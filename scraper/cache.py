"""Persistent cache for analyzed auctions.

Stores the post-audit, post-comps DataFrame for each auction on disk so we
don't have to re-run the expensive steps when revisiting the same auction.
Current bids / time-left / status come fresh from Phase 1 Discovery; only
the immutable analysis columns (verdict, est_resale, ebay_str, etc.) are
read from cache.

Eviction:
- auction closing_date has passed -> purge
- cached_at older than ttl_days   -> purge
- user manually clears            -> purge
"""
import os
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import pandas as pd

# Columns we trust from cache (they don't change within an auction's life).
# Everything else (current_bid, bid_count, est_cost, time_left, status)
# must come from a fresh Phase 1 fetch.
CACHED_ANALYSIS_COLS = [
    'enriched_title', 'enriched_title_pre_image',
    'verdict', 'confidence', 'red_flag',
    'est_resale', 'price_low', 'price_high', 'comp_count',
    'ebay_comps', 'mercari_comps',
    'price_source', 'ebay_str', 'str_source',
    # Image-enrichment columns (from scraper/vision_enrich.py)
    'img_enriched_title', 'img_confidence', 'img_comp_count',
    'img_top_match', 'img_top_price', 'img_error',
]

CACHE_DIR = Path(".cache") / "auctions"


class AuctionCache:
    """Thin wrapper around a filesystem-backed auction-analysis cache."""

    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, auction_id) -> Path:
        return self.cache_dir / f"{auction_id}.pkl"

    def save(self, auction_id, auction_name: str, df: pd.DataFrame,
             closing_date: str = "") -> None:
        """Persist the analyzed DataFrame for this auction."""
        if df is None or df.empty:
            return
        # Keep only what we'll trust on reload + identity columns
        keep_cols = ['lot_id'] + [c for c in CACHED_ANALYSIS_COLS if c in df.columns]
        slim = df[keep_cols].copy() if 'lot_id' in df.columns else df.copy()

        payload = {
            "auction_id": auction_id,
            "auction_name": auction_name,
            "cached_at": datetime.now().isoformat(),
            "closing_date": closing_date,
            "df": slim,
        }
        with open(self._path(auction_id), "wb") as f:
            pickle.dump(payload, f)

    def load(self, auction_id) -> Optional[Dict]:
        """Return the cached payload dict, or None if not cached / corrupt."""
        path = self._path(auction_id)
        if not path.exists():
            return None
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            # Corrupt / incompatible cache — delete and bail
            try:
                path.unlink()
            except Exception:
                pass
            return None

    def is_fresh(self, payload: Dict, ttl_days: int = 14) -> bool:
        """Check whether a payload is still within the TTL and the auction
        hasn't closed."""
        if not payload:
            return False

        # 1. TTL check
        try:
            cached_at = datetime.fromisoformat(payload.get("cached_at", ""))
        except (ValueError, TypeError):
            return False
        if datetime.now() - cached_at > timedelta(days=ttl_days):
            return False

        # 2. Closing-date check — if the auction closed, purge the cache.
        closing = payload.get("closing_date", "")
        if closing:
            try:
                end_dt = datetime.fromisoformat(str(closing).replace("Z", ""))
                if datetime.now() > end_dt:
                    return False
            except (ValueError, TypeError):
                pass  # Can't parse -> trust TTL alone

        return True

    def clear_all(self) -> int:
        """Delete every cache file. Returns count deleted."""
        count = 0
        for p in self.cache_dir.glob("*.pkl"):
            try:
                p.unlink()
                count += 1
            except Exception:
                pass
        return count

    def purge_expired(self, ttl_days: int = 14) -> int:
        """Delete cache entries past their TTL or with a closed auction.
        Returns count deleted."""
        count = 0
        for p in self.cache_dir.glob("*.pkl"):
            try:
                with open(p, "rb") as f:
                    payload = pickle.load(f)
                if not self.is_fresh(payload, ttl_days=ttl_days):
                    p.unlink()
                    count += 1
            except Exception:
                # Corrupt file — nuke it
                try:
                    p.unlink()
                    count += 1
                except Exception:
                    pass
        return count

    def list_all(self, ttl_days: int = 14) -> List[Dict]:
        """Return lightweight metadata for every cached auction.

        Includes fresh/stale status so the UI can warn about stale entries
        without forcing a purge.
        """
        out = []
        for p in self.cache_dir.glob("*.pkl"):
            try:
                with open(p, "rb") as f:
                    payload = pickle.load(f)
                out.append({
                    "auction_id": payload.get("auction_id"),
                    "auction_name": payload.get("auction_name", "(unknown)"),
                    "cached_at": payload.get("cached_at", ""),
                    "closing_date": payload.get("closing_date", ""),
                    "items": len(payload.get("df", pd.DataFrame())),
                    "fresh": self.is_fresh(payload, ttl_days=ttl_days),
                })
            except Exception:
                continue
        # Newest first
        out.sort(key=lambda r: r.get("cached_at", ""), reverse=True)
        return out


def merge_cached_analysis(fresh_df: pd.DataFrame,
                          cached_payload: Dict) -> pd.DataFrame:
    """Overlay the cached analysis columns onto a fresh Phase 1 DataFrame.

    Fresh bids / time-left come from fresh_df. Audit verdicts, price comps,
    STR etc. come from the cache (joined by lot_id). Any lot in fresh_df
    that isn't in the cache gets NaN in the analysis columns — those are
    new lots the user can re-run audit/comps on.

    est_roi is recomputed from the fresh est_cost + cached est_resale, so
    it stays accurate as bids climb.
    """
    if not cached_payload or 'df' not in cached_payload:
        return fresh_df.copy()

    cached_df = cached_payload['df']
    if 'lot_id' not in fresh_df.columns or 'lot_id' not in cached_df.columns:
        return fresh_df.copy()

    # Only bring over the analysis columns
    analysis_cols = [c for c in CACHED_ANALYSIS_COLS if c in cached_df.columns]
    if not analysis_cols:
        return fresh_df.copy()

    cached_slim = cached_df[['lot_id'] + analysis_cols].drop_duplicates(subset=['lot_id'])
    merged = fresh_df.merge(cached_slim, on='lot_id', how='left')

    # Recompute est_roi from fresh est_cost (bids may have climbed).
    # We compute on raw numpy arrays here — earlier attempts using
    # `pd.to_numeric(...).round()` still produced "Expected numeric dtype,
    # got object" failures on certain cached payloads (likely Decimal or
    # nullable-extension internals that survive coercion but break
    # Series.round). Going via np.array(..., dtype=float) bottoms out at
    # plain float64 so np.round always works.
    # cost==0 (no bids yet) gets a 9999% sentinel so the row sorts to
    # the top of the results table rather than vanishing as None.
    if 'est_resale' in merged.columns and 'est_cost' in merged.columns:
        merged['est_roi'] = None
        resale = _to_float_array(merged['est_resale'])
        cost = _to_float_array(merged['est_cost'])
        mask = ~np.isnan(resale) & ~np.isnan(cost) & (cost > 0)
        if mask.any():
            roi = np.round((resale[mask] - cost[mask]) / cost[mask] * 100, 0)
            merged.loc[mask, 'est_roi'] = roi
        no_bid = ~np.isnan(resale) & (resale > 0) & (
            np.isnan(cost) | (cost == 0)
        )
        if no_bid.any():
            merged.loc[no_bid, 'est_roi'] = 9999

    return merged


def _to_float_array(series: pd.Series) -> np.ndarray:
    """Coerce a Series to a numpy float64 array, NaN for unconvertible cells.

    `pd.to_numeric(errors='coerce')` is supposed to do this but doesn't
    always — Decimal values, nullable Float64 extension dtype, and certain
    pickled-cache shapes keep the resulting Series at object dtype, which
    then breaks `.round()`. Doing the coercion element-wise via float()
    is slow on huge frames but bulletproof.
    """
    out = np.empty(len(series), dtype='float64')
    for i, v in enumerate(series):
        try:
            if v is None or (isinstance(v, float) and v != v):
                out[i] = np.nan
            else:
                out[i] = float(v)
        except (TypeError, ValueError):
            out[i] = np.nan
    return out
