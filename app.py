# =============================================================================
# Auction Intelligence Dashboard — Streamlit app entry point.
#
# File layout (search by these section markers to jump around):
#   IMPORTS + CONSTANTS                  — top of file
#   STATE MANAGEMENT                     — `# --- STATE MANAGEMENT ---`
#   PERSISTENT DISCOVERY CACHE           — `# --- Persistent discovery-result cache ---`
#   ASYNC WRAPPER + RUN HELPERS          — `# --- ASYNC WRAPPER ---`
#   PAGE CONFIG + CSS                    — `# --- PAGE CONFIGURATION ---`
#   AUDIT / IMAGE / COMP / FILTER FNS    — `def _run_ai_audit`, `def _run_ebay_comps`
#                                          `def _apply_comps_filters`, etc.
#   RENDER HELPERS                       — `def _render_results_table`,
#                                          `def _render_live_status_panel`,
#                                          `def _render_live_hibid_view`, etc.
#   HEADER + STICKY BANNER               — `# --- MAIN DASHBOARD UI ---`
#   AUTO-DISCOVER TRIGGER                — `# Auto-discover on first page load`
#   PHASE WORK BLOCKS                    — `# WORK BLOCK: Phase 1a discovery`
#                                          (1a / 1b / 2 / 3 / 4 / 5)
#   ANALYSIS VIEW                        — `# ---- ANALYSIS VIEW`
#   DEFAULT VIEW (no auction loaded)     — `# ---- DEFAULT VIEW`
#
# Companion modules in scraper/:
#   bolo.py            — BOLO matcher with hot-reload + chunked match
#   cache.py           — per-auction analyzed-DataFrame pickle cache
#   comped_lots.py     — lot-level comp registry (.cache/comped_lots.json)
#   my_bids.py         — bid tracker (.cache/my_bids.json)
#   pass1.py           — HiBid GraphQL discovery + lot fetch
#   pass2.py           — Anthropic-based audit (verdict / red_flag)
#   ebay_prices.py     — eBay sold-listings comp scraper
#   pricecharting.py   — curated TCG/video-game/comic prices (free)
#   (gocollect.py removed 7/6/26 — API application rejected)
#   vision_enrich.py   — Claude Vision title rewriter for vague lots
#
# Data files in data/*.json:
#   13 BOLO category files (clothing, household_parts, watch_accessories,
#   precious_metals, nostalgia_collectibles, vintage_video_games, etc.)
# =============================================================================
# --- TLS bridge to OS native cert store ---------------------------------
# Some environments (Norton/Kaspersky/ZScaler/corp MITM proxies) re-sign
# HTTPS traffic with a private root CA that is trusted by the OS cert
# store but NOT by certifi (which is what httpx/requests use by default).
# truststore.inject_into_ssl() makes ssl.create_default_context() honor
# the OS store, so every downstream HTTPS call (httpx, requests, urllib)
# automatically picks up the inspector's root and stops throwing
# CERTIFICATE_VERIFY_FAILED. Must run BEFORE any module that opens TLS.
try:
    import truststore  # type: ignore
    truststore.inject_into_ssl()
except ImportError:
    # truststore is optional. On environments without a SSL-inspecting
    # AV / proxy, certifi works fine and this import isn't needed.
    pass

import streamlit as st # type: ignore
import numpy as np
import pandas as pd
import asyncio
import os
import pickle
import re
import sys
import time as _time
from datetime import datetime, timedelta
from pathlib import Path


# ---------------------------------------------------------------------
# Terminal logging helper. Streamlit reruns the script on every user
# interaction, so naive prints are noisy. `tlog()` writes a timestamped
# tag-prefixed line straight to stderr (which Streamlit pipes to the
# terminal where you ran `streamlit run`). Use it at PHASE TRANSITIONS
# (start/end of discovery, fetch, audit, comps) — not inside hot loops.
# ---------------------------------------------------------------------
def tlog(tag: str, *parts) -> None:
    """Print a timestamped log line to the terminal (stderr).

    Examples:
        tlog("DISCOVER", "Phase 1a starting", f"radius={radius}")
        tlog("FETCH", f"got {len(df)} lots in {elapsed:.1f}s")
    """
    msg = " ".join(str(p) for p in parts)
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{tag}] {msg}", file=sys.stderr, flush=True)


# Print one banner per Streamlit script run so the user can see how often
# the rerun loop fires when interacting with widgets. Cheap (1 line, no
# work) and useful for diagnosing "is it stuck" vs "is it working".
tlog("RUN", f"Streamlit script run #{st.session_state.get('_run_count', 0) + 1}")
st.session_state._run_count = st.session_state.get('_run_count', 0) + 1


_LAST_SCAN_VIEW_PATH = Path(".cache/last_scan_view.pkl")


def _save_last_scan_view(label: str, df: pd.DataFrame):
    """Persist a completed synthetic scan (BOLO / keyword / basket).

    A browser disconnect mid-session cancels the Streamlit script and
    wipes session_state — a nearly-finished multi-minute scan used to
    evaporate with it (7/11: user lost an 86k-lot BOLO scan seconds
    before completion). The matched subset is small (dozens-hundreds
    of rows), so persist it the moment it exists; the Discover view
    offers a one-click restore for 24h.
    """
    try:
        _LAST_SCAN_VIEW_PATH.parent.mkdir(exist_ok=True)
        with open(_LAST_SCAN_VIEW_PATH, "wb") as fh:
            pickle.dump({
                "label": label,
                "df": df,
                "saved_at": datetime.now().isoformat(),
            }, fh)
        tlog("SCAN-SAVE", f"persisted '{label}' ({len(df)} rows)")
    except Exception as e:
        tlog("SCAN-SAVE", f"failed (non-fatal): {e}")


def _load_last_scan_view():
    """Return {label, df, saved_at} if a fresh (<24h) scan snapshot
    exists on disk, else None."""
    try:
        if not _LAST_SCAN_VIEW_PATH.exists():
            return None
        with open(_LAST_SCAN_VIEW_PATH, "rb") as fh:
            payload = pickle.load(fh)
        saved_at = datetime.fromisoformat(payload.get("saved_at", ""))
        if (datetime.now() - saved_at).total_seconds() > 24 * 3600:
            return None
        if not isinstance(payload.get("df"), pd.DataFrame):
            return None
        return payload
    except Exception:
        return None


# ---------------------------------------------------------------------
# Interrupted-scan recovery (7/13). The scan-view snapshot only saves
# AFTER the BOLO/keyword match completes — a session that dies during
# the multi-minute FETCH or the match itself saved nothing and lost
# the expensive fetch. This persists the fetched frame + the pending
# mode flags the instant fetch finishes, so a mid-match death can
# resume by re-running the (cache-fast) match on the already-fetched
# lots — no re-fetch.
# ---------------------------------------------------------------------
_FETCHED_FRAME_PATH = Path(".cache/interrupted_scan_fetch.pkl")


def _save_fetched_frame(df: pd.DataFrame, mode: dict) -> None:
    """Persist a just-fetched scan frame + its pending mode flags."""
    try:
        _FETCHED_FRAME_PATH.parent.mkdir(exist_ok=True)
        with open(_FETCHED_FRAME_PATH, "wb") as fh:
            pickle.dump({
                "df": df,
                "mode": mode,
                "saved_at": datetime.now().isoformat(),
            }, fh)
        tlog("SCAN-FETCH-SAVE", f"persisted {len(df)} fetched lots for resume")
    except Exception as e:
        tlog("SCAN-FETCH-SAVE", f"failed (non-fatal): {e}")


def _load_fetched_frame():
    """Return a fresh (<6h) interrupted-fetch payload, or None."""
    try:
        if not _FETCHED_FRAME_PATH.exists():
            return None
        with open(_FETCHED_FRAME_PATH, "rb") as fh:
            payload = pickle.load(fh)
        saved_at = datetime.fromisoformat(payload.get("saved_at", ""))
        if (datetime.now() - saved_at).total_seconds() > 6 * 3600:
            return None
        if not isinstance(payload.get("df"), pd.DataFrame):
            return None
        return payload
    except Exception:
        return None


def _clear_fetched_frame() -> None:
    """Delete the interrupted-fetch file (match completed / new load)."""
    try:
        _FETCHED_FRAME_PATH.unlink(missing_ok=True)
    except Exception:
        pass


def _render_brand_bar_chart(brand_counts: dict, title: str = "BOLO matches by brand"):
    """Render a brand-count bar chart using Streamlit's native st.bar_chart.

    Replaces an earlier plotly.express pie chart — same data, smaller
    dependency footprint, and faster page loads. Top 12 brands plus a
    rolled-up 'Other' bucket so the chart stays readable on long-tail
    scans.

    Returns the brand→count dict that was rendered (caller may want to
    use it for follow-up display); returns empty dict + does nothing if
    `brand_counts` is empty.
    """
    if not brand_counts:
        return {}
    items = sorted(brand_counts.items(), key=lambda kv: kv[1], reverse=True)
    if len(items) > 12:
        head = items[:11]
        tail_total = sum(v for _, v in items[11:])
        head.append((f"Other ({len(items) - 11} brands)", tail_total))
        items = head
    if title:
        st.caption(f"**{title}**")
    chart_df = pd.DataFrame(
        {'Brand': [k for k, _ in items],
         'Matches': [v for _, v in items]}
    ).set_index('Brand')
    st.bar_chart(chart_df, height=300, horizontal=True)
    return dict(items)

# --- IMPORT MODULES ---
from scraper import Phase1Scraper
from scraper.cache import AuctionCache, merge_cached_analysis, CACHED_ANALYSIS_COLS
from scraper.bolo import BoloMatcher
from scraper.auth_check import (
    analyze_description as _auth_analyze_description,
    analyze_photo as _auth_analyze_photo,
    merge_results as _auth_merge_results,
    detect_stylized_replica as _detect_stylized_replica,
    is_dropship_lot as _is_dropship_lot,
)


def _hibid_module():
    """Lazy import of the HiBid live-bids module so cold start doesn't
    pay for it until the user opens the My Bids popover."""
    from scraper import hibid_account as _hibid_account
    return _hibid_account


def _lookup_est_resale_in_session(lot_id):
    """Find est_resale for a lot_id in the current session's audit_results.

    The Live HiBid stop signal can only fire on lots the user has
    analyzed in the current session — anything else shows "❓ NO COMP"
    and the user is prompted to run comps on the source auction.
    """
    if not lot_id:
        return None
    ar = st.session_state.get('audit_results')
    # audit_results is `{}` (empty dict) until an audit runs — guard the
    # DataFrame access so refreshing HiBid bids before any auction is
    # analyzed doesn't blow up ('dict' has no attribute 'columns').
    if (not isinstance(ar, pd.DataFrame) or ar.empty
            or 'lot_id' not in ar.columns or 'est_resale' not in ar.columns):
        return None
    matches = ar[ar['lot_id'].astype(str) == str(lot_id)]
    if matches.empty:
        return None
    val = matches['est_resale'].iloc[0]
    if pd.isna(val):
        return None
    return float(val)

# Single shared cache instance; auto-creates the dir on first touch
_AUCTION_CACHE = AuctionCache()


def _get_cached_auction_list_memo():
    """Return a session-memoized list of every cached auction's metadata.

    Shared by the 💾 Memory popover and the auction-picker — both
    used to call ``_AUCTION_CACHE.list_all()`` separately, doing the
    same disk-glob + pickle-unpack twice per render. This single memo
    is keyed on the cache-dir mtime and always fetches with a 365-day
    TTL (effectively unfiltered) so callers can derive their own
    'fresh' buckets without invalidating each other.
    """
    try:
        mtime = _AUCTION_CACHE.cache_dir.stat().st_mtime
    except (OSError, AttributeError):
        mtime = 0
    memo = st.session_state.get('_auction_list_memo')
    if memo is not None and memo.get('mtime') == mtime:
        return memo['entries']
    try:
        entries = _AUCTION_CACHE.list_all(ttl_days=365)
    except Exception:
        entries = []
    st.session_state._auction_list_memo = {
        'mtime': mtime,
        'entries': entries,
    }
    return entries

# Single shared BOLO matcher. Hot-reloads from data/*.json whenever
# any file's mtime changes — the user can drop in new quarterly
# brand lists without restarting Streamlit.
#
# Wrapped in a lazy-load proxy because eager construction reads 15+
# JSON files + compiles ~750 regex patterns (~1-2s of blocking work)
# on every first Streamlit script run. The header, sticky banner,
# auto-discover trigger, and Phase 1a/1b work blocks DON'T need the
# matcher. Lazy-loading lets the "🔍 Discovering open auctions"
# banner appear ~2 seconds sooner on first open. The first call to
# any matcher attribute (typically the sidebar's BOLO-only filter
# label rendering its brand_count) triggers the actual load.
@st.cache_resource(show_spinner=False)
def _build_bolo_matcher():
    """Construct (or return the cached) BoloMatcher singleton.

    `@st.cache_resource` survives Streamlit reruns — without it, the
    module-level `_BOLO_MATCHER = _LazyBoloMatcher()` line re-runs on
    every page interaction and the underlying BoloMatcher() constructor
    re-reads 17 JSON files + recompiles ~750 regex patterns (~1-2s).
    The matcher's own `_load_if_stale()` mtime check keeps hot-reload
    working — edits to the BOLO JSONs still pick up on the next match()
    call without needing a full restart.
    """
    return BoloMatcher()


class _LazyBoloMatcher:
    __slots__ = ('_instance',)

    def __init__(self):
        self._instance = None

    def _real(self):
        if self._instance is None:
            # cache_resource means this is free on every rerun after
            # the first construction. The lazy wrapper still defers
            # the FIRST call so the discovery banner paints first.
            self._instance = _build_bolo_matcher()
        return self._instance

    @property
    def loaded(self):
        # IMPORTANT: do NOT route through __getattr__ for `loaded` —
        # that would trigger a build just to answer "are you loaded?".
        # `loaded` returning False on first render is exactly what we
        # want: it tells the sidebar render to skip the matcher-heavy
        # paths and render the auction list immediately. The matcher
        # builds the first time someone actually NEEDS it (during the
        # BOLO scan, post-fetch matching, audit scope, etc.).
        return self._instance is not None

    def __getattr__(self, name):
        if name in ('_instance', 'loaded'):
            raise AttributeError(name)
        return getattr(self._real(), name)


_BOLO_MATCHER = _LazyBoloMatcher()


# ⭐ Proven-lane categories — the three lanes the user's actual sales
# history proved are high-margin AND easy-ship. Keyed on bolo_category
# so any brand filed under these tags inherits the flag:
#   Loungefly            → pop_culture_bag
#   Appliance/cookware/  → appliance_parts, cookware_parts, tool_parts,
#     tool parts             kitchen_knives
#   Small accessories    → watch_accessory, designer_eyewear
# (Deliberately NOT sterling/precious_metal or diecast — those LOST
# money in the 12-month readout.)
_PROVEN_LANE_CATEGORIES = frozenset({
    'pop_culture_bag',
    'appliance_parts', 'cookware_parts', 'tool_parts', 'kitchen_knives',
    'watch_accessory', 'designer_eyewear',
})


def _compute_bolo_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add bolo_brand / bolo_tier / bolo_target_buy_low/high columns.

    Vectorized over duplicate (title, description) pairs: the BOLO
    matcher is called ONCE per unique (title, desc) tuple within a
    given dataframe, then the result is broadcast back to every row
    sharing that pair. In dedup-rich datasets (estate-sale jewelry,
    repeated commodity Pyrex titles) this often cuts the work 50%+.

    Done lazily (during render) rather than baked into the audit
    pipeline so the user can swap the JSON file at any time and have
    results reflect the new list on the next rerun. No API cost,
    just regex.
    """
    if df is None or len(df) == 0 or 'title' not in df.columns:
        return df
    titles = df['title'].fillna('').astype(str)
    descs = (
        df['description'].fillna('').astype(str)
        if 'description' in df.columns
        else pd.Series([''] * len(df), index=df.index)
    )

    # Per-call cache: BOLO match + stylized check are identical for
    # identical haystacks, so memoize keyed on (title, description).
    _match_cache: dict = {}
    _stylized_cache: dict = {}
    _auth_cache: dict = {}

    def _cached_match(t: str, d: str):
        key = (t, d)
        cached = _match_cache.get(key, _SENTINEL)
        if cached is _SENTINEL:
            cached = _BOLO_MATCHER.match(t, d)
            _match_cache[key] = cached
        return cached

    def _cached_stylized(t: str, d: str):
        key = (t, d)
        cached = _stylized_cache.get(key, _SENTINEL)
        if cached is _SENTINEL:
            cached = _detect_stylized_replica(t, d)
            _stylized_cache[key] = cached
        return cached

    def _cached_auth(t: str, d: str, match):
        # Auth analysis only runs for auth_required brands; key on
        # (t, d, brand) so a different brand on the same haystack
        # gets its own score.
        key = (t, d, match.get('brand') if match else None)
        cached = _auth_cache.get(key, _SENTINEL)
        if cached is _SENTINEL:
            cached = _auth_analyze_description(t, d, match)
            _auth_cache[key] = cached
        return cached

    brands = []
    categories = []
    tiers = []
    models = []
    buy_lows = []
    buy_highs = []
    confs = []
    auth_required = []
    auth_scores = []
    auth_red_summaries = []
    auth_green_summaries = []
    stylized_flags = []
    stylized_phrases = []
    for t, d in zip(titles, descs):
        match = _cached_match(t, d)
        # Run the stylized/replica detector on EVERY row, not just
        # BOLO matches — a "Rolex style watch" lot might not match our
        # BOLO for Rolex but still has the contamination problem in
        # comps. The flag downstream gates whether comps actually run.
        stylized = _cached_stylized(t, d)
        stylized_flags.append(stylized is not None)
        stylized_phrases.append(stylized or "")
        if match:
            brands.append(match.get('brand'))
            categories.append(match.get('category'))
            tiers.append(match.get('tier'))
            models.append(match.get('matched_model'))
            buy_lows.append(match.get('target_buy_low'))
            buy_highs.append(match.get('target_buy_high'))
            confs.append(match.get('confidence'))
            auth_req = bool(match.get('auth_required'))
            auth_required.append(auth_req)
            # Description-based authenticity scoring runs only on
            # auth-required brands (tier 3 luxury + Polo sublines).
            if auth_req:
                auth = _cached_auth(t, d, match)
                auth_scores.append(auth['auth_score'] if auth else None)
                auth_red_summaries.append(
                    ", ".join(auth['red_flags']) if auth and auth.get('red_flags') else ""
                )
                auth_green_summaries.append(
                    ", ".join(auth['green_flags']) if auth and auth.get('green_flags') else ""
                )
            else:
                auth_scores.append(None)
                auth_red_summaries.append("")
                auth_green_summaries.append("")
        else:
            brands.append(None)
            categories.append(None)
            tiers.append(None)
            models.append(None)
            buy_lows.append(None)
            buy_highs.append(None)
            confs.append(None)
            auth_required.append(False)
            auth_scores.append(None)
            auth_red_summaries.append("")
            auth_green_summaries.append("")
    df = df.copy()
    df['bolo_brand'] = brands
    df['bolo_category'] = categories
    df['bolo_tier'] = tiers
    # ⭐ Proven-lane flag (7/13). The user's 12-month Nifty sales data
    # showed three lanes that are BOTH high-margin AND mailbox-
    # shippable — the money lanes worth prioritizing over the 300+
    # other BOLO brands (many of which, like weighed sterling and
    # mainline diecast, LOST money last year). Flagging by
    # bolo_category so the results table can surface / filter them.
    df['bolo_proven_lane'] = (
        pd.Series(categories, index=df.index)
        .isin(_PROVEN_LANE_CATEGORIES)
    )
    df['bolo_model'] = models
    df['bolo_target_buy_low'] = buy_lows
    df['bolo_target_buy_high'] = buy_highs
    df['bolo_confidence'] = confs
    df['bolo_auth_required'] = auth_required
    df['bolo_auth_score'] = auth_scores
    df['bolo_auth_red'] = auth_red_summaries
    df['bolo_auth_green'] = auth_green_summaries
    df['is_stylized_replica'] = stylized_flags
    df['stylized_phrase'] = stylized_phrases
    # Tag CPU-category lots as bare_cpu / system / unclear so the
    # results table can show whether a lot is a chip or a whole
    # machine — those comp very differently.
    df = _compute_cpu_form_factor_columns(df)
    return df


# Sentinel for the per-call memoization in _compute_bolo_columns —
# need a distinct value because `None` is a valid match() result
# (means "no BOLO match for this lot").
_SENTINEL = object()


def _compute_bolo_columns_chunked(df: pd.DataFrame, chunk_size: int = 2000,
                                  progress_callback=None) -> pd.DataFrame:
    """Same output as _compute_bolo_columns, but processes in chunks of
    `chunk_size` rows so the UI can show running progress.

    progress_callback signature: (current, total, hits_so_far, top_brands_dict) -> None
    Called after every chunk so the caller can refresh a progress bar
    plus a live "X matches so far, top brands: Trifari, Pyrex, …" line.

    Falls back to the unchunked path when df is small (≤ chunk_size).
    """
    if df is None or len(df) == 0 or 'title' not in df.columns:
        return df
    n = len(df)
    if n <= chunk_size:
        out = _compute_bolo_columns(df)
        if progress_callback is not None:
            hits = (
                int(out['bolo_brand'].notna().sum())
                if 'bolo_brand' in out.columns else 0
            )
            top = (
                out.loc[out['bolo_brand'].notna(), 'bolo_brand']
                .value_counts().head(8).to_dict()
                if 'bolo_brand' in out.columns else {}
            )
            progress_callback(n, n, hits, top)
        return out

    parts: list = []
    hits_so_far = 0
    from collections import Counter
    brand_counter: Counter = Counter()
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = _compute_bolo_columns(df.iloc[start:end])
        # Skip empty chunks — pandas 2.x emits a FutureWarning when
        # concat sees empty/all-NA frames because it'll change dtype
        # behavior in a future release. Filtering here is a no-op for
        # results but silences the warning.
        if not chunk.empty:
            parts.append(chunk)
        if 'bolo_brand' in chunk.columns:
            chunk_brands = chunk['bolo_brand'].dropna()
            hits_so_far += len(chunk_brands)
            brand_counter.update(chunk_brands.tolist())
        if progress_callback is not None:
            top = dict(brand_counter.most_common(8))
            progress_callback(end, n, hits_so_far, top)
    if not parts:
        return df.iloc[0:0]
    return pd.concat(parts, ignore_index=False)

# --- Persistent discovery-result cache ---
# Streamlit session state is wiped on browser refresh / app restart, which
# means a successful "Discover Auctions" run only sticks around for the
# current session. Persist the candidate list + the sourcing config that
# produced it to disk so we can rehydrate on the next app load without
# making the user click the button again. TTL: 24 hours (bids/closing
# times move fast enough that day-stale data is the outer edge of useful).
_DISCOVERY_CACHE_PATH = Path(".cache") / "last_discovery.pkl"
_DISCOVERY_CACHE_TTL = timedelta(hours=24)
# Bump this when the cached payload shape changes — old caches are silently
# discarded so users don't have to delete the file by hand. v4 drops the
# thumbnail_urls field (rendering 4 images per card across 200+ auctions
# was too slow to load); images are now reserved for the lot-results table.
_DISCOVERY_CACHE_VERSION = 4


def _save_cached_discovery(candidates, sourcing_cfg, category_samples=None):
    """Persist a successful discovery result for the next session."""
    try:
        _DISCOVERY_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": _DISCOVERY_CACHE_VERSION,
            "saved_at": datetime.now().isoformat(),
            "sourcing_cfg": dict(sourcing_cfg or {}),
            "candidates": list(candidates or []),
            # Sampled lot previews (categories + titles + thumbnail_url) so
            # the picker rehydrates on reload.
            "category_samples": dict(category_samples or {}),
        }
        with open(_DISCOVERY_CACHE_PATH, "wb") as fh:
            pickle.dump(payload, fh)
    except Exception:
        # Cache persistence is best-effort; never let it break the app.
        pass


def _load_cached_discovery():
    """Return a dict with candidates/sourcing_cfg/age if fresh, else None.

    Caches written by an older schema (no schema_version, or a smaller
    number than the current one) are silently discarded — re-running
    Discover repopulates the file with the new shape.
    """
    try:
        if not _DISCOVERY_CACHE_PATH.exists():
            return None
        with open(_DISCOVERY_CACHE_PATH, "rb") as fh:
            payload = pickle.load(fh)
        if payload.get("schema_version", 1) < _DISCOVERY_CACHE_VERSION:
            return None
        saved_at = datetime.fromisoformat(payload.get("saved_at", ""))
        age = datetime.now() - saved_at
        if age > _DISCOVERY_CACHE_TTL:
            return None
        return {
            "candidates": payload.get("candidates") or [],
            "sourcing_cfg": payload.get("sourcing_cfg") or {},
            "category_samples": payload.get("category_samples") or {},
            "saved_at": saved_at,
            "age": age,
        }
    except Exception:
        return None





# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="H-Town TX Finds: ROI Engine",
    page_icon="🛰️",
    layout="wide",
    # auto = expanded on wide screens, collapsed-overlay on narrow ones.
    # Hard-coding "expanded" forced the sidebar to occupy a fixed chunk
    # of the screen at every width, which squeezed the analysis split-
    # screen at sizes between desktop and mobile.
    initial_sidebar_state="auto",
)

# Sidebar sizing — only force the wider 320–360 px footprint when the
# viewport actually has room (≥1024 px). Below that we let Streamlit's
# default responsive behavior take over: the sidebar auto-collapses to
# a hamburger and overlays content only when the user opens it.
st.markdown(
    """
    <style>
    @media (min-width: 1024px) {
        [data-testid="stSidebar"] {
            min-width: 320px !important;
            max-width: 360px !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- MOBILE-RESPONSIVE CSS ---
st.markdown("""
<style>
/* ---- Touch-friendly tabs ---- */
button[data-baseweb="tab"] {
    padding: 12px 8px !important;
    font-size: 14px !important;
}

/* ---- Larger tap targets for buttons ---- */
.stButton > button {
    min-height: 48px !important;
    font-size: 15px !important;
}

/* ---- Compact title on small screens ---- */
@media (max-width: 640px) {
    /* Stack metric columns vertically */
    [data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important;
        gap: 4px !important;
    }
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        flex: 1 1 45% !important;
        min-width: 45% !important;
    }

    /* Shrink header */
    h1 { font-size: 1.5rem !important; }
    h2 { font-size: 1.2rem !important; }
    h3 { font-size: 1.1rem !important; }

    /* Metric cards: tighter padding */
    [data-testid="stMetric"] {
        padding: 8px 4px !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 11px !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 18px !important;
    }

    /* Dataframe horizontal scroll hint */
    [data-testid="stDataFrame"] {
        overflow-x: auto !important;
        -webkit-overflow-scrolling: touch !important;
    }

    /* Tab text: shorter labels */
    button[data-baseweb="tab"] {
        padding: 10px 4px !important;
        font-size: 12px !important;
    }
}

/* Removed: a global @media (max-width: 960px) block here used to wrap
   every stHorizontalBlock to 48% min-width, which broke the picker
   table at any narrow-but-not-mobile width. The <640px rule above
   still handles real phones. */

/* Removed sticky-results CSS — the analysis view now uses a single
   full-width results panel with manual controls in a collapsed
   expander below, so there's no second column to make sticky. */
</style>
""", unsafe_allow_html=True)

# --- STATE MANAGEMENT ---
# All session-state defaults consolidated into a single dict. setdefault()
# is idempotent: if the key exists in session state (e.g. user already
# tweaked the slider), the default is ignored. Comments organize the keys
# by purpose so it's easy to add/remove without scrolling 200 lines.
_STATE_DEFAULTS = {
    # --- Pipeline result frames ---
    'phase1_leads':              pd.DataFrame(),
    'selected_leads':            pd.DataFrame(),
    'current_auction':           None,
    'audit_results':             {},

    # --- Pipeline running flags ---
    'discover_running':          False,
    'fetch_lots_running':        False,
    '_sampling_pending':         False,   # Phase 1b background sampling
    'audit_running':             False,
    'comps_running':             False,

    # --- Audit knobs ---
    # Phase 4 parallelism. 12 = Haiku-tier sweet spot (~33% faster than
    # the old 8-default; 16+ starts hitting rate limits).
    'audit_workers':             12,
    # Which engine runs the audit AI (text + photo tiers). 'gemini' routes
    # to Google's gemini-flash-lite (cheaper than Claude Haiku, matched its
    # quality on real auction photos); 'claude' keeps the original path.
    # Per-run selector in ⚙️ Audit settings. Falls back to config
    # gemini.provider on first load if the user hasn't picked one.
    'vision_provider':           'gemini',

    # --- Pre-comps filter knobs ---
    # Defaults match the 🌐 Aggressive preset — no top-N cap, no
    # min-bid floor, no bid-ceiling. Per user preference: maximum
    # coverage out of the box. Tighter modes (Standard / Conservative)
    # are still one click away in the spend-confirm gate.
    'comps_max_lots':            0,       # 0 = no cap
    'comps_skip_above_bid':      0.0,     # 0 = no skip-above-bid cap
    'comps_skip_below_str':      0.0,     # skip categories below X% STR
    'comps_dedup_titles':        True,    # fingerprint dedup
    'comps_use_melt_floor':      False,   # melt-value for precious metals
    'comps_melt_premium_factor': 1.4,
    'comps_exclude_hard':        True,    # drop HARD logistics
    'comps_easy_ship_only':      True,    # drop anything not mailbox-easy
    'comps_pc_check_stylized':   True,
    'comps_skip_unknown_verdict': True,   # don't price condition-unknown lots
    'comps_use_mercari':         False,   # 0% success rate; off by default
    'comps_chunk_size':          5000,    # process N at a time
    'comps_use_auction_str':     True,    # auction-level STR sampling
    'comps_workers':             8,       # comp thread-pool size

    # --- Multi-auction BOLO scan flow ---
    '_comps_free_only_mode':     False,
    '_bolo_scan_all_pending':    False,
    '_bolo_scan_summary':        None,

    # --- 🕵️ Sleeper hunt (automatic stage of the BOLO scan-all) ---
    # Container-titled lots ("Jewelry Box", "Lot of Toys") whose value
    # is only visible in photos. Candidates are scored on free signals,
    # the top N get vision enrichment, and enriched titles re-run
    # through the BOLO matcher + melt detector.
    'sleeper_hunt_enabled':      True,
    'sleeper_max_lots':          300,   # vision-enrichment budget cap

    # --- Multi-auction KEYWORD scan flow ---
    # When set to a non-empty string, the post-fetch handler filters
    # every fetched lot by case-insensitive substring match against
    # title + description (instead of running the BOLO matcher) and
    # loads the matched subset as a synthetic "🔍 Keyword: <term>"
    # auction. Cleared back to '' on auction load / scan completion.
    '_keyword_scan_pending':     '',
    '_keyword_scan_summary':     None,

    # --- Retail-anchored pricing (Amazon-return auctions) ---
    'use_retail_anchor':         True,
    'use_amazon_live':           True,   # live buy-box fetch for big lots
    'amazon_live_min_retail':    100.0,  # only lots worth the credits
    'amazon_live_max_lookups':   20,     # per comps batch
    'retail_anchor_factor':      0.5,   # est_resale = retail x this

    # --- Pre-audit preview filters ---
    'preaudit_exclude_categories': [],
    'preaudit_exclude_metals':   False,

    # --- ScrapingBee spend ledger ---
    # Cross-session lot-level record of every paid lookup. Lots priced
    # within the TTL restore for free; known-empty attempts are
    # blocked from re-spending.
    'use_spend_ledger':          True,
    'spend_ledger_ttl_days':     7.0,

    # --- Multi-select combined analysis ---
    # True while a multi-auction grid selection is being fetched; the
    # post-fetch handler combines the lots into one synthetic
    # "🧺 basket" analysis view.
    '_multi_select_pending':     False,

    # --- Cache TTLs ---
    'cache_ttl_days':            1,
}

for _k, _v in _STATE_DEFAULTS.items():
    st.session_state.setdefault(_k, _v)

# --- Special-case initializations ---
# `cache_purged_this_session` triggers a side effect (purge_expired) the
# first time per session, so it can't go through setdefault.
if 'cache_purged_this_session' not in st.session_state:
    _AUCTION_CACHE.purge_expired(ttl_days=st.session_state.cache_ttl_days)
    st.session_state.cache_purged_this_session = True

# `auction_candidates` rehydrates from disk on first load if there's a
# fresh cached discovery — saves the user a Discover click on tab refresh.
if 'auction_candidates' not in st.session_state:
    _cached_disc = _load_cached_discovery()
    if _cached_disc and _cached_disc["candidates"]:
        st.session_state.auction_candidates = _cached_disc["candidates"]
        st.session_state._discovery_restored_from = _cached_disc["saved_at"]
        st.session_state._sourcing_cfg = _cached_disc["sourcing_cfg"]
        st.session_state.category_samples = _cached_disc.get(
            "category_samples", {}
        )
        tlog("CACHE",
             f"restored {len(_cached_disc['candidates'])} auctions from disk",
             f"(saved {_cached_disc['saved_at']})")
    else:
        st.session_state.auction_candidates = []
        st.session_state.category_samples = {}
        tlog("CACHE", "no cached discovery on disk — auto-discover will fire")

# `category_samples` may not have been set by the rehydration branch
# (when there's no cached discovery), so guard separately.
st.session_state.setdefault('category_samples', {})

# `known_categories` — starter list of HiBid categories the picker uses.
# Grows over time from scrape results, so it's stored in session state
# rather than being a constant.
st.session_state.setdefault('known_categories', [
    "Antiques", "Art", "Automotive", "Books & Media",
    "Clothing & Accessories", "Coins & Currency", "Collectibles",
    "Electronics", "Firearms", "Fishing", "Furniture",
    "Glassware", "Home & Garden", "Hunting", "Jewelry",
    "Kitchen", "Music & Instruments", "Outdoors", "Pottery",
    "Sporting Goods", "Sports Memorabilia", "Tools",
    "Toys & Games", "Vintage",
])

# Stub left in place so existing call sites don't error. The mobile
# wake-lock JS injection was removed in the simplification pass —
# this is a desktop-first app and the script was firing the Streamlit
# components.html deprecation warning on every long op.
def _keep_screen_awake():
    """No-op (was a mobile keep-awake script; removed for desktop)."""
    pass


# --- ASYNC WRAPPER ---
def run_async(coro):
    """Run an arbitrary coroutine from Streamlit's sync thread."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


def _extract_auction_id(auction_df: pd.DataFrame):
    """Pull the HiBid auction_id from a DataFrame.

    Discovery rows store it as a URL like https://hibid.com/auction/12345
    in the 'auction_link' column. We parse the trailing int.

    Defined this early in the file because the header refresh button
    (rendered above the rest of the analysis-view code) needs it to
    route a single-auction refresh.
    """
    if 'auction_link' not in auction_df.columns or auction_df.empty:
        return None
    link = str(auction_df['auction_link'].iloc[0])
    if '/auction/' not in link:
        return None
    try:
        return int(link.rsplit('/auction/', 1)[1].split('/')[0].split('?')[0])
    except (ValueError, IndexError):
        return None


def _aid_from_link(link) -> int | None:
    """Parse a HiBid auction_link → integer auction_id (None if unparseable).

    Row-level helper used by the multi-auction cache merge to identify
    which auction each lot belongs to without an extra join.
    """
    s = str(link or '')
    if '/auction/' not in s:
        return None
    try:
        return int(s.rsplit('/auction/', 1)[1].split('/')[0].split('?')[0])
    except (ValueError, IndexError):
        return None


def _merge_cached_analysis_multi(df: pd.DataFrame):
    """Overlay cached analysis onto a multi-auction DataFrame.

    Walks every unique auction_id in ``df``, loads the per-auction
    cache pickle (skipping stale ones), and merges the cached analysis
    columns (verdict, est_resale, etc.) onto matching rows by ``lot_id``.

    The downstream BOLO scan + audit + comps see lots with
    pre-populated ``verdict`` / ``est_resale`` / ``red_flag`` and skip
    them automatically (audit fast-path on lots with verdicts; comps
    skip lots with non-NaN ``est_resale``). On a re-scan of mostly the
    same auctions, this turns a 30-min comp run into a sub-minute
    pass over just the new auctions.

    Returns ``(merged_df, stats)`` where stats is a dict with:
        cached_auctions, new_auctions, cached_lots, total_auctions
    Stats are ``{}`` when the input is empty or unmergeable.
    """
    if (
        df is None or df.empty
        or 'auction_link' not in df.columns
        or 'lot_id' not in df.columns
    ):
        return df, {}

    ttl_days = int(st.session_state.cache_ttl_days)

    # Per-row auction_id, then unique list of auctions in this batch.
    aids = df['auction_link'].apply(_aid_from_link)
    unique_aids = sorted({a for a in aids.tolist() if a is not None})
    if not unique_aids:
        return df, {}

    # Walk each auction's cache, collect just the analysis columns.
    cached_slim_frames: list = []
    cached_aids: set = set()
    for aid in unique_aids:
        try:
            payload = _AUCTION_CACHE.load(aid)
        except Exception:
            continue
        if not payload or not _AUCTION_CACHE.is_fresh(
            payload, ttl_days=ttl_days
        ):
            continue
        cached_df = payload.get('df')
        if (
            not isinstance(cached_df, pd.DataFrame)
            or cached_df.empty
            or 'lot_id' not in cached_df.columns
        ):
            continue
        keep_cols = ['lot_id'] + [
            c for c in CACHED_ANALYSIS_COLS if c in cached_df.columns
        ]
        if len(keep_cols) <= 1:
            continue
        cached_slim_frames.append(
            cached_df[keep_cols].drop_duplicates(subset=['lot_id'])
        )
        cached_aids.add(aid)

    stats = {
        'total_auctions': len(unique_aids),
        'cached_auctions': len(cached_aids),
        'new_auctions': len(unique_aids) - len(cached_aids),
        'cached_lots': 0,
    }

    if not cached_slim_frames:
        return df, stats

    # Build one combined cache lookup keyed by lot_id (HiBid lot_ids
    # are globally unique so we can dedup across auctions safely).
    combined = pd.concat(cached_slim_frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=['lot_id'])

    # Drop columns from fresh df that the cache will provide so the
    # merge doesn't create _x/_y suffix collisions. Phase 1 doesn't
    # populate any of CACHED_ANALYSIS_COLS, but be defensive.
    fresh = df.copy()
    overlap = [c for c in combined.columns if c != 'lot_id' and c in fresh.columns]
    if overlap:
        fresh = fresh.drop(columns=overlap)

    merged = fresh.merge(combined, on='lot_id', how='left')

    # Recompute est_roi from FRESH est_cost + CACHED est_resale so it
    # reflects whatever the bid has climbed to since the cache landed.
    if 'est_resale' in merged.columns and 'est_cost' in merged.columns:
        resale = pd.to_numeric(merged['est_resale'], errors='coerce')
        cost = pd.to_numeric(merged['est_cost'], errors='coerce')
        mask = resale.notna() & cost.notna() & (cost > 0)
        if mask.any():
            roi = ((resale - cost) / cost * 100).round(0)
            if 'est_roi' not in merged.columns:
                merged['est_roi'] = pd.NA
            merged.loc[mask, 'est_roi'] = roi[mask]

    if 'est_resale' in merged.columns:
        stats['cached_lots'] = int(merged['est_resale'].notna().sum())

    return merged, stats


def _load_auction_for_analysis(auction_name, auction_df):
    """Replace the current analysis target with the given auction's items.

    If a fresh cached analysis exists for this auction, overlay its audit
    verdicts and price comps onto the fresh Phase 1 data so the user sees
    results immediately (with current bids, recomputed ROI).

    Hoisted up here so the Memory popover (rendered near the top of the
    script) can call it when the user clicks a cached-auction button —
    Streamlit executes top-to-bottom, so the function has to exist before
    the click handler runs.
    """
    st.session_state.selected_leads = auction_df.copy()
    st.session_state.current_auction = auction_name
    st.session_state.audit_results = {}
    # Fresh auction → reset chunked-comps state so the first comps click
    # starts a new run instead of trying to continue from the previous one.
    st.session_state.pop('_comps_has_more', None)
    st.session_state.pop('_comps_auction_str_map', None)
    st.session_state.pop('_comps_stats', None)
    # Force the credit-spend confirmation to fire again for this auction.
    # Each auction gets its own explicit confirmation before the comp
    # run burns ScrapingBee credits.
    st.session_state.pop('_comps_credit_confirmed', None)
    # Reset the audit-scope chooser. Big auctions (>500 lots) with at
    # least one BOLO match get a "full vs BOLO-only" pre-audit gate so
    # the user doesn't accidentally burn Claude API + ScrapingBee
    # credits on 11k irrelevant lots when 30 are real.
    st.session_state.pop('_audit_scope', None)
    st.session_state.pop('_audit_scope_total_lots', None)
    # Reset the free-only-mode flag — each auction's credit gate
    # makes its own free-vs-paid decision.
    st.session_state._comps_free_only_mode = False
    # Clear any pending multi-auction BOLO-scan state so a
    # subsequent single-auction click doesn't accidentally re-enter
    # scan mode on the next fetch.
    st.session_state._bolo_scan_all_pending = False
    st.session_state._bolo_scan_summary = None
    # Clear pending keyword-scan state too (parallel to BOLO scan)
    st.session_state._keyword_scan_pending = ''
    st.session_state._keyword_scan_summary = None
    st.session_state._multi_select_pending = False
    # Reset all "in-progress" flags. These get set by their respective
    # buttons and cleared in `finally` blocks inside the analysis view —
    # but if the user refreshed mid-run, the analysis branch never
    # executed the finally, so the flags stay True and disable the
    # buttons on the new auction. Belt-and-suspenders reset here.
    st.session_state.audit_running = False
    st.session_state.comps_running = False
    # Reset the auto-pipeline tracker so a new auction starts fresh
    # (audit auto-fires, then first comp chunk auto-fires). Failed
    # attempts on the previous auction don't poison this one.
    st.session_state.pop('_auto_pipeline_attempts', None)
    # Consecutive comps-failure counter is per-auction too — a stall
    # on the previous auction shouldn't suppress auto-resume here.
    st.session_state.pop('_comps_error_count', None)
    # Pre-audit preview gate re-arms on every new auction load.
    st.session_state.pop('_audit_confirmed', None)

    # Consult disk cache, keyed by auction_id (pulled from auction_link or a
    # dedicated column if present)
    auction_id = _extract_auction_id(auction_df)
    if auction_id is None:
        return

    payload = _AUCTION_CACHE.load(auction_id)
    if not payload:
        return
    if not _AUCTION_CACHE.is_fresh(payload, ttl_days=st.session_state.cache_ttl_days):
        return

    merged = merge_cached_analysis(auction_df, payload)
    # Only treat as full audit_results if it actually has verdicts
    if 'verdict' in merged.columns and merged['verdict'].notna().any():
        st.session_state.selected_leads = merged
        st.session_state.audit_results = merged


# ================================================================
# PIPELINE PREFLIGHT — runs ONCE per session, before any UI.
# Two consecutive production auctions (7/6) silently produced garbage
# for two different invisible reasons: (1) the `anthropic` package
# was missing from the venv (audit degraded to no_api_key for every
# lot), then (2) Norton's TLS inspection killed every SDK call with
# APIConnectionError (audit degraded to image_api_failed). Neither
# had a visible symptom until a CSV export was audited by hand.
# The preflight makes both failure classes — and their cousins —
# loud at startup:
#   1. Critical imports (anthropic, truststore, httpx)
#   2. Config keys present (anthropic / scrapingbee / ebay)
#   3. ONE live 1-token Anthropic API ping — the only check that
#      catches SSL interception, dead keys, and network problems.
# Results are cached in session_state so reruns don't re-ping.
# ================================================================
def _gemini_ping(gemini_api_key: str, model: str = "gemini-flash-lite-latest"):
    """1-token Gemini liveness check. Returns None if healthy, else a short
    human-readable reason (credits depleted, key rejected, model retired,
    TLS/network).

    Cheap insurance against the failures we hit in testing: a valid key on
    a $0-balance project returns 429 RESOURCE_EXHAUSTED, and a pinned-but-
    retired model returns 404 NOT_FOUND — both of which the vision wrappers
    swallow to None, so without this every lot would silently fall back to
    keyword-only.
    """
    if not gemini_api_key:
        return "no Gemini key in config.json under gemini.api_key"
    try:
        from scraper.vision_provider import _client, _GEMINI_ENDPOINT
        r = _client().post(
            _GEMINI_ENDPOINT.format(model=model),
            headers={"x-goog-api-key": gemini_api_key,
                     "Content-Type": "application/json"},
            json={"contents": [{"role": "user", "parts": [{"text": "ping"}]}],
                  "generationConfig": {"maxOutputTokens": 1}},
        )
        if r.status_code == 200:
            return None
        body = r.json() if r.headers.get("content-type", "").startswith(
            "application/json") else {}
        err = body.get("error", {}) if isinstance(body, dict) else {}
        status = err.get("status") or f"HTTP {r.status_code}"
        msg = str(err.get("message", ""))[:110]
        if status == "RESOURCE_EXHAUSTED":
            return (f"credits/quota exhausted (RESOURCE_EXHAUSTED) — {msg} "
                    "Top up at ai.studio/projects or switch provider to Claude.")
        if r.status_code in (401, 403) or status in (
                "UNAUTHENTICATED", "PERMISSION_DENIED"):
            return f"key rejected ({status}) — {msg}"
        if r.status_code == 404 or status == "NOT_FOUND":
            return (f"model '{model}' unavailable ({status}) — {msg} "
                    "Set a current model in config.json under gemini.model "
                    "(e.g. gemini-flash-lite-latest).")
        return f"ping failed ({status}) — {msg}"
    except Exception as e:
        return (f"ping error ({type(e).__name__}: {str(e)[:80]}) — "
                "possible TLS interception or network problem")


def _run_preflight() -> list:
    """Return a list of human-readable problem strings (empty = healthy)."""
    issues = []

    # 1. Critical imports
    for pkg, why in (
        ('anthropic', 'AI audit + vision enrichment will silently mark '
                      'every lot no_api_key'),
        ('truststore', 'HTTPS breaks under Norton/corp TLS inspection '
                       '(HiBid + Anthropic calls fail)'),
        ('httpx', 'all HTTP calls depend on it'),
    ):
        try:
            __import__(pkg)
        except ImportError:
            issues.append(
                f"❌ Python package `{pkg}` is not installed — {why}. "
                f"Fix: `pip install {pkg}`"
            )

    # 2. Config keys
    try:
        from scraper.config_loader import load_config
        _cfg = load_config()
        if not (_cfg.get('anthropic') or {}).get('api_key'):
            issues.append(
                "❌ No Anthropic API key in config.json — audit + vision "
                "tiers are disabled."
            )
        if not (_cfg.get('scrapingbee') or {}).get('api_key'):
            issues.append(
                "⚠️ No ScrapingBee key in config.json — eBay sold-comps "
                "will fall back to unreliable active listings."
            )
    except Exception as e:
        issues.append(f"❌ config.json failed to load: {e}")

    # 3. Live API ping — catches what static checks can't (SSL
    # interception, revoked key, network). ~1 token, fractions of a
    # cent, once per session.
    if not any('anthropic' in s or 'Anthropic' in s for s in issues):
        try:
            from scraper.pass2 import Phase2Scraper
            _p2 = Phase2Scraper()
            if _p2.client is None:
                issues.append(
                    "❌ Anthropic client failed to initialize (key set "
                    "but SDK unavailable)."
                )
            else:
                _p2.client.messages.create(
                    model=_p2.model_name, max_tokens=1,
                    messages=[{"role": "user", "content": "ping"}],
                )
        except Exception as e:
            issues.append(
                f"❌ Live Anthropic API ping FAILED "
                f"({type(e).__name__}: {str(e)[:120]}) — the audit will "
                f"mark every lot api_failed. Common causes: TLS "
                f"inspection (Norton/corp proxy), revoked key, network."
            )

    # 4. Live Gemini ping — ONLY when Gemini is the chosen audit provider
    # (session pick, else config default). Skipped entirely on Claude so
    # we never ping a provider the user isn't using.
    try:
        from scraper.config_loader import load_config as _lc
        _gcfg = (_lc().get("gemini") or {})
        _prov = str(
            st.session_state.get("vision_provider")
            or _gcfg.get("provider") or "claude"
        ).lower()
        if _prov == "gemini":
            _err = _gemini_ping(
                _gcfg.get("api_key"),
                _gcfg.get("model") or "gemini-flash-lite-latest",
            )
            if _err:
                issues.append(
                    f"❌ Gemini audit provider is selected but its live ping "
                    f"FAILED — {_err} The audit would silently drop to "
                    f"keyword-only. Fix the key/billing or switch **Audit AI "
                    f"provider** to Claude in ⚙️ Audit settings."
                )
    except Exception:
        pass
    return issues


if '_preflight_issues' not in st.session_state:
    with st.spinner("🩺 Preflight: checking pipeline health…"):
        st.session_state._preflight_issues = _run_preflight()
    _pf = st.session_state._preflight_issues
    tlog("PREFLIGHT",
         f"{'healthy' if not _pf else str(len(_pf)) + ' issue(s)'}")
    for _issue in _pf:
        tlog("PREFLIGHT", _issue)

if st.session_state.get('_preflight_issues'):
    st.error(
        "🩺 **Pipeline preflight found problems** — analysis quality "
        "will degrade silently until these are fixed:\n\n"
        + "\n".join(f"- {s}" for s in st.session_state._preflight_issues)
    )
    if st.button("🔁 Re-run preflight", key="rerun_preflight"):
        st.session_state.pop('_preflight_issues', None)
        st.rerun()


# --- MAIN DASHBOARD UI ---
# Compact top bar: title on the left, settings popovers + Refresh button
# on the right. Replaces the old left sidebar — settings are tucked behind
# a popover instead of taking permanent screen space, since the user
# rarely changes them.
discover_running = st.session_state.get('discover_running', False)
fetch_lots_running = st.session_state.get('fetch_lots_running', False)
_audit_running = st.session_state.get('audit_running', False)
_comps_running = st.session_state.get('comps_running', False)
_sampling_pending = st.session_state.get('_sampling_pending', False)

# --- WATCHDOG: detect wedged "running" flags ---
# A network hang or unhandled exception can leave a *_running flag
# set forever — the user sees a sticky "loading" banner with no way
# out. Stamp the start time when each flag flips True; render a
# manual reset button if too much wall-clock has elapsed without
# the flag clearing. Thresholds chosen to be well above the slowest
# expected path: discovery 60s, fetch 600s (10min for huge scans).
_now = datetime.now()
_running_start_keys = {
    'discover_running': ('_discover_started_at', 60),
    'fetch_lots_running': ('_fetch_started_at', 600),
    '_sampling_pending': ('_sampling_started_at', 300),
}
for flag_key, (ts_key, max_seconds) in _running_start_keys.items():
    if st.session_state.get(flag_key):
        if not st.session_state.get(ts_key):
            st.session_state[ts_key] = _now
    else:
        st.session_state.pop(ts_key, None)


def _is_wedged(flag_key, max_seconds):
    ts = st.session_state.get(_running_start_keys[flag_key][0])
    if not ts:
        return False
    return (datetime.now() - ts).total_seconds() > max_seconds


_wedged_phase = None
for k, (_, secs) in _running_start_keys.items():
    if _is_wedged(k, secs):
        _wedged_phase = k
        break

if _wedged_phase:
    elapsed = (
        _now - st.session_state[_running_start_keys[_wedged_phase][0]]
    ).total_seconds()
    st.error(
        f"⚠️ **Stuck for {int(elapsed)}s** in `{_wedged_phase}`. "
        "Most likely a hung HiBid request that didn't honor the 15s "
        "timeout, or an unhandled exception. Click the reset button "
        "below to clear the wedged state and try again."
    )
    if st.button(
        "🔄 Reset wedged state and retry",
        type="primary",
        key="reset_wedged_state",
    ):
        # Clear ALL running flags + their timestamps so the next
        # rerun lands cleanly. Don't clear cached candidates / leads
        # — those are still useful.
        for k in list(_running_start_keys.keys()):
            st.session_state[k] = False
        for _, (ts_k, _) in _running_start_keys.items():
            st.session_state.pop(ts_k, None)
        st.session_state.pop('_auto_discover_triggered', None)
        st.rerun()
    st.stop()

# First-page-open detection: when no cached discovery exists AND
# auto-discover hasn't been triggered yet, we want the banner +
# sidebar loading card visible IMMEDIATELY (not after a 50-200ms
# rerun gap). We do this by flipping the real session-state flag
# and stopping the script — Streamlit picks up the flag on the next
# rerun and runs the actual discovery. Previously this code set a
# LOCAL discover_running=True for banner purposes only, which had
# a nasty side effect: any_running became True, which made the
# downstream auto-discover trigger refuse to fire, leaving the user
# stuck on "Loading auctions…" forever with no actual work running.
_first_open_pending_discover = (
    not st.session_state.get('_auto_discover_triggered', False)
    and not st.session_state.get('auction_candidates')
    and st.session_state.phase1_leads.empty
    and not discover_running
    and not fetch_lots_running
)

any_running = (
    discover_running or fetch_lots_running or _sampling_pending
    or _audit_running or _comps_running
)
_restored_at = st.session_state.get('_discovery_restored_from')

# --- STICKY TOP-OF-PAGE STATUS BANNER ---
# When ANY long-running operation is in flight, render a banner pinned to
# the top of the viewport that names the current phase. The detailed
# st.status panels below still drive minute-by-minute progress; this is
# a single-line "what is the algorithm doing right now" indicator the
# user can rely on regardless of scroll position.
#
# Special "just kicked off" branch: when the user just clicked the
# scan button, we render an extra-prominent variant for the first
# few seconds (before phase status panels have caught up) so the
# click is unambiguously confirmed. Detected via the
# _scan_just_started marker stamped at click time.
_scan_kickoff = st.session_state.get('_scan_just_started')
if _scan_kickoff:
    try:
        _kick_age = (
            datetime.now()
            - datetime.fromisoformat(_scan_kickoff.get('started_at', ''))
        ).total_seconds()
    except (ValueError, TypeError):
        _kick_age = 999
    # Keep the kickoff banner up for ~6 seconds; after that the regular
    # phase banner takes over. Either way clear the marker once Phase 2
    # is well underway so it doesn't linger.
    if _kick_age > 6 or not any_running:
        st.session_state.pop('_scan_just_started', None)
        _scan_kickoff = None

if any_running or _scan_kickoff:
    # CSS animation — pulse the left border so the banner reads as
    # "actively working" even without scrolling.
    st.markdown(
        """
        <style>
        @keyframes hto_pulse {
            0%   { border-left-color: #fbbf24; box-shadow: 0 0 0 0 rgba(251,191,36,0.5); }
            50%  { border-left-color: #f59e0b; box-shadow: 0 0 12px 4px rgba(251,191,36,0.25); }
            100% { border-left-color: #fbbf24; box-shadow: 0 0 0 0 rgba(251,191,36,0.5); }
        }
        .hto-running-banner {
            position: sticky; top: 3.5rem; z-index: 999;
            background: linear-gradient(90deg, #1e3a8a 0%, #1e40af 100%);
            color: white; padding: 6px 14px; border-radius: 4px;
            margin-bottom: 8px; font-size: 13px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.15);
            border-left: 4px solid #fbbf24;
            animation: hto_pulse 1.6s ease-in-out infinite;
        }
        .hto-kickoff-banner {
            position: sticky; top: 3.5rem; z-index: 1000;
            background: linear-gradient(90deg, #166534 0%, #15803d 60%, #16a34a 100%);
            color: white; padding: 14px 20px; border-radius: 8px;
            margin-bottom: 12px; font-size: 15px;
            box-shadow: 0 4px 16px rgba(22,163,74,0.35);
            border-left: 6px solid #facc15;
            animation: hto_pulse 1s ease-in-out infinite;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if _scan_kickoff:
        # Loud "click landed, scan starting" confirmation. Renders for
        # the first ~6 seconds after the button click, then yields
        # to the regular phase banner.
        _ac = _scan_kickoff.get('auction_count', 0)
        _lc = _scan_kickoff.get('lot_count', 0)
        _eta = _scan_kickoff.get('eta', '')
        st.markdown(
            f"""
            <div class="hto-kickoff-banner">
                <div style="font-weight: 700; font-size: 16px;">
                    🚀 BOLO scan kicked off — fetching {_ac} auctions ({_lc:,} lots)
                </div>
                <div style="font-size: 13px; opacity: 0.95; margin-top: 4px;">
                    ETA {_eta}. Phase 1 (HiBid GraphQL fetch) starting now —
                    detailed progress panel will appear below momentarily.
                    Don't refresh the page.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        if discover_running:
            if _first_open_pending_discover:
                _phase_label = "🔍 Loading auctions — first-time fetch from HiBid"
                _phase_detail = (
                    "No cached discovery found, so we're pulling the full "
                    "open-auction list now. This runs once per session "
                    "(~5–10s). The list will populate the sidebar as soon "
                    "as it lands."
                )
            else:
                _phase_label = "🔍 Discovering open auctions on HiBid"
                _phase_detail = "Querying nationwide + local listings, filtering by closing date…"
        elif _sampling_pending:
            _phase_label = "🔍 Loading content previews in background"
            _phase_detail = (
                "The auction list is already usable — go ahead and "
                "scan or click an auction. This step just fills in "
                "the per-auction preview column + BOLO hints."
            )
        elif fetch_lots_running:
            _scan_all = st.session_state.get('_bolo_scan_all_pending', False)
            if _scan_all:
                _phase_label = "📥 Fetching every lot for BOLO scan"
                _phase_detail = (
                    "Pulling all lots from HiBid GraphQL in batches of 20 "
                    "(free, no ScrapingBee credits). BOLO regex match runs "
                    "next once lots are in."
                )
            else:
                _phase_label = "📥 Fetching lots for selected auctions"
                _phase_detail = "Pulling lot data from HiBid GraphQL (free, no credits)…"
        elif _audit_running:
            _phase_label = "🧠 AI condition audit"
            _phase_detail = "Claude reviewing each lot's title/description for verdicts + red flags…"
        elif _comps_running:
            _phase_label = "💰 Price comps + sell-through rate"
            _phase_detail = (
                "Fetching eBay sold comps + STR (uses ScrapingBee credits). "
                "PriceCharting-routed lots are free."
            )
        else:
            _phase_label = "⏳ Working…"
            _phase_detail = ""
        st.markdown(
            f"""
            <div class="hto-running-banner">
                <div style="font-weight: 600;">{_phase_label}</div>
                <div style="font-size: 12px; opacity: 0.9; margin-top: 2px;">
                    {_phase_detail}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# Refresh-button context: when the user has drilled into an auction, the
# button should re-fetch JUST that auction (not re-run discovery against
# the entire HiBid catalog, which is what "Discover" / "Refresh" did
# historically). We detect that here so the header can render the right
# label and dispatch the right action.
_in_analysis_view = (
    bool(st.session_state.get('current_auction'))
    and isinstance(st.session_state.get('selected_leads'), pd.DataFrame)
    and not st.session_state.selected_leads.empty
)

# Phase-view refactor (7/9): nothing renders to st.sidebar anymore —
# the Discover content lives full-width in the main area and only
# renders when no auction is loaded. Hide the sidebar chrome (empty
# rail + expand arrow) unconditionally so Streamlit never shows the
# vestigial drawer.
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] {
        display: none !important;
    }
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapsedControl"],
    [data-testid="stExpandSidebarButton"] {
        display: none !important;
    }
    /* Trim Streamlit's default ~6rem top padding so the Discover
       controls + auction grid start near the top of the viewport. */
    .block-container {
        padding-top: 1.2rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

_HIBID_AUCTION_ID_RE = re.compile(
    # Captures the integer auction ID from any HiBid URL flavor:
    #   /livecatalog/735012/...    (current live-catalog view)
    #   /auction/735012            (legacy direct link)
    #   /catalog/735012/...        (older catalog)
    # Also accepts a bare number.
    r'(?:livecatalog|auction|catalog)/(\d{5,8})|^\s*(\d{5,8})\s*$',
    flags=re.IGNORECASE,
)


def _parse_hibid_auction_id(text: str):
    """Extract a HiBid auction ID from a URL or bare number.

    Returns the integer ID or None if the text doesn't match.
    """
    if not text:
        return None
    m = _HIBID_AUCTION_ID_RE.search(text)
    if not m:
        return None
    captured = m.group(1) or m.group(2)
    try:
        return int(captured)
    except (TypeError, ValueError):
        return None


def _hibid_popover_label() -> str:
    """Compute the 💰 My Bids popover label inline.

    Cheap (session_state lookup + a filter count) so no cache layer
    needed — runs on every script rerun. Shows the count of currently-
    OPEN bids when we have cached data, falls through to 🔐 when the
    user hasn't pasted a HiBid token.
    """
    try:
        from pathlib import Path
        has_token = (Path(".cache") / "hibid_session.json").exists()
        if not has_token:
            return "💰 My Bids 🔐"
        cache = st.session_state.get("_hibid_live_cache")
        if not cache:
            return "💰 My Bids"
        lots = cache.get("lots") or []
        n_open = sum(
            1 for l in lots
            if not l.get("is_closed") and not l.get("is_archived")
        )
        if n_open:
            return f"💰 My Bids ({n_open})"
        return "💰 My Bids"
    except Exception:
        return "💰 My Bids"


def _compute_max_bid_single(est_resale, auction_buyer_premium_pct,
                            ship_cost, target_roi):
    """Single-lot max-bid calculator — mirrors `_compute_max_bid`.

    Returns the max bid amount (rounded $) such that paying it lands
    at exactly target_roi × cost. Returns None when:
      - est_resale missing / non-positive
      - shipping + premium eat the entire margin (negative max_bid)

    HiBid's `auction_buyer_premium_pct` is the multiplier form
    (1.18 = 18% premium). When HiBid returns 1.0 the rate wasn't
    parseable from the auction's free-text buyer-premium description —
    fall back to cfg default ~1.15 so we don't massively over-state
    the stop signal.
    """
    if est_resale is None:
        return None
    try:
        est_resale = float(est_resale)
    except (TypeError, ValueError):
        return None
    if est_resale <= 0:
        return None
    EBAY_FEE_PCT = 0.1325
    EBAY_FEE_FLAT = 0.30
    net_resale = est_resale * (1 - EBAY_FEE_PCT) - EBAY_FEE_FLAT
    bp_mult = float(auction_buyer_premium_pct or 1.0)
    if bp_mult <= 1.0:
        bp_mult = 1.15
    # Shipping excluded from bid guidance (7/10) — consistent with
    # the main results table; ship_cost param retained for signature
    # compatibility but no longer subtracted.
    max_bid = (net_resale / float(target_roi)) / bp_mult
    return round(max_bid, 2) if max_bid > 0 else None


def _render_live_hibid_view():
    """Render the contents of the 💰 My Bids popover.

    Primary view: currently-OPEN HiBid bids with a STOP / NEAR MAX /
    GO indicator per lot, computed by comparing the current high_bid
    to the max_bid that hits target ROI for the lot's est_resale
    (pulled from the comped_lots registry). The user explicitly
    asked to keep live bids visible — token-management setup lives
    in a collapsed expander at the bottom.
    """
    _hibid_account = _hibid_module()
    st.markdown("### 💰 My Bids")

    token_meta = _hibid_account.session_metadata()
    has_token = bool(token_meta.get("has_token"))

    if not has_token:
        st.warning(
            "🔐 Not signed in. Paste a HiBid authToken in **Setup** "
            "below to see your live bids and stop signals."
        )
        st.divider()
        _render_hibid_setup_panel(_hibid_account, expand_token=True)
        return

    # ---- Refresh row + cache age ----------------------------------
    cache = st.session_state.get("_hibid_live_cache")
    cache_ts = st.session_state.get("_hibid_live_cache_ts", 0.0)
    cache_age_s = (_time.time() - cache_ts) if cache_ts else None

    rb1, rb2 = st.columns([1, 3])
    with rb1:
        if st.button(
            "🔄 Refresh", key="hibid_live_refresh",
            help="Pull fresh HiBid bids now (bypasses the 5-min "
                 "background sync).",
            width='stretch',
        ):
            st.session_state.pop("_hibid_autosync_at", None)
            st.session_state.pop("_hibid_live_cache", None)
            st.session_state.pop("_hibid_live_cache_ts", None)
            _maybe_auto_sync_hibid(force=True)
            st.rerun()
    with rb2:
        if cache_age_s is None:
            st.caption("_No bid data cached — click Refresh._")
        else:
            mins = int(cache_age_s // 60)
            secs = int(cache_age_s % 60)
            st.caption(f"Last sync: {mins}m {secs}s ago")

    # ---- Stop-signal table ---------------------------------------
    if cache:
        _render_active_bids_with_stop_signal(cache)

    # ---- Setup section (collapsed) -------------------------------
    st.divider()
    _render_hibid_setup_panel(_hibid_account, expand_token=False)


def _render_active_bids_with_stop_signal(cache):
    """Show open HiBid bids + STOP / NEAR MAX / GO per lot.

    Status logic (per lot):
      - missing est_resale  → ❓ NO COMP (can't compute max)
      - high_bid >= max_bid → 🛑 STOP (next bid loses money at target ROI)
      - high_bid >= 85% max → 🟡 NEAR MAX (small headroom, watch closely)
      - else                → 🟢 GO (safe to bid up to max)

    max_bid uses the same formula as `_compute_max_bid` in the main
    results table: backs out from net resale (after eBay fees) /
    target_roi, less shipping, divided by the auction's buyer's
    premium multiplier.
    """
    from scraper.config_loader import load_config

    cfg = load_config()
    ship_cost_default = float(
        cfg.get("shipping", {}).get("bundled_ship_cost", 25.0)
    )
    target_roi = float(st.session_state.get("target_roi_live", 3.0) or 3.0)

    auctions = cache.get("auctions") or []
    auctions_by_id = {a.get("id"): a for a in auctions}
    lots = cache.get("lots") or []

    # Filter: open = not closed AND not archived. The cached payload
    # spans 3 months including closed lots (used by auto-record).
    open_lots = [
        l for l in lots
        if not l.get("is_closed") and not l.get("is_archived")
    ]

    if not open_lots:
        st.info("No open HiBid bids right now. ✅")
        # Compact tail: surface recent closures so the user knows the
        # cache is alive — without rebuilding the (deleted) outcomes
        # dashboard.
        won = sum(1 for l in lots if l.get("may_have_won"))
        lost = sum(1 for l in lots
                   if l.get("is_closed") and not l.get("may_have_won"))
        if won or lost:
            st.caption(
                f"Recent closed (3 mo): 🏆 {won} won · ❌ {lost} lost"
            )
        return

    # Score each open lot
    rows = []
    for lot in open_lots:
        lot_id = str(lot.get("lot_id"))
        title = (lot.get("title") or "(untitled)").strip()
        high_bid = float(lot.get("high_bid") or 0)
        my_bid = float(lot.get("my_bid") or 0)
        bid_status = lot.get("bid_status") or "?"
        time_left = (lot.get("time_left") or "").strip()
        auction_id = lot.get("auction_id")
        bp_mult = float(lot.get("auction_buyer_premium_pct") or 1.0)
        auction = auctions_by_id.get(auction_id) or {}
        auction_name = (auction.get("eventName")
                        or f"Auction {auction_id}")

        # Pull est_resale from the current session's audit_results
        # (if any). With the cross-session comped_lots registry gone,
        # the stop signal only fires on lots the user has analyzed
        # in the current session — uncomped lots show "❓ NO COMP".
        est_resale = _lookup_est_resale_in_session(lot_id)
        max_bid = _compute_max_bid_single(
            est_resale, bp_mult, ship_cost_default, target_roi,
        )

        if est_resale is None or max_bid is None:
            status = "no_comp"
        elif high_bid >= max_bid:
            status = "stop"
        elif high_bid >= 0.85 * max_bid:
            status = "caution"
        else:
            status = "go"

        rows.append({
            "lot_id": lot_id,
            "title": title,
            "auction_name": auction_name,
            "auction_id": auction_id,
            "high_bid": high_bid,
            "my_bid": my_bid,
            "bid_status": bid_status,
            "time_left": time_left,
            "est_resale": float(est_resale) if est_resale else None,
            "max_bid": max_bid,
            "status": status,
        })

    # Summary chips at the top
    n_stop = sum(1 for r in rows if r["status"] == "stop")
    n_caution = sum(1 for r in rows if r["status"] == "caution")
    n_go = sum(1 for r in rows if r["status"] == "go")
    n_nocomp = sum(1 for r in rows if r["status"] == "no_comp")
    chip_bits = []
    if n_stop:
        chip_bits.append(f"🛑 **{n_stop} stop**")
    if n_caution:
        chip_bits.append(f"🟡 {n_caution} near max")
    if n_go:
        chip_bits.append(f"🟢 {n_go} go")
    if n_nocomp:
        chip_bits.append(f"❓ {n_nocomp} uncomped")
    st.markdown(
        " · ".join(chip_bits) if chip_bits else f"{len(rows)} active bids"
    )
    st.caption(
        f"Stop signal uses target ROI **{target_roi:.1f}×** "
        f"(set in the comps panel) and ${ship_cost_default:.0f} default "
        "ship cost. Adjust ROI to retighten."
    )

    # ---- CSV export of the live-bid view ---------------------------
    # Same in-memory data we use to render the cards, materialized
    # to a flat DataFrame. Includes derived columns (status_label,
    # headroom, lot_url) so the CSV is useful in Excel without
    # post-processing.
    status_label_map = {
        "stop": "STOP",
        "caution": "NEAR MAX",
        "go": "GO",
        "no_comp": "NO COMP",
    }
    export_rows = []
    for r in rows:
        max_bid = r["max_bid"]
        high_bid = r["high_bid"]
        export_rows.append({
            "status": status_label_map[r["status"]],
            "lot_id": r["lot_id"],
            "title": r["title"],
            "auction_name": r["auction_name"],
            "auction_id": r["auction_id"],
            "current_high": high_bid,
            "my_bid": r["my_bid"],
            "bid_status": r["bid_status"],
            "time_left": r["time_left"],
            "est_resale": r["est_resale"],
            "max_bid": max_bid,
            "headroom": (
                round(max_bid - high_bid, 2)
                if max_bid is not None else None
            ),
            "lot_url": f"https://hibid.com/lot/{r['lot_id']}",
        })
    export_df = pd.DataFrame(export_rows)
    csv_bytes = export_df.to_csv(index=False).encode("utf-8")
    fname = f"htown-hibid-live-{datetime.now():%Y%m%d-%H%M}.csv"
    st.download_button(
        f"📥 Export {len(export_rows)} live bids to CSV",
        data=csv_bytes,
        file_name=fname,
        mime="text/csv",
        key="hibid_live_export_csv",
        width="stretch",
        help="Download the table above with one row per open bid, "
             "plus status + headroom columns for spreadsheet filtering.",
    )

    # Sort: stop first (most urgent), then near-max, no-comp, go last
    sort_priority = {"stop": 0, "caution": 1, "no_comp": 2, "go": 3}
    rows.sort(key=lambda r: (sort_priority[r["status"]], r["title"]))

    status_label = {
        "stop": "🛑 STOP",
        "caution": "🟡 NEAR MAX",
        "go": "🟢 GO",
        "no_comp": "❓ NO COMP",
    }
    for r in rows:
        # Leading vs outbid is the FIRST visual signal — bigger anxiety
        # driver than the stop signal itself. When someone else bid $590
        # on a lot where the user only bid $50, the user shouldn't have
        # to puzzle out "is that 'Current high $590' MY bid or theirs?"
        # — the chip + plain-language summary line below resolve it.
        is_leading = r["bid_status"] == "WINNING"
        is_outbid = r["bid_status"] == "OUTBID"
        leading_chip = (
            "🏆 LEADING" if is_leading
            else "🚨 OUTBID" if is_outbid
            else f"❓ {r['bid_status']}"
        )

        with st.container(border=True):
            c1, c2 = st.columns([4, 1])
            with c1:
                st.markdown(
                    f"**{leading_chip}** · "
                    f"**{status_label[r['status']]}** — "
                    f"{r['title'][:75]}"
                )
                tail = f" · ⏱ {r['time_left']}" if r["time_left"] else ""
                st.caption(f"{r['auction_name'][:55]}{tail}")
                # Plain-language summary — the anti-anxiety line.
                # Reframes "current high $590 / max $92 / -$498 over"
                # so the user sees instantly that the $590 is the
                # LEADER's bid, not theirs.
                if is_leading:
                    if (r["max_bid"] is not None
                            and r["my_bid"] is not None
                            and r["my_bid"] > r["max_bid"]):
                        over = r["my_bid"] - r["max_bid"]
                        summary = (
                            f"🏆 **You're leading at "
                            f"${r['my_bid']:,.0f}** — "
                            f"but ${over:,.0f} over your max"
                        )
                    else:
                        summary = (
                            f"🏆 **You're leading at "
                            f"${r['my_bid']:,.0f}**"
                        )
                elif is_outbid:
                    gap = r["high_bid"] - (r["my_bid"] or 0)
                    summary = (
                        f"🚨 Leader is at **${r['high_bid']:,.0f}** "
                        f"· you bid ${r['my_bid'] or 0:,.0f} "
                        f"(${gap:,.0f} below)"
                    )
                else:
                    summary = f"Current high: ${r['high_bid']:,.0f}"
                st.markdown(summary)
            with c2:
                lot_url = f"https://hibid.com/lot/{r['lot_id']}"
                st.markdown(f"[Open ↗]({lot_url})")

            cm1, cm2, cm3 = st.columns(3)
            with cm1:
                # Always primary: YOUR bid. Label changes based on
                # leading state so the user knows whether the dollar
                # figure is "you're winning" or "you got outbid".
                first_label = (
                    "Your bid (leading)" if is_leading
                    else "Your bid (outbid)" if is_outbid
                    else "Your bid"
                )
                first_value = r["my_bid"] if r["my_bid"] is not None else 0
                st.metric(first_label, f"${first_value:,.0f}")
            with cm2:
                if r["max_bid"] is not None:
                    headroom = r["max_bid"] - r["high_bid"]
                    # "normal" colors negative deltas red-down, positive
                    # green-up. Headroom uses the leader's high_bid as
                    # the floor — meaningful for both leading (your
                    # cushion before someone else outbids you past
                    # max) and outbid (whether you can counter-bid
                    # without going over).
                    if headroom > 0:
                        delta_text = f"+${headroom:,.0f} room"
                    else:
                        delta_text = f"-${abs(headroom):,.0f} over"
                    st.metric(
                        "Max bid", f"${r['max_bid']:,.0f}",
                        delta=delta_text, delta_color="normal",
                    )
                else:
                    st.metric("Max bid", "—",
                              delta="no comp", delta_color="off")
            with cm3:
                if r["est_resale"]:
                    st.metric("Est resale", f"${r['est_resale']:,.0f}")
                else:
                    st.metric("Est resale", "—")


def _render_hibid_setup_panel(_hibid_account, *, expand_token: bool):
    """Render the HiBid token-management expander below the live-bid view."""
    token_meta = _hibid_account.session_metadata()
    has_token = bool(token_meta.get("has_token"))

    with st.expander(
        "🔐 HiBid token",
        expanded=expand_token and not has_token,
    ):
        if has_token:
            try:
                acct = _hibid_account.fetch_account_info()
                st.success(
                    f"Signed in as **{acct.get('username') or '?'}**"
                )
            except Exception as e:
                st.error(f"⚠️ Token invalid: {e}")
            st.caption(
                f"Saved {token_meta.get('saved_at')} · "
                f"{token_meta.get('token_length')} chars"
            )
            if st.button("🗑️ Clear token", key="hibid_clear_token"):
                _hibid_account.clear_auth_token()
                st.session_state.pop("_hibid_live_cache", None)
                st.session_state.pop("_hibid_live_cache_ts", None)
                st.session_state.pop("_hibid_autosync_at", None)
                st.rerun()
        else:
            st.caption(
                "Log into hibid.com → DevTools → Network → /graphql "
                "POST → `GetAccountInfo` response → copy the `authToken` "
                "field (long hex string ~768 chars)."
            )
        new_token = st.text_area(
            "Paste HiBid authToken",
            value="",
            height=100,
            key="hibid_token_paste",
        )
        if st.button(
            "💾 Save token", key="hibid_save_token",
            disabled=not (new_token or "").strip(),
            width="stretch",
        ):
            _hibid_account.set_auth_token(new_token)
            tlog("HIBID", "authToken saved")
            st.session_state.pop("_hibid_autosync_at", None)
            st.rerun()


# ---------------------------------------------------------------------
# Background HiBid sync — fires on page render with debounce
# ---------------------------------------------------------------------
# Default: every 5 minutes since the last successful sync. Without this,
# auto-record only fired when the user opened the (now-deleted) Live
# HiBid bids tab. Now it runs invisibly: token saved + auto-record
# toggled on + 5min elapsed → fetch + auto-record.

_HIBID_AUTOSYNC_INTERVAL_SECONDS = 300  # 5 minutes


def _maybe_auto_sync_hibid(force: bool = False) -> None:
    """Pull HiBid current bids into the live cache, debounced.

    Safe to call on every Streamlit script run. Returns immediately
    when no HiBid token is saved or the last sync was within
    `_HIBID_AUTOSYNC_INTERVAL_SECONDS`. Pass ``force=True`` to bypass
    the time-window check.
    """
    _hibid_account = _hibid_module()
    if not _hibid_account.session_metadata().get("has_token"):
        return
    last = st.session_state.get("_hibid_autosync_at", 0.0)
    now = _time.time()
    if not force and (now - last) < _HIBID_AUTOSYNC_INTERVAL_SECONDS:
        return
    try:
        cache_full = _hibid_account.fetch_all_current_bids(
            only_open=False, only_winning=False,
        )
        st.session_state._hibid_live_cache = cache_full
        st.session_state._hibid_live_cache_ts = now
        st.session_state._hibid_autosync_at = now
    except Exception as e:
        # Don't crash the page over a network blip — log + back off.
        tlog("HIBID", f"auto-sync failed: {type(e).__name__}: {e}")
        st.session_state._hibid_autosync_at = now


# ---------------------------------------------------------------------
# TOP BAR: title + Sourcing / Memory / My Bids popovers + sidebar toggle.
# Defined here at module scope so the widget return values
# (user_zip, radius, include_nationwide, closing_days, category_filter)
# are visible to `_render_sidebar_refresh_button()` later in the script
# — Streamlit widget returns are scoped to the `with` block they're
# defined in, but module-level assignments stay in module-level scope.
# Placed after the My Bids helper functions so all referenced helpers
# (_hibid_popover_label, _render_live_hibid_view) are already defined.
# ---------------------------------------------------------------------
header_title_col, header_actions_col = st.columns([3, 4])
with header_title_col:
    st.markdown("## 🛰️ Auction Intelligence Dashboard")

# Sourcing controls live at the TOP of the sidebar, above the auction
# list. Previous design put them in a popover in the header — but users
# typically tweak sourcing settings BEFORE scanning the auction list, so
# having them above the list is more natural. Widget keys stay the same
# (`sb_*`) so session state persists across the move. Variable
# assignments are at module scope and remain visible to downstream
# refresh logic (see line ~2250 where they're bundled into _sourcing_cfg).
# ================================================================
# PHASE VIEWS
# The app has two workflow phases and renders one at a time:
#   🔎 DISCOVER — no auction loaded. Sourcing controls + scan
#      buttons + the auction grid render FULL-WIDTH in the main
#      area (this replaced the old persistent left sidebar, which
#      was only ever useful up-front and then ate 300px of table
#      width for the rest of the session).
#   🔬 ANALYZE — an auction/scan is loaded. Nothing renders to the
#      sidebar or the discover area; the results table gets the
#      whole viewport. "← Back to auctions" returns to Discover.
# During a fetch (fetch_lots_running) the discover UI also hides so
# the phase status panels sit at the top of the page.
# ================================================================
_discover_phase = (
    not st.session_state.get('current_auction')
    and not st.session_state.get('fetch_lots_running', False)
)

# Fixed sourcing values (were sliders/multiselects the user never
# changed — removed 7/9 to reclaim vertical space in Discover):
closing_days = 1        # always "closing within 1 day"
category_filter = []    # category filter removed — fetch everything
if _discover_phase:
    # Collapsible sourcing + filters drawer. Sits ABOVE the search
    # bar, collapsed by default — these settings rarely change
    # mid-session. Row 1 (sourcing) renders here at module level so
    # the variable assignments stay module-scoped; row 2 (list
    # filters + refresh) is appended into the SAME expander object
    # by the auction-list renderer.
    _discover_filters_exp = st.expander(
        "⚙️ Sourcing & filters", expanded=False,
    )
    with _discover_filters_exp:
        _src_c1, _src_c2, _src_c3, _src_spacer = st.columns(
            [1.0, 1.2, 1.6, 3.2]
        )
        with _src_c1:
            user_zip = st.text_input(
                "Home Zip", value="77058", key="sb_user_zip",
            )
        with _src_c2:
            radius = st.number_input(
                "Pickup radius (mi)", min_value=5, max_value=100,
                value=20, step=5, key="sb_radius",
            )
        with _src_c3:
            include_nationwide = st.checkbox(
                "Include Nationwide (Ship-to-Me)", value=True,
                key="sb_include_nationwide",
            )
    radius = int(radius or 20)
else:
    # Analyze phase / fetch in flight: no sourcing widgets rendered,
    # but the discover work blocks still read these variables — pull
    # the persisted values straight from session state.
    user_zip = st.session_state.get('sb_user_zip', '77058') or '77058'
    radius = int(st.session_state.get('sb_radius', 20) or 20)
    include_nationwide = bool(
        st.session_state.get('sb_include_nationwide', True)
    )

with header_actions_col:
    pop_memory, pop_mybids = st.columns(2)

    with pop_memory:
        with st.popover("💾 Memory", width='stretch'):
            cached_list = _AUCTION_CACHE.list_all(
                ttl_days=st.session_state.cache_ttl_days
            )
            fresh_count = sum(1 for c in cached_list if c['fresh'])
            st.caption(
                f"**{fresh_count}** auction(s) cached. Audit + comp "
                "results are reused when you re-open an auction."
            )
            st.session_state.cache_ttl_days = st.slider(
                "Auto-purge after (days)",
                min_value=1, max_value=30,
                value=int(st.session_state.cache_ttl_days),
                key="memory_cache_ttl_days",
            )
            if cached_list:
                with st.expander(
                    f"📋 Open a cached auction ({len(cached_list)} entries)",
                    expanded=False,
                ):
                    st.caption(
                        "Click any entry to re-open its previous "
                        "analysis — zero credits, audit + comp data "
                        "read from disk."
                    )
                    for entry in cached_list[:25]:
                        aid = entry.get('auction_id')
                        auction_name = entry.get('auction_name', '(unknown)')
                        items = entry.get('items', 0)
                        fresh = entry.get('fresh', False)
                        badge = "🟢" if fresh else "🔴"
                        try:
                            cached_at = datetime.fromisoformat(entry['cached_at'])
                            age = datetime.now() - cached_at
                            age_str = (
                                f"{age.days}d ago" if age.days > 0
                                else f"{int(age.seconds / 3600)}h ago"
                            )
                        except Exception:
                            age_str = "?"
                        if st.button(
                            f"{badge} **{auction_name}** — "
                            f"{items} items · {age_str}",
                            key=f"open_cached_{aid}",
                            width='stretch',
                        ):
                            if aid is not None:
                                st.session_state._selected_auction_ids = [aid]
                                st.session_state.current_auction = None
                                st.session_state.selected_leads = pd.DataFrame()
                                st.session_state.phase1_leads = pd.DataFrame()
                                st.session_state.fetch_lots_running = True
                                st.rerun()
                    if len(cached_list) > 25:
                        st.caption(
                            f"...and {len(cached_list) - 25} more "
                            "(showing newest 25)."
                        )
            if st.button(
                "🗑️ Clear all memory", width='stretch',
                help="Delete every cached auction analysis.",
            ):
                removed = _AUCTION_CACHE.clear_all()
                st.success(f"Cleared {removed} cached auction(s).")
                st.rerun()

    with pop_mybids:
        with st.popover(_hibid_popover_label(), width='stretch'):
            _render_live_hibid_view()

    # (The old sidebar show/hide toggle was removed with the phase-view
    # refactor — the auction list now renders full-width in the main
    # area during the Discover phase and disappears entirely in the
    # Analyze phase.)



# ================================================================
# SIDEBAR: persistent auction list. Always visible while there are
# discovered candidates. Clicking an auction row triggers a single-
# auction fetch (reuses the same path as the in-page picker's
# "Analyze" button). Selected auction gets a visual highlight.
# ================================================================
# ================================================================
# AUCTION TRIAGE — pre-comp signals so the user can spend ScrapingBee
# credits on auctions that are likely to find arbitrage and skip the
# obvious losers without burning credits to discover that.
#
# Two functions:
#   _estimate_auction_cost(lot_count, sample_payload) -> (pc_pct, credits)
#   _auction_signal(name, items, closes_dt, last_run_dt) -> (rank, reason)
#
# Both run on the existing post-discovery sample data — zero new HTTP.
# ================================================================
# Approximate ScrapingBee cost per non-PC-covered lot. Each lot kicks
# off an eBay-sold scrape (~25 credits) plus a Mercari-sold scrape
# (~25 credits) plus possibly a per-item STR scrape for cards/games/
# comics (~25 credits). Average ~50 credits/lot is a reasonable
# all-in estimate at user volume.
_CREDITS_PER_NON_PC_LOT = 50


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_scrapingbee_usage(api_key: str) -> dict:
    """Hit ScrapingBee's /usage endpoint to get current credits.

    Cached 5 minutes so we don't ping it on every Streamlit rerun.
    Returns {} on any failure (no key, network error, 401, etc.) —
    callers handle the empty-dict case gracefully.
    """
    if not api_key:
        return {}
    try:
        import httpx
        r = httpx.get(
            "https://app.scrapingbee.com/api/v1/usage",
            params={"api_key": api_key}, timeout=8,
        )
        if r.status_code == 200:
            return r.json() or {}
    except Exception:
        pass
    return {}


def _estimate_comp_cost_for_audit(audit_df) -> tuple:
    """Return (eligible_count, est_credits, pc_pct) for the lots the
    next comp run would actually process.

    Eligibility: not red-flagged AND est_resale is currently null
    (un-comped). PC-covered lots cost 0 credits; everything else
    burns ~50 credits (eBay sold + Mercari sold).

    Per-title PC classification is cached in session state so
    spend-cap knob bumps don't re-run the classifier for every
    keystroke.
    """
    from scraper.pricecharting import classify_for_pricecharting
    if not isinstance(audit_df, pd.DataFrame) or audit_df.empty:
        return 0, 0, 0.0
    not_red = ~audit_df.get('red_flag', pd.Series(False, index=audit_df.index)).fillna(False).astype(bool)
    not_done = (
        audit_df['est_resale'].isna()
        if 'est_resale' in audit_df.columns
        else pd.Series(True, index=audit_df.index)
    )
    eligible = audit_df[not_red & not_done]
    if eligible.empty:
        return 0, 0, 0.0
    title_col = 'enriched_title' if 'enriched_title' in eligible.columns else 'title'
    titles = eligible[title_col].fillna('').astype(str).tolist()
    try:
        _pc_cache = st.session_state.setdefault('_pc_classify_cache', {})
    except Exception:
        _pc_cache = {}

    def _is_pc(t: str) -> bool:
        cached = _pc_cache.get(t)
        if cached is None:
            cached = classify_for_pricecharting(t) in (
                'tcg', 'video_game', 'comic',
            )
            _pc_cache[t] = cached
        return cached

    pc_hits = sum(1 for t in titles if _is_pc(t))
    pc_pct = pc_hits / max(len(titles), 1)
    est_credits = int(len(titles) * (1 - pc_pct) * _CREDITS_PER_NON_PC_LOT)
    return len(titles), est_credits, pc_pct


def _estimate_auction_cost(lot_count: int, sample_payload):
    """Return (pc_coverage_pct, est_credits) for a discovered auction.

    pc_coverage is the share of sampled lot titles that route through
    PriceCharting (TCG/video_game/comic) — those don't burn ScrapingBee
    credits. sports_card titles classify but bypass PC and route to
    eBay-sold scraping, so they DO cost.

    With no sample yet, assumes worst case (everything costs).
    """
    from scraper.pricecharting import classify_for_pricecharting
    titles = (sample_payload or {}).get('titles') or []
    if not titles:
        return 0.0, int(lot_count * _CREDITS_PER_NON_PC_LOT)
    pc_hits = sum(
        1 for t in titles
        if classify_for_pricecharting(t or '')
           in ('tcg', 'video_game', 'comic')
    )
    pc_coverage = pc_hits / len(titles)
    est_credits = int(lot_count * (1 - pc_coverage) * _CREDITS_PER_NON_PC_LOT)
    return pc_coverage, est_credits


# Auction-name keywords that signal "avoid" — base products are too
# commoditized for arbitrage even when prices are accurate.
_AVOID_KEYWORDS = (
    'liquidation', 'overstock', 'new returns', 'warehouse',
    'wholesale', 'pallet', 'truckload', 'amazon return',
)
# Name keywords that signal "good" — variety + older inventory means
# more chance of finding an underpriced item.
_PREFER_KEYWORDS = (
    'estate', 'storage', 'collection', 'consignment', 'antique',
    'vintage',
)

# Auctioneer-template / AI-generated listing copy patterns. When >30%
# of sample lot titles contain one of these phrases, the auction is
# almost certainly a dropship operation (mass-uploaded fake-vintage
# inventory with computer-generated descriptions). Real auctioneer
# descriptions don't have boilerplate phrases like "Hook Discover Item
# Details Age" repeated across dozens of lots.
#
# Confirmed examples observed in the wild:
# - 'Vintage & Collectible Sale — Medals, Porcelain' had "Uncommon
#   Vintage" in 46% of titles + "Hook Discover" in 31%
# - 'New Pokemon Cards 03-05-2026' had "Pokemon Cards English" template
#   with auction-marketing adjectives in 100% of titles
_DROPSHIP_TEMPLATE_PHRASES = (
    'uncommon vintage',
    'rare vintage',
    'unique vintage',
    'hook discover',
    'item details age',
    'item details type',
    'introduction discover',
    'introduction elevate',
    'introduction add',
    'introduction this',
    'a rare find',
    'why this item',
    'step item details',
    'pokemon cards english',
    'cards english',
)
_DROPSHIP_THRESHOLD_RATIO = 0.30  # 30% of sample titles → flag as dropship


def _detect_dropship_signal(sample_titles):
    """Return (is_dropship, ratio, top_phrase) for a list of sample titles.

    Computes the fraction of titles containing ANY of the auctioneer-
    template phrases. When the fraction crosses _DROPSHIP_THRESHOLD_RATIO,
    we flag the auction as a dropship/AI-listing operation.

    Returns:
        (False, 0.0, None) when sample is empty or below threshold
        (True, ratio_as_float, top_matching_phrase) when at/above threshold
    """
    titles = [t for t in (sample_titles or []) if isinstance(t, str)]
    if len(titles) < 5:  # need a meaningful sample to flag confidently
        return False, 0.0, None
    lower = [t.lower() for t in titles]
    matched_titles = 0
    phrase_counts = {}
    for title in lower:
        hit_in_this_title = False
        for phrase in _DROPSHIP_TEMPLATE_PHRASES:
            if phrase in title:
                phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1
                hit_in_this_title = True
        if hit_in_this_title:
            matched_titles += 1
    ratio = matched_titles / len(titles)
    if ratio < _DROPSHIP_THRESHOLD_RATIO:
        return False, ratio, None
    top_phrase = max(phrase_counts.items(), key=lambda kv: kv[1])[0] if phrase_counts else None
    return True, ratio, top_phrase


# ---------------------------------------------------------------------
# 🕵️ Sleeper-hunt candidate scoring.
# A "sleeper" is a container-titled lot ("Jewelry Box", "Lot of Toys",
# "Estate Assortment") whose value is only visible in the photos —
# invisible to the text-only BOLO matcher. Candidates are gated on a
# container-style title, then scored on signals that are FREE because
# they're already in the Phase-1 fetch. Only the top N by score get
# vision enrichment (real but tiny API cost — eBay image search is
# free tier 1, Claude Haiku vision fallback is ~$0.005/lot).
# ---------------------------------------------------------------------
_SLEEPER_CONTAINER_RE = re.compile(
    r"\b(?:box(?:es)?|lot|lots|tray|bin|bag|basket|flat|case|drawer|"
    r"tote|tub|crate)\s+(?:of|full|w/|with)\b"
    r"|\b(?:assorted|assortment|mixed|misc|miscellaneous|various|"
    r"estate|mystery|grab\s*bag|junk\s*drawer|unsorted)\b"
    r"|\bjewelry\s+box(?:es)?\b|\bjewelry\s+lot\b",
    re.IGNORECASE,
)
_SLEEPER_DESC_TREASURE_RE = re.compile(
    r"\b(?:sterling|925|1[048]k|22k|24k|hallmark\w*|signed|stamped|"
    r"marked|coins?|gold|silver|turquoise|military|antique|bakelite)\b",
    re.IGNORECASE,
)
_SLEEPER_CATEGORY_RE = re.compile(
    r"jewelr|coin|watch|collectib|antique|sterling|gold|silver|toy",
    re.IGNORECASE,
)


def _score_sleeper_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Return container-titled rows scored + sorted by treasure signals.

    Gate: title must look like a container lot. Score components (all
    free — already in the fetch): HiBid category, auctioneer estimate,
    bid activity, photo count, treasure words in the description.
    Rows scoring < 2 are dropped — zero-signal containers aren't worth
    even a cheap vision call.
    """
    if df is None or df.empty or 'title' not in df.columns:
        return pd.DataFrame()
    titles = df['title'].fillna('').astype(str)
    container_mask = titles.str.contains(_SLEEPER_CONTAINER_RE, na=False)
    cand = df[container_mask].copy()
    if cand.empty:
        return cand
    score = pd.Series(0, index=cand.index)
    if 'category' in cand.columns:
        score += cand['category'].fillna('').astype(str).str.contains(
            _SLEEPER_CATEGORY_RE, na=False).astype(int) * 3
    ae = pd.to_numeric(cand.get('auctioneer_est_high'), errors='coerce')
    if ae is not None:
        score += (ae.fillna(0) >= 50).astype(int) * 2
        score += (ae.fillna(0) >= 200).astype(int)
    bc = pd.to_numeric(cand.get('bid_count'), errors='coerce')
    if bc is not None:
        score += (bc.fillna(0) >= 1).astype(int)
        score += (bc.fillna(0) >= 3).astype(int)
    ic = pd.to_numeric(cand.get('image_count'), errors='coerce')
    if ic is not None:
        score += (ic.fillna(0) >= 5).astype(int)
    if 'description' in cand.columns:
        score += cand['description'].fillna('').astype(str).str.contains(
            _SLEEPER_DESC_TREASURE_RE, na=False).astype(int) * 2
    cand['_sleeper_score'] = score
    cand = cand[cand['_sleeper_score'] >= 2]
    return cand.sort_values('_sleeper_score', ascending=False)


# Canadian-province codes — auctions located in Canada are flagged
# 'avoid' for a US-based buyer because most don't ship to US (or
# import/clearance fees make it uneconomical).
_CANADIAN_PROVINCE_CODES = frozenset({
    'ON', 'QC', 'BC', 'AB', 'MB', 'SK',
    'NS', 'NB', 'NL', 'PE', 'YT', 'NT', 'NU',
})


def _auction_signal(name, items, closes_dt, last_run_dt,
                    sample_titles=None, state=None, city=None):
    """Return (rank, reason) where rank is 'good' | 'caution' | 'avoid'.

    Heuristics, ranked first-match-wins:
      - 'avoid': Canadian-located auction (most don't ship to US),
        name matches commoditized-source keywords, dropship template
        language detected in lot titles, lot count > 1500, or closing
        in <2 hours.
      - 'caution': already analyzed in the last 24h, or outside the
        100-500 lot sweet spot but otherwise fine.
      - 'good': in the sweet spot, optionally with a 'prefer' keyword.
    """
    nm = (name or '').lower()
    state_upper = (state or '').strip().upper()
    if state_upper and state_upper in _CANADIAN_PROVINCE_CODES:
        return 'avoid', (
            f"Located in {state_upper} (Canada) — most Canadian "
            "auctioneers don't ship to US"
        )
    if (
        'canada' in nm
        or ' canadian ' in f' {nm} '
        or 'ontario' in nm or 'quebec' in nm
        or 'british columbia' in nm or 'alberta' in nm
    ):
        return 'avoid', (
            "Auction name suggests Canadian location — most don't "
            "ship to US"
        )
    for kw in _AVOID_KEYWORDS:
        if kw in nm:
            return 'avoid', f"Name contains '{kw}' — base prices typically too compressed for arbitrage"
    # Dropship / AI-generated listing detector. Runs on the cat_samples
    # titles passed in by the sidebar builder. Catches auctions where
    # >30% of titles share auctioneer-template / dropship boilerplate
    # like "Uncommon Vintage" + "Hook Discover" — these are almost
    # always low-quality sourcing (Asian decor reproductions, niche
    # militaria, AI-generated descriptions).
    if sample_titles:
        is_dropship, ratio, top_phrase = _detect_dropship_signal(sample_titles)
        if is_dropship:
            pct = int(round(ratio * 100))
            reason = (
                f"Dropship signals: {pct}% of sample titles contain "
                f"auctioneer-template language"
            )
            if top_phrase:
                reason += f" (e.g. '{top_phrase}')"
            reason += " — likely AI-generated listings, low-quality inventory"
            return 'avoid', reason
    if items > 1500:
        return 'avoid', f"{items:,} lots — too expensive at current credit rates"
    if closes_dt:
        secs = (closes_dt - datetime.now()).total_seconds()
        if 0 < secs < 7200:
            return 'avoid', f"Closes in {int(secs / 60)} min — comp run wouldn't finish in time"
    if last_run_dt:
        age = (datetime.now() - last_run_dt).total_seconds()
        if age < 86400:
            return 'caution', f"Already analyzed {int(age / 3600)}h ago — re-running may be redundant"
    if not (100 <= items <= 500):
        if items < 100:
            return 'caution', f"Only {items} lots — small sample, finds may be sparse"
        # 500 < items <= 1500
        return 'caution', f"{items:,} lots — moderate cost, worth analyzing if categories look good"
    # In the 100-500 sweet spot
    for kw in _PREFER_KEYWORDS:
        if kw in nm:
            return 'good', f"Sweet-spot lot count + name suggests '{kw}' (good variety)"
    return 'good', "Sweet-spot lot count (100-500)"


def _render_sidebar_refresh_button():
    """Render the primary Refresh / Discover button at the top of the sidebar.

    Behavior is context-aware:
      - In an analysis view → 🔄 Refresh auction (re-fetches just the
        currently loaded auction's lots, preserves cached audit/comps).
      - With cached discovery present → 🔄 Refresh (re-runs full discovery).
      - Empty state → 🔍 Discover (kicks off the first discovery).
    """
    if discover_running:
        refresh_label = "⏳ Discovering…"
    elif fetch_lots_running:
        refresh_label = "⏳ Refreshing…"
    elif _in_analysis_view:
        refresh_label = "🔄 Refresh auction"
    elif _restored_at:
        refresh_label = "🔄 Refresh"
    else:
        refresh_label = "🔍 Discover"

    refresh_help = (
        "Re-fetch lots for the auction you're currently viewing. "
        "Pulls fresh bids; cached audit + comp results are preserved."
        if _in_analysis_view
        else "Re-fetch the open-auction list with the current Sourcing "
             "settings. Successful runs are cached for 24h and "
             "auto-restored on reload."
    )

    if st.button(
        refresh_label,
        type="primary",
        width='stretch',
        disabled=any_running,
        key="discover_btn",
        help=refresh_help,
    ):
        if _in_analysis_view:
            # Single-auction refresh path. Pull the auction_id off the
            # loaded analysis, queue a fetch_lots run for just that ID,
            # and clear the analysis-view state so the auto-load step at
            # the top of the dispatch picks the freshly fetched data back
            # up (with cached audit + comps merged on top via
            # _load_auction_for_analysis).
            aid = _extract_auction_id(st.session_state.selected_leads)
            if aid is not None:
                st.session_state._selected_auction_ids = [aid]
                st.session_state.current_auction = None
                st.session_state.selected_leads = pd.DataFrame()
                st.session_state.fetch_lots_running = True
                st.rerun()
            # If we somehow can't extract the auction_id, fall through
            # to a full discovery as a safe default.
        st.session_state._sourcing_cfg = {
            "zip": user_zip,
            "radius": radius,
            "include_nationwide": include_nationwide,
            "closing_days": closing_days,
            "category_filter": category_filter,
        }
        st.session_state.discover_running = True
        st.rerun()


def _render_sidebar_auction_list():
    """Render the Discover-phase auction browser (full-width main area).

    Historically this lived in st.sidebar; the phase-view refactor
    moved it into the main content area, shown only when no auction
    is loaded. The name is kept so grep/history still land here.
    """
    candidates = st.session_state.get('auction_candidates') or []
    cat_samples = st.session_state.get('category_samples', {}) or {}

    with st.container():
        # Row 2 of the collapsible filters drawer (row 1 = sourcing,
        # rendered at module level into the same expander): refresh
        # action + sort + hide-filters.
        with _discover_filters_exp:
            _fx1, _fx2, _fx3, _fx4 = st.columns([1.0, 1.4, 1.0, 1.4])
            with _fx1:
                _render_sidebar_refresh_button()
            with _fx2:
                sb_sort = st.selectbox(
                    "Sort by",
                    options=[
                        "BOLO hits",
                        "Best Match",
                        "Highest PC %",
                        "Lowest credit cost",
                        "Most items",
                        "Closing soonest",
                        "Fewest items",
                        "Closing latest",
                    ],
                    key="sidebar_picker_sort",
                    label_visibility="collapsed",
                )
            with _fx3:
                sb_hide_avoid = st.checkbox(
                    "Hide 🔴 avoid",
                    value=True,
                    key="sidebar_hide_avoid",
                    help="Filter out auctions flagged as low-priority: "
                         "liquidation/pallet naming, dropship template "
                         "language in sample titles, lot count > 1500, "
                         "or closing in <2 hours.",
                )
            with _fx4:
                sb_hide_local_pickup = st.checkbox(
                    "Hide 📦 local pickup",
                    value=False,
                    key="sidebar_hide_local_pickup",
                    help="Filter out every Local Pickup auction, even "
                         "ones within your sourcing radius.",
                )
        # ---- Resume an INTERRUPTED scan (died during match/load) ----
        # A scan whose session died before the match finished saved no
        # scan-view snapshot, but its fetched frame WAS persisted right
        # after fetch. Offer to resume from that — re-runs the match on
        # the already-fetched lots (cache-fast), no re-fetch.
        _pending_fetch = _load_fetched_frame()
        if (_pending_fetch is not None
                and not st.session_state.get('fetch_lots_running')):
            _pf_mode = _pending_fetch.get('mode') or {}
            _pf_kind = (
                'BOLO' if _pf_mode.get('bolo')
                else f"keyword '{_pf_mode.get('keyword')}'"
                if _pf_mode.get('keyword')
                else 'basket'
            )
            try:
                _pf_age = int(
                    (datetime.now()
                     - datetime.fromisoformat(_pending_fetch['saved_at'])
                     ).total_seconds() // 60
                )
                _pf_age_s = (f"{_pf_age}m ago" if _pf_age < 120
                             else f"{_pf_age // 60}h ago")
            except Exception:
                _pf_age_s = "earlier"
            if st.button(
                f"⏮ Resume interrupted {_pf_kind} scan — re-match "
                f"{len(_pending_fetch['df']):,} already-fetched lots "
                f"(no re-fetch · {_pf_age_s})",
                key="resume_interrupted_fetch_btn",
                type="primary",
                width='stretch',
                help="Your last scan's session ended before matching "
                     "finished. The fetched lots were saved — this "
                     "re-runs the (cache-fast) match on them without "
                     "re-fetching from HiBid.",
            ):
                st.session_state.phase1_leads = _pending_fetch['df']
                # Restore the pending-mode flags so the post-fetch
                # handler runs the right match.
                if _pf_mode.get('bolo'):
                    st.session_state._bolo_scan_all_pending = True
                if _pf_mode.get('keyword'):
                    st.session_state._keyword_scan_pending = _pf_mode['keyword']
                if _pf_mode.get('multi'):
                    st.session_state._multi_select_pending = True
                if _pf_mode.get('selected_ids'):
                    st.session_state._selected_auction_ids = _pf_mode['selected_ids']
                st.session_state.current_auction = None
                st.session_state.selected_leads = pd.DataFrame()
                st.rerun()

        # ---- Restore last scan (disconnect insurance) ----
        # A completed BOLO/keyword/basket scan is persisted to disk
        # the moment it loads; if the browser disconnected and the
        # session evaporated, this brings it back in one click —
        # no refetch, no rematch.
        _last_scan = _load_last_scan_view()
        if _last_scan is not None:
            try:
                _ls_saved = datetime.fromisoformat(_last_scan['saved_at'])
                _ls_age_min = int(
                    (datetime.now() - _ls_saved).total_seconds() // 60
                )
                _ls_age = (
                    f"{_ls_age_min}m ago" if _ls_age_min < 120
                    else f"{_ls_age_min // 60}h ago"
                )
            except Exception:
                _ls_age = "earlier"
            if st.button(
                f"⏮ Restore last scan: {_last_scan['label']}  "
                f"({len(_last_scan['df']):,} lots · saved {_ls_age})",
                key="restore_last_scan_btn",
                width='stretch',
                help="Reloads the matched lots from the last completed "
                     "scan without refetching or rematching. The "
                     "pipeline gates (lot preview, audit, comps "
                     "credit confirmation) apply as usual.",
            ):
                st.session_state.selected_leads = (
                    _last_scan['df'].reset_index(drop=True)
                )
                st.session_state.current_auction = _last_scan['label']
                st.session_state.audit_results = {}
                for _k in ('_comps_has_more', '_comps_auction_str_map',
                           '_comps_stats', '_comps_credit_confirmed',
                           '_audit_scope', '_audit_scope_total_lots',
                           '_comps_error_count',
                           '_auto_pipeline_attempts', '_audit_confirmed'):
                    st.session_state.pop(_k, None)
                st.session_state._comps_free_only_mode = False
                st.session_state.audit_running = False
                st.session_state.comps_running = False
                st.rerun()

        # Full-width search bar. One box, four behaviors: filter the
        # auction list by name, search HIBID ITSELF for auctions by
        # name/contents (works before Discover has ever run), surface
        # the deep lot-scan button, or paste an auction ID/URL for a
        # one-click load. (Sort + hide-filters live in the ⚙️ drawer.)
        if st.session_state.pop('_clear_picker_search', False):
            # Set BEFORE the widget instantiates this run — clearing
            # after a HiBid search so the name-filter doesn't hide
            # freshly found auctions whose lots (not names) matched.
            st.session_state['sidebar_picker_search'] = ''
        _sb_search_raw = st.text_input(
            "🔎 Filter",
            key="sidebar_picker_search",
            placeholder="Search auction names… deep-scan lots… or paste "
                        "an auction ID/URL",
            label_visibility="collapsed",
        ).strip()
        sb_search = _sb_search_raw.lower()
        # Remote pickup-only auctions (nationwide-search results that
        # are actually pickup-only in Lubbock/Boise/etc.) are ALWAYS
        # hidden — was a checkbox, but there was never a reason to
        # uncheck it. The unreachable-pickup guard downstream covers
        # the per-lot version of the same trap.
        sb_hide_remote_pickup = True

        # ---- Open-by-ID via the search box ----
        # A bare auction ID ("735012") or any HiBid URL in the search
        # field surfaces a one-click loader. Fetches every lot for
        # that auction regardless of the search radius — useful for
        # out-of-area auctions someone linked directly. Button-gated
        # (not automatic) so partial numbers typed mid-search don't
        # trigger spurious fetches.
        _parsed_aid = _parse_hibid_auction_id(_sb_search_raw or "")
        if _parsed_aid is not None:
            if st.button(
                f"📥 Load auction #{_parsed_aid} from HiBid",
                key="sb_load_by_id_btn",
                type="primary",
                disabled=bool(st.session_state.get('fetch_lots_running')),
                help="Fetches every lot for this auction regardless of "
                     "your search radius.",
            ):
                # Inject a synthetic candidate so the fetch work block
                # can resolve `_selected_auction_ids` against
                # `auction_candidates`. Downstream code pulls the real
                # name/source from the lots themselves.
                existing = st.session_state.get('auction_candidates') or []
                if not any(
                    c.get('auction_id') == _parsed_aid for c in existing
                ):
                    existing.append({
                        'auction_id': _parsed_aid,
                        'name': f'(loading auction #{_parsed_aid}…)',
                        'source': 'Ship',
                        'date_end': '',
                        'date_info': '',
                        'lot_count': 0,
                        'city': '',
                        'state': '',
                        'auctioneer': '',
                    })
                    st.session_state.auction_candidates = existing
                st.session_state._selected_auction_ids = [_parsed_aid]
                st.session_state.current_auction = None
                st.session_state.selected_leads = pd.DataFrame()
                st.session_state.phase1_leads = pd.DataFrame()
                st.session_state.fetch_lots_running = True
                st.rerun()

        # ---- Search HiBid's auction catalog directly ----
        # Runs BEFORE/without Discover: one GraphQL call with the term
        # as serverside searchText (names + contents). Results merge
        # into the grid as normal candidates — click to analyze.
        if _sb_search_raw and _parsed_aid is None:
            if st.button(
                f"🌐 Search HiBid for auctions matching "
                f"“{_sb_search_raw}”",
                key="hibid_auction_search_btn",
                disabled=bool(st.session_state.get('fetch_lots_running')),
                help="Searches HiBid's live auction catalog (names and "
                     "contents, nationwide) and adds matches to the "
                     "grid below. Free — one API call, no credits.",
            ):
                from scraper.pass1 import Phase1Scraper as _P1S, _HTTPX_VERIFY as _HV
                import httpx as _hx

                async def _hibid_auction_search():
                    _s = _P1S()
                    async with _hx.AsyncClient(
                        verify=_HV, headers=_s.headers, timeout=30,
                    ) as _c:
                        return await _s.fetch_auctions(
                            _c, "", 0, search_text=_sb_search_raw,
                        )
                with st.spinner(
                    f"Searching HiBid for '{_sb_search_raw}'…"
                ):
                    try:
                        _found = run_async(_hibid_auction_search())
                    except Exception as _hse:
                        st.error(f"HiBid search failed: {_hse}")
                        _found = []
                _existing = st.session_state.get('auction_candidates') or []
                _known = {c.get('auction_id') for c in _existing}
                _added = 0
                for _fa in _found:
                    if _fa.get('auction_id') in _known:
                        continue
                    _fa['source'] = 'Ship'  # locality unknown; per-lot
                    #                         flags resolve it at fetch
                    _existing.append(_fa)
                    _added += 1
                st.session_state.auction_candidates = _existing
                tlog("SEARCH",
                     f"HiBid auction search '{_sb_search_raw}':",
                     f"{len(_found)} results, {_added} new")
                st.toast(
                    f"🌐 Found {len(_found)} auction(s) — "
                    f"{_added} new added to the grid",
                    icon="✅",
                )
                st.session_state['_clear_picker_search'] = True
                st.rerun()

        if not candidates:
            if discover_running:
                # Show a friendly loading state instead of "click discover" —
                # we're already discovering. A spinning emoji + text is a
                # cheap way to communicate "the algorithm is working" in
                # the sidebar (where the user is most likely looking)
                # without needing JS animations.
                st.markdown(
                    """
                    <div style="
                        padding: 14px 12px;
                        background: rgba(59,130,246,0.1);
                        border-left: 3px solid #3b82f6;
                        border-radius: 4px;
                        font-size: 13px;
                    ">
                        <div style="font-weight: 600; margin-bottom: 4px;">
                            ⏳ Loading auctions…
                        </div>
                        <div style="font-size: 12px; opacity: 0.85;">
                            Fetching the open-auction list from HiBid.
                            The sidebar will populate in a few seconds.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                # Streamlit's built-in spinner gives the user a small
                # animated indicator on top of the text card.
                with st.spinner("Querying HiBid…"):
                    # Empty placeholder — the spinner shows for as long as
                    # discover_running is True. Streamlit auto-cleans it
                    # up when the script finishes / reruns.
                    st.empty()
            else:
                st.caption(
                    "Nothing discovered yet — type an auction name above to "
                    "search HiBid directly, or open **⚙️ Sourcing & "
                    "filters** and click **🔍 Discover** for the full "
                    "radius + nationwide sweep."
                )
            return

        # The "Only BOLO matches" sidebar toggle was removed — at the
        # discovery stage the BOLO hints are based on auction NAME +
        # 8-lot title sample, so they hide auctions that may actually
        # have BOLO content in lots we haven't sampled yet. Better UX:
        # show every discovered auction, let the user click "🔍 Search
        # for BOLOs" to do the full scan, and surface the hit counts
        # in the auction list afterwards.

        # Scan-all-for-BOLO button placeholder — the actual button is
        # rendered below, AFTER all visibility filters are applied to
        # `rows`, so its label + scope reflect exactly what the user
        # sees in the picker (not the unfiltered candidate list).
        _bolo_scan_button_slot = st.empty()

        # Build display rows. Reuse the same parsing/closing-time logic
        # the in-page picker uses, kept lightweight.
        rows = []
        for c in candidates:
            aid = c['auction_id']
            raw_sample = cat_samples.get(aid)
            if isinstance(raw_sample, dict):
                sample_payload = raw_sample
            elif isinstance(raw_sample, list):
                sample_payload = {"categories": raw_sample, "cat_counts": {}, "titles": []}
            else:
                sample_payload = None
            summary = (
                Phase1Scraper.generate_auction_summary(c, sample_payload)
                if sample_payload is not None else ""
            )
            closing_raw = c.get('date_end', '')
            date_info = c.get('date_info', '') or ''
            closing_fmt = closing_raw
            closes_dt = None
            try:
                if closing_raw:
                    day_dt = datetime.fromisoformat(closing_raw)
                    date_part = day_dt.strftime("%b %d")
                    time_match = re.findall(
                        r'(\d{1,2})(?::(\d{2}))?\s*([ap])\.?m\.?',
                        date_info, flags=re.IGNORECASE,
                    )
                    if time_match:
                        h, m, mer = time_match[-1]
                        hour24 = int(h) % 12 + (12 if mer.lower() == 'p' else 0)
                        minute = int(m) if m else 0
                        closes_dt = day_dt.replace(hour=hour24, minute=minute)
                        time_str = f"{int(h)}:{minute:02d}{mer.upper()}M"
                        closing_fmt = f"{date_part} @ {time_str}"
                    else:
                        closes_dt = day_dt.replace(hour=23, minute=59)
                        closing_fmt = date_part
            except (ValueError, TypeError):
                pass
            # Cost + signal for triage. Both are pure functions of the
            # data we already have — zero additional HTTP.
            pc_pct, est_credits = _estimate_auction_cost(
                int(c.get('lot_count') or 0), sample_payload,
            )
            # Pickup-only override: HiBid auctions found via the
            # nationwide search get source='Ship' by default, but the
            # auction house can still be physically pickup-only at a
            # remote location (Lubbock TX, Boise ID, etc.). Detect
            # explicit pickup-only / no-shipping language in the
            # auction name + summary and re-classify so the
            # "Hide local pickup" filter catches them.
            raw_source = c.get('source') or ''
            text_for_pickup_check = (
                f"{c.get('name', '')} {summary or ''}"
            ).lower()
            pickup_phrases = (
                'pickup only',
                'pick up only',
                'pick-up only',
                'local pickup only',
                'no shipping',
                'shipping not available',
                'shipping unavailable',
                'buyer must pick up',
                'buyer to pick up',
                'must be picked up',
                'in-person pickup',
            )
            if any(p in text_for_pickup_check for p in pickup_phrases):
                effective_source = 'Local Pickup'
            else:
                effective_source = raw_source
            # Extract sample titles for the dropship-detector. Empty
            # list when no sample payload exists — _auction_signal
            # gracefully short-circuits in that case.
            sample_titles_for_signal = []
            if isinstance(sample_payload, dict):
                raw_titles = sample_payload.get('titles') or []
                sample_titles_for_signal = [
                    str(t) for t in raw_titles if t
                ][:50]
            rows.append({
                'auction_id': aid,
                'name': c.get('name') or '(unnamed)',
                'items': int(c.get('lot_count') or 0),
                'closes_fmt': closing_fmt or '—',
                'closes_dt': closes_dt,
                'summary': summary or '',
                'source': effective_source,
                'raw_source': raw_source,  # used by REMOTE-pickup filter below
                'city': c.get('city') or '',
                'state': c.get('state') or '',
                'pc_pct': pc_pct,
                'est_credits': est_credits,
                'sample_titles': sample_titles_for_signal,
            })

        # Filter + sort.
        # Keep a pre-name-filter copy for the deep keyword scan: the
        # search box narrows the AUCTION LIST by name/summary, but a
        # deep lot-keyword scan for the same term should sweep every
        # visible auction — an auction doesn't need "ink" in its NAME
        # to contain ink lots.
        rows_for_deep_scan = list(rows)
        if sb_search:
            rows = [
                r for r in rows
                if sb_search in r['name'].lower() or sb_search in r['summary'].lower()
            ]

        # Determine the currently-loaded auction (for highlighting)
        active_aid = None
        sel_leads = st.session_state.get('selected_leads')
        if isinstance(sel_leads, pd.DataFrame) and not sel_leads.empty:
            active_aid = _extract_auction_id(sel_leads)

        sb_fetch_lots_running = st.session_state.get('fetch_lots_running', False)

        # Build a {auction_id: cached_at_datetime} map for the "last
        # analyzed" labels + a {auction_id: (a_count, b_count,
        # total_count)} map so cached auctions can show their grade
        # summary (replaces the old green_pct / green_count fields).
        # Single list_all() call, then we look up by id in the loop.
        # Pass a generous TTL so stale entries still surface a timestamp
        # — the user wants to see "12 days ago" on a stale auction, not
        # nothing.
        #
        # PERFORMANCE: list_all() unpickles every cached auction file
        # (~50-200KB each). With 30+ cached auctions that's ~5MB of
        # pickle parsing on every render. The shared
        # `_get_cached_auction_list_memo()` helper memoizes per-session
        # keyed on the cache-dir's directory mtime — rebuild only when
        # a new auction is cached or one is purged. The 💾 Memory
        # popover hits the same memo, so we only do that pickle parse
        # once per cache-dir change.
        _entries = _get_cached_auction_list_memo()

        last_run_map = {}
        grade_map = {}
        bolo_cached_map = {}
        bids_per_lot_map = {}
        dropship_pct_map = {}
        try:
            for entry in _entries:
                ts_raw = entry.get('cached_at') or ''
                try:
                    last_run_map[entry.get('auction_id')] = (
                        datetime.fromisoformat(ts_raw) if ts_raw else None
                    )
                except (ValueError, TypeError):
                    pass
                ac = entry.get('a_count')
                bc_grade = entry.get('b_count')
                tc = entry.get('total_count')
                if ac is not None or bc_grade is not None:
                    grade_map[entry.get('auction_id')] = (
                        ac or 0, bc_grade or 0, tc or 0,
                    )
                bc = entry.get('bolo_count')
                if bc is not None:
                    bolo_cached_map[entry.get('auction_id')] = bc
                bpl = entry.get('bids_per_lot')
                if bpl is not None:
                    bids_per_lot_map[entry.get('auction_id')] = bpl
                dpc = entry.get('dropship_pct')
                if dpc is not None:
                    dropship_pct_map[entry.get('auction_id')] = dpc
        except Exception:
            pass  # Cache read failure shouldn't break the picker

        # For each auction, compute a bolo_hint count:
        #   - cached auctions: the persisted full-scan count (authoritative)
        #   - uncached auctions: count of brand hits in name + sample
        #     lot titles. Less complete than a full lot scan, but free
        #     and good enough to surface "this auction probably has
        #     Lululemon / Patagonia / etc. inside" as a heads-up.
        # The flag distinguishes the two so the UI can label hints
        # differently from full scans (badge color/wording).
        #
        # PERFORMANCE: this used to recompute every render — at 200
        # auctions × 50 sample titles × ~200 BOLO patterns that's ~2M
        # regex calls per render, making every search keystroke /
        # sort change feel laggy. Now we memoize per
        # (auction_id, samples_fingerprint, matcher_mtime). The cache
        # invalidates automatically when the BOLO files change (mtime)
        # or when fresh samples land for an auction (fingerprint).
        bolo_hint_map = {}  # aid -> (count, is_full_scan)

        # FAST PATH: only populate hints from the DISK CACHE (cheap dict
        # lookup, zero regex work). The previous "compute hints from
        # auction name + sample titles" branch was eagerly loading the
        # BOLO matcher (1-2s) and running ~2M regex calls (~1s) on every
        # page render — making the auction list appear ~2-3s late after
        # cache rehydrate. The hint-badge sample-based path was also the
        # least-reliable signal (8 sampled titles per auction); the user
        # gets ground-truth match counts after clicking "🔍 Search for
        # BOLOs" anyway. Cached full-scan counts (from prior scans) are
        # still surfaced — those don't require the matcher.
        for c in candidates:
            aid_local = c['auction_id']
            cached_count = bolo_cached_map.get(aid_local)
            if cached_count is not None and int(cached_count) > 0:
                bolo_hint_map[aid_local] = (int(cached_count), True)

        def _format_last_run(when):
            """Compact relative timestamp for the per-row label."""
            if when is None:
                return ""
            delta = datetime.now() - when
            if delta.total_seconds() < 60:
                return "just now"
            if delta.total_seconds() < 3600:
                return f"{int(delta.total_seconds() / 60)}m ago"
            if delta.total_seconds() < 86400:
                return f"{int(delta.total_seconds() / 3600)}h ago"
            return f"{delta.days}d ago"

        # Compute the triage signal for every row using auction-name
        # heuristics + lot count + closing time + last-run age + a
        # dropship-detector that runs against the sample lot titles.
        for row in rows:
            rank, reason = _auction_signal(
                row['name'], row['items'], row['closes_dt'],
                last_run_map.get(row['auction_id']),
                sample_titles=row.get('sample_titles'),
                state=row.get('state'),
                city=row.get('city'),
            )
            row['signal_rank'] = rank
            row['signal_reason'] = reason

        # Apply the "hide avoid" filter after computing signals.
        if sb_hide_avoid:
            rows = [r for r in rows if r['signal_rank'] != 'avoid']
        # Hide local-pickup-only auctions when the toggle is on. Compare
        # against the source string ('Local Pickup' / 'Ship') populated
        # in pass1.py + the post-discovery override.
        if sb_hide_local_pickup:
            rows = [
                r for r in rows
                if (r.get('source') or '').strip().lower() != 'local pickup'
            ]
        # Hide REMOTE pickup-only — auctions whose raw_source was 'Ship'
        # (i.e., found via nationwide search, outside the sourcing
        # radius) but our pickup-only language detector re-classified
        # them to 'Local Pickup'. These are the "drive 500 mi" trap.
        # On by default; "Hide ALL local pickup" supersedes it.
        if sb_hide_remote_pickup and not sb_hide_local_pickup:
            rows = [
                r for r in rows
                if not (
                    (r.get('raw_source') or '').strip().lower() == 'ship'
                    and (r.get('source') or '').strip().lower() == 'local pickup'
                )
            ]
        # Stash the BOLO match count on each row so the sort below
        # (and any future label decoration) can read it directly.
        # is_full = True means a cached full-scan count; False means
        # a sample-titles hint. We sort on the raw count regardless,
        # but tie-break by full-scan first since those are higher
        # signal.
        for r in rows:
            entry = bolo_hint_map.get(r['auction_id'])
            if entry:
                r['bolo_count'] = int(entry[0])
                r['bolo_full'] = bool(entry[1])
            else:
                r['bolo_count'] = 0
                r['bolo_full'] = False

        # 🎯 Render the Scan-all-for-BOLO button NOW that all visibility
        # filters have been applied to `rows`. The button's scope is
        # exactly the visible auction list — toggling "Hide local
        # pickup" / "Hide remote pickup" / "Only BOLO matches" /
        # search shrinks the scan to match. This was the #1 footgun of
        # the previous build: the button ignored filters and hammered
        # HiBid for auctions the user had explicitly hidden.
        # NOTE: previously gated on `_BOLO_MATCHER.loaded` — that's now
        # a non-build-triggering check that returns False until the
        # matcher is first touched. The button itself doesn't need a
        # loaded matcher to RENDER; the matcher builds when the button
        # is clicked + the BOLO scan path enters its match phase.
        if len(rows) >= 2:
            _filtered_lots = sum(int(r.get('items') or 0) for r in rows)
            _filtered_mega = sum(
                1 for r in rows if int(r.get('items') or 0) > 1000
            )
            _filtered_batches = max(1, (len(rows) + 19) // 20)
            _filtered_eta_sec = _filtered_batches * 4 + _filtered_mega * 3
            _filtered_eta_str = (
                f"~{_filtered_eta_sec}s"
                if _filtered_eta_sec < 60
                else f"~{_filtered_eta_sec // 60}m {_filtered_eta_sec % 60:02d}s"
            )
            _scan_disabled = (
                discover_running
                or fetch_lots_running
                or st.session_state.get('audit_running', False)
                or st.session_state.get('comps_running', False)
            )
            _filtered_count = len(rows)
            _hidden_count = len(candidates) - _filtered_count
            _hidden_suffix = (
                f" · {_hidden_count} hidden by filters"
                if _hidden_count > 0 else ""
            )
            with _bolo_scan_button_slot.container():
                if st.button(
                    f"🔍 Search for BOLOs across {_filtered_count} auctions "
                    f"({_filtered_lots:,} lots, {_filtered_eta_str})",
                    key="bolo_scan_all_btn",
                    disabled=_scan_disabled,
                    type="primary",
                    width='stretch',
                    help=(
                        f"Searches the {_filtered_count} auctions currently "
                        f"visible in your sidebar for BOLO matches — respects "
                        f"every filter (search, hide local pickup, hide "
                        f"remote pickup, hide avoid).{_hidden_suffix}. "
                        f"Estimated scope: {_filtered_lots:,} total lots, "
                        f"~{_filtered_batches} concurrent fetch batches, "
                        f"ETA {_filtered_eta_str}. Free (HiBid GraphQL — "
                        f"no ScrapingBee credits). BOLO regex match runs "
                        f"after fetch, then audit + comps run on the "
                        f"matched subset (credits gate fires before any "
                        f"ScrapingBee spend)."
                    ),
                ):
                    tlog("CLICK",
                         f"🔍 Search for BOLOs clicked",
                         f"· {len(rows)} auctions in scope",
                         f"· {_filtered_lots:,} lots,",
                         f"ETA {_filtered_eta_str}")
                    # Use the FILTERED list of auction IDs — not the raw
                    # `candidates` — so hidden auctions stay hidden.
                    st.session_state._selected_auction_ids = [
                        r['auction_id'] for r in rows
                    ]
                    st.session_state.current_auction = None
                    st.session_state.selected_leads = pd.DataFrame()
                    st.session_state.phase1_leads = pd.DataFrame()
                    st.session_state.audit_results = {}
                    st.session_state._bolo_scan_all_pending = True
                    st.session_state.fetch_lots_running = True
                    # Stash a "scan just kicked off" marker so the next
                    # render (post-rerun) can pop a prominent banner +
                    # toast confirming the click landed. The marker
                    # carries scope info so the banner shows scope
                    # before phase status panels start rendering.
                    st.session_state._scan_just_started = {
                        "started_at": datetime.now().isoformat(),
                        "auction_count": len(rows),
                        "lot_count": _filtered_lots,
                        "eta": _filtered_eta_str,
                    }
                    # st.toast renders on the NEXT script run (post-
                    # rerun), so this is the right place to fire it —
                    # the user sees the confirmation pop in the top-
                    # right immediately on rerun, before any heavy
                    # work starts.
                    st.toast(
                        f"🎯 BOLO scan kicked off — "
                        f"{len(rows)} auctions, {_filtered_lots:,} lots, "
                        f"ETA {_filtered_eta_str}",
                        icon="🚀",
                    )
                    st.rerun()

        # Sort. "Best fit first" uses the signal as the primary key
        # (good > caution > avoid), then est_credits ascending, then
        # closing time. The other sorts are unchanged.
        if sb_sort.startswith("🎯 Best"):
            _rank_order = {'good': 0, 'caution': 1, 'avoid': 2}
            rows.sort(key=lambda r: (
                _rank_order.get(r['signal_rank'], 3),
                r['est_credits'],
                r['closes_dt'] or datetime.max,
            ))
        elif sb_sort.startswith("🎯 Most BOLO"):
            # Most BOLO hits first. Full-scan counts beat sample-hint
            # counts at the same numeric value (full=False sorts after
            # full=True via the tiebreak), and ties break by closing
            # soonest so a 5-hit auction ending tonight floats above
            # a 5-hit auction ending Friday.
            rows.sort(key=lambda r: (
                -int(r.get('bolo_count') or 0),
                0 if r.get('bolo_full') else 1,
                r['closes_dt'] or datetime.max,
            ))
        elif sb_sort.startswith("💸 Highest PC"):
            # Higher PC % = cheaper to comp. Tiebreak by closing soonest
            # so finds with same coverage but ending sooner float up.
            rows.sort(key=lambda r: (
                -(r.get('pc_pct') or 0),
                r['closes_dt'] or datetime.max,
            ))
        elif sb_sort.startswith("💸 Lowest credit"):
            # Cheapest absolute spend — useful when budget is tight and
            # you want to rip through several small auctions cheaply.
            rows.sort(key=lambda r: (
                r['est_credits'],
                r['closes_dt'] or datetime.max,
            ))
        elif sb_sort.startswith("🔢 Most"):
            rows.sort(key=lambda r: (-r['items'], r['closes_dt'] or datetime.max))
        elif sb_sort.startswith("🔢 Fewest"):
            rows.sort(key=lambda r: (r['items'], r['closes_dt'] or datetime.max))
        elif sb_sort.startswith("⏰ Closing soonest"):
            rows.sort(key=lambda r: r['closes_dt'] or datetime.max)
        else:
            rows.sort(key=lambda r: r['closes_dt'] or datetime.min, reverse=True)

        # --- Deep keyword scan (driven by the ONE search box) ---
        # The single filter box drives both behaviors: typing narrows
        # the auction list by NAME live, and this button deep-scans
        # every visible auction's LOTS for the same term. Rendered
        # BEFORE the empty-list early-return on purpose — a term like
        # "ink" usually matches zero auction NAMES while matching
        # plenty of lots, and the deep scan is the whole point then.
        # Scope: rows_for_deep_scan = every sidebar filter EXCEPT the
        # name filter.
        _deep_term = (_sb_search_raw or '').strip()
        if _deep_term and _parsed_aid is None and len(rows_for_deep_scan) >= 1:
            _deep_rows = rows_for_deep_scan
            _deep_lots = sum(int(r.get('items') or 0) for r in _deep_rows)
            _deep_batches = max(1, (len(_deep_rows) + 19) // 20)
            _deep_eta_sec = _deep_batches * 4
            _deep_eta = (
                f"~{_deep_eta_sec}s" if _deep_eta_sec < 60
                else f"~{_deep_eta_sec // 60}m {_deep_eta_sec % 60:02d}s"
            )
            _deep_disabled = (
                discover_running
                or st.session_state.get('fetch_lots_running', False)
                or st.session_state.get('audit_running', False)
                or st.session_state.get('comps_running', False)
            )
            if st.button(
                f"🔍 Deep-scan {len(_deep_rows)} auctions' lots for "
                f"“{_deep_term}” ({_deep_lots:,} lots, {_deep_eta})",
                key="keyword_scan_btn",
                disabled=_deep_disabled,
                width='stretch',
                help=(
                    "Searches lot title + description across every "
                    "visible auction — not just auction names. Free "
                    "(HiBid server-side search, no ScrapingBee "
                    "credits). Word-boundary stem match: 'ink' hits "
                    "ink/inks/inkjet, not pink/drink; multi-word "
                    "terms match in any order."
                ),
            ):
                tlog("CLICK",
                     f"🔍 Keyword deep-scan clicked: '{_deep_term}'",
                     f"· {len(_deep_rows)} auctions in scope")
                st.session_state._selected_auction_ids = [
                    r['auction_id'] for r in _deep_rows
                ]
                st.session_state.current_auction = None
                st.session_state.selected_leads = pd.DataFrame()
                st.session_state.phase1_leads = pd.DataFrame()
                st.session_state.audit_results = {}
                st.session_state._keyword_scan_pending = _deep_term
                st.session_state.fetch_lots_running = True
                st.session_state._scan_just_started = {
                    "started_at": datetime.now().isoformat(),
                    "auction_count": len(_deep_rows),
                    "lot_count": _deep_lots,
                    "eta": _deep_eta,
                    "kind": "keyword",
                    "term": _deep_term,
                }
                st.toast(
                    f"🔍 Deep-scanning {len(_deep_rows)} auctions "
                    f"for '{_deep_term}'…",
                    icon="🚀",
                )
                st.rerun()

        if not rows:
            st.caption(
                "_No auction names match — use the deep-scan button "
                "above to search inside their lots instead._"
                if _deep_term else "_No auctions match the filter._"
            )
            return

        # Triage icon for the row label. Prefixed before the auction
        # name so "Best fit" sorting visually surfaces 🟢 rows.
        _signal_icon = {'good': '🟢', 'caution': '🟡', 'avoid': '🔴'}

        def _format_credits(c: int) -> str:
            """Compact credit count for the per-row label."""
            if c == 0:
                return "free (PC)"
            if c < 1000:
                return f"~{c} cr"
            return f"~{c / 1000:.1f}k cr"

        # ================================================================
        # AUCTION GRID — one row per auction, full-width, sortable by
        # any column header. Clicking a row loads that auction for
        # analysis (same dispatch as the old per-auction buttons).
        # Replaced the sidebar card stack in the phase-view refactor.
        # ================================================================
        _grid_records = []
        for row in rows:
            aid = row['auction_id']
            last_run_str = _format_last_run(last_run_map.get(aid))
            # Grade summary from the cached payload ("3A · 5B /42")
            grades_txt = ''
            grade_entry = grade_map.get(aid)
            if grade_entry is not None:
                ac, bc_grade, tc = grade_entry
                parts = []
                if ac:
                    parts.append(f"🟢{ac}A")
                if bc_grade:
                    parts.append(f"🟡{bc_grade}B")
                if parts:
                    grades_txt = " ".join(parts) + (
                        f" /{tc}" if tc else ""
                    )
            # BOLO count: full-scan counts are bare numbers, sample
            # hints get a ~ prefix.
            bolo_txt = ''
            bolo_entry = bolo_hint_map.get(aid)
            if bolo_entry:
                bcount, is_full = bolo_entry
                bolo_txt = f"🎯 {bcount}" if is_full else f"🎯 ~{bcount}?"
            # Bids/lot with the 🔥/🟡/🧊 competitiveness marker
            bpl = bids_per_lot_map.get(aid)
            if bpl is None:
                bpl_txt = ''
            elif bpl >= 8:
                bpl_txt = f"🔥 {bpl:.1f}"
            elif bpl >= 3:
                bpl_txt = f"🟡 {bpl:.1f}"
            else:
                bpl_txt = f"🧊 {bpl:.1f}"
            dpc = dropship_pct_map.get(aid)
            dropship_txt = (
                f"🚨 {dpc:.0f}%" if (dpc is not None and dpc >= 20) else ''
            )
            _src = (row.get('source') or '').strip().lower()
            _raw_src = (row.get('raw_source') or '').strip().lower()
            if _src == 'local pickup' and _raw_src == 'ship':
                ship_txt = "📦⚠️ remote"
            elif _src == 'local pickup':
                ship_txt = "📦 pickup"
            elif _src == 'ship':
                ship_txt = "🚚 ships"
            else:
                ship_txt = ""
            pc_pct_txt = (
                f"{int(round(row['pc_pct'] * 100))}%"
                if row.get('pc_pct') and row['pc_pct'] > 0 else ''
            )
            _grid_records.append({
                '_aid': aid,
                '🚦': _signal_icon.get(row['signal_rank'], '⚪'),
                'Auction': row['name'],
                'Lots': int(row['items']),
                'Closes': row['closes_fmt'],
                'Ship': ship_txt,
                'Credits': _format_credits(row['est_credits']),
                'PC%': pc_pct_txt,
                'BOLO': bolo_txt,
                'Bids/lot': bpl_txt,
                'Dropship': dropship_txt,
                'Analyzed': last_run_str or '',
                'Grades': grades_txt,
            })
        grid_df = pd.DataFrame(_grid_records)
        # Drop signal columns that are entirely blank for this list.
        # They're cache-derived (Analyzed / Grades / Bids-lot /
        # Dropship) or sample-derived (BOLO, PC%) and only populate
        # for auctions that have been analyzed or sampled — on a
        # fresh discovery they're all empty and just eat grid width.
        # They reappear automatically once any row has data.
        for _col in ('PC%', 'BOLO', 'Bids/lot', 'Dropship',
                     'Analyzed', 'Grades', 'Dropship%'):
            if _col in grid_df.columns:
                _vals = grid_df[_col].astype(str).str.strip()
                if (_vals == '').all() or (_vals == 'None').all():
                    grid_df = grid_df.drop(columns=[_col])
        st.caption(
            f"**{len(grid_df)}** auctions shown — check one or more "
            f"rows, then hit **Analyze**. Sort by clicking a column "
            f"header."
        )
        # Key includes a nonce so we can clear the selection after
        # handling it — without this, the same selection re-fires on
        # every rerun and traps the user in a load loop.
        _grid_key = f"auction_grid_{st.session_state.get('_grid_nonce', 0)}"
        _grid_event = st.dataframe(
            grid_df.drop(columns=['_aid']),
            key=_grid_key,
            width='stretch',
            height=min(600, 60 + 35 * len(grid_df)),
            hide_index=True,
            on_select='rerun',
            selection_mode='multi-row',
            column_config={
                'Auction': st.column_config.TextColumn(
                    'Auction', width='large',
                ),
            },
        )
        _sel_rows = []
        try:
            _sel_rows = list(_grid_event.selection.rows)
        except Exception:
            pass
        if _sel_rows:
            _sel_aids = [
                int(grid_df.iloc[i]['_aid']) for i in _sel_rows
            ]
            _sel_lots = int(sum(
                int(grid_df.iloc[i]['Lots'] or 0) for i in _sel_rows
            ))
            _btn_label = (
                f"🔬 Analyze this auction ({_sel_lots:,} lots)"
                if len(_sel_aids) == 1 else
                f"🔬 Analyze {len(_sel_aids)} auctions together "
                f"({_sel_lots:,} lots)"
            )
            if st.button(
                _btn_label,
                key="analyze_selected_btn",
                type="primary",
                width='stretch',
                disabled=sb_fetch_lots_running,
                help=(
                    "Fetches every lot from the checked auction(s) and "
                    "runs the full pipeline. Multiple auctions load as "
                    "one combined analysis view — audit + comps + "
                    "grading across all of them, with per-row auction "
                    "links in the results table."
                ),
            ):
                # Bump the nonce so the fresh grid widget starts with
                # no selection when the user comes back to Discover.
                st.session_state._grid_nonce = (
                    st.session_state.get('_grid_nonce', 0) + 1
                )
                # Clear the current analysis state so the auto-load
                # step in the dispatch picks up the newly-fetched lots.
                st.session_state._selected_auction_ids = _sel_aids
                st.session_state.current_auction = None
                st.session_state.selected_leads = pd.DataFrame()
                st.session_state.phase1_leads = pd.DataFrame()
                # >1 auction → the post-fetch handler combines them
                # into one synthetic "🧺 basket" analysis instead of
                # the single-auction loader (which would refuse a
                # multi-auction frame).
                st.session_state._multi_select_pending = (
                    len(_sel_aids) > 1
                )
                st.session_state.fetch_lots_running = True
                st.rerun()


# Render the Discover-phase auction browser (main area, full width).
# Skipped entirely in the Analyze phase and while a fetch is running —
# that's the whole point of the phase-view refactor: the list is a
# selection tool, not a permanent fixture.
if _discover_phase:
    _render_sidebar_auction_list()


# Subtle "showing cached results" caption under the header, only when
# we restored from disk (so the user knows the results aren't fresh).
if _restored_at and not discover_running:
    _age = datetime.now() - _restored_at
    if _age.total_seconds() < 3600:
        _age_str = f"{int(_age.total_seconds() / 60)} min ago"
    else:
        _age_str = f"{_age.total_seconds() / 3600:.1f}h ago"
    st.caption(
        f"♻️ Showing cached results from **{_age_str}** "
        f"({len(st.session_state.auction_candidates)} auctions). "
        f"Click 🔄 Refresh to re-fetch."
    )

# --- Surface any persisted discover/fetch status so errors don't vanish on rerun ---
for _key, _tb_key, _label in (
    ('_discover_status', '_last_discover_traceback', 'Discovery'),
    ('_fetch_status', '_last_fetch_traceback', 'Lot fetch'),
):
    _status = st.session_state.pop(_key, None)
    if not _status:
        continue
    if _status.get('error'):
        st.error(f"❌ {_label} failed: {_status['error']}")
        tb = st.session_state.get(_tb_key)
        if tb:
            with st.expander("🔍 Full traceback (share this if you ask for help)"):
                st.code(tb, language="python")
    elif _status.get('msg'):
        msg = _status['msg']
        if msg.startswith("⚠️"):
            st.warning(msg)
        else:
            st.success(msg)

    # Render full fetch diagnostics so the user can see exactly what HiBid
    # returned even after the rerun wipes the in-status widgets.
    _diag = _status.get('diag') if isinstance(_status, dict) else None
    if _diag:
        with st.expander(
            f"🔬 {_label} diagnostics "
            f"(raw: {_diag.get('raw_count', 0)} · "
            f"kept: {_diag.get('kept', 0)} · "
            f"closed: {_diag.get('filtered_status', 0)} · "
            f"category: {_diag.get('filtered_cat', 0)})",
            expanded=(_diag.get('kept', 0) == 0),
        ):
            st.write(
                f"- **Raw lots from HiBid:** {_diag.get('raw_count', 0)}\n"
                f"- **Kept after filtering:** {_diag.get('kept', 0)}\n"
                f"- **Dropped as CLOSED / Bidding Closed:** {_diag.get('filtered_status', 0)}\n"
                f"- **Dropped by category filter:** {_diag.get('filtered_cat', 0)}"
            )
            _sv = _diag.get('status_values') or {}
            if _sv:
                st.write(f"**lotState.status values seen:** `{_sv}`")
            _pa = _diag.get('per_auction') or []
            if _pa:
                st.caption(f"Per-auction breakdown ({len(_pa)}):")
                st.table(pd.DataFrame(_pa))
            _errs = _diag.get('errors') or []
            if _errs:
                st.warning(f"{len(_errs)} auction(s) errored during fetch:")
                st.table(pd.DataFrame(_errs))


# ================================================================
# WORK BLOCK: Discover Auctions (lives in main area so mobile users
# with collapsed sidebar can actually SEE progress / errors)
# ================================================================
if st.session_state.get('discover_running'):
    _keep_screen_awake()
    discover_error = None
    discover_result_msg = None
    with st.status("🔍 Discovering open auctions…", expanded=True) as status_box:
        try:
            cfg = st.session_state.get('_sourcing_cfg', {})
            scraper = Phase1Scraper(config_path="config.json")
            scraper.zip_code = cfg.get("zip", "")
            scraper.radius = cfg.get("radius", 20)
            scraper.include_nationwide = cfg.get("include_nationwide", True)
            scraper.closing_within_days = cfg.get("closing_days", 1)
            scraper.category_filter = cfg.get("category_filter", [])

            tlog("DISCOVER",
                 f"Phase 1a starting · zip={scraper.zip_code}",
                 f"radius={scraper.radius}mi",
                 f"closing<={scraper.closing_within_days}d",
                 f"nationwide={scraper.include_nationwide}",
                 f"category_filter={scraper.category_filter or 'none'}")
            _phase1a_t0 = _time.time()

            st.write(
                f"Querying HiBid near **{scraper.zip_code}** within "
                f"**{scraper.radius} mi**, closing within "
                f"**{scraper.closing_within_days} day(s)**"
                + (", including nationwide shippable." if scraper.include_nationwide else ".")
            )
            scan_progress = st.progress(0, text="Starting...")

            def discover_prog(current, total, label=""):
                if total == 0:
                    pct, text = 0.0, (label or "Done")
                elif current == 0 and total == 1:
                    pct, text = 0.0, (label or "Working...")
                else:
                    pct = current / total if total > 0 else 0
                    text = label or f"{current}/{total}"
                scan_progress.progress(min(pct, 1.0), text=text)

            candidates = run_async(
                scraper.fetch_auction_candidates(progress_callback=discover_prog)
            )
            scan_progress.empty()
            _phase1a_elapsed = _time.time() - _phase1a_t0
            _total_lots = sum(int(c.get('lot_count') or 0) for c in candidates)
            tlog("DISCOVER",
                 f"Phase 1a done in {_phase1a_elapsed:.1f}s",
                 f"· got {len(candidates)} auctions",
                 f"({_total_lots:,} total lots)")

            # Reset downstream state: a new candidate list invalidates prior picks + lots
            st.session_state.auction_candidates = candidates
            st.session_state.category_samples = {}
            st.session_state.phase1_leads = pd.DataFrame()
            st.session_state.audit_results = {}
            st.session_state.selected_leads = pd.DataFrame()
            st.session_state.current_auction = None

            # STREAMED DISCOVERY: stop here, render the auction list
            # immediately, then continue sampling per-auction in the
            # background phase below. The previous flow blocked the
            # entire discovery on the slow per-auction sample step
            # (~25-30s for 200 auctions); now the list is usable in
            # ~5-10s and samples trickle in without blocking the user.
            if candidates:
                discover_result_msg = (
                    f"✅ Found {len(candidates)} candidate auction(s). "
                    "Loading content previews in the background…"
                )
                status_box.update(
                    label=(
                        f"✅ Found {len(candidates)} auctions — "
                        f"samples loading in background"
                    ),
                    state="complete", expanded=False,
                )
                # Persist the candidate list now (without samples) so
                # the next page load / tab refresh restores it. The
                # samples will be added to the same cache file on the
                # background sampling pass below.
                _save_cached_discovery(
                    candidates,
                    st.session_state.get('_sourcing_cfg', {}),
                    {},  # samples filled in by Phase 1b
                )
                st.session_state.pop('_discovery_restored_from', None)
                # NOTE: Phase 1b background sampling NO LONGER auto-fires.
                # The user explicitly asked for the auction list to render
                # immediately without further work blocking the main panel.
                # Sampling now runs only when the user clicks "🎯 Scan
                # visible auctions" (which triggers the full BOLO scan)
                # or clicks an individual auction row to drill in. Setting
                # to False so any prior pending flag is cleared.
                st.session_state._sampling_pending = False
            else:
                discover_result_msg = "⚠️ No auctions matched your filters."
                status_box.update(
                    label="⚠️ No matching auctions",
                    state="error", expanded=True,
                )
                st.session_state._sampling_pending = False
        except Exception as e:
            import traceback
            discover_error = f"{type(e).__name__}: {e}"
            st.session_state._last_discover_traceback = traceback.format_exc()
            st.error(f"❌ {discover_error}")
            st.code(traceback.format_exc(), language="python")
            status_box.update(label="❌ Discovery failed", state="error", expanded=True)
        finally:
            st.session_state.discover_running = False

    st.session_state._discover_status = {
        "error": discover_error,
        "msg": discover_result_msg,
    }
    st.rerun()


# ================================================================
# WORK BLOCK: Phase 1b — background auction-content sampling
# ================================================================
# Streamed discovery splits the slow part (sample lot titles per
# auction) out of the discovery work block above. The candidate
# list renders immediately after Phase 1a completes; this block
# fires on the next render and fills in the samples without
# blocking the user. The picker tolerates empty samples (the
# "What's in this auction" column is just blank for unsampled
# rows) so the user can already start scanning / filtering.
if st.session_state.get('_sampling_pending'):
    _keep_screen_awake()
    candidates = st.session_state.get('auction_candidates', []) or []
    if not candidates:
        st.session_state._sampling_pending = False
    else:
        cfg = st.session_state.get('_sourcing_cfg', {})
        scraper = Phase1Scraper(config_path="config.json")
        scraper.zip_code = cfg.get("zip", "")
        scraper.radius = cfg.get("radius", 20)
        scraper.include_nationwide = cfg.get("include_nationwide", True)

        with st.status(
            f"🔍 Loading content previews for "
            f"{len(candidates)} auctions in the background…",
            expanded=True,
        ) as sample_status_box:
            st.write(
                "The auction list above is already usable — this step "
                "fills in the **'What's in this auction'** preview "
                "column + BOLO hints. Click an auction or run "
                "**🔍 Search for BOLOs** anytime; you don't have to wait."
            )
            sample_progress = st.progress(
                0, text=f"Sampling 0/{len(candidates)}…"
            )

            def _bg_sample_prog(current, total, label=""):
                pct = current / total if total > 0 else 1.0
                sample_progress.progress(
                    min(pct, 1.0),
                    text=label or f"Sampled {current}/{total}",
                )

            cat_samples_map: dict = {}
            try:
                cat_samples_map = run_async(
                    scraper.sample_categories_batch(
                        candidates,
                        sample_size=8,    # was 20 — 60% smaller payload
                        batch_size=30,    # was 15 — 2x concurrency
                        progress_callback=_bg_sample_prog,
                    )
                )
            except Exception:
                # Sampling is a nice-to-have — never stall the picker
                # because a preview call blew up. Empty dict means the
                # picker just shows blank previews; user can still
                # operate normally.
                cat_samples_map = {}
            sample_progress.empty()

            st.session_state.category_samples = cat_samples_map
            # Refresh the persisted cache with the sample data so the
            # next session restore includes them. The candidate list
            # was already saved in Phase 1a; this update merges the
            # samples into the same file.
            try:
                _save_cached_discovery(
                    candidates,
                    st.session_state.get('_sourcing_cfg', {}),
                    cat_samples_map,
                )
            except Exception:
                pass

            sampled = len(cat_samples_map)
            sample_status_box.update(
                label=(
                    f"✅ Loaded previews for "
                    f"{sampled} of {len(candidates)} auctions"
                ),
                state="complete", expanded=False,
            )

        st.session_state._sampling_pending = False
        st.rerun()


# ================================================================
# WORK BLOCK: Fetch lots for selected auctions
# ================================================================
if st.session_state.get('fetch_lots_running'):
    _keep_screen_awake()
    fetch_error = None
    fetch_result_msg = None
    _scan_all_label = (
        "📥 Fetching every lot for BOLO scan…"
        if st.session_state.get('_bolo_scan_all_pending')
        else "📥 Fetching lots for selected auctions…"
    )
    with st.status(_scan_all_label, expanded=True) as status_box:
        try:
            candidates = st.session_state.get('auction_candidates', [])
            sel_ids = set(st.session_state.get('_selected_auction_ids', []))
            selected_candidates = [c for c in candidates if c['auction_id'] in sel_ids]

            # Fallback: synthesize candidates from session-state DataFrames
            # when the candidate list doesn't have these IDs. Common when
            # refreshing a cached analysis after the 24h discovery cache
            # expired — we still know enough about the auction (link,
            # name, source, closing date) to refetch its lots.
            if not selected_candidates and sel_ids:
                seen_ids = {c['auction_id'] for c in selected_candidates}
                for source_df_name in ('selected_leads', 'audit_results',
                                        'phase1_leads'):
                    src = st.session_state.get(source_df_name)
                    if not isinstance(src, pd.DataFrame) or src.empty:
                        continue
                    if 'auction_link' not in src.columns:
                        continue
                    for aid in sel_ids - seen_ids:
                        sub = src[src['auction_link'].fillna('').astype(str)
                                  .str.contains(f'/auction/{aid}', regex=False)]
                        if sub.empty:
                            continue
                        first = sub.iloc[0]
                        selected_candidates.append({
                            'auction_id': aid,
                            'name': str(first.get('auction') or '(unknown)'),
                            'source': str(first.get('source') or ''),
                            'date_end': str(first.get('closing_date') or ''),
                            'lot_count': len(sub),
                            'date_info': '',
                            'city': '',
                            'state': '',
                        })
                        seen_ids.add(aid)

            if not selected_candidates:
                raise RuntimeError(
                    f"No matching auctions found for selected IDs ({len(sel_ids)} ids, "
                    f"{len(candidates)} candidates). This usually means the candidate "
                    "list was reset between clicks."
                )

            st.write(
                f"Deep-scanning **{len(selected_candidates)}** auction(s): "
                + ", ".join(
                    f"*{a.get('name', a.get('auction_id'))}*"
                    for a in selected_candidates[:3]
                )
                + (f" + {len(selected_candidates) - 3} more"
                   if len(selected_candidates) > 3 else "")
                + "."
            )

            cfg = st.session_state.get('_sourcing_cfg', {})
            scraper = Phase1Scraper(config_path="config.json")
            scraper.category_filter = cfg.get("category_filter", [])

            fetch_progress = st.progress(0, text="Fetching lots…")
            fetch_live_detail = st.empty()

            # Detect what kind of scan triggered this fetch so the
            # progress line can explain that the running lot count is
            # the FETCH total — the keyword / BOLO filter runs AFTER
            # this finishes. Without this context, users see e.g.
            # "15,000 lots fetched" while searching for 'rolex box'
            # and worry that 15k lots actually match.
            _scan_kind_for_label = None
            _scan_term_for_label = None
            if st.session_state.get('_keyword_scan_pending'):
                _scan_kind_for_label = 'keyword'
                _scan_term_for_label = (
                    st.session_state.get('_keyword_scan_pending') or ''
                ).strip()
            elif st.session_state.get('_bolo_scan_all_pending'):
                _scan_kind_for_label = 'bolo'

            def fetch_prog(current, total, label="", extras=None):
                if total == 0:
                    pct, text = 0.0, (label or "Done")
                else:
                    pct = current / total if total > 0 else 0
                    text = (
                        f"{label} · {current}/{total} auctions fetched"
                        if label
                        else f"{current}/{total} auctions fetched"
                    )
                fetch_progress.progress(min(pct, 1.0), text=text)
                # Live detail line: running lot count + names of the
                # auctions just completed this batch + any errors so far
                if extras:
                    running = extras.get('running_kept', 0)
                    batch_lots = extras.get('batch_lots', 0)
                    batch_names = extras.get('batch_names') or []
                    errs = extras.get('errors_so_far', 0)
                    name_preview = ", ".join(
                        n[:30] for n in batch_names[:3]
                    )
                    if len(batch_names) > 3:
                        name_preview += f" + {len(batch_names) - 3} more"
                    err_bit = (
                        f" · ⚠️ {errs} errors so far" if errs else ""
                    )
                    # Reframe the running count when we're in a
                    # filter-mode scan so the user understands this
                    # is "lots downloaded so far" — not "lots that
                    # match your keyword".
                    if _scan_kind_for_label == 'keyword':
                        running_label = (
                            f"📥 Downloading lots to search "
                            f"(**{running:,}** scanned so far · keyword "
                            f"filter for '{_scan_term_for_label}' runs after "
                            f"fetch completes)"
                        )
                    elif _scan_kind_for_label == 'bolo':
                        running_label = (
                            f"📥 Downloading lots to scan "
                            f"(**{running:,}** scanned so far · BOLO match "
                            f"runs after fetch completes)"
                        )
                    else:
                        running_label = (
                            f"📦 **{running:,}** lots fetched so far"
                        )
                    fetch_live_detail.markdown(
                        f"{running_label} · "
                        f"+{batch_lots:,} from this batch · "
                        f"just finished: {name_preview}{err_bit}"
                    )

            _is_bolo_scan_all = bool(st.session_state.get('_bolo_scan_all_pending'))
            _expected_lots = sum(int(c.get('lot_count') or 0) for c in selected_candidates)
            # When the keyword-scan path is active, pass the term down
            # to the scraper so HiBid filters server-side. Most auctions
            # return 0 lots → no pagination → a 97-auction scan drops
            # from ~2 minutes to ~5 seconds. Empty string = no filter,
            # which is what BOLO scan and single drills use.
            _server_search_text = (
                st.session_state.get('_keyword_scan_pending') or ''
            ).strip()
            _mode_label = (
                'keyword scan' if _server_search_text
                else 'BOLO scan all' if _is_bolo_scan_all
                else 'single drill'
            )
            tlog("FETCH",
                 f"Phase 1 starting · {len(selected_candidates)} auctions",
                 f"· ~{_expected_lots:,} expected lots",
                 f"· mode={_mode_label}",
                 f"· searchText='{_server_search_text}'" if _server_search_text else "")
            _fetch_t0 = _time.time()

            df = run_async(
                scraper.fetch_lots_for_selected(
                    selected_candidates, progress_callback=fetch_prog,
                    search_text=_server_search_text,
                )
            )
            fetch_progress.empty()
            fetch_live_detail.empty()
            _fetch_elapsed = _time.time() - _fetch_t0
            tlog("FETCH",
                 f"Phase 1 done in {_fetch_elapsed:.1f}s",
                 f"· kept {len(df):,} lots",
                 f"(raw {df.attrs.get('raw_count', 0):,} ·",
                 f"dropped status {df.attrs.get('filtered_by_status', 0)} ·",
                 f"dropped category {df.attrs.get('filtered_by_category', 0)})")

            # ---- Multi-auction cache merge ----
            # Overlay cached analysis (verdict, est_resale, comp data)
            # onto fresh Phase 1 lots from auctions we've already
            # analyzed within the TTL window. Saves the audit + comp
            # spend on auctions that haven't changed. The audit
            # fast-path naturally skips lots with verdicts; the comp
            # pipeline naturally skips lots with non-NaN est_resale.
            if ((st.session_state.get('_bolo_scan_all_pending')
                    or st.session_state.get('_multi_select_pending'))
                    and not df.empty):
                df, _cache_stats = _merge_cached_analysis_multi(df)
                if _cache_stats and _cache_stats.get('cached_auctions', 0) > 0:
                    st.write(
                        f"💾 Reused cached analysis for "
                        f"**{_cache_stats['cached_auctions']}** of "
                        f"**{_cache_stats['total_auctions']}** auctions "
                        f"(**{_cache_stats['cached_lots']:,}** lots already "
                        f"analyzed) · running fresh on "
                        f"**{_cache_stats['new_auctions']}** new auction(s)."
                    )

            st.session_state.phase1_leads = df

            # Interrupted-scan insurance: for the long scan modes (BOLO
            # scan-all / keyword / multi-select basket), persist the
            # freshly-fetched frame + pending flags NOW — before the
            # match phase — so a session death during match/load can
            # resume without re-fetching. Single-auction drills are
            # cheap to re-fetch and skip this.
            _scan_mode = {
                'bolo': bool(st.session_state.get('_bolo_scan_all_pending')),
                'keyword': (st.session_state.get('_keyword_scan_pending') or ''),
                'multi': bool(st.session_state.get('_multi_select_pending')),
                'selected_ids': st.session_state.get('_selected_auction_ids'),
            }
            if _scan_mode['bolo'] or _scan_mode['keyword'] or _scan_mode['multi']:
                _save_fetched_frame(df, _scan_mode)

            # Stash the actual fetch scope so the post-fetch keyword /
            # BOLO branches can show "searched X of Y auctions" where
            # Y is what we ASKED HiBid about — not what came back.
            # With server-side `searchText` filtering, auctions with
            # zero matches return empty payloads → `df['auction'].nunique()`
            # undercounts. We need the queried count, which is the
            # length of `selected_candidates` (the auction list passed
            # to the scraper). `df.attrs['per_auction']` is also a
            # reliable source (one entry per queried auction, even
            # when the auction returned 0 lots).
            st.session_state._last_fetch_scope = {
                "auctions_queried": len(selected_candidates),
                "lots_expected": _expected_lots,
            }

            # Grow the known-category list for the sidebar filter
            if not df.empty and 'category' in df.columns:
                seen = {c for c in df['category'].dropna().astype(str).tolist() if c}
                st.session_state.known_categories = sorted(
                    set(st.session_state.known_categories) | seen
                )

            diag = dict(df.attrs)  # copy BEFORE we mutate / rerun
            raw_count = diag.get('raw_count', 0)
            filtered_cat = diag.get('filtered_by_category', 0)
            filtered_status = diag.get('filtered_by_status', 0)
            per_auction = diag.get('per_auction', []) or []
            errors = diag.get('errors', []) or []
            status_values = diag.get('status_values_seen', {}) or {}

            # Show the diagnostic breakdown inside the status box — one place
            # to understand exactly where the user's items went.
            st.write(
                f"**Raw lots from HiBid:** {raw_count} · "
                f"**Kept:** {len(df)} · "
                f"**Dropped by status (CLOSED / Bidding Closed):** {filtered_status} · "
                f"**Dropped by category filter:** {filtered_cat}"
            )

            if status_values:
                st.write(f"**lotState.status values seen:** `{status_values}`")

            if per_auction:
                with st.expander(f"Per-auction breakdown ({len(per_auction)})", expanded=(len(df) == 0)):
                    st.table(pd.DataFrame(per_auction))

            if errors:
                st.warning(f"**{len(errors)} auction(s) errored during fetch** — see details:")
                st.table(pd.DataFrame(errors))

            local_count = int((df['source'] == "Local Pickup").sum()) if not df.empty and 'source' in df.columns else 0
            ship_count = int((df['source'] == "Ship").sum()) if not df.empty and 'source' in df.columns else 0
            auction_count = df['auction'].nunique() if not df.empty else 0
            hard_count = int((df['logistics_ease'] == "HARD").sum()) if not df.empty and 'logistics_ease' in df.columns else 0
            easy_count = int((df['logistics_ease'] == "EASY").sum()) if not df.empty and 'logistics_ease' in df.columns else 0

            if df.empty:
                # Be honest about which of the three possible causes it was.
                if raw_count == 0:
                    if errors:
                        reason = (
                            f"HiBid returned **0 lots** for every selected auction. "
                            f"{len(errors)} auction(s) threw errors above — that's the likely cause."
                        )
                    else:
                        reason = (
                            "HiBid returned **0 lots** for every selected auction. "
                            "The GraphQL call succeeded but the response contained no lots. "
                            "This can happen for auctions in a pre-opening preview state, "
                            "or if HiBid's schema changed."
                        )
                elif filtered_status == raw_count:
                    reason = (
                        f"All {raw_count} lots were dropped as CLOSED / 'Bidding Closed'. "
                        f"Status values seen: `{status_values}`."
                    )
                elif filtered_cat == raw_count:
                    reason = f"All {raw_count} lots were excluded by the category filter."
                else:
                    reason = f"{raw_count} raw lots, all dropped by a mix of status + category filters."
                fetch_result_msg = f"⚠️ Scan complete — 0 items survived. {reason}"
                status_box.update(
                    label="⚠️ No open lots matched",
                    state="error", expanded=True,
                )
            else:
                breakdown_bits = [f"📦 {easy_count} easy-ship"]
                if hard_count:
                    breakdown_bits.append(f"🏋️ {hard_count} HARD (hidden by default)")
                fetch_result_msg = (
                    f"✅ Scanned {len(df)} items across {auction_count} auction(s)"
                    f" — {local_count} local, {ship_count} shippable"
                    f" · {' · '.join(breakdown_bits)}."
                )
                status_box.update(
                    label=f"✅ {len(df)} items from {auction_count} auction(s)",
                    state="complete", expanded=False,
                )
        except Exception as e:
            import traceback
            fetch_error = f"{type(e).__name__}: {e}"
            st.session_state._last_fetch_traceback = traceback.format_exc()
            st.error(f"❌ {fetch_error}")
            st.code(traceback.format_exc(), language="python")
            status_box.update(label="❌ Fetch failed", state="error", expanded=True)
        finally:
            st.session_state.fetch_lots_running = False

    # Persist full diagnostics so the next rerun can re-render them —
    # otherwise everything written inside the status box above is wiped.
    try:
        _diag_payload = {
            "raw_count": int(raw_count),
            "kept": int(len(df)),
            "filtered_status": int(filtered_status),
            "filtered_cat": int(filtered_cat),
            "per_auction": per_auction,
            "errors": errors,
            "status_values": status_values,
        }
    except NameError:
        # We errored before computing diagnostics (exception path)
        _diag_payload = None

    st.session_state._fetch_status = {
        "error": fetch_error,
        "msg": fetch_result_msg,
        "diag": _diag_payload,
    }
    st.rerun()


# --- Shared column config for discovery tables ---
DISCOVERY_COL_CONFIG = {
    "lot_link": st.column_config.LinkColumn("Item", display_text="Open"),
    "current_bid": st.column_config.NumberColumn("Bid", format="$%.2f"),
    "bid_count": st.column_config.NumberColumn("Bids", format="%d"),
    "est_cost": st.column_config.NumberColumn("Est. Cost", format="$%.2f"),
    "logistics_ease": st.column_config.TextColumn("Logistics"),
    "time_left": st.column_config.TextColumn("Time Left"),
}
DISCOVERY_COL_ORDER = ["title", "current_bid", "est_cost", "bid_count",
                       "time_left", "lot_link", "category", "logistics_ease"]


# `_extract_auction_id` and `_load_auction_for_analysis` were both
# hoisted near the top of the file so the Memory popover (cached-auction
# buttons) and the header refresh button can call them. Their original
# definition sites were here.


def _compute_grade_stats(ar: pd.DataFrame):
    """Compute (a_count, b_count, total) from an analyzed df.

    Replaces the old _compute_green_stats that used ROI/STR thresholds.
    Now driven by the buy_grade column — counts A and B grades among
    comp-able lots (shippable + not red-flagged + has est_resale).
    The sidebar shows these counts to help the user compare auctions
    at a glance ("12 A · 30 B" vs "0 A · 2 B").
    """
    if ar is None or not isinstance(ar, pd.DataFrame) or ar.empty:
        return None, None, None
    if 'buy_grade' not in ar.columns:
        return None, None, None

    # Comp-able universe: shippable, not red-flagged, has a price.
    eligible = (
        ~ar.get('red_flag', pd.Series(False, index=ar.index)).fillna(False).astype(bool)
        & (ar.get('logistics_ease', pd.Series('', index=ar.index)) != 'HARD')
        & ar.get('est_resale', pd.Series(pd.NA, index=ar.index)).notna()
    )
    total = int(eligible.sum())
    if total == 0:
        return 0, 0, 0

    grades = ar['buy_grade'].fillna('').astype(str)
    a_mask = grades.str.startswith('🟢') & eligible
    b_mask = grades.str.startswith('🟡') & eligible
    return int(a_mask.sum()), int(b_mask.sum()), total


def _save_current_auction_to_cache():
    """Persist the current audit_results DataFrame to the disk cache."""
    ar = st.session_state.get('audit_results')
    if not isinstance(ar, pd.DataFrame) or ar.empty:
        return
    auction_name = st.session_state.get('current_auction') or ""
    # Skip per-auction cache writes for synthetic multi-auction views
    # (BOLO scan across all auctions). The lots come from many
    # different auction IDs, so a single cache key isn't meaningful —
    # and re-loading would need to be triggered by the scan button,
    # not by clicking a cached entry. The COMPED-LOTS REGISTRY still
    # gets written below regardless, so future scans can reuse this
    # comp data.
    if (auction_name.startswith("🎯 BOLO scan")
            or auction_name.startswith("🔍 Keyword:")
            or auction_name.startswith("🧺")):
        return
    auction_id = _extract_auction_id(ar)
    if auction_id is None:
        return
    closing_date = ""
    if 'closing_date' in ar.columns and not ar.empty:
        closing_date = str(ar['closing_date'].iloc[0])

    # Refresh buy_grade for sidebar accuracy. ar may not have max_bid
    # populated (or may have stale buy_grade from before the render
    # path computed max_bid). Recompute both on a throwaway copy so
    # the persisted a_count / b_count reflect the actual grades.
    _ar_scored = ar
    try:
        target_roi_val = float(
            st.session_state.get("target_roi_live", 3.0) or 3.0
        )
        _ar_scored = _compute_max_bid(ar, target_roi_val)
        _ar_scored = _compute_buy_score(_ar_scored)
    except Exception:
        pass
    a_count, b_count, total_count = _compute_grade_stats(_ar_scored)
    # Compute BOLO match count using the current brand list. Cheap
    # regex pass over titles + descriptions; result persists in the
    # cache payload so the sidebar can show "🎯 N BOLO" badges
    # without rehydrating every auction's lots on each render.
    bolo_count = 0
    try:
        if 'title' in ar.columns:
            bolo_df = _compute_bolo_columns(ar)
            if 'bolo_brand' in bolo_df.columns:
                bolo_count = int(bolo_df['bolo_brand'].notna().sum())
    except Exception:
        bolo_count = 0

    # Competitiveness signal — average bids per lot. Persists alongside
    # the green % so the sidebar can show 🔥/🟡/🧊 markers without
    # rehydrating the whole auction on each render. Read by the
    # auction-list renderer in `_render_sidebar_auction_list`.
    bids_per_lot = None
    try:
        if 'bid_count' in ar.columns and len(ar) > 0:
            bc = pd.to_numeric(ar['bid_count'], errors='coerce').fillna(0)
            bids_per_lot = float(bc.mean())
    except Exception:
        bids_per_lot = None

    # Drop-ship-channel signature — % of lots whose titles match
    # known SEO-spam patterns (DOTEFFIL silver-plated, eye-mask, etc.).
    # Persists for sidebar display so the user can spot pure-dropship
    # auctions like Mystery Mega before clicking in. Threshold for the
    # 🚨 badge applied at render time, so we only need the raw %.
    dropship_pct = None
    try:
        if 'title' in ar.columns and len(ar) > 0:
            n_dropship = sum(
                1 for t in ar['title'].fillna('').astype(str)
                if _is_dropship_lot(t)
            )
            dropship_pct = round(100.0 * n_dropship / len(ar), 1)
    except Exception:
        dropship_pct = None

    try:
        _AUCTION_CACHE.save(
            auction_id, auction_name, ar, closing_date,
            a_count=a_count,
            b_count=b_count,
            total_count=total_count,
            bolo_count=bolo_count,
            bids_per_lot=bids_per_lot,
            dropship_pct=dropship_pct,
        )
    except Exception as e:
        # Don't crash the app over a cache write failure
        st.warning(f"Could not save analysis to cache: {e}")




@st.cache_resource(show_spinner=False)
def _get_auditor(model_name: str, vision_provider: str = "claude"):
    """Load (and cache) the Phase2Scraper.

    The constructor reads the Anthropic + Gemini keys from `config.json`
    and lazy-creates the API client on first use — no model download,
    instant init. Cached per (model_name, vision_provider) so switching
    the provider mid-session hands back a correctly-configured instance
    instead of the stale one.
    """
    from scraper import Phase2Scraper
    return Phase2Scraper(model_name=model_name, vision_provider=vision_provider)


# Words/phrases that indicate something MIGHT be wrong with a lot,
# even if it's a tier-1 BOLO match. If any of these appear in the
# title or description, the lot does NOT qualify for the audit-skip
# fast path — it must run through the real audit. We compile each as
# a word-boundaried regex so "stain" matches "stain"/"stains"/"stained"
# but NOT "stainless", and "rip" matches "rip"/"rips"/"ripped" but NOT
# "stripe"/"grip"/"trip".
_AUDIT_SKIP_RED_FLAG_PATTERN = re.compile(
    r"\b("
    # Damage / breakage. Each verb enumerates explicit conjugations so
    # word-boundary anchors fire correctly — `stain\w*` would gobble
    # "stainless", `chip\w*` would match "chipset", etc. Stick to
    # specific conjugated forms wrapped in \b.
    r"broken|break|breaks|damage|damages|damaged|"
    r"crack|cracks|cracked|"
    r"chip|chipped|"
    r"dent|dents|dented|"
    r"scratch|scratches|scratched|scratching|"
    r"stain|stains|stained|staining|"
    r"rip|rips|ripped|tear|tears|torn|"
    # Functionality issues
    r"untested|not\s+working|doesn'?t\s+work|does\s+not\s+work|"
    r"no\s+power|won'?t\s+turn\s+on|won'?t\s+work|needs\s+repair|"
    # Cleaning / contamination
    r"smoke|moldy|mildew|water\s+damage|pet\s+hair|stinks|"
    r"needs\s+cleaning|"
    # Counterfeit / replica
    r"replica|counterfeit|knockoff|knock\s+off|not\s+authentic|"
    r"fake|reproduction|repro|"
    # Auction-house "as-is" code phrases
    r"as[-\s]is|as[-\s]found|for\s+parts|parts\s+only|missing"
    r")\b",
    re.IGNORECASE,
)


def _qualifies_for_audit_skip(row) -> bool:
    """Return True when a lot can skip the AI audit — tier-1 BOLO
    match + clean specific title + no red-flag phrases in description.

    The skip path saves ~$0.001-0.005 per lot in Anthropic tokens
    (text-API tier) or up to $0.01 (image-API tier on lots with
    short descriptions). On a 1,200-lot BOLO scan that's $1-4 per
    run — modest but free if we can do it without sacrificing
    accuracy.

    Criteria (all must hold):
      1. BOLO tier == 1 (curated highest-value brands we trust)
      2. NOT HARD logistics (those get auto-flagged anyway)
      3. NOT a stylized replica candidate (the matcher would mark it)
      4. Title is "specific" — has year / model # / or 3+ informative
         words. _is_generic_title is the existing heuristic.
      5. Description (if present) doesn't contain any red-flag
         phrase from _AUDIT_SKIP_RED_FLAG_PHRASES.
      6. Title doesn't contain any red-flag phrase either.

    We're intentionally conservative: when in doubt, fall through to
    the real audit. False positives here mean a broken/damaged lot
    gets through with a "Looks good" verdict — that's a wasted comp
    spend later (~10× more expensive than the audit we just skipped).
    Net is still positive but only if the skip rate is high AND
    error rate stays low.
    """
    bolo_tier = row.get('bolo_tier')
    if bolo_tier != 1 and bolo_tier != "1":
        return False
    if str(row.get('logistics_ease') or '').upper() == 'HARD':
        return False
    if bool(row.get('is_stylized_replica')):
        return False
    title = str(row.get('title') or '')
    if _is_generic_title(title):
        return False
    desc = str(row.get('description') or '')
    haystack = f"{title} {desc}"
    if _AUDIT_SKIP_RED_FLAG_PATTERN.search(haystack):
        return False
    return True


def _run_ai_audit(leads_df, on_progress=None):
    """Run Phase 2 AI condition audit with detailed phase-by-phase status."""
    from scraper import Phase2Scraper

    total = len(leads_df)
    workers = int(st.session_state.get('audit_workers', 8))
    _prov_label = (
        "Gemini"
        if str(st.session_state.get('vision_provider', 'claude')).lower() == 'gemini'
        else "Claude"
    )

    with st.status(f"🧠 AI Condition Audit ({_prov_label})…", expanded=True) as status:
        # ---- Pre-pass: skip lots that already have a verdict ----
        # When the multi-auction cache merge runs, lots from previously
        # analyzed auctions land here with `verdict` already populated
        # from cache. Those don't need to hit Claude again — the
        # verdict is stable for the life of the auction.
        cached_verdict_mask = pd.Series(False, index=leads_df.index)
        if 'verdict' in leads_df.columns:
            cached_verdict_mask = leads_df['verdict'].notna() & (
                leads_df['verdict'].astype(str).str.strip() != ''
            )
        n_cached_audit = int(cached_verdict_mask.sum())
        if n_cached_audit > 0:
            st.write(
                f"💾 **Cache hit:** {n_cached_audit} of {total} lots "
                f"already have a verdict from a prior run — skipping "
                f"the audit pass for those."
            )

        # ---- Pre-pass: tier-1 BOLO clean-title fast path ----
        # Identify rows that qualify for an audit-skip — tier-1 BOLO
        # matches with specific titles + no red-flag phrases. These
        # get a synthetic "Looks good" verdict without an API call;
        # everything else goes through the normal three-tier flow.
        skip_mask = pd.Series(False, index=leads_df.index)
        if 'bolo_tier' in leads_df.columns:
            skip_mask = leads_df.apply(
                _qualifies_for_audit_skip, axis=1
            ).fillna(False).astype(bool)
        # Don't count lots already covered by cache — they're handled
        # via a separate concat below.
        skip_mask = skip_mask & ~cached_verdict_mask
        n_skip = int(skip_mask.sum())
        n_audit = total - n_skip - n_cached_audit

        if n_skip > 0:
            st.write(
                f"⚡ **Audit-skip fast path:** {n_skip} of {total} lots "
                f"qualify for the BOLO clean-skip "
                f"({100 * n_skip / total:.0f}%). These are tier-1 BOLO "
                f"matches with specific titles + clean descriptions — "
                f"synthesizing a 'Looks good' verdict without an API call. "
                f"Estimated savings: ~${n_skip * 0.0025:.2f} in Claude "
                f"tokens. Remaining {n_audit} lots run through the "
                f"normal three-tier audit."
            )

        # Build the synthetic-verdict df for skipped rows. Mirrors the
        # column shape Phase2Scraper.batch_audit produces so downstream
        # code (comps gate, results table) doesn't need to special-case.
        if n_skip > 0:
            skipped_df = leads_df[skip_mask].copy()
            skipped_df['enriched_title'] = skipped_df.get('title', '')
            skipped_df['verdict'] = "Looks good (BOLO tier-1 fast path)"
            skipped_df['confidence'] = 90.0
            skipped_df['red_flag'] = False
            skipped_df['audit_source'] = "skip_bolo_clean"
        else:
            skipped_df = None

        # Cache-hit rows already carry verdict/red_flag from the prior
        # auction analysis. Stamp `audit_source` so the results-view
        # status panel can credit them to "cache" instead of fresh API.
        if n_cached_audit > 0:
            cached_audit_df = leads_df[cached_verdict_mask].copy()
            if 'audit_source' not in cached_audit_df.columns:
                cached_audit_df['audit_source'] = 'cache'
            else:
                cached_audit_df.loc[
                    cached_audit_df['audit_source'].isna(), 'audit_source'
                ] = 'cache'
        else:
            cached_audit_df = None

        # If everything qualified, no need to run the auditor at all.
        if n_audit == 0:
            status.update(
                label=(
                    f"⚡ Audit fast-skipped: {n_skip} BOLO clean + "
                    f"{n_cached_audit} cached"
                ),
                state="complete", expanded=False,
            )
            parts = [d for d in (cached_audit_df, skipped_df) if d is not None]
            if not parts:  # Defensive — leads_df was empty
                return leads_df.copy()
            return pd.concat(parts, ignore_index=False).sort_index()

        # Otherwise narrow leads_df to just the rows that need real audit
        # (NOT in the BOLO-clean fast path AND NOT already cached).
        audit_target_mask = ~(skip_mask | cached_verdict_mask)
        audit_target_df = leads_df[audit_target_mask].copy()

        # Phase 1: pre-flight
        _vprov = str(st.session_state.get('vision_provider', 'claude')).lower()
        auditor = _get_auditor(Phase2Scraper.DEFAULT_MODEL, _vprov)
        # Which key must be present depends on the resolved engine: Gemini
        # runs on the Google key, Claude on the Anthropic key.
        _eng = auditor._image_provider()
        _key_ok = (bool(auditor.gemini_api_key) if _eng == "gemini"
                   else bool(auditor.api_key) if _eng == "claude"
                   else False)
        # If the cached Phase2Scraper instance is missing the key it needs
        # but config now HAS one, force a rebuild — this happens when the
        # user added the key to config.json AFTER Streamlit started, and
        # @st.cache_resource has been holding the stale instance ever
        # since. Without this, the audit silently downgrades to
        # keyword-only and every non-keyword lot gets `no_api_key`. Guard
        # on config actually having the key so we don't clear+rebuild on
        # every rerun when the user simply has no key configured at all.
        if not _key_ok:
            from scraper.config_loader import load_config
            _cfg_now = load_config()
            _needed = (
                (_cfg_now.get("gemini") or {}).get("api_key")
                if _vprov == "gemini"
                else (_cfg_now.get("anthropic") or {}).get("api_key")
            )
            if _needed:
                _get_auditor.clear()
                auditor = _get_auditor(Phase2Scraper.DEFAULT_MODEL, _vprov)
                _eng = auditor._image_provider()
                _key_ok = (bool(auditor.gemini_api_key) if _eng == "gemini"
                           else bool(auditor.api_key) if _eng == "claude"
                           else False)
        if not _key_ok:
            if _vprov == "gemini":
                st.warning(
                    "⚠️ Vision provider is **Gemini** but no Google key found "
                    "in `config.json` under `gemini.api_key`. The audit will "
                    "run keyword-only. Grab a free key at "
                    "aistudio.google.com/apikey, or switch the provider back "
                    "to Claude in ⚙️ Audit settings."
                )
            else:
                st.warning(
                    "⚠️ No Anthropic API key found in `config.json`. The audit "
                    "will run keyword-only — items without a keyword match will "
                    "be marked 'Unknown' instead of red-flagged. Add your key "
                    "under `anthropic.api_key` to enable AI classification."
                )

        # Gemini hard-gate: a valid key on a $0-balance project returns 429
        # RESOURCE_EXHAUSTED, which the vision wrappers swallow to None — so
        # without this, a whole run would silently classify every lot
        # keyword-only. Ping once up front; on failure ABORT loudly (raised
        # here, caught by the caller's try/except → st.error) rather than
        # burning the run or, worse, silently degrading. No Claude fallback
        # — the user chose Gemini to avoid Claude spend; respect that.
        if _eng == "gemini" and _key_ok:
            _gp_err = _gemini_ping(auditor.gemini_api_key, auditor.gemini_model)
            if _gp_err:
                raise RuntimeError(
                    f"Gemini provider unavailable — {_gp_err} "
                    "(No lots were audited and no credits were spent. Fix the "
                    "Gemini key/billing, or switch Audit AI provider to Claude "
                    "in ⚙️ Audit settings, then re-run.)"
                )

        # Pre-count how many lots will be pre-filtered (HARD logistics) so
        # the user sees the savings up front.
        hard_preview = 0
        if 'logistics_ease' in audit_target_df.columns:
            hard_preview = int((audit_target_df['logistics_ease'] == 'HARD').sum())

        st.write(
            f"**🔎 Step 1/2 — Three-tier condition classification** "
            f"on {n_audit} items"
            + (f" (after fast-skipping {n_skip} clean tier-1 BOLO matches)"
               if n_skip > 0 else "")
            + "."
        )
        _eng_name = "Gemini flash-lite" if _prov_label == "Gemini" else "Claude Haiku"
        st.caption(
            "**Tier 1 — keyword regex** over the description (instant, free): "
            "matches phrases like *'untested'*, *'doesn't work'*, *'factory "
            "sealed'*. Most lots short-circuit here.  \n"
            f"**Tier 2 — {_eng_name} text** (~500ms/lot, parallel): for lots "
            "with a substantial description but no keyword hit.  \n"
            f"**Tier 3 — {_eng_name} vision** (~2s/lot, parallel): for lots "
            "with a short description, classifies from the thumbnail."
        )
        if hard_preview > 0:
            st.caption(
                f"⏭️ Skipping AI on **{hard_preview}** HARD-logistics lots "
                "(mattresses, vehicles, furniture, real estate, etc.) — "
                "auto-flagged as Unshippable, no API call needed."
            )
        progress_bar = st.progress(0, text=f"Starting — 0/{n_audit}")
        current_item_placeholder = st.empty()

        def ai_progress(current, total_items):
            pct = current / total_items if total_items > 0 else 1.0
            # With batching we don't know "which single item just finished" —
            # show the most recently processed row instead.
            try:
                preview_idx = min(max(current - 1, 0), total_items - 1)
                row = audit_target_df.iloc[preview_idx]
                title_preview = str(row.get('title', ''))[:70]
            except Exception:
                title_preview = ""
            progress_bar.progress(
                min(pct, 1.0),
                text=f"Analyzing {current}/{total_items}…",
            )
            if title_preview:
                current_item_placeholder.caption(
                    f"🔎 Last batch ended near: *{title_preview}*"
                )

        # Live callback: every batch boundary, push the partial df into
        # session_state and notify the caller (which throttles UI refresh).
        # We merge the in-progress audit subset back with the skipped
        # subset so the live UI sees BOTH the synthetic clean rows AND
        # whatever's been audited so far — otherwise the table would
        # show only the in-progress half during the run.
        def _audit_live_cb(processed, total_items, partial_df):
            try:
                if skipped_df is not None:
                    merged = pd.concat(
                        [skipped_df, partial_df], ignore_index=False
                    ).sort_index()
                else:
                    merged = partial_df
                st.session_state.audit_results = merged
                if on_progress is not None:
                    on_progress(processed, total_items)
            except Exception:
                pass

        results_df = auditor.batch_audit(
            audit_target_df,
            progress_callback=ai_progress,
            batch_size=workers,
            live_callback=_audit_live_cb,
        )

        # Merge skipped-clean + cached rows back so downstream code sees
        # one coherent results df. Preserve original row order via
        # sorted-index concat.
        extra_parts = [d for d in (cached_audit_df, skipped_df) if d is not None]
        if extra_parts:
            audit_attrs = dict(results_df.attrs or {})
            results_df = pd.concat(
                extra_parts + [results_df], ignore_index=False
            ).sort_index()
            results_df.attrs.update(audit_attrs)
            results_df.attrs['audit_skipped_bolo_clean'] = n_skip
            results_df.attrs['audit_skipped_cached'] = n_cached_audit

        # Phase 2: summarize (counts + per-tier breakdown)
        # red_flag may round-trip as int via cache — coerce to bool first.
        if 'red_flag' in results_df.columns:
            _rf_sum = results_df['red_flag'].fillna(False).astype(bool)
            good = int((~_rf_sum).sum())
            flagged = int(_rf_sum.sum())
        else:
            good = 0
            flagged = 0
        skipped_hard = int(results_df.attrs.get('audit_skipped_hard', 0) or 0)
        skipped_collectible = int(results_df.attrs.get('audit_skipped_collectible', 0) or 0)
        skipped_empty = int(results_df.attrs.get('audit_skipped_empty', 0) or 0)
        skipped_bolo_clean = int(results_df.attrs.get('audit_skipped_bolo_clean', 0) or 0)
        fashion_jewelry = int(results_df.attrs.get('audit_fashion_jewelry', 0) or 0)
        keyword_hits = int(results_df.attrs.get('audit_keyword_hits', 0) or 0)
        text_api_calls = int(results_df.attrs.get('audit_text_api_calls', 0) or 0)
        image_api_calls = int(results_df.attrs.get('audit_image_api_calls', 0) or 0)
        text_failed = int(results_df.attrs.get('audit_text_api_failed', 0) or 0)
        image_failed = int(results_df.attrs.get('audit_image_api_failed', 0) or 0)
        no_signal = int(results_df.attrs.get('audit_no_signal', 0) or 0)

        summary_parts = [f"✅ {good} good-condition", f"⚠️ {flagged} red-flagged"]
        if skipped_bolo_clean > 0:
            summary_parts.append(
                f"⚡ {skipped_bolo_clean} BOLO clean-skip "
                f"(saved ~${skipped_bolo_clean * 0.0025:.2f})"
            )
        if skipped_hard > 0:
            summary_parts.append(f"🚚 {skipped_hard} HARD-logistics (pre-filtered)")
        if skipped_collectible > 0:
            summary_parts.append(f"🎴 {skipped_collectible} collectibles (pass-through)")
        if fashion_jewelry > 0:
            summary_parts.append(
                f"💍 {fashion_jewelry} fashion jewelry (silver-plated, "
                "lab-stone — auto red-flagged)"
            )
        if skipped_empty > 0:
            summary_parts.append(f"❓ {skipped_empty} empty rows")
        st.write(f"**📊 Step 2/2 — Summary:** " + " · ".join(summary_parts))

        tier_parts = []
        if keyword_hits:
            tier_parts.append(f"🔍 {keyword_hits} by keyword")
        if text_api_calls:
            tier_parts.append(f"📝 {text_api_calls} by text API")
        if image_api_calls:
            tier_parts.append(f"🖼️ {image_api_calls} by image API")
        if no_signal:
            tier_parts.append(f"❓ {no_signal} no signal (short desc + no thumb)")
        if tier_parts:
            st.caption("**Tier breakdown:** " + " · ".join(tier_parts))
        if text_failed or image_failed:
            st.caption(
                f"⚠️ {text_failed + image_failed} API call(s) failed and were "
                "marked Unknown — check your network/key if this is high."
            )
        status.update(label="✅ AI audit complete", state="complete", expanded=False)

    return results_df


# Tokens that don't add product signal — used by _is_generic_title to
# determine whether a title has enough text-based hooks for eBay/Mercari
# search to identify the product on its own.
_GENERIC_TITLE_TOKENS = {
    'lot', 'lots', 'box', 'boxes', 'mystery', 'misc', 'miscellaneous',
    'various', 'assorted', 'mixed', 'collection', 'group', 'bundle',
    'set', 'sets', 'piece', 'pieces', 'items', 'item', 'things',
    'stuff', 'estate', 'find', 'finds', 'see', 'photo', 'photos',
    'picture', 'pictures', 'a', 'an', 'the', 'and', 'or', 'of',
    'with', 'in', 'on', 'at', 'by', 'as', 'to', 'for', 'from',
    'no', 'reserve', 'vintage', 'antique', 'old', 'used', 'new',
    'huge', 'large', 'small', 'great', 'nice', 'good', 'fair',
    'multiple', 'numerous', 'several', 'many', 'plus', 'unknown',
}


def _is_generic_title(title: str) -> bool:
    """Return True when the lot title has too little signal to identify
    the product from text alone.

    "Specific" titles (year + manufacturer + card #, model number,
    3+ informative words, etc.) are confidently searchable on text —
    eBay's image-search tier produces wrong matches on such titles
    (Topps↔OPC mix-ups, grade swaps on slabbed cards). We reserve
    image enrichment for lots like 'Box of misc items' / 'Lot 47' /
    'Mystery box' where the photo is the only way to identify the
    product.
    """
    if not title or len(title.strip()) < 5:
        return True

    # Year (1900-2030) → temporal anchor → specific
    if re.search(r'\b(19\d{2}|20\d{2})\b', title):
        return False
    # Decade marker like "1980s" / "70s" / "'90s" → specific
    if re.search(r"['']?\d{2,4}\s*['']?s\b", title):
        return False
    # Card # / lot # / SKU like #250, No. 14, #88AS-40 → specific
    if re.search(r'#\s*\w*\d+', title):
        return False
    # Model-number-ish token (alphanumeric SKU) → specific
    if re.search(r'\b[A-Z]{1,4}[\-.]?\d{2,}[\w\-.]*\b', title):
        return False

    # Count "significant" words (drop generic filler tokens). 3+ → specific.
    clean = re.sub(r'[^\w\s\-]', ' ', title.lower())
    words = [w for w in clean.split() if len(w) >= 2]
    significant = [w for w in words if w not in _GENERIC_TITLE_TOKENS]
    if len(significant) >= 3:
        return False

    return True



# Generic tokens stripped from `_title_fingerprint`. Words that are
# common across many lots provide no discriminative power for the
# fingerprint dedup — keeping them would make unrelated lots collide
# (e.g. "vintage gold ring" and "vintage gold necklace" wouldn't dedup
# against their direct duplicates if "vintage" / "gold" dominated).
_FINGERPRINT_GENERIC_TOKENS = frozenset({
    # Generic descriptors
    'lot', 'lots', 'set', 'sets', 'pair', 'pairs', 'bundle', 'bundles',
    'box', 'boxes', 'bag', 'bags', 'piece', 'pieces', 'pcs', 'pc',
    'item', 'items', 'group', 'groups', 'collection', 'collections',
    'assortment', 'assorted', 'mixed', 'misc', 'miscellaneous',
    'various', 'sundry', 'estate', 'auction',
    # Condition / quality words
    'new', 'used', 'old', 'mint', 'good', 'great', 'excellent',
    'nice', 'fine', 'beautiful', 'rare', 'unique', 'antique',
    'vintage', 'modern', 'authentic', 'original', 'genuine',
    'pre-owned', 'preowned', 'sealed', 'open', 'opened', 'unopened',
    'nib', 'nwt', 'mip', 'mib', 'nos',
    # Size / generic adjectives
    'small', 'medium', 'large', 'huge', 'mini', 'big', 'little',
    'tiny', 'jumbo', 'oversized',
    # Color words (rarely discriminate the model itself)
    'red', 'blue', 'green', 'yellow', 'black', 'white', 'gray',
    'grey', 'brown', 'pink', 'purple', 'orange', 'silver', 'gold',
    'multicolor', 'multicolored', 'colorful',
    # Articles / fillers
    'and', 'the', 'with', 'for', 'from', 'that', 'this', 'these',
    'those', 'has', 'have', 'plus', 'including', 'includes', 'incl',
    # Common HiBid / auction phrasing
    'see', 'photos', 'photo', 'pic', 'pics', 'picture', 'pictures',
    'description', 'shown', 'shows', 'as-is', 'asis',
    'ship', 'ships', 'shipping', 'shippable',
})


def _title_fingerprint(title: str) -> str:
    """Normalize a title to a comparable fingerprint string.

    Drops numbers, punctuation, generic tokens, and short words; sorts
    the remaining tokens so word order doesn't matter. Two lots with
    the same fingerprint will receive the same eBay comp lookup.
    """
    if not title:
        return ''
    s = title.lower()
    # Remove anything that's not a letter, space, or hyphen
    s = re.sub(r'[^a-z\s\-]', ' ', s)
    # Collapse runs of whitespace
    s = re.sub(r'\s+', ' ', s).strip()
    # Tokenize, drop generic + short
    tokens = [
        t for t in s.split()
        if t not in _FINGERPRINT_GENERIC_TOKENS and len(t) >= 3
    ]
    if not tokens:
        return ''
    # Sort so word order doesn't matter
    return ' '.join(sorted(set(tokens)))


# Precious-metals spot prices for melt-value computation. Refreshed
# manually (USD per gram, typical 2026 levels). The melt-value floor
# is conservative — actual eBay clearing prices are usually 30-50%
# above melt for jewelry, but melt is a no-credit proxy that lets us
# skip the eBay comp lookup entirely on weight-stamped precious-metal
# lots. Update these values periodically as spot moves.
_SPOT_PRICES_USD_PER_GRAM = {
    # 10K = 41.7% gold, 14K = 58.3%, 18K = 75%, 22K = 91.7%, 24K = 99.9%
    '10k': 16.5,
    '14k': 23.0,
    '18k': 30.0,
    '22k': 36.5,
    '24k': 39.5,
    'sterling': 0.95,    # 92.5% silver
    '925': 0.95,
    'platinum': 32.0,
    'gold-filled-1-20': 1.5,  # 1/20 12K GF roughly 1.5/gram
}

# Regex to extract karat + weight from a title.
# Matches "14K 5.2g", "10kt 3 grams", "18K 7.5gm", ".925 12g", and
# comma-formatted flatware weights ("2,045 grams" — the Towle
# Chippendale lot 7/8 stated its sterling weight exactly this way
# in the description and the old digits-only pattern missed it).
_KARAT_RE = re.compile(
    r'\b(10|14|18|22|24)[ ]*[kK][tT]?\b', re.IGNORECASE,
)
_WEIGHT_RE = re.compile(
    r'(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)\s*(?:g|gm|gms?|grams?)\b',
    re.IGNORECASE,
)
# Troy-ounce weights ("2.5 ozt", "65 troy oz") — common on scrap-
# silver and bullion-adjacent lots. Converted to grams (×31.103).
_WEIGHT_OZT_RE = re.compile(
    r'([\d,]+(?:\.\d+)?)\s*(?:ozt\b|troy\s+o(?:z\b|unces?\b))',
    re.IGNORECASE,
)
_STERLING_RE = re.compile(
    r'(?:\.925|s925|925\s*sterling|sterling\s*silver|\bsterling\b)',
    re.IGNORECASE,
)
_PLATINUM_RE = re.compile(
    r'\b(?:pt950|pt900|950\s*platinum|platinum)\b', re.IGNORECASE,
)


# ---------------------------------------------------------------------
# Retail-anchored pricing (7/12). Liquidation auctions selling Amazon
# returns embed the retail price directly: A2Z prefixes titles
# ("$241 Aquastrong PB4-60 Pool Booster Pump"), others put
# "Retail Price: $252" in the description. Amazon-return goods clear
# on eBay at a fairly stable fraction of retail (~45-60% for boxed
# name-brand, less for no-name), so the retail figure prices the
# long tail of lots that eBay comps miss — at zero credits — and
# sanity-caps comps that drift ABOVE retail (nobody pays over retail
# for a customer return).
# ---------------------------------------------------------------------
_RETAIL_TITLE_RE = re.compile(r"^\s*\$(\d{1,5}(?:\.\d{2})?)\b(?!/)")
_RETAIL_DESC_RE = re.compile(
    r"(?:retail\s*(?:price|value)?|msrp)\s*[:\-]?\s*\$(\d{1,5}(?:\.\d{2})?)",
    re.IGNORECASE,
)


_AMAZON_URL_RE = re.compile(
    r"Retailer\s+Product\s+URL:\s*(https?://\S*amazon\.\S+)",
    re.IGNORECASE,
)
_AMAZON_ASIN_RE = re.compile(r"(?:/dp/|/gp/product/)([A-Z0-9]{10})")


def _extract_amazon_url(description: str):
    """Amazon product URL from a liquidation-lot description, or None.

    Prefers the explicit 'Retailer Product URL:' field; falls back to
    rebuilding a canonical /dp/ URL from any ASIN found in the text.
    """
    if not description:
        return None
    d = str(description)
    m = _AMAZON_URL_RE.search(d)
    if m:
        return m.group(1).strip().rstrip('.,)')
    m = _AMAZON_ASIN_RE.search(d)
    if m:
        return f"https://www.amazon.com/dp/{m.group(1)}"
    return None


def _extract_retail_price(title: str, description: str = ''):
    """Return the stated retail price (float) or None.

    Title form: leading "$241 …" (A2Z / liquidation-house format;
    the (?!/) guard rejects "$1/2- GoldBack" fraction titles).
    Description form: "Retail Price: $252" / "MSRP $89.99".
    Sanity bounds $5-$20,000 — below that it's probably a lot number
    or the GoldBack style, above it a typo.
    """
    for text, pat in ((title, _RETAIL_TITLE_RE),
                      (description, _RETAIL_DESC_RE)):
        if not text:
            continue
        m = pat.search(str(text))
        if m:
            try:
                v = float(m.group(1))
            except (TypeError, ValueError):
                continue
            if 5.0 <= v <= 20000.0:
                return v
    return None


def _extract_weight_grams(text: str):
    """Pull a weight in grams from text (gram or troy-oz notation)."""
    m = _WEIGHT_RE.search(text)
    if m:
        try:
            return float(m.group(1).replace(',', ''))
        except ValueError:
            pass
    m = _WEIGHT_OZT_RE.search(text)
    if m:
        try:
            return float(m.group(1).replace(',', '')) * 31.103
        except ValueError:
            pass
    return None


def _estimate_melt_value(title: str, description: str = ''):
    """Extract metal + weight from title/description → (usd, metal, grams).

    Returns (None, None, 0.0) when nothing usable is found. Scans the
    DESCRIPTION too — auctioneers frequently state exact weights there
    ("total tare weight is 2,045 grams") while the title stays generic.

    Used by the comp pipeline as a no-credit fast-path for precious-
    metal lots. Melt is a conservative floor; actual eBay clearing is
    typically 30-100% above melt for jewelry pieces (much closer to
    melt for flatware/scrap).
    """
    haystack = " ".join(filter(None, [title or '', description or '']))
    if not haystack.strip():
        return None, None, 0.0
    grams = _extract_weight_grams(haystack)
    if not grams or grams <= 0:
        return None, None, 0.0
    # Gold karat first (most specific)
    karat_m = _KARAT_RE.search(haystack)
    if karat_m:
        karat = f"{karat_m.group(1)}k"
        ppg = _SPOT_PRICES_USD_PER_GRAM.get(karat.lower())
        if ppg:
            return round(ppg * grams, 2), 'gold', grams
    if _STERLING_RE.search(haystack):
        return round(_SPOT_PRICES_USD_PER_GRAM['925'] * grams, 2), 'sterling', grams
    if _PLATINUM_RE.search(haystack):
        return round(_SPOT_PRICES_USD_PER_GRAM['platinum'] * grams, 2), 'platinum', grams
    return None, None, 0.0


def _dedup_identical_text_rows(df):
    """Pre-comp dedup of rows with byte-identical (title, description).

    Two lots in the same fetch with identical text MUST get the same
    comp result. Without this, separate eBay scrapes for the same query
    can return wildly different comp sets — one pulls outlier listings,
    the other pulls clean comparables — leading to identical pink
    moissanite rings being green-flagged at $1,022 resale and commodity-
    flagged at $111 in the same scan.

    Strategy: group rows by case-insensitive, whitespace-normalized
    (title, description). For each group of N>1, the first row is
    canonical; siblings are dropped from the eligible frame AND a
    clone map (sibling_lot_id -> canonical_lot_id) is stashed in
    `_comps_byte_dedup_map` for the post-comp clone step
    (``_apply_dedup_clones``) to consume.

    The byte-identical map is kept SEPARATE from the fingerprint+brand
    dedup map so the two passes don't clobber each other.

    Returns (df_with_siblings_dropped, n_siblings_deferred).
    """
    if 'lot_id' not in df.columns or 'title' not in df.columns:
        st.session_state._comps_byte_dedup_map = {}
        return df, 0

    def _norm(s):
        return ' '.join(str(s or '').lower().split())

    title_col = (
        'enriched_title' if 'enriched_title' in df.columns else 'title'
    )
    has_desc = 'description' in df.columns

    df = df.copy()
    # Compute normalized identity key per row. Tuples can't be passed
    # cleanly through groupby in some pandas versions, so we join with
    # a sentinel separator that never appears in cleaned text.
    sep = '\x1f'
    df['_idtext_key'] = df.apply(
        lambda r: _norm(r.get(title_col)) + sep + (
            _norm(r.get('description')) if has_desc else ''
        ),
        axis=1,
    )

    clones_map: dict = {}
    drop_indices: set = set()
    for key, grp in df.groupby('_idtext_key', sort=False):
        # Skip empty-title groups (key starts with sep when title blank).
        if not key or key.startswith(sep):
            continue
        if len(grp) < 2:
            continue
        canonical_idx = grp.index[0]
        canonical_lot_id = str(grp.loc[canonical_idx, 'lot_id'])
        for sibling_idx in grp.index[1:]:
            sibling_lot_id = str(grp.loc[sibling_idx, 'lot_id'])
            if sibling_lot_id == canonical_lot_id:
                continue
            clones_map[sibling_lot_id] = canonical_lot_id
            drop_indices.add(sibling_idx)

    if drop_indices:
        df = df[~df.index.isin(drop_indices)]
    df = df.drop(columns=['_idtext_key'], errors='ignore')

    # Always overwrite — each _apply_comps_filters call gets a fresh map.
    st.session_state._comps_byte_dedup_map = clones_map
    return df, len(clones_map)


def _apply_comps_filters(good_df):
    """Apply the Step 2 pre-comps filters to good_df.

    Returns (eligible_df, skipped_df, filter_summary) so the caller can
    run comps on just the eligible rows while preserving skipped rows
    (without resale data) in the final merged output.
    """
    df = good_df.copy()
    reasons = []

    # ---- Identical-text dedup (RUNS FIRST, before any cache merge) ----
    # Two lots in the same fetch with byte-identical (title, description)
    # MUST get the same comp result. Without this, separate eBay scrapes
    # for the same query can return wildly different comp sets (one pulls
    # outlier listings, the other pulls clean comparables) — leading to
    # one pink moissanite ring being green-flagged at $1,022 resale and
    # an identical sibling being flagged commodity at $111. Real bug
    # observed in 7 of 113 lots in a recent jewelry-heavy scan.
    #
    # This dedup is STRICTER than the fingerprint+brand dedup downstream:
    # it requires byte-identity, not just fingerprint match, AND it
    # works regardless of whether bolo_brand is set. When found, the
    # row with the MOST comp data (priority: most ebay_comps, then has
    # est_resale) becomes the canonical row; siblings clone its comp
    # data and get marked `price_source='clone:<rep_id>'`.
    if 'lot_id' in df.columns and 'title' in df.columns:
        df, n_id_dedup = _dedup_identical_text_rows(df)
        if n_id_dedup > 0:
            reasons.append(
                f"{n_id_dedup} byte-identical-title duplicates "
                f"normalized to canonical row (0 credits)"
            )

    # Free-only mode: route every NON-BOLO row through the PC-only
    # path (eBay/Mercari skipped, zero credits). BOLO matches DO get
    # full eBay/Mercari comps because those are exactly the lots
    # where the comp spend is worth it — the brand is on the watch
    # list specifically because the user wants real resale data on
    # them. Same per-row mechanism used for stylized lots.
    free_only = bool(st.session_state.get('_comps_free_only_mode', False))
    # Saturated-BOLO mode: when "every eligible lot is a BOLO match"
    # (multi-auction scan-all), the user can opt into truly-free
    # comps that skip eBay/Mercari for BOLO too. The flag is set at
    # the credit gate's Free button when _is_bolo_saturated.
    free_skip_bolo = bool(st.session_state.get('_comps_free_skip_bolo', False))
    if free_only:
        if free_skip_bolo:
            # Genuinely free: every row goes PriceCharting-only,
            # including BOLO matches. Used in BOLO-saturated runs
            # where running eBay on every BOLO hit would equal
            # a full paid run.
            df['_pc_only_stylized'] = True
            reasons.append(
                f"all {len(df)} → PriceCharting only "
                "(saturated BOLO free mode)"
            )
        else:
            # Compute BOLO matches inline. The comp pipeline runs before
            # the results-table render that normally adds bolo_brand,
            # so we have to do this here. Cheap regex pass.
            title_col_for_bolo = 'enriched_title' if 'enriched_title' in df.columns else 'title'
            desc_col_for_bolo = 'description' if 'description' in df.columns else None
            if title_col_for_bolo in df.columns and _BOLO_MATCHER.loaded:
                def _has_bolo(r):
                    t = str(r.get(title_col_for_bolo) or '')
                    d = str(r.get(desc_col_for_bolo) or '') if desc_col_for_bolo else ''
                    return _BOLO_MATCHER.match(t, d) is not None
                has_bolo_mask = df.apply(_has_bolo, axis=1)
                df['_pc_only_stylized'] = ~has_bolo_mask
                bolo_count = int(has_bolo_mask.sum())
                pc_only_count = int((~has_bolo_mask).sum())
                reasons.append(
                    f"free mode: {bolo_count} BOLO → full comps, "
                    f"{pc_only_count} → PriceCharting only"
                )
            else:
                # No BOLO available — every row goes PC-only
                df['_pc_only_stylized'] = True
                reasons.append(f"all {len(df)} → PriceCharting only (free mode)")

    # Stylized/replica handling — running eBay comps on a "Gucci style
    # hat" returns authentic-Gucci sold-comps that DO NOT apply to the
    # lot. Two policies based on `comps_pc_check_stylized`:
    #
    #   On (default):  KEEP stylized lots, mark them with _pc_only_stylized
    #                  = True so the comp lookup skips eBay/Mercari but
    #                  still runs PriceCharting (curated, free, no
    #                  contamination risk). PC will return None for lots
    #                  it doesn't cover (handbags, jewelry, etc.) which
    #                  is the correct outcome.
    #
    #   Off:           DROP stylized lots from eligible entirely — no
    #                  comp data of any kind is gathered for them.
    # Skip per-row stylized detection when free_only is already True
    # for everything — saves a regex pass and avoids overwriting the
    # all-True _pc_only_stylized column with a stylized-only mask.
    pc_check_stylized = bool(st.session_state.get('comps_pc_check_stylized', True))
    title_col_for_filter = 'enriched_title' if 'enriched_title' in df.columns else 'title'
    desc_col_for_filter = 'description' if 'description' in df.columns else None
    if not free_only and title_col_for_filter in df.columns:
        def _is_stylized(r):
            t = r.get(title_col_for_filter) or ''
            d = r.get(desc_col_for_filter) if desc_col_for_filter else ''
            return _detect_stylized_replica(str(t), str(d) if d else None) is not None
        stylized_mask = df.apply(_is_stylized, axis=1)
        n_stylized = int(stylized_mask.sum())
        if n_stylized:
            if pc_check_stylized:
                # Keep them in eligible but mark — the lookup phase
                # will route to PC-only for these rows.
                df['_pc_only_stylized'] = stylized_mask
                reasons.append(
                    f"{n_stylized} stylized/replica → PC-only "
                    "(eBay/Mercari skipped, free PC check)"
                )
            else:
                df = df[~stylized_mask]
                reasons.append(
                    f"{n_stylized} stylized/replica skipped entirely "
                    "(comps_pc_check_stylized off)"
                )

    # ---- Dropship-title lots → PC-only (no eBay credits) ----
    # AliExpress/Temu listings pasted verbatim into auction catalogs
    # ("Hot sale 925...", "New Original...", "...for Women Wedding
    # Party Jewelry Gifts"). Their tokens GENUINELY match mid-market
    # eBay solds — a "925 sterling silver ring" query returns $25-50
    # comps for what is a $3 zircon ring — so the relevance filter
    # can't catch them (LotPop 7/8: false A at score 97). Route to
    # PriceCharting-only, which returns nothing for fashion jewelry —
    # the honest outcome. The buy-score also caps these at C.
    if not free_only and title_col_for_filter in df.columns:
        def _is_ds(r):
            t = str(r.get(title_col_for_filter) or '')
            d = str(r.get(desc_col_for_filter) or '') if desc_col_for_filter else ''
            return _is_dropship_lot(t, d)
        ds_mask = df.apply(_is_ds, axis=1)
        n_ds = int(ds_mask.sum())
        if n_ds:
            if '_pc_only_stylized' in df.columns:
                df['_pc_only_stylized'] = (
                    df['_pc_only_stylized'].fillna(False).astype(bool) | ds_mask
                )
            else:
                df['_pc_only_stylized'] = ds_mask
            reasons.append(
                f"{n_ds} dropship-pattern lots → PC-only "
                "(eBay comps would return wrong market tier)"
            )

    # Apply HARD-logistics + easy-ship-only filters via the shared helper.
    # NOTE: The BOLO-scan stage now applies the same filter upstream,
    # so this pass typically catches the small remainder of lots that
    # didn't go through that path (single-auction analyses).
    df, _ship_reasons = _apply_easy_ship_filter(df)
    reasons.extend(_ship_reasons)


    # Skip comps when next-bid floor exceeds the user's threshold.
    # Same helper used upstream by the BOLO scan; this pass catches
    # any rows that didn't go through that path (single-auction view).
    df, _bid_reason = _apply_bid_cap_filter(df)
    if _bid_reason:
        reasons.append(_bid_reason)

    # ---- Unreachable pickup-only skip ----
    # Lots flagged pickup-only inside auctions that came from the
    # NATIONWIDE discovery query (i.e., outside the user's pickup
    # radius) AND whose auction offers no conditional shipping.
    # The auctioneer won't ship them and the user can't drive there —
    # comping them wastes credits pricing items that can never be
    # acquired. Conditional-shipping auctions ("contact us to
    # confirm") keep their lots: those ARE acquirable shipped, and
    # the buy-score prices the shipping in.
    if 'unreachable_pickup' in df.columns and len(df) > 0:
        _unreach_mask = df['unreachable_pickup'].fillna(False).astype(bool)
        if 'auction_cond_ship' in df.columns:
            _unreach_mask &= ~df['auction_cond_ship'].fillna(False).astype(bool)
        n_unreach = int(_unreach_mask.sum())
        if n_unreach:
            df = df[~_unreach_mask]
            reasons.append(
                f"{n_unreach} unreachable pickup-only lots skipped "
                "(auction outside pickup radius, no shipping option)"
            )

    # ---- Unknown-verdict skip (default on) ----
    # verdict == "Unknown" means the audit COULDN'T assess the lot
    # (API call failed, no text/image signal) — not that it passed.
    # Spending eBay credits pricing condition-unknown lots is usually
    # waste: an "untested / for parts" item hiding behind a failed
    # audit call comps identically to a working one. Bypass when it
    # would empty the run entirely (globally-dead audit is handled by
    # the pipeline gate upstream; if the user insists via the manual
    # button, let it through rather than silently doing nothing).
    if (bool(st.session_state.get('comps_skip_unknown_verdict', True))
            and 'verdict' in df.columns and len(df) > 0):
        _unk_mask = df['verdict'].fillna('') == 'Unknown'
        n_unk = int(_unk_mask.sum())
        if 0 < n_unk < len(df):
            df = df[~_unk_mask]
            reasons.append(
                f"{n_unk} Unknown-verdict lots skipped "
                "(audit couldn't assess condition)"
            )
        elif n_unk == len(df):
            reasons.append(
                f"all {n_unk} lots are Unknown-verdict — skip bypassed "
                "(would empty the run; fix the audit instead)"
            )

    # Top-N by bid
    max_lots = int(st.session_state.get('comps_max_lots', 0) or 0)
    if max_lots > 0 and len(df) > max_lots and 'current_bid' in df.columns:
        dropped = len(df) - max_lots
        df = df.sort_values('current_bid', ascending=False).head(max_lots)
        reasons.append(f"trimmed to top {max_lots} by bid ({dropped} cut)")

    # ---- Melt-value floor for precious metals ----
    # When enabled, lots whose title contains both a karat + weight
    # (e.g. "14K 5.2g pendant") get est_resale computed from melt
    # value × jewelry-premium-factor instead of running an eBay comp.
    # Skips ScrapingBee credits entirely on these — fast and free.
    # Default OFF because eBay comps are still more accurate for
    # signed maker jewelry (Cartier 14k vs generic 14k); use this
    # for liquidation / volume-melt plays.
    melt_enabled = bool(
        st.session_state.get('comps_use_melt_floor', False)
    )
    melt_premium = float(
        st.session_state.get('comps_melt_premium_factor', 1.4) or 1.4
    )
    melt_priced_indices = []
    if 'title' in df.columns and len(df) > 0:
        df = df.copy()
        # Make sure the destination columns exist before we set values
        for col in ('est_resale', 'price_low', 'price_high',
                    'comp_count', 'price_source'):
            if col not in df.columns:
                df[col] = None
        _melt_sterling_n = 0
        for idx in df.index:
            title = str(df.at[idx, 'title'] or '')
            desc = str(df.at[idx, 'description'] or '') \
                if 'description' in df.columns else ''
            melt, metal, grams = _estimate_melt_value(title, desc)
            if melt is None or melt <= 0:
                continue
            # Gate: sterling at scrap/flatware weight (≥100g) is
            # ALWAYS melt-priced — when the auctioneer states 2,045g
            # of sterling, melt IS the comp (Towle Chippendale 7/8:
            # nine bidders priced it to melt while the tool showed
            # nothing). Everything else (gold, platinum, small
            # sterling jewelry where signed pieces beat melt) stays
            # behind the comps_use_melt_floor toggle.
            sterling_scrap = (metal == 'sterling' and grams >= 100)
            if not (melt_enabled or sterling_scrap):
                continue
            # Flatware/scrap clears near melt; jewelry clears above it.
            factor = 1.0 if sterling_scrap else melt_premium
            est = round(melt * factor, 2)
            df.at[idx, 'est_resale'] = est
            df.at[idx, 'price_low'] = round(melt, 2)  # melt floor
            df.at[idx, 'price_high'] = round(melt * 2.0, 2)  # ceiling
            df.at[idx, 'comp_count'] = 1  # synthetic
            df.at[idx, 'price_source'] = f'melt ({metal} {grams:g}g)'
            melt_priced_indices.append(idx)
            if sterling_scrap:
                _melt_sterling_n += 1
        if melt_priced_indices:
            reasons.append(
                f"{len(melt_priced_indices)} precious-metal lots priced "
                f"from melt weight (0 credits"
                + (f"; {_melt_sterling_n} sterling-scrap always-on"
                   if _melt_sterling_n else "")
                + ")"
            )

    # ---- Title-fingerprint dedup ----
    # Group rows by (title_fingerprint, bolo_brand). For each group
    # with N>1, keep the highest-current-bid lot as the representative
    # (most likely to actually clear and become the comp anchor) and
    # mark the others as siblings. Siblings store a pointer to their
    # representative's lot_id; the post-comp step in `_run_ebay_comps`
    # clones the comp data from rep → siblings without paying
    # ScrapingBee for the duplicates. Requires title + bolo_brand
    # to be present; pure-title dedup risks cross-brand cross-pricing
    # (e.g. "vintage chair" matching across totally different brands).
    dedup_enabled = bool(
        st.session_state.get('comps_dedup_titles', True)
    )
    title_col_for_fp = 'enriched_title' if 'enriched_title' in df.columns else 'title'
    if (
        dedup_enabled
        and title_col_for_fp in df.columns
        and 'bolo_brand' in df.columns
        and 'lot_id' in df.columns
        and 'current_bid' in df.columns
    ):
        # Build fingerprint per row
        df = df.copy()
        df['_fp'] = df[title_col_for_fp].fillna('').astype(str).map(
            _title_fingerprint
        )
        # Only dedupe rows with non-empty fingerprint AND a brand
        # (so we know the comp would route the same way)
        eligible_for_dedup = df[
            (df['_fp'] != '') & df['bolo_brand'].notna()
        ]
        # Group by fp+brand
        clones_map: dict = {}  # sibling_lot_id -> representative_lot_id
        rep_indices = set()
        for (_fp, _brand), grp in eligible_for_dedup.groupby(
            ['_fp', 'bolo_brand'], sort=False,
        ):
            if len(grp) < 2:
                continue
            # Sort by current_bid desc; keep first as rep
            sorted_grp = grp.sort_values('current_bid', ascending=False)
            rep_idx = sorted_grp.index[0]
            rep_lot_id = str(sorted_grp.loc[rep_idx, 'lot_id'])
            rep_indices.add(rep_idx)
            for sibling_idx in sorted_grp.index[1:]:
                sibling_lot_id = str(sorted_grp.loc[sibling_idx, 'lot_id'])
                clones_map[sibling_lot_id] = rep_lot_id
        # Stash the clone-map in session state for the post-comp step
        st.session_state._comps_dedup_clone_map = clones_map
        if clones_map:
            # Drop siblings from eligible df
            sibling_indices = set()
            for sibling_lot_id in clones_map.keys():
                # Find sibling row indices in df by lot_id
                matches = df[df['lot_id'].astype(str) == sibling_lot_id].index
                sibling_indices.update(matches.tolist())
            df = df[~df.index.isin(sibling_indices)]
            reasons.append(
                f"{len(clones_map)} duplicate-title siblings "
                f"deferred (clone from rep after comp)"
            )
        # Drop the temporary _fp column before returning
        df = df.drop(columns=['_fp'], errors='ignore')
    else:
        st.session_state._comps_dedup_clone_map = {}

    # Pull melt-priced rows out of eligible — they already have a
    # synthetic price; the comp pipeline would overwrite that. Move
    # them to skipped (with their pre-filled est_resale intact) so
    # they appear in the final results without paying credits.
    melt_rows = pd.DataFrame()
    if melt_priced_indices:
        melt_rows = df.loc[melt_priced_indices].copy()
        df = df.drop(index=melt_priced_indices)

    eligible_ids = set(df.index)
    skipped = good_df[~good_df.index.isin(eligible_ids)].copy()
    # Merge melt-priced rows into skipped (with their pre-filled comp
    # data preserved). Use combine_first so the melt values take
    # precedence over the empty cells in skipped.
    if not melt_rows.empty:
        # Drop the same indices from skipped first to avoid duplicates
        skipped = skipped[~skipped.index.isin(melt_rows.index)]
        skipped = pd.concat([skipped, melt_rows], ignore_index=False)
    summary = " · ".join(reasons) if reasons else "all good+ items included"
    return df, skipped, summary


def _apply_dedup_clones(results_df):
    """After comps run, clone comp data from each representative lot
    to its dedup siblings. Reads the clone-map stashed in session
    state by `_apply_comps_filters`.

    Modifies `results_df` in place AND returns it for chaining.
    """
    # Two pre-comp passes can populate clone maps:
    #   1. byte-identical (title, description) dedup — ``_comps_byte_dedup_map``
    #   2. fingerprint+brand dedup           — ``_comps_dedup_clone_map``
    # Merge them; the fingerprint map wins on conflicts (it's the more
    # recent pass and its rep-pick logic is bid-aware).
    clone_map = dict(st.session_state.get('_comps_byte_dedup_map') or {})
    clone_map.update(st.session_state.get('_comps_dedup_clone_map') or {})
    if not clone_map or 'lot_id' not in results_df.columns:
        return results_df, 0
    # Build lookup: lot_id -> dict of comp columns
    comp_cols = (
        'est_resale', 'price_low', 'price_high', 'comp_count',
        'ebay_comps', 'mercari_comps', 'pricecharting_comps',
        'gocollect_comps', 'price_source', 'ebay_str', 'str_source',
    )
    available_cols = [c for c in comp_cols if c in results_df.columns]
    if not available_cols:
        return results_df, 0
    by_lot = {
        str(r.get('lot_id')): r
        for _, r in results_df.iterrows()
        if r.get('lot_id') is not None
    }
    cloned = 0
    for sibling_id, rep_id in clone_map.items():
        rep = by_lot.get(rep_id)
        if rep is None:
            continue
        # Only clone if rep actually has comp data
        if rep.get('est_resale') in (None, '') or (
            isinstance(rep.get('est_resale'), float)
            and rep['est_resale'] != rep['est_resale']
        ):
            continue
        # Find sibling rows in results_df by lot_id
        sibling_mask = results_df['lot_id'].astype(str) == sibling_id
        if not sibling_mask.any():
            continue
        for col in available_cols:
            results_df.loc[sibling_mask, col] = rep[col]
        # Mark the source so the user knows it was cloned
        if 'price_source' in results_df.columns:
            results_df.loc[sibling_mask, 'price_source'] = (
                f"clone:{rep_id}"
            )
        cloned += 1
    return results_df, cloned


# Title-specificity markers — when present, single-comp catalog matches
# (PriceCharting / GoCollect) are usually trustworthy because the title
# carries enough info to disambiguate. When absent, single-comp matches
# can drift catastrophically (see Pokemon dropship audit 5/3 — 5 lots
# matched the 1999 Base Set Booster Box catalog at $10,736 each).
_TITLE_SPECIFICITY_MARKERS = (
    re.compile(r"#\d+", re.IGNORECASE),                   # issue number
    re.compile(r"\b(?:CGC|PSA|BGS|SGC|CBCS|ANACS)\b", re.IGNORECASE),  # graders
    re.compile(r"\b1st\s+edition\b", re.IGNORECASE),
    re.compile(r"\b(?:base|jungle|fossil|gym|neo|EX|XY|"
               r"sun\s+&\s+moon|sword\s+&\s+shield|"
               r"scarlet\s+&\s+violet)\s+set\b", re.IGNORECASE),
    re.compile(r"\b(?:19|20)\d{2}\b"),                    # 4-digit year
    re.compile(r"\b(?:autograph(?:ed)?|signed|auto)\b", re.IGNORECASE),
    re.compile(r"\bsealed\b", re.IGNORECASE),
)


def _has_title_specificity(title: str) -> bool:
    """True when the title contains an identifier that disambiguates a
    single-comp catalog match (issue number, grading code, set name,
    year, autograph/sealed callout)."""
    if not title:
        return False
    return any(p.search(title) for p in _TITLE_SPECIFICITY_MARKERS)


def _parse_time_left_hours(s):
    """Convert HiBid `time_left` strings ('22h 16m', '1d 4h', 'Webcast')
    to total hours until close. Returns None when unparseable so the
    caller can treat it as "far from close" instead of triggering false
    market-signal-mismatch warnings."""
    if not s or not isinstance(s, str):
        return None
    sl = s.lower().strip()
    if not sl or 'closed' in sl or 'webcast' in sl:
        return None
    total = 0.0
    days = re.search(r'(\d+)\s*d', sl)
    hours = re.search(r'(\d+)\s*h', sl)
    minutes = re.search(r'(\d+)\s*m', sl)
    if days:
        total += int(days.group(1)) * 24
    if hours:
        total += int(hours.group(1))
    if minutes:
        total += int(minutes.group(1)) / 60.0
    return total if total > 0 else None


def _compute_manual_check_flags(df):
    """Tag every priced lot with reasons it might need a manual sanity-
    check.

    Returns the dataframe with two new columns:
      ``manual_check``         — bool, True if at least one flag fired
      ``manual_check_reasons`` — semicolon-joined string of human-readable
                                 reasons (so the user can see WHY it was
                                 flagged at a glance)

    Triggered by signals where the comp pipeline has structural
    uncertainty rather than a hard wrong answer:
      • Wide-spread comps (already marked in price_source) → variance
        threshold caught it; a human eye on the actual lot is the only
        way to know if the median lies high or is correct
      • Single-comp catalog match on a generic title → the lot's title
        wasn't specific enough to anchor the catalog match (see Pokemon
        dropship case)
      • High est_resale with no auctioneer estimate to cross-check
      • Bid-trap potential — someone's already bid on a lot we couldn't
        comp (they may know something we don't, OR may be wrong)
      • BOLO hit but no comp returned — manual eBay check warranted
      • Auctioneer uncertainty marker (`?` in title)
      • System estimate >3× the auctioneer's high estimate — we may be
        seeing variant contamination they're not

    Defensive on missing columns so it works on legacy cached frames.
    """
    if df is None or len(df) == 0:
        return df

    n = len(df)
    flags = [False] * n
    reasons = [[] for _ in range(n)]

    title = df['title'].fillna('').astype(str) if 'title' in df.columns else pd.Series([''] * n)
    est_resale = pd.to_numeric(df['est_resale'], errors='coerce') if 'est_resale' in df.columns else pd.Series([float('nan')] * n)
    comp_count = pd.to_numeric(df['comp_count'], errors='coerce') if 'comp_count' in df.columns else pd.Series([float('nan')] * n)
    price_source = df['price_source'].fillna('').astype(str) if 'price_source' in df.columns else pd.Series([''] * n)
    auctioneer_high = pd.to_numeric(df['auctioneer_est_high'], errors='coerce') if 'auctioneer_est_high' in df.columns else pd.Series([float('nan')] * n)
    current_bid = pd.to_numeric(df['current_bid'], errors='coerce') if 'current_bid' in df.columns else pd.Series([0] * n)
    bid_count = pd.to_numeric(df['bid_count'], errors='coerce') if 'bid_count' in df.columns else pd.Series([float('nan')] * n)
    time_left_str = df['time_left'].fillna('').astype(str) if 'time_left' in df.columns else pd.Series([''] * n)
    bolo_brand = df['bolo_brand'] if 'bolo_brand' in df.columns else pd.Series([None] * n)
    red_flag = df['red_flag'].fillna(False).astype(bool) if 'red_flag' in df.columns else pd.Series([False] * n)

    for i in range(n):
        # Skip red-flagged rows — they're already handled, no need to
        # double-flag for review.
        if red_flag.iloc[i]:
            continue

        ps = price_source.iloc[i]
        er = est_resale.iloc[i]
        cc = comp_count.iloc[i]
        cb = current_bid.iloc[i] if not pd.isna(current_bid.iloc[i]) else 0
        ah = auctioneer_high.iloc[i]
        t = title.iloc[i]
        is_clone = isinstance(ps, str) and ps.startswith("clone:")

        # 1. Wide-spread comp (already flagged by variance check)
        if 'wide-spread' in ps:
            if '(capped)' in ps:
                reasons[i].append("median capped from wide-spread comps")
            else:
                reasons[i].append("wide-spread comps (Q3/Q1 > 3×)")

        # 2. Single-comp catalog match with generic title — the
        # Pokemon-dropship failure mode. Specificity gate prevents
        # false positives on legitimately-titled comics.
        if (not is_clone and not pd.isna(cc) and cc == 1
                and not pd.isna(er) and er > 100
                and not _has_title_specificity(t)):
            reasons[i].append(
                f"single catalog comp ${er:.0f}+ on generic title"
            )

        # 3. High system estimate, no auctioneer cross-check
        if (not pd.isna(er) and er > 200 and pd.isna(ah)):
            reasons[i].append(
                f"${er:.0f} resale w/ no auctioneer estimate to anchor"
            )

        # 4. System estimate WAY above auctioneer estimate — we may be
        # seeing variant contamination they're not.
        if (not pd.isna(er) and not pd.isna(ah)
                and ah > 0 and er > 3.0 * ah):
            reasons[i].append(
                f"system ${er:.0f} >3× auctioneer's ${ah:.0f} estimate"
            )

        # 5. Bid trap — active bidding on uncomped lot
        if pd.isna(er) and cb >= 20 and not isinstance(bolo_brand.iloc[i], str):
            reasons[i].append(
                f"already ${cb:.0f} bid with no comp data"
            )

        # 6. BOLO match but no comp
        if (isinstance(bolo_brand.iloc[i], str)
                and bolo_brand.iloc[i].strip()
                and pd.isna(er)):
            reasons[i].append(
                f"BOLO ({bolo_brand.iloc[i][:30]}) but no comp returned"
            )

        # 7. Auctioneer uncertain — '?' in title is their tell
        if '?' in t:
            reasons[i].append("auctioneer uncertain ('?' in title)")

        # 8. Market-signal mismatch — high system resale + closing soon
        # + no bidders. The market is the ultimate signal: when a $200+
        # resale lot has near-zero bid 24h before close, smart money is
        # rejecting our estimate. Caught the 3.7Ct CZ "marquise diamond"
        # ring at $0 bid / $1,100 system estimate / 22h to close.
        bc = bid_count.iloc[i]
        tlh = _parse_time_left_hours(time_left_str.iloc[i])
        if (not pd.isna(er) and er > 200
                and cb < 5
                and (pd.isna(bc) or bc <= 1)
                and tlh is not None and tlh < 48):
            reasons[i].append(
                f"market-signal mismatch — ${er:.0f} resale, "
                f"${cb:.0f} bid, ~{tlh:.0f}h to close"
            )

        flags[i] = bool(reasons[i])

    df = df.copy()
    df['manual_check'] = flags
    df['manual_check_reasons'] = ['; '.join(r) for r in reasons]
    return df


def _compute_buy_score(df):
    """Compute a single 0-100 'Should I buy it?' score per lot.

    Combines all the per-lot signals into one number so the user can
    sort and filter on a single column instead of squinting at a
    dozen. Tunable weights at the top of the function — the four
    components must sum to 100 (max).

    Score components (when all signals are favorable):
      • margin       (0-45) — how much profit at max bid, after fees
      • bid_headroom (0-20) — room left to bid before someone else
                              pushes past max
      • str          (0-15) — sell-through rate (how fast it'll sell)
      • confidence   (0-20) — comp-quality penalties (wide spread,
                              generic-title single-comp, no auctioneer
                              anchor, etc.) deducted from a 20-point
                              starting allotment

    Hard zeros: red-flagged, no resale, max_bid <= 0, profit-at-max <= 0.

    Returns the dataframe with two columns added:
      buy_score (int 0-100) and buy_grade (string like "🟢 A").
    """
    if df is None or len(df) == 0:
        return df

    n = len(df)
    scores = [0] * n
    grades = [""] * n

    # Pull columns once — defensive on missing columns so legacy frames work
    er_arr = pd.to_numeric(df.get('est_resale', pd.Series([float('nan')] * n)), errors='coerce')
    mb_arr = pd.to_numeric(df.get('max_bid', pd.Series([float('nan')] * n)), errors='coerce')
    cb_arr = pd.to_numeric(df.get('current_bid', pd.Series([0] * n)), errors='coerce').fillna(0)
    str_arr = pd.to_numeric(df.get('ebay_str', pd.Series([float('nan')] * n)), errors='coerce')
    ps_arr = df.get('price_source', pd.Series([''] * n)).fillna('').astype(str)
    ah_arr = pd.to_numeric(df.get('auctioneer_est_high', pd.Series([float('nan')] * n)), errors='coerce')
    rf_arr = df.get('red_flag', pd.Series([False] * n)).fillna(False).astype(bool)
    bolo_arr = df.get('bolo_brand', pd.Series([None] * n))
    # Bid_count is the strongest market-validation signal we have on
    # uncomped lots — the antique-auction audit (5/9/26 Coin & Adv)
    # showed bidders pricing real value into lots the comp pipeline
    # couldn't touch.
    bid_count_arr = pd.to_numeric(df.get('bid_count', pd.Series([0] * n)), errors='coerce').fillna(0)
    # comp_count + manual_check drive hard grade caps below. A 1-2
    # comp sample is barely a signal, and manual_check=True literally
    # means "a human should look before trusting this" — neither is
    # compatible with an A grade ("high confidence, buy it").
    cc_arr = pd.to_numeric(df.get('comp_count', pd.Series([float('nan')] * n)), errors='coerce')
    mc_arr = df.get('manual_check', pd.Series([False] * n)).fillna(False).astype(bool)
    # Per-auction commercial terms from Phase 1 — actual buyer premium
    # (0-22% observed in the wild) and shipping-cost hint from the
    # auction's shippingAndPickupInfo. NaN → the BP_DEFAULT /
    # SHIP_DEFAULT constants below.
    bp_row_arr = pd.to_numeric(df.get('auction_buyer_premium_pct', pd.Series([float('nan')] * n)), errors='coerce')
    ship_hint_arr = pd.to_numeric(df.get('auction_ship_hint', pd.Series([float('nan')] * n)), errors='coerce')
    logistics_arr = df.get('logistics_ease', pd.Series([''] * n)).fillna('').astype(str)
    # Pickup-only lots in auctions OUTSIDE the pickup radius. Two
    # sub-cases, split by the auction's conditional-shipping terms:
    #   - cond_ship True ("contact us to confirm shipping"): the lot
    #     IS acquirable — but only by shipping. Grade with the ship
    #     cost added instead of the $0 pickup assumption.
    #   - cond_ship False: definitively unobtainable → hard F.
    unreachable_arr = df.get('unreachable_pickup', pd.Series([False] * n)).fillna(False).astype(bool)
    cond_ship_arr = df.get('auction_cond_ship', pd.Series([False] * n)).fillna(False).astype(bool)
    # Dropship-title detection (AliExpress listings pasted into auction
    # catalogs). Their comps match the wrong market tier — a $3 zircon
    # ring comps against $36 mid-market sterling solds — so the grade
    # is capped at C no matter how clean the comp band looks.
    _ds_titles = df.get('title', pd.Series([''] * n)).fillna('').astype(str)
    _ds_descs = df.get('description', pd.Series([''] * n)).fillna('').astype(str)
    dropship_arr = pd.Series(
        [_is_dropship_lot(t, d) for t, d in zip(_ds_titles, _ds_descs)],
        index=df.index,
    )
    # Pickup-only auctions have $0 ship cost. Without per-row ship,
    # the margin formula double-counts: max_bid was already computed
    # with $0 ship by _compute_max_bid, then we'd re-add $25. Caught
    # in Final Kemah audit 5/11 where Canali jacket at max $45 / resale
    # $179 scored only 75 (B) — should be 85+ (A) with $0 ship math.
    source_arr = df.get('source', pd.Series([''] * n)).fillna('').astype(str)

    # Approximate fee structure for margin calculation. Same constants
    # as `_compute_max_bid` so the score and the max-bid number are
    # consistent.
    EBAY_FEE_PCT = 0.1325
    EBAY_FEE_FLAT = 0.30
    BP_DEFAULT = 1.15
    # (Shipping constants removed 7/10 — shipping no longer factors
    # into grades; it's surfaced as a banner above the results.)
    # Per user preference: skip anything with realized profit below
    # this floor regardless of margin %. A 97% margin on a $5 boot
    # still nets $5; not worth the listing + shipping time.
    MIN_PROFIT_FLOOR = 20.0

    for i in range(n):
        if rf_arr.iloc[i]:
            scores[i] = 0
            grades[i] = "⚫ F"
            continue
        if unreachable_arr.iloc[i] and not cond_ship_arr.iloc[i]:
            # Pickup-only + auction outside pickup radius + no
            # conditional-shipping option = can't acquire at any
            # price. Hard F before any margin math — $0-ship
            # "profit" on these is pure fantasy. (When the auction
            # DOES offer confirm-with-us shipping, the lot survives
            # and gets the ship cost added below instead.)
            scores[i] = 0
            grades[i] = "⚫ F"
            continue

        er = er_arr.iloc[i]
        mb = mb_arr.iloc[i]
        cb = float(cb_arr.iloc[i] or 0)
        st_pct = str_arr.iloc[i]
        ps = ps_arr.iloc[i]
        ah = ah_arr.iloc[i]
        bolo = bolo_arr.iloc[i] if i < len(bolo_arr) else None

        # ---- No resale data at all — signal-based fallback ----
        # F is reserved for *comped and confirmed bad*; lots we can't
        # price get their own bucket so the user knows the difference
        # between "skip this" and "I can't tell — eyeball it."
        if pd.isna(er) or er <= 0:
            bc = float(bid_count_arr.iloc[i] or 0)
            has_bolo = isinstance(bolo, str) and bolo.strip() != ''
            has_auctioneer = (not pd.isna(ah)) and ah > 0
            signal_score = 0
            if has_bolo:
                signal_score += 15
            if has_auctioneer:
                signal_score += 10
            if bc >= 10:
                signal_score += 15
            elif bc >= 3:
                signal_score += 8
            elif bc >= 1:
                signal_score += 3
            if signal_score >= 10:
                # At least one strong signal OR a couple of weak ones —
                # tell the user "manual research candidate" instead of
                # lumping it with confirmed-bad F lots.
                scores[i] = min(40, signal_score)
                grades[i] = "🔍 ?"
            else:
                # No signals at all — pure unknown, sorts to the bottom.
                scores[i] = signal_score
                grades[i] = "❓ -"
            continue

        # ---- Resale exists; decide effective bid for grading ----
        # Bug fix 5/21: when est_resale exists but max_bid is NaN, the
        # OLD code lumped these into ❓- alongside "no info" lots, even
        # though the lot was perfectly gradeable against current_bid.
        # Hill Country audit had 91/163 (56%) in ❓- because the 3× ROI
        # default puts max_bid below zero on capped-wide-spread comps —
        # but a $30-resale lot at $5 current_bid still nets ~$10 profit,
        # which the algorithm should reflect, not hide.
        #
        # Also collapses the BOLO-but-already-overbid case (Coleman
        # lanterns at $50 cb vs $20 resale) into ⚫F automatically: if
        # the floor bid is unprofitable, no fallback grade is awarded.
        # Shipping is NOT part of the grade (user decision 7/10).
        # Every shipping assumption we tried ($25 bundle, $17 USPS,
        # $6 First-Class for EASY lots, auctioneer hints) whipsawed
        # grades on the same lot across runs. Grades now reflect pure
        # item economics — bid × premium vs. net resale — and the
        # shipping situation is displayed as a banner above the
        # results instead. Unreachable pickup-only lots still hard-F
        # above (that's acquirability, not cost).
        net_resale = er * (1 - EBAY_FEE_PCT) - EBAY_FEE_FLAT
        has_max_bid = (not pd.isna(mb)) and mb > 0
        if has_max_bid:
            # Grade against the HIGHER of your ROI ceiling and the
            # current bid. When current_bid is below max_bid you can
            # still buy profitably up to your ceiling → grade at the
            # ceiling. When the market has ALREADY bid past your
            # ceiling (current_bid > max_bid), you can only win by
            # overpaying — so grade at that reality, which makes the
            # margin (and grade) reflect the loss. Without this, lots
            # bid far past profitability graded A/B on a max_bid you
            # can no longer place: a 2014 Gold Eagle bid to $2,550
            # against a $465 resale graded B (7/23 scan, 58 of 117
            # graded lots were already priced out).
            effective_bid = max(float(mb), cb)
        else:
            # Floor case: target-ROI math didn't produce a positive
            # max_bid. Grade against the cheapest bid the user could
            # realistically place — current bid if active, else $1.
            effective_bid = max(cb, 1.0)

        _bp_row = bp_row_arr.iloc[i]
        _bp_mult = float(_bp_row) if (not pd.isna(_bp_row) and _bp_row > 0) else BP_DEFAULT
        cost_at_bid = effective_bid * _bp_mult
        if cost_at_bid <= 0 or net_resale <= 0:
            scores[i] = 0
            grades[i] = "⚫ F"
            continue
        margin_pct = (net_resale - cost_at_bid) / cost_at_bid
        if margin_pct <= 0:
            scores[i] = 0
            grades[i] = "⚫ F"
            continue
        # Absolute-profit floor — listing + shipping + handling time
        # makes <$20 profits not worth the user's bother regardless
        # of how good the margin % looks. The Cole Haan boots at $11
        # resale / $3 max bid were getting A grade (~95) by % margin
        # alone; nominal profit was only ~$6.
        profit_at_max = net_resale - cost_at_bid
        if profit_at_max < MIN_PROFIT_FLOOR:
            scores[i] = 0
            grades[i] = "⚫ F"
            continue
        margin_score = min(45.0, margin_pct * 22.5)
        # Discount the margin score when the comp itself is suspect.
        # Without this, a uncapped wide-spread comp with $1000 resale
        # would still produce 100% margin and dominate the score even
        # though the resale is potentially 5-10× inflated. Pink
        # Moissanite Ring case from May 5 jewelry-auction audit:
        # comp range $60-2000 with median $1023 — 87/A score without
        # the discount, ~60/C with it.
        if 'generic-title single-comp' in ps:
            # Already capped to low; don't double-discount.
            pass
        elif 'wide-spread (NOS floor)' in ps:
            # NOS exception — the spread came from mixing new+used
            # comps, our lot is new, no margin penalty needed.
            pass
        elif 'wide-spread (capped)' in ps:
            margin_score *= 0.75
        elif 'wide-spread' in ps:
            margin_score *= 0.5

        # ---- 2. Bid headroom (0-20) ----
        # Fresh auctions ($0 bid) → full points; bid at max → 0 points;
        # halfway → 10 points. Linear in current_bid / max_bid.
        # When max_bid is missing (5/21 fallback path), give half-credit:
        # we DO know the lot is profitable at the floor, just not whether
        # the user has bid-room left at their target-ROI ceiling.
        if not has_max_bid:
            bid_score = 10.0
        elif cb >= mb:
            bid_score = 0.0
        elif mb > 0:
            ratio = cb / mb
            bid_score = max(0.0, 20.0 * (1.0 - ratio))
        else:
            bid_score = 0.0

        # ---- 3. STR score (0-15) ----
        # 60% sell-through (typical-strong) → 15 points; linear below.
        if pd.isna(st_pct) or st_pct < 0:
            str_score = 7.5  # neutral default when STR data missing
        else:
            str_score = min(15.0, st_pct * 0.25)

        # ---- 4. Confidence (0-20, starts at 20, subtract penalties) ----
        confidence = 20.0
        if 'wide-spread (NOS floor)' in ps:
            # NOS-detected wide-spread is by design, not a comp problem
            confidence -= 2
        elif 'wide-spread (capped)' in ps:
            confidence -= 5
        elif 'wide-spread' in ps:
            confidence -= 10
        if 'generic-title single-comp' in ps:
            confidence -= 12
        if 'short query' in ps:
            confidence -= 3
        if pd.isna(ah):
            confidence -= 3
        elif ah > 0 and er > 3.0 * ah:
            confidence -= 5
        # Modest BOLO bonus when a brand match was found — gives a small
        # nudge for items that hit the watch list (better-targeted
        # comps tend to be cleaner).
        if isinstance(bolo, str) and bolo.strip():
            confidence += 3
        # Grading against current_bid (no max_bid anchor) is less certain
        # than grading against a target-ROI ceiling — knock confidence
        # down so these lots stay distinguishable from full A grades.
        if not has_max_bid:
            confidence -= 5
        confidence = max(0.0, min(20.0, confidence))

        total = margin_score + bid_score + str_score + confidence
        score = int(round(min(100.0, max(0.0, total))))

        # ---- Hard grade caps (trust gates) ----
        # An A grade means "high confidence, buy it" — these gates
        # keep thin or suspect evidence out of that bucket no matter
        # how good the margin math looks.
        #
        # Cap 1: fewer than 3 scraped comps caps at B (79). One or two
        # eBay solds are barely a sample; noise dominates. Curated
        # sources (PriceCharting, GoCollect) are exempt — a single
        # catalog price is authoritative in a way one scraped sold
        # listing isn't.
        # Cap 2: manual_check=True caps at B. The flag literally says
        # "human review recommended" — contradictory with an auto-A.
        # (The Notebook Paper & Cards false A from the Longview audit
        # motivated both: 21 wrong-product comps in a tight band beat
        # every soft confidence penalty.)
        _cc = cc_arr.iloc[i]
        _ps_lower = ps.lower()
        _curated = ('pricecharting' in _ps_lower or 'gocollect' in _ps_lower
                    # retail-anchor's synthetic comp_count=1 shouldn't
                    # trip the thin-sample C cap — its own B cap below
                    # is the intended ceiling (stated retail is exact;
                    # only the resale factor is an estimate).
                    or 'retail-anchor' in _ps_lower
                    or 'amazon-live' in _ps_lower)
        # < 5 scraped comps can't earn an A (was < 3 — a 3-comp
        # tight-band of quantity-mismatched bulk listings A-graded a
        # $30 diecast lot at $150 on 7/12; 3 wrong comps agreeing is
        # still 3 wrong comps). 3-4 comps cap at B; 1-2 cap at C.
        if (not _curated) and (not pd.isna(_cc)):
            if _cc < 3:
                score = min(score, 59)
            elif _cc < 5:
                score = min(score, 79)
        if mc_arr.iloc[i]:
            score = min(score, 79)
        # Retail-anchored prices are a factor-of-retail ESTIMATE, not
        # market evidence — solid enough for triage, not for an A.
        if 'retail-anchor' in _ps_lower or 'amazon-live' in _ps_lower:
            score = min(score, 79)
        # Cap 3: dropship-pattern title caps at C (59). The comps are
        # for a different market tier than the lot — no comp count or
        # band tightness redeems that (LotPop 7/8: "New Original 925
        # Sterling Silver Rings" scored 97/A against mid-market comps
        # for what is a $3 AliExpress zircon ring).
        if dropship_arr.iloc[i]:
            score = min(score, 59)

        scores[i] = score

        if score >= 80:
            grades[i] = "🟢 A"
        elif score >= 65:
            grades[i] = "🟡 B"
        elif score >= 50:
            grades[i] = "🟠 C"
        elif score >= 35:
            grades[i] = "🔴 D"
        else:
            grades[i] = "⚫ F"

    df = df.copy()
    df['buy_score'] = scores
    df['buy_grade'] = grades
    return df


def _run_ebay_comps(results_df):
    """Run eBay + Mercari price comps on filtered good+ items in results_df.

    Applies the Step 2 pre-comps filters (min bid, logistics, top-N, etc.),
    then:
      1. Samples STR per auction (~3 lots each) — fast, replaces per-lot scrape.
      2. Runs price comps on every eligible lot with the auction-level STR.

    Max bid is NOT computed here — it's recomputed on every render so the
    Target ROI slider in the results section updates it live.
    """
    # Clear previous comp data so re-runs start fresh
    for col in ['est_resale', 'price_low', 'price_high', 'comp_count',
                'ebay_comps', 'mercari_comps', 'pricecharting_comps',
                'price_source', 'ebay_str', 'str_source',
                'est_roi', 'max_bid']:
        if col in results_df.columns:
            results_df = results_df.drop(columns=[col])

    # Coerce red_flag to bool — cache round-trips can land it as int.
    # ~int gives bitwise inverse (-1, -2) which pandas treats as
    # label-based selection and crashes. Same fix applied at the
    # comps-tab pre-filter (line ~9689).
    _rf = results_df['red_flag'].fillna(False).astype(bool)
    good_df = results_df[~_rf].copy()
    flagged_df = results_df[_rf].copy()

    # Apply pre-comps filters — skipped rows come back with no resale data
    eligible_df, skipped_df, filter_summary = _apply_comps_filters(good_df)

    from scraper.ebay_prices import EbayPriceLookup
    from scraper.pricecharting import PriceChartingLookup
    from scraper.config_loader import load_config
    cfg = load_config()
    pc_token = (cfg.get("pricecharting") or {}).get("token") or None
    pc_client = PriceChartingLookup(pc_token)
    sb_key = (cfg.get("scrapingbee") or {}).get("api_key") or None
    ebay = EbayPriceLookup(
        cfg["ebay"]["app_id"], cfg["ebay"]["cert_id"],
        pricecharting=pc_client,
        scrapingbee_key=sb_key,
        mercari_enabled=bool(st.session_state.get('comps_use_mercari', False)),
    )

    total = len(eligible_df)

    with st.status("💰 Price Comps & Sell-Through Rate…", expanded=True) as status:
        if total == 0:
            st.warning(
                "No items matched the pre-comps filters. Loosen the filters above and try again."
            )
            status.update(label="⚠️ Nothing to comp", state="error", expanded=True)
            combined = pd.concat([good_df, flagged_df], ignore_index=True)
            return combined, 0, 0

        st.write(
            f"**🎯 Comp target:** {total} lots "
            f"({len(skipped_df)} of {len(good_df)} good+ lots skipped by filters)."
        )
        st.caption(f"Filters applied — {filter_summary}")

        # ---------- STR: per-category sampling (if enabled) ----------
        auction_str_map = None
        use_auction_str = st.session_state.get('comps_use_auction_str', True)
        if use_auction_str and 'auction' in eligible_df.columns:
            # Count distinct (auction, category) buckets so the user knows
            # how many STR scrapes we're doing vs per-lot
            if 'category' in eligible_df.columns:
                buckets = eligible_df.groupby(
                    ['auction', eligible_df['category'].fillna('').replace('', '(uncategorized)')]
                ).ngroups
            else:
                buckets = eligible_df['auction'].nunique()
            st.write(
                f"**📈 Sampling STR across {buckets} category bucket(s)** "
                "(2 lots each — gives per-category variance without per-lot cost)."
            )
            str_progress = st.progress(0, text=f"Sampling STR — 0/{buckets}")

            def str_progress_cb(current, total_buckets):
                pct = current / total_buckets if total_buckets > 0 else 1.0
                str_progress.progress(
                    min(pct, 1.0),
                    text=f"Sampled STR for {current}/{total_buckets} bucket(s)",
                )

            auction_str_map = ebay.sample_auction_str(
                eligible_df, sample_size=2,
                progress_callback=str_progress_cb,
                granularity="category",
            )
            usable = sum(
                1 for k, v in auction_str_map.items()
                if k != "__granularity__" and v and v[0] is not None
            )
            st.caption(f"✓ STR resolved for {usable}/{buckets} bucket(s).")

        # ---------- Price comps ----------
        st.write(
            f"**🔗 Looking up eBay sold listings + Mercari sold listings** "
            f"for {total} lots."
        )
        st.caption(
            "Per lot: scrape recent sold prices from both marketplaces, "
            "apply IQR outlier filtering, pool into median / 25th / 75th percentile."
        )
        progress_bar = st.progress(0, text=f"Starting — 0/{total}")
        current_item_placeholder = st.empty()
        workers = int(st.session_state.get('comps_workers', 8))
        if workers > 1:
            st.caption(f"⚡ Running {workers} parallel workers.")

        def price_progress(current, total_items, title_preview=""):
            # Called from the main thread (as_completed drains on-thread).
            pct = current / total_items if total_items > 0 else 1.0
            progress_bar.progress(
                min(pct, 1.0),
                text=f"Priced {current}/{total_items}…",
            )
            if title_preview:
                current_item_placeholder.caption(
                    f"🔎 Just priced: *{title_preview}*"
                )

        comps_df = ebay.batch_lookup(
            eligible_df,
            progress_callback=price_progress,
            auction_str_map=auction_str_map,
            max_workers=workers,
        )

        # ROI = (resale - cost) / cost. est_cost is computed from
        # max(current_bid, next_bid) in pass1, so it's > 0 whenever the
        # auction has a starting bid (= virtually always). Rows with
        # cost==0 fall through to None — the table sorts them last.
        comps_df['est_roi'] = None
        resale_num = pd.to_numeric(comps_df['est_resale'], errors='coerce')
        cost_num = pd.to_numeric(comps_df['est_cost'], errors='coerce')
        normal = resale_num.notna() & cost_num.gt(0)
        comps_df.loc[normal, 'est_roi'] = (
            (resale_num[normal] - cost_num[normal]) / cost_num[normal] * 100
        ).round(0)

        found = int(comps_df['est_resale'].notna().sum())
        st.write(f"**📊 Summary:** found price comps for {found}/{total} items.")
        status.update(label="✅ Price comps complete", state="complete", expanded=False)

    # Stitch comps_df (with resale) back together with skipped + flagged rows
    # so the results table still shows every lot, just with NaN resale for
    # skipped ones.
    combined = pd.concat([comps_df, skipped_df, flagged_df], ignore_index=True)

    # Post-comp dedup-clone step: copy comp data from representative
    # lots to their duplicate-title siblings (which were skipped from
    # the comp run to save credits).
    combined, n_cloned = _apply_dedup_clones(combined)
    if n_cloned > 0:
        st.toast(
            f"💎 Cloned comp data to {n_cloned} duplicate-title sibling(s) "
            f"— saved ~{n_cloned * _CREDITS_PER_NON_PC_LOT} credits",
            icon="💎",
        )

    # Tag low-confidence rows for manual review. Free pass — purely
    # computed from columns we already have.
    combined = _compute_manual_check_flags(combined)
    n_manual = int(combined.get('manual_check', pd.Series(False)).sum())
    if n_manual > 0:
        tlog("COMPS", f"flagged {n_manual} lots for manual review")

    # NOTE: buy_score is intentionally NOT computed here. It depends on
    # max_bid which is computed later in _render_results_table via
    # _compute_max_bid. Computing buy_score before max_bid would route
    # every comped lot through the "no comp" gate (since max_bid is
    # still NaN at this point) — see Final Kemah audit 5/11 where 207
    # max_bid > 0 lots all graded ❓ - because of this ordering bug.
    # buy_score now runs in the render path after max_bid is populated.

    return combined, found, total


# Comp columns that batch_lookup adds — used by the chunked variant to
# initialize empty placeholders and merge per-chunk results back into the
# accumulating audit_results DataFrame.
_COMP_COLUMNS = (
    'est_resale', 'price_low', 'price_high', 'comp_count',
    'ebay_comps', 'mercari_comps', 'pricecharting_comps', 'gocollect_comps',
    'price_source', 'ebay_str', 'str_source',
)


def _run_ebay_comps_chunk(audit_df, chunk_size: int = 200, on_lot_priced=None):
    """Run price comps on the NEXT batch of eligible-but-uncomped lots.

    Lets the user review earlier results while later chunks process.
    Each call:
      1. Identifies rows that are good (not red-flagged), pass the
         pre-comps filters, AND don't yet have an est_resale value.
      2. Runs comps on the first chunk_size of those.
      3. Merges the chunk's comp columns back into the input df at
         the original row indices.

    Args:
        audit_df: current full audit_results DataFrame.
        chunk_size: max number of pending lots to comp this call.
        on_lot_priced: optional callable(completed, total, last_lot)
            that fires after EACH individual lot finishes pricing.
            `last_lot` is a small dict with keys 'title', 'resale',
            'roi', 'ebay_comps', 'mercari_comps' so the caller can
            render a per-lot ticker without re-scanning the df.
            Lets the caller stream live updates into the UI without
            waiting for the whole chunk to finish. The function is
            also responsible for any UI refresh; this scope only
            updates session_state.

    Returns:
        (updated_df, found_in_chunk, processed_in_chunk, has_more)

    `has_more` is True when more eligible-uncomped rows remain after
    this chunk — the caller surfaces a "Continue" button when so.
    """
    df = audit_df.copy().reset_index(drop=True)

    # Initialize comp columns on first run so the .isna() check below
    # can find rows that haven't been processed yet.
    for col in _COMP_COLUMNS:
        if col not in df.columns:
            df[col] = None
    if 'est_roi' not in df.columns:
        df['est_roi'] = None

    # ---- ScrapingBee spend ledger (cross-session, lot-level) ----
    # Restore comp data for lots we've ALREADY paid to price (any
    # session, any auction view) and block re-attempts on lots that
    # already came back empty within the TTL. Rebuilt 7/11 so the
    # same lot never spends credits twice.
    _ledger_filled = 0
    _ledger_blocked: set = set()
    if st.session_state.get('use_spend_ledger', True):
        try:
            from scraper import comped_lots as _spend_ledger
            _ttl = float(
                st.session_state.get('spend_ledger_ttl_days', 7.0) or 7.0
            )
            df, _ledger_filled, _ledger_blocked = (
                _spend_ledger.overlay_onto_df(df, ttl_days=_ttl)
            )
            if _ledger_filled or _ledger_blocked:
                tlog("LEDGER",
                     f"restored {_ledger_filled} priced lots ·",
                     f"blocked {len(_ledger_blocked)} known-empty",
                     f"re-attempts (0 credits)")
            st.session_state._last_ledger_filled = _ledger_filled
            st.session_state._last_ledger_blocked = len(_ledger_blocked)
        except Exception as _le:
            tlog("LEDGER", f"overlay failed (non-fatal): {_le}")

    # "Pending" = good (not red-flagged) AND no est_resale yet AND not
    # a known-empty ledger entry within TTL.
    not_red = (
        ~df['red_flag'].fillna(False).astype(bool)
        if 'red_flag' in df.columns
        else pd.Series(True, index=df.index)
    )
    not_processed = df['est_resale'].isna()
    if _ledger_blocked and 'lot_id' in df.columns:
        not_blocked = ~df['lot_id'].astype(str).isin(_ledger_blocked)
    else:
        not_blocked = pd.Series(True, index=df.index)
    pending_df = df[not_red & not_processed & not_blocked]

    eligible_df, _skipped_df, filter_summary = _apply_comps_filters(pending_df)

    # Melt-priced rows come back in the SKIPPED frame with synthetic
    # comp values pre-filled (est_resale from metal weight, 0 credits).
    # The chunk merge below only covers rows that went through the
    # eBay lookup, so without this copy the melt prices were silently
    # dropped — a dormant bug the whole time the melt floor existed
    # (default-off toggle meant nobody noticed).
    melt_found = 0
    if (isinstance(_skipped_df, pd.DataFrame) and not _skipped_df.empty
            and 'price_source' in _skipped_df.columns):
        _melt_mask = (
            _skipped_df['price_source'].fillna('').astype(str)
            .str.startswith('melt')
        )
        for _midx in _skipped_df.index[_melt_mask]:
            if _midx not in df.index:
                continue
            for _mcol in ('est_resale', 'price_low', 'price_high',
                          'comp_count', 'price_source'):
                if _mcol in _skipped_df.columns and _mcol in df.columns:
                    df.at[_midx, _mcol] = _skipped_df.at[_midx, _mcol]
            melt_found += 1

    total_pending = len(eligible_df)
    if total_pending == 0:
        return df, melt_found, melt_found, False

    chunk = eligible_df.head(chunk_size).copy()
    chunk_indices = chunk.index.tolist()  # original positions in df

    from scraper.ebay_prices import EbayPriceLookup
    from scraper.pricecharting import PriceChartingLookup
    from scraper.config_loader import load_config
    cfg = load_config()
    pc_token = (cfg.get("pricecharting") or {}).get("token") or None
    sb_key = (cfg.get("scrapingbee") or {}).get("api_key") or None
    ebay = EbayPriceLookup(
        cfg["ebay"]["app_id"], cfg["ebay"]["cert_id"],
        pricecharting=PriceChartingLookup(pc_token),
        scrapingbee_key=sb_key,
    )

    label = f"💰 Comping next {len(chunk)} of {total_pending} pending lot(s)…"
    with st.status(label, expanded=True) as status:
        st.caption(f"Filters applied — {filter_summary}")

        # STR sampling — cache the auction-level map across chunks so
        # we don't re-scrape it every batch. Only sample once per fetch.
        auction_str_map = st.session_state.get('_comps_auction_str_map')
        if (
            auction_str_map is None
            and st.session_state.get('comps_use_auction_str', True)
            and 'auction' in chunk.columns
        ):
            # Sample over the FULL eligible pending pool (not just chunk)
            # so the per-category buckets reflect the whole auction.
            all_eligible = pd.concat([eligible_df], ignore_index=False)
            buckets = (
                all_eligible.groupby([
                    'auction',
                    all_eligible['category'].fillna('').replace('', '(uncategorized)')
                ]).ngroups
                if 'category' in all_eligible.columns
                else all_eligible['auction'].nunique()
            )
            st.write(f"📈 Sampling STR across {buckets} category bucket(s) (one-time per fetch).")
            str_progress = st.progress(0)
            def str_cb(current, total_buckets):
                pct = current / total_buckets if total_buckets > 0 else 1.0
                str_progress.progress(min(pct, 1.0),
                                      text=f"Sampled STR for {current}/{total_buckets}…")
            auction_str_map = ebay.sample_auction_str(
                all_eligible, sample_size=2,
                progress_callback=str_cb, granularity="category",
            )
            st.session_state._comps_auction_str_map = auction_str_map

        st.write(f"🔗 Looking up sold prices for {len(chunk)} lots in this batch.")
        progress_bar = st.progress(0)
        current_item = st.empty()

        def price_cb(current, total_items, title_preview=""):
            pct = current / total_items if total_items > 0 else 1.0
            progress_bar.progress(min(pct, 1.0),
                                  text=f"Priced {current}/{total_items}…")
            if title_preview:
                current_item.caption(f"🔎 Just priced: *{title_preview}*")

        # The live callback receives the raw `lookup_price_range()` dict
        # whose keys differ from our DataFrame column names. Translate.
        _PAYLOAD_KEY_MAP = {
            'median': 'est_resale',
            'low': 'price_low',
            'high': 'price_high',
            'count': 'comp_count',
            'ebay_count': 'ebay_comps',
            'mercari_count': 'mercari_comps',
            'pricecharting_count': 'pricecharting_comps',
            'source': 'price_source',
        }

        # Per-lot live callback. Fires from the main thread (drained via
        # as_completed) so it's safe to touch Streamlit state. Merges the
        # individual lot's comp result into df + session_state on the
        # spot, then defers to the caller's UI-refresh hook.
        def _live_cb(chunk_idx, payload, completed, total_items):
            try:
                target_idx = chunk_indices[chunk_idx]
                for src_key, col in _PAYLOAD_KEY_MAP.items():
                    if src_key in payload:
                        df.at[target_idx, col] = payload[src_key]
                # ROI on this row only — cheap. Full ROI recompute happens
                # at the bottom of this function after the chunk completes.
                # est_cost is built from max(current_bid, next_bid) in pass1
                # so it's > 0 whenever the auction has a starting bid; rows
                # where cost==0 leave est_roi as None.
                cost = pd.to_numeric(df.at[target_idx, 'est_cost'], errors='coerce')
                resale = pd.to_numeric(df.at[target_idx, 'est_resale'], errors='coerce')
                if pd.notna(resale) and pd.notna(cost) and cost > 0:
                    df.at[target_idx, 'est_roi'] = round(
                        (resale - cost) / cost * 100, 0
                    )
                st.session_state.audit_results = df
                if on_lot_priced is not None:
                    # Pull a small "last lot" payload so the caller can
                    # render a per-lot ticker without re-scanning the df.
                    row = df.iloc[target_idx]
                    last_lot = {
                        'title': (
                            row.get('enriched_title')
                            or row.get('title')
                            or ''
                        ),
                        'resale': row.get('est_resale'),
                        'roi': row.get('est_roi'),
                        'ebay_comps': row.get('ebay_comps') or 0,
                        'mercari_comps': row.get('mercari_comps') or 0,
                    }
                    on_lot_priced(completed, total_items, last_lot)
            except Exception:
                # Live updates are best-effort — never fail the comp run
                # because of a UI hiccup.
                pass

        chunk_with_comps = ebay.batch_lookup(
            chunk,
            progress_callback=price_cb,
            auction_str_map=auction_str_map,
            max_workers=int(st.session_state.get('comps_workers', 8)),
            live_callback=_live_cb,
        )

        # Merge comp columns from chunk back into df at the original
        # row indices. .values bypasses index alignment (chunk_with_comps
        # was reset to 0..n-1 inside batch_lookup).
        for col in _COMP_COLUMNS:
            if col in chunk_with_comps.columns:
                df.loc[chunk_indices, col] = chunk_with_comps[col].values

        # ---- Retail-anchored pricing (Amazon-return auctions) ----
        # Runs AFTER the merge so real sold-comps always win. Two
        # moves, both keyed on the retail price stated in the lot
        # itself ("$241 …" title prefix / "Retail Price: $252" desc):
        #   1. FALLBACK: comp-missed lots get retail × factor as
        #      est_resale (zero credits — this prices the long tail).
        #   2. CAP: comp medians ABOVE retail get clamped to retail —
        #      nobody pays over retail for a customer return.
        if st.session_state.get('use_retail_anchor', True):
            _ra_factor = float(
                st.session_state.get('retail_anchor_factor', 0.5) or 0.5
            )
            # Per-chunk live-lookup budget (mutable for closure use)
            _amz_budget = [int(
                st.session_state.get('amazon_live_max_lookups', 20) or 20
            )]
            _n_anchored = 0
            _n_capped = 0
            for _ri in chunk_indices:
                _retail = _extract_retail_price(
                    str(df.at[_ri, 'title'] or ''),
                    str(df.at[_ri, 'description'] or '')
                    if 'description' in df.columns else '',
                )
                # Live Amazon price upgrade for high-value lots:
                # the stated retail can be stale/inflated; the live
                # buy-box price is truth. Budget-capped per chunk and
                # gated on value so credits go where margin lives.
                _live_used = False
                if (st.session_state.get('use_amazon_live', True)
                        and _amz_budget[0] > 0
                        and (_retail or 0) >= float(
                            st.session_state.get(
                                'amazon_live_min_retail', 100.0) or 100.0)):
                    _amz_url = _extract_amazon_url(
                        str(df.at[_ri, 'description'] or '')
                        if 'description' in df.columns else ''
                    )
                    if _amz_url:
                        _amz_budget[0] -= 1
                        _live = ebay.fetch_amazon_price(_amz_url)
                        if _live and _live > 0:
                            _retail = _live
                            _live_used = True
                if not _retail:
                    continue
                df.at[_ri, 'retail_price'] = _retail
                _er_val = df.at[_ri, 'est_resale']
                try:
                    _er_missing = pd.isna(_er_val)
                except (TypeError, ValueError):
                    _er_missing = _er_val is None
                if _er_missing:
                    df.at[_ri, 'est_resale'] = round(_retail * _ra_factor, 2)
                    df.at[_ri, 'price_low'] = round(_retail * 0.35, 2)
                    df.at[_ri, 'price_high'] = round(_retail * 0.7, 2)
                    df.at[_ri, 'comp_count'] = 1
                    df.at[_ri, 'price_source'] = (
                        f"{'amazon-live' if _live_used else 'retail-anchor'}"
                        f" ({int(_ra_factor * 100)}% of ${_retail:g})"
                    )
                    _n_anchored += 1
                else:
                    try:
                        if float(_er_val) > _retail * 1.15:
                            df.at[_ri, 'est_resale'] = round(_retail, 2)
                            df.at[_ri, 'price_source'] = (
                                str(df.at[_ri, 'price_source'] or '')
                                + f" ⚠ capped@retail ${_retail:g}"
                            )
                            _n_capped += 1
                    except (TypeError, ValueError):
                        pass
            if _n_anchored or _n_capped:
                tlog("RETAIL",
                     f"anchored {_n_anchored} comp-missed lots ·",
                     f"capped {_n_capped} over-retail comps")

        # Ledger every row that just went through the PAID lookup —
        # priced or empty, it consumed ScrapingBee credits and must
        # never be paid for again within the TTL.
        if st.session_state.get('use_spend_ledger', True):
            try:
                from scraper import comped_lots as _spend_ledger
                _n_led = _spend_ledger.record_from_df(
                    df.loc[chunk_indices]
                )
                tlog("LEDGER", f"recorded {_n_led} spend events")
            except Exception as _le:
                tlog("LEDGER", f"record failed (non-fatal): {_le}")

        # Recompute ROI on every row that now has resale + cost. With
        # est_cost = max(current_bid, next_bid) + premium from pass1,
        # cost > 0 unless the auction omitted a starting bid (rare).
        cost_col = pd.to_numeric(df.get('est_cost'), errors='coerce')
        resale_col = pd.to_numeric(df.get('est_resale'), errors='coerce')
        has_data = resale_col.notna() & cost_col.gt(0)
        df.loc[has_data, 'est_roi'] = (
            (resale_col[has_data] - cost_col[has_data]) / cost_col[has_data] * 100
        ).round(0)

        found = int(chunk_with_comps['est_resale'].notna().sum())
        has_more = total_pending > len(chunk)

        # On the FINAL chunk, recompute manual-check flags across the
        # whole frame (not just the chunk) so the user sees a complete
        # review list once comps are done. Skipping per-chunk to keep
        # the flagging consistent — we only know the final state of
        # cloned siblings + dedup after all chunks merge.
        if not has_more:
            df = _compute_manual_check_flags(df)
            n_manual = int(df.get('manual_check', pd.Series(False)).sum())
            if n_manual > 0:
                tlog("COMPS",
                     f"flagged {n_manual} lots for manual review")
            # buy_score runs in render after max_bid is computed; see
            # the note in _run_ebay_comps for why we don't do it here.

        status.update(
            label=f"✅ Batch complete — {found}/{len(chunk)} priced "
                  f"({total_pending - len(chunk)} still pending)" if has_more
                  else f"✅ All chunks complete — {found}/{len(chunk)} priced in final batch",
            state="complete", expanded=False,
        )

    return df, found + melt_found, len(chunk) + melt_found, has_more


# Expanded easy-ship keyword pattern. Pass1's classify_logistics() only
# flags HARD (oversized) and EASY (coin/jewelry/watch/card mailbox tier);
# everything else is NEUTRAL. When the user opts into "Easy-ship only"
# mode, we promote NEUTRAL → EASY for any lot whose title hits this
# expanded shippable-goods pattern (toys, books, clothing, electronics,
# bags, etc.). Anything that doesn't match — and isn't already EASY —
# gets dropped from the comp queue.
_EASY_SHIP_TITLE_RE = re.compile(
    # Mailbox tier (already EASY) — repeated here so the combined
    # regex covers both buckets in one pass.
    r"\b(?:jewelry|watch|camera|card|cards|game|games|gold|silver|"
    r"nintendo|apple|ink|pen|coin|coins|currency|stamp|numismatic|"
    # Bags / accessories
    r"backpack|purse|handbag|wallet|crossbody|tote|clutch|fanny\s*pack|"
    r"loungefly|sling\s*bag|messenger\s*bag|"
    # Toys / collectibles small
    r"figure|figures|figurine|funko|pop\s*vinyl|plush|plushie|stuffed|"
    r"doll|barbie|hot\s*wheels|matchbox|lego|legos|playset|action\s*figure|"
    r"webkinz|tamagotchi|furby|pokemon|yu-?gi-?oh|booster|tcg|"
    # Media
    r"book|books|comic|comics|graphic\s*novel|manga|"
    r"\bdvd\b|blu-?ray|\bcd\b|\bcds\b|vinyl|record|cassette|"
    r"poster|print|magazine|"
    # Clothing / footwear / accessories
    r"shirt|t-?shirt|tee|hoodie|sweater|jacket|coat|jersey|"
    r"hat|cap|beanie|scarf|gloves|belt|tie|"
    r"shoe|shoes|sneaker|sneakers|boot|boots|heels|"
    r"dress|skirt|jeans|pants|shorts|leggings|"
    r"sunglasses|glasses|eyeglasses|"
    # Electronics small
    r"phone|iphone|ipad|airpods|earbuds|earphones|headphone|headphones|"
    r"speaker|charger|cable|adapter|mouse|keyboard|kindle|tablet|"
    r"console|controller|cartridge|gameboy|switch|playstation|xbox|"
    r"router|drone|gopro|"
    # Beauty / health
    r"perfume|cologne|fragrance|makeup|cosmetic|cosmetics|lipstick|"
    r"skincare|"
    # Crafts / small office
    r"yarn|fabric|notebook|journal|stationery|sticker|stickers|"
    # Tools small
    r"knife|knives|pocketknife|multitool|flashlight|"
    # Generic small-lot signals
    r"box\s*of|set\s*of\s*(?:\d+|small)|sealed|new\s*in\s*box|nib\b|"
    r"vintage|antique"
    r")\b",
    re.IGNORECASE,
)


# --- CPU lot form-factor classification ----------------------------
# A lot titled "Intel i5-8500 CPU Processor SR3XE" comps very
# differently than "HP ProDesk 600 G4 DM w/ Intel Core i5-8500T 8GB
# RAM" — one is a $40 chip, the other is a $70-150 system. We tag
# every CPU-category BOLO match with `cpu_form_factor` so the user
# can see at a glance whether a lot is bare silicon or a complete
# system, and the comp pipeline can route the comp query accordingly.
#
# Classification is conservative: when both bare-CPU and system signals
# fire (e.g., "Lot of CPUs and a desktop"), we return 'unclear'.

# Strong system signals — branded device names, or storage-spec
# language that bare CPUs never have. Listed in priority order.
_SYSTEM_SIGNAL_RE = re.compile(
    r'\b(?:'
    # Branded systems. "desktop" excludes "Desktop CPU/Processor/chip"
    # — Intel uses that as a tier descriptor on bare-chip listings
    # ("6-Core Desktop CPU Processor SR3XE") and we don't want to
    # mis-classify those as systems.
    r'laptop|notebook|desktop(?!\s+(?:cpu|processor|chip|class))|'
    r'workstation|tower|server\s+chassis|'
    r'all-?in-?one|\baio\b|mini\s*pc|micro\s*pc|sff\b|tiny\s+desktop|'
    # Branded model lines
    r'thinkpad|latitude|inspiron|elitebook|probook|macbook|imac|'
    r'precision|optiplex|prodesk|elitedesk|thinkcentre|'
    r'ideapad|legion|envy|pavilion|spectre|surface(?:\s+pro|\s+laptop)?|'
    r'chromebook|ultrabook|netbook|gaming\s+laptop|business\s+laptop|'
    # System-only specs (bare CPUs never have these)
    r'\d+\s*gb\s+(?:ram|memory|ddr)|'
    r'\d+\s*(?:gb|tb)\s+(?:ssd|hdd|nvme|m\.?2)|'
    r'\bno\s+(?:os|hdd|ssd)\b|\bw/?\s*os\b|\bwith\s+os\b|'
    r'windows\s+(?:10|11|7|xp|server)|'
    # Power / boot states only relevant to systems
    r'(?:powers?|boots?)\s+(?:on|up)|posts?\s+to\s+bios|'
    r'\bdoes\s+not\s+(?:power|boot|post)|'
    # Network / display ports indicate a chassis
    r'\bvga\b|\bhdmi\b|\bdisplayport\b|\bdvi\b|\bethernet\b|'
    # Server-rack form
    r'\d+\s*u\s+rack|rackmount|blade\s+server|tower\s+server'
    r')\b',
    re.IGNORECASE,
)

# Strong bare-CPU signals — language and SKU patterns only present
# when listing the chip alone.
_BARE_CPU_SIGNAL_RE = re.compile(
    r'(?:'
    # "CPU Processor" / "Processor CPU" combos are a bare-chip giveaway
    r'\bcpu\s+processor\b|\bprocessor\s+cpu\b|\bcpu\s+only\b|'
    # S-Spec / Q-Spec is etched on the heat spreader, listed in bare-CPU titles
    r'\bsr[a-z0-9]{3,5}\b|\bs-?spec\s*:?\s*[a-z0-9]+\b|'
    # Engineering / qualification samples
    r'\b(?:es|qs)\s+sample\b|engineering\s+sample|qualification\s+sample|'
    # Tray / OEM packaging
    r'\boem\s+tray\b|\btray\s+pack\b|\bbulk\s+tray\b|'
    # Socket-only mention without chassis context
    r'\b(?:fclga|lga)(?:1150|1151|1155|1156|1200|1700|2011|2066|3647|4189)\b'
    r')',
    re.IGNORECASE,
)


def _classify_cpu_form_factor(title: str, description: str = '') -> str:
    """Return 'bare_cpu', 'system', or 'unclear' for a CPU-category lot.

    Strategy:
    - Strong system signal → 'system' (wins ties because system-spec
      language is more specific than bare-CPU language)
    - Bare-CPU signal AND no system signal → 'bare_cpu'
    - Otherwise → 'unclear' (caller treats as bare-CPU comp by default
      but flags lower confidence)
    """
    haystack = f"{title or ''} {description or ''}"
    has_system = bool(_SYSTEM_SIGNAL_RE.search(haystack))
    has_bare = bool(_BARE_CPU_SIGNAL_RE.search(haystack))
    if has_system:
        # Even when bare-CPU language is present too, the system spec
        # dominates — "HP ProDesk w/ Intel i5-8500T CPU" is a system.
        return 'system'
    if has_bare:
        return 'bare_cpu'
    return 'unclear'


def _compute_cpu_form_factor_columns(df):
    """Add `cpu_form_factor` column for CPU-category BOLO matches.

    Only runs on rows with bolo_category == 'cpu' or
    bolo_category == 'mini_pc' (where the embedded-CPU value still
    matters as a floor). Other rows get NaN.

    Idempotent — can be called multiple times safely.
    """
    if df is None or df.empty or 'bolo_category' not in df.columns:
        return df
    out = df.copy()
    cpu_mask = out['bolo_category'].isin(['cpu', 'mini_pc'])
    if not cpu_mask.any():
        out['cpu_form_factor'] = pd.NA
        return out
    titles = out['title'].fillna('').astype(str) if 'title' in out.columns else pd.Series([''] * len(out), index=out.index)
    descs = (
        out['description'].fillna('').astype(str)
        if 'description' in out.columns
        else pd.Series([''] * len(out), index=out.index)
    )
    # Per-call cache keyed on (title, desc) — same dedup pattern as
    # _compute_bolo_columns. Estate-sale CPU lots have lots of repeats.
    _cache: dict = {}
    forms = pd.Series(pd.NA, index=out.index, dtype='object')
    for idx in out.index[cpu_mask]:
        key = (titles.loc[idx], descs.loc[idx])
        cached = _cache.get(key, _SENTINEL)
        if cached is _SENTINEL:
            cached = _classify_cpu_form_factor(*key)
            _cache[key] = cached
        forms.loc[idx] = cached
    out['cpu_form_factor'] = forms
    return out


def _apply_bid_cap_filter(df):
    """Drop lots whose next_bid (or current_bid) is above the user's cap.

    Reads ``comps_skip_above_bid`` from session state (default $100).
    Lots already bid above the cap have squeezed margins — not worth
    burning BOLO scan / audit / comp time on. Lots with no bid data
    yet (NaN) are KEPT — those haven't been priced and shouldn't get
    filtered blindly on missing data.

    Returns ``(filtered_df, reason_str_or_None)``. The reason string
    is non-None when at least one lot was dropped, e.g.
    ``"3,212 skipped — next-bid > $100"``.
    """
    if df is None or df.empty:
        return df, None
    skip_above = float(
        st.session_state.get('comps_skip_above_bid', 100.0) or 0
    )
    if skip_above <= 0:
        return df, None
    if 'next_bid' in df.columns:
        bid_col = pd.to_numeric(df['next_bid'], errors='coerce')
    elif 'current_bid' in df.columns:
        bid_col = pd.to_numeric(df['current_bid'], errors='coerce')
    else:
        return df, None
    before = len(df)
    keep_mask = bid_col.fillna(0) <= skip_above
    df = df[keep_mask]
    dropped = before - len(df)
    if dropped:
        return df, f"{dropped:,} bid > ${skip_above:.0f}"
    return df, None


def _apply_easy_ship_filter(df, *, drop_hard: bool = True):
    """Return ``(filtered_df, reasons)`` after applying the easy-ship filter.

    Combines the HARD-logistics drop and the easy-ship-only NEUTRAL
    drop into one helper so it can be reused at multiple pipeline
    stages (BOLO match, audit, comps). Both behaviors are gated by
    the matching `comps_*` session-state toggles.

    Parameters
    ----------
    df : pd.DataFrame
        Lot rows. Expected columns: ``logistics_ease``, ``title`` /
        ``enriched_title``. Missing columns are tolerated — the
        filter degrades gracefully (skips that pass).
    drop_hard : bool
        If ``True`` (default), apply the HARD-logistics drop.
        Caller can disable when the upstream pipeline already
        cut HARD lots and only the easy-ship pass is wanted.

    Returns
    -------
    (filtered_df, reasons) : tuple
        Filtered rows + a list of human-readable reason strings
        like ``"3,212 HARD-logistics"`` for the calling status panel.
    """
    reasons: list = []
    if df is None or df.empty:
        return df, reasons

    if (
        drop_hard
        and st.session_state.get('comps_exclude_hard', True)
        and 'logistics_ease' in df.columns
    ):
        before = len(df)
        df = df[df['logistics_ease'] != 'HARD']
        dropped = before - len(df)
        if dropped:
            reasons.append(f"{dropped:,} HARD-logistics")

    if st.session_state.get('comps_easy_ship_only', True):
        before = len(df)
        title_col = 'enriched_title' if 'enriched_title' in df.columns else 'title'
        if title_col in df.columns:
            titles = df[title_col].fillna('').astype(str)
            easy_re_match = titles.str.contains(
                _EASY_SHIP_TITLE_RE, regex=True, na=False,
            )
            if 'logistics_ease' in df.columns:
                already_easy = df['logistics_ease'] == 'EASY'
                keep_mask = already_easy | easy_re_match
            else:
                keep_mask = easy_re_match
            df = df[keep_mask]
            dropped = before - len(df)
            if dropped:
                reasons.append(f"{dropped:,} not-easy-ship")
    return df, reasons


# Set of BOLO brand names that signal "luxury watch / high-end jewelry"
# where a $5 next_bid is a starter-bid placeholder, not a real entry
# point. Used by `_compute_realistic_cost_columns` to flag rows whose
# displayed ROI is misleading until bidding opens.
_LUXURY_STARTER_BID_BRANDS = frozenset({
    'Rolex accessories',
    'Omega accessories',
    'Patek Philippe accessories',
    'Audemars Piguet accessories',
    'Cartier accessories',
    'Other Swiss luxury accessories',
})


def _compute_realistic_cost_columns(df):
    """Add `realistic_cost`, `realistic_roi`, and `bid_trap_warn` columns.

    Background: when an auction lot has a low starter `next_bid` ($5)
    but a high `auctioneer_est_low` ($16,000), the displayed est_roi
    is wildly inflated — once bidding opens, the lot clears closer
    to the auctioneer's estimate.

    realistic_cost = MAX(est_cost, auctioneer_est_low × 0.5)
        — half of the auctioneer's low estimate is a reasonable
        proxy for "what this lot will actually clear at."

    realistic_roi = (est_resale - realistic_cost) / realistic_cost × 100

    bid_trap_warn = True when:
        - bolo_brand ∈ luxury-watch set, AND
        - auctioneer_est_low > $1000, AND
        - current_bid < auctioneer_est_low × 0.1
    """
    out = df.copy()
    out['realistic_cost'] = pd.NA
    out['realistic_roi'] = np.nan
    out['bid_trap_warn'] = False
    if 'est_cost' not in out.columns:
        return out
    est_cost = pd.to_numeric(out['est_cost'], errors='coerce')
    est_low = (
        pd.to_numeric(out['auctioneer_est_low'], errors='coerce')
        if 'auctioneer_est_low' in out.columns
        else pd.Series(np.nan, index=out.index, dtype='float64')
    )
    est_resale = (
        pd.to_numeric(out['est_resale'], errors='coerce')
        if 'est_resale' in out.columns
        else pd.Series(np.nan, index=out.index, dtype='float64')
    )
    current_bid = (
        pd.to_numeric(out['current_bid'], errors='coerce')
        if 'current_bid' in out.columns
        else pd.Series(0, index=out.index, dtype='float64')
    )

    floor_from_estimate = est_low.fillna(0) * 0.5
    realistic = pd.concat(
        [est_cost.fillna(0), floor_from_estimate], axis=1
    ).max(axis=1)
    # Mask out rows where neither cost nor auctioneer estimate was
    # available — those have no basis for a "realistic cost" floor.
    # Use NaN (not pd.NA) so downstream arithmetic works element-wise
    # without hitting `float(pd.NA)` errors.
    has_basis = est_cost.notna() | est_low.notna()
    realistic = realistic.where(has_basis, np.nan)
    # Force float dtype so subsequent comparisons + arithmetic don't
    # trip on object-dtype mixed pd.NA values.
    realistic_f = pd.to_numeric(realistic, errors='coerce')
    out['realistic_cost'] = realistic_f

    has_resale = est_resale.notna() & realistic_f.notna() & (realistic_f > 0)
    if has_resale.any():
        roi = (est_resale - realistic_f) / realistic_f * 100
        out.loc[has_resale, 'realistic_roi'] = np.round(roi[has_resale], 0)

    if 'bolo_brand' in out.columns:
        is_luxury = out['bolo_brand'].isin(_LUXURY_STARTER_BID_BRANDS)
        est_low_high = est_low.fillna(0) > 1000
        bid_low = current_bid.fillna(0) < (est_low.fillna(0) * 0.1)
        out['bid_trap_warn'] = (is_luxury & est_low_high & bid_low).fillna(False)

    return out


# --- Shipping-cost extraction ----------------------------------------
# HiBid auctions advertise their shipping policy in the auction name
# and/or in a boilerplate paragraph that gets copied onto every lot's
# description. Common patterns:
#   "MAY 6TH WEDNESDAY HUMP DAY AUCTION FREE SHIPPING"     → $0
#   "Shipping: $15 flat rate"                              → $15
#   "Flat rate shipping is $12 per item"                   → $12
#   "Shipping starts at $8"                                → $8
#   "$15 shipping"                                         → $15
# The configured `bundled_ship_cost` ($25) is just a fallback when no
# pattern matches. Without per-auction extraction, every lot in a
# free-shipping auction gets falsely penalized $25 → Target Buy is
# under-stated (sometimes to $0 / NaN).

_SHIP_COST_PATTERNS = (
    # Free shipping is the most important signal — match before $-patterns
    re.compile(r'\bfree\s+ship(?:ping)?\b', re.IGNORECASE),
)

# Numeric extractors. Return the captured float when matched.
_SHIP_COST_PRICE_PATTERNS = (
    re.compile(
        r'(?:flat[\s-]*rate\s+)?ship(?:ping)?'
        r'(?:\s+(?:is|starts?\s+at|cost|costs?|fee|fees|per\s+item|per\s+lot))?'
        r'\s*[:\-]?\s*\$\s*(\d+(?:\.\d{1,2})?)',
        re.IGNORECASE,
    ),
    re.compile(
        r'\$\s*(\d+(?:\.\d{1,2})?)\s*(?:flat[\s-]*rate\s+)?ship(?:ping)?',
        re.IGNORECASE,
    ),
)


def _extract_shipping_cost(text):
    """Parse a shipping-cost dollar amount from auction-name / description text.

    Returns ``0.0`` for free-shipping language, a positive float when a
    dollar amount is found near a "ship/shipping" keyword, or ``None``
    when nothing matches (caller falls back to the configured default).
    """
    if not text:
        return None
    s = str(text)
    for p in _SHIP_COST_PATTERNS:
        if p.search(s):
            return 0.0
    for p in _SHIP_COST_PRICE_PATTERNS:
        m = p.search(s)
        if m:
            try:
                v = float(m.group(1))
                # Sanity bound — keep results in the $1-$200 range.
                # Outside that we're probably matching a price quote
                # ("$3,250 retail") rather than a shipping fee.
                if 0.5 <= v <= 200:
                    return v
            except (ValueError, IndexError):
                pass
    return None


def _build_auction_ship_map(df, default_cost: float):
    """Return ``{auction_id: ship_cost_float}`` for every auction in df.

    Strategy per auction:
      1. Check the auction NAME first — "FREE SHIPPING" / "$15 SHIP"
         language there is the most authoritative signal.
      2. If the name is silent, scan up to 25 lot descriptions in
         that auction. The shipping boilerplate, when present, is
         usually copied verbatim onto every lot. Take the most
         common non-None match across the sample.
      3. Fall back to ``default_cost`` when no signal anywhere.

    The map is built once per render and used to vectorize Target Buy.
    """
    out: dict = {}
    if df is None or df.empty:
        return out
    if 'auction_link' not in df.columns:
        return out
    aid_series = df['auction_link'].apply(_aid_from_link)
    name_col = df['auction'] if 'auction' in df.columns else None
    desc_col = df['description'] if 'description' in df.columns else None

    # Pre-group lot descriptions per auction once
    desc_by_aid: dict = {}
    if desc_col is not None:
        for aid, group_idx in df.groupby(aid_series).groups.items():
            if aid is None:
                continue
            samples = desc_col.loc[group_idx].dropna().astype(str).head(25).tolist()
            desc_by_aid[aid] = samples

    seen_aids: set = set()
    for aid in aid_series.tolist():
        if aid is None or aid in seen_aids:
            continue
        seen_aids.add(aid)

        # 1. Auction name signal (cheapest + most authoritative)
        name_text = ''
        if name_col is not None:
            mask = aid_series == aid
            if mask.any():
                first = name_col[mask].dropna().head(1).tolist()
                name_text = first[0] if first else ''
        signal = _extract_shipping_cost(name_text)

        # 2. Scan lot descriptions for boilerplate
        if signal is None:
            samples = desc_by_aid.get(aid, [])
            hits: list = []
            for s in samples:
                v = _extract_shipping_cost(s)
                if v is not None:
                    hits.append(v)
            if hits:
                # Mode (most common) — defends against an outlier
                # description that slipped a non-shipping dollar amount
                # past our sanity bounds.
                from collections import Counter
                signal = Counter(hits).most_common(1)[0][0]

        out[aid] = signal if signal is not None else default_cost
    return out


def _compute_max_bid(df, target_roi_val):
    """Back out the max bid that still hits target_roi_val × cost.

    Returns a new DataFrame with 'max_bid' column set (or left as None where
    no est_resale is available).
    """
    from scraper.config_loader import load_config
    cfg = load_config()
    ebay_fee_pct = 0.1325
    ebay_fee_flat = 0.30
    buyer_premium_pct = cfg.get("shipping", {}).get("buyer_premium_pct", 15.0) / 100.0

    out = df.copy()
    out['max_bid'] = None
    if 'est_resale' not in out.columns:
        return out

    # (Per-auction ship-map plumbing removed 7/10 — shipping no
    # longer factors into max_bid; see the shipping banner instead.)

    # Per-auction buyer-premium multiplier (parsed from HiBid's
    # buyerPremium text at Phase-1 time). Real premiums range 0-22%;
    # the flat config default understates cost on high-premium
    # auctions. Rows without a parsed value fall back to config.
    _bp_default_mult = 1.0 + buyer_premium_pct
    if 'auction_buyer_premium_pct' in out.columns:
        _bp_vec = _to_float_array(out['auction_buyer_premium_pct'])
        _bp_vec = np.where(
            np.isnan(_bp_vec) | (_bp_vec <= 0), _bp_default_mult, _bp_vec
        )
    else:
        _bp_vec = np.full(len(out), _bp_default_mult, dtype='float64')

    # Cached est_resale can land as `object` dtype with internals
    # (Decimal, nullable extension dtype) that survive `pd.to_numeric`
    # but break Series.round. Coerce element-wise to a plain float64
    # numpy array first — np.round is dtype-strict from the start.
    resale = _to_float_array(out['est_resale'])
    resale_mask = ~np.isnan(resale)
    if resale_mask.any():
        net_resale = resale[resale_mask] * (1 - ebay_fee_pct) - ebay_fee_flat
        # Shipping deliberately excluded (7/10) — max_bid answers
        # "what bid hits target ROI on the item itself"; the shipping
        # situation is a banner, not a hidden subtraction.
        max_bid = (net_resale / target_roi_val) / _bp_vec[resale_mask.nonzero()[0]]
        # Negative max_bid means shipping + premium eats the entire
        # resale margin at target ROI — there's no positive bid that
        # hits the target. Show as NaN so the column renders blank
        # ("don't bid"), not "$0.00" (which reads as broken).
        max_bid_arr = np.round(max_bid, 2)
        max_bid_arr = np.where(max_bid_arr > 0, max_bid_arr, np.nan)
        out.loc[resale_mask, 'max_bid'] = max_bid_arr
    return out


def _normalize_nullable_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert pandas nullable extension dtypes to plain numpy dtypes.

    Float64/Int64 → float64 (pd.NA becomes np.nan); boolean → bool
    (pd.NA becomes False). pd.NA is hostile to scalar code paths —
    float(pd.NA), `pd.NA or x`, and `pd.NA > 0` all raise TypeError,
    while np.nan flows through them harmlessly. Cache round-trips and
    dataframe merges are the usual source of these dtypes.
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    for col in out.columns:
        dt = str(out[col].dtype)
        if dt in ('Float64', 'Int64'):
            out[col] = out[col].astype('float64')
        elif dt == 'boolean':
            out[col] = out[col].fillna(False).astype(bool)
        elif dt == 'object':
            # Object columns can carry pd.NA SCALARS after a merge even
            # though the dtype stays object. Swap them (and None) for
            # np.nan, which downstream float()/comparison code tolerates.
            s = out[col]
            mask = s.isna()
            if mask.any():
                out[col] = s.where(~mask, np.nan)
    return out


def _to_float_array(series: pd.Series) -> np.ndarray:
    """Coerce a Series to a numpy float64 array element-wise.

    Element-wise float() avoids the dtype gotchas of pandas'
    `to_numeric` on cached payloads — same helper as in scraper/cache.py.
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


def _render_red_flag_editor(ar_full):
    """Render the red-flag review expander (bulk-clear button + editor).

    Pulled out of the inline analysis-view so the right-side results panel
    can render it from anywhere. ar_full is mutated in place when the
    user clicks the bulk-clear button or unchecks individual rows;
    audit_results in session_state is also written to keep things in sync.
    """
    flagged_mask = ar_full['red_flag'].fillna(False).astype(bool)
    flagged_count = int(flagged_mask.sum())
    with st.expander(
        f"🚩 Review {flagged_count} red-flagged item(s) "
        "— uncheck any the audit got wrong",
        expanded=False,
    ):
        # Bulk-clear escape hatch for auctions where the audit is mostly
        # wrong (typically sports cards / TCG / video games — the AI's
        # risk labels don't apply to collectibles whose condition is
        # encoded as a grade in the title).
        if st.button(
            f"⚡ Clear all {flagged_count} flag(s) in this auction",
            help="Set red_flag=False on every flagged row. "
                 "Use when the audit was systematically wrong "
                 "(e.g. an auction full of cards or comics). "
                 "Items will rejoin the next comps batch.",
            key=f"clear_all_flags_{st.session_state.get('current_auction', '')}",
        ):
            ar_full.loc[flagged_mask, 'red_flag'] = False
            st.session_state.audit_results = ar_full
            st.session_state._comps_has_more = True
            _save_current_auction_to_cache()
            st.success(
                f"✓ Cleared all {flagged_count} red flag(s). "
                "They'll be included in the next comps batch."
            )
            st.rerun()

        title_col = (
            'enriched_title' if 'enriched_title' in ar_full.columns
            else 'title'
        )
        review_cols = ['red_flag', title_col, 'verdict']
        if 'confidence' in ar_full.columns:
            review_cols.append('confidence')
        if 'description' in ar_full.columns:
            review_cols.append('description')
        if 'lot_link' in ar_full.columns:
            review_cols.append('lot_link')
        review_cols = [c for c in review_cols if c in ar_full.columns]
        review_df = ar_full.loc[flagged_mask, review_cols].copy()
        editor_sig = (
            st.session_state.get('current_auction', ''),
            flagged_count,
            tuple(review_df.index.tolist()[:50]),
        )
        editor_key = f"red_flag_review_{hash(editor_sig)}"

        edited_review = st.data_editor(
            review_df,
            width='stretch',
            hide_index=True,
            column_order=review_cols,
            disabled=[c for c in review_cols if c != 'red_flag'],
            column_config={
                "red_flag": st.column_config.CheckboxColumn(
                    "🚩 Flagged",
                    help="Uncheck to clear the flag and let this "
                         "item join the next comps batch.",
                    width="small",
                ),
                title_col: st.column_config.TextColumn(
                    "Title", width="large",
                ),
                "verdict": st.column_config.TextColumn(
                    "Reason", width="medium",
                ),
                "confidence": st.column_config.ProgressColumn(
                    "Confidence", min_value=0, max_value=100,
                    format="%.0f%%",
                ),
                "description": st.column_config.TextColumn(
                    "Description", width="large",
                    help="Original lot description — gives the "
                         "context the AI scored against.",
                ),
                "lot_link": st.column_config.LinkColumn(
                    "Open", display_text="🔗",
                ),
            },
            key=editor_key,
        )

        unchecked = review_df.index[
            review_df['red_flag'].fillna(False).astype(bool)
            & ~edited_review['red_flag'].fillna(False).astype(bool).values
        ]
        if len(unchecked) > 0:
            ar_full.loc[unchecked, 'red_flag'] = False
            st.session_state.audit_results = ar_full
            st.session_state._comps_has_more = True
            _save_current_auction_to_cache()
            st.success(
                f"✓ Unflagged {len(unchecked)} item(s). They'll be "
                "included in the next price-comps batch."
            )
            st.rerun()


def _compute_resale_confidence(row):
    """Score est_resale credibility on a 0-100 scale.

    Three independent signals get a 0-1 score and are then weighted:
      - Source quality (50% weight): sold > active. eBay+Mercari sold
        is ideal; PriceCharting is good when its product match is
        right (we can't verify that here); 'thin comps' is the
        pricer's own self-flag for 1-2 sold listings; 'active'
        fallback has no outlier protection at all.
      - Sample size (30% weight): more comps = more robust median.
        PriceCharting's single match counts as ~5-comp because their
        pricing is server-side aggregated across many transactions.
      - Spread tightness (20% weight): wider (high-low)/median ratio
        signals outlier listings dragging the median around.

    Returns None when est_resale is missing.
    """
    median = row.get('est_resale')
    if median is None or pd.isna(median):
        return None
    src = str(row.get('price_source') or '').lower()

    def _int0(v):
        # NaN-safe int coercion. `v or 0` is NOT safe here: NaN is
        # truthy, so `NaN or 0` returns NaN and int(NaN) raises —
        # crashed the whole results table on 7/11 when ledger-
        # overlaid frames carried NaN comp counts.
        try:
            if v is None or pd.isna(v):
                return 0
            return int(float(v))
        except (TypeError, ValueError):
            return 0

    comp_count = _int0(row.get('comp_count'))
    pc_comps = _int0(row.get('pricecharting_comps'))
    gc_comps = _int0(row.get('gocollect_comps'))
    low = row.get('price_low')
    high = row.get('price_high')
    low = low if pd.notna(low) else median
    high = high if pd.notna(high) else median

    # Source quality (0-1)
    if gc_comps > 0 or 'gocollect' in src:
        # GoCollect: curated CGC/BGS/SGC-graded data, matched at the
        # issue+grade level. Highest signal we have for graded comics.
        source_score = 0.95
    elif 'sold' in src and 'thin' not in src:
        source_score = 1.0 if 'ebay+mercari' in src else 0.9
    elif pc_comps > 0 or 'pricecharting' in src:
        source_score = 0.75
    elif 'amazon-live' in src:
        # Factor of the LIVE Amazon buy-box price — the strongest
        # anchor variant (no staleness, no auctioneer inflation).
        source_score = 0.65
    elif 'retail-anchor' in src:
        # Factor-of-stated-retail estimate: better than nothing,
        # weaker than any real sold comp.
        source_score = 0.55
    elif 'thin' in src:
        source_score = 0.4
    elif 'active' in src:
        source_score = 0.25
    else:
        source_score = 0.5

    # Sample size (0-1)
    if gc_comps > 0:
        # GoCollect's single match is a curated issue+grade record
        # backed by Heritage / eBay / ComicConnect aggregation —
        # treat like a high-comp PC hit.
        sample_score = 0.9
    elif pc_comps > 0:
        sample_score = 0.85
    elif comp_count >= 5:
        sample_score = 1.0
    elif comp_count >= 4:
        sample_score = 0.9
    elif comp_count >= 3:
        sample_score = 0.75
    elif comp_count >= 2:
        sample_score = 0.55
    elif comp_count >= 1:
        sample_score = 0.35
    else:
        sample_score = 0.0

    # Spread tightness (0-1)
    if median and median > 0:
        spread_ratio = (high - low) / median
        if spread_ratio <= 0.1:
            spread_score = 1.0
        elif spread_ratio <= 0.25:
            spread_score = 0.9
        elif spread_ratio <= 0.5:
            spread_score = 0.75
        elif spread_ratio <= 1.0:
            spread_score = 0.55
        else:
            spread_score = 0.35
    else:
        spread_score = 0.7

    base_score = (source_score * 0.5 + sample_score * 0.3 + spread_score * 0.2)

    # Reality check: if the live current_bid is multiples of our est,
    # the market is loudly disagreeing with us. Our query probably
    # matched the wrong product / grade / variant. Knock confidence
    # down hard so the user can spot these in the table.
    try:
        bid = float(row.get('current_bid') or 0)
        if bid > 0 and median > 0:
            ratio = bid / median
            if ratio >= 10:
                base_score *= 0.25      # bid >= 10x est → est is almost certainly wrong
            elif ratio >= 5:
                base_score *= 0.4
            elif ratio >= 2.5:
                base_score *= 0.6
    except (TypeError, ValueError):
        pass

    # Authenticity gate: when the BOLO matcher flagged this as
    # auth_required AND the description-based auth check produced a
    # low score, halve resale confidence. The eBay sold-comps for
    # luxury are flooded with replicas, so a low auth_score means
    # our est_resale is probably already mixing real-and-fake comps.
    # Even if the est is "right" for the contaminated market, the
    # user can't realize it without authenticating, so the practical
    # confidence in the resale is lower.
    if row.get('bolo_auth_required'):
        auth_score = row.get('bolo_auth_score')
        try:
            auth_score = float(auth_score) if auth_score is not None else None
        except (TypeError, ValueError):
            auth_score = None
        if auth_score is not None:
            if auth_score < 30:
                base_score *= 0.3       # strong red flags → near-zero usable confidence
            elif auth_score < 50:
                base_score *= 0.55
            elif auth_score < 70:
                base_score *= 0.8
            # 70+ : description supports authenticity, no penalty
        else:
            # Auth-required but no score (no description text to
            # analyze) — treat as ambiguous, mild discount.
            base_score *= 0.7

    return int(round(base_score * 100))


def _render_results_table(results_df):
    """Render the results table with live ROI/STR threshold highlighting.

    Rows are color-coded by `buy_grade` (the composite "Should I buy?"
    score that already factors in margin, headroom, STR, and comp
    confidence). The old ROI / STR threshold sliders + green/yellow
    tier system were removed — they fragmented signal across two
    sliders that the user had to interpret separately, and produced
    misleading "all green" calls on contaminated comps. The buy_score
    consolidates everything into one number with one color scale.
    """
    # Target ROI is still needed internally to compute max_bid (the
    # "what's the highest bid that hits 3× cost" column). Kept as a
    # session_state default so power users could override via session
    # state, but no widget surfaces it — 3× is the right default for
    # most resale categories and the buy_score absorbs threshold-tuning
    # into its margin component.
    target_roi_val = float(
        st.session_state.get("target_roi_live", 3.0) or 3.0
    )

    # --- Normalize pandas nullable extension dtypes FIRST ---
    # Cache round-trips and merges can land columns as Float64 / Int64 /
    # boolean extension dtypes whose missing value is pd.NA (NAType).
    # Unlike np.nan, float(pd.NA) raises TypeError and `pd.NA or x` /
    # `pd.NA > 0` raise too — any bare float()/comparison downstream
    # becomes a render crash ("float() argument must be a string or a
    # real number, not 'NAType'", seen 7/6 on the Hayworth reload).
    # Convert once here: numeric extensions → float64 (pd.NA → np.nan),
    # boolean → plain bool with NA → False.
    results_df = _normalize_nullable_dtypes(results_df)

    # --- Recompute max_bid with current target (dynamic) ---
    working = _compute_max_bid(results_df, target_roi_val)
    # --- Buy-grade RUNS HERE, after max_bid is populated. Computing it
    # at the end of the comp pipeline (where max_bid is still NaN)
    # routed every comped lot through the "uncomped" gate. Caught in
    # Final Kemah audit 5/11.
    working = _compute_buy_score(working)

    # --- Tag every row with brand-BOLO matches.  Free regex pass so
    #     it's safe to rerun on every render — picks up changes to
    #     the underlying clothing_brand_bolo.json file mid-session.
    #     The matcher is vectorized + has an internal per-call (title,
    #     description) cache, so a fresh call on ~2000 lots is ~100ms.
    #     We used to wrap this in a session-state cache keyed on a
    #     2000-element tuple of lot_ids, but hashing/comparing that
    #     tuple cost more than the work we were trying to skip. ---
    working = _compute_bolo_columns(working)
    bolo_total = int(working['bolo_brand'].notna().sum()) if 'bolo_brand' in working.columns else 0

    # --- CAP `bolo_target_buy_high` BY `max_bid` ---
    # The BOLO file's target_buy_usd.high is a category-level
    # guideline; it doesn't know the comp-derived resale of THIS
    # specific lot. Take the MIN(static, max_bid) so the displayed
    # Target Buy never exceeds the resell-ROI ceiling. Also: when
    # the cap drops to 0 (shipping eats the margin), set NaN so
    # the column renders blank ("don't bid") instead of "$0".
    if (
        'bolo_target_buy_high' in working.columns
        and 'max_bid' in working.columns
    ):
        _static = pd.to_numeric(working['bolo_target_buy_high'], errors='coerce')
        _math = pd.to_numeric(working['max_bid'], errors='coerce')
        _capped = _static.where(_math.isna() | (_static <= _math), _math)
        # Use NaN (not pd.NA) so downstream consumers calling `float()`
        # on the column don't blow up — float(NaN) is a real float,
        # float(pd.NA) raises TypeError.
        _capped = _capped.where(_capped > 0, np.nan)
        working['bolo_target_buy_high'] = _capped

    # --- Realistic cost / ROI / bid-trap-warn columns ---
    # When a luxury-watch lot has $5 next-bid but $16K auctioneer
    # estimate, est_roi is wildly inflated — once bidding opens the
    # lot clears closer to the auctioneer estimate. Compute a second
    # cost basis from MAX(est_cost, auctioneer_est_low × 0.5) and
    # flag obvious starter-bid traps.
    working = _compute_realistic_cost_columns(working)

    # --- Sort: actionable-first ordering ---
    # Was: est_roi % desc alone, which put luxury-watch starter-bid
    # traps (Cartier $5→$3K = 50000% ROI) at the top of the table.
    # Now: traps last, then ROI tier (≥200% high-prio first),
    # then absolute resale within tier (so $1K tennis bracelet
    # floats above $200 small pendant), then realistic_roi tiebreak.
    if 'est_roi' in working.columns:
        _trap = working.get(
            'bid_trap_warn', pd.Series(False, index=working.index)
        ).fillna(False).astype(bool)
        _real_roi = pd.to_numeric(
            working.get('realistic_roi', pd.Series(np.nan, index=working.index)),
            errors='coerce',
        )
        _est_roi = pd.to_numeric(working['est_roi'], errors='coerce')
        _effective_roi = _real_roi.fillna(_est_roi)
        _resale = pd.to_numeric(
            working.get('est_resale', pd.Series(np.nan, index=working.index)),
            errors='coerce',
        ).fillna(0)
        working['_trap_key'] = _trap.astype(int)
        working['_roi_tier'] = (_effective_roi.fillna(-1) >= 200).astype(int) * -1
        working['_resale_sort'] = _resale
        working['_roi_sort'] = _effective_roi.fillna(-9999)
        working = working.sort_values(
            ['_trap_key', '_roi_tier', '_resale_sort', '_roi_sort'],
            ascending=[True, True, False, False],
            na_position='last',
        ).drop(
            columns=['_trap_key', '_roi_tier', '_resale_sort', '_roi_sort']
        ).reset_index(drop=True)

    # --- Estimated profit per lot (after eBay fees) ---
    # Formula matches the gross-of-fees ROI used everywhere else but
    # nets the eBay take rate so the "profit" number is what would
    # actually hit the bank account, not the headline revenue. Same
    # 13.25% + $0.30 fee structure used by _compute_max_bid().
    # Lots without est_resale or est_cost get NaN → excluded from sums.
    if 'est_resale' in working.columns and 'est_cost' in working.columns:
        _resale = pd.to_numeric(working['est_resale'], errors='coerce')
        _cost = pd.to_numeric(working['est_cost'], errors='coerce')
        _net_resale = _resale * (1 - 0.1325) - 0.30
        working['est_profit'] = (_net_resale - _cost).round(2)
    else:
        working['est_profit'] = pd.Series(pd.NA, index=working.index, dtype="Float64")

    # --- Buy-grade aggregate counts for the status line ---
    # buy_grade is the composite "Should I buy?" letter (🟢 A, 🟡 B,
    # 🟠 C, 🔴 D, ⚫ F). Replaces the old meets_roi / meets_str /
    # meets_both / meets_either masks — that system fragmented signal
    # across two sliders and produced misleading "all green" calls
    # on contaminated comps. The buy_grade absorbs threshold tuning
    # into the score.
    grade_col = working.get('buy_grade')
    n_a = n_b = n_c = n_d = n_q = 0
    if grade_col is not None:
        gc = grade_col.fillna('').astype(str)
        # Use the emoji prefix to identify each grade unambiguously
        # (substring matches like "A" used to false-match "🔍 ?" via
        # the question mark — emojis ensure clean buckets).
        n_a = int(gc.str.startswith('🟢').sum())
        n_b = int(gc.str.startswith('🟡').sum())
        n_c = int(gc.str.startswith('🟠').sum())
        n_d = int(gc.str.startswith('🔴').sum())
        n_q = int(gc.str.startswith('🔍').sum())

    # Profit aggregate for the A+B (likely-buy) tier — gives the user
    # a "what's at stake if I bid all the green / yellow lots?" number.
    _profit_num = pd.to_numeric(
        working.get('est_profit', pd.Series(dtype=float)),
        errors='coerce',
    )
    _ab_profit = 0.0
    if grade_col is not None:
        ab_mask = grade_col.fillna('').astype(str).str.startswith(
            ('🟢', '🟡')
        )
        _ab_profit = float(_profit_num[ab_mask].sum() or 0.0)

    # --- One-line status row ---
    status_bits = [f"**{len(working)}** leads"]
    if 'est_resale' in working.columns:
        status_bits.append(f"**{int(working['est_resale'].notna().sum())}** comped")
    grade_chip = []
    if n_a: grade_chip.append(f"🟢 **{n_a}** A")
    if n_b: grade_chip.append(f"🟡 **{n_b}** B")
    if n_c: grade_chip.append(f"🟠 **{n_c}** C")
    if n_d: grade_chip.append(f"🔴 **{n_d}** D")
    if n_q: grade_chip.append(f"🔍 **{n_q}** uncomped w/ signals")
    if grade_chip:
        status_bits.append(" · ".join(grade_chip))
    if (n_a + n_b) > 0:
        status_bits.append(f"💰 **${_ab_profit:,.0f}** est. profit (A+B)")
    if 'red_flag' in working.columns:
        status_bits.append(f"🚩 **{int(working['red_flag'].sum())}** red-flagged")
    if 'manual_check' in working.columns:
        n_manual = int(working['manual_check'].fillna(False).astype(bool).sum())
        if n_manual > 0:
            status_bits.append(f"🔍 **{n_manual}** need manual review")
    if 'ebay_str' in working.columns:
        has_str = int(working['ebay_str'].notna().sum())
        if has_str:
            avg_str = pd.to_numeric(working['ebay_str'], errors='coerce').mean()
            status_bits.append(f"avg STR **{avg_str:.0f}%**")
    if bolo_total > 0:
        status_bits.append(f"🎯 **{bolo_total}** BOLO match{'es' if bolo_total != 1 else ''}")
    # Stylized/replica count — independent of BOLO matching, since
    # "Rolex style watch" might not even hit the BOLO matcher but
    # still has the contaminated-comps problem.
    if 'is_stylized_replica' in working.columns:
        stylized_total = int(working['is_stylized_replica'].fillna(False).astype(bool).sum())
        if stylized_total > 0:
            status_bits.append(f"⚠️ **{stylized_total}** stylized/replica")
    # Auth-required count: how many of the BOLO matches are tier-3
    # luxury or Polo sublines that need authentication. Surfacing
    # this here helps the user gauge how much of the auction needs
    # extra scrutiny before bidding.
    if 'bolo_auth_required' in working.columns:
        auth_req_total = int(working['bolo_auth_required'].fillna(False).astype(bool).sum())
        if auth_req_total > 0:
            low_auth = 0
            if 'bolo_auth_score' in working.columns:
                low_auth = int(
                    (pd.to_numeric(working['bolo_auth_score'], errors='coerce') < 30).sum()
                )
            piece = f"🛡️ **{auth_req_total}** auth-required"
            if low_auth:
                piece += f" ({low_auth} low-score)"
            status_bits.append(piece)
    st.caption(" · ".join(status_bits))

    # Stylized/replica banner — shown unconditionally whenever there's
    # at least one such lot, because these are NEVER eligible for
    # authentic-brand comps and the user benefits from seeing them
    # called out distinctly from auth-ambiguous-but-real lots.
    if 'is_stylized_replica' in working.columns:
        stylized_mask = working['is_stylized_replica'].fillna(False).astype(bool)
        n_stylized = int(stylized_mask.sum())
        if n_stylized:
            sample_phrases = (
                working.loc[stylized_mask, 'stylized_phrase']
                .dropna().astype(str).unique().tolist()[:5]
            )
            phrase_str = ", ".join(f"'{p}'" for p in sample_phrases if p)
            pc_check_on = bool(st.session_state.get('comps_pc_check_stylized', True))
            mode_caption = (
                "**eBay/Mercari are skipped** for these (authentic-brand "
                "comps don't apply to a stylized/inspired piece) but "
                "**PriceCharting still runs** — free curated catalog "
                "lookup, no contamination risk. PC returns nothing for "
                "products it doesn't cover (handbags / jewelry / "
                "decorative sculpture), which is the safe outcome."
                if pc_check_on else
                "All comp sources are skipped for these (no est_resale "
                "data will populate). Toggle the *Run PC on stylized "
                "lots* checkbox in comp settings to enable free PC "
                "fallback."
            )
            st.error(
                f"⚠️ **{n_stylized} lot(s)** contain stylized/replica "
                f"language ({phrase_str}). {mode_caption}"
            )

    # Inline warning banner: when any auth-required lot has score < 30,
    # the auction is contaminated by replica-market comps and the user
    # should know before they look at ROI numbers. Placed right above
    # the threshold inputs so it's the first thing they see.
    if 'bolo_auth_required' in working.columns and 'bolo_auth_score' in working.columns:
        scores = pd.to_numeric(working['bolo_auth_score'], errors='coerce')
        very_low = (
            working['bolo_auth_required'].fillna(False).astype(bool)
            & scores.notna()
            & (scores < 30)
        )
        n_very_low = int(very_low.sum())
        if n_very_low:
            st.warning(
                f"🛡️ **{n_very_low} lot(s)** are flagged auth-required with "
                "auth_score < 30 (auctioneer used 'as-is' / 'designer-style' / "
                "no-auth language). For luxury BOLO matches, eBay sold-comps "
                "are flooded with replicas — est_resale on these lots is "
                "almost certainly mixing real-and-fake comps. **Do not bid "
                "without independent authentication.**"
            )
        else:
            # Softer info banner when there are auth-required lots but
            # nothing is flagrantly red. Reminds the user that even
            # neutral-scored luxury lots need authentication before
            # bidding — replica markets are big and the eBay sold-
            # comp pool is contaminated regardless of how nice the
            # auctioneer's description sounds.
            auth_req_total_for_banner = int(
                working['bolo_auth_required'].fillna(False).astype(bool).sum()
            )
            if auth_req_total_for_banner > 0:
                ambiguous = int(
                    (
                        working['bolo_auth_required'].fillna(False).astype(bool)
                        & scores.notna()
                        & (scores < 60)
                    ).sum()
                )
                if ambiguous:
                    st.info(
                        f"🛡️ **{auth_req_total_for_banner} auth-required lot(s)** "
                        f"({ambiguous} with ambiguous score < 60). The "
                        "eBay sold-comp pool for tier-3 luxury is heavily "
                        "contaminated by replicas, so est_resale is best "
                        "treated as 'what the contaminated market clears at,' "
                        "not 'what an authenticated piece is worth.' Use the "
                        "🛡️ Photo authentication panel below to spot-check "
                        "specific lots before bidding."
                    )

    # ---- Photo-based verification panel ----
    # When the auction has auth-required BOLO matches, expose a popover
    # that lets the user pick a specific lot and run Claude vision on
    # its photo against the brand's era_markers checklist. One vision
    # call per lot (~$0.005 with claude-haiku-4-5), explicit rather
    # than auto so the user controls the spend.
    if 'bolo_auth_required' in working.columns:
        auth_lots = working[
            working['bolo_auth_required'].fillna(False).astype(bool)
        ]
        if len(auth_lots) > 0 and 'thumbnail_url' in auth_lots.columns:
            with st.expander(
                f"🛡️ Photo authentication ({len(auth_lots)} auth-required lot(s) — "
                "one Claude vision call per click, ~$0.005 each)",
                expanded=False,
            ):
                st.caption(
                    "Pick a lot and click **Verify**. Claude will look at the "
                    "photo against the brand's era_markers checklist and report "
                    "what it can/can't see plus any obvious red flags. This is "
                    "NOT a replacement for professional authentication — but it "
                    "catches the obvious fakes (off-center monogram, wrong "
                    "hardware color, blurry heat-stamp) that experienced eyes "
                    "would also flag."
                )
                # Build a label list: brand · model · est_resale · current_bid
                lot_labels = []
                for _, lot_row in auth_lots.iterrows():
                    brand = lot_row.get('bolo_brand') or '?'
                    model = lot_row.get('bolo_model') or ''
                    title_short = (str(lot_row.get('title') or '')[:60])
                    bid = lot_row.get('current_bid') or 0
                    label = (f"{brand} · {title_short}"
                             f" — bid ${bid}")
                    lot_labels.append(label)
                pick = st.selectbox(
                    "Pick a lot to verify",
                    options=list(range(len(lot_labels))),
                    format_func=lambda i: lot_labels[i],
                    key="auth_verify_pick",
                )
                picked_row = auth_lots.iloc[pick]
                # Store per-lot results in session_state keyed by lot_id
                # so re-clicking the popover doesn't re-fire the API call.
                lot_id = picked_row.get('lot_id')
                cache_key = f"_auth_photo_result_{lot_id}"
                vc1, vc2 = st.columns([1, 1])
                with vc1:
                    if st.button(
                        "🔍 Verify with Claude vision",
                        key=f"auth_verify_btn_{lot_id}",
                        type="primary",
                        width='stretch',
                        help="Spends one Claude API call. Result is cached "
                             "per-lot for the rest of the session.",
                    ):
                        # Pull the API key from config
                        from scraper.config_loader import load_config
                        try:
                            _cfg = load_config()
                            ant_cfg = _cfg.get('anthropic') or {}
                            ant_key = ant_cfg.get('api_key')
                            ant_model = ant_cfg.get('model') or 'claude-haiku-4-5'
                        except Exception:
                            ant_key = None
                            ant_model = 'claude-haiku-4-5'

                        if not ant_key:
                            st.error(
                                "No Anthropic API key in config.json. "
                                "Add `anthropic.api_key` to enable photo "
                                "verification."
                            )
                        else:
                            # Pull image URL — prefer fullsize then HD then
                            # plain thumbnail. Smaller images = lower
                            # signal but still usable for obvious fakes.
                            img_url = (
                                picked_row.get('fullsize_url')
                                or picked_row.get('hd_thumbnail_url')
                                or picked_row.get('thumbnail_url')
                            )
                            # Reconstruct the brand_match dict from the
                            # row's BOLO columns so analyze_photo has
                            # era_markers + notes to work with.
                            brand_match = _BOLO_MATCHER.match(
                                str(picked_row.get('title') or ''),
                                str(picked_row.get('description') or ''),
                            )
                            with st.spinner("Asking Claude to look at the photo…"):
                                result = _auth_analyze_photo(
                                    img_url, brand_match, ant_key, model=ant_model
                                )
                            st.session_state[cache_key] = result or {
                                'auth_score': None,
                                'photo_notes': '(no result returned — check API key / image URL)',
                                'red_flags': [], 'green_flags': [],
                                'era_seen': [], 'era_missing': [],
                            }
                            st.rerun()
                with vc2:
                    if st.button(
                        "🗑️ Clear result",
                        key=f"auth_clear_btn_{lot_id}",
                        width='stretch',
                        disabled=cache_key not in st.session_state,
                    ):
                        st.session_state.pop(cache_key, None)
                        st.rerun()

                # Render the cached result if present
                if cache_key in st.session_state:
                    result = st.session_state[cache_key]
                    score = result.get('auth_score')
                    quality = result.get('image_quality') or 'n/a'
                    rc1, rc2, rc3 = st.columns(3)
                    with rc1:
                        st.metric(
                            "Photo auth score",
                            f"{score}" if score is not None else "n/a",
                            help="0-100, Claude's confidence based on photo only",
                        )
                    with rc2:
                        st.metric("Image quality", quality)
                    with rc3:
                        # Merge with description-based score for a
                        # combined view if both exist.
                        desc_score = picked_row.get('bolo_auth_score')
                        if desc_score is not None and score is not None:
                            try:
                                merged = int(round(0.6 * float(score) + 0.4 * float(desc_score)))
                            except (TypeError, ValueError):
                                merged = None
                            st.metric(
                                "Merged score",
                                f"{merged}" if merged is not None else "n/a",
                                help="0.6 × photo + 0.4 × description",
                            )

                    if result.get('photo_notes'):
                        st.caption(f"📝 {result['photo_notes']}")

                    rf = result.get('red_flags') or []
                    gf = result.get('green_flags') or []
                    es = result.get('era_seen') or []
                    em = result.get('era_missing') or []
                    if rf:
                        st.error("🚩 **Red flags Claude saw:**\n\n- " + "\n- ".join(rf))
                    if gf:
                        st.success("✅ **Supports authenticity:**\n\n- " + "\n- ".join(gf))
                    if es:
                        st.markdown("**Era markers visible:**\n\n- " + "\n- ".join(es))
                    if em:
                        st.markdown("**Era markers NOT visible (need to verify in person):**\n\n- " + "\n- ".join(em))

    filtered_df = working

    # ⭐ Proven-lanes filter — one-click narrow to the three lanes the
    # user's sales history proved are high-margin AND easy-ship
    # (Loungefly, appliance/tool parts, small watch/eyewear accessories).
    # Only offered when the loaded set actually contains any — no point
    # showing a toggle that would empty the table.
    if ('bolo_proven_lane' in filtered_df.columns
            and filtered_df['bolo_proven_lane'].fillna(False).astype(bool).any()):
        _n_proven = int(
            filtered_df['bolo_proven_lane'].fillna(False).astype(bool).sum()
        )
        _proven_only = st.checkbox(
            f"⭐ Proven lanes only ({_n_proven} lots) — "
            f"Loungefly · appliance/tool parts · watch & eyewear "
            f"accessories",
            key="results_proven_lanes_only",
            help="Filter to BOLO matches in the three categories your "
                 "12-month sales history proved are both high-margin "
                 "and mailbox-shippable. Hides the rest (including the "
                 "sterling/diecast lanes that lost money last year).",
        )
        if _proven_only:
            filtered_df = filtered_df[
                filtered_df['bolo_proven_lane'].fillna(False).astype(bool)
            ]

    # Columns
    title_col = 'enriched_title' if 'enriched_title' in filtered_df.columns else 'title'
    # Lead with the lot thumbnail when we have one — Streamlit's ImageColumn
    # renders the URL inline so the user can scan visually before reading.
    display_cols = []
    # Lot link goes FIRST and is pinned to the left so it's always
    # reachable without scrolling — that's the most-clicked control.
    if 'lot_link' in filtered_df.columns:
        display_cols.append('lot_link')
    # Buy score sits right after the link — single-column "should I bid?"
    # answer, sortable to surface the best lots without scanning all
    # the underlying metrics.
    if 'buy_grade' in filtered_df.columns:
        display_cols.append('buy_grade')
    # ⭐ proven-lane marker column — a star on rows in the three money
    # lanes so they're spottable even without the filter on.
    if ('bolo_proven_lane' in filtered_df.columns
            and filtered_df['bolo_proven_lane'].fillna(False).astype(bool).any()):
        filtered_df = filtered_df.copy()
        filtered_df['lane'] = filtered_df['bolo_proven_lane'].map(
            lambda v: '⭐' if bool(v) else ''
        )
        display_cols.append('lane')
    if 'thumbnail_url' in filtered_df.columns:
        display_cols.append('thumbnail_url')
    display_cols.append(title_col)
    # When we're displaying enriched titles AND any row's enriched value
    # actually differs from the HiBid original, expose the original title
    # too so the user can see what was changed.
    show_original = (
        title_col == 'enriched_title'
        and 'title' in filtered_df.columns
        and (
            filtered_df['title'].fillna('').astype(str)
            != filtered_df['enriched_title'].fillna('').astype(str)
        ).any()
    )
    if show_original:
        display_cols.append('title')
    # 🎯 BOLO match column — only added when at least one row in the
    # current auction matched the brand list. Shows the brand name
    # (with tier emoji) so the user can scan for premium hits without
    # opening individual lots. The target-buy ceiling lives next to
    # current_bid so the comparison is one glance.
    if bolo_total > 0 and 'bolo_brand' in filtered_df.columns:
        display_cols.append('bolo_brand')
    # CPU Form column — only show when there's at least one CPU /
    # mini_pc category match in the table. Distinguishes bare chips
    # from systems-with-CPU at a glance (very different comp tiers).
    has_cpu_match = (
        'cpu_form_factor' in filtered_df.columns
        and filtered_df['cpu_form_factor'].notna().any()
    )
    if has_cpu_match:
        display_cols.append('cpu_form_factor')
        # bolo_model surfaces the specific chip identifier (e.g.
        # "i5-8500T", "Xeon Gold 6138"). For 'system' rows it tells
        # the user which CPU is inside the laptop/desktop — useful for
        # judging whether the chip itself has resale value as a floor.
        if 'bolo_model' in filtered_df.columns:
            display_cols.append('bolo_model')
    # 🛡️ Auth Score column — only when the auction has at least one
    # auth-required (tier-3 luxury / Polo subline) BOLO match. Below
    # 30 is "do not bid without authentication"; surfaces the same
    # red/green flag detail in the column tooltip.
    auth_required_count = (
        int(filtered_df['bolo_auth_required'].fillna(False).astype(bool).sum())
        if 'bolo_auth_required' in filtered_df.columns else 0
    )
    if auth_required_count > 0 and 'bolo_auth_score' in filtered_df.columns:
        display_cols.append('bolo_auth_score')
    # Multi-auction BOLO scan: when lots come from many auctions,
    # surface the auction name + a per-row link so the user can
    # navigate to each lot's source. Single-auction views keep the
    # one-time auction link at the top of the page (cleaner).
    is_multi_auction_view = (
        'auction' in filtered_df.columns
        and int(filtered_df['auction'].nunique()) > 1
    )
    if is_multi_auction_view:
        if 'auction' in filtered_df.columns:
            display_cols.append('auction')
        if 'auction_link' in filtered_df.columns:
            display_cols.append('auction_link')
    display_cols += ['category', 'current_bid']
    if 'next_bid' in filtered_df.columns:
        display_cols.append('next_bid')
    display_cols.append('est_cost')
    # Target-buy ceiling sits next to est_cost so the user can see at a
    # glance whether the current bid has already crossed the BOLO's
    # recommended max. Only shown when matches exist.
    if bolo_total > 0 and 'bolo_target_buy_high' in filtered_df.columns:
        display_cols.append('bolo_target_buy_high')
    col_config = {
        "thumbnail_url": st.column_config.ImageColumn(
            "📷",
            help="Lot photo from HiBid (click to enlarge).",
            width="small",
        ),
        "enriched_title": st.column_config.TextColumn("Title (Enriched)"),
        "title": st.column_config.TextColumn(
            "Original Title",
            width="medium",
            help="The HiBid lot title before enrichment. Step 1 (audit) "
                 "and Step 1.5 (image match) can rewrite the title with "
                 "brand / model / year details pulled from the "
                 "description or matching eBay listings — this column "
                 "preserves what the lot actually said on HiBid.",
        ),
        "lot_link": st.column_config.LinkColumn(
            "🔗",
            display_text="Open",
            width="small",
            pinned=True,  # always visible during horizontal scroll
            help="Open this lot on HiBid in a new tab.",
        ),
        "buy_grade": st.column_config.TextColumn(
            "Buy?",
            width="small",
            help="Composite 'Should I buy it?' grade:\n\n"
                 "🟢 A (80+) — strong buy, clean comp & margin\n"
                 "🟡 B (65-79) — solid\n"
                 "🟠 C (50-64) — borderline\n"
                 "🔴 D (35-49) — risky\n"
                 "🔍 ? — uncomped but market signals present "
                 "(BOLO match, active bidding, auctioneer estimate) "
                 "— manual research candidate\n"
                 "⚫ F — comped & confirmed bad (margin negative, "
                 "past max bid, red-flagged)\n"
                 "❓ - — uncomped with no market signal — can't tell",
        ),
        "buy_score": st.column_config.ProgressColumn(
            "Score", min_value=0, max_value=100, format="%d",
            help="0-100 composite score: margin (≤45) + bid "
                 "headroom (≤20) + STR (≤15) + comp confidence "
                 "(≤20). 80+ = A, 65+ = B, 50+ = C, 35+ = D, "
                 "below = F.",
        ),
        "auction_link": st.column_config.LinkColumn("Auction", display_text="Open"),
        "current_bid": st.column_config.NumberColumn("Current Bid", format="$%.2f"),
        "next_bid": st.column_config.NumberColumn(
            "Next Bid", format="$%.2f",
            help="Minimum next acceptable bid (HiBid `lotState.minBid`). "
                 "Used as the cost-basis floor when current bid is $0.",
        ),
        "est_cost": st.column_config.NumberColumn(
            "Est. Cost", format="$%.2f",
            help="max(current_bid, next_bid) + buyer_premium_pct. "
                 "This is what you'd pay if you won at the minimum next bid.",
        ),
        "bid_count": st.column_config.NumberColumn("Bids", format="%d"),
        "bolo_brand": st.column_config.TextColumn(
            "🎯 BOLO",
            help="Brand match against data/clothing_brand_bolo.json. "
                 "Tier 1 = high sell-through, Tier 2 = vintage/heritage, "
                 "Tier 3 = luxury/situational. Hover the row's "
                 "bolo_target_buy_high to see the recommended max bid.",
        ),
        "bolo_target_buy_high": st.column_config.NumberColumn(
            "Target Buy ≤", format="$%.0f",
            help="Highest price you should pay. Computed as MIN(BOLO "
                 "category guideline, comp-derived max_bid for THIS "
                 "lot at target ROI). BLANK = don't bid — shipping + "
                 "premium eat the resale margin at target ROI. Common "
                 "on low-resale lots from remote-ship auctions.",
        ),
        "bolo_tier": st.column_config.NumberColumn(
            "Tier", format="%d",
            help="1 = athleisure / outdoor / boho contemporary "
                 "(reliable sell-through). 2 = vintage denim / heritage. "
                 "3 = luxury / sneakers / high-volume mid-luxury "
                 "(highest ceiling, lowest STR).",
        ),
        "cpu_form_factor": st.column_config.TextColumn(
            "CPU Form",
            help="For CPU-category lots: 'bare_cpu' = chip alone "
                 "(comps to bare-CPU sold-listings, ~$15-150 typical). "
                 "'system' = laptop/desktop/server containing the CPU "
                 "(comps to system sold-listings, ~$70-300 typical — "
                 "but the bare-CPU value floor is still useful intel "
                 "if you can harvest the chip). 'unclear' = ambiguous "
                 "title; treat as bare-CPU but inspect manually.",
        ),
        "bolo_model": st.column_config.TextColumn(
            "Model",
            help="Specific BOLO model identifier matched in the title "
                 "(e.g. 'i5-8500T', 'Xeon Gold 6138', 'Raspberry Pi 4 "
                 "Model B'). For 'system' form-factor rows, this is "
                 "the CPU inside the laptop/desktop — handy for "
                 "judging whether the chip alone has resale value "
                 "even if the system itself isn't a primary target.",
        ),
        "bolo_auth_score": st.column_config.ProgressColumn(
            "🛡️ Auth",
            min_value=0, max_value=100, format="%d",
            help="Description-based authenticity score (0-100) for "
                 "auth-required BOLO matches. <30 = do not bid without "
                 "authentication (auctioneer used 'as-is' or 'designer-"
                 "style' or similar disclaimer). 30-60 = ambiguous, "
                 "inspect carefully. 60+ = description supports "
                 "authenticity (made-in tag cited, dust bag mentioned, "
                 "estate provenance, etc.). Resale confidence is "
                 "automatically discounted when this score is low.",
        ),
    }

    if 'est_resale' in filtered_df.columns:
        display_cols += ['est_resale']
        col_config["est_resale"] = st.column_config.NumberColumn("Est. Resale (median)", format="$%.2f")

        # Net-of-eBay-fees profit per row, computed earlier in
        # _render_results_table. Inserted right after est_resale so
        # the user reads "Est. Resale → Est. Profit" left-to-right.
        if 'est_profit' in filtered_df.columns:
            display_cols += ['est_profit']
            col_config["est_profit"] = st.column_config.NumberColumn(
                "Est. Profit (net)",
                format="$%.2f",
                help="est_resale × (1 - 13.25% eBay fees) - $0.30 - est_cost. "
                     "What would actually hit your bank after eBay's take.",
            )

        if 'price_low' in filtered_df.columns:
            display_cols += ['price_low', 'price_high']
            col_config["price_low"] = st.column_config.NumberColumn("Low (25%)", format="$%.2f")
            col_config["price_high"] = st.column_config.NumberColumn("High (75%)", format="$%.2f")

        if 'ebay_comps' in filtered_df.columns:
            display_cols += ['ebay_comps']
            col_config["ebay_comps"] = st.column_config.NumberColumn("eBay Comps", format="%d")

            # Numerical resale-confidence score (0-100). Replaces the
            # earlier binary low_comp_confidence flag with a richer
            # signal — see _compute_resale_confidence() for the formula
            # (source quality 50%, sample size 30%, spread tightness
            # 20%). Display as a ProgressColumn so the bar visualizes
            # the score at a glance.
            filtered_df = filtered_df.copy()
            filtered_df['resale_confidence'] = filtered_df.apply(
                _compute_resale_confidence, axis=1,
            )
            display_cols += ['resale_confidence']
            col_config["resale_confidence"] = st.column_config.ProgressColumn(
                "Conf.",
                min_value=0,
                max_value=100,
                format="%d%%",
                help=(
                    "Trust score for est_resale on a 0-100 scale. "
                    "Combines source quality (50% weight: eBay/Mercari "
                    "sold > PriceCharting > active fallback), sample "
                    "size (30%; PC's single match counts ~5-comp since "
                    "it's server-aggregated), and spread tightness "
                    "(20%; wider (high-low)/median means outliers). "
                    "Rough thresholds: >=80 high, 50-79 medium, <50 "
                    "rough estimate at best."
                ),
            )
        if 'mercari_comps' in filtered_df.columns:
            display_cols += ['mercari_comps']
            col_config["mercari_comps"] = st.column_config.NumberColumn("Mercari Comps", format="%d")
        if 'pricecharting_comps' in filtered_df.columns:
            display_cols += ['pricecharting_comps']
            col_config["pricecharting_comps"] = st.column_config.NumberColumn(
                "PC Hit", format="%d",
                help="1 = PriceCharting matched this lot to a canonical product.",
            )
        if 'price_source' in filtered_df.columns:
            display_cols += ['price_source']
            col_config["price_source"] = st.column_config.TextColumn("Price Src")

        # ----- Auctioneer-estimate fact-check -----
        # When the auctioneer published an estimate range, surface it
        # alongside our comp-derived est_resale and flag rows where the
        # comp is wildly outside the auctioneer's range. Wrapped in
        # try/except so a bad payload (mixed dtypes, weird estimate
        # strings) can't blank the whole results table.
        try:
            has_est_low = 'auctioneer_est_low' in filtered_df.columns
            has_est_high = 'auctioneer_est_high' in filtered_df.columns
            if has_est_low or has_est_high:
                filtered_df = filtered_df.copy()
                est_low = (pd.to_numeric(filtered_df['auctioneer_est_low'], errors='coerce')
                           if has_est_low else pd.Series(np.nan, index=filtered_df.index, dtype='float64'))
                est_high = (pd.to_numeric(filtered_df['auctioneer_est_high'], errors='coerce')
                            if has_est_high else pd.Series(np.nan, index=filtered_df.index, dtype='float64'))
                resale_n = pd.to_numeric(filtered_df['est_resale'], errors='coerce')
                # Convert to plain numpy arrays — element-wise math on
                # mixed-dtype Series can hit "Expected numeric dtype, got
                # object" the same way other parts of this app did.
                est_low_arr = est_low.to_numpy(dtype='float64', na_value=np.nan)
                est_high_arr = est_high.to_numpy(dtype='float64', na_value=np.nan)
                resale_arr = resale_n.to_numpy(dtype='float64', na_value=np.nan)

                verdicts = []
                for low, high, resale in zip(est_low_arr, est_high_arr, resale_arr):
                    if np.isnan(resale):
                        verdicts.append('')
                        continue
                    low_ok = not np.isnan(low)
                    high_ok = not np.isnan(high)
                    if not low_ok and not high_ok:
                        verdicts.append('—')
                        continue
                    lo = low if low_ok else high
                    hi = high if high_ok else low
                    if lo > 0 and resale < lo * 0.5:
                        verdicts.append('⚠ comp <50% of est')
                    elif hi > 0 and resale > hi * 2.0:
                        verdicts.append('⚠ comp >2× est')
                    elif (lo <= 0 or resale >= lo * 0.8) and (hi <= 0 or resale <= hi * 1.5):
                        verdicts.append('✓ in range')
                    else:
                        verdicts.append('near range')

                filtered_df['comp_vs_est'] = verdicts
                if has_est_low:
                    display_cols += ['auctioneer_est_low']
                    col_config['auctioneer_est_low'] = st.column_config.NumberColumn(
                        "Auct. Low", format="$%.2f",
                        help="Auctioneer's published low estimate.",
                    )
                if has_est_high:
                    display_cols += ['auctioneer_est_high']
                    col_config['auctioneer_est_high'] = st.column_config.NumberColumn(
                        "Auct. High", format="$%.2f",
                        help="Auctioneer's published high estimate.",
                    )
                display_cols += ['comp_vs_est']
                col_config['comp_vs_est'] = st.column_config.TextColumn(
                    "Comp vs. Est.",
                    help=(
                        "Sanity check between the comp-derived est_resale "
                        "and the auctioneer's published estimate range. "
                        "'⚠' rows have comp values that disagree strongly "
                        "— verify before bidding."
                    ),
                )
        except Exception as _est_err:
            st.caption(f"_(comp-vs-estimate fact-check skipped: {_est_err})_")

        display_cols += ['est_roi']
        col_config["est_roi"] = st.column_config.NumberColumn(
            "ROI %", format="%.0f%%",
            help="ROI based on CURRENT bid + cost. Misleading on luxury "
                 "lots where bidding hasn't opened (see Realistic ROI %).",
        )

    # Realistic ROI based on max(est_cost, auctioneer_est_low × 0.5).
    # Fairer for luxury-watch lots starting at $5 with $16K estimates.
    if 'realistic_roi' in filtered_df.columns:
        display_cols.append('realistic_roi')
        col_config["realistic_roi"] = st.column_config.NumberColumn(
            "Realistic ROI %", format="%.0f%%",
            help="ROI computed against MAX(est_cost, auctioneer_est_low × 0.5). "
                 "On lots already bid up to estimate range, this matches ROI %; "
                 "on luxury starter-bid lots it gives a much more honest picture.",
        )

    # Bid-trap warning marker
    if 'bid_trap_warn' in filtered_df.columns:
        display_cols.append('bid_trap_warn')
        col_config["bid_trap_warn"] = st.column_config.CheckboxColumn(
            "⚠️",
            help="Starter-bid trap: luxury-watch / high-end jewelry lot "
                 "with current bid < 10% of auctioneer estimate AND "
                 "estimate > $1000. Displayed ROI is inflated; the lot "
                 "will clear closer to the auctioneer's estimate. "
                 "Sort puts these BELOW real opportunities.",
        )

    if 'max_bid' in filtered_df.columns:
        display_cols.append('max_bid')
        col_config["max_bid"] = st.column_config.NumberColumn("Max Bid", format="$%.2f")

    if 'ebay_str' in filtered_df.columns:
        display_cols.append('ebay_str')
        col_config["ebay_str"] = st.column_config.ProgressColumn("STR %", min_value=0, max_value=100, format="%.0f%%")

    display_cols.append('bid_count')

    if 'verdict' in filtered_df.columns:
        display_cols += ['verdict', 'confidence', 'red_flag']
        col_config["verdict"] = st.column_config.TextColumn(
            "Flag Reason",
            help=(
                "Why the audit flagged the item. "
                "'Unshippable (HARD logistics)' = pickup-only or oversized. "
                "'broken, damaged, or for parts' = AI flagged the description "
                "as condition risk. 'untested or unknown condition' = "
                "description suggests the seller couldn't verify it works. "
                "'Unknown' = description was empty or too short to classify."
            ),
        )
        col_config["confidence"] = st.column_config.ProgressColumn(
            "Audit Conf.",
            min_value=0, max_value=100, format="%.1f%%",
            help="The AI condition-audit's confidence in its verdict "
                 "(see 'Flag Reason' column). Distinct from the "
                 "'Resale Conf.' column which scores trust in the "
                 "est_resale price. Often 0 for collectible-bypass "
                 "lots (cards/comics/games) where the audit is "
                 "skipped on purpose.",
        )
        col_config["red_flag"] = st.column_config.CheckboxColumn("Red Flag")

    final_cols = [c for c in display_cols if c in filtered_df.columns]

    # --- Column show/hide controls ---
    # Auto-detect columns that have no useful data on this auction
    # (all-null or all-zero) and default-hide them. The user can
    # toggle visibility per column from the expander above the table.
    # Common case this addresses: coin auctions where STR scraping
    # returns nothing, or audit was bypassed so 'confidence' is 0.
    def _column_is_empty(col: str) -> bool:
        if col not in filtered_df.columns:
            return True
        s = filtered_df[col]
        if s.isna().all():
            return True
        # Numeric-like check: treat all-zero as empty too. Don't apply
        # to text columns where '0' could be a real category label.
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            try:
                numeric = pd.to_numeric(s, errors='coerce').fillna(0)
                if (numeric == 0).all():
                    return True
            except (TypeError, ValueError):
                pass
        return False

    # Friendly labels for the checkbox UI. Falls back to the column
    # name when not specified.
    _col_labels = {
        'thumbnail_url': '📷 Image',
        'enriched_title': 'Title (Enriched)',
        'title': 'Original Title',
        'lot_link': 'Item link',
        'auction_link': 'Auction link',
        'category': 'Category',
        'current_bid': 'Current Bid',
        'next_bid': 'Next Bid',
        'est_cost': 'Est. Cost',
        'est_resale': 'Est. Resale',
        'price_low': 'Price Low',
        'price_high': 'Price High',
        'comp_count': 'Comp Count',
        'ebay_comps': 'eBay Comps',
        'mercari_comps': 'Mercari Comps',
        'pricecharting_comps': 'PC Hit',
        'gocollect_comps': 'GoCollect Hit',
        'price_source': 'Price Source',
        'resale_confidence': 'Resale Conf.',
        'ebay_str': 'STR %',
        'str_source': 'STR Source',
        'bid_count': 'Bids',
        'verdict': 'Flag Reason',
        'confidence': 'Audit Conf.',
        'red_flag': 'Red Flag',
        'est_roi': 'ROI %',
        'max_bid': 'Max Bid',
        'low_comp_confidence': 'Low Conf. (legacy)',
        'bolo_brand': '🎯 BOLO Brand',
        'bolo_tier': '🎯 BOLO Tier',
        'bolo_target_buy_high': '🎯 Target Buy',
        'bolo_model': '🎯 BOLO Model',
        'bolo_auth_score': '🛡️ Auth Score',
        'bolo_auth_red': '🛡️ Auth Red Flags',
        'bolo_auth_green': '🛡️ Auth Green Flags',
    }
    # Persist user picks across reruns. Key includes a fingerprint of
    # the current column set so a different auction's columns don't
    # inherit a stale picks set.
    _empty_cols = {c: _column_is_empty(c) for c in final_cols}
    n_empty = sum(_empty_cols.values())
    n_visible = len(final_cols) - n_empty
    with st.expander(
        f"📋 Show / hide columns "
        f"({n_visible} visible · {n_empty} auto-hidden as empty)",
        expanded=False,
    ):
        st.caption(
            "Empty columns (all-null or all-zero) are auto-hidden by "
            "default. Tick a checkbox to force-show one anyway."
        )
        cb_cols = st.columns(4)
        kept_cols = []
        for i, c in enumerate(final_cols):
            with cb_cols[i % 4]:
                empty = _empty_cols[c]
                label = _col_labels.get(c, c)
                if empty:
                    label = f"{label}  _(empty)_"
                # Default: hide if empty, show otherwise. The state
                # key is per-column so picks persist across reruns
                # but reset cleanly between auctions (different
                # current_auction → different audit_results df shape).
                shown = st.checkbox(
                    label,
                    value=not empty,
                    key=f"colshow_{c}",
                )
                if shown:
                    kept_cols.append(c)
        final_cols = kept_cols

    if not final_cols:
        st.info("All columns hidden — tick at least one in the expander above.")
        return

    display_df = filtered_df[final_cols].copy()

    display_df = display_df.reset_index(drop=True)

    # Row coloring is now driven by `buy_grade` (composite score). The
    # old ROI/STR-threshold green/yellow tier system was removed — it
    # fragmented signal across two sliders and produced misleading
    # "all green" calls on contaminated comps. Buy_grade absorbs all
    # the threshold tuning into one number with a familiar A-F scale.
    _GRADE_BG = {
        '🟢 A': 'background-color: rgba(46, 204, 113, 0.28)',   # green
        '🟡 B': 'background-color: rgba(241, 196, 15, 0.22)',   # yellow
        '🟠 C': 'background-color: rgba(230, 126, 34, 0.18)',   # orange
        '🔴 D': 'background-color: rgba(231, 76, 60, 0.18)',    # red
        '🔍 ?': 'background-color: rgba(52, 152, 219, 0.18)',   # blue — manual research
        '⚫ F': '',                                                # no tint
        '❓ -': 'color: rgba(127, 140, 141, 0.7)',                # grey-out
    }

    def _row_style(row):
        # buy_grade column may be missing on legacy frames or when
        # the comp pipeline hasn't run yet — default to no styling.
        if 'buy_grade' not in display_df.columns:
            return [''] * len(row)
        grade = display_df.at[row.name, 'buy_grade']
        css = _GRADE_BG.get(grade, '')
        return [css] * len(row)

    styled = display_df.style.apply(_row_style, axis=1)
    st.dataframe(
        styled,
        width='stretch',
        column_config=col_config,
    )

    # --- Export controls: share results for debugging ---
    # Two formats. CSV is the full thing (every column, attach as a file
    # in chat or open in a spreadsheet). The Markdown snippet is a
    # paste-friendly view of the visible columns + a sample of rows —
    # useful when only a few lots look wrong and you want to ask a
    # specific question without sharing the whole sheet.
    with st.expander("📤 Export results", expanded=False):
        auction_slug = re.sub(
            r'[^a-z0-9]+', '-',
            (st.session_state.get('current_auction') or 'auction').lower(),
        ).strip('-')[:50] or 'auction'
        timestamp = datetime.now().strftime('%Y%m%d-%H%M')
        csv_name = f"htown-results-{auction_slug}-{timestamp}.csv"

        # Single Full CSV download — all rows + all columns. Use the
        # styled-table's working df so est_roi / max_bid reflect the
        # current target_roi setting.
        st.download_button(
            label=f"📥 Full CSV ({len(working)})",
            data=working.to_csv(index=False).encode('utf-8'),
            file_name=csv_name,
            mime="text/csv",
            key="export_results_csv",
            help="All rows + all columns (verdict, comps, ROI, max_bid, "
                 "links, image-enrichment fields).",
            width='stretch',
        )

        # Markdown-snippet row config (was row 2 before the green/yellow
        # buttons came out — kept the same widget below).
        snippet_n = st.number_input(
            "Markdown snippet rows", min_value=5, max_value=100, step=5,
            value=20, key="export_snippet_n",
            help="How many rows to include in the markdown snippet below.",
        )
        st.caption(
            "💡 If specific lots look wrong, share the Markdown snippet. "
            "If you want me to look at the whole auction, attach a CSV."
        )

        snippet_cols = [
            c for c in (
                'title', 'enriched_title', 'category', 'current_bid',
                'est_resale', 'price_low', 'price_high',
                'ebay_comps', 'mercari_comps', 'pricecharting_comps',
                'price_source', 'est_roi', 'ebay_str',
                'verdict', 'red_flag', 'audit_source',
                'img_source', 'lot_link',
            ) if c in working.columns
        ]
        snippet_df = working[snippet_cols].head(int(snippet_n)).copy()
        # Trim long titles so the markdown table doesn't get unreadable
        for col in ('title', 'enriched_title'):
            if col in snippet_df.columns:
                snippet_df[col] = (
                    snippet_df[col].fillna('').astype(str).str.slice(0, 60)
                )
        try:
            md_table = snippet_df.to_markdown(index=False)
        except (ImportError, AttributeError):
            # `tabulate` package may not be installed — fall back to CSV-as-code
            md_table = "```csv\n" + snippet_df.to_csv(index=False) + "```"

        st.code(md_table, language="markdown")


# ================================================================
# SINGLE-PAGE UI: switches between Discovery view and Analysis view
# ================================================================
# Auto-skip the per-auction Discovery results page: when lots have just
# been fetched (phase1_leads filled) but no auction has been loaded into
# analysis yet, jump straight into the analysis view for the fetched
# auction. The picker only fetches one auction per click, so this is
# unambiguous.
if (
    not st.session_state.phase1_leads.empty
    and not st.session_state.get('current_auction')
):
    _df = st.session_state.phase1_leads

    # Multi-auction BOLO-scan path: filter every fetched lot
    # against the BOLO matcher and load just the matches as a
    # synthetic "🎯 BOLO scan" auction. The standard analysis
    # pipeline (audit → comps → results table) takes over from
    # there — same code path, just with a pre-filtered subset.
    #
    # IMPORTANT: do NOT gate this on `_BOLO_MATCHER.loaded`. After the
    # lazy-loading change, `.loaded` returns False until the matcher
    # is first accessed — and on the rerun after fetch completes, it
    # hasn't been touched yet in this script run. Gating on it here
    # caused the scan path to silently skip and fall through to the
    # single-auction fallback below, narrowing a multi-auction scan
    # down to just the first auction's lots. The matcher builds
    # itself the moment `.brand_count` / `.match()` is called inside
    # this block (and `_bolo_scan_all_pending` is the authoritative
    # signal that we WANT to do BOLO matching).
    if st.session_state.get('_bolo_scan_all_pending'):
        # Stage 2 of the multi-auction BOLO scan: regex-match every fetched
        # lot title/description against the loaded BOLO list. Cheap (regex
        # only), but we surface a visible status panel so the user knows
        # WHY the page seems to pause between fetch and audit.
        with st.status(
            "🎯 Matching lots against BOLO list…",
            expanded=True,
        ) as _bolo_match_status:
            # Pre-filter BEFORE running the BOLO match. Three passes:
            #   1. HARD-logistics drop
            #   2. Easy-ship-only drop (NEUTRAL lots whose titles don't
            #      hit the mailbox-shippable keyword set)
            #   3. Bid-cap drop (next-bid > comps_skip_above_bid)
            # The match is regex-only and not slow per-lot, but at 14K+
            # fetched lots dropping unactionable rows up front cuts the
            # scan time proportionally and skips spending audit + comp
            # credits on lots we'd reject anyway. Keep the original
            # count so the status panel can show the savings.
            n_total_pre = len(_df)
            _df, _ship_reasons = _apply_easy_ship_filter(_df)
            _df, _bid_reason = _apply_bid_cap_filter(_df)
            _all_reasons = list(_ship_reasons)
            if _bid_reason:
                _all_reasons.append(_bid_reason)
            if _all_reasons:
                _dropped_n = n_total_pre - len(_df)
                st.write(
                    f"📦 Pre-filtered to **{len(_df):,}** actionable lots "
                    f"(dropped {_dropped_n:,}: "
                    f"{', '.join(_all_reasons)})."
                )
            n_total = len(_df)
            n_brands = _BOLO_MATCHER.brand_count
            st.write(
                f"Scanning **{n_total:,}** fetched lots against "
                f"**{n_brands}** BOLO brands…"
            )
            # Chunked match with a live progress bar + running brand
            # tally + a live pie chart. Chunk size 2000 gives ~20
            # progress ticks for a 40K-lot scan — visible motion without
            # spamming reruns.
            _bolo_progress = st.progress(0.0, text="Starting…")
            _bolo_eta_label = st.empty()
            _bolo_live_stats = st.empty()
            # ETA tracking: stamp the start time before the first chunk
            # callback fires. Each chunk reports elapsed seconds and a
            # linear projection of the remaining time. Linear projection
            # is fine here because regex match cost is essentially flat
            # per row — there's no warm-up curve like there is for
            # ScrapingBee.
            _bolo_start_ts = datetime.now()

            def _fmt_secs(s: float) -> str:
                if s < 1:
                    return "<1s"
                if s < 60:
                    return f"{s:.0f}s"
                m, s = divmod(int(s), 60)
                return f"{m}m {s:02d}s"

            def _bolo_chunk_cb(current, total, hits, top_brands):
                pct = current / total if total else 1.0
                _elapsed = (datetime.now() - _bolo_start_ts).total_seconds()
                # Project remaining time from rate-so-far. Guard against
                # divide-by-zero on the first chunk (current can equal 0
                # in pathological cases).
                _rate = current / max(_elapsed, 0.001)  # rows/sec
                _remaining = (total - current) / _rate if _rate > 0 else 0
                _bolo_progress.progress(
                    min(pct, 1.0),
                    text=f"Matching… {current:,} / {total:,} lots scanned "
                         f"· {hits:,} BOLO hits so far",
                )
                # Separate line for elapsed + ETA so the progress bar
                # text stays readable. Once we're at 99%+ progress, the
                # ETA isn't meaningful — show "wrapping up…" instead.
                if pct >= 0.99:
                    _bolo_eta_label.markdown(
                        f"⏱️ Elapsed: **{_fmt_secs(_elapsed)}** · "
                        f"wrapping up…"
                    )
                else:
                    _bolo_eta_label.markdown(
                        f"⏱️ Elapsed: **{_fmt_secs(_elapsed)}** · "
                        f"ETA remaining: **{_fmt_secs(_remaining)}** · "
                        f"rate: ~{_rate:,.0f} lots/sec"
                    )
                if top_brands:
                    _summary = " · ".join(
                        f"**{b}** ({c})" for b, c in top_brands.items()
                    )
                    _bolo_live_stats.markdown(
                        f"🔥 Top brands so far: {_summary}"
                    )

            _scan_df = _compute_bolo_columns_chunked(
                _df, chunk_size=2000, progress_callback=_bolo_chunk_cb,
            )
            # Stash the final elapsed time so we can show "completed
            # in Xs" in the closing summary instead of just "done".
            _bolo_total_secs = (
                datetime.now() - _bolo_start_ts
            ).total_seconds()
            _bolo_progress.empty()
            _bolo_eta_label.empty()
            _bolo_live_stats.empty()

            if 'bolo_brand' in _scan_df.columns:
                _bolo_subset = _scan_df[_scan_df['bolo_brand'].notna()].copy()
            else:
                _bolo_subset = _scan_df.iloc[0:0]
            n_matches = len(_bolo_subset)
            n_auctions = (
                int(_df['auction'].nunique())
                if 'auction' in _df.columns else 1
            )
            n_match_auctions = (
                int(_bolo_subset['auction'].nunique())
                if 'auction' in _bolo_subset.columns and not _bolo_subset.empty
                else 0
            )
            _match_pct = (100 * n_matches / n_total) if n_total else 0.0
            tlog("BOLO",
                 f"matched {n_matches:,}/{n_total:,} lots ({_match_pct:.1f}%)",
                 f"in {_bolo_total_secs:.1f}s",
                 f"· {n_match_auctions}/{n_auctions} auctions hit")
            st.write(
                f"✅ **{n_matches:,}** BOLO matches found "
                f"({_match_pct:.1f}% hit rate) across "
                f"**{n_match_auctions}** of **{n_auctions}** auctions."
            )
            if n_matches > 0:
                _top_brands = (
                    _bolo_subset['bolo_brand']
                    .value_counts()
                    .head(10)
                    .to_dict()
                )
                _brand_summary = " · ".join(
                    f"**{b}** ({c})" for b, c in _top_brands.items()
                )
                st.write(f"Top brands hit: {_brand_summary}")

                # Per-category breakdown so the user knows what tiers are coming
                if 'bolo_category' in _bolo_subset.columns:
                    _cat_breakdown = (
                        _bolo_subset['bolo_category']
                        .fillna('(uncategorized)')
                        .value_counts()
                        .head(8)
                        .to_dict()
                    )
                    _cat_summary = " · ".join(
                        f"{c} ({n})" for c, n in _cat_breakdown.items()
                    )
                    st.write(f"By category: {_cat_summary}")

                # Top auctions by BOLO match count — surface where the
                # densest BOLO-friendly inventory lives
                if 'auction' in _bolo_subset.columns:
                    _top_auctions = (
                        _bolo_subset['auction']
                        .value_counts()
                        .head(5)
                        .to_dict()
                    )
                    _auc_summary = " · ".join(
                        f"*{a[:40]}* ({n})" for a, n in _top_auctions.items()
                    )
                    st.write(f"🏆 Densest BOLO auctions: {_auc_summary}")

                st.write(
                    "Loading the matched subset into the analysis view; "
                    "audit + price comps run next."
                )
            else:
                st.write(
                    "No BOLO matches in this batch. Check that BOLO files "
                    "are loaded and that auction titles look reasonable."
                )
            _bolo_match_status.update(
                label=(
                    f"✅ {n_matches:,} BOLO matches "
                    f"across {n_match_auctions} auctions "
                    f"(completed in {_fmt_secs(_bolo_total_secs)})"
                ),
                state="complete", expanded=False,
            )

        # ================================================================
        # 🕵️ SLEEPER HUNT (automatic stage of every BOLO scan)
        # Container-titled lots the text matcher can't see into
        # ("Jewelry Box", "Lot of Toys"). Candidates score on free
        # signals; the top N get vision enrichment (eBay image search
        # first — free; Claude Haiku vision fallback — ~$0.005/lot).
        # Enriched titles re-run through the BOLO matcher and the
        # melt detector; hits merge into the results tagged 🕵️.
        # Non-fatal by design — any failure logs and the scan
        # proceeds with the text-only matches.
        # ================================================================
        _sleeper_hits = pd.DataFrame()
        _sleeper_cands_n = 0
        if (st.session_state.get('sleeper_hunt_enabled', True)
                and 'bolo_brand' in _scan_df.columns):
            try:
                with st.status(
                    "🕵️ Sleeper hunt — reading photos of container lots…",
                    expanded=True,
                ) as _sl_status:
                    _unmatched = _scan_df[_scan_df['bolo_brand'].isna()]
                    _cands = _score_sleeper_candidates(_unmatched)
                    _cap = int(st.session_state.get('sleeper_max_lots', 300) or 300)
                    _sleeper_cands_n = len(_cands)
                    _cands = _cands.head(_cap)
                    st.write(
                        f"**{_sleeper_cands_n}** container-titled lots with "
                        f"treasure signals; vision-reading the top "
                        f"**{len(_cands)}** (cap {_cap})."
                    )
                    if not _cands.empty and 'thumbnail_url' in _cands.columns:
                        from scraper.vision_enrich import EbayImageEnricher
                        from scraper.config_loader import load_config
                        _cfg = load_config()
                        _enricher = EbayImageEnricher(
                            _cfg['ebay']['app_id'], _cfg['ebay']['cert_id'],
                            anthropic_api_key=(
                                (_cfg.get('anthropic') or {}).get('api_key')
                            ),
                            gemini_api_key=(
                                (_cfg.get('gemini') or {}).get('api_key')
                            ),
                            vision_provider=str(
                                st.session_state.get('vision_provider', 'claude')
                            ).lower(),
                        )
                        _sl_prog = st.progress(0.0, text="Reading photos…")
                        from concurrent.futures import (
                            ThreadPoolExecutor as _SlPool,
                            as_completed as _sl_done,
                        )

                        def _do_enrich(idx, url, ot):
                            try:
                                r = _enricher.enrich_one(url, original_title=ot)
                                return idx, r.get('img_enriched_title')
                            except Exception:
                                return idx, None

                        _enriched_titles = {}
                        _n_done = 0
                        _sl_t0 = _time.time()
                        with _SlPool(max_workers=8) as _ex:
                            _futs = [
                                _ex.submit(
                                    _do_enrich, idx,
                                    str(_cands.at[idx, 'thumbnail_url'] or ''),
                                    str(_cands.at[idx, 'title'] or ''),
                                )
                                for idx in _cands.index
                            ]
                            for _f in _sl_done(_futs):
                                idx, _et = _f.result()
                                _n_done += 1
                                if _et:
                                    _enriched_titles[idx] = _et
                                _sl_prog.progress(
                                    _n_done / len(_futs),
                                    text=(
                                        f"Reading photos… {_n_done}/{len(_futs)}"
                                        f" · {len(_enriched_titles)} identified"
                                    ),
                                )
                        _sl_prog.empty()
                        tlog("SLEEPER",
                             f"enriched {len(_enriched_titles)}/{len(_cands)}",
                             f"in {_time.time() - _sl_t0:.1f}s")
                        if _enriched_titles:
                            # Re-match on the ENRICHED text: swap titles on
                            # a copy, run the standard BOLO column pass
                            # (reuses disqualifiers / guards / auth), then
                            # restore original titles and keep the vision
                            # text in enriched_title.
                            _re_rows = _cands.loc[list(_enriched_titles)].copy()
                            _orig_titles = _re_rows['title'].copy()
                            _re_rows['title'] = pd.Series(_enriched_titles)
                            _re_rows = _compute_bolo_columns(_re_rows)
                            _hit_mask = (
                                _re_rows['bolo_brand'].notna()
                                if 'bolo_brand' in _re_rows.columns
                                else pd.Series(False, index=_re_rows.index)
                            )
                            # Melt check on the vision text — a "Jewelry
                            # Box" that vision reads as sterling flatware
                            # is a hit even without a brand.
                            _melt_idx = []
                            for _mi in _re_rows.index:
                                _mv, _metal, _g = _estimate_melt_value(
                                    str(_re_rows.at[_mi, 'title']),
                                    str(_re_rows.at[_mi, 'description'] or '')
                                    if 'description' in _re_rows.columns else '',
                                )
                                if _mv and _mv >= 40:
                                    _melt_idx.append(_mi)
                            _keep = _re_rows.index[_hit_mask].union(
                                pd.Index(_melt_idx)
                            )
                            _sleeper_hits = _re_rows.loc[_keep].copy()
                            if not _sleeper_hits.empty:
                                _sleeper_hits['enriched_title'] = (
                                    _sleeper_hits['title']
                                )
                                _sleeper_hits['title'] = _orig_titles.loc[_keep]
                                _sleeper_hits['sleeper_hit'] = True
                                _preview = " · ".join(
                                    f"*{str(t)[:40]}* → "
                                    f"{str(_sleeper_hits.at[i, 'enriched_title'])[:45]}"
                                    for i, t in
                                    _sleeper_hits['title'].head(4).items()
                                )
                                st.write(
                                    f"🕵️ **{len(_sleeper_hits)}** sleeper "
                                    f"hit(s) — value visible only in "
                                    f"photos: {_preview}"
                                )
                    _sl_status.update(
                        label=(
                            f"🕵️ Sleeper hunt: {len(_sleeper_hits)} hit(s) "
                            f"from {min(_sleeper_cands_n, _cap)} photo-reads"
                        ),
                        state="complete",
                        expanded=bool(len(_sleeper_hits)),
                    )
            except Exception as _sl_exc:
                tlog("SLEEPER",
                     f"hunt failed (non-fatal): "
                     f"{type(_sl_exc).__name__}: {_sl_exc}")

        if not _sleeper_hits.empty:
            _sleeper_hits = _sleeper_hits.drop(
                columns=['_sleeper_score'], errors='ignore'
            )
            _bolo_subset = pd.concat(
                [_bolo_subset, _sleeper_hits], ignore_index=False
            )
            # Downstream code prefers enriched_title when the column
            # exists — fill non-sleeper rows with their original title
            # so they don't resolve to NaN.
            if 'enriched_title' in _bolo_subset.columns:
                _bolo_subset['enriched_title'] = (
                    _bolo_subset['enriched_title']
                    .fillna(_bolo_subset['title'])
                )
            n_matches = len(_bolo_subset)
            if 'auction' in _bolo_subset.columns:
                n_match_auctions = int(_bolo_subset['auction'].nunique())

        scan_label = (
            f"🎯 BOLO scan — {n_matches} matches across "
            f"{n_match_auctions} of {n_auctions} auctions"
        )
        # Stash the summary before _load_auction_for_analysis (which
        # clears the flag). The synthetic-auction load path mirrors
        # the standard one but skips cache lookups since the auction_id
        # space is mixed across many auctions.
        st.session_state._bolo_scan_summary = {
            'total_lots': n_total,
            'matches': n_matches,
            'auctions': n_auctions,
            'match_auctions': n_match_auctions,
            'sleepers': int(len(_sleeper_hits)),
        }
        # Load directly without going through _load_auction_for_analysis
        # (which would try to merge cached data keyed by a single
        # auction_id — wrong for a multi-auction frame).
        st.session_state.selected_leads = _bolo_subset.reset_index(drop=True)
        st.session_state.current_auction = scan_label
        _save_last_scan_view(scan_label, st.session_state.selected_leads)
        _clear_fetched_frame()  # match done — resume file no longer needed
        # Free the full fetched frame — the 86k-row original isn't
        # needed once the subset is extracted (held ~300MB+).
        st.session_state.phase1_leads = pd.DataFrame()
        st.session_state.audit_results = {}
        # Reset all the auto-pipeline / scope flags so the standard
        # post-load flow (audit → credit gate → comps) fires fresh.
        st.session_state.pop('_comps_has_more', None)
        st.session_state.pop('_comps_auction_str_map', None)
        st.session_state.pop('_comps_stats', None)
        st.session_state.pop('_comps_credit_confirmed', None)
        st.session_state.pop('_audit_scope', None)
        st.session_state.pop('_audit_scope_total_lots', None)
        st.session_state._comps_free_only_mode = False
        st.session_state.audit_running = False
        st.session_state.comps_running = False
        st.session_state.pop('_auto_pipeline_attempts', None)
        st.session_state.pop('_audit_confirmed', None)
        # Clear the flag — single-auction loads from here on don't
        # re-enter scan mode unless the button is clicked again.
        st.session_state._bolo_scan_all_pending = False
        st.rerun()

    # Multi-auction KEYWORD-scan path: filter every fetched lot by a
    # case-insensitive substring against title + description and load
    # just the matches as a synthetic "🔍 Keyword: <term>" auction.
    # Parallel to the BOLO scan branch above — same fetch path, same
    # downstream pipeline; only the filter step differs.
    kw_term = (st.session_state.get('_keyword_scan_pending') or '').strip()
    if kw_term:
        with st.status(
            f"🔍 Filtering lots by keyword '{kw_term}'…",
            expanded=True,
        ) as _kw_match_status:
            # Same pre-filter as BOLO scan — drop HARD logistics + bid-cap
            # rejects up front so we don't waste audit / comp credits on
            # lots the user would reject anyway.
            n_total_pre = len(_df)
            _df, _ship_reasons = _apply_easy_ship_filter(_df)
            _df, _bid_reason = _apply_bid_cap_filter(_df)
            _all_reasons = list(_ship_reasons)
            if _bid_reason:
                _all_reasons.append(_bid_reason)
            if _all_reasons:
                _dropped_n = n_total_pre - len(_df)
                st.write(
                    f"📦 Pre-filtered to **{len(_df):,}** actionable lots "
                    f"(dropped {_dropped_n:,}: "
                    f"{', '.join(_all_reasons)})."
                )
            n_total = len(_df)
            # auctions_queried = how many auctions we asked HiBid about
            # (the actual scan breadth). df['auction'].nunique() would
            # undercount because HiBid's server-side searchText filter
            # returns nothing for auctions with no matches — those
            # auctions never appear in _df. We stashed the queried
            # count in session_state at fetch time.
            _scope = st.session_state.get('_last_fetch_scope') or {}
            n_auctions = int(_scope.get('auctions_queried') or 0)
            if n_auctions == 0:
                # Fallback for non-keyword paths that didn't set the
                # scope marker. Use df['auction'].nunique() as before.
                n_auctions = (
                    int(_df['auction'].nunique())
                    if 'auction' in _df.columns else 1
                )
            st.write(
                f"HiBid pre-filtered to **{n_total:,}** lots across "
                f"**{n_auctions}** auctions for '**{kw_term}**' — "
                f"refining locally to title-only matches…"
            )
            _kw_t0 = _time.time()
            # PREFIX word-boundary match per word, order-independent.
            # Each search word must appear in the haystack as a
            # word-start prefix (\b<word>\w*). The leading \b prevents
            # mid-word matches ('ink' won't match 'pink/drink/thinking').
            # The trailing \w* allows stem matching ('sunglass' matches
            # 'sunglasses', 'ink' matches 'inkjet/inks').
            #
            # Multi-word terms use AND-of-lookaheads so order doesn't
            # matter and the words don't have to be adjacent — 'rolex
            # box' matches 'Rolex Submariner box' AND 'box of Rolex
            # parts'. This matches how most search engines work and
            # how HiBid's own server-side searchText behaves.
            #
            # User-supplied term goes through re.escape so regex
            # metacharacters in the term ('.', '(', etc.) are treated
            # as literals.
            _kw_parts = [re.escape(p) for p in kw_term.lower().split()]
            if len(_kw_parts) == 1:
                kw_pattern = r"\b" + _kw_parts[0] + r"\w*"
            else:
                # Lookahead per word. (?=.*\bword\w*) succeeds whenever
                # the word appears anywhere as a prefix. Combining
                # several lookaheads at position 0 requires ALL words
                # to be findable somewhere in the string, in any order.
                kw_pattern = "".join(
                    rf"(?=.*\b{p}\w*)" for p in _kw_parts
                )
            title_lower = (
                _df.get('title', pd.Series(dtype=str))
                .fillna('').astype(str).str.lower()
            )
            desc_lower = (
                _df.get('description', pd.Series(dtype=str))
                .fillna('').astype(str).str.lower()
            )
            title_match = title_lower.str.contains(kw_pattern, regex=True, na=False)
            desc_match = desc_lower.str.contains(kw_pattern, regex=True, na=False)
            _kw_subset = _df[title_match | desc_match].copy()
            _kw_elapsed = _time.time() - _kw_t0
            n_matches = len(_kw_subset)
            n_match_auctions = (
                int(_kw_subset['auction'].nunique())
                if 'auction' in _kw_subset.columns and not _kw_subset.empty
                else 0
            )
            _match_pct = (100 * n_matches / n_total) if n_total else 0.0
            tlog("KEYWORD",
                 f"matched {n_matches:,}/{n_total:,} lots ({_match_pct:.1f}%)",
                 f"in {_kw_elapsed:.2f}s",
                 f"· {n_match_auctions}/{n_auctions} auctions hit",
                 f"· term='{kw_term}'")
            st.write(
                f"✅ **{n_matches:,}** lots match '{kw_term}' "
                f"({_match_pct:.1f}% hit rate) across "
                f"**{n_match_auctions}** of **{n_auctions}** auctions."
            )
            if n_matches > 0:
                # Top auctions by match count
                if 'auction' in _kw_subset.columns:
                    _top_auctions = (
                        _kw_subset['auction']
                        .value_counts()
                        .head(5)
                        .to_dict()
                    )
                    _auc_summary = " · ".join(
                        f"*{a[:40]}* ({n})" for a, n in _top_auctions.items()
                    )
                    st.write(f"🏆 Densest auctions: {_auc_summary}")
                st.write(
                    "Loading the matched subset into the analysis view; "
                    "audit + price comps run next."
                )
            else:
                st.write(
                    f"No lots match '{kw_term}'. Try a different keyword "
                    "or check that the visible auctions actually contain "
                    "items related to your search."
                )
            _kw_match_status.update(
                label=(
                    f"✅ {n_matches:,} matches for '{kw_term}' "
                    f"across {n_match_auctions} auctions"
                ),
                state="complete", expanded=False,
            )

        # Stash last-scan result so the default view can show a
        # persistent "0 matches" banner after the rerun. Without this,
        # a 0-match scan would bounce the user back to default view
        # with no explanation — feeling like the search "did nothing".
        st.session_state._keyword_scan_last_result = {
            'term': kw_term,
            'matches': n_matches,
            'match_auctions': n_match_auctions,
            'auctions_queried': n_auctions,
            'lots_scanned': n_total,
            'finished_at': datetime.now().isoformat(),
        }
        st.session_state._keyword_scan_summary = {
            'term': kw_term,
            'total_lots': n_total,
            'matches': n_matches,
            'auctions': n_auctions,
            'match_auctions': n_match_auctions,
        }
        # Clear the pending flag FIRST (before rerun) so subsequent
        # auction loads don't re-enter keyword-scan mode.
        st.session_state._keyword_scan_pending = ''

        if n_matches == 0:
            # 0-match path: don't enter the analysis view. Leave the
            # user on the default view; the persisted last-scan-result
            # banner explains what happened. Phase1 leads get cleared
            # so the post-fetch handler doesn't loop on the next rerun.
            st.session_state.phase1_leads = pd.DataFrame()
            st.session_state.selected_leads = pd.DataFrame()
            st.session_state.current_auction = None
            st.rerun()

        # Match path: load matched subset into the analysis view.
        scan_label = (
            f"🔍 Keyword: {kw_term} — {n_matches} matches across "
            f"{n_match_auctions} of {n_auctions} auctions"
        )
        st.session_state.selected_leads = _kw_subset.reset_index(drop=True)
        st.session_state.current_auction = scan_label
        _save_last_scan_view(scan_label, st.session_state.selected_leads)
        _clear_fetched_frame()  # match done — resume file no longer needed
        st.session_state.phase1_leads = pd.DataFrame()
        st.session_state.audit_results = {}
        # Reset all auto-pipeline / scope flags (same as BOLO branch).
        st.session_state.pop('_comps_has_more', None)
        st.session_state.pop('_comps_auction_str_map', None)
        st.session_state.pop('_comps_stats', None)
        st.session_state.pop('_comps_credit_confirmed', None)
        st.session_state.pop('_audit_scope', None)
        st.session_state.pop('_audit_scope_total_lots', None)
        st.session_state._comps_free_only_mode = False
        st.session_state.audit_running = False
        st.session_state.comps_running = False
        st.session_state.pop('_auto_pipeline_attempts', None)
        st.session_state.pop('_audit_confirmed', None)
        st.rerun()

    # Multi-select basket: the user checked several auctions in the
    # Discover grid and hit Analyze. Load ALL their lots as one
    # combined synthetic view — no BOLO/keyword filtering; the full
    # pipeline (audit → credit gate → comps → grading) runs across
    # the whole basket. Cached analyses for previously-run auctions
    # were already merged at fetch time.
    if st.session_state.get('_multi_select_pending'):
        _n_basket_auctions = (
            int(_df['auction'].nunique()) if 'auction' in _df.columns else 1
        )
        basket_label = (
            f"🧺 {_n_basket_auctions} auctions — {len(_df):,} lots"
        )
        tlog("MULTI",
             f"basket load: {_n_basket_auctions} auctions",
             f"· {len(_df):,} lots")
        st.session_state.selected_leads = _df.reset_index(drop=True)
        st.session_state.current_auction = basket_label
        _save_last_scan_view(basket_label, st.session_state.selected_leads)
        _clear_fetched_frame()  # match done — resume file no longer needed
        st.session_state.phase1_leads = pd.DataFrame()
        st.session_state.audit_results = {}
        # Reset auto-pipeline / scope flags (same as the scan branches).
        st.session_state.pop('_comps_has_more', None)
        st.session_state.pop('_comps_auction_str_map', None)
        st.session_state.pop('_comps_stats', None)
        st.session_state.pop('_comps_credit_confirmed', None)
        st.session_state.pop('_audit_scope', None)
        st.session_state.pop('_audit_scope_total_lots', None)
        st.session_state.pop('_comps_error_count', None)
        st.session_state._comps_free_only_mode = False
        st.session_state.audit_running = False
        st.session_state.comps_running = False
        st.session_state.pop('_auto_pipeline_attempts', None)
        st.session_state.pop('_audit_confirmed', None)
        st.session_state._multi_select_pending = False
        st.rerun()

    # Standard single-auction load (default behavior). Belt-and-
    # suspenders guard: if the fetched frame contains lots from MORE
    # THAN ONE auction (which means the user clicked "Search for
    # BOLOs across N auctions"), we should NEVER silently load just
    # the first auction. That's what caused the "scanned 1002, only
    # 1 showed" bug — a missing guard let single-auction fallback
    # narrow a multi-auction frame. If somehow we land here with
    # multi-auction data and the BOLO scan path didn't catch it,
    # surface a hard error rather than silently truncating.
    if 'auction' in _df.columns and len(_df):
        _unique_auctions = _df['auction'].nunique()
        if _unique_auctions > 1:
            tlog("ERROR",
                 f"Multi-auction frame ({_unique_auctions} auctions, "
                 f"{len(_df)} lots) reached single-auction fallback.",
                 "_bolo_scan_all_pending was probably False when it",
                 "should have been True. Investigate the BOLO scan path.")
            st.error(
                f"⚠️ Internal routing bug: a multi-auction fetch "
                f"({_unique_auctions} auctions, {len(_df):,} lots) reached "
                f"the single-auction loader. Refusing to truncate to one "
                f"auction. Click **🔍 Search for BOLOs** again to retry, "
                f"or report this with your terminal output."
            )
            st.stop()
        _auction_name = _df['auction'].iloc[0]
        _auction_df = _df[_df['auction'] == _auction_name].reset_index(drop=True)
        _load_auction_for_analysis(_auction_name, _auction_df)
        st.rerun()

current_auction = st.session_state.get('current_auction')

# ---- LIVE STATUS RIGHT PANEL ----
# A fixed-position floating panel pinned to the right edge of the
# viewport. Always visible while the analysis view is active so the
# user can see live activity (current phase, lot counts, credit
# budget, registry hits) without scrolling. Re-renders on every
# Streamlit interaction.
def _render_live_status_panel(leads_df=None):
    """Render a fixed-position status panel on the right edge."""
    # Aggregate live stats
    n_lots = len(leads_df) if leads_df is not None else 0
    n_comped = 0
    n_a = n_b = 0
    n_bolo = 0
    n_tier1 = n_tier2 = n_tier3 = 0
    if leads_df is not None and not leads_df.empty:
        if 'est_resale' in leads_df.columns:
            n_comped = int(leads_df['est_resale'].notna().sum())
        # Buy-grade counts replace the old ROI/STR threshold-based
        # green/yellow counts. The buy_grade column is populated by
        # `_compute_buy_score` after comps run.
        if 'buy_grade' in leads_df.columns:
            grades = leads_df['buy_grade'].fillna('').astype(str)
            n_a = int(grades.str.startswith('🟢').sum())
            n_b = int(grades.str.startswith('🟡').sum())
        if 'bolo_brand' in leads_df.columns:
            n_bolo = int(leads_df['bolo_brand'].notna().sum())
        if 'bolo_tier' in leads_df.columns:
            tier = pd.to_numeric(leads_df['bolo_tier'], errors='coerce')
            n_tier1 = int((tier == 1).sum())
            n_tier2 = int((tier == 2).sum())
            n_tier3 = int((tier == 3).sum())

    # Active phase
    if discover_running:
        phase = "🔍 Discovering"
        phase_color = "#3b82f6"
    elif _sampling_pending:
        phase = "🔍 Loading samples"
        phase_color = "#3b82f6"
    elif fetch_lots_running:
        phase = "📥 Fetching lots"
        phase_color = "#3b82f6"
    elif _audit_running:
        phase = "🧠 AI audit"
        phase_color = "#a855f7"
    elif _comps_running:
        phase = "💰 Comps running"
        phase_color = "#f59e0b"
    elif _comps_running == False and n_comped > 0:
        phase = "✅ Idle (comp'd)"
        phase_color = "#22c55e"
    else:
        phase = "⚪ Idle"
        phase_color = "#6b7280"

    # Registry stats
    # ScrapingBee budget remaining (from session-cached usage)
    sb_used_pct = None
    try:
        from scraper.config_loader import load_config as _cfg
        sb_cfg = _cfg().get('scrapingbee') or {}
        if sb_cfg.get('api_key'):
            usage = _fetch_scrapingbee_usage(sb_cfg.get('api_key'))
            used = int(usage.get('used_api_credit') or 0)
            cap = int(usage.get('max_api_credit') or 0)
            if cap:
                sb_used_pct = round(100 * used / cap, 0)
    except Exception:
        pass

    bolo_bits = ""
    if n_bolo > 0:
        bolo_bits = (
            f"<div class='lsp-row'>"
            f"<span class='lsp-label'>BOLO hits</span>"
            f"<span class='lsp-val'>{n_bolo}</span>"
            f"</div>"
        )
        tier_parts = []
        if n_tier1: tier_parts.append(f"T1:{n_tier1}")
        if n_tier2: tier_parts.append(f"T2:{n_tier2}")
        if n_tier3: tier_parts.append(f"T3:{n_tier3}")
        if tier_parts:
            bolo_bits += (
                f"<div class='lsp-row'>"
                f"<span class='lsp-label'>by tier</span>"
                f"<span class='lsp-val' style='font-size:10px;'>"
                f"{' · '.join(tier_parts)}</span>"
                f"</div>"
            )

    sb_bit = ""
    if sb_used_pct is not None:
        bar_color = (
            '#22c55e' if sb_used_pct < 50 else
            '#f59e0b' if sb_used_pct < 80 else
            '#ef4444'
        )
        sb_bit = (
            f"<div class='lsp-row'><span class='lsp-label'>"
            f"SB credits used</span>"
            f"<span class='lsp-val' style='color:{bar_color};'>"
            f"{int(sb_used_pct)}%</span></div>"
        )

    html = f"""
    <style>
      .lsp-panel {{
        position: fixed;
        right: 8px;
        top: 90px;
        width: 220px;
        z-index: 998;
        background: rgba(15,23,42,0.94);
        color: #e5e7eb;
        border-left: 4px solid {phase_color};
        border-radius: 6px;
        padding: 10px 12px;
        font-size: 12px;
        box-shadow: -2px 2px 12px rgba(0,0,0,0.3);
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        max-height: calc(100vh - 110px);
        overflow-y: auto;
      }}
      .lsp-title {{
        font-weight: 700;
        font-size: 13px;
        color: {phase_color};
        margin-bottom: 8px;
        letter-spacing: 0.3px;
      }}
      .lsp-section {{
        margin-bottom: 8px;
        padding-bottom: 6px;
        border-bottom: 1px solid rgba(255,255,255,0.08);
      }}
      .lsp-section:last-child {{ border-bottom: none; }}
      .lsp-section-title {{
        font-size: 9px;
        text-transform: uppercase;
        letter-spacing: 0.7px;
        color: #9ca3af;
        margin-bottom: 3px;
      }}
      .lsp-row {{
        display: flex;
        justify-content: space-between;
        padding: 1px 0;
      }}
      .lsp-label {{
        color: #9ca3af;
        font-size: 11px;
      }}
      .lsp-val {{
        color: #e5e7eb;
        font-weight: 600;
        font-size: 12px;
        font-variant-numeric: tabular-nums;
      }}
      .lsp-green {{ color: #22c55e; }}
      .lsp-yellow {{ color: #f59e0b; }}

      /* Hide on small screens — would overlap with content */
      @media (max-width: 1280px) {{
        .lsp-panel {{ display: none; }}
      }}
    </style>
    <div class="lsp-panel">
      <div class="lsp-title">{phase}</div>

      <div class="lsp-section">
        <div class="lsp-section-title">This view</div>
        <div class="lsp-row"><span class="lsp-label">Lots loaded</span>
          <span class="lsp-val">{n_lots}</span></div>
        <div class="lsp-row"><span class="lsp-label">Comped</span>
          <span class="lsp-val">{n_comped}</span></div>
        <div class="lsp-row"><span class="lsp-label">🟢 A grade</span>
          <span class="lsp-val lsp-green">{n_a}</span></div>
        <div class="lsp-row"><span class="lsp-label">🟡 B grade</span>
          <span class="lsp-val lsp-yellow">{n_b}</span></div>
      </div>

      {('<div class="lsp-section">'
        '<div class="lsp-section-title">BOLO matches</div>'
        + bolo_bits + '</div>') if n_bolo > 0 else ''}

      {('<div class="lsp-section">'
        '<div class="lsp-section-title">Budget</div>'
        + sb_bit + '</div>') if sb_bit else ''}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# ---- ANALYSIS VIEW: one auction is loaded ----
if current_auction and not st.session_state.selected_leads.empty:
    leads_df = st.session_state.selected_leads

    # Live status panel pinned to the right edge of the viewport.
    # Re-renders on every Streamlit interaction so counts update
    # in real time as filters / target sliders / table sort change.
    # CSS hides it on screens narrower than 1280px (would overlap
    # with the main table).
    _render_live_status_panel(
        st.session_state.get('audit_results')
        if isinstance(st.session_state.get('audit_results'), pd.DataFrame)
        and not st.session_state.audit_results.empty
        else leads_df
    )

    # Back button + Refresh-bids button + header.
    # The Refresh-bids button re-runs Phase 1 for just this auction so the
    # user gets fresh current_bid / bid_count / time_left without burning
    # any ScrapingBee credits — _load_auction_for_analysis overlays the
    # cached audit + comps onto the freshly fetched lots, and the
    # auto-pipeline + credit gate both no-op because has_audit and
    # has_comps_data are still True after the merge.
    bc1, bc2, bc3 = st.columns([1, 1, 3])
    with bc1:
        if st.button("← Back to auctions", width='stretch'):
            st.session_state.selected_leads = pd.DataFrame()
            st.session_state.current_auction = None
            st.session_state.audit_results = {}
            # Also clear phase1_leads so the auto-load step above doesn't
            # immediately bounce us back into analysis — we want to land
            # on the picker (Selection view) instead.
            st.session_state.phase1_leads = pd.DataFrame()
            st.rerun()
    with bc2:
        _refresh_bids_running = (
            st.session_state.get('fetch_lots_running', False)
            or st.session_state.get('discover_running', False)
        )
        _refresh_label = (
            "⏳ Refreshing…" if _refresh_bids_running else "🔁 Refresh bids"
        )
        if st.button(
            _refresh_label,
            width='stretch',
            disabled=_refresh_bids_running,
            key="header_refresh_bids_btn",
            help=(
                "Re-fetch just this auction's current bids / time-left "
                "without re-running audit or comps. Spends 0 ScrapingBee "
                "credits — cached analysis is preserved."
            ),
        ):
            aid = _extract_auction_id(leads_df)
            if aid is not None:
                st.session_state._selected_auction_ids = [aid]
                st.session_state.current_auction = None
                st.session_state.selected_leads = pd.DataFrame()
                st.session_state.fetch_lots_running = True
                st.rerun()
    with bc3:
        # Make the auction name itself a clickable link to the HiBid
        # auction page. Replaces the per-row 'Auction' column we used
        # to render in the results table — the auction is the same
        # for every row, so one link at the top is plenty.
        # EXCEPT for multi-auction BOLO scans, where lots come from
        # many sources — show a plain title and let the per-row
        # Auction column handle navigation.
        _auction_url = None
        is_multi_auction = (
            current_auction.startswith("🎯 BOLO scan")
            or current_auction.startswith("🔍 Keyword:")
            or current_auction.startswith("🧺")
        )
        if (not is_multi_auction
                and 'auction_link' in leads_df.columns and not leads_df.empty):
            link_val = leads_df['auction_link'].dropna().head(1)
            if not link_val.empty:
                _auction_url = str(link_val.iloc[0]).strip() or None
        if _auction_url:
            # Render as an h3 with the auction name as a clickable link
            # that opens the HiBid auction page in a new tab. target=_blank
            # is the natural choice — clicking shouldn't navigate away
            # from the analysis view.
            st.markdown(
                f"### 🔬 <a href='{_auction_url}' target='_blank' "
                f"rel='noopener'>{current_auction}</a>  "
                f"<span style='font-size:0.7em;opacity:0.6'>↗</span>",
                unsafe_allow_html=True,
            )
        else:
            st.subheader(f"🔬 {current_auction}")
        caption_bits = [f"{len(leads_df)} items loaded"]

        # Scope marker: when the user picked "BOLO only" on a big
        # auction, the loaded leads are a filtered subset. Make that
        # explicit so the user understands why the count looks low.
        _scope = st.session_state.get('_audit_scope')
        _scope_total = st.session_state.get('_audit_scope_total_lots')
        if _scope == 'bolo' and _scope_total:
            caption_bits.append(
                f"🎯 BOLO-only scope ({len(leads_df)} of {_scope_total:,} lots)"
            )
        # Free-mode marker: signals to the user that est_resale data
        # only comes from PC + GoCollect for non-BOLO lots — empty
        # est_resale on most non-BOLO rows is expected (not a
        # comp-failure to investigate). BOLO lots get the full
        # eBay/Mercari treatment so they aren't affected.
        if st.session_state.get('_comps_free_only_mode'):
            caption_bits.append(
                "🆓 Free + BOLO comps "
                "(non-BOLO: PriceCharting only · BOLO: full eBay/Mercari)"
            )

        # Multi-auction BOLO scan: when the loaded leads came from a
        # consolidated scan across every discovered auction, surface
        # the breadth (X auctions had at least one BOLO match out of
        # Y discovered, scanned Z total lots). Helps the user gauge
        # how thoroughly their watch list is being matched.
        scan_summary = st.session_state.get('_bolo_scan_summary')
        if scan_summary and current_auction.startswith("🎯 BOLO scan"):
            caption_bits.append(
                f"📡 scanned {scan_summary['total_lots']:,} total lots "
                f"({scan_summary['match_auctions']} of "
                f"{scan_summary['auctions']} auctions had matches)"
            )
            if scan_summary.get('sleepers'):
                caption_bits.append(
                    f"🕵️ {scan_summary['sleepers']} sleeper hit(s) — "
                    f"container lots identified from photos"
                )

        # Keyword scan summary — parallel to the BOLO scan banner above.
        kw_summary = st.session_state.get('_keyword_scan_summary')
        if kw_summary and current_auction.startswith("🔍 Keyword:"):
            caption_bits.append(
                f"📡 searched {kw_summary['total_lots']:,} lots for "
                f"'{kw_summary['term']}' "
                f"({kw_summary['match_auctions']} of "
                f"{kw_summary['auctions']} auctions had matches)"
            )

        # Conditional-shipping badge: the auction's terms say shipping
        # is only available on SOME lots ("contact prior to bidding to
        # confirm"). The Ship / Local Pickup classification is soft
        # here — confirm with the auctioneer before counting on either.
        if ('auction_cond_ship' in leads_df.columns
                and leads_df['auction_cond_ship']
                    .fillna(False).astype(bool).any()):
            caption_bits.append(
                "📦 shipping conditional — auctioneer confirms per lot"
            )

        # Unreachable pickup-only badges: pickup-only lots inside an
        # auction that's OUTSIDE the pickup radius. Split by whether
        # the auction offers conditional shipping:
        #   hard: no shipping option → auto-F, excluded from comps
        #   soft: "contact us" shipping → graded WITH ship cost added
        if 'unreachable_pickup' in leads_df.columns:
            _unreach_m = (
                leads_df['unreachable_pickup'].fillna(False).astype(bool)
            )
            _cond_m = (
                leads_df['auction_cond_ship'].fillna(False).astype(bool)
                if 'auction_cond_ship' in leads_df.columns
                else pd.Series(False, index=leads_df.index)
            )
            _n_hard = int((_unreach_m & ~_cond_m).sum())
            _n_soft = int((_unreach_m & _cond_m).sum())
            if _n_hard:
                caption_bits.append(
                    f"🚫 {_n_hard} pickup-only lots unreachable "
                    f"(outside radius, no shipping) — auto-F"
                )
            if _n_soft:
                caption_bits.append(
                    f"🚚 {_n_soft} pickup-flagged lots — auctioneer "
                    f"ships on request; confirm shipping + cost "
                    f"before bidding"
                )

        # Per-auction buyer-premium badge — show the ACTUAL premium in
        # use when it was parsed from HiBid (vs the config default).
        if 'auction_buyer_premium_pct' in leads_df.columns:
            _bp_vals = pd.to_numeric(
                leads_df['auction_buyer_premium_pct'], errors='coerce'
            ).dropna().unique()
            if len(_bp_vals) == 1 and _bp_vals[0] > 0:
                caption_bits.append(
                    f"💳 {round((_bp_vals[0] - 1) * 100)}% buyer premium"
                )

        # Persistent BOLO brand chart + per-brand lot preview. Renders
        # for any view where leads_df has BOLO brand data. The bar chart
        # gives a quick visual of which brands are in the loaded set;
        # the selectbox below it drives an inline lot preview without
        # disrupting the main results table.
        if (
            'bolo_brand' in leads_df.columns
            and leads_df['bolo_brand'].notna().sum() >= 2
        ):
            with st.expander(
                "📊 BOLO brand breakdown (chart + per-brand preview)",
                expanded=False,
            ):
                _brand_counts_full = (
                    leads_df['bolo_brand']
                    .dropna()
                    .value_counts()
                    .to_dict()
                )
                _render_brand_bar_chart(
                    _brand_counts_full,
                    title=(
                        f"BOLO matches by brand "
                        f"({sum(_brand_counts_full.values())} total)"
                    ),
                )
                # Brand selector for the lot preview. Sorted by count
                # desc so the top-volume brand is the default. Empty
                # placeholder is the no-selection state.
                _brand_options = sorted(
                    _brand_counts_full.keys(),
                    key=lambda b: -_brand_counts_full[b],
                )
                _selected_brand = st.selectbox(
                    "Preview lots for brand",
                    options=[""] + _brand_options,
                    format_func=(
                        lambda b: f"{b} ({_brand_counts_full.get(b, 0)})"
                        if b else "(pick a brand…)"
                    ),
                    key="bolo_brand_preview_picker",
                    label_visibility="collapsed",
                )
                if _selected_brand:
                    _slice_df = leads_df[
                        leads_df['bolo_brand'] == _selected_brand
                    ].copy()
                    st.markdown(
                        f"### 📌 {_selected_brand} ({len(_slice_df)} lots)"
                    )
                    _preview_cols = [
                        c for c in (
                            'auction', 'title', 'current_bid',
                            'est_resale', 'est_roi', 'verdict',
                            'bolo_tier', 'bolo_category', 'lot_link',
                        ) if c in _slice_df.columns
                    ]
                    if _preview_cols:
                        _preview_df = _slice_df[_preview_cols].head(50)
                        st.dataframe(
                            _preview_df, width='stretch', hide_index=True,
                            column_config={
                                "lot_link": st.column_config.LinkColumn(
                                    "Link", display_text="open ↗",
                                ) if 'lot_link' in _preview_df.columns
                                else None,
                            },
                        )
                        if len(_slice_df) > 50:
                            st.caption(
                                f"Showing first 50 of {len(_slice_df)} "
                                "matching lots."
                            )

        # If the current analysis came from cache, indicate that + when.
        auction_id = _extract_auction_id(leads_df)
        if auction_id is not None:
            payload = _AUCTION_CACHE.load(auction_id)
            if payload and _AUCTION_CACHE.is_fresh(
                payload, ttl_days=st.session_state.cache_ttl_days
            ):
                try:
                    cached_at = datetime.fromisoformat(payload.get('cached_at', ''))
                    age = datetime.now() - cached_at
                    age_str = (
                        f"{age.days}d ago" if age.days > 0
                        else f"{int(age.seconds / 3600)}h ago"
                    )
                except Exception:
                    age_str = "earlier"
                caption_bits.append(f"💾 cached analysis from {age_str} · current bids refreshed")
        st.caption(" · ".join(caption_bits))

        # When in BOLO-only scope, expose a one-click "↗ expand to full"
        # button that re-fetches the auction from Phase 1 and resets
        # the scope to None so the chooser fires again. The user gets
        # a fresh dataset; the BOLO-only audit work doesn't carry over
        # (Phase 1 doesn't save partial audits to the cache yet) but
        # current bids are fresh and they can pick "full" this time.
        if _scope == 'bolo':
            if st.button(
                "↗ Expand to full auction",
                key="expand_to_full_scope",
                help="Re-fetch the entire auction's lots and re-run the "
                     "scope chooser. Use this if the BOLO subset turned "
                     "up promising signal and you now want to comp the "
                     "rest of the auction too.",
            ):
                aid_for_expand = _extract_auction_id(leads_df)
                if aid_for_expand is not None:
                    st.session_state._selected_auction_ids = [aid_for_expand]
                    st.session_state.current_auction = None
                    st.session_state.selected_leads = pd.DataFrame()
                    st.session_state.phase1_leads = pd.DataFrame()
                    st.session_state.audit_results = {}
                    st.session_state.pop('_audit_scope', None)
                    st.session_state.pop('_audit_scope_total_lots', None)
                    st.session_state.pop('_auto_pipeline_attempts', None)
                    st.session_state.fetch_lots_running = True
                    st.rerun()

    has_audit = (
        isinstance(st.session_state.get('audit_results'), pd.DataFrame)
        and not st.session_state.audit_results.empty
        and 'verdict' in st.session_state.audit_results.columns
    )

    # Hoisted "running" flags so each tab can disable buttons consistently
    # regardless of which tab the user has open.
    audit_running = st.session_state.get('audit_running', False)
    comps_running = st.session_state.get('comps_running', False)

    # ================================================================
    # AUTO-PIPELINE TRIGGER (early-fire, no UI flicker)
    # Runs BEFORE any heavy rendering so that the auto-`st.rerun()`
    # below doesn't briefly flash the empty placeholders before
    # transitioning into the audit/comps status block.
    # ================================================================
    # "comps_data" used to mean "ANY est_resale present", but cached
    # auctions can have 1-3 stale values from earlier runs that confuse
    # the auto-pipeline into skipping a fresh comps pass. Better signal:
    # at least 30% of audit-eligible (good+, non-HARD, has-thumbnail)
    # lots must have a comp value before we treat comps as "done".
    has_comps_data = False
    ar_state = st.session_state.get('audit_results')
    if (isinstance(ar_state, pd.DataFrame)
            and 'est_resale' in ar_state.columns
            and not ar_state.empty):
        eligible_mask = (
            ~ar_state.get('red_flag', pd.Series(False, index=ar_state.index)).fillna(False).astype(bool)
            & (ar_state.get('logistics_ease', pd.Series('', index=ar_state.index)) != 'HARD')
        )
        eligible_count = int(eligible_mask.sum())
        if eligible_count > 0:
            comped_count = int(
                ar_state.loc[eligible_mask, 'est_resale'].notna().sum()
            )
            has_comps_data = (comped_count / eligible_count) >= 0.30

    # Detect "stale audit" — when a cache-restored auction has the audit
    # columns but most rows are `audit_source=no_api_key` (because the
    # cache was made before the API key was wired up). Those rows show
    # verdict=Unknown / confidence=0 across the whole auction, and items
    # that should be red-flagged ("broken", "untested", "for parts")
    # leak into the comps stage and waste lookups.
    audit_looks_stale = False
    # audit_is_dead: the audit RAN but produced nothing usable on most
    # rows — key missing, SDK missing, or every API call failing (SSL /
    # revoked key). Distinct from "stale": dead gates the comps stage
    # so ScrapingBee credits aren't spent pricing unvetted lots (the
    # 7/6 Hayworth run comped 11 unvetted lots before this existed).
    audit_is_dead = False
    _DEAD_AUDIT_SOURCES = ('no_api_key', 'text_api_failed', 'image_api_failed')
    if (isinstance(ar_state, pd.DataFrame)
            and 'audit_source' in ar_state.columns
            and len(ar_state) > 0):
        _src = ar_state['audit_source'].fillna('')
        dead_count = int(_src.isin(_DEAD_AUDIT_SOURCES).sum())
        no_key_count = int((_src == 'no_api_key').sum())
        # Only consider it "stale" if our config actually has an API key
        # to use — otherwise re-running won't help.
        try:
            from scraper.config_loader import load_config
            _cfg = load_config()
            _has_key = bool((_cfg.get("anthropic") or {}).get("api_key"))
        except Exception:
            _has_key = False
        # Failed-API rows count toward the stale retry too — one free
        # re-attempt is worth it for transient failures; if the retry
        # also dies, audit_is_dead persists and comps stay gated.
        if _has_key and (dead_count / len(ar_state)) >= 0.50:
            audit_looks_stale = True
        if (dead_count / len(ar_state)) >= 0.50:
            audit_is_dead = True

    auto_attempts: set = st.session_state.setdefault(
        '_auto_pipeline_attempts', set()
    )
    has_more_chunks = bool(st.session_state.get('_comps_has_more'))

    # Detect generic-titled lots that would benefit from image enrichment
    # before comps run. We check the post-audit state so red-flagged and
    # HARD-logistics lots are already filtered out.
    needs_image_enrich = False

    # ================================================================
    # PRE-AUDIT PREVIEW GATE
    # Freshly loaded auction (no audit yet, nothing spent): show the
    # raw lot table FIRST and wait for an explicit "Run pipeline"
    # click. Free signal shown up-front: current bids, bid activity,
    # est_cost at the auction's real premium, logistics class, and a
    # zero-cost BOLO regex pass. Cached auctions skip this entirely
    # (has_audit is already True on load).
    # ================================================================
    if (not has_audit and not audit_running and not comps_running
            and not st.session_state.get('_audit_confirmed', False)):
        st.markdown("### 👀 Lot preview — nothing has run yet")
        _prev_df = _compute_bolo_columns(leads_df)
        _n_prev_bolo = (
            int(_prev_df['bolo_brand'].notna().sum())
            if 'bolo_brand' in _prev_df.columns else 0
        )
        _prev_bits = [f"**{len(_prev_df):,}** lots"]
        if _n_prev_bolo:
            _prev_bits.append(f"🎯 **{_n_prev_bolo}** BOLO matches")
        _bids_active = int(
            (pd.to_numeric(_prev_df.get('bid_count'), errors='coerce')
             .fillna(0) > 0).sum()
        )
        _prev_bits.append(f"{_bids_active} lots have bids")
        st.caption(
            " · ".join(_prev_bits)
            + " — browse below, then run the pipeline when ready."
        )

        # ---- ⚡ Pre-audit filters (same knobs as the comps gate) ----
        # These share session keys with the credit-gate spend caps, so
        # setting them here means they're already set when the comps
        # gate renders later. Applying them NOW trims the lot set
        # before the AI audit runs — saving Claude spend and wall-
        # clock, not just ScrapingBee credits. Requested 7/11 for the
        # full-BOLO-scan workflow.
        with st.expander(
            "⚡ Pre-audit filters — trim before ANY spend (audit + comps)",
            expanded=bool(len(_prev_df) >= 100),
        ):
            _paf1, _paf2, _paf3 = st.columns(3)
            with _paf1:
                st.number_input(
                    "Min bid floor ($)",
                    min_value=0.0, step=1.0, format="%.2f",
                    key="_gate_min_bid_filter",
                    help="Drop lots whose current bid is below this. "
                         "Cheap-junk filter — try $5-10.",
                )
            with _paf2:
                st.number_input(
                    "Skip if next-bid > ($)  (0 = no cap)",
                    min_value=0.0, step=10.0, format="%.2f",
                    key="comps_skip_above_bid",
                    help="Drop lots already bid above this — squeezed "
                         "margins rarely pay back the spend.",
                )
            with _paf3:
                st.number_input(
                    "Cap to top N by bid (0 = no cap)",
                    min_value=0, step=50,
                    key="_gate_top_n_filter",
                    help="Keep only the N highest-bid lots.",
                )
            _paf4, _paf5, _paf6 = st.columns(3)
            with _paf4:
                _prev_has_tiers = (
                    'bolo_tier' in _prev_df.columns
                    and _prev_df['bolo_tier'].notna().any()
                )
                st.checkbox(
                    "Tier 1 BOLO only",
                    key="_gate_tier1_only_filter",
                    disabled=not _prev_has_tiers,
                    help="Restrict to tier-1 BOLO matches (curated "
                         "highest-resale brands).",
                )
            with _paf5:
                st.checkbox(
                    "Easy-ship only (mailbox-size)",
                    key="comps_easy_ship_only",
                    help="Keep only items that look mailbox-shippable.",
                )
            with _paf6:
                st.checkbox(
                    "Exclude HARD logistics",
                    key="comps_exclude_hard",
                    help="Skip items flagged hard to ship/pick up.",
                )
            _paf7, _paf8 = st.columns([2, 1])
            with _paf7:
                _prev_cat_counts = (
                    _prev_df['category'].fillna('(none)').astype(str)
                    .value_counts().to_dict()
                    if 'category' in _prev_df.columns else {}
                )
                st.multiselect(
                    "Exclude lot categories",
                    options=sorted(_prev_cat_counts.keys()),
                    format_func=lambda c: (
                        f"{c} ({_prev_cat_counts.get(c, 0)})"
                    ),
                    key="preaudit_exclude_categories",
                    help="Drop every lot whose HiBid category is in "
                         "this list before the audit runs.",
                )
            with _paf8:
                st.checkbox(
                    "Skip 🥈 sterling / gold",
                    key="preaudit_exclude_metals",
                    help="Drop precious-metal lots (sterling / 925 / "
                         "karat gold / platinum in the title) — the "
                         "weighed-metal market is priced to melt by "
                         "other bidders anyway. Regex-based, so it "
                         "catches metals even when the HiBid category "
                         "is vague.",
                )

        # Apply the filters to the preview so the table + count show
        # exactly what the audit will receive.
        _filt_df = _prev_df
        _filt_df, _ = _apply_easy_ship_filter(_filt_df)
        _filt_df, _ = _apply_bid_cap_filter(_filt_df)
        _paf_min_bid = float(
            st.session_state.get('_gate_min_bid_filter', 0.0) or 0.0
        )
        if _paf_min_bid > 0 and 'current_bid' in _filt_df.columns:
            _filt_df = _filt_df[
                pd.to_numeric(_filt_df['current_bid'], errors='coerce')
                .fillna(0) >= _paf_min_bid
            ]
        if (st.session_state.get('_gate_tier1_only_filter', False)
                and 'bolo_tier' in _filt_df.columns):
            _filt_df = _filt_df[
                pd.to_numeric(_filt_df['bolo_tier'], errors='coerce') == 1
            ]
        _paf_top_n = int(
            st.session_state.get('_gate_top_n_filter', 0) or 0
        )
        if (_paf_top_n > 0 and len(_filt_df) > _paf_top_n
                and 'current_bid' in _filt_df.columns):
            _filt_df = _filt_df.sort_values(
                'current_bid', ascending=False
            ).head(_paf_top_n)
        _excl_cats = set(
            st.session_state.get('preaudit_exclude_categories', []) or []
        )
        if _excl_cats and 'category' in _filt_df.columns:
            _filt_df = _filt_df[
                ~_filt_df['category'].fillna('(none)').astype(str)
                .isin(_excl_cats)
            ]
        if (st.session_state.get('preaudit_exclude_metals', False)
                and 'title' in _filt_df.columns):
            _t = _filt_df['title'].fillna('').astype(str)
            _metal_mask = (
                _t.str.contains(_STERLING_RE, na=False)
                | _t.str.contains(_KARAT_RE, na=False)
                | _t.str.contains(_PLATINUM_RE, na=False)
                | _t.str.contains(r'solid\s+gold', case=False, na=False)
                # Bare "925" — the melt regex is deliberately strict
                # (avoids model numbers), but for a SKIP filter a
                # slightly wider net is the right trade.
                | _t.str.contains(r'925', na=False)
            )
            _filt_df = _filt_df[~_metal_mask]
        if len(_filt_df) != len(_prev_df):
            st.caption(
                f"⚡ Filters keep **{len(_filt_df):,}** of "
                f"{len(_prev_df):,} lots — only these go to the "
                f"audit + comps."
            )
        _prev_df = _filt_df
        _prev_cols = [
            c for c in (
                'title', 'current_bid', 'next_bid', 'bid_count',
                'est_cost', 'time_left', 'category', 'logistics_ease',
                'source', 'bolo_brand', 'auction', 'lot_link',
            ) if c in _prev_df.columns
        ]
        # Single-auction views don't need the auction column
        if ('auction' in _prev_cols
                and _prev_df['auction'].nunique() <= 1):
            _prev_cols.remove('auction')
        st.dataframe(
            _prev_df[_prev_cols],
            width='stretch',
            height=min(560, 60 + 35 * len(_prev_df)),
            hide_index=True,
            column_config={
                'lot_link': st.column_config.LinkColumn(
                    'Link', display_text='open ↗',
                ),
                'current_bid': st.column_config.NumberColumn(
                    'Bid', format='$%.2f',
                ),
                'next_bid': st.column_config.NumberColumn(
                    'Next', format='$%.2f',
                ),
                'est_cost': st.column_config.NumberColumn(
                    'Est. Cost', format='$%.2f',
                ),
                'bolo_brand': st.column_config.TextColumn('🎯 BOLO'),
                'title': st.column_config.TextColumn(
                    'Title', width='large',
                ),
            },
        )
        _pg1, _pg2 = st.columns([2, 1])
        with _pg1:
            if st.button(
                f"🛡️ Run audit + comps pipeline on "
                f"{len(_prev_df):,} lots",
                type="primary",
                width='stretch',
                key="confirm_run_pipeline",
                help="Fires the AI condition audit (Claude API), then "
                     "the comps credit gate, then eBay pricing. Big "
                     "auctions with BOLO matches get a scope chooser "
                     "first.",
            ):
                # Persist the FILTERED set as the working frame so the
                # audit only ever sees the trimmed lots. (The pre-
                # filter counts stay visible via the caption above.)
                st.session_state.selected_leads = (
                    _prev_df.reset_index(drop=True)
                )
                st.session_state._audit_confirmed = True
                st.rerun()
        with _pg2:
            if st.button(
                "← Back to auctions",
                width='stretch',
                key="preview_back_btn",
            ):
                st.session_state.selected_leads = pd.DataFrame()
                st.session_state.current_auction = None
                st.session_state.audit_results = {}
                st.session_state.phase1_leads = pd.DataFrame()
                st.rerun()
        st.stop()

    # ================================================================
    # AUDIT-SCOPE CHOOSER (pre-audit gate for big auctions)
    # When an auction has > BIG_AUCTION_THRESHOLD lots AND at least one
    # BOLO match AND no audit has run yet, block the auto-audit and ask
    # the user whether to analyze the full auction or just the BOLO
    # subset. Solves the "11000 lots but only 30 BOLOs" cost-blowup
    # case the user flagged: running Claude audit on 11000 generic
    # estate-sale lots burns API budget without finding the brand
    # hits any faster than scoping straight to BOLO.
    # ================================================================
    BIG_AUCTION_THRESHOLD = 500
    audit_scope = st.session_state.get('_audit_scope')
    # Multi-auction BOLO scan-all loads `selected_leads` already pre-
    # filtered to BOLO matches — there is no "full auction" inside this
    # view to expand to (you'd have to drill into individual auctions
    # via the per-row links). The scope chooser shouldn't fire because
    # both choices would point at the same dataset. Detect via the
    # synthetic-auction name prefix the BOLO scan uses.
    _is_multi_auction_bolo_scan = bool(
        current_auction
        and isinstance(current_auction, str)
        and current_auction.startswith("🎯 BOLO scan")
    )
    _is_multi_auction_keyword_scan = bool(
        current_auction
        and isinstance(current_auction, str)
        and current_auction.startswith("🔍 Keyword:")
    )
    needs_scope_choice = (
        not has_audit
        and not audit_running
        and audit_scope is None
        and len(leads_df) > BIG_AUCTION_THRESHOLD
        and _BOLO_MATCHER.loaded
        and not _is_multi_auction_bolo_scan
        and not _is_multi_auction_keyword_scan
    )
    # Multi-auction scan-all path: auto-mark the scope as 'bolo' so
    # downstream gates (the audit's `_audit_scope` checks, the comps
    # credit gate's BOLO-aware spend estimate) treat the data correctly.
    # Keyword scan uses the same scope marker — the loaded set is
    # already a curated subset of the full auction universe, so the
    # scope chooser shouldn't try to ask "BOLO or full?" again.
    if (
        (_is_multi_auction_bolo_scan or _is_multi_auction_keyword_scan)
        and audit_scope is None
        and not has_audit
    ):
        st.session_state._audit_scope = 'bolo'
        st.session_state._audit_scope_total_lots = len(leads_df)
        audit_scope = 'bolo'

    # Compute BOLO count on the loaded lots once so both the chooser
    # and the in-place filter share the same number. Cheap (regex).
    if needs_scope_choice or audit_scope == 'bolo':
        _scope_df = _compute_bolo_columns(leads_df)
        _bolo_match_count = int(_scope_df['bolo_brand'].notna().sum()) \
            if 'bolo_brand' in _scope_df.columns else 0
    else:
        _scope_df = None
        _bolo_match_count = 0

    # Render the chooser as a card and st.stop() until the user picks.
    # Only show it when there's actually something to choose between
    # — zero BOLO matches means the only sensible scope is the full
    # auction, so we auto-select 'full' instead of asking.
    if needs_scope_choice:
        if _bolo_match_count == 0:
            st.session_state._audit_scope = 'full'
            st.session_state._audit_scope_total_lots = len(leads_df)
            st.rerun()
        st.markdown("---")
        st.markdown("### 🎯 Big auction — pick a scope before running audit")
        st.caption(
            f"This auction has **{len(leads_df):,}** lots. Running audit "
            "on the whole thing burns Claude API on every generic estate-"
            f"sale lot. The BOLO matcher already spotted **"
            f"{_bolo_match_count}** lot(s) matching a brand on your watch "
            "list — analyzing just those is much cheaper if the rest of "
            "the auction is unlikely to have premium items."
        )
        sc1, sc2 = st.columns([1, 1])
        with sc1:
            if st.button(
                f"🎯 BOLO only — analyze {_bolo_match_count:,} matching lot(s)",
                type="primary",
                width='stretch',
                key="audit_scope_bolo",
                help="Filter selected_leads down to BOLO-matched lots, "
                     "then run audit + comps only on those. Cheapest "
                     "path. You can switch to the full auction later "
                     "via the analysis-view header.",
            ):
                # Filter selected_leads in place so all downstream
                # steps see only the BOLO subset. Save the original
                # lot count for the header caption.
                st.session_state._audit_scope = 'bolo'
                st.session_state._audit_scope_total_lots = len(leads_df)
                st.session_state.selected_leads = (
                    _scope_df[_scope_df['bolo_brand'].notna()]
                    .reset_index(drop=True)
                )
                st.rerun()
        with sc2:
            if st.button(
                f"🌐 Full auction — analyze all {len(leads_df):,} lots",
                width='stretch',
                key="audit_scope_full",
                help="Run audit + comps across every lot. Choose this "
                     "when the auction's mix is unknown or you want "
                     "the BOLO matcher to confirm against post-audit "
                     "enriched titles.",
            ):
                st.session_state._audit_scope = 'full'
                st.session_state._audit_scope_total_lots = len(leads_df)
                st.rerun()
        st.markdown("---")
        st.stop()

    if not (audit_running or comps_running):
        # Stale-audit retry: if the cached audit columns are mostly
        # `no_api_key` and we now DO have a key, re-fire the audit so
        # genuinely-broken/untested items get red-flagged before they
        # reach the comps stage. Tracked separately from the regular
        # 'audit' attempt so a one-time stale retry doesn't loop.
        if (has_audit and audit_looks_stale
                and 'audit_stale_retry' not in auto_attempts):
            auto_attempts.add('audit_stale_retry')
            auto_attempts.discard('audit')  # let audit step re-run
            st.session_state.audit_running = True
            st.rerun()
        elif not has_audit and 'audit' not in auto_attempts:
            auto_attempts.add('audit')
            st.session_state.audit_running = True
            st.rerun()
        elif (has_audit and not audit_is_dead
              and not has_comps_data and 'comps_first' not in auto_attempts):
            # Credit-spend confirmation gate. Don't auto-fire the
            # first comps run until the user has explicitly confirmed
            # the credit cost for THIS auction. The flag is per-auction
            # (cleared by _load_auction_for_analysis on every new
            # auction load) so each one needs its own confirmation.
            # `audit_is_dead` blocks this branch entirely — pricing
            # unvetted lots wastes ScrapingBee credits (see the dead-
            # audit banner below).
            if st.session_state.get('_comps_credit_confirmed', False):
                auto_attempts.add('comps_first')
                st.session_state.comps_running = True
                st.rerun()
            # else: fall through; the confirmation gate below renders
        elif (has_audit and not audit_is_dead and has_more_chunks
              and st.session_state.get('_comps_credit_confirmed', False)
              and st.session_state.get('_comps_error_count', 0) < 2):
            # Auto-continue chunks. _comps_has_more flips False when we
            # genuinely run out, so this self-terminates.
            #
            # NOTE: this used to also require `has_comps_data` (≥30% of
            # eligible lots priced). That gate silently killed resumption
            # whenever an early batch errored out: 35/155 priced = 23%,
            # threshold not met, `comps_first` already consumed → the
            # pipeline hung forever with 100+ lots unpriced and no error
            # shown (Longview 6/12 auction). `_comps_has_more` is the
            # authoritative "work remains" signal — trust it alone.
            # `_comps_error_count` (incremented in the comps exception
            # handler, reset on any successful batch) breaks the loop
            # after 2 consecutive failures so a persistent error (dead
            # API key, out of credits) can't rerun-spin forever; the
            # stall banner below takes over from there.
            st.session_state.comps_running = True
            st.rerun()

    # ================================================================
    # DEAD-AUDIT BANNER — audit ran but >50% of rows got no usable
    # verdict (no_api_key / api_failed). Comps are gated off above;
    # this explains why and offers the retry. The one-shot stale
    # retry has usually already fired by the time this renders, so
    # landing here means the retry ALSO failed — check the preflight
    # banner + terminal for the root cause.
    # ================================================================
    if audit_is_dead and not audit_running and not comps_running:
        st.error(
            "🛑 **Audit is not producing verdicts** — most lots have "
            "audit_source `no_api_key` / `api_failed`. Price comps are "
            "**blocked** so ScrapingBee credits aren't spent on unvetted "
            "lots (no condition red-flagging is active). Check the "
            "preflight banner at the top of the page and the terminal "
            "`[AUDIT]` lines for the root cause, fix it, then retry."
        )
        if st.button("🔁 Re-run audit", key="dead_audit_retry",
                     type="primary"):
            st.session_state.pop('_auto_pipeline_attempts', None)
            st.session_state.pop('_preflight_issues', None)
            st.session_state.audit_running = True
            st.rerun()

    # ================================================================
    # COMPS-STALLED BANNER
    # Renders when work remains but auto-resume has given up after
    # repeated failures. Without this, a stalled run looks identical
    # to a finished one — the user has no idea 100+ lots were never
    # priced (they'd have to notice the ❓ count themselves).
    # Two entry conditions:
    #   - has_more_chunks + 2 consecutive failures: mid-run stall
    #     (auto-resume retried once and gave up)
    #   - not has_comps_data + any failure: FIRST batch died before
    #     `_comps_has_more` was ever written, so has_more can't be
    #     the signal — the near-empty est_resale column is.
    # ================================================================
    if (not audit_running and not comps_running
            and st.session_state.get('_comps_error_count', 0) >= 1
            and (
                (has_more_chunks
                 and st.session_state._comps_error_count >= 2)
                or not has_comps_data
            )):
        _last_err = (st.session_state.get('_comps_stats') or {}).get(
            'last_msg', '(no error captured)'
        )
        st.error(
            f"⚠️ **Comps stalled** — {st.session_state._comps_error_count} "
            f"consecutive batch failures; lots remain unpriced. "
            f"Last error: {_last_err}"
        )
        if st.button("🔁 Resume comps", key="resume_stalled_comps",
                     type="primary"):
            st.session_state._comps_error_count = 0
            st.session_state.comps_running = True
            st.rerun()

    # ================================================================
    # CREDIT-SPEND CONFIRMATION GATE
    # The auto-pipeline used to fire comps automatically after the
    # audit completed — burning 5–25k ScrapingBee credits without any
    # explicit user action. The user asked for a "Are you sure? This
    # spends X credits out of Y available" interstitial before any
    # such spend.
    #
    # Render conditions: audit is done, comps haven't run yet, and the
    # per-auction confirmation flag isn't set. The user clicks
    # "Confirm and run comps" -> _comps_credit_confirmed flips True ->
    # the auto-pipeline above re-fires comps on the next rerun.
    # ================================================================
    needs_credit_confirmation = (
        has_audit
        and not audit_is_dead   # dead audit → comps gated, no point confirming spend
        and not has_comps_data
        and not audit_running
        and not comps_running
        and not st.session_state.get('_comps_credit_confirmed', False)
    )
    if needs_credit_confirmation:
        ar_for_estimate_raw = st.session_state.get('audit_results')

        # Spend-ledger heads-up: how much of this auction is already
        # paid for from previous sessions.
        try:
            from scraper import comped_lots as _sl_stats_mod
            _sl_stats = _sl_stats_mod.stats()
            if _sl_stats['total']:
                st.caption(
                    f"💾 Spend ledger: **{_sl_stats['priced']:,}** lots "
                    f"already priced + {_sl_stats['empty_attempts']:,} "
                    f"known-empty across all sessions — any overlap "
                    f"with this auction restores free."
                )
        except Exception:
            pass

        # ---- Spend-cap knobs ----
        # Surface trim controls right at the gate so the user can dial
        # cost down BEFORE confirming. Three knobs:
        #   - Min bid floor: drop lots below $X (kills cheap junk)
        #   - Tier 1 BOLO only: drop tier 2/3 BOLO matches
        #   - Top-N cap: only comp the N highest-bid lots
        # Each knob is wired to session_state so it persists across
        # reruns within the same gate. Defaults are the existing
        # session values when the user has set them previously.
        ar_for_estimate = ar_for_estimate_raw

        # Apply BOLO column for tier filtering
        ar_with_bolo = (
            _compute_bolo_columns(ar_for_estimate_raw)
            if isinstance(ar_for_estimate_raw, pd.DataFrame)
            and not ar_for_estimate_raw.empty
            else None
        )

        # Detect "100% BOLO saturation" — typical of the multi-auction
        # scan-all flow where every loaded lot is already a BOLO match.
        # In that mode the "BOLO subset" framing is misleading because
        # subset == full set; we adjust labeling and offer different
        # cost-cutting knobs (tier filter + bid floor + top-N cap).
        if ar_with_bolo is not None and 'bolo_brand' in ar_with_bolo.columns:
            _bolo_saturation_pct = (
                ar_with_bolo['bolo_brand'].notna().sum()
                / max(len(ar_with_bolo), 1)
            )
        else:
            _bolo_saturation_pct = 0.0
        _is_bolo_saturated = _bolo_saturation_pct >= 0.98

        # ---- Preset picker ----
        # Three presets cover ~90% of usage: Conservative (tight budget,
        # high-quality only), Standard (balanced default), Aggressive
        # (max coverage). Clicking a preset writes values into the
        # underlying knob session-state keys; the Advanced expander
        # below still exposes every individual knob for power users.
        st.caption("**Pick a spend preset** _(or fine-tune in Advanced below)_")
        pp_c1, pp_c2, pp_c3 = st.columns(3)
        with pp_c1:
            if st.button(
                "⚡ Conservative",
                width='stretch',
                key="spend_preset_conservative",
                help="Top 50 by bid · Tier-1 BOLO only · $20 floor · "
                     "$50 next-bid cap · STR ≥ 50% · dedupe ON. "
                     "Targets the highest-quality bargains; ~1k credits "
                     "typical on a 2k-lot scan.",
            ):
                st.session_state['_gate_min_bid_filter'] = 20.0
                st.session_state['comps_skip_above_bid'] = 50.0
                st.session_state['_gate_top_n_filter'] = 50
                st.session_state['_gate_tier1_only_filter'] = True
                st.session_state['comps_skip_below_str'] = 50.0
                st.session_state['comps_dedup_titles'] = True
                st.session_state['comps_use_melt_floor'] = False
                st.session_state.pop('_gate_category_filter', None)
                st.toast("⚡ Conservative preset applied", icon="⚡")
                st.rerun()
        with pp_c2:
            if st.button(
                "💰 Standard",
                width='stretch',
                key="spend_preset_standard",
                help="Top 200 by bid · all tiers · $5 floor · $100 "
                     "next-bid cap · dedupe ON. Tighter than Aggressive "
                     "(saves credits) but skips low-bid lots that often "
                     "have the best margins.",
            ):
                st.session_state['_gate_min_bid_filter'] = 5.0
                st.session_state['comps_skip_above_bid'] = 100.0
                st.session_state['_gate_top_n_filter'] = 200
                st.session_state['_gate_tier1_only_filter'] = False
                st.session_state['comps_skip_below_str'] = 0.0
                st.session_state['comps_dedup_titles'] = True
                st.session_state['comps_use_melt_floor'] = False
                st.session_state.pop('_gate_category_filter', None)
                st.toast("💰 Standard preset applied", icon="💰")
                st.rerun()
        with pp_c3:
            if st.button(
                "🌐 Aggressive",
                type="primary",
                width='stretch',
                key="spend_preset_aggressive",
                help="No top-N cap · all tiers · $0 floor · no next-bid "
                     "cap · all categories · dedupe ON. **Default** — "
                     "maximum coverage. Expensive on BOLO-saturated "
                     "scans; click Standard to clamp down.",
            ):
                st.session_state['_gate_min_bid_filter'] = 0.0
                st.session_state['comps_skip_above_bid'] = 0.0
                st.session_state['_gate_top_n_filter'] = 0
                st.session_state['_gate_tier1_only_filter'] = False
                st.session_state['comps_skip_below_str'] = 0.0
                st.session_state['comps_dedup_titles'] = True
                st.session_state['comps_use_melt_floor'] = False
                st.session_state.pop('_gate_category_filter', None)
                st.toast("🌐 Aggressive preset applied", icon="🌐")
                st.rerun()

        # Draw the per-knob spend caps in an expander. Auto-expanded
        # for BOLO-saturated runs (where the user probably wants to
        # see the cuts before confirming a big spend). Power users
        # can override any individual knob; the preset picker just
        # writes sensible starting values.
        with st.expander(
            "🎚️ Advanced — fine-tune individual spend caps",
            expanded=_is_bolo_saturated,
        ):
            cap_c1, cap_c2 = st.columns(2)
            with cap_c1:
                _gate_min_bid = st.number_input(
                    "Min bid floor ($)",
                    min_value=0.0,
                    value=float(
                        st.session_state.get('_gate_min_bid_filter', 0.0)
                    ),
                    step=1.0, format="%.2f",
                    key="_gate_min_bid_filter",
                    help="Skip lots whose current bid is below this. "
                         "Cheap-junk filter — at low bids the comp spend "
                         "rarely pays back. Try $5–10.",
                )
            with cap_c2:
                # Default to whatever's in session state (initial $100)
                # so adjusting it persists across reruns. Set to 0 to
                # disable the cap.
                _gate_skip_above = st.number_input(
                    "Skip if next-bid > ($)  (0 = no cap)",
                    min_value=0.0,
                    value=float(
                        st.session_state.get('comps_skip_above_bid', 100.0)
                    ),
                    step=10.0, format="%.2f",
                    key="comps_skip_above_bid",
                    help="Skip the comp step on lots whose next-bid floor "
                         "is above this dollar amount. Lots already bid "
                         "up that high have squeezed margins — the comp "
                         "spend rarely pays back. Default $100. Lowest "
                         "spend impact: try $50 if you only want to "
                         "auto-evaluate true bargain finds.",
                )

            cap_c3, cap_c4 = st.columns(2)
            with cap_c3:
                _gate_top_n = st.number_input(
                    "Cap to top N by bid (0 = no cap)",
                    min_value=0,
                    value=int(
                        st.session_state.get('_gate_top_n_filter', 0)
                    ),
                    step=50,
                    key="_gate_top_n_filter",
                    help="Sort eligible lots by current bid, descending, "
                         "and keep only the top N. Set to e.g. 200 to "
                         "spend on the highest-value lots first; the "
                         "long tail can be re-run later if needed.",
                )
            with cap_c4:
                _has_tier_data = (
                    ar_with_bolo is not None
                    and 'bolo_tier' in ar_with_bolo.columns
                    and ar_with_bolo['bolo_tier'].notna().any()
                )
                _gate_tier1_only = st.checkbox(
                    "Tier 1 BOLO only",
                    value=bool(
                        st.session_state.get('_gate_tier1_only_filter', False)
                    ),
                    key="_gate_tier1_only_filter",
                    disabled=not _has_tier_data,
                    help="Restrict to tier-1 BOLO matches (the curated "
                         "highest-resale brands). Drops tier 2/3 hits "
                         "for cheaper, higher-quality runs.",
                )

            # Title-fingerprint dedup checkbox. Big credit-saver in
            # jewelry / commodity-glass scans with many near-identical
            # lots. The post-comp clone step ensures siblings still
            # show comp data in the results table — they just don't
            # cost credits to compute.
            cap_d1, cap_d2 = st.columns([1, 1])
            with cap_d1:
                _gate_dedup = st.checkbox(
                    "Dedupe near-identical titles",
                    value=bool(
                        st.session_state.get('comps_dedup_titles', True)
                    ),
                    key="comps_dedup_titles",
                    help="Group lots whose titles match a normalized "
                         "fingerprint AND share a BOLO brand. Comp "
                         "ONE per group, clone result to siblings. "
                         "Saves ~30-70% credits on jewelry-saturated "
                         "auctions. Default ON.",
                )
            with cap_d2:
                _gate_melt = st.checkbox(
                    "Melt-value floor for precious metals",
                    value=bool(
                        st.session_state.get('comps_use_melt_floor', False)
                    ),
                    key="comps_use_melt_floor",
                    help="When a lot's title contains both karat "
                         "(10K/14K/18K/22K/24K, sterling, platinum) "
                         "AND weight (Xg), price est_resale = melt "
                         "value × premium factor (default 1.4) "
                         "instead of running an eBay comp. Skips "
                         "ScrapingBee credits entirely on these lots. "
                         "Best for liquidation / volume-melt plays. "
                         "Conservative — signed maker jewelry "
                         "(Cartier 14k) clears 30-100% above this "
                         "estimate via eBay; uncheck for those.",
                )

        # Build the post-cap dataframe used for cost estimates and for
        # the actual comp run. Cap state is stashed so the comps run
        # below knows to apply the same trim.
        if isinstance(ar_for_estimate_raw, pd.DataFrame) and not ar_for_estimate_raw.empty:
            _capped = ar_for_estimate_raw.copy()
            _trim_reasons = []

            if _gate_min_bid > 0 and 'current_bid' in _capped.columns:
                _before = len(_capped)
                _capped = _capped[
                    pd.to_numeric(_capped['current_bid'], errors='coerce')
                    .fillna(0) >= _gate_min_bid
                ]
                _cut = _before - len(_capped)
                if _cut:
                    _trim_reasons.append(
                        f"{_cut} dropped below ${_gate_min_bid:.2f} bid"
                    )

            # Skip-above-bid filter: drop lots whose next-bid floor is
            # above the threshold. Mirrors the runtime filter inside
            # _apply_comps_filters so the metric/button cost reflects
            # the same trim.
            if _gate_skip_above > 0:
                if 'next_bid' in _capped.columns:
                    _bid_col = pd.to_numeric(
                        _capped['next_bid'], errors='coerce'
                    )
                elif 'current_bid' in _capped.columns:
                    _bid_col = pd.to_numeric(
                        _capped['current_bid'], errors='coerce'
                    )
                else:
                    _bid_col = None
                if _bid_col is not None:
                    _before = len(_capped)
                    _capped = _capped[_bid_col.fillna(0) <= _gate_skip_above]
                    _cut = _before - len(_capped)
                    if _cut:
                        _trim_reasons.append(
                            f"{_cut} skipped — next-bid > ${_gate_skip_above:.0f}"
                        )

            if _gate_tier1_only and ar_with_bolo is not None:
                _before = len(_capped)
                _capped_ids = set(_capped.index)
                _tier1_ids = set(
                    ar_with_bolo[ar_with_bolo['bolo_tier'] == 1].index
                )
                _capped = _capped[_capped.index.isin(_tier1_ids)]
                _cut = _before - len(_capped)
                if _cut:
                    _trim_reasons.append(
                        f"{_cut} non-tier-1 BOLO dropped"
                    )

            # NOTE: The category-STR threshold filter that lived here
            # was removed when the cross-session comped_lots registry
            # was dropped. Without that registry there's no historical
            # category-level STR data to threshold against. If you want
            # category-aware filtering back, add a session-only STR
            # tracker first.

            if (_gate_top_n > 0 and len(_capped) > _gate_top_n
                    and 'current_bid' in _capped.columns):
                _before = len(_capped)
                _capped = _capped.sort_values(
                    'current_bid', ascending=False
                ).head(_gate_top_n)
                _cut = _before - len(_capped)
                _trim_reasons.append(f"trimmed to top {_gate_top_n} by bid")

            if _trim_reasons:
                st.success(
                    f"✂️ Caps applied: {' · '.join(_trim_reasons)}"
                )
            ar_for_estimate = _capped
        # Stash the caps so the actual comp run picks them up.
        # `comps_max_lots` already exists in the comps pipeline; the
        # min-bid + tier1-only filters are applied here at the gate
        # by replacing audit_results with the capped df at confirm time.
        if _gate_top_n > 0:
            st.session_state.comps_max_lots = int(_gate_top_n)

        # ---- Spend-ledger overlap: don't quote for lots we already
        # paid for. Lots priced (or attempted-empty) within the ledger
        # TTL restore free at run time — remove them from the estimate
        # frame so the quoted credit cost matches what a re-run will
        # actually spend. This is also the visible proof the ledger
        # saved last run's scrapes: re-running the same scan shows
        # "💾 N already paid" and a much smaller quote.
        _ledger_covered = 0
        if (st.session_state.get('use_spend_ledger', True)
                and isinstance(ar_for_estimate, pd.DataFrame)
                and 'lot_id' in ar_for_estimate.columns
                and not ar_for_estimate.empty):
            try:
                from scraper import comped_lots as _sl_gate
                _ttl_gate = float(
                    st.session_state.get('spend_ledger_ttl_days', 7.0)
                    or 7.0
                )
                _, _n_priced_gate, _blocked_gate = (
                    _sl_gate.overlay_onto_df(
                        ar_for_estimate, ttl_days=_ttl_gate,
                    )
                )
                _covered_ids = set(_blocked_gate)
                # overlay marks priced rows in its returned copy; get
                # their ids by re-checking which input rows the ledger
                # knows as priced within TTL
                _lots_map = _sl_gate._load().get('lots', {})
                from datetime import datetime as _dtg
                for _lid in ar_for_estimate['lot_id'].astype(str):
                    _e = _lots_map.get(_lid)
                    if not _e or _e.get('est_resale') is None:
                        continue
                    try:
                        _age = (
                            _dtg.now()
                            - _dtg.fromisoformat(_e['attempted_at'])
                        ).total_seconds() / 86400.0
                        if _age <= _ttl_gate:
                            _covered_ids.add(_lid)
                    except (ValueError, TypeError, KeyError):
                        pass
                _ledger_covered = int(
                    ar_for_estimate['lot_id'].astype(str)
                    .isin(_covered_ids).sum()
                )
                if _ledger_covered:
                    ar_for_estimate = ar_for_estimate[
                        ~ar_for_estimate['lot_id'].astype(str)
                        .isin(_covered_ids)
                    ]
                    st.success(
                        f"💾 **{_ledger_covered}** of these lots were "
                        f"already paid for in previous runs — they "
                        f"restore free and are excluded from the "
                        f"estimate below."
                    )
            except Exception as _sl_e:
                tlog("LEDGER", f"gate overlap check failed: {_sl_e}")

        eligible_count, est_credits, pc_pct = _estimate_comp_cost_for_audit(
            ar_for_estimate
        )

        bolo_subset_count = 0
        bolo_subset_credits = 0
        bolo_subset_pc_pct = 0.0
        if (ar_with_bolo is not None and 'bolo_brand' in ar_with_bolo.columns):
            bolo_only_df = ar_with_bolo[ar_with_bolo['bolo_brand'].notna()]
            if not bolo_only_df.empty:
                bolo_subset_count, bolo_subset_credits, bolo_subset_pc_pct = (
                    _estimate_comp_cost_for_audit(bolo_only_df)
                )

        offer_bolo_scope = (
            bolo_subset_count > 0
            and eligible_count > BIG_AUCTION_THRESHOLD
            and bolo_subset_count < eligible_count
            and not _is_bolo_saturated
        )

        # Live ScrapingBee usage. Cached 5 min so we don't ping it
        # every rerun. Empty dict on any failure (no key, network).
        from scraper.config_loader import load_config as _cfg_load
        try:
            _cfg = _cfg_load()
            _sb_key = (_cfg.get("scrapingbee") or {}).get("api_key") or ""
        except Exception:
            _sb_key = ""
        usage = _fetch_scrapingbee_usage(_sb_key) if _sb_key else {}
        used = int(usage.get('used_api_credit') or 0)
        cap = int(usage.get('max_api_credit') or 0)
        remaining = max(cap - used, 0) if cap else None

        # Card layout — three metrics + the confirm button.
        st.markdown("---")
        st.markdown("### 💸 Confirm credit spend")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric(
                "Eligible lots",
                f"{eligible_count:,}",
                help="Non-red-flagged lots that need pricing.",
            )
        with m2:
            cost_str = (
                "free (all PC)" if est_credits == 0
                else f"~{est_credits:,}"
            )
            delta = (
                f"{int(round(pc_pct * 100))}% PC-covered"
                if pc_pct > 0 else None
            )
            st.metric(
                "Estimated cost",
                cost_str, delta=delta, delta_color="off",
            )
        with m3:
            if cap:
                st.metric(
                    "ScrapingBee budget",
                    f"{remaining:,} left",
                    delta=f"of {cap:,} this month",
                    delta_color="off",
                )
            else:
                st.metric("ScrapingBee budget", "no key set")

        # Affordability check — if the estimate exceeds what's left,
        # warn the user explicitly instead of letting them click
        # through into a quota-exhausted run.
        affordable = (
            cap == 0 or est_credits == 0
            or (remaining is not None and est_credits <= remaining)
        )
        if not affordable:
            st.error(
                f"⚠️ Estimated cost (**~{est_credits:,}**) exceeds "
                f"remaining budget (**{remaining:,}**). The run will "
                "stall partway through when ScrapingBee returns 401. "
                "Skip this auction or upgrade your plan."
            )
        elif est_credits > 0 and cap and remaining is not None:
            pct_of_budget = est_credits / cap * 100
            if pct_of_budget > 10:
                st.warning(
                    f"This run will spend **{pct_of_budget:.0f}%** of "
                    f"your monthly budget. Consider whether the auction "
                    "is worth that share."
                )

        st.caption(
            "Cost = ~50 credits per non-PC-covered lot (eBay sold + "
            "Mercari sold). PriceCharting lookups are "
            "free; only the ScrapingBee-routed scrapes consume credits."
        )

        if offer_bolo_scope:
            st.info(
                f"🎯 **BOLO scope available**: {bolo_subset_count} of "
                f"{eligible_count} eligible lots match the brand watch "
                f"list. Comping just those costs **~{bolo_subset_credits:,} "
                f"credits** vs **~{est_credits:,}** for the full set "
                f"(saves ~{est_credits - bolo_subset_credits:,})."
            )

        # Compute BOLO count + cost for the Free button label and
        # for the preview banner below. Must be defined BEFORE the
        # preview-banner conditional reads it. When offer_bolo_scope
        # already computed bolo_subset_count/credits, reuse those;
        # otherwise compute from the BOLO-tagged audit_results.
        free_bolo_count = bolo_subset_count if offer_bolo_scope else 0
        free_bolo_credits = bolo_subset_credits if offer_bolo_scope else 0
        if not offer_bolo_scope and ar_with_bolo is not None:
            bolo_only_df = ar_with_bolo[ar_with_bolo['bolo_brand'].notna()]
            if not bolo_only_df.empty:
                free_bolo_count, free_bolo_credits, _ = (
                    _estimate_comp_cost_for_audit(bolo_only_df)
                )

        # Free-mode preview — shown whenever there's a credit gate so the
        # user knows it's available. Estimate which lots WOULD get
        # est_resale via PriceCharting: any lot whose title classifies
        # as tcg / video_game / comic via the PC classifier.
        if pc_pct > 0 or free_bolo_count > 0:
            covered = int(round(pc_pct * eligible_count))
            if _is_bolo_saturated:
                # All eligible lots are BOLO matches (multi-auction
                # scan-all flow). The "Free + BOLO" framing is wrong
                # here because BOLO subset == full set — re-running
                # eBay/Mercari on every BOLO match equals "full". So
                # we surface the genuinely-free option clearly.
                st.info(
                    f"🆓 **Genuinely-free mode available**: every "
                    f"eligible lot is already a BOLO match, so "
                    f"'Free + BOLO' would equal 'Full'. Click the "
                    f"**🆓 Free comps only** button below for a true "
                    f"zero-credit run — only ~{covered} lots will get "
                    f"prices (those covered by PriceCharting / "
                    f"curated catalog); the rest stay "
                    f"un-comped. Use the spend caps above (top N by "
                    f"bid, tier 1 only, min bid floor) to right-size "
                    f"a paid run before confirming."
                )
            else:
                bolo_clause = (
                    f" Plus full eBay/Mercari comps on **{free_bolo_count} "
                    f"BOLO match(es)** (~{free_bolo_credits:,} credits)."
                    if free_bolo_count > 0 else ""
                )
                st.info(
                    f"🆓 **Free + BOLO mode available**: ~{covered} of "
                    f"{eligible_count} eligible lots are likely covered "
                    "by PriceCharting (curated catalog, free)."
                    + bolo_clause
                )

        # Button layout grows with the available scopes:
        #   - Always: 🆓 Free+BOLO / 🌐 Full / 🚫 Skip
        #   - Plus 🎯 BOLO-only when offer_bolo_scope is True
        # 🆓 Free+BOLO mode runs PC + GoCollect for non-BOLO lots
        # (zero credits) AND full eBay/Mercari for BOLO matches
        # (because those are exactly the lots where the comp spend
        # is worth it — they're on the watch list for a reason).
        # When there are no BOLO matches, this is functionally a
        # "0 credits" run.
        free_help = (
            "Run the free curated source (PriceCharting) on "
            "every lot, AND full eBay/Mercari comps on BOLO matches "
            "specifically. BOLO matches are on the watch list because "
            "you want real resale data on them — those still get the "
            "full treatment. Everything else: PriceCharting only "
            "(lots they cover get est_resale; the rest stay empty). "
            f"Cost: ~{free_bolo_credits:,} ScrapingBee credits "
            f"({free_bolo_count} BOLO × ~50 cr each)."
        )
        if offer_bolo_scope:
            cb_bolo, cb_free, cb_full, cb_skip = st.columns([1, 1, 1, 0.6])
            with cb_bolo:
                if st.button(
                    f"🎯 BOLO ({bolo_subset_count} lots, "
                    f"~{bolo_subset_credits:,} cr)",
                    type="primary", width='stretch',
                    disabled=not affordable or bolo_subset_count == 0,
                    key="confirm_comp_credits_bolo",
                    help="Scope to BOLO-matched lots only. Cheapest "
                         "paid path. Use ↗ Expand to full auction in "
                         "the header to comp the rest later.",
                ):
                    bolo_filtered = ar_with_bolo[
                        ar_with_bolo['bolo_brand'].notna()
                    ].reset_index(drop=True)
                    st.session_state.audit_results = bolo_filtered
                    st.session_state.selected_leads = bolo_filtered.copy()
                    st.session_state._audit_scope = 'bolo'
                    st.session_state._audit_scope_total_lots = (
                        st.session_state.get('_audit_scope_total_lots')
                        or len(ar_for_estimate)
                    )
                    st.session_state._comps_credit_confirmed = True
                    st.rerun()
            with cb_free:
                free_label = (
                    f"🆓 Free + BOLO (~{free_bolo_credits:,} cr)"
                    if free_bolo_count > 0
                    else f"🆓 Free ({eligible_count} lots, 0 cr)"
                )
                if st.button(
                    free_label,
                    width='stretch',
                    disabled=eligible_count == 0,
                    key="confirm_comp_credits_free",
                    help=free_help,
                ):
                    if isinstance(ar_for_estimate, pd.DataFrame):
                        st.session_state.audit_results = ar_for_estimate
                        st.session_state.selected_leads = ar_for_estimate.copy()
                    st.session_state._comps_free_only_mode = True
                    st.session_state._comps_free_skip_bolo = False
                    st.session_state._comps_credit_confirmed = True
                    st.rerun()
            with cb_full:
                if st.button(
                    f"🌐 Full ({eligible_count} lots, ~{est_credits:,} cr)",
                    width='stretch',
                    disabled=not affordable or eligible_count == 0,
                    key="confirm_comp_credits_full",
                ):
                    if isinstance(ar_for_estimate, pd.DataFrame):
                        st.session_state.audit_results = ar_for_estimate
                        st.session_state.selected_leads = ar_for_estimate.copy()
                    st.session_state._comps_credit_confirmed = True
                    st.rerun()
            with cb_skip:
                if st.button(
                    "🚫 Skip",
                    width='stretch',
                    key="skip_comp_credits",
                ):
                    auto_attempts.add('comps_first')
                    st.rerun()
        else:
            cb_free, cb_full, cb_skip = st.columns([1, 1.3, 0.7])
            with cb_free:
                # When BOLO-saturated, "Free + BOLO" is misleading —
                # treat it as a true zero-credit run by ALSO setting
                # the BOLO-eBay-skip flag the comp run picks up. When
                # not saturated, free mode runs full eBay/Mercari on
                # BOLO matches (the original behavior).
                if _is_bolo_saturated:
                    free_label = (
                        f"🆓 Free comps only ({eligible_count} lots, 0 cr)"
                    )
                else:
                    free_label = (
                        f"🆓 Free + BOLO comps (~{free_bolo_credits:,} cr)"
                        if free_bolo_count > 0
                        else f"🆓 Free comps only ({eligible_count} lots, 0 cr)"
                    )
                if st.button(
                    free_label,
                    width='stretch',
                    disabled=eligible_count == 0,
                    key="confirm_comp_credits_free",
                    help=(
                        "Zero-credit run — PriceCharting only. "
                        "Skips eBay/Mercari entirely (including on BOLO "
                        "matches) because every eligible lot is already "
                        "BOLO-matched and running eBay on all of them "
                        "would equal a full paid run."
                        if _is_bolo_saturated
                        else free_help
                    ),
                ):
                    # Commit the capped df so the comp run sees only
                    # the trimmed subset.
                    if isinstance(ar_for_estimate, pd.DataFrame):
                        st.session_state.audit_results = ar_for_estimate
                        st.session_state.selected_leads = ar_for_estimate.copy()
                    st.session_state._comps_free_only_mode = True
                    if _is_bolo_saturated:
                        # In saturated mode, "Free" really means free —
                        # skip BOLO eBay/Mercari too. The comp run reads
                        # _comps_free_skip_bolo to decide whether to
                        # exempt BOLO matches from the free-only filter.
                        st.session_state._comps_free_skip_bolo = True
                    else:
                        st.session_state._comps_free_skip_bolo = False
                    st.session_state._comps_credit_confirmed = True
                    st.rerun()
            with cb_full:
                if st.button(
                    f"✅ Confirm and run comps (~{est_credits:,} credits)",
                    type="primary", width='stretch',
                    disabled=not affordable or eligible_count == 0,
                    key="confirm_comp_credits",
                ):
                    # Commit the capped df so the comp run respects
                    # min-bid / tier-1 / top-N caps.
                    if isinstance(ar_for_estimate, pd.DataFrame):
                        st.session_state.audit_results = ar_for_estimate
                        st.session_state.selected_leads = ar_for_estimate.copy()
                    st.session_state._comps_credit_confirmed = True
                    st.rerun()
            with cb_skip:
                if st.button(
                    "🚫 Skip",
                    width='stretch',
                    key="skip_comp_credits",
                    help="Bail on the comp run for this auction. The audit "
                         "results render below; you can still hit the "
                         "🔄 Re-run comps button later if you change your "
                         "mind.",
                ):
                    auto_attempts.add('comps_first')
                    st.rerun()

        st.markdown("---")
        st.stop()  # Don't render results panel until confirmed

    # ================================================================
    # PIPELINE-ACTIVE GATE
    # When the auto-pipeline is mid-flight (audit running, comps
    # running, image enrichment running, OR comp chunks still pending),
    # we hide the results panel entirely and show only a "🍳 Cooking…"
    # screen with the audit/comps status panels. The user said the
    # streaming-results layout was wonky — they prefer to wait and see
    # final results all at once. The split layout + tabs render below
    # only when the pipeline is fully done.
    # ================================================================
    # img_enrich_running is included now that the auto-pipeline fires
    # image enrichment between audit and comps for generic-titled lots.
    # The execution block is hoisted below alongside audit/comps so the
    # cooking screen covers all three steps uniformly.
    pipeline_active = (
        audit_running or comps_running or has_more_chunks
    )

    # No-op stubs for live-update hooks — used by the audit/comps
    # execution blocks below. During cooking we don't render results,
    # so these stubs make the live_callback hooks safe to call without
    # an active placeholder.
    class _NullPlaceholder:
        def markdown(self, *a, **k):
            pass

        def container(self):
            from contextlib import nullcontext
            return nullcontext()

    last_priced_placeholder = _NullPlaceholder()

    def _render_live_results():
        # No-op during cooking. Real rendering happens inline in the
        # `if not pipeline_active` block below once everything finishes.
        pass

    def _render_red_flag_review():
        pass

    if pipeline_active:
        # While cooking, show a heading. The audit/comps st.status
        # blocks below render their progress directly underneath.
        st.markdown("### 🍳 Cooking your auction…")
        st.caption(
            "Audit + price comps run automatically. Final results "
            "appear once everything completes — hang tight."
        )

    # ================================================================
    # AUDIT-RUNNING execution block — hoisted out of tab_audit so the
    # live st.status updates show in the main flow regardless of which
    # tab the user has open.
    # ================================================================
    if audit_running:
        _keep_screen_awake()
        st.info(
            "🔋 Keeping screen awake while this runs. "
            "If your phone still locks, set **Auto-Lock → Never** in your phone's "
            "display settings before kicking off long runs."
        )
        try:
            _audit_last = [0.0]
            def _on_audit_progress(processed, total_items):
                now = _time.time()
                if (now - _audit_last[0]) < 0.3 and processed < total_items:
                    return
                _audit_last[0] = now
                _render_live_results()

            tlog("AUDIT", f"starting · {len(leads_df):,} leads to audit")
            _audit_t0 = _time.time()
            st.session_state.audit_results = _run_ai_audit(
                leads_df, on_progress=_on_audit_progress,
            )
            _audit_elapsed = _time.time() - _audit_t0
            tlog("AUDIT",
                 f"done in {_audit_elapsed:.1f}s",
                 f"· {len(st.session_state.audit_results)} verdicts written")
            _save_current_auction_to_cache()
        except Exception as e:
            tlog("AUDIT", f"FAILED: {type(e).__name__}: {e}")
            st.error(f"Audit failed: {e}")
        finally:
            st.session_state.audit_running = False
        st.rerun()

    # ================================================================
    # IMAGE-ENRICHMENT execution block — hoisted out of tab_audit so the
    # auto-pipeline can run it between audit and comps for generic-titled
    # lots. The cooking screen covers it via pipeline_active.

    # ================================================================
    # COMPS-RUNNING execution block — same hoist as audit. Reads filter
    # values from session_state (set by the Comps tab settings panel
    # below) so it's decoupled from tab rendering.
    # ================================================================
    if comps_running:
        _keep_screen_awake()
        st.info(
            "🔋 Keeping screen awake while this runs. "
            "If your phone still locks, set **Auto-Lock → Never** in your phone's "
            "display settings before kicking off long runs."
        )
        ar = st.session_state.audit_results
        chunk_size = int(st.session_state.get('comps_chunk_size', 200))
        any_comped = (
            isinstance(ar, pd.DataFrame)
            and 'est_resale' in ar.columns
            and ar['est_resale'].notna().any()
        )
        try:
            # First chunk on a fresh auction? Reset the cached STR map so
            # we re-sample for the new auction. Also reset the scrape-
            # stats counters so the "scraper blocked" warning below
            # reflects this run only, not lifetime.
            if not any_comped:
                st.session_state.pop('_comps_auction_str_map', None)
                from scraper.ebay_prices import EbayPriceLookup
                EbayPriceLookup.reset_scrape_stats()

            # Two streams of updates per lot:
            #   - Per-lot ticker (no throttle): a one-line summary that
            #     updates instantly. Cheap, makes "live" obvious.
            #   - Full table re-render (throttled to 300ms).
            import time as _time
            _last = [0.0]

            def _on_lot_priced(completed, total_items, last_lot=None):
                if last_lot is not None:
                    title = (last_lot.get('title') or '')[:60]
                    resale = last_lot.get('resale')
                    roi = last_lot.get('roi')
                    def _tick_int(v):
                        try:
                            return 0 if (v is None or pd.isna(v)) else int(v)
                        except (TypeError, ValueError):
                            return 0
                    comps_n = (
                        _tick_int(last_lot.get('ebay_comps'))
                        + _tick_int(last_lot.get('mercari_comps'))
                    )
                    bits = [f"🔥 **{completed}/{total_items}** priced"]
                    if title:
                        if pd.notna(resale):
                            detail = f"*{title}* → **${float(resale):.2f}**"
                            if pd.notna(roi):
                                detail += f" (ROI {int(roi)}%)"
                            if comps_n:
                                detail += f" · {comps_n} comps"
                            bits.append(detail)
                        else:
                            bits.append(f"*{title}* → no comps found")
                    try:
                        last_priced_placeholder.markdown(" · ".join(bits))
                    except Exception:
                        pass
                now = _time.time()
                if (now - _last[0]) < 0.3 and completed < total_items:
                    return
                _last[0] = now
                _render_live_results()

            tlog("COMPS",
                 f"chunk starting · chunk_size={chunk_size}",
                 f"· {len(ar):,} audit rows in scope")
            _comps_t0 = _time.time()
            updated, found, processed, has_more = _run_ebay_comps_chunk(
                ar, chunk_size=chunk_size,
                on_lot_priced=_on_lot_priced,
            )
            _comps_elapsed = _time.time() - _comps_t0
            tlog("COMPS",
                 f"chunk done in {_comps_elapsed:.1f}s",
                 f"· processed {processed} · priced {found}",
                 f"· has_more={has_more}")
            st.session_state.audit_results = updated
            st.session_state._comps_has_more = has_more
            # Any successful batch resets the consecutive-failure
            # counter — auto-resume stays alive as long as batches
            # keep landing, even if occasional ones error.
            st.session_state._comps_error_count = 0
            # Accumulate per-batch stats so the post-pipeline view can show
            # exactly what happened (helps explain "nothing highlights"
            # without forcing the user to dig through st.status blocks).
            stats_total = st.session_state.setdefault('_comps_stats', {
                'batches': 0, 'attempted': 0, 'priced': 0,
                'last_msg': '', 'has_more': False,
            })
            stats_total['batches'] += 1
            stats_total['attempted'] += processed
            stats_total['priced'] += found
            stats_total['has_more'] = has_more
            _save_current_auction_to_cache()
            if processed:
                tail = ("  More lots remain — continuing automatically…"
                        if has_more else
                        "  ✅ All eligible lots have been comped.")
                _lf = int(st.session_state.get('_last_ledger_filled', 0) or 0)
                _lb = int(st.session_state.get('_last_ledger_blocked', 0) or 0)
                _ledger_bit = ""
                if _lf or _lb:
                    _ledger_bit = (
                        f"  💾 {_lf} restored free from the spend ledger"
                        + (f" + {_lb} known-empty skipped" if _lb else "")
                        + "."
                    )
                stats_total['last_msg'] = (
                    f"Batch complete — priced {found}/{processed} "
                    f"lot(s).{_ledger_bit}{tail}"
                )
                st.success(stats_total['last_msg'])
            else:
                good_count = int(
                    (~ar.get('red_flag', pd.Series([], dtype=bool))
                       .fillna(False)).sum()
                ) if isinstance(ar, pd.DataFrame) else 0
                try:
                    preview, _, summary = _apply_comps_filters(
                        ar[~ar['red_flag'].fillna(False)]
                        if isinstance(ar, pd.DataFrame) and 'red_flag' in ar.columns
                        else ar
                    )
                    preview_count = len(preview)
                except Exception:
                    preview_count = 0
                    summary = "(filter preview failed)"
                msg = (
                    f"No eligible lots to comp this batch. "
                    f"**{good_count}** good+ lots in the audit, "
                    f"**{preview_count}** survive the current filters "
                    f"({summary}). "
                    "Loosen the filters in the **💰 Comps** tab "
                    "(*Narrow down what to comp* expander) and use the "
                    "manual run button to retry."
                )
                st.warning(msg)
                stats_total = st.session_state.setdefault('_comps_stats', {
                    'batches': 0, 'attempted': 0, 'priced': 0,
                    'last_msg': '', 'has_more': False,
                })
                stats_total['last_msg'] = msg
                stats_total['has_more'] = False
        except Exception as e:
            import traceback
            err = f"Price comps failed: {type(e).__name__}: {e}"
            st.error(err)
            with st.expander("Show traceback"):
                st.code(traceback.format_exc(), language="python")
            # Count consecutive failures. The auto-pipeline retries
            # while this is < 2; after that the stall banner renders
            # a manual Resume button instead (prevents rerun-spin on
            # persistent errors like a dead ScrapingBee key).
            # IMPORTANT: do NOT touch `_comps_has_more` here — work
            # genuinely remains, and wiping the flag was what made
            # stalled runs indistinguishable from finished ones.
            st.session_state._comps_error_count = (
                st.session_state.get('_comps_error_count', 0) + 1
            )
            tlog("COMPS",
                 f"batch FAILED ({st.session_state._comps_error_count} "
                 f"consecutive) · {err}")
            stats_total = st.session_state.setdefault('_comps_stats', {
                'batches': 0, 'attempted': 0, 'priced': 0,
                'last_msg': '', 'has_more': False,
            })
            stats_total['last_msg'] = err
        finally:
            st.session_state.comps_running = False

        # Surface scraper-blocked status. Both eBay sold and Mercari
        # silently fall through to "active eBay listings" when their
        # anti-bot trips a 403 — the user otherwise sees no signal that
        # their entire sold-history pipeline is dead and every priced
        # row in the table came from an untrustworthy active fallback.
        try:
            from scraper.ebay_prices import EbayPriceLookup
            stats = EbayPriceLookup.get_scrape_stats()
            ebay_total = stats.get('ebay_sold_attempts', 0)
            ebay_blocked = stats.get('ebay_sold_blocked', 0)
            merc_total = stats.get('mercari_attempts', 0)
            merc_blocked = stats.get('mercari_blocked', 0)
            sb_calls = stats.get('scrapingbee_calls', 0)
            sb_credits = stats.get('scrapingbee_credits', 0)
            ebay_block_rate = (ebay_blocked / ebay_total) if ebay_total else 0.0
            merc_block_rate = (merc_blocked / merc_total) if merc_total else 0.0

            # ScrapingBee-specific errors get top billing — a quota or
            # auth failure means the proxy isn't even reaching eBay,
            # so the lower-level "scraper blocked" message would be
            # misleading.
            sb_quota = stats.get('scrapingbee_quota_fail', 0)
            sb_auth = stats.get('scrapingbee_auth_fail', 0)
            if sb_quota > 0:
                st.error(
                    f"💸 **ScrapingBee plan is exhausted** — "
                    f"{sb_quota} request(s) returned 'Monthly API "
                    "calls limit reached'. eBay/Mercari sold scraping "
                    "is dead until your plan resets or you upgrade. "
                    "Visit https://app.scrapingbee.com/account/usage "
                    "to check your quota or upgrade to the Freelance "
                    "tier ($49/mo for 100k credits)."
                )
            elif sb_auth > 0:
                st.error(
                    f"🔑 **ScrapingBee auth failed** on {sb_auth} "
                    "request(s) — your `scrapingbee.api_key` in "
                    "config.json is likely wrong. Re-copy the key "
                    "from https://app.scrapingbee.com/account."
                )

            # Positive signal: ScrapingBee was used and the eBay sold
            # path is actually getting through. Show a small caption
            # so the user can track credit burn against their plan.
            if sb_calls > 0:
                st.caption(
                    f"🐝 ScrapingBee used for {sb_calls} scrape(s) "
                    f"(~{sb_credits} credits)."
                )

            if (ebay_total >= 3 and ebay_block_rate > 0.5) or \
               (merc_total >= 3 and merc_block_rate > 0.5):
                bits = []
                if ebay_block_rate > 0.5:
                    bits.append(
                        f"**eBay sold** blocked on {ebay_blocked}/{ebay_total} "
                        f"requests ({int(ebay_block_rate * 100)}%)"
                    )
                if merc_block_rate > 0.5:
                    bits.append(
                        f"**Mercari sold** blocked on {merc_blocked}/{merc_total} "
                        f"requests ({int(merc_block_rate * 100)}%)"
                    )
                st.error(
                    "🚫 **Sold-listing scrapers are being blocked** — "
                    + "; ".join(bits) + ". "
                    "Most prices in this run came from eBay's **active** "
                    "listings (asking-prices) instead of real sold "
                    "history, which is much less reliable. The Conf "
                    "column will rate active-source rows in the 25–50% "
                    "range so they stand out. Long-term fix: apply for "
                    "eBay's Marketplace Insights API (free, gated) — "
                    "uses the same OAuth as your existing Browse API "
                    "key. See https://developer.ebay.com/api-docs/buy/"
                    "marketplace-insights/static/overview.html"
                )
        except Exception:
            pass  # Stats reporting is best-effort

        st.rerun()

    # ================================================================
    # PIPELINE-DONE rendering: split layout with results + tabs.
    # Skipped while cooking — see PIPELINE-ACTIVE GATE comment above.
    # The audit/comps execution blocks above end with st.rerun(), so
    # this rendering only runs once the pipeline is fully complete.
    # ================================================================
    if pipeline_active:
        # Cooking — st.status panels above are doing the talking.
        # Skip the rest of the analysis-view rendering entirely.
        st.stop()

    # ----- Full-width results panel -----
    # Auction list lives in the Streamlit sidebar (left). The main content
    # area is now dedicated entirely to the results table — manual
    # audit/comps overrides are tucked into a collapsed expander below
    # since the auto-pipeline runs them without user input.
    st.markdown("### 📊 Results")

    # ---- Shipping info banner (7/10) ----
    # Shipping is NOT factored into grades or max bids anymore — every
    # flat assumption we tried mis-graded some category (the $25
    # default auto-F'd $6-to-ship sterling rings; $6 would understate
    # furniture). Instead: state the auction's shipping situation
    # once, up front, and let the user mentally net it out per lot.
    _ship_bits = []
    _src_series = leads_df.get('source')
    if _src_series is not None:
        _n_ship = int((_src_series == 'Ship').sum())
        _n_pickup = int((_src_series == 'Local Pickup').sum())
        if _n_ship and _n_pickup:
            _ship_bits.append(
                f"{_n_ship} shippable · {_n_pickup} pickup-only lots"
            )
        elif _n_pickup and not _n_ship:
            _ship_bits.append("all lots pickup-only")
        else:
            _ship_bits.append("all lots shippable")
    _hint_vals = pd.to_numeric(
        leads_df.get('auction_ship_hint', pd.Series(dtype=float)),
        errors='coerce',
    ).dropna().unique()
    if len(_hint_vals) == 1:
        _ship_bits.append(
            "auctioneer ships **FREE**" if _hint_vals[0] == 0
            else f"auctioneer rate ~**${_hint_vals[0]:.2f}/lot**"
        )
    else:
        _ship_bits.append(
            "rate unstated — typical: $5-10 small items, $15-30 boxed"
        )
    st.info(
        "🚚 **Shipping is NOT included in grades or max bids** — "
        + " · ".join(_ship_bits)
        + ". Net it out of the profit column before bidding."
    )

    # Surface the most recent comps-batch outcome so the user can see
    # exactly what happened (priced X/Y, filter dropped everything, an
    # exception, etc.) without scrolling through the collapsed st.status
    # panels above. This persists across reruns until a new auction loads.
    _comp_stats = st.session_state.get('_comps_stats') or {}
    if _comp_stats:
        if _comp_stats.get('attempted'):
            tail = ("  More pending — auto-continuing…"
                    if _comp_stats.get('has_more') else "")
            st.success(
                f"💰 Comps: priced **{_comp_stats['priced']}/"
                f"{_comp_stats['attempted']}** lot(s) across "
                f"**{_comp_stats['batches']}** batch(es).{tail}"
            )
        elif _comp_stats.get('last_msg'):
            st.warning(_comp_stats['last_msg'])

    ar = st.session_state.get('audit_results')
    if isinstance(ar, pd.DataFrame) and not ar.empty:
        try:
            _render_results_table(ar)
        except Exception as e:
            import traceback
            st.error(f"⚠️ Failed to render results table: {e}")
            with st.expander("Show traceback"):
                st.code(traceback.format_exc(), language="python")
            # Fall back to a plain table so the user can still see something
            st.dataframe(ar, width='stretch')
        if 'red_flag' in ar.columns and ar['red_flag'].fillna(False).any():
            try:
                _render_red_flag_editor(ar)
            except Exception as e:
                st.warning(f"Red-flag editor unavailable: {e}")
    else:
        st.info(
            "Audit hasn't produced any rows yet. Try the manual "
            "**🔄 Re-run audit** button in the controls below."
        )

    # ----- Manual overrides (collapsed) -----
    # Audit + comps run automatically when an auction loads. These tabs
    # are escape hatches: re-run after editing flags, change batch size,
    # tweak comp filters, run image enrichment, etc.
    st.markdown("---")
    _ctrls_expander = st.expander(
        "⚙️ Manual controls — re-run audit / comps / image enrichment",
        expanded=False,
    )
    with _ctrls_expander:
        tab_audit, tab_comps = st.tabs(["🛡️ Audit", "💰 Comps"])

    with tab_audit:
        # Step 1: AI audit
        st.markdown("### Step 1: AI Condition Audit")

        if has_audit:
            ar = st.session_state.audit_results
            # Bool-coerce — see comp-tab fix at line ~9689 for context.
            _rf = ar['red_flag'].fillna(False).astype(bool)
            good_count = int((~_rf).sum())
            flagged_count = int(_rf.sum())
            st.success(f"Audit complete — **{good_count} good+** condition, {flagged_count} red-flagged")

        audit_btn_label = "⏳ Running audit…" if audit_running else "🔄 Re-run audit"

        # ---- Audit knobs ----
        with st.expander("⚙️ Audit settings (optional)", expanded=False):
            _vp_opts = ["gemini", "claude"]
            _vp_labels = {
                "gemini": "Gemini flash-lite 💸",
                "claude": "Claude Haiku",
            }
            st.radio(
                "Audit AI provider",
                options=_vp_opts,
                format_func=lambda v: _vp_labels.get(v, v),
                key="vision_provider",
                horizontal=True,
                help="Which model runs the audit's text + photo tiers. "
                     "**Gemini** (gemini-flash-lite) is ~cheaper than Haiku "
                     "and matched its quality on real auction photos — the "
                     "default. **Claude** (Haiku) is the original path. "
                     "Gemini needs a key + credits in config.json under "
                     "`gemini.api_key` (aistudio.google.com). A dead key/"
                     "empty balance is caught by preflight before a run.",
            )
            st.slider(
                "Parallel workers",
                min_value=1, max_value=16, step=1,
                key="audit_workers",
                help="Concurrent API calls during the AI tier. "
                     "Default 8 — fast without tripping rate limits. "
                     "Drop to 1-2 if you see HTTP 429s; push to 12-16 on "
                     "a high-tier API plan. Gemini's free tier is rate-"
                     "limited — keep this at 2-4 on Gemini.",
            )

        if st.button(
            audit_btn_label,
            type="secondary",
            width='stretch',
            disabled=audit_running or comps_running,
            key="run_audit_btn",
            help="Manual override — the audit normally auto-fires when you "
                 "load an auction. Click this to re-run it.",
        ):
            # Manual click — clear the auto-pipeline tracker so the
            # forced re-run isn't blocked by the "already attempted" guard.
            st.session_state.pop('_auto_pipeline_attempts', None)
            st.session_state.audit_running = True
            st.rerun()

        # The audit-running execution block has been hoisted out of this
        # tab to the top-level analysis view (so the live st.status panel
        # is visible regardless of which tab the user is on). See the
        # auto-pipeline section above tabs.

        if has_audit and not (audit_running):
            st.caption("✅ Audit done — comps auto-fire next.")

    with tab_comps:
        # Step 2: eBay + Mercari comps (only after audit)
        st.markdown("### Step 2: eBay + Mercari Price Comps & STR")

        if not has_audit:
            st.info(
                "Audit hasn't completed yet — comps auto-fire as soon as it does. "
                "Hold on…"
            )
        else:
            ar = st.session_state.audit_results
            # Coerce red_flag to bool defensively — cache round-trips
            # can land it as int 0/1 (pyarrow / CSV roundtrip) or
            # numpy.int64. `~` on int gives bitwise inverse (-1, -2)
            # which pandas then tries to interpret as column labels,
            # crashing with KeyError("None of [Index([-1, -1, ...])]").
            red_mask = ar['red_flag'].fillna(False).astype(bool)
            good_df = ar[~red_mask]
            flagged_df = ar[red_mask]
            st.caption(f"💰 {len(good_df)} good+ items eligible for lookup ({len(flagged_df)} red-flagged skipped)")

            # ---- Pre-comps filter panel ----
            # Comps are ~3s/lot; a 1000-lot auction takes ~50 min unfiltered.
            # Let the user trim the target set before launch.
            with st.expander("⚙️ Narrow down what to comp (optional — big speedup on large auctions)",
                             expanded=(len(good_df) >= 300)):
                f_col1, f_col2 = st.columns(2)
                with f_col1:
                    st.number_input(
                        "Cap total lots to comp (0 = no cap)",
                        min_value=0, max_value=5000, step=50,
                        key="comps_max_lots",
                        help="Keep only the top-N by current bid. Useful for massive auctions.",
                    )
                with f_col2:
                    st.checkbox(
                        "Exclude HARD logistics lots",
                        key="comps_exclude_hard",
                        help="Skip items flagged as hard to ship/pick up.",
                    )
                    st.checkbox(
                        "🏷️ Retail-anchor pricing (Amazon-return lots)",
                        key="use_retail_anchor",
                        help="Liquidation lots state their retail price "
                             "('$241 …' titles / 'Retail Price: $252' "
                             "descriptions). When eBay comps miss, "
                             "est_resale falls back to 50% of retail "
                             "(free); when comps exceed retail, they're "
                             "capped to it. Anchored prices cap at B "
                             "grade and show 'retail-anchor' as source.",
                    )
                    st.checkbox(
                        "💾 Use spend ledger (never re-pay for a lot)",
                        key="use_spend_ledger",
                        help="Cross-session record of every lot that "
                             "ever consumed ScrapingBee credits. Lots "
                             "priced within the TTL restore for free "
                             "(marked 💾 in price source); lots that "
                             "came back empty are not re-attempted "
                             "until the TTL lapses (default 7 days).",
                    )
                    st.checkbox(
                        "Skip Unknown-verdict lots",
                        key="comps_skip_unknown_verdict",
                        help="verdict=Unknown means the audit couldn't "
                             "assess condition (API failure / no signal) "
                             "— NOT that the lot passed. Pricing these "
                             "spends credits on items that may be "
                             "untested or broken. Uncheck to comp them "
                             "anyway.",
                    )
                    st.checkbox(
                        "Easy-ship only (mailbox-size items)",
                        key="comps_easy_ship_only",
                        help="Aggressive logistics filter: keep ONLY items "
                             "that look mailbox-shippable — jewelry, watches, "
                             "cards, coins, electronics, small toys, books, "
                             "clothing, shoes, bags, etc. Drops anything "
                             "ambiguous (NEUTRAL logistics). Big lot-count "
                             "reduction; some legit small items may slip "
                             "through if the title is vague.",
                    )
                    st.checkbox(
                        "Run PC on stylized/replica lots (free, no credits)",
                        key="comps_pc_check_stylized",
                        value=True,
                        help="Stylized lots ('Gucci style hat', 'inspired by') "
                             "are skipped from eBay/Mercari comps because the "
                             "scraped sold-listings would return authentic-brand "
                             "prices that don't apply. PriceCharting matches "
                             "against curated product catalogs (no replica "
                             "contamination), so it's safe to run AND it's free. "
                             "Off = stylized lots get zero comp data; on = PC "
                             "still tries (and naturally returns nothing for "
                             "lots PC doesn't cover, like handbags/jewelry).",
                    )
                    # Mercari toggle removed — scraper has 0% hit rate
                    # across 7,700+ production lookups. Setting stays
                    # in session state defaults (False) for forward-compat
                    # with cached payloads.
                    st.checkbox(
                        "Fast STR (sample 3 lots per auction)",
                        key="comps_use_auction_str",
                        help="STR is a marketplace signal, not per-lot. Sampling "
                             "replaces ~1000 scrapes with ~15 on a typical run. "
                             "Big time saver — leave on.",
                    )
                    st.slider(
                        "Parallel workers",
                        min_value=1, max_value=16, step=1,
                        key="comps_workers",
                        help="Thread pool size for price lookups. Default 8 — "
                             "roughly 8x faster than serial. Drop to 1 if you "
                             "suspect rate-limiting; push to 12-16 on a fast "
                             "connection.",
                    )
                    st.slider(
                        "Comp batch size",
                        min_value=50, max_value=5000, step=50,
                        key="comps_chunk_size",
                        help="Process eligible lots N at a time. Default "
                             "5000 = whole auction in one batch (the results "
                             "table is hidden between batches anyway). Lower "
                             "this only if you specifically want to see "
                             "partial progress in chunks.",
                    )

                # Live preview of how many lots will actually be comped
                try:
                    preview_df, preview_skipped, preview_summary = _apply_comps_filters(good_df)
                    st.caption(
                        f"**🎯 Will comp {len(preview_df)} lots** "
                        f"(of {len(good_df)} good+) · {preview_summary}"
                    )
                except Exception:
                    pass

            # Decide which button label to show. The first chunk uses
            # "Run Price Comps"; subsequent chunks use "Continue with next
            # batch". `_comps_has_more` is set after each chunk completes.
            chunk_size = int(st.session_state.get('comps_chunk_size', 200))
            has_more = st.session_state.get('_comps_has_more', False)
            any_comped = (
                'est_resale' in ar.columns
                and ar['est_resale'].notna().any()
            )
            hard_btn_label = (
                "⏳ Running price comps…" if comps_running
                else "💥 Hard refresh"
            )
            if st.button(
                hard_btn_label,
                type="primary",
                width='stretch',
                disabled=audit_running or comps_running,
                key="hard_refresh_comps_btn",
                help="Wipe every existing comp value and clear the "
                     "BOLO match cache, then re-run comps fresh. Use "
                     "after pipeline changes (variance threshold, "
                     "fashion-jewelry detector, etc.) so stale "
                     "values don't bleed through.",
            ):
                ar_current = st.session_state.audit_results
                # Clear the BOLO match cache so pattern changes
                # (moissanite, white sapphire, etc.) propagate through
                # cached negatives.
                try:
                    _BOLO_MATCHER.clear_match_cache()
                except Exception:
                    pass
                tlog("HARD-REFRESH", "cleared BOLO match cache")

                if isinstance(ar_current, pd.DataFrame):
                    # Clear all comp data so the chunked runner sees
                    # every non-red-flagged row as 'pending' again.
                    for col in (*_COMP_COLUMNS, 'est_roi', 'max_bid'):
                        if col in ar_current.columns:
                            ar_current[col] = None
                    st.session_state.audit_results = ar_current
                    _save_current_auction_to_cache()
                # Reset chunk-pipeline tracker + sampled STR map so
                # the forced re-run resamples freshly.
                st.session_state.pop('_auto_pipeline_attempts', None)
                st.session_state.pop('_comps_has_more', None)
                st.session_state.pop('_comps_auction_str_map', None)
                # Manual re-run = explicit confirmation. Skip credit
                # re-confirm.
                st.session_state._comps_credit_confirmed = True
                st.session_state.comps_running = True
                st.toast(
                    "💥 Hard refresh — BOLO cache cleared, comps re-running",
                    icon="💥",
                )
                st.rerun()

            # The comps-running execution block has been hoisted out of
            # this tab to the top-level analysis view (so the live
            # st.status panel is visible regardless of which tab the
            # user is on). See the auto-pipeline section above tabs.

            if any_comped and not comps_running:
                st.caption("📊 Priced lots stream live to the right panel.")

# ---- DEFAULT VIEW: pick an auction from the sidebar ----
else:
    # Persistent banner for the most recent keyword-scan result.
    # A 0-match scan would otherwise bounce the user back to the
    # default view with no feedback, making the search feel like
    # it did nothing. This banner says "Yes the scan ran, here's
    # the term, here's the auction count — try again or pick a
    # different keyword".
    _last_kw_result = st.session_state.get('_keyword_scan_last_result')
    if _last_kw_result:
        _term = _last_kw_result.get('term', '')
        _matches = _last_kw_result.get('matches', 0)
        _auctions_q = _last_kw_result.get('auctions_queried', 0)
        _lots_scanned = _last_kw_result.get('lots_scanned', 0)
        if _matches == 0:
            st.warning(
                f"🔍 **No lots match '{_term}'** — scanned "
                f"**{_auctions_q}** auction(s), "
                f"HiBid returned {_lots_scanned:,} lots for "
                f"server-side pre-filtering but none survived the "
                f"local refinement. Try a broader keyword, a less "
                f"specific stem (e.g. 'rolex' instead of 'rolex box'), "
                f"or remove sidebar filters that might be hiding "
                f"matching auctions."
            )
        else:
            _match_auctions = _last_kw_result.get('match_auctions', 0)
            st.success(
                f"🔍 Last scan: **'{_term}'** → "
                f"**{_matches:,}** matches across "
                f"**{_match_auctions}** of **{_auctions_q}** auctions."
            )
        # Dismiss button so the banner doesn't linger forever
        if st.button("Dismiss", key="dismiss_kw_result"):
            st.session_state.pop('_keyword_scan_last_result', None)
            st.rerun()

    if st.session_state.get("auction_candidates"):
        st.info(
            "👆 Click a row in the auction grid above to analyze it. "
            "Audit + price comps run automatically once it loads."
        )
    elif discover_running:
        # First-page-open auto-discovery is in flight. The sticky banner
        # at the top already announces this, but a center-of-page card
        # gives the user something to look at while the sidebar loads —
        # otherwise they're staring at empty space.
        st.info(
            "⏳ **Loading auctions…**\n\n"
            "Fetching the open-auction list from HiBid. This typically "
            "takes 10–20 seconds on first open. The sidebar will "
            "populate automatically when the list arrives — no further "
            "action needed."
        )
    else:
        st.info(
            "No auctions discovered yet. Click 🔍 Discover in the top right "
            "to fetch the open-auction list. (We do this automatically on first page load.)"
        )
