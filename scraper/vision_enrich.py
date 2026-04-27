"""Zero-cost image enrichment using eBay's Browse image_search API.

Workflow:
  1. Download the HiBid thumbnail (Referer header required by hibid CDN).
  2. POST to https://api.ebay.com/buy/browse/v1/item_summary/search_by_image
     with the image base64-encoded.
  3. Inspect the returned listings. If the top results share coherent
     product words (e.g. all contain "Nintendo Switch"), build a new
     enriched_title from the most common n-grams. Otherwise, leave the
     lot alone — a bad image match is worse than no match.

Why this design:
  - No paid API, no local model — just two HTTP calls per lot.
  - Gated: we only call this on lots the user has decided are worth
    analyzing (EASY logistics, not a red flag, bid above some floor).
  - Cached per lot_id so re-scanning an auction is free.
"""
from __future__ import annotations

import base64
import re
import statistics
from collections import Counter
from typing import Callable, Dict, List, Optional, Tuple

import httpx
import pandas as pd


# Words that add no signal to a product title — dropped when picking the
# "most common terms" across image_search hits.
_STOP = {
    'a', 'an', 'the', 'and', 'or', 'of', 'in', 'on', 'at', 'to', 'for',
    'with', 'from', 'by', 'as', 'is', 'are', 'was', 'were', 'be', 'this',
    'that', 'it', 'its', 'new', 'used', 'lot', 'item', 'items', 'size',
    'vintage', 'rare', 'authentic', 'original', 'men', 'women', 'mens',
    'womens', 'black', 'white', 'red', 'blue', 'green', 'gold', 'silver',
    'pink', 'brown', 'gray', 'grey', 'small', 'medium', 'large',
    'w', 'h', 'l', 'x', 'by', 'set', 'piece', 'pcs', 'pc',
}


class EbayImageEnricher:
    """Turn an auction photo into an eBay-searchable title via image_search."""

    def __init__(self, app_id: str, cert_id: str,
                 hibid_user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"):
        self.app_id = app_id
        self.cert_id = cert_id
        self.hibid_user_agent = hibid_user_agent
        self._token: Optional[str] = None
        self._client = httpx.Client(timeout=30.0)

    # ------------------------------------------------------------------ auth
    def _get_token(self) -> str:
        if self._token:
            return self._token
        creds = base64.b64encode(f"{self.app_id}:{self.cert_id}".encode()).decode()
        resp = self._client.post(
            "https://api.ebay.com/identity/v1/oauth2/token",
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Authorization": f"Basic {creds}",
            },
            data={
                "grant_type": "client_credentials",
                "scope": "https://api.ebay.com/oauth/api_scope",
            },
        )
        resp.raise_for_status()
        self._token = resp.json()["access_token"]
        return self._token

    # ----------------------------------------------------------- image fetch
    def _download_image(self, url: str) -> Optional[bytes]:
        """Pull a thumbnail from HiBid's CDN. Returns None on failure.

        HiBid's CDN requires a Referer header — without it, requests 4xx.
        """
        if not url:
            return None
        try:
            r = self._client.get(
                url,
                follow_redirects=True,
                headers={
                    "Referer": "https://hibid.com/",
                    "User-Agent": self.hibid_user_agent,
                },
            )
            if r.status_code == 200 and len(r.content) > 1000:
                return r.content
        except Exception:
            pass
        return None

    # --------------------------------------------------------- image search
    def _search_by_image(self, img_bytes: bytes, limit: int = 8) -> List[Dict]:
        """Call eBay's image_search endpoint. Returns list of item summaries."""
        token = self._get_token()
        b64 = base64.b64encode(img_bytes).decode()
        r = self._client.post(
            "https://api.ebay.com/buy/browse/v1/item_summary/search_by_image",
            headers={
                "Authorization": f"Bearer {token}",
                "X-EBAY-C-MARKETPLACE-ID": "EBAY_US",
                "Content-Type": "application/json",
            },
            params={"limit": str(limit)},
            json={"image": b64},
        )
        if r.status_code != 200:
            return []
        return r.json().get("itemSummaries", []) or []

    # -------------------------------------------------- confidence + title
    @staticmethod
    def _tokenize(title: str) -> List[str]:
        """Lowercase word tokens from a title, stopwords stripped."""
        words = re.findall(r"[A-Za-z0-9]+", title.lower())
        return [w for w in words if len(w) >= 2 and w not in _STOP]

    @classmethod
    def _build_enriched_title(cls, items: List[Dict],
                              original_title: str,
                              min_hits: int = 3) -> Tuple[Optional[str], float, int]:
        """From image_search hits, build a new title if the results are coherent.

        Returns (new_title, confidence 0.0-1.0, coherent_hit_count).
        `new_title` is None when confidence is too low to trust the match.

        Confidence heuristic:
          - Tokenize each hit's title, drop stop-words.
          - Count token occurrences across all hits.
          - If >=min_hits hits share the SAME top 2-4 tokens, that's a real
            product match; build a title from those shared tokens.
          - If tokens are scattered (no common terms), image_search likely
            returned unrelated items — return (None, low_confidence, 0).
        """
        if not items:
            return None, 0.0, 0

        # Per-hit token sets
        hit_tokens = [set(cls._tokenize(it.get("title", ""))) for it in items]
        hit_tokens = [s for s in hit_tokens if s]  # drop empty
        if not hit_tokens:
            return None, 0.0, 0

        # Count tokens across hits (DOCUMENT frequency — 1 per hit max)
        token_hits = Counter()
        for s in hit_tokens:
            for t in s:
                token_hits[t] += 1

        # Tokens that appear in at least half the hits are "shared"
        n = len(hit_tokens)
        threshold = max(min_hits, (n + 1) // 2)
        shared = [(t, c) for t, c in token_hits.most_common() if c >= threshold]

        # Keep the top 4 shared tokens in frequency order (drop purely numeric
        # unless they look model-number-ish — 3+ digits or alphanumerics)
        shared = [(t, c) for t, c in shared
                  if not (t.isdigit() and len(t) < 3)]
        top = shared[:5]

        if len(top) < 2:
            # Results were scattered — not a confident match
            return None, 0.1, 0

        # Compute confidence from coverage of the top tokens
        coverage = [c / n for _, c in top]
        confidence = round(statistics.mean(coverage), 2)

        # Coherent hit count: hits containing at least half of the top tokens
        half = max(1, len(top) // 2)
        top_set = {t for t, _ in top}
        coherent = sum(1 for s in hit_tokens if len(s & top_set) >= half)

        if coherent < min_hits:
            return None, confidence, coherent

        # Build the new title. Prefer the shortest matching item title as
        # the base (it's usually the cleanest), then enrich with any shared
        # tokens it doesn't already contain.
        candidates = sorted(
            (it.get("title", "") for it in items if it.get("title")),
            key=len,
        )
        base = ""
        for cand in candidates:
            cand_tokens = set(cls._tokenize(cand))
            if len(cand_tokens & top_set) >= half:
                base = cand
                break
        if not base:
            # Fallback: just join the top tokens titlecase
            base = " ".join(t.title() for t, _ in top)

        # Keep under ~80 chars for good eBay search behavior
        base = re.sub(r"\s+", " ", base).strip()
        if len(base) > 80:
            base = base[:80].rsplit(" ", 1)[0]

        return base, confidence, coherent

    # ------------------------------------------------------------ public API
    def enrich_one(self, thumbnail_url: str,
                   original_title: str = "") -> Dict:
        """Try to enrich a single lot's title from its thumbnail image.

        Returns a dict with:
          img_enriched_title : Optional[str]   — None when we can't identify
          img_confidence     : float           — 0.0 to 1.0
          img_comp_count     : int             — coherent hits found
          img_top_match      : Optional[str]   — raw title of the best hit
          img_top_price      : Optional[float] — price of the best hit (if any)
          img_error          : Optional[str]   — error message, if any
        """
        result = {
            "img_enriched_title": None,
            "img_confidence": 0.0,
            "img_comp_count": 0,
            "img_top_match": None,
            "img_top_price": None,
            "img_error": None,
        }
        if not thumbnail_url:
            result["img_error"] = "no_thumbnail"
            return result
        try:
            img = self._download_image(thumbnail_url)
            if not img:
                result["img_error"] = "image_fetch_failed"
                return result

            items = self._search_by_image(img, limit=8)
            if not items:
                result["img_error"] = "no_ebay_matches"
                return result

            top = items[0]
            result["img_top_match"] = top.get("title")
            try:
                result["img_top_price"] = float(
                    (top.get("price") or {}).get("value") or 0
                ) or None
            except (ValueError, TypeError):
                pass

            new_title, confidence, coherent = self._build_enriched_title(
                items, original_title=original_title
            )
            result["img_enriched_title"] = new_title
            result["img_confidence"] = confidence
            result["img_comp_count"] = coherent
        except Exception as e:
            result["img_error"] = f"{type(e).__name__}: {e}"
        return result

    def batch_enrich(
        self,
        df: pd.DataFrame,
        gate_fn: Optional[Callable[[pd.Series], bool]] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> pd.DataFrame:
        """Run image enrichment across a DataFrame.

        Args:
          df: must contain 'thumbnail_url' column. 'title' recommended.
          gate_fn: Optional callable(row) -> bool. If provided, only rows
            where gate_fn returns True are analyzed. Gated-out rows get
            img_error='skipped_gate' so the UI can explain the skip.
          progress_callback: (current, total, label) -> None

        Returns a copy of df with six new columns (img_enriched_title,
        img_confidence, img_comp_count, img_top_match, img_top_price,
        img_error). Does NOT overwrite 'enriched_title' — the caller
        decides whether to promote img_enriched_title when confidence is
        high enough.
        """
        df = df.copy()
        for col, default in [
            ("img_enriched_title", None),
            ("img_confidence", 0.0),
            ("img_comp_count", 0),
            ("img_top_match", None),
            ("img_top_price", None),
            ("img_error", None),
        ]:
            if col not in df.columns:
                df[col] = default

        total = len(df)
        for i, (idx, row) in enumerate(df.iterrows()):
            if gate_fn is not None and not gate_fn(row):
                df.at[idx, "img_error"] = "skipped_gate"
                if progress_callback:
                    progress_callback(i + 1, total, "gated")
                continue

            thumb = row.get("thumbnail_url") or ""
            orig = row.get("title") or ""
            out = self.enrich_one(thumb, original_title=orig)
            for k, v in out.items():
                df.at[idx, k] = v

            if progress_callback:
                title_preview = (out.get("img_enriched_title") or
                                 out.get("img_top_match") or
                                 row.get("title") or "")[:60]
                progress_callback(i + 1, total, title_preview)

        return df


def promote_image_titles(df: pd.DataFrame,
                         min_confidence: float = 0.5,
                         min_hits: int = 3) -> pd.DataFrame:
    """Where img_enriched_title is present and confident, promote it to
    `enriched_title`. The original `enriched_title` is kept in
    `enriched_title_pre_image` for traceability.
    """
    df = df.copy()
    if "enriched_title" not in df.columns:
        df["enriched_title"] = df.get("title", "")
    if "img_enriched_title" not in df.columns:
        return df

    mask = (
        df["img_enriched_title"].notna()
        & (df["img_confidence"].fillna(0) >= min_confidence)
        & (df["img_comp_count"].fillna(0) >= min_hits)
    )
    df["enriched_title_pre_image"] = df["enriched_title"]
    df.loc[mask, "enriched_title"] = df.loc[mask, "img_enriched_title"]
    return df
