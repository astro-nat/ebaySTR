"""Two-tier image enrichment.

Workflow per lot:
  1. **eBay tier** (free, fast — preferred):
     - Download the HiBid thumbnail (Referer header required by hibid CDN).
     - POST to eBay's Browse `search_by_image` endpoint.
     - If the top results share coherent product words (e.g. all contain
       "Nintendo Switch"), build a new title from the most common n-grams.
  2. **Claude vision fallback** (~$0.001/lot):
     - Triggers ONLY when the eBay tier didn't produce a confident match
       (no items returned, or build_enriched_title declined to match).
     - Asks Claude Haiku to identify brand/model/product from the image.
     - Output is structured JSON enforced by the API's output_config schema.

Why this design:
  - eBay is free and tied to the actual marketplace catalog — when it
    works, it produces titles that match real eBay listings exactly.
  - Claude vision generalizes better when eBay returns nothing (obscure
    items, single high-quality photos) — covers the long tail.
  - Gated: we only call either tier on lots the user has decided are
    worth analyzing (EASY logistics, not a red flag, bid above floor).
  - Cached per lot_id so re-scanning an auction is free.
"""
from __future__ import annotations

import base64
import json
import re
import statistics
from collections import Counter
from typing import Callable, Dict, List, Optional, Tuple

import httpx
import pandas as pd

from scraper._ssl_compat import make_ssl_context

# OS-native cert store bridge for Windows/AV-MITM environments.
_HTTPX_VERIFY = make_ssl_context()


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


# --- Claude vision tier ---
# Output schema enforced server-side; Claude's response is guaranteed to
# parse into this shape, so no defensive json handling needed.
_CLAUDE_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {
            "type": "string",
            "description": "eBay-searchable product title under 80 characters.",
        },
        "confident": {
            "type": "boolean",
            "description": (
                "true only when brand AND model/specific product type are "
                "identifiable. false for blurry, mixed-lot, or ambiguous photos."
            ),
        },
        "reason": {
            "type": "string",
            "description": "Brief justification under 80 characters.",
        },
    },
    "required": ["title", "confident", "reason"],
    "additionalProperties": False,
}

# Same shape as _CLAUDE_OUTPUT_SCHEMA but Gemini-flavored: no
# `additionalProperties` (Gemini rejects it) and an explicit
# `propertyOrdering` so the JSON comes back in a stable order.
_GEMINI_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {
            "type": "string",
            "description": "eBay-searchable product title under 80 characters.",
        },
        "confident": {
            "type": "boolean",
            "description": (
                "true only when brand AND model/specific product type are "
                "identifiable. false for blurry, mixed-lot, or ambiguous photos."
            ),
        },
        "reason": {
            "type": "string",
            "description": "Brief justification under 80 characters.",
        },
    },
    "required": ["title", "confident", "reason"],
    "propertyOrdering": ["title", "confident", "reason"],
}

_CLAUDE_SYSTEM_PROMPT = """You identify products from auction-listing photos so they can be searched on eBay's marketplace. Given an image, produce the most specific eBay-searchable title you can.

Title format:
- Lead with brand if visible/recognizable: "Sony", "Nintendo", "Pyrex", "Roseville", etc.
- Include model name/number if printed/visible
- Include year/era if printed or strongly indicated by style (e.g. "1960s", "MCM")
- End with the product type: "console", "skillet", "watch", "vase", "creamer", etc.
- Under 80 characters total
- No marketing fluff ("brand new!", "RARE!!"), no condition language ("mint", "broken")

Vintage / antique / decorative items (the hardest case — most lots fall here):
When no brand mark is visible, build a searchable title from VISUAL specifics. eBay's
search will match similar listings as long as the title carries enough identifying
features. Use any of these that apply:
- Material: "milk glass", "carnival glass", "pressed glass", "porcelain", "stoneware",
  "bone china", "brass", "copper", "pewter", "sterling silver", "walnut", "teak"
- Color/finish: "cobalt blue", "amber", "ruby", "iridescent", "satin", "matte"
- Pattern/motif: "hobnail", "fluted", "ribbed", "spatterware", "transferware",
  "floral", "rose pattern", "Greek key", "lattice"
- Era cues: "Victorian", "Art Deco", "Mid-Century Modern", "Mid-Century", "Atomic",
  "Depression-era", "Edwardian", "1950s", "1970s"
- Shape/form: "Cinderella bowl", "creamer", "salt cellar", "footed compote",
  "pedestal cake stand", "gravy boat", "candy dish", "trinket box"
- Maker/style families when shape is iconic: "Fenton-style hobnail",
  "Roseville-style pottery", "Stickley-style oak"
A vintage title without a brand IS confident if it carries 3+ specifics like
material + color + form + era. "Decorative item" or "Vintage piece" alone is NOT
confident.

Confidence rules:
- "confident": true → either (a) brand AND model/product type identifiable, OR
  (b) 3+ visual specifics that would match similar listings on eBay.
- "confident": false → blurry, mixed-lot, or so generic the title would only be
  "decorative item" / "vintage piece" / "old box" with nothing else to anchor it.

Bundled lots — a tray of mixed jewelry, a box of random tools, an estate-sale
grouping — should be marked "confident": false. They cannot be searched as a
single product.

Examples:
- Photo of a Nintendo Switch console in original box → {"title": "Nintendo Switch Console Gray Joy-Con HAC-001", "confident": true, "reason": "switch console clearly visible with model number"}
- Single 1922 silver dollar → {"title": "1922 Peace Silver Dollar", "confident": true, "reason": "date and denomination visible on coin"}
- Tray of mixed costume jewelry → {"title": "Costume jewelry lot", "confident": false, "reason": "multiple unrelated items, not a single product"}
- Vintage milk-glass vase with hobnail texture, no brand mark → {"title": "Milk glass hobnail vase mid-century 6 inch", "confident": true, "reason": "iconic hobnail pattern + milk glass material + form"}
- Mid-century walnut sideboard with tapered legs → {"title": "Mid-Century Modern walnut sideboard tapered legs", "confident": true, "reason": "MCM era + material + form + leg style identifiable"}
- Set of cobalt blue Depression-era glass plates → {"title": "Cobalt blue Depression glass plate set", "confident": true, "reason": "color + era + form clear"}
- Generic decorative brass figurine, no detail → {"title": "Decorative brass figurine", "confident": false, "reason": "no era/style/maker cues — too generic"}
- Blurry photo of an unidentifiable hardcover book → {"title": "Vintage hardcover book", "confident": false, "reason": "title and author not legible"}"""


class EbayImageEnricher:
    """Turn an auction photo into an eBay-searchable title.

    Two-tier flow:
      1. eBay `search_by_image` — free, fast, preferred when results agree.
      2. Claude Haiku vision (optional fallback) — runs only when eBay
         doesn't produce a confident match and an Anthropic key is wired
         up via the `anthropic_api_key` constructor arg.
    """

    def __init__(self, app_id: str, cert_id: str,
                 hibid_user_agent: str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                 anthropic_api_key: Optional[str] = None,
                 anthropic_model: str = "claude-haiku-4-5",
                 gemini_api_key: Optional[str] = None,
                 gemini_model: str = "gemini-flash-lite-latest",
                 vision_provider: str = "auto"):
        self.app_id = app_id
        self.cert_id = cert_id
        self.hibid_user_agent = hibid_user_agent
        self._token: Optional[str] = None
        self._client = httpx.Client(timeout=30.0, verify=_HTTPX_VERIFY)
        # Claude vision config — fallback tier. None disables the fallback.
        self.anthropic_api_key = anthropic_api_key or None
        self.anthropic_model = anthropic_model
        self._anthropic_client = None  # lazy
        # Gemini vision config — free-tier alternative to the Claude
        # fallback. `vision_provider` picks which engine the fallback uses:
        #   'claude'    → Claude only  (None if no Anthropic key)
        #   'gemini'    → Gemini only  (None if no Google key)
        #   'ebay_only' → skip the LLM fallback entirely
        #   'auto'      → free first: Gemini if keyed, else Claude
        self.gemini_api_key = gemini_api_key or None
        self.gemini_model = gemini_model
        self.vision_provider = (vision_provider or "auto").lower()

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

    # ----------------------------------------------------- claude vision tier
    @property
    def anthropic_client(self):
        """Lazy-init the Anthropic client. Returns None if no key configured."""
        if self._anthropic_client is None and self.anthropic_api_key:
            try:
                import anthropic
                import httpx
                # Explicit truststore-backed SSL context — the SDK's
                # default httpx client trusts only certifi, which dies
                # under Norton/corp TLS inspection. Same fix as pass2.
                self._anthropic_client = anthropic.Anthropic(
                    api_key=self.anthropic_api_key,
                    http_client=httpx.Client(verify=make_ssl_context()),
                )
            except ImportError:
                return None
        return self._anthropic_client

    @staticmethod
    def _detect_media_type(url: str) -> str:
        """Best-effort media-type guess from a URL extension."""
        url_low = (url or "").lower().split("?")[0]
        if url_low.endswith(".png"):
            return "image/png"
        if url_low.endswith(".gif"):
            return "image/gif"
        if url_low.endswith(".webp"):
            return "image/webp"
        # Default for HiBid CDN — most thumbnails are JPEG.
        return "image/jpeg"

    def _claude_identify(self, image_bytes: bytes,
                         thumbnail_url: str = "") -> Optional[Dict]:
        """Ask Claude vision to identify the product in a thumbnail.

        Takes already-downloaded image bytes (HiBid's CDN needs a Referer
        header that Anthropic's server-side fetcher won't send, so we
        upload the bytes via base64 instead of passing the URL). Returns
        a dict on success: {title, confidence (0.0-1.0), reason}, or
        None on any failure (no client, network error, parse error).
        """
        if not self.anthropic_client or not image_bytes:
            return None
        try:
            b64 = base64.standard_b64encode(image_bytes).decode("ascii")
            response = self.anthropic_client.messages.create(
                model=self.anthropic_model,
                max_tokens=300,
                system=_CLAUDE_SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": self._detect_media_type(thumbnail_url),
                                "data": b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "Identify this product for an eBay search. "
                                "Return JSON per the schema."
                            ),
                        },
                    ],
                }],
                output_config={
                    "format": {
                        "type": "json_schema",
                        "schema": _CLAUDE_OUTPUT_SCHEMA,
                    }
                },
            )
            text = next(
                (b.text for b in response.content if b.type == "text"), ""
            )
            data = json.loads(text)
            title = (data.get("title") or "").strip()
            if not title:
                return None
            # Trim to 80 chars on a word boundary.
            if len(title) > 80:
                title = title[:80].rsplit(" ", 1)[0]
            return {
                "title": title,
                # Map Claude's boolean confident → 0.85 (clears the 0.5
                # promotion threshold) or 0.3 (below it). The exact
                # number doesn't matter beyond passing/failing the bar.
                "confidence": 0.85 if data.get("confident") else 0.3,
                "reason": (data.get("reason") or "")[:120],
            }
        except Exception:
            return None

    # ----------------------------------------------------- gemini vision tier
    def _resolve_provider(self) -> Optional[str]:
        """Pick the LLM fallback engine per `vision_provider`.

        Explicit choices are strict: 'gemini' with no Google key returns
        None (no silent, cost-incurring Claude fallback), and vice-versa.
        'auto' prefers the free tier (Gemini) when a key is present.
        Returns 'gemini', 'claude', or None (skip the LLM fallback).
        """
        p = self.vision_provider
        if p == "ebay_only":
            return None
        if p == "gemini":
            return "gemini" if self.gemini_api_key else None
        if p == "claude":
            return "claude" if self.anthropic_api_key else None
        # auto
        if self.gemini_api_key:
            return "gemini"
        if self.anthropic_api_key:
            return "claude"
        return None

    def _gemini_identify(self, image_bytes: bytes,
                         thumbnail_url: str = "") -> Optional[Dict]:
        """Gemini twin of `_claude_identify` — same prompt, same output.

        Returns {title, confidence (0.0-1.0), reason} or None on any
        failure. Uses the shared, SSL-safe REST helper.
        """
        if not self.gemini_api_key or not image_bytes:
            return None
        from scraper.vision_provider import gemini_vision_json
        data = gemini_vision_json(
            api_key=self.gemini_api_key,
            model=self.gemini_model,
            system_prompt=_CLAUDE_SYSTEM_PROMPT,
            image_bytes=image_bytes,
            media_type=self._detect_media_type(thumbnail_url),
            user_text=(
                "Identify this product for an eBay search. "
                "Return JSON per the schema."
            ),
            response_schema=_GEMINI_OUTPUT_SCHEMA,
        )
        if not data:
            return None
        title = (data.get("title") or "").strip()
        if not title:
            return None
        if len(title) > 80:
            title = title[:80].rsplit(" ", 1)[0]
        return {
            "title": title,
            "confidence": 0.85 if data.get("confident") else 0.3,
            "reason": (data.get("reason") or "")[:120],
        }

    # ------------------------------------------------------------ public API
    def enrich_one(self, thumbnail_url: str,
                   original_title: str = "") -> Dict:
        """Try to enrich a single lot's title from its thumbnail image.

        Tier 1: eBay search_by_image. Tier 2: Claude vision (only if eBay
        didn't produce a confident match AND an Anthropic key is wired up).

        Returns a dict with:
          img_enriched_title : Optional[str]   — None when neither tier identified the lot
          img_confidence     : float           — 0.0 to 1.0
          img_comp_count     : int             — coherent eBay hits, or 99 sentinel for Claude
          img_top_match      : Optional[str]   — raw title of the best eBay hit
          img_top_price      : Optional[float] — price of the best eBay hit
          img_source         : Optional[str]   — 'ebay' or 'claude' (which tier matched)
          img_error          : Optional[str]   — error message, if any
        """
        result = {
            "img_enriched_title": None,
            "img_confidence": 0.0,
            "img_comp_count": 0,
            "img_top_match": None,
            "img_top_price": None,
            "img_source": None,
            "img_error": None,
        }
        if not thumbnail_url:
            result["img_error"] = "no_thumbnail"
            return result

        ebay_failed_reason: Optional[str] = None
        img: Optional[bytes] = None
        try:
            img = self._download_image(thumbnail_url)
            if not img:
                ebay_failed_reason = "image_fetch_failed"
            else:
                items = self._search_by_image(img, limit=8)
                if not items:
                    ebay_failed_reason = "no_ebay_matches"
                else:
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
                    if new_title is not None:
                        result["img_enriched_title"] = new_title
                        result["img_confidence"] = confidence
                        result["img_comp_count"] = coherent
                        result["img_source"] = "ebay"
                    else:
                        ebay_failed_reason = "ebay_low_confidence"
        except Exception as e:
            ebay_failed_reason = f"{type(e).__name__}: {e}"

        # --- LLM vision fallback (Gemini or Claude) ---
        # Triggers only when eBay didn't produce a confident match AND we
        # successfully downloaded the image. We pass the bytes (not the
        # URL) because neither provider's server-side fetcher can reach
        # HiBid's CDN — HiBid requires a Referer header.
        _provider = self._resolve_provider()
        if (
            result["img_enriched_title"] is None
            and _provider is not None
            and img is not None
        ):
            if _provider == "gemini":
                ident = self._gemini_identify(img, thumbnail_url=thumbnail_url)
            else:
                # Touch the property so a missing SDK still lazy-warns once.
                ident = (self._claude_identify(img, thumbnail_url=thumbnail_url)
                         if self.anthropic_client is not None else None)
            if ident is not None:
                result["img_enriched_title"] = ident["title"]
                result["img_confidence"] = ident["confidence"]
                # Sentinel: 99 hits flags this as an LLM match for the
                # promotion logic. Real eBay match counts top out around 8.
                result["img_comp_count"] = 99
                result["img_source"] = _provider
                # Surface the model's reason in img_top_match when eBay had
                # no top match at all — gives the user a debug breadcrumb.
                if not result["img_top_match"]:
                    result["img_top_match"] = f"[{_provider}] {ident['reason']}"
            elif ebay_failed_reason:
                result["img_error"] = f"{ebay_failed_reason}+{_provider}_failed"

        if result["img_enriched_title"] is None and not result["img_error"]:
            result["img_error"] = ebay_failed_reason or "no_match"
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
            ("img_source", None),
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

    Two acceptance paths:
      - eBay match: `img_source == 'ebay'` AND img_confidence ≥ min_confidence
        AND img_comp_count ≥ min_hits (need multiple coherent eBay hits).
      - LLM match: `img_source` in {'claude', 'gemini'} AND img_confidence ≥
        min_confidence (the model already self-reports confident=true/false,
        so the hit-count check doesn't apply).
    """
    df = df.copy()
    if "enriched_title" not in df.columns:
        df["enriched_title"] = df.get("title", "")
    if "img_enriched_title" not in df.columns:
        return df

    has_title = df["img_enriched_title"].notna()
    confidence = df["img_confidence"].fillna(0) >= min_confidence

    if "img_source" in df.columns:
        source = df["img_source"].fillna("")
        is_ebay = (
            (source == "ebay")
            & confidence
            & (df["img_comp_count"].fillna(0) >= min_hits)
        )
        is_llm = source.isin(["claude", "gemini"]) & confidence
        accept = is_ebay | is_llm
    else:
        # Older DataFrames pre-Claude tier — fall back to old check.
        accept = confidence & (df["img_comp_count"].fillna(0) >= min_hits)

    mask = has_title & accept
    df["enriched_title_pre_image"] = df["enriched_title"]
    df.loc[mask, "enriched_title"] = df.loc[mask, "img_enriched_title"]
    return df
