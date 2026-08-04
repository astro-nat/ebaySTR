import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import pandas as pd

from .config_loader import load_config
from .pricecharting import classify_for_pricecharting
from .auth_check import detect_fashion_jewelry, detect_fake_pokemon

# Re-check pickup-only language at audit time in case the auction was loaded
# from cache (where logistics_ease was computed with an older, narrower regex).
# Keep this list in sync with Phase1Scraper._PICKUP_ONLY_RE.
_PICKUP_ONLY_AUDIT_RE = re.compile(
    r'local\s+pick\s*-?\s*up\s+only'
    r'|pick\s*-?\s*up\s+only'
    r'|local\s+pickup\s+only'
    r'|pickup\s+only'
    r'|in[- ]?(store|person)\s+pick\s*-?\s*up\s+only'
    r'|in[- ]?(store|person)\s+only'
    r'|on[- ]?site\s+pick\s*-?\s*up'
    r'|must\s+pick\s*-?\s*up'
    r'|must\s+be\s+picked\s+up'
    r'|buyer\s+(must\s+)?(pick\s*-?\s*up|arrange\s+pickup|arrange\s+shipp?ing|arrange\s+transport)'
    r'|pick\s*-?\s*up\s+(required|mandatory)'
    r'|no\s+shipp?ing'
    r'|will\s+not\s+ship'
    r'|do(es)?\s+not\s+ship'
    r'|cannot\s+be\s+shipped'
    r'|can\s*n?o?t\s+ship'
    r'|not\s+available\s+for\s+shipp?ing'
    r'|shipping\s+(is\s+)?not\s+available'
    r'|shipping\s*:\s*(not\s+available|none|no\b|unavailable)'
    r'|unable\s+to\s+ship'
    r'|no\s+ship\b'
    r'|local\s+(delivery|sale|buyers?)\s+only'
    r'|ships?\s+locally\s+only'
    r'|ships?\s+only\s+(locally|to\s+local)'
    r'|this\s+lot\s+(is|will\s+be)\s+(a\s+)?pick\s*-?\s*up'
    r'|available\s+for\s+pickup\s+only',
    re.IGNORECASE,
)

# Common filler words to skip when extracting details from descriptions
_FILLER = {
    'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'be',
    'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
    'would', 'could', 'should', 'may', 'might', 'shall', 'can', 'this',
    'that', 'these', 'those', 'it', 'its', 'of', 'in', 'on', 'at', 'to',
    'for', 'with', 'from', 'by', 'as', 'into', 'about', 'up', 'out',
    'lot', 'item', 'items', 'listing', 'auction', 'bid', 'bidding',
    'see', 'photos', 'photo', 'pictures', 'picture', 'image', 'images',
    'please', 'note', 'description', 'details', 'condition', 'shipping',
    'sold', 'buyer', 'seller', 'payment', 'terms', 'pickup',
    'click', 'here', 'more', 'info', 'information', 'view', 'all',
    'no', 'yes', 'not', 'we', 'our', 'you', 'your', 'if', 'so',
}

# HiBid lot descriptions are littered with section-header tokens that look
# like product features to the regex extractors. These leak into enriched
# titles ("Fog Machine ... Remote Condition Very Good Damaged No In Packaging")
# and pollute the eBay search query. Treat them as zero-signal tokens.
_HIBID_HEADER_TOKENS = {
    'condition', 'damaged', 'packaging', 'package', 'packed', 'boxed',
    'new', 'used', 'good', 'fair', 'poor', 'excellent', 'mint',
    'very', 'like', 'original', 'retail', 'msrp', 'value', 'estimated',
    'est', 'worth', 'tested', 'untested', 'working', 'nonworking',
    'broken', 'parts', 'only', 'sold', 'final', 'sale', 'available',
    'included', 'includes', 'missing', 'complete', 'incomplete',
    'pickup', 'shipping', 'shipped', 'ship', 'delivery', 'local',
    'returns', 'warranty', 'guarantee', 'lot', 'unit', 'units',
}


# --- Verdict labels ---
# Kept in sync with the historical zero-shot classifier output so cached
# audit rows from older runs still merge cleanly.
_RISK_LABELS = [
    "broken, damaged, or for parts",
    "untested or unknown condition",
    "mint condition or working perfectly",
    "normal wear and tear",
]
_RED_FLAG_LABELS = {
    "broken, damaged, or for parts",
    "untested or unknown condition",
}


# --- Tier 1: keyword classification ---
# Rule-based first pass. Hits short-circuit the API call entirely. Patterns
# are intentionally narrow — we only match unambiguous condition language
# ("untested", "doesn't work") rather than soft signals ("as-is", "no
# returns") that show up in auction-house boilerplate regardless of item
# condition. Anything ambiguous falls through to the API for context-aware
# classification.
#
# Each tuple: (compiled regex, verdict, red_flag).
_KEYWORD_PATTERNS = [
    # --- Broken / damaged / for-parts ---
    (re.compile(r"\b(?:doesn'?t|does\s+not)\s+work\b", re.IGNORECASE),
     "broken, damaged, or for parts", True),
    (re.compile(r"\bnon[\s-]*work(?:ing|s)?\b", re.IGNORECASE),
     "broken, damaged, or for parts", True),
    (re.compile(r"\bnot\s+working\b", re.IGNORECASE),
     "broken, damaged, or for parts", True),
    (re.compile(r"\bfor\s+parts(?:\s+(?:only|or\s+repair))?\b", re.IGNORECASE),
     "broken, damaged, or for parts", True),
    (re.compile(r"\bparts\s+only\b", re.IGNORECASE),
     "broken, damaged, or for parts", True),
    (re.compile(r"\b(?:is|was|are|were)\s+broken\b", re.IGNORECASE),
     "broken, damaged, or for parts", True),
    (re.compile(r"\bshattered\b", re.IGNORECASE),
     "broken, damaged, or for parts", True),
    (re.compile(r"\bwon'?t\s+(?:turn\s+on|power|start|charge|work)\b", re.IGNORECASE),
     "broken, damaged, or for parts", True),
    (re.compile(r"\bwill\s+not\s+(?:turn\s+on|power|start|work)\b", re.IGNORECASE),
     "broken, damaged, or for parts", True),

    # --- Untested / unknown condition ---
    (re.compile(r"\buntested\b", re.IGNORECASE),
     "untested or unknown condition", True),
    (re.compile(r"\bnot\s+tested\b", re.IGNORECASE),
     "untested or unknown condition", True),
    (re.compile(r"\b(?:unknown|unspecified)\s+condition\b", re.IGNORECASE),
     "untested or unknown condition", True),
    (re.compile(r"\bcondition\s+(?:unknown|unspecified)\b", re.IGNORECASE),
     "untested or unknown condition", True),
    (re.compile(r"\b(?:could\s+not|cannot|can'?t|unable\s+to)\s+test\b", re.IGNORECASE),
     "untested or unknown condition", True),
    (re.compile(r"\bnot\s+able\s+to\s+test\b", re.IGNORECASE),
     "untested or unknown condition", True),

    # --- Positive signals (NEW IN BOX, sealed, tested-working) ---
    (re.compile(r"\b(?:brand\s+new|new\s+in\s+(?:box|package|wrapper))\b", re.IGNORECASE),
     "mint condition or working perfectly", False),
    (re.compile(r"\bfactory\s+sealed\b", re.IGNORECASE),
     "mint condition or working perfectly", False),
    (re.compile(r"\bstill\s+sealed\b", re.IGNORECASE),
     "mint condition or working perfectly", False),
    (re.compile(r"\b(?:NIB|NWT|NWB)\b"),  # acronyms — case-sensitive on purpose
     "mint condition or working perfectly", False),
    (re.compile(r"\bunopened\b", re.IGNORECASE),
     "mint condition or working perfectly", False),
    (re.compile(r"\btested\s+(?:and\s+)?(?:working|functional)\b", re.IGNORECASE),
     "mint condition or working perfectly", False),
    (re.compile(r"\bfully\s+functional\b", re.IGNORECASE),
     "mint condition or working perfectly", False),
    (re.compile(r"\bworks?\s+(?:perfectly|great|like\s+new|as\s+intended)\b", re.IGNORECASE),
     "mint condition or working perfectly", False),
]


def _classify_by_keyword(desc: str):
    """Return a classification dict if a keyword pattern matches, else None.

    Keyword hits are 95%-confidence — we only match unambiguous phrases.
    """
    if not desc:
        return None
    for pattern, verdict, red_flag in _KEYWORD_PATTERNS:
        if pattern.search(desc):
            return {
                "verdict": verdict,
                "confidence": 95.0,
                "red_flag": red_flag,
                "source": "keyword",
            }
    return None


# --- Tier 2 / 3: Claude API ---
# JSON Schema enforced server-side via output_config — Claude is guaranteed
# to return text that parses into this shape, no defensive parsing needed.
_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": _RISK_LABELS},
        "confidence": {
            "type": "number",
            "description": "Confidence 0-100 in the verdict.",
        },
        "reason": {
            "type": "string",
            "description": "Short justification under 80 characters.",
        },
    },
    "required": ["verdict", "confidence", "reason"],
    "additionalProperties": False,
}

# Gemini-flavored twin of _OUTPUT_SCHEMA: no `additionalProperties`
# (Gemini rejects it) and an explicit `propertyOrdering`. Used only when
# the photo tier is routed to Gemini via `vision_provider`.
_GEMINI_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": _RISK_LABELS},
        "confidence": {
            "type": "number",
            "description": "Confidence 0-100 in the verdict.",
        },
        "reason": {
            "type": "string",
            "description": "Short justification under 80 characters.",
        },
    },
    "required": ["verdict", "confidence", "reason"],
    "propertyOrdering": ["verdict", "confidence", "reason"],
}


# --- Pokémon card authenticity (vision tier) ---
# Bootleg cards carry honest titles, classify as collectibles, skip the
# audit, and would comp against real-card prices. This vision pass reads
# the photo for fake tells and red-flags confident fakes so they never
# comp. Deliberately conservative — a too-small/blurry photo returns
# authentic=true (never flags a fake on uncertainty), so legit cards
# aren't punished.
_CARD_AUTH_SYSTEM_PROMPT = """You authenticate Pokémon trading cards from a single auction photo. Decide whether the card(s) shown are AUTHENTIC official Pokémon TCG cards or FAKE (proxy / bootleg / counterfeit / custom / gold-metal novelty).

Fake tells: wrong or blurry card back, off-center or crooked printing, pixelated / low-res text or art, non-standard holo or foil pattern, wrong fonts, missing or incorrect copyright line, gold-metal or plastic novelty cards, 'GX' / 'V' / 'VMAX' branding on cards from eras that never had it, energy-symbol errors, or a bulk lot of obviously reprinted commons.

Rules:
- Only call a card FAKE when you can SEE a concrete tell on an actual card.
- If the photo shows Pokémon MERCHANDISE that is not a trading card (plush, blanket, toy, figure, mug, clothing, sealed video game), it is NOT fake — return authentic=true. 'Fake' applies ONLY to counterfeit trading cards.
- If the photo is too small, blurry, or angled to judge, return authentic=true with LOW confidence — never flag a fake on uncertainty.
- Graded slabs (PSA / BGS / CGC) are authentic.

Return JSON: authentic (boolean = false ONLY for a counterfeit CARD), confidence (0-100), reason (<=90 chars — name the tell, or why it looks genuine / not-a-card)."""

_CARD_AUTH_OUTPUT_SCHEMA = {  # Claude flavor
    "type": "object",
    "properties": {
        "authentic": {"type": "boolean",
                      "description": "true if a genuine official Pokémon card."},
        "confidence": {"type": "number", "description": "0-100."},
        "reason": {"type": "string", "description": "Under 90 chars."},
    },
    "required": ["authentic", "confidence", "reason"],
    "additionalProperties": False,
}
_CARD_AUTH_GEMINI_SCHEMA = {  # Gemini flavor (no additionalProperties)
    "type": "object",
    "properties": {
        "authentic": {"type": "boolean",
                      "description": "true if a genuine official Pokémon card."},
        "confidence": {"type": "number", "description": "0-100."},
        "reason": {"type": "string", "description": "Under 90 chars."},
    },
    "required": ["authentic", "confidence", "reason"],
    "propertyOrdering": ["authentic", "confidence", "reason"],
}

# Routes a collectible lot to the card-authenticity vision check: needs
# BOTH a Pokémon reference AND card context, so Pokémon plush / blankets /
# toys (which classify_for_pricecharting over-claims as 'tcg') don't get
# sent to the card check and mislabeled.
_POKEMON_LOT_RE = re.compile(r"\b(?:pok[eé]mon|pikachu|charizard)\b", re.IGNORECASE)
_CARD_CONTEXT_RE = re.compile(
    r"\b(?:cards?|holo(?:graphic)?|tcg|psa|bgs|cgc|graded|booster|promo|"
    r"blister|1st\s*edition|first\s*edition|reverse\s*holo|base\s*set|"
    r"full\s*art|secret\s*rare|ex|gx|vmax|vstar|\bv\b)\b",
    re.IGNORECASE)


_SYSTEM_PROMPT = """You evaluate auction lot descriptions and photos and classify each item's condition. You will be given either a text description or an image of a single auction lot, and must return a JSON object with the verdict, a confidence score, and a brief reason.

Choose ONE verdict from this exact list:
- "broken, damaged, or for parts" — the item is explicitly broken, non-functional, sold for parts, or damaged in a way that prevents normal use.
- "untested or unknown condition" — the seller has not tested the item or cannot verify it works. Use this for items where the seller explicitly disclaims condition knowledge.
- "mint condition or working perfectly" — the item is described as new, sealed, unopened, or explicitly tested-and-working.
- "normal wear and tear" — used but implied to function normally. This is the DEFAULT for typical resale items where the description doesn't say anything alarming.

CRITICAL RULES:
- Auction-house boilerplate ("all items sold as-is", "no returns", "all sales final", "buyer beware") is generic legal language used regardless of item condition — do NOT classify based on it. Only classify based on item-specific condition language.
- "Used" or "previously owned" alone → "normal wear and tear", NOT untested.
- "Lot of N items" with no individual condition info and items appear intact → "normal wear and tear".
- For images: visible cracks, missing pieces, or obvious damage → "broken, damaged, or for parts". A normal-looking item in a generic photo → "normal wear and tear".
- For collectibles where condition is encoded by grade (PSA/BGS-graded cards, sealed video games, mint-stamped coins) → "mint condition or working perfectly" if the grade indicates good condition.
- Be conservative — when in doubt, prefer "normal wear and tear" over red-flagging. False red flags cost the user good resale opportunities."""


class Phase2Scraper:
    """Audits HiBid lot descriptions for condition risk.

    Three-tier classification, in order of cost:

      1. **Keyword regex** over the description — instant, free. Only
         matches unambiguous phrases ("untested", "doesn't work",
         "factory sealed"). Most resale-worthy lots short-circuit here.
      2. **Claude Haiku text classification** — when the description has
         enough material (≥ ~80 chars after HTML strip) for the model to
         extract signal.
      3. **Claude Haiku vision classification** — when the description
         is too short, falls back to analyzing the lot's thumbnail. The
         URL is sent directly; Anthropic fetches it server-side.

    Items pre-flagged by Phase 1 (HARD logistics, collectibles, totally
    empty rows) skip all three tiers — Phase 1 already has higher-
    confidence signal for those buckets.
    """

    # Back-compat aliases — both old constants now point to Haiku 4.5.
    # The local-model fast/accurate split was retired when we moved off
    # transformers/torch onto the Anthropic API. Old session state that
    # references DEFAULT_MODEL_FAST keeps working without code changes.
    DEFAULT_MODEL = "claude-haiku-4-5"
    DEFAULT_MODEL_FAST = "claude-haiku-4-5"
    DEFAULT_MODEL_ACCURATE = "claude-haiku-4-5"

    # Below this many chars of cleaned description, we skip the text API
    # and go straight to image analysis. Picked empirically — anything
    # shorter than ~80 chars is usually just the title repeated and gives
    # the text classifier nothing to chew on.
    _MIN_DESC_FOR_TEXT_API = 80

    def __init__(self, model_name: str = None, vision_provider: str = None):
        cfg = load_config()
        anth = cfg.get("anthropic", {}) or {}
        self.api_key = (
            anth.get("api_key")
            or os.environ.get("ANTHROPIC_API_KEY")
            or None
        )
        self.model_name = (
            model_name or anth.get("model") or self.DEFAULT_MODEL
        )
        self._client = None  # lazy-init on first API call
        # Gemini config for the PHOTO tier only (the text tier always
        # stays on Claude — it's cheap and tuned). `vision_provider`:
        #   'claude'    → Claude photo classify (None if no Anthropic key)
        #   'gemini'    → Gemini photo classify (None if no Google key)
        #   'auto'/None → free first: Gemini if keyed, else Claude
        gem = cfg.get("gemini", {}) or {}
        self.gemini_api_key = (
            gem.get("api_key")
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
            or None
        )
        self.gemini_model = gem.get("model") or "gemini-flash-lite-latest"
        self.vision_provider = (
            vision_provider or gem.get("provider") or "auto"
        ).lower()

    @property
    def client(self):
        """Lazy-init the Anthropic client. Returns None if no key configured."""
        if self._client is None and self.api_key:
            try:
                import anthropic
                import httpx
                from scraper._ssl_compat import make_ssl_context
                # Explicit truststore-backed SSL context. The SDK's
                # default httpx client trusts only certifi, which
                # Norton/corp-MITM re-signed certs are not in — every
                # API call died with APIConnectionError on this box
                # (7/6 Hayworth run: 160/160 image_api_failed).
                # `truststore.inject_into_ssl()` at app startup is NOT
                # sufficient for httpx (see scraper/_ssl_compat.py
                # docstring) — pass the context explicitly, same as
                # pass1 does for HiBid calls.
                self._client = anthropic.Anthropic(
                    api_key=self.api_key,
                    http_client=httpx.Client(verify=make_ssl_context()),
                )
            except ImportError:
                # A missing SDK used to fail SILENTLY here — every lot
                # got labeled `no_api_key` even though the key was fine,
                # and the 7/6 Hayworth Creek auction ran a full keyless
                # audit (150 Unknown verdicts, comps spent on unvetted
                # lots) before anyone noticed. Shout to the terminal
                # once per process so the root cause is visible.
                import sys as _sys
                if not getattr(Phase2Scraper, '_sdk_warned', False):
                    Phase2Scraper._sdk_warned = True
                    print(
                        "[AUDIT] FATAL: `anthropic` package is not "
                        "installed in this venv — API key is configured "
                        "but unusable. Every lot will be marked "
                        "audit_source=no_api_key. Fix with: "
                        "pip install anthropic",
                        file=_sys.stderr, flush=True,
                    )
                return None
        return self._client

    def _enrich_title(self, original_title: str, description: str) -> str:
        """Build a detailed, eBay-searchable title from auction title + description.

        Extracts brand names, model numbers, product specifics, and key attributes
        from the description and combines them with the original title.
        Returns the enriched title (max ~80 chars for good eBay search results).
        """
        if not description or len(description.strip()) < 10:
            return original_title

        # Strip HTML tags and normalize whitespace
        clean = re.sub(r'<[^<]+?>', ' ', description)
        clean = re.sub(r'&\w+;', ' ', clean)
        clean = re.sub(r'\s+', ' ', clean).strip()

        # Words already in the original title (lowercase)
        title_words = set(re.findall(r'[a-z0-9]+', original_title.lower()))

        new_details = []
        seen = set()

        def _add(term):
            t = term.strip()
            if not t or len(t) < 2:
                return
            low = t.lower()
            if low in seen or low in _FILLER:
                return
            if low in _HIBID_HEADER_TOKENS:
                return
            words = set(re.findall(r'[a-z0-9]+', low))
            if not words:
                return
            informative = words - title_words - _FILLER - _HIBID_HEADER_TOKENS
            if not informative:
                return
            seen.add(low)
            new_details.append(t)

        # 1. Brand / model numbers (e.g. "XR-500", "Model 42B", "HP LaserJet")
        for m in re.finditer(r'\b([A-Z][A-Za-z]*[\s-]?[A-Z0-9][\w-]*(?:[\s-][A-Z0-9][\w-]*)*)\b', clean):
            _add(m.group(1))

        # 2. Model / part numbers: alphanumeric with hyphens or dots
        for m in re.finditer(r'\b([A-Z]{1,4}[\-.]?\d{2,}[\w\-.]*)\b', clean):
            _add(m.group(1))

        # 3. Year mentions
        for m in re.finditer(r'\b(1[89]\d{2}|20[0-2]\d)\b', clean):
            _add(m.group(1))

        # 4. Quoted product names
        for m in re.finditer(r'["“]([^"”]{3,40})["”]', clean):
            _add(m.group(1))

        # 5. Capitalized multi-word phrases in the first 300 chars
        first_chunk = clean[:300]
        for m in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', first_chunk):
            phrase = m.group(1)
            if len(phrase.split()) <= 4:
                _add(phrase)

        # 6. First meaningful sentence as fallback context
        sentences = re.split(r'[.!?\n]', first_chunk)
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 15 and not any(skip in sent.lower() for skip in
                                           ['shipping', 'payment', 'pickup', 'bid', 'click', 'terms']):
                for word in sent.split():
                    w_clean = re.sub(r'[^a-zA-Z0-9\-]', '', word)
                    if (len(w_clean) > 3
                            and w_clean.lower() not in _FILLER
                            and w_clean[0].isupper()):
                        _add(w_clean)
                break

        enriched = original_title.rstrip('.')
        for detail in new_details:
            candidate = f"{enriched} {detail}"
            if len(candidate) > 80:
                break
            enriched = candidate

        return enriched

    def _classify_text_gemini(self, snippet: str):
        """Gemini text-condition classify. Same output shape as the Claude
        text tier; returns None on any failure."""
        from scraper.vision_provider import gemini_text_json
        data = gemini_text_json(
            api_key=self.gemini_api_key,
            model=self.gemini_model,
            system_prompt=_SYSTEM_PROMPT,
            user_text=(
                "Classify the condition of this auction lot.\n\n"
                f"DESCRIPTION:\n{snippet}"
            ),
            response_schema=_GEMINI_OUTPUT_SCHEMA,
        )
        if not data or "verdict" not in data:
            return None
        verdict = data["verdict"]
        try:
            conf = float(data.get("confidence", 70))
        except (TypeError, ValueError):
            conf = 70.0
        return {
            "verdict": verdict,
            "confidence": conf,
            "red_flag": verdict in _RED_FLAG_LABELS,
            "source": "text_api",
            "reason": data.get("reason", ""),
        }

    def _classify_by_text_api(self, clean_desc: str):
        """Tier 2: send the cleaned description to the text model.

        Routes to Gemini or Claude per `vision_provider` (the selector
        governs the whole audit AI, not just photos, so 'Gemini' means
        zero Claude spend). Returns the parsed result dict on success, or
        None if the call failed (network error, rate limit, parse error,
        no client). Caller falls through to the image tier or marks as
        Unknown on None.
        """
        # Trim to a generous limit — input cost is dwarfed by per-call
        # latency, so giving the model more context is cheap.
        snippet = clean_desc[:2500]
        if self._image_provider() == "gemini":
            return self._classify_text_gemini(snippet)
        if not self.client:
            return None
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=200,
                system=_SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": (
                        "Classify the condition of this auction lot.\n\n"
                        f"DESCRIPTION:\n{snippet}"
                    ),
                }],
                output_config={
                    "format": {
                        "type": "json_schema",
                        "schema": _OUTPUT_SCHEMA,
                    }
                },
            )
            text = next(
                (b.text for b in response.content if b.type == "text"), ""
            )
            data = json.loads(text)
            verdict = data["verdict"]
            return {
                "verdict": verdict,
                "confidence": float(data.get("confidence", 70)),
                "red_flag": verdict in _RED_FLAG_LABELS,
                "source": "text_api",
                "reason": data.get("reason", ""),
            }
        except Exception:
            return None

    def _download_thumbnail(self, url: str):
        """Download a HiBid CDN image → (bytes, media_type) or (None, None).

        HiBid's CDN requires a Referer header — that's exactly why the
        old URL-source approach died: Anthropic's server-side fetcher
        sends no Referer, HiBid blocks it, and the API returns 400
        "Unable to download the file" for EVERY image call (7/6
        Hayworth: 160/160 image_api_failed). Same download pattern as
        vision_enrich._download_image, plus the truststore SSL context
        for Norton/corp TLS inspection.
        """
        if not url:
            return None, None
        try:
            import httpx
            from scraper._ssl_compat import make_ssl_context
            if getattr(self, '_img_client', None) is None:
                # Benign race under the audit thread pool — double
                # creation just wastes one client object.
                self._img_client = httpx.Client(
                    verify=make_ssl_context(), timeout=20,
                    follow_redirects=True,
                )
            r = self._img_client.get(url, headers={
                "Referer": "https://hibid.com/",
                "User-Agent": "Mozilla/5.0",
            })
            if r.status_code == 200 and len(r.content) > 1000:
                mt = (r.headers.get('content-type') or 'image/jpeg')
                mt = mt.split(';')[0].strip()
                if not mt.startswith('image/'):
                    mt = 'image/jpeg'
                return r.content, mt
        except Exception:
            pass
        return None, None

    def _image_provider(self) -> Optional[str]:
        """Which engine runs the audit AI, per `vision_provider`.

        Honors the user's CHOICE authoritatively — an explicit 'gemini'
        returns 'gemini' even if the key is missing, so the tiers no-op to
        None (keyword-only) rather than silently cross-falling-back to the
        paid engine and spending credits the user meant to avoid. 'auto'
        picks whichever engine has a key, free tier first. Returns
        'gemini', 'claude', or None (no engine available at all).
        """
        p = self.vision_provider
        if p == "gemini":
            return "gemini"
        if p == "claude":
            return "claude"
        # auto: free first, then paid, else nothing
        if self.gemini_api_key:
            return "gemini"
        if self.api_key:
            return "claude"
        return None

    def _classify_image_gemini(self, img_bytes: bytes, media_type: str):
        """Gemini photo-condition classify. Same output shape as the
        Claude image tier; returns None on any failure."""
        from scraper.vision_provider import gemini_vision_json
        data = gemini_vision_json(
            api_key=self.gemini_api_key,
            model=self.gemini_model,
            system_prompt=_SYSTEM_PROMPT,
            image_bytes=img_bytes,
            media_type=media_type,
            user_text=(
                "Classify the condition of this auction lot based on the "
                "photo. The description was too short to use, so this is "
                "your only signal."
            ),
            response_schema=_GEMINI_OUTPUT_SCHEMA,
        )
        if not data or "verdict" not in data:
            return None
        verdict = data["verdict"]
        try:
            conf = float(data.get("confidence", 60))
        except (TypeError, ValueError):
            conf = 60.0
        return {
            "verdict": verdict,
            "confidence": conf,
            "red_flag": verdict in _RED_FLAG_LABELS,
            "source": "image_api",
            "reason": data.get("reason", ""),
        }

    def _classify_by_image_api(self, thumbnail_url: str):
        """Tier 3: download the thumbnail, send base64 to the vision model.

        Routes to Gemini or Claude per `vision_provider`. Returns None on
        failure so the caller can mark the lot as Unknown. A failed
        download RAISES so the batch loop's first-error logger surfaces it
        (a silent None here hid the HiBid-Referer failure for an entire
        production run).
        """
        _provider = self._image_provider()
        if not _provider or not thumbnail_url:
            return None
        img_bytes, media_type = self._download_thumbnail(thumbnail_url)
        if img_bytes is None:
            raise RuntimeError(
                f"thumbnail download failed (HiBid CDN): {thumbnail_url[:90]}"
            )
        if _provider == "gemini":
            return self._classify_image_gemini(img_bytes, media_type)
        # --- Claude path (default) ---
        if not self.client:
            return None
        import base64 as _b64
        b64_data = _b64.standard_b64encode(img_bytes).decode("ascii")
        try:
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=200,
                system=_SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": b64_data,
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "Classify the condition of this auction lot "
                                "based on the photo. The description was too "
                                "short to use, so this is your only signal."
                            ),
                        },
                    ],
                }],
                output_config={
                    "format": {
                        "type": "json_schema",
                        "schema": _OUTPUT_SCHEMA,
                    }
                },
            )
            text = next(
                (b.text for b in response.content if b.type == "text"), ""
            )
            data = json.loads(text)
            verdict = data["verdict"]
            return {
                "verdict": verdict,
                "confidence": float(data.get("confidence", 60)),
                "red_flag": verdict in _RED_FLAG_LABELS,
                "source": "image_api",
                "reason": data.get("reason", ""),
            }
        except Exception:
            return None

    def _assess_card_authenticity(self, thumbnail_url: str):
        """Vision check: is this Pokémon card authentic or a fake/bootleg?

        Routes to the same provider as the audit (Gemini or Claude).
        Returns {authentic: bool, confidence: float, reason: str} or None
        on any failure (no provider, download fail, parse error). Callers
        treat None as 'couldn't check — leave the lot alone'.
        """
        provider = self._image_provider()
        if not provider or not thumbnail_url:
            return None
        img_bytes, media_type = self._download_thumbnail(thumbnail_url)
        if img_bytes is None:
            return None
        _user = ("Is this an authentic Pokémon card or a fake / bootleg? "
                 "Return JSON per the schema.")
        data = None
        try:
            if provider == "gemini":
                from scraper.vision_provider import gemini_vision_json
                data = gemini_vision_json(
                    api_key=self.gemini_api_key, model=self.gemini_model,
                    system_prompt=_CARD_AUTH_SYSTEM_PROMPT,
                    image_bytes=img_bytes, media_type=media_type,
                    user_text=_user, response_schema=_CARD_AUTH_GEMINI_SCHEMA)
            elif self.client is not None:
                import base64 as _b64
                b64 = _b64.standard_b64encode(img_bytes).decode("ascii")
                resp = self.client.messages.create(
                    model=self.model_name, max_tokens=200,
                    system=_CARD_AUTH_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": [
                        {"type": "image", "source": {
                            "type": "base64", "media_type": media_type,
                            "data": b64}},
                        {"type": "text", "text": _user},
                    ]}],
                    output_config={"format": {
                        "type": "json_schema",
                        "schema": _CARD_AUTH_OUTPUT_SCHEMA}})
                text = next(
                    (b.text for b in resp.content if b.type == "text"), "")
                data = json.loads(text) if text else None
        except Exception:
            return None
        if not data or "authentic" not in data:
            return None
        try:
            conf = float(data.get("confidence", 60) or 60)
        except (TypeError, ValueError):
            conf = 60.0
        return {
            "authentic": bool(data.get("authentic")),
            "confidence": conf,
            "reason": str(data.get("reason") or "")[:90],
        }

    def analyze_condition(self, description_text: str) -> dict:
        """Single-description classification — keyword pass, then text API.

        Kept for backward compat with anything that imports this method
        directly. Most callers should use batch_audit().
        """
        if not description_text or len(description_text.strip()) < 10:
            return {"verdict": "Unknown", "confidence": 0.0, "red_flag": False}
        kw = _classify_by_keyword(description_text)
        if kw is not None:
            return {
                "verdict": kw["verdict"],
                "confidence": kw["confidence"],
                "red_flag": kw["red_flag"],
            }
        clean = re.sub(r'<[^<]+?>', ' ', description_text)
        result = self._classify_by_text_api(clean.strip())
        if result is not None:
            return {
                "verdict": result["verdict"],
                "confidence": result["confidence"],
                "red_flag": result["red_flag"],
            }
        return {"verdict": "Unknown", "confidence": 0.0, "red_flag": False}

    def batch_audit(self, df: pd.DataFrame, progress_callback=None,
                    batch_size: int = 8, live_callback=None) -> pd.DataFrame:
        """Three-tier condition audit. See class docstring for the flow.

        Args:
            df: DataFrame with 'title', 'description', and (ideally)
                'thumbnail_url' columns.
            progress_callback: Optional callable(current, total) for progress
                updates. Fires after each lot is classified.
            batch_size: Max parallel API workers. Renamed from "batch size"
                in the old transformer-based version; signature kept for
                compatibility with app.py session state. 8 is a sensible
                default — pushes through ~8 lots/sec when hot.
            live_callback: Optional callable(processed, total, partial_df)
                that fires periodically (~every 600ms) with the in-progress
                results so the UI can stream updates.

        Returns:
            DataFrame with 'enriched_title', 'verdict', 'confidence',
            'red_flag', and 'audit_source' columns added. The audit_source
            column tells the UI which tier classified each lot ('keyword',
            'text_api', 'image_api', 'skip_hard', 'skip_collectible',
            'skip_empty', 'no_signal', or '*_api_failed').
        """
        total = len(df)
        titles = (
            df['title'].fillna('').astype(str).tolist()
            if 'title' in df.columns else [''] * total
        )
        descs = (
            df['description'].fillna('').astype(str).tolist()
            if 'description' in df.columns else [''] * total
        )
        thumbs = (
            df['thumbnail_url'].fillna('').astype(str).tolist()
            if 'thumbnail_url' in df.columns else [''] * total
        )
        logistics = (
            df['logistics_ease'].fillna('').astype(str).tolist()
            if 'logistics_ease' in df.columns else [''] * total
        )

        # --- Step 1: enrich titles (regex-only, microseconds) ---
        enriched_titles = [
            self._enrich_title(t, d) for t, d in zip(titles, descs)
        ]

        # --- Step 2: pre-classify (skip flags) ---
        verdicts = [None] * total
        confidences = [0.0] * total
        red_flags = [False] * total
        sources = [''] * total

        skip_indices = set()
        hard_count = 0
        collectible_count = 0
        empty_count = 0
        fashion_jewelry_count = 0
        fake_pokemon_count = 0
        card_auth_pending = []   # [(i, thumbnail_url), ...] pokemon cards

        for i, (t, d, thumb, log) in enumerate(
            zip(titles, descs, thumbs, logistics)
        ):
            pickup_only_in_desc = bool(
                d and _PICKUP_ONLY_AUDIT_RE.search(d)
            )
            is_hard = (log == 'HARD') or pickup_only_in_desc
            is_collectible = (
                not is_hard and bool(classify_for_pricecharting(t))
            )
            # Fashion-jewelry detector — Chinese drop-ship costume
            # jewelry laundered through US auction houses (Property 1
            # Vegas LLC etc.). Pattern matches "GRA cert", "diamond
            # moissanite", "gold over silver", "925 + diamond/sapphire",
            # "adjustable size 6-10", etc. — see auth_check.py for the
            # full list. Triggered lots get red-flagged here, never
            # reach AI audit, never reach comps.
            jewelry_match = (
                None if (is_hard or is_collectible)
                else detect_fashion_jewelry(t, d)
            )
            # Fake-Pokémon check runs BEFORE the collectible skip — a
            # bootleg card classifies as a collectible (tcg) and would
            # otherwise sail straight to comps against real-card prices.
            # Text tier here; the vision tier flags photo-only fakes.
            fake_pokemon = (None if is_hard else detect_fake_pokemon(t, d))
            if is_hard:
                verdicts[i] = "Unshippable (HARD logistics)"
                confidences[i] = 100.0
                red_flags[i] = True
                sources[i] = "skip_hard"
                skip_indices.add(i)
                hard_count += 1
            elif fake_pokemon is not None:
                verdicts[i] = f"Likely fake Pokémon: {fake_pokemon}"
                confidences[i] = 100.0
                red_flags[i] = True
                sources[i] = "fake_pokemon"
                skip_indices.add(i)
                fake_pokemon_count += 1
            elif is_collectible:
                # Collectibles pass straight through to comps; the AI's
                # risk labels don't apply when condition is encoded as a
                # grade in the title (PSA/BGS, sealed games, etc.)
                verdicts[i] = "Skipped (collectible)"
                confidences[i] = 0.0
                red_flags[i] = False
                sources[i] = "skip_collectible"
                skip_indices.add(i)
                collectible_count += 1
                # Pokémon cards get a photo authenticity check (Step 4b)
                # before they comp — bootlegs have honest titles and would
                # otherwise comp against real-card prices. Require card
                # context so plush / blankets / toys aren't sent here.
                if (thumbs[i] and _POKEMON_LOT_RE.search(t)
                        and _CARD_CONTEXT_RE.search(t)):
                    card_auth_pending.append((i, thumbs[i]))
            elif jewelry_match is not None:
                verdicts[i] = (
                    f"Fashion jewelry: {jewelry_match}"
                )
                confidences[i] = 100.0
                red_flags[i] = True
                sources[i] = "fashion_jewelry"
                skip_indices.add(i)
                fashion_jewelry_count += 1
            elif (not d or len(d.strip()) < 10) and not thumb:
                # Nothing to work with — no description, no image
                verdicts[i] = "Unknown"
                confidences[i] = 0.0
                red_flags[i] = False
                sources[i] = "skip_empty"
                skip_indices.add(i)
                empty_count += 1

        # --- Step 3: keyword pass on remaining lots ---
        keyword_hits = 0
        text_pending = []   # [(i, clean_desc), ...]
        image_pending = []  # [(i, thumbnail_url), ...]
        for i in range(total):
            if i in skip_indices:
                continue
            d = descs[i]
            kw_result = _classify_by_keyword(d)
            if kw_result is not None:
                verdicts[i] = kw_result["verdict"]
                confidences[i] = kw_result["confidence"]
                red_flags[i] = kw_result["red_flag"]
                sources[i] = kw_result["source"]
                keyword_hits += 1
                continue
            # Fallthrough: route to AI tier based on description length.
            clean_desc = re.sub(r'<[^<]+?>', ' ', d).strip() if d else ''
            clean_desc = re.sub(r'\s+', ' ', clean_desc)
            if len(clean_desc) >= self._MIN_DESC_FOR_TEXT_API:
                text_pending.append((i, clean_desc))
            elif thumbs[i]:
                image_pending.append((i, thumbs[i]))
            else:
                # Short description AND no image — can't classify
                verdicts[i] = "Unknown"
                confidences[i] = 0.0
                red_flags[i] = False
                sources[i] = "no_signal"

        # --- Step 4: parallel API calls (text + image, same worker pool) ---
        api_total = len(text_pending) + len(image_pending)
        api_done = 0
        # Skipped + keyword-classified lots are already done — count them
        # toward the progress total so the bar reaches 100% naturally.
        progress_offset = total - api_total

        def _emit_progress():
            if progress_callback:
                progress_callback(progress_offset + api_done, total)

        def _build_partial_df():
            out = df.copy()
            out['enriched_title'] = enriched_titles
            out['verdict'] = verdicts
            out['confidence'] = confidences
            out['red_flag'] = red_flags
            out['audit_source'] = sources
            return out

        def _emit_live():
            if live_callback:
                try:
                    live_callback(
                        progress_offset + api_done,
                        total,
                        _build_partial_df(),
                    )
                except Exception:
                    pass

        _emit_progress()
        _emit_live()

        if api_total > 0 and self.client is not None:
            max_workers = max(1, int(batch_size or 8))
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {}
                for i, clean in text_pending:
                    fut = ex.submit(self._classify_by_text_api, clean)
                    futures[fut] = ('text', i)
                for i, thumb in image_pending:
                    fut = ex.submit(self._classify_by_image_api, thumb)
                    futures[fut] = ('image', i)

                last_live = time.time()
                _first_api_error_logged = False
                for fut in as_completed(futures):
                    kind, i = futures[fut]
                    try:
                        result = fut.result()
                    except Exception as _api_exc:
                        # Log the FIRST failure per run with full detail.
                        # These used to vanish entirely — the 7/6 Hayworth
                        # run failed 160/160 API calls (SSL) and the only
                        # visible symptom was `image_api_failed` in a CSV
                        # export. One loud line makes systemic failures
                        # (bad key, SSL, dead model name) diagnosable
                        # without spamming 160 tracebacks.
                        if not _first_api_error_logged:
                            _first_api_error_logged = True
                            import sys as _sys
                            print(
                                f"[AUDIT] {kind}_api call failed "
                                f"({type(_api_exc).__name__}: {_api_exc}) "
                                f"— further failures this run logged "
                                f"only in audit_source counts.",
                                file=_sys.stderr, flush=True,
                            )
                        result = None
                    if result is None:
                        # API failed — don't punish the lot, mark Unknown
                        # so it still flows through to comps.
                        verdicts[i] = "Unknown"
                        confidences[i] = 0.0
                        red_flags[i] = False
                        sources[i] = f"{kind}_api_failed"
                    else:
                        verdicts[i] = result["verdict"]
                        confidences[i] = result["confidence"]
                        red_flags[i] = result["red_flag"]
                        sources[i] = result["source"]
                    api_done += 1
                    _emit_progress()
                    # Throttle live updates to ~600ms so the UI isn't
                    # rebuilding the partial df 8x/sec under load.
                    now = time.time()
                    if (now - last_live) > 0.6 or api_done == api_total:
                        last_live = now
                        _emit_live()
        elif api_total > 0:
            # No API key configured — mark everything pending as Unknown
            # so the audit completes without crashing. The UI surfaces
            # this via the diagnostic counts so the user knows to add
            # their key.
            for i, _ in text_pending + image_pending:
                verdicts[i] = "Unknown"
                confidences[i] = 0.0
                red_flags[i] = False
                sources[i] = "no_api_key"
                api_done += 1
            _emit_progress()
            _emit_live()

        # --- Step 4b: Pokémon card authenticity (vision) ---
        # Bootleg cards classify as collectibles and skip the audit → they'd
        # comp against real-card prices. Vision-check the pokemon ones; a
        # CONFIDENT fake gets red-flagged so it never comps. A None/uncertain
        # result leaves the lot as-is (skip_collectible → comps), so a blurry
        # photo never punishes a genuine card.
        if card_auth_pending and self._image_provider() is not None:
            _ca_workers = max(1, int(batch_size or 8))
            with ThreadPoolExecutor(max_workers=_ca_workers) as _cx:
                _cfuts = {
                    _cx.submit(self._assess_card_authenticity, _thumb): _i
                    for _i, _thumb in card_auth_pending
                }
                for _cf in as_completed(_cfuts):
                    _i = _cfuts[_cf]
                    try:
                        _res = _cf.result()
                    except Exception:
                        _res = None
                    if (_res and not _res.get("authentic", True)
                            and _res.get("confidence", 0) >= 55):
                        verdicts[_i] = (
                            f"Likely fake Pokémon (photo): "
                            f"{_res.get('reason', '')}"
                        )
                        confidences[_i] = float(_res.get("confidence", 60))
                        red_flags[_i] = True
                        sources[_i] = "fake_pokemon_vision"
                        fake_pokemon_count += 1

        if progress_callback:
            progress_callback(total, total)

        # --- Step 5: build result df + diagnostics ---
        out = _build_partial_df()

        # Promote logistics_ease to HARD for rows we re-flagged via the
        # description regex, so comps filtering / cache / styling agree.
        if 'logistics_ease' in out.columns:
            newly_hard_mask = [
                s == 'skip_hard' and log != 'HARD'
                for s, log in zip(sources, logistics)
            ]
            if any(newly_hard_mask):
                out.loc[newly_hard_mask, 'logistics_ease'] = 'HARD'

        text_api_count = sum(1 for s in sources if s == "text_api")
        image_api_count = sum(1 for s in sources if s == "image_api")
        text_api_failed = sum(1 for s in sources if s == "text_api_failed")
        image_api_failed = sum(1 for s in sources if s == "image_api_failed")
        no_api_key_count = sum(1 for s in sources if s == "no_api_key")
        no_signal_count = sum(1 for s in sources if s == "no_signal")

        out.attrs['audit_skipped_hard'] = hard_count
        out.attrs['audit_skipped_collectible'] = collectible_count
        out.attrs['audit_skipped_empty'] = empty_count
        out.attrs['audit_fashion_jewelry'] = fashion_jewelry_count
        out.attrs['audit_fake_pokemon'] = fake_pokemon_count
        out.attrs['audit_keyword_hits'] = keyword_hits
        out.attrs['audit_text_api_calls'] = text_api_count
        out.attrs['audit_image_api_calls'] = image_api_count
        out.attrs['audit_text_api_failed'] = text_api_failed
        out.attrs['audit_image_api_failed'] = image_api_failed
        out.attrs['audit_no_api_key'] = no_api_key_count
        out.attrs['audit_no_signal'] = no_signal_count
        out.attrs['audit_classified'] = (
            keyword_hits + text_api_count + image_api_count
        )
        out.attrs['audit_newly_hard_from_desc'] = sum(
            1 for s, log in zip(sources, logistics)
            if s == 'skip_hard' and log != 'HARD'
        )
        return out
