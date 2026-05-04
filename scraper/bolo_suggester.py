"""Mine cached analyzed auctions for BOLO-list candidate brands.

Each comp run produces ``est_resale × comp_count × ebay_str`` per lot
and persists those columns to ``.cache/auctions/<id>.pkl``. Lots whose
titles AREN'T already matched by the BOLO matcher but DO comp well are
the user's strongest signal of what should be added to the BOLO list
next quarter.

This module walks every cached auction, identifies BOLO-misses with
solid comp signal, extracts a brand candidate from each title via
heuristic, aggregates per candidate, and ranks. Output is a
DataFrame the UI renders + a JSON scaffolding helper that turns a
candidate into the BOLO file's brand-entry shape.

Pure read-only — no API calls, no scraping.
"""
from __future__ import annotations

import json
import pickle
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


CACHE_DIR_DEFAULT = Path(".cache") / "auctions"


# ---------------------------------------------------------------------
# Brand-candidate extraction. The goal is to pull out the brand-shaped
# token sequence at the start of an auction title so we can group.
#
# Common HiBid title patterns we want to handle:
#   "Lululemon Align Leggings Size 6 Black"        -> "lululemon"
#   "Yeti Rambler 30oz Tumbler Stainless"          -> "yeti"
#   "Free People Movement Floral Mini Dress"       -> "free people"
#   "1 Box Pokemon Cards English ..."              -> drop the "1 box"
#   "Vintage Pyrex Cinderella Bowl Set"            -> "pyrex"
#   "Sterling Silver Necklace ..."                 -> nothing useful
#                                                     (sterling silver
#                                                     is material, not
#                                                     brand)
# ---------------------------------------------------------------------

# Words that should be stripped from the LEAD of a title because they're
# adjectives / quantifiers / generic descriptors, not brand tokens.
# Order matters: longest phrases first so we strip "set of" before "of".
_LEAD_STRIPPERS = [
    "lot of", "set of", "pair of", "box of", "case of",
    "group of", "collection of", "estate of",
    "vintage", "antique", "modern", "rare", "unique",
    "uncommon", "common", "scarce", "decorative",
    "new", "used", "preowned", "pre-owned", "second-hand",
    "mint", "excellent", "good", "fair", "poor",
    "mens", "men's", "womens", "women's", "kids", "kid's",
    "boys", "boy's", "girls", "girl's", "unisex",
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
    "1 box", "1 set", "1 lot", "1 pair", "1 piece",
    "10 pcs", "5 pcs", "3 piece",
    "gold", "silver", "sterling silver", "rose gold", "white gold",
    "yellow gold", "platinum",
    "leather", "suede", "velvet", "satin", "wool",
    "chinese", "japanese", "german", "french", "italian", "asian",
    "european", "oriental", "russian", "english",
    "the",
]

# Tokens that should NEVER be the start of a brand candidate. If after
# lead-stripping the first token is in this set, we drop the candidate.
_INVALID_LEAD_TOKENS = {
    # Materials — descriptive, not brand
    "leather", "suede", "velvet", "satin", "wool", "cotton",
    "silk", "linen", "denim", "canvas",
    "gold", "silver", "sterling", "platinum", "brass", "copper",
    "wood", "wooden", "glass", "ceramic", "porcelain", "marble",
    "metal", "plastic", "rubber", "stone",
    # Colors
    "black", "white", "red", "blue", "green", "yellow", "pink",
    "purple", "orange", "gray", "grey", "brown", "tan", "beige",
    "cream", "navy", "teal", "burgundy",
    # Generic noun lead-ins
    "set", "pair", "lot", "box", "group", "collection",
    "piece", "pieces", "pcs", "pc",
    # Sizes / dimensions
    "small", "medium", "large", "xl", "xxl",
    # Empty / numbers we missed
    "",
}

# Product-type words that indicate the candidate is genuinely the
# brand (preceded a noun like "necklace" or "skillet"). We DON'T strip
# these — they help validate that the preceding tokens were the brand.
# Used as anti-stoplist to avoid extracting "necklace" as a brand.
_PRODUCT_TYPE_WORDS = {
    "bag", "bags", "handbag", "handbags", "tote", "totes",
    "purse", "clutch", "wallet", "backpack",
    "shirt", "tshirt", "t-shirt", "tee", "blouse", "top",
    "pants", "jeans", "shorts", "leggings", "tights",
    "dress", "skirt", "jumpsuit", "romper", "gown",
    "jacket", "coat", "parka", "puffer", "pullover", "sweater",
    "shoe", "shoes", "sneaker", "sneakers", "boot", "boots",
    "watch", "watches", "ring", "necklace", "bracelet", "earring",
    "earrings", "pendant", "brooch", "pin",
    "skillet", "pan", "pot", "lid", "knob", "bowl", "bowls",
    "plate", "plates", "tumbler", "mug", "cup",
    "battery", "batteries", "drill", "saw", "wrench", "ratchet",
    "knife", "knives", "blade", "cleaver",
    "container", "attachment", "mixer", "blender",
}


def _normalize_title(title: str) -> str:
    """Lowercase + strip punctuation while preserving whitespace."""
    if not isinstance(title, str):
        return ""
    t = title.lower()
    # Remove parentheticals and brackets entirely
    t = re.sub(r"[\(\[].*?[\)\]]", " ", t)
    # Replace punctuation with spaces (keeps tokens separated)
    t = re.sub(r"[^a-z0-9 \-/]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _strip_lead(title: str) -> str:
    """Remove leading filler/quantifier words to expose the brand."""
    t = title
    changed = True
    while changed:
        changed = False
        for prefix in _LEAD_STRIPPERS:
            # Only strip whole-word prefixes followed by space
            if t.startswith(prefix + " "):
                t = t[len(prefix) + 1:]
                changed = True
    return t


def extract_brand_candidate(title: Optional[str]) -> Optional[str]:
    """Return the most-likely brand prefix from a title, or None.

    Strategy: normalize → strip leading filler → take first 1-2 tokens.
    A 2-token candidate wins when both tokens are plausibly part of a
    brand name (Free People, North Face, Le Creuset). Otherwise fall
    back to the first single token.
    """
    if not title:
        return None
    norm = _normalize_title(title)
    if not norm:
        return None
    norm = _strip_lead(norm)
    if not norm:
        return None
    tokens = norm.split()
    if not tokens:
        return None
    first = tokens[0]
    # Reject obvious non-brand leads
    if first in _INVALID_LEAD_TOKENS or first in _PRODUCT_TYPE_WORDS:
        return None
    if len(first) < 3:
        return None
    if first.isdigit():
        return None
    # Try a 2-token candidate when the second token isn't a product
    # type word (so we keep "le creuset" but drop "yeti rambler" → just "yeti").
    if len(tokens) >= 2:
        second = tokens[1]
        if (second not in _PRODUCT_TYPE_WORDS
                and second not in _INVALID_LEAD_TOKENS
                and not second.isdigit()
                and len(second) >= 3):
            # Preserve as 2-token candidate when the second word is
            # alphabetic and looks brand-like (Le Creuset, North Face,
            # Air Jordan). The single-token version still gets counted
            # because we group by the canonical brand name extracted.
            return f"{first} {second}"
    return first


def scan_cache(cache_dir: Path = CACHE_DIR_DEFAULT) -> pd.DataFrame:
    """Walk every .pkl in the cache and return one flat DataFrame.

    Each row is one analyzed lot. Includes the auction id/name as
    columns for traceability ("which auctions did this brand show
    up in").
    """
    if not cache_dir.exists():
        return pd.DataFrame()
    rows: List[pd.DataFrame] = []
    for pkl_path in sorted(cache_dir.glob("*.pkl")):
        try:
            with open(pkl_path, "rb") as f:
                payload = pickle.load(f)
        except Exception:
            continue
        df = payload.get("df")
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        df = df.copy()
        df["_auction_id"] = payload.get("auction_id")
        df["_auction_name"] = payload.get("auction_name", "(unknown)")
        df["_cached_at"] = payload.get("cached_at", "")
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True, sort=False)


def suggest_from_cache(
    bolo_matcher: Any,
    cache_dir: Path = CACHE_DIR_DEFAULT,
    min_resale: float = 20.0,
    min_comps: int = 3,
    min_occurrences: int = 2,
    max_results: int = 30,
) -> pd.DataFrame:
    """Mine the cache for brand candidates not yet in the BOLO file.

    Filtering chain:
      1. est_resale >= min_resale       (no point flagging $5 items)
      2. comp_count >= min_comps        (at least some sample size)
      3. NOT already matched by the     (skip whatever is already covered)
         current BOLO matcher
      4. brand candidate appears in     (one-off occurrences are noise)
         >= min_occurrences lots

    Score: occurrences × avg_resale × log(avg_comps + 1). Bigger when a
    brand shows up often, comps richly, AND has many comps to back the
    median up. Outputs a sorted DataFrame.
    """
    cache_df = scan_cache(cache_dir)
    if cache_df.empty:
        return pd.DataFrame()

    # Pick the best title column available. enriched_title is the
    # post-audit normalized version; fall back to img_enriched_title
    # (vision-based rewrite) or the original title if it's preserved
    # in the cached slim df.
    title_col = None
    for c in ("enriched_title", "img_enriched_title"):
        if c in cache_df.columns and cache_df[c].notna().any():
            title_col = c
            break
    if title_col is None:
        return pd.DataFrame()

    # Coerce numeric cols (cache may have object-dtype after pickle round-trip)
    for col in ("est_resale", "comp_count", "ebay_str"):
        if col in cache_df.columns:
            cache_df[col] = pd.to_numeric(cache_df[col], errors="coerce")

    # Cached auctions saved BEFORE the comps step ran (e.g., audit-
    # only cache writes from the streamed-discovery flow, or a comp
    # run that errored out partway through) don't have the
    # est_resale column. Without comps there's no way to score
    # brand candidates, so return early — the user gets an empty
    # suggestion list with no false drama.
    if "est_resale" not in cache_df.columns:
        return pd.DataFrame()

    # Filter to lots with meaningful comp signal
    eligible_mask = (
        cache_df["est_resale"].notna()
        & (cache_df["est_resale"] >= min_resale)
    )
    if "comp_count" in cache_df.columns:
        eligible_mask &= (
            cache_df["comp_count"].notna()
            & (cache_df["comp_count"] >= min_comps)
        )
    eligible = cache_df[eligible_mask].copy()
    if eligible.empty:
        return pd.DataFrame()

    # Identify which lots are already BOLO-matched. Run the matcher
    # against title + (description if any). The cache doesn't store
    # description for slim frames, so this is title-only most of the
    # time — that's fine, BoloMatcher.match() handles None description.
    eligible["_bolo_match"] = eligible[title_col].apply(
        lambda t: bool(bolo_matcher.match(str(t or ""), None)) if bolo_matcher.loaded else False
    )
    misses = eligible[~eligible["_bolo_match"]].copy()
    if misses.empty:
        return pd.DataFrame()

    # Extract a brand candidate per row
    misses["_brand"] = misses[title_col].apply(extract_brand_candidate)
    misses = misses[misses["_brand"].notna() & (misses["_brand"] != "")]
    if misses.empty:
        return pd.DataFrame()

    # Aggregate by candidate. We compute both the 1-token AND 2-token
    # versions so a brand like "Free People" rolls up correctly even
    # when some titles only say "Free" or "People" alone — but for now,
    # the candidate IS the rolled-up key.
    grouped = (
        misses.groupby("_brand")
        .agg(
            occurrences=("_brand", "size"),
            avg_resale=("est_resale", "mean"),
            max_resale=("est_resale", "max"),
            total_resale=("est_resale", "sum"),
            avg_comps=("comp_count", "mean") if "comp_count" in misses.columns else ("est_resale", "mean"),
            avg_str=("ebay_str", "mean") if "ebay_str" in misses.columns else ("est_resale", "mean"),
            sample_titles=(title_col, lambda s: list(s.head(3))),
            auctions_seen=("_auction_name", lambda s: list(pd.Series(list(s)).drop_duplicates().head(3))),
        )
        .reset_index()
        .rename(columns={"_brand": "brand_candidate"})
    )

    # Filter to candidates with enough repeated signal
    grouped = grouped[grouped["occurrences"] >= min_occurrences].copy()
    if grouped.empty:
        return pd.DataFrame()

    # Score: high-frequency × high-value × broad comp evidence. Log on
    # comps prevents one-comp 30-occurrence brands from dominating.
    avg_comps_safe = grouped["avg_comps"].fillna(1).clip(lower=1)
    grouped["score"] = (
        grouped["occurrences"]
        * grouped["avg_resale"].fillna(0)
        * np.log(avg_comps_safe + 1)
    ).round(0)
    grouped["avg_resale"] = grouped["avg_resale"].round(2)
    grouped["max_resale"] = grouped["max_resale"].round(2)
    grouped["total_resale"] = grouped["total_resale"].round(2)
    grouped["avg_comps"] = grouped["avg_comps"].round(1)
    grouped["avg_str"] = grouped["avg_str"].round(1) if "avg_str" in grouped.columns else None

    grouped = grouped.sort_values("score", ascending=False).head(max_results)
    grouped = grouped.reset_index(drop=True)
    return grouped


# ---------------------------------------------------------------------
# JSON scaffolding for "I want to add this to the BOLO file."
# ---------------------------------------------------------------------

def to_bolo_entry_scaffold(brand_candidate: str,
                           sample_titles: List[str],
                           avg_resale: float,
                           max_resale: float) -> Dict[str, Any]:
    """Generate a JSON-shaped entry the user can paste into a BOLO file.

    Numerical fields are seeded from the user's cache stats so the
    estimated comp range and target buy aren't pure guesses. The user
    still needs to fill in models, era_markers, and notes by hand —
    those are domain-knowledge fields the data can't synthesize.
    """
    # Title-case the brand candidate for the JSON
    pretty_brand = " ".join(w.capitalize() for w in brand_candidate.split())

    # Comp range: scale from cached observations
    suggested_low = max(round(avg_resale * 0.5), 5)
    suggested_high = round(max_resale)
    target_low = max(round(avg_resale * 0.05), 1)        # rough 5% of avg
    target_high = max(round(avg_resale * 0.20), 5)       # rough 20% of avg

    return {
        "brand": pretty_brand,
        "category": "TODO_PICK_CATEGORY",
        "tier": "TODO_1_OR_2_OR_3",
        "models": ["TODO_MODEL_1", "TODO_MODEL_2"],
        "era_markers": [
            "TODO: brand stamp / made-in tag / model number format / etc."
        ],
        "comp_ranges_usd": [
            {"item": "general", "low": suggested_low, "high": suggested_high}
        ],
        "target_buy_usd": {"low": target_low, "high": target_high},
        "platform_primary": "eBay",
        "platform_secondary": None,
        "comp_query_template": f"{pretty_brand.lower()} {{model}}",
        "ship_class": "TODO_poly_mailer_or_small_box_or_medium_box",
        "notes": (
            f"Auto-suggested from {len(sample_titles)} sample title(s). "
            f"Sample: {sample_titles[0][:80] if sample_titles else 'n/a'}"
        ),
    }


def format_scaffold_json(scaffold: Dict[str, Any]) -> str:
    """Pretty-print the scaffold dict as JSON the user can paste."""
    return json.dumps(scaffold, indent=2, ensure_ascii=False)
