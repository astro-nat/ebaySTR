"""Authenticity scoring for luxury / vintage lots.

Two layers, separated by cost:

1. ``analyze_description(title, description, brand_match)`` is **free**:
   pure regex/keyword matching against the lot's title + description.
   Looks for:
   - Auctioneer disclosure language ("as-is", "no authentication") that
     is a strong signal the auctioneer themselves doubt the item.
   - Provenance language ("with original receipt", "from estate of...",
     "made in France", "with dust bag") that supports authenticity.
   - Era-marker hits — when the description mentions phrases from the
     BOLO file's per-brand ``era_markers`` list (date code, creed
     stamp, single-stitch, Big E, etc.), it's a positive signal.

2. ``analyze_photo(image_url, brand_match, anthropic_key, model)``
   spends a Claude vision call. Asks the model to look at the
   provided image and report which era_markers it can / can't see,
   plus any obvious red flags (off-center monogram, wrong hardware
   color, sloppy stitching). Returns a structured dict.

Both layers return the same shape so the caller can merge / compare:

    {
        "auth_score": 0..100,
        "red_flags": [str, ...],
        "green_flags": [str, ...],
        "era_seen": [str, ...],
        "era_missing": [str, ...],
        "notes": [str, ...],
        "method": "description" | "photo" | "merged",
    }

The description analyzer is automatic; the photo verifier should be
fired on demand (one vision call per lot is ~$0.005 with Haiku, so
running it on 50 BOLO matches is ~$0.25 — fine, but explicit is better
than auto).
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------
# Stylized / replica detection — whole-word regex patterns that catch
# the "Gucci style hat" / "Chanel-inspired bag" / "Rolex knockoff watch"
# family. These are CRITICAL because:
#
#   1. The auctioneer is telling you it's not authentic (using "style"
#      / "inspired" / "knockoff" is a tacit admission).
#   2. The comp pipeline will happily query eBay for "Gucci hat" and
#      return authentic-Gucci sold-comps, producing a wildly inflated
#      est_resale that doesn't apply to the lot at all.
#   3. The user almost gets burned bidding $20 expecting $300 resale.
#
# Whole-word matching prevents false positives:
#   "lifestyle"      → not "style"  (no boundary between "life" and "style")
#   "stylish"        → not "style"  (would need exact match)
#   "replica car"    → MATCH "replica"  (correct; a real fake)
# ---------------------------------------------------------------------
_STYLIZED_REPLICA_PATTERNS: List[Tuple[str, re.Pattern]] = [
    # Bare "[X] style" / "[X] styled" — catches "Gucci style", "Chanel style"
    ("style",          re.compile(r"\bstyle\b",            re.IGNORECASE)),
    ("styled",         re.compile(r"\bstyled\b",           re.IGNORECASE)),
    ("-style",         re.compile(r"\b\w+-style\b",        re.IGNORECASE)),
    # Inspiration / homage language
    ("inspired by",    re.compile(r"\binspired\s+by\b",    re.IGNORECASE)),
    ("inspired",       re.compile(r"\binspired\b",         re.IGNORECASE)),
    ("in the style of", re.compile(r"\bin the style of\b", re.IGNORECASE)),
    ("homage",         re.compile(r"\bhomage\b",           re.IGNORECASE)),
    # Direct fake / replica language
    ("replica",        re.compile(r"\breplica\b",          re.IGNORECASE)),
    ("imitation",      re.compile(r"\bimitation\b",        re.IGNORECASE)),
    ("knockoff",       re.compile(r"\bknock[-\s]?off\b",   re.IGNORECASE)),
    ("look-alike",     re.compile(r"\blook[-\s]?alike\b",  re.IGNORECASE)),
    # Auctioneer disclosure
    ("not authentic",   re.compile(r"\bnot\s+authentic\b",     re.IGNORECASE)),
    ("non-authentic",   re.compile(r"\bnon[-\s]?authentic\b",  re.IGNORECASE)),
    ("not genuine",     re.compile(r"\bnot\s+genuine\b",       re.IGNORECASE)),
    ("authenticity not", re.compile(r"\bauthenticity\s+not\b", re.IGNORECASE)),
    # Synthetic-stone disclosures — the canonical jewelry-replica
    # signal. Auctioneers list "1ct pink diamond moissanite ring" and
    # post a $850 estimate; eBay returns wide-spread comps that mix
    # genuine-diamond and moissanite listings, pulling the median into
    # the genuine-diamond zone. Flagging these as stylized prevents the
    # comp pipeline from treating them as real-diamond comparables.
    # Whole-word matching avoids false positives on "moissanite-style"
    # or "lab" appearing in other contexts.
    ("moissanite",      re.compile(r"\bmoissanite\b",          re.IGNORECASE)),
    ("CZ",              re.compile(r"\bCZ\b",                  re.IGNORECASE)),
    ("cubic zirconia",  re.compile(r"\bcubic\s+zirconi[ae]\b", re.IGNORECASE)),
    ("lab-grown",       re.compile(r"\blab[-\s]?grown\b",      re.IGNORECASE)),
    ("lab-created",     re.compile(r"\blab[-\s]?created\b",    re.IGNORECASE)),
    ("lab diamond",     re.compile(r"\blab\s+diamond\b",       re.IGNORECASE)),
    ("synthetic",       re.compile(r"\bsynthetic\b",           re.IGNORECASE)),
    ("simulated",       re.compile(r"\bsimulated\s+(?:diamond|sapphire|ruby|emerald|pearl|stone)\b",
                                   re.IGNORECASE)),
    ("faux",            re.compile(r"\bfaux\b",                re.IGNORECASE)),
    # White Sapphire is a common diamond stand-in in costume jewelry.
    # Real diamond rings rarely advertise stones as "white sapphire";
    # auctioneers using this language are flagging the synthetic.
    ("white sapphire",  re.compile(r"\bwhite\s+sapphire\b",    re.IGNORECASE)),
]


# Exemptions: titles with these tokens are NOT stylized/replica even
# if they contain "replica"/"style". Reason: graded comics + cards are
# real products with real comps; an issue happening to be named "X
# Replica #1" doesn't make the lot a fake. Same for art-grading
# services (PSA/CGC). We use whole-word regex to avoid false positives
# on words like "psalm" or "scgar" or whatever.
_STYLIZED_EXEMPT_PATTERNS: List[re.Pattern] = [
    re.compile(r"\bCGC\b",  re.IGNORECASE),
    re.compile(r"\bCBCS\b", re.IGNORECASE),
    re.compile(r"\bPSA\b",  re.IGNORECASE),
    re.compile(r"\bSGC\b",  re.IGNORECASE),
    re.compile(r"\bBGS\b",  re.IGNORECASE),
]


def detect_stylized_replica(title: Optional[str],
                            description: Optional[str] = None) -> Optional[str]:
    """Return the matched stylized/replica phrase or None.

    Used by the comp pipeline to skip lots whose title says "Gucci
    style hat" — running eBay comps on those returns authentic-Gucci
    prices that don't apply to the lot. Used by auth_check to force
    auth_score to 0 on these.

    Returns the FIRST matched phrase (a short string we can surface
    in the UI, e.g. "Gucci ⚠️ stylized: 'style'") so the user knows
    what triggered the flag.

    Exemption: titles with a recognized grading-service code (CGC,
    CBCS, PSA, SGC, BGS) are graded comics/cards — the issue name
    might happen to include "Replica" or "Style" but the lot is a
    real product with real comps. Bypass the flag to avoid skipping
    legitimate comp lookups on graded collectibles.
    """
    haystack = " ".join(filter(None, [title or "", description or ""]))
    if not haystack:
        return None
    for exempt_pattern in _STYLIZED_EXEMPT_PATTERNS:
        if exempt_pattern.search(haystack):
            return None
    for label, pattern in _STYLIZED_REPLICA_PATTERNS:
        if pattern.search(haystack):
            return label
    return None


# ---------------------------------------------------------------------
# Fashion-jewelry detector — Chinese drop-shipped costume jewelry
# laundered through US auction houses (Property 1 Vegas LLC, etc.).
# Every pattern below is a near-zero-false-positive tell that the
# lot is silver-plated lab-stone fashion jewelry being marketed as
# something more valuable.
#
# Real precious-metal jewelry NEVER:
#   - Sets diamonds/sapphires/emeralds in sterling silver (silver
#     tarnishes; diamonds get 14k+ gold or platinum)
#   - Comes with a "GRA Certificate" — GRA certifies moissanite, not
#     diamond. Real diamonds get GIA, IGI, AGS, AGL.
#   - Uses the phrase "Diamond Moissanite" — moissanite is not diamond.
#   - Says "Plated in white gold" / "gold over 925 silver" — these
#     describe silver-plated costume jewelry.
#   - Uses "Adjustable size 6-10" — adjustable rings are made of soft
#     costume metal that bends.
#
# Triggered lots get red-flagged at the audit stage with
# audit_source='fashion_jewelry' — never reach AI audit, never reach
# comps. Eliminates the entire class of "$1000 estimate, $25-90
# realized, $30 actual resale" auction-house traps in one pass.
# ---------------------------------------------------------------------
_FASHION_JEWELRY_PATTERNS: List[Tuple[str, re.Pattern]] = [
    # GRA = Gemological Research Association — certifies moissanite
    # ONLY. Real diamonds get GIA, IGI, AGS, AGL. Seeing "GRA" near
    # "certificate" or "diamond" or "moissanite" = lab moissanite.
    ("GRA cert (moissanite)", re.compile(
        r"\bGRA\b\s*(?:certificate|certified|cert\b|certif)",
        re.IGNORECASE)),
    # "Diamond Moissanite" — moissanite is NOT diamond. SEO spam from
    # Chinese fashion-jewelry drop-shippers.
    ("'diamond moissanite'", re.compile(
        r"\bdiamond\s+moissanite\b", re.IGNORECASE)),
    # Created / SIMULATED stones — jeweler boilerplate for FAKE stones
    # in otherwise-precious-looking listings. Reads the DESCRIPTION,
    # where the tell usually hides: the 7/23 scan had five "A/B"
    # gemstone lots whose titles said only "3.70 Ct Marquise
    # Engagement Ring" but whose descriptions said "Gemstones: created
    # simulated" — comps then priced them against REAL diamonds ($810
    # phantom profit). "created simulated" / "simulated <gem>" /
    # "cubic zirconia" / "diamonique" are unambiguous simulants.
    # DELIBERATELY EXCLUDES bare "lab grown/created diamond" — lab-grown
    # diamonds are chemically real and hold genuine (lower) resale
    # value; only simulants (glass/CZ) are the costume junk we skip.
    ("created/simulated stone", re.compile(
        r"\b(?:created\s+simulated|simulated\s+created)\b"
        r"|\bsimulated\s+(?:diamond|sapphire|emerald|ruby|gemstone|stone)\b"
        r"|\b(?:diamond|sapphire|emerald|ruby)\s+simulant\b"
        r"|\bcubic\s+zirconia\b|\bdiamonique\b",
        re.IGNORECASE)),
    # "[Color] gold over 925 / sterling / silver" — silver plated.
    ("gold-over-silver (plated)", re.compile(
        r"\b(?:white|yellow|rose)\s+gold\s+over\s+"
        r"(?:925|sterling|silver)\b",
        re.IGNORECASE)),
    # "Plated in [color] gold" — explicit gold plating.
    ("gold plated", re.compile(
        r"\bplated\s+in\s+(?:white\s+|yellow\s+|rose\s+)?gold\b",
        re.IGNORECASE)),
    # NOTE: "gold filled" and "gold overlay" patterns intentionally
    # removed (after Estate Deceased 5/9 audit). They false-positive
    # on legit antique vintage jewelry — antique 12K gold-filled chains
    # ($40-150 resale), vintage gold-filled watches ($30-200), etc.
    # The Chinese fashion-jewelry drop-shippers we're targeting say
    # "plated in white gold" or "white gold over 925", never "gold
    # filled" — so removing GF/GO patterns loses zero true positives.
    # "Stamped S925" / "Stamped 925" with a precious-stone claim
    # within ~80 chars — fashion stamping. Real jewelry is hallmarked,
    # not "stamped."
    ("stamped 925 + stone", re.compile(
        r"\bstamped\s+s?\.?925\b[^.]{0,80}\b"
        r"(?:diamond|sapphire|ruby|emerald|moissanite)\b",
        re.IGNORECASE | re.DOTALL)),
    # "Sterling silver" / "925 silver" + precious-stone claim within
    # ~80 chars. Real diamonds, sapphires, etc. aren't set in silver.
    # Bidirectional check (silver-then-stone OR stone-then-silver).
    ("silver + precious stone", re.compile(
        r"\b(?:925\s+silver|sterling\s+silver|s925)\b[^.]{0,80}\b"
        r"(?:diamond|sapphire|ruby|emerald)\b",
        re.IGNORECASE | re.DOTALL)),
    ("precious stone + silver", re.compile(
        r"\b(?:diamond|sapphire|ruby|emerald)\b[^.]{0,80}\b"
        r"(?:925\s+silver|sterling\s+silver|s925)\b",
        re.IGNORECASE | re.DOTALL)),
    # "Adjustable size 6-10" / "size 6 to 10" — costume soft-metal ring.
    # Real precious-metal rings are sized to one number.
    ("adjustable ring", re.compile(
        r"\badjustable\s+size\s+\d+\s*[\-to]+\s*\d+\b",
        re.IGNORECASE)),
    # "925 Sterling silver plated" / "Sterling silver plated" — Chinese
    # dropship signature (DOTEFFIL etc.). Different from honest "silver
    # plated" antique flatware because of the "925/sterling" prefix that
    # tries to imply real silver content. Caught after Mystery Mega
    # Auction 5/9 audit revealed 11 lots like "Hot 925 Sterling silver
    # plated Photo Frame" slipping through the existing patterns.
    ("silver-plated (dropship)", re.compile(
        r"\b(?:925|sterling|s925)\s+silver\s+plated",
        re.IGNORECASE)),
    # Known Chinese dropship vendor brands. These are pure-fabricated
    # SEO names that never appear on real jewelry. Cheap to add, near-
    # zero false-positive risk because they're nonsense English.
    ("dropship brand prefix", re.compile(
        r"\b(?:DOTEFFIL|JIAYIQI|VRIUA|QIHE|MOSTYLE|EUQEE|RUIYUNS)\b",
        re.IGNORECASE)),
]

# Carat-weight-vs-stone extractor for the synthetic heuristic below.
# Matches a weight adjacent (within ~40 chars) to an expensive-if-
# natural stone, in either order. "emerald" is guarded against
# "emerald cut" (a shape, not the stone). Groups: (1)=wt / (2)=stone
# for weight-first; (3)=stone / (4)=wt for stone-first.
_CARAT_STONE_RE = re.compile(
    r"(\d{1,3}(?:\.\d{1,2})?)\s*(?:ct|ctw|tcw|cttw|carats?)\b"
    r"[^.]{0,40}?\b(diamond|sapphire|emerald(?!\s*cut)|ruby)\b"
    r"|\b(diamond|sapphire|emerald(?!\s*cut)|ruby)\b[^.]{0,40}?"
    r"(\d{1,3}(?:\.\d{1,2})?)\s*(?:ct|ctw|tcw|cttw|carats?)\b",
    re.IGNORECASE,
)
# Above this total carat weight, a natural diamond/sapphire/emerald/
# ruby piece would be worth five figures — never a cheap estate lot.
_SYNTH_CARAT_THRESHOLD = 25.0


def detect_fashion_jewelry(title: Optional[str],
                           description: Optional[str] = None
                           ) -> Optional[str]:
    """Return the matched fashion-jewelry phrase or None.

    Used by the audit pre-classify step to red-flag silver-plated
    fashion jewelry (Chinese drop-ship channels — Property 1 Vegas
    LLC and similar) before any AI / comp credits are spent.

    Returns the FIRST matched label so the UI can surface what tripped
    the flag (e.g. "fashion jewelry: GRA cert (moissanite)").
    """
    haystack = " ".join(filter(None, [title or "", description or ""]))
    if not haystack:
        return None
    for label, pattern in _FASHION_JEWELRY_PATTERNS:
        if pattern.search(haystack):
            return label
    # Implausible natural carat weight → almost-certainly synthetic even
    # when the listing discloses nothing (7/23 scan: "76.80 CTW Yellow
    # Sapphire Tennis Bracelet" said nothing fake in title OR desc, but
    # 76 ct of natural sapphire would be a five-figure piece, never a
    # $600 estate lot). Only the expensive-if-natural stones (diamond /
    # sapphire / emerald / ruby) trip this — large NATURAL amethyst /
    # citrine / topaz / quartz are common and cheap, so they're excluded.
    # Threshold 25 CTW stays well clear of legit diamond tennis
    # bracelets (typ. 2-10 CTW).
    for m in _CARAT_STONE_RE.finditer(haystack):
        _wt = m.group(1) or m.group(4)
        try:
            if float(_wt) >= _SYNTH_CARAT_THRESHOLD:
                return f"implausible {_wt}ct natural stone (synthetic)"
        except (TypeError, ValueError):
            continue
    return None


# ---------------------------------------------------------------------
# Drop-ship-channel signature — broader than fashion-jewelry detection.
# Used at the AUCTION level to flag whole auctions whose lots are
# overwhelmingly Chinese SEO-spam product (DOTEFFIL silver-plated
# jewelry, hot-compress towels, eye masks, acrylic mirror stickers,
# etc.). When >30% of lots match, the auction gets a 🚨 dropship badge
# in the sidebar so the user can skip without analyzing.
#
# Each pattern below is an INDEPENDENT signal — a lot only needs ONE
# hit to count toward the auction-level percentage. The patterns are
# cheap regex; no per-lot iteration cost beyond the existing audit
# loop. Maintained as a separate set from fashion-jewelry because
# these patterns are non-jewelry-specific (towels, mats, gadgets).
# ---------------------------------------------------------------------
_DROPSHIP_LOT_PATTERNS = [
    # Marketing-adjective leads at start-of-title. "Hot 925", "Hot Sale",
    # "Wholesale", etc. — real auctions don't open this way. Trailing
    # `\d+\b` (not just `\d\b`) because "Hot 925" must match the FULL
    # number — `\b` between digits doesn't exist.
    re.compile(
        r"^\s*(?:Hot\s+(?:Sale|Selling|Compress|\d+)|"
        r"Fashion\s+Hot|Wholesale|Premium\s+Quality|Brand\s+New)\b",
        re.IGNORECASE),
    # Quantity-prefix titles ("60pcs Eye Mask", "5 Pairs Cat Lashes").
    re.compile(
        r"^\s*\d+\s*(?:pcs|pairs|sets|bag|box)\b",
        re.IGNORECASE),
    # Silver-plated product (catches auctioneer-style listings of the
    # dropship-jewelry pattern). No trailing `\b` because the dropship
    # CSVs frequently concatenate "plated" with the next word
    # ("silver platedPhoto Frame"), which would otherwise miss.
    re.compile(
        r"\b(?:925|sterling|s925)\s+silver\s+plated",
        re.IGNORECASE),
    # Common dropship product types. Bath(room) Mat handles both the
    # "Bath Mat" and the more common "Bathroom Mat" wording.
    re.compile(
        r"\b(?:Cat\s+Eye\s+Lashes|Faux\s+Lashes|Eye\s+Mask\s+Anti|"
        r"Hot\s+compress|Acrylic\s+Mirror\s+(?:Wall\s+)?Sticker|"
        r"Bath(?:room)?\s+Mat\s+(?:Cobblestone|Embossed)|"
        r"Refrigerator\s+Liner\s+Mat|"
        r"Aloe\s+Vera\s+Collagen|EVA\s+Waterproof)\b",
        re.IGNORECASE),
    # Made-up Chinese vendor brands.
    re.compile(
        r"\b(?:DOTEFFIL|JIAYIQI|VRIUA|QIHE|MOSTYLE|EUQEE|RUIYUNS)\b",
        re.IGNORECASE),
    # LotPop 7/8 signatures — AliExpress listing titles pasted verbatim.
    # Start-of-title marketing leads: "New Original 925...", "Fit
    # Original 925 Sterling Silver Charms" (Pandora-clone compat),
    # "Fine 925 Sterling Silver Luxury...", "Hot 925...".
    re.compile(
        r"^\s*(?:New\s+Original|Fit(?:s)?\s+Original|Original\s+925|"
        r"Fine\s+925\s+Sterling\s+Silver\s+(?:Luxury|Fashion)|"
        r"Hot\s+925|Luxury\s+Vintage\s+Titanium)\b",
        re.IGNORECASE),
    # Noun-soup gift endings: "...for Women Wedding Party Jewelry Fine
    # Pretty Holiday Gifts", "...Birthday Jewelry Gift Wholesale
    # Direct Sales". Real auctioneers never write like this; the
    # phrasing is SEO keyword-stuffing from marketplace exports.
    re.compile(
        r"\bfor\s+wom[ea]n\b[^.]{0,60}\b(?:wedding|party|birthday|"
        r"anniversary|holiday)\b[^.]{0,40}\bgifts?\b",
        re.IGNORECASE),
    re.compile(r"\bwholesale\s+direct\s+sales\b", re.IGNORECASE),
    # Karat-marked "gold plated" fashion jewelry (A2Z 7/12: "14K Gold
    # Plated" birth-flower ring comped $439 against real-gold solds;
    # actual value ~$5). Genuine gold lots never say "plated".
    re.compile(
        r"\b1[048]k\s+gold[\s-]+plated\b|\b2[24]k\s+gold[\s-]+plated\b",
        re.IGNORECASE),
    # Plated-steel posing as gold: "18K Gold ... Titanium Steel" /
    # "Titanium Steel ... 24k gold" in either order.
    re.compile(
        r"\b(?:titanium|stainless)\s+steel\b[^.]{0,40}\b(?:10|14|18|22|24)k\b"
        r"|\b(?:10|14|18|22|24)k\s+gold\b[^.]{0,40}\b(?:titanium|stainless)\s+steel\b",
        re.IGNORECASE),
]


def is_dropship_lot(title: Optional[str],
                    description: Optional[str] = None) -> bool:
    """True when a lot's title/description carries a drop-ship signature.

    Run this against every lot title; aggregate the True-rate per
    auction to get a `dropship_pct`. Above ~30% the auction is almost
    certainly a Chinese SEO-spam dumping ground. The optional
    description (first ~250 chars) catches listings whose keyword-
    stuffed marketing copy continues past the title (LotPop pattern:
    title is the product half, description is the "...for Women
    Wedding Party Jewelry Gifts" half).
    """
    if not title and not description:
        return False
    haystack = " ".join(filter(None, [
        title or "", (description or "")[:250],
    ]))
    for pattern in _DROPSHIP_LOT_PATTERNS:
        if pattern.search(haystack):
            return True
    return False


# ---------------------------------------------------------------------
# Keyword sets. Tuned for HiBid auctioneer copy specifically — they
# tend to use a small vocabulary of disclaimers that are highly
# diagnostic when present.
# ---------------------------------------------------------------------

# Strong signals that the auctioneer themselves doubt authenticity, OR
# that the lot is explicitly an inspired-by / replica piece. Each hit
# subtracts heavily from the score.
_AUCTIONEER_RED_FLAGS = [
    # Generic auction-house disclaimer language
    "as-is", "as is condition",
    "no authentication", "no auth",
    "buyer to verify", "buyer to authenticate",
    "buyer beware", "no returns",
    "not authenticated", "authenticity not guaranteed",
    # Explicit replica / homage / inspired-by language. These shouldn't
    # appear on a real luxury lot — when they do, the auctioneer is
    # telling you the item isn't authentic without saying so directly.
    "designer-style", "designer style",
    "in the style of", "inspired by",
    "replica", "homage piece",
    # "by Some Nobody" pattern from the May 4th HiBid auction —
    # framed art with designer names in titles. Treated as a hard
    # red flag because the lot is decorative, not the actual product.
    "by some nobody",
]

# Provenance / specificity signals. Each green hit adds modestly. Real
# auctioneers describe authenticated luxury pieces with these phrases;
# fake-flooded auctions tend to skip them.
_AUCTIONEER_GREEN_FLAGS = [
    # Documentation
    "with certificate", "authenticated by", "authenticity card",
    "comes with authenticity", "coa included",
    "original receipt", "purchase receipt", "with receipt",
    "original tag", "original tags",
    # Packaging signals
    "with original box", "original packaging",
    "original dust bag", "with dust bag",
    "original storage bag",
    # Provenance
    "from estate of", "estate find", "estate sale",
    "single owner", "one owner",
    # Country-of-origin tags (luxury houses cite these specifically)
    "made in italy", "made in france", "made in spain",
    "made in usa", "made in canada", "made in scotland",
    "italy stamp", "france stamp",
    # Specific authentication services / platforms
    "fashionphile", "the realreal", "trr authenticated",
    "ebay authenticity", "entrupy",
]

# Specifically watch-house disclosures common on jewelry/handbag lots
# that we treat as moderate red flags (auctioneer inviting buyer
# to inspect = they themselves haven't).
_INSPECTION_INVITES = [
    "buyer responsible for", "all sales final",
    "no warranty", "sold without warranty",
    "see photos for condition", "see photos for details",
]


def _lower(s: Optional[str]) -> str:
    return (s or "").lower()


def _scan_keywords(haystack: str, keywords: List[str]) -> List[str]:
    """Return the keyword phrases present in haystack, in order."""
    out = []
    for kw in keywords:
        if kw in haystack:
            out.append(kw)
    return out


def _scan_era_markers(haystack: str, era_markers: List[str]) -> Dict[str, List[str]]:
    """Detect era_marker hits in the description.

    Era markers from the BOLO JSON are descriptive phrases like
    "Older oval logo (pre-2000) is premium over current logo" or
    "Made in France/Italy/Spain/USA". To detect them we extract a
    short signature phrase (the first 2-3 distinctive words) and
    search for that. Imperfect but works for the common cases —
    "single-stitch", "creed stamp", "date code", "made in usa", etc.

    Returns ``{"seen": [...], "missing": [...]}``.
    """
    seen: List[str] = []
    missing: List[str] = []
    for marker in era_markers:
        if not isinstance(marker, str):
            continue
        # Pull a searchable signature out of the marker phrase.
        signature_candidates = _marker_signatures(marker)
        if not signature_candidates:
            missing.append(marker)
            continue
        if any(sig in haystack for sig in signature_candidates):
            seen.append(marker)
        else:
            missing.append(marker)
    return {"seen": seen, "missing": missing}


# Map common era-marker phrases to the literal substrings we search for.
# When the marker doesn't match a known pattern we fall back to its
# first 2-3 words.
_MARKER_SIGNATURE_MAP = {
    "single-stitch": ["single-stitch", "single stitch"],
    "single stitch": ["single-stitch", "single stitch"],
    "made in usa": ["made in usa", "made in u.s.a", "u.s.a."],
    "made in france": ["made in france"],
    "made in italy": ["made in italy"],
    "made in canada": ["made in canada"],
    "made in spain": ["made in spain"],
    "made in scotland": ["made in scotland"],
    "made in england": ["made in england"],
    "selvedge": ["selvedge", "selvage"],
    "big e": ["big e", "capital e"],
    "creed stamp": ["creed stamp", "creed"],
    "date code": ["date code"],
    "serial number": ["serial number", "serial #"],
    "ykk": ["ykk"],
    "aquaguard": ["aquaguard"],
    "gore-tex": ["gore-tex", "goretex", "gore tex"],
    "summit series": ["summit series"],
    "vachetta": ["vachetta"],
    "hidden rivets": ["hidden rivet"],
    "leather patch": ["leather patch"],
    "blanket-lined": ["blanket-lined", "blanket lined"],
    "stitched logo": ["stitched logo", "embroidered logo"],
    "gray tag": ["gray tag", "grey tag"],
    "union label": ["union label", "union-made", "union made"],
    "sanforized": ["sanforized"],
    "tin cloth": ["tin cloth"],
    "oil-finish": ["oil-finish", "oil finish"],
    "burberrys": ["burberrys"],
    "snow beach": ["snow beach"],
    "polo bear": ["polo bear"],
    "stadium": ["polo stadium"],
    "rrl": [" rrl ", "double rl"],
    "scot schmidt": ["scot schmidt"],
    "made with love": ["made with love"],
    "rip tag": ["rip tag"],
    "bonnie cashin": ["bonnie cashin"],
}


def _marker_signatures(marker: str) -> List[str]:
    """Build a list of substrings to search for when checking marker presence."""
    m = marker.lower()
    # Look up known patterns first — these are highly accurate.
    for key, sigs in _MARKER_SIGNATURE_MAP.items():
        if key in m:
            return sigs
    # Fallback: take the first two non-stop words. Don't include
    # parentheticals or filler ("is premium", "older").
    cleaned = re.sub(r"[\(\[].*?[\)\]]", "", m)
    cleaned = re.sub(r"[^a-z0-9 \-/]", " ", cleaned)
    tokens = [t for t in cleaned.split() if len(t) >= 3 and t not in
              {"and", "or", "the", "for", "with", "older", "newer",
               "older/desirable", "premium", "over", "current",
               "indicates", "tag", "tags"}]
    if len(tokens) >= 2:
        return [" ".join(tokens[:2])]
    if tokens:
        return [tokens[0]]
    return []


def analyze_description(title: Optional[str],
                        description: Optional[str],
                        brand_match: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Free description-based authenticity scan.

    Returns None when ``brand_match`` is None or has no useful data.
    Otherwise returns a dict with auth_score (0-100), found markers,
    and red/green flag lists.

    Score model (additive, clamped to 0-100):
      Base for auth-required lots              40   (tier-3 luxury where
                                                    neutral = ambiguous)
      Base for non-auth-required lots          50
      + 10 per era_marker matched in text      ([0, 30] capped)
      +  8 per green-flag phrase matched       ([0, 32] capped)
      - 25 per red-flag phrase matched         (no cap — auctioneer
                                                disclosure of doubt
                                                is the strongest signal
                                                we have)
      -  5 per inspection-invite phrase        (mild — common boilerplate)

    A score below 30 should be treated as "do not bid without
    authentication"; 30-60 is "ambiguous, inspect carefully"; 60+ is
    "supports authenticity, still verify hardware/stitching at the
    auction site".

    The auth-required base is intentionally pessimistic: a luxury lot
    that says nothing about provenance, dust bag, made-in-tag, or any
    era marker IS suspicious by default. Counterfeit dropshippers
    write generic catalog copy; real estate auctions describe the
    specifics.
    """
    if not brand_match:
        return None
    haystack = " ".join(filter(None, [_lower(title), _lower(description)]))
    if not haystack.strip():
        return None

    red = _scan_keywords(haystack, _AUCTIONEER_RED_FLAGS)
    green = _scan_keywords(haystack, _AUCTIONEER_GREEN_FLAGS)
    invites = _scan_keywords(haystack, _INSPECTION_INVITES)
    era = _scan_era_markers(haystack, brand_match.get("era_markers") or [])

    # Stylized / replica detection trumps everything. When the title
    # says "Gucci style" or "Chanel inspired" or "Rolex knockoff", the
    # lot isn't authentic and no comp from authentic eBay sales should
    # apply. Force the score to 0 and add a strong, named red flag so
    # the user sees exactly which phrase tripped it.
    stylized = detect_stylized_replica(title, description)
    if stylized:
        red = list(red) + [f"stylized: {stylized!r}"]

    score = 40 if brand_match.get("auth_required") else 50
    score += min(len(era["seen"]) * 10, 30)
    score += min(len(green) * 8, 32)
    score -= len(red) * 25
    score -= len(invites) * 5
    score = max(0, min(100, score))
    if stylized:
        score = 0  # hard floor — replica/style language is dispositive

    return {
        "auth_score": score,
        "red_flags": red,
        "green_flags": green,
        "inspection_invites": invites,
        "era_seen": era["seen"],
        "era_missing": era["missing"],
        "method": "description",
    }


# ---------------------------------------------------------------------
# Photo-based verification via Claude vision. One API call per lot,
# returns structured JSON. Caller decides when to fire it (per-lot
# button is a common UI choice — auto-running on 50+ BOLOs eats budget).
# ---------------------------------------------------------------------

_PHOTO_AUTH_SCHEMA = {
    "type": "object",
    "properties": {
        "era_seen": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Era markers from the brand list that you can VISUALLY confirm in the photo.",
        },
        "era_missing": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Era markers you specifically cannot see (hidden by angle / lighting / out of frame).",
        },
        "red_flags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Specific visual cues suggesting non-authenticity: off-center monogram, wrong hardware color, sloppy stitching, font mismatch on tag, etc.",
        },
        "green_flags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Visual cues supporting authenticity: clean stitching, correct hardware engraving, proper canvas alignment, etc.",
        },
        "confidence": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
            "description": "Your confidence (0-100) that this item is authentic given ONLY the photo evidence. Account for image quality.",
        },
        "image_quality": {
            "type": "string",
            "enum": ["clear", "ok", "poor"],
            "description": "Whether the image quality is sufficient to make an authentication judgment.",
        },
        "notes": {
            "type": "string",
            "description": "One-sentence summary of the strongest signal you saw, positive or negative.",
        },
    },
    "required": ["era_seen", "era_missing", "red_flags", "green_flags",
                 "confidence", "image_quality", "notes"],
}


_PHOTO_AUTH_SYSTEM = """You are an authentication assistant for second-hand luxury and vintage items. You will be shown a photo of a single item from an estate-auction catalog and asked to evaluate it against a brand-specific authentication checklist.

You are NOT a professional authenticator. Your job is to:
1. Report which items on the checklist you can or cannot see in the photo.
2. Flag specific visual cues that look "off" (a real authenticator would investigate further).
3. Flag specific visual cues that support authenticity.
4. Give an honest confidence number that accounts for image quality.

Be conservative. If you can't see a detail clearly, say "not visible" rather than guessing. A photo of just a logo close-up is rarely enough to authenticate; say so.

When the brand is luxury (Louis Vuitton, Chanel, Hermès, Gucci, Prada, Dior, etc.), specific things to look at:
- Monogram canvas alignment at seams (real LV monograms align across stitch lines)
- Heat-stamped logo crispness vs. printed-looking logos
- Hardware: brand engraving depth, gold/silver tone consistency, YKK or branded zippers
- Lining: brand-correct lining color and tag positioning
- Stitching count and tension
- Date code / serial number visibility (most luxury brands stamp inside)
- Color: real LV vachetta has a honey patina; obvious orange-leather is suspect"""


def analyze_photo(image_url: str,
                  brand_match: Dict[str, Any],
                  anthropic_key: Optional[str],
                  model: str = "claude-haiku-4-5",
                  timeout: float = 30.0) -> Optional[Dict[str, Any]]:
    """Vision-based authentication check. One Claude call per invocation.

    Caller is responsible for fetching ``image_url`` (we hand it to the
    Anthropic SDK as a URL block, which does the download server-side).

    Returns the photo dict shape, or None on any failure / no API key.
    """
    if not image_url or not brand_match or not anthropic_key:
        return None

    try:
        import anthropic  # type: ignore
    except ImportError:
        return None

    brand = brand_match.get("brand") or "unknown"
    era_markers = brand_match.get("era_markers") or []
    notes_for_prompt = brand_match.get("notes") or ""

    # Compose the per-lot user prompt: brand + era_markers + notes.
    checklist = "\n".join(f"- {m}" for m in era_markers) or "- (no specific markers — use general luxury heuristics)"
    extra = f"\n\nBrand-specific notes: {notes_for_prompt}" if notes_for_prompt else ""
    user_text = (
        f"Brand: {brand}\n\n"
        f"Authentication checklist (era markers from brand watch list):\n"
        f"{checklist}{extra}\n\n"
        "Evaluate the photo against this checklist. Return JSON matching "
        "the provided schema. Be specific — name the marker you saw or "
        "didn't see; don't summarize."
    )

    client = anthropic.Anthropic(api_key=anthropic_key)
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=1024,
            timeout=timeout,
            system=_PHOTO_AUTH_SYSTEM,
            tools=[{
                "name": "report_authentication",
                "description": "Report what you saw in the photo against the brand's authentication checklist.",
                "input_schema": _PHOTO_AUTH_SCHEMA,
            }],
            tool_choice={"type": "tool", "name": "report_authentication"},
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "url", "url": image_url},
                    },
                    {"type": "text", "text": user_text},
                ],
            }],
        )
    except Exception:
        return None

    # Pull the tool_use payload
    payload = None
    for block in resp.content:
        if getattr(block, "type", None) == "tool_use":
            payload = block.input
            break
    if not isinstance(payload, dict):
        return None

    # Translate confidence → auth_score for shape parity with the
    # description-based result. Penalize unusable images.
    confidence = int(payload.get("confidence") or 0)
    quality = payload.get("image_quality") or "ok"
    if quality == "poor":
        confidence = min(confidence, 40)

    return {
        "auth_score": confidence,
        "red_flags": list(payload.get("red_flags") or []),
        "green_flags": list(payload.get("green_flags") or []),
        "inspection_invites": [],
        "era_seen": list(payload.get("era_seen") or []),
        "era_missing": list(payload.get("era_missing") or []),
        "image_quality": quality,
        "photo_notes": payload.get("notes") or "",
        "method": "photo",
    }


def merge_results(desc_result: Optional[Dict[str, Any]],
                  photo_result: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Combine description + photo results into a single merged view.

    Score: weighted average favoring photo (60/40) when both exist.
    Lists: union, deduplicated.
    """
    if not desc_result and not photo_result:
        return None
    if not desc_result:
        return {**photo_result, "method": "photo"}
    if not photo_result:
        return {**desc_result, "method": "description"}

    merged_score = int(round(
        0.6 * (photo_result.get("auth_score") or 0)
        + 0.4 * (desc_result.get("auth_score") or 0)
    ))

    def _union(a, b):
        seen = set()
        out = []
        for item in (a or []) + (b or []):
            if item not in seen:
                out.append(item)
                seen.add(item)
        return out

    return {
        "auth_score": merged_score,
        "red_flags": _union(desc_result.get("red_flags"),
                            photo_result.get("red_flags")),
        "green_flags": _union(desc_result.get("green_flags"),
                              photo_result.get("green_flags")),
        "inspection_invites": _union(desc_result.get("inspection_invites"),
                                     photo_result.get("inspection_invites")),
        "era_seen": _union(desc_result.get("era_seen"),
                           photo_result.get("era_seen")),
        "era_missing": _union(desc_result.get("era_missing"),
                              photo_result.get("era_missing")),
        "image_quality": photo_result.get("image_quality") or "n/a",
        "photo_notes": photo_result.get("photo_notes") or "",
        "method": "merged",
    }
