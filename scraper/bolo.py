"""Brand BOLO ("be on lookout") matcher for clothing/accessories.

Loads a JSON brand-list (default: ``data/clothing_brand_bolo.json``) and
exposes ``BoloMatcher.match(title, description)`` so the rest of the app
can flag lots whose titles/descriptions hit a known resale-worthy brand.

The JSON file is meant to be swappable: the user updates it quarterly
with new brands/models/comp ranges and the matcher auto-reloads on
file-mtime change so a Streamlit rerun picks up the new list without
restart. Schema (abbreviated):

    {
      "brands": [
        {
          "brand": "Lululemon",
          "tier": 1,
          "category": "athleisure",
          "models": ["Align", "Scuba", ...],
          "target_buy_usd": {"low": 5, "high": 15},
          ...
        },
        ...
      ],
      "skip_list": ["Old Navy", "H&M", ...]
    }

Matching logic
--------------
Three layers, all whole-word + case-insensitive against ``title + " " +
description``:

1. Skip-list: if any skip-list entry is in the haystack AND no brand
   keyword is, return None (these are explicitly "don't bother" brands).
2. Brand exact name: case-insensitive, with smart aliasing for the
   awkward header rows in the JSON (e.g. "Polo Ralph Lauren premium
   sublines" → matches the literal "polo ralph lauren" + any subline
   token). Compound aliases like "Wrangler / Lee vintage" are split
   into both "wrangler" and "lee".
3. Model name: matched only when the brand name is also present in the
   haystack — prevents "Align" alone (a generic word) from triggering
   a Lululemon match.

A match returns a dict with brand / tier / category / matched_model /
target_buy_low / target_buy_high / platform_primary / notes — enough
for the UI to badge the lot, sort by tier, and surface a target-buy
ceiling next to current_bid.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union


# Default file paths, relative to repo root. Callers can override.
# Multiple files are loaded into one combined matcher so a single
# match() call covers all domains. The user can drop a new file in
# (e.g. data/sports_card_bolo.json) and add it here later — the
# matcher handles file-mtime hot-reloading per file.
DEFAULT_BOLO_PATHS: List[Path] = [
    # Pop-culture collectibles loads FIRST so toy/figure aliases
    # (Hot Wheels, Funko, Kenner) take precedence over auto-parts
    # brand aliases that share keywords (e.g., "Hot Wheels Brembo
    # die-cast" routes to Hot Wheels collector variants instead of
    # Brembo brakes).
    Path("data") / "pop_culture_collectibles_bolo.json",
    Path("data") / "clothing_brand_bolo.json",
    Path("data") / "household_parts_bolo.json",
    Path("data") / "fishing_tackle_bolo.json",
    # Fishing reels load BEFORE watch_accessories so "Zebco Omega Pro 3"
    # routes to Zebco fishing reels (the right answer) instead of
    # "Omega accessories" (the luxury-watch brand) — the word "Omega"
    # is the collision point.
    Path("data") / "fishing_reels_bolo.json",
    Path("data") / "printer_ink_bolo.json",
    Path("data") / "computer_parts_bolo.json",
    Path("data") / "apple_products_bolo.json",
    Path("data") / "golf_equipment_bolo.json",
    Path("data") / "audio_watches_bolo.json",
    # Luxury watches load BEFORE watch_accessories so a "Rolex Submariner
    # 116610LN" with a watch model code routes to the watch entry
    # ($8-12K) instead of the accessory entry ($60-700).
    Path("data") / "luxury_watches_bolo.json",
    Path("data") / "watch_accessories_bolo.json",
    # Designer eyewear / sunglass-case-as-collectible — loads BEFORE
    # clothing_brand designer entries so Chrome Hearts / Bentley OEM /
    # Oakley X-Metal / Tom Ford / Jacques Marie Mage route to the
    # eyewear-specific tier with case-only comp ranges, instead of the
    # generic "Designer luxury" clothing entry.
    Path("data") / "designer_eyewear_bolo.json",
    Path("data") / "musical_instruments_bolo.json",
    Path("data") / "camera_equipment_bolo.json",
    Path("data") / "auto_parts_bolo.json",
    Path("data") / "lightweight_collectibles_bolo.json",
    Path("data") / "estate_collectibles_bolo.json",
    Path("data") / "nostalgia_collectibles_bolo.json",
    Path("data") / "vintage_video_games_bolo.json",
    Path("data") / "western_wear_bolo.json",
    # Precious-metals loads LAST so brand-specific entries (Tiffany,
    # Cartier, Native American jewelry) take precedence — those have
    # higher resale ceilings than the generic precious-metal floor.
    Path("data") / "precious_metals_bolo.json",
]
# Back-compat single-path constant for callers that import it.
DEFAULT_BOLO_PATH = DEFAULT_BOLO_PATHS[0]


# ---------------------------------------------------------------------
# Persistent match-result cache — see BoloMatcher.match()
# ---------------------------------------------------------------------
# Path to the on-disk cache. Single file (not per-fingerprint) so we
# don't accumulate cruft as the BOLO files evolve. The fingerprint is
# stored INSIDE the file as a top-level key.
_BOLO_MATCH_CACHE_PATH = Path(".cache") / "bolo_match_cache.json"
# How many writes to absorb before flushing the in-memory cache to
# disk. Tuned for "scan 14k lots" use case — flushing every 500 misses
# means ~30 disk writes during a full scan, each ~50ms (negligible
# vs the ~6s of regex work the cache saves).
_BOLO_CACHE_FLUSH_EVERY = 500
# Sentinel object distinguishing "key not in cache" from "key in cache
# with value=None" (negative cache — title was checked before, no
# match found). Without this, dict.get(key) returns None for both.
_CACHE_SENTINEL = object()


# ---------------------------------------------------------------------
# Brand-name normalization. The JSON has some compound entries like
# "Wrangler / Lee vintage" or "Designer luxury (Gucci/Chanel/...)" that
# need to expand into multiple searchable aliases. We special-case them
# explicitly rather than try to derive aliases programmatically — the
# JSON header rows are a small, hand-curated list and the special cases
# don't change often.
#
# Each alias becomes a whole-word regex below. Order matters only for
# tiebreaking — when two aliases match, we keep the first (which is
# usually the more specific one).
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# "Hard kill" phrases — when ANY of these are in a lot title /
# description, we return None from match() regardless of which alias
# would otherwise hit. Use sparingly — these are unambiguous deal-
# breakers where the lot has near-zero resale value REGARDLESS of
# brand match.
# ---------------------------------------------------------------------
_HARD_SKIP_PATTERNS: List[str] = [
    # Apple-specific authentication locks — make device functionally
    # worthless to buyer. Always a no-bid.
    "icloud locked",
    "activation lock",
    "find my locked",
    # Carrier/IMEI status — heavily depresses but doesn't zero
    # resale; flag as hard-skip because a buyer can't unlock without
    # the original-account holder.
    "blacklisted imei",
    "bad imei",
    "blocked imei",
    # Unambiguous parts-only language. NOT included: "broken screen"
    # / "cracked screen" / "no power" because a cracked-screen
    # MacBook or "no power" iMac often has parts value (logic board,
    # battery, RAM, etc.) AND is sometimes a known easy fix that
    # restores full value. Those are auth_check-tier signals, not
    # hard kills.
    "for parts only",
    "sold for parts",
    "parts or repair only",
    # Diecast disqualifiers for AUTO PARTS specifically. Originally
    # included "hot wheels" + "matchbox" so a "Hot Wheels Brembo"
    # toy wouldn't false-match Brembo brakes. Now that we have a
    # dedicated 'Hot Wheels collector variants' BOLO entry, it
    # correctly catches collector Hot Wheels first (matcher's
    # JSON-order strong-match wins). The remaining hard-kill
    # phrases here are scale-model + kit signals that are too
    # broad / generic to put in any single BOLO entry.
    "1:18 scale",
    "1:24 scale",
    "1:43 scale",
    "1:64 scale",
    "model car kit",
]


# ---------------------------------------------------------------------
# Accessory-context guard ("compatible-with disqualifier").
# "Case for iPad", "lens for Canon EOS", "band fits Apple Watch" are
# cheap third-party accessories — the brand name appears in the title
# but the lot is NOT a brand item. When EVERY occurrence of a brand
# alias in the haystack is directly preceded by one of these phrases,
# the alias hit is rejected. One un-prefixed occurrence anywhere keeps
# the match alive ("Canon EOS R5 with lens for Canon" is still an R5).
# The regex is anchored at end-of-prefix ($) and tested against the
# ~40 chars before each alias occurrence.
# Production false positives that motivated this: "Noise Canceling
# Headphone Case for Bose" (A-graded as Bose headphones), "for Canon",
# "for iPad" accessory lots in the 743768 audit.
# ---------------------------------------------------------------------
_COMPAT_CONTEXT_RE = re.compile(
    r"(?:\bfor|\bfits?|\bcompatible\s+with|\bworks\s+with|"
    r"\breplacement\s+for|\bdesigned\s+for|\bmade\s+for|\bto\s+fit)"
    r"\s+(?:the\s+|your\s+|all\s+|select\s+|most\s+)?$"
)

# ---------------------------------------------------------------------
# Media-context guard for apparel-family brands.
# Movie/TV titles collide with clothing brand names: "DVD Movies -
# Buck, Wrangler, Jericho, Dakota" matched the Wrangler/Lee vintage
# BOLO in production (6/12 Longview audit). When media markers are
# present, ALIAS-ONLY matches for apparel-family categories are
# rejected. Strong (alias + model) matches still pass — a real
# "Wrangler 936 Cowboy Cut" mentioned alongside a DVD keeps matching.
# Deliberately NOT applied to nostalgia/electronics categories, where
# DVD-bearing lots are legitimate BOLO targets (Hannah Montana DVD
# player, Tamagotchi + DVD bundles).
# "film" is deliberately absent — it would kill camera-film lots.
# ---------------------------------------------------------------------
_MEDIA_CONTEXT_RE = re.compile(
    r"\b(?:dvds?|blu-?rays?|vhs|laserdisc|movies?|tv\s+series|"
    r"episodes|season\s+\d|complete\s+series)\b"
)
# Trigger-index tokenizer (PERF 7/13). Extracts literal alphanumeric
# tokens (≥2 chars) from alias pattern sources and from lot haystacks
# so the matcher can gate on a fast word-set intersection instead of a
# 361KB combined regex. Stopwords drop the boundary-class scaffolding
# artifacts ("za"/"z0" from `[A-Za-z0-9]`) that leak out of pattern
# sources — over-capturing a real trigger would only cost speed, but
# dropping one would miss matches, so the stopword set stays minimal.
_TRIGGER_TOKEN_RE = re.compile(r'[a-z0-9]{2,}')
_TRIGGER_STOPWORDS = frozenset({'za', 'z0'})

_MEDIA_GUARDED_CATEGORIES = frozenset({
    # Full apparel-family category list from clothing_brand_bolo.json,
    # western_wear_bolo.json, and vintage_clothing_bolo.json. Keep in
    # sync when adding apparel categories — a category missing here
    # means movie titles can alias-match that category's brands
    # (the original Wrangler bug was exactly this: 'vintage_denim'
    # wasn't guarded).
    "clothing", "athleisure", "boho_contemporary", "boots", "luxury",
    "luxury_mid", "outdoor", "premium_contemporary", "sneakers",
    "vintage_denim", "vintage_graphic", "vintage_heritage",
    "western_wear", "workwear", "denim",
})


def _all_alias_hits_are_accessory_context(alias_pat, haystack: str) -> bool:
    """True when every alias occurrence sits in 'for {brand}' context.

    Scans each regex hit of ``alias_pat`` and tests the preceding ~40
    characters against _COMPAT_CONTEXT_RE. Returns False the moment
    any occurrence is NOT accessory-prefixed (that occurrence keeps
    the brand match alive). Cheap: only runs on alias hits, which are
    rare relative to total lots scanned.
    """
    saw_any = False
    for m in alias_pat.finditer(haystack):
        saw_any = True
        prefix = haystack[max(0, m.start() - 40):m.start()]
        if not _COMPAT_CONTEXT_RE.search(prefix):
            return False
    return saw_any


_BRAND_ALIASES: Dict[str, List[str]] = {
    # JSON header → list of literal phrases to match in the haystack
    "Loungefly":                              ["loungefly", "loungefy", "lounge fly", "stitch shoppe", "mickey main attraction", "main attraction series"],
    # Nostalgia collectibles — late-90s through 2010s items with
    # sustained millennial-collector resale. Aliases are narrow
    # because we want the era markers ("Pleasant Company",
    # "Bluebird", "Coleco", "Tiger Electronics") doing the
    # disambiguation against modern reissues.
    "Webkinz":                                ["webkinz", "ganz webkinz", "lil kinz", "lil'kinz", "lilkinz", "mazin hamsters", "mazin' hamsters", "mazin hamster", "wkse", "wks1006", "wks1008", "wks1009", "wks1026", "wks1044", "wks1049"],
    "Neopets collectibles":                   ["neopets"],
    "Tamagotchi (vintage P1/P2)":             ["tamagotchi", "devilgotchi", "deviltchi", "angelgotchi", "yasashii", "tamawalkie", "tamatown", "tama-go", "twistetchi", "mametchi", "memetchi", "tamaotch", "hexagontchi", "morinotchi", "genjincchi", "tama deco", "deco pierce"],
    "Vintage Gymboree (Rainbow Tag era)":     ["gymboree", "vtg gymboree", "vintage gymboree", "rainbow tag gymboree", "gymboree rainbow tag", "gymbores"],
    "Chrome Hearts eyewear & cases":          ["chrome hearts", "chromehearts"],
    "Bentley OEM eyewear cases":              ["bentley continental", "bentley flying spur", "bentley oem", "bentley sunglasses case", "bentley glasses case", "bentley eyeglass case", "bentley console case"],
    "Maui Jim dealer display cases":          ["maui jim display", "maui jim sunglasses display", "maui readers display"],
    "Oakley X-Metal vault & display cases":   ["oakley x-metal", "oakley x metal", "oakley xmetal", "oakley vault", "oakley romeo", "oakley juliet", "oakley mars", "oakley penny", "oakley c six", "oakley c-six", "oakley pit boss", "oakley plasma", "oakley display case", "oakley display cabinet", "oakley sunglasses tower", "ferrari oakley", "ferrari × oakley", "oakley ferrari"],
    "Premium designer eyewear (Tom Ford / Persol Ratti / DITA / Oliver Peoples / Jacques Marie Mage)": [
        "tom ford sunglasses", "tom ford eyeglasses", "tom ford optical",
        "tom ford lennox", "tom ford prescott", "tom ford velvet case",
        "tom ford case", "tom ford brown velvet",
        "persol ratti", "persol meflecto",
        "dita eyewear", "dita eclipse",
        "oliver peoples", "oliver peoples lilletto",
        "jacques marie mage", "marie mage", "jmm sunglasses",
    ],
    "Designer sunglass cases & pouches (Christian Louboutin / Valentino / Celine / Brighton / BAPE / Miu Miu / Saint Laurent / Balmain / Harveys)": [
        "christian louboutin pouch", "louboutin pouch eyewear",
        "louboutin sole logo pouch",
        "valentino garavani sunglass case",
        "valentino sunglass case",
        "celine sunglasses case", "celine sunglass case",
        "brighton fashionista", "brighton pretty tough",
        "brighton sunglass case", "brighton eyeglass case",
        "miu miu sunglasses case", "miu miu glasses case",
        "miu miu velvet hard case",
        "saint laurent sl 51", "saint laurent sunglasses cloth case",
        "balmain sunglasses case", "balmain oval case",
        "bape camo clamshell", "bathing ape clamshell",
        "harveys seatbelt barbie",
        "tiffany tf4105hb", "tiffany co havana blue",
        "chopard schf",
        "wildfox lolita",
        "jean paul gaultier 75-8207", "jean paul gaultier limited 500",
        "gentle monster", "gm sunglasses maison",
        "jean lafont paris",
        "budd leather sunglasses case",
        "vintage ray-ban bausch lomb leather case",
        "chopard sunglasses", "chopard schf",
        "maybach the primadonna", "maybach primadonna",
        "maybach sunglasses",
        "elvis presley tcb", "elvis tcb sunglasses",
        "tcb sunglasses concert",
        "gargoyles 85s", "gargoyles made in usa",
        "vtg gargoyles",
        "balenciaga demna led", "balenciaga led sunglasses",
        "balenciaga by demna led",
        "vintage ray ban b&l", "ray-ban b&l 12k", "ray ban b&l 12k gf",
        "1/10 12k gf aviator",
        "louis vuitton x kusama", "lv x kusama", "kusama yayoi sunglasses",
        "ray ban saint laurent wayfarer",
        "meta ray-ban display ai", "ray-ban meta gen",
        "cartier ct0092o", "cartier moustique",
    ],
    "Original Furby (1998-2000)":             [
        # Bare "furby" hits modern Hasbro reissues too — narrow
        # to vintage-context aliases via models[] in the JSON.
        # Brand alias kept narrow.
        "tiger electronics", "vintage furby",
        "original furby", "furby babies", "furby baby",
    ],
    "Polly Pocket (Bluebird era)":            [
        "bluebird polly pocket", "polly pocket bluebird",
        "bluebird toys",
        # Bare "polly pocket" matches the modern Mattel reissues
        # too — disqualifier list rejects Mattel/post-1998 cases.
    ],
    "Vintage Cabbage Patch Coleco":           [
        "coleco cabbage patch", "cabbage patch coleco",
        "xavier roberts", "original appalachian artworks",
        "oaa cabbage patch",
    ],
    "American Girl Pleasant Company":         [
        "pleasant company", "pleasant co.", "pleasant co ",
        "pleasant rowland",
        "american girl doll", "american girl dolls",
        "american girl historical", "american girl truly me",
        "american girl just like you", "american girl jly",
        "american girl wicked", "american girl elphaba",
        "american girl glinda",
        "american girl grace", "american girl kanani",
        "american girl caroline", "american girl claudie",
        "american girl blaire", "american girl josefina",
        "american girl felicity", "american girl samantha",
        "american girl molly", "american girl kirsten",
        "american girl addy", "american girl kit",
        # "american girl" alone matches Mattel Barbie #1070 too —
        # the AG entry has explicit Barbie-context disqualifiers,
        # but we keep aliases narrow to specific AG product lines.
    ],
    "Pokemon TCG (vintage 1st Edition)":      [
        "pokemon 1st edition", "pokemon first edition",
        "pokemon 1st ed", "wotc pokemon",
        "wizards of the coast pokemon",
        "pokemon base set", "pokemon shadowless",
        "pokemon neo genesis", "pokemon neo discovery",
        "pokemon neo revelation", "pokemon neo destiny",
    ],
    "Yu-Gi-Oh TCG (vintage 1st Edition)":     [
        "yu-gi-oh 1st edition", "yu-gi-oh first edition",
        "yu-gi-oh 1st ed", "yugioh 1st edition",
        "yugioh first edition", "yugioh 1st ed",
        "yugioh lob", "legend of blue-eyes",
    ],
    "Vintage Beanie Babies (errors + apex)":  [
        # Bare "beanie baby" fires too aggressively on commodity
        # plush — narrow to error-and-apex compound aliases. The
        # model list inside the JSON catches the specific named
        # premium variants (Princess Diana, Peace, etc.).
        "1st gen beanie baby", "first gen beanie baby",
        "beanie baby pvc pellets", "beanie baby tag error",
        "princess diana beanie",
    ],
    # Vintage dolls — collector market (added 2026-05-05)
    "Vintage Mattel Barbie collector": [
        "vintage barbie", "vintage mattel barbie",
        "ponytail barbie", "bubblecut barbie",
        "side-part barbie", "side part barbie",
        "american girl barbie",
        "swirl ponytail barbie",
        "color magic barbie", "color magic fashion doll",
        "twist n turn barbie", "tnt barbie",
        "mod era barbie", "mod stacey",
        "stacey #1165", "stacey 1165",
        "francie no bangs", "francie barbie",
        "skipper barbie", "skooter barbie",
        "tutti barbie", "todd barbie", "chris barbie",
        "twiggy doll", "casey doll", "christie barbie",
        "live action barbie", "talking barbie",
        "hair fair barbie", "walking jamie",
        "teen age fashion model",
        "barbie #1190", "barbie #1160", "barbie #1144",
        "barbie 1190", "barbie 1160", "barbie 1144",
        "growin' pretty hair barbie", "growin pretty hair",
        "growin' pretty hair", "growing pretty hair",
        "quick curl barbie", "sweet 16 barbie",
        "malibu barbie 1971", "spanish barbie",
        "living barbie", "now look barbie",
        "free moving barbie",
        "rotoplast barbie", "tropical miko",
        "rotoplast tropical",
        "cipsa barbie", "venezuela barbie",
        "spanish speaking barbie",
        "talking brad", "stacey talking",
        "mib barbie 1958", "mib barbie 1964",
        "barbie 1958", "barbie 1959",
        "barbie 1960s", "barbie 1970s",
        "platinum blonde barbie", "pale blonde barbie",
        "sun kissed blonde barbie",
        "vintage mattel doll", "mattel japan barbie",
    ],
    "Madame Alexander vintage dolls": [
        "madame alexander", "alexander doll co",
        "madame alexander cissy", "madame alexander cissette",
        "madame alexander wendy", "madame alexander wendy-kins",
        "madame alexander maggie", "madame alexander margaret",
        "madame alexander sonja", "madame alexander margot",
        "madame alexander elise", "madame alexander madeline",
        "madame alexander dionne", "dionne quintuplets",
        "madame alexander quints",
        "madame alexander alice", "madame alexander snow white",
        "madame alexander scarlett",
        "madame alexander gone with",
        "madame alexander little women",
        "madame alexander beth", "madame alexander jo",
        "madame alexander meg", "madame alexander amy",
        "madame alexander storybook",
        "madame alexander international",
        "madame alexander americana",
        "cissy doll", "cissy 21 inch", "cissy 21\"",
        "cissette doll", "cissette 10",
        "bride cissy", "cissy in box",
        "wendy 8 inch", "wendy-kins",
    ],
    "Vintage doll collectors (general)": [
        "effanbee patsy", "effanbee dy-dee",
        "ideal shirley temple", "ideal crissy",
        "ideal saucy walker", "ideal toni",
        "vogue ginny", "vogue jill",
        "horsman bright star",
        "composition doll antique", "china head doll",
        "bisque head doll", "tin head doll",
        "antique german doll", "antique french doll",
        "bébé bru", "bebe bru", "jumeau doll",
        "bru jne", "sfbj doll",
        "kammer reinhardt", "simon halbig",
        "armand marseille", "gebruder heubach",
        "kestner doll", "kewpie doll", "bye-lo baby",
        "patsy doll", "dy-dee baby", "tiny tears",
        "betsy wetsy", "chatty cathy", "mrs. beasley",
        "raggedy ann vintage", "raggedy andy vintage",
        "volland raggedy", "knickerbocker raggedy",
        "baby dear wood knob", "baby dear wilkins",
        "wilkins baby dear",
        "lonely lisa", "lovely lori",
        "thumbelina doll", "tippy toes doll",
        "madame hendren", "heubach babies",
        "lenci doll", "käthe kruse", "kathe kruse",
        "steiff doll", "steiff teddy",
        "boudoir doll antique",
        "cloth body doll antique",
    ],
    # Vintage motorcycle parts (added 2026-05-05)
    "Vintage Indian motorcycle parts": [
        "indian scout", "indian chief", "indian four",
        "indian sport scout", "indian junior scout",
        "indian big chief", "indian 101 scout",
        "indian roadmaster", "indian warrior",
        "indian standard scout",
        "indian fender lamp", "indian fender light",
        "indian front fender lamp",
        "indian face front fender",
        "indian chrome lamp", "indian scout headlamp",
        "indian chief headlight",
        "indian war chief", "indian police chief",
        "indian little twin", "indian brockhouse",
        "indian engine", "indian gas tank",
        "indian fender skirt", "indian seat",
        "indian saddle", "indian foot board",
        "indian carburetor", "indian magneto",
        "indian speedometer", "indian spark plug",
        "indian decal", "indian emblem",
        "indian motocycle", "indian motorcycle co",
        "indian retro vintage scout", "scout chief v-twin",
    ],
    "Vintage Harley parts (Panhead/Shovelhead/Knucklehead)": [
        "harley knucklehead", "harley panhead",
        "harley shovelhead", "harley servicar",
        "harley flh", "harley flt",
        "harley fxr", "harley xlh",
        "harley xlch", "harley fxst", "harley flhr",
        "panhead saddlebag", "shovelhead saddlebag",
        "panhead saddlebag lock", "shovelhead lock",
        "fl flh saddlebag", "panhead flh fl shovelhead",
        "panhead seat", "shovelhead seat",
        "panhead gas tank", "shovelhead gas tank",
        "fl flh gas tank",
        "panhead speedometer", "shovelhead speedometer",
        "harley servicar speedometer",
        "panhead fender", "shovelhead fender",
        "panhead carburetor", "shovelhead carburetor",
        "panhead magneto", "shovelhead magneto",
        "panhead engine", "shovelhead engine",
        "knucklehead engine", "knucklehead carburetor",
        "harley servicar trike",
        "vintage harley davidson",
        "antique harley davidson",
        "panhead bullseye", "shovelhead bullseye",
    ],
    "Hannah Montana collectibles":            [
        "hannah montana", "miley stewart",
    ],
    "High School Musical collectibles":       [
        "high school musical",
        "hsm doll", "hsm mattel",
        "hsm 1", "hsm 2", "hsm 3",
        "hsm senior year",
        "sharpay evans", "gabriella montez",
        "troy bolton", "wildcats jersey",
    ],
    # Vintage video games — narrow brand aliases keyed off the
    # specific high-value game names. Pass 2g (added in match)
    # additionally fires on console+CIB/sealed compounds.
    "Vintage NES games (rare)":               [
        "stadium events", "nintendo world championships",
        "cheetahmen ii", "cheetahmen 2",
        "little samson", "panic restaurant",
        "bonk's adventure nes", "bucky o'hare nes",
        "mr. gimmick", "earthbound beginnings",
    ],
    "Vintage SNES games (rare)":              [
        "earthbound snes", "earthbound super",
        "chrono trigger", "hagane", "wild guns",
        "pocky and rocky", "demon's crest",
        "rendering ranger", "donkey kong country competition",
    ],
    "Vintage N64 games (rare)":               [
        "conker's bad fur day", "clayfighter sculptor",
        "clayfighter 63 1/3 sculptor",
        "ogre battle 64", "sin and punishment",
        "mischief makers", "snowboard kids 2",
    ],
    "Vintage Game Boy / GBA games (rare)":    [
        "pokemon red game boy", "pokemon blue game boy",
        "pokemon yellow game boy", "pokemon crystal game boy",
        "pokemon gold game boy", "pokemon silver game boy",
        "mother 3 gba", "boktai", "drill dozer",
        "klonoa empire of dreams",
        "metroid fusion gba", "metroid zero mission",
    ],
    "Vintage GameCube / Wii games (rare)":    [
        "skies of arcadia legends", "cubivore",
        "gotcha force", "pac-man vs",
        "fire emblem path of radiance",
        "metroid prime trilogy",
        "xenoblade chronicles wii",
        "the last story wii", "pandora's tower wii",
        "sin and punishment star successor",
        "geist gamecube",
    ],
    "Vintage PlayStation 1 / 2 games (rare)": [
        "suikoden ii", "suikoden 2 ps1",
        "suikoden iii", "suikoden v",
        "misadventures of tron bonne",
        "mega man legends 2",
        "klonoa door to phantomile",
        "lsd dream emulator", "tail concerto",
        "persona 2 eternal punishment",
        "persona 2 innocent sin",
        "persona 3 fes", "persona 3 portable",
        "persona 4 ps2", "shin megami tensei",
        "rule of rose", "haunting ground",
        "kuon ps2", "forbidden siren",
        "panzer dragoon saga", "snatcher sega",
        "radiant silvergun",
        "vagrant story", "xenogears",
        "chrono cross",
    ],
    "Vintage gaming consoles (rare/sealed)":  [
        "atari 2600 heavy sixer", "neo geo aes",
        "n64 pikachu", "jungle green n64",
        "game boy pocket pikachu",
        "spice orange gamecube", "hyrule edition wii",
    ],
    # Western wear — premium maker brands. Bare "Justin" / "Bailey" /
    # "Crockett" too generic; we use compound aliases that include
    # the western context so a "Justin Timberlake" lot doesn't fire
    # boot-tier matches.
    "Premium cowboy boots":                   [
        "lucchese", "tony lama",
        "olathe boot", "anderson bean",
        "rios of mercedes", "stallion boot",
        "old gringo", "rocketbuster",
        "liberty boot", "lane boots",
        # "Justin" alone is too broad (common name); compound only
        "justin exotic", "justin bent rail",
        "ariat caldera", "ariat heritage",
    ],
    "Premium western hats":                   [
        "stetson", "resistol",
        "american hat co", "american hat company",
        "greeley hat", "charlie 1 horse",
        "charlie one horse",
        # "Bailey" alone too generic — use compound
        "bailey hat", "bailey western",
        "atwood hat", "sunbody hat",
    ],
    "Western snap-front shirts":              [
        "rockmount", "h bar c", "h-bar-c",
        "panhandle slim", "scully western",
        "scully snap", "scully embroidered",
        "karman western",
        "tem tex", "cumberland outfitters",
        "miller international ranch",
    ],
    "Vintage western belt buckles":           [
        "montana silversmiths", "crumrine",
        "edward h bohlin", "bohlin",
        "sunset trails", "tom taylor santa fe",
        "comstock heritage", "carol felley",
        "hesston nfr", "prca championship",
        "trophy buckle vintage",
    ],
    "Bolo ties + cowboy western jewelry":     [
        "vintage bolo tie", "sterling bolo",
        "navajo bolo", "hopi bolo", "zuni bolo",
        "turquoise bolo", "concho belt",
        "squash blossom necklace",
    ],
    "Vintage spurs + western tack hardware":  [
        "crockett spurs", "buermann spurs",
        "august buermann",
        "north & judd spurs", "north judd",
        "garcia bit", "garcia spur",
        "g.s. garcia", "vintage spurs",
        "hand-engraved spurs",
        "rod mcafee spurs", "greg darnall",
        "mike morales spurs", "wade hubbard bit",
    ],
    # Musical instruments — narrow brand aliases. Pass 2f handles
    # implicit matches via model-name + instrument-context words.
    "Premium harmonicas":                     [
        "hohner", "lee oskar", "seydel",
        # "suzuki" is too broad (cars, motorcycles); we rely on
        # model-context matching for it via Pass 2f.
    ],
    "Premium microphones":                    [
        "shure", "neumann", "sennheiser",
        # AKG, Royer, Telefunken are audio-only brands — safe to
        # alias even without context. Bare "akg" risks people's
        # initials but the model number adjacent disambiguates.
        "akg", "royer", "telefunken",
        # "rca" / "coles" / "electro-voice" / "heil" — bare brand
        # too generic / overlapping (RCA = electronics, EV = trucks),
        # so model-context matching via Pass 2f handles them.
        # Bare Shure SKUs without "shure" prefix — eBay sellers
        # often write just "SM57 Cardioid Dynamic" without brand.
        "sm57 cardioid", "sm57 dynamic", "sm58 dynamic",
        "sm7b cardioid", "sm7b dynamic", "sm7b vocal",
        "sm7b microphone",
        "beta 58a supercardioid", "beta58 supercardioid",
        "beta 58a microphone", "beta58 microphone",
        "beta 58a vocal", "beta58 vocal",
        # Bare Sennheiser codes
        "e935 dynamic", "md421 dynamic",
        "ew 100 g", "ew 500 g", "evolution wireless",
        # Bare AKG codes
        "c414 condenser", "c451 condenser", "d112 dynamic",
        # --- Newer / streaming-tier microphone brands ---
        "blue snowball", "blue yeti", "blue spark",
        "blue baby bottle", "blue ember", "blue raspberry",
        "logitech blue", "blue bluebird",
        "audio-technica", "audio technica",
        "at2020", "at2035", "at2050", "at4040", "at4050",
        "at-829", "at829cw", "at-lavalier",
        "rode podmic", "rode podcaster",
        "rode nt1", "rode nt2", "rode nt5",
        "rode wireless go", "rode videomic",
        "rode lavalier",
        "elgato wave", "elgato wave xlr",
        "hyperx quadcast", "hyperx solocast",
        "hyperx duocast",
        "razer seiren", "razer seiren mini",
        "fifine k688", "fifine k669",
        "samson c01", "samson g-track",
        "samson q2u", "samson meteor",
        # --- Wireless lavalier / streaming systems ---
        "hollyland lark", "hollyland lark m2",
        "hollyland lark a1", "hollyland lark c1",
        "hollyland lark m1", "hollyland pyro",
        "hollyland mars",
        "rode wireless go ii", "rode wireless go 2",
        "dji mic", "dji mic 2", "dji mic mini",
        "comica vimo", "comica boomx",
        "saramonic blink", "saramonic blink me",
        "saramonic ulock", "saramonic uwmic",
        "boya by-wm4", "boya wireless",
        "lekato wireless mic", "lekato microphone",
        "lekato 5.8g",
        # --- Karaoke / party-box wireless systems ---
        "jbl partybox", "jbl wireless mic",
        "jbl wireless 2", "jblwirelessmicam",
        "jbl wireless microphones",
        "bose s1 pro mic",
        # --- Generic budget OEM ---
        "5core microphone", "5core xlr",
    ],
    "Effects pedals (guitar/bass)":           [
        "boss pedal", "boss compact",
        "mxr", "ibanez", "electro-harmonix",
        "klon centaur", "strymon", "eventide",
        "empress effects", "chase bliss",
        "jhs pedals", "jhs morning glory", "jhs bonsai",
        "earthquaker devices",
        "walrus audio", "proco rat", "pro co rat",
        "pro co rat2", "proco rat2",
        "analogman", "analog man", "analogman sun face",
        # Bare "ibanez" is fine (most contexts are guitar/pedal),
        # bare "boss" is too generic so we use compound aliases.
        # --- Premium / boutique brands from eBay sold-research ---
        "mesa boogie", "mesa/boogie", "mesa-boogie",
        "tone king", "schumann pll",
        "neural dsp", "quad cortex", "nano cortex",
        "marshall jfx", "marshall ed-1", "marshall ms-2",
        "marshall shred master", "marshall guv'nor",
        "marshall guvnor",
        "chandler tube driver", "chandler 12ax7",
        "foxx tone machine", "foxx fuzz",
        "foxx guitar synthesizer",
        "4ms pedals", "4ms tremulus", "4ms cv max",
        "4ms swash",
        "tc electronic", "tc sentry", "tc hall of fame",
        "tc flashback", "tc polytune", "tc ditto",
        "tc helix",
        "aer pocket tools", "aer colourizer", "aer bingo",
        "aer acoustic preamp",
        "headrush vx5", "headrush pedalboard",
        "headrush gigboard", "headrush mx5", "headrush prime",
        "trace elliot transit", "trace elliot elf",
        "digitech whammy", "digitech drop",
        "digitech trio", "digitech rp1000",
        "digitech rp500", "digitech rp360",
        "digitech jamman", "digitech hardwire",
        "rockman x100", "dunlop rockman",
        "korg miku stomp", "korg pitchblack",
        "korg toneworks", "korg volca",
        "korg mini kaoss",
        "moog moogerfooger", "moogerfooger",
        "mf-101 filter", "mf-102 ring",
        "mf-103 phaser", "mf-104 delay",
        "mf-105 murf", "mf-105m midi murf",
        "mf-107 freq box", "mf-108m cluster flux",
        "moog minifooger", "moog subharmonicon",
        "ehx microsynth", "ehx pitch fork",
        "ehx pico pog", "ehx hum debugger",
        "ehx bass mono synth", "ehx synth9",
        "ehx key9", "ehx cathedral", "ehx oceans 11",
        "jhs colour box", "jhs 3-series",
        "jhs pollinator", "jhs charlie brown",
        "jhs notaklon", "jhs notaklön",
        "earthquaker plumes", "earthquaker hizumitas",
        "earthquaker westwood", "earthquaker afterneath",
        "earthquaker dispatch master",
        "walrus mako", "walrus iron horse",
        "walrus r1", "walrus slötvå", "walrus lillian",
        "strymon iridium", "strymon nightsky",
        "strymon lex", "strymon riverside",
        "strymon sunset", "strymon compadre",
        "empress heavy", "empress effectsystem",
        "empress paraeq", "empress buffer",
        "empress tape delay",
        "chase bliss bliss factory", "chase bliss audio",
        "chase bliss cxm 1978", "chase bliss wombtone",
        "chase bliss brothers am", "chase bliss mood",
        "chase bliss blooper", "chase bliss tonal recall",
        "king of tone", "analog man king",
        "analog man sun face", "analog man beano",
        "analog man prince of tone",
        "ik multimedia tonex", "tonex one", "tonex pedal",
        "ik multimedia amplitube",
        "uafx", "uafx astra", "uafx dream 65",
        "uafx ox stomp", "uafx ruby", "uafx lion",
        "uafx heavenly", "uafx galaxy",
        "universal audio uafx",
        # --- Boss specific newer / vintage models ---
        "boss tb-2w", "boss tb2w", "boss tone bender",
        "boss gm-800", "boss gm800",
        "boss vg-800", "boss vg800",
        "boss sg-1 slow gear", "boss bcb-6", "boss bcb-60",
        "boss ge-7", "boss ge7",
        # --- MXR M-codes for specific volume models ---
        "mxr m101", "mxr m102", "mxr m133",
        "mxr m134", "mxr m135", "mxr m169",
        "mxr m196", "mxr mb301",
        "mxr m81", "mxr m87",
        # --- Budget Chinese OEM volume movers ---
        "sonicake", "sonicake portal",
        "sonicake compressor", "sonicake tone group",
        "sonicake pocket power", "sonicake matribox",
        "m-vave", "mvave", "m-vave mini universe",
        "m-vave cube baby", "m-vave bb-1", "m-vave cucurbit",
        "mooer hustle drive", "mooer yellow comp",
        "mooer audiofile", "mooer reverie",
        "mooer black truck", "mooer red truck",
        "donner yellow fall", "donner multi drive",
        "donner looper", "donner tutti love",
        "donner white tape", "donner tape delay",
        "donner pedal", "donner echo",
        "donner reverb",
        "joyo american sound", "joyo british sound",
        "joyo sweet baby", "joyo ultimate drive",
        "joyo ironman",
        # --- Distortion / overdrive specialists ---
        "maxon ts9", "maxon ts808", "maxon st9",
        "maxon od9", "maxon od808",
        "maxon overdrive",
        "fender godzilla", "fender hammertone",
        "fender pugilist", "fender santa ana",
        "fender mtg tube",
        "greer amps", "greer lightspeed",
        "greer 320 overdrive",
        "animals pedal", "animals push & pull",
        "animals push pull", "coalowl pedal",
        "univox super fuzz", "univox uni-fuzz",
        "vemuram jan ray", "vemuram myriad fuzz",
        "vemuram josh smith",
        "behringer centaur", "behringer tube monster",
        "behringer vintage tube",
        "way huge pork loin", "way huge pork pickle",
        "way huge aqua puss", "way huge green rhino",
        "way huge russian pickle", "way huge supa puss",
        "way huge swollen pickle", "way huge red llama",
        "way huge atreides", "way huge wm91",
        "wampler tumnus", "wampler plexi drive",
        "wampler sovereign", "wampler triple wreck",
        "wampler pinnacle", "wampler euphoria",
        "wampler ego comp", "wampler gearbox",
        "wampler golden jubilee", "wampler clarksdale",
        "wampler dracarys", "wampler velvet fuzz",
        "wampler tweed 57",
        "revv g3", "revv g4", "revv g8", "revv g20",
        "revv shawn tubbs",
        "dod 250", "dod overdrive 250", "dod 280",
        "dod fx", "dod gunslinger",
        "jca pedals", "jca texas flood",
        "origin effects", "origin cali76",
        "origin revival drive", "origin sliderig",
        "origin halcyon",
        # Marshall pedal-specific models
        "marshall 1959", "marshall jcm800",
        "marshall bluesbreaker", "marshall drivemaster",
        "marshall guv'nor reissue", "marshall guvnor reissue",
        # Boss specific overdrive/distortion variants
        "boss hm-2w", "boss hm2w",
        "boss od-1 over drive", "boss od1",
        "boss bd-2w", "boss bd2w",
        "boss sy-300", "boss sy300",
        "boss re-2", "boss re2",
        # EHX overdrive/distortion variants
        "ehx hot wax", "ehx hot tubes", "ehx crayon",
        "ehx little big muff", "ehx nano big muff",
        "ehx green russian", "ehx sovtek big muff",
        "ehx rams head big muff", "nano rams head",
        # --- Delay / echo / reverb specialists ---
        "roland space echo", "roland re-201", "roland re-301",
        "roland re-501", "roland re-555", "roland re-101",
        "re-201 space echo", "re-501 tape echo",
        "boss re-2", "boss re2",
        "boss re-202", "boss re202",
        "boss sde-3000", "boss sde3000", "sde-3000evh",
        "boss dd-200", "boss dd200",
        "boss dd-500", "boss dd500", "boss md-500",
        "meris mercuryx", "meris mercury x",
        "meris lvx", "meris polymoon", "meris hedra",
        "meris ottobit", "meris enzo",
        "death by audio", "death-by-audio", "deathbyaudio",
        "dba echo master", "dba echo dream",
        "dba rooms",
        "hotone verbera", "hotone skyline",
        "hotone soul press", "hotone xtomp",
        "hotone ampero", "hotone ravo",
        "source audio", "source audio collider",
        "source audio ventris", "source audio nemesis",
        "source audio eq2", "source audio true spring",
        "source audio sa263", "source audio sa270",
        "echoplex ep-3", "echoplex echo chamber",
        "echoplex digital pro",
        "moogerfooger mf-104", "mf-104z analog delay",
        "moog mf-104z",
        "keeley halo", "keeley zoma",
        "keeley caverns", "keeley eccos",
        "keeley dyna trem", "keeley compressor plus",
        "keeley electronics",
        "kinotone ribbons", "kinotone granular",
        "surfy industries", "surfybear",
        "surfy spring reverb",
        "universal audio golden reverberator",
        "universal audio dream 65",
        "universal audio orion", "universal audio astra",
        "ua golden reverberator",
        # Generic vintage/legacy delay model codes
        "dd-1100",
        # --- Phasers / pitch shifters ---
        "maestro ps-1a", "maestro ps1a", "maestro phase shifter",
        "maestro brassmaster", "maestro echoplex",
        "maestro ring modulator", "maestro fz-1",
        "maestro fuzztone",
        "maxon ph-350", "maxon ph350",
        "maxon ph-1", "maxon ch-1",
        "lekato pitch", "lekato pitch box",
        "lekato pitch shifter", "lekato guitar pedal",
        "roland jet phaser", "roland ap-7", "roland ap7",
        # Boss pitch shifter series
        "boss ps-2", "boss ps-3", "boss ps-5",
        "boss ps-6", "boss ps2", "boss ps3",
        "boss ps5", "boss ps6",
        "boss xs-1", "boss xs-100", "poly shifter",
        "boss ve-2", "boss ve-20", "boss ve-500",
        # MXR phaser-specific models
        "mxr m290", "mxr m107", "phase 95",
        "phase 100", "phase 90 evh", "phase 90 script",
        "evh phase 90", "phase tone",
        # --- Fuzz pedal specialists ---
        "jhs coyote", "jhs 1966 bender",
        "jhs 3-series fuzz",
        "dallas arbiter", "dallas-arbiter",
        "fuzz face vintage", "1969 fuzz face",
        "tone ar dallas arbiter",
        "cbs arbiter", "cbs/arbiter",
        "vox tone bender", "vox v828",
        "vox v829", "vox toneb",
        "prescription electronics", "prescription experience",
        "experience fuzz", "experience octave",
        "noisekick fx", "noisekick diabeetus",
        "zvex", "zvex effects",
        "zvex woolly mammoth", "woolly mammoth",
        "zvex fuzz factory", "fuzz factory vertical",
        "zvex lo-fi junky", "lo-fi junky vertical",
        "instant lo-fi junky",
        "zvex 59 sound", "zvex super hard on",
        "zvex mastotron", "zvex super duper",
        "dr. no", "dr no skull fuzz",
        "dr no superfuzz", "dr no octafuzz",
        "fuzzrite", "fuzz rite",
        "mosrite fuzzrite",
        "pigdog juju", "pigdog effects",
        "roger mayer", "roger mayer octavia",
        "roger mayer voodoo",
        "fender hello kitty", "hello kitty fuzz",
        "fender shields blender", "shields blender fuzz",
        "fender pugilist", "fender mtg",
        "fender level set",
        "bsm fuzzbender", "bsm pedals",
        "bsm fuzz machines",
        "r2r electric", "r2r nkt", "r2r ge distorter",
        "guyatone fuzz", "guyatone fz-2000",
        "guyatone fz2000",
        "king tone guitar", "kingtone",
        "kingtone octaland", "kingtone mini fuzz",
        "king tone duellist",
        "catalinbread", "catalinbread fuzzrite",
        "catalinbread moseley", "catalinbread sft",
        "pedal pawn", "pedal pawn fuzz",
        "pedal pawn ac128",
        "danelectro", "danelectro nichols",
        "danelectro fuzz drive",
        "maestro fuzz-tone", "maestro fz1",
        "maestro fz1b", "maestro fz-1b",
        "british pedal company", "british tonebender",
        "british pedal oc81d",
        "keeley fuzz bender",
        "dunlop fuzz face", "dunlop hendrix",
        "jim dunlop fuzz",
        "earthquaker hizumitas",
        "earthquaker hoof", "earthquaker erupter",
        "earthquaker bit commander",
        "way huge swollen pickle", "way huge atreides",
        # --- Chorus / vibrato / vibe specialists ---
        "shin ei", "shin-ei", "shinei",
        "shin ei univibe", "shin-ei uni-vibe",
        "honey vibe", "uni-vibe",
        "boss dc-2", "boss dc2", "boss dc-2w",
        "boss dc2w", "boss dc-3", "boss dc3",
        "boss ce-1", "boss ce1", "boss ce-2",
        "boss ce-2w", "boss ce2w",
        "boss ce-3", "boss ce-300", "boss ce300",
        "boss ce-5", "boss ce-20",
        "tech 21 sansamp", "tech 21 geddy lee",
        "tech 21 amalgamation", "tech 21 trademark",
        "tech 21 character", "tech 21 fly rig",
        "red witch", "red witch synthotron",
        "red witch familiar", "red witch pentavocal",
        "red witch grace", "red witch deluxe moon",
        "mu-tron", "mu tron", "mutron",
        "musitronics", "mu-tron iii", "mu-tron biphase",
        "mu-tron octave", "mu-tron flanger",
        "rivera 3d shaman", "rivera sedona",
        "rivera amplification",
        "fulltone", "fulltone ocd", "fulltone plimsoul",
        "fulltone dejavibe", "fulltone deja vibe",
        "fulltone mini dejavibe", "fulltone mdv",
        "fulltone fat-boost", "fulltone clyde",
        "fulltone supa-trem", "fulltone soulbender",
        "dunlop uv-1", "dunlop uv1",
        "dunlop univibe", "dunlop rotovibe",
        "beetronics", "beetronics fx",
        "beetronics seabee", "beetronics royal jelly",
        "beetronics whoctahell",
        "way huge smalls wm61", "blue hippo",
        "way huge blue hippo",
        "dawner prince", "dawner prince viberator",
        "dawner prince boonar",
        "arion sch-1", "arion sfl-1", "arion smm-1",
        "arion stereo chorus", "arion ews",
        "fjord fuzz", "fjord froy",
        "jam pedals", "jam waterfall",
        "jam wahcko", "jam delay llama",
        "jam ripply fall", "jam tubedreamer",
    ],
    "Sax / clarinet mouthpieces":             [
        "otto link", "vandoren",
        "berg larsen", "dukoff", "lawton mouthpiece",
        "brillhart mouthpiece", "ponzol mouthpiece",
        # "selmer" / "meyer" too generic — model-context matched.
    ],
    "Guitar pickups (standalone)":            [
        "seymour duncan", "dimarzio", "bill lawrence pickup",
        "lollar pickup", "tv jones", "fralin pickup",
        # "EMG" only in audio context, "Gibson" / "Fender" too
        # broad — model-context matched.
    ],
    "Music boxes + cylinder boxes":           [
        "reuge", "thorens music", "lador",
        "polyphon", "symphonion",
        # "sankyo" / "regina" / "stella" too generic; model-context.
    ],
    "Drum machines + samplers (small format)": [
        "teenage engineering", "akai mpc",
        "elektron model:",
        # "roland" / "korg" / "boss" / "akai" / "pioneer" all
        # too generic — model-context matched.
    ],
    "DJ cartridges / needles":                [
        "ortofon concorde",
        # "shure" already aliased on microphones, "stanton"
        # / "audio-technica" model-context matched.
    ],
    "Premium ukuleles":                       [
        "kamaka", "koaloha", "kanile'a", "kanilea",
        "ukulele",
        # "kala" / "pono" / "martin" too generic; "ukulele"
        # word in title is the clearest signal.
    ],
    # Precious metals — alias keys ARE the metal-content markers
    # themselves (no brand). Disqualifier guards (silver-tone /
    # gold-plate / etc.) are applied per-brand in Pass 1 from the
    # JSON's `_disqualifiers` array. We use specific compound
    # phrases rather than bare "silver" / "gold" to avoid
    # false-positive runs on costume jewelry.
    "Sterling silver":                        [
        ".925", "925 silver", "925 sterling",
        "sterling silver", "solid sterling", "solid silver",
        "s925",
    ],
    "Solid gold (10K-24K)":                   [
        "10k gold", "14k gold", "18k gold", "22k gold", "24k gold",
        "10kt gold", "14kt gold", "18kt gold", "22kt gold",
        "10k solid gold", "14k solid gold", "18k solid gold",
        "solid 14k", "solid 18k", "solid 10k",
    ],
    "Gold-filled":                            [
        "gold filled", "gold-filled",
        "12k gf", "14k gf", "12k g.f.", "14k g.f.",
        "1/20 12k gf", "1/20 14k gf", "1/10 12k gf",
        "12k gold filled", "14k gold filled",
        # Bare "1/20" / "1/10" / "1/40" notation = gold-filled by trade
        # convention (the fraction is the gold weight ratio). Without
        # these aliases, sellers writing "Real Solid 1/20 14K" misroute
        # to Solid Gold — which is wrong (5% gold ≠ 100% gold).
        "1/20 14k", "1/20 14kt", "1/20 14 k",
        "1/20 12k", "1/20 12kt",
        "1/20 10k", "1/20 10kt",
        "1/20 18k", "1/20 18kt",
        "1/10 14k", "1/10 12k", "1/10 18k",
        "1/40 14k", "1/40 14kt",
        "rolled gold plate", "rolled-gold plate",
    ],
    "Platinum":                               [
        "pt950", "pt900", "950 platinum", "900 platinum",
        "platinum ring", "platinum band", "platinum chain",
        "platinum necklace", "platinum bracelet",
        "platinum earring", "platinum pendant",
        "platinum wedding band", "platinum engagement",
        "iridplat", "iridium platinum", "solid platinum",
    ],
    # Luxury watches (data/luxury_watches_bolo.json — added 2026-05-05)
    # These entries come BEFORE the accessories entries below in load
    # order, so a "Rolex Submariner 116610LN" routes to the watch
    # entry ($8-12K) rather than the accessory entry ($60-700).
    "Rolex watches": [
        "rolex submariner", "rolex gmt-master", "rolex gmt master",
        "rolex daytona", "rolex cosmograph daytona",
        "rolex day-date", "rolex datejust",
        "rolex explorer", "rolex yacht-master", "rolex yacht master",
        "rolex sea-dweller", "rolex deepsea",
        "rolex sky-dweller", "rolex air-king", "rolex milgauss",
        "rolex oyster perpetual", "rolex president",
        "rolex cellini", "rolex land-dweller",
        # Common bare reference numbers (Rolex-only patterns)
        "submariner 116610", "submariner 126610", "submariner 16610",
        "submariner 5512", "submariner 5513", "submariner 1680",
        "gmt-master 116710", "gmt-master 126710",
        "gmt-master 116719", "gmt-master 126719",
        "daytona 116500", "daytona 116508", "daytona 126500",
        "daytona 6263", "daytona 6265", "daytona 6239",
        "day-date 228235", "day-date 228238", "day-date 228239",
        "datejust 126200", "datejust 126233", "datejust 126234",
        "datejust 126300", "datejust 16234",
        "explorer 214270", "explorer 124270",
        "yacht-master 116622", "yacht-master 126622",
        "sea-dweller 126600", "sky-dweller 326933",
    ],
    "Tudor watches": [
        "tudor black bay", "tudor pelagos",
        "tudor heritage", "tudor ranger", "tudor royal",
        "tudor 1926", "tudor north flag", "tudor glamour",
        "tudor style", "tudor fastrider",
        "tudor submariner", "tudor mini-sub",
        "tudor 79030", "tudor 79230", "tudor 79090",
        "black bay 58", "black bay gmt", "black bay pro",
        "black bay chrono", "bb58", "bb58 79030",
        "pelagos 39", "pelagos fxd", "pelagos lhd",
    ],
    "Patek Philippe watches": [
        "patek philippe nautilus", "patek nautilus",
        "patek philippe aquanaut", "patek aquanaut",
        "patek philippe calatrava", "patek calatrava",
        "patek philippe complications",
        "patek philippe annual calendar",
        "patek philippe perpetual calendar",
        "patek philippe world time", "patek philippe twenty-4",
        "patek philippe gondolo", "patek philippe ellipse",
        "patek golden ellipse", "patek philippe pilot",
        "patek philippe cubitus",
        "nautilus 5711", "nautilus 5712", "nautilus 5740",
        "nautilus 5980", "nautilus 5990", "nautilus 5811",
        "aquanaut 5167", "aquanaut 5168g", "aquanaut 5164",
        "aquanaut 5172", "aquanaut 5269",
        "calatrava 5196", "calatrava 5119", "calatrava 5227",
        "calatrava 5320g", "calatrava 6119",
        "patek 5930", "patek 5930p",
        "twenty-4 7300", "twenty-4 4910",
    ],
    "Audemars Piguet watches": [
        "audemars piguet royal oak", "ap royal oak",
        "audemars piguet code", "ap code 11.59",
        "royal oak 15202", "royal oak 15400", "royal oak 15500",
        "royal oak 15510", "royal oak 15710",
        "royal oak 26331", "royal oak 26334", "royal oak 26320",
        "royal oak 26240", "royal oak 26420", "royal oak 26430",
        "royal oak offshore",
        "code 11.59 41", "code 11.59 26393",
        "royal oak concept", "royal oak frosted",
    ],
    "Cartier watches": [
        "cartier tank", "cartier santos", "cartier panthère",
        "cartier panthere", "cartier ballon bleu",
        "cartier calibre", "cartier drive", "cartier roadster",
        "cartier pasha", "cartier crash", "cartier tortue",
        "cartier tonneau", "cartier coussin", "cartier baignoire",
        "cartier 21 must", "cartier la dona",
        "cartier cle", "cartier privé", "cartier prive",
        "tank française", "tank francaise",
        "tank americaine", "tank anglaise",
        "tank louis", "tank cintrée", "tank asymmetrique",
        "santos galbée", "santos galbee",
        "santos dumont", "santos 100",
        "santos de cartier",
        "panthère de cartier", "panthere de cartier",
        # Bare Cartier reference codes
        "wspn0006", "wspn0007", "wspn0009", "wspn0010", "wspn0019",
        "wssa0006", "wssa0010", "wssa0029", "wssa0030",
        "wjsa0021",
    ],
    "Richard Mille watches": [
        "richard mille",
        "rm 011", "rm 010", "rm 016", "rm 027", "rm 028",
        "rm 030", "rm 035", "rm 052", "rm 055", "rm 056",
        "rm 057", "rm 067", "rm 38", "rm 50", "rm 65",
        "rm 72-01", "rm 35-01", "rm 35-02", "rm 35-03",
        "rm up-01", "rm bonbon",
        "rm tourbillon", "rm skeleton",
        "rm016", "rm011", "rm010", "rm035", "rm055",
    ],
    "A. Lange & Söhne watches": [
        "a. lange & söhne", "a. lange söhne", "a lange & sohne",
        "a lange sohne", "a lange & söhne", "a.lange&sohne",
        "lange söhne", "lange sohne",
        "a. lange",
        "lange 1", "lange 1 moon", "lange 1 time zone",
        "lange 31", "lange zeitwerk",
        "lange saxonia", "lange datograph",
        "lange odysseus", "lange 1815",
        "lange richard lange", "lange cabaret",
        "datograph up down", "datograph perpetual",
        "saxonia thin", "saxonia outsize",
        "odysseus steel", "odysseus white gold",
        "zeitwerk date", "zeitwerk striking",
        "1815 up down", "1815 chronograph",
        "tourbograph pour le mérite",
        "ALS lange", "109.049", "191.039",
    ],
    "Omega watches": [
        "omega speedmaster", "omega seamaster",
        "omega constellation", "omega de ville",
        "omega railmaster", "omega aqua terra",
        "omega specialities",
        "speedmaster moonwatch", "speedmaster professional",
        "speedmaster moonphase", "speedmaster reduced",
        "speedmaster mark ii", "speedmaster snoopy",
        "speedmaster apollo", "speedmaster dark side",
        "speedmaster grey side",
        "seamaster diver", "seamaster 300", "seamaster 300m",
        "seamaster aqua terra", "seamaster planet ocean",
        "seamaster bullhead",
        "constellation globemaster",
        "constellation manhattan",
        "de ville trésor", "de ville prestige",
        "de ville tourbillon",
        # Bare Omega ref codes — full reference numbers only.
        # Short forms like "311.32" are too broad (false-positive on
        # any "311 ... 32" combo in jewelry / silverware titles).
        "311.30.42.30", "310.30.42.50", "210.30.42.20",
        "220.10.41.21", "232.30.42.21",
        "145.022", "105.012", "145.012",
    ],
    "Other Swiss luxury watches": [
        # IWC
        "iwc pilot", "iwc big pilot", "iwc top gun",
        "iwc mark xviii", "iwc spitfire",
        "iwc portuguese", "iwc portugieser",
        "iwc portofino", "iwc aquatimer",
        "iwc da vinci", "iwc ingenieur",
        "iw501001", "iw387901", "iw500705", "iw371417",
        # Breitling
        "breitling navitimer", "breitling superocean",
        "breitling avenger", "breitling premier",
        "breitling endurance", "breitling aerospace",
        "breitling chronomat", "breitling top time",
        "navitimer b01", "ab0142", "a1738830",
        # Vacheron Constantin
        "vacheron constantin overseas", "vacheron overseas",
        "vacheron constantin patrimony",
        "vacheron constantin traditionnelle",
        "vacheron constantin fiftysix",
        "vacheron constantin historiques",
        "4500v", "4500s", "5500v", "7900v",
        # Jaeger-LeCoultre
        "jaeger-lecoultre reverso", "jlc reverso",
        "reverso tribute", "reverso classic",
        "reverso duoface",
        "jaeger-lecoultre master",
        "jlc master ultra thin", "jlc master compressor",
        "jlc polaris", "jlc memovox",
        "jlc master geographic",
        # Panerai
        "panerai luminor", "panerai luminor marina",
        "panerai luminor 1950", "panerai submersible",
        "panerai radiomir", "panerai mare nostrum",
        "pam00422", "pam00111", "pam00112", "pam00382",
        # TAG Heuer (NB: bare 'tag' is too generic)
        "tag heuer carrera", "tag heuer monaco",
        "tag heuer aquaracer", "tag heuer formula 1",
        "tag heuer connected", "tag heuer autavia",
        "tag heuer link",
        "cv2a1u", "cbn2010", "cbm2110",
        # Hublot
        "hublot big bang", "hublot classic fusion",
        "hublot spirit of big bang", "hublot mp",
        "hublot king power",
        # Zenith
        "zenith el primero", "zenith defy",
        "zenith chronomaster", "zenith pilot",
        "zenith elite",
        # Bell & Ross
        "bell & ross br", "bell ross br",
        "bell & ross 03-92", "bell ross 03-92",
        "bell & ross br-x1",
        "bell & ross vintage",
        # Grand Seiko
        "grand seiko", "grand-seiko",
        "sbga211", "sbga", "sbge", "sbgn",
        "sbgm", "sbgr", "slga",
        "snowflake spring drive", "9f quartz",
        # Blancpain
        "blancpain fifty fathoms", "blancpain bathyscaphe",
        "blancpain villeret",
        # Glashütte Original
        "glashütte original", "glashutte original",
        "glashutte senator", "glashutte sixties",
        "panomatic",
        # Chopard
        "chopard l.u.c", "chopard luc",
        "chopard mille miglia", "chopard alpine eagle",
        "chopard happy sport",
    ],
    # Watch accessories — luxury-only, brand-name aliases. Implicit
    # matching (e.g., "Submariner box" without "Rolex" in title) is
    # handled by the Pass 2c branch in the matcher.
    "Rolex accessories":                      ["rolex", "tudor"],
    "Omega accessories":                      ["omega"],
    "Patek Philippe accessories":             ["patek philippe", "patek"],
    # Bare " ap " alias removed 7/12 — beyond the obvious acronym
    # collisions (AP exam, AP news), the title+description JOIN can
    # split a word across the boundary and manufacture the token out
    # of thin air: a truncated title ending "…Ankle Wr" + description
    # starting "ap for Sprain" matched 4 junk lots in one A2Z run.
    "Audemars Piguet accessories":            ["audemars piguet", "ap royal oak"],
    "Cartier accessories":                    ["cartier"],
    "Other Swiss luxury accessories":         [
        "vacheron constantin", "vacheron",
        "breitling", "iwc", "panerai",
        "jaeger-lecoultre", "jaeger lecoultre", "jlc",
        "tag heuer", "heuer",
        "zenith", "grand seiko", "hublot",
        "bell & ross", "bell ross",
    ],
    "Lululemon":                              ["lululemon"],
    "Alo Yoga":                               ["alo yoga", "alo "],
    "Athleta":                                ["athleta"],
    "Patagonia":                              ["patagonia"],
    "Vuori":                                  ["vuori"],
    "Gymshark":                               ["gymshark"],
    "Free People":                            ["free people", "fp movement"],
    "Anthropologie":                          ["anthropologie", "anthro "],
    "Reformation":                            ["reformation"],
    "Madewell":                               ["madewell"],
    "The North Face":                         ["the north face", "north face", " tnf "],
    "Arc'teryx":                              ["arc'teryx", "arcteryx", "arc teryx"],
    "Carhartt":                               ["carhartt"],
    "Columbia PFG":                           ["columbia pfg", " pfg "],
    "Filson":                                 ["filson"],
    "Levi's vintage":                         ["levi's", "levis ", "levi strauss"],
    # 'lee' alone false-matches "LEE CORN CUTTER" / "Lee Iacocca" /
    # other Lee surnames. Gate to denim/jeans/jacket/sanforized
    # context — Lee jeans are the actual BOLO target.
    "Wrangler / Lee vintage": [
        "wrangler",
        "lee riders", "lee storm rider", "lee jeans",
        "lee denim", "lee 101", "lee buddy lee",
        "lee jacket", "lee union", "lee work shirt",
        "lee sanforized", "vintage lee jeans",
    ],
    "Pendleton":                              ["pendleton"],
    "Woolrich vintage":                       ["woolrich"],
    "Polo Ralph Lauren premium sublines":     ["polo ralph lauren", "ralph lauren", "rrl ", "double rl"],
    "Tommy Hilfiger vintage 90s":             ["tommy hilfiger"],
    # Bare 'theory' false-matched "Theory of a Deadman Signed CD"
    # (band name). Gate to fashion-specific context.
    "Theory": [
        "theory blazer", "theory pant", "theory dress",
        "theory cashmere", "theory silk", "theory wool",
        "theory sweater", "theory jacket", "theory blouse",
        "theory treeca", "theory max",
    ],
    # Bare 'vince ' false-matched "Vince Coleman Signed Baseball"
    # (the player). Gate to fashion-specific context.
    "Vince": [
        "vince cashmere", "vince silk", "vince leather",
        "vince blouse", "vince dress", "vince sweater",
        "vince jacket", "vince pant", "vince camuto",
    ],
    "Eileen Fisher":                          ["eileen fisher"],
    "J.Crew Collection":                      ["j.crew", "jcrew", "j crew"],
    "Torrid":                                 ["torrid"],
    "Coach vintage":                          ["coach"],
    "Burberry vintage":                       ["burberry", "burberrys"],
    "Louis Vuitton":                          ["louis vuitton", " lv "],
    # Compound luxury group — expand each brand into its own alias
    "Designer luxury (Gucci/Chanel/Prada/Hermes/YSL/Fendi/Dior/Bottega)": [
        "gucci", "chanel", "prada", "hermes", "hermès",
        "ysl", "saint laurent", "fendi", "dior", "bottega",
    ],
    "Tory Burch / Kate Spade / Michael Kors": [
        "tory burch", "kate spade", "michael kors",
    ],
    "Frye / Dr. Martens / Allen Edmonds / Ariat": [
        "frye", "dr. martens", "doc martens", "dr martens",
        "allen edmonds", "ariat",
    ],
    "Vintage band tees (high-value music)":   [],   # model-only matches
    "Vintage movie/TV/brand promo tees":      [],   # model-only matches
    # Bare " starter " false-matched "NEW STARTER UNKNOWN VEHICLE"
    # (automotive starter motor). Gate to clothing context.
    "Vintage sports/college (Champion/Nike/Starter/Salem)": [
        "champion reverse weave",
        "starter jacket", "starter satin", "starter pullover",
        "starter team", "starter nfl", "starter mlb", "starter nba",
        "salem sportswear",
        "russell athletic", "mitchell & ness", "mitchell and ness",
    ],
    # 'jordan 1' / 'jordan 3' false-matched basketball CARDS where
    # 'JORDAN ... 1' appeared across word-gap proximity (e.g. "1993
    # STADIUM CLUB 1 MICHAEL JORDAN"). Tightened to require shoe-
    # specific suffix (retro / high / low / og) on the Jordan model
    # numbers, so Michael Jordan trading cards / coins / signed jerseys
    # don't false-match.
    "Sneakers (Air Jordan/Yeezy/Nike SB/New Balance)": [
        "air jordan",
        "jordan retro", "jordan 1 retro", "jordan 1 high",
        "jordan 1 low", "jordan 1 mid",
        "jordan 3 retro", "jordan 4 retro", "jordan 5 retro",
        "jordan 6 retro", "jordan 11 retro", "jordan 12 retro",
        "jordan og", "jordan bred", "jordan flu game",
        "jordan chicago", "jordan sneaker", "jordan shoes",
        "yeezy", "nike sb",
        "new balance 9", "new balance 5", "new balance 2002r",
        "new balance 990", "new balance 991", "new balance 992",
        "new balance 993", "new balance 9060", "new balance 550",
    ],
    # Household parts (data/household_parts_bolo.json)
    "Le Creuset":                             ["le creuset", "lecreuset"],
    "Pyrex vintage":                          [
        "pyrex",
        # Pyrex-specific pattern names that almost never appear without
        # context elsewhere — catches collector listings that drop the
        # "Pyrex" word ("Blue Dianthus 443 Cinderella Bowl").
        "blue dianthus", "dianthus 443",
        "lady on the left", "amish butterprint",
        "pumpkin butterprint", "orange butterprint",
        "robin egg gooseberry", "jaj gooseberry",
        "rigopal", "horno argentina",
        "snowflake garland 024", "sandalwood ivy",
        "atomic starburst 575", "atomic eyes chip",
        "ufo sputnik atomic", "compass c1959",
        "nemacolin country club", "nemacolin zodiac",
        "duchess pink gold", "stanley promotional",
        "sage scroll oblong", "pink stems 043",
        "fish dish 1959", "black star casserole",
        "gypsy caravan 4 quart", "gypsy caravan bowl",
        "pyrex 924cm", "pyrex 575-b",
        "cinderella mixing bowl 441", "cinderella mixing bowl 444",
        "mixing bowl set 441-444",
    ],
    "Corelle":                                ["corelle", "corning ware"],
    "Anchor Hocking / Fire-King":             [
        "anchor hocking", "fire-king", "fire king", "fireking",
        "jadeite", "jade-ite", "vitrock",
    ],
    # KitchenAid entry covers stand mixer ATTACHMENTS / BOWLS only —
    # not whole stand mixers (heavy / bulky). Aliases gate on
    # accessory-specific words. Quart-size aliases ('5qt' / '6qt')
    # were removed because they false-matched whole mixer titles
    # like "KitchenAid Stand Mixer 5qt KSM150".
    "KitchenAid": [
        "kitchenaid bowl", "kitchenaid attachment",
        "kitchenaid pasta roller", "kitchenaid food grinder",
        "kitchenaid ice cream maker", "kitchenaid spiralizer",
        "kitchenaid grain mill", "kitchenaid sausage stuffer",
        "kitchenaid juicer attachment", "kitchenaid shaver",
        "kitchenaid burnished", "kitchenaid flat beater",
        "kitchenaid wire whip", "kitchenaid dough hook",
        "kitchen aid attachment", "kitchen aid bowl",
    ],
    # Vitamix entry covers CONTAINERS / TAMPERS / BLADES only —
    # not whole blender bases (heavier).
    "Vitamix": [
        "vitamix container", "vitamix 64oz", "vitamix 48oz",
        "vitamix 32oz", "vitamix wet container", "vitamix dry container",
        "vitamix tamper", "vitamix blade",
        "vitamix aer disc", "vitamix lid",
        "vitamix blade assembly",
        "vita-mix container", "vita mix container",
    ],
    # Cuisinart entry covers REPLACEMENT PARTS only (bowls, blades,
    # discs, lids) — not whole appliances. User preference: no
    # mid-size kitchen appliances. Bare 'cuisinart' would match
    # "Cuisinart Waffle Maker", "Cuisinart Coffee Maker", etc. — those
    # should miss. Aliases gate on part-specific words.
    "Cuisinart food processor": [
        "cuisinart bowl", "cuisinart work bowl",
        "cuisinart blade", "cuisinart s blade",
        "cuisinart disc", "cuisinart slicing disc",
        "cuisinart shredding disc", "cuisinart lid",
        "cuisinart pusher", "cuisinart feed tube",
        "cuisinart dlc-7", "cuisinart dlc-8", "cuisinart dlc-10",
        "cuisinart dfp-14", "cuisinart fp-14", "cuisinart fp-13",
        "cuisinart elite", "cuisinart custom 14",
        "cuisinart food processor parts",
    ],
    "Lodge cast iron":                        ["lodge cast iron", "lodge skillet", "lodge logic", "lodge dutch"],
    "Tupperware vintage":                     ["tupperware"],
    "Wusthof / Henckels / Shun / Global":     [
        "wusthof", "wüsthof", "henckels", "j.a. henckels",
        "shun ", "global g-", "global gs-",
    ],
    # User preference: NO power tools (heavy / awkward to ship). The
    # battery-only aliases gate to specific battery model numbers and
    # 'battery' / 'pack' context — bare 'dewalt' / 'milwaukee' would
    # match drills / saws / sanders that the user doesn't want flagged.
    "Dewalt batteries (OEM only)":            [
        "dewalt battery", "dewalt batteries",
        "dcb200", "dcb201", "dcb203", "dcb204", "dcb205",
        "dcb206", "dcb207", "dcb208", "dcb209",
        "dcb606", "dcb609", "dcb612",
        "20v max battery", "flexvolt battery", "60v max battery",
        "dewalt charger", "dcb107", "dcb115",
    ],
    "Milwaukee batteries (OEM only)":         [
        "milwaukee battery", "milwaukee batteries",
        "m18 battery", "m12 battery", "m18 pack",
        "m18 5.0ah", "m18 8.0ah", "m18 12ah", "m18 high output",
        "m12 6.0ah", "m12 4.0ah", "m12 12.0ah",
        "48-11-1850", "48-11-1860", "48-11-1862",
        "48-11-1865", "48-11-1880", "48-11-2460",
        "48-11-2440", "48-11-2402",
        "milwaukee charger", "48-59-",
    ],
    "Snap-On hand tools":                     [
        "snap-on", "snap on", "snapon",
    ],
    "Marantz / Pioneer / Sansui vintage audio parts": [
        "marantz", "pioneer sx", "sansui",
    ],
    "Singer Featherweight":                   [
        # 'featherweight' alone false-matches "BISSELL Featherweight"
        # vacuum cleaners. Require 'singer' adjacent to disambiguate
        # from BISSELL / vacuum / iron / other lightweight products.
        "singer featherweight", "singer 221", "singer 222",
        "featherweight 221", "featherweight 222",
        "singer 221 sewing", "centennial featherweight",
    ],
    # Fishing tackle (data/fishing_tackle_bolo.json)
    "Heddon":                                 ["heddon"],
    "Creek Chub Bait Co":                     [
        "creek chub", "creek-chub", "creekchub",
    ],
    "Arbogast":                               [
        "arbogast", "fred arbogast", "jitterbug",
    ],
    "South Bend Bait Co":                     [
        "south bend bait", "south-bend bait", "bass-oreno",
        "bass oreno", "pike-oreno", "pike oreno",
    ],
    "Paw Paw Bait Co":                        [
        "paw paw", "paw-paw bait",
    ],
    "Les Davis":                              [
        "les davis", "les-davis",
    ],
    "Beau Mac":                               [
        "beau mac", "beau-mac", "beaumac",
    ],
    "Rapala":                                 ["rapala"],
    "Storm":                                  [
        # 'storm' is generic; require a model word context. Use the
        # iconic models so we don't false-match "perfect storm" or
        # weather references.
        "wiggle wart", "thunderstick", "hot 'n tot", "hot-n-tot",
        "hot n tot", "storm lure", "storm wiggle",
    ],
    "Bagley":                                 [
        "bagley", "balsa b ", "killer b", "kill'r b",
    ],
    "Luhr-Jensen":                            [
        "luhr-jensen", "luhr jensen", "luhrjensen",
        "krocodile spoon", "kwikfish", "kwik fish", "kwik-fish",
        "j-plug", "j plug",
    ],
    "Mann's Bait Company":                    [
        "mann's bait", "manns bait", "mann bait",
        "stretch 5+", "stretch 10+", "stretch 15+",
        "stretch 20+", "stretch 25+", "stretch 30+",
        "1 minus lure", "augusta jelly",
    ],
    # Fishing reels (data/fishing_reels_bolo.json — added 2026-05-05)
    "Shimano fishing reels": [
        "shimano stella", "shimano twin power", "shimano saragosa",
        "shimano sustain", "shimano stradic", "shimano sahara",
        "shimano spheros", "shimano vanford", "shimano vanquish",
        "shimano sienna", "shimano syncopate", "shimano symetre",
        "shimano sedona", "shimano nasci",
        "shimano fx", "shimano ix",
        "shimano curado", "shimano chronarch", "shimano calais",
        "shimano aldebaran", "shimano metanium", "shimano antares",
        "shimano bantam", "shimano slx",
        "shimano tranx", "shimano calcutta",
        "shimano tekota", "shimano talica", "shimano tiagra",
        "shimano tld", "shimano trinidad", "shimano torium",
        "shimano tyrnos", "shimano charter special", "shimano triton",
        # Bare model codes from listings
        "fx4000fc", "fx2000",
        "sc2500fg", "sn4000fg", "ix2000r",
        "c5000xg", "c4000xg", "c3000xg", "c2500hg", "c2000hgs",
        "tek600hga", "tek500hga", "tek700hga",
        "bnt 3906", "bnt3906",
    ],
    "Daiwa fishing reels": [
        "daiwa saltiga", "daiwa steez", "daiwa zillion",
        "daiwa tatula", "daiwa fuego",
        "daiwa bg", "daiwa certate", "daiwa exist",
        "daiwa caldia", "daiwa lexa", "daiwa coastal",
        "daiwa procyon", "daiwa aird", "daiwa crossfire",
        "daiwa regal", "daiwa silvercast", "daiwa goldcast",
        "daiwa underspin", "daiwa sealine", "daiwa saltist",
        "daiwa tanacom", "daiwa seaborg", "daiwa ballistic",
        "daiwa theory", "daiwa capricorn",
        # Bare model codes
        "sc100a", "sc100a-cp", "sc120-cp", "sc200a",
    ],
    "Penn fishing reels": [
        "penn slammer", "penn battle", "penn spinfisher",
        "penn senator", "penn international",
        "penn squall", "penn fathom", "penn torque",
        "penn authority", "penn pursuit", "penn wrath",
        "penn conflict", "penn conquer", "penn clash",
        "penn defiance", "penn captiva",
        "penn jigmaster", "penn levelmatic", "penn peerless",
        "penn surfmaster",
        # Common Penn model codes
        "slaiv7500", "slaiv5500", "slaiv3500",
        "penn 113h", "penn 114h", "penn 115l",
        "penn 9/0", "penn 12/0", "penn 6/0",
        "penn 209", "penn 309", "penn 209m",
    ],
    "Lew's fishing reels": [
        "lew's speed spool", "lew's tournament",
        "lew's pro sp", "lew's pro ti",
        "lew's custom pro", "lew's custom inshore",
        "lew's custom speed",
        "lew's kvd", "lew's hypermag",
        "lew's bb1", "lew's american hero",
        "lew's mach", "lew's inshore speed",
        "lew's xfinity", "lew's laser mg",
        "lew's beta", "lew's wally marshall",
        "team lew's",
        # Bare model codes
        "kvd1xhl", "kvd1hl", "kvd1sh", "kvd1mgl",
        "ci1sh", "xs30a", "xs30b",
    ],
    "Zebco fishing reels": [
        "zebco omega pro", "zebco omega",
        "zebco 33", "zebco 808", "zebco 404",
        "zebco 22", "zebco 11",
        "zebco bullet", "zebco slingshot", "zebco spyn",
        "zebco stinger", "zebco splash", "zebco quantum",
        # Bare Zebco model codes
        "z03pro", "z02pro", "z01pro",
    ],
    "Pflueger fishing reels": [
        "pflueger president", "pflueger patriarch",
        "pflueger trion", "pflueger supreme",
        "pflueger asaro", "pflueger echelon",
        "pflueger lady president", "pflueger monarch",
        "pflueger spincast", "pflueger cetina",
        "pflueger medalist", "pflueger purist", "pflueger akron",
        # Bare Pflueger model codes
        "triosp25b", "triosp35b",
    ],
    "Abu Garcia fishing reels": [
        "abu garcia", "abu-garcia", "abugarcia",
        "ambassadeur",
        # Pure-Fishing era models that may drop "Abu" prefix in titles
        "revo sx", "revo mgx", "revo beast", "revo rocket",
        "revo toro", "revo stx", "revo ike", "revo premier",
        "abu black max", "abu silver max", "abu pro max",
        "max 4 pro", "max4pro",
        "cardinal stx", "cardinal sx",
        "abu orra", "abu vendetta", "abu veritas",
        # Bare codes
        "abulp", "max4pro-c",
    ],
    "Okuma fishing reels": [
        "okuma ceymar", "okuma avenger", "okuma cedros",
        "okuma andros", "okuma magda", "okuma convector",
        "okuma cold water", "okuma coronado", "okuma trio",
        "okuma helios", "okuma calera", "okuma citrix",
        "okuma komodo", "okuma stratus", "okuma rtx",
        "okuma slv", "okuma integrity", "okuma tundra",
        "okuma surf 8k",
        # Bare model codes
        "c-4000a", "ma-15dxt", "ma-20dxt", "av-3000",
    ],
    "KastKing fishing reels": [
        "kastking",
        "royale legend", "royale legend ii", "royale legend iii",
        "royale legend elite",
        # Standalone KastKing model lines (without 'kastking' prefix)
        "kastking centron", "kastking brutus",
        "kastking mega jaws", "kastking speed demon",
        "kastking sharky", "kastking crixus",
        "kastking skeet reese", "kastking bassinator",
        "kastking spartacus", "kastking stealth",
        "kastking summer", "kastking zephyr",
        "kastking eagle", "kastking ireel",
        "kastking saiblade", "kastking verus",
        "kastking kapstan", "kastking triton",
    ],
    "13 Fishing reels": [
        "13 fishing concept", "13 fishing inception",
        "13 fishing origin", "13 fishing defy",
        "13 fishing source", "13 fishing architect",
        "13 fishing audacity", "13 fishing microtech",
        "13 fishing black betty",
        # Bare model codes
        "concept tx2", "concept tx", "tx2-6.8",
        "tx2-7.5", "tx2-8.3",
    ],
    "Quantum fishing reels": [
        "quantum smoke", "quantum energy", "quantum optix",
        "quantum throttle", "quantum tour",
        "quantum iron", "quantum cabo", "quantum boca",
        "quantum kinetic", "quantum vapor",
        "quantum strategy", "quantum reliance",
        "quantum snapshot", "quantum magnum",
    ],
    "Premium fly fishing reels": [
        "hardy perfect", "hardy bougle", "hardy marquis",
        "hardy princess", "hardy featherweight", "hardy lrh",
        "hardy cascapedia", "hardy ultralite", "hardy demon",
        "hardy zephrus",
        "ross evolution", "ross reach", "ross animas",
        "ross san miguel",
        "tibor riptide", "tibor everglades", "tibor gulfstream",
        "tibor pacific", "tibor tail water",
        "tibor backcountry", "tibor spring creek",
        "galvan torque", "galvan brookie", "galvan rush",
        "galvan euro nymph",
        "lamson litespeed", "lamson liquid", "lamson speedster",
        "lamson remix", "lamson guru", "lamson cobalt",
        "sage click", "sage spectrum", "sage trout spey",
        "sage arbor xl", "sage domain",
        "orvis hydros", "orvis mirage", "orvis battenkill",
        "orvis clearwater", "orvis cfo", "orvis access",
        "bauer rx", "bauer sst",
        "abel super", "abel tr", "abel vaya",
        "nautilus ccf", "nautilus nv-g",
    ],
    "Newell vintage saltwater reels": [
        "newell 220", "newell 322", "newell 332",
        "newell 338", "newell 422", "newell 432",
        "newell 446", "newell 454", "newell 533",
        "newell 540", "newell 622", "newell 633",
        "newell 646", "newell 700", "newell 740",
        "newell p220", "newell p322", "newell p332",
        "newell g440", "newell g450",
        "newell clicker",
    ],
    # Printer ink (data/printer_ink_bolo.json — added 2026-05-05)
    # Cartridge SKU-driven matching. The brand prefix ("epson",
    # "hp", "canon", "brother") is the disambiguator since pure
    # SKUs like "67XL" or "PG-275" might collide with other domains.
    "Epson printer ink": [
        "epson 502", "epson t502", "epson 522", "epson t522",
        "epson 542", "epson t542", "epson 552", "epson t552",
        "epson 532", "epson t532", "epson 504", "epson t504",
        "epson 220", "epson 232", "epson 252", "epson 273",
        "epson 312", "epson 410", "epson 524", "epson 588",
        "epson 786", "epson 802", "epson 822", "epson 902",
        "epson 215", "epson 200", "epson 124", "epson 125",
        "epson 126", "epson 127", "epson 78", "epson 88",
        "epson 60", "epson 68", "epson 69", "epson 99",
        # Specific industrial ink lines (kept narrow — bare "epson
        # ecotank" without ink-SKU context routes to printer hardware).
        "epson dg white", "epson dtg white",
        # Bare cartridge codes
        "t725a", "t725b", "t725c", "t725d", "t725e",
    ],
    "HP printer ink": [
        "hp 60", "hp 60xl", "hp 60 xl",
        "hp 61", "hp 61xl", "hp 61 xl",
        "hp 62", "hp 62xl", "hp 62 xl",
        "hp 63", "hp 63xl", "hp 63 xl",
        "hp 64", "hp 64xl",
        "hp 65", "hp 65xl",
        "hp 66", "hp 66xl",
        "hp 67", "hp 67xl", "hp 67 xl",
        "hp 75", "hp 75xl", "hp 78",
        "hp 92", "hp 93", "hp 95", "hp 96", "hp 97", "hp 99", "hp 100",
        "hp 902", "hp 902xl",
        "hp 910", "hp 910xl", "hp 910 xl",
        "hp 915", "hp 915xl",
        "hp 920", "hp 920xl",
        "hp 932", "hp 932xl", "hp 933", "hp 933xl",
        "hp 934", "hp 934xl", "hp 935", "hp 935xl",
        "hp 950", "hp 950xl", "hp 951", "hp 951xl",
        "hp 952", "hp 952xl", "hp 953", "hp 953xl",
        "hp 962", "hp 962xl", "hp 963", "hp 963xl",
        "hp 964", "hp 964xl", "hp 965", "hp 965xl",
        "hp 970", "hp 970xl", "hp 971", "hp 971xl",
        "hp 972", "hp 972xl", "hp 973", "hp 973xl",
        "hp 974", "hp 974xl",
        "hp 980", "hp 980xl", "hp 981", "hp 981xl",
        "hp 982", "hp 982xl", "hp 983", "hp 983xl",
        "hp 564", "hp 564xl", "hp 711", "hp 711xl",
        "hp 727", "hp 728",
        "hp 901", "hp 901xl",
        "hp 940", "hp 940xl",
        "hp designjet ink", "hp latex ink",
        "hp smart tank ink", "hp instant ink",
        "c1q10a", "c1q11a",
    ],
    "Canon printer ink": [
        "canon pg-210", "canon cl-211",
        "canon pg-240", "canon cl-241",
        "canon pg-243", "canon cl-244",
        "canon pg-245", "canon cl-246",
        "canon pg-247",
        "canon pg-275", "canon cl-276", "canon pg-276",
        "canon pgi-220", "canon cli-221",
        "canon pgi-225", "canon cli-226",
        "canon pgi-250", "canon cli-251",
        "canon pgi-270", "canon cli-271",
        "canon pgi-280", "canon cli-281",
        "canon pgi-1200", "canon pgi-2200",
        "canon gi-21", "canon gi-22", "canon gi-23",
        "canon gi-25", "canon gi-26",
        "canon bci-3", "canon bci-6", "canon bci-15", "canon bci-16",
        "canon cli-8", "canon cli-36", "canon cli-42",
        "canon lucia ink", "canon imageprograf ink",
        "canon pfi-300", "canon pfi-1100", "canon pfi-1300",
        "canon pfi-1700",
        # Bare SKU codes (without "canon" prefix)
        "pg-275xl", "cl-276xl", "pg-245xl", "cl-246xl",
        "pgi-280xl", "cli-281xl",
        "pgi-1200xl", "pgi-2200xl",
    ],
    "HP LaserJet toner": [
        # Requested 7/12 — genuine HP laser toner. "A" = standard
        # yield, "X" = high yield; sealed genuine boxes resell $25-150.
        "hp toner", "hp laserjet toner",
        "hp 05a", "hp 12a", "hp 26a", "hp 26x", "hp 30a", "hp 30x",
        "hp 42a", "hp 48a", "hp 55a", "hp 58a", "hp 58x", "hp 64a",
        "hp 78a", "hp 80a", "hp 83a", "hp 85a", "hp 89a", "hp 90a",
        "hp 202a", "hp 202x", "hp 206a", "hp 206x",
        "hp 410a", "hp 410x", "hp 414a", "hp 414x",
        # Bare part numbers (with and without HP prefix on the lot)
        "cf226a", "cf226x", "cf258a", "cf258x", "cf230a", "cf230x",
        "cf248a", "cf280a", "cf280x", "cf283a", "ce285a", "ce255a",
        "cc364a", "q2612a", "cf500a", "cf510a",
        "w2110a", "w2111a", "w2112a", "w2113a",
        "w2020a", "w2021a", "w2022a", "w2023a",
        "cf410a", "cf410x", "cf411a", "cf412a", "cf413a",
    ],
    "Canon toner": [
        # Requested 7/12 — Canon imageCLASS / CRG laser toner.
        "canon toner", "canon imageclass toner",
        "canon 045", "canon 046", "canon 051", "canon 052",
        "canon 054", "canon 055", "canon 055h", "canon 057",
        "canon 057h", "canon 067", "canon 118", "canon 119",
        "canon 121", "canon 128", "canon 131", "canon 137",
        "crg-045", "crg-046", "crg-051", "crg-052", "crg-054",
        "crg-055", "crg-057", "crg-067", "crg-118", "crg-119",
        "crg-121", "crg-128", "crg-131", "crg-137",
    ],
    "Brother printer supplies": [
        "brother lc101", "brother lc103", "brother lc203",
        "brother lc205", "brother lc207", "brother lc209",
        "brother lc401", "brother lc404", "brother lc406",
        "brother lc3017", "brother lc3019",
        "brother lc3029", "brother lc3033",
        "brother lc3035", "brother lc3039",
        "brother lc51", "brother lc61", "brother lc65",
        "brother lc75", "brother lc79",
        "brother btd60", "brother bt5000", "brother bt-d60",
        "brother tn-450", "brother tn-660", "brother tn-630",
        "brother tn-720", "brother tn-730", "brother tn-750",
        "brother tn-760", "brother tn-770",
        "brother tn-820", "brother tn-830", "brother tn-850",
        "brother tn-880", "brother tn-890",
        "brother tn-227", "brother tn-223",
        "brother dr-360", "brother dr-630", "brother dr-720",
        "brother dr-730", "brother dr-820", "brother dr-830",
        "brother dr-223", "brother dr-227",
        # Bare/unhyphenated toner codes (7/12) — estate-lot titles
        # often write "TN760" or lead with the bare code ("Lot: TN-760
        # x3 sealed"). The brother-prefixed hyphen forms above miss
        # those.
        "brother tn760", "brother tn850", "brother tn660",
        "brother tn450", "brother tn730", "brother tn880",
        "tn-450", "tn-660", "tn-730", "tn-760", "tn-770",
        "tn-820", "tn-850", "tn-880", "tn-223", "tn-227",
        "tn450", "tn660", "tn730", "tn760", "tn770",
        "tn820", "tn850", "tn880",
        "dr-420", "dr-630", "dr-730", "dr-820",
        "brother toner", "brother drum unit",
    ],
    "Specialty / industrial inks": [
        "fujifilm dx 100", "fujifilm dx100",
        "fujifilm frontier ink", "fuji minilab ink",
        "dtf ink", "dtf white ink", "dtf black ink",
        "dtf cmykw", "dtg ink", "dtg white ink",
        "direct to garment ink", "direct to film ink",
        "uv led ink", "uv led inks", "uv curable ink",
        "sublimation ink", "sublimation inks",
        "eco-solvent ink", "eco solvent ink",
        "mimaki ink", "mimaki bs3", "mimaki bs4",
        "roland eco-sol", "roland texart",
        "mutoh universal",
        "sawgrass ink", "sublijet hd",
        "cobra ink", "inktec ink", "hilord ink",
    ],
    "Inkjet printers (Epson / HP / Canon)": [
        "epson ecotank et-2400", "epson ecotank et-2800",
        "epson ecotank et-2850", "epson ecotank et-2980",
        "epson ecotank et-3850", "epson ecotank et-4760",
        "epson ecotank et-4800", "epson ecotank et-4850",
        "epson ecotank et-4900", "epson ecotank et-5800",
        "epson ecotank et-5850", "epson ecotank et-15000",
        "epson ecotank et-16500", "epson ecotank et-16650",
        "epson ecotank et-8500", "epson ecotank et-8550",
        "epson workforce wf-2820", "epson workforce wf-2830",
        "epson workforce wf-2860", "epson workforce wf-2870",
        "epson workforce wf-3720", "epson workforce wf-3733",
        "epson workforce wf-7710", "epson workforce wf-7720",
        "epson workforce pro wf-3820",
        "epson wf-3823", "epson wf-4830", "epson wf-4834",
        "epson xp-4100", "epson xp-4105",
        "epson xp-7100", "epson xp-15000",
        "epson surecolor f570", "epson surecolor p700",
        "epson surecolor p900", "epson surecolor p5000",
        "epson surecolor f2100", "epson surecolor f2000",
        "epson surecolor f6300", "epson surecolor f6370",
        "epson surecolor f9470", "epson surecolor f7200",
        "epson colorworks c3500", "epson colorworks c4000u",
        "epson colorworks c7500",
        "hp deskjet 2855e", "hp deskjet 2755e",
        "hp deskjet 4155e", "hp deskjet 4255e",
        "hp deskjet 3755", "hp deskjet plus 4155",
        "hp officejet pro 8020", "hp officejet pro 8025",
        "hp officejet pro 9015", "hp officejet pro 9020",
        "hp officejet pro 9025", "hp officejet pro 9135",
        "hp officejet 250", "hp officejet 200",
        "hp envy pro 6455", "hp envy 6055",
        "hp envy inspire 7220", "hp envy inspire 7958",
        "hp laserjet pro m404", "hp laserjet pro m428",
        "hp smart tank 5101", "hp smart tank 7301",
        "hp designjet t120", "hp designjet t520",
        "hp designjet t630", "hp designjet t650",
        "hp designjet t730",
        "canon pixma tr4720", "canon pixma tr4722",
        "canon pixma ts3520", "canon pixma ts3522",
        "canon pixma ts6420", "canon pixma ts9520",
        "canon pixma ts9521",
        "canon pixma g3260", "canon pixma g6020",
        "canon pixma g7020", "canon maxify gx5020",
        "canon maxify gx6020", "canon maxify gx7020",
        "brother mfc-j895dw", "brother mfc-j995dw",
        "brother mfc-j4335dw", "brother mfc-j4535dw",
        "brother hl-l2350dw", "brother hl-l2370dw",
        "brother hl-l3290cdw", "brother hl-l3270cdw",
        "leibinger jet3up",
        # Specialty hardware that shows under "ink" / "printer" search
        "dtf printer", "dtg printer", "uv printer",
        "uv led printer", "sublimation printer",
        "a3 dtf", "a3 dtg", "a4 dtf", "a4 uv",
        "direct to film printer", "direct to garment printer",
        "eufymake e1",
    ],
    "E-ink tablets / readers": [
        "kindle paperwhite", "kindle oasis", "kindle scribe",
        "kindle voyage",
        "remarkable 2", "remarkable paper pro",
        "kobo libra", "kobo sage", "kobo elipsa", "kobo clara",
        "boox note", "boox tab", "boox go",
        "boox note air", "boox page", "boox palma",
        "ruertu e-ink", "ruertu color e-ink",
        "color e-ink", "e-ink writing tablet",
        "supernote a5", "supernote a6", "supernote a6x",
        "pocketbook era", "pocketbook inkpad",
        "onyx boox",
    ],
    # Appliance parts (data/household_parts_bolo.json — added 2026-05-04)
    # The bare brand name is fine for these — Whirlpool / Maytag /
    # Frigidaire / Amana / Jenn-Air don't sell anything but appliances,
    # so any title containing the brand is plausibly an appliance part.
    "Whirlpool appliance parts":              ["whirlpool"],
    "Maytag appliance parts":                 ["maytag"],
    # KitchenAid appliance parts shares its alias with the existing
    # KitchenAid stand-mixer entry. The matcher's strong-match-wins
    # logic + JSON ordering handle the routing — appliance-specific
    # models (control board, dishwasher rack, etc.) match this entry
    # while stand-mixer-specific models (6qt bowl, pasta roller) match
    # the original KitchenAid entry.
    # Gate on dishwasher / refrigerator / range / control board /
    # ice maker context — bare 'kitchenaid' would false-match whole
    # stand mixers (mid-size, not desired) plus any other product
    # in the broader brand line.
    "KitchenAid appliance parts": [
        "kitchenaid dishwasher", "kitchenaid refrigerator",
        "kitchenaid range", "kitchenaid wall oven",
        "kitchenaid microwave", "kitchenaid control board",
        "kitchenaid ice maker", "kitchenaid water inlet",
        "kitchenaid drain pump", "kitchenaid superba",
        "kitchenaid pro line", "kitchenaid architect",
        "kitchenaid kscs", "kitchenaid kfcs", "kitchenaid krmf",
    ],
    "Amana":                                  ["amana"],
    "Jenn-Air":                               ["jenn-air", "jenn air", "jennair"],
    # GE / LG / Samsung also sell consumer electronics (TVs, phones).
    # Qualify with appliance-context words to avoid false-positives on
    # those product lines.
    "GE Appliances": [
        "ge profile", "ge cafe", "ge café", "ge monogram",
        "monogram refrigerator", "monogram dishwasher",
        "monogram range", "ge appliance", "general electric appliance",
        "ge refrigerator", "ge dishwasher", "ge washer", "ge dryer",
        "ge range", "ge oven", "ge microwave", "ge spacemaker",
        "ge adora",
    ],
    "Frigidaire":                             ["frigidaire"],
    "Bosch appliance parts": [
        "bosch dishwasher", "bosch washer", "bosch dryer",
        "bosch refrigerator", "bosch range", "bosch oven",
        "bosch appliance", "silence plus", "bosch benchmark",
        "bosch ascenta",
    ],
    "LG appliance parts": [
        "lg refrigerator", "lg washer", "lg dryer",
        "lg dishwasher", "lg range", "lg oven", "lg microwave",
        "lg appliance", "lg tromm",
    ],
    "Samsung appliance parts": [
        "samsung refrigerator", "samsung washer", "samsung dryer",
        "samsung dishwasher", "samsung range", "samsung oven",
        "samsung microwave", "samsung appliance",
        "flexzone", "twin cooling", "family hub",
    ],
    # Computer parts (data/computer_parts_bolo.json — added 2026-05-04)
    # CPUs
    "AMD CPUs": [
        "amd ryzen", "ryzen 5", "ryzen 7", "ryzen 9",
        "threadripper", "amd epyc", "amd 5800x", "amd 5900x",
        "5800x3d", "7800x3d", "7950x3d", "9950x",
    ],
    "Intel CPUs": [
        "intel core", "core i3-", "core i5-", "core i7-", "core i9-",
        "core i3 12", "core i5 12", "core i7 12", "core i9 12",
        "core i3 13", "core i5 13", "core i7 13", "core i9 13",
        "core i3 14", "core i5 14", "core i7 14", "core i9 14",
        "intel xeon", "core ultra",
        # Older mainstream desktop CPUs from eBay sold-research
        # Volume movers — bare model numbers in titles like
        # "Intel i5-8500" or "i7-7700 SR338" should fire on these.
        "intel i3", "intel i5", "intel i7", "intel i9",
        "i3-4", "i3-6", "i3-7", "i3-8", "i3-9", "i3-10", "i3-12", "i3-13",
        "i5-4", "i5-6", "i5-7", "i5-8", "i5-9", "i5-10", "i5-11",
        "i7-4", "i7-6", "i7-7", "i7-8", "i7-9", "i7-10", "i7-11",
        # Server-tier (LGA2011-3 / LGA3647)
        "xeon e3", "xeon e5", "xeon e-2",
        "xeon gold", "xeon silver", "xeon bronze", "xeon platinum",
    ],
    # Single-board computers — Raspberry Pi 3 Model B was the #1 + #2
    # best-seller in CPU-category eBay sold-research (past 30 days).
    "Raspberry Pi": [
        "raspberry pi", "rpi", "pi zero", "pi pico",
        "pi 4 model b", "pi 3 model b", "pi 5",
        "compute module 4", "compute module 5",
    ],
    # Intel mini-PCs — discontinued 2023 (now ASUS NUC line). Older
    # NUC models still trade well to homelab buyers.
    "Intel NUC": [
        "intel nuc",
        # Whole-word alias hits — match the model prefix verbatim
        # in titles like "NUC10i3FNH" or "NUC8i7HVK". These share a
        # word with the suffix (no boundary) so we list both with
        # AND without the trailing letters/digits.
        "nuc8i3", "nuc8i5", "nuc8i7", "nuc8i7beh", "nuc8i7hvk",
        "nuc10i3", "nuc10i5", "nuc10i7",
        "nuc10i3fnh", "nuc10i5fnh", "nuc10i7fnh",
        "nuc11i3", "nuc11i5", "nuc11i7",
        "nuc11pahi5", "nuc11pahi7",
        "nuc12i3", "nuc12i5", "nuc12i7",
        "nuc13i3", "nuc13i5", "nuc13i7",
        "hades canyon", "ghost canyon", "phantom canyon",
        "beast canyon", "dragon canyon",
    ],
    # SFF / Mini desktops — corporate off-lease refurbs flowing
    # into homelab / Plex / pfSense / Proxmox builds.
    "SFF mini desktops": [
        "hp prodesk", "hp elitedesk",
        "prodesk 600", "prodesk 400", "prodesk 405",
        "elitedesk 800", "elitedesk 705", "elitedesk 705 g",
        "dell optiplex 3050", "dell optiplex 3060", "dell optiplex 3070",
        "dell optiplex 5050", "dell optiplex 5060", "dell optiplex 5070",
        "dell optiplex 7050", "dell optiplex 7060", "dell optiplex 7070",
        "dell optiplex 7080", "dell optiplex micro",
        "lenovo thinkcentre", "thinkcentre m715q", "thinkcentre m720q",
        "thinkcentre m75q", "thinkcentre m90q", "thinkcentre tiny",
    ],
    # Thermal compound — surprisingly high volume from eBay research
    # (Arctic MX-6 hit 193 units in 30 days). Generic 5-pack syringes
    # also move steadily.
    "PC thermal compound": [
        "arctic mx-", "arctic mx ", "arctic silver",
        "thermal grizzly", "kryonaut", "conductonaut",
        "noctua nt-h", "mastergel",
        "thermal paste", "thermal grease", "thermal compound",
        "heatsink compound",
        "silver thermal", "gold thermal",
    ],
    # NVIDIA Tesla data-center GPUs — DOMINANT volume on eBay sold-
    # research (P100 cleared 500+ units across 5 listings in 30 days).
    "NVIDIA Tesla data-center GPUs": [
        "nvidia tesla", "tesla v100", "tesla p100",
        "tesla p40", "tesla p4", "tesla t4",
        "tesla a100", "tesla a30", "tesla a40", "tesla a10",
        "tesla h100", "tesla h200",
        "tesla l40", "tesla l40s", "tesla l4",
        "tesla m40", "tesla m60", "tesla k80", "tesla k40",
        # NVIDIA part-number prefixes that show up in raw titles
        "699-2g503", "699-2h400", "699-2g610", "699-2g414",
        "699-21x10", "699-21001",
        # Also catch "V100 16GB SXM2 TESLA GPU" word-order variations
        "v100 16gb", "v100 32gb", "v100 sxm2",
        "p100 16gb", "p100 hbm2",
        "p40 24gb", "p4 8gb",
    ],
    "NVIDIA Quadro / RTX workstation GPUs": [
        "nvidia quadro", "quadro p1000", "quadro p2000",
        "quadro p4000", "quadro p5000", "quadro p6000",
        "quadro k420", "quadro k600", "quadro k2200",
        "quadro k4200", "quadro k5200", "quadro m2000",
        "quadro m4000", "quadro m5000", "quadro m6000",
        "quadro rtx 4000", "quadro rtx 5000",
        "quadro rtx 6000", "quadro rtx 8000",
        "rtx a2000", "rtx a4000", "rtx a4500", "rtx a5000",
        "rtx a5500", "rtx a6000",
        "rtx 4000 ada", "rtx 5000 ada", "rtx 6000 ada",
        "rtx 4500 ada", "rtx 5880 ada",
    ],
    "AMD Radeon Pro server GPUs": [
        "radeon pro v620", "radeon pro v340", "radeon pro v520",
        "radeon pro v710", "radeon pro vii",
        "radeon pro wx", "radeon pro w7",
        "radeon pro w6800", "radeon pro w6600",
        "radeon pro w5700", "radeon pro w5500",
        "radeon instinct", "amd instinct",
        "firepro w7100", "firepro w8100",
        "firepro s9150", "firepro s9300",
        # Common AMD part-number prefixes from eBay sold listings
        "102-d60305", "102-d05318", "102-d32302",
    ],
    "NVIDIA CMP mining GPUs": [
        "cmp 170hx", "cmp 90hx", "cmp 50hx",
        "cmp 40hx", "cmp 30hx",
        "p102-100", "p104-100", "p106-100",
        "p102-90", "p104-90", "p106-90",
        "mining edition", "mining gpu",
    ],
    "Older AMD consumer GPUs": [
        "rx 470", "rx 480", "rx 570", "rx 580",
        "rx 590", "rx 5500 xt", "rx 5600 xt",
        "rx 5700", "rx 5700 xt",
        "rx 6500 xt", "rx 6600", "rx 6600 xt",
        "rx 6650 xt", "rx 6700", "rx 6700 xt",
        "rx 6750 xt", "rx 6800", "rx 6800 xt",
        "rx 6900 xt", "rx 6950 xt",
        "radeon hd 7970",
        "r9 290", "r9 290x", "r9 390", "r9 390x",
        "r9 fury", "r9 nano",
    ],
    "Older NVIDIA consumer GPUs": [
        "gtx 1050", "gtx 1060", "gtx 1070", "gtx 1080",
        "gtx 1080 ti", "gtx 1070 ti",
        "gtx 1650", "gtx 1660",
        "rtx 2060", "rtx 2070", "rtx 2080",
        "rtx 3050", "rtx 3060", "rtx 3070",
        "rtx 3080", "rtx 3090",
        "gtx 980", "gtx 980 ti", "gtx 970", "gtx 960", "gtx 950",
        "gtx 750", "gtx 750 ti", "gtx 760",
        "gtx 770", "gtx 780", "gtx 780 ti",
        "titan xp", "titan v", "titan rtx",
        "titan x pascal", "nvidia titan",
    ],
    "GPU power cables / accessories": [
        "12vhpwr", "12v-2x6", "12v 2x6",
        "12vhpwr cable", "12vhpwr adapter",
        "16-pin power adapter", "16-pin power cable",
        "8-pin to 12-pin", "8-pin to 16-pin",
        "8-pin to dual 8", "8-pin to dual 8-pin",
        "pcie 5.0 power", "pcie 5.0 12vhpwr",
        "8 pin gpu", "6+2 pin gpu",
        "modular psu gpu", "supermicro gpu cable",
        "cbl-pwex",
    ],
    "Mini-ITX motherboards": [
        "mini-itx", "mini itx", "itx motherboard",
        "celeron j1900", "celeron j3160", "celeron j4125",
        "celeron n5105", "atom n270", "atom n450",
        "ga-j1900n", "ga-n3150n", "ga-n3160n",
        "asrock j5040", "asrock j4125", "asrock n100m",
        "asus h110i", "asus b660i",
        "b450 i aorus", "b550i aorus", "b550m-itx",
        "x570 phantom gaming-itx", "b650e pg-itx",
    ],
    # HP enterprise PSUs — corporate-refurb staples (DPS-XXXAB,
    # HSTNS-PD, HP Z-workstation chassis fits, ProDesk/EliteDesk
    # SFF PSUs). Listed brand prefixes match how eBay sellers
    # write the part numbers verbatim.
    "HP enterprise PSUs": [
        "dps-750rb", "dps-700ab",
        "dps-1125ab", "dps-1200fb",
        "hstns-pd18", "hstns-pd05", "hstns-pd30", "hstns-pd41",
        "hp 506822", "hp 511778", "hp 511777",
        "hp 660185", "hp 660183",
        "hp 754381", "hp 758467",
        "hp 796348", "hp 796418", "hp 702309", "hp 751886",
        "hp 809053", "hp 860477", "hp 719795",
        "l05757", "l77487", "l89233",
        "z230 psu", "z240 psu", "z420 psu", "z440 psu",
        "z620 psu", "z640 psu", "z820 psu", "z840 psu",
        "prodesk 600 psu", "elitedesk 800 psu",
        "envy 795 psu", "hp 280 g8 psu",
        "pro z2 psu", "te01 psu", "tg01 psu",
    ],
    # Dell enterprise PSUs — OptiPlex SFF, Precision workstation,
    # PowerEdge rack-server PSUs.
    "Dell enterprise PSUs": [
        "h240as-00", "h180as-00", "h255as-00",
        "l255as-00", "h320as-00", "h460am",
        "n750e-s0", "l685em-00", "l685em",
        "h750ef-00", "h750em-00",
        # Dell part-number prefixes from eBay sold listings
        "2xk8w", "3xrjo", "fp16x", "h1fwx",
        "nt1xp", "hxrpx", "3wn11", "2txym",
        "rv1c4", "05xv5k", "8tvyy",
        "04fhyw", "033vt", "565yr",
        # PowerEdge rack server PSUs — eBay sellers often write the
        # model number standalone ("DELL R740 PSU") without the
        # "PowerEdge" prefix.
        "dell r720", "dell r730", "dell r740", "dell r740xd",
        "dell r830", "dell r840", "dell r930", "dell r940",
        "poweredge r720", "poweredge r730", "poweredge r740",
        "poweredge r830", "poweredge r840", "poweredge r930",
        "poweredge r940",
        # Precision tower workstation PSUs
        "precision t5810", "precision t7810",
        "precision t7910", "precision t3640",
        "dell precision t5810", "dell precision t7810",
    ],
    "SuperMicro PSUs": [
        "supermicro psu", "supermicro power supply",
        "pws-203-1h", "pws-280-1h", "pws-350-1h",
        "pws-500-1h", "pws-600-1h", "pws-700-1h", "pws-865-1h",
        "pws-1k24a-1r", "pws-1k28a-1r", "pws-2k04a-1r",
        "pws-3k06g-2r", "pws-3k01f-1r", "pws-2k01f-1r",
        "5018a-ftn4",
    ],
    # Mid-tier consumer PSU brands not covered by Corsair / EVGA /
    # Seasonic / be quiet entries.
    "Mid-tier consumer ATX PSUs": [
        "msi mag a550bn", "msi mag a650bn", "msi mag a750bn",
        "msi mag a550gl", "msi mag a650gl", "msi mag a750gl", "msi mag a850gl",
        "msi mpg a1000g", "msi mpg a850g", "msi mpg a750g",
        "msi a1000g", "msi a850gl",
        "gp-p450b", "gp-p550b", "gp-p650b",
        "gp-p750gm", "gp-p850gm", "gp-p1000gm",
        "gigabyte aorus p850w", "gigabyte aorus p1000w",
        "gigabyte aorus p1200w",
        "gigabyte ud850gm", "gigabyte ud1000gm",
        "cooler master mwe", "cooler master v850",
        "cooler master v1300", "cooler master m2 silent",
        "thermaltake smart series", "thermaltake smart 500w",
        "thermaltake smart 600w", "thermaltake smart 700w",
        "thermaltake smart 750w",
        "thermaltake smart pro", "thermaltake toughpower gf",
        "thermaltake toughpower",
        "super flower leadex",
        "segotep gaming", "segotep psu",
        "fsp hydro g pro", "fsp hydro ptm pro", "fsp dagger pro",
        "fsp270-60le", "fsp210-20tgbab",
    ],
    "Apple legacy PSUs": [
        "apple imac psu", "imac power supply",
        "a1419 psu", "a2115 psu", "a1311 psu", "a1312 psu", "a1418 psu",
        "mac pro 980w", "mac pro 1,1 psu", "mac pro 2,1 psu",
        "apple thunderbolt display psu", "apple cinema display psu",
        "a1407 psu", "apple a1407",
        # Apple-specific Delta PSU model + Apple service-part numbers
        "dps-980ab", "614-0383", "614-0436", "614-0454",
        # Catch-all phrasings the eBay sellers use
        "apple 980w", "apple imac",
    ],
    # Premium custom water cooling — EKWB / Alphacool / XSPC /
    # Hardware Labs / Bitspower / NZXT Kraken / Lian Li Galahad.
    "Premium custom water cooling": [
        "ekwb", "ek water blocks", "ek-quantum",
        "ek quantum surface", "ek quantum velocity",
        "ek quantum magnitude", "ek quantum vector",
        "ek quantum reflection", "ek quantum kinetic",
        "ek-quantum surface", "ek-quantum velocity",
        "ek-quantum magnitude", "ek-quantum vector",
        "ek-quantum reflection", "ek-quantum kinetic",
        "ek-quantum convection", "ek-quantum inertia",
        "ek-quantum power",
        "ek blitz", "ek-coolstream", "ek-loop",
        "ek-tube", "ek-torque", "ek-furious",
        "ek-vardar evo",
        "alphacool eisstation", "alphacool eisbecher",
        "alphacool eisbaer", "alphacool eiswolf",
        "alphacool nexxxos", "alphacool nexxos",
        "alphacool st30", "alphacool xt45", "alphacool ut60",
        "alphacool vpp755", "alphacool vpp solo",
        "alphacool dc-lt", "alphacool d5",
        "alphacool es bay",
        "xspc raystorm", "xspc rx240", "xspc rx360",
        "xspc rx480", "xspc ex240", "xspc ex360",
        "xspc tx240", "xspc tx360",
        "xspc drive d5", "xspc d5t", "xspc photon",
        "xspc bay reservoir",
        "hardware labs black ice", "black ice nemesis",
        "black ice gts", "black ice gtx", "black ice stealth",
        "black ice sr2", "black ice sr1",
        "bitspower touchaqua", "bitspower premium",
        "bitspower mod-reservoir",
        "watercool heatkiller", "heatkiller iv",
        "heatkiller tube", "heatkiller mobo",
        "phanteks glacier", "phanteks glacier one",
        "bykski water block", "bykski radiator",
        "bykski reservoir",
        "singularity computers",
        "aquacomputer aqualis", "aquacomputer aquaero",
        "aquacomputer octo", "aquacomputer ultitube",
        "nzxt kraken x53", "nzxt kraken x63", "nzxt kraken x73",
        "nzxt kraken z53", "nzxt kraken z63", "nzxt kraken z73",
        "nzxt kraken elite", "nzxt kraken 240", "nzxt kraken 280",
        "nzxt kraken 360", "nzxt kraken 420",
        "lian li galahad", "lian li galahad ii",
        "lian li galahad ii trinity", "lian li galahad ii performance",
        "deepcool lt720", "deepcool ls720", "deepcool ls520",
        "deepcool lt520",
        "id-cooling frozn", "id-cooling as500",
        "thermalright frozen notte", "thermalright frozen edge",
        "thermalright frozen magic",
    ],
    # Budget water cooling components — Chinese OEM.
    "Budget water cooling parts (FreezeMod / Barrow / etc.)": [
        "freezemod",
        "barrow rgbs", "barrow water block", "barrow waterblock",
        "barrow reservoir", "barrow fitting",
        "barrow pvc", "barrow soft tube",
        "barrow cpu block", "barrow gpu block",
        "barrow distro plate", "barrow hdm2280",
        "bykski radiator", "bykski fitting",
        "bykski reservoir", "bykski distro plate",
        "dracaena radiator", "dracaena pc radiator",
        "ococoo radiator", "ococoo pc",
        "ocool nexxos",
        "g1/4 fitting", "g1/4 thread fitting",
        "g1/4 compression fitting", "g1/4 stop plug",
        "g1/4 barb connector", "g1/4 two touch",
        "g1/4 water pump", "g1/4 thread water pump",
        "high flow g1/4",
        "soft pvc tube", "soft tubing fitting",
        "acrylic rigid tube", "petg tubing",
        "dc12v water pump", "12v dc water pump",
        "sc-300t", "sc-600t", "sc-450",
        "ddc pump",
    ],
    # Server / enterprise replacement fans — corporate refurb supply.
    # Distinct from generic PC case fans — proprietary hot-swap form
    # factor only fits the matching server / network-gear chassis.
    "Server / enterprise replacement fans": [
        "poweredge r750 fan", "poweredge r760 fan",
        "poweredge r7525 fan", "poweredge r7625 fan",
        "poweredge r740 fan", "poweredge r740xd fan",
        "poweredge r730 fan", "poweredge r720 fan",
        "poweredge r650 fan", "poweredge r660 fan",
        "poweredge r840 fan", "poweredge r940 fan",
        # Dell fan part-numbers from sold listings
        "2nd0r", "0y4f46", "0wc8n4", "03rkjc", "0rv5m3",
        "n5t36",
        # HPE server fans
        "proliant dl3x0", "proliant dl360 fan", "proliant dl380 fan",
        "proliant dl385 fan", "proliant dl580 fan", "proliant dl560 fan",
        "hpe gen10 fan", "hpe gen11 fan", "hpe gen12 fan",
        "p48908-b21", "p48908b21",
        # HP workstation fans
        "hp z4 g4 fan", "hp z6 g4 fan", "hp z8 g4 fan",
        "hp z440 fan", "hp z640 fan", "hp z840 fan",
        "927570-002", "927570-001", "1xm34aa",
        # Supermicro server fans
        "supermicro fan-0100l4", "supermicro fan-0104l4",
        "supermicro fan-0124l4", "supermicro fan-0144l4",
        "supermicro fan-0166l4", "fan-0100l4", "fan-0104l4",
        # Networking gear fans
        "arista 7280r fan", "arista 7050x fan", "arista 7060 fan",
        "arista 7280r", "arista 7050x", "arista 7060",
        "ex4400-fan-afo", "juniper ex4400 fan", "juniper ex4600 fan",
        "juniper qfx5100 fan", "juniper qfx5200 fan",
        "cisco nexus fan tray", "cisco catalyst fan tray",
    ],
    # Premium PC case fans — RGB / LCD / daisy-chain ecosystems.
    "Premium PC case fans": [
        "darkrock f120", "darkrock f140", "darkrock infinite mirror",
        "lian li uni fan", "lian li uni fan tl",
        "lian li uni fan sl", "lian li uni fan al",
        "lian li uni fan p28", "lian li tl wireless",
        "lian li tl lcd", "12tllcd",
        "phanteks t30-120", "phanteks t30 120",
        "phanteks m25", "phanteks d30",
        "phanteks halos lux", "phanteks eclipse fan",
        "icue link qx120", "icue link qx140",
        "icue link rx120", "icue link rx140",
        "icue link lx120", "icue link lx140",
        "corsair ml120", "corsair ml140", "corsair ml pro",
        "corsair ll120", "corsair ll140",
        "corsair ql120", "corsair ql140",
        "corsair af120", "corsair af140", "corsair af elite",
        "corsair sp120", "corsair sp140",
        "thermaltake riing", "thermaltake ct120", "thermaltake ct140",
        "thermaltake swafan", "swafan ex",
        "ek-vardar", "ek-loop fan", "ek quantum fan",
        "masterfan", "sickleflow",
        "cooler master mf120", "cooler master mf140",
        "cooler master halo", "cooler master mobius",
        "deepcool fk120", "deepcool fc120", "deepcool ls520",
        "deepcool cf120", "deepcool tf120",
        "msi silent gale",
        "rog strix xf120", "rog strix lc fan",
        "gamdias aeolus", "gamdias cyclops",
        "inwin saturn", "inwin mercury", "inwin polaris",
        # Generic ARGB / hub aliases
        "argb fan hub", "argb fan controller",
        "infinite mirror", "infinite mirror design",
        "rgb fan 6 pack", "rgb fan 6-pack",
    ],
    # EBM Papst industrial — German premium fans for inverters,
    # servo motors, AC drives.
    "EBM Papst industrial fans": [
        "ebm papst", "ebmpapst", "ebm-papst",
        "papst k3g250", "papst k3g220", "papst k3g310",
        "papst r3g225", "papst r3g310", "papst r3g450",
        "papst r4d450", "papst r4d560",
        "papst w2d250", "papst w2d210", "papst w2d225",
        "papst w2e250", "papst w2e142",
        "papst d2d146", "papst d3g220",
        "papst g2e140", "papst g3g250",
        # Common bare model-code prefixes (no "Papst" word)
        "k3g250-rr17", "k3g250-rr",
        "r4d450-ak03", "r4d450-ak",
        "w2d250-ga04", "w2d210-eb10", "w2d225-ea18",
        "r3g225-re07", "d2d146-bg03",
    ],
    # Industrial commodity fans — Delta / Sunon / Minebea / NMB / Vantec.
    "Industrial / commodity AC fans (Delta / Sunon / Minebea)": [
        "delta pfb1224", "delta pfb1212", "delta pfb1248",
        "delta thb1748", "delta thb1648", "delta thb1448",
        "delta afb1212", "delta afb1224",
        "delta afc1212", "delta afc1224",
        "delta tfb1224", "delta tfc1248", "delta tfb1212",
        "delta qfr1212", "delta qfr1248",
        # Bare Delta model codes
        "pfb1224uhec", "pfb1224uhec8x",
        "thb1748bg", "thb1648bg", "thb1448bg",
        "sunon dp200a", "sunon sp100a", "sunon sp101a",
        "sunon dp201a", "sunon dp203a",
        "sunon me60252", "sunon ee92252", "sunon ee80252",
        "sunon ha40201", "sunon mf50101",
        "sunon psd1208", "sunon psd1212",
        # Bare Sunon model codes
        "dp200a-2123", "sp100a-1123",
        "minebea 2410ml", "minebea 2410sl", "minebea 3110kl",
        "minebea 4710kl", "minebea 4715kl",
        "vantec sf8025", "vantec stealth", "vantec tf7025",
        "nmb-mat", "nmb 4715", "nmb 3610",
        "sanyo denki 9g", "sanyo denki 109",
    ],
    "Specialty PSUs (Pico / Flex ATX / mining)": [
        "pico atx", "pico psu", "pico-psu",
        "pico 12v", "pico atx switch",
        "flex atx psu", "flex atx power", "flex atx 24-pin",
        "fsp270-60le", "fsp300-60ghs",
        "mining breakout", "gpu breakout board", "breakout board 16 port",
        "parallel miner zsx", "parallel miner x7", "parallel miner x11",
        "zsx-amp", "16 port server psu",
        "mining kit", "server psu mining",
        "compaq dps-1200fb",
    ],
    # GPUs + motherboards (cross-category brands)
    "ASUS PC parts": [
        "asus rog", "asus tuf", "asus prime", "asus proart",
        "asus dual", "asus phoenix", "asus geforce", "asus radeon",
        "rog strix", "rog maximus", "rog crosshair", "tuf gaming",
    ],
    "MSI PC parts": [
        "msi gaming", "msi suprim", "msi ventus",
        "msi geforce", "msi radeon",
        "msi meg", "msi mpg", "msi mag", "msi pro",
        "tomahawk msi", "msi tomahawk", "msi carbon",
    ],
    "Gigabyte PC parts": [
        "gigabyte aorus", "gigabyte gaming", "gigabyte eagle",
        "gigabyte windforce", "gigabyte vision",
        "gigabyte geforce", "gigabyte radeon",
        "aorus master", "aorus elite", "aorus ultra", "aorus xtreme",
    ],
    # Single-domain GPU brands
    "ASRock motherboards": [
        "asrock taichi", "asrock phantom gaming", "asrock steel legend",
        "asrock pro", "asrock mainboard", "asrock motherboard",
        # AM4 / AM5 / Intel chipset variants — common eBay title patterns
        "asrock b550", "asrock b450", "asrock b650",
        "asrock x570", "asrock x670", "asrock x670e",
        "asrock z690", "asrock z790", "asrock z890",
        "asrock h510", "asrock h610", "asrock h670", "asrock h770",
        "asrock b660", "asrock b760",
        "asrock pro4", "asrock pro rs",
        "asrock itx", "asrock mini-itx",
    ],
    "Zotac GPUs": [
        "zotac amp", "zotac trinity", "zotac twin edge",
        "zotac gaming", "zotac geforce", "zotac rtx",
    ],
    "EVGA PC parts": [
        "evga ftw", "evga xc", "evga sc", "evga kingpin",
        "evga supernova", "evga geforce", "evga gtx", "evga rtx",
        "evga psu",
    ],
    "PNY": [
        "pny xlr8", "pny verto", "pny epic-x",
        "pny geforce", "pny rtx", "pny quadro",
        "pny cs3", "pny ssd",
    ],
    "Sapphire Radeon GPUs": [
        "sapphire nitro", "sapphire pulse", "sapphire pure",
        "sapphire toxic", "sapphire vapor-x", "sapphire radeon",
    ],
    "XFX Radeon GPUs": [
        "xfx speedster", "xfx merc", "xfx qick", "xfx swft",
        "xfx radeon", "xfx zero",
    ],
    "PowerColor Radeon GPUs": [
        "powercolor red devil", "powercolor liquid devil",
        "powercolor hellhound", "powercolor fighter",
        "powercolor reaper", "powercolor radeon",
    ],
    # RAM / SSD / HDD / PSU / Case / Cooler — bare brand aliases when
    # the brand only makes PC parts. Multi-product brands (Corsair,
    # Crucial, Kingston) get bare aliases too because their non-PC
    # presence is minimal.
    "Corsair PC components": [
        "corsair vengeance", "corsair dominator",
        "corsair rm", "corsair hx", "corsair ax", "corsair cx",
        # PSU model+wattage forms ("Corsair RM1000e" etc.) — needed
        # because "corsair rm" alone won't whole-word-match "RM1000e"
        # (the digit 1 right after 'm' breaks the trailing boundary).
        "corsair rm550", "corsair rm650", "corsair rm750",
        "corsair rm850", "corsair rm1000",
        "corsair rm550e", "corsair rm650e", "corsair rm750e",
        "corsair rm850e", "corsair rm1000e",
        "corsair rm550x", "corsair rm650x", "corsair rm750x",
        "corsair rm850x", "corsair rm1000x", "corsair rm1200x",
        "corsair hx750", "corsair hx850", "corsair hx1000",
        "corsair hx1200", "corsair hx1500", "corsair hx1500i",
        "corsair cx450", "corsair cx550", "corsair cx650",
        "corsair cx750", "corsair cx850",
        "corsair cx550m", "corsair cx650m", "corsair cx750m",
        "corsair cx850m",
        "corsair sf450", "corsair sf600", "corsair sf750",
        "corsair sf850",
        "corsair icue", "corsair crystal", "corsair obsidian",
        "corsair h100", "corsair h115", "corsair h150", "corsair h170",
        "corsair hydro",
    ],
    "G.Skill RAM": [
        "g.skill", "gskill", "g skill",
        "trident z", "ripjaws", "f4-3200", "f4-3600", "f5-6000",
        "f5-7200", "f5-7600",
    ],
    "Crucial": [
        "crucial ballistix", "crucial mx500", "crucial p3", "crucial p5",
        "crucial t700", "crucial t705", "crucial pro ddr",
        "crucial ddr5", "crucial ddr4",
    ],
    "Kingston": [
        "kingston fury", "kingston hyperx", "kingston kc3000",
        "kingston nv2", "kingston nv3", "kingston valueram",
        "kingston dc600", "kingston a2000",
    ],
    "TeamGroup": [
        "teamgroup", "team group", "t-force", "t force",
    ],
    "Samsung SSDs": [
        "samsung 980", "samsung 990", "samsung 970",
        "samsung 870", "samsung evo", "samsung qvo",
        "samsung pro plus", "samsung t7", "samsung t9", "samsung t5",
        "pm9a1", "pm9a3",
    ],
    "WD storage": [
        "wd black", "wd blue", "wd red", "wd purple", "wd gold",
        "western digital",
        "wd elements", "wd easystore", "wd my passport", "wd my book",
        "wd ultrastar", "sn850x", "sn770", "sn570", "sn550",
    ],
    "Seagate": [
        "seagate ironwolf", "seagate exos", "seagate barracuda",
        "seagate firecuda", "seagate skyhawk",
        "seagate backup plus", "seagate expansion", "seagate one touch",
    ],
    "Toshiba storage": [
        "toshiba mg", "toshiba n300", "toshiba p300",
        "toshiba canvio", "toshiba rc500", "toshiba rd400",
        "toshiba enterprise",
    ],
    "Sabrent": [
        "sabrent rocket", "sabrent",
    ],
    "Seasonic PSUs": [
        "seasonic prime", "seasonic focus", "seasonic vertex",
        "seasonic snow silent", "seasonic", "seasonic platinum",
        "seasonic titanium",
    ],
    "be quiet!": [
        "be quiet", "bequiet", "dark power", "dark rock",
        "pure power", "straight power", "silent loop",
    ],
    "Fractal Design Cases": [
        "fractal design", "fractal define", "fractal torrent",
        "fractal meshify", "fractal north", "fractal pop",
    ],
    "Lian Li Cases": [
        "lian li", "lian-li",
        "o11 dynamic", "lancool", "dan a4-sfx", "dan-a4",
    ],
    "NZXT Cases": [
        "nzxt h7", "nzxt h6", "nzxt h5", "nzxt h1", "nzxt h510",
        "nzxt phantom", "nzxt source", "nzxt kraken",
        "nzxt cam",
    ],
    "Cooler Master Cases": [
        "cooler master", "coolermaster",
        "mastercase", "haf 700", "haf 922", "haf x",
        "cosmos c700", "nr200",
    ],
    "Noctua": ["noctua"],
    "Thermalright": [
        "thermalright peerless", "thermalright phantom",
        "thermalright frost", "thermalright le grand",
        "thermalright true spirit", "thermalright silver arrow",
        "thermalright assassin",
    ],
    "Arctic Cooling": [
        "arctic liquid freezer", "arctic freezer 34", "arctic freezer 36",
        "arctic freezer 7", "arctic bionix",
        # Arctic case-fan model variants — eBay sellers write
        # "ARCTIC P12 5 Pack" without "PWM"
        "arctic p8", "arctic p12", "arctic p14",
        "arctic p12 pwm", "arctic p14 pwm",
        "arctic p12 pst", "arctic p14 pst",
        "arctic p12 slim", "arctic p14 slim",
        "arctic p12 pro", "arctic p14 pro",
        "arctic p8 slim",
        "acfan00295a", "acfan00319a",
    ],
    # Apple products (data/apple_products_bolo.json — added 2026-05-04)
    # Aliases are DELIBERATELY SPECIFIC. Bare 'apple', 'apple tv', 'mac',
    # 'iphone' are NEVER aliases — they'd match dead-tier products.
    # Each alias requires a generation/chip/series marker.
    "Apple Mac": [
        # MacBook Pro (Apple Silicon only — Intel = lower resale)
        "macbook pro m1", "macbook pro m2", "macbook pro m3", "macbook pro m4",
        "macbook pro 14", "macbook pro 16",
        # MacBook Air (Apple Silicon only)
        "macbook air m1", "macbook air m2", "macbook air m3", "macbook air m4",
        # Reverse word-order variants — eBay sellers often write
        # "Apple M1 MacBook Air" instead of "MacBook Air M1"
        "m1 macbook air", "m2 macbook air", "m3 macbook air", "m4 macbook air",
        "m1 macbook pro", "m2 macbook pro", "m3 macbook pro", "m4 macbook pro",
        # Apple model identifiers — distinctive A-codes from MacBook bottoms
        "a2179", "a2337", "a2338", "a3113",
        "a2442", "a2485", "a2779", "a2918", "a2991", "a2992",
        # 12" Retina MacBook (cult following)
        "macbook 12 retina",
        # Mac mini (Apple Silicon only)
        "mac mini m1", "mac mini m2", "mac mini m4",
        # Mac Studio (always premium)
        "mac studio",
        # iMac (M-series + last 5K Intel)
        "imac m1", "imac m3", "imac m4",
        "imac 27 5k", "imac 27 retina", "imac pro",
        # Mac Pro (cylinder + cheese-grater + M-series)
        "mac pro 2013", "mac pro 2019", "mac pro 2023",
        "mac pro m2", "mac pro cylinder", "mac pro cheese grater",
    ],
    "Apple iPhone": [
        # iPhone 12+ only (older = skip-listed)
        "iphone 12", "iphone 13", "iphone 14", "iphone 15",
        "iphone 16", "iphone 17", "iphone 18",
        "iphone se 2nd", "iphone se 3rd", "iphone se 2022",
    ],
    "Apple iPad": [
        # iPad Pro (always Pro tier)
        "ipad pro 11", "ipad pro 12.9", "ipad pro 13",
        "ipad pro m1", "ipad pro m2", "ipad pro m4",
        # iPad Air (4+ only — earlier gens were weaker)
        "ipad air 4", "ipad air 5", "ipad air 6", "ipad air m2",
        # iPad mini (6+ only — 1-4 are skip-listed)
        "ipad mini 6", "ipad mini 7", "ipad mini a17",
        # Standard iPad (9+ only)
        "ipad 9th gen", "ipad 10th gen", "ipad 11th gen",
        # Accessories
        "magic keyboard for ipad", "apple pencil 2",
        "apple pencil pro", "apple pencil usb-c",
    ],
    "Apple Watch (modern)": [
        # Series 7+ only (1-3 are skip-listed; 4-6 weaker, omit)
        "apple watch series 7", "apple watch series 8",
        "apple watch series 9", "apple watch series 10",
        "apple watch series 11",
        # Ultra line (always premium)
        "apple watch ultra",
        # SE 2nd gen (current SE)
        "apple watch se 2",
    ],
    "Apple AirPods Pro / Max": [
        "airpods pro 2", "airpods pro 3",
        "airpods max", "airpods 4 anc",
    ],
    # Apple TV — DELIBERATELY EXCLUDES BARE 'apple tv' to skip 2nd/3rd
    # gen units that have no app support (the user's specific pain
    # point). Only 4K and HD/4th-gen aliases match.
    "Apple TV 4K / HD": [
        "apple tv 4k", "apple tv hd",
        "apple tv 4th gen", "apple tv 5th gen",
        "apple tv 6th gen", "apple tv 7th gen",
        "apple tv a1842", "apple tv a1625",
        "apple tv a2169", "apple tv a2737",
    ],
    "Apple HomePod (modern)": [
        # 2nd gen + mini only (1st gen skip-listed)
        "homepod 2nd gen", "homepod second gen",
        "homepod mini", "homepod 2023",
    ],
    "Apple Display": [
        "apple studio display", "studio display nano",
        "pro display xdr", "apple pro display",
        "apple pro stand",
    ],
    "Apple Magic accessories": [
        "magic keyboard with touch id",
        "magic keyboard with numeric",
        "magic trackpad",
        "magic mouse",
    ],
    # Golf equipment (data/golf_equipment_bolo.json — added 2026-05-04)
    "Callaway Golf": [
        "callaway", "callaway golf", "callaway pre-owned",
        "callaway preowned", "odyssey putter", "odyssey white hot",
        "odyssey 2-ball", "odyssey ai-one",
    ],
    "Ping Golf": [
        # Bare 'ping' is risky (could match unrelated noun usages),
        # so qualify with golf context. The trending data shows
        # "ping" alone appearing — but in HiBid auction titles
        # they'd typically write "Ping golf clubs" or "Ping G430"
        "ping golf", "ping g430", "ping g425", "ping g410",
        "ping g400", "ping g30", "ping g25",
        "ping i230", "ping i525", "ping i59", "ping i210", "ping i200",
        "ping anser", "ping blueprint", "ping glide", "ping hoofer",
        "ping driver", "ping iron",
    ],
    "TaylorMade": [
        "taylormade", "taylor made",
        "tm stealth", "tm qi10", "tm spider",
    ],
    "Titleist": [
        "titleist", "vokey sm9", "vokey sm10", "vokey wedge",
        "pro v1", "pro v1x",
    ],
    "Mizuno Golf": [
        "mizuno jpx", "mizuno mp", "mizuno pro 2",
        "mizuno t22", "mizuno t20", "mizuno st",
        "mizuno irons", "mizuno golf", "mizuno m.craft",
    ],
    "Cobra Golf": [
        "cobra darkspeed", "cobra aerojet", "cobra ltdx",
        "cobra radspeed", "cobra king forged",
        "cobra speedzone", "cobra golf",
    ],
    "Cleveland Golf": [
        "cleveland golf", "cleveland rtx", "cleveland cbx",
        "cleveland zipcore", "cleveland smart sole",
        "cleveland huntington beach", "cleveland frontline",
    ],
    "Srixon": [
        "srixon", "srixon zx5", "srixon zx7",
        "srixon z-star", "srixon q-star",
    ],
    "Scotty Cameron Putters": [
        "scotty cameron", "newport 2", "phantom x",
        "circle t putter", "tour only cameron",
    ],
    "Bettinardi Putters": [
        "bettinardi", "queen b", "studio stock putter",
        "bb1 putter", "bb8 putter", "inovai putter",
    ],
    "Stix Golf": [
        "stix golf", "stix stride", "stix beginner set",
    ],
    "Takomo": [
        "takomo", "takomo 101", "takomo 201", "takomo 301",
    ],
    "Sun Mountain Golf Bags": [
        "sun mountain", "c-130 bag", "clubglider",
        "h2no tour", "sun mountain golf",
    ],
    # Audio (data/audio_watches_bolo.json — added 2026-05-04)
    # Sony's bare brand alias is too broad (TVs, phones, cameras), so
    # gate on the headphone-specific WH/WF/MDR series prefixes.
    "Sony noise-canceling headphones": [
        "sony wh-1000xm", "sony wh1000xm",
        "sony wh-ch", "sony wf-1000xm", "sony wf1000xm",
        "sony linkbuds", "sony ult wear",
        "sony mdr-7506", "sony mdr-v6",
        "sony noise canceling", "sony noise-canceling",
        "wh-1000xm5", "wh-1000xm4", "wh-1000xm3", "wh-1000xm6",
        "wf-1000xm5", "wf-1000xm4",
    ],
    "Bose noise-canceling": [
        "bose quietcomfort", "bose qc ultra", "bose qc 45",
        "bose qc 35", "bose qc 25", "bose qc earbuds",
        "bose noise cancelling headphones 700",
        "bose 700 headphones",
    ],
    "Sennheiser premium audio": [
        "sennheiser hd",
        "sennheiser momentum",
        "sennheiser ie 900", "sennheiser ie 600", "sennheiser ie 200",
        "sennheiser accentum",
        "hd 800s", "hd 660s", "hd 650", "hd 600", "hd 25",
    ],
    "Audio-Technica": [
        "audio-technica", "audio technica",
        "ath-m50x", "ath-m40x", "ath-r70x", "ath-ad",
        "ath-w5000", "ath-wp900",
        "at-lp120", "at-lp140", "at-lp3",
    ],
    "Beats by Dr. Dre": [
        "beats studio", "beats solo", "beats pro",
        "powerbeats", "beats fit pro", "beats studio buds",
        "beats pill", "beats by dre", "beats by dr. dre",
    ],
    # Casio: bare 'casio' is fine — they essentially only make watches
    # and calculators, and HiBid auction titles for Casio-branded
    # items are nearly always watches. Skip-list catches replicas.
    "Casio G-Shock + vintage": [
        "casio g-shock", "casio gshock", "g-shock",
        "casio mr-g", "casio mtg", "mt-g",
        "casio frogman", "casio mudmaster", "casio rangeman",
        "casio edifice", "casio pro trek",
        "casio a158", "casio a168", "casio f-91w", "casio f91w",
        "casio ae-1200", "casio ae1200", "casio ae-1300",
        "casio ga-2100", "casio ga-2200", "casio dw-5600",
        "casio gw-9400", "casio gw-9500", "casio gw-m5610",
        "casio men's watch", "casio mens watch",
    ],
    "Citizen": [
        "citizen eco-drive", "citizen eco drive",
        "citizen promaster", "citizen tsuyosa",
        "citizen series 8", "citizen chronomaster",
        "citizen aqualand", "citizen skyhawk",
        "citizen caliber 0100",
    ],
    "Seiko": [
        "seiko",
    ],
    "Hamilton Watch": [
        "hamilton khaki", "hamilton jazzmaster",
        "hamilton ventura", "hamilton intra-matic",
        "hamilton american classic", "hamilton murph",
        "hamilton pan europ",
    ],
    "Bose SoundLink speakers": [
        "bose soundlink",
    ],
    # Premium home audio + audiophile (audio_watches_bolo.json)
    "Premium home audio / hi-fi": [
        "klipsch", "klipsch rp-", "klipsch heresy",
        "klipsch cornwall", "klipsch la scala",
        "klipsch khorn", "klipsch forte", "klipsch chorus",
        "klipsch the sixes", "klipsch the fives",
        "klipsch the nines", "klipsch the sevens",
        "naim audio", "naim mu-so", "naim atom",
        "naim uniti", "naim nap", "naim nac",
        "accuphase",
        "bang olufsen", "bang & olufsen", "bang and olufsen",
        "b&o beoplay", "b&o beosound", "b&o beolab",
        "b&o beocenter", "b&o beosystem", "b&o beolit",
        "b&o beolink", "beoplay", "beosound", "beolab",
        "mcintosh ma-", "mcintosh mc-",
        "mcintosh ma8950", "mcintosh mc462",
        "mark levinson", "mark-levinson",
        "krell k-300i", "krell vanguard",
        "wilson audio",
        "magnepan", "magnepan mg",
        "martin logan", "martin-logan",
        "b&w 805", "b&w 803", "b&w 802", "b&w 705",
        "bowers wilkins", "bowers & wilkins",
        "b&w signature diamond",
        "focal aria", "focal sopra", "focal stellia",
        "focal clear", "focal utopia",
        "kef ls50", "kef ls60", "kef reference",
        "kef q150", "kef q-",
        "tannoy stirling", "tannoy cheviot",
        "tannoy westminster",
        "wharfedale diamond", "wharfedale linton",
        "mission speakers",
        "cerwin-vega", "cerwin vega",
        "jbl l100", "jbl 4310", "jbl 4311", "jbl 4312",
        "jbl l65", "jbl 250ti",
        "polk rti", "polk audio lsim",
        "yamaha ns-10", "yamaha ns-1000",
        "yamaha cr ", "yamaha a-s",
        "pioneer sx-", "pioneer hpm",
        "sansui au-", "sansui g-",
        "niles cm8", "niles in ceiling",
        "definitive technology bp",
        "boston acoustics",
        "acoustic research", "ar speakers",
    ],
    "Pro audio / DJ / broadcast": [
        "technics sl-1200", "technics sl-1210",
        "technics sl-1500c", "technics 1200",
        "pioneer dj djm", "pioneer dj cdj",
        "pioneer dj xdj", "pioneer dj ddj",
        "pioneer dj xdj-rx3", "pioneer dj xdj-xz",
        "cdj-3000", "cdj-2000", "djm-900", "djm-750",
        "numark mixtrack", "numark nv", "numark ns6",
        "numark party mix",
        "denon dj prime", "denon dj sc6000",
        "denon dj x1850", "denon dj mcx8000",
        "denon dn-s1200",
        "allen heath xone", "allen & heath xone",
        "allen & heath gld",
        "reloop rmx", "reloop beatpad",
        "reloop beatmix", "reloop mixon",
        "hercules djcontrol", "hercules inpulse",
        "behringer ddm4000", "behringer nox1010",
        "gemini mxr-01", "gemini mxr01",
        "gemini mxr-01bt", "gemini mxr01bt",
        "mxr-01bt", "mxr01bt",
        "gemini pmx-300", "gemini pt-2400",
        "riedel pro-d1", "riedel bolero",
        "riedel artist", "riedel rsp",
        "clear-com", "clear com pl-pro",
        "clear com freespeak", "clear-com kb-202",
        "sound devices mixpre", "sound devices 833",
        "sound devices 888", "sound devices a20",
        "mackie profx", "mackie 1604vlz",
        "yamaha mg mixer", "yamaha 01v",
        "soundcraft notepad", "soundcraft signature",
        "behringer x32", "behringer xr18",
        "presonus studiolive",
        "tascam model 12", "tascam model 24",
        "tascam dp-32",
        "zoom livetrak", "zoom l-12", "zoom l-20",
        "zoom f8", "zoom h8", "zoom h6",
    ],
    "Budget DIY audio (Aiyima / Fosi / Topping / SMSL)": [
        "aiyima a07", "aiyima a04", "aiyima a03",
        "aiyima t9", "aiyima t2",
        "aiyima tweeter", "aiyima 12 ohm",
        "aiyima",
        "fosi audio", "fosi v3", "fosi bt20a",
        "fosi tb10d", "fosi tb10a",
        "fosi za3", "fosi mc101",
        "topping e30", "topping l30", "topping d90",
        "topping a30", "topping dx5",
        "smsl su-9", "smsl da-8s", "smsl m500",
        "smsl vmv", "smsl ho150",
        "douk audio", "nobsound",
        "fx audio fx-502", "fx audio fx-1002",
        "drok amplifier", "s.m.s.l",
    ],
    # Camera equipment (data/camera_equipment_bolo.json — added 2026-05-04)
    # Canon: bare 'canon' is risky (printers, calculators, copiers).
    # Gate on EOS / RF / EF / specific body model number.
    "Canon Cameras + Lenses": [
        "canon eos", "canon rf", "canon ef-",
        "canon r5", "canon r6", "canon r3", "canon r7",
        "canon r8", "canon r10", "canon r50", "canon r100",
        "canon 5d mark", "canon 6d mark", "canon 7d mark",
        "canon 1dx", "canon 80d", "canon 90d", "canon 70d",
        "canon m50", "canon dslr", "canon mirrorless",
        "canon l usm", "canon l is", "canon l-series",
    ],
    # Nikon: bare 'nikon' false-matches binoculars (Aculon, Monarch).
    # Gate on camera-specific model markers + Nikkor (lens brand).
    "Nikon Cameras + Lenses": [
        "nikon z9", "nikon z8", "nikon z7", "nikon z6", "nikon z5",
        "nikon z fc", "nikon zf", "nikon z50", "nikon z30",
        "nikon z mount", "nikon z body", "nikon z camera",
        "nikon d6", "nikon d5", "nikon d4", "nikon df",
        "nikon d850", "nikon d810", "nikon d800",
        "nikon d750", "nikon d780",
        "nikon d500", "nikon d7500", "nikon d7200", "nikon d7100",
        "nikon dslr", "nikon mirrorless", "nikon body",
        "nikkor", "af-s nikkor", "afs nikkor",
    ],
    # Sony Alpha: gated on alpha / a-series / FE / GM / camera body
    # codes to avoid conflict with Sony headphones (already a separate
    # BOLO entry) and Sony TVs / PlayStation.
    "Sony Alpha Cameras + Lenses": [
        "sony alpha", "sony a7", "sony a9", "sony a1",
        "sony a7r", "sony a7s", "sony a7c",
        # Specific a6xxx model variants — the bare "sony a6" alias
        # didn't match "sony a6700" because the digit-trailing
        # boundary blocks it. Each variant is its own alias.
        "sony a6000", "sony a6300", "sony a6400",
        "sony a6500", "sony a6600", "sony a6700",
        "sony zv-e10", "sony zv-1", "sony fx3", "sony fx30",
        "sony fe ", "sony g master", "sony gm lens",
    ],
    # Fujifilm: bare 'fuji' / 'fujifilm' is fine — they primarily make
    # cameras + film + medical imaging (the latter not appearing in
    # auction context).
    "Fujifilm Cameras + Lenses": [
        "fujifilm", "fuji x-", "fuji x100", "fuji gfx",
        "fujinon", "fuji film camera",
    ],
    "Leica": [
        "leica",
    ],
    "Hasselblad": [
        "hasselblad",
    ],
    # Sigma: alias 'sigma' is OK — they're primarily a lens maker now.
    # Older sigma alone might match other meanings, so qualify with
    # Art / Sport / Contemporary / DG / DN / camera-specific suffixes.
    "Sigma lenses": [
        "sigma art", "sigma sport", "sigma contemporary",
        "sigma dg dn", "sigma dg hsm", "sigma fp",
        # Focal-length aliases tend to fail on titles like
        # "Sigma 24-70mm" (the "70mm" tokenizes as one word and the
        # trailing boundary blocks "70" alone). Use the lens name
        # patterns instead — "sigma art" / "sigma dg dn" / "sigma
        # sport" cover most well-formatted lens titles.
        "sigma 150-600", "sigma 60-600",
        "sigma 18-35 f", "sigma 50-100 f",
    ],
    "Tamron lenses": [
        "tamron",
    ],
    "Zeiss lenses": [
        "zeiss otus", "zeiss milvus", "zeiss loxia",
        "zeiss batis", "zeiss touit",
        "carl zeiss", "zeiss planar", "zeiss distagon",
        "zeiss sonnar", "zeiss apo",
    ],
    "Olympus + OM SYSTEM": [
        "olympus om", "olympus pen", "olympus e-m1",
        "olympus e-m5", "om system", "om-d e-m",
        "m.zuiko", "zuiko",
    ],
    "Panasonic Lumix": [
        "panasonic lumix", "lumix s1", "lumix s5",
        "lumix s9", "lumix gh", "lumix g9", "lumix g100",
        "leica dg",
    ],
    "DJI Drones + Gimbals": [
        "dji mavic", "dji mini ", "dji phantom",
        "dji inspire", "dji avata", "dji fpv",
        "dji ronin", "dji osmo", "dji rs ",
    ],
    "Profoto / Godox lighting": [
        "profoto",
        # Godox specific model variants. Bare "godox ad" doesn't
        # match "Godox AD600" because "AD600" tokenizes as one
        # word and the trailing boundary blocks "ad" alone.
        "godox ad200", "godox ad300", "godox ad400", "godox ad600",
        "godox v1", "godox v860", "godox sl",
        "godox vl150", "godox vl300", "godox movelink",
    ],
    # Auto parts (data/auto_parts_bolo.json — added 2026-05-04)
    # Bosch automotive shares 'bosch' alias root with Bosch appliance
    # parts (existing entry). The matcher's strong-match-wins logic
    # picks whichever entry's models match the title — automotive
    # tokens (spark plug / alternator / etc.) route here, dishwasher
    # tokens route to the appliance entry.
    "Bosch automotive": [
        "bosch spark plug", "bosch double iridium", "bosch platinum",
        "bosch alternator", "bosch starter motor",
        "bosch fuel injector", "bosch fuel pump",
        "bosch oxygen sensor", "bosch lambda sensor",
        "bosch ignition coil", "bosch glow plug",
        "bosch wiper blade", "bosch mass air flow",
        "bosch abs module", "bosch headlight",
        "bosch motronic", "bosch cp4",
    ],
    "Denso": [
        "denso iridium", "denso platinum", "denso alternator",
        "denso starter", "denso a/c compressor",
        "denso ac compressor", "denso fuel injector",
        "denso oxygen sensor", "denso ignition coil",
        "denso radiator", "denso fuel pump",
        "denso ac condenser",
    ],
    "NGK ignition": [
        "ngk iridium", "ngk laser", "ngk g-power",
        "ngk v-power", "ngk standard",
        "ngk ignition", "ngk cop", "ngk coil",
        "ngk wire", "ngk o2 sensor", "ngk spark plug",
        "ntk lambda", "ntk oxygen sensor",
    ],
    "ACDelco": [
        "acdelco", "ac delco", "ac-delco",
    ],
    "Motorcraft": [
        "motorcraft", "ford motorcraft", "fl-820s", "fl820s",
    ],
    "Mopar": [
        "mopar", "mopar performance", "mopar genuine",
        "mopar hellcat", "mopar srt", "mopar hemi",
        "mopar demon", "mopar trx",
    ],
    "Brembo brakes": [
        "brembo",
    ],
    "Akebono brakes": [
        "akebono",
    ],
    "Hawk Performance brakes": [
        "hawk hps", "hawk hp+", "hawk dtc",
        "hawk performance ceramic", "hawk blue 9012",
        "hawk pc lts",
    ],
    "PowerStop / EBC brakes": [
        "powerstop", "power stop z", "powerstop k-series",
        "ebc yellowstuff", "ebc redstuff", "ebc greenstuff",
        "ebc bluestuff", "ebc orangestuff",
        "ebc usr rotor", "ebc gd rotor",
    ],
    "K&N filters": [
        "k&n", "k & n", "kn cold air intake",
        "k n filter", "k&n cai",
    ],
    "Premium OEM filters": [
        "mann-filter", "mann filter", "mann + hummel",
        "mahle filter", "mahle ox", "mahle oc",
        "wix filter", "wix xp", "wix pro-tec",
        "hengst filter",
    ],
    "Gates belts and hoses": [
        "gates serpentine", "gates timing belt",
        "gates micro-v", "gates fluidshield",
        "gates green stripe", "gates k-belt",
        "gates radiator hose", "gates powergrip",
    ],
    "Hella + Philips automotive lighting": [
        "hella headlight", "hella h7", "hella h4",
        "hella h11", "hella h8", "hella xenon", "hella hid",
        "philips x-tremevision", "philips racingvision",
        "philips diamond vision", "philips ultinon",
        "philips d2s", "philips d3s",
        "osram night breaker", "osram cool blue",
    ],
    "WeatherTech": [
        "weathertech", "weather tech",
    ],
    "Magnaflow exhaust": [
        "magnaflow",
    ],
    "Borla exhaust": [
        "borla atak", "borla s-type", "borla touring",
        "borla cat-back", "borla axle-back", "borla pro xs",
    ],
    "HKS Japanese performance": [
        "hks hi-power", "hks legamax", "hks ssqv",
        "hks gt2", "hks gt3", "hks type-s",
        "hks hipermax", "hks turbo", "hks bov",
    ],
    "Mishimoto cooling": [
        "mishimoto",
    ],
    "BBS Wheels": [
        "bbs lm", "bbs rs", "bbs rm", "bbs rc",
        "bbs ch-r", "bbs ch ", "bbs fi-r", "bbs fs ",
        "bbs rgr", "bbs sr ", "bbs super rs",
        "bbs forged",
    ],
    "Volk Racing + Enkei wheels": [
        "volk te37", "volk ce28n", "volk ze40", "volk g25",
        "volk racing", "rays engineering",
        "enkei rpf1", "enkei nt03", "enkei pf01",
        "enkei rc-t5", "enkei ts-10", "enkei kojin",
    ],
    # Lightweight collectibles (data/lightweight_collectibles_bolo.json
    # — added 2026-05-04). User preference: items <5lbs, ship in
    # poly mailer or small box.
    # Coleman: SPECIFIC to vintage lanterns. Bare 'coleman' would
    # match Coleman coolers / water jugs which are too bulky.
    "Coleman vintage lanterns": [
        # SPECIFIC to lanterns. 'vintage coleman' alone false-matched
        # "Vintage Coleman Cooler" — too broad, removed. Coolers,
        # water jugs, propane fuel canisters are too bulky to ship
        # per user preference, so we WANT them to miss.
        "coleman 200a", "coleman 200 ",
        "coleman 220", "coleman 228",
        "coleman 242", "coleman 275", "coleman 282",
        "coleman 286", "coleman 290", "coleman 295",
        "coleman 321",
        "coleman lantern", "coleman gas lantern",
        "coleman propane lantern", "coleman kerosene lantern",
    ],
    # Yeti: tumblers / drinkware ONLY. Excludes Tundra / Hopper /
    # Loadout coolers (too bulky).
    "Yeti drinkware": [
        "yeti rambler", "yeti tumbler", "yeti colster",
        "yeti yonder", "yeti daytrip", "yeti stackable",
        "yeti wine tumbler", "yeti half gallon",
        "yeti 1 gallon",
    ],
    "NOCO + Schumacher chargers": [
        "noco genius", "noco g750", "noco g3500", "noco g7200",
        "noco gb40", "noco gb50", "noco gb70", "noco gb150",
        "noco boost",
        "noco genius1", "noco genius5", "noco genius10",
        "schumacher sc1280", "schumacher sc1281", "schumacher sp1297",
        "schumacher dsr131", "schumacher se-",
        "ctek mus", "ctek mxs", "ctek ct5",
        "battery tender junior", "battery tender plus",
    ],
    "Vintage motorcycle helmets": [
        # 'bell helmet' / 'bell 500' / 'bell star' are specific enough
        # to avoid matching plain 'bell' words.
        "bell 500", "bell star helmet", "bell magnum helmet",
        "bell toptex", "bell moto-3", "bell moto-4", "bell moto-5",
        "bell r-t helmet", "bell motorcycle helmet",
        "vintage bell helmet",
        "arai quantum", "arai corsair", "arai signet",
        "shoei rf-1400", "shoei x-spirit", "shoei gt-air",
        "agv k6", "agv pista",
        "buco helmet", "buco defender",
    ],
    "Vintage radios": [
        "zenith transoceanic", "zenith royal 3000",
        "zenith royal 1000", "zenith t600", "zenith h500", "zenith g500",
        "zenith wave magnet",
        "sony icf-7600", "sony icf-2010", "sony icf-sw77",
        "sony tr-63", "sony tr-65", "sony tfm",
        "grundig satellit", "grundig yacht boy", "grundig g3",
        "panasonic rf-2200", "panasonic rf-4900",
        "realistic dx-440",
        "hallicrafters s-38", "hallicrafters s-40", "hallicrafters s-120",
    ],
    "Pelican cases": [
        "pelican 1170", "pelican 1200", "pelican 1300",
        "pelican 1400", "pelican 1450", "pelican 1500",
        "pelican 1510", "pelican 1520", "pelican 1535",
        "pelican 1610", "pelican 1620", "pelican 1650",
        "pelican vault", "pelican air", "pelican storm",
        "pelican ruck", "pelican case",
    ],
    "Premium pocket knives": [
        "spyderco para military", "spyderco pm2",
        "spyderco paramilitary", "spyderco manix",
        "spyderco endura", "spyderco delica", "spyderco native",
        "spyderco tenacious", "spyderco sage",
        "benchmade bugout", "benchmade griptilian",
        "benchmade 535", "benchmade 551",
        "benchmade adamas", "benchmade bailout",
        "benchmade infidel", "benchmade 940",
        "microtech ultratech", "microtech combat troodon",
        "microtech halo", "microtech utx",
        "zero tolerance", "zt 0350", "zt 0562", "zt 0566",
        "hinderer xm-18", "hinderer halftrack",
        "crkt m16", "cold steel recon",
    ],
    "Multi-tools": [
        "leatherman wave", "leatherman charge",
        "leatherman surge", "leatherman skeletool",
        "leatherman wingman", "leatherman sidekick",
        "leatherman free p4", "leatherman free p2",
        "leatherman signal", "leatherman oht",
        "leatherman style", "leatherman squirt",
        "gerber suspension", "gerber mp600", "gerber mp1",
        "gerber center-drive", "gerber truss",
        "victorinox swisstool",
        "sog powerassist", "sog powerlock",
    ],
    "Metal detectors": [
        "bounty hunter discovery", "bounty hunter tracker",
        "bounty hunter land ranger", "bounty hunter pioneer",
        "bounty hunter time ranger", "bounty hunter quick draw",
        "garrett at pro", "garrett at max", "garrett at gold",
        "garrett ace 200", "garrett ace 300", "garrett ace 400",
        "garrett apex", "garrett pro-pointer",
        "whites spectra", "whites mx sport", "whites coinmaster",
        "whites treasuremaster",
        "minelab equinox", "minelab manticore", "minelab ctx",
        "xp deus", "xp orx",
        "fisher f22", "fisher f44", "fisher f75",
        "metal detector",
    ],
    # Estate collectibles (data/estate_collectibles_bolo.json — added 2026-05-04)
    "Hummel / Goebel figurines": [
        "hummel", "m.i. hummel", "mi hummel", "goebel hummel",
        "goebel figurine",
    ],
    # Disney character figurines (disney_figurine, added 2026-07-28). All
    # distinctive enough for alias-only matches; the JSON models[] refine
    # to specific pieces when present.
    "Jim Shore Disney Traditions": [
        "jim shore", "disney traditions", "heartwood creek",
    ],
    "Disney Showcase / Couture de Force": [
        "disney showcase", "couture de force",
    ],
    "Walt Disney Classics Collection (WDCC)": [
        "wdcc", "walt disney classics collection", "walt disney classics",
    ],
    "Grand Jester Studios Disney": [
        "grand jester studios", "grand jester",
    ],
    "Lladró figurines": [
        "lladro", "lladró",
    ],
    "Royal Doulton": [
        "royal doulton", "royal-doulton",
    ],
    "Wedgwood": [
        "wedgwood", "wedgewood",  # common misspelling
        "jasperware", "jasper ware",
    ],
    "Norman Rockwell collectibles": [
        "norman rockwell",
    ],
    "Roseville Pottery": [
        # 'roseville' alone is risky (Roseville CA city). Gate on
        # pottery / vase / pattern context.
        "roseville pottery", "roseville vase",
        "roseville pinecone", "roseville magnolia",
        "roseville zephyr", "roseville apple blossom",
        "roseville bushberry", "roseville snowberry",
        "roseville freesia", "roseville iris",
        "roseville foxglove", "roseville clematis",
        "roseville water lily", "roseville wisteria",
        "roseville bleeding heart", "roseville donatello",
    ],
    "Hull / McCoy / Weller / Rookwood pottery": [
        "hull pottery", "hull vase", "hull cookie jar",
        "hull little red riding hood",
        # McCoy aliases broadened: bare "vtg mccoy" / "mccoy vase" /
        # "mccoy art deco" weren't matching. Adding more variants.
        "mccoy pottery", "mccoy usa", "mccoy cookie jar",
        "mccoy planter", "mccoy vase", "mccoy art deco",
        "vtg mccoy", "vintage mccoy",
        "weller pottery", "weller forest", "weller hudson",
        "weller coppertone", "weller sicard", "vtg weller",
        "rookwood pottery", "rookwood vellum",
    ],
    "Fenton Art Glass": [
        "fenton art glass", "fenton glass",
        "fenton carnival", "fenton hobnail",
        "fenton burmese", "fenton custard",
        "fenton opalescent", "fenton silvercrest",
        "fenton milk glass",
    ],
    "Imperial Candlewick + premium glassware": [
        "imperial candlewick", "candlewick punch",
        "candlewick glass", "candlewick crystal",
        "heisey orchid", "heisey rose", "heisey crystal",
        "heisey glass",
        "cambridge rose point", "cambridge caprice",
        "cambridge glass",
        "fostoria american", "fostoria coin", "fostoria crystal",
        "westmoreland milk glass",
    ],
    "Vintage signed costume jewelry": [
        # Each brand alias requires the SIGNED context to avoid
        # matching unrelated mentions (e.g. 'monet print').
        "trifari signed", "crown trifari", "trifari brooch",
        "trifari jewelry", "trifari earrings", "trifari necklace",
        "eisenberg ice", "eisenberg original", "eisenberg signed",
        "weiss signed", "weiss rhinestone", "weiss brooch",
        "weiss jewelry",
        "miriam haskell",
        "hobé jewelry", "hobe signed", "hobé signed",
        "coro craft", "coro duette", "coro signed",
        "sarah coventry",
        "marcel boucher",
        "hattie carnegie", "schiaparelli",
        "joseff of hollywood",
        "kramer jewelry", "schreiner jewelry",
        "lisner jewelry", "monet brooch", "napier brooch",
        "bogoff jewelry",
        # Krementz: signed mid-tier costume jewelry (1865-2009),
        # often gold-filled / 14kt. Sustained collector demand.
        "krementz",
    ],
    "James Avery + Brighton + Pandora": [
        "james avery",
        "brighton heart bracelet", "brighton charm",
        "brighton silver bracelet", "brighton necklace",
        "pandora bracelet", "pandora charm",
        "pandora ale", "pandora ring",
    ],
    "Swarovski crystal": [
        "swarovski",
    ],
    "Steiff + vintage stuffed animals": [
        "steiff",
    ],
    "Vintage Fisher-Price": [
        "vintage fisher-price", "vintage fisher price",
        "fisher-price little people", "fisher price little people",
        "fisher-price chatter telephone",
        "fisher-price farmer in the dell",
        "fisher-price pull toy", "fisher price pull toy",
        "fisher-price wooden", "fisher-price music box",
        "fisher-price family farm", "fisher-price castle",
        "fisher-price adventure people",
    ],
    "Longaberger baskets": [
        "longaberger",
    ],
    "Rae Dunn pottery": [
        "rae dunn",
    ],
    "Lenox limited editions": [
        # Bare 'lenox' would match commodity Lenox china. Gate on
        # premium / limited / Disney / figurine / Carousel / fine-
        # china (24k gold trimmed) / made-in-USA context.
        "lenox limited", "lenox figurine",
        "lenox disney", "lenox carousel",
        "lenox heritage", "lenox christmas figurine",
        "lenox bone china figurine",
        "lenox fine china", "lenox 24k gold",
        "lenox gold trimmed", "lenox made in usa",
        "lenox crystal", "lenox vase",
    ],
    "Steuben + Waterford crystal": [
        "steuben crystal", "steuben glass", "steuben sculpture",
        "waterford crystal", "waterford lismore",
        "waterford marquis", "waterford alana",
        "waterford colleen",
    ],
    "Belleek Irish porcelain": [
        "belleek",
    ],
    "Vintage Lionel + Marx trains": [
        "lionel train", "lionel postwar", "lionel prewar",
        "lionel locomotive", "lionel boxcar",
        "lionel o gauge", "lionel o scale",
        "marx train", "marx tin litho",
        "american flyer train",
    ],
    # Pop culture collectibles (data/pop_culture_collectibles_bolo.json)
    "Funko Pop": [
        "funko pop", "funko",
        "funko soda", "funko mystery mini",
    ],
    # Hot Wheels: alias picks up legitimate collector variants. The
    # "Hot Wheels Brembo die-cast collectible" case routes here first
    # (correctly identifying it as a Hot Wheels toy, not a Brembo
    # auto part) due to JSON-order strong-match precedence. Modern
    # mainline blue-card Hot Wheels (commodity tier) match alias-only
    # which the user can manually skip.
    "Hot Wheels collector variants": [
        # Specific collector variants only — DO NOT add a generic
        # "hot wheels" alias here. The Wolf/Bone 5/21 audit had 21 BOLO
        # false-positives on modern mainline cars (Spongebob, Cosworth,
        # BMW 507, etc.) because the previous catch-all alias fired on
        # every Hot Wheels lot. Mainline blue-card resells at $1-3 and
        # should NOT surface as a BOLO hit. Treasure Hunts / Super THs /
        # Redlines / RLC / pop-culture lines are the real premium tier.
        "hot wheels treasure hunt", "hot wheels th",
        "hot wheels super treasure hunt", "hot wheels super th",
        "hot wheels redline", "hot wheels rlc",
        "hot wheels red line club",
        "hot wheels premium", "hot wheels boulevard",
        "hot wheels car culture", "hot wheels cars culture",
        "hot wheels convention", "hot wheels mattel creations",
        "hot wheels star wars", "hot wheels marvel",
        "hot wheels pop culture", "hot wheels entertainment",
        "hot wheels hall of fame",
        "matchbox lesney", "matchbox models of yesteryear",
        "matchbox 1-75",
    ],
    "Vintage Kenner Star Wars": [
        "kenner star wars", "vintage kenner",
        "kenner action figure",
        "power of the force kenner", "potf kenner",
        "death star playset", "kenner death star",
        "millennium falcon kenner", "kenner millennium falcon",
        "kenner at-at", "kenner x-wing",
        "kenner carded", "kenner cardback",
        "12-back", "20-back", "21-back", "31-back",
        "32-back", "41-back", "45-back", "47-back",
        "65-back", "77-back", "92-back",
    ],
    "Topps trading cards": [
        "topps factory sealed", "topps wax box",
        "topps wax pack", "topps star wars",
        "topps heritage", "topps chrome",
        "topps allen", "topps update",
        "topps bowman", "topps stadium club",
        "topps tier one", "topps triple threads",
        "topps definitive", "topps dynasty",
        "topps now", "topps project 70",
        "topps garbage pail", "topps gpk",
        "topps wacky packages",
        "topps mickey mantle", "topps rookie",
    ],
    "Vintage Transformers + GI Joe + Masters": [
        "g1 transformers", "generation 1 transformers",
        "optimus prime g1", "megatron g1",
        "transformers takara", "soundwave g1",
        "vintage gi joe", "gi joe arah",
        "gi joe real american hero",
        "snake eyes gi joe", "cobra commander",
        "masters of the universe", "motu",
        "he-man vintage", "she-ra vintage",
        "motu origins", "motu classics",
        "skeletor vintage", "castle grayskull",
        "vintage mego", "mego world's greatest",
    ],
}


def _whole_word_pattern(literal: str, max_gap_words: int = 6) -> re.Pattern:
    """Build a case-insensitive whole-word regex from a literal phrase.

    Single-token aliases get strict whole-word boundaries.

    Multi-token aliases get a proximity match: each consecutive pair
    of alias tokens can have up to ``max_gap_words`` intervening words
    between them in the haystack. This handles titles like:

        "Bosch SHX863WD55N Dishwasher Drain Pump"
            ↑ alias 'bosch dishwasher' matches because the model
              number sits between the two alias tokens (within
              the 4-word proximity window).

    Both leading and trailing boundaries block letters AND digits — so
    'iphone 12' won't match 'iPhone 128GB', 'series 1' won't match
    'Series 10', '501' won't be a fragment of '200501'. Multi-token
    aliases like 'bosch dishwasher' still match titles where a model
    number sits between the alias tokens (via the proximity gap
    mechanism), so we don't need to relax boundaries to handle the
    'Bosch SHX863... Dishwasher' case.
    """
    tokens = re.findall(r"\w+", literal)
    if not tokens:
        return re.compile(r"$.^")  # never matches
    if len(tokens) == 1:
        token_re = re.escape(tokens[0])
        return re.compile(
            rf"(?<![A-Za-z0-9]){token_re}(?![A-Za-z0-9])",
            re.IGNORECASE,
        )
    # Multi-token: connect tokens with `\W+` PLUS an optional run of
    # up to max_gap_words intervening word-blobs.
    parts = [re.escape(tokens[0])]
    for t in tokens[1:]:
        # `(?:\w+\W+){0,N}` = up to N intervening (word + non-word).
        parts.append(rf"\W+(?:\w+\W+){{0,{max_gap_words}}}")
        parts.append(re.escape(t))
    inner = "".join(parts)
    return re.compile(
        rf"(?<![A-Za-z0-9]){inner}(?![A-Za-z0-9])",
        re.IGNORECASE,
    )


class BoloMatcher:
    """Stateful, hot-reloadable brand-list matcher.

    Construct once at app startup; ``match()`` is cheap enough to call
    per-row during render. The underlying JSON is reloaded automatically
    if its mtime changes, so the user can drop a new file in and the
    next Streamlit rerun will use it.
    """

    def __init__(self, path: Union[Path, Sequence[Path], None] = None):
        # Accept either a single Path (back-compat) or a list of Paths.
        # When None, use DEFAULT_BOLO_PATHS — caller doesn't need to
        # know about household_parts being a separate file.
        if path is None:
            self.paths: List[Path] = list(DEFAULT_BOLO_PATHS)
        elif isinstance(path, (str, Path)):
            self.paths = [Path(path)]
        else:
            self.paths = [Path(p) for p in path]
        # Back-compat single-path attribute (some old callers read .path)
        self.path = self.paths[0] if self.paths else DEFAULT_BOLO_PATH
        self._lock = threading.Lock()
        # Track per-file mtime so we can detect any file changing.
        self._mtimes: Dict[str, float] = {}
        self._brand_patterns: List[tuple] = []
        # tuple = (brand_canonical, alias_pattern, models, brand_meta)
        self._model_patterns: List[tuple] = []
        # tuple = (brand_canonical, model_name, model_pattern)
        self._skip_patterns: List[re.Pattern] = []
        self._brand_meta: Dict[str, Dict[str, Any]] = {}
        self._raw: Dict[str, Any] = {}
        # Persistent match-result cache (in-memory dict, lazily loaded
        # from disk + flushed on a write counter). See match() docstring
        # for invalidation semantics.
        self._match_cache: Optional[Dict[str, Any]] = None
        self._match_cache_loaded_for_fingerprint: Optional[str] = None
        self._match_cache_dirty_writes: int = 0
        self._match_cache_lock = threading.Lock()
        self._load_if_stale()

    # ---- file I/O -------------------------------------------------

    # Throttle for the per-match staleness check. _load_if_stale runs
    # 22 os.stat calls on the JSON files which costs ~1.4ms each call.
    # In a 14k-lot scan that's 20s of wasted file-stat overhead. We
    # debounce: only re-stat after this many seconds since the last
    # successful check. Editing a JSON file means the new content is
    # picked up at most _STALE_CHECK_INTERVAL seconds later — fine
    # for normal workflow.
    _STALE_CHECK_INTERVAL = 2.0

    def _load_if_stale(self) -> None:
        """Reload the JSON files when any source mtime changes.

        Throttled — runs the actual file-stat check at most every
        ``_STALE_CHECK_INTERVAL`` seconds. Without throttling, a
        14k-lot match scan pays 20+ seconds of os.stat overhead.

        With multiple files, any single file changing triggers a full
        rebuild on the next throttle window.
        """
        import time as _t
        now = _t.monotonic()
        last = getattr(self, '_stale_check_at', 0.0)
        if (now - last) < self._STALE_CHECK_INTERVAL and self._brand_patterns:
            # Still in the debounce window AND the matcher is loaded.
            # Skip the file-stat ritual.
            return
        current_mtimes: Dict[str, float] = {}
        for p in self.paths:
            try:
                current_mtimes[str(p)] = p.stat().st_mtime
            except OSError:
                # Missing file is OK — just skip it. The other file(s)
                # still load. If ALL files are missing the matcher is
                # empty and match() returns None.
                continue
        # Update the throttle marker even when no reload happens — the
        # check itself is what we're rate-limiting, not the reload.
        self._stale_check_at = now
        if current_mtimes == self._mtimes and self._brand_patterns:
            return
        with self._lock:
            if current_mtimes == self._mtimes and self._brand_patterns:
                return
            self._reload(current_mtimes)

    def _reload(self, mtimes: Dict[str, float]) -> None:
        # Concatenate brands + skip_lists from every loadable file.
        # A failure on one file shouldn't take down the others.
        merged_brands: List[Dict] = []
        merged_skip: List[str] = []
        loaded_raw: Dict[str, Any] = {"sources": {}}
        for p in self.paths:
            try:
                with open(p, "r", encoding="utf-8") as f:
                    file_data = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            loaded_raw["sources"][str(p)] = file_data
            merged_brands.extend(file_data.get("brands") or [])
            merged_skip.extend(file_data.get("skip_list") or [])

        # The rest of the build runs on the unioned brand list.
        data = {"brands": merged_brands, "skip_list": merged_skip}

        brand_patterns: List[tuple] = []
        model_patterns: List[tuple] = []
        skip_patterns: List[re.Pattern] = []
        brand_meta: Dict[str, Dict[str, Any]] = {}

        for entry in data.get("brands", []) or []:
            canonical = entry.get("brand") or ""
            if not canonical:
                continue
            aliases = _BRAND_ALIASES.get(canonical, [canonical.lower()])
            alias_pats = [_whole_word_pattern(a) for a in aliases if a]
            models = entry.get("models") or []
            # Deduplicate model patterns and skip ones too short to be
            # specific (a 2-char model would match too aggressively).
            model_pats = []
            for m in models:
                if not isinstance(m, str) or len(m) < 4:
                    continue
                model_pats.append((m, _whole_word_pattern(m)))

            # Stash metadata once per brand for the UI to pick up.
            target = entry.get("target_buy_usd") or {}
            tier = entry.get("tier")
            category = entry.get("category") or ""
            # auth_required heuristic: tier-3 luxury + the iconic-vintage
            # tier-2 "premium subline" entries (Polo Snow Beach, RRL,
            # Stadium) where counterfeits are common. Tier-1 athleisure /
            # outdoor doesn't get auth-required because Lululemon Aligns
            # don't have a fake market worth worrying about.
            auth_required = (
                (tier == 3 and category in ("luxury", "luxury_mid", "sneakers"))
                or canonical == "Polo Ralph Lauren premium sublines"
            )
            # Per-brand disqualifier list: when any of these phrases
            # appears in the haystack, the brand alias match is REJECTED.
            # Used by precious-metals entries to block "silver-plated" /
            # "gold-tone" / "platinum membership" false positives.
            # Phrases are case-insensitive substring checks (NOT regex)
            # for speed and predictable behavior.
            disqualifier_phrases = entry.get("_disqualifiers") or []
            disqualifier_phrases = [
                str(p).lower() for p in disqualifier_phrases
                if isinstance(p, str) and p
            ]

            brand_meta[canonical] = {
                "tier": tier,
                "category": category,
                "platform_primary": entry.get("platform_primary") or "",
                "platform_secondary": entry.get("platform_secondary") or "",
                "ship_class": entry.get("ship_class") or "",
                "target_buy_low": target.get("low"),
                "target_buy_high": target.get("high"),
                "notes": entry.get("notes") or "",
                "era_markers": entry.get("era_markers") or [],
                "auth_required": auth_required,
                "disqualifier_phrases": disqualifier_phrases,
            }

            for ap in alias_pats:
                brand_patterns.append((canonical, ap, model_pats))
            for m_name, m_pat in model_pats:
                model_patterns.append((canonical, m_name, m_pat))

        for skip in data.get("skip_list", []) or []:
            if not isinstance(skip, str):
                continue
            # Skip-list entries are case-insensitive whole-phrase. Keep
            # the parenthetical clarifications ("modern Coach (post-2000
            # China-made)") matching just the leading brand fragment.
            head = skip.split("(", 1)[0].strip()
            if head:
                skip_patterns.append(_whole_word_pattern(head))

        # Combined OR-regex of ALL alias patterns. Used as a fast
        # "is there any alias match potential?" pre-filter inside
        # match() — replaces a Python-level `any(p.search(...) for p
        # in patterns)` loop over ~750 patterns with a single C-level
        # regex search. Behavior-equivalent: if this combined regex
        # matches, at least one of the per-brand alias patterns will
        # also match in the subsequent loop.
        any_alias_re = None
        if brand_patterns:
            try:
                # Each alias pattern's source is a `(?<![A-Za-z0-9])
                # phrase(?![A-Za-z0-9])` shape. Combining them with
                # `|` and a single non-capturing group preserves the
                # boundary semantics. Use re.IGNORECASE since the
                # matcher lowercases its haystack already; this just
                # mirrors the per-pattern flag.
                pattern_sources = [
                    p.pattern for _, p, _ in brand_patterns
                ]
                # De-duplicate identical alias regexes (some brands
                # share aliases like "boss") to keep the combined
                # regex compact.
                pattern_sources = list(dict.fromkeys(pattern_sources))
                any_alias_re = re.compile(
                    "|".join(pattern_sources), re.IGNORECASE,
                )
            except re.error:
                # If combining fails (some pathological combination of
                # patterns), fall back to per-pattern loop in match()
                any_alias_re = None

        self._raw = loaded_raw
        self._brand_patterns = brand_patterns
        self._model_patterns = model_patterns
        self._skip_patterns = skip_patterns
        self._brand_meta = brand_meta
        self._mtimes = mtimes
        self._any_alias_re = any_alias_re

        # ---- Word-level trigger index (PERF 7/13) --------------------
        # The 361KB / 5,333-alternative combined regex above cost ~210ms
        # PER LOT because a per-branch lookbehind kills the engine's
        # first-char optimization. Replace it in the hot path with an
        # inverted index: every literal token in a brand's alias
        # pattern → the indices of brand_patterns needing it. A lot is
        # tokenized ONCE, and only brands whose trigger words actually
        # appear get their (real, boundary-correct) alias_pat tested.
        # Superset-safe: each literal token IS required by its pattern,
        # so a brand absent from the candidate set provably cannot match.
        # Patterns with no extractable literal token (rare) go in
        # `_always_test` and are checked every lot.
        self._trigger_index: Dict[str, list] = {}
        self._always_test: list = []
        for _bi, (_canon, _ap, _mp) in enumerate(brand_patterns):
            _toks = _TRIGGER_TOKEN_RE.findall(_ap.pattern.lower())
            # Drop regex keyword noise that appears literally in sources.
            _toks = [t for t in _toks if t not in _TRIGGER_STOPWORDS]
            if not _toks:
                self._always_test.append(_bi)
                continue
            for _t in set(_toks):
                self._trigger_index.setdefault(_t, []).append(_bi)

        # ---- Model-pattern buckets (PERF 7/13) -----------------------
        # The Pass-2 model-only sub-passes each filtered ALL 9,735 model
        # patterns by category/name on EVERY lot (precious-metal Pass 2e
        # ran with no context gate at all). Precompute the buckets once
        # so each pass iterates only its own handful. Built by iterating
        # model_patterns in order, so within-bucket order — and the
        # first-match-wins behavior — is identical to the old filtered
        # loops.
        self._model_pats_by_cat: Dict[str, list] = {}
        for _mp in model_patterns:
            _c = (brand_meta.get(_mp[0]) or {}).get('category')
            self._model_pats_by_cat.setdefault(_c, []).append(_mp)
        self._model_pats_band_tee = [
            _mp for _mp in model_patterns
            if "Vintage" in _mp[0] and "tee" in _mp[0].lower()
        ]
        self._model_pats_watch_acc = [
            _mp for _mp in model_patterns if "accessories" in (_mp[0] or "")
        ]
        self._model_pats_loungefly = [
            _mp for _mp in model_patterns if _mp[0] == "Loungefly"
        ]
        # Trigger-word set for the precious-metal Pass 2e — the only
        # model-only pass with no context gate (it ran its ~65 patterns
        # + a 133-phrase disqualifier check on EVERY lot, ~23% of match
        # time). Gate it on a fast word-set intersection: if the lot
        # shares none of these patterns' literal tokens, no precious-
        # metal model can match. Superset-safe (each token is required
        # by its pattern), so behavior is unchanged.
        self._precious_trigger_words: set = set()
        for _c, _n, _p in self._model_pats_by_cat.get('precious_metal', ()):
            for _t in _TRIGGER_TOKEN_RE.findall(_p.pattern.lower()):
                if _t not in _TRIGGER_STOPWORDS:
                    self._precious_trigger_words.add(_t)
        # Mtime fingerprint changed → in-memory cache is now stale.
        # We reset to None; next match() call will lazy-load (which
        # will see the fingerprint mismatch and start fresh).
        self._match_cache = None
        self._match_cache_loaded_for_fingerprint = None
        self._match_cache_dirty_writes = 0

    # ---- match-cache helpers --------------------------------------

    def _fingerprint(self) -> str:
        """Stable string representing the loaded BOLO file set + mtimes.

        Used as the cache-validity key — when ANY file's mtime changes,
        the fingerprint changes and the cache is treated as cold. Hex
        digest keeps the on-disk JSON key short.
        """
        try:
            items = sorted(self._mtimes.items())
        except (AttributeError, TypeError):
            items = []
        # Include the path strings so a config change (different files
        # loaded) also invalidates the cache.
        material = "|".join(f"{k}={v}" for k, v in items).encode("utf-8")
        return hashlib.sha256(material).hexdigest()[:16]

    @staticmethod
    def _haystack_key(haystack: str) -> str:
        """Hash a normalized haystack into a compact cache key.

        SHA-256 truncated to 16 hex chars = 64-bit collision space.
        At our scale (millions of unique titles max) the collision
        probability is vanishingly small.
        """
        return hashlib.sha256(haystack.encode("utf-8")).hexdigest()[:16]

    def _get_match_cache(self) -> Optional[Dict[str, Any]]:
        """Return the in-memory cache dict, lazily loading from disk.

        First call (or call after fingerprint change) reads the JSON
        file. Subsequent calls return the same dict reference, so
        match() can mutate it directly.
        """
        fp = self._fingerprint()
        if (
            self._match_cache is not None
            and self._match_cache_loaded_for_fingerprint == fp
        ):
            return self._match_cache

        # Lazy load from disk — accept only if fingerprint matches.
        loaded: Dict[str, Any] = {}
        try:
            if _BOLO_MATCH_CACHE_PATH.exists():
                with _BOLO_MATCH_CACHE_PATH.open("r", encoding="utf-8") as f:
                    raw = json.load(f)
                if isinstance(raw, dict) and raw.get("fingerprint") == fp:
                    entries = raw.get("entries") or {}
                    if isinstance(entries, dict):
                        loaded = dict(entries)
        except (OSError, json.JSONDecodeError):
            # Treat any read failure as cold cache. Don't crash the
            # match path over a corrupt cache file.
            loaded = {}

        with self._match_cache_lock:
            self._match_cache = loaded
            self._match_cache_loaded_for_fingerprint = fp
            self._match_cache_dirty_writes = 0
        return self._match_cache

    def _maybe_flush_cache(self) -> None:
        """Write cache to disk every Nth write — bounds I/O cost.

        Called from match() after each cache miss. The flush itself is
        cheap-ish (~50ms for 50k entries) so we batch.
        """
        with self._match_cache_lock:
            self._match_cache_dirty_writes += 1
            if self._match_cache_dirty_writes < _BOLO_CACHE_FLUSH_EVERY:
                return
            cache = self._match_cache or {}
            fp = self._match_cache_loaded_for_fingerprint or self._fingerprint()
            self._match_cache_dirty_writes = 0
        # Drop the lock before doing I/O.
        try:
            _BOLO_MATCH_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "fingerprint": fp,
                "entries": cache,
                "saved_at": datetime.now().isoformat(),
            }
            tmp = _BOLO_MATCH_CACHE_PATH.with_suffix(".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(payload, f)
            tmp.replace(_BOLO_MATCH_CACHE_PATH)
        except OSError:
            # Disk full / read-only fs — skip flush, keep matching.
            pass

    def flush_match_cache(self) -> None:
        """Public flush — call at end of a scan to persist any pending writes."""
        with self._match_cache_lock:
            # Force flush by setting dirty_writes above threshold.
            if self._match_cache is None:
                return
            self._match_cache_dirty_writes = _BOLO_CACHE_FLUSH_EVERY
        self._maybe_flush_cache()

    def clear_match_cache(self) -> int:
        """Wipe the persistent match cache. Returns count cleared.

        Useful when the user wants to force every lot to re-match
        (e.g., debugging a brand routing issue). Doesn't touch the
        BOLO JSON files — only the cached results.
        """
        with self._match_cache_lock:
            n = len(self._match_cache or {})
            self._match_cache = {}
            self._match_cache_loaded_for_fingerprint = self._fingerprint()
            self._match_cache_dirty_writes = 0
        try:
            if _BOLO_MATCH_CACHE_PATH.exists():
                _BOLO_MATCH_CACHE_PATH.unlink()
        except OSError:
            pass
        return n

    # ---- public API -----------------------------------------------

    @property
    def loaded(self) -> bool:
        return bool(self._brand_patterns or self._model_patterns)

    @property
    def brand_count(self) -> int:
        return len(self._brand_meta)

    def match(self, title: Optional[str], description: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Return a match dict for the most-specific brand hit, or None.

        Persistent cache wrapper around `_match_uncached`. Results are
        memoized in-process AND on disk at ``.cache/bolo_match_cache.json``,
        keyed by haystack hash + matcher's combined-mtime fingerprint.
        When the user edits any BOLO JSON file, the fingerprint changes
        and the cache is treated as cold.
        """
        self._load_if_stale()
        if not self._brand_patterns and not self._model_patterns:
            return None

        haystack = " ".join(filter(None, [title or "", description or ""])).lower()
        if not haystack:
            return None

        # ---- Fast path: persistent match cache ----
        cache = self._get_match_cache()
        cache_key = self._haystack_key(haystack) if cache is not None else None
        if cache is not None and cache_key is not None:
            cached = cache.get(cache_key, _CACHE_SENTINEL)
            if cached is not _CACHE_SENTINEL:
                return cached  # may be None (negative cache) or a dict

        # Cold path: run the actual matching and write the result back.
        result = self._match_uncached(haystack)
        if cache is not None and cache_key is not None:
            cache[cache_key] = result
            self._maybe_flush_cache()
        return result

    def _match_uncached(self, haystack: str) -> Optional[Dict[str, Any]]:
        """The actual matching logic — see public `match()` for cache wrapper.

        Match rules:
          (a) An alias pattern hits AND a model pattern from the same
              brand also hits — strongest signal, returns matched_model.
          (b) An alias pattern hits with no model — returns the brand
              with matched_model = None. Still useful (the user may
              want to see "any Patagonia" hits) but lower confidence.
          (c) Skip-list dominates: if a skip entry hits and no brand
              alias hits, the haystack is treated as non-matching even
              if a model token coincidentally appears.

        Tier-1 hits beat tier-2 beat tier-3 only when the brand-name
        confidence is the same (alias+model > alias-only). Within a
        tier, first-found wins — the alias list is hand-ordered.
        """

        # Hard-skip override — these phrases zero out the lot
        # regardless of brand match (iCloud locked iPhone, parts-only
        # MacBook, blacklisted IMEI, etc.). Returns None even when
        # an alias would otherwise hit. Reserved for unambiguous
        # deal-breakers; soft skip-list (per-file skip_list) keeps
        # the existing alias-precedence behavior.
        for hard_kill in _HARD_SKIP_PATTERNS:
            if hard_kill in haystack:
                return None

        # Skip-list shortcut: but only when no brand-alias hits. We
        # don't want to skip a "Coach made in USA" lot just because
        # "modern Coach" is on the skip list — the alias-match wins.
        # Use the precompiled combined-OR regex as a fast pre-filter
        # (~10-50× faster than a Python loop over all alias patterns
        # for the no-match-anywhere case, which is most rows).
        # Candidate gate via the word-level trigger index (PERF 7/13,
        # replaces the 361KB / 5,333-alt combined regex that cost ~210ms
        # PER LOT). Tokenize the haystack ONCE; a brand is a candidate
        # only if one of its literal trigger words is present. Superset-
        # safe — each literal token IS required by its pattern, so a
        # brand with no trigger word present provably cannot match, and
        # the real boundary-correct alias_pat.search below confirms the
        # rest. Cold 82k-lot scans drop from ~55 min to seconds.
        _hay_words = set(_TRIGGER_TOKEN_RE.findall(haystack))
        _cand: set = set(self._always_test)
        for _w in _hay_words:
            _bis = self._trigger_index.get(_w)
            if _bis:
                _cand.update(_bis)

        # Pass 1: alias + model (strongest match). Iterate candidates in
        # JSON/tier order so curated tier-1 brands are evaluated first.
        # When an alias hits but no model matches, register an
        # alias-only fallback and KEEP SCANNING — a later entry might
        # have a strong (alias + model) match for the same brand family.
        # `_matched_any_alias` mirrors the old `any_alias_hit`: set the
        # instant a raw alias pattern matches (before disqualifier /
        # guard rejection), because the skip-list must be suppressed
        # whenever ANY alias matched, exactly as before.
        fallback_alias_only: Optional[Dict[str, Any]] = None
        _matched_any_alias = False
        if _cand:
            # Media context computed once per haystack (cheap regex) —
            # consumed by the apparel-category alias-only guard below.
            _has_media_context = _MEDIA_CONTEXT_RE.search(haystack) is not None
            for _bi in sorted(_cand):
                canonical, alias_pat, model_pats = self._brand_patterns[_bi]
                if not alias_pat.search(haystack):
                    continue
                _matched_any_alias = True
                # Per-brand disqualifier check: e.g., precious-metals
                # entries reject when "silver-plated" / "gold-tone" /
                # "platinum membership" appears in the haystack — those
                # are base-metal commodity / unrelated-context lots that
                # would otherwise false-positive on a metal-content alias.
                _meta = self._brand_meta.get(canonical) or {}
                _disqs = _meta.get("disqualifier_phrases") or []
                if _disqs and any(d in haystack for d in _disqs):
                    continue
                # Accessory-context guard: "case for iPad" / "lens for
                # Canon" is a third-party accessory, not a brand item.
                # Rejects the entry only when EVERY alias occurrence is
                # accessory-prefixed — applies to strong AND alias-only
                # paths ("lens for Canon EOS R5" would otherwise be a
                # strong alias+model match on the R5 model pattern).
                if _all_alias_hits_are_accessory_context(alias_pat, haystack):
                    continue
                for m_name, m_pat in model_pats:
                    if m_pat.search(haystack):
                        return self._build_match(canonical, m_name, confidence="strong")
                # Alias hit, no model — keep this as the best fallback
                # but keep scanning in case a later entry matches stronger.
                # Media-context guard: movie/TV titles collide with apparel
                # brand names ("DVD Movies - Buck, Wrangler, Jericho").
                # Alias-only apparel matches are rejected on media lots;
                # strong matches above already returned and are unaffected.
                if fallback_alias_only is None:
                    if (_has_media_context
                            and (_meta.get("category") or "")
                            in _MEDIA_GUARDED_CATEGORIES):
                        continue
                    fallback_alias_only = self._build_match(
                        canonical, None, confidence="alias_only"
                    )

        if fallback_alias_only is not None:
            return fallback_alias_only

        # Skip-list — runs only when NO alias actually matched (same as
        # the original `if not any_alias_hit` gate, just relocated to
        # after Pass 1 since the trigger index yields candidates, not a
        # definitive alias-hit boolean). A skip phrase suppresses a
        # would-be model-only (Pass 2) match; a real alias match above
        # already returned and is unaffected.
        if not _matched_any_alias:
            for sp in self._skip_patterns:
                if sp.search(haystack):
                    return None

        # Pass 2: model-only. For brands like "Vintage band tees" that
        # have no alias (because the brand IS the model — "Metallica"
        # tee, "Pink Floyd" shirt). Lower confidence — model token
        # alone in a non-clothing lot is noisy. We require at least
        # one tee/shirt context word in the haystack to fire.
        if any(w in haystack for w in (" tee", "t-shirt", "tshirt", " shirt", "tour shirt", "concert")):
            for canonical, m_name, m_pat in self._model_pats_band_tee:
                if m_pat.search(haystack):
                    return self._build_match(canonical, m_name, confidence="model_only")

        # Pass 2e: Precious-metal model-only.
        # Titles like "Taxco Mexican sterling brooch" or "Sterling
        # spoon Gorham" don't contain the alias phrase "sterling
        # silver" / ".925" — they use the bare word "sterling" plus
        # a jewelry/flatware context. The model list in the JSON
        # ("Sterling chain", "Taxco sterling", "Sterling spoon",
        # etc.) covers these compound forms. Fire model-only matching
        # for precious_metal-category brands so they catch the
        # implicit cases. Per-brand disqualifier check still applies
        # so silver-plate / gold-tone / platinum-membership false
        # positives stay rejected.
        if _hay_words & self._precious_trigger_words:
            for canonical, m_name, m_pat in self._model_pats_by_cat.get('precious_metal', ()):
                _meta = self._brand_meta.get(canonical) or {}
                _disqs = _meta.get("disqualifier_phrases") or []
                if _disqs and any(d in haystack for d in _disqs):
                    continue
                if m_pat.search(haystack):
                    return self._build_match(
                        canonical, m_name, confidence="model_only"
                    )

        # Pass 2f: Musical-instrument model-only.
        # Many titles include the model name without a brand-aliased
        # phrase — "TR-606 vintage drum machine", "M44-7 cartridge",
        # "Tube Screamer pedal", "TS-808 vintage". Fire model-only
        # matching for the musical_instrument category gated on
        # instrument-context words so a "Roland TR-606" tractor or
        # something equally weird doesn't false-positive.
        #
        # Negative guards: skip lots that are clearly the BIG-format
        # versions we explicitly excluded (electric guitar, sax body,
        # full drum kit, amp head). Even if a model word matches
        # somewhere in the title, those are not what we want.
        _music_skip = (
            "electric guitar" in haystack
            or "acoustic guitar" in haystack
            or "bass guitar" in haystack
            or " guitar body" in haystack
            or " saxophone" in haystack
            or " trumpet" in haystack
            or " trombone" in haystack
            or " drum kit" in haystack
            or " drum set" in haystack
            or " amplifier head" in haystack
            or "amp combo" in haystack
            or " tuba" in haystack
            or " french horn" in haystack
            or " cello" in haystack
            or " viola" in haystack
        )
        if not _music_skip and any(w in haystack for w in (
            "harmonica", "microphone", " mic ", " mic,", " mic.",
            "pedal", "stompbox", "stomp box",
            "mouthpiece", "pickup", "humbucker", " single coil",
            "music box", "cylinder box",
            "drum machine", "groovebox", "synthesizer", " synth ",
            "sampler", "sequencer",
            "cartridge", "needle", "stylus", "turntable",
            "ukulele",
        )):
            for canonical, m_name, m_pat in self._model_pats_by_cat.get('musical_instrument', ()):
                if m_pat.search(haystack):
                    return self._build_match(
                        canonical, m_name, confidence="model_only"
                    )

        # Pass 2c: Watch-accessory model-only. Auction listings often
        # describe luxury watch accessories without mentioning the
        # brand — they just say "Submariner box", "Speedmaster papers",
        # "Daytona bracelet", "Nautilus papers", etc. The model name
        # alone is sufficient signal because these names are
        # essentially trademarks (no other brand makes a "Submariner"
        # or "Speedmaster"). Gate on accessory-context words
        # (box / papers / bracelet / certificate / warranty / pouch /
        # winder / hangtag) so a "Submariner watch" alone doesn't
        # match (that's the watch itself, which is a different
        # category — we want the accessory).
        #
        # Negative guards: skip generic display boxes / kid's
        # toys / aftermarket organizers, which are NOT luxury OEM
        # accessories.
        _watch_acc_skip = (
            "watch box organizer" in haystack
            or "watch box display" in haystack
            or "watch case organizer" in haystack
            or "for 6 watches" in haystack
            or "for 12 watches" in haystack
            or "for 24 watches" in haystack
            or "watch box for" in haystack
            or "kids watch" in haystack
            or "toy watch" in haystack
            or "aftermarket box" in haystack
        )
        if not _watch_acc_skip and any(w in haystack for w in (
            " box", " papers", " certificate", " warranty",
            " bracelet", " pouch", " winder", " hangtag",
            " hang tag", " booklet", " case",
        )):
            for canonical, m_name, m_pat in self._model_pats_watch_acc:
                if m_pat.search(haystack):
                    return self._build_match(
                        canonical, m_name, confidence="model_only"
                    )

        # Pass 2b: Loungefly model-only. Auction listings frequently
        # describe Loungefly products without mentioning the brand
        # name — they just say "Disney mini backpack", "Stitch
        # backpack", "Marvel mini backpack", etc. Loungefly is the
        # dominant maker of licensed-character mini backpacks, so
        # these implicit listings are almost always Loungefly. We
        # gate on backpack / wallet / crossbody context words to
        # prevent matching against unrelated character merchandise
        # (a "Disney Stitch tumbler" shouldn't match Loungefly).
        #
        # Negative guard: skip when the title clearly describes a
        # kid/toddler/school backpack — those are licensed-character
        # backpacks but NOT Loungefly (Loungefly is adult-oriented
        # PU faux-leather construction).
        _loungefly_skip = (
            "toddler" in haystack
            or "kindergarten" in haystack
            or " preschool" in haystack
            or "kids backpack" in haystack
            or "kid's backpack" in haystack
            or "children's backpack" in haystack
            or "school backpack" in haystack
            or "rolling backpack" in haystack
            or "bookbag" in haystack
            or "lunch box" in haystack
        )
        if not _loungefly_skip and any(w in haystack for w in (
            "backpack", "mini back pack", "wallet",
            "crossbody", "cross body", "cross-body",
        )):
            # Per-brand disqualifier check: prevents "Loungefly inspired",
            # "fits Loungefly", "Loungefly logo sticker pack",
            # "Loungefly enamel pin only" etc. from false-positive
            # matching via Pass 2b's model-only path.
            _lf_meta = self._brand_meta.get("Loungefly") or {}
            _lf_disqs = _lf_meta.get("disqualifier_phrases") or []
            _lf_blocked = _lf_disqs and any(d in haystack for d in _lf_disqs)
            if not _lf_blocked:
                for canonical, m_name, m_pat in self._model_pats_loungefly:
                    if m_pat.search(haystack):
                        return self._build_match(
                            canonical, m_name, confidence="model_only"
                        )

        return None

    def _build_match(self, canonical: str, model: Optional[str],
                     confidence: str) -> Dict[str, Any]:
        meta = self._brand_meta.get(canonical, {})
        return {
            "brand": canonical,
            "matched_model": model,
            "confidence": confidence,
            "tier": meta.get("tier"),
            "category": meta.get("category"),
            "platform_primary": meta.get("platform_primary"),
            "platform_secondary": meta.get("platform_secondary"),
            "ship_class": meta.get("ship_class"),
            "target_buy_low": meta.get("target_buy_low"),
            "target_buy_high": meta.get("target_buy_high"),
            "notes": meta.get("notes"),
            "era_markers": meta.get("era_markers") or [],
            "auth_required": bool(meta.get("auth_required")),
        }
