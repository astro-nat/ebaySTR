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

import json
import re
import threading
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
    Path("data") / "computer_parts_bolo.json",
    Path("data") / "apple_products_bolo.json",
    Path("data") / "golf_equipment_bolo.json",
    Path("data") / "audio_watches_bolo.json",
    Path("data") / "watch_accessories_bolo.json",
    Path("data") / "musical_instruments_bolo.json",
    Path("data") / "camera_equipment_bolo.json",
    Path("data") / "auto_parts_bolo.json",
    Path("data") / "lightweight_collectibles_bolo.json",
    Path("data") / "estate_collectibles_bolo.json",
    # Precious-metals loads LAST so brand-specific entries (Tiffany,
    # Cartier, Native American jewelry) take precedence — those have
    # higher resale ceilings than the generic precious-metal floor.
    Path("data") / "precious_metals_bolo.json",
]
# Back-compat single-path constant for callers that import it.
DEFAULT_BOLO_PATH = DEFAULT_BOLO_PATHS[0]


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


_BRAND_ALIASES: Dict[str, List[str]] = {
    # JSON header → list of literal phrases to match in the haystack
    "Loungefly":                              ["loungefly"],
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
    ],
    "Effects pedals (guitar/bass)":           [
        "boss pedal", "boss compact",
        "mxr", "ibanez", "electro-harmonix",
        "klon centaur", "strymon", "eventide",
        "empress effects", "chase bliss",
        "jhs pedals", "earthquaker devices",
        "walrus audio", "proco rat",
        # Bare "ibanez" is fine (most contexts are guitar/pedal),
        # bare "boss" is too generic so we use compound aliases.
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
    ],
    "Platinum":                               [
        "pt950", "pt900", "950 platinum", "900 platinum",
        "platinum ring", "platinum band", "platinum chain",
        "platinum necklace", "platinum bracelet",
        "platinum earring", "platinum pendant",
        "platinum wedding band", "platinum engagement",
        "iridplat", "iridium platinum", "solid platinum",
    ],
    # Watch accessories — luxury-only, brand-name aliases. Implicit
    # matching (e.g., "Submariner box" without "Rolex" in title) is
    # handled by the Pass 2c branch in the matcher.
    "Rolex accessories":                      ["rolex", "tudor"],
    "Omega accessories":                      ["omega"],
    "Patek Philippe accessories":             ["patek philippe", "patek"],
    "Audemars Piguet accessories":            ["audemars piguet", " ap "],
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
    "Pyrex vintage":                          ["pyrex"],
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
        "corsair rm", "corsair hx", "corsair ax",
        "corsair icue", "corsair crystal", "corsair obsidian",
        "corsair h100", "corsair h115", "corsair h150", "corsair h170",
        "corsair hydro", "corsair sf",
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
        "arctic freezer 7", "arctic bionix", "arctic p12 pwm",
        "arctic p14 pwm",
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
        "hot wheels",  # broad fallback (alias_only; specific aliases above route to strong match)
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
        self._load_if_stale()

    # ---- file I/O -------------------------------------------------

    def _load_if_stale(self) -> None:
        """Reload the JSON files when any source mtime changes.

        Cheap enough to run on every match() call — st.cache_data isn't
        appropriate here because the cache should invalidate on file
        modification, not session boundary. With multiple files, any
        single file changing triggers a full rebuild.
        """
        current_mtimes: Dict[str, float] = {}
        for p in self.paths:
            try:
                current_mtimes[str(p)] = p.stat().st_mtime
            except OSError:
                # Missing file is OK — just skip it. The other file(s)
                # still load. If ALL files are missing the matcher is
                # empty and match() returns None.
                continue
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

        self._raw = loaded_raw
        self._brand_patterns = brand_patterns
        self._model_patterns = model_patterns
        self._skip_patterns = skip_patterns
        self._brand_meta = brand_meta
        self._mtimes = mtimes

    # ---- public API -----------------------------------------------

    @property
    def loaded(self) -> bool:
        return bool(self._brand_patterns or self._model_patterns)

    @property
    def brand_count(self) -> int:
        return len(self._brand_meta)

    def match(self, title: Optional[str], description: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Return a match dict for the most-specific brand hit, or None.

        A "match" requires either:
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
        self._load_if_stale()
        if not self._brand_patterns and not self._model_patterns:
            return None

        haystack = " ".join(filter(None, [title or "", description or ""])).lower()
        if not haystack:
            return None

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
        any_alias_hit = any(
            ap.search(haystack) for _, ap, _ in self._brand_patterns
        )
        if not any_alias_hit:
            for sp in self._skip_patterns:
                if sp.search(haystack):
                    return None

        # Pass 1: alias + model (strongest match). Iterate in JSON
        # order so the curated tier-1 brands get evaluated first.
        # When an alias hits but no model matches, register an
        # alias-only fallback and KEEP SCANNING — a later entry
        # might have a strong (alias + model) match for the same
        # brand family. Common case: two entries share an alias
        # ("KitchenAid" → stand mixer + appliance parts) and we
        # want to surface whichever model list matches the title.
        # Strong always beats alias-only.
        fallback_alias_only: Optional[Dict[str, Any]] = None
        for canonical, alias_pat, model_pats in self._brand_patterns:
            if not alias_pat.search(haystack):
                continue
            # Per-brand disqualifier check: e.g., precious-metals
            # entries reject when "silver-plated" / "gold-tone" /
            # "platinum membership" appears in the haystack — those
            # are base-metal commodity / unrelated-context lots that
            # would otherwise false-positive on a metal-content alias.
            _meta = self._brand_meta.get(canonical) or {}
            _disqs = _meta.get("disqualifier_phrases") or []
            if _disqs and any(d in haystack for d in _disqs):
                continue
            for m_name, m_pat in model_pats:
                if m_pat.search(haystack):
                    return self._build_match(canonical, m_name, confidence="strong")
            # Alias hit, no model — keep this as the best fallback
            # but keep scanning in case a later entry matches stronger.
            if fallback_alias_only is None:
                fallback_alias_only = self._build_match(
                    canonical, None, confidence="alias_only"
                )

        if fallback_alias_only is not None:
            return fallback_alias_only

        # Pass 2: model-only. For brands like "Vintage band tees" that
        # have no alias (because the brand IS the model — "Metallica"
        # tee, "Pink Floyd" shirt). Lower confidence — model token
        # alone in a non-clothing lot is noisy. We require at least
        # one tee/shirt context word in the haystack to fire.
        if any(w in haystack for w in (" tee", "t-shirt", "tshirt", " shirt", "tour shirt", "concert")):
            for canonical, m_name, m_pat in self._model_patterns:
                # Only fire model-only for the "Vintage band tees" /
                # "Vintage movie/TV/brand promo" entries — every other
                # brand has its name in the haystack anyway and pass 1
                # would have caught it.
                if "Vintage" not in canonical or "tee" not in canonical.lower():
                    continue
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
        for canonical, m_name, m_pat in self._model_patterns:
            _meta = self._brand_meta.get(canonical) or {}
            if _meta.get("category") != "precious_metal":
                continue
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
            for canonical, m_name, m_pat in self._model_patterns:
                _meta = self._brand_meta.get(canonical) or {}
                if _meta.get("category") != "musical_instrument":
                    continue
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
            for canonical, m_name, m_pat in self._model_patterns:
                if "accessories" not in (canonical or ""):
                    continue
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
            for canonical, m_name, m_pat in self._model_patterns:
                if canonical != "Loungefly":
                    continue
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
