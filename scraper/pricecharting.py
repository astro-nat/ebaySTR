"""PriceCharting API integration — pricing for video games, trading cards, comics.

PriceCharting aggregates sold-listing history from eBay, Amazon, online retailers
and trading card marketplaces. For lots that are clearly games / cards / comics,
their data is materially better than scraping eBay sold for ourselves: prices
are normalized by condition (loose / CIB / new), updated continuously, and
matched to a canonical product ID rather than fuzzy keyword search.

API spec
--------
- Auth: `?t=API_TOKEN` query param (Premium subscription required)
- Search:  GET https://www.pricecharting.com/api/products?t=TOKEN&q=KEYWORDS
- Product: GET https://www.pricecharting.com/api/product?t=TOKEN&q=KEYWORDS  (first match)
- Prices come back in CENTS (loose-price, cib-price, new-price, etc.)
"""
import re
import time
import threading
from typing import Optional, Dict, Any

import httpx


_BASE_URL = "https://www.pricecharting.com/api"


# Lots that should be tried against PriceCharting. High-precision keywords —
# we'd rather skip a PC-eligible lot than waste tokens on a "Mario-themed
# coffee mug" lot. The classifier returns a category hint we can use for
# logging but PriceCharting's search itself is category-agnostic.
_PC_TRIGGERS = [
    # Order matters — first category match wins. We list comic BEFORE
    # trading_card so a 'Marvel Comics ... CGC 9.8' title doesn't get
    # claimed by trading_card's bare ` cgc ` trigger first.
    # ---- Video game consoles & branded handhelds ----
    ("video_game", [
        "nintendo 64", " n64 ", "n64,", "n64 ",
        "playstation", " ps1 ", " ps2 ", " ps3 ", " ps4 ", " ps5 ",
        "xbox", "xbox 360", "xbox one", "xbox series",
        "gamecube", "game cube",
        "nintendo switch", " switch lite ", " switch oled ",
        "nintendo wii", " wii u ",
        "nintendo ds", " 3ds ", " 2ds ",
        "game boy", "gameboy", "gba ",
        "sega genesis", "sega saturn", "sega cd", "sega dreamcast",
        " snes ", "super nintendo", " nes ",
        "atari 2600", "atari 5200", "atari 7800", "atari jaguar", "atari lynx",
        "video game", "videogame",
        " rom cartridge", "game cartridge",
    ]),
    # ---- Comics ----
    # The bare "comic book" / "comic books" triggers used to live here
    # but they false-matched indie/art-book lots ("Superhumanity Super
    # Australians Comic Book Vol 1" was a $70 phantom). Almost every
    # legitimate single-comic lot also carries a publisher name or a
    # CGC/CBCS grade callout, so requiring those is a much sharper
    # signal.
    ("comic", [
        "marvel comics", "dc comics", "image comics",
        "dark horse comics", "idw comics", "boom! studios",
        "vertigo comics",
        "cgc comic", "cgc 9.", "cgc 8.", " cbcs ",
    ]),
    # ---- TCGs (Pokemon, Magic, Yu-Gi-Oh, sealed product) ----
    # PriceCharting's pricing model maps cleanly here: loose-price IS
    # the raw single-card / sealed-pack price, cib/new are alternates.
    ("tcg", [
        "pokemon", "pokémon", "pokemon card",
        "magic the gathering", " mtg ", "mtg booster",
        "yugioh", "yu-gi-oh", "yu gi oh",
        "trading card", " tcg ", "tcg booster",
        "booster pack", "booster box",
    ]),
    # ---- Sports cards (Topps, Panini, Bowman, etc.) ----
    # Audited 5/3 — PriceCharting returns inflated graded-tier prices
    # in the loose-price field for older popular cards (1990 Fleer
    # #513 Griffey returned $600 from PC vs $1.72-$3.00 raw on eBay;
    # 2023 Panini Absolute #100 DeVonta Smith $106 vs $0.79). The
    # loose/cib/new field model doesn't map to raw/graded for sports
    # cards. classify_for_pricecharting still tags these so they bypass
    # the AI condition audit, but PriceChartingLookup.lookup() returns
    # None for sports_card so the eBay sold-listing scraper handles
    # them — eBay has plenty of raw-card sales and IQR outlier filtering
    # produces a sensible median.
    ("sports_card", [
        # Brand names — matched as space-padded tokens so they don't
        # false-match unrelated words.
        " topps ", " bowman ", " donruss ", " fleer ", " panini ",
        " upper deck ", " stadium club ", " prizm ",
        " select baseball ", " select football ", " select basketball ",
        # Grading callouts. Universal across TCG and sports cards but
        # since they always indicate "this is a card lot", grouping with
        # sports_card is fine — PC is suppressed for both anyway when
        # the lot is graded (graded cards need eBay's actual graded-comp
        # data, not PC's tier estimate).
        " psa ", " bgs ", " sgc ", " cgc ",
        "psa 10", "psa 9", "bgs graded", "psa graded", "cgc graded",
        # Rookie card markers — "rookie card" only (the bare word
        # "rookie" matches "rookie season"/"of the year").
        "rookie card",
        # Card-specific markers
        "card #", " autograph ", " refractor ",
    ]),
]


# Anti-triggers — words that strongly indicate the lot is a physical
# collectible, NOT a tradable card / game / comic. If any of these
# appear in the title we suppress the classification entirely. The
# motivation: PriceCharting will obligingly return a product match
# even when the lot isn't really one of its products (e.g. a Pokemon
# Nanoblock toy returned $110 because PC matched it to some Pokemon
# card). PC has no idea Nanoblocks exist; the price is noise.
_PC_ANTI_TRIGGERS = [
    # Building toys
    "nanoblock", "lego ", "mega bloks", "minifig", "minifigure",
    # Figurines / statues
    "figurine", "figurines", "statue", "statuette", " bust ",
    "funko", "pop figure", "pop! figure", "pop vinyl",
    "iron studios", "minico",
    # Plush
    "plush", "plushie", "plushies", "stuffed animal",
    # Paper / 3D toys
    "puzzle", "jigsaw",
    # Music / audio
    "vinyl ", " lp ", "lp record", "soundtrack",
    # Wall / room decor
    "poster", "wall art", "tapestry",
    # Wearables
    " shirt", "t-shirt", "hoodie", " hat ", " cap ",
    "blanket", " towel",
    # Drinkware
    " mug ", "tumbler", "water bottle",
    # Smaller merch
    "keychain", "lanyard", "enamel pin", "pin set",
    # Stationery
    "notebook", "journal", " sticker",
]



def classify_for_pricecharting(title: str) -> Optional[str]:
    """Return category label ('video_game' / 'trading_card' / 'comic') if
    the title is likely a PriceCharting-covered item, else None.

    Pads with spaces on both ends so word-boundary matching works for the
    one-letter abbreviations like ' ps2 ' that can't be safely searched
    without surrounding whitespace.

    Anti-triggers run first: a single physical-collectible word
    (nanoblock, statue, plush, vinyl, funko, etc.) suppresses the
    classification regardless of any matching category trigger. PC's
    database doesn't cover those products, and a stray product match
    on a related card/game is just noise.
    """
    if not title:
        return None
    padded = f" {title.lower()} "
    for kw in _PC_ANTI_TRIGGERS:
        if kw in padded:
            return None
    for category, keywords in _PC_TRIGGERS:
        for kw in keywords:
            if kw in padded:
                return category
    return None


class PriceChartingLookup:
    """Thin client over PriceCharting's `/api/product` endpoint.

    Token comes from `pricecharting.token` in config.json (or st.secrets).
    Without a token, every method returns None — call sites can safely
    construct a no-op instance with `PriceChartingLookup(None)`.
    """

    def __init__(self, token: Optional[str]):
        self.token = token
        self._lock = threading.Lock()
        # Tiny in-process cache so repeated calls during a single comp run
        # don't hammer the API. PriceCharting prices barely change minute-
        # to-minute and we routinely re-scan the same lots.
        self._cache: Dict[str, Optional[Dict[str, Any]]] = {}

    @property
    def enabled(self) -> bool:
        return bool(self.token)

    @staticmethod
    def _clean_query(title: str) -> str:
        """Strip price hints, retail boilerplate, and condition noise.

        PriceCharting's search is far smarter than eBay's, so we don't
        need progressive shortening — but we still strip the obvious
        garbage so the query matches a canonical product.
        """
        clean = re.sub(r'\$\d+(?:\.\d{1,2})?\b', '', title)
        clean = re.sub(
            r'\b(retail(\s+value)?|msrp|est(\.|imated)?\s*(value|worth))\b',
            '', clean, flags=re.IGNORECASE,
        )
        clean = re.sub(r'\bQty[:\-]?\s*\d+\s*', '', clean, flags=re.IGNORECASE)
        clean = re.sub(r'\([^)]{1,25}\)', '', clean)
        # Strip dashes — PriceCharting's product matcher treats them as
        # token separators which can cause partial-match weirdness, and
        # mirrors the same fix applied to eBay queries.
        clean = re.sub(r'[,;:/\\|\-]+', ' ', clean)
        clean = re.sub(r'\s+', ' ', clean).strip(' .,-')
        return clean

    def lookup(self, title: str, timeout: float = 8.0) -> Optional[Dict[str, Any]]:
        """Fetch the best-matching product from PriceCharting.

        Returns a dict in the same shape as `EbayPriceLookup.lookup_price_range`
        so the caller can plug it into the existing pricing pipeline:
            {median, low, high, count, source, ebay_count, mercari_count,
             pricecharting_count, query, pc_product, pc_console}
        Returns None when the token isn't set, the title doesn't classify,
        or PriceCharting has no match.

        Pricing model: median = loose-price (most realistic flip outcome),
        low = box-only or 70% of loose, high = cib-price (or new-price if
        cib is missing). PriceCharting prices are pre-aggregated across
        many sales, so a single match counts as high-confidence — it isn't
        a single eBay listing.
        """
        if not self.enabled:
            return None

        category = classify_for_pricecharting(title)
        if not category:
            return None

        # Sports cards bypass PriceCharting entirely. PC's pricing fields
        # (loose / cib / new) don't map to raw vs graded condition for
        # sports cards — older popular cards return graded-tier prices
        # in the loose-price field, producing wildly inflated estimates
        # ($600 for a 1990 Fleer Griffey raw card that sells for $2 on
        # eBay; $106 for a $1 DeVonta Smith base card). The eBay sold
        # scraper handles these correctly with plenty of raw-card comps.
        if category == "sports_card":
            return None

        query = self._clean_query(title)
        if not query:
            return None

        cache_key = query.lower()
        with self._lock:
            if cache_key in self._cache:
                return self._cache[cache_key]

        try:
            resp = httpx.get(
                f"{_BASE_URL}/product",
                params={"t": self.token, "q": query},
                timeout=timeout,
            )
        except Exception:
            with self._lock:
                self._cache[cache_key] = None
            return None

        if resp.status_code != 200:
            with self._lock:
                self._cache[cache_key] = None
            return None

        try:
            data = resp.json()
        except ValueError:
            with self._lock:
                self._cache[cache_key] = None
            return None

        if data.get("status") != "success":
            with self._lock:
                self._cache[cache_key] = None
            return None

        # Cents → dollars. Any field can be missing for niche products.
        loose = _cents_to_dollars(data.get("loose-price"))
        cib = _cents_to_dollars(data.get("cib-price"))
        new = _cents_to_dollars(data.get("new-price"))
        box_only = _cents_to_dollars(data.get("box-only-price"))

        # Prefer loose (used) as the headline — it's what most flippers
        # realize from a mixed lot. Fall back to CIB then new.
        median = loose or cib or new
        if median is None:
            with self._lock:
                self._cache[cache_key] = None
            return None

        low = box_only or (loose and round(loose * 0.7, 2)) or median
        high = new or cib or median

        # Sanity: low ≤ median ≤ high
        low, high = min(low, median), max(high, median)

        # Throttle a touch so a 200-lot batch doesn't burst-fire the API.
        time.sleep(0.15)

        result = {
            "median": round(median, 2),
            "low": round(low, 2),
            "high": round(high, 2),
            "count": 1,
            "source": f"pricecharting ({category}, loose)",
            "ebay_count": 0,
            "mercari_count": 0,
            "pricecharting_count": 1,
            "query": query,
            "pc_product": data.get("product-name", ""),
            "pc_console": data.get("console-name", ""),
            "pc_id": data.get("id", ""),
        }
        with self._lock:
            self._cache[cache_key] = result
        return result


def _cents_to_dollars(value) -> Optional[float]:
    """PriceCharting returns prices in cents as ints. Missing fields are
    sometimes 0, sometimes None — treat 0 as missing too (a $0.00 game
    is a missing price, not a real one)."""
    if value in (None, 0, "0", ""):
        return None
    try:
        return round(int(value) / 100, 2)
    except (TypeError, ValueError):
        return None
