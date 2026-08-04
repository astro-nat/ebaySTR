import asyncio
import httpx
import pandas as pd
import re
import json
import os
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Optional

from scraper._ssl_compat import make_ssl_context

# Built once at import — handed to every httpx client in this module.
# On Windows boxes with Norton/Kaspersky/corp-proxy TLS inspection, this
# is a truststore-backed OS-native context; otherwise it's True (certifi).
_HTTPX_VERIFY = make_ssl_context()


# HiBid lot.lotState.timeLeft strings look like:
#   "2d 5h 30m"  /  "5h 30m"  /  "30m 12s"  /  "Bidding Closed"
# Compact regex: capture an optional integer + a unit letter for each
# of d/h/m/s. Missing units default to 0. Returns a timedelta or None.
_TIMELEFT_RE = re.compile(
    r'(?:(?P<d>\d+)\s*d)?\s*(?:(?P<h>\d+)\s*h)?\s*'
    r'(?:(?P<m>\d+)\s*m)?\s*(?:(?P<s>\d+)\s*s)?',
    re.IGNORECASE,
)


def _parse_time_left(time_left: str) -> Optional[timedelta]:
    """Parse a HiBid timeLeft string into a timedelta.

    Returns None for closed/empty/unparseable strings. Used as a
    fallback closing-time signal when the auction's eventDateEnd is
    missing — every individual lot still carries `timeLeft` and we
    can derive an auction-level closing time from the first lot.
    """
    if not time_left or not isinstance(time_left, str):
        return None
    s = time_left.strip().lower()
    if not s or 'closed' in s:
        return None
    m = _TIMELEFT_RE.fullmatch(s)
    if not m:
        return None
    parts = m.groupdict()
    if not any(parts.values()):
        return None
    return timedelta(
        days=int(parts['d'] or 0),
        hours=int(parts['h'] or 0),
        minutes=int(parts['m'] or 0),
        seconds=int(parts['s'] or 0),
    )


def _derive_auction_closing_from_lots(lots: List[Dict]) -> Optional[datetime]:
    """Compute an auction-level closing datetime from its lots.

    Returns the FIRST lot's closing time (lots in HiBid auctions
    typically close in lot-number order, with subsequent lots
    spaced 10-30 seconds apart). Falls back to scanning subsequent
    lots if the first lot's `time_left` doesn't parse. Returns
    None if no lot in the auction has a parseable timeLeft.
    """
    now = datetime.now()
    for lot in lots:
        delta = _parse_time_left(lot.get('time_left') or lot.get('timeLeft', ''))
        if delta is not None:
            return now + delta
    return None

AUCTION_MAP_QUERY = """
query AuctionMap($zip: String, $miles: Int, $searchText: String, $categoryId: CategoryId, $filter: AuctionLotFilter, $status: AuctionLotStatus, $eventIds: [Int!] = null) {
  auctionMap(
    input: {zip: $zip, miles: $miles, searchText: $searchText, category: $categoryId, filter: $filter, status: $status, eventIds: $eventIds}
  ) {
    mapMarkers {
      auction {
        id
        eventName
        auctioneer { name __typename }
        lotCount
        geoLong
        geoLat
        eventAddress
        eventCity
        eventZip
        eventState
        eventDateBegin
        eventDateInfo
        eventDateEnd
        __typename
      }
      __typename
    }
    __typename
  }
}
"""

# Per-auction commercial terms, fetched by eventIds at Phase-1 time for
# just the selected auctions (cheap — one call per 50 auctions).
#   buyerPremium          — free-text, e.g. "10% Buyers Premium with Cash
#                           & Check", "No Buyer's Premium", "18%"
#   buyerPremiumRate      — structured multiplier, but UNRELIABLE: most
#                           auctioneers leave it at 1 even when the text
#                           says 19-22%. Only trusted when > 1.
#   shippingAndPickupInfo — the "Shipping / Pick Up" section from the
#                           auction page. Source for conditional-shipping
#                           detection + shipping-cost hints.
AUCTION_META_QUERY = """
query AuctionMeta($eventIds: [Int!]) {
  auctionMap(input: {zip: "", miles: 0, searchText: "", category: -1, filter: ALL, status: ALL, eventIds: $eventIds}) {
    mapMarkers {
      auction {
        id
        buyerPremium
        buyerPremiumRate
        shippingAndPickupInfo
        termsAndConditions
      }
    }
  }
}
"""

# --- Buyer-premium text parsing -------------------------------------
_BP_PCT_RE = re.compile(r"(\d{1,2}(?:\.\d{1,2})?)\s*%")
_NO_BP_RE = re.compile(r"\bno\s+buyer'?s?\s+premium\b", re.IGNORECASE)


def parse_buyer_premium_pct(text, rate=None):
    """Parse a buyer-premium multiplier (1.10 for '10%') from HiBid data.

    Precedence:
      1. "No Buyer's Premium" text → 1.0
      2. First percentage in the text → 1 + pct/100. When the text
         lists cash vs card tiers ("10% with Cash & Check"), the first
         (lower/cash) number wins — matches a cash-paying buyer.
      3. buyerPremiumRate when > 1 (rate == 1 means "field not filled"
         for most auctioneers, NOT "no premium" — 19% and 22% auctions
         ship with rate=1).
      4. None — caller falls back to the config default.
    """
    t = (text or "").strip()
    if t:
        if _NO_BP_RE.search(t):
            return 1.0
        m = _BP_PCT_RE.search(t)
        if m:
            pct = float(m.group(1))
            if 0 <= pct <= 50:
                return 1.0 + pct / 100.0
    try:
        r = float(rate or 0)
        if r > 1.0:
            return r
    except (TypeError, ValueError):
        pass
    return None


# Premium-from-terms fallback. Some auctioneers put only boilerplate
# in buyerPremium ("BUYERS PREMIUM APPLIES TO ALL PURCHASES") and bury
# the actual number in termsAndConditions ("ALL ITEMS HAVE A 10% (TEN
# PERCENT) BUYERS PREMIUM AND 6% PA SALES TAX" — Johnston 752316).
# The % must be ADJACENT to the words "buyer's premium" in either
# order, so the 6% sales tax two clauses over doesn't get grabbed.
_BP_NEAR_TERMS_RES = (
    # "10% ... buyers premium" — percent first
    re.compile(
        r"(\d{1,2}(?:\.\d{1,2})?)\s*%[^%]{0,45}?buyer'?s?\s+premium",
        re.IGNORECASE,
    ),
    # "buyers premium of 15%" — premium first
    re.compile(
        r"buyer'?s?\s+premium[^.%\n]{0,35}?(\d{1,2}(?:\.\d{1,2})?)\s*%",
        re.IGNORECASE,
    ),
)


def parse_premium_from_terms(terms_text):
    """Extract a buyer-premium multiplier from termsAndConditions text.

    Returns 1 + pct/100 or None. Strips HTML first (terms are often
    rich text). Only fires on a % within ~45 chars of the phrase
    "buyer(')s premium" so unrelated percentages (sales tax, card
    fees) don't false-match.
    """
    if not terms_text:
        return None
    clean = re.sub(r"<[^>]+>", " ", str(terms_text))
    clean = re.sub(r"\s+", " ", clean)
    for pat in _BP_NEAR_TERMS_RES:
        m = pat.search(clean)
        if m:
            try:
                pct = float(m.group(1))
            except (TypeError, ValueError):
                continue
            if 0 <= pct <= 50:
                return 1.0 + pct / 100.0
    return None


# --- shippingAndPickupInfo parsing -----------------------------------
_FREE_SHIP_RE = re.compile(r"free\s+(?:domestic\s+)?shipping", re.IGNORECASE)
_USPS_PRIORITY_RE = re.compile(r"usps\s+priority", re.IGNORECASE)
# Estimated auctioneer-shipped cost when terms mention USPS Priority
# boxes but no dollar figure — medium flat-rate box + modest handling.
USPS_PRIORITY_SHIP_ESTIMATE = 17.0
# "Shipping is NOT available on all lots" / "contact us prior to
# bidding to confirm shipping" style language → shipping is per-lot
# conditional; the Ship/Local Pickup source classification is soft.
_COND_SHIP_RE = re.compile(
    r"not\s+available\s+(?:on|for)\s+all\s+(?:lots|items)"
    r"|contact\s+[^.\n]{0,80}?(?:prior\s+to|before)\s+bidding"
    r"|shipping\s+(?:available\s+)?on\s+select(?:ed)?\s+(?:lots|items)"
    r"|some\s+(?:lots|items)\s+(?:cannot|can\s*not|won'?t)\s+be\s+shipped"
    # "PLEASE DO NOT ASSUME ALL ITEMS ARE SHIPPABLE" (Johnston 752316)
    r"|do\s+not\s+assume\s+all\s+(?:items|lots)\s+are\s+shippable"
    r"|not\s+all\s+(?:items|lots)\s+(?:are\s+)?shippable"
    # "email auctioneer with any/all shipping related questions"
    r"|email\s+[^.\n]{0,40}?shipping\s+related\s+questions",
    re.IGNORECASE,
)
_SHIP_DOLLAR_RE = re.compile(
    r"ship\w*[^$\n]{0,40}\$\s*(\d{1,3}(?:\.\d{2})?)"
    r"|\$\s*(\d{1,3}(?:\.\d{2})?)[^.\n]{0,30}ship",
    re.IGNORECASE,
)


def parse_shipping_info(ship_text):
    """Extract (cond_ship: bool, ship_hint: float|None) from the
    auction's shippingAndPickupInfo blurb.

    ship_hint precedence: free-shipping language → 0.0; explicit
    dollar figure near 'ship' → that amount; USPS-Priority mention
    → USPS_PRIORITY_SHIP_ESTIMATE; otherwise None (caller uses the
    config default).
    """
    t = ship_text or ""
    cond = bool(_COND_SHIP_RE.search(t))
    hint = None
    if _FREE_SHIP_RE.search(t):
        hint = 0.0
    else:
        m = _SHIP_DOLLAR_RE.search(t)
        if m:
            try:
                v = float(m.group(1) or m.group(2))
                if 0.5 <= v <= 200:
                    hint = v
            except (TypeError, ValueError):
                pass
        if hint is None and _USPS_PRIORITY_RE.search(t):
            hint = USPS_PRIORITY_SHIP_ESTIMATE
    return cond, hint


LOT_SEARCH_QUERY = """
query LotSearch($auctionId: Int!, $pageNumber: Int!, $searchText: String!) {
  lotSearch(input: {auctionId: $auctionId, searchText: $searchText}, pageNumber: $pageNumber) {
    pagedResults {
      totalCount
      pageNumber
      results {
        id
        lotNumber
        lead
        description
        estimate
        category { categoryName }
        lotState { highBid minBid bidCount status timeLeft }
        pictures { thumbnailLocation hdThumbnailLocation fullSizeLocation }
        shippingOffered
      }
    }
  }
}
"""

# HiBid's current GraphQL schema (Apr 2026): `pageNumber` is a sibling
# argument to `input` (not inside it), and page size is fixed at 100 lots.
# The old `pageSize` / `pageIndex` input fields were removed, which is what
# caused the HTTP 400 "Unknown field" errors.
LOT_PAGE_SIZE = 100             # fixed by the server
# HiBid caps pagination at page 100 (i.e. 10,000 lots max, confirmed via probe
# against auction 734754 which has 10,817 lots). Going beyond page 100 returns
# an empty batch. MAX_LOT_PAGES stays slightly above 100 in case the server
# cap shifts.
MAX_LOT_PAGES = 120


class Phase1Scraper:
    def __init__(self, config_path: str = "config.json"):
        self.config = self._load_config(config_path)

        self.graphql_url = "https://hibid.com/graphql"
        self.timeout = self.config["api"]["timeout_seconds"]
        self.headers = {
            "User-Agent": self.config["api"]["user_agent"],
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Origin": "https://hibid.com",
            "Referer": "https://hibid.com/",
        }

        self.zip_code = self.config["sourcing"]["zip_code"]
        self.radius = self.config["sourcing"]["radius_miles"]
        self.page_size = self.config["sourcing"]["page_size"]

        self.ship_killers = self.config["logistics"]["ship_killers"]
        self.mailbox_winners = self.config["logistics"]["mailbox_winners"]

        ship_cfg = self.config.get("shipping", {})
        self.bundled_ship_cost = ship_cfg.get("bundled_ship_cost", 25.0)
        self.buyer_premium_pct = ship_cfg.get("buyer_premium_pct", 15.0)

        self.include_nationwide = False
        self.closing_within_days = 7  # Only include auctions closing within this many days
        self.category_filter: List[str] = []  # Optional list of category substrings (case-insensitive)

    def _load_config(self, filepath: str) -> dict:
        from .config_loader import load_config
        return load_config(filepath)

    # Patterns that indicate an item is pickup-only (checked against description too).
    # Expanded to cover common HiBid phrasings — easier to over-match here
    # (HARD just skips AI + comps, which is recoverable) than to miss real
    # pickup-only items.
    _PICKUP_ONLY_RE = re.compile(
        # explicit pickup-only phrasings
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
        # explicit no-ship phrasings
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
        # local-only / regional phrasings
        r'|local\s+(delivery|sale|buyers?)\s+only'
        r'|ships?\s+locally\s+only'
        r'|ships?\s+only\s+(locally|to\s+local)'
        # specific HiBid boilerplate
        r'|this\s+lot\s+(is|will\s+be)\s+(a\s+)?pick\s*-?\s*up'
        r'|available\s+for\s+pickup\s+only',
        re.IGNORECASE,
    )

    def classify_logistics(self, title: str, category: str, description: str = "") -> str:
        text = f"{title} {category}".lower()

        # 1. Description "pickup only" language trumps everything — it's an
        #    explicit seller statement, not a title heuristic. A jewelry lot
        #    that says "must pick up in person" really is pickup-only.
        if description and self._PICKUP_ONLY_RE.search(description):
            return "HARD"

        # 2. Mailbox-winner keywords (coin, jewelry, watch, card, etc.) are
        #    very specific small-item signals — they beat generic size
        #    words in ship_killers. Avoids false HARD on titles like
        #    "Large Morgan Silver Dollar Collection" (gold/silver/coin
        #    matches mailbox_winners; "large" matches ship_killers).
        if re.search(self.mailbox_winners, text):
            return "EASY"

        # 3. Otherwise fall back to the ship_killers heuristic.
        if re.search(self.ship_killers, text):
            return "HARD"

        return "NEUTRAL"

    def estimate_total_cost(self, bid: float, premium_mult: float = None) -> float:
        """Estimate acquisition cost per item: bid + buyer premium.

        ``premium_mult`` is the per-auction multiplier parsed from
        HiBid's buyerPremium text (1.10 for a 10% auction). When None
        (auction meta unavailable / unparseable), falls back to the
        config-wide default percentage.
        """
        if premium_mult is not None and premium_mult > 0:
            return bid * premium_mult
        premium = bid * (self.buyer_premium_pct / 100.0)
        return bid + premium

    async def fetch_auction_meta(self, client: httpx.AsyncClient,
                                 auction_ids) -> Dict[int, Dict]:
        """Fetch per-auction commercial terms (premium, shipping blurb).

        Returns ``{auction_id: {premium_mult, cond_ship, ship_hint}}``.
        Chunked at 50 ids per GraphQL call. Failures degrade to an
        empty dict for the affected chunk — callers treat missing
        auctions as "use config defaults", never as an error.
        """
        meta: Dict[int, Dict] = {}
        ids = [int(a) for a in dict.fromkeys(auction_ids) if a]
        for i in range(0, len(ids), 50):
            chunk = ids[i:i + 50]
            try:
                data = await self._graphql(
                    client, "AuctionMeta", AUCTION_META_QUERY,
                    {"eventIds": chunk},
                )
            except Exception:
                continue
            markers = (data.get("auctionMap") or {}).get("mapMarkers") or []
            for mk in markers:
                a = mk.get("auction") or {}
                aid = a.get("id")
                if aid is None:
                    continue
                cond, hint = parse_shipping_info(
                    a.get("shippingAndPickupInfo") or ""
                )
                premium_mult = parse_buyer_premium_pct(
                    a.get("buyerPremium"), a.get("buyerPremiumRate")
                )
                if premium_mult is None:
                    # buyerPremium had no number ("BUYERS PREMIUM
                    # APPLIES TO ALL PURCHASES") — the actual % is
                    # often buried in the terms text instead.
                    premium_mult = parse_premium_from_terms(
                        a.get("termsAndConditions")
                    )
                meta[int(aid)] = {
                    "premium_mult": premium_mult,
                    "cond_ship": cond,
                    "ship_hint": hint,
                }
        return meta

    @staticmethod
    def _parse_estimate(raw: str):
        """Parse HiBid's `estimate` string into (low, high) floats.

        Examples we handle:
          - "10.00 - 50.00 USD"  → (10.0, 50.0)
          - "$75"                → (75.0, 75.0)
          - "100 to 200"         → (100.0, 200.0)
          - "" / None            → (None, None)
        """
        if not raw:
            return None, None
        nums = re.findall(r'\d+(?:\.\d+)?', str(raw).replace(',', ''))
        if not nums:
            return None, None
        if len(nums) >= 2:
            return float(nums[0]), float(nums[1])
        v = float(nums[0])
        return v, v

    async def _graphql(self, client: httpx.AsyncClient, operation: str, query: str, variables: dict) -> dict:
        payload = {
            "operationName": operation,
            "query": query,
            "variables": variables,
        }
        # Retry ladder with escalating timeouts. A single 100-lot page
        # with heavy descriptions can exceed the base timeout on a slow
        # HiBid evening, and with no retry ONE timeout used to kill an
        # entire 800-lot auction fetch (user hit exactly this 7/12).
        # Three attempts at 1x / 2x / 3x the configured timeout with a
        # short backoff; transient transport hiccups (connection reset,
        # DNS blip) retry the same way. Non-timeout HTTP errors still
        # raise immediately below.
        last_exc: Exception = None
        for _attempt in range(3):
            try:
                response = await client.post(
                    self.graphql_url,
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout * (_attempt + 1),
                )
                break
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last_exc = e
                if _attempt < 2:
                    await asyncio.sleep(1.5 * (_attempt + 1))
        else:
            raise RuntimeError(
                f"HiBid GraphQL {operation} timed out after 3 attempts "
                f"(final timeout {self.timeout * 3:.0f}s): {last_exc}"
            )
        # Capture the body on error — a bare raise_for_status() swallows the
        # response text, and HiBid's 400s explain exactly what they dislike
        # (e.g. "pageSize exceeds maximum") in the body.
        if response.status_code >= 400:
            body_snippet = ""
            try:
                body_snippet = response.text[:500]
            except Exception:
                pass
            raise RuntimeError(
                f"HiBid GraphQL {operation} returned HTTP {response.status_code}: "
                f"{body_snippet or '(empty body)'} | variables={variables}"
            )
        data = response.json()
        if "errors" in data:
            raise RuntimeError(f"GraphQL errors ({operation}): {data['errors']}")
        return data["data"]

    async def fetch_auctions(self, client: httpx.AsyncClient, zip_code: str, radius: int,
                             search_text: str = "") -> List[Dict]:
        """Fetch auctions from HiBid. Use empty zip for nationwide.

        ``search_text`` searches HiBid's auction catalog server-side
        (auction names + content) — powers the "find an auction by
        name before running Discover" flow. Empty = full listing
        (the discovery default).
        """
        variables = {
            "zip": zip_code,
            "miles": radius,
            "searchText": search_text or "",
            "categoryId": -1,
            "filter": "ALL",
            "status": "OPEN",
            "eventIds": None,
        }

        try:
            data = await self._graphql(client, "AuctionMap", AUCTION_MAP_QUERY, variables)
            markers = data.get("auctionMap", {}).get("mapMarkers", [])

            auctions = [
                {
                    "auction_id": m["auction"]["id"],
                    "name": m["auction"]["eventName"],
                    "city": m["auction"].get("eventCity", ""),
                    "state": m["auction"].get("eventState", ""),
                    "lot_count": m["auction"].get("lotCount", 0),
                    "auctioneer": m["auction"].get("auctioneer", {}).get("name", ""),
                    "date_begin": m["auction"].get("eventDateBegin", ""),
                    "date_end": m["auction"].get("eventDateEnd", ""),
                    "date_info": m["auction"].get("eventDateInfo", ""),
                }
                for m in markers
                if m.get("auction")
            ]
            return auctions
        except Exception as e:
            raise RuntimeError(f"Auction fetch failed: {e}")

    def _filter_by_closing_date(self, auctions: List[Dict]) -> List[Dict]:
        """Filter auctions to only those closing within self.closing_within_days."""
        if not self.closing_within_days:
            return auctions

        now = datetime.now()
        cutoff = now + timedelta(days=self.closing_within_days)
        filtered = []
        for a in auctions:
            end_dt = self._resolve_auction_end(a, now=now)
            if end_dt is None:
                # Couldn't resolve a closing time from any field —
                # keep the auction (err on inclusion). The downstream
                # lot-fetch step recomputes closing time from the
                # first lot's `timeLeft` and writes it back.
                filtered.append(a)
                continue
            # Lower bound uses DATE comparison rather than timestamp:
            # an auction whose end_dt is today at 3pm should still
                # appear in the morning list AND in the late-afternoon
            # list, even if we couldn't recover an exact closing time
            # and assumed midnight. This protects against HiBid
            # date-only payloads slipping past _resolve_auction_end's
            # enrichment path.
            if end_dt.date() >= now.date() and end_dt <= cutoff:
                filtered.append(a)
        return filtered

    @staticmethod
    def _resolve_auction_end(a: Dict, now: Optional[datetime] = None) -> Optional[datetime]:
        """Resolve an auction-level closing datetime from available fields.

        Tries in order:
          1. `date_end` parsed as ISO (HiBid's `eventDateEnd`)
          2. `date_info` free-text scanned for date-like patterns
             ("Closes Wed May 6 7:00 PM", "Ends 5/6/2026", etc.)

        Lot-level `timeLeft` fallback can't fire here because the
        candidate-discovery step doesn't fetch lots. That fallback
        runs in `_fetch_lots_for_auction` after lots arrive.
        """
        date_end = a.get('date_end') or ''
        if date_end:
            try:
                parsed_end = datetime.fromisoformat(date_end.replace('Z', ''))
                # HiBid sometimes returns eventDateEnd as a date-only
                # string (no time component), which fromisoformat reads
                # as midnight. That breaks "closes today" filtering —
                # by the time the user checks the dashboard at 2pm,
                # 00:00 is already in the past and the auction gets
                # filtered out even though it closes at 7pm tonight.
                # When time is missing, enrich from date_info if it
                # carries an "h:mm AM/PM" token; otherwise default to
                # 23:59 so the auction stays in view for the full day.
                if parsed_end.hour == 0 and parsed_end.minute == 0:
                    date_info_str = (a.get('date_info') or '').strip()
                    time_match = re.findall(
                        r'(\d{1,2})(?::(\d{2}))?\s*([ap])\.?m\.?',
                        date_info_str, flags=re.IGNORECASE,
                    )
                    if time_match:
                        h, m, mer = time_match[-1]
                        hour24 = int(h) % 12 + (12 if mer.lower() == 'p' else 0)
                        minute = int(m) if m else 0
                        parsed_end = parsed_end.replace(hour=hour24, minute=minute)
                    else:
                        parsed_end = parsed_end.replace(hour=23, minute=59)
                return parsed_end
            except (ValueError, TypeError):
                pass

        date_info = (a.get('date_info') or '').strip()
        if not date_info:
            return None

        # Strip leading prepositions/verbs to leave a date-shaped tail.
        # HiBid date_info commonly looks like:
        #   "Closes Wed May 6, 2026 7:00 PM CDT"
        #   "Bidding ends May 6 at 7:00 PM"
        #   "Begins May 1 — Ends May 6"
        # We pluck out the part after "ends"/"closes" if present.
        s = date_info
        m = re.search(r'(?:ends|closes)[:\s]+(.+)$', s, re.IGNORECASE)
        if m:
            s = m.group(1).strip()

        # Try a handful of common formats. Year defaults to current
        # year if missing — this is risky around year-end but matches
        # how a human reads "Closes Jan 4" in late December.
        ref_year = (now or datetime.now()).year
        formats = [
            "%a %b %d, %Y %I:%M %p %Z",  # "Wed May 6, 2026 7:00 PM CDT"
            "%a %b %d, %Y %I:%M %p",     # "Wed May 6, 2026 7:00 PM"
            "%b %d, %Y %I:%M %p",        # "May 6, 2026 7:00 PM"
            "%b %d, %Y",                 # "May 6, 2026"
            "%b %d %I:%M %p",            # "May 6 7:00 PM"  (no year)
            "%b %d",                     # "May 6"          (no year)
            "%m/%d/%Y %I:%M %p",         # "5/6/2026 7:00 PM"
            "%m/%d/%Y",                  # "5/6/2026"
            "%m/%d %I:%M %p",            # "5/6 7:00 PM"    (no year)
            "%m/%d",                     # "5/6"            (no year)
        ]
        # Strip filler words ("at"), then strip trailing timezone
        # abbreviations the strptime parser can't handle (CDT, EST,
        # PST, etc.). EXCLUDE AM/PM from the strip — those are part
        # of the time format, not a timezone, and an earlier version
        # of this code ate them and broke 7:00 PM parsing.
        cleaned = re.sub(r'\s+at\s+', ' ', s, flags=re.IGNORECASE)
        cleaned = re.sub(
            r'\s+(?!(?:AM|PM)\b)(?:[A-Z]{2,4})\s*$',
            '',
            cleaned,
        ).strip()
        for fmt in formats:
            try:
                parsed = datetime.strptime(cleaned, fmt)
                # Backfill year if format omitted it.
                if '%Y' not in fmt:
                    parsed = parsed.replace(year=ref_year)
                return parsed
            except (ValueError, TypeError):
                continue
        return None

    async def _fetch_all_lot_pages(
        self,
        client: httpx.AsyncClient,
        auction_id: int,
        search_text: str = "",
    ) -> List[Dict]:
        """Fetch every lot in an auction.

        HiBid's current schema pages at a fixed size of 100 lots with
        `pageNumber` (1-based) as a sibling arg to `input`. We call page 1
        to get totalCount, then paginate through the remaining pages.

        When ``search_text`` is non-empty, HiBid does the filtering
        server-side — instead of downloading 10k lots per auction and
        filtering locally, we get back only the matching lots (often
        zero). This is what powers the "keyword across auctions" scan.
        HiBid's searchText is case-insensitive and matches against
        both title (lead) and description, so callers should still
        apply a refinement filter if they want title-only matching.
        """
        lots: List[Dict] = []
        total_count = 0

        # First request: page 1 gives us both results and totalCount
        variables = {
            "auctionId": auction_id,
            "pageNumber": 1,
            "searchText": search_text or "",
        }
        data = await self._graphql(client, "LotSearch", LOT_SEARCH_QUERY, variables)
        paged = (data.get("lotSearch") or {}).get("pagedResults") or {}
        batch = paged.get("results") or []
        total_count = paged.get("totalCount") or 0
        lots.extend(batch)

        if total_count == 0 or len(lots) >= total_count:
            return lots

        # Remaining pages. ceil(total/100) = total pages; we've fetched page 1.
        import math
        last_page = min(math.ceil(total_count / LOT_PAGE_SIZE), MAX_LOT_PAGES)
        for page_number in range(2, last_page + 1):
            variables = {
                "auctionId": auction_id,
                "pageNumber": page_number,
                "searchText": search_text or "",
            }
            data = await self._graphql(client, "LotSearch", LOT_SEARCH_QUERY, variables)
            paged = (data.get("lotSearch") or {}).get("pagedResults") or {}
            batch = paged.get("results") or []
            if not batch:
                break  # empty page = we're past the end
            lots.extend(batch)
            if len(lots) >= total_count:
                break

        return lots

    @staticmethod
    def _extract_category_name(raw) -> str:
        """HiBid returns `category` sometimes as a list of {categoryName}
        dicts, sometimes as a single dict, sometimes as None. Handle all."""
        if not raw:
            return ''
        if isinstance(raw, list):
            if not raw:
                return ''
            first = raw[0]
            return first.get('categoryName', '') if isinstance(first, dict) else ''
        if isinstance(raw, dict):
            return raw.get('categoryName', '')
        return ''

    async def fetch_lots_for_auction(self, client: httpx.AsyncClient, auction_id: int,
                                     auction_name: str = "", date_end: str = "",
                                     source: str = "Local Pickup",
                                     search_text: str = "",
                                     auction_meta: Dict = None):
        """Fetch + process one auction's lots.

        When ``search_text`` is non-empty, HiBid filters lots
        server-side — most auctions return zero matching lots, so the
        keyword-scan path completes in seconds instead of minutes.

        Returns a dict: {
            "lots": [processed_lot, ...],
            "raw_count": int,                # lots returned by HiBid (pre-filter)
            "filtered_by_category": int,     # lots dropped by sidebar category filter
            "error": Optional[str],          # exception message if something blew up
            "auction_id": int,
            "auction_name": str,
        }
        """
        result = {
            "lots": [],
            "raw_count": 0,
            "filtered_by_category": 0,
            "error": None,
            "auction_id": auction_id,
            "auction_name": auction_name,
        }
        try:
            lots = await self._fetch_all_lot_pages(
                client, auction_id, search_text=search_text,
            )
            result["raw_count"] = len(lots)

            # Compute closing_fmt for the auction. Primary source: the
            # `date_end` we already had from auctionMap discovery. When
            # that's missing (some HiBid responses omit eventDateEnd
            # for in-progress timed auctions), fall back to deriving it
            # from the first lot's `lotState.timeLeft` — every lot
            # carries a "Nd Nh Nm" countdown that resolves to a real
            # closing datetime relative to now. Lots within an auction
            # close sequentially in lot-number order, so the first
            # lot's countdown is the earliest auction-level closing.
            closing_fmt = ""
            if date_end:
                try:
                    closing_fmt = datetime.fromisoformat(
                        date_end.replace('Z', '')
                    ).strftime("%b %d")
                except (ValueError, TypeError):
                    closing_fmt = date_end
            else:
                # Fallback: derive from first lot's timeLeft. Walks
                # the lot list looking for the first parseable
                # countdown. Returns None if every lot is closed/
                # missing — leaves closing_fmt blank in that case.
                derived = _derive_auction_closing_from_lots(lots)
                if derived is not None:
                    closing_fmt = derived.strftime("%b %d")

            cat_keywords = [c.strip().lower() for c in (self.category_filter or []) if c and c.strip()]

            processed_lots = []
            for lot in lots:
                title = lot.get('lead', '') or ''
                category = self._extract_category_name(lot.get('category'))
                description = lot.get('description', '') or ''

                if cat_keywords:
                    haystack = f"{category} {title}".lower()
                    if not any(kw in haystack for kw in cat_keywords):
                        result["filtered_by_category"] += 1
                        continue

                state = lot.get('lotState') or {}
                logistics = self.classify_logistics(title, category, description)
                current_bid = state.get('highBid', 0.0) or 0.0
                # `minBid` is the minimum acceptable next bid. When the lot
                # has zero bids, current_bid==0 but minBid is still the
                # auctioneer's starting bid — that's the real acquisition
                # cost basis, not $0. Pick whichever is higher to handle
                # both no-bids and active-bidding states.
                next_bid = state.get('minBid', 0.0) or 0.0
                effective_bid = max(current_bid, next_bid)
                # Per-auction premium (parsed from HiBid buyerPremium
                # text) beats the config-wide default. Real premiums
                # observed: 0% (US Marshals) through 22% — the flat
                # 15% default understates cost on high-premium
                # auctions, the dangerous direction.
                _premium_mult = (auction_meta or {}).get('premium_mult')
                total_cost = self.estimate_total_cost(
                    effective_bid, premium_mult=_premium_mult,
                )

                # Auctioneer's value range, e.g. "10.00 - 50.00 USD".
                # Used as a sanity check against eBay/Mercari comp medians.
                est_low, est_high = self._parse_estimate(lot.get('estimate'))

                lot_id = lot.get('id')

                # Pull the first thumbnail URL — that's what we feed to eBay
                # image_search during vision enrichment. HiBid's CDN requires
                # a Referer header; we only store the URL here.
                pictures = lot.get('pictures') or []
                thumbnail_url = ''
                hd_thumbnail_url = ''
                fullsize_url = ''
                if pictures:
                    first = pictures[0] or {}
                    thumbnail_url = first.get('thumbnailLocation') or ''
                    hd_thumbnail_url = first.get('hdThumbnailLocation') or ''
                    fullsize_url = first.get('fullSizeLocation') or ''

                # Per-lot shipping flag from HiBid's GraphQL. Many
                # auctions (especially estate-liquidation and Hill-Country
                # style) mix shippable small items with pickup-only large
                # items in the same catalog. Trust the per-lot flag when
                # present; fall back to the auction-level `source` arg
                # only when the field is unexpectedly null.
                lot_ship = lot.get('shippingOffered')
                if lot_ship is True:
                    effective_source = 'Ship'
                elif lot_ship is False:
                    effective_source = 'Local Pickup'
                elif 'local pickup only' in (description or '').lower():
                    # Belt-and-suspenders: some auctioneers skip the
                    # structured flag and write "Local Pickup Only" in
                    # the lot description instead. Text is authoritative.
                    effective_source = 'Local Pickup'
                else:
                    effective_source = source

                # UNREACHABLE pickup-only detection. `source` (the arg)
                # is the AUCTION-level classification: 'Ship' means the
                # auction was found via the nationwide query — it is NOT
                # within the user's pickup radius. A pickup-only lot in
                # such an auction can never be acquired: the auctioneer
                # won't ship it and the user won't drive 1,200 miles.
                # 7/6 Hayworth (Pompeii, MI — user in Houston): 6-foot
                # metal shelves and post-hole diggers were A-graded at
                # $0 ship because their pickup-only flag made them look
                # like local wins. Consumed by the buy-score (hard F),
                # the comps filter (skip — no credits), and the header
                # badge.
                unreachable_pickup = (
                    source == 'Ship' and effective_source == 'Local Pickup'
                )

                processed_lots.append({
                    "unreachable_pickup": unreachable_pickup,
                    "lot_id": lot_id,
                    "auction": auction_name,
                    "auction_link": f"https://hibid.com/auction/{auction_id}",
                    "closing_date": closing_fmt,
                    "source": effective_source,
                    "title": title,
                    "lot_link": f"https://hibid.com/lot/{lot_id}",
                    "category": category,
                    "current_bid": current_bid,
                    "next_bid": round(next_bid, 2),
                    "bid_count": state.get('bidCount', 0) or 0,
                    "est_cost": round(total_cost, 2),
                    "auctioneer_est_low": est_low,
                    "auctioneer_est_high": est_high,
                    "status": state.get('status', '') or '',
                    "time_left": state.get('timeLeft', '') or '',
                    "description": description,
                    "logistics_ease": logistics,
                    "thumbnail_url": thumbnail_url,
                    "hd_thumbnail_url": hd_thumbnail_url,
                    "fullsize_url": fullsize_url,
                    "image_count": len(pictures),
                    # Per-auction commercial terms (same value on every
                    # lot of an auction). Consumed by _compute_max_bid /
                    # _compute_buy_score (premium + ship hint) and the
                    # analysis-view header (conditional-shipping badge).
                    "auction_buyer_premium_pct": _premium_mult,
                    "auction_cond_ship": (auction_meta or {}).get('cond_ship', False),
                    "auction_ship_hint": (auction_meta or {}).get('ship_hint'),
                })
            result["lots"] = processed_lots
        except Exception as e:
            result["error"] = f"{type(e).__name__}: {e}"
        return result

    async def _fetch_lots_batch(self, client: httpx.AsyncClient, auctions: List[Dict],
                                source: str, batch_size: int = 20,
                                progress_callback=None, progress_offset: int = 0,
                                grand_total: int = None, phase_label: str = "",
                                search_text: str = "",
                                auction_meta_map: Dict[int, Dict] = None):
        """Fetch lots for a list of auctions in concurrent batches.

        Returns a dict: {
            "lots": [...],
            "raw_count": int,
            "filtered_by_category": int,
            "errors": [{"auction_id", "auction_name", "error"}, ...],
            "per_auction": [{"auction_id", "auction_name", "raw_count", "kept": int}, ...]
        }

        progress_callback signature:
          (current, total, label, extras) -> None
        where `extras` is a dict carrying running lot counts + names of
        the auctions just completed in this batch, so the UI can show
        "Fetched 12,450 lots so far · just finished: Estate Sale 4/29".
        Older single-line callbacks are still supported via 3-arg call
        when the callback only accepts 3 positional args.
        """
        agg = {
            "lots": [],
            "raw_count": 0,
            "filtered_by_category": 0,
            "errors": [],
            "per_auction": [],
        }
        total = len(auctions)
        effective_total = grand_total if grand_total is not None else total
        running_lots = 0

        for i in range(0, total, batch_size):
            batch = auctions[i:i + batch_size]
            tasks = [
                self.fetch_lots_for_auction(
                    client, a['auction_id'], a['name'], a.get('date_end', ''), source,
                    search_text=search_text,
                    auction_meta=(auction_meta_map or {}).get(
                        int(a['auction_id'])
                    ),
                )
                for a in batch
            ]
            results = await asyncio.gather(*tasks)
            batch_lots = 0
            batch_names = []
            for r in results:
                agg["lots"].extend(r["lots"])
                agg["raw_count"] += r["raw_count"]
                agg["filtered_by_category"] += r["filtered_by_category"]
                batch_lots += len(r["lots"])
                batch_names.append(r["auction_name"])
                if r["error"]:
                    agg["errors"].append({
                        "auction_id": r["auction_id"],
                        "auction_name": r["auction_name"],
                        "error": r["error"],
                    })
                agg["per_auction"].append({
                    "auction_id": r["auction_id"],
                    "auction_name": r["auction_name"],
                    "raw_count": r["raw_count"],
                    "kept": len(r["lots"]),
                })
            running_lots += batch_lots

            if progress_callback:
                current = progress_offset + min(i + batch_size, total)
                extras = {
                    "running_lots": agg["raw_count"] + progress_offset * 0,
                    "running_kept": len(agg["lots"]),
                    "batch_names": batch_names,
                    "batch_lots": batch_lots,
                    "errors_so_far": len(agg["errors"]),
                }
                # Try the new 4-arg signature first; fall back to 3-arg
                # for older callers that don't take extras.
                try:
                    progress_callback(current, effective_total, phase_label, extras)
                except TypeError:
                    progress_callback(current, effective_total, phase_label)

        return agg

    async def fetch_auction_candidates(self, progress_callback=None) -> List[Dict]:
        """Return the combined local + nationwide auction list WITHOUT fetching lots.

        This is the cheap first step of the two-step discovery flow: the user
        picks which auctions are worth a deep scan, then we only pay the
        per-lot cost on their selection.

        Each returned dict carries `auction_id`, `name`, `city`, `state`,
        `lot_count`, `date_begin`, `date_end`, `date_info`, `auctioneer`,
        plus a `source` field ('Local Pickup' or 'Ship') so the caller can
        preserve that semantic when lots are later fetched.
        """
        def _report(current, total, label):
            if progress_callback:
                progress_callback(current, total, label)

        async with httpx.AsyncClient(verify=_HTTPX_VERIFY) as client:
            # Parallelize the local + nationwide GraphQL calls. They were
            # sequential before — total Phase 1a was ~10s for two ~5s
            # calls in series. Running them concurrently cuts that to
            # ~5s wall-clock. asyncio.gather waits for both; if either
            # raises, propagate (we want both to succeed for a correct
            # candidate list).
            _report(
                0, 2,
                f"Querying HiBid (local within {self.radius} mi of "
                f"{self.zip_code or '(any)'}, nationwide in parallel)…"
            )
            # Hard outer timeout so a hung GraphQL request can't wedge
            # the whole discovery indefinitely. httpx has its own per-
            # request timeout (15s from config), but rare proxy /
            # connection-pool edge cases can let a request sit
            # waiting. The asyncio.wait_for here is a watchdog: 45s
            # ceiling for the parallel pair, well above the expected
            # ~5s.
            try:
                if self.include_nationwide:
                    local_raw, nationwide_raw = await asyncio.wait_for(
                        asyncio.gather(
                            self.fetch_auctions(
                                client, self.zip_code, self.radius
                            ),
                            self.fetch_auctions(client, "", 0),
                        ),
                        timeout=45.0,
                    )
                else:
                    local_raw = await asyncio.wait_for(
                        self.fetch_auctions(
                            client, self.zip_code, self.radius
                        ),
                        timeout=45.0,
                    )
                    nationwide_raw = []
            except asyncio.TimeoutError as e:
                raise RuntimeError(
                    "HiBid discovery timed out after 45s. The GraphQL "
                    "endpoint is hung or the network connection is "
                    "blocked. Try again in a moment."
                ) from e

            # Local processing (filter by closing date, mark source).
            local_raw_count = len(local_raw)
            local_auctions = self._filter_by_closing_date(local_raw)
            local_ids = {a['auction_id'] for a in local_auctions}
            for a in local_auctions:
                a['source'] = 'Local Pickup'
            _report(
                1, 2,
                f"Local: {len(local_auctions)} kept "
                f"(of {local_raw_count} returned, filtered by closing date)",
            )

            # Nationwide processing.
            remote_auctions: List[Dict] = []
            if self.include_nationwide:
                remote_auctions = [
                    a for a in nationwide_raw
                    if a['auction_id'] not in local_ids
                ]
                pre_close_count = len(remote_auctions)
                remote_auctions = self._filter_by_closing_date(remote_auctions)
                remote_auctions = sorted(
                    # `or ''` (not a dict default): date_end is always
                    # PRESENT but can be None when HiBid returns
                    # eventDateEnd:null — sorting None against ISO
                    # strings raises TypeError and killed nationwide
                    # discovery (confirmed by the 7/11 NaN sweep).
                    remote_auctions, key=lambda a: a.get('date_end') or '',
                )
                for a in remote_auctions:
                    a['source'] = 'Ship'
                _report(
                    2, 2,
                    f"Nationwide: {len(remote_auctions)} kept "
                    f"(of {len(nationwide_raw)} returned, "
                    f"{pre_close_count - len(remote_auctions)} dropped "
                    f"by closing date)",
                )

            all_auctions = local_auctions + remote_auctions
            _report(len(all_auctions), max(len(all_auctions), 1),
                    f"Found {len(all_auctions)} auctions")
            return all_auctions

    async def sample_lot_categories(
        self, client: httpx.AsyncClient, auction_id: int, sample_size: int = 20
    ) -> Dict[str, list]:
        """Fetch a small lot sample and return a preview payload.

        Used by the two-step picker so the user can see what KINDS of stuff
        are in an auction without paying to fetch every lot. Cheap — one
        GraphQL call per auction (the server returns a 100-lot page; we
        slice `sample_size` off the front).

        Returns a dict:
            {
                "categories": [category_name, ...] — unique, sorted
                "cat_counts": {category_name: int} — counts within sample
                "titles":     [lot lead, ...] up to sample_size
            }

        Kept backward-compat: callers that previously received a plain
        list of categories can treat the dict as an iterable of "categories".
        """
        # Note: the GraphQL LotSearch query takes `pageNumber` as a sibling
        # of `input` (per the schema note at the top of this file). The old
        # `pageIndex`/`pageSize` form was removed by HiBid; using it here
        # silently failed and made the preview column blank.
        variables = {"auctionId": auction_id, "pageNumber": 1}
        empty = {"categories": [], "cat_counts": {}, "titles": []}
        try:
            data = await self._graphql(client, "LotSearch", LOT_SEARCH_QUERY, variables)
        except Exception:
            return empty
        paged = data.get("lotSearch", {}).get("pagedResults", {}) or {}
        lots = paged.get("results", []) or []
        # Only inspect the first sample_size lots to keep the cost bounded.
        lots = lots[:sample_size]

        cat_counts: Dict[str, int] = {}
        titles: List[str] = []
        for lot in lots:
            name = self._extract_category_name(lot.get('category'))
            if name:
                cat_counts[name] = cat_counts.get(name, 0) + 1
            lead = (lot.get('lead') or '').strip()
            if lead:
                titles.append(lead)

        return {
            "categories": sorted(cat_counts.keys()),
            "cat_counts": cat_counts,
            "titles": titles,
        }

    async def sample_categories_batch(
        self, auctions: List[Dict], sample_size: int = 8,
        batch_size: int = 30, progress_callback=None,
    ) -> Dict[int, Dict[str, list]]:
        """Sample categories + titles for a batch of auctions concurrently.

        Returns {auction_id: {"categories": [...], "cat_counts": {...},
        "titles": [...]}}. Useful for the picker UI to show "what's in
        this auction" without fetching every lot.
        """
        out: Dict[int, Dict[str, list]] = {}
        total = len(auctions)
        async with httpx.AsyncClient(verify=_HTTPX_VERIFY) as client:
            for i in range(0, total, batch_size):
                chunk = auctions[i:i + batch_size]
                tasks = [
                    self.sample_lot_categories(client, a['auction_id'], sample_size)
                    for a in chunk
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for a, r in zip(chunk, results):
                    if isinstance(r, dict):
                        out[a['auction_id']] = r
                    elif isinstance(r, list):
                        # Backward compat if an override returns the old shape
                        out[a['auction_id']] = {
                            "categories": r, "cat_counts": {}, "titles": [],
                        }
                    else:
                        out[a['auction_id']] = {
                            "categories": [], "cat_counts": {}, "titles": [],
                        }
                if progress_callback:
                    progress_callback(
                        min(i + batch_size, total), total,
                        f"Sampling categories ({min(i + batch_size, total)}/{total})",
                    )
        return out

    @staticmethod
    def generate_auction_summary(
        auction: Dict, sample_payload: Dict = None
    ) -> str:
        """Build a short "what's in this auction" blurb from the sampled
        category distribution.

        Example output:
            "Mostly Furniture (40%), Tools (25%), Kitchen (15%)"
            "Mix of Jewelry (35%), Watches (20%), Coins (15%)"

        The lead word switches to "Mostly" when the top category alone
        accounts for >=50% of the sample, otherwise "Mix of".
        """
        sample_payload = sample_payload or {}
        cat_counts: Dict[str, int] = sample_payload.get("cat_counts") or {}
        total_sample = sum(cat_counts.values())

        if cat_counts and total_sample:
            top = sorted(cat_counts.items(), key=lambda kv: -kv[1])[:3]
            pieces = [
                f"{name} ({int(round(100 * n / total_sample))}%)"
                for name, n in top
            ]
            lead_word = "Mostly" if top[0][1] / total_sample >= 0.5 else "Mix of"
            return f"{lead_word} {', '.join(pieces)}"
        if sample_payload.get("categories"):
            return "Categories: " + ", ".join(sample_payload["categories"][:5])
        return ""

    async def fetch_lots_for_selected(
        self, selected_auctions: List[Dict], progress_callback=None,
        search_text: str = "",
    ) -> pd.DataFrame:
        """Fetch full lot detail for a caller-supplied list of auctions.

        When ``search_text`` is non-empty, HiBid filters lots server-side
        across every auction in ``selected_auctions``. Auctions with zero
        matching lots return an empty payload (single GraphQL call, no
        pagination needed) — turning a multi-minute full fetch into a
        seconds-long targeted scan. Used by the "keyword across auctions"
        feature in the dashboard.

        Returns a DataFrame. Diagnostic counts (raw_count, filtered_by_*,
        per_auction breakdown, errors) are attached to `df.attrs` so the UI
        can show the user exactly where their items went.
        """
        empty_attrs = {
            "raw_count": 0,
            "filtered_by_category": 0,
            "filtered_by_status": 0,
            "per_auction": [],
            "errors": [],
            "status_values_seen": {},
        }
        if not selected_auctions:
            df = pd.DataFrame()
            df.attrs.update(empty_attrs)
            return df

        local_auctions = [a for a in selected_auctions if a.get('source') != 'Ship']
        remote_auctions = [a for a in selected_auctions if a.get('source') == 'Ship']
        grand_total = len(local_auctions) + len(remote_auctions)

        agg_lots: List[Dict] = []
        raw_count = 0
        filtered_by_category = 0
        per_auction: List[Dict] = []
        errors: List[Dict] = []

        async with httpx.AsyncClient(verify=_HTTPX_VERIFY) as client:
            # Per-auction commercial terms (buyer premium, shipping
            # blurb) — one cheap call per 50 auctions, fetched before
            # the lot pages so every processed lot can carry its
            # auction's premium multiplier + ship hint.
            try:
                _meta_map = await self.fetch_auction_meta(
                    client,
                    [a['auction_id'] for a in selected_auctions],
                )
            except Exception:
                _meta_map = {}

            if local_auctions:
                # Phase label gets joined with "{label} — {current}/{total}
                # auctions fetched" downstream, so keep the label compact
                # and human-readable. e.g. "Local-pickup phase (9 auctions)".
                local_label = (
                    f"Local-pickup phase ({len(local_auctions)} auctions)"
                    if not remote_auctions
                    else f"Local-pickup phase "
                         f"({len(local_auctions)} of {grand_total} are pickup-only)"
                )
                r = await self._fetch_lots_batch(
                    client, local_auctions, "Local Pickup",
                    progress_callback=progress_callback, progress_offset=0,
                    grand_total=grand_total, phase_label=local_label,
                    search_text=search_text,
                    auction_meta_map=_meta_map,
                )
                agg_lots.extend(r["lots"])
                raw_count += r["raw_count"]
                filtered_by_category += r["filtered_by_category"]
                per_auction.extend(r["per_auction"])
                errors.extend(r["errors"])

            if remote_auctions:
                # e.g. "Nationwide-shipping phase (993 of 1002 are shippable)"
                nationwide_label = (
                    f"Nationwide-shipping phase "
                    f"({len(remote_auctions)} of {grand_total} are shippable)"
                )
                r = await self._fetch_lots_batch(
                    client, remote_auctions, "Ship",
                    progress_callback=progress_callback,
                    progress_offset=len(local_auctions),
                    grand_total=grand_total, phase_label=nationwide_label,
                    search_text=search_text,
                    auction_meta_map=_meta_map,
                )
                agg_lots.extend(r["lots"])
                raw_count += r["raw_count"]
                filtered_by_category += r["filtered_by_category"]
                per_auction.extend(r["per_auction"])
                errors.extend(r["errors"])

        df = pd.DataFrame(agg_lots)

        # Track raw distribution of status values for diagnostics (so we can
        # SEE whether everything's actually 'CLOSED' or something weird)
        status_values_seen = {}
        if not df.empty and 'status' in df.columns:
            status_values_seen = df['status'].fillna('').astype(str).value_counts().to_dict()

        filtered_by_status = 0
        if not df.empty:
            pre_count = len(df)
            df = df[df['status'] != "CLOSED"]
            df = df[df['time_left'] != "Bidding Closed"]
            filtered_by_status = pre_count - len(df)
            df = df.sort_values('closing_date').reset_index(drop=True)

        df.attrs.update({
            "raw_count": raw_count,
            "filtered_by_category": filtered_by_category,
            "filtered_by_status": filtered_by_status,
            "per_auction": per_auction,
            "errors": errors,
            "status_values_seen": status_values_seen,
        })
        return df

    async def run(self, progress_callback=None) -> pd.DataFrame:
        """Run the full scrape (discover + fetch lots for everything).

        Kept for backward compatibility. The two-step flow used by the UI
        is `fetch_auction_candidates()` + `fetch_lots_for_selected()`.

        progress_callback signature: (current:int, total:int, label:str) -> None
        """
        candidates = await self.fetch_auction_candidates(
            progress_callback=progress_callback
        )
        if not candidates:
            if progress_callback:
                progress_callback(0, 0, "No auctions matched the filters")
            return pd.DataFrame()
        return await self.fetch_lots_for_selected(
            candidates, progress_callback=progress_callback
        )
