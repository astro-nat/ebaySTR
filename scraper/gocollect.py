"""GoCollect API integration — curated CGC/BGS/SGC-graded comic prices.

GoCollect aggregates sold-listing data from eBay, Heritage Auctions,
ComicConnect, MyComicShop and other comic marketplaces, and indexes
prices by *issue + specific grade*. That solves a problem the eBay-sold
scraping path can't: for very rare graded books (Golden Age keys, SOTI
issues, etc.) eBay returns a handful of contaminated comps including
modern reprints, while GoCollect returns the actual graded fair-market
value because the matching is by canonical issue ID, not keyword search.

API spec (per https://gocollect.com/api-docs and developer docs)
----------------------------------------------------------------
- Auth:   `Authorization: Bearer <api_key>` header
- Base:   https://api.gocollect.com/v1
- Search: GET /collectibles?q=<query>&type=comics  → list of items
- Value:  GET /insights/item/<item_id>?grade=<X.X> → fair-market value

Tier limits (as of 2026):
  Free: 50 API calls/day
  Pro:  $9/mo, 100 API calls/day

We aggressively in-process-cache search + value lookups to amortize
the small daily quota across an entire comp run.
"""
import re
import time
import threading
from typing import Optional, Dict, Any, List

import httpx


_BASE_URL = "https://api.gocollect.com/v1"


# Recognize professional-grading-service callouts in lot titles.
# Captures the grader name + numeric grade so we can pass the grade
# straight into GoCollect's insights endpoint.
#   "Amazing Spider-Man 14 CGC 7.0" → ("CGC", "7.0")
#   "Action Comics 1 CBCS 9.6"      → ("CBCS", "9.6")
#   "Detective 27 PSA 8"            → ("PSA", "8")
_GRADE_RE = re.compile(
    r'\b(CGC|BGS|SGC|CBCS|PSA)\s*([0-9]+(?:\.[0-9])?)',
    re.IGNORECASE,
)


def extract_grade(title: str) -> Optional[Dict[str, str]]:
    """Pull a {service, grade} dict out of a lot title, or None.

    Used by both this module's classifier (gate calls to graded books
    only) and the pipeline (decide whether to route here vs. PC/eBay).
    """
    if not title:
        return None
    m = _GRADE_RE.search(title)
    if not m:
        return None
    return {"service": m.group(1).upper(), "grade": m.group(2)}


class GoCollectLookup:
    """Thin client for GoCollect's collectibles + insights API.

    Mirrors PriceChartingLookup's interface so the pricing pipeline
    can route to it the same way: classifier check, single lookup
    call, returns either a price dict or None.

    Without an api_key the instance is a no-op — every call returns
    None and consumes nothing. This lets app code construct the
    lookup unconditionally and skip integration when the user
    hasn't signed up yet.

    GoCollect requires explicit API access approval (a multi-step
    application process). An ``api_key`` in config alone isn't
    enough — the key must also be APPROVED on their end. To prevent
    every comp run from wasting one transport timeout testing an
    unapproved key, we gate on a separate ``approved`` flag in
    config.json:

        "gocollect": {
            "api_key": "Gqc73BZGico...",
            "approved": true   <-- flip this once approval lands
        }

    When approved=False, the instance is a no-op even with a key
    present. Flip approved=True after GoCollect's email confirming
    API access lands, and the next comp run starts using it.
    """

    def __init__(self, api_key: Optional[str], approved: bool = False):
        self.api_key = api_key
        self.approved = bool(approved)
        self._lock = threading.Lock()
        self._cache: Dict[str, Optional[Dict[str, Any]]] = {}
        # Track whether we've hit the daily quota — once a 429 comes
        # back, stop firing more requests for the rest of the run.
        self._quota_exhausted = False

    @property
    def enabled(self) -> bool:
        return (
            bool(self.api_key)
            and self.approved
            and not self._quota_exhausted
        )

    @staticmethod
    def _clean_query(title: str) -> str:
        """Strip price hints, condition noise, and grading callouts.

        We pull the grade out via extract_grade() and pass it
        separately, so it shouldn't poison the search query.
        """
        clean = re.sub(r'\$\d+(?:\.\d{1,2})?\b', '', title)
        # Remove the grading callout itself — already captured separately
        clean = _GRADE_RE.sub('', clean)
        # Remove page-quality codes that are CGC-specific noise
        clean = re.sub(
            r'\b(OW|CRM|WP|OW/W|CRM/OW|White|Cream|OffWhite)\b\s*Pages?\b',
            '', clean, flags=re.IGNORECASE,
        )
        # Strip parentheticals and standard noise punctuation
        clean = re.sub(r'\([^)]{1,25}\)', '', clean)
        clean = re.sub(r'[,;:/\\|\-]+', ' ', clean)
        clean = re.sub(r'\s+', ' ', clean).strip(' .,-')
        return clean

    def _api_get(self, path: str, params: Optional[Dict] = None,
                timeout: float = 10) -> Optional[Dict]:
        """Make an authenticated GET. Returns parsed JSON or None on
        any failure. Detects quota exhaustion (429 / explicit body
        message) and trips the run-level kill switch."""
        if not self.enabled:
            return None
        try:
            resp = httpx.get(
                f"{_BASE_URL}{path}",
                headers={"Authorization": f"Bearer {self.api_key}"},
                params=params or {},
                timeout=timeout,
            )
        except (httpx.ReadTimeout, httpx.ConnectTimeout,
                httpx.RemoteProtocolError, httpx.NetworkError):
            # Total transport failure (no response at all) — same
            # treatment as a 5xx: trip the kill switch so we don't
            # spin through 50 lots × 30s of timeouts.
            self._quota_exhausted = True
            return None
        except Exception:
            return None
        if resp.status_code == 429:
            self._quota_exhausted = True
            return None
        if resp.status_code in (401, 403):
            # Bad key or permission denied — disable for this run so
            # subsequent calls don't repeat the auth failure.
            self._quota_exhausted = True
            return None
        if resp.status_code >= 500:
            # Origin outage (522 from Cloudflare, 502/503/504 etc.).
            # Trip the kill switch for the rest of the run so we don't
            # burn ~30s per lot waiting on a dead service. Each new
            # GoCollectLookup instance resets, so the next comp run
            # will retry naturally.
            self._quota_exhausted = True
            return None
        if resp.status_code != 200:
            return None
        try:
            return resp.json()
        except ValueError:
            return None

    def _search_item_id(self, query: str) -> Optional[str]:
        """Search the collectibles endpoint for the best match.

        Returns the first result's item_id (a string ID) or None.
        Cached per query — repeated lookups for the same comic
        across an auction don't re-hit the search endpoint.
        """
        cache_key = f"search:{query.lower()}"
        with self._lock:
            if cache_key in self._cache:
                cached = self._cache[cache_key]
                return cached.get('item_id') if isinstance(cached, dict) else None
        data = self._api_get("/collectibles", params={
            "q": query,
            "type": "comics",
        })
        if not isinstance(data, dict):
            with self._lock:
                self._cache[cache_key] = None
            return None
        # Response shape (best-effort guess from public docs):
        #   {"data": [{"id": "12345", "title": "...", ...}, ...]}
        # OR:
        #   {"items": [...]}
        # OR top-level list. Try several shapes.
        items: List[Dict] = []
        if isinstance(data.get('data'), list):
            items = data['data']
        elif isinstance(data.get('items'), list):
            items = data['items']
        elif isinstance(data.get('results'), list):
            items = data['results']
        if not items:
            with self._lock:
                self._cache[cache_key] = None
            return None
        first = items[0] or {}
        item_id = (
            first.get('id')
            or first.get('item_id')
            or first.get('collectible_id')
        )
        result = {'item_id': str(item_id)} if item_id else None
        with self._lock:
            self._cache[cache_key] = result
        return result.get('item_id') if result else None

    def _get_insights(self, item_id: str, grade: str) -> Optional[Dict]:
        """Fetch fair-market-value data for a specific item + grade."""
        cache_key = f"insights:{item_id}:{grade}"
        with self._lock:
            if cache_key in self._cache:
                return self._cache[cache_key]
        data = self._api_get(
            f"/insights/item/{item_id}",
            params={"grade": grade},
        )
        with self._lock:
            self._cache[cache_key] = data
        return data

    def lookup(self, title: str) -> Optional[Dict[str, Any]]:
        """Return a price dict in the same shape as
        PriceChartingLookup.lookup, or None if no match.

        Only fires for titles with an explicit grading callout —
        GoCollect's value comes from grade-specific data, so an
        ungraded book wouldn't benefit. A daily-quota-exhausted
        instance returns None silently.
        """
        if not self.enabled:
            return None
        grade_info = extract_grade(title)
        if not grade_info:
            return None
        query = self._clean_query(title)
        if not query:
            return None
        item_id = self._search_item_id(query)
        if not item_id:
            return None
        insights = self._get_insights(item_id, grade_info['grade'])
        if not isinstance(insights, dict):
            return None

        # Response field names guessed from public docs — we try
        # several plausible names so a key drift doesn't break us.
        # Preferred: fair_market_value > recent_avg > median
        fmv = (
            insights.get('fair_market_value')
            or insights.get('fmv')
            or insights.get('value')
        )
        recent_avg = (
            insights.get('recent_average_sale')
            or insights.get('recent_avg')
            or insights.get('average_sale')
        )
        last_sold = (
            insights.get('last_sold_price')
            or insights.get('last_sale')
        )
        # Pick the most authoritative number we got.
        median = fmv or recent_avg or last_sold
        try:
            median = float(median) if median else None
        except (TypeError, ValueError):
            median = None
        if median is None or median <= 0:
            return None

        # Range: try to find low/high from the insights payload.
        low = insights.get('low_sale') or insights.get('min_price')
        high = insights.get('high_sale') or insights.get('max_price')
        try:
            low = float(low) if low else round(median * 0.85, 2)
            high = float(high) if high else round(median * 1.15, 2)
        except (TypeError, ValueError):
            low, high = round(median * 0.85, 2), round(median * 1.15, 2)
        # Sanity: low ≤ median ≤ high
        low, high = min(low, median), max(high, median)

        # Throttle so a 200-lot batch doesn't burst-fire and immediately
        # exhaust the daily quota.
        time.sleep(0.2)

        return {
            "median": round(median, 2),
            "low": round(low, 2),
            "high": round(high, 2),
            "count": 1,
            "source": (
                f"gocollect ({grade_info['service']} "
                f"{grade_info['grade']})"
            ),
            "ebay_count": 0,
            "mercari_count": 0,
            "pricecharting_count": 0,
            "gocollect_count": 1,
            "query": query,
            "gc_item_id": item_id,
            "gc_grade": grade_info['grade'],
            "gc_service": grade_info['service'],
        }
