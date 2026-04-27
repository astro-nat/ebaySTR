import asyncio
import httpx
import pandas as pd
import re
import json
import os
from datetime import datetime, timedelta, timezone
from typing import List, Dict

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

LOT_SEARCH_QUERY = """
query LotSearch($auctionId: Int!, $pageNumber: Int!) {
  lotSearch(input: {auctionId: $auctionId, searchText: ""}, pageNumber: $pageNumber) {
    pagedResults {
      totalCount
      pageNumber
      results {
        id
        lotNumber
        lead
        description
        category { categoryName }
        lotState { highBid bidCount status timeLeft }
        pictures { thumbnailLocation hdThumbnailLocation fullSizeLocation }
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
        if re.search(self.ship_killers, text):
            return "HARD"
        # Check description for pickup-only language
        if description and self._PICKUP_ONLY_RE.search(description):
            return "HARD"
        if re.search(self.mailbox_winners, text):
            return "EASY"
        return "NEUTRAL"

    def estimate_total_cost(self, bid: float) -> float:
        """Estimate acquisition cost per item: bid + buyer premium."""
        premium = bid * (self.buyer_premium_pct / 100.0)
        return bid + premium

    async def _graphql(self, client: httpx.AsyncClient, operation: str, query: str, variables: dict) -> dict:
        payload = {
            "operationName": operation,
            "query": query,
            "variables": variables,
        }
        response = await client.post(
            self.graphql_url,
            headers=self.headers,
            json=payload,
            timeout=self.timeout,
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

    async def fetch_auctions(self, client: httpx.AsyncClient, zip_code: str, radius: int) -> List[Dict]:
        """Fetch auctions from HiBid. Use empty zip for nationwide."""
        variables = {
            "zip": zip_code,
            "miles": radius,
            "searchText": "",
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
            date_end = a.get('date_end', '')
            if not date_end:
                filtered.append(a)  # Keep auctions with no end date (can't filter)
                continue
            try:
                # HiBid returns naive datetimes like "2026-04-16T00:00:00"
                end_dt = datetime.fromisoformat(date_end.replace('Z', ''))
                if now <= end_dt <= cutoff:
                    filtered.append(a)
            except (ValueError, TypeError):
                filtered.append(a)  # Keep if we can't parse the date
        return filtered

    async def _fetch_all_lot_pages(self, client: httpx.AsyncClient, auction_id: int) -> List[Dict]:
        """Fetch every lot in an auction.

        HiBid's current schema pages at a fixed size of 100 lots with
        `pageNumber` (1-based) as a sibling arg to `input`. We call page 1
        to get totalCount, then paginate through the remaining pages.
        """
        lots: List[Dict] = []
        total_count = 0

        # First request: page 1 gives us both results and totalCount
        variables = {"auctionId": auction_id, "pageNumber": 1}
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
            variables = {"auctionId": auction_id, "pageNumber": page_number}
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
                                     source: str = "Local Pickup"):
        """Fetch + process one auction's lots.

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
            lots = await self._fetch_all_lot_pages(client, auction_id)
            result["raw_count"] = len(lots)

            try:
                closing_fmt = datetime.fromisoformat(date_end).strftime("%b %d") if date_end else ""
            except (ValueError, TypeError):
                closing_fmt = date_end

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
                total_cost = self.estimate_total_cost(current_bid)

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

                processed_lots.append({
                    "lot_id": lot_id,
                    "auction": auction_name,
                    "auction_link": f"https://hibid.com/auction/{auction_id}",
                    "closing_date": closing_fmt,
                    "source": source,
                    "title": title,
                    "lot_link": f"https://hibid.com/lot/{lot_id}",
                    "category": category,
                    "current_bid": current_bid,
                    "bid_count": state.get('bidCount', 0) or 0,
                    "est_cost": round(total_cost, 2),
                    "status": state.get('status', '') or '',
                    "time_left": state.get('timeLeft', '') or '',
                    "description": description,
                    "logistics_ease": logistics,
                    "thumbnail_url": thumbnail_url,
                    "hd_thumbnail_url": hd_thumbnail_url,
                    "fullsize_url": fullsize_url,
                    "image_count": len(pictures),
                })
            result["lots"] = processed_lots
        except Exception as e:
            result["error"] = f"{type(e).__name__}: {e}"
        return result

    async def _fetch_lots_batch(self, client: httpx.AsyncClient, auctions: List[Dict],
                                source: str, batch_size: int = 20,
                                progress_callback=None, progress_offset: int = 0,
                                grand_total: int = None, phase_label: str = ""):
        """Fetch lots for a list of auctions in concurrent batches.

        Returns a dict: {
            "lots": [...],
            "raw_count": int,
            "filtered_by_category": int,
            "errors": [{"auction_id", "auction_name", "error"}, ...],
            "per_auction": [{"auction_id", "auction_name", "raw_count", "kept": int}, ...]
        }
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

        for i in range(0, total, batch_size):
            batch = auctions[i:i + batch_size]
            tasks = [
                self.fetch_lots_for_auction(
                    client, a['auction_id'], a['name'], a.get('date_end', ''), source
                )
                for a in batch
            ]
            results = await asyncio.gather(*tasks)
            for r in results:
                agg["lots"].extend(r["lots"])
                agg["raw_count"] += r["raw_count"]
                agg["filtered_by_category"] += r["filtered_by_category"]
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

            if progress_callback:
                current = progress_offset + min(i + batch_size, total)
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

        async with httpx.AsyncClient() as client:
            _report(0, 1, "Discovering local auctions...")
            local_auctions = await self.fetch_auctions(client, self.zip_code, self.radius)
            local_auctions = self._filter_by_closing_date(local_auctions)
            local_ids = {a['auction_id'] for a in local_auctions}
            for a in local_auctions:
                a['source'] = 'Local Pickup'

            remote_auctions: List[Dict] = []
            if self.include_nationwide:
                _report(0, 1, "Discovering nationwide auctions...")
                nationwide_raw = await self.fetch_auctions(client, "", 0)
                remote_auctions = [a for a in nationwide_raw if a['auction_id'] not in local_ids]
                remote_auctions = self._filter_by_closing_date(remote_auctions)
                remote_auctions = sorted(remote_auctions, key=lambda a: a.get('date_end', ''))
                for a in remote_auctions:
                    a['source'] = 'Ship'

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
                "thumbnail_url": str — first lot's thumbnail (auction "cover")
            }

        Kept backward-compat: callers that previously received a plain
        list of categories can treat the dict as an iterable of "categories".
        """
        # Note: the GraphQL LotSearch query takes `pageNumber` as a sibling
        # of `input` (per the schema note at the top of this file). The old
        # `pageIndex`/`pageSize` form was removed by HiBid; using it here
        # silently failed and made the preview column blank.
        variables = {"auctionId": auction_id, "pageNumber": 1}
        empty = {"categories": [], "cat_counts": {}, "titles": [], "thumbnail_url": ""}
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
        thumbnail_url = ""
        for lot in lots:
            name = self._extract_category_name(lot.get('category'))
            if name:
                cat_counts[name] = cat_counts.get(name, 0) + 1
            lead = (lot.get('lead') or '').strip()
            if lead:
                titles.append(lead)
            # Use the first lot with a picture as the auction's "cover"
            # image. HiBid's lot 1 is usually a representative or featured
            # piece, so this doubles as a decent visual identifier.
            if not thumbnail_url:
                pictures = lot.get('pictures') or []
                if pictures:
                    first = pictures[0] or {}
                    thumbnail_url = (
                        first.get('hdThumbnailLocation')
                        or first.get('thumbnailLocation')
                        or ''
                    )

        return {
            "categories": sorted(cat_counts.keys()),
            "cat_counts": cat_counts,
            "titles": titles,
            "thumbnail_url": thumbnail_url,
        }

    async def sample_categories_batch(
        self, auctions: List[Dict], sample_size: int = 20,
        batch_size: int = 15, progress_callback=None,
    ) -> Dict[int, Dict[str, list]]:
        """Sample categories + titles for a batch of auctions concurrently.

        Returns {auction_id: {"categories": [...], "cat_counts": {...},
        "titles": [...]}}. Useful for the picker UI to show "what's in
        this auction" without fetching every lot.
        """
        out: Dict[int, Dict[str, list]] = {}
        total = len(auctions)
        async with httpx.AsyncClient() as client:
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
                            "thumbnail_url": "",
                        }
                    else:
                        out[a['auction_id']] = {
                            "categories": [], "cat_counts": {}, "titles": [],
                            "thumbnail_url": "",
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
    ) -> pd.DataFrame:
        """Fetch full lot detail for a caller-supplied list of auctions.

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

        async with httpx.AsyncClient() as client:
            if local_auctions:
                local_label = (
                    f"Local pickup ({len(local_auctions)})"
                    if not remote_auctions
                    else f"Local pickup ({len(local_auctions)} of {grand_total})"
                )
                r = await self._fetch_lots_batch(
                    client, local_auctions, "Local Pickup",
                    progress_callback=progress_callback, progress_offset=0,
                    grand_total=grand_total, phase_label=local_label,
                )
                agg_lots.extend(r["lots"])
                raw_count += r["raw_count"]
                filtered_by_category += r["filtered_by_category"]
                per_auction.extend(r["per_auction"])
                errors.extend(r["errors"])

            if remote_auctions:
                nationwide_label = f"Nationwide ({len(remote_auctions)} of {grand_total})"
                r = await self._fetch_lots_batch(
                    client, remote_auctions, "Ship",
                    progress_callback=progress_callback,
                    progress_offset=len(local_auctions),
                    grand_total=grand_total, phase_label=nationwide_label,
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
