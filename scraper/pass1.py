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
query LotSearch($auctionId: Int!) {
  lotSearch(input: {auctionId: $auctionId, searchText: ""}) {
    pagedResults {
      totalCount
      results {
        id
        lotNumber
        lead
        description
        category { categoryName }
        lotState { highBid bidCount status timeLeft }
      }
    }
  }
}
"""


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

    # Patterns that indicate an item is pickup-only (checked against description too)
    _PICKUP_ONLY_RE = re.compile(
        r'local\s+pick\s*-?\s*up\s+only'
        r'|pick\s*-?\s*up\s+only'
        r'|no\s+shipp?ing'
        r'|will\s+not\s+ship'
        r'|cannot\s+be\s+shipped'
        r'|can\s*n?o?t\s+ship'
        r'|must\s+pick\s*-?\s*up'
        r'|in[- ]?store\s+pick\s*-?\s*up\s+only'
        r'|not\s+available\s+for\s+shipp?ing',
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
        response.raise_for_status()
        data = response.json()
        if "errors" in data:
            raise RuntimeError(f"GraphQL errors: {data['errors']}")
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

    async def fetch_lots_for_auction(self, client: httpx.AsyncClient, auction_id: int,
                                     auction_name: str = "", date_end: str = "",
                                     source: str = "Local Pickup") -> List[Dict]:
        variables = {"auctionId": auction_id}

        try:
            data = await self._graphql(client, "LotSearch", LOT_SEARCH_QUERY, variables)
            paged = data.get("lotSearch", {}).get("pagedResults", {})
            lots = paged.get("results", [])

            try:
                closing_fmt = datetime.fromisoformat(date_end).strftime("%b %d") if date_end else ""
            except (ValueError, TypeError):
                closing_fmt = date_end

            # Normalize category filter once per batch
            cat_keywords = [c.strip().lower() for c in (self.category_filter or []) if c and c.strip()]

            processed_lots = []
            for lot in lots:
                title = lot.get('lead', '')
                categories = lot.get('category', [])
                category = categories[0]['categoryName'] if categories else ''
                description = lot.get('description', '')

                # Apply category filter (substring, case-insensitive, against
                # category name OR title — auctioneer tagging is inconsistent
                # so falling back to title catches e.g. a "fishing rod" lot
                # mis-tagged as "Sporting Goods"-only)
                if cat_keywords:
                    haystack = f"{category} {title}".lower()
                    if not any(kw in haystack for kw in cat_keywords):
                        continue

                state = lot.get('lotState', {})
                logistics = self.classify_logistics(title, category, description)
                current_bid = state.get('highBid', 0.0)
                total_cost = self.estimate_total_cost(current_bid)

                lot_id = lot.get('id')
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
                    "bid_count": state.get('bidCount', 0),
                    "est_cost": round(total_cost, 2),
                    "status": state.get('status', ''),
                    "time_left": state.get('timeLeft', ''),
                    "description": lot.get('description', ''),
                    "logistics_ease": logistics,
                })
            return processed_lots
        except Exception as e:
            print(f"Warning: failed to fetch lots for auction {auction_id}: {e}")
            return []

    async def _fetch_lots_batch(self, client: httpx.AsyncClient, auctions: List[Dict],
                                source: str, batch_size: int = 20,
                                progress_callback=None, progress_offset: int = 0,
                                grand_total: int = None, phase_label: str = "") -> List[Dict]:
        """Fetch lots for a list of auctions in concurrent batches.

        `grand_total` is the total auction count across ALL phases; when
        provided, progress is reported against it so the bar doesn't reset
        when switching from local -> nationwide. `phase_label` is included
        in the progress text so the user knows which phase is running.
        """
        all_lots = []
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
            for sublist in results:
                all_lots.extend(sublist)

            if progress_callback:
                current = progress_offset + min(i + batch_size, total)
                progress_callback(current, effective_total, phase_label)

        return all_lots

    async def run(self, progress_callback=None) -> pd.DataFrame:
        """Run the full scrape.

        progress_callback signature: (current:int, total:int, label:str) -> None
        The label describes the current phase so the UI can show e.g.
        "Discovering local auctions..." vs "Fetching nationwide lots".
        """
        def _report(current, total, label):
            if progress_callback:
                progress_callback(current, total, label)

        async with httpx.AsyncClient() as client:
            # --- Phase 1: fetch BOTH auction lists up front so we know the
            # grand total before starting any lot-fetching. That way the
            # progress bar and count are honest from the first tick.
            _report(0, 1, "Discovering local auctions...")
            local_auctions = await self.fetch_auctions(client, self.zip_code, self.radius)
            local_auctions = self._filter_by_closing_date(local_auctions)
            local_ids = {a['auction_id'] for a in local_auctions}

            remote_auctions: List[Dict] = []
            if self.include_nationwide:
                _report(0, 1, "Discovering nationwide auctions...")
                nationwide_raw = await self.fetch_auctions(client, "", 0)
                remote_auctions = [a for a in nationwide_raw if a['auction_id'] not in local_ids]
                remote_auctions = self._filter_by_closing_date(remote_auctions)
                remote_auctions = sorted(remote_auctions, key=lambda a: a.get('date_end', ''))

            grand_total = len(local_auctions) + len(remote_auctions)

            # If nothing to do, short-circuit with a clean 0/0 tick.
            if grand_total == 0:
                _report(0, 0, "No auctions matched the filters")
                return pd.DataFrame()

            # --- Phase 2: fetch lots for local, then nationwide, using the
            # grand total so the bar only moves forward.
            local_label = (
                f"Local pickup ({len(local_auctions)})"
                if not remote_auctions
                else f"Local pickup ({len(local_auctions)} of {grand_total})"
            )
            all_lots = await self._fetch_lots_batch(
                client, local_auctions, "Local Pickup",
                progress_callback=progress_callback, progress_offset=0,
                grand_total=grand_total, phase_label=local_label,
            )

            if remote_auctions:
                nationwide_label = f"Nationwide ({len(remote_auctions)} of {grand_total})"
                nationwide_lots = await self._fetch_lots_batch(
                    client, remote_auctions, "Ship",
                    progress_callback=progress_callback, progress_offset=len(local_auctions),
                    grand_total=grand_total, phase_label=nationwide_label,
                )
                all_lots.extend(nationwide_lots)

            df = pd.DataFrame(all_lots)

            if not df.empty:
                df = df[df['logistics_ease'] != "HARD"]
                df = df[df['status'] != "CLOSED"]
                df = df[df['time_left'] != "Bidding Closed"]
                df = df.sort_values('closing_date').reset_index(drop=True)
            return df
