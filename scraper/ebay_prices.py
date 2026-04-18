import httpx
import requests as _requests
import base64
import json
import re
import time
import statistics
import pandas as pd
from typing import Optional


class EbayPriceLookup:
    def __init__(self, app_id: str, cert_id: str):
        self.app_id = app_id
        self.cert_id = cert_id
        self._token: Optional[str] = None

    def _get_token(self) -> str:
        if self._token:
            return self._token
        credentials = base64.b64encode(f"{self.app_id}:{self.cert_id}".encode()).decode()
        resp = httpx.post(
            "https://api.ebay.com/identity/v1/oauth2/token",
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Authorization": f"Basic {credentials}",
            },
            data={
                "grant_type": "client_credentials",
                "scope": "https://api.ebay.com/oauth/api_scope",
            },
            timeout=15,
        )
        resp.raise_for_status()
        self._token = resp.json()["access_token"]
        return self._token

    def _clean_title(self, title: str) -> str:
        """Clean auction titles for better eBay search results."""
        # Remove leading price like "$20 "
        clean = re.sub(r'^\$\d+\.?\d*\s*', '', title)
        # Remove trailing ellipsis
        clean = clean.rstrip('.').strip()
        # Remove quantity prefixes like "Qty-2 " or "Qty:5 "
        clean = re.sub(r'^Qty[:\-]\d+\s*', '', clean, flags=re.IGNORECASE)
        # Truncate very long titles to first ~8 words for better search
        words = clean.split()
        if len(words) > 8:
            clean = ' '.join(words[:8])
        return clean

    def _filter_outliers(self, prices: list) -> list:
        """Remove statistical outliers using IQR method."""
        if len(prices) < 4:
            return prices
        sorted_p = sorted(prices)
        q1 = statistics.quantiles(sorted_p, n=4)[0]
        q3 = statistics.quantiles(sorted_p, n=4)[2]
        iqr = q3 - q1
        lo = q1 - 1.5 * iqr
        hi = q3 + 1.5 * iqr
        return [p for p in prices if lo <= p <= hi]

    def _scrape_ebay_sold_prices(self, query: str, max_prices: int = 30) -> list:
        """Scrape actual sold prices from eBay's sold listings page.

        Returns a list of sold prices (float). Empty list if scraping fails.
        """
        session = self._get_scrape_session()
        params = {
            "_nkw": query,
            "LH_Sold": "1",
            "LH_Complete": "1",
            "_ipg": "60",  # 60 results per page
        }
        try:
            resp = session.get(
                "https://www.ebay.com/sch/i.html",
                params=params,
                timeout=15,
            )
            if resp.status_code != 200 or len(resp.text) < 1000:
                return []

            html = resp.text

            # Extract prices from s-item__price spans
            # Handles: <span class="s-item__price">$24.99</span>
            # And: <span class="s-item__price"><span class="POSITIVE">$24.99</span></span>
            # Skip range listings ("$10.00 to $20.00") since those are ambiguous
            prices = []
            # Find all s-item__price blocks
            blocks = re.findall(
                r'class="s-item__price"[^>]*>(.*?)</span>\s*</div>',
                html,
                flags=re.DOTALL,
            )
            if not blocks:
                # Fallback: simpler pattern
                blocks = re.findall(r'class="s-item__price"[^>]*>([^<]+(?:<[^>]+>[^<]+</[^>]+>)*)', html)

            for block in blocks:
                # Skip ranges ("X to Y")
                if ' to ' in block.lower():
                    continue
                # Extract the dollar amount
                m = re.search(r'\$([\d,]+(?:\.\d{2})?)', block)
                if m:
                    try:
                        p = float(m.group(1).replace(",", ""))
                        if 0.99 < p < 50000:  # Sanity bounds
                            prices.append(p)
                    except ValueError:
                        pass
                if len(prices) >= max_prices:
                    break

            # First result is often the "Shop on eBay" placeholder — skip if wildly inconsistent
            # Actually, placeholder has no price, so list naturally excludes it.
            return prices
        except Exception:
            return []

    def _scrape_mercari_sold_prices(self, query: str, max_prices: int = 30) -> list:
        """Scrape sold prices from Mercari search results.

        Mercari's search page renders a Next.js app with initial state JSON
        embedded in a __NEXT_DATA__ script tag. We parse that JSON and extract
        sold item prices.

        Returns a list of sold prices (float). Empty list if scraping fails.
        """
        session = self._get_scrape_session()
        params = {
            "keyword": query,
            "itemStatuses": "ITEM_STATUS_SOLD_OUT",
        }
        try:
            resp = session.get(
                "https://www.mercari.com/search/",
                params=params,
                timeout=15,
            )
            if resp.status_code != 200 or len(resp.text) < 1000:
                return []

            html = resp.text
            prices = []

            # 1. Try __NEXT_DATA__ JSON blob (Mercari uses Next.js)
            next_data_match = re.search(
                r'<script[^>]*id="__NEXT_DATA__"[^>]*>({.*?})</script>',
                html,
                flags=re.DOTALL,
            )
            if next_data_match:
                try:
                    data = json.loads(next_data_match.group(1))
                    # Walk the JSON tree looking for items with numeric 'price'
                    prices = self._extract_mercari_prices(data, max_prices)
                except (json.JSONDecodeError, ValueError):
                    prices = []

            # 2. Fallback: regex for prices in specific Mercari markup
            if not prices:
                # Mercari inline prices: "price":"1234" (cents) or "price":12.34
                for m in re.finditer(r'"price":\s*"?(\d+(?:\.\d+)?)"?', html):
                    try:
                        raw = float(m.group(1))
                        # Mercari sometimes stores price in cents — heuristic normalize
                        p = raw / 100 if raw > 1000 else raw
                        if 0.99 < p < 50000:
                            prices.append(round(p, 2))
                    except ValueError:
                        pass
                    if len(prices) >= max_prices:
                        break

            return prices[:max_prices]
        except Exception:
            return []

    def _extract_mercari_prices(self, node, max_prices: int, found=None) -> list:
        """Recursively walk Mercari's JSON tree collecting item prices."""
        if found is None:
            found = []
        if len(found) >= max_prices:
            return found

        if isinstance(node, dict):
            # Mercari items have 'price' and 'status' fields
            # status is typically 'ITEM_STATUS_SOLD_OUT' for sold items
            if 'price' in node:
                price_val = node.get('price')
                status = node.get('status', '')
                # Accept if explicitly sold or if we don't know status (already filtered URL)
                if status in ('ITEM_STATUS_SOLD_OUT', '', None) or 'SOLD' in str(status):
                    try:
                        p = float(price_val)
                        if 0.99 < p < 50000:
                            found.append(round(p, 2))
                    except (TypeError, ValueError):
                        pass
            for v in node.values():
                if len(found) >= max_prices:
                    break
                self._extract_mercari_prices(v, max_prices, found)
        elif isinstance(node, list):
            for v in node:
                if len(found) >= max_prices:
                    break
                self._extract_mercari_prices(v, max_prices, found)
        return found

    def _price_stats(self, prices: list) -> Optional[dict]:
        """Compute median, low (Q1), high (Q3) from a list of prices."""
        if not prices:
            return None
        sorted_p = sorted(prices)
        median = statistics.median(sorted_p)
        if len(sorted_p) >= 4:
            q = statistics.quantiles(sorted_p, n=4)
            low, high = q[0], q[2]
        else:
            low, high = min(sorted_p), max(sorted_p)
        return {
            "median": round(median, 2),
            "low": round(low, 2),
            "high": round(high, 2),
        }

    def lookup_price(self, title: str, limit: int = 8) -> Optional[float]:
        """Look up median resale price. Returns just the median (back-compat)."""
        result = self.lookup_price_range(title, limit=limit)
        return result["median"] if result else None

    def lookup_price_range(self, title: str, limit: int = 8) -> Optional[dict]:
        """Look up combined resale price statistics from eBay + Mercari sold data.

        Combines actual sold prices from both platforms for a more reliable
        cross-marketplace median. Falls back to eBay active listings if neither
        marketplace returns sold comps.

        Returns:
            {
                "median": float,
                "low": float,           # 25th percentile across all comps
                "high": float,          # 75th percentile across all comps
                "count": int,           # Total comp count
                "source": str,          # Combined source label
                "ebay_count": int,      # eBay sold comps contributed
                "mercari_count": int,   # Mercari sold comps contributed
            }
            or None if no data available.
        """
        query = self._clean_title(title)
        if len(query) < 5:
            return None

        # Scrape both marketplaces (throttled)
        ebay_prices = []
        mercari_prices = []
        try:
            time.sleep(0.3)
            ebay_prices = self._scrape_ebay_sold_prices(query)
            ebay_prices = self._filter_outliers(ebay_prices)
        except Exception:
            ebay_prices = []

        try:
            time.sleep(0.3)
            mercari_prices = self._scrape_mercari_sold_prices(query)
            mercari_prices = self._filter_outliers(mercari_prices)
        except Exception:
            mercari_prices = []

        combined = ebay_prices + mercari_prices

        # If we have enough real sold comps (≥3 combined), use them
        if len(combined) >= 3:
            combined = self._filter_outliers(combined)
            stats = self._price_stats(combined)
            if stats:
                if ebay_prices and mercari_prices:
                    source = "sold (eBay+Mercari)"
                elif ebay_prices:
                    source = "sold (eBay)"
                else:
                    source = "sold (Mercari)"
                return {
                    **stats,
                    "count": len(combined),
                    "source": source,
                    "ebay_count": len(ebay_prices),
                    "mercari_count": len(mercari_prices),
                }

        # Fallback: eBay active listings via Browse API
        try:
            token = self._get_token()
            resp = httpx.get(
                "https://api.ebay.com/buy/browse/v1/item_summary/search",
                headers={
                    "Authorization": f"Bearer {token}",
                    "X-EBAY-C-MARKETPLACE-ID": "EBAY_US",
                },
                params={
                    "q": query,
                    "filter": "buyingOptions:{FIXED_PRICE}",
                    "sort": "price",
                    "limit": str(limit),
                },
                timeout=10,
            )
            if resp.status_code != 200:
                return None

            items = resp.json().get("itemSummaries", [])
            prices = [
                float(item["price"]["value"])
                for item in items
                if float(item.get("price", {}).get("value", 0)) > 0.99
            ]
            prices = self._filter_outliers(prices)

            if not prices:
                return None

            stats = self._price_stats(prices)
            if not stats:
                return None

            return {
                **stats,
                "count": len(prices),
                "source": "active (eBay)",
                "ebay_count": len(prices),
                "mercari_count": 0,
            }
        except Exception:
            return None

    _scrape_session = None

    @classmethod
    def _get_scrape_session(cls):
        """Reusable requests session with browser-like headers for eBay scraping."""
        if cls._scrape_session is None:
            cls._scrape_session = _requests.Session()
            cls._scrape_session.headers.update({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-User": "?1",
            })
        return cls._scrape_session

    def _scrape_ebay_count(self, query: str, sold: bool = False) -> Optional[int]:
        """Scrape eBay search result count from the website.

        Args:
            query: Search query string
            sold: If True, search sold/completed items; otherwise active listings

        Returns:
            Total result count, or None if scraping fails
        """
        session = self._get_scrape_session()
        params = {"_nkw": query}
        if sold:
            params["LH_Sold"] = "1"
            params["LH_Complete"] = "1"

        try:
            resp = session.get(
                "https://www.ebay.com/sch/i.html",
                params=params,
                timeout=15,
            )
            if resp.status_code != 200 or len(resp.text) < 1000:
                return None

            # eBay embeds the count in JSON as "count":"1234,"
            match = re.search(r'"count":\s*"?([\d,]+)', resp.text)
            if match:
                return int(match.group(1).replace(",", ""))

            # Fallback: "X results" in heading
            match2 = re.search(r'([\d,]+)\s+results', resp.text)
            if match2:
                return int(match2.group(1).replace(",", ""))

            return None
        except Exception:
            return None

    def _demand_score(self, title: str) -> Optional[float]:
        """Fallback demand score using Browse API when scraping fails.

        Uses active listing volume and price consistency as a proxy.
        Returns a 0-100 score, or None.
        """
        query = self._clean_title(title)
        if len(query) < 5:
            return None

        try:
            token = self._get_token()
            resp = httpx.get(
                "https://api.ebay.com/buy/browse/v1/item_summary/search",
                headers={
                    "Authorization": f"Bearer {token}",
                    "X-EBAY-C-MARKETPLACE-ID": "EBAY_US",
                },
                params={
                    "q": query,
                    "filter": "buyingOptions:{FIXED_PRICE}",
                    "sort": "price",
                    "limit": "10",
                },
                timeout=10,
            )
            if resp.status_code != 200:
                return None

            data = resp.json()
            total = data.get("total", 0)
            items = data.get("itemSummaries", [])
            if total == 0 or not items:
                return None

            prices = [
                float(item["price"]["value"])
                for item in items
                if float(item.get("price", {}).get("value", 0)) > 0.99
            ]
            if len(prices) < 2:
                return 30.0

            mean_price = statistics.mean(prices)
            cv = (statistics.stdev(prices) / mean_price * 100) if mean_price > 0 else 100
            price_score = max(20, min(80, 80 - cv * 0.5))

            if total >= 50:
                size_bonus = 15
            elif total >= 10:
                size_bonus = 5
            else:
                size_bonus = -10

            return round(max(10, min(95, price_score + size_bonus)), 0)
        except Exception:
            return None

    def lookup_str(self, title: str) -> tuple:
        """Look up actual sell-through rate by scraping eBay sold vs active listings.

        STR = sold / (sold + active) * 100

        Falls back to a demand score from the Browse API if scraping fails.

        Returns:
            (str_value, source) where source is "sold" or "demand", or (None, None)
        """
        query = self._clean_title(title)
        if len(query) < 5:
            return None, None

        try:
            # Small delay to avoid rate limiting
            time.sleep(0.5)

            sold_count = self._scrape_ebay_count(query, sold=True)
            if sold_count is None:
                score = self._demand_score(title)
                return (score, "demand") if score is not None else (None, None)

            time.sleep(0.5)

            active_count = self._scrape_ebay_count(query, sold=False)
            if active_count is None:
                score = self._demand_score(title)
                return (score, "demand") if score is not None else (None, None)

            total = sold_count + active_count
            if total == 0:
                return None, None

            return round((sold_count / total) * 100, 1), "sold"
        except Exception:
            score = self._demand_score(title)
            return (score, "demand") if score is not None else (None, None)

    def sample_auction_str(self, df: pd.DataFrame, sample_size: int = 3,
                           progress_callback=None) -> dict:
        """Estimate STR per auction by sampling a few items from each.

        Args:
            df: Full Phase 1 DataFrame with 'auction' and 'title' columns
            sample_size: Number of items to sample per auction
            progress_callback: Optional callable(current, total) for progress updates

        Returns:
            Dict mapping auction name -> estimated STR % (or None)
        """
        auction_strs = {}
        auctions = df.groupby('auction')
        auction_names = list(auctions.groups.keys())
        total = len(auction_names)

        for i, auction_name in enumerate(auction_names):
            group = auctions.get_group(auction_name)

            # Sample items: prefer items with longer titles (more searchable)
            ranked = group.copy()
            ranked['_title_len'] = ranked['title'].str.len()
            ranked = ranked.sort_values('_title_len', ascending=False)
            sample = ranked.head(sample_size)

            str_results = []
            for _, row in sample.iterrows():
                result, _ = self.lookup_str(row.get('title', ''))
                if result is not None:
                    str_results.append(result)

            auction_strs[auction_name] = round(statistics.mean(str_results), 1) if str_results else None

            if progress_callback:
                progress_callback(i + 1, total)

        return auction_strs

    def batch_lookup(self, df: pd.DataFrame, progress_callback=None) -> pd.DataFrame:
        """Add price range, STR, and source columns to a DataFrame.

        For each row, performs an eBay sold-prices lookup (falling back to active
        listings) and an STR lookup. Median price populates 'est_resale' for
        back-compat with ROI calculations.

        Args:
            df: DataFrame with a 'title' column (or 'enriched_title')
            progress_callback: Optional callable(current, total)

        Returns:
            DataFrame with these columns added:
                est_resale, price_low, price_high, comp_count, price_source,
                ebay_str, str_source
        """
        df = df.copy()
        medians = []
        lows = []
        highs = []
        counts = []
        ebay_counts = []
        mercari_counts = []
        price_sources = []
        str_values = []
        str_sources = []
        total = len(df)

        for i, (_, row) in enumerate(df.iterrows()):
            # Prefer enriched title (from AI audit) for better eBay search results
            title = row.get('enriched_title') or row.get('title', '')
            price_info = self.lookup_price_range(title)
            if price_info:
                medians.append(price_info["median"])
                lows.append(price_info["low"])
                highs.append(price_info["high"])
                counts.append(price_info["count"])
                ebay_counts.append(price_info.get("ebay_count", 0))
                mercari_counts.append(price_info.get("mercari_count", 0))
                price_sources.append(price_info["source"])
            else:
                medians.append(None)
                lows.append(None)
                highs.append(None)
                counts.append(0)
                ebay_counts.append(0)
                mercari_counts.append(0)
                price_sources.append(None)

            str_pct, str_src = self.lookup_str(title)
            str_values.append(str_pct)
            str_sources.append(str_src)

            if progress_callback:
                progress_callback(i + 1, total)

        df['est_resale'] = medians
        df['price_low'] = lows
        df['price_high'] = highs
        df['comp_count'] = counts
        df['ebay_comps'] = ebay_counts
        df['mercari_comps'] = mercari_counts
        df['price_source'] = price_sources
        df['ebay_str'] = str_values
        df['str_source'] = str_sources
        return df
