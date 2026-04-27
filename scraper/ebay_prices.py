import httpx
import requests as _requests
import base64
import json
import re
import time
import statistics
import threading
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional


class EbayPriceLookup:
    def __init__(self, app_id: str, cert_id: str, pricecharting=None):
        """eBay/Mercari price lookup, optionally augmented with PriceCharting.

        Args:
            app_id, cert_id: eBay developer credentials.
            pricecharting: Optional PriceChartingLookup instance. When set,
                lots whose titles classify as games / cards / comics get a
                PriceCharting lookup BEFORE the eBay/Mercari scrape. PC's
                aggregated sold data is materially better for these niches.
                Pass None (or omit) to disable.
        """
        self.app_id = app_id
        self.cert_id = cert_id
        self.pricecharting = pricecharting
        self._token: Optional[str] = None
        # Guard token fetch under parallel workers — avoids redundant OAuth calls
        self._token_lock = threading.Lock()

    def _get_token(self) -> str:
        if self._token:
            return self._token
        with self._token_lock:
            if self._token:  # double-check after acquiring lock
                return self._token
            return self._fetch_token()

    def _fetch_token(self) -> str:
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

    # Condition / packaging tokens HiBid often appends to lot titles ("Very
    # Good", "Damaged", "No In Packaging", "New Open Box", etc.). Stripping
    # these tightens the eBay query — a full-title search including
    # "Damaged No In Packaging" matches almost nothing.
    _CONDITION_NOISE_RE = re.compile(
        r'\b(very\s+good|like\s+new|brand\s+new|open\s+box|no\s+in\s+packaging|'
        r'in\s+original\s+packaging|no\s+packaging|condition|damaged|untested|'
        r'for\s+parts|as[- ]is|sealed|unopened|unused|new\b|used\b|good\b|fair\b|'
        r'poor\b|mint\b|excellent\b)\b',
        re.IGNORECASE,
    )

    def _clean_title(self, title: str, max_words: int = 6) -> str:
        """Clean auction titles for better eBay search results.

        max_words caps the query length. eBay's search is extremely
        sensitive to extra terms — 7+ word queries commonly return ZERO
        matches even for well-known products. 4-6 words is the sweet
        spot for most lots; progressive shortening downstream handles
        cases where even that is too specific.
        """
        # Remove ALL $NN / $NN.NN tokens anywhere in the title — HiBid often
        # sprinkles retail-value hints like "Gucci Bag $250 Retail" that
        # poison the eBay search (we'd get $250-priced listings regardless
        # of the actual product).
        clean = re.sub(r'\$\d+(?:\.\d{1,2})?\b', '', title)
        # Remove "Retail Value" / "MSRP" / "Est. Value" boilerplate so those
        # words don't leak into the query
        clean = re.sub(
            r'\b(retail(\s+value)?|msrp|est(\.|imated)?\s*(value|worth))\b',
            '', clean, flags=re.IGNORECASE,
        )
        # Remove quantity prefixes like "Qty-2 " or "Qty:5 "
        clean = re.sub(r'\bQty[:\-]?\s*\d+\s*', '', clean, flags=re.IGNORECASE)
        # Remove HiBid condition/packaging boilerplate that makes queries
        # over-specific (see _CONDITION_NOISE_RE above)
        clean = self._CONDITION_NOISE_RE.sub('', clean)
        # Strip parenthetical asides like "(Renewed)", "(Open Box)" — these
        # are usually marketplace qualifiers, not product features
        clean = re.sub(r'\([^)]{1,25}\)', '', clean)
        # Collapse punctuation into whitespace so eBay's tokenizer works
        clean = re.sub(r'[,;:/\\|]+', ' ', clean)
        # Normalize whitespace
        clean = re.sub(r'\s+', ' ', clean).strip(' .,-')
        # Truncate to max_words for better search. Drop trailing stopwords
        # and single-letter fragments so the tail of the query isn't junk.
        words = [w for w in clean.split() if w]
        if len(words) > max_words:
            words = words[:max_words]
        while words and (
            len(words[-1]) <= 1
            or words[-1].lower() in {'and', 'or', 'the', 'with', 'for', '&'}
        ):
            words.pop()
        return ' '.join(words)

    def _query_variants(self, title: str) -> list:
        """Produce a list of progressively shorter eBay queries from a title.

        eBay search often returns zero hits for long, specific queries but
        plenty for shorter keyword-only ones. Rather than guess the right
        length up front, we try full → 4 → 3 words in order and stop as
        soon as we get enough comps. The caller is responsible for early-
        exit; this just hands back the candidate list.

        Dedupes adjacent identical queries (e.g. when max_words == actual
        word count).
        """
        variants: list = []
        for cap in (6, 4, 3):
            q = self._clean_title(title, max_words=cap)
            if q and len(q) >= 5 and q not in variants:
                variants.append(q)
        return variants

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

        Uses progressive query shortening: we try the full cleaned title first
        (6 words), then fall back to 4 words and finally 3 words if no matches
        turn up. eBay's search is extremely intolerant of extra terms — a
        7-word query commonly returns zero, while the same first 3 words
        return hundreds. We stop as soon as we clear the ≥3 sold-comp bar.

        Returns:
            {
                "median": float,
                "low": float,           # 25th percentile across all comps
                "high": float,          # 75th percentile across all comps
                "count": int,           # Total comp count
                "source": str,          # Combined source label
                "ebay_count": int,      # eBay sold comps contributed
                "mercari_count": int,   # Mercari sold comps contributed
                "query": str,           # The query variant that produced hits
            }
            or None if no data available.
        """
        # Category-specific: try PriceCharting first for games / TCG / comics.
        # When their classifier matches and they have the product, the data
        # is materially better than scraping eBay sold (already aggregated,
        # condition-normalized, canonical product ID). Falls through to the
        # eBay/Mercari path on miss.
        if self.pricecharting is not None and self.pricecharting.enabled:
            try:
                pc_result = self.pricecharting.lookup(title)
                if pc_result is not None:
                    return pc_result
            except Exception:
                pass  # Don't let PC outages break the scan

        variants = self._query_variants(title)
        if not variants:
            return None

        # Try each query variant in descending specificity. First one to
        # clear the ≥3-comp bar wins. Also remember the best "partial"
        # (1-2 comp) result in case nothing clears the bar — still better
        # than falling all the way through to active listings.
        best_partial = None  # (combined, ebay, mercari, query)

        for idx, query in enumerate(variants):
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
                    # Annotate source with the fallback level if we had to
                    # drop down — helps the user eyeball whether the comps
                    # were for the specific product vs a generic keyword.
                    if idx > 0:
                        source = f"{source} [short query]"
                    return {
                        **stats,
                        "count": len(combined),
                        "source": source,
                        "ebay_count": len(ebay_prices),
                        "mercari_count": len(mercari_prices),
                        "query": query,
                    }

            # Remember the first variant that produced ANY comps so we can
            # surface at least a rough number if none hit the ≥3 bar.
            if combined and best_partial is None:
                best_partial = (list(combined), list(ebay_prices),
                                list(mercari_prices), query)

        # All sold-comp variants failed to clear ≥3. Use the best partial
        # if we have one (1-2 comps) before falling back to active listings.
        if best_partial is not None:
            combined, ebay_prices, mercari_prices, matched_q = best_partial
            stats = self._price_stats(combined)
            if stats:
                return {
                    **stats,
                    "count": len(combined),
                    "source": "sold (thin comps)",
                    "ebay_count": len(ebay_prices),
                    "mercari_count": len(mercari_prices),
                    "query": matched_q,
                }

        # Nothing sold. Fall back to eBay active listings via Browse API —
        # use the SHORTEST variant since the longer ones already struck out.
        fallback_query = variants[-1]
        try:
            token = self._get_token()
            resp = httpx.get(
                "https://api.ebay.com/buy/browse/v1/item_summary/search",
                headers={
                    "Authorization": f"Bearer {token}",
                    "X-EBAY-C-MARKETPLACE-ID": "EBAY_US",
                },
                params={
                    "q": fallback_query,
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
                "query": fallback_query,
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

    def sample_auction_str(self, df: pd.DataFrame, sample_size: int = 2,
                           progress_callback=None,
                           granularity: str = "category") -> dict:
        """Sample STR per (auction, category) bucket — category-level sampling.

        Rationale: STR is a category signal, not per-lot. A "jewelry" STR and
        a "tools" STR are very different, but every "Nintendo Switch" lot in
        the same auction has essentially the same STR. Sampling per-category
        gives realistic per-row variance (jewelry rows show jewelry STR,
        tools rows show tools STR) at a fraction of the per-lot cost.

        For a 1000-lot auction with, say, 12 distinct categories: we do
        ~12 × 2 = 24 STR scrapes instead of 1000. The per-row values then
        differ by category so the user actually sees variance.

        Args:
            df: DataFrame with 'auction' + 'category' + 'title' columns
            sample_size: Items to sample per bucket (2 is usually enough —
                STR is noisy per-query, averaging 2 stabilizes it)
            progress_callback: Optional callable(current, total)
            granularity: "category" (default) = group by (auction, category);
                "auction" = back-compat, one STR per auction.

        Returns:
            Dict with a unified lookup key. For category granularity the key
            is (auction_name, category_name); for auction granularity it's
            just auction_name. Callers should use the helper `get_str()` or
            check granularity to know how to look things up.

            Each value is (str_value, source). Source is e.g.
            "sold (sampled, 3 lots)" so the UI can show what we did.
        """
        result: dict = {"__granularity__": granularity}

        if granularity == "category" and 'category' in df.columns:
            # Build (auction, category) buckets. Drop empty category -> treat as
            # its own bucket per auction so we don't lump categorized with
            # un-categorized.
            working = df.copy()
            working['_cat_key'] = working['category'].fillna('').astype(str).str.strip()
            working.loc[working['_cat_key'] == '', '_cat_key'] = '(uncategorized)'
            buckets = working.groupby(['auction', '_cat_key'])
            bucket_keys = list(buckets.groups.keys())
        else:
            # Fallback / back-compat: group by auction only
            granularity = "auction"
            result["__granularity__"] = "auction"
            buckets = df.groupby('auction')
            bucket_keys = [(a,) for a in buckets.groups.keys()]

        total = len(bucket_keys)
        title_col = 'enriched_title' if 'enriched_title' in df.columns else 'title'

        for i, key in enumerate(bucket_keys):
            if granularity == "category":
                group = buckets.get_group(key)
            else:
                group = buckets.get_group(key[0])

            # Pick longest-title samples — usually the most searchable
            ranked = group.copy()
            ranked['_title_len'] = ranked[title_col].fillna('').astype(str).str.len()
            ranked = ranked.sort_values('_title_len', ascending=False)
            sample = ranked.head(sample_size)

            str_results = []
            source_counts: dict = {}
            for _, row in sample.iterrows():
                title = row.get(title_col) or row.get('title', '')
                res, src = self.lookup_str(title)
                if res is not None:
                    str_results.append(res)
                    if src:
                        source_counts[src] = source_counts.get(src, 0) + 1

            if str_results:
                avg = round(statistics.mean(str_results), 1)
                best_src = (
                    max(source_counts.items(), key=lambda kv: kv[1])[0]
                    if source_counts else "sold"
                )
                src_label = f"{best_src} (sampled, {len(str_results)} lots)"
                result[key] = (avg, src_label)
            else:
                result[key] = (None, None)

            if progress_callback:
                progress_callback(i + 1, total)

        return auction_strs

    def batch_lookup(self, df: pd.DataFrame, progress_callback=None,
                     auction_str_map: Optional[dict] = None,
                     max_workers: int = 8) -> pd.DataFrame:
        """Add price range, STR, and source columns to a DataFrame.

        Runs `lookup_price_range()` in parallel across a thread pool. STR is
        either scraped per-lot (slow) or looked up from a precomputed
        per-auction map (fast — recommended for 500+ lots).

        Threading notes:
          - requests.Session and httpx.get are thread-safe.
          - Token fetch is guarded by a lock so 8 workers starting at once
            don't mint 8 tokens.
          - Results are collected via `as_completed` so Streamlit progress
            callbacks fire only from the main thread.
          - The per-call `time.sleep(0.3)` inside lookup_price_range is kept
            as a per-worker throttle; with 8 workers that's ~8 req/sec,
            which eBay/Mercari scraping tolerates well.

        Args:
            df: DataFrame with a 'title' column (or 'enriched_title')
            progress_callback: Optional callable(current, total) OR
                callable(current, total, title_preview). Extra arg is optional.
            auction_str_map: Optional {auction_name: (str_value, source)} dict,
                typically from sample_auction_str(). When provided, per-lot
                STR scraping is SKIPPED and the auction-level value is applied
                to every row.
            max_workers: Thread pool size. Set to 1 for serial (useful for
                debugging or if you're getting rate-limited).

        Returns:
            DataFrame with these columns added:
                est_resale, price_low, price_high, comp_count, ebay_comps,
                mercari_comps, price_source, ebay_str, str_source
        """
        df = df.copy().reset_index(drop=True)  # 0..n-1 positional keys
        total = len(df)
        use_auction_str = auction_str_map is not None

        # Pre-extract titles + auction names so worker threads don't touch
        # pandas (which isn't thread-safe for concurrent .at assignments).
        titles = [
            (row.get('enriched_title') or row.get('title', '') or '')
            for _, row in df.iterrows()
        ]
        auctions = (
            df['auction'].tolist() if 'auction' in df.columns else [None] * total
        )
        categories = (
            df['category'].fillna('').astype(str).tolist()
            if 'category' in df.columns else [''] * total
        )

        # Result slots, indexed positionally
        price_results: list = [None] * total
        str_results: list = [(None, None)] * total

        def _work_price(i: int):
            try:
                return i, self.lookup_price_range(titles[i])
            except Exception:
                return i, None

        def _work_str(i: int):
            try:
                pct, src = self.lookup_str(titles[i])
                return i, (pct, src)
            except Exception:
                return i, (None, None)

        # Fill STR from the sampled map (no HTTP — cheap pass).
        # The map can be keyed per-auction (back-compat) or per-(auction,category)
        # (new). __granularity__ tells us which.
        if use_auction_str:
            granularity = auction_str_map.get("__granularity__", "auction")
            for i in range(total):
                entry = None
                if granularity == "category":
                    cat = categories[i].strip() or '(uncategorized)'
                    entry = auction_str_map.get((auctions[i], cat))
                    # Fallback: try just auction-level (if this row's category
                    # had no usable sample, borrow auction-avg from another
                    # bucket in the same auction — better than None)
                    if not entry or entry[0] is None:
                        # Average across all buckets for this auction
                        auction_vals = [
                            v[0] for k, v in auction_str_map.items()
                            if isinstance(k, tuple) and k[0] == auctions[i]
                            and v and v[0] is not None
                        ]
                        if auction_vals:
                            avg = round(sum(auction_vals) / len(auction_vals), 1)
                            entry = (avg, "sampled (auction avg)")
                else:
                    entry = auction_str_map.get(auctions[i])
                str_results[i] = entry if entry else (None, None)

        completed = 0

        def _emit(current: int, title: str = ""):
            if not progress_callback:
                return
            # Try (current, total, title) then fall back to (current, total)
            try:
                progress_callback(current, total, title)
            except TypeError:
                try:
                    progress_callback(current, total)
                except Exception:
                    pass

        workers = max(1, int(max_workers))

        # Serial path — avoid thread overhead when max_workers == 1
        if workers == 1:
            for i in range(total):
                _, price_info = _work_price(i)
                price_results[i] = price_info
                if not use_auction_str:
                    _, sr = _work_str(i)
                    str_results[i] = sr
                completed += 1
                _emit(completed, titles[i][:70])
        else:
            # Parallel path — submit all price jobs; interleave STR jobs only
            # when we don't have a precomputed auction map.
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = {ex.submit(_work_price, i): ('price', i) for i in range(total)}
                if not use_auction_str:
                    for i in range(total):
                        futures[ex.submit(_work_str, i)] = ('str', i)

                for fut in as_completed(futures):
                    kind, i = futures[fut]
                    try:
                        idx, payload = fut.result()
                    except Exception:
                        continue
                    if kind == 'price':
                        price_results[idx] = payload
                        completed += 1
                        _emit(completed, titles[idx][:70])
                    else:
                        str_results[idx] = payload

        # Unpack price_results into column lists
        medians, lows, highs, counts = [], [], [], []
        ebay_counts, mercari_counts, pc_counts, price_sources = [], [], [], []
        for info in price_results:
            if info:
                medians.append(info["median"])
                lows.append(info["low"])
                highs.append(info["high"])
                counts.append(info["count"])
                ebay_counts.append(info.get("ebay_count", 0))
                mercari_counts.append(info.get("mercari_count", 0))
                pc_counts.append(info.get("pricecharting_count", 0))
                price_sources.append(info["source"])
            else:
                medians.append(None)
                lows.append(None)
                highs.append(None)
                counts.append(0)
                ebay_counts.append(0)
                mercari_counts.append(0)
                pc_counts.append(0)
                price_sources.append(None)

        str_values = [s[0] for s in str_results]
        str_sources = [s[1] for s in str_results]

        df['est_resale'] = medians
        df['price_low'] = lows
        df['price_high'] = highs
        df['comp_count'] = counts
        df['ebay_comps'] = ebay_counts
        df['mercari_comps'] = mercari_counts
        df['pricecharting_comps'] = pc_counts
        df['price_source'] = price_sources
        df['ebay_str'] = str_values
        df['str_source'] = str_sources
        return df
