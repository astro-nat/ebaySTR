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


def _nan_safe_str(v) -> str:
    """str(v) with NaN/None/pd.NA collapsed to ''.

    DataFrame-sourced values are NaN-capable, and NaN is TRUTHY — so
    `row.get('enriched_title') or row.get('title')` returns NaN
    instead of falling through, and str(NaN) == 'nan' silently
    poisons queries/URLs. Confirmed crash path 7/11: cache-merged
    frames carry NaN enriched_title for uncached lots → NaN reached
    title.lower() in the PC classifier → AttributeError aborted the
    whole comps batch.
    """
    if v is None:
        return ''
    try:
        if pd.isna(v):
            return ''
    except (TypeError, ValueError):
        pass
    return str(v)


class _CompPriceList(list):
    """A list of comp prices that also reports relevance-filter stats.

    Behaves exactly like list[float] for existing callers; the lookup
    flow reads `.relevance_dropped` to annotate price_source when
    off-topic comps were discarded. Per-call instance → thread-safe
    under the comp ThreadPoolExecutor (unlike an attribute on the
    shared EbayPriceLookup instance).
    """
    relevance_dropped = 0


# Title-specificity markers — when present in a lot title, single-comp
# catalog matches (PriceCharting / GoCollect, count=1) are usually
# trustworthy because the title carries enough info to disambiguate the
# match. When ABSENT, single-comp catalog matches can drift catastrophic-
# ally — see the Pokemon dropship audit (5 lots matched the 1999 Base
# Set Booster Box catalog at $10,736 each because their generic
# "1 Box Pokémon Cards English – Surprise Gift Box" titles got mapped
# to the actual Base Set product). Specificity gating preserves
# legitimate matches like "X-Men #281 Jim Lee Auto" while caging the
# generic dropship pattern.
_TITLE_SPECIFICITY_MARKERS = (
    re.compile(r"#\d+", re.IGNORECASE),                   # issue/lot number
    re.compile(r"\b(?:CGC|PSA|BGS|SGC|CBCS|ANACS)\b", re.IGNORECASE),
    re.compile(r"\b1st\s+edition\b", re.IGNORECASE),
    re.compile(r"\b(?:base|jungle|fossil|gym|neo|EX|XY|"
               r"sun\s+&\s+moon|sword\s+&\s+shield|"
               r"scarlet\s+&\s+violet)\s+set\b", re.IGNORECASE),
    re.compile(r"\b(?:19|20)\d{2}\b"),                    # 4-digit year
    re.compile(r"\b(?:autograph(?:ed)?|signed|auto)\b", re.IGNORECASE),
    re.compile(r"\bsealed\b", re.IGNORECASE),
)


def _has_title_specificity(title: str) -> bool:
    """True when the title carries an identifier that disambiguates a
    single-comp catalog match (issue#, grading code, set name, year,
    autograph/sealed callout)."""
    if not title:
        return False
    return any(p.search(title) for p in _TITLE_SPECIFICITY_MARKERS)


# Title markers signaling new-in-box / sealed / never-used condition.
# These lots sit at the HIGH end of comp distributions — applying the
# Q1-anchored variance cap drags resale down to used-comp territory,
# under-pricing NOS items by 50-80%. Hill Country 5/21: Henckels Zwilling
# 8pc Steak Knife Set NOS scored profit -$5 with $45 capped resale; real
# NOS Zwilling steak sets clear $80-180.
_NOS_PATTERNS = (
    re.compile(r"\bN\.?O\.?S\.?\b", re.IGNORECASE),
    re.compile(r"\bnew\s+old\s+stock\b", re.IGNORECASE),
    re.compile(r"\bMIB\b", re.IGNORECASE),
    re.compile(r"\bNIB\b", re.IGNORECASE),
    re.compile(r"\bM\.?I\.?S\.?B\.?\b", re.IGNORECASE),  # MISB: mint in sealed box
    re.compile(r"\bsealed\b", re.IGNORECASE),
    re.compile(r"\bunopened\b", re.IGNORECASE),
    re.compile(r"\bnew\s+in\s+(?:box|package|pkg)\b", re.IGNORECASE),
    re.compile(r"\bfactory[\s-]?sealed\b", re.IGNORECASE),
    re.compile(r"\bbrand[\s-]?new\b", re.IGNORECASE),
    re.compile(r"\bdeadstock\b", re.IGNORECASE),
)


def _has_nos_marker(title: str) -> bool:
    """True when the title signals new-in-box / sealed / NOS condition.

    Used to short-circuit the wide-spread variance cap — NOS lots
    belong in the high-end of the comp distribution, so anchoring to
    Q1 mis-prices them by a factor of 2-3×.
    """
    if not title:
        return False
    return any(p.search(title) for p in _NOS_PATTERNS)


class EbayPriceLookup:
    # Class-level scrape stats. Updated by _scrape_ebay_sold_prices on
    # every call. Read by the UI after a comp run finishes so the user
    # can see when the scraper is being blocked (eBay returns HTTP 403
    # for almost every request from a non-residential IP). Without
    # this, the silent fall-through to the eBay-Browse-API "active
    # listings" path looks like normal data when in fact the entire
    # sold-history pipeline is dead.
    # NOTE: Mercari + ScrapingBee integrations removed; counters trimmed.
    _scrape_stats = {
        'ebay_sold_attempts': 0,
        'ebay_sold_blocked': 0,
        'ebay_sold_success': 0,
        # Off-topic comps discarded by the relevance filter (comp
        # title doesn't contain the query's terms). High counts mean
        # eBay's keyword match is drifting away from the lot titles.
        'relevance_dropped': 0,
        # SoldComps API (paid eBay sold-data provider) — the primary
        # sold-comp source now that eBay captcha-blocks the direct scrape.
        'soldcomps_calls': 0,
        'soldcomps_hits': 0,        # calls that returned >= 1 sold item
        'soldcomps_auth_fail': 0,   # 401/403 — bad key
        'soldcomps_quota_fail': 0,  # 429 — monthly request cap hit
    }

    @classmethod
    def reset_scrape_stats(cls):
        """Zero every counter — call before each new comp run so the UI
        warning reflects the current run's failure rate, not lifetime."""
        for k in cls._scrape_stats:
            cls._scrape_stats[k] = 0

    @classmethod
    def get_scrape_stats(cls) -> dict:
        """Snapshot of the scrape counters since the last reset."""
        return dict(cls._scrape_stats)

    def __init__(self, app_id: str, cert_id: str, pricecharting=None,
                 soldcomps_key: Optional[str] = None,
                 gocollect=None,   # deprecated no-op — account never approved
                 mercari_enabled: bool = False,
                 **_ignored):      # swallows a legacy scrapingbee_key= call
        """eBay-only price lookup, optionally augmented with PriceCharting.

        Args:
            app_id, cert_id: eBay developer credentials.
            pricecharting: Optional PriceChartingLookup instance. When set,
                lots whose titles classify as games / cards / comics get a
                PriceCharting lookup BEFORE the eBay scrape. PC's
                aggregated sold data is materially better for these niches.
                Pass None (or omit) to disable.
            soldcomps_key: SoldComps API key — the primary sold-comp source.
            mercari_enabled: Deprecated/no-op. Mercari integration has
                been removed; this parameter is retained only for
                signature compatibility with existing call sites.

        NOTE: ScrapingBee was removed 7/2026 (subscription cancelled). A
        legacy `scrapingbee_key=` kwarg is accepted and ignored via
        **_ignored so existing call sites don't break.
        """
        self.app_id = app_id
        self.cert_id = cert_id
        self.pricecharting = pricecharting
        # SoldComps API key — primary sold-comp source (real eBay sold
        # data via https://sold-comps.com). eBay now captcha-blocks the
        # direct sold-listings scrape, so this is how we get true sold
        # prices; the old scrape is kept only as a fallback. Per-instance
        # cache so repeated identical queries in one run cost one request.
        self.soldcomps_key = soldcomps_key or None
        self._soldcomps_cache: dict = {}
        # Mercari integration removed; flag retained as no-op for
        # backward compatibility with existing call sites.
        self.mercari_enabled = False
        # GoCollect integration removed 7/6 — the API-access
        # application was rejected, so the lookup tier never had a
        # working key. The constructor arg is retained as a no-op for
        # signature compat; graded comics now route through
        # PriceCharting + the eBay-sold scrape like everything else.
        self.gocollect = None
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
        # Strip dashes too — eBay's search engine interprets " - " (space-
        # dash-space) as the NOT operator. Titles like "Phantom Lady 17 -
        # CGC 4.0" become "Phantom Lady 17 NOT CGC NOT 4.0", excluding
        # exactly the graded copies the user wants. Replace with spaces.
        clean = re.sub(r'[,;:/\\|\-]+', ' ', clean)
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

    def _proxied_get(self, target_url: str, params: dict, timeout: float = 30):
        """Direct HTTP GET (ScrapingBee removed 7/2026).

        Returns the response object (or None on transport failure). eBay
        tends to 403 direct requests, but sold comps now come from the
        SoldComps API and STR falls back to the free Browse-API demand
        score — so this direct path is only a last-ditch attempt, not the
        primary comp source it once was.
        """
        try:
            session = self._get_scrape_session()
            return session.get(target_url, params=params, timeout=timeout)
        except Exception:
            return None

    # --- Comp-relevance filtering ------------------------------------
    # eBay's keyword search is permissive: "Notebook Paper & Cards"
    # (a real Longview lot) returned 21 trading-card listings in a
    # tight $136-147 band — consistently priced, consistently the
    # WRONG product, and invisible to the variance detector. The fix
    # is to keep the comp TITLES during scraping and require each comp
    # to actually contain the query's terms.
    _RELEVANCE_STOPWORDS = frozenset({
        'the', 'and', 'for', 'with', 'of', 'to', 'in', 'on', 'a', 'an',
        'by', 'or', 'new', 'used', 'set', 'size',
    })

    @classmethod
    def _relevance_tokens(cls, text: str) -> list:
        """Lowercased alnum tokens (len >= 3) minus stopwords."""
        return [
            t for t in re.findall(r'[a-z0-9]{3,}', (text or '').lower())
            if t not in cls._RELEVANCE_STOPWORDS
        ]

    # Bulk-listing indicators. When the COMP title screams "big
    # collection" but the QUERY doesn't, the comp is a quantity
    # mismatch: a 3-car diecast lot comping against 50-car crate
    # listings (7/12: "MAISTO EXPLORER, MAJORETTE COBRA, TC NASCAR"
    # — three ~$10 cars — drew 3 collection comps at $125-150 and
    # A-graded at $150 resale).
    _BULK_TITLE_RE = re.compile(
        r"\b(?:lot\s+of\s+\d+|\d{2,}\s*(?:pcs?|pieces?|cars?|count)\b|"
        r"collection|bundle|huge\s+lot|large\s+lot|case\s+of|"
        r"wholesale\s+lot|dealer\s+lot|estate\s+lot)\b",
        re.IGNORECASE,
    )

    @classmethod
    def _quantity_mismatch(cls, query: str, comp_title: str) -> bool:
        """True when the comp is a bulk listing but the query isn't
        (or vice versa) — those prices describe a different amount of
        stuff and poison the median in either direction."""
        if not comp_title:
            return False
        q_bulk = bool(cls._BULK_TITLE_RE.search(query or ''))
        c_bulk = bool(cls._BULK_TITLE_RE.search(comp_title))
        return q_bulk != c_bulk

    @classmethod
    def _comp_is_relevant(cls, query_tokens: list, comp_title: str) -> bool:
        """True when the comp title contains >= 50% of the query's tokens.

        Prefix matching per token ('lantern' hits 'lanterns'). Comps
        with no title (price-only legacy markup) are treated as
        relevant — no information is not negative information.
        """
        if not query_tokens:
            return True
        if not comp_title:
            return True
        comp_tokens = cls._relevance_tokens(comp_title)
        if not comp_tokens:
            return True
        hits = 0
        for q in query_tokens:
            if any(c.startswith(q) or q.startswith(c) for c in comp_tokens):
                hits += 1
        need = max(1, -(-len(query_tokens) // 2))  # ceil(n/2)
        return hits >= need

    def _scrape_ebay_sold_listings(self, query: str,
                                   max_prices: int = 30) -> list:
        """Scrape (price, title) pairs from eBay's sold-listings page.

        Handles three markup generations observed in production:
          1. su-item-card (mid-2026)  — per-card title + price
          2. s-card__price (early 2026) — price spans only (no title)
          3. s-item__price (legacy)     — price spans only (no title)
        Titles are None for generations 2-3; the relevance filter
        passes those through unchecked.
        """
        type(self)._scrape_stats['ebay_sold_attempts'] += 1
        params = {
            "_nkw": query,
            "LH_Sold": "1",
            "LH_Complete": "1",
            "_ipg": "60",  # 60 results per page
        }
        try:
            resp = self._proxied_get(
                "https://www.ebay.com/sch/i.html",
                params=params,
                timeout=30,
            )
            if resp is None:
                return []
            _txt = resp.text or ''
            _low = _txt.lower()
            # Does the page actually contain listing markup? (any markup
            # generation). If not, it's not a results page.
            _has_listings = (
                'su-item-card' in _txt
                or 's-item__price' in _txt
                or 's-card__price' in _txt
            )
            # 403 / very-short body = classic anti-bot block. NEW (7/2026):
            # eBay now serves a FULL-SIZE captcha / "verify yourself" /
            # sign-in interstitial (HTTP 200, 40KB+) on the sold-listings
            # endpoint — it sails past the tiny-body check and looks like
            # "0 results", silently poisoning every comp with the active-
            # listing fallback. Detect the challenge page explicitly:
            # listing markup ABSENT and a challenge marker PRESENT.
            _is_challenge = not _has_listings and (
                'pardon our interruption' in _low
                or 'enter the characters you see' in _low
                or 'please verify yourself' in _low
                or 'checking your browser' in _low
                or ('captcha' in _low and 'signin' in _low)
            )
            if (resp.status_code in (403, 429)
                    or len(_txt) < 500
                    or _is_challenge):
                type(self)._scrape_stats['ebay_sold_blocked'] += 1
                return []
            if resp.status_code != 200 or len(_txt) < 1000:
                return []

            html = resp.text
            pairs = []  # (price: float, title: str|None)

            def _parse_price(text):
                m = re.search(r'\$([\d,]+(?:\.\d{2})?)', text)
                if not m:
                    return None
                try:
                    p = float(m.group(1).replace(",", ""))
                except ValueError:
                    return None
                return p if 0.99 < p < 50000 else None

            # GEN 1 (mid-2026): one <div class="su-item-card s-item-card">
            # per listing; title in a nested span under the
            # su-item-card__title link, price in su-item-card__price.
            # eBay pads the top of results with fake "Shop on eBay"
            # placeholder cards ($20.00) — skipped by title match.
            cards = re.split(r'<div class="su-item-card s-item-card"', html)
            if len(cards) > 1:
                title_re = re.compile(
                    r'su-item-card__title.*?<span[^>]*>([^<]{5,250})</span>',
                    re.DOTALL,
                )
                price_re = re.compile(
                    r'su-item-card__price[^>]*>([^<]+)<'
                )
                for card in cards[1:]:
                    chunk = card[:6000]
                    tm = title_re.search(chunk)
                    title = tm.group(1).strip() if tm else None
                    if title and title.lower() == 'shop on ebay':
                        continue
                    pm = price_re.search(chunk)
                    if not pm or ' to ' in pm.group(1).lower():
                        continue
                    p = _parse_price(pm.group(1))
                    if p is not None:
                        pairs.append((p, title))
                    if len(pairs) >= max_prices:
                        break

            # GEN 2 (early 2026): s-card__price spans. Price-only —
            # titles aren't reliably adjacent in this markup.
            # 'strikethrough' variants are crossed-out original prices.
            if not pairs:
                new_blocks = re.findall(
                    r'class="([^"]*s-card__price[^"]*)"[^>]*>([^<]+)</span>',
                    html,
                )
                for class_str, content in new_blocks:
                    if 'strikethrough' in class_str:
                        continue
                    if ' to ' in content.lower():
                        continue
                    p = _parse_price(content)
                    if p is not None:
                        pairs.append((p, None))
                    if len(pairs) >= max_prices:
                        break

            # GEN 3 (legacy): s-item__price spans, price-only.
            if not pairs:
                blocks = re.findall(
                    r'class="s-item__price"[^>]*>(.*?)</span>\s*</div>',
                    html,
                    flags=re.DOTALL,
                )
                if not blocks:
                    blocks = re.findall(
                        r'class="s-item__price"[^>]*>'
                        r'([^<]+(?:<[^>]+>[^<]+</[^>]+>)*)', html,
                    )
                for block in blocks:
                    if ' to ' in block.lower():
                        continue
                    p = _parse_price(block)
                    if p is not None:
                        pairs.append((p, None))
                    if len(pairs) >= max_prices:
                        break

            if pairs:
                type(self)._scrape_stats['ebay_sold_success'] += 1
            return pairs
        except Exception:
            return []

    def _fetch_soldcomps_pairs(self, query: str, count: int = 120):
        """Fetch (price, title) sold pairs from the SoldComps API.

        The paid drop-in replacement for the eBay sold-listings scrape
        (which eBay now captcha-blocks). Returns a list of (float, str)
        pairs, or None on any failure (no key, HTTP error, quota, parse).
        Cached per query on the instance so repeated identical lookups in
        one run cost a single API request.
        """
        if not self.soldcomps_key or not query:
            return None
        if query in self._soldcomps_cache:
            return self._soldcomps_cache[query]
        pairs = None
        try:
            # Explicit truststore SSL context — raw httpx.get trusts only
            # certifi, which dies under Norton/corp TLS inspection. Build
            # the client once per instance (same pattern as pass2/vision).
            if getattr(self, '_sc_client', None) is None:
                from scraper._ssl_compat import make_ssl_context
                self._sc_client = httpx.Client(
                    verify=make_ssl_context(), timeout=40,
                )
            type(self)._scrape_stats['soldcomps_calls'] += 1
            resp = self._sc_client.get(
                "https://api.sold-comps.com/v1/scrape",
                headers={"Authorization": f"Bearer {self.soldcomps_key}"},
                params={
                    "keyword": query,
                    "count": min(max(int(count), 1), 240),
                    "daysToScrape": 90,
                },
            )
            if resp.status_code == 200:
                items = (resp.json() or {}).get("items") or []
                out = []
                for it in items:
                    raw = it.get("soldPrice")
                    try:
                        p = float(str(raw).replace(",", "").replace("$", "").strip())
                    except (TypeError, ValueError):
                        continue
                    if 0.99 < p < 50000:
                        out.append((p, it.get("title") or ""))
                pairs = out
                if out:
                    type(self)._scrape_stats['soldcomps_hits'] += 1
            elif resp.status_code in (401, 403):
                type(self)._scrape_stats['soldcomps_auth_fail'] += 1
            elif resp.status_code == 429:
                type(self)._scrape_stats['soldcomps_quota_fail'] += 1
        except Exception:
            pairs = None
        self._soldcomps_cache[query] = pairs
        return pairs

    def _scrape_ebay_sold_prices(self, query: str, max_prices: int = 30) -> list:
        """Get relevance-filtered sold prices for `query`.

        Source priority: SoldComps API (real sold data — primary now that
        eBay captcha-blocks the direct scrape) → the legacy eBay scrape as
        fallback. Returns a _CompPriceList (a list subclass) of floats
        whose `.relevance_dropped` reports how many comps were discarded
        for not containing the query's terms, and whose `.sold_via`
        records which source produced the data ('SoldComps' or 'eBay').
        Callers that treat it as a plain list are unaffected.
        """
        sold_via = 'eBay'
        pairs = None
        if self.soldcomps_key:
            # SoldComps is the authoritative sold-data source. If it returns
            # nothing (quota hit or genuine no-match), do NOT fall back to
            # the legacy eBay sold scrape — eBay captcha-blocks it, so it
            # never returns data and only burns latency + ScrapingBee
            # credits on a dead page. Let lookup_price_range drop straight
            # to the free active-listing Browse API instead.
            pairs = self._fetch_soldcomps_pairs(query, count=max(120, max_prices))
            if pairs:
                sold_via = 'SoldComps'
        else:
            # No SoldComps key configured — the (now usually blocked) scrape
            # is the only sold path available; keep trying it.
            pairs = self._scrape_ebay_sold_listings(query, max_prices=max_prices)
        if pairs is None:
            pairs = []
        q_tokens = self._relevance_tokens(query)
        kept = _CompPriceList()
        dropped = 0
        for price, title in pairs:
            if (self._comp_is_relevant(q_tokens, title)
                    and not self._quantity_mismatch(query, title)):
                kept.append(price)
            else:
                dropped += 1
        kept.relevance_dropped = dropped
        kept.sold_via = sold_via
        if dropped:
            type(self)._scrape_stats['relevance_dropped'] = (
                type(self)._scrape_stats.get('relevance_dropped', 0) + dropped
            )
        return kept

    # NOTE: Mercari sold-listings integration was removed. The previous
    # _scrape_mercari_sold_prices and _extract_mercari_prices helpers
    # have been deleted; the comp pipeline is now eBay-only (with
    # PriceCharting / GoCollect tiers preserved).

    def fetch_amazon_price(self, url: str) -> Optional[float]:
        """Disabled 7/2026 — always returns None.

        This used to fetch Amazon's live buy-box price for liquidation
        lots carrying a 'Retailer Product URL: amazon.com/dp/ASIN'. The
        fetch only ever succeeded through ScrapingBee's residential
        proxy (direct requests get 403'd from datacenter IPs). With the
        ScrapingBee subscription cancelled there's no working transport,
        so this is a no-op; the pipeline falls back to the auctioneer's
        stated retail (soft-capped elsewhere). Retained so existing call
        sites keep working without a signature change.
        """
        return None

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

    def lookup_price_range(self, title: str, limit: int = 8,
                           pc_only: bool = False) -> Optional[dict]:
        """Look up resale price statistics from eBay sold data (eBay only).

        Uses progressive query shortening: we try the full cleaned title first
        (6 words), then fall back to 4 words and finally 3 words if no matches
        turn up. eBay's search is extremely intolerant of extra terms — a
        7-word query commonly returns zero, while the same first 3 words
        return hundreds. We stop as soon as we clear the ≥3 sold-comp bar.

        ``pc_only=True`` restricts the lookup to PriceCharting
        (curated catalog only) and skips the eBay sold-listings scrape
        entirely. Used for stylized/replica lots where eBay scraped
        comps would return authentic-brand contamination, but PC's
        catalog-keyed matching is safe — PC simply returns None for
        products it doesn't cover (handbags, jewelry, decorative
        sculpture, etc.).

        Returns:
            {
                "median": float,
                "low": float,           # 25th percentile across all comps
                "high": float,          # 75th percentile across all comps
                "count": int,           # Total comp count
                "source": str,          # Combined source label
                "ebay_count": int,      # eBay sold comps contributed
                "mercari_count": int,   # Always 0 (Mercari removed; key
                                        # retained for app.py compat)
                "query": str,           # The query variant that produced hits
            }
            or None if no data available.
        """
        # Tier 1: PriceCharting for games / TCG / comics. Aggregated
        # sold data, condition-normalized, canonical product ID.
        # Strong on Pokemon/MTG/video games. (A GoCollect tier for
        # graded comics used to sit above this — removed 7/6 when the
        # API application was rejected; graded comics now comp via PC
        # + the eBay-sold scrape.)
        if self.pricecharting is not None and self.pricecharting.enabled:
            try:
                pc_result = self.pricecharting.lookup(title)
                if pc_result is not None:
                    return pc_result
            except Exception:
                pass  # Don't let PC outages break the scan

        # PC-only mode bails here: eBay sold-comps would contaminate
        # stylized/replica lookups with authentic-brand prices that
        # don't apply.
        if pc_only:
            return None

        variants = self._query_variants(title)
        if not variants:
            return None

        # Try each query variant in descending specificity. First one to
        # clear the ≥3-comp bar wins. Also remember the best "partial"
        # (1-2 comp) result in case nothing clears the bar — still better
        # than falling all the way through to active listings.
        best_partial = None  # (combined, ebay, query)

        for idx, query in enumerate(variants):
            ebay_prices = []
            rel_dropped = 0
            sold_via = 'eBay'
            try:
                # No inter-request sleep needed on the SoldComps path (it's
                # an API, not a scrape); keep the polite delay only when
                # falling back to the direct scrape.
                if not self.soldcomps_key:
                    time.sleep(0.3)
                ebay_prices = self._scrape_ebay_sold_prices(query)
                # Capture attrs BEFORE outlier filtering (which returns a
                # plain list, losing them).
                rel_dropped = getattr(ebay_prices, 'relevance_dropped', 0)
                sold_via = getattr(ebay_prices, 'sold_via', 'eBay')
                ebay_prices = self._filter_outliers(ebay_prices)
            except Exception:
                ebay_prices = []

            combined = list(ebay_prices)

            if len(combined) >= 3:
                combined = self._filter_outliers(combined)
                stats = self._price_stats(combined)
                if stats:
                    source = f"sold ({sold_via})"
                    # Annotate source with the fallback level if we had to
                    # drop down — helps the user eyeball whether the comps
                    # were for the specific product vs a generic keyword.
                    if idx > 0:
                        source = f"{source} [short query]"
                    # Surface how many off-topic comps the relevance
                    # filter discarded — a high count means eBay's
                    # keyword match drifted and the survivors deserve
                    # a skeptical eye.
                    if rel_dropped:
                        source = f"{source} [{rel_dropped} off-topic dropped]"
                    return {
                        **stats,
                        "count": len(combined),
                        "source": source,
                        "ebay_count": len(ebay_prices),
                        # Mercari integration removed; zero-out for app.py compat.
                        "mercari_count": 0,
                        "query": query,
                    }

            # Remember the first variant that produced ANY comps so we can
            # surface at least a rough number if none hit the ≥3 bar.
            if combined and best_partial is None:
                best_partial = (
                    list(combined), list(ebay_prices), query, rel_dropped,
                    sold_via,
                )

        # All sold-comp variants failed to clear ≥3. Use the best partial
        # if we have one (1-2 comps) before falling back to active listings.
        if best_partial is not None:
            combined, ebay_prices, matched_q, rel_dropped, sold_via = best_partial
            stats = self._price_stats(combined)
            if stats:
                source = f"sold (thin comps · {sold_via})"
                if rel_dropped:
                    source = f"{source} [{rel_dropped} off-topic dropped]"
                return {
                    **stats,
                    "count": len(combined),
                    "source": source,
                    "ebay_count": len(ebay_prices),
                    # Mercari integration removed; zero-out for app.py compat.
                    "mercari_count": 0,
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
                title = (_nan_safe_str(row.get(title_col))
                 or _nan_safe_str(row.get('title')))
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

        return result

    def batch_lookup(self, df: pd.DataFrame, progress_callback=None,
                     auction_str_map: Optional[dict] = None,
                     max_workers: int = 8,
                     live_callback=None) -> pd.DataFrame:
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
            which eBay scraping tolerates well.

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
            (_nan_safe_str(row.get('enriched_title'))
                 or _nan_safe_str(row.get('title')))
            for _, row in df.iterrows()
        ]
        auctions = (
            df['auction'].tolist() if 'auction' in df.columns else [None] * total
        )
        categories = (
            df['category'].fillna('').astype(str).tolist()
            if 'category' in df.columns else [''] * total
        )
        # Per-row "PC only" flag — set by _apply_comps_filters when a lot
        # is stylized/replica. The lookup runs GoCollect + PriceCharting
        # but skips eBay, because authentic-brand sold-comps don't apply
        # to a "Gucci style" lot. PC will return None when it doesn't
        # cover the product (handbags, jewelry, etc.) — that's the safe
        # outcome.
        pc_only_flags = (
            df['_pc_only_stylized'].fillna(False).astype(bool).tolist()
            if '_pc_only_stylized' in df.columns else [False] * total
        )

        # Result slots, indexed positionally
        price_results: list = [None] * total
        str_results: list = [(None, None)] * total

        # Per-item STR override mask. STR varies wildly within
        # card/game/comic categories — Charizard PSA 10 sells in days,
        # a generic Pokemon common takes months. The category-level
        # average smears that signal. For rows where the classifier
        # tags the lot as tcg / sports_card / comic / video_game, we
        # do per-item STR scrapes regardless of use_auction_str.
        # Furniture/tools/jewelry rows still use the cheaper
        # category sampling.
        from .pricecharting import classify_for_pricecharting
        needs_per_item_str = [
            classify_for_pricecharting(titles[i]) is not None
            for i in range(total)
        ]

        def _work_price(i: int):
            try:
                return i, self.lookup_price_range(
                    titles[i], pc_only=pc_only_flags[i]
                )
            except Exception:
                return i, None

        def _work_str(i: int):
            try:
                pct, src = self.lookup_str(titles[i])
                # Tag the source so the UI distinguishes per-item vs
                # category-sampled STR.
                if pct is not None and src and 'sampled' not in src:
                    src = f"{src} (per-item)"
                return i, (pct, src)
            except Exception:
                return i, (None, None)

        # Fill STR from the sampled map (no HTTP — cheap pass).
        # The map can be keyed per-auction (back-compat) or per-(auction,category)
        # (new). __granularity__ tells us which. Rows in the per-item
        # mask are LEFT at (None, None) here so the parallel-path STR
        # job below picks them up.
        if use_auction_str:
            granularity = auction_str_map.get("__granularity__", "auction")
            for i in range(total):
                if needs_per_item_str[i]:
                    continue  # leave at (None, None) for per-item scrape
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

        def _fire_live(idx: int, payload):
            """Call live_callback with (chunk_idx, payload, completed, total).

            Lets the caller stream a single lot's price result into the UI
            as soon as it completes, rather than waiting for the whole
            batch to finish. Errors in the callback are swallowed so a
            UI hiccup never poisons the price run.
            """
            if live_callback is None or payload is None:
                return
            try:
                live_callback(idx, payload, completed, total)
            except Exception:
                pass

        # Serial path — avoid thread overhead when max_workers == 1
        if workers == 1:
            for i in range(total):
                _, price_info = _work_price(i)
                price_results[i] = price_info
                # Per-item STR scrape when EITHER (a) we don't have an
                # auction map at all, OR (b) the row was tagged for
                # per-item even though we have a map.
                if (not use_auction_str) or needs_per_item_str[i]:
                    _, sr = _work_str(i)
                    str_results[i] = sr
                completed += 1
                _emit(completed, titles[i][:70])
                _fire_live(i, price_info)
        else:
            # Parallel path — submit all price jobs; interleave STR jobs
            # for rows that don't have a precomputed map value (either
            # use_auction_str is False entirely, or the row was tagged
            # as a card/game/comic that needs per-item STR).
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = {ex.submit(_work_price, i): ('price', i) for i in range(total)}
                for i in range(total):
                    if (not use_auction_str) or needs_per_item_str[i]:
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
                        _fire_live(idx, payload)
                    else:
                        str_results[idx] = payload

        # Unpack price_results into column lists.
        # NOTE: Mercari integration removed — mercari_counts is no
        # longer accumulated per-row; the df['mercari_comps'] column
        # is zero-filled below for app.py back-compat.
        medians, lows, highs, counts = [], [], [], []
        ebay_counts, pc_counts, gc_counts, price_sources = (
            [], [], [], []
        )
        variance_flags = []
        for idx, info in enumerate(price_results):
            if info:
                median = info["median"]
                low = info["low"]
                high = info["high"]
                count = info["count"]
                source = info["source"]
                title = titles[idx] if idx < len(titles) else ""
                # ---- Variance contamination guard ----
                # Wide low/high spread (>3× Q3/Q1 ratio) on a population
                # of ≥5 comps is the signature of variant contamination
                # in eBay search results: the query "Funko Pop Batwing
                # #500" pulls in the regular vinyl figure, Comic-Con
                # exclusives, chase variants, sealed multi-lots, signed
                # editions, etc. Median sits on the MIX, not on the
                # actual lot — for the regular figure the user probably
                # has, the comp lies high by 5-10×.
                #
                # Cap est_resale at 2.5 × the Q1 (low) so the resale
                # estimate stays anchored to the cheaper end of the
                # distribution where the regular product sits. Annotate
                # the source string with a ⚠ marker so the warning is
                # visible in the results table without a new column
                # being required for display.
                variance_flag = False
                if (low and high and count and count >= 5
                        and low > 0 and (high / low) > 3.0):
                    variance_flag = True
                    # Tier the cap by spread severity. A 3-5× spread is
                    # mild contamination — the median is still mostly
                    # right, just a little inflated. A 10×+ spread is
                    # near-total contamination (variant + outlier
                    # comps) and the median is essentially noise; the
                    # honest floor is closer to Q1 itself. Funko Pop
                    # Batwing #500 case: $24-$399 = 16.8× spread, real
                    # eBay value $7-14, our prior 2.5×-low cap of $59
                    # was still 5× too high.
                    spread = high / low
                    # NOS / sealed / MIB exception: these lots belong
                    # in the upper half of the comp distribution, not
                    # anchored to Q1. Use the unmodified median (no
                    # cap) since the wide-spread comes from mixing
                    # used + new comps and the user's lot IS new.
                    # Hill Country 5/21 Henckels Zwilling 8pc NOS:
                    # capped to $45 → profit -$5; uncapped median ~$120
                    # → profit ~$60.
                    if _has_nos_marker(title):
                        source = f"{source} ⚠ wide-spread (NOS floor)"
                    else:
                        if spread > 10.0:
                            cap_mult = 1.0
                        elif spread > 5.0:
                            cap_mult = 1.5
                        else:
                            cap_mult = 2.5
                        cap = round(cap_mult * low, 2)
                        if median is not None and median > cap:
                            median = cap
                            source = f"{source} ⚠ wide-spread (capped)"
                        else:
                            source = f"{source} ⚠ wide-spread"

                # ---- Single-comp catalog match ceiling ----
                # PriceCharting / GoCollect catalog matches return count=1
                # by design (one product → one price). When the lot title
                # is GENERIC enough that the catalog match probably mis-
                # fired, the median can be wildly wrong (Pokemon dropship
                # audit 5/3 — five "1 Box Pokémon Cards English – Surprise
                # Gift Box" lots matched the 1999 Base Set Booster Box
                # catalog at $10,736 each).
                #
                # Specificity gate: trust single-comp matches when the
                # title contains an issue#, grading code, set name, year,
                # autograph/sealed callout — these disambiguate the match.
                # When NO specificity marker is present, cap est_resale at
                # the catalog low (= same as median for single-comp) and
                # re-route to a "generic-title single-comp" warning. The
                # downstream manual_check flag will surface this.
                if (count == 1 and median is not None and median > 100
                        and not _has_title_specificity(title)):
                    if low is not None and low < median:
                        median = low
                    source = f"{source} ⚠ generic-title single-comp"

                medians.append(median)
                lows.append(low)
                highs.append(high)
                counts.append(count)
                ebay_counts.append(info.get("ebay_count", 0))
                pc_counts.append(info.get("pricecharting_count", 0))
                gc_counts.append(info.get("gocollect_count", 0))
                price_sources.append(source)
                variance_flags.append(variance_flag)
            else:
                medians.append(None)
                lows.append(None)
                highs.append(None)
                counts.append(0)
                ebay_counts.append(0)
                pc_counts.append(0)
                gc_counts.append(0)
                price_sources.append(None)
                variance_flags.append(False)

        str_values = [s[0] for s in str_results]
        str_sources = [s[1] for s in str_results]

        df['est_resale'] = medians
        df['price_low'] = lows
        df['price_high'] = highs
        df['comp_count'] = counts
        df['ebay_comps'] = ebay_counts
        # Mercari integration removed; column kept zero-filled because
        # app.py's _COMP_COLUMNS still references it downstream.
        df['mercari_comps'] = [0] * len(df)
        df['pricecharting_comps'] = pc_counts
        df['gocollect_comps'] = gc_counts
        df['price_source'] = price_sources
        df['comp_variance_flag'] = variance_flags
        df['ebay_str'] = str_values
        df['str_source'] = str_sources
        return df
