"""HiBid account integration — authenticated GraphQL queries.

Provides Path A from the Personal Comps plan: user pastes their authToken
once, the dashboard pulls live current bids from HiBid's GraphQL endpoint
on every refresh.

The authToken is a 768-character hex string returned in the
``GetAccountInfo`` GraphQL response. It's used as
``Authorization: Bearer <token>`` on subsequent /graphql calls. Token
appears to last for the duration of the user's HiBid session
(weeks-to-months); when it expires the API returns 401 "Buyer is
required" and the user re-pastes.

Storage: ``.cache/hibid_session.json``. JSON is fine here — the file is
gitignored, lives on the user's local machine, and the token is the
SAME credential the user's browser stores plaintext in localStorage.
We're not making the security situation worse.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from scraper._ssl_compat import make_ssl_context


_SESSION_PATH = Path(".cache") / "hibid_session.json"
_LOCK = threading.Lock()
_GRAPHQL_URL = "https://hibid.com/graphql"

# Shared truststore-backed client. Raw httpx.post() trusts only certifi,
# which dies under Norton / corp TLS inspection (SSL: CERTIFICATE_VERIFY_
# FAILED) — the same MITM issue that killed the Anthropic/HiBid calls until
# each module passed an explicit context. Build once, reuse.
_SSL_CLIENT: Optional[httpx.Client] = None
_CLIENT_LOCK = threading.Lock()


def _client() -> httpx.Client:
    global _SSL_CLIENT
    if _SSL_CLIENT is None:
        with _CLIENT_LOCK:
            if _SSL_CLIENT is None:
                _SSL_CLIENT = httpx.Client(verify=make_ssl_context(), timeout=30.0)
    return _SSL_CLIENT

# Standard headers every authenticated HiBid GraphQL request uses.
# Mirrors the Chrome HAR captures so we look like a real browser.
_BASE_HEADERS = {
    "accept": "application/json, text/plain, */*",
    "accept-language": "en-US,en;q=0.9",
    "content-type": "application/json",
    "origin": "https://hibid.com",
    "referer": "https://hibid.com/account/currentbids",
    "site_subdomain": "hibid.com",
    "user-agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/147.0.0.0 Safari/537.36"
    ),
}


# ---------------------------------------------------------------------
# Session persistence
# ---------------------------------------------------------------------
def _ensure_dir() -> None:
    _SESSION_PATH.parent.mkdir(parents=True, exist_ok=True)


def set_auth_token(token: str) -> None:
    """Persist the authToken to disk. Strips whitespace.

    Overwrites any existing token. Set ``token=""`` to clear.
    """
    cleaned = (token or "").strip()
    with _LOCK:
        _ensure_dir()
        payload = {"auth_token": cleaned, "saved_at": datetime.now().isoformat()}
        with _SESSION_PATH.open("w", encoding="utf-8") as f:
            json.dump(payload, f)


def get_auth_token() -> Optional[str]:
    """Return the saved authToken, or None if none stored."""
    if not _SESSION_PATH.exists():
        return None
    try:
        with _SESSION_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        token = (data or {}).get("auth_token") or None
        return token if token else None
    except (json.JSONDecodeError, OSError):
        return None


def session_metadata() -> Dict[str, Any]:
    """Diagnostic metadata about the saved session.

    Returns ``{has_token: bool, token_length: int, saved_at: str}``.
    Doesn't return the token itself — callers that need it use
    ``get_auth_token()``.
    """
    if not _SESSION_PATH.exists():
        return {"has_token": False, "token_length": 0, "saved_at": None}
    try:
        with _SESSION_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        token = (data or {}).get("auth_token") or ""
        return {
            "has_token": bool(token),
            "token_length": len(token),
            "saved_at": data.get("saved_at"),
        }
    except (json.JSONDecodeError, OSError):
        return {"has_token": False, "token_length": 0, "saved_at": None}


def clear_auth_token() -> None:
    """Remove the saved session. Idempotent."""
    with _LOCK:
        try:
            _SESSION_PATH.unlink()
        except FileNotFoundError:
            pass


# ---------------------------------------------------------------------
# Internal: GraphQL caller
# ---------------------------------------------------------------------
def _post_graphql(
    operation_name: str,
    query: str,
    variables: Dict[str, Any],
    token: Optional[str] = None,
    timeout: float = 30.0,
) -> Dict[str, Any]:
    """Send a single authenticated GraphQL request.

    Raises:
      ValueError if no token is available.
      RuntimeError on non-200 / GraphQL errors.

    Returns the ``data`` payload directly (the wrapping ``{data, errors}``
    is unwrapped here so callers don't have to).
    """
    token = token or get_auth_token()
    if not token:
        raise ValueError(
            "No HiBid authToken stored. Call set_auth_token() first or "
            "have the user paste a token via the dashboard."
        )

    headers = {
        **_BASE_HEADERS,
        "authorization": f"Bearer {token}",
    }
    payload = {
        "operationName": operation_name,
        "variables": variables,
        "query": query,
    }
    try:
        resp = _client().post(
            _GRAPHQL_URL, headers=headers, json=payload, timeout=timeout,
        )
    except httpx.HTTPError as e:
        raise RuntimeError(f"HiBid GraphQL request failed: {e}") from e

    if resp.status_code == 401:
        raise RuntimeError(
            "HiBid returned 401 — the authToken is invalid or expired. "
            "Re-paste a fresh token from your browser."
        )
    if resp.status_code != 200:
        raise RuntimeError(
            f"HiBid GraphQL returned {resp.status_code}: {resp.text[:300]}"
        )
    try:
        body = resp.json()
    except json.JSONDecodeError as e:
        raise RuntimeError(f"HiBid response was not JSON: {e}") from e

    if "errors" in body:
        msgs = "; ".join(
            err.get("message", "?") for err in body.get("errors", [])[:3]
        )
        raise RuntimeError(f"HiBid GraphQL errors: {msgs}")
    return body.get("data") or {}


# ---------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------
_GET_ACCOUNT_INFO_QUERY = """
query GetAccountInfo {
  account {
    email
    username
    ... on Buyer {
      id
      first
      last
      company
      city
      state
      country
      universalBidderEntityID
      authToken
      emailverified
      __typename
    }
    __typename
  }
}
"""


def fetch_account_info(token: Optional[str] = None) -> Dict[str, Any]:
    """Return account profile data for the signed-in user.

    Useful for two purposes:
      1. Prove the token is valid (200 + a populated account object).
      2. Display "Signed in as <username>" in the dashboard UI.

    Returns: ``{email, username, id, first, last, company, city, state,
    country, universalBidderEntityID, authToken, emailverified}``

    Note: the response also contains the user's ``authToken`` — we
    DON'T overwrite the saved one with this, because the token the
    user pasted is what we know to work. The response's authToken is
    the same value (or a refreshed one if HiBid rotates it).
    """
    data = _post_graphql("GetAccountInfo", _GET_ACCOUNT_INFO_QUERY, {}, token=token)
    return data.get("account") or {}


# ---------------------------------------------------------------------
# Current bids — the headline integration
# ---------------------------------------------------------------------
# Slimmed-down version of HiBid's CurrentBidsSearch query — same shape
# as the browser sends but trimmed to fields we actually need. The full
# version pulls 60+ fields per lot (auctioneer info, payment options,
# pictures, etc.) which we don't render.
_CURRENT_BIDS_QUERY = """
query CurrentBidsSearch(
  $pageNumber: Int!,
  $pageLength: Int!,
  $isArchived: Boolean = false,
  $hideClosedLots: Boolean = false,
  $auctionId: Int = null,
  $buyerLotStatusGroup: BuyerLotStatusGroup = null,
  $sortOrder: BuyerEventItemSortOrder = null,
  $monthRange: AltBidPastBidsRange = null,
  $sortDirection: SortDirection = DESC,
  $groupByAuction: Boolean = true,
  $auctionSortDirection: SortDirection = ASC
) {
  currentBids(
    input: {
      isArchived: $isArchived,
      groupByAuction: $groupByAuction,
      auctionSortDirection: $auctionSortDirection,
      hideClosedLots: $hideClosedLots,
      auctionId: $auctionId,
      buyerLotStatusGroup: $buyerLotStatusGroup,
      sortOrder: $sortOrder,
      monthRange: $monthRange
    }
    pageNumber: $pageNumber
    pageLength: $pageLength
    sortDirection: $sortDirection
  ) {
    auctions {
      id
      eventName
      lotCount
      bidCloseDateTime
      eventDateEnd
      eventCity
      eventState
      buyerPremium
      buyerPremiumRate
      shippingAndPickupInfo
    }
    pagedResults {
      totalCount
      filteredCount
      pageNumber
      pageLength
      results {
        id
        lotNumber
        lead
        bidAmount
        auction {
          id
          buyerPremium
          buyerPremiumRate
        }
        lotState {
          highBid
          buyerHighBid
          buyerBidStatus
          minBid
          status
          timeLeft
          isClosed
          isArchived
          mayHaveWonStatus
        }
      }
    }
  }
}
"""


_DEFAULT_CURRENT_BIDS_VARS: Dict[str, Any] = {
    "isArchived": False,
    "groupByAuction": True,
    "auctionSortDirection": "ASC",
    "hideClosedLots": False,
    "auctionId": 0,
    "buyerLotStatusGroup": "ALL",
    "sortOrder": "SALES_ORDER",
    "monthRange": "THREE_MONTHS",
    "sortDirection": "ASC",
    "pageNumber": 1,
    "pageLength": 100,
}


def fetch_current_bids(
    page_length: int = 100,
    page_number: int = 1,
    only_open: bool = False,
    only_winning: bool = False,
    month_range: str = "THREE_MONTHS",
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """Fetch the user's current bids from HiBid.

    Returns:
      {
        'auctions': [
          {'id', 'eventName', 'lotCount', 'bidCloseDateTime',
           'eventDateEnd', 'eventCity', 'eventState'},
          ...
        ],
        'lots': [
          {'lot_id', 'lot_number', 'title', 'auction_id',
           'high_bid', 'my_bid', 'min_bid', 'bid_status',
           'time_left', 'is_closed', 'is_archived',
           'may_have_won', 'status'},
          ...
        ],
        'total_count': int,
        'filtered_count': int,
      }

    Parameters
    ----------
    page_length
        Up to 100 lots per call (HiBid caps at 100; pagination needed
        for users with more open bids — uncommon for individual buyers).
    only_open
        Filter out closed lots client-side. Set when you only want
        "still bidding" rows.
    only_winning
        Filter to lots where buyer_high_bid >= high_bid (i.e., user is
        currently leading). Subset of ``only_open``.
    month_range
        HiBid's ``AltBidPastBidsRange`` enum. ``THREE_MONTHS`` is the
        UI default; other useful values: ``ONE_WEEK``, ``ONE_MONTH``,
        ``ALL_TIME``.
    token
        Override the saved token (rare; mostly useful in tests).
    """
    variables = dict(_DEFAULT_CURRENT_BIDS_VARS)
    variables.update({
        "pageNumber": int(page_number),
        "pageLength": int(min(max(page_length, 1), 100)),
        "monthRange": month_range,
    })
    data = _post_graphql(
        "CurrentBidsSearch", _CURRENT_BIDS_QUERY, variables, token=token,
    )
    cb = data.get("currentBids") or {}
    auctions = cb.get("auctions") or []
    paged = cb.get("pagedResults") or {}
    raw_results = paged.get("results") or []

    # The lot.auction subfield gives us the auction_id and per-auction
    # buyer's premium directly — no fragile positional joining needed.
    # The auctions array is still useful for rendering "which auctions
    # do I have bids in", but per-lot economics come from lot.auction.
    lots: List[Dict[str, Any]] = []
    for r in raw_results:
        ls = r.get("lotState") or {}
        if only_open and ls.get("isClosed"):
            continue
        high = ls.get("highBid") or 0.0
        my = ls.get("buyerHighBid") or 0.0
        if only_winning and not (my and my >= high):
            continue
        a = r.get("auction") or {}
        bp_pct = _normalize_buyer_premium(
            a.get("buyerPremium"), a.get("buyerPremiumRate"),
        )
        lots.append({
            "lot_id": str(r.get("id") or ""),
            "lot_number": r.get("lotNumber"),
            "title": r.get("lead") or "",
            "auction_id": a.get("id"),
            # Premium expressed as a decimal fraction (0.15 = 15%) for
            # arithmetic. None means we couldn't determine it from the
            # response — caller falls back to a default.
            "auction_buyer_premium_pct": bp_pct,
            "high_bid": high,
            "my_bid": my,
            "min_bid": ls.get("minBid"),
            "bid_status": ls.get("buyerBidStatus") or "",
            "time_left": ls.get("timeLeft") or "",
            "is_closed": bool(ls.get("isClosed")),
            "is_archived": bool(ls.get("isArchived")),
            "may_have_won": bool(ls.get("mayHaveWonStatus")),
            "status": ls.get("status") or "",
        })

    return {
        "auctions": auctions,
        "lots": lots,
        "total_count": paged.get("totalCount") or 0,
        "filtered_count": paged.get("filteredCount") or 0,
        "page_number": paged.get("pageNumber") or 1,
        "page_length": paged.get("pageLength") or 0,
    }


def fetch_all_current_bids(
    only_open: bool = False,
    only_winning: bool = False,
    month_range: str = "THREE_MONTHS",
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """Convenience wrapper that paginates through all current bids.

    Same return shape as ``fetch_current_bids`` but ``lots`` contains
    every result across pages.
    """
    page = 1
    all_lots: List[Dict[str, Any]] = []
    auctions: List[Dict[str, Any]] = []
    auction_ids_seen: set = set()
    total = 0
    filtered = 0
    while True:
        result = fetch_current_bids(
            page_length=100, page_number=page,
            only_open=only_open, only_winning=only_winning,
            month_range=month_range, token=token,
        )
        all_lots.extend(result["lots"])
        for a in result.get("auctions") or []:
            if a.get("id") not in auction_ids_seen:
                auctions.append(a)
                auction_ids_seen.add(a.get("id"))
        total = result["total_count"]
        filtered = result["filtered_count"]
        # Stop when we've fetched everything OR the page returned fewer
        # than page_length (safer than relying on totalCount alone).
        if len(result["lots"]) < 100 or len(all_lots) >= total:
            break
        page += 1
        if page > 50:
            # Safety: cap at 5000 lots. Anyone with more open bids is
            # an edge case worth investigating manually.
            break
    return {
        "auctions": auctions,
        "lots": all_lots,
        "total_count": total,
        "filtered_count": filtered,
    }


# ---------------------------------------------------------------------
# Watchlist — the lots the user has starred on HiBid (hibid.com/account/
# watchlist). Same GraphQL shape as CurrentBids (both return BuyerEvent
# lot lists), but the `watchList` root field also carries description /
# pictures / estimate per lot, so ONE call yields everything the comps +
# audit + vision pipeline needs — no per-lot fetches.
# ---------------------------------------------------------------------
_WATCHLIST_QUERY = """
query WatchListSearch(
  $pageNumber: Int!,
  $pageLength: Int!,
  $isArchived: Boolean = false,
  $hideClosedLots: Boolean = false,
  $auctionId: Int = null,
  $buyerLotStatusGroup: BuyerLotStatusGroup = null,
  $sortOrder: BuyerEventItemSortOrder = null,
  $monthRange: AltBidPastBidsRange = null,
  $sortDirection: SortDirection = DESC,
  $groupByAuction: Boolean = true,
  $auctionSortDirection: SortDirection = ASC
) {
  watchList(
    input: {
      isArchived: $isArchived,
      groupByAuction: $groupByAuction,
      auctionSortDirection: $auctionSortDirection,
      hideClosedLots: $hideClosedLots,
      auctionId: $auctionId,
      buyerLotStatusGroup: $buyerLotStatusGroup,
      sortOrder: $sortOrder,
      monthRange: $monthRange
    }
    pageNumber: $pageNumber
    pageLength: $pageLength
    sortDirection: $sortDirection
  ) {
    auctions {
      id
      eventName
      lotCount
      eventDateEnd
      eventCity
      eventState
      buyerPremium
      buyerPremiumRate
      shippingAndPickupInfo
    }
    pagedResults {
      totalCount
      filteredCount
      pageNumber
      pageLength
      results {
        id
        lotNumber
        lead
        description
        pictureCount
        estimate
        bidAmount
        auction {
          id
          eventName
          eventDateEnd
          buyerPremium
          buyerPremiumRate
          shippingAndPickupInfo
        }
        pictures {
          thumbnailLocation
          hdThumbnailLocation
          fullSizeLocation
        }
        lotState {
          highBid
          buyerHighBid
          buyerBidStatus
          minBid
          status
          timeLeft
          isClosed
          isArchived
        }
      }
    }
  }
}
"""


def _parse_watchlist_lot(r: Dict[str, Any]) -> Dict[str, Any]:
    """Shape one raw watchList result into a lead-ready dict."""
    ls = r.get("lotState") or {}
    a = r.get("auction") or {}
    pics = r.get("pictures") or []
    first = pics[0] if pics else {}
    # HiBid's watchlist exposes buyerPremium as FREE TEXT ("Buyers premium
    # is 20%") with buyerPremiumRate=1 as a "not filled" flag — so the
    # numeric _normalize_buyer_premium wrongly returns 100%. Use pass1's
    # text parser, which returns the MULTIPLIER form (1.20) the leads-df
    # grading pipeline expects, first-percent-wins for cash/card tiers.
    from .pass1 import parse_buyer_premium_pct
    bp_mult = parse_buyer_premium_pct(
        a.get("buyerPremium"), a.get("buyerPremiumRate"),
    )
    high = ls.get("highBid") or 0.0
    my = ls.get("buyerHighBid") or 0.0
    return {
        "lot_id": str(r.get("id") or ""),
        "lot_number": r.get("lotNumber"),
        "title": r.get("lead") or "",
        "description": r.get("description") or "",
        "auction_id": a.get("id"),
        "auction": a.get("eventName") or "",
        "closing_date": a.get("eventDateEnd") or "",
        "auction_cond_ship": a.get("shippingAndPickupInfo") or "",
        "auction_buyer_premium_pct": bp_mult,
        "current_bid": high,
        "my_bid": my,
        "next_bid": ls.get("minBid"),
        "bid_status": ls.get("buyerBidStatus") or "",
        "time_left": ls.get("timeLeft") or "",
        "is_closed": bool(ls.get("isClosed")),
        "auctioneer_est": r.get("estimate") or "",
        "image_count": r.get("pictureCount") or 0,
        "thumbnail_url": first.get("thumbnailLocation") or "",
        "hd_thumbnail_url": first.get("hdThumbnailLocation") or "",
        "fullsize_url": first.get("fullSizeLocation") or "",
        "lot_link": f"https://hibid.com/lot/{r.get('id')}" if r.get("id") else "",
    }


def fetch_watchlist(
    page_length: int = 100,
    page_number: int = 1,
    only_open: bool = False,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """Fetch one page of the user's HiBid watchlist (starred lots).

    Returns ``{auctions, lots, total_count, filtered_count, page_number,
    page_length}`` where each lot is a lead-ready dict (see
    ``_parse_watchlist_lot``): title, description, current_bid, next_bid,
    thumbnail_url, auction, buyer premium, etc.
    """
    variables = {
        "isArchived": False,
        "groupByAuction": True,
        "auctionSortDirection": "ASC",
        "hideClosedLots": bool(only_open),
        "auctionId": 0,
        "buyerLotStatusGroup": "ALL",
        "sortOrder": "SALES_ORDER",
        "monthRange": "THREE_MONTHS",
        "sortDirection": "ASC",
        "pageNumber": int(page_number),
        "pageLength": int(min(max(page_length, 1), 100)),
    }
    data = _post_graphql(
        "WatchListSearch", _WATCHLIST_QUERY, variables, token=token,
    )
    wl = data.get("watchList") or {}
    auctions = wl.get("auctions") or []
    paged = wl.get("pagedResults") or {}
    raw = paged.get("results") or []
    lots: List[Dict[str, Any]] = []
    for r in raw:
        if only_open and (r.get("lotState") or {}).get("isClosed"):
            continue
        lots.append(_parse_watchlist_lot(r))
    return {
        "auctions": auctions,
        "lots": lots,
        "total_count": paged.get("totalCount") or 0,
        "filtered_count": paged.get("filteredCount") or 0,
        "page_number": paged.get("pageNumber") or 1,
        "page_length": paged.get("pageLength") or 0,
    }


def fetch_all_watchlist(
    only_open: bool = False,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """Paginate through the entire watchlist. Same shape as
    ``fetch_watchlist`` but ``lots`` holds every page."""
    page = 1
    all_lots: List[Dict[str, Any]] = []
    auctions: List[Dict[str, Any]] = []
    seen: set = set()
    total = 0
    filtered = 0
    while True:
        result = fetch_watchlist(
            page_length=100, page_number=page,
            only_open=only_open, token=token,
        )
        all_lots.extend(result["lots"])
        for a in result.get("auctions") or []:
            if a.get("id") not in seen:
                auctions.append(a)
                seen.add(a.get("id"))
        total = result["total_count"]
        filtered = result["filtered_count"]
        if len(result["lots"]) < 100 or len(all_lots) >= (filtered or total):
            break
        page += 1
        if page > 50:  # safety cap — 5000 watched lots is an edge case
            break
    return {
        "auctions": auctions,
        "lots": all_lots,
        "total_count": total,
        "filtered_count": filtered,
    }


def _normalize_buyer_premium(
    buyer_premium: Any,
    buyer_premium_rate: Any,
) -> Optional[float]:
    """Resolve the auction's buyer-premium percentage to a 0.0-1.0 float.

    HiBid's GraphQL exposes both ``buyerPremium`` and ``buyerPremiumRate``
    on the auction type. From spot-checks across multiple auctioneers:
      - ``buyerPremium`` is an integer/float in 0-100 (a percent value)
      - ``buyerPremiumRate`` is sometimes a 0-1 decimal, sometimes
        the same percent value, depending on the auctioneer's setup.

    Strategy: prefer ``buyerPremium`` interpreted as percent
    (15.0 → 0.15). If that's missing/zero, fall back to
    ``buyerPremiumRate``, deciding based on magnitude whether it's
    a decimal (≤1) or a percent (>1).

    Returns None when neither field has a usable value — caller falls
    back to a flat default.
    """
    bp = _safe_float(buyer_premium)
    if bp and bp > 0:
        # Percent value: 15.0 → 0.15. Cap at 50% as a sanity check;
        # anything higher is almost certainly a decimal.
        if bp > 50:
            return None
        return round(bp / 100.0, 4)
    rate = _safe_float(buyer_premium_rate)
    if rate and rate > 0:
        if rate <= 1.0:
            return round(float(rate), 4)
        if rate <= 100.0:
            return round(rate / 100.0, 4)
    return None


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


__all__ = [
    "set_auth_token", "get_auth_token", "clear_auth_token", "session_metadata",
    "fetch_account_info", "fetch_current_bids", "fetch_all_current_bids",
    "fetch_watchlist", "fetch_all_watchlist",
    "auto_record_won_purchases",
]


# ---------------------------------------------------------------------
# Auto-record won purchases (the headline UX shortcut)
# ---------------------------------------------------------------------
# Default buyer's-premium estimate when we can't determine the
# auction-specific value. 15% covers the median HiBid auctioneer.
# Stored as a per-auction override in `.cache/auction_premiums.json`
# in a future iteration; today we just use the default.
_DEFAULT_BUYER_PREMIUM_PCT = 0.15
_DEFAULT_SHIP_COST = 25.0


def auto_record_won_purchases(
    bids_payload: Optional[Dict[str, Any]] = None,
    buyer_premium_pct: float = _DEFAULT_BUYER_PREMIUM_PCT,
    ship_cost_default: float = _DEFAULT_SHIP_COST,
) -> Dict[str, Any]:
    """Detect closed-and-won lots, write outcomes.purchase events for any
    that aren't yet recorded.

    This is the "no clicks needed" path for capturing purchases — we
    fetch the user's CurrentBids, scan for lots in a "you won this"
    state, and append ``outcomes.purchase`` rows for any whose
    (lot_id, event_type='purchase', event_ts) tuple isn't already in
    the outcomes table.

    Defaults that the user can correct later via a manual edit button:
      - paid_amount  = lot.highBid (HiBid's hammer price for the lot)
      - fees         = paid * buyer_premium_pct + ship_cost_default
      - won_at       = current timestamp (won_ts isn't in the response)
      - notes        = 'auto-recorded from HiBid current bids'

    Returns ``{'considered': N, 'new_purchases': N, 'already_recorded':
    N, 'skipped': N, 'errors': [..]}``.

    Why a snapshot from after-the-fact is OK: the predicted_resale we
    freeze comes from cached comp data (``comped_lots.json``) at the
    time we record. It's not "decision time" exactly, but it's
    same-day-or-week, which is good enough for calibration training
    pairs since comps don't drift more than a few % over a week.

    Lazy-imports outcomes + comped_lots to avoid circular dependency.
    """
    from . import outcomes
    from . import comped_lots

    if bids_payload is None:
        bids_payload = fetch_all_current_bids(only_open=False)

    lots = bids_payload.get('lots') or []

    stats = {
        'considered': 0,
        'new_purchases': 0,
        'already_recorded': 0,
        'skipped': 0,
        'errors': [],
    }

    for lot in lots:
        lot_id = str(lot.get('lot_id') or '')
        if not lot_id:
            continue
        if not _is_won_state(lot):
            continue
        stats['considered'] += 1

        # Skip if there's already a purchase event for this lot.
        existing = outcomes.get_events_for_lot(lot_id)
        if any(e.get('event_type') == 'purchase' for e in existing):
            stats['already_recorded'] += 1
            continue

        try:
            paid = float(lot.get('high_bid') or lot.get('my_bid') or 0.0)
            if paid <= 0:
                stats['skipped'] += 1
                continue
            # Use this lot's auction-specific buyer premium when HiBid
            # gave us one; fall back to the function-level default
            # (15%) for auctions that didn't expose the field.
            per_lot_pct = lot.get('auction_buyer_premium_pct')
            applied_pct = (
                float(per_lot_pct)
                if per_lot_pct is not None and per_lot_pct > 0
                else float(buyer_premium_pct)
            )
            premium = paid * applied_pct
            ship = float(ship_cost_default)
            title = lot.get('title') or ''
            premium_source = (
                f"{int(round(applied_pct * 100))}% (auction-specific)"
                if per_lot_pct is not None and per_lot_pct > 0
                else f"{int(round(applied_pct * 100))}% (default)"
            )

            # Pull predicted_resale snapshot from comped_lots.json if we
            # ever ran comps on this lot. Falls through to None when
            # the lot wasn't comped (typical for low-engagement lots).
            snapshot = _build_snapshot_from_cache(lot_id, title)

            ok = outcomes.record_purchase(
                lot_id=lot_id,
                paid_amount=paid,
                won_at=datetime.now().isoformat(),
                fees_paid=premium,
                ship_cost=ship,
                notes=(
                    f"auto-recorded from HiBid current bids · "
                    f"premium {premium_source} · "
                    f"ship ${ship:.0f} (default — review per auction)"
                ),
                snapshot=snapshot,
            )
            if ok:
                stats['new_purchases'] += 1
            else:
                stats['already_recorded'] += 1
        except Exception as e:
            stats['errors'].append({
                'lot_id': lot_id,
                'error': f"{type(e).__name__}: {e}",
            })

    return stats


def _is_won_state(lot: Dict[str, Any]) -> bool:
    """True iff this lot is in a "you won it" state.

    We require BOTH (a) the auction is closed and (b) the user is the
    high bidder. HiBid's ``buyerBidStatus`` and ``mayHaveWonStatus``
    fields together cover the common cases; ``isClosed`` is the
    safety check that prevents auto-recording WHILE bidding is
    still active (e.g., a lot temporarily at WINNING_NOW that the
    user later loses).
    """
    if not lot.get('is_closed'):
        return False
    status = (lot.get('bid_status') or '').upper()
    if status in ('WON', 'WINNING_END'):
        return True
    if lot.get('may_have_won') and status not in ('OUTBID', 'LOST'):
        return True
    return False


def _build_snapshot_from_cache(lot_id: str, title: str) -> Dict[str, Any]:
    """Construct the predicted-resale snapshot from cached comp data.

    Joins on lot_id with ``.cache/comped_lots.json``. When the lot
    was comped previously, captures predicted_resale + brand +
    category + tier from that cache. Otherwise tries the BOLO matcher
    on the title alone for at least the brand/category/tier.

    Returns the dict expected by ``outcomes.record_purchase`` —
    keys all set to None when no data is available.
    """
    from . import comped_lots
    from .bolo import BoloMatcher

    snapshot: Dict[str, Any] = {
        'predicted_resale': None,
        'predicted_profit': None,
        'brand': None,
        'category': None,
        'bolo_tier': None,
        'match_title': title,
    }

    # Layer 1: comped_lots.json (most authoritative when present)
    try:
        comp = comped_lots.get_any(lot_id)
        if comp:
            snapshot['predicted_resale'] = comp.get('est_resale')
            # No predicted_profit in comped_lots — we'd need est_cost too.
            # Leave as None; downstream computes it from
            # actual_paid + sale data instead.
    except Exception:
        pass

    # Layer 2: BOLO matcher on the title for brand/category/tier
    try:
        match = BoloMatcher().match(title or '', '')
        if match:
            snapshot['brand'] = match.get('brand') or snapshot['brand']
            snapshot['category'] = match.get('category') or snapshot['category']
            snapshot['bolo_tier'] = match.get('tier') or snapshot['bolo_tier']
    except Exception:
        pass

    return snapshot
