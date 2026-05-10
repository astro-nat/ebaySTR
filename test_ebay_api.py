"""Smoke-test the eBay Browse API price-lookup path.

Loads credentials from the same config_loader the rest of the app uses
(reads `config.yaml` -> ebay.app_id / ebay.cert_id, then falls back to
EBAY_APP_ID / EBAY_CERT_ID env vars). Never hardcode credentials in
this file — push protection blocks it.
"""
import base64
import os
import re
import statistics

import httpx


def _load_credentials():
    """Pull eBay creds from config.yaml or env vars; bail if missing."""
    try:
        from scraper.config_loader import load_config
        cfg = load_config().get("ebay", {}) or {}
        app_id = cfg.get("app_id")
        cert_id = cfg.get("cert_id")
        if app_id and cert_id:
            return app_id, cert_id
    except Exception:
        pass
    app_id = os.environ.get("EBAY_APP_ID")
    cert_id = os.environ.get("EBAY_CERT_ID")
    if not (app_id and cert_id):
        raise SystemExit(
            "eBay credentials not found. Set ebay.app_id / ebay.cert_id "
            "in config.yaml or export EBAY_APP_ID / EBAY_CERT_ID."
        )
    return app_id, cert_id


def get_token(app_id: str, cert_id: str) -> str:
    credentials = base64.b64encode(f"{app_id}:{cert_id}".encode()).decode()
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
    return resp.json()["access_token"]


def search_active_prices(token: str, query: str, limit: int = 10) -> list:
    """Search currently listed items to estimate market value."""
    # Clean up auction-style titles
    clean = re.sub(r"^\$\d+\s*", "", query)  # remove leading "$20 "
    clean = clean.rstrip(".").strip()
    if len(clean) < 5:
        return []

    resp = httpx.get(
        "https://api.ebay.com/buy/browse/v1/item_summary/search",
        headers={
            "Authorization": f"Bearer {token}",
            "X-EBAY-C-MARKETPLACE-ID": "EBAY_US",
        },
        params={
            "q": clean,
            "filter": "buyingOptions:{FIXED_PRICE}",
            "sort": "price",
            "limit": str(limit),
        },
        timeout=15,
    )

    if resp.status_code != 200:
        return []

    items = resp.json().get("itemSummaries", [])
    prices = []
    for item in items:
        price = float(item.get("price", {}).get("value", 0))
        if price > 0.99:
            prices.append(price)
    return prices


if __name__ == "__main__":
    app_id, cert_id = _load_credentials()
    token = get_token(app_id, cert_id)
    print("Got token\n")

    test_items = [
        "1988 Proof American Silver Eagle 1oz .999",
        "Apple Watch Series 2 42mm Smartwatch",
        "Coach Signature Dome Satchel Handbag",
        "$20 Red LED Flashing Jelly Bumpy Light Up Rings Pack of 36",
        "Autographed Rawlings NCAA Baseball",
    ]

    for item in test_items:
        print(f"Searching: '{item[:50]}'")
        prices = search_active_prices(token, item)
        if prices:
            med = statistics.median(prices)
            avg = statistics.mean(prices)
            print(f"  {len(prices)} listings, median=${med:.2f}, avg=${avg:.2f}")
            print(f"  Prices: {['${:.2f}'.format(p) for p in prices[:6]]}")
        else:
            print("  No results")
        print()
