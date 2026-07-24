"""
Flask backend API server for the Resale Analyzer Android app.

Exposes POST /analyze:
  - Receives base64 JPEG image + purchase cost
  - Uses Claude Vision to identify the item
  - Queries eBay for sell-through rate (STR) and median sale price
  - Returns BUY / PASS verdict based on >=80% STR and >=500% ROI after fees

Run: python api_server.py
Default port: 5001
"""

import base64
import os

from dotenv import load_dotenv
from flask import Flask, jsonify, request

load_dotenv()

app = Flask(__name__)

EBAY_APP_ID = os.environ.get("EBAY_APP_ID", "")
EBAY_CERT_ID = os.environ.get("EBAY_CERT_ID", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
SCRAPINGBEE_KEY = os.environ.get("SCRAPINGBEE_KEY", "")

# eBay fee structure (buyer pays shipping, so no shipping deducted from proceeds)
EBAY_FINAL_VALUE_FEE_RATE = 0.1325  # 13.25% of sale price
EBAY_FIXED_FEE = 0.30               # $0.30 per order

# Decision thresholds
MIN_STR_PCT = 80.0   # 80% sell-through rate minimum
MIN_ROI = 5.0        # 500% ROI minimum (5.0x cost)


def identify_item(image_base64: str) -> str:
    """Use Claude Vision to produce a concise, eBay-searchable product name."""
    import anthropic

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=150,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_base64,
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            "Identify this item for an eBay search. "
                            "Reply with ONLY a concise product name (brand + model + key detail). "
                            "Examples: 'Nike Air Jordan 1 Retro High OG', "
                            "'Sony WH-1000XM4 Wireless Headphones', "
                            "'Nintendo Game Boy Color Purple'. "
                            "No explanations, no punctuation beyond the name itself."
                        ),
                    },
                ],
            }
        ],
    )
    return response.content[0].text.strip()


def calc_roi(sale_price: float, your_cost: float) -> float:
    """ROI after eBay fees. Buyer pays shipping so shipping is not deducted."""
    net = sale_price * (1 - EBAY_FINAL_VALUE_FEE_RATE) - EBAY_FIXED_FEE
    if your_cost <= 0:
        return float("inf")
    return (net - your_cost) / your_cost


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    image_b64 = data.get("image", "")
    try:
        cost = float(data.get("cost", 0))
    except (TypeError, ValueError):
        return jsonify({"error": "cost must be a number"}), 400

    if not image_b64:
        return jsonify({"error": "image field is required (base64 JPEG)"}), 400
    if cost <= 0:
        return jsonify({"error": "cost must be greater than 0"}), 400

    # --- Step 1: Identify item via Claude Vision ---
    try:
        item_name = identify_item(image_b64)
    except Exception as exc:
        return jsonify({"error": f"Item identification failed: {exc}"}), 500

    if not item_name:
        return jsonify({"error": "Could not identify item from photo"}), 422

    # --- Step 2: eBay price + STR lookup ---
    from scraper.ebay_prices import EbayPriceLookup

    lookup = EbayPriceLookup(
        app_id=EBAY_APP_ID,
        cert_id=EBAY_CERT_ID,
        scrapingbee_key=SCRAPINGBEE_KEY or None,
    )

    price_info = lookup.lookup_price_range(item_name)
    str_pct, str_source = lookup.lookup_str(item_name)

    if price_info is None:
        return jsonify(
            {
                "item_name": item_name,
                "verdict": "PASS",
                "reason": "No eBay sold data found — cannot verify price",
                "str_pct": str_pct,
                "str_source": str_source,
                "avg_sale_price": None,
                "price_low": None,
                "price_high": None,
                "comp_count": 0,
                "net_proceeds": None,
                "roi_pct": None,
                "ebay_fee": None,
                "your_cost": cost,
                "passes_str": False,
                "passes_roi": False,
            }
        )

    avg_price = price_info["median"]
    ebay_fee = round(avg_price * EBAY_FINAL_VALUE_FEE_RATE + EBAY_FIXED_FEE, 2)
    net_proceeds = round(avg_price - ebay_fee, 2)
    roi = calc_roi(avg_price, cost)
    roi_pct = round(roi * 100, 1)

    passes_str = str_pct is not None and str_pct >= MIN_STR_PCT
    passes_roi = roi >= MIN_ROI

    verdict = "BUY" if (passes_str and passes_roi) else "PASS"

    failure_reasons = []
    if not passes_str:
        str_display = f"{str_pct:.1f}%" if str_pct is not None else "N/A"
        failure_reasons.append(f"STR {str_display} (need ≥80%)")
    if not passes_roi:
        failure_reasons.append(f"ROI {roi_pct:.0f}% (need ≥500%)")

    reason = "; ".join(failure_reasons) if failure_reasons else "Meets all criteria"

    return jsonify(
        {
            "item_name": item_name,
            "verdict": verdict,
            "reason": reason,
            "str_pct": str_pct,
            "str_source": str_source,
            "avg_sale_price": round(avg_price, 2),
            "price_low": round(price_info.get("low", 0), 2),
            "price_high": round(price_info.get("high", 0), 2),
            "comp_count": price_info.get("count", 0),
            "net_proceeds": net_proceeds,
            "roi_pct": roi_pct,
            "ebay_fee": ebay_fee,
            "your_cost": cost,
            "passes_str": passes_str,
            "passes_roi": passes_roi,
        }
    )


if __name__ == "__main__":
    port = int(os.environ.get("API_PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)
