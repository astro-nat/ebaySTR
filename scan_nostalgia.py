"""Live BOLO scan against auctions likely to contain nostalgia brands.

Runs Phase 1 GraphQL scraper against a curated list of estate/consignment/
vintage auctions, then runs each lot through the BoloMatcher and reports
hits grouped by brand.

If `.cache/nostalgia_scan.pkl` exists, reuses cached lots instead of
re-fetching (set REFETCH=1 to force a fresh fetch).
"""
import asyncio, sys, pickle, time, os, io
from collections import defaultdict
from pathlib import Path

# Force UTF-8 stdout on Windows so we can print a wide range of chars
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, '.')

from scraper.pass1 import Phase1Scraper
from scraper.bolo import BoloMatcher

# Targeted: a mix of estate/consignment/vintage auctions where nostalgia
# brands are statistically most likely to appear, plus the dedicated
# Barbie Bonanza for Mattel Barbie / American Girl signal.
TARGET_AUCTIONS = [
    {"auction_id": 732210, "name": "Barbie Bonanza!", "lot_count": 477, "source": "Ship"},
    {"auction_id": 735677, "name": "April Take 2 JBS Vintage Auctions", "lot_count": 180, "source": "Ship"},
    {"auction_id": 735710, "name": "MONTPELIER UNIQUE ANTIQUES / VINTAGE AUCTION 1", "lot_count": 213, "source": "Ship"},
    {"auction_id": 736683, "name": "Living Estate Of Mary Ann Sturms Potter & Artist 1700 + Lots", "lot_count": 1778, "source": "Ship"},
    {"auction_id": 728478, "name": "HUGE Multi Estate Online Auction", "lot_count": 810, "source": "Ship"},
    {"auction_id": 731504, "name": "Midland Online Estate Auction", "lot_count": 1166, "source": "Ship"},
    {"auction_id": 738340, "name": "Gene & Marilyn Marsh Consignment Auction", "lot_count": 310, "source": "Ship"},
    {"auction_id": 737864, "name": "The Allen & Joan Hanawalt Estate Auction 3", "lot_count": 745, "source": "Ship"},
]


async def main():
    cache_pkl = Path(".cache") / "nostalgia_scan.pkl"
    use_cache = cache_pkl.exists() and not os.environ.get("REFETCH")

    if use_cache:
        print("=" * 80)
        print(f"Loading cached lots from {cache_pkl}")
        print("=" * 80)
        with cache_pkl.open("rb") as f:
            df = pickle.load(f)
        print(f"DataFrame shape: {df.shape}")
        print()
    else:
        print("=" * 80)
        print(f"Scanning {len(TARGET_AUCTIONS)} likely-nostalgia auctions")
        print(f"Total expected lots: {sum(a['lot_count'] for a in TARGET_AUCTIONS):,}")
        print("=" * 80)
        print()
        scraper = Phase1Scraper()
        t0 = time.time()
        df = await scraper.fetch_lots_for_selected(TARGET_AUCTIONS)
        elapsed = time.time() - t0
        print(f"Fetch elapsed: {elapsed:.1f}s")
        print(f"DataFrame shape: {df.shape}")
        if df.empty:
            print("No lots returned - exiting.")
            return
        cache_pkl.parent.mkdir(parents=True, exist_ok=True)
        with cache_pkl.open("wb") as f:
            pickle.dump(df, f)
        print(f"Cached to {cache_pkl}")
        print()

    # Now run each lot through the matcher
    m = BoloMatcher()
    print(f"Matcher loaded: {m.brand_count} brand patterns from {len(m.paths)} files")
    print()

    # Identify the title column
    title_col = None
    for c in ("title", "lead", "name"):
        if c in df.columns:
            title_col = c
            break
    if title_col is None:
        print(f"Couldn't find title column. Available: {list(df.columns)}")
        return

    desc_col = "description" if "description" in df.columns else None

    hits_by_brand = defaultdict(list)
    for i, row in df.iterrows():
        title = str(row.get(title_col, "") or "")
        desc = str(row.get(desc_col, "") or "") if desc_col else ""
        r = m.match(title=title, description=desc)
        if r:
            hits_by_brand[r["brand"]].append({
                "title": title,
                "model": r.get("matched_model"),
                "confidence": r.get("confidence"),
                "tier": r.get("tier"),
                "auction_id": row.get("auction_id"),
            })

    total_lots = len(df)
    total_hits = sum(len(v) for v in hits_by_brand.values())
    print("=" * 80)
    print(f"BOLO SCAN RESULTS")
    print("=" * 80)
    print(f"Total lots scanned: {total_lots:,}")
    print(f"Total BOLO hits: {total_hits:,} ({100.0 * total_hits / max(1, total_lots):.1f}%)")
    print(f"Unique brands hit: {len(hits_by_brand)}")
    print()

    # Brands we extra-care about (this session + prior session expansions)
    EXPANDED_BRANDS = {
        "Loungefly", "Webkinz", "Tamagotchi (vintage P1/P2)",
        "American Girl Pleasant Company",
        "Vintage Gymboree (Rainbow Tag era)",
        "Hannah Montana collectibles",
        "Original Furby (1998-2000)",
        "Polly Pocket (Bluebird era)",
        "Vintage Cabbage Patch Coleco",
        "Vintage Mattel Barbie collector",
        "Madame Alexander vintage dolls",
        "Pokemon TCG (vintage 1st Edition)",
        "Yu-Gi-Oh TCG (vintage 1st Edition)",
        "Vintage Beanie Babies (errors + apex)",
        "High School Musical collectibles",
        "Neopets collectibles",
        "Chrome Hearts eyewear & cases",
        "Bentley OEM eyewear cases",
        "Maui Jim dealer display cases",
        "Oakley X-Metal vault & display cases",
        "Premium designer eyewear (Tom Ford / Persol Ratti / DITA / Oliver Peoples / Jacques Marie Mage)",
        "Designer sunglass cases & pouches (Christian Louboutin / Valentino / Celine / Brighton / BAPE / Miu Miu / Saint Laurent / Balmain / Harveys)",
    }

    # Sort: expanded brands first by hit count desc, then others
    sorted_brands = sorted(
        hits_by_brand.items(),
        key=lambda kv: (kv[0] not in EXPANDED_BRANDS, -len(kv[1]))
    )

    print("HITS BY BRAND (* = expanded this/prior sessions):")
    print()
    for brand, hits in sorted_brands:
        marker = "*" if brand in EXPANDED_BRANDS else " "
        print(f"  {marker} {brand:55} {len(hits):>4} hits")

    print()
    print("=" * 80)
    print("SAMPLE HITS FOR EXPANDED BRANDS (up to 8 per brand)")
    print("=" * 80)
    for brand, hits in sorted_brands:
        if brand not in EXPANDED_BRANDS:
            continue
        print()
        print(f"* {brand} ({len(hits)} hits)")
        for h in hits[:8]:
            ttl = h["title"][:100]
            print(f"    [{h['confidence']:>11}] {ttl}")
            if h["model"]:
                print(f"      -> matched_model: {h['model']}")


if __name__ == "__main__":
    asyncio.run(main())
