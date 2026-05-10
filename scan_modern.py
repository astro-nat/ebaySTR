"""Second scan against modern multi-consignment auctions where late-90s/
2000s millennial-era nostalgia (Webkinz, Tamagotchi, Loungefly, AG modern,
Hannah Montana, Pokemon, Yu-Gi-Oh) is statistically more common."""
import asyncio, sys, pickle, time, os, io
from collections import defaultdict, Counter
from pathlib import Path
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, '.')
from scraper.pass1 import Phase1Scraper
from scraper.bolo import BoloMatcher

# Targeted: large multi-consignment / Texas Houston-area / general-mixed
# auctions where modern millennial nostalgia is more common than estate
# sales of older Boomer collectibles.
TARGET_AUCTIONS = [
    {"auction_id": 738781, "name": "TRUSTBID INC 81", "lot_count": 3258, "source": "Ship"},
    {"auction_id": 738176, "name": "MAY 6TH WEDNESDAY MORNING AUCTION FREE SHIPPING", "lot_count": 1363, "source": "Ship"},
    {"auction_id": 738807, "name": "050626 Auctionfuel Indy SHIPPING AVAILABLE", "lot_count": 1984, "source": "Ship"},
    {"auction_id": 738015, "name": "Open Consignmnet Auction 5/6/26 9AM", "lot_count": 756, "source": "Ship"},
    {"auction_id": 738206, "name": "April 29-May 6 Multi Consignment Auction 588", "lot_count": 573, "source": "Ship"},
    {"auction_id": 737922, "name": "Electronics & Home Goods Sugar Land 04/29-05/06", "lot_count": 321, "source": "Ship"},
]


async def main():
    cache_pkl = Path(".cache") / "modern_scan.pkl"
    use_cache = cache_pkl.exists() and not os.environ.get("REFETCH")
    if use_cache:
        print(f"Using cached scan from {cache_pkl}")
        with cache_pkl.open("rb") as f:
            df = pickle.load(f)
    else:
        scraper = Phase1Scraper()
        t0 = time.time()
        print(f"Fetching {sum(a['lot_count'] for a in TARGET_AUCTIONS):,} expected lots from {len(TARGET_AUCTIONS)} auctions...")
        df = await scraper.fetch_lots_for_selected(TARGET_AUCTIONS)
        elapsed = time.time() - t0
        print(f"Fetch elapsed: {elapsed:.1f}s; got {len(df):,} lots")
        cache_pkl.parent.mkdir(parents=True, exist_ok=True)
        with cache_pkl.open("wb") as f:
            pickle.dump(df, f)

    if df.empty:
        print("No lots returned.")
        return

    m = BoloMatcher()
    print(f"Matcher: {m.brand_count} brands from {len(m.paths)} files")
    print()

    EXPANDED = {
        "Loungefly", "Webkinz", "Tamagotchi (vintage P1/P2)",
        "American Girl Pleasant Company", "Vintage Gymboree (Rainbow Tag era)",
        "Hannah Montana collectibles", "Original Furby (1998-2000)",
        "Polly Pocket (Bluebird era)", "Vintage Cabbage Patch Coleco",
        "Vintage Mattel Barbie collector", "Madame Alexander vintage dolls",
        "Pokemon TCG (vintage 1st Edition)", "Yu-Gi-Oh TCG (vintage 1st Edition)",
        "Vintage Beanie Babies (errors + apex)",
        "High School Musical collectibles", "Neopets collectibles",
        "Chrome Hearts eyewear & cases", "Bentley OEM eyewear cases",
        "Maui Jim dealer display cases", "Oakley X-Metal vault & display cases",
        "Premium designer eyewear (Tom Ford / Persol Ratti / DITA / Oliver Peoples / Jacques Marie Mage)",
        "Designer sunglass cases & pouches (Christian Louboutin / Valentino / Celine / Brighton / BAPE / Miu Miu / Saint Laurent / Balmain / Harveys)",
    }

    by_brand = defaultdict(list)
    for _, row in df.iterrows():
        title = str(row.get('title', '') or '')
        desc = str(row.get('description', '') or '')
        r = m.match(title=title, description=desc)
        if r:
            by_brand[r['brand']].append({
                'title': title, 'model': r.get('matched_model'),
                'confidence': r.get('confidence'),
            })

    total = len(df)
    hits = sum(len(v) for v in by_brand.values())
    print('=' * 80)
    print(f'RESULTS: {hits:,} BOLO hits across {len(by_brand)} brands ({100*hits/max(1,total):.1f}% of {total:,} lots)')
    print('=' * 80)
    print()

    sorted_brands = sorted(by_brand.items(),
                           key=lambda kv: (kv[0] not in EXPANDED, -len(kv[1])))
    print('HITS BY BRAND (* = expanded this/prior sessions):')
    for brand, hs in sorted_brands:
        marker = '*' if brand in EXPANDED else ' '
        print(f'  {marker} {brand:60} {len(hs):>4} hits')

    print()
    print('=' * 80)
    print('SAMPLE HITS FOR EXPANDED BRANDS (up to 6 each)')
    print('=' * 80)
    for brand, hs in sorted_brands:
        if brand not in EXPANDED:
            continue
        print()
        print(f'* {brand} ({len(hs)} hits)')
        for h in hs[:6]:
            ttl = h['title'][:100]
            print(f"    [{h['confidence']:>11}] {ttl}")
            if h['model']:
                print(f"      -> matched_model: {h['model']}")


if __name__ == "__main__":
    asyncio.run(main())
