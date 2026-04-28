import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import asyncio
import os
import pickle
import re
from datetime import datetime, timedelta
from pathlib import Path

# --- IMPORT MODULES ---
from scraper import Phase1Scraper
from scraper.cache import AuctionCache, merge_cached_analysis

# Single shared cache instance; auto-creates the dir on first touch
_AUCTION_CACHE = AuctionCache()

# --- Persistent discovery-result cache ---
# Streamlit session state is wiped on browser refresh / app restart, which
# means a successful "Discover Auctions" run only sticks around for the
# current session. Persist the candidate list + the sourcing config that
# produced it to disk so we can rehydrate on the next app load without
# making the user click the button again. TTL: 24 hours (bids/closing
# times move fast enough that day-stale data is the outer edge of useful).
_DISCOVERY_CACHE_PATH = Path(".cache") / "last_discovery.pkl"
_DISCOVERY_CACHE_TTL = timedelta(hours=24)
# Bump this when the cached payload shape changes — old caches are silently
# discarded so users don't have to delete the file by hand. v4 drops the
# thumbnail_urls field (rendering 4 images per card across 200+ auctions
# was too slow to load); images are now reserved for the lot-results table.
_DISCOVERY_CACHE_VERSION = 4


def _save_cached_discovery(candidates, sourcing_cfg, category_samples=None):
    """Persist a successful discovery result for the next session."""
    try:
        _DISCOVERY_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": _DISCOVERY_CACHE_VERSION,
            "saved_at": datetime.now().isoformat(),
            "sourcing_cfg": dict(sourcing_cfg or {}),
            "candidates": list(candidates or []),
            # Sampled lot previews (categories + titles + thumbnail_url) so
            # the picker rehydrates on reload.
            "category_samples": dict(category_samples or {}),
        }
        with open(_DISCOVERY_CACHE_PATH, "wb") as fh:
            pickle.dump(payload, fh)
    except Exception:
        # Cache persistence is best-effort; never let it break the app.
        pass


def _load_cached_discovery():
    """Return a dict with candidates/sourcing_cfg/age if fresh, else None.

    Caches written by an older schema (no schema_version, or a smaller
    number than the current one) are silently discarded — re-running
    Discover repopulates the file with the new shape.
    """
    try:
        if not _DISCOVERY_CACHE_PATH.exists():
            return None
        with open(_DISCOVERY_CACHE_PATH, "rb") as fh:
            payload = pickle.load(fh)
        if payload.get("schema_version", 1) < _DISCOVERY_CACHE_VERSION:
            return None
        saved_at = datetime.fromisoformat(payload.get("saved_at", ""))
        age = datetime.now() - saved_at
        if age > _DISCOVERY_CACHE_TTL:
            return None
        return {
            "candidates": payload.get("candidates") or [],
            "sourcing_cfg": payload.get("sourcing_cfg") or {},
            "category_samples": payload.get("category_samples") or {},
            "saved_at": saved_at,
            "age": age,
        }
    except Exception:
        return None


# --- CATEGORY GROUPING ---
# HiBid emits 30+ fine-grained category labels per discovery run (Halloween
# Decor, Other Baby, Linens / Curtains, Outdoor Games & Sports Equipment,
# …). Showing 30 checkboxes is unusable, so we collapse similar categories
# into ~12 broad groups via keyword match. Groups are checked in order —
# first match wins — and anything that matches no group falls into "Other".
#
# Each entry: (emoji + label, [lowercase keywords]). Keywords are matched
# as substrings against the lowercased HiBid category; grouping a new
# category that HiBid starts emitting just requires its label containing
# one of these keywords.
_CATEGORY_GROUPS = [
    ("🎮 Electronics", [
        "electronic", "computer", "laptop", "tablet", "phone", "camera",
        "audio", "video", "tv ", " tv", "gaming", "console", "headphone",
        "speaker", "drone",
    ]),
    ("🔧 Tools & Hardware", [
        "tool", "hardware", "power tool", "workshop", "garage", "drill",
        "saw", "welding", "mechanic",
    ]),
    ("🏠 Home & Kitchen", [
        "kitchen", "cookware", "bakeware", "appliance", "linen", "curtain",
        "bedding", "bath", "lighting", "lamp", "cleaning", "vacuum",
        "laundry", "storage", "organizer", "home decor", "decor ", " decor",
        "furniture", "rug",
    ]),
    ("🧥 Clothing & Accessories", [
        "clothing", "apparel", "shoe", "footwear", "jewelry", "watch",
        "handbag", "purse", "wallet", "accessories", "hat", "scarf",
    ]),
    ("🧸 Toys & Baby", [
        "toy", "game", "baby", "infant", "kid", "children", "nursery",
        "stroller", "puzzle", "doll", "lego", "action figure",
    ]),
    ("🎯 Sporting & Outdoors", [
        "sport", "fitness", "exercise", "hunting", "fishing", "camping",
        "hiking", "bike", "cycling", "golf", "outdoor games", "archery",
        "firearm", "ammo",
    ]),
    ("🚗 Automotive", [
        "automotive", "auto ", " auto", "vehicle", "motorcycle", "atv",
        "utv", "boat", "rv ", " rv", "trailer", "tire", "car ",
    ]),
    ("🎃 Seasonal & Decor", [
        "halloween", "christmas", "holiday", "easter", "thanksgiving",
        "seasonal", "party", "wedding",
    ]),
    ("🎨 Art & Collectibles", [
        "art", "antique", "vintage", "collectible", "coin", "currency",
        "stamp", "glassware", "pottery", "sculpture", "painting",
        "memorabilia", "trading card",
    ]),
    ("🎵 Music, Books & Media", [
        "music", "musical instrument", "guitar", "piano", "record",
        "vinyl", "cd", "dvd", "book", "magazine", "movie",
    ]),
    ("🍔 Food, Health & Beauty", [
        "food", "beverage", "drink", "snack", "supplement", "vitamin",
        "health", "beauty", "cosmetic", "skincare", "personal care",
        "hair", "bath & body",
    ]),
    ("🐕 Pets", [
        "pet", "dog", "cat", "aquarium", "bird", "animal",
    ]),
    ("🌱 Yard & Garden", [
        "garden", "lawn", "yard", "landscap", "plant", "patio",
        "greenhouse", "mower",
    ]),
    ("🏢 Business & Industrial", [
        "office", "industrial", "commercial", "medical equipment",
        "janitorial", "retail fixture", "restaurant equipment",
    ]),
]


def _classify_category(raw_category: str) -> str:
    """Return the group label for a HiBid category. 'Other' if nothing matches."""
    if not raw_category:
        return "❓ Uncategorized"
    low = str(raw_category).lower()
    for label, keywords in _CATEGORY_GROUPS:
        for kw in keywords:
            if kw in low:
                return label
    return "📦 Other"


def _build_category_filter(df, state_key: str = "category_group_picks"):
    """Render a checkbox row for category filtering, return the filtered df.

    The UI shows one checkbox per group that has lots in the current df,
    labeled with the per-group count. Selections persist in session_state
    so they survive reruns. When nothing is ticked, everything passes
    through — same semantics as the old multiselect.
    """
    if 'category' not in df.columns or df.empty:
        return df

    # Classify every row and count per group
    groups = df['category'].fillna('').astype(str).apply(_classify_category)
    counts = groups.value_counts()
    if counts.empty:
        return df

    # Order: keep _CATEGORY_GROUPS order, then Other, then Uncategorized.
    # Only show groups that actually have lots.
    ordered_labels = [g[0] for g in _CATEGORY_GROUPS] + ["📦 Other", "❓ Uncategorized"]
    visible = [g for g in ordered_labels if g in counts.index]

    # Seed session state if empty
    if state_key not in st.session_state:
        st.session_state[state_key] = set()

    with st.expander(
        f"🏷️ Filter by category group ({len(visible)} groups, "
        f"{len(st.session_state[state_key])} selected)",
        expanded=False,
    ):
        # Toolbar row: Select all / Clear
        tc1, tc2, _ = st.columns([1, 1, 4])
        with tc1:
            if st.button("Select all", key=f"{state_key}_all",
                         use_container_width=True):
                st.session_state[state_key] = set(visible)
                st.rerun()
        with tc2:
            if st.button("Clear", key=f"{state_key}_clear",
                         use_container_width=True):
                st.session_state[state_key] = set()
                st.rerun()

        # Checkbox grid — 4 per row on wide screens, wraps on mobile via CSS
        cols_per_row = 4
        for i in range(0, len(visible), cols_per_row):
            chunk = visible[i:i + cols_per_row]
            cols = st.columns(cols_per_row)
            for label, col in zip(chunk, cols):
                with col:
                    checked_now = label in st.session_state[state_key]
                    new = st.checkbox(
                        f"{label} ({counts[label]})",
                        value=checked_now,
                        key=f"{state_key}_cb_{label}",
                    )
                    if new and not checked_now:
                        st.session_state[state_key].add(label)
                    elif not new and checked_now:
                        st.session_state[state_key].discard(label)

    picks = st.session_state[state_key] & set(visible)
    if not picks:
        return df
    # Keep rows whose computed group is in picks
    mask = groups.isin(picks)
    return df[mask]


# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="H-Town TX Finds: ROI Engine",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- MOBILE-RESPONSIVE CSS ---
st.markdown("""
<style>
/* ---- Touch-friendly tabs ---- */
button[data-baseweb="tab"] {
    padding: 12px 8px !important;
    font-size: 14px !important;
}

/* ---- Larger tap targets for buttons ---- */
.stButton > button {
    min-height: 48px !important;
    font-size: 15px !important;
}

/* ---- Compact title on small screens ---- */
@media (max-width: 640px) {
    /* Stack metric columns vertically */
    [data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important;
        gap: 4px !important;
    }
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        flex: 1 1 45% !important;
        min-width: 45% !important;
    }

    /* Shrink header */
    h1 { font-size: 1.5rem !important; }
    h2 { font-size: 1.2rem !important; }
    h3 { font-size: 1.1rem !important; }

    /* Wider sidebar when open on mobile */
    [data-testid="stSidebar"] {
        min-width: 85vw !important;
        max-width: 85vw !important;
    }

    /* Metric cards: tighter padding */
    [data-testid="stMetric"] {
        padding: 8px 4px !important;
    }
    [data-testid="stMetricLabel"] {
        font-size: 11px !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 18px !important;
    }

    /* Dataframe horizontal scroll hint */
    [data-testid="stDataFrame"] {
        overflow-x: auto !important;
        -webkit-overflow-scrolling: touch !important;
    }

    /* Tab text: shorter labels */
    button[data-baseweb="tab"] {
        padding: 10px 4px !important;
        font-size: 12px !important;
    }
}

/* Removed: a global @media (max-width: 960px) block here used to wrap
   every stHorizontalBlock to 48% min-width, which broke the picker
   table at any narrow-but-not-mobile width. The <640px rule above
   still handles real phones. */
</style>
""", unsafe_allow_html=True)

# --- STATE MANAGEMENT ---
if 'phase1_leads' not in st.session_state:
    st.session_state.phase1_leads = pd.DataFrame()

if 'selected_leads' not in st.session_state:
    st.session_state.selected_leads = pd.DataFrame()

if 'current_auction' not in st.session_state:
    st.session_state.current_auction = None

if 'audit_results' not in st.session_state:
    st.session_state.audit_results = {}

if 'audit_running' not in st.session_state:
    st.session_state.audit_running = False

# --- Audit speed/accuracy knobs ---
# Default: bart-large-mnli (already cached on-disk, loads instantly).
# User can opt into the smaller/faster distilbart model via UI toggle — that
# one triggers a ~500MB download the first time it's used.
if 'audit_fast_mode' not in st.session_state:
    st.session_state.audit_fast_mode = False
if 'audit_batch_size' not in st.session_state:
    # 8 is a safe CPU default. Bigger = faster until RAM runs out.
    st.session_state.audit_batch_size = 8

if 'comps_running' not in st.session_state:
    st.session_state.comps_running = False

if 'img_enrich_running' not in st.session_state:
    st.session_state.img_enrich_running = False

if 'img_enrich_min_bid' not in st.session_state:
    # Skip image enrichment on lots below this bid (junk filter). Tunable
    # from the Step 1.5 UI panel.
    st.session_state.img_enrich_min_bid = 5.0

# --- Pre-comps filter knobs (Step 2 panel) ---
# Comps are the most expensive step — letting the user prune the target set
# before launch can cut runtime by 2–10x on big auctions.
if 'comps_min_bid' not in st.session_state:
    st.session_state.comps_min_bid = 5.0
if 'comps_max_lots' not in st.session_state:
    # 0 = no cap
    st.session_state.comps_max_lots = 0
if 'comps_exclude_hard' not in st.session_state:
    st.session_state.comps_exclude_hard = True
if 'comps_only_img_promoted' not in st.session_state:
    st.session_state.comps_only_img_promoted = False
if 'comps_use_auction_str' not in st.session_state:
    # When True, sample STR per-auction (3 lots each) instead of scraping
    # STR for every lot. Huge speedup for big auctions.
    st.session_state.comps_use_auction_str = True
if 'comps_workers' not in st.session_state:
    # Thread-pool size for parallel price comps. 8 hits a good balance:
    # ~8x speedup without getting throttled by eBay/Mercari scraping.
    st.session_state.comps_workers = 8

if 'cache_ttl_days' not in st.session_state:
    st.session_state.cache_ttl_days = 14

if 'cache_purged_this_session' not in st.session_state:
    # Purge expired entries once per session, not every rerun
    _AUCTION_CACHE.purge_expired(ttl_days=st.session_state.cache_ttl_days)
    st.session_state.cache_purged_this_session = True

if 'auction_candidates' not in st.session_state:
    # List of dicts from Phase1Scraper.fetch_auction_candidates() — the
    # "step 1" output, before the user picks which auctions to deep-scan.
    # First-load hydration: if a successful discovery <24h old is on disk,
    # restore it so the user doesn't have to re-click "Discover Auctions"
    # every time they reopen the app / refresh the tab.
    _cached_disc = _load_cached_discovery()
    if _cached_disc and _cached_disc["candidates"]:
        st.session_state.auction_candidates = _cached_disc["candidates"]
        st.session_state._discovery_restored_from = _cached_disc["saved_at"]
        st.session_state._sourcing_cfg = _cached_disc["sourcing_cfg"]
        # Also rehydrate the sampled lot previews so the picker's
        # "What's in this auction" column works after a tab refresh.
        st.session_state.category_samples = _cached_disc.get(
            "category_samples", {}
        )
    else:
        st.session_state.auction_candidates = []

if 'category_samples' not in st.session_state:
    # {auction_id: {"categories": [...], "cat_counts": {...}, "titles": [...]}}
    # from sample_categories_batch(). Older versions stored a plain list of
    # category names — the picker reader tolerates both shapes.
    st.session_state.category_samples = {}

if 'discover_running' not in st.session_state:
    st.session_state.discover_running = False

if 'fetch_lots_running' not in st.session_state:
    st.session_state.fetch_lots_running = False

if 'known_categories' not in st.session_state:
    # Common HiBid lot categories as a starter set. Grown over time from any
    # unique category strings we see in scrape results.
    st.session_state.known_categories = [
        "Antiques", "Art", "Automotive", "Books & Media",
        "Clothing & Accessories", "Coins & Currency", "Collectibles",
        "Electronics", "Firearms", "Fishing", "Furniture",
        "Glassware", "Home & Garden", "Hunting", "Jewelry",
        "Kitchen", "Music & Instruments", "Outdoors", "Pottery",
        "Sporting Goods", "Sports Memorabilia", "Tools",
        "Toys & Games", "Vintage",
    ]

# --- SCREEN WAKE LOCK (mobile keep-awake) ---
def _keep_screen_awake():
    """Ask the browser to keep the screen on during a long-running op.

    Uses the Wake Lock API (Chrome/Android, iOS 16.4+). The earlier button
    click counts as the user gesture most browsers require. On devices
    where Wake Lock isn't supported this is a silent no-op — users need
    to set their phone's auto-lock setting manually.

    The script is injected into a zero-height component iframe; it also
    re-requests the lock when the tab becomes visible again (handy if
    the user briefly switches apps).
    """
    components.html(
        """
        <script>
        (async () => {
            if (!('wakeLock' in navigator)) return;
            try {
                const lock = await navigator.wakeLock.request('screen');
                window._htFindsWakeLock = lock;
                lock.addEventListener('release', () => {
                    window._htFindsWakeLock = null;
                });
            } catch (e) { /* user-gesture / permission issue; fail silent */ }
        })();
        document.addEventListener('visibilitychange', async () => {
            if (document.visibilityState === 'visible' &&
                !window._htFindsWakeLock &&
                'wakeLock' in navigator) {
                try {
                    window._htFindsWakeLock = await navigator.wakeLock.request('screen');
                } catch (e) {}
            }
        });
        </script>
        """,
        height=0,
    )


# --- ASYNC WRAPPER ---
def run_async_scraper(scraper_instance, progress_callback=None):
    """Safely runs the asyncio scraper within Streamlit's synchronous thread."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(scraper_instance.run(progress_callback=progress_callback))


def run_async(coro):
    """Run an arbitrary coroutine from Streamlit's sync thread."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)

# --- MAIN DASHBOARD UI ---
st.title("🛰️ Auction Intelligence Dashboard")
st.markdown("Automated sourcing and risk-assessment for H-Town TX Finds.")

# --- Top-of-page controls (formerly the left sidebar) ---
# The sidebar pushed the main content around when resizing the window, so
# both panels now live as expanders directly under the title. Sourcing
# starts open because users typically tweak it before each run; Memory
# stays collapsed since it's a less-frequent peek.
with st.expander("📍 Sourcing", expanded=True):
    sc_a, sc_b = st.columns(2)
    with sc_a:
        user_zip = st.text_input("Home Zip Code", value="77058")
        radius = st.slider("Local Pickup Radius (mi)", 5, 100, 20)
        include_nationwide = st.checkbox(
            "Include Nationwide (Ship-to-Me)", value=True,
        )
    with sc_b:
        closing_days = st.slider("Closing Within (days)", 1, 30, 1)
        category_filter = st.multiselect(
            "🏷️ Categories (optional)",
            options=sorted(set(st.session_state.known_categories)),
            placeholder="All categories",
            help=(
                "Only keep lots whose category matches any selected term "
                "(substring, case-insensitive). Saves time in Phase 2 by "
                "dropping irrelevant items. Leave blank to fetch everything."
            ),
        )

    # --- Step 1: discover auction candidates (cheap — no per-lot fetch) ---
    discover_running = st.session_state.get('discover_running', False)
    fetch_lots_running = st.session_state.get('fetch_lots_running', False)
    any_running = discover_running or fetch_lots_running

    # Label the button differently when we're sitting on restored-from-disk
    # results, so the click reads as "refresh" rather than "start from zero".
    _restored_at = st.session_state.get('_discovery_restored_from')
    if discover_running:
        discover_label = "⏳ Discovering…"
    elif _restored_at:
        discover_label = "🔄 Refresh Auctions"
    else:
        discover_label = "🔍 Discover Auctions"
    if st.button(
        discover_label,
        type="primary",
        use_container_width=True,
        disabled=any_running,
        key="discover_btn",
        help="Step 1 of 2: fetch the LIST of open auctions (no per-lot data yet). "
             "You'll then pick which ones are worth a deep scan. "
             "Successful runs are cached for 24h and auto-restored on reload.",
    ):
        st.session_state._sourcing_cfg = {
            "zip": user_zip,
            "radius": radius,
            "include_nationwide": include_nationwide,
            "closing_days": closing_days,
            "category_filter": category_filter,
        }
        st.session_state.discover_running = True
        st.rerun()

    if _restored_at and not discover_running:
        _age = datetime.now() - _restored_at
        if _age.total_seconds() < 3600:
            _age_str = f"{int(_age.total_seconds() / 60)} min ago"
        else:
            _age_str = f"{_age.total_seconds() / 3600:.1f}h ago"
        st.caption(
            f"♻️ Showing results from **{_age_str}** "
            f"({len(st.session_state.auction_candidates)} auctions). "
            f"Click to refresh."
        )

with st.expander("💾 Memory", expanded=False):
    cached_list = _AUCTION_CACHE.list_all(ttl_days=st.session_state.cache_ttl_days)
    fresh_count = sum(1 for c in cached_list if c['fresh'])
    st.caption(
        f"**{fresh_count}** auction(s) cached. "
        f"Audit + price-comp results are reused when you re-open an auction — "
        f"current bids refresh every discovery run."
    )
    st.session_state.cache_ttl_days = st.slider(
        "Auto-purge after (days)",
        min_value=1, max_value=30,
        value=int(st.session_state.cache_ttl_days),
        help="Cached analyses older than this get deleted automatically. "
             "Auctions are also purged as soon as their closing date passes.",
    )

    if cached_list:
        with st.expander(f"📋 View {len(cached_list)} cached entries", expanded=False):
            for entry in cached_list[:25]:
                badge = "🟢" if entry['fresh'] else "🔴 stale"
                try:
                    cached_at = datetime.fromisoformat(entry['cached_at'])
                    age = datetime.now() - cached_at
                    age_str = f"{age.days}d ago" if age.days > 0 else f"{int(age.seconds / 3600)}h ago"
                except Exception:
                    age_str = "?"
                st.caption(f"{badge} **{entry['auction_name']}** — {entry['items']} items · {age_str}")
            if len(cached_list) > 25:
                st.caption(f"...and {len(cached_list) - 25} more")

    if st.button("🗑️ Clear all memory", use_container_width=True,
                 help="Delete every cached auction analysis. Use if results feel stale."):
        removed = _AUCTION_CACHE.clear_all()
        st.success(f"Cleared {removed} cached auction(s).")
        st.rerun()

# --- Surface any persisted discover/fetch status so errors don't vanish on rerun ---
for _key, _tb_key, _label in (
    ('_discover_status', '_last_discover_traceback', 'Discovery'),
    ('_fetch_status', '_last_fetch_traceback', 'Lot fetch'),
):
    _status = st.session_state.pop(_key, None)
    if not _status:
        continue
    if _status.get('error'):
        st.error(f"❌ {_label} failed: {_status['error']}")
        tb = st.session_state.get(_tb_key)
        if tb:
            with st.expander("🔍 Full traceback (share this if you ask for help)"):
                st.code(tb, language="python")
    elif _status.get('msg'):
        msg = _status['msg']
        if msg.startswith("⚠️"):
            st.warning(msg)
        else:
            st.success(msg)

    # Render full fetch diagnostics so the user can see exactly what HiBid
    # returned even after the rerun wipes the in-status widgets.
    _diag = _status.get('diag') if isinstance(_status, dict) else None
    if _diag:
        with st.expander(
            f"🔬 {_label} diagnostics "
            f"(raw: {_diag.get('raw_count', 0)} · "
            f"kept: {_diag.get('kept', 0)} · "
            f"closed: {_diag.get('filtered_status', 0)} · "
            f"category: {_diag.get('filtered_cat', 0)})",
            expanded=(_diag.get('kept', 0) == 0),
        ):
            st.write(
                f"- **Raw lots from HiBid:** {_diag.get('raw_count', 0)}\n"
                f"- **Kept after filtering:** {_diag.get('kept', 0)}\n"
                f"- **Dropped as CLOSED / Bidding Closed:** {_diag.get('filtered_status', 0)}\n"
                f"- **Dropped by sidebar category filter:** {_diag.get('filtered_cat', 0)}"
            )
            _sv = _diag.get('status_values') or {}
            if _sv:
                st.write(f"**lotState.status values seen:** `{_sv}`")
            _pa = _diag.get('per_auction') or []
            if _pa:
                st.caption(f"Per-auction breakdown ({len(_pa)}):")
                st.table(pd.DataFrame(_pa))
            _errs = _diag.get('errors') or []
            if _errs:
                st.warning(f"{len(_errs)} auction(s) errored during fetch:")
                st.table(pd.DataFrame(_errs))


# ================================================================
# WORK BLOCK: Discover Auctions (lives in main area so mobile users
# with collapsed sidebar can actually SEE progress / errors)
# ================================================================
if st.session_state.get('discover_running'):
    _keep_screen_awake()
    discover_error = None
    discover_result_msg = None
    with st.status("🔍 Discovering auctions…", expanded=True) as status_box:
        try:
            cfg = st.session_state.get('_sourcing_cfg', {})
            scraper = Phase1Scraper(config_path="config.json")
            scraper.zip_code = cfg.get("zip", "")
            scraper.radius = cfg.get("radius", 20)
            scraper.include_nationwide = cfg.get("include_nationwide", True)
            scraper.closing_within_days = cfg.get("closing_days", 1)
            scraper.category_filter = cfg.get("category_filter", [])

            st.write(
                f"Querying HiBid near **{scraper.zip_code}** within "
                f"**{scraper.radius} mi**, closing within "
                f"**{scraper.closing_within_days} day(s)**"
                + (", including nationwide shippable." if scraper.include_nationwide else ".")
            )
            scan_progress = st.progress(0, text="Starting...")

            def discover_prog(current, total, label=""):
                if total == 0:
                    pct, text = 0.0, (label or "Done")
                elif current == 0 and total == 1:
                    pct, text = 0.0, (label or "Working...")
                else:
                    pct = current / total if total > 0 else 0
                    text = label or f"{current}/{total}"
                scan_progress.progress(min(pct, 1.0), text=text)

            candidates = run_async(
                scraper.fetch_auction_candidates(progress_callback=discover_prog)
            )
            scan_progress.empty()

            # Reset downstream state: a new candidate list invalidates prior picks + lots
            st.session_state.auction_candidates = candidates
            st.session_state.category_samples = {}
            st.session_state.phase1_leads = pd.DataFrame()
            st.session_state.audit_results = {}
            st.session_state.selected_leads = pd.DataFrame()
            st.session_state.current_auction = None

            # Auto-sample lot previews for every candidate so the picker's
            # "What's in this auction" column is populated without the user
            # having to click a second button. Cheap (one GraphQL call per
            # auction, batched 15-wide).
            cat_samples_map: dict = {}
            if candidates:
                st.write(f"Previewing lots for **{len(candidates)}** auctions…")
                sample_progress = st.progress(0, text="Sampling 0/…")

                def _auto_sample_prog(current, total, label=""):
                    pct = current / total if total > 0 else 1.0
                    sample_progress.progress(
                        min(pct, 1.0), text=label or f"{current}/{total}",
                    )

                try:
                    cat_samples_map = run_async(
                        scraper.sample_categories_batch(
                            candidates, sample_size=20,
                            progress_callback=_auto_sample_prog,
                        )
                    )
                except Exception:
                    # Sampling is a nice-to-have; never fail the whole
                    # discovery just because a preview call blew up.
                    cat_samples_map = {}
                sample_progress.empty()
                st.session_state.category_samples = cat_samples_map

            if candidates:
                discover_result_msg = (
                    f"✅ Found {len(candidates)} candidate auction(s). "
                    "Pick which to deep-scan below."
                )
                status_box.update(
                    label=f"✅ Found {len(candidates)} auctions",
                    state="complete", expanded=False,
                )
                # Persist so the next page load / tab refresh restores
                # this list automatically (24h TTL).
                _save_cached_discovery(
                    candidates,
                    st.session_state.get('_sourcing_cfg', {}),
                    cat_samples_map,
                )
                # Fresh run supersedes any restored-from-disk marker.
                st.session_state.pop('_discovery_restored_from', None)
            else:
                discover_result_msg = "⚠️ No auctions matched your filters."
                status_box.update(
                    label="⚠️ No matching auctions",
                    state="error", expanded=True,
                )
        except Exception as e:
            import traceback
            discover_error = f"{type(e).__name__}: {e}"
            st.session_state._last_discover_traceback = traceback.format_exc()
            st.error(f"❌ {discover_error}")
            st.code(traceback.format_exc(), language="python")
            status_box.update(label="❌ Discovery failed", state="error", expanded=True)
        finally:
            st.session_state.discover_running = False

    st.session_state._discover_status = {
        "error": discover_error,
        "msg": discover_result_msg,
    }
    st.rerun()


# ================================================================
# WORK BLOCK: Fetch lots for selected auctions
# ================================================================
if st.session_state.get('fetch_lots_running'):
    _keep_screen_awake()
    fetch_error = None
    fetch_result_msg = None
    with st.status("📥 Fetching lots for selected auctions…", expanded=True) as status_box:
        try:
            candidates = st.session_state.get('auction_candidates', [])
            sel_ids = set(st.session_state.get('_selected_auction_ids', []))
            selected_candidates = [c for c in candidates if c['auction_id'] in sel_ids]

            if not selected_candidates:
                raise RuntimeError(
                    f"No matching auctions found for selected IDs ({len(sel_ids)} ids, "
                    f"{len(candidates)} candidates). This usually means the candidate "
                    "list was reset between clicks."
                )

            st.write(
                f"Deep-scanning **{len(selected_candidates)}** auction(s): "
                + ", ".join(
                    f"*{a.get('name', a.get('auction_id'))}*"
                    for a in selected_candidates[:3]
                )
                + (f" + {len(selected_candidates) - 3} more"
                   if len(selected_candidates) > 3 else "")
                + "."
            )

            cfg = st.session_state.get('_sourcing_cfg', {})
            scraper = Phase1Scraper(config_path="config.json")
            scraper.category_filter = cfg.get("category_filter", [])

            fetch_progress = st.progress(0, text="Fetching lots…")

            def fetch_prog(current, total, label=""):
                if total == 0:
                    pct, text = 0.0, (label or "Done")
                else:
                    pct = current / total if total > 0 else 0
                    text = (f"{label} — {current}/{total} auctions"
                            if label else f"{current}/{total}")
                fetch_progress.progress(min(pct, 1.0), text=text)

            df = run_async(
                scraper.fetch_lots_for_selected(
                    selected_candidates, progress_callback=fetch_prog
                )
            )
            fetch_progress.empty()

            st.session_state.phase1_leads = df

            # Grow the known-category list for the sidebar filter
            if not df.empty and 'category' in df.columns:
                seen = {c for c in df['category'].dropna().astype(str).tolist() if c}
                st.session_state.known_categories = sorted(
                    set(st.session_state.known_categories) | seen
                )

            diag = dict(df.attrs)  # copy BEFORE we mutate / rerun
            raw_count = diag.get('raw_count', 0)
            filtered_cat = diag.get('filtered_by_category', 0)
            filtered_status = diag.get('filtered_by_status', 0)
            per_auction = diag.get('per_auction', []) or []
            errors = diag.get('errors', []) or []
            status_values = diag.get('status_values_seen', {}) or {}

            # Show the diagnostic breakdown inside the status box — one place
            # to understand exactly where the user's items went.
            st.write(
                f"**Raw lots from HiBid:** {raw_count} · "
                f"**Kept:** {len(df)} · "
                f"**Dropped by status (CLOSED / Bidding Closed):** {filtered_status} · "
                f"**Dropped by category filter:** {filtered_cat}"
            )

            if status_values:
                st.write(f"**lotState.status values seen:** `{status_values}`")

            if per_auction:
                with st.expander(f"Per-auction breakdown ({len(per_auction)})", expanded=(len(df) == 0)):
                    st.table(pd.DataFrame(per_auction))

            if errors:
                st.warning(f"**{len(errors)} auction(s) errored during fetch** — see details:")
                st.table(pd.DataFrame(errors))

            local_count = int((df['source'] == "Local Pickup").sum()) if not df.empty and 'source' in df.columns else 0
            ship_count = int((df['source'] == "Ship").sum()) if not df.empty and 'source' in df.columns else 0
            auction_count = df['auction'].nunique() if not df.empty else 0
            hard_count = int((df['logistics_ease'] == "HARD").sum()) if not df.empty and 'logistics_ease' in df.columns else 0
            easy_count = int((df['logistics_ease'] == "EASY").sum()) if not df.empty and 'logistics_ease' in df.columns else 0

            if df.empty:
                # Be honest about which of the three possible causes it was.
                if raw_count == 0:
                    if errors:
                        reason = (
                            f"HiBid returned **0 lots** for every selected auction. "
                            f"{len(errors)} auction(s) threw errors above — that's the likely cause."
                        )
                    else:
                        reason = (
                            "HiBid returned **0 lots** for every selected auction. "
                            "The GraphQL call succeeded but the response contained no lots. "
                            "This can happen for auctions in a pre-opening preview state, "
                            "or if HiBid's schema changed."
                        )
                elif filtered_status == raw_count:
                    reason = (
                        f"All {raw_count} lots were dropped as CLOSED / 'Bidding Closed'. "
                        f"Status values seen: `{status_values}`."
                    )
                elif filtered_cat == raw_count:
                    reason = f"All {raw_count} lots were excluded by the sidebar category filter."
                else:
                    reason = f"{raw_count} raw lots, all dropped by a mix of status + category filters."
                fetch_result_msg = f"⚠️ Scan complete — 0 items survived. {reason}"
                status_box.update(
                    label="⚠️ No open lots matched",
                    state="error", expanded=True,
                )
            else:
                breakdown_bits = [f"📦 {easy_count} easy-ship"]
                if hard_count:
                    breakdown_bits.append(f"🏋️ {hard_count} HARD (hidden by default)")
                fetch_result_msg = (
                    f"✅ Scanned {len(df)} items across {auction_count} auction(s)"
                    f" — {local_count} local, {ship_count} shippable"
                    f" · {' · '.join(breakdown_bits)}."
                )
                status_box.update(
                    label=f"✅ {len(df)} items from {auction_count} auction(s)",
                    state="complete", expanded=False,
                )
        except Exception as e:
            import traceback
            fetch_error = f"{type(e).__name__}: {e}"
            st.session_state._last_fetch_traceback = traceback.format_exc()
            st.error(f"❌ {fetch_error}")
            st.code(traceback.format_exc(), language="python")
            status_box.update(label="❌ Fetch failed", state="error", expanded=True)
        finally:
            st.session_state.fetch_lots_running = False

    # Persist full diagnostics so the next rerun can re-render them —
    # otherwise everything written inside the status box above is wiped.
    try:
        _diag_payload = {
            "raw_count": int(raw_count),
            "kept": int(len(df)),
            "filtered_status": int(filtered_status),
            "filtered_cat": int(filtered_cat),
            "per_auction": per_auction,
            "errors": errors,
            "status_values": status_values,
        }
    except NameError:
        # We errored before computing diagnostics (exception path)
        _diag_payload = None

    st.session_state._fetch_status = {
        "error": fetch_error,
        "msg": fetch_result_msg,
        "diag": _diag_payload,
    }
    st.rerun()


# --- Shared column config for discovery tables ---
DISCOVERY_COL_CONFIG = {
    "lot_link": st.column_config.LinkColumn("Item", display_text="Open"),
    "current_bid": st.column_config.NumberColumn("Bid", format="$%.2f"),
    "bid_count": st.column_config.NumberColumn("Bids", format="%d"),
    "est_cost": st.column_config.NumberColumn("Est. Cost", format="$%.2f"),
    "logistics_ease": st.column_config.TextColumn("Logistics"),
    "time_left": st.column_config.TextColumn("Time Left"),
}
DISCOVERY_COL_ORDER = ["title", "current_bid", "est_cost", "bid_count",
                       "time_left", "lot_link", "category", "logistics_ease"]


def _load_auction_for_analysis(auction_name, auction_df):
    """Replace the current analysis target with the given auction's items.

    If a fresh cached analysis exists for this auction, overlay its audit
    verdicts and price comps onto the fresh Phase 1 data so the user sees
    results immediately (with current bids, recomputed ROI).
    """
    st.session_state.selected_leads = auction_df.copy()
    st.session_state.current_auction = auction_name
    st.session_state.audit_results = {}

    # Consult disk cache, keyed by auction_id (pulled from auction_link or a
    # dedicated column if present)
    auction_id = _extract_auction_id(auction_df)
    if auction_id is None:
        return

    payload = _AUCTION_CACHE.load(auction_id)
    if not payload:
        return
    if not _AUCTION_CACHE.is_fresh(payload, ttl_days=st.session_state.cache_ttl_days):
        return

    merged = merge_cached_analysis(auction_df, payload)
    # Only treat as full audit_results if it actually has verdicts
    if 'verdict' in merged.columns and merged['verdict'].notna().any():
        st.session_state.selected_leads = merged
        st.session_state.audit_results = merged


def _extract_auction_id(auction_df: pd.DataFrame):
    """Pull the HiBid auction_id from the DataFrame.

    Discovery rows store it as a URL like https://hibid.com/auction/12345 in
    the 'auction_link' column. We parse the trailing int.
    """
    if 'auction_link' not in auction_df.columns or auction_df.empty:
        return None
    link = str(auction_df['auction_link'].iloc[0])
    if '/auction/' not in link:
        return None
    try:
        return int(link.rsplit('/auction/', 1)[1].split('/')[0].split('?')[0])
    except (ValueError, IndexError):
        return None


def _save_current_auction_to_cache():
    """Persist the current audit_results DataFrame to the disk cache."""
    ar = st.session_state.get('audit_results')
    if not isinstance(ar, pd.DataFrame) or ar.empty:
        return
    auction_id = _extract_auction_id(ar)
    if auction_id is None:
        return
    auction_name = st.session_state.get('current_auction') or ""
    closing_date = ""
    if 'closing_date' in ar.columns and not ar.empty:
        closing_date = str(ar['closing_date'].iloc[0])
    try:
        _AUCTION_CACHE.save(auction_id, auction_name, ar, closing_date)
    except Exception as e:
        # Don't crash the app over a cache write failure
        st.warning(f"Could not save analysis to cache: {e}")


def _render_auction_card(auction_name, auction_df):
    """Render one auction's expander with a single 'Analyze' button.

    Button click triggers a full app rerun so the analysis view renders in
    the main content area (not inside this expander).
    """
    closing = auction_df['closing_date'].iloc[0] if 'closing_date' in auction_df.columns else ""
    source = auction_df['source'].iloc[0] if 'source' in auction_df.columns else ""
    item_count = len(auction_df)
    avg_bid = auction_df['current_bid'].mean() if 'current_bid' in auction_df.columns else 0
    easy_count = (auction_df['logistics_ease'] == "EASY").sum() if 'logistics_ease' in auction_df.columns else 0

    # Cache hit indicator
    auction_id = _extract_auction_id(auction_df)
    cache_prefix = ""
    if auction_id is not None:
        payload = _AUCTION_CACHE.load(auction_id)
        if payload and _AUCTION_CACHE.is_fresh(payload, ttl_days=st.session_state.cache_ttl_days):
            cache_prefix = "💾 "

    # Easy-ship count goes FIRST in the label and is bolded so it's the
    # easiest thing to scan when picking which auction to dive into.
    if easy_count:
        easy_badge = f"📦 **{easy_count} easy-ship**"
    else:
        easy_badge = "📦 0 easy-ship"

    subtitle_parts = [f"{item_count} items"]
    if closing:
        subtitle_parts.append(f"closes {closing}")
    if source:
        subtitle_parts.append(source)
    subtitle_parts.append(f"avg bid ${avg_bid:.2f}")

    label = (
        f"{cache_prefix}{easy_badge}  ·  🏷️ {auction_name}  —  "
        + "  ·  ".join(subtitle_parts)
    )
    with st.expander(label, expanded=False):
        if cache_prefix:
            st.caption("💾 Previously analyzed — cached audit + price comps will load instantly on Analyze.")
        if st.button(
            f"🎯 Analyze This Auction ({item_count} items)",
            key=f"load_{auction_name}",
            type="primary",
            use_container_width=True,
        ):
            _load_auction_for_analysis(auction_name, auction_df)
            st.rerun()

        st.dataframe(
            auction_df,
            use_container_width=True,
            key=f"table_{auction_name}",
            column_config=DISCOVERY_COL_CONFIG,
            column_order=DISCOVERY_COL_ORDER,
            hide_index=True,
        )


@st.cache_resource(show_spinner=False)
def _get_auditor(model_name: str):
    """Load (and cache) the Phase2Scraper with the named NLP model.

    Streamlit keys the cache on args, so switching model_name swaps which
    cached instance is returned. First call per model: download + load
    (~30s-2min). Subsequent calls: instant.
    """
    from scraper import Phase2Scraper
    return Phase2Scraper(model_name=model_name)


def _run_ai_audit(leads_df):
    """Run Phase 2 AI condition audit with detailed phase-by-phase status."""
    from scraper import Phase2Scraper

    total = len(leads_df)

    # Default model is bart-large-mnli (already cached on most dev machines).
    # Opt-in to the faster distilbart model via "Fast mode" checkbox — that
    # one triggers a ~500MB download the first time it's used.
    fast_mode = st.session_state.get('audit_fast_mode', False)
    model_name = (
        Phase2Scraper.DEFAULT_MODEL_FAST if fast_mode
        else Phase2Scraper.DEFAULT_MODEL
    )
    batch_size = int(st.session_state.get('audit_batch_size', 8))

    with st.status("🧠 Running AI Condition Audit…", expanded=True) as status:
        # Phase 1: model load
        size_note = "~500MB" if fast_mode else "~1.6GB"
        speed_note = "fast mode, ~3x speedup" if fast_mode else "accuracy mode"
        st.write(
            f"**📥 Step 1/3 — Loading NLP model** "
            f"(`{model_name}`, {size_note}, {speed_note}). "
            "Downloads on first run, cached after."
        )
        if fast_mode:
            st.caption(
                "⚠️ First time running Fast mode? Expect a one-time ~500MB "
                "download. The UI won't update during the download — be "
                "patient for 1–3 minutes on a decent connection."
            )
        auditor = _get_auditor(model_name)
        st.write("✅ Model ready.")

        # Pre-count how many lots will be pre-filtered (HARD logistics) so
        # the user sees the savings up front.
        hard_preview = 0
        if 'logistics_ease' in leads_df.columns:
            hard_preview = int((leads_df['logistics_ease'] == 'HARD').sum())

        # Phase 2: title enrichment + condition classification
        if hard_preview > 0:
            will_classify = total - hard_preview
            st.write(
                f"**🔍 Step 2/3 — Classifying {will_classify} items** "
                f"(batched {batch_size} at a time). "
                f"⏭️ Skipping **{hard_preview}** HARD-logistics lots "
                "(mattresses, vehicles, furniture, real estate, etc.) — "
                "auto-flagged as Unshippable, no AI needed."
            )
        else:
            st.write(
                f"**🔍 Step 2/3 — Enriching titles and classifying condition** "
                f"for {total} items (batched {batch_size} at a time)."
            )
        st.caption(
            "Each item: pull model numbers / brands from the description into "
            "the title, then run zero-shot classification across "
            "*mint / normal-wear / untested / broken*."
        )
        progress_bar = st.progress(0, text=f"Starting — 0/{total}")
        current_item_placeholder = st.empty()

        def ai_progress(current, total_items):
            pct = current / total_items if total_items > 0 else 1.0
            # With batching we don't know "which single item just finished" —
            # show the most recently processed row instead.
            try:
                preview_idx = min(max(current - 1, 0), total_items - 1)
                row = leads_df.iloc[preview_idx]
                title_preview = str(row.get('title', ''))[:70]
            except Exception:
                title_preview = ""
            progress_bar.progress(
                min(pct, 1.0),
                text=f"Analyzing {current}/{total_items}…",
            )
            if title_preview:
                current_item_placeholder.caption(
                    f"🔎 Last batch ended near: *{title_preview}*"
                )

        results_df = auditor.batch_audit(
            leads_df,
            progress_callback=ai_progress,
            batch_size=batch_size,
        )

        # Phase 3: summarize
        good = int((~results_df['red_flag']).sum()) if 'red_flag' in results_df.columns else 0
        flagged = int(results_df['red_flag'].sum()) if 'red_flag' in results_df.columns else 0
        skipped_hard = int(results_df.attrs.get('audit_skipped_hard', 0) or 0)
        skipped_empty = int(results_df.attrs.get('audit_skipped_empty', 0) or 0)
        classified = int(results_df.attrs.get('audit_classified', 0) or 0)

        summary_parts = [f"✅ {good} good-condition", f"⚠️ {flagged} red-flagged"]
        if skipped_hard > 0:
            summary_parts.append(f"🚚 {skipped_hard} HARD-logistics (pre-filtered)")
        if skipped_empty > 0:
            summary_parts.append(f"❓ {skipped_empty} empty description")
        st.write(f"**📊 Step 3/3 — Summary:** " + " · ".join(summary_parts))
        if skipped_hard > 0 and classified > 0:
            saved_est = skipped_hard * 0.2  # ~200ms per classification on CPU
            st.caption(
                f"💨 Skipped AI on {skipped_hard} obviously-unshippable lots "
                f"(~{saved_est:.0f}s saved). Adjust the HARD-logistics "
                "ship_killers regex in `config.json` if items are being "
                "wrongly pre-filtered."
            )
        status.update(label="✅ AI audit complete", state="complete", expanded=False)

    return results_df


def _run_image_enrichment(audit_df, min_bid: float = 5.0):
    """Run eBay image_search-based title enrichment on promising lots.

    Gated to skip:
      - red-flagged lots (condition audit says broken/untested)
      - HARD logistics (we're not buying furniture to ship)
      - lots below min_bid (junk filter — don't burn API calls on $1 items)
      - lots with no thumbnail_url

    Returns the DataFrame with six new img_* columns plus (where confidence
    is high enough) a promoted `enriched_title` that now carries brand +
    model + year pulled straight from matching eBay listings.
    """
    from scraper.vision_enrich import EbayImageEnricher, promote_image_titles
    from scraper.config_loader import load_config

    cfg = load_config()
    enricher = EbayImageEnricher(
        cfg["ebay"]["app_id"], cfg["ebay"]["cert_id"],
        hibid_user_agent=cfg.get("api", {}).get(
            "user_agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"),
    )

    total = len(audit_df)

    def gate(row):
        if row.get('red_flag'):
            return False
        if row.get('logistics_ease') == 'HARD':
            return False
        if not (row.get('thumbnail_url') or ''):
            return False
        try:
            if float(row.get('current_bid') or 0) < min_bid:
                return False
        except (ValueError, TypeError):
            return False
        return True

    # Pre-count how many lots will actually be analyzed vs skipped, so the
    # progress bar can reflect real work (not padded by skipped rows).
    eligible_ids = {
        row.get('lot_id')
        for _, row in audit_df.iterrows() if gate(row)
    }
    eligible_count = len(eligible_ids)

    with st.status(
        f"🖼️ Enriching {eligible_count} titles via eBay image_search…",
        expanded=True,
    ) as status:
        st.write(
            f"**Gated to {eligible_count} of {total} items** — skipping "
            "red-flagged, HARD-logistics, missing-image, and sub-${:.2f} lots."
            .format(min_bid)
        )
        st.caption(
            "For each eligible item: download the HiBid thumbnail, POST it "
            "to eBay's Browse `search_by_image` endpoint, and if the top "
            "hits agree on a product, rewrite the title to match what eBay "
            "actually sells it as. Zero API cost — reuses your existing "
            "Browse API credentials."
        )

        progress_bar = st.progress(0, text=f"Starting — 0/{eligible_count}")
        current_item_placeholder = st.empty()
        progress_state = {"done": 0, "hits": 0}

        def img_progress(current, tot_with_skips, label):
            # `current` counts every row including gated skips. We only
            # want to advance the bar on rows we actually analyzed — so
            # we track that ourselves and derive progress from it.
            lot_row = audit_df.iloc[current - 1] if current - 1 < len(audit_df) else None
            if lot_row is None:
                return
            if gate(lot_row):
                progress_state["done"] += 1
                pct = (progress_state["done"] / eligible_count
                       if eligible_count else 1.0)
                progress_bar.progress(
                    min(pct, 1.0),
                    text=f"Identifying {progress_state['done']}/{eligible_count}…",
                )
                if label and label != "gated":
                    current_item_placeholder.caption(f"🔎 Matched: *{label[:70]}*")

        result_df = enricher.batch_enrich(
            audit_df, gate_fn=gate, progress_callback=img_progress,
        )

        # Promote high-confidence matches to `enriched_title`
        promoted_df = promote_image_titles(
            result_df, min_confidence=0.5, min_hits=3,
        )
        promoted = int((promoted_df['enriched_title']
                        != promoted_df.get('enriched_title_pre_image',
                                           promoted_df['enriched_title'])).sum())
        skipped_gate = int(
            (promoted_df['img_error'] == 'skipped_gate').sum()
        ) if 'img_error' in promoted_df.columns else 0
        errored = int(
            promoted_df['img_error'].notna().sum()
            - (promoted_df['img_error'] == 'skipped_gate').sum()
        ) if 'img_error' in promoted_df.columns else 0

        st.write(
            f"**📊 Summary:** "
            f"✅ {promoted} titles upgraded from image matches · "
            f"⏭️ {skipped_gate} gated-out · "
            f"⚠️ {errored} couldn't identify."
        )
        status.update(
            label=f"✅ Image enrichment — {promoted} titles upgraded",
            state="complete", expanded=False,
        )

    return promoted_df


def _apply_comps_filters(good_df):
    """Apply the Step 2 pre-comps filters to good_df.

    Returns (eligible_df, skipped_df, filter_summary) so the caller can
    run comps on just the eligible rows while preserving skipped rows
    (without resale data) in the final merged output.
    """
    df = good_df.copy()
    reasons = []

    # Min bid filter
    min_bid = float(st.session_state.get('comps_min_bid', 0) or 0)
    if min_bid > 0 and 'current_bid' in df.columns:
        before = len(df)
        df = df[df['current_bid'].fillna(0) >= min_bid]
        dropped = before - len(df)
        if dropped:
            reasons.append(f"{dropped} under ${min_bid:g} bid")

    # Exclude HARD logistics
    if st.session_state.get('comps_exclude_hard', True) and 'logistics_ease' in df.columns:
        before = len(df)
        df = df[df['logistics_ease'] != 'HARD']
        dropped = before - len(df)
        if dropped:
            reasons.append(f"{dropped} HARD-logistics")

    # Only image-promoted titles
    if st.session_state.get('comps_only_img_promoted', False):
        if 'img_enriched_title' in df.columns and 'enriched_title_pre_image' in df.columns:
            before = len(df)
            promoted = (
                df['img_enriched_title'].notna()
                & (df['enriched_title'].fillna('') != df['enriched_title_pre_image'].fillna(''))
            )
            df = df[promoted]
            dropped = before - len(df)
            if dropped:
                reasons.append(f"{dropped} not image-promoted")

    # Top-N by bid
    max_lots = int(st.session_state.get('comps_max_lots', 0) or 0)
    if max_lots > 0 and len(df) > max_lots and 'current_bid' in df.columns:
        dropped = len(df) - max_lots
        df = df.sort_values('current_bid', ascending=False).head(max_lots)
        reasons.append(f"trimmed to top {max_lots} by bid ({dropped} cut)")

    eligible_ids = set(df.index)
    skipped = good_df[~good_df.index.isin(eligible_ids)].copy()
    summary = " · ".join(reasons) if reasons else "all good+ items included"
    return df, skipped, summary


def _run_ebay_comps(results_df):
    """Run eBay + Mercari price comps on filtered good+ items in results_df.

    Applies the Step 2 pre-comps filters (min bid, logistics, top-N, etc.),
    then:
      1. Samples STR per auction (~3 lots each) — fast, replaces per-lot scrape.
      2. Runs price comps on every eligible lot with the auction-level STR.

    Max bid is NOT computed here — it's recomputed on every render so the
    Target ROI slider in the results section updates it live.
    """
    # Clear previous comp data so re-runs start fresh
    for col in ['est_resale', 'price_low', 'price_high', 'comp_count',
                'ebay_comps', 'mercari_comps', 'pricecharting_comps',
                'price_source', 'ebay_str', 'str_source',
                'est_roi', 'max_bid']:
        if col in results_df.columns:
            results_df = results_df.drop(columns=[col])

    good_df = results_df[~results_df['red_flag']].copy()
    flagged_df = results_df[results_df['red_flag']].copy()

    # Apply pre-comps filters — skipped rows come back with no resale data
    eligible_df, skipped_df, filter_summary = _apply_comps_filters(good_df)

    from scraper.ebay_prices import EbayPriceLookup
    from scraper.pricecharting import PriceChartingLookup
    from scraper.config_loader import load_config
    cfg = load_config()
    pc_token = (cfg.get("pricecharting") or {}).get("token") or None
    pc_client = PriceChartingLookup(pc_token)
    ebay = EbayPriceLookup(
        cfg["ebay"]["app_id"], cfg["ebay"]["cert_id"],
        pricecharting=pc_client,
    )

    total = len(eligible_df)

    with st.status("💰 Running Price Comps & STR…", expanded=True) as status:
        if total == 0:
            st.warning(
                "No items matched the pre-comps filters. Loosen the filters above and try again."
            )
            status.update(label="⚠️ Nothing to comp", state="error", expanded=True)
            combined = pd.concat([good_df, flagged_df], ignore_index=True)
            return combined, 0, 0

        st.write(
            f"**🎯 Comp target:** {total} lots "
            f"({len(skipped_df)} of {len(good_df)} good+ lots skipped by filters)."
        )
        st.caption(f"Filters applied — {filter_summary}")

        # ---------- STR: per-category sampling (if enabled) ----------
        auction_str_map = None
        use_auction_str = st.session_state.get('comps_use_auction_str', True)
        if use_auction_str and 'auction' in eligible_df.columns:
            # Count distinct (auction, category) buckets so the user knows
            # how many STR scrapes we're doing vs per-lot
            if 'category' in eligible_df.columns:
                buckets = eligible_df.groupby(
                    ['auction', eligible_df['category'].fillna('').replace('', '(uncategorized)')]
                ).ngroups
            else:
                buckets = eligible_df['auction'].nunique()
            st.write(
                f"**📈 Sampling STR across {buckets} category bucket(s)** "
                "(2 lots each — gives per-category variance without per-lot cost)."
            )
            str_progress = st.progress(0, text=f"Sampling STR — 0/{buckets}")

            def str_progress_cb(current, total_buckets):
                pct = current / total_buckets if total_buckets > 0 else 1.0
                str_progress.progress(
                    min(pct, 1.0),
                    text=f"Sampled STR for {current}/{total_buckets} bucket(s)",
                )

            auction_str_map = ebay.sample_auction_str(
                eligible_df, sample_size=2,
                progress_callback=str_progress_cb,
                granularity="category",
            )
            usable = sum(
                1 for k, v in auction_str_map.items()
                if k != "__granularity__" and v and v[0] is not None
            )
            st.caption(f"✓ STR resolved for {usable}/{buckets} bucket(s).")

        # ---------- Price comps ----------
        st.write(
            f"**🔗 Looking up eBay sold listings + Mercari sold listings** "
            f"for {total} lots."
        )
        st.caption(
            "Per lot: scrape recent sold prices from both marketplaces, "
            "apply IQR outlier filtering, pool into median / 25th / 75th percentile."
        )
        progress_bar = st.progress(0, text=f"Starting — 0/{total}")
        current_item_placeholder = st.empty()
        workers = int(st.session_state.get('comps_workers', 8))
        if workers > 1:
            st.caption(f"⚡ Running {workers} parallel workers.")

        def price_progress(current, total_items, title_preview=""):
            # Called from the main thread (as_completed drains on-thread).
            pct = current / total_items if total_items > 0 else 1.0
            progress_bar.progress(
                min(pct, 1.0),
                text=f"Priced {current}/{total_items}…",
            )
            if title_preview:
                current_item_placeholder.caption(
                    f"🔎 Just priced: *{title_preview}*"
                )

        comps_df = ebay.batch_lookup(
            eligible_df,
            progress_callback=price_progress,
            auction_str_map=auction_str_map,
            max_workers=workers,
        )

        # ROI
        comps_df['est_roi'] = None
        mask = comps_df['est_resale'].notna() & (comps_df['est_cost'] > 0)
        comps_df.loc[mask, 'est_roi'] = (
            (comps_df.loc[mask, 'est_resale'] - comps_df.loc[mask, 'est_cost'])
            / comps_df.loc[mask, 'est_cost'] * 100
        ).round(0)

        found = int(comps_df['est_resale'].notna().sum())
        st.write(f"**📊 Summary:** found price comps for {found}/{total} items.")
        status.update(label="✅ Price comps complete", state="complete", expanded=False)

    # Stitch comps_df (with resale) back together with skipped + flagged rows
    # so the results table still shows every lot, just with NaN resale for
    # skipped ones.
    combined = pd.concat([comps_df, skipped_df, flagged_df], ignore_index=True)
    return combined, found, total


def _compute_max_bid(df, target_roi_val):
    """Back out the max bid that still hits target_roi_val × cost.

    Returns a new DataFrame with 'max_bid' column set (or left as None where
    no est_resale is available).
    """
    from scraper.config_loader import load_config
    cfg = load_config()
    ebay_fee_pct = 0.1325
    ebay_fee_flat = 0.30
    buyer_premium_pct = cfg.get("shipping", {}).get("buyer_premium_pct", 15.0) / 100.0
    ship_cost = cfg.get("shipping", {}).get("bundled_ship_cost", 25.0)

    out = df.copy()
    out['max_bid'] = None
    if 'est_resale' not in out.columns:
        return out
    resale_mask = out['est_resale'].notna()
    if resale_mask.any():
        resale = out.loc[resale_mask, 'est_resale']
        net_resale = resale * (1 - ebay_fee_pct) - ebay_fee_flat
        item_ship = out.loc[resale_mask, 'source'].apply(
            lambda s: ship_cost if s == "Ship" else 0
        ) if 'source' in out.columns else 0
        max_bid = (net_resale / target_roi_val - item_ship) / (1 + buyer_premium_pct)
        out.loc[resale_mask, 'max_bid'] = max_bid.clip(lower=0).round(2)
    return out


def _render_results_table(results_df):
    """Render the results table with live ROI/STR threshold highlighting.

    Rather than hiding items below threshold, rows are color-coded:
      - green = meets BOTH target ROI and target STR
      - yellow = meets ONE of them
      - no tint = below both thresholds (or missing data)

    Sorted by est_roi descending by default.
    """
    # --- Threshold controls (live) ---
    st.markdown("#### 🎯 Profitability Targets")
    st.caption(
        "Adjust these knobs to see which items meet your goals. "
        "**Green** rows meet both targets, **yellow** meet one, **uncolored** miss both or lack data. "
        "The **Max Bid** column recomputes from the ROI target."
    )
    tc1, tc2 = st.columns(2)
    with tc1:
        target_roi_val = st.number_input(
            "Target ROI Multiplier",
            value=3.0, step=0.5, format="%.1f",
            min_value=1.0,
            help="Sell for Nx what you paid (3x = sell for 3× total cost). "
                 "Drives both the highlight and the Max Bid column.",
            key="target_roi_live",
        )
    with tc2:
        target_str_val = st.number_input(
            "Target eBay STR %",
            value=70.0, step=5.0, format="%.0f",
            min_value=0.0, max_value=100.0,
            help="Minimum sell-through rate on eBay. Higher = faster-selling items.",
            key="target_str_live",
        )

    # --- Recompute max_bid with current target (dynamic) ---
    working = _compute_max_bid(results_df, target_roi_val)

    # --- Sort by ROI descending ---
    if 'est_roi' in working.columns:
        working['_roi_sort'] = pd.to_numeric(working['est_roi'], errors='coerce')
        working = working.sort_values(
            '_roi_sort', ascending=False, na_position='last'
        ).drop(columns=['_roi_sort']).reset_index(drop=True)

    # --- Threshold masks (for highlight + metrics) ---
    roi_threshold = (target_roi_val - 1) * 100
    meets_roi = (
        working['est_roi'].notna() & (pd.to_numeric(working['est_roi'], errors='coerce') >= roi_threshold)
        if 'est_roi' in working.columns
        else pd.Series(False, index=working.index)
    )
    meets_str_mask = (
        working['ebay_str'].notna() & (pd.to_numeric(working['ebay_str'], errors='coerce') >= target_str_val)
        if 'ebay_str' in working.columns
        else pd.Series(False, index=working.index)
    )
    meets_both = meets_roi & meets_str_mask
    meets_either = (meets_roi | meets_str_mask) & ~meets_both

    # --- Metrics ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Leads", len(working))
    if 'est_resale' in working.columns:
        col2.metric("Comps", working['est_resale'].notna().sum())
    col3.metric("✅ Meets Both Targets", int(meets_both.sum()))

    col4, col5, col6 = st.columns(3)
    col4.metric("🟡 Meets One", int(meets_either.sum()))
    if 'ebay_str' in working.columns:
        has_str = working['ebay_str'].notna().sum()
        avg_str = pd.to_numeric(working['ebay_str'], errors='coerce').mean()
        col5.metric("Avg STR", f"{avg_str:.0f}%" if has_str > 0 else "N/A")
    if 'red_flag' in working.columns:
        col6.metric("Red Flags", int(working['red_flag'].sum()))

    filtered_df = working

    # Columns
    title_col = 'enriched_title' if 'enriched_title' in filtered_df.columns else 'title'
    # Lead with the lot thumbnail when we have one — Streamlit's ImageColumn
    # renders the URL inline so the user can scan visually before reading.
    display_cols = []
    if 'thumbnail_url' in filtered_df.columns:
        display_cols.append('thumbnail_url')
    display_cols += [title_col, 'lot_link', 'auction_link', 'category', 'current_bid', 'est_cost']
    col_config = {
        "thumbnail_url": st.column_config.ImageColumn(
            "📷",
            help="Lot photo from HiBid (click to enlarge).",
            width="small",
        ),
        "enriched_title": st.column_config.TextColumn("Title (Enriched)"),
        "lot_link": st.column_config.LinkColumn("Item", display_text="Open"),
        "auction_link": st.column_config.LinkColumn("Auction", display_text="Open"),
        "current_bid": st.column_config.NumberColumn("Current Bid", format="$%.2f"),
        "est_cost": st.column_config.NumberColumn("Est. Cost", format="$%.2f"),
        "bid_count": st.column_config.NumberColumn("Bids", format="%d"),
    }

    if 'est_resale' in filtered_df.columns:
        display_cols += ['est_resale']
        col_config["est_resale"] = st.column_config.NumberColumn("Est. Resale (median)", format="$%.2f")

        if 'price_low' in filtered_df.columns:
            display_cols += ['price_low', 'price_high']
            col_config["price_low"] = st.column_config.NumberColumn("Low (25%)", format="$%.2f")
            col_config["price_high"] = st.column_config.NumberColumn("High (75%)", format="$%.2f")

        if 'ebay_comps' in filtered_df.columns:
            display_cols += ['ebay_comps']
            col_config["ebay_comps"] = st.column_config.NumberColumn("eBay Comps", format="%d")

            # Low-confidence flag: with fewer than 4 sold comps, the IQR
            # outlier filter in ebay_prices._filter_outliers can't run, so a
            # single pricey listing can drag the median resale way up.
            # PriceCharting hits are pre-aggregated \u2014 never low-confidence.
            filtered_df = filtered_df.copy()
            pc_hits = (
                filtered_df['pricecharting_comps'].fillna(0).astype(int).gt(0)
                if 'pricecharting_comps' in filtered_df.columns
                else False
            )
            filtered_df['low_comp_confidence'] = (
                filtered_df['ebay_comps'].fillna(0).astype(int).lt(4)
                & filtered_df['est_resale'].notna()
                & ~pc_hits
            )
            display_cols += ['low_comp_confidence']
            col_config["low_comp_confidence"] = st.column_config.CheckboxColumn(
                "Low Conf.",
                help=(
                    "Checked when fewer than 4 eBay sold comps were found "
                    "(and the lot didn't get a PriceCharting hit). The "
                    "outlier filter needs \u22654 data points, so with 1\u20133 "
                    "comps the est_resale can be skewed by a single "
                    "high-priced listing."
                ),
            )
        if 'mercari_comps' in filtered_df.columns:
            display_cols += ['mercari_comps']
            col_config["mercari_comps"] = st.column_config.NumberColumn("Mercari Comps", format="%d")
        if 'pricecharting_comps' in filtered_df.columns:
            display_cols += ['pricecharting_comps']
            col_config["pricecharting_comps"] = st.column_config.NumberColumn(
                "PC Hit", format="%d",
                help="1 = PriceCharting matched this lot to a canonical product.",
            )
        if 'price_source' in filtered_df.columns:
            display_cols += ['price_source']
            col_config["price_source"] = st.column_config.TextColumn("Price Src")

        display_cols += ['est_roi']
        col_config["est_roi"] = st.column_config.NumberColumn("ROI %", format="%.0f%%")

    if 'max_bid' in filtered_df.columns:
        display_cols.append('max_bid')
        col_config["max_bid"] = st.column_config.NumberColumn("Max Bid", format="$%.2f")

    if 'ebay_str' in filtered_df.columns:
        display_cols.append('ebay_str')
        col_config["ebay_str"] = st.column_config.ProgressColumn("STR %", min_value=0, max_value=100, format="%.0f%%")

    display_cols.append('bid_count')

    if 'verdict' in filtered_df.columns:
        display_cols += ['verdict', 'confidence', 'red_flag']
        col_config["confidence"] = st.column_config.ProgressColumn("Confidence", min_value=0, max_value=100, format="%.1f%%")
        col_config["red_flag"] = st.column_config.CheckboxColumn("Red Flag")

    final_cols = [c for c in display_cols if c in filtered_df.columns]
    display_df = filtered_df[final_cols].copy()

    # Row-level highlighting based on the threshold masks computed above.
    # Re-index the masks to match display_df (preserve filter_df's row order).
    local_meets_both = meets_both.reindex(filtered_df.index).fillna(False).reset_index(drop=True)
    local_meets_either = meets_either.reindex(filtered_df.index).fillna(False).reset_index(drop=True)
    display_df = display_df.reset_index(drop=True)

    def _row_style(row):
        i = row.name
        if local_meets_both.iloc[i]:
            return ['background-color: rgba(46, 204, 113, 0.28)'] * len(row)
        if local_meets_either.iloc[i]:
            return ['background-color: rgba(241, 196, 15, 0.22)'] * len(row)
        return [''] * len(row)

    styled = display_df.style.apply(_row_style, axis=1)

    st.dataframe(
        styled,
        use_container_width=True,
        column_config=col_config,
    )


# ================================================================
# SINGLE-PAGE UI: switches between Discovery view and Analysis view
# ================================================================
current_auction = st.session_state.get('current_auction')

# ---- ANALYSIS VIEW: one auction is loaded ----
if current_auction and not st.session_state.selected_leads.empty:
    leads_df = st.session_state.selected_leads

    # Back button + header
    bc1, bc2 = st.columns([1, 4])
    with bc1:
        if st.button("← Back to auctions", use_container_width=True):
            st.session_state.selected_leads = pd.DataFrame()
            st.session_state.current_auction = None
            st.session_state.audit_results = {}
            st.rerun()
    with bc2:
        st.subheader(f"🔬 {current_auction}")
        caption_bits = [f"{len(leads_df)} items loaded"]

        # If the current analysis came from cache, indicate that + when.
        auction_id = _extract_auction_id(leads_df)
        if auction_id is not None:
            payload = _AUCTION_CACHE.load(auction_id)
            if payload and _AUCTION_CACHE.is_fresh(
                payload, ttl_days=st.session_state.cache_ttl_days
            ):
                try:
                    cached_at = datetime.fromisoformat(payload.get('cached_at', ''))
                    age = datetime.now() - cached_at
                    age_str = (
                        f"{age.days}d ago" if age.days > 0
                        else f"{int(age.seconds / 3600)}h ago"
                    )
                except Exception:
                    age_str = "earlier"
                caption_bits.append(f"💾 cached analysis from {age_str} · current bids refreshed")
        st.caption(" · ".join(caption_bits))

    has_audit = (
        isinstance(st.session_state.get('audit_results'), pd.DataFrame)
        and not st.session_state.audit_results.empty
        and 'verdict' in st.session_state.audit_results.columns
    )

    # Step 1: AI audit
    st.markdown("---")
    st.markdown("### Step 1: AI Condition Audit")

    if has_audit:
        ar = st.session_state.audit_results
        good_count = (~ar['red_flag']).sum()
        flagged_count = ar['red_flag'].sum()
        st.success(f"Audit complete — **{good_count} good+** condition, {flagged_count} red-flagged")

    audit_running = st.session_state.get('audit_running', False)
    comps_running = st.session_state.get('comps_running', False)
    audit_btn_label = "⏳ Running audit…" if audit_running else "🧠 Run AI Condition Audit"

    # ---- Audit speed/accuracy knobs ----
    with st.expander("⚙️ Audit speed vs accuracy (optional)", expanded=False):
        ac_col1, ac_col2 = st.columns(2)
        with ac_col1:
            st.checkbox(
                "🚀 Fast mode (smaller model)",
                key="audit_fast_mode",
                help="Use distilbart-mnli-12-3 (~500MB, ~3x faster) instead "
                     "of bart-large-mnli. First run triggers a one-time "
                     "~500MB download — the UI may appear frozen for 1–3 "
                     "minutes during that download. Leave off to use the "
                     "already-cached bart-large-mnli.",
            )
        with ac_col2:
            st.slider(
                "Batch size (per forward pass)",
                min_value=1, max_value=32, step=1,
                key="audit_batch_size",
                help="Descriptions classified in one forward pass. Bigger = "
                     "faster but more RAM. Start at 8; drop to 4 if OOM, "
                     "push to 16-32 on a beefy box. Batching alone gives a "
                     "~2-3x speedup over the old serial code.",
            )

    if st.button(
        audit_btn_label,
        type="primary",
        use_container_width=True,
        disabled=audit_running or comps_running,
        key="run_audit_btn",
    ):
        st.session_state.audit_running = True
        st.rerun()

    # If the flag is set, we're on the second rerun — do the actual work
    # with the button now rendered disabled above. Clear the flag when done.
    if audit_running:
        _keep_screen_awake()
        st.info(
            "🔋 Keeping screen awake while this runs. "
            "If your phone still locks, set **Auto-Lock → Never** in your phone's "
            "display settings before kicking off long runs."
        )
        try:
            st.session_state.audit_results = _run_ai_audit(leads_df)
            _save_current_auction_to_cache()
        except Exception as e:
            st.error(f"Audit failed: {e}")
        finally:
            st.session_state.audit_running = False
        st.rerun()

    # Step 1.5: Image-based title enrichment (optional; improves Step 2 quality)
    img_enrich_running = st.session_state.get('img_enrich_running', False)
    st.markdown("---")
    st.markdown("### Step 1.5: Upgrade Titles via eBay Image Match  ·  *optional*")

    if not has_audit:
        st.info(
            "Run the AI audit first — image enrichment only runs on **good+** lots "
            "that pass the condition filter."
        )
    else:
        ar = st.session_state.audit_results
        # Count what would actually be analyzed so the user knows the scope
        if 'thumbnail_url' in ar.columns:
            with_thumbs = int(ar['thumbnail_url'].fillna('').astype(bool).sum())
        else:
            with_thumbs = 0

        img_upgraded = 0
        if 'img_enriched_title' in ar.columns:
            img_upgraded = int(ar['img_enriched_title'].notna().sum())

        caption_bits = [f"🖼️ {with_thumbs} items have thumbnails"]
        if img_upgraded:
            caption_bits.append(f"✨ {img_upgraded} already upgraded")
        st.caption(" · ".join(caption_bits))

        st.caption(
            "Downloads each lot's first thumbnail, runs it through eBay's "
            "`search_by_image`, and rewrites the title with brand / model / "
            "year pulled from matching listings. **Zero-cost** — reuses your "
            "Browse API credentials. Skips red-flagged, HARD-to-ship, "
            "and low-bid lots by default."
        )

        c1, c2 = st.columns([2, 1])
        with c2:
            min_bid = st.number_input(
                "Skip lots below bid $",
                min_value=0.0, max_value=500.0, step=1.0,
                value=float(st.session_state.img_enrich_min_bid),
                key="img_enrich_min_bid_input",
                help="Junk filter. Don't burn cycles identifying $1 lots.",
            )
            st.session_state.img_enrich_min_bid = float(min_bid)

        img_btn_label = ("⏳ Identifying items…" if img_enrich_running
                         else "🖼️ Upgrade Titles from Images")
        if c1.button(
            img_btn_label,
            type="secondary",
            use_container_width=True,
            disabled=audit_running or comps_running or img_enrich_running,
            key="run_img_enrich_btn",
        ):
            st.session_state.img_enrich_running = True
            st.rerun()

        if img_enrich_running:
            _keep_screen_awake()
            try:
                st.session_state.audit_results = _run_image_enrichment(
                    st.session_state.audit_results,
                    min_bid=st.session_state.img_enrich_min_bid,
                )
                _save_current_auction_to_cache()
            except Exception as e:
                import traceback
                st.error(f"Image enrichment failed: {e}")
                st.code(traceback.format_exc(), language="python")
            finally:
                st.session_state.img_enrich_running = False
            st.rerun()

    # Step 2: eBay + Mercari comps (only after audit)
    st.markdown("---")
    st.markdown("### Step 2: eBay + Mercari Price Comps & STR")

    if not has_audit:
        st.info("Run the AI audit first — price lookups only run on **good+ condition** items.")
    else:
        ar = st.session_state.audit_results
        good_df = ar[~ar['red_flag']]
        flagged_df = ar[ar['red_flag']]
        st.caption(f"💰 {len(good_df)} good+ items eligible for lookup ({len(flagged_df)} red-flagged skipped)")

        # ---- Pre-comps filter panel ----
        # Comps are ~3s/lot; a 1000-lot auction takes ~50 min unfiltered.
        # Let the user trim the target set before launch.
        with st.expander("⚙️ Narrow down what to comp (optional — big speedup on large auctions)",
                         expanded=(len(good_df) >= 300)):
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                st.number_input(
                    "Minimum current bid ($)",
                    min_value=0.0, max_value=500.0, step=1.0,
                    key="comps_min_bid",
                    help="Skip lots with bids below this — cheap bids usually = junk.",
                )
                st.number_input(
                    "Cap total lots to comp (0 = no cap)",
                    min_value=0, max_value=5000, step=50,
                    key="comps_max_lots",
                    help="Keep only the top-N by current bid. Useful for massive auctions.",
                )
            with f_col2:
                st.checkbox(
                    "Exclude HARD logistics lots",
                    key="comps_exclude_hard",
                    help="Skip items flagged as hard to ship/pick up.",
                )
                st.checkbox(
                    "Only image-promoted titles",
                    key="comps_only_img_promoted",
                    help="Only comp lots whose title was upgraded in Step 1.5 "
                         "(highest-confidence product matches). Leave off unless "
                         "you've already run image enrichment.",
                )
                st.checkbox(
                    "Fast STR (sample 3 lots per auction)",
                    key="comps_use_auction_str",
                    help="STR is a marketplace signal, not per-lot. Sampling "
                         "replaces ~1000 scrapes with ~15 on a typical run. "
                         "Big time saver — leave on.",
                )
                st.slider(
                    "Parallel workers",
                    min_value=1, max_value=16, step=1,
                    key="comps_workers",
                    help="Thread pool size for price lookups. Default 8 — "
                         "roughly 8x faster than serial. Drop to 1 if you "
                         "suspect rate-limiting; push to 12-16 on a fast "
                         "connection.",
                )

            # Live preview of how many lots will actually be comped
            try:
                preview_df, preview_skipped, preview_summary = _apply_comps_filters(good_df)
                st.caption(
                    f"**🎯 Will comp {len(preview_df)} lots** "
                    f"(of {len(good_df)} good+) · {preview_summary}"
                )
            except Exception:
                pass

        comps_btn_label = "⏳ Running price comps…" if comps_running else "💰 Run Price Comps on Filtered Items"
        if st.button(
            comps_btn_label,
            type="primary",
            use_container_width=True,
            disabled=audit_running or comps_running or img_enrich_running,
            key="run_comps_btn",
        ):
            st.session_state.comps_running = True
            st.rerun()

        if comps_running:
            _keep_screen_awake()
            st.info(
                "🔋 Keeping screen awake while this runs. "
                "If your phone still locks, set **Auto-Lock → Never** in your phone's "
                "display settings before kicking off long runs."
            )
            try:
                combined, found, total = _run_ebay_comps(ar)
                st.session_state.audit_results = combined
                _save_current_auction_to_cache()
                st.success(f"Found price comps for {found}/{total} good+ leads.")
            except Exception as e:
                st.error(f"Price comps failed: {e}")
            finally:
                st.session_state.comps_running = False
            st.rerun()

    # Results table (if anything exists)
    if (
        isinstance(st.session_state.get('audit_results'), pd.DataFrame)
        and not st.session_state.audit_results.empty
    ):
        st.markdown("---")
        st.markdown("### Results")
        _render_results_table(st.session_state.audit_results)

# ---- SELECTION VIEW: candidates loaded, user picking which to deep-scan ----
elif st.session_state.get('auction_candidates') and st.session_state.phase1_leads.empty:
    candidates = st.session_state.auction_candidates
    cat_samples = st.session_state.get('category_samples', {})

    st.subheader(f"📋 Step 2: Pick which auctions to deep-scan")
    st.caption(
        f"Found **{len(candidates)}** open auctions matching your filters. "
        "Check the ones worth a full item-level scan, then click **📥 Fetch items** "
        "to pull every lot, current bids, and descriptions for those auctions only. "
        "Big auctions (1000+ items) take noticeably longer — pick selectively."
    )

    # --- Build the picker DataFrame ---
    rows = []
    for c in candidates:
        aid = c['auction_id']
        raw_sample = cat_samples.get(aid)
        # Back-compat: old session state stored a plain List[str]. Normalize.
        if isinstance(raw_sample, list):
            sample_payload = {
                "categories": raw_sample, "cat_counts": {}, "titles": [],
            }
        elif isinstance(raw_sample, dict):
            sample_payload = raw_sample
        else:
            sample_payload = None

        cats = (sample_payload or {}).get("categories") or []
        cat_preview = ", ".join(cats[:6]) + (f" (+{len(cats) - 6})" if len(cats) > 6 else "")

        # Auto-generated blurb: "450 lots · Mostly Tools (40%), Kitchen (25%) ·
        # Examples: Craftsman drill press, KitchenAid mixer, Oak dining table"
        if sample_payload is not None:
            summary = Phase1Scraper.generate_auction_summary(c, sample_payload)
        else:
            summary = "(sample categories to see a preview)"
        if not summary:
            summary = "—"

        # HiBid's eventDateEnd is date-only (always 00:00:00 — useless for the
        # actual closing time). The closing time, when known, lives as free
        # text in eventDateInfo (e.g. "Bidding closing Monday, April 27 at
        # 7:00 PM CST" or "@ 7pm"). Extract the LAST AM/PM time mentioned —
        # when info strings name multiple dates/times, the close time is
        # almost always the trailing one.
        closing_raw = c.get('date_end', '')
        date_info = c.get('date_info', '') or ''
        closing_fmt = closing_raw
        # Build a real datetime for sorting too. When no time was parseable,
        # use 23:59 so unknown-time auctions sort AFTER known ones on the
        # same day (you'd rather see a known 6pm close before an unknown).
        closes_dt = None
        try:
            if closing_raw:
                day_dt = datetime.fromisoformat(closing_raw)
                date_part = day_dt.strftime("%b %d")
                time_match = re.findall(
                    r'(\d{1,2})(?::(\d{2}))?\s*([ap])\.?m\.?',
                    date_info, flags=re.IGNORECASE,
                )
                if time_match:
                    h, m, mer = time_match[-1]
                    hour24 = int(h) % 12 + (12 if mer.lower() == 'p' else 0)
                    minute = int(m) if m else 0
                    closes_dt = day_dt.replace(hour=hour24, minute=minute)
                    time_str = f"{int(h)}:{minute:02d}{mer.upper()}M"
                    closing_fmt = f"{date_part} @ {time_str}"
                else:
                    closes_dt = day_dt.replace(hour=23, minute=59)
                    closing_fmt = date_part
        except (ValueError, TypeError):
            closing_fmt = closing_raw
        rows.append({
            "select": False,
            "auction_id": aid,
            "name": c.get('name', ''),
            "items": c.get('lot_count', 0),
            "source": c.get('source', ''),
            "location": f"{c.get('city', '')}, {c.get('state', '')}".strip(", "),
            "closes": closing_fmt,
            "closes_dt": closes_dt,
            "categories_sampled": cat_preview or ("—" if aid in cat_samples else "(not sampled)"),
            "summary": summary,
            "auction_link": f"https://hibid.com/auction/{aid}",
        })
    picker_df = pd.DataFrame(rows)

    # --- Filters: search + source ---
    picker_search = st.text_input(
        "🔎 Search auction name / contents",
        key="picker_search",
        placeholder="e.g. 'fishing', 'estate', 'tools'",
    ).strip().lower()
    sources_avail = picker_df['source'].unique().tolist()
    source_filter = st.radio(
        "Source:", ["All"] + sources_avail,
        horizontal=True, key="picker_source",
    ) if len(sources_avail) > 1 else "All"

    # Sort selector. Default is "most items first" — big auctions are the
    # ones most worth deep-scanning, so surfacing them first matches the
    # typical workflow.
    sort_choice = st.radio(
        "Sort by:",
        options=[
            "🔢 Most items first",
            "🔢 Fewest items first",
            "⏰ Closing soonest",
            "⏰ Closing latest",
        ],
        horizontal=True,
        key="picker_sort",
    )

    shown = picker_df.copy()
    if picker_search:
        mask = (
            shown['name'].fillna("").str.lower().str.contains(picker_search, regex=False)
            | shown['summary'].fillna("").str.lower().str.contains(picker_search, regex=False)
        )
        shown = shown[mask]
    if source_filter != "All":
        shown = shown[shown['source'] == source_filter]

    # Apply sort. Closing-time sorts use the parsed `closes_dt` datetime;
    # rows where time was unknown end up with 23:59 of the closing day so
    # they tail the time-based sorts (you'd rather see known 6pm closes
    # first). For item-count sorts, ties break by closing date.
    if sort_choice.startswith("🔢 Most"):
        shown = shown.sort_values(['items', 'closes_dt'], ascending=[False, True])
    elif sort_choice.startswith("🔢 Fewest"):
        shown = shown.sort_values(['items', 'closes_dt'], ascending=[True, True])
    elif sort_choice.startswith("⏰ Closing soonest"):
        shown = shown.sort_values('closes_dt', ascending=True, na_position='last')
    else:  # Closing latest
        shown = shown.sort_values('closes_dt', ascending=False, na_position='last')
    shown = shown.reset_index(drop=True)

    fetch_lots_running = st.session_state.get('fetch_lots_running', False)

    # --- Selection state (single source of truth, keyed by auction_id) ---
    # Avoids st.data_editor's "lose-all-edits" footgun when the underlying
    # DataFrame changes (e.g. after filter changes or re-sort). Picks
    # survive any rerender because they live in session_state.
    if '_picked_auction_ids' not in st.session_state:
        st.session_state._picked_auction_ids = set()
    picked: set = st.session_state._picked_auction_ids

    all_ids = set(shown['auction_id'].tolist())
    # Drop any ids no longer in candidates (e.g. after re-discovery)
    picked &= set(picker_df['auction_id'].tolist())

    # --- Bulk actions row ---
    ac1, ac2 = st.columns([1, 1])
    with ac1:
        if st.button("✅ Select all shown", use_container_width=True,
                     disabled=fetch_lots_running):
            picked |= all_ids
            st.session_state._picked_auction_ids = picked
            st.rerun()
    with ac2:
        if st.button("⬜ Deselect all shown", use_container_width=True,
                     disabled=fetch_lots_running):
            picked -= all_ids
            st.session_state._picked_auction_ids = picked
            st.rerun()

    # --- Picker UI: card list ---
    # Streamlit's data_editor is a canvas grid that truncates long names
    # with no cell-wrap. We mimic a table with st.columns: every row uses
    # the same column proportions so they align vertically like a real
    # table, and titles wrap naturally without truncation. CSS shrinks
    # the per-row gap and adds a subtle bottom border so rows are dense.
    if shown.empty:
        st.info("No auctions match the current filters.")
    else:
        st.markdown(
            """
            <style>
            div[data-testid="stVerticalBlock"]:has(> div.picker-row) {
                gap: 0 !important;
            }
            .picker-row {
                border-bottom: 1px solid rgba(255,255,255,0.08);
                padding: 0;
            }
            .picker-row strong { font-weight: 600; }
            .picker-row.header { border-bottom: 2px solid rgba(255,255,255,0.18); }
            .picker-row.header strong { white-space: nowrap; }

            /* Crush the default vertical breathing room on markdown text
               that sits inside any column on the page. Streamlit wraps
               every st.markdown in <p> tags with default ~16px margins;
               that's the main reason picker rows look tall. Scoping to
               stColumn descendants keeps standalone st.markdown blocks
               (page captions, info banners) untouched. */
            [data-testid="stColumn"] [data-testid="stMarkdown"] {
                margin-bottom: 0 !important;
            }
            [data-testid="stColumn"] [data-testid="stMarkdown"] p {
                margin: 0 !important;
                line-height: 1.3 !important;
            }
            /* Streamlit also pads each stColumn and the surrounding
               stHorizontalBlock by default, which adds another chunk
               of vertical space per row. Crush both. The flex gap=0
               between rows means consecutive picker rows sit flush
               against each other separated only by the 1px border. */
            [data-testid="stColumn"] {
                padding-top: 0 !important;
                padding-bottom: 0 !important;
            }
            [data-testid="stHorizontalBlock"] {
                margin-bottom: 0 !important;
                margin-top: 0 !important;
            }

            /* Inline labels appear only on mobile so stacked cells make
               sense without a header row. Hidden on desktop where the
               column header already labels each cell. */
            .cell-label { display: none; opacity: 0.55; font-size: 0.85em; }

            /* DESKTOP / TABLET: 5 columns side-by-side. Override the
               global mobile-CSS wrap rule that would otherwise stack. */
            .picker-row [data-testid="stHorizontalBlock"] {
                flex-wrap: nowrap !important;
            }
            .picker-row [data-testid="stColumn"] {
                flex: 1 1 0% !important;
                min-width: 0 !important;
            }

            /* MOBILE: stack into card form. Checkbox + name on first
               line, items/closes/summary below indented under the name.
               Header row is hidden — inline labels show instead. */
            @media (max-width: 640px) {
                .picker-row.header { display: none !important; }
                .cell-label { display: inline; }

                .picker-row [data-testid="stHorizontalBlock"] {
                    flex-wrap: wrap !important;
                }
                /* Pick column: narrow, stays at the left */
                .picker-row [data-testid="stColumn"]:nth-child(1) {
                    flex: 0 0 36px !important;
                    min-width: 36px !important;
                    max-width: 36px !important;
                }
                /* Name column: takes the rest of the first line */
                .picker-row [data-testid="stColumn"]:nth-child(2) {
                    flex: 1 1 calc(100% - 40px) !important;
                    min-width: calc(100% - 40px) !important;
                }
                /* Items / Closes / Summary: each on its own line, indented */
                .picker-row [data-testid="stColumn"]:nth-child(n+3) {
                    flex: 1 1 100% !important;
                    min-width: 100% !important;
                    padding-left: 40px !important;
                    font-size: 0.92em;
                    opacity: 0.9;
                }
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        # Same column proportions for every row so they line up like a
        # table. Auction name gets the lion's share so long titles wrap
        # in place rather than overflowing. The Pick column is wide enough
        # to fit the header text "Pick" plus the checkbox without wrapping.
        col_widths = [0.07, 0.38, 0.10, 0.16, 0.29]

        # Header row
        st.markdown('<div class="picker-row header">', unsafe_allow_html=True)
        h = st.columns(col_widths)
        h[0].markdown("**Pick**")
        h[1].markdown("**Auction**")
        h[2].markdown("**Items**")
        h[3].markdown("**Closes**")
        h[4].markdown("**What's in this auction**")
        st.markdown('</div>', unsafe_allow_html=True)

        for _, row in shown.iterrows():
            aid = row['auction_id']
            is_picked = aid in picked
            st.markdown('<div class="picker-row">', unsafe_allow_html=True)
            ck, name_c, items_c, closes_c, summary_c = st.columns(col_widths)
            with ck:
                new_state = st.checkbox(
                    "pick",
                    value=is_picked,
                    key=f"pick_compact_{aid}",
                    label_visibility="collapsed",
                    disabled=fetch_lots_running,
                )
            name_c.markdown(f"**{row['name']}**")
            items_c.markdown(
                f'<span class="cell-label">Items: </span>{int(row["items"]):,}',
                unsafe_allow_html=True,
            )
            closes_c.markdown(
                f'<span class="cell-label">Closes: </span>{row["closes"] or "—"}',
                unsafe_allow_html=True,
            )
            summary_c.markdown(row['summary'] or "—")
            st.markdown('</div>', unsafe_allow_html=True)
            if new_state and not is_picked:
                picked.add(aid)
            elif not new_state and is_picked:
                picked.discard(aid)

        st.session_state._picked_auction_ids = picked

    selected_ids = picked  # everything picked across all filters
    selected_item_total = int(
        picker_df.loc[picker_df['auction_id'].isin(selected_ids), 'items'].sum()
    ) if selected_ids else 0

    st.markdown("---")
    sc1, sc2 = st.columns([1, 2])
    with sc1:
        st.metric("Selected", f"{len(selected_ids)} auctions")
    with sc2:
        st.metric("Items to scan", f"≈{selected_item_total:,}")

    fetch_disabled = len(selected_ids) == 0 or fetch_lots_running
    fetch_label = "⏳ Fetching lots…" if fetch_lots_running else (
        f"📥 Fetch items from {len(selected_ids)} selected auction(s)"
        if selected_ids else "📥 Select at least one auction"
    )
    if st.button(
        fetch_label, type="primary", use_container_width=True,
        disabled=fetch_disabled, key="fetch_lots_btn",
    ):
        st.session_state._selected_auction_ids = list(selected_ids)
        st.session_state.fetch_lots_running = True
        st.rerun()

    # Sampling + fetch-lots work blocks run at the TOP of the main viewport
    # (see the "WORK BLOCKS" section below the title) so progress is always
    # visible. This branch only renders the picker UI.

    # --- Reset / back control ---
    st.markdown("---")
    if st.button("🔄 Start over (discard candidate list)", use_container_width=False):
        st.session_state.auction_candidates = []
        st.session_state.category_samples = {}
        st.session_state.phase1_leads = pd.DataFrame()
        st.rerun()


# ---- DISCOVERY VIEW: no auction loaded ----
elif not st.session_state.phase1_leads.empty:
    df = st.session_state.phase1_leads
    total_items = len(df)
    total_auctions = df['auction'].nunique() if 'auction' in df.columns else 0

    # --- Filters ---
    with st.container():
        search_query = st.text_input(
            "🔎 Search",
            placeholder="Search auctions, item titles, or descriptions (e.g. 'fishing', 'vintage', 'Weber')",
            key="discovery_search",
        ).strip()

        # Category-group filter — checkbox row grouping HiBid's 30+ granular
        # categories into ~12 broad buckets (Electronics, Tools, Home, …).
        # Returns the df filtered to the selected groups, or unchanged if
        # nothing is checked.
        df = _build_category_filter(df, state_key="discovery_category_groups")

        # Source filter (only if multiple sources)
        sources = df['source'].unique().tolist() if 'source' in df.columns else []
        if len(sources) > 1:
            selected_source = st.radio("Source:", ["All"] + sources, horizontal=True)
            if selected_source != "All":
                df = df[df['source'] == selected_source]

        # Logistics filter — HARD items are hidden by default since they're
        # expensive to ship to eBay buyers, but for local-pickup auctions
        # (where you grab the item in person) they're still fair game.
        hard_total = int((df['logistics_ease'] == "HARD").sum()) if 'logistics_ease' in df.columns else 0
        show_hard = st.checkbox(
            f"🏋️ Include HARD-to-ship items ({hard_total} hidden)" if hard_total else "🏋️ Include HARD-to-ship items",
            value=False,
            key="discovery_show_hard",
            help=(
                "Items matching the 'ship_killers' regex (furniture, heavy, "
                "large, mowers, pickup-only, etc.) are hidden by default "
                "because they're costly to re-ship to an eBay buyer. Turn on "
                "for local-pickup auctions where you plan to move the item "
                "yourself, or to see ALL lots regardless of shipability."
            ),
            disabled=(hard_total == 0),
        )
        if not show_hard and 'logistics_ease' in df.columns:
            df = df[df['logistics_ease'] != "HARD"]

    # --- Apply search + category filters ---
    if search_query:
        q = search_query.lower()
        title_col = 'title' if 'title' in df.columns else None
        desc_col = 'description' if 'description' in df.columns else None
        auction_col = 'auction' if 'auction' in df.columns else None

        mask = pd.Series(False, index=df.index)
        if auction_col:
            mask = mask | df[auction_col].fillna("").str.lower().str.contains(q, regex=False)
        if title_col:
            mask = mask | df[title_col].fillna("").str.lower().str.contains(q, regex=False)
        if desc_col:
            mask = mask | df[desc_col].fillna("").str.lower().str.contains(q, regex=False)
        df = df[mask]

    if df.empty:
        hints = ["broaden the search", "clear the category filter"]
        if hard_total and not show_hard:
            hints.append(f"tick **🏋️ Include HARD-to-ship items** ({hard_total} available)")
        st.warning(
            f"No matches for your filters. (Started with {total_auctions} auctions, "
            f"{total_items} items.) Try: {', '.join(hints)}."
        )
        st.stop()

    auction_groups = df.groupby('auction', sort=False)

    # --- Per-auction metrics we can sort on ---
    has_easy = 'logistics_ease' in df.columns
    metrics = df.groupby('auction').agg(
        items=('title', 'count'),
        closing=('closing_date', 'first'),
    )
    if has_easy:
        metrics['easy_ship'] = df.groupby('auction')['logistics_ease'].apply(
            lambda s: int((s == 'EASY').sum())
        )
    else:
        metrics['easy_ship'] = 0

    sort_choice = st.radio(
        "Sort auctions by:",
        options=[
            "📦 Easy-ship count (most first)",
            "⏰ Closing soonest",
            "🔢 Item count (most first)",
        ],
        horizontal=True,
        index=0,
    )
    if sort_choice.startswith("📦"):
        # Easy-ship desc, then closing soonest as tiebreaker so same-count
        # auctions still show the ones ending first at the top.
        auction_order = metrics.sort_values(
            ['easy_ship', 'closing'], ascending=[False, True]
        )
    elif sort_choice.startswith("⏰"):
        auction_order = metrics.sort_values('closing', ascending=True)
    else:
        auction_order = metrics.sort_values(
            ['items', 'closing'], ascending=[False, True]
        )

    filter_bits = []
    if search_query:
        filter_bits.append(f"search \"{search_query}\"")
    category_picks = st.session_state.get("discovery_category_groups", set())
    if category_picks:
        filter_bits.append(f"{len(category_picks)} category group(s)")
    filter_suffix = f" — filtered by {', '.join(filter_bits)}" if filter_bits else ""

    st.subheader(f"Discovery Results — {len(auction_order)} auctions, {len(df)} items{filter_suffix}")
    st.caption("Expand an auction to preview items, then click **🎯 Analyze This Auction** to run the full Phase 2 analysis.")

    for auction_name in auction_order.index:
        auction_df = auction_groups.get_group(auction_name).reset_index(drop=True)
        _render_auction_card(auction_name, auction_df)

    # Nifty CSV export (bottom, collapsed)
    with st.expander("📦 Nifty.ai CSV Export"):
        st.write("Download all discovered items as a Nifty.ai bulk-import CSV.")
        st.download_button(
            label="📥 Download CSV",
            data=st.session_state.phase1_leads.to_csv(index=False).encode('utf-8'),
            file_name="htown_finds_nifty_import.csv",
            mime="text/csv",
        )

# ---- EMPTY STATE: nothing discovered yet ----
else:
    st.info(
        "👋 **Two-step discovery:**\n\n"
        "1. Configure your filters in the sidebar and click **🔍 Discover Auctions** — "
        "pulls the list of open auctions and a lot-preview per auction so you can "
        "see what's in each.\n"
        "2. Pick which auctions are worth deep-scanning based on the "
        "*What's in this auction* column.\n"
        "3. Click **📥 Fetch items** to pull every lot for just your picks."
    )
