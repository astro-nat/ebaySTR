import streamlit as st # type: ignore
import streamlit.components.v1 as components # type: ignore
import numpy as np
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
    # auto = expanded on wide screens, collapsed-overlay on narrow ones.
    # Hard-coding "expanded" forced the sidebar to occupy a fixed chunk
    # of the screen at every width, which squeezed the analysis split-
    # screen at sizes between desktop and mobile.
    initial_sidebar_state="auto",
)

# Sidebar sizing — only force the wider 320–360 px footprint when the
# viewport actually has room (≥1024 px). Below that we let Streamlit's
# default responsive behavior take over: the sidebar auto-collapses to
# a hamburger and overlays content only when the user opens it.
st.markdown(
    """
    <style>
    @media (min-width: 1024px) {
        [data-testid="stSidebar"] {
            min-width: 320px !important;
            max-width: 360px !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
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

/* Removed sticky-results CSS — the analysis view now uses a single
   full-width results panel with manual controls in a collapsed
   expander below, so there's no second column to make sticky. */
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

# --- Audit knobs ---
# `audit_workers` controls how many parallel Claude API calls run during
# the AI tier of the condition audit. 8 hits a sweet spot — fast without
# tripping rate limits. The legacy `audit_fast_mode` and `audit_batch_size`
# keys are kept (no-ops) so older session state from before the API
# migration doesn't crash on read.
if 'audit_workers' not in st.session_state:
    st.session_state.audit_workers = 8
if 'audit_fast_mode' not in st.session_state:
    st.session_state.audit_fast_mode = False
if 'audit_batch_size' not in st.session_state:
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
if 'comps_max_lots' not in st.session_state:
    # 0 = no cap
    st.session_state.comps_max_lots = 0
if 'comps_exclude_hard' not in st.session_state:
    st.session_state.comps_exclude_hard = True
if 'comps_only_img_promoted' not in st.session_state:
    st.session_state.comps_only_img_promoted = False
if 'comps_chunk_size' not in st.session_state:
    # Process eligible lots N at a time. The pipeline hides the results
    # table between chunks (cooking screen), so multi-chunk runs feel
    # like the table never appears. Default 5000 makes virtually every
    # auction complete in a single chunk — the user sees one cooking
    # screen, then full results. Multi-chunk is still available via the
    # Comps tab settings for very large auctions where progress visibility
    # matters more than layout stability.
    st.session_state.comps_chunk_size = 5000
if 'comps_use_auction_str' not in st.session_state:
    # When True, sample STR per-auction (3 lots each) instead of scraping
    # STR for every lot. Huge speedup for big auctions.
    st.session_state.comps_use_auction_str = True
if 'comps_workers' not in st.session_state:
    # Thread-pool size for parallel price comps. 8 hits a good balance:
    # ~8x speedup without getting throttled by eBay/Mercari scraping.
    st.session_state.comps_workers = 8

if 'cache_ttl_days' not in st.session_state:
    st.session_state.cache_ttl_days = 1

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


def _extract_auction_id(auction_df: pd.DataFrame):
    """Pull the HiBid auction_id from a DataFrame.

    Discovery rows store it as a URL like https://hibid.com/auction/12345
    in the 'auction_link' column. We parse the trailing int.

    Defined this early in the file because the header refresh button
    (rendered above the rest of the analysis-view code) needs it to
    route a single-auction refresh.
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


# --- MAIN DASHBOARD UI ---
# Compact top bar: title on the left, settings popovers + Refresh button
# on the right. Replaces the old left sidebar — settings are tucked behind
# a popover instead of taking permanent screen space, since the user
# rarely changes them.
discover_running = st.session_state.get('discover_running', False)
fetch_lots_running = st.session_state.get('fetch_lots_running', False)
any_running = discover_running or fetch_lots_running
_restored_at = st.session_state.get('_discovery_restored_from')

# Refresh-button context: when the user has drilled into an auction, the
# button should re-fetch JUST that auction (not re-run discovery against
# the entire HiBid catalog, which is what "Discover" / "Refresh" did
# historically). We detect that here so the header can render the right
# label and dispatch the right action.
_in_analysis_view = (
    bool(st.session_state.get('current_auction'))
    and isinstance(st.session_state.get('selected_leads'), pd.DataFrame)
    and not st.session_state.selected_leads.empty
)

# Auto-collapse the sidebar when the user starts working an auction.
# Streamlit doesn't expose a programmatic "collapse sidebar" call,
# and the JS-click-via-components-html approach is unreliable because
# of iframe sandboxing across Streamlit versions. CSS-based hiding is
# the bulletproof alternative: track a `_sidebar_force_collapsed`
# session-state flag and inject a `display: none` rule when it's set.
#
# Auto-set the flag on the (was-not, is-now)-in-analysis transition,
# auto-clear it when the user goes back to the discovery view, and
# expose a manual toggle button so the user can override either way.
_was_in_analysis = st.session_state.get('_was_in_analysis', False)
_just_entered_analysis = _in_analysis_view and not _was_in_analysis
_just_left_analysis = _was_in_analysis and not _in_analysis_view
st.session_state._was_in_analysis = _in_analysis_view

if _just_entered_analysis:
    st.session_state._sidebar_force_collapsed = True
elif _just_left_analysis:
    # Back to the picker — show the auction list again automatically.
    st.session_state._sidebar_force_collapsed = False

if st.session_state.get('_sidebar_force_collapsed', False):
    # Hide the sidebar AND the collapse-toggle button that Streamlit
    # otherwise renders in the corner. We provide our own manual
    # reopen button below the header so the toggle stays in a
    # predictable place.
    st.markdown(
        """
        <style>
        section[data-testid="stSidebar"] {
            display: none !important;
        }
        [data-testid="collapsedControl"] {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

header_title_col, header_actions_col = st.columns([3, 2])
with header_title_col:
    st.markdown("## 🛰️ Auction Intelligence Dashboard")

with header_actions_col:
    # Three slots: Sourcing popover, Memory popover, and a sidebar
    # toggle that's always reachable so the user can manually reopen
    # the auction list after the auto-collapse fires.
    pop_sourcing, pop_memory, btn_sidebar = st.columns(3)

    with pop_sourcing:
        with st.popover("📍 Sourcing", use_container_width=True):
            user_zip = st.text_input("Home Zip Code", value="77058")
            radius = st.slider("Local Pickup Radius (mi)", 5, 100, 20)
            include_nationwide = st.checkbox(
                "Include Nationwide (Ship-to-Me)", value=True,
            )
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

    with pop_memory:
        with st.popover("💾 Memory", use_container_width=True):
            cached_list = _AUCTION_CACHE.list_all(ttl_days=st.session_state.cache_ttl_days)
            fresh_count = sum(1 for c in cached_list if c['fresh'])
            st.caption(
                f"**{fresh_count}** auction(s) cached. "
                "Audit + price-comp results are reused when you re-open an auction — "
                "current bids refresh every discovery run."
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

    # The Refresh / Discover button moved to the top of the sidebar
    # auction list (see _render_sidebar_refresh_button below) so it's
    # adjacent to the list it refreshes.

    with btn_sidebar:
        # Manual sidebar toggle so the user can override the auto-
        # collapse behavior in either direction. Label flips based on
        # the current force-collapsed state.
        _is_hidden = bool(st.session_state.get('_sidebar_force_collapsed', False))
        _toggle_label = "📋 Show auctions" if _is_hidden else "🔽 Hide auctions"
        if st.button(
            _toggle_label,
            use_container_width=True,
            key="sidebar_toggle_btn",
            help=("Show the auction-list sidebar."
                  if _is_hidden else
                  "Hide the auction-list sidebar to free up screen space."),
        ):
            st.session_state._sidebar_force_collapsed = not _is_hidden
            st.rerun()

# Auto-discover on first page load when there's no cached discovery to
# show — saves the user a click. The flag prevents re-triggering across
# Streamlit reruns within the same session if the discover fails or
# returns zero candidates.
if (
    not st.session_state.get('_auto_discover_triggered', False)
    and not any_running
    and st.session_state.phase1_leads.empty
    and not st.session_state.get('auction_candidates')
):
    st.session_state._auto_discover_triggered = True
    st.session_state._sourcing_cfg = {
        "zip": user_zip,
        "radius": radius,
        "include_nationwide": include_nationwide,
        "closing_days": closing_days,
        "category_filter": category_filter,
    }
    st.session_state.discover_running = True
    st.rerun()

# ================================================================
# SIDEBAR: persistent auction list. Always visible while there are
# discovered candidates. Clicking an auction row triggers a single-
# auction fetch (reuses the same path as the in-page picker's
# "Analyze" button). Selected auction gets a visual highlight.
# ================================================================
# ================================================================
# AUCTION TRIAGE — pre-comp signals so the user can spend ScrapingBee
# credits on auctions that are likely to find arbitrage and skip the
# obvious losers without burning credits to discover that.
#
# Two functions:
#   _estimate_auction_cost(lot_count, sample_payload) -> (pc_pct, credits)
#   _auction_signal(name, items, closes_dt, last_run_dt) -> (rank, reason)
#
# Both run on the existing post-discovery sample data — zero new HTTP.
# ================================================================
# Approximate ScrapingBee cost per non-PC-covered lot. Each lot kicks
# off an eBay-sold scrape (~25 credits) plus a Mercari-sold scrape
# (~25 credits) plus possibly a per-item STR scrape for cards/games/
# comics (~25 credits). Average ~50 credits/lot is a reasonable
# all-in estimate at user volume.
_CREDITS_PER_NON_PC_LOT = 50


def _estimate_auction_cost(lot_count: int, sample_payload):
    """Return (pc_coverage_pct, est_credits) for a discovered auction.

    pc_coverage is the share of sampled lot titles that route through
    PriceCharting (TCG/video_game/comic) — those don't burn ScrapingBee
    credits. sports_card titles classify but bypass PC and route to
    eBay-sold scraping, so they DO cost.

    With no sample yet, assumes worst case (everything costs).
    """
    from scraper.pricecharting import classify_for_pricecharting
    titles = (sample_payload or {}).get('titles') or []
    if not titles:
        return 0.0, int(lot_count * _CREDITS_PER_NON_PC_LOT)
    pc_hits = sum(
        1 for t in titles
        if classify_for_pricecharting(t or '')
           in ('tcg', 'video_game', 'comic')
    )
    pc_coverage = pc_hits / len(titles)
    est_credits = int(lot_count * (1 - pc_coverage) * _CREDITS_PER_NON_PC_LOT)
    return pc_coverage, est_credits


# Auction-name keywords that signal "avoid" — base products are too
# commoditized for arbitrage even when prices are accurate.
_AVOID_KEYWORDS = (
    'liquidation', 'overstock', 'new returns', 'warehouse',
    'wholesale', 'pallet', 'truckload', 'amazon return',
)
# Name keywords that signal "good" — variety + older inventory means
# more chance of finding an underpriced item.
_PREFER_KEYWORDS = (
    'estate', 'storage', 'collection', 'consignment', 'antique',
    'vintage',
)


def _auction_signal(name, items, closes_dt, last_run_dt):
    """Return (rank, reason) where rank is 'good' | 'caution' | 'avoid'.

    Heuristics, ranked first-match-wins:
      - 'avoid': name matches commoditized-source keywords, lot count
        > 1500, or closing in <2 hours (comp run wouldn't finish).
      - 'caution': already analyzed in the last 24h, or outside the
        100-500 lot sweet spot but otherwise fine.
      - 'good': in the sweet spot, optionally with a 'prefer' keyword.
    """
    nm = (name or '').lower()
    for kw in _AVOID_KEYWORDS:
        if kw in nm:
            return 'avoid', f"Name contains '{kw}' — base prices typically too compressed for arbitrage"
    if items > 1500:
        return 'avoid', f"{items:,} lots — too expensive at current credit rates"
    if closes_dt:
        secs = (closes_dt - datetime.now()).total_seconds()
        if 0 < secs < 7200:
            return 'avoid', f"Closes in {int(secs / 60)} min — comp run wouldn't finish in time"
    if last_run_dt:
        age = (datetime.now() - last_run_dt).total_seconds()
        if age < 86400:
            return 'caution', f"Already analyzed {int(age / 3600)}h ago — re-running may be redundant"
    if not (100 <= items <= 500):
        if items < 100:
            return 'caution', f"Only {items} lots — small sample, finds may be sparse"
        # 500 < items <= 1500
        return 'caution', f"{items:,} lots — moderate cost, worth analyzing if categories look good"
    # In the 100-500 sweet spot
    for kw in _PREFER_KEYWORDS:
        if kw in nm:
            return 'good', f"Sweet-spot lot count + name suggests '{kw}' (good variety)"
    return 'good', "Sweet-spot lot count (100-500)"


def _render_sidebar_refresh_button():
    """Render the primary Refresh / Discover button at the top of the sidebar.

    Behavior is context-aware:
      - In an analysis view → 🔄 Refresh auction (re-fetches just the
        currently loaded auction's lots, preserves cached audit/comps).
      - With cached discovery present → 🔄 Refresh (re-runs full discovery).
      - Empty state → 🔍 Discover (kicks off the first discovery).
    """
    if discover_running:
        refresh_label = "⏳ Discovering…"
    elif fetch_lots_running:
        refresh_label = "⏳ Refreshing…"
    elif _in_analysis_view:
        refresh_label = "🔄 Refresh auction"
    elif _restored_at:
        refresh_label = "🔄 Refresh"
    else:
        refresh_label = "🔍 Discover"

    refresh_help = (
        "Re-fetch lots for the auction you're currently viewing. "
        "Pulls fresh bids; cached audit + comp results are preserved."
        if _in_analysis_view
        else "Re-fetch the open-auction list with the current Sourcing "
             "settings. Successful runs are cached for 24h and "
             "auto-restored on reload."
    )

    if st.button(
        refresh_label,
        type="primary",
        use_container_width=True,
        disabled=any_running,
        key="discover_btn",
        help=refresh_help,
    ):
        if _in_analysis_view:
            # Single-auction refresh path. Pull the auction_id off the
            # loaded analysis, queue a fetch_lots run for just that ID,
            # and clear the analysis-view state so the auto-load step at
            # the top of the dispatch picks the freshly fetched data back
            # up (with cached audit + comps merged on top via
            # _load_auction_for_analysis).
            aid = _extract_auction_id(st.session_state.selected_leads)
            if aid is not None:
                st.session_state._selected_auction_ids = [aid]
                st.session_state.current_auction = None
                st.session_state.selected_leads = pd.DataFrame()
                st.session_state.fetch_lots_running = True
                st.rerun()
            # If we somehow can't extract the auction_id, fall through
            # to a full discovery as a safe default.
        st.session_state._sourcing_cfg = {
            "zip": user_zip,
            "radius": radius,
            "include_nationwide": include_nationwide,
            "closing_days": closing_days,
            "category_filter": category_filter,
        }
        st.session_state.discover_running = True
        st.rerun()


def _render_sidebar_auction_list():
    candidates = st.session_state.get('auction_candidates') or []
    cat_samples = st.session_state.get('category_samples', {}) or {}

    with st.sidebar:
        # Refresh button lives at the very top of the sidebar so it's
        # always reachable without scrolling, regardless of how long
        # the auction list grows.
        _render_sidebar_refresh_button()
        st.markdown("### 📋 Auctions")
        if not candidates:
            st.caption(
                "Nothing discovered yet. Click **🔍 Discover** above "
                "to fetch the open-auction list."
            )
            return

        st.caption(f"**{len(candidates)}** open auction(s)")

        # Compact search + sort
        sb_search = st.text_input(
            "🔎 Filter",
            key="sidebar_picker_search",
            placeholder="Search name or contents…",
            label_visibility="collapsed",
        ).strip().lower()
        sb_sort = st.selectbox(
            "Sort",
            options=[
                "🎯 Best fit first",
                "💸 Highest PC % first",
                "💸 Lowest credit cost first",
                "🔢 Most items first",
                "⏰ Closing soonest",
                "🔢 Fewest items first",
                "⏰ Closing latest",
            ],
            key="sidebar_picker_sort",
            label_visibility="collapsed",
        )
        sb_hide_avoid = st.checkbox(
            "Hide 🔴 avoid",
            key="sidebar_hide_avoid",
            help="Filter out auctions flagged as low-priority: name "
                 "contains 'liquidation'/'overstock'/'pallet'/etc., lot "
                 "count > 1500, or closing in <2 hours.",
        )

        # Build display rows. Reuse the same parsing/closing-time logic
        # the in-page picker uses, kept lightweight.
        rows = []
        for c in candidates:
            aid = c['auction_id']
            raw_sample = cat_samples.get(aid)
            if isinstance(raw_sample, dict):
                sample_payload = raw_sample
            elif isinstance(raw_sample, list):
                sample_payload = {"categories": raw_sample, "cat_counts": {}, "titles": []}
            else:
                sample_payload = None
            summary = (
                Phase1Scraper.generate_auction_summary(c, sample_payload)
                if sample_payload is not None else ""
            )
            closing_raw = c.get('date_end', '')
            date_info = c.get('date_info', '') or ''
            closing_fmt = closing_raw
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
                pass
            # Cost + signal for triage. Both are pure functions of the
            # data we already have — zero additional HTTP.
            pc_pct, est_credits = _estimate_auction_cost(
                int(c.get('lot_count') or 0), sample_payload,
            )
            rows.append({
                'auction_id': aid,
                'name': c.get('name') or '(unnamed)',
                'items': int(c.get('lot_count') or 0),
                'closes_fmt': closing_fmt or '—',
                'closes_dt': closes_dt,
                'summary': summary or '',
                'source': c.get('source') or '',
                'pc_pct': pc_pct,
                'est_credits': est_credits,
            })

        # Filter + sort
        if sb_search:
            rows = [
                r for r in rows
                if sb_search in r['name'].lower() or sb_search in r['summary'].lower()
            ]

        # Determine the currently-loaded auction (for highlighting)
        active_aid = None
        sel_leads = st.session_state.get('selected_leads')
        if isinstance(sel_leads, pd.DataFrame) and not sel_leads.empty:
            active_aid = _extract_auction_id(sel_leads)

        sb_fetch_lots_running = st.session_state.get('fetch_lots_running', False)

        # Build a {auction_id: cached_at_datetime} map for the "last
        # analyzed" labels. Single list_all() call, then we look up by
        # id in the loop. Pass a generous TTL so stale entries still
        # surface a timestamp — the user wants to see "12 days ago" on
        # a stale auction, not nothing.
        last_run_map = {}
        try:
            for entry in _AUCTION_CACHE.list_all(ttl_days=365):
                ts_raw = entry.get('cached_at') or ''
                try:
                    last_run_map[entry.get('auction_id')] = (
                        datetime.fromisoformat(ts_raw) if ts_raw else None
                    )
                except (ValueError, TypeError):
                    pass
        except Exception:
            pass  # Cache read failure shouldn't break the picker

        def _format_last_run(when):
            """Compact relative timestamp for the per-row label."""
            if when is None:
                return ""
            delta = datetime.now() - when
            if delta.total_seconds() < 60:
                return "just now"
            if delta.total_seconds() < 3600:
                return f"{int(delta.total_seconds() / 60)}m ago"
            if delta.total_seconds() < 86400:
                return f"{int(delta.total_seconds() / 3600)}h ago"
            return f"{delta.days}d ago"

        # Compute the triage signal for every row using auction-name
        # heuristics + lot count + closing time + last-run age.
        for row in rows:
            rank, reason = _auction_signal(
                row['name'], row['items'], row['closes_dt'],
                last_run_map.get(row['auction_id']),
            )
            row['signal_rank'] = rank
            row['signal_reason'] = reason

        # Apply the "hide avoid" filter after computing signals.
        if sb_hide_avoid:
            rows = [r for r in rows if r['signal_rank'] != 'avoid']

        # Sort. "Best fit first" uses the signal as the primary key
        # (good > caution > avoid), then est_credits ascending, then
        # closing time. The other sorts are unchanged.
        if sb_sort.startswith("🎯 Best"):
            _rank_order = {'good': 0, 'caution': 1, 'avoid': 2}
            rows.sort(key=lambda r: (
                _rank_order.get(r['signal_rank'], 3),
                r['est_credits'],
                r['closes_dt'] or datetime.max,
            ))
        elif sb_sort.startswith("💸 Highest PC"):
            # Higher PC % = cheaper to comp. Tiebreak by closing soonest
            # so finds with same coverage but ending sooner float up.
            rows.sort(key=lambda r: (
                -(r.get('pc_pct') or 0),
                r['closes_dt'] or datetime.max,
            ))
        elif sb_sort.startswith("💸 Lowest credit"):
            # Cheapest absolute spend — useful when budget is tight and
            # you want to rip through several small auctions cheaply.
            rows.sort(key=lambda r: (
                r['est_credits'],
                r['closes_dt'] or datetime.max,
            ))
        elif sb_sort.startswith("🔢 Most"):
            rows.sort(key=lambda r: (-r['items'], r['closes_dt'] or datetime.max))
        elif sb_sort.startswith("🔢 Fewest"):
            rows.sort(key=lambda r: (r['items'], r['closes_dt'] or datetime.max))
        elif sb_sort.startswith("⏰ Closing soonest"):
            rows.sort(key=lambda r: r['closes_dt'] or datetime.max)
        else:
            rows.sort(key=lambda r: r['closes_dt'] or datetime.min, reverse=True)

        if not rows:
            st.caption("_No auctions match the filter._")
            return

        # Triage icon for the row label. Prefixed before the auction
        # name so "Best fit" sorting visually surfaces 🟢 rows.
        _signal_icon = {'good': '🟢', 'caution': '🟡', 'avoid': '🔴'}

        def _format_credits(c: int) -> str:
            """Compact credit count for the per-row label."""
            if c == 0:
                return "free (PC)"
            if c < 1000:
                return f"~{c} cr"
            return f"~{c / 1000:.1f}k cr"

        for row in rows:
            aid = row['auction_id']
            is_active = (aid == active_aid)
            last_run_str = _format_last_run(last_run_map.get(aid))
            icon = '🟢 ' if is_active else _signal_icon.get(row['signal_rank'], '⚪ ')
            credits_label = _format_credits(row['est_credits'])
            pc_label = (
                f" · {int(round(row['pc_pct'] * 100))}% PC"
                if row.get('pc_pct') and row['pc_pct'] > 0 else ""
            )
            # 🕒 Last analyzed line: only when we have a cache entry.
            footer = f"\n\n🕒 _Last analyzed: {last_run_str}_" if last_run_str else ""
            label = (
                f"{icon}**{row['name']}**\n\n"
                f"{row['items']:,} lots · {row['closes_fmt']}\n\n"
                f"💸 {credits_label}{pc_label}"
                f"{footer}"
            )
            # Tooltip combines the auction summary with the triage
            # reason so the user understands why a row is flagged.
            tip_bits = []
            if row.get('signal_reason'):
                tip_bits.append(row['signal_reason'])
            if row['summary']:
                tip_bits.append(row['summary'])
            tooltip = " — ".join(tip_bits)[:300] if tip_bits else None

            # Each row uses its own button — clicking dispatches to the
            # single-auction fetch path. We disable while a fetch is
            # already in flight to avoid racing.
            if st.button(
                label,
                key=f"sidebar_pick_{aid}",
                use_container_width=True,
                disabled=sb_fetch_lots_running,
                type="primary" if is_active else "secondary",
                help=tooltip,
            ):
                # Switching auctions: clear the current analysis state
                # so the auto-load step in the dispatch picks up the
                # newly-fetched lots and lands us on the new auction.
                st.session_state._selected_auction_ids = [aid]
                st.session_state.current_auction = None
                st.session_state.selected_leads = pd.DataFrame()
                st.session_state.fetch_lots_running = True
                st.rerun()


# Render the persistent auction-list sidebar
_render_sidebar_auction_list()


# Subtle "showing cached results" caption under the header, only when
# we restored from disk (so the user knows the results aren't fresh).
if _restored_at and not discover_running:
    _age = datetime.now() - _restored_at
    if _age.total_seconds() < 3600:
        _age_str = f"{int(_age.total_seconds() / 60)} min ago"
    else:
        _age_str = f"{_age.total_seconds() / 3600:.1f}h ago"
    st.caption(
        f"♻️ Showing cached results from **{_age_str}** "
        f"({len(st.session_state.auction_candidates)} auctions). "
        f"Click 🔄 Refresh to re-fetch."
    )

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
                f"- **Dropped by category filter:** {_diag.get('filtered_cat', 0)}"
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

            # Fallback: synthesize candidates from session-state DataFrames
            # when the candidate list doesn't have these IDs. Common when
            # refreshing a cached analysis after the 24h discovery cache
            # expired — we still know enough about the auction (link,
            # name, source, closing date) to refetch its lots.
            if not selected_candidates and sel_ids:
                seen_ids = {c['auction_id'] for c in selected_candidates}
                for source_df_name in ('selected_leads', 'audit_results',
                                        'phase1_leads'):
                    src = st.session_state.get(source_df_name)
                    if not isinstance(src, pd.DataFrame) or src.empty:
                        continue
                    if 'auction_link' not in src.columns:
                        continue
                    for aid in sel_ids - seen_ids:
                        sub = src[src['auction_link'].fillna('').astype(str)
                                  .str.contains(f'/auction/{aid}', regex=False)]
                        if sub.empty:
                            continue
                        first = sub.iloc[0]
                        selected_candidates.append({
                            'auction_id': aid,
                            'name': str(first.get('auction') or '(unknown)'),
                            'source': str(first.get('source') or ''),
                            'date_end': str(first.get('closing_date') or ''),
                            'lot_count': len(sub),
                            'date_info': '',
                            'city': '',
                            'state': '',
                        })
                        seen_ids.add(aid)

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
                    reason = f"All {raw_count} lots were excluded by the category filter."
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
    # Fresh auction → reset chunked-comps state so the first comps click
    # starts a new run instead of trying to continue from the previous one.
    st.session_state.pop('_comps_has_more', None)
    st.session_state.pop('_comps_auction_str_map', None)
    st.session_state.pop('_comps_stats', None)
    # Reset all "in-progress" flags. These get set by their respective
    # buttons and cleared in `finally` blocks inside the analysis view —
    # but if the user refreshed mid-run, the analysis branch never
    # executed the finally, so the flags stay True and disable the
    # buttons on the new auction. Belt-and-suspenders reset here.
    st.session_state.audit_running = False
    st.session_state.comps_running = False
    st.session_state.img_enrich_running = False
    # Reset the auto-pipeline tracker so a new auction starts fresh
    # (audit auto-fires, then first comp chunk auto-fires). Failed
    # attempts on the previous auction don't poison this one.
    st.session_state.pop('_auto_pipeline_attempts', None)

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


# `_extract_auction_id` was hoisted near the top of the file so the
# header refresh button could call it. Kept the original definition site
# blank to preserve line numbers; the function is the same.


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
    """Load (and cache) the Phase2Scraper.

    The constructor reads the Anthropic key from `config.json` and
    lazy-creates the API client on first use — no model download, instant
    init. Cached so the same instance is reused across reruns.
    """
    from scraper import Phase2Scraper
    return Phase2Scraper(model_name=model_name)


def _run_ai_audit(leads_df, on_progress=None):
    """Run Phase 2 AI condition audit with detailed phase-by-phase status."""
    from scraper import Phase2Scraper

    total = len(leads_df)
    workers = int(st.session_state.get('audit_workers', 8))

    with st.status("🧠 Running AI Condition Audit…", expanded=True) as status:
        # Phase 1: pre-flight
        auditor = _get_auditor(Phase2Scraper.DEFAULT_MODEL)
        # If the cached Phase2Scraper instance has api_key=None but the
        # config now has one, force a rebuild — this happens when the
        # user added the key to config.json AFTER Streamlit started, and
        # @st.cache_resource has been holding the stale instance ever
        # since. Without this, the audit silently downgrades to
        # keyword-only and every non-keyword lot gets `no_api_key`.
        if not auditor.api_key:
            from scraper.config_loader import load_config
            config_key = (load_config().get("anthropic") or {}).get("api_key")
            if config_key:
                _get_auditor.clear()
                auditor = _get_auditor(Phase2Scraper.DEFAULT_MODEL)
        if not auditor.api_key:
            st.warning(
                "⚠️ No Anthropic API key found in `config.json`. The audit "
                "will run keyword-only — items without a keyword match will "
                "be marked 'Unknown' instead of red-flagged. Add your key "
                "under `anthropic.api_key` to enable AI classification."
            )

        # Pre-count how many lots will be pre-filtered (HARD logistics) so
        # the user sees the savings up front.
        hard_preview = 0
        if 'logistics_ease' in leads_df.columns:
            hard_preview = int((leads_df['logistics_ease'] == 'HARD').sum())

        st.write(
            f"**🔎 Step 1/2 — Three-tier condition classification** "
            f"on {total} items."
        )
        st.caption(
            "**Tier 1 — keyword regex** over the description (instant, free): "
            "matches phrases like *'untested'*, *'doesn't work'*, *'factory "
            "sealed'*. Most lots short-circuit here.  \n"
            "**Tier 2 — Claude Haiku text** (~500ms/lot, parallel): for lots "
            "with a substantial description but no keyword hit.  \n"
            "**Tier 3 — Claude Haiku vision** (~2s/lot, parallel): for lots "
            "with a short description, classifies from the thumbnail."
        )
        if hard_preview > 0:
            st.caption(
                f"⏭️ Skipping AI on **{hard_preview}** HARD-logistics lots "
                "(mattresses, vehicles, furniture, real estate, etc.) — "
                "auto-flagged as Unshippable, no API call needed."
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

        # Live callback: every batch boundary, push the partial df into
        # session_state and notify the caller (which throttles UI refresh).
        def _audit_live_cb(processed, total_items, partial_df):
            try:
                st.session_state.audit_results = partial_df
                if on_progress is not None:
                    on_progress(processed, total_items)
            except Exception:
                pass

        results_df = auditor.batch_audit(
            leads_df,
            progress_callback=ai_progress,
            batch_size=workers,
            live_callback=_audit_live_cb,
        )

        # Phase 2: summarize (counts + per-tier breakdown)
        good = int((~results_df['red_flag']).sum()) if 'red_flag' in results_df.columns else 0
        flagged = int(results_df['red_flag'].sum()) if 'red_flag' in results_df.columns else 0
        skipped_hard = int(results_df.attrs.get('audit_skipped_hard', 0) or 0)
        skipped_collectible = int(results_df.attrs.get('audit_skipped_collectible', 0) or 0)
        skipped_empty = int(results_df.attrs.get('audit_skipped_empty', 0) or 0)
        keyword_hits = int(results_df.attrs.get('audit_keyword_hits', 0) or 0)
        text_api_calls = int(results_df.attrs.get('audit_text_api_calls', 0) or 0)
        image_api_calls = int(results_df.attrs.get('audit_image_api_calls', 0) or 0)
        text_failed = int(results_df.attrs.get('audit_text_api_failed', 0) or 0)
        image_failed = int(results_df.attrs.get('audit_image_api_failed', 0) or 0)
        no_signal = int(results_df.attrs.get('audit_no_signal', 0) or 0)

        summary_parts = [f"✅ {good} good-condition", f"⚠️ {flagged} red-flagged"]
        if skipped_hard > 0:
            summary_parts.append(f"🚚 {skipped_hard} HARD-logistics (pre-filtered)")
        if skipped_collectible > 0:
            summary_parts.append(f"🎴 {skipped_collectible} collectibles (pass-through)")
        if skipped_empty > 0:
            summary_parts.append(f"❓ {skipped_empty} empty rows")
        st.write(f"**📊 Step 2/2 — Summary:** " + " · ".join(summary_parts))

        tier_parts = []
        if keyword_hits:
            tier_parts.append(f"🔍 {keyword_hits} by keyword")
        if text_api_calls:
            tier_parts.append(f"📝 {text_api_calls} by text API")
        if image_api_calls:
            tier_parts.append(f"🖼️ {image_api_calls} by image API")
        if no_signal:
            tier_parts.append(f"❓ {no_signal} no signal (short desc + no thumb)")
        if tier_parts:
            st.caption("**Tier breakdown:** " + " · ".join(tier_parts))
        if text_failed or image_failed:
            st.caption(
                f"⚠️ {text_failed + image_failed} API call(s) failed and were "
                "marked Unknown — check your network/key if this is high."
            )
        status.update(label="✅ AI audit complete", state="complete", expanded=False)

    return results_df


# Tokens that don't add product signal — used by _is_generic_title to
# determine whether a title has enough text-based hooks for eBay/Mercari
# search to identify the product on its own.
_GENERIC_TITLE_TOKENS = {
    'lot', 'lots', 'box', 'boxes', 'mystery', 'misc', 'miscellaneous',
    'various', 'assorted', 'mixed', 'collection', 'group', 'bundle',
    'set', 'sets', 'piece', 'pieces', 'items', 'item', 'things',
    'stuff', 'estate', 'find', 'finds', 'see', 'photo', 'photos',
    'picture', 'pictures', 'a', 'an', 'the', 'and', 'or', 'of',
    'with', 'in', 'on', 'at', 'by', 'as', 'to', 'for', 'from',
    'no', 'reserve', 'vintage', 'antique', 'old', 'used', 'new',
    'huge', 'large', 'small', 'great', 'nice', 'good', 'fair',
    'multiple', 'numerous', 'several', 'many', 'plus', 'unknown',
}


def _is_generic_title(title: str) -> bool:
    """Return True when the lot title has too little signal to identify
    the product from text alone.

    "Specific" titles (year + manufacturer + card #, model number,
    3+ informative words, etc.) are confidently searchable on text —
    eBay's image-search tier produces wrong matches on such titles
    (Topps↔OPC mix-ups, grade swaps on slabbed cards). We reserve
    image enrichment for lots like 'Box of misc items' / 'Lot 47' /
    'Mystery box' where the photo is the only way to identify the
    product.
    """
    if not title or len(title.strip()) < 5:
        return True

    # Year (1900-2030) → temporal anchor → specific
    if re.search(r'\b(19\d{2}|20\d{2})\b', title):
        return False
    # Decade marker like "1980s" / "70s" / "'90s" → specific
    if re.search(r"['']?\d{2,4}\s*['']?s\b", title):
        return False
    # Card # / lot # / SKU like #250, No. 14, #88AS-40 → specific
    if re.search(r'#\s*\w*\d+', title):
        return False
    # Model-number-ish token (alphanumeric SKU) → specific
    if re.search(r'\b[A-Z]{1,4}[\-.]?\d{2,}[\w\-.]*\b', title):
        return False

    # Count "significant" words (drop generic filler tokens). 3+ → specific.
    clean = re.sub(r'[^\w\s\-]', ' ', title.lower())
    words = [w for w in clean.split() if len(w) >= 2]
    significant = [w for w in words if w not in _GENERIC_TITLE_TOKENS]
    if len(significant) >= 3:
        return False

    return True


def _run_image_enrichment(audit_df, min_bid: float = 5.0):
    """Run image-based title enrichment on lots with too-generic titles.

    Gated to skip:
      - red-flagged lots (condition audit says broken/untested)
      - HARD logistics (we're not buying furniture to ship)
      - lots below min_bid (junk filter — don't burn API calls on $1 items)
      - lots with no thumbnail_url
      - **lots with already-specific titles** (year, card #, brand, 3+
        informative words). Image search frequently produces wrong
        matches on already-identifiable products (Topps/OPC mix-ups,
        grade swaps on slabbed cards), so we restrict it to lots where
        the title alone can't drive a comp search.

    Returns the DataFrame with six new img_* columns plus (where confidence
    is high enough) a promoted `enriched_title` that now carries brand +
    model + year pulled straight from matching eBay listings.
    """
    from scraper.vision_enrich import EbayImageEnricher, promote_image_titles
    from scraper.config_loader import load_config

    cfg = load_config()
    anth_cfg = cfg.get("anthropic") or {}
    enricher = EbayImageEnricher(
        cfg["ebay"]["app_id"], cfg["ebay"]["cert_id"],
        hibid_user_agent=cfg.get("api", {}).get(
            "user_agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"),
        anthropic_api_key=anth_cfg.get("api_key"),
        anthropic_model=anth_cfg.get("model", "claude-haiku-4-5"),
    )
    has_claude_fallback = bool(anth_cfg.get("api_key"))

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
        # Only run image matching when the title is too generic to
        # comp on text alone. Specific titles (year + brand + card # /
        # model number / 3+ informative words) have been a reliable
        # source of wrong image matches — Topps/OPC mix-ups on cards,
        # grade swaps on slabbed product, etc.
        title = str(
            row.get('enriched_title') or row.get('title') or ''
        )
        if not _is_generic_title(title):
            return False
        return True

    # Pre-count how many lots will actually be analyzed vs skipped, so the
    # progress bar can reflect real work (not padded by skipped rows).
    eligible_ids = {
        row.get('lot_id')
        for _, row in audit_df.iterrows() if gate(row)
    }
    eligible_count = len(eligible_ids)

    status_label = (
        f"🖼️ Enriching {eligible_count} titles via image search…"
        if has_claude_fallback
        else f"🖼️ Enriching {eligible_count} titles via eBay image_search…"
    )
    with st.status(status_label, expanded=True) as status:
        st.write(
            f"**Gated to {eligible_count} of {total} items** — skipping "
            "red-flagged, HARD-logistics, missing-image, sub-${:.2f}, "
            "and any lot whose title is already specific enough to comp "
            "on text alone (year, card #, brand, 3+ informative words)."
            .format(min_bid)
        )
        if has_claude_fallback:
            st.caption(
                "Image enrichment now only fires on **generic-titled lots** "
                "(*'Box of misc items'*, *'Lot 47'*, *'Mystery box'* — where "
                "the photo is the only way to identify the product). Two-tier "
                "flow when it does run: **eBay image_search first** (free, "
                "fast), then **Claude vision as a fallback**. Specific titles "
                "skip both tiers because eBay's image search frequently "
                "produces wrong matches on already-identifiable products "
                "(Topps↔OPC mix-ups on cards, grade swaps on slabs)."
            )
        else:
            st.caption(
                "For each eligible item: download the HiBid thumbnail, POST it "
                "to eBay's Browse `search_by_image` endpoint, and if the top "
                "hits agree on a product, rewrite the title to match what eBay "
                "actually sells it as. Add an `anthropic.api_key` to "
                "`config.json` to enable the Claude vision fallback for items "
                "eBay can't identify."
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
        promoted_mask = (
            promoted_df['enriched_title']
            != promoted_df.get('enriched_title_pre_image',
                               promoted_df['enriched_title'])
        )
        promoted = int(promoted_mask.sum())

        # Per-tier breakdown so the user can see how often each tier saved
        # the day. img_source is set per row by enrich_one.
        ebay_promoted = 0
        claude_promoted = 0
        if 'img_source' in promoted_df.columns:
            sources = promoted_df.loc[promoted_mask, 'img_source'].fillna('')
            ebay_promoted = int((sources == 'ebay').sum())
            claude_promoted = int((sources == 'claude').sum())

        skipped_gate = int(
            (promoted_df['img_error'] == 'skipped_gate').sum()
        ) if 'img_error' in promoted_df.columns else 0
        errored = int(
            promoted_df['img_error'].notna().sum()
            - (promoted_df['img_error'] == 'skipped_gate').sum()
        ) if 'img_error' in promoted_df.columns else 0

        summary_bits = [f"✅ {promoted} titles upgraded"]
        if has_claude_fallback and (ebay_promoted or claude_promoted):
            summary_bits.append(
                f"(🛒 {ebay_promoted} via eBay · 🤖 {claude_promoted} via Claude)"
            )
        summary_bits.append(f"⏭️ {skipped_gate} gated-out")
        summary_bits.append(f"⚠️ {errored} couldn't identify")
        st.write("**📊 Summary:** " + " · ".join(summary_bits))
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
    from scraper.gocollect import GoCollectLookup
    from scraper.config_loader import load_config
    cfg = load_config()
    pc_token = (cfg.get("pricecharting") or {}).get("token") or None
    pc_client = PriceChartingLookup(pc_token)
    gc_key = (cfg.get("gocollect") or {}).get("api_key") or None
    gc_client = GoCollectLookup(gc_key)
    sb_key = (cfg.get("scrapingbee") or {}).get("api_key") or None
    ebay = EbayPriceLookup(
        cfg["ebay"]["app_id"], cfg["ebay"]["cert_id"],
        pricecharting=pc_client,
        scrapingbee_key=sb_key,
        gocollect=gc_client,
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

        # ROI = (resale - cost) / cost. est_cost is computed from
        # max(current_bid, next_bid) in pass1, so it's > 0 whenever the
        # auction has a starting bid (= virtually always). Rows with
        # cost==0 fall through to None — the table sorts them last.
        comps_df['est_roi'] = None
        resale_num = pd.to_numeric(comps_df['est_resale'], errors='coerce')
        cost_num = pd.to_numeric(comps_df['est_cost'], errors='coerce')
        normal = resale_num.notna() & cost_num.gt(0)
        comps_df.loc[normal, 'est_roi'] = (
            (resale_num[normal] - cost_num[normal]) / cost_num[normal] * 100
        ).round(0)

        found = int(comps_df['est_resale'].notna().sum())
        st.write(f"**📊 Summary:** found price comps for {found}/{total} items.")
        status.update(label="✅ Price comps complete", state="complete", expanded=False)

    # Stitch comps_df (with resale) back together with skipped + flagged rows
    # so the results table still shows every lot, just with NaN resale for
    # skipped ones.
    combined = pd.concat([comps_df, skipped_df, flagged_df], ignore_index=True)
    return combined, found, total


# Comp columns that batch_lookup adds — used by the chunked variant to
# initialize empty placeholders and merge per-chunk results back into the
# accumulating audit_results DataFrame.
_COMP_COLUMNS = (
    'est_resale', 'price_low', 'price_high', 'comp_count',
    'ebay_comps', 'mercari_comps', 'pricecharting_comps', 'gocollect_comps',
    'price_source', 'ebay_str', 'str_source',
)


def _run_ebay_comps_chunk(audit_df, chunk_size: int = 200, on_lot_priced=None):
    """Run price comps on the NEXT batch of eligible-but-uncomped lots.

    Lets the user review earlier results while later chunks process.
    Each call:
      1. Identifies rows that are good (not red-flagged), pass the
         pre-comps filters, AND don't yet have an est_resale value.
      2. Runs comps on the first chunk_size of those.
      3. Merges the chunk's comp columns back into the input df at
         the original row indices.

    Args:
        audit_df: current full audit_results DataFrame.
        chunk_size: max number of pending lots to comp this call.
        on_lot_priced: optional callable(completed, total, last_lot)
            that fires after EACH individual lot finishes pricing.
            `last_lot` is a small dict with keys 'title', 'resale',
            'roi', 'ebay_comps', 'mercari_comps' so the caller can
            render a per-lot ticker without re-scanning the df.
            Lets the caller stream live updates into the UI without
            waiting for the whole chunk to finish. The function is
            also responsible for any UI refresh; this scope only
            updates session_state.

    Returns:
        (updated_df, found_in_chunk, processed_in_chunk, has_more)

    `has_more` is True when more eligible-uncomped rows remain after
    this chunk — the caller surfaces a "Continue" button when so.
    """
    df = audit_df.copy().reset_index(drop=True)

    # Initialize comp columns on first run so the .isna() check below
    # can find rows that haven't been processed yet.
    for col in _COMP_COLUMNS:
        if col not in df.columns:
            df[col] = None
    if 'est_roi' not in df.columns:
        df['est_roi'] = None

    # "Pending" = good (not red-flagged) AND no est_resale yet.
    not_red = (
        ~df['red_flag'].fillna(False).astype(bool)
        if 'red_flag' in df.columns
        else pd.Series(True, index=df.index)
    )
    not_processed = df['est_resale'].isna()
    pending_df = df[not_red & not_processed]

    eligible_df, _skipped_df, filter_summary = _apply_comps_filters(pending_df)
    total_pending = len(eligible_df)
    if total_pending == 0:
        return df, 0, 0, False

    chunk = eligible_df.head(chunk_size).copy()
    chunk_indices = chunk.index.tolist()  # original positions in df

    from scraper.ebay_prices import EbayPriceLookup
    from scraper.pricecharting import PriceChartingLookup
    from scraper.gocollect import GoCollectLookup
    from scraper.config_loader import load_config
    cfg = load_config()
    pc_token = (cfg.get("pricecharting") or {}).get("token") or None
    gc_key = (cfg.get("gocollect") or {}).get("api_key") or None
    sb_key = (cfg.get("scrapingbee") or {}).get("api_key") or None
    ebay = EbayPriceLookup(
        cfg["ebay"]["app_id"], cfg["ebay"]["cert_id"],
        pricecharting=PriceChartingLookup(pc_token),
        scrapingbee_key=sb_key,
        gocollect=GoCollectLookup(gc_key),
    )

    label = f"💰 Comping next {len(chunk)} of {total_pending} pending lot(s)…"
    with st.status(label, expanded=True) as status:
        st.caption(f"Filters applied — {filter_summary}")

        # STR sampling — cache the auction-level map across chunks so
        # we don't re-scrape it every batch. Only sample once per fetch.
        auction_str_map = st.session_state.get('_comps_auction_str_map')
        if (
            auction_str_map is None
            and st.session_state.get('comps_use_auction_str', True)
            and 'auction' in chunk.columns
        ):
            # Sample over the FULL eligible pending pool (not just chunk)
            # so the per-category buckets reflect the whole auction.
            all_eligible = pd.concat([eligible_df], ignore_index=False)
            buckets = (
                all_eligible.groupby([
                    'auction',
                    all_eligible['category'].fillna('').replace('', '(uncategorized)')
                ]).ngroups
                if 'category' in all_eligible.columns
                else all_eligible['auction'].nunique()
            )
            st.write(f"📈 Sampling STR across {buckets} category bucket(s) (one-time per fetch).")
            str_progress = st.progress(0)
            def str_cb(current, total_buckets):
                pct = current / total_buckets if total_buckets > 0 else 1.0
                str_progress.progress(min(pct, 1.0),
                                      text=f"Sampled STR for {current}/{total_buckets}…")
            auction_str_map = ebay.sample_auction_str(
                all_eligible, sample_size=2,
                progress_callback=str_cb, granularity="category",
            )
            st.session_state._comps_auction_str_map = auction_str_map

        st.write(f"🔗 Looking up sold prices for {len(chunk)} lots in this batch.")
        progress_bar = st.progress(0)
        current_item = st.empty()

        def price_cb(current, total_items, title_preview=""):
            pct = current / total_items if total_items > 0 else 1.0
            progress_bar.progress(min(pct, 1.0),
                                  text=f"Priced {current}/{total_items}…")
            if title_preview:
                current_item.caption(f"🔎 Just priced: *{title_preview}*")

        # The live callback receives the raw `lookup_price_range()` dict
        # whose keys differ from our DataFrame column names. Translate.
        _PAYLOAD_KEY_MAP = {
            'median': 'est_resale',
            'low': 'price_low',
            'high': 'price_high',
            'count': 'comp_count',
            'ebay_count': 'ebay_comps',
            'mercari_count': 'mercari_comps',
            'pricecharting_count': 'pricecharting_comps',
            'source': 'price_source',
        }

        # Per-lot live callback. Fires from the main thread (drained via
        # as_completed) so it's safe to touch Streamlit state. Merges the
        # individual lot's comp result into df + session_state on the
        # spot, then defers to the caller's UI-refresh hook.
        def _live_cb(chunk_idx, payload, completed, total_items):
            try:
                target_idx = chunk_indices[chunk_idx]
                for src_key, col in _PAYLOAD_KEY_MAP.items():
                    if src_key in payload:
                        df.at[target_idx, col] = payload[src_key]
                # ROI on this row only — cheap. Full ROI recompute happens
                # at the bottom of this function after the chunk completes.
                # est_cost is built from max(current_bid, next_bid) in pass1
                # so it's > 0 whenever the auction has a starting bid; rows
                # where cost==0 leave est_roi as None.
                cost = pd.to_numeric(df.at[target_idx, 'est_cost'], errors='coerce')
                resale = pd.to_numeric(df.at[target_idx, 'est_resale'], errors='coerce')
                if pd.notna(resale) and pd.notna(cost) and cost > 0:
                    df.at[target_idx, 'est_roi'] = round(
                        (resale - cost) / cost * 100, 0
                    )
                st.session_state.audit_results = df
                if on_lot_priced is not None:
                    # Pull a small "last lot" payload so the caller can
                    # render a per-lot ticker without re-scanning the df.
                    row = df.iloc[target_idx]
                    last_lot = {
                        'title': (
                            row.get('enriched_title')
                            or row.get('title')
                            or ''
                        ),
                        'resale': row.get('est_resale'),
                        'roi': row.get('est_roi'),
                        'ebay_comps': row.get('ebay_comps') or 0,
                        'mercari_comps': row.get('mercari_comps') or 0,
                    }
                    on_lot_priced(completed, total_items, last_lot)
            except Exception:
                # Live updates are best-effort — never fail the comp run
                # because of a UI hiccup.
                pass

        chunk_with_comps = ebay.batch_lookup(
            chunk,
            progress_callback=price_cb,
            auction_str_map=auction_str_map,
            max_workers=int(st.session_state.get('comps_workers', 8)),
            live_callback=_live_cb,
        )

        # Merge comp columns from chunk back into df at the original
        # row indices. .values bypasses index alignment (chunk_with_comps
        # was reset to 0..n-1 inside batch_lookup).
        for col in _COMP_COLUMNS:
            if col in chunk_with_comps.columns:
                df.loc[chunk_indices, col] = chunk_with_comps[col].values

        # Recompute ROI on every row that now has resale + cost. With
        # est_cost = max(current_bid, next_bid) + premium from pass1,
        # cost > 0 unless the auction omitted a starting bid (rare).
        cost_col = pd.to_numeric(df.get('est_cost'), errors='coerce')
        resale_col = pd.to_numeric(df.get('est_resale'), errors='coerce')
        has_data = resale_col.notna() & cost_col.gt(0)
        df.loc[has_data, 'est_roi'] = (
            (resale_col[has_data] - cost_col[has_data]) / cost_col[has_data] * 100
        ).round(0)

        found = int(chunk_with_comps['est_resale'].notna().sum())
        has_more = total_pending > len(chunk)
        status.update(
            label=f"✅ Batch complete — {found}/{len(chunk)} priced "
                  f"({total_pending - len(chunk)} still pending)" if has_more
                  else f"✅ All chunks complete — {found}/{len(chunk)} priced in final batch",
            state="complete", expanded=False,
        )

    return df, found, len(chunk), has_more


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
    # Cached est_resale can land as `object` dtype with internals
    # (Decimal, nullable extension dtype) that survive `pd.to_numeric`
    # but break Series.round. Coerce element-wise to a plain float64
    # numpy array first — np.round is dtype-strict from the start.
    resale = _to_float_array(out['est_resale'])
    resale_mask = ~np.isnan(resale)
    if resale_mask.any():
        net_resale = resale[resale_mask] * (1 - ebay_fee_pct) - ebay_fee_flat
        if 'source' in out.columns:
            item_ship = np.array([
                ship_cost if s == "Ship" else 0
                for s in out.loc[resale_mask, 'source'].tolist()
            ], dtype='float64')
        else:
            item_ship = 0.0
        max_bid = (net_resale / target_roi_val - item_ship) / (1 + buyer_premium_pct)
        max_bid = np.round(np.clip(max_bid, 0, None), 2)
        out.loc[resale_mask, 'max_bid'] = max_bid
    return out


def _to_float_array(series: pd.Series) -> np.ndarray:
    """Coerce a Series to a numpy float64 array element-wise.

    Element-wise float() avoids the dtype gotchas of pandas'
    `to_numeric` on cached payloads — same helper as in scraper/cache.py.
    """
    out = np.empty(len(series), dtype='float64')
    for i, v in enumerate(series):
        try:
            if v is None or (isinstance(v, float) and v != v):
                out[i] = np.nan
            else:
                out[i] = float(v)
        except (TypeError, ValueError):
            out[i] = np.nan
    return out


def _render_red_flag_editor(ar_full):
    """Render the red-flag review expander (bulk-clear button + editor).

    Pulled out of the inline analysis-view so the right-side results panel
    can render it from anywhere. ar_full is mutated in place when the
    user clicks the bulk-clear button or unchecks individual rows;
    audit_results in session_state is also written to keep things in sync.
    """
    flagged_mask = ar_full['red_flag'].fillna(False).astype(bool)
    flagged_count = int(flagged_mask.sum())
    with st.expander(
        f"🚩 Review {flagged_count} red-flagged item(s) "
        "— uncheck any the audit got wrong",
        expanded=False,
    ):
        # Bulk-clear escape hatch for auctions where the audit is mostly
        # wrong (typically sports cards / TCG / video games — the AI's
        # risk labels don't apply to collectibles whose condition is
        # encoded as a grade in the title).
        if st.button(
            f"⚡ Clear all {flagged_count} flag(s) in this auction",
            help="Set red_flag=False on every flagged row. "
                 "Use when the audit was systematically wrong "
                 "(e.g. an auction full of cards or comics). "
                 "Items will rejoin the next comps batch.",
            key=f"clear_all_flags_{st.session_state.get('current_auction', '')}",
        ):
            ar_full.loc[flagged_mask, 'red_flag'] = False
            st.session_state.audit_results = ar_full
            st.session_state._comps_has_more = True
            _save_current_auction_to_cache()
            st.success(
                f"✓ Cleared all {flagged_count} red flag(s). "
                "They'll be included in the next comps batch."
            )
            st.rerun()

        title_col = (
            'enriched_title' if 'enriched_title' in ar_full.columns
            else 'title'
        )
        review_cols = ['red_flag', title_col, 'verdict']
        if 'confidence' in ar_full.columns:
            review_cols.append('confidence')
        if 'description' in ar_full.columns:
            review_cols.append('description')
        if 'lot_link' in ar_full.columns:
            review_cols.append('lot_link')
        review_cols = [c for c in review_cols if c in ar_full.columns]
        review_df = ar_full.loc[flagged_mask, review_cols].copy()
        editor_sig = (
            st.session_state.get('current_auction', ''),
            flagged_count,
            tuple(review_df.index.tolist()[:50]),
        )
        editor_key = f"red_flag_review_{hash(editor_sig)}"

        edited_review = st.data_editor(
            review_df,
            use_container_width=True,
            hide_index=True,
            column_order=review_cols,
            disabled=[c for c in review_cols if c != 'red_flag'],
            column_config={
                "red_flag": st.column_config.CheckboxColumn(
                    "🚩 Flagged",
                    help="Uncheck to clear the flag and let this "
                         "item join the next comps batch.",
                    width="small",
                ),
                title_col: st.column_config.TextColumn(
                    "Title", width="large",
                ),
                "verdict": st.column_config.TextColumn(
                    "Reason", width="medium",
                ),
                "confidence": st.column_config.ProgressColumn(
                    "Confidence", min_value=0, max_value=100,
                    format="%.0f%%",
                ),
                "description": st.column_config.TextColumn(
                    "Description", width="large",
                    help="Original lot description — gives the "
                         "context the AI scored against.",
                ),
                "lot_link": st.column_config.LinkColumn(
                    "Open", display_text="🔗",
                ),
            },
            key=editor_key,
        )

        unchecked = review_df.index[
            review_df['red_flag'].fillna(False).astype(bool)
            & ~edited_review['red_flag'].fillna(False).astype(bool).values
        ]
        if len(unchecked) > 0:
            ar_full.loc[unchecked, 'red_flag'] = False
            st.session_state.audit_results = ar_full
            st.session_state._comps_has_more = True
            _save_current_auction_to_cache()
            st.success(
                f"✓ Unflagged {len(unchecked)} item(s). They'll be "
                "included in the next price-comps batch."
            )
            st.rerun()


def _compute_resale_confidence(row):
    """Score est_resale credibility on a 0-100 scale.

    Three independent signals get a 0-1 score and are then weighted:
      - Source quality (50% weight): sold > active. eBay+Mercari sold
        is ideal; PriceCharting is good when its product match is
        right (we can't verify that here); 'thin comps' is the
        pricer's own self-flag for 1-2 sold listings; 'active'
        fallback has no outlier protection at all.
      - Sample size (30% weight): more comps = more robust median.
        PriceCharting's single match counts as ~5-comp because their
        pricing is server-side aggregated across many transactions.
      - Spread tightness (20% weight): wider (high-low)/median ratio
        signals outlier listings dragging the median around.

    Returns None when est_resale is missing.
    """
    median = row.get('est_resale')
    if median is None or pd.isna(median):
        return None
    src = str(row.get('price_source') or '').lower()
    comp_count = int(row.get('comp_count') or 0)
    pc_comps = int(row.get('pricecharting_comps') or 0)
    gc_comps = int(row.get('gocollect_comps') or 0)
    low = row.get('price_low')
    high = row.get('price_high')
    low = low if pd.notna(low) else median
    high = high if pd.notna(high) else median

    # Source quality (0-1)
    if gc_comps > 0 or 'gocollect' in src:
        # GoCollect: curated CGC/BGS/SGC-graded data, matched at the
        # issue+grade level. Highest signal we have for graded comics.
        source_score = 0.95
    elif 'sold' in src and 'thin' not in src:
        source_score = 1.0 if 'ebay+mercari' in src else 0.9
    elif pc_comps > 0 or 'pricecharting' in src:
        source_score = 0.75
    elif 'thin' in src:
        source_score = 0.4
    elif 'active' in src:
        source_score = 0.25
    else:
        source_score = 0.5

    # Sample size (0-1)
    if gc_comps > 0:
        # GoCollect's single match is a curated issue+grade record
        # backed by Heritage / eBay / ComicConnect aggregation —
        # treat like a high-comp PC hit.
        sample_score = 0.9
    elif pc_comps > 0:
        sample_score = 0.85
    elif comp_count >= 5:
        sample_score = 1.0
    elif comp_count >= 4:
        sample_score = 0.9
    elif comp_count >= 3:
        sample_score = 0.75
    elif comp_count >= 2:
        sample_score = 0.55
    elif comp_count >= 1:
        sample_score = 0.35
    else:
        sample_score = 0.0

    # Spread tightness (0-1)
    if median and median > 0:
        spread_ratio = (high - low) / median
        if spread_ratio <= 0.1:
            spread_score = 1.0
        elif spread_ratio <= 0.25:
            spread_score = 0.9
        elif spread_ratio <= 0.5:
            spread_score = 0.75
        elif spread_ratio <= 1.0:
            spread_score = 0.55
        else:
            spread_score = 0.35
    else:
        spread_score = 0.7

    base_score = (source_score * 0.5 + sample_score * 0.3 + spread_score * 0.2)

    # Reality check: if the live current_bid is multiples of our est,
    # the market is loudly disagreeing with us. Our query probably
    # matched the wrong product / grade / variant. Knock confidence
    # down hard so the user can spot these in the table.
    try:
        bid = float(row.get('current_bid') or 0)
        if bid > 0 and median > 0:
            ratio = bid / median
            if ratio >= 10:
                base_score *= 0.25      # bid >= 10x est → est is almost certainly wrong
            elif ratio >= 5:
                base_score *= 0.4
            elif ratio >= 2.5:
                base_score *= 0.6
    except (TypeError, ValueError):
        pass

    return int(round(base_score * 100))


def _render_results_table(results_df):
    """Render the results table with live ROI/STR threshold highlighting.

    Rather than hiding items below threshold, rows are color-coded:
      - green = meets BOTH target ROI and target STR
      - yellow = meets ONE of them
      - no tint = below both thresholds (or missing data)

    Sorted by est_roi descending by default.
    """
    # --- Compact filter bar: ROI + STR inputs side-by-side. The 6-cell
    #     metric grid that used to live here is replaced with a one-line
    #     status caption below, after the masks are computed.
    tc1, tc2 = st.columns(2)
    with tc1:
        target_roi_val = st.number_input(
            "🎯 Target ROI ×",
            value=3.0, step=0.5, format="%.1f",
            min_value=1.0,
            help="Sell for Nx total cost (3x = sell for 3× what you paid). "
                 "Drives the green/yellow row tint and the Max Bid column. "
                 "Green rows meet both ROI + STR targets, yellow meet one.",
            key="target_roi_live",
        )
    with tc2:
        target_str_val = st.number_input(
            "🎯 Target eBay STR %",
            value=70.0, step=5.0, format="%.0f",
            min_value=0.0, max_value=100.0,
            help="Minimum eBay sell-through %. Higher = faster-selling items.",
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

    # --- Threshold masks (for highlight + status counts) ---
    # When STR data is unavailable for the entire auction (e.g. coin
    # auctions where STR scraping doesn't reliably resolve), don't
    # require STR for the green highlight. Fall back to ROI-only so
    # rows still flag profitable. We detect "no STR data available"
    # by checking if any row has a non-null STR value.
    roi_threshold = (target_roi_val - 1) * 100
    meets_roi = (
        working['est_roi'].notna() & (pd.to_numeric(working['est_roi'], errors='coerce') >= roi_threshold)
        if 'est_roi' in working.columns
        else pd.Series(False, index=working.index)
    )
    str_known_anywhere = (
        'ebay_str' in working.columns
        and working['ebay_str'].notna().any()
    )
    meets_str_mask = (
        working['ebay_str'].notna() & (pd.to_numeric(working['ebay_str'], errors='coerce') >= target_str_val)
        if 'ebay_str' in working.columns
        else pd.Series(False, index=working.index)
    )
    if str_known_anywhere:
        meets_both = meets_roi & meets_str_mask
        meets_either = (meets_roi | meets_str_mask) & ~meets_both
    else:
        # No STR data at all — promote ROI-only to "meets both" so the
        # green highlight still works on auctions where STR can't be
        # scraped (coins, bullion, niche collectibles).
        meets_both = meets_roi
        meets_either = pd.Series(False, index=working.index)

    # --- One-line status row (replaces the old 6-cell stMetric grid) ---
    status_bits = [f"**{len(working)}** leads"]
    if 'est_resale' in working.columns:
        status_bits.append(f"**{int(working['est_resale'].notna().sum())}** comped")
    status_bits.append(f"✅ **{int(meets_both.sum())}** meet both")
    status_bits.append(f"🟡 **{int(meets_either.sum())}** meet one")
    if 'red_flag' in working.columns:
        status_bits.append(f"🚩 **{int(working['red_flag'].sum())}** red-flagged")
    if 'ebay_str' in working.columns:
        has_str = int(working['ebay_str'].notna().sum())
        if has_str:
            avg_str = pd.to_numeric(working['ebay_str'], errors='coerce').mean()
            status_bits.append(f"avg STR **{avg_str:.0f}%**")
    st.caption(" · ".join(status_bits))

    filtered_df = working

    # Columns
    title_col = 'enriched_title' if 'enriched_title' in filtered_df.columns else 'title'
    # Lead with the lot thumbnail when we have one — Streamlit's ImageColumn
    # renders the URL inline so the user can scan visually before reading.
    display_cols = []
    if 'thumbnail_url' in filtered_df.columns:
        display_cols.append('thumbnail_url')
    display_cols.append(title_col)
    # When we're displaying enriched titles AND any row's enriched value
    # actually differs from the HiBid original, expose the original title
    # too so the user can see what was changed.
    show_original = (
        title_col == 'enriched_title'
        and 'title' in filtered_df.columns
        and (
            filtered_df['title'].fillna('').astype(str)
            != filtered_df['enriched_title'].fillna('').astype(str)
        ).any()
    )
    if show_original:
        display_cols.append('title')
    display_cols += ['lot_link', 'auction_link', 'category', 'current_bid']
    if 'next_bid' in filtered_df.columns:
        display_cols.append('next_bid')
    display_cols.append('est_cost')
    col_config = {
        "thumbnail_url": st.column_config.ImageColumn(
            "📷",
            help="Lot photo from HiBid (click to enlarge).",
            width="small",
        ),
        "enriched_title": st.column_config.TextColumn("Title (Enriched)"),
        "title": st.column_config.TextColumn(
            "Original Title",
            width="medium",
            help="The HiBid lot title before enrichment. Step 1 (audit) "
                 "and Step 1.5 (image match) can rewrite the title with "
                 "brand / model / year details pulled from the "
                 "description or matching eBay listings — this column "
                 "preserves what the lot actually said on HiBid.",
        ),
        "lot_link": st.column_config.LinkColumn("Item", display_text="Open"),
        "auction_link": st.column_config.LinkColumn("Auction", display_text="Open"),
        "current_bid": st.column_config.NumberColumn("Current Bid", format="$%.2f"),
        "next_bid": st.column_config.NumberColumn(
            "Next Bid", format="$%.2f",
            help="Minimum next acceptable bid (HiBid `lotState.minBid`). "
                 "Used as the cost-basis floor when current bid is $0.",
        ),
        "est_cost": st.column_config.NumberColumn(
            "Est. Cost", format="$%.2f",
            help="max(current_bid, next_bid) + buyer_premium_pct. "
                 "This is what you'd pay if you won at the minimum next bid.",
        ),
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

            # Numerical resale-confidence score (0-100). Replaces the
            # earlier binary low_comp_confidence flag with a richer
            # signal — see _compute_resale_confidence() for the formula
            # (source quality 50%, sample size 30%, spread tightness
            # 20%). Display as a ProgressColumn so the bar visualizes
            # the score at a glance.
            filtered_df = filtered_df.copy()
            filtered_df['resale_confidence'] = filtered_df.apply(
                _compute_resale_confidence, axis=1,
            )
            display_cols += ['resale_confidence']
            col_config["resale_confidence"] = st.column_config.ProgressColumn(
                "Conf.",
                min_value=0,
                max_value=100,
                format="%d%%",
                help=(
                    "Trust score for est_resale on a 0-100 scale. "
                    "Combines source quality (50% weight: eBay/Mercari "
                    "sold > PriceCharting > active fallback), sample "
                    "size (30%; PC's single match counts ~5-comp since "
                    "it's server-aggregated), and spread tightness "
                    "(20%; wider (high-low)/median means outliers). "
                    "Rough thresholds: >=80 high, 50-79 medium, <50 "
                    "rough estimate at best."
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

        # ----- Auctioneer-estimate fact-check -----
        # When the auctioneer published an estimate range, surface it
        # alongside our comp-derived est_resale and flag rows where the
        # comp is wildly outside the auctioneer's range. Wrapped in
        # try/except so a bad payload (mixed dtypes, weird estimate
        # strings) can't blank the whole results table.
        try:
            has_est_low = 'auctioneer_est_low' in filtered_df.columns
            has_est_high = 'auctioneer_est_high' in filtered_df.columns
            if has_est_low or has_est_high:
                filtered_df = filtered_df.copy()
                est_low = (pd.to_numeric(filtered_df['auctioneer_est_low'], errors='coerce')
                           if has_est_low else pd.Series(np.nan, index=filtered_df.index, dtype='float64'))
                est_high = (pd.to_numeric(filtered_df['auctioneer_est_high'], errors='coerce')
                            if has_est_high else pd.Series(np.nan, index=filtered_df.index, dtype='float64'))
                resale_n = pd.to_numeric(filtered_df['est_resale'], errors='coerce')
                # Convert to plain numpy arrays — element-wise math on
                # mixed-dtype Series can hit "Expected numeric dtype, got
                # object" the same way other parts of this app did.
                est_low_arr = est_low.to_numpy(dtype='float64', na_value=np.nan)
                est_high_arr = est_high.to_numpy(dtype='float64', na_value=np.nan)
                resale_arr = resale_n.to_numpy(dtype='float64', na_value=np.nan)

                verdicts = []
                for low, high, resale in zip(est_low_arr, est_high_arr, resale_arr):
                    if np.isnan(resale):
                        verdicts.append('')
                        continue
                    low_ok = not np.isnan(low)
                    high_ok = not np.isnan(high)
                    if not low_ok and not high_ok:
                        verdicts.append('—')
                        continue
                    lo = low if low_ok else high
                    hi = high if high_ok else low
                    if lo > 0 and resale < lo * 0.5:
                        verdicts.append('⚠ comp <50% of est')
                    elif hi > 0 and resale > hi * 2.0:
                        verdicts.append('⚠ comp >2× est')
                    elif (lo <= 0 or resale >= lo * 0.8) and (hi <= 0 or resale <= hi * 1.5):
                        verdicts.append('✓ in range')
                    else:
                        verdicts.append('near range')

                filtered_df['comp_vs_est'] = verdicts
                if has_est_low:
                    display_cols += ['auctioneer_est_low']
                    col_config['auctioneer_est_low'] = st.column_config.NumberColumn(
                        "Auct. Low", format="$%.2f",
                        help="Auctioneer's published low estimate.",
                    )
                if has_est_high:
                    display_cols += ['auctioneer_est_high']
                    col_config['auctioneer_est_high'] = st.column_config.NumberColumn(
                        "Auct. High", format="$%.2f",
                        help="Auctioneer's published high estimate.",
                    )
                display_cols += ['comp_vs_est']
                col_config['comp_vs_est'] = st.column_config.TextColumn(
                    "Comp vs. Est.",
                    help=(
                        "Sanity check between the comp-derived est_resale "
                        "and the auctioneer's published estimate range. "
                        "'⚠' rows have comp values that disagree strongly "
                        "— verify before bidding."
                    ),
                )
        except Exception as _est_err:
            st.caption(f"_(comp-vs-estimate fact-check skipped: {_est_err})_")

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
        col_config["verdict"] = st.column_config.TextColumn(
            "Flag Reason",
            help=(
                "Why the audit flagged the item. "
                "'Unshippable (HARD logistics)' = pickup-only or oversized. "
                "'broken, damaged, or for parts' = AI flagged the description "
                "as condition risk. 'untested or unknown condition' = "
                "description suggests the seller couldn't verify it works. "
                "'Unknown' = description was empty or too short to classify."
            ),
        )
        col_config["confidence"] = st.column_config.ProgressColumn(
            "Audit Conf.",
            min_value=0, max_value=100, format="%.1f%%",
            help="The AI condition-audit's confidence in its verdict "
                 "(see 'Flag Reason' column). Distinct from the "
                 "'Resale Conf.' column which scores trust in the "
                 "est_resale price. Often 0 for collectible-bypass "
                 "lots (cards/comics/games) where the audit is "
                 "skipped on purpose.",
        )
        col_config["red_flag"] = st.column_config.CheckboxColumn("Red Flag")

    final_cols = [c for c in display_cols if c in filtered_df.columns]

    # --- Column show/hide controls ---
    # Auto-detect columns that have no useful data on this auction
    # (all-null or all-zero) and default-hide them. The user can
    # toggle visibility per column from the expander above the table.
    # Common case this addresses: coin auctions where STR scraping
    # returns nothing, or audit was bypassed so 'confidence' is 0.
    def _column_is_empty(col: str) -> bool:
        if col not in filtered_df.columns:
            return True
        s = filtered_df[col]
        if s.isna().all():
            return True
        # Numeric-like check: treat all-zero as empty too. Don't apply
        # to text columns where '0' could be a real category label.
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            try:
                numeric = pd.to_numeric(s, errors='coerce').fillna(0)
                if (numeric == 0).all():
                    return True
            except (TypeError, ValueError):
                pass
        return False

    # Friendly labels for the checkbox UI. Falls back to the column
    # name when not specified.
    _col_labels = {
        'thumbnail_url': '📷 Image',
        'enriched_title': 'Title (Enriched)',
        'title': 'Original Title',
        'lot_link': 'Item link',
        'auction_link': 'Auction link',
        'category': 'Category',
        'current_bid': 'Current Bid',
        'next_bid': 'Next Bid',
        'est_cost': 'Est. Cost',
        'est_resale': 'Est. Resale',
        'price_low': 'Price Low',
        'price_high': 'Price High',
        'comp_count': 'Comp Count',
        'ebay_comps': 'eBay Comps',
        'mercari_comps': 'Mercari Comps',
        'pricecharting_comps': 'PC Hit',
        'gocollect_comps': 'GoCollect Hit',
        'price_source': 'Price Source',
        'resale_confidence': 'Resale Conf.',
        'ebay_str': 'STR %',
        'str_source': 'STR Source',
        'bid_count': 'Bids',
        'verdict': 'Flag Reason',
        'confidence': 'Audit Conf.',
        'red_flag': 'Red Flag',
        'est_roi': 'ROI %',
        'max_bid': 'Max Bid',
        'low_comp_confidence': 'Low Conf. (legacy)',
    }
    # Persist user picks across reruns. Key includes a fingerprint of
    # the current column set so a different auction's columns don't
    # inherit a stale picks set.
    _empty_cols = {c: _column_is_empty(c) for c in final_cols}
    n_empty = sum(_empty_cols.values())
    n_visible = len(final_cols) - n_empty
    with st.expander(
        f"📋 Show / hide columns "
        f"({n_visible} visible · {n_empty} auto-hidden as empty)",
        expanded=False,
    ):
        st.caption(
            "Empty columns (all-null or all-zero) are auto-hidden by "
            "default. Tick a checkbox to force-show one anyway."
        )
        cb_cols = st.columns(4)
        kept_cols = []
        for i, c in enumerate(final_cols):
            with cb_cols[i % 4]:
                empty = _empty_cols[c]
                label = _col_labels.get(c, c)
                if empty:
                    label = f"{label}  _(empty)_"
                # Default: hide if empty, show otherwise. The state
                # key is per-column so picks persist across reruns
                # but reset cleanly between auctions (different
                # current_auction → different audit_results df shape).
                shown = st.checkbox(
                    label,
                    value=not empty,
                    key=f"colshow_{c}",
                )
                if shown:
                    kept_cols.append(c)
        final_cols = kept_cols

    if not final_cols:
        st.info("All columns hidden — tick at least one in the expander above.")
        return

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

    # --- Export controls: share results for debugging ---
    # Two formats. CSV is the full thing (every column, attach as a file
    # in chat or open in a spreadsheet). The Markdown snippet is a
    # paste-friendly view of the visible columns + a sample of rows —
    # useful when only a few lots look wrong and you want to ask a
    # specific question without sharing the whole sheet.
    with st.expander("📤 Export results", expanded=False):
        auction_slug = re.sub(
            r'[^a-z0-9]+', '-',
            (st.session_state.get('current_auction') or 'auction').lower(),
        ).strip('-')[:50] or 'auction'
        timestamp = datetime.now().strftime('%Y%m%d-%H%M')
        csv_name = f"htown-results-{auction_slug}-{timestamp}.csv"

        ec1, ec2 = st.columns([1, 1])
        with ec1:
            # Full audit_results — every column. Use the styled-table's
            # working df so est_roi / max_bid reflect current target_roi.
            st.download_button(
                label="📥 Download full CSV",
                data=working.to_csv(index=False).encode('utf-8'),
                file_name=csv_name,
                mime="text/csv",
                key="export_results_csv",
                help="All columns (verdict, comps, ROI, max_bid, links, image-enrichment fields). "
                     "Attach this in chat to share the full results.",
                use_container_width=True,
            )
        with ec2:
            # Snippet pre-trimmed to the columns most useful for triage —
            # paste straight into chat without attaching a file.
            snippet_n = st.number_input(
                "Snippet rows", min_value=5, max_value=100, step=5,
                value=20, key="export_snippet_n",
                help="How many rows to include in the markdown snippet below.",
            )
        st.caption(
            "💡 If specific lots look wrong, share the Markdown snippet. "
            "If you want me to look at the whole auction, attach the CSV."
        )

        snippet_cols = [
            c for c in (
                'title', 'enriched_title', 'category', 'current_bid',
                'est_resale', 'price_low', 'price_high',
                'ebay_comps', 'mercari_comps', 'pricecharting_comps',
                'price_source', 'est_roi', 'ebay_str',
                'verdict', 'red_flag', 'audit_source',
                'img_source', 'lot_link',
            ) if c in working.columns
        ]
        snippet_df = working[snippet_cols].head(int(snippet_n)).copy()
        # Trim long titles so the markdown table doesn't get unreadable
        for col in ('title', 'enriched_title'):
            if col in snippet_df.columns:
                snippet_df[col] = (
                    snippet_df[col].fillna('').astype(str).str.slice(0, 60)
                )
        try:
            md_table = snippet_df.to_markdown(index=False)
        except (ImportError, AttributeError):
            # `tabulate` package may not be installed — fall back to CSV-as-code
            md_table = "```csv\n" + snippet_df.to_csv(index=False) + "```"

        st.code(md_table, language="markdown")


# ================================================================
# SINGLE-PAGE UI: switches between Discovery view and Analysis view
# ================================================================
# Auto-skip the per-auction Discovery results page: when lots have just
# been fetched (phase1_leads filled) but no auction has been loaded into
# analysis yet, jump straight into the analysis view for the fetched
# auction. The picker only fetches one auction per click, so this is
# unambiguous.
if (
    not st.session_state.phase1_leads.empty
    and not st.session_state.get('current_auction')
):
    _df = st.session_state.phase1_leads
    if 'auction' in _df.columns and len(_df):
        _auction_name = _df['auction'].iloc[0]
        _auction_df = _df[_df['auction'] == _auction_name].reset_index(drop=True)
        _load_auction_for_analysis(_auction_name, _auction_df)
        st.rerun()

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
            # Also clear phase1_leads so the auto-load step above doesn't
            # immediately bounce us back into analysis — we want to land
            # on the picker (Selection view) instead.
            st.session_state.phase1_leads = pd.DataFrame()
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

    # Hoisted "running" flags so each tab can disable buttons consistently
    # regardless of which tab the user has open.
    audit_running = st.session_state.get('audit_running', False)
    comps_running = st.session_state.get('comps_running', False)
    img_enrich_running = st.session_state.get('img_enrich_running', False)

    # ================================================================
    # AUTO-PIPELINE TRIGGER (early-fire, no UI flicker)
    # Runs BEFORE any heavy rendering so that the auto-`st.rerun()`
    # below doesn't briefly flash the empty placeholders before
    # transitioning into the audit/comps status block.
    # ================================================================
    # "comps_data" used to mean "ANY est_resale present", but cached
    # auctions can have 1-3 stale values from earlier runs that confuse
    # the auto-pipeline into skipping a fresh comps pass. Better signal:
    # at least 30% of audit-eligible (good+, non-HARD, has-thumbnail)
    # lots must have a comp value before we treat comps as "done".
    has_comps_data = False
    ar_state = st.session_state.get('audit_results')
    if (isinstance(ar_state, pd.DataFrame)
            and 'est_resale' in ar_state.columns
            and not ar_state.empty):
        eligible_mask = (
            ~ar_state.get('red_flag', pd.Series(False, index=ar_state.index)).fillna(False).astype(bool)
            & (ar_state.get('logistics_ease', pd.Series('', index=ar_state.index)) != 'HARD')
        )
        eligible_count = int(eligible_mask.sum())
        if eligible_count > 0:
            comped_count = int(
                ar_state.loc[eligible_mask, 'est_resale'].notna().sum()
            )
            has_comps_data = (comped_count / eligible_count) >= 0.30

    # Detect "stale audit" — when a cache-restored auction has the audit
    # columns but most rows are `audit_source=no_api_key` (because the
    # cache was made before the API key was wired up). Those rows show
    # verdict=Unknown / confidence=0 across the whole auction, and items
    # that should be red-flagged ("broken", "untested", "for parts")
    # leak into the comps stage and waste lookups.
    audit_looks_stale = False
    if (isinstance(ar_state, pd.DataFrame)
            and 'audit_source' in ar_state.columns
            and len(ar_state) > 0):
        no_key_count = int(
            (ar_state['audit_source'].fillna('') == 'no_api_key').sum()
        )
        # Only consider it "stale" if our config actually has an API key
        # to use — otherwise re-running won't help.
        try:
            from scraper.config_loader import load_config
            _cfg = load_config()
            _has_key = bool((_cfg.get("anthropic") or {}).get("api_key"))
        except Exception:
            _has_key = False
        if _has_key and (no_key_count / len(ar_state)) >= 0.50:
            audit_looks_stale = True

    auto_attempts: set = st.session_state.setdefault(
        '_auto_pipeline_attempts', set()
    )
    has_more_chunks = bool(st.session_state.get('_comps_has_more'))

    # Detect generic-titled lots that would benefit from image enrichment
    # before comps run. We check the post-audit state so red-flagged and
    # HARD-logistics lots are already filtered out.
    needs_image_enrich = False
    if has_audit and 'img_enrich' not in auto_attempts:
        ar_check = st.session_state.audit_results
        if isinstance(ar_check, pd.DataFrame) and 'thumbnail_url' in ar_check.columns:
            mask_eligible = (
                ~ar_check.get('red_flag', pd.Series(False, index=ar_check.index)).fillna(False).astype(bool)
                & (ar_check.get('logistics_ease', pd.Series('', index=ar_check.index)) != 'HARD')
                & ar_check['thumbnail_url'].fillna('').astype(bool)
            )
            # Skip lots that already have a Claude/eBay-promoted title
            if 'img_enriched_title' in ar_check.columns:
                mask_eligible &= ar_check['img_enriched_title'].isna()
            generic_count = 0
            for title in ar_check.loc[mask_eligible].apply(
                lambda r: r.get('enriched_title') or r.get('title') or '', axis=1
            ):
                if _is_generic_title(str(title)):
                    generic_count += 1
            needs_image_enrich = generic_count >= 1

    if not (audit_running or comps_running or img_enrich_running):
        # Stale-audit retry: if the cached audit columns are mostly
        # `no_api_key` and we now DO have a key, re-fire the audit so
        # genuinely-broken/untested items get red-flagged before they
        # reach the comps stage. Tracked separately from the regular
        # 'audit' attempt so a one-time stale retry doesn't loop.
        if (has_audit and audit_looks_stale
                and 'audit_stale_retry' not in auto_attempts):
            auto_attempts.add('audit_stale_retry')
            auto_attempts.discard('audit')  # let audit step re-run
            st.session_state.audit_running = True
            st.rerun()
        elif not has_audit and 'audit' not in auto_attempts:
            auto_attempts.add('audit')
            st.session_state.audit_running = True
            st.rerun()
        elif has_audit and needs_image_enrich:
            # Auto-fire image enrichment between audit and comps so lots
            # with generic titles ("Box of misc", "Vintage vase") get a
            # vision-based identification before we hit eBay/Mercari.
            auto_attempts.add('img_enrich')
            st.session_state.img_enrich_running = True
            st.rerun()
        elif has_audit and not has_comps_data and 'comps_first' not in auto_attempts:
            auto_attempts.add('comps_first')
            st.session_state.comps_running = True
            st.rerun()
        elif has_audit and has_comps_data and has_more_chunks:
            # Auto-continue chunks. _comps_has_more flips False when we
            # genuinely run out, so this self-terminates.
            st.session_state.comps_running = True
            st.rerun()

    # ================================================================
    # PIPELINE-ACTIVE GATE
    # When the auto-pipeline is mid-flight (audit running, comps
    # running, image enrichment running, OR comp chunks still pending),
    # we hide the results panel entirely and show only a "🍳 Cooking…"
    # screen with the audit/comps status panels. The user said the
    # streaming-results layout was wonky — they prefer to wait and see
    # final results all at once. The split layout + tabs render below
    # only when the pipeline is fully done.
    # ================================================================
    # img_enrich_running is included now that the auto-pipeline fires
    # image enrichment between audit and comps for generic-titled lots.
    # The execution block is hoisted below alongside audit/comps so the
    # cooking screen covers all three steps uniformly.
    pipeline_active = (
        audit_running or comps_running or img_enrich_running or has_more_chunks
    )

    # No-op stubs for live-update hooks — used by the audit/comps
    # execution blocks below. During cooking we don't render results,
    # so these stubs make the live_callback hooks safe to call without
    # an active placeholder.
    class _NullPlaceholder:
        def markdown(self, *a, **k):
            pass

        def container(self):
            from contextlib import nullcontext
            return nullcontext()

    last_priced_placeholder = _NullPlaceholder()

    def _render_live_results():
        # No-op during cooking. Real rendering happens inline in the
        # `if not pipeline_active` block below once everything finishes.
        pass

    def _render_red_flag_review():
        pass

    if pipeline_active:
        # While cooking, show a heading. The audit/comps st.status
        # blocks below render their progress directly underneath.
        st.markdown("### 🍳 Cooking your auction…")
        st.caption(
            "Audit + price comps run automatically. Final results "
            "appear once everything completes — hang tight."
        )

    # ================================================================
    # AUDIT-RUNNING execution block — hoisted out of tab_audit so the
    # live st.status updates show in the main flow regardless of which
    # tab the user has open.
    # ================================================================
    if audit_running:
        _keep_screen_awake()
        st.info(
            "🔋 Keeping screen awake while this runs. "
            "If your phone still locks, set **Auto-Lock → Never** in your phone's "
            "display settings before kicking off long runs."
        )
        try:
            import time as _time
            _audit_last = [0.0]
            def _on_audit_progress(processed, total_items):
                now = _time.time()
                if (now - _audit_last[0]) < 0.3 and processed < total_items:
                    return
                _audit_last[0] = now
                _render_live_results()

            st.session_state.audit_results = _run_ai_audit(
                leads_df, on_progress=_on_audit_progress,
            )
            _save_current_auction_to_cache()
        except Exception as e:
            st.error(f"Audit failed: {e}")
        finally:
            st.session_state.audit_running = False
        st.rerun()

    # ================================================================
    # IMAGE-ENRICHMENT execution block — hoisted out of tab_audit so the
    # auto-pipeline can run it between audit and comps for generic-titled
    # lots. The cooking screen covers it via pipeline_active.
    # ================================================================
    if img_enrich_running:
        _keep_screen_awake()
        try:
            st.session_state.audit_results = _run_image_enrichment(
                st.session_state.audit_results,
                min_bid=float(
                    st.session_state.get('img_enrich_min_bid', 0.0) or 0.0
                ),
            )
            _save_current_auction_to_cache()
        except Exception as e:
            import traceback
            st.error(f"Image enrichment failed: {e}")
            with st.expander("Show traceback"):
                st.code(traceback.format_exc(), language="python")
        finally:
            st.session_state.img_enrich_running = False
        st.rerun()

    # ================================================================
    # COMPS-RUNNING execution block — same hoist as audit. Reads filter
    # values from session_state (set by the Comps tab settings panel
    # below) so it's decoupled from tab rendering.
    # ================================================================
    if comps_running:
        _keep_screen_awake()
        st.info(
            "🔋 Keeping screen awake while this runs. "
            "If your phone still locks, set **Auto-Lock → Never** in your phone's "
            "display settings before kicking off long runs."
        )
        ar = st.session_state.audit_results
        chunk_size = int(st.session_state.get('comps_chunk_size', 200))
        any_comped = (
            isinstance(ar, pd.DataFrame)
            and 'est_resale' in ar.columns
            and ar['est_resale'].notna().any()
        )
        try:
            # First chunk on a fresh auction? Reset the cached STR map so
            # we re-sample for the new auction. Also reset the scrape-
            # stats counters so the "scraper blocked" warning below
            # reflects this run only, not lifetime.
            if not any_comped:
                st.session_state.pop('_comps_auction_str_map', None)
                from scraper.ebay_prices import EbayPriceLookup
                EbayPriceLookup.reset_scrape_stats()

            # Two streams of updates per lot:
            #   - Per-lot ticker (no throttle): a one-line summary that
            #     updates instantly. Cheap, makes "live" obvious.
            #   - Full table re-render (throttled to 300ms).
            import time as _time
            _last = [0.0]

            def _on_lot_priced(completed, total_items, last_lot=None):
                if last_lot is not None:
                    title = (last_lot.get('title') or '')[:60]
                    resale = last_lot.get('resale')
                    roi = last_lot.get('roi')
                    comps_n = (
                        int(last_lot.get('ebay_comps') or 0)
                        + int(last_lot.get('mercari_comps') or 0)
                    )
                    bits = [f"🔥 **{completed}/{total_items}** priced"]
                    if title:
                        if pd.notna(resale):
                            detail = f"*{title}* → **${float(resale):.2f}**"
                            if pd.notna(roi):
                                detail += f" (ROI {int(roi)}%)"
                            if comps_n:
                                detail += f" · {comps_n} comps"
                            bits.append(detail)
                        else:
                            bits.append(f"*{title}* → no comps found")
                    try:
                        last_priced_placeholder.markdown(" · ".join(bits))
                    except Exception:
                        pass
                now = _time.time()
                if (now - _last[0]) < 0.3 and completed < total_items:
                    return
                _last[0] = now
                _render_live_results()

            updated, found, processed, has_more = _run_ebay_comps_chunk(
                ar, chunk_size=chunk_size,
                on_lot_priced=_on_lot_priced,
            )
            st.session_state.audit_results = updated
            st.session_state._comps_has_more = has_more
            # Accumulate per-batch stats so the post-pipeline view can show
            # exactly what happened (helps explain "nothing highlights"
            # without forcing the user to dig through st.status blocks).
            stats_total = st.session_state.setdefault('_comps_stats', {
                'batches': 0, 'attempted': 0, 'priced': 0,
                'last_msg': '', 'has_more': False,
            })
            stats_total['batches'] += 1
            stats_total['attempted'] += processed
            stats_total['priced'] += found
            stats_total['has_more'] = has_more
            _save_current_auction_to_cache()
            if processed:
                tail = ("  More lots remain — continuing automatically…"
                        if has_more else
                        "  ✅ All eligible lots have been comped.")
                stats_total['last_msg'] = (
                    f"Batch complete — priced {found}/{processed} lot(s).{tail}"
                )
                st.success(stats_total['last_msg'])
            else:
                good_count = int(
                    (~ar.get('red_flag', pd.Series([], dtype=bool))
                       .fillna(False)).sum()
                ) if isinstance(ar, pd.DataFrame) else 0
                try:
                    preview, _, summary = _apply_comps_filters(
                        ar[~ar['red_flag'].fillna(False)]
                        if isinstance(ar, pd.DataFrame) and 'red_flag' in ar.columns
                        else ar
                    )
                    preview_count = len(preview)
                except Exception:
                    preview_count = 0
                    summary = "(filter preview failed)"
                msg = (
                    f"No eligible lots to comp this batch. "
                    f"**{good_count}** good+ lots in the audit, "
                    f"**{preview_count}** survive the current filters "
                    f"({summary}). "
                    "Loosen the filters in the **💰 Comps** tab "
                    "(*Narrow down what to comp* expander) and use the "
                    "manual run button to retry."
                )
                st.warning(msg)
                stats_total = st.session_state.setdefault('_comps_stats', {
                    'batches': 0, 'attempted': 0, 'priced': 0,
                    'last_msg': '', 'has_more': False,
                })
                stats_total['last_msg'] = msg
                stats_total['has_more'] = False
        except Exception as e:
            err = f"Price comps failed: {e}"
            st.error(err)
            stats_total = st.session_state.setdefault('_comps_stats', {
                'batches': 0, 'attempted': 0, 'priced': 0,
                'last_msg': '', 'has_more': False,
            })
            stats_total['last_msg'] = err
            stats_total['has_more'] = False
        finally:
            st.session_state.comps_running = False

        # Surface scraper-blocked status. Both eBay sold and Mercari
        # silently fall through to "active eBay listings" when their
        # anti-bot trips a 403 — the user otherwise sees no signal that
        # their entire sold-history pipeline is dead and every priced
        # row in the table came from an untrustworthy active fallback.
        try:
            from scraper.ebay_prices import EbayPriceLookup
            stats = EbayPriceLookup.get_scrape_stats()
            ebay_total = stats.get('ebay_sold_attempts', 0)
            ebay_blocked = stats.get('ebay_sold_blocked', 0)
            merc_total = stats.get('mercari_attempts', 0)
            merc_blocked = stats.get('mercari_blocked', 0)
            sb_calls = stats.get('scrapingbee_calls', 0)
            sb_credits = stats.get('scrapingbee_credits', 0)
            ebay_block_rate = (ebay_blocked / ebay_total) if ebay_total else 0.0
            merc_block_rate = (merc_blocked / merc_total) if merc_total else 0.0

            # ScrapingBee-specific errors get top billing — a quota or
            # auth failure means the proxy isn't even reaching eBay,
            # so the lower-level "scraper blocked" message would be
            # misleading.
            sb_quota = stats.get('scrapingbee_quota_fail', 0)
            sb_auth = stats.get('scrapingbee_auth_fail', 0)
            if sb_quota > 0:
                st.error(
                    f"💸 **ScrapingBee plan is exhausted** — "
                    f"{sb_quota} request(s) returned 'Monthly API "
                    "calls limit reached'. eBay/Mercari sold scraping "
                    "is dead until your plan resets or you upgrade. "
                    "Visit https://app.scrapingbee.com/account/usage "
                    "to check your quota or upgrade to the Freelance "
                    "tier ($49/mo for 100k credits)."
                )
            elif sb_auth > 0:
                st.error(
                    f"🔑 **ScrapingBee auth failed** on {sb_auth} "
                    "request(s) — your `scrapingbee.api_key` in "
                    "config.json is likely wrong. Re-copy the key "
                    "from https://app.scrapingbee.com/account."
                )

            # Positive signal: ScrapingBee was used and the eBay sold
            # path is actually getting through. Show a small caption
            # so the user can track credit burn against their plan.
            if sb_calls > 0:
                st.caption(
                    f"🐝 ScrapingBee used for {sb_calls} scrape(s) "
                    f"(~{sb_credits} credits)."
                )

            if (ebay_total >= 3 and ebay_block_rate > 0.5) or \
               (merc_total >= 3 and merc_block_rate > 0.5):
                bits = []
                if ebay_block_rate > 0.5:
                    bits.append(
                        f"**eBay sold** blocked on {ebay_blocked}/{ebay_total} "
                        f"requests ({int(ebay_block_rate * 100)}%)"
                    )
                if merc_block_rate > 0.5:
                    bits.append(
                        f"**Mercari sold** blocked on {merc_blocked}/{merc_total} "
                        f"requests ({int(merc_block_rate * 100)}%)"
                    )
                st.error(
                    "🚫 **Sold-listing scrapers are being blocked** — "
                    + "; ".join(bits) + ". "
                    "Most prices in this run came from eBay's **active** "
                    "listings (asking-prices) instead of real sold "
                    "history, which is much less reliable. The Conf "
                    "column will rate active-source rows in the 25–50% "
                    "range so they stand out. Long-term fix: apply for "
                    "eBay's Marketplace Insights API (free, gated) — "
                    "uses the same OAuth as your existing Browse API "
                    "key. See https://developer.ebay.com/api-docs/buy/"
                    "marketplace-insights/static/overview.html"
                )
        except Exception:
            pass  # Stats reporting is best-effort

        st.rerun()

    # ================================================================
    # PIPELINE-DONE rendering: split layout with results + tabs.
    # Skipped while cooking — see PIPELINE-ACTIVE GATE comment above.
    # The audit/comps execution blocks above end with st.rerun(), so
    # this rendering only runs once the pipeline is fully complete.
    # ================================================================
    if pipeline_active:
        # Cooking — st.status panels above are doing the talking.
        # Skip the rest of the analysis-view rendering entirely.
        st.stop()

    # ----- Full-width results panel -----
    # Auction list lives in the Streamlit sidebar (left). The main content
    # area is now dedicated entirely to the results table — manual
    # audit/comps overrides are tucked into a collapsed expander below
    # since the auto-pipeline runs them without user input.
    st.markdown("### 📊 Results")

    # Surface the most recent comps-batch outcome so the user can see
    # exactly what happened (priced X/Y, filter dropped everything, an
    # exception, etc.) without scrolling through the collapsed st.status
    # panels above. This persists across reruns until a new auction loads.
    _comp_stats = st.session_state.get('_comps_stats') or {}
    if _comp_stats:
        if _comp_stats.get('attempted'):
            tail = ("  More pending — auto-continuing…"
                    if _comp_stats.get('has_more') else "")
            st.success(
                f"💰 Comps: priced **{_comp_stats['priced']}/"
                f"{_comp_stats['attempted']}** lot(s) across "
                f"**{_comp_stats['batches']}** batch(es).{tail}"
            )
        elif _comp_stats.get('last_msg'):
            st.warning(_comp_stats['last_msg'])

    ar = st.session_state.get('audit_results')
    if isinstance(ar, pd.DataFrame) and not ar.empty:
        try:
            _render_results_table(ar)
        except Exception as e:
            import traceback
            st.error(f"⚠️ Failed to render results table: {e}")
            with st.expander("Show traceback"):
                st.code(traceback.format_exc(), language="python")
            # Fall back to a plain table so the user can still see something
            st.dataframe(ar, use_container_width=True)
        if 'red_flag' in ar.columns and ar['red_flag'].fillna(False).any():
            try:
                _render_red_flag_editor(ar)
            except Exception as e:
                st.warning(f"Red-flag editor unavailable: {e}")
    else:
        st.info(
            "Audit hasn't produced any rows yet. Try the manual "
            "**🔄 Re-run audit** button in the controls below."
        )

    # ----- Manual overrides (collapsed) -----
    # Audit + comps run automatically when an auction loads. These tabs
    # are escape hatches: re-run after editing flags, change batch size,
    # tweak comp filters, run image enrichment, etc.
    st.markdown("---")
    _ctrls_expander = st.expander(
        "⚙️ Manual controls — re-run audit / comps / image enrichment",
        expanded=False,
    )
    with _ctrls_expander:
        tab_audit, tab_comps = st.tabs(["🛡️ Audit", "💰 Comps"])

    with tab_audit:
        # Step 1: AI audit
        st.markdown("### Step 1: AI Condition Audit")

        if has_audit:
            ar = st.session_state.audit_results
            good_count = (~ar['red_flag']).sum()
            flagged_count = ar['red_flag'].sum()
            st.success(f"Audit complete — **{good_count} good+** condition, {flagged_count} red-flagged")

        audit_btn_label = "⏳ Running audit…" if audit_running else "🔄 Re-run audit"

        # ---- Audit knobs ----
        with st.expander("⚙️ Audit settings (optional)", expanded=False):
            st.slider(
                "Parallel workers",
                min_value=1, max_value=16, step=1,
                key="audit_workers",
                help="Concurrent Claude API calls during the AI tier. "
                     "Default 8 — fast without tripping rate limits. "
                     "Drop to 1-2 if you see HTTP 429s; push to 12-16 on "
                     "a high-tier API plan.",
            )

        if st.button(
            audit_btn_label,
            type="secondary",
            use_container_width=True,
            disabled=audit_running or comps_running,
            key="run_audit_btn",
            help="Manual override — the audit normally auto-fires when you "
                 "load an auction. Click this to re-run it.",
        ):
            # Manual click — clear the auto-pipeline tracker so the
            # forced re-run isn't blocked by the "already attempted" guard.
            st.session_state.pop('_auto_pipeline_attempts', None)
            st.session_state.audit_running = True
            st.rerun()

        # The audit-running execution block has been hoisted out of this
        # tab to the top-level analysis view (so the live st.status panel
        # is visible regardless of which tab the user is on). See the
        # auto-pipeline section above tabs.

        # Step 1.5: Image-based title enrichment (optional; improves Step 2 quality)
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

            # The image-enrichment execution block has been hoisted out of
            # this tab to the top-level analysis view (so the live
            # st.status panel is visible regardless of which tab the user
            # is on). See the auto-pipeline section above tabs.

        if has_audit and not (audit_running or img_enrich_running):
            st.caption("✅ Audit done — comps auto-fire next.")

    with tab_comps:
        # Step 2: eBay + Mercari comps (only after audit)
        st.markdown("### Step 2: eBay + Mercari Price Comps & STR")

        if not has_audit:
            st.info(
                "Audit hasn't completed yet — comps auto-fire as soon as it does. "
                "Hold on…"
            )
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
                    st.slider(
                        "Comp batch size",
                        min_value=50, max_value=5000, step=50,
                        key="comps_chunk_size",
                        help="Process eligible lots N at a time. Default "
                             "5000 = whole auction in one batch (the results "
                             "table is hidden between batches anyway). Lower "
                             "this only if you specifically want to see "
                             "partial progress in chunks.",
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

            # Decide which button label to show. The first chunk uses
            # "Run Price Comps"; subsequent chunks use "Continue with next
            # batch". `_comps_has_more` is set after each chunk completes.
            chunk_size = int(st.session_state.get('comps_chunk_size', 200))
            has_more = st.session_state.get('_comps_has_more', False)
            any_comped = (
                'est_resale' in ar.columns
                and ar['est_resale'].notna().any()
            )
            if comps_running:
                comps_btn_label = "⏳ Running price comps…"
            else:
                comps_btn_label = "🔄 Re-run comps from scratch"

            if st.button(
                comps_btn_label,
                type="secondary",
                use_container_width=True,
                disabled=audit_running or comps_running or img_enrich_running,
                key="run_comps_btn",
                help="Wipe every existing comp value on this auction and "
                     "re-comp from scratch with the current filters. Use "
                     "after upgrading the classifier or fixing a bad rule "
                     "— stale est_resale rows would otherwise be skipped "
                     "by the 'pending lots' gate.",
            ):
                # Clear all comp data so the chunked runner sees every
                # non-red-flagged row as 'pending' again. Without this
                # the est_resale.isna() & ~red_flag gate inside
                # _run_ebay_comps_chunk would skip every row that already
                # has a stale value.
                ar_current = st.session_state.audit_results
                if isinstance(ar_current, pd.DataFrame):
                    for col in (*_COMP_COLUMNS, 'est_roi', 'max_bid'):
                        if col in ar_current.columns:
                            ar_current[col] = None
                    st.session_state.audit_results = ar_current
                    _save_current_auction_to_cache()
                # Reset the chunk-pipeline tracker so the forced re-run
                # isn't blocked by the 'already attempted' guard, and the
                # cached STR map is dropped so we re-sample for the new
                # comp pass.
                st.session_state.pop('_auto_pipeline_attempts', None)
                st.session_state.pop('_comps_has_more', None)
                st.session_state.pop('_comps_auction_str_map', None)
                st.session_state.comps_running = True
                st.rerun()

            # The comps-running execution block has been hoisted out of
            # this tab to the top-level analysis view (so the live
            # st.status panel is visible regardless of which tab the
            # user is on). See the auto-pipeline section above tabs.

            if any_comped and not comps_running:
                st.caption("📊 Priced lots stream live to the right panel.")

# ---- DEFAULT VIEW: pick an auction from the sidebar ----
else:
    if st.session_state.get("auction_candidates"):
        st.info("""👈 Pick an auction from the sidebar to analyze.
Audit + price comps run automatically once you click one.""")
    else:
        st.info(
            "No auctions discovered yet. Click 🔍 Discover in the top right "
            "to fetch the open-auction list. (We do this automatically on first page load.)"
        )
