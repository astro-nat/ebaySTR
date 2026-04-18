import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import asyncio
import os
from datetime import datetime

# --- IMPORT MODULES ---
from scraper import Phase1Scraper
from scraper.cache import AuctionCache, merge_cached_analysis

# Single shared cache instance; auto-creates the dir on first touch
_AUCTION_CACHE = AuctionCache()

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="H-Town TX Finds: ROI Engine",
    page_icon="🛰️",
    layout="wide"
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

/* ---- Mid-size tablets ---- */
@media (max-width: 960px) {
    [data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important;
    }
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        flex: 1 1 48% !important;
        min-width: 48% !important;
    }
}
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

if 'comps_running' not in st.session_state:
    st.session_state.comps_running = False

if 'cache_ttl_days' not in st.session_state:
    st.session_state.cache_ttl_days = 14

if 'cache_purged_this_session' not in st.session_state:
    # Purge expired entries once per session, not every rerun
    _AUCTION_CACHE.purge_expired(ttl_days=st.session_state.cache_ttl_days)
    st.session_state.cache_purged_this_session = True

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

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("📍 Sourcing")
    user_zip = st.text_input("Home Zip Code", value="77058")
    radius = st.slider("Local Pickup Radius (mi)", 5, 100, 20)
    include_nationwide = st.checkbox("Include Nationwide (Ship-to-Me)", value=True)
    closing_days = st.slider("Closing Within (days)", 1, 30, 1)

    category_filter = st.multiselect(
        "🏷️ Categories (optional)",
        options=sorted(set(st.session_state.known_categories)),
        placeholder="All categories",
        help=(
            "Only keep lots whose category matches any selected term (substring, "
            "case-insensitive). Saves time in Phase 2 by dropping irrelevant items. "
            "Leave blank to fetch everything. The list grows with any new categories "
            "seen in prior scrapes."
        ),
    )

    st.markdown("---")

    if st.button("🚀 Run Phase 1 Discovery", type="primary", use_container_width=True):
        _keep_screen_awake()
        try:
            scraper = Phase1Scraper(config_path="config.json")

            # Apply UI settings
            scraper.zip_code = user_zip
            scraper.radius = radius
            scraper.include_nationwide = include_nationwide
            scraper.closing_within_days = closing_days
            scraper.category_filter = category_filter

            scan_progress = st.progress(0, text="Starting discovery...")

            def scan_prog(current, total, label=""):
                pct = (current / total) if total > 0 else 0
                if total == 0:
                    text = label or "Done"
                elif current == 0 and total == 1:
                    # Indeterminate / "working on it" tick — label carries the meaning.
                    text = label or "Working..."
                    pct = 0
                else:
                    prefix = label if label else "Fetching lots"
                    text = f"{prefix} — {current}/{total} auctions"
                scan_progress.progress(min(pct, 1.0), text=text)

            df = run_async_scraper(scraper, progress_callback=scan_prog)
            scan_progress.empty()

            st.session_state.phase1_leads = df
            st.session_state.audit_results = {}
            st.session_state.selected_leads = pd.DataFrame()
            st.session_state.current_auction = None

            # Grow the known-category list so future runs can pick from what
            # HiBid actually returns for this user's area.
            if not df.empty and 'category' in df.columns:
                seen = {c for c in df['category'].dropna().astype(str).tolist() if c}
                st.session_state.known_categories = sorted(
                    set(st.session_state.known_categories) | seen
                )

            local_count = len(df[df['source'] == "Local Pickup"]) if not df.empty and 'source' in df.columns else len(df)
            ship_count = len(df[df['source'] == "Ship"]) if not df.empty and 'source' in df.columns else 0
            auction_count = df['auction'].nunique() if not df.empty else 0
            msg = f"Found {len(df)} items across {auction_count} auctions! ({local_count} local"
            if ship_count:
                msg += f", {ship_count} shippable"
            msg += ")"
            st.success(msg)
            st.rerun()
        except Exception as e:
            st.error(f"Scraper failed: {e}")

    # --- Memory / Cache controls ---
    st.markdown("---")
    st.header("💾 Memory")
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

# --- MAIN DASHBOARD UI ---
st.title("🛰️ Auction Intelligence Dashboard")
st.markdown("Automated sourcing and risk-assessment for H-Town TX Finds.")

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

    subtitle_parts = [f"{item_count} items"]
    if closing:
        subtitle_parts.append(f"closes {closing}")
    if source:
        subtitle_parts.append(source)
    subtitle_parts.append(f"avg bid ${avg_bid:.2f}")
    if easy_count:
        subtitle_parts.append(f"{easy_count} easy-ship")

    # Cache hit indicator on the expander label
    auction_id = _extract_auction_id(auction_df)
    cache_prefix = ""
    if auction_id is not None:
        payload = _AUCTION_CACHE.load(auction_id)
        if payload and _AUCTION_CACHE.is_fresh(payload, ttl_days=st.session_state.cache_ttl_days):
            cache_prefix = "💾 "

    with st.expander(f"{cache_prefix}🏷️ **{auction_name}** — {' · '.join(subtitle_parts)}", expanded=False):
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
def _get_auditor():
    """Load (and cache) the Phase2Scraper with its NLP model.

    First call: ~1.6GB download on cold cache, ~30s model load on warm cache.
    Subsequent calls in the same session return instantly.
    """
    from scraper import Phase2Scraper
    return Phase2Scraper()


def _run_ai_audit(leads_df):
    """Run Phase 2 AI condition audit with detailed phase-by-phase status."""
    total = len(leads_df)

    with st.status("🧠 Running AI Condition Audit…", expanded=True) as status:
        # Phase 1: model load
        st.write(
            "**📥 Step 1/3 — Loading NLP model** "
            "(`facebook/bart-large-mnli`, ~1.6GB). "
            "Downloads on first run, cached after — may take a minute."
        )
        auditor = _get_auditor()
        st.write("✅ Model ready.")

        # Phase 2: title enrichment + condition classification
        st.write(
            f"**🔍 Step 2/3 — Enriching titles and classifying condition** "
            f"for {total} items."
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
            # Show the title of the item just processed for a live "what's happening now" feel
            try:
                row = leads_df.iloc[current - 1]
                title_preview = str(row.get('title', ''))[:70]
            except Exception:
                title_preview = ""
            progress_bar.progress(
                min(pct, 1.0),
                text=f"Analyzing condition {current}/{total_items}…",
            )
            if title_preview:
                current_item_placeholder.caption(f"🔎 Just analyzed: *{title_preview}*")

        results_df = auditor.batch_audit(leads_df, progress_callback=ai_progress)

        # Phase 3: summarize
        good = int((~results_df['red_flag']).sum()) if 'red_flag' in results_df.columns else 0
        flagged = int(results_df['red_flag'].sum()) if 'red_flag' in results_df.columns else 0
        st.write(
            f"**📊 Step 3/3 — Summary:** "
            f"✅ {good} good-condition · ⚠️ {flagged} red-flagged"
        )
        status.update(label="✅ AI audit complete", state="complete", expanded=False)

    return results_df


def _run_ebay_comps(results_df):
    """Run eBay + Mercari price comps on the good+ items in results_df.

    Max bid is NOT computed here — it's recomputed on every render so the
    Target ROI slider in the results section updates it live.
    """
    # Clear previous comp data so re-runs start fresh
    for col in ['est_resale', 'price_low', 'price_high', 'comp_count',
                'ebay_comps', 'mercari_comps',
                'price_source', 'ebay_str', 'str_source',
                'est_roi', 'max_bid']:
        if col in results_df.columns:
            results_df = results_df.drop(columns=[col])

    good_df = results_df[~results_df['red_flag']].copy()
    flagged_df = results_df[results_df['red_flag']].copy()

    from scraper.ebay_prices import EbayPriceLookup
    from scraper.config_loader import load_config
    cfg = load_config()
    ebay = EbayPriceLookup(cfg["ebay"]["app_id"], cfg["ebay"]["cert_id"])

    total = len(good_df)

    with st.status("💰 Running Price Comps & STR…", expanded=True) as status:
        st.write(
            f"**🔗 Looking up eBay sold listings + Mercari sold listings** "
            f"for {total} good-condition items."
        )
        st.caption(
            "For each item: scrape recent sold prices from both marketplaces, "
            "apply IQR outlier filtering, pool into median / 25th / 75th percentile, "
            "then compute eBay sell-through rate from the active vs sold ratio."
        )
        progress_bar = st.progress(0, text=f"Starting — 0/{total}")
        current_item_placeholder = st.empty()

        def price_progress(current, total_items):
            pct = current / total_items if total_items > 0 else 1.0
            try:
                row = good_df.iloc[current - 1]
                title_preview = str(
                    row.get('enriched_title') or row.get('title') or ''
                )[:70]
            except Exception:
                title_preview = ""
            progress_bar.progress(
                min(pct, 1.0),
                text=f"Looking up item {current}/{total_items}…",
            )
            if title_preview:
                current_item_placeholder.caption(f"🔎 Just priced: *{title_preview}*")

        comps_df = ebay.batch_lookup(good_df, progress_callback=price_progress)

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

    combined = pd.concat([comps_df, flagged_df], ignore_index=True)
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
    display_cols = [title_col, 'lot_link', 'auction_link', 'category', 'current_bid', 'est_cost']
    col_config = {
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
        if 'mercari_comps' in filtered_df.columns:
            display_cols += ['mercari_comps']
            col_config["mercari_comps"] = st.column_config.NumberColumn("Mercari Comps", format="%d")
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

        comps_btn_label = "⏳ Running price comps…" if comps_running else "💰 Run Price Comps on Good+ Items"
        if st.button(
            comps_btn_label,
            type="primary",
            use_container_width=True,
            disabled=audit_running or comps_running,
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

# ---- DISCOVERY VIEW: no auction loaded ----
elif not st.session_state.phase1_leads.empty:
    df = st.session_state.phase1_leads
    total_items = len(df)
    total_auctions = df['auction'].nunique() if 'auction' in df.columns else 0

    # --- Filters ---
    with st.container():
        fc1, fc2 = st.columns([2, 1])
        with fc1:
            search_query = st.text_input(
                "🔎 Search",
                placeholder="Search auctions, item titles, or descriptions (e.g. 'fishing', 'vintage', 'Weber')",
                key="discovery_search",
            ).strip()
        with fc2:
            categories = (
                sorted(df['category'].dropna().unique().tolist())
                if 'category' in df.columns else []
            )
            selected_categories = st.multiselect(
                "🏷️ Category",
                options=categories,
                placeholder="All categories",
                key="discovery_category",
            )

        # Source filter (only if multiple sources)
        sources = df['source'].unique().tolist() if 'source' in df.columns else []
        if len(sources) > 1:
            selected_source = st.radio("Source:", ["All"] + sources, horizontal=True)
            if selected_source != "All":
                df = df[df['source'] == selected_source]

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

    if selected_categories and 'category' in df.columns:
        df = df[df['category'].isin(selected_categories)]

    if df.empty:
        st.warning(
            f"No matches for your filters. (Started with {total_auctions} auctions, {total_items} items.) "
            "Try broadening the search or clearing the category filter."
        )
        st.stop()

    auction_groups = df.groupby('auction', sort=False)
    auction_order = df.groupby('auction')['closing_date'].first().sort_values()

    filter_bits = []
    if search_query:
        filter_bits.append(f"search \"{search_query}\"")
    if selected_categories:
        filter_bits.append(f"{len(selected_categories)} category filter(s)")
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
    st.info("👋 Configure your filters in the sidebar and click **🚀 Run Phase 1 Discovery** to get started.")
