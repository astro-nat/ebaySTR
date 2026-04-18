import streamlit as st
import pandas as pd
import asyncio
import os

# --- IMPORT MODULES ---
from scraper import Phase1Scraper

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
    closing_days = st.slider("Closing Within (days)", 1, 30, 7)

    st.markdown("---")

    if st.button("🚀 Run Phase 1 Discovery", type="primary", use_container_width=True):
        try:
            scraper = Phase1Scraper(config_path="config.json")

            # Apply UI settings
            scraper.zip_code = user_zip
            scraper.radius = radius
            scraper.include_nationwide = include_nationwide
            scraper.closing_within_days = closing_days

            status_text = st.empty()
            scan_progress = st.progress(0, text="Discovering auctions...")

            status_text.caption("Finding open auctions on HiBid...")

            def scan_prog(current, total):
                pct = current / total if total > 0 else 0
                scan_progress.progress(min(pct, 1.0), text=f"Fetching lots: auction {current}/{total}...")

            df = run_async_scraper(scraper, progress_callback=scan_prog)
            scan_progress.empty()
            status_text.empty()

            st.session_state.phase1_leads = df
            st.session_state.audit_results = {}
            st.session_state.selected_leads = pd.DataFrame()
            st.session_state.current_auction = None

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

    Clears prior audit/comp results since they're tied to the old auction.
    """
    st.session_state.selected_leads = auction_df.copy()
    st.session_state.current_auction = auction_name
    st.session_state.audit_results = {}


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

    with st.expander(f"🏷️ **{auction_name}** — {' · '.join(subtitle_parts)}", expanded=False):
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


def _run_ai_audit(leads_df):
    """Run Phase 2 AI condition audit on the leads DataFrame."""
    from scraper import Phase2Scraper

    progress_bar = st.progress(0, text="Loading NLP model...")
    auditor = Phase2Scraper()

    def ai_progress(current, total):
        progress_bar.progress(current / total, text=f"Analyzing condition {current}/{total}...")

    results_df = auditor.batch_audit(leads_df, progress_callback=ai_progress)
    progress_bar.empty()
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

    progress_bar = st.progress(0, text="Looking up eBay + Mercari prices & STR...")

    def price_progress(current, total):
        progress_bar.progress(current / total, text=f"Looking up item {current}/{total}...")

    comps_df = ebay.batch_lookup(good_df, progress_callback=price_progress)

    # ROI
    comps_df['est_roi'] = None
    mask = comps_df['est_resale'].notna() & (comps_df['est_cost'] > 0)
    comps_df.loc[mask, 'est_roi'] = (
        (comps_df.loc[mask, 'est_resale'] - comps_df.loc[mask, 'est_cost'])
        / comps_df.loc[mask, 'est_cost'] * 100
    ).round(0)

    combined = pd.concat([comps_df, flagged_df], ignore_index=True)
    progress_bar.empty()
    return combined, comps_df['est_resale'].notna().sum(), len(good_df)


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
        st.caption(f"{len(leads_df)} items loaded")

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

    if st.button("🧠 Run AI Condition Audit", type="primary", use_container_width=True):
        st.session_state.audit_results = _run_ai_audit(leads_df)
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

        if st.button("💰 Run Price Comps on Good+ Items", type="primary", use_container_width=True):
            combined, found, total = _run_ebay_comps(ar)
            st.session_state.audit_results = combined
            st.success(f"Found price comps for {found}/{total} good+ leads.")
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
