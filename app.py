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

    st.header("💰 Financial Targets")
    target_roi = st.number_input("Target ROI Multiplier", value=3.0, step=0.5, format="%.1f",
                                  help="Minimum resale-to-cost ratio. e.g. 3x means sell for 3x what you paid.")
    target_str = st.number_input("Target eBay STR %", value=70.0, step=5.0, format="%.0f",
                                  min_value=0.0, max_value=100.0,
                                  help="Minimum sell-through rate on eBay. Higher = faster-selling items.")

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

tab1, tab2, tab3 = st.tabs(["🔍 Phase 1: Lead Discovery", "📊 Phase 2: AI Audit", "📦 Nifty.ai Export"])

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


@st.fragment
def _render_auction_expander(auction_name, auction_df, col_config, col_order):
    """Render a single auction's expander as an isolated fragment.

    Button clicks inside a fragment only re-render the fragment itself, not the
    entire page — critical when there are hundreds of auctions on screen.
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

    is_active = st.session_state.get('current_auction') == auction_name
    prefix = "⭐ " if is_active else "🏷️ "

    with st.expander(f"{prefix}**{auction_name}** — {' · '.join(subtitle_parts)}", expanded=False):
        button_label = f"🎯 Analyze This Auction ({item_count} items)"
        if is_active:
            button_label = "✅ Currently Loaded — Re-analyze"

        if st.button(button_label, key=f"load_{auction_name}", type="primary", use_container_width=True):
            _load_auction_for_analysis(auction_name, auction_df)
            st.success(f"Loaded {item_count} items from **{auction_name}**. Open the **Phase 2** tab to run the audit.")

        st.dataframe(
            auction_df,
            use_container_width=True,
            key=f"table_{auction_name}",
            column_config=col_config,
            column_order=col_order,
            hide_index=True,
        )


# --- TAB 1: DISCOVERY ---
with tab1:
    if st.session_state.phase1_leads.empty:
        st.info("No leads discovered yet. Configure your filters in the sidebar and click 'Run Phase 1 Discovery'.")
    else:
        df = st.session_state.phase1_leads

        # Source filter
        sources = df['source'].unique().tolist() if 'source' in df.columns else []
        if len(sources) > 1:
            selected_source = st.radio("Filter by source:", ["All"] + sources, horizontal=True)
            if selected_source != "All":
                df = df[df['source'] == selected_source]

        # Active auction banner
        active_auction = st.session_state.get('current_auction')
        if active_auction:
            active_count = len(st.session_state.selected_leads)
            lc1, lc2 = st.columns([3, 1])
            with lc1:
                st.info(f"⭐ **Loaded for analysis:** {active_auction} — {active_count} items. Go to **Phase 2** to audit.")
            with lc2:
                if st.button("🗑️ Clear", use_container_width=True):
                    st.session_state.selected_leads = pd.DataFrame()
                    st.session_state.current_auction = None
                    st.session_state.audit_results = {}
                    st.rerun()

        # Group by auction, sorted by closing date
        auction_groups = df.groupby('auction', sort=False)
        # Sort auctions by closing date
        auction_order = df.groupby('auction')['closing_date'].first().sort_values()

        st.subheader(f"Discovery Results — {len(auction_order)} auctions, {len(df)} items")
        st.caption("Expand an auction to preview items, then click **🎯 Analyze This Auction** to run Phase 2 on all of its items.")

        for auction_name in auction_order.index:
            auction_df = auction_groups.get_group(auction_name).reset_index(drop=True)
            _render_auction_expander(auction_name, auction_df,
                                     DISCOVERY_COL_CONFIG, DISCOVERY_COL_ORDER)

# --- TAB 2: AI AUDIT ---
with tab2:
    active_auction = st.session_state.get('current_auction')
    if active_auction:
        st.subheader(f"🔬 Analyzing: {active_auction}")
    else:
        st.subheader("Lead Deep Dive")
    st.markdown("**Step 1:** AI condition audit → **Step 2:** eBay price comps & STR on good+ items only.")

    if st.session_state.selected_leads.empty:
        st.warning("No auction loaded yet. Go to **Phase 1** → expand an auction → click **🎯 Analyze This Auction**.")
    else:
        leads_df = st.session_state.selected_leads
        has_audit = isinstance(st.session_state.get('audit_results'), pd.DataFrame) and not st.session_state.audit_results.empty and 'verdict' in st.session_state.audit_results.columns
        has_comps = has_audit and 'est_resale' in st.session_state.audit_results.columns

        # --- STEP 1: AI Condition Audit ---
        st.markdown("---")
        st.markdown("### Step 1: AI Condition Audit")
        st.caption(f"🎯 {len(leads_df)} item(s) from this auction ready for analysis")

        if has_audit:
            results_df = st.session_state.audit_results
            good_count = (~results_df['red_flag']).sum()
            flagged_count = results_df['red_flag'].sum()
            st.success(f"Audit complete — **{good_count} good+** condition, {flagged_count} red-flagged")

        if st.button("🧠 Run AI Condition Audit", type="primary", use_container_width=True):
            from scraper import Phase2Scraper

            progress_bar = st.progress(0, text="Loading NLP model...")
            auditor = Phase2Scraper()

            def ai_progress(current, total):
                progress_bar.progress(current / total, text=f"Analyzing condition {current}/{total}...")

            results_df = auditor.batch_audit(leads_df, progress_callback=ai_progress)
            st.session_state.audit_results = results_df
            progress_bar.empty()
            st.rerun()

        # --- STEP 2: eBay Price Comps (only after audit, only good+ items) ---
        st.markdown("---")
        st.markdown("### Step 2: eBay Price Comps & STR")

        if not has_audit:
            st.info("Run the AI audit first — eBay lookups will only run on items in **good+ condition** to save API calls.")
        else:
            results_df = st.session_state.audit_results
            good_df = results_df[~results_df['red_flag']].copy()
            flagged_df = results_df[results_df['red_flag']].copy()

            st.caption(f"💰 {len(good_df)} good+ items eligible for eBay lookup ({len(flagged_df)} red-flagged skipped)")

            if st.button("💰 Run eBay Comps on Good+ Items", type="primary", use_container_width=True):
                # Clear previous comps data so we start fresh
                results_df = st.session_state.audit_results
                for col in ['est_resale', 'price_low', 'price_high', 'comp_count',
                            'ebay_comps', 'mercari_comps',
                            'price_source', 'ebay_str', 'str_source',
                            'est_roi', 'max_bid']:
                    if col in results_df.columns:
                        results_df = results_df.drop(columns=[col])
                st.session_state.audit_results = results_df
                good_df = results_df[~results_df['red_flag']].copy()
                flagged_df = results_df[results_df['red_flag']].copy()

                from scraper.ebay_prices import EbayPriceLookup
                from scraper.config_loader import load_config
                cfg = load_config()
                ebay = EbayPriceLookup(cfg["ebay"]["app_id"], cfg["ebay"]["cert_id"])

                progress_bar = st.progress(0, text="Looking up eBay prices & STR...")

                def price_progress(current, total):
                    progress_bar.progress(current / total, text=f"Looking up item {current}/{total} on eBay...")

                comps_df = ebay.batch_lookup(good_df, progress_callback=price_progress)

                # Calculate ROI where we have resale data
                comps_df['est_roi'] = None
                mask = comps_df['est_resale'].notna() & (comps_df['est_cost'] > 0)
                comps_df.loc[mask, 'est_roi'] = (
                    (comps_df.loc[mask, 'est_resale'] - comps_df.loc[mask, 'est_cost'])
                    / comps_df.loc[mask, 'est_cost'] * 100
                ).round(0)

                # Calculate max bid to hit target ROI
                # You pay: bid + buyer_premium (15%) + shipping
                # You receive: resale - eBay fees (13.25% FVF + $0.30)
                # Target: receive >= pay * ROI
                # So: max_bid = (resale_after_fees / ROI - shipping) / (1 + premium_pct)
                ebay_fee_pct = 0.1325
                ebay_fee_flat = 0.30
                buyer_premium_pct = cfg.get("shipping", {}).get("buyer_premium_pct", 15.0) / 100.0
                ship_cost = cfg.get("shipping", {}).get("bundled_ship_cost", 25.0)

                comps_df['max_bid'] = None
                resale_mask = comps_df['est_resale'].notna()
                if resale_mask.any():
                    resale = comps_df.loc[resale_mask, 'est_resale']
                    net_resale = resale * (1 - ebay_fee_pct) - ebay_fee_flat
                    # For local pickup, no shipping cost
                    item_ship = comps_df.loc[resale_mask, 'source'].apply(
                        lambda s: ship_cost if s == "Ship" else 0
                    )
                    max_bid = (net_resale / target_roi - item_ship) / (1 + buyer_premium_pct)
                    comps_df.loc[resale_mask, 'max_bid'] = max_bid.clip(lower=0).round(2)

                # Merge comps back with flagged items (flagged stay without comps)
                combined = pd.concat([comps_df, flagged_df], ignore_index=True)
                st.session_state.audit_results = combined
                progress_bar.empty()
                found = comps_df['est_resale'].notna().sum()
                st.success(f"Found eBay comps for {found}/{len(good_df)} good+ leads!")
                st.rerun()

    if 'audit_results' in st.session_state and isinstance(st.session_state.audit_results, pd.DataFrame) and not st.session_state.audit_results.empty:
        results_df = st.session_state.audit_results

        # Filter by targets from sidebar
        filtered_df = results_df.copy()
        filters_applied = []

        if 'ebay_str' in filtered_df.columns:
            meets_str = filtered_df['ebay_str'].notna() & (filtered_df['ebay_str'] >= target_str)
            no_str = filtered_df['ebay_str'].isna()
            filtered_df = filtered_df[meets_str | no_str]
            filters_applied.append(f"STR ≥ {target_str:.0f}%")

        if 'est_roi' in filtered_df.columns:
            meets_roi = filtered_df['est_roi'].notna() & (filtered_df['est_roi'] >= (target_roi - 1) * 100)
            no_roi = filtered_df['est_roi'].isna()
            filtered_df = filtered_df[meets_roi | no_roi]
            filters_applied.append(f"ROI ≥ {target_roi:.1f}x")

        if filters_applied:
            hidden = len(results_df) - len(filtered_df)
            if hidden > 0:
                st.caption(f"Filters applied: {', '.join(filters_applied)} — {hidden} item(s) hidden")

        # Summary metrics — 3 columns wraps nicely on mobile
        col1, col2, col3 = st.columns(3)
        col1.metric("Leads", len(filtered_df))

        if 'est_resale' in filtered_df.columns:
            has_comps = filtered_df['est_resale'].notna().sum()
            col2.metric("Comps", has_comps)
            if 'est_roi' in filtered_df.columns:
                profitable = (filtered_df['est_roi'].notna() & (filtered_df['est_roi'] > 0)).sum()
                col3.metric("Profitable", profitable)

        col4, col5 = st.columns(2)
        if 'ebay_str' in filtered_df.columns:
            has_str = filtered_df['ebay_str'].notna().sum()
            avg_str = filtered_df['ebay_str'].mean()
            col4.metric("Avg STR", f"{avg_str:.0f}%" if has_str > 0 else "N/A")

        if 'red_flag' in filtered_df.columns:
            flagged = filtered_df['red_flag'].sum()
            col5.metric("Red Flags", int(flagged))

        # Build display columns dynamically
        # Show enriched title if available, otherwise original
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

            # Price range — show low/high
            if 'price_low' in filtered_df.columns:
                display_cols += ['price_low', 'price_high']
                col_config["price_low"] = st.column_config.NumberColumn("Low (25%)", format="$%.2f")
                col_config["price_high"] = st.column_config.NumberColumn("High (75%)", format="$%.2f")

            # Comp counts per source
            if 'ebay_comps' in filtered_df.columns:
                display_cols += ['ebay_comps']
                col_config["ebay_comps"] = st.column_config.NumberColumn(
                    "eBay Comps",
                    format="%d",
                    help="Number of eBay sold listings used"
                )
            if 'mercari_comps' in filtered_df.columns:
                display_cols += ['mercari_comps']
                col_config["mercari_comps"] = st.column_config.NumberColumn(
                    "Mercari Comps",
                    format="%d",
                    help="Number of Mercari sold listings used"
                )
            if 'price_source' in filtered_df.columns:
                display_cols += ['price_source']
                col_config["price_source"] = st.column_config.TextColumn(
                    "Price Src",
                    help="Shows which marketplace(s) contributed sold data"
                )

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

        st.dataframe(
            filtered_df[[c for c in display_cols if c in filtered_df.columns]],
            use_container_width=True,
            column_config=col_config,
        )

# --- TAB 3: NIFTY EXPORT ---
with tab3:
    st.subheader("Batch Manifest Generation")

    if st.session_state.phase1_leads.empty:
        st.info("No data available for export.")
    else:
        st.write("Generate a CSV formatted specifically for Nifty.ai bulk ingestion.")

        export_df = st.session_state.phase1_leads.copy()

        st.download_button(
            label="📥 Download Nifty.ai CSV",
            data=export_df.to_csv(index=False).encode('utf-8'),
            file_name="htown_finds_nifty_import.csv",
            mime="text/csv",
            type="primary"
        )
