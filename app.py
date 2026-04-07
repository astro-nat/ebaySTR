import streamlit as st
import pandas as pd
import asyncio
import os

# --- IMPORT MODULES ---
# These assume you have your __init__.py files set up correctly
from scraper import Phase1Scraper
# from scraper.pass2 import Phase2Scraper # Uncomment when built
# from utils import TitleSanitizer, FinancialEngine, EbayClient # Uncomment when built

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="H-Town TX Finds: ROI Engine",
    page_icon="🛰️",
    layout="wide"
)

# --- STATE MANAGEMENT ---
if 'phase1_leads' not in st.session_state:
    st.session_state.phase1_leads = pd.DataFrame()

if 'audit_results' not in st.session_state:
    st.session_state.audit_results = {}

# --- ASYNC WRAPPER ---
def run_async_scraper(scraper_instance):
    """Safely runs the asyncio scraper within Streamlit's synchronous thread."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    return loop.run_until_complete(scraper_instance.run())

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("📍 Logistics Filters")
    user_zip = st.text_input("Home Zip Code", value="77058")
    radius = st.slider("Local Pickup Radius (mi)", 5, 100, 20)
    ship_only = st.checkbox("Prioritize Shipping Auctions", value=True)
    
    st.header("💰 Financial Targets")
    target_roi = st.number_input("Target Net ROI", value=5.0, step=0.5, format="%.1f")
    
    st.markdown("---")
    
    if st.button("🚀 Run Phase 1 Discovery", type="primary", use_container_width=True):
        with st.spinner("Scraping HiBid JSON endpoints..."):
            try:
                # Initialize the scraper (assumes config.json exists)
                scraper = Phase1Scraper(config_path="config.json")
                
                # Update parameters based on UI
                scraper.zip_code = user_zip
                scraper.radius = radius
                
                # Run and save to session state
                df = run_async_scraper(scraper)
                st.session_state.phase1_leads = df
                st.success(f"Discovered {len(df)} potential leads!")
            except Exception as e:
                st.error(f"Scraper failed: {e}")

# --- MAIN DASHBOARD UI ---
st.title("🛰️ Auction Intelligence Dashboard")
st.markdown("Automated sourcing and risk-assessment for H-Town TX Finds.")

# Create the navigation tabs
tab1, tab2, tab3 = st.tabs(["🔍 Phase 1: Lead Discovery", "📊 Phase 2: AI Audit", "📦 Nifty.ai Export"])

# --- TAB 1: DISCOVERY ---
with tab1:
    if st.session_state.phase1_leads.empty:
        st.info("No leads discovered yet. Configure your filters in the sidebar and click 'Run Phase 1 Discovery'.")
    else:
        st.subheader("Raw Discovery Data (Logistics Filtered)")
        st.write("These items passed the initial regex logistics check (no oversized/heavy items).")
        
        df = st.session_state.phase1_leads
        
        # Display the dataframe with basic Streamlit styling
        st.dataframe(
            df,
            use_container_width=True,
            column_config={
                "current_bid": st.column_config.NumberColumn("Current Bid", format="$%.2f"),
                "logistics_ease": st.column_config.TextColumn("Logistics Tier")
            }
        )

# --- TAB 2: AI AUDIT ---
with tab2:
    st.subheader("Deep Dive Risk Assessment")
    st.markdown("Select a high-value lead to trigger the HuggingFace Zero-Shot Classification audit.")
    
    if st.session_state.phase1_leads.empty:
        st.warning("Please run Phase 1 first to populate leads.")
    else:
        # Create a dropdown to select a specific item
        titles = st.session_state.phase1_leads['title'].tolist()
        selected_item = st.selectbox("Select an item to audit:", titles)
        
        if st.button("🧠 Run AI Audit"):
            with st.spinner(f"Downloading HiBid HTML and analyzing {selected_item}..."):
                # TODO: Wire up the actual Pass 2 Scraper and HuggingFace model here
                import time
                time.sleep(1.5) # Simulating processing time
                
                # Mock result for layout purposes
                st.success("Audit Complete!")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(label="AI Verdict", value="Light Wear", delta="88% Confidence", delta_color="normal")
                with col2:
                    st.write("**Identified Red Flags:**")
                    st.write("- 'Untested' not found.")
                    st.write("- 'As-is' found in footer, standard auction house boilerplate.")

# --- TAB 3: NIFTY EXPORT ---
with tab3:
    st.subheader("Batch Manifest Generation")
    
    if st.session_state.phase1_leads.empty:
        st.info("No data available for export.")
    else:
        st.write("Generate a CSV formatted specifically for Nifty.ai bulk ingestion.")
        
        # In a real run, you would filter by a "Gold Mine" boolean column calculated by your financials engine
        export_df = st.session_state.phase1_leads.copy()
        
        st.download_button(
            label="📥 Download Nifty.ai CSV",
            data=export_df.to_csv(index=False).encode('utf-8'),
            file_name="htown_finds_nifty_import.csv",
            mime="text/csv",
            type="primary"
        )