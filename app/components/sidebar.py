import streamlit as st
import datetime
from db import get_filter_options, get_filtered_companies, get_db_stats

def render_sidebar():
    """Renders the custom sidebar with filters and metadata."""
    
    # -----------------------------
    # Sidebar Header & Branding
    # -----------------------------
    st.sidebar.markdown("""
        <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1.5rem;">
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 2C6.48 2 2 6.48 2 12C2 17.52 6.48 22 12 22C17.52 22 22 17.52 22 12C22 6.48 17.52 2 12 2ZM13 17H11V15H13V17ZM13 13H11V7H13V13Z" fill="#3b82f6"/>
            </svg>
            <span style="font-family: 'Outfit', sans-serif; font-size: 1.6rem; font-weight: 800; background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">RiskPulse</span>
        </div>
    """, unsafe_allow_html=True)
    
    # -----------------------------
    # Fetch Data Stats and Options
    # -----------------------------
    stats = get_db_stats()
    opts = get_filter_options()
    
    # Standardize date types
    min_date = opts["min_date"]
    max_date = opts["max_date"]
    if isinstance(min_date, str):
        min_date = datetime.datetime.strptime(min_date, "%Y-%m-%d").date()
    if isinstance(max_date, str):
        max_date = datetime.datetime.strptime(max_date, "%Y-%m-%d").date()

    # -----------------------------
    # Session State Initialization
    # -----------------------------
    if "industry" not in st.session_state:
        st.session_state.industry = "All"
    if "country" not in st.session_state:
        st.session_state.country = "All"
    if "risk_level" not in st.session_state:
        st.session_state.risk_level = "All"
    if "start_date" not in st.session_state:
        st.session_state.start_date = min_date
    if "end_date" not in st.session_state:
        st.session_state.end_date = max_date
    if "search_company" not in st.session_state:
        st.session_state.search_company = "Amazon"  # Default fallback selection
        
    # Reset button handler
    def handle_reset():
        st.session_state.industry = "All"
        st.session_state.country = "All"
        st.session_state.risk_level = "All"
        st.session_state.start_date = min_date
        st.session_state.end_date = max_date
        st.session_state.search_company = "Amazon"
        
    # -----------------------------
    # Filter Widgets
    # -----------------------------
    st.sidebar.subheader("🎯 Filters")
    
    st.session_state.industry = st.sidebar.selectbox(
        "Industry",
        opts["industries"],
        index=opts["industries"].index(st.session_state.industry)
    )
    
    st.session_state.country = st.sidebar.selectbox(
        "Country",
        opts["countries"],
        index=opts["countries"].index(st.session_state.country)
    )
    
    st.session_state.risk_level = st.sidebar.selectbox(
        "Risk Level",
        opts["risk_levels"],
        index=opts["risk_levels"].index(st.session_state.risk_level)
    )
    
    # Date Range Filter
    date_choice = st.sidebar.date_input(
        "Date Range",
        value=(st.session_state.start_date, st.session_state.end_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Safe unpack date choice
    if isinstance(date_choice, tuple) and len(date_choice) == 2:
        st.session_state.start_date, st.session_state.end_date = date_choice
    elif isinstance(date_choice, datetime.date):
        st.session_state.start_date = date_choice
        st.session_state.end_date = date_choice
        
    st.sidebar.button("🔄 Reset Filters", on_click=handle_reset, use_container_width=True)
    
    # -----------------------------
    # Filter Company List dynamically
    # -----------------------------
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    st.sidebar.subheader("🔍 Company Selection")
    
    filtered_companies = get_filtered_companies(
        industry=st.session_state.industry,
        country=st.session_state.country,
        risk_level=st.session_state.risk_level,
        start_date=st.session_state.start_date,
        end_date=st.session_state.end_date
    )
    
    # Validate selected company is in the filtered list
    if not filtered_companies:
        st.sidebar.warning("No companies match the active filters.")
        st.session_state.search_company = None
        selected_company = None
    else:
        if st.session_state.search_company not in filtered_companies:
            st.session_state.search_company = filtered_companies[0]
            
        selected_company = st.sidebar.selectbox(
            "Search Company",
            filtered_companies,
            index=filtered_companies.index(st.session_state.search_company),
            key="selected_company_widget"
        )
        st.session_state.search_company = selected_company
        
    # -----------------------------
    # Database and Health Status
    # -----------------------------
    st.sidebar.markdown("<br><hr style='border: 0.5px solid rgba(255,255,255,0.08); margin: 1.5rem 0;'>", unsafe_allow_html=True)
    st.sidebar.subheader("📊 System Status")
    
    db_dot = "dot-green" if stats.get("db_status") == "Connected" else "dot-red"
    api_dot = "dot-green" if stats.get("news_api_status") == "Active" else "dot-red"
    
    st.sidebar.markdown(f"""
        <div class="sidebar-status-item">
            <span>Database Status</span>
            <span class="status-indicator">
                <span class="status-dot {db_dot}"></span>
                {stats.get("db_status", "Disconnected")}
            </span>
        </div>
        <div class="sidebar-status-item">
            <span>News API Status</span>
            <span class="status-indicator">
                <span class="status-dot {api_dot}"></span>
                {stats.get("news_api_status", "Missing")}
            </span>
        </div>
        <div class="sidebar-status-item">
            <span>Companies Loaded</span>
            <span style="font-weight: 600;">{stats.get("num_companies", 0):,}</span>
        </div>
        <div class="sidebar-status-item">
            <span>Layoff Events</span>
            <span style="font-weight: 600;">{stats.get("num_layoffs", 0):,}</span>
        </div>
        <div class="sidebar-status-item">
            <span>Last Risk Refresh</span>
            <span style="font-weight: 600; font-size: 0.7rem;">{stats.get("last_updated", "N/A")}</span>
        </div>
        <div class="sidebar-status-item" style="border-bottom: none;">
            <span>System Version</span>
            <span style="color: #6B7280; font-weight: 600;">v1.1.0</span>
        </div>
    """, unsafe_allow_html=True)
    
    return selected_company
