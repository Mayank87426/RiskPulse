# ── CRITICAL: Virtual-env path shim (must run before ANY library import) ─────
# Models are pickled under .venv with scikit-learn 1.9.x.  If Streamlit is
# launched from a different Python (e.g. global install with sklearn 1.8.x),
# unpickling crashes with "No module named '_loss'".  We pre-insert the .venv
# site-packages so the correct sklearn is always resolved first.
import sys, os
from pathlib import Path as _Path
_venv_site = _Path(__file__).resolve().parent.parent / ".venv" / "Lib" / "site-packages"
if _venv_site.exists() and str(_venv_site) not in sys.path:
    sys.path.insert(0, str(_venv_site))
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd

# Add 'app' directory to system path to ensure reliable module imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import Database Helper Functions
from db import get_company_overview, get_layoff_history, get_company_news

# Import Global Styling and Helper Modules
from utils.styles import inject_custom_css
from utils.helpers import get_sentiment_summary

# Import Dashboard UI Components
from components.sidebar import render_sidebar
from components.header import render_header
from components.kpi_cards import render_kpi_cards
from components.risk_gauge import render_risk_gauge
from components.overview import render_company_overview
from components.executive_summary import render_executive_summary
from components.charts import render_layoff_combination_chart, render_risk_sparkline
from components.news_panel import render_news_panel
from components.ai_prediction import render_ai_prediction_tab

# ---------------------------------------------------------
# Page Configurations & Styling Injections
# ---------------------------------------------------------
st.set_page_config(
    page_title="RiskPulse - Enterprise Workforce Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject modern custom stylesheet rules (glassmorphism cards, badges, grids, scrollbars)
inject_custom_css()

# ---------------------------------------------------------
# Sidebar Component (Filters & Database Health Stats)
# ---------------------------------------------------------
selected_company = render_sidebar()

# ---------------------------------------------------------
# Dashboard Main Content Assembly
# ---------------------------------------------------------
if selected_company:
    # 1. Fetch data for selected company
    overview = get_company_overview(selected_company)
    layoff_history = get_layoff_history(selected_company)
    news_list = get_company_news(overview["id"]) if overview else []
    
    # 2. Convert layoff history to Pandas DataFrame
    history_df = pd.DataFrame(layoff_history) if layoff_history else pd.DataFrame()
    
    # 3. Analyze news articles sentiment on-the-fly (prepares for FinBERT)
    news_summary = get_sentiment_summary(news_list)
    
    if overview:
        # A. Premium Dashboard Header Badge Panel
        render_header(overview)
        
        # B. High-Fidelity KPI Metrics Cards Row (Includes custom circular workforce Health Score)
        render_kpi_cards(overview, news_summary, history_df)
        
        # C. Two-Column Layout for Corporate Profile details and Speedometer gauge
        col_profile, col_gauge = st.columns([2, 1])
        
        with col_profile:
            render_company_overview(overview, history_df)
            
        with col_gauge:
            # Generate the professional risk speedometer indicator
            risk_score = overview["workforce_risk_score"]
            risk_level = overview["risk_level"]
            fig_gauge = render_risk_gauge(risk_score, risk_level)
            
            st.subheader("⚡ Risk Velocity Meter")
            st.plotly_chart(fig_gauge, use_container_width=True)
            
        # D. Dynamic Executive Assessment Summary Callout
        render_executive_summary(overview, news_summary)
        
        # E. Corporate Intelligence Tabs (Filters visual charts from news articles lists)
        tab_layoffs, tab_news, tab_ai = st.tabs([
            "📊 Volatility & Layoff Analytics", 
            "📰 News Coverage & Sentiment",
            "🤖 AI Risk Forecast"
        ])
        
        with tab_layoffs:
            if layoff_history:
                # Dual Y-Axis combination chart
                fig_combination = render_layoff_combination_chart(layoff_history)
                if fig_combination:
                    st.plotly_chart(fig_combination, use_container_width=True)
                    
                # Custom Simulated Risk Evolution Sparkline
                fig_sparkline = render_risk_sparkline(layoff_history, risk_score, risk_level)
                if fig_sparkline:
                    st.plotly_chart(fig_sparkline, use_container_width=True)
            else:
                st.info("No layoff chronology available for this organization.")
                
        with tab_news:
            render_news_panel(news_list)

        with tab_ai:
            render_ai_prediction_tab(selected_company)
            
    else:
        st.error(f"Failed to load risk data for company '{selected_company}'. Please verify database synchronization.")
else:
    st.info("Please select or search a company in the sidebar to review workforce intelligence indicators.")