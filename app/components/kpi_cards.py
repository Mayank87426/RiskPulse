import streamlit as st
from utils.config import COLORS, BG_COLORS
from utils.helpers import format_number, calculate_days_since, calculate_health_score

def render_kpi_cards(overview, news_summary, history_df):
    """Renders high-fidelity HTML metric cards and the Health Score circle in columns."""
    if not overview:
        return
        
    risk_score = overview["workforce_risk_score"]
    risk_level = overview["risk_level"]
    layoff_events = overview.get("layoff_events", 0)
    total_laid_off = overview.get("total_employees_laid_off", 0)
    max_pct = float(overview.get("max_percentage_laid_off", 0) or 0)
    latest_layoff = overview.get("latest_layoff")
    
    # Calculate Days Since Last Layoff
    days_since = calculate_days_since(latest_layoff)
    days_str = f"{days_since} days" if days_since is not None else "No layoffs"
    
    # Calculate Largest Layoff from history
    largest_layoff = 0
    avg_layoff = 0
    if history_df is not None and not history_df.empty:
        largest_layoff = history_df["employees_laid_off"].max()
        avg_layoff = history_df["employees_laid_off"].mean()
        
    # Calculate Health Score
    news_sentiment_score = news_summary.get("avg_score", 0.0)
    health_score = calculate_health_score(
        risk_score, 
        news_sentiment_score, 
        layoff_events, 
        days_since
    )
    
    # Determine Health Label and Color
    if health_score >= 80:
        health_label = "Excellent"
        health_color = "#10B981"  # Emerald
    elif health_score >= 60:
        health_label = "Stable"
        health_color = "#34D399"  # Light Emerald
    elif health_score >= 40:
        health_label = "Vulnerable"
        health_color = "#F59E0B"  # Yellow/Amber
    elif health_score >= 20:
        health_label = "Stressed"
        health_color = "#F97316"  # Orange
    else:
        health_label = "Critical Crisis"
        health_color = "#EF4444"  # Red
        
    # Circunference of R=50 is 314
    dashoffset = 314 - (health_score / 100.0) * 314
    
    # Border class based on Risk Level
    border_class = f"border-{risk_level.lower().replace(' ', '-')}"
    
    # -----------------------------
    # Render layout columns
    # -----------------------------
    
    # Grid Row 1: Risk, Events, Laid Off, Health
    col1, col2, col3, col4 = st.columns(4)
    
    # Card 1: Risk Score
    with col1:
        st.markdown(f"""
            <div class="metric-card {border_class}">
                <div class="kpi-title">
                    <span>🔥</span>
                    <span>Risk Score</span>
                </div>
                <div class="kpi-value">{risk_score}<span style="font-size: 1rem; font-weight: 500; color: #6B7280;">/100</span></div>
                <div class="kpi-subtitle">Risk Level: <b>{risk_level}</b></div>
                <div class="kpi-trend trend-{"up" if risk_score >= 60 else "down" if risk_score <= 20 else "neutral"}">
                    { "↑ Elevated Risk" if risk_score >= 60 else "↓ Low Volatility" if risk_score <= 20 else "→ Standard Volatility" }
                </div>
            </div>
        """, unsafe_allow_html=True)
        
    # Card 2: Layoff Events
    with col2:
        events_trend = "up" if layoff_events >= 4 else "neutral" if layoff_events >= 1 else "down"
        st.markdown(f"""
            <div class="metric-card {border_class}">
                <div class="kpi-title">
                    <span>📊</span>
                    <span>Layoff Events</span>
                </div>
                <div class="kpi-value">{layoff_events}</div>
                <div class="kpi-subtitle">Total historical actions</div>
                <div class="kpi-trend trend-{events_trend}">
                    { "↑ High Frequency" if layoff_events >= 4 else "→ Recurring" if layoff_events >= 2 else "↓ Stable Capacity" }
                </div>
            </div>
        """, unsafe_allow_html=True)
        
    # Card 3: Employees Laid Off
    with col3:
        employees_trend = "up" if total_laid_off >= 1000 else "down" if total_laid_off == 0 else "neutral"
        laid_off_str = format_number(total_laid_off, "0")
        st.markdown(f"""
            <div class="metric-card {border_class}">
                <div class="kpi-title">
                    <span>👥</span>
                    <span>Employees Laid Off</span>
                </div>
                <div class="kpi-value">{laid_off_str}</div>
                <div class="kpi-subtitle">Total workforce reduction</div>
                <div class="kpi-trend trend-{employees_trend}">
                    { "↑ High Volume" if total_laid_off >= 1000 else "↓ Zero cuts" if total_laid_off == 0 else "→ Moderate Volume" }
                </div>
            </div>
        """, unsafe_allow_html=True)
        
    # Card 4: Circular Health Score
    with col4:
        st.markdown(f"""
            <div class="metric-card {border_class}" style="padding: 0.75rem;">
                <div class="health-score-container">
                    <div class="circular-progress">
                        <svg>
                            <circle class="bg-circle" cx="60" cy="60" r="50"></circle>
                            <circle class="progress-circle" cx="60" cy="60" r="50" 
                                    style="stroke: {health_color}; stroke-dasharray: 314; stroke-dashoffset: {dashoffset};"></circle>
                        </svg>
                        <div class="percentage-text" style="color: {health_color};">{health_score}%</div>
                    </div>
                    <div style="font-size: 0.75rem; text-transform: uppercase; color: #9CA3AF; margin-top: 0.5rem; font-weight: 600;">Workforce Health</div>
                    <div style="font-size: 0.7rem; color: {health_color}; font-weight: 500;">{health_label}</div>
                </div>
            </div>
        """, unsafe_allow_html=True)

    # Grid Row 2: Largest Layoff, Max reduction %, Days Since last layoff, average layoff
    col1_2, col2_2, col3_2, col4_2 = st.columns(4)
    
    # Card 5: Largest Layoff
    with col1_2:
        largest_str = format_number(largest_off := int(largest_layoff) if largest_layoff else 0, "0")
        largest_trend = "up" if largest_off >= 500 else "neutral" if largest_off > 0 else "down"
        st.markdown(f"""
            <div class="metric-card {border_class}">
                <div class="kpi-title">
                    <span>💥</span>
                    <span>Largest Layoff</span>
                </div>
                <div class="kpi-value">{largest_str}</div>
                <div class="kpi-subtitle">Single event record</div>
                <div class="kpi-trend trend-{largest_trend}">
                    { "↑ Massive Cut" if largest_off >= 500 else "↓ None" if largest_off == 0 else "→ Standard Cut" }
                </div>
            </div>
        """, unsafe_allow_html=True)
        
    # Card 6: Max Workforce Reduction %
    with col2_2:
        reduction_trend = "up" if max_pct >= 25.0 else "neutral" if max_pct >= 10.0 else "down"
        st.markdown(f"""
            <div class="metric-card {border_class}">
                <div class="kpi-title">
                    <span>📉</span>
                    <span>Max Workforce Reduction</span>
                </div>
                <div class="kpi-value">{max_pct:.1f}%</div>
                <div class="kpi-subtitle">Peak workforce reduction ratio</div>
                <div class="kpi-trend trend-{reduction_trend}">
                    { "↑ Severe (25%+)" if max_pct >= 25.0 else "↓ Minimal" if max_pct == 0.0 else "→ Moderate" }
                </div>
            </div>
        """, unsafe_allow_html=True)
        
    # Card 7: Days Since Last Layoff
    with col3_2:
        if days_since is None:
            days_trend = "down"
            days_status = "↓ Stable"
        else:
            days_trend = "down" if days_since > 180 else "up" if days_since < 90 else "neutral"
            days_status = "↓ Aging Risk" if days_since > 180 else "↑ Active Risk" if days_since < 90 else "→ Normalizing"
            
        st.markdown(f"""
            <div class="metric-card {border_class}">
                <div class="kpi-title">
                    <span>⏱️</span>
                    <span>Days Since Layoff</span>
                </div>
                <div class="kpi-value">{days_str}</div>
                <div class="kpi-subtitle">Time elapsed since last action</div>
                <div class="kpi-trend trend-{days_trend}">
                    {days_status}
                </div>
            </div>
        """, unsafe_allow_html=True)
        
    # Card 8: Average Layoff Volume
    with col4_2:
        avg_str = format_number(int(avg_layoff) if avg_layoff else 0, "0")
        avg_trend = "up" if avg_layoff >= 300 else "neutral" if avg_layoff > 0 else "down"
        st.markdown(f"""
            <div class="metric-card {border_class}">
                <div class="kpi-title">
                    <span>📐</span>
                    <span>Average Layoff Size</span>
                </div>
                <div class="kpi-value">{avg_str}</div>
                <div class="kpi-subtitle">Mean staff reduction per event</div>
                <div class="kpi-trend trend-{avg_trend}">
                    { "↑ Severe Size" if avg_layoff >= 300 else "↓ Safe Bounds" if avg_layoff == 0 else "→ Moderate Size" }
                </div>
            </div>
        """, unsafe_allow_html=True)
