import streamlit as st
from utils.helpers import format_number
import textwrap

def render_executive_summary(overview, news_summary):
    """Generates and renders a professional executive summary for the company."""
    if not overview:
        return
        
    company_name = overview["name"]
    risk_score = overview["workforce_risk_score"]
    risk_level = overview["risk_level"]
    events = overview.get("layoff_events", 0)
    total_laid_off = overview.get("total_employees_laid_off", 0)
    max_pct = overview.get("max_percentage_laid_off", 0)
    news_label = news_summary.get("label", "Neutral")
    
    # 1. Workforce Risk Sentence
    risk_str = f"**{company_name}** currently exhibits a **{risk_level} Workforce Risk Profile** with an intelligence score of **{risk_score}/100**."
    
    # 2. Layoff Chronology Sentence
    if events > 0:
        laid_off_formatted = format_number(total_laid_off)
        events_str = "layoff event" if events == 1 else "layoff events"
        layoff_str = f"The organization has undergone **{events} {events_str}**, resulting in the separation of **{laid_off_formatted} employees**."
        
        # Max reduction mention
        if max_pct and max_pct > 0:
            layoff_str += f" The largest single event reduced the company workforce by **{max_pct:.1f}%**."
    else:
        layoff_str = "There are no recorded layoff events for this organization in our database, suggesting high historical workforce capacity stability."
        
    # 3. Operational Risk Factors (Conditional)
    if risk_score >= 80:
        factor_str = "Frequent or large-scale workforce reductions indicate elevated operational and cultural risks, suggesting potential structural challenges or severe realignment efforts."
    elif risk_score >= 40:
        factor_str = "Moderate layoff activity indicates structured organizational adjustment, presenting moderate operational risk but pointing toward ongoing optimization."
    else:
        factor_str = "Minimal or non-existent layoffs reflect strong employee retention metrics, minimizing immediate business continuity and transition risks."
        
    # 4. News Sentiment Context
    if news_label == "Negative":
        news_str = "Public news sentiment is currently **Negative**, driven by recent corporate announcements, market challenges, or layoff publicity."
    elif news_label == "Positive":
        news_str = "External public sentiment remains **Positive**, highlighting product growth, capital raises, or favorable strategic directions."
    else:
        news_str = "Public news sentiment is currently **Neutral**, indicating stable public relations and normal operational reporting."
        
    # 5. Conclusion
    if risk_score >= 80:
        conclusion_str = "Overall, workforce stability is **highly volatile**, requiring close monitoring and risk mitigation protocols."
    elif risk_score >= 40:
        conclusion_str = "Overall, workforce stability is **moderate**, with stability indices tracking within normal bounds for the industry."
    else:
        conclusion_str = "Overall, workforce stability remains **exceptionally strong**, positioning the company as a low-risk counterparty."
        
    st.subheader("📝 Executive Summary")
    
    summary_html = textwrap.dedent(f"""
        <div class="exec-summary-box">
            <div style="font-size: 0.95rem; line-height: 1.6; color: inherit;">
                <p style="margin-bottom: 0.75rem;">{risk_str} {layoff_str}</p>
                <p style="margin-bottom: 0.75rem;">{factor_str} {news_str}</p>
                <p style="margin-bottom: 0;">{conclusion_str}</p>
            </div>
        </div>
    """)
    st.markdown(summary_html, unsafe_allow_html=True)
