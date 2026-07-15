import streamlit as st
from utils.helpers import format_number
import textwrap

def render_company_overview(overview, history_df):
    """Renders the Company Overview card and the Risk Factors list."""
    if not overview:
        return
        
    industry = overview.get("industry", "N/A")
    country = overview.get("country", "N/A")
    stage = overview.get("stage", "N/A")
    raised = overview.get("money_raised_mil")
    raised_str = f"${float(raised):,.1f} Million" if raised else "N/A"
    
    # Extract Company Size from the history if available
    size_before = "N/A"
    size_after = "N/A"
    if history_df is not None and not history_df.empty:
        # Sort by date descending to find the latest size metrics
        df_sorted = history_df.sort_values(by="layoff_date", ascending=False)
        for _, row in df_sorted.iterrows():
            if row.get("company_size_before") and row.get("company_size_before") > 0:
                size_before = format_number(int(row["company_size_before"]))
                size_after = format_number(int(row["company_size_after"])) if row.get("company_size_after") else "N/A"
                break
                
    st.subheader("🏢 Corporate Profile")
    
    profile_html = textwrap.dedent(f"""
        <div style="background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.05); border-radius: 12px; padding: 1.25rem; margin-bottom: 1.5rem;">
            <table style="width: 100%; border-collapse: collapse; font-size: 0.85rem;">
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                    <td style="padding: 0.6rem 0; color: #9CA3AF; font-weight: 500;">Sector / Industry</td>
                    <td style="padding: 0.6rem 0; text-align: right; font-weight: 600;">{industry}</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                    <td style="padding: 0.6rem 0; color: #9CA3AF; font-weight: 500;">Country Headquarters</td>
                    <td style="padding: 0.6rem 0; text-align: right; font-weight: 600;">{country}</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                    <td style="padding: 0.6rem 0; color: #9CA3AF; font-weight: 500;">Corporate Stage</td>
                    <td style="padding: 0.6rem 0; text-align: right; font-weight: 600;">{stage}</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                    <td style="padding: 0.6rem 0; color: #9CA3AF; font-weight: 500;">Total Funding</td>
                    <td style="padding: 0.6rem 0; text-align: right; font-weight: 600; color: #3b82f6;">{raised_str}</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                    <td style="padding: 0.6rem 0; color: #9CA3AF; font-weight: 500;">Staff Count Before Layoffs</td>
                    <td style="padding: 0.6rem 0; text-align: right; font-weight: 600;">{size_before}</td>
                </tr>
                <tr style="border-bottom: none;">
                    <td style="padding: 0.6rem 0; color: #9CA3AF; font-weight: 500;">Staff Count Post Layoffs</td>
                    <td style="padding: 0.6rem 0; text-align: right; font-weight: 600;">{size_after}</td>
                </tr>
            </table>
        </div>
    """)
    st.markdown(profile_html, unsafe_allow_html=True)
    
    # Render Risk Factors Badges
    st.subheader("⚠️ Risk Identifiers")
    reasons_str = overview.get("risk_reasons", "")
    if reasons_str:
        reasons = [r.strip() for r in reasons_str.split(";") if r.strip()]
        badges_html = ""
        for reason in reasons:
            badges_html += textwrap.dedent(f"""
                <div style="display: flex; align-items: center; gap: 0.75rem; padding: 0.65rem 0.85rem; margin-bottom: 0.5rem; background: rgba(239, 68, 68, 0.05); border: 1px solid rgba(239, 68, 68, 0.15); border-radius: 8px;">
                    <span style="color: #EF4444; font-size: 1rem; font-weight: bold;">✓</span>
                    <span style="font-size: 0.85rem; font-weight: 500; color: #F3F4F6;">{reason}</span>
                </div>
            """)
        st.markdown(f"<div>{badges_html}</div>", unsafe_allow_html=True)
    else:
        no_flags_html = textwrap.dedent("""
            <div style="display: flex; align-items: center; gap: 0.75rem; padding: 0.65rem 0.85rem; background: rgba(16, 185, 129, 0.05); border: 1px solid rgba(16, 185, 129, 0.15); border-radius: 8px;">
                <span style="color: #10B981; font-size: 1rem; font-weight: bold;">✓</span>
                <span style="font-size: 0.85rem; font-weight: 500; color: #F3F4F6;">No risk flags detected in this period</span>
            </div>
        """)
        st.markdown(no_flags_html, unsafe_allow_html=True)
