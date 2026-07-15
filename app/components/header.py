import streamlit as st
from utils.config import COLORS, RISK_EMOJIS
from utils.helpers import format_number
import textwrap

def render_header(overview):
    """Renders a premium enterprise dashboard header for the selected company."""
    if not overview:
        return
        
    company_name = overview["name"]
    industry = overview.get("industry", "N/A")
    country = overview.get("country", "N/A")
    stage = overview.get("stage", "N/A")
    risk_level = overview.get("risk_level", "Medium")
    latest_layoff = overview.get("latest_layoff")
    generated_at = overview.get("generated_at")
    
    # Format dates nicely
    layoff_str = latest_layoff.strftime("%b %d, %Y") if latest_layoff else "No layoffs recorded"
    updated_str = generated_at.strftime("%b %d, %Y at %H:%M") if generated_at else "N/A"
    
    # Custom Risk Badge Color CSS Mapping
    risk_class = f"badge-{risk_level.lower().replace(' ', '-')}"
    emoji = RISK_EMOJIS.get(risk_level, "⚪")
    
    # Money Raised
    raised = overview.get("money_raised_mil")
    raised_str = f"${float(raised):,.1f}M" if raised else "N/A"
    
    header_html = textwrap.dedent(f"""
        <div style="margin-bottom: 2rem; padding: 1.5rem; background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.05); border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.15);">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 1rem;">
                <div>
                    <div style="display: flex; align-items: center; gap: 1rem; flex-wrap: wrap; margin-bottom: 0.5rem;">
                        <h1 style="margin: 0; font-size: 2.2rem; font-weight: 800; font-family: 'Outfit', sans-serif; display: flex; align-items: center; gap: 0.5rem;">
                            🏢 {company_name}
                        </h1>
                        <span class="pulse-badge {risk_class}">
                            {emoji} {risk_level} Risk
                        </span>
                    </div>
                    <div style="display: flex; gap: 1.5rem; flex-wrap: wrap; font-size: 0.85rem; color: #9CA3AF; margin-top: 0.5rem;">
                        <span style="display: flex; align-items: center; gap: 0.35rem;">
                            📂 <b>Industry:</b> {industry}
                        </span>
                        <span style="display: flex; align-items: center; gap: 0.35rem;">
                            📍 <b>Location:</b> {country}
                        </span>
                        <span style="display: flex; align-items: center; gap: 0.35rem;">
                            🚀 <b>Stage:</b> {stage}
                        </span>
                        <span style="display: flex; align-items: center; gap: 0.35rem;">
                            💰 <b>Funding:</b> {raised_str}
                        </span>
                    </div>
                </div>
                <div style="text-align: right; min-width: 200px; display: flex; flex-direction: column; gap: 0.25rem;">
                    <div style="font-size: 0.8rem; color: #6B7280;">
                        📅 <b>Latest Layoff:</b> <span style="color: #E5E7EB; font-weight: 500;">{layoff_str}</span>
                    </div>
                    <div style="font-size: 0.8rem; color: #6B7280;">
                        ⏱️ <b>Data Updated:</b> <span style="color: #E5E7EB; font-weight: 500;">{updated_str}</span>
                    </div>
                </div>
            </div>
        </div>
    """)
    st.markdown(header_html, unsafe_allow_html=True)
