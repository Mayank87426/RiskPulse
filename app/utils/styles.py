import streamlit as st

def inject_custom_css():
    st.markdown("""
    <style>
    /* Import Premium Typography */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global Font Overrides */
    html, body, [class*="css"], .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        letter-spacing: -0.02em;
    }

    /* Remove Streamlit default header/footer padding */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        padding-left: 3rem !important;
        padding-right: 3rem !important;
    }
    
    /* Styled Containers & Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        height: 100%;
        position: relative;
        overflow: hidden;
    }
    
    /* Light/Dark mode compatibility via transparent borders & soft shadows */
    [data-theme="light"] .metric-card {
        background: rgba(255, 255, 255, 0.85);
        border: 1px solid rgba(0, 0, 0, 0.08);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.2);
        border-color: rgba(255, 255, 255, 0.15);
    }
    
    [data-theme="light"] .metric-card:hover {
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        border-color: rgba(0, 0, 0, 0.12);
    }

    /* KPI Left-Borders by Risk Level */
    .border-very-low { border-left: 5px solid #10B981 !important; }
    .border-low { border-left: 5px solid #34D399 !important; }
    .border-medium { border-left: 5px solid #F59E0B !important; }
    .border-high { border-left: 5px solid #F97316 !important; }
    .border-critical { border-left: 5px solid #EF4444 !important; }
    
    /* KPI Content Styles */
    .kpi-title {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #9CA3AF;
        margin-bottom: 0.5rem;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    [data-theme="light"] .kpi-title {
        color: #4B5563;
    }

    .kpi-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: inherit;
        line-height: 1.2;
        font-family: 'Outfit', sans-serif;
    }

    .kpi-subtitle {
        font-size: 0.75rem;
        color: #6B7280;
        margin-top: 0.25rem;
    }

    .kpi-trend {
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 0.25rem;
        margin-top: 0.5rem;
    }
    .trend-up { color: #EF4444; }
    .trend-down { color: #10B981; }
    .trend-neutral { color: #9CA3AF; }

    /* Custom Badges */
    .pulse-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 0.35rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        gap: 0.35rem;
        border: 1px solid transparent;
    }
    
    .badge-very-low {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border-color: rgba(16, 185, 129, 0.25);
    }
    
    .badge-low {
        background-color: rgba(52, 211, 153, 0.15);
        color: #34D399;
        border-color: rgba(52, 211, 153, 0.25);
    }
    
    .badge-medium {
        background-color: rgba(245, 158, 11, 0.15);
        color: #F59E0B;
        border-color: rgba(245, 158, 11, 0.25);
    }
    
    .badge-high {
        background-color: rgba(249, 115, 22, 0.15);
        color: #F97316;
        border-color: rgba(249, 115, 22, 0.25);
    }
    
    .badge-critical {
        background-color: rgba(239, 68, 68, 0.15);
        color: #EF4444;
        border-color: rgba(239, 68, 68, 0.25);
        box-shadow: 0 0 10px rgba(239, 68, 68, 0.1);
        animation: pulse-border 2s infinite;
    }

    @keyframes pulse-border {
        0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.2); }
        70% { box-shadow: 0 0 0 6px rgba(239, 68, 68, 0); }
        100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }

    /* Sentiment Badges */
    .badge-positive {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10B981;
        border-color: rgba(16, 185, 129, 0.25);
    }
    
    .badge-neutral {
        background-color: rgba(156, 163, 175, 0.15);
        color: #9CA3AF;
        border-color: rgba(156, 163, 175, 0.25);
    }
    
    .badge-negative {
        background-color: rgba(239, 68, 68, 0.15);
        color: #EF4444;
        border-color: rgba(239, 68, 68, 0.25);
    }

    /* Info Badge (for overview) */
    .info-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.25rem 0.5rem;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #E5E7EB;
        font-size: 0.75rem;
        border-radius: 4px;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    [data-theme="light"] .info-badge {
        background: rgba(0, 0, 0, 0.03);
        border: 1px solid rgba(0, 0, 0, 0.08);
        color: #374151;
    }

    /* Premium News Article Cards */
    .news-card-link {
        text-decoration: none !important;
        color: inherit !important;
    }
    
    .news-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        transition: all 0.2s ease-in-out;
        cursor: pointer;
    }
    
    [data-theme="light"] .news-card {
        background: rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(0, 0, 0, 0.06);
    }

    .news-card:hover {
        background: rgba(255, 255, 255, 0.05);
        border-color: rgba(255, 255, 255, 0.12);
        transform: scale(1.01);
    }
    
    [data-theme="light"] .news-card:hover {
        background: rgba(249, 250, 251, 1);
        border-color: rgba(0, 0, 0, 0.1);
    }

    .news-meta {
        display: flex;
        justify-content: space-between;
        font-size: 0.7rem;
        color: #9CA3AF;
        margin-bottom: 0.4rem;
        align-items: center;
    }
    
    [data-theme="light"] .news-meta {
        color: #6B7280;
    }

    .news-headline {
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 0.25rem;
        line-height: 1.3;
    }

    .news-snippet {
        font-size: 0.8rem;
        color: #9CA3AF;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    
    [data-theme="light"] .news-snippet {
        color: #4B5563;
    }

    /* Executive Summary Callout */
    .exec-summary-box {
        background: linear-gradient(135deg, rgba(255,255,255,0.02) 0%, rgba(255,255,255,0.01) 100%);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        position: relative;
    }
    
    [data-theme="light"] .exec-summary-box {
        background: linear-gradient(135deg, rgba(255,255,255,1) 0%, rgba(249,250,251,1) 100%);
        border: 1px solid rgba(0, 0, 0, 0.06);
    }
    
    .exec-summary-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 4px;
        height: 100%;
        background: linear-gradient(to bottom, #3b82f6, #8b5cf6);
        border-top-left-radius: 12px;
        border-bottom-left-radius: 12px;
    }

    /* Circular Progress Indicator Container */
    .health-score-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 1rem;
        height: 100%;
    }

    .circular-progress {
        position: relative;
        width: 120px;
        height: 120px;
    }

    .circular-progress svg {
        width: 100%;
        height: 100%;
        transform: rotate(-90deg);
    }

    .circular-progress circle {
        fill: none;
        stroke-width: 10;
    }

    .circular-progress .bg-circle {
        stroke: rgba(255, 255, 255, 0.05);
    }
    
    [data-theme="light"] .circular-progress .bg-circle {
        stroke: rgba(0, 0, 0, 0.05);
    }

    .circular-progress .progress-circle {
        stroke-linecap: round;
        transition: stroke-dashoffset 0.8s ease-in-out;
    }

    .circular-progress .percentage-text {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        font-size: 1.5rem;
        font-weight: 700;
        font-family: 'Outfit', sans-serif;
        color: inherit;
    }

    /* Sidebar Status Badges */
    .sidebar-status-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.35rem 0;
        font-size: 0.75rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    [data-theme="light"] .sidebar-status-item {
        border-bottom: 1px solid rgba(0, 0, 0, 0.05);
    }

    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        font-weight: 600;
    }

    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
    }
    
    .dot-green {
        background-color: #10B981;
        box-shadow: 0 0 8px #10B981;
    }
    
    .dot-red {
        background-color: #EF4444;
        box-shadow: 0 0 8px #EF4444;
    }

    /* Custom scrollbars for columns */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
    }
    [data-theme="light"] ::-webkit-scrollbar-thumb {
        background: rgba(0, 0, 0, 0.1);
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(255, 255, 255, 0.2);
    }
    </style>
    """, unsafe_allow_html=True)
