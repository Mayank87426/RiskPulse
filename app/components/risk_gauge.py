import plotly.graph_objects as go
from utils.config import COLORS

def render_risk_gauge(score, risk_level):
    """Generates a premium, styled Plotly indicator gauge matching the design system."""
    
    # Select color matching the risk level
    score_color = COLORS.get(risk_level, "#F59E0B")
    
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={
                'text': f"Risk Status: <span style='color:{score_color}; font-weight:700;'>{risk_level}</span>",
                'font': {'size': 18, 'family': 'Outfit, sans-serif'}
            },
            number={
                'font': {'size': 54, 'family': 'Outfit, sans-serif'},
                'suffix': ""
            },
            gauge={
                'axis': {
                    'range': [0, 100],
                    'tickwidth': 1,
                    'tickcolor': "#4B5563",
                    'tickvals': [0, 20, 40, 60, 80, 100],
                    'ticktext': ['0', '20', '40', '60', '80', '100']
                },
                'bar': {
                    'color': '#374151',
                    'thickness': 0.25
                },
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 20], 'color': 'rgba(16, 185, 129, 0.7)'},    # Emerald Green
                    {'range': [20, 40], 'color': 'rgba(52, 211, 153, 0.7)'},   # Light Emerald
                    {'range': [40, 60], 'color': 'rgba(245, 158, 11, 0.7)'},   # Yellow
                    {'range': [60, 80], 'color': 'rgba(249, 115, 22, 0.7)'},   # Orange
                    {'range': [80, 100], 'color': 'rgba(239, 68, 68, 0.7)'}    # Red
                ],
                'threshold': {
                    'line': {'color': score_color, 'width': 6},
                    'thickness': 0.8,
                    'value': score
                }
            }
        )
    )
    
    # Styling updates for a modern dashboard widget
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#E5E7EB", 'family': "Inter, sans-serif"},
        margin=dict(l=20, r=20, t=50, b=20),
        height=260
    )
    
    return fig
