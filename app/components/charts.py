import plotly.graph_objects as go
import pandas as pd
import datetime
from utils.config import COLORS

def render_layoff_combination_chart(history):
    """Generates a high-fidelity combination chart (dual y-axis) for layoff history."""
    if not history:
        return None
        
    df = pd.DataFrame(history)
    df["layoff_date"] = pd.to_datetime(df["layoff_date"])
    df = df.sort_values(by="layoff_date")
    
    fig = go.Figure()
    
    # 1. Bar Chart: Employees Laid Off (Primary Y)
    fig.add_trace(
        go.Bar(
            x=df["layoff_date"],
            y=df["employees_laid_off"],
            name="Employees Laid Off",
            marker=dict(
                color="rgba(59, 130, 246, 0.75)", # Soft blue
                line=dict(color="#3b82f6", width=1.5)
            ),
            yaxis="y1",
            hovertemplate="<b>Date:</b> %{x|%b %d, %Y}<br>" +
                          "<b>Employees Laid Off:</b> %{y:,.0f}<extra></extra>"
        )
    )
    
    # 2. Line Chart: Percentage Laid Off (Secondary Y)
    fig.add_trace(
        go.Scatter(
            x=df["layoff_date"],
            y=df["percentage_laid_off"],
            name="Workforce Reduction %",
            mode="lines+markers",
            line=dict(color="#EF4444", width=3, shape="spline"),
            marker=dict(size=7, color="#EF4444", line=dict(color="#FFFFFF", width=1)),
            yaxis="y2",
            hovertemplate="<b>Date:</b> %{x|%b %d, %Y}<br>" +
                          "<b>Workforce Reduction:</b> %{y:.1f}%<extra></extra>"
        )
    )
    
    # Dual Y-Axis layout
    fig.update_layout(
        title={
            "text": "Workforce Reduction Chronology",
            "font": {"family": "Outfit, sans-serif", "size": 18}
        },
        xaxis=dict(
            title="Timeline",
            gridcolor="rgba(255, 255, 255, 0.05)",
            showgrid=True
        ),
        yaxis=dict(
            title=dict(
                text="Employees Laid Off",
                font=dict(color="#3b82f6")
            ),
            tickfont=dict(color="#3b82f6"),
            gridcolor="rgba(255, 255, 255, 0.05)",
            showgrid=True
        ),
        yaxis2=dict(
            title=dict(
                text="Workforce Reduction (%)",
                font=dict(color="#EF4444")
            ),
            tickfont=dict(color="#EF4444"),
            overlaying="y",
            side="right",
            showgrid=False
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#E5E7EB"),
        margin=dict(l=20, r=20, t=60, b=20),
        height=320
    )
    
    return fig

def render_risk_sparkline(history, current_score, risk_level):
    """
    Simulates and renders an elegant sparkline showing historical risk score evolution.
    If historical scores do not exist, they are simulated based on layoff chronology
    with a daily decay of -0.05 points and boosting spikes on layoff events.
    """
    # Define start date (e.g. 3 years ago or 1 year before first layoff)
    today = datetime.date.today()
    start_date = today - datetime.timedelta(days=365 * 3)
    
    points = []
    
    # Default time series if there is no layoff history
    if not history or len(history) == 0:
        # Create a stable low risk line
        date_range = [start_date + datetime.timedelta(days=x) for x in range(0, 3 * 365, 30)]
        for d in date_range:
            points.append({"date": d, "score": current_score})
    else:
        df = pd.DataFrame(history)
        df["layoff_date"] = pd.to_datetime(df["layoff_date"]).dt.date
        df = df.sort_values(by="layoff_date")
        
        # Sort and group by date (in case of multiple events on the same day)
        grouped = df.groupby("layoff_date").agg({
            "employees_laid_off": "sum",
            "percentage_laid_off": "max"
        }).reset_index()
        
        # Simulation parameters
        base_score = 10.0
        running_score = base_score
        
        # Start timeline one month before the first layoff
        first_layoff_date = grouped["layoff_date"].iloc[0]
        curr_date = first_layoff_date - datetime.timedelta(days=30)
        points.append({"date": curr_date, "score": running_score})
        
        for _, row in grouped.iterrows():
            event_date = row["layoff_date"]
            laid_off = float(row["employees_laid_off"] or 0)
            pct = float(row["percentage_laid_off"] or 0)
            
            # Days since last point
            days = (event_date - curr_date).days
            if days > 0:
                # 1. Decay the score daily (-0.08 risk point decay per day)
                running_score = max(base_score, running_score - (0.08 * days))
                
            # 2. Boost the score based on layoff magnitude
            # Base boost + head count penalty + percentage penalty
            boost = 15.0
            boost += min(laid_off / 150.0, 20.0) # Up to 20 pts for size
            boost += min(pct * 0.4, 25.0)        # Up to 25 pts for reduction percentage
            
            running_score = min(100.0, running_score + boost)
            points.append({"date": event_date, "score": running_score})
            curr_date = event_date
            
        # Decay to today
        days_to_today = (today - curr_date).days
        if days_to_today > 0:
            running_score = max(base_score, running_score - (0.08 * days_to_today))
            
        # Match today's simulated score with the actual current score
        points.append({"date": today, "score": float(current_score)})
        
    spark_df = pd.DataFrame(points)
    spark_df["date"] = pd.to_datetime(spark_df["date"])
    
    # Generate Sparkline Figure
    spark_color = COLORS.get(risk_level, "#F59E0B")
    
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=spark_df["date"],
            y=spark_df["score"],
            mode="lines",
            line=dict(color=spark_color, width=2.5, shape="spline"),
            fill="tozeroy",
            fillcolor=f"rgba{tuple(list(int(spark_color[i:i+2], 16) for i in (1, 3, 5)) + [0.08])}", # Dynamic RGBA from HEX
            hovertemplate="<b>Date:</b> %{x|%b %Y}<br><b>Risk Index:</b> %{y:.1f}/100<extra></extra>"
        )
    )
    
    fig.update_layout(
        title={
            "text": "Risk Score Evolution Sparkline",
            "font": {"family": "Outfit, sans-serif", "size": 15}
        },
        xaxis=dict(
            visible=True,
            showgrid=False,
            gridcolor="rgba(255,255,255,0.02)",
            tickfont=dict(size=9, color="#9CA3AF")
        ),
        yaxis=dict(
            range=[0, 105],
            showgrid=True,
            gridcolor="rgba(255,255,255,0.05)",
            tickfont=dict(size=9, color="#9CA3AF")
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#E5E7EB"),
        margin=dict(l=10, r=10, t=40, b=20),
        height=140
    )
    
    return fig
