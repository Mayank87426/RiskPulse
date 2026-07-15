import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from utils.config import COLORS
from utils.helpers import get_sentiment_summary
import textwrap

def render_news_panel(news_list):
    """Renders the sentiment analytics cards, pie chart, and list of news articles."""
    st.subheader("📰 Sentiment Analysis & News Intelligence")
    
    if not news_list:
        st.info("No news articles currently cached in database for this company. News sentiment is neutral.")
        # Renders an empty-state sentiment block
        render_sentiment_summary(None)
        return
        
    # Get sentiment summary (calculates values on-the-fly and modifies dicts in-place)
    summary = get_sentiment_summary(news_list)
    
    # -----------------------------
    # Render layout columns
    # -----------------------------
    left_col, right_col = st.columns([1, 1])
    
    with left_col:
        render_sentiment_summary(summary)
        
    with right_col:
        render_sentiment_pie(summary)
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("Latest Coverage")
    
    # Renders news cards list
    for item in news_list:
        title = item.get("title", "N/A")
        source = item.get("source", "Unknown Source")
        published_at = item.get("published_at")
        published_str = published_at.strftime("%b %d, %Y") if published_at else "Recent"
        url = item.get("article_url", "#")
        description = item.get("description", "")
        sentiment = item.get("sentiment", "Neutral")
        score = float(item.get("sentiment_score", 0.0))
        
        # Calculate a mock confidence score to structure FinBERT compatibility
        confidence = round(0.70 + abs(score) * 0.25, 2)
        confidence_str = f"{int(confidence * 100)}%"
        
        sent_class = f"badge-{sentiment.lower()}"
        
        card_html = textwrap.dedent(f"""
            <a href="{url}" target="_blank" class="news-card-link">
                <div class="news-card">
                    <div class="news-meta">
                        <span>📰 {source} • {published_str}</span>
                        <div style="display: flex; gap: 0.5rem; align-items: center;">
                            <span class="pulse-badge {sent_class}">{sentiment}</span>
                            <span style="color: #6B7280; font-size: 0.65rem;">Conf: <b>{confidence_str}</b></span>
                        </div>
                    </div>
                    <div class="news-headline">{title}</div>
                    <div class="news-snippet">{description}</div>
                </div>
            </a>
        """)
        st.markdown(card_html, unsafe_allow_html=True)

def render_sentiment_summary(summary):
    """Renders the HTML sentiment score metric and progress bar."""
    if not summary:
        avg_score = 0.0
        percentage = 50.0
        label = "Neutral"
        pos, neu, neg = 0, 0, 0
    else:
        avg_score = summary["avg_score"]
        percentage = summary["percentage"]
        label = summary["label"]
        pos, neu, neg = summary["Positive"], summary["Neutral"], summary["Negative"]
        
    # Get color
    sent_color = COLORS.get(label, "#9CA3AF")
    
    summary_html = textwrap.dedent(f"""
        <div style="background: rgba(255,255,255,0.01); border: 1px solid rgba(255,255,255,0.05); border-radius: 12px; padding: 1.25rem; height: 100%; display: flex; flex-direction: column; justify-content: center;">
            <div style="font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; color: #9CA3AF; margin-bottom: 0.5rem; font-weight: 500;">
                Overall News Sentiment
            </div>
            <div style="font-size: 2.2rem; font-weight: 800; color: {sent_color}; font-family: 'Outfit', sans-serif;">
                {avg_score:+.2f} 
                <span style="font-size: 1.1rem; font-weight: 500; color: #6B7280;">({label})</span>
            </div>
            <div style="margin-top: 1rem; width: 100%;">
                <div style="width: 100%; height: 8px; background: rgba(255, 255, 255, 0.06); border-radius: 4px; position: relative;">
                    <div style="position: absolute; left: 0; top: 0; height: 100%; width: {percentage}%; background: {sent_color}; border-radius: 4px; transition: width 0.5s ease-in-out;"></div>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 0.75rem; color: #6B7280; margin-top: 0.5rem;">
                    <span>Negative (-1.0)</span>
                    <span>Neutral (0.0)</span>
                    <span>Positive (+1.0)</span>
                </div>
            </div>
            <div style="display: flex; justify-content: space-between; margin-top: 1rem; font-size: 0.8rem; color: #9CA3AF; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 0.75rem;">
                <span>🟢 <b>Positive:</b> {pos}</span>
                <span>⚪ <b>Neutral:</b> {neu}</span>
                <span>🔴 <b>Negative:</b> {neg}</span>
            </div>
        </div>
    """)
    st.markdown(summary_html, unsafe_allow_html=True)

def render_sentiment_pie(summary):
    """Renders a styled Plotly Pie chart showing sentiment shares."""
    if not summary or (summary["Positive"] == 0 and summary["Neutral"] == 0 and summary["Negative"] == 0):
        # Empty placeholder
        fig = go.Figure()
        fig.add_annotation(text="No article distribution data", showarrow=False, font=dict(size=14, color="#6B7280"))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=180,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)
        return
        
    labels = ["Positive", "Neutral", "Negative"]
    values = [summary["Positive"], summary["Neutral"], summary["Negative"]]
    colors = [COLORS["Positive"], COLORS["Neutral"], COLORS["Negative"]]
    
    # Filter out labels with 0 counts
    filtered_labels = []
    filtered_values = []
    filtered_colors = []
    for l, v, c in zip(labels, values, colors):
        if v > 0:
            filtered_labels.append(l)
            filtered_values.append(v)
            filtered_colors.append(c)
            
    fig = go.Figure(
        go.Pie(
            labels=filtered_labels,
            values=filtered_values,
            hole=0.45,
            marker=dict(colors=filtered_colors, line=dict(color="#1F2937", width=1.5)),
            textinfo="percent+label",
            textfont=dict(size=10, family="Inter, sans-serif", color="#FFFFFF"),
            hoverinfo="label+value"
        )
    )
    
    fig.update_layout(
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=10),
        height=180
    )
    
    st.plotly_chart(fig, use_container_width=True)
