from datetime import datetime, date
import re

def format_number(val, default="N/A", format_str="{:,}"):
    """Helper to format numbers with commas and handle None values."""
    if val is None:
        return default
    try:
        return format_str.format(val)
    except (ValueError, TypeError):
        return default

def calculate_days_since(target_date):
    """Calculates the number of days between target_date and today."""
    if not target_date:
        return None
    if isinstance(target_date, str):
        try:
            target_date = datetime.strptime(target_date, "%Y-%m-%d").date()
        except ValueError:
            return None
    elif isinstance(target_date, datetime):
        target_date = target_date.date()
        
    today = date.today()
    return (today - target_date).days

def analyze_article_sentiment(title, description, content):
    """
    Perform a fast keyword-based sentiment analysis for an article.
    Prepares the architecture for a future FinBERT integration.
    """
    title_text = str(title or "").lower()
    desc_text = str(description or "").lower()
    content_text = str(content or "").lower()
    
    # Combined search spaces with weighting
    full_text = f"{title_text} {title_text} {desc_text} {content_text}"
    
    # Word list triggers
    neg_words = [
        r"\blayoffs?\b", r"\bcuts?\b", r"\bdownsized?\b", r"\bdownsizings?\b", 
        r"\breductions?\b", r"\blosses?\b", r"\bdecline\b", r"\bwarns?\b", 
        r"\bshutting\b", r"\bbankruptcy\b", r"\bbankrupt\b", r"\blawsuits?\b", 
        r"\bsued?\b", r"\bcrisis\b", r"\bplunges?\b", r"\bfired?\b", 
        r"\bfirings?\b", r"\bdebt\b", r"\bdeficit\b", r"\bstruggles?\b"
    ]
    
    pos_words = [
        r"\bhiring\b", r"\bhires?\b", r"\bgrows?\b", r"\bgrowth\b", 
        r"\bprofits?\b", r"\bprofitable\b", r"\bexpansions?\b", r"\bexpands?\b", 
        r"\bacquires?\b", r"\bacquisitions?\b", r"\bpartnerships?\b", r"\bpartners?\b", 
        r"\bsuccess\b", r"\bsuccessful\b", r"\braises?\b", r"\bincreases?\b", 
        r"\bbullish\b", r"\binnovations?\b", r"\binnovative\b", r"\brevenues?\b"
    ]
    
    neg_score = sum(len(re.findall(w, full_text)) for w in neg_words)
    pos_score = sum(len(re.findall(w, full_text)) for w in pos_words)
    
    total = neg_score + pos_score
    if total == 0:
        return "Neutral", 0.0
        
    sentiment_score = (pos_score - neg_score) / total
    
    if sentiment_score <= -0.15:
        sentiment = "Negative"
    elif sentiment_score >= 0.15:
        sentiment = "Positive"
    else:
        sentiment = "Neutral"
        
    return sentiment, round(sentiment_score, 2)

def get_sentiment_summary(articles):
    """
    Computes sentiment stats for a list of articles.
    Returns counts, overall average score (-1 to 1), and progress-bar representation (0-100).
    """
    if not articles:
        return {
            "Positive": 0, "Neutral": 0, "Negative": 0,
            "avg_score": 0.0, "percentage": 50.0, "label": "Neutral"
        }
        
    pos_cnt, neu_cnt, neg_cnt = 0, 0, 0
    scores = []
    
    for a in articles:
        # Check if DB already has it, if not compute on the fly
        sent = a.get("sentiment")
        score = a.get("sentiment_score")
        
        if sent is None or score is None:
            sent, score = analyze_article_sentiment(
                a.get("title"), a.get("description"), a.get("content")
            )
            
        scores.append(float(score))
        if sent == "Positive":
            pos_cnt += 1
        elif sent == "Negative":
            neg_cnt += 1
        else:
            neu_cnt += 1
            
    avg_score = sum(scores) / len(scores) if scores else 0.0
    
    # Scale from [-1, 1] to [0, 100]
    percentage = (avg_score + 1.0) * 50.0
    
    if avg_score <= -0.15:
        overall_label = "Negative"
    elif avg_score >= 0.15:
        overall_label = "Positive"
    else:
        overall_label = "Neutral"
        
    return {
        "Positive": pos_cnt,
        "Neutral": neu_cnt,
        "Negative": neg_cnt,
        "avg_score": round(avg_score, 2),
        "percentage": round(percentage, 1),
        "label": overall_label
    }

def calculate_health_score(risk_score, sentiment_score, layoff_events, days_since_last):
    """
    Formula:
    Health Score (0-100) = 100 - (
        0.4 * Risk Score 
        + 0.25 * (100 - Sentiment Percentage)
        + 0.2 * Layoff Frequency Penalty
        + 0.15 * Recency Penalty
    )
    """
    # 1. Risk Score Penalty (0 to 40 points)
    penalty_risk = 0.4 * risk_score
    
    # 2. News Sentiment Penalty (0 to 25 points)
    # Convert sentiment score (-1 to 1) to percentage (0 to 100). Higher percentage is better, so 100 - percentage is the penalty.
    sent_percentage = (sentiment_score + 1.0) * 50.0
    penalty_sentiment = 0.25 * (100 - sent_percentage)
    
    # 3. Layoff Frequency Penalty (0 to 20 points)
    # Scale from 0 (no layoffs) to 10+ layoff events (max penalty)
    penalty_events = min(layoff_events * 2.0, 20.0)
    
    # 4. Layoff Recency Penalty (0 to 15 points)
    # If recent layoff (within 30 days) -> max penalty. Decay to 0 after 1 year (365 days).
    if days_since_last is None:
        penalty_recency = 0.0
    else:
        if days_since_last <= 30:
            penalty_recency = 15.0
        elif days_since_last >= 365:
            penalty_recency = 0.0
        else:
            # Linear decay from 15 to 0 between 30 and 365 days
            penalty_recency = 15.0 * (1.0 - (days_since_last - 30) / 335.0)
            
    total_penalty = penalty_risk + penalty_sentiment + penalty_events + penalty_recency
    health_score = max(0, min(100, 100 - total_penalty))
    return round(health_score)
