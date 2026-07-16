"""
fetch_news.py — Fetches news articles for all companies via NewsAPI and
stores them in the news_articles table with computed sentiment scores.
"""

import os
import sys
import time
import requests
from pathlib import Path
from dotenv import load_dotenv

# Allow importing from app/ (for sentiment helper) and scripts/
SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "app"))

from database import get_connection
from utils.helpers import analyze_article_sentiment

# ── Configuration ──────────────────────────────────────────────────────────
load_dotenv(PROJECT_ROOT / ".env")
API_KEY = os.getenv("NEWS_API_KEY")

if not API_KEY:
    raise EnvironmentError("NEWS_API_KEY not found in .env file. Please add it and retry.")

NEWS_API_URL = "https://newsapi.org/v2/everything"
ARTICLES_PER_COMPANY = 5   # NewsAPI free tier: max 100 req/day
REQUEST_DELAY_SEC = 0.5     # Polite delay between API calls


def fetch_articles_for_company(company_name: str, api_key: str) -> list:
    """Calls the NewsAPI and returns a list of raw article dicts for a company."""
    params = {
        "q": f'"{company_name}" layoff OR workforce OR employees',
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": ARTICLES_PER_COMPANY,
        "apiKey": api_key,
    }
    try:
        resp = requests.get(NEWS_API_URL, params=params, timeout=10)
        if resp.status_code != 200:
            print(f"  [WARN] API returned {resp.status_code} for '{company_name}'")
            return []
        data = resp.json()
        return data.get("articles", [])
    except Exception as e:
        print(f"  [ERROR] Request failed for '{company_name}': {e}")
        return []


def upsert_article(cur, company_id: int, article: dict) -> bool:
    """Inserts a single article with sentiment scores. Returns True if inserted."""
    title       = article.get("title") or ""
    source      = (article.get("source") or {}).get("name") or ""
    author      = article.get("author") or ""
    published   = article.get("publishedAt") or None
    url         = article.get("url") or ""
    description = article.get("description") or ""
    content     = article.get("content") or ""

    if not url or not title:
        return False

    # Compute sentiment using the keyword-based analyzer already in helpers.py
    sentiment_label, sentiment_score = analyze_article_sentiment(title, description, content)

    cur.execute(
        """
        INSERT INTO news_articles (
            company_id, title, source, author, published_at,
            article_url, description, content, sentiment, sentiment_score
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (article_url) DO UPDATE
            SET sentiment       = EXCLUDED.sentiment,
                sentiment_score = EXCLUDED.sentiment_score;
        """,
        (
            company_id, title, source, author, published,
            url, description, content, sentiment_label, round(float(sentiment_score), 4),
        ),
    )
    return True


def main() -> None:
    conn = get_connection()
    cur  = conn.cursor()

    # Fetch ALL companies
    cur.execute("SELECT id, name FROM companies ORDER BY id;")
    companies = cur.fetchall()
    print(f"Fetching news for {len(companies)} companies...\n")

    total_inserted = 0

    for company_id, company_name in companies:
        print(f"  Searching: {company_name}")
        articles = fetch_articles_for_company(company_name, API_KEY)
        print(f"    Found {len(articles)} articles")

        inserted = 0
        for article in articles:
            if upsert_article(cur, company_id, article):
                inserted += 1

        conn.commit()
        total_inserted += inserted
        print(f"    Saved {inserted} articles (with sentiment scores)")
        print(f"    {'-' * 56}")

        # Polite delay to stay within rate limits
        time.sleep(REQUEST_DELAY_SEC)

    cur.close()
    conn.close()
    print(f"\nNews fetching complete. Total articles saved/updated: {total_inserted}")


if __name__ == "__main__":
    main()