import os
import requests
from dotenv import load_dotenv
from database import get_connection

# -----------------------------
# Load Environment Variables
# -----------------------------
load_dotenv()

API_KEY = os.getenv("NEWS_API_KEY")

if not API_KEY:
    raise Exception("NEWS_API_KEY not found in .env")

# -----------------------------
# Connect to PostgreSQL
# -----------------------------
conn = get_connection()
cur = conn.cursor()

# -----------------------------
# Get Companies
# -----------------------------
cur.execute("""
SELECT id, name
FROM companies
ORDER BY id
LIMIT 10;
""")

companies = cur.fetchall()

print(f"Fetching news for {len(companies)} companies...\n")

# -----------------------------
# Fetch News
# -----------------------------
for company_id, company_name in companies:

    print(f"Searching: {company_name}")

    url = "https://newsapi.org/v2/everything"

    params = {
        "q": f'"{company_name}"',
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 5,
        "apiKey": API_KEY
    }

    try:

        response = requests.get(url, params=params)

        if response.status_code != 200:
            print(f"API Error ({response.status_code})")
            continue

        data = response.json()

        articles = data.get("articles", [])

        print(f"Found {len(articles)} articles")

        for article in articles:

            title = article.get("title")
            source = article.get("source", {}).get("name")
            author = article.get("author")
            published_at = article.get("publishedAt")
            url = article.get("url")
            description = article.get("description")
            content = article.get("content")

            cur.execute("""
            INSERT INTO news_articles(
                company_id,
                title,
                source,
                author,
                published_at,
                article_url,
                description,
                content
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (article_url) DO NOTHING;
            """,
            (
                company_id,
                title,
                source,
                author,
                published_at,
                url,
                description,
                content
            ))

        conn.commit()

    except Exception as e:
        print(e)

    print("-" * 60)

cur.close()
conn.close()

print("\nNews fetching completed!")