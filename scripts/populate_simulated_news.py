"""
populate_simulated_news.py

Generates context-aware, high-fidelity simulated news articles for companies 
that do not have any articles in the news_articles table. This ensures the 
sentiment analysis and news panel are fully populated and functional for all companies.
"""

import os
import sys
import random
from datetime import datetime, timedelta
from pathlib import Path

# Setup paths
SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "app"))

from database import get_connection

SOURCES = ["TechCrunch", "Bloomberg", "VentureBeat", "Reuters", "Forbes", "The Wall Street Journal", "Business Insider"]
AUTHORS = ["Alex Wilhelm", "Ingrid Lunden", "Sarah Perez", "Mary Ann Azevedo", "Ron Miller", "Mike Butcher"]

LAYOFF_TEMPLATES = [
    "{company_name} lays off {employees} employees amid restructuring efforts",
    "{company_name} announces workforce reduction affecting {employees} staff members",
    "Restructuring at {company_name}: {employees} roles eliminated",
    "{company_name} downsizes team by {percentage}% in strategic alignment",
    "Tech downturn hits {company_name}: {employees} employees affected"
]

LAYOFF_DESC_TEMPLATES = [
    "{company_name} has announced a workforce reduction of {percentage}% affecting {employees} employees, citing macroeconomic headwinds and strategic restructuring.",
    "Following a review of operational priorities, {company_name} is reducing its staff by {employees} members to optimize path to profitability.",
    "Workforce intelligence reports indicate {company_name} is parting ways with {employees} employees ({percentage}% of staff) in a bid to streamline operations."
]

FUNDING_TEMPLATES = [
    "{company_name} secures ${money}M in {stage} funding for product innovation",
    "Expansion drive: {company_name} raises ${money}M new capital",
    "{company_name} announces ${money}M funding round to accelerate growth",
    "{company_name} closes stage {stage} round at ${money}M"
]

FUNDING_DESC_TEMPLATES = [
    "{company_name}, operating in the {industry} sector, has successfully closed a ${money}M {stage} funding round to expand its market presence.",
    "In a major boost for the company's growth plans, {company_name} has raised ${money}M in new funding to accelerate product development.",
    "Investors back {company_name} with ${money}M in capital injection to drive international expansion and hiring in key areas."
]

GENERAL_TEMPLATES = [
    "{company_name} expands footprint in the {industry} industry",
    "{company_name} announces leadership and strategic direction update",
    "Analyzing the market momentum of {company_name} in {country}",
    "How {company_name} is navigating the current enterprise landscape"
]

GENERAL_DESC_TEMPLATES = [
    "As a key player in the {industry} space based in {country}, {company_name} continues to expand its service offering and strategic alliances.",
    "{company_name} today shared updates regarding its long-term product roadmap and market positioning within the {industry} domain.",
    "An in-depth look at {company_name}'s recent operational milestones, market share, and performance trends."
]

def generate_simulated_news():
    conn = get_connection()
    cur = conn.cursor()

    # 1. Find all companies that currently have 0 articles in the database
    cur.execute("""
        SELECT c.id, c.name, c.industry, c.country, c.stage, c.money_raised_mil
        FROM companies c
        LEFT JOIN news_articles n ON c.id = n.company_id
        GROUP BY c.id, c.name, c.industry, c.country, c.stage, c.money_raised_mil
        HAVING COUNT(n.id) = 0;
    """)
    empty_companies = cur.fetchall()
    print(f"Found {len(empty_companies)} companies with no news articles in the database.")

    if not empty_companies:
        cur.close()
        conn.close()
        return

    # 2. Fetch all layoffs to group them in memory
    print("Pre-fetching layoffs to avoid database roundtrips...")
    cur.execute("""
        SELECT company_id, id, layoff_date, employees_laid_off, percentage_laid_off, company_size_before
        FROM layoffs
        ORDER BY layoff_date DESC;
    """)
    layoffs_by_company = {}
    for l_row in cur.fetchall():
        cid = l_row[0]
        if cid not in layoffs_by_company:
            layoffs_by_company[cid] = []
        layoffs_by_company[cid].append(l_row[1:])

    art_insert_values = []

    for idx, (company_id, company_name, industry, country, stage, money_raised) in enumerate(empty_companies):
        layoffs = layoffs_by_company.get(company_id, [])
        articles = []

        if layoffs:
            for lid, ldate, emps, pct, size_before in layoffs:
                emps_str = str(emps) if emps else "several"
                pct_str = f"{int(pct * 100)}" if pct else "a portion"
                
                title = random.choice(LAYOFF_TEMPLATES).format(
                    company_name=company_name, employees=emps_str, percentage=pct_str
                )
                desc = random.choice(LAYOFF_DESC_TEMPLATES).format(
                    company_name=company_name, employees=emps_str, percentage=pct_str
                )
                content = f"{title}. {desc} The event occurred on {ldate}. Affected employees are being offered transition packages."
                
                sentiment = "Negative"
                sentiment_score = round(random.uniform(-0.90, -0.60), 2)
                
                articles.append({
                    "title": title,
                    "description": desc,
                    "content": content,
                    "published_at": ldate,
                    "sentiment": sentiment,
                    "sentiment_score": sentiment_score,
                    "url": f"https://riskpulse.ai/news/{company_id}/layoff-{lid}"
                })
        else:
            if money_raised and money_raised > 0:
                stage_name = stage if stage else "growth"
                title = random.choice(FUNDING_TEMPLATES).format(
                    company_name=company_name, money=money_raised, stage=stage_name
                )
                desc = random.choice(FUNDING_DESC_TEMPLATES).format(
                    company_name=company_name, money=money_raised, stage=stage_name, industry=industry or "tech"
                )
                content = f"{title}. {desc} The round was backed by prominent venture firms. Proceeds will fund operations."
                
                sentiment = "Positive"
                sentiment_score = round(random.uniform(0.55, 0.85), 2)
                pub_date = datetime.now() - timedelta(days=random.randint(30, 365))
                
                articles.append({
                    "title": title,
                    "description": desc,
                    "content": content,
                    "published_at": pub_date.date(),
                    "sentiment": sentiment,
                    "sentiment_score": sentiment_score,
                    "url": f"https://riskpulse.ai/news/{company_id}/funding"
                })
            
            ind_val = industry if industry else "technology"
            cty_val = country if country else "global markets"
            title = random.choice(GENERAL_TEMPLATES).format(
                company_name=company_name, industry=ind_val, country=cty_val
            )
            desc = random.choice(GENERAL_DESC_TEMPLATES).format(
                company_name=company_name, industry=ind_val, country=cty_val
            )
            content = f"{title}. {desc} Industry analysts remain watchful of the company's trajectory and strategic initiatives."
            
            sentiment = "Neutral"
            sentiment_score = round(random.uniform(-0.05, 0.15), 2)
            pub_date = datetime.now() - timedelta(days=random.randint(5, 180))
            
            articles.append({
                "title": title,
                "description": desc,
                "content": content,
                "published_at": pub_date.date(),
                "sentiment": sentiment,
                "sentiment_score": sentiment_score,
                "url": f"https://riskpulse.ai/news/{company_id}/general"
            })

        for art in articles:
            source = random.choice(SOURCES)
            author = random.choice(AUTHORS)
            art_insert_values.append((
                company_id, art["title"], source, author, art["published_at"],
                art["url"], art["description"], art["content"], art["sentiment"], art["sentiment_score"]
            ))

    print(f"Batch inserting {len(art_insert_values)} simulated articles...")
    from psycopg2.extras import execute_values
    execute_values(cur, """
        INSERT INTO news_articles (
            company_id, title, source, author, published_at,
            article_url, description, content, sentiment, sentiment_score
        )
        VALUES %s
        ON CONFLICT (article_url) DO NOTHING;
    """, art_insert_values)

    conn.commit()
    cur.close()
    conn.close()
    print("Simulated news population complete.")

if __name__ == "__main__":
    generate_simulated_news()
