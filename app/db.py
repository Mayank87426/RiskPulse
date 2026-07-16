import psycopg2
from psycopg2.extras import RealDictCursor
import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_connection():
    """Establishes connection to the database. Does not use cache to avoid connection state leaks."""
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        return psycopg2.connect(db_url)
        
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        database=os.getenv("DB_NAME", "RIskPulse_DB"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "Mk123.jha@post"),
        port=os.getenv("DB_PORT", "5432")
    )

@st.cache_data(ttl=600)
def get_all_companies():
    """Fetch all unique company names sorted alphabetically."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT name
        FROM companies
        ORDER BY name;
    """)
    companies = [row[0] for row in cur.fetchall()]
    cur.close()
    conn.close()
    return companies

@st.cache_data(ttl=600)
def get_company_overview(company_name):
    """Fetch overview metrics and risk scores for a specific company."""
    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("""
        SELECT
            c.id,
            c.name,
            c.industry,
            c.country,
            c.stage,
            c.money_raised_mil,
            wr.workforce_risk_score,
            wr.risk_level,
            wr.layoff_events,
            wr.total_employees_laid_off,
            wr.latest_layoff,
            wr.max_percentage_laid_off,
            wr.risk_reasons,
            wr.generated_at,
            wr.model_version
        FROM companies c
        JOIN workforce_risk wr
        ON c.id = wr.company_id
        WHERE c.name = %s;
    """, (company_name,))
    result = cur.fetchone()
    cur.close()
    conn.close()
    return result

@st.cache_data(ttl=600)
def get_layoff_history(company_name):
    """Fetch layoff timeline events for a specific company."""
    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("""
        SELECT
            l.layoff_date,
            l.employees_laid_off,
            l.percentage_laid_off,
            l.company_size_before,
            l.company_size_after
        FROM layoffs l
        JOIN companies c
            ON l.company_id = c.id
        WHERE c.name = %s
        ORDER BY l.layoff_date;
    """, (company_name,))
    history = cur.fetchall()
    cur.close()
    conn.close()
    return history

@st.cache_data(ttl=600)
def get_company_news(company_id):
    """Fetch cached news articles for a company."""
    conn = get_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("""
        SELECT
            title,
            source,
            author,
            published_at,
            article_url,
            description,
            content,
            sentiment,
            sentiment_score
        FROM news_articles
        WHERE company_id = %s
        ORDER BY published_at DESC;
    """, (company_id,))
    news = cur.fetchall()
    cur.close()
    conn.close()
    return news

@st.cache_data(ttl=3600)
def get_filter_options():
    """Fetch unique values for Industry, Country, and Risk Level filters."""
    conn = get_connection()
    cur = conn.cursor()
    
    # Get Industries
    cur.execute("SELECT DISTINCT industry FROM companies WHERE industry IS NOT NULL AND industry != '' ORDER BY industry")
    industries = ["All"] + [r[0] for r in cur.fetchall()]
    
    # Get Countries
    cur.execute("SELECT DISTINCT country FROM companies WHERE country IS NOT NULL AND country != '' ORDER BY country")
    countries = ["All"] + [r[0] for r in cur.fetchall()]
    
    # Get Date bounds
    cur.execute("SELECT MIN(layoff_date), MAX(layoff_date) FROM layoffs")
    min_date, max_date = cur.fetchone()
    if min_date is None:
        min_date = "2020-01-01"
    if max_date is None:
        max_date = "2026-12-31"
        
    cur.close()
    conn.close()
    
    return {
        "industries": industries,
        "countries": countries,
        "risk_levels": ["All", "Very Low", "Low", "Medium", "High", "Critical"],
        "min_date": min_date,
        "max_date": max_date
    }

@st.cache_data(ttl=60)
def get_filtered_companies(industry="All", country="All", risk_level="All", start_date=None, end_date=None):
    """Filter list of company names matching sidebar filters dynamically."""
    conn = get_connection()
    cur = conn.cursor()
    
    query = """
        SELECT DISTINCT c.name
        FROM companies c
        JOIN workforce_risk wr ON c.id = wr.company_id
        LEFT JOIN layoffs l ON c.id = l.company_id
        WHERE 1=1
    """
    params = []
    
    if industry != "All":
        query += " AND c.industry = %s"
        params.append(industry)
        
    if country != "All":
        query += " AND c.country = %s"
        params.append(country)
        
    if risk_level != "All":
        query += " AND wr.risk_level = %s"
        params.append(risk_level)
        
    if start_date and end_date:
        query += " AND (l.layoff_date BETWEEN %s AND %s OR l.layoff_date IS NULL)"
        params.append(start_date)
        params.append(end_date)
        
    query += " ORDER BY c.name"
    
    cur.execute(query, tuple(params))
    companies = [r[0] for r in cur.fetchall()]
    cur.close()
    conn.close()
    return companies

@st.cache_data(ttl=300)
def get_db_stats():
    """Fetch database health statistics for status display."""
    stats = {}
    try:
        conn = get_connection()
        cur = conn.cursor()
        
        # Company Count
        cur.execute("SELECT COUNT(*) FROM companies")
        stats["num_companies"] = cur.fetchone()[0]
        
        # Layoff Count
        cur.execute("SELECT COUNT(*) FROM layoffs")
        stats["num_layoffs"] = cur.fetchone()[0]
        
        # Last Updated from workforce_risk
        cur.execute("SELECT MAX(generated_at) FROM workforce_risk")
        max_gen = cur.fetchone()[0]
        stats["last_updated"] = max_gen.strftime("%Y-%m-%d %H:%M:%S") if max_gen else "N/A"
        
        # News API Key Verification
        api_key = os.getenv("NEWS_API_KEY")
        stats["news_api_status"] = "Active" if (api_key and len(api_key) > 10) else "Missing"
        stats["db_status"] = "Connected"
        
        cur.close()
        conn.close()
    except Exception as e:
        stats["db_status"] = "Disconnected"
        stats["num_companies"] = 0
        stats["num_layoffs"] = 0
        stats["last_updated"] = "N/A"
        stats["news_api_status"] = "Unknown"
        stats["error"] = str(e)
        
    return stats
