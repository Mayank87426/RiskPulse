import psycopg2
import os

def main():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("DATABASE_URL is not set!")
        return
        
    print("Connecting to database...")
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    
    print("Dropping existing tables if any...")
    cur.execute("DROP TABLE IF EXISTS news_articles CASCADE;")
    cur.execute("DROP TABLE IF EXISTS layoffs CASCADE;")
    cur.execute("DROP TABLE IF EXISTS workforce_risk CASCADE;")
    cur.execute("DROP TABLE IF EXISTS companies CASCADE;")
    
    print("Creating 'companies' table...")
    cur.execute("""
        CREATE TABLE companies (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255) NOT NULL UNIQUE,
            industry VARCHAR(255),
            country VARCHAR(255),
            company_type VARCHAR(255),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            location_hq VARCHAR(255),
            continent VARCHAR(255),
            stage VARCHAR(255),
            money_raised_mil NUMERIC,
            latitude NUMERIC,
            longitude NUMERIC
        );
    """)
    
    print("Creating 'workforce_risk' table...")
    cur.execute("""
        CREATE TABLE workforce_risk (
            company_id INTEGER PRIMARY KEY REFERENCES companies(id) ON DELETE CASCADE,
            workforce_risk_score INTEGER NOT NULL,
            risk_level VARCHAR(50) NOT NULL,
            layoff_events INTEGER,
            total_employees_laid_off BIGINT,
            latest_layoff DATE,
            max_percentage_laid_off NUMERIC,
            risk_reasons TEXT,
            generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            model_version VARCHAR(50)
        );
    """)
    
    print("Creating 'layoffs' table...")
    cur.execute("""
        CREATE TABLE layoffs (
            id SERIAL PRIMARY KEY,
            company_id INTEGER NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
            layoff_date DATE,
            employees_laid_off BIGINT,
            percentage_laid_off NUMERIC,
            company_size_before BIGINT,
            company_size_after BIGINT
        );
    """)
    
    print("Creating 'news_articles' table...")
    cur.execute("""
        CREATE TABLE news_articles (
            id SERIAL PRIMARY KEY,
            company_id INTEGER NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
            title TEXT NOT NULL,
            source VARCHAR(255),
            author VARCHAR(255),
            published_at TIMESTAMP,
            article_url TEXT UNIQUE,
            description TEXT,
            content TEXT,
            sentiment VARCHAR(50),
            sentiment_score NUMERIC,
            risk_phrases TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    
    conn.commit()
    cur.close()
    conn.close()
    print("All tables created successfully!")

if __name__ == "__main__":
    main()
