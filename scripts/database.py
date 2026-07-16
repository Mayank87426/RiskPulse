import psycopg2
import os

def get_connection():
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        return psycopg2.connect(db_url)
        
    return psycopg2.connect(
        host="localhost",
        database="RIskPulse_DB",
        user="postgres",
        password="Mk123.jha@post",
        port="5432"
    )