import psycopg2

def get_connection():
    return psycopg2.connect(
        host="localhost",
        database="RIskPulse_DB",
        user="postgres",
        password="Mk123.jha@post",
        port="5432"
    )