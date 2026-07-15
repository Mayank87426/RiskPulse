import pandas as pd
from database import get_connection

df = pd.read_csv("../data/layoffs.csv")

conn = get_connection()
cur = conn.cursor()

for _, row in df.iterrows():

    cur.execute("""
        SELECT id
        FROM companies
        WHERE name = %s
    """, (row["company"],))

    company_id = cur.fetchone()[0]

    cur.execute("""
        INSERT INTO layoffs(
            company_id,
            layoff_date,
            employees_laid_off,
            percentage_laid_off
        )
        VALUES (%s,%s,%s,%s)
    """,
    (
        company_id,
        row["layoff_date"],
        row["employees_laid_off"],
        row["percentage_laid_off"]
    ))

conn.commit()

print("Layoffs loaded!")

cur.close()
conn.close()