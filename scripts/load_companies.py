import pandas as pd
from database import get_connection
import os

data_path = os.path.join(os.path.dirname(__file__), "..", "data", "tech_layoffs_till_2025.csv")
df = pd.read_csv(data_path)

companies = df.drop_duplicates(subset=["Company"])

from psycopg2.extras import execute_values

companies = companies.where(pd.notnull(companies), None)

conn = get_connection()
cur = conn.cursor()

# Prepare values for batch insertion
values = [
    (
        row["Company"],
        row["Industry"],
        row["Country"],
        row["Location_HQ"],
        row["Continent"],
        row["Stage"],
        row["Money_Raised_in__mil"],
        row["latitude"],
        row["longitude"]
    )
    for _, row in companies.iterrows()
]

print(f"Batch inserting {len(values)} companies...")
execute_values(cur, """
    INSERT INTO companies(
        name,
        industry,
        country,
        location_hq,
        continent,
        stage,
        money_raised_mil,
        latitude,
        longitude
    )
    VALUES %s
""", values)

conn.commit()
cur.close()
conn.close()

print("Companies Loaded Successfully!")