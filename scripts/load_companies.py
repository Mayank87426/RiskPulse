import pandas as pd
from database import get_connection

df = pd.read_csv("../data/tech_layoffs_till_2025.csv")

companies = df.drop_duplicates(subset=["Company"])

companies = companies.where(pd.notnull(companies), None)

conn = get_connection()
cur = conn.cursor()

for _, row in companies.iterrows():

    cur.execute("""
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
    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """,
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
    ))

conn.commit()

cur.close()
conn.close()

print("Companies Loaded")