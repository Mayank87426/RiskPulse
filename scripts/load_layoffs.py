import pandas as pd
from database import get_connection
import os

# Read dataset
data_path = os.path.join(os.path.dirname(__file__), "..", "data", "tech_layoffs_till_2025.csv")
df = pd.read_csv(data_path)

conn = get_connection()
cur = conn.cursor()

loaded = 0
skipped = 0

# Fetch all company IDs first to build a lookup mapping in memory
cur.execute("SELECT id, name FROM companies")
company_map = {name: cid for cid, name in cur.fetchall()}

values = []
loaded = 0
skipped = 0

for _, row in df.iterrows():
    company_id = company_map.get(row["Company"])
    if company_id is None:
        skipped += 1
        continue

    # Handle missing values
    laid_off = None if pd.isna(row["Laid_Off"]) else int(row["Laid_Off"])
    percentage = None if pd.isna(row["Percentage"]) else float(row["Percentage"])
    size_before = None if pd.isna(row["Company_Size_before_Layoffs"]) else int(row["Company_Size_before_Layoffs"])
    size_after = None if pd.isna(row["Company_Size_after_layoffs"]) else int(row["Company_Size_after_layoffs"])

    values.append((
        company_id,
        row["Date_layoffs"],
        laid_off,
        percentage,
        size_before,
        size_after
    ))
    loaded += 1

print(f"Batch inserting {len(values)} layoffs...")
from psycopg2.extras import execute_values
execute_values(cur, """
    INSERT INTO layoffs(
        company_id,
        layoff_date,
        employees_laid_off,
        percentage_laid_off,
        company_size_before,
        company_size_after
    )
    VALUES %s
""", values)

conn.commit()
cur.close()
conn.close()

print(f"\nLoaded: {loaded}")
print(f"Skipped: {skipped}")
print("Layoffs Loaded Successfully!")