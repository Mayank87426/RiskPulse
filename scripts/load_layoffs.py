import pandas as pd
from database import get_connection

# Read dataset
df = pd.read_csv("../data/tech_layoffs_till_2025.csv")

conn = get_connection()
cur = conn.cursor()

loaded = 0
skipped = 0

for _, row in df.iterrows():

    try:
        # Find company id
        cur.execute(
            "SELECT id FROM companies WHERE name = %s",
            (row["Company"],)
        )

        result = cur.fetchone()

        if result is None:
            skipped += 1
            continue

        company_id = result[0]

        # Handle missing values
        laid_off = None if pd.isna(row["Laid_Off"]) else int(row["Laid_Off"])

        percentage = None if pd.isna(row["Percentage"]) else float(row["Percentage"])

        size_before = (
            None
            if pd.isna(row["Company_Size_before_Layoffs"])
            else int(row["Company_Size_before_Layoffs"])
        )

        size_after = (
            None
            if pd.isna(row["Company_Size_after_layoffs"])
            else int(row["Company_Size_after_layoffs"])
        )

        # Insert layoff event
        cur.execute("""
            INSERT INTO layoffs(
                company_id,
                layoff_date,
                employees_laid_off,
                percentage_laid_off,
                company_size_before,
                company_size_after
            )
            VALUES (%s,%s,%s,%s,%s,%s)
        """,
        (
            company_id,
            row["Date_layoffs"],
            laid_off,
            percentage,
            size_before,
            size_after
        ))

        loaded += 1

    except Exception as e:
        print("\nERROR:")
        print(row["Company"])
        print(e)
        skipped += 1

conn.commit()

cur.close()
conn.close()

print(f"\nLoaded: {loaded}")
print(f"Skipped: {skipped}")
print("Layoffs Loaded Successfully!")