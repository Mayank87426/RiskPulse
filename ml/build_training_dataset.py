import sys
from pathlib import Path
import pandas as pd

# --------------------------------------------------
# Project path
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from scripts.database import get_connection


# --------------------------------------------------
# Load Data
# --------------------------------------------------

def load_data():
    conn = get_connection()

    query = """
    SELECT
        c.id AS company_id,
        c.name,
        c.industry,
        c.country,
        c.company_type,
        c.created_at,

        l.layoff_date,
        l.employees_laid_off,
        l.percentage_laid_off,
        l.company_size_before,
        l.company_size_after

    FROM companies c
    JOIN layoffs l
        ON c.id = l.company_id

    ORDER BY
        c.id,
        l.layoff_date;
    """

    df = pd.read_sql(query, conn)
    conn.close()
    return df


# --------------------------------------------------
# Feature Engineering
# --------------------------------------------------

def engineer_features(df):
    df["layoff_date"] = pd.to_datetime(df["layoff_date"])

    df = df.sort_values(["company_id", "layoff_date"])

    g = df.groupby("company_id")

    # ------------------------------------------
    # Previous layoff count
    # ------------------------------------------
    df["previous_layoff_events"] = g.cumcount()

    # ------------------------------------------
    # Historical total employees laid off
    # ------------------------------------------
    df["total_previous_laid_off"] = (
        g["employees_laid_off"]
        .cumsum()
        .shift(fill_value=0)
    )

    # ------------------------------------------
    # Historical largest layoff
    # ------------------------------------------
    df["largest_previous_layoff"] = (
        g["employees_laid_off"]
        .cummax()
        .shift(fill_value=0)
    )

    # ------------------------------------------
    # Historical average layoff
    # ------------------------------------------
    df["average_previous_layoff"] = (
        g["employees_laid_off"]
        .expanding()
        .mean()
        .shift()
        .reset_index(level=0, drop=True)
    )
    df["average_previous_layoff"] = df["average_previous_layoff"].fillna(0)

    # ------------------------------------------
    # Historical maximum percentage
    # ------------------------------------------
    df["maximum_previous_percentage"] = (
        g["percentage_laid_off"]
        .cummax()
        .shift(fill_value=0)
    )

    # ------------------------------------------
    # Historical average percentage
    # ------------------------------------------
    df["average_previous_percentage"] = (
        g["percentage_laid_off"]
        .expanding()
        .mean()
        .shift()
        .reset_index(level=0, drop=True)
    )
    df["average_previous_percentage"] = df["average_previous_percentage"].fillna(0)

    # ------------------------------------------
    # Previous layoff date
    # ------------------------------------------
    df["previous_layoff_date"] = g["layoff_date"].shift()

    # ------------------------------------------
    # Days since previous layoff
    # ------------------------------------------
    df["days_since_last_layoff"] = (
        df["layoff_date"] - df["previous_layoff_date"]
    ).dt.days
    df["days_since_last_layoff"] = df["days_since_last_layoff"].fillna(9999)

    # ------------------------------------------
    # Next layoff date
    # ------------------------------------------
    df["next_layoff_date"] = g["layoff_date"].shift(-1)

    # ------------------------------------------
    # Days until next layoff
    # ------------------------------------------
    df["days_until_next_layoff"] = (
        df["next_layoff_date"] - df["layoff_date"]
    ).dt.days

    # ------------------------------------------
    # Binary Target (1 = Another layoff within 180 days)
    # ------------------------------------------
    df["future_layoff"] = (
        (df["days_until_next_layoff"] <= 180) & 
        (df["days_until_next_layoff"].notna())
    ).astype(int)

    # ------------------------------------------
    # Calendar Features
    # ------------------------------------------
    df["year"] = df["layoff_date"].dt.year
    df["month"] = df["layoff_date"].dt.month
    df["quarter"] = df["layoff_date"].dt.quarter

    return df


# --------------------------------------------------
# Main Pipeline Execution
# --------------------------------------------------
if __name__ == "__main__":

    df = load_data()

    df = engineer_features(df)

    # Keep only useful columns
    dataset = df[[
        "company_id",
        "industry",
        "country",
        "company_type",

        "employees_laid_off",
        "percentage_laid_off",
        "company_size_before",
        "company_size_after",

        "previous_layoff_events",
        "total_previous_laid_off",
        "largest_previous_layoff",
        "average_previous_layoff",
        "maximum_previous_percentage",
        "average_previous_percentage",
        "days_since_last_layoff",

        "year",
        "month",
        "quarter",

        "future_layoff"
    ]]

    output_dir = PROJECT_ROOT / "ml" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "training_dataset.csv"

    dataset.to_csv(output_path, index=False)

    print("=" * 60)
    print("Training Dataset Created Successfully")
    print("=" * 60)

    print(f"Rows: {len(dataset)}")
    print(f"Columns: {len(dataset.columns)}")

    print("\nTarget Distribution:")
    print(dataset["future_layoff"].value_counts())

    print("\nSaved to:")
    print(output_path)