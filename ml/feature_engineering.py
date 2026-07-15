from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT = PROJECT_ROOT / "ml" / "data" / "training_dataset.csv"

OUTPUT = PROJECT_ROOT / "ml" / "data" / "enhanced_training_dataset.csv"

# --------------------------------------------------

df = pd.read_csv(INPUT)

print("=" * 60)
print("Feature Engineering")
print("=" * 60)

# --------------------------------------------------
# Fill Missing Values
# --------------------------------------------------

numeric = [
    "employees_laid_off",
    "percentage_laid_off",
    "company_size_before",
    "company_size_after",
    "largest_previous_layoff",
    "average_previous_layoff",
    "maximum_previous_percentage",
    "average_previous_percentage",
]

for col in numeric:
    df[col] = df[col].fillna(0)

# --------------------------------------------------
# Workforce Reduction Ratio
# --------------------------------------------------

df["workforce_reduction_ratio"] = np.where(
    df["company_size_before"] > 0,
    df["employees_laid_off"] /
    df["company_size_before"],
    0
)

# --------------------------------------------------
# Remaining Workforce
# --------------------------------------------------

df["remaining_workforce_ratio"] = np.where(
    df["company_size_before"] > 0,
    df["company_size_after"] /
    df["company_size_before"],
    1
)

# --------------------------------------------------
# Workforce Shrinkage
# --------------------------------------------------

df["workforce_shrinkage"] = (
    df["company_size_before"]
    -
    df["company_size_after"]
)

# --------------------------------------------------
# Previous Layoff Size
# --------------------------------------------------

df["previous_layoff_size"] = (

    df.groupby("company_id")

    ["employees_laid_off"]

    .shift()

)

df["previous_layoff_size"] = (
    df["previous_layoff_size"]
    .fillna(0)
)

# --------------------------------------------------
# Layoff Growth
# --------------------------------------------------

df["layoff_growth"] = np.where(
    df["previous_layoff_size"] > 0,
    df["employees_laid_off"] /
    df["previous_layoff_size"],
    1
)

# --------------------------------------------------
# Rolling Average (3)
# --------------------------------------------------

df["rolling_avg_layoff"] = (

    df.groupby("company_id")

    ["employees_laid_off"]

    .transform(

        lambda x:

        x.shift()

        .rolling(3, min_periods=1)

        .mean()

    )

)

df["rolling_avg_layoff"] = (
    df["rolling_avg_layoff"]
    .fillna(0)
)

# --------------------------------------------------
# Rolling Percentage
# --------------------------------------------------

df["rolling_avg_percentage"] = (

    df.groupby("company_id")

    ["percentage_laid_off"]

    .transform(

        lambda x:

        x.shift()

        .rolling(3, min_periods=1)

        .mean()

    )

)

df["rolling_avg_percentage"] = (
    df["rolling_avg_percentage"]
    .fillna(0)
)

# --------------------------------------------------
# Layoff Velocity
# --------------------------------------------------

df["layoff_velocity"] = np.where(
    df["days_since_last_layoff"] > 0,
    df["employees_laid_off"] /
    df["days_since_last_layoff"],
    0
)

# --------------------------------------------------
# Large Layoff
# --------------------------------------------------

df["large_layoff"] = (
    df["employees_laid_off"] >= 500
).astype(int)

# --------------------------------------------------
# Major Workforce Cut
# --------------------------------------------------

df["major_workforce_cut"] = (
    df["percentage_laid_off"] >= 25
).astype(int)

# --------------------------------------------------
# Cumulative Workforce Lost
# --------------------------------------------------

df["cumulative_reduction"] = (

    df.groupby("company_id")

    ["workforce_reduction_ratio"]

    .cumsum()

)

# --------------------------------------------------
# Previous Company Size
# --------------------------------------------------

df["previous_company_size"] = (

    df.groupby("company_id")

    ["company_size_before"]

    .shift()

)

df["previous_company_size"] = (
    df["previous_company_size"]
    .fillna(df["company_size_before"])
)

# --------------------------------------------------
# Company Size Change
# --------------------------------------------------

df["company_size_change"] = (

    df["company_size_before"]

    -

    df["previous_company_size"]

)

# --------------------------------------------------
# Layoffs Per Year
# --------------------------------------------------

df["company_age_proxy"] = (
    df["previous_layoff_events"] + 1
)

df["layoffs_per_year"] = (
    df["previous_layoff_events"]
    /
    df["company_age_proxy"]
)

# --------------------------------------------------

df.to_csv(
    OUTPUT,
    index=False
)

print()

print("Enhanced Dataset Saved")

print()

print("Rows:", len(df))

print("Columns:", len(df.columns))

print()

print(df.head())
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT = PROJECT_ROOT / "ml" / "data" / "training_dataset.csv"

OUTPUT = PROJECT_ROOT / "ml" / "data" / "enhanced_training_dataset.csv"

# --------------------------------------------------

df = pd.read_csv(INPUT)

print("=" * 60)
print("Feature Engineering")
print("=" * 60)

# --------------------------------------------------
# Fill Missing Values
# --------------------------------------------------

numeric = [
    "employees_laid_off",
    "percentage_laid_off",
    "company_size_before",
    "company_size_after",
    "largest_previous_layoff",
    "average_previous_layoff",
    "maximum_previous_percentage",
    "average_previous_percentage",
]

for col in numeric:
    df[col] = df[col].fillna(0)

# --------------------------------------------------
# Workforce Reduction Ratio
# --------------------------------------------------

df["workforce_reduction_ratio"] = np.where(
    df["company_size_before"] > 0,
    df["employees_laid_off"] /
    df["company_size_before"],
    0
)

# --------------------------------------------------
# Remaining Workforce
# --------------------------------------------------

df["remaining_workforce_ratio"] = np.where(
    df["company_size_before"] > 0,
    df["company_size_after"] /
    df["company_size_before"],
    1
)

# --------------------------------------------------
# Workforce Shrinkage
# --------------------------------------------------

df["workforce_shrinkage"] = (
    df["company_size_before"]
    -
    df["company_size_after"]
)

# --------------------------------------------------
# Previous Layoff Size
# --------------------------------------------------

df["previous_layoff_size"] = (

    df.groupby("company_id")

    ["employees_laid_off"]

    .shift()

)

df["previous_layoff_size"] = (
    df["previous_layoff_size"]
    .fillna(0)
)

# --------------------------------------------------
# Layoff Growth
# --------------------------------------------------

df["layoff_growth"] = np.where(
    df["previous_layoff_size"] > 0,
    df["employees_laid_off"] /
    df["previous_layoff_size"],
    1
)

# --------------------------------------------------
# Rolling Average (3)
# --------------------------------------------------

df["rolling_avg_layoff"] = (

    df.groupby("company_id")

    ["employees_laid_off"]

    .transform(

        lambda x:

        x.shift()

        .rolling(3, min_periods=1)

        .mean()

    )

)

df["rolling_avg_layoff"] = (
    df["rolling_avg_layoff"]
    .fillna(0)
)

# --------------------------------------------------
# Rolling Percentage
# --------------------------------------------------

df["rolling_avg_percentage"] = (

    df.groupby("company_id")

    ["percentage_laid_off"]

    .transform(

        lambda x:

        x.shift()

        .rolling(3, min_periods=1)

        .mean()

    )

)

df["rolling_avg_percentage"] = (
    df["rolling_avg_percentage"]
    .fillna(0)
)

# --------------------------------------------------
# Layoff Velocity
# --------------------------------------------------

df["layoff_velocity"] = np.where(
    df["days_since_last_layoff"] > 0,
    df["employees_laid_off"] /
    df["days_since_last_layoff"],
    0
)

# --------------------------------------------------
# Large Layoff
# --------------------------------------------------

df["large_layoff"] = (
    df["employees_laid_off"] >= 500
).astype(int)

# --------------------------------------------------
# Major Workforce Cut
# --------------------------------------------------

df["major_workforce_cut"] = (
    df["percentage_laid_off"] >= 25
).astype(int)

# --------------------------------------------------
# Cumulative Workforce Lost
# --------------------------------------------------

df["cumulative_reduction"] = (

    df.groupby("company_id")

    ["workforce_reduction_ratio"]

    .cumsum()

)

# --------------------------------------------------
# Previous Company Size
# --------------------------------------------------

df["previous_company_size"] = (

    df.groupby("company_id")

    ["company_size_before"]

    .shift()

)

df["previous_company_size"] = (
    df["previous_company_size"]
    .fillna(df["company_size_before"])
)

# --------------------------------------------------
# Company Size Change
# --------------------------------------------------

df["company_size_change"] = (

    df["company_size_before"]

    -

    df["previous_company_size"]

)

# --------------------------------------------------
# Layoffs Per Year
# --------------------------------------------------

df["company_age_proxy"] = (
    df["previous_layoff_events"] + 1
)

df["layoffs_per_year"] = (
    df["previous_layoff_events"]
    /
    df["company_age_proxy"]
)

# --------------------------------------------------

df.to_csv(
    OUTPUT,
    index=False
)

print()

print("Enhanced Dataset Saved")

print()

print("Rows:", len(df))

print("Columns:", len(df.columns))

print()

print(df.head())