import pandas as pd

df = pd.read_csv("../data/tech_layoffs_till_2025.csv")

df["Date_layoffs"] = pd.to_datetime(df["Date_layoffs"])

df = df.where(pd.notnull(df), None)

print(df.info())