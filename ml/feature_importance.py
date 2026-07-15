from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "ml" / "models"

DATA_PATH = PROJECT_ROOT / "ml" / "data" / "training_dataset.csv"

# --------------------------------------------------

model = joblib.load(MODEL_DIR / "random_forest.pkl")

preprocessor = joblib.load(MODEL_DIR / "preprocessor.pkl")

df = pd.read_csv(DATA_PATH)

feature_names = preprocessor.get_feature_names_out()

importance = pd.DataFrame({
    "Feature": feature_names,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print()

print(importance.head(20))

plt.figure(figsize=(10,8))

plt.barh(
    importance["Feature"][:20],
    importance["Importance"][:20]
)

plt.gca().invert_yaxis()

plt.tight_layout()

plt.show()