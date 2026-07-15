from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "ml" / "data" / "enhanced_training_dataset.csv"

MODEL_DIR = PROJECT_ROOT / "ml" / "models"
MODEL_DIR.mkdir(exist_ok=True)

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("Dataset Loaded")
print("=" * 60)
print(df.shape)

from ml_config import MLConfig
config = MLConfig()

# --------------------------------------------------
# Target
# --------------------------------------------------

y = df[config.TARGET_COL]

cols_to_drop = [c for c in [config.TARGET_COL] + config.IGNORE_COLS if c in df.columns]
X = df.drop(columns=cols_to_drop)

# --------------------------------------------------
# Feature Types
# --------------------------------------------------

categorical_features = [c for c in config.CATEGORICAL_COLS if c in X.columns]

numeric_features = [
    col
    for col in X.columns
    if col not in categorical_features
]

# --------------------------------------------------
# Numeric Pipeline
# --------------------------------------------------

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)

# --------------------------------------------------
# Categorical Pipeline
# --------------------------------------------------

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)

# --------------------------------------------------
# Full Preprocessor
# --------------------------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_pipeline,
            numeric_features
        ),
        (
            "cat",
            categorical_pipeline,
            categorical_features
        )
    ]
)

# --------------------------------------------------
# Train Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print()

print("Train:", X_train.shape)
print("Test :", X_test.shape)

# --------------------------------------------------
# Fit Transformer
# --------------------------------------------------

X_train = preprocessor.fit_transform(X_train)

X_test = preprocessor.transform(X_test)

# --------------------------------------------------
# Save Everything
# --------------------------------------------------

joblib.dump(
    preprocessor,
    MODEL_DIR / "preprocessor.pkl"
)

joblib.dump(
    X_train,
    MODEL_DIR / "X_train.pkl"
)

joblib.dump(
    X_test,
    MODEL_DIR / "X_test.pkl"
)

joblib.dump(
    y_train,
    MODEL_DIR / "y_train.pkl"
)

joblib.dump(
    y_test,
    MODEL_DIR / "y_test.pkl"
)

print()

print("=" * 60)
print("Preprocessing Complete")
print("=" * 60)

print("Training Samples :", len(y_train))
print("Testing Samples  :", len(y_test))

print()

print("Files Saved")

print("[OK] preprocessor.pkl")
print("[OK] X_train.pkl")
print("[OK] X_test.pkl")
print("[OK] y_train.pkl")
print("[OK] y_test.pkl")