from pathlib import Path
import joblib

from xgboost import XGBClassifier

# -----------------------------
# Paths
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "ml" / "models"

# -----------------------------
# Load Data
# -----------------------------

X_train = joblib.load(MODEL_DIR / "X_train.pkl")
X_test = joblib.load(MODEL_DIR / "X_test.pkl")

y_train = joblib.load(MODEL_DIR / "y_train.pkl")
y_test = joblib.load(MODEL_DIR / "y_test.pkl")

# -----------------------------
# Handle Imbalanced Dataset
# -----------------------------

negative = (y_train == 0).sum()
positive = (y_train == 1).sum()

scale = negative / positive

print(f"Negative: {negative}")
print(f"Positive: {positive}")
print(f"Scale Weight: {scale:.2f}")

# -----------------------------
# Build Model
# -----------------------------

model = XGBClassifier(

    n_estimators=400,

    learning_rate=0.05,

    max_depth=5,

    min_child_weight=3,

    subsample=0.8,

    colsample_bytree=0.8,

    gamma=0.1,

    objective="binary:logistic",

    eval_metric="logloss",

    scale_pos_weight=scale,

    random_state=42,

    n_jobs=-1
)

print("\nTraining XGBoost...\n")

model.fit(

    X_train,

    y_train,

    eval_set=[(X_test, y_test)],

    verbose=False
)

# -----------------------------
# Save
# -----------------------------

joblib.dump(

    model,

    MODEL_DIR / "xgboost.pkl"
)

print("Model Saved Successfully")