from pathlib import Path
import joblib

from sklearn.ensemble import RandomForestClassifier

# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "ml" / "models"

# --------------------------------------------------
# Load Processed Data
# --------------------------------------------------

X_train = joblib.load(MODEL_DIR / "X_train.pkl")
X_test = joblib.load(MODEL_DIR / "X_test.pkl")

y_train = joblib.load(MODEL_DIR / "y_train.pkl")
y_test = joblib.load(MODEL_DIR / "y_test.pkl")

print("=" * 60)
print("Training Random Forest...")
print("=" * 60)

# --------------------------------------------------
# Build Model
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

# --------------------------------------------------
# Train
# --------------------------------------------------

model.fit(X_train, y_train)

# --------------------------------------------------
# Save
# --------------------------------------------------

joblib.dump(
    model,
    MODEL_DIR / "random_forest.pkl"
)

print()

print("Model Saved Successfully")

print()

print("Location:")
print(MODEL_DIR / "random_forest.pkl")