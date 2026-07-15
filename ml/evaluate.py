from pathlib import Path

import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "ml" / "models"

# --------------------------------------------------
# Load
# --------------------------------------------------

model = joblib.load(MODEL_DIR / "random_forest.pkl")

X_test = joblib.load(MODEL_DIR / "X_test.pkl")

y_test = joblib.load(MODEL_DIR / "y_test.pkl")

# --------------------------------------------------
# Predict
# --------------------------------------------------

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]

# --------------------------------------------------
# Metrics
# --------------------------------------------------

print("=" * 60)
print("Model Evaluation")
print("=" * 60)

print(f"Accuracy : {accuracy_score(y_test,y_pred):.4f}")

print(f"Precision: {precision_score(y_test,y_pred):.4f}")

print(f"Recall   : {recall_score(y_test,y_pred):.4f}")

print(f"F1 Score : {f1_score(y_test,y_pred):.4f}")

print(f"ROC AUC  : {roc_auc_score(y_test,y_prob):.4f}")

print()

print(classification_report(y_test,y_pred))

# --------------------------------------------------
# Confusion Matrix
# --------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(cm)

disp.plot()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

output_dir = PROJECT_ROOT / "ml" / "outputs"
output_dir.mkdir(exist_ok=True)

plt.tight_layout()
plt.savefig(output_dir / "confusion_matrix.png", dpi=300)
print("Confusion matrix saved to:", output_dir / "confusion_matrix.png")
plt.close()