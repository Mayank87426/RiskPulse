from pathlib import Path

import joblib

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.metrics import *

# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "ml" / "models"

OUTPUT_DIR = PROJECT_ROOT / "ml" / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

# -----------------------------

model = joblib.load(MODEL_DIR / "xgboost.pkl")

X_test = joblib.load(MODEL_DIR / "X_test.pkl")

y_test = joblib.load(MODEL_DIR / "y_test.pkl")

# -----------------------------

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:,1]

# -----------------------------

print("="*60)

print("XGBoost Evaluation")

print("="*60)

print(f"Accuracy : {accuracy_score(y_test,y_pred):.4f}")

print(f"Precision: {precision_score(y_test,y_pred):.4f}")

print(f"Recall   : {recall_score(y_test,y_pred):.4f}")

print(f"F1 Score : {f1_score(y_test,y_pred):.4f}")

print(f"ROC AUC  : {roc_auc_score(y_test,y_prob):.4f}")

print()

print(classification_report(y_test,y_pred))

cm = confusion_matrix(y_test,y_pred)

disp = ConfusionMatrixDisplay(cm)

disp.plot()

plt.tight_layout()

plt.savefig(

    OUTPUT_DIR/"xgboost_confusion_matrix.png",

    dpi=300

)

plt.close()

print()

print("Confusion Matrix Saved")