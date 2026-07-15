"""calibration.py -- Probability calibration via Platt scaling and Isotonic regression."""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss
from ml_config import MLConfig, get_logger

logger = get_logger("Calibration")

def expected_calibration_error(y_true, y_prob, n_bins=10) -> float:
    """Calculates the Expected Calibration Error (ECE) for binary classification."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n_samples = len(y_true)
    
    # Handle pandas Series index alignment
    y_true_arr = np.array(y_true)
    y_prob_arr = np.array(y_prob)
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Select items in bin
        in_bin = (y_prob_arr >= bin_lower) & (y_prob_arr < bin_upper)
        if i == n_bins - 1:  # Include 1.0 in the last bin
            in_bin = in_bin | (y_prob_arr == bin_upper)
            
        bin_size = np.sum(in_bin)
        if bin_size > 0:
            bin_acc = np.mean(y_true_arr[in_bin] == 1)
            bin_conf = np.mean(y_prob_arr[in_bin])
            ece += (bin_size / n_samples) * np.abs(bin_acc - bin_conf)
            
    return ece

def calibrate_model(model, config: MLConfig):
    """Calibrates predictions from a trained model using Platt scaling and Isotonic regression."""
    logger.info("=" * 60)
    logger.info("Probability Calibration")
    logger.info("=" * 60)
    
    # Load training & test datasets
    X_train = joblib.load(config.MODEL_DIR / "X_train.pkl")
    X_test = joblib.load(config.MODEL_DIR / "X_test.pkl")
    y_train = joblib.load(config.MODEL_DIR / "y_train.pkl")
    y_test = joblib.load(config.MODEL_DIR / "y_test.pkl")
    
    # Convert sparse matrices to dense arrays for compatibility with models like HistGradientBoosting
    if hasattr(X_train, "toarray"):
        X_train = X_train.toarray()
    if hasattr(X_test, "toarray"):
        X_test = X_test.toarray()
    
    # Base probabilities on test set
    y_prob_base = model.predict_proba(X_test)[:, 1]
    brier_base = brier_score_loss(y_test, y_prob_base)
    ece_base = expected_calibration_error(y_test, y_prob_base)
    logger.info("Base Model - Brier Score: %.4f, ECE: %.4f", brier_base, ece_base)
    
    # 1. Sigmoid Calibration (Platt Scaling) — use cv=5 (prefit removed in sklearn 1.8)
    logger.info("Fitting Platt Scaling (Sigmoid)...")
    try:
        cal_sigmoid = CalibratedClassifierCV(estimator=model, method="sigmoid", cv=5)
        cal_sigmoid.fit(X_train, y_train)
        y_prob_sigmoid = cal_sigmoid.predict_proba(X_test)[:, 1]
        brier_sigmoid = brier_score_loss(y_test, y_prob_sigmoid)
        ece_sigmoid = expected_calibration_error(y_test, y_prob_sigmoid)
        logger.info("Platt Scaling - Brier Score: %.4f, ECE: %.4f", brier_sigmoid, ece_sigmoid)
    except Exception as e:
        logger.warning("Platt scaling failed (%s), using base probabilities as fallback.", e)
        cal_sigmoid = model
        y_prob_sigmoid = y_prob_base
        brier_sigmoid = brier_base
        ece_sigmoid = ece_base
    
    # 2. Isotonic Calibration
    logger.info("Fitting Isotonic Regression...")
    try:
        cal_isotonic = CalibratedClassifierCV(estimator=model, method="isotonic", cv=5)
        cal_isotonic.fit(X_train, y_train)
        y_prob_isotonic = cal_isotonic.predict_proba(X_test)[:, 1]
        brier_isotonic = brier_score_loss(y_test, y_prob_isotonic)
        ece_isotonic = expected_calibration_error(y_test, y_prob_isotonic)
        logger.info("Isotonic Calibration - Brier: %.4f, ECE: %.4f", brier_isotonic, ece_isotonic)
    except Exception as e:
        logger.warning("Isotonic calibration failed (%s), using base probabilities as fallback.", e)
        cal_isotonic = model
        y_prob_isotonic = y_prob_base
        brier_isotonic = brier_base
        ece_isotonic = ece_base
    
    # 3. Generate Reliability Plot
    plt.figure(figsize=(8, 6))
    
    # Perfect calibration line
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")
    
    # Base model curve
    fraction_of_positives_base, mean_predicted_value_base = calibration_curve(y_test, y_prob_base, n_bins=10)
    plt.plot(mean_predicted_value_base, fraction_of_positives_base, "s-", label=f"Base (Brier: {brier_base:.3f})", color="gray")
    
    # Sigmoid curve
    f_pos_sig, m_pred_sig = calibration_curve(y_test, y_prob_sigmoid, n_bins=10)
    plt.plot(m_pred_sig, f_pos_sig, "s-", label=f"Platt Sigmoid (Brier: {brier_sigmoid:.3f})", color="blue")
    
    # Isotonic curve
    f_pos_iso, m_pred_iso = calibration_curve(y_test, y_prob_isotonic, n_bins=10)
    plt.plot(m_pred_iso, f_pos_iso, "s-", label=f"Isotonic (Brier: {brier_isotonic:.3f})", color="green")
    
    plt.xlabel("Mean Predicted Probability", fontsize=11, fontname="DejaVu Sans")
    plt.ylabel("Fraction of Positives", fontsize=11, fontname="DejaVu Sans")
    plt.title("Reliability Curve / Probability Calibration Curve", fontsize=13, fontweight="bold")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.5)
    
    plot_path = config.OUTPUT_DIR / "probability_calibration_curve.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logger.info("Reliability Curve saved to %s", plot_path)
    
    # Select the best model based on lowest Brier score
    scores = {
        "base": brier_base,
        "sigmoid": brier_sigmoid,
        "isotonic": brier_isotonic
    }
    
    best_method = min(scores, key=scores.get)
    logger.info("Selecting best calibration method: %s", best_method.upper())
    
    if best_method == "sigmoid":
        best_model = cal_sigmoid
    elif best_method == "isotonic":
        best_model = cal_isotonic
    else:
        best_model = model  # Keep uncalibrated if it scored lowest
        
    # Save the selected calibrated model
    model_path = config.MODEL_DIR / "calibrated_xgboost.pkl"
    joblib.dump(best_model, model_path)
    logger.info("Calibrated model saved to: %s", model_path)
    
    calibration_metrics = {
        "calibration_method": best_method,
        "brier_base": brier_base,
        "brier_sigmoid": brier_sigmoid,
        "brier_isotonic": brier_isotonic,
        "ece_base": ece_base,
        "ece_sigmoid": ece_sigmoid,
        "ece_isotonic": ece_isotonic
    }
    
    return best_model, calibration_metrics
