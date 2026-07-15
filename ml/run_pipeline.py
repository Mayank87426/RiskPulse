"""
run_pipeline.py — End-to-end orchestrator for the RiskPulse ML pipeline.

Executes all training, optimisation, calibration, and explainability steps
in the correct order with smart caching so expensive stages (HPO, benchmark)
are only re-run when their output artefacts are missing.
"""

import time
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import pipeline modules
from ml_config import MLConfig, get_logger
from tracker import ExperimentTracker
from hpo import run_hpo
from benchmark import run_benchmarking
from feature_selector import run_feature_selection
from validation import run_time_aware_validation, run_threshold_optimization
from calibration import calibrate_model
from explainability import generate_global_shap_explanations

logger = get_logger("Pipeline")


def execute_pipeline() -> None:
    """Runs the full workforce risk ML pipeline end-to-end."""
    logger.info("=" * 70)
    logger.info("RISKPULSE Predictive Intelligence Optimization Pipeline")
    logger.info("=" * 70)

    start_time = time.time()
    config = MLConfig()
    tracker = ExperimentTracker(config)

    # 1. Dataset Verification
    enhanced_data_path = config.DATA_DIR / "enhanced_training_dataset.csv"
    if not enhanced_data_path.exists():
        logger.error("Enhanced training dataset not found. Run feature engineering first.")
        return

    # 2. Verify preprocessed matrices exist
    required_files = ["X_train.pkl", "X_test.pkl", "y_train.pkl", "y_test.pkl", "preprocessor.pkl"]
    missing = [f for f in required_files if not (config.MODEL_DIR / f).exists()]

    if missing:
        logger.error("Missing preprocessed matrices: %s. Run preprocess.py first.", missing)
        return

    # Step 1: Hyperparameter Optimization (Optuna HPO)
    hpo_meta_path = config.MODEL_DIR / "xgboost_hpo_metadata.json"
    if hpo_meta_path.exists():
        logger.info("[STEP 1] Loading cached HPO metadata...")
        with open(hpo_meta_path, "r") as f:
            hpo_meta = json.load(f)
            best_xgb_params: Dict[str, Any] = hpo_meta["best_params"]
        logger.info("Loaded HPO Best Params (CV ROC-AUC: %.4f)", hpo_meta.get("best_roc_auc", 0.0))
    else:
        logger.info("[STEP 1] Running Bayesian Hyperparameter Optimization...")
        best_xgb_params = run_hpo(config)

    # Step 2: Feature Selection (skip if already run)
    selected_feat_path = config.MODEL_DIR / "selected_features.json"
    if selected_feat_path.exists():
        logger.info("[STEP 2] Loading cached feature selection results...")
        with open(selected_feat_path, "r") as f:
            production_features: List[str] = json.load(f)["features"]
        logger.info("Loaded %d production features.", len(production_features))
    else:
        logger.info("[STEP 2] Evaluating Feature Selection Subsets...")
        _, production_features = run_feature_selection(config)

    # Step 3: Benchmarking & Ensembles (skip if best_model.pkl already exists)
    bench_path = config.MODEL_DIR / "best_model.pkl"
    bench_csv = config.OUTPUT_DIR / "model_benchmark_comparison.csv"
    if bench_path.exists() and bench_csv.exists():
        import pandas as pd
        logger.info("[STEP 3] Loading cached benchmark results...")
        df_bench = pd.read_csv(bench_csv)
        best_model = joblib.load(bench_path)
        best_idx = df_bench["roc_auc"].idxmax()
        best_model_name: str = df_bench.loc[best_idx, "model_name"]
        logger.info("Loaded best model: %s (ROC-AUC: %.4f)", best_model_name, df_bench.loc[best_idx, "roc_auc"])
    else:
        logger.info("[STEP 3] Running Classifier Benchmarks...")
        df_bench, best_model, best_model_name = run_benchmarking(config, best_xgb_params)

    # Step 4: Time-Aware Validation
    logger.info("[STEP 4] Running Chronological Split Validation...")
    df_time_comp = run_time_aware_validation(config)

    # Step 5: Probability Calibration
    logger.info("[STEP 5] Calibrating Base Probabilities...")
    calibrated_model, calibration_metrics = calibrate_model(best_model, config)

    # Step 6: Threshold Optimization
    logger.info("[STEP 6] Tuning Decision Threshold for Imbalanced Classes...")
    opt_threshold = run_threshold_optimization(calibrated_model, config)

    # Step 7: Global SHAP Explanations
    logger.info("[STEP 7] Extracting SHAP Explanations...")
    generate_global_shap_explanations(best_model, config)

    # Step 8: Log Experiment Tracker Summary
    total_duration = time.time() - start_time
    logger.info("[PIPELINE COMPLETED] Total Elapsed Time: %.2f minutes", total_duration / 60)

    # Compile metrics dict
    best_metrics = df_bench[df_bench["model_name"] == best_model_name].iloc[0].to_dict()
    best_metrics_clean = {k: float(v) if isinstance(v, (np.float64, np.int64)) else v for k, v in best_metrics.items()}
    best_metrics_clean.pop("model_name", None)

    tracker.log_run(
        model_name=best_model_name,
        parameters=best_xgb_params if best_model_name == "XGBoost" else {"info": "ensemble/baseline parameters"},
        metrics=best_metrics_clean,
        additional_meta={
            "optimized_threshold": opt_threshold,
            "calibration_method": calibration_metrics["calibration_method"],
            "total_training_duration_min": round(total_duration / 60, 2),
            "features_selected_count": len(production_features),
            "date": time.strftime("%Y-%m-%d"),
        },
    )

    logger.info("=" * 70)
    logger.info("All optimization artifacts successfully compiled and saved.")
    logger.info("=" * 70)


if __name__ == "__main__":
    execute_pipeline()
