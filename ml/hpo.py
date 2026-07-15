"""hpo.py -- Bayesian hyperparameter optimisation for XGBoost via Optuna."""

import json
import time
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import optuna
from optuna.samplers import TPESampler
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from ml_config import MLConfig, get_logger

logger = get_logger("HPO")

# Suppress Optuna verbose logging unless desired
optuna.logging.set_verbosity(optuna.logging.WARNING)

def run_hpo(config: MLConfig) -> Dict[str, Any]:
    """Runs Optuna hyperparameter optimization for XGBoost using Stratified K-Fold CV."""
    logger.info("=" * 60)
    logger.info("XGBoost Hyperparameter Optimization (Optuna)")
    logger.info("=" * 60)
    
    # Load training data
    X_train = joblib.load(config.MODEL_DIR / "X_train.pkl")
    y_train = joblib.load(config.MODEL_DIR / "y_train.pkl")
    
    # If X_train is a sparse matrix, convert to dense if needed (XGBoost handles sparse natively)
    if hasattr(X_train, "toarray"):
        # XGBoost handles sparse matrices directly, but keeping it sparse is faster
        pass
        
    # Class weight scaling
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_pos_weight = neg_count / pos_count
    
    # Define objective function
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.25, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "scale_pos_weight": scale_pos_weight,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": config.RANDOM_SEED,
            "n_jobs": -1
        }
        
        # Stratified K-Fold CV
        skf = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)
        cv_scores = []
        
        for train_idx, val_idx in skf.split(X_train, y_train):
            # Split
            X_tr, y_tr = X_train[train_idx], y_train.iloc[train_idx]
            X_val, y_val = X_train[val_idx], y_train.iloc[val_idx]
            
            # Train
            clf = XGBClassifier(**params)
            clf.fit(X_tr, y_tr, verbose=False)
            
            # Predict probabilities
            val_probs = clf.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, val_probs)
            cv_scores.append(score)
            
        return np.mean(cv_scores)

    # Run Optuna Study
    sampler = TPESampler(seed=config.RANDOM_SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    
    start_time = time.time()
    study.optimize(objective, n_trials=config.OPTUNA_TRIALS, timeout=config.HPO_TIMEOUT)
    duration = time.time() - start_time
    
    # Save optimization history & best results
    best_params = study.best_params
    best_roc_auc = study.best_value
    
    logger.info("HPO Completed in: %.2fs", duration)
    logger.info("Best CV ROC-AUC: %.4f", best_roc_auc)
    logger.info("Best Trial Params: %s", json.dumps(best_params, indent=2))
    
    # Add non-tuned params
    best_params["scale_pos_weight"] = scale_pos_weight
    best_params["objective"] = "binary:logistic"
    best_params["eval_metric"] = "logloss"
    best_params["random_state"] = config.RANDOM_SEED
    best_params["n_jobs"] = -1
    
    # Save HPO Metadata
    metadata = {
        "best_roc_auc": best_roc_auc,
        "n_trials": len(study.trials),
        "duration_sec": duration,
        "best_params": best_params,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    meta_path = config.MODEL_DIR / "xgboost_hpo_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
        
    logger.info("Best parameters saved to: %s", meta_path)
    
    return best_params
