import joblib
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, 
    HistGradientBoostingClassifier, VotingClassifier, StackingClassifier
)
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, brier_score_loss, roc_curve, precision_recall_curve
)
from ml_config import MLConfig

def run_benchmarking(config: MLConfig, best_xgboost_params=None):
    """Trains and compares multiple baseline classifiers and ensemble methods."""
    print("=" * 60)
    print("Model Benchmarking & Ensembles")
    print("=" * 60)
    
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
    
    # Calculate scale pos weight for imbalanced data
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_pos_weight = neg_count / pos_count
    
    # Initialize base models
    models = {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced", random_state=config.RANDOM_SEED, max_iter=1000
        ),
        "RandomForest": RandomForestClassifier(
            class_weight="balanced", random_state=config.RANDOM_SEED, n_jobs=-1
        ),
        "ExtraTrees": ExtraTreesClassifier(
            class_weight="balanced", random_state=config.RANDOM_SEED, n_jobs=-1
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            class_weight="balanced", random_state=config.RANDOM_SEED
        ),
        "LightGBM": LGBMClassifier(
            scale_pos_weight=scale_pos_weight, random_state=config.RANDOM_SEED, 
            n_jobs=-1, verbose=-1
        )
    }
    
    # Add XGBoost with optimized HPO parameters if available, otherwise use defaults
    if best_xgboost_params:
        models["XGBoost"] = XGBClassifier(**best_xgboost_params)
    else:
        models["XGBoost"] = XGBClassifier(
            scale_pos_weight=scale_pos_weight, random_state=config.RANDOM_SEED, 
            n_jobs=-1, eval_metric="logloss"
        )
        
    trained_models = {}
    metrics_list = []
    
    # Set up plots
    plt.figure(figsize=(10, 8))
    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    fig_pr, ax_pr = plt.subplots(figsize=(8, 6))
    
    # Train and evaluate individual models
    for name, model in models.items():
        print(f"Training {name}...")
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Evaluate
        start_time = time.time()
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        infer_time = time.time() - start_time
        
        trained_models[name] = model
        
        metrics = {
            "model_name": name,
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
            "roc_auc": roc_auc_score(y_test, probs),
            "brier_score": brier_score_loss(y_test, probs),
            "train_time_sec": train_time,
            "inference_time_sec": infer_time
        }
        metrics_list.append(metrics)
        
        # Add to curves
        fpr, tpr, _ = roc_curve(y_test, probs)
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC: {metrics['roc_auc']:.3f})")
        
        prec, rec, _ = precision_recall_curve(y_test, probs)
        ax_pr.plot(rec, prec, label=f"{name} (F1: {metrics['f1']:.3f})")
        
    # Initialize and train Ensembles
    # 1. Soft Voting Ensemble
    print("Training Soft Voting Ensemble...")
    voting_clf = VotingClassifier(
        estimators=[(k, v) for k, v in trained_models.items()],
        voting="soft"
    )
    start_time = time.time()
    voting_clf.fit(X_train, y_train)
    voting_train_time = time.time() - start_time
    
    start_time = time.time()
    voting_preds = voting_clf.predict(X_test)
    voting_probs = voting_clf.predict_proba(X_test)[:, 1]
    voting_infer_time = time.time() - start_time
    
    voting_metrics = {
        "model_name": "SoftVotingEnsemble",
        "accuracy": accuracy_score(y_test, voting_preds),
        "precision": precision_score(y_test, voting_preds, zero_division=0),
        "recall": recall_score(y_test, voting_preds, zero_division=0),
        "f1": f1_score(y_test, voting_preds, zero_division=0),
        "roc_auc": roc_auc_score(y_test, voting_probs),
        "brier_score": brier_score_loss(y_test, voting_probs),
        "train_time_sec": voting_train_time,
        "inference_time_sec": voting_infer_time
    }
    metrics_list.append(voting_metrics)
    trained_models["SoftVotingEnsemble"] = voting_clf
    
    fpr, tpr, _ = roc_curve(y_test, voting_probs)
    ax_roc.plot(fpr, tpr, label=f"SoftVotingEnsemble (AUC: {voting_metrics['roc_auc']:.3f})", linestyle="--")
    
    prec, rec, _ = precision_recall_curve(y_test, voting_probs)
    ax_pr.plot(rec, prec, label=f"SoftVoting (F1: {voting_metrics['f1']:.3f})", linestyle="--")
    
    # 2. Stacking Ensemble
    print("Training Stacking Ensemble...")
    stacking_clf = StackingClassifier(
        estimators=[(k, v) for k, v in trained_models.items() if k != "SoftVotingEnsemble"],
        final_estimator=LogisticRegression(max_iter=1000),
        n_jobs=-1
    )
    start_time = time.time()
    stacking_clf.fit(X_train, y_train)
    stacking_train_time = time.time() - start_time
    
    start_time = time.time()
    stacking_preds = stacking_clf.predict(X_test)
    stacking_probs = stacking_clf.predict_proba(X_test)[:, 1]
    stacking_infer_time = time.time() - start_time
    
    stacking_metrics = {
        "model_name": "StackingClassifier",
        "accuracy": accuracy_score(y_test, stacking_preds),
        "precision": precision_score(y_test, stacking_preds, zero_division=0),
        "recall": recall_score(y_test, stacking_preds, zero_division=0),
        "f1": f1_score(y_test, stacking_preds, zero_division=0),
        "roc_auc": roc_auc_score(y_test, stacking_probs),
        "brier_score": brier_score_loss(y_test, stacking_probs),
        "train_time_sec": stacking_train_time,
        "inference_time_sec": stacking_infer_time
    }
    metrics_list.append(stacking_metrics)
    trained_models["StackingClassifier"] = stacking_clf
    
    fpr, tpr, _ = roc_curve(y_test, stacking_probs)
    ax_roc.plot(fpr, tpr, label=f"StackingClassifier (AUC: {stacking_metrics['roc_auc']:.3f})", linestyle="-.")
    
    prec, rec, _ = precision_recall_curve(y_test, stacking_probs)
    ax_pr.plot(rec, prec, label=f"Stacking (F1: {stacking_metrics['f1']:.3f})", linestyle="-.")
    
    # Format and save ROC plot
    ax_roc.plot([0, 1], [0, 1], "k:", label="Random Guess")
    ax_roc.set_xlabel("False Positive Rate", fontsize=11, fontname="DejaVu Sans")
    ax_roc.set_ylabel("True Positive Rate", fontsize=11, fontname="DejaVu Sans")
    ax_roc.set_title("ROC Curve Benchmarking", fontsize=13, fontweight="bold")
    ax_roc.legend(loc="lower right")
    ax_roc.grid(True, linestyle="--", alpha=0.5)
    fig_roc.tight_layout()
    roc_plot_path = config.OUTPUT_DIR / "benchmark_roc_curve.png"
    fig_roc.savefig(roc_plot_path, dpi=300)
    plt.close(fig_roc)
    
    # Format and save PR plot
    ax_pr.set_xlabel("Recall", fontsize=11, fontname="DejaVu Sans")
    ax_pr.set_ylabel("Precision", fontsize=11, fontname="DejaVu Sans")
    ax_pr.set_title("Precision-Recall Curve Benchmarking", fontsize=13, fontweight="bold")
    ax_pr.legend(loc="lower left")
    ax_pr.grid(True, linestyle="--", alpha=0.5)
    fig_pr.tight_layout()
    pr_plot_path = config.OUTPUT_DIR / "benchmark_pr_curve.png"
    fig_pr.savefig(pr_plot_path, dpi=300)
    plt.close(fig_pr)
    
    # Convert results list to DataFrame
    df_results = pd.DataFrame(metrics_list)
    print("\nBenchmark Summary Results:")
    print(df_results.to_string(index=False))
    
    # Save benchmark table
    table_path = config.OUTPUT_DIR / "model_benchmark_comparison.csv"
    df_results.to_csv(table_path, index=False)
    print(f"\nBenchmarking tables saved to: {table_path}")
    
    # Identify the best production model (Highest ROC-AUC)
    best_idx = df_results["roc_auc"].idxmax()
    best_model_name = df_results.loc[best_idx, "model_name"]
    best_model = trained_models[best_model_name]
    
    print(f"\n[BEST MODEL] Identified BEST Model for Production: {best_model_name} (ROC-AUC: {df_results.loc[best_idx, 'roc_auc']:.4f})")
    
    # Persist the best model
    joblib.dump(best_model, config.MODEL_DIR / "best_model.pkl")
    print(f"Production model saved to: {config.MODEL_DIR / 'best_model.pkl'}")
    
    # Save a small JSON file pointing to the best model type
    with open(config.MODEL_DIR / "production_model_meta.json", "w") as f:
        json.dump({
            "model_name": best_model_name,
            "metrics": df_results.loc[best_idx].to_dict(),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=4)
        
    return df_results, best_model, best_model_name
