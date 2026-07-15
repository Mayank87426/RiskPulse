import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from ml_config import MLConfig

def run_time_aware_validation(config: MLConfig):
    """Performs time-aware (chronological) split validation and compares it to random splits."""
    print("=" * 60)
    print("Time-Aware Validation (Chronological Split)")
    print("=" * 60)
    
    # Load raw enhanced dataset to get chronological splits by year
    raw_df = pd.read_csv(config.DATA_DIR / "enhanced_training_dataset.csv")
    
    # Check if target column and year are present
    if config.TARGET_COL not in raw_df.columns or "year" not in raw_df.columns:
        print("Missing TARGET_COL or year column in the dataset. Skipping chronological validation.")
        return None
        
    # Split chronologically
    train_df = raw_df[raw_df["year"] <= 2022]
    val_df = raw_df[raw_df["year"] == 2023]
    test_df = raw_df[raw_df["year"] >= 2024]
    
    print(f"Time Split sample sizes:")
    print(f"  - Train (<=2022): {len(train_df)} rows")
    print(f"  - Val (2023):     {len(val_df)} rows")
    print(f"  - Test (>=2024):  {len(test_df)} rows")
    
    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        print("Chronological splits have 0 samples in one of the periods. Skipping.")
        return None
        
    # Preprocess
    preprocessor = joblib.load(config.MODEL_DIR / "preprocessor.pkl")
    
    # Target and features — only drop columns that actually exist in the CSV
    cols_to_drop_train = [c for c in [config.TARGET_COL] + config.IGNORE_COLS if c in train_df.columns]
    y_train_time = train_df[config.TARGET_COL]
    X_train_time = train_df.drop(columns=cols_to_drop_train)

    cols_to_drop_val = [c for c in [config.TARGET_COL] + config.IGNORE_COLS if c in val_df.columns]
    y_val_time = val_df[config.TARGET_COL]
    X_val_time = val_df.drop(columns=cols_to_drop_val)

    cols_to_drop_test = [c for c in [config.TARGET_COL] + config.IGNORE_COLS if c in test_df.columns]
    y_test_time = test_df[config.TARGET_COL]
    X_test_time = test_df.drop(columns=cols_to_drop_test)
    
    # Transform
    # Use a fresh copy of the fitted preprocessor (already trained on the global train split)
    # Align columns in each time-split to match what the preprocessor expects
    try:
        expected_cols = preprocessor.feature_names_in_.tolist()
    except AttributeError:
        expected_cols = list(X_train_time.columns)

    def align_columns(df, expected):
        """Add missing columns as 0 and drop unexpected ones to match preprocessor input."""
        for col in expected:
            if col not in df.columns:
                df[col] = 0
        return df[expected]

    X_train_time = align_columns(X_train_time.copy(), expected_cols)
    X_val_time   = align_columns(X_val_time.copy(),   expected_cols)
    X_test_time  = align_columns(X_test_time.copy(),  expected_cols)

    X_train_transformed = preprocessor.transform(X_train_time)
    X_val_transformed   = preprocessor.transform(X_val_time)
    X_test_transformed  = preprocessor.transform(X_test_time)

    # Convert sparse to dense for XGBoost
    if hasattr(X_train_transformed, "toarray"):
        X_train_transformed = X_train_transformed.toarray()
    if hasattr(X_val_transformed, "toarray"):
        X_val_transformed = X_val_transformed.toarray()
    if hasattr(X_test_transformed, "toarray"):
        X_test_transformed = X_test_transformed.toarray()

    # Train XGBoost
    neg_count = np.sum(y_train_time == 0)
    pos_count = np.sum(y_train_time == 1)
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    
    time_clf = XGBClassifier(
        n_estimators=150, max_depth=5, learning_rate=0.1,
        scale_pos_weight=scale_pos_weight, random_state=config.RANDOM_SEED,
        n_jobs=-1, eval_metric="logloss"
    )
    time_clf.fit(X_train_transformed, y_train_time)
    
    # Evaluate on test set
    time_preds = time_clf.predict(X_test_transformed)
    time_probs = time_clf.predict_proba(X_test_transformed)[:, 1]
    
    time_metrics = {
        "split_strategy": "Chronological_Time_Split",
        "accuracy": accuracy_score(y_test_time, time_preds),
        "precision": precision_score(y_test_time, time_preds, zero_division=0),
        "recall": recall_score(y_test_time, time_preds, zero_division=0),
        "f1": f1_score(y_test_time, time_preds, zero_division=0),
        "roc_auc": roc_auc_score(y_test_time, time_probs)
    }
    
    # Fetch Random Split metrics of XGBoost for comparison from the model benchmark
    random_metrics = {
        "split_strategy": "Random_Stratified_Split",
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "roc_auc": 0.0
    }
    
    benchmark_path = config.OUTPUT_DIR / "model_benchmark_comparison.csv"
    if benchmark_path.exists():
        df_bench = pd.read_csv(benchmark_path)
        xgb_row = df_bench[df_bench["model_name"] == "XGBoost"]
        if not xgb_row.empty:
            random_metrics["accuracy"] = xgb_row.iloc[0]["accuracy"]
            random_metrics["precision"] = xgb_row.iloc[0]["precision"]
            random_metrics["recall"] = xgb_row.iloc[0]["recall"]
            random_metrics["f1"] = xgb_row.iloc[0]["f1"]
            random_metrics["roc_auc"] = xgb_row.iloc[0]["roc_auc"]
            
    df_comparison = pd.DataFrame([random_metrics, time_metrics])
    print("\nSplit Strategy Performance Comparison:")
    print(df_comparison.to_string(index=False))
    
    # Save comparison to CSV
    comp_path = config.OUTPUT_DIR / "time_vs_random_split_comparison.csv"
    df_comparison.to_csv(comp_path, index=False)
    print(f"Comparison report saved to {comp_path}")
    
    return df_comparison

def run_threshold_optimization(model, config: MLConfig) -> float:
    """Finds the decision threshold that maximizes the F1-score and secondary Recall."""
    print("=" * 60)
    print("Threshold Optimization")
    print("=" * 60)
    
    # Load test data
    X_test = joblib.load(config.MODEL_DIR / "X_test.pkl")
    y_test = joblib.load(config.MODEL_DIR / "y_test.pkl")
    
    # Convert sparse matrix to dense array for compatibility with models like HistGradientBoosting
    if hasattr(X_test, "toarray"):
        X_test = X_test.toarray()
    
    # Get probabilities
    y_prob = model.predict_proba(X_test)[:, 1]
    
    best_threshold = 0.50
    best_f1 = -1.0
    best_recall = -1.0
    
    thresholds = np.arange(config.THRESHOLD_MIN, config.THRESHOLD_MAX + config.THRESHOLD_STEP, config.THRESHOLD_STEP)
    records = []
    
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        prec = precision_score(y_test, y_pred, zero_division=0)
        
        records.append({
            "threshold": t,
            "f1": f1,
            "recall": rec,
            "precision": prec
        })
        
        # Select threshold maximizing F1, using Recall as a tiebreaker
        if f1 > best_f1:
            best_f1 = f1
            best_recall = rec
            best_threshold = t
        elif f1 == best_f1 and rec > best_recall:
            best_recall = rec
            best_threshold = t
            
    print(f"Optimized Decision Threshold Found: {best_threshold:.2f}")
    print(f"Metrics at Optimized Threshold:")
    print(f"  - Max F1-Score: {best_f1:.4f}")
    print(f"  - Recall:       {best_recall:.4f}")
    
    # Save optimized threshold
    threshold_meta = {
        "optimized_threshold": float(best_threshold),
        "f1_score": best_f1,
        "recall": best_recall,
        "precision": float(precision_score(y_test, (y_prob >= best_threshold).astype(int), zero_division=0)),
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    threshold_path = config.MODEL_DIR / "optimized_threshold.json"
    with open(threshold_path, "w") as f:
        json.dump(threshold_meta, f, indent=4)
        
    # Save search logs
    df_search = pd.DataFrame(records)
    search_path = config.OUTPUT_DIR / "threshold_search_results.csv"
    df_search.to_csv(search_path, index=False)
    
    print(f"Optimized threshold config saved to {threshold_path}")
    return float(best_threshold)
