import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score
import shap
from ml_config import MLConfig

def run_feature_selection(config: MLConfig):
    """Computes and compares feature importances and tests feature subsets."""
    print("=" * 60)
    print("Feature Selection & Importances")
    print("=" * 60)
    
    # Load preprocessed arrays
    X_train = joblib.load(config.MODEL_DIR / "X_train.pkl")
    X_test = joblib.load(config.MODEL_DIR / "X_test.pkl")
    y_train = joblib.load(config.MODEL_DIR / "y_train.pkl")
    y_test = joblib.load(config.MODEL_DIR / "y_test.pkl")
    
    # Load preprocessor to retrieve column names
    preprocessor = joblib.load(config.MODEL_DIR / "preprocessor.pkl")
    
    # Extract feature names
    feature_names = preprocessor.get_feature_names_out()
    print(f"Total features after preprocessing: {len(feature_names)}")
    
    # If X_train is sparse, convert to dense for RFE and Permutation Importance
    if hasattr(X_train, "toarray"):
        X_train_dense = X_train.toarray()
        X_test_dense = X_test.toarray()
    else:
        X_train_dense = X_train
        X_test_dense = X_test
        
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_pos_weight = neg_count / pos_count
    
    # Base estimator for feature ranking (fast XGBoost)
    base_clf = XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1, 
        scale_pos_weight=scale_pos_weight, random_state=config.RANDOM_SEED, 
        n_jobs=-1, eval_metric="logloss"
    )
    
    print("Computing Mutual Information...")
    mi_scores = mutual_info_classif(X_train_dense, y_train, random_state=config.RANDOM_SEED)
    
    print("Computing Permutation Importance...")
    base_clf.fit(X_train, y_train)
    perm_importances = permutation_importance(
        base_clf, X_test_dense, y_test, n_repeats=5, random_state=config.RANDOM_SEED, n_jobs=-1
    )
    perm_scores = perm_importances.importances_mean
    
    print("Computing SHAP Feature Importance...")
    explainer = shap.TreeExplainer(base_clf)
    shap_values = explainer.shap_values(X_train_dense)
    shap_scores = np.abs(shap_values).mean(axis=0)
    
    print("Computing Recursive Feature Elimination (RFE) Rank...")
    # Reduce step for speed
    rfe = RFE(estimator=base_clf, n_features_to_select=10, step=0.2)
    rfe.fit(X_train, y_train)
    rfe_ranking = rfe.ranking_
    
    # Combine ranks
    df_features = pd.DataFrame({
        "feature_name": feature_names,
        "mutual_info": mi_scores,
        "permutation_importance": perm_scores,
        "shap_importance": shap_scores,
        "rfe_rank": rfe_ranking
    })
    
    # Calculate a composite importance score (higher is better)
    # We rank each feature per method and average the ranks (RFE rank is inverted so smaller is better)
    df_features["mi_rank"] = df_features["mutual_info"].rank(ascending=False)
    df_features["perm_rank"] = df_features["permutation_importance"].rank(ascending=False)
    df_features["shap_rank"] = df_features["shap_importance"].rank(ascending=False)
    df_features["composite_rank"] = (df_features["mi_rank"] + df_features["perm_rank"] + df_features["shap_rank"] + df_features["rfe_rank"]) / 4.0
    
    df_features = df_features.sort_values(by="composite_rank", ascending=True)
    
    # Save feature rank
    rank_path = config.OUTPUT_DIR / "feature_importance_ranking.csv"
    df_features.to_csv(rank_path, index=False)
    print(f"Feature importance ranks saved to {rank_path}")
    
    # Test performance on subsets: All, 30, 20, 15, 10
    ordered_features = df_features["feature_name"].tolist()
    subsets = {
        "All": len(feature_names),
        "Top 30": min(30, len(feature_names)),
        "Top 20": min(20, len(feature_names)),
        "Top 15": min(15, len(feature_names)),
        "Top 10": min(10, len(feature_names))
    }
    
    subset_metrics = []
    skf = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)
    
    for label, count in subsets.items():
        print(f"Evaluating {label} ({count} features)...")
        # Extract columns indices
        top_feature_names = ordered_features[:count]
        feature_indices = [np.where(feature_names == f)[0][0] for f in top_feature_names]
        
        X_train_sub = X_train_dense[:, feature_indices]
        X_test_sub = X_test_dense[:, feature_indices]
        
        # Train and validate XGBoost on the subset
        cv_scores = []
        for train_idx, val_idx in skf.split(X_train_sub, y_train):
            X_tr, y_tr = X_train_sub[train_idx], y_train.iloc[train_idx]
            X_val, y_val = X_train_sub[val_idx], y_train.iloc[val_idx]
            
            clf = XGBClassifier(
                n_estimators=150, max_depth=5, learning_rate=0.1, 
                scale_pos_weight=scale_pos_weight, random_state=config.RANDOM_SEED, 
                n_jobs=-1, eval_metric="logloss"
            )
            clf.fit(X_tr, y_tr)
            probs = clf.predict_proba(X_val)[:, 1]
            cv_scores.append(roc_auc_score(y_val, probs))
            
        mean_cv = np.mean(cv_scores)
        
        # Test score
        clf = XGBClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.1, 
            scale_pos_weight=scale_pos_weight, random_state=config.RANDOM_SEED, 
            n_jobs=-1, eval_metric="logloss"
        )
        clf.fit(X_train_sub, y_train)
        test_probs = clf.predict_proba(X_test_sub)[:, 1]
        test_auc = roc_auc_score(y_test, test_probs)
        
        subset_metrics.append({
            "subset_name": label,
            "feature_count": count,
            "cv_roc_auc": mean_cv,
            "test_roc_auc": test_auc
        })
        
    df_subsets = pd.DataFrame(subset_metrics)
    print("\nFeature Selection Performance Comparison:")
    print(df_subsets.to_string(index=False))
    
    # Save comparison report
    comp_path = config.OUTPUT_DIR / "feature_selection_comparison.csv"
    df_subsets.to_csv(comp_path, index=False)
    
    # Persist the list of Top 30 features as the production feature set
    production_features = ordered_features[:30]
    selected_features_path = config.MODEL_DIR / "selected_features.json"
    with open(selected_features_path, "w") as f:
        json.dump({
            "features": production_features,
            "all_features_count": len(feature_names),
            "production_features_count": len(production_features)
        }, f, indent=4)
        
    print(f"Production features list saved to {selected_features_path}")
    return df_subsets, production_features
