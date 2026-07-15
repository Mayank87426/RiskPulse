"""explainability.py -- SHAP-based global and local model explainability."""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from ml_config import MLConfig, get_logger

logger = get_logger("Explainability")

def generate_global_shap_explanations(model, config: MLConfig):
    """Generates and saves global SHAP explanations for the production model."""
    logger.info("=" * 60)
    logger.info("Generating Global SHAP Explanations")
    logger.info("=" * 60)
    
    # Load test dataset
    X_train = joblib.load(config.MODEL_DIR / "X_train.pkl")
    X_test = joblib.load(config.MODEL_DIR / "X_test.pkl")
    
    # Check if sparse and convert to dense for SHAP explainer compatibility
    if hasattr(X_train, "toarray"):
        X_train_dense = X_train.toarray()
        X_test_dense = X_test.toarray()
    else:
        X_train_dense = X_train
        X_test_dense = X_test
        
    preprocessor = joblib.load(config.MODEL_DIR / "preprocessor.pkl")
    feature_names = preprocessor.get_feature_names_out()
    
    # Extract underlying tree model for explainability if best model is an ensemble
    model_name = type(model).__name__
    explainer_model = model
    
    if model_name in ["VotingClassifier", "StackingClassifier"]:
        logger.info("Production model is an ensemble (%s). Extracting underlying XGBoost/RF for SHAP tree explainer...", model_name)
        if hasattr(model, "named_estimators_") and "XGBoost" in model.named_estimators_:
            explainer_model = model.named_estimators_["XGBoost"]
        elif hasattr(model, "estimators_") and len(model.estimators_) > 0:
            explainer_model = model.estimators_[0]
        else:
            # Fall back to load the base trained xgboost
            xgb_path = config.MODEL_DIR / "xgboost.pkl"
            if xgb_path.exists():
                explainer_model = joblib.load(xgb_path)
                
    try:
        # Create TreeExplainer
        explainer = shap.TreeExplainer(explainer_model)
        
        # Calculate SHAP values
        # To speed up and avoid memory blowup, take a subset of test set (e.g. 300 samples) if test set is large
        sample_size = min(300, X_test_dense.shape[0])
        X_test_sample = X_test_dense[:sample_size]
        
        shap_values = explainer.shap_values(X_test_sample)
        
        # In SHAP v0.40+, binary classification output might be a list or 3D array
        if isinstance(shap_values, list) and len(shap_values) == 2:
            # Use positive class values
            shap_values_to_plot = shap_values[1]
        else:
            shap_values_to_plot = shap_values
            
        # 1. Summary Dot Plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values_to_plot, X_test_sample, feature_names=feature_names, show=False
        )
        plt.title("SHAP Feature Importance (Summary Dot Plot)", fontsize=13, fontweight="bold", pad=20)
        plt.tight_layout()
        dot_path = config.SHAP_DIR / "shap_summary_dot.png"
        plt.savefig(dot_path, dpi=300)
        plt.close()
        logger.info("SHAP Dot plot saved to %s", dot_path)
        
        # 2. Summary Bar Plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values_to_plot, X_test_sample, feature_names=feature_names, plot_type="bar", show=False
        )
        plt.title("SHAP Feature Importance (Summary Bar Plot)", fontsize=13, fontweight="bold", pad=20)
        plt.tight_layout()
        bar_path = config.SHAP_DIR / "shap_summary_bar.png"
        plt.savefig(bar_path, dpi=300)
        plt.close()
        logger.info("SHAP Bar plot saved to %s", bar_path)
        
        # 3. Dependence Plot for the top feature
        # Calculate mean absolute SHAP values to find top feature
        mean_shap = np.abs(shap_values_to_plot).mean(axis=0)
        top_idx = np.argmax(mean_shap)
        top_feature_name = feature_names[top_idx]
        
        plt.figure(figsize=(8, 5))
        shap.dependence_plot(
            top_idx, shap_values_to_plot, X_test_sample, feature_names=feature_names, show=False
        )
        plt.title(f"SHAP Dependence Plot: {top_feature_name}", fontsize=12, fontweight="bold", pad=15)
        plt.tight_layout()
        dep_path = config.SHAP_DIR / "shap_dependence_plot.png"
        plt.savefig(dep_path, dpi=300)
        plt.close()
        logger.info("SHAP Dependence plot saved to %s", dep_path)
        
        # Save explainer object for fast local inference later
        joblib.dump(explainer, config.MODEL_DIR / "shap_explainer.pkl")
        logger.info("SHAP Explainer saved to models directory successfully.")
        
    except Exception as e:
        logger.error("Error generating SHAP explanations: %s", e)
        # Create a mock/empty file to prevent downstream script crashes
        config.SHAP_DIR.mkdir(parents=True, exist_ok=True)
        # Create a simple matplotlib placeholder chart
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "SHAP explainability not supported\nfor current ensemble model", 
                ha="center", va="center", fontsize=12, color="gray")
        fig.savefig(config.SHAP_DIR / "shap_summary_dot.png")
        fig.savefig(config.SHAP_DIR / "shap_summary_bar.png")
        fig.savefig(config.SHAP_DIR / "shap_dependence_plot.png")
        plt.close(fig)

def get_local_shap_explanation(preprocessed_row, feature_names, config: MLConfig):
    """Computes local SHAP explanation for a single prediction row."""
    # Convert preprocessed row to dense numpy array if needed
    if hasattr(preprocessed_row, "toarray"):
        row_dense = preprocessed_row.toarray()
    else:
        row_dense = np.array(preprocessed_row)
        
    if len(row_dense.shape) == 1:
        row_dense = row_dense.reshape(1, -1)
        
    explainer_path = config.MODEL_DIR / "shap_explainer.pkl"
    if not explainer_path.exists():
        return {"error": "SHAP Explainer not available"}
        
    try:
        explainer = joblib.load(explainer_path)
        shap_values = explainer.shap_values(row_dense)
        
        # Handle list/multi-class outputs from SHAP TreeExplainer
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values_row = shap_values[1][0]
        else:
            shap_values_row = shap_values[0] if len(shap_values.shape) > 1 else shap_values
            
        base_value = float(explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value)
        
        # Map feature names to shap values
        contribs = []
        for name, val in zip(feature_names, shap_values_row):
            contribs.append({
                "feature": name,
                "shap_value": float(val)
            })
            
        df_contribs = pd.DataFrame(contribs)
        
        # Sort contributions
        positive_contribs = df_contribs[df_contribs["shap_value"] > 0].sort_values(by="shap_value", ascending=False).head(3).to_dict(orient="records")
        negative_contribs = df_contribs[df_contribs["shap_value"] < 0].sort_values(by="shap_value", ascending=True).head(3).to_dict(orient="records")
        
        # Format names for presentation (remove numerical prefixes from ColumnTransformer encoding)
        for c in positive_contribs + negative_contribs:
            clean_name = c["feature"]
            if clean_name.startswith("num__"):
                clean_name = clean_name.replace("num__", "")
            elif clean_name.startswith("cat__"):
                clean_name = clean_name.replace("cat__", "")
            # Replace underscores with spaces and capitalize
            c["feature_clean"] = clean_name.replace("_", " ").title()
            
        return {
            "base_value": base_value,
            "positive_factors": positive_contribs,
            "negative_factors": negative_contribs
        }
        
    except Exception as e:
        return {"error": f"Failed to compute local explanation: {str(e)}"}

def generate_local_shap_waterfall_plot(preprocessed_row, feature_names, company_name: str, config: MLConfig) -> str:
    """Generates and saves a local SHAP waterfall plot for a specific company's prediction."""
    # Convert preprocessed row to dense numpy array if needed
    if hasattr(preprocessed_row, "toarray"):
        row_dense = preprocessed_row.toarray()
    else:
        row_dense = np.array(preprocessed_row)
        
    if len(row_dense.shape) == 1:
        row_dense = row_dense.reshape(1, -1)
        
    explainer_path = config.MODEL_DIR / "shap_explainer.pkl"
    if not explainer_path.exists():
        return ""
        
    try:
        explainer = joblib.load(explainer_path)
        shap_values = explainer.shap_values(row_dense)
        
        # Handle list/multi-class outputs from SHAP TreeExplainer
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values_row = shap_values[1][0]
        else:
            shap_values_row = shap_values[0] if len(shap_values.shape) > 1 else shap_values
            
        base_value = float(explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value)
        
        # Format feature names to remove prefixes
        clean_names = []
        for name in feature_names:
            clean_name = name
            if clean_name.startswith("num__"):
                clean_name = clean_name.replace("num__", "")
            elif clean_name.startswith("cat__"):
                clean_name = clean_name.replace("cat__", "")
            clean_names.append(clean_name.replace("_", " ").title())
            
        # Construct Explanation object
        exp = shap.Explanation(
            values=shap_values_row,
            base_values=base_value,
            data=row_dense[0],
            feature_names=clean_names
        )
        
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(exp, show=False)
        plt.title(f"SHAP Local Explanation Waterfall for {company_name}", fontsize=12, fontweight="bold", pad=15)
        plt.tight_layout()
        
        # Save plot
        # Sanitize company name for filename
        safe_name = "".join([c if c.isalnum() else "_" for c in company_name]).lower()
        waterfall_path = config.SHAP_DIR / f"shap_waterfall_{safe_name}.png"
        plt.savefig(waterfall_path, dpi=300)
        plt.close()
        return str(waterfall_path)
    except Exception as e:
        logger.error("Error generating local SHAP waterfall plot: %s", e)
        return ""
