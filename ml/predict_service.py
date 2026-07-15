"""predict_service.py -- Production prediction service for real-time company risk inference."""

import joblib
import json
import time
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from app.db import get_connection
from ml_config import MLConfig, get_logger
from explainability import get_local_shap_explanation

logger = get_logger("PredictionService")

class PredictionService:
    def __init__(self, config: MLConfig):
        self.config = config
        self.preprocessor_path = config.MODEL_DIR / "preprocessor.pkl"
        self.model_path = config.MODEL_DIR / "calibrated_xgboost.pkl"
        self.best_model_path = config.MODEL_DIR / "best_model.pkl"
        self.threshold_path = config.MODEL_DIR / "optimized_threshold.json"
        
        # Load preprocessor and model
        self.preprocessor = joblib.load(self.preprocessor_path) if self.preprocessor_path.exists() else None
        
        # Load model (prefer calibrated model, fallback to best baseline)
        if self.model_path.exists():
            self.model = joblib.load(self.model_path)
        elif self.best_model_path.exists():
            self.model = joblib.load(self.best_model_path)
        else:
            self.model = None
            
        # Load optimized threshold
        if self.threshold_path.exists():
            with open(self.threshold_path, "r") as f:
                self.threshold = json.load(f).get("optimized_threshold", 0.50)
        else:
            self.threshold = 0.50
            
    def get_company_layoff_features(self, company_name: str) -> pd.DataFrame:
        """Fetches layoff history for a company from DB and constructs the latest feature vector."""
        conn = get_connection()
        
        # 1. Fetch company profile metadata
        comp_query = """
            SELECT id, name, industry, country, company_type 
            FROM companies 
            WHERE name = %s
        """
        df_comp = pd.read_sql(comp_query, conn, params=(company_name,))
        if df_comp.empty:
            conn.close()
            return pd.DataFrame()
            
        company_id = int(df_comp.iloc[0]["id"])
        industry = df_comp.iloc[0]["industry"]
        country = df_comp.iloc[0]["country"]
        company_type = df_comp.iloc[0]["company_type"]
        
        # 2. Fetch layoff events sorted by date
        layoff_query = """
            SELECT layoff_date, employees_laid_off, percentage_laid_off, 
                   company_size_before, company_size_after
            FROM layoffs 
            WHERE company_id = %s 
            ORDER BY layoff_date ASC
        """
        df_layoffs = pd.read_sql(layoff_query, conn, params=(company_id,))
        conn.close()
        
        if df_layoffs.empty:
            # Company has no layoff history
            return pd.DataFrame()
            
        # 3. Construct chronological aggregations (same logic as build_training_dataset.py)
        df_layoffs["layoff_date"] = pd.to_datetime(df_layoffs["layoff_date"])
        df_layoffs["employees_laid_off"] = df_layoffs["employees_laid_off"].fillna(0).astype(float)
        df_layoffs["percentage_laid_off"] = df_layoffs["percentage_laid_off"].fillna(0).astype(float)
        df_layoffs["company_size_before"] = df_layoffs["company_size_before"].fillna(0).astype(float)
        df_layoffs["company_size_after"] = df_layoffs["company_size_after"].fillna(0).astype(float)
        
        # We need to construct the features for the LATEST event
        n_events = len(df_layoffs)
        latest_event = df_layoffs.iloc[-1]
        
        # History excluding the latest event (prior events)
        df_prior = df_layoffs.iloc[:-1] if n_events > 1 else pd.DataFrame()
        
        # Build features
        prev_events = n_events - 1
        total_prev = df_prior["employees_laid_off"].sum() if not df_prior.empty else 0.0
        largest_prev = df_prior["employees_laid_off"].max() if not df_prior.empty else 0.0
        avg_prev = df_prior["employees_laid_off"].mean() if not df_prior.empty else 0.0
        max_prev_pct = df_prior["percentage_laid_off"].max() if not df_prior.empty else 0.0
        avg_prev_pct = df_prior["percentage_laid_off"].mean() if not df_prior.empty else 0.0
        
        if n_events > 1:
            days_since = (latest_event["layoff_date"] - df_prior.iloc[-1]["layoff_date"]).days
            prev_layoff_size = float(df_prior.iloc[-1]["employees_laid_off"])
            prev_company_size = float(df_prior.iloc[-1]["company_size_before"])
        else:
            days_since = 9999.0
            prev_layoff_size = 0.0
            prev_company_size = float(latest_event["company_size_before"])
            
        # Feature Engineering (feature_engineering.py)
        reduction_ratio = (
            latest_event["employees_laid_off"] / latest_event["company_size_before"]
            if latest_event["company_size_before"] > 0 else 0.0
        )
        remaining_ratio = (
            latest_event["company_size_after"] / latest_event["company_size_before"]
            if latest_event["company_size_before"] > 0 else 1.0
        )
        shrinkage = latest_event["company_size_before"] - latest_event["company_size_after"]
        
        layoff_growth = (
            latest_event["employees_laid_off"] / prev_layoff_size 
            if prev_layoff_size > 0 else 1.0
        )
        
        # Rolling averages (3 events prior to the latest)
        rolling_layoffs = df_prior["employees_laid_off"].tail(3) if not df_prior.empty else pd.Series(dtype=float)
        rolling_pcts = df_prior["percentage_laid_off"].tail(3) if not df_prior.empty else pd.Series(dtype=float)
        
        rolling_avg_layoff = rolling_layoffs.mean() if not rolling_layoffs.empty else 0.0
        rolling_avg_percentage = rolling_pcts.mean() if not rolling_pcts.empty else 0.0
        
        layoff_velocity = (
            latest_event["employees_laid_off"] / days_since 
            if days_since > 0 else 0.0
        )
        
        # Cumulative prior reduction
        # Sum reduction ratio of all events up to the latest
        prior_reduction_ratios = []
        for idx, r in df_layoffs.iterrows():
            size_b = r["company_size_before"]
            laid = r["employees_laid_off"]
            ratio = laid / size_b if size_b > 0 else 0.0
            prior_reduction_ratios.append(ratio)
        cumulative_reduction = sum(prior_reduction_ratios)
        
        company_size_change = latest_event["company_size_before"] - prev_company_size
        company_age_proxy = prev_events + 1
        layoffs_per_year = prev_events / company_age_proxy
        
        # 4. Construct Row Dictionary
        feat_dict = {
            "company_id": company_id,
            "industry": industry,
            "country": country,
            "company_type": company_type,
            
            "employees_laid_off": latest_event["employees_laid_off"],
            "percentage_laid_off": latest_event["percentage_laid_off"],
            "company_size_before": latest_event["company_size_before"],
            "company_size_after": latest_event["company_size_after"],
            
            "previous_layoff_events": prev_events,
            "total_previous_laid_off": total_prev,
            "largest_previous_layoff": largest_prev,
            "average_previous_layoff": avg_prev,
            "maximum_previous_percentage": max_prev_pct,
            "average_previous_percentage": avg_prev_pct,
            "days_since_last_layoff": days_since,
            
            "year": latest_event["layoff_date"].year,
            "month": latest_event["layoff_date"].month,
            "quarter": latest_event["layoff_date"].quarter,
            
            # Engineered features
            "workforce_reduction_ratio": reduction_ratio,
            "remaining_workforce_ratio": remaining_ratio,
            "workforce_shrinkage": shrinkage,
            "previous_layoff_size": prev_layoff_size,
            "layoff_growth": layoff_growth,
            "rolling_avg_layoff": rolling_avg_layoff,
            "rolling_avg_percentage": rolling_avg_percentage,
            "layoff_velocity": layoff_velocity,
            "large_layoff": int(latest_event["employees_laid_off"] >= 500),
            "major_workforce_cut": int(latest_event["percentage_laid_off"] >= 25),
            "cumulative_reduction": cumulative_reduction,
            "previous_company_size": prev_company_size,
            "company_size_change": company_size_change,
            "company_age_proxy": company_age_proxy,
            "layoffs_per_year": layoffs_per_year
        }
        
        return pd.DataFrame([feat_dict])
        
    def predict_recurrence(self, company_name: str) -> dict:
        """Runs the prediction pipeline for a selected company."""
        if not self.model or not self.preprocessor:
            return {"error": "Prediction service model or preprocessor is not loaded."}
            
        # 1. Build features from database
        df_feat = self.get_company_layoff_features(company_name)
        if df_feat.empty:
            return {
                "company_name": company_name,
                "has_history": False,
                "probability": 0.0,
                "risk_category": "Very Low",
                "prediction_label": 0,
                "message": "No historical layoff events recorded. AI recurrence prediction not applicable."
            }
            
        # 2. Extract metadata
        company_id = int(df_feat.iloc[0]["company_id"])
        
        # 3. Preprocess
        cols_to_drop = [c for c in self.config.IGNORE_COLS if c in df_feat.columns]
        X_raw = df_feat.drop(columns=cols_to_drop)
        X_trans = self.preprocessor.transform(X_raw)
        if hasattr(X_trans, "toarray"):
            X_trans = X_trans.toarray()
        
        # 4. Predict probability
        prob = float(self.model.predict_proba(X_trans)[0, 1])
        
        # 5. Classify using optimized threshold
        predicted_label = 1 if prob >= self.threshold else 0
        
        # Map dynamic risk category
        t = self.threshold
        if prob < t * 0.4:
            risk_category = "Very Low"
        elif prob < t:
            risk_category = "Low"
        elif prob < t + (1.0 - t) * 0.33:
            risk_category = "Medium"
        elif prob < t + (1.0 - t) * 0.66:
            risk_category = "High"
        else:
            risk_category = "Critical"
            
        # 6. Fetch SHAP Explainability
        feature_names = self.preprocessor.get_feature_names_out()
        shap_explanation = get_local_shap_explanation(X_trans, feature_names, self.config)
        
        # 7. Generate local SHAP waterfall plot image
        from explainability import generate_local_shap_waterfall_plot
        waterfall_path = generate_local_shap_waterfall_plot(X_trans, feature_names, company_name, self.config)
        
        # Calculate mock confidence score based on distance from threshold
        confidence = round(0.70 + abs(prob - self.threshold) * 0.25, 2)
        confidence = min(0.99, max(0.50, confidence))
        
        result = {
            "company_name": company_name,
            "company_id": company_id,
            "has_history": True,
            "probability": prob,
            "probability_percentage": round(prob * 100, 1),
            "risk_category": risk_category,
            "decision_threshold": self.threshold,
            "prediction_label": predicted_label,
            "confidence": confidence,
            "confidence_percentage": round(confidence * 100, 1),
            "prediction_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "shap_explanation": shap_explanation,
            "waterfall_plot_path": waterfall_path
        }
        
        # Log prediction to local history registry
        self._log_prediction_to_history(result)
        return result
        
    def _log_prediction_to_history(self, result: dict) -> None:
        """Appends prediction results to history registry file."""
        history_path = self.config.OUTPUT_DIR / "prediction_history.json"
        history = []
        if history_path.exists():
            try:
                with open(history_path, "r") as f:
                    history = json.load(f)
            except Exception:
                history = []
                
        # Clean result dictionary of SHAP objects for simple logging
        log_entry = {
            "company_name": result["company_name"],
            "probability": result["probability"],
            "risk_category": result["risk_category"],
            "prediction_label": result["prediction_label"],
            "confidence": result["confidence"],
            "timestamp": result["prediction_timestamp"]
        }
        
        history.append(log_entry)
        # Limit history to latest 100 runs
        history = history[-100:]
        
        try:
            with open(history_path, "w") as f:
                json.dump(history, f, indent=4)
        except Exception as e:
            logger.error("Error logging prediction history: %s", e)
            
if __name__ == "__main__":
    config = MLConfig()
    service = PredictionService(config)
    logger.info("Testing inference service for Google...")
    res = service.predict_recurrence("Google")
    logger.info(json.dumps(res, indent=2))
