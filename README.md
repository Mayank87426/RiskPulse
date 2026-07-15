# 📊 RiskPulse Predictive Intelligence Platform

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![ML Pipeline](https://img.shields.io/badge/Pipeline-Optuna%20%7C%20SHAP%20%7C%20Calibration-indigo.svg)](#key-platform-features)
[![UI Integration](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)](app/main.py)
[![Database](https://img.shields.io/badge/Database-PostgreSQL-blue.svg)](scripts/database.py)

RiskPulse is an enterprise-grade workforce intelligence platform that transforms raw company profiles, historical layoff patterns, and real-time news coverage into forward-looking predictive risk insights. The system aids corporate leaders and executive planners in proactively identifying workforce stability issues and forecasting the recurrence of structural downsizing events before they happen.

---

## 🎯 Platform Evolution: Rule-Based vs. ML-Driven Intelligence

In legacy iterations, workforce risk scoring relied on a static, heuristic **Rule-Based Model** with arbitrary conditions and weight distributions (e.g., assigning a risk category purely based on whether a layoff exceeded 500 people, or counting previous layoff events). This naive approach suffered from severe limitations:
* **High False Alarm Rates**: Heuristic metrics did not account for historical context or industry baseline variance.
* **Lack of Probability Calibration**: Standard scoring couldn't represent actual statistical probability (e.g., what does a score of "85" mean in empirical terms?).
* **Inability to Learn Complex Relationships**: Interaction terms between layoff velocity, shrinkage ratio, and rolling averages were completely ignored.

The platform has been overhauled into an **ML-Driven Predictive System**. RiskPulse now runs an end-to-end, multi-stage machine learning pipeline that uses hyperparameter-optimized estimators, probability calibration techniques, and deep explainability tools to offer mathematically sound predictions.

---

## 🏗️ System Architecture

The following diagram illustrates the lifecycle of the RiskPulse platform—from ingestion to production deployment:

```mermaid
flowchart TD
    %% Styling Classes
    classDef db fill:#1e293b,stroke:#3b82f6,stroke-width:2px,color:#f8fafc;
    classDef process fill:#334155,stroke:#94a3b8,stroke-width:1px,color:#f8fafc;
    classDef pipeline fill:#1e1b4b,stroke:#6366f1,stroke-width:2px,color:#f8fafc;
    classDef calibration fill:#064e3b,stroke:#10b981,stroke-width:2px,color:#f8fafc;
    classDef explain fill:#7c2d12,stroke:#f97316,stroke-width:2px,color:#f8fafc;
    classDef ui fill:#0f172a,stroke:#ec4899,stroke-width:2px,color:#f8fafc;

    %% Data layer
    subgraph Storage ["1. Relational Storage Layer"]
        DB[(PostgreSQL Database<br>companies | layoffs | news)]:::db
    end

    %% Pipeline Stage 1
    subgraph DataPrep ["2. Feature Pipeline"]
        FE[Feature Engineering<br>- Layoff Growth & Velocity<br>- Remaining Workforce Ratio<br>- Rolling Aggregations]:::process
        PP[Preprocessing<br>- Missing Value Imputers<br>- One-Hot Encoder<br>- ColumnTransformer]:::process
    end

    %% Pipeline Stage 2
    subgraph Optimization ["3. Core Optimization Engine"]
        FS[Feature Selection<br>- Mutual Info | SHAP | RFE]:::pipeline
        HPO[Bayesian HPO<br>- Optuna 100 trials<br>- Stratified K-Fold CV]:::pipeline
        BM[Classifier Benchmarking<br>- LGBM | XGBoost | RF<br>- Stacking & Voting Ensembles]:::pipeline
    end

    %% Pipeline Stage 3
    subgraph PostTraining ["4. Post-Training Calibration & Tuning"]
        CAL[Probability Calibration<br>- Platt Scaling vs Isotonic<br>- Minimizing Brier Score & ECE]:::calibration
        THR[Decision Threshold Tuning<br>- Imbalanced Step-Search<br>- F1 & Recall Optimization]:::calibration
    end

    %% Pipeline Stage 4
    subgraph Explainability ["5. Interpretability Engine"]
        SHAP[SHAP Explainability<br>- Global Summary Dot & Bar Plots<br>- Local Waterfall Diagrams]:::explain
    end

    %% Dashboard layer
    subgraph Delivery ["6. Deployment & UI Delivery"]
        APP[Streamlit Web UI<br>- Live AI Inference Tab<br>- Gauge Charts & SHAP Plots]:::ui
        SRV[Prediction Service<br>- DB-Backed Feature Builder<br>- Inference Logging Registry]:::ui
    end

    %% Relationships
    DB -->|Raw Query Ingestion| FE
    FE -->|Enhanced Training Data| PP
    PP -->|Train-Test Split Matrices| FS
    FS & HPO --> BM
    BM -->|Identify Best Model| CAL
    CAL --> THR
    THR -->|Calibrated Stacking Model| SHAP
    SHAP --> SRV
    SRV --> APP
    APP -->|Logs Prediction History| DB
```

---

## 🌟 Key Platform Features

### 1. Automated Bayesian Hyperparameter Optimization (HPO)
Using [ml/hpo.py](ml/hpo.py), the platform optimizes XGBoost's non-linear parameters via **Optuna** running a 100-trial study. Evaluated using a 5-fold Stratified Cross-Validation strategy, it tunes:
* Learning rate, max depth, subsample, and colsample parameters.
* Regularization parameters (`reg_alpha` and `reg_lambda`) to prevent overfitting.
* Target-class balance weight scaling (`scale_pos_weight`) to accommodate severe class imbalances.

### 2. Multi-Model Benchmarking & Stacking Ensembles
The [ml/benchmark.py](ml/benchmark.py) engine evaluates a suite of diverse classifiers:
* **Baseline models**: Logistic Regression, Random Forest, ExtraTrees, HistGradientBoosting, LightGBM, and XGBoost.
* **Soft Voting Ensemble**: Combines the prediction probabilities across base estimators.
* **Stacking Classifier**: Leverages all base models as first-level estimators, feeding predictions into a Meta-Logistic Regression model to find optimal ensemble weights.

### 3. Comprehensive Feature Selection
To avoid overfitting and improve inference speed, the feature selection module [ml/feature_selector.py](ml/feature_selector.py) computes a ranking consensus from four methodologies:
* **Mutual Information**: Measures statistical dependency.
* **Permutation Importance**: Captures out-of-bag accuracy drops on Random Forests.
* **SHAP Mean Absolute Magnitude**: Estimates global model contributions.
* **Recursive Feature Elimination (RFE)**: Systematically drops weak features.

### 4. Probability Calibration
Raw ML models often yield uncalibrated confidence scores. In [ml/calibration.py](ml/calibration.py), Platt Scaling (Sigmoid) and Isotonic Regression are fitted on validation splits to align predicted probabilities with empirical likelihoods. Performance is evaluated using Brier Score and Expected Calibration Error (ECE) curves.

### 5. Imbalanced Class Threshold Tuning
Standard classification uses a static 0.5 decision threshold, which degrades performance on imbalanced targets. The [ml/validation.py](ml/validation.py) script scans thresholds from 0.05 to 0.95 to locate the decision cutoff that maximizes the **F1-Score** while maintaining **Recall** as a secondary objective.

### 6. Time-Aware Chronological Validation
To combat data leakage and evaluate real-world generalization, the platform tests models under a temporal split:
* **Train**: Records prior to and including 2022.
* **Validation**: Records from 2023.
* **Test**: Records from 2024 onwards.
This chronological validation exposes how macroeconomic shifts and structural layout changes affect model accuracy over time.

### 7. Deep Interpretability (SHAP)
Utilizing the game-theoretic framework SHAP (SHapley Additive exPlanations) via [ml/explainability.py](ml/explainability.py), RiskPulse generates:
* **Global Explanations**: Summary bar and dot charts plotting feature importance across the entire dataset.
* **Local Explanations**: Waterfall plots explaining individual company forecasts, showing feature shifts relative to the base output rate.

### 8. Production Prediction Service
The [ml/predict_service.py](ml/predict_service.py) provides a database-backed inference wrapper:
1. Ingests a company name.
2. Performs SQL queries to retrieve its entire historical layoff track record.
3. Automatically constructs the engineered features (rolling averages, velocities, metrics) on-the-fly.
4. Feeds the data through the preprocessor pipeline and calibrated Stacking model.
5. Emits the probability score, risk category, and local SHAP explanation values.
6. Automatically appends the prediction result to the persistent history registry.

---

## 💻 How to Run & Deploy

### 1. Environment Setup
Create a virtual environment (using Python 3.10 or 3.11) and install the necessary dependencies:

```powershell
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install core requirements
pip install -r requirements.txt

# Install ML optimization dependencies
pip install optuna shap
```

Create a `.env` file in the project root directory and define your API key:
```env
NEWS_API_KEY=your_news_api_key_here
```

### 2. Database Initialization & Data Loading
Ensure PostgreSQL is running locally and your credentials in `app/db.py` and `scripts/database.py` are properly configured. Run the following ingestion scripts using the virtual environment python:

```powershell
# Create tables and clean raw files
.\.venv\Scripts\python.exe scripts/data_cleaning.py

# Ingest metadata and layoff events
.\.venv\Scripts\python.exe scripts/load_companies.py
.\.venv\Scripts\python.exe scripts/load_layoffs.py

# Generate baseline heuristic risk scores
.\.venv\Scripts\python.exe scripts/generate_risk_scores.py

# Fetch relevant news articles
.\.venv\Scripts\python.exe scripts/fetch_news.py
```

### 3. Running the ML Pipeline
Execute the machine learning pipeline in sequence using the virtual environment python. This guarantees the model pickle files match the package versions in Streamlit:

```powershell
# Step A: Perform time-series rolling feature engineering
.\.venv\Scripts\python.exe ml/feature_engineering.py

# Step B: Impute, scale, encode, and split dataset matrices
.\.venv\Scripts\python.exe ml/preprocess.py

# Step C: Run the end-to-end Optimization Pipeline (HPO, Benchmark, Calibration, SHAP)
.\.venv\Scripts\python.exe ml/run_pipeline.py
```

> [!NOTE]
> The `run_pipeline.py` script checks for existing cached results (like HPO parameters or feature subsets) to bypass expensive computations. To perform a complete refresh, delete the contents of `ml/models/` and re-run.

### 4. Booting the Streamlit Web Application
Launch the interactive enterprise dashboard using the virtual environment's Streamlit executable:

```powershell
.\.venv\Scripts\streamlit.exe run app/main.py
```
Open your browser and navigate to `http://localhost:8501`. Choose the **AI Risk Forecast** tab on any company detail screen to view the predictive indicators.

---

## 📁 Repository Directory Structure

```text
RiskPulse/
│
├── .env                              # Application environment variables (News API keys)
├── requirements.txt                  # Consolidated python dependency list
│
├── app/                              # Streamlit Application Source Code
│   ├── main.py                       # Main application layout and dashboard router
│   ├── db.py                         # PostgreSQL database connector & raw querying utils
│   │
│   ├── components/                   # Modular dashboard UI panels
│   │   ├── ai_prediction.py          # AI tab showing gauges, SHAP factors, and metrics
│   │   ├── risk_gauge.py             # Plotly speedometer indicators
│   │   ├── charts.py                 # Historical layoff combination charts & sparklines
│   │   ├── kpi_cards.py              # Statistical summaries & circular progress indicators
│   │   ├── executive_summary.py      # LLM-ready summary card templates
│   │   ├── news_panel.py             # News feed container with sentiment indicators
│   │   └── sidebar.py                # Company selectors and DB synchronization stats
│   │
│   └── utils/                        # Design styles and configuration
│       ├── config.py                 # Application metadata configurations
│       ├── helpers.py                # Text processing and sentiment analyzers
│       └── styles.py                 # Custom CSS stylesheet injections
│
├── ml/                               # Machine Learning Pipeline
│   ├── ml_config.py                  # Hard-coded configuration parameters & log setups
│   ├── feature_engineering.py        # Generates time-series lagging & rolling aggregations
│   ├── preprocess.py                 # Creates sklearn ColumnTransformers and test splits
│   ├── hpo.py                        # Optuna Bayesian hyperparameter search routines
│   ├── feature_selector.py           # Features importance filtering consensus
│   ├── benchmark.py                  # Trains and compares base classifiers & ensembles
│   ├── validation.py                 # Temporal splits analysis & threshold search
│   ├── calibration.py                # Sigmoid & Isotonic probability calibration
│   ├── explainability.py             # SHAP local and global explainers
│   ├── predict_service.py            # Live inference builder with feature assembly
│   ├── run_pipeline.py               # E2E pipeline orchestrator
│   │
│   ├── data/                         # Train-ready CSV data exports
│   │   ├── training_dataset.csv      # Baseline raw training dataset
│   │   └── enhanced_training_dataset.csv # Post feature-engineered data export
│   │
│   ├── models/                       # Serialized models and configurations
│   │   ├── best_model.pkl            # Candidate baseline or ensemble model
│   │   ├── calibrated_xgboost.pkl    # Best production-ready calibrated model
│   │   ├── preprocessor.pkl          # Scikit-learn ColumnTransformer instance
│   │   ├── selected_features.json    # Features chosen by the selector module
│   │   ├── optimized_threshold.json  # Decision threshold maximizing F1 & Recall
│   │   └── production_model_meta.json# Performance metrics summary file
│   │
│   └── outputs/                      # Saved validation charts, plots and logs
│       ├── shap/                     # Global SHAP summary bar/dot png images
│       ├── logs/                     # Rotation log files for run_pipeline
│       ├── model_benchmark_comparison.csv # Performance metrics table
│       ├── threshold_search_results.csv   # Threshold optimization scan outputs
│       └── time_vs_random_split_comparison.csv # Split strategy results
│
└── scripts/                          # Data Ingestion & Populating Scripts
    ├── database.py                   # PostgreSQL schema structures
    ├── data_cleaning.py              # Cleans raw layouts datasets
    ├── load_companies.py             # Loads company metadata from source lists
    ├── load_layoffs.py               # Populates layoffs events database
    ├── fetch_news.py                 # Syncs articles from NewsAPI
    └── generate_risk_scores.py       # Populates legacy rule-based metrics in DB
```

---

## 📈 Production Metrics Summary

### Model Benchmarking Comparison
The following performance metrics were captured from the latest pipeline execution comparing all baseline models, voting, and stacking classifier ensembles on the test set:

| Model Name | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Brier Score | Train Time (s) | Inference Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| StackingClassifier | 89.6% | 0.769 | 0.175 | 0.286 | **0.777** | **0.088** | 40.112 | 0.372 |
| **SoftVotingEnsemble** | **85.9%** | **0.388** | **0.333** | **0.358** | **0.772** | **0.109** | 16.672 | 0.592 |
| XGBoost (HPO Optimized) | 73.5% | 0.233 | 0.544 | 0.326 | 0.763 | 0.163 | 0.312 | 0.008 |
| ExtraTrees | 88.6% | 0.556 | 0.175 | 0.267 | 0.762 | 0.092 | 0.641 | 0.149 |
| RandomForest | 86.1% | 0.400 | 0.351 | 0.374 | 0.765 | 0.105 | 0.610 | 0.152 |
| Logistic Regression | 71.6% | 0.214 | 0.526 | 0.305 | 0.738 | 0.205 | 2.128 | 0.003 |
| HistGradientBoosting | 84.9% | 0.367 | 0.386 | 0.376 | 0.728 | 0.113 | 13.396 | 0.099 |
| LightGBM | 83.4% | 0.311 | 0.333 | 0.322 | 0.689 | 0.119 | 1.281 | 0.108 |

> [!IMPORTANT]
> The **StackingClassifier** achieved the highest generalization capability with an **ROC-AUC of 0.776** and a **Brier score of 0.088**, establishing it as the core production estimator.

### Imbalanced Class Threshold Settings
To recover high-risk downsizing events effectively, a decision threshold search was conducted on the calibrated model probabilities:
* **Default Decision Threshold**: `0.50` (Precision: `0.769`, Recall: `0.175`, F1-Score: `0.286`)
* **Optimized Decision Threshold**: `0.20` (Precision: `0.364`, Recall: `0.509`, F1-Score: **`0.423`**)

By shifting the classification threshold to **0.20**, the platform dramatically increases the recall rate to **50.9%** (capturing over half of actual layoff recurrences) while boosting the model's overall F1-score to **0.423**.

### Temporal Generalization Analysis
Testing the model's performance on chronological (out-of-time) validation data highlights the impact of macroeconomic trends:

| Split Strategy | Test Accuracy | Test Precision | Test Recall | Test F1-Score | Test ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Stratified Split** | **73.5%** | **0.233** | **0.544** | **0.326** | **0.763** |
| **Chronological Time Split** | **78.4%** | **0.195** | **0.152** | **0.171** | **0.622** |

> [!WARNING]
> The drop in ROC-AUC from **0.763** to **0.622** under chronological validation demonstrates that workforce changes are subject to significant temporal shifts. Model performance can decay when encountering structural changes in the macroeconomic environment, emphasizing the need for periodic pipeline retraining.
