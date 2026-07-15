"""
ml_config.py — Central configuration module for the RiskPulse ML pipeline.

All tuneable parameters, paths, feature lists, and logging setup are
consolidated here so that no module contains hard-coded magic values.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List
import logging
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def get_logger(name: str = "RiskPulseML") -> logging.Logger:
    """Returns a shared, production-grade logger with console + file handlers.

    Args:
        name: Logger namespace (default ``RiskPulseML``).

    Returns:
        A configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console handler (force UTF-8 on Windows to avoid charmap errors)
        console = logging.StreamHandler(
            stream=open(sys.stdout.fileno(), mode="w", encoding="utf-8", closefd=False)
        )
        console.setFormatter(formatter)
        logger.addHandler(console)

        # File handler — rotates into ml/outputs/logs/
        log_dir = PROJECT_ROOT / "ml" / "outputs" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            log_dir / "pipeline.log", encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


@dataclass
class MLConfig:
    """Immutable configuration for the entire ML training & inference pipeline."""

    # Random State for Reproducibility
    RANDOM_SEED: int = 42
    
    # Dataset Versioning
    DATASET_VERSION: str = "enhanced_v1.0"
    
    # Paths
    ML_DIR: Path = PROJECT_ROOT / "ml"
    DATA_DIR: Path = PROJECT_ROOT / "ml" / "data"
    MODEL_DIR: Path = PROJECT_ROOT / "ml" / "models"
    OUTPUT_DIR: Path = PROJECT_ROOT / "ml" / "outputs"
    SHAP_DIR: Path = PROJECT_ROOT / "ml" / "outputs" / "shap"
    LOG_DIR: Path = PROJECT_ROOT / "ml" / "outputs" / "logs"
    
    # Target Column
    TARGET_COL: str = "future_layoff"
    
    # Ignored Features during Model Training
    IGNORE_COLS: List[str] = field(default_factory=lambda: [
        "company_id", "name", "layoff_date", "previous_layoff_date", 
        "next_layoff_date", "days_until_next_layoff"
    ])
    
    # Feature Classifications
    CATEGORICAL_COLS: List[str] = field(default_factory=lambda: [
        "industry", "country", "company_type"
    ])
    
    # Numeric Columns will be inferred as:
    # All columns in the dataset except TARGET_COL, CATEGORICAL_COLS, and IGNORE_COLS
    
    # Cross Validation Settings
    CV_FOLDS: int = 5
    VAL_STRATEGY: str = "stratified_cv"  # "stratified_cv" or "time_split"
    
    # Hyperparameter Optimization (Optuna)
    OPTUNA_TRIALS: int = 100
    HPO_TIMEOUT: int = 600  # Timeout in seconds
    
    # Threshold Optimization Search Range
    THRESHOLD_MIN: float = 0.05
    THRESHOLD_MAX: float = 0.95
    THRESHOLD_STEP: float = 0.01
    
    # Model Benchmarking List
    BENCHMARK_MODELS: List[str] = field(default_factory=lambda: [
        "LogisticRegression", "RandomForest", "ExtraTrees", 
        "HistGradientBoosting", "LightGBM", "XGBoost"
    ])
    
    # Probability Calibration Methods
    CALIBRATION_METHODS: List[str] = field(default_factory=lambda: [
        "platt", "isotonic"
    ])
    
    def __post_init__(self) -> None:
        """Create output directories if they do not exist."""
        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.SHAP_DIR.mkdir(parents=True, exist_ok=True)
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)

