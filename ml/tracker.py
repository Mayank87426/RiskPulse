"""tracker.py -- ML experiment tracking and persistent run logging."""

import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import pandas as pd
from ml_config import MLConfig, get_logger

logger = get_logger("Tracker")

class ExperimentTracker:
    def __init__(self, config: MLConfig):
        self.config = config
        self.json_path = config.OUTPUT_DIR / "experiment_logs.json"
        self.csv_path = config.OUTPUT_DIR / "experiment_logs.csv"
        
    def log_run(self, model_name: str, parameters: Dict[str, Any], metrics: Dict[str, float], additional_meta: Dict[str, Any] = None) -> None:
        """Logs model run details, parameters, and metrics to persistent files."""
        run_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        run_data = {
            "model_name": model_name,
            "dataset_version": self.config.DATASET_VERSION,
            "run_date": run_date,
            "validation_strategy": self.config.VAL_STRATEGY,
            "parameters": parameters,
            "metrics": metrics,
            **(additional_meta or {})
        }
        
        # 1. Update JSON logs
        logs = []
        if self.json_path.exists():
            try:
                with open(self.json_path, "r") as f:
                    logs = json.load(f)
            except Exception:
                logs = []
        logs.append(run_data)
        with open(self.json_path, "w") as f:
            json.dump(logs, f, indent=4)
            
        # 2. Update CSV logs
        # Flatten dictionary for tabular logging
        flat_data = {
            "model_name": model_name,
            "dataset_version": self.config.DATASET_VERSION,
            "run_date": run_date,
            "validation_strategy": self.config.VAL_STRATEGY,
            **{f"param_{k}": v for k, v in parameters.items()},
            **{f"metric_{k}": v for k, v in metrics.items()},
            **(additional_meta or {})
        }
        
        # Convert any list/dict values to string for CSV
        for k, v in flat_data.items():
            if isinstance(v, (dict, list)):
                flat_data[k] = str(v)
                
        file_exists = self.csv_path.exists()
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(flat_data.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(flat_data)
            
        logger.info("Logged experiment metrics for '%s' successfully to %s", model_name, self.json_path)
