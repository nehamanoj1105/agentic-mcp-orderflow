"""
Model Performance Degradation Monitor.

Tracks rolling model performance metrics (Accuracy, F1, IC, Sharpe)
and triggers alerts when performance drops below expected thresholds.
"""

import numpy as np
from typing import Dict, Any
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import spearmanr


class PerformanceMonitor:
    """
    Monitors live/rolling model performance against baseline benchmarks.
    """

    def __init__(
        self,
        min_accuracy_threshold: float = 0.40,
        min_f1_threshold: float = 0.35,
        min_ic_threshold: float = 0.01
    ):
        self.min_accuracy_threshold = min_accuracy_threshold
        self.min_f1_threshold = min_f1_threshold
        self.min_ic_threshold = min_ic_threshold

    def evaluate_performance(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        is_classification: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluates current window performance and flags degradation status.
        """
        if is_classification:
            acc = float(accuracy_score(y_true, y_pred))
            f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
            is_degraded = (acc < self.min_accuracy_threshold) or (f1 < self.min_f1_threshold)

            return {
                "performance_status": "DEGRADED" if is_degraded else "STABLE",
                "accuracy": round(acc, 4),
                "f1_score": round(f1, 4),
                "accuracy_threshold": self.min_accuracy_threshold,
                "f1_threshold": self.min_f1_threshold
            }
        else:
            ic, _ = spearmanr(y_true, y_pred)
            ic_val = float(ic) if not np.isnan(ic) else 0.0
            is_degraded = ic_val < self.min_ic_threshold

            return {
                "performance_status": "DEGRADED" if is_degraded else "STABLE",
                "ic": round(ic_val, 4),
                "ic_threshold": self.min_ic_threshold
            }
