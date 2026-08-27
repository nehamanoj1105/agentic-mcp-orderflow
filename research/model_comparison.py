"""
Model Benchmark & Side-by-Side Comparison Engine.

Evaluates multiple model families on identical out-of-sample data splits.
Calculates Accuracy, Precision, Recall, F1, ROC-AUC, Log Loss, MAE, RMSE, R2, and IC.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, mean_absolute_error, mean_squared_error, r2_score, confusion_matrix
)
from scipy.stats import spearmanr


class ModelComparator:
    """
    Evaluates and compares performance across diverse model architectures.
    """

    @staticmethod
    def evaluate_classification(
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """
        Calculates full suite of classification metrics.
        """
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y_test, y_pred)

        auc_score = 0.5
        loss_val = 1.0
        try:
            probs = model.predict_proba(X_test)
            loss_val = log_loss(y_test, probs)
            if len(np.unique(y_test)) > 2:
                auc_score = roc_auc_score(y_test, probs, multi_class="ovr")
            else:
                auc_score = roc_auc_score(y_test, probs[:, 1])
        except Exception:
            pass

        return {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "roc_auc": float(auc_score),
            "log_loss": float(loss_val),
            "confusion_matrix": cm.tolist()
        }

    @staticmethod
    def evaluate_regression(
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculates full suite of regression metrics.
        """
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        ic, p_val = spearmanr(y_pred, y_test)
        if np.isnan(ic):
            ic = 0.0

        return {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
            "ic": float(ic),
            "ic_p_val": float(p_val)
        }

    def compare_models(
        self,
        models_dict: Dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        is_classification: bool = True
    ) -> pd.DataFrame:
        """
        Trains and evaluates all provided models on the specified dataset split.
        """
        records = []
        for name, model in models_dict.items():
            model.fit(X_train, y_train)

            if is_classification:
                metrics = self.evaluate_classification(model, X_test, y_test)
            else:
                metrics = self.evaluate_regression(model, X_test, y_test)

            metrics["model_name"] = name
            records.append(metrics)

        res_df = pd.DataFrame(records)
        cols = ["model_name"] + [c for c in res_df.columns if c not in ["model_name", "confusion_matrix"]]
        return res_df[cols]
