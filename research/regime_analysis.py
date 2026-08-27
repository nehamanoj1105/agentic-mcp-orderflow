"""
Market Regime Clustering & Regime-Sliced Model Performance Analysis.

Uses Gaussian Mixture Models (GMM) / K-Means clustering on volatility and imbalance
features to classify market regimes (Low Vol vs High Vol, High Buy vs High Sell).
Evaluates model performance degradation across distinct market environments.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from .model_comparison import ModelComparator


class MarketRegimeAnalyzer:
    """
    Identifies market regimes and calculates sliced performance metrics per regime.
    """

    def __init__(self, n_regimes: int = 3, seed: int = 42):
        self.n_regimes = n_regimes
        self.seed = seed
        self.scaler = StandardScaler()
        self.gmm = GaussianMixture(n_components=n_regimes, random_state=seed)
        self.comparator = ModelComparator()

    def fit_predict_regimes(self, df: pd.DataFrame, regime_features: List[str]) -> np.ndarray:
        """
        Fits unsupervised GMM clustering on specified volatility and imbalance features.
        Returns regime cluster label assignment (0, 1, 2, ...).
        """
        X_reg = df[regime_features].fillna(0.0).values
        X_scaled = self.scaler.fit_transform(X_reg)
        regime_labels = self.gmm.fit_predict(X_scaled)
        return regime_labels

    def evaluate_by_regime(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        regime_labels: np.ndarray,
        is_classification: bool = True
    ) -> pd.DataFrame:
        """
        Evaluates model metrics sliced separately for each identified market regime.
        """
        records = []
        unique_regimes = np.unique(regime_labels)

        for reg in unique_regimes:
            idx = np.where(regime_labels == reg)[0]
            if len(idx) < 5:
                continue

            X_sub = X_test[idx]
            y_sub = y_test[idx]

            if is_classification:
                metrics = self.comparator.evaluate_classification(model, X_sub, y_sub)
            else:
                metrics = self.comparator.evaluate_regression(model, X_sub, y_sub)

            metrics["regime"] = int(reg)
            metrics["sample_count"] = len(idx)
            records.append(metrics)

        res_df = pd.DataFrame(records)
        cols = ["regime", "sample_count"] + [c for c in res_df.columns if c not in ["regime", "sample_count", "confusion_matrix"]]
        return res_df[cols]
