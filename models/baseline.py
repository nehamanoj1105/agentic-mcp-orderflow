"""
Baseline Machine Learning Models.

Provides Logistic Regression, Ridge Regression, and Random Forest models
for classification and regression targets.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler


class BaselineModel:
    """
    Wrapper for Baseline ML models (Logistic Regression & Random Forest).
    """

    def __init__(self, model_type: str = "rf", is_classification: bool = True, seed: int = 42):
        self.model_type = model_type.lower()
        self.is_classification = is_classification
        self.seed = seed
        self.scaler = StandardScaler()

        if self.is_classification:
            if self.model_type == "logistic":
                self.model = LogisticRegression(max_iter=1000, random_state=seed)
            else:
                self.model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=seed, n_jobs=-1)
        else:
            if self.model_type == "ridge":
                self.model = Ridge(alpha=1.0, random_state=seed)
            else:
                self.model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=seed, n_jobs=-1)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_classification:
            raise ValueError("predict_proba is only available for classification models.")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
