"""
XGBoost Model Architecture.

Provides XGBoost Classifier and Regressor wrappers with hyperparameter options
and feature importance extraction.
"""

import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from typing import Optional


class XGBoostModel:
    """
    XGBoost Classifier / Regressor wrapper.
    """

    def __init__(self, is_classification: bool = True, n_estimators: int = 100, max_depth: int = 5, seed: int = 42):
        self.is_classification = is_classification
        self.seed = seed
        self.scaler = StandardScaler()

        if self.is_classification:
            self.model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=0.05,
                random_state=seed,
                eval_metric="mlogloss",
                n_jobs=-1
            )
        else:
            self.model = xgb.XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=0.05,
                random_state=seed,
                eval_metric="rmse",
                n_jobs=-1
            )

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

    def get_feature_importances(self) -> np.ndarray:
        return self.model.feature_importances_
