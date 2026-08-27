"""
LightGBM Model Architecture with Fallback Support.

Provides LightGBM Classifier/Regressor wrappers with fallback to GradientBoosting.
"""

import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    logger.warning("LightGBM not found. Using GradientBoosting fallback.")


class LightGBMModel:
    """
    LightGBM Classifier / Regressor wrapper with fallback support.
    """

    def __init__(self, is_classification: bool = True, n_estimators: int = 100, max_depth: int = 5, seed: int = 42):
        self.is_classification = is_classification
        self.seed = seed
        self.scaler = StandardScaler()

        if HAS_LIGHTGBM:
            if self.is_classification:
                self.model = lgb.LGBMClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=0.05,
                    random_state=seed,
                    verbosity=-1,
                    n_jobs=-1
                )
            else:
                self.model = lgb.LGBMRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=0.05,
                    random_state=seed,
                    verbosity=-1,
                    n_jobs=-1
                )
        else:
            if self.is_classification:
                self.model = GradientBoostingClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=seed
                )
            else:
                self.model = GradientBoostingRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=seed
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
