"""
Weighted Meta-Ensemble Model Architecture.

Combines predictions from multiple distinct model families (XGBoost, LightGBM,
RandomForest, Neural Net) using simple averaging, performance-weighted averaging,
or probability ensembling.
"""

import numpy as np
from typing import List, Dict, Optional, Any


class WeightedEnsembleModel:
    """
    Ensemble model combining base predictions with custom/dynamic weights.
    """

    def __init__(
        self,
        base_models: List[Any],
        weights: Optional[List[float]] = None,
        is_classification: bool = True
    ):
        self.base_models = base_models
        self.is_classification = is_classification

        if weights is None:
            # Default to equal weighting
            self.weights = np.ones(len(base_models)) / len(base_models)
        else:
            weights_arr = np.array(weights, dtype=float)
            self.weights = weights_arr / np.sum(weights_arr)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fits all underlying base models.
        """
        for model in self.base_models:
            model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Computes weighted average class probability distribution across base models.
        """
        if not self.is_classification:
            raise ValueError("predict_proba is only available for classification models.")

        prob_list = []
        for model in self.base_models:
            prob_list.append(model.predict_proba(X))

        # Weighted sum of probabilities
        weighted_probs = np.zeros_like(prob_list[0])
        for w, prob in zip(self.weights, prob_list):
            weighted_probs += w * prob

        return weighted_probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generates ensemble predictions.
        """
        if self.is_classification:
            probs = self.predict_proba(X)
            return np.argmax(probs, axis=1)
        else:
            preds_list = [model.predict(X) for model in self.base_models]
            weighted_preds = np.zeros_like(preds_list[0])
            for w, pred in zip(self.weights, preds_list):
                weighted_preds += w * pred
            return weighted_preds
