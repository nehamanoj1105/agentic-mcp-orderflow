"""
Walk-Forward Time-Series Validation Framework.

Implements chronological expanding and rolling window walk-forward validation
to evaluate out-of-sample stability without look-ahead leakage.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Type
from .model_comparison import ModelComparator


class WalkForwardEvaluator:
    """
    Executes walk-forward time-series cross-validation.
    """

    def __init__(
        self,
        n_splits: int = 5,
        train_ratio: float = 0.6,
        expanding: bool = True
    ):
        self.n_splits = n_splits
        self.train_ratio = train_ratio
        self.expanding = expanding
        self.comparator = ModelComparator()

    def generate_folds(self, n_samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Generates chronological fold index splits.
        """
        folds = []
        init_train_size = int(n_samples * self.train_ratio)
        test_size = (n_samples - init_train_size) // self.n_splits

        for k in range(self.n_splits):
            if self.expanding:
                train_start = 0
            else:
                train_start = k * test_size

            train_end = init_train_size + k * test_size
            test_start = train_end
            test_end = test_start + test_size if k < self.n_splits - 1 else n_samples

            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, test_end)
            folds.append((train_idx, test_idx))

        return folds

    def evaluate_walk_forward(
        self,
        model_cls: Any,
        model_kwargs: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        is_classification: bool = True
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Runs walk-forward evaluation across folds. Returns fold metrics summary,
        out-of-sample predictions array, and ground truth target array.
        """
        folds = self.generate_folds(len(X))
        fold_results = []

        oof_preds = []
        oof_targets = []

        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_te, y_te = X[test_idx], y[test_idx]

            model = model_cls(**model_kwargs)
            model.fit(X_tr, y_tr)

            if is_classification:
                metrics = self.comparator.evaluate_classification(model, X_te, y_te)
                preds = model.predict(X_te)
            else:
                metrics = self.comparator.evaluate_regression(model, X_te, y_te)
                preds = model.predict(X_te)

            metrics["fold"] = fold_idx + 1
            metrics["train_size"] = len(train_idx)
            metrics["test_size"] = len(test_idx)
            fold_results.append(metrics)

            oof_preds.extend(preds)
            oof_targets.extend(y_te)

        fold_df = pd.DataFrame(fold_results)
        return fold_df, np.array(oof_preds), np.array(oof_targets)
