"""
Rolling Statistical Features.

Computes rolling statistics (mean, std, min, max, z-scores) for orderbook
and market signals over multiple window horizons.
"""

import numpy as np
import pandas as pd
from typing import List


class StatisticalFeatures:
    """
    Computes rolling statistical metrics and normalized z-scores.
    """

    @staticmethod
    def calculate_rolling_stats(
        series: pd.Series,
        feature_name: str,
        windows: List[int] = [10, 30, 60]
    ) -> pd.DataFrame:
        res = pd.DataFrame(index=series.index)
        for w in windows:
            roll = series.rolling(window=w, min_periods=1)
            mean = roll.mean()
            std = roll.std().replace(0, 1e-8).fillna(1e-8)
            
            res[f"{feature_name}_roll_mean_{w}"] = mean
            res[f"{feature_name}_roll_std_{w}"] = std
            res[f"{feature_name}_roll_zscore_{w}"] = (series - mean) / std
            res[f"{feature_name}_roll_min_{w}"] = roll.min()
            res[f"{feature_name}_roll_max_{w}"] = roll.max()

        return res
