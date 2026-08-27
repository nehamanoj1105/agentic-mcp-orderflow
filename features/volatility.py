"""
Volatility Feature Engineering.

Computes realized volatility over multiple lookback windows, log returns, momentum,
and High-Low price range metrics.
"""

import numpy as np
import pandas as pd
from typing import List


class VolatilityFeatures:
    """
    Computes market volatility metrics.
    """

    @staticmethod
    def calculate_volatility(df: pd.DataFrame, windows: List[int] = [10, 30, 60]) -> pd.DataFrame:
        res = pd.DataFrame(index=df.index)
        mid = df["mid_price"]

        # Log returns
        log_ret = np.log(mid / mid.shift(1)).fillna(0.0)
        res["log_return_1"] = log_ret

        # Realized volatility over rolling windows
        for w in windows:
            res[f"realized_vol_{w}"] = log_ret.rolling(window=w, min_periods=1).std().fillna(0.0)
            res[f"momentum_{w}"] = np.log(mid / mid.shift(w)).fillna(0.0)

        # High-low range proxy if level 1 is available
        if "ask_price_1" in df and "bid_price_1" in df:
            high_proxy = df["ask_price_1"]
            low_proxy = df["bid_price_1"]
            res["high_low_range"] = (high_proxy - low_proxy) / mid

        return res
