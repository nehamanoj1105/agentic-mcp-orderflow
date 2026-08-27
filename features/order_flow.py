"""
Order Flow Feature Engineering.

Computes Multi-Level Order Book Imbalance (OBI), EWMA Imbalances across decay rates,
Trade Imbalance, and Trade Intensity.
"""

import numpy as np
import pandas as pd
from typing import List


class OrderFlowFeatures:
    """
    Computes quantitative order flow features.
    """

    @staticmethod
    def calculate_obi(df: pd.DataFrame, depth_level: int = 1) -> pd.Series:
        """
        Calculates Order Book Imbalance at level k:
        OBI_k = (sum_{i=1}^k BidQty_i - sum_{i=1}^k AskQty_i) / (sum_{i=1}^k BidQty_i + sum_{i=1}^k AskQty_i)
        """
        bid_vols = np.zeros(len(df))
        ask_vols = np.zeros(len(df))

        for lvl in range(1, depth_level + 1):
            if f"bid_qty_{lvl}" in df and f"ask_qty_{lvl}" in df:
                bid_vols += df[f"bid_qty_{lvl}"].fillna(0.0).values
                ask_vols += df[f"ask_qty_{lvl}"].fillna(0.0).values

        total_vol = bid_vols + ask_vols
        obi = np.where(total_vol > 0, (bid_vols - ask_vols) / total_vol, 0.0)
        return pd.Series(obi, index=df.index, name=f"obi_lvl_{depth_level}")

    @staticmethod
    def calculate_ewma_imbalance(obi_series: pd.Series, alphas: List[float] = [0.01, 0.05, 0.1, 0.2]) -> pd.DataFrame:
        """
        Computes Exponentially Weighted Moving Average (EWMA) of imbalance:
        EWMA_t = alpha * x_t + (1 - alpha) * EWMA_{t-1}
        """
        ewma_df = pd.DataFrame(index=obi_series.index)
        for alpha in alphas:
            ewma_df[f"ewma_obi_alpha_{alpha}"] = obi_series.ewm(alpha=alpha, adjust=False).mean()
        return ewma_df

    @staticmethod
    def calculate_trade_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates trade side imbalance, buy/sell volumes, and trade intensity.
        """
        res = pd.DataFrame(index=df.index)
        buy_vol = df.get("buy_volume", pd.Series(0.0, index=df.index)).fillna(0.0)
        sell_vol = df.get("sell_volume", pd.Series(0.0, index=df.index)).fillna(0.0)
        total_trade_vol = buy_vol + sell_vol

        res["trade_volume_total"] = total_trade_vol
        res["trade_imbalance"] = np.where(
            total_trade_vol > 0,
            (buy_vol - sell_vol) / total_trade_vol,
            0.0
        )
        res["trade_count"] = df.get("trade_count", pd.Series(0.0, index=df.index)).fillna(0.0)
        res["avg_trade_size"] = np.where(
            res["trade_count"] > 0,
            total_trade_vol / np.maximum(1.0, res["trade_count"]),
            0.0
        )
        return res
