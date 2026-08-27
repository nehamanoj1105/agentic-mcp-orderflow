"""
Market Microstructure Features.

Computes bid-ask spread, relative spread, depth imbalance, microprice deviations,
and volume-weighted average price (VWAP) features.
"""

import numpy as np
import pandas as pd


class MicrostructureFeatures:
    """
    Computes microstructure price and liquidity metrics.
    """

    @staticmethod
    def calculate_microstructure(df: pd.DataFrame) -> pd.DataFrame:
        res = pd.DataFrame(index=df.index)

        b1 = df["bid_price_1"]
        a1 = df["ask_price_1"]
        mid = df["mid_price"]
        bq1 = df["bid_qty_1"]
        aq1 = df["ask_qty_1"]

        # Bid-Ask Spread & Relative Spread
        spread = a1 - b1
        res["bid_ask_spread"] = spread
        res["relative_spread"] = np.where(mid > 0, spread / mid, 0.0)

        # Microprice deviation from mid price
        microprice = np.where((bq1 + aq1) > 0, (b1 * aq1 + a1 * bq1) / (bq1 + aq1), mid)
        res["microprice"] = microprice
        res["microprice_dev"] = microprice - mid

        # Depth ratio and depth imbalance
        total_depth_bids = sum(df[f"bid_qty_{i}"].fillna(0.0) for i in range(1, 11) if f"bid_qty_{i}" in df)
        total_depth_asks = sum(df[f"ask_qty_{i}"].fillna(0.0) for i in range(1, 11) if f"ask_qty_{i}" in df)

        total_depth = total_depth_bids + total_depth_asks
        res["total_depth"] = total_depth
        res["depth_imbalance"] = np.where(total_depth > 0, (total_depth_bids - total_depth_asks) / total_depth, 0.0)

        # VWAP deviation if available
        if "vwap" in df:
            res["vwap_dev"] = df["vwap"] - mid

        return res
