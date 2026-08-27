"""
Order Book Alignment & Time-Bar Preprocessor.

Aligns tick-level orderbook snapshots and trade execution logs into uniform
time bars (e.g., 100ms or 1s intervals). Performs quote validation and gap handling.
"""

import pandas as pd
import numpy as np
from typing import Tuple


class OrderBookPreprocessor:
    """
    Cleans, validates, and aligns orderbook and trade data into synchronized time series.
    """

    @staticmethod
    def compute_microprice(bid_price_1: float, bid_qty_1: float, ask_price_1: float, ask_qty_1: float) -> float:
        """
        Computes the volume-weighted mid price (Microprice):
        P_micro = (BidPrice * AskQty + AskPrice * BidQty) / (BidQty + AskQty)
        """
        denom = bid_qty_1 + ask_qty_1
        if denom == 0:
            return (bid_price_1 + ask_price_1) / 2.0
        return (bid_price_1 * ask_qty_1 + ask_price_1 * bid_qty_1) / denom

    def clean_and_align(
        self,
        orderbook_df: pd.DataFrame,
        trades_df: pd.DataFrame,
        bar_interval: str = "100ms"
    ) -> pd.DataFrame:
        """
        Cleans data and aligns orderbook snapshots with trade aggregations.
        """
        ob = orderbook_df.copy()
        tr = trades_df.copy()

        # Ensure datetime indexing
        ob["timestamp"] = pd.to_datetime(ob["timestamp"])
        ob = ob.sort_values("timestamp").reset_index(drop=True)

        # Validate orderbook quotes
        invalid_quotes = (ob["bid_price_1"] >= ob["ask_price_1"]) | (ob["bid_qty_1"] <= 0) | (ob["ask_qty_1"] <= 0)
        if invalid_quotes.any():
            ob = ob[~invalid_quotes].reset_index(drop=True)

        # Compute Microprice
        ob["microprice"] = ob.apply(
            lambda r: self.compute_microprice(r["bid_price_1"], r["bid_qty_1"], r["ask_price_1"], r["ask_qty_1"]),
            axis=1
        )

        # Process trades if available
        if not tr.empty:
            tr["timestamp"] = pd.to_datetime(tr["timestamp"])
            tr["buy_volume"] = np.where(tr["side"] == "BUY", tr["volume"], 0.0)
            tr["sell_volume"] = np.where(tr["side"] == "SELL", tr["volume"], 0.0)

            # Aggregate trades into time bars
            tr_agg = tr.set_index("timestamp").resample(bar_interval).agg({
                "volume": "sum",
                "buy_volume": "sum",
                "sell_volume": "sum",
                "price": ["count", "mean"]
            })
            tr_agg.columns = ["trade_volume", "buy_volume", "sell_volume", "trade_count", "vwap"]
            tr_agg = tr_agg.reset_index()

            # Merge with orderbook on closest timestamp
            merged = pd.merge_asof(
                ob,
                tr_agg,
                on="timestamp",
                direction="backward"
            )
            merged["trade_volume"] = merged["trade_volume"].fillna(0.0)
            merged["buy_volume"] = merged["buy_volume"].fillna(0.0)
            merged["sell_volume"] = merged["sell_volume"].fillna(0.0)
            merged["trade_count"] = merged["trade_count"].fillna(0.0)
            merged["vwap"] = merged["vwap"].fillna(merged["mid_price"])
        else:
            merged = ob.copy()
            merged["trade_volume"] = 0.0
            merged["buy_volume"] = 0.0
            merged["sell_volume"] = 0.0
            merged["trade_count"] = 0.0
            merged["vwap"] = merged["mid_price"]

        return merged
