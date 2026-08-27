"""
Unified Feature Engineering & Target Labelling Pipeline.

Extracts all quantitative order flow, microstructure, volatility, and rolling statistical features.
Generates multi-horizon prediction targets with strict look-ahead leakage prevention.

LEAKAGE PREVENTION GUARANTEE:
----------------------------
1. Features are computed strictly using backward-looking windows (e.g., rolling means, EWMA).
2. Targets (future log returns r_{t+h}) are generated using forward shift `shift(-horizon)`.
3. Trailing horizon samples with NaN target values are explicitly dropped before model training.
4. Predictor variables X_t contain ZERO future price or volume information.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any

from .order_flow import OrderFlowFeatures
from .microstructure import MicrostructureFeatures
from .volatility import VolatilityFeatures
from .statistics import StatisticalFeatures


class FeaturePipeline:
    """
    Orchestrates market microstructure feature extraction and target labelling.
    """

    def __init__(self, target_horizons: List[int] = [1, 5, 10, 30, 60], neutral_threshold_std: float = 0.5):
        self.target_horizons = target_horizons
        self.neutral_threshold_std = neutral_threshold_std

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts all market microstructure features from raw aligned orderbook dataframe.
        """
        features_df = pd.DataFrame(index=df.index)
        features_df["mid_price"] = df["mid_price"]
        features_df["timestamp"] = df["timestamp"]

        # 1. Order Flow Features (Multi-level OBI)
        for lvl in [1, 3, 5, 10]:
            obi_s = OrderFlowFeatures.calculate_obi(df, depth_level=lvl)
            features_df[f"obi_lvl_{lvl}"] = obi_s

        # EWMA Imbalance
        obi_1 = features_df["obi_lvl_1"]
        ewma_df = OrderFlowFeatures.calculate_ewma_imbalance(obi_1, alphas=[0.01, 0.05, 0.1, 0.2])
        features_df = pd.concat([features_df, ewma_df], axis=1)

        # Trade features
        trade_df = OrderFlowFeatures.calculate_trade_features(df)
        features_df = pd.concat([features_df, trade_df], axis=1)

        # 2. Microstructure & Liquidity Features
        micro_df = MicrostructureFeatures.calculate_microstructure(df)
        features_df = pd.concat([features_df, micro_df], axis=1)

        # 3. Volatility Features
        vol_df = VolatilityFeatures.calculate_volatility(df, windows=[10, 30, 60])
        features_df = pd.concat([features_df, vol_df], axis=1)

        # 4. Rolling Statistical Features (z-scores on OBI & Microprice Dev)
        stats_obi = StatisticalFeatures.calculate_rolling_stats(obi_1, "obi_1", windows=[10, 30, 60])
        stats_micro = StatisticalFeatures.calculate_rolling_stats(features_df["microprice_dev"], "micro_dev", windows=[10, 30, 60])
        features_df = pd.concat([features_df, stats_obi, stats_micro], axis=1)

        return features_df

    def create_targets(self, features_df: pd.DataFrame, primary_horizon: int = 5) -> Tuple[pd.DataFrame, List[str]]:
        """
        Generates multi-horizon future return regression and 3-class classification targets.

        Target definitions:
        - target_return_h: log(mid_price[t+h] / mid_price[t])
        - target_class_h: 0 (DOWN), 1 (NEUTRAL), 2 (UP)
        """
        df = features_df.copy()
        target_cols = []

        mid = df["mid_price"]

        for h in self.target_horizons:
            # Forward shift to look into future t+h
            future_mid = mid.shift(-h)
            fut_return = np.log(future_mid / mid)
            ret_col = f"target_return_{h}"
            df[ret_col] = fut_return
            target_cols.append(ret_col)

            # Dynamic volatility threshold for classification
            ret_std = fut_return.std()
            thresh = max(1e-5, self.neutral_threshold_std * ret_std)

            # Class mapping: 0=DOWN, 1=NEUTRAL, 2=UP
            class_col = f"target_class_{h}"
            conditions = [
                fut_return < -thresh,
                (fut_return >= -thresh) & (fut_return <= thresh),
                fut_return > thresh
            ]
            choices = [0, 1, 2] # 0: DOWN, 1: NEUTRAL, 2: UP
            df[class_col] = np.select(conditions, choices, default=1)
            target_cols.append(class_col)

        # Drop trailing rows with missing targets to prevent leakage & NaNs
        max_h = max(self.target_horizons)
        cleaned_df = df.iloc[:-max_h].dropna().reset_index(drop=True)

        return cleaned_df, target_cols


if __name__ == "__main__":
    from data.ingestion.synthetic_gen import SyntheticMarketDataGenerator
    from data.preprocessing.orderbook_builder import OrderBookPreprocessor

    gen = SyntheticMarketDataGenerator()
    ob, tr = gen.generate_orderbook_and_trades(n_steps=500)
    prep = OrderBookPreprocessor()
    aligned = prep.clean_and_align(ob, tr)

    pipe = FeaturePipeline()
    feats = pipe.extract_features(aligned)
    full_dataset, target_cols = pipe.create_targets(feats)
    print(f"Extracted {len(feats.columns)} features and generated target columns: {target_cols}")
    print(f"Dataset shape after leakage-safe cleaning: {full_dataset.shape}")
