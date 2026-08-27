"""
Unit Tests for Market Microstructure Feature Calculations.
"""

import pytest
import numpy as np
import pandas as pd

from data.ingestion.synthetic_gen import SyntheticMarketDataGenerator
from data.preprocessing.orderbook_builder import OrderBookPreprocessor
from features.order_flow import OrderFlowFeatures
from features.microstructure import MicrostructureFeatures
from features.volatility import VolatilityFeatures
from features.statistics import StatisticalFeatures
from features.feature_pipeline import FeaturePipeline


def test_order_book_imbalance():
    df = pd.DataFrame({
        "bid_qty_1": [10.0, 20.0],
        "ask_qty_1": [5.0, 20.0]
    })
    obi_1 = OrderFlowFeatures.calculate_obi(df, depth_level=1)
    assert len(obi_1) == 2
    assert pytest.approx(obi_1.iloc[0], 0.001) == (10.0 - 5.0) / 15.0
    assert pytest.approx(obi_1.iloc[1], 0.001) == 0.0


def test_ewma_imbalance():
    s = pd.Series([0.5, -0.2, 0.8, 0.1])
    ewma_df = OrderFlowFeatures.calculate_ewma_imbalance(s, alphas=[0.1, 0.5])
    assert "ewma_obi_alpha_0.1" in ewma_df.columns
    assert "ewma_obi_alpha_0.5" in ewma_df.columns
    assert len(ewma_df) == 4


def test_microprice_and_spread():
    df = pd.DataFrame({
        "bid_price_1": [100.0],
        "ask_price_1": [101.0],
        "mid_price": [100.5],
        "bid_qty_1": [10.0],
        "ask_qty_1": [30.0]
    })
    micro_df = MicrostructureFeatures.calculate_microstructure(df)
    assert pytest.approx(micro_df["bid_ask_spread"].iloc[0], 0.001) == 1.0
    # Microprice weighted towards bid when ask quantity is larger
    expected_micro = (100.0 * 30.0 + 101.0 * 10.0) / 40.0
    assert pytest.approx(micro_df["microprice"].iloc[0], 0.001) == expected_micro


def test_feature_pipeline_integration():
    gen = SyntheticMarketDataGenerator(seed=42)
    ob, tr = gen.generate_orderbook_and_trades(n_steps=200)
    prep = OrderBookPreprocessor()
    aligned = prep.clean_and_align(ob, tr)

    pipe = FeaturePipeline()
    feats = pipe.extract_features(aligned)
    assert "obi_lvl_1" in feats.columns
    assert "microprice_dev" in feats.columns
    assert "realized_vol_30" in feats.columns
