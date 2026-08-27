"""
Unit Tests for Target Generation & Look-Ahead Leakage Prevention.
"""

import pytest
import numpy as np
import pandas as pd

from features.feature_pipeline import FeaturePipeline


def test_target_generation_no_leakage():
    prices = np.array([100.0, 102.0, 101.0, 105.0, 104.0, 108.0, 107.0, 110.0])
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-08-27", periods=len(prices), freq="1s"),
        "mid_price": prices,
        "bid_price_1": prices - 0.5,
        "ask_price_1": prices + 0.5,
        "bid_qty_1": np.ones(len(prices)) * 10.0,
        "ask_qty_1": np.ones(len(prices)) * 10.0
    })

    pipe = FeaturePipeline(target_horizons=[1, 2])
    feats = pipe.extract_features(df)
    clean_df, target_cols = pipe.create_targets(feats)

    # Verify no NaN values remain in clean_df
    assert clean_df[target_cols].isna().sum().sum() == 0

    # Check forward log return logic for horizon 1
    # row 0 target should be log(102.0 / 100.0)
    expected_ret_0 = np.log(102.0 / 100.0)
    assert pytest.approx(clean_df["target_return_1"].iloc[0], 1e-4) == expected_ret_0

    # Verify classification targets are within {0, 1, 2}
    for col in target_cols:
        if "class" in col:
            unique_vals = clean_df[col].unique()
            assert set(unique_vals).issubset({0, 1, 2})
