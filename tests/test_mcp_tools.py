"""
Unit Tests for MCP Analytical Server Tools.
"""

import pytest
from mcp_server.server import (
    get_market_data,
    get_orderbook,
    calculate_order_imbalance,
    calculate_ewma,
    calculate_zscore,
    calculate_microstructure_features,
    detect_market_regime,
    predict,
    evaluate_model,
    detect_feature_drift
)


def test_mcp_get_market_data():
    res = get_market_data(n_steps=50)
    assert res["n_steps"] == 50
    assert "latest_mid_price" in res


def test_mcp_orderbook():
    res = get_orderbook(depth_level=3)
    assert len(res["bids"]) == 3
    assert len(res["asks"]) == 3


def test_mcp_calculate_order_imbalance():
    res = calculate_order_imbalance(depth_level=1)
    assert "latest_obi" in res
    assert -1.0 <= res["latest_obi"] <= 1.0


def test_mcp_predict():
    res = predict(model_type="xgboost")
    assert res["predicted_signal"] in ["UP", "DOWN", "NEUTRAL"]
    assert 0.0 <= res["confidence"] <= 1.0


def test_mcp_evaluate_model():
    res = evaluate_model(model_type="xgboost")
    assert "accuracy" in res
    assert "f1_score" in res


def test_mcp_detect_drift():
    res = detect_feature_drift()
    assert res["overall_feature_drift"] in ["NORMAL", "DETECTED"]
