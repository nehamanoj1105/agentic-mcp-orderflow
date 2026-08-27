"""
Model Context Protocol (MCP) Analytical Tool Server.

Exposes deterministic quantitative, microstructure, model evaluation, and monitoring
tools to AI agentic workflows.
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from fastmcp import FastMCP

# Ensure root workspace is in import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.ingestion.synthetic_gen import SyntheticMarketDataGenerator
from data.preprocessing.orderbook_builder import OrderBookPreprocessor
from features.feature_pipeline import FeaturePipeline
from features.order_flow import OrderFlowFeatures
from features.statistics import StatisticalFeatures
from models.xgboost_model import XGBoostModel
from research.model_comparison import ModelComparator
from research.regime_analysis import MarketRegimeAnalyzer
from monitoring.drift import DriftDetector
from monitoring.performance import PerformanceMonitor

mcp = FastMCP("AgenticMarketMLServer")

# Global cached dataset & pipeline instance for deterministic tool calls
GEN = SyntheticMarketDataGenerator(seed=42)
PREP = OrderBookPreprocessor()
PIPE = FeaturePipeline()
RAW_OB, RAW_TR = GEN.generate_orderbook_and_trades(n_steps=1000)
ALIGNED_DF = PREP.clean_and_align(RAW_OB, RAW_TR)
FEATS_DF, TARGET_COLS = PIPE.create_targets(PIPE.extract_features(ALIGNED_DF))
FEATURE_COLS = [c for c in FEATS_DF.columns if c not in ["timestamp", "mid_price"] + TARGET_COLS]


@mcp.tool()
def get_market_data(n_steps: int = 200) -> Dict[str, Any]:
    """
    Retrieves synchronized L2 market depth and trade snapshot summary.
    """
    sub = ALIGNED_DF.tail(n_steps)
    return {
        "n_steps": len(sub),
        "start_time": str(sub["timestamp"].iloc[0]),
        "end_time": str(sub["timestamp"].iloc[-1]),
        "latest_mid_price": float(sub["mid_price"].iloc[-1]),
        "latest_spread": float(sub["spread"].iloc[-1]),
        "latest_microprice": float(sub["microprice"].iloc[-1])
    }


@mcp.tool()
def get_orderbook(depth_level: int = 5) -> Dict[str, Any]:
    """
    Retrieves current L2 orderbook bids and asks up to depth_level.
    """
    latest = ALIGNED_DF.iloc[-1]
    bids = []
    asks = []
    for i in range(1, depth_level + 1):
        if f"bid_price_{i}" in latest and f"ask_price_{i}" in latest:
            bids.append({"level": i, "price": float(latest[f"bid_price_{i}"]), "qty": float(latest[f"bid_qty_{i}"])})
            asks.append({"level": i, "price": float(latest[f"ask_price_{i}"]), "qty": float(latest[f"ask_qty_{i}"])})
    return {
        "timestamp": str(latest["timestamp"]),
        "mid_price": float(latest["mid_price"]),
        "bids": bids,
        "asks": asks
    }


@mcp.tool()
def calculate_order_imbalance(depth_level: int = 1) -> Dict[str, Any]:
    """
    Calculates Multi-Level Order Book Imbalance (OBI).
    """
    obi_series = OrderFlowFeatures.calculate_obi(ALIGNED_DF, depth_level=depth_level)
    return {
        "depth_level": depth_level,
        "latest_obi": float(obi_series.iloc[-1]),
        "mean_obi": float(obi_series.mean()),
        "std_obi": float(obi_series.std())
    }


@mcp.tool()
def calculate_ewma(alpha: float = 0.05) -> Dict[str, Any]:
    """
    Calculates Exponentially Weighted Moving Average (EWMA) of Order Book Imbalance.
    """
    obi_series = OrderFlowFeatures.calculate_obi(ALIGNED_DF, depth_level=1)
    ewma_df = OrderFlowFeatures.calculate_ewma_imbalance(obi_series, alphas=[alpha])
    val = ewma_df[f"ewma_obi_alpha_{alpha}"].iloc[-1]
    return {
        "alpha": alpha,
        "latest_ewma_obi": float(val)
    }


@mcp.tool()
def calculate_zscore(window: int = 30) -> Dict[str, Any]:
    """
    Calculates rolling Z-Score for Order Book Imbalance over specified window.
    """
    obi_series = OrderFlowFeatures.calculate_obi(ALIGNED_DF, depth_level=1)
    stats = StatisticalFeatures.calculate_rolling_stats(obi_series, "obi", windows=[window])
    zscore_val = stats[f"obi_roll_zscore_{window}"].iloc[-1]
    return {
        "window": window,
        "latest_obi_zscore": float(zscore_val)
    }


@mcp.tool()
def calculate_microstructure_features() -> Dict[str, Any]:
    """
    Calculates current market microstructure features (Microprice dev, relative spread, depth ratio).
    """
    latest = FEATS_DF.iloc[-1]
    return {
        "mid_price": float(latest["mid_price"]),
        "bid_ask_spread": float(latest["bid_ask_spread"]),
        "relative_spread": float(latest["relative_spread"]),
        "microprice_dev": float(latest["microprice_dev"]),
        "depth_imbalance": float(latest["depth_imbalance"])
    }


@mcp.tool()
def detect_market_regime() -> Dict[str, Any]:
    """
    Clusters current market state into volatility/imbalance market regimes.
    """
    analyzer = MarketRegimeAnalyzer(n_regimes=3)
    regimes = analyzer.fit_predict_regimes(FEATS_DF, regime_features=["realized_vol_30", "obi_lvl_1"])
    current_regime = int(regimes[-1])
    regime_names = {0: "Low Volatility / Normal", 1: "High Volatility / Shock", 2: "Directional Imbalance"}
    return {
        "current_regime_id": current_regime,
        "regime_name": regime_names.get(current_regime, f"Regime {current_regime}"),
        "total_observed_regimes": len(set(regimes))
    }


@mcp.tool()
def predict(model_type: str = "xgboost") -> Dict[str, Any]:
    """
    Runs model inference on latest market feature vector.
    """
    X = FEATS_DF[FEATURE_COLS].values
    y = FEATS_DF["target_class_5"].values
    
    model = XGBoostModel(is_classification=True)
    model.fit(X[:-50], y[:-50])
    
    latest_X = X[-1:]
    pred_class = int(model.predict(latest_X)[0])
    probs = model.predict_proba(latest_X)[0].tolist()

    class_names = {0: "DOWN", 1: "NEUTRAL", 2: "UP"}
    return {
        "model_type": model_type,
        "predicted_signal": class_names.get(pred_class, str(pred_class)),
        "confidence": float(np.max(probs)),
        "class_probabilities": {"DOWN": float(probs[0]), "NEUTRAL": float(probs[1]), "UP": float(probs[2])}
    }


@mcp.tool()
def evaluate_model(model_type: str = "xgboost") -> Dict[str, Any]:
    """
    Evaluates specified model on out-of-sample test split.
    """
    X = FEATS_DF[FEATURE_COLS].values
    y = FEATS_DF["target_class_5"].values
    
    split = int(len(X) * 0.8)
    X_tr, y_tr = X[:split], y[:split]
    X_te, y_te = X[split:], y[split:]

    model = XGBoostModel(is_classification=True)
    model.fit(X_tr, y_tr)

    comparator = ModelComparator()
    metrics = comparator.evaluate_classification(model, X_te, y_te)
    metrics["model_type"] = model_type
    return metrics


@mcp.tool()
def compare_models() -> Dict[str, Any]:
    """
    Compares baseline, tree, and ensemble models side-by-side on test data.
    """
    X = FEATS_DF[FEATURE_COLS].values
    y = FEATS_DF["target_class_5"].values

    split = int(len(X) * 0.8)
    X_tr, y_tr = X[:split], y[:split]
    X_te, y_te = X[split:], y[split:]

    from models.baseline import BaselineModel
    from models.lightgbm_model import LightGBMModel
    from models.neural_model import NeuralNetworkModel

    models = {
        "RandomForest": BaselineModel(model_type="rf"),
        "XGBoost": XGBoostModel(),
        "LightGBM": LightGBMModel(),
        "NeuralNet": NeuralNetworkModel()
    }

    comparator = ModelComparator()
    df_res = comparator.compare_models(models, X_tr, y_tr, X_te, y_te, is_classification=True)
    return {"comparison": df_res.to_dict(orient="records")}


@mcp.tool()
def detect_feature_drift() -> Dict[str, Any]:
    """
    Audits feature distribution drift between reference window and recent production window.
    """
    split = int(len(FEATS_DF) * 0.5)
    ref_df = FEATS_DF.iloc[:split]
    cur_df = FEATS_DF.iloc[split:]

    detector = DriftDetector()
    return detector.detect_feature_drift(ref_df, cur_df, feature_cols=["obi_lvl_1", "realized_vol_30"])


@mcp.tool()
def detect_prediction_drift() -> Dict[str, Any]:
    """
    Detects prediction probability distribution drift.
    """
    X = FEATS_DF[FEATURE_COLS].values
    y = FEATS_DF["target_class_5"].values

    split = int(len(X) * 0.5)
    model = XGBoostModel()
    model.fit(X[:split], y[:split])

    ref_probs = model.predict_proba(X[:split])[:, 2]
    cur_probs = model.predict_proba(X[split:])[:, 2]

    detector = DriftDetector()
    return detector.detect_prediction_drift(ref_probs, cur_probs)


@mcp.tool()
def get_model_performance() -> Dict[str, Any]:
    """
    Retrieves current model health and performance degradation metrics.
    """
    X = FEATS_DF[FEATURE_COLS].values
    y = FEATS_DF["target_class_5"].values

    split = int(len(X) * 0.8)
    model = XGBoostModel()
    model.fit(X[:split], y[:split])
    y_pred = model.predict(X[split:])

    monitor = PerformanceMonitor()
    return monitor.evaluate_performance(y[split:], y_pred, is_classification=True)


if __name__ == "__main__":
    print("Starting FastMCP Market Microstructure Server...")
    mcp.run()
