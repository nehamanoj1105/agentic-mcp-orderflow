"""
End-to-End System Integration Test.

Validates end-to-end pipeline execution:
Market Data -> Features -> Model Training -> Real-Time Inference -> Monitoring -> Agentic Orchestrator
"""

import pytest

from data.ingestion.synthetic_gen import SyntheticMarketDataGenerator
from data.preprocessing.orderbook_builder import OrderBookPreprocessor
from features.feature_pipeline import FeaturePipeline
from models.xgboost_model import XGBoostModel
from monitoring.drift import DriftDetector
from agents.orchestrator import AgentOrchestrator


def test_full_system_pipeline_integration():
    # 1. Ingest synthetic data
    gen = SyntheticMarketDataGenerator(seed=42)
    ob, tr = gen.generate_orderbook_and_trades(n_steps=300)
    assert len(ob) == 300

    # 2. Align orderbook & trades
    prep = OrderBookPreprocessor()
    aligned = prep.clean_and_align(ob, tr)
    assert len(aligned) > 0

    # 3. Extract features & targets
    pipe = FeaturePipeline()
    feats = pipe.extract_features(aligned)
    df, target_cols = pipe.create_targets(feats)
    feature_cols = [c for c in df.columns if c not in ["timestamp", "mid_price"] + target_cols]

    # 4. Train model & infer
    X = df[feature_cols].values
    y = df["target_class_5"].values
    model = XGBoostModel()
    model.fit(X[:-20], y[:-20])
    preds = model.predict(X[-20:])
    assert len(preds) == 20

    # 5. Drift audit
    detector = DriftDetector()
    drift = detector.detect_feature_drift(df.iloc[:100], df.iloc[100:200], feature_cols=["obi_lvl_1"])
    assert "overall_feature_drift" in drift

    # 6. Agentic Orchestration query
    orchestrator = AgentOrchestrator()
    res = orchestrator.process_query("Is the current BTC order flow unusual and does our model still perform reliably?")
    assert "workflow" in res
    assert res["workflow"] == "market_and_model_reliability_audit"
    assert "response" in res
