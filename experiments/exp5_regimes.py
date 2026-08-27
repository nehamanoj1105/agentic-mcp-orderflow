"""
Experiment 5: Model Robustness Across Market Regimes.

Evaluates how model predictive performance degrades or shifts when evaluated
across different market volatility and order flow imbalance regimes.
"""

import os
import json
import numpy as np
import pandas as pd
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.ingestion.synthetic_gen import SyntheticMarketDataGenerator
from data.preprocessing.orderbook_builder import OrderBookPreprocessor
from features.feature_pipeline import FeaturePipeline
from models.xgboost_model import XGBoostModel
from research.regime_analysis import MarketRegimeAnalyzer


def run_experiment_5():
    print("=" * 60)
    print("RUNNING EXPERIMENT 5: Model Robustness Across Market Regimes")
    print("=" * 60)

    gen = SyntheticMarketDataGenerator(seed=42)
    ob, tr = gen.generate_orderbook_and_trades(n_steps=2000)
    prep = OrderBookPreprocessor()
    aligned = prep.clean_and_align(ob, tr)

    pipe = FeaturePipeline()
    feats = pipe.extract_features(aligned)
    df, target_cols = pipe.create_targets(feats)

    feature_cols = [c for c in df.columns if c not in ["timestamp", "mid_price"] + target_cols]

    reg_analyzer = MarketRegimeAnalyzer(n_regimes=3)
    regimes = reg_analyzer.fit_predict_regimes(df, regime_features=["realized_vol_30", "obi_lvl_1"])

    X = df[feature_cols].values
    y = df["target_class_5"].values

    split = int(len(X) * 0.75)
    X_tr, y_tr = X[:split], y[:split]
    X_te, y_te = X[split:], y[split:]
    regimes_te = regimes[split:]

    model = XGBoostModel()
    model.fit(X_tr, y_tr)

    regime_metrics_df = reg_analyzer.evaluate_by_regime(model, X_te, y_te, regimes_te, is_classification=True)

    os.makedirs("experiments/results", exist_ok=True)
    res_dict = {
        "experiment": "exp5_regimes",
        "regime_sliced_performance": regime_metrics_df.to_dict(orient="records")
    }

    with open("experiments/results/exp5_regimes.json", "w") as f:
        json.dump(res_dict, f, indent=2)

    print("\nModel Performance Sliced by Market Regime:")
    print(regime_metrics_df)
    print("\n[SUCCESS] Experiment 5 Completed. Results saved to experiments/results/exp5_regimes.json\n")
    return res_dict


if __name__ == "__main__":
    run_experiment_5()
