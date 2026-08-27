"""
Experiment 2: Model Architecture Benchmark Comparison.

Compares Baseline (Logistic Regression, Random Forest), XGBoost, LightGBM,
and Neural Network models on identical out-of-sample datasets.
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
from models.baseline import BaselineModel
from models.xgboost_model import XGBoostModel
from models.lightgbm_model import LightGBMModel
from models.neural_model import NeuralNetworkModel
from research.model_comparison import ModelComparator


def run_experiment_2():
    print("=" * 60)
    print("RUNNING EXPERIMENT 2: Model Architecture Benchmark Comparison")
    print("=" * 60)

    gen = SyntheticMarketDataGenerator(seed=42)
    ob, tr = gen.generate_orderbook_and_trades(n_steps=2000)
    prep = OrderBookPreprocessor()
    aligned = prep.clean_and_align(ob, tr)

    pipe = FeaturePipeline()
    feats = pipe.extract_features(aligned)
    df, target_cols = pipe.create_targets(feats)

    feature_cols = [c for c in df.columns if c not in ["timestamp", "mid_price"] + target_cols]

    X = df[feature_cols].values
    y = df["target_class_5"].values

    split = int(len(X) * 0.75)
    X_tr, y_tr = X[:split], y[:split]
    X_te, y_te = X[split:], y[split:]

    models = {
        "LogisticRegression": BaselineModel(model_type="logistic", is_classification=True),
        "RandomForest": BaselineModel(model_type="rf", is_classification=True),
        "XGBoost": XGBoostModel(is_classification=True),
        "LightGBM": LightGBMModel(is_classification=True),
        "NeuralNetwork": NeuralNetworkModel(is_classification=True)
    }

    comparator = ModelComparator()
    bench_df = comparator.compare_models(models, X_tr, y_tr, X_te, y_te, is_classification=True)

    os.makedirs("experiments/results", exist_ok=True)
    res_dict = {
        "experiment": "exp2_model_benchmark",
        "sample_size_train": len(X_tr),
        "sample_size_test": len(X_te),
        "target": "target_class_5",
        "benchmark_summary": bench_df.to_dict(orient="records")
    }

    with open("experiments/results/exp2_model_benchmark.json", "w") as f:
        json.dump(res_dict, f, indent=2)

    print("\nModel Architecture Comparison Benchmark:")
    print(bench_df)
    print("\n[SUCCESS] Experiment 2 Completed. Results saved to experiments/results/exp2_model_benchmark.json\n")
    return res_dict


if __name__ == "__main__":
    run_experiment_2()
