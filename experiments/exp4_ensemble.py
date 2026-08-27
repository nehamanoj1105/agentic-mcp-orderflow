"""
Experiment 4: Weighted Ensemble Out-of-Sample Performance.

Evaluates whether combining XGBoost, LightGBM, RandomForest, and Neural Network models
into a Weighted Ensemble outperforms individual standalone models.
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
from models.ensemble import WeightedEnsembleModel
from research.model_comparison import ModelComparator


def run_experiment_4():
    print("=" * 60)
    print("RUNNING EXPERIMENT 4: Weighted Ensemble Performance Evaluation")
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

    rf = BaselineModel(model_type="rf", is_classification=True)
    xgb_m = XGBoostModel(is_classification=True)
    lgb_m = LightGBMModel(is_classification=True)
    nn_m = NeuralNetworkModel(is_classification=True)

    ensemble_simple = WeightedEnsembleModel(base_models=[rf, xgb_m, lgb_m, nn_m], weights=[0.25, 0.25, 0.25, 0.25])
    ensemble_weighted = WeightedEnsembleModel(base_models=[rf, xgb_m, lgb_m, nn_m], weights=[0.1, 0.4, 0.4, 0.1])

    models_to_test = {
        "RandomForest_Single": rf,
        "XGBoost_Single": xgb_m,
        "LightGBM_Single": lgb_m,
        "NeuralNet_Single": nn_m,
        "Ensemble_Equal_Weights": ensemble_simple,
        "Ensemble_Weighted": ensemble_weighted
    }

    comparator = ModelComparator()
    bench_df = comparator.compare_models(models_to_test, X_tr, y_tr, X_te, y_te, is_classification=True)

    os.makedirs("experiments/results", exist_ok=True)
    res_dict = {
        "experiment": "exp4_ensemble",
        "results": bench_df.to_dict(orient="records")
    }

    with open("experiments/results/exp4_ensemble.json", "w") as f:
        json.dump(res_dict, f, indent=2)

    print("\nEnsemble vs Single Model Performance:")
    print(bench_df[["model_name", "accuracy", "f1_score", "roc_auc", "log_loss"]])
    print("\n[SUCCESS] Experiment 4 Completed. Results saved to experiments/results/exp4_ensemble.json\n")
    return res_dict


if __name__ == "__main__":
    run_experiment_4()
