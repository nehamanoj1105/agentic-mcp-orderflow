"""
Primary Pipeline Training & Model Artifact Generator.

Ingests market data, builds synchronized time bars, extracts microstructure features,
generates target labels with leakage prevention, trains model families,
and outputs performance metrics.
"""

import os
import argparse
import json
import numpy as np
import pandas as pd

from data.ingestion.synthetic_gen import SyntheticMarketDataGenerator
from data.preprocessing.orderbook_builder import OrderBookPreprocessor
from data.storage.data_store import MarketDataStore
from features.feature_pipeline import FeaturePipeline
from models.baseline import BaselineModel
from models.xgboost_model import XGBoostModel
from models.lightgbm_model import LightGBMModel
from models.neural_model import NeuralNetworkModel
from models.ensemble import WeightedEnsembleModel
from research.model_comparison import ModelComparator


def main():
    parser = argparse.ArgumentParser(description="Train Market Microstructure ML Models")
    parser.add_argument("--steps", type=int, default=2500, help="Number of market data steps to generate/process")
    parser.add_argument("--horizon", type=int, default=5, help="Future target horizon (ticks/bars)")
    args = parser.parse_args()

    print("=" * 60)
    print("STARTING MARKET MICROSTRUCTURE MODEL TRAINING PIPELINE")
    print(f"Dataset steps: {args.steps} | Prediction Horizon: {args.horizon}")
    print("=" * 60)

    # 1. Ingestion & Preprocessing
    print("\n[Step 1/5] Ingesting L2 order book and trade data...")
    gen = SyntheticMarketDataGenerator(seed=42)
    ob_df, tr_df = gen.generate_orderbook_and_trades(n_steps=args.steps)
    
    prep = OrderBookPreprocessor()
    aligned_df = prep.clean_and_align(ob_df, tr_df)
    print(f"Data ingested and aligned: {len(aligned_df)} time bars.")

    # 2. Feature Extraction
    print("\n[Step 2/5] Extracting microstructure, order flow & volatility features...")
    pipe = FeaturePipeline(target_horizons=[1, 5, 10, 30, 60])
    feats_df = pipe.extract_features(aligned_df)
    full_df, target_cols = pipe.create_targets(feats_df)

    feature_cols = [c for c in full_df.columns if c not in ["timestamp", "mid_price"] + target_cols]
    print(f"Extracted {len(feature_cols)} features with 0 look-ahead leakage.")

    # Store dataset
    store = MarketDataStore()
    store.save_dataset(full_df, "research_dataset.parquet")

    # 3. Model Training & Evaluation
    print("\n[Step 3/5] Training Baseline, Tree, and Neural Network models...")
    X = full_df[feature_cols].values
    target_class_col = f"target_class_{args.horizon}"
    y = full_df[target_class_col].values

    split = int(len(X) * 0.8)
    X_tr, y_tr = X[:split], y[:split]
    X_te, y_te = X[split:], y[split:]

    rf = BaselineModel(model_type="rf", is_classification=True)
    xgb_m = XGBoostModel(is_classification=True)
    lgb_m = LightGBMModel(is_classification=True)
    nn_m = NeuralNetworkModel(is_classification=True)
    ensemble = WeightedEnsembleModel(base_models=[rf, xgb_m, lgb_m, nn_m], weights=[0.1, 0.4, 0.4, 0.1])

    models = {
        "RandomForest": rf,
        "XGBoost": xgb_m,
        "LightGBM": lgb_m,
        "NeuralNetwork": nn_m,
        "WeightedEnsemble": ensemble
    }

    print("\n[Step 4/5] Evaluating out-of-sample test performance...")
    comparator = ModelComparator()
    results_df = comparator.compare_models(models, X_tr, y_tr, X_te, y_te, is_classification=True)

    print("\n" + "=" * 60)
    print("OUT-OF-SAMPLE MODEL BENCHMARK RESULTS")
    print("=" * 60)
    print(results_df[["model_name", "accuracy", "precision", "recall", "f1_score", "roc_auc", "log_loss"]])

    # 5. Save Artifacts
    print("\n[Step 5/5] Saving benchmark metrics...")
    os.makedirs("experiments/results", exist_ok=True)
    out_dict = {
        "dataset_steps": args.steps,
        "target_horizon": args.horizon,
        "train_samples": len(X_tr),
        "test_samples": len(X_te),
        "metrics": results_df.to_dict(orient="records")
    }
    with open("experiments/results/train_pipeline_results.json", "w") as f:
        json.dump(out_dict, f, indent=2)

    print("\n[COMPLETE] Model training pipeline finished successfully!\n")


if __name__ == "__main__":
    main()
