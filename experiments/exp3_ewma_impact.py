"""
Experiment 3: Impact of EWMA Decay Rates on Predictive Power.

Evaluates how exponentially weighted moving average (EWMA) decay parameters (alpha)
affect short-term return correlation (Spearman IC).
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
from research.feature_analysis import FeatureAnalyzer


def run_experiment_3():
    print("=" * 60)
    print("RUNNING EXPERIMENT 3: Impact of EWMA Decay Rates")
    print("=" * 60)

    gen = SyntheticMarketDataGenerator(seed=42)
    ob, tr = gen.generate_orderbook_and_trades(n_steps=2000)
    prep = OrderBookPreprocessor()
    aligned = prep.clean_and_align(ob, tr)

    pipe = FeaturePipeline()
    feats = pipe.extract_features(aligned)
    df, target_cols = pipe.create_targets(feats)

    ewma_cols = [c for c in df.columns if "ewma_obi" in c]

    analyzer = FeatureAnalyzer()
    res_df = analyzer.analyze_features(df, ewma_cols, target_col="target_return_5", is_classification=False)

    os.makedirs("experiments/results", exist_ok=True)
    res_dict = {
        "experiment": "exp3_ewma_impact",
        "target": "target_return_5",
        "ewma_decay_comparison": res_df[["feature", "spearman_ic", "pearson_corr"]].to_dict(orient="records")
    }

    with open("experiments/results/exp3_ewma_impact.json", "w") as f:
        json.dump(res_dict, f, indent=2)

    print("\nEWMA Decay Alpha Comparison (Spearman IC):")
    print(res_df[["feature", "spearman_ic", "pearson_corr"]])
    print("\n[SUCCESS] Experiment 3 Completed. Results saved to experiments/results/exp3_ewma_impact.json\n")
    return res_dict


if __name__ == "__main__":
    run_experiment_3()
