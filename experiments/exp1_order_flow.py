"""
Experiment 1: Order-Flow Feature Predictive Power Analysis.

Investigates whether multi-level order book imbalance (OBI), trade imbalance,
and microprice deviations contain predictive information for future log returns.
Outputs Pearson correlation, Spearman IC, Mutual Information, and research plots.
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


def run_experiment_1():
    print("=" * 60)
    print("RUNNING EXPERIMENT 1: Order-Flow Predictive Power Analysis")
    print("=" * 60)

    # 1. Generate dataset
    gen = SyntheticMarketDataGenerator(seed=42)
    ob, tr = gen.generate_orderbook_and_trades(n_steps=2000)
    prep = OrderBookPreprocessor()
    aligned = prep.clean_and_align(ob, tr)

    # 2. Extract features & targets
    pipe = FeaturePipeline(target_horizons=[1, 5, 10, 30, 60])
    feats = pipe.extract_features(aligned)
    df, target_cols = pipe.create_targets(feats)

    feature_cols = [c for c in df.columns if c not in ["timestamp", "mid_price"] + target_cols]

    # 3. Analyze feature predictive power
    analyzer = FeatureAnalyzer(output_dir="experiments/plots")
    results_df = analyzer.analyze_features(df, feature_cols, target_col="target_return_5", is_classification=False)

    # Save plots
    corr_plot = analyzer.plot_correlation_matrix(df, feature_cols[:10], title="Exp 1: Order Flow Feature Correlation")
    feat_plot = analyzer.plot_feature_vs_target(df, feature_name="obi_lvl_1", target_col="target_return_5")

    # 4. Save structured results
    os.makedirs("experiments/results", exist_ok=True)
    res_dict = {
        "experiment": "exp1_order_flow",
        "sample_size": len(df),
        "target": "target_return_5",
        "top_features_by_ic": results_df[["feature", "spearman_ic", "pearson_corr", "mutual_info"]].head(10).to_dict(orient="records"),
        "plots_saved": [corr_plot, feat_plot]
    }

    with open("experiments/results/exp1_order_flow.json", "w") as f:
        json.dump(res_dict, f, indent=2)

    print("\nTop 5 Order-Flow Features by Spearman IC:")
    print(results_df[["feature", "spearman_ic", "pearson_corr", "mutual_info"]].head(5))
    print("\n[SUCCESS] Experiment 1 Completed. Results saved to experiments/results/exp1_order_flow.json\n")
    return res_dict


if __name__ == "__main__":
    run_experiment_1()
