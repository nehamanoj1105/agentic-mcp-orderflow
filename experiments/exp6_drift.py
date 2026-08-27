"""
Experiment 6: Distribution Drift & Degradation Alert Validation.

Simulates synthetic distribution shift in order flow features and evaluates
whether Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) tests trigger alerts.
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
from monitoring.drift import DriftDetector
from monitoring.performance import PerformanceMonitor


def run_experiment_6():
    print("=" * 60)
    print("RUNNING EXPERIMENT 6: Distribution Drift & Degradation Alerts")
    print("=" * 60)

    gen = SyntheticMarketDataGenerator(seed=42)
    ob, tr = gen.generate_orderbook_and_trades(n_steps=1500)
    prep = OrderBookPreprocessor()
    aligned = prep.clean_and_align(ob, tr)

    pipe = FeaturePipeline()
    feats = pipe.extract_features(aligned)
    df, target_cols = pipe.create_targets(feats)

    # Clean reference window
    ref_df = df.iloc[:500].copy()
    
    # Injected drift window (shift mean and std of OBI and volatility)
    drifted_df = df.iloc[500:1000].copy()
    drifted_df["obi_lvl_1"] = drifted_df["obi_lvl_1"] + np.random.normal(0.8, 0.5, len(drifted_df))
    drifted_df["realized_vol_30"] = drifted_df["realized_vol_30"] * 3.5

    detector = DriftDetector()
    drift_audit = detector.detect_feature_drift(ref_df, drifted_df, feature_cols=["obi_lvl_1", "realized_vol_30"])

    os.makedirs("experiments/results", exist_ok=True)
    res_dict = {
        "experiment": "exp6_drift",
        "reference_samples": len(ref_df),
        "drifted_samples": len(drifted_df),
        "overall_feature_drift": drift_audit["overall_feature_drift"],
        "drift_audit_details": drift_audit["features"]
    }

    with open("experiments/results/exp6_drift.json", "w") as f:
        json.dump(res_dict, f, indent=2)

    print("\nFeature Drift Audit Results (Injected Drift Scenario):")
    print(json.dumps(drift_audit, indent=2))
    print("\n[SUCCESS] Experiment 6 Completed. Results saved to experiments/results/exp6_drift.json\n")
    return res_dict


if __name__ == "__main__":
    run_experiment_6()
