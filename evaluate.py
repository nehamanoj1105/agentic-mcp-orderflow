"""
Walk-Forward Validation & Backtest Strategy Evaluator.

Executes walk-forward cross-validation across time-series folds and evaluates
realistic transaction-cost-adjusted backtest metrics (Sharpe ratio, Max Drawdown, Net Return).
"""

import argparse
import json
import numpy as np
import pandas as pd

from data.ingestion.synthetic_gen import SyntheticMarketDataGenerator
from data.preprocessing.orderbook_builder import OrderBookPreprocessor
from features.feature_pipeline import FeaturePipeline
from models.xgboost_model import XGBoostModel
from research.walk_forward import WalkForwardEvaluator
from research.statistical_tests import StatisticalBacktester


def main():
    parser = argparse.ArgumentParser(description="Evaluate Walk-Forward & Backtest Strategy")
    parser.add_argument("--folds", type=int, default=5, help="Number of walk-forward folds")
    parser.add_argument("--cost_bps", type=float, default=2.0, help="Transaction cost in basis points per side")
    args = parser.parse_args()

    print("=" * 60)
    print("WALK-FORWARD CROSS-VALIDATION & STRATEGY BACKTEST EVALUATION")
    print(f"Folds: {args.folds} | Transaction Costs: {args.cost_bps} bps")
    print("=" * 60)

    # 1. Dataset generation & feature extraction
    gen = SyntheticMarketDataGenerator(seed=42)
    ob, tr = gen.generate_orderbook_and_trades(n_steps=2000)
    prep = OrderBookPreprocessor()
    aligned = prep.clean_and_align(ob, tr)

    pipe = FeaturePipeline()
    feats = pipe.extract_features(aligned)
    df, target_cols = pipe.create_targets(feats)

    feature_cols = [c for c in df.columns if c not in ["timestamp", "mid_price"] + target_cols]
    X = df[feature_cols].values
    y_class = df["target_class_5"].values
    y_ret = df["target_return_5"].values

    # 2. Walk-forward cross validation
    wf_eval = WalkForwardEvaluator(n_splits=args.folds, expanding=True)
    fold_df, oof_preds, oof_targets = wf_eval.evaluate_walk_forward(
        XGBoostModel,
        {"is_classification": True},
        X,
        y_class,
        is_classification=True
    )

    print("\nWalk-Forward Fold Results:")
    print(fold_df[["fold", "accuracy", "precision", "recall", "f1_score", "roc_auc"]])

    # 3. Strategy Backtest Evaluation
    backtester = StatisticalBacktester()
    bt_metrics = backtester.evaluate_trading_backtest(
        predictions=oof_preds,
        future_returns=y_ret[-len(oof_preds):],
        cost_bps=args.cost_bps,
        is_classification=True
    )

    print("\n" + "=" * 60)
    print("STRATEGY BACKTEST PERFORMANCE METRICS")
    print("=" * 60)
    for k, v in bt_metrics.items():
        if "return" in k or "cost" in k:
            print(f"- {k:20s}: {v*100:8.4f}%")
        elif "ratio" in k or "turnover" in k or "win" in k:
            print(f"- {k:20s}: {v:8.4f}")
        else:
            print(f"- {k:20s}: {v:8.4f}")

    out_dict = {
        "folds": fold_df.to_dict(orient="records"),
        "backtest_metrics": bt_metrics
    }
    with open("experiments/results/evaluate_walk_forward_results.json", "w") as f:
        json.dump(out_dict, f, indent=2)

    print("\n[COMPLETE] Evaluation script completed successfully!\n")


if __name__ == "__main__":
    main()
