"""
Unit Tests for Distribution Drift & Performance Monitoring.
"""

import pytest
import numpy as np
import pandas as pd

from monitoring.drift import DriftDetector
from monitoring.performance import PerformanceMonitor


def test_psi_calculation():
    detector = DriftDetector()
    ref = np.random.normal(0, 1, 500)
    # Same distribution -> low PSI
    cur_same = np.random.normal(0, 1, 500)
    psi_same = detector.calculate_psi(ref, cur_same)
    assert psi_same < 0.15

    # Shifted distribution -> high PSI
    cur_shifted = np.random.normal(2.5, 1, 500)
    psi_shifted = detector.calculate_psi(ref, cur_shifted)
    assert psi_shifted > 0.25


def test_feature_drift_detection():
    detector = DriftDetector()
    ref_df = pd.DataFrame({"feat1": np.random.normal(0, 1, 200)})
    cur_df = pd.DataFrame({"feat1": np.random.normal(3, 1, 200)})

    res = detector.detect_feature_drift(ref_df, cur_df, feature_cols=["feat1"])
    assert res["overall_feature_drift"] == "DETECTED"
    assert res["features"]["feat1"]["drift_status"] == "DETECTED"


def test_performance_monitoring():
    monitor = PerformanceMonitor(min_accuracy_threshold=0.5)
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 2])
    eval_res = monitor.evaluate_performance(y_true, y_pred)
    assert eval_res["performance_status"] == "STABLE"
    assert eval_res["accuracy"] == 1.0
