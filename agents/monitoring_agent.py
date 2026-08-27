"""
Specialized Monitoring Agent.

Interprets feature drift, prediction drift, data quality alerts,
and model performance degradation across market regimes.
"""

from typing import Dict, Any
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from mcp_server.server import detect_feature_drift, detect_prediction_drift, get_model_performance, detect_market_regime


class MonitoringAgent:
    """
    Agent responsible for system health, drift detection, and alert interpretation.
    """

    def audit_system_health(self) -> Dict[str, Any]:
        feat_drift = detect_feature_drift()
        pred_drift = detect_prediction_drift()
        perf_health = get_model_performance()
        regime_info = detect_market_regime()

        overall_status = "HEALTHY"
        alerts = []

        if feat_drift["overall_feature_drift"] == "DETECTED":
            overall_status = "WARNING"
            alerts.append("Feature distribution drift detected via PSI/KS tests.")

        if pred_drift["prediction_drift_status"] == "DETECTED":
            overall_status = "WARNING"
            alerts.append("Prediction distribution drift detected.")

        if perf_health["performance_status"] == "DEGRADED":
            overall_status = "CRITICAL"
            alerts.append("Model accuracy/F1 score has degraded below benchmark threshold!")

        summary = (
            f"System Health Status: {overall_status}. "
            f"Active Regime: {regime_info['regime_name']}. "
            f"Model Accuracy: {perf_health.get('accuracy', 0.0):.2%}, "
            f"Feature Drift: {feat_drift['overall_feature_drift']}, "
            f"Prediction Drift: {pred_drift['prediction_drift_status']}."
        )

        return {
            "agent": "MonitoringAgent",
            "overall_status": overall_status,
            "summary": summary,
            "alerts": alerts,
            "regime_info": regime_info,
            "feature_drift": feat_drift,
            "prediction_drift": pred_drift,
            "performance_health": perf_health
        }
