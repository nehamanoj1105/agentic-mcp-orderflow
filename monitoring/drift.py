"""
Feature & Prediction Distribution Drift Monitoring Engine.

Computes Population Stability Index (PSI) and Kolmogorov-Smirnov (KS) tests
to detect distribution shifts between baseline reference and current production windows.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from scipy.stats import ks_2samp


class DriftDetector:
    """
    Monitors data and prediction distribution drift using statistical distance metrics.
    """

    @staticmethod
    def calculate_psi(reference: np.ndarray, current: np.ndarray, num_bins: int = 10) -> float:
        """
        Calculates Population Stability Index (PSI):
        PSI = sum( (Actual% - Expected%) * ln(Actual% / Expected%) )

        PSI < 0.1: No significant change
        0.1 <= PSI < 0.25: Moderate shift
        PSI >= 0.25: Significant drift
        """
        ref_clean = reference[np.isfinite(reference)]
        cur_clean = current[np.isfinite(current)]

        if len(ref_clean) < 10 or len(cur_clean) < 10:
            return 0.0

        bins = np.linspace(min(ref_clean.min(), cur_clean.min()), max(ref_clean.max(), cur_clean.max()), num_bins + 1)
        bins[0] -= 1e-5
        bins[-1] += 1e-5

        ref_counts, _ = np.histogram(ref_clean, bins=bins)
        cur_counts, _ = np.histogram(cur_clean, bins=bins)

        ref_pct = (ref_counts + 1e-4) / np.sum(ref_counts + 1e-4)
        cur_pct = (cur_counts + 1e-4) / np.sum(cur_counts + 1e-4)

        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        return float(psi)

    def detect_feature_drift(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        feature_cols: list,
        psi_threshold: float = 0.25
    ) -> Dict[str, Any]:
        """
        Performs PSI and KS-test feature drift audit across all specified features.
        """
        drift_results = {}
        drift_detected = False

        for col in feature_cols:
            if col in reference_df and col in current_df:
                ref_vals = reference_df[col].values
                cur_vals = current_df[col].values

                psi_val = self.calculate_psi(ref_vals, cur_vals)
                ks_stat, ks_pval = ks_2samp(ref_vals, cur_vals)

                is_drift = (psi_val >= psi_threshold) or (ks_pval < 0.01)
                if is_drift:
                    drift_detected = True

                drift_results[col] = {
                    "psi": round(psi_val, 4),
                    "ks_stat": round(float(ks_stat), 4),
                    "ks_p_value": round(float(ks_pval), 6),
                    "drift_status": "DETECTED" if is_drift else "NORMAL"
                }

        return {
            "overall_feature_drift": "DETECTED" if drift_detected else "NORMAL",
            "features": drift_results
        }

    def detect_prediction_drift(
        self,
        ref_preds: np.ndarray,
        cur_preds: np.ndarray,
        psi_threshold: float = 0.2
    ) -> Dict[str, Any]:
        """
        Detects output prediction probability distribution drift.
        """
        psi_val = self.calculate_psi(ref_preds, cur_preds)
        ks_stat, ks_pval = ks_2samp(ref_preds, cur_preds)

        is_drift = psi_val >= psi_threshold or ks_pval < 0.01

        return {
            "prediction_drift_status": "DETECTED" if is_drift else "NORMAL",
            "psi": round(psi_val, 4),
            "ks_stat": round(float(ks_stat), 4),
            "ks_p_value": round(float(ks_pval), 6)
        }
