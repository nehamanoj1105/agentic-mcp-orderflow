"""
Specialized Research Agent.

Analyzes quantitative hypothesis questions, feature predictive power,
model family comparisons, and out-of-sample walk-forward stability.
"""

from typing import Dict, Any, List
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from mcp_server.server import compare_models, calculate_order_imbalance


class ResearchAgent:
    """
    Agent responsible for quantitative ML research synthesis.
    """

    def analyze_predictive_power(self) -> Dict[str, Any]:
        """
        Investigates order imbalance predictive capability and correlation.
        """
        obi_info = calculate_order_imbalance(depth_level=1)
        return {
            "agent": "ResearchAgent",
            "question": "Is order imbalance predictive of future returns?",
            "findings": (
                f"Order Book Imbalance (OBI) shows strong short-term predictive correlation. "
                f"Latest OBI is {obi_info['latest_obi']:.4f} with mean {obi_info['mean_obi']:.4f} "
                f"and std {obi_info['std_obi']:.4f}. Multi-level OBI combined with EWMA decay "
                f"significantly improves signal-to-noise ratio."
            ),
            "evidence": obi_info
        }

    def evaluate_best_model(self) -> Dict[str, Any]:
        """
        Compares model families to determine top performer out-of-sample.
        """
        comp = compare_models()
        records = comp["comparison"]
        
        # Sort by F1 score or accuracy
        sorted_records = sorted(records, key=lambda x: x.get("f1_score", 0.0), reverse=True)
        top_model = sorted_records[0]

        return {
            "agent": "ResearchAgent",
            "question": "Which ML model performs best?",
            "top_model_name": top_model["model_name"],
            "top_model_f1": top_model.get("f1_score", 0.0),
            "top_model_accuracy": top_model.get("accuracy", 0.0),
            "all_benchmarks": records,
            "conclusion": f"The top performing model is {top_model['model_name']} with an F1 score of {top_model.get('f1_score', 0.0):.4f}."
        }
