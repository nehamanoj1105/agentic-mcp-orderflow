"""
Specialized Signal Agent.

Analyzes current market microstructure conditions, order flow imbalance,
and real-time prediction confidence.
"""

from typing import Dict, Any
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from mcp_server.server import predict, get_orderbook, calculate_microstructure_features


class SignalAgent:
    """
    Agent responsible for real-time market signal synthesis.
    """

    def analyze_market_signal(self) -> Dict[str, Any]:
        ob_info = get_orderbook(depth_level=1)
        micro_info = calculate_microstructure_features()
        pred_info = predict(model_type="xgboost")

        signal = pred_info["predicted_signal"]
        conf = pred_info["confidence"]

        summary = (
            f"Market Signal: {signal} (Confidence: {conf:.2%}). "
            f"Current Mid Price: ${ob_info['mid_price']:.2f}, "
            f"Spread: ${micro_info['bid_ask_spread']:.2f}, "
            f"Microprice Dev: ${micro_info['microprice_dev']:.4f}."
        )

        return {
            "agent": "SignalAgent",
            "signal": signal,
            "confidence": conf,
            "summary": summary,
            "orderbook": ob_info,
            "microstructure": micro_info,
            "prediction_details": pred_info
        }
