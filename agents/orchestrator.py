"""
Agentic Multi-Agent Orchestrator.

Coordinates specialized sub-agents (ResearchAgent, SignalAgent, MonitoringAgent)
and routes user natural language queries to deterministic MCP tool calls.
"""

from typing import Dict, Any, List
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from .research_agent import ResearchAgent
from .signal_agent import SignalAgent
from .monitoring_agent import MonitoringAgent
from mcp_server.server import (
    calculate_microstructure_features,
    calculate_zscore,
    detect_market_regime,
    get_model_performance,
    detect_feature_drift,
    detect_prediction_drift
)


class AgentOrchestrator:
    """
    Central Orchestrator for routing queries and invoking MCP tool chains.
    """

    def __init__(self):
        self.research_agent = ResearchAgent()
        self.signal_agent = SignalAgent()
        self.monitoring_agent = MonitoringAgent()

    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Processes natural language prompt by identifying required tools and sub-agents.
        """
        query_lower = query.lower()

        # Workflow 1: Market Flow & Model Health Audit
        if "unusual" in query_lower or "reliably" in query_lower or "health" in query_lower:
            micro_info = calculate_microstructure_features()
            zscore_info = calculate_zscore(window=30)
            regime_info = detect_market_regime()
            perf_info = get_model_performance()
            feat_drift = detect_feature_drift()
            pred_drift = detect_prediction_drift()
            health_audit = self.monitoring_agent.audit_system_health()

            synthesis = (
                f"### Market Microstructure & Model Reliability Report\n\n"
                f"- **Market State**: {regime_info['regime_name']} (Regime ID {regime_info['current_regime_id']})\n"
                f"- **Order Imbalance Z-Score**: {zscore_info['latest_obi_zscore']:.2f}\n"
                f"- **Microprice Deviation**: ${micro_info['microprice_dev']:.4f}\n"
                f"- **Feature Drift Status**: {feat_drift['overall_feature_drift']}\n"
                f"- **Prediction Drift Status**: {pred_drift['prediction_drift_status']}\n"
                f"- **Model Performance**: {perf_info['performance_status']} (Accuracy: {perf_info.get('accuracy', 0.0):.2%})\n\n"
                f"**Overall Assessment**: {health_audit['summary']}"
            )

            return {
                "query": query,
                "workflow": "market_and_model_reliability_audit",
                "tools_executed": [
                    "calculate_microstructure_features",
                    "calculate_zscore",
                    "detect_market_regime",
                    "get_model_performance",
                    "detect_feature_drift",
                    "detect_prediction_drift"
                ],
                "response": synthesis,
                "data": {
                    "microstructure": micro_info,
                    "zscore": zscore_info,
                    "regime": regime_info,
                    "performance": perf_info,
                    "feature_drift": feat_drift,
                    "prediction_drift": pred_drift
                }
            }

        # Workflow 2: Research & Model Benchmark
        elif "predictive" in query_lower or "best" in query_lower or "model" in query_lower:
            res_analysis = self.research_agent.evaluate_best_model()
            obi_analysis = self.research_agent.analyze_predictive_power()

            synthesis = (
                f"### Quantitative Model Benchmark & Predictive Power Report\n\n"
                f"- **Top Performing Model**: {res_analysis['top_model_name']} (F1 Score: {res_analysis['top_model_f1']:.4f})\n"
                f"- **Predictive Power**: {obi_analysis['findings']}\n\n"
                f"**Conclusion**: {res_analysis['conclusion']}"
            )

            return {
                "query": query,
                "workflow": "research_and_benchmark",
                "tools_executed": ["compare_models", "calculate_order_imbalance"],
                "response": synthesis,
                "data": res_analysis
            }

        # Default Workflow: Real-time Signal Analysis
        else:
            signal_res = self.signal_agent.analyze_market_signal()
            return {
                "query": query,
                "workflow": "realtime_signal_analysis",
                "tools_executed": ["get_orderbook", "calculate_microstructure_features", "predict"],
                "response": signal_res["summary"],
                "data": signal_res
            }
