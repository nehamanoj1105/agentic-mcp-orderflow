"""
FastAPI Real-Time Inference & Agent Service.

Exposes low-latency REST endpoints for real-time model inference, latency profiling,
agentic query execution, and monitoring health checks.
"""

import sys
import os
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.orchestrator import AgentOrchestrator
from mcp_server.server import predict, get_market_data, get_model_performance, detect_feature_drift

app = FastAPI(
    title="Agentic Market Microstructure Research & ML System",
    version="1.0.0",
    description="Real-time orderbook microstructure inference and agentic monitoring server"
)

orchestrator = AgentOrchestrator()


class QueryRequest(BaseModel):
    query: str


class PredictRequest(BaseModel):
    model_type: Optional[str] = "xgboost"


@app.get("/")
def read_root():
    return {
        "status": "online",
        "system": "Agentic Market Microstructure Research & ML Monitoring System",
        "version": "1.0.0"
    }


@app.post("/predict")
def run_prediction(req: PredictRequest):
    """
    Performs real-time order-flow model inference and measures feature/inference latency.
    """
    start_t = time.time()
    res = predict(model_type=req.model_type or "xgboost")
    total_latency_ms = (time.time() - start_t) * 1000.0

    res["inference_latency_ms"] = round(total_latency_ms, 3)
    return res


@app.post("/agent/query")
def execute_agent_query(req: QueryRequest):
    """
    Executes multi-agent orchestrator query workflow.
    """
    try:
        return orchestrator.process_query(req.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
def get_system_metrics():
    """
    Returns data quality, drift status, and model performance metrics.
    """
    market = get_market_data(n_steps=50)
    perf = get_model_performance()
    drift = detect_feature_drift()

    return {
        "market": market,
        "performance": perf,
        "drift": drift
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
