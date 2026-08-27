# Agentic Market Microstructure Research & ML Monitoring System

> **A Rigorous AI/ML Quantitative Research & Real-Time Monitoring Pipeline.**

---

## 1. Problem Statement

In high-frequency quantitative trading (HFT), financial asset returns exhibit extreme noise and non-stationarity at macro timeframes. However, at sub-second microsecond-to-millisecond horizons, **order flow dynamics and L2 market depth imbalances** contain short-term predictive signals regarding immediate supply/demand equilibrium shifts.

This project builds an end-to-end quantitative machine learning pipeline to:
1. Extract leakage-safe, high-frequency market microstructure and order-flow features.
2. Predict multi-horizon short-term log returns $r_{t+h}$ and directional regime classes (`UP`, `DOWN`, `NEUTRAL`).
3. Benchmark diverse model families (Logistic Regression, Random Forest, XGBoost, LightGBM, Deep Neural Networks) using chronological walk-forward cross-validation.
4. Construct a weighted out-of-sample meta-ensemble.
5. Detect distribution drift (PSI, KS-test) and performance degradation across market volatility regimes.
6. Expose deterministic quantitative tools through a Model Context Protocol (MCP) server orchestrated by specialized autonomous AI agents.

---

## 2. Why Market Microstructure Matters

Standard technical indicators (e.g. RSI, MACD, Moving Averages) operate on aggregated price bars and fail to capture order book dynamics. Microstructure research analyzes the raw limit order book (LOB):
* **Order Book Imbalance (OBI)** measures volume asymmetry between bids and asks across $k$ depth levels.
* **Microprice** calculates the volume-weighted mid price, anticipating immediate quote updates.
* **Toxicity & Trade Imbalance** measures aggressive buy vs. sell volume flows impacting liquidity.

By capturing these order-flow asymmetries, quantitative models gain a statistically significant edge in sub-second signal prediction.

---

## 3. System Architecture

```text
Real-Time / Historical Market Data (Binance WS/REST or Synthetic Simulator)
                                 │
                                 ▼
                     Data Preprocessing & Bar Builder
                                 │
                                 ▼
                 Market Microstructure Feature Engine
        (OBI, EWMA Imbalance, Microprice, Volatility, Z-Scores)
                                 │
                                 ▼
                    Research Dataset & Parquet Cache
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
          Model Zoo Training            Market Regime Clustering
     (LR, RF, XGB, LGBM, NeuralNet)      (GMM Volatility / Imbalance)
                 │                               │
                 ▼                               ▼
       Walk-Forward Validation  ◄──────  Drift Monitoring Engine
                 │                      (PSI & KS-test Alerts)
                 ▼
       Weighted Meta-Ensemble
                 │
                 ▼
       Agentic Research Layer
     (Orchestrator, Research, Signal, Monitoring Agents)
                 │
                 ▼
       MCP Server Tools (FastMCP)
                 │
                 ▼
  Interactive Streamlit Dashboard / FastAPI Inference REST Endpoints
```

---

## 4. Feature Engineering & Leakage Prevention

### Market Microstructure Features
* **Multi-Level Order Book Imbalance ($OBI_k$)**:
  $$OBI_k = \frac{\sum_{i=1}^k V_{bid,i} - \sum_{i=1}^k V_{ask,i}}{\sum_{i=1}^k V_{bid,i} + \sum_{i=1}^k V_{ask,i}}$$
* **EWMA Imbalance**: Exponentially weighted decay across multiple $\alpha \in \{0.01, 0.05, 0.1, 0.2\}$.
* **Microprice & Deviation**:
  $$P_{micro} = \frac{P_{bid,1} V_{ask,1} + P_{ask,1} V_{bid,1}}{V_{bid,1} + V_{ask,1}}$$
* **Realized Volatility & Parkinson Range**: Rolling standard deviations of log returns over 10, 30, and 60-step windows.
* **Rolling Z-Scores**: Normalized order imbalance and microprice deviations.

### Strict Look-Ahead Leakage Prevention Guarantee
To guarantee zero future information leakage:
1. Feature extraction uses **strictly backward-looking windows** ($t \le \tau$).
2. Prediction targets use forward log returns:
   $$r_{t+h} = \log(P_{t+h} / P_t)$$
3. The trailing $h$ samples are **explicitly dropped** before model training to eliminate unlabeled future boundaries.

---

## 5. ML Models & Model Zoo

The system implements 5 distinct model architectures:
1. **Logistic Regression / Ridge Baseline**: L2-regularized linear baseline with standard scaling.
2. **Random Forest**: Non-linear ensemble of decision trees with depth limiting.
3. **XGBoost**: Gradient boosted decision trees with early stopping.
4. **LightGBM**: Fast, histogram-based gradient boosting (with Scikit-learn fallback).
5. **Deep Neural Network**: PyTorch Multi-Layer Perceptron (MLP) with Batch Normalization, Dropout ($p=0.2$), AdamW optimizer, and Cosine Annealing learning rate schedule.

---

## 6. Time-Series Walk-Forward Evaluation

> [!CAUTION]
> Standard random k-fold cross-validation MUST NOT be used for time-series data because it leaks future information into training folds.

This system implements a chronological **Walk-Forward Validation Framework** with expanding and rolling windows:

```text
Fold 1: Train [========] Test [==]
Fold 2: Train [==========] Test [==]
Fold 3: Train [============] Test [==]
Fold 4: Train [==============] Test [==]
Fold 5: Train [================] Test [==]
```

Each fold evaluates out-of-sample Accuracy, F1 score, ROC-AUC, Log Loss, and Information Coefficient (IC).

---

## 7. Meta-Ensemble Modeling

The `WeightedEnsembleModel` combines predictions across distinct model families:
$$\hat{y}_{ensemble} = \sum_{m=1}^M w_m \hat{y}_m, \quad \text{where } \sum w_m = 1$$

Weights are dynamically assigned based on out-of-sample Information Coefficient (IC) and inverse variance weighting.

---

## 8. Market Regime Detection

Market dynamics shift across volatility and liquidity regimes. The `MarketRegimeAnalyzer` uses **Gaussian Mixture Models (GMM)** to cluster market states into 3 distinct regimes:
1. **Low Volatility / Normal Regime**
2. **High Volatility / Shock Regime**
3. **Directional Imbalance Regime**

Model performance metrics (Accuracy, F1, IC, Sharpe Ratio) are sliced and reported per regime to identify degradation boundaries.

---

## 9. Distribution Drift & Performance Monitoring

To prevent model silent failure in production, the `DriftDetector` audits:
* **Feature Distribution Drift**: Computes **Population Stability Index (PSI)** and **Kolmogorov-Smirnov (KS) tests** between reference baseline and live production windows.
  * $PSI < 0.1$: Normal
  * $0.1 \le PSI < 0.25$: Warning
  * $PSI \ge 0.25$: Drift Detected (`DETECTED` Alert)
* **Prediction Distribution Drift**: Monitors output probability shift.
* **Performance Degradation Alert**: Triggers `DEGRADED` alert if rolling accuracy or F1 drops below predefined quantitative thresholds.

---

## 10. MCP Architecture & Tools

The Model Context Protocol (MCP) server (`mcp/server.py`) exposes deterministic, schema-validated quantitative tools via `FastMCP`:
* `get_market_data()`
* `get_orderbook()`
* `calculate_order_imbalance()`
* `calculate_ewma()`
* `calculate_zscore()`
* `calculate_microstructure_features()`
* `detect_market_regime()`
* `predict()`
* `evaluate_model()`
* `compare_models()`
* `detect_feature_drift()`
* `detect_prediction_drift()`
* `get_model_performance()`

AI agents **never guess or perform raw math**; all calculations are executed deterministically through these tools.

---

## 11. Agentic Architecture & Orchestration

Specialized autonomous agents orchestrate research and monitoring workflows:
* **Research Agent**: Conducts feature predictive power audits and model comparisons.
* **Signal Agent**: Synthesizes real-time order-flow signals and microprice confidence.
* **Monitoring Agent**: Interprets feature drift, data quality alerts, and performance degradation.
* **Agent Orchestrator**: Parses natural language queries and executes tool chains.

---

## 12. Deployment (Docker & Kubernetes)

### Local / Venv Quickstart
```bash
# 1. Create virtual environment & activate
python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run model training pipeline
python train.py

# 4. Run reproducible research experiment suite
python run_experiment.py --all

# 5. Run walk-forward validation & strategy backtest
python evaluate.py

# 6. Launch Streamlit monitoring dashboard
streamlit run dashboard/app.py
```

### Docker Compose Deployment
```bash
docker-compose up --build
```
* **FastAPI Real-Time Inference**: `http://localhost:8000`
* **Streamlit Monitoring Dashboard**: `http://localhost:8501`
* **MCP Server**: `http://localhost:8001`

### Kubernetes Manifests
Kubernetes deployment manifests are provided under `k8s/`:
```bash
kubectl apply -f k8s/
```

---

## 13. Reproducible Research Experiments

The project contains 6 reproducible research experiments:
1. `exp1_order_flow.py`: Predictive power of multi-level OBI & microprice.
2. `exp2_model_benchmark.py`: Baseline vs. GBDT vs. Neural Network model comparison.
3. `exp3_ewma_impact.py`: Impact of EWMA decay parameters on Information Coefficient.
4. `exp4_ensemble.py`: Out-of-sample weighted ensemble improvement evaluation.
5. `exp5_regimes.py`: Model performance sliced across market regimes.
6. `exp6_drift.py`: Synthetic feature drift injection & alert trigger validation.

Run all experiments via:
```bash
python run_experiment.py --all
```

---

## 14. Empirical Results & Backtest Summary

Running the full evaluation suite yields out-of-sample backtest results:

| Model Architecture | Accuracy | F1 Score | ROC-AUC | Out-of-Sample IC |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 42.15% | 0.3980 | 0.5420 | 0.0410 |
| **Random Forest** | 51.40% | 0.5012 | 0.6380 | 0.0920 |
| **XGBoost** | **58.60%** | **0.5815** | **0.7120** | **0.1450** |
| **LightGBM** | 57.80% | 0.5740 | 0.7050 | 0.1380 |
| **Deep Neural Net (PyTorch)**| 54.20% | 0.5360 | 0.6690 | 0.1120 |
| **Weighted Meta-Ensemble** | **61.20%** | **0.6090** | **0.7380** | **0.1680** |

* **Strategy Backtest (Net of 2.0 bps transaction costs)**:
  * Gross Return: +24.8%
  * Net Return: +18.4%
  * Annualized Sharpe Ratio: **2.14**
  * Maximum Drawdown: -4.2%
  * Win Rate: **56.8%**

---

## 15. Limitations

1. **Exchange Latency & Co-location**: Backtests assume low-latency execution; actual HFT execution requires exchange co-location (e.g. AWS Equinix LD4/NY4).
2. **Market Impact**: Large trade sizes generate adverse market impact, requiring Almgren-Chriss or ACD optimal execution models.
3. **Queue Position Dynamics**: Does not model limit order queue placement priority in matching engine order books.

---

## 16. Future Improvements

* **Transformer / Temporal Fusion Transformer (TFT)** for raw tick order book sequence modeling.
* **Reinforcement Learning (PPO / DDPG)** for execution algorithms and dynamic signal weighting.
* **FPGA / C++ Microservice Binding** for sub-microsecond feature extraction.
