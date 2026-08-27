"""
Streamlit Monitoring & Research Dashboard.

Provides an interactive user interface featuring:
1. Market Microstructure View (Real-time L2 orderbook, OBI, Microprice, Spread)
2. ML Model Performance View (Live signals, confidence, side-by-side benchmarks)
3. Monitoring & Drift View (PSI feature drift, prediction drift, regime indicators)
4. Quantitative Research View (Feature IC rankings, walk-forward folds, agentic workflow query interface)
"""

import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.ingestion.synthetic_gen import SyntheticMarketDataGenerator
from data.preprocessing.orderbook_builder import OrderBookPreprocessor
from features.feature_pipeline import FeaturePipeline
from models.xgboost_model import XGBoostModel
from research.model_comparison import ModelComparator
from research.feature_analysis import FeatureAnalyzer
from research.walk_forward import WalkForwardEvaluator
from research.regime_analysis import MarketRegimeAnalyzer
from monitoring.drift import DriftDetector
from monitoring.performance import PerformanceMonitor
from agents.orchestrator import AgentOrchestrator

st.set_page_config(
    page_title="Market Microstructure ML Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Agentic Market Microstructure Research & ML Monitoring System")
st.markdown("*Quantitative ML Pipeline & Real-Time Order Flow Monitoring for High-Frequency Trading*")

# Cache data loading for smooth UI rendering
@st.cache_data
def load_research_dataset():
    gen = SyntheticMarketDataGenerator(seed=42)
    ob, tr = gen.generate_orderbook_and_trades(n_steps=1200)
    prep = OrderBookPreprocessor()
    aligned = prep.clean_and_align(ob, tr)
    pipe = FeaturePipeline()
    feats = pipe.extract_features(aligned)
    df, target_cols = pipe.create_targets(feats)
    feature_cols = [c for c in df.columns if c not in ["timestamp", "mid_price"] + target_cols]
    return df, feature_cols, target_cols

df, feature_cols, target_cols = load_research_dataset()

# Sidebar Navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Select View:", [
    "📊 Market Microstructure",
    "🤖 ML Models & Signals",
    "⚠️ Monitoring & Drift Alerts",
    "🔬 Research & Experiments",
    "💬 Agentic Copilot"
])

# ----------------------------------------------------
# View 1: Market Microstructure
# ----------------------------------------------------
if page == "📊 Market Microstructure":
    st.header("📊 Market Microstructure & Order Flow")
    
    col1, col2, col3, col4 = st.columns(4)
    latest = df.iloc[-1]
    col1.metric("Mid Price", f"${latest['mid_price']:.2f}")
    col2.metric("Spread", f"${latest['bid_ask_spread']:.2f}")
    col3.metric("Order Imbalance (OBI)", f"{latest['obi_lvl_1']:.4f}")
    col4.metric("Microprice Dev", f"${latest['microprice_dev']:.4f}")

    st.subheader("Price & Order Flow Imbalance Time Series")
    st.line_chart(df.set_index("timestamp")[["mid_price", "microprice"]])
    
    st.subheader("Multi-Level Order Book Imbalance (OBI)")
    st.line_chart(df.set_index("timestamp")[["obi_lvl_1", "obi_lvl_3", "obi_lvl_5", "obi_lvl_10"]])

# ----------------------------------------------------
# View 2: ML Models & Signals
# ----------------------------------------------------
elif page == "🤖 ML Models & Signals":
    st.header("🤖 ML Model Predictions & Benchmarks")

    X = df[feature_cols].values
    y = df["target_class_5"].values
    split = int(len(X) * 0.8)
    X_tr, y_tr = X[:split], y[:split]
    X_te, y_te = X[split:], y[split:]

    model = XGBoostModel()
    model.fit(X_tr, y_tr)
    probs = model.predict_proba(X_te[-1:]) [0]
    pred_class = int(np.argmax(probs))
    class_names = {0: "DOWN 🔴", 1: "NEUTRAL ⚪", 2: "UP 🟢"}

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Live Real-Time Prediction")
        st.write(f"**Predicted Signal**: {class_names[pred_class]}")
        st.write(f"**Confidence**: {np.max(probs):.2%}")
        st.progress(float(np.max(probs)))

    with col2:
        st.subheader("Class Probabilities")
        prob_df = pd.DataFrame({"Class": ["DOWN", "NEUTRAL", "UP"], "Probability": probs})
        st.bar_chart(prob_df.set_index("Class"))

    st.subheader("Side-by-Side Model Comparison Benchmark")
    from models.baseline import BaselineModel
    from models.lightgbm_model import LightGBMModel
    from models.neural_model import NeuralNetworkModel

    models = {
        "RandomForest": BaselineModel(model_type="rf"),
        "XGBoost": XGBoostModel(),
        "LightGBM": LightGBMModel(),
        "NeuralNet": NeuralNetworkModel()
    }
    comparator = ModelComparator()
    bench_df = comparator.compare_models(models, X_tr, y_tr, X_te, y_te, is_classification=True)
    st.dataframe(bench_df, use_container_width=True)

# ----------------------------------------------------
# View 3: Monitoring & Drift Alerts
# ----------------------------------------------------
elif page == "⚠️ Monitoring & Drift Alerts":
    st.header("⚠️ Model Monitoring, Drift & Regime Analysis")

    split = int(len(df) * 0.5)
    ref_df = df.iloc[:split]
    cur_df = df.iloc[split:]

    detector = DriftDetector()
    drift_info = detector.detect_feature_drift(ref_df, cur_df, feature_cols=["obi_lvl_1", "realized_vol_30"])

    col1, col2, col3 = st.columns(3)
    col1.metric("Feature Drift", drift_info["overall_feature_drift"], delta_color="inverse")

    perf_mon = PerformanceMonitor()
    X = df[feature_cols].values
    y = df["target_class_5"].values
    model = XGBoostModel().fit(X[:split], y[:split])
    y_pred = model.predict(X[split:])
    perf_info = perf_mon.evaluate_performance(y[split:], y_pred)

    col2.metric("Model Performance", perf_info["performance_status"])
    col3.metric("Test Accuracy", f"{perf_info['accuracy']:.2%}")

    st.subheader("Feature PSI & Kolmogorov-Smirnov Audit")
    st.json(drift_info["features"])

    st.subheader("Market Regime Classification")
    reg_analyzer = MarketRegimeAnalyzer(n_regimes=3)
    regimes = reg_analyzer.fit_predict_regimes(df, regime_features=["realized_vol_30", "obi_lvl_1"])
    df["regime"] = regimes
    st.line_chart(df.set_index("timestamp")["regime"])

# ----------------------------------------------------
# View 4: Research & Experiments
# ----------------------------------------------------
elif page == "🔬 Research & Experiments":
    st.header("🔬 Quantitative Research & Walk-Forward Validation")

    st.subheader("Feature Information Coefficient (IC) Rankings")
    analyzer = FeatureAnalyzer()
    ic_df = analyzer.analyze_features(df, feature_cols, target_col="target_return_5", is_classification=False)
    st.dataframe(ic_df, use_container_width=True)

    st.subheader("Walk-Forward Cross-Validation Folds")
    X = df[feature_cols].values
    y = df["target_class_5"].values
    wf_eval = WalkForwardEvaluator(n_splits=5)
    fold_df, oof_preds, oof_targets = wf_eval.evaluate_walk_forward(XGBoostModel, {"is_classification": True}, X, y)
    st.dataframe(fold_df, use_container_width=True)

# ----------------------------------------------------
# View 5: Agentic Copilot
# ----------------------------------------------------
elif page == "💬 Agentic Copilot":
    st.header("💬 Agentic Research Copilot")
    st.markdown("Ask natural language quantitative questions regarding market flow, model benchmarks, or drift alerts.")

    user_query = st.text_input("Enter your research question:", "Is the current BTC order flow unusual and does our model still perform reliably?")
    if st.button("Run Agent Query"):
        orchestrator = AgentOrchestrator()
        result = orchestrator.process_query(user_query)
        st.markdown(result["response"])
        st.subheader("MCP Tools Executed")
        st.json(result["tools_executed"])
