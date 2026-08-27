"""
Unit Tests for ML Model Architectures and Ensembles.
"""

import pytest
import numpy as np

from models.baseline import BaselineModel
from models.xgboost_model import XGBoostModel
from models.lightgbm_model import LightGBMModel
from models.neural_model import NeuralNetworkModel
from models.ensemble import WeightedEnsembleModel


@pytest.fixture
def synthetic_train_test_data():
    np.random.seed(42)
    X = np.random.randn(200, 10)
    y = np.random.choice([0, 1, 2], size=200)
    return X[:150], y[:150], X[150:], y[150:]


def test_baseline_models(synthetic_train_test_data):
    X_tr, y_tr, X_te, y_te = synthetic_train_test_data
    rf = BaselineModel(model_type="rf", is_classification=True)
    rf.fit(X_tr, y_tr)
    preds = rf.predict(X_te)
    probs = rf.predict_proba(X_te)

    assert len(preds) == len(X_te)
    assert probs.shape == (len(X_te), 3)


def test_xgboost_model(synthetic_train_test_data):
    X_tr, y_tr, X_te, y_te = synthetic_train_test_data
    xgb_m = XGBoostModel(is_classification=True)
    xgb_m.fit(X_tr, y_tr)
    preds = xgb_m.predict(X_te)
    probs = xgb_m.predict_proba(X_te)

    assert len(preds) == len(X_te)
    assert probs.shape == (len(X_te), 3)


def test_lightgbm_model(synthetic_train_test_data):
    X_tr, y_tr, X_te, y_te = synthetic_train_test_data
    lgb_m = LightGBMModel(is_classification=True)
    lgb_m.fit(X_tr, y_tr)
    preds = lgb_m.predict(X_te)

    assert len(preds) == len(X_te)


def test_neural_network_model(synthetic_train_test_data):
    X_tr, y_tr, X_te, y_te = synthetic_train_test_data
    nn_m = NeuralNetworkModel(is_classification=True, epochs=5)
    nn_m.fit(X_tr, y_tr)
    preds = nn_m.predict(X_te)
    probs = nn_m.predict_proba(X_te)

    assert len(preds) == len(X_te)
    assert probs.shape == (len(X_te), 3)


def test_weighted_ensemble_model(synthetic_train_test_data):
    X_tr, y_tr, X_te, y_te = synthetic_train_test_data
    m1 = BaselineModel(model_type="rf", is_classification=True)
    m2 = XGBoostModel(is_classification=True)
    ensemble = WeightedEnsembleModel(base_models=[m1, m2], weights=[0.4, 0.6])
    ensemble.fit(X_tr, y_tr)

    preds = ensemble.predict(X_te)
    probs = ensemble.predict_proba(X_te)
    assert len(preds) == len(X_te)
    assert probs.shape == (len(X_te), 3)
