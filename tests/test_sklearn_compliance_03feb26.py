"""
sklearn API compliance tests for ERBFRegressor.

Created: 03Feb26
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_friedman1
from sklearn.utils.estimator_checks import parametrize_with_checks

from erbf import ERBFRegressor


def _erbf_expected_failed_checks(estimator):
    """Return dict of {check_name: reason} for checks expected to fail."""
    return {
        'check_sample_weights_pandas_series': 'sample_weight not yet supported',
        'check_sample_weights_not_an_array': 'sample_weight not yet supported',
        'check_sample_weights_list': 'sample_weight not yet supported',
        'check_sample_weights_shape': 'sample_weight not yet supported',
        'check_sample_weights_not_overwritten': 'sample_weight not yet supported',
        'check_sample_weight_equivalence_on_dense_data': 'sample_weight not yet supported',
        'check_fit2d_1sample': 'RBF networks require more than 1 sample',
    }


@parametrize_with_checks(
    [ERBFRegressor(n_rbf=5, random_state=42, width_optim_iters=10)],
    expected_failed_checks=_erbf_expected_failed_checks,
)
def test_sklearn_compatible(estimator, check):
    check(estimator)


# --- Targeted smoke tests ---

@pytest.fixture
def fitted_model():
    X, y = make_friedman1(n_samples=200, random_state=42)
    m = ERBFRegressor(n_rbf=10, random_state=42)
    m.fit(X, y)
    return m, X


def test_n_features_in(fitted_model):
    m, X = fitted_model
    assert m.n_features_in_ == 10


def test_n_outputs(fitted_model):
    m, _ = fitted_model
    assert m.n_outputs_ == 1


def test_n_params(fitted_model):
    m, _ = fitted_model
    K, d = m.centres_.shape
    assert m.n_params_ == K * d * 2 + K + 1


def test_feature_names_in():
    X, y = make_friedman1(n_samples=200, random_state=42)
    df = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(10)])
    m = ERBFRegressor(n_rbf=10, random_state=42)
    m.fit(df, y)
    assert list(m.feature_names_in_) == [f'feat_{i}' for i in range(10)]


def test_feature_names_not_set_for_array(fitted_model):
    m, _ = fitted_model
    assert not hasattr(m, 'feature_names_in_')


def test_width_summary_shape(fitted_model):
    m, _ = fitted_model
    ws = m.width_summary()
    assert ws.shape == (10, 4)
    assert list(ws.columns) == ['mean', 'std', 'min', 'max']


def test_width_summary_with_feature_names():
    X, y = make_friedman1(n_samples=200, random_state=42)
    df = pd.DataFrame(X, columns=[f'x{i}' for i in range(10)])
    m = ERBFRegressor(n_rbf=10, random_state=42)
    m.fit(df, y)
    ws = m.width_summary()
    assert list(ws.index) == [f'x{i}' for i in range(10)]


def test_sample_weight_rejection(fitted_model):
    m, X = fitted_model
    y = np.ones(X.shape[0])
    with pytest.raises(NotImplementedError, match='sample_weight'):
        m.fit(X, y, sample_weight=np.ones(X.shape[0]))
