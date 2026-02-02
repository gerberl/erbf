"""Tests for ERBF components (activations, weights, auto params)."""

import numpy as np
from erbf import compute_activations, solve_weights_ridge, recommend_erbf_hyperparams, ERBFHyperparams


def test_compute_activations_shape():
    """Output shape (n, K)."""
    rng = np.random.RandomState(0)
    X = rng.randn(50, 4)
    centres = rng.randn(10, 4)
    widths = np.ones((10, 4))
    phi = compute_activations(X, centres, widths)
    assert phi.shape == (50, 10)
    assert np.all(phi >= 0) and np.all(phi <= 1)


def test_solve_weights_ridge():
    """Known solution on simple data."""
    rng = np.random.RandomState(0)
    Phi = rng.randn(100, 5)
    true_w = np.array([1.0, -2.0, 0.5, 0.0, 3.0])
    y = Phi @ true_w
    w_hat = solve_weights_ridge(Phi, y, alpha=0.001)
    np.testing.assert_allclose(w_hat, true_w, atol=0.1)


def test_recommend_hyperparams():
    """Returns ERBFHyperparams with sensible values."""
    params = recommend_erbf_hyperparams(n_samples=1000, n_features=10)
    assert isinstance(params, ERBFHyperparams)
    assert 20 <= params.n_rbf <= 200
    assert params.alpha > 0
    d = params.to_dict()
    assert 'n_rbf' in d and 'alpha' in d
