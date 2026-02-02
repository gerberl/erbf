"""
Shared RBF activation computation with numba acceleration.

All regressor classes should use these functions for consistent performance.

Created: 03Jan26
"""

import os
# Suppress OpenMP deprecation warning (omp_set_nested)
os.environ.setdefault('KMP_WARNINGS', '0')

import numpy as np

# Try to import numba for fast activation computation
try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


if NUMBA_AVAILABLE:
    @numba.jit(nopython=True, parallel=True, fastmath=True, cache=True)
    def _activations_numba(X, centres, inv_widths_sq):
        """
        Compute ellipsoidal RBF activations using numba.

        phi[i, j] = exp(-sum_d (x[i,d] - c[j,d])^2 / w[j,d]^2)

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        centres : ndarray of shape (n_rbf, n_features)
        inv_widths_sq : ndarray of shape (n_rbf, n_features)
            Inverse squared widths (1/w^2)

        Returns
        -------
        phi : ndarray of shape (n_samples, n_rbf)
        """
        n_samples, n_features = X.shape
        n_rbf = centres.shape[0]
        phi = np.empty((n_samples, n_rbf))

        for i in numba.prange(n_samples):
            for j in range(n_rbf):
                dist_sq = 0.0
                for d in range(n_features):
                    diff = X[i, d] - centres[j, d]
                    dist_sq += diff * diff * inv_widths_sq[j, d]
                phi[i, j] = np.exp(-dist_sq)

        return phi

    @numba.jit(nopython=True, parallel=True, fastmath=True, cache=True)
    def _activations_weighted_numba(X, centres, inv_widths_sq, feature_weights):
        """
        Compute weighted ellipsoidal RBF activations using numba.

        phi[i, j] = exp(-sum_d importance_d * (x[i,d] - c[j,d])^2 / w[j,d]^2)

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        centres : ndarray of shape (n_rbf, n_features)
        inv_widths_sq : ndarray of shape (n_rbf, n_features)
        feature_weights : ndarray of shape (n_features,)
            Feature importance weights

        Returns
        -------
        phi : ndarray of shape (n_samples, n_rbf)
        """
        n_samples, n_features = X.shape
        n_rbf = centres.shape[0]
        phi = np.empty((n_samples, n_rbf))

        for i in numba.prange(n_samples):
            for j in range(n_rbf):
                dist_sq = 0.0
                for d in range(n_features):
                    diff = X[i, d] - centres[j, d]
                    dist_sq += diff * diff * inv_widths_sq[j, d] * feature_weights[d]
                phi[i, j] = np.exp(-dist_sq)

        return phi


def _activations_numpy(X, centres, widths):
    """
    Compute ellipsoidal RBF activations using pure numpy.

    Fallback when numba is not available.
    """
    n_samples = X.shape[0]
    n_rbf = centres.shape[0]

    phi = np.zeros((n_samples, n_rbf))

    for j in range(n_rbf):
        diff = X - centres[j]
        w = np.maximum(widths[j], 1e-10)
        sq_dist = np.sum((diff / w) ** 2, axis=1)
        phi[:, j] = np.exp(-sq_dist)

    return phi


def _activations_weighted_numpy(X, centres, widths, feature_weights):
    """
    Compute weighted ellipsoidal RBF activations using pure numpy.

    Fallback when numba is not available.
    """
    n_samples = X.shape[0]
    n_rbf = centres.shape[0]

    phi = np.zeros((n_samples, n_rbf))

    for j in range(n_rbf):
        diff = X - centres[j]
        w = np.maximum(widths[j], 1e-10)
        sq_dist_per_dim = (diff / w) ** 2
        sq_dist = np.sum(sq_dist_per_dim * feature_weights, axis=1)
        phi[:, j] = np.exp(-sq_dist)

    return phi


def compute_activations(X, centres, widths, feature_weights=None):
    """
    Compute ellipsoidal RBF activations.

    Automatically uses numba if available for ~10-25x speedup.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input samples.
    centres : ndarray of shape (n_rbf, n_features)
        RBF centres.
    widths : ndarray of shape (n_rbf, n_features)
        Per-dimension widths for each RBF.
    feature_weights : ndarray of shape (n_features,), optional
        Feature importance weights. If provided, important features
        contribute more to activation distance.

    Returns
    -------
    phi : ndarray of shape (n_samples, n_rbf)
        Activation matrix.
    """
    if NUMBA_AVAILABLE:
        # Precompute inverse squared widths for numba
        inv_widths_sq = 1.0 / (np.maximum(widths, 1e-10) ** 2)

        if feature_weights is not None:
            return _activations_weighted_numba(
                X.astype(np.float64),
                centres.astype(np.float64),
                inv_widths_sq.astype(np.float64),
                feature_weights.astype(np.float64)
            )
        else:
            return _activations_numba(
                X.astype(np.float64),
                centres.astype(np.float64),
                inv_widths_sq.astype(np.float64)
            )
    else:
        if feature_weights is not None:
            return _activations_weighted_numpy(X, centres, widths, feature_weights)
        else:
            return _activations_numpy(X, centres, widths)
