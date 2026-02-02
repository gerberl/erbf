"""
Center initialization strategies for ERBFN.

Provides various methods for placing RBF centers in the feature space.

Created: 03Jan26
"""

import numpy as np
from sklearn.cluster import KMeans
from .local_analysis import data_lipschitz


def init_centers_kmeans(X, n_rbf, random_state=None):
    """Initialize centers using K-means clustering.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Training features.
    n_rbf : int
        Number of RBF centers.
    random_state : int or None
        Random seed.

    Returns
    -------
    centres : ndarray of shape (n_rbf, n_features)
        RBF center locations.
    """
    n_samples = len(X)
    n_rbf = min(n_rbf, n_samples)
    km = KMeans(n_clusters=n_rbf, random_state=random_state, n_init=10)
    km.fit(X)
    return km.cluster_centers_


def init_centers_lipschitz(X, y, n_rbf, random_state=None):
    """Initialize centers by sampling from high |dy/dx| regions.

    Places centers where the target function changes rapidly,
    which typically needs more RBF coverage.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Training features.
    y : ndarray of shape (n_samples,)
        Training target.
    n_rbf : int
        Number of RBF centers.
    random_state : int or None
        Random seed.

    Returns
    -------
    centres : ndarray of shape (n_rbf, n_features)
        RBF center locations.
    """
    n_samples = len(X)
    n_rbf = min(n_rbf, n_samples)

    if random_state is not None:
        np.random.seed(random_state)

    # Estimate local Lipschitz constants
    local_L = data_lipschitz(X, y, k=5)

    if local_L.sum() < 1e-10:
        weights = np.ones(n_samples) / n_samples
    else:
        weights = local_L / local_L.sum()

    # Ensure enough non-zero weights for sampling without replacement
    n_nonzero = np.count_nonzero(weights)
    if n_nonzero < n_rbf:
        # Add small epsilon to zero weights to allow sampling
        epsilon = 1e-10
        weights = np.where(weights == 0, epsilon, weights)
        weights = weights / weights.sum()

    indices = np.random.choice(n_samples, size=n_rbf, replace=False, p=weights)
    return X[indices].copy()


def init_centers_residual(X, residuals, n_rbf, random_state=None):
    """Initialize centers by sampling proportional to |residual|.

    Places more centers where the current model has large errors.
    Useful for boosting/multi-layer approaches.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Training features.
    residuals : ndarray of shape (n_samples,)
        Current residuals (y - y_pred).
    n_rbf : int
        Number of RBF centers.
    random_state : int or None
        Random seed.

    Returns
    -------
    centres : ndarray of shape (n_rbf, n_features)
        RBF center locations.
    """
    n_samples = len(X)
    n_rbf = min(n_rbf, n_samples)

    if random_state is not None:
        np.random.seed(random_state)

    abs_res = np.abs(residuals)
    if abs_res.sum() < 1e-10:
        weights = np.ones(n_samples) / n_samples
    else:
        weights = abs_res / abs_res.sum()

    # Ensure enough non-zero weights for sampling without replacement
    n_nonzero = np.count_nonzero(weights)
    if n_nonzero < n_rbf:
        # Add small epsilon to zero weights to allow sampling
        epsilon = 1e-10
        weights = np.where(weights == 0, epsilon, weights)
        weights = weights / weights.sum()

    indices = np.random.choice(n_samples, size=n_rbf, replace=False, p=weights)
    return X[indices].copy()


def init_centers_random(X, n_rbf, random_state=None):
    """Initialize centers by random sampling from training data.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Training features.
    n_rbf : int
        Number of RBF centers.
    random_state : int or None
        Random seed.

    Returns
    -------
    centres : ndarray of shape (n_rbf, n_features)
        RBF center locations.
    """
    n_samples = len(X)
    n_rbf = min(n_rbf, n_samples)

    if random_state is not None:
        np.random.seed(random_state)

    indices = np.random.choice(n_samples, size=n_rbf, replace=False)
    return X[indices].copy()


def init_centers(X, y, n_rbf, method='kmeans', residuals=None, random_state=None):
    """Initialize RBF centers using the specified method.

    Dispatcher function that routes to specific initialization strategies.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Training features.
    y : ndarray of shape (n_samples,)
        Training target.
    n_rbf : int
        Number of RBF centers.
    method : str, default='kmeans'
        Initialization method:
        - 'kmeans': K-means clustering
        - 'lipschitz': Sample from high |dy/dx| regions
        - 'residual': Sample proportional to |residual|
        - 'random': Random sampling
    residuals : ndarray or None
        Current residuals (for 'residual' method).
    random_state : int or None
        Random seed.

    Returns
    -------
    centres : ndarray of shape (n_rbf, n_features)
        RBF center locations.
    """
    if method == 'kmeans':
        return init_centers_kmeans(X, n_rbf, random_state)

    elif method == 'lipschitz':
        return init_centers_lipschitz(X, y, n_rbf, random_state)

    elif method == 'residual':
        if residuals is None:
            residuals = y
        return init_centers_residual(X, residuals, n_rbf, random_state)

    elif method == 'random':
        return init_centers_random(X, n_rbf, random_state)

    else:
        raise ValueError(f"Unknown center initialization method: {method}")
