"""
Width initialization strategies for ERBFN.

Provides various methods for initializing RBF width parameters.

Created: 03Jan26
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from .local_analysis import lipschitz_guided_init, local_ridge_widths


def init_widths_local_ridge(X, y, centres, k=None, alpha=1.0, tau=None):
    """Initialize widths using local Ridge regression.

    For each RBF center, fits a local Ridge regression to nearby points
    and uses the residual variance to estimate appropriate widths per dimension.

    This accounts for local linear trends - if a region has a strong linear
    trend, raw variance overestimates RBF width needed. By fitting and
    subtracting the linear trend first, we get the true "residual" variance.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Training features.
    y : ndarray of shape (n_samples,)
        Training target.
    centres : ndarray of shape (n_rbf, n_features)
        RBF center locations.
    k : int or None
        Number of neighbors for local analysis. If None, uses adaptive k.
    alpha : float
        Ridge regularization strength.
    tau : float or None
        Width scaling factor. If None, uses 1.5 * sqrt(n_features).

    Returns
    -------
    widths : ndarray of shape (n_rbf, n_features)
        Initialized width parameters.
    """
    n_samples, n_features = X.shape
    n_rbf = len(centres)

    if k is None:
        k = min(100, n_samples // max(n_rbf, 1))
        k = max(k, 10)

    if tau is None:
        tau = 1.5 * np.sqrt(n_features)

    widths, _ = local_ridge_widths(
        X, y, centres, k=k, alpha=alpha,
        tau=tau, min_width_ratio=0.1
    )
    return widths


def init_widths_local_variance(X, centres, k=None):
    """Initialize widths using k-NN variance around each center.

    Formula: width_j = local_std_j * sqrt(n_features)
    This gives activation ~ 1/e at the typical neighbor distance.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Training features.
    centres : ndarray of shape (n_rbf, n_features)
        RBF center locations.
    k : int or None
        Number of neighbors. If None, uses adaptive k.

    Returns
    -------
    widths : ndarray of shape (n_rbf, n_features)
        Initialized width parameters.
    """
    n_samples, n_features = X.shape
    n_rbf = len(centres)

    if k is None:
        k = max(10, n_samples // 20)

    nn = NearestNeighbors(n_neighbors=k).fit(X)
    _, indices = nn.kneighbors(centres)
    dim_scale = np.sqrt(n_features)

    widths = np.zeros((n_rbf, n_features))
    for j in range(n_rbf):
        neighbor_X = X[indices[j]]
        local_std = np.std(neighbor_X, axis=0)
        local_std = np.maximum(local_std, 0.01)
        widths[j] = local_std * dim_scale

    # Clip to avoid numerical issues
    widths = np.clip(widths, 0.1, 10.0)
    return widths


def init_widths_hybrid(X, y, centres, k=None):
    """Initialize widths using local variance + correlation-based expansion.

    Formula: width[i,j] = local_std[j] * sqrt(d) * (1 + sqrt(1 - r[j]^2))

    Components:
    1. local_std[j] * sqrt(d): principled base (activation ~ 1/e at neighbor distance)
    2. (1 + sqrt(1 - r^2)): correlation-based expansion
       - r = |corr(feature_j, target)|
       - High |r| -> scale ~ 1 -> base width (precise on predictive features)
       - Low |r| -> scale ~ 2 -> wider (smooth over uninformative features)

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Training features.
    y : ndarray of shape (n_samples,)
        Training target.
    centres : ndarray of shape (n_rbf, n_features)
        RBF center locations.
    k : int or None
        Number of neighbors. If None, uses adaptive k.

    Returns
    -------
    widths : ndarray of shape (n_rbf, n_features)
        Initialized width parameters.
    """
    n_samples, n_features = X.shape
    n_rbf = len(centres)

    if k is None:
        k = max(10, n_samples // 20)

    dim_scale = np.sqrt(n_features)

    # Compute global correlations with target (once per layer)
    correlations = np.zeros(n_features)
    for j in range(n_features):
        r = np.corrcoef(X[:, j], y)[0, 1]
        correlations[j] = np.abs(r) if not np.isnan(r) else 0.0

    # Correlation scaling: 1 + sqrt(1 - r^2), range [1, 2]
    corr_scale = 1.0 + np.sqrt(1 - correlations**2)

    widths = np.zeros((n_rbf, n_features))
    for i in range(n_rbf):
        # Find k nearest neighbors
        sq_distances = np.sum((X - centres[i])**2, axis=1)
        nearest_k = np.argsort(sq_distances)[:k]

        # Local spread in each dimension
        local_std = np.std(X[nearest_k], axis=0)
        local_std = np.maximum(local_std, 0.01)

        # Combine: base width * correlation scaling
        widths[i] = local_std * dim_scale * corr_scale

    widths = np.clip(widths, 0.1, 10.0)
    return widths


def init_widths_lipschitz(X, y, n_rbf, k=10, random_state=None):
    """Initialize widths from local gradient magnitudes.

    Uses Lipschitz-guided initialization which estimates local gradients
    and sets widths based on how fast the function changes.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Training features.
    y : ndarray of shape (n_samples,)
        Training target.
    n_rbf : int
        Number of RBF units.
    k : int
        Number of neighbors for gradient estimation.
    random_state : int or None
        Random seed.

    Returns
    -------
    widths : ndarray of shape (n_rbf, n_features)
        Initialized width parameters.
    """
    # lipschitz_guided_init returns (centers, widths, L, scores)
    # We only use the widths here
    _, widths, _, _ = lipschitz_guided_init(
        X, y, n_rbf=n_rbf, k=k, random_state=random_state
    )
    return widths


def init_widths_uniform(X, n_rbf):
    """Initialize widths uniformly based on feature standard deviations.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Training features.
    n_rbf : int
        Number of RBF units.

    Returns
    -------
    widths : ndarray of shape (n_rbf, n_features)
        Initialized width parameters.
    """
    base_width = np.std(X, axis=0) + 1e-10
    widths = np.tile(base_width, (n_rbf, 1))
    return widths


def apply_width_mode(widths, mode='full'):
    """Apply width parameterization mode.

    Transforms full width matrix based on parameterization constraints.

    Parameters
    ----------
    widths : ndarray of shape (n_rbf, n_features)
        Full width parameters (K x d).
    mode : str
        Parameterization mode:
        - 'full': K x d params (no change)
        - 'shared': d params, shared across RBFs
        - 'isotropic': K params, one scalar per RBF
        - 'rank1': K+d params, factorized as u @ v.T

    Returns
    -------
    widths : ndarray of shape (n_rbf, n_features)
        Transformed width parameters.
    """
    n_rbf, n_features = widths.shape

    if mode == 'full':
        return widths  # K x d, no change

    elif mode == 'shared':
        # Average across RBFs to get single d-vector
        shared = np.mean(widths, axis=0, keepdims=True)
        return np.tile(shared, (n_rbf, 1))

    elif mode == 'isotropic':
        # One scalar per RBF (average across features)
        iso = np.mean(widths, axis=1, keepdims=True)
        return np.tile(iso, (1, n_features))

    elif mode == 'rank1':
        # Factorized: approximate widths as outer product u @ v.T
        # Use SVD to find best rank-1 approximation
        U, S, Vt = np.linalg.svd(widths, full_matrices=False)
        u = U[:, 0:1] * np.sqrt(S[0])  # K x 1
        v = Vt[0:1, :] * np.sqrt(S[0])  # 1 x d
        return u @ v  # K x d, rank-1

    else:
        raise ValueError(f"Unknown width_mode: {mode}")


def init_widths(X, y, centres, method='local_ridge', mode='full',
                k=None, alpha=1.0, tau=None, random_state=None):
    """Initialize RBF widths using the specified method.

    Dispatcher function that routes to specific initialization strategies
    and applies the width parameterization mode.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Training features.
    y : ndarray of shape (n_samples,)
        Training target.
    centres : ndarray of shape (n_rbf, n_features)
        RBF center locations.
    method : str, default='local_ridge'
        Initialization method:
        - 'local_ridge': Supervised, uses local ridge coefficients
        - 'local_variance': Unsupervised, k-NN variance
        - 'hybrid': local_variance + correlation-based expansion
        - 'lipschitz': From local gradient magnitudes
        - 'uniform': Scaled by feature std
    mode : str, default='full'
        Width parameterization mode ('full', 'shared', 'isotropic', 'rank1').
    k : int or None
        Number of neighbors (for applicable methods).
    alpha : float
        Ridge regularization strength (for local_ridge).
    tau : float or None
        Width scaling factor (for local_ridge).
    random_state : int or None
        Random seed.

    Returns
    -------
    widths : ndarray of shape (n_rbf, n_features)
        Initialized width parameters.
    """
    n_rbf = len(centres)

    if method == 'local_ridge':
        widths = init_widths_local_ridge(X, y, centres, k=k, alpha=alpha, tau=tau)

    elif method == 'local_variance':
        widths = init_widths_local_variance(X, centres, k=k)

    elif method == 'hybrid':
        widths = init_widths_hybrid(X, y, centres, k=k)

    elif method == 'lipschitz':
        widths = init_widths_lipschitz(X, y, n_rbf, k=k or 10, random_state=random_state)

    elif method == 'uniform':
        widths = init_widths_uniform(X, n_rbf)

    else:
        raise ValueError(f"Unknown width initialization method: {method}")

    # Apply width mode
    widths = apply_width_mode(widths, mode)

    return widths
