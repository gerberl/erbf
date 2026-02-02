"""
Local Analysis for RBF Initialization

Lipschitz-based and LIME-style gradient estimation for:
- RBF centre placement (more RBFs where function changes rapidly)
- Width initialization (narrow where sharp, wide where smooth)
- Per-feature anisotropic widths

Created: 01Jan26
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import Ridge


def data_lipschitz(X, y, k=5, aggregation='max', robust=True):
    """
    Estimate local Lipschitz constant at each training point.

    No model needed - estimates directly from (X, y) data.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix
    y : ndarray of shape (n_samples,)
        Target values
    k : int, default=5
        Number of neighbors to consider
    aggregation : str, default='max'
        How to aggregate neighbor ratios: 'max' or 'mean'
    robust : bool, default=True
        If True, clip extreme Lipschitz values to 99th percentile.
        Helps with outliers in real datasets.

    Returns
    -------
    local_L : ndarray of shape (n_samples,)
        Local Lipschitz estimate at each point.
        High value = function changes rapidly here.
    """
    X = np.asarray(X)
    y = np.asarray(y).ravel()
    n_samples = len(X)

    # Fit k-NN (k+1 to exclude self)
    nn = NearestNeighbors(n_neighbors=min(k + 1, n_samples)).fit(X)
    distances, indices = nn.kneighbors(X)

    local_L = np.zeros(n_samples)

    for i in range(n_samples):
        # Neighbors (excluding self at index 0)
        neighbor_idx = indices[i, 1:]
        neighbor_dist = distances[i, 1:]

        # |y_i - y_j| / ||x_i - x_j||
        dy = np.abs(y[i] - y[neighbor_idx])
        ratios = dy / (neighbor_dist + 1e-10)

        if aggregation == 'max':
            local_L[i] = np.max(ratios)
        elif aggregation == 'mean':
            local_L[i] = np.mean(ratios)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

    # Robust: clip extreme values to prevent outlier domination
    if robust and len(local_L) > 10:
        p99 = np.percentile(local_L, 99)
        local_L = np.clip(local_L, 0, p99)

    return local_L


def local_gradient(X, y, idx, k=10, nn_precomputed=None):
    """
    Estimate local gradient at point idx via weighted local linear fit.

    LIME-style approach: fit Ridge regression in a local neighborhood,
    weighted by proximity (Gaussian kernel).

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix
    y : ndarray of shape (n_samples,)
        Target values
    idx : int
        Index of point to estimate gradient at
    k : int, default=10
        Number of neighbors for local fit
    nn_precomputed : NearestNeighbors or None
        Pre-fitted NearestNeighbors object for efficiency

    Returns
    -------
    gradient : ndarray of shape (n_features,)
        Estimated local gradient (coefficients of local linear fit)
    """
    X = np.asarray(X)
    y = np.asarray(y).ravel()
    n_samples, n_features = X.shape

    # Get neighbors
    if nn_precomputed is not None:
        nn = nn_precomputed
    else:
        nn = NearestNeighbors(n_neighbors=min(k + 1, n_samples)).fit(X)

    distances, indices = nn.kneighbors(X[idx:idx+1])
    neighbor_idx = indices[0, 1:]  # exclude self
    neighbor_dist = distances[0, 1:]

    if len(neighbor_idx) < 2:
        # Not enough neighbors
        return np.zeros(n_features)

    # Local data centered at point idx
    X_local = X[neighbor_idx] - X[idx]
    y_local = y[neighbor_idx] - y[idx]

    # Proximity weights (Gaussian kernel)
    sigma = np.median(neighbor_dist) + 1e-10
    weights = np.exp(-neighbor_dist**2 / (2 * sigma**2))

    # Weighted local linear fit
    lr = Ridge(alpha=0.01, fit_intercept=False)
    lr.fit(X_local, y_local, sample_weight=weights)

    return lr.coef_


def all_local_gradients(X, y, indices, k=10):
    """
    Compute local gradients at multiple points efficiently.

    Pre-computes k-NN once and reuses for all gradient computations.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix
    y : ndarray of shape (n_samples,)
        Target values
    indices : array-like of int
        Indices of points to compute gradients at
    k : int, default=10
        Number of neighbors for local fits

    Returns
    -------
    gradients : ndarray of shape (len(indices), n_features)
        Local gradient at each requested point
    """
    X = np.asarray(X)
    y = np.asarray(y).ravel()
    indices = np.asarray(indices)
    n_features = X.shape[1]

    # Pre-compute k-NN
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(X))).fit(X)

    gradients = np.zeros((len(indices), n_features))
    for i, idx in enumerate(indices):
        gradients[i] = local_gradient(X, y, idx, k=k, nn_precomputed=nn)

    return gradients


def lipschitz_guided_init(X, y, n_rbf=20, k=10, aggregation='max',
                          random_state=None):
    """
    Place RBFs and set widths based on local smoothness.

    1. Estimates local Lipschitz at all training points
    2. Samples centres with probability proportional to Lipschitz (more RBFs
       where function changes rapidly)
    3. Computes local gradients at centres
    4. Sets per-feature widths inversely proportional to gradient magnitude

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix
    y : ndarray of shape (n_samples,)
        Target values
    n_rbf : int, default=20
        Number of RBFs to place
    k : int, default=10
        Number of neighbors for gradient estimation
    aggregation : str, default='max'
        How to aggregate Lipschitz: 'max' or 'mean'
    random_state : int or None
        Random seed for reproducibility

    Returns
    -------
    centres : ndarray of shape (n_rbf, n_features)
        RBF centre locations
    widths : ndarray of shape (n_rbf, n_features)
        Per-RBF per-feature widths (anisotropic)
    centre_indices : ndarray of shape (n_rbf,)
        Indices of selected centres in X
    diagnostics : dict
        Additional information (local_L, gradients, etc.)
    """
    X = np.asarray(X)
    y = np.asarray(y).ravel()
    n_samples, n_features = X.shape

    if random_state is not None:
        np.random.seed(random_state)

    # Clamp n_rbf
    n_rbf = min(n_rbf, n_samples)

    # 1. Estimate local Lipschitz
    local_L = data_lipschitz(X, y, k=k, aggregation=aggregation)

    # Handle edge case: all Lipschitz values are zero or very small
    if local_L.sum() < 1e-10:
        # Fall back to uniform sampling
        weights = np.ones(n_samples) / n_samples
    else:
        # Normalize to probability distribution
        weights = local_L / local_L.sum()

    # 2. Sample centres weighted by Lipschitz
    centre_indices = np.random.choice(
        n_samples, size=n_rbf, replace=False, p=weights
    )
    centres = X[centre_indices].copy()

    # 3. Compute local gradients at centres
    gradients = all_local_gradients(X, y, centre_indices, k=k)

    # 4. Set widths inversely proportional to gradient magnitude
    base_width = np.std(X, axis=0) + 1e-10  # data scale per feature
    grad_mag = np.abs(gradients)  # shape (n_rbf, n_features)

    # Normalize gradient magnitudes to [0, 1] per feature
    max_grad = grad_mag.max(axis=0, keepdims=True) + 1e-10
    grad_mag_norm = grad_mag / max_grad

    # Width formula: wide where gradient is small, narrow where large
    # Range: [base_width, 2 * base_width]
    # High gradient → grad_mag_norm ≈ 1 → width ≈ base_width (narrow)
    # Low gradient → grad_mag_norm ≈ 0 → width ≈ 2 * base_width (wide)
    widths = base_width * (2 - grad_mag_norm)

    diagnostics = {
        'local_L': local_L,
        'gradients': gradients,
        'grad_mag': grad_mag,
        'sampling_weights': weights,
    }

    return centres, widths, centre_indices, diagnostics


def local_ridge_widths(X, y, centres, k=100, alpha=1.0, tau=1.0,
                       min_width_ratio=0.1, use_tstat=False):
    """
    Compute supervised width initialization using local ridge sensitivity.

    For each centre, fits a local ridge model in its k-NN neighborhood,
    then sets widths inversely proportional to coefficient magnitude.

    This is "prediction-aware" - features that predict y locally get narrow
    widths (more sensitivity), while irrelevant features get wide widths.

    Algorithm (per centre j):
        1. Find k nearest neighbors in X
        2. Fit ridge: y_local ~ β_j0 + Σ β_jd * (x_d - c_jd)
        3. width_jd² ∝ Var(X_local,d) / (|β_jd| + ε)

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Training features (should be scaled).
    y : ndarray of shape (n_samples,)
        Training target (or residuals in boosting).
    centres : ndarray of shape (n_rbf, n_features)
        RBF centre locations.
    k : int, default=100
        Number of neighbors for local ridge fits.
    alpha : float, default=1.0
        Ridge regularization for local fits.
    tau : float, default=1.0
        Overall width scaling factor.
    min_width_ratio : float, default=0.1
        Minimum width as fraction of max width per dimension.
        Prevents widths from collapsing to zero.
    use_tstat : bool, default=False
        If True, use t-statistic (β/SE) instead of raw |β|.
        More robust to different coefficient scales.

    Returns
    -------
    widths : ndarray of shape (n_rbf, n_features)
        Per-RBF per-dimension widths.
    diagnostics : dict
        Contains 'betas', 'local_vars', 'relevance' for inspection.

    Notes
    -----
    Based on ChatGPT suggestion "Local supervised sensitivity (ARD-lite)":
    - Keeps local density via variance term
    - Injects predictive relevance via |β|
    - Naturally shrinks widths on locally predictive dimensions

    Reference: ChatGPT analysis 03Jan26
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    centres = np.asarray(centres, dtype=np.float64)

    n_samples, n_features = X.shape
    n_rbf = centres.shape[0]

    # Clamp k to available samples
    k = min(k, n_samples - 1)

    # Fit k-NN on training data
    nn = NearestNeighbors(n_neighbors=k).fit(X)

    # Find neighbors of each centre
    distances, indices = nn.kneighbors(centres)

    # Storage
    betas = np.zeros((n_rbf, n_features))
    local_vars = np.zeros((n_rbf, n_features))

    for j in range(n_rbf):
        neighbor_idx = indices[j]

        # Local data centered at centre
        X_local = X[neighbor_idx] - centres[j]
        y_local = y[neighbor_idx]

        # Local variance per dimension
        local_vars[j] = np.var(X_local, axis=0) + 1e-10

        # Fit local ridge model
        ridge = Ridge(alpha=alpha, fit_intercept=True)
        ridge.fit(X_local, y_local)

        if use_tstat:
            # Compute t-statistics for more robust relevance
            # t = β / SE(β), where SE ≈ sqrt(MSE * (X'X)^-1)
            y_pred = ridge.predict(X_local)
            mse = np.mean((y_local - y_pred) ** 2) + 1e-10

            # Approximate SE using diagonal of (X'X + αI)^-1
            XtX_diag = np.sum(X_local ** 2, axis=0) + alpha
            se = np.sqrt(mse / (XtX_diag + 1e-10))

            betas[j] = np.abs(ridge.coef_) / (se + 1e-10)
        else:
            betas[j] = np.abs(ridge.coef_)

    # Compute relevance: higher |β| = more relevant = narrower width
    # Add small epsilon to avoid division by zero
    epsilon = 1e-10
    relevance = betas + epsilon

    # Width formula: w_jd² ∝ Var(X_local,d) / relevance_jd
    # Take sqrt to get widths (not squared widths)
    widths_sq = local_vars / relevance
    widths = tau * np.sqrt(widths_sq)

    # Apply minimum width constraint per dimension
    # Prevents any width from being too small relative to the max
    max_widths = widths.max(axis=0, keepdims=True)
    min_widths = min_width_ratio * max_widths
    widths = np.maximum(widths, min_widths)

    # Ensure no NaN/Inf
    widths = np.nan_to_num(widths, nan=1.0, posinf=1.0, neginf=1.0)

    diagnostics = {
        'betas': betas,
        'local_vars': local_vars,
        'relevance': relevance,
        'k_used': k,
    }

    return widths, diagnostics


def lipschitz_summary(X, y, k=5, robust=True):
    """
    Compute summary statistics of data Lipschitz.

    Useful as a diagnostic for data complexity.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,)
    k : int, default=5
    robust : bool, default=True
        If True, clip extreme values to 99th percentile.

    Returns
    -------
    summary : dict
        Statistics: mean, median, max, std of local Lipschitz
    """
    local_L = data_lipschitz(X, y, k=k, aggregation='max', robust=robust)

    return {
        'mean': np.mean(local_L),
        'median': np.median(local_L),
        'max': np.max(local_L),
        'std': np.std(local_L),
        'p90': np.percentile(local_L, 90),
        'p10': np.percentile(local_L, 10),
    }
