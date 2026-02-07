"""
ERBFRegressor: Single-layer Ellipsoidal RBF Network.

A clean API for single-layer RBF networks with gradient-optimized widths.

Validated Configuration
-----------------------
**Locked-in settings** (always use):
- width_optim='gradient' - L-BFGS-B with analytical gradients
- width_mode='full' - per-RBF per-feature widths (+5% R2 vs shared)

**Recommended defaults**:
- n_rbf=40, center_init='lipschitz', width_init='local_ridge', alpha=1.0
"""

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted, validate_data

from .base import _ERBFBase
from .activations import compute_activations
from .auto_params import get_n_rbf_auto


class ERBFRegressor(_ERBFBase):
    """
    Ellipsoidal RBF Network Regressor.

    A single-layer RBF network with gradient-optimized anisotropic widths.

    Parameters
    ----------
    n_rbf : int or 'auto', default=40
        Number of RBF centers.
        - int: Fixed number (validated: 40 optimal for d <= 20)
        - 'auto': Adaptive based on n_samples and n_features
          Formula: n_rbf = clip(max(40, 2*d), min=20, max=min(200, n//10))

    center_init : str, default='lipschitz'
        How to place RBF centers:
        - 'kmeans': K-means clustering
        - 'lipschitz': Sample from high |dy/dx| regions (recommended)
        - 'residual': Sample proportional to |residual|
        - 'random': Random sample from training data

    width_init : str, default='local_ridge'
        How to initialize widths:
        - 'local_ridge': Supervised, uses local ridge coefficients (recommended)
        - 'local_variance': Unsupervised, k-NN variance
        - 'hybrid': local_variance + correlation-based expansion
        - 'lipschitz': From local gradient magnitudes
        - 'uniform': Scaled by feature std

    width_mode : str, default='full'
        How widths are parameterized:
        - 'full': Per-RBF per-feature (K*d params) - recommended
        - 'shared': One width vector for all RBFs (d params)
        - 'isotropic': One scalar per RBF (K params)

    width_optim : str or None, default='gradient'
        How to optimize widths:
        - 'gradient': L-BFGS-B with analytical gradients (recommended)
        - None: Use initialized widths directly

    width_optim_iters : int, default=30
        Number of optimization iterations.

    regularization : str, default='ridge'
        Regularization for output weights ('ridge', 'lasso', 'elasticnet').

    alpha : float, default=1.0
        Regularization strength.

    l1_ratio : float, default=0.5
        ElasticNet mixing parameter (only used if regularization='elasticnet').

    standardize : bool, default=True
        Standardize features before fitting.

    random_state : int or None, default=None
        Random seed for reproducibility.

    verbose : int, default=0
        Verbosity level.

    Attributes
    ----------
    centres_ : ndarray of shape (n_rbf, n_features)
        RBF center locations.
    widths_ : ndarray of shape (n_rbf, n_features)
        RBF width parameters.
    weights_ : ndarray of shape (n_rbf,)
        Output layer weights.
    intercept_ : float
        Bias term (mean of training y).
    scaler_ : StandardScaler or None
        Feature scaler (if standardize=True).

    Examples
    --------
    >>> from erbf import ERBFRegressor
    >>> from sklearn.datasets import make_regression
    >>> from sklearn.model_selection import train_test_split
    >>>
    >>> X, y = make_regression(n_samples=500, n_features=10, random_state=42)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    >>>
    >>> model = ERBFRegressor(n_rbf=40, random_state=42)
    >>> model.fit(X_train, y_train)
    >>> print(f"R2: {model.score(X_test, y_test):.4f}")
    """

    def __init__(
        self,
        n_rbf=40,
        center_init='lipschitz',
        width_init='local_ridge',
        width_mode='full',
        width_optim='gradient',
        width_optim_iters=30,
        regularization='ridge',
        alpha=1.0,
        l1_ratio=0.5,
        standardize=True,
        random_state=None,
        verbose=0,
    ):
        super().__init__(
            n_rbf=n_rbf,
            center_init=center_init,
            width_init=width_init,
            width_mode=width_mode,
            width_optim=width_optim,
            width_optim_iters=width_optim_iters,
            regularization=regularization,
            alpha=alpha,
            l1_ratio=l1_ratio,
            standardize=standardize,
            random_state=random_state,
            verbose=verbose,
        )

    def fit(self, X, y, sample_weight=None):
        """
        Fit the ERBF model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        sample_weight : array-like of shape (n_samples,), default=None
            Not yet supported. Raises NotImplementedError if provided.

        Returns
        -------
        self : ERBFRegressor
            Fitted estimator.
        """
        if sample_weight is not None:
            raise NotImplementedError("sample_weight is not yet supported")

        # sklearn input validation (sets n_features_in_, feature_names_in_)
        X, y = validate_data(
            self, X, y,
            accept_sparse=False,
            dtype=np.float64,
            multi_output=False,
            y_numeric=True,
        )
        y = y.ravel()

        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Standardize features
        X_scaled = self._standardize_input(X, fit=True)

        # Fit the single-layer model
        self._fit_single_layer(X_scaled, y)

        self.n_outputs_ = 1

        return self

    def predict(self, X):
        """
        Predict target values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        check_is_fitted(self)
        X = validate_data(self, X, accept_sparse=False, dtype=np.float64, reset=False)
        X_scaled = self._standardize_input(X, fit=False)

        activations = compute_activations(X_scaled, self.centres_, self.widths_)
        return activations @ self.weights_ + self.intercept_

    def _fit_single_layer(self, X, y):
        """Fit a single-layer ERBFN."""
        n_samples, n_features = X.shape
        n_rbf = self._resolve_n_rbf(n_samples, n_features)

        if self.verbose >= 1:
            print(f"Single layer mode: n_rbf={n_rbf}")

        # Store bias as mean(y), fit RBFs to residuals
        bias = np.mean(y)
        residuals = y - bias

        # Fit the RBF layer
        layer = self._fit_rbf_layer(X, residuals, n_rbf)

        # Store results
        self.centres_ = layer['centres']
        self.widths_ = layer['widths']
        self.weights_ = layer['weights']
        self.intercept_ = bias

        if self.verbose >= 1:
            activations = compute_activations(X, self.centres_, self.widths_)
            y_pred = activations @ self.weights_ + bias
            r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2)
            print(f"Training R2: {r2:.4f}")

    def _resolve_n_rbf(self, n_samples, n_features):
        """Resolve 'auto' n_rbf to actual value."""
        return self.n_rbf if self.n_rbf != 'auto' else get_n_rbf_auto(n_samples, n_features)

    @property
    def n_params_(self):
        """Total number of fitted parameters: K*d (centres) + K*d (widths) + K (weights) + 1 (intercept)."""
        K, d = self.centres_.shape
        return K * d + K * d + K + 1

    def width_summary(self):
        """Summary statistics of widths across RBFs, per feature.

        Returns
        -------
        df : DataFrame
            Columns: mean, std, min, max. One row per feature.
            Narrow widths indicate local importance; wide widths indicate
            the feature is ignored by that RBF.
        """
        w = self.widths_  # (K, d)
        index = (self.feature_names_in_.tolist()
                 if hasattr(self, 'feature_names_in_')
                 else list(range(w.shape[1])))
        return pd.DataFrame({
            'mean': w.mean(axis=0),
            'std': w.std(axis=0),
            'min': w.min(axis=0),
            'max': w.max(axis=0),
        }, index=index)
