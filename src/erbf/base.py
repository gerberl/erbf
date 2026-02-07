"""
Base class for ERBF regressors.

Contains shared functionality for ERBFRegressor and BoostedERBFRegressor.

Created: 18Jan26
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, ElasticNet

from .activations import compute_activations
from .center_initialization import init_centers
from .width_initialization import init_widths, apply_width_mode
from .width_optimization import optimize_widths, solve_weights_ridge


class _ERBFBase(RegressorMixin, BaseEstimator):
    """
    Base class for Ellipsoidal RBF Network Regressors.

    Provides shared functionality:
    - Standardization
    - Weight solving (Ridge/Lasso/ElasticNet)
    - Single RBF layer fitting

    Subclasses: ERBFRegressor, BoostedERBFRegressor
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
        self.n_rbf = n_rbf
        self.center_init = center_init
        self.width_init = width_init
        self.width_mode = width_mode
        self.width_optim = width_optim
        self.width_optim_iters = width_optim_iters
        self.regularization = regularization
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.standardize = standardize
        self.random_state = random_state
        self.verbose = verbose

    def _standardize_input(self, X, fit=False):
        """
        Standardize features if enabled.

        Parameters
        ----------
        X : ndarray
            Input features.
        fit : bool
            If True, fit the scaler (call during fit()).
            If False, transform only (call during predict()).

        Returns
        -------
        X_scaled : ndarray
            Standardized features (or original if standardize=False).
        """
        if not self.standardize:
            self.scaler_ = None
            return X

        if fit:
            self.scaler_ = StandardScaler()
            return self.scaler_.fit_transform(X)
        else:
            return self.scaler_.transform(X)

    def _solve_weights(self, activations, y):
        """
        Solve for output weights with regularization.

        Parameters
        ----------
        activations : ndarray of shape (n_samples, n_rbf)
            RBF activation matrix.
        y : ndarray of shape (n_samples,)
            Target values (or residuals).

        Returns
        -------
        weights : ndarray of shape (n_rbf,)
            Solved output weights.
        """
        if self.regularization == 'ridge':
            return solve_weights_ridge(activations, y, self.alpha)
        elif self.regularization == 'lasso':
            model = Lasso(alpha=self.alpha, fit_intercept=False, max_iter=1000)
            model.fit(activations, y)
            return model.coef_
        elif self.regularization == 'elasticnet':
            model = ElasticNet(
                alpha=self.alpha,
                l1_ratio=self.l1_ratio,
                fit_intercept=False,
                max_iter=1000
            )
            model.fit(activations, y)
            return model.coef_
        else:
            raise ValueError(f"Unknown regularization: {self.regularization}")

    def _fit_rbf_layer(self, X, y, n_rbf, width_scale=1.0, width_optim_iters=None):
        """
        Fit a single RBF layer.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Standardized input features.
        y : ndarray of shape (n_samples,)
            Target values (or residuals for boosting).
        n_rbf : int
            Number of RBF centers.
        width_scale : float, default=1.0
            Scale factor for widths (for coarse-to-fine scheduling).
        width_optim_iters : int, optional
            Override self.width_optim_iters for this layer.

        Returns
        -------
        layer : dict
            Dictionary with 'centres', 'widths', 'weights'.
        """
        if width_optim_iters is None:
            width_optim_iters = self.width_optim_iters

        # Initialize centers
        centres = init_centers(
            X, y, n_rbf,
            method=self.center_init,
            residuals=y,
            random_state=self.random_state
        )

        # Initialize widths
        mode = 'full' if self.width_optim == 'abc' else self.width_mode
        widths = init_widths(
            X, y, centres,
            method=self.width_init,
            mode=mode,
            random_state=self.random_state
        )

        # Apply width mode if using ABC
        if self.width_optim == 'abc' and mode == 'full':
            widths = apply_width_mode(widths, self.width_mode)

        # Apply width scaling (for coarse-to-fine)
        if width_scale != 1.0:
            widths = widths * width_scale

        # Optimize widths
        if self.width_optim is not None:
            widths, _ = optimize_widths(
                X, y, centres, widths,
                method=self.width_optim,
                alpha=self.alpha,
                width_mode=self.width_mode,
                max_iters=width_optim_iters,
                random_state=self.random_state,
                verbose=self.verbose - 1 if self.verbose > 0 else 0
            )

        # Compute activations and solve for weights
        activations = compute_activations(X, centres, widths)
        weights = self._solve_weights(activations, y)

        return {
            'centres': centres,
            'widths': widths,
            'weights': weights,
        }
