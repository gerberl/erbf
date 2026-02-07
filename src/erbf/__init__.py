"""
ERBF - Ellipsoidal RBF Network Regressor

A scikit-learn compatible RBF network with anisotropic (per-feature)
widths optimised via L-BFGS-B with analytical gradients.
"""

from .regressor import ERBFRegressor
from .activations import compute_activations
from .center_initialization import init_centers
from .width_initialization import init_widths
from .width_optimization import optimize_widths_gradient, solve_weights_ridge
from .auto_params import (
    recommend_erbf_hyperparams, get_n_rbf_auto, get_alpha_auto, ERBFHyperparams
)

__all__ = [
    'ERBFRegressor',
    'compute_activations',
    'init_centers',
    'init_widths',
    'optimize_widths_gradient',
    'solve_weights_ridge',
    'recommend_erbf_hyperparams',
    'get_n_rbf_auto',
    'get_alpha_auto',
    'ERBFHyperparams',
]

__version__ = '0.1.0'
