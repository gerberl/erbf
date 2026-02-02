"""
ERBF Auto-Hyperparameter Recommendations

Principled heuristics for recommending n_rbf and alpha based on data
characteristics.

Key insight: ERBF collapse at high dimensionality (d > 50) is NOT inherent
--- it is due to insufficient RBF centers for coverage. Scaling n_rbf with
d rescues performance: n_rbf ~ 2*d for high-d.
"""

import numpy as np
from typing import List
from dataclasses import dataclass, field


@dataclass
class ERBFHyperparams:
    """Recommended ERBF hyperparameters with explanations.

    Attributes
    ----------
    n_rbf : int
        Recommended number of RBF centers.
    alpha : float
        Recommended Ridge regularization strength.
    center_init : str
        Recommended center initialization method.
    width_init : str
        Recommended width initialization method.
    width_optim : str
        Recommended width optimization method.
    n_rbf_rationale : str
        Explanation for n_rbf choice.
    alpha_rationale : str
        Explanation for alpha choice.
    warnings : list
        List of warning messages for edge cases.
    """
    n_rbf: int
    alpha: float
    center_init: str
    width_init: str
    width_optim: str
    n_rbf_rationale: str = ""
    alpha_rationale: str = ""
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Return as dict for model instantiation.

        Returns
        -------
        dict
            Parameters suitable for ERBFRegressor(**params.to_dict())
        """
        return {
            'n_rbf': self.n_rbf,
            'alpha': self.alpha,
            'center_init': self.center_init,
            'width_init': self.width_init,
            'width_optim': self.width_optim,
        }

    def __repr__(self):
        s = "ERBFHyperparams(\n"
        s += f"  n_rbf={self.n_rbf}  # {self.n_rbf_rationale}\n"
        s += f"  alpha={self.alpha}  # {self.alpha_rationale}\n"
        s += f"  center_init='{self.center_init}'\n"
        s += f"  width_init='{self.width_init}'\n"
        s += f"  width_optim='{self.width_optim}'\n"
        if self.warnings:
            s += f"  warnings={self.warnings}\n"
        s += ")"
        return s


def recommend_erbf_hyperparams(
    X: np.ndarray = None,
    n_samples: int = None,
    n_features: int = None,
    verbose: bool = False
) -> ERBFHyperparams:
    """
    Recommend ERBF hyperparameters based on data characteristics.

    Parameters
    ----------
    X : array-like, optional
        Training data with shape (n_samples, n_features).
        If provided, n_samples and n_features are inferred.
    n_samples : int, optional
        Number of training samples (if X not provided).
    n_features : int, optional
        Number of features (if X not provided).
    verbose : bool, default=False
        Print detailed recommendations and warnings.

    Returns
    -------
    ERBFHyperparams
        Recommended hyperparameters with rationales.

    Examples
    --------
    >>> from erbf import recommend_erbf_hyperparams, ERBFRegressor
    >>>
    >>> params = recommend_erbf_hyperparams(X_train)
    >>> model = ERBFRegressor(**params.to_dict())
    >>>
    >>> params = recommend_erbf_hyperparams(n_samples=5000, n_features=100)
    >>> print(params)
    """
    if X is not None:
        X = np.asarray(X)
        n_samples, n_features = X.shape

    if n_samples is None or n_features is None:
        raise ValueError("Provide either X or both n_samples and n_features")

    n, d = n_samples, n_features
    warnings = []

    # n_rbf: Number of RBF centers
    # Formula: n_rbf = clip(max(40, 2*d), min=20, max=min(200, n//10))
    n_rbf_base = 40
    n_rbf_scaled = max(n_rbf_base, int(2 * d))
    n_rbf_max_by_samples = max(20, n // 10)
    n_rbf_max_compute = 200

    n_rbf = min(n_rbf_scaled, n_rbf_max_by_samples, n_rbf_max_compute)
    n_rbf = max(20, n_rbf)

    if d <= 20:
        n_rbf_rationale = f"d={d} <= 20: base value sufficient"
    elif n_rbf == n_rbf_max_compute:
        n_rbf_rationale = f"d={d}: scaled to 2*d={2*d}, capped at {n_rbf_max_compute}"
        warnings.append(f"High dimensionality (d={d}): consider feature selection")
    elif n_rbf == n_rbf_max_by_samples:
        n_rbf_rationale = f"d={d}: limited by n/10={n_rbf_max_by_samples} (sample constraint)"
        warnings.append(f"Small sample size relative to d: n/d = {n/d:.1f}")
    else:
        n_rbf_rationale = f"d={d} > 20: scaled to 2*d for coverage"

    # alpha: Ridge regularization
    # Formula: alpha = 1.0 * (1 + max(0, d/n - 0.1) * 5)
    d_over_n = d / n

    if d_over_n > 0.2:
        alpha = 1.0 * (1 + (d_over_n - 0.1) * 10)
        alpha = min(alpha, 10.0)
        alpha_rationale = f"d/n={d_over_n:.2f} > 0.2: increased regularization"
        warnings.append(f"High d/n ratio ({d_over_n:.2f}): overfitting risk")
    elif d > 50:
        alpha = 0.5
        alpha_rationale = f"d={d} > 50 with adequate n: reduced alpha for flexibility"
    else:
        alpha = 1.0
        alpha_rationale = "standard setting"

    alpha = round(alpha, 2)

    center_init = 'lipschitz'
    width_init = 'local_ridge'
    width_optim = 'gradient'

    if d > 100:
        warnings.append(
            f"Very high dimensionality (d={d}): strongly recommend feature selection first"
        )
    if n < 100:
        warnings.append(f"Very small sample size (n={n}): results may be unstable")
    if n < 5 * n_rbf:
        warnings.append(
            f"n={n} < 5*n_rbf={5*n_rbf}: may overfit, consider reducing n_rbf"
        )

    params = ERBFHyperparams(
        n_rbf=n_rbf,
        alpha=alpha,
        center_init=center_init,
        width_init=width_init,
        width_optim=width_optim,
        n_rbf_rationale=n_rbf_rationale,
        alpha_rationale=alpha_rationale,
        warnings=warnings,
    )

    if verbose:
        print(params)
        if warnings:
            print("\nWarnings:")
            for w in warnings:
                print(f"  - {w}")

    return params


def get_n_rbf_auto(n_samples: int, n_features: int) -> int:
    """Quick utility to get recommended n_rbf.

    For use in ERBFRegressor when n_rbf='auto'.

    Parameters
    ----------
    n_samples : int
        Number of training samples.
    n_features : int
        Number of features.

    Returns
    -------
    int
        Recommended number of RBF centers.
    """
    params = recommend_erbf_hyperparams(n_samples=n_samples, n_features=n_features)
    return params.n_rbf


def get_alpha_auto(n_samples: int, n_features: int) -> float:
    """Quick utility to get recommended alpha.

    Parameters
    ----------
    n_samples : int
        Number of training samples.
    n_features : int
        Number of features.

    Returns
    -------
    float
        Recommended alpha (regularization strength).
    """
    params = recommend_erbf_hyperparams(n_samples=n_samples, n_features=n_features)
    return params.alpha
