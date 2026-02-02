# erbf

Ellipsoidal RBF network regressor with gradient-optimised anisotropic widths.

[![PyPI version](https://img.shields.io/pypi/v/erbf.svg)](https://pypi.org/project/erbf/)
[![Tests](https://github.com/gerberl/erbf/actions/workflows/test.yml/badge.svg)](https://github.com/gerberl/erbf/actions)

## Installation

```bash
pip install erbf
```

## Quick start

```python
from erbf import ERBFRegressor
from sklearn.datasets import make_friedman1
from sklearn.model_selection import train_test_split

X, y = make_friedman1(n_samples=1000, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

model = ERBFRegressor(n_rbf=40, random_state=42)
model.fit(X_train, y_train)
print(f"R2: {model.score(X_test, y_test):.3f}")
```

## Key features

- **Anisotropic widths** --- per-feature per-RBF width parameters capture axis-aligned structure
- **Gradient optimisation** --- L-BFGS-B in log-space with analytical gradients; O(n K d) per iteration
- **Modular initialisation** --- Lipschitz-guided centre placement, local-Ridge width initialisation
- **scikit-learn compatible** --- `fit`/`predict`/`score`, `clone()`, `get_params()`/`set_params()`

## How it works

ERBFRegressor places K radial basis functions in the feature space, each with a
per-dimension width vector that controls sensitivity along each axis. Centres are
initialised by sampling from high-gradient regions (Lipschitz-guided), widths are
initialised via local Ridge regression coefficients, then both widths and output
weights are jointly refined. Width optimisation operates in log-space via L-BFGS-B
with analytical gradients, preventing width collapse and ensuring a smooth loss
landscape.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_rbf` | `40` | Number of RBF centres (`'auto'` for adaptive) |
| `center_init` | `'lipschitz'` | Centre placement: `'lipschitz'`, `'kmeans'`, `'random'` |
| `width_init` | `'local_ridge'` | Width initialisation: `'local_ridge'`, `'local_variance'`, `'uniform'` |
| `width_mode` | `'full'` | Parameterisation: `'full'` (K*d), `'shared'` (d), `'isotropic'` (K) |
| `width_optim` | `'gradient'` | Optimisation: `'gradient'` (L-BFGS-B) or `None` |
| `alpha` | `1.0` | Ridge regularisation strength |
| `standardize` | `True` | Standardise features before fitting |

## Citation

```bibtex
@article{gerber2026revisiting,
  title={Revisiting Chebyshev Polynomial and Anisotropic RBF Models for Tabular Regression},
  author={Gerber, Luciano and Lloyd, Chris},
  year={2026}
}
```

## Licence

MIT
