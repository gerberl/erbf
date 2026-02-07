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

## Tuning with GridSearchCV

```python
from sklearn.model_selection import GridSearchCV

param_grid = {'n_rbf': [20, 40], 'alpha': [0.1, 1.0]}
gs = GridSearchCV(
    ERBFRegressor(random_state=42, width_optim_iters=50),
    param_grid, cv=3, scoring='r2',
)
gs.fit(X_train, y_train)
print(gs.best_params_)   # {'alpha': 0.1, 'n_rbf': 40}
print(gs.best_score_)    # 0.996
```

## Tuning with Optuna

```python
import optuna
from sklearn.model_selection import cross_val_score

def objective(trial):
    model = ERBFRegressor(
        n_rbf=trial.suggest_int('n_rbf', 10, 80),
        alpha=trial.suggest_float('alpha', 1e-3, 10.0, log=True),
        center_init=trial.suggest_categorical('center_init', ['lipschitz', 'kmeans']),
        width_init=trial.suggest_categorical('width_init', ['local_ridge', 'local_variance']),
        random_state=42,
    )
    return cross_val_score(model, X_train, y_train, cv=3, scoring='r2').mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)
print(study.best_params)
```

## Inspecting fitted models

After fitting, the model exposes its internal components:

```python
model = ERBFRegressor(n_rbf=40, alpha=0.1, random_state=42)
model.fit(X_train, y_train)

model.centres_.shape   # (40, 10) --- K centres in d-dimensional space
model.widths_.shape    # (40, 10) --- per-RBF per-feature width parameters
model.weights_.shape   # (40,)    --- output layer weights
model.intercept_       # 14.296   --- bias term (mean of training y)
model.n_params_        # 841      --- total fitted parameters: 2*K*d + K + 1
```

`width_summary()` returns per-feature statistics across all RBFs. Features with
narrow mean widths are locally important; features with uniformly wide widths are
effectively ignored:

```python
print(model.width_summary())
#        mean       std       min        max
# 0   6.809     9.248     0.341    31.623      <-- x0: narrow, important
# ...
# 8  29.247     4.845    12.298    31.623      <-- x8: wide, ignored (noise)
```

On Friedman-1, features 0--4 are the true inputs; features 5--9 are noise. The
width summary reflects this: active features have narrower mean widths.

## Roadmap

- [ ] Deterministic centre placement for improved retraining stability
- [ ] Explicit linear term (global trend + localised RBF residuals)
- [ ] Width regularisation to prevent localised steep gradients
- [ ] Adaptive handling of near-discrete and categorical features
- [ ] Mini-batch width optimisation for scaling beyond 50K samples
- [ ] Sparse activation via spatial indexing (BallTree) for large K

## Citation

```bibtex
@article{gerber2026revisiting,
  title={Revisiting Chebyshev Polynomial and Anisotropic RBF Models for Tabular Regression},
  author={Gerber, Luciano and Lloyd, Huw},
  year={2026}
}
```

## Licence

MIT
