"""
Optuna hyperparameter tuning for ERBFRegressor.

Search space adapted from the poly_erbf_benchmark project.

Usage:
    python examples/optuna_tuning.py
"""

import numpy as np
import optuna
from sklearn.datasets import make_friedman1
from sklearn.model_selection import cross_val_score, KFold

from erbf import ERBFRegressor


def erbf_objective(trial: optuna.Trial, X, y):
    """Optuna objective: 5-fold CV R² for ERBFRegressor."""
    use_auto_rbf = trial.suggest_categorical("n_rbf_auto", [True, False])
    n_rbf = "auto" if use_auto_rbf else trial.suggest_int("n_rbf", 10, 80)

    model = ERBFRegressor(
        n_rbf=n_rbf,
        alpha=trial.suggest_float("alpha", 1e-3, 1e3, log=True),
        center_init=trial.suggest_categorical("center_init", ["lipschitz", "kmeans"]),
        width_init=trial.suggest_categorical("width_init", ["local_ridge", "local_variance"]),
        width_optim_iters=30,
        width_mode="full",
        width_optim="gradient",
        standardize=True,
        random_state=42,
    )

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
    return scores.mean()


if __name__ == "__main__":
    # Generate synthetic data
    X, y = make_friedman1(n_samples=500, n_features=5, noise=0.5, random_state=42)

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: erbf_objective(trial, X, y), n_trials=30)

    print(f"\nBest R²: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    # Refit with best params
    bp = study.best_params
    n_rbf = "auto" if bp["n_rbf_auto"] else bp.get("n_rbf", "auto")
    best_model = ERBFRegressor(
        n_rbf=n_rbf,
        alpha=bp["alpha"],
        center_init=bp["center_init"],
        width_init=bp["width_init"],
        width_optim_iters=30,
        width_mode="full",
        width_optim="gradient",
        standardize=True,
        random_state=42,
    )
    best_model.fit(X, y)
    print(f"Train R²: {best_model.score(X, y):.4f}")
    print(f"Centres shape: {best_model.centres_.shape}")
    print(f"Widths shape:  {best_model.widths_.shape}")
