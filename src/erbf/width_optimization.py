"""
Width optimization strategies for ERBFN.

The gradient-based optimizer is the key innovation, providing ~20x speedup over
metaheuristic alternatives for full width mode while achieving better results.
"""

import numpy as np
from .activations import compute_activations


def solve_weights_ridge(activations, y, alpha=1.0):
    """Solve for output weights using Ridge regression.

    Uses direct numpy solve: (Phi^T Phi + alpha I)^-1 Phi^T y
    Much faster than sklearn Ridge for repeated calls.

    Parameters
    ----------
    activations : ndarray of shape (n_samples, n_rbf)
        RBF activation matrix.
    y : ndarray of shape (n_samples,)
        Target values.
    alpha : float
        Regularization strength.

    Returns
    -------
    weights : ndarray of shape (n_rbf,)
        Output layer weights.
    """
    n_rbf = activations.shape[1]
    regularization = alpha * np.eye(n_rbf)
    try:
        weights = np.linalg.solve(
            activations.T @ activations + regularization,
            activations.T @ y
        )
    except np.linalg.LinAlgError:
        # Fallback to least squares if singular
        weights, _, _, _ = np.linalg.lstsq(activations, y, rcond=None)
    return weights


def optimize_widths_gradient(X, y, centres, widths, alpha=1.0,
                             width_mode='full', max_iters=30, verbose=0):
    """Optimize widths using L-BFGS-B with analytical gradients.

    Uses analytical gradient formula:
        dMSE/dw[j,d] = -4*weight[j]/w[j,d]^3 * sum_i residual[i]*Phi[i,j]*diff[i,j,d]^2

    Works in log-space for positivity, so gradient becomes:
        dMSE/dlog_w[j,d] = dMSE/dw[j,d] * w[j,d]

    This is O(n*K*d) per iteration - same as forward pass, enabling
    ~20x speedup over metaheuristics for full width mode.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Training features.
    y : ndarray of shape (n_samples,)
        Training target.
    centres : ndarray of shape (n_rbf, n_features)
        RBF center locations.
    widths : ndarray of shape (n_rbf, n_features)
        Initial width parameters.
    alpha : float
        Ridge regularization strength.
    width_mode : str
        Width parameterization mode:
        - 'full': K*d parameters (per-RBF per-feature)
        - 'shared': d parameters (shared across RBFs)
        - 'rank1': K+d parameters (factorized: w[j,d] = u[j] * v[d])
    max_iters : int
        Maximum L-BFGS-B iterations.
    verbose : int
        Verbosity level.

    Returns
    -------
    widths : ndarray of shape (n_rbf, n_features)
        Optimized width parameters.
    info : dict
        Optimization info (iterations, final MSE, success).
    """
    from scipy.optimize import minimize

    n_samples, n_features = X.shape
    n_rbf = centres.shape[0]

    # Precompute squared differences (fixed during optimization)
    diff = X[:, None, :] - centres[None, :, :]  # (n, K, d)
    sq_diff = diff ** 2

    # Set bounds in log-space
    base_width = np.sqrt(n_features)
    log_lower = np.log(0.1 * base_width)
    log_upper = np.log(10.0 * base_width)

    # Track iterations
    state = {'iter': 0, 'best_mse': np.inf}

    def objective_and_gradient(log_params_flat):
        """Compute MSE and analytical gradient w.r.t. log(widths)."""
        state['iter'] += 1

        # Reconstruct widths based on mode
        if width_mode == 'shared':
            log_w = np.tile(log_params_flat.reshape(1, -1), (n_rbf, 1))
        elif width_mode == 'rank1':
            log_u = log_params_flat[:n_rbf]
            log_v = log_params_flat[n_rbf:]
            log_w = log_u[:, None] + log_v[None, :]
        else:  # full
            log_w = log_params_flat.reshape(n_rbf, n_features)

        w = np.exp(log_w)
        inv_w_sq = 1.0 / (w ** 2)

        # Forward pass: compute activations
        exponent = np.sum(sq_diff * inv_w_sq, axis=2)  # (n, K)
        Phi = np.exp(-exponent)

        # Solve for weights
        weights = solve_weights_ridge(Phi, y, alpha)

        # Compute residuals and MSE
        y_pred = Phi @ weights
        residuals = y - y_pred
        mse = np.mean(residuals ** 2)

        if mse < state['best_mse']:
            state['best_mse'] = mse

        # Analytical gradient
        term = residuals[:, None] * Phi  # (n, K)
        weighted_sq_diff = np.einsum('ij,ijd->jd', term, sq_diff)  # (K, d)
        grad_w = -(4.0 / n_samples) * (weights[:, None] / (w ** 3)) * weighted_sq_diff

        # Chain rule for log-space
        grad_log_w = grad_w * w

        # Reduce gradients based on mode
        if width_mode == 'shared':
            grad_params = np.sum(grad_log_w, axis=0)
        elif width_mode == 'rank1':
            grad_log_u = np.sum(grad_log_w, axis=1)
            grad_log_v = np.sum(grad_log_w, axis=0)
            grad_params = np.concatenate([grad_log_u, grad_log_v])
        else:  # full
            grad_params = grad_log_w.ravel()

        if verbose >= 2 and state['iter'] % 10 == 0:
            print(f"  L-BFGS-B iter {state['iter']}: MSE={mse:.6f}")

        return mse, grad_params

    # Initial parameters in log-space
    if width_mode == 'shared':
        log_params_init = np.log(widths[0, :])
        bounds = [(log_lower, log_upper)] * n_features
    elif width_mode == 'rank1':
        log_w_init = np.log(widths)
        log_u_init = np.mean(log_w_init, axis=1)
        log_v_init = np.mean(log_w_init, axis=0)
        log_params_init = np.concatenate([log_u_init, log_v_init])
        bounds = [(log_lower, log_upper)] * (n_rbf + n_features)
    else:  # full
        log_params_init = np.log(widths).ravel()
        bounds = [(log_lower, log_upper)] * (n_rbf * n_features)

    # Run L-BFGS-B
    result = minimize(
        objective_and_gradient,
        log_params_init,
        method='L-BFGS-B',
        jac=True,
        bounds=bounds,
        options={
            'maxiter': max_iters,
            'ftol': 1e-8,
            'gtol': 1e-6,
            'disp': False
        }
    )

    if verbose >= 1:
        n_params = len(log_params_init)
        print(f"L-BFGS-B ({width_mode}, {n_params} params): "
              f"{state['iter']} evals, MSE={result.fun:.6f}, success={result.success}")

    # Reconstruct widths
    if width_mode == 'shared':
        opt_widths = np.exp(result.x)
        final_widths = np.tile(opt_widths.reshape(1, -1), (n_rbf, 1))
    elif width_mode == 'rank1':
        log_u_opt = result.x[:n_rbf]
        log_v_opt = result.x[n_rbf:]
        u_opt = np.exp(log_u_opt)
        v_opt = np.exp(log_v_opt)
        final_widths = np.outer(u_opt, v_opt)
    else:  # full
        final_widths = np.exp(result.x).reshape(n_rbf, n_features)

    info = {
        'n_iters': state['iter'],
        'final_mse': result.fun,
        'success': result.success
    }

    return final_widths, info


def optimize_widths(X, y, centres, widths, method='gradient',
                    alpha=1.0, width_mode='full', max_iters=30,
                    random_state=None, verbose=0):
    """Optimize widths using the specified method.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Training features.
    y : ndarray of shape (n_samples,)
        Training target.
    centres : ndarray of shape (n_rbf, n_features)
        RBF center locations.
    widths : ndarray of shape (n_rbf, n_features)
        Initial width parameters.
    method : str, default='gradient'
        Optimization method. Currently only 'gradient' is supported.
    alpha : float
        Ridge regularization strength.
    width_mode : str
        'full', 'shared', or 'rank1'.
    max_iters : int
        Maximum iterations.
    random_state : int or None
        Random seed (unused, kept for API compatibility).
    verbose : int
        Verbosity level.

    Returns
    -------
    widths : ndarray of shape (n_rbf, n_features)
        Optimized width parameters.
    info : dict
        Optimization info.
    """
    if method == 'gradient':
        return optimize_widths_gradient(
            X, y, centres, widths,
            alpha=alpha, width_mode=width_mode,
            max_iters=max_iters, verbose=verbose
        )
    else:
        raise ValueError(f"Unknown width optimization method: {method}")
