"""
lambda_tuning.py

Automatic lambda selection for the per-node LASSO problem:

    min_{x >= 0}  ||x||_1 + lambda * ||Ax - y||_2^2

All constants imported from constants.py.
"""

import numpy as np
import cvxpy as cp
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Dict


# =============================================================
# Core LASSO solver
# =============================================================
def solve_lasso(A: np.ndarray,
                y: np.ndarray,
                lambda_val: float,
                solver: str = "CLARABEL") -> np.ndarray:
    """
    Solve one non-negative LASSO instance:

        min_{x >= 0}  ||x||_1 + lambda * ||Ax - y||_2^2

    Returns
    -------
    x : np.ndarray, shape (n_unknowns,)
    """
    n = A.shape[1]
    x = cp.Variable(n, nonneg=True)

    prob = cp.Problem(cp.Minimize(cp.norm1(x) + lambda_val * cp.sum_squares(A @ x - y)))

    try:
        prob.solve(solver=solver, verbose=False)
    except cp.SolverError:
        warnings.warn(f"Solver {solver} failed, falling back to SCS.")
        try:
            prob.solve(solver="SCS", verbose=False)
        except cp.SolverError:
            warnings.warn("SCS also failed. Returning zeros.")
            return np.zeros(n)

    if x.value is None:
        warnings.warn("Solver returned None. Returning zeros.")
        return np.zeros(n)

    return np.clip(x.value, 0, None)


# =============================================================
# Lambda zero
# =============================================================
def compute_lambda_zero(A: np.ndarray, y: np.ndarray) -> float:
    """
    Compute the smallest lambda above which x becomes non-zero.

        lambda_zero = 1 / (2 * max_j (A^T y)_j)

    Returns
    -------
    lambda_zero : float
    """
    ATy_pos = np.maximum(A.T @ y, 0)
    max_val = np.max(ATy_pos)

    if max_val == 0:
        warnings.warn("A^T y has no positive entries — y may be all zeros.")
        return 1e-6

    return 1.0 / (2.0 * max_val)


# =============================================================
# L-curve method
# =============================================================
def l_curve_method(A: np.ndarray,
                   y: np.ndarray,
                   n_points: int = 40,
                   lambda_min_factor: float = 1.5,
                   lambda_max_factor: float = 1e5,
                   solver: str = "CLARABEL",
                   plot: bool = False,
                   plot_dir: str = ".",
                   node_name: str = "") -> Tuple[float, Dict]:
    """
    Automatic lambda selection using the L-curve method.

    Returns
    -------
    lambda_opt : float
    info       : dict
        Keys: method, lambda_zero, lambda_grid, l1_norms,
              residuals, corner_idx
    """
    lambda_zero = compute_lambda_zero(A, y)
    lambda_grid = np.logspace(
        np.log10(lambda_zero * lambda_min_factor),
        np.log10(lambda_zero * lambda_max_factor),
        n_points
    )

    print(f"    lambda_zero = {lambda_zero:.3e}")
    print(f"    lambda grid : [{lambda_grid[0]:.3e}, {lambda_grid[-1]:.3e}]"
          f"  ({n_points} points)")

    l1_norms  = np.zeros(n_points)
    residuals = np.zeros(n_points)

    for i, lam in enumerate(lambda_grid):
        x_i = solve_lasso(A, y, lam, solver)
        l1_norms[i] = np.sum(x_i)
        residuals[i] = np.linalg.norm(A @ x_i - y)

    lambda_opt, corner_idx = _find_lcurve_corner(lambda_grid, l1_norms, residuals)

    print(f"    lambda_opt  = {lambda_opt:.3e}  "
          f"(index {corner_idx}/{n_points-1})")
    print(f"    ||x||_1     = {l1_norms[corner_idx]:.2f}  "
          f"||Ax-y||_2 = {residuals[corner_idx]:.2f}")

    info = {
        "method"      : "lcurve",
        "lambda_zero" : lambda_zero,
        "lambda_grid" : lambda_grid,
        "l1_norms"    : l1_norms,
        "residuals"   : residuals,
        "corner_idx"  : corner_idx,
    }

    if plot:
        _plot_lcurve(info, lambda_opt, node_name, plot_dir)

    return lambda_opt, info


def _find_lcurve_corner(lambda_grid: np.ndarray, 
                        l1_norms: np.ndarray, 
                        residuals: np.ndarray) -> Tuple[float, int]:
    """Find L-curve corner via maximum perpendicular distance."""
    log_res = np.log10(residuals + 1e-30)
    log_l1 = np.log10(l1_norms  + 1e-30)

    def normalize(v: np.ndarray) -> np.ndarray:
        r = v.max() - v.min()
        return (v - v.min()) / r if r > 1e-12 else np.zeros_like(v)

    log_res_n = normalize(log_res)
    log_l1_n  = normalize(log_l1)

    x0, y0 = log_res_n[0], log_l1_n[0]
    x1, y1 = log_res_n[-1], log_l1_n[-1]
    dx, dy = x1 - x0, y1 - y0
    denom = np.sqrt(dx**2 + dy**2) + 1e-30

    distances = np.abs(dy * log_res_n - dx * log_l1_n + x1 * y0 - y1 * x0) / denom

    corner_idx = int(np.argmax(distances))
    return float(lambda_grid[corner_idx]), corner_idx


# =============================================================
# LOCO-CV method
# =============================================================
def loco_cv_method(A: np.ndarray,
                   y: np.ndarray,
                   n_points: int = 30,
                   lambda_min_factor: float = 1.5,
                   lambda_max_factor: float = 1e5,
                   solver: str = "CLARABEL",
                   node_name: str = "") -> Tuple[float, Dict]:
    """
    Automatic lambda selection via Leave-One-Counter-Out CV.

    Returns
    -------
    lambda_opt : float
    info       : dict
        Keys: method, lambda_zero, lambda_grid, cv_errors, best_idx
    """
    two_M = len(y)
    lambda_zero = compute_lambda_zero(A, y)
    lambda_grid = np.logspace(
        np.log10(lambda_zero * lambda_min_factor),
        np.log10(lambda_zero * lambda_max_factor),
        n_points
    )

    print(f"    lambda_zero = {lambda_zero:.3e}")
    print(f"    lambda grid : [{lambda_grid[0]:.3e}, {lambda_grid[-1]:.3e}]"
          f"  ({n_points} points, {two_M} folds each)")

    cv_errors = np.zeros(n_points)

    for i, lam in enumerate(lambda_grid):
        fold_errors = np.zeros(two_M)
        for k in range(two_M):
            mask = np.ones(two_M, dtype=bool)
            mask[k] = False
            x_k = solve_lasso(A[mask, :], y[mask], lam, solver)
            fold_errors[k] = (A[k, :] @ x_k - y[k]) ** 2
        cv_errors[i] = np.mean(fold_errors)

    best_idx   = int(np.argmin(cv_errors))
    lambda_opt = float(lambda_grid[best_idx])

    print(f"    lambda_opt = {lambda_opt:.3e}  "
          f"(index {best_idx}/{n_points-1})")
    print(f"    CV error   = {cv_errors[best_idx]:.4f}")

    info = {
        "method"      : "loco_cv",
        "lambda_zero" : lambda_zero,
        "lambda_grid" : lambda_grid,
        "cv_errors"   : cv_errors,
        "best_idx"    : best_idx,
    }

    return lambda_opt, info


# =============================================================
# Unified interface
# =============================================================
def auto_tune_lambda(A: np.ndarray,
                     y: np.ndarray,
                     method: str = "lcurve",
                     n_points: int = 40,
                     solver: str = "CLARABEL",
                     plot: bool = False,
                     plot_dir: str = ".",
                     node_name: str = "") -> Tuple[float, Dict]:
    """
    Automatically select lambda for one node.

    Returns
    -------
    lambda_opt : float
    info       : dict
    """
    if method == "lcurve":
        return l_curve_method(
            A, y,
            n_points  = n_points,
            solver    = solver,
            plot      = plot,
            plot_dir  = plot_dir,
            node_name = node_name
        )
    elif method == "loco_cv":
        return loco_cv_method(
            A, y,
            n_points  = n_points,
            solver    = solver,
            node_name = node_name
        )
    else:
        raise ValueError(
            f"Unknown method '{method}'. Choose 'lcurve' or 'loco_cv'."
        )


# =============================================================
# Plot helpers
# =============================================================
def _plot_lcurve(info: Dict,
                 lambda_opt: float,
                 node_name: str,
                 plot_dir: str = ".") -> None:
    """Save L-curve and tradeoff plots for one node."""
    Path(plot_dir).mkdir(parents=True, exist_ok=True)

    lambda_grid = info["lambda_grid"]
    l1_norms = info["l1_norms"]
    residuals = info["residuals"]
    corner_idx = info["corner_idx"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.loglog(residuals, l1_norms, "b.-", linewidth=1.5, markersize=4)
    ax.loglog(residuals[corner_idx], l1_norms[corner_idx],
              "ro", markersize=10, label=f"Corner  λ={lambda_opt:.2e}")
    ax.set_xlabel(r"Residual $\|Ax - y\|_2$")
    ax.set_ylabel(r"Sparsity $\|x\|_1$")
    ax.set_title(f"L-curve  [{node_name}]")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax  = axes[1]
    ax2 = ax.twinx()
    ax.semilogx(lambda_grid, l1_norms,  "b.-", linewidth=1.5,
                markersize=4, label=r"$\|x\|_1$")
    ax2.semilogx(lambda_grid, residuals, "g.-", linewidth=1.5,
                 markersize=4, label=r"$\|Ax-y\|_2$")
    ax.axvline(lambda_opt, color="r", linestyle="--",
               label=f"λ_opt={lambda_opt:.2e}")
    ax.set_xlabel("λ")
    ax.set_ylabel(r"$\|x\|_1$",     color="b")
    ax2.set_ylabel(r"$\|Ax-y\|_2$", color="g")
    ax.set_title(f"Tradeoff  [{node_name}]")
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    outfile = str(Path(plot_dir) / f"lcurve_{node_name}.png")
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    L-curve saved: {outfile}")