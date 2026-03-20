"""
solve_global.py

Solves the MPI communication model for all nodes.

All constants imported from constants.py.
"""

import numpy as np
from typing import Tuple, List, Optional

from constants import MSG_SIZES
from lambda_tuning import solve_lasso, auto_tune_lambda


# =============================================================
# Per-node solver
# =============================================================
def solve_per_node(A: np.ndarray,
                   y_n: np.ndarray,
                   lambda_val: float | str = "auto",
                   method: str = "lcurve",
                   n_points: int = 40,
                   solver: str = "CLARABEL",
                   plot: bool = False,
                   plot_dir: str = ".",
                   node_name: str = "") -> Tuple[np.ndarray, float]:
    """
    Solve per-node LASSO with optional automatic lambda tuning.

        min_{x >= 0}  ||x||_1 + lambda * ||Ax - y||_2^2

    Returns
    -------
    x_n         : np.ndarray, shape (2*N_msg,)
                  x_n[0:N_msg]   = send counts per message size bin
                  x_n[N_msg:]    = recv counts per message size bin
    lambda_used : float
    """
    if lambda_val == "auto":
        print(f"  Auto-tuning lambda [{method}]:")
        lambda_used, _ = auto_tune_lambda(
            A, y_n,
            method = method,
            n_points = n_points,
            solver = solver,
            plot = plot,
            plot_dir = plot_dir,
            node_name = node_name
        )
    else:
        lambda_used = float(lambda_val)

    x_n = solve_lasso(A, y_n, lambda_used, solver)
    return x_n, lambda_used


# =============================================================
# Per-node stacked solver (multiple runs)
# =============================================================
def solve_per_node_stacked(A: np.ndarray,
                           Y_runs: np.ndarray,
                           lambda_val: float | str = "auto",
                           method: str = "lcurve",
                           n_points: int = 40,
                           solver: str = "CLARABEL",
                           plot: bool = False,
                           plot_dir: str = ".",
                           node_name: str = "") -> Tuple[np.ndarray, float]:
    """
    Solve per-node LASSO using stacked observations from K runs.

    Parameters
    ----------
    A      : np.ndarray, shape (TWO_M, 2*N_msg)
    Y_runs : np.ndarray, shape (K, TWO_M)
             Y_runs[k, :] = counter vector from run k
    """
    K, two_M = Y_runs.shape

    if two_M != A.shape[0]:
        raise ValueError(f"Y_runs has {two_M} counters per run but A has {A.shape[0]} rows.")

    if K == 1:
        print(f"  Single run ({two_M} obs, {A.shape[1]} unknowns)")
        A_eff = A
        y_eff = Y_runs[0, :]
    else:
        A_eff = np.tile(A, (K, 1))   # (K*TWO_M, 2*N_msg)
        y_eff = Y_runs.flatten()      # (K*TWO_M,)
        print(f"  Stacking {K} runs → "
              f"({K * two_M} obs, {A.shape[1]} unknowns)")

    return solve_per_node(
        A_eff, y_eff,
        lambda_val = lambda_val,
        method = method,
        n_points = n_points,
        solver = solver,
        plot = plot,
        plot_dir = plot_dir,
        node_name = node_name
    )


# =============================================================
# Global solver — single run
# =============================================================
def solve_global(A: np.ndarray,
                 Y: np.ndarray,
                 lambda_val: float | str = "auto",
                 method: str = "lcurve",
                 n_points: int = 40,
                 node_names: Optional[List[str]] = None,
                 solver: str = "CLARABEL",
                 plot: bool = False,
                 plot_dir: str = ".") -> Tuple[np.ndarray, List[float]]:
    """
    Solve global optimization for all nodes (single run).

    Returns
    -------
    X            : np.ndarray, shape (N, 2*N_msg)
    lambdas_used : list of float, length N
    """
    N, two_M = Y.shape

    if two_M != A.shape[0]:
        raise ValueError(f"Y has {two_M} counters per node but A has {A.shape[0]} rows.")

    if node_names is None:
        node_names = [f"node_{n}" for n in range(N)]

    mode_str = (f"auto ({method})" if lambda_val == "auto"
                else f"fixed ({float(lambda_val):.3e})")
    print(f"\nsolve_global: {N} nodes | lambda={mode_str} | solver={solver}")
    print(f"  A : {A.shape}   Y : {Y.shape}   X : ({N}, {A.shape[1]})\n")

    X = np.zeros((N, A.shape[1]))
    lambdas_used = []

    for n in range(N):
        print(f"--- Node [{n+1}/{N}]: {node_names[n]} ---")
        x_n, lam = solve_per_node(
            A, Y[n, :],
            lambda_val = lambda_val,
            method     = method,
            n_points   = n_points,
            solver     = solver,
            plot       = plot,
            plot_dir   = plot_dir,
            node_name  = node_names[n]
        )
        X[n, :] = x_n
        lambdas_used.append(lam)

        active = int(np.sum(x_n > 0.5))
        resid = np.linalg.norm(A @ x_n - Y[n, :])
        print(f"  lambda={lam:.3e}  active_bins={active}  "
              f"residual={resid:.2f}\n")

    return X, lambdas_used


# =============================================================
# Global solver — multi-run stacked
# =============================================================
def solve_global_stacked(A: np.ndarray,
                         Y_multi: np.ndarray,
                         lambda_val: float | str = "auto",
                         method: str = "lcurve",
                         n_points: int = 40,
                         node_names: Optional[List[str]] = None,
                         solver: str = "CLARABEL",
                         plot: bool = False,
                         plot_dir: str = ".") -> Tuple[np.ndarray, List[float]]:
    """Solve global optimization using stacked multi-run observations."""
    N, K, two_M = Y_multi.shape

    if node_names is None:
        node_names = [f"node_{n}" for n in range(N)]

    mode_str = (f"auto ({method})" if lambda_val == "auto"
                else f"fixed ({float(lambda_val):.3e})")
    print(f"\nsolve_global_stacked: {N} nodes, {K} runs | "
          f"lambda={mode_str} | solver={solver}\n")

    X = np.zeros((N, A.shape[1]))
    lambdas_used = []

    for n in range(N):
        print(f"--- Node [{n+1}/{N}]: {node_names[n]} ---")
        x_n, lam = solve_per_node_stacked(
            A,
            Y_runs = Y_multi[n, :, :],
            lambda_val = lambda_val,
            method = method,
            n_points = n_points,
            solver = solver,
            plot = plot,
            plot_dir = plot_dir,
            node_name = node_names[n]
        )
        X[n, :]  = x_n
        lambdas_used.append(lam)

        active = int(np.sum(x_n > 0.5))
        print(f"  lambda={lam:.3e}  active_bins={active}\n")

    return X, lambdas_used


# =============================================================
# Post-processing
# =============================================================
def extract_send_recv(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split X into send and recv distributions."""
    if X.shape[1] % 2 != 0:
        raise ValueError(
            f"X has {X.shape[1]} columns — expected an even number (2*n_msg)."
        )
    n_msg = X.shape[1] // 2
    return X[:, :n_msg], X[:, n_msg:]


def print_solution_summary(A: np.ndarray,
                           X: np.ndarray,
                           Y: np.ndarray,
                           node_names: List[str],
                           msg_sizes: np.ndarray = MSG_SIZES) -> None:
    """Print per-node solution quality and active message bins."""
    N, two_N = X.shape
    n_msg = two_N // 2

    print("\n=== Solution Summary ===")
    print(f"  {'Node':<20} {'Active bins':>11} "
          f"{'Residual':>10} {'Rel. residual':>14}")
    print("  " + "-" * 58)

    for n in range(N):
        x_n = X[n, :]
        residual = np.linalg.norm(A @ x_n - Y[n, :])
        y_norm = np.linalg.norm(Y[n, :])
        rel_resid = residual / y_norm if y_norm > 0 else 0.0
        active = int(np.sum(x_n > 0.5))
        print(f"  {node_names[n]:<20} {active:>11} "
              f"{residual:>10.2f} {rel_resid:>13.2%}")

    print()
    print("Active message sizes:")
    for n in range(N):
        x_send = X[n, :n_msg]
        x_recv = X[n, n_msg:]
        print(f"  {node_names[n]}:")
        for j in np.where(x_send > 0.5)[0]:
            m = msg_sizes[j]
            size_str = f"{int(m/1024)}KB" if m >= 1024 else f"{int(m)}B"
            print(f"    Send bin {j:>2} ({size_str:>8}): {x_send[j]:.1f}")
        for j in np.where(x_recv > 0.5)[0]:
            m = msg_sizes[j]
            size_str = f"{int(m/1024)}KB" if m >= 1024 else f"{int(m)}B"
            print(f"    Recv bin {j:>2} ({size_str:>8}): {x_recv[j]:.1f}")


def print_lambda_summary(node_names: List[str],
                         lambdas_used: List[float],
                         X: np.ndarray) -> None:
    """Print per-node lambda selection summary."""
    if X.shape[1] % 2 != 0:
        raise ValueError(
            f"X has {X.shape[1]} columns — expected an even number (2*n_msg)."
        )
    n_msg = X.shape[1] // 2

    print("\n=== Lambda Summary ===")
    print(f"  {'Node':<20} {'lambda':>12} {'active_bins':>12} "
          f"{'total_sends':>12} {'total_recvs':>12}")
    print("  " + "-" * 70)
    for n, (name, lam) in enumerate(zip(node_names, lambdas_used)):
        x_send = X[n, :n_msg]
        x_recv = X[n, n_msg:]
        active   = int(np.sum(X[n, :] > 0.5))
        tot_send = int(np.sum(x_send))
        tot_recv = int(np.sum(x_recv))
        print(f"  {name:<20} {lam:>12.3e} {active:>12} "
              f"{tot_send:>12} {tot_recv:>12}")

        
