"""
solve_global.py

Solves the MPI communication model for all nodes.

All constants imported from constants.py.
"""
import numpy as np
from typing import Tuple, List, Optional, Dict
import warnings
import cvxpy as cp


# =============================================================
# Global solver — single run
# =============================================================
def solve_global(A: np.ndarray,
                 Y: np.ndarray,
                 node_names: Optional[List[str]] = None) -> Tuple[np.ndarray, List[float]]:
    """
    Solve global optimization for all nodes (single run).

    Returns
    -------
    X            : np.ndarray, shape (num_nodes, 2*num_msg_size_bins)
    lambdas_used : list of float, length num_nodes
    """
    num_nodes, all_counters_txrx = Y.shape

    if all_counters_txrx != A.shape[0]:
        raise ValueError(f"Y has {all_counters_txrx} counters per node but A has {A.shape[0]} rows.")

    if node_names is None:
        node_names = [f"node_{n}" for n in range(num_nodes)]

    print(f"  A : {A.shape}   Y : {Y.shape}   X : ({num_nodes}, {A.shape[1]})\n")

    X = np.zeros((num_nodes, A.shape[1]))

    lambdas_used = []

    for node_idx in range(num_nodes):
        print(f"--- Node [{node_idx+1}/{num_nodes}]: {node_names[node_idx]} ---")

        # solve per-node LASSO with automatic lambda tuning
        print(f"  Auto-tuning lambda:")
        y_nid = Y[node_idx, :]
        lam = loco_cv_method(A, y_nid)
        x_nid = solve_lasso(A, y_nid, lam)
        X[node_idx, :] = x_nid
        lambdas_used.append(lam)

        # active_bins counts how many elements of x_nid are greater than 0.5
        active_bins = int(np.sum(x_nid > 0.5))
        # residual is ||Ax - y||, measuring how well the solution fits the data for this node.
        residual = np.linalg.norm(A @ x_nid - y_nid)
        print(f"  lambda={lam:.3e} active_bins={active_bins} residual={residual:.2f}\n")

    return X, lambdas_used


# =============================================================
# Leave-One-Counter-Out Cross-Validation (LOCO-CV) for lambda selection
# =============================================================
def loco_cv_method(A: np.ndarray, y: np.ndarray) -> Tuple[float, Dict]:
    """
    Automatic lambda selection via Leave-One-Counter-Out CV.

    Returns
    -------
    lambda_opt : float
    info       : dict
        Keys: method, lambda_zero, lambda_grid, cv_errors, best_idx
    """
    two_num_all_cntrs = len(y)
    
    lam_n_points = 40
    lam_min_factor = 1e-4
    lam_max_factor = 1.0

    # lambda_zero is the starting point for lambda grid search
    lambda_baseline = compute_lambda_baseline(A, y)
    lambda_grid = np.logspace(
        np.log10(lambda_baseline * lam_min_factor),
        np.log10(lambda_baseline * lam_max_factor),
        lam_n_points
    )

    print(f"    lambda_baseline = {lambda_baseline:.3e}")
    print(f"    lambda grid : [{lambda_grid[0]:.3e}, {lambda_grid[-1]:.3e}]"
          f"  ({lam_n_points} points, {two_num_all_cntrs} folds each)")

    cv_errors = np.zeros(lam_n_points)

    for i, lam in enumerate(lambda_grid):
        # initialize array to hold errors for each fold
        fold_errors = np.zeros(two_num_all_cntrs)

        # LOCO-CV: For each counter k, leave it out, solve LASSO on the remaining counters
        # and compute the squared error on the left-out counter k.
        for k in range(two_num_all_cntrs):
            mask = np.ones(two_num_all_cntrs, dtype=bool)
            # False means "exclude this element" using the boolean mask
            mask[k] = False
            x_k = solve_lasso(A[mask, :], y[mask], lam)
            fold_errors[k] = (A[k, :] @ x_k - y[k]) ** 2

        cv_errors[i] = np.mean(fold_errors)

    # find the lambda with the lowest CV error
    best_idx = int(np.argmin(cv_errors))
    lambda_opt = float(lambda_grid[best_idx])

    print(f"    lambda_opt = {lambda_opt:.3e}  "
          f"(index {best_idx}/{lam_n_points-1})")
    print(f"    CV error   = {cv_errors[best_idx]:.4f}")

    return lambda_opt


# =============================================================
# Core LASSO solver
# =============================================================
def solve_lasso(A: np.ndarray, y: np.ndarray, lambda_val: float) -> np.ndarray:
    """
    Solve one non-negative LASSO instance:

        min_{x >= 0}  ||x||_1 + lambda * ||Ax - y||_2^2
    """
    two_num_msg_sizes = A.shape[1]
    x = cp.Variable(two_num_msg_sizes, nonneg=True)

    # Note: Because x >= 0, the L1 norm ||x||_1 simplifies completely to the sum of x.
    # Defining it as cp.sum(x) reformulates the setup as a pure Quadratic Program (QP) 
    # instead of a Second-Order Cone Program (SOCP), which is far faster for solvers.
    prob = cp.Problem(cp.Minimize(lambda_val * cp.sum(x) + cp.sum_squares(A @ x - y)))

    try:
        prob.solve(solver="CLARABEL", verbose=False)
    except cp.SolverError:
        warnings.warn("CLARABEL also failed. Returning zeros.")
        return np.zeros(two_num_msg_sizes)
            
    if x.value is None:
        warnings.warn("Solver returned None. Returning zeros.")
        return np.zeros(two_num_msg_sizes)

    # make sure to clip small negative values to zero
    return np.clip(x.value, 0, None)


# =============================================================
# Mathematically grounded starting point
# =============================================================
def compute_lambda_baseline(A: np.ndarray, y: np.ndarray) -> float:
    """
    Compute the baseline lambda value for lambda grid search.
    lambda_baseline = 2 * max_j (A^T y)_j
    """
    # A.T @ y is essentially asking: 
    # "which features are most correlated with what I'm trying to predict?"
    max_val = np.max(np.abs(A.T @ y))
    if max_val == 0:
        warnings.warn("A^T y has no positive entries — y may be all zeros.")
        return 1e-6

    return 2.0 * max_val


# =============================================================
# Utility function to print solution summary
# =============================================================
def print_solution_summary(node_names: List[str],
                           lambdas_used: List[float],
                           X: np.ndarray,
                           msg_size_sets: List[int]) -> None:
    """Print per-node lambda selection summary."""
    if X.shape[1] % 2 != 0:
        raise ValueError(f"X has {X.shape[1]} columns — expected an even number.")
    
    num_msg_sizes = X.shape[1] // 2

    print("\n=== Lambda Summary ===")
    print(f"  {'Node':<20} {'lambda':>12} {'active_bins':>12} "
          f"{'total_sends':>12} {'total_recvs':>12}")
    print("  " + "-" * 70)
    for node_idx, (name, lam) in enumerate(zip(node_names, lambdas_used)):
        x_send = X[node_idx, :num_msg_sizes]
        x_recv = X[node_idx, num_msg_sizes:]
        active   = int(np.sum(X[node_idx, :] > 0.5))
        tot_send = int(np.sum(x_send))
        tot_recv = int(np.sum(x_recv))
        print(f"  {name:<20} {lam:>12.3e} {active:>12} "
              f"{tot_send:>12} {tot_recv:>12}")
        
        # Active bin indices and values (0-based)
        active_send_bins = np.where(x_send > 0.5)[0].tolist()
        active_recv_bins = np.where(x_recv > 0.5)[0].tolist()

        if active_send_bins:
            send_str = ", ".join([f"{msg_size_sets[i]}: {x_send[i]:.2f}" for i in active_send_bins])
            print(f"    send bins : [{send_str}]")
        else:
            print(f"    send bins : (none)")

        if active_recv_bins:
            recv_str = ", ".join([f"{msg_size_sets[i]}: {x_recv[i]:.2f}" for i in active_recv_bins])
            print(f"    recv bins : [{recv_str}]")
        else:
            print(f"    recv bins : (none)")