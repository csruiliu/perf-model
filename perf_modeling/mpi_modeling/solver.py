"""
solve_global.py

Solves the MPI communication model for all nodes.

All constants imported from constants.py.
"""
import numpy as np
from typing import Tuple, List, Optional
import warnings
import cvxpy as cp
from scipy.optimize import nnls


# =============================================================
# Global solver — single run
# =============================================================
def solve_global(A: np.ndarray,
                 Y: np.ndarray,
                 node_names: Optional[List[str]] = None) -> Tuple[np.ndarray, List[float]]:
    """
    Solve global constrained optimization for all nodes (single run).

        min_{x >= 0}  ||x||_1
        s.t.          ||Ax - y||_2^2 <= epsilon

    Returns
    -------
    X            : np.ndarray, shape (num_nodes, 2*num_msg_size_bins)
    lambda_used_list : list of float, length num_nodes
    """
    num_nodes, all_counters_txrx = Y.shape

    if all_counters_txrx != A.shape[0]:
        raise ValueError(f"Y has {all_counters_txrx} counters per node but A has {A.shape[0]} rows.")

    if node_names is None:
        node_names = [f"node_{n}" for n in range(num_nodes)]

    print(f"  A : {A.shape}   Y : {Y.shape}   X : ({num_nodes}, {A.shape[1]})\n")

    X = np.zeros((num_nodes, A.shape[1]))
    lambda_used_list = []

    for node_idx in range(num_nodes):
        print(f"--- Node [{node_idx+1}/{num_nodes}]: {node_names[node_idx]} ---")

        y_nid = Y[node_idx, :]
        print(f"  Auto-tuning lambda (residual tolerance):")
        lam = find_lambda_cv(A, y_nid)
        x_nid = solve_constrained_optimization(A, y_nid, lam)
        X[node_idx, :] = x_nid
        lambda_used_list.append(lam)

        active_bins = int(np.sum(x_nid > 0.5))
        residual = np.linalg.norm(A @ x_nid - y_nid)
        print(f"  lambda={lam:.3e}  active_bins={active_bins}  residual={residual:.2f}\n")

    return X, lambda_used_list


# =============================================================
# Lambda selection via Leave-One-Counter-Out Cross Validation
# =============================================================
def find_lambda_cv(A: np.ndarray, y: np.ndarray) -> float:
    """
    Automatic lambda selection via Leave-One-Counter-Out CV.

    Returns
    -------
    lambda_opt : float
    """
    m = len(y)
    
    lam_n_points = 50

    lam_min, lam_balance = compute_lambda_baseline(A, y)

    # Grid from the KKT transition point to well above the balance point
    lam_lo = lam_min
    lam_hi = lam_balance * 1000.0
    lambda_grid = np.logspace(
        np.log10(lam_lo),
        np.log10(lam_hi),
        lam_n_points
    )

    print(f"    lam_min = {lam_min:.3e}")
    print(f"    lam_balance = {lam_balance:.3e}")
    print(f"    lambda grid : [{lambda_grid[0]:.3e}, {lambda_grid[-1]:.3e}]"
          f"  ({lam_n_points} points, {m} folds each)")

    cv_errors = np.zeros(lam_n_points)

    for i, lam in enumerate(lambda_grid):
        # initialize array to hold errors for each fold
        fold_errors = np.zeros(m)

        # LOCO-CV: For each counter k, leave it out, solve LASSO on the remaining counters
        # and compute the squared error on the left-out counter k.
        for k in range(m):
            mask = np.ones(m, dtype=bool)
            # False means "exclude this element" using the boolean mask
            mask[k] = False
            x_k = solve_constrained_optimization(A[mask, :], y[mask], lam)
            fold_errors[k] = (A[k, :] @ x_k - y[k]) ** 2

        cv_errors[i] = np.mean(fold_errors)

    # find the lambda with the lowest CV error
    best_idx = int(np.argmin(cv_errors))
    lambda_opt = float(lambda_grid[best_idx])

    # Warn if the optimum is at a grid boundary — grid may be too narrow
    if best_idx == 0:
        warnings.warn(
            f"Optimal lambda is at the lower grid boundary ({lambda_opt:.3e}). "
            f"Consider extending the grid downward."
        )
    elif best_idx == lam_n_points - 1:
        warnings.warn(
            f"Optimal lambda is at the upper grid boundary ({lambda_opt:.3e}). "
            f"Consider extending the grid upward."
        )

    print(f"    lambda_opt = {lambda_opt:.3e}  (index {best_idx}/{lam_n_points-1})")
    print(f"    CV error   = {cv_errors[best_idx]:.4f}")

    return lambda_opt


# =============================================================
# Core constrained optimization solver
# =============================================================
def solve_constrained_optimization(A: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """
    Solve the constrained optimization problem:

        min_{x >= 0}  ||x||_1
        s.t.          ||Ax - y||_2^2 <= epsilon

    The above is equivalent to the LASSO formulation:

        min_{x >= 0}  ||x||_1 + lambda * ||Ax - y||_2^2

    Since x >= 0, ||x||_1 = sum(x), so the objective simplifies to:

        min_{x >= 0}  sum(x) + lambda * ||Ax - y||_2^2

    Minimizing the L1 norm encourages sparsity. The squared L2 term
    penalizes the residual between the reconstructed vector Ax and the
    observed vector y. lambda controls the trade-off between sparsity
    and reconstruction accuracy.

    Args:
        A       : System matrix (shape: m x n)
        y       : Observation vector (shape: m,)
        lam     : Regularization parameter.
                  Larger lam → tighter fit, denser x.
                  Smaller lam → looser fit, sparser x.
    """
    n = A.shape[1]
    x = cp.Variable(n, nonneg=True)  # Non-negative variable for message size bins

    # Since x >= 0, ||x||_1 = sum(x), so we can simplify the objective
    objective = cp.Minimize(cp.sum(x) + lam * cp.sum_squares(A @ x - y))

    prob = cp.Problem(objective)  # No explicit constraints since nonneg is handled by the variable

    # Try multiple solvers in order of preference
    solver_list = ["CLARABEL", "SCS", "ECOS", "OSQP", "CVXOPT"]

    for solver in solver_list:
        try:
            prob.solve(solver=solver, verbose=False)
            if x.value is not None and prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                break
            warnings.warn(f"Solver {solver} returned None or failed to converge. Trying the next solver.")
        except cp.SolverError:
            warnings.warn(f"Solver {solver} failed. Trying the next solver.")

    if x.value is None:
        warnings.warn("All solvers failed. Returning zeros.")
        return np.zeros(n)

    return x.value


# =============================================================
# Mathematically grounded starting point
# =============================================================
def compute_lambda_baseline(A: np.ndarray, y: np.ndarray) -> float:
    """
    Compute principled lambda search bounds using KKT theory and NNLS.

    lam_min     : Transition point from KKT conditions. Below this, x=0 is
                  optimal. lam_min = 1 / (2 * max_j (A^T y)_j)

    lam_balance : Lambda where L1 and residual terms are equally weighted,
                  estimated from the NNLS solution.
                  lam_balance = ||x_nnls||_1 / ||Ax_nnls - y||_2^2

    Returns
    -------
    lam_min     : float
    lam_balance : float
    """
    # A.T @ y is essentially asking: 
    # "which features are most correlated with what I'm trying to predict?"
    aty = A.T @ y
    max_aty = np.max(aty[aty > 0]) if np.any(aty > 0) else 0.0
    
    if max_aty == 0:
        warnings.warn("A^T y has no positive entries — y may be all zeros.")
        lam_min = 1e-6
    else:
        lam_min = 1.0 / (2.0 * max_aty)
    
    # NNLS gives the best-fit x >= 0, used to calibrate the upper grid bound
    x_nnls, res_norm = nnls(A, y)   # res_norm = ||Ax_nnls - y||_2
    res_sq   = res_norm ** 2
    l1_nnls  = np.sum(x_nnls)

    if res_sq > 0 and l1_nnls > 0:
        lam_balance = l1_nnls / res_sq
    else:
        warnings.warn("NNLS solution is degenerate. Falling back to lam_min * 1e4.")
        lam_balance = lam_min * 1e4

    return lam_min, lam_balance


# =============================================================
# Utility function to print solution summary
# =============================================================
def print_solution_summary(node_names: List[str],
                           lambda_used: List[float],
                           X: np.ndarray,
                           msg_size_sets: List[int]) -> None:
    """Print per-node lambda selection and solution summary."""
    if X.shape[1] % 2 != 0:
        raise ValueError(f"X has {X.shape[1]} columns — expected an even number.")

    num_msg_sizes = X.shape[1] // 2

    print("\n=== Residual Tolerance Summary ===")
    print(f"  {'Node':<20} {'lambda':>12} {'active_bins':>12} {'total_sends':>12} {'total_recvs':>12}")
    print("  " + "-" * 70)

    for node_idx, (name, lambda_val) in enumerate(zip(node_names, lambda_used)):
        x_send = X[node_idx, :num_msg_sizes]
        x_recv = X[node_idx, num_msg_sizes:]
        active   = int(np.sum(X[node_idx, :] > 0.5))
        tot_send = int(np.sum(x_send))
        tot_recv = int(np.sum(x_recv))

        print(f"  {name:<20} {lambda_val:>12.3e} {active:>12} {tot_send:>12} {tot_recv:>12}")

        active_send_bins = np.where(x_send > 0.5)[0].tolist()
        active_recv_bins = np.where(x_recv > 0.5)[0].tolist()

        if active_send_bins or active_recv_bins:
            send_pairs = [f"{msg_size_sets[i]}: {x_send[i]:.2f}" for i in active_send_bins]
            recv_pairs = [f"{msg_size_sets[i]}: {x_recv[i]:.2f}" for i in active_recv_bins]

            max_len = max(len(send_pairs), len(recv_pairs))
            send_pairs += [""] * (max_len - len(send_pairs))
            recv_pairs += [""] * (max_len - len(recv_pairs))

            pad = max(len(s) for s in send_pairs) + 2 if send_pairs else 0

            print(f"    bins :")
            for s, r in zip(send_pairs, recv_pairs):
                print(f"        send: {s:<{pad}} recv: {r}")
        else:
            print(f"    bins : (none)")