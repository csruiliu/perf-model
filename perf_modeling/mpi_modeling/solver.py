"""
solver.py

Solves the MPI communication model for all nodes.

All constants imported from constants.py.
"""

import warnings

import cvxpy as cp
import numpy as np
from scipy.optimize import nnls


# =============================================================
# Global solver — single run
# =============================================================
def solve_global(
    matrix_a: np.ndarray, vec_y: np.ndarray, node_names: list[str] | None = None
) -> tuple[np.ndarray, list[float]]:
    """
    Solve global constrained optimization for all nodes (single run).

        min_{x >= 0}  ||x||_1
        s.t.          ||Ax - y||_2^2 <= epsilon

    Returns
    -------
    X            : np.ndarray, shape (num_nodes, 2*num_msg_size_bins)
    lambda_used_list : list of float, length num_nodes
    """
    num_nodes, all_counters_txrx = vec_y.shape

    if all_counters_txrx != matrix_a.shape[0]:
        raise ValueError(
            f"Y has {all_counters_txrx} counters per node but A has {matrix_a.shape[0]} rows."
        )

    if node_names is None:
        node_names = [f"node_{n}" for n in range(num_nodes)]

    print(f"  A : {matrix_a.shape}   Y : {vec_y.shape}   X : ({num_nodes}, {matrix_a.shape[1]})\n")

    vec_x = np.zeros((num_nodes, matrix_a.shape[1]))
    lambda_used_list = []

    for node_idx in range(num_nodes):
        print(f"--- Node [{node_idx + 1}/{num_nodes}]: {node_names[node_idx]} ---")

        y_nid = vec_y[node_idx, :]
        print("  Auto-tuning lambda (residual tolerance):")
        lam = find_lambda_cv(matrix_a, y_nid, max_extensions=5)
        x_nid = solve_constrained_optimization(matrix_a, y_nid, lam)
        vec_x[node_idx, :] = x_nid
        lambda_used_list.append(lam)

        active_bins = int(np.sum(x_nid > 0.5))
        residual = np.linalg.norm(matrix_a @ x_nid - y_nid)
        print(f"  lambda={lam:.3e}  active_bins={active_bins}  residual={residual:.2f}\n")

    return vec_x, lambda_used_list


# =============================================================
# Lambda selection via Leave-One-Counter-Out Cross Validation
# =============================================================
def find_lambda_cv(matrix_a: np.ndarray, vec_y: np.ndarray, max_extensions: int = 5) -> float:
    """
    Automatic lambda selection via Leave-One-Counter-Out CV.

    If the optimal lambda lands on a grid boundary, the grid is automatically
    extended in that direction and CV is re-run on the new region only.
    Results are accumulated across all extensions and the global best is returned.

    Parameters
    ----------
    A : System matrix (shape: m x n)
    y : Observation vector (shape: m,)
    max_extensions : Maximum number of grid extensions before giving up (default: 5)

    Returns
    -------
    lambda_opt : float
    """
    m = len(vec_y)
    lam_n_points = 50
    extend_factor = 10.0

    lam_min, lam_balance = compute_lambda_baseline(matrix_a, vec_y)
    lam_lo = lam_min
    lam_hi = lam_balance * 10.0

    print(f"    lam_min (KKT transition) = {lam_min:.3e}")
    print(f"    lam_balance (NNLS)       = {lam_balance:.3e}")

    # Accumulate all (lambda, cv_error) pairs across extensions
    # so we can find the global best at the end
    all_lambdas = []
    all_cv_errors = []

    found_interior = False

    for attempt in range(max_extensions + 1):
        lambda_grid = np.logspace(np.log10(lam_lo), np.log10(lam_hi), lam_n_points)

        print(
            f"    Attempt {attempt + 1}: searching "
            f"[{lambda_grid[0]:.3e}, {lambda_grid[-1]:.3e}] "
            f"({lam_n_points} points, {m} LOCO folds each)"
        )

        # Run LOCO-CV on this grid segment
        cv_errors = np.zeros(lam_n_points)
        for i, lam in enumerate(lambda_grid):
            fold_errors = np.zeros(m)
            for k in range(m):
                mask = np.ones(m, dtype=bool)
                mask[k] = False
                x_k = solve_constrained_optimization(matrix_a[mask, :], vec_y[mask], lam)
                fold_errors[k] = (matrix_a[k, :] @ x_k - vec_y[k]) ** 2
            cv_errors[i] = np.mean(fold_errors)

        # Accumulate results
        all_lambdas.extend(lambda_grid.tolist())
        all_cv_errors.extend(cv_errors.tolist())

        local_best_idx = int(np.argmin(cv_errors))
        at_lower = local_best_idx == 0
        at_upper = local_best_idx == lam_n_points - 1

        if not at_lower and not at_upper:
            # Optimum is interior — no need to extend further
            found_interior = True
            break

        if attempt < max_extensions:
            if at_lower:
                print(
                    f"    Optimal at lower boundary ({lambda_grid[0]:.3e}). "
                    f"Extending grid downward by factor {extend_factor}."
                )
                # Shift the search window downward, no overlap with current window
                lam_hi = lam_lo
                lam_lo = lam_lo / extend_factor
            else:
                print(
                    f"    Optimal at upper boundary ({lambda_grid[-1]:.3e}). "
                    f"Extending grid upward by factor {extend_factor}."
                )
                # Shift the search window upward, no overlap with current window
                lam_lo = lam_hi
                lam_hi = lam_hi * extend_factor

    if not found_interior:
        warnings.warn(
            f"Grid extension limit ({max_extensions}) reached. "
            f"lambda_opt may not be the true optimum. "
            f"Consider increasing max_extensions or checking your data.",
            stacklevel=2,
        )

    # Find global best across all accumulated searches
    all_lambdas_arr = np.array(all_lambdas)
    all_cv_errors_arr = np.array(all_cv_errors)
    global_best_idx = int(np.argmin(all_cv_errors_arr))
    lambda_opt = float(all_lambdas_arr[global_best_idx])

    print(f"    lambda_opt = {lambda_opt:.3e}  (found in attempt {attempt + 1})")
    print(f"    CV error   = {all_cv_errors_arr[global_best_idx]:.4f}")

    return lambda_opt


# =============================================================
# Core constrained optimization solver
# =============================================================
def solve_constrained_optimization(
    matrix_a: np.ndarray, vec_y: np.ndarray, lam: float
) -> np.ndarray:
    """
    Solve sparse recovery using L1 + LS (two-stage):

    Stage 1 - Plain L1 (support identification):
        min_{x >= 0}  sum(x) + lambda * ||Ax - y||_2^2

    Stage 2 - NNLS on identified support:
        min_{x_S >= 0}  ||A[:, S] x_S - y||_2

    Args:
        A: System matrix (shape: m x n)
        y: Observation vector (shape: m,)
        lam: Regularization parameter
    """
    n = matrix_a.shape[1]
    solver_list = ["CLARABEL", "SCS", "ECOS", "OSQP", "CVXOPT"]

    # -------------------------
    # Stage 1: Plain L1
    # -------------------------
    vec_x = cp.Variable(n, nonneg=True)
    objective = cp.Minimize(cp.sum(vec_x) + lam * cp.sum_squares(matrix_a @ vec_x - vec_y))
    prob = cp.Problem(objective)

    for solver in solver_list:
        try:
            prob.solve(solver=solver, verbose=False)
            if vec_x.value is not None and prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                break
            warnings.warn(f"Solver {solver} failed. Trying next solver.", stacklevel=2)
        except cp.SolverError:
            warnings.warn(f"Solver {solver} failed. Trying next solver.", stacklevel=2)

    if vec_x.value is None:
        warnings.warn("All solvers failed. Returning zeros.", stacklevel=2)
        return np.zeros(n)

    # -------------------------
    # Stage 2: NNLS on support
    # -------------------------
    # support_threshold: Variables below this after Stage 1 are treated as zero
    support_threshold = 0.01

    active_mask = vec_x.value > support_threshold

    if not np.any(active_mask):
        warnings.warn(
            "No active variables found after L1. Consider lowering support_threshold or adjusting lambda.",
            stacklevel=2,
        )
        return vec_x.value

    x_active, _ = nnls(matrix_a[:, active_mask], vec_y)
    x_refined = np.zeros(n)
    x_refined[active_mask] = x_active

    return x_refined


# =============================================================
# Mathematically grounded starting point
# =============================================================
def compute_lambda_baseline(matrix_a: np.ndarray, vec_y: np.ndarray) -> float:
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
    aty = matrix_a.T @ vec_y
    max_aty = np.max(aty[aty > 0]) if np.any(aty > 0) else 0.0

    if max_aty == 0:
        warnings.warn("A^T y has no positive entries — y may be all zeros.", stacklevel=2)
        lam_min = 1e-6
    else:
        lam_min = 1.0 / (2.0 * max_aty)

    # NNLS gives the best-fit x >= 0, used to calibrate the upper grid bound
    x_nnls, res_norm = nnls(matrix_a, vec_y)  # res_norm = ||Ax_nnls - y||_2
    res_sq = res_norm**2
    l1_nnls = np.sum(x_nnls)

    if res_sq > 0 and l1_nnls > 0:
        lam_balance = l1_nnls / res_sq
    else:
        warnings.warn("NNLS solution is degenerate. Falling back to lam_min * 1e4.", stacklevel=2)
        lam_balance = lam_min * 1e4

    return lam_min, lam_balance


# =============================================================
# Utility function to print solution summary
# =============================================================
def print_solution_summary(
    node_names: list[str], lambda_used: list[float], vec_x: np.ndarray, msg_size_sets: list[int]
) -> None:
    """Print per-node lambda selection and solution summary."""
    if vec_x.shape[1] % 2 != 0:
        raise ValueError(f"X has {vec_x.shape[1]} columns — expected an even number.")

    num_msg_sizes = vec_x.shape[1] // 2

    print("\n=== Residual Tolerance Summary ===")
    print(
        f"  {'Node':<20} {'lambda':>12} {'active_bins':>12} {'total_sends':>12} {'total_recvs':>12}"
    )
    print("  " + "-" * 70)

    for node_idx, (name, lambda_val) in enumerate(zip(node_names, lambda_used, strict=True)):
        x_send = vec_x[node_idx, :num_msg_sizes]
        x_recv = vec_x[node_idx, num_msg_sizes:]
        active = int(np.sum(vec_x[node_idx, :] > 0.5))
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

            print("    bins :")
            for s, r in zip(send_pairs, recv_pairs, strict=True):
                print(f"        send: {s:<{pad}} recv: {r}")
        else:
            print("    bins : (none)")
