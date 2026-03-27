"""
solve_global.py

Solves the MPI communication model for all nodes.

All constants imported from constants.py.
"""
import numpy as np
from typing import Tuple, List, Optional, Dict
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
    Solve global constrained least-squares optimization for all nodes (single run).

        min_{x >= 0}  ||Ax - y||_2^2
        s.t.          ||x||_1 <= c

    Returns
    -------
    X       : np.ndarray, shape (num_nodes, 2*num_msg_size_bins)
    t_used  : list of float, length num_nodes
    """
    num_nodes, all_counters_txrx = Y.shape

    if all_counters_txrx != A.shape[0]:
        raise ValueError(f"Y has {all_counters_txrx} counters per node but A has {A.shape[0]} rows.")

    if node_names is None:
        node_names = [f"node_{n}" for n in range(num_nodes)]

    print(f"  A : {A.shape}   Y : {Y.shape}   X : ({num_nodes}, {A.shape[1]})\n")

    X = np.zeros((num_nodes, A.shape[1]))
    c_used_list = []

    for node_idx in range(num_nodes):
        print(f"--- Node [{node_idx+1}/{num_nodes}]: {node_names[node_idx]} ---")

        y_nid = Y[node_idx, :]
        print(f"  Auto-tuning c (resources budget):")
        c = loco_cv_method(A, y_nid)
        x_nid = solve_constrained_optimization(A, y_nid, c)
        X[node_idx, :] = x_nid
        c_used_list.append(c)

        # printout summary for this node
        active_bins = int(np.sum(x_nid > 0.5))
        residual = np.linalg.norm(A @ x_nid - y_nid)
        print(f"  C={c:.3e}  active_bins={active_bins}  residual={residual:.2f}\n")

    return X, c_used_list


# =============================================================
# Leave-One-Counter-Out Cross-Validation (LOCO-CV) for t selection
# =============================================================
def loco_cv_method(A: np.ndarray, y: np.ndarray) -> float:
    """
    Automatic t selection via Leave-One-Counter-Out CV.

    The grid spans from a very tight L1 budget (sparse solution) up to the
    unconstrained NNLS solution norm (no effective constraint).

    Returns
    -------
    c_opt : float
        Optimal budget minimizing the leave-one-out CV error.
    """
    two_num_all_cntrs = len(y)

    c_n_points  = 40
    c_min_factor = 0.01   # 1%  of unconstrained norm → very sparse
    c_max_factor = 1.00   # 100% of unconstrained norm → effectively unconstrained

    # c_baseline is the L1 norm of the unconstrained NNLS solution.
    # It defines the upper end of our search: beyond this, the constraint is inactive.
    c_baseline = compute_c_baseline(A, y)

    # Linear spacing is natural for c since it is a direct budget on ||x||_1
    c_grid = np.linspace(
        c_baseline * c_min_factor,
        c_baseline * c_max_factor,
        c_n_points
    )

    print(f"    c_baseline = {c_baseline:.3e}")
    print(f"    c grid : [{c_grid[0]:.3e}, {c_grid[-1]:.3e}]"
          f"  ({c_n_points} points, {two_num_all_cntrs} folds each)")

    cv_errors = np.zeros(c_n_points)

    for i, c in enumerate(c_grid):
        fold_errors = np.zeros(two_num_all_cntrs)

        # LOCO-CV: for each counter k, leave it out, solve the constrained problem
        # on the remaining counters, and compute squared error on the held-out counter.
        for k in range(two_num_all_cntrs):
            mask = np.ones(two_num_all_cntrs, dtype=bool)
            mask[k] = False   # False means "exclude this counter"
            x_k = solve_constrained_optimization(A[mask, :], y[mask], c)
            fold_errors[k] = (A[k, :] @ x_k - y[k]) ** 2

        cv_errors[i] = np.mean(fold_errors)

    best_idx = int(np.argmin(cv_errors))
    c_opt    = float(c_grid[best_idx])

    print(f"    c_opt    = {c_opt:.3e}  (index {best_idx}/{c_n_points-1})")
    print(f"    CV error = {cv_errors[best_idx]:.4f}")

    return c_opt


# =============================================================
# Core constrained optimization solver
# =============================================================
def solve_constrained_optimization(A: np.ndarray, y: np.ndarray, c: float) -> np.ndarray:
    """
    Solve one non-negative constrained least-squares instance:

        min_{x >= 0}  ||Ax - y||_2^2
        s.t.          ||x||_1 <= c

    Args:
        A : System matrix
        y : Observation vector
        c : budget (smaller c → sparser solution)
    """
    two_num_msg_sizes = A.shape[1]
    x = cp.Variable(two_num_msg_sizes, nonneg=True)

    objective   = cp.Minimize(cp.sum_squares(A @ x - y))
    constraints = [cp.sum(x) <= c]

    prob = cp.Problem(objective, constraints)

    try:
        prob.solve(solver="SCS", verbose=False)
    except cp.SolverError:
        warnings.warn("SCS failed. Returning zeros.")
        return np.zeros(two_num_msg_sizes)

    if x.value is None:
        warnings.warn("Solver returned None. Returning zeros.")
        return np.zeros(two_num_msg_sizes)

    return np.clip(x.value, 0, None)


# =============================================================
# Compute c baseline from the unconstrained NNLS solution
# =============================================================
def compute_c_baseline(A: np.ndarray, y: np.ndarray) -> float:
    """
    Compute the baseline c value for the c grid search.

        c_baseline = ||x_nnls||_1

    where x_nnls is the unconstrained non-negative least-squares solution.

    This represents the largest meaningful c: setting c = c_baseline
    makes the constraint inactive (equivalent to no regularization).
    Smaller c values progressively tighten the budget and sparsify x.
    """
    x_nnls, _ = nnls(A, y)
    c_baseline = float(np.sum(x_nnls))

    if c_baseline == 0:
        warnings.warn("Unconstrained NNLS solution is zero — y may be all zeros.")
        return 1e-6

    return c_baseline


# =============================================================
# Utility function to print solution summary
# =============================================================
def print_solution_summary(node_names: List[str],
                           t_used: List[float],
                           X: np.ndarray,
                           msg_size_sets: List[int]) -> None:
    """Print per-node t selection and solution summary."""
    if X.shape[1] % 2 != 0:
        raise ValueError(f"X has {X.shape[1]} columns — expected an even number.")

    num_msg_sizes = X.shape[1] // 2

    print("\n=== L1 Budget (t) Summary ===")
    print(f"  {'Node':<20} {'t (budget)':>12} {'active_bins':>12} "
          f"{'total_sends':>12} {'total_recvs':>12}")
    print("  " + "-" * 70)

    for node_idx, (name, t) in enumerate(zip(node_names, t_used)):
        x_send = X[node_idx, :num_msg_sizes]
        x_recv = X[node_idx, num_msg_sizes:]
        active   = int(np.sum(X[node_idx, :] > 0.5))
        tot_send = int(np.sum(x_send))
        tot_recv = int(np.sum(x_recv))

        print(f"  {name:<20} {t:>12.3e} {active:>12} "
              f"{tot_send:>12} {tot_recv:>12}")

        active_send_bins = np.where(x_send > 0.5)[0].tolist()
        active_recv_bins = np.where(x_recv > 0.5)[0].tolist()

        if active_send_bins or active_recv_bins:
            send_pairs = [f"{msg_size_sets[i]}: {x_send[i]:.2f}" for i in active_send_bins]
            recv_pairs = [f"{msg_size_sets[i]}: {x_recv[i]:.2f}" for i in active_recv_bins]

            # Pad the shorter list with empty strings
            max_len = max(len(send_pairs), len(recv_pairs))
            send_pairs += [""] * (max_len - len(send_pairs))
            recv_pairs += [""] * (max_len - len(recv_pairs))

            # Calculate padding based on longest send pair string
            pad = max(len(s) for s in send_pairs) + 2 if send_pairs else 0

            print(f"    bins :")
            for s, r in zip(send_pairs, recv_pairs):
                print(f"        send: {s:<{pad}} recv: {r}")
        else:
            print(f"    bins : (none)")