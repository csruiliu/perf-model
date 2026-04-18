"""
solver.py

Solves the MPI communication model for all nodes.

All constants imported from constants.py.
"""

import warnings

import cvxpy as cp
import numpy as np
from constants import COUNTER_GROUPS, RX_HIST_SLICE, TX_HIST_SLICE
from scipy.optimize import nnls


# =============================================================
# Global solver — single run
# =============================================================
def solve_global(
    matrix_a: np.ndarray, vec_y: np.ndarray, node_names: list[str] | None = None
) -> tuple[np.ndarray, list[float]]:
    """
    Solve the sparse recovery problem independently for every node in a single pass.

    For each node, the system matrix is rescaled using Poisson variance weights
    derived from that node's observation vector, and then the optimal regularization
    parameter lambda is selected via cross-validation before solving:

        min_{x >= 0}  w^T x + lambda * ||A_scaled x - y_scaled||_2^2

    where w are packet weights (column sums of the original A) and the scaled
    versions of A and y account for Poisson noise in each row.
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

    # 1. Compute baseline packet weights using the ORIGINAL, unscaled matrix
    packet_weights = np.sum(matrix_a, axis=0)

    for node_idx in range(num_nodes):
        print(f"--- Node [{node_idx + 1}/{num_nodes}]: {node_names[node_idx]} ---")
        y_nid = vec_y[node_idx, :]

        # 2. Compute Poisson variance weights for this node
        # We add 1.0 to avoid division by zero on empty counters
        row_scales = 1.0 / np.sqrt(np.maximum(y_nid, 1.0))

        # 3. Scale the matrix and target vector
        matrix_a_scaled = matrix_a * row_scales[:, np.newaxis]
        y_scaled = y_nid * row_scales

        print("  Auto-tuning lambda (residual tolerance):")
        lam = find_lambda_cv(matrix_a_scaled, y_scaled, packet_weights, max_extensions=5)
        x_nid = solve_constrained_optimization(matrix_a_scaled, y_scaled, packet_weights, lam)
        vec_x[node_idx, :] = x_nid
        lambda_used_list.append(lam)

        active_bins = int(np.sum(x_nid > 0.5))
        # Print original unscaled residual for user visibility
        residual = np.linalg.norm(matrix_a @ x_nid - y_nid)
        print(f"  lambda={lam:.3e}  active_bins={active_bins}  residual={residual:.2f}\n")

    return vec_x, lambda_used_list


# =============================================================
# Lambda selection via Leave-One-Counter-Out Cross Validation
# =============================================================
def find_lambda_cv(
    matrix_a: np.ndarray, vec_y: np.ndarray, packet_weights: np.ndarray, max_extensions: int = 5
) -> float:
    """
    Select the regularization parameter lambda via Leave-One-Counter-Out cross-validation.

    A logarithmic grid of lambda candidates is evaluated; for each candidate, every
    counter is held out in turn and the squared prediction error on that counter is
    recorded. The lambda that minimises the mean CV error is returned.

    If the best lambda falls on a grid boundary, the grid is automatically shifted
    in that direction by `extend_factor` and CV is re-run over the new region.
    Results are accumulated across all extensions and the global minimum is returned.
    """
    m = len(vec_y)
    lam_n_points = 50
    extend_factor = 10.0

    lam_min, lam_balance = compute_lambda_baseline(matrix_a, vec_y, packet_weights)
    lam_lo, lam_hi = lam_min, lam_balance * 10.0

    print(f"    lam_min (KKT transition) = {lam_min:.3e}")
    print(f"    lam_balance (NNLS)       = {lam_balance:.3e}")

    all_lambdas, all_cv_errors = [], []

    for attempt in range(max_extensions + 1):
        lambda_grid = np.logspace(np.log10(lam_lo), np.log10(lam_hi), lam_n_points)

        cv_errors = np.zeros(lam_n_points)
        for i, lam in enumerate(lambda_grid):
            fold_errors = np.zeros(m)
            for k in range(m):
                mask = np.ones(m, dtype=bool)
                mask[k] = False
                x_k = solve_constrained_optimization(
                    matrix_a[mask, :], vec_y[mask], packet_weights, lam
                )
                fold_errors[k] = (matrix_a[k, :] @ x_k - vec_y[k]) ** 2
            cv_errors[i] = np.mean(fold_errors)

        all_lambdas.extend(lambda_grid.tolist())
        all_cv_errors.extend(cv_errors.tolist())

        local_best_idx = int(np.argmin(cv_errors))
        at_lower = local_best_idx == 0
        at_upper = local_best_idx == lam_n_points - 1

        if not at_lower and not at_upper:
            break

        if attempt < max_extensions:
            if at_lower:
                lam_hi, lam_lo = lam_lo, lam_lo / extend_factor
            else:
                lam_lo, lam_hi = lam_hi, lam_hi * extend_factor

    all_lambdas_arr = np.array(all_lambdas)
    all_cv_errors_arr = np.array(all_cv_errors)
    global_best_idx = int(np.argmin(all_cv_errors_arr))

    return float(all_lambdas_arr[global_best_idx])


# =============================================================
# Core constrained optimization solver
# =============================================================
def solve_constrained_optimization(
    matrix_a: np.ndarray, vec_y: np.ndarray, packet_weights: np.ndarray, lam: float
) -> np.ndarray:
    """
    Recover a sparse, non-negative solution via two-stage L1-regularized least squares.

    Stage 1 — Support identification (L1 + LS):
        Solve the penalised problem using a convex solver:

            min_{x >= 0}  w^T x + lambda * ||Ax - y||_2^2

        where w are the packet weights. The solution identifies which bins
        are likely non-zero (the *active support*).

    Stage 2 — Solution polishing (NNLS):
        Restrict to the active support S = {j : x_j > threshold} and solve
        the unpenalised non-negative least-squares problem:

            min_{x_S >= 0}  ||A[:, S] x_S - y||_2

        This removes the shrinkage bias introduced by the L1 penalty in Stage 1.
    """
    n = matrix_a.shape[1]
    solver_list = ["CLARABEL", "SCS", "ECOS", "OSQP", "CVXOPT"]

    vec_x = cp.Variable(n, nonneg=True)

    # We now map the original packet weights against the scaled Sum of Squares
    objective = cp.Minimize(
        (packet_weights @ vec_x) + lam * cp.sum_squares(matrix_a @ vec_x - vec_y)
    )
    prob = cp.Problem(objective)

    for solver in solver_list:
        try:
            prob.solve(solver=solver, verbose=False)
            if vec_x.value is not None and prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                break
        except cp.SolverError:
            continue

    if vec_x.value is None:
        warnings.warn("All solvers failed. Returning zeros.", stacklevel=2)
        return np.zeros(n)

    support_threshold = 0.01
    active_mask = vec_x.value > support_threshold

    if not np.any(active_mask):
        return vec_x.value

    # Polish the active variables using the scaled matrices
    x_active, _ = nnls(matrix_a[:, active_mask], vec_y)
    x_refined = np.zeros(n)
    x_refined[active_mask] = x_active

    return x_refined


# =============================================================
# Mathematically grounded starting point
# =============================================================
def compute_lambda_baseline(
    matrix_a: np.ndarray, vec_y: np.ndarray, packet_weights: np.ndarray
) -> float:
    """
    Compute principled lower and upper bounds for the lambda search range.

    Two reference values are derived:

    lam_min — KKT transition point:
        The smallest lambda for which x = 0 is no longer optimal. Derived from
        the KKT stationarity condition of the penalised objective:

            lam_min = 1 / (2 * max_j { (A^T y)_j / w_j })

        For lambda below this threshold, x = 0 satisfies the KKT conditions and
        is a valid solution. Above it, at least one bin must be active.

    lam_balance — L1 / residual balance point:
        The lambda at which the weighted L1 term and the squared residual term
        contribute equally, estimated from the unconstrained NNLS solution:

            lam_balance = (w^T x_nnls) / ||A x_nnls - y||_2^2

        This acts as a soft upper bound: beyond this value the penalty dominates
        and the solution is driven toward more sparsity than the data supports.

    A.T @ y is essentially asking:
    "which features are most correlated with what I'm trying to predict?"
    """
    aty = matrix_a.T @ vec_y

    valid_mask = aty > 0
    if np.any(valid_mask):
        max_ratio = np.max(aty[valid_mask] / packet_weights[valid_mask])
        lam_min = 1.0 / (2.0 * max_ratio)
    else:
        warnings.warn("A^T y has no positive entries — y may be all zeros.", stacklevel=2)
        lam_min = 1e-6

    x_nnls, res_norm = nnls(matrix_a, vec_y)
    res_sq = res_norm**2

    l1_nnls = np.sum(packet_weights * x_nnls)

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


def validate_solution(
    matrix_a: np.ndarray,
    vec_y: np.ndarray,
    vec_x: np.ndarray,
    node_names: list[str] | None = None,
    rel_tol: float = 0.10,
) -> bool:
    """
    Validate that the predicted counters from solution x match the observed counters y.

    For each node, computes y_hat = A @ x and compares it against y across four
    counter groups: TX histogram, TX TC, RX histogram, RX TC.

    A counter passes if:
        |y_hat_i - y_i| / y_i <= rel_tol   (when y_i > 0)
        y_hat_i == 0                         (when y_i == 0)

    Packet conservation is checked separately by summing the TX and RX histogram
    counters — the only counters that obey a strict conservation law (every packet
    counted exactly once in exactly one bucket).
    """
    num_nodes = vec_y.shape[0]

    if node_names is None:
        node_names = [f"node_{n}" for n in range(num_nodes)]

    if len(node_names) != num_nodes:
        raise ValueError(
            f"node_names has {len(node_names)} entries but vec_y has {num_nodes} rows."
        )

    all_pass = True

    for node_idx in range(num_nodes):
        node_pass = _validate_node(
            matrix_a, vec_y[node_idx, :], vec_x[node_idx, :], node_names[node_idx], rel_tol
        )
        all_pass = all_pass and node_pass

    width = 70
    print(f"\n{'=' * width}")
    print(f"  Overall validation : {'PASS' if all_pass else 'WARN'}")
    print(f"{'=' * width}\n")

    return all_pass


def _validate_node(
    matrix_a: np.ndarray, y_obs: np.ndarray, x_node: np.ndarray, node_name: str, rel_tol: float
) -> bool:
    """
    Validate a single node's solution against its observed counters.

    Reports three things:
      1. Per-counter table: observed, predicted, absolute error, relative error, status.
      2. Packet conservation summary: total TX/RX histogram packets observed vs predicted.
      3. Overall L2 residual (absolute and relative).

    Returns True if all per-counter checks and packet conservation pass.
    """
    y_hat = matrix_a @ x_node

    width = 70
    print(f"\n{'=' * width}")
    print(f"  Node : {node_name}")
    print(f"{'=' * width}")

    all_counters_pass = True

    # ------------------------------------------------------------------
    # 1. Per-counter breakdown across all four counter groups
    # ------------------------------------------------------------------
    col_w = 30
    header = (
        f"  {'Counter':<{col_w}} {'Observed':>12} {'Predicted':>12} "
        f"{'Abs Err':>12} {'Rel Err':>10} {'Status':>8}"
    )
    sep = "  " + "-" * (len(header) - 2)

    for group_name, slc, counter_names in COUNTER_GROUPS:
        print(f"\n  {group_name}:")
        print(header)
        print(sep)

        y_obs_grp = y_obs[slc]
        y_hat_grp = y_hat[slc]

        for i, name in enumerate(counter_names):
            obs = float(y_obs_grp[i])
            pred = float(y_hat_grp[i])
            abs_err = abs(pred - obs)

            # Relative error definition:
            #   - y_i > 0  : standard relative error
            #   - y_i == 0 and pred == 0 : perfect match, rel_err = 0
            #   - y_i == 0 and pred != 0 : unbounded error, flag as WARN
            if obs > 0:
                rel_err = abs_err / obs
                status = "OK" if rel_err <= rel_tol else "WARN"
            elif pred == 0.0:
                rel_err = 0.0
                status = "OK"
            else:
                rel_err = float("inf")
                status = "WARN"

            if status != "OK":
                all_counters_pass = False

            rel_err_str = f"{rel_err:.1%}" if rel_err != float("inf") else "inf"
            print(
                f"  {name:<{col_w}} {obs:>12.0f} {pred:>12.0f} "
                f"{abs_err:>12.0f} {rel_err_str:>10} {status:>8}"
            )

    # ------------------------------------------------------------------
    # 2. Packet conservation summary (histogram totals only)
    #
    # We sum only the 8 TX histogram and 8 RX histogram counters because
    # each physical packet is counted exactly once in exactly one bucket —
    # their sum is the true total packet count. TC counters classify the
    # same packets by traffic class and would double-count if included.
    # ------------------------------------------------------------------
    total_tx_obs = float(np.sum(y_obs[TX_HIST_SLICE]))
    total_tx_hat = float(np.sum(y_hat[TX_HIST_SLICE]))
    total_rx_obs = float(np.sum(y_obs[RX_HIST_SLICE]))
    total_rx_hat = float(np.sum(y_hat[RX_HIST_SLICE]))

    tx_rel_err = abs(total_tx_hat - total_tx_obs) / total_tx_obs if total_tx_obs > 0 else 0.0
    rx_rel_err = abs(total_rx_hat - total_rx_obs) / total_rx_obs if total_rx_obs > 0 else 0.0
    tx_ok = tx_rel_err <= rel_tol
    rx_ok = rx_rel_err <= rel_tol

    pkt_header = f"  {'':30} {'Observed':>12} {'Predicted':>12} {'Rel Err':>10} {'Status':>8}"
    pkt_sep = "  " + "-" * (len(pkt_header) - 2)

    print("\n  Packet Conservation (histogram totals):")
    print(pkt_header)
    print(pkt_sep)
    print(
        f"  {'Total TX Histogram Pkts':<30} {total_tx_obs:>12.0f} {total_tx_hat:>12.0f} "
        f"{tx_rel_err:>9.1%} {'OK' if tx_ok else 'WARN':>8}"
    )
    print(
        f"  {'Total RX Histogram Pkts':<30} {total_rx_obs:>12.0f} {total_rx_hat:>12.0f} "
        f"{rx_rel_err:>9.1%} {'OK' if rx_ok else 'WARN':>8}"
    )

    # ------------------------------------------------------------------
    # 3. Overall L2 residual
    # ------------------------------------------------------------------
    residual = float(np.linalg.norm(y_hat - y_obs))
    norm_y_obs = float(np.linalg.norm(y_obs))
    rel_residual = residual / norm_y_obs if norm_y_obs > 0 else 0.0

    print(f"\n  L2 residual : {residual:.2f}  (relative : {rel_residual:.2%})")

    node_pass = all_counters_pass and tx_ok and rx_ok
    verdict = "PASS" if node_pass else "WARN"
    direction = "all counters within" if node_pass else "some counters exceed"
    print(f"  Node status : {verdict} — {direction} {rel_tol * 100:.0f}% relative tolerance")

    return node_pass
