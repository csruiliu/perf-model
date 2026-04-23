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
    matrix_a: np.ndarray,
    vec_y: np.ndarray,
    node_names: list[str] | None = None,
    total_messages: dict[str, int] | None = None,
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

    if total_messages is not None:
        validate_message_counts(total_messages, node_names)

    print(f"  A : {matrix_a.shape}   Y : {vec_y.shape}   X : ({num_nodes}, {matrix_a.shape[1]})\n")

    num_msg_sizes = matrix_a.shape[1] // 2
    vec_x = np.zeros((num_nodes, matrix_a.shape[1]))
    lambda_used_list = []

    # 1. Compute baseline packet weights using the ORIGINAL, unscaled matrix
    packet_weights = np.sum(matrix_a, axis=0)

    for node_idx in range(num_nodes):
        node_name = node_names[node_idx]

        print(f"--- Node [{node_idx + 1}/{num_nodes}]: {node_name} ---")
        y_nid = vec_y[node_idx, :]

        n_total = _get_node_message_total(total_messages, node_name)
        if n_total is not None:
            print(f"  Message count constraint: n_total={int(n_total)}")
        else:
            print("  Message count constraint: none")

        # 2. Compute Poisson variance weights for this node
        # We add 1.0 to avoid division by zero on empty counters
        row_scales = 1.0 / np.sqrt(np.maximum(y_nid, 1.0))

        # 3. Scale the matrix and target vector
        matrix_a_scaled = matrix_a * row_scales[:, np.newaxis]
        y_scaled = y_nid * row_scales

        print("  Auto-tuning lambda (residual tolerance):")
        lam = find_lambda_cv(
            matrix_a_scaled,
            y_scaled,
            packet_weights,
            num_msg_sizes=num_msg_sizes,
            n_total=n_total,
            max_extensions=5,
        )
        x_nid = solve_constrained_optimization(
            matrix_a_scaled,
            y_scaled,
            packet_weights,
            lam,
            num_msg_sizes=num_msg_sizes,
            n_total=n_total,
        )
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
    matrix_a: np.ndarray,
    vec_y: np.ndarray,
    packet_weights: np.ndarray,
    num_msg_sizes: int,
    n_total: float | None = None,
    max_extensions: int = 5,
) -> float:
    """
    Select the regularization parameter lambda via Leave-One-Counter-Out cross-validation.

    When n_total is provided, the message count equality constraint is enforced in
    every CV fold so that the selected lambda is consistent with the constrained
    problem that will be solved at inference time.

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
                    matrix_a[mask, :],
                    vec_y[mask],
                    packet_weights,
                    lam,
                    num_msg_sizes=num_msg_sizes,
                    n_total=n_total,
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
    matrix_a: np.ndarray,
    vec_y: np.ndarray,
    packet_weights: np.ndarray,
    lam: float,
    num_msg_sizes: int,
    n_total: float | None = None,
) -> np.ndarray:
    """
    Recover a sparse, non-negative solution via two-stage L1-regularized least squares.

    Stage 1 — Support identification (L1 + LS):
        Solve the penalised problem using a convex solver:

            min_{x >= 0}  w^T x + lambda * ||Ax - y||_2^2
            subject to:
                sum(x) == n_total   (if n_total is not None)

        where x[:N] are send-counts and x[N:] are recv-counts, and the constraint
        fixes their combined total while leaving the send/recv split free.

    Stage 2 — Solution polishing (NNLS):
        Restrict to the active support S = {j : x_j > threshold} and solve
        the unpenalised non-negative least-squares problem:

            min_{x_S >= 0}  ||A[:, S] x_S - y||_2

        This removes the shrinkage bias introduced by the L1 penalty in Stage 1.
    """
    n = matrix_a.shape[1]
    solver_list = ["CLARABEL", "SCS", "ECOS", "OSQP", "CVXOPT"]

    # --- merge identical columns before solving ---
    matrix_a_merged, weights_merged, groups = merge_identical_columns(matrix_a, packet_weights)
    n_merged = matrix_a_merged.shape[1]

    vec_x = cp.Variable(n_merged, nonneg=True)
    objective = cp.Minimize(
        (weights_merged @ vec_x) + lam * cp.sum_squares(matrix_a_merged @ vec_x - vec_y)
    )
    constraints = _build_message_count_constraints(vec_x, n_total)
    prob = cp.Problem(objective, constraints)

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
    active_mask_merged = vec_x.value > support_threshold

    if not np.any(active_mask_merged):
        return expand_solution(vec_x.value, groups, n)

    # --- Stage 2: polish on the merged active support ---
    x_refined_merged = _polish_active_support(
        matrix_a_merged, vec_y, active_mask_merged, n_total, n_merged
    )

    # --- expand back to original column space ---
    return expand_solution(x_refined_merged, groups, n)


# =============================================================
# Mathematically grounded starting point
# =============================================================
def compute_lambda_baseline(
    matrix_a: np.ndarray,  # A_scaled — already Poisson-scaled
    vec_y: np.ndarray,  # y_scaled — already Poisson-scaled
    packet_weights: np.ndarray,  # w = sum(A_unscaled, axis=0) — unscaled column sums
) -> tuple[float, float]:
    # ------------------------------------------------------------------
    # lam_min = min_j { W_j / (2 * (A^T 1)_j) }
    #
    # (A^T 1)_j = column sum of UNSCALED A = packet_weights[j]
    # NOT np.sum(matrix_a, axis=0) which would give column sums of A_scaled
    # ------------------------------------------------------------------
    col_sums_unscaled = np.maximum(packet_weights, 1e-12)  # guard zero columns
    lam_min = float(np.min(packet_weights / (2.0 * col_sums_unscaled)))

    # ------------------------------------------------------------------
    # lam_balance = W^T X_nnls / sum_i (A[i,:] X_nnls - Y[i])^2 / Y[i]
    #
    # Solved on the scaled system since:
    #   sum_i (A_scaled[i,:] X - y_scaled[i])^2
    #       == sum_i (A[i,:] X - Y[i])^2 / Y[i]
    # ------------------------------------------------------------------
    x_nnls, _ = nnls(matrix_a, vec_y)

    # Poisson-weighted residual via scaled equivalence
    poisson_residual_sq = float(np.sum((matrix_a @ x_nnls - vec_y) ** 2))
    l1_cost = float(packet_weights @ x_nnls)

    if poisson_residual_sq < 1e-12:
        # Perfect fit — balance point is undefined; return a safe fallback
        lam_balance = lam_min * 100.0
    else:
        lam_balance = l1_cost / poisson_residual_sq

    return lam_min, lam_balance


def merge_identical_columns(
    matrix_a: np.ndarray, packet_weights: np.ndarray, tol: float = 1e-6
) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
    """
    Detect groups of identical columns in matrix_a, merge each group into a
    single representative column, and accumulate the corresponding packet weights.

    Returns
    -------
    matrix_a_merged : np.ndarray, shape (m, n_unique)
        Reduced matrix with one column per unique fingerprint.
    weights_merged : np.ndarray, shape (n_unique,)
        Accumulated packet weights for each merged column.
    groups : list[list[int]]
        groups[k] is the list of original column indices that were merged into
        merged column k.  len(groups) == n_unique.
    """
    n = matrix_a.shape[1]
    visited = np.zeros(n, dtype=bool)
    groups = []

    for i in range(n):
        if visited[i]:
            continue
        group = [i]
        for j in range(i + 1, n):
            if not visited[j] and np.linalg.norm(matrix_a[:, i] - matrix_a[:, j]) < tol:
                group.append(j)
        for idx in group:
            visited[idx] = True
        groups.append(group)

    n_unique = len(groups)
    matrix_a_merged = np.zeros((matrix_a.shape[0], n_unique), dtype=np.float64)
    weights_merged = np.zeros(n_unique, dtype=np.float64)

    for k, group in enumerate(groups):
        matrix_a_merged[:, k] = matrix_a[:, group[0]]
        weights_merged[k] = np.sum(packet_weights[group])

    return matrix_a_merged, weights_merged, groups


def expand_solution(x_merged: np.ndarray, groups: list[list[int]], n_original: int) -> np.ndarray:
    """
    Expand a solution on the merged system back to the original column space.

    The entire value of each merged column is placed on the first member of
    the group; all other members are set to zero.  This is the accumulation
    behaviour of the original NNLS code.

    Parameters
    ----------
    x_merged : np.ndarray, shape (n_unique,)
        Solution in the merged column space.
    groups : list[list[int]]
        Column grouping returned by merge_identical_columns.
    n_original : int
        Number of columns in the original (unmerged) matrix.
    """
    x_full = np.zeros(n_original, dtype=np.float64)
    for k, group in enumerate(groups):
        x_full[group[0]] = x_merged[k]  # accumulate into first member
    return x_full


def _polish_active_support(
    matrix_a: np.ndarray,
    vec_y: np.ndarray,
    active_mask: np.ndarray,
    n_total: float | None,
    n_total_vars: int,
) -> np.ndarray:
    """
    Polish the Stage-1 solution by solving an unpenalised NNLS restricted to the
    active support, while re-applying the message count constraint.

    When n_total is None, scipy.optimize.nnls is used for speed.
    When n_total is provided, CVXPY enforces sum(x_active) == n_total.
    Falls back to unconstrained NNLS if the constrained polish fails.
    """
    x_refined = np.zeros(n_total_vars)

    if n_total is None:
        # No constraint — fast path via scipy NNLS
        x_active, _ = nnls(matrix_a[:, active_mask], vec_y)
        x_refined[active_mask] = x_active
        return x_refined

    # Constrained polish via CVXPY
    n_active = int(np.sum(active_mask))
    x_var = cp.Variable(n_active, nonneg=True)
    objective = cp.Minimize(cp.sum_squares(matrix_a[:, active_mask] @ x_var - vec_y))
    polish_constraints = [cp.sum(x_var) == n_total]

    prob = cp.Problem(objective, polish_constraints)
    solver_list = ["CLARABEL", "SCS", "ECOS", "OSQP", "CVXOPT"]

    for solver in solver_list:
        try:
            prob.solve(solver=solver, verbose=False)
            if x_var.value is not None and prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                break
        except cp.SolverError:
            continue

    if x_var.value is not None:
        x_refined[active_mask] = x_var.value
    else:
        warnings.warn(
            "Constrained polish failed; falling back to unconstrained NNLS.", stacklevel=3
        )
        x_active, _ = nnls(matrix_a[:, active_mask], vec_y)
        x_refined[active_mask] = x_active

    return x_refined


def _get_node_message_total(total_messages: dict[str, int] | None, node_name: str) -> float | None:
    """Return the total message count (send + recv) for a node, or None if unconstrained."""
    if total_messages is None or node_name not in total_messages:
        return None
    return float(total_messages[node_name])


def _build_message_count_constraints(vec_x: cp.Variable, n_total: float | None) -> list:
    """
    Return a single CVXPY equality constraint:
        sum(x) == n_total
    meaning sum(x_send) + sum(x_recv) == n_total.
    The send/recv split is left free to the solver.
    Returns an empty list when n_total is None.
    """
    if n_total is None:
        return []
    return [cp.sum(vec_x) == n_total]


# =============================================================
# Utility function to print solution summary
# =============================================================
def print_solution_summary(
    node_names: list[str],
    lambda_used: list[float],
    vec_x: np.ndarray,
    msg_size_sets: np.ndarray,
    total_messages: dict[str, int] | None = None,
) -> None:
    if vec_x.shape[1] % 2 != 0:
        raise ValueError(f"X has {vec_x.shape[1]} columns — expected an even number.")
    num_msg_sizes = vec_x.shape[1] // 2

    print("\n=== Solution Summary ===")
    header = (
        f"  {'Node':<20} {'lambda':>12} {'active_bins':>12} "
        f"{'total_sends':>14} {'total_recvs':>14} {'solved_total':>14}"
    )
    if total_messages:
        header += f"  {'expected_total':>15}"
    print(header)
    print("  " + "-" * (90 + (17 if total_messages else 0)))

    for node_idx, (name, lambda_val) in enumerate(zip(node_names, lambda_used, strict=True)):
        x_send = vec_x[node_idx, :num_msg_sizes]
        x_recv = vec_x[node_idx, num_msg_sizes:]

        active = int(np.sum(x_send > 0.5) + np.sum(x_recv > 0.5))
        tot_send = int(np.round(np.sum(x_send)))
        tot_recv = int(np.round(np.sum(x_recv)))
        solved_total = tot_send + tot_recv

        row = (
            f"  {name:<20} {lambda_val:>12.3e} {active:>12} "
            f"{tot_send:>14} {tot_recv:>14} {solved_total:>14}"
        )
        if total_messages and name in total_messages:
            row += f"  {int(total_messages[name]):>15}"
        print(row)

        active_send_bins = np.where(x_send > 0.5)[0].tolist()
        active_recv_bins = np.where(x_recv > 0.5)[0].tolist()

        if active_send_bins or active_recv_bins:
            send_pairs = [f"{int(msg_size_sets[i])}B: {x_send[i]:.2f}" for i in active_send_bins]
            recv_pairs = [f"{int(msg_size_sets[i])}B: {x_recv[i]:.2f}" for i in active_recv_bins]

            max_len = max(len(send_pairs), len(recv_pairs))
            send_pairs += [""] * (max_len - len(send_pairs))
            recv_pairs += [""] * (max_len - len(recv_pairs))
            pad = max(len(s) for s in send_pairs) + 2 if send_pairs else 0

            print("    bins :")
            for s, r in zip(send_pairs, recv_pairs, strict=True):
                print(f"        send: {s:<{pad}} recv: {r}")
        else:
            print("    bins : (none)")


def validate_message_counts(total_messages: dict[str, int], node_names: list[str]) -> None:
    unknown = set(total_messages) - set(node_names)
    if unknown:
        raise ValueError(
            "total_messages contains node names not present in the loaded data:\n"
            + "\n".join(f"  {n}" for n in sorted(unknown))
        )

    for name, n_total in total_messages.items():
        if n_total < 0:
            raise ValueError(
                f"Node '{name}': total message count must be non-negative (got n_total={n_total})."
            )

    print(f"  [OK] total_messages validated for {len(total_messages)} node(s).")


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
