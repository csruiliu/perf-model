"""
solver.py

Solves the MPI communication model for all nodes.

All constants imported from constants.py.
"""

import warnings

import cvxpy as cp
import numpy as np
from scipy.optimize import nnls

from counter_model.cxi.constants import COUNTER_GROUPS, RX_HIST_SLICE, TX_HIST_SLICE


# =============================================================
# Global solver — single run
# =============================================================
def solve_global(
    matrix_a: np.ndarray,
    vec_y: np.ndarray,
    node_names: list[str],
    msg_size_sets: np.ndarray,
    node_send_msgs: dict[str, int],
    node_recv_msgs: dict[str, int],
) -> tuple[np.ndarray, list[float]]:

    num_nodes, all_counters_txrx = vec_y.shape

    if all_counters_txrx != matrix_a.shape[0]:
        raise ValueError(
            f"Y has {all_counters_txrx} counters per node but A has {matrix_a.shape[0]} rows."
        )

    print(f"  A : {matrix_a.shape}   Y : {vec_y.shape}   X : ({num_nodes}, {matrix_a.shape[1]})\n")

    vec_x = np.zeros((num_nodes, matrix_a.shape[1]))
    lambda_used_list = []

    # 1. Compute baseline packet weights using the ORIGINAL, unscaled matrix
    packet_weights = np.sum(matrix_a, axis=0)
    num_msg_sizes = matrix_a.shape[1] // 2

    for node_idx in range(num_nodes):
        node_name = node_names[node_idx]

        print(f"--- Node [{node_idx + 1}/{num_nodes}]: {node_name} ---")
        y_nid = vec_y[node_idx, :]

        # 2. Compute Poisson variance weights for this node
        # We add 1.0 to avoid division by zero on empty counters
        row_scales = 1.0 / np.sqrt(np.maximum(y_nid, 1.0))

        # 3. Scale the matrix and target vector
        matrix_a_scaled = matrix_a * row_scales[:, np.newaxis]
        y_scaled = y_nid * row_scales

        print("  Auto-tuning lambda (residual tolerance):")
        lam = _find_lambda_cv(
            matrix_a_scaled,
            y_scaled,
            packet_weights,
            num_msg_sizes,
            n_send=node_send_msgs[node_name],
            n_recv=node_recv_msgs[node_name],
            max_extensions=5,
        )
        # Solve the equation using the chosen lambda and all the data
        x_nid = _solve_constrained_optimization(
            matrix_a_scaled,
            y_scaled,
            packet_weights,
            lam,
            num_msg_sizes=num_msg_sizes,
            n_send=node_send_msgs[node_name],
            n_recv=node_recv_msgs[node_name],
        )
        vec_x[node_idx, :] = x_nid
        lambda_used_list.append(lam)

        active_bins = int(np.sum(x_nid > 0.5))
        # Print original unscaled residual for user visibility
        residual = np.linalg.norm(matrix_a @ x_nid - y_nid)
        print(f"  lambda={lam:.3e}  active_bins={active_bins}  residual={residual:.2f}\n")

    vec_x_merged = _merge_indistinguishable_bins(vec_x, msg_size_sets)

    return vec_x_merged, lambda_used_list


# =============================================================
# Lambda selection via Leave-One-Counter-Out Cross Validation
# =============================================================
def _find_lambda_cv(
    matrix_a: np.ndarray,
    vec_y: np.ndarray,
    packet_weights: np.ndarray,
    num_msg_sizes: int,
    n_send: float,
    n_recv: int,
    max_extensions: int = 5,
) -> float:
    m = len(vec_y)
    lam_n_points = 50
    extend_factor = 10.0

    # consider lam_min as lam_lo and lam_balance as lam_hi
    lam_lo, lam_hi = _compute_lambda_baseline(matrix_a, vec_y, packet_weights)
    print(f"    lam_min (KKT transition) = {lam_lo:.3e}")
    print(f"    lam_balance (NNLS)       = {lam_hi:.3e}")

    all_lambdas, all_cv_errors = [], []

    for attempt in range(max_extensions + 1):
        lambda_grid = np.logspace(np.log10(lam_lo), np.log10(lam_hi), lam_n_points)

        cv_errors = np.zeros(lam_n_points)
        for i, lam in enumerate(lambda_grid):
            fold_errors = np.zeros(m)
            for k in range(m):
                mask = np.ones(m, dtype=bool)
                mask[k] = False
                x_k = _solve_constrained_optimization(
                    matrix_a[mask, :],
                    vec_y[mask],
                    packet_weights,
                    lam,
                    num_msg_sizes=num_msg_sizes,
                    n_recv=n_recv,
                    n_send=n_send,
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
# Mathematically grounded starting point
# =============================================================
def _compute_lambda_baseline(
    matrix_a: np.ndarray, vec_y: np.ndarray, packet_weights: np.ndarray
) -> tuple[float, float]:
    # Calculate min lambda according to the equation, always 0.5
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


# =============================================================
# Core constrained optimization solver
# =============================================================
def _solve_constrained_optimization(
    matrix_a: np.ndarray,
    vec_y: np.ndarray,
    packet_weights: np.ndarray,
    lam: float,
    num_msg_sizes: int,
    n_send: float,
    n_recv: int,
) -> np.ndarray:
    """
    Solve the regularised non-negative least-squares problem.
    """
    n = matrix_a.shape[1]
    solver_list = ["CLARABEL", "SCS", "ECOS", "OSQP", "CVXOPT"]

    vec_x = cp.Variable(n, nonneg=True)

    objective = cp.Minimize(
        (packet_weights @ vec_x) + lam * cp.sum_squares(matrix_a @ vec_x - vec_y)
    )

    # Create constraint using the estimated total send and recv messages
    constraints: list = []
    if n_recv is not None:
        constraints.append(cp.sum(vec_x[num_msg_sizes:]) == n_recv)
    if n_send is not None:
        constraints.append(cp.sum(vec_x[:num_msg_sizes]) == n_send)

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

    return vec_x.value


def _merge_indistinguishable_bins(
    vec_x: np.ndarray, msg_size_sets: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:

    if vec_x.shape[1] % 2 != 0:
        raise ValueError(f"vec_x has {vec_x.shape[1]} columns — expected an even number.")

    num_msg_sizes = vec_x.shape[1] // 2
    vec_x_merged = vec_x.copy()
    merged_msg_sizes = msg_size_sets.copy()

    for half_start, half_name in [(0, "send"), (num_msg_sizes, "recv")]:
        half = vec_x_merged[:, half_start : half_start + num_msg_sizes]

        # Round to fixed precision so floating-point near-duplicates are treated
        # as identical, shape (num_nodes, num_msg_sizes) -> use columns as signatures.
        signatures = np.round(half, decimals=6)

        assigned = np.zeros(num_msg_sizes, dtype=bool)

        for j in range(num_msg_sizes):
            if assigned[j]:
                continue

            # Find ALL other unassigned bins with the same value signature.
            # Skip all-zero bins to avoid grouping empty bins together.
            if np.all(signatures[:, j] == 0.0):
                assigned[j] = True
                continue

            group = [
                k
                for k in range(num_msg_sizes)
                if not assigned[k] and k != j and np.array_equal(signatures[:, j], signatures[:, k])
            ]

            if not group:
                assigned[j] = True
                continue

            # All bins in the group including j; pick the one with the
            # smallest message size as the representative.
            full_group = [j] + group
            rep = min(full_group, key=lambda k: msg_size_sets[k])
            non_rep = [k for k in full_group if k != rep]

            # Sum all mass onto the representative bin, zero out the rest.
            total_mass = np.sum(half[:, full_group], axis=1)
            half[:, full_group] = 0.0
            half[:, rep] = total_mass

            # Update labels so every group member carries the smallest size.
            for k in non_rep:
                merged_msg_sizes[k] = msg_size_sets[rep]

            assigned[full_group] = True

            print(
                f"  [{half_name}] merged bins {full_group} "
                f"({[int(msg_size_sets[k]) for k in full_group]}B) "
                f"→ rep={int(msg_size_sets[rep])}B  "
                f"mass={total_mass.tolist()}"
            )

    return vec_x_merged


# =============================================================
# Utility function to print solution summary
# =============================================================
def print_solution_summary(
    node_names: list[str],
    lambda_used: list[float],
    vec_x: np.ndarray,
    msg_size_sets: np.ndarray,
    n_send: dict[str, float],
    n_recv: dict[str, int],
) -> None:
    """
    Print a per-node summary of the MPI model solution.
    """
    num_msg_sizes = vec_x.shape[1] // 2

    print("\n=== Solution Summary ===")
    header = (
        f"  {'Node':<20} {'lambda':>12} {'active_bins':>12} "
        f"{'solved_send':>14} {'solved_recv':>14} "
        f"{'expect_send':>14} {'expect_recv':>14}"
    )
    print(header)
    print("  " + "-" * 114)

    for node_idx, (name, lambda_val) in enumerate(zip(node_names, lambda_used, strict=True)):
        x_send = vec_x[node_idx, :num_msg_sizes]
        x_recv = vec_x[node_idx, num_msg_sizes:]

        active = int(np.sum(x_send > 0.5) + np.sum(x_recv > 0.5))
        solved_send = int(np.round(np.sum(x_send)))
        solved_recv = int(np.round(np.sum(x_recv)))

        expect_send_str = f"{n_send[name]:.1f}" if n_send is not None and name in n_send else "N/A"
        expect_recv_str = f"{n_recv[name]}" if n_recv is not None and name in n_recv else "N/A"

        row = (
            f"  {name:<20} {lambda_val:>12.3e} {active:>12} "
            f"{solved_send:>14} {solved_recv:>14} "
            f"{expect_send_str:>14} {expect_recv_str:>14}"
        )
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


def validate_solution(
    matrix_a: np.ndarray,
    vec_y: np.ndarray,
    vec_x: np.ndarray,
    node_names: list[str],
    rel_tol: float = 0.10,
) -> bool:
    num_nodes = vec_y.shape[0]

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
