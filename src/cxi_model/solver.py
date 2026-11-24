"""
solver.py

Solves the MPI communication model for all nodes.

All constants imported from constants.py.
"""

import warnings

import cvxpy as cp
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, hstack, kron
from scipy.sparse import eye as sp_eye

from cxi_model.constants import (
    COUNTER_GROUPS,
    FIT_ABS_TOL,
    FIT_REL_TOL_LADDER,
    OBJ_ABS_TOL,
    OBJ_REL_TOL,
    QP_SOLVER_LIST,
    RX_HIST_SLICE,
    TX_HIST_SLICE,
)


# =============================================================
# Global solver — single run
# =============================================================
def solve_global(
    matrix_a: np.ndarray,
    vec_y: np.ndarray,
    nic_names: list[str],
    msg_size_sets: np.ndarray,
    nic_recv_msgs: dict[str, int],
) -> np.ndarray:
    """
    Reconstruct send/recv message counts per size bin for every NIC of a job.

    Parameters
    ----------
    matrix_a : (H, 2M) signature matrix shared by all NICs
    vec_y    : (N, H) observed counter increments, one row per NIC
    nic_names : NIC labels in the same order as vec_y rows
    msg_size_sets : (M,) representative message size per bin (ascending)
    nic_recv_msgs : L_k, received-message total per NIC

    Returns
    -------
    vec_x : (N, 2M) selected counts, [x_send | x_recv] per NIC
    """
    num_nics, all_counters_txrx = vec_y.shape
    num_msg_sizes = matrix_a.shape[1] // 2

    if all_counters_txrx != matrix_a.shape[0]:
        raise ValueError(
            f"Y has {all_counters_txrx} counters per NIC but A has {matrix_a.shape[0]} rows."
        )
    if num_msg_sizes != len(msg_size_sets):
        raise ValueError(f"A has {matrix_a.shape[1]} columns but {len(msg_size_sets)} size bins.")

    print(f"  A : {matrix_a.shape}   Y : {vec_y.shape}   X : ({num_nics}, {matrix_a.shape[1]})\n")

    recv_totals = np.array([float(nic_recv_msgs[name]) for name in nic_names])

    # ----------------------------------------------------------
    # Stage 1 — joint weighted least squares (Eq. 8)
    # ----------------------------------------------------------
    print("  Stage 1: joint weighted least-squares fit (Eq. 8)")
    x_fit, obj_fit = _solve_joint_fit(matrix_a, vec_y, recv_totals, num_msg_sizes)
    y_hat_star = x_fit @ matrix_a.T  # unique optimal fitted counters, one row per NIC
    print(
        f"    objective = {obj_fit:.6e}   bins with mass = {_count_used_bins(x_fit, num_msg_sizes)}"
    )

    # ----------------------------------------------------------
    # Stage 2 — fewest distinct bins among solutions with A x_k = y_hat*_k
    # ----------------------------------------------------------
    print("  Stage 2: minimum-bin selection among optimal fits")
    obj_limit = obj_fit + OBJ_ABS_TOL + OBJ_REL_TOL * obj_fit
    vec_x = None

    # (a) MILP proposal: global minimum number of bins, verified by refitting.
    #     At large counter magnitudes the MILP can be numerically unreliable
    #     (one-count precision on ~1e9 totals), so every proposal is checked.
    for rel_tol in FIT_REL_TOL_LADDER:
        support = _select_min_bins(
            matrix_a, y_hat_star, recv_totals, num_msg_sizes, rel_tol=rel_tol
        )
        if support is None:
            print(f"    rel_tol={rel_tol:.0e}: no feasible support")
            continue

        # Refit on the candidate support and check that it is still optimal.
        x_restricted, obj_restricted = _solve_joint_fit(
            matrix_a, vec_y, recv_totals, num_msg_sizes, support=support
        )
        accepted = obj_restricted <= obj_limit
        print(
            f"    rel_tol={rel_tol:.0e}: {int(support.sum())} bins, refit objective = "
            f"{obj_restricted:.6e}  [{'accepted' if accepted else 'rejected'}]"
        )
        if accepted:
            vec_x = x_restricted
            break

    if vec_x is None:
        print("    no verified MILP support; starting elimination from the stage-1 fit")
        vec_x, support = x_fit, _used_bins_mask(x_fit, num_msg_sizes)

    # Backward elimination: drop further bins while the optimum is preserved.
    # Guarantees an inclusion-minimal support even when the MILP is unreliable.
    vec_x, support = _prune_bins(
        matrix_a, vec_y, recv_totals, num_msg_sizes, vec_x, support, obj_limit
    )
    sizes = [int(msg_size_sets[m]) for m in np.flatnonzero(support)]
    print(f"    final support: {int(support.sum())} bins: {sizes} B")

    for nic_idx, name in enumerate(nic_names):
        x_nic = vec_x[nic_idx, :]
        active_bins = int(np.sum(x_nic > 0.5))
        residual = np.linalg.norm(matrix_a @ x_nic - vec_y[nic_idx, :])
        print(f"    {name:<24} active_bins={active_bins}  residual={residual:.2f}")
    print()

    return vec_x


# =============================================================
# Stage 1 — joint constrained weighted least squares
# =============================================================
def _solve_joint_fit(
    matrix_a: np.ndarray,
    vec_y: np.ndarray,
    recv_totals: np.ndarray,
    num_msg_sizes: int,
    support: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """
    Solve Eq. (8) for all NICs jointly.

    If `support` (bool, shape (M,)) is given, bins outside it are fixed to zero
    in both directions for every NIC.

    Returns the message counts x (N, 2M) and the objective value
    sum_k sum_h [(A x_k)_h - y_{k,h}]^2 / max(y_{k,h}, 1).

    Numerics: counters can reach ~1e9 while a zero counter must be matched to
    within one count, so the problem is solved in packet units,
        u_{k,j} = pkts_per_msg_j * x_{k,j} / count_scale,
    which keeps every column of the scaled signature matrix in [0, 1].
    The minimizer is unchanged.
    """
    num_nics = vec_y.shape[0]
    n_cols = matrix_a.shape[1]

    pkts_per_msg = np.maximum(np.sum(matrix_a, axis=0), 1.0)  # send and recv columns of a bin match
    count_scale = max(1.0, float(np.max(vec_y)))
    matrix_a_pkt = matrix_a / pkts_per_msg
    inv_send = 1.0 / pkts_per_msg[:num_msg_sizes]
    inv_recv = 1.0 / pkts_per_msg[num_msg_sizes:]

    # sqrt of the Eq. (8) weights, in scaled units
    row_weights = count_scale / np.sqrt(np.maximum(vec_y, 1.0))

    vec_u = cp.Variable((num_nics, n_cols), nonneg=True)
    msgs_send = cp.multiply(vec_u[:, :num_msg_sizes], inv_send)  # x_send / count_scale
    msgs_recv = cp.multiply(vec_u[:, num_msg_sizes:], inv_recv)  # x_recv / count_scale

    # Row k of (U A_pkt^T) is (A x_k) / count_scale, so each entry of
    # weighted_residual is ((A x_k)_h - y_{k,h}) / sqrt(max(y_{k,h}, 1)) as in Eq. (8)
    weighted_residual = cp.multiply(row_weights, vec_u @ matrix_a_pkt.T - vec_y / count_scale)
    objective = cp.Minimize(cp.sum_squares(weighted_residual))

    constraints: list = [
        cp.sum(msgs_recv, axis=1) == recv_totals / count_scale,  # receive totals anchor each NIC
        cp.sum(msgs_send, axis=0) == cp.sum(msgs_recv, axis=0),  # per-bin job-wide conservation
    ]
    if support is not None:
        off = np.flatnonzero(~support)
        if off.size > 0:
            constraints.append(vec_u[:, off] == 0)
            constraints.append(vec_u[:, num_msg_sizes + off] == 0)

    prob = cp.Problem(objective, constraints)

    for solver in QP_SOLVER_LIST:
        try:
            prob.solve(solver=solver, verbose=False, canon_backend=cp.SCIPY_CANON_BACKEND)
            if vec_u.value is not None and prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                break
        except cp.SolverError:
            continue

    if vec_u.value is None:
        warnings.warn("All solvers failed. Returning zeros.", stacklevel=2)
        return np.zeros((num_nics, n_cols)), float("inf")

    # back to message counts; clip solver round-off below zero
    x_val = np.maximum(vec_u.value, 0.0) * count_scale / pkts_per_msg
    obj_val = float(np.sum((x_val @ matrix_a.T - vec_y) ** 2 / np.maximum(vec_y, 1.0)))
    return x_val, obj_val


# =============================================================
# Stage 2 — fewest distinct message-size bins (MILP)
# =============================================================
def _select_min_bins(
    matrix_a: np.ndarray,
    y_hat_star: np.ndarray,
    recv_totals: np.ndarray,
    num_msg_sizes: int,
    rel_tol: float,
    abs_tol: float = FIT_ABS_TOL,
    time_limit: float = 300.0,
) -> np.ndarray | None:
    """
    Find the smallest set of size bins that reproduces y_hat* for every NIC.

    Variables: x (N * 2M, continuous >= 0, NIC-major) and z (M, binary),
    where z_m = 1 if bin m is used by any NIC in either direction.
    Because of per-bin conservation, a bin used for receiving anywhere in the
    job is also used for sending somewhere, so one indicator per bin suffices.

    Ties between equally small supports (e.g. bins with identical signature
    columns) are broken in favor of smaller message sizes.

    Returns a bool mask of shape (M,), or None if the MILP fails.
    """
    num_nics, num_cntrs = y_hat_star.shape
    n_cols = 2 * num_msg_sizes
    n_x = num_nics * n_cols
    eye_n = sp_eye(num_nics, format="csr")

    # Normalize message counts so the big-M bounds below are <= 1
    count_scale = max(1.0, float(np.sum(recv_totals)))
    y_tilde = y_hat_star / count_scale
    l_tilde = recv_totals / count_scale

    # --- A x_k within tolerance of y_hat*_k for every NIC ---
    a_block = kron(eye_n, csr_matrix(matrix_a), format="csr")  # (N*H, N*2M)
    y_flat = y_tilde.ravel()
    tol = np.maximum(abs_tol / count_scale, rel_tol * np.abs(y_flat))
    fit_con = LinearConstraint(
        hstack([a_block, csr_matrix((num_nics * num_cntrs, num_msg_sizes))]),
        y_flat - tol,
        y_flat + tol,
    )

    # --- sum_m x^recv_{k,m} = L_k ---
    recv_row = np.concatenate([np.zeros(num_msg_sizes), np.ones(num_msg_sizes)])
    recv_con = LinearConstraint(
        hstack([kron(eye_n, csr_matrix(recv_row)), csr_matrix((num_nics, num_msg_sizes))]),
        l_tilde,
        l_tilde,
    )

    # --- sum_k x^send_{k,m} - sum_k x^recv_{k,m} = 0 ---
    eye_m = sp_eye(num_msg_sizes, format="csr")
    per_nic = hstack([eye_m, -eye_m])
    cons_con = LinearConstraint(
        hstack([kron(np.ones((1, num_nics)), per_nic), csr_matrix((num_msg_sizes, num_msg_sizes))]),
        0.0,
        0.0,
    )

    # --- x^d_{k,m} <= U_{k,d} * z_m  (big-M link) ---
    # recv counts are bounded by L_k; send counts by the job-wide receive total.
    upper = np.concatenate(
        [
            np.concatenate([np.full(num_msg_sizes, l_tilde.sum()), np.full(num_msg_sizes, l_k)])
            for l_k in l_tilde
        ]
    )
    bin_of_col = np.tile(np.arange(num_msg_sizes), 2 * num_nics)  # bin index of each x entry
    link = csr_matrix((-upper, (np.arange(n_x), bin_of_col)), shape=(n_x, num_msg_sizes))
    link_con = LinearConstraint(hstack([sp_eye(n_x, format="csr"), link]), -np.inf, 0.0)

    # --- objective: number of bins, plus a tie-break (< 1 in total) toward small sizes ---
    tie_break = np.arange(num_msg_sizes) / (num_msg_sizes * (num_msg_sizes + 1.0))
    cost = np.concatenate([np.zeros(n_x), 1.0 + tie_break])

    integrality = np.concatenate([np.zeros(n_x), np.ones(num_msg_sizes)])
    bounds = Bounds(
        np.zeros(n_x + num_msg_sizes),
        np.concatenate([upper, np.ones(num_msg_sizes)]),
    )

    res = milp(
        cost,
        constraints=[fit_con, recv_con, cons_con, link_con],
        integrality=integrality,
        bounds=bounds,
        options={"time_limit": time_limit, "disp": False},
    )

    if res.x is None:
        return None
    if res.status != 0:
        warnings.warn(f"MILP not proven optimal ({res.message}); using best found.", stacklevel=2)

    return np.round(res.x[n_x:]).astype(bool)


# =============================================================
# Stage 2 (fallback/refinement) — backward elimination of bins
# =============================================================
def _prune_bins(
    matrix_a: np.ndarray,
    vec_y: np.ndarray,
    recv_totals: np.ndarray,
    num_msg_sizes: int,
    vec_x: np.ndarray,
    support: np.ndarray,
    obj_limit: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove bins from `support` one at a time while refitting Eq. (8) on the
    remaining bins stays within `obj_limit`.

    Bins with the least job-wide mass are tried first; among equal masses
    (e.g. bins with identical signature columns) the larger size is dropped
    first, so the smaller representative is kept. Every accepted removal is
    verified, so the result is an optimal fit on an inclusion-minimal support.
    """
    support = support.copy()

    # First, drop in one shot every bin carrying less than half a message job-wide.
    tiny = support & (_bin_mass(vec_x, num_msg_sizes) < 0.5)
    if tiny.any() and tiny.sum() < support.sum():
        trial = support & ~tiny
        x_trial, obj_trial = _solve_joint_fit(
            matrix_a, vec_y, recv_totals, num_msg_sizes, support=trial
        )
        if obj_trial <= obj_limit:
            support, vec_x = trial, x_trial
            print(f"    dropped {int(tiny.sum())} near-empty bins")

    changed = True
    while changed and support.sum() > 1:
        changed = False
        mass = _bin_mass(vec_x, num_msg_sizes)
        scale = max(float(mass.sum()), 1.0)
        candidates = sorted(np.flatnonzero(support), key=lambda m: (round(mass[m] / scale, 6), -m))
        for m in candidates:
            trial = support.copy()
            trial[m] = False
            x_trial, obj_trial = _solve_joint_fit(
                matrix_a, vec_y, recv_totals, num_msg_sizes, support=trial
            )
            if obj_trial <= obj_limit:
                support, vec_x = trial, x_trial
                changed = True
                break  # masses changed; re-rank the remaining bins

    return vec_x, support


def _bin_mass(vec_x: np.ndarray, num_msg_sizes: int) -> np.ndarray:
    """Job-wide number of messages per size bin (send + recv)."""
    return np.sum(vec_x[:, :num_msg_sizes] + vec_x[:, num_msg_sizes:], axis=0)


def _used_bins_mask(vec_x: np.ndarray, num_msg_sizes: int, threshold: float = 0.5) -> np.ndarray:
    """Bool mask of size bins with more than `threshold` messages anywhere in the job."""
    used = (vec_x[:, :num_msg_sizes] > threshold) | (vec_x[:, num_msg_sizes:] > threshold)
    return np.any(used, axis=0)


def _count_used_bins(vec_x: np.ndarray, num_msg_sizes: int, threshold: float = 0.5) -> int:
    """Number of size bins with more than `threshold` messages anywhere in the job."""
    return int(np.sum(_used_bins_mask(vec_x, num_msg_sizes, threshold)))


# =============================================================
# Utility function to print solution summary
# =============================================================
def print_solution_summary(
    nic_names: list[str],
    vec_x: np.ndarray,
    msg_size_sets: np.ndarray,
    n_recv: dict[str, int],
) -> None:
    """
    Print a per-NIC summary of the reconstructed message-size distribution.
    """
    num_msg_sizes = vec_x.shape[1] // 2

    print("\n=== Solution Summary ===")
    header = (
        f"  {'NIC':<24} {'active_bins':>12} "
        f"{'solved_send':>14} {'solved_recv':>14} {'L_k (recv)':>14}"
    )
    print(header)
    print("  " + "-" * 82)

    for nic_idx, name in enumerate(nic_names):
        x_send = vec_x[nic_idx, :num_msg_sizes]
        x_recv = vec_x[nic_idx, num_msg_sizes:]

        active = int(np.sum(x_send > 0.5) + np.sum(x_recv > 0.5))
        solved_send = int(np.round(np.sum(x_send)))
        solved_recv = int(np.round(np.sum(x_recv)))

        expect_recv_str = f"{n_recv[name]}" if n_recv is not None and name in n_recv else "N/A"

        row = f"  {name:<24} {active:>12} {solved_send:>14} {solved_recv:>14} {expect_recv_str:>14}"
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

    # Job-wide view: per-bin conservation and number of distinct bins
    job_send = np.sum(vec_x[:, :num_msg_sizes], axis=0)
    job_recv = np.sum(vec_x[:, num_msg_sizes:], axis=0)
    used = np.flatnonzero((job_send > 0.5) | (job_recv > 0.5))
    max_gap = float(np.max(np.abs(job_send - job_recv))) if num_msg_sizes else 0.0

    print(f"\n  Job-wide distinct bins : {len(used)}")
    for m in used:
        print(
            f"    {int(msg_size_sets[m]):>10}B   "
            f"send={job_send[m]:>14.2f}   recv={job_recv[m]:>14.2f}"
        )
    print(f"  Max per-bin |send - recv| : {max_gap:.3e}")


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
    print(f"  NIC : {node_name}")
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
            elif pred < 0.5:
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
    print(f"  NIC status : {verdict} — {direction} {rel_tol * 100:.0f}% relative tolerance")

    return node_pass
