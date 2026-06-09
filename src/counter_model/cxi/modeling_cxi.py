"""
modeling_cxi.py

Full pipeline for the network time model.
"""

import argparse
from pathlib import Path

from counter_model.cxi.build_matrix import build_matrix_a, validate_matrix_a
from counter_model.cxi.constants import MSG_SIZE_SETS
from counter_model.cxi.load_counters import load_counters_single_job
from counter_model.cxi.solver import print_solution_summary, solve_global, validate_solution
from counter_model.cxi.time_estimator import (
    build_latency_model_from_config,
    build_latency_model_from_file,
    time_estimation,
)
from counter_model.hw_config.hw_specs import HostSpec
from counter_model.hw_config.pm_config import LATENCY_TABLES


def main():
    parser = argparse.ArgumentParser(description="Communication Time Modeling Based on CXI")

    parser.add_argument(
        "--counter_dir",
        type=Path,
        required=True,
        help="One SLURM job directories with collected counters (OMB_<job_id>)",
    )

    parser.add_argument(
        "--msg_set",
        default="fine",
        type=str,
        required=True,
        choices=list(MSG_SIZE_SETS.keys()),
        help="Message size bin set: fine | coarse | pm (default: fine)",
    )

    parser.add_argument(
        "-rh",
        "--ref_host",
        type=str,
        required=True,
        choices=list(HostSpec.keys()),
        help="Reference Host",
    )

    parser.add_argument(
        "--latency_file",
        type=Path,
        default=None,
        help="Optional path to latency file (e.g., osu_latency.out). If provided, overrides the built-in table.",
    )

    parser.add_argument(
        "--latency_table",
        type=str,
        default="omb",
        choices=list(LATENCY_TABLES.keys()),
        help="Which latency table from pm_config.py to use.",
    )

    args = parser.parse_args()

    # Resolve message size set
    msg_size_sets = MSG_SIZE_SETS[args.msg_set]
    print(f"Message size set : '{msg_size_sets}'  ({len(msg_size_sets)} bins)")

    # ----------------------------------------------------------
    # Step 1 — Load hardware counters
    # ----------------------------------------------------------
    print("=" * 60)
    print("Step 1: Load hardware counters")
    print("=" * 60)

    # Y has shape (num_nodes, 2 * NUM_ALL_CNTRS)
    # The first NUM_ALL_CNTRS columns are TX counters and the next NUM_ALL_CNTRS columns are RX counters
    y_solver, n_names, node_send_msgs, node_recv_msgs = load_counters_single_job(args.counter_dir)
    # ----------------------------------------------------------
    # Step 2 — Build signature matrix A
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 2: Build system signature matrix A")
    print("=" * 60)

    # Matrix A's shape is (2 * NUM_ALL_CNTRS, 2 * num_msg_sizes)
    matrix_a = build_matrix_a(msg_size_sets)
    validate_matrix_a(
        matrix_a,
        msg_size_sets,
        target_size=128,
        count=100000,
        case_name="TEST CASE 1 (Eager Protocol)",
    )

    # ----------------------------------------------------------
    # Step 3 — Solve global optimization
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 3: Solve global optimization")
    print("=" * 60)

    # X's shape is (num_nodes, 2 * num_msg_size_bins)
    vec_x, lambdas_used = solve_global(
        matrix_a, y_solver, n_names, msg_size_sets, node_send_msgs, node_recv_msgs
    )

    print_solution_summary(
        n_names, lambdas_used, vec_x, msg_size_sets, node_send_msgs, node_recv_msgs
    )

    # Validate that predicted counters match observed counters
    validate_solution(matrix_a, y_solver, vec_x, n_names, rel_tol=0.05)

    # ----------------------------------------------------------
    # Step 4 — Estimate Communication Time Upper Bound
    # ----------------------------------------------------------
    # Choose the latency source: explicit file overrides the built-in table.
    if args.latency_file and args.latency_file.exists():
        print(f"Using raw latency lookup from file: {args.latency_file}")
        latency_model = build_latency_model_from_file(args.latency_file)
    else:
        if args.latency_file:
            print(
                f"[INFO] Latency file '{args.latency_file}' not found — "
                f"Using pm_config table '{args.latency_table}'."
            )
        else:
            print(f"Using built-in latency table from pm_config: '{args.latency_table}'")
        latency_model = build_latency_model_from_config(args.latency_table)

    n_msg_sizes = len(msg_size_sets)
    x_send = vec_x[:, :n_msg_sizes]
    x_recv = vec_x[:, n_msg_sizes:]

    overlap_time_us, sequential_time_us = time_estimation(
        x_send, x_recv, msg_size_sets, latency_model, args.ref_host
    )

    print("\n  === Time Estimation Results ===")
    print(f"  Full Parallelism: {overlap_time_us / 1e6:.4f} seconds")
    print(f"  Full Sequentiality: {sequential_time_us / 1e6:.4f} seconds")


if __name__ == "__main__":
    main()
