"""
modeling_cxi.py

Full pipeline for the network time model.
"""

import argparse
from pathlib import Path

from build_matrix import build_matrix_a, validate_matrix_a
from constants import MSG_SIZE_SETS
from hw_config.hw_specs import HostSpec
from load_counters import load_counters_single_job
from solver import print_solution_summary, solve_global, validate_solution
from time_estimator import create_direct_lookup_model, time_estimation


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
        "--latency_file",
        type=Path,
        default=None,
        help="Path to OSU latency benchmark output (e.g., osu_latency.out)",
    )

    parser.add_argument(
        "-rh",
        "--ref_host",
        type=str,
        required=True,
        choices=list(HostSpec.keys()),
        help="Reference Host",
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
    if not args.latency_file or not args.latency_file.exists():
        print(
            "\n[INFO] No OSU latency file provided or founded — skipping Step 4 (time estimation)."
        )
        return

    print("\n" + "=" * 60)
    print("Step 4: Estimate Communication Time Upper Bound")
    print("=" * 60)

    # Choose the latency calculation method
    print("Using direct raw latency lookup...")
    latency_model = create_direct_lookup_model(args.latency_file)

    n_msg_sizes = len(msg_size_sets)
    x_send = vec_x[:, :n_msg_sizes]
    x_recv = vec_x[:, n_msg_sizes:]

    overlap_time_us, sequential_time_us, heaviest_node_idx = time_estimation(
        x_send, x_recv, msg_size_sets, latency_model, args.ref_host
    )

    heaviest_node_name = n_names[heaviest_node_idx] if n_names else str(heaviest_node_idx)

    print("\n  === Time Estimation Results ===")
    print(f"  Heaviest communicating node : {heaviest_node_name}")
    print(f"  Overlap mode (Max node time): {overlap_time_us / 1e6:.4f} seconds")
    print(f"  Sequential mode (Total time): {sequential_time_us / 1e6:.4f} seconds")


if __name__ == "__main__":
    main()
