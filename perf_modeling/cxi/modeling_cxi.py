"""
mpi_modeling.py

Full pipeline entry point for the MPI communication model.
"""

import argparse
from pathlib import Path

from build_matrix import build_matrix_a, validate_matrix_a
from constants import MSG_SIZE_SETS
from load_counters import load_counters_single_job
from solver import print_solution_summary, solve_global, validate_solution
from time_estimator import compute_upper_bound_time, create_direct_lookup_model


def main():
    parser = argparse.ArgumentParser(
        description="MPI Communication Profiler — hardware counter model"
    )

    parser.add_argument(
        "--counter_dir",
        type=Path,
        help="One SLURM job directories with collected counters (OMB_<job_id>)",
    )
    parser.add_argument(
        "--msg_set",
        default="fine",
        choices=list(MSG_SIZE_SETS.keys()),
        help="Message size bin set: fine | coarse | pm (default: fine)",
    )
    parser.add_argument(
        "--latency_file",
        type=Path,
        default=None,
        help="Path to OSU latency benchmark output (e.g., osu_latency.txt)",
    )
    parser.add_argument(
        "--latency_method",
        type=str,
        choices=["fit", "direct"],
        default="fit",
        help="Choose 'fit' for the piecewise Hockney model or 'direct' for raw benchmark lookup",
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
    y_for_solver, node_names, node_send_msgs, node_recv_msgs = load_counters_single_job(
        args.counter_dir
    )
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
        matrix_a, y_for_solver, node_names, msg_size_sets, node_send_msgs, node_recv_msgs
    )

    print_solution_summary(
        node_names, lambdas_used, vec_x, msg_size_sets, node_send_msgs, node_recv_msgs
    )

    # Validate that predicted counters match observed counters
    validate_solution(matrix_a, y_for_solver, vec_x, node_names, rel_tol=0.05)

    # ----------------------------------------------------------
    # Step 4 — Estimate Communication Time Upper Bound
    # ----------------------------------------------------------
    if not args.osu_latency_file or not args.osu_latency_file.exists():
        print(
            "\n[INFO] No OSU latency file provided or founded — skipping Step 4 (time estimation)."
        )
        return

    print("\n" + "=" * 60)
    print("Step 4: Estimate Communication Time Upper Bound")
    print("=" * 60)

    # Choose the latency calculation method
    print("Using direct raw latency lookup...")
    latency_model = create_direct_lookup_model(args.osu_latency_file)

    n_msg_sizes = len(msg_size_sets)
    x_send = vec_x[:, :n_msg_sizes]
    x_recv = vec_x[:, n_msg_sizes:]

    # current only consider latency and placeholder for gap model
    gap_model = None

    overlap_time_us, sequential_time_us, heaviest_node_idx = compute_upper_bound_time(
        x_send, x_recv, msg_size_sets, latency_model, gap_model
    )

    heaviest_node_name = node_names[heaviest_node_idx] if node_names else str(heaviest_node_idx)

    print("\n  === Time Estimation Results ===")
    print(f"  Heaviest communicating node : {heaviest_node_name}")
    print(f"  Overlap mode (Max node time): {overlap_time_us / 1e6:.4f} seconds")
    print(f"  Sequential mode (Total time): {sequential_time_us / 1e6:.4f} seconds")


if __name__ == "__main__":
    main()
