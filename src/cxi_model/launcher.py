"""
modeling_cxi.py

Full pipeline for the network time model.
"""

import argparse
from pathlib import Path

from cxi_model.constants import MSG_SIZE_SETS
from cxi_model.hw_specs import HostSpec
from cxi_model.job_parser import load_counters_single_job
from cxi_model.matrix_builder import build_matrix_a, validate_matrix_a
from cxi_model.solver import print_solution_summary, solve_global, validate_solution


def main():
    parser = argparse.ArgumentParser(description="Communication Time Modeling Based on CXI")

    parser.add_argument(
        "--counter_dir",
        type=Path,
        required=True,
        help="One job directories with collected counters (e.g., OMB_<job_id>)",
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
    y_solver, n_names, nic_recv_msgs = load_counters_single_job(args.counter_dir)
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
    vec_x = solve_global(matrix_a, y_solver, n_names, msg_size_sets, nic_recv_msgs)

    print_solution_summary(n_names, vec_x, msg_size_sets, nic_recv_msgs)

    # Validate that predicted counters match observed counters
    validate_solution(matrix_a, y_solver, vec_x, n_names, rel_tol=0.05)


if __name__ == "__main__":
    main()
