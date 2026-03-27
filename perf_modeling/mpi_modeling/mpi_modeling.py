"""
mpi_modeling.py

Full pipeline entry point for the MPI communication model.

Usage:
    # Single run, auto lambda
    python main.py /pscratch/.../results/OMB_12345678

    # Multiple runs stacked, auto lambda
    python main.py /pscratch/.../results/OMB_111 OMB_222 OMB_333

    # Fixed lambda, no tuning
    python main.py /pscratch/.../results/OMB_12345678 --lambda_val 0.05

    # With L-curve plots saved
    python main.py /pscratch/.../results/OMB_12345678 --plot
"""

import argparse
from pathlib import Path

from constants import MSG_SIZE_SETS
from load_counters import load_counters_single_job
from build_matrix import build_matrixA, validate_matrixA
from solver import print_solution_summary, solve_global


def main():
    parser = argparse.ArgumentParser(description="MPI Communication Profiler — hardware counter model")
    
    parser.add_argument("-f", "--counter_dir", type=Path,
                        help="One SLURM job directories with collected counters (OMB_<job_id>)")
    
    parser.add_argument("--msg_set", default="fine", choices=list(MSG_SIZE_SETS.keys()),
                        help="Message size bin set: fine | coarse | pm (default: fine)")
    
    parser.add_argument("--lambda_val", default="auto",
                        help="Regularization lambda: float or 'auto' (default: auto)")
    
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
    Y, node_names = load_counters_single_job(args.counter_dir)    
    Y_for_solver  = Y

    # ----------------------------------------------------------
    # Step 2 — Build signature matrix A
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 2: Build system signature matrix A")
    print("=" * 60)
    
    # A's shape is (2 * NUM_ALL_CNTRS, 2 * num_msg_sizes)
    A = build_matrixA(msg_size_sets)
    #validate_matrixA(A, msg_size_sets, target_size=128, count=100000, case_name="TEST CASE 1 (Eager Protocol)")

    # ----------------------------------------------------------
    # Step 3 — Solve global optimization
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 3: Solve global optimization")
    print("=" * 60)

    # X's shape is (num_nodes, 2 * num_msg_size_bins)
    X, lambdas_used = solve_global(A, Y_for_solver, node_names)
    print_solution_summary(node_names, lambdas_used, X, msg_size_sets)




if __name__ == "__main__":
    main()