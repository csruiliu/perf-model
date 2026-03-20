"""
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

from constants import print_constants, MSG_SIZE_SETS
from load_counters import load_counters, load_multiple_runs
from build_matrix import build_matrixA, print_matrix_summary
from solve_global import (
    solve_global,
    solve_global_stacked,
    extract_send_recv,
    print_solution_summary,
    print_lambda_summary,
)


def main():
    parser = argparse.ArgumentParser(description="MPI Communication Profiler — hardware counter model")
    parser.add_argument("results_dirs", type=Path, nargs="+",
                        help="One or more SLURM job result directories (OMB_<job_id>)")
    parser.add_argument("--msg_set", default="default", choices=list(MSG_SIZE_SETS.keys()),
                        help="Message size bin set: default | fine | coarse  (default: default)")
    parser.add_argument("--lambda_val", default="auto",
                        help="Regularization lambda: float or 'auto' (default: auto)")
    parser.add_argument("--method", default="lcurve", choices=["lcurve", "loco_cv"],
                        help="Lambda tuning method (default: lcurve)")
    parser.add_argument("--n_points", type=int, default=40,
                        help="Lambda grid size for auto tuning (default: 40)")
    parser.add_argument("--solver", default="CLARABEL", choices=["CLARABEL", "SCS", "ECOS"],
                        help="CVXPY solver backend (default: CLARABEL)")
    parser.add_argument("--plot", action="store_true", help="Save L-curve plots per node")
    parser.add_argument("--plot_dir", default="plots", help="Directory for L-curve plots (default: plots/)")
    parser.add_argument("--print_constants", action="store_true", help="Print all constants and exit")
    args = parser.parse_args()

    if args.print_constants:
        print_constants()

    # Resolve message size set
    msg_sizes = MSG_SIZE_SETS[args.msg_set]
    n_msg     = len(msg_sizes)
    print(f"Message size set : '{args.msg_set}'  ({n_msg} bins)")

    # Parse lambda_val — float or 'auto'
    if args.lambda_val == "auto":
        lambda_val = "auto"
    else:
        try:
            lambda_val = float(args.lambda_val)
        except ValueError:
            parser.error(f"--lambda_val must be a float or 'auto', got '{args.lambda_val}'")

    # ----------------------------------------------------------
    # Step 1 — Load hardware counters
    # ----------------------------------------------------------
    print("=" * 60)
    print("Step 1: Load hardware counters")
    print("=" * 60)

    n_runs = len(args.results_dirs)

    if n_runs == 1:
        Y, node_names = load_counters(args.results_dirs[0])
        Y_for_solver  = Y
        Y_for_report  = Y
    else:
        Y_multi, node_names = load_multiple_runs(args.results_dirs)
        Y_for_solver        = Y_multi
        Y_for_report        = Y_multi[:, 0, :]

    # ----------------------------------------------------------
    # Step 2 — Build signature matrix A
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 2: Build system signature matrix A")
    print("=" * 60)

    A = build_matrixA(msg_sizes)
    print_matrix_summary(A, msg_sizes)

    # ----------------------------------------------------------
    # Step 3 — Solve global optimization
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 3: Solve global optimization")
    print("=" * 60)

    solver_kwargs = dict(
        lambda_val = lambda_val,
        method = args.method,
        n_points = args.n_points,
        node_names = node_names,
        solver = args.solver,
        plot = args.plot,
        plot_dir = args.plot_dir,
    )

    if n_runs == 1:
        X, lambdas_used = solve_global(A, Y_for_solver, **solver_kwargs)
    else:
        X, lambdas_used = solve_global_stacked(A, Y_for_solver, **solver_kwargs)

    # ----------------------------------------------------------
    # Step 4 — Report results
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 4: Results")
    print("=" * 60)
    
    print_lambda_summary(node_names, lambdas_used, X)
    print()
    print_solution_summary(A, X, Y_for_report, node_names, msg_sizes=msg_sizes)

    # ----------------------------------------------------------
    # Final output shapes — ready for downstream use
    # ----------------------------------------------------------
    x_send = X[:, :n_msg]
    x_recv = X[:, n_msg:] 

    print("\n" + "=" * 60)
    print("Output shapes ready for downstream use:")
    print("=" * 60)
    print(f"  X      : {X.shape}  (N nodes, 2*N_MSG bins)")
    print(f"  x_send : {x_send.shape}  (N nodes, N_MSG send bins)")
    print(f"  x_recv : {x_recv.shape}  (N nodes, N_MSG recv bins)")


if __name__ == "__main__":
    main()