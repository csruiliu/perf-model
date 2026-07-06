import argparse
import os

from counter_model.dcgm.dispatcher import Dispatcher


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================
def valid_input(path: str) -> str:
    """Validate that the input is an existing folder, or a file with an
    allowed extension (.txt, .out, .pkl)."""
    valid_extensions = (".txt", ".out", ".pkl")
    if os.path.isdir(path):
        return path
    if os.path.isfile(path):
        if path.lower().endswith(valid_extensions):
            return path
        raise argparse.ArgumentTypeError(
            f"File must have one of these extensions: {', '.join(valid_extensions)}"
        )
    raise argparse.ArgumentTypeError(f"Path does not exist: {path}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DCGM-based Performance Modeling for Single Node",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- the two mode axes (node is always single) ---
    parser.add_argument(
        "--job_mode", required=True, choices=["single", "multi"], help="Single-job or multi-job"
    )
    parser.add_argument("--num_gpu", type=int, help="[multi-gpu] Number of GPUs")

    # --- shared ---
    parser.add_argument(
        "--dcgm_input",
        required=True,
        type=valid_input,
        help="DCGM input: file (.txt, .out, .pkl) or folder path",
    )
    parser.add_argument("-d", "--sample_interval_ms", type=int, required=True)
    parser.add_argument("-o", "--overall_runtime_ms", type=int, required=True)
    parser.add_argument("-st", "--start_timestamp", type=int, default=0)
    parser.add_argument("-et", "--end_timestamp", type=int, default=None)
    parser.add_argument("-rg", "--ref_gpu", type=str, required=True)
    parser.add_argument("-tg", "--tgt_gpu", type=str, default=None)
    parser.add_argument("-rh", "--ref_host", type=str, required=True)
    parser.add_argument("-th", "--tgt_host", type=str, default=None)
    parser.add_argument("--metrics", type=lambda s: s.split(","), required=True)
    parser.add_argument("--cores_alloc", choices=["same", "all"], help="CPU Cores Allocation")

    # --- conditionally-required ---
    parser.add_argument("--agg_interval_ms", type=int, help="[multi-gpu] Aggregation interval (ms)")

    args = parser.parse_args()
    _validate(parser, args)
    return args


def _validate(parser, args):
    """Requirements attach to an axis, not to a specific mode combination."""
    required = {}
    if args.job_mode == "multi":
        required.update({"agg_interval_ms": args.agg_interval_ms})

    missing = [n for n, v in required.items() if v is None]
    if missing:
        parser.error("missing required for this mode: " + ", ".join("--" + m for m in missing))


def main():
    args = parse_arguments()
    dispatcher = Dispatcher(args)
    dispatcher.dispatch()


if __name__ == "__main__":
    main()
