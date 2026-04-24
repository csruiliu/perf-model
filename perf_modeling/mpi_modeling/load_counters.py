"""
load_counters.py

Loads Cassini hardware counter data from the two-level directory
structure produced by sbatch scripts we used on Perlmutter.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from constants import ALL_CNTRS, MISC_CNTRS, NUM_ALL_CNTRS, RX_HIST_SLICE, TX_HIST_SLICE


# =============================================================
# Top-level loader — single run
# =============================================================
def load_counters_single_job(counter_dir: str | Path) -> tuple[np.ndarray, list[str]]:
    """
    Load all hardware counters from the three-level results directory.

    Auto-discovers node directories by finding any subdirectory
    containing cxi0/counters.csv. All other files are ignored.

    Parameters
    ----------
    counter_dir : str or Path
        Top-level directory named by SLURM job ID.
        Example: "/pscratch/.../results/OMB_12345678"
    """
    counter_dir = Path(counter_dir)

    if not counter_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {counter_dir}")
    if not counter_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {counter_dir}")

    print(f"Loading counters from: {counter_dir}")

    # find and sort all subdirs inside counter_dir that contain a cxi0/counters.csv file.
    node_dirs = sorted(
        [d for d in counter_dir.iterdir() if d.is_dir() and (d / "cxi0" / "counters.csv").exists()]
    )

    if len(node_dirs) == 0:
        raise FileNotFoundError(
            f"No node directories with counters.csv found under:\n"
            f"  {counter_dir}\n"
            f"Expected: {counter_dir}/<node_name>/cxi0/counters.csv"
        )

    num_nodes = len(node_dirs)

    # create Y vector with shape (num_nodes, 2 * NUM_ALL_CNTRS)
    vector_y = np.zeros((num_nodes, 2 * NUM_ALL_CNTRS), dtype=np.float64)
    # create list of node names in the same order as Y
    node_names: list[str] = []
    total_messages: dict[str, int] = {}

    for node_idx, node_dir in enumerate(node_dirs):
        node_name = node_dir.name
        node_names.append(node_name)

        y_node, n_total = load_counters_single_node(node_dir / "cxi0" / "counters.csv")
        vector_y[node_idx, :] = y_node
        total_messages[node_name] = n_total

        print(f"  Loaded [{node_idx + 1}/{num_nodes}] {node_name}  (n_total={n_total})")

    # check loaded data for common issues before returning
    _validate(vector_y, node_names)

    print(
        f"\nDone. Y shape: {vector_y.shape}  (N={num_nodes} nodes, each has {2 * NUM_ALL_CNTRS} counters)"
    )

    return vector_y, node_names, total_messages


# =============================================================
# Single node loader
# =============================================================
def load_counters_single_node(counter_file: Path) -> np.ndarray:
    """
    Load counters.csv for a single node.

    counters.csv is the delta (after - before).
    It has three columns:
        counter_name, direction, value

    Returns
    -------
    y_node : np.ndarray, ordered counter vector y for single node following ALL_CNTRS.
    """
    df = pd.read_csv(counter_file)

    required_cols = {"counter_name", "direction", "value"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"{counter_file}: missing required columns.\n"
            f"  Required : {required_cols}\n"
            f"  Found    : {set(df.columns)}"
        )

    counter_lookup: dict[str, float] = dict(zip(df["counter_name"], df["value"], strict=True))

    # check that all expected counters are present
    missing = [name for name in ALL_CNTRS if name not in counter_lookup]
    if missing:
        raise KeyError(
            f"{counter_file}: missing counters:\n" + "\n".join(f"  - {m}" for m in missing)
        )

    # create y_node vector in the order of ALL_CNTRS, with TX counters followed by RX counters
    y_node = np.array([counter_lookup[name] for name in ALL_CNTRS], dtype=np.float64)

    if np.any(y_node < 0):
        raise ValueError(f"{counter_file}: negative counter values detected.")

    # --- total message count ---
    missing_msg = [name for name in MISC_CNTRS if name not in counter_lookup]
    if missing_msg:
        raise KeyError(
            f"{counter_file}: missing message-total counters:\n"
            + "\n".join(f"  - {m}" for m in missing_msg)
        )

    n_total = int(counter_lookup["lpe_net_match_priority_0"]) + int(
        counter_lookup["lpe_net_match_overflow_0"]
    )

    if n_total < 0:
        raise ValueError(f"{counter_file}: negative total message count ({n_total}).")

    return y_node, n_total


# =============================================================
# Validation
# =============================================================
def _validate(vector_y: np.ndarray, node_names: list[str]) -> None:
    """Check loaded counter data for common issues."""
    num_nodes = vector_y.shape[0]
    print("\n  Validating...")

    # make sure all counters are non-negative
    if np.any(vector_y < 0):
        raise ValueError("Negative counter values detected.")
    print("    [OK] All counters non-negative")

    # make sure no node has all-zero counters (indicating missing data)
    for node_idx in range(num_nodes):
        if np.all(vector_y[node_idx, :] == 0):
            print(f"    [WARN] {node_names[node_idx]}: all counters zero — possible missing data")

    # Collect all packet counts from histogram counters
    total_tx = np.sum(vector_y[:, TX_HIST_SLICE])
    total_rx = np.sum(vector_y[:, RX_HIST_SLICE])

    # Check that TX and RX totals are within a reasonable ratio.
    if total_tx > 0 and total_rx > 0:
        ratio = total_tx / total_rx
        status = "[OK]" if 0.5 <= ratio <= 2.0 else "[WARN]"
        print(f"    {status} TX/RX packet ratio = {ratio:.2f}")

    print("    Validation complete.")
