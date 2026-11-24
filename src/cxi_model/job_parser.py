"""
load_counters.py

Loads Cassini hardware counter data from the two-level directory
structure produced by sbatch scripts we used on Perlmutter.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from cxi_model.constants import ALL_CNTRS, LPE_CNTRS, NUM_ALL_CNTRS, RX_HIST_SLICE, TX_HIST_SLICE


# =============================================================
# Top-level loader — single run
# =============================================================
def load_counters_single_job(counter_dir: str | Path) -> tuple[np.ndarray, list[str]]:
    """
    Load all hardware counters from the three-level results directory.

    Auto-discovers every NIC directory matching <node>/cxi*/counters.csv.
    Each NIC is one row of Y (NIC k in Sec. III-B). All other files are ignored.

    Parameters
    ----------
    counter_dir : str or Path
        Top-level directory named by SLURM job ID.
        Example: "/pscratch/.../results/OMB_12345678"

    Returns
    -------
    vector_y      : (N, 2 * NUM_ALL_CNTRS) counter increments, one row per NIC
    nic_name_list : "<node>/<cxiX>" labels in the same order as Y
    nic_recv_msgs : L_k, received-message total per NIC (priority + overflow matches)
    """
    counter_dir = Path(counter_dir)

    if not counter_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {counter_dir}")
    if not counter_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {counter_dir}")

    print(f"Loading counters from: {counter_dir}")

    # find and sort all <node>/cxi*/counters.csv files inside counter_dir
    nic_files = sorted(f for f in counter_dir.glob("*/cxi*/counters.csv") if f.is_file())

    if len(nic_files) == 0:
        raise FileNotFoundError(
            f"No node directories with counters.csv found under:\n"
            f"  {counter_dir}\n"
            f"Expected: {counter_dir}/<node_name>/cxi0/counters.csv"
        )

    num_nics = len(nic_files)

    # create Y vector with shape (num_nodes, 2 * NUM_ALL_CNTRS)
    vector_y = np.zeros((num_nics, 2 * NUM_ALL_CNTRS), dtype=np.float64)
    # create list of NIC names in the same order as Y
    nic_name_list: list[str] = []

    # L_k: received-message total of each NIC.
    # Send totals are not estimated here; the solver couples sends and receives
    # through per-bin conservation across the job (Eq. 8).
    nic_recv_msgs: dict[str, int] = {}

    for nic_idx, nic_file in enumerate(nic_files):
        nic_name = f"{nic_file.parent.parent.name}/{nic_file.parent.name}"
        nic_name_list.append(nic_name)

        y_nic, n_total = load_counters_single_node(nic_file)
        vector_y[nic_idx, :] = y_nic
        nic_recv_msgs[nic_name] = n_total

        print(f"  Loaded [{nic_idx + 1}/{num_nics}] {nic_name}  (L_k={n_total})")

    # check loaded data for common issues before returning
    _validate(vector_y, nic_name_list, nic_recv_msgs)

    print(
        f"\nDone. Y shape: {vector_y.shape}  "
        f"(N={num_nics} NICs, each has {2 * NUM_ALL_CNTRS} counters)"
    )

    return vector_y, nic_name_list, nic_recv_msgs


# =============================================================
# Single node loader
# =============================================================
def load_counters_single_node(counter_file: Path) -> tuple[np.ndarray, int]:
    """
    Load counters.csv for a single NIC.

    counters.csv is the delta (after - before).
    It has three columns:
        counter_name, direction, value

    Returns
    -------
    y_node  : np.ndarray, ordered counter vector y for a single NIC following ALL_CNTRS.
    n_total : int, received-message total L_k (priority + overflow list matches).
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
    missing_msg = [name for name in LPE_CNTRS if name not in counter_lookup]
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
def _validate(
    vector_y: np.ndarray,
    node_names: list[str],
    node_recv_msg_counts: dict[str, int],
) -> None:
    """Check loaded counter data for common issues."""
    if not node_names:
        raise ValueError("node_names is empty!")

    if not node_recv_msg_counts:
        raise ValueError("node_recv_msg_counts is empty!")

    if set(node_recv_msg_counts) != set(node_names):
        raise ValueError("node_recv_msg_counts contains NIC names not present in the loaded data")

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
