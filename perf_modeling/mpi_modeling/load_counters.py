"""
load_counters.py

Loads Cassini hardware counter data from the two-level directory
structure produced by sbatch scripts we used on Perlmutter.

All constants imported from constants.py.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict

from constants import (
    COUNTER_ORDER,
    TX_HIST_SLICE,
    RX_HIST_SLICE,
    N_HIST,
    TWO_M,
)


# =============================================================
# Single node loader
# =============================================================
def load_node_counters(counter_file: Path) -> np.ndarray:
    """
    Load counters.csv for a single node.

    counters.csv is the delta (after - before). 
    It has three columns:
        counter_name, direction, value

    Parameters
    ----------
    counter_file : Path
        Path to <node_name>/counters.csv

    Returns
    -------
    y_n : np.ndarray, shape (TWO_M,) = (30,)
        Ordered counter vector following COUNTER_ORDER.
    """
    df = pd.read_csv(counter_file)

    required_cols = {"counter_name", "direction", "value"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"{counter_file}: missing required columns.\n"
            f"  Required : {required_cols}\n"
            f"  Found    : {set(df.columns)}"
        )

    counter_lookup: Dict[str, float] = dict(
        zip(df["counter_name"], df["value"])
    )

    missing = [
        name for name in COUNTER_ORDER
        if name not in counter_lookup
    ]
    if missing:
        raise KeyError(
            f"{counter_file}: missing counters:\n" +
            "\n".join(f"  - {m}" for m in missing)
        )

    y_n = np.array(
        [counter_lookup[name] for name in COUNTER_ORDER],
        dtype=np.float64
    )

    if np.any(y_n < 0):
        raise ValueError(
            f"{counter_file}: negative counter values detected."
        )

    return y_n


# =============================================================
# Top-level loader — single run
# =============================================================
def load_counters(results_dir: str | Path) -> Tuple[np.ndarray, List[str]]:
    """
    Load all hardware counters from the two-level results directory.

    Auto-discovers node directories by finding any subdirectory
    containing counters.csv. All other files are ignored.

    Parameters
    ----------
    results_dir : str or Path
        Top-level results directory named by SLURM job ID.
        Example: "/pscratch/.../results/OMB_12345678"

    Returns
    -------
    Y : np.ndarray, shape (N, TWO_M) = (N, 30)
    node_names : list of str, length N
    """
    results_dir = Path(results_dir)

    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    if not results_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {results_dir}")

    print(f"Loading counters from: {results_dir}")

    node_dirs = sorted([
        d for d in results_dir.iterdir()
        if d.is_dir() and (d / "counters.csv").exists()
    ])

    if len(node_dirs) == 0:
        raise FileNotFoundError(
            f"No node directories with counters.csv found under:\n"
            f"  {results_dir}\n"
            f"Expected: {results_dir}/<node_name>/counters.csv"
        )

    N = len(node_dirs)
    Y = np.zeros((N, TWO_M), dtype=np.float64)
    node_names: List[str] = []

    for n, node_dir in enumerate(node_dirs):
        node_names.append(node_dir.name)
        Y[n, :] = load_node_counters(node_dir / "counters.csv")
        print(f"  Loaded [{n+1}/{N}] {node_dir.name}")

    _validate(Y, node_names)

    print(f"\nDone. Y shape: {Y.shape}  "
          f"(N={N} nodes, TWO_M={TWO_M} counters)")

    return Y, node_names


# =============================================================
# Multi-run loader
# =============================================================
def load_multiple_runs(
        run_dirs: List[str | Path]
) -> Tuple[np.ndarray, List[str]]:
    """
    Load counter data from multiple runs of the same workload.

    Parameters
    ----------
    run_dirs : list of str or Path
        SLURM job result directories, one per run.

    Returns
    -------
    Y_multi : np.ndarray, shape (N, K, TWO_M)
    node_names : list of str
    """
    K = len(run_dirs)
    all_Y: List[np.ndarray] = []
    ref_names: List[str] = []

    for k, run_dir in enumerate(run_dirs):
        Y_k, names_k = load_counters(Path(run_dir))

        if k == 0:
            ref_names = names_k
        elif names_k != ref_names:
            raise ValueError(
                f"Node mismatch between runs:\n"
                f"  Run 0  : {ref_names}\n"
                f"  Run {k} : {names_k}"
            )
        all_Y.append(Y_k)

    Y_multi = np.stack(all_Y, axis=1)   # (N, K, TWO_M)

    print(f"\nY_multi shape: {Y_multi.shape}  "
          f"(N={Y_multi.shape[0]} nodes, "
          f"K={Y_multi.shape[1]} runs, "
          f"TWO_M={Y_multi.shape[2]} counters)")

    return Y_multi, ref_names


# =============================================================
# Multi-run loader
# =============================================================
def load_multiple_runs(run_dirs: List[str | Path]) -> Tuple[np.ndarray, List[str]]:
    """
    Load counter data from multiple runs of the same workload.

    Parameters
    ----------
    run_dirs : list of str or Path
        SLURM job result directories, one per run.

    Returns
    -------
    Y_multi : np.ndarray, shape (N, K, TWO_M)
    node_names : list of str
    """
    K = len(run_dirs)
    all_Y: List[np.ndarray] = []
    ref_names: List[str] = []

    for k, run_dir in enumerate(run_dirs):
        Y_k, names_k = load_counters(Path(run_dir))

        if k == 0:
            ref_names = names_k
        elif names_k != ref_names:
            raise ValueError(
                f"Node mismatch between runs:\n"
                f"  Run 0  : {ref_names}\n"
                f"  Run {k} : {names_k}"
            )
        all_Y.append(Y_k)

    Y_multi = np.stack(all_Y, axis=1)   # (N, K, TWO_M)

    print(f"\nY_multi shape: {Y_multi.shape}  "
          f"(N={Y_multi.shape[0]} nodes, "
          f"K={Y_multi.shape[1]} runs, "
          f"TWO_M={Y_multi.shape[2]} counters)")

    return Y_multi, ref_names


# =============================================================
# Validation
# =============================================================
def _validate(Y: np.ndarray, node_names: List[str]) -> None:
    """Check loaded counter data for common issues."""
    N = Y.shape[0]
    print("\n  Validating...")

    if np.any(Y < 0):
        raise ValueError("Negative counter values detected.")
    print("    [OK] All counters non-negative")

    for n in range(N):
        if np.all(Y[n, :] == 0):
            print(f"    [WARN] {node_names[n]}: "
                  f"all counters zero — possible missing data")

    total_tx = np.sum(Y[:, TX_HIST_SLICE])
    total_rx = np.sum(Y[:, RX_HIST_SLICE])
    if total_tx > 0 and total_rx > 0:
        ratio  = total_tx / total_rx
        status = "[OK]" if 0.5 <= ratio <= 2.0 else "[WARN]"
        print(f"    {status} TX/RX packet ratio = {ratio:.2f}")

    print("    Validation complete.")