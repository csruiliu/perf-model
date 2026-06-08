"""
time_estimator.py
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np

from counter_model.hw_config.hw_specs import Host
from counter_model.hw_config.pm_config import LATENCY_TABLES


def parse_osu_benchmark(filepath: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Parses standard output from OSU Micro-Benchmarks.
    Works for both osu_latency (Size vs Latency) and osu_bw/osu_bibw (Size vs MB/s).
    """
    sizes, values = [], []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    sizes.append(float(parts[0]))
                    values.append(float(parts[1]))
                except ValueError:
                    continue
    return np.array(sizes, dtype=np.float64), np.array(values, dtype=np.float64)


def _build_interp_model(
    sizes: np.ndarray, latencies: np.ndarray, source_label: str
) -> Callable[[np.ndarray], np.ndarray]:
    """Builds a linear-interpolation latency lookup from size/latency arrays."""
    if len(sizes) == 0:
        raise ValueError(f"No latency data available from {source_label}")

    sort_idx = np.argsort(sizes)
    sorted_sizes = sizes[sort_idx]
    sorted_latencies = latencies[sort_idx]

    print(f"  [Latency Lookup] Loaded {len(sorted_sizes)} points from {source_label}")

    def predict_latency(msg_sizes_array: np.ndarray) -> np.ndarray:
        return np.interp(msg_sizes_array, sorted_sizes, sorted_latencies)

    return predict_latency


def build_latency_model_from_file(osu_filepath: str | Path) -> Callable[[np.ndarray], np.ndarray]:
    """Creates a direct lookup model for latency using linear interpolation (file-based)."""
    sizes, latencies = parse_osu_benchmark(osu_filepath)
    return _build_interp_model(sizes, latencies, Path(osu_filepath).name)


def build_latency_model_from_config(table_name: str) -> Callable[[np.ndarray], np.ndarray]:
    """Creates a latency lookup model from a named table in pm_config.py."""
    if table_name not in LATENCY_TABLES:
        raise KeyError(
            f"No latency table named '{table_name}'. Available: {list(LATENCY_TABLES.keys())}"
        )

    table = LATENCY_TABLES[table_name]
    if not table:
        raise ValueError(f"Latency table '{table_name}' is empty in pm_config.py")

    sizes = np.array(list(table.keys()), dtype=np.float64)
    latencies = np.array(list(table.values()), dtype=np.float64)

    return _build_interp_model(sizes, latencies, f"pm_config[{table_name}]")


def fit_gap_model(
    osu_bw_filepath: str | Path, rdzv_threshold: int = 16384
) -> Callable[[np.ndarray], np.ndarray]:
    """Fits a piecewise linear model to the message Gap derived from bandwidth data."""
    sizes, bandwidths = parse_osu_benchmark(osu_bw_filepath)
    if len(sizes) == 0:
        raise ValueError(f"Could not parse bandwidth data from {osu_bw_filepath}")

    # Convert bandwidth (MB/s) to Gap (us) -> Gap = size(Bytes) / BW(MB/s)
    safe_bw = np.where(bandwidths == 0, 1e-9, bandwidths)
    gaps_us = sizes / safe_bw

    eager_mask = sizes <= rdzv_threshold
    rdzv_mask = sizes > rdzv_threshold

    sizes_eager, gaps_eager = sizes[eager_mask], gaps_us[eager_mask]
    sizes_rdzv, gaps_rdzv = sizes[rdzv_mask], gaps_us[rdzv_mask]

    if len(sizes_eager) > 1:
        beta_eager, alpha_eager = np.polyfit(sizes_eager, gaps_eager, 1)
    else:
        beta_eager, alpha_eager = 0.0, np.mean(gaps_eager) if len(gaps_eager) else 0.0

    if len(sizes_rdzv) > 1:
        beta_rdzv, alpha_rdzv = np.polyfit(sizes_rdzv, gaps_rdzv, 1)
    else:
        beta_rdzv, alpha_rdzv = 0.0, np.mean(gaps_rdzv) if len(gaps_rdzv) else 0.0

    print(f"  [Gap Fit - Eager]      alpha: {alpha_eager:>6.3f} us,  beta: {beta_eager:.6f} us/B")
    print(f"  [Gap Fit - Rendezvous] alpha: {alpha_rdzv:>6.3f} us,  beta: {beta_rdzv:.6f} us/B")

    def predict_gap(msg_sizes_array: np.ndarray) -> np.ndarray:
        predicted = np.zeros_like(msg_sizes_array, dtype=np.float64)
        eager_idx = msg_sizes_array <= rdzv_threshold
        rdzv_idx = msg_sizes_array > rdzv_threshold

        predicted[eager_idx] = alpha_eager + beta_eager * msg_sizes_array[eager_idx]
        predicted[rdzv_idx] = alpha_rdzv + beta_rdzv * msg_sizes_array[rdzv_idx]
        return np.maximum(predicted, 0.0)

    return predict_gap


def create_direct_gap_model(osu_bw_filepath: str | Path) -> Callable[[np.ndarray], np.ndarray]:
    """Creates a model that returns the raw interpolated gap based on bandwidth benchmarks."""
    sizes, bandwidths = parse_osu_benchmark(osu_bw_filepath)
    if len(sizes) == 0:
        raise ValueError(f"Could not parse bandwidth data from {osu_bw_filepath}")

    sort_idx = np.argsort(sizes)
    sorted_sizes = sizes[sort_idx]
    sorted_bw = bandwidths[sort_idx]

    safe_bw = np.where(sorted_bw == 0, 1e-9, sorted_bw)
    gaps_us = sorted_sizes / safe_bw

    print(f"  [Gap Lookup] Loaded {len(sorted_sizes)} points from {Path(osu_bw_filepath).name}")

    def predict_gap(msg_sizes_array: np.ndarray) -> np.ndarray:
        return np.interp(msg_sizes_array, sorted_sizes, gaps_us)

    return predict_gap


def time_estimation(
    x_send: np.ndarray,
    x_recv: np.ndarray,
    msg_sizes: np.ndarray,
    latency_model: Callable[[np.ndarray], np.ndarray],
    ref_host_name: str,
) -> tuple[float, float]:
    """
    Estimate communication time bounds for a node-level message pattern.

    Returns two values:
      - overlap_time:    Theoretical *lower bound* on time, assuming all messages
                         are launched simultaneously with perfect overlap.
      - sequential_time: Reference time assuming no overlap (every message's
                         latency summed), i.e. the fully-serialized case.

    Model assumptions:
      - Network is a FULL-BISECTION FAT TREE
      - Links are FULL-DUPLEX: send and receive flow concurrently and do not
        compete, so the bandwidth floor uses max(send, recv).
      - This is a lower bound only, it does NOT capture endpoint contention
        (e.g. incast / many-to-one).
    """

    # Per-message wire latency for each message-size bin.
    bin_latencies = latency_model(msg_sizes)

    # Ensure 2D so rows = nodes, columns = size bins.
    x_send_2d = np.atleast_2d(x_send)
    x_recv_2d = np.atleast_2d(x_recv)

    num_nodes = x_send_2d.shape[0]
    node_sequential_time = np.zeros(num_nodes)
    node_parallelism_time = np.zeros(num_nodes)
    node_parallelism_max_latency = np.zeros(num_nodes)
    node_parallelism_bandwidth_floor = np.zeros(num_nodes)

    ref_host = Host(host_name=ref_host_name)

    # we assume the network is full-duplex
    network_bw = num_nodes * ref_host.get_specs("network_bw")

    for nidx in range(num_nodes):
        # --- Latency floor ---
        # Under perfect overlap you still must wait for the slowest single message.
        # Take the worse of send/recv per bin, then the max over bins.
        node_parallelism_max_latency[nidx] = np.max(
            np.maximum(x_send_2d[nidx, :] * bin_latencies, x_recv_2d[nidx, :] * bin_latencies)
        )

        # --- Bandwidth floor ---
        # Time to push the heavier direction's bytes through the link at full
        # rate. max(send, recv) because full-duplex links don't contend.
        send_bytes = np.sum(x_send_2d[nidx, :] * msg_sizes)
        recv_bytes = np.sum(x_recv_2d[nidx, :] * msg_sizes)
        node_parallelism_bandwidth_floor[nidx] = max(send_bytes, recv_bytes) / network_bw

        node_parallelism_time[nidx] = max(
            node_parallelism_max_latency[nidx], node_parallelism_bandwidth_floor[nidx]
        )

        # No-overlap reference: sum the latency of every message (both dirs).
        node_sequential_time[nidx] = np.sum(
            (x_send_2d[nidx, :] + x_recv_2d[nidx, :]) * bin_latencies
        )

    overlap_time = np.max(node_parallelism_time)
    sequential_time = np.sum(node_sequential_time)

    return overlap_time, sequential_time
