"""
time_estimator.py

Estimates the theoretical upper bound of MPI communication time using both 
a latency model (for sparse communication) and a gap/bandwidth model 
(for pipelined, back-to-back communication).
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Callable, Optional


def parse_osu_benchmark(filepath: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parses standard output from OSU Micro-Benchmarks.
    Works for both osu_latency (Size vs Latency) and osu_bw/osu_bibw (Size vs MB/s).
    """
    sizes, values = [], []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    sizes.append(float(parts[0]))
                    values.append(float(parts[1]))
                except ValueError:
                    continue
    return np.array(sizes, dtype=np.float64), np.array(values, dtype=np.float64)


def fit_latency_model(osu_filepath: str | Path, rdzv_threshold: int = 16384) -> Callable[[np.ndarray], np.ndarray]:
    """Fits a piecewise Hockney model to the OSU latency data."""
    sizes, latencies = parse_osu_benchmark(osu_filepath)
    if len(sizes) == 0:
        raise ValueError(f"Could not parse latency data from {osu_filepath}")

    eager_mask = sizes <= rdzv_threshold
    rdzv_mask = sizes > rdzv_threshold
    
    sizes_eager, lat_eager = sizes[eager_mask], latencies[eager_mask]
    sizes_rdzv, lat_rdzv = sizes[rdzv_mask], latencies[rdzv_mask]
    
    if len(sizes_eager) > 1:
        beta_eager, alpha_eager = np.polyfit(sizes_eager, lat_eager, 1)
    else:
        beta_eager, alpha_eager = 0.0, np.mean(lat_eager) if len(lat_eager) else 0.0
        
    if len(sizes_rdzv) > 1:
        beta_rdzv, alpha_rdzv = np.polyfit(sizes_rdzv, lat_rdzv, 1)
    else:
        beta_rdzv, alpha_rdzv = 0.0, np.mean(lat_rdzv) if len(lat_rdzv) else 0.0

    print("  [Latency Fit - Eager]      alpha: {:>6.3f} us,  beta: {:.6f} us/B".format(alpha_eager, beta_eager))
    print("  [Latency Fit - Rendezvous] alpha: {:>6.3f} us,  beta: {:.6f} us/B".format(alpha_rdzv, beta_rdzv))

    def predict_latency(msg_sizes_array: np.ndarray) -> np.ndarray:
        predicted = np.zeros_like(msg_sizes_array, dtype=np.float64)
        eager_idx = msg_sizes_array <= rdzv_threshold
        rdzv_idx = msg_sizes_array > rdzv_threshold
        
        predicted[eager_idx] = alpha_eager + beta_eager * msg_sizes_array[eager_idx]
        predicted[rdzv_idx] = alpha_rdzv + beta_rdzv * msg_sizes_array[rdzv_idx]
        return np.maximum(predicted, 0.0)
        
    return predict_latency


def create_direct_lookup_model(osu_filepath: str | Path) -> Callable[[np.ndarray], np.ndarray]:
    """Creates a direct lookup model for latency using linear interpolation."""
    sizes, latencies = parse_osu_benchmark(osu_filepath)
    if len(sizes) == 0:
        raise ValueError(f"Could not parse latency data from {osu_filepath}")

    sort_idx = np.argsort(sizes)
    sorted_sizes = sizes[sort_idx]
    sorted_latencies = latencies[sort_idx]

    print(f"  [Latency Lookup] Loaded {len(sorted_sizes)} points from {Path(osu_filepath).name}")

    def predict_latency(msg_sizes_array: np.ndarray) -> np.ndarray:
        return np.interp(msg_sizes_array, sorted_sizes, sorted_latencies)

    return predict_latency


def fit_gap_model(osu_bw_filepath: str | Path, rdzv_threshold: int = 16384) -> Callable[[np.ndarray], np.ndarray]:
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

    print("  [Gap Fit - Eager]      alpha: {:>6.3f} us,  beta: {:.6f} us/B".format(alpha_eager, beta_eager))
    print("  [Gap Fit - Rendezvous] alpha: {:>6.3f} us,  beta: {:.6f} us/B".format(alpha_rdzv, beta_rdzv))

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


def compute_upper_bound_time(x_send: np.ndarray, 
                             x_recv: np.ndarray, 
                             msg_sizes: np.ndarray, 
                             latency_model: Callable[[np.ndarray], np.ndarray],
                             gap_model: Optional[Callable[[np.ndarray], np.ndarray]] = None) -> Tuple[float, float, int]:
    """
    Computes theoretical communication time evaluating both sparse latency constraints
    and dense injection gap constraints.
    """
    bin_latencies = latency_model(msg_sizes)
    
    if gap_model is not None:
        bin_gaps = gap_model(msg_sizes)
    else:
        bin_gaps = np.zeros_like(bin_latencies)
    
    x_send_2d = np.atleast_2d(x_send)
    x_recv_2d = np.atleast_2d(x_recv)
    
    num_nodes = x_send_2d.shape[0]
    node_times = np.zeros(num_nodes)
    
    for n in range(num_nodes):
        latency_bound = np.sum((x_send_2d[n, :] + x_recv_2d[n, :]) * bin_latencies)
        gap_bound = np.sum((x_send_2d[n, :] + x_recv_2d[n, :]) * bin_gaps)
        node_times[n] = max(latency_bound, gap_bound)
        
    overlap_time = np.max(node_times)
    sequential_time = np.sum(node_times)
    heaviest_node = int(np.argmax(node_times))
    
    return overlap_time, sequential_time, heaviest_node