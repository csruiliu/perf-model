"""
time_estimator.py

Estimates the theoretical upper bound of MPI communication time using a
piecewise Hockney model (T = alpha + beta * size) fitted to OSU latency data.

Usage:
    # First, run standard osu_latency and save the output:
    # srun -n 2 ./osu_latency > osu_latency.txt
    
    # Then in your main pipeline:
    from time_estimator import fit_latency_model, compute_upper_bound_time
    
    latency_model = fit_latency_model("osu_latency.txt", rdzv_threshold=16384)
    total_time_us = compute_upper_bound_time(x_send, MSG_SIZES, latency_model)
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Callable, Dict

def parse_osu_latency(filepath: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """Parses standard output from the OSU Micro-Benchmark (osu_latency)."""
    sizes, latencies = [], []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    sizes.append(float(parts[0]))
                    latencies.append(float(parts[1]))
                except ValueError:
                    continue
    return np.array(sizes, dtype=np.float64), np.array(latencies, dtype=np.float64)


def fit_latency_model(osu_filepath: str | Path, rdzv_threshold: int = 16384) -> Callable[[np.ndarray], np.ndarray]:
    """
    Fits a piecewise Hockney model to the OSU latency data.
    
    Splits the data at the rendezvous threshold to account for the RTS/CTS 
    handshake overhead. Fits T(s) = alpha + beta * s for both regimes.
    
    Returns a vectorized function that takes an array of message sizes 
    and returns an array of estimated latencies in microseconds.
    """
    sizes, latencies = parse_osu_latency(osu_filepath)
    if len(sizes) == 0:
        raise ValueError(f"Could not parse latency data from {osu_filepath}")

    # Split data into Eager and Rendezvous regimes
    eager_mask = sizes <= rdzv_threshold
    rdzv_mask = sizes > rdzv_threshold
    
    sizes_eager = sizes[eager_mask]
    lat_eager = latencies[eager_mask]
    
    sizes_rdzv = sizes[rdzv_mask]
    lat_rdzv = latencies[rdzv_mask]
    
    # Fit Eager model (np.polyfit returns [slope, intercept])
    if len(sizes_eager) > 1:
        beta_eager, alpha_eager = np.polyfit(sizes_eager, lat_eager, 1)
    else:
        beta_eager, alpha_eager = 0.0, np.mean(lat_eager) if len(lat_eager) else 0.0
        
    # Fit Rendezvous model
    if len(sizes_rdzv) > 1:
        beta_rdzv, alpha_rdzv = np.polyfit(sizes_rdzv, lat_rdzv, 1)
    else:
        beta_rdzv, alpha_rdzv = 0.0, np.mean(lat_rdzv) if len(lat_rdzv) else 0.0

    print("  [Eager Protocol]      alpha: {:>6.3f} us,  beta: {:.6f} us/B".format(alpha_eager, beta_eager))
    print("  [Rendezvous Protocol] alpha: {:>6.3f} us,  beta: {:.6f} us/B".format(alpha_rdzv, beta_rdzv))

    def predict_latency(msg_sizes_array: np.ndarray) -> np.ndarray:
        predicted = np.zeros_like(msg_sizes_array, dtype=np.float64)
        eager_idx = msg_sizes_array <= rdzv_threshold
        rdzv_idx = msg_sizes_array > rdzv_threshold
        
        predicted[eager_idx] = alpha_eager + beta_eager * msg_sizes_array[eager_idx]
        predicted[rdzv_idx] = alpha_rdzv + beta_rdzv * msg_sizes_array[rdzv_idx]
        return np.maximum(predicted, 0.0)
        
    return predict_latency


def create_direct_lookup_model(osu_filepath: str | Path) -> Callable[[np.ndarray], np.ndarray]:
    """
    Creates a model that directly returns the raw benchmark latencies.
    Uses linear interpolation to perfectly match exact sizes and safely
    estimate any missing intermediate sizes.
    """
    sizes, latencies = parse_osu_latency(osu_filepath)
    if len(sizes) == 0:
        raise ValueError(f"Could not parse latency data from {osu_filepath}")

    # Ensure the arrays are sorted by size, which is required for np.interp
    sort_idx = np.argsort(sizes)
    sorted_sizes = sizes[sort_idx]
    sorted_latencies = latencies[sort_idx]

    print(f"  [Direct Lookup] Loaded {len(sorted_sizes)} raw data points from {Path(osu_filepath).name}")

    def predict_latency(msg_sizes_array: np.ndarray) -> np.ndarray:
        # np.interp returns exact latencies for exact size matches
        return np.interp(msg_sizes_array, sorted_sizes, sorted_latencies)

    return predict_latency



def compute_upper_bound_time(x_send: np.ndarray, 
                             x_recv: np.ndarray, 
                             msg_sizes: np.ndarray, 
                             latency_model: Callable) -> Tuple[float, float, int]:
    """
    Computes theoretical communication time for both overlap and sequential modes.
    
    Returns:
        overlap_time (float): The max time spent by a single node.
        sequential_time (float): The total time spent by all nodes combined.
        heaviest_node (int): The index of the node with the maximum time.
    """
    bin_latencies = latency_model(msg_sizes)
    
    if x_send.ndim == 1:
        # For a single node, overlap and sequential times are identical
        total_time = np.sum((x_send + x_recv) * bin_latencies)
        return total_time, total_time, 0
    else:
        # Multi-node calculation
        node_times = np.zeros(x_send.shape[0])
        for n in range(x_send.shape[0]):
            node_times[n] = np.sum((x_send[n, :] + x_recv[n, :]) * bin_latencies)
            
        overlap_time = np.max(node_times)
        sequential_time = np.sum(node_times)
        heaviest_node = int(np.argmax(node_times))
        
        return overlap_time, sequential_time, heaviest_node