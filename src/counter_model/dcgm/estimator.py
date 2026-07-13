import argparse
from abc import ABC, abstractmethod

import pandas as pd
from scipy.stats import trim_mean

from counter_model.dcgm.gpu_metrics import MetricValues
from counter_model.dcgm.gpu_time import TimeSlicer
from counter_model.dcgm.scaler import GpuScaler, HostScaler, get_tf_weights
from counter_model.dcgm.utils import print_target_results
from counter_model.hw_config.hw_specs import GPU, Host


class BaseEstimator(ABC):
    """Abstract base class for profilers"""

    def __init__(self, sample_interval_ms: float, ref_gpu: GPU):
        self.time_slicer = TimeSlicer(sample_interval_ms, ref_gpu)

    @abstractmethod
    def run(self, *args, **kwargs):
        """Run the profiling/prediction"""
        pass


class SingleGpuEstimator(BaseEstimator):
    """Estimating performance on target GPU"""

    # Class-level constants
    SMOCC_LEVELS = ["lower", "mid", "upper", "mock"]

    def __init__(self, args: argparse.Namespace):
        self.ref_gpu = GPU(gpu_name=args.ref_gpu)
        self.tgt_gpu = GPU(gpu_name=args.tgt_gpu)
        self.ref_host = Host(host_name=args.ref_host)
        self.tgt_host = Host(host_name=args.tgt_host)
        super().__init__(args.sample_interval_ms, self.ref_gpu)

    def run(self, dcgm_df: pd.DataFrame, args: argparse.Namespace, is_printout: bool):
        """Predict performance on target hardware"""
        # Calculate target metrics
        target_metrics = self._scale_metrics(dcgm_df, args.metrics, args.cores_alloc)

        # Get time slice
        time_window = self.time_slicer.get_time_window(
            args.overall_runtime_ms,
            args.start_timestamp,
            args.end_timestamp,
            len(target_metrics["t_total_lower"]),
        )

        # Slice metrics
        ws = time_window.extract_from_dict(target_metrics)

        # Calculate estimated FLOPS and memory bandwidth
        est_flops = self._estimate_peak_rate(ws, "flops")
        est_membw = self._estimate_peak_rate(ws, "dram")

        # Print predictions
        if is_printout:
            print_target_results(ws, est_flops, est_membw, self.tgt_gpu.get_name())

        return {level: float(sum(ws[f"t_total_{level}"])) for level in self.SMOCC_LEVELS}

    def _scale_metrics(
        self, dcgm_df: pd.DataFrame, metrics: list[str], cores_alloc: str
    ) -> dict[str, list[float]]:
        """Calculate metrics for target hardware"""
        time_results = ["t_kernel", "t_total", "dram", "flops"]

        results = {f"{metric}_{key}": [] for metric in time_results for key in self.SMOCC_LEVELS}

        # host and pcie time are not scaled by smocc
        results["t_host"] = []
        results["t_pcie"] = []

        gpu_scaler = GpuScaler(self.ref_gpu, self.tgt_gpu, self.SMOCC_LEVELS)
        host_scaler = HostScaler(self.ref_host, self.tgt_host)

        for row in dcgm_df.itertuples(index=False):
            mv = MetricValues.from_row(row, metrics)
            mv_gract_norm = mv.gract_normalization()

            # Calculate weights for this row
            tf_weights = get_tf_weights(
                mv_gract_norm["fp64a_gract"],
                mv_gract_norm["fp32a_gract"],
                mv_gract_norm["fp16a_gract"],
            )

            tf_precisions = ("tf64", "tf32", "tf16")
            tf_tgt = sum(tf_weights[p] * self.tgt_gpu.get_specs(p) for p in tf_precisions)
            tf_ref = sum(tf_weights[p] * self.ref_gpu.get_specs(p) for p in tf_precisions)

            # Calculate time fraction on ref gpu
            time_frac_ref = self.time_slicer.time_fraction_single_gpu(mv)

            # Update SMOCC and calculate all scales
            gpu_scaler.update_smocc(mv_gract_norm["smocc_gract"])
            gpu_scaler.update_scale_kernel(mv_gract_norm, tf_weights)

            # PCIe Time
            t_pcie_tgt = time_frac_ref.t_pcie / gpu_scaler.pcie_scale()
            results["t_pcie"].append(t_pcie_tgt)

            # Other node time
            t_host_tgt = time_frac_ref.t_host / host_scaler.host_scale(cores_alloc)
            results["t_host"].append(t_host_tgt)

            # Process each SMOCC key
            for key in self.SMOCC_LEVELS:
                # Calculate kernel and total time
                t_kernel_tgt = time_frac_ref.t_kernel / gpu_scaler.scale_kernel.get(key)
                results[f"t_kernel_{key}"].append(t_kernel_tgt)
                results[f"t_total_{key}"].append(t_kernel_tgt + t_pcie_tgt + t_host_tgt)
                mem_bw_tgt = min(
                    self.ref_gpu.get_specs("mem_bw")
                    * mv_gract_norm["drama_gract"]
                    * gpu_scaler.scale_smocc[key],
                    self.tgt_gpu.get_specs("mem_bw"),
                )
                results[f"dram_{key}"].append(mem_bw_tgt)

                flops_tgt = min(
                    self.ref_gpu.get_specs("tf64")
                    * mv_gract_norm["tenso_gract"]
                    * gpu_scaler.scale_smocc[key],
                    self.tgt_gpu.get_specs("tf64"),
                )
                results[f"flops_{key}"].append(flops_tgt)

        return results

    def _estimate_peak_rate(self, metrics: dict[str, list[float]], prefix: str) -> dict[str, float]:
        """Generic method to calculate aggregated metrics (FLOPS or memory bandwidth)"""
        return {
            f"{prefix}_{key}": float(trim_mean(metrics[f"{prefix}_{key}"], 0.10))
            for key in self.SMOCC_LEVELS
        }
