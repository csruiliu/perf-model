import argparse
from abc import ABC, abstractmethod

import pandas as pd

from counter_model.dcgm.gpu_metrics import MetricValues
from counter_model.dcgm.gpu_time import TimeSlicer
from counter_model.dcgm.scaler import GpuScaler, HostScaler, get_tf_weights
from counter_model.dcgm.utils import ResultsFormatter
from counter_model.hw_config.hw_specs import GPU, Host


class BaseEstimator(ABC):
    """Abstract base class for profilers"""

    def __init__(self, sample_interval_ms: float, ref_gpu: GPU):
        self.time_slicer = TimeSlicer(sample_interval_ms, ref_gpu)
        self.formatter = ResultsFormatter()

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

    def run(self, dcgm_df: pd.DataFrame, args: argparse.Namespace):
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
        windowed_metrics = time_window.extract_from_dict(target_metrics)

        # Print predictions
        self.formatter.print_target_results(windowed_metrics, self.tgt_gpu.get_name())

    def _scale_metrics(
        self, dcgm_df: pd.DataFrame, metrics: list[str], cores_alloc: str
    ) -> dict[str, list[float]]:
        """Calculate metrics for target hardware"""
        time_results = ["t_kernel", "t_total"]

        results = {f"{metric}_{key}": [] for metric in time_results for key in self.SMOCC_LEVELS}

        results["t_host"] = []
        results["t_pcie"] = []

        gpu_scaler = GpuScaler(self.ref_gpu, self.tgt_gpu)
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

            # Calculate time fraction on ref gpu
            time_frac_ref = self.time_slicer.time_fraction_single_gpu(mv)

            # Update SMOCC and calculate all scales
            gpu_scaler.update_smocc(mv_gract_norm["smocc_gract"])
            kernel_metrics_tgt = self._scale_kernel_metrics(gpu_scaler, mv_gract_norm, tf_weights)

            # PCIe Time
            t_pcie_tgt = time_frac_ref.t_pcie / gpu_scaler.pcie_scale()
            results["t_pcie"].append(t_pcie_tgt)

            # Other node time
            t_host_tgt = time_frac_ref.t_host / host_scaler.host_scale(cores_alloc)
            results["t_host"].append(t_host_tgt)

            # Process each SMOCC key
            for i, key in enumerate(self.SMOCC_LEVELS):
                # Calculate kernel scale (minimum of all constraints)
                kernel_scale = min(
                    x
                    for x in [
                        gpu_scaler.scale_smocc[key],
                        kernel_metrics_tgt["dram"][i],
                        kernel_metrics_tgt["tensor"][i],
                        kernel_metrics_tgt["fp64"][i],
                        kernel_metrics_tgt["fp32"][i],
                        kernel_metrics_tgt["fp16"][i],
                    ]
                    if x != 0
                )

                # Calculate kernel and total time
                t_kernel_tgt = time_frac_ref.t_kernel / kernel_scale
                results[f"t_kernel_{key}"].append(t_kernel_tgt)
                results[f"t_total_{key}"].append(t_kernel_tgt + t_pcie_tgt + t_host_tgt)

        return results

    def _scale_kernel_metrics(
        self, gpu_scaler: GpuScaler, mv_gract_norm: dict, tf_weights: dict
    ) -> dict[str, tuple]:
        """Calculate all scale factors in one place"""
        # scale_calc.smocc_scale() need to be invoked first
        return {
            "dram": gpu_scaler.dram_scale(mv_gract_norm["drama_gract"]),
            "tensor": gpu_scaler.tensor_scale_weighted(mv_gract_norm["tenso_gract"], tf_weights),
            "fp64": gpu_scaler.fp64_scale(mv_gract_norm["fp64a_gract"]),
            "fp32": gpu_scaler.fp32_scale(mv_gract_norm["fp32a_gract"]),
            "fp16": gpu_scaler.fp16_scale(mv_gract_norm["fp16a_gract"]),
        }
