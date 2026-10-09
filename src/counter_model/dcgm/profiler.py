import argparse
from abc import ABC, abstractmethod

import pandas as pd

from counter_model.dcgm.constants import TF_TO_FP
from counter_model.dcgm.data_classes import MetricValues
from counter_model.dcgm.scaler import effective_tensor_peak, get_tf_weights
from counter_model.dcgm.time_aggregator import TimeAggregator
from counter_model.hw_config.hw_specs import GPU


class BaseProfiler(ABC):
    """Abstract base class for profilers"""

    def __init__(self, sample_interval_ms: float, gpu_name: str):
        self.gpu = GPU(gpu_name=gpu_name)
        self.time_aggregator = TimeAggregator(sample_interval_ms, self.gpu)

    @abstractmethod
    def run(self, *args, **kwargs):
        """Run the profiling/prediction"""
        pass


class SingleGpuProfiler(BaseProfiler):
    """Profiles performance on reference hardware"""

    def run(self, profiled_df: pd.DataFrame, args: argparse.Namespace, is_printout: bool) -> float:
        """Model performance on reference hardware"""
        flop_sum = 0.0
        dram_sum = 0.0

        results = {
            "t_kernel": [],
            "t_pcie": [],
            "t_kernel_pcie": [],
            "t_residual": [],
            "flops": [],
            "dram": [],
        }

        for row in profiled_df.itertuples(index=False):
            mv = MetricValues.from_row(row)
            mv_gract_norm = mv.gract_normalization()

            # Calculate weights for this row
            # Eq. (10b): FLOP-share weights and harmonic-mean tensor peak on the reference
            tf_weights = get_tf_weights(
                mv_gract_norm["fp64a_gract"],
                mv_gract_norm["fp32a_gract"],
                mv_gract_norm["fp16a_gract"],
                self.gpu,
            )
            tf_ref = effective_tensor_peak(tf_weights, self.gpu)  # β_tensor^ref

            # Achieved tensor rate: A_tensor^ref · β_tensor^ref (denominator in Eq. 10a)
            tensor_flop = mv_gract_norm["tenso_gract"] * tf_ref

            # Achieved non-tensor rate: Σ_p A_fpp^ref · β_fpp^ref
            regular_flop = sum(
                mv_gract_norm[f"{p}a_gract"] * self.gpu.get_specs(p) for p in list(TF_TO_FP.values())
            )

            results["flops"].append(tensor_flop + regular_flop)
            results["dram"].append(mv_gract_norm["drama_gract"] * self.gpu.get_specs("mem_bw"))

            # Time fractions on the reference GPU
            time_frac_ref = self.time_aggregator.time_fraction_single_gpu_ref(mv)
            results["t_kernel"].append(time_frac_ref.t_kernel)
            results["t_pcie"].append(time_frac_ref.t_pcie)
            results["t_kernel_pcie"].append(time_frac_ref.t_kernel_pcie)
            results["t_residual"].append(time_frac_ref.t_residual)

        time_window = self.time_aggregator.get_time_window(
            args.overall_runtime_ms,
            args.start_timestamp,
            args.end_timestamp,
            len(results["t_residual"]),
        )

        ws = time_window.extract_from_dict(results)
        flops = flop_sum / len(profiled_df)
        membw = dram_sum / len(profiled_df)

        if is_printout:
            self.print_reference_results(ws, flops, membw, self.gpu.get_name())

        return float(
            sum(ws["t_kernel"])
            + sum(ws["t_pcie"])
            - sum(ws["t_kernel_pcie"])
            + sum(ws["t_residual"])
        )

    def print_reference_results(
        self, est_component_sample: dict[str, list[float]], flops: float, mem_bw: float, gpu: str
    ):
        """Print reference hardware results, convert runtime(ms) to second"""
        """No need to have total time"""
        print(f"\n{'=' * 60}")
        print(f"Reference Hardware: {gpu}\n")
        print(f"Estimated TFLOPS: {flops:.2f}")
        print(f"Estimated GPU Memory Bandwidth: {mem_bw:.2f} GB/s")
        print(f"\nEstimated Kernel Time: {sum(est_component_sample['t_kernel']) / 1000:.2f} s")
        print(f"\nEstimated PCIe Time: {sum(est_component_sample['t_pcie']) / 1000:.2f} s")
        print(
            f"\nEstimated Kernel and PCIe Aggregation Time: {sum(est_component_sample['t_kernel_pcie']) / 1000:.2f} s"
        )
        print(f"\nEstimated Residual Time: {sum(est_component_sample['t_residual']) / 1000:.2f} s")
        print(f"{'=' * 60}\n")
