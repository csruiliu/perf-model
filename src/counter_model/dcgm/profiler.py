import argparse
from abc import ABC, abstractmethod

import pandas as pd

from counter_model.dcgm.gpu_metrics import MetricValues
from counter_model.dcgm.gpu_time import TimeSlicer
from counter_model.dcgm.scaler import get_tf_weights
from counter_model.dcgm.utils import print_reference_results
from counter_model.hw_config.hw_specs import GPU


class BaseProfiler(ABC):
    """Abstract base class for profilers"""

    def __init__(self, sample_interval_ms: float, gpu_name: str):
        self.gpu = GPU(gpu_name=gpu_name)
        self.time_slicer = TimeSlicer(sample_interval_ms, self.gpu)

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

        results = {}
        results["t_kernel"] = []
        results["t_pcie"] = []
        results["t_host"] = []

        for row in profiled_df.itertuples(index=False):
            mv = MetricValues.from_row(row, args.metrics)
            mv_gract_norm = mv.gract_normalization()

            # Calculate weights for this row
            tf_weights = get_tf_weights(
                mv_gract_norm["fp64a_gract"],
                mv_gract_norm["fp32a_gract"],
                mv_gract_norm["fp16a_gract"],
            )

            tensor_flop = mv_gract_norm["tenso_gract"] * (
                tf_weights["tf64"] * self.gpu.get_specs("tf64")
                + tf_weights["tf32"] * self.gpu.get_specs("tf32")
                + tf_weights["tf16"] * self.gpu.get_specs("tf16")
            )

            regular_flop = sum(
                mv_gract_norm[f"{p}a_gract"] * self.gpu.get_specs(p)
                for p in ("fp64", "fp32", "fp16")
            )

            flop_sum += tensor_flop + regular_flop
            dram_sum += mv_gract_norm["drama_gract"] * self.gpu.get_specs("mem_bw")

            # Calculate time fraction on ref gpu
            time_frac_ref = self.time_slicer.time_fraction_single_gpu(mv)
            results["t_kernel"].append(time_frac_ref.t_kernel)
            results["t_pcie"].append(time_frac_ref.t_pcie)
            results["t_host"].append(time_frac_ref.t_host)

        time_window = self.time_slicer.get_time_window(
            args.overall_runtime_ms,
            args.start_timestamp,
            args.end_timestamp,
            len(results["t_host"]),
        )

        ws = time_window.extract_from_dict(results)
        flops = flop_sum / len(profiled_df)
        membw = dram_sum / len(profiled_df)

        if is_printout:
            print_reference_results(ws, flops, membw, self.gpu.get_name())

        return float(sum(ws["t_kernel"]) + sum(ws["t_pcie"]) + sum(ws["t_host"]))
