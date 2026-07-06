import argparse
from abc import ABC, abstractmethod

import pandas as pd

from counter_model.dcgm.gpu_metrics import MetricValues
from counter_model.dcgm.gpu_time import TimeFraction, TimeSlicer
from counter_model.dcgm.scaler import get_tf_weights
from counter_model.dcgm.utils import ResultsFormatter
from counter_model.hw_config.hw_specs import GPU


class BaseProfiler(ABC):
    """Abstract base class for profilers"""

    def __init__(self, sample_interval_ms: float, gpu_name: str):
        self.gpu = GPU(gpu_name=gpu_name)
        self.time_slicer = TimeSlicer(sample_interval_ms, self.gpu)
        self.formatter = ResultsFormatter()

    @abstractmethod
    def run(self, *args, **kwargs):
        """Run the profiling/prediction"""
        pass

    @staticmethod
    def _get_time_fraction(df: pd.DataFrame, metrics: list[str], calc_fn) -> list[TimeFraction]:
        """Calculate time fraction for all rows using the given calculator."""
        return [calc_fn(MetricValues.from_row(row, metrics)) for row in df.itertuples(index=False)]


class SingleGpuProfiler(BaseProfiler):
    """Profiles performance on reference hardware"""

    def run(self, profiled_df: pd.DataFrame, args: argparse.Namespace):
        """Model performance on reference hardware"""
        time_frac_ref = self._get_time_fraction(
            profiled_df, args.metrics, self.time_slicer.time_fraction_single_gpu
        )

        time_window = self.time_slicer.get_time_window(
            args.overall_runtime_ms, args.start_timestamp, args.end_timestamp, len(time_frac_ref)
        )

        windowed_df = time_window.extract_from_dataframe(profiled_df)
        flops = self._calc_flops(windowed_df, args.smetrics)
        membw = self._calc_membw(windowed_df, args.metrics)

        self.formatter.print_reference_results(windowed_df, flops, membw, self.gpu.get_name())

    def _estimate_flops(self, profiled_df: pd.DataFrame, metrics: list[str]) -> float:
        """Calculate FLOPS"""
        flop_sum = 0.0

        for row in profiled_df.itertuples(index=False):
            mv = MetricValues.from_row(row, metrics)
            mv_gract_norm = mv.gract_normalization()
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

        return flop_sum / len(profiled_df)

    def _estimate_membw(self, profiled_df: pd.DataFrame, metrics: list[str]) -> float:
        """Calculate memory bandwidth"""
        dram_sum = 0.0
        for row in profiled_df.itertuples(index=False):
            mv = MetricValues.from_row(row, metrics)
            mv_gract_norm = mv.gract_normalization()
            dram_sum += mv_gract_norm["drama_gract"] * self.gpu.get_specs("mem_bw")

        return dram_sum / len(profiled_df)


class MultiGpuProfiler(BaseProfiler):
    """Profiles performance on reference hardware for multiple GPUs"""

    def __init__(self, sample_interval_ms: float, gpu_name: str):
        self.gpu = GPU(gpu_name=gpu_name)
        super().__init__(sample_interval_ms, self.gpu)

    def run(
        self,
        gpu_dfs: list[pd.DataFrame],
        metrics: list[str],
        overall_runtime_ms: float,
        agg_interval_ms: float,
        start_ts: float | None,
        end_ts: float | None,
        tensor_prec: str,
    ):
        pass
