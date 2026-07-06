from dataclasses import dataclass

import pandas as pd

from counter_model.dcgm.gpu_metrics import MetricValues
from counter_model.hw_config.hw_specs import GPU


@dataclass
class TimeFraction:
    """Container for calculated time components"""

    t_kernel: float = 0.0
    t_pcie: float = 0.0
    t_host: float = 0.0
    t_nvlink: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary"""
        return {
            "t_kernel": self.t_kernel,
            "t_pcie": self.t_pcie,
            "t_host": self.t_host,
            "t_nvlink": self.t_nvlink,
        }


@dataclass
class TimeWindow:
    """Container for time-based metrics with slicing functionality"""

    start_idx: int = 0
    end_idx: int | None = None

    def extract_from_list(self, data: list) -> list:
        """Apply time window to a list"""
        return data[self.start_idx : self.end_idx]

    def extract_from_dict(self, data: dict[str, list]) -> dict[str, list]:
        """Apply time window to all lists in a dictionary"""
        return {key: values[self.start_idx : self.end_idx] for key, values in data.items()}

    def extract_from_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply time window to a list"""
        return data[self.start_idx : self.end_idx]


class TimeSlicer:
    """Handles time-related calculations"""

    def __init__(self, sample_interval_ms: float, ref_gpu: GPU):
        self.sample_intv = sample_interval_ms / 1000
        self.gpu = ref_gpu

    def time_fraction_single_gpu(self, metrics: MetricValues) -> TimeFraction:
        """Calculate time fraction from metrics for single gpu"""
        t_kernel = self.sample_intv * metrics.gract
        t_pcie = (
            self.sample_intv
            * (metrics.pcitx + metrics.pcirx)
            / (self.gpu.get_specs("pcie_bw") * 1e9)
        )
        t_host = max(self.sample_intv - t_kernel - t_pcie, 0)

        return TimeFraction(t_kernel, t_pcie, t_host, t_nvlink=0)

    def time_fraction_multi_gpu(self, metrics: MetricValues) -> TimeFraction:
        """Calculate time fraction from metrics for multi-gpu"""
        pass

    def get_time_window(
        self,
        overall_runtime_ms: float,
        start_ts: float | None,
        end_ts: float | None,
        data_length: int,
    ) -> TimeWindow:
        """Calculate time window indices"""
        finish_idx = min(int(overall_runtime_ms / (self.sample_intv * 1000)), data_length)
        start_idx = int((start_ts or 0) / (self.sample_intv * 1000))

        if end_ts is not None:
            end_idx = min(finish_idx, int(end_ts / (self.sample_intv * 1000)))
            if start_idx > end_idx:
                raise ValueError("End timestamp is earlier than start timestamp")
        else:
            end_idx = finish_idx

        return TimeWindow(start_idx=start_idx, end_idx=end_idx)
