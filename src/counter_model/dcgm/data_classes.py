from dataclasses import dataclass

import pandas as pd
from constants import DCGM_COUNTERS_MAP


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


@dataclass
class MetricValues:
    """Data class for extracted metrics from a row"""

    gract: float = 0.0
    drama: float = 0.0
    tenso: float = 0.0
    fp64a: float = 0.0
    fp32a: float = 0.0
    fp16a: float = 0.0
    smocc: float = 0.0
    pcitx: float = 0.0
    pcirx: float = 0.0
    nvltx: float = 0.0
    nvlrx: float = 0.0

    @classmethod
    def from_row(cls, row) -> "MetricValues":
        """Create MetricValues from a dataframe row"""
        metric_set = set(DCGM_COUNTERS_MAP.keys())

        return cls(
            **{
                field: getattr(row, col, 0.0) if col in metric_set else 0.0
                for col, field in DCGM_COUNTERS_MAP.items()
            }
        )

    def get_flop_sum(self) -> float:
        """Sum of all FLOP-related metrics"""
        return self.tenso + self.fp64a + self.fp32a + self.fp16a

    def gract_normalization(self) -> dict[str, float]:
        """Calculate all metrics normalized by graphic engine active fraction (GRACT).

        Returns zero if gract is 0 to avoid division errors.
        """
        # GRACT-normalized metrics -> source field on MetricValues.
        gract_attrs = {
            "drama_gract": "drama",
            "tenso_gract": "tenso",
            "fp64a_gract": "fp64a",
            "fp32a_gract": "fp32a",
            "fp16a_gract": "fp16a",
            "smocc_gract": "smocc",
        }
        if self.gract == 0:
            return {key: 0.0 for key in gract_attrs}
        return {key: getattr(self, attr) / self.gract for key, attr in gract_attrs.items()}
