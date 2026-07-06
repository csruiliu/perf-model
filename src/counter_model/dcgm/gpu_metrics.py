from dataclasses import dataclass

# Mapping of DCGM metrics name -> MetricValues field name.
_ROW_FIELD_MAP = {
    "GRACT": "gract",
    "DRAMA": "drama",
    "TENSO": "tenso",
    "FP64A": "fp64a",
    "FP32A": "fp32a",
    "FP16A": "fp16a",
    "SMOCC": "smocc",
    "PCITX": "pcitx",
    "PCIRX": "pcirx",
    "NVLTX": "nvltx",
    "NVLRX": "nvlrx",
}

# GRACT-normalized metrics -> source field on MetricValues.
_GRACT_ATTRS = {
    "drama_gract": "drama",
    "tenso_gract": "tenso",
    "fp64a_gract": "fp64a",
    "fp32a_gract": "fp32a",
    "fp16a_gract": "fp16a",
    "smocc_gract": "smocc",
}


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
    def from_row(cls, row, metrics: list[str]) -> "MetricValues":
        """Create MetricValues from a dataframe row"""
        metric_set = set(metrics)

        return cls(
            **{
                field: getattr(row, col, 0.0) if col in metric_set else 0.0
                for col, field in _ROW_FIELD_MAP.items()
            }
        )

    def get_flop_sum(self) -> float:
        """Sum of all FLOP-related metrics"""
        return self.tenso + self.fp64a + self.fp32a + self.fp16a

    def gract_normalization(self) -> dict[str, float]:
        """Calculate all metrics normalized by graphic engine active fraction (GRACT).

        Returns zero if gract is 0 to avoid division errors.
        """
        if self.gract == 0:
            return {key: 0.0 for key in _GRACT_ATTRS}
        return {key: getattr(self, attr) / self.gract for key, attr in _GRACT_ATTRS.items()}
