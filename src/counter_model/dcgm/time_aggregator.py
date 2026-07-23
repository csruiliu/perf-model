from counter_model.dcgm.data_classes import MetricValues, TimeFraction, TimeWindow
from counter_model.hw_config.hw_specs import GPU


class TimeSlicer:
    """Handles time-related calculations"""

    def __init__(self, sample_interval_ms: float, ref_gpu: GPU):
        self.sample_intv_ms = sample_interval_ms
        self.gpu = ref_gpu

    def time_fraction_single_gpu(self, metrics: MetricValues) -> TimeFraction:
        """Calculate time fraction from metrics for single gpu"""
        t_kernel = self.sample_intv_ms * metrics.gract
        t_pcie = (
            self.sample_intv_ms
            * (metrics.pcitx + metrics.pcirx)
            / (self.gpu.get_specs("pcie_bw") * 1e9)
        )
        t_host = max(self.sample_intv_ms - t_kernel - t_pcie, 0)

        return TimeFraction(t_kernel, t_pcie, t_host, t_nvlink=0)

    def time_fraction_multi_gpu(self, metrics: MetricValues) -> TimeFraction:
        """Calculate time fraction from metrics for multi-gpu"""
        pass

    def get_time_window(
        self,
        overall_runtime_ms: float | None,
        start_ts: float | None,
        end_ts: float | None,
        data_length: int,
    ) -> TimeWindow:
        """Calculate time window indices"""
        if overall_runtime_ms is None:
            finish_idx = data_length
        else:
            finish_idx = min(int(overall_runtime_ms / self.sample_intv_ms), data_length)

        start_idx = int((start_ts or 0) / self.sample_intv_ms)

        if end_ts is not None:
            end_idx = min(finish_idx, int(end_ts / self.sample_intv_ms))
            if start_idx > end_idx:
                raise ValueError("End timestamp is earlier than start timestamp")
        else:
            end_idx = finish_idx

        return TimeWindow(start_idx=start_idx, end_idx=end_idx)
