import numpy as np

from counter_model.hw_config.hw_specs import GPU, Host


def get_tf_weights(fp64a: float, fp32a: float, fp16a: float, threshold=0.01) -> dict[str, float]:
    """
    Calculate weights for tf64, tf32, tf16 based on FP operations
    """
    # Apply threshold - treat values < 0.01 as 0
    fp64a = fp64a if fp64a >= threshold else 0.0
    fp32a = fp32a if fp32a >= threshold else 0.0
    fp16a = fp16a if fp16a >= threshold else 0.0

    total = fp64a + fp32a + fp16a

    # If total is 0 (and TENSO > 0.01, this is guaranteed in other function), give equal weights
    if total == 0:
        return {"tf64": 1 / 3, "tf32": 1 / 3, "tf16": 1 / 3}

    return {"tf64": fp64a / total, "tf32": fp32a / total, "tf16": fp16a / total}


class HostScaler:
    """Calculates scale factors for host"""

    def __init__(self, ref_host: Host, tgt_host: Host):
        self.ref_host = ref_host
        self.tgt_host = tgt_host
        self._precompute_common_ratios()

    def _precompute_common_ratios(self):
        cpu_clock_ratio_ref = np.mean(
            [self.ref_host.get_specs("cpu_clock_base"), self.ref_host.get_specs("cpu_clock_boost")]
        )
        cpu_clock_ratio_tgt = np.mean(
            [self.tgt_host.get_specs("cpu_clock_base"), self.tgt_host.get_specs("cpu_clock_boost")]
        )
        self.cpu_clock_ratio = cpu_clock_ratio_tgt / cpu_clock_ratio_ref

        # self.cpu_clock_ratio = self._get_ratio("cpu_clock_boost")
        self.dram_ratio = self._get_ratio("mem_bw")
        self.pcie_ratio = self._get_ratio("pcie")
        self.cpu_cores_ratio = self._get_ratio("cpu_cores")

    def _get_ratio(self, spec: str) -> float:
        """Helper to compute target/reference ratio for a given spec"""
        return self.tgt_host.get_specs(spec) / self.ref_host.get_specs(spec)

    def host_scale(self, cores_alloc: str) -> float:
        if cores_alloc == "same":
            return self.cpu_clock_ratio
        else:
            return self.cpu_clock_ratio * self.cpu_cores_ratio


class GpuScaler:
    """Calculates computational intensities"""

    METRIC_THRESHOLD = 0.01

    def __init__(self, ref_gpu: GPU, tgt_gpu: GPU, smocc_levels: list[str]):
        self.ref_gpu = ref_gpu
        self.tgt_gpu = tgt_gpu
        self.smocc_levels = smocc_levels

        # Initialize state
        self.cur_smocc = 0
        self.cur_warps_ref = 0
        self.cur_warps_tgt = {level: 0 for level in smocc_levels}
        self.scale_smocc = {level: 0 for level in smocc_levels}
        self.scale_kernel = {level: 0 for level in smocc_levels}

        # Precompute common ratios
        self._precompute_common_ratios()

    def update_smocc(self, smocc: float):
        self.cur_smocc = smocc
        self._estimate_warps()

        for key in self.scale_smocc.keys():
            k_smocc_tgt = self._compute_k_smocc(self.cur_warps_tgt[key], self.tgt_gpu)
            k_smocc_ref = self._compute_k_smocc(self.cur_warps_ref, self.ref_gpu)
            if k_smocc_ref == 0 or k_smocc_tgt == 0:
                self.scale_smocc[key] = np.inf
            else:
                self.scale_smocc[key] = k_smocc_tgt / k_smocc_ref

    def update_scale_kernel(self, mv_gract_norm: dict, tf_weights: dict):
        tf_precisions = ("tf64", "tf32", "tf16")
        tf_tgt = sum(tf_weights[p] * self.tgt_gpu.get_specs(p) for p in tf_precisions)
        tf_ref = sum(tf_weights[p] * self.ref_gpu.get_specs(p) for p in tf_precisions)

        # Below-threshold intensities are treated as infinite (i.e. non-binding),
        # so their ratio contribution collapses to 0 and won't drive the min.
        gract_keys = ("tenso_gract", "drama_gract", "fp64a_gract", "fp32a_gract", "fp16a_gract")
        for key in gract_keys:
            if mv_gract_norm[key] < self.METRIC_THRESHOLD:
                mv_gract_norm[key] = np.inf

        # Candidate scale ratio per resource; the tightest one governs, ignore zeros
        for level in self.smocc_levels:
            self.scale_kernel[level] = min(
                x
                for x in [
                    self.scale_smocc[level],
                    tf_tgt / (tf_ref * mv_gract_norm["tenso_gract"]),
                    self._get_ratio("mem_bw") / mv_gract_norm["drama_gract"],
                    self._get_ratio("fp64") / mv_gract_norm["fp64a_gract"],
                    self._get_ratio("fp32") / mv_gract_norm["fp32a_gract"],
                    self._get_ratio("fp16") / mv_gract_norm["fp16a_gract"],
                ]
                if x != 0
            )

    def pcie_scale(self):
        return self._get_ratio("pcie_bw")

    def _precompute_common_ratios(self):
        """Compute GPU spec ratios that don't depend on tensor precision"""
        self.reg_sm_limit = self._get_ratio("reg_size_sm")
        self.shmem_sm_limit = self._get_ratio("shmem_sm")
        self.bw_ratio = self._get_ratio("mem_bw")

        # Store specs locally to avoid repeated method calls
        self.ref_max_warps = self.ref_gpu.get_specs("max_warps_sm")
        self.tgt_max_warps = self.tgt_gpu.get_specs("max_warps_sm")

        self.fp64_ratio = self._get_ratio("fp64")
        self.fp32_ratio = self._get_ratio("fp32")
        self.fp16_ratio = self._get_ratio("fp16")

    def _get_tensor_ratio(self, precision: str) -> float:
        """Compute ratio for a specific precision"""
        if precision not in self.precision_ratios:
            self.precision_ratios[precision] = self._get_ratio(precision)
        return self.precision_ratios[precision]

    def _get_ratio(self, spec: str) -> float:
        """Helper to compute target/reference ratio for a given spec"""
        return self.tgt_gpu.get_specs(spec) / self.ref_gpu.get_specs(spec)

    def _estimate_warps(self):
        self.cur_warps_ref = min(self.cur_smocc * self.ref_max_warps, self.ref_max_warps)

        self.cur_warps_tgt["lower"] = min(
            self.cur_warps_ref * max(self.reg_sm_limit, self.shmem_sm_limit), self.tgt_max_warps
        )

        self.cur_warps_tgt["mid"] = min(
            self.cur_warps_ref * (self.reg_sm_limit + self.shmem_sm_limit) / 2, self.tgt_max_warps
        )

        self.cur_warps_tgt["upper"] = min(
            self.cur_warps_ref * max(self.reg_sm_limit, self.shmem_sm_limit), self.tgt_max_warps
        )
        self.cur_warps_tgt["mock"] = self.cur_warps_ref

    def _compute_k_smocc(self, warps: float, gpu: GPU) -> float:
        """Compute k_smocc value for given warps and GPU"""
        return warps * gpu.get_specs("num_sm") * gpu.get_specs("boost_clock")
