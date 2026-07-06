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

    INTENSITY_THRESHOLD = 0.01

    def __init__(self, ref_gpu: GPU, tgt_gpu: GPU):
        self.ref_gpu = ref_gpu
        self.tgt_gpu = tgt_gpu

        # Initialize state
        self.cur_smocc = 0
        self.cur_warps_ref = 0
        self.cur_warps_tgt = {"lower": 0, "mid": 0, "upper": 0, "mock": 0}
        self.scale_smocc = {"lower": 0, "mid": 0, "upper": 0, "mock": 0}

        # Precompute common ratios
        self._precompute_common_ratios()

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
            self.cur_warps_ref * self.reg_sm_limit,
            self.cur_warps_ref * self.shmem_sm_limit,
            self.tgt_max_warps,
        )

        self.cur_warps_tgt["mid"] = min(
            self.cur_warps_ref * (self.reg_sm_limit + self.shmem_sm_limit) * 0.5, self.tgt_max_warps
        )

        self.cur_warps_tgt["upper"] = min(
            max(self.cur_warps_ref * self.reg_sm_limit, self.cur_warps_ref * self.shmem_sm_limit),
            self.tgt_max_warps,
        )
        self.cur_warps_tgt["mock"] = self.cur_warps_ref

    def _compute_k_smocc(self, warps: float, gpu: GPU) -> float:
        """Compute k_smocc value for given warps and GPU"""
        return warps * gpu.get_specs("num_sm") * gpu.get_specs("boost_clock")

    def _scale_metric(
        self, intensity_ref: float, ratio: float
    ) -> tuple[float, float, float, float]:
        """Generic method to compute scaling factors for any intensity metric"""
        # this is important, make sure dcgm metrics that are too small will not be considered
        if intensity_ref < self.INTENSITY_THRESHOLD:
            return np.inf, np.inf, np.inf, np.inf

        scale_factor = ratio / intensity_ref
        return tuple(
            min(self.scale_smocc[key], scale_factor) for key in ("lower", "mid", "upper", "mock")
        )

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

    def tensor_scale_weighted(
        self, tensor_ref: float, weights: dict[str, float]
    ) -> tuple[float, float, float, float]:
        """Calculate weighted tensor core scaling factors"""
        tf_tgt = sum(
            weights[prec] * self.tgt_gpu.get_specs(prec) for prec in ["tf64", "tf32", "tf16"]
        )
        tf_ref = sum(
            weights[prec] * self.ref_gpu.get_specs(prec) for prec in ["tf64", "tf32", "tf16"]
        )

        return self._compute_scale(tensor_ref, tf_tgt / tf_ref)

    def pcie_scale(self):
        return self._get_ratio("pcie_bw")

    def dram_scale(self, dram_ref: float) -> tuple[float, float, float]:
        """Calculate DRAM bandwidth scaling factors"""
        return self._compute_scale(dram_ref, self.bw_ratio)

    def dram_l2_scale(self, intensities: dict) -> tuple[float, float, float, float]:
        """Calculate DRAM bandwidth with L2 cache scaling factors (tentative)"""
        if intensities["drama_gract"] < self.INTENSITY_THRESHOLD:
            return np.inf, np.inf, np.inf, np.inf

        scale_factor = self.bw_ratio / intensities["drama_gract"]
        l2_cache_ratio = self.tgt_gpu.get_specs("l2_cache") / self.ref_gpu.get_specs("l2_cache")

        def calculate_lambda_factor(key):
            l_factor = 1 / (1 - intensities["smocc_gract"] * (l2_cache_ratio - 1))
            return min(self.scale_smocc[key], scale_factor) * l_factor

        return tuple(calculate_lambda_factor(key) for key in ["lower", "mid", "upper", "mock"])

    def fp64_scale(self, fp64_ref: float) -> tuple[float, float, float, float]:
        """Calculate FP64 scaling factors"""
        return self._compute_scale(fp64_ref, self.fp64_ratio)

    def fp32_scale(self, fp32_ref: float) -> tuple[float, float, float, float]:
        """Calculate FP32 scaling factors"""
        return self._compute_scale(fp32_ref, self.fp32_ratio)

    def fp16_scale(self, fp16_ref: float) -> tuple[float, float, float, float]:
        """Calculate FP16 scaling factors"""
        return self._compute_scale(fp16_ref, self.fp16_ratio)

    def est_flop_tgt(
        self, tf_weights: dict[str, float], intensities: dict[str, float], smocc_scale: float
    ) -> tuple[float, float, float, float]:
        # Compute tf_tgt once
        tf_tgt = (
            tf_weights["tf64"] * self.ref_gpu["tf64"]
            + tf_weights["tf32"] * self.ref_gpu["tf32"]
            + tf_weights["tf16"] * self.ref_gpu["tf16"]
        )

        # Define precision types and their references
        fps = ["fp64", "fp32", "fp16"]
        fp_refs = [
            intensities["fp64a_gract"],
            intensities["fp32a_gract"],
            intensities["fp16a_gract"],
        ]

        tensor_tgt = tf_tgt * intensities["tenso_gract"] * smocc_scale

        fp_tgts = [
            min(self.ref_gpu[prec] * ref * smocc_scale, self.tgt_gpu[prec])
            for prec, ref in zip(fps, fp_refs, strict=True)
        ]

        flop_tgt = tensor_tgt + sum(fp_tgts)

        return flop_tgt
