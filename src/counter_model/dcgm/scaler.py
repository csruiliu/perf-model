import numpy as np

from counter_model.dcgm.constants import GPU_MIN_INTENSITY_THRESHOLD, TF_TO_FP
from counter_model.hw_config.hw_specs import GPU, Host


def get_tf_weights(
    fp64a: float, fp32a: float, fp16a: float, ref_gpu: GPU, threshold: float = 0.01
) -> dict[str, float]:
    """
    FLOP-share weights for the tensor precision mix (numerator terms of Eq. 10b):
        w_p ∝ A_fpp^ref * β_fpp^ref
    Precisions with no tensor support on the reference GPU are excluded, since
    tensor activity observed on the reference cannot have been at that precision.
    """
    activity = {"tf64": fp64a, "tf32": fp32a, "tf16": fp16a}
    supported = [p for p in TF_TO_FP if ref_gpu.get_specs(p) > 0]

    flop_share = {p: 0.0 for p in TF_TO_FP}
    for p in supported:
        a = activity[p]
        if a >= threshold:
            flop_share[p] = a * ref_gpu.get_specs(TF_TO_FP[p])

    total = sum(flop_share.values())
    if total == 0.0:
        # Conservative fallback: equal FLOP shares over reference-supported precisions
        n = len(supported)
        return {p: (1.0 / n if p in supported else 0.0) for p in TF_TO_FP}

    return {p: s / total for p, s in flop_share.items()}


def effective_tensor_peak(tf_weights: dict[str, float], gpu: GPU) -> float:
    """
    Weighted harmonic mean of per-precision tensor peaks (Eq. 10b):
        β_tensor^a = 1 / Σ_p ( ŵ_p / β_tfp^a )
    """
    denom = 0.0
    for p, w in tf_weights.items():
        if w == 0.0:
            continue
        denom += w / gpu.get_specs(p)
    
    return 1.0 / denom if denom > 0.0 else 0.0


class HostScaler:
    """Calculates scale factors for host"""

    def __init__(self, ref_host: Host, tgt_host: Host):
        self.ref_host = ref_host
        self.tgt_host = tgt_host
        self._precompute_common_ratios()

    def _precompute_common_ratios(self):
        cpu_clock_ratio_ref = self.ref_host.get_specs("cpu_clock_boost")
        cpu_clock_ratio_tgt = self.tgt_host.get_specs("cpu_clock_boost")
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

    def __init__(self, ref_gpu: GPU, tgt_gpu: GPU, smocc_levels: list[str]):
        self.ref_gpu = ref_gpu
        self.tgt_gpu = tgt_gpu
        self.smocc_levels = smocc_levels

        # Initialize state
        self.cur_smocc = 0
        self.cur_warps_ref = 0
        self.cur_warps_tgt = {level: 0.0 for level in smocc_levels}
        self.scale_smocc = {level: 0.0 for level in smocc_levels}
        self.scale_kernel = {level: 0.0 for level in smocc_levels}

        # Precompute common ratios
        self._precompute_common_ratios()

    def update_smocc(self, smocc: float):
        self.cur_smocc = smocc
        self._estimate_warps()

        for key in self.smocc_levels:
            k_smocc_tgt = self._compute_k_smocc(self.cur_warps_tgt[key], self.tgt_gpu)
            k_smocc_ref = self._compute_k_smocc(self.cur_warps_ref, self.ref_gpu)
            if k_smocc_ref == 0 or k_smocc_tgt == 0:
                self.scale_smocc[key] = np.inf
            else:
                self.scale_smocc[key] = k_smocc_tgt / k_smocc_ref

    def update_scale_kernel(self, mv_gract_norm: dict, tf_ref: float, tf_tgt: float):
        """
        Per-resource scale factors K^tgt/K^ref (Eqs. 9c, 10a); the tightest governs (Eq. 11).
        Below-threshold activities are non-binding and simply omitted. The input dict
        is NOT mutated.
        """
        ceilings = []
        # Eq. (10a), second argument: β_tensor^tgt / (A_tensor^ref β_tensor^ref)
        if mv_gract_norm["tenso_gract"] >= GPU_MIN_INTENSITY_THRESHOLD and tf_ref > 0.0:
            ceilings.append(tf_tgt / (tf_ref * mv_gract_norm["tenso_gract"]))

        # Eq. (9c) and the analogous non-tensor FP ceilings
        for key, ratio in (
            ("drama_gract", self.bw_ratio),
            ("fp64a_gract", self.fp64_ratio),
            ("fp32a_gract", self.fp32_ratio),
            ("fp16a_gract", self.fp16_ratio),
        ):
            if mv_gract_norm[key] >= GPU_MIN_INTENSITY_THRESHOLD:
                ceilings.append(ratio / mv_gract_norm[key])

        # min(γ, ·) for every ceiling == global min including γ (Eq. 11)
        for level in self.smocc_levels:
            self.scale_kernel[level] = min(ceilings + [self.scale_smocc[level]])

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

    def _get_ratio(self, spec: str) -> float:
        """Helper to compute target/reference ratio for a given spec"""
        return self.tgt_gpu.get_specs(spec) / self.ref_gpu.get_specs(spec)

    def _estimate_warps(self):
        self.cur_warps_ref = min(self.cur_smocc * self.ref_max_warps, self.ref_max_warps)

        self.cur_warps_tgt["lower"] = min(
            self.cur_warps_ref * min(self.reg_sm_limit, self.shmem_sm_limit), self.tgt_max_warps
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
