"""
constants.py

All constants used across the MPI model.

All other modules import from here — never define constants elsewhere.
"""

# Mapping of DCGM metrics name -> MetricValues field name.
DCGM_COUNTERS_MAP = {
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

# Column name mappings for multi-job (PKL) DCGM data.
LDMS_COLUMN_RENAME_MAP = {
    "nersc_ldms_dcgm_gr_engine_active": "GRACT",
    "nersc_ldms_dcgm_dram_active": "DRAMA",
    "nersc_ldms_dcgm_sm_occupancy": "SMOCC",
    "nersc_ldms_dcgm_tensor_active": "TENSO",
    "nersc_ldms_dcgm_fp16_active": "FP16A",
    "nersc_ldms_dcgm_fp32_active": "FP32A",
    "nersc_ldms_dcgm_fp64_active": "FP64A",
    "nersc_ldms_dcgm_nvlink_rx_bytes": "NVLRX",
    "nersc_ldms_dcgm_nvlink_tx_bytes": "NVLTX",
    "nersc_ldms_dcgm_pcie_rx_bytes": "PCIRX",
    "nersc_ldms_dcgm_pcie_tx_bytes": "PCITX",
}

# Column used to decide whether a GPU was actively used (long/raw name).
GPU_ACTIVE_THRESHOLD_METRIC = "nersc_ldms_dcgm_gr_engine_active"

# A GPU counts as "active" if at least MIN_ACTIVE_FRACTION of its samples
# have an activity value of at least MIN_ACTIVE_VALUE.
GPU_INTENSIVE_MIN_ACTIVE_FRACTION = 0.5
GPU_INTENSIVE_MIN_ACTIVE_VALUE = 0.5

# Intensities below this are considered negligible and set to infinity so
# their scale-ratio contribution collapses toward 0, excluding them from the
# governing min() in update_scale_kernel().
GPU_MIN_INTENSITY_THRESHOLD = 0.01

# SMOCC-level constants for estimation
SMOCC_LEVELS = ["lower", "mid", "upper", "mock"]
