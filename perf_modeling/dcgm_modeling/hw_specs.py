from dataclasses import dataclass
from typing import Dict

# GPU Specifications
# References: 
# 1. https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf
# 2. https://www.techpowerup.com/gpu-specs
# 3. NVIDIA GPU Data Sheet Webpage
# Note: Unified Cache includes L1 cache and shared memory
GPUSpec = {
    "A100-40": {
        "fp64": 9.7, "tf64": 19.5, "fp32": 19.5, "tf32": 156, "fp16": 78, "tf16": 312, 
        "mem_bw": 1555, "pcie_bw": 64, "nvlink_bw": 600, "l2_cache": 40,
        "base_clock": 765, "boost_clock": 1410, "mem_clock": 1215,
        "max_warps_sm": 64, "reg_size_sm": 256, "shmem_sm": 164, "num_sm": 108
    },
    "A100-80": {
        "fp64": 9.7, "tf64": 19.5, "fp32": 19.5, "tf32": 156, "fp16": 78, "tf16": 312, 
        "mem_bw": 1935, "pcie_bw": 64, "nvlink_bw": 600, "l2_cache": 40,
        "base_clock": 1065, "boost_clock": 1410, "mem_clock": 1512,
        "max_warps_sm": 64, "reg_size_sm": 256, "shmem_sm": 164, "num_sm": 108 
    },
    "A40": {
        "fp64": 0.58, "tf64": 0, "fp32": 37.4, "tf32": 74.8, "fp16": 37.4, "tf16": 149.7, 
        "mem_bw": 696, "pcie_bw": 64, "nvlink_bw": 112.5, "l2_cache": 6,
        "base_clock": 1305, "boost_clock": 1740, "mem_clock": 1812, 
        "max_warps_sm": 48, "reg_size_sm": 256, "shmem_sm": 100, "num_sm": 84
    },
    "H100-SXM": {
        "fp64": 34, "tf64": 67, "fp32": 67, "tf32": 495, "fp16": 267.6, "tf16": 990, 
        "mem_bw": 3350, "pcie_bw": 128, "nvlink_bw": 900, "l2_cache": 50,
        "base_clock": 1590, "boost_clock": 1980, "mem_clock": 1313, 
        "max_warps_sm": 64, "reg_size_sm": 256, "shmem_sm": 228, "num_sm": 132
    },
    "H200-SXM": {
        "fp64": 34, "tf64": 67, "fp32": 67, "tf32": 495, "fp16": 267.6, "tf16": 990, 
        "mem_bw": 4890, "pcie_bw": 128, "nvlink_bw": 900, "l2_cache": 50,
        "base_clock": 1590, "boost_clock": 1980, "mem_clock": 1593, 
        "max_warps_sm": 64, "reg_size_sm": 256, "shmem_sm": 228, "num_sm": 132
    },
    "RTX8000": {
        "fp64": 0.51, "tf64": 0, "fp32": 16.31, "tf32": 0, "fp16": 32.62, "tf16": 130.5, 
        "mem_bw": 672, "pcie_bw": 15.75, "nvlink_bw": 100, "l2_cache": 6,
        "base_clock": 1395, "boost_clock": 1770, "mem_clock": 1750,
        "max_warps_sm": 32, "reg_size_sm": 256, "shmem_sm": 96, "num_sm": 72
    },
    "V100-SXM2": {
        "fp64": 7.8, "tf64": 0, "fp32": 15.7, "tf32": 0, "fp16": 31.3, "tf16": 125, 
        "mem_bw": 900, "pcie_bw": 15.75, "nvlink_bw": 100, "l2_cache": 6,
        "base_clock": 1312, "boost_clock": 1530, "mem_clock": 876,
        "max_warps_sm": 64, "reg_size_sm": 256, "shmem_sm": 96, "num_sm": 80
    }
}


HostSpec = {
    "Perlmutter": {
        "cpu_clock_base": 2.45, "cpu_clock_boost": 3.5, "cpu_cores": 64, "mem_bw": 204.8, "pcie": 32
    },
    "Einsteinium-H100": {
        "cpu_clock_base": 2, "cpu_clock_boost": 3.8, "cpu_cores": 112, "mem_bw": 307.2, "pcie": 32
    },
    "Einsteinium-A40": {
        "cpu_clock_base": 2.25, "cpu_clock_boost": 3.4, "cpu_cores": 64, "mem_bw": 204.8, "pcie": 32
    },
    "Einsteinium-RTX8000": {
        "cpu_clock_base": 2.45, "cpu_clock_boost": 3.5, "cpu_cores": 64, "mem_bw": 204.8, "pcie": 32
    }
}


@dataclass
class GPU:
    """Encapsulates GPU specifications"""
    name: str
    specs: Dict[str, float]
    
    def __init__(self, gpu_name: str):
        self.name = gpu_name
        self.specs = GPUSpec.get(gpu_name)

    def get_name(self) -> str:
        return self.name

    def get_specs(self, key: str, default: float = 0.0) -> float:
        """Safe getter for spec values"""
        return self.specs.get(key, default)
    
    def __getitem__(self, key: str) -> float:
        """Allow dictionary-style access"""
        return self.specs[key]
    

@dataclass
class Host:
    """Encapsulates Host specifications"""
    name: str
    specs: Dict[str, float]
    
    def __init__(self, host_name: str):
        self.name = host_name
        self.specs = HostSpec.get(host_name)

    def get_name(self) -> str:
        return self.name

    def get_specs(self, key: str, default: float = 0.0) -> float:
        """Safe getter for spec values"""
        return self.specs.get(key, default)
    
    def __getitem__(self, key: str) -> float:
        """Allow dictionary-style access"""
        return self.specs[key]