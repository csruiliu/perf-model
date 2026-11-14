from dataclasses import dataclass

# GPU Specifications
# References:
# 1. https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf
# 2. https://www.techpowerup.com/gpu-specs
# 3. NVIDIA GPU Data Sheet Webpage
# Note: Unified Cache includes L1 cache and shared memory

# Ruff won't format this code
# fmt: off

HostSpec = {
    "Perlmutter": {
        "cpu_clock_base": 2.45, "cpu_clock_boost": 3.5, "cpu_cores": 64, "mem_bw": 204.8, "pcie": 32,
        "network_bw": 25
    },
    "Einsteinium-H100": {
        "cpu_clock_base": 2, "cpu_clock_boost": 3.8, "cpu_cores": 112, "mem_bw": 307.2, "pcie": 32
    },
    "Einsteinium-A40": {
        "cpu_clock_base": 2.25, "cpu_clock_boost": 3.4, "cpu_cores": 64, "mem_bw": 204.8, "pcie": 32
    },
    "Einsteinium-RTX8000": {
        "cpu_clock_base": 2.45, "cpu_clock_boost": 3.5, "cpu_cores": 64, "mem_bw": 204.8, "pcie": 32
    },
    ############# Hypothetical Hosts #############
    "Perlmutter-NG2": {
        "cpu_clock_base": 4.9, "cpu_clock_boost": 7, "cpu_cores": 64, "mem_bw": 204.8, "pcie": 32,
        "network_bw": 25
    },
    "Perlmutter-NG4": {
        "cpu_clock_base": 9.8, "cpu_clock_boost": 14, "cpu_cores": 64, "mem_bw": 204.8, "pcie": 32,
        "network_bw": 25
    },
}


@dataclass
class Host:
    """Encapsulates Host specifications"""
    name: str
    specs: dict[str, float]

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
