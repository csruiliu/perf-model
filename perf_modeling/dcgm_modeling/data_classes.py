import pandas as pd

from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class MetricValues:
    """Container for extracted metric values from a row"""
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
    def from_row(cls, row, metrics: List[str]) -> 'MetricValues':
        """Create MetricValues from a dataframe row"""
        return cls(
            gract=getattr(row, 'GRACT', 0.0) if 'GRACT' in metrics else 0.0,
            drama=getattr(row, 'DRAMA', 0.0) if 'DRAMA' in metrics else 0.0,
            tenso=getattr(row, 'TENSO', 0.0) if 'TENSO' in metrics else 0.0,
            fp64a=getattr(row, 'FP64A', 0.0) if 'FP64A' in metrics else 0.0,
            fp32a=getattr(row, 'FP32A', 0.0) if 'FP32A' in metrics else 0.0,
            fp16a=getattr(row, 'FP16A', 0.0) if 'FP16A' in metrics else 0.0,
            smocc=getattr(row, 'SMOCC', 0.0) if 'SMOCC' in metrics else 0.0,
            pcitx=getattr(row, 'PCITX', 0.0) if 'SMOCC' in metrics else 0.0,
            pcirx=getattr(row, 'PCIRX', 0.0) if 'SMOCC' in metrics else 0.0,
            nvltx=getattr(row, 'NVLTX', 0.0) if 'SMOCC' in metrics else 0.0,
            nvlrx=getattr(row, 'NVLRX', 0.0) if 'SMOCC' in metrics else 0.0,
        )
    
    def get_flop_sum(self) -> float:
        """Sum of all FLOP-related metrics"""
        return self.tenso + self.fp64a + self.fp32a + self.fp16a
    

@dataclass
class TimeComponents:
    """Container for calculated time components"""
    t_flop: float = 0.0
    t_dram: float = 0.0
    t_kernel: float = 0.0
    t_pcie: float = 0.0
    t_nvlink: float = 0.0
    t_othernode: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            't_flop': self.t_flop,
            't_dram': self.t_dram,
            't_kernel': self.t_kernel,
            't_pcie': self.t_pcie,
            't_nvlink': self.t_nvlink,
            't_othernode': self.t_othernode
        }
    

@dataclass
class TimeSlice:
    """Container for time-based metrics with slicing functionality"""
    start_idx: int = 0
    end_idx: Optional[int] = None
    
    def slice_list(self, data: List) -> List:
        """Apply slicing to a list"""
        return data[self.start_idx:self.end_idx]
    
    def slice_dict(self, data: Dict[str, List]) -> Dict[str, List]:
        """Apply slicing to all lists in a dictionary"""
        return {
            key: values[self.start_idx:self.end_idx]
            for key, values in data.items()
        }
    
    def slice_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply slicing to a list"""
        return data[self.start_idx:self.end_idx]