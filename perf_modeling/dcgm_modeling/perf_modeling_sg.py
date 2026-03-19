import argparse
import pandas as pd
import numpy as np

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from hw_specs import GPU, GPUSpec, Host, HostSpec
from data_classes import MetricValues, TimeComponents, TimeSlice
from job_processor import JobProcessor 
from utils import ResultsFormatter
from performance_calculators import (
    MetricIntensityCalculator,
    GPUScaleCalculator,
    TimeCalculator,
    TFWeightCalculator,
    HostScaleCalculator,
)


class BaseProfiler(ABC):
    """Abstract base class for profilers"""
    
    def __init__(self, sample_interval_ms: float, ref_gpu: GPU):
        self.time_calc = TimeCalculator(sample_interval_ms, ref_gpu)
        self.intensity_calc = MetricIntensityCalculator()
        self.formatter = ResultsFormatter()
        
    @abstractmethod
    def run(self, *args, **kwargs):
        """Run the profiling/prediction"""
        pass


class ReferenceProfiler(BaseProfiler):
    """Profiles performance on reference hardware"""
    def __init__(self, sample_interval_ms, gpu_name):
        self.gpu = GPU(gpu_name=gpu_name)
        super().__init__(sample_interval_ms, self.gpu)

    def run(self, profiled_df: pd.DataFrame, metrics: List[str], 
            overall_runtime_ms: float, start_ts: Optional[float], 
            end_ts: Optional[float]):
        """Model performance on reference hardware"""        
        # Calculate components for all rows
        components_list = self._calc_all_components(profiled_df, metrics)
        
        # Get time slice
        time_slice = self.time_calc.get_time_slice(
            overall_runtime_ms, start_ts, end_ts, len(components_list)
        )
        
        # Slice and aggregate components
        sliced = self._slice_and_aggregate(components_list, time_slice)
        
        # Calculate performance metrics
        flops = self._calc_flops(time_slice.slice_dataframe(profiled_df), metrics)
        membw = self._calc_membw(time_slice.slice_dataframe(profiled_df), metrics)
        
        # Print results
        self.formatter.print_reference_results(sliced, flops, membw, self.gpu.get_name())
    
    def _calc_all_components(self, profiled_df: pd.DataFrame, metrics: List[str]) -> List[TimeComponents]:
        """Calculate time components for all rows"""
        return [
            self.time_calc.calc_components_sg(MetricValues.from_row(row, metrics))
            for row in profiled_df.itertuples(index=False)
        ]
    
    def _slice_and_aggregate(self, components_list: List[TimeComponents], time_slice: TimeSlice) -> Dict[str, List[float]]:
        """Slice components and add total time"""
        sliced = {
            key: [comp.to_dict()[key] for comp in components_list][time_slice.start_idx:time_slice.end_idx]
            for key in components_list[0].to_dict().keys()
        }
        
        # Add total time
        sliced['t_total'] = [
            sliced['t_kernel'][i] + sliced['t_othernode'][i]
            for i in range(len(sliced['t_kernel']))
        ]
        
        return sliced
    
    def _calc_flops(self, profiled_df: pd.DataFrame, metrics: List[str]) -> float:
        """Calculate FLOPS"""
        flop_sum = 0
        tf_weight_calc = TFWeightCalculator()

        for row in profiled_df.itertuples(index=False):
            mv = MetricValues.from_row(row, metrics)
            intensities = self.intensity_calc.metric_intensities(mv)

             # Calculate weights for this row
            tf_weights = tf_weight_calc.calculate_weights(mv.fp64a, mv.fp32a, mv.fp16a)

            tensor_util = intensities['tenso_gract'] * (
                tf_weights['tf64'] * self.gpu.get_specs("tf64") +
                tf_weights['tf32'] * self.gpu.get_specs("tf32") +
                tf_weights['tf16'] * self.gpu.get_specs("tf16")
            )
            
            fp64_util = intensities['fp64a_gract'] * self.gpu.get_specs("fp64")
            fp32_util = intensities['fp32a_gract'] * self.gpu.get_specs("fp32")
            fp16_util = intensities['fp16a_gract'] * self.gpu.get_specs("fp16")
            flop_sum += tensor_util + fp64_util + fp32_util + fp16_util

        return flop_sum / len(profiled_df)
    
    def _calc_membw(self, profiled_df: pd.DataFrame, metrics: List[str]) -> float:
        """Calculate memory bandwidth"""
        dram_sum = 0
        for row in profiled_df.itertuples(index=False):
            mv = MetricValues.from_row(row, metrics)
            intensities = self.intensity_calc.metric_intensities(mv)
            dram_sum += intensities['drama_gract'] * self.gpu.get_specs("mem_bw")
        
        return dram_sum / len(profiled_df)


class TargetPredictor(BaseProfiler):
    """Predicts performance on target hardware"""

    # Class-level constants
    SMOCC_LEVELS = ['lower', 'mid', 'upper', 'mock']

    def __init__(self, sample_interval_ms, ref_gpu_name, tgt_gpu_name, ref_host_name, tgt_host_name):
        self.ref_gpu = GPU(gpu_name=ref_gpu_name)
        self.tgt_gpu = GPU(gpu_name=tgt_gpu_name)
        self.ref_host = Host(host_name=ref_host_name)
        self.tgt_host = Host(host_name=tgt_host_name)
        super().__init__(sample_interval_ms, self.ref_gpu)

    def run(self, profiled_df: pd.DataFrame, metrics: List[str],
            overall_runtime_ms: float, start_ts: Optional[float], 
            end_ts: Optional[float], cores_alloc: str):
        """Predict performance on target hardware"""        
        # Calculate target metrics
        target_metrics = self._calc_target_metrics(profiled_df, metrics, cores_alloc)
        
        # Get time slice
        time_slice = self.time_calc.get_time_slice(
            overall_runtime_ms, start_ts, end_ts, len(target_metrics['t_total_lower'])
        )
        
        # Slice metrics
        sliced_metrics = time_slice.slice_dict(target_metrics)
        
        # Calculate estimated FLOPS and memory bandwidth
        #est_flops = self._calc_aggregated_metrics(sliced_metrics, 'total_flop_tgt', 'flop_smocc')
        #est_membw = self._calc_aggregated_metrics(sliced_metrics, 'total_dram_tgt', 'dram_smocc')
        
        # Print predictions
        self.formatter.print_target_results(sliced_metrics, self.tgt_gpu.get_name())
    
    def _calc_target_metrics(self, profiled_df: pd.DataFrame, metrics: List[str], cores_alloc: str) -> Dict[str, List[float]]:
        """Calculate metrics for target hardware"""

        metric_types = ['t_kernel', 't_total', 'total_dram_tgt', 'total_flop_tgt']
        
        results = {f'{metric}_{key}': [] 
                   for metric in metric_types 
                   for key in self.SMOCC_LEVELS}

        results['t_othernode'] = []
        results['t_pcie'] = []
        
        gpu_scale_calc = GPUScaleCalculator(self.ref_gpu, self.tgt_gpu)
        host_scale_calc = HostScaleCalculator(self.ref_host, self.tgt_host)
        tf_weight_calc = TFWeightCalculator()
        
        for row in profiled_df.itertuples(index=False):
            mv = MetricValues.from_row(row, metrics)
            
            # Calculate intensities
            intensities = self.intensity_calc.metric_intensities(mv)

            # Calculate weights for this row
            tf_weights = tf_weight_calc.calculate_weights(mv.fp64a, mv.fp32a, mv.fp16a)

            # Calculate reference components
            ref_components = self.time_calc.calc_components_sg(mv)
            
            # Update SMOCC and calculate all scales
            gpu_scale_calc.update_smocc(intensities['smocc_gract'])
            all_scales = self._calculate_all_scales(gpu_scale_calc, intensities, tf_weights)
            
            # PCIe Time
            t_pcie_tgt = ref_components.t_pcie / gpu_scale_calc.pcie_scale()
            results['t_pcie'].append(t_pcie_tgt)

            # Other node time
            t_othernode_tgt = ref_components.t_othernode / host_scale_calc.othernode_scale(cores_alloc)
            results['t_othernode'].append(t_othernode_tgt)

            # Process each SMOCC key
            for i, key in enumerate(self.SMOCC_LEVELS):
                # Calculate kernel scale (minimum of all constraints)
                kernel_scale = min(
                    x for x in [
                        gpu_scale_calc.scale_smocc[key],
                        all_scales['dram'][i],
                        all_scales['tensor'][i],
                        all_scales['fp64'][i],
                        all_scales['fp32'][i],
                        all_scales['fp16'][i]
                    ] if x != 0
                )

                # Calculate kernel and total time
                t_kernel_tgt = ref_components.t_kernel / kernel_scale
                results[f't_kernel_{key}'].append(t_kernel_tgt)
                results[f't_total_{key}'].append(t_kernel_tgt + t_pcie_tgt + t_othernode_tgt)
        
        return results
        
    def _calculate_all_scales(self, scale_calc: GPUScaleCalculator, intensities: Dict, tf_weights: Dict) -> Dict[str, tuple]:
        """Calculate all scale factors in one place"""
        # scale_calc.smocc_scale() need to be invoked first
        return {
            'dram': scale_calc.dram_scale(intensities['drama_gract']),
            'tensor': scale_calc.tensor_scale_weighted(intensities['tenso_gract'], tf_weights),
            'fp64': scale_calc.fp64_scale(intensities['fp64a_gract']),
            'fp32': scale_calc.fp32_scale(intensities['fp32a_gract']),
            'fp16': scale_calc.fp16_scale(intensities['fp16a_gract'])
        }

    def _calc_aggregated_metrics(self, results: Dict[str, List], src_prefix: str, tgt_prefix: str) -> Dict[str, float]:
        """Generic method to calculate aggregated metrics (FLOPS or memory bandwidth)"""
        return {
            f"{tgt_prefix}_{key}": np.mean(results[f'{src_prefix}_{key}'])
            for key in self.SMOCC_LEVELS
        }


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='GPU Performance Profiler and Predictor',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('-f', '--dcgm_file', required=True, help='DCGM output file path')
    parser.add_argument('-d', '--sample_interval_ms', type=int, required=True, help='Sample interval in milliseconds')
    parser.add_argument('-st', '--start_timestamp', type=int, default=0, help='Start timestamp (ms, default: 0)')
    parser.add_argument('-et', '--end_timestamp', type=int, default=None, help='End timestamp (ms, default: None)')
    parser.add_argument('-o', '--overall_runtime_ms', type=int, required=True, help='Overall runtime in milliseconds')
    parser.add_argument('-rg', '--ref_gpu', required=True, choices=list(GPUSpec.keys()), help='Reference GPU')
    parser.add_argument('-tg', '--tgt_gpu', choices=list(GPUSpec.keys()), help='Target Host (optional)')
    parser.add_argument('-rh', '--ref_host', required=True, choices=list(HostSpec.keys()), help='Reference Host')
    parser.add_argument('-th', '--tgt_host', choices=list(HostSpec.keys()), help='Target Host (optional)')
    parser.add_argument('-ca', '--cores_alloc', type=str, required=True, choices=['same', 'all'], help='CPU Cores Allocation Strategy (optional)')
    parser.add_argument('--metrics', type=lambda s: s.split(','), required=True, help='Comma-separated list of metrics (e.g., GRACT, DRAMA, SMOCC)')
        
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    # Process metrics file for a job
    job_processor = JobProcessor(1, args.metrics)
    profiled_df = job_processor.process_files(args.dcgm_file)

    # Create and run reference profiler
    ref_profiler = ReferenceProfiler(args.sample_interval_ms, args.ref_gpu)
    ref_profiler.run(
        profiled_df, args.metrics, args.overall_runtime_ms,
        args.start_timestamp, args.end_timestamp
    )
    
    # Create target predictor and run if specified
    if args.tgt_gpu:
        tgt_predictor = TargetPredictor(args.sample_interval_ms, args.ref_gpu, args.tgt_gpu, args.ref_host, args.tgt_host)
        tgt_predictor.run(
            profiled_df, args.metrics, args.overall_runtime_ms,
            args.start_timestamp, args.end_timestamp, args.cores_alloc
        )


if __name__=="__main__":
    main()