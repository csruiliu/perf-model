import argparse
import pandas as pd
import numpy as np

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from hw_specs import GPU, GPUSpec
from data_classes import MetricValues, TimeComponents, TimeSlice
from job_processor import JobProcessor
from performance_calculators import MetricIntensityCalculator, ScaleCalculator, TimeCalculator
from utils import ResultsFormatter


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
    """Profiles performance on reference hardware for multiple GPUs"""
    
    def __init__(self, sample_interval_ms: float, gpu_name: str):
        self.gpu = GPU(gpu_name=gpu_name)
        super().__init__(sample_interval_ms, self.gpu)

    def run(self, gpu_dfs: List[pd.DataFrame], metrics: List[str], 
            overall_runtime_ms: float, agg_interval_ms: float,
            start_ts: Optional[float], end_ts: Optional[float], 
            tensor_prec: str):
        """Model performance on reference hardware for multiple GPUs"""
        
        # Process each GPU
        all_gpu_metrics = []
        for gpu_id, df in enumerate(gpu_dfs):
            if df.empty:
                raise ValueError(f"GPU {gpu_id} DataFrame is empty")
            
            # print(f"\nProcessing GPU {gpu_id}...")
            gpu_metrics = self._process_single_gpu(df, metrics, overall_runtime_ms, 
                                                   start_ts, end_ts, tensor_prec)
            all_gpu_metrics.append(gpu_metrics)
        
        # Aggregate across GPUs
        aggregated_metrics = self._aggregate_multi_gpu(all_gpu_metrics, agg_interval_ms)
        
        # Calculate overall performance metrics
        flops = self._calc_flops(aggregated_metrics, tensor_prec)
        mem_bw = self._calc_mem_bw(aggregated_metrics)
        
        # Print results
        self.formatter.print_reference_results(aggregated_metrics, flops, mem_bw, self.gpu.get_name())
    
    def _process_single_gpu(self, df: pd.DataFrame, metrics: List[str],
                           overall_runtime_ms: float, start_ts: Optional[float],
                           end_ts: Optional[float]) -> Dict[str, List[float]]:
        """Process a single GPU's data"""
        # Calculate components for all rows
        components_list = [
            self.time_calc.calc_components(MetricValues.from_row(row, metrics))
            for row in df.itertuples(index=False)
        ]
        
        # Get time slice
        time_slice = self.time_calc.get_time_slice(
            overall_runtime_ms, start_ts, end_ts, len(components_list)
        )
        
        # Slice and aggregate components
        sliced = self._slice_and_aggregate(components_list, time_slice)
        
        return sliced
    
    def _slice_and_aggregate(self, components_list: List[TimeComponents], time_slice: TimeSlice) -> Dict[str, List[float]]:
        """Slice components and add total time"""
        sliced = {
            key: [comp.to_dict()[key] for comp in components_list][time_slice.start_idx:time_slice.end_idx]
            for key in components_list[0].to_dict().keys()
        }
        
        # Add total time
        sliced['t_total'] = [
            sliced['t_kernel'][i] + sliced['t_pcie'][i] + 
            sliced['t_nvlink'][i] + sliced['t_othernode'][i]
            for i in range(len(sliced['t_kernel']))
        ]
        
        return sliced
    
    def _aggregate_multi_gpu(self, all_gpu_metrics: List[Dict[str, List[float]]], 
                            agg_interval_ms: float) -> Dict[str, List[float]]:
        """Aggregate metrics across multiple GPUs"""
        # Check all GPUs have same length
        lengths = [len(m['t_total']) for m in all_gpu_metrics]
        if len(set(lengths)) != 1:
            raise ValueError("Not all GPU metric lists are of the same length!")
        
        num_rows = lengths[0]
        agg_samples = agg_interval_ms // self.sample_interval_ms
        
        # Aggregate by taking max across GPUs for each time window
        aggregated = {key: [] for key in all_gpu_metrics[0].keys()}
        
        for start in range(0, num_rows, agg_samples):
            end = min(start + agg_samples, num_rows)
            
            for key in aggregated.keys():
                # Sum within each GPU's window, then take max across GPUs
                window_values = [
                    sum(gpu_metrics[key][row_idx] for row_idx in range(start, end))
                    for gpu_metrics in all_gpu_metrics
                ]
                aggregated[key].append(max(window_values))
        
        return aggregated
    
    def _calc_flops(self, sliced: Dict[str, List[float]], tensor_prec: str) -> float:
        """Calculate FLOPS"""
        return (np.mean(sliced['t_flop']) / self.time_calc.sample_intv * 
                self.gpu.get_specs(tensor_prec))
    
    def _calc_mem_bw(self, sliced: Dict[str, List[float]]) -> float:
        """Calculate memory bandwidth"""
        return (np.mean(sliced['t_dram']) / self.time_calc.sample_intv * 
                self.gpu.get_specs("mem_bw"))


class TargetPredictor(BaseProfiler):
    """Predicts performance on target hardware for multiple GPUs"""
    
    def __init__(self, sample_interval_ms: float, ref_gpu_name: str, tgt_gpu_name: str):
        self.ref_gpu = GPU(gpu_name=ref_gpu_name)
        self.tgt_gpu = GPU(gpu_name=tgt_gpu_name)
        super().__init__(sample_interval_ms)
        self.time_calc = TimeCalculator(sample_interval_ms, self.ref_gpu)

    def run(self, gpu_dfs: List[pd.DataFrame], metrics: List[str],
            overall_runtime_ms: float, agg_interval_ms: float,
            start_ts: Optional[float], end_ts: Optional[float], 
            tensor_prec: str):
        """Predict performance on target hardware for multiple GPUs"""
        
        # Process each GPU
        all_gpu_metrics = []
        for gpu_id, df in enumerate(gpu_dfs):
            if df.empty:
                raise ValueError(f"GPU {gpu_id} DataFrame is empty")
            
            # print(f"\nPredicting for GPU {gpu_id}...")
            gpu_metrics = self._predict_single_gpu(df, metrics, overall_runtime_ms,
                                                   start_ts, end_ts, tensor_prec)
            all_gpu_metrics.append(gpu_metrics)
        
        # Aggregate across GPUs
        aggregated_metrics = self._aggregate_multi_gpu(all_gpu_metrics, agg_interval_ms)
        
        # Calculate estimated FLOPS and memory bandwidth
        est_flops = self._calc_est_flops(aggregated_metrics, tensor_prec)
        est_mem_bw = self._calc_est_mem_bw(aggregated_metrics)
        
        # Print predictions
        self.formatter.print_target_results(aggregated_metrics, est_flops, est_mem_bw, 
                                           self.tgt_gpu.get_name())
    
    def _predict_single_gpu(self, df: pd.DataFrame, metrics: List[str],
                           overall_runtime_ms: float, start_ts: Optional[float],
                           end_ts: Optional[float], tensor_prec: str) -> Dict[str, List[float]]:
        """Predict performance for a single GPU"""
        # Calculate target metrics
        target_metrics = self._calc_target_metrics(df, metrics, tensor_prec)
        
        # Get time slice
        time_slice = self.time_calc.get_time_slice(
            overall_runtime_ms, start_ts, end_ts, len(target_metrics['t_total_lower'])
        )
        
        # Slice metrics
        sliced_metrics = time_slice.slice_dict(target_metrics)
        
        return sliced_metrics
    
    def _calc_target_metrics(self, df: pd.DataFrame, metrics: List[str],
                            tensor_prec: str) -> Dict[str, List[float]]:
        """Calculate metrics for target hardware"""
        results = {
            't_kernel_lower': [], 't_kernel_mid': [], 't_kernel_upper': [],
            't_pcie': [], 't_nvlink': [], 't_othernode': [],
            't_total_lower': [], 't_total_mid': [], 't_total_upper': [],
            'drama_ref': [], 'tensor_ref': [], 'fp64a_ref': [], 
            'fp32a_ref': [], 'fp16a_ref': []
        }
        
        scale_calc = ScaleCalculator(self.ref_gpu, self.tgt_gpu, tensor_prec)
        
        for row in df.itertuples(index=False):
            mv = MetricValues.from_row(row, metrics)
            
            # Store reference metrics
            results['drama_ref'].append(mv.drama)
            results['tensor_ref'].append(mv.tenso)
            results['fp64a_ref'].append(mv.fp64a)
            results['fp32a_ref'].append(mv.fp32a)
            results['fp16a_ref'].append(mv.fp16a)
            
            # Calculate intensities
            intensities = self.intensity_calc.metric_intensities(mv)
            
            # Calculate reference components
            ref_components = self.time_calc.calc_components(mv)
            
            # Estimate Warps on target GPU
            scale_calc.refresh_smocc(intensities['smocc_gract'])
            
            # Calculate kernel scales
            smocc_lower, smocc_mid, smocc_upper = scale_calc.smocc_scale()
            dram_lower, dram_mid, dram_upper = scale_calc.dram_scale(intensities['drama_gract'])
            tensor_lower, tensor_mid, tensor_upper = scale_calc.tensor_scale(intensities['tenso_gract'])
            fp64_lower, fp64_mid, fp64_upper = scale_calc.fp64_scale(intensities['fp64a_gract'])
            fp32_lower, fp32_mid, fp32_upper = scale_calc.fp32_scale(intensities['fp32a_gract'])
            fp16_lower, fp16_mid, fp16_upper = scale_calc.fp16_scale(intensities['fp16a_gract'])
            
            kernel_scale_lower = min(smocc_lower, dram_lower, tensor_lower, 
                                    fp64_lower, fp32_lower, fp16_lower)
            kernel_scale_mid = min(smocc_mid, dram_mid, tensor_mid, 
                                  fp64_mid, fp32_mid, fp16_mid)
            kernel_scale_upper = min(smocc_upper, dram_upper, tensor_upper, 
                                    fp64_upper, fp32_upper, fp16_upper)
            
            # Calculate kernel times for each scenario
            for scale, suffix in [(kernel_scale_lower, 'lower'), 
                                 (kernel_scale_mid, 'mid'), 
                                 (kernel_scale_upper, 'upper')]:
                t_kernel = ref_components.t_kernel / scale if scale != 0 else 0
                results[f't_kernel_{suffix}'].append(t_kernel)
            
            # Calculate communication times
            pcie_scale = scale_calc.pcie_scale()
            nvlink_scale = scale_calc.nvlink_scale()
            
            t_pcie = ref_components.t_pcie / pcie_scale if pcie_scale != 0 else 0
            t_nvlink = ref_components.t_nvlink / nvlink_scale if nvlink_scale != 0 else 0
            
            results['t_pcie'].append(t_pcie)
            results['t_nvlink'].append(t_nvlink)
            
            # Other node time (unchanged)
            t_othernode = ref_components.t_othernode
            results['t_othernode'].append(t_othernode)
            
            # Calculate totals
            results['t_total_lower'].append(results['t_kernel_lower'][-1] + t_pcie + t_nvlink + t_othernode)
            results['t_total_mid'].append(results['t_kernel_mid'][-1] + t_pcie + t_nvlink + t_othernode)
            results['t_total_upper'].append(results['t_kernel_upper'][-1] + t_pcie + t_nvlink + t_othernode)
        
        return results
    
    def _aggregate_multi_gpu(self, all_gpu_metrics: List[Dict[str, List[float]]], 
                            agg_interval_ms: float) -> Dict[str, List[float]]:
        """Aggregate metrics across multiple GPUs"""
        # Check all GPUs have same length
        lengths = [len(m['t_total_lower']) for m in all_gpu_metrics]
        if len(set(lengths)) != 1:
            raise ValueError("Not all GPU metric lists are of the same length!")
        
        num_rows = lengths[0]
        agg_samples = agg_interval_ms // self.sample_interval_ms
        
        # Aggregate by taking max across GPUs for each time window
        aggregated = {key: [] for key in all_gpu_metrics[0].keys()}
        
        for start in range(0, num_rows, agg_samples):
            end = min(start + agg_samples, num_rows)
            
            for key in aggregated.keys():
                # Sum within each GPU's window, then take max across GPUs
                window_values = [
                    sum(gpu_metrics[key][row_idx] for row_idx in range(start, end))
                    for gpu_metrics in all_gpu_metrics
                ]
                aggregated[key].append(max(window_values))
        
        return aggregated
    
    def _calc_est_flops(self, sliced_metrics: Dict[str, List[float]], 
                       tensor_prec: str) -> float:
        """Calculate estimated FLOPS"""
        return (
            np.mean(sliced_metrics.get('tensor_ref')) * self.tgt_gpu.get_specs(tensor_prec) +
            np.mean(sliced_metrics.get('fp64a_ref')) * self.tgt_gpu.get_specs("fp64") +
            np.mean(sliced_metrics.get('fp32a_ref')) * self.tgt_gpu.get_specs("fp32") +
            np.mean(sliced_metrics.get('fp16a_ref')) * self.tgt_gpu.get_specs("fp16")
        )
    
    def _calc_est_mem_bw(self, sliced_metrics: Dict[str, List[float]]) -> float:
        """Calculate estimated memory bandwidth"""
        return np.mean(sliced_metrics.get('drama_ref')) * self.tgt_gpu.get_specs("mem_bw")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Multi-GPU Performance Profiler and Predictor',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('-f', '--dcgm_input', required=True, help='DCGM input: file or folder path')
    parser.add_argument('-n', '--num_gpu', type=int, required=True, help='Number of GPUs')
    parser.add_argument('-d', '--sample_interval_ms', type=int, required=True, help='Sample interval in milliseconds')
    parser.add_argument('-a', '--agg_interval_ms', type=int, required=True, help='Aggregation interval in milliseconds')
    parser.add_argument('-st', '--start_timestamp', type=int, default=0, help='Start timestamp (ms, default: 0)')
    parser.add_argument('-et', '--end_timestamp', type=int, default=None, help='End timestamp (ms, default: None)')
    parser.add_argument('-o', '--overall_runtime_ms', type=int, required=True, help='Overall runtime in milliseconds')
    parser.add_argument('-rg', '--ref_gpu', required=True, choices=list(GPUSpec.keys()), help='Reference GPU')
    parser.add_argument('-tg', '--tgt_gpu', choices=list(GPUSpec.keys()), help='Target GPU (optional)')
    parser.add_argument('--metrics', type=lambda s: s.split(','), required=True, help='Comma-separated list of metrics')
    parser.add_argument('-tp', '--tensor_precision', required=True, choices=['tf64', 'tf32', 'tf16'], help='Tensor precision type')
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    # Process metrics file for multiple jobs
    job_processor = JobProcessor(args.num_gpu, args.metrics)
    gpu_dfs = job_processor.process_files(args.dcgm_input)

    print(f"\nProcessed {len(gpu_dfs)} GPUs")
    for i, df in enumerate(gpu_dfs):
        print(f"GPU {i}: {len(df)} samples")

    # Create and run reference profiler
    ref_profiler = ReferenceProfiler(args.sample_interval_ms, args.ref_gpu)
    ref_profiler.run(
        gpu_dfs, args.metrics, args.overall_runtime_ms, args.agg_interval_ms,
        args.start_timestamp, args.end_timestamp, args.tensor_precision
    )

    # Create target predictor and run if specified
    if args.tgt_gpu:
        tgt_predictor = TargetPredictor(args.sample_interval_ms, args.ref_gpu, args.tgt_gpu)
        tgt_predictor.run(
            gpu_dfs, args.metrics, args.overall_runtime_ms, args.agg_interval_ms,
            args.start_timestamp, args.end_timestamp, args.tensor_precision
        )


if __name__=="__main__":
    main()