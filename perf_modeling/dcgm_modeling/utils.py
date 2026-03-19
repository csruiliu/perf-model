from typing import Dict, List


class ResultsFormatter:
    """Formats and prints results"""
    
    @staticmethod
    def print_reference_results(metrics: Dict[str, List[float]], flops: float, mem_bw: float, gpu_name: str):
        """Print reference hardware results"""
        print(f"\n{'='*60}")
        print(f"Reference Hardware: {gpu_name}\n")
        print(f"Estimated TFLOPS: {flops:.2f}")
        print(f"Estimated GPU Memory Bandwidth: {mem_bw:.2f} GB/s")
        
        print(f"\nEstimated Kernel Time: {sum(metrics['t_kernel']):.2f} s")
        print(f"Estimated Other Node Time: {sum(metrics['t_othernode']):.2f} s")
        print(f"Estimated Total Runtime: {sum(metrics['t_total']):.2f} s")
        print(f"{'='*60}\n")
    
    @staticmethod
    def print_target_results(metrics: Dict[str, List[float]], gpu_name: str):
        """Print target hardware results"""
        print(f"\n{'='*60}")
        print(f"Target Hardware: {gpu_name}")
        
        #print(f'Estimated TFLOPS [Lower SMOCC]: {flops.get("flop_smocc_lower"):.2f} GB/s')
        #print(f'Estimated TFLOPS [Mid SMOCC]: {flops.get("flop_smocc_mid"):.2f} GB/s')
        #print(f'Estimated TFLOPS [Upper SMOCC]: {flops.get("flop_smocc_upper"):.2f} GB/s')
        #print(f'Estimated TFLOPS [Mock SMOCC]: {flops.get("flop_smocc_mock"):.2f} GB/s')

        #print(f'Estimated GPU Memory Bandwidth [Lower SMOCC]: {mem_bw.get("dram_smocc_lower"):.2f} GB/s')
        #print(f'Estimated GPU Memory Bandwidth [Mid SMOCC]: {mem_bw.get("dram_smocc_mid"):.2f} GB/s')
        #print(f'Estimated GPU Memory Bandwidth [Upper SMOCC]: {mem_bw.get("dram_smocc_upper"):.2f} GB/s')
        #print(f'Estimated GPU Memory Bandwidth [Mock SMOCC]: {mem_bw.get("dram_smocc_mock"):.2f} GB/s')

        print(f"\nEstimated Kernel Time [Lower SMOCC]: {sum(metrics['t_kernel_lower']):.2f} s")
        print(f"Estimated Kernel Time [Mid SMOCC]:   {sum(metrics['t_kernel_mid']):.2f} s")
        print(f"Estimated Kernel Time [Upper SMOCC]: {sum(metrics['t_kernel_upper']):.2f} s")
        print(f"Estimated Kernel Time [Mock SMOCC]: {sum(metrics['t_kernel_mock']):.2f} s")
        
        print(f"\nEstimated Other Node Time: {sum(metrics['t_othernode']):.2f} s")
        
        print(f"\nEstimated Total Runtime [Lower SMOCC]: {sum(metrics['t_total_lower']):.2f} s")
        print(f"Estimated Total Runtime [Mid SMOCC]:   {sum(metrics['t_total_mid']):.2f} s")
        print(f"Estimated Total Runtime [Upper SMOCC]: {sum(metrics['t_total_upper']):.2f} s")
        print(f"Estimated Total Runtime [Mock SMOCC]: {sum(metrics['t_total_mock']):.2f} s")
        print(f"{'='*60}\n")