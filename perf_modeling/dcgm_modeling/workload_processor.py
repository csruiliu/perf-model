import pandas as pd
import numpy as np
import json
import ast
from pathlib import Path
from typing import Dict, Tuple, Optional, Set, List, Any
from hw_specs import GPU
import traceback

from concurrent.futures import ProcessPoolExecutor

from pm_node_config import nodes_80gb_set


class WorkloadProcessor:
    """
    Processes GPU workload data and models performance across different GPU architectures
    """
    
    # Column name mappings
    COLUMN_RENAME_MAP = {
        'nersc_ldms_dcgm_gr_engine_active': 'GRACT',
        'nersc_ldms_dcgm_dram_active': 'DRAMA',
        'nersc_ldms_dcgm_sm_occupancy': 'SMOCC',
        'nersc_ldms_dcgm_tensor_active': 'TENSO',
        'nersc_ldms_dcgm_fp16_active': 'FP16A',
        'nersc_ldms_dcgm_fp32_active': 'FP32A',
        'nersc_ldms_dcgm_fp64_active': 'FP64A',
        'nersc_ldms_dcgm_nvlink_rx_bytes': 'NVRX',
        'nersc_ldms_dcgm_nvlink_tx_bytes': 'NVTX',
        'nersc_ldms_dcgm_pcie_rx_bytes': 'PCRX',
        'nersc_ldms_dcgm_pcie_tx_bytes': 'PCTX'
    }

    def __init__(
        self, 
        workload_metadata_file: str, 
        workload_data_path: str, 
        max_workers: int, 
        chunk_size: int,  
        ref_gpu_name: str,
        tgt_gpu_name: str,
        sampling_interval_ms: int = 1000
    ):
        """
        Initialize the WorkloadProcessor.
        
        Args:
            workload_metadata_file: Path to workload metadata CSV file
            workload_data_path: Path to directory containing parquet files
            max_workers: Maximum number of parallel workers
            chunk_size: Number of files to process per chunk
            ref_gpu_name: Reference GPU name 
            tgt_gpu_name: Target GPU name
            sampling_interval: Optional sampling interval for data processing
        """
        self.max_workers = max_workers
        self.chunk_size = chunk_size
        self.sampling_interval_ms = sampling_interval_ms
        self.workload_metadata_file = workload_metadata_file
        self.workload_data_path = workload_data_path
        
        # initlize GPUs
        self.ref_gpu = GPU(gpu_name=ref_gpu_name) 
        self.tgt_gpu = GPU(gpu_name=tgt_gpu_name) 

        # Placeholder for workload
        self.workload_metadata = None
        self.workload_set = None


    @staticmethod
    def _read_metadata_file_json(filepath: str) -> dict:
        """Read and parse a JSON metadata file."""
        with open(filepath, 'r') as file:
            return json.load(file)


    @staticmethod
    def _get_all_parquet_files(job_data_path: str) -> List[str]:
        """
        Recursively find all parquet files in job_data_path and its subdirectories.
        """
        job_path = Path(job_data_path)
        return [str(f) for f in job_path.rglob("*.pq")]


    @staticmethod
    def _get_job_node_hours(job_metadata: Dict[str, Any]):
        """
        Get node-hours for a job, 1 node-hour means one hour of continuous operation of a single node.
        """
        return sum(
            (entry['end_ts'] - entry['start_ts']) * len(entry['nodelist']) / 3_600_000.0
            for entry in job_metadata['entries']
        )


    @staticmethod
    def _get_job_overall_runtime(job_metadata: Dict[str, Any]):
        runtime = 0
        for entry in job_metadata['entries']:
            runtime += entry['end_ts'] - entry['start_ts']
        return runtime // 1000


    def _preprocess_job_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess job data by converting timestamps, renaming columns, and filtering.
        """
        # Convert to 1s and 10s intervals
        df['timestamp_1s'] = df['timestamp'] // 1000
        df['timestamp_10s'] = df['timestamp'] // 10000
        
        COLUMNS_TO_DROP = [
            'timestamp',
            'nersc_ldms_dcgm_power_usage',
            'nersc_ldms_dcgm_total_energy_consumption',
            'nersc_ldms_dcgm_tensor_hmma_active'
        ]

        # Drop unnecessary columns
        df.drop(COLUMNS_TO_DROP, axis=1, inplace=True, errors='ignore')

        # Rename columns for brevity
        df.rename(columns=self.COLUMN_RENAME_MAP, inplace=True)

        # Filter out metric values that don't fall in [0,1]
        metric_columns = ['FP64A', 'FP32A', 'FP16A', 'TENSO', 'DRAMA', 'GRACT']
        filter_mask = pd.Series(True, index=df.index)
        
        for col in metric_columns:
            if col in df.columns:
                filter_mask &= df[col].between(0, 1)
        
        filtered_df = df.loc[filter_mask].copy()
        
        # Calculate combined metrics
        if 'PCRX' in filtered_df.columns and 'PCTX' in filtered_df.columns:
            filtered_df['PCIE_TRX'] = filtered_df['PCRX'] + filtered_df['PCTX']
        
        if 'NVRX' in filtered_df.columns and 'NVTX' in filtered_df.columns:
            filtered_df['NVLINK_TRX'] = filtered_df['NVRX'] + filtered_df['NVTX']
        
        return filtered_df


    def process_workload_metadata(self):
        """Identify non-interactive jobs running on A100-40GB GPUs."""
        
        self.workload_metadata = pd.read_csv(self.workload_metadata_file)
        
        # Identify 40GB nodes (not in 80GB set)
        nodes_40gb_mask = self.workload_metadata['nodes'].apply(
            lambda x: ast.literal_eval(x)[0] not in nodes_80gb_set
        )
        
        # Identify non-interactive jobs
        non_interactive_mask = self.workload_metadata['QOS'].isin(
            ["gpu_regular", "gpu_premium"]
        )

        # Combine filters and extract job identifiers
        self.workload_set = set(
            self.workload_metadata.loc[
                non_interactive_mask & nodes_40gb_mask,
                ['JobID', 'User']
            ].itertuples(index=False, name=None)
        )

        print(f"The number of non-interactive jobs using A100-40GB: {len(self.workload_set)}")


    def _perf_model_per_job(self, 
                            job_pq_df, 
                            jobid_userid: str, 
                            job_total_runtime: float, 
                            job_node_hours: float):
        rg = self.ref_gpu
        tg = self.tgt_gpu
        mv_ref = self._preprocess_job_data(job_pq_df, self.ref_gpu)

        # Calculate occupancy on reference GPU
        mv_ref['theta_ref'] = (mv_ref['SMOCC'] / mv_ref['GRACT']) * rg.get_specs("max_warps_sm")

        # Calculate both resource-based occupancy estimates
        occ_reg = mv_ref['theta_ref'] * (tg.get_specs("reg_size_sm") / rg.get_specs("reg_size_sm"))
        occ_shmem = mv_ref['theta_ref'] * (tg.get_specs("shmem_sm") / rg.get_specs("shmem_sm"))

        # Estimate occupancy on target GPU for lower, middle, upper
        smocc_tgt_lower = np.minimum(
            np.minimum(occ_reg, occ_shmem),
            tg.get_specs("max_warps_sm")
        )
        smocc_tgt_middle = np.minimum(
            (occ_reg + occ_shmem) * 0.5,
            tg.get_specs("max_warps_sm")
        )
        smocc_tgt_upper = np.minimum(
            np.minimum(occ_reg, occ_shmem),
            tg.get_specs("max_warps_sm")
        )

        # Store all three variants
        smocc_tgt = {
            'lower': smocc_tgt_lower,
            'middle': smocc_tgt_middle,
            'upper': smocc_tgt_upper
        }

        # Calculate kernel and other runtime on reference GPU 
        intv = self.sampling_interval_ms // 1000
        t_kernel_ref = intv * mv_ref['GRACT']
        t_other_ref = intv * (1 - mv_ref['GRACT'])

        # Use other runtime on reference gpu as other runtime on target gpu
        t_other_tgt = t_other_ref

        # Get FP activities - vectorized threshold operation
        fp_cols = ['FP16A', 'FP32A', 'FP64A']
        mv_ref[fp_cols] = mv_ref[fp_cols].where(mv_ref[fp_cols] >= 0.01, 0)

        fp_total = mv_ref[fp_cols].sum(axis=1)

        # Compute weights and tensor values
        mask = fp_total != 0
        weights = mv_ref[fp_cols].div(fp_total, axis=0).where(mask, 1/3)

        tensor_ref = sum(weights.iloc[:, i] * rg.get_specs(f"tf{precision}") 
                         for i, precision in enumerate([16, 32, 64]))
        tensor_tgt = sum(weights.iloc[:, i] * tg.get_specs(f"tf{precision}") 
                         for i, precision in enumerate([16, 32, 64]))
        tensor_ratio = tensor_tgt / tensor_ref

        # Pre-compute spec ratios 
        spec_ratios = {
            'mem_bw': tg.get_specs("mem_bw") / rg.get_specs("mem_bw"),
            'fp64': tg.get_specs("fp64") / rg.get_specs("fp64"),
            'fp32': tg.get_specs("fp32") / rg.get_specs("fp32"),
            'fp16': tg.get_specs("fp16") / rg.get_specs("fp16")
        }

        # Pre-compute common values
        k_smocc_base_tgt = tg.get_specs("num_sm") * tg.get_specs("boost_clock")
        k_smocc_base_ref = rg.get_specs("num_sm") * rg.get_specs("boost_clock") * mv_ref['theta_ref']
        gract_safe = mv_ref['GRACT'].replace(0, np.nan)  # Avoid division by zero
        
        time_distribution_per_job = dict()
        # Calculate all SMOCC lower, middle, upper
        for smocc_name, smocc_tgt in smocc_tgt.items():
            # Calculate k_smocc scaling factor
            k_smocc_tgt = k_smocc_base_tgt * smocc_tgt
            k_smocc_ratio = np.where(k_smocc_base_ref == 0, 0, k_smocc_tgt / k_smocc_base_ref)

            # Pre-compute ratios for each metric
            ratios = {
                'dram': spec_ratios['mem_bw'] / (mv_ref['DRAMA'] / gract_safe),
                'tensor': tensor_ratio / (mv_ref['TENSOR'] / gract_safe),
                'fp64': spec_ratios['fp64'] / (mv_ref['FP64'] / gract_safe),
                'fp32': spec_ratios['fp32'] / (mv_ref['FP32'] / gract_safe),
                'fp16': spec_ratios['fp16'] / (mv_ref['FP16'] / gract_safe)
            }

            # Replace NaN with 0 and compute minimum with k_smocc_ratio
            k_dram_ratio = np.minimum(k_smocc_ratio, ratios['dram'].fillna(0))
            k_tensor_ratio = np.minimum(k_smocc_ratio, ratios['tensor'].fillna(0))
            k_fp64_ratio = np.minimum(k_smocc_ratio, ratios['fp64'].fillna(0))
            k_fp32_ratio = np.minimum(k_smocc_ratio, ratios['fp32'].fillna(0))
            k_fp16_ratio = np.minimum(k_smocc_ratio, ratios['fp16'].fillna(0))

            t_kernel_tgt = t_kernel_ref / np.minimum(k_dram_ratio, k_tensor_ratio, k_fp64_ratio, k_fp32_ratio, k_fp16_ratio)


        job_time_distribution_tgt_gpu = {
            'jobid_userid': jobid_userid, 
            'total_measured_runtime':job_total_runtime,
            'total_node_hours_job':job_node_hours,
            f'kernel_time_{tg.get_name()}': t_kernel_tgt.sum(axis=1),
            f'othernode_time_{tg.get_name()}': t_other_tgt.sum(axis=1)
        }

        time_distribution_per_job[tg.get_name()] = job_time_distribution_tgt_gpu

        return time_distribution_per_job


    def _process_workload(self, chunk: List[str], chunk_idx: int) -> Dict:
        """Worker function to process a subset of parquet files"""
        local_summaries = dict()

        for pq_file in chunk:
            jobid, userid = Path(pq_file).stem.rsplit('_', 1)
            if (jobid, userid) not in self.workload_set:
                continue
            
            jobid_userid = f"{jobid}_{userid}"
            job_metadata_file = pq_file.replace(".pq", ".json")
            job_metadata = self._read_metadata_file_json(job_metadata_file)

            job_pq_df = pd.read_parquet(pq_file, engine='pyarrow')

            time_distribution_per_job = self._perf_model_per_job(
                job_pq_df, 
                jobid_userid, 
                self._get_job_overall_runtime(job_metadata), 
                self._get_node_hours_job(job_metadata), 
                self.ref_gpu, 
                self.tgt_gpu
            )

            local_summaries[jobid_userid] = time_distribution_per_job

        print(f"Chunk {chunk_idx} has been processed")
        return local_summaries


    def run(self, parallel_process: bool = True):
        """
        Execute the workload processing pipeline.
        """
        job_file_list = self._get_all_parquet_files(self.workload_data_path)
        print(f"{self.workload_data_path} has {len(job_file_list)} parquet files (including subdirectories)")

        if not job_file_list:
            print("No parquet files found. Exiting...")
            return

        # Split files into chunks for parallel processing
        job_file_chunks = [
            job_file_list[i:i + self.chunk_size] 
            for i in range(0, len(job_file_list), self.chunk_size)
        ]
        print(f"Processing with {self.max_workers} workers, each has {self.chunk_size} files")

        global_output = {}
        if parallel_process:
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [
                    executor.submit(
                        self._process_workload, 
                        chunk, 
                        chunk_idx
                    )
                    for chunk_idx, chunk in enumerate(job_file_chunks)
                ]

            # Collect results from all processes
            for future in enumerate(futures):
                try:
                    local_results = future.result()
                    global_output.update()

                except Exception as e:
                    print(f"Error processing chunk: {e}")
                    traceback.print_exc()
        
        else:
            self._process_workload(job_file_list, 1)
        