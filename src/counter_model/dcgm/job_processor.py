import ast
import json
import os
import re
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from counter_model.hw_config.hw_specs import GPU
from counter_model.hw_config.pm_config import nodes_80gb_set


class SingleJobProcessor:
    """Handles metrics file processing"""

    GPU_PATTERN = re.compile(r"^GPU \d+\s")
    HEADER_PATTERN = re.compile(r"^#Entity")

    def __init__(self, num_gpu: int, metric_names: list[str]):
        self.num_gpu = num_gpu
        self.metric_names = metric_names

    @staticmethod
    def is_float(value: str) -> bool:
        """Check if a string can be converted to float"""
        try:
            float(value)
            return True
        except ValueError:
            return False

    @staticmethod
    def _count_zero(profiled_df: pd.DataFrame):
        # Filter for gract > 0.9
        filtered = profiled_df[profiled_df["GRACT"] > 0.9]
        total_samples = len(filtered)

        # Count zeros for each metric
        tensor_zeros = ((filtered["GRACT"] > 0.9) & (filtered["TENSO"] < 0.01)).sum()
        drama_zeros = ((filtered["GRACT"] > 0.9) & (filtered["DRAMA"] < 0.01)).sum()
        fp64_zeros = ((filtered["GRACT"] >= 0.9) & (filtered["FP64A"] < 0.01)).sum()
        fp32_zeros = ((filtered["GRACT"] >= 0.9) & (filtered["FP32A"] < 0.01)).sum()
        fp16_zeros = ((filtered["GRACT"] >= 0.9) & (filtered["FP16A"] < 0.01)).sum()
        print(
            f"Total Samples: {total_samples}, DRAMA Zero Samples: {drama_zeros}, TENSO Zero Samples: {tensor_zeros}, "
            f"FP64A Zero Samples: {fp64_zeros}, FP32A Zero Samples: {fp32_zeros}, FP16A Zero Samples: {fp16_zeros}"
        )

    def _organize_by_file_content(self, all_files: list[Path]) -> list[str]:
        """Organize files by analyzing their content"""
        file_info = []

        for file_path in all_files:
            try:
                with open(file_path) as f:
                    content = ""
                    for i, line in enumerate(f):
                        content += line
                        if i > 10:
                            break

                gpu_pattern = re.compile(r"GPU (\d+)")
                gpu_matches = gpu_pattern.findall(content)

                if gpu_matches:
                    gpu_counter = Counter(gpu_matches)
                    most_common_gpu_id = int(gpu_counter.most_common(1)[0][0])
                    total_lines = len(gpu_matches)
                    file_info.append((file_path, most_common_gpu_id, total_lines))
                else:
                    print(f"Warning: No GPU data found in {file_path}")
                    file_info.append((file_path, -1, 0))

            except Exception as e:
                print(f"Warning: Could not read file {file_path}: {e}")
                file_info.append((file_path, -1, 0))

        # Sort by filename for deterministic ordering
        file_info.sort(key=lambda x: x[0].name)

        if len(file_info) != self.num_gpu:
            print(
                f"Content-based matching found {len(file_info)} valid files, expected {self.num_gpu}."
            )
            raise ValueError(f"Expected {self.num_gpu} files but found {len(file_info)}")

        # Sort by GPU ID and line count
        file_info.sort(key=lambda x: (x[1], x[2]))

        organized_files = [str(info[0]) for info in file_info]

        print("File organization by content analysis:")
        for logical_gpu_id, (file_path, detected_gpu_id, line_count) in enumerate(file_info):
            if detected_gpu_id >= 0:
                print(
                    f"  Logical GPU {logical_gpu_id}: {file_path.name} "
                    f"(detected GPU {detected_gpu_id}, {line_count} data lines)"
                )
            else:
                print(
                    f"  Logical GPU {logical_gpu_id}: {file_path.name} "
                    f"(GPU ID unknown, {line_count} data lines)"
                )

        return organized_files

    def _scan_and_organize_gpu_files(self, folder_path: str) -> list[str]:
        """Scan a folder for GPU data files and organize them by logical GPU ID"""
        folder_path = Path(folder_path)
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        file_patterns = ["*.out", "*.txt"]
        all_files = []
        for pattern in file_patterns:
            all_files.extend(folder_path.glob(pattern))

        if not all_files:
            raise FileNotFoundError(f"No data files found in {folder_path}")

        print(f"Found {len(all_files)} potential GPU data files in {folder_path}")

        return self._organize_by_file_content(all_files)

    def process_files(self, dcgm_input: str) -> list[pd.DataFrame]:
        """Process input files or directory"""
        if os.path.isdir(dcgm_input):
            print(f"Processing folder: {dcgm_input}")
            file_paths = self._scan_and_organize_gpu_files(dcgm_input)
            return self._process_multiple_files(file_paths)
        elif os.path.isfile(dcgm_input):
            print(f"Processing single file with {self.num_gpu} GPUs: {dcgm_input}")
            return self._process_single_file(dcgm_input)
        else:
            raise ValueError(f"Input path '{dcgm_input}' is neither a valid file nor a directory")

    def _process_multiple_files(self, file_paths: list[str]) -> list[pd.DataFrame]:
        """Process multiple files, each containing single GPU data"""
        profiled_data = list()

        for logical_gpu_id, file_path in enumerate(file_paths):
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            print(f"Processing file {file_path} as logical GPU {logical_gpu_id}")

            with open(file_path) as file:
                lines = file.readlines()

            header_columns, metric_indices = self._parse_header(lines)
            gpu_data = self._extract_single_gpu_data(lines, metric_indices, len(header_columns))

            if gpu_data:
                df = pd.DataFrame(gpu_data, columns=self.metric_names)
                profiled_data.append(df)
                print(f"Logical GPU {logical_gpu_id}: Created DataFrame with {len(gpu_data)} rows")
            else:
                profiled_data.append(pd.DataFrame(columns=self.metric_names))
                print(
                    f"Warning: No data found for logical GPU {logical_gpu_id} in file {file_path}"
                )

        return profiled_data

    def _process_single_file(self, file_path: str) -> list[pd.DataFrame]:
        """Process a single file containing multiple GPU data"""
        with open(file_path) as file:
            lines = file.readlines()

        header_columns, metric_indices = self._parse_header(lines)
        gpu_data = self._extract_gpu_data(lines, metric_indices, len(header_columns))

        # only one gpu so just fetch the first item
        profiled_data = self._create_dataframes(gpu_data)[0]

        # count number of lines with (nearly) "zero" activities
        self._count_zero(profiled_data)

        return profiled_data

    def _extract_gpu_data(
        self, lines: list[str], metric_indices: list[int], num_columns: int
    ) -> dict[int, list[list[float]]]:
        """Extract data from a file with multiple GPUs"""
        gpu_data = {}

        for line in lines:
            if self.HEADER_PATTERN.match(line):
                continue
            if self.GPU_PATTERN.match(line):
                parts = re.split(r"\s{3,}", line.strip())

                # Extract GPU ID
                gpu_match = re.search(r"GPU (\d+)", parts[0])
                if not gpu_match:
                    continue

                gpu_id = int(gpu_match.group(1))

                # Extract numeric values
                values = parts[1:]
                # Creates a list of numeric values, converting 'n/a' strings to 0.0 and other valid numbers to floats
                numeric_values = [
                    0.0 if v.strip().lower() == "n/a" else float(v)
                    for v in values
                    if self.is_float(v) or v.strip().lower() == "n/a"
                ]

                if len(numeric_values) >= num_columns - 1:
                    selected_values = [numeric_values[i] for i in metric_indices]

                    if gpu_id not in gpu_data:
                        gpu_data[gpu_id] = []
                    gpu_data[gpu_id].append(selected_values)
                else:
                    print(f"Warning: Line has insufficient data columns: {line.strip()}")

        return gpu_data

    def _create_dataframes(self, gpu_data: list[list[float]]) -> list[pd.DataFrame]:
        """Create DataFrames from GPU data dictionary"""
        gpu_dfs = []
        for gpu_id in sorted(gpu_data.keys()):
            if gpu_data[gpu_id]:
                df = pd.DataFrame(gpu_data[gpu_id], columns=self.metric_names)
                gpu_dfs.append(df)
            else:
                gpu_dfs.append(pd.DataFrame(columns=self.metric_names))

        return gpu_dfs

    def _parse_header(self, lines: list[str]) -> tuple[list[str], list[int]]:
        """Parse header and find metric column indices"""
        for line in lines:
            if self.HEADER_PATTERN.match(line):
                header_columns = [col.strip() for col in re.split(r"\s{2,}", line.strip())]
                metric_indices = self._get_metric_indices(header_columns)
                return header_columns, metric_indices
        raise ValueError("Could not find header line in the data file")

    def _get_metric_indices(self, header_columns: list[str]) -> list[int]:
        """Map requested metrics to their column indices"""
        metric_indices = []
        for metric in self.metric_names:
            if metric not in header_columns:
                raise ValueError(
                    f"Metric '{metric}' not found in data file. "
                    f"Available metrics: {header_columns[1:]}"
                )
            metric_indices.append(header_columns.index(metric) - 1)
        return metric_indices


class MultiJobProcessor:
    """
    Processes GPU workload data and models performance across different GPU architectures
    """

    # Column name mappings
    COLUMN_RENAME_MAP = {
        "nersc_ldms_dcgm_gr_engine_active": "GRACT",
        "nersc_ldms_dcgm_dram_active": "DRAMA",
        "nersc_ldms_dcgm_sm_occupancy": "SMOCC",
        "nersc_ldms_dcgm_tensor_active": "TENSO",
        "nersc_ldms_dcgm_fp16_active": "FP16A",
        "nersc_ldms_dcgm_fp32_active": "FP32A",
        "nersc_ldms_dcgm_fp64_active": "FP64A",
        "nersc_ldms_dcgm_nvlink_rx_bytes": "NVRX",
        "nersc_ldms_dcgm_nvlink_tx_bytes": "NVTX",
        "nersc_ldms_dcgm_pcie_rx_bytes": "PCRX",
        "nersc_ldms_dcgm_pcie_tx_bytes": "PCTX",
    }

    def __init__(
        self,
        workload_metadata_file: str,
        workload_data_path: str,
        max_workers: int,
        chunk_size: int,
        ref_gpu_name: str,
        tgt_gpu_name: str,
        sampling_interval_ms: int = 1000,
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
        with open(filepath) as file:
            return json.load(file)

    @staticmethod
    def _get_all_parquet_files(job_data_path: str) -> list[str]:
        """
        Recursively find all parquet files in job_data_path and its subdirectories.
        """
        job_path = Path(job_data_path)
        return [str(f) for f in job_path.rglob("*.pq")]

    @staticmethod
    def _get_job_node_hours(job_metadata: dict[str, Any]):
        """
        Get node-hours for a job, 1 node-hour means one hour of continuous operation of a single node.
        """
        return sum(
            (entry["end_ts"] - entry["start_ts"]) * len(entry["nodelist"]) / 3_600_000.0
            for entry in job_metadata["entries"]
        )

    @staticmethod
    def _get_job_overall_runtime(job_metadata: dict[str, Any]):
        runtime = 0
        for entry in job_metadata["entries"]:
            runtime += entry["end_ts"] - entry["start_ts"]
        return runtime // 1000

    def _preprocess_job_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess job data by converting timestamps, renaming columns, and filtering.
        """
        # Convert to 1s and 10s intervals
        df["timestamp_1s"] = df["timestamp"] // 1000
        df["timestamp_10s"] = df["timestamp"] // 10000

        cols_to_drop = [
            "timestamp",
            "nersc_ldms_dcgm_power_usage",
            "nersc_ldms_dcgm_total_energy_consumption",
            "nersc_ldms_dcgm_tensor_hmma_active",
        ]

        # Drop unnecessary columns
        df.drop(cols_to_drop, axis=1, inplace=True, errors="ignore")

        # Rename columns for brevity
        df.rename(columns=self.COLUMN_RENAME_MAP, inplace=True)

        # Filter out metric values that don't fall in [0,1]
        metric_columns = ["FP64A", "FP32A", "FP16A", "TENSO", "DRAMA", "GRACT"]
        filter_mask = pd.Series(True, index=df.index)

        for col in metric_columns:
            if col in df.columns:
                filter_mask &= df[col].between(0, 1)

        filtered_df = df.loc[filter_mask].copy()

        # Calculate combined metrics
        if "PCRX" in filtered_df.columns and "PCTX" in filtered_df.columns:
            filtered_df["PCIE_TRX"] = filtered_df["PCRX"] + filtered_df["PCTX"]

        if "NVRX" in filtered_df.columns and "NVTX" in filtered_df.columns:
            filtered_df["NVLINK_TRX"] = filtered_df["NVRX"] + filtered_df["NVTX"]

        return filtered_df

    def process_workload_metadata(self):
        """Identify non-interactive jobs running on A100-40GB GPUs."""

        self.workload_metadata = pd.read_csv(self.workload_metadata_file)

        # Identify 40GB nodes (not in 80GB set)
        nodes_40gb_mask = self.workload_metadata["nodes"].apply(
            lambda x: ast.literal_eval(x)[0] not in nodes_80gb_set
        )

        # Identify non-interactive jobs
        non_interactive_mask = self.workload_metadata["QOS"].isin(["gpu_regular", "gpu_premium"])

        # Combine filters and extract job identifiers
        self.workload_set = set(
            self.workload_metadata.loc[
                non_interactive_mask & nodes_40gb_mask, ["JobID", "User"]
            ].itertuples(index=False, name=None)
        )

        print(f"The number of non-interactive jobs using A100-40GB: {len(self.workload_set)}")

    def _perf_model_per_job(
        self, job_pq_df, jobid_userid: str, job_total_runtime: float, job_node_hours: float
    ):
        rg = self.ref_gpu
        tg = self.tgt_gpu
        mv_ref = self._preprocess_job_data(job_pq_df, self.ref_gpu)

        # Calculate occupancy on reference GPU
        mv_ref["theta_ref"] = (mv_ref["SMOCC"] / mv_ref["GRACT"]) * rg.get_specs("max_warps_sm")

        # Calculate both resource-based occupancy estimates
        occ_reg = mv_ref["theta_ref"] * (tg.get_specs("reg_size_sm") / rg.get_specs("reg_size_sm"))
        occ_shmem = mv_ref["theta_ref"] * (tg.get_specs("shmem_sm") / rg.get_specs("shmem_sm"))

        # Estimate occupancy on target GPU for lower, middle, upper
        smocc_tgt_lower = np.minimum(np.minimum(occ_reg, occ_shmem), tg.get_specs("max_warps_sm"))
        smocc_tgt_middle = np.minimum((occ_reg + occ_shmem) * 0.5, tg.get_specs("max_warps_sm"))
        smocc_tgt_upper = np.minimum(np.minimum(occ_reg, occ_shmem), tg.get_specs("max_warps_sm"))

        # Store all three variants
        smocc_tgt_dict = {
            "lower": smocc_tgt_lower,
            "middle": smocc_tgt_middle,
            "upper": smocc_tgt_upper,
        }

        # Calculate kernel and other runtime on reference GPU
        intv = self.sampling_interval_ms // 1000
        t_kernel_ref = intv * mv_ref["GRACT"]
        t_other_ref = intv * (1 - mv_ref["GRACT"])

        # Use other runtime on reference gpu as other runtime on target gpu
        t_other_tgt = t_other_ref

        # Get FP activities - vectorized threshold operation
        fp_cols = ["FP16A", "FP32A", "FP64A"]
        mv_ref[fp_cols] = mv_ref[fp_cols].where(mv_ref[fp_cols] >= 0.01, 0)

        fp_total = mv_ref[fp_cols].sum(axis=1)

        # Compute weights and tensor values
        mask = fp_total != 0
        weights = mv_ref[fp_cols].div(fp_total, axis=0).where(mask, 1 / 3)

        tensor_ref = sum(
            weights.iloc[:, i] * rg.get_specs(f"tf{precision}")
            for i, precision in enumerate([16, 32, 64])
        )
        tensor_tgt = sum(
            weights.iloc[:, i] * tg.get_specs(f"tf{precision}")
            for i, precision in enumerate([16, 32, 64])
        )
        tensor_ratio = tensor_tgt / tensor_ref

        # Pre-compute spec ratios
        spec_ratios = {
            "mem_bw": tg.get_specs("mem_bw") / rg.get_specs("mem_bw"),
            "fp64": tg.get_specs("fp64") / rg.get_specs("fp64"),
            "fp32": tg.get_specs("fp32") / rg.get_specs("fp32"),
            "fp16": tg.get_specs("fp16") / rg.get_specs("fp16"),
        }

        # Pre-compute common values
        k_smocc_base_tgt = tg.get_specs("num_sm") * tg.get_specs("boost_clock")
        k_smocc_base_ref = (
            rg.get_specs("num_sm") * rg.get_specs("boost_clock") * mv_ref["theta_ref"]
        )
        gract_safe = mv_ref["GRACT"].replace(0, np.nan)  # Avoid division by zero

        time_distribution_per_job = dict()
        # Calculate all SMOCC lower, middle, upper
        for _, smocc_tgt in smocc_tgt_dict.items():
            # Calculate k_smocc scaling factor
            k_smocc_tgt = k_smocc_base_tgt * smocc_tgt
            k_smocc_ratio = np.where(k_smocc_base_ref == 0, 0, k_smocc_tgt / k_smocc_base_ref)

            # Pre-compute ratios for each metric
            ratios = {
                "dram": spec_ratios["mem_bw"] / (mv_ref["DRAMA"] / gract_safe),
                "tensor": tensor_ratio / (mv_ref["TENSOR"] / gract_safe),
                "fp64": spec_ratios["fp64"] / (mv_ref["FP64"] / gract_safe),
                "fp32": spec_ratios["fp32"] / (mv_ref["FP32"] / gract_safe),
                "fp16": spec_ratios["fp16"] / (mv_ref["FP16"] / gract_safe),
            }

            # Replace NaN with 0 and compute minimum with k_smocc_ratio
            k_dram_ratio = np.minimum(k_smocc_ratio, ratios["dram"].fillna(0))
            k_tensor_ratio = np.minimum(k_smocc_ratio, ratios["tensor"].fillna(0))
            k_fp64_ratio = np.minimum(k_smocc_ratio, ratios["fp64"].fillna(0))
            k_fp32_ratio = np.minimum(k_smocc_ratio, ratios["fp32"].fillna(0))
            k_fp16_ratio = np.minimum(k_smocc_ratio, ratios["fp16"].fillna(0))

            t_kernel_tgt = t_kernel_ref / np.minimum(
                k_dram_ratio, k_tensor_ratio, k_fp64_ratio, k_fp32_ratio, k_fp16_ratio
            )

        job_time_distribution_tgt_gpu = {
            "jobid_userid": jobid_userid,
            "total_measured_runtime": job_total_runtime,
            "total_node_hours_job": job_node_hours,
            f"kernel_time_{tg.get_name()}": t_kernel_tgt.sum(axis=1),
            f"othernode_time_{tg.get_name()}": t_other_tgt.sum(axis=1),
        }

        time_distribution_per_job[tg.get_name()] = job_time_distribution_tgt_gpu

        return time_distribution_per_job

    def _process_workload(self, chunk: list[str], chunk_idx: int) -> dict:
        """Worker function to process a subset of parquet files"""
        local_summaries = dict()

        for pq_file in chunk:
            jobid, userid = Path(pq_file).stem.rsplit("_", 1)
            if (jobid, userid) not in self.workload_set:
                continue

            jobid_userid = f"{jobid}_{userid}"
            job_metadata_file = pq_file.replace(".pq", ".json")
            job_metadata = self._read_metadata_file_json(job_metadata_file)

            job_pq_df = pd.read_parquet(pq_file, engine="pyarrow")

            time_distribution_per_job = self._perf_model_per_job(
                job_pq_df,
                jobid_userid,
                self._get_job_overall_runtime(job_metadata),
                self._get_node_hours_job(job_metadata),
                self.ref_gpu,
                self.tgt_gpu,
            )

            local_summaries[jobid_userid] = time_distribution_per_job

        print(f"Chunk {chunk_idx} has been processed")
        return local_summaries

    def run(self, parallel_process: bool = True):
        """
        Execute the workload processing pipeline.
        """
        job_file_list = self._get_all_parquet_files(self.workload_data_path)
        print(
            f"{self.workload_data_path} has {len(job_file_list)} parquet files (including subdirectories)"
        )

        if not job_file_list:
            print("No parquet files found. Exiting...")
            return

        # Split files into chunks for parallel processing
        job_file_chunks = [
            job_file_list[i : i + self.chunk_size]
            for i in range(0, len(job_file_list), self.chunk_size)
        ]
        print(f"Processing with {self.max_workers} workers, each has {self.chunk_size} files")

        global_output = {}
        if parallel_process:
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [
                    executor.submit(self._process_workload, chunk, chunk_idx)
                    for chunk_idx, chunk in enumerate(job_file_chunks)
                ]

            # Collect results from all processes
            for future in enumerate(futures):
                try:
                    local_results = future.result()
                    global_output.update(local_results)

                except Exception as e:
                    print(f"Error processing chunk: {e}")
                    traceback.print_exc()

        else:
            self._process_workload(job_file_list, 1)
