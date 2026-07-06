import pickle
import re
from pathlib import Path

import pandas as pd


class JobParser:
    """Handles DCGM metrics file processing for single-job and multi-job modes."""

    GPU_PATTERN = re.compile(r"^GPU \d+\s")
    HEADER_PATTERN = re.compile(r"^#Entity")

    # Column name mappings for multi-job (PKL) DCGM data.
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

    def __init__(self, dcgm_file: str, metric_names: list[str]):
        self.dcgm_file = dcgm_file
        self.metric_names = metric_names

    # ------------------------------------------------------------------ #
    # Single-job: read a single DCGM file
    #   - 1 GPU        -> DataFrame
    #   - multiple GPU -> {gpu_id: DataFrame}
    # ------------------------------------------------------------------ #
    def parsing_single_job(self, num_gpu) -> pd.DataFrame:
        """Read a single DCGM file and return a DataFrame of the requested metrics."""
        print(f"Processing single DCGM file (expecting {num_gpu} GPU(s)): {self.dcgm_file}")

        with open(self.dcgm_file) as file:
            lines = file.readlines()

        header_columns, metric_indices = self._parse_header(lines)
        gpu_data = self._extract_gpu_data(lines, metric_indices, len(header_columns))

        if not gpu_data:
            raise ValueError(f"No GPU data found in file: {self.dcgm_file}")

        if len(gpu_data) != num_gpu:
            raise ValueError(
                f"Expected {num_gpu} GPU(s) in {self.dcgm_file}, "
                f"but found {len(gpu_data)}: GPU IDs {sorted(gpu_data.keys())}"
            )

        gpu_dfs = {
            gpu_id: pd.DataFrame(gpu_data[gpu_id], columns=self.metric_names)
            for gpu_id in sorted(gpu_data.keys())
        }

        # Single-GPU path -> return one DataFrame.
        if num_gpu == 1:
            single_df = next(iter(gpu_dfs.values()))
            print(f"Single GPU: {len(single_df)} rows")
            self._count_zero(single_df)
            return single_df

        # Multi-GPU path -> return {gpu_id: DataFrame}.
        for gpu_id, df in gpu_dfs.items():
            print(f"GPU {gpu_id}: {len(df)} rows")
            self._count_zero(df)
        return gpu_dfs

    # ------------------------------------------------------------------ #
    # Multi-job: read a PKL file -> {job_id: DataFrame}
    # ------------------------------------------------------------------ #
    def parsing_multi_job(self) -> dict[str, pd.DataFrame]:
        """Reads a PKL file containing {job_id: DataFrame}."""
        path = Path(self.dcgm_file)
        if not path.is_file():
            raise FileNotFoundError(f"PKL file not found: {self.dcgm_file}")
        if path.suffix.lower() != ".pkl":
            raise ValueError(f"Expected a .pkl file, got: {self.dcgm_file}")

        with open(path, "rb") as f:
            data = pickle.load(f)

        if not isinstance(data, dict):
            raise ValueError(
                f"PKL file must contain a dict of {{job_id: DataFrame}}, got {type(data).__name__}"
            )

        processed: dict[str, pd.DataFrame] = {}
        for job_id, df in data.items():
            if not isinstance(df, pd.DataFrame):
                raise ValueError(
                    f"Value for job '{job_id}' must be a DataFrame, got {type(df).__name__}"
                )
            processed[job_id] = self._preprocess_job_data(df)

        print(f"Loaded {len(processed)} jobs from {self.dcgm_file}")
        return processed

    # ------------------------------------------------------------------ #
    # Single-job helpers (raw DCGM text parsing)
    # ------------------------------------------------------------------ #
    @staticmethod
    def is_float(value: str) -> bool:
        """Check if a string can be converted to float."""
        try:
            float(value)
            return True
        except ValueError:
            return False

    def _extract_gpu_data(
        self, lines: list[str], metric_indices: list[int], num_columns: int
    ) -> dict[int, list[list[float]]]:
        """Extract per-GPU numeric data from a DCGM file."""
        gpu_data: dict[int, list[list[float]]] = {}

        for line in lines:
            if self.HEADER_PATTERN.match(line):
                continue
            if not self.GPU_PATTERN.match(line):
                continue

            parts = re.split(r"\s{3,}", line.strip())

            gpu_match = re.search(r"GPU (\d+)", parts[0])
            if not gpu_match:
                continue
            gpu_id = int(gpu_match.group(1))

            values = parts[1:]
            numeric_values = [
                0.0 if v.strip().lower() == "n/a" else float(v)
                for v in values
                if self.is_float(v) or v.strip().lower() == "n/a"
            ]

            if len(numeric_values) >= num_columns - 1:
                selected_values = [numeric_values[i] for i in metric_indices]
                gpu_data.setdefault(gpu_id, []).append(selected_values)
            else:
                print(f"Warning: Line has insufficient data columns: {line.strip()}")

        return gpu_data

    def _create_dataframes(self, gpu_data: dict[int, list[list[float]]]) -> list[pd.DataFrame]:
        """Create a list of DataFrames (one per GPU) from the extracted data dict."""
        gpu_dfs = []
        for gpu_id in sorted(gpu_data.keys()):
            if gpu_data[gpu_id]:
                gpu_dfs.append(pd.DataFrame(gpu_data[gpu_id], columns=self.metric_names))
            else:
                gpu_dfs.append(pd.DataFrame(columns=self.metric_names))
        return gpu_dfs

    def _parse_header(self, lines: list[str]) -> tuple[list[str], list[int]]:
        """Parse the header line and find metric column indices."""
        for line in lines:
            if self.HEADER_PATTERN.match(line):
                header_columns = [col.strip() for col in re.split(r"\s{2,}", line.strip())]
                metric_indices = self._get_metric_indices(header_columns)
                return header_columns, metric_indices
        raise ValueError("Could not find header line in the data file")

    def _get_metric_indices(self, header_columns: list[str]) -> list[int]:
        """Map requested metrics to their column indices."""
        metric_indices = []
        for metric in self.metric_names:
            if metric not in header_columns:
                raise ValueError(
                    f"Metric '{metric}' not found in data file. "
                    f"Available metrics: {header_columns[1:]}"
                )
            metric_indices.append(header_columns.index(metric) - 1)
        return metric_indices

    @staticmethod
    def _count_zero(profiled_df: pd.DataFrame) -> None:
        """Report samples with (nearly) zero activity while GRACT is high."""
        required = {"GRACT", "TENSO", "DRAMA", "FP64A", "FP32A", "FP16A"}
        if not required.issubset(profiled_df.columns):
            return

        filtered = profiled_df[profiled_df["GRACT"] > 0.9]
        total_samples = len(filtered)

        tensor_zeros = (filtered["TENSO"] < 0.01).sum()
        drama_zeros = (filtered["DRAMA"] < 0.01).sum()
        fp64_zeros = (filtered["FP64A"] < 0.01).sum()
        fp32_zeros = (filtered["FP32A"] < 0.01).sum()
        fp16_zeros = (filtered["FP16A"] < 0.01).sum()

        print(
            f"Total Samples: {total_samples}, DRAMA Zero Samples: {drama_zeros}, "
            f"TENSO Zero Samples: {tensor_zeros}, FP64A Zero Samples: {fp64_zeros}, "
            f"FP32A Zero Samples: {fp32_zeros}, FP16A Zero Samples: {fp16_zeros}"
        )

    # ------------------------------------------------------------------ #
    # Multi-job helper (PKL DataFrame preprocessing)
    # ------------------------------------------------------------------ #
    def _preprocess_job_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess a job's DCGM DataFrame from the PKL file."""
        df = df.copy()

        if "timestamp" in df.columns:
            df["timestamp_1s"] = df["timestamp"] // 1000
            df["timestamp_10s"] = df["timestamp"] // 10000

        cols_to_drop = [
            "timestamp",
            "nersc_ldms_dcgm_power_usage",
            "nersc_ldms_dcgm_total_energy_consumption",
            "nersc_ldms_dcgm_tensor_hmma_active",
        ]
        df.drop(cols_to_drop, axis=1, inplace=True, errors="ignore")

        df.rename(columns=self.COLUMN_RENAME_MAP, inplace=True)

        metric_columns = ["FP64A", "FP32A", "FP16A", "TENSO", "DRAMA", "GRACT"]
        filter_mask = pd.Series(True, index=df.index)
        for col in metric_columns:
            if col in df.columns:
                filter_mask &= df[col].between(0, 1)

        filtered_df = df.loc[filter_mask].copy()

        if "PCRX" in filtered_df.columns and "PCTX" in filtered_df.columns:
            filtered_df["PCIE_TRX"] = filtered_df["PCRX"] + filtered_df["PCTX"]
        if "NVRX" in filtered_df.columns and "NVTX" in filtered_df.columns:
            filtered_df["NVLINK_TRX"] = filtered_df["NVRX"] + filtered_df["NVTX"]

        return filtered_df
