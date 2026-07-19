import pickle
import re

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
        "nersc_ldms_dcgm_nvlink_rx_bytes": "NVLRX",
        "nersc_ldms_dcgm_nvlink_tx_bytes": "NVLTX",
        "nersc_ldms_dcgm_pcie_rx_bytes": "PCIRX",
        "nersc_ldms_dcgm_pcie_tx_bytes": "PCITX",
    }

    # Column used to decide whether a GPU was actively used (long/raw name).
    ACTIVITY_METRIC = "nersc_ldms_dcgm_gr_engine_active"

    # A GPU counts as "active" if at least MIN_ACTIVE_FRACTION of its samples
    # have an activity value of at least MIN_ACTIVE_VALUE.
    MIN_ACTIVE_FRACTION = 0.1
    MIN_ACTIVE_VALUE = 0.1

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
    # Multi-job: read a PKL file -> {job_key: DataFrame}
    # ------------------------------------------------------------------ #
    def parsing_multi_job(self, num_gpu: int = 1) -> dict[str, pd.DataFrame]:
        """Read a PKL file containing {job_id: dcgm_df} and return a dict of
        cleaned/renamed DataFrames ready for the profiler/estimator.

        Each raw dcgm_df contains data for 4 GPUs. We treat every GPU as an
        individual single-GPU job: each raw dcgm_df is split into up to four
        per-GPU DataFrames, and only GPUs that were actively used are kept.

        A GPU is considered active when at least ``MIN_ACTIVE_FRACTION`` of its
        samples have a ``ACTIVITY_METRIC`` value of at least ``MIN_ACTIVE_VALUE``.

        The ``num_gpu`` argument is retained for API compatibility; each returned
        entry corresponds to a single active GPU. Returned dict keys use the
        composite form ``f"{job_id}_gpu{gpu_id}"`` to stay unique across the
        GPUs of a job.
        """
        print(f"Processing multi-job PKL file (splitting into single-GPU jobs): {self.dcgm_file}")

        with open(self.dcgm_file, "rb") as f:
            job_to_df: dict = pickle.load(f)

        print(f"Loaded {len(job_to_df)} job(s) from PKL file")

        processed: dict[str, pd.DataFrame] = {}
        skipped: list = []

        for job_id, raw_df in job_to_df.items():
            try:
                gpu_results = self._process_pkl_job(job_id, raw_df)
            except ValueError as exc:
                print(f"Warning: skipping job {job_id}: {exc}")
                skipped.append(job_id)
                continue

            if not gpu_results:
                print(f"Warning: job {job_id} has no active GPU; skipping")
                skipped.append(job_id)
                continue

            for gpu_id, gpu_df in gpu_results.items():
                job_key = f"{job_id}_gpu{gpu_id}"
                processed[job_key] = gpu_df

        print(
            f"Successfully produced {len(processed)} single-GPU job(s); "
            f"skipped {len(skipped)} source job(s)"
        )
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
            f"Total Samples (GRACT > 0.9): {total_samples}, DRAMA Zero Samples: {drama_zeros}, "
            f"TENSO Zero Samples: {tensor_zeros}, FP64A Zero Samples: {fp64_zeros}, "
            f"FP32A Zero Samples: {fp32_zeros}, FP16A Zero Samples: {fp16_zeros}"
        )

    # ------------------------------------------------------------------ #
    # Multi-job helpers (PKL DataFrame preprocessing)
    # ------------------------------------------------------------------ #
    def _process_pkl_job(self, job_id, raw_df: pd.DataFrame) -> dict[int, pd.DataFrame]:
        """Split a single raw dcgm_df (4 GPUs) into per-GPU single-GPU jobs.

        Each raw dcgm_df is assumed to contain data for multiple GPUs. We treat
        each GPU as an individual job, keeping only GPUs that were actively used.

        Steps:
          1. Validate the raw DataFrame and locate its distinct gpu_ids.
          2. For each GPU, slice out its rows.
          3. Decide whether the GPU was active (based on ACTIVITY_METRIC).
          4. For active GPUs: rename long -> short column names, select/order the
             requested metrics, coerce to numeric, and report zero counts.

        Returns a dict {gpu_id: cleaned_df} containing only active GPUs.
        """
        if not isinstance(raw_df, pd.DataFrame):
            raise ValueError(f"expected a DataFrame, got {type(raw_df)}")

        if raw_df.empty:
            raise ValueError("empty DataFrame")

        gpu_ids = self._get_gpu_ids(raw_df)

        if not gpu_ids:
            raise ValueError("no GPU found")

        results: dict[int, pd.DataFrame] = {}

        for gpu_id in gpu_ids:
            gpu_df = raw_df[raw_df["gpu_id"] == gpu_id].copy()

            if gpu_df.empty:
                continue

            # Decide activity on the raw (long-named) activity column.
            if not self._is_gpu_active(gpu_df):
                print(f"Job {job_id}: GPU {gpu_id} inactive; skipping")
                continue

            cleaned = self._clean_gpu_df(gpu_df)

            print(f"Job {job_id}: GPU {gpu_id} active, {len(cleaned)} rows")
            self._count_zero(cleaned)
            results[gpu_id] = cleaned

        return results

    def _is_gpu_active(self, gpu_df: pd.DataFrame) -> bool:
        """Return True if the GPU was actively used.

        A GPU is active when at least ``MIN_ACTIVE_FRACTION`` of its samples have
        an ``ACTIVITY_METRIC`` value of at least ``MIN_ACTIVE_VALUE``.
        The check runs on the raw (long) column name before renaming.
        """
        if self.ACTIVITY_METRIC not in gpu_df.columns:
            raise ValueError(
                f"activity metric column '{self.ACTIVITY_METRIC}' not found in DataFrame"
            )

        activity = pd.to_numeric(gpu_df[self.ACTIVITY_METRIC], errors="coerce").fillna(0.0)

        if len(activity) == 0:
            return False

        active_fraction = (activity >= self.MIN_ACTIVE_VALUE).mean()
        return active_fraction >= self.MIN_ACTIVE_FRACTION

    def _clean_gpu_df(self, gpu_df: pd.DataFrame) -> pd.DataFrame:
        """Rename, select, order, and numeric-coerce a single-GPU DataFrame."""
        # Keep only mapped columns, then rename long -> short names.
        keep_cols = [c for c in gpu_df.columns if c in self.COLUMN_RENAME_MAP]
        gpu_df = gpu_df[keep_cols].rename(columns=self.COLUMN_RENAME_MAP)

        # Validate all requested metrics are available after renaming.
        missing = [m for m in self.metric_names if m not in gpu_df.columns]
        if missing:
            raise ValueError(f"missing required metric column(s) after renaming: {missing}")

        # Select and order exactly the requested metrics (mirrors the text path,
        # whose DataFrame columns == self.metric_names).
        result_df = gpu_df[self.metric_names].reset_index(drop=True)

        # Ensure numeric dtype (raw PKL may store objects / strings / N/A).
        result_df = result_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

        return result_df

    def _get_gpu_ids(self, raw_df: pd.DataFrame) -> list:
        """Return the sorted list of distinct gpu_ids present in the DataFrame."""
        if "gpu_id" not in raw_df.columns:
            raise ValueError("raw DataFrame has no 'gpu_id' column")

        gpu_ids = raw_df["gpu_id"].dropna().unique()
        return sorted(gpu_ids)
