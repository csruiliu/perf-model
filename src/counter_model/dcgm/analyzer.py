import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from counter_model.dcgm.estimator import SingleGpuEstimator
from counter_model.dcgm.profiler import SingleGpuProfiler


class SystemWideAnalyzer:
    """Handles metrics file processing"""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.max_workers = min(args.max_workers, os.cpu_count())
        self.job_node_hours: dict = getattr(args, "job_node_hours", {}) or {}

    def _process_job(self, job_id: str, df: pd.DataFrame, smocc_level: str):
        """Process a single job. Returns (job_id, record) or (job_id, None)."""
        if df.empty:
            return job_id, None

        # Reference runtime (A100).
        ref_profiler = SingleGpuProfiler(self.args.sample_interval_ms, self.args.ref_gpu)
        ref_runtime = ref_profiler.run(df, self.args, False)

        # Target runtime for the target GPU.
        estimator = SingleGpuEstimator(self.args)
        tgt_runtime_dict = estimator.run(df, self.args, is_printout=False)
        tgt_runtime = tgt_runtime_dict.get(smocc_level)

        record = {
            "ref_runtime": ref_runtime,
            "tgt_runtime": tgt_runtime,
            "speedup": (ref_runtime / tgt_runtime) if tgt_runtime > 0 else 0.0,
            "node_hours": ref_runtime / (3600 * 1000),
        }
        return job_id, record

    def run(self, job_to_df: dict[str, pd.DataFrame], smocc_level: str = "mid"):
        records = {}

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._process_job, job_id, df, smocc_level): job_id
                for job_id, df in job_to_df.items()
            }
            for future in as_completed(futures):
                job_id, record = future.result()
                if record is not None:
                    records[job_id] = record

        result_df = pd.DataFrame.from_dict(records, orient="index")
        result_df.index.name = "job_id"
        result_df.to_parquet(self.args.agg_results_dir + "/speedup_dist.parquet")
        return result_df
