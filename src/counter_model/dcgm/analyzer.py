import argparse

import pandas as pd

from counter_model.dcgm.estimator import SingleGpuEstimator
from counter_model.dcgm.profiler import SingleGpuProfiler


class SystemWideAnalyzer:
    """Handles metrics file processing"""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        # Node-hours per job. Supply via a dict {job_id: node_hours};
        # node_hours = job_runtime works for single-node jobs.
        self.job_node_hours: dict = getattr(args, "job_node_hours", {}) or {}

    def run(self, job_to_df: dict[str, pd.DataFrame], smocc_level: str = "mid"):
        """
        job_to_df:          {job_id: cleaned single-GPU DataFrame}
        Save a DataFrame indexed by job_id with columns:
            ref_runtime, <name>_runtime, <name>_speedup, node_hours
        """
        records = {}

        for job_id, df in job_to_df.items():
            if df.empty:
                continue

            row = {}

            # Reference runtime (A100).
            ref_profiler = SingleGpuProfiler(self.args.sample_interval_ms, self.args.ref_gpu)
            ref_runtime = ref_profiler.run(df, self.args, False)
            row["ref_runtime"] = ref_runtime

            # Target runtime for the target GPU
            estimator = SingleGpuEstimator(self.args)
            tgt_runtime_dict = estimator.run(df, self.args, is_printout=False)
            tgt_runtime = tgt_runtime_dict.get(smocc_level)

            records[job_id] = {
                "ref_runtime": ref_runtime,
                "tgt_runtime": tgt_runtime,
                "speedup": (ref_runtime / tgt_runtime) if tgt_runtime > 0 else 0.0,
                "node_hours": self.job_node_hours.get(job_id, ref_runtime) / (3600 * 1000),
            }

        result_df = pd.DataFrame.from_dict(records, orient="index")
        result_df.index.name = "job_id"
        result_df.to_parquet("speedup_dist.parquet")
