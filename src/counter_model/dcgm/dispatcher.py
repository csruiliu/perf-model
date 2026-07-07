import argparse
import os

from counter_model.dcgm.analyzer import SystemWideAnalyzer
from counter_model.dcgm.estimator import SingleGpuEstimator
from counter_model.dcgm.job_parser import JobParser
from counter_model.dcgm.profiler import SingleGpuProfiler
from counter_model.dcgm.utils import plot_speedup_distribution


class Dispatcher:
    """Handles metrics file processing"""

    def __init__(self, args: argparse.Namespace):
        self.args = args

    def dispatch(self):
        if self.args.job_mode == "single" and self.args.num_gpu == 1:
            self.single_job_single_gpu()
        elif self.args.job_mode == "single" and self.args.num_gpu > 1:
            self.single_job_multi_gpu()
        elif self.args.job_mode == "multi" and self.args.num_gpu == 1:
            self.multi_job_single_gpu()
        elif self.args.job_mode == "multi" and self.args.num_gpu > 1:
            self.multi_job_multi_gpu()
        else:
            raise SystemExit(
                f"Unsupported combination: jobs={self.args.job_mode}, gpus={self.args.num_gpu}"
            )

    def single_job_single_gpu(self):
        job_parser = JobParser(self.args.dcgm_input, self.args.metrics)
        profiled_df = job_parser.parsing_single_job(num_gpu=1)

        # Create and run reference profiler
        ref_profiler = SingleGpuProfiler(self.args.sample_interval_ms, self.args.ref_gpu)
        ref_profiler.run(profiled_df, self.args, is_printout=True)
        # Create target estimator and run if specified
        if self.args.tgt_gpu:
            tgt_estimator = SingleGpuEstimator(self.args)
            tgt_estimator.run(profiled_df, self.args, is_printout=True)

    def single_job_multi_gpu(self):
        pass

    def multi_job_single_gpu(self):
        job_parser = JobParser(self.args.dcgm_input, self.args.metrics)
        job_to_df = job_parser.parsing_multi_job(num_gpu=1)
        gpu_tag = self.args.tgt_gpu

        analyzer = SystemWideAnalyzer(self.args)
        result_df = analyzer.run(job_to_df)
        speedup_dist_fig_path = os.path.join(self.args.plot_dir, f"speedup_dist_{gpu_tag}.png")
        plot_speedup_distribution(result_df, gpu_name=gpu_tag, outpath=speedup_dist_fig_path)

    def multi_job_multi_gpu(self):
        pass
