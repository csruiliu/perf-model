import argparse

from counter_model.dcgm.estimator import SingleGpuEstimator
from counter_model.dcgm.job_parser import JobParser
from counter_model.dcgm.profiler import SingleGpuProfiler


class Dispatcher:
    """Handles metrics file processing"""

    def __init__(self, args: argparse.Namespace):
        self.args = args

    def dispatch(self):
        table = {
            ("single", "single"): self.single_job_single_gpu,
            ("single", "multi"): self.single_job_multi_gpu,
            ("multi", "single"): self.multi_job_single_gpu,
            ("multi", "multi"): self.multi_job_multi_gpu,
        }
        try:
            handler = table[(self.args.jobs, self.args.gpus)]
        except KeyError as exc:
            raise SystemExit(
                f"Unsupported combination: jobs={self.args.jobs}, gpus={self.args.gpus}"
            ) from exc
        return handler()

    def single_job_single_gpu(self):
        job_parser = JobParser(self.args.dcgm_input, self.args.metrics)
        profiled_df = job_parser.parsing_single_job(num_gpu=1)

        # Create and run reference profiler
        ref_profiler = SingleGpuProfiler(self.args.sample_interval_ms, self.args.ref_gpu)
        ref_profiler.run(profiled_df, self.args)
        # Create target estimator and run if specified
        if self.args.tgt_gpu:
            tgt_estimator = SingleGpuEstimator(self.args)
            tgt_estimator.run(profiled_df, self.args)

    def single_job_multi_gpu(self):
        pass

    def multi_job_single_gpu(self):
        job_parser = JobParser(self.args.dcgm_input, self.args.metrics)
        job_parser.parsing_multi_job(num_gpu=1)
        pass

    def multi_job_multi_gpu(self):
        pass
