import argparse

from hw_specs import GPUSpec

from workload_processor import WorkloadProcessor

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Combined GPU Performance Modeling and Analysis Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('-rg', '--ref_gpu', required=True, choices=list(GPUSpec.keys()), help='Reference GPU')
    parser.add_argument('-tg', '--tgt_gpu', choices=list(GPUSpec.keys()), help='Target GPU (optional)')
    parser.add_argument('-d', '--sample_interval_ms', type=int, required=True, help='Sample interval in milliseconds')
    parser.add_argument('-wm', '--workload_metadata_file', action='store', type=str, required=True, help='indicate the job master file')
    parser.add_argument('-wd', '--workload_data_path', action='store', type=str, required=True, help='indicate the job path that consists of various jobs')
    parser.add_argument('--max_workers', action='store', type=int, default=32, help='maximum number of worker processes (defaults to CPU count)')    
    parser.add_argument('--chunk_size', action='store', type=int, default=1, help='the number of chunks that each worker processes, 1 is maximum parallelism, large size has less overhead')
    
    args = parser.parse_args()

    workload_processor = WorkloadProcessor(
        args.workload_metadata_file, args.workload_data_path, args.max_workers, args.chunk_size, args.sample_interval_ms
    )

    workload_processor.process_workload_metadata()

    workload_processor.run()


if __name__=="__main__":
    main()