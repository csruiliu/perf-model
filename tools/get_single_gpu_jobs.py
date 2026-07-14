import argparse
import gc
import pickle

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter a job_to_df pickle down to genuinely single-GPU jobs."
    )
    parser.add_argument(
        "--input-path",
        default="job_to_df.pkl",
        help="Path to the input pickle (default: job_to_df.pkl)",
    )
    parser.add_argument(
        "--output-path",
        default="job_to_df_single_gpu.pkl",
        help="Path for the output pickle (default: job_to_df_single_gpu.pkl)",
    )
    parser.add_argument(
        "--activity-threshold",
        type=float,
        default=0.01,
        help="A sample counts as 'active' when ACTIVITY_COL exceeds this value "
        "(gr_engine_active is in [0, 1]). Default: 0.01",
    )
    parser.add_argument(
        "--min-active-fraction",
        type=float,
        default=0.01,
        help="Fraction of samples that must be active for a GPU to count as used. Default: 0.01",
    )
    parser.add_argument(
        "--activity-col",
        default="nersc_ldms_dcgm_gr_engine_active",
        help="Column used as the activity signal. Default: nersc_ldms_dcgm_gr_engine_active",
    )
    return parser.parse_args()


def active_gpus(df, activity_col, activity_threshold, min_active_fraction):
    """Return the set of gpu_ids that were genuinely used during the job."""
    active = set()
    for gpu_id, sub in df.groupby("gpu_id"):
        vals = sub[activity_col].to_numpy(dtype=float)
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            continue
        active_fraction = np.mean(vals > activity_threshold)
        if active_fraction >= min_active_fraction:
            active.add(gpu_id)
    return active


def main():
    args = parse_args()

    print("Configuration:")
    print(f"  input_path          = {args.input_path}")
    print(f"  output_path         = {args.output_path}")
    print(f"  activity_threshold  = {args.activity_threshold}")
    print(f"  min_active_fraction = {args.min_active_fraction}")
    print(f"  activity_col        = {args.activity_col}\n")

    print("Loading input pickle (this may take a while)...")
    with open(args.input_path, "rb") as f:
        job_to_df = pickle.load(f)
    total_jobs = len(job_to_df)
    print(f"Loaded {total_jobs} jobs.")

    single_gpu = {}
    n_processed = 0

    for jobid, df in job_to_df.items():
        n_processed += 1

        if args.activity_col not in df.columns or "gpu_id" not in df.columns:
            continue

        used = active_gpus(df, args.activity_col, args.activity_threshold, args.min_active_fraction)
        if len(used) == 1:
            (gpu_id,) = used  # unpack the single active gpu_id
            # Keep only the rows belonging to the active GPU.
            filtered = df[df["gpu_id"] == gpu_id].copy()
            single_gpu[jobid] = filtered

        if n_processed % 1000 == 0:
            print(f"Processed {n_processed}/{total_jobs}; single-GPU so far: {len(single_gpu)}")

    n_single = len(single_gpu)
    print("\n" + "=" * 50)
    print(f"Total jobs:        {total_jobs}")
    print(f"Single-GPU jobs:   {n_single}")
    if total_jobs:
        print(f"Fraction:          {n_single / total_jobs:.2%}")
    print("=" * 50 + "\n")

    # Free the original before writing, to reduce peak memory.
    del job_to_df
    gc.collect()

    print("Writing output pickle...")
    with open(args.output_path, "wb") as f:
        pickle.dump(single_gpu, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Wrote {n_single} single-GPU jobs to {args.output_path}.")


if __name__ == "__main__":
    main()
