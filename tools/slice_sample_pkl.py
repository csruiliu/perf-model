"""
Slice a large job_to_df pickle file into a smaller random sample.

Example:
    python slice_pkl.py --input job_to_df.pkl --output sample.pkl --n 20 --seed 42
    python slice_pkl.py --min-rows 1000 --n 50 --verify
"""

import argparse
import pickle
import random


def main():
    parser = argparse.ArgumentParser(
        description="Randomly sample N jobs from a large job_to_df pickle file."
    )
    parser.add_argument(
        "--input", default="job_to_df.pkl", help="Path to the large input pickle file."
    )
    parser.add_argument(
        "--output", default="job_to_df_sample.pkl", help="Path for the smaller output pickle file."
    )
    parser.add_argument("--n", type=int, default=20, help="Number of jobs to randomly select.")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility. Use -1 for fully random.",
    )
    parser.add_argument(
        "--min-rows",
        type=int,
        default=0,
        help="Only sample jobs whose DataFrame has at least this many rows.",
    )
    parser.add_argument(
        "--protocol",
        type=int,
        default=pickle.HIGHEST_PROTOCOL,
        help="Pickle protocol for the output file.",
    )
    parser.add_argument(
        "--verify", action="store_true", help="Reload the output file and print DataFrame shapes."
    )
    args = parser.parse_args()

    # Load the original large pickle file
    print(f"Loading {args.input} ... (this may take a while)")
    with open(args.input, "rb") as f:
        job_to_df = pickle.load(f)

    total_jobs = len(job_to_df)
    print(f"Loaded dict with {total_jobs} jobs.")

    # Set up the random seed
    if args.seed != -1:
        random.seed(args.seed)

    # Build the pool of candidate job IDs, applying the min-rows filter if requested
    if args.min_rows > 0:
        candidates = [
            jid
            for jid, df in job_to_df.items()
            if hasattr(df, "shape") and df.shape[0] >= args.min_rows
        ]
        print(f"{len(candidates)} jobs have >= {args.min_rows} rows.")
    else:
        candidates = list(job_to_df.keys())

    if not candidates:
        raise ValueError("No jobs matched the filtering criteria.")

    # Randomly select N job IDs
    n_select = min(args.n, len(candidates))
    if n_select < args.n:
        print(f"Warning: only {n_select} jobs available; sampling all of them.")
    selected_jobids = random.sample(candidates, n_select)

    print(f"Selected {n_select} job IDs: {selected_jobids}")

    # Build the smaller dict (references only, no copy)
    small_dict = {jid: job_to_df[jid] for jid in selected_jobids}

    # Save to a new pickle file
    print(f"Saving to {args.output} ...")
    with open(args.output, "wb") as f:
        pickle.dump(small_dict, f, protocol=args.protocol)
    print("Done.")

    # Optional verification
    if args.verify:
        print("\nVerifying output file ...")
        with open(args.output, "rb") as f:
            check = pickle.load(f)
        print(f"Reloaded {len(check)} jobs:")
        for jid, df in check.items():
            shape = getattr(df, "shape", "N/A")
            print(f"  {jid}: shape={shape}")


if __name__ == "__main__":
    main()
