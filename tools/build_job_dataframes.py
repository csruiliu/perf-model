#!/usr/bin/env python3
"""
Generate three pickle files, each containing a dict:  job_to_df[jobid] -> dcgm_df

    all jobs
    jobs that used more than one node
    jobs that used exactly one node

The DCGM DataFrame for each job is read from its `.pq` file.
Node count is derived from the `nodelist` field in the JSON metadata.
"""

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build per-job DCGM DataFrame dicts and save them as pickle files."
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=Path,
        required=True,
        help="Root folder containing the daily subfolders (e.g. jobwise_dcgm_march_2026_gpu).",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory to write the pickle files (default: current directory).",
    )
    parser.add_argument(
        "--all-name",
        default="all_jobs.pkl",
        help="Filename for the all-jobs pickle (default: all_jobs.pkl).",
    )
    parser.add_argument(
        "--multi-name",
        default="multi_node_jobs.pkl",
        help="Filename for the multi-node pickle (default: multi_node_jobs.pkl).",
    )
    parser.add_argument(
        "--single-name",
        default="single_node_jobs.pkl",
        help="Filename for the single-node pickle (default: single_node_jobs.pkl).",
    )
    return parser.parse_args()


def get_node_count(meta: dict) -> int:
    """Return the number of unique nodes used by a job."""
    nodes = set()
    for entry in meta.get("entries", []):
        nodes.update(entry.get("nodelist", []))
    return len(nodes)


def main():
    args = parse_args()
    root = args.input_dir

    if not root.is_dir():
        raise SystemExit(f"[ERROR] Input directory not found: {root}")

    job_to_df_all = {}
    job_to_df_multi = {}
    job_to_df_single = {}

    for subfolder in sorted(root.iterdir()):
        if not subfolder.is_dir():
            continue

        for json_path in sorted(subfolder.glob("*.json")):
            stem = json_path.stem  # <job_id>_<user_id>
            jobid = stem.split("_")[0]
            pq_path = subfolder / f"{stem}.pq"

            if not pq_path.exists():
                print(f"[WARN] Missing .pq for {stem}, skipping.")
                continue

            try:
                with open(json_path) as f:
                    meta = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                print(f"[WARN] Could not read {json_path}: {e}")
                continue

            n_nodes = get_node_count(meta)

            try:
                dcgm_df = pd.read_parquet(pq_path)
            except Exception as e:
                print(f"[WARN] Could not read {pq_path}: {e}")
                continue

            job_to_df_all[jobid] = dcgm_df
            if n_nodes > 1:
                job_to_df_multi[jobid] = dcgm_df
            elif n_nodes == 1:
                job_to_df_single[jobid] = dcgm_df

    # --- Save ---
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, d in [
        (args.all_name, job_to_df_all),
        (args.multi_name, job_to_df_multi),
        (args.single_name, job_to_df_single),
    ]:
        path = out_dir / name
        with open(path, "wb") as f:
            pickle.dump(d, f)
        print(f"Saved {len(d):>6} jobs to {path}")


if __name__ == "__main__":
    main()
