#!/usr/bin/env python3
"""
Compute statistics on the number of jobs by node count.

For each job, the node count is derived from the `nodelist` field in the
job's JSON metadata (unique nodes across all entries).
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute statistics on the number of jobs by node count."
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
        "--output-csv",
        type=Path,
        default=Path("node_stats.csv"),
        help="Path to write the summary CSV (default: node_stats.csv).",
    )
    return parser.parse_args()


def get_node_count(json_path: Path) -> int:
    """Return the number of unique nodes used by a job."""
    with open(json_path) as f:
        meta = json.load(f)

    nodes = set()
    for entry in meta.get("entries", []):
        nodes.update(entry.get("nodelist", []))
    return len(nodes)


def main():
    args = parse_args()
    root = args.input_dir

    if not root.is_dir():
        raise SystemExit(f"[ERROR] Input directory not found: {root}")

    # Map: node_count -> number of jobs
    nodecount_to_jobs = Counter()
    nodecount_to_jobids = defaultdict(list)
    total_jobs = 0

    for subfolder in sorted(root.iterdir()):
        if not subfolder.is_dir():
            continue

        for json_path in sorted(subfolder.glob("*.json")):
            # Filename format: <job_id>_<user_id>.json
            jobid = json_path.stem.split("_")[0]

            try:
                n_nodes = get_node_count(json_path)
            except (json.JSONDecodeError, OSError) as e:
                print(f"[WARN] Skipping {json_path}: {e}")
                continue

            nodecount_to_jobs[n_nodes] += 1
            nodecount_to_jobids[n_nodes].append(jobid)
            total_jobs += 1

    # --- Build a summary DataFrame ---
    rows = []
    for n_nodes in sorted(nodecount_to_jobs):
        count = nodecount_to_jobs[n_nodes]
        rows.append(
            {
                "num_nodes": n_nodes,
                "num_jobs": count,
                "pct_of_jobs": round(100 * count / total_jobs, 2) if total_jobs else 0,
            }
        )

    summary = pd.DataFrame(rows)

    # --- Report ---
    print(f"Total jobs: {total_jobs}\n")
    print("Distribution of jobs by node count:")
    print(summary.to_string(index=False))

    single_node = nodecount_to_jobs.get(1, 0)
    multi_node = total_jobs - single_node
    print(f"\nSingle-node jobs: {single_node}")
    print(f"Multi-node jobs:  {multi_node}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_csv, index=False)
    print(f"\nSaved summary to {args.output_csv}")


if __name__ == "__main__":
    main()
