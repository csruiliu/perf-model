#!/usr/bin/env python3
"""
Compute statistics on the number of jobs by node count.

For each job, the node count is derived from the `nodelist` field in the
job's JSON metadata (unique nodes across all entries).
"""

import json
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd

# --- Configuration ---
ROOT = Path("jobwise_dcgm_march_2026_gpu")
OUTPUT_CSV = Path("node_stats.csv")


def get_node_count(json_path: Path) -> int:
    """Return the number of unique nodes used by a job."""
    with open(json_path, "r") as f:
        meta = json.load(f)

    nodes = set()
    for entry in meta.get("entries", []):
        nodes.update(entry.get("nodelist", []))
    return len(nodes)


def main():
    # Map: node_count -> number of jobs
    nodecount_to_jobs = Counter()
    # Optional: keep track of which jobs fall into each bucket
    nodecount_to_jobids = defaultdict(list)

    total_jobs = 0

    for subfolder in sorted(ROOT.iterdir()):
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

    summary.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved summary to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()