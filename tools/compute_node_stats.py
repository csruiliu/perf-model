"""
job_nodecount_hist.py

Distribution of jobs by distinct node count.

A "job" is the (jobid, userid) pair read from inside each JSON, merged
across all date folders. A job's node count is the number of DISTINCT
nodes it touched (union of every entry's 'nodelist').

Output CSV columns:
    num_nodes,num_jobs,pct_of_jobs
one row per node-count value that occurs, sorted ascending.
pct_of_jobs is rounded to 2 decimals.
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Distribution of jobs by distinct node count.")
    p.add_argument(
        "-i",
        "--input-dir",
        type=Path,
        required=True,
        help="Root folder containing the daily subfolders.",
    )
    p.add_argument(
        "-o",
        "--output-csv",
        type=Path,
        default=Path("job_nodecount_hist.csv"),
        help="Output CSV (default: job_nodecount_hist.csv).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    root = args.input_dir
    if not root.is_dir():
        raise SystemExit(f"[ERROR] Input directory not found: {root}")

    # (jobid, userid) -> set of distinct nodes (union across entries/folders)
    job_nodes = defaultdict(set)

    for subfolder in sorted(root.iterdir()):
        if not subfolder.is_dir():
            continue
        for json_path in sorted(subfolder.glob("*.json")):
            try:
                with open(json_path) as f:
                    meta = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                print(f"[WARN] Could not read {json_path}: {e}")
                continue

            jobid = meta.get("jobid")
            userid = meta.get("userid")
            if jobid is None or userid is None:
                print(f"[WARN] Missing jobid/userid in {json_path}")
                continue

            key = (str(jobid), str(userid))
            for e in meta.get("entries", []):
                nodelist = e.get("nodelist") or []
                if not isinstance(nodelist, list):
                    nodelist = [nodelist]
                for n in nodelist:
                    job_nodes[key].add(str(n))

    # Count jobs by their distinct node count (skip jobs with 0 nodes).
    counts = [len(nodes) for nodes in job_nodes.values() if nodes]
    total_jobs = len(counts)
    if total_jobs == 0:
        raise SystemExit("[ERROR] No jobs with node information found.")

    hist = Counter(counts)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["num_nodes", "num_jobs", "pct_of_jobs"])
        for num_nodes in sorted(hist):
            num_jobs = hist[num_nodes]
            pct = round(100.0 * num_jobs / total_jobs, 2)
            writer.writerow([num_nodes, num_jobs, pct])

    print(f"Total jobs: {total_jobs}")
    print(f"Written to: {args.output_csv}")


if __name__ == "__main__":
    main()
