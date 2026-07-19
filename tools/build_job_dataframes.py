#!/usr/bin/env python3
"""
Generate three pickle files, each containing a dict:  job_to_df[jobid] -> dcgm_df

    all_jobs.pkl          : all jobs
    multi_node_jobs.pkl   : jobs that used more than one node
    single_node_jobs.pkl  : jobs that used exactly one node

The DCGM DataFrame for each job is read from its `.pq` file.
Node count is derived from the `nodelist` field in the JSON metadata.
"""

import json
import pickle
from pathlib import Path

import pandas as pd

# --- Configuration ---
ROOT = Path("jobwise_dcgm_march_2026_gpu")
OUT_ALL = Path("all_jobs.pkl")
OUT_MULTI = Path("multi_node_jobs.pkl")
OUT_SINGLE = Path("single_node_jobs.pkl")


def get_node_count(meta: dict) -> int:
    """Return the number of unique nodes used by a job."""
    nodes = set()
    for entry in meta.get("entries", []):
        nodes.update(entry.get("nodelist", []))
    return len(nodes)


def main():
    job_to_df_all = {}
    job_to_df_multi = {}
    job_to_df_single = {}

    for subfolder in sorted(ROOT.iterdir()):
        if not subfolder.is_dir():
            continue

        for json_path in sorted(subfolder.glob("*.json")):
            stem = json_path.stem              # <job_id>_<user_id>
            jobid = stem.split("_")[0]
            pq_path = subfolder / f"{stem}.pq"

            if not pq_path.exists():
                print(f"[WARN] Missing .pq for {stem}, skipping.")
                continue

            # Load metadata to determine node count
            try:
                with open(json_path, "r") as f:
                    meta = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                print(f"[WARN] Could not read {json_path}: {e}")
                continue

            n_nodes = get_node_count(meta)

            # Load the DCGM DataFrame
            try:
                dcgm_df = pd.read_parquet(pq_path)
            except Exception as e:
                print(f"[WARN] Could not read {pq_path}: {e}")
                continue

            # Populate the dicts
            job_to_df_all[jobid] = dcgm_df
            if n_nodes > 1:
                job_to_df_multi[jobid] = dcgm_df
            elif n_nodes == 1:
                job_to_df_single[jobid] = dcgm_df
            # n_nodes == 0 (no nodelist) is included in "all" only

    # --- Save ---
    for path, d in [
        (OUT_ALL, job_to_df_all),
        (OUT_MULTI, job_to_df_multi),
        (OUT_SINGLE, job_to_df_single),
    ]:
        with open(path, "wb") as f:
            pickle.dump(d, f)
        print(f"Saved {len(d):>6} jobs to {path}")


if __name__ == "__main__":
    main()