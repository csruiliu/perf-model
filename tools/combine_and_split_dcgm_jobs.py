#!/usr/bin/env python3
"""
Process NVIDIA DCGM job data across dated sub-folders.

Steps:
  1. Walk all MM_DD_YYYY sub-folders and group PQ files by job, where a job is
     identified by the (jobid, userid) pair read from the paired JSON file.
     Concatenate a job's PQ files in sub-folder date order.
  2. Split each job's combined data by gpu_id (0, 1, 2, 3).
  3. Write all per-GPU PQ files into a single output folder, named
     <jobid>_<userid>_<gpuid>.pq.
"""

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

# Sub-folder names look like MM_DD_YYYY, e.g. 03_06_2026
FOLDER_DATE_RE = re.compile(r"^\d{2}_\d{2}_\d{4}$")
GPU_IDS = [0, 1, 2, 3]


def parse_folder_date(folder_name: str) -> datetime:
    """Convert an MM_DD_YYYY folder name into a datetime for sorting."""
    return datetime.strptime(folder_name, "%m_%d_%Y")


def collect_jobs(root: Path):
    """
    Walk all dated sub-folders and build a mapping:
        (jobid, userid) -> list of (folder_date, pq_path)

    The jobid and userid are read from each JSON file's fields (not the
    file name), and the JSON's stem is used to locate the paired PQ file.
    """
    jobs = defaultdict(list)

    subfolders = [d for d in root.iterdir() if d.is_dir() and FOLDER_DATE_RE.match(d.name)]
    # Sort so files for a job are appended in chronological folder order.
    subfolders.sort(key=lambda d: parse_folder_date(d.name))

    for folder in subfolders:
        folder_date = parse_folder_date(folder.name)

        for json_path in folder.glob("*.json"):
            pq_path = json_path.with_suffix(".pq")
            if not pq_path.exists():
                print(f"[WARN] No matching PQ for {json_path}, skipping.")
                continue

            try:
                with open(json_path) as f:
                    meta = json.load(f)
                jobid = str(meta["jobid"])
                userid = str(meta["userid"])
            except (json.JSONDecodeError, KeyError) as e:
                print(f"[WARN] Could not read jobid/userid from {json_path}: {e}")
                continue

            jobs[(jobid, userid)].append((folder_date, pq_path))

    return jobs


def process_jobs(jobs, output_dir: Path):
    """
    For each job, concatenate its PQ files in date order, split by gpu_id,
    and write one PQ file per GPU into output_dir.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for (jobid, userid), entries in jobs.items():
        # Concatenate in ascending folder-date order.
        entries.sort(key=lambda x: x[0])

        frames = []
        for _, pq_path in entries:
            try:
                frames.append(pd.read_parquet(pq_path))
            except Exception as e:
                print(f"[WARN] Failed to read {pq_path}: {e}")

        if not frames:
            print(f"[WARN] No readable PQ data for job {jobid}_{userid}, skipping.")
            continue

        combined = pd.concat(frames, ignore_index=True)

        n_folders = len(entries)
        print(f"[INFO] Job {jobid}_{userid}: combined {n_folders} file(s), {len(combined)} rows.")

        # Split by gpu_id and write one file per GPU.
        for gpu_id in GPU_IDS:
            gpu_df = combined[combined["gpu_id"] == gpu_id]

            if gpu_df.empty:
                print(f"[WARN] Job {jobid}_{userid}: no rows for gpu_id={gpu_id}.")
                continue

            out_name = f"{jobid}_{userid}_{gpu_id}.pq"
            out_path = output_dir / out_name
            gpu_df.to_parquet(out_path, index=False)
            print(f"[INFO]   wrote {out_path} ({len(gpu_df)} rows).")


def main():
    parser = argparse.ArgumentParser(
        description="Combine job PQ files across dated sub-folders and split each job by GPU."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        dest="root",
        help="Root folder containing MM_DD_YYYY sub-folders.",
    )
    parser.add_argument(
        "-o", "--output", type=Path, required=True, help="Output folder for the per-GPU PQ files."
    )
    args = parser.parse_args()

    if not args.root.is_dir():
        parser.error(f"Root folder does not exist: {args.root}")

    jobs = collect_jobs(args.root)
    print(f"[INFO] Found {len(jobs)} unique job(s).")

    process_jobs(jobs, args.output)
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
