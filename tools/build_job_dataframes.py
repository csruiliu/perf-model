#!/usr/bin/env python3
"""
Generate three pickle files, each containing a dict:
    job_to_df[(jobid, userid)] -> dcgm_df

A "job" is the (jobid, userid) pair read from INSIDE each JSON -- not from
the filename -- so array tasks (jobid '49735483_362') and underscore
userids ('yu_yao') are handled correctly.

A job whose files are spread across multiple date folders is ONE job: all
of its .pq files are read and concatenated, ordered by folder date
(folder names are 'MM_DD_YYYY').

Processes one category per pass to keep peak memory low.
"""

import argparse
import json
import pickle
from collections import defaultdict
from datetime import datetime
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
        help="Root folder containing the daily subfolders.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory to write the pickle files.",
    )
    parser.add_argument("--all-name", default="all_jobs.pkl")
    parser.add_argument("--multi-name", default="multi_node_jobs.pkl")
    parser.add_argument("--single-name", default="single_node_jobs.pkl")
    parser.add_argument(
        "--category",
        choices=["all", "multi", "single"],
        default=None,
        help="If set, only build this one category (recommended on memory-limited nodes).",
    )
    return parser.parse_args()


def folder_date_key(folder_name: str):
    """Parse a 'MM_DD_YYYY' folder name into a date for chronological sorting.

    Falls back to the raw string (sorted last) if it doesn't match, so an
    unexpected folder name never crashes the run.
    """
    try:
        return (0, datetime.strptime(folder_name, "%m_%d_%Y").date())
    except ValueError:
        print(f"[WARN] Folder name '{folder_name}' is not MM_DD_YYYY; ordering it last.")
        return (1, folder_name)


def get_node_count(meta: dict) -> int:
    """Distinct nodes a job touched (union of every entry's nodelist)."""
    nodes = set()
    for entry in meta.get("entries", []):
        nodes.update(entry.get("nodelist", []))
    return len(nodes)


def scan_jobs(root: Path):
    """
    Group all data for each (jobid, userid) pair across every folder.

    Returns:
        jobs: dict[(jobid, userid)] -> {
            "pq_files": list of (folder_name, pq_path),  # unsorted
            "nodes": set of node names (union across folders/entries),
        }
    """
    jobs = defaultdict(lambda: {"pq_files": [], "nodes": set()})

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
            stem = json_path.stem
            pq_path = subfolder / f"{stem}.pq"
            if not pq_path.exists():
                print(f"[WARN] Missing .pq for {stem}, skipping this file.")
                continue

            entry = jobs[key]
            entry["pq_files"].append((subfolder.name, pq_path))
            for e in meta.get("entries", []):
                entry["nodes"].update(e.get("nodelist", []))

    return jobs


def read_job_df(pq_files) -> pd.DataFrame | None:
    """
    Read and concatenate a job's .pq files, ordered by folder date.

    pq_files: list of (folder_name, pq_path). Sorted chronologically by the
    parsed MM_DD_YYYY folder date so the resulting DataFrame is in date order.
    """
    frames = []
    for folder_name, pq_path in sorted(pq_files, key=lambda t: folder_date_key(t[0])):
        try:
            frames.append(pd.read_parquet(pq_path))
        except Exception as e:
            print(f"[WARN] Could not read {pq_path}: {e}")
    if not frames:
        return None
    if len(frames) == 1:
        return frames[0]
    return pd.concat(frames, ignore_index=True)


def build_category(jobs: dict, category: str) -> dict:
    """Build the job_to_df dict for one category from the pre-scanned jobs."""
    job_to_df = {}
    for key, info in jobs.items():
        n_nodes = len(info["nodes"])
        if category == "multi" and n_nodes <= 1:
            continue
        if category == "single" and n_nodes != 1:
            continue
        df = read_job_df(info["pq_files"])
        if df is not None:
            job_to_df[key] = df
    return job_to_df


def main():
    args = parse_args()
    root = args.input_dir
    if not root.is_dir():
        raise SystemExit(f"[ERROR] Input directory not found: {root}")

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    name_map = {"all": args.all_name, "multi": args.multi_name, "single": args.single_name}
    categories = [args.category] if args.category else ["all", "multi", "single"]

    # Scan once; reuse the grouping for every category pass.
    jobs = scan_jobs(root)
    print(f"Discovered {len(jobs)} unique (jobid, userid) jobs.")
    multi_folder = sum(1 for info in jobs.values() if len({fn for fn, _ in info["pq_files"]}) > 1)
    print(f"  of which span >1 folder (merged): {multi_folder}")

    for cat in categories:
        d = build_category(jobs, cat)
        path = out_dir / name_map[cat]
        with open(path, "wb") as f:
            pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved {len(d):>6} jobs to {path}")
        del d  # free memory before the next pass


if __name__ == "__main__":
    main()
