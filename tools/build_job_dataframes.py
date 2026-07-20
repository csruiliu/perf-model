#!/usr/bin/env python3
"""
Generate pickle files, each containing a dict:
    job_to_df[(jobid, userid)] -> dcgm_df

A "job" is the (jobid, userid) pair read from INSIDE each JSON. A job whose
files span multiple date folders is ONE job (all .pq concatenated in date
order). Within each category, jobs are sharded across N pickle files, and
shards are built in parallel across worker processes.
"""

import argparse
import hashlib
import json
import pickle
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
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
        help="If set, only build this one category.",
    )
    parser.add_argument(
        "-n",
        "--num-shards",
        type=int,
        default=1,
        help="Split each category's output across this many pickle files.",
    )
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes. Default: os.cpu_count().",
    )
    parser.add_argument(
        "--scan-workers",
        type=int,
        default=None,
        help="Processes for the JSON-scanning phase. "
        "Default: min(workers, number of date folders).",
    )
    return parser.parse_args()


def folder_date_key(folder_name: str):
    try:
        return (0, datetime.strptime(folder_name, "%m_%d_%Y").date())
    except ValueError:
        return (1, folder_name)


# ----------------------------------------------------------------------------
# Scanning phase (parallel over date folders)
# ----------------------------------------------------------------------------


def scan_one_folder(subfolder: Path):
    """Scan a single date folder. Returns a list of
    (key, folder_name, pq_path, nodes_set) records, plus warnings.
    Runs in a worker process, so it returns picklable primitives only.
    """
    records = []
    warnings = []
    if not subfolder.is_dir():
        return records, warnings

    for json_path in sorted(subfolder.glob("*.json")):
        try:
            with open(json_path) as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            warnings.append(f"[WARN] Could not read {json_path}: {e}")
            continue

        jobid = meta.get("jobid")
        userid = meta.get("userid")
        if jobid is None or userid is None:
            warnings.append(f"[WARN] Missing jobid/userid in {json_path}")
            continue

        stem = json_path.stem
        pq_path = subfolder / f"{stem}.pq"
        if not pq_path.exists():
            warnings.append(f"[WARN] Missing .pq for {stem}, skipping this file.")
            continue

        nodes = set()
        for e in meta.get("entries", []):
            nodes.update(e.get("nodelist", []))

        records.append(((str(jobid), str(userid)), subfolder.name, str(pq_path), nodes))

    return records, warnings


def scan_jobs(root: Path, scan_workers: int):
    """Parallel scan of all date folders, merged into the same structure as before:
    jobs: dict[key] -> {"pq_files": [(folder_name, pq_path), ...], "nodes": set}
    """
    subfolders = [p for p in sorted(root.iterdir()) if p.is_dir()]
    jobs = defaultdict(lambda: {"pq_files": [], "nodes": set()})

    if scan_workers <= 1 or len(subfolders) <= 1:
        results = [scan_one_folder(sf) for sf in subfolders]
    else:
        with ProcessPoolExecutor(max_workers=scan_workers) as ex:
            results = list(ex.map(scan_one_folder, subfolders))

    for records, warnings in results:
        for w in warnings:
            print(w)
        for key, folder_name, pq_path, nodes in records:
            entry = jobs[key]
            entry["pq_files"].append((folder_name, pq_path))
            entry["nodes"].update(nodes)

    return jobs


# ----------------------------------------------------------------------------
# Sharding helpers
# ----------------------------------------------------------------------------


def hash_key(key) -> int:
    s = "\x00".join(str(part) for part in key)
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)


def shard_index(key, num_shards: int) -> int:
    return hash_key(key) % num_shards


def shard_output_path(out_dir: Path, base_name: str, shard: int, num_shards: int) -> Path:
    if num_shards == 1:
        return out_dir / base_name
    p = Path(base_name)
    stem, suffix = p.stem, p.suffix or ".pkl"
    width = len(str(num_shards))
    return out_dir / f"{stem}_{shard:0{width}d}of{num_shards:0{width}d}{suffix}"


def category_keys(jobs: dict, category: str):
    keys = []
    for key, info in jobs.items():
        n_nodes = len(info["nodes"])
        if category == "multi" and n_nodes <= 1:
            continue
        if category == "single" and n_nodes != 1:
            continue
        keys.append(key)
    return keys


# ----------------------------------------------------------------------------
# Reading phase (parallel over shards)
# ----------------------------------------------------------------------------


def read_job_df(pq_files):
    """pq_files: list of (folder_name, pq_path_str), read in date order."""
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


def build_one_shard(task):
    """Build and write a single shard file, entirely inside a worker process.

    task = (out_path_str, jobs_for_this_shard)
    where jobs_for_this_shard = list of (key, pq_files_list).

    Returns (out_path_str, n_jobs_written). No DataFrames cross the process
    boundary -- the worker writes its own file.
    """
    out_path, jobs_for_shard = task
    job_to_df = {}
    for key, pq_files in jobs_for_shard:
        df = read_job_df(pq_files)
        if df is not None:
            job_to_df[key] = df

    with open(out_path, "wb") as f:
        pickle.dump(job_to_df, f, protocol=pickle.HIGHEST_PROTOCOL)
    n = len(job_to_df)
    del job_to_df
    return out_path, n


def build_category_parallel(jobs, category, out_dir, base_name, num_shards, workers):
    keys = category_keys(jobs, category)

    # Group keys by shard, carrying only the pq_files each key needs so the
    # tasks stay lightweight to pickle across processes.
    shard_buckets = defaultdict(list)
    for key in keys:
        shard = shard_index(key, num_shards)
        shard_buckets[shard].append((key, jobs[key]["pq_files"]))

    tasks = []
    for shard in range(num_shards):
        out_path = str(shard_output_path(out_dir, base_name, shard, num_shards))
        tasks.append((out_path, shard_buckets.get(shard, [])))

    total = 0
    if workers <= 1 or len(tasks) <= 1:
        for t in tasks:
            path, n = build_one_shard(t)
            print(f"Saved {n:>6} jobs to {path}")
            total += n
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as ex:
            futures = {ex.submit(build_one_shard, t): t[0] for t in tasks}
            for fut in as_completed(futures):
                path, n = fut.result()
                print(f"Saved {n:>6} jobs to {path}")
                total += n

    print(f"Category '{category}': {total} jobs across {num_shards} shard(s).")


def main():
    import os

    args = parse_args()
    root = args.input_dir
    if not root.is_dir():
        raise SystemExit(f"[ERROR] Input directory not found: {root}")
    if args.num_shards < 1:
        raise SystemExit("[ERROR] --num-shards must be >= 1")

    workers = args.workers or os.cpu_count() or 1
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    name_map = {"all": args.all_name, "multi": args.multi_name, "single": args.single_name}
    categories = [args.category] if args.category else ["all", "multi", "single"]

    n_folders = sum(1 for p in root.iterdir() if p.is_dir())
    scan_workers = args.scan_workers or min(workers, max(1, n_folders))

    jobs = scan_jobs(root, scan_workers)
    print(
        f"Discovered {len(jobs)} unique (jobid, userid) jobs "
        f"(scanned with {scan_workers} worker(s))."
    )
    multi_folder = sum(1 for info in jobs.values() if len({fn for fn, _ in info["pq_files"]}) > 1)
    print(f"  of which span >1 folder (merged): {multi_folder}")

    for cat in categories:
        build_category_parallel(jobs, cat, out_dir, name_map[cat], args.num_shards, workers)


if __name__ == "__main__":
    main()
