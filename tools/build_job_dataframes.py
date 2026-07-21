#!/usr/bin/env python3
"""
Generate pickle files, each containing a dict:
    job_to_df[(jobid, userid)] -> dcgm_df

A "job" is the (jobid, userid) pair read from INSIDE each JSON -- not from
the filename -- so array tasks (jobid '49735483_362') and underscore
userids ('yu_yao') are handled correctly.

A job whose files are spread across multiple date folders is ONE job: all
of its .pq files are read and concatenated, ordered by folder date
(folder names are 'MM_DD_YYYY').

Only a fixed subset of DCGM columns is kept (KEEP_COLUMNS); the projection is
pushed down to the Parquet reader so unwanted columns are never read off disk.

Within each category the jobs are sharded across N pickle files
(--num-shards), and shards are built in parallel across worker processes
(--workers). Each worker builds and writes one shard end-to-end, so no large
DataFrames cross the process boundary.

Designed to run under a SLURM allocation (e.g. Perlmutter salloc): worker
count defaults to the cores actually available to the process, and each build
worker is recycled after finishing its shard to keep peak memory bounded.
"""

import argparse
import gc
import hashlib
import json
import os
import pickle
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd

# Columns to keep from each DCGM DataFrame. Everything else is discarded at
# read time via PyArrow column pushdown, so unwanted columns are never loaded.
KEEP_COLUMNS = [
    "gpu_id",
    "nersc_ldms_dcgm_gr_engine_active",
    "nersc_ldms_dcgm_dram_active",
    "nersc_ldms_dcgm_sm_occupancy",
    "nersc_ldms_dcgm_tensor_active",
    "nersc_ldms_dcgm_fp16_active",
    "nersc_ldms_dcgm_fp32_active",
    "nersc_ldms_dcgm_fp64_active",
    "nersc_ldms_dcgm_nvlink_rx_bytes",
    "nersc_ldms_dcgm_nvlink_tx_bytes",
    "nersc_ldms_dcgm_pcie_rx_bytes",
    "nersc_ldms_dcgm_pcie_tx_bytes",
]


# ----------------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------------


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
    parser.add_argument(
        "-n",
        "--num-shards",
        type=int,
        default=1,
        help="Split each category's output across this many pickle files (shards). "
        "Use a value >= --workers (or a small multiple) to keep all workers busy.",
    )
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes for building shards. "
        "Default: cores available to this process (SLURM-aware).",
    )
    parser.add_argument(
        "--scan-workers",
        type=int,
        default=None,
        help="Processes for the JSON-scanning phase. "
        "Default: min(workers, number of date folders).",
    )
    return parser.parse_args()


# ----------------------------------------------------------------------------
# Core-count detection (SLURM / Perlmutter aware)
# ----------------------------------------------------------------------------


def available_cores() -> int:
    """Cores this process may actually use.

    On SLURM/Perlmutter, os.cpu_count() reports the whole physical node,
    ignoring the cgroup allocation. Prefer SLURM's own hint, then the affinity
    mask, and only fall back to cpu_count().
    """
    slurm_cpus = os.environ.get("SLURM_CPUS_ON_NODE")
    if slurm_cpus:
        try:
            return int(slurm_cpus)
        except ValueError:
            pass
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # not available on all platforms
        return os.cpu_count() or 1


# ----------------------------------------------------------------------------
# Folder-date parsing
# ----------------------------------------------------------------------------


def folder_date_key(folder_name: str):
    """Parse a 'MM_DD_YYYY' folder name into a date for chronological sorting.

    Falls back to the raw string (sorted last) if it doesn't match, so an
    unexpected folder name never crashes the run.
    """
    try:
        return (0, datetime.strptime(folder_name, "%m_%d_%Y").date())
    except ValueError:
        return (1, folder_name)


# ----------------------------------------------------------------------------
# Scanning phase (parallel over date folders)
# ----------------------------------------------------------------------------


def scan_one_folder(subfolder: Path):
    """Scan a single date folder.

    Returns (records, warnings) where each record is
    (key, folder_name, pq_path_str, nodes_set). Runs in a worker process, so
    it returns only picklable primitives.
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
    """Parallel scan of all date folders.

    Returns:
        jobs: dict[(jobid, userid)] -> {
            "pq_files": list of (folder_name, pq_path_str),  # unsorted
            "nodes": set of node names (union across folders/entries),
        }
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
    """Deterministic non-negative hash of a (jobid, userid) tuple.

    Python's built-in hash() is randomized per-process (PYTHONHASHSEED), so we
    use a stable hash to keep shard assignment reproducible across runs.
    """
    s = "\x00".join(str(part) for part in key)
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)


def shard_index(key, num_shards: int) -> int:
    """Deterministically assign a job key to a shard in [0, num_shards)."""
    return hash_key(key) % num_shards


def shard_output_path(out_dir: Path, base_name: str, shard: int, num_shards: int) -> Path:
    """Build a shard filename, e.g. all_jobs.pkl -> all_jobs_003of010.pkl.

    With num_shards == 1 the original name is used unchanged.
    """
    if num_shards == 1:
        return out_dir / base_name
    p = Path(base_name)
    stem, suffix = p.stem, p.suffix or ".pkl"
    width = len(str(num_shards))
    return out_dir / f"{stem}_{shard:0{width}d}of{num_shards:0{width}d}{suffix}"


def category_keys(jobs: dict, category: str):
    """Return the list of job keys belonging to a category."""
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
    """Read and concatenate a job's .pq files (date order), keeping only KEEP_COLUMNS.

    The column projection is pushed down to the Parquet reader so unwanted
    columns are never read off disk or materialized in memory.

    pq_files: list of (folder_name, pq_path_str), sorted chronologically by the
    parsed MM_DD_YYYY folder date so the resulting DataFrame is in date order.
    """
    frames = []
    for folder_name, pq_path in sorted(pq_files, key=lambda t: folder_date_key(t[0])):
        try:
            df = pd.read_parquet(pq_path, columns=KEEP_COLUMNS)
        except Exception as e:
            # A file missing an expected column, or unreadable, is skipped with
            # a clear message rather than killing the whole shard.
            print(f"[WARN] Could not read {pq_path} with requested columns: {e}")
            continue
        frames.append(df)

    if not frames:
        return None
    if len(frames) == 1:
        return frames[0]
    return pd.concat(frames, ignore_index=True)


def build_one_shard(task):
    """Build and write a single shard file, entirely inside a worker process.

    task = (out_path_str, jobs_for_this_shard)
    where jobs_for_this_shard = list of (key, pq_files_list).

    Reads jobs one at a time and stores each DataFrame into the shard dict.
    Returns (out_path_str, n_jobs_written); no DataFrames cross the process
    boundary -- the worker writes its own file.
    """
    out_path, jobs_for_shard = task
    job_to_df = {}
    for key, pq_files in jobs_for_shard:
        try:
            df = read_job_df(pq_files)
        except MemoryError:
            # Re-raise as a clear, picklable error so the parent sees the cause.
            raise RuntimeError(f"MemoryError while reading job {key} for {out_path}")
        if df is not None:
            job_to_df[key] = df

    with open(out_path, "wb") as f:
        pickle.dump(job_to_df, f, protocol=pickle.HIGHEST_PROTOCOL)
    n = len(job_to_df)
    del job_to_df
    gc.collect()
    return out_path, n


def build_category_parallel(jobs, category, out_dir, base_name, num_shards, workers):
    """Build one category's job_to_df dict, sharded across files, in parallel."""
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
        # max_tasks_per_child=1 recycles each worker after its shard, releasing
        # native (PyArrow) memory so peak usage stays bounded.
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks)), max_tasks_per_child=1) as ex:
            futures = {ex.submit(build_one_shard, t): t[0] for t in tasks}
            for fut in as_completed(futures):
                shard_path = futures[fut]
                try:
                    path, n = fut.result()
                except Exception as e:
                    print(f"[ERROR] Shard {shard_path} failed: {e!r}")
                    raise
                print(f"Saved {n:>6} jobs to {path}")
                total += n

    print(f"Category '{category}': {total} jobs across {num_shards} shard(s).")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main():
    args = parse_args()

    root = args.input_dir
    if not root.is_dir():
        raise SystemExit(f"[ERROR] Input directory not found: {root}")
    if args.num_shards < 1:
        raise SystemExit("[ERROR] --num-shards must be >= 1")

    workers = args.workers or available_cores()
    print(f"Using {workers} build worker process(es).")

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    name_map = {"all": args.all_name, "multi": args.multi_name, "single": args.single_name}
    categories = [args.category] if args.category else ["all", "multi", "single"]

    n_folders = sum(1 for p in root.iterdir() if p.is_dir())
    scan_workers = args.scan_workers or min(workers, max(1, n_folders))

    # Scan once; reuse the grouping for every category pass.
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
