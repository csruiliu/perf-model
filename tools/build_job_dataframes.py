#!/usr/bin/env python3
"""
Generate three pickle files, each containing a dict:  job_to_df[jobid] -> dcgm_df

Processes one category per pass to keep peak memory low.
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


def get_node_count(meta: dict) -> int:
    nodes = set()
    for entry in meta.get("entries", []):
        nodes.update(entry.get("nodelist", []))
    return len(nodes)


def iter_jobs(root: Path):
    """Yield (jobid, json_path, pq_path, n_nodes) for every valid job."""
    for subfolder in sorted(root.iterdir()):
        if not subfolder.is_dir():
            continue
        for json_path in sorted(subfolder.glob("*.json")):
            stem = json_path.stem
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
            yield jobid, pq_path, get_node_count(meta)


def build_category(root: Path, category: str) -> dict:
    """Build the job_to_df dict for one category."""
    job_to_df = {}
    for jobid, pq_path, n_nodes in iter_jobs(root):
        if category == "multi" and n_nodes <= 1:
            continue
        if category == "single" and n_nodes != 1:
            continue
        try:
            job_to_df[jobid] = pd.read_parquet(pq_path)
        except Exception as e:
            print(f"[WARN] Could not read {pq_path}: {e}")
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

    for cat in categories:
        d = build_category(root, cat)
        path = out_dir / name_map[cat]
        with open(path, "wb") as f:
            pickle.dump(d, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved {len(d):>6} jobs to {path}")
        del d  # free memory before the next pass


if __name__ == "__main__":
    main()
