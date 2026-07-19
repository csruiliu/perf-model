"""
Detect jobs (identified by the JSON's own jobid + userid) that appear across
multiple date folders, and record exactly which folders each one spans.

Reads ids from inside each JSON rather than parsing filenames, since job ids
may be Slurm array tasks (e.g. '49735483_362') and user ids may contain
underscores (e.g. 'yu_yao').
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description="Find (jobid, userid) pairs that span multiple date folders."
    )
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
        default=Path("cross_folder_jobs.csv"),
        help="CSV listing every cross-folder job and its folders (default: cross_folder_jobs.csv).",
    )
    p.add_argument(
        "--show",
        type=int,
        default=20,
        help="How many example duplicated pairs to print to stdout "
        "(default: 20). The CSV always contains all of them.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    root = args.input_dir
    if not root.is_dir():
        raise SystemExit(f"[ERROR] Input directory not found: {root}")

    # (jobid, userid) -> set of date-folder names it appears in
    pair_to_folders = defaultdict(set)

    total_files = 0
    bad_files = 0

    for subfolder in sorted(root.iterdir()):
        if not subfolder.is_dir():
            continue
        for json_path in sorted(subfolder.glob("*.json")):
            total_files += 1
            try:
                with open(json_path) as f:
                    meta = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                print(f"[WARN] Could not read {json_path}: {e}")
                bad_files += 1
                continue

            jobid = meta.get("jobid")
            userid = meta.get("userid")
            if jobid is None or userid is None:
                print(f"[WARN] Missing jobid/userid in {json_path}")
                bad_files += 1
                continue

            pair_to_folders[(str(jobid), str(userid))].add(subfolder.name)

    # Jobs spanning more than one date folder
    multi_folder = {k: v for k, v in pair_to_folders.items() if len(v) > 1}

    # --- Write the full list to CSV ---
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["jobid", "userid", "num_folders", "folders"])
        # Sort by how many folders each spans (descending), then by jobid
        for (jobid, userid), folders in sorted(
            multi_folder.items(), key=lambda kv: (-len(kv[1]), kv[0])
        ):
            writer.writerow(
                [
                    jobid,
                    userid,
                    len(folders),
                    ";".join(sorted(folders)),  # semicolon-separated folder list
                ]
            )

    # --- Report to stdout ---
    print("\n===== Summary =====")
    print(f"Total JSON files scanned:            {total_files}")
    print(f"Unreadable / missing-id files:       {bad_files}")
    print(f"Unique (jobid, userid) pairs:        {len(pair_to_folders)}")
    print(f"Pairs spanning >1 date folder:       {len(multi_folder)}")
    print(f"\nFull list written to: {args.output_csv}")

    if multi_folder:
        print(f"\n===== Examples (showing up to {args.show}; see CSV for all) =====")
        for i, ((jobid, userid), folders) in enumerate(
            sorted(multi_folder.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        ):
            if i >= args.show:
                break
            print(f"jobid={jobid!r} userid={userid!r} -> {len(folders)} folders: {sorted(folders)}")


if __name__ == "__main__":
    main()
