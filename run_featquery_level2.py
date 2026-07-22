"""
run_featquery_level2.py

Runs FSL featquery on all matching cope*.feat directories inside .gfeat
(group/level-2) directories, for each ROI/task combination defined in
config_featquery_level2.py.

Checks which stats images actually exist in each cope*.feat dir before
building the command, so copes with fewer stats images won't crash.

Usage:
    python run_featquery_level2.py
    python run_featquery_level2.py --dry-run
    python run_featquery_level2.py --subject sub-CP016
    python run_featquery_level2.py --workers 4
"""

import os
import argparse
import subprocess
import pandas as pd
from multiprocessing import Pool, cpu_count
from config.config_featquery_level2 import (
    FEATQUERY_BIN, FEAT_BASE, STATS_IMAGES,
    FEATQUERY_FLAGS, ROI_TASK_MAP, SUBJECTS
)
from util.discovery import find_cope_dirs, get_existing_stats


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def run_featquery(cope_dir, roi_path, roi_name, stats_images, flags, dry_run=False):
    """Build and run the featquery command for one cope*.feat dir + ROI."""
    output_name   = f"{roi_name}.featquery"
    featquery_dir = os.path.join(cope_dir, output_name)
    mask_out      = os.path.join(featquery_dir, "mask.nii.gz")

    if os.path.exists(mask_out):
        print(f"    [SKIP] Already exists: {featquery_dir}")
        return "already_done"

    available_stats = get_existing_stats(cope_dir, stats_images)
    if not available_stats:
        print(f"    [SKIP] No stats images found in {cope_dir}")
        return "skip_no_stats"

    cmd = (
        [FEATQUERY_BIN, "1", cope_dir, str(len(available_stats))]
        + available_stats
        + [output_name]
        + flags
        + [roi_path]
    )

    print(f"    [CMD] {' '.join(cmd)}")

    if dry_run:
        return "dry_run"

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=cope_dir)
        if result.returncode != 0:
            print(f"    [ERROR] returncode={result.returncode}")
            print(f"    [STDERR] {result.stderr.strip()}")
            return "failed"
        print(f"    [OK] Output: {featquery_dir}")
        return "ok"
    except Exception as e:
        print(f"    [EXCEPTION] {e}")
        return "failed"


def _run_one(args):
    """Top-level wrapper so a cope_dir + ROI job is picklable for Pool.map."""
    gfeat_dir, cope_dir, roi_path, roi_name, stats_images, flags, dry_run = args
    label = os.path.join(os.path.basename(gfeat_dir), os.path.basename(cope_dir))
    print(f"  -- {label}")

    if not os.path.exists(os.path.join(cope_dir, "design.fsf")):
        print(f"    [SKIP] No design.fsf found")
        return {"roi": roi_name, "cope": label, "status": "skip_no_fsf"}

    status = run_featquery(
        cope_dir     = cope_dir,
        roi_path     = roi_path,
        roi_name     = roi_name,
        stats_images = stats_images,
        flags        = flags,
        dry_run      = dry_run,
    )
    return {"roi": roi_name, "cope": label, "status": status}


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main(dry_run=False, subject_override=None, n_workers=1):
    subjects = subject_override if subject_override else SUBJECTS
    summary  = []

    for roi_cfg in ROI_TASK_MAP:
        roi_path = roi_cfg["roi_path"]
        roi_name = roi_cfg["roi_name"]
        tasks    = roi_cfg["tasks"]
        copes    = roi_cfg["copes"]

        print(f"\n{'='*70}")
        print(f"ROI  : {roi_name}")
        print(f"Tasks: {tasks}")
        print(f"Copes: {copes}")
        print(f"{'='*70}")

        if not os.path.exists(roi_path):
            print(f"  [ERROR] ROI file not found: {roi_path} — skipping")
            continue

        cope_dirs = find_cope_dirs(FEAT_BASE, tasks, copes, subjects=subjects)
        print(f"  Found {len(cope_dirs)} matching cope directories")
        if n_workers > 1:
            print(f"  Running with {n_workers} parallel workers\n")
        else:
            print()

        job_args = [
            (gfeat_dir, cope_dir, roi_path, roi_name, STATS_IMAGES, FEATQUERY_FLAGS, dry_run)
            for gfeat_dir, cope_dir in cope_dirs
        ]

        if n_workers > 1 and not dry_run:
            with Pool(processes=n_workers) as pool:
                results = pool.map(_run_one, job_args)
        else:
            results = [_run_one(a) for a in job_args]

        summary.extend(results)

    df = pd.DataFrame(summary)
    if not df.empty:
        print(f"\n{'='*70}")
        print("FEATQUERY SUMMARY")
        print(df.groupby(["roi", "status"]).size().to_string())
        failed = df[df["status"] == "failed"]
        if not failed.empty:
            print(f"\nFailed runs:")
            print(failed.to_string(index=False))

    if dry_run:
        print(f"\nDRY RUN COMPLETE — nothing executed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch FSL featquery runner (level 2 / .gfeat).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing them")
    parser.add_argument("--subject", type=str, default=None,
                        help="Process a single subject (e.g. sub-CP016). "
                             "Overrides SUBJECTS in config.")
    parser.add_argument("--workers", type=int, default=1,
                        help=f"Number of parallel workers (default: 1, max: {cpu_count()})")
    args = parser.parse_args()

    subject_list = [args.subject] if args.subject else None
    main(dry_run=args.dry_run, subject_override=subject_list, n_workers=args.workers)
