"""
run_featquery.py

Runs FSL featquery on all matching .feat directories for each ROI/task
combination defined in config_featquery.py.

Checks which stats images actually exist in each .feat dir before
building the command, so subjects with fewer PEs won't crash.

Usage:
    python run_featquery.py
    python run_featquery.py --dry-run
"""

import os
import glob
import argparse
import subprocess
import pandas as pd
from config_featquery import (
    FEATQUERY_BIN, FEAT_BASE, STATS_IMAGES,
    FEATQUERY_FLAGS, ROI_TASK_MAP, SUBJECTS
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def find_feat_dirs(base_dir, task_filters, subjects=None):
    """
    Find all .feat dirs whose basename contains any task in task_filters.
    If subjects is a list, only include dirs matching those subject IDs.
    """
    all_dirs = sorted(glob.glob(os.path.join(base_dir, "sub-*", "ses-*", "*.feat")))
    filtered = [d for d in all_dirs
                if any(task in os.path.basename(d) for task in task_filters)]
    if subjects is not None:
        filtered = [d for d in filtered
                    if any(sub in os.path.basename(d) for sub in subjects)]
    return filtered


def get_existing_stats(feat_dir, stats_images):
    """Return only the stats image paths that actually exist in feat_dir."""
    existing = []
    for img in stats_images:
        # FSL stores without extension internally; check both with and without
        path_no_ext = os.path.join(feat_dir, img)
        path_nii_gz = os.path.join(feat_dir, img + ".nii.gz")
        if os.path.exists(path_nii_gz) or os.path.exists(path_no_ext):
            existing.append(img)
    return existing


def run_featquery(feat_dir, roi_path, roi_name, stats_images,
                  flags, dry_run=False):
    """
    Build and run the featquery command for one .feat dir + ROI.

    Command format:
        featquery 1 <feat_dir> <n_stats> <stats...> <output_name> \
            [flags] -b <roi_path>
    """
    output_name   = f"{roi_name}.featquery"
    featquery_dir = os.path.join(feat_dir, output_name)
    mask_out      = os.path.join(featquery_dir, "mask.nii.gz")

    if os.path.exists(mask_out):
        print(f"    [SKIP] Already exists: {featquery_dir}")
        return "already_done"

    # Only request stats images that exist in this feat dir
    available_stats = get_existing_stats(feat_dir, stats_images)
    if not available_stats:
        print(f"    [SKIP] No stats images found in {feat_dir}")
        return "skip_no_stats"

    cmd = (
        [FEATQUERY_BIN,
         "1",
         feat_dir,
         str(len(available_stats))]
        + available_stats
        + [output_name]
        + flags
        + [roi_path]
    )

    print(f"    [CMD] {' '.join(cmd)}")

    if dry_run:
        return "dry_run"

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=feat_dir
        )
        if result.returncode != 0:
            print(f"    [ERROR] returncode={result.returncode}")
            print(f"    [STDERR] {result.stderr.strip()}")
            return "failed"
        print(f"    [OK] Output: {featquery_dir}")
        return "ok"
    except Exception as e:
        print(f"    [EXCEPTION] {e}")
        return "failed"


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main(dry_run=False):
    summary = []

    for roi_cfg in ROI_TASK_MAP:
        roi_path = roi_cfg["roi_path"]
        roi_name = roi_cfg["roi_name"]
        tasks    = roi_cfg["tasks"]

        print(f"\n{'='*70}")
        print(f"ROI  : {roi_name}")
        print(f"Tasks: {tasks}")
        print(f"{'='*70}")

        if not os.path.exists(roi_path):
            print(f"  [ERROR] ROI file not found: {roi_path} — skipping")
            continue

        feat_dirs = find_feat_dirs(FEAT_BASE, tasks, subjects=SUBJECTS)
        print(f"  Found {len(feat_dirs)} matching .feat directories\n")

        for feat_dir in feat_dirs:
            feat_name = os.path.basename(feat_dir)
            print(f"  -- {feat_name}")

            if not os.path.exists(os.path.join(feat_dir, "design.fsf")):
                print(f"    [SKIP] No design.fsf found")
                summary.append({"roi": roi_name, "feat": feat_name, "status": "skip_no_fsf"})
                continue

            status = run_featquery(
                feat_dir     = feat_dir,
                roi_path     = roi_path,
                roi_name     = roi_name,
                stats_images = STATS_IMAGES,
                flags        = FEATQUERY_FLAGS,
                dry_run      = dry_run,
            )
            summary.append({"roi": roi_name, "feat": feat_name, "status": status})

    # Print summary
    df = pd.DataFrame(summary)
    if not dry_run and not df.empty:
        print(f"\n{'='*70}")
        print("FEATQUERY SUMMARY")
        print(df.groupby(["roi", "status"]).size().to_string())
        failed = df[df["status"] == "failed"]
        if not failed.empty:
            print(f"\nFailed runs:")
            print(failed.to_string(index=False))
    else:
        print(f"\nDRY RUN COMPLETE — nothing executed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch FSL featquery runner.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing them")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
