"""
extract_timeseries.py

Extracts the mean BOLD time series from the largest activation cluster
that overlaps with each ROI mask (produced by run_featquery.py).

For each .feat dir + ROI combination:
  - Finds the largest cluster (by ROI overlap) in cluster_mask_zstat1.nii.gz
  - Averages filtered_func_data.nii.gz across all voxels in cluster ∩ ROI
  - Saves a CSV with columns: timepoint, mean_bold
  - Saves a QC mask (_mask.nii.gz) showing the exact voxels extracted from

Usage:
    python extract_timeseries.py
    python extract_timeseries.py --dry-run
"""

import os
import glob
import argparse
import numpy as np
import nibabel as nib
import pandas as pd
from config_extract import FEAT_BASE, OUTPUT_BASE, ROI_TASK_MAP, SUBJECTS


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


def get_output_paths(feat_dir, roi_name, output_base):
    """Return (csv_path, mask_path) for a given feat dir and ROI."""
    feat_name = os.path.basename(feat_dir).replace(".feat", "")
    out_dir   = os.path.join(output_base, roi_name, "timeseries")
    csv_path  = os.path.join(out_dir, f"{feat_name}_{roi_name}_timeseries.csv")
    mask_path = os.path.join(out_dir, f"{feat_name}_{roi_name}_mask.nii.gz")
    return csv_path, mask_path


def extract(feat_dir, roi_name, csv_path, mask_path, dry_run=False):
    """
    Extract mean BOLD time series from largest cluster within ROI.
    Returns status string.
    """
    cluster_mask_path  = os.path.join(feat_dir, "cluster_mask_zstat1.nii.gz")
    cluster_txt_path   = os.path.join(feat_dir, "cluster_zstat1.txt")
    filtered_func_path = os.path.join(feat_dir, "filtered_func_data.nii.gz")
    roi_mask_path      = os.path.join(feat_dir, f"{roi_name}.featquery", "mask.nii.gz")

    # Check required files
    required = {
        "cluster_mask"  : cluster_mask_path,
        "cluster_txt"   : cluster_txt_path,
        "filtered_func" : filtered_func_path,
        "roi_mask"      : roi_mask_path,
    }
    for label, path in required.items():
        if not os.path.exists(path):
            print(f"    [SKIP] Missing {label}: {path}")
            return f"skip_missing_{label}"

    if os.path.exists(csv_path):
        print(f"    [SKIP] Already exists: {csv_path}")
        return "already_done"

    if dry_run:
        print(f"    [DRY RUN] Would extract -> {csv_path}")
        return "dry_run"

    # Load images
    cluster_mask_img  = nib.load(cluster_mask_path)
    cluster_mask_data = cluster_mask_img.get_fdata()
    roi_bin           = (nib.load(roi_mask_path).get_fdata() > 0.5).astype(int)

    if roi_bin.sum() == 0:
        print(f"    [SKIP] ROI mask is empty after thresholding")
        return "skip_empty_roi"

    # Read cluster table
    cluster_df = pd.read_csv(cluster_txt_path, sep="\t")
    cluster_df.columns = [c.strip() for c in cluster_df.columns]

    # Find cluster with most overlap with ROI
    best_idx, best_overlap = None, 0
    for _, row in cluster_df.iterrows():
        cidx    = int(row["Cluster Index"])
        overlap = int(np.sum((cluster_mask_data == cidx) * roi_bin))
        if overlap > best_overlap:
            best_overlap = overlap
            best_idx     = cidx

    if best_idx is None or best_overlap == 0:
        print(f"    [SKIP] No clusters overlap with ROI mask")
        return "skip_no_overlap"

    print(f"    Cluster {best_idx} selected — {best_overlap} voxels overlap with ROI")

    # Build combined mask (cluster ∩ ROI)
    combined_mask = ((cluster_mask_data == best_idx) * roi_bin).astype(np.int16)
    n_vox         = int(combined_mask.sum())

    # Save QC mask
    os.makedirs(os.path.dirname(mask_path), exist_ok=True)
    nib.save(nib.Nifti1Image(combined_mask, cluster_mask_img.affine), mask_path)

    # Extract mean time series across mask voxels
    func_data = nib.load(filtered_func_path).get_fdata()
    mean_ts   = func_data[combined_mask.astype(bool), :].mean(axis=0)
    n_tp      = len(mean_ts)

    # Save CSV
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    pd.DataFrame({
        "timepoint": np.arange(n_tp),
        "mean_bold": mean_ts,
    }).to_csv(csv_path, index=False)

    print(f"    Saved: {csv_path} ({n_vox} voxels, {n_tp} timepoints)")
    return "ok"


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main(dry_run=False):
    summary = []

    for roi_cfg in ROI_TASK_MAP:
        roi_name = roi_cfg["roi_name"]
        tasks    = roi_cfg["tasks"]

        print(f"\n{'='*70}")
        print(f"ROI  : {roi_name}")
        print(f"Tasks: {tasks}")
        print(f"{'='*70}")

        feat_dirs = find_feat_dirs(FEAT_BASE, tasks, subjects=SUBJECTS)
        print(f"  Found {len(feat_dirs)} matching .feat directories\n")

        for feat_dir in feat_dirs:
            feat_name         = os.path.basename(feat_dir)
            csv_path, mask_path = get_output_paths(feat_dir, roi_name, OUTPUT_BASE)

            print(f"  -- {feat_name}")

            status = extract(
                feat_dir  = feat_dir,
                roi_name  = roi_name,
                csv_path  = csv_path,
                mask_path = mask_path,
                dry_run   = dry_run,
            )
            summary.append({"roi": roi_name, "feat": feat_name, "status": status})

    # Summary
    df = pd.DataFrame(summary)
    if not df.empty:
        print(f"\n{'='*70}")
        print("EXTRACTION SUMMARY")
        print(df.groupby(["roi", "status"]).size().to_string())
        failed = df[~df["status"].isin(["ok", "already_done", "dry_run"])]
        if not failed.empty:
            print(f"\nSkipped / failed:")
            print(failed.to_string(index=False))

        if not dry_run:
            os.makedirs(OUTPUT_BASE, exist_ok=True)
            df.to_csv(os.path.join(OUTPUT_BASE, "extraction_summary.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract mean BOLD time series from ROI clusters.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview without executing")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
