"""
extract_timeseries.py

Extracts the mean BOLD time series from the largest activation cluster
that overlaps with each ROI mask (produced by run_featquery.py).

For each .feat dir + ROI combination:
  - Finds the largest cluster (by ROI overlap) in cluster_mask_zstat1.nii.gz
  - Averages filtered_func_data.nii.gz across all voxels in cluster ∩ ROI
  - Saves a CSV with columns:
      timepoint, mean_bold,
      rot_x, rot_y, rot_z, trans_x, trans_y, trans_z, fd
  - Saves a QC mask (_mask.nii.gz) showing the exact voxels extracted from

Also saves a responder summary CSV with per-subject ROI voxel counts, mean
z-stat (raw and suprathreshold-only), suprathreshold voxel count, and
responder status (responder = any cluster voxels within ROI > 0).

Usage:
    python extract_timeseries.py
    python extract_timeseries.py --dry-run
"""

import os
import argparse
import numpy as np
import nibabel as nib
import pandas as pd
from config.config_extract import FEAT_BASE, OUTPUT_BASE, ROI_TASK_MAP, SUBJECTS
from util.discovery import find_feat_dirs
from util.motion import load_motion_params, motion_columns
from util.zstat import compute_zstat_metrics


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

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

    Returns a dict with keys:
        status                     : string status code
        roi_vox                    : number of voxels in the native-space ROI mask
        extraction_vox             : number of voxels in the largest cluster ∩ ROI mask
        all_cluster_vox            : number of voxels across all significant clusters ∩ ROI
        mean_zstat_roi             : mean raw z-stat across the whole ROI mask
        peak_zstat_roi             : max raw z-stat across the whole ROI mask
        peak_vox_roi_x/y/z         : voxel coords (native space) of that peak
        mean_zstat_extraction      : mean raw z-stat across the extraction mask
        peak_zstat_extraction      : max raw z-stat across the extraction mask
        peak_vox_extraction_x/y/z  : voxel coords (native space) of that peak
        n_suprathreshold_vox       : voxels in the ROI surviving cluster-corrected threshold
        mean_suprathreshold_zstat  : mean thresholded z-stat among suprathreshold voxels
    """
    cluster_mask_path  = os.path.join(feat_dir, "cluster_mask_zstat1.nii.gz")
    cluster_txt_path   = os.path.join(feat_dir, "cluster_zstat1.txt")
    filtered_func_path = os.path.join(feat_dir, "filtered_func_data.nii.gz")
    roi_mask_path      = os.path.join(feat_dir, f"{roi_name}.featquery", "mask.nii.gz")

    result = {
        "status"                   : None,
        "roi_vox"                  : None,
        "extraction_vox"           : None,
        "all_cluster_vox"          : None,
        "mean_zstat_roi"           : np.nan,
        "peak_zstat_roi"           : np.nan,
        "peak_vox_roi_x"           : None,
        "peak_vox_roi_y"           : None,
        "peak_vox_roi_z"           : None,
        "mean_zstat_extraction"    : np.nan,
        "peak_zstat_extraction"    : np.nan,
        "peak_vox_extraction_x"    : None,
        "peak_vox_extraction_y"    : None,
        "peak_vox_extraction_z"    : None,
        "n_suprathreshold_vox"     : None,
        "mean_suprathreshold_zstat": np.nan,
    }

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
            result["status"] = f"skip_missing_{label}"
            return result

    if os.path.exists(csv_path):
        roi_bin            = (nib.load(roi_mask_path).get_fdata() > 0.5).astype(int)
        cluster_mask_data  = nib.load(cluster_mask_path).get_fdata()
        extraction_mask    = None
        result["roi_vox"]         = int(roi_bin.sum())
        result["status"]          = "already_done"
        result["all_cluster_vox"] = int(((cluster_mask_data > 0) * roi_bin).sum())
        if os.path.exists(mask_path):
            extraction_mask = (nib.load(mask_path).get_fdata() > 0.5).astype(int)
            result["extraction_vox"] = int(extraction_mask.sum())
        result.update(compute_zstat_metrics(feat_dir, roi_bin, extraction_mask))
        print(f"    [SKIP] Already exists: {csv_path}")
        return result

    if dry_run:
        print(f"    [DRY RUN] Would extract -> {csv_path}")
        result["status"] = "dry_run"
        return result

    # Load images
    cluster_mask_img  = nib.load(cluster_mask_path)
    cluster_mask_data = cluster_mask_img.get_fdata()
    roi_bin           = (nib.load(roi_mask_path).get_fdata() > 0.5).astype(int)
    result["roi_vox"] = int(roi_bin.sum())

    if roi_bin.sum() == 0:
        print(f"    [SKIP] ROI mask is empty after thresholding")
        result["status"] = "skip_empty_roi"
        return result

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
        result["status"] = "skip_no_overlap"
        return result

    print(f"    Cluster {best_idx} selected — {best_overlap} voxels overlap with ROI")

    # Build combined mask (largest cluster ∩ ROI)
    combined_mask = ((cluster_mask_data == best_idx) * roi_bin).astype(np.int16)
    n_vox         = int(combined_mask.sum())
    result["extraction_vox"]  = n_vox
    result["all_cluster_vox"] = int(((cluster_mask_data > 0) * roi_bin).sum())
    result.update(compute_zstat_metrics(feat_dir, roi_bin, combined_mask))

    # Save QC mask
    os.makedirs(os.path.dirname(mask_path), exist_ok=True)
    nib.save(nib.Nifti1Image(combined_mask, cluster_mask_img.affine), mask_path)

    # Extract mean time series across mask voxels
    func_data = nib.load(filtered_func_path).get_fdata()
    mean_ts   = func_data[combined_mask.astype(bool), :].mean(axis=0)
    n_tp      = len(mean_ts)

    # Load motion parameters
    motion_params = load_motion_params(feat_dir)
    if motion_params is None:
        print(f"    [WARN] No .par file found — motion columns will be NaN")
    mot_cols = motion_columns(motion_params, n_tp)

    # Save CSV
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    pd.DataFrame({
        "timepoint": np.arange(n_tp),
        "mean_bold": mean_ts,
        **mot_cols,
    }).to_csv(csv_path, index=False)

    print(f"    Saved: {csv_path} ({n_vox} voxels, {n_tp} timepoints)")
    result["status"] = "ok"
    return result


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main(dry_run=False):
    summary    = []
    responders = []

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
            feat_name            = os.path.basename(feat_dir)
            csv_path, mask_path  = get_output_paths(feat_dir, roi_name, OUTPUT_BASE)

            print(f"  -- {feat_name}")

            result = extract(
                feat_dir  = feat_dir,
                roi_name  = roi_name,
                csv_path  = csv_path,
                mask_path = mask_path,
                dry_run   = dry_run,
            )

            summary.append({
                "roi"   : roi_name,
                "feat"  : feat_name,
                "status": result["status"],
            })

            if result["roi_vox"] is not None:
                responders.append({
                    "feat"                     : feat_name,
                    "roi"                      : roi_name,
                    "roi_vox"                  : result["roi_vox"],
                    "extraction_vox"           : result["extraction_vox"],
                    "all_cluster_vox"          : result["all_cluster_vox"],
                    "mean_zstat_roi"           : result["mean_zstat_roi"],
                    "peak_zstat_roi"           : result["peak_zstat_roi"],
                    "peak_vox_roi_x"           : result["peak_vox_roi_x"],
                    "peak_vox_roi_y"           : result["peak_vox_roi_y"],
                    "peak_vox_roi_z"           : result["peak_vox_roi_z"],
                    "mean_zstat_extraction"    : result["mean_zstat_extraction"],
                    "peak_zstat_extraction"    : result["peak_zstat_extraction"],
                    "peak_vox_extraction_x"    : result["peak_vox_extraction_x"],
                    "peak_vox_extraction_y"    : result["peak_vox_extraction_y"],
                    "peak_vox_extraction_z"    : result["peak_vox_extraction_z"],
                    "n_suprathreshold_vox"     : result["n_suprathreshold_vox"],
                    "mean_suprathreshold_zstat": result["mean_suprathreshold_zstat"],
                    "is_responder"             : (
                        result["extraction_vox"] is not None
                        and result["extraction_vox"] > 0
                    ),
                })

    # Extraction summary
    df_summary = pd.DataFrame(summary)
    if not df_summary.empty:
        print(f"\n{'='*70}")
        print("EXTRACTION SUMMARY")
        print(df_summary.groupby(["roi", "status"]).size().to_string())
        failed = df_summary[~df_summary["status"].isin(["ok", "already_done", "dry_run"])]
        if not failed.empty:
            print(f"\nSkipped / failed:")
            print(failed.to_string(index=False))

        if not dry_run:
            os.makedirs(OUTPUT_BASE, exist_ok=True)
            df_summary.to_csv(
                os.path.join(OUTPUT_BASE, "extraction_summary.csv"), index=False
            )

    # Responder summary
    if responders and not dry_run:
        df_resp = pd.DataFrame(responders)
        print(f"\n{'='*70}")
        print("RESPONDER SUMMARY")
        print(df_resp.groupby(["roi", "is_responder"]).size().rename("n_runs").to_string())
        resp_path = os.path.join(OUTPUT_BASE, "responder_summary.csv")
        df_resp.to_csv(resp_path, index=False)
        print(f"\nSaved: {resp_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract mean BOLD time series from ROI clusters.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview without executing")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
