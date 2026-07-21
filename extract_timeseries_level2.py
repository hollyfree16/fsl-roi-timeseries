"""
extract_timeseries_level2.py

Extracts the mean BOLD time series from the largest activation cluster
that overlaps with each ROI mask, from group/level-2 (.gfeat/cope*.feat)
outputs produced by run_featquery_level2.py.

For each cope*.feat dir + ROI combination:
  - Finds the largest cluster (by ROI overlap) in cluster_mask_zstat1.nii.gz,
    using the cluster table cluster_zstat1_std.txt (level 2 names this
    differently from level 1's cluster_zstat1.txt)
  - Averages filtered_func_data.nii.gz across all voxels in cluster ∩ ROI
  - Saves a CSV with columns:
      timepoint, mean_bold,
      rot_x, rot_y, rot_z, trans_x, trans_y, trans_z, fd
    Level-2 runs have no motion .par file, so the motion columns are NaN.
  - Saves a QC mask (_mask.nii.gz) showing the exact voxels extracted from

Also saves a responder summary CSV with per-subject ROI voxel counts, mean
z-stat (raw and suprathreshold-only), suprathreshold voxel count, and
responder status (responder = any cluster voxels within ROI > 0).

Usage:
    python extract_timeseries_level2.py
    python extract_timeseries_level2.py --dry-run
"""

import os
import glob
import argparse
import numpy as np
import nibabel as nib
import pandas as pd
from config_extract_level2 import FEAT_BASE, OUTPUT_BASE, ROI_TASK_MAP, SUBJECTS


# -----------------------------------------------------------------------------
# Motion helpers
# -----------------------------------------------------------------------------

def load_motion_params(cope_dir):
    """
    Load the FSL motion parameter file from the standard location.
    Returns an (N, 6) array [rotX, rotY, rotZ, transX, transY, transZ]
    or None if the file is not found (expected at level 2 — no .par file).
    """
    par_path = os.path.join(cope_dir, "mc", "prefiltered_func_data_mcf.par")
    if not os.path.exists(par_path):
        return None
    try:
        return np.loadtxt(par_path)
    except Exception:
        return None


def compute_fd(motion_params):
    """
    Compute framewise displacement from an (N, 6) motion parameter array.
    Rotations (first 3 columns) are converted from radians to mm using a
    50 mm head-radius assumption. Returns a 1-D array of length N (FD[0] = 0).
    """
    diff = np.diff(motion_params, axis=0, prepend=motion_params[[0]])
    diff[:, :3] *= 50.0
    return np.abs(diff).sum(axis=1)


def motion_columns(motion_params, n_tp):
    """
    Return a dict of motion column arrays aligned to n_tp timepoints.
    If motion_params is None or length-mismatched, returns NaN arrays.
    """
    col_names = ["rot_x", "rot_y", "rot_z", "trans_x", "trans_y", "trans_z", "fd"]
    nan_cols  = {c: np.full(n_tp, np.nan) for c in col_names}

    if motion_params is None:
        return nan_cols

    fd = compute_fd(motion_params)

    if len(fd) != n_tp:
        print(f"    [WARN] Motion params length ({len(fd)}) != timeseries length ({n_tp}) — filling with NaN")
        return nan_cols

    return {
        "rot_x"  : motion_params[:, 0],
        "rot_y"  : motion_params[:, 1],
        "rot_z"  : motion_params[:, 2],
        "trans_x": motion_params[:, 3],
        "trans_y": motion_params[:, 4],
        "trans_z": motion_params[:, 5],
        "fd"     : fd,
    }


# -----------------------------------------------------------------------------
# Z-stat helpers
# -----------------------------------------------------------------------------

def _peak_voxel(data, mask_bool):
    """
    Return (peak_value, (i, j, k)) for the maximum of data within mask_bool.
    Returns (nan, (None, None, None)) if mask_bool is empty.
    """
    if mask_bool.sum() == 0:
        return np.nan, (None, None, None)
    masked = np.where(mask_bool, data, -np.inf)
    idx    = np.unravel_index(np.argmax(masked), masked.shape)
    return float(data[idx]), idx


def compute_zstat_metrics(cope_dir, roi_bin, extraction_mask=None):
    """
    Compute average/peak z-stat and suprathreshold voxel count for an ROI.

    Reads directly from the cope*.feat dir's own stats images (independent
    of featquery/cluster_mask), so this works even for non-responders. Peak
    voxel coordinates are native functional-space voxel indices (i, j, k),
    matching the space of the QC mask, for easy lookup in FSLeyes.

    Returns a dict with keys:
        mean_zstat_roi             : mean of raw stats/zstat1.nii.gz across the
                                      whole native-space ROI mask
        peak_zstat_roi              : max of raw z-stat across the whole ROI mask
        peak_vox_roi_x/y/z          : voxel coords of that peak (native space)
        mean_zstat_extraction       : mean of raw z-stat across the extraction
                                       mask (largest cluster ∩ ROI), NaN if no
                                       extraction mask was passed
        peak_zstat_extraction       : max of raw z-stat across the extraction mask
        peak_vox_extraction_x/y/z   : voxel coords of that peak (native space)
        n_suprathreshold_vox       : voxels in the ROI where thresh_zstat1.nii.gz
                                      (FSL's cluster-corrected thresholded map)
                                      is nonzero
        mean_suprathreshold_zstat  : mean of thresh_zstat1.nii.gz among those
                                      suprathreshold voxels only, NaN if none
    """
    metrics = {
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

    zstat_path        = os.path.join(cope_dir, "stats", "zstat1.nii.gz")
    thresh_zstat_path = os.path.join(cope_dir, "thresh_zstat1.nii.gz")
    roi_bool           = roi_bin.astype(bool)

    if os.path.exists(zstat_path):
        zstat_data = nib.load(zstat_path).get_fdata()
        if roi_bool.sum() > 0:
            metrics["mean_zstat_roi"] = float(zstat_data[roi_bool].mean())
            peak_val, (px, py, pz) = _peak_voxel(zstat_data, roi_bool)
            metrics["peak_zstat_roi"] = peak_val
            metrics["peak_vox_roi_x"], metrics["peak_vox_roi_y"], metrics["peak_vox_roi_z"] = px, py, pz
        if extraction_mask is not None:
            extraction_bool = extraction_mask.astype(bool)
            if extraction_bool.sum() > 0:
                metrics["mean_zstat_extraction"] = float(zstat_data[extraction_bool].mean())
                peak_val, (px, py, pz) = _peak_voxel(zstat_data, extraction_bool)
                metrics["peak_zstat_extraction"] = peak_val
                metrics["peak_vox_extraction_x"], metrics["peak_vox_extraction_y"], metrics["peak_vox_extraction_z"] = px, py, pz
    else:
        print(f"    [WARN] No stats/zstat1.nii.gz found — z-stat columns will be NaN")

    if os.path.exists(thresh_zstat_path):
        thresh_data  = nib.load(thresh_zstat_path).get_fdata()
        supra_in_roi = (thresh_data != 0) & roi_bool
        metrics["n_suprathreshold_vox"] = int(supra_in_roi.sum())
        if supra_in_roi.sum() > 0:
            metrics["mean_suprathreshold_zstat"] = float(thresh_data[supra_in_roi].mean())
    else:
        print(f"    [WARN] No thresh_zstat1.nii.gz found — suprathreshold columns will be NaN")

    return metrics


# -----------------------------------------------------------------------------
# Helpers — file discovery
# -----------------------------------------------------------------------------

def find_cope_dirs(base_dir, task_filters, copes, subjects=None):
    """
    Find all cope*.feat dirs inside matching .gfeat directories.
    Returns a list of (gfeat_dir, cope_dir) tuples.
    """
    all_gfeats = sorted(glob.glob(os.path.join(base_dir, "sub-*", "ses-*", "*.gfeat")))
    filtered   = [d for d in all_gfeats
                  if any(task in os.path.basename(d) for task in task_filters)]
    if subjects is not None:
        filtered = [d for d in filtered
                    if any(sub in os.path.basename(d) for sub in subjects)]

    results = []
    for gfeat_dir in filtered:
        for cope in copes:
            cope_dir = os.path.join(gfeat_dir, cope)
            if os.path.isdir(cope_dir):
                results.append((gfeat_dir, cope_dir))
            else:
                print(f"  [WARN] cope dir not found: {cope_dir}")
    return results


def get_output_paths(gfeat_dir, cope_dir, roi_name, output_base):
    """Return (csv_path, mask_path, run_name) for a given gfeat/cope dir and ROI."""
    run_name  = f"{os.path.basename(gfeat_dir).replace('.gfeat', '')}_{os.path.basename(cope_dir).replace('.feat', '')}"
    out_dir   = os.path.join(output_base, roi_name, "timeseries")
    csv_path  = os.path.join(out_dir, f"{run_name}_{roi_name}_timeseries.csv")
    mask_path = os.path.join(out_dir, f"{run_name}_{roi_name}_mask.nii.gz")
    return csv_path, mask_path, run_name


# -----------------------------------------------------------------------------
# Extraction
# -----------------------------------------------------------------------------

def extract(cope_dir, roi_name, csv_path, mask_path, dry_run=False):
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
    # Level 2 (.gfeat) names the cluster table cluster_zstat1_std.txt,
    # unlike level 1's cluster_zstat1.txt
    cluster_mask_path  = os.path.join(cope_dir, "cluster_mask_zstat1.nii.gz")
    cluster_txt_path   = os.path.join(cope_dir, "cluster_zstat1_std.txt")
    filtered_func_path = os.path.join(cope_dir, "filtered_func_data.nii.gz")
    roi_mask_path       = os.path.join(cope_dir, f"{roi_name}.featquery", "mask.nii.gz")

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
        result.update(compute_zstat_metrics(cope_dir, roi_bin, extraction_mask))
        print(f"    [SKIP] Already exists: {csv_path}")
        return result

    if dry_run:
        print(f"    [DRY RUN] Would extract -> {csv_path}")
        result["status"] = "dry_run"
        return result

    cluster_mask_img  = nib.load(cluster_mask_path)
    cluster_mask_data = cluster_mask_img.get_fdata()
    roi_bin           = (nib.load(roi_mask_path).get_fdata() > 0.5).astype(int)
    result["roi_vox"] = int(roi_bin.sum())

    if roi_bin.sum() == 0:
        print(f"    [SKIP] ROI mask is empty after thresholding")
        result["status"] = "skip_empty_roi"
        return result

    cluster_df = pd.read_csv(cluster_txt_path, sep="\t")
    cluster_df.columns = [c.strip() for c in cluster_df.columns]

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

    combined_mask = ((cluster_mask_data == best_idx) * roi_bin).astype(np.int16)
    n_vox         = int(combined_mask.sum())
    result["extraction_vox"]  = n_vox
    result["all_cluster_vox"] = int(((cluster_mask_data > 0) * roi_bin).sum())
    result.update(compute_zstat_metrics(cope_dir, roi_bin, combined_mask))

    os.makedirs(os.path.dirname(mask_path), exist_ok=True)
    nib.save(nib.Nifti1Image(combined_mask, cluster_mask_img.affine), mask_path)

    func_data = nib.load(filtered_func_path).get_fdata()
    mean_ts   = func_data[combined_mask.astype(bool), :].mean(axis=0)
    n_tp      = len(mean_ts)

    motion_params = load_motion_params(cope_dir)
    mot_cols = motion_columns(motion_params, n_tp)

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
        copes    = roi_cfg["copes"]

        print(f"\n{'='*70}")
        print(f"ROI  : {roi_name}")
        print(f"Tasks: {tasks}")
        print(f"Copes: {copes}")
        print(f"{'='*70}")

        cope_dirs = find_cope_dirs(FEAT_BASE, tasks, copes, subjects=SUBJECTS)
        print(f"  Found {len(cope_dirs)} matching cope directories\n")

        for gfeat_dir, cope_dir in cope_dirs:
            csv_path, mask_path, run_name = get_output_paths(
                gfeat_dir, cope_dir, roi_name, OUTPUT_BASE
            )

            print(f"  -- {run_name}")

            result = extract(
                cope_dir  = cope_dir,
                roi_name  = roi_name,
                csv_path  = csv_path,
                mask_path = mask_path,
                dry_run   = dry_run,
            )

            summary.append({
                "roi"   : roi_name,
                "run"   : run_name,
                "status": result["status"],
            })

            if result["roi_vox"] is not None:
                responders.append({
                    "run"                      : run_name,
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

    if responders and not dry_run:
        df_resp = pd.DataFrame(responders)
        print(f"\n{'='*70}")
        print("RESPONDER SUMMARY")
        print(df_resp.groupby(["roi", "is_responder"]).size().rename("n_runs").to_string())
        resp_path = os.path.join(OUTPUT_BASE, "responder_summary.csv")
        df_resp.to_csv(resp_path, index=False)
        print(f"\nSaved: {resp_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract mean BOLD time series from level-2 ROI clusters.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview without executing")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
