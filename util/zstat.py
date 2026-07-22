"""
util/zstat.py

Shared helpers for computing z-stat metrics (mean/peak/suprathreshold
voxel counts) over an ROI, read directly from a FEAT dir's own stats
images. Used by extract_timeseries.py and extract_timeseries_level2.py.
"""

import os
import numpy as np
import nibabel as nib


def peak_voxel(data, mask_bool):
    """
    Return (peak_value, (i, j, k)) for the maximum of data within mask_bool.
    Returns (nan, (None, None, None)) if mask_bool is empty.
    """
    if mask_bool.sum() == 0:
        return np.nan, (None, None, None)
    masked = np.where(mask_bool, data, -np.inf)
    idx    = np.unravel_index(np.argmax(masked), masked.shape)
    return float(data[idx]), idx


def compute_zstat_metrics(feat_dir, roi_bin, extraction_mask=None):
    """
    Compute average/peak z-stat and suprathreshold voxel count for an ROI.

    Reads directly from the FEAT (or cope*.feat) dir's own stats images
    (independent of featquery/cluster_mask), so this works even for
    non-responders. Peak voxel coordinates are native functional-space
    voxel indices (i, j, k), matching the space of the QC mask, for easy
    lookup in FSLeyes.

    Returns a dict with keys:
        mean_zstat_roi            : mean of raw stats/zstat1.nii.gz across the
                                     whole native-space ROI mask
        peak_zstat_roi            : max of raw z-stat across the whole ROI mask
        peak_vox_roi_x/y/z        : voxel coords of that peak (native space)
        mean_zstat_extraction     : mean of raw z-stat across the extraction
                                     mask (largest cluster ∩ ROI), NaN if no
                                     extraction mask was passed
        peak_zstat_extraction     : max of raw z-stat across the extraction mask
        peak_vox_extraction_x/y/z : voxel coords of that peak (native space)
        n_suprathreshold_vox      : voxels in the ROI where thresh_zstat1.nii.gz
                                     (FSL's cluster-corrected thresholded map)
                                     is nonzero
        mean_suprathreshold_zstat : mean of thresh_zstat1.nii.gz among those
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

    zstat_path        = os.path.join(feat_dir, "stats", "zstat1.nii.gz")
    thresh_zstat_path = os.path.join(feat_dir, "thresh_zstat1.nii.gz")
    roi_bool           = roi_bin.astype(bool)

    if os.path.exists(zstat_path):
        zstat_data = nib.load(zstat_path).get_fdata()
        if roi_bool.sum() > 0:
            metrics["mean_zstat_roi"] = float(zstat_data[roi_bool].mean())
            peak_val, (px, py, pz) = peak_voxel(zstat_data, roi_bool)
            metrics["peak_zstat_roi"] = peak_val
            metrics["peak_vox_roi_x"], metrics["peak_vox_roi_y"], metrics["peak_vox_roi_z"] = px, py, pz
        if extraction_mask is not None:
            extraction_bool = extraction_mask.astype(bool)
            if extraction_bool.sum() > 0:
                metrics["mean_zstat_extraction"] = float(zstat_data[extraction_bool].mean())
                peak_val, (px, py, pz) = peak_voxel(zstat_data, extraction_bool)
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
