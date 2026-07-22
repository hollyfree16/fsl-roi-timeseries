"""
util/motion.py

Shared helpers for loading FSL motion parameters and computing framewise
displacement. Used by the extract and plot scripts at level 1; level-2
(.gfeat) runs have no .par file, so these degrade gracefully to None/NaN.
"""

import os
import numpy as np


def load_motion_params(feat_dir):
    """
    Load the FSL motion parameter file from the standard location.
    Returns an (N, 6) array [rotX, rotY, rotZ, transX, transY, transZ]
    or None if the file is not found.
    """
    par_path = os.path.join(feat_dir, "mc", "prefiltered_func_data_mcf.par")
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


def find_outliers(fd, threshold=0.5):
    """Return volume indices where FD exceeds threshold."""
    return list(np.where(fd > threshold)[0])
