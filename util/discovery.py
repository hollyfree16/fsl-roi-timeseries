"""
util/discovery.py

Shared helpers for locating FEAT/gfeat directories and checking which
stats images they contain. Used by the featquery, extract, and plot
scripts at both level 1 (.feat) and level 2 (.gfeat).
"""

import os
import glob


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
