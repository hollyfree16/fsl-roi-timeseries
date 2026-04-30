"""
plot_timeseries.py

Plots raw and z-scored mean BOLD time series for all ROI/task combinations.
Reads CSVs produced by extract_timeseries.py and timing from design.fsf.

For task runs: overlays coloured bars showing task ON blocks.
For rest runs: plots signal only (no block markers).

Saves two PNGs per subject/ROI/run:
  <feat_name>_<roi_name>_timeseries.png
  <feat_name>_<roi_name>_timeseries_zscored.png

Usage:
    python plot_timeseries.py
    python plot_timeseries.py --dry-run
"""

import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from config_plot import FEAT_BASE, OUTPUT_BASE, TASK_COLOR, ROI_TASK_MAP, SUBJECTS


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


def get_paths(feat_dir, roi_name, output_base):
    """Return (csv_path, png_path, fsf_path) for a given feat dir and ROI."""
    feat_name = os.path.basename(feat_dir).replace(".feat", "")
    csv_path  = os.path.join(output_base, roi_name, "timeseries",
                              f"{feat_name}_{roi_name}_timeseries.csv")
    png_path  = os.path.join(output_base, roi_name, "plots",
                              f"{feat_name}_{roi_name}_timeseries.png")
    fsf_path  = os.path.join(feat_dir, "design.fsf")
    return csv_path, png_path, fsf_path, feat_name


def parse_fsf_block_timing(fsf_path):
    """Parse TR, ndelete, and block onsets/offsets from a FEAT design.fsf."""
    with open(fsf_path, "r") as f:
        content = f.read()

    def get_val(key, cast=float):
        m = re.search(rf'set fmri\({re.escape(key)}\)\s+([^\s#]+)', content)
        if m:
            return cast(m.group(1))
        raise ValueError(f"Key not found in fsf: {key}")

    tr      = get_val("tr",      float)
    npts    = get_val("npts",    int)
    ndelete = get_val("ndelete", int)

    n_effective    = npts - ndelete
    total_duration = n_effective * tr

    block_onsets  = []
    block_offsets = []

    try:
        if get_val("shape1", int) == 0:   # square wave = block design
            off      = get_val("off1",   float)
            on       = get_val("on1",    float)
            phase    = get_val("phase1", float)
            stop     = get_val("stop1",  float)
            end_time = total_duration if stop < 0 else stop
            t = phase + off
            while t < end_time:
                block_onsets.append(t)
                block_offsets.append(min(t + on, end_time))
                t += off + on
    except ValueError:
        pass  # rest run or no EV — no blocks

    return {
        "tr"           : tr,
        "ndelete"      : ndelete,
        "block_onsets" : block_onsets,
        "block_offsets": block_offsets,
        "has_blocks"   : len(block_onsets) > 0,
    }


def save_plot(time_s, signal, timing, title, ylabel, output_path, task_color):
    """Render and save a single plot (raw or z-scored)."""
    fig, (ax_bar, ax) = plt.subplots(
        nrows=2, figsize=(14, 5),
        gridspec_kw={"height_ratios": [1, 8], "hspace": 0.05}
    )

    ax_bar.set_xlim(time_s[0], time_s[-1])
    ax_bar.set_ylim(0, 1)
    ax_bar.set_axis_off()
    ax_bar.set_title(title, fontsize=12, pad=8)

    if timing["has_blocks"]:
        # Derive task label from title (e.g. "task-hand" -> "Hand (ON)")
        task_match = re.search(r"task-([^_|\s]+)", title)
        task_label = (task_match.group(1).capitalize() + " (ON)") if task_match else "Task (ON)"

        for onset, offset in zip(timing["block_onsets"], timing["block_offsets"]):
            ax_bar.barh(y=0.5, width=offset - onset, left=onset,
                        height=0.6, color=task_color, align="center")
            ax.axvspan(onset, offset, color=task_color, alpha=0.12, zorder=1)

        task_patch = mpatches.Patch(color=task_color, label=task_label, alpha=0.8)
        ax.legend(handles=[task_patch], loc="upper right", fontsize=9, framealpha=0.7)

    ax.plot(time_s, signal, color="black", linewidth=0.9, zorder=3)

    if "Z-score" in ylabel:
        ax.axhline(0, color="gray", linewidth=0.7, linestyle="--", zorder=2)

    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlim(time_s[0], time_s[-1])
    ax.tick_params(labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot(csv_path, fsf_path, png_path, roi_name,
         feat_name, task_color, dry_run=False):
    """Generate raw and z-scored plots for one feat dir + ROI."""

    raw_path = png_path
    z_path   = png_path.replace(".png", "_zscored.png")

    if os.path.exists(raw_path) and os.path.exists(z_path):
        print(f"    [SKIP] Plots already exist")
        return "already_done"

    if not os.path.exists(csv_path):
        print(f"    [SKIP] CSV not found: {csv_path}")
        return "skip_no_csv"

    if not os.path.exists(fsf_path):
        print(f"    [SKIP] design.fsf not found: {fsf_path}")
        return "skip_no_fsf"

    if dry_run:
        print(f"    [DRY RUN] Would save -> {raw_path}")
        return "dry_run"

    df     = pd.read_csv(csv_path)
    bold   = df["mean_bold"].values
    timing = parse_fsf_block_timing(fsf_path)
    time_s = np.arange(len(bold)) * timing["tr"] + timing["ndelete"] * timing["tr"]

    title_base = f"{feat_name} | {roi_name}"

    # Raw plot
    save_plot(
        time_s     = time_s,
        signal     = bold,
        timing     = timing,
        title      = title_base + " — Raw",
        ylabel     = "Mean BOLD (a.u.)",
        output_path= raw_path,
        task_color = task_color,
    )
    print(f"    Saved: {raw_path}")

    # Z-scored plot
    bold_z = (bold - bold.mean()) / bold.std()
    save_plot(
        time_s     = time_s,
        signal     = bold_z,
        timing     = timing,
        title      = title_base + " — Z-scored",
        ylabel     = "Z-score",
        output_path= z_path,
        task_color = task_color,
    )
    print(f"    Saved: {z_path}")

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
            csv_path, png_path, fsf_path, feat_name = get_paths(
                feat_dir, roi_name, OUTPUT_BASE
            )
            print(f"  -- {feat_name}")

            status = plot(
                csv_path   = csv_path,
                fsf_path   = fsf_path,
                png_path   = png_path,
                roi_name   = roi_name,
                feat_name  = feat_name,
                task_color = TASK_COLOR,
                dry_run    = dry_run,
            )
            summary.append({"roi": roi_name, "feat": feat_name, "status": status})

    # Summary
    df = pd.DataFrame(summary)
    if not df.empty:
        print(f"\n{'='*70}")
        print("PLOTTING SUMMARY")
        print(df.groupby(["roi", "status"]).size().to_string())
        failed = df[~df["status"].isin(["ok", "already_done", "dry_run"])]
        if not failed.empty:
            print(f"\nSkipped / failed:")
            print(failed.to_string(index=False))

        if not dry_run:
            os.makedirs(OUTPUT_BASE, exist_ok=True)
            df.to_csv(os.path.join(OUTPUT_BASE, "plotting_summary.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot mean BOLD time series with task blocks.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview without saving any files")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
