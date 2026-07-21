"""
plot_timeseries_level2.py

Plots raw and z-scored mean BOLD time series for group/level-2 (.gfeat)
ROI/task combinations. Reads CSVs produced by extract_timeseries_level2.py.

Timing source:
  1. The "timing" entry in config_plot_level2.py's ROI_TASK_MAP (required
     for a task overlay — a level-2 design.fsf has no block-design EV), or
  2. The cope's design.fsf, as a fallback (TR/ndelete always come from here)

Level-2 runs have no motion .par file, so the HTML QC output shows only the
raw/z-scored BOLD panels — the rotation/translation/FD panels are omitted.

Saves per subject/ROI/cope:
  PNG outputs (raw + z-scored):
    <run_name>_<roi_name>_timeseries.png
    <run_name>_<roi_name>_timeseries_zscored.png

  Interactive HTML (BOLD only, no motion panels):
    <run_name>_<roi_name>_timeseries_qc.html

Usage:
    python plot_timeseries_level2.py
    python plot_timeseries_level2.py --dry-run
"""

import os
import re
import glob
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from config_plot_level2 import FEAT_BASE, OUTPUT_BASE, TASK_COLOR, ROI_TASK_MAP, SUBJECTS


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


def get_paths(gfeat_dir, cope_dir, roi_name, output_base):
    """Return (csv_path, png_path, html_path, fsf_path, run_name)."""
    run_name  = f"{os.path.basename(gfeat_dir).replace('.gfeat', '')}_{os.path.basename(cope_dir).replace('.feat', '')}"
    ts_dir    = os.path.join(output_base, roi_name, "timeseries")
    plot_dir  = os.path.join(output_base, roi_name, "plots")
    csv_path  = os.path.join(ts_dir,   f"{run_name}_{roi_name}_timeseries.csv")
    png_path  = os.path.join(plot_dir, f"{run_name}_{roi_name}_timeseries.png")
    html_path = os.path.join(plot_dir, f"{run_name}_{roi_name}_timeseries_qc.html")
    fsf_path  = os.path.join(cope_dir, "design.fsf")
    return csv_path, png_path, html_path, fsf_path, run_name


# -----------------------------------------------------------------------------
# Helpers — timing
# -----------------------------------------------------------------------------

def parse_fsf_tr_ndelete(fsf_path):
    """Parse just TR and ndelete from a level-2 design.fsf (no block EVs)."""
    with open(fsf_path, "r") as f:
        content = f.read()

    def get_val(key, cast=float):
        m = re.search(rf'set fmri\({re.escape(key)}\)\s+([^\s#]+)', content)
        if m:
            return cast(m.group(1))
        raise ValueError(f"Key not found in fsf: {key}")

    tr      = get_val("tr",      float)
    ndelete = get_val("ndelete", int)
    return tr, ndelete


def resolve_timing(fsf_path, config_timing=None):
    """
    Return a timing dict for plotting.

    A level-2 design.fsf has no block-design EV to parse, so block
    onsets/offsets come only from config_timing (ROI_TASK_MAP["timing"]).
    TR and ndelete are always read from the fsf.
    """
    tr, ndelete = parse_fsf_tr_ndelete(fsf_path)

    onsets  = (config_timing or {}).get("block_onsets",  [])
    offsets = (config_timing or {}).get("block_offsets", [])

    return {
        "tr"           : tr,
        "ndelete"      : ndelete,
        "block_onsets" : onsets,
        "block_offsets": offsets,
        "has_blocks"   : len(onsets) > 0,
        "timing_source": "config" if config_timing else "none",
    }


# -----------------------------------------------------------------------------
# PNG output
# -----------------------------------------------------------------------------

def save_plot(time_s, signal, timing, title, ylabel, output_path, task_color):
    """Render and save a single matplotlib plot (raw or z-scored)."""
    fig, (ax_bar, ax) = plt.subplots(
        nrows=2, figsize=(14, 5),
        gridspec_kw={"height_ratios": [1, 8], "hspace": 0.05}
    )

    ax_bar.set_xlim(time_s[0], time_s[-1])
    ax_bar.set_ylim(0, 1)
    ax_bar.set_axis_off()
    ax_bar.set_title(title, fontsize=12, pad=8)

    if timing["has_blocks"]:
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


# -----------------------------------------------------------------------------
# HTML output (BOLD panels only — no motion .par at level 2)
# -----------------------------------------------------------------------------

def _parse_feat_meta(run_name):
    """Extract sub/ses/task/run from a BIDS-style gfeat/cope run name."""
    fields = {}
    for key in ("sub", "ses", "task", "run"):
        m = re.search(rf'({key}-[^_]+)', run_name)
        fields[key] = m.group(1) if m else ""
    return fields


def save_html(bold, timing, run_name, roi_name, output_path):
    """Render a two-panel interactive HTML QC plot (raw + z-scored BOLD)."""
    tr      = timing["tr"]
    ndelete = timing["ndelete"]
    n_vols  = len(bold)

    meta = _parse_feat_meta(run_name)
    meta_str = " · ".join(v for v in [
        meta["sub"], meta["ses"], meta["task"], meta["run"], roi_name
    ] if v)
    meta_str += f" &nbsp;·&nbsp; TR = {tr} s · {n_vols} volumes &nbsp;·&nbsp; level 2 (no motion data)"

    bold_z = ((bold - bold.mean()) / bold.std()).tolist()

    block_onsets_js  = json.dumps(timing["block_onsets"])
    block_offsets_js = json.dumps(timing["block_offsets"])
    bold_js          = json.dumps([round(float(v), 3) for v in bold])
    bold_z_js        = json.dumps([round(float(v), 4) for v in bold_z])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{run_name} | {roi_name} — QC</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 1.5rem 2rem; background: #fff; color: #222; }}
  h1   {{ font-size: 15px; font-weight: 600; margin: 0 0 0.4rem 0; }}
  .fp-meta      {{ font-size: 13px; color: #666; margin: 0 0 0.75rem 0; }}
  .fp-legend    {{ display: flex; flex-wrap: wrap; gap: 14px; font-size: 11px;
                   color: #666; margin-bottom: 12px; }}
  .fp-legend span    {{ display: flex; align-items: center; gap: 4px; }}
  .fp-swatch         {{ width: 20px; height: 2.5px; border-radius: 2px; }}
  .fp-swatchbox      {{ width: 9px; height: 9px; border-radius: 2px; }}
  .fp-panel          {{ margin-bottom: 4px; }}
  .fp-label          {{ font-size: 11px; color: #666; margin: 0 0 2px 0; font-weight: 500; }}
</style>
</head>
<body>
<h1>{run_name} | {roi_name}</h1>
<p class="fp-meta">{meta_str}</p>

<div class="fp-legend">
  <span><span class="fp-swatchbox" style="background:#4C9BE8;opacity:0.5"></span>Task ON</span>
  <span><span class="fp-swatch" style="background:#2C2C2A"></span>BOLD</span>
</div>

<div class="fp-panel"><p class="fp-label">Raw BOLD (a.u.)</p>
<div style="position:relative;width:100%;height:140px;">
  <canvas id="c1" role="img" aria-label="Raw BOLD signal"></canvas>
</div></div>

<div class="fp-panel"><p class="fp-label">Z-scored BOLD</p>
<div style="position:relative;width:100%;height:120px;">
  <canvas id="c2" role="img" aria-label="Z-scored BOLD signal"></canvas>
</div></div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<script>
const TR      = {tr};
const ndelete = {ndelete};
const Y_AXIS_WIDTH = 62;

const bold         = {bold_js};
const boldZ        = {bold_z_js};
const blockOnsets  = {block_onsets_js};
const blockOffsets = {block_offsets_js};

const overlayPlugin = {{
  id: 'overlay',
  beforeDraw(chart) {{
    const {{ ctx, chartArea: {{ left, right, top, bottom }}, scales: {{ x }} }} = chart;
    if (!x) return;
    ctx.save();
    blockOnsets.forEach((onset, i) => {{
      const vs = onset / TR - ndelete;
      const ve = blockOffsets[i] / TR - ndelete;
      const x0 = Math.max(left,  x.getPixelForValue(vs));
      const x1 = Math.min(right, x.getPixelForValue(ve));
      ctx.fillStyle = 'rgba(76,155,232,0.11)';
      ctx.fillRect(x0, top, x1 - x0, bottom - top);
    }});
    ctx.restore();
  }}
}};

const afterFit = scale => {{ scale.width = Y_AXIS_WIDTH; }};

const xAxis = (showLabels) => ({{
  ticks: {{
    callback: (v) => {{
      if (!showLabels) return '';
      const t = v * TR + ndelete * TR;
      return Math.round(t) % 25 === 0 ? `${{Math.round(t)}}s` : '';
    }},
    autoSkip: false, maxRotation: 0, font: {{ size: 10 }}, color: '#888780'
  }},
  grid: {{ color: 'rgba(136,135,128,0.12)' }}
}});

const yAxis = (label) => ({{
  title: {{ display: true, text: label, font: {{ size: 10 }}, color: '#888780' }},
  ticks: {{ font: {{ size: 10 }}, color: '#888780', maxTicksLimit: 4 }},
  grid:  {{ color: 'rgba(136,135,128,0.12)' }},
  afterFit
}});

const baseOpts = (yLabel, showXLabels) => ({{
  responsive: true, maintainAspectRatio: false, animation: false,
  layout: {{ padding: {{ right: 8 }} }},
  plugins: {{
    legend: {{ display: false }},
    tooltip: {{ callbacks: {{
      title: items => {{
        const i = items[0].dataIndex;
        return `vol ${{i}} · t=${{(i * TR + ndelete * TR).toFixed(2)}}s`;
      }}
    }} }}
  }},
  scales: {{ x: xAxis(showXLabels), y: yAxis(yLabel) }}
}});

const vols = bold.map((_, i) => i);

new Chart(document.getElementById('c1'), {{
  type: 'line', plugins: [overlayPlugin],
  data: {{ labels: vols, datasets: [{{
    data: bold, borderColor: '#2C2C2A', borderWidth: 1.1, pointRadius: 0, tension: 0.3
  }}] }},
  options: baseOpts('a.u.', false)
}});

new Chart(document.getElementById('c2'), {{
  type: 'line', plugins: [overlayPlugin],
  data: {{ labels: vols, datasets: [{{
    data: boldZ, borderColor: '#2C2C2A', borderWidth: 1.1, pointRadius: 0, tension: 0.3
  }}] }},
  options: baseOpts('Z', true)
}});
</script>
</body>
</html>
"""

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)


# -----------------------------------------------------------------------------
# Main per-run function
# -----------------------------------------------------------------------------

def plot(csv_path, fsf_path, png_path, html_path, roi_name, run_name,
         task_color, config_timing=None, dry_run=False):
    """Generate PNG (raw + z-scored) and HTML QC plot for one cope dir + ROI."""

    raw_path = png_path
    z_path   = png_path.replace(".png", "_zscored.png")

    all_exist = (os.path.exists(raw_path) and
                 os.path.exists(z_path) and
                 os.path.exists(html_path))
    if all_exist:
        print(f"    [SKIP] All outputs already exist")
        return "already_done"

    if not os.path.exists(csv_path):
        print(f"    [SKIP] CSV not found: {csv_path}")
        return "skip_no_csv"

    if not os.path.exists(fsf_path):
        print(f"    [SKIP] design.fsf not found: {fsf_path}")
        return "skip_no_fsf"

    if dry_run:
        print(f"    [DRY RUN] Would save:")
        print(f"      PNG raw  -> {raw_path}")
        print(f"      PNG z-sc -> {z_path}")
        print(f"      HTML QC  -> {html_path}")
        return "dry_run"

    df     = pd.read_csv(csv_path)
    bold   = df["mean_bold"].values
    timing = resolve_timing(fsf_path, config_timing=config_timing)
    time_s = np.arange(len(bold)) * timing["tr"] + timing["ndelete"] * timing["tr"]
    print(f"    Timing source: {timing['timing_source']}")
    if timing["timing_source"] == "none":
        print(f"    [WARN] No 'timing' entry in config — plot will have no task overlay")

    title_base = f"{run_name} | {roi_name}"

    if not os.path.exists(raw_path):
        save_plot(
            time_s=time_s, signal=bold, timing=timing,
            title=title_base + " — Raw",
            ylabel="Mean BOLD (a.u.)",
            output_path=raw_path, task_color=task_color,
        )
        print(f"    Saved PNG (raw): {raw_path}")
    else:
        print(f"    [SKIP] PNG (raw) already exists")

    if not os.path.exists(z_path):
        bold_z = (bold - bold.mean()) / bold.std()
        save_plot(
            time_s=time_s, signal=bold_z, timing=timing,
            title=title_base + " — Z-scored",
            ylabel="Z-score",
            output_path=z_path, task_color=task_color,
        )
        print(f"    Saved PNG (z):   {z_path}")
    else:
        print(f"    [SKIP] PNG (z-scored) already exists")

    if not os.path.exists(html_path):
        save_html(
            bold=bold, timing=timing,
            run_name=run_name, roi_name=roi_name,
            output_path=html_path,
        )
        print(f"    Saved HTML QC:   {html_path}")
    else:
        print(f"    [SKIP] HTML already exists")

    return "ok"


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main(dry_run=False):
    summary = []

    for roi_cfg in ROI_TASK_MAP:
        roi_name      = roi_cfg["roi_name"]
        tasks         = roi_cfg["tasks"]
        copes         = roi_cfg["copes"]
        config_timing = roi_cfg.get("timing", None)

        print(f"\n{'='*70}")
        print(f"ROI  : {roi_name}")
        print(f"Tasks: {tasks}")
        print(f"Copes: {copes}")
        print(f"{'='*70}")

        cope_dirs = find_cope_dirs(FEAT_BASE, tasks, copes, subjects=SUBJECTS)
        print(f"  Found {len(cope_dirs)} matching cope directories\n")

        for gfeat_dir, cope_dir in cope_dirs:
            csv_path, png_path, html_path, fsf_path, run_name = get_paths(
                gfeat_dir, cope_dir, roi_name, OUTPUT_BASE
            )
            print(f"  -- {run_name}")

            status = plot(
                csv_path      = csv_path,
                fsf_path      = fsf_path,
                png_path      = png_path,
                html_path     = html_path,
                roi_name      = roi_name,
                run_name      = run_name,
                task_color    = TASK_COLOR,
                config_timing = config_timing,
                dry_run       = dry_run,
            )
            summary.append({"roi": roi_name, "run": run_name, "status": status})

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
    parser = argparse.ArgumentParser(description="Plot mean BOLD time series for level-2 (.gfeat) ROI runs.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview without saving any files")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
