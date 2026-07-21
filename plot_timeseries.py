"""
plot_timeseries.py

Plots raw and z-scored mean BOLD time series for all ROI/task combinations.
Reads CSVs produced by extract_timeseries.py and timing from either:
  1. The ROI_TASK_MAP entry in config_plot.py (takes precedence if defined), or
  2. The run's design.fsf (fallback)

For task runs: overlays coloured bars showing task ON blocks.
For rest runs: plots signal only (no block markers).

Saves per subject/ROI/run:
  PNG outputs (raw + z-scored):
    <feat_name>_<roi_name>_timeseries.png
    <feat_name>_<roi_name>_timeseries_zscored.png

  Interactive five-panel HTML (BOLD + motion QC):
    <feat_name>_<roi_name>_timeseries_qc.html

Usage:
    python plot_timeseries.py
    python plot_timeseries.py --dry-run
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
from config_plot import FEAT_BASE, OUTPUT_BASE, TASK_COLOR, ROI_TASK_MAP, SUBJECTS


# -----------------------------------------------------------------------------
# Helpers — file discovery
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
    """Return (csv_path, png_path, html_path, fsf_path, feat_name)."""
    feat_name = os.path.basename(feat_dir).replace(".feat", "")
    ts_dir    = os.path.join(output_base, roi_name, "timeseries")
    plot_dir  = os.path.join(output_base, roi_name, "plots")
    csv_path  = os.path.join(ts_dir,   f"{feat_name}_{roi_name}_timeseries.csv")
    png_path  = os.path.join(plot_dir, f"{feat_name}_{roi_name}_timeseries.png")
    html_path = os.path.join(plot_dir, f"{feat_name}_{roi_name}_timeseries_qc.html")
    fsf_path  = os.path.join(feat_dir, "design.fsf")
    return csv_path, png_path, html_path, fsf_path, feat_name


# -----------------------------------------------------------------------------
# Helpers — timing
# -----------------------------------------------------------------------------

def parse_fsf_block_timing(fsf_path):
    """
    Parse TR, ndelete, and block onsets/offsets from a FEAT design.fsf.

    Handles:
      - Square-wave EVs     (shape1 = 0)
      - Custom 3-column EVs (shape1 = 3)
    """
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
        shape = get_val("shape1", int)

        if shape == 0:
            # Square wave
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

        elif shape == 3:
            # Custom 3-column format
            custom_path = get_val("custom1", str)
            if not os.path.isabs(custom_path):
                custom_path = os.path.join(os.path.dirname(fsf_path), custom_path)
            with open(custom_path, "r") as cf:
                for line in cf:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    if len(parts) < 3:
                        continue
                    onset, duration, weight = float(parts[0]), float(parts[1]), float(parts[2])
                    if weight > 0:
                        block_onsets.append(onset)
                        block_offsets.append(onset + duration)

    except ValueError:
        pass  # rest run or unrecognised EV type — no blocks

    return {
        "tr"           : tr,
        "ndelete"      : ndelete,
        "block_onsets" : block_onsets,
        "block_offsets": block_offsets,
        "has_blocks"   : len(block_onsets) > 0,
    }


def resolve_timing(fsf_path, config_timing=None):
    """
    Return a timing dict for plotting.

    Priority:
      1. config_timing — if provided in ROI_TASK_MAP, always used
      2. parse_fsf_block_timing(fsf_path) — fallback

    config_timing format (seconds):
        {
            "block_onsets"  : [<onset_s>, ...],
            "block_offsets" : [<offset_s>, ...],
        }
    TR and ndelete are always read from the fsf regardless.
    """
    fsf = parse_fsf_block_timing(fsf_path)

    if config_timing is not None:
        onsets  = config_timing.get("block_onsets",  [])
        offsets = config_timing.get("block_offsets", [])
        return {
            "tr"           : fsf["tr"],
            "ndelete"      : fsf["ndelete"],
            "block_onsets" : onsets,
            "block_offsets": offsets,
            "has_blocks"   : len(onsets) > 0,
            "timing_source": "config",
        }

    fsf["timing_source"] = "fsf"
    return fsf


# -----------------------------------------------------------------------------
# Helpers — motion parameters
# -----------------------------------------------------------------------------

def load_motion_params(feat_dir):
    """
    Load the FSL motion parameter file (.par) from the standard location.
    Returns an (N, 6) numpy array [rotX, rotY, rotZ, transX, transY, transZ],
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
    Compute framewise displacement from a (N, 6) motion parameter array.
    Rotations (first 3 columns) are converted from radians to mm
    using a 50 mm head-radius assumption.
    Returns a 1-D array of length N (FD[0] = 0).
    """
    diff = np.diff(motion_params, axis=0, prepend=motion_params[[0]])
    diff[:, :3] *= 50.0   # rad -> mm
    return np.abs(diff).sum(axis=1)


def find_outliers(fd, threshold=0.5):
    """Return volume indices where FD exceeds threshold."""
    return list(np.where(fd > threshold)[0])


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


# -----------------------------------------------------------------------------
# HTML output
# -----------------------------------------------------------------------------

def _parse_feat_meta(feat_name):
    """Extract sub/ses/task/run from a BIDS-style feat directory name."""
    fields = {}
    for key in ("sub", "ses", "task", "run"):
        m = re.search(rf'({key}-[^_]+)', feat_name)
        fields[key] = m.group(1) if m else ""
    return fields


def save_html(bold, time_s, timing, motion_params, feat_name, roi_name,
              fd_threshold, output_path):
    """
    Render the five-panel interactive HTML QC plot (Chart.js).

    Panels:
      1. Raw BOLD
      2. Z-scored BOLD
      3. Rotations (rad)        — hidden if no .par file
      4. Translations (mm)      — hidden if no .par file
      5. Framewise displacement — hidden if no .par file
    """
    tr      = timing["tr"]
    ndelete = timing["ndelete"]
    n_vols  = len(bold)

    meta = _parse_feat_meta(feat_name)
    meta_str = " · ".join(v for v in [
        meta["sub"], meta["ses"], meta["task"], meta["run"], roi_name
    ] if v)
    meta_str += f" &nbsp;·&nbsp; TR = {tr} s · {n_vols} volumes"

    bold_z = ((bold - bold.mean()) / bold.std()).tolist()

    has_motion = motion_params is not None
    if has_motion:
        fd       = compute_fd(motion_params)
        outliers = find_outliers(fd, threshold=fd_threshold)
        par_list = motion_params.tolist()
        fd_list  = [round(float(v), 4) for v in fd]
    else:
        outliers = []
        par_list = []
        fd_list  = []

    spikes_js        = json.dumps([1 if i in outliers else 0 for i in range(n_vols)])
    outlier_labels    = ", ".join(f"vol {i}" for i in outliers) if outliers else "none"
    block_onsets_js   = json.dumps(timing["block_onsets"])
    block_offsets_js  = json.dumps(timing["block_offsets"])
    bold_js           = json.dumps([round(float(v), 3) for v in bold])
    bold_z_js         = json.dumps([round(float(v), 4) for v in bold_z])
    par_js            = json.dumps([[round(x, 8) for x in row] for row in par_list])
    fd_js             = json.dumps(fd_list)

    motion_height  = "110px" if has_motion else "0"
    motion_display = "block" if has_motion else "none"
    fd_height      = "100px" if has_motion else "0"

    motion_legend = ""
    if has_motion:
        motion_legend = """
  <span><span class="fp-swatch" style="background:#534AB7"></span>rot X</span>
  <span><span class="fp-swatch" style="background:#0F6E56"></span>rot Y</span>
  <span><span class="fp-swatch" style="background:#993C1D"></span>rot Z</span>
  <span><span class="fp-swatch" style="background:#185FA5"></span>trans X</span>
  <span><span class="fp-swatch" style="background:#854F0B"></span>trans Y</span>
  <span><span class="fp-swatch" style="background:#993556"></span>trans Z</span>
  <span><span class="fp-swatch" style="background:#888780"></span>FD</span>"""

    outlier_legend = ""
    if outliers:
        outlier_legend = (
            f'<span><span class="fp-swatchbox" style="background:#E24B4A;opacity:0.5">'
            f'</span>Motion outlier ({outlier_labels})</span>'
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{feat_name} | {roi_name} — QC</title>
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
<h1>{feat_name} | {roi_name}</h1>
<p class="fp-meta">{meta_str}</p>

<div class="fp-legend">
  <span><span class="fp-swatchbox" style="background:#4C9BE8;opacity:0.5"></span>Task ON</span>
  {outlier_legend}
  <span><span class="fp-swatch" style="background:#2C2C2A"></span>BOLD</span>
  {motion_legend}
</div>

<div class="fp-panel"><p class="fp-label">Raw BOLD (a.u.)</p>
<div style="position:relative;width:100%;height:140px;">
  <canvas id="c1" role="img" aria-label="Raw BOLD signal"></canvas>
</div></div>

<div class="fp-panel"><p class="fp-label">Z-scored BOLD</p>
<div style="position:relative;width:100%;height:120px;">
  <canvas id="c2" role="img" aria-label="Z-scored BOLD signal"></canvas>
</div></div>

<div class="fp-panel" style="display:{motion_display}"><p class="fp-label">Rotations (rad)</p>
<div style="position:relative;width:100%;height:{motion_height};">
  <canvas id="c3" role="img" aria-label="Rotation parameters"></canvas>
</div></div>

<div class="fp-panel" style="display:{motion_display}"><p class="fp-label">Translations (mm)</p>
<div style="position:relative;width:100%;height:{motion_height};">
  <canvas id="c4" role="img" aria-label="Translation parameters"></canvas>
</div></div>

<div class="fp-panel" style="display:{motion_display}"><p class="fp-label">Framewise displacement (mm)</p>
<div style="position:relative;width:100%;height:{fd_height};">
  <canvas id="c5" role="img" aria-label="Framewise displacement"></canvas>
</div></div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
<script>
const TR      = {tr};
const ndelete = {ndelete};
const Y_AXIS_WIDTH = 62;

const bold         = {bold_js};
const boldZ        = {bold_z_js};
const par          = {par_js};
const fd           = {fd_js};
const spikes       = {spikes_js};
const blockOnsets  = {block_onsets_js};
const blockOffsets = {block_offsets_js};
const hasMotion    = {str(has_motion).lower()};

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
    spikes.forEach((v, i) => {{
      if (!v) return;
      const px = x.getPixelForValue(i);
      ctx.fillStyle   = 'rgba(226,75,74,0.18)';
      ctx.fillRect(px - 3, top, 7, bottom - top);
      ctx.strokeStyle = 'rgba(226,75,74,0.65)';
      ctx.lineWidth   = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(px, top); ctx.lineTo(px, bottom); ctx.stroke();
      ctx.setLineDash([]);
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
        return `vol ${{i}} · t=${{(i * TR + ndelete * TR).toFixed(2)}}s${{spikes[i] ? ' · outlier' : ''}}`;
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
  options: {{
    ...baseOpts('Z', !hasMotion),
    plugins: {{ legend: {{ display: false }}, tooltip: {{ callbacks: {{
      title: items => {{
        const i = items[0].dataIndex;
        return `vol ${{i}} · t=${{(i * TR + ndelete * TR).toFixed(2)}}s${{spikes[i] ? ' · outlier' : ''}}`;
      }},
      label: item => `Z: ${{item.raw.toFixed(3)}}`
    }} }} }}
  }}
}});

if (hasMotion) {{
  new Chart(document.getElementById('c3'), {{
    type: 'line', plugins: [overlayPlugin],
    data: {{ labels: vols, datasets: [
      {{ data: par.map(r => r[0]), borderColor: '#534AB7', borderWidth: 1, pointRadius: 0, tension: 0.3 }},
      {{ data: par.map(r => r[1]), borderColor: '#0F6E56', borderWidth: 1, pointRadius: 0, tension: 0.3 }},
      {{ data: par.map(r => r[2]), borderColor: '#993C1D', borderWidth: 1, pointRadius: 0, tension: 0.3 }},
    ] }},
    options: baseOpts('rad', false)
  }});

  new Chart(document.getElementById('c4'), {{
    type: 'line', plugins: [overlayPlugin],
    data: {{ labels: vols, datasets: [
      {{ data: par.map(r => r[3]), borderColor: '#185FA5', borderWidth: 1, pointRadius: 0, tension: 0.3 }},
      {{ data: par.map(r => r[4]), borderColor: '#854F0B', borderWidth: 1, pointRadius: 0, tension: 0.3 }},
      {{ data: par.map(r => r[5]), borderColor: '#993556', borderWidth: 1, pointRadius: 0, tension: 0.3 }},
    ] }},
    options: baseOpts('mm', false)
  }});

  new Chart(document.getElementById('c5'), {{
    type: 'line', plugins: [overlayPlugin],
    data: {{ labels: vols, datasets: [{{
      data: fd, borderColor: '#888780', borderWidth: 1, pointRadius: 0, tension: 0.2,
      fill: true, backgroundColor: 'rgba(136,135,128,0.08)'
    }}] }},
    options: {{
      ...baseOpts('mm', true),
      plugins: {{ legend: {{ display: false }}, tooltip: {{ callbacks: {{
        title: items => {{
          const i = items[0].dataIndex;
          return `vol ${{i}} · t=${{(i * TR + ndelete * TR).toFixed(2)}}s${{spikes[i] ? ' · outlier' : ''}}`;
        }},
        label: item => `FD: ${{item.raw.toFixed(3)}} mm`
      }} }} }}
    }}
  }});
}}
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

def plot(csv_path, fsf_path, png_path, html_path, roi_name,
         feat_name, feat_dir, task_color, fd_threshold,
         config_timing=None, dry_run=False):
    """Generate PNG (raw + z-scored) and HTML QC plot for one feat dir + ROI."""

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

    title_base = f"{feat_name} | {roi_name}"

    # PNG: raw
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

    # PNG: z-scored
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

    # HTML: five-panel QC
    if not os.path.exists(html_path):
        motion_params = load_motion_params(feat_dir)
        if motion_params is None:
            print(f"    [WARN] No .par file found — HTML will show BOLD panels only")
        save_html(
            bold=bold, time_s=time_s, timing=timing,
            motion_params=motion_params,
            feat_name=feat_name, roi_name=roi_name,
            fd_threshold=fd_threshold,
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
    from config_plot import FD_THRESHOLD
    summary = []

    for roi_cfg in ROI_TASK_MAP:
        roi_name      = roi_cfg["roi_name"]
        tasks         = roi_cfg["tasks"]
        config_timing = roi_cfg.get("timing", None)

        print(f"\n{'='*70}")
        print(f"ROI  : {roi_name}")
        print(f"Tasks: {tasks}")
        print(f"{'='*70}")

        feat_dirs = find_feat_dirs(FEAT_BASE, tasks, subjects=SUBJECTS)
        print(f"  Found {len(feat_dirs)} matching .feat directories\n")

        for feat_dir in feat_dirs:
            csv_path, png_path, html_path, fsf_path, feat_name = get_paths(
                feat_dir, roi_name, OUTPUT_BASE
            )
            print(f"  -- {feat_name}")

            status = plot(
                csv_path      = csv_path,
                fsf_path      = fsf_path,
                png_path      = png_path,
                html_path     = html_path,
                roi_name      = roi_name,
                feat_name     = feat_name,
                feat_dir      = feat_dir,
                task_color    = TASK_COLOR,
                fd_threshold  = FD_THRESHOLD,
                config_timing = config_timing,
                dry_run       = dry_run,
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
