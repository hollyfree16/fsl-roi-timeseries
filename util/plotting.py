"""
util/plotting.py

Shared PNG plotting and BIDS-name-parsing helpers used by both
plot_timeseries.py (level 1) and plot_timeseries_level2.py (level 2).
The HTML QC output differs meaningfully between levels (level 2 has no
motion panels), so save_html stays script-specific.
"""

import os
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


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


def parse_feat_meta(name):
    """Extract sub/ses/task/run from a BIDS-style feat/gfeat/cope run name."""
    fields = {}
    for key in ("sub", "ses", "task", "run"):
        m = re.search(rf'({key}-[^_]+)', name)
        fields[key] = m.group(1) if m else ""
    return fields
