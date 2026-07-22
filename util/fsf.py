"""
util/fsf.py

Shared helpers for parsing timing (TR, ndelete, block onsets/offsets)
out of an FSL design.fsf. Used by plot_timeseries.py (level 1) and
plot_timeseries_level2.py (level 2).
"""

import os
import re


def parse_fsf_block_timing(fsf_path):
    """
    Parse TR, ndelete, and block onsets/offsets from a level-1 FEAT design.fsf.

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
    Return a timing dict for level-1 plotting.

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


def resolve_timing_level2(fsf_path, config_timing=None):
    """
    Return a timing dict for level-2 plotting.

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
