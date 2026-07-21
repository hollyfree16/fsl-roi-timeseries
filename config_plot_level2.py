# =============================================================================
# config_plot_level2.py
# Configuration for plot_timeseries_level2.py
# =============================================================================

# Root directory containing all subject .gfeat directories
FEAT_BASE = "/autofs/space/nicc_006/data/COMPASS/BIDS/derivatives/fsl_feat_v6.0.7.4/higher_level"

# Directory where extract_timeseries_level2.py saved its CSVs
OUTPUT_BASE = "/autofs/space/nicc_006/data/COMPASS/BIDS/derivatives/timeseries_extraction_level2"

# Colour for task ON block bars and shading
TASK_COLOR = "#4C9BE8"

# Framewise displacement threshold (mm) for flagging motion outliers in the
# HTML QC plot. Level-2 (.gfeat) runs have no motion .par file, so the FD
# and motion panels won't appear — this setting is kept only for parity
# with config_plot.py.
FD_THRESHOLD = 0.5

# Optional subject filter (see config_featquery.py for usage notes)
SUBJECTS = None

# ROI -> task mapping
# Must match what was used in config_extract_level2.py
# copes: list of cope*.feat directories to process inside each .gfeat
#
# "timing" is effectively REQUIRED at level 2: a group-level design.fsf has
# no block-design EV to parse (it's a fixed-effects/OLS design over copes),
# so without an explicit "timing" entry, plots will have no task overlay.
# TR and ndelete are still read from design.fsf either way.
#   "timing": {
#       "block_onsets"  : [<onset_s>, ...],
#       "block_offsets" : [<offset_s>, ...],
#   }
ROI_TASK_MAP = [
    {
        "roi_name" : "SMA_PMC",
        "tasks"    : ["task-hand", "task-tennis", "task-rest"],
        "copes"    : ["cope1.feat"],
        "timing"   : None,  # fill in block_onsets / block_offsets (seconds)
    },
    {
        "roi_name" : "Heschl",
        "tasks"    : ["task-language", "task-rest"],
        "copes"    : ["cope1.feat"],
        "timing"   : None,
    },
    {
        "roi_name" : "STG",
        "tasks"    : ["task-language", "task-rest"],
        "copes"    : ["cope1.feat"],
        "timing"   : None,
    },
]
