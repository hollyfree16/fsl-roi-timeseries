# =============================================================================
# config_plot.py
# Configuration for plot_timeseries.py
# =============================================================================

# Root directory containing all subject .feat directories
# (needed to locate design.fsf for timing)
FEAT_BASE = "/autofs/space/nicc_006/data/COMPASS/BIDS/derivatives/fsl_feat_v6.0.7.4/standard"

# Directory where extract_timeseries.py saved its CSVs
# Plots will be saved under OUTPUT_BASE/<roi_name>/plots/
OUTPUT_BASE = "/autofs/space/nicc_006/data/COMPASS/BIDS/derivatives/timeseries_extraction"

# Colour for task ON block bars and shading
TASK_COLOR = "#4C9BE8"

# Optional subject filter (see config_featquery.py for usage notes)
SUBJECTS = None

# ROI -> task mapping
# Must match what was used in config_extract.py
ROI_TASK_MAP = [
    {
        "roi_name" : "SMA_PMC",
        "tasks"    : ["task-hand", "task-tennis", "task-rest"],
    },
    {
        "roi_name" : "Heschl",
        "tasks"    : ["task-language", "task-rest"],
    },
    {
        "roi_name" : "STG",
        "tasks"    : ["task-language", "task-rest"],
    },
]
