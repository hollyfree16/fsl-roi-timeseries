# =============================================================================
# config_extract.py
# Configuration for extract_timeseries.py
# =============================================================================

# Root directory containing all subject .feat directories
FEAT_BASE = "/autofs/space/nicc_006/data/COMPASS/BIDS/derivatives/fsl_feat_v6.0.7.4/standard"

# Where to save extracted time series CSVs and QC masks
OUTPUT_BASE = "/autofs/space/nicc_006/data/COMPASS/BIDS/derivatives/timeseries_extraction"

# Optional subject filter (see config_featquery.py for usage notes)
SUBJECTS = None

# ROI -> task mapping
# roi_name must match the featquery output dir name (<roi_name>.featquery)
# that was created by run_featquery.py
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
