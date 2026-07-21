# =============================================================================
# config_extract_level2.py
# Configuration for extract_timeseries_level2.py
# =============================================================================

# Root directory containing all subject .gfeat directories
FEAT_BASE = "/autofs/space/nicc_006/data/COMPASS/BIDS/derivatives/fsl_feat_v6.0.7.4/higher_level"

# Where to save extracted time series CSVs and QC masks
OUTPUT_BASE = "/autofs/space/nicc_006/data/COMPASS/BIDS/derivatives/timeseries_extraction_level2"

# Optional subject filter (see config_featquery.py for usage notes)
SUBJECTS = None

# ROI -> task mapping
# roi_name must match the featquery output dir name (<roi_name>.featquery)
# that was created by run_featquery_level2.py
# copes: list of cope*.feat directories to process inside each .gfeat
ROI_TASK_MAP = [
    {
        "roi_name" : "SMA_PMC",
        "tasks"    : ["task-hand", "task-tennis", "task-rest"],
        "copes"    : ["cope1.feat"],
    },
    {
        "roi_name" : "Heschl",
        "tasks"    : ["task-language", "task-rest"],
        "copes"    : ["cope1.feat"],
    },
    {
        "roi_name" : "STG",
        "tasks"    : ["task-language", "task-rest"],
        "copes"    : ["cope1.feat"],
    },
]
