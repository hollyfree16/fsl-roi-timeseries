# =============================================================================
# config_featquery.py
# Configuration for run_featquery.py
# =============================================================================

FEATQUERY_BIN = "/usr/pubsw/packages/fsl/6.0.7.4/bin/featquery"

# Root directory containing all subject .feat directories
# Structure: FEAT_BASE/sub-XX/ses-XX/<feat_dir>.feat
FEAT_BASE = "/autofs/space/nicc_006/data/COMPASS/BIDS/derivatives/fsl_feat_v6.0.7.4/standard"

# Stats images to request from featquery.
# The script will check which of these actually exist in each .feat dir
# before building the command, so missing files won't cause crashes.
STATS_IMAGES = [
    "stats/pe1",
    "stats/pe2",
    "stats/pe3",
    "stats/pe4",
    "stats/pe5",
    "stats/pe6",
    "stats/pe7",
    "stats/pe8",
    "stats/cope1",
    "stats/varcope1",
    "stats/tstat1",
    "stats/fstat1",
    "stats/zstat1",
    "stats/zfstat1",
    "thresh_zstat1",
    "thresh_zfstat1",
]

# featquery flags
# -a 4  : interpolation order for registration
# -p    : output percent signal change
# -s    : use standard space mask
FEATQUERY_FLAGS = ["-a", "4", "-p", "-s"]

# Optional subject filter.
# Set to a list of subject IDs to process only those subjects,
# or None to process all subjects found under FEAT_BASE.
# Examples:
#   SUBJECTS = ["sub-CP016"]
#   SUBJECTS = ["sub-R2c001", "sub-R2p003"]
#   SUBJECTS = None   # run all
SUBJECTS = None

# ROI -> task mapping
# roi_path  : full path to ROI NIfTI in MNI/standard space
# roi_name  : used as the featquery output subdirectory name (<roi_name>.featquery)
# tasks     : list of task- strings to match against .feat directory names
ROI_TASK_MAP = [
    {
        "roi_path" : "/autofs/space/nicc_006/data/COMPASS/BIDS/code/rois/SMA_PMC.nii.gz",
        "roi_name" : "SMA_PMC",
        "tasks"    : ["task-hand", "task-tennis", "task-rest"],
    },
    {
        "roi_path" : "/autofs/space/nicc_006/data/COMPASS/BIDS/code/rois/Heschl.nii.gz",
        "roi_name" : "Heschl",
        "tasks"    : ["task-language", "task-rest"],
    },
    {
        "roi_path" : "/autofs/space/nicc_006/data/COMPASS/BIDS/code/rois/STG.nii.gz",
        "roi_name" : "STG",
        "tasks"    : ["task-language", "task-rest"],
    },
]
