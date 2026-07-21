# =============================================================================
# config_featquery_level2.py
# Configuration for run_featquery_level2.py
# =============================================================================

FEATQUERY_BIN = "/usr/pubsw/packages/fsl/6.0.7.4/bin/featquery"

# Root directory containing all subject .gfeat directories (group/level-2 outputs)
# Structure: FEAT_BASE/sub-XX/ses-XX/<name>.gfeat/cope<N>.feat
FEAT_BASE = "/autofs/space/nicc_006/data/COMPASS/BIDS/derivatives/fsl_feat_v6.0.7.4/higher_level"

# Stats images to request from featquery.
# The script will check which of these actually exist in each cope*.feat dir
# before building the command, so missing files won't cause crashes.
STATS_IMAGES = [
    "stats/cope1",
    "stats/varcope1",
    "stats/tstat1",
    "stats/zstat1",
    "thresh_zstat1",
]

# featquery flags
# -a 4  : interpolation order for registration
# -p    : output percent signal change
# -s    : use standard space mask
FEATQUERY_FLAGS = ["-a", "4", "-p", "-s"]

# Optional subject filter.
# Set to a list of subject IDs to process only those subjects,
# or None to process all subjects found under FEAT_BASE.
SUBJECTS = None

# ROI -> task mapping
# roi_path  : full path to ROI NIfTI in MNI/standard space
# roi_name  : used as the featquery output subdirectory name (<roi_name>.featquery)
# tasks     : list of task- strings to match against .gfeat directory names
# copes     : list of cope*.feat directory names to process inside each .gfeat
ROI_TASK_MAP = [
    {
        "roi_path" : "/autofs/space/nicc_006/data/COMPASS/BIDS/code/rois/SMA_PMC.nii.gz",
        "roi_name" : "SMA_PMC",
        "tasks"    : ["task-hand", "task-tennis", "task-rest"],
        "copes"    : ["cope1.feat"],
    },
    {
        "roi_path" : "/autofs/space/nicc_006/data/COMPASS/BIDS/code/rois/Heschl.nii.gz",
        "roi_name" : "Heschl",
        "tasks"    : ["task-language", "task-rest"],
        "copes"    : ["cope1.feat"],
    },
    {
        "roi_path" : "/autofs/space/nicc_006/data/COMPASS/BIDS/code/rois/STG.nii.gz",
        "roi_name" : "STG",
        "tasks"    : ["task-language", "task-rest"],
        "copes"    : ["cope1.feat"],
    },
]
