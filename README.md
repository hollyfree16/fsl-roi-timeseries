# fsl-roi-timeseries

A pipeline for extracting and visualising mean BOLD time series from FSL FEAT first-level outputs, constrained to user-defined regions of interest (ROIs).

For each subject and task, the pipeline:
1. Projects an MNI-space ROI into native functional space using FSL `featquery`
2. Identifies the largest significant activation cluster overlapping the ROI
3. Extracts the mean BOLD time series across all voxels in that cluster
4. Plots the time series with task block overlays, in both raw and z-scored form

---

## Requirements

- FSL 6.x (specifically `featquery`)
- Python 3.8+
- Python packages: `numpy`, `nibabel`, `pandas`, `matplotlib`

Install Python dependencies:
```bash
pip install numpy nibabel pandas matplotlib
```

---

## Directory structure

```
fsl-roi-timeseries/
├── config_featquery.py     # Config for step 1
├── config_extract.py       # Config for step 2
├── config_plot.py          # Config for step 3
├── run_featquery.py        # Step 1: run FSL featquery
├── extract_timeseries.py   # Step 2: extract mean BOLD time series
└── plot_timeseries.py      # Step 3: plot time series with task blocks
```

---

## Assumptions

- First-level FEAT analyses have already been run
- FEAT directories follow BIDS-style naming:
  ```
  <FEAT_BASE>/sub-<ID>/ses-<ID>/sub-<ID>_ses-<ID>_task-<task>_[run-<N>].feat
  ```
- ROIs are binary NIfTI masks in MNI/standard space
- Task designs are block designs specified as square waves in the `design.fsf`

---

## Configuration

Each script has its own config file. Edit the config before running the corresponding script.

### Key settings (all configs)

| Setting | Description |
|---|---|
| `FEAT_BASE` | Root directory containing all subject `.feat` directories |
| `OUTPUT_BASE` | Where to save outputs (CSVs, masks, plots) |
| `ROI_TASK_MAP` | List of ROI to task mappings (see below) |
| `SUBJECTS` | Optional subject filter — `None` runs all, or provide a list |

### `ROI_TASK_MAP` format

```python
ROI_TASK_MAP = [
    {
        "roi_path" : "/path/to/ROI_MNI.nii.gz",   # MNI-space ROI (featquery config only)
        "roi_name" : "my_roi",                     # used for output naming
        "tasks"    : ["task-hand", "task-rest"],   # task strings to match
    },
]
```

Task matching is a partial string match on the `.feat` directory name, so `"task-rest"` will match `task-rest_run-01`, `task-rest_run-02`, `task-rest_bold`, etc.

### Running a single subject

In any config file, set:
```python
SUBJECTS = ["sub-XX"]
```

To run all subjects:
```python
SUBJECTS = None
```

---

## Usage

Run the scripts in order. Always do a dry run first to verify what will be processed.

### Step 1 — Run featquery

Registers each MNI-space ROI into native functional space and extracts ROI statistics.

```bash
python run_featquery.py --dry-run   # preview
python run_featquery.py
```

Output: `<feat_dir>/<roi_name>.featquery/` inside each matching `.feat` directory.

### Step 2 — Extract time series

Finds the largest significant cluster overlapping the ROI, builds a combined mask,
and extracts the mean BOLD signal across all voxels at each timepoint.

```bash
python extract_timeseries.py --dry-run   # preview
python extract_timeseries.py
```

Output per subject/ROI/run:
```
OUTPUT_BASE/<roi_name>/timeseries/
    <feat_name>_<roi_name>_timeseries.csv    # columns: timepoint, mean_bold
    <feat_name>_<roi_name>_mask.nii.gz       # combined cluster + ROI mask for QC
```

### Step 3 — Plot time series

Reads extracted CSVs and `design.fsf` timing to produce plots with task block overlays.

```bash
python plot_timeseries.py --dry-run   # preview
python plot_timeseries.py
```

Output per subject/ROI/run:
```
OUTPUT_BASE/<roi_name>/plots/
    <feat_name>_<roi_name>_timeseries.png           # raw BOLD signal
    <feat_name>_<roi_name>_timeseries_zscored.png   # z-scored signal
```

---

## Output structure

```
OUTPUT_BASE/
├── <roi_name>/
│   ├── timeseries/
│   │   ├── <feat_name>_<roi_name>_timeseries.csv
│   │   └── <feat_name>_<roi_name>_mask.nii.gz
│   └── plots/
│       ├── <feat_name>_<roi_name>_timeseries.png
│       └── <feat_name>_<roi_name>_timeseries_zscored.png
├── extraction_summary.csv
└── plotting_summary.csv
```

---

## Resuming after interruption

Each script skips outputs that already exist, so re-running after a crash will
pick up where it left off without reprocessing completed subjects.

---

## Notes

- **Temporal alignment**: The time axis accounts for volumes deleted at the start
  of the run (`ndelete` in `design.fsf`), so plotted task blocks align correctly
  with the extracted signal.
- **QC**: Inspect the `_mask.nii.gz` files in FSLeyes overlaid on `example_func.nii.gz`
  to confirm the cluster/ROI intersection is anatomically sensible before interpreting results.
- **Rest runs**: Rest runs with no block design are handled gracefully — plots are
  produced without task block overlays.
- **Missing stats files**: `run_featquery.py` checks which stats images actually
  exist in each `.feat` directory before building the command, so subjects with
  fewer parameter estimates won't cause errors.
