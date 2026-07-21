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
├── config_featquery.py            # Config for step 1 (level 1)
├── config_extract.py              # Config for step 2 (level 1)
├── config_plot.py                 # Config for step 3 (level 1)
├── run_featquery.py                # Step 1: run FSL featquery
├── extract_timeseries.py           # Step 2: extract mean BOLD time series
├── plot_timeseries.py              # Step 3: plot time series with task blocks
│
├── config_featquery_level2.py      # Config for step 1 (level 2 / .gfeat)
├── config_extract_level2.py        # Config for step 2 (level 2 / .gfeat)
├── config_plot_level2.py           # Config for step 3 (level 2 / .gfeat)
├── run_featquery_level2.py         # Step 1: run FSL featquery on .gfeat/cope*.feat
├── extract_timeseries_level2.py    # Step 2: extract mean BOLD time series (level 2)
└── plot_timeseries_level2.py       # Step 3: plot time series (level 2)
```

The `level2` scripts are a parallel pipeline for FSL's higher-level (group,
`.gfeat`) outputs rather than first-level `.feat` runs — see
[Level 2 (group) analysis](#level-2-group-analysis) below. Everything else in
this README refers to the level-1 (first-level `.feat`) pipeline unless noted.

---

## Assumptions

- First-level FEAT analyses have already been run
- FEAT directories follow BIDS-style naming:
  ```
  <FEAT_BASE>/sub-<ID>/ses-<ID>/sub-<ID>_ses-<ID>_task-<task>_[run-<N>].feat
  ```
- ROIs are binary NIfTI masks in MNI/standard space
- Task designs are block designs specified as square waves, or as a custom
  3-column EV file, in the `design.fsf`

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

`run_featquery.py` also accepts `--subject sub-XX` on the command line, which
overrides `SUBJECTS` from the config for that invocation.

---

## Usage

Run the scripts in order. Always do a dry run first to verify what will be processed.

### Step 1 — Run featquery

Registers each MNI-space ROI into native functional space and extracts ROI statistics.

```bash
python run_featquery.py --dry-run              # preview
python run_featquery.py
python run_featquery.py --subject sub-CP016     # single subject, overrides config
python run_featquery.py --workers 4             # parallelize across .feat dirs
```

Output: `<feat_dir>/<roi_name>.featquery/` inside each matching `.feat` directory.

### Step 2 — Extract time series

Finds the largest significant cluster overlapping the ROI, builds a combined mask,
and extracts the mean BOLD signal across all voxels at each timepoint. Also pulls
in motion parameters (if FSL's `mc/prefiltered_func_data_mcf.par` exists) and
computes framewise displacement (FD).

```bash
python extract_timeseries.py --dry-run   # preview
python extract_timeseries.py
```

Output per subject/ROI/run:
```
OUTPUT_BASE/<roi_name>/timeseries/
    <feat_name>_<roi_name>_timeseries.csv    # timepoint, mean_bold,
                                              # rot_x/y/z, trans_x/y/z, fd
    <feat_name>_<roi_name>_mask.nii.gz       # combined cluster + ROI mask for QC
```

Motion columns are `NaN` if no `.par` file is found (e.g. for level-2 runs).

A `responder_summary.csv` is also written to `OUTPUT_BASE`, with one row per
subject/ROI/run:

| Column | Description |
|---|---|
| `roi_vox` | Voxel count of the whole native-space ROI mask |
| `extraction_vox` | Voxel count of the largest cluster ∩ ROI (the extraction mask) |
| `all_cluster_vox` | Voxel count across *all* significant clusters ∩ ROI |
| `mean_zstat_roi` | Mean of the raw (unthresholded) `stats/zstat1.nii.gz` across the whole ROI |
| `mean_zstat_extraction` | Mean of the raw z-stat across just the extraction mask (`NaN` for non-responders) |
| `n_suprathreshold_vox` | Voxels in the ROI where FSL's cluster-corrected `thresh_zstat1.nii.gz` is nonzero |
| `mean_suprathreshold_zstat` | Mean of `thresh_zstat1.nii.gz` among those suprathreshold voxels only (`NaN` if none survive) |
| `is_responder` | `True` if `extraction_vox > 0` |

`mean_zstat_roi`/`n_suprathreshold_vox` are computed directly from the FEAT
dir's own stats images, independent of featquery, so they're populated even
for non-responders (no cluster found).

### Step 3 — Plot time series

Reads extracted CSVs and timing (from `config_plot.py`'s `ROI_TASK_MAP["timing"]`,
falling back to `design.fsf`) to produce plots with task block overlays.

```bash
python plot_timeseries.py --dry-run   # preview
python plot_timeseries.py
```

Output per subject/ROI/run:
```
OUTPUT_BASE/<roi_name>/plots/
    <feat_name>_<roi_name>_timeseries.png             # raw BOLD signal
    <feat_name>_<roi_name>_timeseries_zscored.png      # z-scored signal
    <feat_name>_<roi_name>_timeseries_qc.html          # interactive QC (see below)
```

The HTML QC file is a self-contained, interactive 5-panel view (raw BOLD,
z-scored BOLD, rotations, translations, framewise displacement) with task-block
shading and motion-outlier markers (volumes where FD exceeds `FD_THRESHOLD` in
`config_plot.py`). If no motion `.par` file was found at extraction time, the
motion/FD panels are simply omitted.

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
│       ├── <feat_name>_<roi_name>_timeseries_zscored.png
│       └── <feat_name>_<roi_name>_timeseries_qc.html
├── extraction_summary.csv
├── responder_summary.csv
└── plotting_summary.csv
```

---

## Resuming after interruption

Each script skips outputs that already exist, so re-running after a crash will
pick up where it left off without reprocessing completed subjects. In
`plot_timeseries.py`, each output file (raw PNG, z-scored PNG, HTML QC) is
tracked independently, so a partially-completed run only regenerates what's
missing rather than redoing all three.

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
- **Custom EV timing**: `plot_timeseries.py` can parse either a square-wave block
  design (`shape1 = 0`) or a custom 3-column EV file (`shape1 = 3`) from
  `design.fsf`. A `timing` entry in `config_plot.py`'s `ROI_TASK_MAP` always
  takes precedence over both, if present.

---

## Level 2 (group) analysis

`run_featquery_level2.py`, `extract_timeseries_level2.py`, and
`plot_timeseries_level2.py` mirror the level-1 pipeline above, but operate on
FSL's higher-level (group, `.gfeat`) outputs instead of first-level `.feat`
runs:

```
<FEAT_BASE>/sub-<ID>/ses-<ID>/<name>.gfeat/cope<N>.feat/
```

Differences from the level-1 pipeline:

- Each `ROI_TASK_MAP` entry needs a `"copes"` list (e.g. `["cope1.feat"]`)
  naming which `cope*.feat` directories inside each `.gfeat` to process.
- A level-2 `design.fsf` has no block-design EV to parse, so
  `config_plot_level2.py`'s `ROI_TASK_MAP["timing"]` is effectively
  *required* for a task overlay — without it, plots are produced with no
  block shading. TR and `ndelete` are still read from `design.fsf`.
- Level-2 runs have no `mc/*.par` motion file, so the motion/FD CSV columns
  are always `NaN` and the HTML QC output only shows the BOLD panels.

Usage is otherwise identical:
```bash
python run_featquery_level2.py --dry-run
python run_featquery_level2.py
python extract_timeseries_level2.py --dry-run
python extract_timeseries_level2.py
python plot_timeseries_level2.py --dry-run
python plot_timeseries_level2.py
```
