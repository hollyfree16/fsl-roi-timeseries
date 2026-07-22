import numpy as np
import nibabel as nib

from util.zstat import peak_voxel, compute_zstat_metrics


def test_peak_voxel_empty_mask_returns_nan():
    data = np.zeros((2, 2, 2))
    mask = np.zeros((2, 2, 2), dtype=bool)
    value, idx = peak_voxel(data, mask)
    assert np.isnan(value)
    assert idx == (None, None, None)


def test_peak_voxel_finds_max_within_mask():
    data = np.array([[[1.0, 5.0], [2.0, 3.0]], [[4.0, 9.0], [0.0, 1.0]]])
    mask = np.ones((2, 2, 2), dtype=bool)
    value, idx = peak_voxel(data, mask)
    assert value == 9.0
    assert idx == (1, 0, 1)


def test_peak_voxel_ignores_values_outside_mask():
    data = np.array([[[1.0, 9.0]]])  # the 9.0 lives outside the mask below
    mask = np.array([[[True, False]]])
    value, idx = peak_voxel(data, mask)
    assert value == 1.0
    assert idx == (0, 0, 0)


def _save_nifti(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data.astype(np.float32), affine=np.eye(4)), str(path))


def test_compute_zstat_metrics_with_all_files_present(tmp_path):
    zstat = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])
    thresh = np.array([[[0.0, 2.0], [0.0, 4.0]], [[0.0, 0.0], [0.0, 8.0]]])
    _save_nifti(tmp_path / "stats" / "zstat1.nii.gz", zstat)
    _save_nifti(tmp_path / "thresh_zstat1.nii.gz", thresh)

    roi_bin = np.ones((2, 2, 2), dtype=int)
    extraction_mask = np.zeros((2, 2, 2), dtype=int)
    extraction_mask[1, 1, 1] = 1  # value 8.0

    metrics = compute_zstat_metrics(str(tmp_path), roi_bin, extraction_mask)

    assert metrics["mean_zstat_roi"] == np.mean(zstat)
    assert metrics["peak_zstat_roi"] == 8.0
    assert metrics["mean_zstat_extraction"] == 8.0
    assert metrics["peak_zstat_extraction"] == 8.0
    assert metrics["n_suprathreshold_vox"] == 3  # nonzero entries in thresh
    assert metrics["mean_suprathreshold_zstat"] == np.mean([2.0, 4.0, 8.0])


def test_compute_zstat_metrics_missing_files_returns_nan(tmp_path, capsys):
    roi_bin = np.ones((2, 2, 2), dtype=int)
    metrics = compute_zstat_metrics(str(tmp_path), roi_bin)

    assert np.isnan(metrics["mean_zstat_roi"])
    assert metrics["n_suprathreshold_vox"] is None
    out = capsys.readouterr().out
    assert "zstat1.nii.gz" in out
    assert "thresh_zstat1.nii.gz" in out


def test_compute_zstat_metrics_without_extraction_mask(tmp_path):
    zstat = np.ones((2, 2, 2))
    _save_nifti(tmp_path / "stats" / "zstat1.nii.gz", zstat)

    roi_bin = np.ones((2, 2, 2), dtype=int)
    metrics = compute_zstat_metrics(str(tmp_path), roi_bin, extraction_mask=None)

    assert metrics["mean_zstat_roi"] == 1.0
    assert np.isnan(metrics["mean_zstat_extraction"])
