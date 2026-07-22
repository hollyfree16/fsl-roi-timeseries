import numpy as np
import pytest

from util.motion import load_motion_params, compute_fd, motion_columns, find_outliers


def _write_par(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, array)


def test_load_motion_params_missing_file_returns_none(tmp_path):
    feat_dir = tmp_path / "sub-01_task-hand.feat"
    feat_dir.mkdir()
    assert load_motion_params(str(feat_dir)) is None


def test_load_motion_params_reads_par_file(tmp_path):
    feat_dir = tmp_path / "sub-01_task-hand.feat"
    par = np.array([[0.0, 0, 0, 0, 0, 0], [0.01, 0, 0, 1.0, 0, 0]])
    _write_par(feat_dir / "mc" / "prefiltered_func_data_mcf.par", par)

    loaded = load_motion_params(str(feat_dir))
    assert loaded is not None
    np.testing.assert_allclose(loaded, par)


def test_load_motion_params_unreadable_file_returns_none(tmp_path):
    feat_dir = tmp_path / "sub-01_task-hand.feat"
    mc_dir = feat_dir / "mc"
    mc_dir.mkdir(parents=True)
    (mc_dir / "prefiltered_func_data_mcf.par").write_text("not,a,valid,par,file\n")
    assert load_motion_params(str(feat_dir)) is None


def test_compute_fd_first_value_is_zero():
    motion = np.array([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.01, 0.0, 0.0, 1.0, 0.0, 0.0],
    ])
    fd = compute_fd(motion)
    assert fd[0] == 0.0
    # rotation diff 0.01 rad * 50mm + translation diff 1.0mm
    assert fd[1] == pytest.approx(0.01 * 50.0 + 1.0)


def test_motion_columns_none_returns_nan_arrays():
    cols = motion_columns(None, n_tp=5)
    assert set(cols.keys()) == {"rot_x", "rot_y", "rot_z", "trans_x", "trans_y", "trans_z", "fd"}
    for arr in cols.values():
        assert len(arr) == 5
        assert np.all(np.isnan(arr))


def test_motion_columns_length_mismatch_returns_nan(capsys):
    motion = np.zeros((3, 6))
    cols = motion_columns(motion, n_tp=10)
    assert np.all(np.isnan(cols["fd"]))
    assert len(cols["fd"]) == 10
    assert "WARN" in capsys.readouterr().out


def test_motion_columns_matching_length_returns_values():
    motion = np.array([
        [0.1, 0.2, 0.3, 1.0, 2.0, 3.0],
        [0.1, 0.2, 0.3, 1.0, 2.0, 3.0],
    ])
    cols = motion_columns(motion, n_tp=2)
    np.testing.assert_allclose(cols["rot_x"], [0.1, 0.1])
    np.testing.assert_allclose(cols["trans_z"], [3.0, 3.0])
    assert cols["fd"][0] == 0.0


def test_find_outliers_threshold():
    fd = np.array([0.0, 0.2, 0.6, 0.1, 0.9])
    assert find_outliers(fd, threshold=0.5) == [2, 4]
    assert find_outliers(fd, threshold=1.0) == []
