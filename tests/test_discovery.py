from util.discovery import find_feat_dirs, find_cope_dirs, get_existing_stats


def _mkfeat(base, sub, ses, name):
    d = base / sub / ses / name
    d.mkdir(parents=True)
    return d


def test_find_feat_dirs_filters_by_task(tmp_path):
    _mkfeat(tmp_path, "sub-01", "ses-01", "sub-01_ses-01_task-hand_run-01.feat")
    _mkfeat(tmp_path, "sub-01", "ses-01", "sub-01_ses-01_task-rest.feat")
    _mkfeat(tmp_path, "sub-02", "ses-01", "sub-02_ses-01_task-hand_run-01.feat")

    dirs = find_feat_dirs(str(tmp_path), ["task-hand"])
    assert len(dirs) == 2
    assert all("task-hand" in d for d in dirs)


def test_find_feat_dirs_filters_by_subject(tmp_path):
    _mkfeat(tmp_path, "sub-01", "ses-01", "sub-01_ses-01_task-hand_run-01.feat")
    _mkfeat(tmp_path, "sub-02", "ses-01", "sub-02_ses-01_task-hand_run-01.feat")

    dirs = find_feat_dirs(str(tmp_path), ["task-hand"], subjects=["sub-02"])
    assert len(dirs) == 1
    assert "sub-02" in dirs[0]


def test_find_cope_dirs_returns_gfeat_cope_pairs(tmp_path):
    gfeat = tmp_path / "sub-01" / "ses-01" / "sub-01_ses-01_task-hand.gfeat"
    (gfeat / "cope1.feat").mkdir(parents=True)
    (gfeat / "cope2.feat").mkdir(parents=True)

    results = find_cope_dirs(str(tmp_path), ["task-hand"], ["cope1.feat"])
    assert len(results) == 1
    gfeat_dir, cope_dir = results[0]
    assert gfeat_dir.endswith(".gfeat")
    assert cope_dir.endswith("cope1.feat")


def test_find_cope_dirs_warns_on_missing_cope(tmp_path, capsys):
    gfeat = tmp_path / "sub-01" / "ses-01" / "sub-01_ses-01_task-hand.gfeat"
    gfeat.mkdir(parents=True)

    results = find_cope_dirs(str(tmp_path), ["task-hand"], ["cope1.feat"])
    assert results == []
    assert "WARN" in capsys.readouterr().out


def test_get_existing_stats_checks_both_extensions(tmp_path):
    stats_dir = tmp_path / "stats"
    stats_dir.mkdir()
    (stats_dir / "pe1.nii.gz").touch()
    (tmp_path / "thresh_zstat1").touch()  # no extension, stored FSL-internal style

    existing = get_existing_stats(str(tmp_path), ["stats/pe1", "stats/pe2", "thresh_zstat1"])
    assert existing == ["stats/pe1", "thresh_zstat1"]
