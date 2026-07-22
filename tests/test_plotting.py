import numpy as np

from util.plotting import save_plot, parse_feat_meta


def test_parse_feat_meta_extracts_bids_fields():
    # Trailing fields with no following "_" keep whatever suffix follows
    # (e.g. ".feat") since the field regex stops only at the next underscore.
    meta = parse_feat_meta("sub-CP016_ses-01_task-hand_run-01.feat")
    assert meta["sub"] == "sub-CP016"
    assert meta["ses"] == "ses-01"
    assert meta["task"] == "task-hand"
    assert meta["run"] == "run-01.feat"


def test_parse_feat_meta_missing_fields_are_empty_string():
    meta = parse_feat_meta("sub-CP016_task-hand.feat")
    assert meta["sub"] == "sub-CP016"
    assert meta["ses"] == ""
    assert meta["run"] == ""


def test_save_plot_writes_png_file(tmp_path):
    time_s = np.arange(20, dtype=float)
    signal = np.sin(time_s)
    timing = {
        "has_blocks": True,
        "block_onsets": [2.0, 10.0],
        "block_offsets": [5.0, 14.0],
    }
    output_path = tmp_path / "plots" / "example_timeseries.png"

    save_plot(
        time_s=time_s, signal=signal, timing=timing,
        title="sub-01 | ROI — Raw", ylabel="Mean BOLD (a.u.)",
        output_path=str(output_path), task_color="#4C9BE8",
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_save_plot_without_blocks(tmp_path):
    time_s = np.arange(10, dtype=float)
    signal = np.zeros(10)
    timing = {"has_blocks": False, "block_onsets": [], "block_offsets": []}
    output_path = tmp_path / "rest_timeseries.png"

    save_plot(
        time_s=time_s, signal=signal, timing=timing,
        title="sub-01 | ROI — Raw", ylabel="Z-score",
        output_path=str(output_path), task_color="#4C9BE8",
    )

    assert output_path.exists()
