from util.fsf import (
    parse_fsf_block_timing,
    resolve_timing,
    parse_fsf_tr_ndelete,
    resolve_timing_level2,
)

SQUARE_WAVE_FSF = """\
set fmri(tr) 2.0
set fmri(npts) 100
set fmri(ndelete) 4
set fmri(shape1) 0
set fmri(off1) 20
set fmri(on1) 20
set fmri(phase1) 0
set fmri(stop1) -1
"""

REST_FSF = """\
set fmri(tr) 2.0
set fmri(npts) 50
set fmri(ndelete) 2
"""

CUSTOM_EV_FSF = """\
set fmri(tr) 1.5
set fmri(npts) 80
set fmri(ndelete) 0
set fmri(shape1) 3
set fmri(custom1) timing.txt
"""


def test_parse_fsf_block_timing_square_wave(tmp_path):
    fsf_path = tmp_path / "design.fsf"
    fsf_path.write_text(SQUARE_WAVE_FSF)

    timing = parse_fsf_block_timing(str(fsf_path))
    assert timing["tr"] == 2.0
    assert timing["ndelete"] == 4
    assert timing["has_blocks"] is True
    assert timing["block_onsets"][0] == 20.0
    assert timing["block_offsets"][0] == 40.0
    # blocks alternate every 40s (20s off + 20s on) up to (100-4)*2=192s
    assert timing["block_onsets"][1] == 60.0


def test_parse_fsf_block_timing_rest_run_has_no_blocks(tmp_path):
    fsf_path = tmp_path / "design.fsf"
    fsf_path.write_text(REST_FSF)

    timing = parse_fsf_block_timing(str(fsf_path))
    assert timing["has_blocks"] is False
    assert timing["block_onsets"] == []


def test_parse_fsf_block_timing_custom_ev(tmp_path):
    fsf_path = tmp_path / "design.fsf"
    fsf_path.write_text(CUSTOM_EV_FSF)
    (tmp_path / "timing.txt").write_text("10 5 1\n20 5 0\n35 2.5 1\n")

    timing = parse_fsf_block_timing(str(fsf_path))
    assert timing["has_blocks"] is True
    # weight 0 row should be excluded
    assert timing["block_onsets"] == [10.0, 35.0]
    assert timing["block_offsets"] == [15.0, 37.5]


def test_resolve_timing_prefers_config_over_fsf(tmp_path):
    fsf_path = tmp_path / "design.fsf"
    fsf_path.write_text(SQUARE_WAVE_FSF)

    config_timing = {"block_onsets": [5.0], "block_offsets": [10.0]}
    timing = resolve_timing(str(fsf_path), config_timing=config_timing)

    assert timing["timing_source"] == "config"
    assert timing["block_onsets"] == [5.0]
    assert timing["tr"] == 2.0  # tr/ndelete always come from fsf


def test_resolve_timing_falls_back_to_fsf(tmp_path):
    fsf_path = tmp_path / "design.fsf"
    fsf_path.write_text(SQUARE_WAVE_FSF)

    timing = resolve_timing(str(fsf_path), config_timing=None)
    assert timing["timing_source"] == "fsf"
    assert timing["has_blocks"] is True


def test_parse_fsf_tr_ndelete(tmp_path):
    fsf_path = tmp_path / "design.fsf"
    fsf_path.write_text(SQUARE_WAVE_FSF)

    tr, ndelete = parse_fsf_tr_ndelete(str(fsf_path))
    assert tr == 2.0
    assert ndelete == 4


def test_resolve_timing_level2_without_config_has_no_blocks(tmp_path):
    fsf_path = tmp_path / "design.fsf"
    fsf_path.write_text(SQUARE_WAVE_FSF)

    timing = resolve_timing_level2(str(fsf_path), config_timing=None)
    assert timing["timing_source"] == "none"
    assert timing["has_blocks"] is False


def test_resolve_timing_level2_with_config(tmp_path):
    fsf_path = tmp_path / "design.fsf"
    fsf_path.write_text(SQUARE_WAVE_FSF)

    config_timing = {"block_onsets": [5.0, 15.0], "block_offsets": [10.0, 20.0]}
    timing = resolve_timing_level2(str(fsf_path), config_timing=config_timing)
    assert timing["timing_source"] == "config"
    assert timing["has_blocks"] is True
    assert timing["block_onsets"] == [5.0, 15.0]
