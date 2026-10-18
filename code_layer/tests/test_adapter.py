"""The input contract, and the band-label parser it depends on."""

import json
from pathlib import Path

import pytest

from code_layer.adapter import ContractError, band_midpoints, from_derenderer, parse_band
from code_layer.scope import load_scope

SCOPE = load_scope("us_states")
PROJECT = Path(__file__).resolve().parents[2]
REAL_LABELS = PROJECT / "data" / "kaggle_qwen" / "data" / "test.jsonl"


def a_map(**overrides):
    payload = {"title": "Test", "legend_title": "Value", "legend_type": "continuous",
               "data_type": "number", "values": {c: 1.0 for c in SCOPE.codes}}
    payload.update(overrides)
    return payload


@pytest.mark.parametrize("label,expected", [
    ("10-20", (10.0, 20.0)),
    ("10–20", (10.0, 20.0)),          # en dash
    ("10 – 20", (10.0, 20.0)),
    ("10 to 20", (10.0, 20.0)),
    ("5.5%-7.2%", (5.5, 7.2)),
    ("1,200-3,400", (1200.0, 3400.0)),
    ("$5-$10", (5.0, 10.0)),
    ("-5-10", (-5.0, 10.0)),               # a negative lower bound, not three numbers
    ("69266-73392", (69266.0, 73392.0)),
])
def test_closed_bands_parse(label, expected):
    assert parse_band(label) == expected


@pytest.mark.parametrize("label", [
    "< 10", "Under 10", "Less than 10", "≥ 30", "30 or more", "30 and above", "30+", "over 40",
])
def test_open_ended_bands_are_refused(label):
    """An open band has no midpoint, and inventing one would put a number in the reader's mouth."""
    assert parse_band(label) is None


def test_one_open_band_disables_midpoints_for_the_whole_map():
    """Mixing midpoints and indices would put regions of one map on two different scales."""
    assert band_midpoints(["0-10", "10-20", "20-30", "30-40"]) == [5.0, 15.0, 25.0, 35.0]
    assert band_midpoints(["< 10", "10-20", "20-30", "30+"]) is None


def test_discrete_map_uses_midpoints_when_it_can():
    payload = a_map(legend_type="discrete", ranges=["0-10", "10-20"],
                    values={c: 0 for c in SCOPE.codes})
    data = from_derenderer(payload, "us_states")
    assert data.value_kind == "band_midpoint"
    assert data.values["AL"] == 5.0


def test_discrete_map_falls_back_to_indices():
    payload = a_map(legend_type="discrete", ranges=["< 10", "10-20"],
                    values={c: 1 for c in SCOPE.codes})
    data = from_derenderer(payload, "us_states")
    assert data.value_kind == "band_index"
    assert data.values["AL"] == 1.0


def test_missing_region_is_rejected():
    values = {c: 1.0 for c in SCOPE.codes}
    del values["AL"]
    with pytest.raises(ContractError, match="missing"):
        from_derenderer(a_map(values=values), "us_states")


def test_invented_region_is_rejected():
    values = {c: 1.0 for c in SCOPE.codes}
    values["ZZ"] = 5.0
    with pytest.raises(ContractError, match="unexpected"):
        from_derenderer(a_map(values=values), "us_states")


def test_band_index_outside_the_legend_is_rejected():
    values = {c: 0 for c in SCOPE.codes}
    values["AL"] = 9
    with pytest.raises(ContractError, match="outside"):
        from_derenderer(a_map(legend_type="discrete", ranges=["0-10", "10-20"], values=values),
                        "us_states")


def test_unparseable_output_is_rejected():
    with pytest.raises(ContractError):
        from_derenderer("not json at all", "us_states")


def test_null_stays_null_and_is_not_a_zero():
    values = {c: 1.0 for c in SCOPE.codes}
    values["CA"] = None
    data = from_derenderer(a_map(values=values), "us_states")
    assert data.values["CA"] is None
    assert "CA" not in data.observed and "CA" in data.missing


def test_reported_extremes_are_carried_and_checked():
    data = from_derenderer(a_map(max_state="TX", min_state="CA"), "us_states")
    assert data.reported_extremes == {"max": "TX", "min": "CA"}
    with pytest.raises(ContractError, match="max_state"):
        from_derenderer(a_map(max_state="ZZ"), "us_states")


@pytest.mark.skipif(not REAL_LABELS.exists(), reason="MapQA labels not downloaded")
def test_every_real_mapqa_target_converts():
    """The parser is only useful if it handles the corpus, not just the examples above."""
    rows = [json.loads(line) for line in REAL_LABELS.open()]
    kinds = {"value": 0, "band_midpoint": 0, "band_index": 0}
    for row in rows:
        kinds[from_derenderer(row["target"], "us_states").value_kind] += 1
    assert sum(kinds.values()) == len(rows)
    assert kinds["band_index"] == 0, "MapQA has no open-ended bands; a fallback here means a parser bug"
    assert kinds["value"] and kinds["band_midpoint"]
