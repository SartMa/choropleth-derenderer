"""End to end: hand-built maps with obvious answers, plus the real MapQA predictions."""

import json
from pathlib import Path

import pytest

from code_layer import schema
from code_layer.pipeline import analyse
from code_layer.scope import load_scope

SCOPE = load_scope("us_states")
PROJECT = Path(__file__).resolve().parents[2]
REAL_PREDICTIONS = PROJECT / "outputs" / "mapqa-derender-qwen" / "qwen" / "predictions_mapqa.jsonl"
SOUTH = {c for c in SCOPE.codes if SCOPE.macro_regions[c]["region"] == "South"}
FAST = 199


def payload(values, **extra):
    body = {"title": "Test map", "legend_title": "Value", "legend_type": "continuous",
            "data_type": "number", "values": {c: values.get(c) for c in SCOPE.codes}}
    body.update(extra)
    return json.dumps(body)


def of_type(result, kind):
    return [f for f in result["facts"] if f["type"] == kind]


def test_planted_cluster_is_found_and_labelled_conservatively():
    result = analyse(payload({c: (10.0 if c in SOUTH else 1.0) for c in SCOPE.codes}))
    hotspots = of_type(result, "hotspot_cluster")
    assert hotspots, "a planted cluster must produce a hotspot fact"
    for fact in hotspots:
        assert set(fact["regions"]) <= SOUTH, "the hotspot leaked outside the planted cluster"
        # Only the significant core is reported, so the label describes that core, not what was
        # planted: claiming "the South" from four states is the overclaiming this rule prevents.
        assert fact["label_precision"] == 1.0
        if fact["label_recall"] < 0.5:
            assert fact["label_coverage"] == "part"
            assert fact["phrasing"].startswith("part of")


def test_random_map_reports_no_pattern_and_no_clusters():
    import numpy as np
    values = dict(zip(SCOPE.codes, np.random.default_rng(5).normal(size=len(SCOPE.codes))))
    result = analyse(payload(values))
    assert of_type(result, "no_pattern")
    assert not of_type(result, "hotspot_cluster") and not of_type(result, "coldspot_cluster")


def test_gradient_produces_a_directional_trend():
    result = analyse(payload({c: SCOPE.centroids[c]["x"] / 1e6 for c in SCOPE.codes}))
    trends = of_type(result, "directional_trend")
    assert any(f["axis"] == "east_west" for f in trends)
    assert all("caveat" in f for f in trends)


def test_missing_regions_are_reported_and_never_become_zero():
    values = {c: 5.0 for c in SCOPE.codes}
    for code in ("CA", "TX", "NY"):
        values[code] = None
    result = analyse(payload(values), permutations=FAST)
    missing = of_type(result, "missing_data")
    assert missing and set(missing[0]["regions"]) == {"CA", "TX", "NY"}
    assert result["metadata"]["regions_with_data"] == 47
    assert result["metadata"]["mean"] == 5.0, "a null leaked into the mean as a zero"


def test_banded_map_never_reports_a_measured_mean():
    """The underlying numbers were never in the image, so no fact may state one."""
    body = json.dumps({"title": "Banded", "legend_title": None, "legend_type": "discrete",
                       "data_type": "percentage", "ranges": ["0-10", "10-20", "20-30", "30-40"],
                       "values": {c: (3 if c in SOUTH else 0) for c in SCOPE.codes}})
    result = analyse(body)
    assert result["metadata"]["values_are_banded"] is True
    for fact in result["facts"]:
        assert "cluster_mean" not in fact, "a banded map reported a mean as though measured"
        assert "vs_national_mean_pct" not in fact


def test_banded_extremum_admits_when_it_cannot_single_one_out():
    body = json.dumps({"title": "Banded", "legend_title": None, "legend_type": "discrete",
                       "data_type": "number", "ranges": ["0-10", "10-20"],
                       "values": {c: (1 if c in SOUTH else 0) for c in SCOPE.codes}})
    top = [f for f in of_type(analyse(body, permutations=FAST), "extremum") if f["kind"] == "max"][0]
    assert top["unique"] is False
    assert set(top["tied_regions"]) == SOUTH


def test_reported_extreme_that_disagrees_is_flagged_as_contested():
    values = {c: 1.0 for c in SCOPE.codes}
    values["TX"] = 9.0
    agreeing = analyse(payload(values, max_state="TX"), permutations=FAST)
    top = [f for f in of_type(agreeing, "extremum") if f["kind"] == "max"][0]
    assert top["agrees_with_values"] is True and "contested_with" not in top

    disagreeing = analyse(payload(values, max_state="CA"), permutations=FAST)
    top = [f for f in of_type(disagreeing, "extremum") if f["kind"] == "max"][0]
    assert top["agrees_with_values"] is False and top["contested_with"] == "CA"


def test_output_is_deterministic_for_a_seed():
    body = payload({c: (10.0 if c in SOUTH else 1.0) for c in SCOPE.codes})
    first, second = analyse(body, permutations=FAST), analyse(body, permutations=FAST)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_every_fact_carries_provenance_and_a_unique_id():
    result = analyse(payload({c: (10.0 if c in SOUTH else 1.0) for c in SCOPE.codes}),
                     permutations=FAST)
    ids = [f["fact_id"] for f in result["facts"]]
    assert len(ids) == len(set(ids))
    for fact in result["facts"]:
        assert fact["provenance"]["detector"] and fact["provenance"]["statistic"]


def test_schema_rejects_an_invented_fact_type():
    result = analyse(payload({c: 1.0 for c in SCOPE.codes}), permutations=FAST)
    result["facts"].append({"fact_id": "fX", "type": "vibe_check", "regions": [],
                            "p_value": None, "provenance": {"detector": "d", "statistic": "s"}})
    with pytest.raises(schema.SchemaError, match="unknown fact type"):
        schema.validate(result)


def test_schema_rejects_an_undeclared_field():
    result = analyse(payload({c: 1.0 for c in SCOPE.codes}), permutations=FAST)
    result["facts"][0]["surprise"] = 1
    with pytest.raises(schema.SchemaError, match="outside the schema"):
        schema.validate(result)


@pytest.mark.skipif(not REAL_PREDICTIONS.exists(), reason="model predictions not downloaded")
def test_real_predictions_all_analyse_without_crashing():
    rows = [json.loads(line) for line in REAL_PREDICTIONS.open()][:40]
    for row in rows:
        schema.validate(analyse(row["prediction"], permutations=FAST))


@pytest.mark.skipif(not REAL_PREDICTIONS.exists(), reason="model predictions not downloaded")
def test_false_positive_rate_on_mapqa_is_low():
    """MapQA-S values are randomly generated, so clusters found here are false positives."""
    rows = [json.loads(line) for line in REAL_PREDICTIONS.open()][:60]
    with_clusters = 0
    for row in rows:
        result = analyse(row["prediction"], permutations=FAST)
        if of_type(result, "hotspot_cluster") or of_type(result, "coldspot_cluster"):
            with_clusters += 1
    assert with_clusters / len(rows) <= 0.10, f"{with_clusters}/{len(rows)} random maps grew a cluster"
