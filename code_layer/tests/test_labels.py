"""Region labelling: the rule that stops a two-state cluster being called "the South"."""

from code_layer.labels import direction_label, label_cluster, phrase
from code_layer.scope import load_scope

SCOPE = load_scope("us_states")
SOUTH = [c for c in SCOPE.codes if SCOPE.macro_regions[c]["region"] == "South"]
NEW_ENGLAND = [c for c in SCOPE.codes if SCOPE.macro_regions[c]["division"] == "New England"]


def test_reference_data_covers_every_region():
    assert len(SCOPE.codes) == 50
    assert all(c in SCOPE.macro_regions and c in SCOPE.centroids for c in SCOPE.codes)
    assert sum(len(v) for v in SCOPE.groups("region").values()) == 50
    assert sum(len(v) for v in SCOPE.groups("division").values()) == 50


def test_whole_region_is_named_outright():
    label = label_cluster(SOUTH, SCOPE)
    assert label["label"] == "South" and label["coverage"] == "whole"
    assert phrase(label) == "the South"


def test_two_states_do_not_become_the_whole_region():
    """Precision alone is 1.0 against the South, which is exactly how overclaiming happens."""
    label = label_cluster(["AL", "MS"], SCOPE)
    assert label["precision"] == 1.0
    assert label["label"] != "South", "two states were described as the whole South"
    # The tighter division is the honest label: two of its four states.
    assert label["label"] == "East South Central" and label["recall"] == 0.5
    assert phrase(label) == "much of the East South Central"


def test_a_single_state_is_only_part_of_its_division():
    label = label_cluster(["AL"], SCOPE)
    assert label["coverage"] == "part"
    assert phrase(label).startswith("part of")


def test_the_narrower_division_wins_over_the_broader_region():
    """New England is covered entirely; the Northeast only two thirds."""
    label = label_cluster(NEW_ENGLAND, SCOPE)
    assert label["label"] == "New England" and label["level"] == "division"
    assert label["coverage"] == "whole"


def test_a_cluster_spanning_regions_falls_back_to_direction():
    scattered = ["CA", "ME", "FL", "WA", "TX", "MN"]
    label = label_cluster(scattered, SCOPE)
    assert label["level"] == "direction"
    assert label["precision"] is None


def test_direction_label_reads_off_the_map():
    assert "western" in direction_label(["CA", "OR", "WA", "NV"], SCOPE)["label"]
    assert "eastern" in direction_label(["ME", "NH", "VT", "MA"], SCOPE)["label"]
    assert "southern" in direction_label(["TX", "LA", "MS", "AL"], SCOPE)["label"]


def test_phrasing_never_claims_more_coverage_than_measured():
    for codes in (SOUTH, ["AL", "MS"], NEW_ENGLAND, ["CA", "ME", "FL"]):
        label = label_cluster(codes, SCOPE)
        wording = phrase(label)
        assert wording and isinstance(wording, str)
        if label.get("coverage") == "part":
            assert wording.startswith("part of")


def test_empty_cluster_has_no_label():
    assert label_cluster([], SCOPE) is None
