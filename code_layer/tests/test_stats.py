"""The statistics, against maps with obvious answers and against esda where it agrees.

esda is the reference implementation the spec names. Where a convention differs it is recorded here
rather than papered over, so a later reader knows which form is implemented.
"""

import numpy as np
import pytest

from code_layer import stats, weights
from code_layer.scope import load_scope

SCOPE = load_scope("us_states")
W = weights.build(SCOPE, SCOPE.mainland)
COORDS = np.array([[SCOPE.centroids[c]["x"], SCOPE.centroids[c]["y"]] for c in W.codes])
SOUTH = {c for c in SCOPE.codes if SCOPE.macro_regions[c]["region"] == "South"}

esda = pytest.importorskip("esda", reason="esda not installed")
from esda.getisord import G_Local          # noqa: E402
from esda.moran import Moran, Moran_Local  # noqa: E402
from libpysal.weights import W as PysalW   # noqa: E402


def pysal_weights():
    neighbours = {c: [n for n in SCOPE.adjacency[c] if n in W.index] for c in W.codes}
    w = PysalW(neighbours, silence_warnings=True)
    w.transform = "r"
    return w


def gradient():
    east = COORDS[:, 0]
    return (east - east.mean()) / east.std()


def planted_south():
    return np.array([3.0 if c in SOUTH else 0.0 for c in W.codes])


def test_no_isolates_in_the_analysis_weights():
    """Alaska and Hawaii have no land neighbour; a zero row would divide by zero."""
    assert set(SCOPE.offshore) == {"AK", "HI"}
    assert len(W) == 48
    assert (W.matrix.sum(1) > 0).all()
    assert np.allclose(W.matrix.sum(1), 1.0), "rows must be standardised"


@pytest.mark.parametrize("values", [gradient(), planted_south()])
def test_global_morans_matches_esda(values):
    assert stats.morans_i(values, W) == pytest.approx(Moran(values, pysal_weights()).I, abs=1e-9)


def test_local_morans_matches_esda():
    mine = stats.local_morans(planted_south(), W, permutations=99, seed=0)
    theirs = Moran_Local(planted_south(), pysal_weights(), permutations=99, seed=0)
    assert np.allclose(mine["statistic"], theirs.Is, atol=1e-9)


def test_getis_ord_ranks_like_esda():
    """A different published form (Ord & Getis 1995, as ArcGIS uses), so ranks agree, not values."""
    mine = stats.getis_ord(planted_south(), W, permutations=99, seed=0)["z_score"]
    theirs = G_Local(planted_south(), pysal_weights(), star=True, permutations=99, seed=0).Zs
    assert len(set(np.argsort(-mine)[:10]) & set(np.argsort(-theirs)[:10])) >= 9


def test_clustered_map_is_significant_and_random_is_not():
    clustered = stats.morans_i_test(gradient(), W, permutations=999, seed=0)
    assert clustered["statistic"] > 0.5 and clustered["p_value"] <= 0.05
    scattered = stats.morans_i_test(np.random.default_rng(7).normal(size=len(W)), W,
                                    permutations=999, seed=0)
    assert scattered["p_value"] > 0.05


def test_local_morans_finds_the_planted_cluster_and_nothing_outside_it():
    result = stats.local_morans(planted_south(), W, permutations=999, seed=0)
    hot = {W.codes[i] for i in range(len(W))
           if result["significant"][i] and result["quadrant"][i] == "high_high"}
    assert hot, "a planted cluster must produce significant high-high regions"
    assert hot <= SOUTH, "significance leaked outside the planted cluster"


def test_getis_ord_is_higher_inside_the_planted_cluster():
    z = stats.getis_ord(planted_south(), W, permutations=199, seed=0)["z_score"]
    inside = np.array([z[i] for i, c in enumerate(W.codes) if c in SOUTH])
    outside = np.array([z[i] for i, c in enumerate(W.codes) if c not in SOUTH])
    assert inside.mean() > outside.mean() + 1.0


def test_benjamini_hochberg_is_monotone_and_never_shrinks_a_p_value():
    raw = np.array([0.001, 0.01, 0.02, 0.04, 0.2, 0.9])
    adjusted = stats.benjamini_hochberg(raw)
    assert (adjusted >= raw - 1e-12).all()
    assert (np.diff(adjusted) >= -1e-12).all()
    assert (adjusted <= 1.0).all()


def test_fdr_keeps_false_positives_below_alpha_on_random_maps():
    """50 regions tested at 0.05 would flag 2-3 by chance; the correction is what prevents that."""
    flagged = 0
    trials = 60
    for seed in range(trials):
        values = np.random.default_rng(seed).normal(size=len(W))
        result = stats.local_morans(values, W, permutations=199, seed=seed)
        if result["significant"].any():
            flagged += 1
    assert flagged / trials <= 0.20, f"{flagged}/{trials} random maps produced a significant region"


def test_trend_surface_finds_a_gradient_and_ignores_noise():
    east = COORDS[:, 0]
    found = stats.trend_surface(east, COORDS, W, permutations=299, seed=0)
    assert found["p_east_west"] <= 0.05 and found["r_squared"] > 0.99
    noise = stats.trend_surface(np.random.default_rng(3).normal(size=len(W)), COORDS, W,
                                permutations=299, seed=0)
    assert noise["p_east_west"] > 0.05 and noise["p_south_north"] > 0.05


def test_results_are_reproducible_given_the_seed():
    first = stats.morans_i_test(planted_south(), W, permutations=199, seed=11)
    second = stats.morans_i_test(planted_south(), W, permutations=199, seed=11)
    assert first == second
