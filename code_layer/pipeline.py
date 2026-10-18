"""Run the code layer over one derenderer output and return candidate facts.

    from code_layer.pipeline import analyse
    result = analyse(prediction_json, scope_name="us_states")

Everything here is deterministic given the seed: the permutation tests are the only source of
randomness and they are seeded explicitly, so the same input always produces the same facts.
"""

import numpy as np

from . import detectors, schema, stats, weights
from .adapter import from_derenderer

VERSION = "0.1.0"
DEFAULT_SEED = 0

# Facts are emitted in a fixed order so output is stable and diffable. They are not ranked and none
# are dropped: deciding which facts are worth a sentence is the narrator's job, not this layer's.
FACT_ORDER = ["hotspot_cluster", "coldspot_cluster", "spatial_outlier", "directional_trend",
              "mainland_vs_offshore", "extremum", "no_pattern", "missing_data"]


def prepare(map_data, permutations, seed):
    """Everything the detectors share: weights over the analysable regions, and the statistics."""
    scope = map_data.scope
    analysable = [c for c in scope.mainland if map_data.values[c] is not None]
    w = weights.build(scope, analysable)
    if len(w) < 8:
        return None
    x = w.vector(map_data.values)
    coords = np.array([[scope.centroids[c]["x"], scope.centroids[c]["y"]] for c in w.codes])
    return {
        "weights": w,
        "permutations": permutations,
        "moran": stats.morans_i_test(x, w, permutations, seed),
        "local": stats.local_morans(x, w, permutations, seed, detectors.ALPHA),
        "gi": stats.getis_ord(x, w, permutations, seed, detectors.ALPHA),
        "trend": stats.trend_surface(x, coords, w, permutations, seed),
    }


def analyse(payload, scope_name="us_states", permutations=stats.DEFAULT_PERMUTATIONS,
            seed=DEFAULT_SEED):
    map_data = from_derenderer(payload, scope_name)
    prepared = prepare(map_data, permutations, seed)

    facts = []
    for detector in detectors.ALL:
        if prepared is None and detector in detectors.NEEDS_STATISTICS:
            continue
        facts += detector(map_data, prepared=prepared, seed=seed, permutations=permutations)

    facts.sort(key=lambda f: (FACT_ORDER.index(f["type"]) if f["type"] in FACT_ORDER else len(FACT_ORDER),
                              f["regions"]))
    for number, fact in enumerate(facts, 1):
        fact["fact_id"] = f"f{number}"

    observed = [map_data.values[c] for c in map_data.observed]
    return schema.validate({
        "metadata": {
            "code_layer_version": VERSION,
            "map_scope": scope_name,
            "title": map_data.title,
            "legend_title": map_data.legend_title,
            "legend_type": map_data.legend_type,
            "data_type": map_data.data_type,
            "value_kind": map_data.value_kind,
            "values_are_banded": map_data.is_banded,
            "regions_total": len(map_data.scope.codes),
            "regions_with_data": len(map_data.observed),
            "regions_analysed": 0 if prepared is None else len(prepared["weights"]),
            "regions_excluded_offshore": list(map_data.scope.offshore),
            "data_completeness": round(len(map_data.observed) / len(map_data.scope.codes), 4),
            "mean": round(float(np.mean(observed)), 4) if observed else None,
            "median": round(float(np.median(observed)), 4) if observed else None,
            "global_morans_i": None if prepared is None else round(prepared["moran"]["statistic"], 4),
            "global_morans_p": None if prepared is None else round(prepared["moran"]["p_value"], 4),
            "permutations": permutations,
            "seed": seed,
        },
        "facts": facts,
    })
