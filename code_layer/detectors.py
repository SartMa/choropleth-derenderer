"""One function per kind of fact. Each takes the prepared map and returns zero or more facts.

Every fact carries provenance: which detector produced it, from which statistic, over which regions and
values. That is what lets a later claims-verifier re-derive any sentence in the summary, and it is far
cheaper to record now than to retrofit.

Facts are not ranked or filtered here. Everything the statistics support is emitted, and choosing which
ones are worth saying is the narrator's job.

A fact never states more than its statistic supports. On maps whose legend is a set of bands, no
detector reports a mean as though it were measured: the underlying numbers were never in the image.
"""

import numpy as np

from . import labels as labelling
from . import stats

ALPHA = 0.05


def _fact(fact_type, regions, provenance, p_value=None, **fields):
    fact = {"type": fact_type, "regions": sorted(regions),
            "p_value": None if p_value is None else round(float(p_value), 4)}
    fact.update(fields)
    fact["provenance"] = provenance
    return fact


def _components(codes, scope):
    """Split a set of regions into contiguous groups; two clusters far apart are two facts."""
    remaining, groups = set(codes), []
    while remaining:
        start = remaining.pop()
        group, queue = {start}, [start]
        while queue:
            for neighbour in scope.adjacency[queue.pop()]:
                if neighbour in remaining:
                    remaining.discard(neighbour)
                    group.add(neighbour)
                    queue.append(neighbour)
        groups.append(sorted(group))
    return sorted(groups, key=len, reverse=True)


def _band_summary(map_data, codes):
    """Describe a group of regions on a banded map by its bands, never by an invented mean."""
    bands = [map_data.band_labels.get(c) for c in codes if map_data.band_labels.get(c)]
    if not bands:
        return {}
    ranking = {label: i for i, label in enumerate(map_data.ranges)}
    top = max(bands, key=lambda b: ranking.get(b, -1))
    return {"band_labels": sorted(set(bands), key=lambda b: ranking.get(b, -1)),
            "highest_band": top, "highest_band_index": ranking.get(top)}


def missing_data(map_data, **_):
    absent = map_data.missing
    if not absent:
        return []
    return [_fact("missing_data", absent,
                  {"detector": "missing_data", "statistic": "count of null values"},
                  note="no value shown on the map for these regions",
                  share_missing=round(len(absent) / len(map_data.scope.codes), 4))]


def extremum(map_data, **_):
    """Highest and lowest region, with two honest caveats.

    On a banded map every region in the top band is equally high, so naming one would invent a
    precision the image does not hold. And where the derenderer reported which region looked darkest,
    that is a second, independent reading: if it disagrees with the highest of the 50 values, the two
    are close enough that the map cannot settle it.
    """
    observed = map_data.observed
    if len(observed) < 2:
        return []
    values = {c: map_data.values[c] for c in observed}
    facts = []
    for kind, code in (("max", max(values, key=values.get)), ("min", min(values, key=values.get))):
        fields = {"kind": kind}
        if map_data.is_banded:
            tied = sorted(c for c in observed if map_data.reported[c] == map_data.reported[code])
            fields.update(band_label=map_data.band_labels.get(code),
                          band_index=map_data.reported.get(code),
                          tied_regions=tied, unique=len(tied) == 1)
        else:
            fields.update(value=round(float(values[code]), 4), unique=True)

        claimed = map_data.reported_extremes.get(kind)
        if claimed is not None:
            fields["reported_region"] = claimed
            fields["agrees_with_values"] = claimed == code
            if claimed != code:
                fields["contested_with"] = claimed
                fields["note"] = ("the map was read two ways and they disagree; these regions are too "
                                  "close to separate")
        facts.append(_fact("extremum", [code],
                           {"detector": "extremum", "statistic": f"{kind} of reported values",
                            "values_used": {code: map_data.reported.get(code)}},
                           **fields))
    return facts


def mainland_vs_offshore(map_data, seed=0, permutations=stats.DEFAULT_PERMUTATIONS, **_):
    """Offshore regions have no land neighbour, so contiguity statistics cannot include them.

    They are still worth one comparison: an island that differs sharply from the mainland is a real
    finding, and this is the only place it can surface.
    """
    scope = map_data.scope
    offshore = [c for c in scope.offshore if map_data.values[c] is not None]
    mainland = [c for c in scope.mainland if map_data.values[c] is not None]
    if not offshore or len(mainland) < 5 or map_data.value_kind == "band_index":
        return []
    island = np.array([map_data.values[c] for c in offshore])
    main = np.array([map_data.values[c] for c in mainland])
    spread = main.std()
    if spread == 0:
        return []
    gap = (island.mean() - main.mean()) / spread

    # With only one or two offshore regions a large gap is common by chance, so significance comes
    # from asking how often any same-sized random group of regions differs this much.
    everything = np.array([map_data.values[c] for c in map_data.observed])
    rng = np.random.default_rng(seed)
    draws = np.array([rng.choice(everything, size=len(offshore), replace=False).mean()
                      for _ in range(permutations)])
    reference = everything.mean()
    extreme = int((np.abs(draws - reference) >= abs(island.mean() - reference)).sum())
    p_value = (1 + extreme) / (permutations + 1)
    if p_value > ALPHA:
        return []
    return [_fact("mainland_vs_offshore", offshore,
                  {"detector": "mainland_vs_offshore",
                   "statistic": "offshore mean vs mainland, permutation test over same-sized groups",
                   "params": {"permutations": permutations},
                   "regions_used": {"offshore": offshore, "mainland_count": len(mainland)}},
                  p_value=p_value,
                  offshore_mean=round(float(island.mean()), 4),
                  mainland_mean=round(float(main.mean()), 4),
                  standardised_gap=round(float(gap), 3),
                  direction="higher" if gap > 0 else "lower")]


def global_pattern(map_data, prepared, **_):
    result = prepared["moran"]
    if result["p_value"] <= ALPHA:
        return []
    return [_fact("no_pattern", [],
                  {"detector": "global_pattern", "statistic": "global Moran's I",
                   "params": {"permutations": result["permutations"], "alpha": ALPHA}},
                  p_value=result["p_value"], morans_i=round(result["statistic"], 4),
                  note="values are not spatially clustered beyond chance")]


def clusters(map_data, prepared, **_):
    local, w, scope = prepared["local"], prepared["weights"], map_data.scope
    intensity = prepared["gi"]["z_score"]
    facts = []
    for quadrant, kind in (("high_high", "hotspot_cluster"), ("low_low", "coldspot_cluster")):
        members = [w.codes[i] for i in range(len(w))
                   if local["significant"][i] and local["quadrant"][i] == quadrant]
        for group in _components(members, scope):
            indices = [w.index[c] for c in group]
            worst_p = float(max(local["p_adjusted"][i] for i in indices))
            label = labelling.label_cluster(group, scope)
            fields = {"region_label": label["label"] if label else None,
                      "label_level": label["level"] if label else None,
                      "label_coverage": label["coverage"] if label else None,
                      "label_precision": label["precision"] if label else None,
                      "label_recall": label["recall"] if label else None,
                      "phrasing": labelling.phrase(label),
                      "mean_gi_z": round(float(np.mean([abs(intensity[i]) for i in indices])), 3)}
            if map_data.is_banded:
                fields.update(_band_summary(map_data, group))
                if map_data.value_kind == "band_midpoint":
                    fields["approx_cluster_mean"] = round(
                        float(np.mean([map_data.values[c] for c in group])), 4)
            else:
                values = [map_data.values[c] for c in group]
                national = float(np.mean([map_data.values[c] for c in map_data.observed]))
                fields["cluster_mean"] = round(float(np.mean(values)), 4)
                if national:
                    fields["vs_national_mean_pct"] = round(
                        100 * (float(np.mean(values)) - national) / abs(national), 2)
            facts.append(_fact(kind, group,
                               {"detector": "clusters",
                                "statistic": "local Moran's I, FDR corrected",
                                "params": {"alpha": ALPHA, "permutations": prepared["permutations"]},
                                "values_used": {c: map_data.reported.get(c) for c in group}},
                               p_value=worst_p, **fields))
    return facts


def outliers(map_data, prepared, **_):
    local, w = prepared["local"], prepared["weights"]
    notes = {"high_low": "a high value surrounded by lower neighbours",
             "low_high": "a low value surrounded by higher neighbours"}
    facts = []
    for i in range(len(w)):
        if not local["significant"][i] or local["quadrant"][i] not in notes:
            continue
        code = w.codes[i]
        facts.append(_fact("spatial_outlier", [code],
                           {"detector": "outliers", "statistic": "local Moran's I, FDR corrected",
                            "params": {"alpha": ALPHA},
                            "values_used": {code: map_data.reported.get(code)}},
                           p_value=float(local["p_adjusted"][i]), note=notes[local["quadrant"][i]],
                           pattern=local["quadrant"][i]))
    return facts


def directional_trend(map_data, prepared, **_):
    result = prepared["trend"]
    if result is None:
        return []
    # Two axes are tested per map, so testing each at alpha would flag ~10% of maps by chance.
    corrected = stats.benjamini_hochberg([result["p_east_west"], result["p_south_north"]])
    axes = (("east_west", result["east_west"], corrected[0], "west to east"),
            ("south_north", result["south_north"], corrected[1], "south to north"))
    facts = []
    for axis, coefficient, p, wording in axes:
        if p > ALPHA:
            continue
        # OLS standard errors assume independent residuals. When the residuals are themselves
        # clustered the p-value is optimistic, so the fact says so rather than hiding it.
        reliable = bool(result["residual_moran_p"] > ALPHA)
        facts.append(_fact("directional_trend", [],
                           {"detector": "directional_trend",
                            "statistic": "OLS on equal-area centroids, permutation p-value",
                            "params": {"residual_moran_p": round(result["residual_moran_p"], 4)}},
                           p_value=p, axis=axis,
                           direction="increasing" if coefficient > 0 else "decreasing",
                           wording=f"values {'rise' if coefficient > 0 else 'fall'} from {wording}",
                           r_squared=round(result["r_squared"], 4),
                           p_value_reliable=reliable,
                           caveat=None if reliable else ("residuals are spatially clustered, which can "
                                                         "happen when one cluster drives the gradient; "
                                                         "treat the p-value as optimistic")))
    return facts


ALL = [missing_data, extremum, mainland_vs_offshore, global_pattern, clusters, outliers,
       directional_trend]

NEEDS_STATISTICS = {global_pattern, clusters, outliers, directional_trend}
