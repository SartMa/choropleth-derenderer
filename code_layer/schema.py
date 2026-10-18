"""The closed set of fact types, and a validator for the layer's output.

Stage 3 is trained against this vocabulary, so a detector must not quietly add a type or a field: a
fact the narrator has never seen is a fact it cannot phrase safely. Adding one is a deliberate change
here first.
"""

BASE_REQUIRED = {"fact_id", "type", "regions", "p_value", "provenance"}

# type -> (required fields, optional fields)
FACT_TYPES = {
    "extremum": (
        {"kind", "unique"},
        {"value", "band_label", "band_index", "tied_regions",
         "reported_region", "agrees_with_values", "contested_with", "note"}),
    "hotspot_cluster": (
        {"region_label", "label_level", "label_coverage", "label_precision", "label_recall",
         "phrasing", "mean_gi_z"},
        {"cluster_mean", "vs_national_mean_pct", "band_labels", "highest_band",
         "highest_band_index", "approx_cluster_mean"}),
    "spatial_outlier": ({"note", "pattern"}, set()),
    "directional_trend": (
        {"axis", "direction", "wording", "r_squared", "p_value_reliable"}, {"caveat"}),
    "mainland_vs_offshore": (
        {"offshore_mean", "mainland_mean", "standardised_gap", "direction"}, set()),
    "no_pattern": ({"morans_i", "note"}, set()),
    "missing_data": ({"note", "share_missing"}, set()),
}
FACT_TYPES["coldspot_cluster"] = FACT_TYPES["hotspot_cluster"]

METADATA_REQUIRED = {
    "code_layer_version", "map_scope", "title", "legend_type", "data_type", "value_kind",
    "values_are_banded", "regions_total", "regions_with_data", "regions_analysed",
    "data_completeness", "permutations", "seed",
}


class SchemaError(ValueError):
    """The layer produced something outside its own contract; that is a bug, not bad input."""


def validate(result):
    """Check one analyse() result. Returns it unchanged, or raises SchemaError."""
    if not isinstance(result, dict) or set(result) != {"metadata", "facts"}:
        raise SchemaError(f"top level must be exactly metadata and facts, got {sorted(result)}")

    absent = METADATA_REQUIRED - set(result["metadata"])
    if absent:
        raise SchemaError(f"metadata is missing {sorted(absent)}")

    seen_ids = set()
    for fact in result["facts"]:
        kind = fact.get("type")
        if kind not in FACT_TYPES:
            raise SchemaError(f"unknown fact type {kind!r}; the set is closed, see schema.FACT_TYPES")
        missing_base = BASE_REQUIRED - set(fact)
        if missing_base:
            raise SchemaError(f"{kind} is missing base fields {sorted(missing_base)}")
        if fact["fact_id"] in seen_ids:
            raise SchemaError(f"duplicate fact_id {fact['fact_id']}")
        seen_ids.add(fact["fact_id"])
        if not isinstance(fact["regions"], list):
            raise SchemaError(f"{kind}: regions must be a list")

        required, optional = FACT_TYPES[kind]
        specific = set(fact) - BASE_REQUIRED
        missing = required - specific
        if missing:
            raise SchemaError(f"{kind} is missing {sorted(missing)}")
        unknown = specific - required - optional
        if unknown:
            raise SchemaError(f"{kind} has fields outside the schema: {sorted(unknown)}")

        provenance = fact["provenance"]
        if not isinstance(provenance, dict) or "detector" not in provenance or "statistic" not in provenance:
            raise SchemaError(f"{kind}: provenance needs at least detector and statistic")
    return result
