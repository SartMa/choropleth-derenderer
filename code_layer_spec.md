# Code Layer — Implementation Specification

## 0. Role of this component in the overall pipeline

This is the middle stage of a three-stage system:

```
Map image(s) -> [Derenderer VLM] -> structured region data -> [CODE LAYER] -> candidate facts -> [Narrator LLM] -> summary text
```

The code layer is **fully deterministic** — no machine learning, no LLM calls, no
generation. Its entire job is: take the derenderer's structured output, join it
against known real-world geographic reference data, run established spatial
statistics on it, and emit a list of typed, numerically-grounded "candidate
facts." It must never invent, estimate, or guess anything — every number and
every claim it outputs must be directly computable from its inputs plus its
static reference database.

The reason this layer exists at all: it is what prevents mathematical and
spatial hallucination in the final summary. The narrator LLM downstream is
never allowed to do arithmetic or claim spatial relationships itself — it only
selects and phrases facts that this layer already computed and verified.

---

## 1. Assumptions and dependencies

### 1.1 Geometry source of truth
- All region geometry, region names, canonical region IDs, centroids, and
  adjacency relationships come from **one single source**: geoBoundaries
  (ADM0/ADM1/ADM2 as appropriate), downloaded once and stored locally.
- Do not mix geometry sources (e.g. do not use Natural Earth for one part and
  geoBoundaries for another) — this is a real risk of subtle ID/boundary
  mismatches.
- This same reference database must be the one used to generate synthetic
  training data for the derenderer, so IDs and geometry are guaranteed
  consistent between training and inference.

### 1.2 Fixed canonical region order per "map scope"
- For each supported geography (e.g. `US_states`, `India_states`), there is a
  **fixed, agreed-upon ordered list** of canonical region names (e.g.
  alphabetical). Both the derenderer and this code layer must reference the
  identical, versioned list.
- The derenderer's output is assumed to already be in this canonical order,
  or explicitly tagged per entry with a name from this fixed vocabulary — see
  Section 2.
- This code layer is responsible for maintaining/storing the canonical order
  list per scope (`reference_data/{scope}/region_order.json`), not for
  producing it at runtime.

### 1.3 Precomputed static reference tables (build once, offline, cache)
For every supported `map_scope`, precompute and store:
- `region_order.json` — fixed ordered list of canonical region names/IDs.
- `geometry.gpkg` (or GeoJSON) — polygon geometry per region ID, from
  geoBoundaries.
- `centroids.json` — {region_id: [lat, lon]} for every region.
- `adjacency.json` — {region_id: [neighbor_region_id, ...]}, built once via
  GeoPandas `touches`/`intersects` on the geometry table.
- `named_macro_regions.json` — official administrative groupings where they
  exist (e.g. US Census Bureau's Northeast/Midwest/South/West and finer
  divisions), as {region_id: [macro_region_label, ...]}. Where no official
  convention exists for a scope, this file may be absent — see Section 4.6.
- `value_semantics_keywords.json` — a small curated keyword list mapping
  common variable-name substrings to `higher_is_worse` / `higher_is_better`
  (e.g. "mortality", "poverty", "unemployment" -> worse; "literacy",
  "income", "vaccination" -> better). Anything not matched defaults to
  `neutral`.

### 1.4 Known edge cases to special-case (not solve generally)
- **Non-contiguous entities** (e.g. Alaska/Hawaii inset boxes on standard US
  maps) — adjacency for these should be pulled from a small manual override
  table (`adjacency_overrides.json`) rather than derived purely from
  geometric contiguity, since inset boxes are not geographically adjacent to
  anything in true coordinate space.
- **Disputed/contested boundaries** — regions in this category should be
  excluded from the initially supported scope list rather than silently
  resolved one way, unless a deliberate policy decision is made.
- **Different geographic granularity across two maps being compared** (e.g.
  state-level vs. county-level) — code layer must detect this and refuse
  bivariate comparison (Section 5) until an aggregation/spatial-join step
  aligns them to a common node set. This alignment is out of scope for v1;
  raise a clear error instead of silently proceeding.

---

## 2. Input contract (from the Derenderer)

The code layer receives one JSON object per map image:

```json
{
  "map_scope": "US_states",
  "title_text": "COVID-19 Mortality Rate, 2021",
  "unit_text": "per 100,000",
  "regions": [
    {"name": "Alabama", "value": 34.2},
    {"name": "Alaska", "value": null},
    ...
  ]
}
```

Contract requirements the code layer should validate on receipt (fail loudly,
do not silently proceed, if violated):
- `map_scope` must exactly match a supported, precomputed reference scope
  (Section 1.3). If not recognized, reject the input.
- `regions` must contain exactly one entry per region in that scope's
  `region_order.json` — no missing, no extra, no duplicates. Every `name`
  must be a member of the canonical vocabulary for that scope (this should
  already be guaranteed if the derenderer used constrained decoding, but the
  code layer must not assume this without checking).
- `value` is either a float or `null`. `null` means "no data for this
  region" and must be handled explicitly (Section 4.3), not coerced to 0.
- For time-series / multi-map input, the code layer receives an **array** of
  the above objects, one per time step, all sharing the same `map_scope`,
  ordered chronologically, plus a `time_labels` array (e.g. `["2019",
  "2020", "2021"]`) of the same length.
- For two-variable / bivariate input, the code layer receives **two** such
  objects for the same `map_scope` (same region set), each with its own
  `title_text`/`unit_text`.

---

## 3. Processing pipeline — single map

Run in this order. Each step's output feeds later steps; steps 3.5 onward
only run if earlier steps succeed.

### 3.1 Metadata parsing
- Parse `title_text` / `unit_text` into structured fields:
  - `variable_name` (cleaned title string)
  - `unit_type` — classify into one of: `rate_per_N`, `percentage`, `count`,
    `index`, `unknown`, via simple pattern matching on `unit_text` (e.g.
    "per 100,000" -> `rate_per_N` with N=100000; "%" -> `percentage`).
  - `value_semantics` — look up `variable_name` against
    `value_semantics_keywords.json`; default `neutral` if no match.

### 3.2 Canonicalization
- Join each `regions[i].name` against the scope's reference tables to
  attach: canonical `region_id`, real `geometry`, real `centroid`, and real
  `adjacency` list. (Do not trust any geometry/position from the derenderer
  itself, if present — geometry always comes from the static reference
  data, keyed only by the validated name.)

### 3.3 Missing-data audit
- Compute `data_completeness` = (# non-null values) / (total regions).
- If `data_completeness < 1.0`, emit a `missing_data` fact (Section 6)
  listing the affected regions.
- Exclude null-value regions from all statistical computations below, but
  keep them in the adjacency graph (a missing region can still be a
  "neighbor" of a computed region).

### 3.4 Base statistics
Compute and retain internally (used to populate fact fields, not all
necessarily surfaced as their own facts): mean, median, std dev, quartiles,
min region, max region.
- Emit one `extremum` fact for the max and one for the min (Section 6).

### 3.5 Global spatial autocorrelation
- Build spatial weights from the adjacency graph (first-order contiguity;
  use `libpysal.weights` from the adjacency list, row-standardized).
- Compute **Global Moran's I** (`esda.Moran`) with permutation-based
  significance (default 999 permutations), alpha = 0.05.
- If **not** significant: emit a single `no_pattern` fact and skip 3.6–3.8
  (there is nothing further to report spatially).
- If significant: proceed.

### 3.6 Local clustering
- Compute **Local Moran's I / LISA** (`esda.Moran_Local`) per region, with
  significance flags. Apply a multiple-testing correction (FDR-BH) across
  all regions before thresholding at alpha = 0.05.
- Classify each significant region as High-High (hotspot core), Low-Low
  (coldspot core), or High-Low/Low-High (spatial outlier).
- Also compute **Getis-Ord Gi\*** (`esda.G_Local`) per region as a secondary
  intensity measure (same FDR correction), used to rank/order clusters by
  strength when multiple clusters are found.
- Group contiguous same-class significant regions via connected components
  on the adjacency graph (restricted to the subgraph of significant
  same-class regions) -> one `hotspot_cluster` or `coldspot_cluster` fact
  per resulting connected component (Section 6). Emit standalone
  High-Low/Low-High regions as `spatial_outlier` facts.

### 3.7 Directional trend (trend surface analysis)
- Regress `value ~ latitude + longitude` (region centroids) via OLS.
- If either coefficient is statistically significant (p < 0.05), emit a
  `directional_trend` fact describing the axis (north-south and/or
  east-west) and direction of the significant coefficient(s).

### 3.8 Named-region matching
- For each cluster fact from 3.6, compute the fraction of its member
  regions that belong to each label in `named_macro_regions.json` (if
  present for this scope).
- If any label's overlap >= 0.70, attach that `region_label` to the cluster
  fact. If no `named_macro_regions.json` exists for this scope, or no label
  clears the threshold, fall back to a coarse centroid-based directional
  bucket (e.g. quartile bands of latitude/longitude) instead, or omit the
  label field entirely if even that isn't meaningful, rather than forcing a
  guess.

### 3.9 Group comparison
- Using whichever named/coarse groupings were produced in 3.8 across the
  whole map (not just cluster members), rank groups by mean value.
- Emit a single `regional_comparison` fact for the largest gap (top vs.
  bottom group) only — do not emit exhaustive pairwise comparisons.

---

## 4. Processing pipeline — multi-map / time series

Applies when the input is a time-ordered array of single-map objects
(Section 2) sharing one `map_scope`.

### 4.1 Build the space-time structure
- For each time step, run Section 3 steps 3.1–3.4 independently.
- Construct a "bin" per (region, time_step) pair. A bin's neighbors for
  spatiotemporal statistics = spatially-adjacent regions (from the static
  adjacency graph) at the same time step, plus the same region at
  neighboring time steps within a configurable time window (default: ±1
  step).

### 4.2 Per-time-step local clustering
- Run Getis-Ord Gi* (Section 3.6) independently at each time step, using
  the spatiotemporal neighborhood from 4.1, with FDR correction applied
  per time step.
- Result: for every region, a sequence of (time_step, significant_hot /
  significant_cold / not_significant) labels across the whole series.

### 4.3 Trend test per region
- For each region, run the **Mann-Kendall trend test** (e.g.
  `pymannkendall`) on its sequence of Gi* z-scores across time steps.
- Output: `increasing` / `decreasing` / `no_trend`, each with a p-value.

### 4.4 Category classification (rule-based, not a statistical test itself)
For each region, using its full hot/cold/not-significant timeline (4.2) and
its Mann-Kendall result (4.3), assign exactly one of the following labels
via explicit rule logic (implement as a deterministic decision function, not
a model):

- `new_hotspot` — significant hotspot only in the final time step, never
  before.
- `consecutive_hotspot` — one uninterrupted recent run of significant
  hotspot steps ending at the final step, no earlier significant history,
  <90% of steps significant overall.
- `intensifying_hotspot` — significant hotspot in >=90% of steps
  (including the final step) AND Mann-Kendall trend is significantly
  increasing.
- `persistent_hotspot` — significant hotspot in >=90% of steps, no
  significant Mann-Kendall trend.
- `diminishing_hotspot` — significant hotspot in most steps, Mann-Kendall
  trend is significantly decreasing.
- `sporadic_hotspot` — on/off significance pattern, <90% of steps
  significant, no consistent run.
- `oscillating_hotspot` — significant hotspot currently, with a history of
  also being a significant coldspot at some earlier step.
- `historical_hotspot` — significant hotspot for most of the series, but
  NOT the most recent step.
- Mirror all of the above for `*_coldspot`.
- `no_pattern_detected` — none of the above.

Implement thresholds (90%, window size, alpha) as named constants at the
top of the module, not magic numbers inline, so they can be tuned/cited.

### 4.5 Emit temporal facts
- One `temporal_pattern` fact per region whose category is not
  `no_pattern_detected`; group contiguous regions sharing the same category
  into a single fact the same way 3.6 groups spatial clusters.

---

## 5. Processing pipeline — bivariate (two variables, same time, same scope)

Applies when input is exactly two single-map objects (Section 2) for the
same `map_scope` and same region set.

- Reject (raise a clear validation error) if the two inputs' region sets or
  `map_scope` don't match exactly — do not attempt to auto-align different
  granularities (see Section 1.4).
- Compute plain Pearson and Spearman correlation between the two value
  vectors across all regions with non-null data in both. Emit a
  `global_correlation` fact with both coefficients and p-values.
- Compute **Bivariate Moran's I** (`esda.Moran_BV`) to test spatial
  co-association (does variable A's value in a region associate with
  variable B's value in that region's *neighbors*). Emit a
  `spatial_correlation` fact if significant.
- Optionally compute local bivariate clustering (regions where both
  variables are jointly elevated, or where they diverge sharply) and emit
  as `bivariate_cluster` / `bivariate_outlier` facts, grouped by contiguity
  as in 3.6.

---

## 6. Output contract

Single JSON object per invocation:

```json
{
  "metadata": {
    "variable_name": "COVID-19 mortality rate",
    "unit_type": "rate_per_N",
    "unit_n": 100000,
    "value_semantics": "higher_is_worse",
    "geographic_scope": "US_states",
    "national_mean": 28.4,
    "national_median": 26.1,
    "data_completeness": 0.94
  },
  "facts": [
    {
      "fact_id": "f1",
      "type": "hotspot_cluster",
      "regions": ["US-AL", "US-MS", "US-LA", "US-GA"],
      "region_label": "South",
      "label_overlap": 0.82,
      "cluster_mean": 41.2,
      "vs_national_mean_pct": 45.1,
      "p_value": 0.01
    }
  ]
}
```

### 6.1 Fact schema — shared base fields (every fact has these)
- `fact_id` (string, unique within this output)
- `type` (string enum — see 6.2)
- `regions` (array of canonical region_ids involved)
- `p_value` (float, where statistically applicable; omit/null for facts
  with no associated significance test, e.g. `extremum`)

### 6.2 Fact types and their type-specific fields
- `extremum` — `kind` ("max"/"min"), `value`
- `hotspot_cluster` / `coldspot_cluster` — `region_label` (optional),
  `label_overlap` (optional), `cluster_mean`, `vs_national_mean_pct`
- `spatial_outlier` — `note` (fixed short description, e.g. "high value
  surrounded by lower-value neighbors")
- `directional_trend` — `axis` ("north_south" / "east_west"), `direction`
  ("increasing"/"decreasing")
- `regional_comparison` — `group_a`, `group_b` (labels), `group_a_mean`,
  `group_b_mean`
- `temporal_pattern` — `category` (one of the labels in Section 4.4),
  `time_range`
- `global_correlation` / `spatial_correlation` (bivariate only) —
  `variable_a`, `variable_b`, `coefficient`, `method`
- `bivariate_cluster` / `bivariate_outlier` (bivariate only) — `pattern`
  (e.g. "both_high", "both_low", "high_a_low_b")
- `no_pattern` — no extra fields beyond base
- `missing_data` — `note`, `regions` = affected region list

This is a closed, fixed set for v1. Adding a new fact type later requires
updating this spec and the narrator's training data — do not let an
implementing agent invent ad hoc new types silently.

### 6.3 Traceability requirement
Every fact must be traceable back to the exact computation that produced
it (e.g. log or retain, per fact, which function/statistic and input
region values generated it) — this is required for the future claims-
verifier component (out of scope for this spec, but do not make decisions
now that would block it later, e.g. do not discard intermediate
computation results once a fact is emitted).

---

## 7. Suggested libraries

- `geopandas` — geometry loading, spatial joins
- `libpysal` — spatial weights construction from adjacency
- `esda` — Moran's I (global/local/bivariate), Getis-Ord Gi*
- `pymannkendall` — Mann-Kendall trend test
- `statsmodels` or `scipy.stats` — OLS trend-surface regression, Pearson/
  Spearman correlation
- `networkx` (optional) — connected-components grouping of clusters if not
  done directly via GeoPandas/PySAL utilities

---

## 8. Testing / validation guidance for the implementing agent

- Unit test each statistical step (3.5–3.9, 4.2–4.4, 5) against small
  hand-constructed synthetic inputs with known, obvious answers (e.g. a
  map where one contiguous corner is deliberately set far above the mean
  and everything else is noise — this should reliably produce exactly one
  `hotspot_cluster` fact).
- Test the `no_pattern` path explicitly with a fully randomized (spatially
  uncorrelated) synthetic input — Global Moran's I should come back
  non-significant, and no cluster facts should be emitted.
- Test missing-data handling with a synthetic input containing several
  `null` values, confirming excluded regions don't silently become zeros
  in any statistic.
- Test the Section 1.4 edge cases explicitly (a scope including Alaska/
  Hawaii-style insets; a rejected bivariate call with mismatched scopes).
- Validate that output JSON always conforms exactly to Section 6's schema
  (use a JSON schema validator in CI, not just manual inspection).
