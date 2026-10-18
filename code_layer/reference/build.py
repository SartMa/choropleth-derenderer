#!/usr/bin/env python3
"""Build the code layer's static reference data for a geography, once, offline.

    python code_layer/reference/build.py us_states

Writes adjacency.json, centroids.json and macro_regions.json next to this file. Geometry comes from
geoBoundaries via synth/generate_maps.py, which is also what draws the synthetic training maps, so
region codes cannot drift between the derenderer's training data and this layer's reference tables.

Only the build step needs geopandas; the code layer itself reads the JSON.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent.parent

# US Census Bureau regions and divisions. Divisions are tested first when labelling a cluster: a
# six-state cluster covering New England is better described as New England than as the Northeast.
US_CENSUS = {
    "Northeast": {"New England": "CT ME MA NH RI VT", "Middle Atlantic": "NJ NY PA"},
    "Midwest": {"East North Central": "IL IN MI OH WI", "West North Central": "IA KS MN MO NE ND SD"},
    "South": {"South Atlantic": "DE FL GA MD NC SC VA WV", "East South Central": "AL KY MS TN",
              "West South Central": "AR LA OK TX"},
    "West": {"Mountain": "AZ CO ID MT NV NM UT WY", "Pacific": "AK CA HI OR WA"},
}


def build_us_states(out_dir):
    sys.path.insert(0, str(PROJECT / "synth"))
    import generate_maps as gen

    cache = PROJECT / "data" / "boundaries"
    gdf = gen.load_states(cache)
    gen.load_or_build_adjacency(gdf, cache)
    adjacency = json.loads((cache / "usa_states_adjacency.json").read_text())

    projected = gdf.to_crs(gen.CONUS_CRS)
    planar_centroids = projected.geometry.centroid
    geographic = planar_centroids.to_crs("EPSG:4326")  # reproject the centroid, don't average degrees
    centroids = {}
    for code, planar, degrees in zip(gdf["code"], planar_centroids, geographic):
        # x/y are metres in an equal-area projection: the trend surface must regress on those, not on
        # degrees, whose east-west spacing shrinks towards the poles.
        centroids[code] = {"x": round(planar.x, 1), "y": round(planar.y, 1),
                           "lon": round(degrees.x, 5), "lat": round(degrees.y, 5)}

    macro = {}
    for region, divisions in US_CENSUS.items():
        for division, codes in divisions.items():
            for code in codes.split():
                macro[code] = {"region": region, "division": division}

    codes = sorted(adjacency)
    missing = [c for c in codes if c not in macro or c not in centroids]
    if missing:
        raise SystemExit(f"reference data incomplete for: {missing}")
    for code, neighbours in adjacency.items():
        for other in neighbours:
            if code not in adjacency[other]:
                raise SystemExit(f"adjacency is not symmetric: {code}-{other}")

    out_dir.mkdir(parents=True, exist_ok=True)
    write(out_dir / "adjacency.json", {c: sorted(adjacency[c]) for c in codes})
    write(out_dir / "centroids.json", {c: centroids[c] for c in codes})
    write(out_dir / "macro_regions.json", {c: macro[c] for c in codes})

    isolates = [c for c in codes if not adjacency[c]]
    borders = sum(len(v) for v in adjacency.values()) // 2
    print(f"{len(codes)} regions, {borders} shared borders, isolates: {isolates or 'none'}")


def write(path, obj):
    path.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n")
    print(f"wrote {path.relative_to(PROJECT)}")


SCOPES = {"us_states": build_us_states}

if __name__ == "__main__":
    scope = sys.argv[1] if len(sys.argv) > 1 else "us_states"
    if scope not in SCOPES:
        raise SystemExit(f"unknown scope {scope}; known: {sorted(SCOPES)}")
    SCOPES[scope](HERE / scope)
