#!/usr/bin/env python3
"""Generate synthetic US-state choropleth maps with the JSON the derenderer should output.

The target format is the same as mapqa/convert_mapqa.py, so MapQA and synthetic maps can be trained on together:

    {"title":"Unemployment Rate, 2023","legend_title":"%","legend_type":"discrete","data_type":"percentage",
     "ranges":["2–3","3–4","4–5","5+"],"values":{"AK":2,"AL":0,...,"WY":null}}

    {"title":"Average Gas Price (2024)","legend_title":"USD per gallon","legend_type":"continuous",
     "data_type":"number","values":{"AK":4.12,"AL":3.05,...,"WY":3.4}}

Rules (the same as for MapQA):
- values: all 50 states by 2-letter code, sorted; DC is drawn as part of Maryland and not listed
- continuous: numbers rounded so the color bar has about 100 steps
- discrete: the index of the state's range, 0 = lowest; "ranges" lists the legend labels exactly as drawn,
  lowest first, whatever order the legend shows them in
- no data: null (the legend shows a "No data" box)
- legend_title: the text above the legend exactly as drawn (the unit), or null when the map has none;
  percentage maps always show "%" there, so data_type is always visible
- data_type: "percentage" when the unit is "%", otherwise "number"

Usage (run from the project root; each split has its own seed, so splits never share a map):
    python synth/generate_maps.py --split train --n 10000
    python synth/generate_maps.py --split dev --n 1000
    python synth/generate_maps.py --split test --n 1000

Output (default folder data/synth_usa):
    images/<id>.png or .jpg
    <split>.jsonl            one line per map: id, image, target, legend_type, data_type, true_values, style
    <split>.rejected.jsonl   maps that failed the pixel or text check; not used for training
Also writes <cache>/usa_states_adjacency.json: which states share a border, by state code.
"""

import argparse
import json
import math
import os
import urllib.request
from multiprocessing import Pool
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shapely
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, to_rgb, to_rgba
from matplotlib.patches import Patch
from PIL import Image
from shapely import affinity

API_URL = "https://www.geoboundaries.org/api/current/gbOpen/USA/ADM1/"
TERRITORIES = {
    "American Samoa",
    "Commonwealth of the Northern Mariana Islands",
    "Guam",
    "Puerto Rico",
    "United States Virgin Islands",
}
ISO_FIXES = {"South Dakota": "US-SD"}  # geoBoundaries lists it as "SU-SD"
SMALL_STATES = {"CT", "DE", "HI", "MA", "MD", "NH", "NJ", "RI", "VT"}  # a text label would hide their color
SPLIT_SEEDS = {"train": 0, "dev": 1, "test": 2}

CONUS_CRS = "EPSG:5070"  # Albers equal-area for the lower 48
ALASKA_CRS = "EPSG:3338"
HAWAII_CRS = "+proj=aea +lat_1=8 +lat_2=18 +lat_0=13 +lon_0=-157 +datum=WGS84 +units=m +no_defs"
DPI = 100

# (topic, unit, typical low, typical high) for state-level values
TOPICS = [
    ("Median Household Income", "thousand USD", 45, 100),
    ("Unemployment Rate", "%", 2, 8),
    ("Adult Obesity Rate", "%", 22, 40),
    ("Poverty Rate", "%", 7, 20),
    ("Uninsured Rate", "%", 2, 20),
    ("Median Age", "years", 31, 45),
    ("Life Expectancy", "years", 72, 82),
    ("Households with Broadband", "%", 75, 95),
    ("Voter Turnout", "%", 45, 80),
    ("Average Gas Price", "USD per gallon", 2.8, 5.2),
    ("Violent Crime Rate", "per 100,000 people", 100, 800),
    ("Population Density", "people per sq. mile", 5, 500),
    ("Renewable Electricity Share", "%", 2, 60),
    ("Median Home Value", "thousand USD", 130, 750),
    ("Average January Temperature", "°F", 5, 70),
    ("High School Graduation Rate", "%", 75, 95),
    ("Adult Smoking Rate", "%", 7, 22),
    ("Median Rent", "USD per month", 700, 1900),
]
TITLE_TEMPLATES = [
    "{topic}, {year}",
    "{topic} by State, {year}",
    "{topic} by State ({year})",
    "{year} {topic}",
    "{topic} in the United States, {year}",
    "{topic} ({year})",
]
SEQUENTIAL = ["Blues", "Greens", "Oranges", "Purples", "Reds", "YlOrRd", "YlGnBu", "YlGn", "OrRd",
              "PuBu", "BuPu", "GnBu", "RdPu", "viridis", "plasma", "magma", "inferno"]
DIVERGING = ["RdBu", "RdYlBu", "PiYG", "BrBG", "Spectral", "PuOr"]
NO_DATA_COLORS = ["#a6a6a6", "#cfcfcf", "#808080"]
BACKGROUNDS = ["white", "white", "#f7f7f7", "#fbf8f0"]
RANGE_SEPARATORS = ["–", "-", " – ", " to "]
# (first bin, last bin) label styles; None keeps the plain "a–b" range
OPEN_END_STYLES = [("< {}", "≥ {}"), ("Under {}", "{} or more"), ("Less than {}", "{} and above"), (None, "{}+")]


# ---------------------------------------------------------------- geography

def load_states(cache_dir):
    """The 50 states from geoBoundaries, sorted by state code, in EPSG:4326. DC is merged into Maryland."""
    path = cache_dir / "geoBoundaries-USA-ADM1_simplified.geojson"
    if not path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(API_URL) as response:
            url = json.load(response)["simplifiedGeometryGeoJSON"]
        urllib.request.urlretrieve(url, path)
    gdf = gpd.read_file(path)
    gdf = gdf[~gdf["shapeName"].isin(TERRITORIES)].copy()
    maryland = gdf.index[gdf["shapeName"] == "Maryland"][0]
    dc = gdf.index[gdf["shapeName"] == "District of Columbia"][0]
    gdf.at[maryland, "geometry"] = gdf.at[maryland, "geometry"].union(gdf.at[dc, "geometry"])
    gdf = gdf.drop(index=dc)
    gdf["code"] = [ISO_FIXES.get(name, iso).split("-")[1] for name, iso in zip(gdf["shapeName"], gdf["shapeISO"])]
    gdf = gdf.sort_values("code").reset_index(drop=True)
    if len(gdf) != 50:
        raise ValueError(f"expected 50 states, got {len(gdf)}")
    return gdf


def load_or_build_adjacency(gdf, cache_dir):
    """Neighbor index lists. Two states are neighbors if their borders touch (2 km tolerance)."""
    path = cache_dir / "usa_states_adjacency.json"
    codes = list(gdf["code"])
    if not path.exists():
        buffered = gdf.geometry.to_crs(CONUS_CRS).buffer(2000).values
        left, right = shapely.STRtree(buffered).query(buffered, predicate="intersects")
        neighbors = {code: [] for code in codes}
        for i, j in zip(left, right):
            if i != j:
                neighbors[codes[i]].append(codes[j])
        path.write_text(json.dumps({c: sorted(v) for c, v in neighbors.items()}, indent=1))
    neighbors = json.loads(path.read_text())
    return [[codes.index(m) for m in neighbors[c]] for c in codes]


def fit_into(geom, width, left, bottom):
    """Scale geom to the given width and move its lower-left corner to (left, bottom)."""
    minx, miny, maxx, _ = geom.bounds
    s = width / (maxx - minx)
    geom = affinity.scale(geom, s, s, origin=(minx, miny))
    return affinity.translate(geom, left - minx, bottom - miny)


def layout_geometries(gdf):
    """Shapes to draw: lower 48 in Albers, Alaska and Hawaii moved into lower-left insets."""
    names = list(gdf["shapeName"])
    conus = gdf.geometry[~gdf["shapeName"].isin(["Alaska", "Hawaii"])].to_crs(CONUS_CRS).simplify(1000)
    minx, miny, maxx, maxy = conus.total_bounds
    width, height = maxx - minx, maxy - miny

    def projected(name, lonlat_box, crs, tolerance):
        # clipping drops the far Aleutians and the Northwestern Hawaiian Islands, like most US maps
        geom = shapely.clip_by_rect(gdf.geometry[gdf["shapeName"] == name].iloc[0], *lonlat_box)
        return gpd.GeoSeries([geom], crs=gdf.crs).to_crs(crs).simplify(tolerance).iloc[0]

    placed = {
        "Alaska": fit_into(projected("Alaska", (-180, 50, -129, 72), ALASKA_CRS, 2000),
                           0.24 * width, minx, miny - 0.08 * height),
        "Hawaii": fit_into(projected("Hawaii", (-161, 18, -154, 23), HAWAII_CRS, 500),
                           0.11 * width, minx + 0.27 * width, miny - 0.06 * height),
    }
    return [placed[n] if n in placed else conus.loc[i] for i, n in enumerate(names)]


def prepare_geography(cache_dir):
    gdf = load_states(cache_dir)
    drawn = layout_geometries(gdf)
    label_xy = []
    for geom in drawn:
        part = max(geom.geoms, key=lambda g: g.area) if geom.geom_type == "MultiPolygon" else geom
        point = part.representative_point()
        label_xy.append((point.x, point.y))
    reps = gdf.geometry.representative_point()
    return {
        "codes": list(gdf["code"]),
        "drawn": drawn,
        "label_xy": label_xy,
        "lonlat": np.column_stack([reps.x, reps.y]),
        "adjacency": load_or_build_adjacency(gdf, cache_dir),
    }


# ---------------------------------------------------------------- values and legends

def make_values(rng, adjacency, lonlat, low, high):
    """One value per state, with a random spatial pattern."""
    n = len(adjacency)
    pattern = str(rng.choice(["random", "clustered", "gradient"]))
    if pattern == "random":
        z = rng.normal(size=n) if rng.random() < 0.5 else rng.uniform(size=n)
    elif pattern == "clustered":  # neighbors end up with similar values
        z = rng.normal(size=n)
        for _ in range(int(rng.integers(1, 5))):
            z = np.array([(z[i] + z[nb].mean()) / 2 if nb else z[i] for i, nb in enumerate(adjacency)])
    else:  # values rise in a random compass direction
        angle = rng.uniform(0, 2 * np.pi)
        xy = (lonlat - lonlat.mean(axis=0)) / lonlat.std(axis=0)
        z = xy @ np.array([np.cos(angle), np.sin(angle)]) + rng.normal(scale=rng.uniform(0.2, 0.8), size=n)
    z = (z - z.min()) / (z.max() - z.min())
    if rng.random() < 0.3:
        z = z ** rng.uniform(1.5, 3.0)  # skewed: most states low, a few high
    span = high - low
    lo = low + rng.uniform(0, 0.3) * span
    hi = high - rng.uniform(0, 0.3) * span
    return lo + z * (hi - lo), pattern


def fmt(x, decimals):
    s = f"{x:.{decimals}f}"
    return s.lstrip("-") if float(s) == 0 else s


def decimals_for(step):
    d = max(0, -math.floor(math.log10(step)))
    while d < 6 and abs(round(step, d) - step) > 1e-9:
        d += 1
    return d


def nice_step(span, k):
    """A round step (1, 2, 2.5 or 5 times a power of ten) that splits span into about k parts."""
    raw = span / k
    magnitude = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if m * magnitude >= raw - 1e-12:
            return m * magnitude


def continuous_legend(rng, values):
    lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    step = nice_step(hi - lo, int(rng.integers(3, 6)))
    d = decimals_for(step)
    vmin = round(math.floor(lo / step) * step, d)
    vmax = round(math.ceil(hi / step) * step, d)
    ticks = [round(vmin + i * step, d) for i in range(int(round((vmax - vmin) / step)) + 1)]
    return {"type": "continuous", "vmin": vmin, "vmax": vmax, "ticks": ticks, "tick_decimals": d}


def round_to_bar(values, vmin, vmax):
    """Round so the color bar has about 100 steps (the same rule as mapqa/convert_mapqa.py)."""
    digits = round(-math.log10((vmax - vmin) / 100))
    if digits > 0:
        return [None if v is None else round(float(v), digits) for v in values]
    return [None if v is None else int(round(float(v), digits)) for v in values]


def bin_edges(valid, lo, hi, k, method):
    if method == "nice":
        step = nice_step(hi - lo, k)
        d = decimals_for(step)
        first = math.floor(lo / step)
        n_bins = max(1, math.ceil(hi / step - 1e-9) - first)
        return [round((first + i) * step, d) for i in range(n_bins + 1)], d
    d = max(0, round(-math.log10((hi - lo) / 50)))
    qs = np.linspace(0, 1, k + 1)[1:-1]
    raw = np.quantile(valid, qs) if method == "quantile" else lo + qs * (hi - lo)
    inner = sorted(e for e in {round(float(x), d) for x in raw} if lo < e < hi)
    scale = 10 ** d
    return [math.floor(lo * scale) / scale, *inner, math.ceil(hi * scale) / scale], d


def bin_labels(rng, edges, d):
    sep = str(rng.choice(RANGE_SEPARATORS))
    labels = [f"{fmt(a, d)}{sep}{fmt(b, d)}" for a, b in zip(edges[:-1], edges[1:])]
    if rng.random() < 0.5:
        first, last = OPEN_END_STYLES[int(rng.integers(len(OPEN_END_STYLES)))]
        if first:
            labels[0] = first.format(fmt(edges[1], d))
        labels[-1] = last.format(fmt(edges[-2], d))
    return labels


def binned_legend(rng, values):
    valid = values[~np.isnan(values)]
    lo, hi = float(valid.min()), float(valid.max())
    method = str(rng.choice(["nice", "equal", "quantile"]))
    edges, d = bin_edges(valid, lo, hi, int(rng.integers(3, 8)), method)
    if len(edges) < 4:  # fewer than 3 bins
        method = "equal"
        edges, d = bin_edges(valid, lo, hi, 5, method)
    return {"type": "binned", "method": method, "edges": edges, "decimals": d,
            "labels": bin_labels(rng, edges, d)}


def pick_palette(rng):
    diverging = bool(rng.random() < 0.25)
    name = str(rng.choice(DIVERGING if diverging else SEQUENTIAL))
    cmap = matplotlib.colormaps[name]
    if rng.random() < 0.2:
        cmap, name = cmap.reversed(), name + "_r"
    return name, cmap, diverging


def pick_no_data_color(palette_rgbs):
    """The gray that is least similar to any palette color."""
    return max(NO_DATA_COLORS, key=lambda c: np.abs(palette_rgbs - np.array(to_rgb(c))).max(axis=1).min())


# ---------------------------------------------------------------- drawing

def interior_points(geom, inset, rng, k=24):
    """Up to k random points at least `inset` away from the state's border.

    Random points rather than the representative point, because that is where a state label sits.
    """
    inner = geom.buffer(-inset)
    if not inner.is_empty:
        minx, miny, maxx, maxy = inner.bounds
        xs, ys = rng.uniform(minx, maxx, 400), rng.uniform(miny, maxy, 400)
        inside = shapely.contains_xy(inner, xs, ys)
        if inside.any():
            return np.column_stack([xs[inside], ys[inside]])[:k]
    p = geom.representative_point()
    return np.array([[p.x, p.y]])


def pixel_check(buf, ax, geo, facecolors, rng):
    """Codes of states whose pixels in the image do not have the color the target says."""
    height, width, _ = buf.shape
    (x0, _), (x1, _) = ax.transData.transform([(0, 0), (1000, 0)])
    inset = 3 * 1000 / abs(x1 - x0)  # 3 pixels, in map units
    failed = []
    for i, code in enumerate(geo["codes"]):
        points = interior_points(geo["drawn"][i], inset, rng)
        px = ax.transData.transform(points)
        cols = np.floor(px[:, 0]).astype(int)
        rows = np.floor(height - px[:, 1]).astype(int)
        ok = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
        expected = np.round(np.array(facecolors[i][:3]) * 255)
        hits = (np.abs(buf[rows[ok], cols[ok]].astype(int) - expected).max(axis=1) <= 3).sum()
        if hits * 2 < len(points):  # most sampled pixels must match (a text label may cover a few)
            failed.append(code)
    return failed


def text_outside_image(fig, texts):
    """Names of text blocks (title, legend, color bar) that are cut off by the image edge."""
    renderer = fig.canvas.get_renderer()
    width, height = fig.bbox.width, fig.bbox.height
    clipped = []
    for name, artist in texts.items():
        box = artist.get_tightbbox(renderer)
        if box.x0 < 0 or box.y0 < 0 or box.x1 > width or box.y1 > height:
            clipped.append(name)
    return clipped


def draw_map(rng, geo, facecolors, title, legend_title, leg, cmap, bin_colors, no_data, has_missing, check_rng):
    legend_pos = str(rng.choice(["right", "bottom"]))
    if leg["type"] == "binned" and len(leg["labels"]) + has_missing > 5:
        legend_pos = "right"  # one row at the bottom would get too wide
    width_in = float(rng.uniform(7.5, 10.5))
    height_in = width_in * (0.68 if legend_pos == "right" else 0.8)
    style = {
        "legend_pos": legend_pos,
        "background": str(rng.choice(BACKGROUNDS)),
        "border": str(rng.choice(["white", "black", "#555555", "#999999"])),
        "border_width": round(float(rng.uniform(0.3, 1.0)), 2),
        "font": str(rng.choice(["sans-serif", "serif"])),
        "state_labels": bool(rng.random() < 0.4),
        "title_align": str(rng.choice(["center", "left"])),
    }
    with plt.rc_context({"font.family": style["font"]}):
        fig = plt.figure(figsize=(width_in, height_in), dpi=DPI, facecolor=style["background"])
        ax = fig.add_axes([0.02, 0.02, 0.72, 0.87] if legend_pos == "right" else [0.03, 0.18, 0.94, 0.71])
        ax.set_axis_off()
        ax.margins(0.02)
        gpd.GeoSeries(geo["drawn"]).plot(ax=ax, color=facecolors, edgecolor=style["border"],
                                         linewidth=style["border_width"])

        if style["state_labels"]:
            size = float(rng.uniform(5.5, 8))
            for i, (x, y) in enumerate(geo["label_xy"]):
                if geo["codes"][i] in SMALL_STATES:
                    continue
                r, g, b = facecolors[i][:3]
                ax.text(x, y, geo["codes"][i], ha="center", va="center", fontsize=size,
                        color="black" if 0.299 * r + 0.587 * g + 0.114 * b > 0.5 else "white")

        texts = {"title": fig.text(0.5 if style["title_align"] == "center" else 0.03, 0.96, title,
                                   ha=style["title_align"], va="top", fontsize=float(rng.uniform(13, 18)),
                                   weight=str(rng.choice(["bold", "normal"])))}

        text_size = float(rng.uniform(8, 11))
        no_data_patch = Patch(facecolor=no_data, edgecolor="#666666", linewidth=0.5, label="No data")
        if leg["type"] == "binned":
            handles = [Patch(facecolor=c, edgecolor="#666666", linewidth=0.5, label=label)
                       for c, label in zip(bin_colors, leg["labels"])]
            if rng.random() < 0.5:
                handles.reverse()  # some maps list the highest range first
            if has_missing:
                handles.append(no_data_patch)
            kw = dict(handles=handles, title=legend_title, prop={"size": text_size}, title_fontsize=text_size + 1,
                      frameon=bool(rng.random() < 0.4))
            if legend_pos == "right":
                texts["legend"] = fig.legend(loc="center left", bbox_to_anchor=(0.75, 0.45), **kw)
            else:
                texts["legend"] = fig.legend(loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=len(handles), **kw)
        else:
            mappable = ScalarMappable(norm=Normalize(leg["vmin"], leg["vmax"]), cmap=cmap)
            if legend_pos == "right":
                cax, orientation = fig.add_axes([0.84, 0.28, 0.025, 0.45]), "vertical"
            else:
                cax, orientation = fig.add_axes([0.25, 0.12, 0.5, 0.03]), "horizontal"
            bar = fig.colorbar(mappable, cax=cax, orientation=orientation)
            bar.set_ticks(leg["ticks"], labels=[fmt(t, leg["tick_decimals"]) for t in leg["ticks"]])
            bar.ax.tick_params(labelsize=text_size)
            if legend_title:
                bar.set_label(legend_title, fontsize=text_size + 1)
            bar.outline.set_linewidth(0.5)
            texts["color bar"] = cax
            if has_missing:
                if legend_pos == "right":
                    texts["no-data box"] = fig.legend(handles=[no_data_patch], loc="upper left",
                                                      bbox_to_anchor=(0.82, 0.2), prop={"size": text_size},
                                                      frameon=False)
                else:
                    texts["no-data box"] = fig.legend(handles=[no_data_patch], loc="center left",
                                                      bbox_to_anchor=(0.78, 0.135), prop={"size": text_size},
                                                      frameon=False)

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        failed = pixel_check(buf, ax, geo, facecolors, check_rng)
        clipped = text_outside_image(fig, texts)
        plt.close(fig)
    return buf, failed, clipped, style


# ---------------------------------------------------------------- one sample

_GEO = None


def init_worker(cache_dir):
    global _GEO
    _GEO = prepare_geography(cache_dir)


def make_sample(job):
    index, split, seed, out_dir, jpeg_prob = job
    geo = _GEO
    codes = geo["codes"]
    rng = np.random.default_rng([seed, index])
    sample_id = f"synth_{split}_{index:06d}"

    topic, unit, low, high = TOPICS[int(rng.integers(len(TOPICS)))]
    title = str(rng.choice(TITLE_TEMPLATES)).format(topic=topic, year=int(rng.integers(2005, 2025)))
    data_type = "percentage" if unit == "%" else "number"
    # percentage maps always show "%" above the legend; otherwise nothing in the image would say so
    legend_title = unit if data_type == "percentage" or rng.random() < 0.85 else None

    values, pattern = make_values(rng, geo["adjacency"], geo["lonlat"], low, high)
    if rng.random() < 0.25:
        values[rng.choice(len(codes), size=int(rng.integers(1, 4)), replace=False)] = np.nan
    missing = np.isnan(values)

    palette, cmap, diverging = pick_palette(rng)
    bin_colors = None
    if rng.random() < 0.5:
        leg = continuous_legend(rng, values)
        norm = Normalize(leg["vmin"], leg["vmax"])
        colors = [None if m else cmap(norm(v)) for v, m in zip(values, missing)]
        palette_rgbs = cmap(np.linspace(0, 1, 64))[:, :3]
        rounded = round_to_bar([None if m else v for v, m in zip(values, missing)], leg["vmin"], leg["vmax"])
        target = {"title": title, "legend_title": legend_title, "legend_type": "continuous", "data_type": data_type,
                  "values": dict(zip(codes, rounded))}
    else:
        leg = binned_legend(rng, values)
        start = 0.0 if diverging else float(rng.uniform(0.0, 0.3))
        bin_colors = cmap(np.linspace(start, 1.0, len(leg["labels"])))
        bins = np.searchsorted(np.asarray(leg["edges"][1:-1]), np.nan_to_num(values), side="right")
        colors = [None if m else tuple(bin_colors[b]) for b, m in zip(bins, missing)]
        palette_rgbs = bin_colors[:, :3]
        target = {"title": title, "legend_title": legend_title, "legend_type": "discrete", "data_type": data_type,
                  "ranges": leg["labels"], "values": {c: None if m else int(b) for c, b, m in zip(codes, bins, missing)}}

    no_data = pick_no_data_color(palette_rgbs)
    facecolors = [to_rgba(no_data) if c is None else tuple(c) for c in colors]
    buf, failed, clipped, style = draw_map(rng, geo, facecolors, title, legend_title, leg, cmap, bin_colors,
                                           no_data, bool(missing.any()), np.random.default_rng([seed, index, 1]))

    image = Image.fromarray(buf)
    if rng.random() < jpeg_prob:
        quality = int(rng.integers(60, 96))
        path = out_dir / "images" / f"{sample_id}.jpg"
        image.save(path, quality=quality)
    else:
        quality = None
        path = out_dir / "images" / f"{sample_id}.png"
        image.save(path)

    style.update(palette=palette, no_data_color=no_data, jpeg_quality=quality, size_px=[image.width, image.height])
    return {
        "id": sample_id,
        "image": f"images/{path.name}",
        "target": json.dumps(target, ensure_ascii=False, separators=(",", ":")),
        "legend_type": target["legend_type"],
        "data_type": data_type,
        "true_values": {c: None if m else round(float(v), 6) for c, v, m in zip(codes, values, missing)},
        "legend": leg,
        "value_pattern": pattern,
        "style": style,
        "pixel_check_failed": failed,
        "text_clipped": clipped,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=sorted(SPLIT_SEEDS), default="train")
    parser.add_argument("--n", type=int, default=100, help="number of maps to generate")
    parser.add_argument("--out", type=Path, default=Path("data/synth_usa"))
    parser.add_argument("--cache", type=Path, default=Path("data/boundaries"),
                        help="where the geoBoundaries file and adjacency graph are stored")
    parser.add_argument("--seed", type=int, help="default: 0 for train, 1 for dev, 2 for test")
    parser.add_argument("--start", type=int, default=0, help="first sample index; >0 appends to the split's files")
    parser.add_argument("--jpeg-prob", type=float, default=0.5, help="share of images saved as JPEG")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()
    seed = SPLIT_SEEDS[args.split] if args.seed is None else args.seed

    load_or_build_adjacency(load_states(args.cache), args.cache)  # download and build once, before workers start
    (args.out / "images").mkdir(parents=True, exist_ok=True)
    jobs = [(i, args.split, seed, args.out, args.jpeg_prob) for i in range(args.start, args.start + args.n)]

    counts = {"binned": 0, "continuous": 0}
    rejected = 0
    mode = "a" if args.start > 0 else "w"
    with open(args.out / f"{args.split}.jsonl", mode, encoding="utf-8") as good, \
            open(args.out / f"{args.split}.rejected.jsonl", mode, encoding="utf-8") as bad:
        if args.workers <= 1:
            init_worker(args.cache)
            records = map(make_sample, jobs)
            pool = None
        else:
            pool = Pool(args.workers, initializer=init_worker, initargs=(args.cache,))
            records = pool.imap(make_sample, jobs, chunksize=4)
        for done, rec in enumerate(records, 1):
            problems = rec["pixel_check_failed"] + [f"{name} cut off" for name in rec["text_clipped"]]
            if problems:
                rejected += 1
                bad.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(f"  rejected {rec['id']}: {', '.join(problems)}")
            else:
                good.write(json.dumps(rec, ensure_ascii=False) + "\n")
                counts[rec["legend"]["type"]] += 1
            if done % 500 == 0:
                print(f"  {done}/{args.n}", flush=True)
        if pool:
            pool.close()
            pool.join()

    print(f"{args.split}: kept {counts['binned'] + counts['continuous']} maps in {args.out / (args.split + '.jsonl')} "
          f"({counts['binned']} discrete, {counts['continuous']} continuous); rejected {rejected}.")


if __name__ == "__main__":
    main()
