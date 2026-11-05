"""Read a choropleth with code and a VLM: code finds the regions and their colours, the VLM says what they mean.

The steps:
- template:   project the boundary file into unit coordinates (Albers USA, with Alaska and Hawaii moved)
- fit_map:    find the scale and position that lay the template's borders on the map's borders
- region_colours: take each region's colour from the flat patch of the image inside its outline
- group_by_colour, name_groups: discrete maps; regions of one colour share a class, and the VLM's answers name it
- find_colour_bar, read_positions, read_ticks, scale_from_ticks: continuous maps; OCR reads the bar's numbers,
  and the VLM's answers (scale_bar) are the fallback
"""

import json
import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage, optimize, signal, stats
from scipy.cluster import hierarchy
from skimage import color

STATES = ("AK AL AR AZ CA CO CT DE FL GA HI IA ID IL IN KS KY LA MA MD ME MI MN MO MS MT NC ND NE NH NJ NM NV NY OH "
          "OK OR PA RI SC SD TN TX UT VA VT WA WI WV WY").split()


def conic(lon, lat, centre_lon, centre_lat, parallels):
    """Albers equal-area, as d3 writes it: lon, lat and the result in radians / unit scale, y upwards."""
    lon, lat, lat1, lat2 = np.radians(lon), np.radians(lat), *np.radians(parallels)
    n = (np.sin(lat1) + np.sin(lat2)) / 2
    c = np.cos(lat1) ** 2 + 2 * n * np.sin(lat1)
    rho0 = np.sqrt(c - 2 * n * np.sin(np.radians(centre_lat))) / n
    rho = np.sqrt(c - 2 * n * np.sin(lat)) / n
    theta = n * (lon - np.radians(centre_lon))
    return rho * np.sin(theta), rho0 - rho * np.cos(theta)


def albers_usa(code, lon, lat):
    """d3's albersUsa, the projection Plotly uses: the lower 48, and Alaska and Hawaii moved beside them. y downwards."""
    if code == "AK":
        lon = np.where(lon > 0, lon - 360, lon)  # the Aleutians cross the date line
        x, y = conic(lon, lat, -156, 58.5, (55, 65))
        return 0.35 * x - 0.307, -0.35 * y + 0.201
    if code == "HI":
        x, y = conic(lon, lat, -160, 19.9, (8, 18))
        return x - 0.205, -y + 0.212
    x, y = conic(lon, lat, -96.6, 38.7, (29.5, 45.5))
    return x, -y


def template(geojson, codes=STATES, project=albers_usa):
    """{code: [ring, ...]} with each ring an array of (x, y) in unit coordinates."""
    regions = {}
    for feature in json.loads(Path(geojson).read_text())["features"]:
        code = feature["properties"]["shapeISO"].split("-")[1]
        if code not in codes:
            continue
        geometry = feature["geometry"]
        polygons = geometry["coordinates"] if geometry["type"] == "MultiPolygon" else [geometry["coordinates"]]
        rings = []
        for polygon in polygons:
            ring = np.asarray(polygon[0], dtype=float)[:, :2]
            rings.append(np.stack(project(code, ring[:, 0], ring[:, 1]), axis=1))
        regions[code] = rings
    return regions


def border_points(regions, spacing=0.002):
    """Evenly spaced points along every region's outline; 0.002 is about one pixel at the usual scale."""
    points = []
    for rings in regions.values():
        for ring in rings:
            steps = np.linalg.norm(np.diff(ring, axis=0), axis=1)
            along = np.concatenate([[0], np.cumsum(steps)])
            if along[-1] < 3 * spacing:
                continue
            at = np.arange(0, along[-1], spacing)
            points.append(np.stack([np.interp(at, along, ring[:, 0]), np.interp(at, along, ring[:, 1])], axis=1))
    return np.concatenate(points)


def to_lab(image):
    return color.rgb2lab(image / 255.0)


def edge_distance(lab, threshold=10.0, cap=12.0):
    """For every pixel, how far it is to the nearest edge, capped so that one stray edge cannot dominate a fit."""
    gradient = sum(np.hypot(cv2.Sobel(lab[..., i], cv2.CV_64F, 1, 0), cv2.Sobel(lab[..., i], cv2.CV_64F, 0, 1))
                   for i in range(3))
    edges = (gradient > threshold).astype(np.uint8)
    return np.minimum(cv2.distanceTransform(1 - edges, cv2.DIST_L2, 3), cap)


def fit_map(lab, points, scales=np.arange(380, 800, 8.0), start=None):
    """Find (scale_x, scale_y, x, y) laying the border points on the image's edges.

    Coarse: at each scale, a correlation over the image tries every position at once. Fine: a local search from the
    best one, allowing x and y to scale separately. Returns the fit and the share of points landing on an edge.
    With start, the coarse search is skipped and the fine one begins there.
    """
    distance = edge_distance(lab)
    height, width = distance.shape
    best = (np.inf, start)
    for scale in scales if start is None else ():
        scaled = points * scale
        low = scaled.min(0)
        size = np.ceil(scaled.max(0) - low).astype(int) + 1
        if size[0] >= width or size[1] >= height:
            continue
        ink = np.zeros((size[1], size[0]))
        cells = np.round(scaled - low).astype(int)
        np.add.at(ink, (cells[:, 1], cells[:, 0]), 1.0)
        cost = signal.fftconvolve(distance, ink[::-1, ::-1], mode="valid") / len(points)
        y, x = np.unravel_index(np.argmin(cost), cost.shape)
        if cost[y, x] < best[0]:
            best = (cost[y, x], (scale, scale, x - low[0], y - low[1]))

    def cost(fit, cap=4.0):
        x = points[:, 0] * fit[0] + fit[2]
        y = points[:, 1] * fit[1] + fit[3]
        return np.minimum(ndimage.map_coordinates(distance, [y, x], order=1, mode="nearest"), cap).mean()

    result = optimize.minimize(cost, best[1], method="Powell", options={"xtol": 0.02, "ftol": 1e-4, "maxiter": 40})
    fit = result.x
    x = points[:, 0] * fit[0] + fit[2]
    y = points[:, 1] * fit[1] + fit[3]
    on_edge = (ndimage.map_coordinates(distance, [y, x], order=1, mode="nearest") <= 1.5).mean()
    return tuple(float(v) for v in fit), float(on_edge)


def to_pixels(ring, fit):
    return np.round(ring * fit[:2] + fit[2:]).astype(np.int32)


def flat_patches(lab, threshold=6.0):
    """Label the image's flat patches: connected pixels with no colour edge between them."""
    gradient = sum(np.hypot(cv2.Sobel(lab[..., i], cv2.CV_64F, 1, 0, ksize=1), cv2.Sobel(lab[..., i], cv2.CV_64F, 0, 1, ksize=1))
                   for i in range(3))
    return cv2.connectedComponents((gradient < threshold).astype(np.uint8), connectivity=4)[1]


def region_colours(lab, regions, fit, deepest=9, smallest_patch=4):
    """Each region's colour (median Lab) and how many pixels it came from.

    A boundary file never matches a drawn map exactly, which matters for a state a few pixels wide. So the region's
    colour comes from the flat patch of the image that lies mostly inside its outline; only if there is none, from
    the pixels deepest inside the outline.
    """
    height, width = lab.shape[:2]
    fit = np.asarray(fit)
    patches = flat_patches(lab)
    sizes = np.bincount(patches.ravel())
    colours, pixels = {}, {}
    for code, rings in regions.items():
        mask = np.zeros((height, width), np.uint8)
        cv2.fillPoly(mask, [to_pixels(ring, fit) for ring in rings], 1)
        if not mask.any():
            continue
        overlap = np.bincount(patches[mask > 0], minlength=len(sizes))
        overlap[0] = 0  # patch 0 is the edges themselves
        mostly_inside = np.where((overlap >= 0.5 * sizes) & (sizes >= smallest_patch))[0]
        if len(mostly_inside):
            chosen = patches == mostly_inside[np.argmax(overlap[mostly_inside])]
        else:
            inside = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
            chosen = inside >= max(np.partition(inside.ravel(), -deepest)[-deepest], 1e-6)
        colours[code] = np.median(lab[chosen], axis=0)
        pixels[code] = int(chosen.sum())
    return colours, pixels


def paper_colour(lab):
    """The page background: the commonest colour along the image's edge."""
    edge = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    return np.median(edge, axis=0)


def non_paper(lab, paper, tolerance=2.5):
    return (color.deltaE_cie76(lab, paper[None, None, :]) > tolerance).astype(np.uint8)


def plot_box(found):
    """The bounding box of the biggest thing on the page, which is the map's plot area."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(found, connectivity=8)
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = stats[biggest, :4]
    return x, y, x + w, y + h


def find_colour_bar(lab):
    """The colour bar: (Lab colours low value first, (x, y, w, h)). None if there is no tall bar."""
    found = non_paper(lab, paper_colour(lab))
    x0, y0, x1, y1 = plot_box(found)
    found = found.copy()
    found[y0:y1, x0:x1] = 0
    found = cv2.morphologyEx(found, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(found, connectivity=4)
    if count < 2:
        return None
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, _ = stats[biggest]
    if h < 4 * w or h <= 80:
        return None
    middle = lab[y + 2: y + h - 2, x + w // 2 - 3: x + w // 2 + 4]
    return np.median(middle, axis=1)[::-1], (int(x), int(y), int(w), int(h))  # a tall bar's top is the high end


TICK = re.compile(r"^[-\u2212]?\d[\d,]*(\.\d+)?([kKMBG%]?)$")
UNITS = {"": 1, "%": 1, "k": 1e3, "K": 1e3, "M": 1e6, "B": 1e9, "G": 1e9}


def read_ticks(image, box, ocr):
    """[(row of the image, value)] for the numbers beside the bar, read by OCR at 3x."""
    x, y, w, h = box
    top = max(0, y - 12)
    crop = image[top: y + h + 12, x + w: min(image.shape[1], x + w + 80)].astype(np.uint8)
    crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    ticks = []
    for corners, text, _ in ocr(crop)[0] or []:
        found = TICK.match(text.replace(" ", ""))
        if found:
            number = text.replace(" ", "").replace(",", "").replace("\u2212", "-").rstrip("kKMBG%")
            ticks.append((top + float(np.mean([c[1] for c in corners])) / 3, float(number) * UNITS[found.group(2)]))
    return ticks


def scale_from_ticks(positions, box, ticks):
    """Continuous maps: value = a + b * row through the tick numbers. {code: value}.

    Ticks are evenly spaced, so a misread one sits off the line: the line is the pair of ticks most others agree with.
    """
    _, y, _, h = box
    rows, values = map(np.asarray, zip(*ticks))
    tolerance = 0.25 * np.median(np.abs(np.diff(values)))
    best = np.zeros(len(rows), bool)
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            if rows[i] != rows[j]:
                slope = (values[j] - values[i]) / (rows[j] - rows[i])
                agree = np.abs(values[i] + slope * (rows - rows[i]) - values) <= tolerance
                if agree.sum() > best.sum():
                    best = agree
    slope, intercept = np.polyfit(rows[best], values[best], 1)
    bottom, top = y + h - 3, y + 2
    return {c: float(intercept + slope * (bottom - t * (bottom - top))) for c, t in positions.items()}


def read_positions(colours, bar):
    """{code: position along the colour bar, 0 to 1}."""
    answer = {}
    for code, lab in colours.items():
        gaps = color.deltaE_ciede2000(np.repeat(lab[None], len(bar), axis=0), bar)
        answer[code] = float(np.argmin(gaps) / (len(bar) - 1))
    return answer


def group_by_colour(colours, pixels, same=1.5, far=10.0, reliable=30):
    """Sort regions into groups of one colour: {code: group}. A region outside every group gets group None.

    Groups form from regions with enough pixels to trust. Smaller ones join the nearest group, unless they are
    too far from all of them, which happens when a tiny region is the only one in its class.
    """
    trusted = [c for c in colours if pixels[c] >= reliable]
    table = np.asarray([colours[c] for c in trusted])
    groups = hierarchy.fcluster(hierarchy.linkage(table, "complete"), same, "distance") if len(trusted) > 1 else [1]
    answer = dict(zip(trusted, (int(g) for g in groups)))
    centres = {g: table[np.asarray(groups) == g].mean(0) for g in set(answer.values())}
    for code in colours:
        if code not in answer:
            gaps = {g: float(np.linalg.norm(colours[code] - centre)) for g, centre in centres.items()}
            nearest = min(gaps, key=gaps.get)
            answer[code] = nearest if gaps[nearest] <= far else None
    return answer


def name_groups(groups, vlm_values):
    """Discrete maps: each colour group takes the VLM's commonest answer for it. {code: class}.

    A region in no group keeps the VLM's own answer.
    """
    answer = {}
    for group in set(groups.values()):
        members = [c for c, g in groups.items() if g == group]
        if group is None:
            answer.update({c: vlm_values.get(c) for c in members})
            continue
        votes = Counter(vlm_values.get(c, "missing") for c in members)
        winner = votes.most_common(1)[0][0]
        answer.update({c: None if winner == "missing" else winner for c in members})
    return answer


def scale_bar(positions, vlm_values):
    """Continuous maps: fit value = a + b * position through the VLM's numbers, ignoring its outliers. {code: value}."""
    pairs = [(positions[c], vlm_values[c]) for c in positions if isinstance(vlm_values.get(c), (int, float))]
    x, y = map(np.asarray, zip(*pairs))
    slope, intercept = stats.theilslopes(y, x)[:2]
    return {c: float(intercept + slope * t) for c, t in positions.items()}
