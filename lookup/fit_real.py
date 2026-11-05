"""Fit the India boundary file onto real maps, the first step of cutting a region sheet with no generator to lean on.

Real maps come in several projections, so each is tried and the best fit kept. Each map gets a fit, the share of border
points landing on an image edge, which is the check that decides whether its regions can be trusted.

    .venv/bin/python lookup/fit_real.py --out outputs/real-fit
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from skimage import color

import read_map as rm

GEOJSON = Path("data/boundaries/geoBoundaries-IND-ADM1_simplified.geojson")
REAL = Path("data/region_probe/real")
META = Path("data/region_probe/regions.json")
COARSE = 480  # longest side of the image in the coarse search
SPAN = 500.0  # the template's longest side, in template units
SPACING = 0.5


def plate(code, lon, lat):
    return lon, -lat


def plate_equal_area(code, lon, lat):
    return lon * np.cos(np.radians(22)), -lat


def mercator(code, lon, lat):
    return lon, -np.degrees(np.log(np.tan(np.pi / 4 + np.radians(lat) / 2)))


PROJECTIONS = {"plate": plate, "plate_equal_area": plate_equal_area, "mercator": mercator}


def templates():
    """{name: ({code: rings}, border points)}, each scaled so its longest side is SPAN."""
    codes = [f["properties"]["shapeISO"].split("-")[1] for f in json.loads(GEOJSON.read_text())["features"]]
    answer = {}
    for name, project in PROJECTIONS.items():
        regions = rm.template(GEOJSON, codes, project)
        everything = np.concatenate([ring for rings in regions.values() for ring in rings])
        low, span = everything.min(0), (everything.max(0) - everything.min(0)).max()
        regions = {c: [(ring - low) * SPAN / span for ring in rings] for c, rings in regions.items()}
        answer[name] = regions, rm.border_points(regions, SPACING)
    return answer


TEMPLATES = templates()


def trust(lab, regions, fit, inset=7, smallest=50, same=6.0):
    """Does the fit explain the map? Edges alone do not say: a wrong outline still finds edges to land on.

    uniform:  the typical region's share of interior pixels within `same` of its own median colour
    contrast: the share of the country's interior that differs from the ring of page just outside it
    A right fit gives flat regions on something other than background. Returns (uniform * contrast, uniform, contrast).
    """
    height, width = lab.shape[:2]
    fit = np.asarray(fit)
    country = np.zeros((height, width), np.uint8)
    flat = []
    kernel = np.ones((inset, inset), np.uint8)
    for rings in regions.values():
        mask = np.zeros((height, width), np.uint8)
        cv2.fillPoly(mask, [rm.to_pixels(ring, fit) for ring in rings], 1)
        country |= mask
        inner = lab[cv2.erode(mask, kernel) > 0]
        if len(inner) >= smallest:
            flat.append((color.deltaE_cie76(inner, np.median(inner, axis=0)[None]) <= same).mean())
    if not flat:
        return 0.0, 0.0, 0.0
    ring = cv2.dilate(country, np.ones((25, 25), np.uint8)) - cv2.dilate(country, np.ones((9, 9), np.uint8))
    inside = lab[cv2.erode(country, kernel) > 0]
    if not ring.any() or not len(inside):
        return 0.0, 0.0, 0.0
    uniform = float(np.median(flat))
    contrast = float((color.deltaE_cie76(inside, np.median(lab[ring > 0], axis=0)[None]) > 4).mean())
    return uniform * contrast, uniform, contrast


def fit_one(path):
    """The best fit over the projections: {projection, fit, on_edge}, with on_edge for each one tried."""
    image = Image.open(path).convert("RGB")
    full = rm.to_lab(np.asarray(image, dtype=float))
    small = image.copy()
    small.thumbnail((COARSE, COARSE))
    factor = image.width / small.width
    lab = rm.to_lab(np.asarray(small, dtype=float))
    scales = np.geomspace(0.2, 1.0, 45) * max(small.size) / SPAN
    tried = {}
    for name, (_, points) in TEMPLATES.items():
        (sx, sy, x, y), _ = rm.fit_map(lab, points, scales)
        tried[name] = rm.fit_map(full, points, start=(sx * factor, sy * factor, x * factor, y * factor))
    best = max(tried, key=lambda n: tried[n][1])
    return {"name": path.name, "size": image.size, "projection": best, "fit": tried[best][0],
            "on_edge": tried[best][1], "tried": {n: v[1] for n, v in tried.items()}}


def overlay(rows, path, width=330, columns=6):
    """Contact sheet of fitted outlines in magenta, worst fits first, to judge the check by eye."""
    height = int(width * 0.72)
    shown = sorted(rows, key=lambda r: r["on_edge"])
    sheet = Image.new("RGB", (columns * width, int(np.ceil(len(shown) / columns)) * height), "white")
    for n, row in enumerate(shown):
        image = Image.open(REAL / row["name"]).convert("RGB")
        draw = ImageDraw.Draw(image)
        fit = np.asarray(row["fit"])
        for rings in TEMPLATES[row["projection"]][0].values():
            for ring in rings:
                draw.line([tuple(p) for p in (ring * fit[:2] + fit[2:]).tolist()], fill=(255, 0, 255), width=2)
        image.thumbnail((width - 4, height - 18))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 110, 11), fill="white")
        draw.text((1, 0), f"{row['on_edge']:.3f} {row['projection'][:9]}", fill="black")
        sheet.paste(image, ((n % columns) * width, (n // columns) * height))
    sheet.save(path)


def against_boxes(rows):
    """CMA's boxes on 10 maps, as shares of the image: is each fitted region's centre inside, and how well do boxes overlap?"""
    meta = json.loads(META.read_text())
    inside, overlaps = [], []
    for row in rows:
        original = Path(row["name"]).stem + ".png"
        boxes = meta["real_boxes"].get(original)
        if not boxes:
            continue
        shrink = row["size"][0] / meta["real_sizes"][original][0]
        fit = np.asarray(row["fit"])
        regions = TEMPLATES[row["projection"]][0]
        for code, box in boxes.items():
            top, left, bottom, right = np.asarray(box) * shrink
            points = np.concatenate(regions[code]) * fit[:2] + fit[2:]
            (x0, y0), (x1, y1) = points.min(0), points.max(0)
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            inside.append(left <= cx <= right and top <= cy <= bottom)
            cross = max(0, min(x1, right) - max(x0, left)) * max(0, min(y1, bottom) - max(y0, top))
            overlaps.append(cross / ((x1 - x0) * (y1 - y0) + (right - left) * (bottom - top) - cross))
    return {"regions": len(inside), "centre_in_box": float(np.mean(inside)), "median_box_overlap": float(np.median(overlaps))}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("outputs/real-fit"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    paths = sorted(REAL.glob("*.jpg"))[: args.limit]
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(fit_one, paths))
    (args.out / "fits.json").write_text(json.dumps(rows, indent=1))
    overlay(rows, args.out / "overlay.png")
    edges = np.asarray([r["on_edge"] for r in rows])
    report = {"maps": len(rows), "on_edge_quantiles": np.quantile(edges, [0, .1, .25, .5, .75, .9, 1]).round(3).tolist(),
              "projections": {n: sum(r["projection"] == n for r in rows) for n in PROJECTIONS},
              "against_cma_boxes": against_boxes(rows)}
    (args.out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
