"""Region labels for the synthetic maps, and real India maps, as one small Kaggle dataset for train/region_head.py.

Every synthetic map is the same drawing placed differently, so one raster of region numbers plus each map's size is
enough to know which region lies under any pixel. region_box repeats matplotlib's layout, so no map is redrawn.
Real maps come from the CMA folder, shrunk. Their only state labels are CMA's own Mask R-CNN boxes on 10 test maps.

    .venv/bin/python synth/region_labels.py --out data/region_probe
"""

import argparse
import ast
import csv
import json
import unicodedata
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from shapely import unary_union
from shapely.geometry import MultiPolygon

import generate_maps as gm

WIDTH = 1600  # of the raster; tiny regions keep at least a few cells
MARGIN = 0.02  # ax.margins in draw_map
AXES = {"right": (0.02, 0.02, 0.72, 0.87), "bottom": (0.03, 0.18, 0.94, 0.71)}  # fig.add_axes in draw_map
REAL_LONGEST = 1200


def plain(name):
    """A region's name without accents or ampersands, so the two sources' spellings match."""
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return name.replace("&", "and").casefold()


def polygons(shape):
    return list(shape.geoms) if isinstance(shape, MultiPolygon) else [shape]


def build_raster(geo):
    """A raster of region numbers (0 = nothing, then 1.. in the order of codes), biggest region drawn first."""
    x0, y0, x1, y1 = unary_union(geo["drawn"]).bounds
    scale = WIDTH / (x1 - x0)
    raster = np.zeros((int(np.ceil((y1 - y0) * scale)) + 1, WIDTH + 1), np.uint8)
    order = sorted(range(len(geo["codes"])), key=lambda i: -geo["drawn"][i].area)
    for i in order:
        rings = [np.round((np.asarray(p.exterior.coords)[:, :2] - [x0, y0]) * scale).astype(np.int32)
                 for p in polygons(geo["drawn"][i])]
        cv2.fillPoly(raster, rings, i + 1)
    return raster[::-1], [float(v) for v in (x0, y0, x1, y1)]  # image rows run downwards, map y upwards


def region_box(size, legend_pos, bounds):
    """Where the drawing's bounds land on the image: (left, top, right, bottom) in pixels, like matplotlib's layout."""
    width, height = size
    left, bottom, ax_w, ax_h = AXES[legend_pos]
    x0, y0, x1, y1 = bounds
    span_x, span_y = (x1 - x0) * (1 + 2 * MARGIN), (y1 - y0) * (1 + 2 * MARGIN)
    box_w, box_h = ax_w * width, ax_h * height
    scale = min(box_w / span_x, box_h / span_y)  # aspect equal: the axes shrink around their centre
    centre_x, centre_y = (left + ax_w / 2) * width, height - (bottom + ax_h / 2) * height
    half_w, half_h = (x1 - x0) * scale / 2, (y1 - y0) * scale / 2
    return centre_x - half_w, centre_y - half_h, centre_x + half_w, centre_y + half_h


def placements(folder, splits, bounds):
    placed = {}
    for split in splits:
        for line in open(folder / f"{split}.jsonl"):
            record = json.loads(line)
            placed[record["id"]] = region_box(record["style"]["size_px"], record["style"]["legend_pos"], bounds)
    return placed


def real_maps(cma, out):
    """Shrink the real maps and turn CMA's boxes into shares of the image. {file: {code: [x0, y0, x1, y1]}}."""
    (out / "real").mkdir(parents=True, exist_ok=True)
    folders = ["India/Legend Classification/train", "India/Legend Classification/val", "India/State Segmentation/test"]
    paths = {}
    for folder in folders:
        for path in sorted((cma / folder).rglob("*")):
            if path.suffix.lower() in (".png", ".jpg", ".jpeg"):
                paths.setdefault(path.name, path)
    sizes = {}
    for name, path in paths.items():
        image = Image.open(path).convert("RGB")
        sizes[name] = image.size
        image.thumbnail((REAL_LONGEST, REAL_LONGEST))
        image.save(out / "real" / f"{Path(name).stem}.jpg", quality=92)
    return sizes


def real_boxes(cma, sizes, codes):
    """CMA's boxes, each as (top, left, bottom, right) in the original image's pixels."""
    boxes = {}
    with open(cma / "India/State Segmentation/test_result/output_objects_seg_with_color.csv") as f:
        for row in csv.DictReader(f):
            code = codes[plain(row["Class Name"])]
            if row["File Name"] not in sizes:
                continue
            boxes.setdefault(row["File Name"], {})[code] = ast.literal_eval(row["BoundingBox"])
    return boxes


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("data/region_probe"))
    parser.add_argument("--cache", type=Path, default=Path("data/boundaries"))
    parser.add_argument("--cma", type=Path, default=Path("data/cma"))
    parser.add_argument("--maps", type=Path, default=Path("data/synth_ind"))
    parser.add_argument("--maps-usa", type=Path, default=Path("data/synth_usa"))
    args = parser.parse_args()

    usa = gm.prepare_geography(args.cache, "usa")
    raster, bounds = build_raster(usa)
    args.out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(raster).save(args.out / "regions_usa.png", optimize=True)
    placed = placements(args.maps_usa, ("train", "dev", "test"), bounds)
    (args.out / "regions_usa.json").write_text(json.dumps({"codes": list(usa["codes"]), "placed": placed}))

    geo = gm.prepare_geography(args.cache, "ind")
    raster, bounds = build_raster(geo)
    args.out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(raster).save(args.out / "regions.png", optimize=True)
    names = {plain(f["properties"]["shapeName"]): f["properties"]["shapeISO"].split("-")[1] for f in json.loads(
        (args.cache / "geoBoundaries-IND-ADM1_simplified.geojson").read_text())["features"]}
    sizes = real_maps(args.cma, args.out)
    boxes = real_boxes(args.cma, sizes, names)
    placed = placements(args.maps, ("train", "test"), bounds)
    meta = {"codes": list(geo["codes"]), "listed": [c for c, keep in zip(geo["codes"], geo["listed"]) if keep],
            "bounds": bounds, "placed": placed, "real_sizes": sizes, "real_boxes": boxes}
    (args.out / "regions.json").write_text(json.dumps(meta))
    print(f"{len(meta['codes'])} regions, raster {raster.shape}, {len(sizes)} real maps, boxes on {len(boxes)}")


if __name__ == "__main__":
    main()
