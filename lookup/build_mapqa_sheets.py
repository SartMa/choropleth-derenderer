"""Region sheets for the 400 MapQA test maps every VLM run was scored on, cut with the lookup/ boundary fits.

MapQA's borders are thin white lines, so cells keep the synthetic recipe (eroded interior, commonest colour) without
dropping border colour. Writes data/mapqa_sheets/ in the layout train/read_real.py reads.

    PYTHONPATH=lookup:train .venv/bin/python lookup/build_mapqa_sheets.py
"""

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import build_real_sheets as brs
import find_legend as fl
import read_map as rm
import sheet as sh

OUT = Path("data/mapqa_sheets")
ROOT = Path("data/mapqa_s/MapQA_S")
TEMPLATE = rm.template(Path("data/boundaries/geoBoundaries-USA-ADM1_simplified.geojson"))
FITS = {r["id"]: r["fit"] for r in map(json.loads, open("outputs/lookup-mapqa/read.jsonl"))}
DRAWER = None


def build(map_id):
    global DRAWER
    if DRAWER is None:
        DRAWER = sh.Sheets("data/region_probe")
    path = ROOT / "images" / f"{map_id}.png"
    image = Image.open(path).convert("RGB")
    fit = np.asarray(FITS[map_id])
    country = np.zeros((image.height, image.width), np.uint8)
    for rings in TEMPLATE.values():
        cv2.fillPoly(country, [rm.to_pixels(ring, fit) for ring in rings], 1)
    country = cv2.dilate(country, np.ones((7, 7), np.uint8))
    box, _ = fl.legend_box(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR), country)
    legend = image.crop(box) if box else image.crop((int(0.75 * image.width), 0, image.width, image.height))
    cells = brs.region_cells(np.asarray(image), TEMPLATE, fit, sorted(TEMPLATE), drop_border=False)
    DRAWER.draw(cells, legend).save(OUT / "sheets" / f"{map_id}.png")
    image.save(OUT / "maps" / f"{map_id}.png")
    return {"id": map_id, "map": f"maps/{map_id}.png", "sheet": f"sheets/{map_id}.png", "legend_found": box is not None}


def main():
    (OUT / "maps").mkdir(parents=True, exist_ok=True)
    (OUT / "sheets").mkdir(exist_ok=True)
    ids = json.loads((ROOT.parent / "test400_ids.json").read_text())
    with ProcessPoolExecutor(6) as pool:
        rows = list(pool.map(build, ids, chunksize=4))
    with open(OUT / "records.jsonl", "w") as out:
        for row in rows:
            out.write(json.dumps(row) + "\n")
    print(len(rows), "sheets, legend found on", sum(r["legend_found"] for r in rows))


if __name__ == "__main__":
    main()
