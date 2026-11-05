"""Region labels for the learned answer sheet (train/answer_sheet.py) on maps the generator did not draw.

The region head needs to know which region lies under each patch. Synthetic maps get this from the generator
(data/region_probe). MapQA and real India maps get it from a boundary-file fit, which is done here, on CPU, once:
the fits are used only to teach and to check the head, never when the model reads a map.

Writes data/answer_labels/:
    vocab.json         every region the head knows, "US-AK" ... "IN-WB"; class 0 is "no region"
    mapqa/<id>.png     MapQA maps: 2,000 training maps (head teaching) and the 400 test maps (head checking)
    real/<name>.png    real India maps whose fit passed the trust check (head checking only)
Each label PNG is the image's size, holding class numbers.

    .venv/bin/python lookup/answer_labels.py
"""

import argparse
import json
import random
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import fit_real as fr
import read_map as rm

OUT = Path("data/answer_labels")
ROOT = Path("data/mapqa_s/MapQA_S")
ZIP = Path.home() / "Downloads" / "MapQA_S.zip"
USA = rm.template(Path("data/boundaries/geoBoundaries-USA-ADM1_simplified.geojson"))
USA_POINTS = rm.border_points(USA)
REGIONS = json.loads(Path("data/region_probe/regions.json").read_text())
VOCAB = ["-"] + [f"US-{c}" for c in rm.STATES] + [f"IN-{c}" for c in REGIONS["codes"]]
GOOD_EDGE = 0.95  # MapQA test fits all score at least 0.967


def paint(size, regions, fit, prefix):
    """Class numbers at every pixel: biggest regions first, so small ones drawn inside them stay visible."""
    width, height = size
    label = np.zeros((height, width), np.uint8)
    fit = np.asarray(fit)
    order = sorted(regions, key=lambda c: -sum(cv2.contourArea(rm.to_pixels(r, fit)) for r in regions[c]))
    for code in order:
        cv2.fillPoly(label, [rm.to_pixels(ring, fit) for ring in regions[code]], VOCAB.index(f"{prefix}-{code}"))
    return label


def fit_mapqa(map_id):
    image = Image.open(ROOT / "images" / f"{map_id}.png").convert("RGB")
    fit, on_edge = rm.fit_map(rm.to_lab(np.asarray(image, dtype=float)), USA_POINTS)
    if on_edge >= GOOD_EDGE:
        Image.fromarray(paint(image.size, USA, fit, "US")).save(OUT / "mapqa" / f"{map_id}.png", optimize=True)
    return map_id, on_edge


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-maps", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    for folder in ("mapqa", "real"):
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    (OUT / "vocab.json").write_text(json.dumps(VOCAB))

    # MapQA test maps: the fits made for lookup/ already
    for row in map(json.loads, open("outputs/lookup-mapqa/read.jsonl")):
        size = Image.open(ROOT / "images" / f"{row['id']}.png").size
        Image.fromarray(paint(size, USA, row["fit"], "US")).save(OUT / "mapqa" / f"{row['id']}.png", optimize=True)

    # MapQA training maps: unpacked from the zip and fitted here
    ids = [json.loads(line)["id"] for line in open("data/mapqa_derender/train.jsonl")]
    chosen = random.Random(1).sample(ids, args.train_maps)
    with zipfile.ZipFile(ZIP) as z:
        for map_id in chosen:
            target = ROOT / "images" / f"{map_id}.png"
            if not target.exists():
                target.write_bytes(z.read(f"MapQA_S/images/{map_id}.png"))
    with ProcessPoolExecutor(args.workers) as pool:
        edges = dict(pool.map(fit_mapqa, chosen, chunksize=8))
    kept = [i for i, e in edges.items() if e >= GOOD_EDGE]
    (OUT / "mapqa_train_ids.json").write_text(json.dumps(sorted(kept)))

    # Real India maps that passed the trust check
    trusted = {row[0] for row in json.loads(Path("outputs/real-fit/trust.json").read_text())
               if row[1] >= 0.8 and row[4] >= 0.9}
    for row in json.loads(Path("outputs/real-fit/fits.json").read_text()):
        if row["name"] in trusted:
            regions = fr.TEMPLATES[row["projection"]][0]
            label = paint(row["size"], regions, row["fit"], "IN")
            Image.fromarray(label).save(OUT / "real" / f"{Path(row['name']).stem}.png", optimize=True)
    print(f"{len(VOCAB)} classes; MapQA training fits kept {len(kept)}/{len(chosen)}; real India maps {len(trusted)}")


if __name__ == "__main__":
    main()
