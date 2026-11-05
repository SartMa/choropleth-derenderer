"""Find a real map's legend: the biggest cluster of printed numbers outside the country, read by OCR.

Colours and shapes vary too much between plotting libraries, but every legend prints numbers. The box is grown a little
so swatches or a colour bar beside the numbers come with it. Boxes are saved in the original image's pixels.

    .venv/bin/python lookup/find_legend.py --out outputs/real-fit
"""

import argparse
import json
import random
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

import fit_real as fr
import link_tables as lt
import read_map as rm

FITS_FILE = Path("outputs/real-fit/fits.json")
FITS = {Path(r["name"]).stem: r for r in json.loads(FITS_FILE.read_text())} if FITS_FILE.exists() else {}
READER = None


def country_mask(row, size):
    """The fitted country on the original image, widened a little."""
    width, height = size
    factor = width / row["size"][0]
    fit = np.asarray(row["fit"]) * factor
    mask = np.zeros((height, width), np.uint8)
    for rings in fr.TEMPLATES[row["projection"]][0].values():
        cv2.fillPoly(mask, [rm.to_pixels(ring, fit) for ring in rings], 1)
    grow = max(3, int(0.01 * width)) | 1
    return cv2.dilate(mask, np.ones((grow, grow), np.uint8))


def clusters(boxes, across, down):
    """Groups of boxes (x0, y0, x1, y1) close to each other. Bar ticks sit far apart, hence a longer reach downwards."""
    groups = list(range(len(boxes)))
    def root(i):
        while groups[i] != i:
            i = groups[i]
        return i
    for i, a in enumerate(boxes):
        for j, b in enumerate(boxes[:i]):
            if a[0] - across <= b[2] and b[0] - across <= a[2] and a[1] - down <= b[3] and b[1] - down <= a[3]:
                groups[root(i)] = root(j)
    found = {}
    for i, box in enumerate(boxes):
        found.setdefault(root(i), []).append(box)
    return list(found.values())


def legend_box(image, country):
    """Legend box (x0, y0, x1, y1) on a BGR image, given the country's mask; (None, 0) if no numbers are printed."""
    global READER
    if READER is None:
        from rapidocr_onnxruntime import RapidOCR
        READER = RapidOCR()
    height, width = image.shape[:2]
    boxes = []
    for corners, text, _ in READER(image)[0] or []:
        corners = np.asarray(corners)
        x0, y0, x1, y1 = *corners.min(0), *corners.max(0)
        letters = len(re.findall(r"[A-Za-z]", text))
        if not re.search(r"\d", text) or letters > 4:  # titles hold years among many letters
            continue
        if country[int((y0 + y1) / 2), int((x0 + x1) / 2)]:
            continue
        boxes.append((float(x0), float(y0), float(x1), float(y1)))
    if not boxes:
        return None, 0
    text = float(np.median([b[3] - b[1] for b in boxes]))
    best = max(clusters(boxes, 3 * text, 15 * text), key=len)
    x0, y0 = np.min(best, axis=0)[:2]
    x1, y1 = np.max(best, axis=0)[2:]
    pad_x, pad_y = 5 * text, 3 * text
    box = [max(0, x0 - pad_x), max(0, y0 - pad_y), min(width, x1 + pad_x), min(height, y1 + pad_y)]
    return [int(v) for v in box], text


def find(name):
    """(name, legend box or None, height of its text)."""
    image = cv2.imread(str(lt.originals()[name]))
    box, text = legend_box(image, country_mask(FITS[name], image.shape[1::-1]))
    return name, box, text


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("outputs/real-fit"))
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    links = json.loads(Path("data/cma_tables/links.json").read_text())
    names = sorted(k for k, v in links.items() if v["tier"] != "none" and k in FITS)
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(find, names))
    (args.out / "legends.json").write_text(json.dumps({n: b for n, b, _ in rows}, indent=1))
    print(f"{len(rows)} linked maps, legend found on {sum(b is not None for _, b, _ in rows)}")
    random.seed(3)
    shown = rows if len(rows) <= 36 else random.sample(rows, 36)
    cell_w, cell_h = 330, 236
    sheet = Image.new("RGB", (6 * cell_w, 6 * cell_h), "white")
    for n, (name, box, _) in enumerate(shown):
        image = Image.open(lt.originals()[name]).convert("RGB")
        if box:
            ImageDraw.Draw(image).rectangle(box, outline=(255, 0, 255), width=8)
        image.thumbnail((cell_w - 4, cell_h - 4))
        sheet.paste(image, ((n % 6) * cell_w, (n // 6) * cell_h))
    sheet.save(args.out / "legend_sheet.png")


if __name__ == "__main__":
    main()
