"""Build the region sheet for every real map linked to a table: regions from the fit, legend from the OCR box.

Writes data/real_sheets/ (maps, sheets, records.jsonl), which goes to Kaggle. True values stay here, and are scored by
lookup/score_real.py.

    PYTHONPATH=lookup:train .venv/bin/python lookup/build_real_sheets.py
"""

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import fit_real as fr
import link_tables as lt
import read_map as rm
import sheet as sh

OUT = Path("data/real_sheets")
LEGENDS = Path("outputs/real-fit/legends.json")


def border_colour(pixels, regions, fit):
    """The commonest colour along the fitted borders of the big regions."""
    lines = np.zeros(pixels.shape[:2], np.uint8)
    for rings in regions.values():
        cv2.polylines(lines, [rm.to_pixels(ring, fit) for ring in rings], True, 1, 1)
    coarse = pixels[lines > 0].astype(int) // 16
    colours, counts = np.unique(coarse, axis=0, return_counts=True)
    return darkest(colours, counts)


def darkest(colours, counts):
    """Borders are the commonest dark line colour; region fills also lie along the borders, so look among the dark."""
    dark = colours.sum(1) * 16 < 360
    return (colours[dark][np.argmax(counts[dark])] if dark.any() else colours[np.argmax(counts)]) * 16 + 8


def region_cells(pixels, regions, fit, codes, drop_border=True):
    """(code, pixels) per region, without border ink. A region with no colour of its own left gets an empty cell."""
    border = border_colour(pixels, regions, fit) if drop_border else None
    cells = []
    for code in codes:
        mask = np.zeros(pixels.shape[:2], np.uint8)
        cv2.fillPoly(mask, [rm.to_pixels(ring, fit) for ring in regions[code]], 1)
        chosen = mask
        for size in (3, 5, 9):
            inner = cv2.erode(mask, np.ones((size, size), np.uint8))
            if inner.sum() >= 6:
                chosen = inner
        found = pixels[chosen > 0]
        if border is not None:
            found = found[np.abs(found.astype(int) - border).max(1) > 40]
        cells.append((code, sh.dominant(found) if len(found) >= 6 else found[:0]))
    return cells


def main():
    (OUT / "maps").mkdir(parents=True, exist_ok=True)
    (OUT / "sheets").mkdir(exist_ok=True)
    links = json.loads(Path("data/cma_tables/links.json").read_text())
    legends = json.loads(LEGENDS.read_text())
    fits = {Path(r["name"]).stem: r for r in json.loads(Path("outputs/real-fit/fits.json").read_text())}
    originals = lt.originals()
    drawer = sh.Sheets("data/region_probe")
    with open(OUT / "records.jsonl", "w") as out:
        for name, box in legends.items():
            row = fits[name]
            image = Image.open(fr.REAL / row["name"]).convert("RGB")
            big = Image.open(originals[name]).convert("RGB")
            factor = big.width / row["size"][0]
            legend = big.crop(box)
            legend.thumbnail((image.width // 3, image.height // 2))  # about the size the model saw in training
            regions = fr.TEMPLATES[row["projection"]][0]
            codes = sorted(set(links[name]["values"]) & set(regions))
            fit = np.asarray(row["fit"]) * factor
            sheet = drawer.draw(region_cells(np.asarray(big), regions, fit, codes), legend)
            shutil.copy(fr.REAL / row["name"], OUT / "maps" / row["name"])
            sheet.save(OUT / "sheets" / f"{name}.png")
            out.write(json.dumps({"id": name, "map": f"maps/{row['name']}", "sheet": f"sheets/{name}.png"}) + "\n")
    print(len(legends), "sheets")


if __name__ == "__main__":
    main()
