"""Turn a map into a region sheet: one equal cell per region, labelled with its code, beside the map's legend.

A sheet asks the model only to match cells to the legend, whatever the country, so it should need no geography.
Cells are two tokens tall and line up with the token grid: the code fills the top token, and the region's own
interior pixels fill the bottom one, so a tiny region gets a clean token of its own. Regions come from the drawing,
so this tests the sheet with perfect regions; synth/region_labels.py writes what it needs.

In training each sheet shows a random set of regions under random two-letter codes. The only way to answer is to
copy each code from its cell, which is what lets a model that has only seen US maps write Indian codes.
"""

import json
import random
import string
from pathlib import Path

import cv2
import numpy as np
import torch
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

CELL = 56  # two tokens tall and wide
COLUMNS = 8
GAP = 28  # one empty token between the cells and the legend
LEGEND_BANDS = {"right": (0.75, 0.0, 1.0, 1.0), "bottom": (0.0, 0.82, 1.0, 1.0)}  # draw_map's layout, as shares
SHEET_PROMPT = (
    "The first image is a choropleth map. The second image is a sheet of its regions: each cell shows a region's code "
    "above that region's colour, beside the map's legend. Return only JSON with these keys: "
    '"title"; "legend_title" (null if there is none); "legend_type" ("continuous" or "discrete"); '
    '"data_type" ("percentage" or "number"); "ranges" (discrete legends only: the legend ranges, lowest first); '
    '"values" (every region on the sheet by the code printed in its cell, in sheet order: a number for a continuous '
    "legend, the index of the region's range for a discrete legend, null for no data)."
)


def dominant(pixels):
    """The pixels near the region's commonest colour, dropping printed text and leftover border."""
    if not len(pixels):
        return pixels
    coarse = pixels.astype(int) // 16
    colours, counts = np.unique(coarse, axis=0, return_counts=True)
    return pixels[np.abs(coarse - colours[np.argmax(counts)]).max(1) <= 1]


class Sheets:
    def __init__(self, labels):
        """labels: a folder holding regions.json / .png (India) and regions_usa.json / .png, searched a few levels."""
        folder = next(Path(labels).rglob("regions_usa.json")).parent
        self.maps = {}
        for prefix, name in (("synthind_", "regions"), ("synth_", "regions_usa")):
            self.maps[prefix] = (json.loads((folder / f"{name}.json").read_text()),
                                 np.asarray(Image.open(folder / f"{name}.png")))
        self.typeface = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), 16)

    def lookup(self, record):
        return next(v for prefix, v in self.maps.items() if record["id"].startswith(prefix))

    def region_pixels(self, image, record, codes):
        """{code: (n, 3) array of the pixels inside that region, away from its border where the region allows."""
        meta, raster = self.lookup(record)
        left, top, right, bottom = meta["placed"][record["id"]]
        place = np.array([[(right - left) / raster.shape[1], 0, left], [0, (bottom - top) / raster.shape[0], top]])
        drawn = cv2.warpAffine(raster, place, image.size, flags=cv2.INTER_NEAREST)
        pixels, found = np.asarray(image), {}
        for code in codes:
            mask = (drawn == meta["codes"].index(code) + 1).astype(np.uint8)
            chosen = mask
            for size in (3, 5):
                inner = cv2.erode(mask, np.ones((size, size), np.uint8))
                if inner.sum() >= 6:
                    chosen = inner
            found[code] = dominant(pixels[chosen > 0])
        return found

    def legend(self, image, record):
        """The legend, cut from the band the generator keeps for it and trimmed to its contents."""
        x0, y0, x1, y1 = LEGEND_BANDS[record["style"]["legend_pos"]]
        band = image.crop((int(x0 * image.width), int(y0 * image.height), int(x1 * image.width), int(y1 * image.height)))
        pixels = np.asarray(band).astype(int)
        ink = np.abs(pixels - pixels[0, 0]).sum(2) > 30
        rows, columns = np.where(ink)
        return band.crop((columns.min(), rows.min(), columns.max() + 1, rows.max() + 1)) if len(rows) else band

    def draw(self, cells, legend):
        """cells: (code, pixels) in sheet order. Sides are multiples of the token, so nothing is resampled."""
        rows = int(np.ceil(len(cells) / COLUMNS))
        grid_height = rows * CELL
        width = COLUMNS * CELL + GAP + legend.width
        height = max(grid_height, legend.height)
        sheet = Image.new("RGB", (int(np.ceil(width / 28)) * 28, int(np.ceil(height / 28)) * 28), "white")
        draw = ImageDraw.Draw(sheet)
        half = CELL // 2
        for n, (code, pixels) in enumerate(cells):
            x, y = (n % COLUMNS) * CELL, (n // COLUMNS) * CELL
            draw.text((x + half, y + half // 2), code, fill="black", font=self.typeface, anchor="mm")
            fill = np.resize(pixels.astype(np.uint8), (half, CELL, 3)) if len(pixels) else np.full((half, CELL, 3), 255, np.uint8)
            sheet.paste(Image.fromarray(fill), (x, y + half))
        sheet.paste(legend, (COLUMNS * CELL + GAP, 0))
        return sheet

    def scored(self, image, record):
        """A test map's sheet: every listed region under its real code. The target is unchanged."""
        codes = sorted(json.loads(record["target"])["values"])
        found = self.region_pixels(image, record, codes)
        return self.draw([(c, found[c]) for c in codes], self.legend(image, record))

    def taught(self, image, record, rng):
        """A training map's sheet and target: a random set of regions under random codes, in code order."""
        target = json.loads(record["target"])
        real = list(target["values"])
        chosen = rng.sample(real, rng.randint(16, len(real)))
        names = rng.sample([a + b for a in string.ascii_uppercase for b in string.ascii_uppercase], len(chosen))
        renamed = dict(sorted(zip(names, chosen)))
        found = self.region_pixels(image, record, chosen)
        sheet = self.draw([(name, found[code]) for name, code in renamed.items()], self.legend(image, record))
        target["values"] = {name: target["values"][code] for name, code in renamed.items()}
        return sheet, json.dumps(target, ensure_ascii=False, separators=(",", ":"))


class SheetDataset(torch.utils.data.Dataset):
    """(images, boxes, target, fuse, prompt), as CropDataset gives TrainCollate."""

    def __init__(self, records, sheets):
        self.records, self.sheets = records, sheets

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        record = self.records[i]
        image = Image.open(record["image_path"]).convert("RGB")
        sheet, target = self.sheets.taught(image, record, random.Random())
        return [image, sheet], None, target, False, SHEET_PROMPT
