#!/usr/bin/env python3
"""Does Qwen read India better when the region head marks the map for it?

The same India test maps, drawn with no region codes, are read five ways by the same model:
- plain:     as they are
- true_big:  codes on the 27 big regions at their true positions; the limit for marking
- head_big:  the same codes where the region head puts them
- true_all:  codes on the small regions too, which the model never saw in training
- head_all:  the same, where the head puts them

Passes if head_big is within 5 points of true_big on big regions, and clearly above plain. true_all against plain
says whether the small regions fail from not being found, or from having too few pixels.

    python kaggle/launch.py read --tag probe --synth-dataset none --india-dataset sartma/synth-india-choropleth \
        --extra-dataset sartma/region-probe --from-run sartma/mapqa-derender-qwen \
        --from-run sartma/mapqa-derender-region-probe --no-wait -- --adapter /kaggle/input
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from matplotlib import font_manager
from peft import PeftModel
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from common import evaluate_and_save, find_images_root, find_saved, read_split, sample_records
from region_head import SMALL, GridHead, peaks
from train_qwen import map_tokens, predict, vision_tower

CODE_PIXELS = 10  # the generator's codes are 5.5-8 points at 100 dots per inch


def font():
    return ImageFont.truetype(font_manager.findfont("DejaVu Sans"), CODE_PIXELS)


def true_points(raster, codes):
    """{code: (x, y)}, as shares of the drawing: the point deepest inside each region."""
    points = {}
    for k, code in enumerate(codes, 1):
        rows, columns = np.where(raster == k)
        top, bottom, left, right = rows.min(), rows.max() + 1, columns.min(), columns.max() + 1
        depth = ndimage.distance_transform_edt(np.pad(raster[top: bottom, left: right] == k, 1))
        y, x = np.unravel_index(np.argmax(depth), depth.shape)
        points[code] = ((left + x - 1 + 0.5) / raster.shape[1], (top + y - 1 + 0.5) / raster.shape[0])
    return points


def mark(image, spots, typeface):
    """A copy of the map with each code centred on its spot, black or white against the colour there, as drawn maps do."""
    marked = image.copy()
    draw = ImageDraw.Draw(marked)
    for code, (x, y) in spots.items():
        r, g, b = image.getpixel((min(image.width - 1, int(x)), min(image.height - 1, int(y))))
        ink = "black" if 0.299 * r + 0.587 * g + 0.114 * b > 127 else "white"
        draw.text((x, y), code, fill=ink, font=typeface, anchor="mm")
    return marked


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, help="unused; the launcher always passes it")
    parser.add_argument("--mapqa", type=Path, help="unused; the launcher always passes it")
    parser.add_argument("--india", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=Path("/kaggle/input"), help="holds regions.json and the head")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--max-pixels", type=int, default=640000)
    parser.add_argument("--maps", type=int, default=100)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=600)
    parser.add_argument("--tile-zoom", type=float, default=3.0, help="unused; predict reads it")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    folder = next(args.labels.rglob("regions.json")).parent
    meta = json.loads((folder / "regions.json").read_text())
    raster = np.asarray(Image.open(folder / "regions.png"))
    codes, listed = meta["codes"], meta["listed"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.float16 if device.type == "cuda" else None
    processor = AutoProcessor.from_pretrained(args.model, max_pixels=args.max_pixels)
    processor.tokenizer.padding_side = "left"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.float16,
                                                               attn_implementation="sdpa").to(device)
    model = PeftModel.from_pretrained(model, find_saved(args.adapter, "adapter_config.json")).eval()
    head = GridHead(vision_tower(model).config.out_hidden_size, len(codes) + 1).to(device)
    head.load_state_dict(torch.load(next(args.labels.rglob("picker_region_grid.pt")), map_location=device))
    head.eval()

    root = find_images_root(args.india, "synthind_*")
    plain = [r for r in read_split(root, "test", root) if not r["style"]["state_labels"]]
    records = sample_records(plain, args.maps, seed=0)
    drawing = true_points(raster, codes)
    typeface = font()
    big = [c for c in listed if c not in SMALL]
    versions = {name: [] for name in ("true_big", "head_big", "true_all", "head_all")}
    folders = {name: args.out / "marked" / name for name in versions}
    for path in folders.values():
        path.mkdir(parents=True)
    for record in records:
        image = Image.open(record["image_path"]).convert("RGB")
        left, top, right, bottom = meta["placed"][record["id"]]
        true_spots = {c: (left + x * (right - left), top + y * (bottom - top)) for c, (x, y) in drawing.items()}
        tokens, shape = map_tokens(model, processor, image, device, amp_dtype)
        found = peaks(head, tokens, shape, codes, device)
        head_spots = {c: (v[0] * image.width, v[1] * image.height) for c, v in found.items()}
        for name, spots, chosen in (("true_big", true_spots, big), ("head_big", head_spots, big),
                                    ("true_all", true_spots, listed), ("head_all", head_spots, listed)):
            path = folders[name] / f"{record['id']}.png"
            mark(image, {c: spots[c] for c in chosen}, typeface).save(path)
            versions[name].append({**record, "image_path": str(path)})

    evaluate_and_save(records, predict(model, processor, records, args, device, amp_dtype), args.out, "plain")
    for name, marked in versions.items():
        print(f"reading {name}", flush=True)
        evaluate_and_save(marked, predict(model, processor, marked, args, device, amp_dtype), args.out, name)
    keep = args.out / "samples"
    keep.mkdir()
    for name, path in folders.items():
        for sample in sorted(path.glob("*.png"))[:3]:
            sample.rename(keep / f"picker_{name}_{sample.name}")


if __name__ == "__main__":
    main()
