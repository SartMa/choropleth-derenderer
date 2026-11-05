#!/usr/bin/env python3
"""Can a small head on Qwen's vision tokens say which region lies under each token?

If it can, the model could find a country's regions itself and mark them on the map, which printed codes showed is
all it needs to read them (big Indian states went from 64% to 97%). Nothing here touches the language model.

Three heads learn the same labels from the same maps, with the rest of the model frozen:
- token:    each token judged alone, like the tile picker
- grid:     each token judged with its neighbours
- position: sees only where a token sits on the page; the control. A head that cannot beat it has learned no geography

Synthetic India maps are shrunk, stretched and moved at random, so that a map's position alone teaches little.
Labels come from the drawing itself: the share of each token's square each region covers.

Passes if, on moved test maps, a head finds at least 90% of big states and clearly beats the control, and finds at
least 80% on real maps. Found: the region's best token is mostly that region. On real maps, whose only labels are
CMA's Mask R-CNN boxes: the token's centre is inside the box.

    python kaggle/launch.py region --tag probe --synth-dataset none --india-dataset sartma/synth-india-choropleth \
        --extra-dataset sartma/region-probe --from-run sartma/mapqa-derender-qwen --no-wait -- --adapter /kaggle/input
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from common import find_images_root, find_saved, read_split, sample_records
from train_qwen import map_tokens, vision_tower

SMALL = {"AN", "DL", "GA", "SK", "TR"}  # listed regions under about 1,000 pixels in the generator's maps
SAMPLES = 8  # points per token side when measuring which regions a token covers


def move(image, rng):
    """The whole image shrunk, stretched a little and placed at random on a page of its own colour."""
    scale_x = float(rng.uniform(0.6, 1.0))
    scale_y = scale_x * float(rng.uniform(0.9, 1.1))
    scale_y = min(scale_y, 1.0)
    width, height = max(1, int(image.width * scale_x)), max(1, int(image.height * scale_y))
    x, y = int(rng.integers(0, image.width - width + 1)), int(rng.integers(0, image.height - height + 1))
    page = Image.new("RGB", image.size, image.getpixel((1, 1)))
    page.paste(image.resize((width, height), Image.BILINEAR), (x, y))
    return page, (width / image.width, height / image.height, x, y)


def token_shares(raster, placed, size, rows, columns, moved=(1.0, 1.0, 0, 0)):
    """For each token, the share of its square each region covers; column 0 is everything that is not a region."""
    width, height = size
    scale_x, scale_y, dx, dy = moved
    left, top, right, bottom = placed
    xs = ((np.arange(columns * SAMPLES) + 0.5) / (columns * SAMPLES) * width - dx) / scale_x
    ys = ((np.arange(rows * SAMPLES) + 0.5) / (rows * SAMPLES) * height - dy) / scale_y
    across = np.floor((xs - left) / (right - left) * raster.shape[1]).astype(int)
    down = np.floor((ys - top) / (bottom - top) * raster.shape[0]).astype(int)
    keep_x, keep_y = (across >= 0) & (across < raster.shape[1]), (down >= 0) & (down < raster.shape[0])
    found = np.zeros((len(ys), len(xs)), np.int64)
    found[np.ix_(keep_y, keep_x)] = raster[np.ix_(down[keep_y], across[keep_x])]
    token = (np.arange(len(ys)) // SAMPLES)[:, None] * columns + (np.arange(len(xs)) // SAMPLES)[None, :]
    regions = int(raster.max()) + 1
    counts = np.bincount((token * regions + found).ravel(), minlength=rows * columns * regions)
    return torch.from_numpy(counts.reshape(-1, regions) / SAMPLES ** 2).float()


class TokenHead(torch.nn.Module):
    def __init__(self, width, regions, hidden=256):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(width, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, regions))

    def forward(self, tokens, shape):
        return self.net(tokens)


class GridHead(torch.nn.Module):
    def __init__(self, width, regions, hidden=128):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(width, hidden, 1), torch.nn.GELU(), torch.nn.Conv2d(hidden, hidden, 3, padding=1),
            torch.nn.GELU(), torch.nn.Conv2d(hidden, hidden, 3, padding=2, dilation=2), torch.nn.GELU(),
            torch.nn.Conv2d(hidden, regions, 1))

    def forward(self, tokens, shape):
        rows, columns = shape
        grid = tokens.reshape(-1, rows, columns, tokens.shape[-1]).permute(0, 3, 1, 2)
        return self.net(grid).permute(0, 2, 3, 1).flatten(0, 2)


class PositionHead(torch.nn.Module):
    def __init__(self, width, regions, hidden=256):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(2, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, hidden),
                                       torch.nn.GELU(), torch.nn.Linear(hidden, regions))

    def forward(self, tokens, shape):
        rows, columns = shape
        down, across = torch.meshgrid(torch.arange(rows, device=tokens.device), torch.arange(columns, device=tokens.device),
                                      indexing="ij")
        where = torch.stack([(down.flatten() + 0.5) / rows, (across.flatten() + 0.5) / columns], 1)
        return self.net(where).repeat(len(tokens) // (rows * columns), 1)


HEADS = {"token": TokenHead, "grid": GridHead, "position": PositionHead}


def fit(kind, maps, width, regions, device, epochs, seed):
    """maps: (tokens, shares, shape). Batches hold up to 8 whole maps of one size, so grids stay grids."""
    torch.manual_seed(seed)
    head = HEADS[kind](width, regions).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.01)
    by_shape = {}
    for i, m in enumerate(maps):
        by_shape.setdefault(m[2], []).append(i)
    for epoch in range(epochs):
        batches = []
        for shape, members in by_shape.items():
            members = random.Random(seed + epoch).sample(members, len(members))
            batches += [(shape, members[i: i + 8]) for i in range(0, len(members), 8)]
        random.Random(seed + epoch).shuffle(batches)
        total, seen = 0.0, 0
        for shape, chosen in batches:
            tokens = torch.cat([maps[i][0] for i in chosen]).to(device).float()
            shares = torch.cat([maps[i][1] for i in chosen]).to(device)
            loss = -(shares * head(tokens, shape).log_softmax(-1)).sum(-1).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(chosen)
            seen += len(chosen)
        print(f"{kind} epoch {epoch + 1}: loss {total / seen:.4f}", flush=True)
    return head.eval()


def score_labelled(head, maps, codes, listed, device):
    """On maps with exact labels: how often each region's best token is mostly that region, or touches it."""
    mostly, touches, right, tokens_on_map = {c: [] for c in listed}, {c: [] for c in listed}, 0, 0
    for tokens, shares, shape in maps:
        with torch.no_grad():
            chances = head(tokens.to(device).float(), shape).softmax(-1).cpu()
        on_map = shares[:, 1:].sum(1) > 0.5
        right += int((chances[on_map].argmax(1) == shares[on_map].argmax(1)).sum())
        tokens_on_map += int(on_map.sum())
        for k, code in enumerate(codes, 1):
            if code in mostly and shares[:, k].sum() > 0:
                share = float(shares[int(chances[:, k].argmax()), k])
                mostly[code].append(share >= 0.5)
                touches[code].append(share > 0)
    rate = lambda table, group: float(np.mean([v for c in group for v in table[c]]))
    big = [c for c in listed if c not in SMALL]
    small = [c for c in listed if c in SMALL]
    return {"maps": len(maps), "token_accuracy": right / max(1, tokens_on_map),
            "big_found": rate(mostly, big), "big_touched": rate(touches, big),
            "small_found": rate(mostly, small), "small_touched": rate(touches, small),
            "per_region": {c: round(float(np.mean(v)), 3) for c, v in mostly.items() if v}}


def peaks(head, tokens, shape, codes, device):
    """{code: [x, y, chance]}: each region's best token, as shares of the image."""
    with torch.no_grad():
        chances = head(tokens.to(device).float(), shape).softmax(-1).cpu()
    rows, columns = shape
    found = {}
    for k, code in enumerate(codes, 1):
        best = int(chances[:, k].argmax())
        found[code] = [((best % columns) + 0.5) / columns, ((best // columns) + 0.5) / rows, float(chances[best, k])]
    return found


def score_real(found, meta):
    """Each region's best token against CMA's Mask R-CNN boxes, on the maps that have them."""
    inside = {"big": [], "small": []}
    for name, boxes in meta["real_boxes"].items():
        width, height = meta["real_sizes"][name]
        for code, (top, left, bottom, right) in boxes.items():
            if code in found[name] and code in meta["listed"]:
                x, y = found[name][code][0] * width, found[name][code][1] * height
                inside["small" if code in SMALL else "big"].append(left <= x <= right and top <= y <= bottom)
    return {"maps": len(meta["real_boxes"]), "big_found": float(np.mean(inside["big"])),
            "big_regions": len(inside["big"]), "small_found": float(np.mean(inside["small"]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, help="unused; the launcher always passes it")
    parser.add_argument("--mapqa", type=Path, help="unused; the launcher always passes it")
    parser.add_argument("--india", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=Path("/kaggle/input"), help="holds regions.json, from synth/region_labels.py")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--adapter", type=Path, help="the trained LoRA adapter whose tokens the heads read")
    parser.add_argument("--max-pixels", type=int, default=640000)
    parser.add_argument("--train-maps", type=int, default=2400)
    parser.add_argument("--test-maps", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    folder = next(args.labels.rglob("regions.json")).parent
    meta = json.loads((folder / "regions.json").read_text())
    raster = np.asarray(Image.open(folder / "regions.png"))
    codes, listed = meta["codes"], meta["listed"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.float16 if device.type == "cuda" else None
    processor = AutoProcessor.from_pretrained(args.model, max_pixels=args.max_pixels)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.float16,
                                                               attn_implementation="sdpa").to(device)
    if args.adapter:
        model = PeftModel.from_pretrained(model, find_saved(args.adapter, "adapter_config.json"))
    model.eval()
    width = vision_tower(model).config.out_hidden_size

    root = find_images_root(args.india, "synthind_*")
    train = sample_records(read_split(root, "train", root), args.train_maps, args.seed)
    test = sample_records(read_split(root, "test", root), args.test_maps, args.seed)

    def collect(records, moved, seed):
        rng, maps, began = np.random.default_rng(seed), [], time.time()
        for n, record in enumerate(records, 1):
            image = Image.open(record["image_path"]).convert("RGB")
            place = (1.0, 1.0, 0, 0)
            if moved:
                image, place = move(image, rng)
            tokens, (rows, columns) = map_tokens(model, processor, image, device, amp_dtype)
            shares = token_shares(raster, meta["placed"][record["id"]], image.size, rows, columns, place)
            maps.append((tokens.half().cpu(), shares, (rows, columns)))
            if n % 200 == 0:
                print(f"tokens: {n}/{len(records)} maps ({(time.time() - began) / 60:.1f} min)", flush=True)
        return maps

    training = collect(train, True, args.seed)
    as_drawn, moved = collect(test, False, args.seed + 1), collect(test, True, args.seed + 2)
    real_names, real = sorted(meta["real_sizes"]), []
    for name in real_names:
        image = Image.open(folder / "real" / f"{Path(name).stem}.jpg").convert("RGB")
        tokens, shape = map_tokens(model, processor, image, device, amp_dtype)
        real.append((tokens.half().cpu(), shape))

    report = {"train_maps": len(training), "tokens": sum(len(m[0]) for m in training)}
    answers = {}
    for kind in HEADS:
        head = fit(kind, training, width, len(codes) + 1, device, args.epochs, args.seed)
        torch.save(head.state_dict(), args.out / f"picker_region_{kind}.pt")
        found = {name: peaks(head, tokens, shape, codes, device) for name, (tokens, shape) in zip(real_names, real)}
        answers[kind] = found
        report[kind] = {"as_drawn": score_labelled(head, as_drawn, codes, listed, device),
                        "moved": score_labelled(head, moved, codes, listed, device),
                        "real_cma_boxes": score_real(found, meta)}
        print(kind, json.dumps({k: {a: b for a, b in v.items() if a != "per_region"} for k, v in report[kind].items()}),
              flush=True)
    (args.out / "metrics_regions.json").write_text(json.dumps(report, indent=1))
    with open(args.out / "predictions_regions.jsonl", "w", encoding="utf-8") as f:
        for name in real_names:
            f.write(json.dumps({"map": name, **{kind: answers[kind][name] for kind in HEADS}}) + "\n")


if __name__ == "__main__":
    main()
