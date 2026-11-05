#!/usr/bin/env python3
"""Answer sheet v3: find regions on an S2 (2x) patch grid, and zoom each small region by as much as it needs.

v2 (train/answer_v2.py, outputs/mapqa-derender-v2-zoom/SUMMARY.md) zoomed small regions 3x and fixed many of them,
but its own records show three gaps:
- finding: the head picks where to zoom from the 1x grid, where Delhi is a fifth of a patch. Below a 5% peak it
  skips the region, which happened for Delhi on 58 of 68 real maps and Rhode Island on 169 of 400 MapQA maps
- too little zoom: at 3x, Rhode Island, Delhi and Andaman tokens are still only 56-69% their own pixels
- zooming what is not small: on real maps medium states (Nagaland, Punjab, Himachal) got zoomed and their tokens
  got worse (Himachal 0.74 -> 0.23); about 10 swapped tokens per map is the likely cause of whole-map class shifts

v3, two modes (--mode):
- s2: "scaling on scales" as region tokens. The map is enlarged 2x, cut into 2x2 tiles of the map's own size, each
  run through the tower, and the patch grids stitched into one grid at twice the detail (Delhi: ~0.8 patch). The
  head reads that grid and every region's token is pooled from it. No zoom. Unlike the S2 tried on 2026-09-28
  (train_qwen.py --s2), nothing is averaged back onto the coarse grid and no new weights sit between tower and LLM:
  pooling is per region and goes through Qwen's own merger.
- zoom (default): s2, plus every region the head finds less than `--small-mass` (1x) patches of is re-read in a
  view zoomed so the region covers about `--target-patches` patches (2x-8x). The zoomed token replaces the S2 token
  only when the head is more confident about the region in the view (mean belief over the pooled patches).

The language model still sees the 1x map tokens; only the region tokens change. Continues from v2's weights.

    KAGGLE_API_TOKEN=... python kaggle/launch.py v3 --owner skini501 --tag zoom \\
        --india-dataset sartma/synth-india-choropleth --extra-dataset skini501/answer-v2-inputs \\
        --extra-dataset skini501/real-sheets --from-run skini501/mapqa-derender-v2-zoom --no-wait \\
        -- --adapter /kaggle/input --prefer v2-zoom --mode zoom
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

import answer_sheet as A
import answer_v2 as Z
from common import (add_common_args, autocast, evaluate_and_save, find_images_root, read_split, sample_records,
                    set_seed, train_loop)
from train_qwen import report_lora

SCALE = 2


# ---------------------------------------------------------------- S2 grid

def s2_tiles(image, scale=SCALE):
    """The image enlarged `scale` times and cut into scale x scale tiles of the image's own size, row by row."""
    big = image.resize((image.width * scale, image.height * scale), Image.BICUBIC)
    return [big.crop((c * image.width, r * image.height, (c + 1) * image.width, (r + 1) * image.height))
            for r in range(scale) for c in range(scale)]


def stitch(grids, scale=SCALE):
    """scale * scale tile grids (rows, columns, width), row by row -> one grid."""
    return torch.cat([torch.cat(grids[r * scale:(r + 1) * scale], dim=1) for r in range(scale)], dim=0)


def fine_grid(wrapper, processor, image, device, amp_dtype):
    seen = processor.image_processor(images=s2_tiles(image), return_tensors="pt")
    with torch.no_grad(), autocast(device, amp_dtype):
        _, grids = wrapper.patch_grids(seen["pixel_values"].to(device).type(wrapper.visual.dtype),
                                       seen["image_grid_thw"].to(device))
    return stitch(grids)


def belief(chances, k, sharpen):
    """The head's mean belief in class k over the patches its token would be pooled from."""
    p = chances[:, k]
    w = p ** sharpen
    return float((w * p).sum() / w.sum().clamp_min(1e-6))


def zoom_for(area, args):
    """Zoom so a region of `area` (1x) patches covers about `--target-patches` patches."""
    return float(np.clip(math.sqrt(args.target_patches / max(area, 0.02)), args.min_zoom, args.max_zoom))


def view_around(image, label, cx, cy, zoom, args):
    """A view zoomed `zoom` times around (cx, cy); its side shrinks as zoom grows so views stay one size."""
    side_share = args.view_extent / zoom
    return Z.zoom_view(image, label, Z.zoom_box(image.size, cx, cy, side_share), zoom)


# ---------------------------------------------------------------- the model

class S2Visual(A.AnswerSheetVisual):
    """Region tokens from the S2 grid (`pending["fine"]`), and from zoomed views where `pending["views"]` gives
    them. The 1x map still goes to the language model as before."""

    def forward(self, hidden_states, grid_thw, **kwargs):
        pending = self.pending
        losses = []
        if pending:
            device = hidden_states.device
            override = pending.setdefault("override", {})
            views = pending.get("views")
            if views:
                with torch.no_grad():
                    _, grids = self.patch_grids(views["pixel_values"].to(device).type(hidden_states.dtype),
                                                views["grid_thw"].to(device))
                for (m, j, k), grid, shares in zip(views["owner"], grids, views["labels"]):
                    logits = self.head(grid)
                    if shares is not None:
                        losses.append(-(shares.to(device) * logits.float().log_softmax(-1)).sum(-1).mean())
                    weights = logits.float().softmax(-1).detach()[:, k:k + 1] ** self.sharpen
                    override.setdefault(m, {})[j] = self.pool(grid, weights)[0]
            fine = pending.get("fine")
            if fine:
                with torch.no_grad():
                    _, grids = self.patch_grids(fine["pixel_values"].to(device).type(hidden_states.dtype),
                                                fine["grid_thw"].to(device))
                for m, classes in enumerate(pending["classes"]):
                    grid = stitch(grids[m * SCALE ** 2:(m + 1) * SCALE ** 2])
                    logits = self.head(grid)
                    shares = fine["labels"][m]
                    if shares is not None:
                        losses.append(-(shares.to(device) * logits.float().log_softmax(-1)).sum(-1).mean())
                    regions = self.region_tokens(grid, logits, classes)
                    for j in range(len(classes)):
                        override.setdefault(m, {}).setdefault(j, regions[j])  # zoomed tokens win
        out = super().forward(hidden_states, grid_thw, **kwargs)
        if losses:
            self.aux_loss = torch.stack(losses + ([self.aux_loss] if self.aux_loss is not None else [])).mean()
        return out


def attach(model, classes, device):
    for module in model.modules():
        if "visual" in module._modules and not isinstance(module._modules["visual"], A.AnswerSheetVisual):
            wrapper = S2Visual(module._modules["visual"], classes)
            wrapper.head.to(device=device, dtype=torch.float32)
            module._modules["visual"] = wrapper
            return wrapper
    raise SystemExit("could not find the vision tower")


# ---------------------------------------------------------------- training inputs

def true_centre(label, k):
    ys, xs = np.nonzero(label == k)
    return xs.mean(), ys.mean()


class TrainSet(A.TrainSet):
    """v1's examples, plus S2 tiles for every map and (mode zoom) a sized view for every small shown region."""

    def __init__(self, records, labels, args):
        super().__init__(records, labels)
        self.args = args

    def __getitem__(self, i):
        image, label, names, classes, target = super().__getitem__(i)
        views = []
        if label is not None and self.args.mode == "zoom":
            areas = Z.label_areas(label, self.args.max_pixels, len(self.labels.vocab))
            rng = random.Random()
            small = [(j, k) for j, k in enumerate(classes) if 0 < areas[k] < self.args.small_mass]
            for j, k in rng.sample(small, min(len(small), self.args.max_views)):
                zoom = float(np.clip(zoom_for(areas[k], self.args) * rng.uniform(0.8, 1.25),
                                     self.args.min_zoom, self.args.max_zoom))
                cx, cy = true_centre(label, k)
                side = self.args.view_extent / zoom * max(image.size)
                cx += rng.uniform(-self.args.jitter, self.args.jitter) * side
                cy += rng.uniform(-self.args.jitter, self.args.jitter) * side
                view, crop = view_around(image, label, cx, cy, zoom, self.args)
                views.append((j, k, view, crop))
        return image, label, names, classes, target, views


class Collate(Z.Collate):
    def __call__(self, batch):
        enc, extra = super().__call__(batch)
        tiles = [tile for item in batch for tile in s2_tiles(item[0])]
        seen = self.processor.image_processor(images=tiles, return_tensors="pt")
        thw = seen["image_grid_thw"].tolist()
        shares = []
        for m, item in enumerate(batch):
            _, h, w = thw[m * SCALE ** 2]
            shares.append(None if item[1] is None else A.patch_shares(item[1], h * SCALE, w * SCALE, self.classes))
        extra["fine"] = {"pixel_values": seen["pixel_values"], "grid_thw": seen["image_grid_thw"], "labels": shares}
        return enc, extra


# ---------------------------------------------------------------- stage A

def teach_head(wrapper, processor, records, labels, args, device, amp_dtype):
    """The head alone on frozen features: S2 grids of whole maps (half moved) and, `--zoom-share` of the time in
    mode zoom, sized views around a small region."""
    head = wrapper.head
    optimizer = torch.optim.AdamW(head.parameters(), lr=5e-4, weight_decay=0.01)
    deadline, began, step, window = time.time() + args.head_minutes * 60, time.time(), 0, {"s2": [], "view": []}
    rng = random.Random(args.seed)
    zoom_share = args.zoom_share if args.mode == "zoom" else 0.0
    head.train()
    while time.time() < deadline:
        for record in rng.sample(records, len(records)):
            image = Image.open(record["image_path"]).convert("RGB")
            label = labels.label(record, image.size)
            if rng.random() < zoom_share:
                areas = Z.label_areas(label, args.max_pixels, len(labels.vocab))
                present = [k for k in range(1, len(labels.vocab)) if areas[k] > 0]
                small = [k for k in present if areas[k] < args.small_mass]
                k = rng.choice(small or present)
                zoom = float(np.clip(zoom_for(areas[k], args) * rng.uniform(0.8, 1.25), args.min_zoom, args.max_zoom))
                cx, cy = true_centre(label, k)
                side = args.view_extent / zoom * max(image.size)
                view, crop = view_around(image, label, cx + rng.uniform(-args.jitter, args.jitter) * side,
                                         cy + rng.uniform(-args.jitter, args.jitter) * side, zoom, args)
                grid, target, kind = A.map_grid(wrapper, processor, view, device, amp_dtype), crop, "view"
            else:
                if rng.random() < 0.5:
                    image, label = A.move(image, label, rng)
                grid, target, kind = fine_grid(wrapper, processor, image, device, amp_dtype), label, "s2"
            shares = A.patch_shares(target, grid.shape[0], grid.shape[1], len(labels.vocab)).to(device)
            loss = -(shares * head(grid).log_softmax(-1)).sum(-1).mean()
            (loss / 4).backward()
            window[kind].append(loss.item())
            step += 1
            if step % 4 == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if step % 200 == 0:
                print(json.dumps({"head_step": step, **{f"loss_{n}": round(float(np.mean(v)), 4)
                                                         for n, v in window.items() if v},
                                  "minutes": round((time.time() - began) / 60, 1)}), flush=True)
                window = {"s2": [], "view": []}
            if time.time() > deadline:
                break
    head.eval()
    print(f"head taught on {step} images", flush=True)


# ---------------------------------------------------------------- head report: finding and purity, 1x against S2

def head_report(wrapper, processor, sets, labels, args, device, amp_dtype):
    """Per set: country right, patch accuracy, token purity on the S2 grid, and for each small region how often
    the head finds it at all (peak >= --min-peak) on the 1x grid against the S2 grid."""
    report = {}
    for name, records in sets:
        right_country, patch_right, patches = 0, 0, 0
        purity = {"big": [], "small": []}
        found = {}
        for record in records:
            image = Image.open(record["image_path"]).convert("RGB")
            label = labels.label(record, image.size)
            if label is None:
                continue
            country = A.Labels.country(record)
            coarse = A.map_grid(wrapper, processor, image, device, amp_dtype)
            grid = fine_grid(wrapper, processor, image, device, amp_dtype)
            with torch.no_grad():
                chances = wrapper.head(grid).float().softmax(-1).cpu()
                coarse_chances = wrapper.head(coarse).float().softmax(-1).cpu()
            shares = A.patch_shares(label, grid.shape[0], grid.shape[1], len(labels.vocab))
            right_country += A.country_of(chances, labels.vocab) == country
            on_map = shares[:, 1:].sum(1) > 0.5
            patch_right += int((chances[on_map].argmax(1) == shares[on_map].argmax(1)).sum())
            patches += int(on_map.sum())
            for code in labels.listed[country]:
                k = labels.index[f"{country}-{code}"]
                if shares[:, k].sum() == 0:
                    continue
                weights = chances[:, k] ** wrapper.sharpen
                weights = weights / weights.sum().clamp_min(1e-6)
                small = code in A.SMALL[country]
                purity["small" if small else "big"].append(float((weights * shares[:, k]).sum()))
                if small:
                    hit = found.setdefault(code, {"maps": 0, "found_1x": 0, "found_s2": 0})
                    hit["maps"] += 1
                    hit["found_1x"] += float(coarse_chances[:, k].max()) >= args.min_peak
                    hit["found_s2"] += float(chances[:, k].max()) >= args.min_peak
        report[name] = {"maps": len(records), "country_right": right_country / max(1, len(records)),
                        "patch_accuracy": patch_right / max(1, patches),
                        **{f"{size}_purity_median": float(np.median(v)) if v else None for size, v in purity.items()},
                        "found": found}
        print(name, json.dumps(report[name]), flush=True)
    return report


# ---------------------------------------------------------------- reading

def region_overrides(wrapper, processor, image, label, grid, chances, country, codes, labels, args, device,
                     amp_dtype):
    """{position in codes: token} for every region (S2 grid, or a zoomed view where that helps), and per small
    region what happened, with purity where labels exist."""
    rows, columns = grid.shape[:2]
    down, across = torch.meshgrid(torch.arange(rows, device=device), torch.arange(columns, device=device), indexing="ij")
    down, across = down.flatten().float(), across.flatten().float()
    shares = None if label is None else A.patch_shares(label, rows, columns, len(labels.vocab)).to(device)
    classes = [labels.index[f"{country}-{c}"] for c in codes]
    with torch.no_grad(), autocast(device, amp_dtype):
        tokens = wrapper.pool(grid, chances[:, classes] ** wrapper.sharpen)
    overrides = {j: tokens[j] for j in range(len(codes))}
    stats = []
    if args.mode != "zoom":
        return overrides, stats
    for j, (code, k) in enumerate(zip(codes, classes)):
        p = chances[:, k]
        area = float(p.sum()) / SCALE ** 2
        if area >= args.small_mass:
            continue
        stat = {"code": code, "found": float(p.max()) >= args.min_peak, "kept_zoom": False}
        if stat["found"]:
            w = p ** 4
            cx = float((w * (across + 0.5)).sum() / w.sum()) / columns * image.width
            cy = float((w * (down + 0.5)).sum() / w.sum()) / rows * image.height
            zoom = zoom_for(area, args)
            view, crop = view_around(image, label, cx, cy, zoom, args)
            view_grid = A.map_grid(wrapper, processor, view, device, amp_dtype)
            with torch.no_grad():
                view_chances = wrapper.head(view_grid).float().softmax(-1)
            stat.update(zoom=round(zoom, 2), belief_s2=belief(chances, k, wrapper.sharpen),
                        belief_zoom=belief(view_chances, k, wrapper.sharpen))
            if stat["belief_zoom"] > stat["belief_s2"]:
                with torch.no_grad(), autocast(device, amp_dtype):
                    overrides[j] = wrapper.pool(view_grid, view_chances[:, k:k + 1] ** wrapper.sharpen)[0]
                stat["kept_zoom"] = True
            if crop is not None:
                view_shares = A.patch_shares(crop, view_grid.shape[0], view_grid.shape[1], len(labels.vocab))
                weights = view_chances[:, k] ** wrapper.sharpen
                stat["purity_zoom"] = float((weights / weights.sum().clamp_min(1e-6) * view_shares[:, k].to(device)).sum())
        if shares is not None and shares[:, k].sum() > 0:
            weights = p ** wrapper.sharpen
            stat["purity_s2"] = float((weights / weights.sum().clamp_min(1e-6) * shares[:, k]).sum())
        stats.append(stat)
    return overrides, stats


def predict(model, wrapper, processor, records, labels, args, device, amp_dtype):
    answers, countries, stats, began = [], [], [], time.time()
    for start in range(0, len(records), args.eval_batch_size):
        batch = records[start: start + args.eval_batch_size]
        texts, images, classes, overrides = [], [], [], {}
        for m, record in enumerate(batch):
            image = Image.open(record["image_path"]).convert("RGB")
            grid = fine_grid(wrapper, processor, image, device, amp_dtype)
            with torch.no_grad():
                chances = wrapper.head(grid).float().softmax(-1)
            country = A.country_of(chances.cpu(), labels.vocab)
            codes = labels.listed[country]
            label = labels.label(record, image.size)
            overrides[m], found = region_overrides(wrapper, processor, image, label, grid, chances, country, codes,
                                                   labels, args, device, amp_dtype)
            stats += [{"id": record["id"], **s} for s in found]
            countries.append(country)
            texts.append(A.chat(processor, codes))
            images += [image] + [A.PLACEHOLDER] * len(codes)
            classes.append([labels.index[f"{country}-{c}"] for c in codes])
        enc = processor(text=texts, images=images, padding=True, return_tensors="pt").to(device)
        wrapper.pending = {"classes": classes, "labels": [None] * len(batch), "override": overrides}
        with torch.no_grad(), autocast(device, amp_dtype):
            ids = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True)
        answers += processor.batch_decode(ids[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"predicted {len(answers)}/{len(records)} ({(time.time() - began) / 60:.1f} min)", flush=True)
    return answers, countries, stats


def summarise(stats):
    by = {}
    for s in stats:
        by.setdefault(s["code"], []).append(s)

    def median(v, key):
        v = [s[key] for s in v if s.get(key) is not None]
        return float(np.median(v)) if v else None

    return {code: {"small_on_map": len(v), "found": sum(s["found"] for s in v),
                   "kept_zoom": sum(s["kept_zoom"] for s in v), "zoom_median": median(v, "zoom"),
                   "purity_s2": median(v, "purity_s2"), "purity_zoom": median(v, "purity_zoom")}
            for code, v in sorted(by.items(), key=lambda kv: -len(kv[1]))}


# ---------------------------------------------------------------- main

def pick(root, marker, prefer):
    """The one folder under root holding `marker` whose path contains `prefer`."""
    hits = [p for p in Path(root).rglob(marker) if prefer in str(p)]
    if len(hits) != 1:
        raise SystemExit(f"want one {marker} under {root} with '{prefer}' in its path, found {[str(h) for h in hits]}")
    return hits[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--labels", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--real", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--adapter", required=True, help="holds the adapter/ and region_head.pt to continue from")
    parser.add_argument("--prefer", default="", help="pick the adapter and head whose path contains this")
    parser.add_argument("--mode", choices=["s2", "zoom"], default="zoom")
    parser.add_argument("--max-pixels", type=int, default=640_000)
    parser.add_argument("--small-mass", type=float, default=3.0, help="zoom regions smaller than this many 1x patches")
    parser.add_argument("--target-patches", type=float, default=8.0, help="zoom so a small region covers this many")
    parser.add_argument("--min-zoom", type=float, default=2.0)
    parser.add_argument("--max-zoom", type=float, default=8.0)
    parser.add_argument("--view-extent", type=float, default=0.36,
                        help="a view's side after zooming, as a share of the map's longer side (v2: 0.12 x 3)")
    parser.add_argument("--min-peak", type=float, default=0.02, help="in reading, skip regions the head barely sees")
    parser.add_argument("--jitter", type=float, default=0.15)
    parser.add_argument("--max-views", type=int, default=10)
    parser.add_argument("--zoom-share", type=float, default=0.4)
    parser.add_argument("--head-minutes", type=float, default=30)
    parser.add_argument("--head-maps", type=int, default=2000)
    parser.add_argument("--report-maps", type=int, default=100)
    parser.add_argument("--aux-weight", type=float, default=0.5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.set_defaults(epochs=1, hours=3.75, lr=1e-4, batch_size=1, grad_accum=16, eval_batch_size=8,
                        max_new_tokens=600, india_eval_maps=300, synth_train_maps=2000, india_train_maps=2000)
    args = parser.parse_args()
    set_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "args.json").write_text(json.dumps(vars(args), default=str, indent=1))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if device.type == "cpu" else (
        torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16)
    amp_dtype = None if device.type == "cpu" else dtype
    labels = A.Labels(args.labels)
    processor = AutoProcessor.from_pretrained(args.model, max_pixels=args.max_pixels, min_pixels=28 * 28)
    processor.tokenizer.padding_side = "left"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model, torch_dtype=dtype,
                                                               attn_implementation="sdpa").to(device)
    adapter = pick(args.adapter, "adapter_config.json", args.prefer).parent
    head_file = pick(args.adapter, "region_head.pt", args.prefer)
    model = A.load_adapter(model, adapter, trainable=True)
    report_lora(model)
    wrapper = attach(model, len(labels.vocab), device)
    wrapper.head.load_state_dict(torch.load(head_file, map_location=device))
    print(f"mode {args.mode}; continuing from {adapter} and {head_file}", flush=True)

    mapqa_root = find_images_root(args.mapqa, "map_*.png")
    usa_root = find_images_root(args.synth, "synth_*")
    india_root = find_images_root(args.india, "synthind_*")
    labelled_ids = set(json.loads(next(args.labels.rglob("mapqa_train_ids.json")).read_text()))
    mapqa_train = [r for r in read_split(args.data, "train", mapqa_root) if r["id"] in labelled_ids]
    mapqa_train = sample_records(mapqa_train, args.train_maps, args.seed + 11)  # a draw unlike v1's and v2's
    usa_train = sample_records(read_split(usa_root, "train", usa_root), args.synth_train_maps, args.seed + 11)
    india_train = sample_records(read_split(india_root, "train", india_root), args.india_train_maps, args.seed + 11)
    print(f"training maps: {len(mapqa_train)} MapQA + {len(usa_train)} US + {len(india_train)} India", flush=True)

    mapqa_test = sample_records(read_split(args.data, "test", mapqa_root), args.eval_maps, seed=0)
    india_test = sample_records(read_split(india_root, "test", india_root), args.india_eval_maps, seed=0)
    real_probe = [{"id": p.stem, "image_path": str(p)} for p in sorted(labels.real.glob("*.jpg"))
                  if (labels.folder / "real" / f"{p.stem}.png").exists()]
    if args.eval_limit:
        mapqa_test, india_test, real_probe = (x[: args.eval_limit] for x in (mapqa_test, india_test, real_probe))
    head_sets = [("india", india_test[: args.report_maps]), ("mapqa", mapqa_test[: args.report_maps]),
                 ("real_india", real_probe)]
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    report = {"start": head_report(wrapper, processor, head_sets, labels, args, device, amp_dtype)}
    (args.out / "metrics_head.json").write_text(json.dumps(report, indent=1))

    # stage A
    wrapper.head.requires_grad_(True)
    teach = (sample_records(mapqa_train, args.head_maps, 1) + sample_records(usa_train, args.head_maps, 1)
             + sample_records(india_train, args.head_maps, 1))
    teach_head(wrapper, processor, teach, labels, args, device, amp_dtype)
    torch.save(wrapper.head.state_dict(), args.out / "region_head_stage_a.pt")
    report["stage_a"] = head_report(wrapper, processor, head_sets, labels, args, device, amp_dtype)
    (args.out / "metrics_head.json").write_text(json.dumps(report, indent=1))

    # stage B
    lora = [p for n, p in model.named_parameters() if "lora_" in n]
    for p in lora:
        p.requires_grad_(True)
    model.train()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    loader = DataLoader(TrainSet(mapqa_train + usa_train + india_train, labels, args), batch_size=args.batch_size,
                        shuffle=True, num_workers=args.workers, collate_fn=Collate(processor, len(labels.vocab)))

    def forward(batch):
        enc, extra = batch
        wrapper.pending = extra
        loss = model(**enc.to(device)).loss
        if wrapper.aux_loss is not None:
            loss = loss + args.aux_weight * wrapper.aux_loss
        return loss

    groups = [{"params": lora}, {"params": list(wrapper.head.parameters()), "lr": args.head_lr}]
    print(f"{len(loader)} batches per epoch", flush=True)
    train_loop(model, loader, args, forward, args.out, amp_dtype, groups)
    A.save_adapter(model, args.out / "adapter")
    torch.save(wrapper.head.state_dict(), args.out / "region_head.pt")

    model.eval()
    model.config.use_cache = True
    if device.type == "cuda":
        torch.cuda.empty_cache()
    report["stage_b"] = head_report(wrapper, processor, head_sets, labels, args, device, amp_dtype)
    (args.out / "metrics_head.json").write_text(json.dumps(report, indent=1))

    sets = [("real", A.real_records(args.real)[: args.eval_limit]), ("mapqa", mapqa_test), ("india", india_test)]
    for name, records in sets:
        print(f"scoring {len(records)} {name} maps, mode {args.mode}", flush=True)
        answers, countries, stats = predict(model, wrapper, processor, records, labels, args, device, amp_dtype)
        (args.out / f"purity_{name}.json").write_text(json.dumps(summarise(stats), indent=1))
        if name == "real":
            with open(args.out / "predictions_real.jsonl", "w") as f:
                for record, answer, country in zip(records, answers, countries):
                    f.write(json.dumps({"id": record["id"], "prediction": answer, "country": country}) + "\n")
        else:
            evaluate_and_save(records, answers, args.out, name)


if __name__ == "__main__":
    main()
