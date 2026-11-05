#!/usr/bin/env python3
"""Learned answer sheet v2: small regions are read from a zoomed view, and the model is trained that way.

v1 (train/answer_sheet.py) pools every region's token from the map's 14-px patches; Delhi and Rhode Island cover
about a fifth of one patch, so their tokens are mostly their neighbours. Zooming v1 without training failed for two
reasons (outputs/mapqa-derender-zoom-v1/SUMMARY.md): the head had never seen zoomed views, and the reader had never
seen tokens made from them. v2 trains both:

- Stage A (head alone): half the steps are 3x views around a small region, labelled from the same shapefile labels.
- Stage B (LoRA + head): every region smaller than `--small-mass` patches gets its token from a 3x view around it,
  pooled by the head in that view. The head keeps its patch labels as a side loss, on the views too.
- Reading: the head reads the map, every region it finds fewer than `--small-mass` patches of is looked at again in
  a 3x view around where it put that region, and the token is pooled there. Still one model: no fit, no OCR.

In training the view is centred on the region's true middle, moved a little at random; in reading, on the head's.
Continues from the v1 adapter and head.

    KAGGLE_API_TOKEN=... python kaggle/launch.py v2 --owner skini501 --tag v2 --dataset sartma/mapqa-s \
        --india-dataset sartma/synth-india-choropleth --extra-dataset skini501/answer-v2-inputs \
        --extra-dataset skini501/real-sheets --no-wait -- --adapter /kaggle/input
"""

import argparse
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from peft import PeftModel
from PIL import Image
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

import answer_sheet as A
from common import (add_common_args, autocast, evaluate_and_save, find_images_root, find_saved, read_split,
                    sample_records, set_seed, train_loop)
from train_qwen import report_lora


# ---------------------------------------------------------------- zoomed views

def zoom_box(size, cx, cy, side_share):
    width, height = size
    side = side_share * max(size)
    left = int(min(max(0, cx - side / 2), max(0, width - side)))
    top = int(min(max(0, cy - side / 2), max(0, height - side)))
    return left, top, int(min(width, left + side)), int(min(height, top + side))


def zoom_view(image, label, box, zoom):
    size = (max(28, round((box[2] - box[0]) * zoom)), max(28, round((box[3] - box[1]) * zoom)))
    view = image.crop(box).resize(size, Image.BICUBIC)
    crop = None
    if label is not None:
        crop = cv2.resize(np.ascontiguousarray(label[box[1]:box[3], box[0]:box[2]]), size, interpolation=cv2.INTER_NEAREST)
    return view, crop


def label_areas(label, max_pixels, classes):
    """Each class's area in 14-px patches, as the processor will see the image."""
    scale = min(1.0, max_pixels / label.size)
    return np.bincount(label.ravel(), minlength=classes) * scale / 196.0


def true_view(image, label, k, args, rng):
    """A zoomed view around class k's true middle, moved by up to `jitter` of the view's side."""
    ys, xs = np.nonzero(label == k)
    side = args.zoom_side * max(image.size)
    cx = xs.mean() + rng.uniform(-args.jitter, args.jitter) * side
    cy = ys.mean() + rng.uniform(-args.jitter, args.jitter) * side
    return zoom_view(image, label, zoom_box(image.size, cx, cy, args.zoom_side), args.zoom)


# ---------------------------------------------------------------- the model

class ZoomVisual(A.AnswerSheetVisual):
    """v1's tower, plus region tokens taken from zoomed views where `pending["views"]` gives them."""

    def forward(self, hidden_states, grid_thw, **kwargs):
        pending = self.pending
        views = pending.get("views") if pending else None
        view_losses = []
        if views:
            device = hidden_states.device
            with torch.no_grad():
                _, grids = self.patch_grids(views["pixel_values"].to(device).type(hidden_states.dtype),
                                            views["grid_thw"].to(device))
            override = pending.setdefault("override", {})
            for (m, j, k), grid, shares in zip(views["owner"], grids, views["labels"]):
                logits = self.head(grid)
                if shares is not None:
                    view_losses.append(-(shares.to(device) * logits.float().log_softmax(-1)).sum(-1).mean())
                weights = logits.float().softmax(-1).detach()[:, k:k + 1] ** self.sharpen
                override.setdefault(m, {})[j] = self.pool(grid, weights)[0]
        out = super().forward(hidden_states, grid_thw, **kwargs)
        if view_losses:
            losses = view_losses + ([self.aux_loss] if self.aux_loss is not None else [])
            self.aux_loss = torch.stack(losses).mean()
        return out


def attach(model, classes, device):
    for module in model.modules():
        if "visual" in module._modules and not isinstance(module._modules["visual"], A.AnswerSheetVisual):
            wrapper = ZoomVisual(module._modules["visual"], classes)
            wrapper.head.to(device=device, dtype=torch.float32)
            module._modules["visual"] = wrapper
            return wrapper
    raise SystemExit("could not find the vision tower")


# ---------------------------------------------------------------- training inputs

class TrainSet(A.TrainSet):
    """v1's examples, plus a zoomed view (and its labels) for every shown region smaller than `small_mass` patches."""

    def __init__(self, records, labels, args):
        super().__init__(records, labels)
        self.args = args

    def __getitem__(self, i):
        image, label, names, classes, target = super().__getitem__(i)
        views = []
        if label is not None:
            areas = label_areas(label, self.args.max_pixels, len(self.labels.vocab))
            rng = random.Random()
            small = [(j, k) for j, k in enumerate(classes) if 0 < areas[k] < self.args.small_mass]
            for j, k in rng.sample(small, min(len(small), self.args.max_views)):
                view, crop = true_view(image, label, k, self.args, rng)
                views.append((j, k, view, crop))
        return image, label, names, classes, target, views


class Collate(A.Collate):
    def __call__(self, batch):
        enc, extra = super().__call__([item[:5] for item in batch])
        owner, images, crops = [], [], []
        for m, item in enumerate(batch):
            for j, k, view, crop in item[5]:
                owner.append((m, j, k))
                images.append(view)
                crops.append(crop)
        if images:
            seen = self.processor.image_processor(images=images, return_tensors="pt")
            shares = [None if crop is None else A.patch_shares(crop, h, w, self.classes)
                      for crop, (_, h, w) in zip(crops, seen["image_grid_thw"].tolist())]
            extra["views"] = {"pixel_values": seen["pixel_values"], "grid_thw": seen["image_grid_thw"],
                              "owner": owner, "labels": shares}
        return enc, extra


# ---------------------------------------------------------------- stage A

def teach_head(wrapper, processor, records, labels, args, device, amp_dtype):
    """The head alone on frozen features: whole maps (half moved) and, `--zoom-share` of the time, zoomed views."""
    head = wrapper.head
    optimizer = torch.optim.AdamW(head.parameters(), lr=5e-4, weight_decay=0.01)
    deadline, began, step, window = time.time() + args.head_minutes * 60, time.time(), 0, {"map": [], "view": []}
    rng = random.Random(args.seed)
    head.train()
    while time.time() < deadline:
        for record in rng.sample(records, len(records)):
            image = Image.open(record["image_path"]).convert("RGB")
            label = labels.label(record, image.size)
            kind = "map"
            if rng.random() < args.zoom_share:
                areas = label_areas(label, args.max_pixels, len(labels.vocab))
                present = [k for k in range(1, len(labels.vocab)) if areas[k] > 0]
                small = [k for k in present if areas[k] < args.small_mass]
                image, label = true_view(image, label, rng.choice(small or present), args, rng)
                kind = "view"
            elif rng.random() < 0.5:
                image, label = A.move(image, label, rng)
            grid = A.map_grid(wrapper, processor, image, device, amp_dtype)
            shares = A.patch_shares(label, grid.shape[0], grid.shape[1], len(labels.vocab)).to(device)
            loss = -(shares * head(grid).log_softmax(-1)).sum(-1).mean()
            (loss / 4).backward()
            window[kind].append(loss.item())
            step += 1
            if step % 4 == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if step % 200 == 0:
                print(json.dumps({"head_step": step, **{f"loss_{n}": round(float(np.mean(v)), 4) for n, v in window.items() if v},
                                  "minutes": round((time.time() - began) / 60, 1)}), flush=True)
                window = {"map": [], "view": []}
            if time.time() > deadline:
                break
    head.eval()
    print(f"head taught on {step} images", flush=True)


# ---------------------------------------------------------------- reading

def zoom_tokens(wrapper, processor, image, label, grid, chances, country, codes, labels, args, device, amp_dtype):
    """{position in codes: token} for the regions the head finds fewer than `small_mass` patches of, and, where
    labels exist, each one's purity (share of its token that is its own pixels) before and after."""
    rows, columns = grid.shape[:2]
    down, across = torch.meshgrid(torch.arange(rows, device=device), torch.arange(columns, device=device), indexing="ij")
    down, across = down.flatten().float(), across.flatten().float()
    shares = None if label is None else A.patch_shares(label, rows, columns, len(labels.vocab)).to(device)
    overrides, stats = {}, []
    for j, code in enumerate(codes):
        k = labels.index[f"{country}-{code}"]
        p = chances[:, k]
        if p.sum() >= args.small_mass or p.max() < args.min_peak:
            continue
        w = p ** 4
        cx = float((w * (across + 0.5)).sum() / w.sum()) / columns * image.width
        cy = float((w * (down + 0.5)).sum() / w.sum()) / rows * image.height
        view, crop = zoom_view(image, label, zoom_box(image.size, cx, cy, args.zoom_side), args.zoom)
        view_grid = A.map_grid(wrapper, processor, view, device, amp_dtype)
        with torch.no_grad():
            weights = wrapper.head(view_grid).float().softmax(-1)[:, k:k + 1] ** wrapper.sharpen
            with autocast(device, amp_dtype):
                overrides[j] = wrapper.pool(view_grid, weights)[0]
        if shares is not None and shares[:, k].sum() > 0:
            before = p ** wrapper.sharpen
            view_shares = A.patch_shares(crop, view_grid.shape[0], view_grid.shape[1], len(labels.vocab)).to(device)
            after = weights[:, 0] / weights.sum().clamp_min(1e-6)
            stats.append({"code": code, "before": float((before / before.sum() * shares[:, k]).sum()),
                          "after": float((after * view_shares[:, k]).sum())})
    return overrides, stats


def predict(model, wrapper, processor, records, labels, args, device, amp_dtype, zoom=True):
    answers, countries, stats, began = [], [], [], time.time()
    for start in range(0, len(records), args.eval_batch_size):
        batch = records[start: start + args.eval_batch_size]
        texts, images, classes, overrides = [], [], [], {}
        for m, record in enumerate(batch):
            image = Image.open(record["image_path"]).convert("RGB")
            grid = A.map_grid(wrapper, processor, image, device, amp_dtype)
            with torch.no_grad():
                chances = wrapper.head(grid).float().softmax(-1)
            country = A.country_of(chances.cpu(), labels.vocab)
            codes = labels.listed[country]
            if zoom:
                label = labels.label(record, image.size)
                overrides[m], found = zoom_tokens(wrapper, processor, image, label, grid, chances, country, codes,
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
        print(f"predicted {len(answers)}/{len(records)} ({(time.time() - began) / 60:.1f} min), "
              f"{sum(len(o) for o in overrides.values())} regions zoomed in this batch", flush=True)
    return answers, countries, stats


def summarise(stats):
    by = {}
    for s in stats:
        by.setdefault(s["code"], []).append(s)
    return {code: {"zoomed": len(v), "purity_before": float(np.median([s["before"] for s in v])),
                   "purity_after": float(np.median([s["after"] for s in v]))}
            for code, v in sorted(by.items(), key=lambda kv: -len(kv[1]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--labels", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--real", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--adapter", required=True, help="holds v1's adapter/ and region_head.pt")
    parser.add_argument("--max-pixels", type=int, default=640_000)
    parser.add_argument("--zoom", type=float, default=3.0)
    parser.add_argument("--zoom-side", type=float, default=0.12, help="view side, as a share of the map's longer side")
    parser.add_argument("--small-mass", type=float, default=3.0, help="zoom regions smaller than this many patches")
    parser.add_argument("--min-peak", type=float, default=0.05, help="in reading, skip regions the head barely sees")
    parser.add_argument("--jitter", type=float, default=0.15, help="in training, move the view by up to this share")
    parser.add_argument("--max-views", type=int, default=10, help="in training, zoomed views per map")
    parser.add_argument("--zoom-share", type=float, default=0.5, help="stage A: share of steps on zoomed views")
    parser.add_argument("--head-minutes", type=float, default=30)
    parser.add_argument("--head-maps", type=int, default=2000)
    parser.add_argument("--aux-weight", type=float, default=0.5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.set_defaults(epochs=1, hours=4.5, lr=1e-4, batch_size=1, grad_accum=16, eval_batch_size=8,
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
    model = A.load_adapter(model, find_saved(args.adapter, "adapter_config.json"), trainable=True)
    report_lora(model)
    wrapper = attach(model, len(labels.vocab), device)
    head_file = next(Path(args.adapter).rglob("region_head.pt"))
    wrapper.head.load_state_dict(torch.load(head_file, map_location=device))
    print(f"continuing from {head_file}", flush=True)

    mapqa_root = find_images_root(args.mapqa, "map_*.png")
    usa_root = find_images_root(args.synth, "synth_*")
    india_root = find_images_root(args.india, "synthind_*")
    labelled_ids = set(json.loads(next(args.labels.rglob("mapqa_train_ids.json")).read_text()))
    mapqa_train = [r for r in read_split(args.data, "train", mapqa_root) if r["id"] in labelled_ids]
    mapqa_train = sample_records(mapqa_train, args.train_maps, args.seed + 7)  # a different draw from v1's
    usa_train = sample_records(read_split(usa_root, "train", usa_root), args.synth_train_maps, args.seed + 7)
    india_train = sample_records(read_split(india_root, "train", india_root), args.india_train_maps, args.seed + 7)
    print(f"training maps: {len(mapqa_train)} MapQA + {len(usa_train)} US + {len(india_train)} India", flush=True)

    # stage A
    for p in model.parameters():
        p.requires_grad_(False)
    wrapper.head.requires_grad_(True)
    model.eval()
    teach = (sample_records(mapqa_train, args.head_maps, 1) + sample_records(usa_train, args.head_maps, 1)
             + sample_records(india_train, args.head_maps, 1))
    teach_head(wrapper, processor, teach, labels, args, device, amp_dtype)
    torch.save(wrapper.head.state_dict(), args.out / "region_head_stage_a.pt")

    mapqa_test = sample_records(read_split(args.data, "test", mapqa_root), args.eval_maps, seed=0)
    india_test = sample_records(read_split(india_root, "test", india_root), args.india_eval_maps, seed=0)
    real_probe = [{"id": p.stem, "image_path": str(p)} for p in sorted(labels.real.glob("*.jpg"))
                  if (labels.folder / "real" / f"{p.stem}.png").exists()]
    if args.eval_limit:
        mapqa_test, india_test, real_probe = (x[: args.eval_limit] for x in (mapqa_test, india_test, real_probe))
    head_sets = [("india_as_drawn", india_test[:150], False), ("india_moved", india_test[:150], True),
                 ("mapqa", mapqa_test[:150], False), ("real_india", real_probe, False)]
    report = {"stage_a": A.head_report(wrapper, processor, head_sets, labels, device, amp_dtype)}
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
    report["stage_b"] = A.head_report(wrapper, processor, head_sets, labels, device, amp_dtype)
    (args.out / "metrics_head.json").write_text(json.dumps(report, indent=1))

    sets = [("real", A.real_records(args.real)[: args.eval_limit]), ("mapqa", mapqa_test), ("india", india_test)]
    for name, records in sets:
        print(f"scoring {len(records)} {name} maps with zoom", flush=True)
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
