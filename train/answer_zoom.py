#!/usr/bin/env python3
"""Does zooming in fix the small regions of the learned answer sheet? No training: the v1 model, as saved.

v1's small regions are smaller than one 14-px patch (Delhi and Rhode Island cover about a fifth of one), so their
region tokens are mostly their neighbours. Here the head first reads the whole map as before. Every region it finds
less than `--small-mass` patches of is then looked at again in a zoomed view around where the head put it, and that
region's token is pooled from the zoomed patches instead. Everything else is unchanged, so the comparison with v1
is the same weights, zoom off against zoom on.

Variants:
- zoom:         the head picks the region's patches in the zoomed view
- zoom_oracle:  the true labels pick them (only where labels exist). If this fixes reading and `zoom` does not, the
                head needs teaching on zoomed views; if neither does, zooming is not the answer

Also writes purity_<set>_<variant>.json: per zoomed region, how much of its token is its own pixels, before and after.

    python kaggle/launch.py zoom --tag v1 --synth-dataset none --india-dataset sartma/synth-india-choropleth \
        --extra-dataset sartma/region-probe --extra-dataset sartma/answer-labels --extra-dataset sartma/real-sheets \
        --from-run sartma/mapqa-derender-answer-v1 --no-wait -- --adapter /kaggle/input
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

import answer_sheet as A
from common import add_common_args, autocast, evaluate_and_save, find_images_root, find_saved, read_split, sample_records

import cv2


def head_chances(wrapper, processor, image, device, amp_dtype):
    grid = A.map_grid(wrapper, processor, image, device, amp_dtype)
    with torch.no_grad():
        return grid, wrapper.head(grid).float().softmax(-1)


def zoom_overrides(wrapper, processor, image, label, country, codes, labels, args, device, amp_dtype, oracle):
    """{position in codes: token} for the small regions, and purity stats where labels exist."""
    grid, chances = head_chances(wrapper, processor, image, device, amp_dtype)
    rows, columns = grid.shape[:2]
    shares = None if label is None else A.patch_shares(label, rows, columns, len(labels.vocab)).to(device)
    down, across = torch.meshgrid(torch.arange(rows, device=device), torch.arange(columns, device=device), indexing="ij")
    down, across = down.flatten().float(), across.flatten().float()
    side = args.side * max(image.size)
    overrides, stats = {}, []
    for j, code in enumerate(codes):
        k = labels.index[f"{country}-{code}"]
        p = chances[:, k]
        if p.sum() >= args.small_mass or p.max() < args.min_peak:
            continue
        w = p ** 4
        cx = float((w * (across + 0.5)).sum() / w.sum()) / columns * image.width
        cy = float((w * (down + 0.5)).sum() / w.sum()) / rows * image.height
        left = int(min(max(0, cx - side / 2), max(0, image.width - side)))
        top = int(min(max(0, cy - side / 2), max(0, image.height - side)))
        box = (left, top, int(min(image.width, left + side)), int(min(image.height, top + side)))
        size = (round((box[2] - box[0]) * args.zoom), round((box[3] - box[1]) * args.zoom))
        view = image.crop(box).resize(size, Image.BICUBIC)
        view_grid, view_chances = head_chances(wrapper, processor, view, device, amp_dtype)
        view_shares = None
        if label is not None:
            crop = cv2.resize(label[box[1]:box[3], box[0]:box[2]], size, interpolation=cv2.INTER_NEAREST)
            view_shares = A.patch_shares(crop, view_grid.shape[0], view_grid.shape[1], len(labels.vocab)).to(device)
        if oracle:
            if view_shares is None or view_shares[:, k].sum() == 0:
                continue
            weights = view_shares[:, k]
        else:
            weights = view_chances[:, k] ** wrapper.sharpen
        with torch.no_grad(), autocast(device, amp_dtype):
            overrides[j] = wrapper.pool(view_grid, weights[:, None])[0]
        if shares is not None and shares[:, k].sum() > 0:
            before = p ** wrapper.sharpen
            after = weights / weights.sum().clamp_min(1e-6)
            stats.append({"code": code, "before": float((before / before.sum() * shares[:, k]).sum()),
                          "after": None if view_shares is None else float((after * view_shares[:, k]).sum())})
    return overrides, stats


def predict(model, wrapper, processor, records, labels, args, device, amp_dtype, oracle):
    answers, countries, stats, began = [], [], [], time.time()
    for start in range(0, len(records), args.eval_batch_size):
        batch = records[start: start + args.eval_batch_size]
        texts, images, classes, overrides = [], [], [], {}
        for m, record in enumerate(batch):
            image = Image.open(record["image_path"]).convert("RGB")
            _, chances = head_chances(wrapper, processor, image, device, amp_dtype)
            country = A.country_of(chances.cpu(), labels.vocab)
            codes = labels.listed[country]
            label = labels.label(record, image.size)
            overrides[m], found = zoom_overrides(wrapper, processor, image, label, country, codes, labels, args, device,
                                                 amp_dtype, oracle)
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
                   "purity_after": float(np.median([s["after"] for s in v if s["after"] is not None] or [np.nan]))}
            for code, v in sorted(by.items(), key=lambda kv: -len(kv[1]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--labels", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--real", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--adapter", required=True, help="the v1 run's output: adapter/ and region_head.pt")
    parser.add_argument("--max-pixels", type=int, default=640_000)
    parser.add_argument("--zoom", type=float, default=3.0)
    parser.add_argument("--side", type=float, default=0.12, help="zoomed view's side, as a share of the map's longer side")
    parser.add_argument("--small-mass", type=float, default=3.0, help="zoom regions the head finds fewer patches of")
    parser.add_argument("--min-peak", type=float, default=0.05, help="skip regions the head barely sees anywhere")
    parser.add_argument("--variants", default="zoom,zoom_oracle")
    parser.set_defaults(eval_batch_size=8, max_new_tokens=600, india_eval_maps=300)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "args.json").write_text(json.dumps(vars(args), default=str, indent=1))

    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    labels = A.Labels(args.labels)
    processor = AutoProcessor.from_pretrained(args.model, max_pixels=args.max_pixels, min_pixels=28 * 28)
    processor.tokenizer.padding_side = "left"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model, torch_dtype=dtype,
                                                               attn_implementation="sdpa").to(device)
    model = A.load_adapter(model, find_saved(args.adapter, "adapter_config.json"), trainable=False)
    wrapper = A.attach(model, len(labels.vocab), device)
    head = next(p for p in Path(args.adapter).rglob("region_head.pt"))
    wrapper.head.load_state_dict(torch.load(head, map_location=device))
    model.eval()
    print(f"loaded {head}", flush=True)

    mapqa_root = find_images_root(args.mapqa, "map_*.png")
    india_root = find_images_root(args.india, "synthind_*")
    sets = [("real", A.real_records(args.real)),
            ("mapqa", sample_records(read_split(args.data, "test", mapqa_root), args.eval_maps, seed=0)),
            ("india", sample_records(read_split(india_root, "test", india_root), args.india_eval_maps, seed=0))]
    for variant in args.variants.split(","):
        for name, records in sets:
            records = records[: args.eval_limit] if args.eval_limit else records
            if name == "real" and variant == "zoom_oracle":
                continue
            print(f"scoring {len(records)} {name} maps, {variant}", flush=True)
            answers, countries, stats = predict(model, wrapper, processor, records, labels, args, device, dtype,
                                                oracle=variant == "zoom_oracle")
            (args.out / f"purity_{name}_{variant}.json").write_text(json.dumps(summarise(stats), indent=1))
            if name == "real":
                with open(args.out / f"predictions_real_{variant}.jsonl", "w") as f:
                    for record, answer, country in zip(records, answers, countries):
                        f.write(json.dumps({"id": record["id"], "prediction": answer, "country": country}) + "\n")
            else:
                evaluate_and_save(records, answers, args.out, f"{name}_{variant}")


if __name__ == "__main__":
    main()
