#!/usr/bin/env python3
"""Learned answer sheet: one model that finds a map's regions itself and reads each one, with no fitting or OCR.

The region sheet showed the model reads a region well once that region is handed to it by name. Here the model
makes that hand-over itself:

1. Region head. A small head on the vision tower's 14-px patches (before Qwen merges them into 28-px tokens)
   says which region each patch belongs to. Its vocabulary holds every region of every country it is taught
   (US and India here), so it also says which country the map shows.
2. Region tokens. For each region, the patches the head gives it are averaged and passed through Qwen's own
   patch merger, giving one token that looks like a token filled with that region. No new projection is learned.
3. The language model sees the map, then "code: [region token]" for every region of the country, and writes the
   usual JSON, reading the legend from the map itself.

Shapefiles appear only in teaching: they label which region lies under each patch (synthetic maps from the
generator; MapQA and real maps from boundary fits made beforehand by lookup/answer_labels.py).

Stage A teaches the head alone on frozen tokens. Stage B trains LoRA and the head together on the JSON answer,
the head keeping its patch labels as a side loss. In training, half the maps show a random set of regions under
random codes, so the answer has to come from each region token, not from memorised geography.

Scored on: the head's patches (synthetic India, MapQA test, real India), real India maps (predictions only; scored
locally by lookup/score_real.py), 300 synthetic India test maps and the 400 MapQA test maps.

    python kaggle/launch.py answer --tag v1 --india-dataset sartma/synth-india-choropleth \
        --extra-dataset sartma/region-probe --extra-dataset sartma/answer-labels --extra-dataset sartma/real-sheets \
        --from-run sartma/mapqa-derender-qwen --no-wait -- --adapter /kaggle/input
"""

import argparse
import json
import random
import string
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from peft import PeftModel
from PIL import Image
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from common import (add_common_args, autocast, evaluate_and_save, find_images_root, find_saved, read_split,
                    sample_records, set_seed, train_loop)
from train_qwen import report_lora

PROMPT = (
    "The first image is a choropleth map. After this text, each region of the map is listed as its code followed by "
    "one image token showing that region. Return only JSON with these keys: "
    '"title"; "legend_title" (null if there is none); "legend_type" ("continuous" or "discrete"); '
    '"data_type" ("percentage" or "number"); "ranges" (discrete legends only: the legend ranges, lowest first); '
    '"values" (every listed region by its code, in the order listed: a number for a continuous legend, the index of '
    "the region's range for a discrete legend, null for no data). Regions:"
)
SAMPLES = 4  # label points per patch side
SMALL = {"US": {"RI", "DE", "DC", "CT", "NJ", "VT", "NH", "MA", "MD", "HI"}, "IN": {"AN", "DL", "GA", "SK", "TR"}}


# ---------------------------------------------------------------- labels

class Labels:
    """Which region lies under each pixel, as classes of one vocabulary shared by every country."""

    def __init__(self, root):
        root = Path(root)
        self.vocab = json.loads(next(root.rglob("vocab.json")).read_text())
        self.folder = next(root.rglob("vocab.json")).parent
        self.index = {name: i for i, name in enumerate(self.vocab)}
        probe = next(root.rglob("regions_usa.json")).parent
        self.synth = {}
        for prefix, name, country in (("synthind_", "regions", "IN"), ("synth_", "regions_usa", "US")):
            meta = json.loads((probe / f"{name}.json").read_text())
            raster = np.asarray(Image.open(probe / f"{name}.png"))
            lut = np.zeros(256, np.uint8)
            for k, code in enumerate(meta["codes"], 1):
                lut[k] = self.index[f"{country}-{code}"]
            self.synth[prefix] = (meta, lut[raster])
        self.real = probe / "real"
        india = json.loads((probe / "regions.json").read_text())
        usa = json.loads((probe / "regions_usa.json").read_text())
        self.listed = {"IN": india["listed"], "US": usa["codes"]}

    @staticmethod
    def country(record):
        return "US" if record["id"].startswith(("synth_", "map_")) else "IN"

    def label(self, record, size):
        """(height, width) uint8 class numbers for this record's image, or None when it has no labels."""
        name = record["id"]
        for prefix, (meta, raster) in self.synth.items():
            if name.startswith(prefix):
                left, top, right, bottom = meta["placed"][name]
                place = np.array([[(right - left) / raster.shape[1], 0, left], [0, (bottom - top) / raster.shape[0], top]])
                return cv2.warpAffine(raster, place, size, flags=cv2.INTER_NEAREST)
        for folder in ("mapqa", "real"):
            path = self.folder / folder / f"{name}.png"
            if path.exists():
                label = np.asarray(Image.open(path))
                return label if label.shape[::-1] == tuple(size) else cv2.resize(label, size, interpolation=cv2.INTER_NEAREST)
        return None


def move(image, label, rng):
    """Map and labels shrunk, stretched a little and placed at random on a page of the map's own colour."""
    scale_x = rng.uniform(0.6, 1.0)
    scale_y = min(1.0, scale_x * rng.uniform(0.9, 1.1))
    width, height = max(1, int(image.width * scale_x)), max(1, int(image.height * scale_y))
    x, y = rng.randint(0, image.width - width), rng.randint(0, image.height - height)
    page = Image.new("RGB", image.size, image.getpixel((1, 1)))
    page.paste(image.resize((width, height), Image.BILINEAR), (x, y))
    moved = None
    if label is not None:
        moved = np.zeros_like(label)
        moved[y: y + height, x: x + width] = cv2.resize(label, (width, height), interpolation=cv2.INTER_NEAREST)
    return page, moved


def patch_shares(label, rows, columns, classes):
    """(rows * columns, classes): the share of each 14-px patch that each class covers, patches row by row."""
    height, width = label.shape
    xs = np.minimum(((np.arange(columns * SAMPLES) + 0.5) / (columns * SAMPLES) * width).astype(int), width - 1)
    ys = np.minimum(((np.arange(rows * SAMPLES) + 0.5) / (rows * SAMPLES) * height).astype(int), height - 1)
    found = label[np.ix_(ys, xs)].astype(np.int64)
    patch = (np.arange(len(ys)) // SAMPLES)[:, None] * columns + (np.arange(len(xs)) // SAMPLES)[None, :]
    counts = np.bincount((patch * classes + found).ravel(), minlength=rows * columns * classes)
    return torch.from_numpy(counts.reshape(-1, classes) / SAMPLES ** 2).float()


# ---------------------------------------------------------------- the model's new parts

class RegionHead(torch.nn.Module):
    """Patch features (rows, columns, width) -> class logits (rows * columns, classes), seeing a few neighbours."""

    def __init__(self, width, classes, hidden=256):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(width, hidden, 1), torch.nn.GELU(),
            torch.nn.Conv2d(hidden, hidden, 3, padding=1), torch.nn.GELU(),
            torch.nn.Conv2d(hidden, hidden, 3, padding=2, dilation=2), torch.nn.GELU(),
            torch.nn.Conv2d(hidden, hidden, 3, padding=4, dilation=4), torch.nn.GELU(),
            torch.nn.Conv2d(hidden, classes, 1))

    def forward(self, grid):
        out = self.net(grid.permute(2, 0, 1)[None].float())[0]
        return out.permute(1, 2, 0).flatten(0, 1)


class AnswerSheetVisual(torch.nn.Module):
    """The vision tower, plus region tokens for the 28-px placeholder images that follow each map.

    Images arrive as: map, its placeholders, next map, its placeholders... A placeholder is any 2x2-patch image.
    `pending` gives, per map, the class of each of its placeholders and (in training) the map's patch labels.
    """

    def __init__(self, visual, classes, sharpen=2.0):
        super().__init__()
        self.visual = visual
        self.head = RegionHead(visual.config.hidden_size, classes)
        self.sharpen = sharpen
        self.pending, self.aux_loss, self._captured = None, None, None
        visual.merger.register_forward_pre_hook(self._capture)

    def _capture(self, module, args):
        self._captured = args[0]

    def __getattr__(self, name):  # dtype, spatial_merge_size, config and friends come from the tower
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._modules["visual"], name)

    def patch_grids(self, hidden_states, grid_thw):
        """Run the tower on maps only: (merged tokens, [(rows, columns, width) patch features per map])."""
        tokens = self.visual(hidden_states, grid_thw=grid_thw)
        before = self._captured
        window_index, _ = self.visual.get_window_index(grid_thw)
        before = before.reshape(-1, 4, before.shape[-1])[torch.argsort(window_index)]
        grids, start = [], 0
        for _, height, width in grid_thw.tolist():
            count = height * width // 4
            part = before[start: start + count].reshape(height // 2, width // 2, 2, 2, -1)
            grids.append(part.permute(0, 2, 1, 3, 4).reshape(height, width, -1))
            start += count
        return tokens, grids

    def forward(self, hidden_states, grid_thw, **kwargs):
        pending, self.pending = self.pending, None
        placeholder = (grid_thw[:, 1] == 2) & (grid_thw[:, 2] == 2)
        if pending is None or not placeholder.any():
            return self.visual(hidden_states, grid_thw=grid_thw, **kwargs)
        chunks = torch.split(hidden_states, grid_thw.prod(-1).tolist())
        maps = [i for i in range(len(chunks)) if not placeholder[i]]
        tokens, grids = self.patch_grids(torch.cat([chunks[i] for i in maps]), grid_thw[maps])
        sizes = [h * w // 4 for _, h, w in grid_thw[maps].tolist()]
        out, losses = [], []
        for m, (map_tokens, grid) in enumerate(zip(torch.split(tokens, sizes), grids)):
            logits = self.head(grid)
            shares = pending["labels"][m]
            if shares is not None:
                losses.append(-(shares.to(logits.device) * logits.float().log_softmax(-1)).sum(-1).mean())
            regions = self.region_tokens(grid, logits, pending["classes"][m]).to(map_tokens.dtype)
            for j, token in pending.get("override", {}).get(m, {}).items():  # e.g. made from a zoomed view
                regions[j] = token.to(regions.dtype)
            out += [map_tokens, regions]
        self.aux_loss = torch.stack(losses).mean() if losses else None
        return torch.cat(out)

    def region_tokens(self, grid, logits, classes):
        """One token per class: its patches averaged, weighted by the head's (sharpened) belief, then merged."""
        chances = logits.float().softmax(-1).detach()[:, classes] ** self.sharpen
        return self.pool(grid, chances)

    def pool(self, grid, weights):
        """grid (rows, columns, width), weights (rows * columns, k) -> k tokens through Qwen's own merger."""
        weights = weights / weights.sum(0, keepdim=True).clamp_min(1e-6)
        flat = grid.reshape(-1, grid.shape[-1])
        pooled = weights.T.to(flat.dtype) @ flat
        return self.visual.merger(pooled.repeat_interleave(4, dim=0))


def attach(model, classes, device):
    for module in model.modules():
        if "visual" in module._modules and not isinstance(module._modules["visual"], AnswerSheetVisual):
            wrapper = AnswerSheetVisual(module._modules["visual"], classes)
            wrapper.head.to(device=device, dtype=torch.float32)
            module._modules["visual"] = wrapper
            return wrapper
    raise SystemExit("could not find the vision tower")


WRAPPED, PLAIN = ".visual.visual.", ".visual."


def load_adapter(model, folder, trainable):
    """PeftModel.from_pretrained that also reads adapters saved with the wrapper attached (v1's): their vision LoRA
    keys say visual.visual.*, which peft cannot match on a plain model and silently leaves at zero."""
    import shutil
    import tempfile
    from safetensors.torch import load_file, save_file
    folder = Path(folder)
    weights = load_file(folder / "adapter_model.safetensors")
    if any(WRAPPED in k for k in weights):
        fixed = Path(tempfile.mkdtemp())
        for f in folder.iterdir():
            if f.is_file() and f.name != "adapter_model.safetensors":
                shutil.copy(f, fixed / f.name)
        save_file({k.replace(WRAPPED, PLAIN): v for k, v in weights.items()}, fixed / "adapter_model.safetensors")
        print(f"renamed {sum(WRAPPED in k for k in weights)} wrapped vision keys", flush=True)
        folder = fixed
    model = PeftModel.from_pretrained(model, folder, is_trainable=trainable)
    if any("visual" in k for k in weights):  # a fresh LoRA's B is zero, so a zero B means its weights never loaded
        empty = [n for n, p in model.named_parameters() if "visual" in n and "lora_B" in n and not p.detach().abs().sum()]
        if empty:
            raise SystemExit(f"{len(empty)} vision LoRA weights did not load, e.g. {empty[0]}")
    return model


def save_adapter(model, folder):
    """save_pretrained, with the vision keys written as on a plain model so the adapter loads anywhere."""
    from safetensors.torch import load_file, save_file
    model.save_pretrained(folder)
    path = Path(folder) / "adapter_model.safetensors"
    weights = load_file(path)
    save_file({k.replace(WRAPPED, PLAIN): v for k, v in weights.items()}, path)


# ---------------------------------------------------------------- inputs

PLACEHOLDER = Image.new("RGB", (28, 28), (128, 128, 128))


def chat(processor, codes, target=None):
    content = [{"type": "image"}, {"type": "text", "text": PROMPT}]
    for code in codes:
        content += [{"type": "text", "text": f" {code}:"}, {"type": "image"}]
    messages = [{"role": "user", "content": content}]
    if target is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": target}]})
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=target is None)


class TrainSet(torch.utils.data.Dataset):
    def __init__(self, records, labels, move_share=0.5, rename_share=0.5):
        self.records, self.labels = records, labels
        self.move_share, self.rename_share = move_share, rename_share

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        rng = random.Random()
        record = self.records[i]
        image = Image.open(record["image_path"]).convert("RGB")
        label = self.labels.label(record, image.size)
        if rng.random() < self.move_share:
            image, label = move(image, label, rng)
        target = json.loads(record["target"])
        country = Labels.country(record)
        real = list(target["values"])
        if rng.random() < self.rename_share:  # a random set of regions under random codes: copy, don't recall
            chosen = rng.sample(real, rng.randint(min(16, len(real)), len(real)))
            names = rng.sample([a + b for a in string.ascii_uppercase for b in string.ascii_uppercase], len(chosen))
            shown = sorted(zip(names, chosen))
        else:
            shown = [(c, c) for c in real]
        target["values"] = {name: target["values"][code] for name, code in shown}
        classes = [self.labels.index[f"{country}-{code}"] for _, code in shown]
        return image, label, [name for name, _ in shown], classes, json.dumps(target, ensure_ascii=False,
                                                                              separators=(",", ":"))


class Collate:
    def __init__(self, processor, classes):
        self.processor, self.classes = processor, classes
        self.answer_start = processor.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)

    def __call__(self, batch):
        texts, images, labels, classes = [], [], [], []
        for image, label, names, region_classes, target in batch:
            texts.append(chat(self.processor, names, target))
            images += [image] + [PLACEHOLDER] * len(names)
            labels.append(label)
            classes.append(region_classes)
        enc = self.processor(text=texts, images=images, padding=True, return_tensors="pt")
        grids = [g for g in enc["image_grid_thw"].tolist() if g[1:] != [2, 2]]
        shares = [None if label is None else patch_shares(label, h, w, self.classes)
                  for label, (_, h, w) in zip(labels, grids)]
        ids = enc["input_ids"]
        answer = ids.clone()
        answer[enc["attention_mask"] == 0] = -100
        marker = torch.tensor(self.answer_start)
        for row in range(len(ids)):
            hits = (ids[row].unfold(0, len(marker), 1) == marker).all(dim=1).nonzero()
            answer[row, : int(hits[-1]) + len(marker)] = -100
        enc["labels"] = answer
        return enc, {"classes": classes, "labels": shares}


# ---------------------------------------------------------------- stage A: the head alone

def map_grid(wrapper, processor, image, device, amp_dtype):
    seen = processor.image_processor(images=[image], return_tensors="pt")
    with torch.no_grad(), autocast(device, amp_dtype):
        _, grids = wrapper.patch_grids(seen["pixel_values"].to(device).type(wrapper.visual.dtype),
                                       seen["image_grid_thw"].to(device))
    return grids[0]


def teach_head(wrapper, processor, records, labels, device, amp_dtype, minutes, seed):
    """The head alone on frozen patch features, one map at a time, half of them moved."""
    head = wrapper.head
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.01)
    deadline, began, step, window = time.time() + minutes * 60, time.time(), 0, []
    rng = random.Random(seed)
    head.train()
    while time.time() < deadline:
        for record in rng.sample(records, len(records)):
            image = Image.open(record["image_path"]).convert("RGB")
            label = labels.label(record, image.size)
            if rng.random() < 0.5:
                image, label = move(image, label, rng)
            grid = map_grid(wrapper, processor, image, device, amp_dtype)
            shares = patch_shares(label, grid.shape[0], grid.shape[1], len(labels.vocab)).to(device)
            loss = -(shares * head(grid).log_softmax(-1)).sum(-1).mean()
            (loss / 4).backward()
            window.append(loss.item())
            step += 1
            if step % 4 == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if step % 200 == 0:
                print(json.dumps({"head_step": step, "loss": round(float(np.mean(window)), 4),
                                  "minutes": round((time.time() - began) / 60, 1)}), flush=True)
                window.clear()
            if time.time() > deadline:
                break
    head.eval()
    print(f"head taught on {step} maps", flush=True)


# ---------------------------------------------------------------- scoring

def country_of(chances, vocab):
    """The country whose regions hold most of the head's belief over the page."""
    mass = chances[:, 1:].sum(0)
    totals = {}
    for k, name in enumerate(vocab[1:]):
        totals[name[:2]] = totals.get(name[:2], 0.0) + float(mass[k])
    return max(totals, key=totals.get)


def head_report(wrapper, processor, sets, labels, device, amp_dtype):
    """Per set: country right, patch accuracy, and how pure each region's token is (the share of its pooling
    weight that falls on that region's own pixels), big and small regions apart."""
    report = {}
    for name, records, moved in sets:
        rng = random.Random(1)
        right_country, patch_right, patches = 0, 0, 0
        purity = {"big": [], "small": []}
        for record in records:
            image = Image.open(record["image_path"]).convert("RGB")
            label = labels.label(record, image.size)
            if label is None:
                continue
            if moved:
                image, label = move(image, label, rng)
            grid = map_grid(wrapper, processor, image, device, amp_dtype)
            with torch.no_grad():
                chances = wrapper.head(grid).float().softmax(-1).cpu()
            shares = patch_shares(label, grid.shape[0], grid.shape[1], len(labels.vocab))
            country = Labels.country(record)
            right_country += country_of(chances, labels.vocab) == country
            on_map = shares[:, 1:].sum(1) > 0.5
            patch_right += int((chances[on_map].argmax(1) == shares[on_map].argmax(1)).sum())
            patches += int(on_map.sum())
            for code in labels.listed[country]:
                k = labels.index[f"{country}-{code}"]
                if shares[:, k].sum() == 0:
                    continue
                weights = chances[:, k] ** wrapper.sharpen
                weights = weights / weights.sum().clamp_min(1e-6)
                purity["small" if code in SMALL[country] else "big"].append(float((weights * shares[:, k]).sum()))
        report[name] = {"maps": len(records), "country_right": right_country / max(1, len(records)),
                        "patch_accuracy": patch_right / max(1, patches),
                        **{f"{size}_purity_median": float(np.median(v)) if v else None for size, v in purity.items()},
                        **{f"{size}_purity_over_0.8": float(np.mean(np.asarray(v) > 0.8)) if v else None
                           for size, v in purity.items()}}
        print(name, json.dumps(report[name]), flush=True)
    return report


def predict(model, wrapper, processor, records, labels, args, device, amp_dtype, force=None):
    """Each map: the head names the country (or `force` does), then the model reads every listed region of it."""
    answers, countries, began = [], [], time.time()
    for start in range(0, len(records), args.eval_batch_size):
        batch = records[start: start + args.eval_batch_size]
        texts, images, classes = [], [], []
        for record in batch:
            image = Image.open(record["image_path"]).convert("RGB")
            grid = map_grid(wrapper, processor, image, device, amp_dtype)
            with torch.no_grad():
                country = country_of(wrapper.head(grid).float().softmax(-1).cpu(), labels.vocab)
            country = force or country
            codes = labels.listed[country]
            countries.append(country)
            texts.append(chat(processor, codes))
            images += [image] + [PLACEHOLDER] * len(codes)
            classes.append([labels.index[f"{country}-{c}"] for c in codes])
        enc = processor(text=texts, images=images, padding=True, return_tensors="pt").to(device)
        wrapper.pending = {"classes": classes, "labels": [None] * len(batch)}
        with torch.no_grad(), autocast(device, amp_dtype):
            ids = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True)
        answers += processor.batch_decode(ids[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"predicted {len(answers)}/{len(records)} ({(time.time() - began) / 60:.1f} min)", flush=True)
    return answers, countries


def real_records(root):
    """The linked real India maps (records.jsonl beside maps/), with no targets: scored locally."""
    path = next(Path(root).rglob("records.jsonl"), None)
    if path is None:
        return []
    return [{"id": r["id"], "image_path": str(path.parent / r["map"])} for r in map(json.loads, open(path))]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--labels", type=Path, default=Path("/kaggle/input"),
                        help="holds vocab.json (lookup/answer_labels.py) and regions_usa.json (synth/region_labels.py)")
    parser.add_argument("--real", type=Path, default=Path("/kaggle/input"), help="holds the real maps' records.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--adapter", help="the LoRA adapter to continue from")
    parser.add_argument("--max-pixels", type=int, default=640_000)
    parser.add_argument("--head-minutes", type=float, default=40, help="stage A: the head alone")
    parser.add_argument("--head-maps", type=int, default=2000, help="stage A maps per source (US, India, MapQA)")
    parser.add_argument("--aux-weight", type=float, default=0.5, help="stage B: weight of the head's patch loss")
    parser.add_argument("--head-lr", type=float, default=1e-4, help="stage B: the head's learning rate")
    parser.add_argument("--head-only", action="store_true", help="stop after stage A and its report")
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
    labels = Labels(args.labels)
    processor = AutoProcessor.from_pretrained(args.model, max_pixels=args.max_pixels, min_pixels=28 * 28)
    processor.tokenizer.padding_side = "left"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model, torch_dtype=dtype,
                                                               attn_implementation="sdpa").to(device)
    model = load_adapter(model, find_saved(args.adapter, "adapter_config.json"), trainable=True)
    report_lora(model)
    wrapper = attach(model, len(labels.vocab), device)

    mapqa_root = find_images_root(args.mapqa, "map_*.png")
    usa_root = find_images_root(args.synth, "synth_*")
    india_root = find_images_root(args.india, "synthind_*")
    labelled_ids = set(json.loads(next(args.labels.rglob("mapqa_train_ids.json")).read_text()))
    mapqa_train = [r for r in read_split(args.data, "train", mapqa_root) if r["id"] in labelled_ids]
    mapqa_train = sample_records(mapqa_train, args.train_maps, args.seed)
    usa_train = sample_records(read_split(usa_root, "train", usa_root), args.synth_train_maps, args.seed)
    india_train = sample_records(read_split(india_root, "train", india_root), args.india_train_maps, args.seed)
    print(f"training maps: {len(mapqa_train)} MapQA + {len(usa_train)} US + {len(india_train)} India", flush=True)

    # stage A
    for p in model.parameters():
        p.requires_grad_(False)
    wrapper.head.requires_grad_(True)
    teach = (sample_records(mapqa_train, args.head_maps, 1) + sample_records(usa_train, args.head_maps, 1)
             + sample_records(india_train, args.head_maps, 1))
    model.eval()
    teach_head(wrapper, processor, teach, labels, device, amp_dtype, args.head_minutes, args.seed)
    torch.save(wrapper.head.state_dict(), args.out / "region_head_stage_a.pt")

    mapqa_test = sample_records(read_split(args.data, "test", mapqa_root), args.eval_maps, seed=0)
    india_test = sample_records(read_split(india_root, "test", india_root), args.india_eval_maps, seed=0)
    real_probe = [{"id": p.stem, "image_path": str(p)} for p in sorted(labels.real.glob("*.jpg"))
                  if (labels.folder / "real" / f"{p.stem}.png").exists()]
    if args.eval_limit:
        mapqa_test, india_test = mapqa_test[: args.eval_limit], india_test[: args.eval_limit]
        real_probe = real_probe[: args.eval_limit]
    head_sets = [("india_as_drawn", india_test[:150], False), ("india_moved", india_test[:150], True),
                 ("mapqa", mapqa_test[:150], False), ("real_india", real_probe, False)]
    report = {"stage_a": head_report(wrapper, processor, head_sets, labels, device, amp_dtype)}
    (args.out / "metrics_head.json").write_text(json.dumps(report, indent=1))
    if args.head_only:
        return

    # stage B
    lora = [p for n, p in model.named_parameters() if "lora_" in n]
    for p in lora:
        p.requires_grad_(True)
    model.train()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    records = mapqa_train + usa_train + india_train
    loader = DataLoader(TrainSet(records, labels), batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                        collate_fn=Collate(processor, len(labels.vocab)))

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
    save_adapter(model, args.out / "adapter")
    torch.save(wrapper.head.state_dict(), args.out / "region_head.pt")

    model.eval()
    model.config.use_cache = True
    if device.type == "cuda":
        torch.cuda.empty_cache()
    report["stage_b"] = head_report(wrapper, processor, head_sets, labels, device, amp_dtype)
    (args.out / "metrics_head.json").write_text(json.dumps(report, indent=1))

    real = real_records(args.real)[: args.eval_limit]
    if real:
        answers, countries = predict(model, wrapper, processor, real, labels, args, device, amp_dtype)
        with open(args.out / "predictions_real.jsonl", "w") as f:
            for record, answer, country in zip(real, answers, countries):
                f.write(json.dumps({"id": record["id"], "prediction": answer, "country": country}) + "\n")
    for name, records in (("india", india_test), ("mapqa", mapqa_test)):
        answers, countries = predict(model, wrapper, processor, records, labels, args, device, amp_dtype)
        evaluate_and_save(records, answers, args.out, name)
        (args.out / f"countries_{name}.json").write_text(json.dumps(dict(zip([r["id"] for r in records], countries))))
    if report["stage_b"]["india_as_drawn"]["country_right"] < 0.95:  # reading apart from naming the country
        print("head names the country wrongly too often; scoring India again with the country given", flush=True)
        answers, _ = predict(model, wrapper, processor, india_test, labels, args, device, amp_dtype, force="IN")
        evaluate_and_save(india_test, answers, args.out, "india_country_given")


if __name__ == "__main__":
    main()
