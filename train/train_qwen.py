#!/usr/bin/env python3
"""LoRA fine-tune Qwen2.5-VL (3B by default) to turn a choropleth map into the target JSON, then score it.

    python train_qwen.py --data /kaggle/working/data --mapqa /kaggle/input --out /kaggle/working/qwen
    python train_qwen.py ... --synth /kaggle/input            # also train and score on synthetic maps
    python train_qwen.py ... --skip-train                     # score the base model zero-shot, same prompt
    python train_qwen.py ... --adapter <out>/adapter          # continue training, or score a saved adapter

Writes <out>/adapter (LoRA weights), train_log.jsonl, and predictions_<set>.jsonl + metrics_<set>.json for
each test set (mapqa, plus synth when --synth is given).
"""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

# Set before torch initialises CUDA. Two images per example leave a lot of reserved-but-unused memory on a
# T4; letting the allocator grow its segments avoids failing on fragmentation alone.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from peft import LoraConfig, PeftModel, get_peft_model, set_peft_model_state_dict
from PIL import Image
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from common import (add_common_args, autocast, evaluate_and_save, evaluation_sets, find_mapqa_root,
                    find_saved, set_seed, train_loop, training_records)
from tiles import busiest_tiles

PROMPT = (
    "Read this choropleth map and return only JSON with these keys: "
    '"title"; "legend_title" (null if there is none); "legend_type" ("continuous" or "discrete"); '
    '"data_type" ("percentage" or "number"); '
    '"ranges" (discrete legends only: the legend ranges, lowest first); '
    '"values" (all 50 US states by state code: a number for a continuous legend, '
    "the index of the state's range for a discrete legend, null for no data)."
)
# India maps list 32 states and union territories; the four smallest UTs are drawn but not listed (see
# synth/generate_maps.py). Everything else about the task is the same.
INDIA_PROMPT = PROMPT.replace("all 50 US states by state code",
                              "all 32 Indian states and union territories by state code")
assert INDIA_PROMPT != PROMPT


def prompt_for(record):
    return INDIA_PROMPT if record.get("id", "").startswith("synthind_") else PROMPT


CROP_NOTE = (" The second image is a magnified view of part of the same map; "
             "use it to read the small regions it contains.")
TILES_NOTE = (" The extra images are magnified views of the parts of the same map holding the smallest "
              "regions; use them to read the regions they contain. Each one covers this part of the map, "
              "as a fraction of its width and height:")


def tile_images(image, count, zoom, rng=None, jitter=0):
    """Magnified views of wherever this map's region boundaries are densest.

    The boxes come from the picture alone (train/tiles.py), so nothing here knows which country it is
    looking at - the same rule finds the north-east of a US map and the small states of any other.
    """
    views, boxes = [], []
    for left, top, right, bottom in busiest_tiles(image, tiles=count):
        if jitter and rng is not None:
            shift_x, shift_y = rng.randint(-jitter, jitter), rng.randint(-jitter, jitter)
            left, right = max(0, left + shift_x), min(image.width, right + shift_x)
            top, bottom = max(0, top + shift_y), min(image.height, bottom + shift_y)
        view = image.crop((left, top, right, bottom))
        views.append(view.resize((max(28, round(view.width * zoom)), max(28, round(view.height * zoom))),
                                 Image.BICUBIC))
        boxes.append((left / image.width, top / image.height, right / image.width, bottom / image.height))
    return views, boxes
# Matches the attention and MLP layers of the language model, and the MLP layers of the vision encoder.
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
# Adds the vision encoder's attention and its patch merger. Qwen2.5-VL fuses vision attention into "qkv"
# and "proj", which the names above do not match, so the vision tower's attention is otherwise never trained.
VISION_LORA_REGEX = (r"(.*\.(?:q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|qkv))"
                     r"|(.*\.attn\.proj)|(.*visual\.merger\.mlp\.[02])")


def resized_size(width, height, factor, min_pixels, max_pixels):
    """The size Qwen's processor resizes an image to: each side a multiple of factor, area within bounds."""
    height_out = max(factor, round(height / factor) * factor)
    width_out = max(factor, round(width / factor) * factor)
    if height_out * width_out > max_pixels:
        shrink = math.sqrt(height * width / max_pixels)
        height_out = max(factor, math.floor(height / shrink / factor) * factor)
        width_out = max(factor, math.floor(width / shrink / factor) * factor)
    elif height_out * width_out < min_pixels:
        grow = math.sqrt(min_pixels / (height * width))
        height_out = math.ceil(height * grow / factor) * factor
        width_out = math.ceil(width * grow / factor) * factor
    return width_out, height_out


def two_scale_images(processor, images, scale):
    """Each image at the size the tower normally sees, and the same image `scale` times larger."""
    image_processor = processor.image_processor
    factor = image_processor.patch_size * image_processor.merge_size
    coarse, fine = [], []
    for image in images:
        width, height = resized_size(image.width, image.height, factor,
                                     image_processor.min_pixels, image_processor.max_pixels)
        coarse.append(image.resize((width, height), Image.BICUBIC))
        fine.append(image.resize((width * scale, height * scale), Image.BICUBIC))
    return coarse, fine


def pool_to_coarse(fine, fine_grid, coarse_grid, merge, scale):
    """Average the fine pass's tokens over the area of each coarse token, image by image."""
    pooled, at = [], 0
    for (frames, fine_h, fine_w), (_, coarse_h, coarse_w) in zip(fine_grid.tolist(), coarse_grid.tolist()):
        rows, columns = coarse_h // merge, coarse_w // merge
        count = frames * rows * scale * columns * scale
        block = fine[at: at + count].view(frames, rows, scale, columns, scale, -1)
        at += count
        pooled.append(block.mean(dim=(2, 4)).reshape(frames * rows * columns, -1))
    return torch.cat(pooled)


class S2Visual(torch.nn.Module):
    """The vision tower run at two scales and fused - "scaling on scales".

    The second pass sees the map enlarged, so a state covering a fraction of one 28x28-pixel token in the
    first pass covers several tokens here. Those fine tokens are averaged back onto the coarse grid and
    concatenated with it, so the language model still receives exactly as many tokens as before; each one
    now also carries what its own patch of map looks like close up.

    The fusion starts as a pass-through of the coarse pass (the fine half of its weights is zero), so a
    model trained without S2 behaves identically until training moves those weights.
    """

    def __init__(self, visual, scale):
        super().__init__()
        self.visual = visual
        self.scale = scale
        width = visual.config.out_hidden_size
        self.fuse = torch.nn.Linear(2 * width, width, bias=False)
        with torch.no_grad():
            self.fuse.weight.zero_()
            self.fuse.weight[:, :width].copy_(torch.eye(width))
        self.pending = None

    def __getattr__(self, name):  # dtype, spatial_merge_size, config and friends come from the tower
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._modules["visual"], name)

    def forward(self, hidden_states, grid_thw, **kwargs):
        coarse = self.visual(hidden_states, grid_thw=grid_thw, **kwargs)
        pending, self.pending = self.pending, None
        if pending is None:
            return coarse
        pixels, fine_grid = pending
        with torch.no_grad():  # the tower is frozen here; only the fusion learns
            fine = self.visual(pixels.type(self.visual.dtype), grid_thw=fine_grid, **kwargs)
        pooled = pool_to_coarse(fine, fine_grid, grid_thw, self.visual.spatial_merge_size, self.scale)
        return self.fuse(torch.cat([coarse, pooled.to(coarse.dtype)], dim=-1))


def attach_s2(model, scale, device):
    """Replace the vision tower with the two-scale version, wherever it sits in the wrapped model."""
    for module in model.modules():
        if "visual" in module._modules and not isinstance(module._modules["visual"], S2Visual):
            wrapper = S2Visual(module._modules["visual"], scale)
            wrapper.fuse.to(device=device, dtype=torch.float32).requires_grad_(True)
            module._modules["visual"] = wrapper
            return wrapper
    raise SystemExit("could not find the vision tower to attach S2 to")

def fusion_tiles(image, processor, count, cells, zoom):
    """The busiest tiles, snapped to whole cells of the coarse token grid and cut out at `zoom` times size.

    Each tile covers exactly cells x cells coarse tokens, so its zoom x zoom fine tokens per cell line up
    with the coarse grid without any resampling: the tile's tokens can be added to the coarse tokens that
    already sit at those map positions.
    """
    reader = processor.image_processor
    factor = reader.patch_size * reader.merge_size
    width, height = resized_size(image.width, image.height, factor, reader.min_pixels, reader.max_pixels)
    columns, rows = width // factor, height // factor
    scale_x, scale_y = width / image.width, height / image.height
    placed = []
    for left, top, right, bottom in busiest_tiles(image, tiles=count):
        first_column = min(max(0, round((left + right) / 2 * scale_x / factor - cells / 2)), max(0, columns - cells))
        first_row = min(max(0, round((top + bottom) / 2 * scale_y / factor - cells / 2)), max(0, rows - cells))
        box = (first_column * factor / scale_x, first_row * factor / scale_y,
               (first_column + cells) * factor / scale_x, (first_row + cells) * factor / scale_y)
        side = cells * factor * zoom
        placed.append((first_row, first_column, image.resize((side, side), Image.BICUBIC, box=box)))
    return placed


def fusion_inputs(processor, maps, count, cells, zoom):
    """Pixels for every selected tile in a batch, and where each goes: (batch index, first row, first column)."""
    tiles, where = [], []
    for index, image in maps:
        for row, column, tile in fusion_tiles(image, processor, count, cells, zoom):
            tiles.append(tile)
            where.append((index, row, column))
    if not tiles:
        return {}
    seen = processor.image_processor(images=tiles, return_tensors="pt")
    return {"fx_pixel_values": seen["pixel_values"], "fx_grid_thw": seen["image_grid_thw"],
            "fx_where": torch.tensor(where)}


class FusionVisual(torch.nn.Module):
    """The vision tower, plus close-up detail added to the coarse tokens that sit on the busiest tiles.

    A tile is cut from the map, enlarged `zoom` times and run through the same frozen tower. Its tokens
    come back on a grid `zoom` times finer than the coarse one, so each coarse token has zoom x zoom fine
    tokens over exactly the same piece of map. Those are not averaged (averaging is how a state covering
    one fine token gets diluted to a ninth): each is shrunk, the block is laid side by side, and one
    linear layer maps the block to a correction that is added to the coarse token.

    Every other coarse token is untouched, and the language model still receives exactly the tokens it did
    before. The output layer starts at zero, so an untrained fusion changes nothing.
    """

    def __init__(self, visual, cells, zoom, reduced=256):
        super().__init__()
        self.visual, self.cells, self.zoom = visual, cells, zoom
        width = visual.config.out_hidden_size
        self.reduce = torch.nn.Linear(width, reduced)
        self.out = torch.nn.Linear(zoom * zoom * reduced, width, bias=False)
        with torch.no_grad():
            self.out.weight.zero_()
        self.pending = None

    def __getattr__(self, name):  # dtype, spatial_merge_size, config and friends come from the tower
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._modules["visual"], name)

    def forward(self, hidden_states, grid_thw, **kwargs):
        coarse = self.visual(hidden_states, grid_thw=grid_thw, **kwargs)
        pending, self.pending = self.pending, None
        if pending is None:
            return coarse
        pixels, tile_grid, where = pending
        with torch.no_grad():  # the tower is frozen here; only the fusion learns
            fine = self.visual(pixels.type(self.visual.dtype), grid_thw=tile_grid, **kwargs)
        cells, zoom = self.cells, self.zoom
        tiles = len(where)
        shrunk = self.reduce(fine.float()).view(tiles, cells, zoom, cells, zoom, -1)
        blocks = shrunk.permute(0, 1, 3, 2, 4, 5).reshape(tiles, cells * cells, -1)
        added = self.out(blocks)                                       # (tiles, cells * cells, width)
        merge = self.visual.spatial_merge_size
        rows_columns = [(h // merge, w // merge) for _, h, w in grid_thw.tolist()]
        offsets = [0]
        for rows, columns in rows_columns:
            offsets.append(offsets[-1] + rows * columns)
        delta = torch.zeros(coarse.shape, dtype=added.dtype, device=coarse.device)
        steps = torch.arange(cells, device=coarse.device)
        for k, (image, row, column) in enumerate(where.tolist()):
            index = offsets[image] + (row + steps)[:, None] * rows_columns[image][1] + (column + steps)[None, :]
            delta = delta.index_add(0, index.reshape(-1), added[k])
        return coarse + delta.to(coarse.dtype)


def attach_fusion(model, cells, zoom, device):
    """Replace the vision tower with the fusing version, wherever it sits in the wrapped model."""
    for module in model.modules():
        if "visual" in module._modules and not isinstance(module._modules["visual"], (S2Visual, FusionVisual)):
            wrapper = FusionVisual(module._modules["visual"], cells, zoom)
            wrapper.reduce.to(device=device, dtype=torch.float32).requires_grad_(True)
            wrapper.out.to(device=device, dtype=torch.float32).requires_grad_(True)
            module._modules["visual"] = wrapper
            return wrapper
    raise SystemExit("could not find the vision tower to attach the fusion to")


# Boxes in MapQA's 700x500 albers-usa layout; scaled if a map is a different size. NE_BOX was measured
# from the rendered maps: the small north-eastern fills span x 455-549, y 139-258 in every map.
NE_BOX = (440, 138, 565, 263)    # the small north-eastern states: RI, DE, CT, NJ, MA, NH, VT, MD
BODY_BOX = (100, 130, 560, 400)  # the map body, for crops that land elsewhere


def make_crop(image, rng, train):
    """A magnified crop of the map: biased to the north-east in training, fixed there at evaluation.

    Cropping alone would not help - the crop is enlarged so a state that covers a fraction of one
    28x28-pixel vision token in the full map covers several tokens here.
    """
    width, height = image.size
    scale_x, scale_y = width / 700, height / 500
    if train and rng.random() < 0.25:  # elsewhere in the body, so "image 2 zooms image 1" stays general
        left, top, right, bottom = BODY_BOX
        size = rng.uniform(110, 160)
        centre_x = rng.uniform(left + size / 2, right - size / 2)
        centre_y = rng.uniform(top + size / 2, bottom - size / 2)
        zoom = rng.uniform(2.5, 4.0)
    else:
        left, top, right, bottom = NE_BOX
        jitter = rng.uniform(-15, 15) if train else 0.0
        centre_x, centre_y = (left + right) / 2 + jitter, (top + bottom) / 2 + jitter
        size = rng.uniform(110, 165) if train else 130.0
        zoom = rng.uniform(2.5, 4.0) if train else 3.0
    half = size / 2
    box = (max(0, round((centre_x - half) * scale_x)), max(0, round((centre_y - half) * scale_y)),
           min(width, round((centre_x + half) * scale_x)), min(height, round((centre_y + half) * scale_y)))
    crop = image.crop(box)
    return crop.resize((max(28, round(crop.width * zoom)), max(28, round(crop.height * zoom))), Image.BICUBIC)


def tiles_note(boxes):
    """Say where each tile was cut from, so the model can place what it reads back on the map.

    Without this the extra images are just more pixels: the model can see that a region is dark but not
    which region it is, and has to match the tile to the map by eye before it can use it.
    """
    places = [f" view {i} x {left:.2f}-{right:.2f}, y {top:.2f}-{bottom:.2f}."
              for i, (left, top, right, bottom) in enumerate(boxes, start=2)]
    return TILES_NOTE + "".join(places)


def chat_text(processor, target=None, images=1, boxes=None, prompt=PROMPT):
    content = [{"type": "image"} for _ in range(images)]
    if boxes:
        note = tiles_note(boxes)
    else:
        note = "" if images == 1 else (CROP_NOTE if images == 2 else TILES_NOTE)
    content.append({"type": "text", "text": prompt + note})
    messages = [{"role": "user", "content": content}]
    if target is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": target}]})
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=target is None)


class CropDataset(torch.utils.data.Dataset):
    """The full map, and on most examples a magnified crop as a second image.

    A share of examples keep the map alone (--crop-dropout), which keeps single-image inference in
    distribution and lets the crop be ablated at scoring time instead of in a second training run.
    """

    def __init__(self, records, dropout, tiles=0, tile_zoom=3.0, fuse=False):
        self.records, self.dropout = records, dropout
        self.tiles, self.tile_zoom, self.fuse = tiles, tile_zoom, fuse

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        record = self.records[i]
        image = Image.open(record["image_path"]).convert("RGB")
        boxes, fuse = [], False
        if random.random() < self.dropout:
            images = [image]                       # the map alone, so single-image use stays in distribution
        elif self.tiles and self.fuse:
            images, fuse = [image], True           # the tiles are added into the map's own tokens
        elif self.tiles:
            views, boxes = tile_images(image, self.tiles, self.tile_zoom, random, jitter=8)
            images = [image] + views
        else:
            images = [image, make_crop(image, random, train=True)]
        return images, boxes, record["target"], fuse, prompt_for(record)


class TrainCollate:
    """Image(s) + prompt + answer, with labels only on the answer tokens."""

    def __init__(self, processor, fine_processor=None, scale=2, fusion=None):
        self.processor = processor
        self.fusion = fusion  # (tiles, cells, zoom) when tiles are fused into the map's tokens
        self.fine_processor = fine_processor  # same processor, allowed more pixels, for the second scale
        self.scale = scale
        self.answer_start = processor.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)

    def __call__(self, batch):
        texts, images = [], []
        fused = []
        for index, (sample_images, boxes, target, fuse, prompt) in enumerate(batch):
            if fuse:
                fused.append((index, sample_images[0]))
            images.extend(sample_images)
            texts.append(chat_text(self.processor, target, len(sample_images), boxes, prompt))
        fine = None
        if self.fine_processor is not None:
            images, fine_images = two_scale_images(self.processor, images, self.scale)
            fine = self.fine_processor.image_processor(images=fine_images, return_tensors="pt")
        enc = self.processor(text=texts, images=images, padding=True, return_tensors="pt")
        if fine is not None:
            enc["s2_pixel_values"], enc["s2_grid_thw"] = fine["pixel_values"], fine["image_grid_thw"]
        if fused:
            for key, value in fusion_inputs(self.processor, fused, *self.fusion).items():
                enc[key] = value
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        marker = torch.tensor(self.answer_start)
        for row, ids in enumerate(enc["input_ids"]):
            hits = (ids.unfold(0, len(marker), 1) == marker).all(dim=1).nonzero()
            labels[row, : int(hits[-1]) + len(marker)] = -100
        enc["labels"] = labels
        return enc


def predict(model, processor, records, args, device, amp_dtype, crop=False, s2=None, fine_processor=None,
            tiles=0, fusion=None):
    def generate(batch):
        """Predictions for one batch; on running out of GPU memory, the batch is split in two and retried,
        so a long run is not lost to one oversized batch at the very end."""
        try:
            return generate_batch(batch)
        except torch.cuda.OutOfMemoryError:
            if len(batch) == 1:
                raise
            torch.cuda.empty_cache()
            half = len(batch) // 2
            print(f"out of memory on a batch of {len(batch)}; retrying in halves", flush=True)
            return generate(batch[:half]) + generate(batch[half:])

    def generate_batch(batch):
        texts, images, fused = [], [], []
        for record in batch:
            image = Image.open(record["image_path"]).convert("RGB")
            images.append(image)
            extra, boxes = [], []
            if tiles and fusion is not None:
                fused.append((len(texts), image))
            elif tiles:
                extra, boxes = tile_images(image, tiles, args.tile_zoom)
            elif crop:
                extra = [make_crop(image, random, train=False)]
            images.extend(extra)
            texts.append(chat_text(processor, images=1 + len(extra), boxes=boxes, prompt=prompt_for(record)))
        fine = None
        if fine_processor is not None:
            images, fine_images = two_scale_images(processor, images, args.s2_scale)
            fine = fine_processor.image_processor(images=fine_images, return_tensors="pt")
        enc = processor(text=texts, images=images, padding=True, return_tensors="pt").to(device)
        if fused:
            seen = fusion_inputs(processor, fused, tiles, args.fusion_cells, args.fusion_zoom)
            fusion.pending = (seen["fx_pixel_values"].to(device), seen["fx_grid_thw"].to(device),
                              seen["fx_where"].to(device))
        if fine is not None:
            s2.pending = (fine["pixel_values"].to(device), fine["image_grid_thw"].to(device))
        with autocast(device, amp_dtype):
            ids = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 repetition_penalty=1.0, use_cache=True)
        return processor.batch_decode(ids[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    predictions = []
    began = time.time()
    with torch.no_grad():
        for start in range(0, len(records), args.eval_batch_size):
            predictions += generate(records[start : start + args.eval_batch_size])
            print(f"predicted {len(predictions)}/{len(records)} ({(time.time() - began) / 60:.1f} min)", flush=True)
    return predictions


def report_lora(model):
    """Print where LoRA actually landed. PEFT matches module names, so a target that matches nothing is
    silently a no-op; this run's whole question is whether vision attention helps, so it must not pass quietly."""
    names = [name for name, _ in model.named_modules() if name.endswith("lora_A")]
    vision_attention = sum(1 for n in names if ".visual." in n and ".attn." in n)
    vision_mlp = sum(1 for n in names if ".visual." in n and ".mlp." in n and "merger" not in n)
    merger = sum(1 for n in names if "merger" in n)
    print(f"LoRA modules: {len(names)} total | vision attention {vision_attention} | "
          f"vision MLP {vision_mlp} | merger {merger} | language {len(names) - vision_attention - vision_mlp - merger}",
          flush=True)
    return vision_attention


def load_prior_adapter(model, path):
    """Load an earlier adapter into a model whose LoRA now covers more modules.

    LoRA initialises its B matrix to zero, so modules the earlier run did not train start as exact
    no-ops and the model begins exactly where that run ended.
    """
    from safetensors.torch import load_file

    state = load_file(str(Path(path) / "adapter_model.safetensors"))
    result = set_peft_model_state_dict(model, state)
    unexpected = list(getattr(result, "unexpected_keys", []))
    if unexpected:
        raise SystemExit(f"prior adapter does not line up: {len(unexpected)} unexpected keys, "
                         f"e.g. {unexpected[:3]}")
    print(f"loaded {len(state)} tensors from the prior adapter", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--adapter", help="a saved LoRA adapter to continue from or to score")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--max-pixels", type=int, default=640_000,
                        help="larger images are scaled down to this many pixels (MapQA maps have 350,000)")
    parser.add_argument("--vision-lora", action="store_true",
                        help="also train the vision encoder's attention and patch merger, not just its MLPs")
    parser.add_argument("--crop", action="store_true",
                        help="pass a magnified crop of the map as a second image")
    parser.add_argument("--crop-dropout", type=float, default=0.2,
                        help="share of training examples that keep the full map alone")
    parser.add_argument("--eval-both", action="store_true",
                        help="with --crop, also score without the crop, to separate its effect")
    parser.add_argument("--tiles", type=int, default=0,
                        help="magnify this many of the map's most crowded areas and pass them as extra "
                             "images; the areas are found from the picture, not from known geography")
    parser.add_argument("--tile-zoom", type=float, default=3.0, help="how much to magnify each tile")
    parser.add_argument("--fusion", action="store_true",
                        help="with --tiles: add each tile's close-up tokens into the map's own tokens instead of "
                             "passing the tile as an extra image, so the token count does not grow")
    parser.add_argument("--fusion-cells", type=int, default=3,
                        help="a fused tile covers this many coarse tokens on each side")
    parser.add_argument("--fusion-zoom", type=int, default=3,
                        help="whole-number magnification of a fused tile (fine tokens per coarse token, per side)")
    parser.add_argument("--s2", action="store_true",
                        help="run the vision tower at two scales and fuse them, instead of cropping by hand")
    parser.add_argument("--s2-scale", type=int, default=2,
                        help="how many times larger the second pass sees the map")
    # A T4 trains about 2.3 s per MapQA map; 8 h leaves room for scoring inside Kaggle's 12-hour limit
    parser.set_defaults(epochs=1, hours=8.0, lr=2e-4, batch_size=2, grad_accum=8, eval_batch_size=16,
                        max_new_tokens=600)
    args = parser.parse_args()
    set_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "args.json").write_text(json.dumps(vars(args), default=str, indent=1))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":  # T4 and P100 have no bfloat16
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    else:
        dtype = torch.float32
    amp_dtype = dtype if device.type == "cuda" else None
    root = find_mapqa_root(args.mapqa)
    processor = AutoProcessor.from_pretrained(args.model, max_pixels=args.max_pixels)
    processor.tokenizer.padding_side = "left"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model, torch_dtype=dtype,
                                                               attn_implementation="sdpa").to(device)
    targets = VISION_LORA_REGEX if args.vision_lora else LORA_TARGETS
    lora = LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
                      target_modules=targets, task_type="CAUSAL_LM")
    if args.adapter and args.vision_lora:
        # the saved adapter covers fewer modules than this config, so build the wider one and fill it in
        model = get_peft_model(model, lora)
        load_prior_adapter(model, find_saved(args.adapter, "adapter_config.json"))
    elif args.adapter:
        model = PeftModel.from_pretrained(model, find_saved(args.adapter, "adapter_config.json"),
                                          is_trainable=not args.skip_train)
    elif not args.skip_train:
        model = get_peft_model(model, lora)
    if args.adapter or not args.skip_train:
        vision_attention = report_lora(model)
        if args.vision_lora and not vision_attention:
            raise SystemExit("--vision-lora matched no vision attention; check VISION_LORA_REGEX")

    if args.fusion and (args.s2 or not args.tiles):
        raise SystemExit("--fusion needs --tiles N and cannot be combined with --s2")
    fusion = None
    if args.fusion:
        fusion = attach_fusion(model, args.fusion_cells, args.fusion_zoom, device)
        added = sum(p.numel() for p in (*fusion.reduce.parameters(), *fusion.out.parameters()))
        print(f"fusion: {args.tiles} tiles of {args.fusion_cells}x{args.fusion_cells} coarse tokens at "
              f"{args.fusion_zoom}x, {added:,} trainable weights", flush=True)

    s2 = fine_processor = None
    if args.s2:
        s2 = attach_s2(model, args.s2_scale, device)
        fine_processor = AutoProcessor.from_pretrained(args.model,
                                                       max_pixels=args.max_pixels * args.s2_scale ** 2)
        print(f"S2: second pass at {args.s2_scale}x the size, fusing into "
              f"{s2.fuse.weight.numel():,} trainable weights", flush=True)

    if not args.skip_train:
        model.print_trainable_parameters()
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        train = training_records(args, root)
        dataset = CropDataset(train, args.crop_dropout if (args.crop or args.tiles) else 1.0,
                              args.tiles, args.tile_zoom, fuse=args.fusion)
        collate = TrainCollate(processor, fine_processor, args.s2_scale,
                               (args.tiles, args.fusion_cells, args.fusion_zoom) if args.fusion else None)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                            collate_fn=collate)

        def forward(batch):
            batch = batch.to(device)
            if s2 is not None:
                s2.pending = (batch.pop("s2_pixel_values"), batch.pop("s2_grid_thw"))
            if fusion is not None:  # absent when every example in the batch kept the map alone
                pixels = batch.pop("fx_pixel_values", None)
                grid, where = batch.pop("fx_grid_thw", None), batch.pop("fx_where", None)
                fusion.pending = None if pixels is None else (pixels, grid, where)
            return model(**batch).loss

        print(f"{len(loader)} batches per epoch", flush=True)
        train_loop(model, loader, args, forward, args.out, amp_dtype)
        model.save_pretrained(args.out / "adapter")
        if s2 is not None:  # the fusion is not part of the LoRA adapter, so it is saved beside it
            torch.save(s2.fuse.state_dict(), args.out / "s2_fuse.pt")
        if fusion is not None:
            torch.save({"reduce": fusion.reduce.state_dict(), "out": fusion.out.state_dict()},
                       args.out / "fusion.pt")

    model.eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()  # release memory held from training before generating
    variants = [("", args.crop or bool(args.tiles))]
    if (args.crop or args.tiles) and args.eval_both:
        variants.append(("_nocrop", False))  # same weights, extra views withheld: their effect on its own
    for name, records in evaluation_sets(args, root):
        for suffix, use_crop in variants:
            print(f"scoring {len(records)} {name} test maps{' without the crop' if suffix else ''}", flush=True)
            predictions = predict(model, processor, records, args, device, amp_dtype, crop=use_crop,
                                  s2=s2, fine_processor=fine_processor,
                                  tiles=args.tiles if use_crop else 0, fusion=fusion)
            evaluate_and_save(records, predictions, args.out, name + suffix)


if __name__ == "__main__":
    main()
