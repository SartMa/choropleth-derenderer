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

PROMPT = (
    "Read this choropleth map and return only JSON with these keys: "
    '"title"; "legend_title" (null if there is none); "legend_type" ("continuous" or "discrete"); '
    '"data_type" ("percentage" or "number"); '
    '"ranges" (discrete legends only: the legend ranges, lowest first); '
    '"values" (all 50 US states by state code: a number for a continuous legend, '
    "the index of the state's range for a discrete legend, null for no data)."
)
CROP_NOTE = (" The second image is a magnified view of part of the same map; "
             "use it to read the small regions it contains.")
# Matches the attention and MLP layers of the language model, and the MLP layers of the vision encoder.
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
# Adds the vision encoder's attention and its patch merger. Qwen2.5-VL fuses vision attention into "qkv"
# and "proj", which the names above do not match, so the vision tower's attention is otherwise never trained.
VISION_LORA_REGEX = (r"(.*\.(?:q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj|qkv))"
                     r"|(.*\.attn\.proj)|(.*visual\.merger\.mlp\.[02])")

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


def chat_text(processor, target=None, images=1):
    content = [{"type": "image"} for _ in range(images)]
    content.append({"type": "text", "text": PROMPT + (CROP_NOTE if images > 1 else "")})
    messages = [{"role": "user", "content": content}]
    if target is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": target}]})
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=target is None)


class CropDataset(torch.utils.data.Dataset):
    """The full map, and on most examples a magnified crop as a second image.

    A share of examples keep the map alone (--crop-dropout), which keeps single-image inference in
    distribution and lets the crop be ablated at scoring time instead of in a second training run.
    """

    def __init__(self, records, dropout):
        self.records, self.dropout = records, dropout

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        record = self.records[i]
        image = Image.open(record["image_path"]).convert("RGB")
        images = [image] if random.random() < self.dropout else [image, make_crop(image, random, train=True)]
        return images, record["target"]


class TrainCollate:
    """Image(s) + prompt + answer, with labels only on the answer tokens."""

    def __init__(self, processor):
        self.processor = processor
        self.answer_start = processor.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)

    def __call__(self, batch):
        texts, images = [], []
        for sample_images, target in batch:
            images.extend(sample_images)
            texts.append(chat_text(self.processor, target, len(sample_images)))
        enc = self.processor(text=texts, images=images, padding=True, return_tensors="pt")
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        marker = torch.tensor(self.answer_start)
        for row, ids in enumerate(enc["input_ids"]):
            hits = (ids.unfold(0, len(marker), 1) == marker).all(dim=1).nonzero()
            labels[row, : int(hits[-1]) + len(marker)] = -100
        enc["labels"] = labels
        return enc


def predict(model, processor, records, args, device, amp_dtype, crop=False):
    predictions = []
    began = time.time()
    with torch.no_grad():
        for start in range(0, len(records), args.eval_batch_size):
            batch = records[start : start + args.eval_batch_size]
            texts, images = [], []
            for record in batch:
                image = Image.open(record["image_path"]).convert("RGB")
                images.append(image)
                if crop:
                    images.append(make_crop(image, random, train=False))
                texts.append(chat_text(processor, images=2 if crop else 1))
            enc = processor(text=texts, images=images, padding=True, return_tensors="pt").to(device)
            with autocast(device, amp_dtype):
                ids = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                     repetition_penalty=1.0, use_cache=True)
            predictions += processor.batch_decode(ids[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
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

    if not args.skip_train:
        model.print_trainable_parameters()
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        train = training_records(args, root)
        dataset = CropDataset(train, args.crop_dropout if args.crop else 1.0)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                            collate_fn=TrainCollate(processor))

        def forward(batch):
            return model(**batch.to(device)).loss

        print(f"{len(loader)} batches per epoch", flush=True)
        train_loop(model, loader, args, forward, args.out, amp_dtype)
        model.save_pretrained(args.out / "adapter")

    model.eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()  # release memory held from training before generating
    variants = [("", args.crop)]
    if args.crop and args.eval_both:
        variants.append(("_nocrop", False))  # same weights, crop withheld: the crop's effect on its own
    for name, records in evaluation_sets(args, root):
        for suffix, use_crop in variants:
            print(f"scoring {len(records)} {name} test maps{' without the crop' if suffix else ''}", flush=True)
            predictions = predict(model, processor, records, args, device, amp_dtype, crop=use_crop)
            evaluate_and_save(records, predictions, args.out, name + suffix)


if __name__ == "__main__":
    main()
