#!/usr/bin/env python3
"""Plumbing test of the v3 training and reading code on a tiny random Qwen2.5-VL, on CPU, in a few minutes.

Checks that a training step reaches both the region head and the LoRA, that reading works in both modes, and that
training stopped part-way (--halt-after) resumes from its checkpoint in exactly the data order of an uninterrupted
run. The numbers it prints mean nothing (the model is random); only the checks matter.

Needs the Qwen processor files (downloaded from Hugging Face on first use), the label files and a few synthetic
India maps:

    python lab/tiny_test.py --labels data/answer_v2_inputs --india data/synth_ind
"""

import argparse
import shutil
import sys
import tempfile
import types
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from PIL import Image
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))
import answer_sheet as A  # noqa: E402
import answer_v3 as V  # noqa: E402
from common import find_images_root, read_split, train_loop  # noqa: E402

MAX_PIXELS = 100_000


def tiny_model():
    config = Qwen2_5_VLConfig.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    vision = config.vision_config
    vision.depth, vision.hidden_size, vision.intermediate_size, vision.num_heads, vision.out_hidden_size = 4, 64, 128, 2, 64
    vision.fullatt_block_indexes = [1, 3]
    text = config.text_config if hasattr(config, "text_config") else config
    for c in {id(config): config, id(text): text}.values():
        c.hidden_size, c.intermediate_size, c.num_hidden_layers = 64, 128, 2
        c.num_attention_heads, c.num_key_value_heads = 4, 2
        c.rope_scaling = {"type": "default", "rope_type": "default", "mrope_section": [2, 3, 3]}
    torch.manual_seed(0)
    model = Qwen2_5_VLForConditionalGeneration(config)
    return get_peft_model(model, LoraConfig(r=4, target_modules=["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
                                            task_type="CAUSAL_LM"))


def settings(**extra):
    base = dict(mode="zoom", max_pixels=MAX_PIXELS, small_mass=3.0, target_patches=8.0, min_zoom=2.0, max_zoom=8.0,
                view_extent=0.36, min_peak=0.0, jitter=0.15, max_views=2, zoom_share=0.5, head_minutes=0.05, seed=3,
                eval_batch_size=2, max_new_tokens=8, epochs=2, hours=1.0, lr=1e-4, weight_decay=0.01, grad_accum=2,
                log_every=1000, checkpoint_every=0.0001, resume=False, halt_after=None)
    base.update(extra)
    return types.SimpleNamespace(**base)


class WithIds(V.TrainSet):
    """Also returns each map's id, to check the order maps are seen in."""

    def __getitem__(self, i):
        return super().__getitem__(i) + (self.records[i]["id"],)


class CollateIds(V.Collate):
    def __call__(self, batch):
        enc, extra = super().__call__([item[:6] for item in batch])
        extra["ids"] = [item[6] for item in batch]
        return enc, extra


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--labels", type=Path, required=True, help="folder with vocab.json, regions*.json and real/")
    parser.add_argument("--india", type=Path, required=True, help="folder with the synthetic India maps")
    args = parser.parse_args()
    failures = []

    def check(name, ok):
        print(f"{'ok  ' if ok else 'FAIL'} {name}", flush=True)
        if not ok:
            failures.append(name)

    device = torch.device("cpu")
    model = tiny_model()
    labels = A.Labels(args.labels)
    wrapper = V.attach(model, len(labels.vocab), device)
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", max_pixels=MAX_PIXELS, min_pixels=28 * 28)
    processor.tokenizer.padding_side = "left"
    root = find_images_root(args.india, "synthind_*")
    india = read_split(root, "test", root)[:12]

    # one training step, both modes
    for mode in ("zoom", "s2"):
        s = settings(mode=mode)
        dataset = V.TrainSet(india, labels, s)
        enc, extra = V.Collate(processor, len(labels.vocab))([dataset[0], dataset[1]])
        wrapper.pending = extra
        model.train()
        loss = model(**enc).loss + 0.5 * wrapper.aux_loss
        loss.backward()
        head = sum(float(p.grad.abs().sum()) for p in wrapper.head.parameters() if p.grad is not None)
        lora = sum(float(p.grad.abs().sum()) for n, p in model.named_parameters() if "lora_" in n and p.grad is not None)
        check(f"{mode}: a training step reaches the head and the LoRA", head > 0 and lora > 0)
        model.zero_grad()
        image = Image.open(india[0]["image_path"]).convert("RGB")
        coarse = A.map_grid(wrapper, processor, image, device, None)
        fine = V.fine_grid(wrapper, processor, image, device, None)
        check(f"{mode}: the S2 grid has twice the detail", tuple(fine.shape[:2]) == (2 * coarse.shape[0], 2 * coarse.shape[1]))
        model.eval()
        answers, countries, stats = V.predict(model, wrapper, processor, india[:2], labels, s, device, None)
        check(f"{mode}: reading returns an answer per map", len(answers) == 2 and len(countries) == 2)

    # stop part-way and resume
    for n, p in model.named_parameters():
        p.requires_grad_("lora_" in n or ".head." in n)
    seen = []

    def forward(batch):
        enc, extra = batch
        seen.extend(extra.pop("ids"))
        wrapper.pending = extra
        return model(**enc).loss + 0.5 * wrapper.aux_loss

    s = settings()
    loader = DataLoader(WithIds(india, labels, s), batch_size=1, shuffle=True, num_workers=0,
                        collate_fn=CollateIds(processor, len(labels.vocab)))
    out = Path(tempfile.mkdtemp())
    start = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
    model.train()
    updates = train_loop(model, loader, s, forward, out, None)
    uninterrupted = list(seen)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in start:
                p.copy_(start[n])
    (out / "checkpoint.pt").unlink()
    seen.clear()
    s.halt_after = 5
    try:
        train_loop(model, loader, s, forward, out, None)
        check("--halt-after stops training", False)
    except SystemExit:
        check("--halt-after stops training", True)
    before = list(seen)
    seen.clear()
    s.halt_after, s.resume = None, True
    resumed = train_loop(model, loader, s, forward, out, None)
    check(f"resumed run reaches the same {updates} updates", resumed == updates)
    check("resumed run sees the maps in the uninterrupted order, none twice",
          before + seen == uninterrupted and len(before) > 0 and len(seen) > 0)
    seen.clear()
    check("a finished run returns at once on --resume", train_loop(model, loader, s, forward, out, None) == updates
          and not seen)
    shutil.rmtree(out, ignore_errors=True)

    print("all checks passed" if not failures else f"{len(failures)} checks FAILED", flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
