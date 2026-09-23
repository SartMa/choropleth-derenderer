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
import time

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from PIL import Image
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from common import (MapDataset, add_common_args, autocast, evaluate_and_save, evaluation_sets, find_mapqa_root,
                    find_saved, set_seed, train_loop, training_records)

PROMPT = (
    "Read this choropleth map and return only JSON with these keys: "
    '"title"; "legend_title" (null if there is none); "legend_type" ("continuous" or "discrete"); '
    '"data_type" ("percentage" or "number"); '
    '"ranges" (discrete legends only: the legend ranges, lowest first); '
    '"values" (all 50 US states by state code: a number for a continuous legend, '
    "the index of the state's range for a discrete legend, null for no data)."
)
# Matches the attention and MLP layers of the language model, and the MLP layers of the vision encoder.
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def chat_text(processor, target=None):
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    if target is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": target}]})
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=target is None)


class TrainCollate:
    """Image + prompt + answer, with labels only on the answer tokens."""

    def __init__(self, processor):
        self.processor = processor
        self.answer_start = processor.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)

    def __call__(self, batch):
        images, targets = zip(*batch)
        enc = self.processor(text=[chat_text(self.processor, t) for t in targets], images=list(images),
                             padding=True, return_tensors="pt")
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        marker = torch.tensor(self.answer_start)
        for row, ids in enumerate(enc["input_ids"]):
            hits = (ids.unfold(0, len(marker), 1) == marker).all(dim=1).nonzero()
            labels[row, : int(hits[-1]) + len(marker)] = -100
        enc["labels"] = labels
        return enc


def predict(model, processor, records, args, device, amp_dtype):
    predictions = []
    began = time.time()
    with torch.no_grad():
        for start in range(0, len(records), args.eval_batch_size):
            batch = records[start : start + args.eval_batch_size]
            images = [Image.open(r["image_path"]).convert("RGB") for r in batch]
            enc = processor(text=[chat_text(processor)] * len(batch), images=images, padding=True,
                            return_tensors="pt").to(device)
            with autocast(device, amp_dtype):
                ids = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                     repetition_penalty=1.0, use_cache=True)
            predictions += processor.batch_decode(ids[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            print(f"predicted {len(predictions)}/{len(records)} ({(time.time() - began) / 60:.1f} min)", flush=True)
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--adapter", help="a saved LoRA adapter to continue from or to score")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--max-pixels", type=int, default=640_000,
                        help="larger images are scaled down to this many pixels (MapQA maps have 350,000)")
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
    if args.adapter:
        model = PeftModel.from_pretrained(model, find_saved(args.adapter, "adapter_config.json"),
                                          is_trainable=not args.skip_train)
    elif not args.skip_train:
        model = get_peft_model(model, LoraConfig(r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05,
                                                 target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))

    if not args.skip_train:
        model.print_trainable_parameters()
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        train = training_records(args, root)
        loader = DataLoader(MapDataset(train), batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                            collate_fn=TrainCollate(processor))

        def forward(batch):
            return model(**batch.to(device)).loss

        print(f"{len(loader)} batches per epoch", flush=True)
        train_loop(model, loader, args, forward, args.out, amp_dtype)
        model.save_pretrained(args.out / "adapter")

    model.eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()  # release memory held from training before generating
    for name, records in evaluation_sets(args, root):
        print(f"scoring {len(records)} {name} test maps", flush=True)
        evaluate_and_save(records, predict(model, processor, records, args, device, amp_dtype), args.out, name)


if __name__ == "__main__":
    main()
