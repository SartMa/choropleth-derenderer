#!/usr/bin/env python3
"""Fine-tune DePlot (google/deplot) to turn a choropleth map into the target JSON, then score it on test maps.

    python train_deplot.py --data /kaggle/working/data --mapqa /kaggle/input --out /kaggle/working/deplot
    python train_deplot.py ... --synth /kaggle/input     # also train and score on synthetic maps

Writes <out>/model (fine-tuned model), train_log.jsonl, and predictions_<set>.jsonl + metrics_<set>.json for
each test set (mapqa, plus synth when --synth is given).
"""

import argparse
import json
import time

import torch
from torch.utils.data import DataLoader
from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor

from common import (MapDataset, add_common_args, as_list, autocast, evaluate_and_save, evaluation_sets,
                    find_mapqa_root, find_saved, set_seed, train_loop, training_records)

PROMPT = "Generate underlying data table of the figure below:"  # the prompt DePlot was trained with


class TrainCollate:
    def __init__(self, processor, max_patches, max_target_tokens):
        self.processor = processor
        self.max_patches = max_patches
        self.max_target_tokens = max_target_tokens

    def __call__(self, batch):
        images, targets = zip(*batch)
        enc = self.processor(images=list(images), text=[PROMPT] * len(images), max_patches=self.max_patches,
                             return_tensors="pt")
        tokenizer = self.processor.tokenizer
        labels = tokenizer(list(targets), max_length=self.max_target_tokens, truncation=True, padding="longest",
                           return_tensors="pt").input_ids
        labels[labels == tokenizer.pad_token_id] = -100
        return enc["flattened_patches"], enc["attention_mask"], labels


def predict(model, processor, records, args, device, amp_dtype):
    loader = DataLoader(MapDataset(records), batch_size=args.eval_batch_size, num_workers=args.workers,
                        collate_fn=as_list)
    predictions = []
    began = time.time()
    with torch.no_grad():
        for batch in loader:
            images = [image for image, _ in batch]
            enc = processor(images=images, text=[PROMPT] * len(images), max_patches=args.max_patches,
                            return_tensors="pt").to(device)
            with autocast(device, amp_dtype):
                ids = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False, num_beams=1)
            predictions += processor.batch_decode(ids, skip_special_tokens=True)
            print(f"predicted {len(predictions)}/{len(records)} ({(time.time() - began) / 60:.1f} min)", flush=True)
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--model", default="google/deplot")
    parser.add_argument("--max-patches", type=int, default=2048, help="a 700x500 map fits in 2048 patches at full size")
    parser.add_argument("--max-target-tokens", type=int, default=640, help="longest MapQA target is about 530 tokens")
    parser.add_argument("--fp16", action="store_true", help="mixed precision (off by default: T5-style models can overflow)")
    # Pix2Struct has no gradient checkpointing and 2 maps per batch ran out of memory on a 16 GB T4;
    # the effective batch is still 16
    parser.set_defaults(epochs=3, lr=5e-5, batch_size=1, grad_accum=16)
    args = parser.parse_args()
    set_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "args.json").write_text(json.dumps(vars(args), default=str, indent=1))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.float16 if args.fp16 and device.type == "cuda" else None
    root = find_mapqa_root(args.mapqa)
    model_path = find_saved(args.model, "config.json")  # a model id, or a folder holding a saved model
    processor = Pix2StructProcessor.from_pretrained(model_path)
    model = Pix2StructForConditionalGeneration.from_pretrained(model_path).to(device)

    if not args.skip_train:
        train = training_records(args, root)
        loader = DataLoader(MapDataset(train), batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                            collate_fn=TrainCollate(processor, args.max_patches, args.max_target_tokens))

        def forward(batch):
            patches, mask, labels = (t.to(device) for t in batch)
            return model(flattened_patches=patches, attention_mask=mask, labels=labels).loss

        if model.supports_gradient_checkpointing:
            model.gradient_checkpointing_enable()
        print(f"{len(loader)} batches per epoch", flush=True)
        train_loop(model, loader, args, forward, args.out, amp_dtype)
        model.save_pretrained(args.out / "model")
        processor.save_pretrained(args.out / "model")

    model.eval()
    if device.type == "cuda":
        torch.cuda.empty_cache()  # release memory held from training before generating
    for name, records in evaluation_sets(args, root):
        print(f"scoring {len(records)} {name} test maps", flush=True)
        evaluate_and_save(records, predict(model, processor, records, args, device, amp_dtype), args.out, name)


if __name__ == "__main__":
    main()
