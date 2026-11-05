"""Read real maps with the sheet adapter. The sheets are built beforehand; this only runs the model.

    python read_real.py --sheets /kaggle/input --adapter /kaggle/input --out /kaggle/working/real
"""

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from common import autocast, find_saved
from sheet import SHEET_PROMPT
from train_qwen import INDIA_PROMPT, chat_text


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sheets", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--max-pixels", type=int, default=640000)
    parser.add_argument("--max-new-tokens", type=int, default=600)
    parser.add_argument("--plain", action="store_true", help="the map alone with the plain India prompt, as the old model reads it")
    args, _ = parser.parse_known_args()
    args.out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    processor = AutoProcessor.from_pretrained(args.model, max_pixels=args.max_pixels)
    processor.tokenizer.padding_side = "left"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model, torch_dtype=dtype,
                                                               attn_implementation="sdpa").to(device)
    model = PeftModel.from_pretrained(model, find_saved(args.adapter, "adapter_config.json"), is_trainable=False).eval()
    records_file = next(Path(args.sheets).rglob("records.jsonl"))
    root = records_file.parent
    with open(args.out / "predictions_real.jsonl", "w") as out, torch.no_grad():
        for line in open(records_file):
            record = json.loads(line)
            keys = ("map",) if args.plain else ("map", "sheet")
            images = [Image.open(root / record[k]).convert("RGB") for k in keys]
            text = chat_text(processor, images=len(images), prompt=INDIA_PROMPT if args.plain else SHEET_PROMPT)
            enc = processor(text=[text], images=images, padding=True, return_tensors="pt").to(device)
            with autocast(device, dtype):
                ids = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True)
            answer = processor.batch_decode(ids[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)[0]
            out.write(json.dumps({"id": record["id"], "prediction": answer}) + "\n")
            out.flush()
            print(record["id"], flush=True)


if __name__ == "__main__":
    main()
