"""Shared pieces of the derenderer training scripts: loading maps, the training loop, and scoring.

Training data is MapQA (from convert_mapqa.py) plus, with --synth, synthetic maps from
synth/generate_maps.py. Both use the same target JSON.

Scoring (per legend type; MapQA and synthetic test maps are scored separately):
    parsed                 output is a JSON object
    exact_match            output equals the target exactly
    title_correct          title matches (ignoring case and spacing)
    legend_title_correct   legend title matches (ignoring case and spacing; both null also counts)
    legend_type_correct    "continuous" / "discrete" is right
    data_type_correct      "percentage" / "number" is right
    coverage               share of the 50 states present in "values"
    extra_keys             keys in "values" that are not states of this map
  discrete maps:
    ranges_correct         the legend ranges match exactly
    state_accuracy         share of states with the right range index (null for no data)
  continuous maps (errors are relative to the true value range, so 0.05 = 5% of the range):
    nmae, nrmse            mean / root-mean-square error; a missing or non-numeric state counts as 1,
                           and no single error counts more than 1
    within_5pct            share of states within 5% of the range
Maps whose output doesn't parse count as 0 for accuracies and 1 for errors.
"""

import argparse
import contextlib
import json
import math
import random
import time
from pathlib import Path

import torch
from PIL import Image

def argparse_suppress():
    """Help text for testing-only options: hidden from --help."""
    return argparse.SUPPRESS


def add_common_args(parser):
    parser.add_argument("--data", type=Path, required=True, help="folder with MapQA train.jsonl and test.jsonl")
    parser.add_argument("--mapqa", type=Path, required=True, help="folder containing MapQA_S/images (searched a few levels)")
    parser.add_argument("--synth", type=Path, help="folder containing the synthetic maps (train.jsonl next to "
                                                   "images/synth_*, searched a few levels); adds them to training "
                                                   "and scoring")
    parser.add_argument("--india", type=Path, help="folder containing the synthetic India maps (images/synthind_*); "
                                                   "adds a few to training and scores an India test set")
    parser.add_argument("--india-train-maps", type=int, default=1000, help="India maps mixed into training")
    parser.add_argument("--india-eval-maps", type=int, default=100, help="India test maps to score")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-maps", type=int, help="train on a random subset of this many MapQA maps")
    parser.add_argument("--synth-train-maps", type=int, help="train on a random subset of this many synthetic maps")
    parser.add_argument("--eval-maps", type=int, default=400, help="MapQA test maps to score (same maps for every model)")
    parser.add_argument("--synth-eval-maps", type=int, default=200, help="synthetic test maps to score")
    parser.add_argument("--eval-limit", type=int, help="score only the first N sampled test maps; a prefix of the "
                                                       "usual set, so results stay comparable with earlier runs")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--hours", type=float, default=8.0,
                        help="training time budget; the learning-rate schedule is shortened to fit it")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=640, help="longest MapQA target is about 530 tokens")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--skip-train", action="store_true", help="only score the model as loaded")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-every", type=float, default=0,
                        help="minutes between training checkpoints (out/checkpoint.pt); 0 = none")
    parser.add_argument("--resume", action="store_true", help="continue from out/checkpoint.pt if it exists")
    parser.add_argument("--snapshot-every", type=float, default=0,
                        help="hours between kept copies of the model (out/snapshots/); 0 = none")
    parser.add_argument("--halt-after", type=int, help=argparse_suppress())


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def find_images_root(path, pattern):
    """The folder whose images/ holds files matching pattern, searched a few levels under path.

    Finding datasets by their files rather than folder names works however Kaggle mounts or unpacks them.
    """
    for depth in range(5):
        hits = Path(path).glob("*/" * depth + "images/" + pattern)
        hit = next((p for p in hits if "__MACOSX" not in p.parts), None)
        if hit:
            return hit.parent.parent
    raise FileNotFoundError(f"no images/{pattern} under {path}")


def find_mapqa_root(path):
    return find_images_root(path, "map_*.png")


def find_saved(path, marker):
    """A saved model folder: path itself when it holds marker, else a folder a few levels down that does.

    Lets a previous Kaggle run's output be passed as /kaggle/input. A name that is not a path, such as a
    Hugging Face model id, comes back unchanged.
    """
    path = Path(path)
    if not path.exists():
        return str(path)
    if (path / marker).exists():
        return str(path)
    for depth in range(1, 6):
        hit = next(path.glob("*/" * depth + marker), None)
        if hit:
            return str(hit.parent)
    raise FileNotFoundError(f"no {marker} under {path}")


def read_split(data_dir, split, image_root):
    """Records of one split, each with image_path set (a record's "image" is relative to image_root)."""
    records = []
    with open(Path(data_dir) / f"{split}.jsonl", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            record["image_path"] = str(Path(image_root) / record["image"])
            records.append(record)
    return records


def sample_records(records, n, seed):
    if n is None or n >= len(records):
        return list(records)
    return random.Random(seed).sample(records, n)


def training_records(args, mapqa_root):
    records = [] if args.train_maps == 0 else sample_records(read_split(args.data, "train", mapqa_root),
                                                             args.train_maps, args.seed)
    synthetic = []
    if args.synth:
        folder = find_images_root(args.synth, "synth_*")
        synthetic = sample_records(read_split(folder, "train", folder), args.synth_train_maps, args.seed)
    india = []
    if args.india:
        folder = find_images_root(args.india, "synthind_*")
        india = sample_records(read_split(folder, "train", folder), args.india_train_maps, args.seed)
    print(f"training maps: {len(records)} MapQA + {len(synthetic)} synthetic + {len(india)} India", flush=True)
    return records + synthetic + india


def evaluation_sets(args, mapqa_root):
    """(name, records) pairs: MapQA test maps, plus synthetic US / India test maps when those are given."""
    sets = []
    if args.eval_maps:
        sets.append(("mapqa", sample_records(read_split(args.data, "test", mapqa_root), args.eval_maps, seed=0)))
    if args.synth:
        folder = find_images_root(args.synth, "synth_*")
        sets.append(("synth", sample_records(read_split(folder, "test", folder), args.synth_eval_maps, seed=0)))
    if args.india:
        folder = find_images_root(args.india, "synthind_*")
        sets.append(("india", sample_records(read_split(folder, "test", folder), args.india_eval_maps, seed=0)))
    if args.eval_limit:
        sets = [(name, records[: args.eval_limit]) for name, records in sets]
    return sets


class MapDataset(torch.utils.data.Dataset):
    def __init__(self, records):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        record = self.records[i]
        return Image.open(record["image_path"]).convert("RGB"), record["target"]


def as_list(batch):
    return batch


def autocast(device, amp_dtype):
    return torch.autocast(device.type, dtype=amp_dtype) if amp_dtype else contextlib.nullcontext()


def train_loop(model, loader, args, forward, out_dir, amp_dtype=None, groups=None, snapshot=None):
    """AdamW with warmup + cosine decay. Stops after --epochs or --hours, whichever comes first.

    forward(batch) returns the loss. After 20 updates the schedule is shortened, if needed, so the
    learning rate still decays to zero within the time budget. groups: optional AdamW parameter groups,
    for weights that need their own learning rate; the warmup and decay apply to every group alike.

    With --checkpoint-every N (minutes), the trainable weights, optimizer, schedule and position in the data are
    saved to out_dir/checkpoint.pt every N minutes and when training ends; with --resume, training continues from
    that file, in the same data order, and --hours counts the time already spent. Each epoch then draws its order
    from --seed, so a resumed run sees exactly the maps an uninterrupted one would.

    snapshot(update): optional; called every --snapshot-every hours of training to keep a usable copy of the model
    (the caller decides what to save; nothing is overwritten).
    """
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(groups or params, lr=args.lr, weight_decay=args.weight_decay)
    batches = len(loader)
    schedule = {"total": max(1, args.epochs * math.ceil(batches / args.grad_accum))}

    def lr_factor(update):
        warmup = max(1, round(0.03 * schedule["total"]))
        if update < warmup:
            return (update + 1) / warmup
        progress = min(1.0, (update - warmup) / max(1, schedule["total"] - warmup))
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    device = next(model.parameters()).device
    every = getattr(args, "checkpoint_every", 0) or 0
    checkpointing = every > 0 or getattr(args, "resume", False)
    path = Path(out_dir) / "checkpoint.pt"
    update, first_epoch, first_batch, spent = 0, 0, 0, 0.0
    if getattr(args, "resume", False) and path.exists():
        state = torch.load(path, map_location=device, weights_only=False)
        if state.get("finished"):
            load_trainable(model, state)
            print(f"training already finished ({state['update']} updates); weights loaded from {path}", flush=True)
            return state["update"]
        load_trainable(model, state)
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        schedule["total"] = state["schedule_total"]
        update, first_epoch, first_batch, spent = state["update"], state["epoch"], state["batch"], state["spent"]
        scheduler.load_state_dict(state["scheduler"])
        print(f"resuming at update {update}/{schedule['total']} (epoch {first_epoch}, batch {first_batch}, "
              f"{spent / 3600:.2f} h already spent)", flush=True)

    start = time.time()
    deadline = start + args.hours * 3600 - spent
    last_save = time.time()
    halt_after = getattr(args, "halt_after", None)
    period = (getattr(args, "snapshot_every", 0) or 0) * 3600
    next_snapshot = (math.floor(spent / period) + 1) * period if period and snapshot else None

    def save(epoch, batch, finished=False):
        state = {"trainable": {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad},
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "scaler": scaler.state_dict(), "schedule_total": schedule["total"], "update": update,
                 "epoch": epoch, "batch": batch, "spent": spent + time.time() - start, "finished": finished}
        partial = path.with_suffix(".tmp")
        torch.save(state, partial)
        partial.replace(path)  # atomic: a run killed mid-save keeps the previous checkpoint
        print(f"checkpoint saved at update {update}", flush=True)

    def epoch_batches(epoch, skip):
        """(batch number, batch) pairs of one epoch, from batch `skip` + 1 on."""
        if not checkpointing:
            return enumerate(loader, 1)
        order = torch.randperm(len(loader.dataset), generator=torch.Generator().manual_seed(args.seed * 1000 + epoch))
        rest = order[skip * loader.batch_size:].tolist()
        part = torch.utils.data.DataLoader(loader.dataset, batch_size=loader.batch_size, sampler=rest,
                                           num_workers=loader.num_workers, collate_fn=loader.collate_fn,
                                           pin_memory=loader.pin_memory)
        return enumerate(part, skip + 1)

    update_loss, window = 0.0, []
    model.train()
    with open(Path(out_dir) / "train_log.jsonl", "a") as log:
        for epoch in range(first_epoch, args.epochs):
            skip = first_batch if epoch == first_epoch else 0
            for i, batch in epoch_batches(epoch, skip):
                with autocast(device, amp_dtype):
                    loss = forward(batch)
                scaler.scale(loss / args.grad_accum).backward()
                update_loss += loss.item() / args.grad_accum
                if i % args.grad_accum and i != batches:
                    continue
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                update += 1
                window.append(update_loss)
                update_loss = 0.0
                if update == 20 and not spent:
                    seconds_per_update = (time.time() - start) / update
                    fits = update + int((deadline - time.time()) / seconds_per_update)
                    schedule["total"] = max(update + 1, min(schedule["total"], fits))
                if update == 1 or update % args.log_every == 0:
                    entry = {"epoch": epoch, "update": update, "of": schedule["total"],
                             "loss": round(sum(window) / len(window), 4), "lr": scheduler.get_last_lr()[0],
                             "minutes": round((spent + time.time() - start) / 60, 1)}
                    window.clear()
                    print(json.dumps(entry), flush=True)
                    log.write(json.dumps(entry) + "\n")
                    log.flush()
                done = update >= schedule["total"] or time.time() > deadline
                if next_snapshot is not None and spent + time.time() - start >= next_snapshot:
                    snapshot(update)
                    print(f"snapshot kept at update {update} ({next_snapshot / 3600:.0f} h)", flush=True)
                    next_snapshot += period
                if checkpointing and (done or time.time() - last_save > every * 60 or update == halt_after):
                    position = (epoch, i) if i < batches else (epoch + 1, 0)
                    save(*position, finished=done)
                    last_save = time.time()
                if done:
                    print(f"stopping after {update} updates ({(spent + time.time() - start) / 3600:.2f} h)", flush=True)
                    return update
                if update == halt_after:  # testing resume: stop as if the machine had been taken away
                    raise SystemExit(f"halted after update {update} (--halt-after)")
    if checkpointing:
        save(args.epochs, 0, finished=True)
    return update


def load_trainable(model, state):
    """Put a checkpoint's trainable weights back into the model, by name."""
    named = dict(model.named_parameters())
    missing = [n for n in state["trainable"] if n not in named]
    if missing:
        raise SystemExit(f"checkpoint has {len(missing)} weights this model lacks, e.g. {missing[0]}")
    with torch.no_grad():
        for name, value in state["trainable"].items():
            named[name].copy_(value.to(named[name].device, named[name].dtype))


# ---------------------------------------------------------------- scoring

def parse_prediction(text):
    """The first JSON object in the model output, or None."""
    start = text.find("{")
    if start < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _same_text(a, b):
    if isinstance(a, str) and isinstance(b, str):
        return " ".join(a.split()).lower() == " ".join(b.split()).lower()
    return a is None and b is None


def _is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def score_map(record, prediction_text):
    target = json.loads(record["target"])
    pred = parse_prediction(prediction_text)
    score = {"id": record["id"], "legend_type": target["legend_type"], "parsed": pred is not None}
    if pred is None:
        return score
    values = pred.get("values") if isinstance(pred.get("values"), dict) else {}
    codes = list(target["values"])
    score["exact_match"] = pred == target
    score["title_correct"] = _same_text(pred.get("title"), target["title"])
    score["legend_title_correct"] = _same_text(pred.get("legend_title"), target["legend_title"])
    score["legend_type_correct"] = pred.get("legend_type") == target["legend_type"]
    score["data_type_correct"] = pred.get("data_type") == target["data_type"]
    score["coverage"] = sum(c in values for c in codes) / len(codes)
    score["extra_keys"] = sum(k not in target["values"] for k in values)
    if target["legend_type"] == "discrete":
        score["ranges_correct"] = pred.get("ranges") == target["ranges"]
        right = {c: bool(c in values and values[c] == target["values"][c]) for c in codes}
        score["state_accuracy"] = sum(right.values()) / len(codes)
        score["per_state"] = right
    else:
        truth = {c: v for c, v in record["true_values"].items() if v is not None}
        span = (max(truth.values()) - min(truth.values())) or 1.0
        per_state = {c: min(1.0, abs(values[c] - v) / span) if _is_number(values.get(c)) else 1.0
                     for c, v in truth.items()}
        errors = list(per_state.values())
        score["nmae"] = sum(errors) / len(errors)
        score["nrmse"] = math.sqrt(sum(e * e for e in errors) / len(errors))
        score["within_5pct"] = sum(e <= 0.05 for e in errors) / len(errors)
        score["per_state"] = per_state
    return score


def summarize(scores):
    summary = {}
    for kind in ("discrete", "continuous"):
        group = [s for s in scores if s["legend_type"] == kind]
        if not group:
            continue

        def mean(key, if_unparsed):
            return round(sum(float(s.get(key, if_unparsed)) for s in group) / len(group), 4)

        result = {"maps": len(group)}
        for key in ("parsed", "exact_match", "title_correct", "legend_title_correct", "legend_type_correct",
                    "data_type_correct", "coverage"):
            result[key] = mean(key, 0)
        result["extra_keys"] = mean("extra_keys", 0)
        if kind == "discrete":
            result.update(ranges_correct=mean("ranges_correct", 0), state_accuracy=mean("state_accuracy", 0))
        else:
            result.update(nmae=mean("nmae", 1), nrmse=mean("nrmse", 1), within_5pct=mean("within_5pct", 0))
        result["per_state"] = per_state_summary(group, unparsed=0.0 if kind == "discrete" else 1.0)
        summary[kind] = result
    return summary


def per_state_summary(group, unparsed):
    """Per-state accuracy (discrete) or mean error (continuous), worst state first.

    Small states are the known weak point, so the run has to report them separately rather than
    leaving them averaged into the 50.
    """
    every_code = {code for score in group for code in score.get("per_state", {})}
    totals = {}
    for score in group:
        per_state = score.get("per_state") or {}
        for code in every_code:
            got = float(per_state[code]) if code in per_state else unparsed
            hit, count = totals.get(code, (0.0, 0))
            totals[code] = (hit + got, count + 1)
    means = {code: round(total / count, 4) for code, (total, count) in totals.items()}
    worst = sorted(means, key=lambda c: means[c], reverse=unparsed == 1.0)
    return {code: means[code] for code in worst}


def evaluate_and_save(records, predictions, out_dir, name):
    """Score one test set; writes predictions_<name>.jsonl and metrics_<name>.json."""
    with open(out_dir / f"raw_predictions_{name}.jsonl", "w", encoding="utf-8") as f:  # kept if scoring fails
        for record, prediction in zip(records, predictions):
            f.write(json.dumps({"id": record["id"], "prediction": prediction}, ensure_ascii=False) + "\n")
    scores = [score_map(r, p) for r, p in zip(records, predictions)]
    with open(out_dir / f"predictions_{name}.jsonl", "w", encoding="utf-8") as f:
        for record, prediction, score in zip(records, predictions, scores):
            f.write(json.dumps({"id": record["id"], "prediction": prediction, "target": record["target"],
                                "score": score}, ensure_ascii=False) + "\n")
    summary = summarize(scores)
    (out_dir / f"metrics_{name}.json").write_text(json.dumps(summary, indent=2))
    print(f"{name} test maps:\n{json.dumps(summary, indent=2)}", flush=True)
    return summary
