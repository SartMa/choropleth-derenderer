#!/usr/bin/env python3
"""Train and score the derenderer on the lab GPU: the answer sheet (v3, S2 + zoom) and, for comparison, the tiles
model (map + 2 magnified tiles placed by the MLP picker). Both from scratch, on the same maps, scored on the same
test sets.

Run from the repo root, in the conda environment from lab/README.md (native Windows) or inside the Docker image:

    python lab/run.py check        GPU, library versions, free disk
    python lab/run.py prepare      download the data and the Kaggle v3 model (needs KAGGLE_API_TOKEN), convert MapQA

    python lab/run.py smoke        answer sheet: ~30 min on a few maps (training, forced stop and resume, snapshot, scoring)
    python lab/run.py full         answer sheet: the real run; run it again to continue after any stop
    python lab/run.py score <weights folder>   answer sheet: score a snapshot, or the Kaggle v3 model:
                                               python lab/run.py score data/lab/inputs/v3-weights

    python lab/run.py tiles-smoke  tiles model: the same checks
    python lab/run.py tiles-full   tiles model: the real run; run it again to continue after any stop
    python lab/run.py tiles-score <weights folder>   tiles model: score a snapshot (the picker is taught again)

    python lab/run.py status       progress of both full runs

One full run at a time: each needs the whole GPU. Everything goes under data/lab/ (git-ignored): inputs/,
mapqa_data/, hf/ (the Qwen download) and runs/. Full runs keep a resume checkpoint every 30 minutes and a usable
copy of the model every 6 hours of training (runs/<run>/snapshots/update_NNNNN/), never overwritten.
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LAB = ROOT / "data" / "lab"
INPUTS = LAB / "inputs"
RUNS = LAB / "runs"
DATASETS = {  # folder under inputs/ -> Kaggle dataset
    "mapqa-s": "sartma/mapqa-s",
    "synth-usa": "sartma/synth-usa-choropleth",
    "synth-india": "sartma/synth-india-choropleth",
    "answer-inputs": "skini501/answer-v2-inputs",
    "real-sheets": "skini501/real-sheets",
}
WEIGHTS = ("v3-weights", "skini501/mapqa-derender-v3-zoom")  # the Kaggle v3 model, scored for comparison only

# Both models train on exactly the same maps, for the same number of passes, with the same batch and learning rate:
# the 3,704 MapQA maps that have shapefile labels (the answer sheet needs them; the tiles model is restricted to the
# same ones), all 10,000 synthetic US and all 10,000 synthetic India training maps.
EPOCHS = 2
TRAIN_MAPS = 3704 + 10000 + 10000
SAME = ["--train-maps", "100000", "--synth-train-maps", "10000", "--india-train-maps", "10000",
        "--epochs", str(EPOCHS), "--hours", "300",  # two full passes; the hours only cap it
        "--batch-size", "1", "--grad-accum", "16", "--lr", "1e-4",
        "--eval-maps", "100000", "--india-eval-maps", "1000",  # the whole MapQA and synthetic India test sets
        "--checkpoint-every", "30", "--snapshot-every", "6"]
FULL = SAME + ["--head-minutes", "60"]  # a fresh head gets longer on its own first
TILES_FULL = SAME + ["--picker-maps", "2000", "--synth-eval-maps", "200"]

SMOKE_SAME = ["--train-maps", "40", "--synth-train-maps", "40", "--india-train-maps", "40", "--eval-maps", "8",
              "--india-eval-maps", "8", "--eval-limit", "8", "--epochs", "1", "--hours", "1", "--batch-size", "1",
              "--grad-accum", "4", "--lr", "1e-4", "--log-every", "1", "--checkpoint-every", "1",
              "--snapshot-every", "0.05"]
SMOKE = SMOKE_SAME + ["--head-maps", "20", "--head-minutes", "2", "--report-maps", "5"]
TILES_SMOKE = SMOKE_SAME + ["--picker-maps", "40", "--synth-eval-maps", "8"]
# Native Windows starts DataLoader workers by re-importing everything, which is fragile and slow to start; there the
# maps are prepared in the main process instead (about 10% slower). Linux (Docker) uses worker processes.
WORKERS = ["--workers", "0" if os.name == "nt" else "4"]


def run(command, **kwargs):
    print("$", " ".join(str(c) for c in command), flush=True)
    return subprocess.run([str(c) for c in command], cwd=ROOT, **kwargs)


def environment():
    env = dict(os.environ)
    env.setdefault("HF_HOME", str(LAB / "hf"))  # keep the 7.5 GB Qwen download between runs
    env["PYTHONUTF8"] = "1"  # Windows would otherwise read text files as cp1252 (map ranges contain "–")
    env.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    return env


def data_args():
    return ["--data", LAB / "mapqa_data", "--mapqa", INPUTS / "mapqa-s", "--synth", INPUTS / "synth-usa",
            "--india", INPUTS / "synth-india", "--real", INPUTS / "real-sheets"]


def v3_command(out, weights, extra):
    """The answer sheet (train/answer_v3.py); weights 'none' = from scratch."""
    return [sys.executable, ROOT / "train" / "answer_v3.py", *data_args(), "--labels", INPUTS / "answer-inputs",
            "--adapter", weights, "--mode", "zoom", "--out", out, *WORKERS, *extra]


def tiles_command(out, extra, weights=None):
    """The tiles model (train/train_qwen.py): the map plus 2 tiles magnified 3x, placed by the rule in training
    (20% of maps without tiles), then an MLP picker taught on the trained model places them when reading.
    No weights = from scratch."""
    return [sys.executable, ROOT / "train" / "train_qwen.py", *data_args(), "--mapqa-ids", INPUTS / "answer-inputs",
            "--tiles", "2", "--tile-zoom", "3", "--crop-dropout", "0.2", "--picker", "mlp", "--skip-rule-eval",
            "--eval-batch-size", "8", *(["--adapter", weights] if weights else []), "--out", out, *WORKERS, *extra]


def check(_):
    import torch
    import transformers
    import peft
    print(f"python {sys.version.split()[0]}, torch {torch.__version__}, transformers {transformers.__version__}, "
          f"peft {peft.__version__}")
    if not torch.cuda.is_available():
        sys.exit("no CUDA GPU visible: install the CUDA build of PyTorch (lab/README.md), or in Docker use --gpus all")
    name = torch.cuda.get_device_name(0)
    memory = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    bf16 = torch.cuda.is_bf16_supported()
    print(f"GPU: {name}, {memory:.1f} GB, bf16 {'yes' if bf16 else 'no'}")
    x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16 if bf16 else torch.float16)
    torch.cuda.synchronize()
    began = time.time()
    for _ in range(50):
        x @ x
    torch.cuda.synchronize()
    print(f"matmul speed: {50 * 2 * 4096 ** 3 / (time.time() - began) / 1e12:.1f} TFLOPS")
    LAB.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(LAB).free / 2 ** 30
    print(f"free disk where data/lab lives: {free:.0f} GB (need about 30 GB)")


def unzip_inner(folder):
    """Unpack zips left inside a downloaded dataset (folders uploaded zipped, e.g. images.zip) where they lie."""
    import zipfile
    for archive in sorted(Path(folder).rglob("*.zip")):
        print(f"unpacking {archive.relative_to(folder)}", flush=True)
        with zipfile.ZipFile(archive) as z:
            # kaggle -r zip packs a folder's contents without the folder's name: put them back under it
            names = z.namelist()
            inside = all(n.startswith(archive.stem + "/") for n in names)
            z.extractall(archive.parent if inside else archive.parent / archive.stem)
        archive.unlink()


def prepare(_):
    if not os.environ.get("KAGGLE_API_TOKEN") and not (Path.home() / ".kaggle").exists():
        sys.exit("set KAGGLE_API_TOKEN first (the skini501 token); it is read from the environment, never saved")
    kaggle = shutil.which("kaggle") or sys.exit("kaggle CLI not found: pip install kaggle")
    for folder, dataset in DATASETS.items():
        target = INPUTS / folder
        if target.exists() and any(target.iterdir()):
            print(f"have {folder}")
            continue
        target.mkdir(parents=True, exist_ok=True)
        if run([kaggle, "datasets", "download", dataset, "-p", target, "--unzip"]).returncode:
            shutil.rmtree(target)
            sys.exit(f"download of {dataset} failed. If its Kaggle page says 'awaiting compression', wait and retry")
        unzip_inner(target)
    folder, kernel = WEIGHTS
    target = INPUTS / folder
    if not (target.exists() and any(target.rglob("adapter_config.json"))):
        target.mkdir(parents=True, exist_ok=True)
        if run([kaggle, "kernels", "output", kernel, "-p", target]).returncode:
            sys.exit(f"download of {kernel}'s output failed")
    if not (LAB / "mapqa_data" / "train.jsonl").exists():
        if run([sys.executable, ROOT / "mapqa" / "convert_mapqa.py", "--mapqa", INPUTS / "mapqa-s",
                "--out", LAB / "mapqa_data"], env=environment()).returncode:
            sys.exit("converting MapQA failed")
    print("prepared: inputs in", INPUTS)


def smoke_test(name, command, needed):
    """command(extra args) -> the command line. Stops training on purpose, resumes, checks the output."""
    out = RUNS / name
    shutil.rmtree(out, ignore_errors=True)
    env = environment()
    print("1/2: training (from scratch) stops itself after 6 updates, as if the machine had been taken away",
          flush=True)
    first = run(command(out, ["--resume", "--halt-after", "6"]), env=env)
    if first.returncode == 0 or not (out / "checkpoint.pt").exists():
        sys.exit(f"{name} FAILED: the forced stop did not happen or left no checkpoint")
    print("2/2: the same command again resumes, finishes training and scores", flush=True)
    if run(command(out, ["--resume"]), env=env).returncode:
        sys.exit(f"{name} FAILED: the resumed run crashed (see the output above)")
    log = [json.loads(line) for line in open(out / "train_log.jsonl")]
    snapshots = sorted((out / "snapshots").glob("update_*")) if (out / "snapshots").exists() else []
    print(f"\ntraining updates logged: {[e['update'] for e in log]}")
    print(f"snapshots kept: {[p.name for p in snapshots]}")
    print(f"files: {sorted(p.name for p in out.iterdir() if p.is_file())}")
    missing = [n for n in needed if not (out / n).exists()]
    if not snapshots or missing:
        sys.exit(f"{name} FAILED: missing {missing or 'a snapshot'}")
    per_map = log[-1]["minutes"] * 60 / (log[-1]["update"] * 4)
    print(f"\n{name} PASSED. Training speed here: about {per_map:.1f} s per map, so the full run's training "
          f"({EPOCHS} passes over {TRAIN_MAPS:,} maps) takes about {EPOCHS * TRAIN_MAPS * per_map / 3600:.0f} h, "
          f"plus scoring ~5,200 test maps.")


def smoke(_):
    smoke_test("smoke", lambda out, extra: v3_command(out, "none", SMOKE + extra),
               ["metrics_mapqa.json", "metrics_india.json", "predictions_real.jsonl"])


def tiles_smoke(_):
    smoke_test("tiles-smoke", lambda out, extra: tiles_command(out, TILES_SMOKE + extra),
               ["picker.pt", "metrics_mapqa_mlp.json", "metrics_india_mlp.json", "predictions_real_mlp.jsonl"])


def run_logged(out, command):
    """A long run, its output shown and appended to out/run.log."""
    out.mkdir(parents=True, exist_ok=True)
    print(f"output and log in {out}. Stop any time (Ctrl+C); run the same command to continue.", flush=True)
    with open(out / "run.log", "a", encoding="utf-8") as f:
        f.write(f"\n===== started {time.strftime('%Y-%m-%d %H:%M')} =====\n")
        f.flush()
        process = subprocess.Popen([str(c) for c in command], cwd=ROOT, env=environment(), stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", bufsize=1)
        for line in process.stdout:
            sys.stdout.write(line)
            f.write(line)
            f.flush()
    sys.exit(process.wait())


def full(_):
    run_logged(RUNS / "full", v3_command(RUNS / "full", "none", FULL + ["--resume"]))


def tiles_full(_):
    run_logged(RUNS / "tiles-full", tiles_command(RUNS / "tiles-full", TILES_FULL + ["--resume"]))


def status(_):
    for name, final in (("full", "metrics_india.json"), ("tiles-full", "metrics_india_mlp.json")):
        out = RUNS / name
        print(f"== {name} ({'answer sheet' if name == 'full' else 'tiles model'})")
        if not out.exists():
            print("   not started")
            continue
        log = out / "train_log.jsonl"
        lines = [json.loads(line) for line in open(log)] if log.exists() else []
        if lines:
            last = lines[-1]
            print(f"   training: update {last['update']} of {last['of']}, loss {last['loss']}, "
                  f"{last['minutes'] / 60:.1f} h of training so far")
        for partial in sorted(out.glob("partial_*.jsonl")):
            print(f"   scoring {partial.stem[len('partial_'):]}: {sum(1 for _ in open(partial))} maps read")
        snapshots = sorted((out / "snapshots").glob("update_*")) if (out / "snapshots").exists() else []
        print(f"   snapshots: {[p.name for p in snapshots] or 'none yet'}")
        print(f"   finished: {'yes' if (out / final).exists() else 'not yet'}")


def scoring_args(flags):
    keep = {"--train-maps", "--synth-train-maps", "--india-train-maps", "--eval-maps", "--india-eval-maps",
            "--synth-eval-maps", "--picker-maps"}
    return [x for flag, value in zip(flags[::2], flags[1::2]) if flag in keep for x in (flag, value)]


def score(rest):
    if not rest:
        sys.exit("give the weights folder, e.g. data/lab/runs/full/snapshots/update_01234")
    weights = Path(rest[0]).resolve()
    out = RUNS / f"score_{weights.name}"
    sys.exit(run(v3_command(out, weights, scoring_args(FULL) + ["--skip-train", "--resume"]),
                 env=environment()).returncode)


def tiles_score(rest):
    if not rest:
        sys.exit("give the weights folder, e.g. data/lab/runs/tiles-full/snapshots/update_01234")
    weights = Path(rest[0]).resolve()
    out = RUNS / f"tiles_score_{weights.name}"
    sys.exit(run(tiles_command(out, scoring_args(TILES_FULL) + ["--skip-train", "--resume"], weights=weights),
                 env=environment()).returncode)


def main():
    commands = {"check": check, "prepare": prepare, "smoke": smoke, "full": full, "score": score,
                "tiles-smoke": tiles_smoke, "tiles-full": tiles_full, "tiles-score": tiles_score, "status": status}
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        sys.exit(__doc__)
    commands[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    main()
