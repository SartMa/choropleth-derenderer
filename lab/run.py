#!/usr/bin/env python3
"""Train and score the derenderer (answer sheet v3, S2 + zoom) on the lab GPU.

Run from the repo root, in the conda environment from lab/README.md (native Windows) or inside the Docker image:

    python lab/run.py check      GPU, library versions, free disk
    python lab/run.py prepare    download the data and v3's weights from Kaggle (needs KAGGLE_API_TOKEN), convert MapQA
    python lab/run.py smoke      ~30 min on a few maps: training, a forced stop and resume, a snapshot, scoring
    python lab/run.py full       the real run, from scratch (plain Qwen); run it again to continue after any stop
    python lab/run.py status     progress of the full run
    python lab/run.py score <weights folder>   score a kept snapshot, or the Kaggle v3 model for comparison:
                                               python lab/run.py score data/lab/inputs/v3-weights

Everything goes under data/lab/ (git-ignored): inputs/, mapqa_data/, hf/ (the Qwen download) and runs/.
The full run keeps a resume checkpoint every 30 minutes (runs/full/checkpoint.pt) and a usable copy of the model
every 6 hours of training (runs/full/snapshots/update_NNNNN/), which are never overwritten.
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
DATASETS = {  # folder under inputs/ -> Kaggle dataset
    "mapqa-s": "sartma/mapqa-s",
    "synth-usa": "sartma/synth-usa-choropleth",
    "synth-india": "sartma/synth-india-choropleth",
    "answer-inputs": "skini501/answer-v2-inputs",
    "real-sheets": "skini501/real-sheets",
}
WEIGHTS = ("v3-weights", "skini501/mapqa-derender-v3-zoom")  # the Kaggle v3 model, scored for comparison only

EPOCHS = 2
FULL = ["--train-maps", "100000",          # every MapQA training map that has shapefile labels (~3,700)
        "--synth-train-maps", "8000",      # synthetic US
        "--india-train-maps", "10000",     # synthetic India: India is where the gap is
        "--epochs", str(EPOCHS), "--hours", "300",  # two full passes; the hours only cap it
        "--head-minutes", "60",            # a fresh head gets longer on its own first
        "--eval-maps", "100000",           # the whole MapQA test set (4,157 maps)
        "--india-eval-maps", "1000",       # the whole synthetic India test set
        "--checkpoint-every", "30", "--snapshot-every", "6"]
SMOKE = ["--train-maps", "40", "--synth-train-maps", "40", "--india-train-maps", "40", "--head-maps", "20",
         "--head-minutes", "2", "--report-maps", "5", "--eval-maps", "8", "--india-eval-maps", "8", "--eval-limit", "8",
         "--epochs", "1", "--hours", "1", "--grad-accum", "4", "--log-every", "1", "--checkpoint-every", "1",
         "--snapshot-every", "0.05"]
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


def v3_command(out, weights, extra):
    return [sys.executable, ROOT / "train" / "answer_v3.py",
            "--data", LAB / "mapqa_data", "--mapqa", INPUTS / "mapqa-s",
            "--synth", INPUTS / "synth-usa", "--india", INPUTS / "synth-india",
            "--labels", INPUTS / "answer-inputs", "--real", INPUTS / "real-sheets",
            "--adapter", weights, "--mode", "zoom", "--out", out, *WORKERS, *extra]


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
            sys.exit(f"download of {dataset} failed")
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


def smoke(_):
    out = LAB / "runs" / "smoke"
    shutil.rmtree(out, ignore_errors=True)
    env = environment()
    print("1/2: training (from scratch) stops itself after 6 updates, as if the machine had been taken away",
          flush=True)
    first = run(v3_command(out, "none", SMOKE + ["--resume", "--halt-after", "6"]), env=env)
    if first.returncode == 0 or not (out / "checkpoint.pt").exists():
        sys.exit("smoke FAILED: the forced stop did not happen or left no checkpoint")
    print("2/2: the same command again resumes, finishes training and scores", flush=True)
    if run(v3_command(out, "none", SMOKE + ["--resume"]), env=env).returncode:
        sys.exit("smoke FAILED: the resumed run crashed (see the output above)")
    log = [json.loads(line) for line in open(out / "train_log.jsonl")]
    snapshots = sorted((out / "snapshots").glob("update_*")) if (out / "snapshots").exists() else []
    scored = [p.name for p in out.glob("metrics_*.json")]
    print(f"\ntraining updates logged: {[e['update'] for e in log]}")
    print(f"snapshots kept: {[p.name for p in snapshots]}")
    print(f"score files: {scored}")
    if not snapshots or "metrics_mapqa.json" not in scored or not (out / "predictions_real.jsonl").exists():
        sys.exit("smoke FAILED: a snapshot or a score file is missing")
    per_map = log[-1]["minutes"] * 60 / (log[-1]["update"] * 4)
    maps = 3700 + 8000 + 10000
    print(f"\nsmoke PASSED. Training speed here: about {per_map:.1f} s per map, so the full run's training "
          f"({EPOCHS} passes over ~{maps:,} maps) takes about {EPOCHS * maps * per_map / 3600:.0f} h, plus scoring "
          f"~5,200 test maps.")


def full(_):
    out = LAB / "runs" / "full"
    out.mkdir(parents=True, exist_ok=True)
    log = out / "run.log"
    print(f"full run; output and log in {out}. Stop any time (Ctrl+C or docker stop); run the same command to continue.")
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"\n===== started {time.strftime('%Y-%m-%d %H:%M')} =====\n")
        f.flush()
        process = subprocess.Popen([str(c) for c in v3_command(out, "none", FULL + ["--resume"])],
                                   cwd=ROOT, env=environment(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace", bufsize=1)
        for line in process.stdout:
            sys.stdout.write(line)
            f.write(line)
            f.flush()
    sys.exit(process.wait())


def status(_):
    out = LAB / "runs" / "full"
    if not out.exists():
        sys.exit("no full run yet")
    log = out / "train_log.jsonl"
    if log.exists():
        lines = [json.loads(line) for line in open(log)]
        if lines:
            last = lines[-1]
            print(f"training: update {last['update']} of {last['of']}, loss {last['loss']}, "
                  f"{last['minutes'] / 60:.1f} h of training so far")
    for name in ("real", "mapqa", "india"):
        partial = out / f"partial_{name}.jsonl"
        if partial.exists():
            print(f"scoring {name}: {sum(1 for _ in open(partial))} maps read")
    snapshots = sorted((out / "snapshots").glob("update_*")) if (out / "snapshots").exists() else []
    print(f"snapshots: {[p.name for p in snapshots] or 'none yet'}")
    print("finished: yes" if (out / "metrics_india.json").exists() else "finished: not yet")


def score(rest):
    if not rest:
        sys.exit("give the weights folder, e.g. data/lab/runs/full/snapshots/update_01234")
    weights = Path(rest[0]).resolve()
    out = LAB / "runs" / f"score_{weights.name}"
    keep = {"--train-maps", "--synth-train-maps", "--india-train-maps", "--eval-maps", "--india-eval-maps"}
    extra = [x for flag, value in zip(FULL[::2], FULL[1::2]) if flag in keep for x in (flag, value)]
    sys.exit(run(v3_command(out, weights, extra + ["--skip-train", "--resume"]), env=environment()).returncode)


def main():
    commands = {"check": check, "prepare": prepare, "smoke": smoke, "full": full, "status": status, "score": score}
    if len(sys.argv) < 2 or sys.argv[1] not in commands:
        sys.exit(__doc__)
    commands[sys.argv[1]](sys.argv[2:])


if __name__ == "__main__":
    main()
