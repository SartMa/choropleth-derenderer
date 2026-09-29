#!/usr/bin/env python3
"""Run a training script on a Kaggle GPU from this machine, wait for it to finish, and download the results.

The launcher packs mapqa/convert_mapqa.py, train/common.py and the training script into one Kaggle
notebook. The notebook installs pinned libraries, converts MapQA-S, then trains on MapQA plus the synthetic
maps dataset and scores the model on both test sets.
The launcher pushes it with the Kaggle CLI, checks its status every few minutes, and downloads
metrics.json, predictions.jsonl and the logs into outputs/<run name>/.

Needs the Kaggle CLI (pip install kaggle) and a Kaggle API token in ~/.kaggle.

    python kaggle/launch.py deplot --smoke              # ~15-minute check that the whole setup works
    python kaggle/launch.py deplot                      # full run
    python kaggle/launch.py qwen -- --hours 6           # arguments after -- go to the training script
    python kaggle/launch.py qwen --no-wait              # push and exit
    python kaggle/launch.py qwen --wait-only            # wait for the last pushed qwen run, then download

To run on another Kaggle account, set KAGGLE_API_TOKEN to that account's token and pass --owner with its
username; the datasets stay under their own owner and must be shared with (or public to) that account.
"""

import argparse
import json
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
SCRIPTS = {"deplot": PROJECT / "train" / "train_deplot.py", "qwen": PROJECT / "train" / "train_qwen.py"}
PACKAGES = "transformers==4.57.1 peft==0.17.1 accelerate"
SMOKE_ARGS = ["--train-maps", "64", "--synth-train-maps", "32", "--eval-maps", "16", "--synth-eval-maps", "8",
              "--epochs", "1", "--hours", "0.25", "--log-every", "5"]
SMOKE_TIMEOUT_SECONDS = 45 * 60
RESULT_FILES = r".*(metrics.*\.json|predictions.*\.jsonl|train_log\.jsonl|args\.json|\.log)$"

RUNNER = '''import subprocess

def run(command):
    """Run a shell command, show its output live, and fail the notebook if the command fails."""
    print("$", command, flush=True)
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in process.stdout:
        print(line, end="", flush=True)
    if process.wait() != 0:
        raise RuntimeError(f"exit code {process.returncode}: {command}")
'''


def code_cell(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {"language": "python"}, "outputs": [],
            "source": source}


def build_notebook(model, train_args, synth=False, india=False, variants=()):
    """variants: (name, extra args) pairs run one after another in the same notebook, each writing to
    /kaggle/working/<model>/<name>; one variant failing does not stop the others."""
    script = SCRIPTS[model]
    cells = [code_cell(RUNNER), code_cell(f'run("nvidia-smi")\nrun("pip install -q {PACKAGES}")')]
    modules = [PROJECT / "mapqa" / "convert_mapqa.py", PROJECT / "train" / "common.py",
               PROJECT / "train" / "tiles.py", script]
    for path in modules:
        cells.append(code_cell(f"%%writefile {path.name}\n{path.read_text()}"))
    cells.append(code_cell('run("python convert_mapqa.py --mapqa /kaggle/input --out /kaggle/working/data")'))
    synth_args = (["--synth", "/kaggle/input"] if synth else []) + (["--india", "/kaggle/input"] if india else [])

    def command(out, extra=()):
        return shlex.join(["python", script.name, "--data", "/kaggle/working/data", "--mapqa", "/kaggle/input",
                           "--out", out, *synth_args, *train_args, *extra])

    if not variants:
        cells.append(code_cell(f"run({command(f'/kaggle/working/{model}')!r})"))
    for name, extra in variants:
        cells.append(code_cell(f"try:\n    run({command(f'/kaggle/working/{model}/{name}', extra)!r})\n"
                               f"except RuntimeError as error:\n    print('variant {name} failed:', error)"))
    return {"cells": cells, "nbformat": 4, "nbformat_minor": 4,
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                         "language_info": {"name": "python"}}}


def kaggle(cli, *args, check=True):
    result = subprocess.run([cli, *args], capture_output=True, text=True)
    output = (result.stdout + result.stderr).strip()
    if check and result.returncode != 0:
        sys.exit(f"kaggle {' '.join(args)} failed:\n{output}")
    return output


def push(cli, ref, slug, model, train_args, datasets, machine, timeout, kernel_sources=(), use_synth=False,
         use_india=False, variants=()):
    """datasets: the MapQA dataset first, then any others (synthetic maps, a saved model)."""
    folder = PROJECT / "kaggle" / "runs" / slug
    folder.mkdir(parents=True, exist_ok=True)
    notebook = build_notebook(model, train_args, synth=use_synth, india=use_india, variants=variants)
    (folder / "notebook.ipynb").write_text(json.dumps(notebook, indent=1))
    (folder / "kernel-metadata.json").write_text(json.dumps({
        "id": ref, "title": slug, "code_file": "notebook.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
        "machine_shape": machine, "dataset_sources": datasets, "competition_sources": [],
        "kernel_sources": list(kernel_sources),
        "model_sources": []}, indent=1))
    extra = ["-t", str(timeout)] if timeout else []
    output = kaggle(cli, "kernels", "push", "-p", str(folder), *extra)
    print(output)
    if "error" in output.lower():  # the CLI can report a refused push and still exit 0
        sys.exit("push refused; nothing is running")


def wait(cli, ref, every_minutes, just_pushed=False):
    """Poll until the run ends.

    Right after a push the status can still show how the previous version ended, so an end state only
    counts once the new version has been seen queued or running, or after 10 minutes.
    """
    last, failures, started = None, 0, time.time()
    seen_active = not just_pushed
    while True:
        output = kaggle(cli, "kernels", "status", ref, check=False)
        match = re.search(r'status "([^"]+)"', output)
        if not match:
            failures += 1
            print(f"{time.strftime('%H:%M')} could not read status ({failures}/5): {output}", flush=True)
            if failures >= 5:
                sys.exit("giving up on status checks; rerun with --wait-only")
        else:
            failures = 0
            status = match.group(1).split(".")[-1].lower()
            if status != last:
                print(f"{time.strftime('%H:%M')} {ref}: {status}", flush=True)
                last = status
            if not any(word in status for word in ("complete", "error", "cancel")):
                seen_active = True
            elif seen_active or time.time() - started > 600:
                return status
        time.sleep(every_minutes * 60)


def download(cli, ref, slug, everything):
    dest = PROJECT / "outputs" / slug
    dest.mkdir(parents=True, exist_ok=True)
    args = ["kernels", "output", ref, "-p", str(dest), "-o"]
    if not everything:
        args += ["--file-pattern", RESULT_FILES]
    print(kaggle(cli, *args, check=False))
    for metrics in sorted(dest.rglob("metrics*.json")):
        print(f"\n{metrics.relative_to(PROJECT)}:\n{metrics.read_text()}")
    print(f"results in {dest}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", choices=sorted(SCRIPTS))
    parser.add_argument("--smoke", action="store_true", help="tiny run: 64 training maps, 16 test maps, 45-minute cap")
    parser.add_argument("--tag", help="suffix for the run name, e.g. --tag 1h gives mapqa-derender-deplot-1h")
    parser.add_argument("--dataset", default="sartma/mapqa-s", help="Kaggle dataset with MapQA_S")
    parser.add_argument("--synth-dataset", default="sartma/synth-usa-choropleth",
                        help="Kaggle dataset with synth_usa/; 'none' trains on MapQA only")
    parser.add_argument("--india-dataset", default="none",
                        help="Kaggle dataset with the synthetic India maps (e.g. sartma/synth-india-choropleth); "
                             "a few are mixed into training and an India test set is scored")
    parser.add_argument("--variant", action="append", default=[], metavar="NAME:ARGS",
                        help="run several configurations in one notebook, one after another, e.g. "
                             "--variant 't1:--tiles 1' --variant 't3:--tiles 3'; ARGS are added to those after --")
    parser.add_argument("--owner", help="Kaggle account that runs the notebook (default: the dataset's owner)")
    parser.add_argument("--from-run", action="append", default=[],
                        help="attach a previous run's output (e.g. mapqa-derender-deplot-1h); Kaggle only "
                             "allows this for runs that finished successfully")
    parser.add_argument("--extra-dataset", action="append", default=[],
                        help="attach another dataset, such as a saved model uploaded as one; it appears "
                             "under /kaggle/input, so --model /kaggle/input finds it")
    parser.add_argument("--machine", default="NvidiaTeslaT4", help="Kaggle machine_shape: NvidiaTeslaT4 or NvidiaL4")
    parser.add_argument("--every", type=float, default=5, help="minutes between status checks")
    parser.add_argument("--no-wait", action="store_true", help="push and exit")
    parser.add_argument("--wait-only", action="store_true", help="don't push; wait for the last run and download it")
    parser.add_argument("--download-all", action="store_true", help="also download the model weights")
    parser.add_argument("--kaggle", default=shutil.which("kaggle"), help="path to the kaggle CLI")
    argv = sys.argv[1:]
    train_args = []
    if "--" in argv:
        split = argv.index("--")
        argv, train_args = argv[:split], argv[split + 1:]
    args = parser.parse_args(argv)
    if not args.kaggle:
        sys.exit("kaggle CLI not found: pip install kaggle (or pass --kaggle /path/to/kaggle)")

    tag = args.tag or ("smoke" if args.smoke else None)
    slug = f"mapqa-derender-{args.model}" + (f"-{tag}" if tag else "")
    ref = f"{args.owner or args.dataset.split('/')[0]}/{slug}"
    if not args.wait_only:
        run_args = (SMOKE_ARGS if args.smoke else []) + train_args
        owner = args.dataset.split("/")[0]
        datasets = [args.dataset] + ([] if args.synth_dataset == "none" else [args.synth_dataset])
        datasets += [] if args.india_dataset == "none" else [args.india_dataset]
        datasets += [d if "/" in d else f"{owner}/{d}" for d in args.extra_dataset]
        sources = [s if "/" in s else f"{owner}/{s}" for s in args.from_run]
        push(args.kaggle, ref, slug, args.model, run_args, datasets, args.machine,
             SMOKE_TIMEOUT_SECONDS if args.smoke else None, sources, args.synth_dataset != "none",
             args.india_dataset != "none",
             [(v.split(":", 1)[0], shlex.split(v.split(":", 1)[1])) for v in args.variant])
        if args.no_wait:
            flags = ((" --smoke" if args.smoke else "") + (f" --tag {args.tag}" if args.tag else "")
                     + (f" --owner {args.owner}" if args.owner else ""))
            print(f"pushed; later run: python kaggle/launch.py {args.model}{flags} --wait-only")
            return
        time.sleep(30)
    status = wait(args.kaggle, ref, args.every, just_pushed=not args.wait_only)
    download(args.kaggle, ref, slug, args.download_all)
    if "complete" not in status:
        sys.exit(f"run ended with status {status}; see the .log file in outputs/{slug}")


if __name__ == "__main__":
    main()
