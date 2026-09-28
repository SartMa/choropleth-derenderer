#!/usr/bin/env python3
"""Generate a synthetic map dataset on a Kaggle CPU notebook, then publish it as a (private) Kaggle dataset.

Two steps, because generating 12,000 maps takes about an hour:

    python kaggle/generate_synth.py --country ind              # push the CPU notebook and exit
    python kaggle/generate_synth.py --country ind --finish     # wait for it, download, create the dataset

The notebook runs synth/generate_maps.py for train, dev and test and zips the result, so the download is one
file instead of 12,000. --finish unpacks it into data/synth_<country>, uploads it as
<owner>/synth-<name>-choropleth with the same layout as sartma/synth-usa-choropleth (split files at the top,
images/ below), then deletes the local zip; --keep-local keeps the unpacked copy.
"""

import argparse
import json
import shlex
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from launch import PROJECT, RUNNER, code_cell, download, kaggle, wait  # noqa: E402

NAMES = {"ind": "india", "usa": "usa"}
SPLITS = {"train": 10_000, "dev": 1_000, "test": 1_000}


def build_notebook(country, counts):
    generator = PROJECT / "synth" / "generate_maps.py"
    out = f"/kaggle/working/synth_{country}"
    commands = [shlex.join(["python", "generate_maps.py", "--country", country, "--split", split, "--n", str(n),
                            "--out", out, "--cache", "/kaggle/working/boundaries"])
                for split, n in counts.items()]
    cells = [code_cell(RUNNER),
             code_cell('run("python -c \'import geopandas\' || pip install -q geopandas")'),
             code_cell(f"%%writefile generate_maps.py\n{generator.read_text()}")]
    cells += [code_cell(f"run({command!r})") for command in commands]
    cells.append(code_cell(  # one zip, so the download is one file rather than 12,000
        "import shutil\n"
        f"shutil.make_archive('/kaggle/working/synth_{country}', 'zip', '/kaggle/working', 'synth_{country}')\n"
        f"shutil.rmtree('{out}')\nshutil.rmtree('/kaggle/working/boundaries')\n"
        'run("ls -la /kaggle/working")'))
    return {"cells": cells, "nbformat": 4, "nbformat_minor": 4,
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                         "language_info": {"name": "python"}}}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--country", choices=sorted(NAMES), default="ind")
    parser.add_argument("--owner", default="sartma")
    parser.add_argument("--train", type=int, default=SPLITS["train"])
    parser.add_argument("--dev", type=int, default=SPLITS["dev"])
    parser.add_argument("--test", type=int, default=SPLITS["test"])
    parser.add_argument("--finish", action="store_true", help="wait for the notebook, then create the dataset")
    parser.add_argument("--keep-local", action="store_true", help="keep the unpacked maps in data/ afterwards")
    parser.add_argument("--kaggle", default=shutil.which("kaggle"), help="path to the kaggle CLI")
    args = parser.parse_args()
    if not args.kaggle:
        sys.exit("kaggle CLI not found: pip install kaggle (or pass --kaggle /path/to/kaggle)")

    name = NAMES[args.country]
    slug = f"synth-{name}-maps"
    ref = f"{args.owner}/{slug}"
    if not args.finish:
        folder = PROJECT / "kaggle" / "runs" / slug
        folder.mkdir(parents=True, exist_ok=True)
        counts = {"train": args.train, "dev": args.dev, "test": args.test}
        (folder / "notebook.ipynb").write_text(json.dumps(build_notebook(args.country, counts), indent=1))
        (folder / "kernel-metadata.json").write_text(json.dumps({
            "id": ref, "title": slug, "code_file": "notebook.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_tpu": False,
            "enable_internet": True, "dataset_sources": [], "competition_sources": [], "kernel_sources": [],
            "model_sources": []}, indent=1))
        print(kaggle(args.kaggle, "kernels", "push", "-p", str(folder)))
        print(f"pushed (CPU only); when it is done: python kaggle/generate_synth.py --country {args.country} --finish")
        return

    status = wait(args.kaggle, ref, every_minutes=5)
    if "complete" not in status:
        sys.exit(f"notebook ended with status {status}; see https://www.kaggle.com/code/{ref}")
    download(args.kaggle, ref, slug, everything=True)
    archive = PROJECT / "outputs" / slug / f"synth_{args.country}.zip"
    data = PROJECT / "data" / f"synth_{args.country}"
    if data.exists():
        shutil.rmtree(data)
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(data.parent)
    for split in ("train", "dev", "test"):
        kept = sum(1 for _ in open(data / f"{split}.jsonl"))
        print(f"{split}: {kept} maps")
    dataset_ref = f"{args.owner}/synth-{name}-choropleth"
    (data / "dataset-metadata.json").write_text(json.dumps({
        "title": f"Synthetic {name.title()} Choropleth Maps", "id": dataset_ref,
        "licenses": [{"name": "CC-BY-4.0"}]}, indent=1))
    print(kaggle(args.kaggle, "datasets", "create", "-p", str(data), "-r", "zip"))
    archive.unlink()
    if not args.keep_local:  # the labels are small and worth keeping; the images live on Kaggle
        shutil.rmtree(data / "images")
    print(f"dataset: https://www.kaggle.com/datasets/{dataset_ref}")


if __name__ == "__main__":
    main()
