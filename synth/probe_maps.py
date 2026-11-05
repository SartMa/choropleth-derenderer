"""Redraw India test maps three ways to ask whether the model fails to *find* Indian states or to *see* them.

For each chosen map (one the model was scored on, drawn without region codes):
- plain:    the map exactly as in the test set (redrawn, and checked to match the original)
- labels:   the same map with every region's code written on it (test A: the model is told which region is which)
- outline:  the same map with one region outlined in magenta, and the prompt naming that region
            (test B: the model is shown where a region is; scored on that region only)

All three come from the same random draws, so values, colours, legend and layout are identical; only the
labels or the outline differ. Writes data/synth_ind_probe/{images/, test.jsonl}.

    .venv/bin/python synth/probe_maps.py --maps 20
"""

import argparse
import json
import random
import shutil
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

import generate_maps as gm

# Big states the model gets wrong most (Chhattisgarh, Bihar, Telangana, Jharkhand, Punjab, Madhya Pradesh) and two
# it always gets right (Rajasthan, Gujarat). Delhi and Goa are left out: a 2.5-point outline would cover them.
OUTLINED = ["CT", "BR", "TG", "JH", "PB", "MP", "RJ", "GJ"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--maps", type=int, default=20, help="half discrete, half continuous")
    parser.add_argument("--src", type=Path, default=Path("data/synth_ind"))
    parser.add_argument("--out", type=Path, default=Path("data/synth_ind_probe"))
    parser.add_argument("--cache", type=Path, default=Path("data/boundaries"))
    args = parser.parse_args()

    names = {f["properties"]["shapeISO"].split("-")[1]: f["properties"]["shapeName"] for f in json.loads(
        (args.cache / "geoBoundaries-IND-ADM1_simplified.geojson").read_text())["features"]}
    test = [json.loads(line) for line in open(args.src / "test.jsonl")]
    scored = random.Random(0).sample(test, 100)  # the 100 India maps every run is scored on (common.sample_records)
    chosen = {"discrete": [], "continuous": []}
    for record in scored:
        bucket = chosen[record["legend_type"]]
        if not record["style"]["state_labels"] and len(bucket) < args.maps // 2:
            bucket.append(record)
    chosen = chosen["discrete"] + chosen["continuous"]

    gm.init_worker(args.cache, "ind")
    if args.out.exists():
        shutil.rmtree(args.out)
    (args.out / "images").mkdir(parents=True)
    seed = gm.SPLIT_SEEDS["test"]
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "images").mkdir()

        def draw(record, condition, labels=False, outline=None, note=""):
            gm.OVERRIDES.update(state_labels=labels, outline=outline)
            index = int(record["id"].rsplit("_", 1)[1])
            sample = gm.make_sample((index, "test", seed, Path(tmp), 0.5))
            gm.OVERRIDES.update(state_labels=False, outline=None)
            assert sample["target"] == record["target"], f"{record['id']} did not redraw the same"
            new_id = f"{record['id']}_{condition}"
            source = Path(tmp) / sample["image"]
            image = f"images/{new_id}{source.suffix}"
            shutil.move(source, args.out / image)
            row = {"id": new_id, "image": image, "target": sample["target"], "legend_type": sample["legend_type"],
                   "data_type": sample["data_type"], "true_values": sample["true_values"], "base_id": record["id"],
                   "condition": condition}
            if outline:
                row.update(outlined=outline, prompt_note=note)
            rows.append(row)
            return args.out / image

        for record in chosen:
            plain = draw(record, "plain")
            original = args.src / record["image"]
            # Fonts differ between machines, so title and legend text can move by a pixel; the map must not.
            a = np.asarray(Image.open(plain).convert("RGB"), dtype=int)
            b = np.asarray(Image.open(original).convert("RGB"), dtype=int)
            assert a.shape == b.shape and (np.abs(a - b).max(-1) > 30).mean() < 0.03, f"{record['id']}: redraw differs"
            draw(record, "labels", labels=True)
            for code in OUTLINED:
                draw(record, f"outline_{code}", outline=code,
                     note=f" The region outlined in magenta is {names[code]} ({code}).")

    with open(args.out / "test.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{len(chosen)} maps -> {len(rows)} images in {args.out}")


if __name__ == "__main__":
    main()
