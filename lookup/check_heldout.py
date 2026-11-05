"""The reader on 400 MapQA test maps never used while building it, with its settings frozen.

There are no saved VLM answers for these maps. So discrete maps are scored on the grouping alone: each colour group
takes its commonest true class, which assumes a VLM that names groups perfectly. Continuous maps use no VLM.
The same measure on the 400 development maps sits beside it, so the two can be compared.

    .venv/bin/python lookup/check_heldout.py --out outputs/lookup-mapqa
"""

import argparse
import json
import random
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import read_map as rm
import score_mapqa as sm

ZIP = Path.home() / "Downloads/MapQA_S.zip"


def extract(ids):
    with zipfile.ZipFile(ZIP) as z:
        z.extractall(sm.ROOT.parent, [f"MapQA_S/images/{i}.png" for i in ids])


def measure(rows, by_id):
    discrete, continuous = [], []
    for row in rows:
        record = by_id[row["id"]]
        target = json.loads(record["target"])["values"]
        if record["legend_type"] == "discrete":
            groups = rm.group_by_colour({c: np.asarray(v) for c, v in row["colours"].items()}, row["pixels"])
            values = rm.name_groups(groups, target)
            discrete.append({c: values.get(c, "missing") == t for c, t in target.items()})
        elif row["positions"] and len(row["ticks"]) >= 3:
            values = rm.scale_from_ticks(row["positions"], row["box"], row["ticks"])
            continuous.append(sm.score({"values": values}, record))
        else:
            continuous.append({"failed": 1.0})
    return {"discrete_maps": len(discrete),
            "states_right": float(np.mean([np.mean(list(d.values())) for d in discrete])),
            "every_state_right": float(np.mean([all(d.values()) for d in discrete])),
            "RI": float(np.mean([d["RI"] for d in discrete])), "DE": float(np.mean([d["DE"] for d in discrete])),
            "continuous_maps": len(continuous),
            "continuous_unread": sum("failed" in c for c in continuous),
            "median_error": float(np.median([np.mean(list(c.values())) for c in continuous])),
            "mean_error": float(np.mean([np.mean(list(c.values())) for c in continuous])),
            "within_5pct": float(np.mean([np.mean([e <= 0.05 for e in c.values()]) for c in continuous]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("outputs/lookup-mapqa"))
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    records = [json.loads(line) for line in open("data/mapqa_derender/test.jsonl")]
    by_id = {r["id"]: r for r in records}
    seen = set(json.loads((sm.ROOT.parent / "test400_ids.json").read_text()))
    fresh = random.Random(1).sample([r for r in records if r["id"] not in seen], 400)
    extract([r["id"] for r in fresh])
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(sm.read_one, fresh, chunksize=4))
    with open(args.out / "read_heldout.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    development = [json.loads(line) for line in open(args.out / "read.jsonl")]
    report = {"held_out": measure(rows, by_id), "development": measure(development, by_id),
              "on_edge_lowest_held_out": float(min(r["on_edge"] for r in rows))}
    (args.out / "report_heldout.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
