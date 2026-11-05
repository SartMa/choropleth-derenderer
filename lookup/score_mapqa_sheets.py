"""Score the sheet read on the 400 MapQA test maps, map by map against the best tiles run.

    PYTHONPATH=train .venv/bin/python lookup/score_mapqa_sheets.py --predictions outputs/mapqa-derender-real-mapqa/real/predictions_real.jsonl
"""

import argparse
import json
from pathlib import Path

import numpy as np

from common import score_map

TILES = Path("outputs/mapqa-derender-qwen-cnnpicker/qwen/predictions_mapqa_mlp.jsonl")


def summary(scores):
    d = [s for s in scores if s["legend_type"] == "discrete"]
    c = [s for s in scores if s["legend_type"] == "continuous"]
    get = lambda group, key: float(np.mean([s.get(key, 0) for s in group]))
    return {"discrete_maps": len(d), "exact": get(d, "exact_match"), "states": get(d, "state_accuracy"),
            "ranges": get(d, "ranges_correct"), "RI": float(np.mean([s.get("per_state", {}).get("RI", 0) for s in d])),
            "DE": float(np.mean([s.get("per_state", {}).get("DE", 0) for s in d])),
            "continuous_maps": len(c), "nmae": get(c, "nmae"), "within_5pct": get(c, "within_5pct")}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", type=Path, required=True)
    args = parser.parse_args()
    records = {r["id"]: r for r in map(json.loads, open("data/mapqa_derender/test.jsonl"))}
    sheet = {r["id"]: score_map(records[r["id"]], r["prediction"]) for r in map(json.loads, open(args.predictions))}
    tiles = {r["id"]: r["score"] for r in map(json.loads, open(TILES)) if r["id"] in sheet}
    report = {"sheet": summary(list(sheet.values())), "tiles": summary(list(tiles.values()))}
    d = [i for i in sheet if sheet[i]["legend_type"] == "discrete"]
    report["exact: sheet fixed / broke vs tiles"] = (sum(sheet[i].get("exact_match", False) and not tiles[i]["exact_match"] for i in d),
                                                     sum(tiles[i]["exact_match"] and not sheet[i].get("exact_match", False) for i in d))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
