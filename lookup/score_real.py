"""Score the sheet read on real maps against the linked source tables.

Discrete: right if the true value lies in the range the model chose. Continuous: error as a share of the value range.
The discrete baseline always answers the commonest true range, so it shows what skewed data gives for nothing.

    .venv/bin/python lookup/score_real.py --predictions outputs/mapqa-derender-real-sheets/real/predictions_real.jsonl
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

SMALL = {"AN", "DL", "GA", "SK", "TR"}
TINY = {"CH", "DH", "LD", "PY"}  # empty cells on most sheets, and not in the taught list
NUMBER = r"-?\d[\d,]*\.?\d*"


def bounds(text):
    found = [float(n.replace(",", "")) for n in re.findall(r"\d[\d,]*\.?\d*", str(text))]
    return (found[0], found[-1]) if len(found) >= 2 else None


def group(code):
    return "tiny" if code in TINY else "small" if code in SMALL else "big"


def score_map(answer, truth):
    """{code: right or error}, kind, baseline share."""
    values, ranges = answer.get("values") or {}, answer.get("ranges")
    truth = {c: v for c, v in truth.items()}
    if answer.get("legend_type") == "discrete" and ranges:
        spans = [bounds(r) for r in ranges]
        if not all(spans):
            return None
        slack = 0.002 * (spans[-1][1] - spans[0][0])
        def where(v):
            return next((i for i, s in enumerate(spans) if s[0] - slack <= v <= s[1] + slack), None)
        result = {c: isinstance(values.get(c), int) and 0 <= values[c] < len(spans) and
                  spans[values[c]][0] - slack <= v <= spans[values[c]][1] + slack for c, v in truth.items()}
        common = Counter(where(v) for c, v in truth.items() if c not in TINY).most_common(1)[0][1]
        return {"kind": "discrete", "per": result, "baseline": common / sum(c not in TINY for c in truth)}
    span = (max(truth.values()) - min(truth.values())) or 1.0
    per = {c: min(1.0, abs(values[c] - v) / span) if isinstance(values.get(c), (int, float)) else 1.0
           for c, v in truth.items()}
    return {"kind": "continuous", "per": per}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("outputs/real-fit/real_scores.json"))
    args = parser.parse_args()
    links = json.loads(Path("data/cma_tables/links.json").read_text())
    trust = {Path(t[0]).stem: t[1] >= 0.8 and t[4] >= 0.9 for t in json.loads(Path("outputs/real-fit/trust.json").read_text())}
    rows, broken = [], []
    for line in open(args.predictions):
        record = json.loads(line)
        try:
            answer = json.loads(record["prediction"])
            scored = score_map(answer, links[record["id"]]["values"])
        except (ValueError, TypeError):
            scored = None
        if scored is None:
            broken.append(record["id"])
            continue
        rows.append({"id": record["id"], "tier": links[record["id"]]["tier"], "trusted": trust.get(record["id"]), **scored})
    report = {"maps": len(rows), "unscored": broken}
    for tier in ("strong", "title"):
        for kind in ("discrete", "continuous"):
            chosen = [r for r in rows if r["tier"] == tier and r["kind"] == kind]
            if not chosen:
                continue
            entry = {"maps": len(chosen), "trusted": sum(bool(r["trusted"]) for r in chosen)}
            for name in ("big", "small", "tiny"):
                entry[name] = round(float(np.mean([np.mean([float(v) for c, v in r["per"].items() if group(c) == name])
                                                   for r in chosen if any(group(c) == name for c in r["per"])])), 4)
            entry["AN"] = round(float(np.mean([float(r["per"]["AN"]) for r in chosen if "AN" in r["per"]])), 4)
            if kind == "discrete":
                entry["baseline_commonest"] = round(float(np.mean([r["baseline"] for r in chosen])), 4)
            report[f"{tier}_{kind}"] = entry
    args.out.write_text(json.dumps({"report": report, "rows": rows}, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
