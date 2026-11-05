"""Score synthetic India predictions on maps without printed state codes: big vs small regions (AN DL GA SK TR),
maps with >= 95% right, and continuous error on big states as a share of the value range.

    .venv/bin/python lookup/score_india.py outputs/<run>/<dir>/predictions_india.jsonl ...
"""
import json, sys
import numpy as np
SMALL = {"AN", "DL", "GA", "SK", "TR"}
test = {}
for line in open("data/synth_ind/test.jsonl"):
    r = json.loads(line); test[r["id"]] = r
def score(path):
    out = {"disc": {"big": [], "small": [], "maps95": []}, "cont": {"big": [], "w5": []}}
    per = {c: [] for c in SMALL}
    n = 0
    for line in open(path):
        p = json.loads(line); r = test[p["id"]]
        if r["style"]["state_labels"]:
            continue
        n += 1
        t = json.loads(r["target"])
        try: a = json.loads(p["prediction"])["values"]
        except Exception: a = {}
        if t["legend_type"] == "discrete":
            right = {c: a.get(c) == v for c, v in t["values"].items() if v is not None}
            out["disc"]["big"] += [x for c, x in right.items() if c not in SMALL]
            out["disc"]["small"] += [x for c, x in right.items() if c in SMALL]
            for c in SMALL:
                if c in right: per[c].append(right[c])
            out["disc"]["maps95"].append(np.mean(list(right.values())) >= 0.95)
        else:
            vals = [v for v in t["values"].values() if v is not None]; span = (max(vals) - min(vals)) or 1
            for c, v in t["values"].items():
                if v is None or c in SMALL: continue
                g = a.get(c); e = 1.0 if not isinstance(g, (int, float)) else abs(g - v) / span
                out["cont"]["big"].append(min(e, 1.0)); out["cont"]["w5"].append(e <= 0.05)
    res = {"maps": n, "discrete_maps": len(out["disc"]["maps95"]),
           "big": np.mean(out["disc"]["big"]), "small": np.mean(out["disc"]["small"]),
           "maps_95": int(sum(out["disc"]["maps95"])),
           **{c: np.mean(v) for c, v in per.items()},
           "cont_err_big": np.mean(out["cont"]["big"]), "cont_within5": np.mean(out["cont"]["w5"])}
    return {k: round(float(v), 3) if isinstance(v, float) else v for k, v in res.items()}
for path in sys.argv[1:]:
    print(path, score(path))
