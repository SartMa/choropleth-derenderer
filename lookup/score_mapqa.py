"""Score the code + VLM reader on the 400 MapQA test maps the VLM runs were scored on, and compare map by map.

The VLM's saved answers on the same maps stand in for a live VLM call. It supplies everything that needs reading
text: title, legend fields and what each colour group means. Code supplies which regions share a colour, each
region's place on the colour bar, and the bar's numbers (by OCR).

    .venv/bin/python lookup/score_mapqa.py --out outputs/lookup-mapqa
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import read_map as rm

GEOJSON = Path("data/boundaries/geoBoundaries-USA-ADM1_simplified.geojson")
ROOT = Path("data/mapqa_s/MapQA_S")
QWEN = Path("outputs/mapqa-derender-qwen-full/qwen")
VLMS = {"tiles": Path("outputs/mapqa-derender-qwen-cnnpicker/qwen/predictions_mapqa_mlp.jsonl"),
        "no tiles": QWEN / "predictions_mapqa_nocrop.jsonl"}
TEMPLATE = rm.template(GEOJSON)
POINTS = rm.border_points(TEMPLATE)
READER = None


def read_one(record):
    """The code's half of one map: its fit, each region's colour and, on continuous maps, place on the bar."""
    image = np.asarray(Image.open(ROOT / record["image"]).convert("RGB"), dtype=float)
    lab = rm.to_lab(image)
    fit, on_edge = rm.fit_map(lab, POINTS)
    colours, pixels = rm.region_colours(lab, TEMPLATE, fit)
    row = {"id": record["id"], "fit": fit, "on_edge": on_edge, "pixels": pixels, "positions": None,
           "colours": {c: v.tolist() for c, v in colours.items()}}
    if record["legend_type"] == "continuous":
        found = rm.find_colour_bar(lab)
        if found is not None:
            bar, box = found
            row.update(positions=rm.read_positions(colours, bar), box=box, ticks=rm.read_ticks(image, box, ocr()))
    return row


def ocr():
    global READER
    if READER is None:
        from rapidocr_onnxruntime import RapidOCR
        READER = RapidOCR()
    return READER


def combine(row, record, prediction):
    """The VLM's whole answer, with the state values replaced by the code's. Falls back to the VLM's values."""
    answer = json.loads(prediction)
    values = answer["values"]
    colours = {c: np.asarray(v) for c, v in row["colours"].items()}
    if record["legend_type"] == "discrete":
        answer["values"] = rm.name_groups(rm.group_by_colour(colours, row["pixels"]), values)
    elif row["positions"] and len(row["ticks"]) >= 3:
        answer["values"] = rm.scale_from_ticks(row["positions"], row["box"], row["ticks"])
    elif row["positions"]:
        answer["values"] = rm.scale_bar(row["positions"], values)
    return answer


def score(answer, record):
    """The same rules as train/common.py: per-state right / wrong, or error as a share of the value range."""
    target = json.loads(record["target"])
    values = answer["values"]
    if record["legend_type"] == "discrete":
        return {c: bool(c in values and values[c] == t) for c, t in target["values"].items()}
    truth = {c: v for c, v in record["true_values"].items() if v is not None}
    span = (max(truth.values()) - min(truth.values())) or 1.0
    return {c: min(1.0, abs(values[c] - v) / span) if values.get(c) is not None else 1.0 for c, v in truth.items()}


def overlay(rows, path, cells=(6, 5), scale=0.5):
    """Contact sheet: the fitted borders in green over the first maps, to judge the fit by eye."""
    w, h = int(700 * scale), int(500 * scale)
    sheet = Image.new("RGB", (cells[0] * w, cells[1] * h), "white")
    for n, row in enumerate(rows[: cells[0] * cells[1]]):
        image = Image.open(ROOT / "images" / f"{row['id']}.png").convert("RGB")
        draw = ImageDraw.Draw(image)
        fit = np.asarray(row["fit"])
        for rings in TEMPLATE.values():
            for ring in rings:
                draw.line([tuple(p) for p in (ring * fit[:2] + fit[2:]).tolist()], fill=(0, 255, 0), width=1)
        sheet.paste(image.resize((w, h)), ((n % cells[0]) * w, (n // cells[0]) * h))
    sheet.save(path)


def compare(rows, records, name, path):
    """One VLM alone against the same VLM with the code's state values, on the same maps."""
    vlm = {r["id"]: r for r in map(json.loads, open(path))}
    groups = {"discrete": [], "continuous": []}
    for row, record in zip(rows, records):
        saved = vlm[row["id"]]
        answer = combine(row, record, saved["prediction"])
        groups[record["legend_type"]].append(
            {"id": row["id"], "code": score(answer, record), "vlm": saved["score"]["per_state"],
             "code_exact": answer == json.loads(record["target"]), "vlm_exact": saved["score"]["exact_match"]})
    report = {}
    d = groups["discrete"]
    both = lambda who, c: float(np.mean([m[who][c] for m in d]))
    fixed = sum(m["code_exact"] and not m["vlm_exact"] for m in d)
    broke = sum(m["vlm_exact"] and not m["code_exact"] for m in d)
    report["discrete"] = {
        "maps": len(d),
        "states": {w: float(np.mean([np.mean(list(m[w].values())) for m in d])) for w in ("code", "vlm")},
        "whole_map_exact": {w: float(np.mean([m[f"{w}_exact"] for m in d])) for w in ("code", "vlm")},
        "RI": {w: both(w, "RI") for w in ("code", "vlm")}, "DE": {w: both(w, "DE") for w in ("code", "vlm")},
        "fixed": int(fixed), "broke": int(broke),
        "still_wrong": [m["id"] for m in d if not m["code_exact"]]}
    c = groups["continuous"]
    report["continuous"] = {
        "maps": len(c),
        "median_error": {w: float(np.median([np.mean(list(m[w].values())) for m in c])) for w in ("code", "vlm")},
        "mean_error": {w: float(np.mean([np.mean(list(m[w].values())) for m in c])) for w in ("code", "vlm")},
        "within_5pct": {w: float(np.mean([np.mean([e <= 0.05 for e in m[w].values()]) for m in c]))
                        for w in ("code", "vlm")},
        "RI_error": {w: float(np.mean([m[w].get("RI", 0) for m in c])) for w in ("code", "vlm")}}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("outputs/lookup-mapqa"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    ids = json.loads((ROOT.parent / "test400_ids.json").read_text())[: args.limit]
    by_id = {r["id"]: r for r in map(json.loads, open("data/mapqa_derender/test.jsonl"))}
    records = [by_id[i] for i in ids]
    cache = args.out / "read.jsonl"
    if cache.exists():
        rows = [json.loads(line) for line in open(cache)]
    else:
        with ProcessPoolExecutor(args.workers) as pool:
            rows = list(pool.map(read_one, records, chunksize=4))
        with open(cache, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
    overlay(rows, args.out / "overlay.png")

    report = {"maps": len(rows), "on_edge_median": float(np.median([r["on_edge"] for r in rows])),
              "on_edge_lowest": float(min(r["on_edge"] for r in rows)),
              "colour_bar_found": sum(r["positions"] is not None for r in rows),
              "bars_with_3_ticks": sum(len(r.get("ticks", [])) >= 3 for r in rows)}
    for name, path in VLMS.items():
        report[name] = compare(rows, records, name, path)
    (args.out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
