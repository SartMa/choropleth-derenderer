#!/usr/bin/env python3
"""Convert MapQA-S into derenderer training files: one map image + the JSON the model should output.

Continuous legend (smooth color bar):
    {"title":"The Percentage of ceremonies in the USA","legend_title":"Value","legend_type":"continuous",
     "data_type":"percentage","values":{"AK":37.9,"AL":39.8,...,"WY":47.4}}

Discrete legend (ranges):
    {"title":"The Number of shots in the USA","legend_title":null,"legend_type":"discrete","data_type":"number",
     "ranges":["7548477-7658039","7668882-7761001","7811458-7859166","7920264-8002936"],
     "values":{"AK":0,"AL":0,...,"WY":1}}

Rules:
- values are keyed by state code, sorted, all 50 states every time
- discrete: each state gets its range number (0 = lowest range); "ranges" lists the legend ranges lowest
  first, whatever order the image shows them in; states with no data get null (the image shows "N/A")
- continuous: numbers are rounded so the color bar has about 100 steps, since the image can't show more
- legend_title is the text above the legend: "Value" on continuous maps, nothing on discrete maps
- data_type is "percentage" or "number": every MapQA title says which, and percentage ranges carry "%"
- train/dev/test follow MapQA's own question files; the few maps that have no questions are skipped

Usage:
    python mapqa/convert_mapqa.py --mapqa ~/Downloads/MapQA_S.zip --out data/mapqa_derender
    python mapqa/convert_mapqa.py --mapqa /kaggle/input/<your-dataset> --out /kaggle/working/mapqa_derender

--mapqa can be the zip or any folder that contains MapQA_S/.

Output: <out>/train.jsonl, dev.jsonl, test.jsonl with one line per map:
    {"id": "map_10", "image": "images/map_10.png", "target": "<the JSON above, as a string>",
     "legend_type": "discrete", "data_type": "number", "true_values": {"AK": 7603168, ...}}
"image" is relative to the MapQA_S folder. true_values are the exact numbers, for evaluation only.
"""

import argparse
import json
import math
import re
import statistics
import zipfile
from pathlib import Path

SPLITS = ("train", "dev", "test")
CONTINUOUS_LEGEND_TITLE = "Value"  # checked on several MapQA-S images; discrete maps show no legend title


class MapQASource:
    """Reads MapQA_S files from the downloaded zip or from an extracted folder."""

    def __init__(self, path):
        path = Path(path).expanduser()
        if path.is_file() and zipfile.is_zipfile(path):
            self.zip = zipfile.ZipFile(path)
            names = [n for n in self.zip.namelist() if not n.startswith("__MACOSX")]
            first_meta = next((n for n in names if re.search(r"(^|/)metadata/map_\d+\.json$", n)), None)
            if first_meta is None:
                raise FileNotFoundError(f"no metadata/map_*.json inside {path}")
            self.prefix = first_meta[: first_meta.index("metadata/")]
            self.names = {n[len(self.prefix):] for n in names if n.startswith(self.prefix)}
        else:
            self.zip = None
            patterns = ["*/" * depth + "metadata" for depth in range(5)]
            root = next((p.parent for pattern in patterns for p in path.glob(pattern)
                         if p.is_dir() and "__MACOSX" not in p.parts), None)
            if root is None:
                raise FileNotFoundError(f"no metadata/ folder found under {path}")
            self.root = root
        self.location = path

    def read(self, rel):
        if self.zip:
            return self.zip.read(self.prefix + rel)
        return (self.root / rel).read_bytes()

    def exists(self, rel):
        return rel in self.names if self.zip else (self.root / rel).exists()

    def metadata_ids(self):
        if self.zip:
            found = (re.fullmatch(r"metadata/map_(\d+)\.json", n) for n in self.names)
        else:
            found = (re.fullmatch(r"map_(\d+)\.json", p.name) for p in (self.root / "metadata").iterdir())
        return sorted(int(m.group(1)) for m in found if m)


def split_of_each_map(source):
    """map id -> split, from the map ids referenced in MapQA's question files."""
    split_of = {}
    for split in SPLITS:
        raw = source.read(f"questions/{split}-QA.json")
        for map_id in re.findall(rb'"map_id"\s*:\s*"map_(\d+)\.png"', raw):
            split_of[int(map_id)] = split
    return split_of


def is_missing(value):
    return value is None or (isinstance(value, float) and math.isnan(value))


def round_for_color_bar(values):
    """Round so the color bar spans about 100 steps: one decimal for a span of ~10, hundreds for ~10,000."""
    present = [v for v in values if v is not None]
    span = max(present) - min(present) if present else 0
    if span <= 0:
        return values
    digits = round(-math.log10(span / 100))
    if digits > 0:
        return [None if v is None else round(v, digits) for v in values]  # never adds digits to the data
    return [None if v is None else int(round(v, digits)) for v in values]


def convert_map(meta):
    """Return (target dict, true values) for one MapQA metadata entry."""
    map_id = meta["map_id"]
    data = sorted(meta["data"], key=lambda d: d["region"])
    if len(data) != 50:
        raise ValueError(f"map {map_id}: expected 50 states, got {len(data)}")
    codes = [d["region"] for d in data]
    true_values = {c: None if is_missing(d["value"]) else d["value"] for c, d in zip(codes, data)}
    title = meta["title"]["text"]
    data_type = meta["data_type"].lower()
    if data_type not in ("percentage", "number"):
        raise ValueError(f"map {map_id}: unknown data_type {meta['data_type']!r}")

    if meta["legend_type"] == "Continuous":
        values = round_for_color_bar([true_values[c] for c in codes])
        target = {"title": title, "legend_title": CONTINUOUS_LEGEND_TITLE, "legend_type": "continuous",
                  "data_type": data_type, "values": dict(zip(codes, values))}
    elif meta["legend_type"] == "Discrete":
        ranges = [c for c in meta["class_descriptions"] if c != "N/A"]
        classes = []
        for d in data:
            if d["class"] < 0:
                if d["description"] != "N/A":
                    raise ValueError(f"map {map_id}: {d['region']} has class {d['class']} but {d['description']!r}")
                classes.append(None)
            else:
                if ranges[d["class"]] != d["description"]:
                    raise ValueError(f"map {map_id}: {d['region']} class {d['class']} != {d['description']!r}")
                classes.append(d["class"])
        target = {"title": title, "legend_title": None, "legend_type": "discrete", "data_type": data_type,
                  "ranges": ranges, "values": dict(zip(codes, classes))}
    else:
        raise ValueError(f"map {map_id}: unknown legend_type {meta['legend_type']!r}")
    return target, true_values


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mapqa", required=True, help="MapQA_S.zip, or a folder containing MapQA_S/")
    parser.add_argument("--out", type=Path, default=Path("data/mapqa_derender"))
    parser.add_argument("--limit", type=int, help="convert only the first N maps (quick test)")
    args = parser.parse_args()

    source = MapQASource(args.mapqa)
    split_of = split_of_each_map(source)
    ids = source.metadata_ids()[: args.limit]
    args.out.mkdir(parents=True, exist_ok=True)

    files = {split: open(args.out / f"{split}.jsonl", "w", encoding="utf-8") for split in SPLITS}
    counts = {split: {"continuous": 0, "discrete": 0} for split in SPLITS}
    no_split, no_image, lengths = [], [], []
    try:
        for map_id in ids:
            split = split_of.get(map_id)
            if split is None:
                no_split.append(map_id)
                continue
            image = f"images/map_{map_id}.png"
            if not source.exists(image):
                no_image.append(map_id)
                continue
            meta = json.loads(source.read(f"metadata/map_{map_id}.json"))
            target, true_values = convert_map(meta)
            target_text = json.dumps(target, ensure_ascii=False, separators=(",", ":"))
            record = {
                "id": f"map_{map_id}",
                "image": image,
                "target": target_text,
                "legend_type": target["legend_type"],
                "data_type": target["data_type"],
                "true_values": true_values,
            }
            files[split].write(json.dumps(record, ensure_ascii=False) + "\n")
            counts[split][target["legend_type"]] += 1
            lengths.append(len(target_text))
    finally:
        for f in files.values():
            f.close()

    print(f"Read {len(ids)} maps from {source.location}")
    for split in SPLITS:
        c = counts[split]
        print(f"  {split:5} {c['continuous'] + c['discrete']:6} maps "
              f"({c['discrete']} discrete, {c['continuous']} continuous) -> {args.out / (split + '.jsonl')}")
    if no_split:
        print(f"  skipped {len(no_split)} map(s) that no question file uses: {no_split[:10]}")
    if no_image:
        print(f"  skipped {len(no_image)} map(s) with no image: {no_image[:10]}")
    if lengths:
        print(f"  target length in characters: min {min(lengths)}, median {int(statistics.median(lengths))}, "
              f"max {max(lengths)}")


if __name__ == "__main__":
    main()
