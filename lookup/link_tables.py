"""Link each real India map to the source table and column it was drawn from, giving real maps true values.

Uses only text: the title and the legend's printed numbers, read by OCR from the original image. State colours are never
used, because reading them is what these values will score.

A strong link means the title picks the table and both ends of the legend equal one column's lowest and highest values.
A title link means the title picks the table and the printed numbers fit inside the column; continuous bars have round
ticks that stop short of the ends, so these are weaker.

    .venv/bin/python lookup/link_tables.py --out outputs/real-fit
"""

import argparse
import csv
import difflib
import json
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

CMA = Path("data/cma/India")
TABLES = Path("data/cma_tables")
GEOJSON = Path("data/boundaries/geoBoundaries-IND-ADM1_simplified.geojson")
FOLDERS = ["Legend Classification/train", "Legend Classification/val", "State Segmentation/test"]
STOP = set("state states ut uts wise number of the in during as on and for by to from total no details".split())
ALIASES = {"nct of delhi": "delhi", "delhi ut": "delhi", "a and n islands": "andaman and nicobar islands",
           "andaman and nicobar": "andaman and nicobar islands", "j and k": "jammu and kashmir", "orissa": "odisha",
           "uttaranchal": "uttarakhand", "pondicherry": "puducherry", "chattisgarh": "chhattisgarh",
           "dandn haveli": "dadra and nagar haveli and daman and diu",
           "dadra and nagar haveli": "dadra and nagar haveli and daman and diu",
           "d and n haveli": "dadra and nagar haveli and daman and diu", "telengana": "telangana"}
READER = None


def ascii_name(text):
    import unicodedata
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().casefold().replace("&", " and ")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", text)).strip()


def state_codes():
    features = json.loads(GEOJSON.read_text())["features"]
    return {ascii_name(f["properties"]["shapeName"]): f["properties"]["shapeISO"].split("-")[1] for f in features}


CODES = state_codes()


def code_of(name):
    name = ascii_name(name)
    name = ALIASES.get(name, name)
    close = difflib.get_close_matches(name, CODES, n=1, cutoff=0.82)
    return CODES[close[0]] if close else None


def number(text):
    try:
        return float(str(text).replace(",", "").replace("%", "").strip())
    except ValueError:
        return None


def read_table(path):
    """{column header: {code: value}} for the columns that hold numbers per region."""
    rows = list(csv.reader(open(path, encoding="utf-8-sig", errors="replace")))
    header = next(i for i, r in enumerate(rows) if sum(bool(c.strip()) for c in r) >= 2)
    names = rows[header]
    best = Counter()
    for row in rows[header + 1:]:
        for i, cell in enumerate(row):
            if cell.strip() and code_of(cell):
                best[i] += 1
    if not best or best.most_common(1)[0][1] < 15:
        return {}
    column = best.most_common(1)[0][0]
    answer = {}
    for i, name in enumerate(names):
        if i == column:
            continue
        values = {}
        for row in rows[header + 1:]:
            if len(row) > max(i, column) and number(row[i]) is not None:
                code = code_of(row[column])
                if code:
                    values[code] = number(row[i])
        if len(values) >= 15 and not re.search(r"s\.? ?no|sl\.? ?no|si\.? ?no", name.casefold()):
            answer[name.strip()] = values
    return answer


def originals():
    paths = {}
    for folder in FOLDERS:
        for path in sorted((CMA / folder).rglob("*")):
            if path.suffix.lower() in (".png", ".jpg", ".jpeg"):
                paths.setdefault(path.stem, path)
    return paths


def read_text(path):
    """(title, numbers printed outside the title) by OCR."""
    global READER
    if READER is None:
        from rapidocr_onnxruntime import RapidOCR
        READER = RapidOCR()
    import cv2
    image = cv2.imread(str(path))
    found = READER(image)[0] or []
    height = image.shape[0]
    tall = max((abs(c[2][1] - c[0][1]) for c, text, _ in found if c[0][1] < 0.2 * height and len(text.split()) >= 3),
               default=0)
    title, numbers = [], []
    for corners, text, _ in found:
        size = abs(corners[2][1] - corners[0][1])
        if corners[0][1] < 0.2 * height and len(text.split()) >= 3 and size >= 0.6 * tall:
            title.append(text)
        else:
            numbers += [float(n.replace(",", "")) for n in re.findall(r"\d[\d,]*\.?\d*", text.replace(",", ""))]
    return {"name": path.stem, "title": " ".join(title), "numbers": numbers}


def words(text):
    return {w for w in re.findall(r"[a-z]+|\d{4}", text.casefold()) if w not in STOP}


def similar(a, b):
    a, b = words(a), words(b)
    return len(a & b) / max(len(a | b), 1)


def present(value, numbers):
    return any(abs(n - value) <= max(1.0, 0.005 * abs(value)) for n in numbers)


def link(page, tables):
    """The best table and column for one map: tier strong, title or none."""
    title = page["title"]
    if len(words(title)) < 3 and not re.fullmatch(r"(map)?[\d_]+(d_)?\d*", page["name"]):
        title = page["name"].replace("_", " ")  # OCR found no title: a descriptive file name stands in
    top = max(similar(title, t["name"]) for t in tables)
    ranked = [t for t in tables if similar(title, t["name"]) >= 0.8 * top]  # legend numbers must not outvote the title
    candidates = []
    for table in ranked:
        score = similar(title, table["name"])
        for column, values in table["columns"].items():
            numbers, low, high = page["numbers"], min(values.values()), max(values.values())
            if high <= low:
                continue
            ends = present(low, numbers) + present(high, numbers)
            inside = [n for n in numbers if low <= n <= high]
            candidates.append({"table": table["name"], "column": column, "title_score": round(score, 3), "ends": int(ends),
                               "inside": len(inside), "low": low, "high": high, "values": values,
                               "column_score": similar(title, column)})
    if not candidates:
        return {"tier": "none"}
    best = max(candidates, key=lambda c: (c["ends"], c["title_score"], c["column_score"], c["inside"]))
    runners = [c for c in candidates if c["values"] != best["values"] and c["ends"] == best["ends"]
               and c["title_score"] == best["title_score"] and c["column_score"] == best["column_score"]]
    best["tier"] = "none"
    if best["title_score"] >= 0.3 and not runners:
        if best["ends"] == 2:
            best["tier"] = "strong"
        elif best["inside"] >= 3:
            best["tier"] = "title"
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("outputs/real-fit"))
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    cache = args.out / "map_text.json"
    if cache.exists():
        pages = json.loads(cache.read_text())
    else:
        with ProcessPoolExecutor(args.workers) as pool:
            pages = list(pool.map(read_text, originals().values()))
        cache.write_text(json.dumps(pages, indent=1))
    tables = []
    for entry in json.loads((TABLES / "index.json").read_text()):
        columns = read_table(TABLES / "raw" / entry["id"])
        if columns:
            tables.append({"name": entry["name"], "columns": columns})
    links = {p["name"]: {"title": p["title"], **link(p, tables)} for p in pages}
    (TABLES / "links.json").write_text(json.dumps(links, indent=1))
    tiers = Counter(l["tier"] for l in links.values())
    print(f"{len(tables)} tables usable, {len(pages)} maps, links {dict(tiers)}")
    print("regions per strong link", np.median([len(l["values"]) for l in links.values() if l["tier"] == "strong"] or [0]))


if __name__ == "__main__":
    main()
