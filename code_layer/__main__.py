#!/usr/bin/env python3
"""Run the code layer over derenderer output.

    python -m code_layer outputs/mapqa-derender-qwen/qwen/predictions_mapqa.jsonl -o facts.jsonl
    python -m code_layer one_map.json --pretty
    echo '{"legend_type":"continuous",...}' | python -m code_layer -

Accepts a .jsonl of prediction records (the "prediction" field is used, as written by the training
scripts), a .jsonl of bare derenderer outputs, a single .json map, or stdin. Writes one result object
per input map.
"""

import argparse
import json
import sys
from pathlib import Path

from .adapter import ContractError
from .pipeline import DEFAULT_SEED, analyse
from .stats import DEFAULT_PERMUTATIONS


def read_maps(source):
    """(identifier, payload) per input map, whatever shape the file is in."""
    text = sys.stdin.read() if source == "-" else Path(source).read_text()
    stripped = text.strip()
    if not stripped:
        return
    if stripped.startswith("{") and "\n" not in stripped.rstrip():
        yield "map", stripped
        return
    for number, line in enumerate(stripped.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            yield f"line{number}", line
            continue
        if isinstance(record, dict) and "prediction" in record:
            yield record.get("id", f"line{number}"), record["prediction"]
        else:
            yield record.get("id", f"line{number}") if isinstance(record, dict) else f"line{number}", record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="a .jsonl or .json file of derenderer output, or - for stdin")
    parser.add_argument("-o", "--out", help="write results here instead of stdout")
    parser.add_argument("--scope", default="us_states")
    parser.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--pretty", action="store_true", help="indent the JSON")
    parser.add_argument("--limit", type=int, help="only the first N maps")
    args = parser.parse_args(argv)

    out = open(args.out, "w", encoding="utf-8") if args.out else sys.stdout
    analysed = failed = 0
    try:
        for identifier, payload in read_maps(args.source):
            if args.limit and analysed + failed >= args.limit:
                break
            try:
                result = analyse(payload, scope_name=args.scope,
                                 permutations=args.permutations, seed=args.seed)
                analysed += 1
            except ContractError as error:
                failed += 1
                result = {"error": str(error)}
            result = {"id": identifier, **result}
            out.write(json.dumps(result, indent=2 if args.pretty else None) + "\n")
    finally:
        if args.out:
            out.close()
    print(f"analysed {analysed} maps, {failed} rejected", file=sys.stderr)
    return 1 if analysed == 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
