"""Turn the derenderer's JSON into the canonical input the detectors work on.

The derenderer emits, per map:

    {"title": ..., "legend_title": ..., "legend_type": "continuous"|"discrete",
     "data_type": "percentage"|"number", "ranges": [...],      # discrete only
     "values": {"AL": 34.2, ...}}                              # or a bin index, for discrete

Continuous maps give a number per region. Discrete maps give the index of the region's legend band, so
the analysis value is the midpoint of that band where the label can be parsed. If any band is open-ended
("> 30") or unparseable, the whole map falls back to bin indices: mixing midpoints and indices in one
map would put the regions on two different scales.

This is the only place region codes and value scales are translated. Everything downstream sees MapData.
"""

import json
import re
from dataclasses import dataclass, field

from .scope import load_scope

DASHES = {"–": "-", "—": "-", "−": "-", "‒": "-"}
OPEN_ENDED = re.compile(r"^\s*(<|>|≤|≥|under|over|less than|more than|at least|"
                        r"up to|.*\+$|.*or more|.*or less|.*and above|.*and below)", re.I)


class ContractError(ValueError):
    """The input does not satisfy the contract; the caller must not proceed."""


@dataclass
class MapData:
    scope_name: str
    title: str
    legend_title: str | None
    legend_type: str
    data_type: str
    ranges: list | None
    values: dict                  # code -> float or None; what the statistics run on
    reported: dict                # code -> what the derenderer literally said
    value_kind: str               # "value" | "band_midpoint" | "band_index"
    band_labels: dict = field(default_factory=dict)   # code -> its legend label, discrete only
    # The derenderer reads the darkest and lightest region directly off the image. That is a separate
    # observation from the 50 values, so where the two disagree the extremum is genuinely uncertain.
    reported_extremes: dict = field(default_factory=dict)   # {"max": code|None, "min": code|None}

    @property
    def scope(self):
        return load_scope(self.scope_name)

    @property
    def observed(self):
        """Codes with a value; regions reported as no-data are excluded from every statistic."""
        return [c for c in self.scope.codes if self.values[c] is not None]

    @property
    def missing(self):
        return [c for c in self.scope.codes if self.values[c] is None]

    @property
    def is_banded(self):
        return self.value_kind != "value"


def clean_number(text):
    text = re.sub(r"[%$€£,\s]", "", text)
    return float(text)


def parse_band(label):
    """(low, high) for a closed band label, or None when it is open-ended or unparseable."""
    text = str(label)
    for dash, plain in DASHES.items():
        text = text.replace(dash, plain)
    if OPEN_ENDED.match(text):
        return None
    if " to " in text.lower():
        left, right = re.split(r"\s+to\s+", text, maxsplit=1, flags=re.I)
    else:
        # split on a "-" that separates two numbers, not one that signs the first number
        match = re.match(r"^\s*(-?[^-]+?)\s*-\s*(-?.+?)\s*$", text)
        if not match:
            return None
        left, right = match.group(1), match.group(2)
    try:
        low, high = clean_number(left), clean_number(right)
    except ValueError:
        return None
    return (low, high) if high >= low else (high, low)


def band_midpoints(ranges):
    """Midpoint per band, or None if any band is open-ended or unparseable."""
    parsed = [parse_band(label) for label in ranges]
    if any(p is None for p in parsed):
        return None
    return [(low + high) / 2 for low, high in parsed]


def from_derenderer(payload, scope_name, strict=True):
    """Validate one derenderer output and convert it. Raises ContractError rather than guessing."""
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ContractError(f"output is not JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ContractError("output is not a JSON object")

    scope = load_scope(scope_name)
    legend_type = payload.get("legend_type")
    if legend_type not in ("continuous", "discrete"):
        raise ContractError(f"legend_type must be continuous or discrete, got {legend_type!r}")

    reported = payload.get("values")
    if not isinstance(reported, dict):
        raise ContractError("values must be an object keyed by region code")
    unknown = sorted(set(reported) - set(scope.codes))
    absent = sorted(set(scope.codes) - set(reported))
    if strict and (unknown or absent):
        raise ContractError(f"values must cover exactly the {len(scope.codes)} regions of "
                            f"{scope_name}; missing {absent}, unexpected {unknown}")

    ranges = payload.get("ranges")
    band_labels = {}
    if legend_type == "discrete":
        if not isinstance(ranges, list) or not ranges:
            raise ContractError("a discrete legend needs a non-empty ranges list")
        midpoints = band_midpoints(ranges)
        value_kind = "band_midpoint" if midpoints else "band_index"
        values = {}
        for code in scope.codes:
            index = reported.get(code)
            if index is None or not isinstance(index, int) or isinstance(index, bool):
                values[code] = None
                continue
            if not 0 <= index < len(ranges):
                raise ContractError(f"{code}: band index {index} outside 0..{len(ranges) - 1}")
            band_labels[code] = ranges[index]
            values[code] = midpoints[index] if midpoints else float(index)
    else:
        value_kind = "value"
        values = {}
        for code in scope.codes:
            raw = reported.get(code)
            ok = isinstance(raw, (int, float)) and not isinstance(raw, bool)
            values[code] = float(raw) if ok else None

    extremes = {}
    for key, field_name in (("max", "max_state"), ("min", "min_state")):
        claimed = payload.get(field_name)
        if claimed is None:
            continue
        if claimed not in scope.codes:
            raise ContractError(f"{field_name}={claimed!r} is not a region of {scope_name}")
        extremes[key] = claimed

    return MapData(scope_name=scope_name, title=payload.get("title") or "",
                   legend_title=payload.get("legend_title"), legend_type=legend_type,
                   data_type=payload.get("data_type") or "unknown",
                   ranges=ranges if legend_type == "discrete" else None,
                   values=values, reported={c: reported.get(c) for c in scope.codes},
                   value_kind=value_kind, band_labels=band_labels, reported_extremes=extremes)
