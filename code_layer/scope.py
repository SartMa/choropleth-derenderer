"""Static reference data for one geography, loaded from code_layer/reference/<scope>/."""

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

REFERENCE = Path(__file__).resolve().parent / "reference"


@dataclass(frozen=True)
class Scope:
    name: str
    codes: tuple                 # canonical region order; every input must match this exactly
    adjacency: dict              # code -> [neighbouring codes]
    centroids: dict              # code -> {x, y, lon, lat}; x/y are metres, equal-area
    macro_regions: dict          # code -> {region, division}

    @property
    def mainland(self):
        """Regions with at least one land neighbour; contiguity statistics only apply to these."""
        return tuple(c for c in self.codes if self.adjacency[c])

    @property
    def offshore(self):
        """Regions with no land neighbour (Alaska, Hawaii, islands). Compared as a group instead."""
        return tuple(c for c in self.codes if not self.adjacency[c])

    def groups(self, level):
        """{label: [codes]} for 'region' or 'division'."""
        out = {}
        for code in self.codes:
            out.setdefault(self.macro_regions[code][level], []).append(code)
        return out


@lru_cache(maxsize=None)
def load_scope(name):
    folder = REFERENCE / name
    if not folder.is_dir():
        known = sorted(p.name for p in REFERENCE.iterdir() if p.is_dir() and p.name != "__pycache__")
        raise ValueError(f"unknown map_scope {name!r}; built scopes: {known}")

    def read(filename):
        return json.loads((folder / filename).read_text())

    adjacency = read("adjacency.json")
    return Scope(name=name, codes=tuple(sorted(adjacency)), adjacency=adjacency,
                 centroids=read("centroids.json"), macro_regions=read("macro_regions.json"))
