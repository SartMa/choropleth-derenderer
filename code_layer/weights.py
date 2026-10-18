"""Spatial weights: the adjacency graph as the matrix the statistics take.

Two regions are neighbours if they share a border. Rows are standardised so each region's neighbours
sum to 1, which makes a spatial lag the mean of the neighbours.

Regions with no value are removed rather than kept at zero: a no-data region cannot contribute to a
neighbour's lag, and treating it as zero would invent a low value. Removing it can strand a region
whose only neighbours were all no-data, so the process repeats until every remaining region has at
least one neighbour.
"""

import numpy as np


class Weights:
    def __init__(self, codes, matrix, dropped):
        self.codes = list(codes)
        self.matrix = matrix              # row-standardised, float
        self.binary = (matrix > 0).astype(float)
        self.dropped = list(dropped)      # removed for having no neighbour left
        self.index = {c: i for i, c in enumerate(self.codes)}

    def __len__(self):
        return len(self.codes)

    def vector(self, values):
        return np.array([float(values[c]) for c in self.codes])

    def lag(self, x):
        return self.matrix @ x


def build(scope, codes):
    """Row-standardised weights over `codes`, after dropping regions left without a neighbour."""
    keep = [c for c in codes if scope.adjacency.get(c)]
    dropped = [c for c in codes if c not in keep]
    while True:
        present = set(keep)
        survivors = [c for c in keep if any(n in present for n in scope.adjacency[c])]
        if len(survivors) == len(keep):
            break
        dropped += [c for c in keep if c not in set(survivors)]
        keep = survivors

    size = len(keep)
    matrix = np.zeros((size, size))
    index = {c: i for i, c in enumerate(keep)}
    for code in keep:
        for neighbour in scope.adjacency[code]:
            if neighbour in index:
                matrix[index[code], index[neighbour]] = 1.0
    totals = matrix.sum(1, keepdims=True)
    if (totals == 0).any():
        raise AssertionError("a region survived with no neighbour; weights would divide by zero")
    return Weights(keep, matrix / totals, dropped)
