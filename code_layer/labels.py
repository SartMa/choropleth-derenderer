"""Describe where a cluster is, in words a reader would use.

Two different questions decide the wording, and using only the first is how a two-state cluster ends
up being called "the South":

    precision - how much of the CLUSTER lies in the named region
    recall    - how much of the REGION the cluster covers

Both must be high before a cluster can be called "the South". High precision with low recall means the
cluster is *in* the South without being it. Where no named label fits, or the geography has no official
grouping at all, the cluster falls back to a compass direction, which works for any scope.
"""

PRECISION_MIN = 0.70
RECALL_FULL = 0.70
RECALL_MUCH = 0.50


def _score(cluster, members):
    shared = len(cluster & members)
    if not shared:
        return 0.0, 0.0, 0.0
    precision = shared / len(cluster)
    recall = shared / len(members)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def label_cluster(codes, scope):
    """The best macro-region label for a cluster, or a compass direction when none fits.

    Divisions are tried before regions: a cluster covering all six New England states is better
    described as New England than as the Northeast, which it only two-thirds covers.
    """
    cluster = set(codes)
    if not cluster:
        return None
    candidates = []
    for level in ("division", "region"):
        for label, members in scope.groups(level).items():
            precision, recall, f1 = _score(cluster, set(members))
            if precision < PRECISION_MIN and recall < RECALL_FULL:
                continue
            if precision >= PRECISION_MIN:
                kind = "whole" if recall >= RECALL_FULL else (
                    "much" if recall >= RECALL_MUCH else "part")
            else:
                kind = "beyond"
            candidates.append((f1, 0 if level == "division" else 1,
                               {"label": label, "level": level, "coverage": kind,
                                "precision": round(precision, 3), "recall": round(recall, 3)}))
    if not candidates:
        return direction_label(cluster, scope)
    # Best fit wins; a tie goes to the narrower label, so a cluster covering exactly New England is
    # not called the Northeast, while one covering the whole South is not called South Atlantic.
    return min(candidates, key=lambda c: (-c[0], c[1]))[2]


def direction_label(codes, scope):
    """Fallback: which part of the map the cluster sits in, from centroids.

    Works for any geography, including those with no official regional scheme.
    """
    cluster = [c for c in codes if c in scope.centroids]
    if not cluster:
        return None
    every = [scope.centroids[c] for c in scope.codes if scope.adjacency.get(c)]
    if not every:
        return None
    xs = [p["x"] for p in every]
    ys = [p["y"] for p in every]
    span_x = (max(xs) - min(xs)) or 1.0
    span_y = (max(ys) - min(ys)) or 1.0
    mean_x = sum(scope.centroids[c]["x"] for c in cluster) / len(cluster)
    mean_y = sum(scope.centroids[c]["y"] for c in cluster) / len(cluster)
    offset_x = (mean_x - (max(xs) + min(xs)) / 2) / span_x
    offset_y = (mean_y - (max(ys) + min(ys)) / 2) / span_y

    words = []
    if offset_y > 0.15:
        words.append("northern")
    elif offset_y < -0.15:
        words.append("southern")
    if offset_x > 0.15:
        words.append("eastern")
    elif offset_x < -0.15:
        words.append("western")
    if not words:
        words.append("central")
    return {"label": " ".join(words), "level": "direction", "coverage": "approximate",
            "precision": None, "recall": None}


def phrase(label):
    """How the narrator should word the label; never asserts more coverage than was measured."""
    if not label:
        return None
    name = label["label"]
    return {"whole": f"the {name}", "much": f"much of the {name}", "part": f"part of the {name}",
            "beyond": f"the {name} and neighbouring areas",
            "approximate": f"the {name} states"}.get(label["coverage"], name)
