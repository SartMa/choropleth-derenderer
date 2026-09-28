#!/usr/bin/env python3
"""Choose where to magnify a map, from the picture alone.

Small regions are where the derenderer fails: a state a few pixels wide is smaller than one 28x28-pixel
vision token, so its colour is averaged with its neighbours before the model sees it. Rather than naming
the crowded corner by hand (which would only ever be right for one country), this picks the tiles with the
densest region boundaries and lets the model magnify those.

    boxes = busiest_tiles(image, tiles=2)     # boxes in the image's own pixels, biggest crowding first

The legend is full of edges too, so boundaries are only counted inside the map body: the largest connected
patch of non-background pixels, which is the mainland because its states all touch.
"""

import numpy as np
from scipy import ndimage

TILE_FRACTION = 0.16   # a tile covers this share of the map body's width
STRIDE_FRACTION = 0.5  # windows step by half a tile


def map_body(image):
    """Mask of the mainland: the largest connected patch that is not the background colour."""
    pixels = np.asarray(image.convert("RGB"), dtype=np.int16)
    corners = np.stack([pixels[0, 0], pixels[0, -1], pixels[-1, 0], pixels[-1, -1]])
    background = np.median(corners, axis=0)
    filled = np.abs(pixels - background).max(axis=2) > 24
    labels, count = ndimage.label(filled)
    if count == 0:
        return np.ones(filled.shape, dtype=bool)
    sizes = ndimage.sum_labels(filled, labels, index=range(1, count + 1))
    return labels == (int(np.argmax(sizes)) + 1)


def boundary_mask(image, body):
    """Where region borders run: colour changes inside the map body."""
    grey = np.asarray(image.convert("L"), dtype=np.float32)
    steps = np.zeros_like(grey)
    steps[:-1, :-1] = np.abs(np.diff(grey, axis=0)[:, :-1]) + np.abs(np.diff(grey, axis=1)[:-1, :])
    return (steps > 20) & body


def busiest_tiles(image, tiles=2, tile_fraction=TILE_FRACTION):
    """Boxes over the parts of the map with the most region boundaries, densest first, non-overlapping."""
    body = map_body(image)
    rows, columns = np.where(body)
    if rows.size == 0:
        return []
    top, bottom, left, right = rows.min(), rows.max(), columns.min(), columns.max()
    size = max(28, int((right - left) * tile_fraction))
    stride = max(7, int(size * STRIDE_FRACTION))
    integral = boundary_mask(image, body).cumsum(0).cumsum(1)

    def density(y, x):
        y2, x2 = y + size, x + size
        return (integral[y2, x2] - integral[y, x2] - integral[y2, x] + integral[y, x])

    windows = [(density(y, x), x, y)
               for y in range(top, max(top + 1, bottom - size), stride)
               for x in range(left, max(left + 1, right - size), stride)
               if y + size < integral.shape[0] and x + size < integral.shape[1]]
    chosen = []
    for score, x, y in sorted(windows, reverse=True):
        if all(abs(x - bx) >= size * 0.6 or abs(y - by) >= size * 0.6 for bx, by, _, _ in chosen):
            chosen.append((x, y, x + size, y + size))
        if len(chosen) == tiles:
            break
    return chosen
