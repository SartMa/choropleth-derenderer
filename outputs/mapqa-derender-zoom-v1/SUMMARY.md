# Zoom check on the answer-sheet v1 model, no training (2026-11-04)

Code: `train/answer_zoom.py`. Run: `sartma/mapqa-derender-zoom-v1` (3.7 h). Weights: v1 as saved.

**Caveat (found later):** this run loaded v1's adapter without its vision LoRA. v1 saved those 192 keys as
`visual.visual.*`, and peft left them at zero with only a warning. So the head and reader here saw different vision
features from the ones v1 was trained on, which explains at least part of the drops below. Fixed in
`answer_sheet.load_adapter`.

Regions the head finds fewer than 3 patches of are looked at again in a 3x view around where the head put them,
and their token is pooled from that view. Two ways of picking the region's patches in the view: the head, or the
true labels (a reference only).

## Purity of zoomed tokens (median share of the token that is the region's own pixels)

| Region | Before | Zoom, head picks | Zoom, true labels |
|---|---|---|---|
| Rhode Island | 0.13 | 0.00 | 0.56 |
| Delaware | 0.16 | 0.01 | 0.60 |
| Delhi | 0.12 | 0.00 | 0.64 |
| Goa | 0.16 | 0.00 | 0.74 |
| Sikkim | 0.46 | 0.00 | 0.81 |

## Reading

| | v1 | Zoom, head | Zoom, true labels |
|---|---|---|---|
| MapQA whole map exact | 32.5% | 20.5% | 31.1% |
| MapQA Rhode Island | 50.7% | 44.0% | 57.0% |
| India (no codes) big states | 99.9% | 81.0% | 82.6% |
| India (no codes) Delhi | 66% | 36% | 51% |
| India (no codes) Tripura | 100% | 31% | 55% |
| India discrete maps with >= 95% states right | 133 / 159 | | 15 / 159 |

## What it shows

- The head cannot find regions in zoomed views: it was never taught on them, and purity falls to about 0.
- With true labels, zoomed tokens are much cleaner (0.6-0.8), and the zoomed region is often read right
  (synthind_test_000334: Delhi right only with zoom). But the model was never trained on tokens made this way,
  and they upset the whole answer: big states that were not zoomed shift by one class.
- So zooming without training does not work, for both reasons. It is not evidence against zooming; both parts
  (head and reader) need to see zoomed views in training. Labels for zoomed views are free from the same shapefiles.
