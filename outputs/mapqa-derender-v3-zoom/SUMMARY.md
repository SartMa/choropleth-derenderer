# Answer sheet v3: S2 grid for finding, zoom sized per region (2026-11-05)

Code: `train/answer_v3.py`. Runs (skini501, both from v2's weights, head 30 min then LoRA + head 3.75 h):
- `mapqa-derender-v3-zoom` (mode zoom, 113 updates): head on the S2 2x tiled grid; every region's token from that
  grid; regions under 3 (1x) patches re-read at a zoom sized to cover ~8 patches (2-8x), kept only if the head's
  belief rises.
- `mapqa-derender-v3-s2` (mode s2, 130 updates): S2 grid for head and every token, no zoom. Outputs in
  `outputs/mapqa-derender-v3-s2/`.

## Reading (same test maps for every run)

| | v1 | v2 | v3 S2 only | **v3 S2 + zoom** | Tiles + MLP | Region sheet |
|---|---|---|---|---|---|---|
| MapQA whole map exact (302) | 32.5% | 70.2% | 71.5% | **92.1%** | 89.7% | 74.8% |
| MapQA Rhode Island | 50.7% | 73.8% | 82.5% | **97.4%** | 94.4% | |
| MapQA Delaware | 73.5% | 94.7% | 90.1% | **98.7%** | 96.4% | |
| MapQA 8 small NE states | 87.7% | 95.0% | 95.0% | **99.0%** | | |
| MapQA other 42 states | 99.4% | 98.4% | 98.1% | 99.4% | | |
| MapQA continuous error | 3.65% | 3.79% | 2.97% | **2.76%** | 3.2% | 6.0% |
| Synthetic India no codes, big | 99.9% | 99.7% | 99.7% | 99.5% | | 100% |
| Synthetic India no codes, small | 89.2% | 93.3% | 93.8% | 93.8% | | 94.8% |
| Synthetic India Delhi / Andaman | 68% / 85% | 84% / 86% | 77% / 93% | 84% / 87% | | |
| Real India strong discrete (31), big | 93.2% | 89.5% | **93.9%** | 91.5% | 62.4% (old full run) | 91.4% |
| Real India strong discrete, small | 72.3% | 76.1% | 77.4% | 77.4% | | **85.2%** |
| Real India Delhi / Goa / Sikkim / Andaman | 35/71/87/71% | 52/81/94/58% | 42/84/100/61% | 45/81/100/68% | | 81/94/97/58% |
| Real India continuous, mean / median error (31) | 12.6% / 9.3% (29) | 15.2% / 6.3% | **10.8% / 6.3%** | 17.2% / 9.3% | | 13.4% |
| Real discrete maps with big < 60% | 1 | 2 | 1 | 1 | | |

v3 against tiles, map by map on MapQA: v3 alone right on 20 maps, tiles alone on 13 (sign test p = 0.30), so level
with tiles, not proven better. S2 only against tiles: 10 against 65 (p < 0.001).

## Finding small regions (head report, peak belief >= 2%)

| | 1x grid | S2 grid |
|---|---|---|
| Real India Delhi, 157 maps (v3 S2 only, after training) | 101 | **157** |
| Real India Goa | 77 | **157** |
| MapQA Rhode Island, 100 maps | 86-100 | **100** |

S2 fixes finding: every small region is now found on every map. In v3, Delhi was zoomed on 65 of 68 real maps
(v2: 10).

## What it shows

- On MapQA, S2 + zoom is the best result so far and level with tiles (92.1% vs 89.7% exact), while also working on
  India, where tiles do not. Continuous error is the lowest of any run.
- S2 alone finds every small region but cannot read the tiniest (RI 82.5%); the sized zoom is what reads them.
- Real India small regions are stuck at ~77% for v2, S2 and v3; Delhi is the gap (45% vs the sheet's 81%). Delhi is
  now found and zoomed, but on real maps its zoomed token is only 0.43 Delhi (Rhode Island on MapQA: 0.79). The
  problem has moved from finding to picking Delhi's pixels inside the zoomed real view.
- v3's head is weaker on whole real maps than S2-only's (patch accuracy 78% vs 89% after training): the 40% of
  head steps spent on zoomed views cost full-map accuracy on real styles. S2-only has the best real big states
  (93.9%) and real continuous error.
- Pass mark set before v2: Rhode Island ~90% met (97.4%); real small ~85% not met (77.4%); real big ~93% not met
  for v3 (91.5%), met by S2 only (93.9%).
