# Answer sheet v2: trained zoom for small regions (2026-11-05)

Code: `train/answer_v2.py`. Run: `skini501/mapqa-derender-v2-zoom` (head 30 min, then LoRA + head 4.46 h, 203 updates of
16 maps, from v1's weights). Data: 2,000 MapQA + 2,000 synthetic US + 2,000 synthetic India, a new draw. No China.
Vision LoRA loaded correctly (192 keys renamed, none at zero).

Regions the head finds less than 3 patches of are cropped where the head put them (12% of the longer side), enlarged
3x, re-read by the vision tower and head, and their token is pooled from that view. Trained the same way.

## Reading, v1 against v2 (same test maps)

| | v1 | v2 | Tiles + MLP picker | Region sheet |
|---|---|---|---|---|
| MapQA whole map exact (302) | 32.5% | **70.2%** | 89.7% | 74.8% |
| MapQA Rhode Island | 50.7% | **73.8%** | 94.4% | |
| MapQA Delaware | 73.5% | **94.7%** | 96.4% | |
| MapQA 8 small NE states | 87.7% | **95.0%** | | |
| MapQA other 42 states | 99.4% | 98.4% | | |
| MapQA continuous error | 3.65% | 3.79% | 3.2% | 6.0% |
| Synthetic India no codes, big (89 discrete) | 99.9% | 99.7% | | 100% |
| Synthetic India no codes, small | 89.2% | **93.3%** | | 94.8% |
| Synthetic India Delhi | 68.2% | **83.5%** | | |
| Synthetic India continuous error, big | 3.5% | 3.1% | | 3.2% |
| Real India strong discrete (31), big | **93.2%** | 89.5% | | 91.4% |
| Real India strong discrete, small | 72.3% | 76.1% | | **85.2%** |
| Real India Delhi / Goa / Sikkim | 35% / 71% / 87% | 52% / 81% / 94% | | 81% / 94% / 97% |
| Real India Andaman | **71.0%** | 58.1% | | 58.1% |
| Real India continuous (29 same maps), mean error | **12.6%** | 15.7% | | 13.4% (31) |
| Real India continuous, median error | 9.3% | **4.5%** | | |

India synthetic figures recomputed for both runs with one scorer (`lookup/score_india.py`: discrete maps
without printed codes, small = AN DL GA SK TR); v1 small is 89.2% here, 88.3% in v1's summary.

## What it shows

- Zoom trained works for small regions: every small region improved on all three sets, and MapQA whole-map exact
  more than doubled. Big states are almost unchanged on synthetic and MapQA (-1 point on MapQA's big states).
- Real India big states: 93.2% -> 89.5% is not a real drop (8 maps better, 8 worse, Wilcoxon p = 0.29). The mean
  falls because of a few whole-map shifts: inmates_male_2020 puts every lowest-class state in class 4 (93% -> 37%),
  discreate_4 shifts big states one class down, 19_d_1 writes class "200" for four states. Same kind of failure
  as the untrained zoom check, but rarer.
- Real India continuous: median error halves (9.3% -> 4.5%, better on 19/29 maps), but two maps blow up
  (map4 10% -> 71%, 12_1 20% -> 69%), which raises the mean.
- Against the pass mark set before the run (RI about 90%, real small about 85%, real big about 93%): not met.
  RI 73.8%, real small 76.1%, real big 89.5%. Tiles are still best on MapQA.
