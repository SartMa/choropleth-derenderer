# Learned answer sheet, first full run (2026-11-04)

Code: `train/answer_sheet.py`. Run: `sartma/mapqa-derender-answer-v1`. Labels: `lookup/answer_labels.py` (Kaggle `sartma/answer-labels`).

One model, no shapefile fit, OCR or legend crop at reading time. A region head on the 14-px patches says which region
(and country) each patch is; each region's patches are pooled into one token through Qwen's own merger; the model
reads "code: [token]" for every region plus the map. Shapefiles only made the training labels.

Training, from the 8-hour adapter: head alone 40 min (3,933 maps), then LoRA + head 4.45 h (227 updates of 16 maps) on
2,000 MapQA + 2,000 synthetic US + 2,000 synthetic India maps. China not used.

## Region head

| Set | Country right | Patch accuracy | Big-region token purity (median) | Small-region token purity (median) |
|---|---|---|---|---|
| Synthetic India, as drawn | 100% | 94.8% | 0.87 | 0.38 |
| Synthetic India, moved | 100% | 92.1% | 0.83 | 0.23 |
| MapQA test | 100% | 94.0% | 0.86 | 0.42 |
| Real India, 157 maps | 100% | 87.1% | 0.82 | 0.26 |

Purity: the share of a region token's pooling weight that falls on that region's own pixels. After stage B.

## Reading

| Real India, 31 strong discrete maps | Answer sheet | Region sheet | Old full run |
|---|---|---|---|
| Big states | **93.2%** | 91.4% | 62.4% |
| Small regions | 72.3% | **85.2%** | 67.1% |
| Andaman | 71.0% | 58.1% | 90.3% |
| Continuous error, big states (title-linked) | **12.6%** (29 maps) | 13.4% (31) | 28.7% |

Commonest-class baseline: 63.5%. Two real maps (17_1, map6) were not scored: their ranges ("< 5,000") don't parse in
`lookup/score_real.py`.

| Synthetic India, 300 test maps, no printed codes (177) | Answer sheet | Region sheet |
|---|---|---|
| Big states | 99.9% | 100% |
| Small regions | 88.3% | **94.8%** |
| Andaman | **83.1%** | 74.2% |
| Continuous error, big states | 3.5% | 3.2% |
| Continuous within 5% | 74.0% | 80.5% |

| MapQA, 400 test maps | Answer sheet | Tiles + MLP picker | 8-hour adapter, no tiles |
|---|---|---|---|
| Whole map exact (302 discrete) | 32.5% | **89.7%** | 33.1% |
| States right | 97.5% | **99.3%** | 97.8% |
| Rhode Island | 50.7% | **94.4%** | 47.0% |
| Delaware | 73.5% | **96.4%** | |
| Continuous error | 3.65% | **3.2%** | 4.1% |

## Verdict

- Works for big regions everywhere, with no fit and no OCR: best real-India big-state score so far (93.2%).
- Small regions are the gap. Their tokens are mostly other regions' pixels (purity 0.2-0.4), because a 14-px patch is
  still about as big as Rhode Island. On MapQA this leaves Rhode Island at the no-tiles level, so whole-map exact is
  no better than the plain model.
- Next: give the head a closer look at small regions (zoomed views where they sit, the tiles idea), and pool small
  regions' tokens from those views.
