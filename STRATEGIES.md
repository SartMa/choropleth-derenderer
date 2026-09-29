# Reading small regions: strategy comparison

**Result:** a learned picker that chooses where to magnify, feeding magnified tiles to the model as extra
images, reads Rhode Island correctly on **94.4%** of maps (baseline 47.0%) and gets the **whole map exactly
right on 89.7%** (baseline 33.1%). Fusing the tiles into the model's own tokens did not work.

All numbers below come from the same 400 MapQA test maps (302 discrete, 98 continuous), and every model
continues from the same 8-hour Qwen2.5-VL-3B LoRA adapter.

---

## 1. The problem

Qwen2.5-VL turns each 28x28-pixel square of the image into one token. Rhode Island covers about 9x13 pixels,
so its colour is averaged with its neighbours before the language model sees it. On the baseline, 48 of 50
states are read almost perfectly; Rhode Island is right on 47% of discrete maps and Delaware on 74%.
Training longer does not help (section 4, "more data, no tiles").

## 2. The strategies

| Strategy | Who picks where to zoom | How the zoom reaches the model | New weights | Level |
|---|---|---|---|---|
| Baseline | nobody | - | - | - |
| Tiles + image (v1) | rule: densest borders | tiles as extra images | none | input |
| Tiles + positions + image (v2) | rule: most regions smaller than a token | tiles as extra images, position in prompt | none | input |
| Fusion | rule (v2) | tile features added into the map's own tokens | 5.2M | model (combining) |
| **MLP picker** | **learned: small MLP on the map's vision tokens** | tiles as extra images, position in prompt | 0.5M | **model (picking)** |

The MLP picker reads the 450 vision tokens the model has already computed for the map, predicts how many
small regions sit in each token's square, and the best-scoring windows become the tiles. It is trained in
minutes on 1,800 maps, with the rule's counts as labels. No image-processing code runs at test time.

## 3. Headline comparison (same 400 maps)

| Strategy | Training after the 8h adapter | Rhode Island | Delaware | All 50 states | Whole map exact | Continuous error (median) | Continuous within 5% |
|---|---|---|---|---|---|---|---|
| Baseline, no tiles | - | 47.0% | 74.2% | 97.83% | 33.1% | 2.98% | 85.0% |
| Fusion (retry, fixed) | 1.5 h | 47.0% | 76.8% | 97.26% | 26.5% | 3.25% | 81.6% |
| Rule tiles (v2) | 1.5 h | 80.8% | 94.4% | 99.03% | 73.8% | 2.44% | 88.4% |
| Rule tiles (v2) | 7 h, more data | 83.8% | 95.7% | 99.05% | 79.8% | 2.12% | 90.0% |
| **MLP picker tiles** | **7 h, more data** | **94.4%** | **96.4%** | **99.34%** | **89.7%** | **2.01%** | **90.4%** |

Continuous error is each state's error as a share of the map's value range (0% = exact).

### All metrics

| Metric | Baseline | Fusion | Rule v2 (1.5 h) | Rule v2 (7 h) | MLP picker (7 h) |
|---|---|---|---|---|---|
| **Discrete maps (302)** | | | | | |
| Valid JSON | 100% | 100% | 100% | 100% | 100% |
| Title correct | 100% | 100% | 100% | 100% | 100% |
| Legend ranges correct | 99.0% | 99.3% | 99.7% | 99.0% | 99.0% |
| All 50 states | 97.83% | 97.26% | 99.03% | 99.05% | **99.34%** |
| Whole map exact | 33.1% | 26.5% | 73.8% | 79.8% | **89.7%** |
| Rhode Island | 47.0% | 47.0% | 80.8% | 83.8% | **94.4%** |
| Delaware | 74.2% | 76.8% | 94.4% | 95.7% | **96.4%** |
| New Hampshire | 91.7% | 93.1% | 96.7% | 97.0% | **99.3%** |
| Connecticut | 96.0% | 95.0% | 96.4% | **99.0%** | **99.0%** |
| Maryland | 97.7% | 95.4% | 98.0% | 98.0% | **98.7%** |
| **Continuous maps (98)** | | | | | |
| Mean error | 4.12% | 3.45% | **2.57%** | 3.31% | 3.24% |
| Median error | 2.98% | 3.25% | 2.44% | 2.12% | **2.01%** |
| Within 5% | 85.0% | 81.6% | 88.4% | 90.0% | **90.4%** |
| Rhode Island error | 25.1% | 25.9% | 8.9% | 7.5% | **5.5%** |
| Delaware error | 12.9% | 10.5% | 5.1% | 4.8% | **4.6%** |

The 7-hour models' continuous *mean* is pulled up by one map whose legend is in millions: the model wrote
3.62 where the answer was 3,620,000, an error capped at 100%. The median, and the per-map comparison
(section 4), show the 7-hour models are better on continuous maps.

## 4. What each change did (one variable at a time)

Each row compares two runs on the same maps, differing in one thing. "Fixed" = maps the first run gets
right and the second gets wrong; "broke" = the reverse. p is an exact two-sided binomial test on those
counts: how likely a split this lopsided is if the change did nothing.

| Change | Rhode Island fixed / broke | p | Whole map exact fixed / broke | p | Continuous maps better / worse |
|---|---|---|---|---|---|
| **MLP picker instead of rule** (same weights) | **38 / 6** | 1e-6 | **41 / 11** | 4e-5 | 55 / 33 |
| Tiles on vs off (same weights) | 125 / 5 | 4e-31 | 146 / 2 | 6e-41 | 74 / 22 |
| MLP picker model vs baseline | 148 / 5 | 1e-37 | 175 / 4 | 1e-46 | 82 / 14 |
| More data and training, with tiles (7 h vs 1.5 h) | 17 / 8 | 0.11 | 24 / 6 | 0.001 | 69 / 27 |
| More data and training, no tiles | 13 / 22 | 0.18 | 14 / 17 | 0.72 | 57 / 38 |
| Fusion on vs off (same weights) | 6 / 3 | 0.51 | 3 / 25 | 3e-5 | 47 / 44 |

Reading it:
- **Tiles are what fix small states.** Same weights, tiles removed: back to baseline.
- **The learned picker beats the rule it was taught by**, on the same model and the same maps.
- **More data helps only together with tiles**: it lifts whole-map accuracy, not the small states alone.
- **Fusion does nothing for Rhode Island** and makes whole maps worse.

## 5. Tile settings (rule v2, 1.5 h each, same maps)

| Tiles | Zoom | Tokens per map | Rhode Island | Whole map exact | vs 2 tiles at 3x (RI fixed / broke) |
|---|---|---|---|---|---|
| 1 | 3x | ~550 | 80.5% | 71.5% | 11 / 12 (no difference) |
| **2** | **3x** | **~650** | **80.8%** | **73.8%** | reference |
| 3 | 3x | ~750 | 78.8% | 69.9% | 13 / 19 (no gain) |
| 2 | 2x | ~550 | 59.9% | 51.7% | 13 / 76 (much worse, p = 5e-12) |

**Magnification matters; the number of tiles does not** (US small states sit in one cluster). So tiles
should go where the map is dense and nowhere else, at 3x.

## 6. What did not work, and why

| Attempt | Rhode Island | Why it failed |
|---|---|---|
| More training alone | ~44-47% | the information is lost in the image tokens; more steps cannot recover it |
| S2 (whole map at 2x, averaged into the tokens) | 41.7% | averaging smears the small state across every token again; 8.4M new weights, too few steps |
| Hand-made north-east crop | 60.4% | helps, but the box is measured for the US and hurt continuous maps |
| Fusion, first try | 43.8% | output layer started at exactly zero, which also blocked the first layer's learning |
| Fusion, retry (10x learning rate, non-zero start) | 47.0% | the layer now learns, but the model cannot use the fused signal |
| MLP picker, first try | 51.9% | my bug: labels were yes/no per token, so the 8-state north-east looked no busier than a lake shore |

(The first three rows are from earlier runs on smaller test subsets, so compare them loosely.)

The pattern across all of them: **anything that reuses the pretrained parts works; anything that needs new
weights trained through the language model's answer did not learn in the budget we had.** The MLP picker is
the exception because it gets its own direct labels on every token.

## 7. Caveats

- **The picker beating its teacher is not fully explained.** It was trained to copy the rule. A likely reason
  is that it learned "small states are usually in the north-east" from US-only maps, which helps here but
  may not carry to other countries. Untested.
- **One geography.** Everything above is US maps. India maps exist (synthetic, 32 regions) but the rule
  misses Delhi and Goa there, and the picker has not been tried on them.
- **400 test maps**, one random sample, the same for every row. Paired differences above are solid; the
  absolute percentages would move a little on the full 4,157-map test set.
- The 7-hour runs saw about 55-60% of the training data once; neither is fully trained.

## 8. Recommendation for the full GPU run

**Tiles as extra images, picked by the MLP picker, 3x zoom, trained on the full dataset (MapQA + synthetic).**
Before that run: one small check of the picker on India maps, to know whether it generalises beyond the US.

---

### Setup, for reference

- Model: Qwen2.5-VL-3B-Instruct, LoRA r=16 on the language model and vision MLPs, fp16 on a Kaggle T4.
- All runs continue from the same 8-hour adapter (8,640 training maps seen).
- "1.5 h" runs: MapQA training maps only. "7 h" runs: MapQA + synthetic US + 1,000 synthetic India maps.
- Tiles: 2 per map, 3x zoom, about 90x90 pixels of the original map each, plus their position in the prompt.
- MLP picker: 2048 -> 256 -> 1 per token, Poisson loss on the rule's small-region counts, 1,800 training maps,
  evaluated with no retraining of the language model (same weights as the rule-picker row).
- Scoring: each state's answer against MapQA's labels; discrete = right range, continuous = error / value range.
