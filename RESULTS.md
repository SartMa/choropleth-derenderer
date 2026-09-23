# Results: what we measured, what it means, and where we stand

This document explains every number the derenderer produced, how each was calculated, what it does and
does not prove, and an honest read on what is presentable today.

Short version: **stage 1 (reading a map into a table) works and is measured**. Range-legend maps are
read almost perfectly. Smooth colour-bar maps are good but clearly weaker, and we know exactly why. The
project is presentable as a finished, measured stage 1 — not as a finished pipeline.

---

## 1. What was run

| | |
|---|---|
| Model | Qwen2.5-VL-3B, LoRA adapters (37M trained of 3.8B) |
| Hardware | one Kaggle T4, fp16 |
| Training | 8.0 hours, 549 updates, effective batch 16 = **8,784 maps seen** |
| Data seen | 39% of one pass over 22,408 training maps (MapQA + synthetic, shuffled together) |
| Loss | 0.41 at the start, about 0.10 at the end, still falling |
| Scoring | 400 MapQA test maps (31 min) and 200 synthetic test maps (13 min) |

The test maps were never trained on. MapQA's own train/dev/test split is used unchanged, so numbers can
be compared with other work on MapQA. Synthetic test maps come from separate generator seeds.

A second model, DePlot (282M), was trained for an hour and has now been scored on the same maps. See
section 7.

---

## 2. How to read the numbers

Every score is worked out for one map first, then averaged across maps, so each map counts equally.

**valid JSON.** We take the model's text, find the first `{` and try to read it as data. Unreadable or
cut-off output scores 0 for the whole map. This is the gate: everything downstream needs data, not prose.

**all 50 states present (coverage).** Of the 50 state codes we expect, how many appear. 48 of 50 = 96%.
We also count keys that are not states of this map; that count is 0 everywhere.

**title / legend title / legend type / data type.** Each is right or wrong for the map. Text is compared
ignoring capitalisation and extra spaces.

**legend ranges copied exactly** (range legends only). The model's list of range labels must match the
target list, character for character and in the same order. All or nothing per map.

**states in the right range** (range legends only). On one map: how many of the 50 states got the right
range number, divided by 50. A state with no data counts as correct only if the model also says "no data".

**every field exactly right.** The strictest: the model's whole output must equal the target — title,
legend fields, and all 50 values. One wrong state loses the map.

**average error, as share of the value range** (colour-bar maps only). For one map:
- the map's **range** = its highest true value minus its lowest. Values from 130 to 292 give a range of 162.
- for each state: `error = |model's number − true number| ÷ range`. Off by 7 on that map = 4.3%.
- a missing or non-numeric answer counts as a full range (100%), and no single state counts as more than 100%.
- average over states, then over maps.

We divide by the range because maps use different units — percentages from 0 to 100, counts in millions.
Without dividing, one map of rents would drown out a hundred maps of rates.

**states within 5% of the range.** The share of states whose error is 5% of the range or less. An average
can hide a few badly wrong states; this line shows how many states are comfortably right.

---

## 3. Results on MapQA test maps (400)

| | Range legends (302 maps) | Colour bars (98 maps) |
|---|---|---|
| valid JSON | 100% | 100% |
| all 50 states present | 100% | 100% |
| invented states | 0 | 0 |
| title correct | 100% | 100% |
| legend title correct | 100% | 100% |
| legend type correct | 100% | 100% |
| data type correct | 100% | 100% |
| legend ranges copied exactly | 99.0% | – |
| states in the right range | **97.8%** | – |
| average error vs range | – | **4.1%** |
| states within 5% of range | – | 85.0% |
| every field exactly right | 33.1% | 1.0% |

**What this means.** Every output was usable data with all 50 states and nothing invented, which was the
main failure of the untrained baseline. On range legends the model gets about 49 of 50 states right per
map. "Every field exactly right" at 33% is not a separate weakness: if each state is right 97.8% of the
time, all 50 are right on 0.978⁵⁰ ≈ 33% of maps. For colour bars, 1% exact matches is expected, because
it needs all 50 numbers to match to the last decimal; the error rows are the meaningful ones there.

---

## 4. Results on synthetic test maps (200)

| | Range legends (86) | Colour bars (114) |
|---|---|---|
| valid JSON / all 50 states | 100% / 100% | 100% / 100% |
| title, legend title, legend type, data type | 100% | 100% |
| legend ranges copied exactly | 95.4% | – |
| states in the right range | **97.3%** | – |
| average error vs range | – | **3.5%** |
| states within 5% of range | – | 78.9% |
| every field exactly right | 37.2% | 0% |

**What this means.** These maps use styles MapQA never shows: different plotting library, fonts, palettes,
legend positions, units such as °F and USD per gallon, state abbreviations, and clustered rather than
random values. Scores hold up, so the model learned to read maps rather than to recognise one house style.
This is the strongest evidence in the project that the result generalises.

---

## 5. Colour-bar maps in depth

One MapQA map, map_16733, is excluded below and explained in section 6.

| | MapQA | Synthetic |
|---|---|---|
| median state error | 2.1% of range | 2.6% |
| 75th percentile | 3.8% | 4.6% |
| 90th percentile | 5.6% | 7.0% |
| 99th percentile | 26.2% | 16.1% |
| average error without RI and DE | 2.5% | 3.2% |
| within 5% of range without RI and DE | 87.8% | 80.2% |

**Compared with range legends on equal footing.** If we pretend a colour-bar map had a 4-range legend and
ask which of the four bands each value falls in, the model is right **93.2%** of the time (89.2% on
synthetic), against 97.8% for real range legends. So reading a smooth gradient is genuinely about five
points harder than matching one of four distinct colours. That is the expected direction: the model has
to judge a shade between two printed tick numbers instead of matching a colour it can see in the legend.

**What the stats stage will actually use.** Order and relative size survive well:

| | MapQA | Synthetic |
|---|---|---|
| order of states preserved (1.0 = identical) | 0.96 | 0.97 |
| top-5 states overlap | 92% | 89% |
| bottom-5 states overlap | 88% | 81% |
| single highest state identified | 88% | 81% |
| single lowest state identified | 85% | 68% |

**This is the weakness to state plainly.** A claim like "Texas has the highest value" would be wrong on
roughly one colour-bar map in eight, because the top two states often differ by less than the model can
see. The fix does not have to be a better model: when the top two values are within the model's typical
error, the stats layer should say "among the highest" instead of naming one state.

---

## 6. Where the errors come from

**1. Rhode Island and Delaware — 73% of all range mistakes.** RI was wrong on 160 of 302 MapQA maps and DE
on 78; together 238 of 327 wrong states. Without those two states, accuracy is **99.4%** (98.6% on
synthetic). In a 700×500 map they are a few pixels across, smaller than one 28×28-pixel square of the
model's view, so their colour is mixed with their neighbours before the model ever sees it. This is a
resolution limit, not a learning failure, and more training will not fix it.

**2. One map with a scaled colour bar.** In map_16733 the bar is labelled `3.35M … 3.8M`. The model wrote
`3.63` where the target says `3,630,000`, for every state. That single map moves the MapQA colour-bar
average from 3.1% to 4.1%, and it is the only map of its kind in the test set. It is a units mistake, not
a colour mistake — the model read the picture correctly and wrote the number without its scale.

**3. Neighbouring ranges.** Of the remaining wrong states, most are one range off: 161 of 246 on MapQA, 92
of 111 on synthetic. These are the cases where two legend colours are nearly the same shade.

**4. No-data states.** 61 times the model gave a range to a grey "no data" state, and 20 times the reverse
(MapQA). Grey can sit close to the pale end of some palettes.

**5. Dash characters.** Range labels were wrong on 7 of 388 maps. Four of those are "–" written as "-" or
the other way round, which no model can settle from an image. Two were a misread digit inside a 7-digit
number, and one was both.

**Difficulty scales with the number of ranges**, as expected: 0.8% of states wrong with 2 ranges, 1.5% with
3, 2.6% with 4 (MapQA); 1.1% with 3 up to 4.7% with 6 or 7 (synthetic).

---

## 7. Comparisons, with their caveats

**Against the same model with no training** (the project's earlier baseline, 20 maps per type):

| | Untrained | After training |
|---|---|---|
| range-legend maps: output usable | 35% | 100% |
| range-legend maps: states matched | 0% | 100% present, 97.8% correct |
| colour-bar maps: output usable | 90% | 100% |
| colour-bar maps: states found | 86% | 100% |

The untrained numbers used a different prompt and a different output format, so treat them as direction,
not as a controlled comparison. The one solid claim: the untrained model could not produce usable data for
range-legend maps at all, and the trained one does it every time.

**Against DePlot, the small-model baseline** (282M parameters, 1 hour of training, scored on the same
100 MapQA and 100 synthetic maps Qwen saw):

| On the same maps | DePlot (1 h) | Qwen (8 h) |
|---|---|---|
| valid JSON, all 50 states | 100% | 100% |
| title, legend type, data type | 100% | 100% |
| legend ranges copied exactly | 90.1% | 100% |
| states in the right range | 38.3% | 98.0% |
| colour-bar error vs range | 46.1% | 3.0% |
| **distinct values per map** | **1.0** | 7.1 (target 7.6) |

The last line explains the rest: **DePlot gives every state the same value.** It learned the output
format quickly — valid JSON, the right title, the legend text copied — and then filled all 50 values with
one constant. Its 38% is simply how often the most common range happens to be right; it is not reading
colours at all.

This is what a model does early in training, before it learns to look at the regions: predicting one
constant is the cheapest way to cut the loss. Qwen escaped that within a few updates because its vision
encoder and instruction following came pretrained for this kind of task; DePlot has to learn the whole
behaviour from our data.

**So this is not yet a fair comparison.** DePlot got 1 hour against Qwen's 8, and its own loss was still
falling (3.64 → 0.38). What can be said today: after an hour, DePlot writes the structure but not the
data. Whether it ever reads colours needs an equal-length run — about 9 GPU hours.

**Against the CMA pipeline figures quoted in the baseline notebook** (nMAE 0.032, nRMSE 0.062 on colour-bar
maps). Using that notebook's own formulas, this model scores **0.022 and 0.058** excluding map_16733, or
0.032 and 0.132 including it. Both should be shown. Note those formulas differ from section 2: there,
nMAE divides the error by the mean value and nRMSE divides by the range, which is why the two sets of
numbers are not interchangeable.

---

## 8. Why the design is the way it is

**One JSON object per map, fixed key order.** Downstream code parses it directly; a fixed order makes
training easier and lets us score each field separately.

**State codes kept, rather than 50 values in a fixed order.** Dropping the codes saves about 100 tokens per
map, but one skipped value would shift every later state, the model would have to count its position in a
long list, and a fixed order would not carry to other countries or partial maps.

**Colour-bar maps give numbers, not a 0–1 position on the bar.** Converting a position back to a value
needs the numbers printed on the bar and where they sit, and MapQA does not record those.

**Range legends give the range index, not the range text.** Copying the text for all 50 states costs 992
tokens against 292 for indices, and the text is already in the output once.

**50 states, DC merged into Maryland.** MapQA has no DC, and DC is 2–3 pixels wide at this size.

**Errors capped at one range, and averaged per map.** Without the cap, the single units mistake in
map_16733 doubles the headline figure for 98 maps. Both versions are reported rather than only the
flattering one.

**Synthetic maps exist for three reasons:** style variety MapQA lacks, exact ground truth we control, and a
path to other countries later. Every generated map passes two automatic checks: each state's colour is
read back from the finished image and compared with its label, and the title and legend must fit inside
the image. 71 of 12,000 maps failed and were excluded.

---

## 9. Where we stand

**Solid enough to present**
- Reading a map into structured data works: 100% usable output, all 50 states, nothing invented.
- Range legends: 97.8% of states correct, 99.4% excluding two tiny states.
- Colour bars: typical error about 2% of the value range; order of states preserved at 0.96.
- It holds on map styles it never trained on, which is the claim that matters.
- Every number is reproducible from files in this repo (section 11).

**Not yet**
- **No real published maps have been tested.** Both test sets are generated. This is the biggest gap
  between "works on benchmarks" and "works".
- **The small-model comparison is not like-for-like.** DePlot was given 1 hour against Qwen's 8. After
  that hour it writes correct structure but one constant value for all 50 states, so it cannot yet be
  said whether a 282M model suffices; that needs an equal-length run.
- **Training is unfinished.** 39% of one pass, loss still falling.
- **Naming the single highest or lowest state is unreliable on colour-bar maps** (88% / 85%).
- **Stages 2 and 3 do not exist yet.** The adjacency graph they need is built; the statistics and the
  narrator are not.

**Is it presentable?** Yes, as "stage 1 is built and measured, with known failure modes", which is a
normal and defensible state for a project at this point. It is not presentable as a working
map-to-summary system, because two of the three stages are not written. The honest framing is: the
perception step is solved well enough for the rest of the pipeline to be built on it, and here is exactly
where it still fails.

---

## 10. What would most improve the numbers

1. **Rebalance the training mix.** Only 36% of training maps had colour bars, since MapQA is 76% range
   legends. Sampling to about half targets the weaker case and costs nothing but a rerun.
2. **Train longer.** The loss was still falling at 39% of one pass.
3. **Give small states more pixels** — a higher-resolution image, or a zoomed crop of the Northeast as a
   second image. This is the only fix for RI and DE.
4. **Add scaled tick labels (`3.5M`, `500k`) to the synthetic maps**, so the units mistake in map_16733 is
   trained away rather than left to chance.
5. **Normalise dash characters** to `-` in the targets, since the distinction is not legible.
6. **Test on real published maps** — Our World in Data, Census, Eurostat — which is what turns this from a
   benchmark result into a usable component.

---

## 11. Where every number comes from

| Number | Source |
|---|---|
| Headline tables (sections 3, 4) | `outputs/mapqa-derender-qwen/qwen/metrics_mapqa.json`, `metrics_synth.json` |
| Every prediction and its per-map score | `outputs/mapqa-derender-qwen/qwen/predictions_*.jsonl` |
| Training curve, updates, timing | `outputs/mapqa-derender-qwen/qwen/train_log.jsonl`, `mapqa-derender-qwen.log` |
| Percentiles, rankings, error sources (sections 5, 6) | recomputed from the prediction files against `data/*/test.jsonl` |
| DePlot training log | `outputs/mapqa-derender-deplot-1h/` (scoring crashed there) |
| DePlot scores | `outputs/mapqa-derender-deplot-rescore/` (same model, scored separately) |
| Settings used for the run | `outputs/mapqa-derender-qwen/qwen/args.json` |

To re-score a saved model without retraining:

```bash
python kaggle/launch.py deplot --tag rescore --no-wait \
  -- --skip-train --model /kaggle/input --eval-batch-size 8
```
