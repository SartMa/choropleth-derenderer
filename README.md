# Choropleth derenderer

Stage 1 of a three-stage pipeline for writing accurate summaries of choropleth maps:

**Derenderer (this repo) → stats code layer → narrator.**

The derenderer reads a map image and writes the data behind it as JSON. Everything downstream
(statistics, then prose) works from that JSON, so no stage after this one has to read pixels or do
arithmetic in generated text.

> **Current state (2026-11-05):** the derenderer is now the learned answer sheet with S2 + zoom
> (`train/answer_v3.py`): one model for US and India, 92.1% of MapQA maps exactly right, real India big states
> 91.5%. Read [docs/ANSWER_SHEET.md](docs/ANSWER_SHEET.md) for the method, results and next steps. The sections
> below describe the earlier plain fine-tuning baseline.

## What the model outputs

One line of compact JSON per map. Keys always appear in this order:

| Key | Value | Rule |
|---|---|---|
| `title` | text | copied exactly as printed |
| `legend_title` | text or `null` | the text above the legend, exactly as printed |
| `legend_type` | `"continuous"` or `"discrete"` | a smooth color bar, or boxes with ranges |
| `data_type` | `"percentage"` or `"number"` | from the title, or the "%" in range labels |
| `ranges` | list of text | discrete maps only: legend labels copied exactly, lowest first |
| `values` | object | all 50 states by 2-letter code, sorted A–Z |

In `values`: a number for continuous maps, the index into `ranges` for discrete maps, `null` where the
map shows no data.

```json
{"title":"The Percentage of ceremonies in the USA","legend_title":"Value","legend_type":"continuous",
 "data_type":"percentage","values":{"AK":37.9,"AL":39.8,"AR":49.8,"...":"...","WY":47.4}}
```
```json
{"title":"The Number of shots in the USA","legend_title":null,"legend_type":"discrete","data_type":"number",
 "ranges":["7548477-7658039","7668882-7761001","7811458-7859166","7920264-8002936"],
 "values":{"AK":0,"AL":0,"...":"...","WY":1}}
```

Two design choices worth stating:

- **State codes are kept**, rather than a bare list of 50 values in a fixed order. One skipped value
  would otherwise shift every later state, the model would have to count its position in a long list,
  and a fixed order would not carry over to other countries or partial maps. The codes cost about 100
  tokens per map.
- **Continuous maps give numbers**, not a 0–1 position on the color bar. Converting a position back to a
  value needs the numbers printed on the bar and where they sit, and MapQA does not record those.
- **Range legends give a range index, not the range text.** Copying the text for every state costs 992
  tokens per map against 292 for indices.

## Data

| Source | Train | Dev | Test | Notes |
|---|---|---|---|---|
| MapQA-S (converted) | 12,470 | 4,155 | 4,157 | MapQA's own split, by map |
| Synthetic (generated here) | 9,938 | 994 | 997 | 71 more maps were generated and rejected by automatic checks |

**MapQA-S** is an existing dataset of US choropleths: real KFF map styles plus maps redrawn with
Geopandas and Plotly. `mapqa/convert_mapqa.py` turns its per-map metadata into the JSON above and keeps
MapQA's own train/dev/test split. Maps that no question file uses (7 of them) are skipped.

**Synthetic maps** come from `synth/generate_maps.py`, drawn from real geoBoundaries state shapes. They
add variety MapQA does not have: 18 topics with real units (USD per gallon, °F, years, %), palettes,
legends on the right or the bottom, several range-label styles, state abbreviations on or off, missing
states, JPEG artifacts, and values that are random, clustered among neighbours, or rising in one
direction. DC is merged into Maryland so every map has the same 50 states as MapQA.

Every generated map passes two automatic checks before it is used for training:

- **Pixel check:** each state's color is read back from the finished image and compared with what the
  target says, so a label can never disagree with the picture.
- **Text check:** the title and legend must fit inside the image, so the model is never asked to read
  text that was cut off.

`data/boundaries/usa_states_adjacency.json` holds which states share a border (50 states, 107 borders),
built from the same shapes. The stats stage needs it.

## Results

Qwen2.5-VL-3B with LoRA, 8 hours on one Kaggle T4, trained on MapQA and synthetic maps together. That
covered 8,784 maps, about 39% of one pass through the data.

| | MapQA test | Synthetic test |
|---|---|---|
| **Discrete maps** | 302 | 86 |
| valid JSON / all 50 states present | 100% / 100% | 100% / 100% |
| title, legend title, legend type, data type | 100% | 100% |
| legend ranges copied exactly | 99.0% | 95.4% |
| states in the right range | **97.8%** | **97.3%** |
| every field of the map exactly right | 33.1% | 37.2% |
| **Continuous maps** | 98 | 114 |
| valid JSON / all 50 states present | 100% / 100% | 100% / 100% |
| average error, as share of the value range | **4.1%** | **3.5%** |
| states within 5% of the range | 85.0% | 78.9% |

Errors are measured against the map's own value range, and a single state's error is capped at 100% of
that range. "Every field exactly right" follows from the per-state rate: 0.978⁵⁰ ≈ 33%.

### Where the errors are

- **Rhode Island and Delaware account for 73% of wrong states.** RI was wrong on 160 of 302 MapQA
  discrete maps and DE on 78. Without those two states, accuracy is **99.4%** on MapQA and 98.6% on
  synthetic maps. In a 700×500 map they are a few pixels wide, smaller than one 28×28-pixel square of
  the model's view.
- **One map caused nearly all large value errors.** In map_16733 the color bar is labelled `3.35M` …
  `3.8M`, and the model wrote `3.63` instead of `3,630,000` for every state. Excluding that map, the
  average continuous error on MapQA is 3.1% of the range instead of 4.1%.
- **Most remaining range mistakes are one range off** (161 of 246 on MapQA), where the two colors are
  close.
- **No-data states:** 61 times the model gave a range to a gray no-data state, 20 times the reverse.
- **Dash characters:** most range-label mismatches are "–" written as "-" or the other way round, which
  an image cannot settle.

### For comparison

- **Same model, no training** (from the project's earlier baseline): on range-legend maps only 35% of
  outputs parsed and none of the states matched; on continuous maps 90% parsed with 86% of states found.
  After training, both are 100%.
- **CMA pipeline figures quoted in the baseline notebook** are nMAE 0.032 and nRMSE 0.062 on continuous
  maps. Using that notebook's own formulas, this model scores nMAE 0.022 and nRMSE 0.058 excluding
  map_16733, or 0.032 and 0.132 including it. The formulas differ from the ones in the table above:
  there, nMAE divides by the mean and nRMSE by the range.

### DePlot baseline

DePlot (the MatCha-family chart-to-table model, 282M parameters) is the small-model baseline. One hour of
training, scored on the same 100 MapQA and 100 synthetic maps as Qwen:

| On the same maps | DePlot (1 h) | Qwen (8 h) |
|---|---|---|
| valid JSON, all 50 states | 100% | 100% |
| legend ranges copied exactly | 90.1% | 100% |
| states in the right range | 38.3% | 98.0% |
| colour-bar error vs range | 46.1% | 3.0% |
| distinct values per map | 1.0 | 7.1 (target 7.6) |

**DePlot gives every state the same value.** It picked up the format quickly — valid JSON, right title,
legend text copied — then filled all 50 values with one constant, which is the cheapest way to cut the
loss before a model learns to look at the regions. Its 38% is just how often the most common range is
right. Qwen escaped that within a few updates, helped by its pretrained vision encoder.

This is not a like-for-like comparison: DePlot had 1 hour against Qwen's 8, and its loss was still falling
(3.64 → 0.38). Settling whether a 282M model suffices needs an equal-length run, about 9 GPU hours.
Scoring also needs a batch of 8 or fewer: at full resolution one attention layer alone asks for 3 GB.

Two facts about DePlot that shape its use here: it has no hard output-length limit (its decoder uses
relative positions), but it was trained on outputs of up to 512 tokens, while our targets run to about
530. And its prompt is not text the model reads as tokens; it is drawn onto the image, so DePlot cannot
be told the output format in words the way Qwen can. It has to learn the format from training alone.

## Running it

```bash
pip install -r requirements.txt

# MapQA-S -> training files
python mapqa/convert_mapqa.py --mapqa ~/Downloads/MapQA_S.zip --out data/mapqa_derender

# synthetic maps (about 45 minutes for 12,000 maps on 14 cores)
python synth/generate_maps.py --split train --n 10000
python synth/generate_maps.py --split dev   --n 1000
python synth/generate_maps.py --split test  --n 1000

# upload them once, so Kaggle runs can use them
kaggle datasets create -p data/synth_usa -r zip
```

Training runs on Kaggle. The launcher packs the converter, the shared code and one training script into
a notebook, attaches both datasets, pushes it, and can download the results:

```bash
python kaggle/launch.py qwen --no-wait          # full run, about 9 hours
python kaggle/launch.py qwen --wait-only        # fetch results into outputs/
python kaggle/launch.py deplot --smoke          # 15-minute check that everything works
python kaggle/launch.py deplot --tag 1h --no-wait -- --hours 1 --eval-batch-size 8
```

Anything after `--` goes to the training script. Useful flags: `--hours` (training budget, and the
learning-rate schedule shrinks to fit it), `--train-maps` / `--synth-train-maps`, `--eval-maps` /
`--synth-eval-maps`, `--skip-train` with `--model` or `--adapter` to score a saved model.

Each run writes `train_log.jsonl`, plus `metrics_<set>.json` and `predictions_<set>.jsonl` for the MapQA
and synthetic test sets.

## Files

```
mapqa/convert_mapqa.py    MapQA-S -> training files
synth/generate_maps.py    synthetic maps + targets, with the pixel and text checks
train/common.py           data loading, training loop, scoring
train/train_deplot.py     fine-tune DePlot
train/train_qwen.py       LoRA fine-tune Qwen2.5-VL
kaggle/launch.py          push a run to Kaggle, wait, download results
train/answer_*.py         the answer sheet (v1, zoom check, v2, v3) - see docs/ANSWER_SHEET.md
lookup/                   shapefile fits, real India maps, scoring
```

## Known limits and next steps

1. **Small states.** RI and DE are the single largest source of error. Options: feed the map at higher
   resolution, or pass a zoomed crop of the Northeast as a second image.
2. **Scaled color bars.** Labels like `3.5M` need the model to write the full number. Synthetic maps
   should include them so this is learned rather than left to chance.
3. **Dash characters** should be normalised to `-` in the targets, since the distinction is not legible.
4. **Training length.** The run covered 39% of one pass and the loss was still falling, so more training
   should still help, though it will not fix the small-state problem.
5. **DePlot needs an equal-length run** (about 9 GPU hours) before the small-model question can be
   answered; after 1 hour it writes structure but a constant value per map.
6. **Only US states so far.** The generator can draw other countries from geoBoundaries; MapQA cannot.
7. **Stages 2 and 3** (statistics, then the narrator) are not built yet. The adjacency graph they need
   is already produced here.
