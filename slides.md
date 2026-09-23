---
marp: true
theme: default
paginate: true
---

# Reading choropleth maps into data

**A derenderer for the map-summarisation pipeline**

Stage 1 of three: map image → structured data → statistics → written summary

---

## The problem

- Goal: long-form summaries of choropleth maps that are **numerically correct**
- A single vision-language model writing the summary directly will invent numbers and spatial claims
- Our own baseline showed it plainly: on range-legend maps, an untrained Qwen2.5-VL produced usable output for **35%** of maps and matched **no states at all**
- Existing chart-to-text models don't transfer: no spatial adjacency, no legend reading, no geography

---

## The design: split perception from arithmetic

**Derenderer → code layer → narrator**

| Stage | Job | Why |
|---|---|---|
| Derenderer (a VLM) | image → data table | only step that looks at pixels |
| Code layer (plain Python) | Moran's I, LISA, Getis-Ord Gi*, trends | computed, never generated |
| Narrator (an LLM) | phrase the computed facts | never does arithmetic |

**This work is stage 1.** The code layer is deterministic, so every number downstream is only as good as the derenderer — which is why stage 1 is measured so carefully.

---

## What the derenderer must output

One line of JSON per map:

```json
{"title":"The Number of shots in the USA","legend_title":null,
 "legend_type":"discrete","data_type":"number",
 "ranges":["7548477-7658039","7668882-7761001","7811458-7859166","7920264-8002936"],
 "values":{"AK":0,"AL":0,"AR":1,"...":"...","WY":1}}
```

- All 50 states, by code, always in the same order
- Smooth colour bar → the number; range legend → which range; no data → `null`
- Everything copied from the image: title, legend text, legend type, percentage or number

---

## Three decisions worth defending

- **Keep state codes** rather than 50 values in a fixed order. One skipped value would shift every later state; the model would have to count its position in a long list; a bare list wouldn't carry to other countries. Costs ~100 tokens per map.
- **Colour-bar maps give numbers, not a 0–1 position on the bar.** Converting a position back needs the numbers printed on the bar and where they sit — MapQA doesn't record that.
- **Range legends give the range index, not the range text.** Repeating the text for 50 states costs 992 tokens against 292.

---

## Data

| Source | Train | Dev | Test |
|---|---|---|---|
| MapQA-S (existing benchmark) | 12,470 | 4,155 | 4,157 |
| Synthetic (built here) | 9,938 | 994 | 997 |

- MapQA-S: US choropleths, real KFF styles plus Geopandas/Plotly redraws. Its own split is kept, so numbers compare with other work.
- Synthetic maps: drawn from real geoBoundaries shapes, with 18 topics and real units (USD per gallon, °F, %), varied palettes, legend positions, labels, missing states, JPEG artifacts, and values that are random, clustered or trending.

---

## The synthetic data checks itself

Every generated map must pass two tests before it is used:

- **Pixel check** — each state's colour is read back out of the finished image and compared with its label. A label can never disagree with the picture.
- **Text check** — the title and legend must fit inside the image, so the model is never asked to read text that was cut off.

71 of 12,000 maps failed and were excluded. This is the advantage synthetic data has over scraped data: the ground truth is verified, not assumed.

---

## Training

- **Model:** Qwen2.5-VL-3B, LoRA adapters (37M trained of 3.8B)
- **Hardware:** one Kaggle T4, 8 hours, pushed and fetched from a script
- **Seen:** 8,784 maps — 39% of one pass over the mixed data
- **Loss:** 0.41 → 0.10, still falling when the budget ran out
- **Scored on:** 400 MapQA and 200 synthetic test maps, never trained on

---

## How the results are measured

- **valid JSON** — can the output be read as data at all
- **all 50 states present** — how many expected states appear
- **states in the right range** — for range legends, of 50 states
- **average error** — for colour bars: |model − truth| ÷ the map's own value range, so maps in percent and maps in millions can be averaged together
- **within 5% of range** — how many states are comfortably right, since an average hides outliers

Every score is computed per map, then averaged over maps.

---

## Results: MapQA test maps

| | Range legends (302) | Colour bars (98) |
|---|---|---|
| valid JSON | 100% | 100% |
| all 50 states present | 100% | 100% |
| title, legend, data type | 100% | 100% |
| legend ranges copied exactly | 99.0% | – |
| **states in the right range** | **97.8%** | – |
| **average error vs range** | – | **4.1%** |
| states within 5% of range | – | 85.0% |

Untrained, the same model produced usable output on 35% of range-legend maps and matched no states.

---

## Results: synthetic test maps

| | Range legends (86) | Colour bars (114) |
|---|---|---|
| valid JSON / all 50 states | 100% / 100% | 100% / 100% |
| legend ranges copied exactly | 95.4% | – |
| **states in the right range** | **97.3%** | – |
| **average error vs range** | – | **3.5%** |
| states within 5% of range | – | 78.9% |

Different plotting library, fonts, palettes, units and legend positions — and the scores hold.
**The model learned to read maps, not one house style.**

---

## Colour bars, in detail

| | MapQA | Synthetic |
|---|---|---|
| median state error | 2.1% of range | 2.6% |
| 90th percentile | 5.6% | 7.0% |
| average without RI and DE | 2.5% | 3.2% |

**On equal footing with range legends:** treat a colour-bar map as if it had 4 ranges, and the model puts the value in the right band **93.2%** of the time, against 97.8% for real range legends.

Reading a shade between two printed tick numbers is genuinely harder than matching one of four colours.

---

## What the statistics stage will actually use

| | MapQA | Synthetic |
|---|---|---|
| order of states preserved | 0.96 | 0.97 |
| top-5 states overlap | 92% | 89% |
| **single highest state correct** | **88%** | 81% |
| single lowest state correct | 85% | 68% |

Ordering survives well. Naming *the* highest state does not: the top two states often differ by less than the model can see.

**Fix without touching the model:** when the top two values are within the model's typical error, the narrator says "among the highest" instead of naming one.

---

## Where the errors are

- **Rhode Island and Delaware: 73% of all range mistakes.** RI wrong on 160 of 302 maps, DE on 78. **Without them, accuracy is 99.4%.**
- **One map with a scaled colour bar.** Labelled `3.35M … 3.8M`; the model wrote `3.63` instead of `3,630,000` for every state. A units mistake, not a colour mistake — and it alone moves the colour-bar average from 3.1% to 4.1%.
- **Most remaining mistakes are one range off**, where two legend colours are nearly identical.
- **No-data states:** 61 grey states were given a range.
- **Dash characters:** "–" versus "-", which no model can settle from an image.

---

## Why the small states fail

- Qwen keeps the aspect ratio and cuts the image into 14-pixel patches, merging each 2×2 group: **one token covers 28×28 pixels**
- In a 700×500 map, **Rhode Island is about 9 pixels wide** — smaller than a single patch, so its colour is averaged with the ocean and its neighbours before the model sees it
- Pix2Struct hit the same wall and solved it on the input side: a variable-resolution representation that fills a patch budget while keeping the aspect ratio. DePlot: *"high input resolution is a key ingredient for chart information extraction"*

| Option | Tokens per map | RI becomes |
|---|---|---|
| now | 450 | ~9 px |
| upscale whole map 3× | ~4,050 | ~27 px |
| **full map + zoomed crop of the Northeast** | **~965** | **~36 px** |

---

## Comparisons

- **Same model, untrained:** 35% usable output on range legends, no states matched → now 100% and 97.8% correct
- **CMA pipeline, as quoted in our baseline notebook:** nMAE 0.032, nRMSE 0.062 on colour-bar maps
  - This model, with the notebook's own formulas: **0.022 and 0.058** excluding the scaled-bar map, 0.032 and 0.132 including it
**DePlot (282M) after 1 hour, on the same maps:**

| | DePlot (1 h) | Qwen (8 h) |
|---|---|---|
| valid JSON, all 50 states | 100% | 100% |
| legend ranges copied exactly | 90.1% | 100% |
| states in the right range | 38.3% | 98.0% |
| distinct values per map | **1.0** | 7.1 (target 7.6) |

DePlot learned the format and the text, then gave **every state the same value** — the cheap solution a model finds before it learns to look at the regions. Not yet a fair test: 1 hour against 8.

---

## What is not yet done

- **No real published maps tested.** Both test sets are generated — the main gap between "works on a benchmark" and "works".
- **The small-model comparison isn't like-for-like:** DePlot got 1 hour against Qwen's 8, and still writes one constant value per map.
- **Training is incomplete:** 39% of one pass, loss still falling.
- **Naming the single highest state on colour-bar maps is unreliable** (88%).
- **Stages 2 and 3 don't exist yet.** The adjacency graph they need is built: 50 states, 107 borders.

---

## Next steps

1. **Free data fixes:** normalise dashes, add `3.5M`-style tick labels, vary the no-data grey
2. **One more training run:** continue from the saved adapter, rebalanced towards colour-bar maps (only 36% of training maps had them)
3. **Resolution test:** add a zoomed crop of the Northeast, measure whether RI and DE improve
4. **Score 20–30 real published maps** — Our World in Data, Census, Eurostat
5. **Build stage 2:** the statistics, using the adjacency graph, with a near-tie guard

---

## Takeaway

- Reading a map into structured data **works**: 100% usable output, all 50 states, nothing invented
- Range legends are close to solved: **97.8%**, or **99.4%** excluding two states a few pixels wide
- Colour bars are good and well understood: typical error **2% of the value range**, ordering preserved at 0.96
- It holds up on map styles it never trained on
- Remaining errors are **few, named, and each has a known fix**

The perception step is solid enough to build the rest of the pipeline on.
