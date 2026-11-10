# Derenderer: the learned answer sheet (S2 + zoom)

Status as of 2026-11-05. This is the current derenderer and everything that led to it since the midterm report
(2026-10-25). Earlier work (plain fine-tuning, tiles, the MLP picker) is in `README.md` and `STRATEGIES.md`.

## In short

- **Best model so far: answer sheet v3, "S2 + zoom"** (`train/answer_v3.py --mode zoom`, Kaggle run
  `skini501/mapqa-derender-v3-zoom`). One model for every country we train on: no shapefile fit, OCR or legend crop
  when reading. Shapefiles are only used to make training labels.
- **MapQA (US), 400 test maps:** whole map exactly right **92.1%** (tiles + MLP picker 89.7%, level: 20 maps vs 13,
  p = 0.30), Rhode Island **97.4%**, Delaware 98.7%, continuous error **2.76%** (lowest of any run).
- **India:** synthetic maps without printed codes, big states 99.5%, small regions 93.8%. Real India (CMA's maps,
  31 matched discrete maps): big states 91.5%, small regions 77.4%. The tiles model scores no better than always
  guessing the commonest class on real India, so v3 is the first model that works on both countries.
- **Main gap:** small regions on real India maps, mostly Delhi (45%; the region sheet gets 81%), and a few real
  continuous maps where the error blows up.

## Decision: the design is frozen at v3 (2026-11-05)

We keep v3 (S2 + zoom) as the derenderer's architecture. The remaining gaps (Delhi on real maps, the head being
weaker on real map styles, a few real continuous maps that blow up) are training and reading-time fixes, not a new
design. Work moves to the code layer and the narrator; derenderer work continues alongside.

**Limitation: a fixed list of 87 regions.** S2 and zoom change how sharply the model sees, not how it names regions.
The region head still chooses from background, 50 US states and 36 Indian regions, so it cannot output a region of
a country it was not trained on (on a Chinese map it would mark provinces as background or as the nearest-looking
known state). Adding a country means:
- adding its regions to the head's outputs (China: about 34 more, ~121 in total), keeping all existing weights;
- labels for its maps, which are free (its shapefile plus our synthetic generator; no hand labelling, unlike CMA's
  per-country Mask R-CNN);
- one training round like v3 (~7 h on a T4).

A class-agnostic head (group pixels into regions first, then name them) would remove the per-country round; that is
a research extension for later, not needed before China.

**On S2 and tiles, for the record.** The September S2 run (`train_qwen.py --s2`, Rhode Island 41.7%) had a design
flaw for our problem, not a bug: it averaged the 2x features back onto the 1x token grid (as in the original S2
paper), which mixes a sub-token region with its neighbours again, and it relied on 8.4M new weights that barely
trained. Pooling per region (v3) is what makes S2 work. S2 alone does not beat tiles on US maps (MapQA exact 71.5% vs
89.7%; tiles alone right on 65 maps, S2 alone on 10); it beats them on India, where the tiles model does not work.
Only S2 + zoom matches tiles on the US.

## How the model works

Base: Qwen2.5-VL-3B with LoRA. Its vision tower cuts an image into 14x14-px patches; every 2x2 patches are merged
into one 28x28-px token for the language model.

**1. Answer sheet (v1).** A small conv head (`RegionHead`, ~2M weights) looks at the vision tower's per-patch
features and says, for every 14-px patch, which region it belongs to, out of a fixed list of 87 classes (background,
50 US states, 36 Indian regions). That grid of guesses is the "answer sheet": the map labelled with its regions,
made inside the model. Summing the head's belief by country gives the country (right on 100% of test maps).
Each region's patch features are then averaged, weighted by the head's belief squared, and passed through Qwen's own
2x2 merger, giving **one token per region**. The language model sees the normal map tokens plus
`AP: [token] AR: [token] ...` and writes the JSON. Half the training maps use random made-up codes and a random
subset of regions, so it learns to copy colours rather than recall names.
(The renaming only changes the text in the prompt, `QX: [token]`, and the key in the answer, `"QX": 2`. The region
head is untouched: it still uses the real region, e.g. class IN-MH, to build the token. The language model writes
free text, so any code works; it learns that a code's value must be read from the token right after it, because it
cannot know QX is Maharashtra. When reading real maps, the real codes are used.)

**2. Why small regions failed.** Delhi and Rhode Island cover about a fifth of one 14-px patch. No patch is mostly
Delhi, so Delhi's token is mostly its neighbours' colour. Accuracy followed region area almost exactly (Delhi 63%,
Goa 88%, Sikkim 97%, Tripura 99%). This is resolution, not too little training.

**3. Zoom (v2).** For each region the head finds less than 3 patches of, crop around where the head put it, enlarge
3x, run the tower and head on the crop, and pool that region's token from the zoomed view. Both head and reader are
trained on zoomed views (zooming without training failed). v2's logs showed three problems: the head often could not
find the tiny region on the 1x grid at all (Delhi skipped on 58 of 68 real maps), 3x was not enough for the tiniest
ones, and medium states were zoomed by mistake and got worse.

**4. S2 + zoom (v3, current).**
- *S2 (scaling on scales):* the map is enlarged 2x and cut into 2x2 tiles of the original size; each tile goes
  through the vision tower and the four patch grids are stitched into one grid with twice the detail. Delhi grows
  from 0.2 to about 0.8 of a patch, which is enough to **find** it. The head reads this 2x grid, and every region's
  token is pooled from it. Unlike the S2 tried in September (`train_qwen.py --s2`), nothing is averaged back onto
  the 1x grid (that blended Delhi into its neighbours again) and no new layers sit before the language model.
- *Zoom sized per region:* each region the head finds less than 3 (1x) patches of is re-read in a view zoomed so it
  covers about 8 patches (2x for Goa-sized regions, up to 8x for Delhi). The zoomed token replaces the S2 token only
  if the head is more confident about the region in the zoomed view.
- Training: from v2's weights, head alone 30 min (S2 grids and sized zoom views), then LoRA + head 3.75 h on
  2,000 MapQA + 2,000 synthetic US + 2,000 synthetic India maps. China not used yet.

S2 fixed finding: the head now finds every small region on every map (real India Delhi: 157 of 157 maps on the 2x
grid against 101 on the 1x grid). The sized zoom is what reads them: S2 alone reaches Rhode Island 82.5%, S2 + zoom
97.4%.

## Results

Same test maps for every column. Real India = CMA's maps with true values from their source tables (see below).

| | v1 | v2 (3x zoom) | v3 S2 only | **v3 S2 + zoom** | Tiles + MLP | Region sheet |
|---|---|---|---|---|---|---|
| MapQA whole map exact (302 discrete) | 32.5% | 70.2% | 71.5% | **92.1%** | 89.7% | 74.8% |
| MapQA Rhode Island | 50.7% | 73.8% | 82.5% | **97.4%** | 94.4% | |
| MapQA Delaware | 73.5% | 94.7% | 90.1% | **98.7%** | 96.4% | |
| MapQA continuous error | 3.65% | 3.79% | 2.97% | **2.76%** | 3.2% | 6.0% |
| Synthetic India (no printed codes), big | 99.9% | 99.7% | 99.7% | 99.5% | | 100% |
| Synthetic India, small | 89.2% | 93.3% | 93.8% | 93.8% | | 94.8% |
| Real India (31 discrete), big | 93.2% | 89.5% | **93.9%** | 91.5% | 62.4%\* | 91.4% |
| Real India, small | 72.3% | 76.1% | 77.4% | 77.4% | | **85.2%** |
| Real India Delhi | 35% | 52% | 42% | 45% | | **81%** |
| Real India continuous, mean / median error | 12.6% / 9.3% | 15.2% / 6.3% | **10.8% / 6.3%** | 17.2% / 9.3% | 28.7%\* | 13.4% |

\* the earlier full-run model without the answer sheet. 62.4% equals always guessing the commonest class (63.5%).

Per-run details: `outputs/mapqa-derender-answer-v1/SUMMARY.md`, `outputs/mapqa-derender-v2-zoom/SUMMARY.md`,
`outputs/mapqa-derender-v3-zoom/SUMMARY.md` (covers both v3 runs).

## How we got here (since the midterm report)

1. **Code + VLM reader, dropped** (`lookup/read_map.py`, `outputs/lookup-mapqa/SUMMARY.md`). Code fits the
   shapefile, reads each state's colour and matches it to the legend (OCR for colour bars); the VLM only names colour
   groups. 96.7% exact on MapQA, but it is essentially CMA (Butani, Sreevalsan-Nair, Kamat, IEEE GRSL 2024), whose
   own stated limitation is that it "could be replaced by a single model". That single model is our goal.
2. **Does the model know Indian geography?** (`train/region_head.py`, `train/region_read.py`). A small head on
   frozen features found Indian states well although the model had never seen India; writing state codes at the
   positions it found took big-state accuracy from 28.6% to 85.4% (true positions: 94.7%). So the India problem was
   not knowing where states are, not reading colours.
3. **Region sheet, paused** (`train/sheet.py`, `docs/region_sheet_summary.md`). Code cuts each region out with a
   shapefile fit and lays them out as cells on a second image. Trained on US only, it read synthetic India almost
   perfectly and real India well (91.4% big), but it needs the shapefile at reading time (close to CMA) and lost
   to tiles on MapQA (74.8% vs 89.7%, mostly readings one shade off).
4. **Real India ground truth** (`lookup/fit_real.py`, `lookup/link_tables.py`, `lookup/score_real.py`). The
   professor's Excel sheet lists 105 source tables (102 distinct files). Matching by title and legend numbers only,
   68 of CMA's 164 India maps are linked to their tables (31 strong discrete links, 37 by title), which gives true
   values. Shapefile fits of the real maps also give per-pixel labels for 157 of them (`lookup/answer_labels.py`).
5. **Answer sheet v1** (`train/answer_sheet.py`): the region sheet's idea moved inside the model. Best real India
   big states at the time (93.2%), small regions weak.
6. **Zoom without training failed** (`train/answer_zoom.py`). That run was also hit by a loading bug: v1 saved its
   vision LoRA under `visual.visual.*` names, which peft silently left at zero. Fixed in `answer_sheet.load_adapter`
   / `save_adapter`, which now stop the run if any vision LoRA weight fails to load.
7. **v2** (`train/answer_v2.py`): trained 3x zoom. Small regions improved everywhere; MapQA exact 32.5% -> 70.2%.
8. **v3** (`train/answer_v3.py`): S2 for finding + sized zoom for reading, as above.

## Weights and data

Model weights are not in git (`*.pt` and adapters are ignored). They are Kaggle run outputs on the skini501 account:

| Model | Kaggle run | Files |
|---|---|---|
| v3 S2 + zoom (best) | `skini501/mapqa-derender-v3-zoom` | `v3/adapter/`, `v3/region_head.pt` |
| v3 S2 only | `skini501/mapqa-derender-v3-s2` | `v3/adapter/`, `v3/region_head.pt` |
| v2 | `skini501/mapqa-derender-v2-zoom` | `v2/adapter/`, `v2/region_head.pt` |
| v1 | `sartma/mapqa-derender-answer-v1` | `answer/adapter/`, `answer/region_head.pt` |

Kaggle datasets: `sartma/mapqa-s` (MapQA), `sartma/synth-usa-choropleth`, `sartma/synth-india-choropleth`,
`skini501/answer-v2-inputs` (label rasters, region files, real India maps and labels, v1 weights),
`skini501/real-sheets` (the 68 linked real maps). `data/` is git-ignored; `lookup/answer_labels.py` rebuilds the
labels locally.

## Running it

```bash
# train + score v3 on Kaggle, continuing from v2 (needs the account's KAGGLE_API_TOKEN in the environment)
python kaggle/launch.py v3 --owner skini501 --tag zoom --india-dataset sartma/synth-india-choropleth \
    --extra-dataset skini501/answer-v2-inputs --extra-dataset skini501/real-sheets \
    --from-run skini501/mapqa-derender-v2-zoom --no-wait -- --adapter /kaggle/input --prefer v2-zoom --mode zoom
python kaggle/launch.py v3 --tag zoom --owner skini501 --wait-only      # later: download the results

# score locally
.venv/bin/python lookup/score_real.py --predictions outputs/mapqa-derender-v3-zoom/v3/predictions_real.jsonl \
    --out outputs/mapqa-derender-v3-zoom/v3/real_scores.json
.venv/bin/python lookup/score_india.py outputs/mapqa-derender-v3-zoom/v3/predictions_india.jsonl
```

MapQA metrics are written by the run itself (`metrics_mapqa.json`). A run takes about 7.3 h on a T4.

**Full-scale run on the lab machine (Windows, RTX A4000), from scratch:** v1, v2 and v3 were each continued from the
previous model (quick Kaggle experiments), so the Kaggle v3 weights carry that whole history. The final model is
trained once with the v3 recipe from plain Qwen2.5-VL-3B (`answer_v3.py --adapter none`: fresh LoRA, rank 16 on the
same layers as before, and a fresh head), two passes over ~23,700 maps (3,704 MapQA + 10,000 synthetic US + 10,000 synthetic India), and evaluated on its own on the full test
sets; the Kaggle v3 model is scored on the same sets for comparison. See `lab/README.md`. One Docker image, then
`python lab/run.py check | prepare | smoke | full | status | score`. Training saves a resume checkpoint every
30 minutes and keeps a usable copy of the model every 6 hours; re-running `full` continues after any stop.

## New files

```
train/answer_sheet.py    answer sheet v1: region head, region tokens, adapter load/save fix
train/answer_zoom.py     zoom check on v1 without training
train/answer_v2.py       v2: trained 3x zoom
train/answer_v3.py       v3: S2 grid + sized zoom (--mode zoom) or S2 only (--mode s2)
train/region_head.py     first test: can a head on frozen features find Indian states?
train/region_read.py     marked-map test: does writing codes at found positions help reading?
train/sheet.py           region sheet images (paused)
train/read_real.py       reading real maps with region sheets
synth/region_labels.py   per-pixel region rasters for the synthetic generator
synth/probe_maps.py      redraw synthetic maps with codes or an outlined region (tests)
lookup/                  shapefile fitting, real-map linking and scoring, label building, code + VLM reader
docs/gemini_prompt.txt   the project summary we gave Gemini for outside ideas
docs/region_sheet_summary.md   the region sheet write-up
```

## Open problems and next steps

1. **Delhi on real maps.** It is found and zoomed on 65 of 68 maps, but the zoomed token is only 43% Delhi
   (Rhode Island on MapQA: 79%). Next: look at the zoomed real views to see why.
2. **The head is weaker on whole real maps in v3** than in S2 only (78% vs 89% of patches right after training);
   time spent on zoomed views cost full-map accuracy on real styles.
3. **Whole-map class shifts** on a few real maps (every state one class off), and two real continuous maps that
   blow up. Likely cause: tokens from zoomed views sit at a different scale from full-map tokens. A cheap test is to
   read those maps with the zoomed tokens rescaled to the full-map tokens' statistics.
4. **S2 tiles see a quarter of the map each.** Qwen's vision tower has full attention only in blocks 7, 15, 23 and
   31, so each 2x tile is normalised on its own. One 2x pass of the whole map may be better.
5. **Ideas from the Gemini review worth trying:** predict each patch's position within the country, so tiny
   regions are located from the big ones around them; give legend swatches their own tokens; an ordinal loss for
   "one shade off" errors.
6. **New countries** need new head outputs and a short training round. A class-agnostic head (group pixels into
   regions, then name them) would remove that; needed before China.
7. **For a paper:** accuracy by region size, frontier models (GPT-4o, Gemini, Qwen-72B) as baselines on the same
   maps, CMA's own numbers on its India maps, and the made-up-codes ablation.
