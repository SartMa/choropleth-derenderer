# Derenderer update: region sheets and real maps

Status as of 2026-11-02. Everything here is uncommitted work in the repo. Result files are listed at the end.

## In short

- **The India problem has a likely fix.** We cut each region out of the map and give the model a second image, a "region
  sheet". A model trained only on US maps then read synthetic India almost perfectly.
- **It also works on real maps, less strongly.** With automatic cutting, big states are 91.4% right on 31 real discrete
  maps, against 63.5% for always guessing the commonest class. Continuous maps are weaker, at 13.4% error.
- **The sheet clearly beats the old model on real maps.** On the same 31 strong maps, the old full-run model got 62.4% of
  big states, no better than always guessing the commonest class (63.5%). The sheet was better on 27 maps, worse on 3.
- **The pipeline stays 3 stages**, but the derenderer gains a code step inside it: code cuts regions, the VLM reads them.

## The problem we started from

Two gaps remained after the tiles work:

- **Small regions.** A vision token is 28x28 px, and Rhode Island is about 9x13 px. Tiles plus the MLP picker fixed this on
  US maps (Rhode Island 94.4%, whole map exact 89.7%).
- **India.** True accuracy is 71.2%, not the 79.2% in the midterm report. Printing state codes on maps took big states
  from 64% to 97%, so the model doesn't know where states are. It can read colours.

## What we tried

| Idea | Result | Decision |
|---|---|---|
| Code reads regions and colours, VLM reads text (`lookup/`) | US MapQA whole map exact 96.7%, continuous median error 0.36%; one Plotly style only | Too close to CMA to be the strategy; keep as a baseline and check |
| Region head: teach Qwen where regions are | Big states 79.7% on moved maps (missed the 90% mark), 96.3% on 10 real maps against CMA's boxes | Possible add-on; needs teaching per country |
| Placing codes on maps using the head | 85.4% against 94.7% with true positions, on 7 shared regions | Missed the mark |
| Region sheet | See below | Current direction |

## The region sheet

Each region gets one equal cell: its code above its own pixels, with the map's legend beside the cells. The sheet goes in
as a second image. Reading becomes "match each cell to the legend", which needs no geography.

Training used synthetic US maps only, under random two-letter codes, so the model has to copy each code from its cell. It
ran 4 hours on top of the 8-hour adapter. No India map was used in training.

Pass mark, set before the run and judged on India maps without printed codes: 95% of big states, with small regions matching.

| Maps without printed codes | Old full run | Sheet |
|---|---|---|
| Big states right | 63.1% | **100%** |
| Small regions right | 48.8% | **94.4%** |
| Continuous error, big states | 16.1% | **3.4%** |
| Continuous maps within 5% | 27.8% | **80.3%** |

These are the 58 maps both runs scored. The old run had about 15 hours of training, including 1,000 India maps. Across all
177 no-code test maps, the sheet scores 100% on big states and 94.8% on small regions.

**Verdict:** passed on big states. Small regions just miss, entirely because of Andaman (74%); the other four are at 100%.

We checked for a leak. The region file gives shapes only, and all colours and the legend come from the map image.

**Andaman:** the model is right 48 of 48 times when the cell colour is right, but four better ways of choosing pixels did
not fix it. The islands are mostly solid border ink, near-black on 136 of 164 real maps. It should probably be reported as
"not readable" instead of guessed.

**Limit:** this test used true regions and a legend crop taken from the generator's known layout. Real maps give neither.

## Cutting real maps

We used the 164 real India maps from the CMA folder. Everything below is on CPU.

| Step | How | Result |
|---|---|---|
| Fit the India shapefile | Tries 3 projections, keeps the best | Works on all 164; 96.9% of state centres inside CMA's boxes |
| Trust check | Region uniformity x contrast with surroundings, plus edges | Keeps 157 of 164 good fits; lets through 1 of 80 wrong-country fits |
| Find the legend | Biggest cluster of OCR numbers outside the country | 68 of 68 found; judged by eye, some bar ends clipped |
| Cut cells | From full-size originals, dropping border ink | Delhi and Goa recovered; the tiniest territories come out empty |

- Edges alone do not work as a check: the US file scored up to 100% on India maps. Adding uniformity fixed it.
- The trust thresholds were chosen on this same data, so those numbers are a little optimistic.
- Finding the country this way is proven only on India maps. CMA's US maps fit neither file, so that side tells us nothing.
- The fit is deterministic code with no learning. It needs a boundary file for the country; geoBoundaries covers most.
- The fit overlaps CMA's region-finding step, but the VLM still does all the reading and there is no per-country Mask R-CNN.

**Agentic version:** skipped for now. We keep one simple part: the VLM names the country, code fits and checks, and a poor
fit falls back to plain reading with the answer flagged.

## True values for real maps

The source tables list has 105 links to 102 files. They were linked to maps using OCR of the title and legend numbers only,
never state colours, so the test stays fair.

| Link quality | Maps |
|---|---|
| Strong: title picks the table, legend ends equal the column's lowest and highest | 31 (23 distinct columns) |
| Title only: weaker, mostly continuous maps | 37 |
| Unlinked, left out of scoring | 96 |

- 27 of the unlinked maps are election maps (electors, seats, voters), which have no table.
- Three pairs of rows in the list share one file, so one table of each pair is missing: rows 24/34, 54/61, 98/104, counted
  from 0 after the header. No other tables are available.
- The links have not been checked by hand.

## First result on real maps

The sheet adapter read all 68 linked maps with automatic cutting (Kaggle run `sartma/mapqa-derender-real-sheets`). A
discrete answer is right if the true value lies in the chosen range.

| Discrete maps | Maps | Big states | Small regions | Tiny territories | Always commonest class |
|---|---|---|---|---|---|
| Strong links | 31 | **91.4%** | 85.2% | 59.7% | 63.5% |
| Title-only links | 6 | 74.1% | 73.3% | 33.3% | 65.6% |

| Continuous maps, all title-only links | Maps | Big states | Small regions | Tiny territories |
|---|---|---|---|---|
| Error as a share of the value range | 31 | 13.4% | 9.9% | 22.2% |

- Real maps score below synthetic (91% against 100%), but well above the baseline.
- The 31 strong maps share 23 columns, so they are fewer independent datasets than they look.
- Title-only links may themselves be wrong, so a low score there can be a linking fault, not a reading fault.
- One map scored far below the rest (`inmates_2020_transgender_discrete_folium`, 19%). The same data in another style
  scored 100%. The cause has not been checked.
- Andaman is 58.1% on strong maps.

## Old model on the same real maps

The old full-run adapter (15 hours, 1,000 India maps, run `skini501/mapqa-derender-real-plain`) read the same 68 maps
with the plain India prompt and no sheet. Tiny territories are left out, because the old prompt does not ask for them.

| Same maps | Sheet | Old model | Sheet better / worse |
|---|---|---|---|
| Strong discrete (31): big states | **91.4%** | 62.4% | 27 / 3 maps |
| Strong discrete (31): small regions | **85.2%** | 67.1% | |
| Title discrete (6): big states | 74.1% | 63.6% | 5 / 1 |
| Title continuous (31): error, big states | **13.4%** | 28.7% | 27 / 4 |
| Title continuous (31): error, small regions | **9.9%** | 31.8% | |

- The old model is no better than always guessing the commonest class (63.5%) on real India maps.
- Andaman is the exception: old 90.3%, sheet 58.1%. That is likely the old model guessing the commonest class, not reading.
- The links were chosen without either model's answers, so the comparison is fair, but the strong maps share 23 columns.

## Next

1. Run the sheet on the 400 MapQA test maps, to see whether it can replace tiles on US maps too (89.7% whole-map exact now).
2. Look at the worst maps, starting with the 19% one.
3. Plan the lab GPU run around the sheet: train on more styles and countries, not only synthetic US.

## Open items

- Sheet not yet run on MapQA, the set behind our main US numbers.
- Continuous within-5% is 80%, weaker than discrete, and that reading still happens in generated tokens.
- No unseen-country test on real maps yet.
- Midterm report corrections: India 79.2% should be 71.2%, and "21% of one pass" is about 59% with the 8-hour adapter.
- The code layer described in the report is not in this repo.

## Files

- `outputs/mapqa-derender-qwen-sheet/SUMMARY.md`: sheet test
- `outputs/real-fit/SUMMARY.md`: real-map cutting and linking
- `outputs/lookup-mapqa/SUMMARY.md`, `outputs/mapqa-derender-region-probe/SUMMARY.md`, `outputs/mapqa-derender-read-probe/SUMMARY.md`: earlier tests
- `train/sheet.py`, `train/read_real.py`, `lookup/fit_real.py`, `lookup/find_legend.py`, `lookup/link_tables.py`, `lookup/build_real_sheets.py`, `lookup/score_real.py`: code
