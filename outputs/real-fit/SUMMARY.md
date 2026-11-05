# Cutting real India maps with the boundary file (2026-11-01)

Scripts: `lookup/fit_real.py` (fit and check), `lookup/find_legend.py` (legend, draft). CPU only. 164 real maps from the
CMA folder, in matplotlib, folium, plotly and other styles. True values come from linked source tables, below.

## Fit

| | |
|---|---|
| Maps | 164, about 1.5 s of CPU each per projection |
| State centres inside CMA's boxes (10 maps, 327 regions) | 96.9% |
| Median overlap with CMA's boxes | 0.84 |
| Best projection | plain lat/lon 62, aspect-corrected 42, Mercator 60 |

CMA's boxes are Mask R-CNN output, not hand truth.

## The check

Border points landing on edges does not work alone: the US file scored up to 100% on India maps, and the right
country won only half the time. Adding region uniformity (the user's idea) and contrast with the surroundings fixed it.

Rule: uniform x contrast >= 0.8 and edges >= 0.9.

- Good India fits kept: 157 of 164. The 7 rejected are mostly pale maps close to the page colour; they take the plain route.
- Wrong-country fits let through: 1 of 80.
- Right country wins on India maps: 40 of 40.
- Thresholds were chosen on this same data, so these are optimistic.
- Country finding on US maps is untested: CMA's US maps fit neither file (edges about 55%). Their projection is unknown.
  The rule does reject most of those, which is the safe outcome.

## Legend

Solid ink outside the country, grown to include its text. Against CMA's legend boxes (9 usable): median overlap 0.54,
5 at 0.5 or better. By eye on 30 maps, most are right, but a few box the title instead. Not ready.

## Andaman

The earlier "cutting fault" was too simple. Four ways of choosing pixels gave at best 40 of 145 right. On real maps the
islands are mostly solid border ink: the deepest pixels are near-black on 136 of 164 maps. Only the largest southern
islands show true colour, and only on some maps. See `andaman_zoomed.png`.

## True values for real maps

`lookup/link_tables.py` links maps to the 102 source tables listed in `~/Downloads/map source tables.xlsx` (downloaded to
`data/cma_tables/`, ignored by git), using OCR of the title and legend numbers only. State colours are not used.

- Strong (title picks the table, both legend ends equal one column's lowest and highest): 31 maps, 23 distinct columns.
- Title only (title picks the table, printed numbers fit inside the column): 37 maps. Weaker; mostly continuous bars.
- Unlinked: 96, kept out of scoring. 27 are election maps (electors, seats, voters), which have no table in the list.
- Three pairs of rows in the list share one file, so one table of each pair is missing (rows 24/34, 54/61, 98/104).
- Links have not been checked by hand.

## Sheet read on real maps

Run `sartma/mapqa-derender-real-sheets`, scored by `lookup/score_real.py` (`real_scores.json`). Legends now come from OCR
number clusters. Cells are cut from the originals without border ink.

- Strong links, discrete, 31 maps: big 91.4%, small 85.2%, tiny 59.7%, Andaman 58.1%; always-commonest baseline 63.5%.
- Title links, discrete, 6 maps: big 74.1%, small 73.3%, tiny 33.3%; baseline 65.6%.
- Title links, continuous, 31 maps: error big 13.4%, small 9.9%, tiny 22.2%.

Old full-run model on the same maps (run `skini501/mapqa-derender-real-plain`, `real_scores_plain.json`): strong
discrete big 62.4% (sheet better on 27 maps, worse on 3), small 67.1%, Andaman 90.3%; continuous error big 28.7%.
