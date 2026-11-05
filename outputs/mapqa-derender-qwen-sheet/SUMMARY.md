# Region sheet: can a model taught only on US sheets read India? (2026-11-01)

Code: `train/sheet.py`, `--sheet` in `train/train_qwen.py`. Run: `sartma/mapqa-derender-qwen-sheet`.

Each map goes in with a second image: one equal cell per region, its code above its own pixels, beside the map's
legend. Training continued the 8-hour adapter for 3.96 h (157 updates, about 2,500 maps) on synthetic US maps only, with
random region subsets under random two-letter codes. No India map was used in training.

Regions came from the true drawing, so this tests reading with perfect regions, not finding them.

Pass mark, set before the run and judged on India maps without printed codes: at least 95% of big states, with small
regions matching.

## 300 India test maps

| | No printed codes | Printed codes |
|---|---|---|
| Discrete maps | 89 | 70 |
| Big states right (27) | **100%** | 98.8% |
| Small regions right (5) | **94.8%** | 89.7% |
| Small regions right, without Andaman | 100% | 98.6% |
| Andaman right | 74.2% | 54.3% |
| Whole map exact | 71.9% | 52.9% |
| Whole map exact, ignoring Andaman | 96.6% | 95.7% |
| Continuous maps | 88 | 53 |
| Continuous error, big states | 3.2% | 3.9% |
| Continuous error, small regions | 8.7% | 8.8% |
| Continuous error, small without Andaman | 3.4% | 4.0% |
| Continuous within 5% | 80.5% | 72.2% |

Verdict: **big-state mark passed. Small regions just miss "matching" (94.8% against 100%), entirely because of Andaman.**

## Same maps as the full run

The full run had about 15 hours of training, 1,000 India maps among them. It scored only 100 of these maps, 58 of them
without printed codes.

| Maps without printed codes | Full run, no tiles | Sheet |
|---|---|---|
| Big states right (25 discrete maps) | 63.1% | 100% |
| Small regions right | 48.8% | 94.4% |
| Continuous error, big states (33 maps) | 16.1% | 3.4% |
| Continuous error, small regions | 24.9% | 7.7% |
| Continuous within 5% | 27.8% | 80.3% |

On maps with printed codes big states are level in this subset (95.9% against 95.7%), because one bad map falls among its
18 discrete maps; small regions go from 32.2% to 84.4%.

US, 100 maps, same adapter family: exact 87.2% against 76.6%, states 99.8% against 99.5%, continuous error 2.6% against
2.9%. Ranges right fell from 97.9% to 93.6%.

## Checks

- No leak: the region file gives shapes only. Cell colours and the legend are cut from the map image itself.
- Andaman is not a reading fault. Its islands are so thin that many cells show border or background (grey stripes,
  near-white). On the 48 of 145 measurable maps where the cell's colour was right, Andaman was right 48 times. All 53
  misses among those 145 were on cells more than 25 levels off. Better pixel choice did not fix it (see
  `outputs/real-fit/SUMMARY.md`): the true colour is barely in the image. Its continuous error was not checked this way,
  but it has the same cells.
- Printed codes do not explain the rest: one map (synthind_test_000844, magma, JPEG quality 61) repeated a legend range,
  shifting 25 states. It accounts for most of the big-state gap between the two columns.

## Limits

- True regions and an exact crop of the legend, both from knowing the generator's layout. Real maps give neither. This
  is the main thing left unproven.
- Synthetic India only, 300 test maps. No real map has been through the sheet.
- Continuous within-5% is 80.5%: reading along a colour bar stays weaker than picking a class.
- The continuous reading is still done in generated tokens, not by code.
