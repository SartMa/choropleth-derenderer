# Code + VLM reader on MapQA (2026-10-30)

Code finds the regions and reads their colours; the VLM reads the text. No GPU, about 3 seconds of CPU per map, plus OCR on continuous maps.
Built by `lookup/read_map.py`, scored by `lookup/score_mapqa.py` and `lookup/check_heldout.py`.

How it works:
- A boundary file is fitted onto the map (scale and position), then each state's colour is read from its flat patch.
- Discrete maps: states of one colour share a class, and the VLM's commonest answer for that colour names the class.
  No legend search.
- Continuous maps: code places each state on the colour bar, and OCR reads the bar's numbers. No VLM.

## Same 400 maps as every VLM run

These maps were used to build the method. The VLM's saved answers stand in for a live call.

| | VLM, tiles (best so far) | Code + VLM, tiles | Code + VLM, no tiles |
|---|---|---|---|
| Discrete: states right | 99.3% | 99.6% | 99.6% |
| Discrete: whole map exact | 89.7% | **96.7%** | 95.4% |
| Rhode Island | 94.4% | 99.0% | 98.3% |
| Delaware | 96.4% | 98.7% | 97.4% |
| Continuous: median error | 2.0% | **0.36%** | 0.36% |
| Continuous: mean error | 3.2% | 0.65% | 0.65% |
| Continuous: within 5% | 90.4% | 98.9% | 98.9% |

Map by map against the same VLM: tiles 23 fixed, 2 broken; no tiles 191 fixed, 0 broken.

## 400 test maps not used to build it

Discrete maps have no saved VLM answers, so colour groups are named from the true labels. That tests grouping only.
It flatters a state alone in its group: 8 of 15,400 readings. Continuous uses no VLM, so it is a full test.

| | Development 400 | Held-out 400 |
|---|---|---|
| Discrete: states right, perfect naming | 99.97% | 99.99% |
| Discrete: every state right, perfect naming | 98.7% | 99.7% |
| Continuous: median error | 0.36% | 0.36% |
| Continuous: mean error | 0.65% | 0.41% |
| Continuous: within 5% | 98.9% | 99.7% |

The tick fix (dropping misread ticks) was made after seeing the held-out maps. Before it, held-out continuous was
1.48% mean and 95.5% within 5%, so those two held-out numbers are not fully untouched.

Lowest share of border points landing on an edge, over all 800 maps: 96.8%. Every colour bar was found.

## Limits

- Every map here is the same Plotly style and projection, the easiest case for fitting a boundary file. Nothing
  here shows it survives real maps; the CMA India overlay is the next test.
- The remaining discrete misses are mostly Delaware or Rhode Island, and the VLM's own legend mistakes, which the
  code inherits.
- Only tall colour bars are handled. All 190 here are tall.
