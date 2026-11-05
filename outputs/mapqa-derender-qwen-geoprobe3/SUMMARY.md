# India: geography or seeing? (20-map probe, 2026-10-07)

Same model as the India results (7-hour run adapter, no tiles). 20 India test maps (10 discrete, 10 continuous),
each redrawn three ways from the same random draws: plain, with region codes written on it (test A), and with one
state outlined in magenta and named in the prompt (test B). 200 images, all answers parsed.

## Test A: write the region codes on the map

| | Plain | Codes drawn |
|---|---|---|
| Big states, discrete: right bucket | 63.9% | **97.3%** |
| Big states, continuous: error (share of value range) | 0.152 | **0.054** |
| Small regions (Delhi, Goa, Sikkim, Tripura, Andaman), discrete | 36% | 40% |
| Small regions, continuous error | 0.253 | 0.225 |

Every weak big state goes to 100% with codes drawn (Chhattisgarh 20% -> 100%, Bihar 33% -> 100%, Jharkhand 50% -> 100%).
**For big states the problem is geography: the model can read the colour, it does not know which region is which.**
With codes drawn it is close to its US accuracy (99%).

The small regions get no codes (the generator never labels them), so test A says nothing about them. They stay
at 36-40% either way; whether that is geography or seeing is still open.

## Test B: outline one state and name it in the prompt — not a valid measurement

Accuracy on the outlined state fell, even for states it always gets right (Rajasthan 100% -> 30%, Gujarat 100% -> 67%).
Other states on the same maps were unaffected (59% vs 60%). So the outline itself breaks the reading of that state:
the model never saw outlines or extra prompt sentences in training. Codes, by contrast, appear on ~40% of training
maps. Test B needs a model trained with outlines before it can answer anything.

## What this means

- The US results reflect a model that knows the US layout, not geography in general.
- Telling the model where regions are (codes, or region positions from the boundary file) recovers almost all of
  the India gap for big states with zero India-specific training. That is the transfer result to build on.
- Next: (1) the same test with codes on small regions too (needs a smaller font or leader lines); (2) train with
  region positions given as input and test on a country it has never seen.

Files: `qwen/predictions_india.jsonl` (per-map answers), probe maps in `data/synth_ind_probe/`, built by
`synth/probe_maps.py`.
