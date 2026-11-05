# Does Qwen read India better when the region head marks the map? (2026-10-31)

Script: `train/region_read.py`. Run: `sartma/mapqa-derender-read-probe`. 100 India test maps with no codes drawn (46
discrete, 54 continuous), each read five ways by the 8-hour adapter, which never saw an India map.

**The run is mostly unscorable.** That adapter never learned Indian codes, so it wrote US codes on 497 of 500 answers
and most regions got no answer. The metrics files mean little and must not be quoted. This was a mistake in choosing
the adapter. The 8 codes both countries share (AR, CT, GA, LA, MN, OR, TN, UT) did get answers, so those are scored.

| Discrete, 46 maps | Plain | True positions | Head's positions |
|---|---|---|---|
| 7 shared big regions | 28.6% | **94.7%** | **85.4%** |
| Arunachal Pradesh (AR) | 20% | 100% | 85% |
| Chhattisgarh (CT) | 35% | 98% | 96% |
| Ladakh (LA) | 35% | 98% | 98% |
| Manipur (MN) | 30% | 74% | 50% |
| Odisha (OR) | 17% | 98% | 91% |
| Tamil Nadu (TN) | 33% | 98% | 98% |
| Uttarakhand (UT) | 30% | 98% | 80% |

Maps better than plain: true positions 45, worse 0; head's positions 45, worse 0. Head against true: 1 better, 26 worse.

Pass mark, set before the run: head within 5 points of true on big regions, and clearly above plain.
**Missed: 9.3 points behind true**, though far above plain. Scored on 7 of 27 big regions, so this is partial.

Continuous maps, same 7 regions, error as a share of the value range: plain 32.7%, true positions 8.6%, head's
positions 10.5%.

Goa (GA), the only small region here: plain 33%, code on big regions only 26%, code on it too 43% (true) and 39%
(head). One region and 46 maps: too little to settle whether small regions fail from location or pixels.

What it shows: a model with no India training reads big Indian regions at about 95% once they are
named on the map. The next run needs an adapter that writes Indian codes, such as `skini501/mapqa-derender-qwen-full`.
