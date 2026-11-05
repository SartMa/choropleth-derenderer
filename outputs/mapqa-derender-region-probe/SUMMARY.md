# Can Qwen's vision tokens say which region is where? (2026-10-31)

Script: `train/region_head.py`; labels: `synth/region_labels.py` (Kaggle dataset `sartma/region-probe`, private).
Run: `sartma/mapqa-derender-region-probe`, 57 minutes, including about 29 unused on the MapQA conversion step.

Three small heads learned which region lies under each token, from 2,400 synthetic India maps shrunk, stretched and
moved at random. The rest of Qwen was frozen. The tokens come from the 8-hour adapter, which never saw an India map.
The position head sees only where a token sits on the page; it is the control.

Pass marks, set before the run: at least 90% of big states found on moved maps, clearly above the control, and at
least 80% on real maps. Found: the region's best token is at least half that region. On real maps: the token's
centre is inside CMA's box.

| 27 big states found | Token head | Grid head | Position control |
|---|---|---|---|
| Synthetic, as drawn (300 maps) | 90.8% | 92.3% | 19.9% |
| Synthetic, moved (300 maps) | 77.7% | **79.7%** | 9.4% |
| Real India, CMA's boxes (10 maps, 270 states) | 95.2% | **96.3%** | 24.1% |

| 5 small regions (Delhi, Goa, Sikkim, Tripura, Andaman) | Token head | Grid head | Position control |
|---|---|---|---|
| Best token touches the region, moved maps | 95.4% | **99.2%** | 6.2% |
| Inside CMA's box, real maps | 52.5% | 55.0% | 0% |

Verdict: **missed the 90% mark on moved maps (79.7%)**; beat the control and passed the real-map mark.

## Reading it

- The miss is mostly narrow states. Nagaland, Meghalaya, Mizoram, Manipur and Kerala score 6-33% on moved maps: once
  shrunk they are too thin to fill half a token, so this measure can hardly be met. Their best token always touches
  them. Without those five, moved is 94.1%, but that split was chosen after seeing the results.
- Weakest of the others on moved maps: Haryana 68%, Punjab 80%, Assam 84%.
- "Found" cannot be met by a small region at all (0% everywhere), which is why touching is reported for them.
- Drawn on six real maps not used for scoring (folium, matplotlib, other styles), the codes land on the right states
  by eye, small regions included. Delhi was below the drawing cutoff (2% chance) on some. This was judged by eye.

## Limits

- CMA's boxes are Mask R-CNN output, not hand truth, and a box is looser than a state; neighbouring boxes overlap.
- 10 real maps have boxes. None has exact outlines here.
- This finds regions only. Whether Qwen reads the map better with codes placed this way is untested, and a printed
  code can hide a tiny region's colour.
- India only, and the head was taught for India. A new country needs new teaching, from its boundary file.
