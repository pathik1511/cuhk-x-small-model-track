"""Re-derive YOLO crop boxes at a different margin. Seconds, not 15 minutes.

The detector output does not depend on the margin. yolo_box.py stores the pre-margin
centre and tight side, so any margin is a arithmetic on the existing CSV.

    python src/remargin.py yolo_boxes.csv 1.55 yolo_boxes_155.csv
"""
import sys
import numpy as np
import pandas as pd

src, margin, dst = sys.argv[1], float(sys.argv[2]), sys.argv[3]
d = pd.read_csv(src)
if "tight" not in d.columns:
    sys.exit("REFUSED: this CSV predates the tight-box change. Re-run yolo_box.py once.")

ok = d.fallback == 0
side = np.minimum(np.maximum(d.tight * margin, 0.34 * np.maximum(d.W, d.H)),
                  np.minimum(d.W, d.H))
bx = np.clip(d.cx - side / 2, 0, d.W - side)
by = np.clip(d.cy - side / 2, 0, d.H - side)
d.loc[ok, "x0"] = bx[ok].astype(int)
d.loc[ok, "y0"] = by[ok].astype(int)
d.loc[ok, "x1"] = (bx + side)[ok].astype(int)
d.loc[ok, "y1"] = (by + side)[ok].astype(int)
d.to_csv(dst, index=False)
print(f"margin {margin} -> {dst}")
print(f"  detected {int(ok.sum())}/{len(d)}")
print(f"  box side mean {float((d.x1-d.x0)[ok].mean()):.0f}  "
      f"min {int((d.x1-d.x0)[ok].min())}  max {int((d.x1-d.x0)[ok].max())}")
print(f"  at the 480 clamp: {100*float(((d.x1-d.x0)[ok] >= 478).mean()):.1f}% of clips")
