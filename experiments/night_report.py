"""Summarize whatever the overnight runs produced. No arguments."""
import glob
from pathlib import Path

import pandas as pd

print(f"{'run':28s} {'n':>5s} {'fold-0 acc':>11s}")
print(f"{'r2p1d (baseline, 128px)':28s} {701:>5d} {0.6020:>11.4f}")
for f in sorted(glob.glob("runs/v*/oof_f0.csv")):
    d = pd.read_csv(f)
    print(f"{Path(f).parent.name:28s} {len(d):>5d} {(d.y_true == d.y_pred).mean():>11.4f}")
print("\nv1_vmae = 4->3 adapter, IR never learned (norm 0.024). Not a fair comparison.")
print("v2_vmae = 4-channel stem, margin 1.35. This is the fair one.")
print("v3_vmae_m155 = 4-channel stem, margin 1.55.")
