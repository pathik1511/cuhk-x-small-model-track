"""Paired comparison against the r19_sadp baseline on identical OOF clips.

The only honest read in this campaign. Reports b and c, not just an accuracy delta:
  b = baseline wrong, candidate correct   (corrections)
  c = baseline correct, candidate wrong   (regressions)
Net is (b-c)/N. A small net built from a large b and c is noise, not a mechanism.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

cand = pd.read_csv(sys.argv[1])
base = pd.read_csv("analysis/oof_ensemble5.csv")[["clip_id", "user", "y_true", "top1_r19_sadp"]]
m = cand.merge(base, on="clip_id", suffixes=("", "_b"))
a = (m.y_pred == m.y_true).values
b_ = (m.top1_r19_sadp == m.y_true).values

b = int((a & ~b_).sum())
c = int((~a & b_).sum())
N = len(m)
print(f"clips compared : {N}")
print(f"r19_sadp       : {b_.sum():>5} = {b_.mean():.4f}")
print(f"candidate      : {a.sum():>5} = {a.mean():.4f}")
print(f"b corrections  : {b}")
print(f"c regressions  : {c}")
print(f"NET            : {b-c:+d} clips = {(b-c)/N:+.4f}")

# subject-clustered bootstrap: 18 subjects is few, so clip-level binomial understates
rng = np.random.default_rng(0)
users = m.user.values
uu = np.unique(users)
d = []
for _ in range(4000):
    s = rng.choice(uu, len(uu), replace=True)
    idx = np.concatenate([np.where(users == u)[0] for u in s])
    d.append(a[idx].mean() - b_[idx].mean())
lo, hi = np.percentile(d, [2.5, 97.5])
print(f"95% CI (subject-clustered bootstrap, {len(uu)} subjects): [{lo:+.4f}, {hi:+.4f}]")
print("\nPROMOTE if NET is positive and the CI excludes 0. "
      "A net inside the CI is the same null as the last four experiments.")
