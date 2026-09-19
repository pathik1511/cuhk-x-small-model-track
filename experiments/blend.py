"""Grid a probability blend of the video model and an old-family model on OOF.

  python src/blend.py --old runs/r19_sadp/oof.csv

Only subject-disjoint predictions are used: both sides are each clip's held-out fold.
No test data, no leaderboard.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

P = [f"p{i}" for i in range(40)]


def load_probs(path):
    d = pd.read_csv(path)
    cols = [c for c in d.columns if c[0] in "tp" and c[1:].isdigit()]
    assert len(cols) == 40, f"{path}: found {len(cols)} class columns"
    m = d[cols].values.astype(np.float64)
    m = m / m.sum(1, keepdims=True)            # renormalize; sources may differ
    return pd.DataFrame(m, columns=P).assign(clip_id=d.clip_id.values)


def main() -> None:
    a = argparse.ArgumentParser()
    a.add_argument("--video", default="runs/r42_r2p1d_e30/oof_probs.csv")
    a.add_argument("--old", default="runs/r19_sadp/oof.csv")
    a.add_argument("--weights", default="0,0.1,0.15,0.2,0.25,0.3,0.4,0.5")
    a = a.parse_args()

    v = pd.read_csv(a.video)
    key = v[["clip_id", "user", "fold", "y_true"]]
    vp = load_probs(a.video)
    op = load_probs(a.old)
    m = key.merge(vp, on="clip_id").merge(op, on="clip_id", suffixes=("_v", "_o"))
    print(f"joined {len(m)} of {len(key)} held-out clips\n")
    V = m[[f"{c}_v" for c in P]].values
    O = m[[f"{c}_o" for c in P]].values
    y = m.y_true.values
    base = (V.argmax(1) == y)
    print(f"{'w':>6} {'acc':>8} {'clips':>7} {'C':>5} {'R':>5}   per-fold")
    for w in [float(x) for x in a.weights.split(",")]:
        pred = ((1 - w) * V + w * O).argmax(1)
        ok = pred == y
        C = int((~base & ok).sum()); R = int((base & ~ok).sum())
        pf = "  ".join(f"{f}:{ok[m.fold.values == f].mean():.3f}" for f in sorted(m.fold.unique()))
        print(f"{w:>6.2f} {ok.mean():>8.4f} {int(ok.sum()) - int(base.sum()):>+7d} "
              f"{C:>5} {R:>5}   {pf}")


if __name__ == "__main__":
    main()
