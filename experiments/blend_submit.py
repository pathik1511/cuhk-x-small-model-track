"""Blend video and old-family TEST probabilities into one submission.

  python src/blend_submit.py --video probs_video.csv --old probs_old.csv \
      --w 0.5 --out sub_blend.csv

Weight comes from src/blend.py on held-out training subjects. Never tune it here.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def norm(path):
    d = pd.read_csv(path)
    c = [x for x in d.columns if x[0] in "tp" and x[1:].isdigit()]
    assert len(c) == 40, f"{path}: {len(c)} class columns"
    m = d[c].values.astype(np.float64)
    return d.clip_id.values, m / m.sum(1, keepdims=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--old", required=True)
    p.add_argument("--w", type=float, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--ref", default="data/raw/Small-Model-Track/Testing/test_file/test.csv")
    a = p.parse_args()

    vi, V = norm(a.video)
    oi, O = norm(a.old)
    assert list(vi) == list(oi), "clip id order differs; align before blending"
    P = (1 - a.w) * V + a.w * O
    sub = pd.DataFrame({"path": [f"small_model_track_test/{i}/" for i in vi],
                        "prediction": P.argmax(1)})
    ref = Path(a.ref)
    if ref.exists():
        sub = pd.read_csv(ref)[["path"]].merge(sub, on="path", how="left")
        miss = int(sub.prediction.isna().sum())
        if miss:
            print(f"  WARNING: {miss} unmatched paths filled with class 0")
        sub["prediction"] = sub.prediction.fillna(0).astype(int)
    sub.to_csv(a.out, index=False)
    nv = int((V.argmax(1) != P.argmax(1)).sum())
    print(f"w={a.w}  {len(sub)} rows -> {a.out}   differs from video-only on {nv}/{len(vi)}")


if __name__ == "__main__":
    main()
