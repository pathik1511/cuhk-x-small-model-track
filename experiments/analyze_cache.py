"""
analyze_cache.py — the last diagnostic before modelling.

Answers, from the built cache:
  1. modality presence split by TRAIN vs TEST      -> how bad is the Radar mismatch
  2. subjects-per-class + per-fold coverage         -> which classes can't generalize
  3. clip length distribution per modality          -> is --frames 16 right
  4. per-modality mean/std                          -> normalization constants
  5. skeleton semantics: is column 3 depth or confidence, and is joint order H36M
     (the left/right swap table for horizontal flip depends on the answer)

Usage:
    python analyze_cache.py --cache data/cache --sample 300
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
MODS = ["Depth_Color", "IR", "Thermal", "Skeleton", "IMU", "Radar"]
H36M_R, H36M_L = [1, 2, 3, 14, 15, 16], [4, 5, 6, 11, 12, 13]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--sample", type=int, default=300)
    a = ap.parse_args()
    cache = Path(a.cache).expanduser()
    df = pd.read_csv(cache / "manifest.csv")
    pd.set_option("display.width", 200)

    print("=" * 74, "\n1. MODALITY PRESENCE BY SPLIT")
    has = [c for c in df.columns if c.startswith("has_")]
    p = df.groupby("split")[has].mean().T.round(4)
    p.columns = [f"{c} (n={int((df.split==c).sum())})" for c in p.columns]
    p["train-test gap"] = (p.iloc[:, 1] - p.iloc[:, 0]).round(4) if p.shape[1] > 1 else np.nan
    print(p.to_string())

    print("\n" + "=" * 74, "\n2. CLASS SUPPORT + FOLD COVERAGE")
    tr = df[df.split == "train"]
    try:
        import cuhk_cv as cv
        s = cv.class_support(tr)
        print("subjects per class (rarest 12):")
        print(s.head(12).T.to_string())
        print(f"\nclasses with <=8 subjects: {sorted(s[s.n_subjects <= 8].index.tolist())}")
        print(f"clips in those classes: {int(tr.action_id.isin(s[s.n_subjects<=8].index).sum())}"
              f" / {len(tr)} ({100*tr.action_id.isin(s[s.n_subjects<=8].index).mean():.1f}%)")
        for seed in (0, 1, 2):
            r = cv.fold_class_coverage(tr, seed=seed)
            bad = r[r.classes_absent_from_train.map(len) > 0]
            print(f"seed {seed}: folds with a class absent from train -> "
                  f"{'NONE' if bad.empty else bad[['fold','val_users','classes_absent_from_train']].to_dict('records')}")
    except Exception as e:
        print(f"  (cuhk_cv.py not importable from here: {e})")

    print("\n" + "=" * 74, "\n3. SOURCE FRAMES PER CLIP  (before the 16-frame resample)")
    ncols = [c for c in df.columns if c.startswith("n_")]
    q = df[df.split == "train"][ncols].replace(0, np.nan).quantile([.05, .25, .5, .75, .95]).round(1)
    print(q.to_string())

    print("\n" + "=" * 74, f"\n4. VALUE STATS  (sampling {a.sample} clips)")
    files = sorted((cache / "train").glob("*.npz"))
    rng = np.random.default_rng(0)
    files = [files[i] for i in rng.choice(len(files), min(a.sample, len(files)), replace=False)]
    acc: dict[str, list] = {m: [] for m in MODS}
    sk = []
    for f in files:
        with np.load(f) as z:
            for m in z.files:
                v = z[m].astype(np.float32)
                acc[m].append((v.mean(), v.std(), v.min(), v.max()))
                if m == "Skeleton":
                    sk.append(z[m].astype(np.float32))
    rows = []
    for m, v in acc.items():
        if v:
            b = np.asarray(v)
            rows.append({"modality": m, "n": len(v), "mean": b[:, 0].mean(), "std": b[:, 1].mean(),
                         "min": b[:, 2].min(), "max": b[:, 3].max()})
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    if sk:
        print("\n" + "=" * 74, "\n5. SKELETON SEMANTICS")
        S = np.concatenate(sk, 0)                       # [N, J, 3]
        print(f"  shape per frame: {S.shape[1:]}   ({S.shape[1]} joints)")
        for c, nm in enumerate("xyz"):
            v = S[..., c]
            print(f"  col {c} ({nm}): min={v.min():+.3f} max={v.max():+.3f} "
                  f"mean={v.mean():+.3f}  negatives={100*(v<0).mean():.1f}%")
        neg = (S[..., 2] < 0).mean()
        print(f"  -> col 2 is {'a 3D coordinate (has negatives)' if neg > 0.01 else 'likely a CONFIDENCE score (never negative)'}")
        j0 = np.abs(S[:, 0, :2]).mean()
        print(f"  joint 0 mean |x|,|y| = {j0:.4f}  -> {'ROOT-CENTERED' if j0 < 1e-3 else 'not root-centered'}")
        if S.shape[1] == 17:
            xr, xl = S[:, H36M_R, 0].mean(), S[:, H36M_L, 0].mean()
            print(f"  H36M test: mean x of joints{H36M_R}={xr:+.4f}  joints{H36M_L}={xl:+.4f}")
            print(f"  -> {'H36M order CONFIRMED — flip swap table is valid' if xr*xl < 0 else 'NOT standard H36M sides — do NOT enable horizontal flip'}")
        b = np.linalg.norm(S[:, 1:] - S[:, :1], axis=-1).mean(-1)
        print(f"  root-to-joint distance: mean={b.mean():.4f} std across clips={b.std():.4f}"
              f"  -> {'body-scale varies; normalize by it' if b.std() > .02 else 'already scale-normalized'}")


if __name__ == "__main__":
    main()
