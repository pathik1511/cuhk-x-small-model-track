"""
analyze_oof.py — where do the errors actually go?

Discriminates between two very different failure modes:

  FINE-GRAINED confusion (wash_face <-> brush_teeth <-> comb_hair): the classes are
  visually similar and the model can't resolve the detail that separates them.
  -> resolution / input representation is the bottleneck. Re-cache larger.

  SCATTERED confusion (errors spread evenly across unrelated classes): the model
  hasn't learned much of anything discriminative yet.
  -> capacity, training length, or modality coverage is the bottleneck.

Also reports per-class accuracy against subject support, which tells you how much of
your loss is structural (classes with too few subjects to ever generalize) versus
fixable.

Usage:
    python analyze_oof.py --oof runs/oof.csv --cache data/cache
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", default="runs/oof.csv")
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--mapping", default="data/raw/Small-Model-Track/class_mapping.csv")
    ap.add_argument("--top", type=int, default=20)
    a = ap.parse_args()
    pd.set_option("display.width", 220)

    oof = pd.read_csv(a.oof)
    man = pd.read_csv(Path(a.cache) / "manifest.csv")
    tr = man[man.split == "train"]
    name = {}
    mp = Path(a.mapping)
    if mp.exists():
        m = pd.read_csv(mp)
        c = [x for x in m.columns if "id" in x.lower()][0]
        n = [x for x in m.columns if "name" in x.lower()][0]
        name = dict(zip(m[c], m[n]))
    else:                                    # fall back to folder names in the manifest
        name = tr.drop_duplicates("action_id").set_index("action_id").action_name.to_dict()
    nm = lambda i: f"{i:>2} {str(name.get(i, '?'))[:26]:<26}"          # noqa: E731

    oof["ok"] = (oof.y_true == oof.y_pred).astype(int)
    print(f"clips {len(oof):,}   accuracy {oof.ok.mean():.4f}\n")

    print("=" * 78, "\nTOP CONFUSIONS  (true -> predicted)")
    bad = oof[oof.ok == 0]
    pair = bad.groupby(["y_true", "y_pred"]).size().sort_values(ascending=False)
    tot = len(bad)
    cum = 0
    for (t, p), n in pair.head(a.top).items():
        cum += n
        print(f"  {n:>4} ({100*n/tot:4.1f}%)  {nm(t)} ->  {nm(p)}")
    print(f"  top {a.top} pairs cover {100*cum/tot:.1f}% of all errors")
    print(f"  distinct confusion pairs: {len(pair)}  (max possible 1560)")
    conc = pair.head(a.top).sum() / tot
    print(f"  -> {'FINE-GRAINED: errors concentrate in few pairs' if conc > 0.35 else 'SCATTERED: errors spread widely — model is broadly weak'}")

    print("\n" + "=" * 78, "\nSYMMETRIC PAIRS  (mutual confusion = genuinely similar classes)")
    d = pair.to_dict()
    sym = sorted({tuple(sorted(k)): d.get(k, 0) + d.get(k[::-1], 0) for k in d}.items(),
                 key=lambda kv: -kv[1])[:10]
    for (i, j), n in sym:
        print(f"  {n:>4}   {nm(i)} <-> {nm(j)}")

    print("\n" + "=" * 78, "\nPER-CLASS ACCURACY vs SUBJECT SUPPORT")
    sup = tr.groupby("action_id").user.nunique()
    per = oof.groupby("y_true").agg(n=("ok", "size"), acc=("ok", "mean"))
    per["subjects"] = sup.reindex(per.index).fillna(0).astype(int)
    per["name"] = [str(name.get(i, "?"))[:26] for i in per.index]
    per = per.sort_values("acc")
    print(per.head(12).round(3).to_string())
    print("\n  best:")
    print(per.tail(5).round(3).to_string())
    lo = per[per.subjects <= 8]
    print(f"\n  classes with <=8 subjects: acc {lo.acc.mean():.3f} on {int(lo.n.sum())} clips")
    hi = per[per.subjects >= 14]
    print(f"  classes with >=14 subjects: acc {hi.acc.mean():.3f} on {int(hi.n.sum())} clips")
    print(f"  -> structural loss from thin classes: "
          f"{(hi.acc.mean()-lo.acc.mean())*lo.n.sum()/len(oof):.3f} accuracy points")

    if "user" in oof:
        print("\n" + "=" * 78, "\nPER-USER")
        u = oof.groupby("user").ok.agg(["size", "mean"]).round(3)
        print(u.to_string())


if __name__ == "__main__":
    main()
