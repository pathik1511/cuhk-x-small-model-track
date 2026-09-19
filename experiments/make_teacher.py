"""Average out-of-fold probability matrices into one teacher target file.

Every row of a run's oof.csv is a prediction made by the fold that did NOT train on that
clip, so averaging preserves the out-of-fold property. Runs whose fold assignment differs
from the reference are refused: mixing them would put in-fold predictions into the target
and distil memorisation instead of knowledge.

    python src/make_teacher.py --runs runs --ref r19_sadp --out runs/teacher_oof.csv

WARNING, READ BEFORE USING THE OUTPUT. These targets are out of fold with respect to the
clip, but NOT with respect to the student's validation subjects. The teacher fold that
predicts a clip of user u was itself trained on every other user, which includes the
subjects the student is being validated on. Distilling it leaks those subjects into the
student and inflates OOF by an unknown amount that will not transfer to the real test
users. Use this file for calibration and analysis, not for a number you intend to report.
See RESULTS.md, leave-two-folds-out construction, for the clean version.
"""
import argparse, glob, os
import numpy as np, pandas as pd

P = [f"p{i}" for i in range(40)]

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--ref", default="r19_sadp", help="run defining the fold assignment")
    ap.add_argument("--out", default="runs/teacher_oof.csv")
    ap.add_argument("--exclude", default="r18_ssl", help="comma separated run names to drop")
    a = ap.parse_args()
    drop = {x for x in a.exclude.split(",") if x}

    R = {}
    for f in sorted(glob.glob(os.path.join(a.runs, "*", "oof.csv"))):
        r = os.path.basename(os.path.dirname(f))
        d = pd.read_csv(f)
        if not set(P) <= set(d.columns) or d.fold.nunique() < 4 or r in drop:
            continue
        R[r] = d.drop_duplicates("clip_id").set_index("clip_id")
    if a.ref not in R:
        raise SystemExit(f"reference run {a.ref} has no 4-fold oof.csv")

    base = R[a.ref][["fold", "y_true"]]
    keep = []
    for r, d in R.items():
        i = base.index.intersection(d.index)
        if (base.loc[i, "fold"] == d.loc[i, "fold"]).all() and \
           (base.loc[i, "y_true"] == d.loc[i, "y_true"]).all():
            keep.append(r)
        else:
            print(f"  refused {r}: different fold assignment")
    idx = R[keep[0]].index
    for r in keep[1:]:
        idx = idx.intersection(R[r].index)

    p = np.mean([R[r].loc[idx, P].values for r in keep], 0)
    p = p / p.sum(1, keepdims=True)
    y = R[a.ref].loc[idx, "y_true"].values
    acc = float((p.argmax(1) == y).mean())
    print(f"  members ({len(keep)}): {', '.join(keep)}")
    print(f"  clips {len(idx)}   teacher OOF accuracy {acc:.4f} "
          f"+/- {np.sqrt(acc * (1 - acc) / len(idx)):.4f}")
    print(f"  mean mass on argmax {p.max(1).mean():.3f}   "
          f"entropy {-(p * np.log(p + 1e-9)).sum(1).mean():.3f} nats")

    out = pd.DataFrame(p, index=idx, columns=[f"t{i}" for i in range(40)])
    out.insert(0, "y_true", y)
    out.to_csv(a.out)
    print(f"  -> {a.out}")
    print("  WARNING: not leak-free against the student's validation subjects; "
          "see the module docstring before training on this.")

if __name__ == "__main__":
    main()
