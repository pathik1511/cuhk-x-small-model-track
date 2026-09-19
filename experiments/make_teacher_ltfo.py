"""Assemble leak-free per-student-fold teacher targets from the six LTFO runs.

A teacher is admissible for student fold F on clip X only if it saw neither val_users(F)
nor X's own user. With 16 users in 4 folds that is exactly the model trained on the 8
users outside folds F and G, where G is X's fold. Six unordered pairs cover everything,
and each run serves two student folds.

    python src/make_teacher_ltfo.py --runs runs --out runs/teacher_ltfo_f{fold}.csv
"""
import argparse, itertools, os, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from cuhk_cv import subject_folds                                   # noqa: E402

P = [f"p{i}" for i in range(40)]
T = [f"t{i}" for i in range(40)]

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--out", default="runs/teacher_ltfo_f{fold}.csv")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="t{a}{b}", help="run dir name pattern")
    a = ap.parse_args()

    folds = [v for _, v in subject_folds(seed=a.seed)]
    src = {}
    for i, j in itertools.combinations(range(4), 2):
        d = os.path.join(a.runs, a.tag.format(a=i, b=j), "oof.csv")
        if not os.path.exists(d):
            raise SystemExit(f"missing {d}; run the six LTFO teachers first")
        src[(i, j)] = pd.read_csv(d).drop_duplicates("clip_id").set_index("clip_id")
        held = set(folds[i]) | set(folds[j])
        seen = set(src[(i, j)].user.unique().tolist())
        if not seen <= held:
            raise SystemExit(f"LTFO({i},{j}) predicted users {sorted(seen - held)} "
                             f"outside its holdout; that run trained on them")
        acc = (src[(i, j)][P].values.argmax(1) == src[(i, j)].y_true.values).mean()
        print(f"  LTFO({i},{j})  n={len(src[(i,j)]):5d}  holdout acc {acc:.4f}")

    for f in range(4):
        parts = []
        for g in range(4):
            if g == f:
                continue
            d = src[tuple(sorted((f, g)))]
            parts.append(d[d.user.isin(folds[g])])
        d = pd.concat(parts)
        assert not d.index.duplicated().any(), "a clip got two teachers"
        assert not set(d.user) & set(folds[f]), "teacher covers the student's own holdout"
        p = d[P].values.astype(np.float32)
        p = p / p.sum(1, keepdims=True)
        out = pd.DataFrame(p, index=d.index, columns=T)
        out.insert(0, "y_true", d.y_true.values)
        path = a.out.format(fold=f)
        out.to_csv(path)
        acc = (p.argmax(1) == d.y_true.values).mean()
        print(f"  student fold {f}: {len(out):5d} clips, users {sorted(set(d.user))}, "
              f"teacher acc {acc:.4f}  -> {path}")

if __name__ == "__main__":
    main()
