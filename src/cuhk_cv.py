"""
cuhk_cv.py — cross-subject CV that mirrors the CUHK-X Small Model Track leaderboard.

LB split:  train users {1..9, 16..24}   test users {10, 11, 25, 26}
Observation: the user IDs form two contiguous blocks (1-11, 16-26), which almost
certainly correspond to the dataset's two indoor recording environments. The test
set draws TWO subjects from each block. So a fold that holds out 4 random subjects
is NOT the same distribution as the LB — it may take all 4 from one environment.

Every fold here holds out 2 subjects from block A and 2 from block B. This matters
for two reasons:
  1. Fold accuracy is an unbiased estimate of LB accuracy (same subject count,
     same environment mix) rather than an optimistic one.
  2. Fold-to-fold spread is an honest proxy for LB spread, so you can tell a real
     gain from noise before you burn a submission on it.

Dependencies: numpy, pandas.  (scikit-learn only for the optional sanity check.)
"""
from __future__ import annotations

import itertools
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Split definition
# --------------------------------------------------------------------------- #
BLOCKS: dict[str, list[int]] = {
    "A": list(range(1, 10)),    # train users 1-9   -> LB test users 10, 11
    "B": list(range(16, 25)),   # train users 16-24 -> LB test users 25, 26
}
TEST_USERS = [10, 11, 25, 26]
HOLDOUT_PER_BLOCK = 2                      # matches LB composition exactly
TRAIN_USERS = sorted(itertools.chain(*BLOCKS.values()))


def subject_folds(n_folds: int = 4, seed: int = 0, per_block: int = HOLDOUT_PER_BLOCK):
    """Yield (train_users, val_users). Val folds are disjoint within a seed.

    9 users per block / 2 held out => 4 disjoint folds, 1 user per block unseen.
    Run 2-3 seeds and average to cover everyone and shrink estimator variance.
    """
    rng = np.random.default_rng(seed)
    perm = {k: rng.permutation(v) for k, v in BLOCKS.items()}
    assert n_folds * per_block <= min(len(v) for v in BLOCKS.values()), "folds exceed block size"
    for i in range(n_folds):
        val = sorted(int(u) for p in perm.values() for u in p[i * per_block:(i + 1) * per_block])
        yield sorted(set(TRAIN_USERS) - set(val)), val


def fold_indices(users: np.ndarray, n_folds: int = 4, seed: int = 0):
    """Same thing, but as boolean masks over a per-clip user array. Use in a DataLoader."""
    users = np.asarray(users)
    for tr, va in subject_folds(n_folds, seed):
        yield np.isin(users, tr), np.isin(users, va)


# --------------------------------------------------------------------------- #
# Leakage + composition guards — run these once, keep them in your test suite
# --------------------------------------------------------------------------- #
def assert_valid(n_folds: int = 4, seed: int = 0) -> None:
    seen: set[int] = set()
    for k, (tr, va) in enumerate(subject_folds(n_folds, seed)):
        assert not set(tr) & set(va), f"fold {k}: user leakage {set(tr) & set(va)}"
        assert not seen & set(va), f"fold {k}: val users repeat across folds"
        assert not set(va) & set(TEST_USERS), f"fold {k}: LB test user in val"
        assert sum(u in BLOCKS["A"] for u in va) == HOLDOUT_PER_BLOCK, f"fold {k}: block A imbalance"
        assert sum(u in BLOCKS["B"] for u in va) == HOLDOUT_PER_BLOCK, f"fold {k}: block B imbalance"
        assert len(tr) + len(va) == len(TRAIN_USERS), f"fold {k}: user count mismatch"
        seen |= set(va)
    print(f"OK  {n_folds} folds, seed={seed}, no leakage, 2+2 block composition held.")


# --------------------------------------------------------------------------- #
# Class x subject coverage — the thing that silently wrecks cross-subject folds
# --------------------------------------------------------------------------- #
def class_support(df: pd.DataFrame, n_classes: int = 40) -> pd.DataFrame:
    """df: manifest rows with columns [user, action_id]. How many SUBJECTS per class."""
    g = df.groupby("action_id").user.nunique().reindex(range(n_classes), fill_value=0)
    return g.rename("n_subjects").sort_values().to_frame()


def fold_class_coverage(df: pd.DataFrame, n_folds: int = 4, seed: int = 0,
                        n_classes: int = 40, min_subjects: int = 2) -> pd.DataFrame:
    """For every fold: which classes end up with too little training support, and how
    many val clips sit in those classes. A class present in only 3 of 18 subjects can
    lose ALL of them to one held-out fold — the model then scores 0 on it by construction.
    """
    out = []
    for k, (tr, va) in enumerate(subject_folds(n_folds, seed)):
        t, v = df[df.user.isin(tr)], df[df.user.isin(va)]
        sup = t.groupby("action_id").user.nunique().reindex(range(n_classes), fill_value=0)
        zero = sorted(sup[sup == 0].index)
        weak = sorted(sup[(sup > 0) & (sup < min_subjects)].index)
        out.append({
            "fold": k, "val_users": va, "train_clips": len(t), "val_clips": len(v),
            "classes_absent_from_train": zero,
            f"classes_under_{min_subjects}_subjects": weak,
            "val_clips_unlearnable": int(v.action_id.isin(zero).sum()),
            "val_pct_unlearnable": round(100 * v.action_id.isin(zero).mean(), 2),
        })
    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# OOF reporting
# --------------------------------------------------------------------------- #
def oof_report(df: pd.DataFrame) -> dict:
    """df columns: user, fold, y_true, y_pred.

    Returns pooled accuracy, the fold mean/std you should compare gains against,
    and per-subject accuracy so you can see WHICH subject your model fails on.
    """
    d = df.assign(ok=(df.y_true == df.y_pred).astype(float))
    by_fold = d.groupby("fold").ok.mean()
    by_user = d.groupby("user").ok.mean().sort_values()
    return {
        "pooled_acc": float(d.ok.mean()),
        "fold_mean": float(by_fold.mean()),
        "fold_std": float(by_fold.std(ddof=1)),
        "per_fold": by_fold.round(4).to_dict(),
        "per_user": by_user.round(4).to_dict(),
        "worst_user": (int(by_user.index[0]), float(by_user.iloc[0])),
        "subject_spread": float(by_user.max() - by_user.min()),
    }


def lb_noise(df: pd.DataFrame, n_test_users: int = 4, n_boot: int = 5000, seed: int = 0) -> dict:
    """How much would the LB move if it had drawn a DIFFERENT 4 subjects?

    Resamples 4 subjects (2 per block, matching the LB) from your OOF predictions.
    The returned std is your noise floor: any LB delta smaller than ~2*std is
    indistinguishable from luck. On a 4-subject test set this is usually large.
    """
    rng = np.random.default_rng(seed)
    d = df.assign(ok=(df.y_true == df.y_pred).astype(float))
    per_user = {int(u): g.ok.values for u, g in d.groupby("user")}
    a = [u for u in per_user if u in BLOCKS["A"]]
    b = [u for u in per_user if u in BLOCKS["B"]]
    k = n_test_users // 2
    accs = np.array([
        np.concatenate([per_user[u] for u in (*rng.choice(a, k, replace=False),
                                              *rng.choice(b, k, replace=False))]).mean()
        for _ in range(n_boot)
    ])
    return {
        "mean": float(accs.mean()),
        "std": float(accs.std(ddof=1)),
        "p05_p95": (float(np.percentile(accs, 5)), float(np.percentile(accs, 95))),
        "min_meaningful_delta": float(2 * accs.std(ddof=1)),
    }


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    assert_valid()
    for i, (tr, va) in enumerate(subject_folds()):
        print(f"fold {i}: val={va}  (train n={len(tr)})")

    # Smoke test of the reporting path on synthetic predictions.
    rng = np.random.default_rng(0)
    rows = []
    for f, (_, va) in enumerate(subject_folds()):
        for u in va:                       # per-subject skill varies -> that's the point
            n, p = 400, rng.uniform(0.40, 0.75)
            y = rng.integers(0, 40, n)
            hit = rng.random(n) < p
            rows.append(pd.DataFrame({"user": u, "fold": f, "y_true": y,
                                      "y_pred": np.where(hit, y, (y + 1) % 40)}))
    oof = pd.concat(rows, ignore_index=True)
    r = oof_report(oof)
    print(f"\npooled={r['pooled_acc']:.4f}  fold={r['fold_mean']:.4f} +/- {r['fold_std']:.4f}"
          f"  subject spread={r['subject_spread']:.3f}")
    n = lb_noise(oof)
    print(f"LB noise: sd={n['std']:.4f}  90% band={n['p05_p95'][0]:.3f}-{n['p05_p95'][1]:.3f}"
          f"  -> ignore LB deltas below {n['min_meaningful_delta']:.4f}")
