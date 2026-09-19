"""Leakage audit for the CUHK-X small model track. Four independent tests.
Run: python src/leak_audit.py [--cache data/cache] [--sub sub_r14_mix16.csv]
"""
import argparse, glob, hashlib, itertools, os
import numpy as np, pandas as pd

def test_a(te, rng):
    """Is the test id order shuffled, or does it follow source order?"""
    te = te.sort_values("idx")
    adj = lambda v: np.nanmean(np.abs(np.diff(np.asarray(v, float))))
    out = []
    for c in ["bytes", "n_Depth_Color", "n_Thermal", "n_IR", "has_Thermal", "has_Radar"]:
        o = adj(te[c]); n = [adj(rng.permutation(te[c].values)) for _ in range(2000)]
        mu, sd = np.mean(n), np.std(n)
        out.append((c, o, mu, sd, (o - mu) / sd if sd > 0 else 0.0))
    return out

def test_b(sub):
    """Do neighbouring test ids get the same predicted label (class-sorted ids)?"""
    lab = sub.sort_values("idx")["prediction"].values
    p = pd.Series(lab).value_counts(normalize=True).values
    exp = (p ** 2).sum(); obs = (lab[:-1] == lab[1:]).mean()
    sd = np.sqrt(exp * (1 - exp) / (len(lab) - 1))
    run = max(len(list(g)) for _, g in itertools.groupby(lab))
    return obs, exp, sd, (obs - exp) / sd, run

def test_c(tr):
    """Can manifest metadata alone predict the class cross-subject?"""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold
    F = [c for c in tr.columns if c.startswith(("n_", "has_"))] + ["bytes"]
    X, y, g = tr[F].values.astype(float), tr.action_id.values, tr.user.values
    a = []
    for i, j in GroupKFold(4).split(X, y, g):
        m = HistGradientBoostingClassifier(max_iter=120, random_state=0).fit(X[i], y[i])
        a.append((m.predict(X[j]) == y[j]).mean())
    return np.mean(a), tr.action_id.value_counts().max() / len(tr)

def test_d(cache):
    """Exact-content duplicates between train and test, via skeleton hash."""
    def sig(f):
        z = np.load(f)
        if "Skeleton" not in z.files: return None
        s = np.nan_to_num(z["Skeleton"].astype(np.float32))
        return hashlib.md5(np.round(s, 3).tobytes()).hexdigest()
    tr, te = {}, {}
    for d, acc in ((cache + "/train", tr), (cache + "/test", te)):
        for f in sorted(glob.glob(d + "/*.npz")):
            h = sig(f)
            if h: acc.setdefault(h, []).append(os.path.basename(f)[:-4])
    dup = lambda d: sum(len(v) - 1 for v in d.values() if len(v) > 1)
    return len(set(tr) & set(te)), dup(tr), dup(te), len(tr), len(te)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--sub", default="sub_r14_mix16.csv")
    a = ap.parse_args()
    m = pd.read_csv(a.cache + "/manifest.csv")
    te = m[m.split == "test"].copy(); te["idx"] = te.clip_id.str.extract(r"(\d+)$")[0].astype(int)
    tr = m[m.split == "train"].copy()
    rng = np.random.default_rng(0)

    print("A  test-id ordering (z < -3 means the ids follow source order)")
    for c, o, mu, sd, z in test_a(te, rng):
        print(f"   {c:16s} obs {o:10.2f}  null {mu:10.2f} +/- {sd:7.2f}  z {z:6.2f}")
    if os.path.exists(a.sub):
        s = pd.read_csv(a.sub); s["idx"] = s["path"].str.extract(r"SM_test_(\d+)")[0].astype(int)
        obs, exp, sd, z, run = test_b(s)
        print(f"B  neighbour-label agreement  obs {obs:.4f}  null {exp:.4f} +/- {sd:.4f}  z {z:6.2f}  longest run {run}")
    acc, maj = test_c(tr)
    print(f"C  metadata-only cross-subject accuracy {acc:.4f}  (chance 0.0250, majority {maj:.4f})")
    col, dtr, dte, ntr, nte = test_d(a.cache)
    print(f"D  train/test content collisions {col}  (train sigs {ntr}, test sigs {nte}); internal dups train {dtr} test {dte}")
