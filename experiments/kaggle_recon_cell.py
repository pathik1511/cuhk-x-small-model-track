# ── CUHK-X Phase-0 recon ── paste into ONE Kaggle cell, hit Run. No downloads.
import os, re, json, collections as C

ROOT = "/kaggle/input"                       # competition data auto-mounts here
DEEP = 2                                     # files to probe per extension

def h(n):
    for u in "B KB MB GB TB".split():
        if n < 1024 or u == "TB": return f"{n:.1f}{u}"
        n /= 1024

def probe(p):
    e = os.path.splitext(p)[1].lower()
    try:
        if e == ".npy":
            import numpy as np; a = np.load(p, mmap_mode="r")
            return f"shape={a.shape} dtype={a.dtype} range=[{a.min():.4g},{a.max():.4g}]"
        if e == ".npz":
            import numpy as np
            with np.load(p) as z: return "npz " + ", ".join(f"{k}{z[k].shape}:{z[k].dtype}" for k in z.files[:12])
        if e in (".csv", ".txt", ".tsv"):
            L = open(p, errors="replace").read().splitlines()
            return f"{len(L)} lines | " + " || ".join(l[:150] for l in L[:3])
        if e == ".json":
            o = json.load(open(p))
            return f"keys={list(o)[:15]}" if isinstance(o, dict) else f"list[{len(o)}] first={str(o[0])[:180]}"
        if e in (".pkl", ".pickle"):
            import pickle; o = pickle.load(open(p, "rb"))
            return f"{type(o).__name__} " + (f"keys={list(o)[:15]}" if isinstance(o, dict) else f"len={len(o)}")
        if e in (".h5", ".hdf5"):
            import h5py, io; out = []
            with h5py.File(p) as f: f.visititems(lambda k, v: out.append(f"{k}{v.shape}") if hasattr(v, "shape") else None)
            return "h5 " + ", ".join(out[:15])
        if e in (".mp4", ".avi", ".mkv"):
            import cv2; c = cv2.VideoCapture(p); g = c.get
            r = f"video {int(g(3))}x{int(g(4))} n={int(g(7))} fps={g(5):.2f}"; c.release(); return r
        if e in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
            from PIL import Image
            with Image.open(p) as im: return f"image {im.size} mode={im.mode}"
    except Exception as ex:
        return f"<probe failed: {type(ex).__name__}: {ex}>"
    return "<no probe>"

F = [(os.path.join(d, f), os.path.getsize(os.path.join(d, f))) for d, _, fs in os.walk(ROOT) for f in fs]
print(f"ROOT {ROOT}\n{len(F):,} files   {h(sum(s for _, s in F))}\n")

print("=" * 70, "\nTREE")
agg = C.defaultdict(lambda: [0, 0])
for p, s in F:
    q = os.path.relpath(os.path.dirname(p), ROOT).split(os.sep)
    for d in range(min(len(q), 4) + 1):
        k = os.sep.join(q[:d]) or "."; agg[k][0] += 1; agg[k][1] += s
for k in sorted(agg)[:100]:
    print(f"  {'  '*k.count(os.sep)}{k:<50}{agg[k][0]:>8,} files {h(agg[k][1]):>9}")

print("\n" + "=" * 70, "\nEXTENSIONS")
ext = C.defaultdict(lambda: [0, 0])
for p, s in F:
    e = os.path.splitext(p)[1].lower() or "<none>"; ext[e][0] += 1; ext[e][1] += s
for e, (n, s) in sorted(ext.items(), key=lambda kv: -kv[1][1]):
    print(f"  {e:<10}{n:>8,} files {h(s):>9}  median {h(s/max(n,1))}")

print("\n" + "=" * 70, "\nMODALITY KEYWORDS")
for m in ("depth", "ir", "infra", "thermal", "imu", "mmwave", "radar", "skel", "pose"):
    n = sum(1 for p, _ in F if m in p.lower())
    if n: print(f"  {m:<9}{n:>8,} paths")

print("\n" + "=" * 70, "\nuser / action IDs PARSED FROM FILENAMES")
U = re.compile(r"(?:^|[^0-9a-z])(?:user|subject|sub|p|s|u)[_\-]?(\d{1,3})(?![0-9])", re.I)
A = re.compile(r"(?:^|[_\-])(?:a|act|action|class|label)[_\-]?(\d{1,3})(?![0-9])", re.I)
us, ac, pr = C.Counter(), C.Counter(), C.Counter()
for p, _ in F:
    b = os.path.relpath(p, ROOT); u, a = U.search(b), A.search(b)
    if u: us[int(u.group(1))] += 1
    if a: ac[int(a.group(1))] += 1
    if u and a: pr[(int(u.group(1)), int(a.group(1)))] += 1
print(f"  users: {sorted(us)}")
print(f"  actions: {len(ac)} distinct, range {min(ac,default='-')}..{max(ac,default='-')}")
if pr:
    miss = [(u, a) for u in us for a in ac if (u, a) not in pr]
    print(f"  (user,action) cells present={len(pr)}  MISSING={len(miss)}  e.g. {miss[:10]}")

print("\n" + "=" * 70, f"\nFILE PROBES (<= {DEEP}/ext)")
seen = C.Counter()
for p, s in sorted(F):
    e = os.path.splitext(p)[1].lower() or "<none>"
    if seen[e] >= DEEP: continue
    seen[e] += 1
    print(f"  {os.path.relpath(p, ROOT)}  [{h(s)}]\n      {probe(p)}")
