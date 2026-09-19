"""
explore_data.py — Phase 0 recon. Run this and paste me the output.

I need the real file layout, tensor shapes, clip lengths, per-user/per-class counts
and missing-modality rates before I write a Dataset class. Guessing these produces
code you throw away.

Usage:
    python explore_data.py /kaggle/input/cuhk-x-competition-small-model-track
    python explore_data.py ~/Documents/.../data --deep 5     # peek into 5 files/type
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys

MODALITY_HINTS = ("depth", "ir", "infra", "thermal", "imu", "mmwave", "radar", "skel", "pose")
USER_RE = re.compile(r"(?:^|[^0-9a-z])(?:user|subject|sub|p|s|u)[_\-]?(\d{1,3})(?![0-9])", re.I)
ACT_RE = re.compile(r"(?:^|[_\-])(?:a|act|action|class|label)[_\-]?(\d{1,3})(?![0-9])", re.I)


def human(n: float) -> str:
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or u == "TB":
            return f"{n:.1f}{u}"
        n /= 1024


def walk(root: str):
    for dp, _, fns in os.walk(root):
        for fn in fns:
            p = os.path.join(dp, fn)
            try:
                yield p, os.path.getsize(p)
            except OSError:
                pass


def peek(path: str) -> str:
    """Best-effort shape/dtype probe. Never raises."""
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".npy":
            import numpy as np
            a = np.load(path, mmap_mode="r", allow_pickle=False)
            return f"ndarray shape={a.shape} dtype={a.dtype} min={a.min():.4g} max={a.max():.4g}"
        if ext == ".npz":
            import numpy as np
            with np.load(path, allow_pickle=False) as z:
                return "npz " + ", ".join(f"{k}{z[k].shape}:{z[k].dtype}" for k in z.files[:12])
        if ext in (".csv", ".txt", ".tsv"):
            with open(path, errors="replace") as f:
                head = [next(f, "").rstrip() for _ in range(3)]
            n = sum(1 for _ in open(path, errors="replace"))
            return f"{n} lines | " + " || ".join(h[:160] for h in head if h)
        if ext == ".json":
            with open(path) as f:
                o = json.load(f)
            return (f"dict keys={list(o)[:15]}" if isinstance(o, dict)
                    else f"list len={len(o)} first={str(o[0])[:200]}")
        if ext in (".pkl", ".pickle"):
            import pickle
            with open(path, "rb") as f:
                o = pickle.load(f)
            return f"{type(o).__name__} " + (f"keys={list(o)[:15]}" if isinstance(o, dict) else f"len={len(o)}")
        if ext in (".h5", ".hdf5"):
            import h5py
            with h5py.File(path, "r") as f:
                out = []
                f.visititems(lambda k, v: out.append(f"{k}{getattr(v,'shape','')}") if hasattr(v, "shape") else None)
            return "h5 " + ", ".join(out[:15])
        if ext in (".mp4", ".avi", ".mkv"):
            import cv2
            c = cv2.VideoCapture(path)
            r = (f"video {int(c.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
                 f"n={int(c.get(cv2.CAP_PROP_FRAME_COUNT))} fps={c.get(cv2.CAP_PROP_FPS):.2f}")
            c.release()
            return r
        if ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
            from PIL import Image
            with Image.open(path) as im:
                return f"image {im.size} mode={im.mode}"
        if ext in (".pt", ".pth"):
            import torch
            o = torch.load(path, map_location="cpu", weights_only=False)
            return f"{type(o).__name__} " + (f"keys={list(o)[:15]}" if isinstance(o, dict) else str(getattr(o, "shape", "")))
    except Exception as e:
        return f"<probe failed: {type(e).__name__}: {e}>"
    return "<no probe for this extension>"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--deep", type=int, default=3, help="files to probe per extension")
    ap.add_argument("--depth", type=int, default=4, help="tree depth to print")
    a = ap.parse_args()
    root = os.path.expanduser(a.root)
    if not os.path.isdir(root):
        sys.exit(f"not a directory: {root}")

    files = list(walk(root))
    print(f"ROOT {root}\nFILES {len(files):,}  TOTAL {human(sum(s for _, s in files))}\n")

    print("=" * 72, "\nTOP-LEVEL TREE (dirs, file count, size)")
    agg = collections.defaultdict(lambda: [0, 0])
    for p, s in files:
        rel = os.path.relpath(os.path.dirname(p), root)
        parts = [] if rel == "." else rel.split(os.sep)
        for d in range(min(len(parts), a.depth) + 1):
            k = os.sep.join(parts[:d]) or "."
            agg[k][0] += 1
            agg[k][1] += s
    for k in sorted(agg)[:120]:
        print(f"  {'  ' * k.count(os.sep)}{k:<52} {agg[k][0]:>8,} files  {human(agg[k][1]):>9}")

    print("\n" + "=" * 72, "\nEXTENSIONS")
    ext = collections.defaultdict(lambda: [0, 0])
    for p, s in files:
        e = os.path.splitext(p)[1].lower() or "<none>"
        ext[e][0] += 1
        ext[e][1] += s
    for e, (n, s) in sorted(ext.items(), key=lambda kv: -kv[1][1]):
        print(f"  {e:<12} {n:>8,} files  {human(s):>9}  (median {human(s / max(n,1))})")

    print("\n" + "=" * 72, "\nMODALITY KEYWORDS IN PATHS")
    for m in MODALITY_HINTS:
        n = sum(1 for p, _ in files if m in p.lower())
        if n:
            print(f"  {m:<10} {n:>8,} paths")

    print("\n" + "=" * 72, "\nPARSED user / action IDS FROM FILENAMES")
    users, acts, pairs = collections.Counter(), collections.Counter(), collections.Counter()
    for p, _ in files:
        b = os.path.relpath(p, root)
        u = USER_RE.search(b)
        c = ACT_RE.search(b)
        if u:
            users[int(u.group(1))] += 1
        if c:
            acts[int(c.group(1))] += 1
        if u and c:
            pairs[(int(u.group(1)), int(c.group(1)))] += 1
    print(f"  users found: {sorted(users)}")
    print(f"  n_actions found: {len(acts)}  range {min(acts, default='-')}..{max(acts, default='-')}")
    if pairs:
        cnt = collections.Counter(u for u, _ in pairs)
        print(f"  distinct (user,action) cells: {len(pairs)}  | classes covered per user: "
              f"min={min(cnt.values())} max={max(cnt.values())}")
        miss = [(u, c) for u in users for c in acts if (u, c) not in pairs]
        print(f"  MISSING (user,action) cells: {len(miss)}  e.g. {miss[:10]}")

    print("\n" + "=" * 72, f"\nFILE PROBES (up to {a.deep} per extension)")
    seen = collections.Counter()
    for p, s in sorted(files, key=lambda kv: kv[0]):
        e = os.path.splitext(p)[1].lower() or "<none>"
        if seen[e] >= a.deep:
            continue
        seen[e] += 1
        print(f"  {os.path.relpath(p, root)}  [{human(s)}]\n      {peek(p)}")


if __name__ == "__main__":
    main()
