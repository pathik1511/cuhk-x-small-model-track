"""Convert NTU RGB+D 120 .skeleton files into the CUHK-X skeleton cache convention.

    python src/ntu_prep.py --root data/ntu/nturgb+d_skeletons --out data/ntu_cache \
                           --missing data/ntu/NTU_RGBD120_samples_with_missing_skeletons.txt

Five conversions, each forced by a difference between the two corpora. Getting any one
of them wrong makes the pretrained weights worse than useless, so each is asserted.

1. JOINTS.   NTU is Kinect 25-joint, CUHK-X is 17-joint H3.6M from a pose estimator.
             Mapped by NTU_TO_H36M below.
2. AXES.     Kinect camera frame is x right, y up, z depth. CUHK-X is
             (x lateral, y depth, z height), so the axes permute to (x, z, y).
3. ORIGIN.   CUHK-X has the root at exactly (0, 0, z) in every frame of the corpus and
             the minimum z exactly 0.000000 in every frame. Reproduced here per frame:
             subtract the root in x and y, subtract the lowest joint in z.
4. RATE.     NTU is 30 fps, CUHK-X is 10 fps (timestamps 100 ms apart). Take every 3rd
             frame BEFORE sampling 16, so the temporal extent matches.
5. BODIES.   NTU has up to 2 people; CUHK-X is single actor throughout. Keep the body
             with the most motion.

Torso-length normalisation is NOT applied here. dataset._skel does it at load time, and
doing it twice would silently halve the scale.
"""
from __future__ import annotations

import argparse, os, re
from pathlib import Path

import numpy as np
import pandas as pd

#                 hip rhip rkne rank lhip lkne lank spn thx nck hea lsh lel lwr rsh rel rwr
NTU_TO_H36M = [0, 16, 17, 18, 12, 13, 14, 1, 20, 2, 3, 4, 5, 6, 8, 9, 10]
NAME_RE = re.compile(r"S(\d{3})C(\d{3})P(\d{3})R(\d{3})A(\d{3})")


def read_skeleton(path: Path) -> np.ndarray | None:
    """-> (T, bodies, 25, 3) in raw Kinect coordinates, or None if unusable."""
    tok = path.read_text().split("\n")
    it = iter(t for t in tok if t.strip())
    try:
        n_frames = int(next(it))
    except (StopIteration, ValueError):
        return None
    frames = []
    try:                                   # 535 NTU-120 samples are truncated or empty
        for _ in range(n_frames):
            bodies = []
            for _ in range(int(next(it))):
                next(it)                                   # body info line
                n_joints = int(next(it))
                j = np.empty((n_joints, 3), np.float32)
                for k in range(n_joints):
                    j[k] = [float(x) for x in next(it).split()[:3]]
                bodies.append(j)
            frames.append(bodies)
    except (StopIteration, ValueError, IndexError):
        return None
    # A frame can legitimately contain ZERO bodies: the tracker loses the person, or the
    # clip starts before they enter. frames[0] is therefore not guaranteed to hold one,
    # and taking frames[0][0] to learn the joint count crashes on those files.
    v = next((b[0].shape[0] for b in frames if b), 0)
    m = max((len(b) for b in frames), default=0)
    if not v or not m:
        return None
    out = np.full((len(frames), m, v, 3), np.nan, np.float32)
    for t, bs in enumerate(frames):
        for i, b in enumerate(bs):
            if b.shape[0] == v:                    # skip a body with an odd joint count
                out[t, i] = b
    return out


def pick_body(x: np.ndarray) -> np.ndarray:
    """(T,M,V,3) -> (T,V,3). The body that moves most, holding gaps at the last frame."""
    energy = [np.nan_to_num(np.nanstd(x[:, i], 0)).sum() for i in range(x.shape[1])]
    y = x[:, int(np.argmax(energy))]
    ok = ~np.isnan(y).any((1, 2))
    if not ok.any():
        return None
    idx = np.maximum.accumulate(np.where(ok, np.arange(len(y)), 0))
    return y[np.where(ok.cumsum() > 0, idx, np.argmax(ok))]


def to_cuhk(x: np.ndarray, frames: int, stride: int) -> np.ndarray:
    """(T,25,3) Kinect -> (frames,17,3) in the CUHK-X convention."""
    x = x[::stride]
    if len(x) == 0:
        return None
    i = np.linspace(0, len(x) - 1, frames).round().astype(int)
    y = x[i][:, NTU_TO_H36M]                                  # joints
    y = y[..., [0, 2, 1]]                                     # axes -> lateral, depth, height
    y[..., :2] -= y[:, :1, :2]                                # root-centre x and y per frame
    y[..., 2] -= y[..., 2].min(1, keepdims=True)              # floor-reference z per frame
    return np.ascontiguousarray(y, np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dir of .skeleton files")
    ap.add_argument("--out", default="data/ntu_cache")
    ap.add_argument("--missing", default=None, help="NTU_RGBD120_samples_with_missing_skeletons.txt")
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--stride", type=int, default=3, help="30 fps -> 10 fps")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    a = ap.parse_args()

    skip = set()
    if a.missing and Path(a.missing).exists():
        skip = {l.strip() for l in Path(a.missing).read_text().split("\n")
                if NAME_RE.fullmatch(l.strip())}     # the file has prose header lines
        print(f"  excluding {len(skip)} samples listed as missing skeletons")

    files = sorted(Path(a.root).rglob("*.skeleton"))
    files = [f for f in files if f.stem not in skip]
    if a.limit:
        files = files[:a.limit]
    print(f"  {len(files)} skeleton files")
    out = Path(a.out); (out / "train").mkdir(parents=True, exist_ok=True)

    from concurrent.futures import ProcessPoolExecutor
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(_one, [(str(f), str(out), a.frames, a.stride) for f in files],
                                     chunksize=64)):
            if r:
                rows.append(r)
            if (i + 1) % 5000 == 0:
                print(f"  {i+1}/{len(files)}  kept {len(rows)}")
    df = pd.DataFrame(rows)
    df.to_csv(out / "manifest.csv", index=False)
    print(f"\n  kept {len(df)} of {len(files)}  ({len(files) - len(df)} unreadable)")
    print(f"  subjects {df.user.nunique()}   classes {df.action_id.nunique()}   "
          f"clips/subject median {int(df.groupby('user').size().median())}")
    print(f"  -> {out/'manifest.csv'}")


def _one(args):
    """One file. Returns None rather than raising: a 113,945-file job must not die
    because one input is malformed."""
    try:
        return _one_inner(args)
    except Exception:
        return None


def _one_inner(args):
    fp, out, frames, stride = args
    fp = Path(fp)
    m = NAME_RE.search(fp.stem)
    if not m:
        return None
    x = read_skeleton(fp)
    if x is None:
        return None
    x = pick_body(x)
    if x is None:
        return None
    y = to_cuhk(x, frames, stride)
    if y is None or not np.isfinite(y).all():
        return None
    np.savez(Path(out) / "train" / f"{fp.stem}.npz", Skeleton=y.astype(np.float16))
    s, c, p, r, act = (int(g) for g in m.groups())
    return dict(clip_id=fp.stem, split="train", action_id=act - 1, user=p,
                setup=s, camera=c, trial=r, has_Skeleton=1, n_Skeleton=len(x))


if __name__ == "__main__":
    main()
