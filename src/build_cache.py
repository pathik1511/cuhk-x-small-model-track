"""
build_cache.py — decode-once preprocessing for CUHK-X Small Model Track.

Turns 46 GB of PNG/JPG/JSON/CSV into per-clip .npz arrays plus a manifest, so
training never touches an image decoder again. Auto-detects train vs test layout.

Layout learned from the archives:
  train  HAR/data/<Modality>/<action_id>_<Action_Name>/<user>/<trial>/<files>
  test   small_model_track_test/<SM_test_NNNN>/<Modality>/<files>

Modalities, as they actually are on disk:
  Depth_Color  640x480 RGB  .png   ~10 fps, filename carries an absolute timestamp
  IR           640x480 L    .png   ~10 fps, timestamps match Depth_Color exactly
  Skeleton     JSON per frame      ~10 fps, timestamps match Depth_Color exactly
  Thermal      320x240 RGB  .jpg   ~25 fps, filenames are frame_NNNNNN — NO timestamp,
                                   so it can only be aligned proportionally
  IMU          2 CSVs/clip         5 devices (LA,RA,C,LL,RL) split across up()/down()
  Radar        1 CSV/clip          point cloud rows; summarized per time-bin

Missing data is normal: Thermal absent in ~10/405 test clips, Radar in ~1, and some
CSVs contain a header and nothing else. Every loader returns None rather than raising,
and the manifest records presence per modality so the model can be trained with
modality dropout.

Usage:
    python build_cache.py --root data/extracted --out data/cache --dry-run
    python build_cache.py --root data/extracted --out data/cache --workers 10
    python build_cache.py --root data/extracted --out data/cache --modalities Depth_Color,Skeleton
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

MODALITIES = ["Depth_Color", "IR", "Thermal", "Skeleton", "IMU", "Radar"]
IMG_MODS = {"Depth_Color": 3, "IR": 1, "Thermal": 3}
DEVICES = ["LA", "RA", "C", "LL", "RL"]          # left arm, right arm, chest, left leg, right leg
IMU_COLS = slice(2, 18)                           # accel3 gyro3 angle3 mag3 quat4 = 16 channels
RADAR_STATS = 12
JUNK = re.compile(r"(^|/)(__MACOSX|\.DS_Store|\._|\.claude|\.git)", re.I)
ACTION_RE = re.compile(r"^(\d{1,2})_(.+)$")
TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2}[_ ]\d{2}-\d{2}-\d{2}\.\d{3})")
FRAME_RE = re.compile(r"(\d{5,})")


# ---------------------------------------------------------------- scanning ---
def scan(root: Path) -> list[dict]:
    """Walk the extracted tree once and group files into clips."""
    clips: dict[tuple, dict] = {}
    for p in root.rglob("*"):
        if not p.is_file() or JUNK.search(p.as_posix()):
            continue
        parts = p.relative_to(root).parts
        mod = next((m for m in MODALITIES if m in parts), None)
        if mod is None:
            continue
        i = parts.index(mod)
        if i + 1 < len(parts) and (m := ACTION_RE.match(parts[i + 1])):   # train
            aid, name = int(m.group(1)), m.group(2)
            user = parts[i + 2] if len(parts) > i + 2 else "?"
            trial = parts[i + 3] if len(parts) > i + 3 else "?"
            key, meta = ("train", aid, user, trial), dict(
                split="train", action_id=aid, action_name=name,
                user=uid(user), user_raw=user, trial=trial,
                clip_id=f"train_a{aid:02d}_u{user}_t{trial}",
                example=p.relative_to(root).as_posix())
        elif i > 0 and parts[i - 1].startswith("SM_test_"):               # test
            key, meta = ("test", parts[i - 1]), dict(
                split="test", action_id=-1, action_name="", user=-1,
                user_raw="", trial="", clip_id=parts[i - 1],
                example=p.relative_to(root).as_posix())
        else:
            continue
        c = clips.setdefault(key, {**meta, "files": defaultdict(list)})
        c["files"][mod].append(p)
    for c in clips.values():
        for m in c["files"]:
            c["files"][m].sort()
    return list(clips.values())


def uid(seg: str) -> int:
    """Subject id from a folder segment. Accepts '7', 'user7', 'P07', 'sub_07', 'u07a'."""
    m = re.search(r"\d+", seg or "")
    return int(m.group()) if m else -1


def pick(n: int, t: int) -> np.ndarray:
    """t uniformly-spaced indices over n items (repeats if n < t)."""
    return np.linspace(0, max(n - 1, 0), t).round().astype(int) if n else np.zeros(0, int)


# ----------------------------------------------------------------- loaders ---
def person_box(files: list[Path], probes: int = 12, min_frac: float = 0.34):
    """Square person box in native image coords, from motion alone. No detector, no
    pretrained weights -- rule-legal.

    The camera is fixed within a trial, so the per-clip temporal MEDIAN is the empty
    room and |frame - median| is the person. Take the largest connected component of
    that motion mask. A subject who barely moves (Watch_TV) shows only a limb, so when
    the blob is small the box is widened and dropped toward the body.

    Returns (x0, y0, x1, y1) or None. Verified visually on Sweep / Write / Watch_TV /
    Walk, which span 0.4% to 35% blob fraction.
    """
    from PIL import Image
    try:
        from scipy import ndimage as ndi
    except ImportError:
        return None
    if len(files) < 3:
        return None
    idx = sorted(set(pick(len(files), min(probes, len(files))).tolist()))
    try:
        V = np.stack([np.asarray(Image.open(files[j]).convert("L"), np.float32) for j in idx])
    except Exception:
        return None
    T, H, W = V.shape
    mo = np.abs(V - np.median(V, 0)).max(0)
    med = float(np.median(mo))
    sig = float(np.median(np.abs(mo - med))) * 1.4826 + 1e-6
    def _mask(centre, scale):
        k = mo > max(centre + 4 * scale, 8.0)
        k = ndi.binary_opening(k, np.ones((3, 3)), iterations=2)
        k = ndi.binary_closing(k, np.ones((9, 9)), iterations=2)
        return k, ndi.label(k)[1]

    m, n = _mask(med, sig)
    if n == 0:
        # FALLBACK. med+4*MAD assumes the moving region is a small share of the frame.
        # A subject who traverses the frame makes the motion mask dense, which lifts both
        # med and MAD above the motion values, so the mask empties and this returned None.
        # Measured: 117 clips (109 train, 8 test) had image data and no box, Walk worst.
        # Re-estimate the background from the lower 60% of motion, which stays background
        # in both regimes. The primary path above is byte-identical, so the 3,218 clips
        # that already had a box keep exactly the box they had.
        lo = mo[mo <= np.percentile(mo, 60)]
        m2 = float(np.median(lo))
        s2 = float(np.median(np.abs(lo - m2))) * 1.4826 + 1e-6
        m, n = _mask(m2, s2)
    if n == 0:
        return None
    lab, _ = ndi.label(m)
    sizes = ndi.sum(m, lab, range(1, n + 1))
    k = int(np.argmax(sizes)) + 1
    frac = float(sizes[k - 1] / (H * W))
    ys, xs = np.nonzero(lab == k)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    margin = 1.35 if frac > 0.02 else 2.4
    # Median-of-per-frame-centres was tried and REVERTED: it fixes stray YOLO detections,
    # which this detector does not have (it takes the largest component of an aggregated
    # mask). Verified on 4 clips: identical on Write/Watch_TV, worse-centred on
    # Sweep/Walk. Extremes of the aggregate mask are correct here.
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2 + (0.0 if frac > 0.02 else 0.35 * max(x1 - x0, y1 - y0))
    side = max(x1 - x0, y1 - y0) * margin
    side = min(max(side, min_frac * max(H, W)), min(H, W))
    bx = int(np.clip(cx - side / 2, 0, W - side))
    by = int(np.clip(cy - side / 2, 0, H - side))
    return (bx, by, int(bx + side), int(by + side))


def hand_box(files: list[Path], pbox, probes: int = 12, frac: float = 0.45,
             min_side: int = 96):
    """Tightest MOVING region inside the person box -- the hands.

    The skeleton is metric 3D with no camera intrinsics, so there are no wrist pixel
    coordinates to crop around. But for a fine-motor action the hands are the only thing
    moving, so the motion peak inside the person box lands on them. Verified visually:
    on 18_Write this returns the hands, the paper and the pen at native resolution, a
    region that is ~15x15 px inside the 128x128 person crop.
    """
    from PIL import Image
    try:
        from scipy import ndimage as ndi
    except ImportError:
        return None
    if pbox is None or len(files) < 3:
        return None
    idx = sorted(set(pick(len(files), min(probes, len(files))).tolist()))
    try:
        V = np.stack([np.asarray(Image.open(files[j]).convert("L"), np.float32) for j in idx])
    except Exception:
        return None
    H, W = V.shape[1:]
    mo = np.abs(V - np.median(V, 0)).max(0)
    med = float(np.median(mo)); sig = float(np.median(np.abs(mo - med))) * 1.4826 + 1e-6
    x0, y0, x1, y1 = pbox
    side = int(max(min_side, (x1 - x0) * frac))
    sub = np.maximum(mo[y0:y1, x0:x1] - (med + 2 * sig), 0)
    if sub.max() <= 0:
        return None
    k = max(9, side // 4) | 1
    sm = ndi.uniform_filter(sub, size=k)
    py, px = np.unravel_index(int(np.argmax(sm)), sm.shape)
    thr = np.percentile(sm, 99.0)
    ys, xs = np.nonzero(sm >= thr)
    w = sm[ys, xs]
    cy = 0.5 * float((ys * w).sum() / w.sum()) + 0.5 * py     # centroid, anchored on peak
    cx = 0.5 * float((xs * w).sum() / w.sum()) + 0.5 * px
    hx = int(np.clip(x0 + cx - side / 2, 0, W - side))
    hy = int(np.clip(y0 + cy - side / 2, 0, H - side))
    return (hx, hy, hx + side, hy + side)


def load_images(files: list[Path], size: tuple[int, int], t: int, ch: int, box=None):
    from PIL import Image
    if not files:
        return None
    idx = pick(len(files), t)
    out = np.zeros((t, size[1], size[0], ch), np.uint8)
    for k, j in enumerate(idx):
        try:
            with Image.open(files[j]) as im:
                im = im.convert("RGB" if ch == 3 else "L")
                if box is not None:
                    im = im.crop(box)
                im = im.resize(size, Image.BILINEAR)
                a = np.asarray(im)
            out[k] = a[..., None] if ch == 1 else a
        except Exception:
            if k:
                out[k] = out[k - 1]                       # hold last good frame
    return out


def load_skeleton(files: list[Path], t: int):
    if not files:
        return None
    idx, arr = pick(len(files), t), None
    for k, j in enumerate(idx):
        try:
            o = json.loads(files[j].read_text())
            kp = np.asarray((o[0] if isinstance(o, list) else o)["keypoints"], np.float32)
        except Exception:
            kp = None
        if arr is None:
            if kp is None:
                continue
            arr = np.zeros((t, *kp.shape), np.float32)
        if kp is not None and kp.shape == arr.shape[1:]:
            arr[k] = kp
        elif k:
            arr[k] = arr[k - 1]
    return None if arr is None else arr.astype(np.float16)


def _read_csv(fp: Path):
    import csv as _csv
    with open(fp, newline="", encoding="utf-8-sig", errors="replace") as f:
        rows = list(_csv.reader(f))
    return (rows[0], rows[1:]) if rows else ([], [])


def load_imu(files: list[Path], t: int):
    """-> float32 [t, 5 devices, 16 channels]. Absent devices stay zero."""
    if not files:
        return None
    per: dict[str, list] = defaultdict(list)
    for fp in files:
        _, rows = _read_csv(fp)
        for r in rows:
            if len(r) < 18:
                continue
            m = re.search(r"WT([A-Z]{1,2})\s*\(", r[1])
            if not m or m.group(1) not in DEVICES:
                continue
            try:
                per[m.group(1)].append([float(x or 0) for x in r[IMU_COLS]])
            except ValueError:
                continue
    if not per:
        return None
    out = np.zeros((t, len(DEVICES), 16), np.float32)
    for d, v in per.items():
        a = np.asarray(v, np.float32)
        out[:, DEVICES.index(d)] = a[pick(len(a), t)]
    return out


def load_radar(files: list[Path], t: int):
    """-> float32 [t, 12]: per-time-bin point-cloud summary."""
    if not files:
        return None
    pts = []
    for fp in files:
        hdr, rows = _read_csv(fp)
        try:
            ci = [hdr.index(c) for c in ("x", "y", "z", "v", "snr")]
        except ValueError:
            continue
        for r in rows:
            try:
                pts.append([float(r[i]) for i in ci])
            except (ValueError, IndexError):
                continue
    if not pts:
        return None
    a = np.asarray(pts, np.float32)
    out = np.zeros((t, RADAR_STATS), np.float32)
    for k, sl in enumerate(np.array_split(np.arange(len(a)), t)):
        if not len(sl):
            continue
        b = a[sl]
        out[k] = [*b.mean(0), *b.std(0), len(b), b[:, 3].max()]
    return out


# ---------------------------------------------------------------- yolo boxes ---
_YOLO: dict | None = None


def yolo_box(clip_id: str, path: str):
    """Precomputed YOLO person box, or None to fall back to the motion crop.

    Loaded once per worker process. Returning None is the documented fallback, not an
    error: the fallback rate is reported by yolo_box.py and recorded per clip in the
    manifest so the comparison can be split on it later.
    """
    global _YOLO
    if _YOLO is None:
        import pandas as _pd
        d = _pd.read_csv(path)
        d = d[d.fallback == 0]
        _YOLO = {r.clip_id: (int(r.x0), int(r.y0), int(r.x1), int(r.y1))
                 for r in d.itertuples()}
        print(f"[yolo] {len(_YOLO)} precomputed boxes from {path}", flush=True)
    return _YOLO.get(clip_id)


# ------------------------------------------------------------------ worker ---
def build_one(clip: dict, out_dir: str, size: tuple[int, int], t: int, mods: list[str],
              crop: bool = False, hand: bool = False, hand_size: int = 96,
              yolo: str = "") -> dict:
    fp = Path(out_dir) / clip["split"] / f"{clip['clip_id']}.npz"
    rec = {k: clip[k] for k in ("clip_id", "split", "action_id", "action_name", "user", "trial")}
    if fp.exists():
        rec["cached"] = 1
        return rec
    try:
        f, data = clip["files"], {}
        # One box per clip, from IR (Depth as fallback), applied to the two modalities
        # that share that 640x480 sensor geometry. Thermal is a different camera with a
        # different FOV -- never reuse the box there.
        box = person_box(f.get("IR") or f.get("Depth_Color") or []) if crop else None
        motion_box = box          # KEEP: the hand crop must stay on the motion signal
        if yolo and crop:
            # Y1 replaces ONLY the whole-person box. `hand_box` below is passed
            # `motion_box`, not `box`, so the hand stream is byte-identical to the
            # motion-crop cache and the comparison moves exactly one variable. Deriving
            # the hand crop from the YOLO box would change two things at once and make
            # any result unattributable.
            yb = yolo_box(clip["clip_id"], yolo)
            rec["yolo"] = int(yb is not None)
            if yb is not None:
                box = yb
        rec["box"] = "" if box is None else "-".join(map(str, box))
        hbox = hand_box(f.get("IR") or f.get("Depth_Color") or [], motion_box) if hand else None
        rec["hand_box"] = "" if hbox is None else "-".join(map(str, hbox))
        for m in mods:
            bx = box if m in ("Depth_Color", "IR") else None
            v = (load_images(f.get(m, []), size, t, IMG_MODS[m], bx) if m in IMG_MODS else
                 load_skeleton(f.get(m, []), t) if m == "Skeleton" else
                 load_imu(f.get(m, []), t) if m == "IMU" else
                 load_radar(f.get(m, []), t))
            rec[f"has_{m}"] = int(v is not None)
            rec[f"n_{m}"] = len(f.get(m, []))
            if v is not None:
                data[m] = v
            # second, tighter view of the same sensors at native resolution
            if hand and hbox is not None and m in ("Depth_Color", "IR"):
                hv = load_images(f.get(m, []), (hand_size, hand_size), t, IMG_MODS[m], hbox)
                if hv is not None:
                    data["Hand_" + m] = hv
        if not data:
            rec["error"] = "no modality loaded"
            return rec
        fp.parent.mkdir(parents=True, exist_ok=True)
        np.savez(fp, **data)
        rec["bytes"] = fp.stat().st_size
        rec["cached"] = 0
    except Exception:
        rec["error"] = traceback.format_exc(limit=2)
    return rec


# -------------------------------------------------------------------- main ---
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dir containing extracted HAR/ and/or small_model_track_test/")
    ap.add_argument("--out", default="data/cache")
    ap.add_argument("--size", default="128x96", help="WxH for image modalities")
    ap.add_argument("--frames", type=int, default=32, help="frames kept per clip per modality")
    ap.add_argument("--modalities", default=",".join(MODALITIES))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--limit", type=int, default=0, help="build only N clips (smoke test)")
    ap.add_argument("--crop", action="store_true",
                    help="crop Depth_Color and IR to a motion-derived person box before "
                         "resizing (classical CV, no detector, rule-legal)")
    ap.add_argument("--hand", action="store_true",
                    help="also cache a tight hand-region crop (implies --crop)")
    ap.add_argument("--hand-size", type=int, default=96)
    ap.add_argument("--yolo", default="", help="CSV from yolo_box.py; replaces the "
                    "motion person box where a detection exists")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    a.crop = a.crop or a.hand
    w, hgt = (int(x) for x in a.size.lower().split("x"))
    size, mods = (w, hgt), [m for m in a.modalities.split(",") if m in MODALITIES]
    root, out = Path(a.root).expanduser(), Path(a.out).expanduser()

    print(f"scanning {root} ...")
    # The rglob over data/extracted walks hundreds of thousands of files and takes
    # minutes, during which the process prints nothing. Three concurrent builds each
    # repeating it thrash one disk. Walk once, reuse the index.
    import pickle
    ix = root.parent / "scan_index.pkl"
    clips = None
    if ix.exists():
        # A pickle written on Windows holds WindowsPath objects, and unpickling those
        # on Linux raises NotImplementedError("cannot instantiate 'WindowsPath'").
        # The index is a local speed cache, never a package artifact: on ANY failure,
        # fall through to a fresh walk rather than killing the run.
        try:
            clips = pickle.loads(ix.read_bytes())
            print(f"reusing file index {ix} (delete it if data/extracted changed)",
                  flush=True)
        except Exception as e:
            print(f"file index unusable ({type(e).__name__}), walking instead", flush=True)
            clips = None
    if clips is None:
        print(f"walking {root} (several minutes, no output until done)", flush=True)
        clips = scan(root)
        try:
            ix.write_bytes(pickle.dumps(clips))
            print(f"wrote file index {ix}", flush=True)
        except Exception as e:
            print(f"could not cache index: {e}", flush=True)
    if not clips:
        sys.exit(f"no clips found under {root} — did the archives extract there?")
    by_split = defaultdict(int)
    for c in clips:
        by_split[c["split"]] += 1
    print(f"{len(clips):,} clips  {dict(by_split)}")

    per = sum(a.frames * hgt * w * IMG_MODS[m] for m in mods if m in IMG_MODS)
    per += a.frames * (17 * 3 * 2 if "Skeleton" in mods else 0)
    per += a.frames * 4 * ((5 * 16 if "IMU" in mods else 0) + (RADAR_STATS if "Radar" in mods else 0))
    print(f"modalities: {mods}\nsize={size} frames={a.frames}"
          f"  ~{per/1e6:.2f} MB/clip  ->  ~{per*len(clips)/1e9:.1f} GB total")

    tr = [c for c in clips if c["split"] == "train"]
    if tr:
        raw = sorted({c["user_raw"] for c in tr})
        users = sorted({c["user"] for c in tr})
        acts = sorted({c["action_id"] for c in tr})
        trials = sorted({c["trial"] for c in tr})
        print(f"\nEXAMPLE TRAIN PATHS")
        for c in tr[:3]:
            print(f"  {c['example']}")
        print(f"\nuser folder names ({len(raw)}): {raw[:40]}")
        print(f"parsed user ids  ({len(users)}): {users}")
        print(f"trial folders    ({len(trials)}): {trials[:20]}")
        print(f"train actions: {len(acts)} ids {min(acts)}..{max(acts)}"
              f"{'  CONTIGUOUS' if acts == list(range(len(acts))) else ''}")
        if users == [-1]:
            sys.exit("\nSTOP: could not parse subject ids — paste the EXAMPLE TRAIN PATHS above to me.")
        cov, cpu = defaultdict(set), defaultdict(int)
        for c in tr:
            cov[c["user"]].add(c["action_id"])
            cpu[c["user"]] += 1
        print(f"clips per user: {dict(sorted(cpu.items()))}")
        print(f"classes per user: min={min(len(v) for v in cov.values())} "
              f"max={max(len(v) for v in cov.values())}")
        holes = {u: sorted(set(acts) - v) for u, v in cov.items() if len(v) < len(acts)}
        print(f"users missing classes: {holes if holes else 'none — full 40x coverage'}")
    if a.dry_run:
        return

    if a.limit:
        clips = clips[:a.limit]
    out.mkdir(parents=True, exist_ok=True)
    recs, errs = [], 0
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(build_one, c, str(out), size, a.frames, mods, a.crop,
                          a.hand, a.hand_size, a.yolo) for c in clips]
        for i, fu in enumerate(as_completed(futs), 1):
            r = fu.result()
            recs.append(r)
            errs += "error" in r
            if i % 200 == 0 or i == len(futs):
                print(f"  {i:,}/{len(futs):,}  errors={errs}", flush=True)

    import pandas as pd
    df = pd.DataFrame(recs)
    mp = out / "manifest.csv"
    df.to_csv(mp, index=False)
    print(f"\nmanifest -> {mp}   {len(df):,} rows, {errs} errors")
    if a.crop and "box" in df.columns:
        got = (df.box.fillna("") != "").mean()
        print(f"person box found for {got*100:.1f}% of clips "
              f"(the rest fall back to the uncropped full frame)")
        if "hand_box" in df.columns:
            print(f"hand box found for {(df.hand_box.fillna('') != '').mean()*100:.1f}% of clips")
    if errs:
        print(df[df.get("error").notna()][["clip_id", "error"]].head(3).to_string())
    pres = [c for c in df.columns if c.startswith("has_")]
    if pres:
        print("\nmodality presence:")
        print((df[pres].mean().rename("fraction of clips").to_frame().round(4)).to_string())
    if "bytes" in df:
        print(f"\ncache size: {df.bytes.sum()/1e9:.2f} GB")


if __name__ == "__main__":
    main()
