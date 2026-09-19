"""YOLO11n person boxes for every clip, written to a CSV the cache builder consumes.

Hypothesis Y1: a learned detector produces better crops than temporal-median motion
segmentation, especially when the performer is mostly stationary. This is NOT the same as
the person_box fallback patch, which only touched clips that had no box at all (2 test
clips, measured null in section 18.1). YOLO changes the box for EVERY clip.

Rules note: YOLO11n is ~2.6M parameters, COCO-pretrained, and is a detector rather than a
recognition backbone. Whether it counts against "no large pretrained backbones" is an
organizer question, not one this script answers.

Output: yolo_boxes.csv with clip_id,x0,y0,x1,y1,n_det,frac_frames_detected,fallback
Clips with no usable detection get fallback=1 and are left for the motion crop.
"""
import argparse, pickle, sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from build_cache import scan, pick


def person_boxes(model, files, probes, conf):
    """Detections on `probes` evenly spaced frames. Returns (boxes, n_frames_with_person)."""
    idx = sorted(set(pick(len(files), min(probes, len(files))).tolist()))
    imgs = []
    for j in idx:
        with Image.open(files[j]) as im:
            a = np.asarray(im.convert("L"), np.float32)
        # IR is single channel with variable gain. Per-frame contrast stretch, then
        # replicate to 3 channels: COCO weights expect RGB statistics.
        lo, hi = np.percentile(a, 1), np.percentile(a, 99)
        a = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1) * 255
        imgs.append(np.repeat(a.astype(np.uint8)[..., None], 3, -1))

    out, hits = [], 0
    for r in model.predict(imgs, conf=conf, classes=[0], verbose=False):
        b = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
        s = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.zeros(0)
        if not len(b):
            continue
        hits += 1
        out.append((b, s))
    return out, hits, len(idx)


MARGIN = 1.35


def consistent_track(dets, H, W):
    """Pick one performer per frame, then union.

    A multi-person frame must not contribute an unrelated highest-confidence box. Anchor on
    the highest-confidence box in the frame with the most confident detection, then in every
    other frame take the box whose centre is nearest that anchor.
    """
    if not dets:
        return None
    k = int(np.argmax([s.max() for _, s in dets]))
    anchor = dets[k][0][int(np.argmax(dets[k][1]))]
    ac = np.array([(anchor[0] + anchor[2]) / 2, (anchor[1] + anchor[3]) / 2])

    keep = []
    for b, s in dets:
        c = np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], 1)
        keep.append(b[int(np.argmin(((c - ac) ** 2).sum(1)))])
    keep = np.stack(keep)

    # Reject outlier frames before the union: a box whose centre is more than 1.5 median
    # absolute deviations from the track centre is a different person or a false positive.
    c = np.stack([(keep[:, 0] + keep[:, 2]) / 2, (keep[:, 1] + keep[:, 3]) / 2], 1)
    med = np.median(c, 0)
    d = np.abs(c - med).sum(1)
    mad = np.median(np.abs(d - np.median(d))) * 1.4826 + 1e-6
    keep = keep[d <= np.median(d) + 3 * mad] if len(keep) > 3 else keep
    if not len(keep):
        return None

    x0, y0 = keep[:, 0].min(), keep[:, 1].min()
    x1, y1 = keep[:, 2].max(), keep[:, 3].max()
    # 15% context, then square while preserving the centre, then clamp inside the frame.
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    # MARGIN MATTERS. build_cache's motion crop uses 1.35; the public 0.716 notebook
    # uses 1.40. A YOLO box at 1.15 is ~8% tighter than the motion box the models were
    # trained under, which makes "better localization" and "tighter crop" two variables
    # in one experiment. Default matched to the motion crop.
    tight = max(x1 - x0, y1 - y0)          # pre-margin, so margins re-derive for free
    side = min(max(tight * MARGIN, 0.34 * max(H, W)), min(H, W))
    bx = int(np.clip(cx - side / 2, 0, W - side))
    by = int(np.clip(cy - side / 2, 0, H - side))
    return bx, by, int(bx + side), int(by + side), float(cx), float(cy), float(tight)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/extracted")
    ap.add_argument("--out", default="yolo_boxes.csv")
    ap.add_argument("--weights", default="yolo11n.pt")
    ap.add_argument("--probes", type=int, default=8)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--margin", type=float, default=1.35,
                    help="crop margin. 1.35 matches build_cache motion crop; "
                         "the public 0.716 notebook uses 1.40")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    global MARGIN
    MARGIN = a.margin
    print(f"crop margin {MARGIN} "
          f"(motion crop uses 1.35, the 0.716 notebook uses 1.40)")
    from ultralytics import YOLO
    model = YOLO(a.weights)

    root = Path(a.root)
    ix = root.parent / "scan_index.pkl"
    clips = pickle.loads(ix.read_bytes()) if ix.exists() else scan(root)
    if a.limit:
        clips = clips[:a.limit]
    print(f"{len(clips)} clips", flush=True)

    rows, nofile = [], 0
    for n, c in enumerate(clips):
        # scan() nests file lists under "files", NOT at the top level of the clip
        # dict. c.get("IR") returns None for every clip and the len<3 branch below
        # then writes fallback=1 silently. That bug produced a 0/3441 detection
        # rate that was mistaken for a measurement.
        fl = c.get("files", {})
        f = fl.get("IR") or fl.get("Depth_Color") or []
        if len(f) < 3:
            nofile += 1
            rows.append(dict(clip_id=c["clip_id"], fallback=1, n_det=0, frac=0.0,
                             reason="fewer than 3 source frames"))
            continue
        try:
            with Image.open(f[0]) as im:
                W, H = im.size
            dets, hits, tot = person_boxes(model, f, a.probes, a.conf)
            box = consistent_track(dets, H, W)
        except Exception as e:
            rows.append(dict(clip_id=c["clip_id"], fallback=1, n_det=0, frac=0.0,
                             error=str(e)[:60]))
            continue
        if box is None:
            rows.append(dict(clip_id=c["clip_id"], fallback=1, n_det=0, frac=0.0))
        else:
            rows.append(dict(clip_id=c["clip_id"], x0=box[0], y0=box[1], x1=box[2],
                             y1=box[3], n_det=hits, frac=hits / max(tot, 1), fallback=0,
                             cx=box[4], cy=box[5], tight=box[6], W=W, H=H))
        if n % 200 == 0:
            print(f"  {n}/{len(clips)}", flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(a.out, index=False)
    print(f"\nwrote {a.out}  {len(d)} rows")
    print(f"detected        : {(d.fallback == 0).sum()} ({(d.fallback == 0).mean():.1%})")
    print(f"clips with <3 source frames: {nofile}")
    if nofile > 0.5 * len(d):
        print("\n*** REFUSED: most clips had no readable frames. That is a path bug,\n    not a detection rate. Check clip['files'] before trusting any number here. ***")
        sys.exit(1)
    print(f"fallback to motion crop: {(d.fallback == 1).sum()}")
    if (d.fallback == 0).any():
        ok = d[d.fallback == 0]
        print(f"frame coverage  : mean {ok.frac.mean():.2f}, min {ok.frac.min():.2f}")
        print(f"box side        : mean {(ok.x1 - ok.x0).mean():.0f}, "
              f"min {(ok.x1 - ok.x0).min()}, max {(ok.x1 - ok.x0).max()}")


if __name__ == "__main__":
    main()
