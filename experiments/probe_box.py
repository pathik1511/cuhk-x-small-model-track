"""Does a fallback threshold recover the clips where person_box returns None?
Controlled: the existing path is untouched, so the 2,822 boxed clips stay byte-identical."""
import sys, numpy as np, pandas as pd
from pathlib import Path
from PIL import Image
from scipy import ndimage as ndi

R = Path("data/extracted/HAR/data")
def pick(n, t): return np.linspace(0, n - 1, t).round().astype(int)

def files_for(cid, mod="IR"):
    p = cid.split("_"); a = int(p[1][1:]); u = p[2][1:]; t = p[3][1:]
    act = next((d for d in (R / mod).iterdir() if d.name.startswith(f"{a}_")), None)
    if act is None: return []
    d = act / u / t
    return sorted(d.glob("*.png")) if d.is_dir() else []

def mask_stats(mo, med, sig):
    m = mo > max(med + 4 * sig, 8.0)
    m = ndi.binary_opening(m, np.ones((3, 3)), iterations=2)
    m = ndi.binary_closing(m, np.ones((9, 9)), iterations=2)
    lab, n = ndi.label(m)
    if n == 0: return None
    sizes = ndi.sum(m, lab, range(1, n + 1)); k = int(np.argmax(sizes)) + 1
    H, W = mo.shape
    ys, xs = np.nonzero(lab == k)
    frac = float(sizes[k - 1] / (H * W))
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    margin = 1.35 if frac > 0.02 else 2.4
    cx = (x0 + x1) / 2; cy = (y0 + y1) / 2 + (0.0 if frac > 0.02 else 0.35 * max(x1-x0, y1-y0))
    side = max(x1 - x0, y1 - y0) * margin
    side = min(max(side, 0.34 * max(H, W)), min(H, W))
    bx = int(np.clip(cx - side/2, 0, W - side)); by = int(np.clip(cy - side/2, 0, H - side))
    return (bx, by, int(bx + side), int(by + side)), frac

def both(files):
    idx = sorted(set(pick(len(files), min(12, len(files))).tolist()))
    V = np.stack([np.asarray(Image.open(files[j]).convert("L"), np.float32) for j in idx])
    mo = np.abs(V - np.median(V, 0)).max(0)
    med = float(np.median(mo)); sig = float(np.median(np.abs(mo - med))) * 1.4826 + 1e-6
    old = mask_stats(mo, med, sig)
    # FALLBACK: estimate background from the lower 60% of motion, which stays background
    # even when the subject covers a large share of the frame.
    lo = mo[mo <= np.percentile(mo, 60)]
    m2 = float(np.median(lo)); s2 = float(np.median(np.abs(lo - m2))) * 1.4826 + 1e-6
    new = mask_stats(mo, m2, s2)
    return old, new, med, sig, m2, s2

man = pd.read_csv("data/cache_hand/manifest.csv")
fail = man[man.box.isna() & (man.n_IR.fillna(0) >= 3)]
ok   = man[man.box.notna() & (man.action_id == 36)]
print(f"clips with data but no box: {len(fail)}  (train {(fail.split=='train').sum()}, test {(fail.split=='test').sum()})")

rec, rows = 0, []
for cid in fail.clip_id:
    f = files_for(cid) if cid.startswith("train") else []
    if cid.startswith("SM_test"):
        d = R.parent.parent / "small_model_track_test" / cid / "IR"
        f = sorted(d.glob("*.png")) if d.is_dir() else []
    if len(f) < 3: rows.append((cid, "nofiles", None)); continue
    try:
        old, new, *_ = both(f)
    except Exception as e:
        rows.append((cid, "err:" + str(e)[:25], None)); continue
    if old is not None: rows.append((cid, "OLD_OK", old[0])); continue
    if new is not None: rec += 1; rows.append((cid, "RECOVERED", new[0]))
    else: rows.append((cid, "still_none", None))
d = pd.DataFrame(rows, columns=["clip", "status", "box"])
print(d.status.value_counts().to_string())
r = d[d.status == "RECOVERED"].copy()
if len(r):
    r["side"] = [b[2] - b[0] for b in r.box]
    r["cx"] = [(b[0] + b[2]) / 2 for b in r.box]
    print(f"\nrecovered box side: mean {r.side.mean():.0f} min {r.side.min()} max {r.side.max()}")
    print(f"recovered box cx:   mean {r.cx.mean():.0f} min {r.cx.min():.0f} max {r.cx.max():.0f}")
    o = ok.copy()
    o["side"] = [int(str(b).split("-")[2]) - int(str(b).split("-")[0]) for b in o.box]
    print(f"CONTROL, existing Walk boxes: side mean {o.side.mean():.0f} min {o.side.min()} max {o.side.max()}")
