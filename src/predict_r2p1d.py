"""Ensemble the R(2+1)D-34 fold checkpoints over the 405 test clips.

Averages softmax across every fold*.pt found in --run, optional horizontal-flip TTA,
and writes the submission in the official row order.

  python src/predict_r2p1d.py --run runs/r42_r2p1d_e30 --out sub_r42_r2p1d.csv --tta
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from r2p1d import R2Plus1D34
from train_r2p1d import DIClips


def tta_probs(model, x, mode):
    """identity | +hflip | +temporal roll -1,+1. The shipped soup used 2-pass; the
    public 0.716 notebook used 4-pass. Roll is cyclic, which at 16 frames shifts the
    clip by one step rather than dropping a frame."""
    p = torch.softmax(model(x).float(), 1)
    if mode == "none":
        return p
    n = 1
    p = p + torch.softmax(model(torch.flip(x, dims=(-1,))).float(), 1); n += 1
    if mode == "4":
        for s in (-1, 1):
            p = p + torch.softmax(model(torch.roll(x, s, dims=1)).float(), 1); n += 1
    return p / n


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="runs/r42_r2p1d_e30")
    p.add_argument("--cache", default="data/cache_yolo")
    p.add_argument("--out", default="sub_r42_r2p1d.csv")
    p.add_argument("--frames", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--tta", default="none", choices=["none", "2", "4"],
                   help="2 = identity+hflip (what the shipped soup used); "
                        "4 = also temporal roll -1,+1")
    p.add_argument("--save-probs", default="", help="also write per-clip probabilities")
    a = p.parse_args()

    ck = sorted(Path(a.run).glob("fold*.pt"))
    assert ck, f"no fold*.pt in {a.run}"
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(Path(a.cache) / "manifest.csv")
    df = df[df.split == "test"].copy()
    ds = DIClips(df, a.cache, train=False, seed=0, frames=a.frames)
    dl = DataLoader(ds, a.batch_size, shuffle=False, num_workers=a.workers,
                    pin_memory=True)

    acc: dict[str, np.ndarray] = {}
    for c in ck:
        s = torch.load(c, map_location="cpu")
        model = R2Plus1D34(pretrained=False).to(dev)
        model.load_state_dict(s["model"])
        model.eval()
        n = 0
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=dev.type == "cuda"):
            for x, _, ids, _ in dl:
                x = x.to(dev, non_blocking=True)
                pr = tta_probs(model, x, a.tta).cpu().numpy()
                for k, cid in enumerate(ids):                 # accumulate by CLIP ID
                    acc[cid] = acc.get(cid, 0.0) + pr[k]
                    n += 1
        print(f"  {c.name}  acc@val {s.get('acc', float('nan')):.4f}  ({n} clips)")
        del model
        torch.cuda.empty_cache()

    ids = sorted(acc)
    probs = np.stack([acc[i] for i in ids])
    print(f"\n{len(ids)} unique test clips, {len(ck)} models, tta={a.tta}")
    if a.save_probs:                       # for cross-family blending, see src/blend.py
        pd.DataFrame(probs / probs.sum(1, keepdims=True),
                     columns=[f"p{i}" for i in range(probs.shape[1])]
                     ).assign(clip_id=ids).to_csv(a.save_probs, index=False)
        print(f"probabilities -> {a.save_probs}")
    sub = pd.DataFrame({"path": [f"small_model_track_test/{i}/" for i in ids],
                        "prediction": probs.argmax(1)})
    ref = Path(a.cache).parent / "raw/Small-Model-Track/Testing/test_file/test.csv"
    if ref.exists():                                   # keep the official row order
        sub = pd.read_csv(ref)[["path"]].merge(sub, on="path", how="left")
        miss = int(sub.prediction.isna().sum())
        if miss:
            print(f"  WARNING: {miss} test paths unmatched -- filling with class 0")
        sub["prediction"] = sub.prediction.fillna(0).astype(int)
    sub.to_csv(a.out, index=False)
    print(f"submission -> {a.out}  ({len(sub)} rows)")
    print(sub.head().to_string(index=False))


if __name__ == "__main__":
    main()
