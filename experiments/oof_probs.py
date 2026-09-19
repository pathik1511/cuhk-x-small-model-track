"""Re-run each r42 fold model over its OWN held-out subjects, saving probabilities.

oof_f*.csv stores only argmax, which cannot be blended. Same TTA as the submission so
the blend weight is chosen under the conditions it will ship under.

  python src/oof_probs.py --run runs/r42_r2p1d_e30 --tta
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
from cuhk_cv import subject_folds
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
    p.add_argument("--frames", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tta", default="none", choices=["none", "2", "4"])
    a = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(Path(a.cache) / "manifest.csv")
    df = df[(df.split == "train") & df.action_id.ge(0)]
    out = []
    for fold in range(4):
        ck = Path(a.run) / f"fold{fold}.pt"
        if not ck.exists():
            print(f"  fold {fold}: no checkpoint, skipped"); continue
        _, va_u = list(subject_folds(seed=a.seed))[fold]
        va = df[df.user.isin(va_u)]
        dl = DataLoader(DIClips(va, a.cache, False, 0, a.frames), a.batch_size,
                        shuffle=False, num_workers=a.workers, pin_memory=True)
        model = R2Plus1D34(pretrained=False).to(dev)
        model.load_state_dict(torch.load(ck, map_location="cpu")["model"])
        model.eval()
        rows = []
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=dev.type == "cuda"):
            for x, y, cid, usr in dl:
                x = x.to(dev, non_blocking=True)
                pr = tta_probs(model, x, a.tta).cpu().numpy()
                for k, c in enumerate(cid):
                    rows.append([c, int(usr[k]), fold, int(y[k]), *pr[k]])
        d = pd.DataFrame(rows, columns=["clip_id", "user", "fold", "y_true",
                                        *[f"p{i}" for i in range(40)]])
        acc = float((d.y_true == d[[f"p{i}" for i in range(40)]].values.argmax(1)).mean())
        print(f"  fold {fold}: {len(d):4d} clips  acc {acc:.4f}", flush=True)
        out.append(d)
        del model; torch.cuda.empty_cache()

    o = pd.concat(out)
    dst = Path(a.run) / f"oof_probs_tta{a.tta}.csv"
    o.to_csv(dst, index=False)
    print(f"-> {dst}  ({len(o)} rows)")


if __name__ == "__main__":
    main()
