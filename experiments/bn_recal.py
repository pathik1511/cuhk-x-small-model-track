"""Re-estimate BatchNorm running statistics after weight averaging.

soup.py averages running_mean and running_var straight across checkpoints. Those
buffers are the activation statistics of PARTICULAR weights; the average of them is
not the statistic of the averaged network. SWA recalibrates with a forward pass for
exactly this reason. This is unrelated to r52 (BN frozen DURING fine-tuning, -0.016);
that froze the statistics of one trained model, this re-derives them for a new one.

Clean test, no test data anywhere:
  python src/bn_recal.py --ckpt runs/r42_r2p1d_e30/fold0.pt --fold 0 --eval
    -> recalibrates on fold 0's TRAINING subjects, scores on its held-out subjects,
       printing before and after.

Apply to the soup (every training subject, because the soup has no holdout anyway):
  python src/bn_recal.py --ckpt runs/r42_soup/fold0.pt --fold -1 \
      --out runs/r42_soup_bn/fold0.pt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from cuhk_cv import subject_folds
from r2p1d import R2Plus1D34
from train_r2p1d import DIClips


def split(cache, fold, seed):
    df = pd.read_csv(Path(cache) / "manifest.csv")
    df = df[(df.split == "train") & df.action_id.ge(0)]
    if fold < 0:                       # no holdout: calibrate on everything
        return df, None
    tr_u, va_u = list(subject_folds(seed=seed))[fold]
    return df[df.user.isin(tr_u)], df[df.user.isin(va_u)]


@torch.no_grad()
def accuracy(model, dl, dev):
    model.eval()
    ok = n = 0
    for x, y, _, _ in dl:
        with torch.amp.autocast("cuda", enabled=dev.type == "cuda"):
            p = model(x.to(dev)).argmax(1).cpu()
        ok += int((p == y).sum()); n += len(y)
    return ok / max(n, 1)


@torch.no_grad()
def recalibrate(model, dl, dev):
    """Reset BN buffers, then accumulate a cumulative (not momentum) average."""
    bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm3d)]
    keep = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None              # cumulative average over batches. NOT an exact
                                       # pooled estimator: unweighted across batches, and
                                       # drops between-batch mean variation from the
                                       # variance. See campaign_summary section 30.
    model.eval()                       # dropout OFF
    for m in bns:
        m.train()                      # only the BN layers update their buffers
    for i, (x, *_) in enumerate(dl):
        model(x.to(dev))               # fp32 on purpose: statistics, not throughput
        if i % 50 == 0:
            print(f"    {i * dl.batch_size:>5d} clips", flush=True)
    for m, mo in zip(bns, keep):
        m.momentum = mo
    model.eval()
    return len(bns)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cache", default="data/cache_yolo")
    p.add_argument("--fold", type=int, default=-1,
                   help="calibrate on this fold's TRAIN users; -1 = all users")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--frames", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--eval", action="store_true", help="score the holdout before/after")
    p.add_argument("--out", default="")
    a = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr, va = split(a.cache, a.fold, a.seed)
    print(f"calibration clips {len(tr)}   holdout clips {0 if va is None else len(va)}")
    if va is not None:
        print(f"holdout users {sorted(int(u) for u in va.user.unique())}")

    # train=False: deterministic, deployment-like inputs. Calibrating on flipped and
    # rolled clips would estimate statistics for inputs inference never sees.
    dl = DataLoader(DIClips(tr, a.cache, False, 0, a.frames), a.batch_size,
                    shuffle=True, num_workers=a.workers, pin_memory=True)
    vl = (DataLoader(DIClips(va, a.cache, False, 0, a.frames), a.batch_size,
                     shuffle=False, num_workers=a.workers, pin_memory=True)
          if va is not None else None)

    s = torch.load(a.ckpt, map_location="cpu")
    model = R2Plus1D34(pretrained=False).to(dev)
    model.load_state_dict(s["model"])

    before = accuracy(model, vl, dev) if vl is not None else float("nan")
    if vl is not None:
        print(f"before  {before:.4f}")

    t0 = time.time()
    n = recalibrate(model, dl, dev)
    print(f"recalibrated {n} BatchNorm3d layers in {time.time()-t0:.0f}s")

    if vl is not None:
        after = accuracy(model, vl, dev)
        d = after - before
        print(f"after   {after:.4f}   delta {d:+.4f}   "
              f"({d*len(va):+.1f} clips of {len(va)})")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        torch.save({**s, "model": model.state_dict(), "bn_recal": True}, a.out)
        print(f"-> {a.out}")


if __name__ == "__main__":
    main()
