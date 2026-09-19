"""Pretrain the EXISTING skeleton encoder on NTU RGB+D 120, then hand it to train.py.

    python src/pretrain_skel.py --cache data/ntu_cache --out runs/ntu_skel.pt --epochs 20

Deliberately reuses `SeqEncoder` from model.py and `_skel`/`_streams`/`_aug_skel` from
dataset.py rather than introducing CTR-GCN at the same time. Pretraining and an
architecture change are two variables; measuring them together would tell you nothing,
and this project has already paid for that lesson twice.

Held-out NTU subjects are reported every epoch. The number to watch is not accuracy, it
is whether the encoder generalises ACROSS PERFORMERS, since that is the only property
that transfers to a cross-subject target task.
"""
from __future__ import annotations

import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))
from dataset import CUHKClips                                  # noqa: E402
from model import SeqEncoder                                   # noqa: E402


class NTUSkel(Dataset):
    """NTU npz written by ntu_prep.py, put through the identical CUHK-X skeleton path."""

    def __init__(self, df: pd.DataFrame, root: Path, frames: int = 16,
                 streams: bool = True, train: bool = True):
        self.df, self.root, self.frames = df.reset_index(drop=True), Path(root), frames
        self.streams, self.train = streams, train
        self.rng = np.random.default_rng(0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        v = np.load(self.root / "train" / f"{r.clip_id}.npz")["Skeleton"]
        x = CUHKClips._skel(v)                                  # [T,17,3] -> [T,17,4]
        if self.train:
            x = CUHKClips._aug_skel(self, x, int(self.rng.integers(-2, 3)),
                                    bool(self.rng.random() < 0.5),
                                    float(self.rng.normal(0, 0.1)))
        if self.streams:
            x = CUHKClips._streams(x)                           # -> [T,17,10]
        return torch.from_numpy(np.ascontiguousarray(x, np.float32)), int(r.action_id)


def _wi(i):
    np.random.seed(torch.initial_seed() % 2 ** 31 + i)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="data/ntu_cache")
    p.add_argument("--out", default="runs/ntu_skel.pt")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--streams", action="store_true", default=True)
    p.add_argument("--holdout", type=int, default=12, help="NTU subjects kept for validation")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    torch.manual_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(Path(a.cache) / "manifest.csv")
    users = np.sort(df.user.unique())
    va_u = set(np.random.default_rng(a.seed).choice(users, a.holdout, replace=False).tolist())
    tr, va = df[~df.user.isin(va_u)], df[df.user.isin(va_u)]
    n_cls = int(df.action_id.max()) + 1
    print(f"device {dev}  {len(tr)} train / {len(va)} val clips  "
          f"{df.user.nunique()} subjects ({len(va_u)} held out)  {n_cls} classes")

    mk = lambda d, t: DataLoader(NTUSkel(d, a.cache, streams=a.streams, train=t),
                                 batch_size=a.batch_size, shuffle=t, num_workers=a.workers,
                                 worker_init_fn=_wi, drop_last=t, pin_memory=True)
    dtr, dva = mk(tr, True), mk(va, False)

    cin = 17 * (10 if a.streams else 4)
    enc = SeqEncoder(cin, a.dim).to(dev)
    head = nn.Sequential(nn.LayerNorm(a.dim), nn.Dropout(0.3), nn.Linear(a.dim, n_cls)).to(dev)
    opt = torch.optim.AdamW([*enc.parameters(), *head.parameters()], lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, epochs=a.epochs,
                                                steps_per_epoch=len(dtr), pct_start=0.25)
    amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    print(f"encoder {sum(q.numel() for q in enc.parameters())/1e6:.2f}M params\n")

    for ep in range(a.epochs):
        enc.train(); head.train(); tot = 0.0
        for x, y in dtr:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast("cuda", enabled=amp):
                loss = F.cross_entropy(head(enc(x)), y, label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_([*enc.parameters(), *head.parameters()], 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item()
        enc.eval(); head.eval(); hit = n = 0
        with torch.no_grad():
            for x, y in dva:
                pred = head(enc(x.to(dev))).argmax(-1).cpu()
                hit += int((pred == y).sum()); n += len(y)
        print(f"  ep {ep+1:2d}/{a.epochs}  loss {tot/max(1,len(dtr)):.4f}  "
              f"held-out-subject acc {hit/max(1,n):.4f}")
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"trunk": enc.state_dict(), "epoch": ep + 1, "cin": cin,
                    "dim": a.dim, "streams": a.streams, "n_classes": n_cls,
                    "val_acc": hit / max(1, n), "source": "NTU RGB+D 120"}, a.out)
    print(f"\n  -> {a.out}   load with train.py --init {a.out}")


if __name__ == "__main__":
    main()
