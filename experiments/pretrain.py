"""pretrain.py — self-supervised pretraining of the image encoder on YOUR OWN frames.

    python src/pretrain.py --cache data/cache_hand --out runs/ssl_di.pt --epochs 30

Why this exists
---------------
The 0.711 leaderboard plateau is an R(2+1)D-34 that saw 65 million videos (IG-65M) then
Kinetics-400. You have 3,036 labelled clips. That gap is a PRETRAINING gap and no
architecture change closes it.

But the cache holds ~91,000 unlabelled training frames and only 3,036 labels. This script
turns every frame into a training sample for the image encoder — roughly a 30x increase
in learning signal for the part of the model that holds most of the parameters — using
only data you already have. No external corpus, no pretrained weights, nothing to declare.

The objective
-------------
NT-Xent (SimCLR) instance discrimination, with two deliberate choices for THIS dataset:

  1. The two views of a "sample" are TWO DIFFERENT FRAMES of the same clip. That builds
     temporal invariance for free and is the standard video-SSL positive pair.

  2. CHANNEL DROPOUT. Depth_Color (3ch) and IR (1ch) are pixel-aligned and frame-locked
     — identical embedded timestamps. Randomly zeroing one modality in each view means a
     depth-only view and an IR-only view of the same clip must land in the same place.
     That folds a cross-modal objective into the single 4-channel encoder the downstream
     model actually uses, so the weights transfer directly instead of needing two towers.

Known limitation, stated rather than hidden: instance discrimination can be partly solved
by recognising the SUBJECT (clothing, body shape), which is the nuisance factor this
project is fighting. The person crop removes the room, strong random-resized-crop removes
some of the rest, and the supervised fine-tune has labels to override what remains — but
if OOF does not move, subject shortcut is the first thing to suspect.

Only the spatial trunk is pretrained (T=1 frames); the temporal convs stay random-init and
are learned during fine-tuning.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent))
from model import FrameCNN                                   # noqa: E402


class FramePairs(Dataset):
    """One item = two augmented frames from the same clip, each [4,1,H,W]."""

    def __init__(self, manifest: pd.DataFrame, cache: Path, repeat: int = 8, seed=0):
        # Only clips that actually hold BOTH modalities. Two zero-filled "views" would be
        # a degenerate positive pair, and a differently-shaped fallback tensor breaks the
        # collate outright -- that was the first version's bug.
        m = manifest
        if "has_Depth_Color" in m.columns and "has_IR" in m.columns:
            m = m[(m.has_Depth_Color == 1) & (m.has_IR == 1)]
        self.df = m.reset_index(drop=True)
        self.dir = Path(cache)
        self.rng = np.random.default_rng(seed)
        # One pair per clip per epoch would be 3,441 samples = ~18 steps: far too few for
        # a contrastive objective. Draw `repeat` independent frame pairs per clip instead.
        self.repeat = repeat

    def __len__(self):
        return len(self.df) * self.repeat

    def _view(self, dep, ir):
        """dep [H,W,3] uint8, ir [H,W,1] uint8 -> [4,1,h,w] float32."""
        H, W = dep.shape[:2]
        # random resized crop, 0.5-1.0 of area, applied identically to both modalities
        s = float(self.rng.uniform(0.5, 1.0)) ** 0.5
        ch, cw = max(8, int(H * s)), max(8, int(W * s))
        y = int(self.rng.integers(0, H - ch + 1)); x = int(self.rng.integers(0, W - cw + 1))
        d = dep[y:y + ch, x:x + cw]; i = ir[y:y + ch, x:x + cw]
        if self.rng.random() < 0.5:
            d, i = d[:, ::-1], i[:, ::-1]
        d = torch.from_numpy(np.ascontiguousarray(d)).permute(2, 0, 1).float() / 255.0
        i = torch.from_numpy(np.ascontiguousarray(i)).permute(2, 0, 1).float() / 255.0
        d = F.interpolate(d[None], size=(H, W), mode="bilinear", align_corners=False)[0]
        i = F.interpolate(i[None], size=(H, W), mode="bilinear", align_corners=False)[0]
        x4 = torch.cat([d, i], 0)
        x4 = (x4 - x4.mean()) / (x4.std() + 1e-5)
        # channel dropout: force depth-only and IR-only views into the same embedding
        r = self.rng.random()
        if r < 0.25:
            x4[:3] = 0.0
        elif r < 0.50:
            x4[3:] = 0.0
        x4 = x4 + torch.from_numpy(self.rng.normal(0, 0.05, x4.shape).astype(np.float32))
        return x4[:, None]                                    # [4,1,H,W]

    def __getitem__(self, idx):
        r = self.df.iloc[idx % len(self.df)]
        with np.load(self.dir / r["split"] / f"{r['clip_id']}.npz") as z:
            dep, ir = z["Depth_Color"], z["IR"]
        T = dep.shape[0]
        a, b = self.rng.integers(0, T), self.rng.integers(0, T)
        return self._view(dep[a], ir[a]), self._view(dep[b], ir[b])


def _worker_init(_wid: int) -> None:
    """Same trap as dataset.py: every worker forks a copy holding the SAME rng seed, so
    without this they all draw an identical augmentation stream. Module-level so Windows
    spawn can pickle it."""
    info = torch.utils.data.get_worker_info()
    if info is not None:
        info.dataset.rng = np.random.default_rng(torch.initial_seed() % 2**31)


class Tower(nn.Module):
    def __init__(self, width=32, dim=256, proj=128):
        super().__init__()
        self.enc = FrameCNN(4, width, dim=dim, temporal=0)     # spatial trunk only
        self.head = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, proj))

    def forward(self, x):
        return F.normalize(self.head(self.enc(x)), dim=-1)


def nt_xent(z1, z2, t=0.1):
    """Standard SimCLR loss. 2N views, the other view of the same clip is the positive."""
    z = torch.cat([z1, z2], 0)
    n = z1.shape[0]
    sim = (z @ z.T) / t
    sim.fill_diagonal_(-1e4)
    tgt = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, tgt)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="data/cache_hand")
    p.add_argument("--out", default="runs/ssl_di.pt")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=192)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--temp", type=float, default=0.1)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--repeat", type=int, default=8,
                   help="frame pairs drawn per clip per epoch")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--train-only", action="store_true",
                   help="exclude test clips (default includes them: unlabelled "
                        "representation learning, NOT pseudo-labelling -- but if you want "
                        "to be maximally conservative about the rules, pass this)")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    torch.manual_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv(Path(a.cache) / "manifest.csv")
    if a.train_only:
        df = df[df.split == "train"]
    ds = FramePairs(df, a.cache, repeat=a.repeat, seed=a.seed)
    drop = len(df) - len(ds.df)
    if drop:
        print(f"  dropped {drop} clips lacking Depth_Color or IR")
    dl = DataLoader(ds, a.batch_size, shuffle=True, num_workers=a.workers,
                    pin_memory=True, drop_last=True, persistent_workers=a.workers > 0,
                    worker_init_fn=_worker_init if a.workers else None)
    model = Tower(a.width, a.dim).to(dev)
    n = sum(p_.numel() for p_ in model.enc.parameters())
    print(f"device {dev}  clips {len(ds.df):,}  cached frames {len(ds.df)*16:,}  "
          f"samples/epoch {len(ds):,}  trunk {n/1e6:.2f}M params  "
          f"{len(dl)} steps/epoch  {'train only' if a.train_only else 'train+test frames'}")

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, epochs=a.epochs,
                                                steps_per_epoch=len(dl), pct_start=0.1)
    amp = dev.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    for ep in range(a.epochs):
        model.train(); tot, acc, t0 = 0.0, 0.0, time.time()
        for v1, v2 in dl:
            v1, v2 = v1.to(dev, non_blocking=True), v2.to(dev, non_blocking=True)
            with torch.autocast("cuda", enabled=amp):
                z1, z2 = model(v1), model(v2)
                loss = nt_xent(z1, z2, a.temp)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item()
            with torch.no_grad():                              # retrieval acc, the metric
                s = z1 @ z2.T
                acc += (s.argmax(1) == torch.arange(len(s), device=dev)).float().mean().item()
        print(f"  ep {ep+1:>2}/{a.epochs}  loss {tot/len(dl):.4f}  "
              f"top1-retrieval {acc/len(dl):.3f}  {time.time()-t0:.0f}s", flush=True)
        torch.save({"trunk": model.enc.state_dict(), "width": a.width, "dim": a.dim,
                    "epoch": ep + 1}, a.out)
    print(f"\nsaved -> {a.out}   load it with:  train.py --init {a.out}")


if __name__ == "__main__":
    main()
