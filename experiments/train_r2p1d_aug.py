"""r42 recipe + stronger augmentation. Separate file so Windows can keep training
fold 1 from an untouched src/train_r2p1d.py while this runs elsewhere.

Fine-tune R(2+1)D-34 (IG65M+Kinetics) on the CUHK-X 4-channel DI input.

Separate from train.py on purpose: the working 12-model pipeline must not change one
day before the freeze. Reuses the same cache, the same manifest and the SAME
subject_folds, so its oof.csv drops straight into merge_oof.py and paired.py.

Two deliberate differences from dataset.py:
  1. KINETICS normalization, not per-clip standardization. dataset.py does
     x/255 -> subtract mean -> divide std per clip, which is right for a scratch CNN
     and wrong for filters pretrained on Kinetics statistics. The 4th channel (IR)
     uses the mean of the three RGB constants, matching the public 0.716 notebook.
  2. DI only. No HAND, no Skeleton. That is the notebook's input and the point here is
     to test the BACKBONE, not to re-litigate fusion.

  python src/train_r2p1d.py --probe                 # find the largest batch that fits
  python src/train_r2p1d.py --fold 0 --out runs/r41_r2p1d
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
from cuhk_cv import subject_folds
from r2p1d import R2Plus1D34

K_MEAN = (0.43216, 0.394666, 0.37645)
K_STD = (0.22803, 0.22145, 0.216989)


class DIClips(Dataset):
    """Depth_Color(3) + IR(1) -> (T, 4, H, W), Kinetics-normalized."""

    def __init__(self, df: pd.DataFrame, cache: Path, train: bool, seed: int = 0,
                 frames: int = 16, aug: int = 0, roll: int = 2):
        self.aug, self.roll = aug, roll
        self.df = df.reset_index(drop=True)
        self.T = frames
        self.cache = Path(cache)
        self.train = train
        self.rng = np.random.default_rng(seed)
        m = (*K_MEAN, sum(K_MEAN) / 3)
        s = (*K_STD, sum(K_STD) / 3)
        self.mean = torch.tensor(m).view(1, 4, 1, 1)
        self.std = torch.tensor(s).view(1, 4, 1, 1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        with np.load(self.cache / r["split"] / f"{r['clip_id']}.npz") as z:
            dc = z["Depth_Color"] if "Depth_Color" in z.files else None
            ir = z["IR"] if "IR" in z.files else None
        T, H, W = self.T, 128, 128

        def fit(a, c):
            # Per stream, BEFORE the concatenate. A missing modality was sized to self.T
            # while the present one kept the cache length, so a Thermal-only clip read
            # from a 32-frame cache at --frames 16 gave dc=16 vs ir=32 and the
            # concatenate raised. Resampling each stream to T removes the mismatch and
            # still lets one 32-frame cache serve the 16-frame control.
            if a is None:
                return np.zeros((T, H, W, c), np.uint8)
            if a.shape[0] == T:
                return a
            idx = np.linspace(0, a.shape[0] - 1, T).round().astype(int)
            return a[idx]

        dc, ir = fit(dc, 3), fit(ir, 1)
        x = np.concatenate([dc, ir], -1).transpose(0, 3, 1, 2)      # (T,4,H,W)
        x = torch.from_numpy(np.ascontiguousarray(x)).float().div_(255.0)
        if self.train:
            if self.rng.random() < 0.5:
                x = torch.flip(x, dims=(-1,))
            x = torch.roll(x, int(self.rng.integers(-self.roll, self.roll + 1)), dims=0)
            if self.aug:
                # One variable against r42: the same cache, recipe and folds, more
                # augmentation. r42 fold 0 ended at train loss 0.7498 with val flat from
                # epoch 18, so the backbone is memorizing 2,306 clips.
                sc = float(self.rng.uniform(0.75, 1.0))              # random resized crop
                h = int(128 * sc)
                i0 = int(self.rng.integers(0, 128 - h + 1))
                j0 = int(self.rng.integers(0, 128 - h + 1))
                x = F.interpolate(x[:, :, i0:i0 + h, j0:j0 + h], size=(128, 128),
                                  mode="bilinear", align_corners=False)
                x = x.mul_(float(self.rng.uniform(0.8, 1.2))).clamp_(0, 1)   # brightness
                if self.rng.random() < 0.25:                         # cutout, box fixed
                    e = int(self.rng.integers(16, 41))               # across all frames
                    a0 = int(self.rng.integers(0, 128 - e))
                    b0 = int(self.rng.integers(0, 128 - e))
                    x[:, :, a0:a0 + e, b0:b0 + e] = 0
        x = x.sub_(self.mean).div_(self.std)
        return x, int(r["action_id"]), r["clip_id"], int(r["user"])


def loaders(cache, fold, seed, bs, workers, frames=16, aug=0, roll=2):
    df = pd.read_csv(Path(cache) / "manifest.csv")
    df = df[(df.split == "train") & df.action_id.ge(0)]
    tr_u, va_u = list(subject_folds(seed=seed))[fold]
    tr, va = df[df.user.isin(tr_u)], df[df.user.isin(va_u)]
    print(f"fold {fold}  val_users={sorted(va_u)}  train {len(tr)}  val {len(va)}")
    return (DataLoader(DIClips(tr, cache, True, seed, frames, aug, roll), bs, shuffle=True,
                       num_workers=workers, pin_memory=True, drop_last=True),
            DataLoader(DIClips(va, cache, False, 0, frames), bs, shuffle=False,
                       num_workers=workers, pin_memory=True))


def probe(a, dev):
    """Largest batch that survives a real forward+backward+step, measured not guessed."""
    m = R2Plus1D34(pretrained=False).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    for bs in (1, 2, 4, 6, 8, 12, 16):
        try:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            x = torch.randn(bs, a.frames, 4, 128, 128, device=dev)
            y = torch.randint(0, 40, (bs,), device=dev)
            with torch.amp.autocast("cuda"):
                loss = F.cross_entropy(m(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            opt.zero_grad(set_to_none=True)
            print(f"  batch {bs:>2}: OK   peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
        except torch.OutOfMemoryError:
            print(f"  batch {bs:>2}: OOM  <- use the largest OK above")
            break


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="data/cache_yolo")
    p.add_argument("--out", default="runs/r41_r2p1d")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--accum", type=int, default=4, help="effective batch = bs * accum")
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--lr-backbone", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--smooth", type=float, default=0.1)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--frames", type=int, default=16,
                   help="clip length. The checkpoint is r2plus1d_34_clip32, PRETRAINED ON 32 "
                        "FRAMES. 16 halves the temporal window its filters were built for. "
                        "Needs a 32-frame cache (data/cache_fixC) and ~2x the memory.")
    p.add_argument("--freeze-bn", type=int, default=1,
                   help="1 = keep BatchNorm in eval mode during fine-tuning. At batch 4 the "
                        "per-batch statistics are computed from 4 samples and OVERWRITE the "
                        "pretrained Kinetics running stats. Affine weight/bias still train.")
    p.add_argument("--ema", type=float, default=0.999,
                   help="0 disables. Weight averaging; the scratch recipe found EMA 0.99 "
                        "worth keeping and this model is 10x larger on the same 2,335 clips.")
    p.add_argument("--probe", action="store_true")
    p.add_argument("--aug", type=int, default=1,
                   help="1 = resized crop + brightness + cutout on top of hflip/roll")
    p.add_argument("--roll", type=int, default=4, help="temporal roll range, +/- frames")
    a = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)
    if a.probe:
        print(f"device {dev}: probing largest batch for R(2+1)D-34 @ "
              f"{a.frames}x128x128x4")
        return probe(a, dev)

    tr, va = loaders(a.cache, a.fold, a.seed, a.batch_size, a.workers, a.frames,
                     a.aug, a.roll)
    print(f"aug {a.aug}  roll +/-{a.roll}")
    model = R2Plus1D34(pretrained=True).to(dev)
    head = {id(q) for q in model.head.parameters()}
    opt = torch.optim.AdamW([
        {"params": [q for q in model.parameters() if id(q) in head], "lr": a.lr_head},
        {"params": [q for q in model.parameters() if id(q) not in head], "lr": a.lr_backbone},
    ], weight_decay=a.wd)
    steps = max(1, len(tr) // a.accum) * a.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[a.lr_head, a.lr_backbone], total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda")
    print(f"effective batch {a.batch_size * a.accum}  optimizer steps {steps}  "
          f"frames {a.frames}")

    n_bn = sum(1 for m in model.modules() if isinstance(m, nn.BatchNorm3d))
    print(f"BatchNorm3d layers: {n_bn}   frozen: {bool(a.freeze_bn)}   EMA: {a.ema}")

    ema = {k: v.detach().clone().float() for k, v in model.state_dict().items()} if a.ema else None

    Path(a.out).mkdir(parents=True, exist_ok=True)
    for ep in range(a.epochs):
        model.train()
        if a.freeze_bn:
            for m in model.modules():
                if isinstance(m, nn.BatchNorm3d):
                    m.eval()          # use pretrained running stats, stop updating them
        tot = n = 0; t0 = time.time()
        opt.zero_grad(set_to_none=True)
        for i, (x, y, _, _) in enumerate(tr):
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.amp.autocast("cuda"):
                loss = F.cross_entropy(model(x), y, label_smoothing=a.smooth) / a.accum
            scaler.scale(loss).backward()
            if (i + 1) % a.accum == 0:
                scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                if sched.last_epoch < steps - 1:
                    sched.step()
                if ema is not None:
                    with torch.no_grad():
                        for k, v in model.state_dict().items():
                            if v.is_floating_point():
                                ema[k].mul_(a.ema).add_(v.float(), alpha=1 - a.ema)
                            else:
                                ema[k] = v.detach().clone().float()
            tot += loss.item() * a.accum; n += 1

        if ema is not None:          # evaluate the averaged weights, restore after
            backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
            model.load_state_dict({k: v.to(backup[k].dtype) for k, v in ema.items()})
        model.eval(); rows = []
        with torch.inference_mode():
            for x, y, cid, usr in va:
                with torch.amp.autocast("cuda"):
                    o = model(x.to(dev))
                rows += [{"clip_id": c, "user": int(u), "fold": a.fold,
                          "y_true": int(t), "y_pred": int(q)}
                         for c, u, t, q in zip(cid, usr, y, o.argmax(1).cpu())]
        d = pd.DataFrame(rows)
        acc = float((d.y_true == d.y_pred).mean())
        if ema is not None:
            model.load_state_dict(backup)
        print(f"  ep {ep+1:>2}/{a.epochs}  loss {tot/max(n,1):.4f}  val {acc:.4f}  "
              f"{time.time()-t0:.0f}s", flush=True)

    d.to_csv(Path(a.out) / f"oof_f{a.fold}.csv", index=False)
    if ema is not None:
        model.load_state_dict({k: v.to(backup[k].dtype) for k, v in ema.items()})
    torch.save({"model": model.state_dict(), "fold": a.fold, "acc": acc,
                "arch": "r2plus1d_34_ig65m_kinetics", "cache": str(a.cache)},
               Path(a.out) / f"fold{a.fold}.pt")
    print(f"\nfold {a.fold} final {acc:.4f}   "
          f"baselines: motion-crop CNN 0.5036, YOLO-crop CNN 0.5193")


if __name__ == "__main__":
    main()
