"""Fine-tune VideoMAE ViT-S/16 (K400) on the CUHK-X 4-channel DI input.

Separate from train_r2p1d.py deliberately: that file is on the verified reproduction
path for the submitted entry and must not change. Same subject_folds, same manifest,
same oof_f*.csv layout, so merge_oof.py and paired.py still apply.

Differences that matter, each one a trap that was identified before the first run:

  * DIClips224 infers H,W from whichever modality is PRESENT. train_r2p1d.DIClips
    hardcodes 128 for its missing-modality zeros, which on a 224 cache reproduces the
    dc=16/ir=32 style crash in a new form.
  * ImageNet normalization, which is what VideoMAE expects. NOT the Kinetics constants
    r2p1d uses. The 4th channel takes the mean of the three.
  * Optimizer learning rates are set DIRECTLY. The original VideoMAE script multiplies
    args.lr by effective_batch/256, so asking for 3e-5 at batch 16 silently gives
    1.875e-6. Every group's actual LR is printed at startup.
  * Layerwise LR decay: block i gets backbone_lr * decay^(depth - i), so early blocks
    move least. Bias and 1-D norm parameters get no weight decay.

  python src/train_vmae.py --probe
  python src/train_vmae.py --fold 0 --out runs/v1_vmae
"""
from __future__ import annotations

import argparse
import math
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
from train_r2p1d import K_MEAN, K_STD
from vmae import VM_MEAN, VM_STD, VideoMAES


class DIClips224(Dataset):
    """Depth_Color(3) + IR(1) -> (T, 4, H, W), ImageNet-normalized."""

    def __init__(self, df, cache, train, seed=0, frames=16, size=224, arch="vmae"):
        self.df = df.reset_index(drop=True)
        self.T, self.S = frames, size
        self.cache = Path(cache)
        self.train = train
        self.rng = np.random.default_rng(seed)
        # VideoMAE expects ImageNet statistics; R(2+1)D expects the Kinetics ones it
        # was pretrained under. Using the wrong pair silently costs accuracy.
        m, sd = (VM_MEAN, VM_STD) if arch == "vmae" else (K_MEAN, K_STD)
        self.mean = torch.tensor((*m, sum(m) / 3)).view(1, 4, 1, 1)
        self.std = torch.tensor((*sd, sum(sd) / 3)).view(1, 4, 1, 1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        with np.load(self.cache / r["split"] / f"{r['clip_id']}.npz") as z:
            dc = z["Depth_Color"] if "Depth_Color" in z.files else None
            ir = z["IR"] if "IR" in z.files else None
        # Size from whatever is present, never from a constant.
        hw = next((a.shape[1:3] for a in (dc, ir) if a is not None), (self.S, self.S))

        def fit(a, c):
            if a is None:
                return np.zeros((self.T, *hw, c), np.uint8)
            if a.shape[0] != self.T:
                idx = np.linspace(0, a.shape[0] - 1, self.T).round().astype(int)
                a = a[idx]
            return a

        x = np.concatenate([fit(dc, 3), fit(ir, 1)], -1).transpose(0, 3, 1, 2)
        x = torch.from_numpy(np.ascontiguousarray(x)).float().div_(255.0)
        if x.shape[-1] != self.S:
            x = F.interpolate(x, size=(self.S, self.S), mode="bilinear",
                              align_corners=False)
        if self.train:
            if self.rng.random() < 0.5:
                x = torch.flip(x, dims=(-1,))
            x = torch.roll(x, int(self.rng.integers(-2, 3)), dims=0)
        x = x.sub_(self.mean).div_(self.std)
        return x, int(r["action_id"]), r["clip_id"], int(r["user"])


def loaders(cache, fold, seed, bs, workers, frames, size, arch="vmae"):
    df = pd.read_csv(Path(cache) / "manifest.csv")
    df = df[(df.split == "train") & df.action_id.ge(0)]
    tr_u, va_u = list(subject_folds(seed=seed))[fold]
    tr, va = df[df.user.isin(tr_u)], df[df.user.isin(va_u)]
    print(f"fold {fold}  val_users={sorted(va_u)}  train {len(tr)}  val {len(va)}")
    return (DataLoader(DIClips224(tr, cache, True, seed, frames, size, arch), bs,
                       shuffle=True, num_workers=workers, pin_memory=True, drop_last=True),
            DataLoader(DIClips224(va, cache, False, 0, frames, size, arch), bs,
                       shuffle=False, num_workers=workers, pin_memory=True))


def param_groups(model, a):
    """Layerwise decay. Printed in full so no LR is ever a silent 1.875e-6."""
    depth = model.net.config.num_hidden_layers
    groups = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("adapt"):
            lr, tag = a.lr_adapter, "adapter"
        elif "patch_embeddings" in n:
            lr, tag = a.lr_adapter, "stem"   # the widened 4-ch embedding
                                             # needs a real LR, not the
                                             # deepest layerwise-decayed one
        elif "classifier" in n:
            lr, tag = a.lr_head, "head"
        elif ".layer." in n:
            i = int(n.split(".layer.")[1].split(".")[0])
            lr, tag = a.lr_backbone * a.llrd ** (depth - i), f"block{i:02d}"
        elif "embeddings" in n:
            lr, tag = a.lr_backbone * a.llrd ** (depth + 1), "embed"
        else:
            lr, tag = a.lr_backbone, "other"
        wd = 0.0 if p.ndim == 1 or n.endswith(".bias") else a.wd
        groups.setdefault((tag, lr, wd), []).append(p)
    out = []
    for (tag, lr, wd), ps in sorted(groups.items()):
        out.append({"params": ps, "lr": lr, "weight_decay": wd})
        print(f"    {tag:9s} lr {lr:.3e}  wd {wd:.3f}  "
              f"{sum(p.numel() for p in ps):>10,} params")
    return out


def build(a, dev):
    if a.arch == "r2p1d":
        return R2Plus1D34(pretrained=True).to(dev)
    m = VideoMAES(pretrained=True, heads=a.heads, stem=a.stem).to(dev)
    m.net.gradient_checkpointing_enable()
    return m


def probe(a, dev):
    model = build(a, dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    scaler = torch.amp.GradScaler("cuda")
    for bs in (1, 2, 4, 8):
        try:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            x = torch.randn(bs, a.frames, 4, a.size, a.size, device=dev)
            y = torch.randint(0, 40, (bs,), device=dev)
            with torch.amp.autocast("cuda"):
                loss = F.cross_entropy(model(x), y)
            scaler.scale(loss).backward(); opt.zero_grad(set_to_none=True)
            print(f"  batch {bs:>2}  OK   peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB")
        except torch.cuda.OutOfMemoryError:
            print(f"  batch {bs:>2}  OOM"); break


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="data/cache_y224")
    p.add_argument("--out", default="runs/v1_vmae")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--accum", type=int, default=8, help="effective batch = bs * accum")
    p.add_argument("--frames", type=int, default=16)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--lr-backbone", type=float, default=3e-5)
    p.add_argument("--lr-head", type=float, default=3e-4)
    p.add_argument("--lr-adapter", type=float, default=1e-4)
    p.add_argument("--llrd", type=float, default=0.9)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--smooth", type=float, default=0.1)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--probe", action="store_true")
    p.add_argument("--heads", type=int, default=0,
                   help="override num_attention_heads; 0 keeps the config")
    p.add_argument("--stem", default="conv4", choices=["conv4", "adapter"])
    p.add_argument("--arch", default="vmae", choices=["vmae", "r2p1d"])
    a = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)
    if a.probe:
        print(f"device {dev}: probing VideoMAE-S @ {a.frames}x{a.size}x{a.size}x4")
        return probe(a, dev)

    tr, va = loaders(a.cache, a.fold, a.seed, a.batch_size, a.workers, a.frames,
                     a.size, a.arch)
    model = build(a, dev)
    steps = max(1, len(tr) // a.accum) * a.epochs

    if a.arch == "r2p1d":
        # EXACTLY the r42 optimizer and schedule, so the only difference from the
        # 0.6020 run is the input resolution. Do not "improve" this.
        head = {id(q) for q in model.head.parameters()}
        opt = torch.optim.AdamW([
            {"params": [q for q in model.parameters() if id(q) in head], "lr": a.lr_head},
            {"params": [q for q in model.parameters() if id(q) not in head],
             "lr": a.lr_backbone}], weight_decay=a.wd)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=[a.lr_head, a.lr_backbone], total_steps=steps, pct_start=0.1)
        print(f"  r2p1d recipe: head {a.lr_head:.1e}  backbone {a.lr_backbone:.1e}  "
              f"wd {a.wd}  OneCycle")
        set_lr = lambda s: sched.step()
    else:
        print("  optimizer groups (ACTUAL learning rates, no /256 scaling):")
        opt = torch.optim.AdamW(param_groups(model, a))
        warm = max(1, len(tr) // a.accum) * a.warmup
        base = [g["lr"] for g in opt.param_groups]

        def set_lr(s):
            f = (s / warm if s < warm else
                 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
            for g, b in zip(opt.param_groups, base):
                g["lr"] = b * f

    scaler = torch.amp.GradScaler("cuda")
    print(f"arch {a.arch}  effective batch {a.batch_size * a.accum}  steps {steps}")
    Path(a.out).mkdir(parents=True, exist_ok=True)

    step = 0
    for ep in range(a.epochs):
        t0 = time.time(); model.train(); tot = n = 0
        opt.zero_grad(set_to_none=True)
        for k, (x, y, *_) in enumerate(tr):
            with torch.amp.autocast("cuda"):
                loss = F.cross_entropy(model(x.to(dev)), y.to(dev),
                                       label_smoothing=a.smooth) / a.accum
            scaler.scale(loss).backward()
            if (k + 1) % a.accum == 0:
                set_lr(step); step += 1
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            tot += loss.item() * a.accum; n += 1

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
        print(f"  ep {ep+1:>2}/{a.epochs}  loss {tot/max(n,1):.4f}  val {acc:.4f}  "
              f"{time.time()-t0:.0f}s", flush=True)

    d.to_csv(Path(a.out) / f"oof_f{a.fold}.csv", index=False)
    torch.save({"model": model.state_dict(), "fold": a.fold, "acc": acc,
                "arch": a.arch, "size": a.size, "cache": str(a.cache)},
               Path(a.out) / f"fold{a.fold}.pt")
    print(f"\nfold {a.fold} final {acc:.4f}   r2p1d same fold: "
          f"{[0.6020, 0.6658, 0.6593, 0.5839][a.fold]:.4f}")


if __name__ == "__main__":
    main()
