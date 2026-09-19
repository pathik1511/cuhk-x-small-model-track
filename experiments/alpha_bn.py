"""Test-time BatchNorm adaptation. Mix source and TARGET statistics at each BN layer.

NOT the same thing as src/bn_recal.py (section 30). That re-derived buffers from TRAINING
clips, fixing stale statistics for averaged weights, and was null. This computes statistics
on the population the model is being evaluated on, addressing a different problem: the test
subjects are a distribution no training subject covers (section 9 measured the two user
blocks as separate domains).

Uses test INPUTS only, never labels. Transductive, and nothing in Rules 3.10a touches it.

    running = a * source + (1 - a) * target       a = 1.0 is the unmodified model

Statistics do not depend on `a`, so they are collected ONCE per fold and every alpha is
scored from the same pass.

  python src/alpha_bn.py --run runs/r42_r2p1d_e30 --tta 2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from cuhk_cv import subject_folds
from predict_r2p1d import tta_probs
from r2p1d import R2Plus1D34
from train_r2p1d import DIClips


class AlphaBN(nn.Module):
    def __init__(self, bn: nn.BatchNorm3d):
        super().__init__()
        self.bn, self.a, self.t = bn, 1.0, None

    def forward(self, x):
        b = self.bn
        if self.t is None or self.a >= 1.0:
            return b(x)
        m, v = self.t
        return F.batch_norm(x, self.a * b.running_mean + (1 - self.a) * m,
                            self.a * b.running_var + (1 - self.a) * v,
                            b.weight, b.bias, False, 0.0, b.eps)


def wrap(model):
    for n, c in model.named_children():
        setattr(model, n, AlphaBN(c)) if isinstance(c, nn.BatchNorm3d) else wrap(c)
    return model


def set_alpha(model, a):
    for m in model.modules():
        if isinstance(m, AlphaBN):
            m.a = a


@torch.no_grad()
def collect(model, dl, dev):
    """One pass over target clips, accumulating per-channel mean and E[x^2]."""
    acc, hooks = {}, []
    for mod in model.modules():
        if isinstance(mod, AlphaBN):
            def hook(_m, inp, _o, key=mod):
                x = inp[0].float()
                s = acc.setdefault(key, [0.0, 0.0, 0])
                s[0] = s[0] + x.sum((0, 2, 3, 4))
                s[1] = s[1] + (x * x).sum((0, 2, 3, 4))
                s[2] = s[2] + x.numel() // x.shape[1]
            hooks.append(mod.bn.register_forward_hook(hook))
    model.eval()
    for x, *_ in dl:
        with torch.amp.autocast("cuda", enabled=dev.type == "cuda"):
            model(x.to(dev))
    for h in hooks:
        h.remove()
    for m, (s, sq, n) in acc.items():
        mu = s / n
        m.t = (mu, (sq / n - mu * mu).clamp_min(0))
    return len(acc)


@torch.no_grad()
def score(model, dl, dev, tta):
    ok = n = 0
    for x, y, *_ in dl:
        with torch.amp.autocast("cuda", enabled=dev.type == "cuda"):
            p = tta_probs(model, x.to(dev), tta).argmax(1).cpu()
        ok += int((p == y).sum()); n += len(y)
    return ok / max(n, 1)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="runs/r42_r2p1d_e30")
    p.add_argument("--cache", default="data/cache_yolo")
    p.add_argument("--alphas", default="1.0,0.7,0.5,0.3,0.1,0.0")
    p.add_argument("--tta", default="2", choices=["none", "2", "4"])
    p.add_argument("--frames", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    al = [float(x) for x in a.alphas.split(",")]
    df = pd.read_csv(Path(a.cache) / "manifest.csv")
    df = df[(df.split == "train") & df.action_id.ge(0)]
    # Baselines are TTA-dependent. Measured by src/oof_probs.py on the same folds.
    # Comparing an a=1.0 run under --tta 2 against the no-TTA numbers is what tripped
    # the guard the first time; the guard was right, the reference table was wrong.
    base = {"none": [0.6020, 0.6658, 0.6593, 0.5839],
            "2":    [0.6106, 0.6740, 0.6667, 0.5990],
            "4":    [0.6120, 0.6795, 0.6696, 0.5990]}[a.tta]
    tab = {}
    for fold in range(4):
        _, va_u = list(subject_folds(seed=a.seed))[fold]
        va = df[df.user.isin(va_u)]
        dl = DataLoader(DIClips(va, a.cache, False, 0, a.frames), a.batch_size,
                        shuffle=False, num_workers=a.workers, pin_memory=True)
        # LOAD FIRST, WRAP SECOND, strict=True. Wrapping renames every BN key from
        # ...weight to ...bn.weight, so loading into an already-wrapped model with
        # strict=False silently drops all 69 layers' weights and running statistics
        # and scores at chance. The a=1.0 identity check below is what caught it.
        model = R2Plus1D34(pretrained=False)
        model.load_state_dict(torch.load(Path(a.run) / f"fold{fold}.pt",
                                         map_location="cpu")["model"], strict=True)
        model = wrap(model).to(dev)
        model.eval()
        n = collect(model, dl, dev)
        print(f"fold {fold}  users {sorted(int(u) for u in va_u)}  n={len(va)}  "
              f"{n} BN layers adapted", flush=True)
        row = {}
        for x in al:
            set_alpha(model, x)
            row[x] = score(model, dl, dev, a.tta)
            flag = ""
            if x >= 1.0:
                ok = abs(row[x] - base[fold]) < 5e-4
                flag = f"  <- identity check vs {base[fold]:.4f}: {'OK' if ok else 'FAILED'}"
                if not ok:
                    raise SystemExit(
                        f"REFUSED: a=1.0 gives {row[x]:.4f}, baseline is {base[fold]:.4f}. "
                        "The wrapper is altering the forward pass; every number below it "
                        "would be noise.")
            print(f"    a={x:<4} {row[x]:.4f}  ({row[x]-row[al[0]]:+.4f} vs a=1.0){flag}",
                  flush=True)
        tab[fold] = row
        del model; torch.cuda.empty_cache()

    ns = [701, 730, 675, 596]
    print(f"\n{'alpha':>6} {'pooled':>8} {'clips':>7}   per fold (delta vs a=1.0)")
    ref = sum(tab[f][al[0]] * ns[f] for f in range(4))
    for x in al:
        tot = sum(tab[f][x] * ns[f] for f in range(4))
        d = "  ".join(f"{f}:{tab[f][x]-tab[f][al[0]]:+.4f}" for f in range(4))
        print(f"{x:>6} {tot/sum(ns):>8.4f} {tot-ref:>+7.1f}   {d}")
    print(f"\nbaseline fold accuracies (tta={a.tta}):", base)


if __name__ == "__main__":
    main()
