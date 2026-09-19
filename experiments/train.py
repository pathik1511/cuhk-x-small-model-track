"""
train.py — cross-subject training + submission for CUHK-X Small Model Track.

    python train.py --cache data/cache --fold 0 --modalities Depth_Color,Skeleton
    python train.py --cache data/cache --all-folds --epochs 30
    python train.py --cache data/cache --predict            # -> submission.csv

Notes that matter:
  * Validation is by SUBJECT, never by clip. Folds come from cuhk_cv.subject_folds,
    which holds out 2 users from each recording block to mirror the leaderboard's
    (10, 11, 25, 26). Clip-level shuffling here would inflate accuracy by ~30 points
    and teach you nothing.
  * Report `fold_std` alongside the mean. With 4 held-out subjects the leaderboard's
    own noise floor is several points; a gain smaller than the fold spread is noise.
  * `--grl` enables the subject-adversarial head: the trunk is pushed to be
    UN-predictive of who is performing. Leave it off for the first honest baseline,
    then turn it on and check whether OOF improves by more than fold_std.
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

sys.path.insert(0, str(Path(__file__).parent))
from cuhk_cv import lb_noise, oof_report, subject_folds     # noqa: E402
from dataset import CUHKClips, make_loaders                 # noqa: E402
from model import FusionNet, budget                         # noqa: E402


def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_dev(b: dict, dev, mods: list[str]) -> dict:
    return {**{m: b[m].to(dev, non_blocking=True) for m in mods},
            "mask": b["mask"].to(dev), "y": b["y"].to(dev)}


@torch.no_grad()
def evaluate(model, loader, dev, mods) -> tuple[float, pd.DataFrame]:
    model.eval()
    rows = []
    for b in loader:
        logits = model(to_dev(b, dev, mods))["logits"]
        p = logits.softmax(-1).cpu().numpy()
        for i, cid in enumerate(b["clip_id"]):
            rows.append({"clip_id": cid, "y_true": int(b["y"][i]),
                         "y_pred": int(p[i].argmax()), **{f"p{j}": p[i, j] for j in range(p.shape[1])}})
    df = pd.DataFrame(rows)
    return float((df.y_true == df.y_pred).mean()), df


def supcon(p: torch.Tensor, y: torch.Tensor, u: torch.Tensor, temp: float = 0.1):
    """Supervised contrastive with SAME-ACTION / DIFFERENT-PERFORMER positives only.

    This is the SkeletonX SADP construction. Ordinary instance discrimination pulls
    together views that share identity, which is why the self-supervised run regressed
    5.4 points -- it learned WHO, not WHAT. Excluding same-subject positives inverts
    that: the only way to satisfy the loss is a representation of the action that is
    invariant to the performer, which is the dominant error term on this split.
    """
    sim = (p @ p.T) / temp
    n = p.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=p.device)
    pos = (y[:, None] == y[None, :]) & (u[:, None] != u[None, :]) & ~eye
    if pos.sum() == 0:
        return p.sum() * 0.0                      # no valid pair in this batch
    sim = sim.masked_fill(eye, -1e4)
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    k = pos.sum(1)
    ok = k > 0
    return -((logp * pos).sum(1)[ok] / k[ok]).mean()


class PKSampler(torch.utils.data.Sampler):
    """P classes x K samples per batch. A batch of 16 drawn at random over 40 classes
    almost never contains a same-action-different-performer pair, so the contrastive
    term would be a no-op. PK sampling guarantees them."""

    def __init__(self, labels, P: int, K: int, batches: int, seed: int = 0):
        self.by = {}
        for i, c in enumerate(labels):
            self.by.setdefault(int(c), []).append(i)
        self.by = {c: v for c, v in self.by.items() if len(v) >= 2}
        self.P, self.K, self.batches = P, K, batches
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.batches * self.P * self.K

    def __iter__(self):
        cls = list(self.by)
        for _ in range(self.batches):
            for c in self.rng.choice(cls, size=min(self.P, len(cls)), replace=False):
                idx = self.by[int(c)]
                take = self.rng.choice(idx, size=self.K, replace=len(idx) < self.K)
                yield from (int(i) for i in take)


def q8(state: dict) -> dict:
    """Per-output-channel symmetric int8 for every >=2-D tensor, fp16 for the rest.
    ~4x smaller than fp32 with no measurable accuracy cost -- this is exactly how the
    public leaderboard baseline fits a 63M-param model under the 100 MB cap, and it is
    what lets you ensemble 20+ models instead of 8."""
    out = {}
    for k, v in state.items():
        if v.dtype.is_floating_point and v.ndim >= 2:
            w = v.detach().float().reshape(v.shape[0], -1)
            sc = w.abs().amax(1).clamp(min=1e-12) / 127.0
            out[k] = {"q": (w / sc[:, None]).round().clamp(-127, 127).to(torch.int8),
                      "s": sc.half(), "shape": tuple(v.shape)}
        elif v.dtype.is_floating_point:
            out[k] = v.detach().half()
        else:
            out[k] = v.detach()
    return out


def dq8(state: dict) -> dict:
    """Inverse of q8. Passes through anything that was never quantized."""
    out = {}
    for k, v in state.items():
        if isinstance(v, dict) and "q" in v:
            out[k] = (v["q"].float() * v["s"].float()[:, None]).reshape(v["shape"])
        elif torch.is_tensor(v) and v.dtype == torch.float16:
            out[k] = v.float()
        else:
            out[k] = v
    return out


class EMA:
    """Exponential moving average of the weights. The public LB baseline scores its EMA
    weights, not the raw ones, and SWA/SWAD is one of the few domain-generalization
    methods that reliably beats plain ERM -- both are the same flat-minimum mechanism."""

    def __init__(self, model, decay=0.99):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)
            else:
                s.copy_(v.detach())

    def copy_to(self, model):
        model.load_state_dict({k: v.to(dtype=p.dtype) for (k, p), v
                               in zip(model.state_dict().items(), self.shadow.values())})


def load_teacher(path: str, fold: int = 0):
    """clip_id -> log teacher probabilities. Log, so a temperature can re-soften them:
    softmax(log p / T) is the standard KD target when only probabilities were saved."""
    if not path:
        return {}
    path = path.format(fold=fold)
    if not Path(path).exists():
        raise SystemExit(f"teacher file not found: {path}")
    d = pd.read_csv(path).drop_duplicates("clip_id").set_index("clip_id")
    cols = [f"t{i}" for i in range(40)]
    lp = np.log(np.clip(d[cols].values.astype(np.float32), 1e-8, None))
    print(f"  teacher: {len(d)} clips from {path}")
    return dict(zip(d.index, lp))


def run_fold(a, fold: int, dev) -> pd.DataFrame:
    mods = a.modalities.split(",")
    vu = [int(x) for x in a.val_users.split(",")] if a.val_users else None
    tr, va = make_loaders(a.cache, fold=fold, seed=a.seed, modalities=mods,
                          batch_size=a.batch_size, workers=a.workers,
                          frames=a.frames, mod_dropout=a.mod_dropout,
                          balanced=not a.no_balance, streams=a.streams, pk=a.pk,
                          val_users=vu, full_fit=a.full_fit)
    tr_u, va_u = list(subject_folds(seed=a.seed))[fold]
    if a.full_fit:
        tr_u = sorted(pd.read_csv(Path(a.cache) / "manifest.csv")
                      .query("split == 'train' and action_id >= 0").user.unique().tolist())
        va_u = ["NONE - full refit, the printed acc is TRAIN data, not validation"]
    elif vu:                            # leave-two-folds-out teacher: 8 users train
        va_u = sorted(vu)
        tr_u = sorted((set(tr_u) | set(list(subject_folds(seed=a.seed))[fold][1])) - set(va_u))
    n_subj = max(tr_u) + 1 if a.grl else 0
    model = FusionNet(mods, n_classes=40, dim=a.dim, width=a.width, n_subjects=n_subj,
                      fusion=a.fusion, temporal=a.temporal, drop_path=a.drop_path,
                      seq_dims={"Skeleton": 17 * 10} if a.streams else None).to(dev)
    if a.init:                                    # load self-supervised trunk weights
        ck = torch.load(a.init, map_location="cpu", weights_only=False)
        hit = []
        tgt = ck.get("target", "Skeleton" if ck.get("source", "").startswith("NTU")
                     else None)
        for m in mods:
            if tgt and m == tgt and m in model.enc:      # NTU-pretrained skeleton trunk
                sd = model.enc[m].state_dict()
                take = {k: v for k, v in ck["trunk"].items()
                        if k in sd and sd[k].shape == v.shape}
                if not take:
                    sys.exit(f"--init: no tensor of {a.init} fits enc[{m}]. "
                             f"Check --streams matches how the trunk was pretrained "
                             f"(cin {ck.get('cin','?')} vs {sd['net.0.weight'].shape[1]}).")
                sd.update(take); model.enc[m].load_state_dict(sd)
                hit.append(f"{m}:{len(take)}/{len(sd)} tensors")
            elif tgt is None and m in ("DI", "HAND") and m in model.enc:
                sd = model.enc[m].state_dict()
                take = {k: v for k, v in ck["trunk"].items()
                        if k in sd and sd[k].shape == v.shape and not k.startswith("temporal")}
                sd.update(take); model.enc[m].load_state_dict(sd)
                hit.append(f"{m}:{len(take)}/{len(sd)} tensors")
        print(f"  init from {a.init} (epoch {ck.get('epoch','?')}) -> {', '.join(hit) or 'NOTHING MATCHED'}")
    n, mb = budget(model)
    print(f"\nfold {fold}  val_users={va_u}  {n/1e6:.2f}M params ({mb:.1f} MB)"
          f"  {len(tr)} train batches")
    if mb >= 100:
        sys.exit(f"model is {mb:.1f} MB — over the 100 MB competition limit")

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, epochs=a.epochs,
                                                steps_per_epoch=len(tr), pct_start=0.25)
    # AMP on CUDA only — MPS autocast is unreliable and CPU gains nothing.
    amp = (dev.type == "cuda") and not a.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    if amp:
        print("  mixed precision: ON")
    u2i = {u: i for i, u in enumerate(tr_u)}
    tch = load_teacher(a.teacher, fold) if a.kd > 0 else {}
    ema = EMA(model, a.ema) if a.ema > 0 else None
    best, best_df = -1.0, None   # -1 so epoch 1 always populates best_df (0.0 acc crashed)
    for ep in range(a.epochs):
        model.train()
        lam = a.grl * min(1.0, ep / max(1, a.epochs * 0.4))
        tot, t0 = 0.0, time.time()
        for b in tr:
            d = to_dev(b, dev, mods)
            with torch.autocast("cuda", enabled=amp):
                out = model(d, lam=lam)
                loss = F.cross_entropy(out["logits"], d["y"], label_smoothing=a.smooth)
                if a.kd > 0 and tch:
                    ok = [i for i, c in enumerate(b["clip_id"]) if c in tch]
                    if ok:
                        tl = torch.as_tensor(np.stack([tch[b["clip_id"][i]] for i in ok]),
                                             device=dev)
                        tt = F.softmax(tl / a.kd_temp, -1)
                        lp = F.log_softmax(out["logits"][ok].float() / a.kd_temp, -1)
                        kd = -(tt * lp).sum(-1).mean()
                        loss = (1 - a.kd) * loss + a.kd * (a.kd_temp ** 2) * kd
                if a.supcon > 0:
                    uu = torch.as_tensor([int(x) for x in b["user"]], device=dev)
                    loss = loss + a.supcon * supcon(out["p"], d["y"], uu, a.supcon_temp)
                if "subject" in out:
                    s = torch.tensor([u2i.get(int(x), 0) for x in b.get("user", d["y"] * 0)],
                                     device=dev)
                    loss = loss + 0.3 * F.cross_entropy(out["subject"], s)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            if ema is not None:
                ema.update(model)
            tot += loss.detach().item()
        if ema is not None:                       # score and save the EMA weights
            raw = {k: v.detach().clone() for k, v in model.state_dict().items()}
            ema.copy_to(model)
        acc, df = evaluate(model, va, dev, mods)
        flag = ""
        last = (ep == a.epochs - 1)
        if a.last_epoch and last:                 # unconditional last epoch: no selection
            best, best_df, flag = acc, df, "  <-saved"
            torch.save({"model": (q8 if a.quant == "int8" else lambda x: x)(model.state_dict()),
                        "mods": mods, "fold": fold, "quant": a.quant, "cache": str(a.cache),
                        "dim": a.dim, "width": a.width, "acc": acc,
                        "fusion": a.fusion, "temporal": a.temporal,
                        "streams": a.streams}, Path(a.out) / f"fold{fold}.pt")
        elif (not a.last_epoch) and acc > best:
            best, best_df, flag = acc, df, "  *"
            torch.save({"model": model.state_dict(), "mods": mods, "fold": fold,
                         "dim": a.dim, "width": a.width, "acc": acc,
                        "fusion": a.fusion, "temporal": a.temporal,
                        "streams": a.streams}, Path(a.out) / f"fold{fold}.pt")
        print(f"  ep {ep+1:>2}/{a.epochs}  loss {tot/len(tr):.4f}  val {acc:.4f}"
              f"  {time.time()-t0:.0f}s{flag}", flush=True)
        if ema is not None:
            model.load_state_dict(raw)            # keep training the raw weights

    print(f"  fold {fold}  best-epoch {best:.4f}   final-epoch {acc:.4f}"
          f"   (report the FINAL number in ablations: best-epoch selects on the same"
          f" fold you are measuring)")
    best_df["fold"], best_df["user"] = fold, -1
    m = pd.read_csv(Path(a.cache) / "manifest.csv").set_index("clip_id").user
    best_df["user"] = best_df.clip_id.map(m).astype(int)
    print(f"  fold {fold} best {best:.4f}")
    return best_df


FLIP_PAIRS = [(1, 4), (2, 5), (3, 6), (14, 11), (15, 12), (16, 13)]
IMU_FLIP = [1, 0, 2, 4, 3]


def roll_batch(d: dict, mods: list[str], k: int) -> dict:
    """Temporal jitter by k frames. Images are [B,C,T,H,W] (T=2), sequences [B,T,...]."""
    o = dict(d)
    for m in mods:
        x = d[m]
        o[m] = torch.roll(x, k, dims=2 if x.dim() == 5 else 1)
    return o


def flip_batch(d: dict, mods: list[str]) -> dict:
    """Mirror a batch the same way dataset.py mirrors it during training."""
    o = dict(d)
    for m in mods:
        x = d[m]
        if m in ("Depth_Color", "IR", "Thermal", "DI"):
            o[m] = torch.flip(x, dims=[-1])
        elif m == "Skeleton":
            x = x.clone()
            x[..., 0] *= -1
            for i, j in FLIP_PAIRS:
                x[:, :, [i, j]] = x[:, :, [j, i]]
            o[m] = x
        elif m == "IMU":
            o[m] = x[:, :, IMU_FLIP]
    return o


def load_ckpt(model, sd: dict, name: str = "") -> None:
    """Load a checkpoint, tolerating ONLY the SADP projection head.

    Runs predating SADP (r16, r17) have no `proj.*` tensors. `proj` feeds out["p"],
    which only the contrastive loss reads, so a random one cannot change a logit.
    Everything else missing or unexpected is a real architecture mismatch and is fatal:
    strict=False everywhere would silently leave half a model at its initialisation.
    """
    r = model.load_state_dict(sd, strict=False)
    bad = [k for k in r.missing_keys if not k.startswith("proj.")] + list(r.unexpected_keys)
    if bad:
        sys.exit(f"{name}: checkpoint does not fit the model: {bad[:8]}")
    if r.missing_keys:
        print(f"  {name}: pre-SADP checkpoint, {len(r.missing_keys)} proj tensors "
              f"left at init (inference-irrelevant)")


@torch.no_grad()
def predict(a, dev) -> None:
    cache = Path(a.cache)
    df = pd.read_csv(cache / "manifest.csv")
    test = df[df.split == "test"].copy()
    ckpts = sorted(Path(a.out).glob("fold*.pt"))
    if not ckpts:
        sys.exit(f"no checkpoints in {a.out} — train first")
    # Accumulate by CLIP ID, never by row position. Two caches can list their test
    # clips in different orders, and summing positionally silently scrambles every
    # prediction -- it produced an ensemble that scored BELOW its worst member.
    acc: dict[str, np.ndarray] = {}
    for c in ckpts:
        s = torch.load(c, map_location=dev, weights_only=False)
        s["model"] = dq8(s["model"]) if s.get("quant") == "int8" else s["model"]
        # Checkpoints trained before the skeleton-3D fix expect 17*3 input channels.
        # Read the width the checkpoint actually wants and match the dataset to it.
        w = s["model"].get("enc.Skeleton.net.0.weight")
        skel_ch = int(w.shape[1] // 17) if w is not None else 4
        streams = s.get("streams", False) or skel_ch == 10
        model = FusionNet(s["mods"], 40, dim=s["dim"], width=s["width"],
                          fusion=s.get("fusion", "mean"),
                          temporal=s.get("temporal", 0),
                          seq_dims={"Skeleton": 17 * skel_ch}).to(dev).eval()
        load_ckpt(model, s["model"], c.name)
        if skel_ch != 4:
            print(f"  {c.name}: skeleton input {skel_ch} channels"
                  f"{' (joint+bone+motion streams)' if skel_ch == 10 else ' (legacy pre-fix)'}")
        ck_cache = Path(s.get("cache") or cache)
        if ck_cache != cache:
            print(f"    (reads its own cache: {ck_cache})")
        test_c = pd.read_csv(ck_cache / "manifest.csv")
        test_c = test_c[test_c.split == "test"].copy()
        ds = CUHKClips(test_c, ck_cache, s["mods"], train=False, mod_dropout=0.0,
                       frames=a.frames, skel_ch=4 if streams else skel_ch,
                       streams=streams)
        dl = torch.utils.data.DataLoader(ds, a.batch_size, shuffle=False, num_workers=a.workers)
        seen = 0
        for b in dl:
            d0 = to_dev(b, dev, s["mods"])
            # 4-pass TTA: identity + hflip + temporal jitter +-1 frame. All four are
            # transforms the model already saw during training (hflip and np.roll are in
            # dataset.py), so none of them shifts the input distribution at inference.
            views = [d0]
            if not a.no_tta:
                views += [flip_batch(d0, s["mods"]),
                          roll_batch(d0, s["mods"], 1),
                          roll_batch(d0, s["mods"], -1)]
            pr = sum(model(v)["logits"].softmax(-1) for v in views) / len(views)
            pr = pr.cpu().numpy()
            for i, cid in enumerate(b["clip_id"]):
                acc[cid] = acc[cid] + pr[i] if cid in acc else pr[i].copy()
                seen += 1
        print(f"  {c.name}  acc@train {s['acc']:.4f}  ({seen} clips)")
    ids = sorted(acc)
    probs = np.stack([acc[i] for i in ids])
    print(f"\n{len(ids)} unique test clips, {len(ckpts)} models")
    if getattr(a, "save_probs", ""):       # for cross-family blending, see src/blend.py
        pd.DataFrame(probs / probs.sum(1, keepdims=True),
                     columns=[f"p{i}" for i in range(probs.shape[1])]
                     ).assign(clip_id=ids).to_csv(a.save_probs, index=False)
        print(f"probabilities -> {a.save_probs}")
    sub = pd.DataFrame({"path": [f"small_model_track_test/{i}/" for i in ids],
                        "prediction": probs.argmax(1)})
    ref = cache.parent / "raw/Small-Model-Track/Testing/test_file/test.csv"
    if ref.exists():                                   # keep the official row order
        sub = pd.read_csv(ref)[["path"]].merge(sub, on="path", how="left")
        miss = int(sub.prediction.isna().sum())
        if miss:
            print(f"  WARNING: {miss} test paths unmatched — filling with class 0")
        sub["prediction"] = sub.prediction.fillna(0).astype(int)
    sub.to_csv(a.submission, index=False)
    print(f"\nsubmission -> {a.submission}  ({len(sub)} rows, {len(ckpts)} models ensembled)")
    print(sub.head().to_string(index=False))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default="data/cache")
    p.add_argument("--out", default="runs")
    p.add_argument("--modalities", default="Depth_Color,IR,Thermal,Skeleton,IMU,Radar")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--all-folds", action="store_true")
    p.add_argument("--predict", action="store_true")
    p.add_argument("--submission", default="submission.csv")
    p.add_argument("--save-probs", default="", help="also write per-clip probabilities")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--smooth", type=float, default=0.1)
    p.add_argument("--mod-dropout", type=float, default=0.25)
    p.add_argument("--no-balance", action="store_true",
                   help="train on the natural class distribution instead of balanced sampling")
    p.add_argument("--frames", type=int, default=16)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--fusion", default="concat", choices=["mean", "concat", "attn"],
                   help="how per-modality embeddings are combined; mean = old behaviour")
    p.add_argument("--temporal", type=int, default=2,
                   help="temporal conv blocks in the image encoders; 0 = old bag-of-frames")
    p.add_argument("--supcon", type=float, default=0.0,
                   help="weight on the same-action/different-performer contrastive loss")
    p.add_argument("--supcon-temp", type=float, default=0.1)
    p.add_argument("--pk", default=None, metavar="P,K",
                   help="P classes x K samples per batch, required for --supcon")
    p.add_argument("--init", default=None,
                   help="self-supervised trunk checkpoint from pretrain.py")
    p.add_argument("--quant", default="none", choices=["none", "int8"],
                   help="int8 = per-channel symmetric quantized checkpoints, ~4x smaller, "
                        "so a large ensemble still fits the 100 MB cap")
    p.add_argument("--ema", type=float, default=0.0,
                   help="EMA decay for the scored/saved weights (0.99 recommended, 0=off)")
    p.add_argument("--last-epoch", action="store_true",
                   help="save the LAST epoch unconditionally instead of the best -- "
                        "removes selection bias on the fold you are measuring")
    p.add_argument("--streams", action="store_true",
                   help="skeleton joint+bone+motion streams (17x10 instead of 17x4)")
    p.add_argument("--drop-path", type=float, default=0.0,
                   help="stochastic depth, linearly scaled with block index")
    p.add_argument("--teacher", default="runs/teacher_oof.csv",
                   help="out-of-fold soft targets from make_teacher.py")
    p.add_argument("--kd", type=float, default=0.0,
                   help="distillation weight; 0 disables, loss = (1-kd)*CE + kd*T^2*KD")
    p.add_argument("--kd-temp", type=float, default=3.0)
    p.add_argument("--full-fit", action="store_true",
                   help="FINAL REFIT on every subject. No validation exists; the printed "
                        "accuracy is on training data. Use with --last-epoch and a "
                        "schedule already chosen from the fold runs.")
    p.add_argument("--val-users", default=None, metavar="U,U,...",
                   help="override the fold holdout with an explicit user list "
                        "(leave-two-folds-out teachers)")
    p.add_argument("--grl", type=float, default=0.0, help="subject-adversarial strength")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--no-tta", action="store_true", help="disable horizontal-flip TTA")
    p.add_argument("--no-amp", action="store_true", help="disable mixed precision on CUDA")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    dev = device()
    torch.manual_seed(a.seed)

    if a.predict:                       # modalities come from each checkpoint, not argv
        print(f"device: {dev}   (modalities read from checkpoints)")
        return predict(a, dev)
    print(f"device: {dev}   modalities: {a.modalities.split(',')}")

    folds = range(4) if a.all_folds else [a.fold]
    oof = pd.concat([run_fold(a, f, dev) for f in folds], ignore_index=True)
    oof.to_csv(Path(a.out) / "oof.csv", index=False)
    r = oof_report(oof[["user", "fold", "y_true", "y_pred"]])
    print(f"\n{'='*60}\nOOF  pooled={r['pooled_acc']:.4f}  "
          f"fold={r['fold_mean']:.4f} +/- {r['fold_std']:.4f}")
    print(f"per-fold: {r['per_fold']}")
    print(f"per-user: {r['per_user']}")
    print(f"worst user {r['worst_user']}  subject spread {r['subject_spread']:.3f}")
    if len(folds) > 1:
        n = lb_noise(oof[["user", "y_true", "y_pred"]])
        print(f"LB noise sd={n['std']:.4f} -> ignore LB deltas below {n['min_meaningful_delta']:.4f}")


if __name__ == "__main__":
    main()
