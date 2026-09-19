"""
dataset.py — PyTorch Dataset over the build_cache.py .npz cache.

Design is driven by one fact: the test subjects are people the model has never seen.
So every normalization here is chosen to DESTROY subject identity while keeping the
activity signal:

  Depth_Color  per-clip standardization      removes body size / distance-to-camera bias
  IR           per-clip standardization      removes reflectivity / skin differences
  Thermal      per-clip MEDIAN subtraction   removes baseline body temperature, which is
                                             a near-perfect subject fingerprint
  Skeleton     root-center + torso scale     removes STATURE (see _skel: the cache holds
                                             METRIC 3D, not 2D+confidence)
  IMU          per-clip per-channel z-score   removes sensor placement / mounting offset
  Radar        per-clip z-score

Missing modalities are normal (Radar is present in <50% of training clips). Every
sample carries a `mask` vector; absent modalities are zero tensors with mask=0, and
`mod_dropout` randomly masks present ones during training so the model can never
become dependent on any single sensor.

Usage:
    from dataset import CUHKClips, make_loaders
    tr, va = make_loaders("data/cache", fold=0, modalities=["Depth_Color", "Skeleton"])
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

MODS = ["Depth_Color", "IR", "Thermal", "Skeleton", "IMU", "Radar"]
# "DI" = Depth_Color(3) + IR(1) stacked into ONE 4-channel tensor and encoded by ONE
# CNN instead of two. Halves the image parameters and matches what the public
# leaderboard baseline does. Frame-locked identical timestamps make this exact.
IMG_MODS = {"Depth_Color": 3, "IR": 1, "Thermal": 3, "DI": 4, "HAND": 4}
# Composite image modalities: one encoder over several cached arrays stacked on channels.
COMPOSITE = {"DI":   [("Depth_Color", 3), ("IR", 1)],
             "HAND": [("Hand_Depth_Color", 3), ("Hand_IR", 1)]}
H36M_PARENT = [0, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
# Human3.6M left/right joint pairs — ONLY valid if analyze_cache.py confirms H36M order.
FLIP_PAIRS = [(1, 4), (2, 5), (3, 6), (14, 11), (15, 12), (16, 13)]
IMU_FLIP = [1, 0, 2, 4, 3]                      # LA<->RA, C, LL<->RL


class CUHKClips(Dataset):
    def __init__(self, manifest: pd.DataFrame, cache_dir: str | Path,
                 modalities: list[str] | None = None, train: bool = True,
                 frames: int = 16, mod_dropout: float = 0.25,
                 hflip: bool = True, seed: int = 0, size: tuple[int, int] | None = None,
                 skel_ch: int = 4, streams: bool = False):
        self.df = manifest.reset_index(drop=True)
        self.dir = Path(cache_dir)
        self.mods = modalities or MODS
        self.train, self.T, self.p_drop, self.hflip = train, frames, mod_dropout, hflip
        self.rng = np.random.default_rng(seed)
        # 4 = metric-3D + ground height (current). 3 = the old (x, y, "confidence")
        # layout, kept ONLY so checkpoints trained before the fix can still predict.
        self.skel_ch = skel_ch
        self.streams = streams
        # Zero tensors for absent modalities must match the cache's real (H,W), or the
        # collate fails. Infer it from the first cached clip rather than hardcoding.
        self.size = size or self._infer_size()
        self.sizes = self._infer_sizes()      # per-modality (H,W); HAND differs from DI
        ct = self._cache_frames()
        if ct and ct != self.T:
            raise ValueError(f"--frames {self.T} but the cache holds {ct} frames per clip. "
                             f"dataset.py does NOT resample; absent-modality zero tensors "
                             f"would mis-shape. Rebuild the cache or pass --frames {ct}.")

    def _infer_size(self) -> tuple[int, int]:
        for _, r in self.df.iterrows():
            fp = self.dir / r["split"] / f"{r['clip_id']}.npz"
            if not fp.exists():
                continue
            with np.load(fp) as z:
                for m in IMG_MODS:
                    if m in z.files:
                        return int(z[m].shape[1]), int(z[m].shape[2])    # (H, W)
        return (96, 128)

    def _infer_sizes(self) -> dict:
        """Absent-modality zero tensors must match each modality's real (H,W). The hand
        crop is cached at a different resolution from the body crop, so one global size
        is not enough."""
        out = {}
        for _, r in self.df.iterrows():
            fp = self.dir / r["split"] / f"{r['clip_id']}.npz"
            if not fp.exists():
                continue
            with np.load(fp) as z:
                for m in self.mods:
                    if m in out or m not in IMG_MODS:
                        continue
                    key = COMPOSITE[m][0][0] if m in COMPOSITE else m
                    if key in z.files:
                        out[m] = (int(z[key].shape[1]), int(z[key].shape[2]))
            if all(m in out for m in self.mods if m in IMG_MODS):
                break
        return out

    def _cache_frames(self) -> int:
        for _, r in self.df.iterrows():
            fp = self.dir / r["split"] / f"{r['clip_id']}.npz"
            if fp.exists():
                with np.load(fp) as z:
                    for m in ("Depth_Color", "IR", "Thermal", "Skeleton"):
                        if m in z.files:
                            return int(z[m].shape[0])
        return 0

    def __len__(self) -> int:
        return len(self.df)

    # ------------------------------------------------------------ normalizers
    @staticmethod
    def _img(v: np.ndarray, thermal: bool) -> np.ndarray:
        """[T,H,W,C] uint8 -> [C,T,H,W] float32, subject-invariant."""
        x = v.astype(np.float32) / 255.0
        x -= np.median(x) if thermal else x.mean()      # thermal: kill baseline body temp
        s = x.std()
        return np.ascontiguousarray((x / (s + 1e-5)).transpose(3, 0, 1, 2))

    @staticmethod
    def _skel(v: np.ndarray) -> np.ndarray:
        """[T,17,3] = METRIC 3D  (x_lateral, y_depth, z_height) in metres, feet at z~=0.

        This is NOT 2D+confidence. Measured on the raw JSON: head joint 10 has mean
        z=1.149 and the ankles (3, 6) have mean z=0.010 -- col 2 is HEIGHT ABOVE THE
        FLOOR, which is why it is never negative. The separate `keypoint_scores` field
        in the JSON is the confidence, it is all 1.0, and build_cache.py discards it.

        The old normalization centred and scaled only cols 0-1 and passed col 2 through
        untouched, which fed the model the subject's ABSOLUTE HEIGHT IN METRES -- a
        near-perfect identity fingerprint, on the split where subject identity is the
        dominant error term.

        Now: scale by torso length (root->thorax, rigid, so pose-invariant), emit
        root-centred normalized 3D in channels 0-2, and keep ground-relative height in
        channel 3 so Stand_up / Sit_down / Lie_down stay separable without stature.
        """
        x = v.astype(np.float32)
        torso = float(np.linalg.norm(x[:, 8] - x[:, 0], axis=-1).mean())
        torso = torso if torso > 1e-3 else 1.0
        ground = x[..., 2:3] / torso                     # height above floor / stature
        c = (x - x[:, :1]) / torso                       # root-centred, stature-free
        return np.ascontiguousarray(np.concatenate([c, ground], -1))   # [T,17,4]

    @staticmethod
    def _streams(x: np.ndarray) -> np.ndarray:
        """Joint + BONE + MOTION, the cheapest large win in the skeleton literature:
        CTR-GCN goes 88.8 joint-only -> 92.4 four-stream on NTU-60. Bone = the vector
        to the H36M parent joint, motion = the temporal difference. Pure arithmetic,
        no new parameters, computed after augmentation so it stays consistent.
        [T,17,4] -> [T,17,10]"""
        bone = x[..., :3] - x[:, H36M_PARENT, :3]
        mot = np.zeros_like(x[..., :3])
        mot[1:] = x[1:, :, :3] - x[:-1, :, :3]
        return np.ascontiguousarray(np.concatenate([x, bone, mot], -1))

    @staticmethod
    def _skel_legacy(v: np.ndarray) -> np.ndarray:
        """Pre-fix normalization. Wrong (leaks stature) but required to load old ckpts."""
        x = v.astype(np.float32).copy()
        x[..., :2] -= x[:, :1, :2]
        scale = np.linalg.norm(x[:, 1:, :2] - x[:, :1, :2], axis=-1).mean()
        x[..., :2] /= (scale + 1e-6)
        return x

    @staticmethod
    def _seq(v: np.ndarray) -> np.ndarray:
        x = np.nan_to_num(v.astype(np.float32))
        return (x - x.mean(0, keepdims=True)) / (x.std(0, keepdims=True) + 1e-5)

    # ---------------------------------------------------------- augmentations
    def _aug_img(self, x: np.ndarray, shift: int, flip: bool) -> np.ndarray:
        if shift:
            x = np.roll(x, shift, axis=1)                # temporal jitter, [C,T,H,W]
        if flip:
            x = x[..., ::-1]
        return np.ascontiguousarray(x)

    def _aug_skel(self, x: np.ndarray, shift: int, flip: bool, ang: float) -> np.ndarray:
        """3D metric pose. Channels 0,1 are lateral/depth, so rotating them is a YAW
        rotation about the vertical axis -- the correct viewpoint augmentation. Channel
        2 (root-relative height) and channel 3 (ground height) must not be rotated."""
        if shift:
            x = np.roll(x, shift, axis=0)
        if flip or ang:
            x = x.copy()
        if flip:
            x[..., 0] *= -1
            for i, j in FLIP_PAIRS:
                x[:, [i, j]] = x[:, [j, i]]
        if ang:
            c, s = np.cos(ang), np.sin(ang)
            u, v = x[..., 0].copy(), x[..., 1].copy()
            x[..., 0], x[..., 1] = c * u - s * v, s * u + c * v
        return np.ascontiguousarray(x)

    # -------------------------------------------------------------- __getitem__
    def __getitem__(self, i: int):
        r = self.df.iloc[i]
        fp = self.dir / r["split"] / f"{r['clip_id']}.npz"
        out, mask = {}, np.zeros(len(self.mods), np.float32)

        shift = int(self.rng.integers(-3, 4)) if self.train else 0
        flip = bool(self.train and self.hflip and self.rng.random() < 0.5)
        ang = float(self.rng.uniform(-0.26, 0.26)) if self.train else 0.0   # +-15 degrees

        with np.load(fp) as z:
            avail = set(z.files)
            for k, m in enumerate(self.mods):
                present = (any(k in avail for k, _ in COMPOSITE[m]) if m in COMPOSITE
                           else m in avail)
                if present and self.train and self.rng.random() < self.p_drop:
                    present = False                       # modality dropout
                if not present:
                    out[m] = torch.zeros(self._shape(m))
                    continue
                if m in COMPOSITE:
                    hw = self.sizes.get(m, self.size)
                    parts = []
                    for sub, ch in COMPOSITE[m]:
                        parts.append(self._img(z[sub], False) if sub in avail
                                     else np.zeros((ch, self.T, *hw), np.float32))
                    x = self._aug_img(np.concatenate(parts, 0), shift, flip)
                    out[m] = torch.from_numpy(np.asarray(x, np.float32))
                    mask[k] = 1.0
                    continue
                v = z[m]
                if m in IMG_MODS:
                    x = self._aug_img(self._img(v, m == "Thermal"), shift, flip)
                elif m == "Skeleton":
                    sk = self._skel(v) if self.skel_ch == 4 else self._skel_legacy(v)
                    x = self._aug_skel(sk, shift, flip, ang)
                    if self.skel_ch == 4 and self.streams:
                        x = self._streams(x)
                else:
                    x = self._seq(v)
                    if self.train:
                        x = np.roll(x, shift, 0) + self.rng.normal(0, .02, x.shape).astype(np.float32)
                    if m == "IMU" and flip:
                        x = x[:, IMU_FLIP]
                out[m] = torch.from_numpy(np.asarray(x, np.float32))
                mask[k] = 1.0

        out["mask"] = torch.from_numpy(mask)
        out["y"] = torch.tensor(int(r["action_id"]), dtype=torch.long)
        out["clip_id"] = r["clip_id"]
        out["user"] = int(r["user"])          # required by --grl; was silently missing
        return out

    def _shape(self, m: str) -> tuple:
        H, W = self.size
        sk = 10 if (self.skel_ch == 4 and self.streams) else self.skel_ch
        seq = {"Skeleton": (self.T, 17, sk), "IMU": (self.T, 5, 16), "Radar": (self.T, 12)}
        if m in seq:
            return seq[m]
        h, w = self.sizes.get(m, (H, W))
        return (IMG_MODS[m], self.T, h, w)


def _worker_init(_wid: int) -> None:
    """Each worker inherits a copy of the Dataset holding the SAME rng seed, so without
    this every worker draws an identical augmentation stream. Must be module-level:
    Windows spawns workers and pickles this by reference; a closure is unpicklable.

    torch.initial_seed() is base_seed + worker_id, and base_seed is redrawn every epoch,
    so this is distinct per worker AND per epoch with nothing captured from the caller.
    """
    info = torch.utils.data.get_worker_info()
    if info is not None:
        info.dataset.rng = np.random.default_rng(torch.initial_seed() % 2**31)


# ------------------------------------------------------------------- loaders --
def make_loaders(cache_dir: str | Path, fold: int = 0, seed: int = 0,
                 modalities: list[str] | None = None, batch_size: int = 32,
                 workers: int = 4, balanced: bool = True, pk: str | None = None,
                 val_users: list[int] | None = None, full_fit: bool = False, **kw):
    """Train/val loaders for one cross-subject fold, with class-balanced sampling."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from cuhk_cv import subject_folds

    cache = Path(cache_dir)
    df = pd.read_csv(cache / "manifest.csv")
    df = df[(df.split == "train") & df.action_id.ge(0)]
    tr_u, va_u = list(subject_folds(seed=seed))[fold]
    if full_fit:                     # FINAL REFIT: train on every available subject.
        # There is no held-out subject left, so nothing here is validation. `va` is a
        # sample of the TRAINING clips, reported only as a progress readout. Select the
        # schedule beforehand from the fold runs and use --last-epoch.
        tr_u = sorted(df.user.unique().tolist())
        va_u = tr_u
        tr = df
        va = df.sample(min(400, len(df)), random_state=0)
    elif val_users:                  # explicit holdout (leave-two-folds-out teachers)
        va_u = sorted(val_users)
        tr_u = sorted(set(df.user.unique().tolist()) - set(va_u))
    if not full_fit:
        tr, va = df[df.user.isin(tr_u)], df[df.user.isin(va_u)]

    # Class-balanced sampling protects the 4 classes with <=8 subjects, but it also
    # shifts the model's implicit class prior toward uniform. If the TEST set mirrors
    # training's imbalance rather than being balanced, that shift costs accuracy.
    # Settle it empirically: train both ways and compare OOF.
    sampler = None
    if pk:
        import sys as _s; _s.path.insert(0, str(Path(__file__).parent))
        from train import PKSampler                       # noqa: E402
        P, K = (int(x) for x in pk.split(","))
        dtr_lab = tr.action_id.tolist()
        # Keep the OPTIMIZER STEP COUNT identical to the non-PK runs. Changing batch
        # size at fixed epochs silently changes steps -- that confound cost this project
        # a wrong conclusion once already (run C).
        sampler = PKSampler(dtr_lab, P, K, batches=max(1, len(tr) // batch_size), seed=seed)
        batch_size = P * K
        balanced = False
    if balanced:
        cnt = tr.action_id.value_counts()
        w = torch.tensor((1.0 / cnt[tr.action_id]).values, dtype=torch.double)
        sampler = torch.utils.data.WeightedRandomSampler(w, len(w), replacement=True)

    val_kw = {k: v for k, v in kw.items() if k != "mod_dropout"}   # val never drops modalities
    dtr = CUHKClips(tr, cache, modalities, train=True, seed=seed, **kw)
    dva = CUHKClips(va, cache, modalities, train=False, mod_dropout=0.0, **val_kw)
    return (DataLoader(dtr, batch_size, sampler=sampler, shuffle=sampler is None,
                       num_workers=workers, pin_memory=True, drop_last=True,
                       worker_init_fn=_worker_init if workers else None),
            DataLoader(dva, batch_size, shuffle=False, num_workers=workers, pin_memory=True))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/cache")
    ap.add_argument("--fold", type=int, default=0)
    a = ap.parse_args()
    tr, va = make_loaders(a.cache, a.fold, workers=0, batch_size=4)
    print(f"train batches {len(tr)}  val batches {len(va)}")
    b = next(iter(tr))
    for k, v in b.items():
        print(f"  {k:<14}{tuple(v.shape) if torch.is_tensor(v) else type(v).__name__}"
              f"{'' if not torch.is_tensor(v) or not v.is_floating_point() else f'  mean={v.mean():+.3f} std={v.std():.3f}'}")
    print(f"  labels: {b['y'].tolist()}   mask: {b['mask'].tolist()}")
