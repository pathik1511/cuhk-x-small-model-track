"""In-fold teacher targets: the CORRECT clean construction, replacing make_teacher.py.

For student outer fold F, a teacher is admissible if it trained on exactly
train_subjects(F), i.e. it excluded val_users(F) from training, preprocessing and
selection. Its predictions on F's own TRAINING clips are IN-SAMPLE. That affects how
useful the targets are; it does not leak, because nothing about val_users(F) ever
reached the teacher.

This is what `make_teacher.py` got wrong. That version required targets to be out of
fold with respect to the CLIP, which forced cross-fold teachers that had trained on the
student's validation subjects. The leave-two-folds-out repair then removed so many
subjects that the teachers came in 0.0471 BELOW the student. Neither was necessary.

    python src/make_teacher_infold.py --cache data/cache_hand \
        --runs r19_sadp,r27_ntu,r16_hand_s0,r17_hand_w24,r21_hand3_ungated \
        --out runs/teacher_infold_f{fold}.csv

PROVENANCE IS ASSERTED, NOT ASSUMED. A file named fold0.pt from a different --seed
excluded different people: seeds re-draw the fold assignment. Every run's actual
validation users are read from its oof.csv and must equal subject_folds(seed)[F][1],
or the build refuses.
"""
from __future__ import annotations

import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from cuhk_cv import subject_folds                                    # noqa: E402
from dataset import CUHKClips                                        # noqa: E402
from model import FusionNet                                          # noqa: E402
from train import dq8, to_dev, flip_batch, roll_batch, device, load_ckpt  # noqa: E402

P = [f"p{i}" for i in range(40)]


def check_provenance(runs: list[str], fold: int, want: set[int]) -> None:
    for r in runs:
        f = Path("runs") / r / "oof.csv"
        if not f.exists():
            sys.exit(f"{r}: no oof.csv, cannot verify which users it held out")
        d = pd.read_csv(f)
        got = set(d[d.fold == fold].user.unique().tolist())
        if got != want:
            sys.exit(f"REFUSED. {r} fold {fold} held out {sorted(got)}, "
                     f"student fold {fold} holds out {sorted(want)}. "
                     f"Different seed, different people. This teacher trained on the "
                     f"student's validation subjects and would leak.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/cache_hand")
    ap.add_argument("--runs", required=True, help="comma separated run dirs under runs/")
    ap.add_argument("--out", default="runs/teacher_infold_f{fold}.csv")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-tta", action="store_true")
    a = ap.parse_args()
    runs = [r.strip() for r in a.runs.split(",") if r.strip()]
    dev = device()
    cache = Path(a.cache)
    man = pd.read_csv(cache / "manifest.csv")
    man = man[(man.split == "train") & man.action_id.ge(0)]
    print(f"device {dev}   teachers: {', '.join(runs)}\n")

    for fold in range(4):
        tr_u, va_u = list(subject_folds(seed=a.seed))[fold]
        check_provenance(runs, fold, set(va_u))
        sub = man[man.user.isin(tr_u)]                # the STUDENT's training clips
        assert not set(sub.user) & set(va_u), "training split touches the holdout"
        acc: dict[str, np.ndarray] = {}
        n_t = 0
        for r in runs:
            c = Path("runs") / r / f"fold{fold}.pt"
            s = torch.load(c, map_location=dev, weights_only=False)
            s["model"] = dq8(s["model"]) if s.get("quant") == "int8" else s["model"]
            w = s["model"].get("enc.Skeleton.net.0.weight")
            skel_ch = int(w.shape[1] // 17) if w is not None else 4
            streams = s.get("streams", False) or skel_ch == 10
            m = FusionNet(s["mods"], 40, dim=s["dim"], width=s["width"],
                          fusion=s.get("fusion", "mean"), temporal=s.get("temporal", 0),
                          seq_dims={"Skeleton": 17 * skel_ch}).to(dev).eval()
            load_ckpt(m, s["model"], f"{r} fold{fold}")
            ds = CUHKClips(sub, cache, s["mods"], train=False, mod_dropout=0.0,
                           frames=a.frames, skel_ch=4 if streams else skel_ch,
                           streams=streams)
            dl = torch.utils.data.DataLoader(ds, a.batch_size, shuffle=False,
                                             num_workers=a.workers)
            with torch.no_grad():
                for b in dl:
                    d0 = to_dev(b, dev, s["mods"])
                    v = [d0] if a.no_tta else [d0, flip_batch(d0, s["mods"]),
                                               roll_batch(d0, s["mods"], 1),
                                               roll_batch(d0, s["mods"], -1)]
                    pr = (sum(m(x)["logits"].softmax(-1) for x in v) / len(v)).cpu().numpy()
                    for i, cid in enumerate(b["clip_id"]):   # by clip id, never by row
                        acc[cid] = acc[cid] + pr[i] if cid in acc else pr[i].copy()
            n_t += 1
            print(f"  fold {fold}  {r:20s} done")
        ids = sorted(acc)
        p = np.stack([acc[i] for i in ids]) / n_t
        y = man.set_index("clip_id").loc[ids, "action_id"].values
        out = pd.DataFrame(p, index=pd.Index(ids, name="clip_id"),
                           columns=[f"t{i}" for i in range(40)])
        out.insert(0, "y_true", y)
        path = a.out.format(fold=fold)
        out.to_csv(path)
        print(f"  fold {fold}  holdout {va_u}  {len(ids)} training clips  "
              f"teacher top1 {(p.argmax(1) == y).mean():.4f} (IN-SAMPLE)  "
              f"mass@argmax {p.max(1).mean():.3f}  -> {path}\n")


if __name__ == "__main__":
    main()
