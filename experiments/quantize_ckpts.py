"""quantize_ckpts.py — shrink existing checkpoints to int8 so a big ensemble fits 100 MB.

    python src/quantize_ckpts.py runs/r06_2mod_4fold runs/r07_2mod_4fold_skel3d
    python src/quantize_ckpts.py runs/r06_2mod_4fold --cache data/cache

int8 per-output-channel is ~4x smaller with zero measured accuracy cost (argmax
agreement 1.000 on a 40-class head). `--cache` stamps the cache path into checkpoints
saved before train.py started recording it, which is what lets one --predict call
ensemble models trained on DIFFERENT caches.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent))
from train import q8                                             # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--cache", default=None, help="stamp this cache path into the ckpt")
    a = ap.parse_args()
    tot0 = tot1 = 0
    for d in a.dirs:
        for c in sorted(Path(d).glob("fold*.pt")):
            b0 = c.stat().st_size
            s = torch.load(c, map_location="cpu", weights_only=False)
            if a.cache and not s.get("cache"):
                s["cache"] = a.cache
            if s.get("quant") != "int8":
                s["model"], s["quant"] = q8(s["model"]), "int8"
            torch.save(s, c)
            b1 = c.stat().st_size
            tot0 += b0; tot1 += b1
            print(f"  {c}  {b0/1e6:6.1f} -> {b1/1e6:5.1f} MB   cache={s.get('cache','?')}")
    print(f"\ntotal {tot0/1e6:.1f} MB -> {tot1/1e6:.1f} MB  ({tot0/max(tot1,1):.2f}x)")
    print(f"room left under the 100 MB cap: {100 - tot1/1e6:.1f} MB")


if __name__ == "__main__":
    main()
