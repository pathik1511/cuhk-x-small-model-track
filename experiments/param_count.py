"""Exact parameter count for the submitted bundle, built the same way predict() builds it.

FusionNet has no `streams` argument. The checkpoint's `streams` flag becomes
seq_dims={"Skeleton": 17 * skel_ch}, and skel_ch is read from the stored weight shape,
not from a flag. Mirroring predict() exactly is the only way this number is the number
that actually runs.
"""
import glob
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from model import FusionNet
from train import dq8

d = sys.argv[1] if len(sys.argv) > 1 else "runs/r33_trim12"
ck = sorted(glob.glob(f"{d}/fold_*.pt")) or sorted(glob.glob(f"{d}/fold*.pt"))
if not ck:
    sys.exit(f"no checkpoints in {d}")

s = torch.load(ck[0], map_location="cpu", weights_only=False)
sd = dq8(s["model"]) if s.get("quant") == "int8" else s["model"]
w = sd.get("enc.Skeleton.net.0.weight")
skel_ch = int(w.shape[1] // 17) if w is not None else 4

m = FusionNet(s["mods"], 40, dim=s["dim"], width=s["width"],
              fusion=s.get("fusion", "mean"), temporal=s.get("temporal", 0),
              seq_dims={"Skeleton": 17 * skel_ch})

n = sum(p.numel() for p in m.parameters())
tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
print(f"checkpoints      : {len(ck)}")
print(f"modalities       : {s['mods']}")
print(f"skeleton channels: {skel_ch} (joint+bone+motion streams)" if skel_ch == 10 else
      f"skeleton channels: {skel_ch}")
print(f"quantization     : {s.get('quant')}")
print(f"trained on cache : {s.get('cache')}")
print(f"dim {s['dim']}, width {s['width']}, fusion {s.get('fusion')}, "
      f"temporal {s.get('temporal')}")
print()
print(f"EXACT PARAMETERS : {n:,}  ({n/1e6:.3f}M)")
print(f"  trainable      : {tr:,}")
print(f"  fp32 payload   : {n*4:,} bytes = {n*4/1e6:.2f} MB")
print(f"  int8 on disk   : {Path(ck[0]).stat().st_size:,} bytes per checkpoint")
