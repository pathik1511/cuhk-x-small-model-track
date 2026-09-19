"""Average fold checkpoints into one model. Ensemble accuracy at single-model size.

All folds fine-tune from the identical IG65M init, so they should still share a basin.
This is a test, not an assumption: predict with it and compare to the N-model ensemble.

  python src/soup.py --run runs/r42_r2p1d_e30 --out runs/r42_soup
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="runs/r42_r2p1d_e30")
    p.add_argument("--out", default="runs/r42_soup")
    a = p.parse_args()

    ck = sorted(Path(a.run).glob("fold*.pt"))
    assert len(ck) > 1, f"need 2+ checkpoints, found {len(ck)} in {a.run}"
    print(f"souping {len(ck)}: {[c.name for c in ck]}")

    sds = [torch.load(c, map_location="cpu")["model"] for c in ck]
    out = {}
    for k, v in sds[0].items():
        out[k] = (torch.stack([s[k].float() for s in sds]).mean(0).to(v.dtype)
                  if v.is_floating_point() else v.clone())   # num_batches_tracked: keep

    Path(a.out).mkdir(parents=True, exist_ok=True)
    torch.save({"model": out, "fold": -1, "acc": float("nan"),
                "arch": "r2plus1d_34_ig65m_kinetics", "soup": [c.name for c in ck]},
               Path(a.out) / "fold0.pt")
    print(f"-> {a.out}/fold0.pt")


if __name__ == "__main__":
    main()
