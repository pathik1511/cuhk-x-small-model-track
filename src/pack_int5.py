"""Pack a checkpoint to int5 and unpack it back, to prove the 100 MB cap is met.

Conv and linear weights -> symmetric per-output-channel int5 (codes 0..30, zero at 15).
Everything else (BN, biases, the 40-way head bias) stays fp16; it is a rounding error
of the total. The packed .npz IS the artifact whose size must clear the cap.

  python src/pack_int5.py pack   in.pt  out.npz [bits]      # bits defaults to 5
  python src/pack_int5.py unpack out.npz back.pt [bits]     # SAME bits as pack
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

BITS, LEVELS = 5, 15          # set by argv; codes -L..L stored +L, fits in BITS bits


def _is_weight(k: str, v: torch.Tensor) -> bool:
    # Quantize only the bulk conv weights. The 40-way head and every small tensor stay
    # fp32: together they are ~0.4 MB against a 100 MB cap, and the head is the most
    # precision-sensitive tensor in the model. BN running_var stays fp32 too; in fp16
    # any variance below 6e-5 underflows to zero and the normalization explodes.
    if not v.is_floating_point() or v.ndim < 2 or v.numel() < 100_000:
        return False
    return "head" not in k and "fc" not in k


def _pack(codes: np.ndarray) -> np.ndarray:            # uint8 in [0,31] -> bitstream
    bits = np.unpackbits(codes.reshape(-1, 1), axis=1)[:, 8 - BITS:]
    return np.packbits(bits.reshape(-1))


def _unpack(buf: np.ndarray, n: int) -> np.ndarray:
    bits = np.unpackbits(buf)[: n * BITS].reshape(n, BITS)
    pad = np.zeros((n, 8 - BITS), np.uint8)
    return np.packbits(np.concatenate([pad, bits], 1), axis=1).ravel()


def pack(src: str, dst: str) -> None:
    sd = torch.load(src, map_location="cpu")["model"]
    out, nq, nf = {}, 0, 0
    for k, v in sd.items():
        if _is_weight(k, v):
            w = v.float().reshape(v.shape[0], -1).numpy()
            s = np.abs(w).max(1) / LEVELS
            s[s == 0] = 1e-8
            q = np.clip(np.rint(w / s[:, None]), -LEVELS, LEVELS).astype(np.int8) + LEVELS
            out[k + "|q"] = _pack(q.astype(np.uint8))
            out[k + "|s"] = s.astype(np.float16)
            out[k + "|d"] = np.array(v.shape, np.int32)
            nq += v.numel()
        else:
            out[k + "|f"] = v.numpy()               # fp32, no precision games
            nf += v.numel()
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    np.savez(dst, **out)
    b = Path(dst).stat().st_size
    print(f"int{BITS} {nq:,} weights | fp32 {nf:,} others")
    print(f"{dst}  {b:,} bytes = {b/1e6:.2f} MB (decimal) / {b/2**20:.2f} MiB")
    print(f"cap 100 MB decimal: {'PASS' if b <= 100_000_000 else 'FAIL'}")


def unpack(src: str, dst: str) -> None:
    z = np.load(src)
    keys = {k.rsplit("|", 1)[0] for k in z.files}
    sd = {}
    for k in sorted(keys):
        if k + "|f" in z.files:
            sd[k] = torch.from_numpy(z[k + "|f"])
        else:
            shp = tuple(int(x) for x in z[k + "|d"])
            n = int(np.prod(shp))
            q = _unpack(z[k + "|q"], n).astype(np.float32) - LEVELS
            s = z[k + "|s"].astype(np.float32)
            sd[k] = torch.from_numpy((q.reshape(shp[0], -1) * s[:, None]).reshape(shp))
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": sd, "fold": -1, "acc": float("nan"),
                "arch": "r2plus1d_34_ig65m_kinetics", "quant": "int5"}, dst)
    print(f"-> {dst}  ({len(sd)} tensors)")


if __name__ == "__main__":
    if len(sys.argv) > 4:                       # optional bit width, default 5
        BITS = int(sys.argv[4])
        LEVELS = 2 ** (BITS - 1) - 1
    (pack if sys.argv[1] == "pack" else unpack)(sys.argv[2], sys.argv[3])
