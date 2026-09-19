"""
check_env.py — run this FIRST on the new machine. 20 seconds, saves hours.

Verifies the three things that silently break a fresh setup:
  1. PyTorch actually has compiled kernels for your GPU's architecture.
     RTX 50-series is Blackwell / sm_120 and needs a cu128+ build. A cu121 wheel
     installs fine, reports cuda=True, then dies on the first conv with
     "no kernel image is available for execution on the device".
  2. A real conv + backward pass runs on the GPU (not just tensor allocation).
  3. The clip cache is present and complete.

Usage:
    python check_env.py
    python check_env.py --cache data/cache
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import sys
from pathlib import Path

OK, BAD = "  [OK] ", "  [!!] "


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data/cache")
    a = ap.parse_args()
    fail = []

    print(f"python  {sys.version.split()[0]}  ({sys.platform})")

    try:
        import torch
    except ImportError:
        sys.exit("torch not installed.  RTX 50-series needs:\n"
                 "  pip install torch --index-url https://download.pytorch.org/whl/cu128")
    print(f"torch   {torch.__version__}")

    # ---- GPU ---------------------------------------------------------------
    if not torch.cuda.is_available():
        print(BAD + "CUDA not available — training will fall back to CPU (very slow)")
        fail.append("cuda")
    else:
        name = torch.cuda.get_device_name(0)
        cc = torch.cuda.get_device_capability(0)
        arch = f"sm_{cc[0]}{cc[1]}"
        built = torch.cuda.get_arch_list()
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"gpu     {name}  {arch}  {vram:.1f} GB")
        print(f"built   {built}")
        if arch not in built:
            print(BAD + f"this torch build has NO kernels for {arch}")
            print("       pip uninstall -y torch")
            print("       pip install torch --index-url https://download.pytorch.org/whl/cu128")
            fail.append("arch")
        else:
            print(OK + f"{arch} kernels present")
            try:                       # real conv + backward, not just allocation
                x = torch.randn(4, 3, 64, 64, device="cuda", requires_grad=True)
                w = torch.nn.Conv2d(3, 8, 3, padding=1).cuda()
                with torch.autocast("cuda"):
                    y = w(x).sum()
                y.backward()
                torch.cuda.synchronize()
                print(OK + "conv2d forward+backward with AMP works on GPU")
            except Exception as e:
                print(BAD + f"GPU compute failed: {type(e).__name__}: {e}")
                fail.append("compute")

    # ---- deps --------------------------------------------------------------
    for mod in ("numpy", "pandas", "PIL"):
        try:
            __import__(mod)
            print(OK + f"{mod}")
        except ImportError:
            print(BAD + f"{mod} missing   ->  pip install "
                  + {"PIL": "pillow"}.get(mod, mod))
            fail.append(mod)

    sz = shutil.which("7z") or shutil.which("7zz") or shutil.which("7za")
    print((OK + f"7-Zip at {sz}") if sz else
          "  [--] 7-Zip not on PATH (only needed to re-extract the raw archives)")

    # ---- cache -------------------------------------------------------------
    c = Path(a.cache)
    if not (c / "manifest.csv").exists():
        print(BAD + f"no manifest.csv under {c.resolve()}")
        fail.append("cache")
    else:
        import pandas as pd
        m = pd.read_csv(c / "manifest.csv")
        ntr = len(glob.glob(str(c / "train" / "*.npz")))
        nte = len(glob.glob(str(c / "test" / "*.npz")))
        gb = sum(os.path.getsize(p) for p in glob.glob(str(c / "*" / "*.npz"))) / 1e9
        print(f"cache   {c.resolve()}")
        print(f"        manifest {len(m):,} rows | files train={ntr:,} test={nte:,} | {gb:.2f} GB")
        if ntr < 2900 or nte != 405:
            print(BAD + "expected ~3036 train and exactly 405 test clips — copy incomplete?")
            fail.append("cache")
        else:
            print(OK + "cache complete")
            users = sorted(m[m.split == "train"].user.unique())
            print(f"        users {users}")

    print("\n" + ("ALL CHECKS PASSED — ready to train" if not fail
                  else f"FAILED: {', '.join(fail)} — fix before training"))
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
