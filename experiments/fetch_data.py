"""
fetch_data.py — download the CUHK-X Small Model Track data from Hugging Face.

47.41 GB across a split archive. Resumable: re-run after an interruption and it
picks up where it stopped.

Prereqs (once):
    pip install huggingface_hub
    hf auth login          # paste a READ token; stored in ~/.cache/huggingface,
                           # never in your shell history or this repo
    # and accept the dataset terms in a browser first:
    # https://huggingface.co/datasets/Kevin-Pal/CUHK-X_Small_Model_Track

Usage:
    python fetch_data.py                 # everything (47.4 GB)
    python fetch_data.py --test-only     # just the 2.8 GB test set, to smoke-test auth
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO = "Kevin-Pal/CUHK-X_Small_Model_Track"
TRAIN_GB, TEST_GB = 44.6, 2.8

TEST_PATTERNS = ["Small-Model-Track/Testing/**", "Small-Model-Track/class_mapping.csv"]
ALL_PATTERNS = ["Small-Model-Track/**"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--test-only", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    out = Path(a.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    need = TEST_GB if a.test_only else TEST_GB + TRAIN_GB
    free = shutil.disk_usage(out).free / 1e9
    print(f"target   {out}\nneed     {need:.1f} GB download"
          f"{'' if a.test_only else ' (+ ~15-20 GB for selective extraction later)'}"
          f"\nfree     {free:.1f} GB")
    if free < need * 1.15:
        sys.exit(f"\nSTOP: not enough free disk. Free up space or pass --out to an external drive.")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("pip install huggingface_hub")

    try:
        p = snapshot_download(
            REPO, repo_type="dataset", local_dir=str(out),
            allow_patterns=TEST_PATTERNS if a.test_only else ALL_PATTERNS,
            max_workers=a.workers,
        )
    except Exception as e:
        n = type(e).__name__
        if "Gated" in n or "401" in str(e):
            sys.exit("\nGATED: open https://huggingface.co/datasets/Kevin-Pal/CUHK-X_Small_Model_Track"
                     "\n       sign in, accept the terms, then run:  hf auth login")
        raise

    print(f"\ndone -> {p}")
    for f in sorted(Path(p).rglob("*")):
        if f.is_file():
            print(f"  {f.stat().st_size/1e9:8.3f} GB  {f.relative_to(p)}")


if __name__ == "__main__":
    main()
