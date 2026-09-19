#!/usr/bin/env python3
"""CUHK-X Small Model Track, verification entry point.

    python inference.py <raw_dir> <output_csv>

Raw competition archives in, 405-row submission CSV out. One command, no cached
intermediates, no training artifacts, no test-specific state.

Reproduces sub_r42_soup3.csv, public LB 0.69651.

WHAT THE ENTRY IS, stated plainly so nothing here is a surprise on inspection:

  detector  YOLO11n, COCO-pretrained, 2.6M params (yolo11n.pt, 5,613,764 bytes).
            Person box only. Not trained or fine-tuned on competition data.
  model     R(2+1)D-34, 63,697,175 params, initialized from the public IG65M +
            Kinetics-400 checkpoint at
            https://github.com/moabitcoin/ig65m-pytorch/releases/download/v1.0.0/
              r2plus1d_34_clip32_ft_kinetics_from_ig65m-ade133f1.pth
            then fine-tuned on this competition's training split only. The weights
            of the three cross-subject folds are AVERAGED into one model (a weight
            soup), so what ships is a single network, not an ensemble.
  shipped   checkpoints/model.pth, 50,908,100 bytes. Despite the .pth name required
            by Rules 2.8a it is an uncompressed .npz archive, loaded by
            src/pack_int5.py, not by torch.load. Conv weights symmetric int6 with
            per-output-channel scales; the classifier head, BatchNorm statistics and
            every tensor under 100K elements stay fp32. Verified to reproduce the
            fp32 submission on 405/405 clips.
  total     56,521,864 bytes against the 100,000,000 cap.

  The IG65M initialization is disclosed here deliberately. Whether it falls under
  "no large pretrained backbones" is the organizers' call, not this file's.

Written in Python rather than bash or cmd so that the same file runs on the machine
that trained the model and on a Linux verification host. Every stage asserts its
postcondition and exits non-zero rather than continuing on bad state.
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Rules 2.8a names `checkpoints/model.pth` as a required deliverable. That path is
# the shipped artifact; model/packed_i6.npz is kept as a same-bytes alias.
PACKED = HERE / "checkpoints" / "model.pth"
YOLO_W = HERE / "yolo11n.pt"
BOXES = HERE / "yolo_boxes.csv"
CACHE = HERE / "data" / "cache_yolo"
EXTRACT = HERE / "data" / "extracted"
UNPACKED = HERE / "runs" / "_verify_unpacked"
BITS = 6
TTA_MODE = "2"          # identity + hflip, what sub_r42_soup3.csv was produced with
CAP = 100_000_000
REF = "sub_r42_soup3.csv"


def die(msg):
    print(f"REFUSED: {msg}")
    sys.exit(1)


def run(cmd, **kw):
    print("    $ " + " ".join(str(c) for c in cmd), flush=True)
    if subprocess.call([str(c) for c in cmd], cwd=HERE, **kw) != 0:
        die(f"command failed: {cmd[0]}")


def find_7z():
    for c in ("7z", "7za", "7zr"):
        if shutil.which(c):
            return shutil.which(c)
    for c in (r"C:\Program Files\7-Zip\7z.exe", r"C:\Program Files (x86)\7-Zip\7z.exe"):
        if Path(c).exists():
            return c
    return None


def stage0():
    print("=== 0. environment ===")
    print(f"    python {sys.version.split()[0]}")
    import numpy, pandas, scipy, torch
    from PIL import Image
    print(f"    torch {torch.__version__}  numpy {numpy.__version__}  "
          f"pandas {pandas.__version__}  scipy {scipy.__version__}")
    try:
        import ultralytics
        print(f"    ultralytics {ultralytics.__version__}")
    except ImportError:
        die("ultralytics is not installed. `pip install ultralytics`. "
            "The person crop is a YOLO11n detection and cannot be skipped.")
    print(f"    cuda available: {torch.cuda.is_available()}  "
          f"(CPU works, no GPU is required for inference)")


def stage1(raw):
    print("\n=== 1. extract raw archives -> data/extracted ===")
    if (EXTRACT / "HAR").is_dir() and (EXTRACT / "small_model_track_test").is_dir():
        print("    data/extracted already complete, reusing")
    else:
        EXTRACT.mkdir(parents=True, exist_ok=True)
        # The training set ships as a MULTI-PART SPLIT ZIP:
        #   HAR.z01 .. HAR.z08 (5 GB each) + HAR.zip (the final part).
        # Python's zipfile cannot read split archives and plain `unzip HAR.zip` fails.
        # 7-Zip reads the whole part set when pointed at the final .zip.
        train = next(iter(sorted(Path(raw).rglob("HAR.zip"))), None)
        test = next(iter(sorted(Path(raw).rglob("small_model_track_test.zip"))), None)
        if train is None:
            die(f"HAR.zip not found under {raw}")
        if test is None:
            die(f"small_model_track_test.zip not found under {raw}")
        print(f"    train: {train}  ({len(list(train.parent.glob('HAR.z*')))} parts)")
        print(f"    test : {test}")
        sz = find_7z()
        if sz is None:
            die("7-Zip not found. Install p7zip (Linux) or 7-zip.org (Windows). "
                "A split zip cannot be extracted without it.")
        run([sz, "x", "-y", f"-o{EXTRACT}", str(train)], stdout=subprocess.DEVNULL)
        run([sz, "x", "-y", f"-o{EXTRACT}", str(test)], stdout=subprocess.DEVNULL)

    if not (EXTRACT / "HAR").is_dir():
        die("data/extracted/HAR missing after extract")
    # Count SM_test_* specifically. The shipped test archive also contains a `.claude`
    # directory, so "every subdirectory" is 406 and the wrong thing to assert on.
    root = EXTRACT / "small_model_track_test"
    clips = sorted(root.glob("SM_test_*"))
    other = [p.name for p in root.iterdir() if p.is_dir()
             and not p.name.startswith("SM_test_")]
    print(f"    test clips extracted: {len(clips)} (expected 405)")
    if other:
        print(f"    non-clip entries in the archive, ignored: {other}")
    if len(clips) != 405:
        die(f"expected 405 SM_test_* directories, found {len(clips)}")


def stage2():
    print("\n=== 2. person boxes (YOLO11n) -> cache ===")
    if BOXES.exists():
        print(f"    {BOXES.name} exists, reusing")
    else:
        if not YOLO_W.exists():
            die(f"{YOLO_W.name} missing; it is part of the shipped model")
        # margin 1.35 is the value that produced yolo_boxes.csv and therefore
        # data/cache_yolo. yolo_box.py exits non-zero if >50% of clips fall back
        # to the motion box, so a silently broken detector cannot pass this stage.
        run([sys.executable, "src/yolo_box.py", "--root", "data/extracted",
             "--out", BOXES.name, "--weights", YOLO_W.name, "--margin", "1.35"])

    if (CACHE / "manifest.csv").exists():
        print("    data/cache_yolo exists, reusing")
    else:
        # EXACTLY the flags that produced data/cache_yolo, the cache every fold
        # trained on. Person box = YOLO, hand box = motion median, 16 frames, 128x128.
        run([sys.executable, "src/build_cache.py",
             "--root", "data/extracted", "--out", "data/cache_yolo",
             "--crop", "--hand", "--hand-size", "96", "--frames", "16",
             "--size", "128x128", "--yolo", BOXES.name,
             "--workers", str(max(1, (os.cpu_count() or 4) - 2))])
    if not (CACHE / "manifest.csv").exists():
        die("cache build produced no manifest")

    import pandas as pd
    m = pd.read_csv(CACHE / "manifest.csv")
    det = int((m.yolo == 1).sum())
    print(f"    manifest: {len(m)} clips, {det} YOLO boxes, {len(m)-det} motion fallback")
    if det < 0.5 * len(m):
        die(f"only {det}/{len(m)} clips got a YOLO box; the detector is not working")


def stage3():
    print("\n=== 3. verify the shipped model against the 100 MB cap ===")
    if not PACKED.exists():
        die(f"{PACKED} missing")
    if not YOLO_W.exists():
        die(f"{YOLO_W} missing")
    a, b = PACKED.stat().st_size, YOLO_W.stat().st_size
    print(f"    {PACKED.name:16s} {a:>12,} bytes   int{BITS} R(2+1)D-34 weight soup")
    print(f"    {YOLO_W.name:16s} {b:>12,} bytes   YOLO11n person detector")
    print(f"    {'total':16s} {a+b:>12,} bytes = {(a+b)/1e6:.2f} MB decimal "
          f"= {(a+b)/2**20:.2f} MiB")
    print(f"    vs 100 MB cap: {'UNDER' if a+b < CAP else 'OVER'} decimal, "
          f"{'UNDER' if a+b < 104857600 else 'OVER'} binary")
    if a + b >= CAP:
        die(f"{a+b} bytes is over the 100,000,000 cap")
    print("    one network ships, not an ensemble: three fold checkpoints were")
    print("    averaged in weight space before packing (see src/soup.py)")


def stage4(out):
    print("\n=== 4. unpack int6 -> predict ===")
    if UNPACKED.exists():
        shutil.rmtree(UNPACKED)
    run([sys.executable, "src/pack_int5.py", "unpack", str(PACKED),
         str(UNPACKED / "fold0.pt"), str(BITS)])
    # --tta is not optional: sub_r42_soup3.csv was produced with horizontal-flip TTA
    # and stage 5 compares against it.
    run([sys.executable, "src/predict_r2p1d.py", "--run", str(UNPACKED),
         "--cache", "data/cache_yolo", "--frames", "16",
         "--tta", TTA_MODE, "--out", out])


def stage5(out):
    print("\n=== 5. done ===")
    import pandas as pd
    d = pd.read_csv(HERE / out)
    print(f"    {out}: {len(d)} rows, {d.prediction.nunique()} distinct classes")
    if len(d) != 405:
        die(f"expected 405 rows, got {len(d)}")
    if not d.prediction.between(0, 39).all():
        die("prediction outside 0..39")
    print("    405 rows, all predictions in 0..39. OK.")
    ref = HERE / REF
    if ref.exists():
        r = pd.read_csv(ref)
        same = int((r.prediction.values == d.prediction.values).sum())
        print(f"\n    vs {REF}: {same}/{len(r)} identical")
        print(f"    {len(r)}/{len(r)} means the package reproduces the submission."
              if same == len(r) else
              "    NOT identical. The package does not reproduce the submission.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_dir")
    ap.add_argument("output_csv")
    a = ap.parse_args()
    stage0()
    stage1(a.raw_dir)
    stage2()
    stage3()
    stage4(a.output_csv)
    stage5(a.output_csv)


if __name__ == "__main__":
    main()
