"""
inspect_archive.py v2 — learn the dataset layout WITHOUT extracting 47 GB.

`7z l` reads only the archive index. With --probe it additionally extracts a
handful of sample files (a few hundred KB) so we learn real image dimensions,
CSV headers and JSON schemas before committing to a cache format.

Install 7z on macOS (Apple's unzip cannot read the split HAR archive):
    brew install sevenzip

Usage:
    python inspect_archive.py data/raw/Small-Model-Track/Testing/data/small_model_track_test.zip --probe 2
    python inspect_archive.py data/raw/Small-Model-Track/Training/data/HAR.zip --probe 2
    python inspect_archive.py --dir data/extracted/HAR
"""
from __future__ import annotations

import argparse
import collections as C
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

MODALITIES = ("Depth_Color", "IR", "Thermal", "Skeleton", "IMU", "Radar")
JUNK = re.compile(r"(^|/)(__MACOSX|\.DS_Store|\._|\.claude|\.git)", re.I)


def sevenzip() -> str:
    for b in ("7zz", "7z", "7za"):
        if shutil.which(b):
            return b
    sys.exit("no 7z found — run:  brew install sevenzip")


def _parse(lines: list[str]) -> list[tuple[str, int]]:
    """7z prints Size right-aligned in a 12-wide field ENDING at the 'Size' header."""
    hdr = next((i for i, l in enumerate(lines) if "Name" in l and "Size" in l), None)
    if hdr is None:
        return []
    ncol = lines[hdr].index("Name")
    send = lines[hdr].index("Size") + 4
    rows, started = [], False
    for l in lines[hdr + 1:]:
        if l.startswith("---"):
            if started:
                break
            started = True
            continue
        if not started or len(l) <= ncol or "D" in l[20:25]:
            continue
        try:
            size = int(l[send - 12:send].strip() or 0)
        except ValueError:
            size = 0
        rows.append((l[ncol:].strip().replace("\\", "/"), size))
    return rows


def from_archive(path: str) -> list[tuple[str, int]]:
    p = Path(path)
    if not p.exists():
        sys.exit(f"archive does not exist: {p}\n       run fetch_data.py first")
    if p.stat().st_size == 0:
        sys.exit(f"archive is 0 bytes (partial download): {p}")
    if p.suffix.lower() == ".zip":
        parts = sorted(p.parent.glob(p.stem + ".z[0-9][0-9]"))
        if parts:
            print(f"split archive: {p.name} + {len(parts)} parts "
                  f"({sum(x.stat().st_size for x in parts)/1e9:.1f} GB)\n")
    out = subprocess.run([sevenzip(), "l", str(p)], capture_output=True, text=True).stdout
    rows = _parse(out.splitlines())
    if not rows:
        sys.exit(f"7z listed no entries for {p} — corrupt, or a split part is missing.\n\n{out[:800]}")
    return rows


def from_dir(root: str) -> list[tuple[str, int]]:
    r = Path(root)
    return [(str(p.relative_to(r)), p.stat().st_size) for p in r.rglob("*") if p.is_file()]


def h(n: float) -> str:
    for u in "B KB MB GB TB".split():
        if n < 1024 or u == "TB":
            return f"{n:.1f}{u}"
        n /= 1024


def modality_of(path: str) -> str | None:
    return next((m for m in MODALITIES if f"/{m}/" in f"/{path}/"), None)


def probe(fp: Path) -> str:
    e = fp.suffix.lower()
    try:
        if e in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
            from PIL import Image
            import numpy as np
            with Image.open(fp) as im:
                a = np.asarray(im)
            return (f"image {im.size[0]}x{im.size[1]} mode={im.mode} "
                    f"shape={a.shape} dtype={a.dtype} range=[{a.min()},{a.max()}]")
        if e == ".csv":
            L = fp.read_text(errors="replace").splitlines()
            return f"{len(L)} rows | header: {L[0][:180]}\n          row1: {L[1][:180] if len(L)>1 else ''}"
        if e == ".json":
            o = json.loads(fp.read_text())
            if isinstance(o, dict):
                return f"dict keys={list(o)[:12]}\n          sample={json.dumps(o)[:300]}"
            return f"list len={len(o)} first={json.dumps(o[0])[:300] if o else ''}"
    except Exception as ex:
        return f"<probe failed: {type(ex).__name__}: {ex}>"
    return "<no probe>"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("archive", nargs="?")
    ap.add_argument("--dir")
    ap.add_argument("--manifest", default="manifest.csv")
    ap.add_argument("--probe", type=int, default=0, help="sample files to extract per (modality, ext)")
    a = ap.parse_args()

    raw = from_dir(a.dir) if a.dir else from_archive(a.archive) if a.archive else sys.exit("need archive or --dir")
    rows = [(p, s) for p, s in raw if not JUNK.search(p)]
    print(f"{len(raw):,} entries -> {len(rows):,} after dropping junk   {h(sum(s for _, s in rows))}\n")

    print("=" * 72, "\nPATH DEPTH SCHEMA")
    parts = [p.split("/") for p, _ in rows]
    for d in range(min(8, max(len(q) for q in parts))):
        vals = C.Counter(q[d] for q in parts if len(q) > d + 1)
        if vals:
            show = [v for v, _ in vals.most_common(10)]
            print(f"  L{d}: {len(vals):>5} distinct | {', '.join(show)}" + (" ..." if len(vals) > 10 else ""))

    print("\n" + "=" * 72, "\nPER-MODALITY BREAKDOWN")
    by_mod: dict[str, list] = C.defaultdict(list)
    for p, s in rows:
        by_mod[modality_of(p) or "<none>"].append((p, s))
    print(f"  {'modality':<14}{'files':>10}{'size':>11}  {'ext':<22}{'leaf dirs':>10}  files/dir min/med/max")
    for m in list(MODALITIES) + ["<none>"]:
        v = by_mod.get(m)
        if not v:
            continue
        ext = C.Counter(os.path.splitext(p)[1].lower() for p, _ in v)
        dirs = C.Counter(os.path.dirname(p) for p, _ in v)
        c = sorted(dirs.values())
        print(f"  {m:<14}{len(v):>10,}{h(sum(s for _, s in v)):>11}  "
              f"{', '.join(f'{e}:{n:,}' for e, n in ext.most_common(3)):<22}{len(dirs):>10}"
              f"  {c[0]}/{c[len(c)//2]}/{c[-1]}")

    print("\n" + "=" * 72, "\nONE COMPLETE CLIP")
    top = C.Counter(q[1] for q in parts if len(q) > 2)
    clip = next((k for k, _ in top.most_common(20) if not JUNK.search(k)), None)
    if clip:
        sel = sorted(p for p, _ in rows if f"/{clip}/" in f"/{p}/")
        print(f"  {clip}  ({len(sel)} files)")
        seen = C.Counter()
        for p in sel:
            k = os.path.dirname(p)
            seen[k] += 1
            if seen[k] <= 3:
                print(f"    {p}")
            elif seen[k] == 4:
                print(f"    {k}/  ... ({sum(1 for q in sel if os.path.dirname(q)==k)} files total)")

    print("\n" + "=" * 72, "\nACTION / USER IDS")
    act = sorted({s for q in parts for s in q if re.fullmatch(r"\d{1,2}_[A-Za-z].*", s)})
    if act:
        ids = sorted(int(s.split("_")[0]) for s in act)
        miss = sorted(set(range(40)) - set(ids))
        print(f"  {len(act)} action folders, ids {min(ids)}..{max(ids)}"
              f"{'  CONTIGUOUS 0-39' if not miss else f'  MISSING={miss}'}")
        print("   ", ", ".join(act[:8]), "...")
        aset = set(act)
        after = C.defaultdict(C.Counter)          # k -> values k segments after the action folder
        for q in parts:
            for i, seg in enumerate(q):
                if seg in aset:
                    for k in (1, 2):
                        if len(q) > i + k:
                            after[k][q[i + k]] += 1
                    break
        for k, lbl in ((1, "users"), (2, "trials")):
            v = sorted((x for x in after[k] if x.isdigit()), key=int)
            if v:
                print(f"  {lbl} (segment +{k} after action): {len(v)} distinct -> {v[:30]}")
    else:
        print("  none (test set is unlabeled — expected)")

    if a.probe and a.archive:
        print("\n" + "=" * 72, f"\nFILE PROBES ({a.probe} per modality+ext)")
        picks, seen = [], C.Counter()
        for m in MODALITIES:
            for p, _ in by_mod.get(m, []):
                k = (m, os.path.splitext(p)[1].lower())
                if seen[k] < a.probe:
                    seen[k] += 1
                    picks.append(p)
        with tempfile.TemporaryDirectory() as td:
            subprocess.run([sevenzip(), "x", a.archive, f"-o{td}", "-y", *picks],
                           capture_output=True, text=True)
            for p in picks:
                fp = Path(td) / p
                print(f"  [{modality_of(p)}] {p}")
                print(f"      {probe(fp) if fp.exists() else '<not extracted>'}")

    with open(a.manifest, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "size"])
        w.writerows(rows)
    print(f"\nmanifest -> {a.manifest}  ({len(rows):,} rows)")


if __name__ == "__main__":
    main()
