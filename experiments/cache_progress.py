"""How far along are the three cache builds, and are they alive?"""
from pathlib import Path
for v in "ABC":
    d = Path(f"data/cache_fix{v}")
    n = len(list(d.rglob("*.npz"))) if d.exists() else 0
    log = Path(f"logs/cache{v}.log")
    tail = ""
    if log.exists() and log.stat().st_size:
        tail = log.read_text(errors="ignore").strip().splitlines()[-1][:70]
    print(f"cache_fix{v}: {n:>5}/3441 clips   log {log.stat().st_size if log.exists() else 0:>7}B  {tail}")
