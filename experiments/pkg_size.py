"""Measured deployed size of a checkpoint directory. python src/pkg_size.py <dir>"""
import glob, os, sys
d = sys.argv[1] if len(sys.argv) > 1 else "runs"
f = sorted(glob.glob(os.path.join(d, "fold*.pt")))
t = sum(os.path.getsize(x) for x in f)
print(f"  {len(f)} files, {t} bytes, {t/1048576:.2f} MiB, {t/1e6:.2f} MB")
print(f"  vs 100 MB cap: {'UNDER' if t < 100e6 else 'OVER'} decimal, "
      f"{'UNDER' if t < 100*1048576 else 'OVER'} binary")
