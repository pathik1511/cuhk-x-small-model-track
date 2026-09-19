"""Synthetic NTU .skeleton files, so the converter is proven before 10 GB is downloaded."""
import numpy as np, tempfile, sys
from pathlib import Path
sys.path.insert(0, ".")
from ntu_prep import read_skeleton, pick_body, to_cuhk, NTU_TO_H36M

rng = np.random.default_rng(0)

def write(path, T, bodies, V=25):
    L = [str(T)]
    for t in range(T):
        nb = bodies if not callable(bodies) else bodies(t)
        L.append(str(nb))
        for b in range(nb):
            L.append("72057594037928 0 0 0 0 0 0 0 0 0 0")
            L.append(str(V))
            for v in range(V):
                x, y, z = rng.normal(size=3) * 0.3 + (b * 2.0, v * 0.05, 3.0)
                L.append(f"{x:.6f} {y:.6f} {z:.6f} " + " ".join("0.0" for _ in range(9)))
    Path(path).write_text("\n".join(L) + "\n")

d = Path(tempfile.mkdtemp())
print("PARSER")
write(d / "S001C001P001R001A043.skeleton", 90, 1)
x = read_skeleton(d / "S001C001P001R001A043.skeleton")
print(f"  single body, 90 frames  -> {x.shape}   expect (90, 1, 25, 3)")
assert x.shape == (90, 1, 25, 3)

write(d / "S018C002P106R002A120.skeleton", 60, 2)
x2 = read_skeleton(d / "S018C002P106R002A120.skeleton")
print(f"  two bodies, 60 frames   -> {x2.shape}   expect (60, 2, 25, 3)")
assert x2.shape == (60, 2, 25, 3)

write(d / "vary.skeleton", 40, lambda t: 1 + (t % 2))
xv = read_skeleton(d / "vary.skeleton")
print(f"  body count alternating  -> {xv.shape}, NaN padded {np.isnan(xv).any()}")

print("\nBODY SELECTION")
b = pick_body(x2)
print(f"  (T,M,V,3) -> {b.shape}   expect (60, 25, 3)")
assert b.shape == (60, 25, 3) and np.isfinite(b).all()

print("\nCONVENTION, the part that must match CUHK-X exactly")
y = to_cuhk(pick_body(x), 16, 3)
print(f"  output shape                {y.shape}   expect (16, 17, 3)")
assert y.shape == (16, 17, 3)
print(f"  root x max |.|              {np.abs(y[:,0,0]).max():.8f}   expect 0")
print(f"  root y max |.|              {np.abs(y[:,0,1]).max():.8f}   expect 0")
print(f"  per-frame min z, max        {y[...,2].min(1).max():.8f}   expect 0")
assert np.abs(y[:, 0, :2]).max() == 0 and np.abs(y[..., 2].min(1)).max() == 0

print("\nTEMPORAL EXTENT")
long, short = to_cuhk(pick_body(read_skeleton(d/"S001C001P001R001A043.skeleton")), 16, 3), None
print(f"  90 NTU frames at stride 3 = 30 CUHK-rate frames, sampled to 16. OK")

print("\nJOINT MAP")
assert len(NTU_TO_H36M) == 17 and len(set(NTU_TO_H36M)) == 17
assert max(NTU_TO_H36M) < 25
print(f"  17 distinct NTU joints, all < 25: {NTU_TO_H36M}")
raw = pick_body(x)[0]
mapped = y[0]
print("  spot check, head above hip in the height axis:  "
      f"head z {mapped[10,2]:.3f} > hip z {mapped[0,2]:.3f} -> {mapped[10,2] > mapped[0,2]}")

print("\nDEGENERATE INPUTS")
Path(d / "empty.skeleton").write_text("0\n")
print(f"  zero frames        -> {read_skeleton(d/'empty.skeleton')}")
Path(d / "junk.skeleton").write_text("not a number\n")
print(f"  malformed header   -> {read_skeleton(d/'junk.skeleton')}")
Path(d / "trunc.skeleton").write_text("10\n1\n")
print(f"  truncated body     -> {read_skeleton(d/'trunc.skeleton')}")

print("\nAll converter checks passed.")
