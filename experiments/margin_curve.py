"""The margin curve on fold 0. Same subjects, same seed, same recipe throughout."""
import re
from pathlib import Path

ROWS = [("motion crop, 1.35", "logs/trainA.log", "=== A fold 0 ==="),
        ("YOLO 1.15", None, None),
        ("YOLO 1.35", "logs/trainY1.log", None),
        ("YOLO 1.40", "logs/train_y140.log", None),
        ("YOLO 1.55", "logs/train_y155.log", None)]
KNOWN = {"motion crop, 1.35": 0.5036, "YOLO 1.15": 0.4964, "YOLO 1.35": 0.5193}


def final_acc(path):
    p = Path(path) if path else None
    if p is None or not p.exists():
        return None
    m = re.findall(r"ep 60/60.*?val (\d\.\d+)", p.read_text(errors="ignore"))
    return float(m[-1]) if m else None


print(f"{'crop':<22} {'fold 0':>8}  {'vs motion':>10}")
print("-" * 44)
base = 0.5036
best, bestname = None, None
for name, log, _ in ROWS:
    a = KNOWN.get(name) or final_acc(log)
    if a is None:
        print(f"{name:<22} {'pending':>8}")
        continue
    print(f"{name:<22} {a:>8.4f}  {a-base:>+10.4f}")
    if name.startswith("YOLO") and (best is None or a > best):
        best, bestname = a, name
print()
if best:
    print(f"best so far: {bestname} at {best:.4f}")
    print("seed sd is 0.0105. A gap under ~0.011 between two margins is not resolvable")
    print("on one fold; pick the larger margin only if it wins by more than that.")
