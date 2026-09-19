"""What is finished, what is in flight, what is left. Safe to run any time."""
from pathlib import Path

print(f"{'stage':<22} {'folds/seeds':<14} {'submission':<26} status")
print("-" * 78)
for tag, run, sub, n in [("B  hand160", "r36_hand160", "sub_r36_hand160.csv", 4),
                         ("A  boxfix", "r35_boxfix", "sub_r35_boxfix.csv", 4),
                         ("C  32frames", "r37_t32", "sub_r37_t32.csv", 4)]:
    got = len(list(Path("runs", run).glob("fold*.pt"))) if Path("runs", run).is_dir() else 0
    done = Path(sub).exists()
    print(f"{tag:<22} {got}/{n} folds{'':<4} {sub:<26} "
          f"{'COMPLETE' if done else ('in progress' if got else 'not started')}")

seeds = sum(Path(f"runs/r34_full46_s{i}/fold0.pt").exists() for i in range(4))
print(f"{'fullfit46':<22} {seeds}/4 seeds{'':<4} {'sub_r35_full46x4.csv':<26} "
      f"{'COMPLETE' if Path('sub_r35_full46x4.csv').exists() else ('in progress' if seeds else 'not started')}")
print("\nexisting submissions:")
for p in sorted(Path(".").glob("sub_*.csv")):
    print(f"  {p.name:<28} {p.stat().st_size:>8} bytes")
