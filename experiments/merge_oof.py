"""Concatenate the per-fold oof_f*.csv that resumable training leaves behind, and
print the same report --all-folds would have printed."""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from train import lb_noise, oof_report

d = Path(sys.argv[1])
parts = sorted(d.glob("oof_f*.csv"))
if len(parts) != 4:
    sys.exit(f"REFUSED. found {len(parts)} per-fold oof files in {d}, expected 4: "
             f"{[p.name for p in parts]}")
oof = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
oof.to_csv(d / "oof.csv", index=False)

r = oof_report(oof[["user", "fold", "y_true", "y_pred"]])
print(f"\n{'='*60}\nOOF  pooled={r['pooled_acc']:.4f}  "
      f"fold={r['fold_mean']:.4f} +/- {r['fold_std']:.4f}")
print(f"per-fold: {r['per_fold']}")
print(f"per-user: {r['per_user']}")
print(f"worst user {r['worst_user']}  subject spread {r['subject_spread']:.3f}")
n = lb_noise(oof[["user", "y_true", "y_pred"]])
print(f"LB noise sd={n['std']:.4f} -> ignore LB deltas below {n['min_meaningful_delta']:.4f}")
print(f"\nwrote {d/'oof.csv'} ({len(oof)} rows)")
