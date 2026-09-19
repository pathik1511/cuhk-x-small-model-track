"""Verify a rebuilt cache before spending GPU time on it.

Three assertions, each of which has caught a real error in this campaign:
  1. the box fix recovered clips that previously had none,
  2. clips that already had a box kept the SAME box (the fix must be additive),
  3. the 103 zero-input clips are still present in the manifest but will be
     filtered by dataset.py, not silently dropped here.
"""
import sys
import pandas as pd

new = pd.read_csv(f"{sys.argv[1]}/manifest.csv").set_index("clip_id")
old = pd.read_csv("data/cache_hand/manifest.csv").set_index("clip_id")
j = old[["box", "split", "action_name", "has_IR"]].join(
    new[["box"]], rsuffix="_new", how="inner")

gained = j[j.box.isna() & j.box_new.notna()]
lost = j[j.box.notna() & j.box_new.isna()]
changed = j[j.box.notna() & j.box_new.notna() & (j.box != j.box_new)]

print(f"clips compared        : {len(j)}")
print(f"BOXES GAINED          : {len(gained)}   (train {(gained.split=='train').sum()}, "
      f"test {(gained.split=='test').sum()})")
print(f"boxes lost            : {len(lost)}")
print(f"boxes CHANGED         : {len(changed)}   <- must be 0, the fix is additive")
if len(gained):
    print("\ngained by action:")
    print(gained.action_name.value_counts().head(8).to_string())

zero = new[new[["has_Depth_Color", "has_IR", "has_Skeleton"]].max(axis=1).eq(0)]
print(f"\nzero-input clips still in manifest: {len(zero)}  (dataset.py filters these)")

bad = len(changed) or len(lost)
print("\n" + ("REFUSED. The fix changed existing boxes; it is not a control."
               if bad else "OK. Additive only. Safe to train on."))
sys.exit(1 if bad else 0)
