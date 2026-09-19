"""Refuse to train on cache_yolo unless it is a clean single-variable control.

Two assertions, both of which have already been broken once in this campaign:
  1. the HAND box is byte-identical to the motion-crop cache (build_one must pass
     motion_box, not the YOLO box, to hand_box),
  2. the PERSON box actually changed for a useful share of clips, otherwise the
     experiment is a no-op dressed as a treatment.
"""
import sys
import pandas as pd

new = pd.read_csv("data/cache_yolo/manifest.csv").set_index("clip_id")
old = pd.read_csv("data/cache_hand/manifest.csv").set_index("clip_id")
j = old[["box", "hand_box", "split"]].join(
    new[["box", "hand_box", "yolo"]], rsuffix="_new", how="inner")

hand_moved = (j.hand_box.fillna("") != j.hand_box_new.fillna("")).sum()
box_moved = (j.box.fillna("") != j.box_new.fillna("")).sum()
det = int(j.yolo.fillna(0).sum())

print(f"clips compared      : {len(j)}")
print(f"YOLO detections     : {det} ({det/len(j):.1%}), fallback {len(j)-det}")
print(f"PERSON boxes changed: {box_moved} ({box_moved/len(j):.1%})   <- the treatment")
print(f"HAND boxes changed  : {hand_moved}   <- MUST BE 0")
print(f"  of which test     : {(j[j.hand_box.fillna('') != j.hand_box_new.fillna('')].split == 'test').sum()}")

bad = []
if hand_moved:
    bad.append(f"hand stream moved on {hand_moved} clips; two variables changed at once")
if box_moved < 0.5 * len(j):
    bad.append(f"only {box_moved/len(j):.1%} of person boxes changed; treatment too weak to read")
print("\n" + ("REFUSED. " + "; ".join(bad) if bad else "OK. Clean single-variable control."))
sys.exit(1 if bad else 0)
