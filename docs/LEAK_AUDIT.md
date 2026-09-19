# Leakage audit

Run: `python src/leak_audit.py --cache data/cache --sub sub_r14_mix16.csv`
Date: 2026-09-04. Verdict: **the released data contains no exploitable leak.**

The public leaderboard is scored on 201 clips. Every score seen so far is an exact
201-denominator fraction, which confirms the count: 0.41791 = 84/201, 0.57711 = 116/201,
0.59701 = 120/201, and the top score 0.98507 = 198/201. Three errors out of 201, against a
dataset whose authors report 0.5638 cross-subject with ImageNet pretraining, contrastive
learning and long-tail class removal. Four tests were run to find the mechanism.

## A. Test-id ordering

Test ids are `SM_test_0001` to `SM_test_0405`, anonymised and sequential. If the organisers
generated them by walking source files grouped by class, the id order would leak the label.
Tested by comparing the mean absolute difference of metadata between neighbouring ids
against a 2000-draw random-permutation null.

| channel | observed | null | z |
|---|---|---|---|
| bytes | 29680.41 | 28980.00 +/- 1359.64 | +0.52 |
| n_Depth_Color | 15.04 | 15.42 +/- 0.42 | -0.90 |
| n_Thermal | 41.26 | 40.83 +/- 1.11 | +0.39 |
| n_IR | 15.04 | 15.43 +/- 0.42 | -0.92 |
| has_Thermal | 0.05 | 0.05 +/- 0.00 | +0.52 |
| has_Radar | 0.47 | 0.50 +/- 0.02 | -1.34 |

Every channel sits inside the null. The ordering is shuffled.

## B. Predicted-label clustering

If ids were class-sorted, a model that is right 58 percent of the time would produce visible
runs. Measured on `sub_r14_mix16.csv` sorted by id.

Neighbouring ids share a predicted label 0.0322 of the time, against 0.0506 expected under
random order (z = -1.69). The longest run is 3. No clustering.

## C. Metadata side channel

A gradient-boosted classifier on manifest metadata only (frame counts, modality presence,
byte size), evaluated cross-subject over 4 folds: **0.0652**. Chance is 0.0250 and the
majority-class rate is 0.1202. The classifier does not reach the majority-class baseline,
so metadata carries no usable class information. Single-feature variants all collapse to
0.1208, which is the majority class.

## D. Train/test content duplication

MD5 over rounded skeleton arrays: 2910 distinct signatures across 2931 train clips, 405
across 405 test clips. **Zero collisions.** No test clip is a copy of a train clip.

Incidental finding: 21 duplicate clips exist inside the training set. Worth removing, but
they sit within a subject and therefore do not cross fold boundaries.

## Conclusion

No identifier leak, no ordering leak, no metadata side channel, no duplication. If 198/201
is not a legitimate model, the mechanism is outside the released data, which leaves matching
the test clips against the labelled parent corpus. That matcher was not built here and will
not be.

The honest ceiling for this pipeline is what the pipeline earns.
