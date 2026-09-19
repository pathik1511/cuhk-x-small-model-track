# CUHK-X Multimodal Human Activity Challenge, Small Model Track

40-class human activity recognition from synchronized depth, IR, thermal, skeleton, IMU and
radar, evaluated **cross-subject**: the test performers appear nowhere in training.

**Final result: rank 127 of 324, private leaderboard 0.71568 (146 of 204 clips).** The
pipeline is a single R(2+1)D-34 packed to 50.9 MB plus a YOLO11n person detector, 56.5 MB
total against a 100 MB cap, reproducible from the raw archives in one command.

| | |
|---|---|
| **private leaderboard (final)** | **0.71568**, rank 127/324 |
| public leaderboard | 0.69651 (140/201) |
| 4-fold cross-subject out-of-fold | 0.6295 (2,702 held-out clips) |
| dataset paper's own cross-subject baseline | 0.5638 |
| shipped size | 56,521,864 bytes (cap 100,000,000) |
| parameters | 63,697,175 |

The leaderboard figures and the cross-subject figure are not comparable. They come from
different subject populations, a different model (one averaged network against its separate
constituents) and different sample sizes.

**The most useful thing in this repository is not the score.** It is section
"The call I got wrong" below, where a correct measurement on 2,702 subject-disjoint clips was
overruled by a 201-clip public probe, and the private leaderboard later showed the measurement
had been right.

---

## What the final pipeline is

```
raw archives
  -> YOLO11n person detection, margin 1.35        src/yolo_box.py
  -> 4-channel cache: Depth_Color(3) + IR(1)      src/build_cache.py
     16 frames, 128x128, Kinetics normalization
  -> R(2+1)D-34, IG65M + Kinetics init            src/r2p1d.py
     fine-tuned per cross-subject fold            src/train_r2p1d.py
  -> weight-space average of 3 folds              src/soup.py
  -> int6 packing, per-output-channel scales      src/pack_int5.py
  -> flip TTA, 405-row submission                 src/predict_r2p1d.py
```

One command, raw archives to submission:

```bash
python inference.py /path/to/raw_archives submission.csv
```

`inference.py` asserts every postcondition and exits non-zero rather than continuing on bad
state. It was verified three times against the submitted CSV, each run removing more cached
state, the last with a full filesystem rescan. All three: **405/405 identical**.

### Two decisions that carried the result

**The backbone.** Initializing R(2+1)D-34 from the public IG65M + Kinetics-400 checkpoint was
worth roughly **+0.10 cross-subject** over a 6.5M-parameter CNN trained from scratch. It is the
only intervention in the entire campaign that moved the number outside the noise band. The
provenance is disclosed in full in `inference.py` and `docs/REPRODUCE.md`.

**Weight averaging instead of ensembling.** Three fp32 checkpoints do not fit the size cap.
Averaging three cross-subject folds tensor-by-tensor produced a single network that beat the
three-model ensemble it was made from on the public set, 140/201 against 137/201, at one third
the size.

Corrected after the private release: **on the larger private half the ensemble won**, 147/204
against 145. The claim that averaging acts as a regularizer was a public-set artifact. What
survives is the size argument: the soup ships under the cap and the ensemble does not.

---

## Experimentation and dead ends

Fourteen interventions were tested. One worked. This section is the actual content of the
project.

| # | what | code | outcome | why |
|---|---|---|---|---|
| 1 | SimCLR self-supervised pretraining | `experiments/pretrain.py` | null | 3,036 clips is far below the scale contrastive pretraining needs; the learned features were no better than random init after fine-tuning |
| 2 | Knowledge distillation from a 12-model teacher | `experiments/train_prekd.py`, `make_teacher*.py` | **-5 clips** | the teacher's advantage was ensemble variance reduction, which does not survive distillation into one student |
| 3 | NTU RGB+D 120 skeleton pretraining | `experiments/ntu_prep.py`, `pretrain_skel.py` | marginal | used to initialize 4 of 12 checkpoints; never isolated as a standalone gain |
| 4 | Derivative hand crop, 160 px hand crop | `experiments/probe_box.py` | null | the hand region at source resolution carries too few pixels; more crop does not create detail |
| 5 | `person_box` recovery fix | `experiments/check_boxfix.py` | +2 clips (predicted 8) | the prediction was made without measuring |
| 6 | Crop margin sweep 1.35 / 1.40 / 1.55 | `experiments/margin_curve.py`, `remargin.py` | **monotonically worse** | 0.5193 / 0.4950 / 0.4836 at 128 px, and null at 224 px. Extra context is noise to the classifier. At margin 1.55, 44% of boxes hit the frame clamp, so that arm was really testing "no crop" |
| 7 | YOLO person crop as an accuracy intervention | `src/yolo_box.py` | null | +0.016 over the motion-median crop, inside the noise band. Kept because it is deterministic, not because it scores better |
| 8 | All-18-subject refit at matched steps | `run_fullfit46.bat` | **-8 clips** | the CNN was already overfitting 3,036 clips; four more subjects added data it could not use |
| 9 | Stronger augmentation: resized crop, brightness, cutout | `experiments/train_r2p1d_aug.py` | null (0.6049 vs 0.6020) | train loss rose 0.7498 to 0.8129, so the regularization worked. Validation did not move. The model is limited by subject shift, not by memorization |
| 10 | BatchNorm recalibration after weight averaging | `experiments/bn_recal.py` | **139/201 vs 140/201** | the mechanism is real (54 of 405 predictions changed) but bought nothing. Correct about the defect, wrong about the number |
| 11 | Cross-family blend with the old hand/skeleton models | `experiments/blend.py`, `blend_submit.py` | **not a failure**, see below | +28 out of fold, -4 public, **+6 private**. Rejected on the wrong evidence |
| 12 | VideoMAE ViT-S/16 (Kinetics-400) | `experiments/vmae.py`, `train_vmae.py` | 0.5749 vs 0.6020 | fit the training set to the label-smoothing floor and transferred worse. IG65M is 65M videos; K400 is ~240K |
| 13 | 224 px resolution, both architectures | `experiments/train_vmae.py --arch r2p1d` | **-11 clips** at 2.5x cost | R(2+1)D-34's Kinetics pretraining uses 112x112 crops, so 128 is near-native and 224 is double |
| 14 | 4-pass TTA (adding temporal roll) | `experiments/oof_probs.py` | +7 clips, below the pre-set bar | not adopted |
| 15 | Test-time BatchNorm adaptation (alpha-BN) | `experiments/alpha_bn.py` | +38 clips OOF, bar was +40 | see below |

### The call I got wrong: #11, cross-family blending

Joining the video model's held-out predictions against the old CNN family's, on 2,702
subject-disjoint clips, found **221 clips the video model got wrong and the old family got
right**. A probability blend, with the weight chosen by a rule written before the grid was
run, gained **+28 clips out of fold**. It then scored **-4 clips** on the 201-clip public
leaderboard, and was rejected.

A mechanism was written to explain the public result: held-out folds are training subjects,
the test set is not, so complementarity measured out of fold overstates what survives on
unseen performers.

**The private leaderboard refuted that.** Final scores, all exact integers over 204:

| submission | public | private |
|---|---|---|
| **the rejected blend** | 0.67661 (worst) | **0.74019** (151/204, best) |
| 3-model ensemble | 0.68159 | 0.72058 (147) |
| 4-fold soup | 0.68656 | 0.71568 (146) |
| **the submitted soup** | **0.69651 (best)** | **0.71078** (145, worst) |

**The public ranking was inverted against the private one.** The best public score had the
worst private score. The rejected blend had the best private score of anything produced, worth
+6 clips over what shipped, which would have moved the final standing above rank 117.

Binomial noise on 201 clips at p = 0.70 has a standard deviation of **6.5 clips**. Nearly every
comparison this campaign settled on the public leaderboard was smaller than that.

> **When a well-powered subject-disjoint validation set and a small public probe disagree,
> believe the validation set.** A probe that cannot resolve the effect size being measured is
> not evidence against the measurement.

The aggravating detail is that this principle was written down at the time, in the same
message that then violated it.

### The near-miss worth reading: #15, test-time normalization

Mixing source and target BatchNorm statistics gained **+1.40 points** pooled, with a clean
interior optimum and the collapse at pure target statistics that the literature predicts. The
pre-registered bar was +1.50. It was not adopted, for three reasons that pointed the same way:
it missed the bar, the predicted mechanism signature (gains concentrating on the weakest
subjects) did not appear, and it makes accuracy depend on statistics computed from the
evaluation set, which degrades violently on small samples and is exactly the wrong property
for a live verification run.

---

## Four silent failures, and what they cost

Every one of these produced a number that looked like a measurement. **None of them made a run
fail.**

| what happened | how it surfaced |
|---|---|
| `yolo_box.py` read the wrong dict key, detected 0 of 3,441 clips, and wrote `fallback=1` for every one | a whole analysis section was built on it and had to be retracted |
| the int5 packer quantized the 40-way classifier head and stored BatchNorm statistics in fp16 | 20 of 405 predictions moved; caught by diffing packed against fp32 |
| `from_pretrained` returned VideoMAE with 36 randomly initialized attention biases and called it a warning | caught by running the pretrained classifier on real clips and seeing "belly dancing" for a person pouring a drink |
| `load_state_dict(..., strict=False)` into a wrapped model silently dropped all 69 BatchNorm layers | caught by an identity check that had to reproduce a known baseline exactly |

The rule this produced, and the reason the last two cost minutes instead of days: **a number in
a report should trace to a command that emitted it, and a loaded model should trace to a check
that exercised it.** A completed run is not evidence.

---

## Reproducing

```bash
pip install -r requirements.txt
python inference.py /path/to/raw_archives submission.csv
```

**The dataset is not in this repository and cannot be.** CUHK-X License v2.0 section 5 makes it
non-redistributable and the Competition Rules forbid making it available to non-participants.
NTU RGB+D 120, used for skeleton pretraining in part of the earlier campaign, is under a
separate ROSE Lab Data Use Agreement and is likewise excluded. Model checkpoints are excluded
for size.

7-Zip is required: the training set ships as a multi-part split zip that Python's `zipfile`
cannot read.

Full method, fold assignments, per-subject accuracy and quantization parity: `docs/REPRODUCE.md`.
The complete campaign record, including every retraction: `docs/campaign_summary.md`.

---

## Honest notes

- Final standing is **rank 127 of 324**. Top 15 advanced to verification; this entry did not.
- The shipped weight soup has **no clean held-out set**. Each fold model was validated on
  subjects the other two trained on, so after averaging no subject group is unseen. 0.69651 is
  a test-set reading, not a cross-validated estimate. The defensible subject-disjoint number is
  0.6425.
- Model selection between packed variants used leaderboard feedback on 201 public clips across
  eight submissions. That is disclosed, not presented as validation.
- The backbone is initialized from public IG65M + Kinetics weights. The source, the parameter
  count and the fact that three folds were averaged are stated in the opening screen of both
  `inference.py` and `docs/REPRODUCE.md`.
- Per-subject accuracy ranges from 0.456 to above 0.70. Any evaluation sample weighted toward
  the harder performers should be expected to score below the leaderboard figure.

## Licence

Code: Apache 2.0. Dataset: not included, see above.
