# CUHK-X Small Model Track: verification package

Submission: `sub_r42_soup3.csv`, public LB **0.69651**.

## Run it

```bash
python inference.py /path/to/raw_archives submission.csv
```

Raw competition archives in, 405-row submission CSV out. No cached intermediates, no
training artifacts, no test-specific state. Every stage asserts its postcondition and
exits non-zero rather than continuing on bad state.

Runtime: person detection ~6 min on a GPU, cache build ~12 min on 20 cores, inference
~3 min on a GPU and ~25 min on CPU. **No GPU is required.**

## What the model is

| | |
|---|---|
| architecture | R(2+1)D-34 (torchvision `VideoResNet`, `Conv2Plus1D`, blocks `[3,4,6,3]`) |
| parameters | **63,697,175** |
| initialization | **IG65M + Kinetics-400**, public checkpoint, see below |
| input | Depth_Color (3ch) + IR (1ch) as one 4-channel clip, 16 frames, 128x128 |
| normalization | Kinetics statistics; the 4th channel uses the mean of the three RGB constants |
| person crop | YOLO11n detection, margin 1.35, falls back to a motion-median box |
| what ships | **one network**, the weight-space average of three cross-subject folds |
| quantization | int6, symmetric, per-output-channel |
| TTA | 2-pass: identity and horizontal flip, probabilities averaged by `clip_id` |
| model bytes | `checkpoints/model.pth` **50,908,100** + `yolo11n.pt` **5,613,764** |
| total | **56,521,864 bytes** = 56.52 MB decimal, 53.90 MiB |
| vs 100 MB cap | UNDER on both the decimal and binary readings |

### Pretrained weights, stated plainly

The backbone is initialized from the public IG65M + Kinetics-400 R(2+1)D-34 checkpoint:

> https://github.com/moabitcoin/ig65m-pytorch/releases/download/v1.0.0/r2plus1d_34_clip32_ft_kinetics_from_ig65m-ade133f1.pth
> 255,061,555 bytes, loaded with `strict=True`.

> D. Ghadiyaram, M. Feiszli, D. Tran, X. Yan, H. Wang, D. Mahajan,
> "Large-scale weakly-supervised pre-training for video action recognition", CVPR 2019.

The detector is stock COCO-pretrained YOLO11n, used for person localisation only and
never trained or fine-tuned on competition data.

Whether an IG65M-initialized backbone satisfies the track's "no large pretrained
backbones" condition is the organizers' determination. It is disclosed here rather than
left to be discovered. No other external data was used.

## The weight soup

Four cross-subject folds were trained with an identical recipe. The three used in the
submission were averaged tensor-by-tensor into a single network (`src/soup.py`), which
is why one model ships rather than an ensemble. All folds fine-tune from the same
initialization, so they remain in a shared loss basin and the average is a valid network.

| fold | validation users | n | accuracy |
|---|---|---|---|
| 0 | 5, 6, 18, 24 | 701 | 0.6020 |
| 1 | 3, 7, 19, 22 | 730 | 0.6658 |
| 2 | 4, 9, 16, 20 | 675 | 0.6593 |
| 3 | 1, 8, 21, 23 | 596 | 0.5839 |
| **pooled, folds 0-2 (the shipped soup)** | | **2106** | **0.6425** |
| pooled, all four folds | | 2702 | 0.6295 |

**The soup itself has no clean held-out set.** Each fold model was validated on subjects
the other two trained on, so after averaging there is no subject group the averaged
network has not seen. **0.69651 is a test-set reading, not a cross-validated estimate.**
The defensible subject-disjoint number is **0.6425**.

Fold spread is driven by which subjects each fold holds out, not by model quality. Per
user accuracy is lowest for user 5 (0.456), 8 (0.544), 6 (0.567), 1 (0.575), 23 (0.589)
and 4 (0.594). Fold 0 holds out user 5 and fold 3 holds out three of those six, which
accounts for both being the low folds.

User 5 has 103 clips carrying Thermal only. The primary encoders see zeros for those.

## Fold assignment

`--seed` re-draws the subject fold split. A file named `fold0.pt` from a different seed
held out different people. Each run's true holdout is recorded in its `oof_f*.csv`; read
that rather than assuming filenames match. `src/cuhk_cv.py::subject_folds` is the single
source of truth. At seed 0, users 2 and 17 fall in no validation fold.

## Quantization is not a claim, it is measured

The fp32 soup and the int6 packed model were both run over all 405 test clips and
compared prediction by prediction:

| packing | bytes | predictions differing from fp32 |
|---|---|---|
| int5 | 43,078,900 | 10 of 405 |
| **int6** | **50,908,100** | **0 of 405** |

int6 ships. "0 of 405" means the packed model produces an identical argmax on every test
clip, not that quantization is lossless: the weights and logits still differ. The score being
defended is the score the shipped weights produce, not the score of weights that stayed on
the training machine. `src/pack_int5.py` packs and
unpacks; `inference.py` stage 4 unpacks the shipped file and predicts from it, so the
packed artifact is on the reproduction path rather than beside it.

Conv weights are quantized. The 40-way classifier head, BatchNorm statistics and every
tensor under 100,000 elements stay fp32, which costs 0.9 MB against a 100 MB cap.

## Reproduction runs

`inference.py` was run three times against `sub_r42_soup3.csv`, each removing more state:

| run | state removed first | result |
|---|---|---|
| 1 | nothing (warm) | 405/405 identical |
| 2 | `data/cache_yolo`, `yolo_boxes.csv` | 405/405 identical |
| 3 | also `data/scan_index.pkl` (full filesystem rescan) | 405/405 identical |

Run 3 is the path a verification host takes. YOLO11n returned exactly 3,290 detections
and 151 motion fallbacks on all three runs, so the crop is deterministic.

## Files

| path | role |
|---|---|
| `inference.py` | entry point, raw archives to CSV |
| `inference.sh` | thin wrapper for hosts that prefer a shell entry point |
| `checkpoints/model.pth` | **the shipped model**, int6 R(2+1)D-34 weight soup. Named per Rules 2.8a; despite the `.pth` extension it is an uncompressed `.npz` read by `src/pack_int5.py`, not by `torch.load` |
| `model/packed_i6.npz` | identical bytes under the campaign's own name |
| `yolo11n.pt` | **the shipped detector**, counted against the cap |
| `yolo_boxes.csv` | detector output, rebuilt by stage 2, kept as a cross-check |
| `src/yolo_box.py` | person boxes; exits non-zero if over half the clips fall back |
| `src/build_cache.py` | raw to cache, person crop, hand crop |
| `src/r2p1d.py` | R(2+1)D-34 loader, 4-channel stem, Caffe2 midplane deviation |
| `src/train_r2p1d.py` | fold training |
| `src/soup.py` | weight-space averaging of the fold checkpoints |
| `src/pack_int5.py` | int5/int6 pack and unpack |
| `src/predict_r2p1d.py` | test-time inference and submission assembly |
| `src/cuhk_cv.py` | `subject_folds`, the fold assignment |
| `src/merge_oof.py`, `src/paired.py` | out-of-fold pooling and paired comparison |
| `src/leak_audit.py` | four independent leakage tests |
| `requirements.txt` | pinned versions |

`runs/` and `data/` are excluded from the package (see `.packageignore`). They contain
~5 GB of fp32 training checkpoints and an 8 GB cache, none of which is the entry, and
all of which `inference.py` rebuilds or supersedes.

## Environment note

Built on **Python 3.14.4** with `torch==2.11.0+cu128` and `ultralytics==8.4.150`. Those
pins will not resolve on an older interpreter. The source contains no 3.14-specific
syntax, so any Python 3.10+ with compatible `torch`, `torchvision`, `ultralytics`,
`numpy`, `pandas`, `scipy` and `pillow` should run it. Stage 0 prints the versions it is
actually using.

7-Zip is required. The training set ships as a multi-part split zip (`HAR.z01`-`HAR.z08`
plus `HAR.zip`) which Python's `zipfile` cannot read.

Inference runs on CPU. No GPU is required.

## Expected accuracy range on unseen subjects

Stated in advance because Rules 2.8b treats a verification-run gap above 10% as
cheating. **This model's accuracy varies by roughly 8 points depending on which
subjects are held out**, and that variance is a property of the data, not of the
pipeline.

| held-out subjects | n | accuracy |
|---|---|---|
| 5, 6, 18, 24 | 701 | 0.6020 |
| 3, 7, 19, 22 | 730 | 0.6658 |
| 4, 9, 16, 20 | 675 | 0.6593 |
| 1, 8, 21, 23 | 596 | 0.5839 |

Per-subject accuracy is lowest for user 5 (0.456), 8 (0.544), 6 (0.567), 1 (0.575),
23 (0.589) and 4 (0.594), and highest above 0.70 for several others. **A fresh sample
weighted toward the harder subjects should be expected to score below the public
leaderboard figure**, and a sample of seen subjects should score above it. The
subject-disjoint estimate for the shipped soup's constituents is 0.6425; the public
figure is 0.69651 on 201 clips.

User 5 has 103 clips carrying Thermal only, where the primary encoders receive zeros.

## Integrity

No test labels were used in training in any form. No manual labelling of test samples.
Every subject-disjoint number reported here is out-of-fold on training subjects only.
`src/leak_audit.py` contains four independent leakage tests over the released data, all
negative.

Model selection between the packed variants used leaderboard feedback. The public score
covers **201** of the 405 test clips; the remaining 204 are private and were never probed.
Eight public submissions were made.

The cross-validated figure is 0.6425 and the public figure is 0.69651. **These two numbers
are not comparable and their difference does not measure anything.** They come from
different subject populations (unseen test performers against held-out training performers),
different models (the averaged network against its separate constituents) and different
sample sizes. Selection optimism from repeated probing is real and is disclosed here; it is
simply not quantified by that subtraction.
