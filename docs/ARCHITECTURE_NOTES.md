# CUHKX_MultiStreamNet: build notes and where the spec meets the measurements

Files: `CUHKX_MultiStreamNet.py`, `test_multistream.py`. Run the test with
`python test_multistream.py` from this directory. All checks pass on torch 2.13 CPU.

## Budget

| stream | params | GMAC (1 clip, 16 frames, 128x128) |
|---|---|---|
| Skeleton, CTR-GCN, 17 joints, 4 channels | 0.438 M | 0.046 |
| Visual, ShuffleNetV2 + TSM, DI (4ch) | 0.203 M | 0.342 |
| Visual, ShuffleNetV2 + TSM, HAND (4ch) | 0.203 M | 0.342 |
| Visual, ShuffleNetV2 + TSM, Thermal (3ch) | 0.202 M | 0.324 |
| Series, dilated TCN + SE, IMU (80x16) | 0.044 M | 0.0004 |
| Series, dilated TCN + SE, Radar (12x1), degenerate | 0.023 M | 0.00002 |
| Fusion gate + head | 0.804 M | |
| **Total, all six modalities** | **1.917 M** | **1.055** |

**7.3 MB fp32, 1.8 MB int8. The 10M parameter budget is not the binding constraint; it
was never close.** The measured-best subset (DI + HAND + Skeleton) is 1.251 M and 4.8 MB.

The current production model is 6.52 M and 26.1 MB int8, so this is 3.4x smaller. That is
not obviously good news: on 3,036 clips the constraint is data, not capacity.

## Verified behaviours

- Five modality-availability patterns forward correctly, including skeleton only and one
  visual stream only. An all-absent batch raises rather than returning garbage.
- **Absence is not zeros.** A zero-filled Thermal tensor and an absent Thermal produce
  different logits (mean delta 0.0268), which is the availability embedding doing its job.
  A model that treats a missing sensor as a zero image is learning "black frame" as a
  class feature.
- **Temporal sensitivity 0.0362 under frame permutation.** This project shipped an encoder
  that was permutation-invariant in time to 1.19e-07 and nobody noticed for weeks. The
  check is in the test file permanently.
- Gradient flows to exactly the present streams; absent encoders receive none.

## Where the spec was followed

CTR-GCN with per-channel topology refinement, TSM inside a ShuffleNetV2 backbone,
multi-scale dilated TCN with residuals and SE, residual modality gating with an
availability embedding, and LayerNorm / Dropout(0.4) / Linear(40) head.

## Where the spec was overruled, and by what

Six changes. Each one is forced by a measurement in `RESULTS.md`, not by preference.

**1. GroupNorm, never BatchNorm.** The split is cross-subject. BN running statistics are
estimated on training subjects and transfer badly to held-out ones. GN also decouples
normalisation from the PK sampler's batch composition.

**2. Skeleton takes 4 channels by default, not 3.** The raw 3 are root-centred metric 3D
with the root at exactly (0, 0, z) in every frame of the corpus. The 4th channel is
floor-referenced height over torso length. Feeding raw 3 channels reverts the stature fix
that was worth +0.0234 OOF, the second-largest measured gain in the project. The
constructor accepts 3 if you insist.

**3. M (bodies) is 1.** Supported in the tensor layout, folded into the batch and
max-pooled back, but every clip in this corpus is single-actor.

**4. The 1D stream degenerates honestly.** Radar is **12 numbers at one timestep** and IMU
has 2 source rows. A multi-scale dilated TCN over T=1 is arithmetic with no signal.
Dilations are clamped to the sequence length and `SeriesStream.degenerate` is set so a
caller cannot mistake an MLP for temporal modelling. The test prints it.

**5. Visual modalities are a constructor argument, defaulting to DI only.** Adding IR,
Thermal, IMU and Radar to Depth+Skeleton measured **-0.071** on 4 folds, with the mechanism
identified: 6-modality train loss reaches 0.80 against 2-modality 1.12 at equal epochs. It
fits harder and generalises worse. The spec asks for all six; the class supports all six;
the default does not use all six.

**6. Depth and IR share one 4-channel encoder ("DI") rather than two encoders.** They come
off the same 640x480 sensor with identical geometry. Thermal is a different camera with a
different field of view and must never share a crop box with them.

## What this architecture will not fix

The measured slope on this dataset is **+0.0102 accuracy per additional training subject**,
against a range of -0.05 to +0.01 for every architectural change tested. CTR-GCN, TSM and
gated fusion are all reasonable and none of them adds a subject.

Two of the four spec pillars have already been measured here in a different form:

- Gated / attention fusion: `--fusion attn` with a modality mask is already implemented in
  `src/model.py` and measured at concat minus 0.001. The availability embedding is the one
  genuinely new piece.
- Temporal shift: `TemporalConv` was added for the same reason and measured at -0.003
  wherever Skeleton is present, because Skeleton already carries the motion.

The two that are actually new are CTR-GCN, which replaces a generic sequence encoder with
a graph one and is the most likely of the four to pay, and the availability embedding.

**If you run one thing from this file, run the skeleton stream alone against the current
skeleton branch, on 4 folds, with everything else held fixed.** That isolates CTR-GCN,
which is the only untested idea here with a real prior behind it. Accept at +0.01 on the
4-fold mean.
