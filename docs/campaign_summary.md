# CUHK-X Small Model Track: campaign summary

Every method attempted, what it returned, and why. Written to be read cold by someone
picking this up with no context. Companion document: `RESULTS.md`, which organises the same
evidence by finding rather than by attempt.

**Final position: public LB 0.62189, OOF 0.5279 for the base recipe, 6.52M parameters,
6,597,019 bytes int8 per checkpoint measured on disk, trained from scratch.**
Exact parameter count **6,521,262**, fp32 payload 26,085,048 bytes, which confirms the
"26.1 MB" quoted through most of this campaign was always the FP32 size (section 15.4). The dataset
authors report 0.5638 cross-subject using ImageNet pretraining, contrastive learning and
long-tail class removal. **That 0.5638 is not a matched benchmark**: section 6.1.3 of the
preprint identifies it as an RGB-only cross-subject result averaged over five LOSO folds
with progressive changes to long-tail classes, contrastive learning and domain inclusion,
and the appendix gives 56.56% for the same thing. It is quoted here for scale only.

**Started at 0.41791. Finished at 0.62189. +41 clips of 201.**

> **Status of this document.** Sections 1 to 14 were written on 11 September and later
> sections supersede them where they conflict. If you read only one thing, read **section
> 18**: it closes the campaign with three null results and the summary table of what the
> whole effort was worth. Section 15 carries ensemble saturation and packaging; section 16
> the user-5 data defect; section 17 the resolution of the subject-count question.
>
> **Final submissions: `sub_r33_trim12.csv` (125 of 201) and `sub_r23_seeds16.csv` (124).**
> Campaign closed 13 September. Nothing after section 15 changed what ships.

---

## 1. Starting position

Inherited a working pipeline, three runs, one leaderboard score.

| run | config | OOF | LB |
|---|---|---|---|
| A | fold 0, Depth+Skeleton, 20 ep | 0.4051 | |
| B | fold 0, Depth+Skeleton, 60 ep | 0.4822 | 0.41791 |
| C | 4 folds, all 6 modalities, bs 32 | 0.4178 | |

The handoff carried several claims marked `[Certain]`. Four were wrong. See section 6.

---

## 2. Code audit, before running anything

Six defects found by reading the source. Each verified, not assumed.

| # | defect | evidence | cost |
|---|---|---|---|
| 1 | `FrameCNN` exactly permutation-invariant in time | permuting 16 frames changed the output by **1.19e-07** | Depth, IR and Thermal carried zero motion information |
| 2 | `--grl` a silent no-op | `b.get("user", d["y"]*0)`; the dataset never emitted `user` | the subject-adversarial head trained against an all-zeros target |
| 3 | every DataLoader worker drew an identical augmentation stream | rng created in `__init__`, forked unchanged | reduced augmentation diversity |
| 4 | `train.py --frames` did nothing except mis-shape absent-modality tensors | the dataset does not resample | collate failure |
| 5 | reported OOF was max-over-epochs on the fold being measured | code read | inflated every number by roughly 0.02 |
| 6 | masked-mean fusion diluted a strong modality by 1/K | hypothesis only | did not survive testing |

---

## 3. Data recovery

The project folder contained one file: the source zip. No cache, no checkpoints, no logs.
Rebuilt from scratch: 47.4 GB downloaded, 46 GB extracted with 7-Zip (the split archive
cannot be read by `unzip` or `Expand-Archive`), cache rebuilt and verified against the
prior session's numbers to four decimals on every modality-presence figure, class support
count and frame percentile.

---

## 4. Methods that worked

| method | effect | mechanism |
|---|---|---|
| **person crop** (temporal-median background subtraction, MAD threshold, largest connected component) | **+0.0697 LB, ~14 clips** | destroys the room-memorisation shortcut. The single largest gain |
| **ensembling, 1 to 16 models** | **~13 clips LB** | variance reduction. Paid enormously up to 16 models and nothing after |
| **hand stream + SADP** (same-action / different-performer supervised contrastive) | **+8 clips LB** at equal model count | positives are same-action, different-performer, so subject identity cannot reduce the loss |
| **skeleton 3D stature fix** | **+0.0234 OOF** | stopped feeding the model the subject's absolute height in metres on a cross-subject split |
| DI 4-channel, EMA 0.99, unconditional last epoch, skeleton joint+bone+motion streams, drop-path 0.1, 4-pass TTA | bundled, not separable | |

**Two of these share a mechanism signature worth recognising.** The skeleton fix moved
subject spread 0.384 to 0.317, worst user +0.056, block A / block B domain gap +0.119 to
+0.066, with the 5 worst users gaining +0.0495 while the 5 best gained -0.0039. SADP moved
spread 0.392 to 0.304, worst user 0.275 to 0.369, 5 worst +0.037 against 5 best +0.008.

Lifting the subjects the model was failing on while leaving the easy ones alone is the
signature of removing an identity shortcut. It is far more convincing than the mean, which
for the skeleton fix was only t = 2.27 on 3 df.

---

## 5. Methods that failed, with the reason for each

### 5.1 Architecture and fusion

| method | result | why |
|---|---|---|
| concat vs masked-mean fusion | -0.001 to -0.003 | the dilution hypothesis was simply wrong. concat equals mean at both 2 and 6 modalities |
| attention fusion with modality mask | -0.001 | same |
| temporal convolution | -0.003 where Skeleton is present | Skeleton already carried the motion. Untested without it |
| 192 vs 128 resolution | -0.002 | |
| all 6 modalities vs Depth+Skeleton | **-0.071** | overfitting, not fusion. 6-mod train loss 0.80 against 2-mod 1.12 at equal epochs: it fits harder and generalises worse. 9.10M params and four extra encoders on 3,036 clips |
| GRL subject-adversarial | n/a | was a silent no-op bug, and DomainBed evidence says skip it regardless |

Raising mod-dropout from 0.1 to 0.25 helps the 6-modality config by +0.019, which is
consistent with the overfitting diagnosis: the extra sensors behave like noise.

### 5.2 Self-supervised pretraining, -0.054, all four folds negative

SimCLR instance discrimination on DI frames. Top-1 retrieval reached **0.315 at epoch one**
against a chance rate of 0.005.

**Why it failed:** the task was trivially solvable from the start because on this data the
thing that makes two frames of the same clip recognisable is the **subject and the scene**.
Fine-tuning then exploited exactly those features: train loss 0.88 against 1.07, with 5
points worse generalisation across subjects. The metric that looked like success was
measuring the failure. Do not use `runs/ssl_di.pt`.

### 5.3 Hand localisation, three attempts, all negative

The original hand crop uses `|frame - temporal median|`. Replaced with a frame-to-frame
derivative `max_t |I(t+1) - I(t)|`, on the argument that the median statistic finds the
region displaced furthest over the clip (torso, hair) rather than the region moving fastest.

Paired against r19 on identical folds, identical everything except the hand cache:

| fold | median crop | derivative | delta | derivative + wrist prior | delta |
|---|---|---|---|---|---|
| 0 | 0.5207 | 0.4822 | -0.0385 | 0.5064 | -0.0143 |
| 1 | 0.5836 | 0.5822 | -0.0014 | 0.5575 | -0.0261 |
| 2 | 0.5807 | 0.5748 | -0.0059 | interrupted | |
| 3 | 0.4849 | 0.4748 | -0.0101 | interrupted | |
| mean | **0.5425** | **0.5285** | **-0.0140** | 0.5319 | -0.0202 |

**Why it failed: the capture is 10 fps.** Consecutive IR timestamps are exactly 100 ms
apart. At 10 fps `|I(t+1) - I(t)|` is not a velocity field: a hand at 1 m/s displaces 10 cm
between frames, so the difference image holds two nearly independent hand positions plus
everything else that moved. The entire argument assumed a frame rate this data does not
have, and checking it cost one directory listing.

The wrist prior moves the crop centre a median of 25.2 px and changes accuracy by nothing.
A visual audit of 14 fine-motor classes at 110 px thumbnails was **not predictive** and made
two variants look identical when they were not.

Code reverted; the derivative version is parked at `src/_attic/build_cache_deriv.py`.

### 5.4 Out-of-fold distillation, and the leak that would have faked +0.03

Averaging the OOF probability matrices of the 8 fold-consistent runs gives a teacher at
**0.5777 +/- 0.0095** against r19's 0.5448. It looks excellent. It is not usable.

**Why: the targets are out of fold with respect to the CLIP but not with respect to the
student's VALIDATION SUBJECTS.** Student fold 0 holds out users 5, 6, 18, 24. Its training
clip from user 3 gets its soft target from teacher fold 1, and teacher fold 1 trained on
users 5, 6, 18, 24. The target carries information derived from the subjects the student is
about to be scored on. Same failure shape as 5.2.

**The clean construction, leave two folds out.** A teacher is admissible for student fold F
on a clip from fold G only if it saw neither `val_users(F)` nor that clip's user. That is a
model trained on the users outside both folds; there are 6 such pairs and each serves two
student folds, so 6 runs cover all four. Five completed:

| pair | LTFO, 10 train users | r19, 14 train users | delta |
|---|---|---|---|
| (0,1) | 0.5262 | 0.5528 | -0.0266 |
| (0,2) | 0.4862 | 0.5501 | -0.0640 |
| (0,3) | 0.4410 | 0.5042 | -0.0632 |
| (1,3) | 0.5068 | 0.5392 | -0.0324 |
| (2,3) | 0.5169 | 0.5358 | -0.0189 |
| mean | **0.4954** | **0.5364** | **-0.0410** |

**The clean teacher is 0.0471 below the student it would teach.** The leaky version promised
+0.0329; the honest one delivers -0.0471. That 0.08 gap is the size of the artifact the
naive construction would have produced. No ensembling rescues it: for student fold F and a
clip in fold G there is exactly one admissible model.

**General result: with 18 subjects, no leak-free teacher on this dataset can beat the
student.** Any construction clean enough to be honest removes enough subjects to be weak.

### 5.5 NTU RGB+D 120 pretraining: zero transfer, but the best ensemble partner

Organisers confirmed external public datasets are allowed. Obtained NTU RGB+D 120 under a
ROSE Lab research licence: 114,480 clips, **106 subjects** against this corpus's 18, with
heavy label overlap (drink, eat, brush teeth, comb hair, writing, typing, phone call,
reading, sit down, stand up, and more).

Conversion (`src/ntu_prep.py`) handles five differences, each verified rather than assumed:
25-joint Kinect to 17-joint H3.6M, axis permutation, per-frame root-centring and
floor-referencing, 30 fps to 10 fps, and two bodies to one. Validation before spending GPU
time: **torso length 0.515 m against 0.516 m** and **motion magnitude 0.0276 against
0.0296**, which rule out unit and frame-rate errors respectively. An apparent axis inversion
was tested anatomically (shoulder and hip lines both land on axis 0 in both corpora at the
same ratio) and proved to be NTU's three-camera view diversity, not a mapping bug.

Pretraining succeeded on its own terms: **0.6214 on 120-way NTU cross-subject** with 12
performers held out, chance 0.0083, climbing 0.19 to 0.62 over 20 epochs rather than solving
instantly. Published comparison: about 0.70 for ST-GCN, 0.88 for CTR-GCN.

**Transfer to the target task is exactly zero.** Paired at seed 0, identical folds:

| fold | r19 | r27 NTU-init | delta |
|---|---|---|---|
| 0 | 0.5207 | 0.4793 | -0.0414 |
| 1 | 0.5836 | 0.6027 | +0.0191 |
| 2 | 0.5807 | 0.5867 | +0.0060 |
| 3 | 0.4849 | 0.5000 | +0.0151 |
| mean | **0.5425** | **0.5422** | **-0.0003** |

**Why so little:** in torso-normalised units NTU shoulders span 0.232 against this corpus's
0.319 and hips 0.111 against 0.174. NTU bodies are 27 to 36 percent narrower across
shoulders and hips relative to torso length, because Kinect places those joints differently
from a pose estimator. The encoder learned real motion structure in a body geometry that
does not match the target.

**What it did buy.** r27 is the best ensemble partner for r19 of every model in the project:

| partner | disagreement | 2-model average | gain over best solo | partner solo |
|---|---|---|---|---|
| **r27 NTU-init** | **0.333** | **0.5644** | **+0.0196** | 0.5440 |
| r17 hand w24 | 0.326 | 0.5581 | +0.0133 | 0.5296 |
| r21 hand deriv | 0.278 | 0.5577 | +0.0130 | 0.5307 |
| r16 hand | 0.319 | 0.5563 | +0.0115 | 0.5337 |
| r13 crop192 | 0.324 | 0.5551 | +0.0104 | 0.5218 |

It is the only partner whose solo accuracy matches r19's. Every other candidate is 0.52 to
0.53. NTU pretraining is a **diversity generator, not an accuracy improvement**.

On the leaderboard that diversity was worth **one clip** (16 models 0.61691, 32 models
0.62189). Real, and below what either instrument resolves.

---

## 6. Inherited claims that were wrong

| claim | truth | cost |
|---|---|---|
| "Skeleton is 2D pose + confidence, not 3D" | metric 3D; the model was being fed subject height on a cross-subject split | ~2.5 points, recovered |
| "LB approximately OOF - 0.06" | measured mean gap -0.018, and one run calibrated to +0.003. The -0.06 came from comparing best-epoch single-fold against the LB | weeks of mis-set expectations |
| "masked-mean dilutes a strong modality by 1/K" | concat equals mean at 2 and at 6 modalities | one wasted experiment arm |
| "Thermal is the strongest single modality at 92.57%" | self-contradictory: the same paper reports 0.5638 for the fusion of that modality plus five others. Adding information does not cost 36 points, so the 92.57 is a different protocol | nearly repriced the whole roadmap around Thermal |

**Rule: a confidence tag on an inherited claim is not evidence. Re-derive before building.**
The pattern repeated four times before it was named.

---

## 7. My own errors during the campaign

Recorded because the process matters more than any single number.

| # | error | how it surfaced | consequence |
|---|---|---|---|
| 1 | ensemble accumulated probabilities by row position while clip-ids were rebuilt per checkpoint | ensemble scored 0.43283, below its worst member, which ensembling cannot do | `data/cache` and `data/cache_crop` order the 405 test clips differently: **333 of 405 rows differ**. Fixed by keying on `clip_id`, verified against a cache with test order deliberately reversed |
| 2 | proposed self-supervised pretraining | -0.054, all four folds | my idea, my call. Retrieval 0.315 at epoch one should have been read as a red flag, not a success |
| 3 | dropped `--modalities` from two commands | two runs mislabelled for a week | r01/r05 silently ran 6-modality while named 2-modality |
| 4 | copy loop failed on a non-existent directory | r15 shipped 20 models while labelled 24 | always read the "N models ensembled" line |
| 5 | quoted "4-fold noise 0.0017" as an instrument | four seeds later measured sd 0.0105 | the 0.0017 came from three runs happening to agree. An sd from n=3 should never have been quoted to four decimals |
| 6 | called 0.5425 the recipe's value | four seeds give 0.5279 mean; 0.5425 is 1.39 sd above it | every "+0.01 against 0.5425" threshold was measured against a lucky draw |
| 7 | offered subject spread as a drop-the-transfer gate | spread ranges 0.304 to 0.454 across identical-recipe runs, sd 0.059 | under-powered by an order of magnitude. Retired |
| 8 | `ntu_prep.py` crashed on the first malformed file | `frames[0][0]` IndexError | a frame can hold zero bodies. A 113,945-file job must never die on one input. Fixed with per-file try/except |
| 9 | `--init` injected one tensor into the DI and HAND image encoders from a skeleton trunk | log read `DI:1/52, HAND:1/52` | a GroupNorm vector matched by name and shape. Negligible after 60 epochs, but unintended. Fixed |
| 10 | recommended a 32-model mix of the 0.617 and 0.577 recipes | retracted before it was run | it would have diluted the better recipe with equal numbers of the weaker one, exactly the r15 failure |
| 11 | gave a curl command writing to a directory never created, and an extract command before the download existed | two failed commands | check the prerequisites of a command before sending it |
| 12 | quoted "26.1 MB int8 per model" | the file on disk is 6,597,019 bytes | 26.1 MB was the FP32 size mislabelled as int8. Every packaging estimate built on it was 4x too large |
| 13 | `run_fullfit.bat` froze epochs, not optimizer steps | full-fit has 189 batches per epoch against the fold runs' 145, so 60 epochs trained 11,340 steps against 8,700 | a 30% step increase inside a comparison meant to isolate subject count. This document already recorded the same trap twice (run C's batch-size confound, r19's PK steps pinned to 145) and I broke the rule anyway. Re-run is 46 epochs = 8,694 steps |
| 14 | inserted a helper between `@torch.no_grad()` and `def predict` | `RuntimeError: Can't call numpy() on Tensor that requires grad` | the decorator silently rebound onto the helper. Never insert a definition directly above a decorated function |

---

## 8. Leakage audit: the released data is clean

The top public score is 0.98507, which is 198/201. Four tests. Script `src/leak_audit.py`,
detail in `LEAK_AUDIT.md`.

| test | result |
|---|---|
| test-id ordering, 6 metadata channels vs a 2000-draw permutation null | worst z -1.34, inside the null. Shuffled |
| predicted-label clustering by id order | 0.0322 against 0.0506 expected, z -1.69, longest run 3. None |
| metadata-only classifier, cross-subject | 0.0652 against a 0.1202 majority-class rate. Below baseline |
| train/test duplication, MD5 over rounded skeletons | 2910 signatures across 2931 train clips, 405 across 405 test. **Zero collisions** |

No identifier leak, no ordering leak, no metadata side channel, no duplication. If 198/201
is not a real model, the mechanism is outside the released data, which leaves matching the
test clips against the labelled parent corpus. That matcher was not built and will not be.

---

## 9. Data facts that constrain what is possible

Each of these killed a plan.

- **No camera translation in the skeleton.** The root joint is exactly `(0, 0, z)` in 3000
  of 3000 frames sampled from 86,050, and minimum z is exactly 0 in all of them. No
  intrinsics can be fitted and no joint can be projected to a pixel. Any plan starting
  "project the wrist into the depth frame" is dead here.
- **`keypoint_scores` is 1.0 everywhere.** The raw JSON carries a confidence field that the
  cache builder discards. Nothing was being lost.
- **All modalities are temporally aligned.** Motion-energy cross-correlation against Depth
  over lags -4 to +4 on 400 clips: every modality peaks at exactly 0.
- **Thermal is looking elsewhere.** It peaks at lag 0 but only 125 of 359 clips put their
  per-clip argmax there, against 345 of 380 for IR. The FOV mismatch showing up in a
  temporal statistic.
- **The capture is 10 fps.** Timestamps 100 ms apart. This kills velocity-based reasoning.
- **18 users, not 16.** Users 2 and 17 appear in no validation fold at seed 0. When seeds 1
  to 3 finally scored them they returned 0.5987 against 0.5146 for everyone else: the two
  easiest subjects, invisible to every earlier measurement.
- **The two user blocks are different domains.** Block A (1-9) and block B (16-24) differed
  by +0.119 accuracy before the skeleton fix, +0.066 after.
- **21 duplicate clips inside the training set.** Within-subject, so no fold leak.

---

## 10. Subject count as the binding constraint. RETRACTED, now unresolved

**This section previously asserted +0.0102 of accuracy per additional training subject as
the finding that explained everything else. That number is withdrawn.** It is left here in
full because the reasoning that produced it is the more useful record.

The claim came from two points, identical clips, both scoring unseen subjects:

| training subjects | accuracy |
|---|---|
| 10 (leave-two-folds-out models) | 0.4954 |
| 14 (standard 4-fold models) | 0.5364 |

0.0410 over 4 subjects = +0.0102 each, and the curve appeared not to be flattening.

**Why it does not hold.** Three separate problems, any one of which is fatal:

1. **Two points cannot establish a slope.** A line through two measurements has zero
   residual degrees of freedom. Nothing about flattening can be read from it.
2. **The gap is inside the noise band.** Four seeds of the identical recipe measure sd
   0.0105. A 0.041 difference between two single constructions is under 4 sd, and the LTFO
   models differ from the 4-fold models in more than subject count.
3. **The direct test came back the wrong way.** `r32_full4` trained on all 18 subjects and
   scored **0.56716 on the leaderboard against `r19_solo`'s 0.59701** at 14 subjects, same
   model count, same recipe. Four extra subjects lost 6 clips.

**And the direct test is itself confounded.** `run_fullfit.bat` froze 60 epochs rather than
the step count. Full-fit has 189 batches per epoch against the fold runs' 145, so it trained
11,340 optimizer steps against 8,700, a 30% increase, on a recipe whose schedule was tuned
at 8,700. The all-18 run is therefore both more-subjects and more-steps, and the two effects
cannot be separated from this pair.

**RESOLVED IN SECTION 17.** `run_fullfit46.bat` ran: all 18 subjects at 8,694 matched steps
scored **0.55721, 112 of 201, against the control's 120**. The slope is not merely
unsupported, it is contradicted. Read section 17 before citing anything in this section.

**What survives the retraction.** Every architectural change tested returned between -0.05
and +0.01, and no intervention of any kind beat the base recipe by more than the noise band.
That is an observation about architecture, not evidence for the subject hypothesis. The
0.71144 plateau remains out of reach from scratch for the plain reason that it is IG-65M and
Kinetics, 65 million videos of diversity bought elsewhere, and NTU pretraining failed for
the reason measured directly in section 5: 106 extra subjects in the wrong body geometry are
not 106 extra subjects in the right one.

---

## 11. Leaderboard history

| submission | models | recipe | LB | clips |
|---|---|---|---|---|
| single fold, uncropped | 1 | Depth + Skeleton | 0.41791 | 84 |
| r07 | 4 | + skeleton 3D fix | 0.47263 | 95 |
| r06 | 4 | | 0.49253 | 99 |
| r09 | 8 | + TTA | 0.50746 | 102 |
| r08 | 4 | **+ person crop** | 0.54228 | 109 |
| r11 | 12 | crop, 3 seeds | 0.57213 | 115 |
| r14 | 16 | crop 128 x3 + 192 | 0.57711 | 116 |
| r15 | 20 | + a weaker member | 0.56218 | 113 |
| r12 | 20 | positional-accumulation bug | 0.43283 | 87 |
| r30_kd4 | 4 | distilled from in-fold teachers | 0.57213 | 115 |
| r32_full4 | 4 | all 18 subjects, steps unmatched | 0.56716 | 114 |
| r35_full46x4 | 4 | all 18 subjects, **steps matched** | 0.55721 | 112 |
| r35_boxfix | 4 | person_box fallback | not submitted | OOF wash, -15 |
| r36_hand160 | 1 fold | hand crop 96 -> 160 px | not submitted | stopped, fold 0 -0.010 |
| r19_solo | 4 | **base recipe, 1 seed, the control** | 0.59701 | 120 |
| r23 | 16 | **hand + SADP, 4 seeds** | 0.61691 | 124 |
| **r33_trim12** | **12** | **r29 minus its 20 weakest** | **0.62189** | **125** |
| r29 | 32 | + 16 NTU-init | 0.62189 | 125 |

Attribution: person crop ~14 clips, ensembling 1 to 16 ~13, hand + SADP ~8, skeleton 3D fix
2 to 5, NTU-init 1, ensembling 16 to 32 zero.

**Final submissions:** `sub_r33_trim12.csv` at 0.62189 and `sub_r23_seeds16.csv` at 0.61691.
`r33_trim12` replaces `r29_final32` as final #1: identical leaderboard score, identical
predictions on the public subset, and 79,164,228 bytes against 211,104,608. See section 15.

---

## 12. Open, not pursued

- **CTR-GCN** on the skeleton branch, implemented and unit-tested in
  `src/CUHKX_MultiStreamNet.py` but never trained. The only untested idea with a real prior.
- **V-REx** (penalise the variance of per-subject risk) in place of the deleted GRL.
- **Thermal per-block affine alignment** by maximising normalised mutual information between
  the depth motion mask and the thermal gradient magnitude. Repriced downward once the
  92.57% claim collapsed.
- Supcon weight sweep, 0.2 and 1.0 against 0.5.
- LTFO(1,2), the sixth teacher. Not needed, the conclusion is in.
- ~~`run_fullfit46.bat`~~ **DONE. Section 17: 112 of 201, worse than the control.**
- ~~hand crop at 160 px~~ **DONE. Section 18.2: fold 0 came back 0.01 lower. Stopped.**
- ~~32-frame cache~~ **DROPPED. Section 18.3: OOM on 12 GB, and the smaller hypothesis.**
- **STILL OPEN AND THE ONLY ITEMS THAT CARRY RISK:**
  - a written record of the organizers' permission for AI-assisted development. The rules
    page still reads "No closed-source APIs or LLMs for development" and verification for
    the top 15 teams is **18 September, 23:59 UTC** (verified on the live challenge page
    13 September; the 22nd was wrong). A verbal yes is not a record.
  - `inference.sh`: raw archives to submission in one script, pinned versions, NTU licence
    and citation recorded, seed and fold assignment written down.
- **Reproducibility package**: raw archives to submission in one script, pinned versions,
  NTU licence and citation recorded, seed and fold assignment written down.

---

## 13. Rules of engagement, each earned by getting it wrong first

1. A `[Certain]` tag on an inherited claim is not evidence. Re-derive it.
2. Check the data before writing the patch. The 10 fps that killed the derivative hand crop
   cost one directory listing to discover, after the code was written.
3. Verify a control is a control. Person boxes were byte-identical across three hand caches,
   which is what made that A/B readable.
4. Prefer paired comparisons: same folds, one change. Paired resolves about 0.005, unpaired
   about 0.02.
5. Never select a subset on the same data you then report. A 255-subset teacher search
   returned +0.0008 over taking everything, inside the standard error.
6. Ask what a metric could be measuring other than what you want. Retrieval 0.315 at epoch
   one, and teacher OOF 0.5777, were both real numbers measuring the wrong thing.
7. Assert the property you rely on, in code, and let it crash. `make_teacher_ltfo.py`
   refuses to build rather than quietly leak.
8. Visual audits do not resolve small effects. 110 px thumbnails hid a 25 px median shift.
9. Read the count lines: "N models ensembled", "n=", "clips per user".
10. Believe a change when it moves the mean AND the subject spread AND the worst user AND
    the domain gap the way the mechanism predicts. When it moves only the mean, do not.
11. A batch job over 100,000 files must never die on one input.
12. An estimate of noise from three runs is not an instrument.
13. Freeze optimizer steps, not epochs. Changing the training-set size changes batches per
    epoch, and a comparison meant to isolate subject count silently becomes a comparison of
    subject count and schedule length.
14. A slope through two points is not a slope. It has zero residual degrees of freedom and
    cannot say anything about curvature.
15. Check where a curve flattens before paying for the tail. Members 13 to 32 of the final
    ensemble bought nothing, and finding that out cost less than training them did.

---

## 14. Error analysis, and the artifact bundle for anyone auditing this

Measured on the 5-model seed-0 ensemble (r19, r27, r16, r17, r21) over 2,702 OOF clips,
accuracy 0.5807, **1,133 errors**. Not estimated, not inferred from the leaderboard.

### Where the errors actually are

| true cluster | to object | to posture | to other | total |
|---|---|---|---|---|
| object / fine-motor | **552** | 39 | 118 | 709 (62.6%) |
| posture / gross motion | 21 | 78 | 70 | 169 (14.9%) |
| other | 105 | 41 | 109 | 255 (22.5%) |

**Object-versus-object confusion is 552 errors, 48.7% of the total.** That is the single
largest addressable block, and it is the category the PoseConv3D RGB results show being
resolved by appearance information the skeleton cannot carry.

**A perfect object discriminator fixes at most 37 of the 76 public-subset errors.**
552/1133 = 48.7% of 76. **CORRECTED from 41**: the original scaled 1,133 OOF errors down to
201 clips (84.3) and took 48.7% of that, instead of using the 76 errors the public score
actually shows. A target of 191/201 needs 66.

Both numbers are extrapolations from a 5-model OOF ensemble to a different deployed
ensemble on different subjects. Treat 37 as an order of magnitude, not a recoverable-error
estimate.

### The errors are diffuse, which rules out pair-targeted fixes

373 distinct confusion pairs, 156 of them occurring exactly once. The top 15 pairs cover
only 20% of errors. The largest single pair is Walk to Massage_oneself at 30 clips, 2.6%.

| count | share | confusion |
|---|---|---|
| 30 | 2.6% | Walk to Massage_oneself |
| 18 | 1.6% | Read_documents to Turn_pages |
| 18 | 1.6% | Turn_pages to Read_documents |
| 16 | 1.4% | Eat_food to Drink_water |
| 15 | 1.3% | Pour_drinks to Take_and_use_tableware |
| 15 | 1.3% | Stand_up to Sit_down |
| 14 | 1.2% | Sweep_the_floor to Mop_the_floor |
| 14 | 1.2% | Stir_drinks to Peel_fruits |

Published cross-dataset results that report 80 to 90 percent reductions on a named
confusion pair work because that corpus concentrates its errors into a few large pairs.
This one does not. Here a pair-targeted fix pays one or two clips.

### The ceiling on every post-hoc method

| | coverage |
|---|---|
| top-1 | 0.5807 |
| top-3 | 0.7576 |
| top-5 | 0.8205 |
| **top-10** | **0.8834** |

**No fusion, calibration, gating, reranking, member selection or distillation over this
model family can exceed 0.8834**, because 11.7% of the time the true label is not in the
ensemble's top ten. Reaching 0.95 requires a representation that puts the truth in its
top-1, not a better way to choose among these.

Oracle for scale: if a perfect selector always picked the correct member, 0.6999 against
the ensemble's 0.5807. Nondeployable; it bounds member selection only, and soft averaging
can pick a class no member had at top-1, so it is not a strict bound on fusion.

### Artifact bundle

Exported to `analysis/` so an external reviewer never has to ask for it again. **Contains
training-subject predictions only. No test clips, no test labels.** Safe to share.

| file | contents |
|---|---|
| `analysis/oof_ensemble5.csv` | 2,702 rows: clip_id, user, fold, y_true, y_pred, p0..p39 (5-model mean), plus each member's top-1 |
| `analysis/oof_<run>.csv` | the five members' individual OOF probability matrices |
| `analysis/class_map.csv` | action_id to action_name, with training clip counts per class |

Code for reproducing every number above: `src/train.py` (training, ensembling, prediction),
`src/dataset.py` (cache reading, skeleton normalisation, augmentation), `src/model.py`
(FusionNet, encoders, `budget()`), `src/build_cache.py` (raw to cache, person and hand
crops), `src/cuhk_cv.py` (`subject_folds`, the fold assignment every provenance check uses),
`src/ntu_prep.py` and `src/pretrain_skel.py` (NTU conversion and pretraining),
`src/make_teacher_infold.py` (clean in-fold teacher targets), `src/leak_audit.py`.

**Fold provenance warning for anyone reusing these:** `--seed` re-draws the fold
assignment. A file named `fold0.pt` from a different seed held out different people. Read
each run's real holdout from its `oof.csv` before treating two runs as comparable.

---

## 15. The last day: saturation, distillation, packaging, and what the leaderboard said

Written after sections 1 to 14. Where it conflicts with them, this section wins.

### 15.1 Ensembling saturates at twelve models

`r29_final32` scored 0.62189 with 32 checkpoints. `r33_trim12` scored **0.62189 with 12**,
the same 125 clips, after dropping the 20 members with the lowest individual OOF accuracy.

**Correction.** This section previously claimed the two submissions "agree on every one of
the 201 public clips." They do not. Diffed row by row, **they differ on 31 of 405 test
clips** and tie at 125 only because the disagreements cancel. Equal accuracy is not equal
predictions, and the earlier sentence asserted a diff that was never run.

| members | LB | clips |
|---|---|---|
| 1 | 0.41791 | 84 |
| 4 | 0.59701 | 120 |
| 16 | 0.61691 | 124 |
| **12 (trimmed)** | **0.62189** | **125** |
| 32 | 0.62189 | 125 |

Four to twelve is worth 5 clips. Twelve to thirty-two is worth zero. Every further
checkpoint after the twelfth was disk space and inference time spent on nothing.

**Why this matters beyond the score.** The submission size cap is 100 MB. At 6,597,019
bytes per int8 checkpoint, 32 models is 211,104,608 bytes and illegal under any reading of
the cap; 12 models is **79,164,228 bytes**, legal under both the decimal (100,000,000) and
binary (104,857,600) readings. The trim was forced by packaging and turned out to cost
nothing, which is the cheapest kind of constraint.

The rule this replaces: "more members is never worse." True, but it stops being better, and
the flat region starts earlier than the 32 this campaign spent GPU time reaching.

### 15.2 Distillation cost five clips

Knowledge distillation from in-fold teachers, `L = (1-a)*CE + a*T^2*KL`, teacher
probabilities re-softened as `softmax(log p / T)`:

| | OOF | LB | clips |
|---|---|---|---|
| base recipe, 4 models (`r19_solo`) | 0.5279 | **0.59701** | 120 |
| distilled, 4 models (`r30_kd4`) | 0.5296 | **0.57213** | 115 |

**+0.0017 on OOF, -0.0249 on the leaderboard.** The OOF gain is a sixth of one seed's
standard deviation, which is to say it is nothing. The leaderboard loss is 5 clips, 0.7 sd
of the 201-clip binomial. Neither number is individually decisive and together they say the
same thing: distillation bought nothing here and plausibly hurt.

**The teacher construction was the hard part and it was gotten right.** Three attempts:

1. `make_teacher.py`, cross-fold OOF averaging. **Leaks.** Its targets were produced by
   models that saw the student's validation subjects. Carries a warning docstring and was
   not used.
2. `make_teacher_ltfo.py`, leave-two-folds-out. Legitimate but weak, because a teacher
   trained on 10 subjects is worse than the student it is teaching. Three assertions in
   code: predicted users lie inside the holdout, no clip receives two teachers, no teacher
   covers the student's holdout.
3. `make_teacher_infold.py`, the correct one. Teachers train on exactly `train_subjects(F)`
   and produce in-sample targets on that same set, which leaks nothing about fold F's
   holdout. Provenance is asserted rather than assumed: `check_provenance` reads each run's
   real holdout from its `oof.csv` and refuses to build if it differs from the student's.

The earlier conclusion that "no leak-free teacher can beat the student" was wrong. In-fold
teachers are legitimate and they do beat the student on their training subjects. They still
did not help.

### 15.3 The all-18 refit lost six clips, and the experiment was broken

Covered in the rewritten section 10. Short version: `r32_full4` at 18 subjects scored
0.56716 against `r19_solo`'s 0.59701 at 14, but trained 30% more optimizer steps because
the batch script froze epochs rather than steps. The comparison is unusable in either
direction. `run_fullfit46.bat` repeats it step-matched.

### 15.4 Packaging, measured rather than estimated

| | bytes | MB (decimal) | MiB |
|---|---|---|---|
| one int8 checkpoint | 6,597,019 | 6.60 | 6.29 |
| 12 models (`r33_trim12`) | **79,164,228** | 79.16 | 75.50 |
| 32 models (`r29_final32`) | 211,104,608 | 211.10 | 201.32 |

Measured with `src/pkg_size.py`, which exists because an inline `py -c "..."` inside a `.bat`
was mangled by cmd.exe quoting. The 26.1 MB figure quoted through most of this campaign was
the FP32 size mislabelled as int8, and it inflated every packaging estimate by 4x.

Still open: whether the 100 MB cap is per-checkpoint or aggregate. Under either reading
`r33_trim12` is legal, so this now affects only final #2 and the verification package.

### 15.5 What the whole campaign is worth, stated plainly

+41 clips of 201 over the starting submission. Of that, **27 came from two changes**: the
person crop (~14) and ensembling from one model to twelve (~13). Hand crop plus SADP added
~8, the skeleton 3D fix 2 to 5, NTU initialisation 1, and everything else, every
architectural variant, every distillation scheme, every transfer idea, returned zero or
negative inside the noise band.

The honest summary is that this problem rewarded data handling and averaging, and punished
cleverness. The measured ceiling on any post-hoc method over this model family is 0.8834
(section 14, top-10 coverage), and the measured error structure, 373 confusion pairs with
the largest at 2.6%, rules out targeted fixes. Reaching 0.95 needs a different
representation, not a better way to combine these.

### 15.6 Still owed before submission closes

1. `run_fullfit46.bat`, four seeds, ~2.7 h, submitted against the 120/201 control.
2. A written record of the organizers' permission for AI-assisted development. The rules
   page still reads "No closed-source APIs or LLMs for development" and a verification stage
   exists, with the verification package due **18 September, 23:59 UTC**.
3. The ruling on the 100 MB cap, per-checkpoint or aggregate.
4. `inference.sh`: raw archives to submission in one script, pinned versions, NTU licence
   and citation recorded, seed and fold assignment written down.
5. Fix the log line reading `legacy 10-channel skeleton checkpoint`. Ten channels is the
   `--streams` joint+bone+motion representation, not a legacy format, and the message will
   mislead anyone reading the verification logs.

---

## 16. The defect that the per-subject numbers were pointing at

Found by querying `analysis/oof_ensemble5.csv` and `data/cache_hand/manifest.csv` directly,
after the logs recorded "one full-fit model reached only 0.556 training accuracy on user 5"
and left it unexplained.

### 16.1 The signature

User 5's out-of-fold accuracy is 0.3500, the worst of 16 subjects and 0.15 below the next
worst. Removing class composition makes it worse, not better: against the per-class global
rates, user 5 underperforms its own class mix by **-0.2390**, twice the next subject.

The deficit is not spread across classes. It is four classes, all gross whole-body motion,
all at exactly zero:

| class | user 5 | n | all subjects |
|---|---|---|---|
| Walk | **0.000** | 30 | 0.863 |
| Stand_up | **0.000** | 9 | 0.571 |
| Lie_down | **0.000** | 6 | 0.806 |
| Do_stretching_exercises | **0.000** | 3 | 0.718 |

**All 48 clips were predicted `Massage_oneself`. All five ensemble members agreed on all 48,
unanimously.** r19_sadp, r27_ntu, r16_hand_s0, r17_hand_w24 and r21_hand3_ungated differ in
seed, modality set and initialisation, and they do not agree like this on genuinely hard
input. Walk accuracy for the other fifteen subjects runs 0.762 to 1.000.

This also dissolves the largest confusion pair in section 14. "Walk to Massage_oneself, 30
clips, the biggest pair in the dataset" is **30 clips from user 5 and zero from anyone
else**. It was never a confusion pair. It was one subject's broken input.

### 16.2 The cause, measured

`data/cache_hand/manifest.csv`, `has_*` columns. User 5's 30 Walk clips:

| modality | present |
|---|---|
| Depth_Color | **0 of 30** |
| IR | **0 of 30** |
| Skeleton | **0 of 30** |
| IMU | **0 of 30** |
| Radar | **0 of 30** |
| Thermal | 30 of 30 |

Confirmed at the filesystem: `data/extracted/HAR/data/IR/36_Walk/` contains 17 user
directories and **user5 is not one of them**. Thermal has user5. The others do not.

**103 training clips carry Thermal and nothing else.** Every one is 786,700 bytes on disk,
the same number, because every one contains a single modality. User 5 owns 60 of them; the
rest spread over 12 subjects. By action: Walk 30, Massage_oneself 12, Stand_up 10,
Read_documents 9, Lie_down 6, Wipe_hands 5.

`src/dataset.py` does not read the `has_*` columns. It takes the manifest as given. For
these 103 clips every primary encoder receives zeros, and the network learns a mapping from
all-zeros to whichever class the optimiser settles on. `Massage_oneself` is that class.
**3.4% of the training set is teaching a degenerate constant.**

### 16.3 A second, separate defect

214 training clips have no person box. 103 of them have no images to compute one from. The
other **111 have full image data and `person_box` returned `None` anyway.**

`person_box` thresholds the motion image at `max(median + 4*MAD, 8.0)`. That assumes the
moving region is a small fraction of the frame. When a subject traverses the frame the
motion mask is dense, the median motion rises, the MAD rises, the threshold rises above the
motion values, `binary_opening` clears the remainder, `n == 0`, and the function returns
`None`. The class distribution is exactly what that predicts: **Walk 117 of 365 clips have
no box (32%)**, then Massage_oneself, Stand_up, Read_documents, Lie_down.

Those 111 clips are cached uncropped. The person crop is the single largest measured gain of
the campaign, roughly 14 leaderboard clips, and 111 training clips never received it.

### 16.4 What this is worth, stated against the test set rather than hoped for

**The test set is clean.** All 405 test clips have Depth_Color, IR and Skeleton at 100%.
Zero test clips are missing a primary modality.

| defect | train clips | test clips | recoverable |
|---|---|---|---|
| Thermal-only, primary inputs zeroed | 103 | **0** | nothing directly |
| `person_box` returned None with data present | 111 | **8** | 8 clips, ~4 in the public 201 |

So the missing-modality defect is worth **removing poison from training, not winning test
clips**. Dropping those 103 rows is a one-line manifest filter and cannot hurt, because the
all-zeros pattern never occurs at test time. The `person_box` defect is worth at most 8 test
clips, of which roughly 4 sit in the public subset, and only a fraction of those flip.

**Honest error budget: +2 to +4 public clips, 0.62189 to about 0.635.** Real, cheap, and
nowhere near a route to 0.95.

### 16.5 Class-prior correction: tested and dead

The test predicted distribution is badly skewed against the training prior (Play_games
predicted at 3.75x its prior, Watch_TV zero times in 405 clips, Massage_oneself at 0.30x).
That invites logit adjustment, `argmax(log p - tau * log prior)`, tuned on OOF:

| tau | OOF acc | change | subjects improved |
|---|---|---|---|
| 0 | 0.5807 | baseline | |
| 0.2 | 0.5818 | +2 clips | 5 of 16 |
| 0.5 | 0.5733 | -20 clips | 4 of 16 |
| 1.0 | 0.5529 | -75 clips | 1 of 16 |

Best case +2 clips of 2,702, paired t of 0.76 across subjects. Dead. Recorded so nobody
spends an afternoon on it.

### 16.6 What the numbers say about 0.95

Reaching 0.95 means 191 of 201, which is 66 net corrections over `r33_trim12`.

The largest coherent error block is seven sedentary object classes: Wipe_bowls, Write,
Read_documents, Turn_pages, Use_a_mobile_phone, Watch_TV, Play_games. 294 clips, 10.9% of
OOF, accuracy 0.2313, **226 errors, 20% of all errors**, and 42% of those errors land on
another class in the same block. **Solving that block perfectly takes OOF from 0.5807 to
0.6643.** Everything else in the dataset would still have to come along too.

Underneath it is a sampling limit that no learning mechanism removes. Classes are not
performed by all subjects:

| performing subjects | classes | clips | accuracy |
|---|---|---|---|
| 3 | 1 (Watch_TV) | 12 | **0.0000** |
| 6 | 2 | 58 | 0.2875 |
| 9 | 1 | 42 | 0.1667 |
| 10 | 4 | 171 | 0.4441 |
| 16 | 4 | 641 | 0.6273 |

Watch_TV is performed by three people in the entire corpus. Under a cross-subject split at
most two are ever in training. Nothing learns a class from two performers, and the measured
accuracy on it is 0.0000 across every model tried.

---

## 17. Section 10 resolved: the all-18 refit is worse, at matched steps

The experiment section 10 said was owed has been run.

| run | training subjects | ensemble made of | optimizer steps | LB | clips |
|---|---|---|---|---|---|
| **r19_solo** | 14 | **4 folds** | 8,700 | **0.59701** | **120** |
| r32_full4 | 18 | 4 seeds | 11,340 | 0.56716 | 114 |
| **r35_full46x4** | 18 | 4 seeds | **8,694, matched** | **0.55721** | **112** |

**Training on every available subject loses 8 clips against the 14-subject control.**

The step confound was real and it pointed the other way. Section 10 warned that `r32_full4`
trained 30% more steps than the recipe was tuned for, and treated that as the reason it
underperformed. Removing those extra steps made the all-18 run **worse**, 112 against 114.
Whatever the extra steps were doing, they were helping.

### The claim this kills

Section 10 originally asserted **+0.0102 accuracy per additional training subject** as "the
finding that explains everything else." It was retracted as unsupported when the two-point
slope was examined. It is now **contradicted by direct measurement**: four extra subjects,
steps held constant, cost 8 of 201 clips.

### The confound that remains, named rather than buried

This is not a clean subject-count experiment and should not be cited as one. `r19_solo`'s
four members are four **folds**, each trained on a different 14 subjects. The full-fit four
are four **seeds** on the identical 18 subjects. Fold members disagree with each other more
than seed members do, and averaging disagreement is most of what ensembling buys.

So the comparison moves two things at once: subject count up, member diversity down. The
measurement cannot separate them, and no run in this campaign can.

| | established |
|---|---|
| refitting on all subjects, then ensembling seeds, is worse than ensembling folds | **yes, twice, 114 and 112 against 120** |
| the +0.0102 per-subject slope | **dead** |
| whether subject count alone helps or hurts | **still unknown, and now unknowable from this data** |

### Practical consequence

Nothing from the full-fit line ships. Finals remain `sub_r33_trim12.csv` (125) and
`sub_r23_seeds16.csv` (124).

More usefully: "refit on everything at the end" is standard practice and this pipeline
punishes it by 8 clips. The fold ensemble is not a stepping stone to a final full-data
model here. It **is** the model.

---

## 18. The last three experiments, all null

Run 12 to 13 September on an RTX 5070 and a rented A100-80GB. All three were aimed at the
largest measured error block: 552 of 1,133 OOF errors are object-versus-object, 48.7%.

### 18.1 The defect fix, isolated (r35_boxfix, variant A)

`person_box` gained a fallback: when the MAD threshold empties the motion mask, re-estimate
the background from the lower 60% of motion. The existing path is untouched and 40 of 40
previously-boxed clips returned byte-identical boxes, so the change is strictly additive.

Paired against `r19_sadp` on the identical 2,702 OOF clips:

| | correct | accuracy |
|---|---|---|
| r19_sadp baseline | 1472 | 0.5448 |
| r35_boxfix | 1457 | 0.5392 |
| **difference** | **-15** | 188 fixed, 203 broken |

391 discordant clips, paired sd ~19.8, so -15 is **0.76 sd**. A wash.

**Not submitted, deliberately.** `check_boxfix` reported 56 boxes gained: 54 train and
**2 test**. Two clips of 405 change, so at most one sits in the public 201. The ceiling was
one clip and the OOF said zero. Spending a submission slot on it would have been waste.

Its value was as a control, not a result: it establishes that the box fix contributes
nothing, which makes variant B attributable to hand resolution alone.

### 18.2 Hand crop at 160 px (r36_hand160, variant B)

The one untested lever aimed at the object errors. The hand stream ran at 96 px for the
entire campaign and was never swept. 160 px is 2.78x the pixels on exactly the region where
the discriminating object sits.

It does not fit in 12 GB. It OOMed on the 5070 inside the Adam step and was moved to a
rented A100-80GB at $1.09/hr, 70 s/epoch.

Fold 0, against variant A's fold 0, same subjects, same recipe, differing only in hand
resolution:

| | A, hand 96 | B, hand 160 |
|---|---|---|
| epoch 33 | 0.5036 | 0.5221 |
| epoch 37 | 0.5093 | 0.5093 |
| **final, epoch 60** | **0.5036** | **0.4936** |
| user 5 | 0.3312 | 0.2812 |

Training losses were near-identical throughout: 2.1490 vs 2.1510 at epoch 33, 2.0477 vs
2.0306 at epoch 37. **2.78x the pixels changed nothing, and fold 0 came out 0.01 lower.**

**Stopped after fold 0.** Not because one fold settles it, but because B could not change
what ships even if folds 1 to 3 came back positive: competing with `sub_r33_trim12.csv`
at 125 requires 12 models, which is 2 more seeds, ~10 hours and ~$11 of A100, and fold 0
said the payoff was zero or negative. The chain was dead at step one.

### 18.3 Thirty-two frames (r37_t32, variant C)

Never trained. `torch.OutOfMemoryError: 10.41 GiB allocated on an 11.91 GiB card`. Doubling
the temporal dimension doubles every branch.

Dropped rather than moved to the A100. Its target was action-phase confusion, and the
measured error structure caps that at roughly 30 to 50 OOF clips against B's 552. Paying
another 8.4 GB upload and another hour of rented GPU for the smaller of two hypotheses,
when the larger one had just come back null, was not defensible.

### 18.4 An observation worth keeping for next time

Both A and B peak mid-training and decline into the saved checkpoint:

| | peak | saved at epoch 60 | cost |
|---|---|---|---|
| A fold 0 | 0.5121 (ep 36) | 0.5036 | -0.0085 |
| B fold 0 | 0.5250 (ep 32) | 0.4936 | **-0.0314** |

`--last-epoch` is unconditional by design, and it should stay that way: picking the best
epoch selects on the fold being measured, which is rule 5. But the **schedule** may be 15 to
25 epochs too long, and schedule length can be chosen honestly on folds other than the one
scored.

[Guessing] a fixed 45-epoch schedule might be worth several clips. Not actionable two days
from the deadline, because testing it means retraining 12 models on a bundle that already
scores 125. Recorded so the next campaign starts by sweeping schedule length.

### 18.5 The engineering failures on the way, all mine

| # | error | how it surfaced | consequence |
|---|---|---|---|
| 15 | raised the loader to 16 workers with `persistent_workers` and `prefetch_factor=4` | CUDA OOM at a RANDOM epoch, 13 on one run, 18 on the next | 64 batches of host-pinned memory alive at once pressures the WDDM driver. The plain 4-worker loader had run 60 epochs x 4 folds repeatedly without a single OOM. Optimised a working pipeline two days before a deadline and broke it. Reverted |
| 16 | set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` as the OOM fix | torch printed `expandable_segments not supported on this platform` and ignored it | Windows does not support it. Removed rather than left in looking useful |
| 17 | resumed an interrupted cache build | `check_boxfix` found 1,309 boxes lost | `build_one()` returns early on an existing `.npz` and never fills in `box` or `has_*`, so 1,411 rows came back with NaN metadata. Cache builds now `rmdir` first and never resume |
| 18 | `goto rebuild%1` in a batch file | nothing rebuilt | cmd.exe labels cannot be dynamic |
| 19 | `call :stage` inside a file that had used `goto` in a subroutine | `The system cannot find the batch label specified` | cmd resolves internal labels by byte offset. Rewrote the orchestrator flat, no labels, no goto |
| 20 | edited a `.bat` while cmd was executing it | unpredictable resume point | cmd reads batch files line by line at run time |
| 21 | told the user the box fix would recover 8 test clips | the real build recovered 2 | the standalone probe and the production build disagree; quote the number the build reports, not the probe |

### 18.6 Where this leaves the campaign

| intervention | result |
|---|---|
| person crop | **+14 clips** |
| ensembling, 1 to 12 models | **+13 clips** |
| hand crop + SADP | +8 clips |
| skeleton 3D fix | +2 to 5 clips |
| NTU initialisation | +1 clip |
| everything else | **zero, inside the noise band** |

That last row now includes: SimCLR pretraining, the derivative hand crop, knowledge
distillation, the all-18 refit at matched steps, the zero-input filter, the `person_box`
recovery, and the 160 px hand crop.

Two changes produced 27 of the 41 clips gained. This problem rewarded data handling and
averaging, and punished cleverness, consistently, for the entire campaign.

---

## 19. External review, 13 September: what it corrected and what it could not change

An independent research review was run against sections 1 to 18. It found four real errors
and one real gap. Recorded because a campaign document that only lists other people's
mistakes is not a record.

### 19.1 Verified timeline, which was wrong here

Checked against the live challenge page on 13 September:

| | |
|---|---|
| leaderboard freeze | **15 September 2026** |
| **verification package** | **18 September 2026, 23:59 UTC** |
| verification and evaluation | 19 to 30 September |
| final top 6 | 1 October |
| grand finals | 12 October, Shanghai |

This document previously said the verification stage was 22 September. **It is the 18th.**
Four days earlier than planned, now corrected in sections 12 and 15.6.

### 19.2 Rules, quoted verbatim from the live page

```
CNN / RNN / Transformer · model size <= 100 MB · no large pretrained backbones
No closed-source APIs or LLMs for development
No API / LLM labeling of training data
```

The LLM prohibition is still live and unqualified. A verbal permission is not a record, and
the verification window opens on the 19th.

### 19.3 Arithmetic and framing errors corrected

- **"at most 41 of the 76 public errors" was wrong; it is 37.** 552/1133 = 48.7% of 76. The
  original scaled OOF errors to 201 clips first and took the fraction of the wrong
  denominator. Fixed in section 14.
- **The 0.5638 paper comparison was oversold.** It is an RGB-only cross-subject result over
  five LOSO folds with long-tail removal, not six-modality fusion, and the appendix reports
  56.56% for the same experiment. Header now says so.
- **The top-10 = 0.8834 bound**, already narrowed in section 15, is restated here: it caps a
  selector over that fixed ten-class candidate set. It does not bound a newly trained model.

### 19.4 The gap this campaign never tested

Two public notebooks score **0.71144** and **0.71641** using **YOLO person crop plus
R(2+1)D video recognition**. That is roughly 18 clips above this campaign's 125.

Section 18.6 concluded that everything beyond person crop and ensembling returned zero.
That conclusion holds **for this representation family**. It was stated too broadly. A
Kinetics-pretrained video backbone is a different family and was never tried here.

Why it was not tried, and why it still was not attempted after the review:

1. **Rule risk.** R(2+1)D-18 is roughly 33M Kinetics-pretrained parameters against a rule
   reading "no large pretrained backbones." A publicly scored notebook is not an
   eligibility ruling. The two leading notebooks may themselves be ineligible.
2. **Time.** The review's own plan is a 24 to 48 hour allocation assuming a working X3D
   pipeline, a YOLO crop cache and fold-matched controls. The freeze was 2 days away.
3. **Hardware.** The 12 GB card already OOMs on a 2.78x hand crop and on 32 frames.

**Recorded as the campaign's main untested hypothesis, not as a missed opportunity that was
available.** If this track runs again, compact pretrained video (X3D-S, ~3.79M parameters,
Kinetics weights) with YOLO localization is where to start, subject to a written eligibility
ruling obtained first.

### 19.5 What did not change

Nothing about what ships. `sub_r33_trim12.csv` (125 of 201) and `sub_r23_seeds16.csv` (124)
remain the finals. Every measurement in sections 1 to 18 stands; what changed is three
numbers, one deadline, and the scope of one conclusion.

---

## 20. Y1, the YOLO person crop: RETRACTED, see section 22

**Everything below this heading is wrong.** The 0/3441 detection rate was a bug in my own
script, not a measurement, and the camera-framing explanation built on it does not hold.
Section 22 has the corrected numbers. This section is kept intact because the reasoning
error is the point: a 0 produced by my own code was read as a fact about the world.

## 20 (retracted). Y1, the YOLO person crop: dead on the camera framing

Run 13 September after the external review ranked a learned person detector as a top-two
experiment. It reached step one and stopped there.

### 20.1 The measurement

YOLO11n, COCO-pretrained, `classes=[0]`, 8 evenly spaced IR frames per clip, confidence
0.25, over all 3,441 clips:

```
detected 0/3441 = 0.0%, fallback 3441
```

Zero. No exceptions raised, weights downloaded correctly, the detector ran on every clip
and returned no person anywhere.

Run directly on individual frames to find the ceiling:

| frame | conf >= 0.25 | >= 0.10 | >= 0.03 | best confidence |
|---|---|---|---|---|
| IR, Walk, frame 48 | 0 | 0 | 2 | **0.091** |
| IR, Walk, frame 51 | 0 | 1 | 3 | **0.218** |
| IR, Read_documents | 0 | 0 | 0 | none |
| Depth_Color, Walk | 0 | 0 | 0 | none |

A per-frame 1st-to-99th percentile contrast stretch changed the results by **exactly
nothing**, identical counts and identical confidences in every row, because these IR frames
already span 0 to 255.

### 20.2 The reason, which is the finding

**The camera is roughly an arm's length behind the subject.** In a Walk clip the person is a
blown-out white mass filling the lower third of the frame, back of head only, truncated at
the frame edge. The matching Depth_Color frame shows the subject as a **black silhouette**,
because they sit inside the depth sensor's minimum range and depth returns zero.

There is no person-shaped object in the image for a COCO detector to find. 0/3441 is the
correct output of a working detector, not a bug and not a threshold to tune.

Lowering confidence to 0.05 would manufacture boxes. At 0.09 to 0.22 confidence on
out-of-domain imagery that is a random box generator, and section 4 of this document already
records what a bad crop costs.

**A second framing exists.** A Read_documents frame is a wide, very dark third-person room
view with the subject small and partly occluded. The dataset mixes near-field
over-the-shoulder and wide third-person camera setups across clips. Worth knowing
independently of this experiment.

### 20.3 What this explains retroactively

The person crop was the single largest win of the campaign, +14 clips, and it came from
temporal-median background subtraction, the crudest method available.

**It works precisely because it never needed a person shape.** It finds what moves. A
semantic detector needs a recognizable human, and at this framing there is not one. The
crude method is not a placeholder that a better detector would improve on; it is the method
the data actually admits.

### 20.4 The public 0.71 notebooks

Two public notebooks report 0.71144 and 0.71641 using "YOLO person crop plus R(2+1)D".
A working YOLO person crop cannot be reconciled with the measurements above on IR.

Possibilities, none verified: the detector was run on a different modality, non-default
thresholds were used, or the gain came from the R(2+1)D video backbone and the crop
contributed little. [Guessing] Not pursued: 36 hours to the freeze, and the R(2+1)D half is
roughly 33M Kinetics-pretrained parameters against a rule reading "no large pretrained
backbones".

### 20.5 Cost and outcome

15 minutes of detector time, one cache build discarded, no training run. The gate that
stopped it was the detection-rate print at step 1, which is the cheapest gate in the
pipeline and the reason this cost 20 minutes rather than 4 hours.

`cache_yolo` was a byte-identical copy of `cache_fixA` (3,441 fallback boxes) and was
deleted rather than left to confuse a later run.

---

## 21. The verification package, run end to end and proven

Built and tested 13 to 14 September, ahead of the **18 September 23:59 UTC** deadline.

### 21.1 Result

```
python inference.py data/raw verify_out.csv
...
vs sub_r33_trim12.csv: 405/405 identical
```

Run twice: once reusing an existing cache, once from nothing but the raw archives with
`data/cache_hand` and `data/scan_index.pkl` deleted first. Both reproduce the submitted CSV
exactly. The cold run rebuilt the 8.02 GB cache from the 45 GB of raw archives with 0 errors.

### 21.2 What is in the package

| file | role |
|---|---|
| `inference.py` | the entry point. Raw archives in, 405-row CSV out |
| `inference.sh` | three-line wrapper so a Linux host runs byte-identical code |
| `REPRODUCE.md` | model description, checkpoint provenance, folds, NTU licence, integrity |
| `requirements.txt` | pinned from the training environment |
| `src/param_count.py` | exact parameter count, built the way `predict()` builds the model |
| `.packageignore` | what must NOT ship |

Measured, not asserted: **6,521,262 parameters**, **79,164,228 bytes** for 12 int8
checkpoints, under the 100 MB cap on both the decimal and binary readings.

### 21.3 Written in Python, not shell, and why

Four rounds of this package were lost to shell syntax, none of them to the pipeline:

| failure | mechanism |
|---|---|
| `goto rebuild%1` | cmd labels cannot be dynamic |
| `call :stage` after a subroutine `goto` | cmd resolves internal labels by byte offset |
| edited a `.bat` while cmd was executing it | cmd reads batch files line by line at run time |
| `if exist A if exist B (...) else (...)` | the `else` binds to the INNER `if`, so extraction was silently skipped |

The logic moved into one Python file and the shell entries became wrappers. Python is
already a hard dependency, so this costs nothing and removes the entire failure class.

### 21.4 Four real defects the end-to-end run found

None were in the model. All four would have fired on the verification host on the 19th.

1. **The training set is a multi-part split zip**, `HAR.z01` to `HAR.z08` plus `HAR.zip`.
   `unzip *.zip` cannot read it. Now extracted with 7-Zip pointed at the final part, with a
   `zip -s 0` recombine fallback, and a clear refusal if neither is present.
2. **The shipped test archive contains a `.claude` directory**, so counting subdirectories
   gives 406. The assertion now counts `SM_test_*` specifically and reports non-clip
   entries as ignored.
3. **`data/scan_index.pkl` holds `WindowsPath` objects.** Unpickling those on Linux raises
   `NotImplementedError: cannot instantiate 'WindowsPath'` and kills `build_cache.py`. The
   index load is now non-fatal and falls through to a fresh walk, and the file is in
   `.packageignore` because it is a local speed cache, never a deliverable.
4. **`legacy 10-channel skeleton checkpoint`** printed twelve times in the verification log.
   Ten channels is the `--streams` joint+bone+motion representation these models were
   deliberately trained with. A verifier reading twelve "legacy" warnings would reasonably
   suspect the wrong checkpoints shipped. Now reads
   `skeleton input 10 channels (joint+bone+motion streams)`.

### 21.5 The assertions that make it trustworthy

`inference.py` refuses rather than continuing on bad state: 405 `SM_test_*` directories
extracted, a manifest produced, exactly 12 checkpoints, total bytes under 100,000,000,
405 output rows, every prediction in 0..39, and a direct diff against the submitted CSV.

Modalities, fold, quantization, dim, width, fusion, temporal and the skeleton channel count
are all read **from each checkpoint**, never from flags in the script. The inference
configuration cannot silently disagree with how the models were trained.

---

## 22. Section 20 retracted: YOLO works, and how the 0.716 notebook actually works

### 22.1 The bug

`src/yolo_box.py` read the source frames as `clip.get("IR")`. `scan()` nests them one level
down, under `clip["files"]["IR"]`. Every lookup returned `None`, every clip fell into the
`len(f) < 3` branch, and the script wrote `fallback=1` with no error for all 3,441 clips.

The output was `detected 0/3441 = 0.0%`. **I read that as a measurement.** Section 20 then
built a camera-framing theory on top of it, illustrated with a genuine near-field photograph
from `36_Walk/user1`, which is a real but unrepresentative clip.

Fixed, plus a guard: `yolo_box.py` now exits non-zero if more than half the clips fall back,
with the message that this is a path bug and not a detection rate.

### 22.2 The corrected measurement

YOLO11n, COCO-pretrained, `classes=[0]`, conf 0.25, 8 IR frames per clip:

| | detected | rate |
|---|---|---|
| test | 393 / 405 | **97.0%** |
| train | 2,897 / 3,036 | **95.4%** |

On individual test frames the confidence is **0.919, 0.919, 0.921**, not the 0.218 ceiling
section 20 quoted from training clips. The public notebook independently reports 397/405,
which matches.

The 30 Walk clips YOLO misses in training are user 5's Thermal-only clips from section 16.
They have no IR to detect in.

**As a treatment it is real but not dramatic**: against the existing motion boxes, mean IoU
0.727, median 0.760, with IoU below 0.5 on 8.1% of clips and a centre shift over 50 px on
13.5%. Most clips barely move; roughly one in ten gets a materially different crop.

### 22.3 What actually produces 0.716, from the notebook's own code

```python
network = ig65m_models.r2plus1d_34_32_kinetics(num_classes=400, pretrained=False)
```

**R(2+1)D-34 pretrained on IG65M plus Kinetics.** Their checkpoint metadata records
`parent_subject_val_accuracy: [0.7156, 0.7187]`. That is cross-subject validation, before
the leaderboard is involved, against this campaign's OOF of 0.5279.

Everything else is close to identical to this campaign:

| | r33_trim12 | notebook v9 |
|---|---|---|
| input | Depth_Color 3ch + IR 1ch | same, 4 channels |
| resolution | 128 x 128 | 128 x 128 |
| frames | 16 | 16 |
| TTA | 4-pass, hflip + temporal roll +/-1 | 4-pass, hflip + temporal roll +/-1 |
| ensemble | **12 models** | 2 models |
| crop | motion median | YOLO11n, margin 1.40 |
| backbone | 6.52M scratch CNN | **R(2+1)D-34, IG65M + Kinetics** |

Six times the ensemble and an 18-clip loss. That rules out ensembling, TTA, resolution,
frame count and modality choice. The backbone is what is left.

They meet the 100 MB cap by packing weights to int5 and int6: 93,688,142 bytes including
the detector. The size rule is satisfied. Whether an IG65M-pretrained R(2+1)D-34 satisfies
**"no large pretrained backbones"** is an organizer question, not one this document answers.

### 22.4 What this changes, and what it does not

Section 10 said the 0.71 plateau was "IG-65M and Kinetics, 65 million videos of diversity
bought elsewhere." That was the right mechanism. The error was assuming nobody in this
track would bring it.

Section 18.6's conclusion holds **for this representation family** and was stated too
broadly. A Kinetics-pretrained video backbone is a different family and was never tested
here.

The YOLO crop remains untested as a treatment. It is a real change on about 10% of clips
and the detection rate is balanced across train and test, so there is no distribution
mismatch blocking it.

### 22.5 Y1 tested properly: null, and it isolates the mechanism

With the path bug fixed, the experiment was run as designed. `check_y1.py` confirmed a clean
single-variable control before any training:

```
YOLO detections     : 3290 (95.6%), fallback 151
PERSON boxes changed: 3209 (93.3%)   <- the treatment
HAND boxes changed  : 0
OK. Clean single-variable control.
```

Fold 0, same subjects, same seed, same recipe, only the person crop differs:

| | fold 0 final | user 5 | user 6 | user 18 | user 24 |
|---|---|---|---|---|---|
| A, motion crop | **0.5036** | 0.3312 | 0.4483 | 0.5618 | 0.6792 |
| Y1, YOLO crop | **0.4964** | 0.2875 | 0.4138 | 0.5922 | 0.7044 |

**-0.0072, about 5 clips of 699.** Two subjects better, two worse. Stopped after one fold,
the same early read that correctly called hand-160.

This is the strongest null in the campaign, because the treatment was not weak: 93.3% of
crops changed, IoU against the motion box averaging 0.727, and the result is flat.

### 22.6 What that proves about the 0.716 notebook

The notebook differs from `r33_trim12` in exactly two ways: the YOLO crop and the
IG65M+Kinetics R(2+1)D-34 backbone. **The crop has now been tested directly in this pipeline
and contributes nothing.**

So the entire 0.19 cross-subject validation gap, 0.716 against 0.5279, is attributable to
the pretrained video backbone. Not the crop, not the TTA, not the ensemble size, not the
input modalities, all of which are the same or favour this campaign.

That is a cleaner attribution than the notebook itself offers, and it came from a null
result.

### 22.7 Final count

Nine consecutive interventions returning zero inside the noise band: SimCLR pretraining, the
derivative hand crop, knowledge distillation, the all-18 refit at matched steps, the
zero-input filter, the `person_box` recovery, the 160 px hand crop, the 32-frame cache, and
now the YOLO person crop.

Person crop and ensembling to twelve produced 27 of the 41 clips gained. Everything else,
across the whole campaign, produced nothing measurable. The binding constraint was never the
crop, the ensemble, the schedule or the fusion. **It is the representation, and buying a
better one costs 65 million videos.**

---

## 23. The backbone hypothesis, executed: four folds of R(2+1)D-34

Section 22 attributed the whole 0.19 gap to the pretrained backbone and stopped there. This
section is that hypothesis carried to a submission.

`src/train_r2p1d.py`, separate from `train.py` on purpose so the working 12-model pipeline
could not be disturbed one day before the freeze. Two deliberate differences from
`dataset.py`: Kinetics normalization instead of per-clip standardization, and DI only, no
HAND and no Skeleton. The point was to test the backbone, not to relitigate fusion.

### 23.1 Recipe search, fold 0 only

| run | config | fold 0 |
|---|---|---|
| CNN, motion crop | the campaign baseline | 0.5036 |
| CNN, YOLO crop 1.35 | | 0.5193 |
| r41 | 16f, b4, BN training, no EMA, 15 ep | 0.5934 |
| **r42** | **16f, b4, BN training, no EMA, 30 ep** | **0.6020** |
| r44 | 16f, b16, BN frozen, EMA .999, 15 ep | 0.5250 |
| r45 (A100) | 32f, b16, BN frozen, EMA .999, 15 ep | 0.5378 |
| r52 | 16f, b4, BN frozen, no EMA, 30 ep | 0.5863 |
| r49 (A100) | 32f, b16, BN training, no EMA, 30 ep | 0.5991 |

Three findings and one confession. BN freezing costs 0.016, measured twice. EMA at decay
0.999 over a 2,175-step run is a 1,000-step horizon and scores at chance in epoch 1. Going
from 16 to 32 frames is neutral at the best recipe. The confession is that r44 changed
three variables at once (batch 4 to 16, BN training to frozen, EMA none to 0.999) after
this document spent twenty-two sections preaching one variable at a time, and cost 0.068 in
doing so. **The first recipe written was the best one, and three hours of hypotheses made
it worse at every step.**

### 23.2 The four folds

Identical recipe, `data/cache_yolo`, seed 0.

| fold | validation users | n | accuracy |
|---|---|---|---|
| 0 | 5, 6, 18, 24 | 701 | 0.6020 |
| 1 | 3, 7, 19, 22 | 730 | 0.6658 |
| 2 | 4, 9, 16, 20 | 675 | 0.6593 |
| 3 | 1, 8, 21, 23 | 596 | 0.5839 |
| **pooled** | | **2702** | **0.6295** |

Against the CNN campaign's 0.5036 on fold 0 and 0.5279 pooled. **The backbone is worth
roughly +0.10 cross-subject**, which is the only intervention in this entire campaign that
moved the number outside the noise band.

The fold spread is subject difficulty, not model quality. Per-user accuracy is worst for
user 5 (0.456), 8 (0.544), 6 (0.567), 1 (0.575), 23 (0.589) and 4 (0.594). Fold 0 holds out
user 5, the Thermal-only defect subject from section 16. Fold 3 holds out three of the six
hardest. Both low folds are explained without appeal to their models.

## 24. Weight averaging beat the ensemble at one third the size

The 100 MB cap does not admit a THREE-model ensemble of this architecture. Three fp32
checkpoints are 763 MB and three int5 checkpoints are 119 MB, both over. That constraint
forced the useful experiment.

CORRECTED 15 September: the original wording said the cap admits no ensemble at all. **Two
int5 networks plus the detector is 91,771,564 bytes and does fit**, and one int6 plus one
int5 is 99,600,764 with 399,236 bytes to spare. Two-model options were never tested. The
int5 caveat from section 26 applies to both.

`src/soup.py` averages the fold checkpoints tensor by tensor. All folds fine-tune from the
identical IG65M initialization, so they should remain in one loss basin and the average
should be a valid network. It was not an assumption; it was submitted.

| submission | models | leaderboard |
|---|---|---|
| `sub_r42_2fold` | folds 0, 1 ensembled | 0.66169 |
| `sub_r42_3fold` | folds 0, 1, 2 ensembled | 0.68159 |
| **`sub_r42_soup3`** | **folds 0, 1, 2 averaged into one** | **0.69651** |
| `sub_r42_soup4` | all four averaged into one | 0.68656 |

**One network beat the three-model ensemble it was made from, at one third the size.**
Averaging is acting as regularization here, not merely as compression. Against the campaign's
standing 0.5638 this is +0.133.

soup4 losing 0.01 to soup3 is **two clips on 201** (138 against 140; see section 29)
and is noise. The tempting story, that fold 3
is a weak ingredient, is contradicted by 23.2: fold 3's model is not worse, its holdout is
harder. No principled reason separates soup3 from soup4.

### 24.1 A non-result recorded as one

`sub_r42_soup5` was meant to add the augmented fold-0 checkpoint as a fifth ingredient. It
scored 0.68656, identical to soup4 to five decimals. Two genuinely different 405-row
submissions do not do that. The files are byte-identical: the `scp` of the aug checkpoint off
the A100 never happened, the copy was a silent no-op, and soup5 is soup4 under another name.
**It is a non-result, not a null.** Recorded because the same silent-copy failure mode
produced section 20.

## 25. Augmentation: a null, called in advance

Fold 0 ended at train loss 0.7498 with validation flat from epoch 18, on 2,335 clips against
63.7M parameters. The existing augmentation was a horizontal flip and a temporal roll of
plus or minus 2. `src/train_r2p1d_aug.py` added a random resized crop (scale 0.75 to 1.0),
brightness jitter of plus or minus 20%, cutout on 25% of clips, and widened the roll to
plus or minus 4.

The decision rule was written down before the run: above 0.62 adopt, 0.59 to 0.62 null,
below 0.59 too aggressive.

Result **0.6049** against r42's **0.6020**. Null, and the rule was honoured.

Train loss rose to 0.8129, so the regularization did what it was designed to do. Validation
did not move. **The model is not limited by memorizing its training clips. It is limited by
subject shift, and augmentation does not address that.** This is the tenth consecutive
intervention returning zero inside the noise band.

## 26. Packing: int5 is lossy here, int6 is exact

`src/pack_int5.py`. Conv weights to symmetric per-output-channel integers, everything else
left alone, bit width on the command line.

The first version differed from the fp32 model on **20 of 405** test clips. Two defects,
both mine, stacked:

1. BatchNorm running statistics were stored in **fp16**. CORRECTED 15 September: the
   original wording here said any `running_var` below 6e-5 underflows to zero. That is
   wrong. 6.1e-5 is fp16's smallest NORMAL value; subnormals reach about 6e-8, and
   BatchNorm carries an epsilon besides. The defect was precision loss in the running
   statistics, not underflow.
2. The **40-way classifier head was quantized** to save 40 KB, which is the single most
   precision-sensitive tensor in the network.

Fixing both, plus excluding every tensor under 100,000 elements, costs 0.9 MB against a
100 MB cap.

| packing | bytes | predictions differing from fp32 |
|---|---|---|
| int5, first version | 40,088,699 | 20 of 405 |
| int5, fixed | 43,078,900 | 10 of 405 |
| **int6, fixed** | **50,908,100** | **0 of 405** |

int5 remains lossy for this model after both fixes. int6 is exact. With 49% of the cap
unused there was never a reason to fight for the smaller number.

Shipped total: `packed_i6.npz` 50,908,100 plus `yolo11n.pt` 5,613,764 equals
**56,521,864 bytes**, 56.52 MB decimal, 53.90 MiB.

## 27. The verification package, rebuilt for the entry that is actually being submitted

Section 21 proved a package that reproduced `sub_r33_trim12.csv`. That package would have
been handed over reproducing a submission that is not the entry.

`inference.py` now goes raw archives to `sub_r42_soup3.csv`: extract, YOLO11n boxes at
margin 1.35, cache build with the exact `cache_yolo` flags, cap check on the two shipped
files, unpack int6, predict with flip TTA, diff against the submitted CSV. **The packed
artifact is on the reproduction path, not beside it**, so the score being defended is the
score the shipped weights produce.

Three packaging defects were found that would have fired on the verification host:

1. `.packageignore` **excluded `yolo11n.pt`**, written when YOLO was not in the shipping
   path. The package could not have built its own cache.
2. `.packageignore` did not exclude `runs/` or `data/cache*/`, roughly 13 GB of training
   checkpoints and cache that are not the entry.
3. `requirements.txt` had no `ultralytics`. Stage 0 would have died.

### 27.1 Three reproduction runs

| run | state removed first | result |
|---|---|---|
| 1 | nothing (warm) | 405/405 identical |
| 2 | `data/cache_yolo`, `yolo_boxes.csv` | 405/405 identical |
| 3 | also `data/scan_index.pkl`, full filesystem rescan | 405/405 identical |

Run 3 is the path a verification host takes. YOLO11n returned exactly 3,290 detections and
151 motion fallbacks on all three runs.

Run 1 alone would have been self-deception: every expensive stage printed "reusing", so it
proved stages 3 through 5 and nothing else.

## 28. Honest accounting

**What is defensible.** The subject-disjoint number is **0.6425** pooled over folds 0 to 2,
the three folds in the shipped soup. The leaderboard number is **0.69651**.

**What is not.** The soup has no clean held-out set. Each fold model was validated on
subjects the other two trained on, so after averaging there is no subject group the network
has not seen. 0.69651 is a test-set reading.

CORRECTED 15 September: this section previously said the gap between 0.69651 and 0.6425
measures leaderboard optimism. **That subtraction is not valid.** The two numbers come from
different subject populations (unseen test users against held-out training users), different
models (the soup against its separate constituents) and different sample sizes. Selection
optimism is real and seven public probes bought some of it, but that difference does not
quantify it.

**The rule.** The track states `no large pretrained backbones`. The entry is 63.7M
parameters initialized from IG65M, 65 million videos. Whether that clears the rule is the
organizers' call. `REPRODUCE.md` and `inference.py` both state the provenance, the source
URL and the parameter count in their opening screens. If it is ruled out, `r33_trim12`
remains fully reproducible at 0.62189 and 79,164,228 bytes.

**The campaign in one line.** Twenty-two sections of crop, fusion, ensemble, schedule and
distillation work moved the number by roughly 0.06. One backbone swap moved it by 0.10, and
averaging three copies of it in weight space moved it another 0.015. Section 22.7 called
this correctly before it was tested: the binding constraint was the representation, and the
only thing that ever bought a better one was 65 million videos of someone else's data.

---

## 29. The public leaderboard is 201 clips, not 405, and a private leaderboard exists

Found 15 September by an external review, confirmed here by arithmetic. **Every leaderboard
score this campaign has ever received is an exact integer over 201, truncated to five
decimals.**

| score | k / 201 |
|---|---|
| 0.55721 | 112 |
| 0.59701 | 120 |
| 0.62189 | 125 |
| 0.66169 | 133 |
| 0.68159 | 137 |
| 0.68656 | 138 |
| **0.69651** | **140** |
| 0.69154 | 139 |
| 0.71641 | 144 |
| 0.98507 | 198 |

Ten for ten, and not one of them is an integer over 405. [Certain]

### 29.1 What this document got wrong because of it

Sections 15 through 28 read the "(125 of 201)" notation as a leaderboard **rank** and
computed every noise claim against the 405 submitted rows. Both readings happen to fit
0.62189, which is why it survived twenty-two sections unchallenged.

1. **Granularity is 1/201 = 0.00498**, twice what was recorded. Every "N clips on 405"
   statement in sections 23 to 28 is roughly half that many public clips.
2. **204 of the 405 rows are a private set that decides the result and was never probed.**
   No section before this one acknowledges that it exists.
3. **The selection advice was wrong.** "Take the higher measured number, there is no
   extrapolation" only holds if the public score covers the whole test set. It does not.
4. Seven public probes against a 201-clip set is more selection pressure than section 28
   allowed for.

### 29.2 What it does not change

The four-fold cross-subject numbers in 23.2 are computed on 2,702 held-out training clips
and are unaffected. So is every reproduction result in section 27. The backbone gain of
roughly +0.10 cross-subject stands on validation, not on the leaderboard.

### 29.3 Final selection

`sub_r42_soup3` (140/201) and `sub_r42_soup4` (138/201) differ by **two public clips**.
Section 24 already established that no principled argument separates them. With a private
set in play, two clips is not a reason to prefer either, and both were selected where a
second slot was available.

## 30. BatchNorm recalibration after averaging: correct in principle, null in measurement

The same external review identified the strongest remaining defect in the shipped model.
`src/soup.py` averages `running_mean` and `running_var` straight across checkpoints. Those
buffers are activation statistics of **particular** weights. The average of them is not the
statistic of the averaged network. Stochastic weight averaging recalibrates with a forward
pass for precisely this reason.

This is unrelated to r52. That froze BatchNorm **during fine-tuning** and cost 0.016. This
re-derives the buffers of a network that was never trained as such.

`src/bn_recal.py`: reset the running statistics, set `momentum = None` for a cumulative
average over batches, put only the BatchNorm layers in training mode so dropout stays off,
and run one deterministic un-augmented pass over the calibration clips.

CORRECTED 15 September: the code comment and the original wording here called that an
**exact** mean. It is not. `momentum = None` averages per-BATCH statistics unweighted, so it
is the exact pooled mean only when every batch has the same element count, and the pooled
VARIANCE is wrong regardless: averaging within-batch variances drops the between-batch
variation of the means. The correct population form is

    M = sum(n_b * m_b) / sum(n_b)
    V = sum[n_b * (v_b + (m_b - M)^2)] / sum(n_b)

with n_b the effective element count per channel (clips times frames times height times
width), not the clip count. fvcore's precise-BN implements this. The estimator used here is
approximate and was never compared against the exact one.

### 30.1 The control

Fold 0's own model, recalibrated on fold 0's **training** subjects only, scored on its
held-out subjects. No test data anywhere.

| | fold 0 holdout (701 clips) |
|---|---|
| before | 0.6020 |
| after | 0.6020 |
| delta | **+0.0000** |

The procedure does no damage. Two caveats, both against the reading this document first
gave it. Equal accuracy does not prove no prediction moved: clips may have flipped both ways
in equal number. And "a model's own statistics cannot be improved by re-measuring them" is
not a theorem, which is how the original wording used it. Training-time buffers are an EMA
collected while the weights, the augmentation and the batch composition were all changing,
so re-estimating them on clean inputs can legitimately differ. +0.0000 here is one
observation on one fold, not a general result.

### 30.2 The soup

Recalibrated on all 3,036 training clips, since the soup has no holdout to protect.

**54 of 405 test predictions changed, 13%.** The averaged buffers were materially wrong for
the averaged weights. The mechanism is real and present in this pipeline, not merely in the
literature.

### 30.3 The probe

No clean holdout exists for the soup and none can be constructed from the saved
checkpoints: a soup of folds 1, 2 and 3 has still seen fold 0's validation users in
training. The only readout was a leaderboard probe, taken knowingly.

| submission | public |
|---|---|
| `sub_r42_soup3` | 140/201 |
| `sub_r42_soup3_bn` | **139/201** |

One clip.

RETRACTED 15 September: this section originally continued "of roughly 27 changed public
clips, about 13 improved and 14 got worse." **That decomposition was invented, not
measured.** Nothing observed says how many of the 54 changed rows fall in the 201 public
clips, and in a 40-class problem a prediction can move from one wrong class to another wrong
class. Writing C for original-wrong/candidate-right, R for the reverse and W for
wrong-to-different-wrong, the only measured facts are `C + R + W = 54` over all 405 rows and
`C - R = -1` over the 201 public rows. That is consistent with C=13,R=14,W=0 and equally
with C=0,R=1,W=26, which imply opposite things about whether the two networks are worth
blending. C, R and W are computable on labeled held-out training clips and were not
computed.

The probe has **limited** discriminatory power, not zero: it rules out a large gain on that
public subset and confirms the function changed. It does not identify which network
generalizes better to unseen subjects.

### 30.4 Why the unrecalibrated model still ships

Not because 139 is below 140. Because under a genuine tie the deciding factor is
operational: `sub_r42_soup3.csv` is reproduced end to end three times, including a full
filesystem rescan, with the packed int6 artifact on the reproduction path and 0 of 405
disagreements. The recalibrated model has none of that and would need a fresh package built
against the 18 September deadline, with int6 parity re-verified from scratch because
packing parity does not transfer to new weights.

[Likely] the recalibrated network is the more principled object. [Certain] it is not
measurably better, and [Certain] adopting it costs verified reproducibility.

### 30.5 The eleventh null, and the only interesting one

SimCLR, the derivative hand crop, distillation, the all-18 refit, the zero-input filter,
the `person_box` recovery, the 160 px hand crop, the 32-frame cache, the YOLO crop,
stronger augmentation, and now BatchNorm recalibration.

This one is different from the other ten. The others tested ideas that turned out not to
apply. This one fixed a defect that demonstrably existed, moved 13% of the predictions, and
still bought nothing measurable. **Being right about the mechanism is not the same as being
right about the number**, and a campaign that cannot tell those apart will keep spending
days on correct ideas with no effect.

## 31. Second external review, 15 September: what it corrected and what it defers

The review that found the 201-clip denominator returned after section 30. Four of its
corrections are applied in place above, in sections 24, 30 and 30.1 and 30.3. The largest
is the retraction in 30.3: a decomposition of the BatchNorm probe that this document
presented as measurement and had invented.

### 31.1 The pattern behind that error

Section 20 was a theory built on a silent `yolo_box` fallback. Section 15.6 reported a
parameter count never measured. Section 30.3 reported an error decomposition never
computed. All three read as observations. **The failure mode is not carelessness about
facts, it is fluency about numbers that were never produced by a tool.** A number in this
document should be traceable to a command that emitted it.

### 31.2 What the review proposes, and why none of it runs now

The submission is frozen and the verification package is due 18 September. The proposals
below need training time, a package rebuild, or both, and are recorded for the next
competition rather than attempted against this deadline.

| # | proposal | why it is plausible | cost |
|---|---|---|---|
| 1 | blend current video model with the old hand/skeleton family | the new recipe is DI only; the old family carries information it discards | inference only, but ships two model families |
| 2 | probability blend of the two BatchNorm states over one stored backbone | 54 disagreements is real prediction diversity; extra bytes are only `8 * sum(BN channels)` | two inference passes, one backbone |
| 3 | exact pooled-variance estimator against the current one | section 30's estimator is approximate, see the correction there | one forward pass |
| 4 | frozen-feature readouts: linear probe, class prototype, subject-balanced prototype | separates a limited head from a limited encoder | feature extraction plus small fits |
| 5 | shared encoder over global and local views | applies the representation that worked to the detail the old 160 px crop failed on | training |
| 6 | snippet sampling across the whole clip | 16 frames from a 10 s clip at 10 FPS is not the same signal as 16 contiguous frames, and the 32-frame test also moved batch size | training |
| 7 | subject-aware contrastive loss on the new backbone | same-action different-performer is the actual objective | training |

Proposal 2 is the cheapest with a real mechanism behind it. Proposal 4 is the cheapest that
would tell this campaign something it does not know: whether the ceiling is the readout or
the representation. Neither was run.

### 31.3 The conclusion in 30.5, amended

Section 30.5 said being right about a mechanism is not the same as being right about the
number. That stands. What it should not be read as saying is that the recalibrated network
was shown to be worse, or that normalization is exhausted. The measured facts are narrower:
one approximate estimator, applied once, on one calibration population, produced no gain on
201 public clips, and the paired decomposition that would say why was never computed.

## 32. Cross-family blending: the largest out-of-fold gain in the campaign, and it did not transfer

The second review's first proposal, executed. It is the only experiment since the backbone
to clear the noise band on held-out data, and the only one whose leaderboard result
contradicted its validation result.

### 32.1 The complementarity is real

Joining the four r42 fold models' held-out predictions against the old family's
out-of-fold predictions, by `clip_id`, subject-disjoint on both sides, n = 2,702:

| | video acc | old acc | C (video wrong, old right) | R (video right, old wrong) | oracle |
|---|---|---|---|---|---|
| vs 12-model teacher | 0.6295 | 0.5777 | **238** | 378 | 0.7176 |
| vs r19_sadp, 4 models | 0.6295 | 0.5448 | **221** | 450 | 0.7113 |

238 recoverable clips, 8.8% of the set. Section 22.7's "everything else produced nothing
measurable" was about interventions inside one family. **The DI-only backbone discards the
hand and skeleton information, and the old family still holds a large amount of it.**

The four-model `fold_a` bundle captures nearly all of the complementarity of the twelve, and
it is the one that fits: 26,388,076 + 50,908,100 + 5,613,764 = **82,909,940 bytes**. Section
24's corrected note applies here too, the cap was never the obstacle it was claimed to be.

### 32.2 The blend gains out of fold

`src/oof_probs.py` regenerated held-out probabilities with the shipped flip TTA, then
`src/blend.py` gridded `p = (1-w) * video + w * old`. Sanity check passed: `w = 1.0`
returns exactly 0.5448 and 0.5777, the solo accuracies.

The TTA baseline is 0.6392, not the 0.6295 in section 23.2. **Flip TTA is worth +26
out-of-fold clips**, measured here for the first time and already present in every
submission.

| w | acc | net clips | fold 3 |
|---|---|---|---|
| 0.00 | 0.6392 | 0 | 0.599 |
| 0.40 | 0.6480 | +24 | 0.601 |
| **0.42** | **0.6495** | **+28** | **0.599** |
| 0.45 | 0.6510 | +32 | 0.591 |
| 0.50 | 0.6528 | **+37** | 0.592 |
| 0.55 | 0.6528 | +37 | 0.587 |
| 0.60 | 0.6451 | +16 | 0.574 |
| 1.00 | 0.5448 | -255 | 0.485 |

An interior maximum with a collapse after it, which is what a real optimum looks like.

`w = 0.42` was selected by a rule written before the grid was run: near-peak pooled gain AND
fold 3 not below baseline. The pooled peak at 0.50 pays for folds 0, 1 and 2 by degrading
fold 3, which holds out three of the six hardest subjects. Choosing 0.42 gave up 9 clips to
avoid that.

### 32.3 The leaderboard disagreed

| submission | public |
|---|---|
| `sub_r42_soup3` | 140/201 |
| `sub_blend042` | **136/201** |

+28 out of fold, -4 on the leaderboard. `sub_r42_soup3.csv` stands.

### 32.4 Why, and this is the part worth carrying forward

Noise covers it: at p = 0.69 on 201 clips the binomial standard deviation is **6.6 clips**,
so 136 and 140 are inside one sigma and a +1.5 expected gain was never measurable there.

But there is a mechanism that fits better. **The held-out folds are training subjects. The
test set is subjects nobody trained on.** C = 221 was measured on the easier population. The
old family degrades harder when the performer is genuinely new: its twelve-model ensemble
scores 125/201 on the real test set against the video soup's 140, a wider gap than the
0.6392 to 0.5448 seen out of fold. Blending at a weight tuned on training subjects then
overweights the model that falls off fastest on unseen ones.

**Out-of-fold validation on held-out training subjects is optimistic for cross-FAMILY
blending in a way it is not for comparing single models.** Nothing in sections 1 to 31 tests
that, because every earlier comparison held the family fixed. The correct protocol would
weight the blend on one held-out subject group and confirm it on a second, which four folds
over 16 of 18 subjects can just barely support.

### 32.5 Standing

Twelve interventions have now returned nothing on the leaderboard. This one is the second
where the mechanism was demonstrably real, after section 30. The campaign's honest summary
is unchanged and now better supported: **the representation was the binding constraint, the
pretrained backbone was the only thing that relieved it, and weight-space averaging of its
folds was the only cheap addition that held up.**

Eight public probes have been spent on 201 clips. No more were taken.

## 33. The new-backbone campaign: VideoMAE-S, resolution, and a third silent failure

Run after the freeze was set, testing whether a different pretrained video family beats
IG65M R(2+1)D. Everything here is fold 0, val users 5, 6, 18, 24, against **0.6020**.

| run | arch | res | margin | stem | fold 0 |
|---|---|---|---|---|---|
| **r42** | **R(2+1)D-34, IG65M** | **128** | **1.35** | 4-ch | **0.6020** |
| v4_r2p1d224 | R(2+1)D-34, IG65M | 224 | 1.35 | 4-ch | 0.5863 |
| v2_vmae | VideoMAE ViT-S, K400 | 224 | 1.35 | 4-ch | 0.5749 |
| v3_vmae_m155 | VideoMAE ViT-S, K400 | 224 | 1.55 | 4-ch | 0.5706 |
| v1_vmae | VideoMAE ViT-S, K400 | 224 | 1.35 | adapter | 0.5635 |

### 33.1 A third silent failure, same shape as the first two

`from_pretrained` on transformers 5.15.1 printed a load report and returned a model with
**36 randomly initialized attention bias tensors**. The published VideoMAE checkpoint
stores attention bias as `q_bias` and `v_bias` with no key bias; transformers 5.x expects
`query/key/value.bias`. The library called this a warning and continued.

It surfaced only because the pretrained 400-class head was run over six real clips first
and returned `belly dancing` for a person pouring a drink. `_load_remapped` in
`src/vmae.py` now remaps explicitly and **refuses** on any residual mismatch.

This is the third time: `yolo_box` wrote `fallback=1` silently (section 20), the packer
quantized the classifier head silently (section 26), and now this. **A completed run with
no error is not evidence of anything.** Section 31.1's rule, that a number should trace to
a command that emitted it, extends here: a loaded model should trace to a check that
exercised it.

### 33.2 The head-count question, settled by data not by config

The published config says 16 attention heads; the original VideoMAE ViT-S constructor uses
6. Q/K/V shapes are identical, so loading succeeds either way and the computation differs.
Running the K400 head over six actions decided it:

- **heads 6**: `Wipe_hands` -> contact juggling, clapping, playing drums, ripping paper,
  playing cymbals. Five of five are two-hands-together manipulations. `Drink_water` ->
  playing recorder, playing trumpet, blowing nose.
- **heads 16**: `Take_and_use_tableware` -> scuba diving 0.352. `Drink_water` -> belly
  dancing 0.619.

Confident and arbitrary is the signature of a wrong attention split. **heads 6**, matching
the original constructor. The published config is wrong. [Likely], on six clips.

### 33.3 The adapter starved IR, and it cost 8 clips

The first design mapped 4 channels to 3 with a 1x1x1 conv initialized to identity on
depth-colour and **zero on IR**, so IR would enter as the adapter learned. It did not. After
40 epochs at lr 1e-4 the IR column norm was **0.024**: the model trained on 3 channels while
r2p1d used 4, making v1 an invalid comparison.

Replacing it with a widened 4-channel patch embedding, seeded with the RGB mean exactly as
`r2p1d.py::_adapt_stem` already did, recovered 0.5635 -> 0.5749. **IR is worth about +8
clips**, a third of the gap, not the gap.

Twelve zero-initialized weights behind a network that can already fit its training set
without them will not learn their way in. Widen the pretrained layer instead.

### 33.4 Crop margin: closed at both resolutions

| margin | 128 (CNN) | 224 (VideoMAE) |
|---|---|---|
| 1.35 | 0.5193 | 0.5749 |
| 1.40 | 0.4950 | |
| 1.55 | 0.4836 | 0.5706 |

Monotonic at 128, null at 224. **1.35 stands.** Note that `remargin.py` reports **44.4% of
boxes at the 480 clamp** at margin 1.55, so half of that run tested "no crop" rather than a
1.55 crop.

v2 and v3 disagree on **159 of 701 predictions, 23%**, and differ by 3 clips of accuracy.
Prediction churn is not evidence of anything, which is the 30.3 retraction restated.

### 33.5 Resolution: 128 beats 224, for both families

r2p1d at 224 scored **0.5863**, eleven clips below itself at 128, at 2.5x the epoch cost.

This closes a confound this document created. The VideoMAE comparison changed architecture
AND resolution together; at matched 224 the two families are **8 clips apart**, not 19. Most
of the gap attributed to architecture was resolution.

[Likely] the mechanism is pretraining scale: R(2+1)D-34's Kinetics training uses 112x112
crops, so 128 is near-native and 224 is double. More pixels also means more to overfit with
2,335 clips, and both runs already sat on the label-smoothing floor of 0.677.

### 33.6 Standing

**Nothing tested beats the submitted model.** A 19 GB cache, four training runs and a margin
sweep produced no configuration above 0.6020, and the questions they closed are: the crop
margin is right at both resolutions, 128 is the right resolution, and a different pretrained
video family at matched settings is roughly tied rather than better.

Section 22.7 holds for a third family: **the representation is the binding constraint, and
IG65M is the only pretraining that has relieved it.**

## 34. The Rules page, read directly, 15 September

Four corrections from the competition's own Rules page, which this document had never
quoted.

### 34.1 There is a disqualification rule nobody accounted for

> §2.8b: The top-ranked Teams on the private leaderboard advance to a verification stage
> comprising a live (Zoom) inference run on freshly released sample data containing both
> seen and unseen subjects, plus independent reproduction by the organizers. **An accuracy
> gap greater than 10% between the verification run and the Team's Kaggle private
> leaderboard score is treated as cheating and results in disqualification.**

Fold accuracies range 0.5839 to 0.6658 and per-subject accuracy ranges 0.456 to above 0.70.
**A 10-point gap is reachable on an unlucky subject draw with no cheating whatsoever.**
`REPRODUCE.md` now states the full cross-subject range in advance so that variance reads as
a known property rather than a discrepancy.

### 34.2 The backbone rule is not where this document said it was

The phrase "no large pretrained backbones" appears on the **Overview** page, not in the
binding Rules. The Rules say the opposite of what sections 19.2 and 28 assumed:

> §2.6: **External data are permitted in the Small Model Track**, including dataset are
> reasonably accessible to all Participants at minimal cost.

> §2.5a: **If input data or pretrained models with an incompatible license are used** to
> generate your winning solution, you do not need to grant an open-source license for that
> data and/or model(s), **provided you clearly identify such components.**

The binding document explicitly contemplates pretrained models and requires only that they
be identified, which `REPRODUCE.md` and `inference.py` already do in their opening screens.
This is not a clearance, because the Overview line exists, but it is much better ground
than sections 19.2, 28 and 31.2 assumed.

### 34.3 A required file was missing

> §2.8a: the code must include the training code, inference code, **model checkpoint
> (checkpoints/model.pth)**, an entry script (**inference.sh**), a README, a signed honor
> declaration, and a description of the required computational environment.

The shipped artifact was at `model/packed_i6.npz`. It is now also at
`checkpoints/model.pth`, byte-identical, and `inference.py` loads from that path so the
required file is on the reproduction path rather than a copy beside it. Despite the `.pth`
name it is an uncompressed `.npz` read by `src/pack_int5.py`; this is stated in both
documents. **Still outstanding: the signed honor declaration and the README.**

### 34.4 Timeline

> §2.3a: opens Jun 18, 2026; submission deadline / leaderboard freeze **Sep 15, 2026**;
> **Selection Stage Sep 16 to 30, 2026**; UbiComp 2026 finals Oct 11, 2026 (Shanghai).

Section 19.1 corrected the date once already, from 22 September to 18 September. The Rules
page says the Selection Stage runs 16 to 30 September. **The 18 September figure in sections
12, 15.6, 19, 27 and 31 has no source on the Rules page and should be treated as unverified.**

### 34.5 Final selection

Two final submissions are allowed (§2.2b), and the private leaderboard alone decides
standing (§2.10b). Left alone, Kaggle auto-selects the two best public scores, which would
pair `sub_r42_soup3` (140/201) with `sub_r42_soup3_bn` (139/201) - the same weights with
recalibrated buffers, a change measured at -1 clip in section 30. The selected pair is
`sub_r42_soup3` and `sub_r42_soup4`, which differ in fold coverage rather than in
normalization buffers.

---

## 35. Final standing, and the retraction that matters most

Private leaderboard, read 19 September. **Rank 127 of 324. Final score 0.71568.** Top 15 per
track advanced; this entry did not. The verification package, the honor declaration and the
AI-assistance permission question are all moot.

Every private score is an exact integer over **204**, confirming the 405 test clips split
201 public / 204 private.

### 35.1 The public leaderboard was inverted against the private one

| submission | public | private | private clips |
|---|---|---|---|
| **`sub_blend042`** | 0.67661 (worst) | **0.74019** | **151/204** |
| `sub_r42_3fold` | 0.68159 | 0.72058 | 147 |
| `sub_r42_soup4` | 0.68656 | 0.71568 | 146 |
| `sub_r42_soup3_bn` | 0.69154 | 0.71568 | 146 |
| **`sub_r42_soup3`** | **0.69651 (best)** | **0.71078** | **145/204** |

**The submission with the best public score had the worst private score of the family.** The
submission section 32 rejected had the best private score of anything this campaign produced.

Selected pair scored 146/204. `sub_blend042` at 151/204 would have placed above rank 117,
somewhere in the 50 to 116 block. Not top 15, which required 0.87745, so the call did not cost
advancement. It cost roughly 10 to 25 places.

### 35.2 Section 32.4 is RETRACTED

32.4 claimed that out-of-fold validation is optimistic for cross-FAMILY blending, because
held-out folds are training subjects while the test set is not, and that this explained a
blend gaining +28 clips out of fold and losing 4 on the public leaderboard.

**That mechanism was invented to explain noise.** Private shows the blend was worth **+6 clips
over `sub_r42_soup3`**, 151 against 145. The out-of-fold measurement on 2,702 subject-disjoint
clips was correct in direction and conservative in size. The 201-clip public probe was wrong,
and the story built on top of it was wrong.

The aggravating detail: the correct principle was written down at the time, in the same
message that then violated it. "Your out-of-fold measurement on 2,702 subject-disjoint clips
is far better evidence than any 201-clip probe." The blend was then rejected because 201 clips
disagreed.

**The rule that should have governed, and the strongest thing this campaign produced:**

> When a well-powered subject-disjoint validation set and a small public probe disagree,
> believe the validation set. A probe that cannot resolve the effect size you are measuring
> is not evidence against the measurement.

Binomial noise on 201 clips at p = 0.70 has a standard deviation of 6.5 clips. Almost every
comparison this campaign litigated on the public leaderboard was smaller than that.

### 35.3 Section 24's soup result is weaker than reported

`sub_r42_3fold`, the three-model ensemble, scored **147/204** private against the three-fold
soup's 145. Section 24 reported the soup beating its own ensemble 140/201 to 137/201 and
called weight averaging a regularizer. **On the larger private half, the ensemble won.** The
soup's advantage was a public-set artifact. What remains true is the size argument: the soup
ships under the cap and the ensemble does not.

### 35.4 Honest accounting, final

Fourteen interventions. One cleared the noise band on validation and shipped: the IG65M
backbone. One more was real, was measured correctly, and was rejected on bad evidence: the
cross-family blend. Twelve were null.

The campaign's recurring failure was not bad modelling. It was treating a 201-clip leaderboard
as an instrument precise enough to overrule a 2,702-clip measurement, and then constructing
mechanisms to justify the reading.
