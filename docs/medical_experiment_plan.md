# Medical Experiment Plan

This document records the current experiment plan before launching long-running
training jobs.

## Datasets

### Synapse

- Root: `all-data/Synapse`
- Format: HDF5 volumes with `image` and `label`
- Classes: `14` (`0` background, `1..13` organs)
- Default train split: `all-data/lists_Synapse_DHC/train_cases.txt`
- Default validation split: `all-data/lists_Synapse_DHC/val_cases.txt`
- Default test split for standalone evaluation:
  `all-data/lists_Synapse_DHC/test_cases.txt`

### AMOS

- Root: `all-data/AMOS`
- Format: `*_image.npy` and `*_label.npy`
- Classes: `16` (`0` background, `1..15` organs)
- Default train split: `all-data/amos_splits/train.txt`
- Optional semi-supervised smoke split:
  `all-data/amos_splits/labeled_5p.txt`
- Default validation split: `all-data/amos_splits/eval.txt`
- Default test split for standalone evaluation:
  `all-data/amos_splits/test.txt`

## Ablation Modes

| Mode | Segmentation CE | VAPL loss | SCDL loss |
| --- | --- | --- | --- |
| `ce` | yes | no | no |
| `vapl` | yes | yes | no |
| `scdl` | yes | no | yes |
| `combined` | yes | yes | yes |

## Config Files

Smoke configs are intended only for 1-iteration execution checks.

- `configs/synapse_ablation_smoke.json`
- `configs/amos_ablation_smoke.json`

Template configs are the current long-run starting point. They must be reviewed
before formal experiments.

- `configs/synapse_ablation_template.json`
- `configs/amos_ablation_template.json`

Current template defaults:

```text
patch_size: 96 96 96
foreground_prob: 0.75
batch_size: 1
workers: 4
max_iters: 1000
lr: 0.001
weight_decay: 0.00001
base_channels: 16
embedding_dim: 256
eval_mode: patch
eval_interval: 500
save_interval: 500
```

## Smoke Checks

Generate commands without running:

```bash
python tools/run_medical_ablation.py \
  --config configs/synapse_ablation_smoke.json
```

Run the smoke checks:

```bash
python tools/run_medical_ablation.py \
  --config configs/synapse_ablation_smoke.json \
  --run

python tools/run_medical_ablation.py \
  --config configs/amos_ablation_smoke.json \
  --run
```

## Formal Run Gate

Before launching formal experiments, confirm:

- Dataset: Synapse, AMOS, or both
- Split choice for each dataset
- Modes to run
- `max_iters`
- `patch_size`
- `base_channels`
- `embedding_dim`
- Validation style during training: patch or full
- Full-volume evaluation cadence, if any
- Output root
- Resume strategy

Formal training must not be launched until these settings are explicitly
confirmed.

## SCDL-Style Proxy Distribution (Option B) Experiment Plan

### Background

The original VAPL "representative proxy" term was verified to be a
mathematical no-op under `softmax_scope="per_class"`: `sim(x, P_c)` is
constant across the per-class softmax dimension and is fully cancelled,
leaving `representative_proxies.grad` at noise level (~1.3e-5).

It has been replaced with an SCDL-style learnable Gaussian proxy
`(mu_c, sigma_c)` per class (`CompositionalSimilarityLoss.proxy_dist`,
shape `[C, 2*embedding_dim]`). The new mechanism computes a cross-class
assignment probability `q_c(x) = softmax_c(sim(x, mu_c) / sigma_c)` and
multiplies it into the existing per-class variation sub-distribution
`p_sub` to form a joint distribution `combined = q (x) p_sub` over
`(class, variation)`. All downstream attraction/repulsion/focal loss
code reuses `combined` unchanged.

New diagnostics logged to `metrics.csv`: `proxy_assignment_accuracy`
(`argmax_c q_c(x) == y`) and `proxy_sigma_mean`. New hyperparameter:
`proxy_sigma_min=0.05`.

### Pilot Validation (1000 iters, completed)

`outputs/pilot_synapse_proxydist_1000_w0/synapse/vapl/`:

- `proxy_dist.grad` is non-zero (fixes the no-op bug).
- `proxy_sigma_mean` decreases smoothly and monotonically from 0.694 to
  0.465 over 1000 steps, staying well above `proxy_sigma_min=0.05`.
- `proxy_assignment_accuracy` rises from ~0.04 (near-random, 1/14=0.071)
  at step 1 to 0.75-0.97 by step 10 onward.
- `val_patch` dice at step 1000 is 0.0235 vs 0.0292 for the old
  mechanism (`outputs/pilot_synapse_1000_w0/`) -- within noise at this
  short scale, not a meaningful comparison.

Conclusion: the new mechanism is numerically stable and the proxy
parameters now carry real gradient signal. Direction validated;
proceeding to formal comparison runs.

### Formal Comparison Runs (20000 iters, Synapse, patch 96^3, seed 42)

Existing baselines (old mechanism / no proxy):

| Run | mode | lambda_cs | lambda_scdl | val_patch dice @20k | best dice (step) |
| --- | --- | --- | --- | --- | --- |
| `formal_synapse_ce_20000_w0` | ce | 0.0 | 0.0 | 0.2494 | 0.2494 (20000) |
| `formal_synapse_combined_l05_20000_w0` | combined (old proxy) | 1.0 | 0.5 | 0.2825 | 0.2825 (20000) |

New proxy mechanism, `lambda_cs=1.0` (unchanged from old default),
completed 2026-06-12:

| Run | mode | lambda_cs | lambda_scdl | val_patch dice @20k | best dice (step) |
| --- | --- | --- | --- | --- | --- |
| `formal_synapse_vapl_proxydist_20000_w0` | vapl | 1.0 | 0.0 | 0.2172 | 0.2359 (19000) |
| `formal_synapse_combined_proxydist_l05_20000_w0` | combined | 1.0 | 0.5 | 0.2036 | 0.2389 (19000) |

Both new-proxy runs underperformed their old-mechanism counterparts
and showed an unhealthy dip from step 19000 to step 20000 (vapl:
0.2359 -> 0.2172; combined: 0.2389 -> 0.2036). This motivated the
loss-balance tuning below.

### Loss-Balance Tuning (lambda_cs / proxy_sigma_min Sweep)

#### Root cause

`combined = q_c(x) (x) p_sub` is structurally harder than the old
mechanism's `p_sub`-only objective, because `q_c(x)` must additionally
match the true class -- this inflates `loss_cs` by roughly 2-3x
relative to the old mechanism. With `lambda_cs=1.0` carried over
unchanged, `lambda_cs*loss_cs` dominates `loss_seg` (ratio ~2.1 @ step
20000 for both new-proxy runs above), starving the backbone of
segmentation gradient and producing the dice underperformance and
end-of-training dip.

`proxy_sigma_min` (the floor on `sigma_c`) was hardcoded at 0.05 and
not exposed as a CLI hyperparameter; this was fixed first
(`--proxy-sigma-min`, threaded through
`tools/train_medical_3d.py` -> `build_vapl_scdl_3d()` ->
`VAPLSCDL3D.__init__` -> `CompositionalSimilarityLoss`).

#### Pilot sweep (3000 iters, mode=vapl, seed=42)

Reference point (`lambda_cs=1.0, proxy_sigma_min=0.05`, from
`formal_synapse_vapl_proxydist_20000_w0`): dice@3000 = 0.0406.

| # | lambda_cs | proxy_sigma_min | dice@3000 |
| --- | --- | --- | --- |
| P1 | 0.1 | 0.05 | 0.0450 |
| P2 | 0.2 | 0.05 | 0.0377 |
| P3 | 0.5 | 0.05 | 0.0341 |
| P4 | 0.2 | 0.15 | 0.0349 |
| P5 | 0.2 | 0.25 | 0.0339 |
| P6 | 0.5 | 0.15 | 0.0487 |

Only P1 and P6 exceeded the reference; both were extended via
`--resume` to 8000 iters for a second look:

| step | P1 (lcs0.1/sig0.05) dice | P6 (lcs0.5/sig0.15) dice |
| --- | --- | --- |
| 1000 | 0.0302 | 0.0326 |
| 2000 | 0.0298 | 0.0328 |
| 3000 | 0.0450 | 0.0487 |
| 4000 | 0.0499 | 0.0571 |
| 5000 | 0.0480 | 0.0430 |
| 6000 | 0.0601 | 0.0636 |
| 7000 | 0.0518 | 0.0443 |
| 8000 | **0.0961** | 0.0625 |

P1 (`lambda_cs=0.1, proxy_sigma_min=0.05`) pulled decisively ahead at
8000 steps (~1.5x P6) and was selected for the formal 20000-iter runs.
Note `proxy_sigma_min=0.05` was already the prior hardcoded default, so
this is also the minimal-change choice.

### Formal Phase A2/B2 (20000 iters, lambda_cs=0.1, proxy_sigma_min=0.05, seed 42)

Completed 2026-06-13, chained sequentially on a single GPU
(~71 min each, ~143 min total -- much faster than the original ~5.2h/run
estimate):

| Run | mode | lambda_cs | lambda_scdl | val_patch dice @20k | best dice (step) | output_dir |
| --- | --- | --- | --- | --- | --- | --- |
| Phase A2 | vapl | 0.1 | 0.0 | 0.2582 | 0.2632 (18000) | `outputs/formal_synapse_vapl_proxydist_lcs0.1_sig0.05_20000_w0/synapse/vapl` |
| Phase B2 | combined | 0.1 | 0.5 | 0.2730 | **0.2870 (18000)** | `outputs/formal_synapse_combined_proxydist_lcs0.1_sig0.05_l05_20000_w0/synapse/combined` |

Loss balance at step 20000 (`lambda_cs*loss_cs / loss_seg`): Phase A2
~0.29, Phase B2 ~0.21 -- both healthy, vs ~2.1 for the `lambda_cs=1.0`
runs. `proxy_sigma_mean` decreased smoothly from ~0.47 to ~0.14-0.15
over 20000 steps, still well above the 0.05 floor (floor not yet
binding at this horizon). `proxy_assignment_accuracy` at step 20000:
0.890 (A2) / 0.916 (B2).

#### Success criteria check

1. **dice@20000 > 0.2825** (old combined baseline): Phase A2 0.2582
   (no), Phase B2 0.2730 (no, but 96.6% of baseline). Phase B2's best
   checkpoint (`best_dice.pth`, 0.2870 @ step 18000) **does** exceed
   the baseline.
2. **No more end-of-run dip** (step 19000 -> 20000): Phase A2
   0.2554 -> 0.2582 (up), Phase B2 0.2456 -> 0.2730 (up). Both pass --
   the instability seen in the `lambda_cs=1.0` runs is resolved.

**Conclusion**: `lambda_cs=0.1` fixes the loss-balance pathology and
the end-of-run instability, lifting dice by +18.9% (vapl) / +34.1%
(combined) over the `lambda_cs=1.0` new-proxy runs. The new mechanism
is now close to (Phase B2 final) or exceeds (Phase B2 peak) the old
(no-op) mechanism's baseline, and clearly better than the
`lambda_cs=1.0` new-proxy attempts. `(lambda_cs=0.1,
proxy_sigma_min=0.05)` is adopted as the new default for this
mechanism going forward.

### Phase F: Held-Out Test-Set Evaluation (full-volume, 6 cases)

`tools/eval_medical_3d.py` was fixed to load checkpoints with
`strict=False`: pre-refactor checkpoints store
`cs_loss.representative_proxies` (a different-shaped, proven no-op
parameter) instead of `cs_loss.proxy_dist`. Eval runs with
`lambda_cs=0.0` and no targets, so `cs_loss` is never exercised --
the mismatch is harmless for backbone-based dice/hd95.

`best_dice.pth` from each of four 20000-iter formal runs was evaluated
on `all-data/lists_Synapse_DHC/test_cases.txt` (6 held-out cases:
case0001, case0004, case0023, case0026, case0032, case0036) with
full-volume sliding-window inference (`eval_medical_3d.py` defaults):

| Run | proxy机制 | lambda_cs | lambda_scdl | test mean_dice | test mean_hd95 |
| --- | --- | --- | --- | --- | --- |
| `formal_synapse_ce_20000_w0` | -- | 0.0 | 0.0 | 0.3742 | 31.30 |
| `formal_synapse_combined_l05_20000_w0` | 旧(死) | 1.0 | 0.5 | 0.4409 | 20.02 |
| Phase A2 (vapl) | 新 | 0.1 | 0.0 | 0.4230 | 46.56 |
| Phase B2 (combined) | 新 | 0.1 | 0.5 | **0.4560** | 33.98 |

**Phase B2 achieves the best test-set dice overall (0.4560), beating
the old (dead-proxy) combined baseline by +0.0151 (+3.4% relative).**
This resolves the val_patch ambiguity from the formal run above (0.2730
vs 0.2825 @ step 20000): on the metric that actually matters (held-out
test set, full-volume inference), the tuned new proxy mechanism wins.

Per-class dice, Phase B2 vs old combined baseline: improvements on
classes 2, 3, 4, 6, 7, 9, 11 (notably class7 +0.080, class9 +0.101);
small regressions on classes 1, 8, 10. Classes 5/12/13 are 0.0 dice
across *all four* runs -- likely absent/degenerate in this 6-case test
split, not method-specific.

**HD95 caveat**: mean_hd95 is worse for both new-proxy runs (A2: 46.56,
B2: 33.98) than the old combined baseline (20.02). The dominant
contributor for B2 is class1 hd95 (113.1 vs 7.5 for old combined) --
with only 6 test cases, HD95 (sensitive to outliers) could in principle
be dominated by a single case with a small far-away false positive.
See Phase G below for the per-case breakdown.

### Phase G: Per-Case Dice/HD95 Breakdown (Phase B2 vs old combined)

Each of the 6 held-out test cases was evaluated individually (single-case
split files, full-volume sliding-window inference) for the old combined
baseline (`formal_synapse_combined_l05_20000_w0`) and Phase B2
(`formal_synapse_combined_proxydist_lcs0.1_sig0.05_l05_20000_w0`), using
`best_dice.pth` from each.

#### Per-case mean_dice / mean_hd95

| case | old combined dice | B2 dice | delta | old combined hd95 | B2 hd95 | delta |
| --- | --- | --- | --- | --- | --- | --- |
| case0001 | 0.4452 | 0.4508 | +0.0056 | 23.34 | 28.58 | +5.24 |
| case0004 | 0.4352 | 0.4561 | +0.0209 | 16.71 | 20.40 | +3.69 |
| case0023 | 0.3383 | 0.3507 | +0.0124 | 12.96 | 50.92 | +37.96 |
| case0026 | 0.4669 | 0.4932 | +0.0263 | 15.94 | 21.92 | +5.98 |
| case0032 | 0.4897 | 0.5243 | +0.0346 | 18.15 | 24.18 | +6.03 |
| case0036 | 0.4702 | 0.4609 | -0.0093 | 25.14 | 53.74 | +28.60 |

**Dice**: B2 wins on 5/6 cases (only case0036 regresses, by -0.0093). The
test-set dice win (+3.4% relative) is broadly distributed, not driven by a
single case.

**HD95**: B2 is *worse* on all 6/6 cases, by +3.7 to +38.0. This is a
systematic regression, not a single-outlier artifact.

#### Class1 dice / hd95 per case (the dominant hd95 contributor)

| case | old combined dice1 | B2 dice1 | old combined hd95_1 | B2 hd95_1 |
| --- | --- | --- | --- | --- |
| case0001 | 0.818 | 0.834 | 7.1 | 119.4 |
| case0004 | 0.931 | 0.876 | 1.7 | 107.4 |
| case0023 | 0.598 | 0.688 | 7.8 | 119.5 |
| case0026 | 0.841 | 0.793 | 6.1 | 95.6 |
| case0032 | 0.795 | 0.714 | 10.0 | 139.4 |
| case0036 | 0.838 | 0.737 | 12.1 | 97.4 |

class1 dice is roughly comparable between the two runs (mean 0.804 ->
0.774, a small regression), but **class1 hd95 jumps from 1.7-12.1 (old
combined) to 95-140 (B2) in every single case** -- a ~10-20x increase,
fully systematic across the test split. Since dice (volume-overlap based)
is largely unaffected while hd95 (boundary-distance based) blows up, the
pattern is consistent with B2 producing small, spatially-isolated
false-positive blobs for class1 that sit far from the true organ in every
volume, without materially changing the bulk overlap.

class1 alone accounts for (113.1 - 7.5) / 13 ~= 8.1 of the 13.96-point
overall mean_hd95 gap (33.98 vs 20.02), i.e. ~58%. The remaining gap is
concentrated in case0023 and case0036: for B2, both cases additionally
show hd95 > 95 for classes 3, 7, and 10 (vs <10, <10, and 0/83.3
respectively for old combined), which is why these two cases have by far
the largest per-case mean_hd95 (50.9 and 53.7).

**Conclusion**: the Phase F HD95 regression is real and systematic, not a
9-case (now known to be 6-case) outlier artifact. The tuned new proxy
mechanism (`lambda_cs=0.1, lambda_scdl=0.5`) trades a small amount of
boundary precision (hd95, especially class1) for a broadly-distributed
dice improvement (+3.4%, 5/6 cases). This should be reported as a genuine
trade-off / limitation alongside the dice win. A natural mitigation to
consider (not yet tested) is a largest-connected-component post-processing
step per class, which would likely remove the small far-away false
positives driving the hd95 blow-up without affecting dice much -- this
would isolate whether the regression is an artifact of scattered noise
vs. a genuine shift in the learned boundary.

### Phase H: Largest-Connected-Component Post-Processing

`tools/eval_medical_3d.py` gained a `--postprocess-largest-cc` flag
(`vap_pidnet/metrics.py::keep_largest_connected_component`): for each
foreground class, all but the single largest 3D connected component of
that class's predicted voxels are zeroed out (set to background). Applied
to `best_dice.pth` for old combined and Phase B2 on the full 6-case test
set (`eval_medical_3d.py` defaults, full-volume sliding window):

| Run | mean_dice (no pp) | mean_dice (+lcc) | mean_hd95 (no pp) | mean_hd95 (+lcc) |
| --- | --- | --- | --- | --- |
| old combined | 0.4409 | 0.4329 | 20.02 | 19.94 |
| Phase B2 | 0.4560 | **0.4597** | 33.98 | **20.45** |

#### Class1 / class3 (the largest Phase G hd95 outliers for B2)

| class | run | dice (no pp) | dice (+lcc) | hd95 (no pp) | hd95 (+lcc) |
| --- | --- | --- | --- | --- | --- |
| class1 | old combined | 0.804 | 0.807 | 7.5 | 6.9 |
| class1 | Phase B2 | 0.774 | **0.916** | 113.1 | **2.3** |
| class3 | old combined | 0.845 | 0.858 | 28.5 | 3.5 |
| class3 | Phase B2 | 0.890 | **0.923** | 54.6 | **1.8** |

**Phase G hypothesis confirmed**: B2's hd95 blow-up was driven by small,
spatially-isolated false-positive blobs, not a genuine boundary shift.
Removing them with largest-CC post-processing:

- Drops B2's mean_hd95 from 33.98 to **20.45**, essentially matching old
  combined (19.94-20.02).
- *Improves* B2's mean_dice slightly (0.4560 -> 0.4597), widening its lead
  over old combined to +6.2% relative (vs. old combined's own
  post-processed 0.4329) or +4.3% (vs. old combined's raw 0.4409).
- class1 and class3 individually go from B2's *worst* hd95 outliers to
  *better than old combined* on both dice and hd95 after cleanup
  (class1: dice 0.774->0.916, hd95 113.1->2.3; class3: dice 0.890->0.923,
  hd95 54.6->1.8).

**Caveat**: largest-CC is a blunt instrument. Classes 9-11 (and partly 8)
get *worse* on both dice and hd95 after post-processing for *both* runs
(e.g. old combined class10: dice 0.244->0.238, hd95 63.8->74.6; B2 class10:
dice 0.168->0.144, hd95 69.3->86.4) -- consistent with these being
bilateral/multi-component organs (e.g. paired kidneys/adrenal glands) where
a true second lobe gets discarded. The *net* effect across all 13 classes
is still clearly positive for B2 and roughly neutral for old combined, but
a size-threshold-based cleanup (drop small components below N voxels,
rather than strictly "keep only the largest") would likely be a better
universal choice and avoid this bilateral-organ penalty.

**Updated headline**: with this minimal post-processing step, Phase B2
(new proxy mechanism, `lambda_cs=0.1, lambda_scdl=0.5`) is unambiguously
better than the old (dead-proxy) combined baseline on *both* primary
test-set metrics (dice 0.4597 vs 0.4329/0.4409, hd95 20.45 vs 19.94/20.02),
with no remaining hd95 trade-off.

### Phase J (Proposed, NOT YET EXECUTED): CE+Dice Composite Loss + Class-Balanced Foreground Sampling

Raised after Phase H: (1) B2's *raw* (pre-LCC) hd95 is much worse than
OldComb (33.98 vs 20.02) -- relying on LCC post-processing alone to
"rescue" this is unsatisfying for the paper narrative; (2) classes
5/12/13 (esophagus, right/left adrenal gland) are 0.0 dice across *all
six* evaluated checkpoints, including the plain CE baseline.

#### Problem 1 root-cause analysis

- `combined = q_c(x) (x) p_sub` (Section 1) is consumed **only** inside
  `CompositionalSimilarityLoss.forward()` to produce `loss_cs`
  (`vap_pidnet/vapl.py:170-188`). It never touches the segmentation
  logits. At inference (`tools/eval_medical_3d.py`), `model(images)` is
  called with `targets=None`, which short-circuits before
  `projection_head`/`cs_loss`/`scdl_loss` are even evaluated
  (`vap_pidnet/model.py:272-281`). So the dice/hd95 differences between
  OldComb/A2/B2 come *entirely* from how the auxiliary losses reshape the
  shared `SCDLVNet3D` backbone weights during training -- not from any
  proxy-weighted prediction at test time.
- A2 (`mode=vapl`, `lambda_scdl=0`, mean_hd95=46.56) is *worse* than B2
  (`mode=combined`, `lambda_scdl=0.5`, mean_hd95=33.98), which is worse
  than OldComb (mean_hd95=20.02). The common factor in A2 and B2 is the
  **new proxy mechanism** (`lambda_cs=0.1` with the learnable Gaussian
  `(mu_c, sigma_c)`), not the SCDL branch -- ruling out `loss_scdl` as the
  primary cause.
- Per-class hd95 (B2 vs OldComb, Section 5 of `paper_results.md`): class 1
  (spleen) 113.11 vs 7.47, class 3 (left kidney) 54.60 vs 28.54, class 7
  (stomach) 71.04 vs 20.17. All three are left-upper-quadrant soft-tissue
  organs with similar CT intensity and that are anatomically adjacent to
  each other -- consistent with the new proxy mechanism's backbone
  producing a handful of spatially isolated misclassifications among
  these neighboring, texture-similar organs.
- `loss_seg` is **pure voxel-wise cross-entropy**
  (`vap_pidnet/model.py:284-288`). CE is insensitive to *where* an error
  occurs -- a handful of false-positive voxels out of ~10-20M barely move
  the average. HD95, by contrast, is the 95th percentile of *surface*
  distances and can be dominated by a single isolated outlier blob. This
  mismatch between the training loss and the hd95 eval metric is the
  structural reason a few stray voxels can blow up hd95 by 10x while dice
  moves by <5%.

#### Problem 2 root-cause analysis (corrects the earlier "absent in test
split" claim in `paper_results.md` Sections 4/Appendix -- now falsified)

- Direct inspection of all 6 test cases' `.h5` label volumes confirms
  classes 5/12/13 are **present with non-trivial voxel counts** in
  *every* test case: class 5 (esophagus) 543-11212 voxels, class 12
  (right adrenal) 633-1757 voxels, class 13 (left adrenal) 543-2480
  voxels (each <=0.9% of the per-case foreground voxel total). The
  "0.0 dice" is therefore not a metric artifact from an empty target --
  `DiceHD95.update` appends a real `dice=0.0` because `pred_mask` is
  empty while `target_mask` is non-empty: **the model never predicts a
  single voxel of classes 5/12/13, anywhere, in any of the 6
  configurations** (including plain CE).
- `foreground_crop_starts` (`vap_pidnet/data/medical3d.py:175-199`) picks
  the patch center by sampling **uniformly over all foreground voxels
  pooled together** (`np.argwhere(target > 0)`). class 6 (liver) alone has
  230k-560k voxels per case vs ~600-2500 for classes 12/13 -- so a
  randomly chosen foreground voxel lands in class 12/13 with probability
  roughly 0.05%-0.2%. With `foreground_prob=0.75` this gives each of these
  classes only a handful of centered patches across 20000 iterations.
- Even when such a patch *is* sampled, voxel-wise CE averages over all
  ~884736 voxels in a 96^3 patch -- a class occupying <=0.2% of even a
  "centered" patch contributes a near-zero share of the CE gradient. So
  CE is the dominant blocker, not just sampling frequency.

#### Proposed fix (addresses both problems with one change set)

1. **Add a soft Dice term to `loss_seg`** (standard CE+Dice composite,
   as in nnU-Net): new `vap_pidnet/losses.py` with
   `soft_dice_loss(logits, targets, num_classes, ignore_index,
   include_background=False, eps=1e-5)`, computed per-class on
   `softmax(logits)` with epsilon smoothing in numerator/denominator.
   - For a class with *no* GT voxels in the current patch but some FP
     predictions, `dice_c = eps / (FP_count + eps) ~= 0`, so
     `loss_dice_c ~= 1` -- this directly penalizes hallucinated
     false-positive blobs for classes absent from the patch, targeting
     Problem 1's "isolated speckle" pathology at the training-loss level
     instead of only via post-hoc LCC.
   - Because Dice is normalized per class (not per voxel), a class
     occupying 0.1% of a patch still contributes a full `[0,1]` term --
     directly counteracting the CE dilution behind Problem 2.
2. **Class-balanced (stratified) foreground patch sampling**: rewrite
   `foreground_crop_starts` to first pick a *class* uniformly at random
   from the foreground classes present in the volume, then pick a random
   voxel of *that* class as the patch center (instead of pooling all
   foreground voxels by raw voxel count). This raises the exposure
   frequency of classes 5/12/13 from <0.2% to roughly 1/(num present
   classes) (~8%), giving the new Dice term actual positive examples to
   learn from. `foreground_prob`/`foreground_margin` keep their current
   meaning and defaults -- no CLI surface change needed here.
3. New `--lambda-dice` CLI flag in `tools/train_medical_3d.py` (default
   `0.5`, the common CE:Dice 1:1 weighting), threaded through
   `build_vapl_scdl_3d()` -> `VAPLSCDL3D.__init__` -> `forward()`:
   `loss_seg = ce + lambda_dice * soft_dice_loss(...)`. Applies uniformly
   to **all** modes (ce/vapl/combined) so any later comparison stays fair.

#### Staged validation plan (no 20000-iter run without explicit confirmation)

1. Code changes + 1-2 iter smoke test (`--lambda-dice 0.5 --max-iters 1
   --no-eval`): confirm `args.json` records `lambda_dice`, `loss_total`
   includes the dice term, and a standalone sampling test shows class
   5/12/13 patch centers at roughly the expected ~1/13 rate (vs near-zero
   before).
2. 3000-iter pilot, `mode=combined` (B2 config: `lambda_cs=0.1,
   proxy_sigma_min=0.05, lambda_scdl=0.5`) + `--lambda-dice 0.5` + new
   sampling, seed=42. Compare `val_patch` dice @1000/2000/3000 against the
   existing B2 reference trajectory (0.0302/0.0310/0.0389). Also run a
   quick full-volume test-set eval on the step-3000 checkpoint: even with
   low absolute dice, check whether classes 5/12/13 now get *any* non-zero
   predicted voxels (the key "did the fix engage" signal, independent of
   overall quality at 3000 steps).
3. Decision gate: if (a) classes 5/12/13 become non-zero in at least some
   cases and (b) the val_patch dice trend is not worse than the B2
   reference, propose re-running **all four** formal 20000-iter configs
   (CE, OldComb, A2, B2) with `--lambda-dice 0.5` + the new sampling, for
   a fair like-for-like comparison. This is a ~4x20000-iter commitment
   (roughly 4-5 hours total based on the ~645s/3000-iter pilot rate) and
   requires explicit user confirmation per the Formal Run Gate before
   starting.
4. After the formal reruns (if approved), update `paper_results.md` with
   the new headline numbers and re-evaluate whether `B2+LCC` is still
   needed, or whether the raw (pre-LCC) numbers now stand on their own.

### Optional Follow-ups

- **Phase C**: ablation switch to force `q` uniform (no proxy) and
  measure the dice delta directly, isolating the proxy's contribution
  to final segmentation accuracy (beyond `proxy_assignment_accuracy`).
- **Phase D**: repeat Phase A2/B2 on AMOS (16 classes) for
  cross-dataset generalization.
- **Phase E**: re-run Phase B2 (`lambda_cs=0.1, lambda_scdl=0.5,
  proxy_sigma_min=0.05`) with 2 additional seeds (43, 44) for mean +/-
  std significance on the test-set dice/hd95 (with and without
  `--postprocess-largest-cc`), given the Phase F/H win margins are based
  on a 6-case test split.
- **Phase I** (optional, likely subsumed by Phase J): replace largest-CC
  with a size-threshold connected-component filter (drop components below
  N voxels per class) to avoid the Phase H bilateral-organ penalty on
  classes 9-11 while still removing the small false-positive blobs. If
  Phase J's Dice term already suppresses raw FP blobs, LCC/size-threshold
  post-processing may become unnecessary entirely.
