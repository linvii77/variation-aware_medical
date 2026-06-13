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

### Phase F: Held-Out Test-Set Evaluation (full-volume, 9 cases)

`tools/eval_medical_3d.py` was fixed to load checkpoints with
`strict=False`: pre-refactor checkpoints store
`cs_loss.representative_proxies` (a different-shaped, proven no-op
parameter) instead of `cs_loss.proxy_dist`. Eval runs with
`lambda_cs=0.0` and no targets, so `cs_loss` is never exercised --
the mismatch is harmless for backbone-based dice/hd95.

`best_dice.pth` from each of four 20000-iter formal runs was evaluated
on `all-data/lists_Synapse_DHC/test_cases.txt` (9 held-out cases) with
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
across *all four* runs -- likely absent/degenerate in this 9-case test
split, not method-specific.

**HD95 caveat**: mean_hd95 is worse for both new-proxy runs (A2: 46.56,
B2: 33.98) than the old combined baseline (20.02). The dominant
contributor for B2 is class1 hd95 (113.1 vs 7.5 for old combined) --
with only 9 test cases, HD95 (sensitive to outliers) can be dominated
by a single case with a small far-away false positive. A per-case
breakdown is needed before drawing conclusions about boundary quality.

### Optional Follow-ups

- **Phase G**: per-case dice/hd95 breakdown for Phase B2 vs old combined
  on class1 (and other regressed classes), to determine whether the
  hd95 regression is a systematic boundary-quality issue or a single
  outlier case with a small far-away false positive.
- **Phase C**: ablation switch to force `q` uniform (no proxy) and
  measure the dice delta directly, isolating the proxy's contribution
  to final segmentation accuracy (beyond `proxy_assignment_accuracy`).
- **Phase D**: repeat Phase A2/B2 on AMOS (16 classes) for
  cross-dataset generalization.
- **Phase E**: re-run Phase B2 (`lambda_cs=0.1, lambda_scdl=0.5,
  proxy_sigma_min=0.05`) with 2 additional seeds (43, 44) for mean +/-
  std significance on the test-set dice/hd95, given the Phase F win
  margin (+3.4% dice) and hd95 regression are both based on a 9-case
  test split.
