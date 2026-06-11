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

| Run | mode | lambda_cs | lambda_scdl | val_patch dice @20k |
| --- | --- | --- | --- | --- |
| `formal_synapse_ce_20000_w0` | ce | 0.0 | 0.0 | 0.2494 |
| `formal_synapse_combined_l05_20000_w0` | combined (old proxy) | 1.0 | 0.5 | 0.2825 |

Planned (new proxy mechanism), launched 2026-06-12, chained
sequentially on a single GPU (~5.2h each, ~10.4h total):

| Run | mode | lambda_cs | lambda_scdl | output_dir |
| --- | --- | --- | --- | --- |
| Phase A | vapl | 1.0 | 0.0 | `outputs/formal_synapse_vapl_proxydist_20000_w0/synapse/vapl` |
| Phase B | combined | 1.0 | 0.5 | `outputs/formal_synapse_combined_proxydist_l05_20000_w0/synapse/combined` |

Combined stdout/stderr log: `outputs/formal_proxydist_phaseAB.log`.

### Optional Follow-ups

- **Phase C**: ablation switch to force `q` uniform (no proxy) and
  measure the dice delta directly, isolating the proxy's contribution
  to final segmentation accuracy (beyond `proxy_assignment_accuracy`).
- **Phase D**: repeat Phase A/B on AMOS (16 classes) for cross-dataset
  generalization.
- **Phase E**: re-run the better of Phase A/B with 2 additional seeds
  (43, 44) for mean +/- std significance.
