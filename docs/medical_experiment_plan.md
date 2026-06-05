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
