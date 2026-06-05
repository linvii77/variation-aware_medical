# VAPL + PIDNet-M Reproduction

This workspace contains a PyTorch reproduction scaffold for
**Variation-aware proxy learning for semantic segmentation** with **PIDNet-M**
as the first backbone.

## What is implemented

- PIDNet-M backbone: `m=2`, `n=3`, `planes=64`, `ppm_planes=96`,
  `head_planes=128`.
- Projection head on PIDNet fused features.
- Representative proxy per class.
- `Kc=5` variation vectors per class.
- Factorized similarity score.
- Negative-only focal modulation.
- Attraction Loss, Repulsion Loss, and Compositional Similarity Loss.
- Training wrapper with:
  - final segmentation CE loss,
  - optional PIDNet P-branch auxiliary CE loss,
  - `lambda_cs * L_cs`.

## Paper-default VAPL hyperparameters

```text
Kc = 5
lambda_var = 1.0
tau = 10.0
gamma = 2.0
tau_R = 0.8
lambda_r = 1.0
lambda_cs = 1.0
```

## Quick check

```bash
python examples/sanity_check.py
```

## Stage 1: Cityscapes PIDNet-M Baseline vs VAPL

Cityscapes data is not bundled with this repo. Put the official dataset in this
layout:

```text
/path/to/cityscapes/
  leftImg8bit/
    train/<city>/*_leftImg8bit.png
    val/<city>/*_leftImg8bit.png
  gtFine/
    train/<city>/*_gtFine_labelIds.png
    val/<city>/*_gtFine_labelIds.png
```

If `*_gtFine_labelTrainIds.png` files are present, the loader uses them.
Otherwise it maps official `labelIds` to the standard 19 Cityscapes train IDs.

Run the PIDNet-M baseline:

```bash
python tools/train_cityscapes.py \
  --data-root /path/to/cityscapes \
  --mode baseline \
  --max-iters 120000 \
  --batch-size 2 \
  --crop-size 1024 1024
```

Run PIDNet-M + the paper's VAPL loss:

```bash
python tools/train_cityscapes.py \
  --data-root /path/to/cityscapes \
  --mode vapl \
  --max-iters 120000 \
  --batch-size 2 \
  --crop-size 1024 1024
```

For a quick wiring check before full training:

```bash
python tools/train_cityscapes.py \
  --data-root /path/to/cityscapes \
  --mode vapl \
  --max-iters 10 \
  --eval-interval 10 \
  --batch-size 2 \
  --crop-size 512 512
```

## Basic usage

```python
import torch
from vap_pidnet import build_vapl_pidnet_m

model = build_vapl_pidnet_m(num_classes=19)
images = torch.randn(2, 3, 1024, 1024)
targets = torch.randint(0, 19, (2, 1024, 1024))

outputs = model(images, targets)
loss = outputs["losses"]["loss_total"]
loss.backward()
```

For inference, call the model without `targets`; the projection head and proxy
loss are skipped.

## Stage 2: 3D Medical Segmentation Backbone

The medical path adds a 3D V-Net/MagicNet-style backbone with optional VAPL and
SCDL-style semantic class distribution losses.

Expected local data layout:

```text
all-data/
  Synapse/*.h5
  lists_Synapse_DHC/{train_cases.txt,val_cases.txt,test_cases.txt}
  AMOS/*_image.npy
  AMOS/*_label.npy
  amos_splits/*.txt
```

`all-data/` is ignored by git and is not uploaded to GitHub.

Run a quick Synapse sanity check:

```bash
python tools/train_medical_3d.py \
  --dataset synapse \
  --mode combined \
  --patch-size 32 32 32 \
  --base-channels 4 \
  --embedding-dim 32 \
  --max-iters 1 \
  --foreground-prob 1.0
```

Validation modes:

```text
patch  center-crop patch validation for quick training feedback
full   full-volume sliding-window validation for final-style metrics
```

Use `full` only for short checks or formal evaluation, because it is much
slower than patch validation:

```bash
python tools/train_medical_3d.py \
  --dataset synapse \
  --mode ce \
  --patch-size 32 32 32 \
  --eval-mode full \
  --eval-stride 32 32 32 \
  --max-iters 1 \
  --eval-interval 1 \
  --max-val-batches 1 \
  --base-channels 4 \
  --embedding-dim 32
```

Ablation modes:

```text
ce        lambda_cs=0, lambda_scdl=0
vapl      lambda_cs=1, lambda_scdl=0
scdl      lambda_cs=0, lambda_scdl=1
combined  lambda_cs=1, lambda_scdl=1
```

Generate the four ablation commands without launching training:

```bash
python tools/run_medical_ablation.py \
  --dataset synapse \
  --patch-size 96 96 96 \
  --max-iters 1000
```

Or load a JSON config:

```bash
python tools/run_medical_ablation.py \
  --config configs/synapse_ablation_smoke.json
```

Add `--run` only when you intentionally want to execute the generated commands.
For a tiny smoke test, restrict the modes and iteration count:

```bash
python tools/run_medical_ablation.py \
  --dataset synapse \
  --modes ce vapl \
  --patch-size 32 32 32 \
  --base-channels 4 \
  --embedding-dim 32 \
  --max-iters 1 \
  --eval-interval 1 \
  --max-val-batches 1
```

Resume an interrupted run from a checkpoint:

```bash
python tools/train_medical_3d.py \
  --dataset synapse \
  --mode combined \
  --resume outputs/synapse_scdl3d_combined/checkpoint_001000.pth
```

Each run writes:

```text
args.json
metrics.csv
checkpoint_*.pth
best_dice.pth
```

Evaluate a saved checkpoint separately:

```bash
python tools/eval_medical_3d.py \
  --checkpoint outputs/synapse_scdl3d_combined/best_dice.pth \
  --dataset synapse \
  --eval-mode full \
  --patch-size 96 96 96
```

AMOS uses 16 classes by default:

```bash
python tools/train_medical_3d.py \
  --dataset amos \
  --mode scdl \
  --split-file all-data/amos_splits/labeled_5p.txt \
  --patch-size 32 32 32 \
  --base-channels 4 \
  --embedding-dim 32 \
  --max-iters 1 \
  --foreground-prob 1.0
```
