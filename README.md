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
