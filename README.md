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
