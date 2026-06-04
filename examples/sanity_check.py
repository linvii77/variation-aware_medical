"""Minimal forward/backward check for VAPL + PIDNet-M."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vap_pidnet import build_vapl_pidnet_m


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_vapl_pidnet_m(num_classes=19).to(device)
    model.train()

    images = torch.randn(2, 3, 256, 512, device=device)
    targets = torch.randint(0, 19, (2, 256, 512), device=device)

    outputs = model(images, targets)
    losses = outputs["losses"]
    assert losses is not None

    loss = losses["loss_total"]
    loss.backward()

    print(f"device: {device}")
    print(f"logits: {tuple(outputs['logits'].shape)}")
    print(f"embeddings: {tuple(outputs['embeddings'].shape)}")
    print(f"loss_total: {loss.item():.4f}")
    print(f"loss_seg: {losses['loss_seg'].item():.4f}")
    print(f"loss_cs: {losses['loss_cs'].item():.4f}")
    print(f"hard_fraction: {losses['hard_fraction'].item():.4f}")


if __name__ == "__main__":
    main()
