"""Train the 3D SCDL-style medical backbone with VAPL.

This script is intentionally small: it verifies the supervised 3D data path
before adding semi-supervised SCDL losses.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vap_pidnet import build_vapl_scdl_3d
from vap_pidnet.data import (
    AMOS_NUM_CLASSES,
    SYNAPSE_NUM_CLASSES_DHC,
    MedicalVolumeDataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["synapse", "amos"], default="synapse")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--patch-size", type=int, nargs=3, default=(96, 96, 96))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-iters", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--lambda-cs", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    fill_defaults(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = MedicalVolumeDataset(
        root=args.data_root,
        dataset=args.dataset,
        split_file=args.split_file,
        patch_size=tuple(args.patch_size),
        train=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    model = build_vapl_scdl_3d(
        num_classes=args.num_classes,
        in_channels=1,
        base_channels=args.base_channels,
        embedding_dim=args.embedding_dim,
        lambda_cs=args.lambda_cs,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = GradScaler(enabled=args.amp and device.type == "cuda")

    train_iter = iter(loader)
    model.train()
    start_time = time.time()

    for iteration in range(args.max_iters):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(loader)
            batch = next(train_iter)

        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=args.amp and device.type == "cuda"):
            outputs = model(images, targets)
            losses = outputs["losses"]
            if losses is None:
                raise RuntimeError("Training forward did not return losses.")
            loss = losses["loss_total"]

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        step = iteration + 1
        if step == 1 or step % args.log_interval == 0:
            elapsed = time.time() - start_time
            print(
                " ".join(
                    [
                        f"iter={step}/{args.max_iters}",
                        f"dataset={args.dataset}",
                        f"case={batch['case_id'][0]}",
                        f"image={tuple(images.shape)}",
                        f"logits={tuple(outputs['logits'].shape)}",
                        f"loss={loss.item():.4f}",
                        f"seg={losses['loss_seg'].item():.4f}",
                        f"cs={losses['loss_cs'].item():.4f}",
                        f"hard={losses['hard_fraction'].item():.4f}",
                        f"time={elapsed:.1f}s",
                    ]
                ),
                flush=True,
            )

        if step % args.save_interval == 0 or step == args.max_iters:
            save_checkpoint(args.output_dir / f"checkpoint_{step:06d}.pth", model, optimizer, step, args)


def fill_defaults(args: argparse.Namespace) -> None:
    data_root = ROOT / "all-data"
    if args.dataset == "synapse":
        args.num_classes = args.num_classes or SYNAPSE_NUM_CLASSES_DHC
        args.data_root = args.data_root or data_root / "Synapse"
        args.split_file = args.split_file or data_root / "lists_Synapse_DHC" / "train_cases.txt"
    else:
        args.num_classes = args.num_classes or AMOS_NUM_CLASSES
        args.data_root = args.data_root or data_root / "AMOS"
        args.split_file = args.split_file or data_root / "amos_splits" / "train.txt"

    if args.output_dir is None:
        args.output_dir = ROOT / "outputs" / f"{args.dataset}_scdl3d_vapl"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    args: argparse.Namespace,
) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
        "args": vars(args),
    }
    torch.save(payload, path)


if __name__ == "__main__":
    main()
