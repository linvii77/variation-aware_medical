"""Train PIDNet-M baseline or PIDNet-M + VAPL on Cityscapes.

Examples:

```bash
python tools/train_cityscapes.py \
  --data-root /path/to/cityscapes \
  --mode baseline \
  --max-iters 120000

python tools/train_cityscapes.py \
  --data-root /path/to/cityscapes \
  --mode vapl \
  --max-iters 120000
```
"""

from __future__ import annotations

import argparse
import json
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

from vap_pidnet import VAPLPIDNetM
from vap_pidnet.data import Cityscapes, CityscapesTransform
from vap_pidnet.data.cityscapes import (
    CITYSCAPES_IGNORE_INDEX,
    CITYSCAPES_NUM_CLASSES,
)
from vap_pidnet.metrics import MeanIoU


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--mode", choices=["baseline", "vapl"], default="vapl")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--num-classes", type=int, default=CITYSCAPES_NUM_CLASSES)
    parser.add_argument("--ignore-index", type=int, default=CITYSCAPES_IGNORE_INDEX)
    parser.add_argument("--crop-size", type=int, nargs=2, default=(1024, 1024))
    parser.add_argument("--scale-range", type=float, nargs=2, default=(0.5, 2.0))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--val-batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-iters", type=int, default=120000)
    parser.add_argument("--lr", type=float, default=1.0e-2)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5.0e-4)
    parser.add_argument("--power", type=float, default=0.9)
    parser.add_argument("--aux-loss-weight", type=float, default=0.4)
    parser.add_argument("--lambda-cs", type=float, default=None)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--eval-interval", type=int, default=5000)
    parser.add_argument("--save-interval", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--no-eval", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.output_dir is None:
        args.output_dir = ROOT / "outputs" / f"cityscapes_pidnetm_{args.mode}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_args(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lambda_cs = args.lambda_cs
    if lambda_cs is None:
        lambda_cs = 1.0 if args.mode == "vapl" else 0.0

    model = VAPLPIDNetM(
        num_classes=args.num_classes,
        lambda_cs=lambda_cs,
        aux_loss_weight=args.aux_loss_weight,
        ignore_index=args.ignore_index,
    ).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    scaler = GradScaler(enabled=args.amp and device.type == "cuda")

    start_iter = 0
    best_miou = 0.0
    if args.resume is not None:
        start_iter, best_miou = load_checkpoint(args.resume, model, optimizer, scaler)

    train_loader = make_loader(args, split="train", train=True)
    val_loader = None
    if not args.no_eval:
        val_loader = make_loader(args, split="val", train=False)

    train_iter = iter(train_loader)
    model.train()
    start_time = time.time()

    for iteration in range(start_iter, args.max_iters):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        lr = poly_lr(args.lr, iteration, args.max_iters, args.power)
        for group in optimizer.param_groups:
            group["lr"] = lr

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
        if step % args.log_interval == 0 or step == 1:
            elapsed = time.time() - start_time
            print(
                " ".join(
                    [
                        f"iter={step}/{args.max_iters}",
                        f"mode={args.mode}",
                        f"lr={lr:.6f}",
                        f"loss={loss.item():.4f}",
                        f"seg={losses['loss_seg'].item():.4f}",
                        f"aux={losses['loss_aux_p'].item():.4f}",
                        f"cs={losses['loss_cs'].item():.4f}",
                        f"hard={losses['hard_fraction'].item():.4f}",
                        f"time={elapsed:.1f}s",
                    ]
                ),
                flush=True,
            )

        if step % args.save_interval == 0:
            save_checkpoint(
                args.output_dir / f"checkpoint_{step:06d}.pth",
                model,
                optimizer,
                scaler,
                step,
                best_miou,
                args,
            )

        if val_loader is not None and step % args.eval_interval == 0:
            metrics = evaluate(model, val_loader, device, args.num_classes)
            miou = float(metrics["miou"])
            print(
                f"eval iter={step} mIoU={miou:.4f} "
                f"pixel_acc={float(metrics['pixel_acc']):.4f}",
                flush=True,
            )
            if miou > best_miou:
                best_miou = miou
                save_checkpoint(
                    args.output_dir / "best.pth",
                    model,
                    optimizer,
                    scaler,
                    step,
                    best_miou,
                    args,
                )
            model.train()

    save_checkpoint(
        args.output_dir / "last.pth",
        model,
        optimizer,
        scaler,
        args.max_iters,
        best_miou,
        args,
    )


def make_loader(args: argparse.Namespace, split: str, train: bool) -> DataLoader:
    transform = CityscapesTransform(
        train=train,
        crop_size=tuple(args.crop_size),
        scale_range=tuple(args.scale_range),
        ignore_index=args.ignore_index,
    )
    dataset = Cityscapes(args.data_root, split=split, transform=transform)
    return DataLoader(
        dataset,
        batch_size=args.batch_size if train else args.val_batch_size,
        shuffle=train,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=train,
    )


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> dict[str, torch.Tensor]:
    model.eval()
    metric = MeanIoU(num_classes=num_classes, ignore_index=CITYSCAPES_IGNORE_INDEX)
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"]
        logits = model(images)["logits"]
        preds = logits.argmax(dim=1).cpu()
        metric.update(preds, targets)
    return metric.compute()


def poly_lr(base_lr: float, iteration: int, max_iters: int, power: float) -> float:
    factor = max(0.0, 1.0 - float(iteration) / float(max_iters))
    return base_lr * (factor**power)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    iteration: int,
    best_miou: float,
    args: argparse.Namespace,
) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "iteration": iteration,
        "best_miou": best_miou,
        "args": vars(args),
    }
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
) -> tuple[int, float]:
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint.get("scaler", {}))
    return int(checkpoint.get("iteration", 0)), float(checkpoint.get("best_miou", 0.0))


def save_args(args: argparse.Namespace) -> None:
    args_path = args.output_dir / "args.json"
    with args_path.open("w", encoding="utf-8") as handle:
        json.dump({k: str(v) for k, v in vars(args).items()}, handle, indent=2)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
