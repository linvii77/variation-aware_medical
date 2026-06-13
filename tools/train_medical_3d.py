"""Train the 3D SCDL-style medical backbone with VAPL.

This script is intentionally small: it verifies the supervised 3D data path
before adding semi-supervised SCDL losses.
"""

from __future__ import annotations

import argparse
import csv
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

from vap_pidnet import build_vapl_scdl_3d
from vap_pidnet.data import (
    AMOS_NUM_CLASSES,
    SYNAPSE_NUM_CLASSES_DHC,
    MedicalVolumeDataset,
)
from vap_pidnet.infer import sliding_window_logits_3d
from vap_pidnet.metrics import DiceHD95


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["synapse", "amos"], default="synapse")
    parser.add_argument(
        "--mode",
        choices=["ce", "vapl", "scdl", "combined"],
        default="vapl",
        help="Ablation mode. Explicit lambda arguments override this default.",
    )
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--val-split-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--patch-size", type=int, nargs=3, default=(96, 96, 96))
    parser.add_argument("--foreground-prob", type=float, default=0.75)
    parser.add_argument("--foreground-margin", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--val-batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-iters", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--lambda-cs", type=float, default=None)
    parser.add_argument("--lambda-scdl", type=float, default=None)
    parser.add_argument("--proxy-sigma-min", type=float, default=None)
    parser.add_argument("--lambda-dice", type=float, default=None)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-mode", choices=["patch", "full"], default="patch")
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-stride", type=int, nargs=3, default=None)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    fill_defaults(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_args(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics_path = args.output_dir / "metrics.csv"

    dataset = MedicalVolumeDataset(
        root=args.data_root,
        dataset=args.dataset,
        split_file=args.split_file,
        patch_size=tuple(args.patch_size),
        train=True,
        foreground_prob=args.foreground_prob,
        foreground_margin=args.foreground_margin,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_loader = None
    if not args.no_eval:
        val_dataset = MedicalVolumeDataset(
            root=args.data_root,
            dataset=args.dataset,
            split_file=args.val_split_file,
            patch_size=tuple(args.patch_size),
            train=False,
            full_volume=args.eval_mode == "full",
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
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
        lambda_scdl=args.lambda_scdl,
        lambda_dice=args.lambda_dice,
        proxy_sigma_min=args.proxy_sigma_min,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = GradScaler(enabled=args.amp and device.type == "cuda")
    start_iter = 0
    best_dice = 0.0
    if args.resume is not None:
        start_iter, best_dice = load_checkpoint(args.resume, model, optimizer, scaler)

    train_iter = iter(loader)
    model.train()
    start_time = time.time()

    for iteration in range(start_iter, args.max_iters):
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
                        f"mode={args.mode}",
                        f"dataset={args.dataset}",
                        f"case={batch['case_id'][0]}",
                        f"image={tuple(images.shape)}",
                        f"fg={(targets > 0).float().mean().item():.4f}",
                        f"logits={tuple(outputs['logits'].shape)}",
                        f"loss={loss.item():.4f}",
                        f"seg={losses['loss_seg'].item():.4f}",
                        f"dice={losses['loss_dice'].item():.4f}",
                        f"cs={losses['loss_cs'].item():.4f}",
                        f"scdl={losses['loss_scdl'].item():.4f}",
                        f"hard={losses['hard_fraction'].item():.4f}",
                        f"proxy_acc={losses['proxy_assignment_accuracy'].item():.4f}",
                        f"proxy_sigma={losses['proxy_sigma_mean'].item():.4f}",
                        f"time={elapsed:.1f}s",
                    ]
                ),
                flush=True,
            )
            append_metrics(
                metrics_path,
                {
                    "step": step,
                    "split": "train",
                    "mode": args.mode,
                    "dataset": args.dataset,
                    "case_id": batch["case_id"][0],
                    "loss_total": loss.item(),
                    "loss_seg": losses["loss_seg"].item(),
                    "loss_dice": losses["loss_dice"].item(),
                    "loss_cs": losses["loss_cs"].item(),
                    "loss_scdl": losses["loss_scdl"].item(),
                    "hard_fraction": losses["hard_fraction"].item(),
                    "proxy_assignment_accuracy": losses["proxy_assignment_accuracy"].item(),
                    "proxy_sigma_mean": losses["proxy_sigma_mean"].item(),
                    "foreground_fraction": (targets > 0).float().mean().item(),
                    "mean_dice": "",
                    "mean_hd95": "",
                    "elapsed_sec": elapsed,
                },
            )

        if step % args.save_interval == 0 or step == args.max_iters:
            save_checkpoint(
                args.output_dir / f"checkpoint_{step:06d}.pth",
                model,
                optimizer,
                scaler,
                step,
                best_dice,
                args,
            )

        if val_loader is not None and (step % args.eval_interval == 0 or step == args.max_iters):
            metrics = evaluate(
                model,
                val_loader,
                device,
                args.num_classes,
                patch_size=tuple(args.patch_size),
                stride=tuple(args.eval_stride),
                eval_mode=args.eval_mode,
                max_batches=args.max_val_batches,
            )
            mean_dice = float(metrics["mean_dice"])
            mean_hd95 = float(metrics["mean_hd95"])
            print(
                f"eval iter={step} mode={args.mode} "
                f"mean_dice={mean_dice:.4f} mean_hd95={mean_hd95:.4f}",
                flush=True,
            )
            append_metrics(
                metrics_path,
                {
                    "step": step,
                    "split": f"val_{args.eval_mode}",
                    "mode": args.mode,
                    "dataset": args.dataset,
                    "case_id": "",
                    "loss_total": "",
                    "loss_seg": "",
                    "loss_dice": "",
                    "loss_cs": "",
                    "loss_scdl": "",
                    "hard_fraction": "",
                    "proxy_assignment_accuracy": "",
                    "proxy_sigma_mean": "",
                    "foreground_fraction": "",
                    "mean_dice": mean_dice,
                    "mean_hd95": mean_hd95,
                    "elapsed_sec": time.time() - start_time,
                },
            )
            if mean_dice > best_dice:
                best_dice = mean_dice
                save_checkpoint(
                    args.output_dir / "best_dice.pth",
                    model,
                    optimizer,
                    scaler,
                    step,
                    best_dice,
                    args,
                )
            model.train()


def fill_defaults(args: argparse.Namespace) -> None:
    mode_lambdas = {
        "ce": (0.0, 0.0),
        "vapl": (1.0, 0.0),
        "scdl": (0.0, 1.0),
        "combined": (1.0, 1.0),
    }
    default_lambda_cs, default_lambda_scdl = mode_lambdas[args.mode]
    if args.lambda_cs is None:
        args.lambda_cs = default_lambda_cs
    if args.lambda_scdl is None:
        args.lambda_scdl = default_lambda_scdl
    if args.proxy_sigma_min is None:
        args.proxy_sigma_min = 0.05
    if args.lambda_dice is None:
        args.lambda_dice = 0.5

    data_root = ROOT / "all-data"
    if args.dataset == "synapse":
        args.num_classes = args.num_classes or SYNAPSE_NUM_CLASSES_DHC
        args.data_root = args.data_root or data_root / "Synapse"
        args.split_file = args.split_file or data_root / "lists_Synapse_DHC" / "train_cases.txt"
        args.val_split_file = args.val_split_file or data_root / "lists_Synapse_DHC" / "val_cases.txt"
    else:
        args.num_classes = args.num_classes or AMOS_NUM_CLASSES
        args.data_root = args.data_root or data_root / "AMOS"
        args.split_file = args.split_file or data_root / "amos_splits" / "train.txt"
        args.val_split_file = args.val_split_file or data_root / "amos_splits" / "eval.txt"

    if args.output_dir is None:
        args.output_dir = ROOT / "outputs" / f"{args.dataset}_scdl3d_{args.mode}"
    if args.eval_stride is None:
        args.eval_stride = tuple(max(1, size // 2) for size in args.patch_size)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_args(args: argparse.Namespace) -> None:
    args_path = args.output_dir / "args.json"
    serializable_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    args_path.write_text(json.dumps(serializable_args, indent=2, sort_keys=True))


def append_metrics(path: Path, row: dict[str, object]) -> None:
    fieldnames = [
        "step",
        "split",
        "mode",
        "dataset",
        "case_id",
        "loss_total",
        "loss_seg",
        "loss_dice",
        "loss_cs",
        "loss_scdl",
        "hard_fraction",
        "proxy_assignment_accuracy",
        "proxy_sigma_mean",
        "foreground_fraction",
        "mean_dice",
        "mean_hd95",
        "elapsed_sec",
    ]
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    iteration: int,
    best_dice: float,
    args: argparse.Namespace,
) -> None:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "iteration": iteration,
        "best_dice": best_dice,
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
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    start_iter = int(checkpoint.get("iteration", 0))
    best_dice = float(checkpoint.get("best_dice", 0.0))
    print(f"resumed from {path} at iter={start_iter} best_dice={best_dice:.4f}", flush=True)
    return start_iter, best_dice


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    patch_size: tuple[int, int, int],
    stride: tuple[int, int, int],
    eval_mode: str,
    max_batches: int | None = None,
) -> dict[str, torch.Tensor]:
    model.eval()
    metric = DiceHD95(num_classes=num_classes)
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        images = batch["image"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)
        if eval_mode == "full":
            if images.shape[0] != 1:
                raise ValueError("full eval currently requires val-batch-size=1.")
            logits = sliding_window_logits_3d(
                model,
                images[0],
                num_classes=num_classes,
                patch_size=patch_size,
                stride=stride,
                device=device,
            )
            preds = logits.argmax(dim=0, keepdim=True)
        else:
            logits = model(images)["logits"]
            preds = logits.argmax(dim=1)
        metric.update(preds, targets)
    return metric.compute()


if __name__ == "__main__":
    main()
