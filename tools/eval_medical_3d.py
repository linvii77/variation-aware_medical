"""Evaluate a 3D medical checkpoint with patch or full-volume inference."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
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
from vap_pidnet.metrics import DiceHD95, keep_largest_connected_component


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", choices=["synapse", "amos"], default="synapse")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--patch-size", type=int, nargs=3, default=(96, 96, 96))
    parser.add_argument("--eval-mode", choices=["patch", "full"], default="full")
    parser.add_argument("--eval-stride", type=int, nargs=3, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--postprocess-largest-cc", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fill_defaults(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = MedicalVolumeDataset(
        root=args.data_root,
        dataset=args.dataset,
        split_file=args.split_file,
        patch_size=tuple(args.patch_size),
        train=False,
        full_volume=args.eval_mode == "full",
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
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
        lambda_cs=0.0,
        lambda_scdl=0.0,
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    incompatible = model.load_state_dict(checkpoint["model"], strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        print(
            f"warning: non-strict checkpoint load, "
            f"missing={incompatible.missing_keys} "
            f"unexpected={incompatible.unexpected_keys}",
            flush=True,
        )

    metrics = evaluate(
        model=model,
        loader=loader,
        device=device,
        num_classes=args.num_classes,
        patch_size=tuple(args.patch_size),
        stride=tuple(args.eval_stride),
        eval_mode=args.eval_mode,
        max_batches=args.max_batches,
        postprocess_largest_cc=args.postprocess_largest_cc,
    )
    write_outputs(args, metrics)
    print(
        f"eval dataset={args.dataset} mode={args.eval_mode} "
        f"mean_dice={float(metrics['mean_dice']):.4f} "
        f"mean_hd95={float(metrics['mean_hd95']):.4f}",
        flush=True,
    )


def fill_defaults(args: argparse.Namespace) -> None:
    data_root = ROOT / "all-data"
    if args.dataset == "synapse":
        args.num_classes = args.num_classes or SYNAPSE_NUM_CLASSES_DHC
        args.data_root = args.data_root or data_root / "Synapse"
        args.split_file = args.split_file or data_root / "lists_Synapse_DHC" / "test_cases.txt"
    else:
        args.num_classes = args.num_classes or AMOS_NUM_CLASSES
        args.data_root = args.data_root or data_root / "AMOS"
        args.split_file = args.split_file or data_root / "amos_splits" / "test.txt"

    if args.eval_stride is None:
        args.eval_stride = tuple(max(1, size // 2) for size in args.patch_size)
    if args.output_dir is None:
        args.output_dir = args.checkpoint.parent / f"eval_{args.dataset}_{args.eval_mode}"


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    patch_size: tuple[int, int, int],
    stride: tuple[int, int, int],
    eval_mode: str,
    max_batches: int | None,
    postprocess_largest_cc: bool = False,
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
                raise ValueError("full eval requires batch-size=1.")
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
        if postprocess_largest_cc:
            preds_np = preds.detach().cpu().numpy()
            for sample in range(preds_np.shape[0]):
                preds_np[sample] = keep_largest_connected_component(
                    preds_np[sample], num_classes
                )
            preds = torch.from_numpy(preds_np).to(preds.device)
        metric.update(preds, targets)
    return metric.compute()


def write_outputs(args: argparse.Namespace, metrics: dict[str, torch.Tensor]) -> None:
    args_payload = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    metrics_payload = {
        "mean_dice": float(metrics["mean_dice"]),
        "mean_hd95": float(metrics["mean_hd95"]),
        "dice": [float(value) for value in metrics["dice"]],
        "hd95": [float(value) for value in metrics["hd95"]],
    }
    (args.output_dir / "eval_args.json").write_text(
        json.dumps(args_payload, indent=2, sort_keys=True)
    )
    (args.output_dir / "eval_metrics.json").write_text(
        json.dumps(metrics_payload, indent=2, sort_keys=True)
    )
    with (args.output_dir / "eval_metrics.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_index", "dice", "hd95"])
        for offset, (dice, hd95) in enumerate(
            zip(metrics_payload["dice"], metrics_payload["hd95"]), start=1
        ):
            writer.writerow([offset, dice, hd95])
        writer.writerow(["mean", metrics_payload["mean_dice"], metrics_payload["mean_hd95"]])


if __name__ == "__main__":
    main()
