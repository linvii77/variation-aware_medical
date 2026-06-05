"""Build or run medical 3D ablation commands.

The default behavior is a dry run that prints reproducible commands for
``ce``, ``vapl``, ``scdl``, and ``combined``. Add ``--run`` only when you
intentionally want to launch training.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODES = ("ce", "vapl", "scdl", "combined")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--dataset", choices=["synapse", "amos"], default=None)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--val-split-file", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--patch-size", type=int, nargs=3, default=None)
    parser.add_argument("--foreground-prob", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--val-batch-size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--base-channels", type=int, default=None)
    parser.add_argument("--embedding-dim", type=int, default=None)
    parser.add_argument("--eval-mode", choices=["patch", "full"], default=None)
    parser.add_argument("--eval-stride", type=int, nargs=3, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute commands. Without this flag commands are only printed.",
    )
    return fill_defaults(load_config(parser.parse_args()))


def main() -> None:
    args = parse_args()
    commands = [build_command(args, mode) for mode in args.modes]

    for command in commands:
        print(format_command(command), flush=True)

    if not args.run:
        return

    for command in commands:
        subprocess.run(command, cwd=ROOT, check=True)


def load_config(args: argparse.Namespace) -> argparse.Namespace:
    if args.config is None:
        return args
    payload = json.loads(args.config.read_text())
    for key, value in payload.items():
        attr = key.replace("-", "_")
        if not hasattr(args, attr):
            raise ValueError(f"Unknown config key: {key}")
        current = getattr(args, attr)
        if current is None or current is False:
            setattr(args, attr, value)
    return args


def fill_defaults(args: argparse.Namespace) -> argparse.Namespace:
    defaults = {
        "dataset": "synapse",
        "modes": list(MODES),
        "output_root": ROOT / "outputs" / "ablations",
        "patch_size": (96, 96, 96),
        "foreground_prob": 0.75,
        "batch_size": 1,
        "val_batch_size": 1,
        "workers": 4,
        "max_iters": 1000,
        "lr": 1.0e-3,
        "weight_decay": 1.0e-5,
        "base_channels": 16,
        "embedding_dim": 256,
        "eval_mode": "patch",
        "eval_interval": 500,
        "save_interval": 500,
        "seed": 42,
    }
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    for key in ("data_root", "split_file", "val_split_file", "output_root", "resume"):
        value = getattr(args, key)
        if value is not None and not isinstance(value, Path):
            setattr(args, key, Path(value))
    return args


def build_command(args: argparse.Namespace, mode: str) -> list[str]:
    output_dir = args.output_root / args.dataset / mode
    command = [
        sys.executable,
        "tools/train_medical_3d.py",
        "--dataset",
        args.dataset,
        "--mode",
        mode,
        "--output-dir",
        str(output_dir),
        "--patch-size",
        *map(str, args.patch_size),
        "--foreground-prob",
        str(args.foreground_prob),
        "--batch-size",
        str(args.batch_size),
        "--val-batch-size",
        str(args.val_batch_size),
        "--workers",
        str(args.workers),
        "--max-iters",
        str(args.max_iters),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--base-channels",
        str(args.base_channels),
        "--embedding-dim",
        str(args.embedding_dim),
        "--eval-mode",
        args.eval_mode,
        "--eval-interval",
        str(args.eval_interval),
        "--save-interval",
        str(args.save_interval),
        "--seed",
        str(args.seed),
    ]

    optional_path_args = {
        "--data-root": args.data_root,
        "--split-file": args.split_file,
        "--val-split-file": args.val_split_file,
    }
    for flag, value in optional_path_args.items():
        if value is not None:
            command.extend([flag, str(value)])

    if args.eval_stride is not None:
        command.extend(["--eval-stride", *map(str, args.eval_stride)])
    if args.max_val_batches is not None:
        command.extend(["--max-val-batches", str(args.max_val_batches)])
    if args.resume is not None:
        command.extend(["--resume", str(args.resume)])
    if args.amp:
        command.append("--amp")
    if args.no_eval:
        command.append("--no-eval")
    return command


def format_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


if __name__ == "__main__":
    main()
