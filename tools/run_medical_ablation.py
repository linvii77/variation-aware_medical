"""Build or run medical 3D ablation commands.

The default behavior is a dry run that prints reproducible commands for
``ce``, ``vapl``, ``scdl``, and ``combined``. Add ``--run`` only when you
intentionally want to launch training.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODES = ("ce", "vapl", "scdl", "combined")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["synapse", "amos"], default="synapse")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--split-file", type=Path, default=None)
    parser.add_argument("--val-split-file", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "ablations")
    parser.add_argument("--patch-size", type=int, nargs=3, default=(96, 96, 96))
    parser.add_argument("--foreground-prob", type=float, default=0.75)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--val-batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-iters", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--eval-mode", choices=["patch", "full"], default="patch")
    parser.add_argument("--eval-stride", type=int, nargs=3, default=None)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute commands. Without this flag commands are only printed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    commands = [build_command(args, mode) for mode in args.modes]

    for command in commands:
        print(format_command(command), flush=True)

    if not args.run:
        return

    for command in commands:
        subprocess.run(command, cwd=ROOT, check=True)


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
    if args.amp:
        command.append("--amp")
    if args.no_eval:
        command.append("--no-eval")
    return command


def format_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


if __name__ == "__main__":
    main()
