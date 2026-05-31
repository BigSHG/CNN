from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run both MNIST and CIFAR-10 experiments")
    parser.add_argument("--data-dir", default=".data")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--mnist-epochs", type=int, default=10)
    parser.add_argument("--cifar-epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    root = Path(args.output_dir)
    py = sys.executable
    run([
        py,
        "src/train_mnist.py",
        "--data-dir",
        args.data_dir,
        "--output-dir",
        str(root / "mnist"),
        "--epochs",
        str(args.mnist_epochs),
        "--batch-size",
        str(args.batch_size),
        "--device",
        args.device,
        "--num-workers",
        str(args.num_workers),
    ])
    run([
        py,
        "src/train_cifar10.py",
        "--data-dir",
        args.data_dir,
        "--output-dir",
        str(root / "cifar10"),
        "--epochs",
        str(args.cifar_epochs),
        "--batch-size",
        str(args.batch_size),
        "--device",
        args.device,
        "--num-workers",
        str(args.num_workers),
    ])


if __name__ == "__main__":
    main()
