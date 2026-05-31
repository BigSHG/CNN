"""Shared training and evaluation helpers."""

from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / self.count


def batch_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def train_one_epoch(
    model: nn.Module,
    loader: Iterable,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(batch_accuracy(logits, labels), batch_size)

    return {"loss": loss_meter.avg, "acc": acc_meter.avg}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: Iterable,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        batch_size = labels.size(0)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(batch_accuracy(logits, labels), batch_size)

    return {"loss": loss_meter.avg, "acc": acc_meter.avg}


def save_history_csv(history: List[Dict[str, float]], path: str | os.PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["epoch", "lr", "train_loss", "train_acc", "test_loss", "test_acc"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def save_json(data: Dict, path: str | os.PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def plot_curves(history: List[Dict[str, float]], output_dir: str | os.PathLike) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]

    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, [row["train_loss"] for row in history], marker="o", label="Training Loss")
    plt.plot(epochs, [row["test_loss"] for row in history], marker="s", label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.plot(epochs, [row["train_acc"] * 100 for row in history], marker="o", label="Training Accuracy")
    plt.plot(epochs, [row["test_acc"] * 100 for row in history], marker="s", label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Curve")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_curve.png", dpi=200)
    plt.close()


def run_training_loop(
    *,
    model: nn.Module,
    train_loader: Iterable,
    test_loader: Iterable,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    epochs: int,
    output_dir: str | os.PathLike,
    experiment_name: str,
) -> Dict[str, float]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history: List[Dict[str, float]] = []
    best_acc = -1.0
    best_epoch = 0

    model.to(device)
    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_metrics = evaluate(model, test_loader, criterion, device)

        if scheduler is not None:
            scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "test_loss": test_metrics["loss"],
            "test_acc": test_metrics["acc"],
        }
        history.append(row)
        print(
            f"[{experiment_name}] Epoch {epoch:03d}/{epochs:03d} "
            f"lr={current_lr:.6f} "
            f"train_loss={row['train_loss']:.4f} train_acc={row['train_acc']*100:.2f}% "
            f"test_loss={row['test_loss']:.4f} test_acc={row['test_acc']*100:.2f}%"
        )

        if row["test_acc"] > best_acc:
            best_acc = row["test_acc"]
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "test_acc": best_acc,
                },
                output_dir / "best_model.pth",
            )

    torch.save(model.state_dict(), output_dir / "last_model_state_dict.pth")
    save_history_csv(history, output_dir / "history.csv")
    plot_curves(history, output_dir)
    final = {
        "experiment": experiment_name,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "best_test_acc": best_acc,
        "final_test_acc": history[-1]["test_acc"],
        "final_train_acc": history[-1]["train_acc"],
        "final_test_loss": history[-1]["test_loss"],
    }
    save_json(final, output_dir / "final_metrics.json")
    print(f"Best test accuracy: {best_acc*100:.2f}% at epoch {best_epoch}")
    return final
