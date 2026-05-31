"""Dataset utilities for MNIST and CIFAR-10.

The implementation intentionally avoids torchvision so that the project does not
load any ready-made CNN structure or pretrained weights. Only raw dataset files
are downloaded and parsed.
"""

from __future__ import annotations

import gzip
import os
import pickle
import random
import struct
import tarfile
import urllib.request
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


MNIST_URLS = {
    "train_images": "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
    "test_images": "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels": "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
}

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"


class DownloadError(RuntimeError):
    """Raised when a dataset file is unavailable and download is disabled/failed."""


def _download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        return
    print(f"Downloading {url} -> {target}")
    try:
        urllib.request.urlretrieve(url, target)
    except Exception as exc:  # pragma: no cover - network dependent
        raise DownloadError(
            f"Failed to download {url}. Please download it manually and place it at {target}."
        ) from exc


def _read_mnist_images(path: Path) -> torch.Tensor:
    with gzip.open(path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Invalid MNIST image file: {path}")
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
    images = torch.from_numpy(data.copy()).float().unsqueeze(1) / 255.0
    return images


def _read_mnist_labels(path: Path) -> torch.Tensor:
    with gzip.open(path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Invalid MNIST label file: {path}")
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    if labels.shape[0] != num:
        raise ValueError(f"Invalid label count in {path}")
    return torch.from_numpy(labels.copy()).long()


class MNISTDataset(Dataset):
    """MNIST dataset parser.

    Images are normalized with the common MNIST mean/std and padded from 28x28 to
    32x32 to match the original LeNet-5 input size.
    """

    def __init__(self, root: str | os.PathLike, train: bool = True, download: bool = True):
        self.root = Path(root) / "MNIST" / "raw"
        self.train = train
        files = {
            "train_images": self.root / "train-images-idx3-ubyte.gz",
            "train_labels": self.root / "train-labels-idx1-ubyte.gz",
            "test_images": self.root / "t10k-images-idx3-ubyte.gz",
            "test_labels": self.root / "t10k-labels-idx1-ubyte.gz",
        }
        if download:
            for key, url in MNIST_URLS.items():
                _download(url, files[key])
        missing = [str(p) for p in files.values() if not p.exists()]
        if missing:
            raise DownloadError("Missing MNIST files: " + ", ".join(missing))

        if train:
            self.images = _read_mnist_images(files["train_images"])
            self.labels = _read_mnist_labels(files["train_labels"])
        else:
            self.images = _read_mnist_images(files["test_images"])
            self.labels = _read_mnist_labels(files["test_labels"])

    def __len__(self) -> int:
        return int(self.labels.numel())

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = self.images[index]
        label = self.labels[index]
        image = F.pad(image, (2, 2, 2, 2))  # 1 x 32 x 32
        image = (image - 0.1307) / 0.3081
        return image, label


def _extract_cifar10(root: Path, archive: Path) -> None:
    if (root / "cifar-10-batches-py").exists():
        return
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(root)


def _load_cifar_batch(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with open(path, "rb") as f:
        batch = pickle.load(f, encoding="latin1")
    data = batch["data"].reshape(-1, 3, 32, 32)
    labels = np.array(batch["labels"], dtype=np.int64)
    return data, labels


class CIFAR10Dataset(Dataset):
    """CIFAR-10 parser with lightweight augmentation.

    Training augmentation: reflection padding + random crop + random horizontal
    flip. Test data only uses normalization.
    """

    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)

    def __init__(self, root: str | os.PathLike, train: bool = True, download: bool = True):
        self.root = Path(root) / "CIFAR10"
        self.train = train
        archive = self.root / "cifar-10-python.tar.gz"
        if download:
            _download(CIFAR10_URL, archive)
            _extract_cifar10(self.root, archive)
        base = self.root / "cifar-10-batches-py"
        if not base.exists():
            raise DownloadError(f"Missing CIFAR-10 directory: {base}")

        if train:
            arrays, labels = [], []
            for i in range(1, 6):
                x, y = _load_cifar_batch(base / f"data_batch_{i}")
                arrays.append(x)
                labels.append(y)
            data = np.concatenate(arrays, axis=0)
            targets = np.concatenate(labels, axis=0)
        else:
            data, targets = _load_cifar_batch(base / "test_batch")

        self.images = torch.from_numpy(data.copy()).float() / 255.0
        self.labels = torch.from_numpy(targets.copy()).long()

    def __len__(self) -> int:
        return int(self.labels.numel())

    def _augment(self, image: torch.Tensor) -> torch.Tensor:
        image = F.pad(image, (4, 4, 4, 4), mode="reflect")
        top = random.randint(0, 8)
        left = random.randint(0, 8)
        image = image[:, top : top + 32, left : left + 32]
        if random.random() < 0.5:
            image = torch.flip(image, dims=[2])
        return image

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = self.images[index]
        label = self.labels[index]
        if self.train:
            image = self._augment(image)
        image = (image - self.mean) / self.std
        return image, label
