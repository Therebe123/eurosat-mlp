from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_SIZE = (64, 64)
IMAGE_CHANNELS = 3
INPUT_DIM = IMAGE_SIZE[0] * IMAGE_SIZE[1] * IMAGE_CHANNELS


@dataclass(frozen=True)
class DatasetSplit:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    train_paths: list[str]
    val_paths: list[str]
    test_paths: list[str]
    class_names: list[str]


def class_directories(data_dir: str | Path) -> list[Path]:
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Data directory not found: {root}")
    dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    if not dirs:
        raise ValueError(f"No class directories found under {root}")
    return dirs


def load_dataset(
    data_dir: str | Path,
    max_per_class: int | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Load EuroSAT RGB images without resizing and flatten to 12288 features."""
    rng = np.random.default_rng(seed)
    xs: list[np.ndarray] = []
    ys: list[int] = []
    paths: list[str] = []
    dirs = class_directories(data_dir)
    class_names = [d.name for d in dirs]

    for label, class_dir in enumerate(dirs):
        image_paths = sorted(
            [
                p
                for p in class_dir.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
            ]
        )
        if max_per_class is not None:
            chosen = rng.permutation(len(image_paths))[:max_per_class]
            image_paths = [image_paths[i] for i in sorted(chosen)]
        for image_path in image_paths:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                if image.size != IMAGE_SIZE:
                    raise ValueError(
                        f"Expected image size {IMAGE_SIZE}, got {image.size} for {image_path}"
                    )
                arr = np.asarray(image, dtype=np.float32).reshape(-1) / 255.0
            xs.append(arr)
            ys.append(label)
            paths.append(str(image_path))

    if not xs:
        raise ValueError(f"No images loaded from {data_dir}")
    x = np.stack(xs).astype(np.float32, copy=False)
    y = np.asarray(ys, dtype=np.int64)
    return x, y, class_names, paths


def stratified_split(
    y: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    for label in np.unique(y):
        label_idx = np.where(y == label)[0]
        rng.shuffle(label_idx)
        n = len(label_idx)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        if n_train + n_val > n:
            n_val = max(0, n - n_train)
        train_idx.extend(label_idx[:n_train].tolist())
        val_idx.extend(label_idx[n_train : n_train + n_val].tolist())
        test_idx.extend(label_idx[n_train + n_val :].tolist())

    for values in (train_idx, val_idx, test_idx):
        rng.shuffle(values)
    return {
        "train": np.asarray(train_idx, dtype=np.int64),
        "val": np.asarray(val_idx, dtype=np.int64),
        "test": np.asarray(test_idx, dtype=np.int64),
    }


def load_splits(
    data_dir: str | Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    max_per_class: int | None = None,
    seed: int = 42,
) -> DatasetSplit:
    x, y, class_names, paths = load_dataset(data_dir, max_per_class=max_per_class, seed=seed)
    split = stratified_split(y, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)
    return DatasetSplit(
        x_train=x[split["train"]],
        y_train=y[split["train"]],
        x_val=x[split["val"]],
        y_val=y[split["val"]],
        x_test=x[split["test"]],
        y_test=y[split["test"]],
        train_paths=[paths[i] for i in split["train"]],
        val_paths=[paths[i] for i in split["val"]],
        test_paths=[paths[i] for i in split["test"]],
        class_names=class_names,
    )


def batch_iterator(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
    seed: int | None = None,
):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(y))
    if shuffle:
        rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        yield x[batch_idx], y[batch_idx]
