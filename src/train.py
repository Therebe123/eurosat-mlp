from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import INPUT_DIM, batch_iterator, load_splits
from src.model import ThreeLayerMLP, accuracy, softmax_cross_entropy


def evaluate_loss(model: ThreeLayerMLP, x: np.ndarray, y: np.ndarray, weight_decay: float, batch_size: int = 512) -> float:
    if len(y) == 0:
        return 0.0
    losses: list[float] = []
    counts: list[int] = []
    for start in range(0, len(y), batch_size):
        xb = x[start : start + batch_size]
        yb = y[start : start + batch_size]
        logits = model.forward(xb)
        loss, _ = softmax_cross_entropy(logits, yb)
        losses.append(loss + weight_decay * model.l2_loss())
        counts.append(len(yb))
    return float(np.average(losses, weights=counts))


def train(
    data_dir: str | Path,
    output_dir: str | Path,
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 0.01,
    hidden_dim: int = 128,
    weight_decay: float = 1e-4,
    activation: str = "relu",
    lr_decay: float = 0.95,
    max_per_class: int | None = None,
    seed: int = 42,
) -> dict[str, float]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = load_splits(data_dir, max_per_class=max_per_class, seed=seed)
    model = ThreeLayerMLP(
        input_dim=INPUT_DIM,
        hidden_dim=hidden_dim,
        output_dim=len(splits.class_names),
        activation=activation,
        seed=seed,
    )
    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "learning_rate": [],
    }
    best_val_accuracy = -1.0
    current_lr = lr
    config = {
        "data_dir": str(data_dir),
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "hidden_dim": hidden_dim,
        "weight_decay": weight_decay,
        "activation": activation,
        "lr_decay": lr_decay,
        "max_per_class": max_per_class,
        "seed": seed,
        "class_names": splits.class_names,
    }

    for epoch in range(1, epochs + 1):
        train_losses: list[float] = []
        for xb, yb in batch_iterator(
            splits.x_train,
            splits.y_train,
            batch_size=batch_size,
            shuffle=True,
            seed=seed + epoch,
        ):
            logits = model.forward(xb)
            loss, grad = softmax_cross_entropy(logits, yb)
            loss += weight_decay * model.l2_loss()
            model.backward(grad, weight_decay=weight_decay)
            model.step(current_lr)
            train_losses.append(loss)

        train_loss = float(np.mean(train_losses))
        val_loss = evaluate_loss(model, splits.x_val, splits.y_val, weight_decay)
        val_acc = accuracy(model, splits.x_val, splits.y_val)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)
        history["learning_rate"].append(current_lr)

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            model.save(output_dir / "best_model.npz", class_names=splits.class_names)
            (output_dir / "best_config.json").write_text(
                json.dumps({**config, "best_epoch": epoch, "best_val_accuracy": best_val_accuracy}, indent=2),
                encoding="utf-8",
            )

        print(
            f"epoch={epoch:03d} lr={current_lr:.6f} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        current_lr *= lr_decay

    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return {"best_val_accuracy": float(best_val_accuracy)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a NumPy three-layer MLP on EuroSAT RGB.")
    parser.add_argument("--data_dir", type=Path, default=Path("EuroSAT_RGB"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--activation", choices=["relu", "tanh"], default="relu")
    parser.add_argument("--lr_decay", type=float, default=0.95)
    parser.add_argument("--max_per_class", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(**vars(args))
