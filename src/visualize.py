from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def plot_history(history_path: str | Path, output_dir: str | Path) -> None:
    history_path = Path(history_path)
    output_dir = Path(output_dir)
    if not history_path.exists():
        print(f"skip history plots: {history_path} not found")
        return
    history = json.loads(history_path.read_text(encoding="utf-8"))
    epochs = np.arange(1, len(history.get("train_loss", [])) + 1)
    if len(epochs) == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["train_loss"], marker="o", label="Train Loss")
    ax.plot(epochs, history["val_loss"], marker="o", label="Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "loss_curve.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history["val_accuracy"], marker="o", color="tab:green")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("Validation Accuracy")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_curve.png", dpi=180)
    plt.close(fig)


def plot_first_layer_weights(checkpoint: str | Path, output_dir: str | Path, image_size: int = 64) -> None:
    checkpoint = Path(checkpoint)
    output_dir = Path(output_dir)
    if not checkpoint.exists():
        print(f"skip weight plot: {checkpoint} not found")
        return
    data = np.load(checkpoint, allow_pickle=True)
    W1 = data["W1"]
    scores = np.linalg.norm(W1, axis=0)
    chosen = np.argsort(scores)[-16:][::-1]
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for ax, index in zip(axes.flat, chosen):
        weight_img = W1[:, index].reshape(image_size, image_size, 3)
        lo = float(weight_img.min())
        hi = float(weight_img.max())
        if hi > lo:
            weight_img = (weight_img - lo) / (hi - lo)
        else:
            weight_img = np.zeros_like(weight_img)
        ax.imshow(np.clip(weight_img, 0, 1))
        ax.set_title(f"h{index}", fontsize=8)
        ax.axis("off")
    fig.suptitle("First Layer Weight Patterns")
    fig.tight_layout()
    fig.savefig(output_dir / "first_layer_weights.png", dpi=180)
    plt.close(fig)


def plot_error_examples(predictions_path: str | Path, output_dir: str | Path, limit: int = 6) -> None:
    predictions_path = Path(predictions_path)
    output_dir = Path(output_dir)
    if not predictions_path.exists():
        print(f"skip error examples: {predictions_path} not found")
        return
    payload = json.loads(predictions_path.read_text(encoding="utf-8"))
    errors = [r for r in payload["predictions"] if r["true_index"] != r["pred_index"]][:limit]
    if not errors:
        errors = payload["predictions"][: min(limit, len(payload["predictions"]))]
    if not errors:
        return

    cols = 3
    rows = int(np.ceil(len(errors) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(10, 3.5 * rows))
    axes_array = np.asarray(axes).reshape(-1)
    for ax in axes_array:
        ax.axis("off")
    for ax, record in zip(axes_array, errors):
        with Image.open(record["path"]) as image:
            ax.imshow(image.convert("RGB"))
        ax.set_title(
            f"true={record['true_label']}\npred={record['pred_label']}",
            fontsize=9,
        )
        ax.axis("off")
    fig.suptitle("Misclassified Test Examples")
    fig.tight_layout()
    fig.savefig(output_dir / "error_examples.png", dpi=180)
    plt.close(fig)


def visualize(
    data_dir: str | Path = "EuroSAT_RGB",
    output_dir: str | Path = "outputs",
    history: str | Path | None = None,
    checkpoint: str | Path | None = None,
    predictions: str | Path | None = None,
) -> None:
    del data_dir
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history = Path(history) if history else output_dir / "history.json"
    checkpoint = Path(checkpoint) if checkpoint else output_dir / "best_model.npz"
    predictions = Path(predictions) if predictions else output_dir / "test_predictions.json"
    plot_history(history, output_dir)
    plot_first_layer_weights(checkpoint, output_dir)
    plot_error_examples(predictions, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate HW1 report visualizations.")
    parser.add_argument("--data_dir", type=Path, default=Path("EuroSAT_RGB"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"))
    parser.add_argument("--history", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--predictions", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    visualize(**vars(args))
