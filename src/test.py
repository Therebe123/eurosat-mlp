from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_splits
from src.model import ThreeLayerMLP


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(y_true, y_pred):
        matrix[int(true_label), int(pred_label)] += 1
    return matrix


def plot_confusion_matrix(matrix: np.ndarray, class_names: list[str], output_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")
    max_value = int(matrix.max()) if matrix.size else 0
    threshold = max_value / 2.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = "white" if matrix[i, j] > threshold else "black"
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color=color, fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def evaluate(
    data_dir: str | Path,
    checkpoint: str | Path,
    output_dir: str | Path = "outputs",
    max_per_class: int | None = None,
    seed: int = 42,
) -> dict[str, float]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = load_splits(data_dir, max_per_class=max_per_class, seed=seed)
    model = ThreeLayerMLP.load(checkpoint)
    y_pred = model.predict(splits.x_test)
    test_accuracy = float(np.mean(y_pred == splits.y_test)) if len(splits.y_test) else 0.0
    matrix = confusion_matrix(splits.y_test, y_pred, len(splits.class_names))
    plot_confusion_matrix(matrix, splits.class_names, output_dir / "confusion_matrix.png")

    records = []
    for path, true_label, pred_label in zip(splits.test_paths, splits.y_test, y_pred):
        records.append(
            {
                "path": path,
                "true_index": int(true_label),
                "pred_index": int(pred_label),
                "true_label": splits.class_names[int(true_label)],
                "pred_label": splits.class_names[int(pred_label)],
            }
        )
    (output_dir / "test_predictions.json").write_text(
        json.dumps(
            {
                "test_accuracy": test_accuracy,
                "class_names": splits.class_names,
                "predictions": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"test_accuracy={test_accuracy:.4f}")
    print(f"confusion_matrix={output_dir / 'confusion_matrix.png'}")
    return {"test_accuracy": test_accuracy}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained NumPy MLP checkpoint.")
    parser.add_argument("--data_dir", type=Path, default=Path("EuroSAT_RGB"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/best_model.npz"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"))
    parser.add_argument("--max_per_class", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(**vars(args))
