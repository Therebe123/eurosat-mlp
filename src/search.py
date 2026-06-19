from __future__ import annotations

import argparse
import csv
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train import train


def run_search(
    data_dir: str | Path,
    output_dir: str | Path = "outputs",
    epochs: int = 8,
    batch_size: int = 128,
    lr_decay: float = 0.95,
    max_per_class: int | None = None,
    seed: int = 42,
    quick: bool = False,
) -> list[dict[str, str | float | int]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    learning_rates = [0.01, 0.005, 0.001]
    hidden_dims = [128, 256]
    weight_decays = [0.0, 1e-4]
    activations = ["relu", "tanh"]
    combos = list(itertools.product(learning_rates, hidden_dims, weight_decays, activations))
    if quick:
        combos = combos[:6]
        epochs = min(epochs, 3)
        max_per_class = max_per_class or 12

    rows: list[dict[str, str | float | int]] = []
    for run_id, (lr, hidden_dim, weight_decay, activation) in enumerate(combos, start=1):
        run_dir = output_dir / f"search_run_{run_id:02d}"
        print(
            f"[search {run_id}/{len(combos)}] lr={lr} hidden_dim={hidden_dim} "
            f"weight_decay={weight_decay} activation={activation}"
        )
        metrics = train(
            data_dir=data_dir,
            output_dir=run_dir,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            hidden_dim=hidden_dim,
            weight_decay=weight_decay,
            activation=activation,
            lr_decay=lr_decay,
            max_per_class=max_per_class,
            seed=seed,
        )
        rows.append(
            {
                "run_id": run_id,
                "lr": lr,
                "hidden_dim": hidden_dim,
                "weight_decay": weight_decay,
                "activation": activation,
                "best_val_accuracy": metrics["best_val_accuracy"],
                "checkpoint": str(run_dir / "best_model.npz"),
            }
        )

    rows.sort(key=lambda row: float(row["best_val_accuracy"]), reverse=True)
    result_path = output_dir / "search_results.csv"
    with result_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"search_results={result_path}")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small hyperparameter search.")
    parser.add_argument("--data_dir", type=Path, default=Path("EuroSAT_RGB"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr_decay", type=float, default=0.95)
    parser.add_argument("--max_per_class", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_search(**vars(args))
