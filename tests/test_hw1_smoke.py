import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class HW1SmokeTests(unittest.TestCase):
    def test_data_loader_keeps_original_64x64_rgb_shape(self):
        from src.data import load_dataset, stratified_split

        x, y, class_names, paths = load_dataset(
            ROOT / "EuroSAT_RGB",
            max_per_class=2,
            seed=123,
        )

        self.assertEqual(x.shape, (20, 64 * 64 * 3))
        self.assertEqual(y.shape, (20,))
        self.assertEqual(len(paths), 20)
        self.assertEqual(len(class_names), 10)
        self.assertGreaterEqual(float(x.min()), 0.0)
        self.assertLessEqual(float(x.max()), 1.0)

        splits = stratified_split(y, train_ratio=0.5, val_ratio=0.25, seed=123)
        self.assertEqual(set(splits), {"train", "val", "test"})
        self.assertEqual(
            len(splits["train"]) + len(splits["val"]) + len(splits["test"]),
            len(y),
        )

    def test_mlp_backward_updates_all_parameters(self):
        from src.model import ThreeLayerMLP, softmax_cross_entropy

        rng = np.random.default_rng(123)
        model = ThreeLayerMLP(
            input_dim=64 * 64 * 3,
            hidden_dim=8,
            output_dim=10,
            activation="relu",
            seed=123,
        )
        x = rng.random((4, 64 * 64 * 3), dtype=np.float32)
        y = np.array([0, 1, 2, 3], dtype=np.int64)

        logits = model.forward(x)
        loss, grad_logits = softmax_cross_entropy(logits, y)
        self.assertTrue(np.isfinite(loss))

        model.backward(grad_logits, weight_decay=1e-4)
        before = {name: value.copy() for name, value in model.params().items()}
        model.step(lr=0.01)

        for name, old_value in before.items():
            self.assertFalse(np.allclose(old_value, model.params()[name]), name)

    def test_train_one_epoch_writes_checkpoint_and_history(self):
        from src.train import train

        output_dir = ROOT / "outputs_test"
        if output_dir.exists():
            for path in output_dir.glob("*"):
                path.unlink()
        metrics = train(
            data_dir=ROOT / "EuroSAT_RGB",
            output_dir=output_dir,
            epochs=1,
            batch_size=8,
            lr=0.01,
            hidden_dim=8,
            weight_decay=1e-4,
            activation="tanh",
            lr_decay=0.95,
            max_per_class=4,
            seed=123,
        )

        self.assertIn("best_val_accuracy", metrics)
        self.assertTrue((output_dir / "best_model.npz").exists())
        self.assertTrue((output_dir / "history.json").exists())
        self.assertTrue((output_dir / "best_config.json").exists())
        history = json.loads((output_dir / "history.json").read_text(encoding="utf-8"))
        self.assertEqual(len(history["train_loss"]), 1)


if __name__ == "__main__":
    unittest.main()
