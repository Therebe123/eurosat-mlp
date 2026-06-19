from __future__ import annotations

from pathlib import Path

import numpy as np


def softmax_cross_entropy(logits: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / np.sum(exp, axis=1, keepdims=True)
    n = logits.shape[0]
    loss = -np.log(probs[np.arange(n), y] + 1e-12).mean()
    grad = probs
    grad[np.arange(n), y] -= 1.0
    grad /= n
    return float(loss), grad.astype(np.float32, copy=False)


class ThreeLayerMLP:
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        activation: str = "relu",
        seed: int = 42,
    ):
        if activation not in {"relu", "tanh"}:
            raise ValueError("activation must be 'relu' or 'tanh'")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.activation = activation
        rng = np.random.default_rng(seed)
        self.W1 = self._init_weight(rng, self.input_dim, self.hidden_dim)
        self.b1 = np.zeros(self.hidden_dim, dtype=np.float32)
        self.W2 = self._init_weight(rng, self.hidden_dim, self.hidden_dim)
        self.b2 = np.zeros(self.hidden_dim, dtype=np.float32)
        self.W3 = self._init_weight(rng, self.hidden_dim, self.output_dim)
        self.b3 = np.zeros(self.output_dim, dtype=np.float32)
        self.grads: dict[str, np.ndarray] = {}
        self.cache: dict[str, np.ndarray] = {}

    def _init_weight(self, rng: np.random.Generator, fan_in: int, fan_out: int) -> np.ndarray:
        scale = np.sqrt(2.0 / fan_in) if self.activation == "relu" else np.sqrt(1.0 / fan_in)
        return rng.normal(0.0, scale, size=(fan_in, fan_out)).astype(np.float32)

    def _activate(self, x: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            return np.maximum(x, 0.0)
        return np.tanh(x)

    def _activation_backward(self, grad: np.ndarray, pre_activation: np.ndarray) -> np.ndarray:
        if self.activation == "relu":
            return grad * (pre_activation > 0)
        activated = np.tanh(pre_activation)
        return grad * (1.0 - activated * activated)

    def forward(self, x: np.ndarray) -> np.ndarray:
        z1 = x @ self.W1 + self.b1
        a1 = self._activate(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = self._activate(z2)
        logits = a2 @ self.W3 + self.b3
        self.cache = {"x": x, "z1": z1, "a1": a1, "z2": z2, "a2": a2}
        return logits

    def l2_loss(self) -> float:
        return 0.5 * float(np.sum(self.W1 * self.W1) + np.sum(self.W2 * self.W2) + np.sum(self.W3 * self.W3))

    def backward(self, grad_logits: np.ndarray, weight_decay: float = 0.0) -> None:
        x = self.cache["x"]
        z1 = self.cache["z1"]
        a1 = self.cache["a1"]
        z2 = self.cache["z2"]
        a2 = self.cache["a2"]

        dW3 = a2.T @ grad_logits + weight_decay * self.W3
        db3 = np.sum(grad_logits, axis=0)
        da2 = grad_logits @ self.W3.T
        dz2 = self._activation_backward(da2, z2)
        dW2 = a1.T @ dz2 + weight_decay * self.W2
        db2 = np.sum(dz2, axis=0)
        da1 = dz2 @ self.W2.T
        dz1 = self._activation_backward(da1, z1)
        dW1 = x.T @ dz1 + weight_decay * self.W1
        db1 = np.sum(dz1, axis=0)

        self.grads = {
            "W1": dW1.astype(np.float32, copy=False),
            "b1": db1.astype(np.float32, copy=False),
            "W2": dW2.astype(np.float32, copy=False),
            "b2": db2.astype(np.float32, copy=False),
            "W3": dW3.astype(np.float32, copy=False),
            "b3": db3.astype(np.float32, copy=False),
        }

    def step(self, lr: float) -> None:
        for name in self.params():
            setattr(self, name, getattr(self, name) - lr * self.grads[name])

    def params(self) -> dict[str, np.ndarray]:
        return {
            "W1": self.W1,
            "b1": self.b1,
            "W2": self.W2,
            "b2": self.b2,
            "W3": self.W3,
            "b3": self.b3,
        }

    def predict(self, x: np.ndarray, batch_size: int = 512) -> np.ndarray:
        preds: list[np.ndarray] = []
        for start in range(0, len(x), batch_size):
            logits = self.forward(x[start : start + batch_size])
            preds.append(np.argmax(logits, axis=1))
        return np.concatenate(preds).astype(np.int64)

    def save(self, path: str | Path, class_names: list[str] | None = None) -> None:
        payload = {
            "input_dim": np.array(self.input_dim),
            "hidden_dim": np.array(self.hidden_dim),
            "output_dim": np.array(self.output_dim),
            "activation": np.array(self.activation),
            **self.params(),
        }
        if class_names is not None:
            payload["class_names"] = np.asarray(class_names)
        np.savez(path, **payload)

    @classmethod
    def load(cls, path: str | Path) -> "ThreeLayerMLP":
        data = np.load(path, allow_pickle=True)
        model = cls(
            input_dim=int(data["input_dim"]),
            hidden_dim=int(data["hidden_dim"]),
            output_dim=int(data["output_dim"]),
            activation=str(data["activation"]),
        )
        for name in model.params():
            setattr(model, name, data[name].astype(np.float32, copy=False))
        return model


def accuracy(model: ThreeLayerMLP, x: np.ndarray, y: np.ndarray) -> float:
    if len(y) == 0:
        return 0.0
    return float(np.mean(model.predict(x) == y))
