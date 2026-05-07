"""Symmetric MLP AutoEncoder anomaly detector (PyTorch, CPU-only)."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, TensorDataset

from src.models.base import BaseDetector


class _MLP(nn.Module):
    """Symmetric MLP: n_in → h0 → h1 → h0 → n_in."""

    def __init__(self, n_in: int, hidden: tuple[int, int]) -> None:
        super().__init__()
        h0, h1 = hidden
        self.net = nn.Sequential(
            nn.Linear(n_in, h0),
            nn.ReLU(),
            nn.Linear(h0, h1),
            nn.ReLU(),
            nn.Linear(h1, h0),
            nn.ReLU(),
            nn.Linear(h0, n_in),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AutoEncoderDetector(BaseDetector):
    """Symmetric MLP autoencoder with early-stopping on validation MSE.

    Score = per-sample MSE between input and reconstruction (higher = more anomalous).
    Trains on CPU only — IMS-sized feature matrices fit comfortably in memory.
    """

    def __init__(
        self,
        hidden: tuple[int, int] = (16, 8),
        lr: float = 1e-3,
        batch_size: int = 32,
        max_epochs: int = 200,
        patience: int = 15,
        val_split: float = 0.15,
        random_state: int = 42,
    ) -> None:
        self.hidden = hidden
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.val_split = val_split
        self.random_state = random_state
        self._scaler: RobustScaler | None = None
        self._model: _MLP | None = None
        self._n_features: int | None = None

    def fit(self, X_healthy: np.ndarray) -> AutoEncoderDetector:
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self._scaler = RobustScaler()
        X_scaled = self._scaler.fit_transform(X_healthy).astype(np.float32)
        self._n_features = X_scaled.shape[1]

        # Train / val split
        n_val = max(1, int(len(X_scaled) * self.val_split))
        idx = np.random.permutation(len(X_scaled))
        X_val = X_scaled[idx[:n_val]]
        X_tr = X_scaled[idx[n_val:]]

        device = torch.device("cpu")
        self._model = _MLP(self._n_features, self.hidden).to(device)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        tr_tensor = torch.from_numpy(X_tr)
        val_tensor = torch.from_numpy(X_val)
        loader = DataLoader(TensorDataset(tr_tensor), batch_size=self.batch_size, shuffle=True)

        best_val = float("inf")
        best_state = None
        no_improve = 0

        self._model.train()
        for _ in range(self.max_epochs):
            for (batch,) in loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                loss_fn(self._model(batch), batch).backward()
                optimizer.step()

            self._model.eval()
            with torch.no_grad():
                val_loss = loss_fn(self._model(val_tensor), val_tensor).item()
            self._model.train()

            if val_loss < best_val - 1e-7:
                best_val = val_loss
                best_state = {k: v.clone() for k, v in self._model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break

        if best_state is not None:
            self._model.load_state_dict(best_state)
        self._model.eval()
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Per-sample MSE reconstruction error (higher = more anomalous)."""
        assert self._model is not None and self._scaler is not None, "call fit() first"
        X_scaled = self._scaler.transform(X).astype(np.float32)
        tensor = torch.from_numpy(X_scaled)
        with torch.no_grad():
            recon = self._model(tensor).numpy()
        return np.mean((X_scaled - recon) ** 2, axis=1)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        assert self._model is not None
        joblib.dump(
            {
                "state_dict": self._model.state_dict(),
                "scaler": self._scaler,
                "n_features": self._n_features,
                "hidden": self.hidden,
                "lr": self.lr,
                "batch_size": self.batch_size,
                "max_epochs": self.max_epochs,
                "patience": self.patience,
                "val_split": self.val_split,
                "random_state": self.random_state,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path) -> AutoEncoderDetector:
        data = joblib.load(Path(path))
        obj = cls(
            hidden=data["hidden"],
            lr=data["lr"],
            batch_size=data["batch_size"],
            max_epochs=data["max_epochs"],
            patience=data["patience"],
            val_split=data["val_split"],
            random_state=data["random_state"],
        )
        obj._scaler = data["scaler"]
        obj._n_features = data["n_features"]
        obj._model = _MLP(data["n_features"], data["hidden"])
        obj._model.load_state_dict(data["state_dict"])
        obj._model.eval()
        return obj
