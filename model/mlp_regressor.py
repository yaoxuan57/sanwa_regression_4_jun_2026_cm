import torch
import torch.nn as nn


class MLPHead(nn.Module):
    """Notebook regression head: n_inputs -> 64 -> 32 -> n_outputs."""

    def __init__(self, n_inputs: int, n_outputs: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_inputs, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, n_outputs),
        )

    def forward(self, x):
        return self.net(x)


class MLPRegressor(nn.Module):
    """Standalone notebook model on flattened tabular features (no transformer encoder)."""

    def __init__(self, n_inputs: int, n_outputs: int, dropout: float = 0.1):
        super().__init__()
        self.head = MLPHead(n_inputs, n_outputs, dropout=dropout)

    def _to_features(self, x):
        if x.ndim == 3:
            # Tabular values are repeated along time; collapse to one value per feature.
            return x.mean(dim=-1)
        return x

    def forward(self, x):
        return self.head(self._to_features(x))

    def predict(self, x):
        return self.forward(x)
