from typing import Optional, List

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None
    F = None


class GraphSAGEModel:
    def __init__(self, input_dim: int, hidden_dim: int = 32, output_dim: int = 1):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for GraphSAGEModel")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )

    def train(self, X, y, epochs: int = 5, lr: float = 0.01):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.BCELoss()
        for _ in range(epochs):
            optimizer.zero_grad()
            preds = self.model(X).squeeze()
            loss = loss_fn(preds, y)
            loss.backward()
            optimizer.step()

    def predict(self, X):
        with torch.no_grad():
            return self.model(X).squeeze().tolist()


class GATModel:
    def __init__(self, input_dim: int, hidden_dim: int = 32, output_dim: int = 1):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for GATModel")
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )

    def train(self, X, y, epochs: int = 5, lr: float = 0.01):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.BCELoss()
        for _ in range(epochs):
            optimizer.zero_grad()
            preds = self.model(X).squeeze()
            loss = loss_fn(preds, y)
            loss.backward()
            optimizer.step()

    def predict(self, X):
        with torch.no_grad():
            return self.model(X).squeeze().tolist()
