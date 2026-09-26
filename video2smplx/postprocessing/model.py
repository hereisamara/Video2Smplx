"""Neural-network loading helpers shared by correction inference programs."""

from __future__ import annotations


def import_torch():
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise SystemExit("PyTorch is required for learned SMPL-X correction.") from exc
    return torch, nn, DataLoader, TensorDataset


def build_model(nn, input_dim: int, hidden_dim: int, layers: int, dropout: float):
    modules = []
    dim = input_dim
    for _ in range(max(1, layers)):
        modules.extend(
            [
                nn.Linear(dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
        )
        dim = hidden_dim
    modules.append(nn.Linear(dim, 6))
    return nn.Sequential(*modules)
