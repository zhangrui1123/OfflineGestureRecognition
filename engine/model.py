"""ST-GCN + MS-TCN style model for offline gesture segmentation."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.features import NUM_NODES
from utils.labels import NUM_CLASSES

# Re-export post-processing for backward compatibility.
from engine.postprocess import probabilities_to_events  # noqa: F401

EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
    (4, 8), (4, 12), (4, 16), (4, 20),
]


def build_adjacency(num_nodes: int = NUM_NODES, edges: list[tuple[int, int]] | None = None) -> torch.Tensor:
    edges = EDGES if edges is None else edges
    a = np.eye(num_nodes, dtype=np.float32)
    for i, j in edges:
        a[i, j] = 1.0
        a[j, i] = 1.0
    d = a.sum(axis=1)
    d_inv = np.where(d > 0, d ** -0.5, 0.0)
    return torch.from_numpy(np.diag(d_inv) @ a @ np.diag(d_inv))


class AdaptiveGraphConv(nn.Module):
    """Graph convolution with a fixed anatomical graph plus a learned residual graph."""

    def __init__(self, c_in: int, c_out: int, adjacency: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("a_fixed", adjacency)
        self.a_residual = nn.Parameter(torch.zeros_like(adjacency))
        self.proj = nn.Conv2d(c_in, c_out, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.a_fixed + torch.tanh(self.a_residual) * 0.25
        x = self.proj(x)
        return torch.einsum("bctv,vw->bctw", x, a)


class STGCNBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, adjacency: torch.Tensor, dropout: float = 0.0) -> None:
        super().__init__()
        self.gcn = AdaptiveGraphConv(c_in, c_out, adjacency)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, kernel_size=(9, 1), padding=(4, 0)),
            nn.BatchNorm2d(c_out),
            nn.Dropout(dropout),
        )
        self.residual = (
            nn.Sequential(nn.Conv2d(c_in, c_out, 1), nn.BatchNorm2d(c_out))
            if c_in != c_out
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.tcn(self.gcn(x)) + self.residual(x), inplace=True)


class STGCNEncoder(nn.Module):
    def __init__(self, input_channels: int, hidden_dim: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        a = build_adjacency()
        self.blocks = nn.Sequential(
            STGCNBlock(input_channels, 48, a, dropout=0.0),
            STGCNBlock(48, 96, a, dropout=dropout * 0.5),
            STGCNBlock(96, hidden_dim, a, dropout=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B,C,T,V] -> [B,D,T]
        return self.blocks(x).mean(dim=-1)


class DilatedResidualLayer(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class TemporalStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: int,
        num_layers: int,
        num_classes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Conv1d(in_channels, channels, 1)
        self.layers = nn.ModuleList(
            DilatedResidualLayer(channels, 2 ** i, dropout)
            for i in range(num_layers)
        )
        self.classifier = nn.Conv1d(channels, num_classes, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.in_proj(x)
        for layer in self.layers:
            feat = layer(feat)
        return self.classifier(feat), feat


class GestureSegmenter(nn.Module):
    """Dense frame-label model.

    Returns:
        logits: [B, num_classes, T]
        boundary_logits: [B, 2, T] for start/end boundary probabilities
        stage_logits: list of refinement stage logits
    """

    def __init__(
        self,
        input_channels: int = 12,
        num_classes: int = NUM_CLASSES,
        hidden_dim: int = 128,
        temporal_channels: int = 128,
        temporal_layers: int = 6,
        temporal_stages: int = 2,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.input_channels = int(input_channels)
        self.num_classes = int(num_classes)
        self.encoder = STGCNEncoder(input_channels, hidden_dim, dropout)
        self.stage0 = TemporalStage(hidden_dim, temporal_channels, temporal_layers, num_classes, dropout)
        self.refiners = nn.ModuleList(
            TemporalStage(num_classes, temporal_channels, temporal_layers, num_classes, dropout)
            for _ in range(max(0, temporal_stages - 1))
        )
        self.boundary_head = nn.Sequential(
            nn.Conv1d(temporal_channels, temporal_channels // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(temporal_channels // 2, 2, 1),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        encoded = self.encoder(x)
        logits, feat = self.stage0(encoded)
        stages = [logits]
        for refiner in self.refiners:
            logits, _ = refiner(F.softmax(logits, dim=1))
            stages.append(logits)
        boundary = self.boundary_head(feat)
        return logits, boundary, stages

