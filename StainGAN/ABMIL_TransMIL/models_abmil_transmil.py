# /homes/wdkarim/code/models_abmil_transmil.py
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ABMIL(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256, attn_dim: int = 128, dropout: float = 0.25):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.attn_V = nn.Linear(hidden_dim, attn_dim)
        self.attn_U = nn.Linear(hidden_dim, attn_dim)
        self.attn_w = nn.Linear(attn_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N_patches, in_dim]
        h = self.fc(x)  # [N, hidden_dim]
        A_V = self.attn_V(h)          # [N, attn_dim]
        A_U = self.attn_U(h)          # [N, attn_dim]
        A = torch.tanh(A_V) * torch.sigmoid(A_U)  # gated attention
        A = self.attn_w(A)            # [N, 1]
        A = torch.softmax(A, dim=0)   # patch weights
        m = torch.sum(A * h, dim=0, keepdim=True)  # [1, hidden_dim]
        return m  # bag-level embedding


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        # src: [1, N_patches, d_model]
        attn_output, _ = self.self_attn(src, src, src)
        src = self.norm1(src + self.dropout1(attn_output))
        ff = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = self.norm2(src + self.dropout2(ff))
        return src


class TransMIL(nn.Module):
    def __init__(self, in_dim: int, d_model: int = 256, nhead: int = 8, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Linear(in_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(d_model, nhead, dim_feedforward=4 * d_model, dropout=dropout)
             for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N_patches, in_dim]
        x = self.embed(x)  # [N, d_model]
        x = x.unsqueeze(0)  # [1, N, d_model]
        cls = self.cls_token.expand(x.size(0), -1, -1)  # [1, 1, d_model]
        x = torch.cat([cls, x], dim=1)  # [1, N+1, d_model]
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        cls_out = x[:, 0, :]  # [1, d_model]
        return cls_out  # bag-level embedding
