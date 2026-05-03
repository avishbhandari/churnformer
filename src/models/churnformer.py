"""
ChurnFormer: Transformer encoder for B2B SaaS customer churn prediction.

Architecture:
  - Event embedding  : learnable token embedding for event types
  - Continuous feats : time delta, session duration, feature depth → projected to d_model/4 each
  - Temporal PE      : continuous-time sinusoidal positional encoding on time_delta
  - Transformer encoder (n_layers × multi-head self-attention + FFN)
  - CLS pooling
  - Multi-task output heads: one sigmoid per prediction horizon (30/60/90d)

Attention weights from the last encoder layer are stored for SHAP attribution.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# Positional / temporal encoding
# ---------------------------------------------------------------------------

class ContinuousTemporalEncoding(nn.Module):
    """
    Encode elapsed time (hours since previous event) as a sinusoidal vector.
    Unlike fixed PE, this operates on actual time gaps, preserving irregular
    sampling information that is critical for detecting disengagement.
    """

    def __init__(self, d_model: int, max_period: float = 8760.0):  # 1 year in hours
        super().__init__()
        self.d_model = d_model
        self.max_period = max_period
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(max_period) / d_model)
        )
        self.register_buffer("div", div)

    def forward(self, time_deltas: torch.Tensor) -> torch.Tensor:
        # time_deltas: (B, L)
        t = time_deltas.unsqueeze(-1)  # (B, L, 1)
        sin_enc = torch.sin(t * self.div)          # (B, L, d/2)
        cos_enc = torch.cos(t * self.div)          # (B, L, d/2)
        enc = torch.zeros(*time_deltas.shape, self.d_model, device=time_deltas.device)
        enc[..., 0::2] = sin_enc
        enc[..., 1::2] = cos_enc
        return enc


# ---------------------------------------------------------------------------
# Input projection
# ---------------------------------------------------------------------------

class EventInputProjection(nn.Module):
    """
    Combines:
      - Event type embedding
      - Time delta encoding (continuous sinusoidal)
      - Session duration (scalar → linear)
      - Feature depth (scalar → linear)
    All projected to d_model via a final linear + layer norm.
    """

    def __init__(self, vocab_size: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.event_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.temporal_enc = ContinuousTemporalEncoding(d_model)

        # Scalar continuous features → d_model/4 each
        cont_dim = d_model // 4
        self.dur_proj = nn.Linear(1, cont_dim)
        self.depth_proj = nn.Linear(1, cont_dim)

        # Fuse: event_embed + temporal + dur + depth → d_model
        fuse_in = d_model + d_model + cont_dim + cont_dim
        self.fuse = nn.Sequential(
            nn.Linear(fuse_in, d_model),
            nn.LayerNorm(d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        event_ids: torch.Tensor,      # (B, L)
        time_deltas: torch.Tensor,    # (B, L)
        session_durs: torch.Tensor,   # (B, L)
        feat_depths: torch.Tensor,    # (B, L)
    ) -> torch.Tensor:
        e = self.event_embed(event_ids)                          # (B, L, d)
        te = self.temporal_enc(time_deltas)                      # (B, L, d)
        sd = self.dur_proj(session_durs.unsqueeze(-1))           # (B, L, d/4)
        fd = self.depth_proj(feat_depths.unsqueeze(-1))          # (B, L, d/4)

        x = torch.cat([e, te, sd, fd], dim=-1)
        x = self.fuse(x)
        return self.dropout(x)


# ---------------------------------------------------------------------------
# Transformer encoder with attention capture
# ---------------------------------------------------------------------------

class AttentionCapturingEncoder(nn.Module):
    """Standard TransformerEncoderLayer that optionally returns attention weights."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        attn_out, attn_weights = self.self_attn(
            x, x, x,
            key_padding_mask=key_padding_mask,
            need_weights=return_attn,
            average_attn_weights=True,
        )
        x = self.norm1(x + self.drop1(attn_out))
        x = self.norm2(x + self.drop2(self.ff(x)))
        return x, attn_weights


# ---------------------------------------------------------------------------
# ChurnFormer
# ---------------------------------------------------------------------------

class ChurnFormer(nn.Module):
    """
    Transformer-based churn prediction model.

    forward() returns:
      logits  : (B, n_horizons)  — raw pre-sigmoid scores
      probs   : (B, n_horizons)  — sigmoid probabilities
      attn    : (B, L) or None   — CLS attention weights from last layer
    """

    def __init__(
        self,
        event_vocab_size: int = 12,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        n_horizons: int = 3,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_horizons = n_horizons

        self.input_proj = EventInputProjection(event_vocab_size, d_model, dropout)

        self.layers = nn.ModuleList([
            AttentionCapturingEncoder(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Per-horizon classification heads (binary)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, 1),
            )
            for _ in range(n_horizons)
        ])

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        event_ids: torch.Tensor,
        time_deltas: torch.Tensor,
        session_durs: torch.Tensor,
        feat_depths: torch.Tensor,
        padding_mask: torch.Tensor,
        return_attn: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:

        x = self.input_proj(event_ids, time_deltas, session_durs, feat_depths)  # (B, L, d)

        last_attn = None
        for i, layer in enumerate(self.layers):
            is_last = (i == len(self.layers) - 1)
            x, attn = layer(x, key_padding_mask=padding_mask, return_attn=(return_attn and is_last))
            if is_last:
                last_attn = attn  # (B, L)

        cls_repr = x[:, 0, :]  # CLS token representation

        logits = torch.cat([head(cls_repr) for head in self.heads], dim=-1)  # (B, n_horizons)
        probs = torch.sigmoid(logits)

        # CLS attention: weights over sequence from CLS token perspective
        cls_attn = None
        if return_attn and last_attn is not None:
            # last_attn averaged over heads: (B, L, L); row 0 = CLS queries
            # For explainability we want how much CLS attends to each token
            # MultiheadAttention with average_attn_weights returns (B, L, L)
            cls_attn = last_attn[:, 0, :]  # (B, L)

        return logits, probs, cls_attn

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_churnformer(config: dict) -> ChurnFormer:
    mc = config["model"]["churnformer"]
    return ChurnFormer(
        event_vocab_size=mc["event_vocab_size"],
        d_model=mc["d_model"],
        n_heads=mc["n_heads"],
        n_layers=mc["n_layers"],
        d_ff=mc["d_ff"],
        dropout=mc["dropout"],
        max_seq_len=config["data"]["max_seq_len"],
        n_horizons=mc["n_horizons"],
    )
