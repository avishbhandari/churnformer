"""
Baseline models for comparison against ChurnFormer.

Three baselines per horizon:
  1. LogisticRegression  — static features only
  2. LightGBM            — static features only (gradient-boosted trees)
  3. LSTMChurner         — sequential model (same inputs as ChurnFormer, no attention)

All sklearn/lightgbm baselines are wrapped in a common interface so the
evaluation script can call them identically.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import joblib
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# sklearn / LightGBM wrappers
# ---------------------------------------------------------------------------

class StaticBaselineModel:
    """
    Wraps sklearn or LightGBM for multi-horizon churn prediction on static features.
    One model is trained per horizon.
    """

    SUPPORTED = {"logreg", "lgbm"}

    def __init__(self, model_type: str, horizons: list[int], **kwargs):
        assert model_type in self.SUPPORTED, f"model_type must be one of {self.SUPPORTED}"
        self.model_type = model_type
        self.horizons = horizons
        self.models: dict[int, object] = {}
        self.kwargs = kwargs

    def _make_model(self, horizon: int):
        if self.model_type == "logreg":
            return Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    max_iter=1000,
                    C=self.kwargs.get("C", 1.0),
                    class_weight="balanced",
                    random_state=42,
                )),
            ])
        elif self.model_type == "lgbm":
            return lgb.LGBMClassifier(
                n_estimators=self.kwargs.get("n_estimators", 500),
                learning_rate=self.kwargs.get("learning_rate", 0.05),
                num_leaves=self.kwargs.get("num_leaves", 63),
                min_child_samples=20,
                class_weight="balanced",
                random_state=42,
                verbose=-1,
            )

    def fit(self, X: np.ndarray, y_dict: dict[int, np.ndarray]) -> "StaticBaselineModel":
        """y_dict: {horizon: binary_label_array}"""
        for h in self.horizons:
            model = self._make_model(h)
            model.fit(X, y_dict[h])
            self.models[h] = model
        return self

    def predict_proba(self, X: np.ndarray) -> dict[int, np.ndarray]:
        return {h: self.models[h].predict_proba(X)[:, 1] for h in self.horizons}

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "StaticBaselineModel":
        return joblib.load(path)


# ---------------------------------------------------------------------------
# LSTM baseline
# ---------------------------------------------------------------------------

class LSTMChurner(nn.Module):
    """
    Bidirectional LSTM over the same event sequences as ChurnFormer.
    Uses mean pooling over non-padding positions for classification.
    """

    def __init__(
        self,
        event_vocab_size: int = 12,
        embed_dim: int = 64,
        hidden_size: int = 128,
        n_layers: int = 2,
        dropout: float = 0.1,
        n_horizons: int = 3,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        self.event_embed = nn.Embedding(event_vocab_size, embed_dim, padding_idx=0)
        # +3 scalar continuous features (time_delta, session_dur, feat_depth)
        lstm_in = embed_dim + 3

        self.lstm = nn.LSTM(
            input_size=lstm_in,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=True,
        )

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, 1),
            )
            for _ in range(n_horizons)
        ])

    def forward(
        self,
        event_ids: torch.Tensor,
        time_deltas: torch.Tensor,
        session_durs: torch.Tensor,
        feat_depths: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        e = self.event_embed(event_ids)  # (B, L, embed_dim)
        cont = torch.stack([time_deltas, session_durs, feat_depths], dim=-1)  # (B, L, 3)
        x = torch.cat([e, cont], dim=-1)  # (B, L, embed_dim+3)

        # Pack for efficiency
        lengths = (~padding_mask).sum(dim=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out_packed, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out_packed, batch_first=True)  # (B, L, 2*H)

        # Mean pool over valid positions
        valid = (~padding_mask).float().unsqueeze(-1)  # (B, L, 1)
        pooled = (out * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)  # (B, 2*H)

        logits = torch.cat([head(pooled) for head in self.heads], dim=-1)  # (B, n_horizons)
        probs = torch.sigmoid(logits)
        return logits, probs

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_lstm_baseline(config: dict) -> LSTMChurner:
    mc = config["model"]
    return LSTMChurner(
        event_vocab_size=mc["churnformer"]["event_vocab_size"],
        embed_dim=mc["lstm"]["hidden_size"] // 2,
        hidden_size=mc["lstm"]["hidden_size"],
        n_layers=mc["lstm"]["n_layers"],
        dropout=mc["lstm"]["dropout"],
        n_horizons=mc["churnformer"]["n_horizons"],
    )
