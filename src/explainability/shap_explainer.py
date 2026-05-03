"""
SHAP-based explainability for ChurnFormer and baseline models.

Two complementary explanation strategies:
  1. SHAP KernelExplainer on static features (model-agnostic, works for all models)
  2. Attention Rollout from ChurnFormer's encoder (sequence-level attribution)

Both produce human-readable explanations for customer success teams.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import shap
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for grayscale-safe rendering
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# SHAP for static feature models (LightGBM / LogReg)
# ---------------------------------------------------------------------------

class StaticSHAPExplainer:
    """TreeExplainer for LightGBM; LinearExplainer for LogReg."""

    def __init__(self, model, model_type: str, feature_names: list[str]):
        self.model_type = model_type
        self.feature_names = feature_names

        if model_type == "lgbm":
            self.explainer = shap.TreeExplainer(model)
        elif model_type == "logreg":
            # Access the clf step inside the Pipeline
            clf = model.named_steps["clf"]
            self.explainer = shap.LinearExplainer(clf, masker=shap.maskers.Independent(
                np.zeros((1, len(feature_names)))
            ))

    def explain(self, X: np.ndarray) -> np.ndarray:
        return self.explainer.shap_values(X)

    def plot_summary(self, X: np.ndarray, save_path: str, horizon_label: str = "30d"):
        shap_vals = self.explain(X)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]  # positive class

        fig, ax = plt.subplots(figsize=(8, 5))
        mean_abs = np.abs(shap_vals).mean(axis=0)
        idx = np.argsort(mean_abs)[::-1][:15]
        bars = ax.barh(
            [self.feature_names[i] for i in reversed(idx)],
            mean_abs[list(reversed(idx))],
            color=["#222222" if j % 2 == 0 else "#888888" for j in range(len(idx))],
        )
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(f"Feature Importance (SHAP) — Churn@{horizon_label}")
        ax.invert_yaxis()
        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Saved SHAP summary plot → {save_path}")


# ---------------------------------------------------------------------------
# Attention rollout for ChurnFormer
# ---------------------------------------------------------------------------

class AttentionRolloutExplainer:
    """
    Computes attention rollout (Abnar & Zuidema, 2020) over all encoder layers
    to attribute the CLS output to input event positions.

    Returns per-token importance scores that can be mapped back to event types
    and timestamps for human-readable explanations.
    """

    def __init__(self, model, device: torch.device, discard_ratio: float = 0.9):
        self.model = model
        self.device = device
        self.discard_ratio = discard_ratio
        self._attn_cache: list[torch.Tensor] = []
        self._hooks: list = []

    def _register_hooks(self):
        self._attn_cache = []

        def make_hook(layer_idx):
            def hook(module, inp, out):
                # MultiheadAttention forward returns (attn_output, attn_weights)
                # We need to capture attn_weights; do a fresh forward with need_weights=True
                pass
            return hook

        # We use the model's built-in return_attn mechanism instead of hooks
        # to avoid double-forward overhead.

    @torch.no_grad()
    def get_attention_maps(self, batch: dict) -> list[torch.Tensor]:
        """Run forward pass per layer capturing attention weights."""
        self.model.eval()
        batch = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in batch.items()}

        x = self.model.input_proj(
            batch["event_ids"],
            batch["time_deltas"],
            batch["session_durs"],
            batch["feat_depths"],
        )

        attn_maps = []
        padding_mask = batch["padding_mask"]
        for layer in self.model.layers:
            attn_out, attn_weights = layer.self_attn(
                x, x, x,
                key_padding_mask=padding_mask,
                need_weights=True,
                average_attn_weights=True,
            )
            residual = x + layer.drop1(attn_out)
            x = layer.norm1(residual)
            x = layer.norm2(x + layer.drop2(layer.ff(x)))
            if attn_weights is not None:
                attn_maps.append(attn_weights.cpu())

        return attn_maps  # list of (B, L, L) per layer

    def rollout(self, attn_maps: list[torch.Tensor], padding_mask: torch.Tensor) -> torch.Tensor:
        """
        Attention rollout: multiply attention matrices across layers with
        residual identity and discard low-attention tokens.
        Returns (B, L) attribution scores for each input position.
        """
        B, L, _ = attn_maps[0].shape
        rollout = torch.eye(L).unsqueeze(0).expand(B, -1, -1)  # (B, L, L)

        for attn in attn_maps:
            # Add identity for residual connection
            attn_with_residual = 0.5 * attn + 0.5 * torch.eye(L).unsqueeze(0)
            # Discard lowest attention (optional noise reduction)
            flat = attn_with_residual.view(B, -1)
            threshold = torch.quantile(flat, self.discard_ratio, dim=-1, keepdim=True).unsqueeze(-1)
            attn_with_residual = torch.where(attn_with_residual >= threshold, attn_with_residual,
                                             torch.zeros_like(attn_with_residual))
            # Row-normalise
            row_sum = attn_with_residual.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            attn_with_residual = attn_with_residual / row_sum
            rollout = torch.bmm(attn_with_residual, rollout)

        # CLS row = how much CLS attends to each token
        cls_attribution = rollout[:, 0, :]  # (B, L)

        # Zero out padding
        valid = ~padding_mask.cpu()
        cls_attribution = cls_attribution * valid.float()

        return cls_attribution  # (B, L)

    def explain_batch(self, batch: dict) -> np.ndarray:
        attn_maps = self.get_attention_maps(batch)
        cls_attr = self.rollout(attn_maps, batch["padding_mask"])
        return cls_attr.numpy()  # (B, L)

    def plot_customer_attribution(
        self,
        attribution: np.ndarray,
        event_ids: list[int],
        timestamps: list,
        event_id_to_name: dict,
        save_path: str,
        customer_id: int = 0,
        top_k: int = 15,
    ):
        """Bar chart of top-k attributed events for a single customer."""
        # Skip CLS (position 0) and padding
        valid_attr = attribution[1:]
        valid_events = event_ids[1:]
        valid_ts = timestamps[1:] if timestamps else list(range(len(valid_attr)))

        top_idx = np.argsort(valid_attr)[::-1][:top_k]
        scores = valid_attr[top_idx]
        labels = [
            f"{event_id_to_name.get(valid_events[i], '?')} (t={i})"
            for i in top_idx
        ]

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["#111111" if s >= np.median(scores) else "#999999" for s in scores]
        ax.barh(labels[::-1], scores[::-1], color=colors[::-1])
        ax.set_xlabel("Attention Rollout Attribution Score")
        ax.set_title(f"Customer {customer_id}: Top-{top_k} Influential Events")
        plt.tight_layout()
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Saved attention attribution plot → {save_path}")


# ---------------------------------------------------------------------------
# Unified explanation report
# ---------------------------------------------------------------------------

def generate_customer_report(
    customer_id: int,
    churn_probs: dict[int, float],
    top_events: list[dict],
    top_static_features: list[dict],
    save_path: str,
):
    """Write a plain-text churn explanation report for a single customer."""
    lines = [
        f"=== ChurnFormer Explanation Report: Customer {customer_id} ===",
        "",
        "Churn probability scores:",
    ]
    for horizon, prob in churn_probs.items():
        risk = "HIGH" if prob > 0.6 else ("MEDIUM" if prob > 0.35 else "LOW")
        lines.append(f"  {horizon}-day horizon: {prob:.1%}  [{risk} RISK]")

    lines += ["", "Top contributing behavioral signals (attention rollout):"]
    for i, ev in enumerate(top_events[:10], 1):
        lines.append(f"  {i:2d}. {ev['event_type']} — score {ev['score']:.4f}")

    lines += ["", "Top static feature contributions (SHAP):"]
    for feat in top_static_features[:10]:
        direction = "↑ increases churn risk" if feat["shap"] > 0 else "↓ decreases churn risk"
        lines.append(f"  {feat['name']}: value={feat['value']:.3f}, SHAP={feat['shap']:+.4f} ({direction})")

    report = "\n".join(lines)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        f.write(report)
    print(f"Saved customer report → {save_path}")
    return report
