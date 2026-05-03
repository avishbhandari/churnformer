"""Evaluation metrics for churn prediction: AUC, F1, Brier, Precision, Recall."""

import numpy as np
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score, brier_score_loss
)


def evaluate_horizon(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan"),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "brier": brier_score_loss(y_true, y_prob),
        "n_pos": int(y_true.sum()),
        "n_total": len(y_true),
    }


def evaluate_all_horizons(
    y_true_dict: dict[int, np.ndarray],
    y_prob_dict: dict[int, np.ndarray],
    threshold: float = 0.5,
) -> dict[int, dict]:
    return {h: evaluate_horizon(y_true_dict[h], y_prob_dict[h], threshold) for h in y_true_dict}


def print_results_table(results: dict[str, dict[int, dict]], horizons: list[int]):
    """Pretty-print comparison table."""
    header = f"{'Model':<20}" + "".join(f"  AUC@{h}d   F1@{h}d" for h in horizons)
    print(header)
    print("-" * len(header))
    for model_name, horizon_results in results.items():
        row = f"{model_name:<20}"
        for h in horizons:
            m = horizon_results.get(h, {})
            row += f"  {m.get('auc', float('nan')):.4f}   {m.get('f1', float('nan')):.4f}"
        print(row)
