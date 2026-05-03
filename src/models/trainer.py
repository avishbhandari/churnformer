"""
Training loop for ChurnFormer and LSTMChurner.

Supports:
  - Multi-task binary cross-entropy (one loss per horizon)
  - Positive class upweighting for class imbalance
  - Early stopping on validation AUC (primary horizon = 30d by default)
  - Gradient clipping
  - Checkpoint saving / loading
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm

from src.evaluation.metrics import evaluate_horizon


class ChurnTrainer:

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        device: torch.device,
        config: dict,
        checkpoint_dir: str = "experiments/results/checkpoints",
        model_name: str = "churnformer",
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.cfg = config["training"]
        self.horizons = config["data"]["horizons"]
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name

        pos_weight = torch.tensor([self.cfg["pos_weight"]], device=device)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self.best_val_auc = -1.0
        self.patience_counter = 0

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for batch in loader:
            batch = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in batch.items()}
            logits = self._forward(batch)          # (B, n_horizons)
            labels = batch["labels"]               # (B, n_horizons)

            loss = sum(
                self.criterion(logits[:, i], labels[:, i])
                for i in range(len(self.horizons))
            ) / len(self.horizons)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg["grad_clip"])
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(loader)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict[int, dict]:
        self.model.eval()
        all_probs = {h: [] for h in self.horizons}
        all_labels = {h: [] for h in self.horizons}

        for batch in loader:
            batch = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in batch.items()}
            logits = self._forward(batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            labels = batch["labels"].cpu().numpy()

            for i, h in enumerate(self.horizons):
                all_probs[h].append(probs[:, i])
                all_labels[h].append(labels[:, i])

        results = {}
        for h in self.horizons:
            y_prob = np.concatenate(all_probs[h])
            y_true = np.concatenate(all_labels[h])
            results[h] = evaluate_horizon(y_true, y_prob)
        return results

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> list[dict]:
        history = []
        for epoch in range(1, self.cfg["epochs"] + 1):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)

            primary_auc = val_metrics[self.horizons[0]]["auc"]
            self.scheduler.step()

            row = {"epoch": epoch, "train_loss": train_loss}
            for h in self.horizons:
                row[f"val_auc_{h}d"] = val_metrics[h]["auc"]
                row[f"val_f1_{h}d"] = val_metrics[h]["f1"]
            history.append(row)

            if epoch % 5 == 0 or epoch == 1:
                auc_str = "  ".join(f"AUC@{h}d={val_metrics[h]['auc']:.4f}" for h in self.horizons)
                print(f"  Epoch {epoch:3d} | loss={train_loss:.4f} | {auc_str}")

            if primary_auc > self.best_val_auc:
                self.best_val_auc = primary_auc
                self.patience_counter = 0
                self._save_checkpoint(epoch)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.cfg["patience"]:
                    print(f"  Early stopping at epoch {epoch}")
                    break

        self._load_best_checkpoint()
        return history

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _forward(self, batch: dict) -> torch.Tensor:
        """Unified forward supporting both ChurnFormer and LSTMChurner."""
        from src.models.churnformer import ChurnFormer
        from src.models.baselines import LSTMChurner

        if isinstance(self.model, ChurnFormer):
            logits, _, _ = self.model(
                batch["event_ids"],
                batch["time_deltas"],
                batch["session_durs"],
                batch["feat_depths"],
                batch["padding_mask"],
            )
        elif isinstance(self.model, LSTMChurner):
            logits, _ = self.model(
                batch["event_ids"],
                batch["time_deltas"],
                batch["session_durs"],
                batch["feat_depths"],
                batch["padding_mask"],
            )
        else:
            raise ValueError(f"Unknown model type: {type(self.model)}")
        return logits

    def _save_checkpoint(self, epoch: int):
        path = self.checkpoint_dir / f"{self.model_name}_best.pt"
        torch.save({
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_val_auc": self.best_val_auc,
        }, path)

    def _load_best_checkpoint(self):
        path = self.checkpoint_dir / f"{self.model_name}_best.pt"
        if path.exists():
            ckpt = torch.load(path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state"])


def get_device(preference: str = "auto") -> torch.device:
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preference)
