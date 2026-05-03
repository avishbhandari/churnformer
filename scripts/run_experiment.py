"""
Main experiment runner: ChurnFormer vs. LightGBM, LogReg, LSTM baselines.

Usage:
  python run_experiment.py                     # full run with synthetic data
  python run_experiment.py --config configs/config.yaml
  python run_experiment.py --skip-generate     # reuse existing data/synthetic/
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))

import yaml
from src.data.synthetic_generator import generate_synthetic_data
from src.data.preprocessor import (
    compute_static_features,
    build_sequences,
    SequenceNormalizer,
    ChurnDataset,
)
from src.models.churnformer import build_churnformer
from src.models.baselines import StaticBaselineModel, build_lstm_baseline
from src.models.trainer import ChurnTrainer, get_device
from src.evaluation.metrics import evaluate_all_horizons, print_results_table
from src.explainability.shap_explainer import (
    StaticSHAPExplainer,
    AttentionRolloutExplainer,
    generate_customer_report,
)


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def split_customer_ids(all_ids: list[int], cfg: dict) -> tuple[list, list, list]:
    seed = cfg["data"]["seed"]
    train_frac = cfg["data"]["train_split"]
    val_frac = cfg["data"]["val_split"]
    train_ids, tmp = train_test_split(all_ids, train_size=train_frac, random_state=seed)
    val_ids, test_ids = train_test_split(
        tmp, train_size=val_frac / (val_frac + cfg["data"]["test_split"]), random_state=seed
    )
    return list(train_ids), list(val_ids), list(test_ids)


def make_dataset(
    split_ids: list[int],
    sequences_norm: dict,
    labels_df: pd.DataFrame,
    static_df: pd.DataFrame,
    horizons: list[int],
    static_scaler=None,
    fit_scaler: bool = False,
) -> ChurnDataset:
    seqs_split = {cid: sequences_norm[cid] for cid in split_ids}
    return ChurnDataset(
        seqs_split, labels_df, static_df, horizons,
        static_scaler=static_scaler, fit_scaler=fit_scaler,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    cfg = load_config(args.config)
    horizons = cfg["data"]["horizons"]
    device = get_device(cfg["training"]["device"])
    print(f"\nDevice: {device}")

    # ── 1. Data ──────────────────────────────────────────────────────────
    if not args.skip_generate:
        print("\n[1/6] Generating synthetic CRM data...")
        dfs = generate_synthetic_data(args.config)
    else:
        print("\n[1/6] Loading existing synthetic data...")
        sd = cfg["data"]["synthetic_dir"]
        dfs = {
            "events": pd.read_parquet(f"{sd}/events.parquet"),
            "labels": pd.read_parquet(f"{sd}/labels.parquet"),
        }

    events_df = dfs["events"]
    labels_df = dfs["labels"]
    all_cids = labels_df["customer_id"].tolist()

    train_ids, val_ids, test_ids = split_customer_ids(all_cids, cfg)
    print(f"Split: {len(train_ids)} train / {len(val_ids)} val / {len(test_ids)} test")

    # ── 2. Preprocessing ─────────────────────────────────────────────────
    print("\n[2/6] Preprocessing sequences and static features...")
    sequences_raw = build_sequences(events_df, max_seq_len=cfg["data"]["max_seq_len"])
    static_df = compute_static_features(events_df, horizons)

    train_seqs = {cid: sequences_raw[cid] for cid in train_ids}
    normalizer = SequenceNormalizer().fit(train_seqs)
    sequences_norm = normalizer.transform(sequences_raw)

    train_ds = make_dataset(train_ids, sequences_norm, labels_df, static_df, horizons, fit_scaler=True)
    val_ds   = make_dataset(val_ids,   sequences_norm, labels_df, static_df, horizons, static_scaler=train_ds.static_scaler)
    test_ds  = make_dataset(test_ids,  sequences_norm, labels_df, static_df, horizons, static_scaler=train_ds.static_scaler)

    bs = cfg["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False, num_workers=0)

    # ── 3. Static baselines ──────────────────────────────────────────────
    print("\n[3/6] Training static baselines (LightGBM, LogReg)...")
    static_cols = train_ds.static_cols
    X_train = train_ds.static_scaler.transform(static_df.set_index("customer_id").loc[train_ids][static_cols].values)
    X_test  = train_ds.static_scaler.transform(static_df.set_index("customer_id").loc[test_ids][static_cols].values)
    y_train = {h: labels_df.set_index("customer_id").loc[train_ids][f"churn_{h}d"].values for h in horizons}
    y_test  = {h: labels_df.set_index("customer_id").loc[test_ids][f"churn_{h}d"].values for h in horizons}

    lgbm_model = StaticBaselineModel("lgbm", horizons)
    lgbm_model.fit(X_train, y_train)

    logreg_model = StaticBaselineModel("logreg", horizons)
    logreg_model.fit(X_train, y_train)

    lgbm_probs  = lgbm_model.predict_proba(X_test)
    logreg_probs = logreg_model.predict_proba(X_test)

    lgbm_results   = evaluate_all_horizons(y_test, lgbm_probs)
    logreg_results = evaluate_all_horizons(y_test, logreg_probs)

    # ── 4. LSTM baseline ─────────────────────────────────────────────────
    print("\n[4/6] Training LSTM baseline...")
    lstm_model = build_lstm_baseline(cfg)
    print(f"  LSTM parameters: {lstm_model.count_parameters():,}")
    lstm_optimizer = torch.optim.AdamW(
        lstm_model.parameters(), lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"]
    )
    lstm_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        lstm_optimizer, T_max=cfg["training"]["epochs"]
    )
    lstm_trainer = ChurnTrainer(
        lstm_model, lstm_optimizer, lstm_scheduler, device, cfg,
        model_name="lstm"
    )
    lstm_trainer.fit(train_loader, val_loader)
    lstm_results = lstm_trainer.evaluate(test_loader)

    # ── 5. ChurnFormer ───────────────────────────────────────────────────
    print("\n[5/6] Training ChurnFormer...")
    churnformer = build_churnformer(cfg)
    print(f"  ChurnFormer parameters: {churnformer.count_parameters():,}")
    cf_optimizer = torch.optim.AdamW(
        churnformer.parameters(), lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"]
    )
    cf_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        cf_optimizer, T_max=cfg["training"]["epochs"]
    )
    cf_trainer = ChurnTrainer(
        churnformer, cf_optimizer, cf_scheduler, device, cfg,
        model_name="churnformer"
    )
    cf_trainer.fit(train_loader, val_loader)
    cf_results = cf_trainer.evaluate(test_loader)

    # ── 6. Results & explainability ─────────────────────────────────────
    print("\n[6/6] Results summary\n")
    all_results = {
        "LogReg":       logreg_results,
        "LightGBM":     lgbm_results,
        "LSTM":         lstm_results,
        "ChurnFormer":  cf_results,
    }
    print_results_table(all_results, horizons)

    # Save JSON
    results_dir = Path("experiments/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "comparison_results.json", "w") as f:
        json.dump({
            name: {str(h): metrics for h, metrics in hr.items()}
            for name, hr in all_results.items()
        }, f, indent=2)
    print(f"\nResults saved → {results_dir}/comparison_results.json")

    # ── SHAP on LightGBM ─────────────────────────────────────────────────
    print("\nGenerating SHAP plots for LightGBM (30d horizon)...")
    lgbm_explainer = StaticSHAPExplainer(
        lgbm_model.models[30], "lgbm", feature_names=static_cols
    )
    lgbm_explainer.plot_summary(
        X_test[:200],
        save_path="experiments/results/figures/lgbm_shap_30d.png",
        horizon_label="30d",
    )

    # ── Attention rollout on ChurnFormer ──────────────────────────────────
    print("Generating attention rollout explanation for sample customer...")
    rollout_exp = AttentionRolloutExplainer(churnformer, device)
    sample_batch = next(iter(test_loader))
    attributions = rollout_exp.explain_batch(sample_batch)

    from src.data.synthetic_generator import EVENT_TYPES
    id_to_event = {v: k for k, v in EVENT_TYPES.items()}

    sample_cid = sample_batch["customer_id"][0]
    sample_event_ids = sample_batch["event_ids"][0].tolist()
    rollout_exp.plot_customer_attribution(
        attribution=attributions[0],
        event_ids=sample_event_ids,
        timestamps=[],
        event_id_to_name=id_to_event,
        save_path="experiments/results/figures/attention_rollout_sample.png",
        customer_id=int(sample_cid),
    )

    print("\nDone. All artifacts in experiments/results/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Run from project root: python scripts/run_experiment.py
  parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--skip-generate", action="store_true")
    args = parser.parse_args()
    main(args)
