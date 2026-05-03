"""
Preprocessing pipeline: raw event sequences → padded tensors + static features.

Each customer produces:
  - event_ids     : (seq_len,)  integer event-type token ids
  - time_deltas   : (seq_len,)  hours since previous event (log-normalised)
  - session_dur   : (seq_len,)  session duration in minutes (log-normalised)
  - feature_depth : (seq_len,)  sub-feature breadth (z-scored)
  - padding_mask  : (seq_len,)  bool, True = padding position
  - static_feats  : (n_static,) aggregate CRM features for baseline models
  - labels        : (n_horizons,) binary churn labels
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib
import yaml


EVENT_PAD_ID = 0
EVENT_CLS_ID = 1


def _load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Feature engineering helpers
# ---------------------------------------------------------------------------

def compute_static_features(events_df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """
    Aggregate per-customer static features for tree / linear baselines.
    These mirror what a conventional ML model would see.
    """
    rows = []
    for cid, grp in events_df.groupby("customer_id"):
        grp = grp.sort_values("timestamp")
        total_events = len(grp)
        span_days = max(1, (grp["timestamp"].max() - grp["timestamp"].min()).days)
        event_counts = grp["event_type"].value_counts()

        rows.append({
            "customer_id": cid,
            # Engagement volume
            "total_events": total_events,
            "events_per_day": total_events / span_days,
            "unique_event_types": grp["event_type"].nunique(),
            # Feature depth
            "avg_feature_depth": grp["feature_depth"].mean(),
            "max_feature_depth": grp["feature_depth"].max(),
            # Session quality
            "avg_session_min": grp["session_duration_min"].mean(),
            "med_session_min": grp["session_duration_min"].median(),
            # Support signal
            "support_tickets": event_counts.get("support_ticket", 0),
            "support_rate": event_counts.get("support_ticket", 0) / total_events,
            # Renewal / billing
            "contract_renewals": event_counts.get("contract_renewal", 0),
            "billing_events": event_counts.get("billing_event", 0),
            # Recency: days since last login
            "days_since_login": (
                (grp["timestamp"].max() - grp[grp["event_type"] == "login"]["timestamp"].max()).days
                if (grp["event_type"] == "login").any() else span_days
            ),
            # Trend: events in last 30 vs prior 30 days
            "recent_event_ratio": _recency_ratio(grp),
        })
    return pd.DataFrame(rows)


def _recency_ratio(grp: pd.DataFrame) -> float:
    last_ts = grp["timestamp"].max()
    last_30 = grp[grp["timestamp"] >= last_ts - pd.Timedelta(days=30)]
    prev_30 = grp[
        (grp["timestamp"] >= last_ts - pd.Timedelta(days=60)) &
        (grp["timestamp"] < last_ts - pd.Timedelta(days=30))
    ]
    return len(last_30) / max(1, len(prev_30))


# ---------------------------------------------------------------------------
# Sequence preparation
# ---------------------------------------------------------------------------

def build_sequences(
    events_df: pd.DataFrame,
    max_seq_len: int = 512,
) -> dict[int, dict]:
    """
    Convert event dataframe → per-customer padded sequence tensors.
    Prepends CLS token; appends PAD to max_seq_len.
    """
    sequences = {}
    for cid, grp in events_df.groupby("customer_id"):
        grp = grp.sort_values("timestamp").reset_index(drop=True)

        event_ids = grp["event_type_id"].tolist()
        timestamps = pd.to_datetime(grp["timestamp"])
        session_durs = grp["session_duration_min"].tolist()
        feat_depths = grp["feature_depth"].tolist()

        # Time deltas in hours (CLS token gets delta=0)
        deltas = [0.0]
        for i in range(1, len(timestamps)):
            delta_h = (timestamps.iloc[i] - timestamps.iloc[i - 1]).total_seconds() / 3600
            deltas.append(float(delta_h))

        # Prepend CLS, truncate to max_seq_len - 1, then pad
        event_ids = [EVENT_CLS_ID] + event_ids[-(max_seq_len - 1):]
        deltas = [0.0] + deltas[-(max_seq_len - 1):]
        session_durs = [0.0] + session_durs[-(max_seq_len - 1):]
        feat_depths = [0.0] + feat_depths[-(max_seq_len - 1):]

        seq_len = len(event_ids)
        pad_len = max_seq_len - seq_len

        padding_mask = [False] * seq_len + [True] * pad_len
        event_ids += [EVENT_PAD_ID] * pad_len
        deltas += [0.0] * pad_len
        session_durs += [0.0] * pad_len
        feat_depths += [0.0] * pad_len

        sequences[cid] = {
            "event_ids": event_ids,
            "time_deltas": deltas,
            "session_durs": session_durs,
            "feat_depths": feat_depths,
            "padding_mask": padding_mask,
            "seq_len": seq_len,
        }
    return sequences


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

class SequenceNormalizer:
    """Log-normalises continuous sequence features; fits only on training data."""

    def __init__(self):
        self._fitted = False
        self.delta_mean = 0.0
        self.delta_std = 1.0
        self.dur_mean = 0.0
        self.dur_std = 1.0
        self.depth_mean = 0.0
        self.depth_std = 1.0

    def fit(self, sequences: dict) -> "SequenceNormalizer":
        all_deltas, all_durs, all_depths = [], [], []
        for s in sequences.values():
            mask = [not m for m in s["padding_mask"]]
            all_deltas.extend([d for d, m in zip(s["time_deltas"], mask) if m])
            all_durs.extend([d for d, m in zip(s["session_durs"], mask) if m])
            all_depths.extend([d for d, m in zip(s["feat_depths"], mask) if m])

        log_deltas = np.log1p(all_deltas)
        log_durs = np.log1p(all_durs)

        self.delta_mean, self.delta_std = float(np.mean(log_deltas)), float(np.std(log_deltas) + 1e-8)
        self.dur_mean, self.dur_std = float(np.mean(log_durs)), float(np.std(log_durs) + 1e-8)
        self.depth_mean = float(np.mean(all_depths))
        self.depth_std = float(np.std(all_depths) + 1e-8)
        self._fitted = True
        return self

    def transform(self, sequences: dict) -> dict:
        assert self._fitted
        out = {}
        for cid, s in sequences.items():
            td = (np.log1p(s["time_deltas"]) - self.delta_mean) / self.delta_std
            sd = (np.log1p(s["session_durs"]) - self.dur_mean) / self.dur_std
            fd = (np.array(s["feat_depths"]) - self.depth_mean) / self.depth_std
            # Zero out padding positions
            mask = np.array(s["padding_mask"])
            td[mask] = 0.0
            sd[mask] = 0.0
            fd[mask] = 0.0
            out[cid] = {**s, "time_deltas": td.tolist(), "session_durs": sd.tolist(), "feat_depths": fd.tolist()}
        return out

    def save(self, path: str):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "SequenceNormalizer":
        return joblib.load(path)


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class ChurnDataset(Dataset):
    """
    Returns per-customer tensors for the transformer and baselines.

    Item keys:
      event_ids     LongTensor   (seq_len,)
      time_deltas   FloatTensor  (seq_len,)
      session_durs  FloatTensor  (seq_len,)
      feat_depths   FloatTensor  (seq_len,)
      padding_mask  BoolTensor   (seq_len,)   True = PAD
      static_feats  FloatTensor  (n_static,)
      labels        FloatTensor  (n_horizons,)
      customer_id   int
    """

    def __init__(
        self,
        sequences: dict,
        labels_df: pd.DataFrame,
        static_df: pd.DataFrame,
        horizons: list[int],
        static_scaler: StandardScaler | None = None,
        fit_scaler: bool = False,
    ):
        self.customer_ids = list(sequences.keys())
        self.sequences = sequences
        self.horizons = horizons

        label_cols = [f"churn_{h}d" for h in horizons]
        self.labels = labels_df.set_index("customer_id")[label_cols]

        static_cols = [c for c in static_df.columns if c != "customer_id"]
        self.static_cols = static_cols
        static_vals = static_df.set_index("customer_id")[static_cols]

        if fit_scaler:
            self.static_scaler = StandardScaler()
            self.static_scaler.fit(static_vals.loc[self.customer_ids].values)
        else:
            self.static_scaler = static_scaler

        self.static_vals = static_vals

    def __len__(self):
        return len(self.customer_ids)

    def __getitem__(self, idx: int) -> dict:
        cid = self.customer_ids[idx]
        seq = self.sequences[cid]

        static_raw = self.static_vals.loc[cid].values.astype(np.float32)
        if self.static_scaler is not None:
            static_raw = self.static_scaler.transform(static_raw.reshape(1, -1)).squeeze(0)

        labels = self.labels.loc[cid].values.astype(np.float32)

        return {
            "event_ids": torch.tensor(seq["event_ids"], dtype=torch.long),
            "time_deltas": torch.tensor(seq["time_deltas"], dtype=torch.float32),
            "session_durs": torch.tensor(seq["session_durs"], dtype=torch.float32),
            "feat_depths": torch.tensor(seq["feat_depths"], dtype=torch.float32),
            "padding_mask": torch.tensor(seq["padding_mask"], dtype=torch.bool),
            "static_feats": torch.tensor(static_raw, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.float32),
            "customer_id": cid,
        }
