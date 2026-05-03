"""
Synthetic CRM behavioral event sequence generator.

Generates realistic B2B SaaS customer interaction histories that mimic
anonymized CRM export data: event type, timestamp, session duration,
feature usage, support tickets, billing events, and contract renewals.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import json
from datetime import datetime, timedelta
import random


EVENT_TYPES = {
    "PAD": 0,
    "CLS": 1,
    "login": 2,
    "feature_use": 3,
    "export": 4,
    "api_call": 5,
    "support_ticket": 6,
    "billing_event": 7,
    "contract_renewal": 8,
    "settings_change": 9,
    "onboarding": 10,
    "report_view": 11,
}


def _load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


class SyntheticCRMGenerator:
    """
    Simulates 3-year customer behavioral histories for B2B SaaS.

    Churned customers show declining engagement patterns 60-90 days before
    churn, increased support tickets, and missed renewal interactions.
    Healthy customers show stable or growing feature adoption.
    """

    def __init__(self, config: dict, seed: int = 42):
        self.cfg = config["data"]["synthetic"]
        self.horizons = config["data"]["horizons"]
        self.max_seq_len = config["data"]["max_seq_len"]
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

        self.n_customers = self.cfg["n_customers"]
        self.churn_rate = self.cfg["churn_rate"]
        self.obs_days = self.cfg["observation_window_days"]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, save_dir: str = "data/synthetic") -> dict[str, pd.DataFrame]:
        """Generate full dataset and return split DataFrames."""
        print(f"Generating {self.n_customers} customer histories ({self.obs_days} days each)...")
        records = []
        for cid in range(self.n_customers):
            is_churner = np.random.rand() < self.churn_rate
            records.extend(self._generate_customer(cid, is_churner))

        events_df = pd.DataFrame(records)
        labels_df = self._build_labels(events_df)

        Path(save_dir).mkdir(parents=True, exist_ok=True)
        events_df.to_parquet(f"{save_dir}/events.parquet", index=False)
        labels_df.to_parquet(f"{save_dir}/labels.parquet", index=False)
        print(f"Saved to {save_dir}/  ({len(events_df):,} events, {len(labels_df):,} customers)")
        return {"events": events_df, "labels": labels_df}

    # ------------------------------------------------------------------
    # Internal generation logic
    # ------------------------------------------------------------------

    def _generate_customer(self, cid: int, is_churner: bool) -> list[dict]:
        start_date = datetime(2021, 1, 1) + timedelta(days=np.random.randint(0, 90))

        if is_churner:
            churn_day = np.random.randint(180, self.obs_days - 30)
        else:
            churn_day = None

        events = []
        # Onboarding burst in first 30 days
        for _ in range(np.random.randint(5, 15)):
            day = np.random.randint(0, 30)
            events.append(self._make_event(cid, start_date, day, "onboarding", is_churner, churn_day))

        # Regular engagement throughout observation window
        day = 0
        while day < self.obs_days:
            daily_events = self._sample_daily_events(day, is_churner, churn_day)
            for etype in daily_events:
                events.append(self._make_event(cid, start_date, day, etype, is_churner, churn_day))
            day += np.random.randint(1, 4)  # customers don't log in every single day

        # Billing / renewal events
        for month in range(0, self.obs_days // 30):
            billing_day = month * 30 + np.random.randint(-3, 4)
            billing_day = max(0, min(billing_day, self.obs_days - 1))
            events.append(self._make_event(cid, start_date, billing_day, "billing_event", is_churner, churn_day))

        # Annual renewal
        for year in [365, 730]:
            if year < self.obs_days:
                renewal_day = year + np.random.randint(-7, 8)
                etype = "contract_renewal" if (not is_churner or renewal_day < churn_day - 90) else "support_ticket"
                events.append(self._make_event(cid, start_date, renewal_day, etype, is_churner, churn_day))

        events.sort(key=lambda e: e["timestamp"])
        # Keep up to max_seq_len most recent events
        if len(events) > self.max_seq_len:
            events = events[-self.max_seq_len:]

        for i, e in enumerate(events):
            e["seq_pos"] = i

        return events

    def _sample_daily_events(self, day: int, is_churner: bool, churn_day) -> list[str]:
        """Sample event types for a given day, with churn-driven degradation."""
        if is_churner and churn_day is not None:
            days_to_churn = churn_day - day
            # Accelerating disengagement 90 days before churn
            if days_to_churn < 0:
                return []  # customer has churned
            decay = max(0.0, 1.0 - (90 - days_to_churn) / 90) if days_to_churn < 90 else 1.0
            support_surge = max(1.0, 3.0 * (1.0 - decay))
        else:
            decay = 1.0
            support_surge = 1.0

        base_events = []

        if np.random.rand() < 0.6 * decay:
            base_events.append("login")
        if np.random.rand() < 0.5 * decay:
            base_events.append("feature_use")
        if np.random.rand() < 0.2 * decay:
            base_events.append("report_view")
        if np.random.rand() < 0.15 * decay:
            base_events.append("export")
        if np.random.rand() < 0.1 * decay:
            base_events.append("api_call")
        if np.random.rand() < 0.05 * decay:
            base_events.append("settings_change")
        if np.random.rand() < 0.08 * support_surge:
            base_events.append("support_ticket")

        return base_events

    def _make_event(
        self, cid: int, start: datetime, day: int, etype: str, is_churner: bool, churn_day
    ) -> dict:
        ts = start + timedelta(days=day, hours=np.random.randint(8, 20), minutes=np.random.randint(0, 60))
        session_minutes = np.random.lognormal(mean=2.5, sigma=0.8)
        if is_churner and churn_day and (churn_day - day) < 60:
            session_minutes *= max(0.1, (churn_day - day) / 60)

        return {
            "customer_id": cid,
            "is_churner": int(is_churner),
            "churn_day": churn_day,
            "event_type": etype,
            "event_type_id": EVENT_TYPES[etype],
            "timestamp": ts,
            "day_offset": day,
            "session_duration_min": round(float(session_minutes), 2),
            "feature_depth": np.random.randint(1, 10),   # how many sub-features used
            "seq_pos": 0,  # filled in after sort
        }

    def _build_labels(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """Build per-customer churn labels for each prediction horizon."""
        rows = []
        for cid, grp in events_df.groupby("customer_id"):
            is_churner = grp["is_churner"].iloc[0]
            churn_day = grp["churn_day"].iloc[0]
            row = {"customer_id": cid, "is_churner": is_churner, "churn_day": churn_day}
            for h in self.horizons:
                if is_churner and churn_day is not None and churn_day <= h:
                    row[f"churn_{h}d"] = 1
                else:
                    row[f"churn_{h}d"] = 0
            rows.append(row)
        return pd.DataFrame(rows)


def generate_synthetic_data(config_path: str = "configs/config.yaml") -> dict[str, pd.DataFrame]:
    cfg = _load_config(config_path)
    gen = SyntheticCRMGenerator(cfg, seed=cfg["data"]["seed"])
    return gen.generate(save_dir=cfg["data"]["synthetic_dir"])


if __name__ == "__main__":
    dfs = generate_synthetic_data()
    print("\nEvents sample:")
    print(dfs["events"].head())
    print("\nLabels distribution:")
    print(dfs["labels"]["churn_30d"].value_counts())
