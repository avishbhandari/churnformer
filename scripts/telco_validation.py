"""
External validity check: static baselines on the IBM Telco Customer Churn dataset.
This validates the LogReg and LightGBM components on a public benchmark.
Note: Telco is B2C (not B2B) and lacks event-log sequences, so ChurnFormer's
sequential encoder cannot be directly evaluated; this script validates the
static-feature baselines only.

Run: python scripts/telco_validation.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, f1_score
from scipy import stats


def load_telco(path: str = "data/raw/telco_churn.csv") -> tuple:
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    cat_cols = [c for c in df.columns if df[c].dtype == object and c != "customerID"]
    for c in cat_cols:
        df[c] = LabelEncoder().fit_transform(df[c].astype(str))

    feature_cols = [c for c in df.columns if c not in ("customerID", "Churn")]
    X = df[feature_cols].values.astype(float)
    y = df["Churn"].values
    return X, y, feature_cols


def run_cv(X, y, n_splits=5, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    results = {"logreg": [], "lgbm": []}

    for fold, (tr, te) in enumerate(skf.split(X, y)):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]

        # LogReg
        lr = Pipeline([("sc", StandardScaler()),
                        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0))])
        lr.fit(X_tr, y_tr)
        p_lr = lr.predict_proba(X_te)[:, 1]
        results["logreg"].append(roc_auc_score(y_te, p_lr))

        # LightGBM
        lgbm = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=63,
                                    class_weight="balanced", random_state=seed, verbose=-1)
        lgbm.fit(X_tr, y_tr)
        p_lgbm = lgbm.predict_proba(X_te)[:, 1]
        results["lgbm"].append(roc_auc_score(y_te, p_lgbm))

    return results


if __name__ == "__main__":
    X, y, features = load_telco()
    print(f"Telco dataset: {X.shape[0]} customers, {y.mean():.1%} churn rate")
    results = run_cv(X, y)
    for model, aucs in results.items():
        print(f"{model}: AUC = {np.mean(aucs):.4f} ± {np.std(aucs):.4f}  (5-fold CV)")
        print(f"  Fold AUCs: {[round(a,4) for a in aucs]}")
