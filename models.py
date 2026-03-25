#!/usr/bin/env python3
"""
Block 1 — Regression: predict levels_completed_per_session
          Features: session-start context only
          Models:   Linear Regression, Decision Tree, Random Forest, SVR, XGBoost

Block 2 — Classification: predict user retention (will_return)
          Features: aggregated first-session behaviour per user
          Models:   Logistic Regression, Decision Tree, Random Forest, SVM, XGBoost
"""
import sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, KFold, cross_validate
from sklearn.metrics import (
    make_scorer,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    RocCurveDisplay,
    mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
N_SPLITS = 5
OUTPUT_DIR = Path("reports")


# Data loading
def find_latest_tsv() -> Path:
    tsvs = sorted(OUTPUT_DIR.glob("metrica-sessions-*.tsv"))
    if not tsvs:
        raise FileNotFoundError(
            "No metrica-sessions-*.tsv in reports/. Run fetch_logs.py first."
        )
    return tsvs[-1]


def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str)
    print(f"Loaded {len(df)} attempt rows from {path.name}")
    return df


# Block 1: levels_completed_per_session (regression)
# Target: how many levels does a user complete in a single session?
# Features: only what is known at session START.
# Dropped intentionally:
#   - hints_used, words_found, time_to_first_word_sec  → mid-game signals
#   - completion_pct, drop_off_pct                     → mid/end-game signals
#   - visit_duration_sec, page_views                   → session outcome, not input

SESSION_NUM = [
    "hour_of_day",   # time of day the session started (0–23)
    "visit_count",   # how many times this user has visited before
]

SESSION_CAT = [
    "ab_group",        # A / B interface variant
    "is_new_user",     # 0 / 1 (Metrica server-side flag)
    "is_returning",    # 0 / 1
    "device_category", # desktop / mobile / tablet
    "utm_source",      # traffic source
    "utm_medium",      # traffic medium
]


def build_session_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """One row per session; target = number of completed levels."""
    completed_per_session = (
        df[df["level_status"] == "completed"]
        .groupby("session_id")
        .size()
        .rename("levels_completed")
    )

    keep = ["session_id"] + SESSION_NUM + SESSION_CAT
    keep = [c for c in keep if c in df.columns]
    sessions = (
        df[keep]
        .drop_duplicates("session_id")
        .set_index("session_id")
        .join(completed_per_session, how="left")
    )
    sessions["levels_completed"] = sessions["levels_completed"].fillna(0).astype(int)
    return sessions.reset_index()


def engineer_session(sessions: pd.DataFrame):
    d = sessions.copy()

    for col in SESSION_NUM:
        d[col] = pd.to_numeric(d.get(col, 0), errors="coerce").fillna(0)

    for col in ("is_new_user", "is_returning"):
        if col in d.columns:
            d[col] = (
                d[col].str.lower()
                .map({"true": 1, "1": 1, "false": 0, "0": 0})
                .fillna(0).astype(int)
            )

    for col in SESSION_CAT:
        if col in ("is_new_user", "is_returning"):
            continue
        if col in d.columns:
            d[col] = LabelEncoder().fit_transform(
                d[col].fillna("unknown").astype(str)
            )
        else:
            d[col] = 0

    feature_cols = [c for c in SESSION_NUM + SESSION_CAT if c in d.columns]
    return d[feature_cols].values, d["levels_completed"].values, feature_cols


def build_regressors() -> dict:
    models = {
        "Linear Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("reg",    LinearRegression()),
        ]),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=5, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=6, random_state=RANDOM_STATE
        ),
        "SVR": Pipeline([
            ("scaler", StandardScaler()),
            ("reg",    SVR(kernel="rbf")),
        ]),
    }
    if HAS_XGBOOST:
        models["XGBoost"] = XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            random_state=RANDOM_STATE, verbosity=0,
        )
    return models


def evaluate_regressors(models: dict, X, y) -> pd.DataFrame:
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scorers = {
        "neg_mae":  "neg_mean_absolute_error",
        "neg_rmse": "neg_root_mean_squared_error",
        "r2":       "r2",
    }
    records = []
    for name, model in models.items():
        print(f"  {name}…")
        s = cross_validate(model, X, y, cv=cv, scoring=scorers)
        records.append({
            "Model": name,
            "MAE":   -s["test_neg_mae"].mean(),
            "RMSE":  -s["test_neg_rmse"].mean(),
            "R²":     s["test_r2"].mean(),
            "MAE±":  s["test_neg_mae"].std(),
            "R²±":   s["test_r2"].std(),
        })
    return pd.DataFrame(records).set_index("Model")


def plot_regression_comparison(results: pd.DataFrame):
    metrics = ["MAE", "RMSE", "R²"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Block 1 — Levels Completed Prediction (5-Fold CV)", fontsize=13)
    colors = plt.cm.tab10.colors

    for ax, metric in zip(axes, metrics):
        vals = results[metric]
        bars = ax.bar(range(len(vals)), vals, color=colors[:len(vals)], alpha=0.85)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(results.index, rotation=25, ha="right", fontsize=9)
        ax.set_title(metric, fontsize=11)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8,
            )

    plt.tight_layout()
    out = OUTPUT_DIR / "regression_comparison.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"  Saved → {out}")


def plot_regression_feature_importance(models: dict, feature_cols: list, X, y):
    """Feature importances for tree-based regressors only."""
    importances = {}
    for name in ("Random Forest", "Decision Tree", "XGBoost"):
        if name not in models:
            continue
        m = models[name]
        m.fit(X, y)
        fi = getattr(m, "feature_importances_", None)
        if fi is not None:
            importances[name] = fi

    if not importances:
        return

    n = len(importances)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    fig.suptitle("Block 1 — Feature Importances", fontsize=13)

    for ax, (name, imp) in zip(axes, importances.items()):
        idx = np.argsort(imp)[::-1]
        ax.barh([feature_cols[i] for i in idx], imp[idx],
                color="steelblue", alpha=0.85)
        ax.set_xlabel("Importance")
        ax.set_title(name, fontsize=11)
        ax.invert_yaxis()

    plt.tight_layout()
    out = OUTPUT_DIR / "regression_feature_importance.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"  Saved → {out}")


def run_block1(df: pd.DataFrame):
    print("\n=== Block 1 — Levels Completed per Session (Regression) ===")
    sessions = build_session_dataset(df)
    print(
        f"  Sessions: {len(sessions)},  "
        f"mean levels completed: {sessions['levels_completed'].mean():.2f},  "
        f"max: {sessions['levels_completed'].max()}"
    )

    if len(sessions) < 20:
        print("Too few sessions — collect more data.")

    X, y, feature_cols = engineer_session(sessions)
    print(f"  Features ({len(feature_cols)}): {feature_cols}")

    models = build_regressors()
    results = evaluate_regressors(models, X, y)

    print("\n" + "=" * 55)
    print("BLOCK 1 RESULTS (mean over 5 folds)")
    print("=" * 55)
    print(results[["MAE", "RMSE", "R²"]].round(3).to_string())
    print("=" * 55)
    print(f"\nBest model by R²: {results['R²'].idxmax()}  ({results['R²'].max():.3f})")

    out = OUTPUT_DIR / "regression_results.csv"
    results.round(4).to_csv(out)
    print(f"Results saved → {out}")

    plot_regression_comparison(results)
    plot_regression_feature_importance(models, feature_cols, X, y)


# Block 2: user retention classification
# Target: will this user ever return? (appears on 2+ distinct dates = 1)
# Features: aggregated from the user's FIRST session only, so we're
#           predicting retention before knowing what the user will do next.

RET_NUM = [
    "completion_rate_first",   # share of levels completed in first session
    "levels_played_first",     # total level attempts in first session
    "avg_hints_first",         # average hints per level
    "avg_time_first_word",     # average seconds to first found word
    "session_duration_first",  # total duration of first session (seconds)
    "hour_of_day",             # hour they first played
]

RET_CAT = [
    "ab_group",        # A / B interface variant
    "device_category", # desktop / mobile / tablet
]


def build_retention_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """One row per user; target = retained (appeared on 2+ distinct dates)."""
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")

    # Retention label: same client_id seen on more than one calendar date
    n_dates = d.groupby("client_id")["date"].nunique()
    retained = (n_dates > 1).astype(int).rename("retained")

    # Keep only first-session rows to avoid using future data as features
    first_date = d.groupby("client_id")["date"].transform("min")
    first = d[d["date"] == first_date].copy()

    for col in ("hints_used", "time_to_first_word_sec", "visit_duration_sec"):
        if col in first.columns:
            first[col] = pd.to_numeric(first[col], errors="coerce").fillna(0)

    def agg_user(g):
        return pd.Series({
            "completion_rate_first": (g["level_status"] == "completed").mean(),
            "levels_played_first":   len(g),
            "avg_hints_first":       g["hints_used"].mean()
                                     if "hints_used" in g.columns else 0,
            "avg_time_first_word":   pd.to_numeric(
                                         g.get("time_to_first_word_sec", 0),
                                         errors="coerce"
                                     ).mean(),
            "session_duration_first": pd.to_numeric(
                                          g["visit_duration_sec"].iloc[0],
                                          errors="coerce"
                                      ) if "visit_duration_sec" in g.columns else 0,
            "hour_of_day":           pd.to_numeric(
                                         g["hour_of_day"].iloc[0],
                                         errors="coerce"
                                     ) if "hour_of_day" in g.columns else 0,
            "ab_group":              g["ab_group"].iloc[0]
                                     if "ab_group" in g.columns else "unknown",
            "device_category":       g["device_category"].iloc[0]
                                     if "device_category" in g.columns else "unknown",
        })

    users = (
        first.groupby("client_id")
        .apply(agg_user, include_groups=False)
        .reset_index()
        .merge(retained.reset_index(), on="client_id")
    )
    return users


def engineer_retention(users: pd.DataFrame):
    d = users.copy()

    for col in RET_NUM:
        d[col] = pd.to_numeric(d.get(col, 0), errors="coerce").fillna(0)

    for col in RET_CAT:
        if col in d.columns:
            d[col] = LabelEncoder().fit_transform(
                d[col].fillna("unknown").astype(str)
            )
        else:
            d[col] = 0

    feature_cols = [c for c in RET_NUM + RET_CAT if c in d.columns]
    return d[feature_cols].values, d["retained"].values, feature_cols


def build_classifiers() -> dict:
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=6, random_state=RANDOM_STATE
        ),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
        ]),
    }
    if HAS_XGBOOST:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            random_state=RANDOM_STATE, eval_metric="logloss", verbosity=0,
        )
    return models


CLF_SCORERS = {
    "accuracy":  make_scorer(accuracy_score),
    "precision": make_scorer(precision_score, zero_division=0),
    "recall":    make_scorer(recall_score, zero_division=0),
    "f1":        make_scorer(f1_score, zero_division=0),
    "roc_auc":   make_scorer(roc_auc_score, needs_proba=True),
}


def evaluate_classifiers(models: dict, X, y) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    records = []
    for name, model in models.items():
        print(f"  {name}…")
        s = cross_validate(model, X, y, cv=cv, scoring=CLF_SCORERS)
        records.append({
            "Model":     name,
            "Accuracy":  s["test_accuracy"].mean(),
            "Precision": s["test_precision"].mean(),
            "Recall":    s["test_recall"].mean(),
            "F1":        s["test_f1"].mean(),
            "ROC-AUC":   s["test_roc_auc"].mean(),
            "Acc±":      s["test_accuracy"].std(),
            "F1±":       s["test_f1"].std(),
            "AUC±":      s["test_roc_auc"].std(),
        })
    return pd.DataFrame(records).set_index("Model")


def plot_clf_comparison(results: pd.DataFrame, title: str, filename: str):
    metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    errors  = {"Accuracy": "Acc±", "F1": "F1±", "ROC-AUC": "AUC±"}
    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 5))
    fig.suptitle(f"{title} — 5-Fold Cross-Validation", fontsize=13)
    colors = plt.cm.tab10.colors

    for ax, metric in zip(axes, metrics):
        vals = results[metric]
        errs = results.get(
            errors.get(metric, ""), pd.Series(0, index=results.index)
        )
        bars = ax.bar(range(len(vals)), vals, yerr=errs, capsize=4,
                      color=colors[:len(vals)], alpha=0.85)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(results.index, rotation=25, ha="right", fontsize=9)
        ax.set_title(metric, fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--")
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8,
            )

    plt.tight_layout()
    out = OUTPUT_DIR / filename
    plt.savefig(out, dpi=150); plt.close()
    print(f"  Saved → {out}")


def plot_retention_roc(models: dict, X, y):
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10.colors

    for (name, model), color in zip(models.items(), colors):
        tprs, aucs, mean_fpr = [], [], np.linspace(0, 1, 100)
        for train_idx, test_idx in cv.split(X, y):
            model.fit(X[train_idx], y[train_idx])
            viz = RocCurveDisplay.from_estimator(
                model, X[test_idx], y[test_idx], ax=ax, alpha=0
            )
            interp = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp[0] = 0.0
            tprs.append(interp)
            aucs.append(viz.roc_auc)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        ax.plot(
            mean_fpr, mean_tpr, color=color, linewidth=2,
            label=f"{name} (AUC={np.mean(aucs):.2f}±{np.std(aucs):.2f})",
        )

    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Block 2 — Retention ROC Curves")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    out = OUTPUT_DIR / "retention_roc.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"  Saved → {out}")


def plot_retention_feature_importance(models: dict, feature_cols: list, X, y):
    importances = {}
    for name in ("Random Forest", "Decision Tree", "XGBoost"):
        if name not in models:
            continue
        m = models[name]
        m.fit(X, y)
        fi = getattr(m, "feature_importances_", None)
        if fi is not None:
            importances[name] = fi

    if not importances:
        return

    n = len(importances)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]
    fig.suptitle("Block 2 — Feature Importances (Retention)", fontsize=13)

    for ax, (name, imp) in zip(axes, importances.items()):
        idx = np.argsort(imp)[::-1]
        ax.barh([feature_cols[i] for i in idx], imp[idx],
                color="coral", alpha=0.85)
        ax.set_xlabel("Importance")
        ax.set_title(name, fontsize=11)
        ax.invert_yaxis()

    plt.tight_layout()
    out = OUTPUT_DIR / "retention_feature_importance.png"
    plt.savefig(out, dpi=150); plt.close()
    print(f"  Saved → {out}")


def run_block2(df: pd.DataFrame):
    print("\n=== Block 2 — User Retention Classification ===")
    users = build_retention_dataset(df)

    ret = int(users["retained"].sum())
    total = len(users)
    print(f"  Users: {total},  retained: {ret},  not retained: {total - ret}")

    if total < 20:
        print("Too few users — collect more sessions.")
        return
    if ret == 0 or ret == total:
        print("Only one class present — need both retained and non-retained users.")
        return
    if min(ret, total - ret) < 5:
        print("Very few samples in minority class — results will be unreliable.")

    X, y, feature_cols = engineer_retention(users)
    print(f"  Features ({len(feature_cols)}): {feature_cols}")

    models = build_classifiers()
    results = evaluate_classifiers(models, X, y)

    print("\n" + "=" * 65)
    print("BLOCK 2 RESULTS (mean over 5 folds)")
    print("=" * 65)
    print(results[["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]].round(3).to_string())
    print("=" * 65)
    print(f"\nBest model by F1: {results['F1'].idxmax()}  ({results['F1'].max():.3f})")

    out = OUTPUT_DIR / "retention_results.csv"
    results.round(4).to_csv(out)
    print(f"Results saved → {out}")

    plot_clf_comparison(results, "Block 2 — User Retention", "retention_comparison.png")
    plot_retention_roc(models, X, y)
    plot_retention_feature_importance(models, feature_cols, X, y)


# Main

def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else find_latest_tsv()
    df = load_raw(path)

    run_block1(df)
    run_block2(df)

    print("\nDone. Check reports/ for all output files.")


if __name__ == "__main__":
    main()
