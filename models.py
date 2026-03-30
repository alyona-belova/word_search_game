#!/usr/bin/env python3
"""
Block 1 — Regression: predict levels_completed_per_session
Features: session-start context only
Models:   Linear Regression, Decision Tree, Random Forest, XGBoost/LightGBM

Block 2 — Classification: predict user retention (will_return)
Features: aggregated first-session behaviour per user + engineered features
Models:   Logistic Regression, Decision Tree, Random Forest, XGBoost/LightGBM

usage: python3 models.py [FILE] [–from YYYY-MM-DD] [–tune] [–cost-fp FLOAT –cost-fn FLOAT]
"""
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    make_scorer,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    RocCurveDisplay, brier_score_loss,
)
from sklearn.model_selection import (
    StratifiedKFold, KFold, cross_validate,
    RandomizedSearchCV, TimeSeriesSplit, learning_curve,
)
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, ElasticNet
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import warnings
import argparse
import logging
import joblib
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")


# gradient boosting: prefer XGBoost → LightGBM → CatBoost

_BOOSTER_NAME = None
try:
    from xgboost import XGBClassifier, XGBRegressor
    _BOOSTER_NAME = "xgboost"
except ImportError:
    try:
        from lightgbm import LGBMClassifier as XGBClassifier, LGBMRegressor as XGBRegressor
        _BOOSTER_NAME = "lightgbm"
    except ImportError:
        try:
            from catboost import CatBoostClassifier, CatBoostRegressor

            class XGBClassifier(CatBoostClassifier):
                def __init__(self, **kw):
                    kw.setdefault("verbose", 0)
                    super().__init__(**kw)

            class XGBRegressor(CatBoostRegressor):
                def __init__(self, **kw):
                    kw.setdefault("verbose", 0)
                    super().__init__(**kw)

            _BOOSTER_NAME = "catboost"
        except ImportError:
            _BOOSTER_NAME = None

HAS_BOOSTER = _BOOSTER_NAME is not None

# optional SMOTE

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    ImbPipeline = Pipeline  # fall back to sklearn Pipeline silently

# optional SHAP

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

warnings.filterwarnings("ignore")

# logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# constants

RANDOM_STATE = 42
N_SPLITS = 5
OUTPUT_DIR = Path("reports")
MODEL_DIR = OUTPUT_DIR / "saved_models"
RARE_CAT_THRESHOLD = 5
LOG_COLS = {"visit_count", "session_duration_first", "avg_time_first_word"}

# Default business cost ratio (can be overridden via CLI)
# cost of predicting return when user won't (wasted spend)
DEFAULT_COST_FP = 1.0
DEFAULT_COST_FN = 3.0   # cost of missing a returner (lost revenue)


# Preprocessing helpers

def cap_rare(series: pd.Series) -> pd.Series:
    rare = series.value_counts()[lambda c: c < RARE_CAT_THRESHOLD].index
    return series.where(~series.isin(rare), other="other")


def freq_encode(series: pd.Series) -> pd.Series:
    return series.map(series.value_counts()).fillna(0).astype(int)


def winsorize(series: pd.Series, upper: float = 0.95) -> pd.Series:
    cap = series.quantile(upper)
    return series.clip(upper=cap) if cap > 0 else series


def hour_to_period(hour: float) -> str:
    """Bin 0-23 hour into named day-parts."""
    if hour < 6:
        return "night"
    if hour < 12:
        return "morning"
    if hour < 18:
        return "afternoon"
    return "evening"

# Data loading


def find_latest_tsvs() -> list:
    ru_tsvs = sorted(OUTPUT_DIR.glob("metrica-sessions-[0-9]*.tsv"))
    eng_tsvs = sorted(OUTPUT_DIR.glob("metrica-sessions-eng-*.tsv"))
    paths = []
    if ru_tsvs:
        paths.append(ru_tsvs[-1])
    if eng_tsvs:
        paths.append(eng_tsvs[-1])
    if not paths:
        raise FileNotFoundError(
            "No metrica-sessions-*.tsv in reports/. Run fetch_logs.py first."
        )
    return paths


def load_raw(paths) -> pd.DataFrame:
    if isinstance(paths, Path):
        paths = [paths]
    frames = []
    for p in paths:
        chunk = pd.read_csv(p, sep="\t", dtype=str)
        log.info("Loaded %d rows from %s", len(chunk), p.name)
        frames.append(chunk)
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    if len(frames) > 1:
        log.info("Combined total: %d rows", len(df))
    return df

# Block 1 — Regression


SESSION_NUM = [
    "hour_of_day",
    "visit_count",
]
SESSION_CAT = [
    "ab_group",
    "is_new_user",
    "is_returning",
    "device_category",
    "utm_source",
    "utm_medium",
    "region",
    "browser",
    "os",
    "theme_letter",
    "session_period",   # morning / afternoon / evening / night
    "day_of_week",      # Monday … Sunday
    "is_weekend",       # 0 / 1
]


def build_session_dataset(df: pd.DataFrame) -> pd.DataFrame:
    completed_per_session = (
        df[df["level_status"] == "completed"]
        .groupby("session_id")
        .size()
        .rename("levels_completed")
    )
    keep = ["session_id"] + SESSION_NUM + [
        c for c in SESSION_CAT
        if c not in ("session_period", "day_of_week", "is_weekend")
    ]
    keep = [c for c in keep if c in df.columns]
    sessions = (
        df[keep]
        .drop_duplicates("session_id")
        .set_index("session_id")
        .join(completed_per_session, how="left")
    )
    sessions["levels_completed"] = sessions["levels_completed"].fillna(
        0).astype(int)

    if "hour_of_day" in sessions.columns:
        hour = pd.to_numeric(
            sessions["hour_of_day"], errors="coerce").fillna(12)
        sessions["session_period"] = hour.apply(hour_to_period)
    if "date" in df.columns:
        date_map = (
            df.drop_duplicates("session_id")
            .set_index("session_id")["date"]
        )
        sessions["_date"] = pd.to_datetime(date_map, errors="coerce")
        sessions["day_of_week"] = sessions["_date"].dt.day_name().fillna(
            "Unknown")
        sessions["is_weekend"] = sessions["_date"].dt.dayofweek.isin([
                                                                     5, 6]).astype(int)
        sessions.drop(columns=["_date"], inplace=True)

    return sessions.reset_index()


def engineer_session(sessions: pd.DataFrame):
    d = sessions.copy()
    available_num = [c for c in SESSION_NUM if c in d.columns]
    available_cat = [
        c for c in SESSION_CAT if c in d.columns or c == "is_weekend"]

    for col in available_num:
        d[col] = pd.to_numeric(d.get(col, 0), errors="coerce").fillna(0)
        d[col] = winsorize(d[col])
        if col in LOG_COLS:
            d[col] = np.log1p(d[col])

    for col in ("is_new_user", "is_returning"):
        if col in d.columns:
            d[col] = (
                d[col].str.lower()
                .map({"true": 1, "1": 1, "false": 0, "0": 0})
                .fillna(0).astype(int)
            )

    for col in available_cat:
        if col in ("is_new_user", "is_returning", "is_weekend"):
            continue
        if col in d.columns:
            d[col] = freq_encode(
                cap_rare(d[col].fillna("unknown").astype(str)))
        else:
            d[col] = 0

    feature_cols = [c for c in available_num + available_cat if c in d.columns]
    return d[feature_cols].values, d["levels_completed"].values, feature_cols


def _reg_param_grids() -> dict:
    """
    Full search spaces for every regressor.
    n_iter is set per-model to cover the grid adequately without being wasteful:
    · small grids  (< 50 combos)  → n_iter = 20
    · medium grids (50–200)       → n_iter = 40
    · large grids  (200+)         → n_iter = 60
    """
    grids: dict[str, dict] = {
        # Ridge: one continuous hyperparameter — sweep log-scale
        "Ridge": {
            "reg__alpha": np.logspace(-3, 3, 50).tolist(),   # 50 values
        },
        # ElasticNet: alpha controls shrinkage, l1_ratio mixes L1/L2
        "ElasticNet": {
            "reg__alpha":    np.logspace(-3, 2, 30).tolist(),
            "reg__l1_ratio": [0.05, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9, 1.0],
        },
        "Decision Tree": {
            "max_depth":         [3, 4, 6, 8, 10, 12, None],
            "min_samples_leaf":  [1, 3, 5, 10, 20],
            "min_samples_split": [2, 5, 10, 20],
            "max_features":      ["sqrt", "log2", None],
        },
        "Random Forest": {
            "n_estimators":      [100, 200, 300, 500],
            "max_depth":         [5, 8, 10, 15, None],
            "max_features":      ["sqrt", "log2", 0.3, 0.5],
            "min_samples_leaf":  [1, 3, 5, 10],
            "min_samples_split": [2, 5, 10],
        },
    }
    if HAS_BOOSTER:
        grids["XGBoost"] = {
            "n_estimators":      [100, 200, 300, 500],
            "max_depth":         [3, 4, 5, 6, 8],
            "learning_rate":     [0.005, 0.01, 0.05, 0.1, 0.15, 0.2],
            "subsample":         [0.6, 0.7, 0.8, 0.9, 1.0],
            "colsample_bytree":  [0.6, 0.7, 0.8, 0.9, 1.0],
            "reg_alpha":         [0, 0.01, 0.1, 1.0],
            "reg_lambda":        [0.5, 1.0, 2.0, 5.0],
        }
    return grids


def _n_iter_for_grid(grid: dict) -> int:
    """Scale RandomizedSearch iterations with grid size."""
    from math import prod
    size = prod(len(v) for v in grid.values())
    if size < 50:
        return 20
    if size < 200:
        return 40
    return 60


def _run_search(name: str, estimator, grid: dict, X, y,
                scoring: str, cv, summary_rows: list) -> object:
    """Run RandomizedSearchCV, log results, append a row to summary_rows."""
    from math import prod
    grid_size = prod(len(v) for v in grid.values())
    n_iter = _n_iter_for_grid(grid)
    log.info("    Tuning %-22s  grid_size=%-6d  n_iter=%d",
             name, grid_size, n_iter)

    search = RandomizedSearchCV(
        estimator, grid,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        refit=True,
        return_train_score=True,
    )
    search.fit(X, y)

    sign = -1 if scoring.startswith("neg_") else 1
    score = sign * search.best_score_
    log.info("    %-22s  best_score=%.4f  params=%s",
             name, score, search.best_params_)

    summary_rows.append({
        "model":       name,
        "best_score":  round(score, 5),
        "n_iter":      n_iter,
        "grid_size":   grid_size,
        **{f"param_{k}": v for k, v in search.best_params_.items()},
    })
    return search.best_estimator_, search.cv_results_


def _save_tuning_summary(summary_rows: list, cv_results_map: dict,
                         block: str, scoring_label: str):
    """Write CSV summary and score-distribution plot for all tuned models."""
    if not summary_rows:
        return

    out_csv = OUTPUT_DIR / f"tuning_summary_{block}.csv"
    pd.DataFrame(summary_rows).to_csv(out_csv, index=False)
    log.info("  Tuning summary → %s", out_csv)

    # score distribution plot (violin per model)
    models_with_results = list(cv_results_map.keys())
    if not models_with_results:
        return

    fig, ax = plt.subplots(figsize=(max(8, 2 * len(models_with_results)), 5))
    data, labels = [], []
    for mname, cvr in cv_results_map.items():
        scores = cvr.get("mean_test_score", np.array([]))
        if len(scores):
            data.append(scores)
            labels.append(mname)

    if data:
        parts = ax.violinplot(data, showmedians=True, showextrema=True)
        for pc in parts["bodies"]:
            pc.set_alpha(0.7)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel(scoring_label)
        ax.set_title(f"{block} — Tuning Score Distribution (all trials)")
        ax.grid(axis="y", linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    out_png = OUTPUT_DIR / f"tune_trials_{block}.png"
    plt.savefig(out_png, dpi=150)
    plt.close()
    log.info("  Tuning plot    → %s", out_png)


def build_regressors(tune: bool = False, X=None, y=None) -> dict:
    """
    Build all regressors.  When tune=True every model is searched over its
    full parameter grid; best estimators are returned.  Results are written
    to reports/tuning_summary_block1.csv and reports/tune_trials_block1.png.
    """
    cv_inner = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    grids = _reg_param_grids()
    summary: list = []
    cv_results: dict = {}

    def maybe_tune(name, estimator):
        grid = grids.get(name, {})
        if not tune or not grid or X is None:
            return estimator
        best, cvr = _run_search(name, estimator, grid, X, y,
                                "neg_root_mean_squared_error", cv_inner, summary)
        cv_results[name] = cvr
        return best

    ridge = maybe_tune("Ridge", Pipeline([
        ("scaler", StandardScaler()),
        ("reg",    Ridge()),
    ]))
    enet = maybe_tune("ElasticNet", Pipeline([
        ("scaler", StandardScaler()),
        ("reg",    ElasticNet(max_iter=5000)),
    ]))
    dt = maybe_tune("Decision Tree",
                    DecisionTreeRegressor(random_state=RANDOM_STATE))
    rf = maybe_tune("Random Forest",
                    RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1))

    models: dict = {
        "Ridge":           ridge,
        "ElasticNet":      enet,
        "Decision Tree":   dt,
        "Random Forest":   rf,
    }

    if HAS_BOOSTER:
        xgb_base = XGBRegressor(
            random_state=RANDOM_STATE,
            **({} if _BOOSTER_NAME == "catboost" else {"verbosity": 0}),
        )
        models["XGBoost"] = maybe_tune("XGBoost", xgb_base)

    if tune:
        _save_tuning_summary(summary, cv_results, "block1",
                             "neg_RMSE (higher=better)")

    return models


def evaluate_regressors(models: dict, X, y) -> pd.DataFrame:
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scorers = {
        "neg_mae":  "neg_mean_absolute_error",
        "neg_rmse": "neg_root_mean_squared_error",
        "r2":       "r2",
    }
    records = []
    fold_scores: dict[str, np.ndarray] = {}
    for name, model in models.items():
        log.info("  Evaluating %s …", name)
        s = cross_validate(model, X, y, cv=cv, scoring=scorers, n_jobs=-1)
        fold_scores[name] = s["test_r2"]
        records.append({
            "Model": name,
            "MAE": -s["test_neg_mae"].mean(),
            "RMSE": -s["test_neg_rmse"].mean(),
            "R²":     s["test_r2"].mean(),
            "MAE±":  s["test_neg_mae"].std(),
            "R²±":   s["test_r2"].std(),
        })
    df = pd.DataFrame(records).set_index("Model")
    _significance_table(fold_scores, metric="R²",
                        output_name="regression_significance.csv")
    return df


def _significance_table(fold_scores: dict, metric: str, output_name: str):
    """Pairwise Wilcoxon signed-rank tests; saves CSV to reports/."""
    names = list(fold_scores.keys())
    rows = []
    for a, b in combinations(names, 2):
        try:
            stat, p = wilcoxon(fold_scores[a], fold_scores[b])
        except Exception:
            stat, p = float("nan"), float("nan")
        rows.append({"ModelA": a, "ModelB": b, "W-stat": stat, "p-value": p,
                     "significant (p<0.05)": p < 0.05})
    if rows:
        out = OUTPUT_DIR / output_name
        pd.DataFrame(rows).to_csv(out, index=False)
        log.info("  Significance tests → %s", out)


def plot_regression_comparison(results: pd.DataFrame):
    metrics = ["MAE", "RMSE", "R²"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(
        "Block 1 — Levels Completed Prediction (5-Fold CV)", fontsize=13)
    colors = plt.cm.tab10.colors

    for ax, metric in zip(axes, metrics):
        vals = results[metric]
        bars = ax.bar(range(len(vals)), vals,
                      color=colors[:len(vals)], alpha=0.85)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(results.index, rotation=25, ha="right", fontsize=9)
        ax.set_title(metric, fontsize=11)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.02,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out = OUTPUT_DIR / "regression_comparison.png"
    plt.savefig(out, dpi=150)
    plt.close()
    log.info("  Saved → %s", out)


def plot_regression_feature_importance(models: dict, feature_cols: list, X, y):
    if HAS_SHAP:
        # Only pass tree-based models to SHAP TreeExplainer;
        # linear models get a coefficient bar chart separately.
        tree_models = {k: v for k, v in models.items()
                       if k not in ("Ridge", "ElasticNet")}
        if tree_models:
            _plot_shap_importance(tree_models, feature_cols, X, y,
                                  title="Block 1 — SHAP Feature Importance",
                                  out_name="regression_shap_importance.png")
        _plot_linear_coefs(models, feature_cols, X, y, block="block1")
        return

    importances = {}
    for name in ("Random Forest", "Decision Tree", "XGBoost"):
        if name not in models:
            continue
        m = models[name]
        m.fit(X, y)
        fi = getattr(m, "feature_importances_", None)
        if fi is not None:
            importances[name] = fi

    _plot_linear_coefs(models, feature_cols, X, y, block="block1")

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
    plt.savefig(out, dpi=150)
    plt.close()
    log.info("  Saved → %s", out)


def _plot_linear_coefs(models: dict, feature_cols: list, X, y, block: str):
    """Bar chart of |coefficients| for Ridge / ElasticNet / Logistic Regression."""
    linear_names = [n for n in ("Ridge", "ElasticNet", "Logistic Regression")
                    if n in models]
    if not linear_names:
        return

    fig, axes = plt.subplots(1, len(linear_names),
                             figsize=(6 * len(linear_names), 5))
    if len(linear_names) == 1:
        axes = [axes]
    fig.suptitle(f"{block} — Linear Model Coefficients", fontsize=13)

    for ax, name in zip(axes, linear_names):
        m = models[name]
        m.fit(X, y)
        # coefficients live inside a Pipeline under step "reg" or "clf"
        inner = m
        for step_name in ("reg", "clf", "model"):
            if hasattr(inner, "named_steps") and step_name in inner.named_steps:
                inner = inner.named_steps[step_name]
                break
        coef = getattr(inner, "coef_", None)
        if coef is None:
            ax.set_title(f"{name}\n(no coef_)")
            continue
        coef = np.abs(coef.ravel())
        idx = np.argsort(coef)[::-1]
        ax.barh([feature_cols[i] for i in idx], coef[idx],
                color="mediumseagreen", alpha=0.85)
        ax.set_xlabel("|coefficient|")
        ax.set_title(name, fontsize=11)
        ax.invert_yaxis()

    plt.tight_layout()
    out = OUTPUT_DIR / f"{block}_linear_coefs.png"
    plt.savefig(out, dpi=150)
    plt.close()
    log.info("  Saved → %s", out)


def _plot_shap_importance(models, feature_cols, X, y, title, out_name):
    target_names = [n for n in ("Random Forest", "XGBoost") if n in models]
    if not target_names:
        target_names = [list(models.keys())[0]]

    n = len(target_names)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5))
    if n == 1:
        axes = [axes]
    fig.suptitle(title, fontsize=13)

    for ax, name in zip(axes, target_names):
        m = models[name]
        m.fit(X, y)
        try:
            explainer = shap.TreeExplainer(m)
            shap_values = explainer.shap_values(X)
            if isinstance(shap_values, list):      # multiclass
                shap_values = np.abs(np.array(shap_values)).mean(axis=0)
            mean_abs = np.abs(shap_values).mean(axis=0)
            idx = np.argsort(mean_abs)[::-1]
            ax.barh([feature_cols[i] for i in idx], mean_abs[idx],
                    color="steelblue", alpha=0.85)
            ax.set_xlabel("|SHAP|")
            ax.set_title(name, fontsize=11)
            ax.invert_yaxis()
        except Exception as e:
            ax.set_title(f"{name}\n(SHAP failed: {e})", fontsize=9)

    plt.tight_layout()
    out = OUTPUT_DIR / out_name
    plt.savefig(out, dpi=150)
    plt.close()
    log.info("  Saved → %s", out)


def plot_learning_curves(models: dict, X, y, task: str = "regression"):
    """Plot train/val learning curves for each model."""
    scoring = "r2" if task == "regression" else "roc_auc"
    cv = (KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
          if task == "regression"
          else StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE))

    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]
    fig.suptitle(f"Learning Curves ({task})", fontsize=13)

    for ax, (name, model) in zip(axes, models.items()):
        try:
            sizes, train_s, val_s = learning_curve(
                model, X, y, cv=cv, scoring=scoring,
                train_sizes=np.linspace(0.2, 1.0, 5),
                n_jobs=-1,
            )
            ax.plot(sizes, train_s.mean(1), "o-",  label="train")
            ax.plot(sizes, val_s.mean(1),   "s--", label="val")
            ax.fill_between(sizes,
                            train_s.mean(1) - train_s.std(1),
                            train_s.mean(1) + train_s.std(1), alpha=0.1)
            ax.fill_between(sizes,
                            val_s.mean(1) - val_s.std(1),
                            val_s.mean(1) + val_s.std(1), alpha=0.1)
        except Exception as e:
            ax.set_title(f"{name}\n(failed: {e})", fontsize=8)
            continue
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Training samples")
        ax.set_ylabel(scoring)
        ax.legend(fontsize=8)

    plt.tight_layout()
    out = OUTPUT_DIR / f"learning_curves_{task}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    log.info("  Saved → %s", out)


def run_block1(df: pd.DataFrame, tune: bool = False):
    log.info("\n=== Block 1 — Levels Completed per Session (Regression) ===")
    sessions = build_session_dataset(df)
    log.info(
        "Sessions: %d,  mean levels completed: %.2f,  max: %d",
        len(sessions), sessions["levels_completed"].mean(
        ), sessions["levels_completed"].max()
    )

    if len(sessions) < 20:
        log.warning("Too few sessions — collect more data.")
        return

    X, y, feature_cols = engineer_session(sessions)
    log.info("Features (%d): %s", len(feature_cols), feature_cols)

    models = build_regressors(tune=tune, X=X, y=y)
    results = evaluate_regressors(models, X, y)

    log.info("\n" + "=" * 55)
    log.info("BLOCK 1 RESULTS (mean over 5 folds)")
    log.info("=" * 55)
    print(results[["MAE", "RMSE", "R²"]].round(3).to_string())
    log.info("=" * 55)
    best = results["R²"].idxmax()
    log.info("Best model by R²: %s  (%.3f)", best, results["R²"].max())

    out = OUTPUT_DIR / "regression_results.csv"
    results.round(4).to_csv(out)
    log.info("Results saved → %s", out)

    # persist best model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_model = models[best]
    best_model.fit(X, y)
    joblib.dump(best_model, MODEL_DIR / "block1_best_model.pkl")
    log.info("Best model saved → %s", MODEL_DIR / "block1_best_model.pkl")

    plot_regression_comparison(results)
    plot_regression_feature_importance(models, feature_cols, X, y)
    plot_learning_curves(models, X, y, task="regression")

# Block 2 — Retention Classification


RET_NUM = [
    "completion_rate_first",
    "levels_played_first",
    "avg_hints_first",
    "avg_time_first_word",
    "session_duration_first",
    "hour_of_day",
    # completion_rate × session_duration (interaction term)
    "engagement_score",
    "hints_per_level",     # avg_hints / levels_played (normalised)
]

RET_CAT = [
    "ab_group",
    "device_category",
    "region",
    "browser",
    "os",
    "theme_letter_first",
    "session_period",      # morning / afternoon / evening / night
    "day_of_week",         # Monday … Sunday
    "is_weekend",          # 0 / 1
]


def build_retention_dataset(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")

    n_dates = d.groupby("client_id")["date"].nunique()
    retained = (n_dates > 1).astype(int).rename("retained")

    first_date = d.groupby("client_id")["date"].transform("min")
    first = d[d["date"] == first_date].copy()

    for col in ("hints_used", "time_to_first_word_sec", "visit_duration_sec"):
        if col in first.columns:
            first[col] = pd.to_numeric(first[col], errors="coerce").fillna(0)

    def agg_user(g):
        hour = pd.to_numeric(g["hour_of_day"].iloc[0], errors="coerce") \
            if "hour_of_day" in g.columns else 12.0
        comp_rate = (g["level_status"] == "completed").mean()
        n_levels = len(g)
        avg_hints = g["hints_used"].mean() if "hints_used" in g.columns else 0
        sess_dur = pd.to_numeric(
            g["visit_duration_sec"].iloc[0], errors="coerce"
        ) if "visit_duration_sec" in g.columns else 0.0

        # date-derived features
        ts = g["date"].iloc[0]
        dow = ts.day_name() if pd.notna(ts) else "Unknown"
        weekend = int(ts.dayofweek in (5, 6)) if pd.notna(ts) else 0

        return pd.Series({
            "completion_rate_first":  comp_rate,
            "levels_played_first":    n_levels,
            "avg_hints_first":        avg_hints,
            "avg_time_first_word":    pd.to_numeric(
                g.get("time_to_first_word_sec", 0),
                errors="coerce"
            ).mean(),
            "session_duration_first": sess_dur,
            "hour_of_day":            hour,
            # interaction features
            "engagement_score":       comp_rate * max(sess_dur, 0),
            "hints_per_level":        avg_hints / max(n_levels, 1),
            # date features
            "session_period":         hour_to_period(hour),
            "day_of_week":            dow,
            "is_weekend":             weekend,
            # categorical
            "ab_group":               g["ab_group"].iloc[0]
            if "ab_group" in g.columns else "unknown",
            "device_category":        g["device_category"].iloc[0]
            if "device_category" in g.columns else "unknown",
            "region":                 g["region"].iloc[0]
            if "region" in g.columns else "unknown",
            "browser":                g["browser"].iloc[0]
            if "browser" in g.columns else "unknown",
            "os":                     g["os"].iloc[0]
            if "os" in g.columns else "unknown",
            "theme_letter_first":     (lambda m: m.iloc[0] if len(m) else "unknown")(
                g["theme_letter"].dropna().mode()
            ) if "theme_letter" in g.columns else "unknown",
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
    available_num = [c for c in RET_NUM if c in d.columns]
    available_cat = [c for c in RET_CAT if c in d.columns]

    for col in available_num:
        d[col] = pd.to_numeric(d.get(col, 0), errors="coerce").fillna(0)
        d[col] = winsorize(d[col])
        if col in LOG_COLS:
            d[col] = np.log1p(d[col])

    for col in available_cat:
        if col == "is_weekend":
            continue
        d[col] = freq_encode(cap_rare(d[col].fillna("unknown").astype(str)))

    feature_cols = [c for c in available_num + available_cat if c in d.columns]
    return d[feature_cols].values, d["retained"].values, feature_cols


def _clf_param_grids(spw: float = 1.0) -> dict:
    """
    Full search spaces for every classifier.
    Pipeline-based estimators use double-underscore param paths.
    """
    grids: dict[str, dict] = {
        # Logistic Regression: regularisation strength × penalty type × solver
        "Logistic Regression": {
            "clf__C":       np.logspace(-3, 3, 40).tolist(),   # 40 values
            "clf__penalty": ["l1", "l2"],
            "clf__solver":  ["liblinear", "saga"],
        },
        "Decision Tree": {
            "max_depth":         [3, 4, 6, 8, 10, 12, None],
            "min_samples_leaf":  [1, 3, 5, 10, 20],
            "min_samples_split": [2, 5, 10, 20],
            "max_features":      ["sqrt", "log2", None],
        },
        "Random Forest": {
            "n_estimators":      [100, 200, 300, 500],
            "max_depth":         [5, 8, 10, 15, None],
            "max_features":      ["sqrt", "log2", 0.3, 0.5],
            "min_samples_leaf":  [1, 3, 5, 10],
            "min_samples_split": [2, 5, 10],
        },
    }
    if HAS_BOOSTER:
        grids["XGBoost"] = {
            "n_estimators":      [100, 200, 300, 500],
            "max_depth":         [3, 4, 5, 6, 8],
            "learning_rate":     [0.005, 0.01, 0.05, 0.1, 0.15, 0.2],
            "subsample":         [0.6, 0.7, 0.8, 0.9, 1.0],
            "colsample_bytree":  [0.6, 0.7, 0.8, 0.9, 1.0],
            "reg_alpha":         [0, 0.01, 0.1, 1.0],
            "reg_lambda":        [0.5, 1.0, 2.0, 5.0],
            "scale_pos_weight":  [spw * f for f in (0.5, 0.75, 1.0, 1.25, 1.5)],
        }
    return grids


def build_classifiers(
    scale_pos_weight: float = 1.0,
    tune: bool = False,
    X=None,
    y=None,
) -> dict:
    """
    Build all classifiers.  When tune=True every model is searched over its
    full grid; best estimators returned.  Results written to
    reports/tuning_summary_block2.csv and reports/tune_trials_block2.png.
    """
    cv_inner = StratifiedKFold(
        n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    grids = _clf_param_grids(spw=scale_pos_weight)
    summary: list = []
    cv_results: dict = {}

    def maybe_tune(name, estimator):
        grid = grids.get(name, {})
        if not tune or not grid or X is None:
            return estimator
        best, cvr = _run_search(name, estimator, grid, X, y,
                                "roc_auc", cv_inner, summary)
        cv_results[name] = cvr
        return best

    # Logistic Regression
    lr_base = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(
            max_iter=2000, random_state=RANDOM_STATE,
            class_weight="balanced",
        )),
    ])
    lr = maybe_tune("Logistic Regression", lr_base)

    # Decision Tree
    dt = maybe_tune("Decision Tree", DecisionTreeClassifier(
        random_state=RANDOM_STATE, class_weight="balanced",
    ))

    # Random Forest
    rf = maybe_tune("Random Forest", RandomForestClassifier(
        random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1,
    ))

    # SMOTE wrapping
    def smote_wrap(estimator):
        if HAS_SMOTE:
            return ImbPipeline([
                ("smote", SMOTE(random_state=RANDOM_STATE)),
                ("model", estimator),
            ])
        return estimator

    models: dict = {
        "Logistic Regression": smote_wrap(lr),
        "Decision Tree":       smote_wrap(dt),
        "Random Forest":       smote_wrap(rf),
    }

    # XGBoost / LightGBM / CatBoost
    if HAS_BOOSTER:
        xgb_kw: dict = {"random_state": RANDOM_STATE}
        if _BOOSTER_NAME != "catboost":
            xgb_kw["verbosity"] = 0
            xgb_kw["scale_pos_weight"] = scale_pos_weight
        else:
            xgb_kw["class_weights"] = {0: 1.0, 1: scale_pos_weight}
        xgb_base = XGBClassifier(**xgb_kw)
        xgb = maybe_tune("XGBoost", xgb_base)
        models["XGBoost"] = smote_wrap(xgb)

    if tune:
        _save_tuning_summary(summary, cv_results, "block2", "ROC-AUC")

    return models


CLF_SCORERS = {
    "accuracy":  make_scorer(accuracy_score),
    "precision": make_scorer(precision_score, zero_division=0),
    "recall":    make_scorer(recall_score, zero_division=0),
    "f1":        make_scorer(f1_score, zero_division=0),
    "roc_auc":   make_scorer(roc_auc_score, needs_proba=True),
    "brier":     make_scorer(brier_score_loss, needs_proba=True,
                             greater_is_better=False),
}


def evaluate_classifiers(models: dict, X, y) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                         random_state=RANDOM_STATE)
    records = []
    fold_scores: dict[str, np.ndarray] = {}
    for name, model in models.items():
        log.info("  Evaluating %s …", name)
        s = cross_validate(model, X, y, cv=cv, scoring=CLF_SCORERS, n_jobs=-1)
        fold_scores[name] = s["test_roc_auc"]
        records.append({
            "Model":     name,
            "Accuracy":  s["test_accuracy"].mean(),
            "Precision": s["test_precision"].mean(),
            "Recall":    s["test_recall"].mean(),
            "F1":        s["test_f1"].mean(),
            "ROC-AUC":   s["test_roc_auc"].mean(),
            "Brier": -s["test_brier"].mean(),
            "Acc±":      s["test_accuracy"].std(),
            "F1±":       s["test_f1"].std(),
            "AUC±":      s["test_roc_auc"].std(),
        })
    df = pd.DataFrame(records).set_index("Model")
    _significance_table(fold_scores, metric="ROC-AUC",
                        output_name="retention_significance.csv")
    return df


def optimise_threshold(model, X, y,
                       cost_fp: float = DEFAULT_COST_FP,
                       cost_fn: float = DEFAULT_COST_FN) -> float:
    """
    Find the probability threshold that minimises expected cost.
    cost = cost_fp * FP + cost_fn * FN
    Returns optimal threshold in [0, 1].
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import confusion_matrix

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    thresholds = np.linspace(0.1, 0.9, 81)
    total_costs = np.zeros(len(thresholds))

    for train_idx, test_idx in cv.split(X, y):
        model.fit(X[train_idx], y[train_idx])
        proba = model.predict_proba(X[test_idx])[:, 1]
        for i, t in enumerate(thresholds):
            preds = (proba >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(y[test_idx], preds,
                                              labels=[0, 1]).ravel()
            total_costs[i] += cost_fp * fp + cost_fn * fn

    best_t = thresholds[np.argmin(total_costs)]
    log.info(
        "  Optimal threshold (cost_fp=%.1f, cost_fn=%.1f): %.2f",
        cost_fp, cost_fn, best_t,
    )
    return best_t


def plot_clf_comparison(results: pd.DataFrame, title: str, filename: str):
    metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    errors = {"Accuracy": "Acc±", "F1": "F1±", "ROC-AUC": "AUC±"}
    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 5))
    fig.suptitle(f"{title} — 5-Fold Cross-Validation", fontsize=13)
    colors = plt.cm.tab10.colors

    for ax, metric in zip(axes, metrics):
        vals = results[metric]
        errs = results.get(errors.get(metric, ""),
                           pd.Series(0, index=results.index))
        bars = ax.bar(range(len(vals)), vals, yerr=errs, capsize=4,
                      color=colors[:len(vals)], alpha=0.85)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(results.index, rotation=25, ha="right", fontsize=9)
        ax.set_title(metric, fontsize=11)
        ax.set_ylim(0, 1.05)
        ax.axhline(0.5, color="gray", linewidth=0.8, linestyle="--")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out = OUTPUT_DIR / filename
    plt.savefig(out, dpi=150)
    plt.close()
    log.info("  Saved → %s", out)


def plot_retention_roc(models: dict, X, y):
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                         random_state=RANDOM_STATE)
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
        ax.plot(mean_fpr, mean_tpr, color=color, linewidth=2,
                label=f"{name} (AUC={np.mean(aucs):.2f}±{np.std(aucs):.2f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Block 2 — Retention ROC Curves")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    out = OUTPUT_DIR / "retention_roc.png"
    plt.savefig(out, dpi=150)
    plt.close()
    log.info("  Saved → %s", out)


def plot_calibration(models: dict, X, y):
    """Reliability (calibration) diagram for probabilistic classifiers."""
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                         random_state=RANDOM_STATE)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    colors = plt.cm.tab10.colors

    for (name, model), color in zip(models.items(), colors):
        # Collect OOF probabilities
        proba_all, y_all = [], []
        for train_idx, test_idx in cv.split(X, y):
            model.fit(X[train_idx], y[train_idx])
            proba_all.append(model.predict_proba(X[test_idx])[:, 1])
            y_all.append(y[test_idx])
        proba_all = np.concatenate(proba_all)
        y_all = np.concatenate(y_all)
        CalibrationDisplay.from_predictions(
            y_all, proba_all, n_bins=10,
            ax=ax, name=name, color=color,
        )

    ax.set_title("Block 2 — Calibration Plot")
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    out = OUTPUT_DIR / "retention_calibration.png"
    plt.savefig(out, dpi=150)
    plt.close()
    log.info("  Saved → %s", out)


def plot_retention_feature_importance(models: dict, feature_cols: list, X, y):
    if HAS_SHAP:
        tree_models = {k: v for k, v in models.items()
                       if k not in ("Logistic Regression",)}
        if tree_models:
            _plot_shap_importance(tree_models, feature_cols, X, y,
                                  title="Block 2 — SHAP Feature Importance (Retention)",
                                  out_name="retention_shap_importance.png")
        _plot_linear_coefs(models, feature_cols, X, y, block="block2")
        return

    importances = {}
    for name in ("Random Forest", "Decision Tree", "XGBoost"):
        if name not in models:
            continue
        m = models[name]
        m.fit(X, y)
        fi = getattr(m, "feature_importances_", None)
        if fi is not None:
            importances[name] = fi

    _plot_linear_coefs(models, feature_cols, X, y, block="block2")

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
    plt.savefig(out, dpi=150)
    plt.close()
    log.info("  Saved → %s", out)


def run_block2(
    df: pd.DataFrame,
    tune: bool = False,
    cost_fp: float = DEFAULT_COST_FP,
    cost_fn: float = DEFAULT_COST_FN,
):
    log.info("\n=== Block 2 — User Retention Classification ===")
    users = build_retention_dataset(df)

    ret = int(users["retained"].sum())
    total = len(users)
    imbalance_ratio = (total - ret) / ret if ret > 0 else float("inf")
    log.info("Users: %d,  retained: %d (%.1f%%),  not retained: %d",
             total, ret, 100 * ret / total if total else 0, total - ret)
    if imbalance_ratio > 5:
        log.warning(
            "Severe class imbalance (ratio %.1f:1). "
            "SMOTE=%s will %s applied.",
            imbalance_ratio, HAS_SMOTE,
            "be" if HAS_SMOTE else "NOT be (install imbalanced-learn)",
        )

    if total < 20:
        log.warning("Too few users — collect more sessions.")
        return
    if ret == 0 or ret == total:
        log.warning(
            "Only one class present — need both retained and non-retained users.")
        return
    if min(ret, total - ret) < 5:
        log.warning(
            "Very few samples in minority class — results will be unreliable.")

    X, y, feature_cols = engineer_retention(users)
    log.info("Features (%d): %s", len(feature_cols), feature_cols)

    spw = (total - ret) / ret if ret > 0 else 1.0
    models = build_classifiers(scale_pos_weight=spw, tune=tune, X=X, y=y)
    results = evaluate_classifiers(models, X, y)

    log.info("\n" + "=" * 65)
    log.info("BLOCK 2 RESULTS (mean over 5 folds)")
    log.info("=" * 65)
    print(results[["Accuracy", "Precision", "Recall",
          "F1", "ROC-AUC", "Brier"]].round(3).to_string())
    log.info("=" * 65)
    best = results["F1"].idxmax()
    log.info("Best model by F1: %s  (%.3f)", best, results["F1"].max())

    out = OUTPUT_DIR / "retention_results.csv"
    results.round(4).to_csv(out)
    log.info("Results saved → %s", out)

    # business-cost threshold optimisation
    best_model = models[best]
    try:
        best_model.fit(X, y)   # refit on full data for threshold search
        opt_t = optimise_threshold(
            best_model, X, y, cost_fp=cost_fp, cost_fn=cost_fn)
        thresh_path = OUTPUT_DIR / "optimal_threshold.txt"
        thresh_path.write_text(
            f"model={best}\nthreshold={opt_t:.4f}\ncost_fp={cost_fp}\ncost_fn={cost_fn}\n"
        )
        log.info("Optimal threshold saved → %s", thresh_path)
    except Exception as e:
        log.warning("Threshold optimisation failed: %s", e)

    # persist best model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_DIR / "block2_best_model.pkl")
    log.info("Best model saved → %s", MODEL_DIR / "block2_best_model.pkl")

    plot_clf_comparison(results, "Block 2 — User Retention",
                        "retention_comparison.png")
    plot_retention_roc(models, X, y)
    plot_calibration(models, X, y)
    plot_retention_feature_importance(models, feature_cols, X, y)
    plot_learning_curves(models, X, y, task="classification")


# Main


def main():
    parser = argparse.ArgumentParser(
        description="Game analytics ML pipeline — regression + retention classification"
    )
    parser.add_argument("file", nargs="?", help="TSV file to load")
    parser.add_argument("--from", dest="from_date", metavar="YYYY-MM-DD",
                        help="Only include rows on or after this date")
    parser.add_argument("--tune", action="store_true",
                        help="Run RandomizedSearchCV hyperparameter tuning")
    parser.add_argument("--cost-fp", type=float, default=DEFAULT_COST_FP,
                        help=f"Business cost of a false positive (default {DEFAULT_COST_FP})")
    parser.add_argument("--cost-fn", type=float, default=DEFAULT_COST_FN,
                        help=f"Business cost of a false negative (default {DEFAULT_COST_FN})")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    paths = [Path(args.file)] if args.file else find_latest_tsvs()
    df = load_raw(paths)

    if args.from_date:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        cutoff = pd.Timestamp(args.from_date)
        before = len(df)
        df = df[df["date"] >= cutoff].reset_index(drop=True)
        log.info("Filtered to %d rows (from %s, dropped %d)",
                 len(df), args.from_date, before - len(df))

    run_block1(df, tune=args.tune)
    run_block2(df, tune=args.tune, cost_fp=args.cost_fp, cost_fn=args.cost_fn)

    log.info("\nDone. Check reports/ for all output files.")


if __name__ == "__main__":
    main()
