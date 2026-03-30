#!/usr/bin/env python3
"""
Block 1 — Level Difficulty Curve
Descriptive. Aggregates drop_off_pct / completion_pct / hints_used
by level + theme_letter across all users. No ML, pure signal for
game designers. Outputs a ranked difficulty table + heatmap.

Block 2 — Session Quality Segmentation
Unsupervised (KMeans). Segments users into Engaged / Struggling /
Bounced based on their first-session trajectory. No label needed.

Block 3 — Level Abandonment Classifier
Binary classification per level attempt: will this level be
abandoned? Uses mid-session signals (level_seq, hints_used,
time_to_first_word_sec, completion_pct, theme_letter …).
Actionable: trigger a hint prompt before the user quits.

Block 4 — D7 Retention Classifier
Binary classification per user: did they return within 7 days?
Uses first-session level-sequence features (improvement trajectory,
last-level completion, engagement depth) that the old block ignored.

All blocks:
- Hyperparameter tuning via RandomizedSearchCV (n_iter scales with grid size)
- Tuning results saved to reports/tuning_summary_block{N}.csv + violin plots
- SHAP importance (tree models) + coefficient plots (linear models)
- Calibration plots + learning curves for classifiers
- Pairwise Wilcoxon significance tests between models
- Best model persisted via joblib
- Business-cost threshold optimisation for Block 4 (--cost-fp / --cost-fn)
- SMOTE when imbalanced-learn is available
- XGBoost → LightGBM → CatBoost fallback chain

usage:
python3 models.py [FILE] [--from YYYY-MM-DD] [--tune]
                  [--cost-fp FLOAT] [--cost-fn FLOAT]
                  [--blocks 1 2 3 4]
"""
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (
    make_scorer,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    RocCurveDisplay, brier_score_loss,
    silhouette_score, confusion_matrix
)
from sklearn.model_selection import (
    StratifiedKFold, cross_validate,
    RandomizedSearchCV, learning_curve,
)
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from scipy.stats import wilcoxon
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import argparse
import logging
import joblib
from pathlib import Path
from itertools import combinations
from math import prod

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")


# gradient boosting: XGBoost → LightGBM → CatBoost
_BOOSTER_NAME = None
try:
    from xgboost import XGBClassifier
    _BOOSTER_NAME = "xgboost"
except ImportError:
    try:
        from lightgbm import LGBMClassifier as XGBClassifier
        _BOOSTER_NAME = "lightgbm"
    except ImportError:
        try:
            from catboost import CatBoostClassifier

            class XGBClassifier(CatBoostClassifier):
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
    ImbPipeline = Pipeline

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
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# constants
RANDOM_STATE = 42
N_SPLITS = 5
INPUT_DIR = Path("reports")
OUTPUT_DIR = Path("reports/models_ver_2")
MODEL_DIR = OUTPUT_DIR / "saved_models"
RARE_THRESHOLD = 5
DEFAULT_COST_FP = 1.0  # wasted marketing spend
DEFAULT_COST_FN = 3.0  # missed returner revenue
D7_WINDOW = 7  # days for retention label


# Shared utilities

def cap_rare(s: pd.Series, threshold: int = RARE_THRESHOLD) -> pd.Series:
    rare = s.value_counts()[lambda c: c < threshold].index
    return s.where(~s.isin(rare), other="other")


def freq_encode(s: pd.Series) -> pd.Series:
    return s.map(s.value_counts()).fillna(0).astype(int)


def winsorize(s: pd.Series, upper: float = 0.95) -> pd.Series:
    cap = s.quantile(upper)
    return s.clip(upper=cap) if cap > 0 else s


def to_num(s, fill=0.0):
    return pd.to_numeric(s, errors="coerce").fillna(fill)


def hour_to_period(h: float) -> str:
    if h < 6:
        return "night"
    if h < 12:
        return "morning"
    if h < 18:
        return "afternoon"
    return "evening"


def _n_iter(grid: dict) -> int:
    size = prod(len(v) for v in grid.values())
    return 20 if size < 50 else (40 if size < 300 else 60)


def _save_csv(df: pd.DataFrame, name: str):
    p = OUTPUT_DIR / name
    df.to_csv(p, index=False)
    log.info(" Saved → %s", p)


def _savefig(name: str):
    p = OUTPUT_DIR / name
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(" Saved → %s", p)


# tuning helpers

def _run_search(name, estimator, grid, X, y, scoring, cv, summary_rows):
    n_iter = _n_iter(grid)
    log.info(" Tuning %-24s grid=%d n_iter=%d",
             name, prod(len(v) for v in grid.values()), n_iter)
    search = RandomizedSearchCV(
        estimator, grid, n_iter=n_iter, cv=cv, scoring=scoring,
        n_jobs=-1, random_state=RANDOM_STATE, refit=True,
        return_train_score=True,
    )
    search.fit(X, y)
    sign = -1 if scoring.startswith("neg_") else 1
    score = sign * search.best_score_
    log.info(" %-24s score=%.4f params=%s", name, score, search.best_params_)
    summary_rows.append({
        "model": name, "best_score": round(score, 5),
        "n_iter": n_iter,
        **{f"param_{k}": v for k, v in search.best_params_.items()},
    })
    return search.best_estimator_, search.cv_results_


def _save_tuning_artifacts(summary_rows, cv_results_map, block, scoring_label):
    if not summary_rows:
        return
    _save_csv(pd.DataFrame(summary_rows), f"tuning_summary_{block}.csv")
    if not cv_results_map:
        return
    fig, ax = plt.subplots(figsize=(max(8, 2 * len(cv_results_map)), 5))
    data, labels = [], []
    for mname, cvr in cv_results_map.items():
        scores = cvr.get("mean_test_score", np.array([]))
        scores = scores[~np.isnan(scores)]
        if len(scores):
            data.append(scores)
            labels.append(mname)
    if data:
        parts = ax.violinplot(data, showmedians=True)
        for pc in parts["bodies"]:
            pc.set_alpha(0.7)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel(scoring_label)
        ax.set_title(f"{block} — Tuning Score Distribution")
        ax.grid(axis="y", linewidth=0.5, alpha=0.5)
        plt.tight_layout()
        _savefig(f"tune_trials_{block}.png")
    plt.close()


# significance tests

def _significance_table(fold_scores: dict, output_name: str):
    rows = []
    for a, b in combinations(fold_scores.keys(), 2):
        try:
            stat, p = wilcoxon(fold_scores[a], fold_scores[b])
        except Exception:
            stat, p = float("nan"), float("nan")
        rows.append({"ModelA": a, "ModelB": b, "W-stat": stat,
                     "p-value": p, "significant": p < 0.05})
    if rows:
        _save_csv(pd.DataFrame(rows), output_name)


# feature importance plots

def _plot_importances(models, feature_cols, X, y, block, color="steelblue"):
    """SHAP for trees; coefficients for linear models."""
    tree_names = [n for n in ("Random Forest", "Decision Tree", "XGBoost")
                  if n in models]
    linear_names = [n for n in ("Logistic Regression", "Ridge", "ElasticNet")
                    if n in models]

    # tree importance / SHAP
    if tree_names:
        n = len(tree_names)
        fig, axes = plt.subplots(1, n, figsize=(7 * n, 5))
        if n == 1:
            axes = [axes]
        for ax, name in zip(axes, tree_names):
            m = models[name]
            m.fit(X, y)
            est = m.named_steps["m"] if hasattr(m, "named_steps") and "m" in m.named_steps else m
            if HAS_SHAP:
                try:
                    exp = shap.TreeExplainer(est)
                    sv = exp.shap_values(X)
                    if isinstance(sv, list):
                        sv = np.abs(np.array(sv)).mean(0)
                    imp = np.abs(sv).mean(0)
                    if imp.ndim > 1:
                        imp = imp.mean(-1)
                    xlabel = "|SHAP|"
                except Exception:
                    imp = getattr(est, "feature_importances_",
                                  np.zeros(len(feature_cols)))
                    xlabel = "Importance"
            else:
                imp = getattr(est, "feature_importances_",
                              np.zeros(len(feature_cols)))
                xlabel = "Importance"
            idx = np.argsort(imp)[::-1]
            ax.barh([feature_cols[i]
                    for i in idx], imp[idx], color=color, alpha=0.85)
            ax.set_xlabel(xlabel)
            ax.set_title(name, fontsize=11)
            ax.invert_yaxis()
        plt.suptitle(f"{block} — Feature Importance", fontsize=13)
        plt.tight_layout()
        _savefig(f"{block}_feature_importance.png")

    # linear coefficients
    if linear_names:
        n = len(linear_names)
        fig, axes = plt.subplots(1, n, figsize=(7 * n, 5))
        if n == 1:
            axes = [axes]
        for ax, name in zip(axes, linear_names):
            m = models[name]
            m.fit(X, y)
            inner = m
            while hasattr(inner, "named_steps"):
                found = False
                for step in ("clf", "reg", "model", "m"):
                    if step in inner.named_steps:
                        inner = inner.named_steps[step]
                        found = True
                        break
                if not found:
                    break
            coef = getattr(inner, "coef_", None)
            if coef is None:
                ax.set_title(f"{name}\n(no coef_)")
                continue
            coef = np.abs(coef.ravel())
            idx = np.argsort(coef)[::-1]
            ax.barh([feature_cols[i] for i in idx], coef[idx],
                    color="mediumseagreen", alpha=0.85)
            ax.set_xlabel("|coef|")
            ax.set_title(name, fontsize=11)
            ax.invert_yaxis()
        plt.suptitle(f"{block} — Linear Coefficients", fontsize=13)
        plt.tight_layout()
        _savefig(f"{block}_linear_coefs.png")


def _plot_learning_curves(models, X, y, block, scoring, cv):
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    fig.suptitle(f"{block} — Learning Curves", fontsize=13)
    for ax, (name, model) in zip(axes, models.items()):
        try:
            sizes, tr, va = learning_curve(
                model, X, y, cv=cv, scoring=scoring,
                train_sizes=np.linspace(0.2, 1.0, 5), n_jobs=-1,
            )
            ax.plot(sizes, tr.mean(1), "o-", label="train")
            ax.plot(sizes, va.mean(1), "s--", label="val")
            ax.fill_between(sizes, tr.mean(1)-tr.std(1),
                            tr.mean(1)+tr.std(1), alpha=0.1)
            ax.fill_between(sizes, va.mean(1)-va.std(1),
                            va.mean(1)+va.std(1), alpha=0.1)
        except Exception as e:
            ax.set_title(f"{name}\n({e})", fontsize=8)
            continue
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Samples")
        ax.set_ylabel(scoring)
        ax.legend(fontsize=8)
    plt.tight_layout()
    _savefig(f"{block}_learning_curves.png")


def _plot_calibration(models, X, y, block):
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                         random_state=RANDOM_STATE)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect")
    for (name, model), color in zip(models.items(), plt.cm.tab10.colors):
        pa, ya = [], []
        for tr, te in cv.split(X, y):
            model.fit(X[tr], y[tr])
            pa.append(model.predict_proba(X[te])[:, 1])
            ya.append(y[te])
        CalibrationDisplay.from_predictions(
            np.concatenate(ya), np.concatenate(pa),
            n_bins=10, ax=ax, name=name, color=color,
        )
    ax.set_title(f"{block} — Calibration")
    ax.legend(fontsize=9)
    plt.tight_layout()
    _savefig(f"{block}_calibration.png")


def _plot_roc(models, X, y, block):
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                         random_state=RANDOM_STATE)
    fig, ax = plt.subplots(figsize=(8, 6))
    mean_fpr = np.linspace(0, 1, 100)
    for (name, model), color in zip(models.items(), plt.cm.tab10.colors):
        tprs, aucs = [], []
        for tr, te in cv.split(X, y):
            try:
                model.fit(X[tr], y[tr])
                viz = RocCurveDisplay.from_estimator(
                    model, X[te], y[te], ax=ax, alpha=0)
                if np.isnan(viz.roc_auc):
                    continue
                tprs.append(np.interp(mean_fpr, viz.fpr, viz.tpr))
                aucs.append(viz.roc_auc)
            except Exception:
                continue
        if not tprs:
            continue
        mt = np.mean(tprs, 0)
        mt[-1] = 1.0
        ax.plot(mean_fpr, mt, color=color, lw=2,
                label=f"{name} AUC={np.mean(aucs):.2f}±{np.std(aucs):.2f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(f"{block} — ROC")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    _savefig(f"{block}_roc.png")


def _evaluate_classifiers(models, X, y, block):
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True,
                         random_state=RANDOM_STATE)
    scorers = {
        "accuracy": make_scorer(accuracy_score),
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
        "roc_auc": make_scorer(roc_auc_score, needs_proba=True),
        "brier": make_scorer(brier_score_loss, needs_proba=True,
                             greater_is_better=False),
    }
    records, fold_aucs = [], {}
    for name, model in models.items():
        log.info(" Evaluating %-24s …", name)
        s = cross_validate(model, X, y, cv=cv, scoring=scorers, n_jobs=-1)
        fold_aucs[name] = s["test_roc_auc"]
        records.append({
            "Model": name,
            "Accuracy": s["test_accuracy"].mean(),
            "Precision": s["test_precision"].mean(),
            "Recall": s["test_recall"].mean(),
            "F1": s["test_f1"].mean(),
            "ROC-AUC": s["test_roc_auc"].mean(),
            "Brier": -s["test_brier"].mean(),
            "F1±": s["test_f1"].std(),
            "AUC±": s["test_roc_auc"].std(),
        })
    df = pd.DataFrame(records).set_index("Model")
    _save_csv(df.round(4).reset_index(), f"{block}_results.csv")
    _significance_table(fold_aucs, f"{block}_significance.csv")
    return df


def _plot_metrics_bar(results, block, title):
    metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    errors = {"F1": "F1±", "ROC-AUC": "AUC±"}
    fig, axes = plt.subplots(1, len(metrics), figsize=(18, 5))
    fig.suptitle(f"{title} — 5-Fold CV", fontsize=13)
    colors = plt.cm.tab10.colors
    for ax, metric in zip(axes, metrics):
        vals = results[metric]
        errs = results.get(errors.get(metric, ""),
                           pd.Series(0, index=results.index))
        bars = ax.bar(range(len(vals)), vals, yerr=errs, capsize=4,
                      color=colors[:len(vals)], alpha=0.85)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(results.index, rotation=25, ha="right", fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.set_title(metric, fontsize=11)
        ax.axhline(0.5, color="gray", lw=0.8, ls="--")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.02,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    _savefig(f"{block}_comparison.png")


def _optimise_threshold(model, X, y, cost_fp, cost_fn):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    thresholds = np.linspace(0.1, 0.9, 81)
    costs = np.zeros(len(thresholds))
    for tr, te in cv.split(X, y):
        model.fit(X[tr], y[tr])
        proba = model.predict_proba(X[te])[:, 1]
        for i, t in enumerate(thresholds):
            preds = (proba >= t).astype(int)
            tn, fp, fn, tp = confusion_matrix(
                y[te], preds, labels=[0, 1]).ravel()
            costs[i] += cost_fp * fp + cost_fn * fn
    best_t = thresholds[np.argmin(costs)]
    log.info(" Optimal threshold (FP=%.1f FN=%.1f): %.2f",
             cost_fp, cost_fn, best_t)
    return best_t


def _build_booster(spw=1.0, **extra):
    if not HAS_BOOSTER:
        return None
    kw = {"random_state": RANDOM_STATE, **extra}
    if _BOOSTER_NAME != "catboost":
        kw["verbosity"] = 0
        kw["scale_pos_weight"] = spw
    else:
        kw["class_weights"] = {0: 1.0, 1: spw}
    return XGBClassifier(**kw)


class _SafeSMOTE(SMOTE):
    """SMOTE that reduces k_neighbors when minority class is too small."""
    def fit_resample(self, X, y):
        min_count = min((y == c).sum() for c in np.unique(y))
        self.k_neighbors = min(self.k_neighbors, int(min_count) - 1)
        if self.k_neighbors < 1:
            return X, y
        return super().fit_resample(X, y)


def _smote_wrap(est):
    if HAS_SMOTE:
        return ImbPipeline([("smote", _SafeSMOTE(random_state=RANDOM_STATE)), ("m", est)])
    return est


# shared classifier grids
def _clf_grids(spw=1.0):
    g = {
        "Logistic Regression": {
            "clf__C": np.logspace(-3, 3, 40).tolist(),
            "clf__penalty": ["l1", "l2"],
            "clf__solver": ["liblinear", "saga"],
        },
        "Decision Tree": {
            "max_depth": [3, 4, 6, 8, 10, None],
            "min_samples_leaf": [1, 3, 5, 10, 20],
            "min_samples_split": [2, 5, 10, 20],
            "max_features": ["sqrt", "log2", None],
        },
        "Random Forest": {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [5, 8, 10, 15, None],
            "max_features": ["sqrt", "log2", 0.3, 0.5],
            "min_samples_leaf": [1, 3, 5, 10],
        },
    }
    if HAS_BOOSTER:
        g["XGBoost"] = {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [3, 4, 5, 6, 8],
            "learning_rate": [0.005, 0.01, 0.05, 0.1, 0.2],
            "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.6, 0.7, 0.8, 1.0],
            "reg_alpha": [0, 0.01, 0.1, 1.0],
            "scale_pos_weight": [spw * f for f in (0.5, 0.75, 1.0, 1.25, 1.5)],
        }
    return g


# data loading

def find_latest_tsvs():
    ru = sorted(INPUT_DIR.glob("metrica-sessions-[0-9]*.tsv"))
    eng = sorted(INPUT_DIR.glob("metrica-sessions-eng-*.tsv"))
    paths = []
    if ru:
        paths.append(ru[-1])
    if eng:
        paths.append(eng[-1])
    if not paths:
        raise FileNotFoundError("No metrica-sessions-*.tsv in reports/.")
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
        log.info("Combined: %d rows", len(df))

    # numeric coercions done once here
    for col in ("duration_sec", "hints_used", "words_found", "words_total",
                "completion_pct", "drop_off_pct", "time_to_first_word_sec",
                "visit_duration_sec", "level_seq", "visit_count", "hour_of_day",
                "page_views"):
        if col in df.columns:
            df[col] = to_num(df[col])

    df["date"] = pd.to_datetime(df.get("date"), errors="coerce")
    return df


# BLOCK 1 — Difficulty Analysis

def run_block1(df: pd.DataFrame):
    log.info("\n=== Block 1 — Difficulty Analysis (position + theme) ===")

    needed = {"level_status", "level_seq"}
    if not needed.issubset(df.columns):
        log.warning("Missing columns %s — skipping Block 1.",
                    needed - set(df.columns))
        return

    d = df.copy()
    d["completed"] = (d["level_status"] == "completed").astype(int)
    d["abandoned"] = (d["level_status"] == "abandoned").astype(int)

    # Position-in-session curve
    # Cap level_seq at the 95th percentile to avoid noise from outlier sessions
    seq_cap = int(d["level_seq"].quantile(
        0.95)) if d["level_seq"].notna().any() else 20
    seq_df = d[d["level_seq"] <= seq_cap].copy()

    seq_agg = seq_df.groupby("level_seq").agg(
        attempts=("level_status", "count"),
        completion_rate=("completed", "mean"),
        abandonment_rate=("abandoned", "mean"),
        avg_completion_pct=("completion_pct", "mean"),
        avg_hints=("hints_used", "mean"),
        avg_time_first_word=("time_to_first_word_sec", "mean"),
        avg_duration_sec=("duration_sec", "mean"),
        avg_drop_off=("drop_off_pct", "mean"),
    ).reset_index()

    # Survival curve: share of sessions still active at each position
    # (attempts at position n / attempts at position 1)
    base_attempts = seq_agg.loc[seq_agg["level_seq"] == seq_agg["level_seq"].min(),
                                "attempts"].values
    if len(base_attempts):
        seq_agg["survival_rate"] = seq_agg["attempts"] / base_attempts[0]
    else:
        seq_agg["survival_rate"] = 1.0

    _save_csv(seq_agg, "block1_position_curve.csv")
    log.info(" Position curve: %d distinct level_seq values (capped at %d)",
             len(seq_agg), seq_cap)

    # Theme-letter breakdown
    theme_agg = None
    if "theme_letter" in d.columns:
        theme_agg = d.groupby("theme_letter").agg(
            attempts=("level_status", "count"),
            completion_rate=("completed", "mean"),
            abandonment_rate=("abandoned", "mean"),
            avg_hints=("hints_used", "mean"),
            avg_drop_off=("drop_off_pct", "mean"),
            avg_time_first_word=("time_to_first_word_sec", "mean"),
            avg_completion_pct=("completion_pct", "mean"),
        ).reset_index()

        # Composite difficulty score per theme
        theme_agg["difficulty_score"] = (
            0.5 * theme_agg["abandonment_rate"] +
            0.3 * (1 - theme_agg["completion_rate"]) +
            0.2 * (theme_agg["avg_hints"] /
                   (theme_agg["avg_hints"].max() + 1e-9))
        ).round(4)
        theme_agg = theme_agg.sort_values("difficulty_score", ascending=False)
        _save_csv(theme_agg, "block1_theme_difficulty.csv")
        log.info(" Theme difficulty (top 5):\n%s",
                 theme_agg[["theme_letter", "attempts", "completion_rate",
                            "abandonment_rate", "difficulty_score"]]
                 .head(5).to_string(index=False))

    # Level-parameter difficulty proxy
    # words_total is a parameter of the generated level; higher = harder
    param_agg = None
    if "words_total" in d.columns and d["words_total"].notna().any():
        d["words_total_bin"] = pd.cut(d["words_total"], bins=5,
                                      labels=["XS", "S", "M", "L", "XL"])
        param_agg = d.groupby("words_total_bin", observed=True).agg(
            attempts=("level_status", "count"),
            completion_rate=("completed", "mean"),
            abandonment_rate=("abandoned", "mean"),
            avg_hints=("hints_used", "mean"),
        ).reset_index()
        _save_csv(param_agg, "block1_wordcount_difficulty.csv")

    # Session-position × theme interaction
    # Which themes get harder as the user progresses through a session?
    interaction = None
    if "theme_letter" in d.columns:
        interaction = (
            d[d["level_seq"] <= seq_cap]
            .groupby(["theme_letter", "level_seq"])
            .agg(completion_rate=("completed", "mean"),
                 abandonment_rate=("abandoned", "mean"))
            .reset_index()
        )
        _save_csv(interaction, "block1_theme_x_position.csv")

    # PLOTS

    fig = plt.figure(figsize=(28, 18))
    fig.suptitle("Block 1 — Difficulty Analysis\n"
                 "(levels are procedurally generated; analysed by position & theme)",
                 fontsize=14)
    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.35)

    # Plot 1: Completion & abandonment rate by position
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(seq_agg["level_seq"], seq_agg["completion_rate"],
            "o-", color="steelblue", lw=2, ms=4, label="Completion")
    ax.plot(seq_agg["level_seq"], seq_agg["abandonment_rate"],
            "s--", color="tomato", lw=2, ms=4, label="Abandonment")
    ax.set_xlabel("Level position in session")
    ax.set_ylabel("Rate")
    ax.set_title("Completion & Abandonment by Position")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Plot 2: Session survival curve
    ax = fig.add_subplot(gs[0, 1])
    ax.fill_between(seq_agg["level_seq"], seq_agg["survival_rate"],
                    alpha=0.3, color="steelblue")
    ax.plot(seq_agg["level_seq"], seq_agg["survival_rate"],
            "o-", color="steelblue", lw=2, ms=4)
    ax.set_xlabel("Level position in session")
    ax.set_ylabel("Fraction of sessions still active")
    ax.set_title("Session Survival Curve")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)

    # Plot 3: Avg hints by position
    ax = fig.add_subplot(gs[0, 2])
    if "avg_hints" in seq_agg.columns and seq_agg["avg_hints"].notna().any():
        ax.bar(seq_agg["level_seq"], seq_agg["avg_hints"],
               color="orange", alpha=0.8)
        ax.set_xlabel("Level position in session")
        ax.set_ylabel("Avg hints used")
        ax.set_title("Hint Usage by Position")
        ax.grid(axis="y", alpha=0.3)
    else:
        ax.set_title("Hint Usage\n(no data)")
        ax.set_visible(False)

    # Plot 4: Theme difficulty bar
    ax = fig.add_subplot(gs[1, 0])
    if theme_agg is not None and len(theme_agg):
        t = theme_agg.sort_values("difficulty_score")
        norm = plt.Normalize(t["difficulty_score"].min(),
                             t["difficulty_score"].max())
        colors_t = plt.cm.RdYlGn_r(norm(t["difficulty_score"]))
        ax.barh(t["theme_letter"], t["difficulty_score"],
                color=colors_t, alpha=0.9)
        ax.set_xlabel("Difficulty score")
        ax.set_title("Theme Difficulty Ranking")
        ax.grid(axis="x", alpha=0.3)
    else:
        ax.set_title("Theme Difficulty\n(no theme_letter column)")
        ax.axis("off")

    # Plot 5: Theme heatmap
    ax = fig.add_subplot(gs[1, 1])
    if theme_agg is not None and len(theme_agg) >= 2:
        heat_cols = ["completion_rate", "abandonment_rate",
                     "avg_hints", "avg_drop_off"]
        heat_cols = [c for c in heat_cols if c in theme_agg.columns]
        heat = theme_agg.set_index("theme_letter")[heat_cols]
        sns.heatmap(heat, annot=True, fmt=".2f", cmap="RdYlGn_r",
                    linewidths=0.5, ax=ax, cbar=False)
        ax.set_title("Theme Metrics Heatmap")
    else:
        ax.set_title("Theme Heatmap\n(not enough themes)")
        ax.axis("off")

    # Plot 6: Word-count difficulty (if available)
    ax = fig.add_subplot(gs[1, 2])
    if param_agg is not None and len(param_agg):
        ax.bar(param_agg["words_total_bin"].astype(str),
               param_agg["completion_rate"], color="steelblue", alpha=0.8,
               label="Completion")
        ax.bar(param_agg["words_total_bin"].astype(str),
               -param_agg["abandonment_rate"], color="tomato", alpha=0.8,
               label="Abandonment")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xlabel("words_total bucket (XS→XL = easier→harder)")
        ax.set_ylabel("Rate")
        ax.set_title("Difficulty by Word Count")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    else:
        ax.set_title("Word-count difficulty\n(words_total not available)")
        ax.axis("off")

    _savefig("block1_difficulty_analysis.png")

    # Bonus: theme × position heatmap (completion rate)
    if interaction is not None and len(interaction["theme_letter"].unique()) >= 2:
        pivot = interaction.pivot_table(
            index="theme_letter", columns="level_seq",
            values="completion_rate", aggfunc="mean",
        )
        fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 0.6),
                                        max(4, len(pivot) * 0.5)))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn",
                    linewidths=0.3, ax=ax, vmin=0, vmax=1)
        ax.set_title("Block 1 — Completion Rate: Theme × Session Position")
        ax.set_xlabel("Level position in session")
        plt.tight_layout()
        _savefig("block1_theme_x_position_heatmap.png")

    log.info(" Block 1 complete.")

# BLOCK 2 — Session Quality Segmentation (KMeans)


def _build_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """One row per user's first session with trajectory features."""
    d = df.copy()
    d["completed"] = (d["level_status"] == "completed").astype(float)
    d["abandoned"] = (d["level_status"] == "abandoned").astype(float)

    # Per-user first-session aggregates
    def agg_first(g):
        g = g.sort_values("level_seq") if "level_seq" in g.columns else g
        n = len(g)
        comp_rate = g["completed"].mean()
        aband_rate = g["abandoned"].mean()
        avg_hints = g["hints_used"].mean() if "hints_used" in g.columns else 0
        avg_tfw = g["time_to_first_word_sec"].mean(
        ) if "time_to_first_word_sec" in g.columns else 0
        sess_dur = g["visit_duration_sec"].iloc[0] if "visit_duration_sec" in g.columns else 0

        # Improvement: did completion_pct rise across the session?
        cp = g["completion_pct"].dropna()
        if len(cp) >= 2:
            improvement = float(np.polyfit(range(len(cp)), cp, 1)[0])
        else:
            improvement = 0.0

        # Did the user finish the last level they started?
        last_status = g["level_status"].iloc[-1] if n > 0 else ""
        last_completed = 1.0 if last_status == "completed" else 0.0

        # First-level engagement
        first_words = g["words_found"].iloc[0] if "words_found" in g.columns else 0
        first_total = g["words_total"].iloc[0] if "words_total" in g.columns else 1
        first_word_rate = float(first_words) / max(float(first_total), 1)

        return pd.Series({
            "n_levels": n,
            "completion_rate": comp_rate,
            "abandonment_rate": aband_rate,
            "avg_hints": avg_hints,
            "avg_time_first_word": avg_tfw,
            "session_duration": sess_dur,
            "improvement": improvement,
            "last_completed": last_completed,
            "first_word_rate": first_word_rate,
            "hints_per_level": avg_hints / max(n, 1),
        })

    first_date = d.groupby("client_id")["date"].transform("min")
    first = d[d["date"] == first_date]
    features = (
        first.groupby("client_id")
        .apply(agg_first, include_groups=False)
        .reset_index()
    )
    return features


def run_block2(df: pd.DataFrame):
    log.info("\n=== Block 2 — Session Quality Segmentation ===")

    needed = {"client_id", "date", "level_status", "completion_pct"}
    if not needed.issubset(df.columns):
        log.warning("Missing columns — skipping Block 2.")
        return

    features = _build_session_features(df)
    feat_cols = ["n_levels", "completion_rate", "abandonment_rate",
                 "avg_hints", "avg_time_first_word", "session_duration",
                 "improvement", "last_completed", "first_word_rate",
                 "hints_per_level"]
    feat_cols = [c for c in feat_cols if c in features.columns]

    X_raw = features[feat_cols].fillna(0).values
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_raw)

    # choose k via silhouette
    k_range = range(2, min(8, len(features) // 5 + 2))
    sil_scores = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_sc)
        try:
            sil_scores.append(silhouette_score(X_sc, labels))
        except Exception:
            sil_scores.append(-1)

    best_k = list(k_range)[int(np.argmax(sil_scores))]
    log.info(" Best k=%d (silhouette=%.3f)", best_k, max(sil_scores))

    km_final = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    features["cluster"] = km_final.fit_predict(X_sc)

    # label clusters by engagement level
    cluster_means = features.groupby("cluster")[feat_cols].mean()
    # Rank clusters by completion_rate descending to assign labels
    ranked = cluster_means["completion_rate"].sort_values(
        ascending=False).index.tolist()
    label_map = {}
    segment_labels = ["Engaged", "Struggling", "Bounced", "Casual",
                      "Power User", "One-and-Done", "Explorers"]
    for i, cid in enumerate(ranked):
        label_map[cid] = segment_labels[i] if i < len(
            segment_labels) else f"Cluster {cid}"
    features["segment"] = features["cluster"].map(label_map)

    segment_profile = features.groupby("segment")[feat_cols].mean().round(3)
    segment_profile["user_count"] = features.groupby("segment").size()
    _save_csv(segment_profile.reset_index(), "block2_segment_profiles.csv")
    _save_csv(features[["client_id", "cluster", "segment"]],
              "block2_user_segments.csv")

    log.info(" Segment sizes:\n%s",
             features["segment"].value_counts().to_string())

    # plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Block 2 — Session Quality Segments", fontsize=14)
    colors = plt.cm.tab10.colors

    # Silhouette curve
    ax = axes[0]
    ax.plot(list(k_range), sil_scores, "o-", color="steelblue")
    ax.axvline(best_k, color="tomato", ls="--", label=f"Best k={best_k}")
    ax.set_xlabel("k")
    ax.set_ylabel("Silhouette score")
    ax.set_title("Optimal k Selection")
    ax.legend()

    # Segment sizes
    ax = axes[1]
    seg_counts = features["segment"].value_counts()
    ax.barh(seg_counts.index, seg_counts.values,
            color=colors[:len(seg_counts)], alpha=0.85)
    ax.set_xlabel("Users")
    ax.set_title("Segment Sizes")
    for i, (idx, val) in enumerate(seg_counts.items()):
        ax.text(val + 0.3, i, str(val), va="center", fontsize=9)

    # PCA scatter
    ax = axes[2]
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_2d = pca.fit_transform(X_sc)
    for i, seg in enumerate(features["segment"].unique()):
        mask = features["segment"] == seg
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   label=seg, alpha=0.6, s=20, color=colors[i % len(colors)])
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.0%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.0%})")
    ax.set_title("PCA — Segment Clusters")
    ax.legend(fontsize=8, markerscale=2)

    plt.tight_layout()
    _savefig("block2_segments.png")

    # Heatmap of segment profiles
    heat_cols = ["completion_rate", "abandonment_rate", "avg_hints",
                 "improvement", "last_completed", "n_levels"]
    heat_cols = [c for c in heat_cols if c in segment_profile.columns]
    if heat_cols:
        fig, ax = plt.subplots(figsize=(10, max(4, best_k)))
        sns.heatmap(segment_profile[heat_cols], annot=True, fmt=".2f",
                    cmap="RdYlGn", linewidths=0.5, ax=ax)
        ax.set_title("Block 2 — Segment Feature Profiles")
        plt.tight_layout()
        _savefig("block2_segment_heatmap.png")

    log.info(" Block 2 complete.")


# BLOCK 3 — Level Abandonment Classifier

def _build_abandonment_dataset(df: pd.DataFrame):
    """One row per level attempt; target = abandoned (1) or not (0)."""
    d = df.copy()
    d["target"] = (d["level_status"] != "completed").astype(int)

    # features available at or early in a level attempt
    num_cols = ["level_seq", "hints_used", "time_to_first_word_sec",
                "completion_pct", "duration_sec", "hour_of_day",
                "visit_count", "page_views"]
    cat_cols = ["theme_letter", "device_category", "browser", "os",
                "ab_group", "session_period", "is_weekend"]

    d["session_period"] = d["hour_of_day"].apply(
        lambda h: hour_to_period(float(h)) if pd.notna(h) else "unknown"
    )
    if "date" in d.columns:
        d["is_weekend"] = d["date"].dt.dayofweek.isin([5, 6]).astype(int)

    # Previous-attempt performance in same session (rolling context)
    d = d.sort_values(["session_id", "level_seq"])
    d["prev_completion_pct"] = d.groupby(
        "session_id")["completion_pct"].shift(1).fillna(0)
    d["prev_hints"] = d.groupby("session_id")["hints_used"].shift(1).fillna(0)
    d["attempts_so_far"] = d.groupby("session_id").cumcount()
    d["session_abandon_rate_so_far"] = (
        d.groupby("session_id")["target"]
        .transform(lambda x: x.shift(1).expanding().mean())
        .fillna(0)
    )
    num_cols += ["prev_completion_pct", "prev_hints",
                 "attempts_so_far", "session_abandon_rate_so_far"]

    for col in num_cols:
        if col in d.columns:
            d[col] = winsorize(to_num(d[col]))

    for col in cat_cols:
        if col in d.columns:
            d[col] = freq_encode(
                cap_rare(d[col].fillna("unknown").astype(str)))
        else:
            d[col] = 0

    feat_cols = [c for c in num_cols + cat_cols if c in d.columns]
    d_clean = d[feat_cols + ["target"]].dropna(subset=feat_cols)
    return d_clean[feat_cols].values, d_clean["target"].values, feat_cols


def _build_clf_models(spw, tune, X, y, grids_override=None, block=""):
    cv_inner = StratifiedKFold(
        n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    grids = grids_override or _clf_grids(spw)
    summary, cv_results = [], {}

    def maybe_tune(name, est):
        g = grids.get(name, {})
        if not tune or not g or X is None:
            return est
        best, cvr = _run_search(
            name, est, g, X, y, "roc_auc", cv_inner, summary)
        cv_results[name] = cvr
        return best

    lr = maybe_tune("Logistic Regression", Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE,
                                   class_weight="balanced")),
    ]))
    dt = maybe_tune("Decision Tree",
                    DecisionTreeClassifier(random_state=RANDOM_STATE,
                                           class_weight="balanced"))
    rf = maybe_tune("Random Forest",
                    RandomForestClassifier(random_state=RANDOM_STATE,
                                           class_weight="balanced", n_jobs=-1))

    models = {
        "Logistic Regression": _smote_wrap(lr),
        "Decision Tree": _smote_wrap(dt),
        "Random Forest": _smote_wrap(rf),
    }

    if HAS_BOOSTER:
        xgb = maybe_tune("XGBoost", _build_booster(spw))
        if xgb:
            models["XGBoost"] = _smote_wrap(xgb)

    if tune and block:
        _save_tuning_artifacts(summary, cv_results, block, "ROC-AUC")

    return models


def run_block3(df: pd.DataFrame, tune: bool = False):
    log.info("\n=== Block 3 — Level Abandonment Classifier ===")

    needed = {"session_id", "level_status", "level_seq", "completion_pct"}
    if not needed.issubset(df.columns):
        log.warning("Missing columns — skipping Block 3.")
        return

    X, y, feat_cols = _build_abandonment_dataset(df)
    n_pos = int(y.sum())
    n_total = len(y)
    log.info(" Attempts: %d, abandoned: %d (%.1f%%)",
             n_total, n_pos, 100 * n_pos / n_total)

    if n_total < 50:
        log.warning("Too few attempts — skipping Block 3.")
        return
    if n_pos == 0:
        log.warning("No abandoned levels found — skipping Block 3.")
        return

    log.info(" Features (%d): %s", len(feat_cols), feat_cols)
    spw = (n_total - n_pos) / max(n_pos, 1)
    models = _build_clf_models(spw, tune, X, y, block="block3")
    results = _evaluate_classifiers(models, X, y, "block3")

    log.info("\n" + "=" * 60)
    print(results[["Accuracy", "Precision", "Recall",
          "F1", "ROC-AUC"]].round(3).to_string())
    log.info("=" * 60)
    best = results["F1"].idxmax()
    log.info("Best by F1: %s (%.3f)", best, results["F1"].max())

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_model = models[best]
    best_model.fit(X, y)
    joblib.dump(best_model, MODEL_DIR / "block3_abandonment_model.pkl")
    log.info(" Model saved → %s", MODEL_DIR / "block3_abandonment_model.pkl")

    cv_lc = StratifiedKFold(n_splits=3, shuffle=True,
                            random_state=RANDOM_STATE)
    _plot_metrics_bar(results, "block3", "Block 3 — Level Abandonment")
    _plot_roc(models, X, y, "block3")
    _plot_calibration(models, X, y, "block3")
    _plot_importances(models, feat_cols, X, y, "block3", color="darkorange")
    _plot_learning_curves(models, X, y, "block3", "roc_auc", cv_lc)

    log.info(" Block 3 complete.")


# BLOCK 4 — D7 Retention Classifier

def _build_d7_dataset(df: pd.DataFrame, window: int = D7_WINDOW):
    """
    One row per user. Label = returned within `window` days of first session.
    Features come entirely from the first session only.
    """
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")

    # D7 label
    first_date = d.groupby("client_id")["date"].min().rename("first_date")
    last_date = d.groupby("client_id")["date"].max().rename("last_date")
    dates_df = pd.concat([first_date, last_date], axis=1)
    dates_df["days_gap"] = (dates_df["last_date"] -
                            dates_df["first_date"]).dt.days
    # Only label users whose first session is old enough to observe D7
    # (i.e. first_date <= max_date - window)
    max_date = d["date"].max()
    observable = dates_df[dates_df["first_date"]
                          <= max_date - pd.Timedelta(days=window)]
    observable["retained_d7"] = (observable["days_gap"] <= window) & \
        (observable["days_gap"] > 0)
    observable["retained_d7"] = observable["retained_d7"].astype(int)

    if len(observable) == 0:
        return None, None, None

    # first-session features
    first = d[d["date"] == d.groupby(
        "client_id")["date"].transform("min")].copy()
    first = first[first["client_id"].isin(observable.index)]

    def agg_user(g):
        g = g.sort_values("level_seq") if "level_seq" in g.columns else g
        n = len(g)
        comp_rate = (g["level_status"] == "completed").mean()
        aband_rate = (g["level_status"] == "abandoned").mean()
        avg_hints = g["hints_used"].mean()
        avg_tfw = g["time_to_first_word_sec"].mean()
        sess_dur = g["visit_duration_sec"].iloc[0] \
            if "visit_duration_sec" in g.columns else 0
        hour = g["hour_of_day"].iloc[0] \
            if "hour_of_day" in g.columns else 12.0

        # Trajectory: slope of completion_pct across level_seq
        cp = g["completion_pct"].dropna()
        improvement = float(np.polyfit(range(len(cp)), cp, 1)[0]) \
            if len(cp) >= 2 else 0.0

        # Last-level outcome
        last_completed = 1.0 if g["level_status"].iloc[-1] == "completed" else 0.0

        # First-level word-finding rate
        fw = g["words_found"].iloc[0] if "words_found" in g.columns else 0
        ft = g["words_total"].iloc[0] if "words_total" in g.columns else 1
        first_word_rate = float(fw) / max(float(ft), 1)

        # Theme diversity: how many distinct themes played?
        n_themes = g["theme_letter"].nunique(
        ) if "theme_letter" in g.columns else 1

        # engagement score: completion × duration
        engagement = comp_rate * max(float(sess_dur), 0)

        ts = g["date"].iloc[0]
        dow = ts.day_name() if pd.notna(ts) else "Unknown"
        weekend = int(ts.dayofweek in (5, 6)) if pd.notna(ts) else 0

        return pd.Series({
            # volume
            "n_levels": n,
            "session_duration": sess_dur,
            "page_views": g["page_views"].iloc[0]
            if "page_views" in g.columns else 0,
            # performance
            "completion_rate": comp_rate,
            "abandonment_rate": aband_rate,
            "avg_hints": avg_hints,
            "hints_per_level": avg_hints / max(n, 1),
            "avg_time_first_word": avg_tfw,
            # trajectory
            "improvement": improvement,
            "last_completed": last_completed,
            "first_word_rate": first_word_rate,
            "n_themes": n_themes,
            # engagement composite
            "engagement_score": engagement,
            # time context
            "hour_of_day": hour,
            "session_period_enc": {"night": 0, "morning": 1, "afternoon": 2, "evening": 3}
            .get(hour_to_period(float(hour)), 1),
            "day_of_week": dow,
            "is_weekend": weekend,
            # acquisition
            "ab_group": g["ab_group"].iloc[0]
            if "ab_group" in g.columns else "unknown",
            "device_category": g["device_category"].iloc[0]
            if "device_category" in g.columns else "unknown",
            "region": g["region"].iloc[0]
            if "region" in g.columns else "unknown",
            "utm_source": g["utm_source"].iloc[0]
            if "utm_source" in g.columns else "unknown",
            "visit_count": g["visit_count"].iloc[0]
            if "visit_count" in g.columns else 1,
        })

    users = (
        first.groupby("client_id")
        .apply(agg_user, include_groups=False)
        .reset_index()
        .merge(observable[["retained_d7"]].reset_index(), on="client_id")
    )
    return users


def _engineer_retention(users: pd.DataFrame):
    d = users.copy()

    num_cols = ["n_levels", "session_duration", "page_views", "completion_rate",
                "abandonment_rate", "avg_hints", "hints_per_level",
                "avg_time_first_word", "improvement", "last_completed",
                "first_word_rate", "n_themes", "engagement_score",
                "hour_of_day", "session_period_enc", "is_weekend", "visit_count"]
    cat_cols = ["day_of_week", "ab_group",
                "device_category", "region", "utm_source"]

    for col in num_cols:
        if col in d.columns:
            d[col] = winsorize(to_num(d[col]))

    for col in cat_cols:
        if col in d.columns:
            d[col] = freq_encode(
                cap_rare(d[col].fillna("unknown").astype(str)))
        else:
            d[col] = 0

    feat_cols = [c for c in num_cols + cat_cols if c in d.columns]
    return d[feat_cols].values, d["retained_d7"].values, feat_cols


def run_block4(df: pd.DataFrame, tune: bool = False,
               cost_fp: float = DEFAULT_COST_FP,
               cost_fn: float = DEFAULT_COST_FN):
    log.info("\n=== Block 4 — D7 Retention Classifier ===")

    needed = {"client_id", "date", "level_status", "completion_pct"}
    if not needed.issubset(df.columns):
        log.warning("Missing columns — skipping Block 4.")
        return

    users = _build_d7_dataset(df, window=D7_WINDOW)
    if users is None or len(users) == 0:
        log.warning("Not enough temporal spread for D7 label — need %d+ days of data.",
                    D7_WINDOW + 1)
        return

    ret = int(users["retained_d7"].sum())
    total = len(users)
    log.info(" Observable users: %d, D7-retained: %d (%.1f%%), not: %d",
             total, ret, 100*ret/total if total else 0, total-ret)

    if total < 30:
        log.warning("Too few users — skipping Block 4.")
        return
    if ret == 0 or ret == total:
        log.warning("Only one class present.")
        return
    if min(ret, total-ret) < 5:
        log.warning("Minority class very small — results unreliable.")

    imbalance = (total - ret) / max(ret, 1)
    if imbalance > 5:
        log.warning("Imbalance ratio %.1f:1. SMOTE=%s.", imbalance, HAS_SMOTE)

    X, y, feat_cols = _engineer_retention(users)
    log.info(" Features (%d): %s", len(feat_cols), feat_cols)

    spw = (total - ret) / max(ret, 1)
    models = _build_clf_models(spw, tune, X, y, block="block4")
    results = _evaluate_classifiers(models, X, y, "block4")

    log.info("\n" + "=" * 65)
    print(results[["Accuracy", "Precision", "Recall",
          "F1", "ROC-AUC", "Brier"]].round(3).to_string())
    log.info("=" * 65)
    best = results["F1"].idxmax()
    log.info("Best by F1: %s (%.3f)", best, results["F1"].max())

    # baseline comparison
    baseline_auc = 0.5
    best_auc = results.loc[best, "ROC-AUC"]
    lift = best_auc - baseline_auc
    log.info(" AUC lift over random baseline: +%.3f", lift)
    if lift < 0.05:
        log.warning(
            " AUC lift is very small — check feature quality and label.")

    # persist
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_model = models[best]
    best_model.fit(X, y)
    joblib.dump(best_model, MODEL_DIR / "block4_retention_model.pkl")
    log.info(" Model saved → %s", MODEL_DIR / "block4_retention_model.pkl")

    # threshold optimisation
    try:
        opt_t = _optimise_threshold(best_model, X, y, cost_fp, cost_fn)
        thresh_path = OUTPUT_DIR / "block4_optimal_threshold.txt"
        thresh_path.write_text(
            f"model={best}\nthreshold={opt_t:.4f}\n"
            f"cost_fp={cost_fp}\ncost_fn={cost_fn}\n"
            f"d7_window={D7_WINDOW}\n"
        )
        log.info(" Threshold → %s", thresh_path)
    except Exception as e:
        log.warning(" Threshold optimisation failed: %s", e)

    # plots
    cv_lc = StratifiedKFold(n_splits=3, shuffle=True,
                            random_state=RANDOM_STATE)
    _plot_metrics_bar(results, "block4", f"Block 4 — D{D7_WINDOW} Retention")
    _plot_roc(models, X, y, "block4")
    _plot_calibration(models, X, y, "block4")
    _plot_importances(models, feat_cols, X, y, "block4", color="mediumpurple")
    _plot_learning_curves(models, X, y, "block4", "roc_auc", cv_lc)

    # feature correlation with retention (quick diagnostic)
    try:
        corr = (
            pd.DataFrame(X, columns=feat_cols)
            .assign(retained=y)
            .corr()["retained"]
            .drop("retained")
            .sort_values(key=abs, ascending=False)
        )
        _save_csv(corr.reset_index().rename(columns={"index": "feature", "retained": "corr_with_d7"}),
                  "block4_feature_correlations.csv")

        fig, ax = plt.subplots(figsize=(8, max(4, len(corr) * 0.3)))
        colors_c = ["steelblue" if v > 0 else "tomato" for v in corr]
        ax.barh(corr.index[::-1], corr.values[::-1],
                color=colors_c[::-1], alpha=0.85)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("Pearson correlation with D7 retention")
        ax.set_title("Block 4 — Feature–Retention Correlation")
        plt.tight_layout()
        _savefig("block4_feature_correlations.png")
    except Exception as e:
        log.warning(" Correlation plot failed: %s", e)

    log.info(" Block 4 complete.")


# Main

def main():
    parser = argparse.ArgumentParser(
        description="Game analytics ML pipeline — 4 focused blocks"
    )
    parser.add_argument("file", nargs="?", help="TSV file to load")
    parser.add_argument("--from", dest="from_date", metavar="YYYY-MM-DD")
    parser.add_argument("--tune", action="store_true",
                        help="Run RandomizedSearchCV tuning for all classifiers")
    parser.add_argument("--cost-fp", type=float, default=DEFAULT_COST_FP)
    parser.add_argument("--cost-fn", type=float, default=DEFAULT_COST_FN)
    parser.add_argument("--blocks", nargs="+", type=int, default=[1, 2, 3, 4],
                        help="Which blocks to run (e.g. --blocks 1 4)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    paths = [Path(args.file)] if args.file else find_latest_tsvs()
    df = load_raw(paths)

    if args.from_date:
        cutoff = pd.Timestamp(args.from_date)
        before = len(df)
        df = df[df["date"] >= cutoff].reset_index(drop=True)
        log.info("Filtered to %d rows (from %s, dropped %d)",
                 len(df), args.from_date, before - len(df))

    if 1 in args.blocks:
        run_block1(df)
    if 2 in args.blocks:
        run_block2(df)
    if 3 in args.blocks:
        run_block3(df, tune=args.tune)
    if 4 in args.blocks:
        run_block4(df, tune=args.tune,
                   cost_fp=args.cost_fp,
                   cost_fn=args.cost_fn)

    log.info("\nDone. All outputs in reports/")


if __name__ == "__main__":
    main()
