"""
Block 0  Positional difficulty curve (descriptive, no ML)
Block 1  Session quality segmentation (KMeans, unsupervised)
Block 2  Level abandonment classifier (binary classification per attempt)
Block 3  Return classifier (binary classification per user)

All classifier blocks:
  - RandomizedSearchCV hyperparameter tuning (--tune flag)
  - ROC-AUC, F1, calibration metrics via 5-fold stratified CV
  - SHAP feature importance for tree models
  - Pairwise Wilcoxon significance tests
  - Best model persisted via joblib

Usage:
    python3 models.py [FILE] [--from YYYY-MM-DD] [--tune] [--blocks 1 2 3 4]
"""
import argparse
import logging
import warnings
from itertools import combinations
from math import prod
from pathlib import Path
import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import norm, wilcoxon
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_curve,
    auc as sk_auc,
    silhouette_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RANDOM_STATE = 42
N_CV_SPLITS = 5  # outer cross-validation folds
N_CV_INNER = 3  # inner folds for hyperparameter search
RARE_THRESHOLD = 5  # min category frequency before collapsing to "other"
WINSORIZE_PCT = 0.95  # upper percentile for winsorisation

INPUT_DIR = Path("reports")
OUTPUT_DIR = Path("reports/models")
MODEL_DIR = OUTPUT_DIR / "saved_models"

# Shared utilities


def cap_rare(s: pd.Series, threshold: int = RARE_THRESHOLD) -> pd.Series:
    """Replace infrequent categories with 'other'."""
    rare = s.value_counts()[lambda c: c < threshold].index
    return s.where(~s.isin(rare), other="other")


def freq_encode(s: pd.Series) -> pd.Series:
    """Encode a categorical series by its value counts."""
    return s.map(s.value_counts()).fillna(0).astype(int)


def winsorize(s: pd.Series, upper: float = WINSORIZE_PCT) -> pd.Series:
    """Clip values above the given percentile."""
    cap = s.quantile(upper)
    return s.clip(upper=cap) if cap > 0 else s


def to_num(s, fill: float = 0.0) -> pd.Series:
    """Coerce to numeric, filling non-parseable values with fill."""
    return pd.to_numeric(s, errors="coerce").fillna(fill)


def hour_to_period(hour: float) -> str:
    """Map an hour of day to a named period."""
    if hour < 6:
        return "night"
    if hour < 12:
        return "morning"
    if hour < 18:
        return "afternoon"
    return "evening"


def _search_budget(grid: dict) -> int:
    """Scale RandomizedSearchCV iterations with grid size."""
    size = prod(len(v) for v in grid.values())
    return 20 if size < 50 else (40 if size < 300 else 60)


def _save_csv(df: pd.DataFrame, name: str) -> None:
    path = OUTPUT_DIR / name
    df.to_csv(path, index=False)
    log.info("Saved → %s", path)


def _save_figure(name: str) -> None:
    path = OUTPUT_DIR / name
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved → %s", path)


# Data loading


def _find_latest_tsv() -> list[Path]:
    tsv = sorted(INPUT_DIR.glob("metrica-sessions-[0-9]*.tsv"))
    if not tsv:
        raise FileNotFoundError("No metrica-sessions-[0-9]*.tsv found in reports/.")
    return [tsv[-1]]


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
        log.info("Combined: %d rows total", len(df))

    numeric_cols = (
        "duration_sec",
        "hints_used",
        "words_found",
        "words_total",
        "completion_pct",
        "drop_off_pct",
        "time_to_first_word_sec",
        "visit_duration_sec",
        "level_seq",
        "visit_count",
        "hour_of_day",
        "page_views",
    )
    for col in numeric_cols:
        if col in df.columns:
            df[col] = to_num(df[col])

    df["date"] = pd.to_datetime(df.get("date"), errors="coerce")
    return df


# Hyperparameter search helpers


def _run_search(name, estimator, grid, X, y, scoring, cv, summary_rows):
    n_iter = _search_budget(grid)
    log.info(
        "Tuning %-24s  grid=%d  n_iter=%d",
        name,
        prod(len(v) for v in grid.values()),
        n_iter,
    )

    search = RandomizedSearchCV(
        estimator,
        grid,
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
    log.info("%-24s  score=%.4f  params=%s", name, score, search.best_params_)

    summary_rows.append(
        {
            "model": name,
            "best_score": round(score, 5),
            "n_iter": n_iter,
            **{f"param_{k}": v for k, v in search.best_params_.items()},
        }
    )
    return search.best_estimator_, search.cv_results_


def _save_tuning_summary(summary_rows, cv_results_map, block, scoring_label):
    if not summary_rows:
        return
    _save_csv(pd.DataFrame(summary_rows), f"tuning_summary_{block}.csv")
    if not cv_results_map:
        return

    fig, ax = plt.subplots(figsize=(max(8, 2 * len(cv_results_map)), 5))
    data, labels = [], []
    for model_name, cvr in cv_results_map.items():
        scores = cvr.get("mean_test_score", np.array([]))
        scores = scores[~np.isnan(scores)]
        if len(scores):
            data.append(scores)
            labels.append(model_name)

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
        _save_figure(f"tune_trials_{block}.png")
    plt.close()


# Evaluation helpers


def _wilcoxon_significance(fold_scores: dict, output_name: str) -> None:
    """Pairwise Wilcoxon tests between models; saves results to CSV."""
    rows = []
    for model_a, model_b in combinations(fold_scores.keys(), 2):
        try:
            stat, p = wilcoxon(fold_scores[model_a], fold_scores[model_b])
        except Exception:
            stat, p = float("nan"), float("nan")
        rows.append(
            {
                "ModelA": model_a,
                "ModelB": model_b,
                "W-stat": stat,
                "p-value": p,
                "significant": p < 0.05,
            }
        )
    if rows:
        _save_csv(pd.DataFrame(rows), output_name)


def _evaluate_classifiers(models: dict, X, y, block: str) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=N_CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scorers = {
        "accuracy": make_scorer(accuracy_score),
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
        "brier": make_scorer(
            brier_score_loss, needs_proba=True, greater_is_better=False
        ),
    }
    records, fold_f1s = [], {}

    for name, model in models.items():
        log.info("Evaluating %-24s …", name)
        scores = cross_validate(model, X, y, cv=cv, scoring=scorers, n_jobs=-1)
        fold_f1s[name] = scores["test_f1"]

        # compute AUC manually per fold
        fold_aucs = []
        for train_idx, test_idx in cv.split(X, y):
            try:
                model.fit(X[train_idx], y[train_idx])
                proba = model.predict_proba(X[test_idx])[:, 1]
                fpr, tpr, _ = roc_curve(y[test_idx], proba)
                fold_aucs.append(sk_auc(fpr, tpr))
            except Exception:
                continue

        mean_auc = float(np.mean(fold_aucs)) if fold_aucs else float("nan")
        std_auc = float(np.std(fold_aucs)) if fold_aucs else float("nan")

        records.append(
            {
                "Model": name,
                "Accuracy": scores["test_accuracy"].mean(),
                "Precision": scores["test_precision"].mean(),
                "Recall": scores["test_recall"].mean(),
                "F1": scores["test_f1"].mean(),
                "ROC-AUC": mean_auc,
                "Brier": -scores["test_brier"].mean(),
                "F1±": scores["test_f1"].std(),
                "AUC±": std_auc,
            }
        )

    results_df = pd.DataFrame(records).set_index("Model")
    _save_csv(results_df.round(4).reset_index(), f"{block}_results.csv")
    _wilcoxon_significance(fold_f1s, f"{block}_significance.csv")
    return results_df


def _compute_roc_curves(models: dict, X, y) -> dict:
    """
    Compute mean ROC curves across CV folds for each model.
    Returns {model_name: (mean_tpr, mean_auc, std_auc)}.
    """
    cv = StratifiedKFold(n_splits=N_CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    mean_fpr = np.linspace(0, 1, 100)
    roc_data = {}

    for name, model in models.items():
        tprs, aucs = [], []
        for train_idx, test_idx in cv.split(X, y):
            try:
                model.fit(X[train_idx], y[train_idx])
                proba = model.predict_proba(X[test_idx])[:, 1]
                fpr, tpr, _ = roc_curve(y[test_idx], proba)
                tprs.append(np.interp(mean_fpr, fpr, tpr))
                aucs.append(sk_auc(fpr, tpr))
            except Exception:
                continue
        if tprs:
            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            roc_data[name] = (mean_tpr, float(np.mean(aucs)), float(np.std(aucs)))

    return roc_data


# Model building


def _build_xgboost(scale_pos_weight: float = 1.0, **kwargs) -> XGBClassifier:
    return XGBClassifier(
        random_state=RANDOM_STATE,
        verbosity=0,
        scale_pos_weight=scale_pos_weight,
        **kwargs,
    )


def _classifier_grids(scale_pos_weight: float = 1.0) -> dict:
    return {
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
        "XGBoost": {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [3, 4, 5, 6, 8],
            "learning_rate": [0.005, 0.01, 0.05, 0.1, 0.2],
            "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.6, 0.7, 0.8, 1.0],
            "reg_alpha": [0, 0.01, 0.1, 1.0],
            "scale_pos_weight": [
                scale_pos_weight * f for f in (0.5, 0.75, 1.0, 1.25, 1.5)
            ],
        },
    }


def _build_classifiers(
    scale_pos_weight: float,
    tune: bool,
    X,
    y,
    block: str = "",
    grids_override: dict | None = None,
    use_smote: bool = False,
) -> dict:
    """
    Build and optionally tune four classifiers.
    use_smote wraps each model in an SMOTE pipeline (Block 2 only).
    """
    cv_inner = StratifiedKFold(
        n_splits=N_CV_INNER, shuffle=True, random_state=RANDOM_STATE
    )
    grids = grids_override or _classifier_grids(scale_pos_weight)
    summary_rows = []
    cv_results_map = {}

    def _maybe_tune(name, estimator):
        grid = grids.get(name, {})
        if not (tune and grid):
            return estimator
        best, cvr = _run_search(
            name, estimator, grid, X, y, "roc_auc", cv_inner, summary_rows
        )
        cv_results_map[name] = cvr
        return best

    lr_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    base_models = {
        "Logistic Regression": _maybe_tune("Logistic Regression", lr_pipeline),
        "Decision Tree": _maybe_tune(
            "Decision Tree",
            DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
        ),
        "Random Forest": _maybe_tune(
            "Random Forest",
            RandomForestClassifier(
                random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1
            ),
        ),
        "XGBoost": _maybe_tune("XGBoost", _build_xgboost(scale_pos_weight)),
    }

    if use_smote:
        base_models = {
            name: ImbPipeline(
                [
                    ("smote", _SafeSMOTE(random_state=RANDOM_STATE)),
                    ("classifier", model),
                ]
            )
            for name, model in base_models.items()
        }

    if tune and block:
        _save_tuning_summary(summary_rows, cv_results_map, block, "ROC-AUC")

    return base_models


class _SafeSMOTE(SMOTE):
    """SMOTE that reduces k_neighbors when the minority class is too small."""

    def fit_resample(self, X, y):
        min_count = min((y == c).sum() for c in np.unique(y))
        self.k_neighbors = min(self.k_neighbors, int(min_count) - 1)
        if self.k_neighbors < 1:
            return X, y
        return super().fit_resample(X, y)


# BLOCK 0 — Positional difficulty curve


def run_block0(df: pd.DataFrame) -> None:
    log.info("\n=== Block 0 — Positional Difficulty Curve ===")

    if not {"level_status", "level_seq"}.issubset(df.columns):
        log.warning("Missing required columns — skipping Block 0.")
        return

    d = df.copy()
    d["completed"] = (d["level_status"] == "completed").astype(int)

    seq_cap = int(d["level_seq"].quantile(0.95)) if d["level_seq"].notna().any() else 20
    agg = (
        d[d["level_seq"] <= seq_cap]
        .groupby("level_seq")
        .agg(
            attempts=("level_status", "count"),
            completion_rate=("completed", "mean"),
            avg_completion_pct=("completion_pct", "mean"),
            avg_hints=("hints_used", "mean"),
            avg_time_first_word=("time_to_first_word_sec", "mean"),
            avg_duration_sec=("duration_sec", "mean"),
        )
        .reset_index()
    )

    base = agg.loc[agg["level_seq"] == agg["level_seq"].min(), "attempts"].values
    agg["survival_rate"] = agg["attempts"] / base[0] if len(base) else 1.0

    _save_csv(agg, "block0_position_curve.csv")

    # Plots
    level_labels = [f"Уровень {int(x) + 1}" for x in agg["level_seq"]]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Позиционная кривая сложности уровней", fontsize=13)

    def _line_panel(ax, col, color, ylabel, title):
        ax.plot(agg["level_seq"], agg[col], "o-", color=color, lw=2, ms=6)
        ax.fill_between(agg["level_seq"], agg[col], alpha=0.15, color=color)
        for x, y_val in zip(agg["level_seq"], agg[col]):
            ax.annotate(
                f"{y_val:.1%}",
                (x, y_val),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=9,
            )
        ax.set_xticks(agg["level_seq"])
        ax.set_xticklabels(level_labels)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_ylim(0, 1.15)
        ax.grid(axis="y", alpha=0.3)

    _line_panel(
        axes[0],
        "survival_rate",
        "steelblue",
        "Доля активных сессий",
        "Кривая выживаемости сессий",
    )
    _line_panel(
        axes[1],
        "completion_rate",
        "seagreen",
        "Доля завершённых уровней",
        "Завершаемость по позиции уровня",
    )

    bars = axes[2].bar(
        agg["level_seq"], agg["avg_hints"], color="darkorange", alpha=0.8, width=0.4
    )
    for bar, val in zip(bars, agg["avg_hints"]):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    axes[2].set_xticks(agg["level_seq"])
    axes[2].set_xticklabels(level_labels)
    axes[2].set_ylabel("Среднее число подсказок")
    axes[2].set_title("Использование подсказок по позиции уровня")
    axes[2].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    _save_figure("block0_position_curve.png")
    log.info("Block 0 complete.")


# BLOCK 1 — Session quality segmentation (KMeans)


def _build_first_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate first-session behaviour into one row per user."""
    d = df.copy()
    d["completed"] = (d["level_status"] == "completed").astype(float)

    def _agg_user(g):
        g = g.sort_values("level_seq") if "level_seq" in g.columns else g
        n = len(g)
        comp_rate = g["completed"].mean()

        cp = g["completion_pct"].dropna()
        improvement = (
            float(np.polyfit(range(len(cp)), cp, 1)[0]) if len(cp) >= 2 else 0.0
        )

        first_words = g["words_found"].iloc[0] if "words_found" in g.columns else 0
        first_total = g["words_total"].iloc[0] if "words_total" in g.columns else 1

        return pd.Series(
            {
                "n_levels": n,
                "completion_rate": comp_rate,
                "abandonment_rate": 1.0 - comp_rate,
                "avg_hints": g["hints_used"].mean()
                if "hints_used" in g.columns
                else 0.0,
                "avg_time_first_word": g["time_to_first_word_sec"].mean()
                if "time_to_first_word_sec" in g.columns
                else 0.0,
                "session_duration": g["visit_duration_sec"].iloc[0]
                if "visit_duration_sec" in g.columns
                else 0.0,
                "improvement": improvement,
                "last_completed": 1.0
                if g["level_status"].iloc[-1] == "completed"
                else 0.0,
                "first_word_rate": float(first_words) / max(float(first_total), 1),
                "hints_per_level": (
                    g["hints_used"].mean() if "hints_used" in g.columns else 0.0
                )
                / max(n, 1),
            }
        )

    first_visit_date = d.groupby("client_id")["date"].transform("min")
    first_session = d[d["date"] == first_visit_date]

    return (
        first_session.groupby("client_id")
        .apply(_agg_user, include_groups=False)
        .reset_index()
    )


def run_block1(df: pd.DataFrame) -> None:
    log.info("\n=== Block 1 — Session Quality Segmentation ===")

    if not {"client_id", "date", "level_status", "completion_pct"}.issubset(df.columns):
        log.warning("Missing required columns — skipping Block 1.")
        return

    features = _build_first_session_features(df)
    feat_cols = [
        c
        for c in (
            "n_levels",
            "completion_rate",
            "abandonment_rate",
            "avg_hints",
            "avg_time_first_word",
            "session_duration",
            "improvement",
            "last_completed",
            "first_word_rate",
            "hints_per_level",
        )
        if c in features.columns
    ]

    X_scaled = StandardScaler().fit_transform(features[feat_cols].fillna(0).values)

    # Choose k by silhouette score
    k_range = range(2, min(12, len(features) // 5 + 2))
    sil_scores = []
    for k in k_range:
        labels = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit_predict(
            X_scaled
        )
        try:
            sil_scores.append(silhouette_score(X_scaled, labels))
        except Exception:
            sil_scores.append(-1.0)

    best_k = list(k_range)[int(np.argmax(sil_scores))]
    log.info("Best k=%d  (silhouette=%.3f)", best_k, max(sil_scores))

    features["cluster"] = KMeans(
        n_clusters=best_k, random_state=RANDOM_STATE, n_init=10
    ).fit_predict(X_scaled)

    # Assign readable labels ordered by completion_rate descending
    labels_names = [
        "Engaged",
        "Active",
        "Casual",
        "Irregular",
        "Explorers",
        "Struggling",
        "Bounced",
        "One-and-Done",
    ]
    ranked = (
        features.groupby("cluster")[feat_cols]
        .mean()["completion_rate"]
        .sort_values(ascending=False)
        .index.tolist()
    )
    label_map = {cid: lbl for cid, lbl in zip(ranked, labels_names)}
    features["segment"] = features["cluster"].map(label_map)

    segment_profile = features.groupby("segment")[feat_cols].mean().round(3)
    segment_profile["user_count"] = features.groupby("segment").size()
    _save_csv(segment_profile.reset_index(), "block1_segment_profiles.csv")
    _save_csv(features[["client_id", "cluster", "segment"]], "block1_user_segments.csv")
    log.info("Segment sizes:\n%s", features["segment"].value_counts().to_string())

    # Plots
    colors = plt.cm.tab10.colors
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Блок 1 — Сегментация пользователей по качеству сессии", fontsize=14)

    # Silhouette curve
    axes[0].plot(list(k_range), sil_scores, "o-", color="steelblue")
    axes[0].axvline(best_k, color="tomato", ls="--", label=f"Оптимальное k={best_k}")
    axes[0].set_xlabel("Число кластеров (k)")
    axes[0].set_ylabel("Силуэтный коэффициент")
    axes[0].set_title("Выбор оптимального числа кластеров")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    # Segment sizes
    seg_counts = features["segment"].value_counts().reindex([s for s in labels_names])
    axes[1].barh(
        seg_counts.index, seg_counts.values, color=colors[: len(seg_counts)], alpha=0.85
    )
    axes[1].set_xlabel("Число пользователей")
    axes[1].set_title("Размеры сегментов")
    for i, (_, val) in enumerate(seg_counts.items()):
        axes[1].text(val + 0.3, i, str(val), va="center", fontsize=9)
    axes[1].grid(axis="x", alpha=0.3)

    plt.tight_layout()
    _save_figure("block1_segments.png")
    log.info("Block 1 complete.")


# BLOCK 2 — Level abandonment classifier


def _build_abandonment_features(df: pd.DataFrame):
    """
    One row per level attempt.
    Target: 1 = attempt did not result in completion, 0 = completed.
    """
    d = df.copy()
    d["target"] = (d["level_status"] != "completed").astype(int)

    d["session_period"] = d["hour_of_day"].apply(
        lambda h: hour_to_period(float(h)) if pd.notna(h) else "unknown"
    )
    if "date" in d.columns:
        d["is_weekend"] = d["date"].dt.dayofweek.isin([5, 6]).astype(int)

    # Rolling context within each session
    d = d.sort_values(["session_id", "level_seq"])
    d["prev_completion_pct"] = (
        d.groupby("session_id")["completion_pct"].shift(1).fillna(0)
    )
    d["prev_hints"] = d.groupby("session_id")["hints_used"].shift(1).fillna(0)
    d["attempts_so_far"] = d.groupby("session_id").cumcount()
    d["session_abandon_rate_so_far"] = (
        d.groupby("session_id")["target"]
        .transform(lambda x: x.shift(1).expanding().mean())
        .fillna(0)
    )

    num_cols = [
        "level_seq",
        "hints_used",
        "time_to_first_word_sec",
        "completion_pct",
        "duration_sec",
        "hour_of_day",
        "visit_count",
        "page_views",
        "prev_completion_pct",
        "prev_hints",
        "attempts_so_far",
        "session_abandon_rate_so_far",
    ]
    cat_cols = [
        "theme_letter",
        "device_category",
        "browser",
        "os",
        "ab_group",
        "session_period",
        "is_weekend",
    ]

    for col in num_cols:
        if col in d.columns:
            d[col] = winsorize(to_num(d[col]))
    for col in cat_cols:
        d[col] = (
            freq_encode(cap_rare(d[col].fillna("unknown").astype(str)))
            if col in d.columns
            else 0
        )

    feat_cols = [c for c in num_cols + cat_cols if c in d.columns]
    clean = d[feat_cols + ["target"]].dropna(subset=feat_cols)
    return clean[feat_cols].values, clean["target"].values, feat_cols


def _plot_shap_importance(model, feat_cols, X, color, title, ax):
    """Draw SHAP bar chart for a tree model on the given axes."""
    est = (
        model.named_steps["classifier"]
        if hasattr(model, "named_steps") and "classifier" in model.named_steps
        else model
    )
    try:
        sv = shap.TreeExplainer(est).shap_values(X)
        if isinstance(sv, list):
            sv = np.abs(np.array(sv)).mean(0)
        imp = np.abs(sv).mean(0)
        if imp.ndim > 1:
            imp = imp.mean(-1)
        xlabel = "|SHAP|"
    except Exception:
        imp = getattr(est, "feature_importances_", np.zeros(len(feat_cols)))
        xlabel = "Важность"

    top_idx = np.argsort(imp)[::-1][:12]
    ax.barh([feat_cols[i] for i in top_idx], imp[top_idx], color=color, alpha=0.85)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)


def run_block2(df: pd.DataFrame, tune: bool = False) -> None:
    log.info("\n=== Block 2 — Level Abandonment Classifier ===")

    if not {"session_id", "level_status", "level_seq", "completion_pct"}.issubset(
        df.columns
    ):
        log.warning("Missing required columns — skipping Block 2.")
        return

    X, y, feat_cols = _build_abandonment_features(df)
    n_abandoned = int(y.sum())
    n_total = len(y)
    log.info(
        "Attempts: %d  abandoned: %d (%.1f%%)",
        n_total,
        n_abandoned,
        100 * n_abandoned / n_total,
    )

    if n_total < 50 or n_abandoned == 0:
        log.warning("Insufficient data — skipping Block 2.")
        return

    scale_pos_weight = (n_total - n_abandoned) / max(n_abandoned, 1)
    models = _build_classifiers(
        scale_pos_weight,
        tune,
        X,
        y,
        block="block2",
        use_smote=False,
    )
    results = _evaluate_classifiers(models, X, y, "block2")
    model_names = list(models.keys())
    roc_data = _compute_roc_curves(models, X, y)

    best_name = results["F1"].idxmax()
    best_model = models[best_name]
    best_model.fit(X, y)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_DIR / "block2_abandonment_model.pkl")
    log.info("Best by F1: %s (%.3f) — model saved.", best_name, results["F1"].max())

    # Plots
    colors = list(plt.cm.tab10.colors)
    mean_fpr = np.linspace(0, 1, 100)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Блок 2 — Классификатор прерывания уровня", fontsize=14)

    # Panel 1: F1 + ROC-AUC bar chart
    ax = axes[0]
    x_pos = np.arange(len(model_names))
    w = 0.35
    f1_vals = results["F1"].values
    f1_err = results["F1±"].values
    auc_vals = np.array([roc_data[n][1] if n in roc_data else 0 for n in model_names])
    auc_err = np.array([roc_data[n][2] if n in roc_data else 0 for n in model_names])

    for bars, vals, errs, color, label in [
        (
            ax.bar(
                x_pos - w / 2,
                f1_vals,
                w,
                yerr=f1_err,
                capsize=4,
                color="steelblue",
                alpha=0.85,
                label="F1",
            ),
            f1_vals,
            f1_err,
            "steelblue",
            "F1",
        ),
        (
            ax.bar(
                x_pos + w / 2,
                auc_vals,
                w,
                yerr=auc_err,
                capsize=4,
                color="darkorange",
                alpha=0.85,
                label="ROC-AUC",
            ),
            auc_vals,
            auc_err,
            "darkorange",
            "ROC-AUC",
        ),
    ]:
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, rotation=15, ha="right", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.axhline(0.5, color="gray", lw=0.8, ls="--")
    ax.set_title("F1 и ROC-AUC (5-кратная кросс-валидация)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Panel 2: ROC curves
    ax = axes[1]
    for (name, (tpr, mean_auc, std_auc)), color in zip(roc_data.items(), colors):
        ax.plot(
            mean_fpr,
            tpr,
            color=color,
            lw=2,
            label=f"{name} AUC={mean_auc:.2f}±{std_auc:.2f}",
        )
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("Доля ложноположительных")
    ax.set_ylabel("Доля истинноположительных")
    ax.set_title("ROC-кривые")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 3: SHAP importance for best tree model
    shap_model_key = results["F1"].idxmax()
    if shap_model_key:
        models[shap_model_key].fit(X, y)
        _plot_shap_importance(
            models[shap_model_key],
            feat_cols,
            X,
            color="darkorange",
            title=f"Важность признаков",
            ax=axes[2],
        )

    plt.tight_layout()
    _save_figure("block2_results.png")
    log.info("Block 2 complete.")


# BLOCK 3 — Return classifier


def _build_return_features(df: pd.DataFrame):
    """
    One row per user.
    Target: 1 = user returned for a second session, 0 = did not.
    All features come from the first session only.
    """
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")

    # Label: any session after the first counts as a return
    n_sessions = d.groupby("client_id")["date"].nunique()
    first_date = d.groupby("client_id")["date"].min().rename("first_date")
    user_meta = pd.concat([first_date, n_sessions.rename("n_sessions")], axis=1)
    user_meta["returned"] = (user_meta["n_sessions"] > 1).astype(int)

    # First-session rows only
    first_session = d[
        d["date"] == d.groupby("client_id")["date"].transform("min")
    ].copy()

    def _agg_user(g):
        g = g.sort_values("level_seq") if "level_seq" in g.columns else g
        n = len(g)
        comp_rate = (g["level_status"] == "completed").mean()

        cp = g["completion_pct"].dropna()
        improvement = (
            float(np.polyfit(range(len(cp)), cp, 1)[0]) if len(cp) >= 2 else 0.0
        )

        fw = g["words_found"].iloc[0] if "words_found" in g.columns else 0
        ft = g["words_total"].iloc[0] if "words_total" in g.columns else 1

        ab = g["ab_group"].iloc[0] if "ab_group" in g.columns else "unknown"
        ab_enc = 1 if ab == "B" else 0
        ts = g["date"].iloc[0]

        return pd.Series(
            {
                # Volume
                "n_levels": n,
                "session_duration": g["visit_duration_sec"].iloc[0]
                if "visit_duration_sec" in g.columns
                else 0.0,
                "page_views": g["page_views"].iloc[0]
                if "page_views" in g.columns
                else 0,
                # Performance
                "completion_rate": comp_rate,
                "abandonment_rate": (g["level_status"] == "abandoned").mean(),
                "avg_hints": g["hints_used"].mean(),
                "hints_per_level": g["hints_used"].mean() / max(n, 1),
                "avg_time_first_word": g["time_to_first_word_sec"].mean(),
                # Trajectory
                "improvement": improvement,
                "last_completed": 1.0
                if g["level_status"].iloc[-1] == "completed"
                else 0.0,
                "first_word_rate": float(fw) / max(float(ft), 1),
                "n_themes": g["theme_letter"].nunique()
                if "theme_letter" in g.columns
                else 1,
                # Engagement composite
                "engagement_score": comp_rate
                * max(
                    float(
                        g["visit_duration_sec"].iloc[0]
                        if "visit_duration_sec" in g.columns
                        else 0
                    ),
                    0,
                ),
                # Temporal context
                "hour_of_day": g["hour_of_day"].iloc[0]
                if "hour_of_day" in g.columns
                else 12.0,
                "session_period_enc": {
                    "night": 0,
                    "morning": 1,
                    "afternoon": 2,
                    "evening": 3,
                }.get(
                    hour_to_period(
                        float(
                            g["hour_of_day"].iloc[0]
                            if "hour_of_day" in g.columns
                            else 12.0
                        )
                    ),
                    1,
                ),
                "day_of_week": ts.day_name() if pd.notna(ts) else "Unknown",
                "is_weekend": int(ts.dayofweek in (5, 6)) if pd.notna(ts) else 0,
                # Acquisition / experiment
                "ab_group": ab,
                "ab_group_enc": ab_enc,
                "ab_x_completion": ab_enc * comp_rate,  # interaction term
                "device_category": g["device_category"].iloc[0]
                if "device_category" in g.columns
                else "unknown",
                "region": g["region"].iloc[0] if "region" in g.columns else "unknown",
                "utm_source": g["utm_source"].iloc[0]
                if "utm_source" in g.columns
                else "unknown",
                "traffic_source": g["traffic_source"].iloc[0]
                if "traffic_source" in g.columns
                else "unknown",
                "visit_count": g["visit_count"].iloc[0]
                if "visit_count" in g.columns
                else 1,
            }
        )

    users = (
        first_session.groupby("client_id")
        .apply(_agg_user, include_groups=False)
        .reset_index()
        .merge(user_meta[["returned"]].reset_index(), on="client_id")
    )
    return users


def _engineer_return_features(users: pd.DataFrame):
    d = users.copy()

    num_cols = [
        "n_levels",
        "session_duration",
        "page_views",
        "completion_rate",
        "abandonment_rate",
        "avg_hints",
        "hints_per_level",
        "avg_time_first_word",
        "improvement",
        "last_completed",
        "first_word_rate",
        "n_themes",
        "engagement_score",
        "hour_of_day",
        "session_period_enc",
        "is_weekend",
        "visit_count",
        "ab_group_enc",
        "ab_x_completion",
    ]
    cat_cols = [
        "day_of_week",
        "ab_group",
        "device_category",
        "region",
        "utm_source",
        "traffic_source",
    ]

    for col in num_cols:
        if col in d.columns:
            d[col] = winsorize(to_num(d[col]))
    for col in cat_cols:
        d[col] = (
            freq_encode(cap_rare(d[col].fillna("unknown").astype(str)))
            if col in d.columns
            else 0
        )

    feat_cols = [c for c in num_cols + cat_cols if c in d.columns]
    return d[feat_cols].values, d["returned"].values, feat_cols


def run_block3(df: pd.DataFrame, tune: bool = False) -> None:
    log.info("\n=== Block 3 — Return Classifier ===")

    if not {"client_id", "date", "level_status", "completion_pct"}.issubset(df.columns):
        log.warning("Missing required columns — skipping Block 3.")
        return

    users = _build_return_features(df)
    if users is None or len(users) == 0:
        log.warning("No user data available — skipping Block 3.")
        return

    n_returned = int(users["returned"].sum())
    n_total = len(users)
    log.info(
        "Users: %d  returned: %d (%.1f%%)",
        n_total,
        n_returned,
        100 * n_returned / n_total if n_total else 0,
    )

    if n_total < 30:
        log.warning("Too few users — skipping Block 3.")
        return
    if n_returned == 0 or n_returned == n_total:
        log.warning("Only one class present — skipping Block 3.")
        return

    imbalance = (n_total - n_returned) / max(n_returned, 1)
    if imbalance > 5:
        log.warning("Class imbalance %.1f:1.", imbalance)

    X, y, feat_cols = _engineer_return_features(users)

    scale_pos_weight = (n_total - n_returned) / max(n_returned, 1)
    models = _build_classifiers(
        scale_pos_weight,
        tune,
        X,
        y,
        block="block3",
        use_smote=True,
    )
    results = _evaluate_classifiers(models, X, y, "block3")
    model_names = list(models.keys())
    roc_data = _compute_roc_curves(models, X, y)

    best_name = results["F1"].idxmax()
    best_model = models[best_name]
    best_model.fit(X, y)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_DIR / "block3_return_model.pkl")
    log.info("Best by F1: %s (%.3f) — model saved.", best_name, results["F1"].max())

    auc_lift = results.loc[best_name, "ROC-AUC"] - 0.5
    log.info("AUC lift over random baseline: +%.3f", auc_lift)
    if auc_lift < 0.05:
        log.warning("AUC lift is very small — features may be uninformative.")

    # Plots
    colors = list(plt.cm.tab10.colors)
    mean_fpr = np.linspace(0, 1, 100)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Блок 3 — Классификатор возврата пользователей", fontsize=14)

    # Panel 1: ROC-AUC + Recall
    ax = axes[0]
    x_pos = np.arange(len(model_names))
    w = 0.35
    recall_vals = results["Recall"].values
    auc_vals    = np.array([roc_data[n][1] if n in roc_data else 0 for n in model_names])
    auc_err     = np.array([roc_data[n][2] if n in roc_data else 0 for n in model_names])

    b1 = ax.bar(x_pos - w/2, recall_vals, w, capsize=4,
                color="steelblue", alpha=0.85, label="Recall")
    b2 = ax.bar(x_pos + w/2, auc_vals, w, yerr=auc_err, capsize=4,
                color="mediumpurple", alpha=0.85, label="ROC-AUC")

    for bar, val in list(zip(b1, recall_vals)) + list(zip(b2, auc_vals)):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(model_names, rotation=15, ha="right", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.axhline(0.5, color="gray", lw=0.8, ls="--")
    ax.set_title("Recall и ROC-AUC (5-кратная кросс-валидация)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Panel 2: ROC curves
    ax = axes[1]
    for (name, (tpr, mean_auc, std_auc)), color in zip(roc_data.items(), colors):
        ax.plot(
            mean_fpr,
            tpr,
            color=color,
            lw=2,
            label=f"{name} AUC={mean_auc:.2f}±{std_auc:.2f}",
        )
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("Доля ложноположительных")
    ax.set_ylabel("Доля истинноположительных")
    ax.set_title("ROC-кривые")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 3: Pearson correlation with return label
    ax = axes[2]
    try:
        corr = (
            pd.DataFrame(X, columns=feat_cols)
            .assign(returned=y)
            .corr()["returned"]
            .drop("returned")
            .sort_values(key=abs, ascending=False)
        )
        _save_csv(
            corr.reset_index().rename(
                columns={"index": "feature", "returned": "pearson_r"}
            ),
            "block3_feature_correlations.csv",
        )
        top = corr.head(15)
        bar_colors = ["steelblue" if v > 0 else "tomato" for v in top[::-1]]
        ax.barh(top.index[::-1], top.values[::-1], color=bar_colors, alpha=0.85)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("Корреляция Пирсона с целевой переменной")
        ax.set_title("Корреляция признаков с целевой переменной")
        ax.grid(axis="x", alpha=0.3)
    except Exception as exc:
        log.warning("Correlation plot failed: %s", exc)
        axes[2].set_visible(False)

    plt.tight_layout()
    _save_figure("block3_results.png")
    log.info("Block 3 complete.")


# A/B Test Analysis


def _cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta: P(B > A) − P(B < A)."""
    n_a, n_b = len(a), len(b)
    greater = sum(1 for ai in a for bj in b if bj > ai)
    less = sum(1 for ai in a for bj in b if bj < ai)
    return (greater - less) / (n_a * n_b)


def _cliffs_delta_ztest(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """
    Two-sided z-test for Cliff's delta (normal approximation).
    Applies a continuity correction when n_A × n_B < 100.
    Returns (delta, p_value).
    """
    n_a, n_b = len(a), len(b)
    delta = _cliffs_delta(a, b)

    if n_a * n_b < 100:
        correction = 1.0 / (n_a * n_b)
        delta = delta - correction if delta > 0 else delta + correction

    var = (n_a + n_b + 1) / (3 * n_a * n_b)
    if var <= 0:
        return delta, 1.0

    p = 2 * norm.sf(abs(delta / np.sqrt(var)))
    return delta, p


def _required_n_cliffs(delta: float, alpha: float, power: float) -> int:
    """Required n per group for Cliff's delta z-test."""
    z_a = norm.ppf(1 - alpha / 2)
    z_b = norm.ppf(power)
    return int(np.ceil(2 / 3 * ((z_a + z_b) / delta) ** 2))


def _required_n_proportions(c_a: float, c_b: float, alpha: float, power: float) -> int:
    """Required n per group for a two-proportion z-test."""
    denom = np.sqrt((c_a + c_b) * (2 - c_a - c_b))
    if denom == 0 or c_a == c_b:
        return 9999
    delta = (c_b - c_a) / denom
    z_a = norm.ppf(1 - alpha / 2)
    z_b = norm.ppf(power)
    return int(np.ceil(((z_a + z_b) / delta) ** 2))


def run_ab_analysis(
    df: pd.DataFrame,
    alpha: float = 0.01,
    power: float = 0.80,
    n_pilot_users: int = 30,
) -> None:
    """
    A/B analysis with a pilot / main split.

    The first n_pilot_users (ordered by first visit date) form the pilot sample
    used to estimate Cliff's delta and the required sample size.
    Hypothesis testing is performed on the remaining users only, preventing
    inflation of Type I error from using the same data twice.
    """
    log.info("\n=== A/B Test Analysis (pilot n=%d) ===", n_pilot_users)

    if "ab_group" not in df.columns:
        log.warning("No ab_group column — skipping A/B analysis.")
        return

    d = df[df["ab_group"].isin(["A", "B"])].copy()
    if "client_id" not in d.columns or "date" not in d.columns:
        log.warning("client_id or date column missing — skipping.")
        return

    # Split by first visit date
    user_order = (
        d.groupby("client_id")["date"]
        .min()
        .sort_values()
        .reset_index()
        .rename(columns={"date": "first_date"})
    )
    # pilot_ids = set(user_order.head(n_pilot_users)["client_id"])
    # main_ids = set(user_order.iloc[n_pilot_users:]["client_id"])
    pilot_ids = set(user_order.iloc[40 : 40 + n_pilot_users]["client_id"])
    main_ids = set(user_order[~user_order["client_id"].isin(pilot_ids)]["client_id"])

    d_pilot = d[d["client_id"].isin(pilot_ids)].copy()
    d_main = d[d["client_id"].isin(main_ids)].copy()

    log.info(
        "Pilot: %d users (%d rows) | Main: %d users (%d rows)",
        len(pilot_ids),
        len(d_pilot),
        len(main_ids),
        len(d_main),
    )

    if len(d_main) < 10:
        log.warning("Too few main-experiment rows — skipping.")
        return

    # Per-session aggregates
    def _session_agg(data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        data["completed"] = (data["level_status"] == "completed").astype(float)
        return (
            data.groupby(["ab_group", "session_id"])
            .agg(
                levels_played=("level_seq", "count"),
                completion_rate=("completed", "mean"),
                avg_hints=("hints_used", "mean"),
                avg_completion_pct=("completion_pct", "mean"),
            )
            .reset_index()
        )

    sess_pilot = _session_agg(d_pilot)
    sess_main = _session_agg(d_main)

    metric_cols = {
        "levels_played": "Уровней за сессию",
        "completion_rate": "Доля завершённых уровней",
        "avg_hints": "Среднее кол-во подсказок",
        "avg_completion_pct": "Средний % завершения уровня",
    }

    n_A_main = int(d_main[d_main["ab_group"] == "A"]["client_id"].nunique())
    n_B_main = int(d_main[d_main["ab_group"] == "B"]["client_id"].nunique())

    # Pilot: estimate effect size and required N
    log.info("\n  --- Pilot experiment ---")
    pilot_rows = []
    sample_size_rows = []

    for col, label in metric_cols.items():
        ga = sess_pilot[sess_pilot["ab_group"] == "A"][col].dropna().values
        gb = sess_pilot[sess_pilot["ab_group"] == "B"][col].dropna().values
        if len(ga) < 3 or len(gb) < 3:
            continue
        delta = _cliffs_delta(ga, gb)
        pilot_rows.append(
            {
                "metric": label,
                "pilot_delta": round(delta, 4),
                "n_A": len(ga),
                "n_B": len(gb),
            }
        )
        if abs(delta) > 0:
            n_needed = _required_n_cliffs(abs(delta), alpha, power)
            sample_size_rows.append(
                {
                    "metric": label,
                    "pilot_delta": round(delta, 4),
                    "n_per_group_needed": n_needed,
                    "n_total_needed": n_needed * 2,
                    "n_A_main": n_A_main,
                    "n_B_main": n_B_main,
                    "powered": "yes" if min(n_A_main, n_B_main) >= n_needed else "no",
                }
            )
        log.info(
            "%-30s  δ=%.4f  n_needed=%d",
            label,
            delta,
            _required_n_cliffs(abs(delta), alpha, power) if abs(delta) > 0 else 9999,
        )

    if pilot_rows:
        _save_csv(pd.DataFrame(pilot_rows), "ab_pilot_deltas.csv")
    if sample_size_rows:
        ss = pd.DataFrame(sample_size_rows)
        _save_csv(ss, "ab_sample_size_pilot.csv")
        log.info("\n%s", ss.to_string(index=False))

    # Main experiment: hypothesis testing
    log.info("\n  --- Main experiment ---")
    log.info("Group A = %d users, Group B = %d users", n_A_main, n_B_main)
    result_rows = []
    sample_size_main = []

    for col, label in metric_cols.items():
        ga = sess_main[sess_main["ab_group"] == "A"][col].dropna().values
        gb = sess_main[sess_main["ab_group"] == "B"][col].dropna().values
        if len(ga) < 3 or len(gb) < 3:
            continue

        delta, p = _cliffs_delta_ztest(ga, gb)
        result_rows.append(
            {
                "metric": label,
                "test": "z-test (Cliff's delta)",
                "A_mean": round(float(np.mean(ga)), 4),
                "B_mean": round(float(np.mean(gb)), 4),
                "A_median": round(float(np.median(ga)), 4),
                "B_median": round(float(np.median(gb)), 4),
                "cliffs_delta": round(delta, 4),
                "p_value": round(p, 4),
                "n_A": len(ga),
                "n_B": len(gb),
                "significant": "yes" if p < alpha else "no",
            }
        )
        if abs(delta) > 0:
            n_needed = _required_n_cliffs(abs(delta), alpha, power)
            sample_size_main.append(
                {
                    "metric": label,
                    "observed_delta": round(delta, 4),
                    "n_per_group_needed": n_needed,
                    "n_total_needed": n_needed * 2,
                    "n_A_actual": len(ga),
                    "n_B_actual": len(gb),
                    "powered": "yes" if min(len(ga), len(gb)) >= n_needed else "no",
                }
            )

    # Retention (proportion z-test)
    max_date = d_main["date"].max()
    first_main = (
        d_main.groupby("client_id")
        .agg(first_date=("date", "min"), ab_group=("ab_group", "first"))
        .join(d_main.groupby("client_id")["date"].max().rename("last_date"))
        .reset_index()
    )
    first_main = first_main[first_main["first_date"] <= max_date - pd.Timedelta(days=7)]
    first_main["returned_again"] = (
        (first_main["last_date"] - first_main["first_date"]).dt.days.between(1, 7)
    ).astype(int)

    for grp in ("A", "B"):
        sub = first_main[first_main["ab_group"] == grp]
        log.info(
            "Group %s retention: %d/%d (%.1f%%)",
            grp,
            sub["returned_again"].sum(),
            len(sub),
            100 * sub["returned_again"].mean() if len(sub) else 0,
        )

    ret_a = first_main[first_main["ab_group"] == "A"]["returned_again"].mean()
    ret_b = first_main[first_main["ab_group"] == "B"]["returned_again"].mean()
    n_a_ret = int((first_main["ab_group"] == "A").sum())
    n_b_ret = int((first_main["ab_group"] == "B").sum())

    denom = np.sqrt((ret_a + ret_b) * (2 - ret_a - ret_b))
    if denom > 0 and n_a_ret > 0 and n_b_ret > 0:
        delta_ret = (ret_b - ret_a) / denom
        var_ret = 0.25 * (1 / n_a_ret + 1 / n_b_ret)
        p_ret = 2 * norm.sf(abs(delta_ret / np.sqrt(var_ret)))
    else:
        delta_ret, p_ret = 0.0, 1.0

    result_rows.append(
        {
            "metric": "Retention rate",
            "test": "z-test (proportions)",
            "A_mean": round(float(ret_a), 4),
            "B_mean": round(float(ret_b), 4),
            "A_median": "-",
            "B_median": "-",
            "cliffs_delta": round(delta_ret, 4),
            "p_value": round(p_ret, 4),
            "n_A": n_a_ret,
            "n_B": n_b_ret,
            "significant": "yes" if p_ret < alpha else "no",
        }
    )
    n_needed_ret = _required_n_proportions(float(ret_a), float(ret_b), alpha, power)
    sample_size_main.append(
        {
            "metric": "Retention rate",
            "observed_delta": round(delta_ret, 4),
            "n_per_group_needed": n_needed_ret,
            "n_total_needed": n_needed_ret * 2,
            "n_A_actual": n_a_ret,
            "n_B_actual": n_b_ret,
            "powered": "yes" if min(n_a_ret, n_b_ret) >= n_needed_ret else "no",
        }
    )

    if not result_rows:
        log.warning("No metrics computed.")
        return

    results_df = pd.DataFrame(result_rows)
    _save_csv(results_df, "ab_test_results.csv")
    log.info(
        "\n%s",
        results_df[
            ["metric", "A_mean", "B_mean", "cliffs_delta", "p_value", "significant"]
        ].to_string(index=False),
    )

    if sample_size_main:
        ss_main = pd.DataFrame(sample_size_main)
        _save_csv(ss_main, "ab_sample_size_main.csv")
        log.info("\n%s", ss_main.to_string(index=False))

    # Plots
    def _sig_label(p_val):
        if p_val < 0.001:
            return "***"
        if p_val < 0.01:
            return "**"
        if p_val < 0.05:
            return "*"
        return "н.з."

    group_colors = {"A": "#4C72B0", "B": "#DD8452"}
    session_rows = [r for r in result_rows if r["metric"] != "Retention rate"]
    retention_row = next(
        (r for r in result_rows if r["metric"] == "Retention rate"), None
    )
    n_panels = len(session_rows) + (1 if retention_row else 0)

    fig, axes_grid = plt.subplots(2, 3, figsize=(15, 12))
    axes_flat = axes_grid.flatten()
    for i in range(n_panels, len(axes_flat)):
        axes_flat[i].set_visible(False)

    for ax, row in zip(axes_flat[: len(session_rows)], session_rows):
        col = next(c for c, l in metric_cols.items() if l == row["metric"])
        ga = sess_main[sess_main["ab_group"] == "A"][col].dropna()
        gb = sess_main[sess_main["ab_group"] == "B"][col].dropna()

        plot_df = pd.DataFrame(
            {
                "value": pd.concat([ga, gb], ignore_index=True),
                "group": ["A"] * len(ga) + ["B"] * len(gb),
            }
        )
        sns.violinplot(
            data=plot_df,
            x="group",
            y="value",
            ax=ax,
            palette=group_colors,
            inner=None,
            alpha=0.45,
            cut=0,
        )
        sns.stripplot(
            data=plot_df,
            x="group",
            y="value",
            ax=ax,
            palette=group_colors,
            size=4,
            jitter=True,
            alpha=0.7,
        )
        for i_grp, gdata in enumerate([ga, gb]):
            ax.plot(
                i_grp,
                gdata.mean(),
                marker="D",
                color="white",
                markeredgecolor="black",
                markersize=7,
                zorder=5,
            )

        p_val = row["p_value"]
        delta = row["cliffs_delta"]
        y_line = max(ga.max(), gb.max()) * 1.18
        ax.plot(
            [0, 0, 1, 1],
            [y_line, y_line * 1.03, y_line * 1.03, y_line],
            lw=1.2,
            color="black",
        )
        ax.text(
            0.5,
            y_line * 1.05,
            f"{_sig_label(p_val)}  p={p_val:.4f}  δ={delta:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
        ax.set_ylim(top=y_line * 1.18)
        ax.set_xticklabels([f"A", f"B"])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title(row["metric"], fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    if retention_row:
        ax = axes_flat[len(session_rows)]
        vals = [retention_row["A_mean"] * 100, retention_row["B_mean"] * 100]
        bars = ax.bar(
            ["A", "B"],
            vals,
            color=[group_colors["A"], group_colors["B"]],
            alpha=0.8,
            width=0.5,
        )
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 0.3,
                f"{v:.1f}%",
                ha="center",
                va="bottom",
                fontsize=10,
            )
        p_val = retention_row["p_value"]
        ax.set_title(
            f"Удержание\np={p_val:.4f}  {_sig_label(p_val)}"
            f"  δ={retention_row['cliffs_delta']:.3f}",
            fontsize=10,
        )
        ax.set_ylabel("Удержание, %")
        ax.set_ylim(0, max(vals) * 2 if max(vals) > 0 else 10)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"A/B Тест — критерий Клиффа (α={alpha}, мощность={power})", fontsize=13, y=1.01
    )
    plt.tight_layout()
    _save_figure("ab_test_comparison.png")
    log.info("A/B analysis complete.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Word Search Game — behavioural analytics pipeline"
    )
    parser.add_argument("file", nargs="?", help="TSV input file")
    parser.add_argument("--from", dest="from_date", metavar="YYYY-MM-DD")
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run RandomizedSearchCV hyperparameter tuning",
    )
    parser.add_argument(
        "--blocks",
        nargs="+",
        type=int,
        default=[1, 2, 3, 4],
        help="Blocks to run, e.g. --blocks 1 3",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    paths = [Path(args.file)] if args.file else _find_latest_tsv()
    df = load_raw(paths)

    if args.from_date:
        cutoff = pd.Timestamp(args.from_date)
        before = len(df)
        df = df[df["date"] >= cutoff].reset_index(drop=True)
        log.info(
            "Filtered to %d rows (from %s, dropped %d)",
            len(df),
            args.from_date,
            before - len(df),
        )

    if 1 in args.blocks:
        run_block0(df)
    if 2 in args.blocks:
        run_block1(df)
    if 3 in args.blocks:
        run_block2(df, tune=args.tune)
    if 4 in args.blocks:
        run_block3(df, tune=args.tune)

    run_ab_analysis(df)
    log.info("\nDone. Outputs in %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
