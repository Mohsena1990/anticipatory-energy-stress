"""
ml_classification.py
─────────────────────
CatBoost binary classifier for High Adaptive Energy Vulnerability (HighAEV).

Target
──────
  high_aev = 1  if aev_score ≥ 75th percentile
            = 0  otherwise

This replaces the old 'high_discomfort' target.  The AEV composite
correctly operationalises energy-poverty vulnerability in the UK ENABLE
sub-sample using validated H15 transition-attitude items and E7 habit
items, rather than the mislabelled H12/H15 'thermal discomfort' proxy.

Leakage prevention
──────────────────
  • aev_score and its direct components are EXCLUDED from features
  • FES variables are EXCLUDED (constant within UK sample → zero variance)
  • Raw item scores that are also construct components are only included
    when using a separate 'item-level generalisable' model variant

Feature groups
──────────────
  1. Construct composite scores (non-AEV): fcp_score, aemc_score,
     bli_score, tcr_score
  2. Energy-poverty proxies: low_income_flag, high_cost_flag, risk_category
  3. Household controls: H1, H2, H3, S2, S3, S5, S6, S8
  4. Energy-related household variables: H5, H6, H13

Usage
─────
  from src.ml_classification import run
  model, results = run(df)   # df from enable_preprocessing.run()
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from src import config, paths

log = logging.getLogger(__name__)

# Features that directly construct or leak the AEV target
_LEAKAGE_COLS: set[str] = {
    "aev_score",
    "high_aev",
    "fcp_score",   # only exclude if testing generalisable model without construct scores
    "aemc_score",
    "bli_score",
    "tcr_score",
}

# FES is constant across UK sample — excluded from ML features
_FES_COLS: set[str] = {"fes_core", "fes_macro", "fes_actual"}

# Preferred feature order (subset will be selected by availability)
_PREFERRED_FEATURES = [
    # Construct-level scores (theoretically motivated)
    "fcp_score", "aemc_score", "bli_score", "tcr_score",
    # Energy-poverty proxies
    "risk_category", "low_income_flag", "high_cost_flag",
    # Household controls
    "S8", "H1", "H2", "H3", "S2", "S3", "S5", "S6",
    # Energy-related household variables
    "H5", "H6", "H13",
]


# =============================================================================
# I/O helpers
# =============================================================================

def _save_csv(df: pd.DataFrame, name: str) -> None:
    paths.ML_TABLES.mkdir(parents=True, exist_ok=True)
    p = paths.ML_TABLES / f"{name}.csv"
    df.to_csv(p, index=False)
    log.info("Saved %s", p.name)


def _save_fig(fig: plt.Figure, name: str) -> None:
    paths.ML_FIGURES.mkdir(parents=True, exist_ok=True)
    p = paths.ML_FIGURES / f"{name}.{config.FIGURE_FORMAT}"
    fig.savefig(p, dpi=config.DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved figure %s", p.name)


# =============================================================================
# Feature preparation
# =============================================================================

def prepare_features(
    df: pd.DataFrame,
    include_construct_scores: bool = True,
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """
    Build feature matrix X and binary target y (high_aev).

    Parameters
    ----------
    include_construct_scores : bool
        If True (default), include FCP/AEMC/BLI/TCR composite scores.
        Set False to test pure item-level prediction (requires separate
        leakage audit).

    Returns (X, y, feature_cols, categorical_cols)
    """
    if "high_aev" not in df.columns:
        raise ValueError("high_aev target missing — run enable_preprocessing first.")

    target_col = "high_aev"
    n_before   = len(df)
    df_clean   = df.dropna(subset=[target_col]).copy().reset_index(drop=True)
    n_dropped  = n_before - len(df_clean)
    if n_dropped > 0:
        log.info("Dropped %d rows with missing target", n_dropped)

    # Build candidate feature list
    leakage = _LEAKAGE_COLS.copy()
    if not include_construct_scores:
        # When construct scores are excluded, item-level n_ columns
        # for non-AEV items can be used — but requires separate leakage check
        leakage.update({"fcp_score", "aemc_score", "bli_score", "tcr_score"})

    candidate = [
        c for c in _PREFERRED_FEATURES
        if c in df_clean.columns
        and c not in leakage
        and c not in _FES_COLS
        and c != target_col
    ]

    # Deduplicate preserving order
    feature_cols = list(dict.fromkeys(candidate))

    X = df_clean[feature_cols].copy()
    y = df_clean[target_col].copy()

    # Bool → int
    for c in X.select_dtypes(include="bool").columns:
        X[c] = X[c].astype(int)

    # Numeric NaN → median
    for c in X.select_dtypes(include=[np.number]).columns:
        if X[c].isna().any():
            X[c] = X[c].fillna(X[c].median())

    # Categorical NaN → "unknown"
    for c in X.select_dtypes(include="object").columns:
        X[c] = X[c].fillna("unknown")

    categorical_cols = list(X.select_dtypes(include="object").columns)
    log.info("Features: %d total  |  %d categorical  |  target balance: %.1f%%",
             len(feature_cols), len(categorical_cols),
             100.0 * y.mean())

    _save_csv(
        pd.DataFrame({"feature": feature_cols,
                      "is_categorical": [c in categorical_cols for c in feature_cols]}),
        "ml_feature_list",
    )
    return X, y, feature_cols, categorical_cols


# =============================================================================
# CatBoost training
# =============================================================================

def train_catboost(
    X: pd.DataFrame,
    y: pd.Series,
    categorical_cols: list[str],
    test_size: float = 0.25,
    seed: int = None,
):
    """
    Fit CatBoost binary classifier (HighAEV target) with early stopping.

    Class imbalance is handled via inverse-frequency class weighting.
    Returns model + train/test split data for downstream evaluation.
    """
    try:
        from catboost import CatBoostClassifier, Pool
    except ImportError:
        raise ImportError("Install CatBoost: pip install catboost")

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score,
        precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score,
        confusion_matrix, classification_report,
    )

    if seed is None:
        seed = config.RANDOM_SEED

    cat_idx = [X.columns.get_loc(c) for c in categorical_cols if c in X.columns]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    log.info("Train=%d  Test=%d", len(y_tr), len(y_te))

    pos_weight = max(1.0, (y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    log.info("Class weight: pos_class=%.2f", pos_weight)

    model = CatBoostClassifier(
        iterations=600,
        learning_rate=0.03,
        depth=4,
        loss_function="Logloss",
        eval_metric="AUC",
        class_weights=[1.0, pos_weight],
        random_seed=seed,
        verbose=False,
        early_stopping_rounds=60,
    )

    train_pool = Pool(X_tr, y_tr, cat_features=cat_idx)
    test_pool  = Pool(X_te, y_te, cat_features=cat_idx)
    model.fit(train_pool, eval_set=test_pool)

    pred = model.predict(X_te)
    prob = model.predict_proba(X_te)[:, 1]

    metrics = {
        "accuracy":           round(accuracy_score(y_te, pred),           4),
        "balanced_accuracy":  round(balanced_accuracy_score(y_te, pred),  4),
        "precision":          round(precision_score(y_te, pred, zero_division=0), 4),
        "recall":             round(recall_score(y_te, pred, zero_division=0),    4),
        "f1":                 round(f1_score(y_te, pred, zero_division=0),        4),
        "roc_auc":            round(roc_auc_score(y_te, prob),                   4),
        "pr_auc":             round(average_precision_score(y_te, prob),          4),
        "n_train":            int(len(y_tr)),
        "n_test":             int(len(y_te)),
        "best_iteration":     int(model.get_best_iteration() or 0),
        "pos_class_weight":   round(pos_weight, 3),
        "target":             "high_aev",
    }

    _save_csv(pd.DataFrame([metrics]), "catboost_performance")

    cm = confusion_matrix(y_te, pred)
    _save_csv(
        pd.DataFrame(cm,
                     index=["actual_low_aev", "actual_high_aev"],
                     columns=["pred_low_aev", "pred_high_aev"]),
        "confusion_matrix",
    )

    report = classification_report(y_te, pred, output_dict=True)
    _save_csv(pd.DataFrame(report).T.reset_index(names=["class"]),
              "classification_report")

    for k, v in metrics.items():
        if isinstance(v, float):
            log.info("  %-22s: %.4f", k, v)

    return model, X_tr, X_te, y_tr, y_te, pred, prob, cat_idx


# =============================================================================
# Evaluation figures
# =============================================================================

def plot_confusion_matrix(
    y_test: pd.Series,
    y_pred: np.ndarray,
) -> None:
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred: Low AEV", "Pred: High AEV"])
    ax.set_yticklabels(["True: Low AEV", "True: High AEV"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=14, color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im, ax=ax)
    ax.set_title("Confusion matrix: High Adaptive Energy Vulnerability")
    fig.tight_layout()
    _save_fig(fig, "confusion_matrix")


def plot_roc_curve(y_test: pd.Series, y_prob: np.ndarray) -> None:
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score   = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, lw=2, color="#2980B9", label=f"AUC = {auc_score:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curve: High Adaptive Energy Vulnerability")
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save_fig(fig, "roc_curve")


def plot_pr_curve(y_test: pd.Series, y_prob: np.ndarray) -> None:
    from sklearn.metrics import precision_recall_curve, average_precision_score
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    ap           = average_precision_score(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(rec, prec, lw=2, color="#27AE60", label=f"AP = {ap:.3f}")
    ax.axhline(y_test.mean(), color="grey", ls="--", lw=1, label="Baseline")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall curve: High Adaptive Energy Vulnerability")
    ax.legend()
    fig.tight_layout()
    _save_fig(fig, "pr_curve")


def plot_risk_distribution(
    y_test: pd.Series,
    y_prob: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, color, ls in [(0, "#2980B9", "-"), (1, "#E74C3C", "--")]:
        mask = y_test == label
        if mask.sum() > 0:
            ax.hist(y_prob[mask], bins=30, alpha=0.6,
                    color=color, ls=ls,
                    label=f"True {'High' if label else 'Low'} AEV (n={mask.sum()})",
                    density=True)
    ax.set_xlabel("Predicted probability of High AEV")
    ax.set_ylabel("Density")
    ax.set_title("Predicted risk distribution by true AEV class")
    ax.legend()
    fig.tight_layout()
    _save_fig(fig, "predicted_risk_distribution")


# =============================================================================
# Full-sample predictions
# =============================================================================

def save_full_predictions(
    df: pd.DataFrame,
    X_full: pd.DataFrame,
    model,
    cat_idx: list[int],
) -> None:
    """Save full-sample predictions + probabilities to CSV."""
    try:
        from catboost import Pool
        pool = Pool(X_full, cat_features=cat_idx)
        preds = model.predict(pool)
        probs = model.predict_proba(pool)[:, 1]
    except Exception as e:
        log.warning("Could not predict full sample: %s", e)
        return

    out = df[["respondent_id"] if "respondent_id" in df.columns else []].copy()
    if "respondent_id" not in out.columns:
        out["respondent_id"] = np.arange(1, len(df) + 1)
    out["aev_score"]        = df.get("aev_score")
    out["high_aev_actual"]  = df.get("high_aev")
    out["high_aev_pred"]    = preds
    out["high_aev_prob"]    = probs
    for fes_col in ["fes_core", "fes_macro", "fes_actual"]:
        if fes_col in df.columns:
            out[fes_col] = df[fes_col]
    out.to_csv(paths.ML_PREDS, index=False)
    log.info("Saved full-sample predictions: %s", paths.ML_PREDS.name)


# =============================================================================
# Main entry point
# =============================================================================

def run(df: pd.DataFrame):
    """
    Full ML classification pipeline for High Adaptive Energy Vulnerability.

    Returns (model, results_dict)
    """
    paths.ML_TABLES.mkdir(parents=True, exist_ok=True)
    paths.ML_FIGURES.mkdir(parents=True, exist_ok=True)

    X, y, feature_cols, cat_cols = prepare_features(df)

    model, X_tr, X_te, y_tr, y_te, pred, prob, cat_idx = \
        train_catboost(X, y, cat_cols)

    plot_confusion_matrix(y_te, pred)
    plot_roc_curve(y_te, prob)
    plot_pr_curve(y_te, prob)
    plot_risk_distribution(y_te, prob)

    save_full_predictions(df, X, model, cat_idx)

    return model, {
        "X_train": X_tr, "X_test": X_te,
        "y_train": y_tr, "y_test": y_te,
        "y_pred":  pred,  "y_prob": prob,
        "cat_idx": cat_idx,
        "feature_cols": feature_cols,
    }
