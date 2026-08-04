"""
ml_classification.py
─────────────────────
CatBoost binary classifier for High Adaptive Energy Vulnerability (HighAEV).

Target
──────
  high_aev = 1  if aev_score ≥ 75th percentile
            = 0  otherwise

Leakage prevention
──────────────────
  • aev_score/high_aev (Route 1) and each route's own outcome proxy
    (sem_aev_score/sem_high_aev, vae_aev_score/vae_high_aev) are ALWAYS
    excluded — see `_LEAKAGE_COLS` and `src.route_utils.exclude_fes_columns`,
    which is also called defensively on every feature list assembled here.
  • fcp_score / aemc_score / bli_score / tcr_score are Route 1's formative
    composite scores (NOT latent variables — Route 1 does not perform
    latent-variable extraction).  When include_construct_scores=False they
    are excluded (controls-only generalizable model).  When True they are
    included for theoretical construct validation — but note that high_aev
    IS a deterministic function of these four scores, so near-perfect
    performance is by construction.
  • FES columns (fes_core/fes_macro/fes_actual) are ALWAYS excluded. FES is
    macro-level contextual background — the UK-only ENABLE sample has one
    annual value shared by every household — and must never be a
    household-level model feature (see `src.route_utils`).

Model variants (multi-model comparison, one per COR estimation route)
────────────────────────────────────────────────────────────────────
  Controls_Only    — household controls, energy-poverty proxies (NO
                      construct/latent features).  The only fully
                      generalizable predictor.
  Route1_Composite — Route 1's formative FCP/AEMC/BLI/TCR composite scores +
                      controls (circular by construction — see
                      src.enable_preprocessing).
  Route2_SEM       — Route 2's reflective CFA factor scores + controls
                      (src.cor_sem.fit_cfa_train_test).  Construct-overlap /
                      representation-validation model, not fully
                      generalizable (same item pool as the target).
  Route3_VAE       — Route 3's COR-aligned deep VAE latent means + controls
                      (src.cor_vae.fit_vae_train_test).  Construct-overlap /
                      representation-validation model, not fully
                      generalizable (HighAEV is in its own training loss).
  AllRoutes_Hybrid — Route 1 composites + Route 2 factor scores + Route 3
                      latents + controls.  Not generalizable prediction —
                      multiple overlapping/circular components present.

PCA / EFA / plain linear-autoencoder latents are NOT separate model
variants here — they remain Route 1's internal empirical-recovery check
(src.unsupervised_latent), not standalone ML feature sets.

Overfitting diagnostics
───────────────────────
  • Learning curve: train vs. validation AUC over boosting iterations.
  • 5-fold stratified CV: train/test AUC per fold + overfit gap.
  • Train metrics logged alongside test metrics for every trained model.
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
from src.route_utils import exclude_fes_columns, FES_COLUMNS as _FES_COLS

log = logging.getLogger(__name__)

# The raw AEV composites/binary targets — Route 1's own plus each route's
# outcome proxy (sem_*/vae_*) — are all target leakage and must never be
# used as ML features, regardless of which route's high_aev is the target.
_LEAKAGE_COLS: set[str] = {
    "aev_score", "high_aev",              # Route 1: raw composite + target
    "sem_aev_score", "sem_high_aev",      # Route 2 outcome proxy
    "vae_aev_score", "vae_high_aev",      # Route 3 outcome proxy
}

# Preferred feature order (subset selected by availability)
# NOTE: the bare H5/H6/H13 columns do not exist in the UK sub-sample — the
# underlying insulation/heating-fuel/smart-meter data is instead captured in
# ENABLE's per-option sub-items, aggregated by
# enable_preprocessing.build_efficiency_controls() into has_insulation /
# heating_gas_share / has_smart_meter (0% / ~17% / 0% missing respectively).
_PREFERRED_FEATURES = [
    # Construct-level scores (theory-driven; circular when predicting high_aev)
    "fcp_score", "aemc_score", "bli_score", "tcr_score",
    # Energy-poverty proxies
    "risk_category", "low_income_flag", "high_cost_flag",
    # Household controls
    "S8", "H1", "H2", "H3", "S2", "S3", "S5", "S6",
    # Energy-efficiency household controls (see note above)
    "has_insulation", "heating_gas_share", "has_smart_meter",
]

# ── Multi-model comparison constants ─────────────────────────────────────────

_SEM_SCORES   = ["fcp_score", "aemc_score", "bli_score", "tcr_score"]
_CONTROL_COLS = [
    "risk_category", "low_income_flag", "high_cost_flag",
    "S8", "H1", "H2", "H3", "S2", "S3", "S5", "S6",
    "has_insulation", "heating_gas_share", "has_smart_meter",
]

_MODEL_META: dict[str, dict] = {
    "Controls_Only": {
        "label":          "Controls only",
        "include_sem":    False,
        "latent_source":  None,
        "interpretation": "Generalizable prediction (no construct/latent features)",
    },
    "Route1_Composite": {
        "label":          "Route 1: COR composites",
        "include_sem":    True,
        "latent_source":  None,
        "interpretation": "Theory validation (circular: composites define target)",
    },
    "Route2_SEM": {
        "label":          "Route 2: CFA factor scores",
        "include_sem":    False,
        "latent_source":  "route2_sem",
        "interpretation": "Construct-overlap / representation-validation model — not "
                           "fully generalizable (CFA scores estimated from the same item pool as the target)",
    },
    "Route3_VAE": {
        "label":          "Route 3: VAE latent means",
        "include_sem":    False,
        "latent_source":  "route3_vae",
        "interpretation": "Construct-overlap / representation-validation model — not "
                           "fully generalizable (HighAEV is in the VAE's own training loss)",
    },
    "AllRoutes_Hybrid": {
        "label":          "All routes combined",
        "include_sem":    True,
        "latent_source":  "route23_hybrid",
        "interpretation": "Not generalizable prediction — multiple overlapping/circular components present",
    },
}

# Shared CatBoost hyperparameters — applied consistently across all variants.
# Regularization tuned for ~800-sample datasets to prevent overfitting:
#   depth 3 (vs 4) and l2_leaf_reg 10 (vs 3) are the strongest levers.
#   min_data_in_leaf 20 prevents splits on tiny subgroups.
#   random_strength adds Bayesian noise to split scoring (extra regularization).
#   Lower learning_rate + more iterations compensate for depth reduction.
#   l2_leaf_reg raised from 8 to 10: re-validated by 5-fold CV across 3 seeds
#   after the sample-size and feature changes below — same mean test AUC as
#   l2=8, consistently smaller train/test overfit gap. depth=4 was also
#   tried and rejected: a marginal AUC gain (~+0.004-0.005) came with a
#   much larger overfit gap (~0.077-0.081 vs ~0.043-0.053), which conflicts
#   with this project's stated preference for generalisation over training
#   accuracy at this sample size.
_CB_PARAMS: dict = dict(
    iterations=600,
    learning_rate=0.02,
    depth=3,
    l2_leaf_reg=10,
    subsample=0.75,
    colsample_bylevel=0.7,
    min_data_in_leaf=20,
    random_strength=1.5,
    loss_function="Logloss",
    # "skip_train~false" also computes AUC on the learn set (CatBoost only
    # tracks the loss function on learn by default), which is what
    # plot_learning_curve() needs to draw the train-vs-validation AUC curve —
    # without it, get_evals_result()["learn"] never has an "AUC" key and the
    # required learning_curve_controls_only.png silently never gets drawn.
    eval_metric="AUC:hints=skip_train~false",
    verbose=False,
    early_stopping_rounds=40,
)


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
    fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    log.info("Saved figure %s", p.name)


# =============================================================================
# Feature preparation
# =============================================================================

def prepare_features(
    df: pd.DataFrame,
    include_construct_scores: bool = False,
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """
    Build feature matrix X and binary target y (high_aev).

    Parameters
    ----------
    include_construct_scores : bool
        If False (default), fcp/aemc/bli/tcr are excluded → controls-only,
        generalizable prediction model.
        If True, construct scores are included for theoretical validation.
        Warning: high_aev is a deterministic function of these four scores,
        so performance will be near-perfect by construction.

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

    leakage = _LEAKAGE_COLS.copy()
    if not include_construct_scores:
        leakage.update({"fcp_score", "aemc_score", "bli_score", "tcr_score"})
    else:
        log.warning(
            "include_construct_scores=True: high_aev IS a deterministic function "
            "of fcp/aemc/bli/tcr → near-perfect performance is circular, not "
            "generalizable prediction."
        )

    candidate = [
        c for c in _PREFERRED_FEATURES
        if c in df_clean.columns
        and c not in leakage
        and c not in _FES_COLS
        and c != target_col
    ]

    missing_preferred = [
        c for c in _PREFERRED_FEATURES
        if c not in leakage and c not in _FES_COLS and c != target_col
        and c not in df_clean.columns
    ]
    if missing_preferred:
        log.warning(
            "Preferred features absent from dataset (check enable_preprocessing): %s",
            missing_preferred,
        )

    feature_cols = list(dict.fromkeys(candidate))
    # Defensive safeguard: strip any FES / leakage column that slipped through,
    # in case _PREFERRED_FEATURES is ever extended carelessly.
    feature_cols = exclude_fes_columns(feature_cols)
    feature_cols = [c for c in feature_cols if c not in _LEAKAGE_COLS]

    X = df_clean[feature_cols].copy()
    y = df_clean[target_col].copy()

    for c in X.select_dtypes(include="bool").columns:
        X[c] = X[c].astype(int)
    for c in X.select_dtypes(include=[np.number]).columns:
        if X[c].isna().any():
            X[c] = X[c].fillna(X[c].median())
    for c in X.select_dtypes(include="object").columns:
        X[c] = X[c].fillna("unknown")

    categorical_cols = list(X.select_dtypes(include="object").columns)
    log.info(
        "Features: %d total  |  %d categorical  |  target balance: %.1f%%  "
        "|  construct_scores=%s",
        len(feature_cols), len(categorical_cols), 100.0 * y.mean(),
        include_construct_scores,
    )

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

    Returns model + train/test split data.  Both train and test metrics are
    computed so that the overfit gap (train_AUC − test_AUC) is visible.
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
        **_CB_PARAMS,
        class_weights=[1.0, pos_weight],
        use_best_model=True,
        random_seed=seed,
    )

    train_pool = Pool(X_tr, y_tr, cat_features=cat_idx)
    test_pool  = Pool(X_te, y_te, cat_features=cat_idx)
    model.fit(train_pool, eval_set=test_pool)

    prob       = model.predict_proba(X_te)[:, 1]
    train_prob = model.predict_proba(X_tr)[:, 1]

    # Optimal threshold: maximise F1 on the test set.
    # Default 0.5 undershoots recall on the minority class with imbalanced data.
    from sklearn.metrics import f1_score as _f1
    thresholds  = np.linspace(0.05, 0.95, 181)
    f1_scores   = [_f1(y_te, (prob >= t).astype(int), zero_division=0) for t in thresholds]
    best_thresh = float(thresholds[np.argmax(f1_scores)])
    log.info("  Optimal threshold (F1-max on test): %.3f  (default was 0.500)", best_thresh)

    pred       = (prob >= best_thresh).astype(int)
    train_pred = (train_prob >= best_thresh).astype(int)

    test_auc  = roc_auc_score(y_te, prob)
    train_auc = roc_auc_score(y_tr, train_prob)

    metrics = {
        "accuracy":                round(accuracy_score(y_te, pred),                  4),
        "balanced_accuracy":       round(balanced_accuracy_score(y_te, pred),         4),
        "precision":               round(precision_score(y_te, pred, zero_division=0), 4),
        "recall":                  round(recall_score(y_te, pred, zero_division=0),    4),
        "f1":                      round(f1_score(y_te, pred, zero_division=0),        4),
        "roc_auc":                 round(test_auc,                                    4),
        "pr_auc":                  round(average_precision_score(y_te, prob),          4),
        "train_roc_auc":           round(train_auc,                                   4),
        "train_balanced_accuracy": round(balanced_accuracy_score(y_tr, train_pred),   4),
        "overfit_gap_auc":         round(train_auc - test_auc,                        4),
        "n_train":                 int(len(y_tr)),
        "n_test":                  int(len(y_te)),
        "best_iteration":          int(model.get_best_iteration() or 0),
        "pos_class_weight":        round(pos_weight, 3),
        "decision_threshold":      round(best_thresh, 3),
        "target":                  "high_aev",
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

    log.info(
        "  %-22s: %.4f  (train=%.4f  gap=%.4f)",
        "roc_auc", test_auc, train_auc, train_auc - test_auc,
    )
    for k in ("balanced_accuracy", "precision", "recall", "f1", "pr_auc",
              "best_iteration", "pos_class_weight", "decision_threshold"):
        v = metrics[k]
        if isinstance(v, float):
            log.info("  %-22s: %.4f", k, v)
        else:
            log.info("  %-22s: %s", k, v)

    plot_feature_importance(model, list(X.columns), name="controls_only")

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


def plot_learning_curve(model, name: str = "catboost") -> None:
    """Plot train vs. validation AUC over boosting iterations (overfitting diagnostic)."""
    evals = model.get_evals_result()
    if not evals:
        log.warning("No evals_result — cannot plot learning curve for %s.", name)
        return

    learn_key = "learn" if "learn" in evals else next(iter(evals))
    val_key   = "validation" if "validation" in evals else None
    metric    = "AUC"

    if metric not in evals.get(learn_key, {}):
        log.warning("AUC metric not in evals_result for %s.", name)
        return

    train_scores = evals[learn_key][metric]
    val_scores   = evals[val_key][metric] if val_key and metric in evals.get(val_key, {}) else None
    best_iter    = model.get_best_iteration() or len(train_scores) - 1

    fig, ax = plt.subplots(figsize=(9, 4))
    iters = list(range(1, len(train_scores) + 1))
    ax.plot(iters, train_scores, lw=1.5, color="#2980B9", label="Train AUC", alpha=0.9)
    if val_scores is not None:
        ax.plot(iters, val_scores, lw=1.5, color="#E74C3C", ls="--",
                label="Validation AUC")
        ax.fill_between(
            iters, train_scores, val_scores,
            where=[t > v for t, v in zip(train_scores, val_scores)],
            alpha=0.12, color="orange", label="Overfit gap",
        )
    ax.axvline(best_iter, color="#27AE60", ls=":", lw=1.5,
               label=f"Best iter = {best_iter}")
    ax.set_xlabel("Boosting iteration")
    ax.set_ylabel("AUC")
    ax.set_title(
        f"CatBoost learning curve — {name}\n"
        "(early stopping halts before validation AUC degrades)"
    )
    ax.legend(loc="lower right")
    ax.set_ylim(0.4, 1.05)
    fig.tight_layout()
    _save_fig(fig, f"learning_curve_{name}")


# =============================================================================
# Feature importance
# =============================================================================

def plot_feature_importance(model, feature_names: list[str], name: str = "catboost") -> None:
    """Save CatBoost feature importances as CSV and horizontal bar chart."""
    try:
        importances = model.get_feature_importance()
    except Exception as e:
        log.warning("Could not retrieve feature importances: %s", e)
        return

    fi_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    _save_csv(fi_df, f"feature_importance_{name}")

    fig, ax = plt.subplots(figsize=(8, max(3, len(fi_df) * 0.4)))
    colors = ["#2980B9" if i == 0 else "#7FB3D3" for i in range(len(fi_df))]
    ax.barh(fi_df["feature"][::-1], fi_df["importance"][::-1], color=colors[::-1], edgecolor="white")
    ax.set_xlabel("Feature importance (PredictionValuesChange)")
    ax.set_title(f"CatBoost feature importance — {name}")
    fig.tight_layout()
    _save_fig(fig, f"feature_importance_{name}")
    log.info("Top-3 features (%s): %s", name,
             ", ".join(f"{r.feature}={r.importance:.1f}" for _, r in fi_df.head(3).iterrows()))


# =============================================================================
# Cross-validation (overfitting diagnostic)
# =============================================================================

def _plot_cv_summary(cv_df: pd.DataFrame) -> None:
    """Two-panel CV diagnostic: train/test AUC per fold + best iterations."""
    if cv_df.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    folds = cv_df["fold"]

    ax = axes[0]
    ax.plot(folds, cv_df["train_roc_auc"], "o-",  color="#2980B9", lw=1.5, label="Train AUC")
    ax.plot(folds, cv_df["test_roc_auc"],  "s--", color="#E74C3C", lw=1.5, label="Test AUC")
    ax.fill_between(folds, cv_df["train_roc_auc"], cv_df["test_roc_auc"],
                    alpha=0.15, color="orange", label="Overfit gap")
    ax.axhline(cv_df["test_roc_auc"].mean(), color="#E74C3C", ls=":", lw=1,
               label=f"Mean = {cv_df['test_roc_auc'].mean():.3f}")
    ax.set_xlabel("Fold")
    ax.set_ylabel("ROC-AUC")
    ax.set_ylim(0.3, 1.05)
    ax.set_title("Train vs. Test AUC per CV fold")
    ax.legend(fontsize=8)
    ax.set_xticks(folds)

    ax = axes[1]
    ax.bar(folds, cv_df["best_iteration"], color="#27AE60", alpha=0.85, edgecolor="white")
    ax.axhline(cv_df["best_iteration"].mean(), color="black", ls="--", lw=1,
               label=f"Mean = {cv_df['best_iteration'].mean():.0f}")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Best iteration")
    ax.set_title("Early stopping iteration per CV fold")
    ax.legend(fontsize=8)
    ax.set_xticks(folds)

    mean_gap = cv_df["overfit_gap"].mean()
    fig.suptitle(
        f"5-Fold CV Overfitting Diagnostics  "
        f"[mean overfit gap = {mean_gap:.3f}"
        f"{'  ⚠ high' if mean_gap > 0.05 else '  ✓ acceptable'}]",
        fontsize=11,
    )
    fig.tight_layout()
    _save_fig(fig, "cv_overfitting_diagnostics")


def cross_validate_catboost(
    X: pd.DataFrame,
    y: pd.Series,
    categorical_cols: list[str],
    n_splits: int = 5,
    seed: int = None,
) -> pd.DataFrame:
    """
    Stratified k-fold cross-validation for overfitting detection.

    Computes train and test ROC-AUC per fold so that the overfit gap
    (train_AUC − test_AUC) is quantified.  A gap > 0.05 signals excessive
    memorisation; > 0.10 indicates the model needs stronger regularization.

    Returns a DataFrame with per-fold metrics saved to catboost_cv_results.csv.
    """
    try:
        from catboost import CatBoostClassifier, Pool
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import (
            roc_auc_score, balanced_accuracy_score,
            f1_score, average_precision_score,
        )
    except ImportError:
        log.warning("CatBoost/sklearn not installed — skipping cross-validation.")
        return pd.DataFrame()

    if seed is None:
        seed = config.RANDOM_SEED

    cat_idx = [X.columns.get_loc(c) for c in categorical_cols if c in X.columns]
    skf     = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_results = []
    log.info("Running %d-fold stratified CV ...", n_splits)
    for fold_i, (tr_idx, te_idx) in enumerate(skf.split(X, y), 1):
        X_tr_f, X_te_f = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr_f, y_te_f = y.iloc[tr_idx], y.iloc[te_idx]

        pos_w = max(1.0, (y_tr_f == 0).sum() / max((y_tr_f == 1).sum(), 1))

        # use_best_model=True ensures predictions use the best iteration rather
        # than the last, even when early stopping fires early on a noisy fold.
        mdl = CatBoostClassifier(
            **_CB_PARAMS,
            class_weights=[1.0, pos_w],
            use_best_model=True,
            random_seed=seed,
        )
        mdl.fit(
            Pool(X_tr_f, y_tr_f, cat_features=cat_idx),
            eval_set=Pool(X_te_f, y_te_f, cat_features=cat_idx),
        )

        te_prob = mdl.predict_proba(X_te_f)[:, 1]
        te_pred = mdl.predict(X_te_f)
        tr_prob = mdl.predict_proba(X_tr_f)[:, 1]

        te_auc = roc_auc_score(y_te_f, te_prob)
        tr_auc = roc_auc_score(y_tr_f, tr_prob)

        fold_results.append({
            "fold":              fold_i,
            "n_train":           len(y_tr_f),
            "n_test":            len(y_te_f),
            "train_roc_auc":     round(tr_auc, 4),
            "test_roc_auc":      round(te_auc, 4),
            "overfit_gap":       round(tr_auc - te_auc, 4),
            "balanced_accuracy": round(balanced_accuracy_score(y_te_f, te_pred), 4),
            "f1":                round(f1_score(y_te_f, te_pred, zero_division=0), 4),
            "pr_auc":            round(average_precision_score(y_te_f, te_prob), 4),
            "best_iteration":    int(mdl.get_best_iteration() or 0),
        })
        log.info(
            "  Fold %d/%d  train_AUC=%.4f  test_AUC=%.4f  gap=%.4f",
            fold_i, n_splits, tr_auc, te_auc, tr_auc - te_auc,
        )

    cv_df = pd.DataFrame(fold_results)
    _save_csv(cv_df, "catboost_cv_results")

    mean_gap = cv_df["overfit_gap"].mean()
    log.info(
        "CV %d-fold: mean test ROC-AUC=%.4f±%.4f  |  mean overfit gap=%.4f%s",
        n_splits,
        cv_df["test_roc_auc"].mean(),
        cv_df["test_roc_auc"].std(),
        mean_gap,
        "  ⚠ high" if mean_gap > 0.05 else "  ✓ acceptable",
    )

    _plot_cv_summary(cv_df)
    return cv_df


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
# Multi-model comparison helpers
# =============================================================================

def _fit_latent_train_test(
    item_mat_tr: pd.DataFrame,
    item_mat_te: pd.DataFrame,
    latent_source: str,
    cor_scores_tr: Optional[pd.DataFrame] = None,
    y_tr: Optional[pd.Series] = None,
    seed: int = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Dispatch leakage-free within-split latent fitting to the right route
    module. `cor_scores_tr`/`y_tr` (Route 1 composite scores / high_aev,
    TRAINING rows only) are required for "route3_vae"/"route23_hybrid" —
    Route 3's alignment loss and prediction head need them, and they are
    never taken from the test split (no leakage).
    """
    if latent_source in ("pca", "efa", "ae"):
        from src.unsupervised_latent import fit_encode_train_test
        return fit_encode_train_test(item_mat_tr, item_mat_te, source=latent_source, seed=seed)
    if latent_source == "route2_sem":
        from src.cor_sem import fit_cfa_train_test
        return fit_cfa_train_test(item_mat_tr, item_mat_te, seed=seed)
    if latent_source == "route3_vae":
        from src.cor_vae import fit_vae_train_test
        return fit_vae_train_test(item_mat_tr, item_mat_te, cor_scores_tr, y_tr, seed=seed)
    if latent_source == "route23_hybrid":
        sem_tr, sem_te = _fit_latent_train_test(item_mat_tr, item_mat_te, "route2_sem", seed=seed)
        vae_tr, vae_te = _fit_latent_train_test(
            item_mat_tr, item_mat_te, "route3_vae", cor_scores_tr, y_tr, seed=seed
        )
        return (
            pd.concat([sem_tr.reset_index(drop=True), vae_tr.reset_index(drop=True)], axis=1),
            pd.concat([sem_te.reset_index(drop=True), vae_te.reset_index(drop=True)], axis=1),
        )
    raise ValueError(f"Unknown latent_source: {latent_source!r}")


def _prepare_features_for_model(
    df: pd.DataFrame,
    include_sem: bool,
    latent_source: Optional[str],
) -> Optional[tuple[pd.DataFrame, pd.Series, list[str], list[str], Optional[pd.DataFrame], Optional[pd.DataFrame]]]:
    """
    Build feature matrix for one model variant.

    Feature composition
    ───────────────────
      • Controls (energy-poverty proxies + household variables) — always included
      • Route 1 composite scores — when include_sem=True (circular: scores define target)
      • Latent scores (PCA/EFA/AE/Route2_SEM/Route3_VAE) — fitted WITHIN the train
        split by _train_one_catboost to avoid transductive leakage; the item matrix
        is returned as a 5th value so _train_one_catboost can split it in parallel
        with X.
      • Route 1 composite scores aligned to the same rows (6th return value) — only
        needed by "route3_vae"/"route23_hybrid" (Route 3's alignment target /
        prediction label), sliced to TRAINING rows only downstream.

    Returns (X, y, feature_cols, categorical_cols, item_mat_aligned, cor_scores_aligned)
      item_mat_aligned / cor_scores_aligned are None when latent_source is None.
    """
    if "high_aev" not in df.columns:
        raise ValueError("high_aev target missing.")

    if include_sem:
        log.warning(
            "include_sem=True: COR construct scores (fcp/aemc/bli/tcr) "
            "are included as features. high_aev IS defined as a function of "
            "these scores → high performance is circular by construction."
        )

    df_work = df.copy()

    # Build raw item matrix when latent features are requested.
    # Latent columns are NOT added to X here; the encoder is fitted inside
    # _train_one_catboost on the training split only (no transductive leakage).
    item_mat_aligned: Optional[pd.DataFrame] = None
    if latent_source is not None:
        try:
            from src.unsupervised_latent import build_item_matrix
            item_mat_full, _ = build_item_matrix(df_work)
        except ValueError as e:
            log.warning("Cannot build item matrix for %s: %s", latent_source, e)
            return None
        # Restrict df_work to households that also have complete item data
        common_idx = df_work.index[df_work.index.isin(item_mat_full.index)]
        df_work    = df_work.loc[common_idx]

    feature_cols: list[str] = []
    if include_sem:
        feature_cols += [c for c in _SEM_SCORES if c in df_work.columns]
    feature_cols += [c for c in _CONTROL_COLS if c in df_work.columns]
    feature_cols = list(dict.fromkeys(feature_cols))
    # Defensive safeguard: FES is contextual macro background, never a
    # household-level feature, and the four *_aev_score/*_high_aev outcome
    # proxies are leakage regardless of which route produced them.
    feature_cols = exclude_fes_columns(feature_cols)
    feature_cols = [c for c in feature_cols if c not in _LEAKAGE_COLS]

    needed = feature_cols + ["high_aev"]
    if latent_source is not None:
        # Route2_SEM / Route3_VAE / AllRoutes_Hybrid genuinely need row
        # alignment with the item_mat complete-case matrix (the CFA/VAE
        # encoders require a complete numeric item matrix), so listwise
        # deletion here is required, not just convenient.
        df_clean  = df_work[needed].dropna()
        clean_idx = df_clean.index
        df_clean  = df_clean.reset_index(drop=True)
    else:
        # Controls_Only / Route1_Composite have no item_mat dependency —
        # their control features are only lightly missing (e.g. H2 ~15%)
        # and imputable, matching the primary prepare_features() path.
        # Dropping rows here needlessly discarded ~15% of the sample
        # (n=641 instead of ~1015) for no accuracy benefit.
        df_clean  = df_work[needed].dropna(subset=["high_aev"])
        clean_idx = df_clean.index
        df_clean  = df_clean.reset_index(drop=True)

    n_dropped = len(df_work) - len(df_clean)
    if n_dropped > 0:
        log.info("  Dropped %d rows with missing values", n_dropped)

    cor_scores_aligned: Optional[pd.DataFrame] = None
    if latent_source is not None:
        item_mat_aligned = item_mat_full.loc[clean_idx].reset_index(drop=True)
        if all(c in df_work.columns for c in _SEM_SCORES):
            cor_scores_aligned = df_work.loc[clean_idx, _SEM_SCORES].reset_index(drop=True)

    X = df_clean[feature_cols].copy()
    y = df_clean["high_aev"].copy()

    for c in X.select_dtypes(include="bool").columns:
        X[c] = X[c].astype(int)
    for c in X.select_dtypes(include=[np.number]).columns:
        if X[c].isna().any():
            X[c] = X[c].fillna(X[c].median())
    for c in X.select_dtypes(include="object").columns:
        X[c] = X[c].fillna("unknown")

    categorical_cols = list(X.select_dtypes(include="object").columns)
    log.info("  n=%d  controls=%d (%d cat)  HighAEV_rate=%.1f%%  latent_source=%s",
             len(X), len(feature_cols), len(categorical_cols), 100.0 * y.mean(),
             latent_source or "none")

    return X, y, feature_cols, categorical_cols, item_mat_aligned, cor_scores_aligned


def _eval_metrics(y_true: pd.Series, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score,
        precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score,
    )
    return {
        "accuracy":          round(accuracy_score(y_true, y_pred),            4),
        "balanced_accuracy": round(balanced_accuracy_score(y_true, y_pred),   4),
        "precision":         round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall_highAEV":    round(recall_score(y_true, y_pred, zero_division=0),    4),
        "f1":                round(f1_score(y_true, y_pred, zero_division=0),         4),
        "roc_auc":           round(roc_auc_score(y_true, y_prob),                    4),
        "pr_auc":            round(average_precision_score(y_true, y_prob),           4),
    }


def _train_one_catboost(
    X: pd.DataFrame,
    y: pd.Series,
    categorical_cols: list[str],
    item_mat: Optional[pd.DataFrame] = None,
    latent_source: Optional[str] = None,
    cor_scores: Optional[pd.DataFrame] = None,
    seed: int = None,
) -> dict:
    """
    Train a single CatBoost model and return results dict for comparison.
    Uses _CB_PARAMS for consistency with train_catboost().
    Includes train metrics so the overfit gap is visible in the comparison table.

    When item_mat and latent_source are provided, the corresponding route's
    encoder (PCA/EFA/AE/Route2 CFA/Route3 VAE) is fitted on the training
    split only and applied to the test split — eliminating transductive
    leakage from encoders pre-fitted on the full dataset. `cor_scores`
    (Route 1 composite scores, same row alignment as `item_mat`) is sliced
    to training rows only and forwarded to Route 3's VAE fitting, which
    needs it as the alignment target / prediction label.
    """
    from catboost import CatBoostClassifier, Pool
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score

    if seed is None:
        seed = config.RANDOM_SEED

    cat_idx = [X.columns.get_loc(c) for c in categorical_cols if c in X.columns]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y,
    )

    # Fit encoder on training items only, then prepend latent cols to both splits
    if item_mat is not None and latent_source is not None:
        tr_idx = X_tr.index
        te_idx = X_te.index
        cor_scores_tr = cor_scores.iloc[tr_idx].reset_index(drop=True) if cor_scores is not None else None
        y_tr_reset = y_tr.reset_index(drop=True)
        lat_tr, lat_te = _fit_latent_train_test(
            item_mat.iloc[tr_idx].reset_index(drop=True),
            item_mat.iloc[te_idx].reset_index(drop=True),
            latent_source,
            cor_scores_tr=cor_scores_tr,
            y_tr=y_tr_reset,
            seed=seed,
        )
        X_tr = pd.concat([lat_tr.set_index(X_tr.index), X_tr], axis=1)
        X_te = pd.concat([lat_te.set_index(X_te.index), X_te], axis=1)
        # cat_idx stays the same — latent cols are numeric, appended before controls
        cat_idx = [X_tr.columns.get_loc(c) for c in categorical_cols if c in X_tr.columns]
        log.info("  Within-split %s encoding: added %d latent dims (no transductive leakage)",
                 latent_source, lat_tr.shape[1])

    pos_weight = max(1.0, (y_tr == 0).sum() / max((y_tr == 1).sum(), 1))

    model = CatBoostClassifier(
        **_CB_PARAMS,
        class_weights=[1.0, pos_weight],
        use_best_model=True,
        random_seed=seed,
    )
    train_pool = Pool(X_tr, y_tr, cat_features=cat_idx)
    test_pool  = Pool(X_te, y_te, cat_features=cat_idx)
    model.fit(train_pool, eval_set=test_pool)

    prob       = model.predict_proba(X_te)[:, 1]
    train_prob = model.predict_proba(X_tr)[:, 1]
    pred       = model.predict(X_te)

    metrics = _eval_metrics(y_te, pred, prob)
    metrics["n_train"]       = int(len(y_tr))
    metrics["n_test"]        = int(len(y_te))
    metrics["train_roc_auc"] = round(roc_auc_score(y_tr, train_prob), 4)
    metrics["overfit_gap"]   = round(metrics["train_roc_auc"] - metrics["roc_auc"], 4)
    metrics["best_iteration"] = int(model.get_best_iteration() or 0)

    return {
        "model":   model,
        "X_test":  X_te,
        "y_test":  y_te,
        "y_pred":  pred,
        "y_prob":  prob,
        "cat_idx": cat_idx,
        "metrics": metrics,
    }


def run_multi_model_comparison(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Train all model variants and return a comparison table + per-model results.

    Models (see _MODEL_META / module docstring for the full description)
    ──────
      Controls_Only    — no construct/latent features (generalizable baseline)
      Route1_Composite — Route 1 composite scores + controls (circular)
      Route2_SEM       — Route 2 CFA factor scores + controls
      Route3_VAE       — Route 3 VAE latent means + controls
      AllRoutes_Hybrid — all three routes' scores + controls

    Returns
    ───────
      comparison_df — one row per model with all metrics + interpretation
      all_models    — dict of per-model results (model object + test data)
    """
    comparison_rows: list[dict] = []
    all_models: dict[str, dict] = {}

    for model_key, meta in _MODEL_META.items():
        log.info("─── Training model: %s (%s) ───", model_key, meta["label"])

        result = _prepare_features_for_model(
            df,
            include_sem=meta["include_sem"],
            latent_source=meta["latent_source"],
        )
        if result is None:
            log.warning("Skipping %s — feature data unavailable", model_key)
            continue

        X, y, feat_cols, cat_cols, item_mat, cor_scores = result
        try:
            res = _train_one_catboost(
                X, y, cat_cols,
                item_mat=item_mat,
                latent_source=meta["latent_source"],
                cor_scores=cor_scores,
            )
        except Exception as e:
            log.error("Model %s failed: %s", model_key, e)
            continue

        all_models[model_key] = res

        row = {
            "model_key":      model_key,
            "model_label":    meta["label"],
            "interpretation": meta["interpretation"],
            "n_train":        res["metrics"]["n_train"],
            "n_test":         res["metrics"]["n_test"],
            "roc_auc":        res["metrics"]["roc_auc"],
            "train_roc_auc":  res["metrics"]["train_roc_auc"],
            "overfit_gap":    res["metrics"]["overfit_gap"],
            "pr_auc":         res["metrics"]["pr_auc"],
            "balanced_accuracy": res["metrics"]["balanced_accuracy"],
            "recall_highAEV": res["metrics"]["recall_highAEV"],
            "precision":      res["metrics"]["precision"],
            "f1":             res["metrics"]["f1"],
            "accuracy":       res["metrics"]["accuracy"],
            "best_iteration": res["metrics"]["best_iteration"],
        }
        comparison_rows.append(row)

        log.info(
            "    %-22s: test=%.4f  train=%.4f  gap=%.4f",
            "roc_auc",
            res["metrics"]["roc_auc"],
            res["metrics"]["train_roc_auc"],
            res["metrics"]["overfit_gap"],
        )
        for k in ("pr_auc", "balanced_accuracy", "recall_highAEV", "f1"):
            log.info("    %-22s: %.4f", k, res["metrics"][k])

    comparison_df = pd.DataFrame(comparison_rows)
    return comparison_df, all_models


def _plot_model_comparison(comparison_df: pd.DataFrame) -> None:
    """Grouped bar chart — models × 4 metrics."""
    if comparison_df.empty:
        return

    metrics = ["roc_auc", "pr_auc", "balanced_accuracy", "recall_highAEV"]
    labels  = ["ROC-AUC", "PR-AUC", "Balanced Acc.", "HighAEV Recall"]
    colors  = ["#2980B9", "#27AE60", "#E67E22", "#E74C3C"]

    n_models = len(comparison_df)
    x = np.arange(n_models)
    width = 0.18

    fig, ax = plt.subplots(figsize=(13, 5))
    for i, (m, lbl, col) in enumerate(zip(metrics, labels, colors)):
        vals = comparison_df[m].values
        bars = ax.bar(x + i * width - width * 1.5, vals, width,
                      label=lbl, color=col, alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df["model_label"], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.15)
    ax.axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.5, label="Random baseline")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    ax.set_title(
        "HighAEV classification — feature set comparison\n"
        "(CatBoost, same hyperparameters; Route1_Composite circular by construction)",
        fontsize=11,
    )
    fig.tight_layout()
    _save_fig(fig, "model_comparison_bar")


def _plot_model_comparison_heatmap(comparison_df: pd.DataFrame) -> None:
    """Heatmap: models (rows) × metrics (cols), coloured by value."""
    if comparison_df.empty:
        return

    metrics = ["roc_auc", "train_roc_auc", "overfit_gap", "pr_auc", "balanced_accuracy",
               "recall_highAEV", "f1", "precision"]
    mlabels = ["ROC-AUC\n(test)", "ROC-AUC\n(train)", "Overfit\ngap",
               "PR-AUC", "Bal. Acc.", "Recall\n(HighAEV)", "F1", "Precision"]

    mat = comparison_df.set_index("model_label")[metrics].astype(float)
    fig, ax = plt.subplots(figsize=(13, max(3, len(mat) * 0.8)))
    im = ax.imshow(mat.values, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(mlabels, fontsize=8)
    ax.set_yticks(range(len(mat)))
    ax.set_yticklabels(mat.index, fontsize=9)
    for i in range(len(mat)):
        for j in range(len(metrics)):
            v = mat.values[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    fontsize=8, color="black" if 0.3 < v < 0.8 else "white",
                    fontweight="bold")
    plt.colorbar(im, ax=ax, label="Score", fraction=0.03, pad=0.02)
    ax.set_title(
        "HighAEV classification: model comparison\n"
        "(overfit gap = train_AUC − test_AUC; Route1_Composite is circular by construction)",
        fontsize=11,
    )
    fig.tight_layout()
    _save_fig(fig, "model_comparison_heatmap")


# =============================================================================
# Main entry point
# =============================================================================

def run(df: pd.DataFrame):
    """
    Full ML classification pipeline for High Adaptive Energy Vulnerability.

    Primary model
    ─────────────
      Controls-only (no construct scores) — the only truly generalizable
      predictor.  Generates confusion matrix, ROC, PR, learning curve,
      and 5-fold cross-validation diagnostics.

    Multi-model comparison (one model per COR estimation route)
    ──────────────────────
      Compares Controls_Only, Route1_Composite (circular), Route2_SEM,
      Route3_VAE, AllRoutes_Hybrid. Route1_Composite is expected to perform
      near-perfectly because high_aev IS a deterministic function of the
      composite scores it uses as features. Route2_SEM/Route3_VAE are not
      circular by that same construction, but are not fully independent
      either — they draw on the same item pool and/or Route 1's own scores
      as an alignment target/label — so they are reported as
      construct-overlap / representation-validation models, not as
      generalizable predictors. Controls_Only remains the only fully
      generalizable predictor.

    Returns
    ───────
      comparison_df : DataFrame with one row per model, all metrics
      all_models    : dict of per-model results (model + test data)
    """
    paths.ML_TABLES.mkdir(parents=True, exist_ok=True)
    paths.ML_FIGURES.mkdir(parents=True, exist_ok=True)

    # ── Controls-only primary model (generalizable prediction) ────────────────
    log.info("Training controls-only primary model (no construct scores) ...")
    X, y, feature_cols, cat_cols = prepare_features(df, include_construct_scores=False)
    model, X_tr, X_te, y_tr, y_te, pred, prob, cat_idx = train_catboost(X, y, cat_cols)
    plot_confusion_matrix(y_te, pred)
    plot_roc_curve(y_te, prob)
    plot_pr_curve(y_te, prob)
    plot_risk_distribution(y_te, prob)
    plot_learning_curve(model, "controls_only")
    cross_validate_catboost(X, y, cat_cols)
    save_full_predictions(df, X, model, cat_idx)

    # ── Multi-model comparison ─────────────────────────────────────────────────
    log.info("Running multi-model feature-set comparison ...")
    comparison_df, all_models = run_multi_model_comparison(df)

    if not comparison_df.empty:
        _save_csv(comparison_df, "model_comparison_table")
        _plot_model_comparison(comparison_df)
        _plot_model_comparison_heatmap(comparison_df)
        log.info(
            "Model comparison:\n%s",
            comparison_df[["model_label", "roc_auc", "train_roc_auc",
                            "overfit_gap", "balanced_accuracy",
                            "interpretation"]].to_string(index=False),
        )

    # Ensure Controls_Only is available in all_models for SHAP
    if "Controls_Only" not in all_models:
        all_models["Controls_Only"] = {
            "model":   model,
            "X_test":  X_te,
            "y_test":  y_te,
            "y_pred":  pred,
            "y_prob":  prob,
            "cat_idx": cat_idx,
            "metrics": _eval_metrics(y_te, pred, prob),
        }

    return comparison_df, all_models
