"""
run_ml_shap.py
--------------
Stage 6: CatBoost + SHAP explainability layer.

Input:
    outputs/social_sem/enable_fes_cor_scored.csv  (produced by run_sem.py)

Goal:
    Predict household-level high thermal discomfort risk from COR-derived
    behavioural scores and household characteristics, then explain
    which features drive predictions using SHAP.

FES policy:
    Annual FES is constant for all UK households (zero within-sample
    variance) so it is EXCLUDED from ML predictors.  It is retained
    only as contextual metadata in the prediction CSV.

Target:
    high_discomfort = 1  if discomfort_score >= 75th percentile
                     0  otherwise

Predictors (no leakage):
    insecurity_score, preservation_score
    risk_category (energy poverty classification)
    household controls: S8, H1, H2, H3
    low_income_flag, high_cost_flag
    normalised COR indicators: n_E2A, n_E2B, n_E7A-E7E,
                               n_H9, n_E5A1-E5A5, n_E5A9,
                               n_E6A1-E6A8
    (discomfort item normalised scores n_H12A, n_H12B, n_H15A-E excluded)

Outputs:
    outputs/ml_shap/tables/      <- 5 CSV tables
    outputs/ml_shap/figures/     <- 7 figures (PNG)
    outputs/ml_shap/enable_ml_predictions.csv
    outputs/ml_shap/README_ml_shap.md

Usage:
    python -m src.run_ml_shap
"""

from __future__ import annotations
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT      = Path(__file__).parent.parent
DATA_PATH = ROOT / "outputs" / "social_sem" / "enable_fes_cor_scored.csv"
OUT_DIR   = ROOT / "outputs" / "ml_shap"
TABLE_DIR = OUT_DIR / "tables"
FIG_DIR   = OUT_DIR / "figures"

# Discomfort item normalised scores — excluded to prevent target leakage
_DISCOMFORT_N_COLS = {"n_H12A", "n_H12B", "n_H15A", "n_H15B", "n_H15C", "n_H15D", "n_H15E"}

# Also exclude n_S8 because raw S8 is already in preferred_features (avoid duplication)
_INSECURITY_CONTROLS = {"n_S8"}

_PALETTE = {
    "bar":  "#2980B9",
    "pos":  "#E74C3C",
    "neg":  "#27AE60",
    "grid": "#EAECEE",
}


# =============================================================================
# I/O helpers
# =============================================================================

def _mkdir():
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def _save_csv(df: pd.DataFrame, name: str) -> None:
    p = TABLE_DIR / f"{name}.csv"
    df.to_csv(p, index=False)
    print(f"  [saved] {p.name}")


def _save_fig(fig: plt.Figure, name: str) -> None:
    p = FIG_DIR / f"{name}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {p.name}")


# =============================================================================
# STEP 1 — Load data
# =============================================================================

def load_data() -> pd.DataFrame:
    print("\n[1] Loading COR-scored dataset...")
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Missing: {DATA_PATH}\n"
            "Run first:  python -m src.run_sem"
        )
    df = pd.read_csv(DATA_PATH)
    print(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


# =============================================================================
# STEP 2 — Create binary target
# =============================================================================

def create_target(df: pd.DataFrame, quantile: float = 0.75) -> pd.DataFrame:
    """
    high_discomfort = 1 if discomfort_score >= 75th percentile, else 0.
    Rows with NaN discomfort_score are dropped (18 of 1015 in UK sub-sample).
    """
    print("\n[2] Creating high-discomfort binary target...")

    if "discomfort_score" not in df.columns:
        raise ValueError("discomfort_score missing -- run src.run_sem first.")

    n_before = len(df)
    df = df.dropna(subset=["discomfort_score"]).reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"  Dropped {n_dropped} rows with missing discomfort_score")

    q = df["discomfort_score"].quantile(quantile)
    df["high_discomfort"] = (df["discomfort_score"] >= q).astype(int)

    vc = df["high_discomfort"].value_counts()
    print(f"  Threshold (p{int(quantile*100)}): {q:.4f}")
    print(f"  high_discomfort=1: {vc.get(1,0):,}   high_discomfort=0: {vc.get(0,0):,}")
    return df


# =============================================================================
# STEP 3 — Feature preparation
# =============================================================================

def prepare_features(df: pd.DataFrame):
    """
    Build feature matrix X and target y.

    Feature groups (no leakage):
      - COR composite scores (insecurity, preservation)
      - Energy poverty flags and category
      - Household controls (S8, H1, H2, H3)
      - Normalised insecurity + preservation item scores
        (H12/H15 discomfort items excluded; n_S8 excluded as redundant with S8)
    """
    print("\n[3] Preparing ML features...")

    leakage_cols = {
        "discomfort_score",
        "high_discomfort",
        *_DISCOMFORT_N_COLS,
    }
    constant_cols = {"fes_core", "fes_macro", "fes_actual"}

    preferred = [
        "insecurity_score",
        "preservation_score",
        "risk_category",          # categorical: energy_poor / at_risk / not_poor
        "low_income_flag",
        "high_cost_flag",
        "S8", "H1", "H2", "H3",
    ]

    # Normalised COR item scores (non-discomfort, non-redundant)
    n_cols = [
        c for c in df.columns
        if c.startswith("n_")
        and c not in _DISCOMFORT_N_COLS
        and c not in _INSECURITY_CONTROLS
    ]

    feature_cols = [
        c for c in preferred + n_cols
        if c in df.columns
        and c not in leakage_cols
        and c not in constant_cols
    ]
    feature_cols = list(dict.fromkeys(feature_cols))  # preserve order, deduplicate

    X = df[feature_cols].copy()
    y = df["high_discomfort"].copy()

    # Bool -> int
    for c in X.select_dtypes(include="bool").columns:
        X[c] = X[c].astype(int)

    # Numeric: fill NaN with column median
    for c in X.select_dtypes(include=[np.number]).columns:
        if X[c].isna().any():
            X[c] = X[c].fillna(X[c].median())

    # Categorical: fill NaN with "unknown"
    for c in X.select_dtypes(include="object").columns:
        X[c] = X[c].fillna("unknown")

    categorical_cols = list(X.select_dtypes(include="object").columns)

    print(f"  Total features: {len(feature_cols)}")
    print(f"  Categorical:    {categorical_cols}")
    print(f"  Numeric:        {len(feature_cols) - len(categorical_cols)}")

    _save_csv(
        pd.DataFrame({"feature": feature_cols,
                      "is_categorical": [c in categorical_cols for c in feature_cols]}),
        "ml_feature_list"
    )
    return X, y, feature_cols, categorical_cols


# =============================================================================
# STEP 4 — Train CatBoost
# =============================================================================

def train_catboost(X: pd.DataFrame, y: pd.Series,
                   categorical_cols: list):
    """
    Fit CatBoost binary classifier with class-weight balancing and early stopping.
    Returns model + split data for downstream evaluation.
    """
    try:
        from catboost import CatBoostClassifier, Pool
    except ImportError:
        raise ImportError("pip install catboost")

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score,
        precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score,
        confusion_matrix, classification_report,
    )

    print("\n[4] Training CatBoost classifier...")

    cat_idx = [X.columns.get_loc(c) for c in categorical_cols if c in X.columns]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    pos_weight = max(1.0, (y_train == 0).sum() / max((y_train == 1).sum(), 1))

    model = CatBoostClassifier(
        iterations=600,
        learning_rate=0.03,
        depth=4,
        loss_function="Logloss",
        eval_metric="AUC",
        class_weights=[1.0, pos_weight],
        random_seed=42,
        verbose=False,
        early_stopping_rounds=60,
    )

    train_pool = Pool(X_train, y_train, cat_features=cat_idx)
    test_pool  = Pool(X_test,  y_test,  cat_features=cat_idx)
    model.fit(train_pool, eval_set=test_pool)

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy":          round(accuracy_score(y_test, pred),          4),
        "balanced_accuracy": round(balanced_accuracy_score(y_test, pred), 4),
        "precision":         round(precision_score(y_test, pred, zero_division=0), 4),
        "recall":            round(recall_score(y_test, pred, zero_division=0),    4),
        "f1":                round(f1_score(y_test, pred, zero_division=0),        4),
        "roc_auc":           round(roc_auc_score(y_test, prob),                   4),
        "pr_auc":            round(average_precision_score(y_test, prob),          4),
        "n_train":           int(len(y_train)),
        "n_test":            int(len(y_test)),
        "best_iteration":    int(model.get_best_iteration() or 0),
        "pos_class_weight":  round(pos_weight, 3),
    }

    _save_csv(pd.DataFrame([metrics]), "catboost_performance")

    cm = confusion_matrix(y_test, pred)
    _save_csv(
        pd.DataFrame(cm,
                     index=["actual_low", "actual_high"],
                     columns=["pred_low", "pred_high"]),
        "confusion_matrix"
    )

    report = classification_report(y_test, pred, output_dict=True)
    _save_csv(pd.DataFrame(report).T.reset_index(names=["class"]),
              "classification_report")

    print("  Performance metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"    {k:<22s}: {v:.4f}")
        else:
            print(f"    {k:<22s}: {v}")

    return model, X_train, X_test, y_train, y_test, pred, prob, cat_idx


# =============================================================================
# STEP 5 — SHAP explainability
# =============================================================================

def run_shap(model, X_test: pd.DataFrame, cat_idx: list):
    """
    Compute SHAP values using CatBoost's native implementation for reliability.
    Falls back to shap.TreeExplainer if native method fails.
    """
    try:
        import shap as _shap
    except ImportError:
        raise ImportError("pip install shap")

    from catboost import Pool

    print("\n[5] Running SHAP explainability...")

    test_pool = Pool(X_test, cat_features=cat_idx)

    # Use CatBoost native SHAP (most reliable with categorical features)
    try:
        sv_raw = model.get_feature_importance(test_pool, type="ShapValues")
        shap_values = sv_raw[:, :-1]   # last column = base value (expected value)
        base_value  = sv_raw[0, -1]
        print(f"  SHAP base value (expected log-odds): {base_value:.4f}")
    except Exception as e:
        print(f"  Native SHAP failed ({e}), falling back to shap.TreeExplainer")
        explainer   = _shap.TreeExplainer(model)
        sv_raw      = explainer.shap_values(X_test)
        shap_values = sv_raw[1] if isinstance(sv_raw, list) else sv_raw
        base_value  = 0.0

    mean_abs = np.abs(shap_values).mean(axis=0)

    importance_df = pd.DataFrame({
        "feature":        X_test.columns,
        "mean_abs_shap":  mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    importance_df["rank"] = np.arange(1, len(importance_df) + 1)

    _save_csv(importance_df, "shap_feature_importance")

    # ── Figure A: SHAP beeswarm summary ──────────────────────────────────────
    try:
        expl = _shap.Explanation(
            values=shap_values,
            base_values=np.full(len(X_test), base_value),
            data=X_test.values,
            feature_names=list(X_test.columns),
        )
        fig, ax = plt.subplots(figsize=(10, 7))
        plt.sca(ax)
        _shap.plots.beeswarm(expl, max_display=20, show=False, color_bar=True)
        ax.set_title("SHAP Beeswarm: High Thermal Discomfort Drivers",
                     fontsize=12, fontweight="bold")
        _save_fig(fig, "shap_beeswarm")
    except Exception:
        # Fallback: dot plot via summary_plot
        fig = plt.figure(figsize=(10, 7))
        _shap.summary_plot(shap_values, X_test, max_display=20, show=False)
        plt.title("SHAP Summary: High Thermal Discomfort Drivers",
                  fontsize=12, fontweight="bold")
        _save_fig(fig, "shap_beeswarm")

    # ── Figure B: Mean |SHAP| bar chart (custom) ──────────────────────────────
    top = importance_df.head(15).iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_facecolor("white"); fig.patch.set_facecolor("white")
    bars = ax.barh(top["feature"], top["mean_abs_shap"],
                   color=_PALETTE["bar"], edgecolor="white", linewidth=0.6)
    for bar, v in zip(bars, top["mean_abs_shap"]):
        ax.text(v + 0.0005, bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}", va="center", fontsize=8.5)
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title("Top 15 SHAP Feature Importances\n"
                 "(mean absolute SHAP value across test set)",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="x", color=_PALETTE["grid"], lw=0.7)
    plt.tight_layout()
    _save_fig(fig, "shap_bar_importance")

    print(f"  Top-5 features by mean |SHAP|:")
    for _, row in importance_df.head(5).iterrows():
        print(f"    {int(row['rank']):2d}. {row['feature']:<30s} {row['mean_abs_shap']:.5f}")

    return shap_values, importance_df


# =============================================================================
# STEP 6 — Performance figures
# =============================================================================

def plot_performance_figures(y_test, pred, prob):
    """Confusion matrix, ROC curve, PR curve, predicted probability distribution."""
    from sklearn.metrics import (
        confusion_matrix, roc_curve, auc,
        precision_recall_curve, average_precision_score,
    )

    print("\n[6] Saving performance figures...")

    # ── Fig 1: Confusion matrix ───────────────────────────────────────────────
    cm = confusion_matrix(y_test, pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    cax = ax.imshow(cm, cmap="Blues")
    fig.colorbar(cax)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred: Low", "Pred: High"])
    ax.set_yticklabels(["Actual: Low", "Actual: High"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=14, fontweight="bold",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title("Confusion Matrix — High Discomfort Prediction",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    _save_fig(fig, "confusion_matrix")

    # ── Fig 2: ROC curve ──────────────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_test, prob)
    roc_auc     = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.plot(fpr, tpr, color="#2980B9", lw=2,
            label=f"CatBoost  AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color="#95A5A6", lw=1, linestyle="--", label="Random")
    ax.fill_between(fpr, tpr, alpha=0.08, color="#2980B9")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curve — High Thermal Discomfort Prediction",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(color=_PALETTE["grid"], lw=0.7)
    plt.tight_layout()
    _save_fig(fig, "roc_curve")

    # ── Fig 3: Precision-Recall curve ─────────────────────────────────────────
    prec, rec, _ = precision_recall_curve(y_test, prob)
    pr_auc       = average_precision_score(y_test, prob)
    baseline     = y_test.mean()
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.plot(rec, prec, color="#8E44AD", lw=2,
            label=f"CatBoost  AP = {pr_auc:.3f}")
    ax.axhline(baseline, color="#95A5A6", lw=1, linestyle="--",
               label=f"Baseline (prevalence = {baseline:.2f})")
    ax.fill_between(rec, prec, alpha=0.08, color="#8E44AD")
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision-Recall Curve — High Thermal Discomfort",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(color=_PALETTE["grid"], lw=0.7)
    plt.tight_layout()
    _save_fig(fig, "pr_curve")

    # ── Fig 4: Predicted probability distribution ─────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.hist(prob[y_test == 0], bins=30, alpha=0.65,
            color=_PALETTE["neg"], label="Low discomfort (actual)", edgecolor="white")
    ax.hist(prob[y_test == 1], bins=30, alpha=0.65,
            color=_PALETTE["pos"], label="High discomfort (actual)", edgecolor="white")
    ax.set_xlabel("Predicted probability of high discomfort", fontsize=11)
    ax.set_ylabel("Households (test set)", fontsize=11)
    ax.set_title("Predicted Risk Score Distribution by True Label",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(color=_PALETTE["grid"], lw=0.7)
    plt.tight_layout()
    _save_fig(fig, "predicted_risk_distribution")


# =============================================================================
# STEP 7 — SHAP dependence plots for top features
# =============================================================================

def plot_shap_dependence(shap_values: np.ndarray,
                         X_test: pd.DataFrame,
                         importance_df: pd.DataFrame,
                         n_top: int = 3) -> None:
    """SHAP value vs raw feature value for the top-n most important features."""
    try:
        import shap as _shap
    except ImportError:
        return

    print("\n[7] Plotting SHAP dependence charts...")

    top_feats = importance_df.head(n_top)["feature"].tolist()
    fig, axes = plt.subplots(1, n_top, figsize=(5 * n_top, 4))
    if n_top == 1:
        axes = [axes]
    fig.patch.set_facecolor("white")

    for ax, feat in zip(axes, top_feats):
        if feat not in X_test.columns:
            continue
        fidx = list(X_test.columns).index(feat)
        xv   = X_test[feat].values
        sv   = shap_values[:, fidx]

        # Colour by second most important numeric feature
        try:
            ax.scatter(xv, sv, c=sv, cmap="coolwarm", alpha=0.55,
                       edgecolors="none", s=18)
        except Exception:
            ax.scatter(xv, sv, alpha=0.55, s=18)

        ax.axhline(0, color="#95A5A6", lw=0.8, linestyle="--")
        ax.set_xlabel(feat, fontsize=10)
        ax.set_ylabel("SHAP value", fontsize=10)
        ax.set_title(f"{feat}", fontsize=11, fontweight="bold")
        ax.set_facecolor("white")
        ax.grid(color=_PALETTE["grid"], lw=0.6)

    fig.suptitle("SHAP Dependence Plots — Top Features", fontsize=13,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    _save_fig(fig, "shap_dependence_top3")


# =============================================================================
# STEP 8 — Save full-sample predictions
# =============================================================================

def save_predictions(df: pd.DataFrame, model,
                     feature_cols: list, categorical_cols: list,
                     cat_idx: list) -> None:
    """
    Score all households (not just test set) and append predictions to df.
    Uses the same feature/encoding logic as training.
    """
    from catboost import Pool

    print("\n[8] Saving full-sample predictions...")

    X_full = df[feature_cols].copy()

    for c in X_full.select_dtypes(include="bool").columns:
        X_full[c] = X_full[c].astype(int)
    for c in X_full.select_dtypes(include=[np.number]).columns:
        if X_full[c].isna().any():
            X_full[c] = X_full[c].fillna(X_full[c].median())
    for c in X_full.select_dtypes(include="object").columns:
        X_full[c] = X_full[c].fillna("unknown")

    pool_full = Pool(X_full, cat_features=cat_idx)
    prob_full = model.predict_proba(pool_full)[:, 1]
    pred_full = model.predict(pool_full)

    out = df.copy()
    out["ml_discomfort_probability"] = prob_full.round(5)
    out["ml_discomfort_prediction"]  = pred_full

    out_path = OUT_DIR / "enable_ml_predictions.csv"
    out.to_csv(out_path, index=False)
    print(f"  [saved] enable_ml_predictions.csv  ({len(out):,} rows x {out.shape[1]} cols)")


# =============================================================================
# STEP 9 — README
# =============================================================================

def write_readme() -> None:
    content = """\
# ML + SHAP Stage — High Thermal Discomfort Prediction

## Overview

This stage uses **CatBoost** gradient boosting to predict whether a UK household
falls in the top quartile of thermal discomfort (`high_discomfort = 1`), based
on COR-derived behavioural scores and household characteristics.

**SHAP (SHapley Additive exPlanations)** then explains which features drive the
prediction for individual households and on average across the test set.

## Target variable

| Variable | Definition |
|----------|-----------|
| `high_discomfort` | 1 if `discomfort_score >= 75th percentile`, else 0 |

Rows with missing `discomfort_score` (18 of 1,015 in the UK sub-sample) are
excluded from training and evaluation.

## Predictors

| Group | Features |
|-------|---------|
| COR composite scores | `insecurity_score`, `preservation_score` |
| Energy poverty | `risk_category` (categorical), `low_income_flag`, `high_cost_flag` |
| Household controls | `S8`, `H1`, `H2`, `H3` |
| Normalised item scores | `n_E2A`, `n_E2B`, `n_E7A-E7E`, `n_H9`, `n_E5A1-E5A5`, `n_E5A9`, `n_E6A1-E6A8` |

**Excluded from ML features:**
- `fes_core / fes_macro / fes_actual` — annual UK-level constants (zero within-sample variance)
- `discomfort_score` — direct target leakage
- `n_H12A, n_H12B, n_H15A-E` — discomfort item scores (target leakage)
- `n_S8` — redundant with raw `S8` already included as control

## Model configuration

| Parameter | Value |
|-----------|-------|
| Algorithm | CatBoostClassifier |
| Iterations | 600 (with early stopping, patience=60) |
| Learning rate | 0.03 |
| Tree depth | 4 |
| Eval metric | AUC |
| Class weights | Balanced by inverse frequency |
| Train/test split | 75 / 25 (stratified, seed=42) |

## Outputs

### Tables

| File | Description |
|------|-------------|
| `ml_feature_list.csv` | Features used with categorical flag |
| `catboost_performance.csv` | Accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, PR-AUC |
| `confusion_matrix.csv` | 2x2 test-set confusion matrix |
| `classification_report.csv` | Per-class precision, recall, F1 |
| `shap_feature_importance.csv` | Features ranked by mean |SHAP| |

### Figures

| File | Description |
|------|-------------|
| `shap_beeswarm.png` | SHAP beeswarm — feature impact distribution |
| `shap_bar_importance.png` | Mean |SHAP| bar chart (top 15 features) |
| `shap_dependence_top3.png` | SHAP dependence plots for top-3 features |
| `confusion_matrix.png` | Confusion matrix heatmap |
| `roc_curve.png` | ROC curve with AUC |
| `pr_curve.png` | Precision-Recall curve with Average Precision |
| `predicted_risk_distribution.png` | Probability distribution by true label |

## Interpretation

SHAP values represent each feature's contribution to the log-odds of the
prediction for each household.  A positive SHAP value pushes the model toward
predicting `high_discomfort = 1`; a negative SHAP value pushes toward 0.

The beeswarm plot shows the distribution of SHAP values across households for
each feature, colour-coded by feature value (red = high, blue = low).
"""
    p = OUT_DIR / "README_ml_shap.md"
    p.write_text(content, encoding="utf-8")
    print(f"  [saved] README_ml_shap.md")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    _mkdir()

    print("=" * 70)
    print("  ML + SHAP PIPELINE: COR FEATURES -> HIGH DISCOMFORT RISK")
    print("=" * 70)

    df = load_data()
    df = create_target(df)
    X, y, feature_cols, categorical_cols = prepare_features(df)

    model, X_train, X_test, y_train, y_test, pred, prob, cat_idx = (
        train_catboost(X, y, categorical_cols)
    )

    shap_values, importance_df = run_shap(model, X_test, cat_idx)
    plot_performance_figures(y_test, pred, prob)
    plot_shap_dependence(shap_values, X_test, importance_df)
    save_predictions(df, model, feature_cols, categorical_cols, cat_idx)
    write_readme()

    print(f"\n{'='*70}")
    print("  PIPELINE COMPLETE")
    print(f"{'='*70}")
    print(f"  Outputs: {OUT_DIR}")
    print(f"  Tables : {TABLE_DIR}")
    print(f"  Figures: {FIG_DIR}")


if __name__ == "__main__":
    main()
