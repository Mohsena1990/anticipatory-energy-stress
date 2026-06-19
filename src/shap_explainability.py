"""
shap_explainability.py
───────────────────────
SHAP (SHapley Additive exPlanations) for the CatBoost HighAEV classifier.

Interpretation note
───────────────────
SHAP values explain the model's predictions — they quantify how much each
feature pushes a prediction above or below the model's base rate.

SHAP does NOT prove causal influence on energy vulnerability.  The correct
interpretation is: "which financial, behavioural, household, and
transition-attitude variables are the most informative predictors in the
model, and in which direction do they push predictions toward high AEV?"

Outputs
───────
  shap_feature_importance.csv   — features ranked by mean |SHAP|
  shap_beeswarm.png             — global SHAP distribution per feature
  shap_bar_importance.png       — top-N feature importance bar chart
  shap_dependence_top_features.png — dependence plots for top-3 features
  (household-level examples via log / CSV rows)

Usage
─────
  from src.shap_explainability import run
  run(model, X_test, y_test, cat_idx)  # from ml_classification.run()
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


# =============================================================================
# I/O helpers
# =============================================================================

def _save_csv(df: pd.DataFrame, name: str) -> None:
    paths.SHAP_TABLES.mkdir(parents=True, exist_ok=True)
    p = paths.SHAP_TABLES / f"{name}.csv"
    df.to_csv(p, index=False)
    log.info("Saved %s", p.name)


def _save_fig(fig: plt.Figure, name: str) -> None:
    paths.SHAP_FIGURES.mkdir(parents=True, exist_ok=True)
    p = paths.SHAP_FIGURES / f"{name}.{config.FIGURE_FORMAT}"
    fig.savefig(p, dpi=config.DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved figure %s", p.name)


# =============================================================================
# SHAP value computation
# =============================================================================

def compute_shap_values(model, X: pd.DataFrame, cat_idx: list[int]) -> np.ndarray:
    """
    Compute SHAP values using CatBoost's native implementation.
    Falls back to shap.TreeExplainer if native method raises an exception.

    Returns a 2D array of shape (n_samples, n_features).
    """
    # CatBoost native SHAP (most reliable for CatBoost models)
    try:
        from catboost import Pool
        pool = Pool(X, cat_features=cat_idx)
        shap_vals = model.get_feature_importance(pool, type="ShapValues")
        # ShapValues returns (n_samples, n_features + 1); last column is base value
        return shap_vals[:, :-1]
    except Exception as e:
        log.warning("CatBoost native SHAP failed: %s — trying shap.TreeExplainer", e)

    try:
        import shap
        explainer  = shap.TreeExplainer(model)
        shap_vals  = explainer.shap_values(X)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]   # class-1 SHAP values
        return shap_vals
    except ImportError:
        raise ImportError(
            "Install shap: pip install shap\n"
            "Or use CatBoost native SHAP by ensuring catboost is installed."
        )


# =============================================================================
# Feature importance table
# =============================================================================

def build_importance_table(
    shap_values: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """Return features ranked by mean absolute SHAP value."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({
        "feature":          feature_names,
        "mean_abs_shap":    np.round(mean_abs, 6),
        "mean_shap":        np.round(shap_values.mean(axis=0), 6),
        "positive_share":   np.round((shap_values > 0).mean(axis=0), 4),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


# =============================================================================
# Figures
# =============================================================================

def _plot_shap_bar(importance_df: pd.DataFrame, top_n: int = 15) -> None:
    """Horizontal bar chart of mean |SHAP| for top-N features."""
    top = importance_df.head(top_n).copy()
    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.45)))
    colors = ["#E74C3C" if v >= 0 else "#2980B9"
              for v in top["mean_shap"]]
    ax.barh(top["feature"][::-1], top["mean_abs_shap"][::-1],
            color=colors[::-1], alpha=0.85)
    ax.set_xlabel("Mean |SHAP value| (impact on model output)")
    ax.set_title(
        f"Top {top_n} features by SHAP importance\n"
        "(CatBoost HighAEV classifier)\n"
        "Note: SHAP explains model predictions, not causal influence"
    )
    fig.tight_layout()
    _save_fig(fig, "shap_bar_importance")


def _plot_beeswarm(shap_values: np.ndarray, X: pd.DataFrame, top_n: int = 15) -> None:
    """SHAP beeswarm plot (uses shap library if available)."""
    try:
        import shap
        mean_abs = np.abs(shap_values).mean(axis=0)
        top_idx  = np.argsort(mean_abs)[::-1][:top_n]
        top_vals = shap_values[:, top_idx]
        top_feat = X.iloc[:, top_idx].copy()

        fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.5)))
        shap_exp = shap.Explanation(
            values=top_vals,
            data=top_feat.values,
            feature_names=[X.columns[i] for i in top_idx],
        )
        shap.plots.beeswarm(shap_exp, max_display=top_n, show=False)
        ax = plt.gca()
        ax.set_title(
            "SHAP beeswarm: feature impact on High AEV prediction\n"
            "Each dot = one household; colour = feature value"
        )
        fig = plt.gcf()
        _save_fig(fig, "shap_beeswarm")
    except (ImportError, Exception) as e:
        log.info("Beeswarm fallback: using scatter approximation (%s)", e)
        _plot_beeswarm_fallback(shap_values, X, top_n)


def _plot_beeswarm_fallback(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    top_n: int = 15,
) -> None:
    """Simple scatter fallback when shap plotting API is unavailable."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    top_idx  = np.argsort(mean_abs)[::-1][:top_n]
    feat_names = [X.columns[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.5)))
    for row_i, feat_i in enumerate(top_idx):
        sv   = shap_values[:, feat_i]
        jitter = np.random.default_rng(feat_i).uniform(-0.3, 0.3, len(sv))
        fv   = X.iloc[:, feat_i].values
        norm_fv = (fv - np.nanmin(fv)) / (np.nanmax(fv) - np.nanmin(fv) + 1e-12)
        ax.scatter(sv, row_i + jitter,
                   c=norm_fv, cmap="RdYlBu_r", s=6, alpha=0.4, vmin=0, vmax=1)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(feat_names)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("SHAP value (impact on HighAEV prediction)")
    ax.set_title("SHAP beeswarm (top features): colour = feature value (blue=low, red=high)")
    fig.tight_layout()
    _save_fig(fig, "shap_beeswarm")


def _plot_dependence(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    feature_names: list[str],
    top_n: int = 3,
) -> None:
    """Scatter SHAP dependence plots for top-N features."""
    n    = min(top_n, len(feature_names))
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, feat in zip(axes, feature_names[:n]):
        if feat not in X.columns:
            continue
        fi  = list(X.columns).index(feat)
        fv  = X[feat].values
        sv  = shap_values[:, fi]
        ax.scatter(fv, sv, alpha=0.4, s=8, color="#2980B9")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xlabel(feat)
        ax.set_ylabel("SHAP value")
        ax.set_title(f"Dependence: {feat}")
    fig.suptitle("SHAP dependence plots (top features)", fontsize=11)
    fig.tight_layout()
    _save_fig(fig, "shap_dependence_top_features")


# =============================================================================
# Household-level examples
# =============================================================================

def household_examples(
    shap_values: np.ndarray,
    X: pd.DataFrame,
    y_test: pd.Series,
    y_prob: np.ndarray,
    n_examples: int = 5,
) -> None:
    """Log and save SHAP explanations for highest/lowest probability households."""
    if len(X) == 0:
        return

    idx_high = np.argsort(y_prob)[::-1][:n_examples]
    idx_low  = np.argsort(y_prob)[:n_examples]

    rows = []
    for label, indices in [("highest_prob", idx_high), ("lowest_prob", idx_low)]:
        for i in indices:
            row = {"example_type": label,
                   "predicted_prob": round(float(y_prob[i]), 4),
                   "actual_high_aev": int(y_test.iloc[i])}
            for feat, sv in zip(X.columns, shap_values[i]):
                row[f"shap_{feat}"] = round(float(sv), 5)
            rows.append(row)

    _save_csv(pd.DataFrame(rows), "shap_household_examples")


# =============================================================================
# Main entry point
# =============================================================================

def run(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_prob: np.ndarray,
    cat_idx: list[int],
    top_n: int = 15,
) -> None:
    """
    Run SHAP explainability for the CatBoost HighAEV classifier.

    Parameters
    ----------
    model      : fitted CatBoostClassifier
    X_test     : test feature matrix
    y_test     : true binary labels
    y_prob     : predicted probabilities (class 1)
    cat_idx    : indices of categorical features
    top_n      : number of top features to show in plots
    """
    paths.SHAP_TABLES.mkdir(parents=True, exist_ok=True)
    paths.SHAP_FIGURES.mkdir(parents=True, exist_ok=True)

    log.info("Computing SHAP values (n_test=%d, n_features=%d)...",
             len(X_test), X_test.shape[1])

    shap_vals = compute_shap_values(model, X_test, cat_idx)

    feature_names = list(X_test.columns)

    # Importance table
    imp = build_importance_table(shap_vals, feature_names)
    _save_csv(imp, "shap_feature_importance")

    top_features = imp["feature"].head(top_n).tolist()

    # Figures
    _plot_shap_bar(imp, top_n=top_n)
    _plot_beeswarm(shap_vals, X_test, top_n=top_n)
    _plot_dependence(shap_vals, X_test, top_features, top_n=3)

    # Household-level examples
    household_examples(shap_vals, X_test, y_test, y_prob, n_examples=5)

    log.info("SHAP explainability stream complete.")
    log.info("Top 5 features by mean |SHAP|:")
    for _, row in imp.head(5).iterrows():
        log.info("  %d. %-22s  mean|SHAP|=%.5f", row["rank"], row["feature"],
                 row["mean_abs_shap"])
