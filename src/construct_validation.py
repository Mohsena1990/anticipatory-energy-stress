"""
construct_validation.py
────────────────────────
Psychometric validation of the four revised COR-based constructs.

Reliability
  • Cronbach's alpha
  • McDonald's omega (from EFA factor loadings)
  • Composite Reliability (CR)

Convergent validity
  • Factor loadings (EFA / PCA)
  • Average Variance Extracted (AVE)

Discriminant validity
  • Fornell–Larcker criterion  (AVE > squared inter-construct correlation)
  • HTMT ratio  (average inter-construct item correlation / average
                  within-construct item correlation)

Factor structure
  • Exploratory Factor Analysis (scikit-learn + factor_analyzer)
  • Correlation matrix (Pearson)

All results are exported to outputs/construct_validation/tables/.
Summary figures go to outputs/construct_validation/figures/.

Usage
─────
  from src.construct_validation import run
  results = run(df)   # df from enable_preprocessing.run()
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
from scipy import stats

warnings.filterwarnings("ignore")

from src import config, paths
from src.construct_mapping import CONSTRUCT_REGISTRY

log = logging.getLogger(__name__)


# =============================================================================
# I/O helpers
# =============================================================================

def _save_csv(df: pd.DataFrame, name: str) -> None:
    paths.CV_TABLES.mkdir(parents=True, exist_ok=True)
    p = paths.CV_TABLES / f"{name}.csv"
    df.to_csv(p, index=False)
    log.info("Saved %s", p.name)


def _save_fig(fig: plt.Figure, name: str) -> None:
    paths.CV_FIGURES.mkdir(parents=True, exist_ok=True)
    p = paths.CV_FIGURES / f"{name}.{config.FIGURE_FORMAT}"
    fig.savefig(p, dpi=config.DPI, bbox_inches="tight")
    fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    log.info("Saved figure %s", p.name)


# =============================================================================
# Reliability
# =============================================================================

def cronbach_alpha(item_matrix: pd.DataFrame) -> float:
    """
    Compute Cronbach's alpha from an n_respondents × n_items matrix of
    normalized item scores.  NaN values are row-wise dropped.
    """
    data = item_matrix.dropna(how="any")
    k = data.shape[1]
    if k < 2:
        return np.nan
    item_vars = data.var(axis=0, ddof=1)
    total_var = data.sum(axis=1).var(ddof=1)
    if total_var < 1e-12:
        return np.nan
    alpha = (k / (k - 1)) * (1.0 - item_vars.sum() / total_var)
    return float(np.clip(alpha, -1.0, 1.0))


def mcdonald_omega(loadings: np.ndarray, item_matrix: pd.DataFrame) -> float:
    """
    Estimate McDonald's omega from a vector of (uni-dimensional) factor loadings.

    omega = (sum_i lambda_i)^2 / [(sum_i lambda_i)^2 + sum_i (1 - lambda_i^2)]

    This is the reliability of the unit-weighted composite when the items
    share a single dominant factor.
    """
    data = item_matrix.dropna(how="any")
    if data.shape[1] < 2 or len(loadings) != data.shape[1]:
        return np.nan
    l = np.array(loadings, dtype=float)
    sum_l   = np.sum(l)
    sum_err = np.sum(1.0 - l ** 2)
    denom   = sum_l ** 2 + sum_err
    if denom < 1e-12:
        return np.nan
    return float(sum_l ** 2 / denom)


def composite_reliability(loadings: np.ndarray) -> float:
    """
    Compute Composite Reliability (Jöreskog's rho_c) from standardized
    factor loadings:

      CR = (sum lambda_i)^2 / [(sum lambda_i)^2 + sum (1 - lambda_i^2)]
    """
    # Identical formula to McDonald's omega for a single-factor CFA
    return mcdonald_omega(loadings, pd.DataFrame(np.ones((2, len(loadings)))))


def average_variance_extracted(loadings: np.ndarray) -> float:
    """
    Compute Average Variance Extracted (AVE):

      AVE = mean(lambda_i^2)
    """
    l = np.array(loadings, dtype=float)
    if len(l) == 0:
        return np.nan
    return float(np.mean(l ** 2))


# =============================================================================
# Factor analysis (EFA)
# =============================================================================

def run_efa(
    item_matrix: pd.DataFrame,
    n_factors: int = 1,
    rotation: str = "varimax",
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Run Exploratory Factor Analysis using factor_analyzer if available,
    falling back to PCA-based loading approximation.

    Returns
    -------
    loadings : np.ndarray  shape (n_items, n_factors)
    communalities : np.ndarray
    explained_variance_ratio : float  (first factor, total if n_factors=1)
    """
    data = item_matrix.dropna(how="any")
    if data.shape[0] < 10 or data.shape[1] < 2:
        log.warning("Insufficient data for EFA (%d rows, %d items)", *data.shape)
        return np.full((data.shape[1], n_factors), np.nan), np.full(data.shape[1], np.nan), np.nan

    try:
        from factor_analyzer import FactorAnalyzer
        fa = FactorAnalyzer(n_factors=n_factors, rotation=rotation, method="ml")
        fa.fit(data)
        loadings      = fa.loadings_
        communalities = fa.get_communalities()
        ev, _         = fa.get_eigenvalues()
        evr           = float(ev[:n_factors].sum() / ev.sum()) if ev.sum() > 0 else np.nan
        return loadings, communalities, evr
    except ImportError:
        pass

    # Fallback: PCA-based loading approximation
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(n_factors, data.shape[1]))
    pca.fit(data.values)
    # Scale eigenvectors by sqrt(eigenvalue) to approximate factor loadings
    evr = float(pca.explained_variance_ratio_[:n_factors].sum())
    loadings = (pca.components_.T *
                np.sqrt(pca.explained_variance_[:n_factors]))
    communalities = np.sum(loadings ** 2, axis=1)
    log.info("EFA fallback: using PCA-based loading approximation")
    return loadings, communalities, evr


def extract_first_factor_loadings(loadings: np.ndarray) -> np.ndarray:
    """Return the first factor column from a loadings matrix."""
    if loadings.ndim == 1:
        return loadings
    return loadings[:, 0]


# =============================================================================
# Discriminant validity
# =============================================================================

def htmt_ratio(
    items_a: pd.DataFrame,
    items_b: pd.DataFrame,
) -> float:
    """
    Compute the HTMT (Heterotrait–Monotrait) ratio between two constructs.

      HTMT = mean(|r_AB|) / sqrt(mean(|r_AA|) * mean(|r_BB|))

    where r_AB are cross-construct correlations, r_AA / r_BB are
    within-construct correlations (excluding self-correlations).

    Values below 0.85 support discriminant validity (Henseler et al. 2015).
    """
    def _mean_abs_corr(m: pd.DataFrame) -> float:
        c = m.dropna(how="any").corr().values
        k = c.shape[0]
        if k < 2:
            return np.nan
        upper = c[np.triu_indices(k, k=1)]
        return float(np.nanmean(np.abs(upper)))

    def _cross_mean_abs_corr(m1: pd.DataFrame, m2: pd.DataFrame) -> float:
        combined = pd.concat([m1, m2], axis=1).dropna(how="any")
        n1 = m1.shape[1]
        full_corr = combined.corr().values
        cross = full_corr[:n1, n1:]
        return float(np.nanmean(np.abs(cross)))

    aa = _mean_abs_corr(items_a)
    bb = _mean_abs_corr(items_b)
    ab = _cross_mean_abs_corr(items_a, items_b)

    denom = np.sqrt(aa * bb)
    if denom < 1e-12 or np.isnan(denom):
        return np.nan
    return float(ab / denom)


def fornell_larcker_check(ave_dict: dict[str, float], corr_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Fornell–Larcker criterion: AVE of each construct should exceed the squared
    correlation with every other construct.

    Returns a DataFrame with True/False for each construct pair.
    """
    constructs = list(ave_dict.keys())
    rows = []
    for c1 in constructs:
        for c2 in constructs:
            if c1 == c2:
                continue
            r = corr_matrix.loc[c1, c2] if (c1 in corr_matrix.index and
                                              c2 in corr_matrix.columns) else np.nan
            r2   = r ** 2 if not np.isnan(r) else np.nan
            ave1 = ave_dict.get(c1, np.nan)
            rows.append({
                "construct_i":  c1,
                "construct_j":  c2,
                "ave_i":        round(float(ave1), 4) if not np.isnan(ave1) else np.nan,
                "r_ij":         round(float(r), 4) if not np.isnan(r) else np.nan,
                "r_ij_squared": round(float(r2), 4) if not np.isnan(r2) else np.nan,
                "fl_satisfied": bool(ave1 > r2) if not np.isnan(ave1) and not np.isnan(r2) else None,
            })
    return pd.DataFrame(rows)


# =============================================================================
# Construct-level correlation matrix
# =============================================================================

def construct_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return Pearson correlation matrix of the four construct scores."""
    score_cols = [c for c in
                  ["fcp_score", "aemc_score", "bli_score", "tcr_score"]
                  if c in df.columns]
    if not score_cols:
        return pd.DataFrame()
    corr = df[score_cols].corr()
    rename = {
        "fcp_score":  "FCP",
        "aemc_score": "AEMC",
        "bli_score":  "BLI",
        "tcr_score":  "TCR",
    }
    corr = corr.rename(index=rename, columns=rename)
    return corr


# =============================================================================
# Figures
# =============================================================================

def _plot_factor_loadings(
    loading_df: pd.DataFrame,
    dest_dir,
) -> None:
    """Bar chart of factor loadings per construct."""
    constructs = loading_df["construct"].unique()
    fig, axes = plt.subplots(1, len(constructs), figsize=(4 * len(constructs), 5))
    if len(constructs) == 1:
        axes = [axes]
    palette = {"FCP": "#E74C3C", "AEMC": "#2980B9", "BLI": "#E67E22", "TCR": "#8E44AD"}
    for ax, c in zip(axes, constructs):
        sub = loading_df[loading_df["construct"] == c].copy()
        color = palette.get(c, "#999")
        ax.barh(sub["variable"], sub["loading_f1"].abs(), color=color, alpha=0.85)
        ax.axvline(config.MIN_LOADING, color="red", ls="--", lw=1,
                   label=f"min={config.MIN_LOADING}")
        ax.set_xlim(0, 1.05)
        ax.set_title(f"{c}\n{CONSTRUCT_REGISTRY.get(c, {}).get('label', c)}", fontsize=9)
        ax.set_xlabel("|Loading|")
        ax.legend(fontsize=7)
    fig.suptitle("Factor loadings by construct (EFA, first factor)", fontsize=11)
    fig.tight_layout()
    _save_fig(fig, "factor_loadings")


def _plot_reliability_avecr(rel_df: pd.DataFrame) -> None:
    """Bar chart comparing alpha, omega, CR, and AVE across constructs."""
    metrics = ["cronbach_alpha", "mcdonald_omega", "composite_reliability", "ave"]
    labels  = ["Cronbach α", "McDonald ω", "CR", "AVE"]
    n = len(rel_df)
    fig, ax = plt.subplots(figsize=(8, 5))
    x    = np.arange(n)
    w    = 0.18
    colors = ["#2980B9", "#E67E22", "#27AE60", "#E74C3C"]
    for i, (m, lab) in enumerate(zip(metrics, labels)):
        if m in rel_df.columns:
            vals = rel_df[m].values.astype(float)
            ax.bar(x + i * w, vals, width=w, label=lab, color=colors[i], alpha=0.85)
    ax.axhline(config.MIN_ALPHA, color="#E74C3C", ls="--", lw=1, label=f"α≥{config.MIN_ALPHA}")
    ax.axhline(config.MIN_AVE, color="#8E44AD", ls=":", lw=1, label=f"AVE≥{config.MIN_AVE}")
    ax.set_xticks(x + w * 1.5)
    ax.set_xticklabels(rel_df["construct"].tolist(), rotation=15)
    ax.set_ylabel("Value")
    ax.set_ylim(0, 1.05)
    ax.set_title("Reliability and validity metrics by construct")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save_fig(fig, "reliability_ave_cr")


def _plot_htmt(htmt_df: pd.DataFrame) -> None:
    """Heatmap of HTMT ratios."""
    constructs = list(htmt_df["construct_i"].unique())
    n = len(constructs)
    if n == 0:
        return
    mat = np.full((n, n), np.nan)
    idx = {c: i for i, c in enumerate(constructs)}
    for _, row in htmt_df.iterrows():
        i, j = idx.get(row["construct_i"]), idx.get(row["construct_j"])
        if i is not None and j is not None:
            mat[i, j] = row["htmt"]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(constructs, rotation=30, ha="right")
    ax.set_yticklabels(constructs)
    for i in range(n):
        for j in range(n):
            v = mat[i, j]
            if not np.isnan(v):
                color = "white" if v > 0.7 else "black"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color=color, fontsize=9)
    plt.colorbar(im, ax=ax, label="HTMT ratio")
    ax.set_title(f"HTMT matrix (threshold < {config.MAX_HTMT})")
    fig.tight_layout()
    _save_fig(fig, "htmt_matrix")


def _plot_corr_matrix(corr: pd.DataFrame) -> None:
    if corr.empty:
        return
    n = len(corr)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr.values, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=30, ha="right")
    ax.set_yticklabels(corr.index)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                    fontsize=9, color="black")
    plt.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title("Construct-level correlation matrix")
    fig.tight_layout()
    _save_fig(fig, "construct_correlation_matrix")


# =============================================================================
# Main entry point
# =============================================================================

def run(df: pd.DataFrame) -> dict:
    """
    Run construct validation on the four revised COR constructs.

    Parameters
    ----------
    df : pd.DataFrame from enable_preprocessing.run()

    Returns
    -------
    dict with keys: 'reliability', 'loadings', 'htmt', 'fl_check', 'corr'
    """
    paths.CV_TABLES.mkdir(parents=True, exist_ok=True)
    paths.CV_FIGURES.mkdir(parents=True, exist_ok=True)

    reg = CONSTRUCT_REGISTRY

    # ── Per-construct analysis ────────────────────────────────────────────────
    rel_rows      = []
    loading_rows  = []
    htmt_data     = {}  # construct → item matrix of normalized scores
    ave_dict      = {}

    for short, info in reg.items():
        label     = info["label"]
        core_cols = info["core_items"]

        # Collect available normalized columns
        n_cols = [f"n_{c}" for c in core_cols if f"n_{c}" in df.columns]
        if not n_cols:
            log.warning("No normalized items found for %s — skipping", short)
            continue

        item_mat = df[n_cols].copy()
        htmt_data[short] = item_mat

        # Reliability
        alpha = cronbach_alpha(item_mat)

        # EFA (single factor)
        efa_loadings, communalities, evr = run_efa(item_mat, n_factors=1)
        f1 = extract_first_factor_loadings(efa_loadings)

        omega = mcdonald_omega(f1, item_mat)
        cr    = composite_reliability(f1)
        ave   = average_variance_extracted(f1)
        ave_dict[short] = ave

        n_items = len(n_cols)
        rel_rows.append({
            "construct":           short,
            "label":               label,
            "n_items":             n_items,
            "n_respondents":       int(item_mat.dropna(how="any").shape[0]),
            "cronbach_alpha":      round(float(alpha), 4) if not np.isnan(alpha) else np.nan,
            "mcdonald_omega":      round(float(omega), 4) if not np.isnan(omega) else np.nan,
            "composite_reliability": round(float(cr), 4) if not np.isnan(cr) else np.nan,
            "ave":                 round(float(ave), 4) if not np.isnan(ave) else np.nan,
            "efa_evr_f1":          round(float(evr), 4) if not np.isnan(evr) else np.nan,
            "alpha_ok":            bool(alpha >= config.MIN_ALPHA) if not np.isnan(alpha) else None,
            "ave_ok":              bool(ave >= config.MIN_AVE) if not np.isnan(ave) else None,
        })

        log.info("%-4s α=%.3f  ω=%.3f  CR=%.3f  AVE=%.3f",
                 short,
                 alpha if not np.isnan(alpha) else -99,
                 omega if not np.isnan(omega) else -99,
                 cr    if not np.isnan(cr)    else -99,
                 ave   if not np.isnan(ave)   else -99)

        # Factor loadings table
        raw_cols = [c.replace("n_", "") for c in n_cols]
        for var, lv in zip(raw_cols, f1):
            loading_rows.append({
                "construct":   short,
                "label":       label,
                "variable":    var,
                "loading_f1":  round(float(lv), 4) if not np.isnan(lv) else np.nan,
                "communality": round(float(communalities[raw_cols.index(var)]), 4)
                               if not np.isnan(communalities[raw_cols.index(var)]) else np.nan,
                "above_min":   bool(abs(lv) >= config.MIN_LOADING) if not np.isnan(lv) else None,
            })

    rel_df      = pd.DataFrame(rel_rows)
    loading_df  = pd.DataFrame(loading_rows)
    _save_csv(rel_df,     "construct_reliability")
    _save_csv(loading_df, "factor_loadings")

    # ── AVE / CR table ────────────────────────────────────────────────────────
    ave_cr_rows = []
    for r in rel_rows:
        ave_cr_rows.append({
            "construct": r["construct"],
            "label":     r["label"],
            "ave":       r["ave"],
            "cr":        r["composite_reliability"],
            "ave>=0.50": r["ave_ok"],
        })
    _save_csv(pd.DataFrame(ave_cr_rows), "ave_cr_table")

    # ── HTMT ─────────────────────────────────────────────────────────────────
    construct_names = list(htmt_data.keys())
    htmt_rows = []
    for i, c1 in enumerate(construct_names):
        for j, c2 in enumerate(construct_names):
            if i >= j:
                continue
            val = htmt_ratio(htmt_data[c1], htmt_data[c2])
            htmt_rows.append({
                "construct_i": c1,
                "construct_j": c2,
                "htmt":        round(float(val), 4) if not np.isnan(val) else np.nan,
                "htmt_ok":     bool(val < config.MAX_HTMT) if not np.isnan(val) else None,
            })
    htmt_df = pd.DataFrame(htmt_rows)
    _save_csv(htmt_df, "htmt_matrix")

    # ── Construct correlation matrix ─────────────────────────────────────────
    corr = construct_correlation_matrix(df)
    if not corr.empty:
        _save_csv(corr.reset_index().rename(columns={"index": "construct"}),
                  "construct_correlation_matrix")

    # ── Fornell–Larcker ───────────────────────────────────────────────────────
    if not corr.empty:
        fl_df = fornell_larcker_check(ave_dict, corr)
        _save_csv(fl_df, "fornell_larcker_check")
    else:
        fl_df = pd.DataFrame()

    # ── Figures ───────────────────────────────────────────────────────────────
    if not loading_df.empty:
        _plot_factor_loadings(loading_df, paths.CV_FIGURES)
    if not rel_df.empty:
        _plot_reliability_avecr(rel_df)
    if not htmt_df.empty:
        _plot_htmt(htmt_df)
    if not corr.empty:
        _plot_corr_matrix(corr)

    return {
        "reliability": rel_df,
        "loadings":    loading_df,
        "htmt":        htmt_df,
        "fl_check":    fl_df,
        "corr":        corr,
        "ave_dict":    ave_dict,
    }
