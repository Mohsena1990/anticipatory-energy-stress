"""
route_comparison.py
────────────────────
Cross-route comparison of the three COR estimation routes:

  Route 1 — COR Composite Route   : formative composites (src.enable_preprocessing)
  Route 2 — COR-Informed SEM      : reflective CFA/SEM latent variables (src.cor_sem)
  Route 3 — COR-Informed VAE      : COR-aligned deep VAE latent representations (src.cor_vae)

Each route estimates the same four COR dimensions (FCP, AEMC, BLI, TCR) and
its own Adaptive Energy Vulnerability outcome proxy (`{route}_aev_score` /
`{route}_high_aev`) from the same survey items, but with a genuinely
different estimation method. This module does not re-estimate anything —
each route already computes its own outcome proxy (Route 1: `aev_score`/
`high_aev` in `src.enable_preprocessing`; Route 2: `sem_aev_score`/
`sem_high_aev` in `src.cor_sem`; Route 3: `vae_aev_score`/`vae_high_aev` in
`src.cor_vae`, all via the shared `src.route_utils.compute_route_aev`
formula). It only consumes those already-computed per-household scores and
asks two questions:

  1. Per-construct agreement — how correlated (Pearson r + Spearman r) are
     the three routes' FCP/AEMC/BLI/TCR estimates with each other?
  2. Outcome agreement — how much do the three routes agree on which
     households are "High AEV"? (Pearson r on the continuous AEV proxy;
     Cohen's kappa + raw agreement rate on the binary HighAEV proxy)

FES is never part of this comparison — see `src.route_utils` for the
project-wide FES safeguard.

Graceful degradation
─────────────────────
If Route 2 and/or Route 3 scores are unavailable (fit failure, or Route 3
skipped via `--skip-vae`), the comparison automatically degrades to
whichever routes are present — Route 1 alone, or Route 1 vs Route 2.

Usage
─────
  from src.route_comparison import run
  run(df, route2_scores, route3_scores)
"""

from __future__ import annotations

import itertools
import logging
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings("ignore")

from src import config, paths
from src.route_utils import CONSTRUCTS as _CONSTRUCTS, sign_canonicalize_against_route1

log = logging.getLogger(__name__)

_ROUTE_LABELS = {"route1": "Route 1: Composite", "route2": "Route 2: SEM", "route3": "Route 3: VAE"}
_PALETTE = {"route1": "#2C3E50", "route2": "#2980B9", "route3": "#E67E22"}

_ROUTE1_CONSTRUCT_COLS = {"fcp_score": "FCP", "aemc_score": "AEMC", "bli_score": "BLI", "tcr_score": "TCR"}
_ROUTE2_CONSTRUCT_COLS = {
    "sem_fcp_latent": "FCP", "sem_aemc_latent": "AEMC", "sem_bli_latent": "BLI", "sem_tcr_latent": "TCR",
}
_ROUTE3_CONSTRUCT_COLS = {
    "vae_fcp_latent": "FCP", "vae_aemc_latent": "AEMC", "vae_bli_latent": "BLI", "vae_tcr_latent": "TCR",
}


# =============================================================================
# I/O helpers
# =============================================================================

def _save_csv(df: pd.DataFrame, name: str) -> None:
    paths.ROUTE_COMPARISON_TABLES.mkdir(parents=True, exist_ok=True)
    p = paths.ROUTE_COMPARISON_TABLES / f"{name}.csv"
    df.to_csv(p, index=False)
    log.info("Saved %s", p.name)


def _save_fig(fig: plt.Figure, name: str) -> None:
    paths.ROUTE_COMPARISON_FIGURES.mkdir(parents=True, exist_ok=True)
    p = paths.ROUTE_COMPARISON_FIGURES / f"{name}.{config.FIGURE_FORMAT}"
    fig.savefig(p, dpi=config.DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved figure %s", p.name)


# =============================================================================
# Build each route's per-construct scores + AEV outcome proxy on a common frame
# =============================================================================

def build_route_frame(
    df: pd.DataFrame,
    route2_scores: pd.DataFrame = None,
    route3_scores: pd.DataFrame = None,
) -> dict[str, pd.DataFrame]:
    """
    Returns {"route1": df6, "route2": df6, "route3": df6}, each a DataFrame
    with columns [FCP, AEMC, BLI, TCR, aev_proxy, high_aev_proxy] indexed
    like `df`. Route 2/Route 3 frames are all-NaN when that route's scores
    were not supplied (fit failure or skipped).

    Route 2/3 construct columns (not the outcome columns) are sign-
    canonicalized against Route 1's composites before being returned, since
    a CFA loading's sign (Route 2) or a VAE alignment correlation's sign
    (Route 3) is statistically free — see `src.route_utils`. Each route's
    own aev_proxy/high_aev_proxy was already computed with this
    canonicalization applied internally (see `src.cor_sem`/`src.cor_vae`),
    so it is used as-is here.
    """
    empty = lambda: pd.DataFrame(
        np.nan, index=df.index, columns=_CONSTRUCTS + ["aev_proxy", "high_aev_proxy"]
    )

    r1 = df.reindex(columns=list(_ROUTE1_CONSTRUCT_COLS) + ["aev_score", "high_aev"])
    r1 = r1.rename(columns={**_ROUTE1_CONSTRUCT_COLS, "aev_score": "aev_proxy", "high_aev": "high_aev_proxy"})

    if route2_scores is not None:
        r2 = route2_scores.reindex(df.index).reindex(
            columns=list(_ROUTE2_CONSTRUCT_COLS) + ["sem_aev_score", "sem_high_aev"]
        )
        r2 = r2.rename(columns={**_ROUTE2_CONSTRUCT_COLS,
                                 "sem_aev_score": "aev_proxy", "sem_high_aev": "high_aev_proxy"})
        r2[_CONSTRUCTS] = sign_canonicalize_against_route1(r2[_CONSTRUCTS], r1[_CONSTRUCTS])
    else:
        r2 = empty()

    if route3_scores is not None:
        r3 = route3_scores.reindex(df.index).reindex(
            columns=list(_ROUTE3_CONSTRUCT_COLS) + ["vae_aev_score", "vae_high_aev"]
        )
        r3 = r3.rename(columns={**_ROUTE3_CONSTRUCT_COLS,
                                 "vae_aev_score": "aev_proxy", "vae_high_aev": "high_aev_proxy"})
        r3[_CONSTRUCTS] = sign_canonicalize_against_route1(r3[_CONSTRUCTS], r1[_CONSTRUCTS])
    else:
        r3 = empty()

    return {"route1": r1, "route2": r2, "route3": r3}


def _available_routes(route2_scores, route3_scores) -> list[str]:
    routes = ["route1"]
    if route2_scores is not None:
        routes.append("route2")
    if route3_scores is not None:
        routes.append("route3")
    return routes


# =============================================================================
# Per-construct cross-route correlation (Pearson + Spearman)
# =============================================================================

def compare_constructs(frames: dict[str, pd.DataFrame], available: list[str]) -> pd.DataFrame:
    rows = []
    for a, b in itertools.combinations(available, 2):
        fa, fb = frames[a], frames[b]
        for construct in _CONSTRUCTS:
            mask = fa[construct].notna() & fb[construct].notna()
            n = int(mask.sum())
            if n < 10:
                r_p, p_p, r_s, p_s = np.nan, np.nan, np.nan, np.nan
            else:
                r_p, p_p = pearsonr(fa.loc[mask, construct], fb.loc[mask, construct])
                r_s, p_s = spearmanr(fa.loc[mask, construct], fb.loc[mask, construct])
            rows.append({
                "route_a": _ROUTE_LABELS[a], "route_b": _ROUTE_LABELS[b],
                "construct": construct, "n": n,
                "pearson_r": round(float(r_p), 4) if pd.notna(r_p) else np.nan,
                "pearson_p": round(float(p_p), 4) if pd.notna(p_p) else np.nan,
                "spearman_r": round(float(r_s), 4) if pd.notna(r_s) else np.nan,
                "spearman_p": round(float(p_s), 4) if pd.notna(p_s) else np.nan,
                "agreement": (
                    "strong" if pd.notna(r_p) and abs(r_p) >= 0.5 else
                    "moderate" if pd.notna(r_p) and abs(r_p) >= 0.3 else
                    "weak" if pd.notna(r_p) else "n/a"
                ),
            })
    return pd.DataFrame(rows)


# =============================================================================
# AEV / HighAEV outcome agreement
# =============================================================================

def compare_outcomes(frames: dict[str, pd.DataFrame], available: list[str]) -> pd.DataFrame:
    from sklearn.metrics import cohen_kappa_score

    rows = []
    for a, b in itertools.combinations(available, 2):
        fa, fb = frames[a], frames[b]
        mask_cont = fa["aev_proxy"].notna() & fb["aev_proxy"].notna()
        n_cont = int(mask_cont.sum())
        r, p = (pearsonr(fa.loc[mask_cont, "aev_proxy"], fb.loc[mask_cont, "aev_proxy"])
                 if n_cont >= 10 else (np.nan, np.nan))

        mask_bin = fa["high_aev_proxy"].notna() & fb["high_aev_proxy"].notna()
        n_bin = int(mask_bin.sum())
        if n_bin >= 10:
            ya = fa.loc[mask_bin, "high_aev_proxy"].astype(int)
            yb = fb.loc[mask_bin, "high_aev_proxy"].astype(int)
            kappa = cohen_kappa_score(ya, yb)
            agree_rate = float((ya == yb).mean())
        else:
            kappa, agree_rate = np.nan, np.nan

        rows.append({
            "route_a": _ROUTE_LABELS[a], "route_b": _ROUTE_LABELS[b],
            "n_continuous": n_cont,
            "aev_pearson_r": round(float(r), 4) if pd.notna(r) else np.nan,
            "aev_pearson_p": round(float(p), 4) if pd.notna(p) else np.nan,
            "n_binary": n_bin,
            "high_aev_agreement_rate": round(agree_rate, 4) if pd.notna(agree_rate) else np.nan,
            "cohen_kappa": round(float(kappa), 4) if pd.notna(kappa) else np.nan,
        })
    return pd.DataFrame(rows)


# =============================================================================
# Structured qualitative comparison table (data, not prose)
# =============================================================================

def build_qualitative_comparison() -> pd.DataFrame:
    rows = [
        {
            "route": "Route 1: COR Composite",
            "main_logic": "Theory-guided formative composite scoring",
            "four_dimensions": "Formative composites (row-mean of normalized items) — not latent-variable extraction",
            "measurement_error": "Not explicitly modelled",
            "non_linearity": "No",
            "theory_use": "Item selection + formula",
            "output": "aev_score / high_aev",
            "main_metrics": "alpha, omega, CR, AVE, HTMT, OLS paths, bootstrap mediation",
            "best_role": "Main transparent, primary route",
        },
        {
            "route": "Route 2: COR-SEM",
            "main_logic": "Reflective latent-variable measurement + structural modelling",
            "four_dimensions": "CFA latent factors, estimated via semopy (maximum-likelihood)",
            "measurement_error": "Explicitly modelled",
            "non_linearity": "Limited (linear factor model)",
            "theory_use": "CFA measurement spec + second-order structural model",
            "output": "sem_aev_score / sem_high_aev",
            "main_metrics": "CFI, TLI, RMSEA, SRMR, standardized loadings, structural paths",
            "best_role": "Comparative/robustness route, evaluated cautiously (see fit indices before trusting)",
        },
        {
            "route": "Route 3: COR-VAE",
            "main_logic": "Deep generative representation learning",
            "four_dimensions": "COR-aligned deep VAE latent representations (mu), regularized against Route 1",
            "measurement_error": "Partly captured through reconstruction noise",
            "non_linearity": "Yes",
            "theory_use": "COR-alignment penalty on the latent space (against Route 1)",
            "output": "vae_aev_score / vae_high_aev",
            "main_metrics": "reconstruction MSE/MAE, KL divergence, alignment r, ROC-AUC, PR-AUC",
            "best_role": "Not independent of Route 1 (alignment target + prediction label) — AI-oriented extension",
        },
    ]
    return pd.DataFrame(rows)


# =============================================================================
# Figures
# =============================================================================

def _plot_construct_agreement(construct_df: pd.DataFrame) -> None:
    if construct_df.empty:
        return
    pivot = construct_df.pivot_table(
        index="construct", columns=["route_a", "route_b"], values="pearson_r"
    )
    pivot.columns = [f"{a}\nvs\n{b}" for a, b in pivot.columns]
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns, fontsize=8)
    ax.set_yticks(range(len(pivot.index)));   ax.set_yticklabels(pivot.index)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if abs(v) > 0.6 else "black", fontsize=9)
    plt.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title("Cross-route construct agreement (Pearson r)")
    fig.tight_layout()
    _save_fig(fig, "cross_route_construct_agreement")


def _plot_outcome_agreement(outcome_df: pd.DataFrame) -> None:
    if outcome_df.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    labels = [f"{a}\nvs {b}" for a, b in zip(outcome_df["route_a"], outcome_df["route_b"])]

    axes[0].bar(labels, outcome_df["aev_pearson_r"], color="#2980B9", alpha=0.85)
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_title("AEV continuous agreement (Pearson r)")
    axes[0].tick_params(axis="x", labelsize=8)

    axes[1].bar(labels, outcome_df["cohen_kappa"], color="#27AE60", alpha=0.85)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_title("HighAEV agreement (Cohen's kappa)")
    axes[1].tick_params(axis="x", labelsize=8)

    fig.suptitle("Cross-route AEV / HighAEV outcome agreement", fontsize=12)
    fig.tight_layout()
    _save_fig(fig, "cross_route_outcome_agreement")


def _plot_aev_distributions(frames: dict[str, pd.DataFrame], available: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for key in available:
        vals = frames[key]["aev_proxy"].dropna()
        if len(vals) == 0:
            continue
        ax.hist(vals, bins=30, alpha=0.5, label=_ROUTE_LABELS[key], color=_PALETTE[key], density=True)
    ax.set_xlabel("AEV proxy (route-specific dimensions, shared Route 1 formula)")
    ax.set_ylabel("Density")
    ax.set_title("AEV proxy distributions by route")
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save_fig(fig, "cross_route_aev_distributions")


# =============================================================================
# Main entry point
# =============================================================================

def run(
    df: pd.DataFrame,
    route2_scores: pd.DataFrame = None,
    route3_scores: pd.DataFrame = None,
) -> dict:
    """
    Cross-route comparison. `route2_scores`/`route3_scores` are the
    `scores_named` DataFrames returned by `src.cor_sem.run()["measurement"]`
    / `src.cor_vae.run()` (columns `sem_*_latent`/`sem_aev_score`/
    `sem_high_aev` or `vae_*_latent`/`vae_aev_score`/`vae_high_aev`). If not
    passed in-memory, falls back to reading the CSVs each module saves
    (`paths.LATENT_SCORES_ROUTE2_SEM` / `paths.LATENT_SCORES_ROUTE3_VAE`);
    if those are also absent, that route is treated as unavailable and the
    comparison degrades gracefully to whichever routes are present.
    """
    paths.ROUTE_COMPARISON_TABLES.mkdir(parents=True, exist_ok=True)
    paths.ROUTE_COMPARISON_FIGURES.mkdir(parents=True, exist_ok=True)

    if route2_scores is None and paths.LATENT_SCORES_ROUTE2_SEM.exists():
        route2_scores = pd.read_csv(paths.LATENT_SCORES_ROUTE2_SEM, index_col=0)
    if route3_scores is None and paths.LATENT_SCORES_ROUTE3_VAE.exists():
        route3_scores = pd.read_csv(paths.LATENT_SCORES_ROUTE3_VAE, index_col=0)

    available = _available_routes(route2_scores, route3_scores)
    log.info("Cross-route comparison: routes available = %s", available)

    frames = build_route_frame(df, route2_scores, route3_scores)

    construct_df = compare_constructs(frames, available)
    _save_csv(construct_df, "cross_route_construct_agreement")
    if not construct_df.empty:
        _plot_construct_agreement(construct_df)

    outcome_df = compare_outcomes(frames, available)
    _save_csv(outcome_df, "cross_route_outcome_agreement")
    if not outcome_df.empty:
        _plot_outcome_agreement(outcome_df)

    _plot_aev_distributions(frames, available)

    qual_df = build_qualitative_comparison()
    _save_csv(qual_df, "route_comparison_summary")

    log.info("Cross-route comparison complete.")
    log.info("Construct agreement:\n%s", construct_df.to_string(index=False))
    log.info("Outcome agreement:\n%s", outcome_df.to_string(index=False))

    return {"frames": frames, "available": available,
            "constructs": construct_df, "outcomes": outcome_df, "qualitative": qual_df}
