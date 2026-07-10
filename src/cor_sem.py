"""
cor_sem.py
──────────
Route 2 (COR-Informed SEM Route) — classical latent-variable estimation of
the four COR constructs via a Confirmatory Factor Analysis (CFA) measurement
model, extended into a second-order structural model for Adaptive Energy
Vulnerability (AEV).

Where Route 1 (`src.enable_preprocessing` + `src.sem_mediation` +
`src.unsupervised_latent`) treats FCP/AEMC/BLI/TCR as FORMATIVE composite
scores (row-means of normalized items, validated post-hoc via EFA), Route 2
specifies them upfront as REFLECTIVE latent variables measured with error:

  FCP  =~ n_S8 + n_E2A + n_E2B
  AEMC =~ n_E5A1 + n_E5A2 + n_E5A3 + n_E5A4 + n_E6A1 + n_E6A2 + n_E6A3
          + n_E6A4 + n_E6A6 + n_E6A7 + n_E6A8
  BLI  =~ n_E7A + n_E7B + n_E7C + n_E7D + n_E7E
  TCR  =~ n_H15A + n_H15B + n_H15C + n_H15D + n_H15E + n_H15F

using the item lists already defined in `src.construct_mapping` and the
`n_{item}` normalized columns already built by `src.enable_preprocessing`
(reverse-coding is already applied at the raw level, so no `_r` suffix is
needed here).  Model estimated via `semopy` (maximum-likelihood, MLW
estimator).

Second-order structural model (Option 2A, primary)
────────────────────────────────────────────────────
  AEV =~ FCP + BLI + TCR + AEMC

AEMC enters directly (not reverse-scored): a CFA loading's sign is free, so
a NEGATIVE AEV=~AEMC loading is the latent-variable equivalent of Route 1's
formative (1 − AEMC) inversion — no synthetic "Low_AEMC" indicator is
needed. Second-order models with weakly-correlated first-order factors can
produce an inadmissible ("Heywood") solution (near-zero or negative AEV
factor variance); `fit_structural_model()` detects and flags this rather
than silently reporting an unstable estimate.

Observed-outcome robustness check (Option 2B)
────────────────────────────────────────────
  high_aev / aev_score ~ FCP_score + AEMC_score + BLI_score + TCR_score

using the CFA factor scores as OLS predictors (`sem_mediation.ols_path`),
flagged circular in the same spirit as Route 1's `SEM_COR` classification
model — the four scores are estimated from the same items that define
`aev_score`/`high_aev` in Route 1.

Factor scoring
──────────────
Out-of-sample factor scoring for `fit_cfa_train_test()` (leakage-free
train/test splits, used by `src.ml_classification`) uses a Thurstone-style
loading-weighted composite of standardized items, computed with train-split
item means/SDs/loadings only, rather than semopy's built-in
`Model.predict_factors()` — the latter inverts a joint covariance system
that is frequently singular for train-only subsamples of this size (~500
complete cases split 75/25, further split 5-fold for CV). This trade-off is
documented in Chapter 8.

Usage
─────
  from src.cor_sem import run
  run(df)   # df from enable_preprocessing.run()
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
from src.construct_mapping import CONSTRUCT_REGISTRY
from src.construct_validation import (
    cronbach_alpha,
    mcdonald_omega,
    composite_reliability,
    average_variance_extracted,
    htmt_ratio,
    fornell_larcker_check,
)
from src.sem_mediation import ols_path, bootstrap_mediation
from src.route_utils import (
    compute_route_aev,
    route_high_aev,
    sign_canonicalize_against_route1,
    minmax_scale_route_scores,
)

log = logging.getLogger(__name__)

_FACTORS = ["FCP", "AEMC", "BLI", "TCR"]
_ROUTE1_SCORE_COLS = ["fcp_score", "aemc_score", "bli_score", "tcr_score"]

# Persisted/returned column names — Route 2 estimates REFLECTIVE latent
# variables through CFA/SEM (as opposed to Route 1's formative composites).
_RENAME_TO_SEM = {
    "FCP": "sem_fcp_latent", "AEMC": "sem_aemc_latent",
    "BLI": "sem_bli_latent", "TCR": "sem_tcr_latent",
}

_PALETTE = {
    "FCP": "#E74C3C", "AEMC": "#2980B9", "BLI": "#E67E22", "TCR": "#8E44AD",
    "AEV": "#2C3E50", "grid": "#EAECEE",
}

_MIN_ITEM_STD = 1e-6   # items below this SD within the complete-case CFA sample are dropped


# =============================================================================
# I/O helpers
# =============================================================================

def _save_csv(df: pd.DataFrame, name: str) -> None:
    paths.COR_SEM_TABLES.mkdir(parents=True, exist_ok=True)
    p = paths.COR_SEM_TABLES / f"{name}.csv"
    df.to_csv(p, index=False)
    log.info("Saved %s", p.name)


def _save_fig(fig: plt.Figure, name: str) -> None:
    paths.COR_SEM_FIGURES.mkdir(parents=True, exist_ok=True)
    p = paths.COR_SEM_FIGURES / f"{name}.{config.FIGURE_FORMAT}"
    fig.savefig(p, dpi=config.DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved figure %s", p.name)


# =============================================================================
# Model specification
# =============================================================================

def _construct_items(short: str) -> list[str]:
    """Core + reverse items for a construct, as raw item names (no n_ prefix)."""
    info = CONSTRUCT_REGISTRY[short]
    return list(info["core_items"]) + list(info.get("reverse_items", []))


def build_item_map(df: pd.DataFrame) -> dict[str, list[str]]:
    """
    Return {construct: [n_item, ...]} after dropping items that are absent
    or have near-zero variance in the complete-case CFA sample.
    """
    all_n_cols: list[str] = []
    construct_of: dict[str, str] = {}
    for short in _FACTORS:
        for item in _construct_items(short):
            col = f"n_{item}"
            all_n_cols.append(col)
            construct_of[col] = short

    available = [c for c in all_n_cols if c in df.columns]
    complete = df[available].dropna()

    kept: dict[str, list[str]] = {short: [] for short in _FACTORS}
    dropped: list[dict] = []
    for col in available:
        std = float(complete[col].std(ddof=0)) if len(complete) else 0.0
        short = construct_of[col]
        if std < _MIN_ITEM_STD:
            dropped.append({"construct": short, "item": col, "std_in_cfa_sample": std,
                             "reason": "near-zero variance in complete-case CFA sample"})
            log.warning("cor_sem: dropping %s (std=%.6f in complete-case sample)", col, std)
        else:
            kept[short].append(col)

    if dropped:
        _save_csv(pd.DataFrame(dropped), "cfa_item_exclusions")

    for short, cols in kept.items():
        if len(cols) < 3:
            log.warning("cor_sem: %s has only %d indicators after exclusions", short, len(cols))
    return kept


def build_measurement_spec(item_map: dict[str, list[str]]) -> str:
    lines = []
    for short in _FACTORS:
        cols = item_map[short]
        if len(cols) >= 2:
            lines.append(f"{short} =~ " + " + ".join(cols))
    return "\n".join(lines)


def build_structural_spec(item_map: dict[str, list[str]]) -> str:
    return build_measurement_spec(item_map) + "\nAEV =~ FCP + BLI + TCR + AEMC\n"


# =============================================================================
# CFA fit indices (semopy + manual SRMR)
# =============================================================================

def _srmr(model) -> float:
    """Standardized Root Mean Square Residual, computed manually (semopy has no built-in SRMR)."""
    try:
        sigma, _ = model.calc_sigma()
        S = model.mx_cov
        d = np.sqrt(np.diag(S))
        resid = (S - sigma) / np.outer(d, d)
        idx = np.tril_indices(S.shape[0])
        return float(np.sqrt(np.mean(resid[idx] ** 2)))
    except Exception as e:
        log.warning("SRMR computation failed: %s", e)
        return np.nan


def _fit_indices_row(model, label: str) -> dict:
    import semopy
    stats = semopy.calc_stats(model)
    row = {"model": label}
    for col in ["DoF", "chi2", "chi2 p-value", "CFI", "TLI", "RMSEA", "GFI", "AIC", "BIC"]:
        if col in stats.columns:
            v = stats[col].iloc[0]
            row[col.lower().replace(" ", "_").replace("-", "_")] = (
                round(float(v), 5) if pd.notna(v) else np.nan
            )
    row["srmr"] = round(_srmr(model), 5)
    row["cfi_ok"]   = bool(row.get("cfi",   np.nan) >= config.MIN_CFI)   if pd.notna(row.get("cfi"))   else None
    row["tli_ok"]   = bool(row.get("tli",   np.nan) >= config.MIN_TLI)   if pd.notna(row.get("tli"))   else None
    row["rmsea_ok"] = bool(row.get("rmsea", np.nan) <= config.MAX_RMSEA) if pd.notna(row.get("rmsea")) else None
    row["srmr_ok"]  = bool(row.get("srmr",  np.nan) <= config.MAX_SRMR)  if pd.notna(row.get("srmr"))  else None
    return row


# =============================================================================
# Loading-weighted (Thurstone-style) factor scoring — robust to small samples
# =============================================================================

def _loadings_from_inspect(inspect_df: pd.DataFrame) -> pd.DataFrame:
    """Standardized measurement loadings: rows where op=='~' and rval is a factor name."""
    load = inspect_df[(inspect_df["op"] == "~") & (inspect_df["rval"].isin(_FACTORS + ["AEV"]))].copy()
    load = load.rename(columns={"lval": "item", "rval": "factor", "Est. Std": "std_loading"})
    return load[["factor", "item", "Estimate", "Std. Err", "z-value", "p-value", "std_loading"]]


def score_from_loadings(
    item_df: pd.DataFrame,
    loadings: pd.DataFrame,
    ref_means: pd.Series,
    ref_stds: pd.Series,
) -> pd.DataFrame:
    """
    Thurstone-style loading-weighted composite per factor:
      score_f = sum_i(lambda_i * z_i) / sum_i(|lambda_i|)
    z_i standardized using ref_means/ref_stds (fit on the reference — i.e.
    training — sample to avoid leakage when called on a held-out split).
    """
    out = {}
    for short in _FACTORS:
        sub = loadings[loadings["factor"] == short]
        if sub.empty:
            out[short] = pd.Series(np.nan, index=item_df.index)
            continue
        items = sub["item"].tolist()
        w = sub.set_index("item")["std_loading"].reindex(items).values.astype(float)
        z = (item_df[items] - ref_means[items]) / ref_stds[items].replace(0, np.nan)
        denom = np.sum(np.abs(w))
        out[short] = (z.values * w).sum(axis=1) / (denom if denom > 1e-12 else 1.0)
    return pd.DataFrame(out, index=item_df.index)


# =============================================================================
# Leakage-free train/test CFA fitting (for src.ml_classification)
# =============================================================================

def fit_cfa_train_test(
    item_mat_tr: pd.DataFrame,
    item_mat_te: pd.DataFrame,
    seed: int = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit the 4-factor CFA measurement model on the TRAINING split only, then
    score both splits via loading-weighted composites (see
    `score_from_loadings`). Mirrors the leakage-free contract of
    `unsupervised_latent.fit_encode_train_test`.

    item_mat_tr / item_mat_te must contain the full `n_{item}` item set
    (same columns as `unsupervised_latent.build_item_matrix`'s output) with
    a sequential 0-based index.

    Returns (scores_tr, scores_te) DataFrames with columns ["FCP","AEMC","BLI","TCR"].
    """
    import semopy

    item_map = {}
    for short in _FACTORS:
        cols = [f"n_{it}" for it in _construct_items(short) if f"n_{it}" in item_mat_tr.columns]
        stds = item_mat_tr[cols].std(ddof=0) if cols else pd.Series(dtype=float)
        item_map[short] = [c for c in cols if stds.get(c, 0.0) >= _MIN_ITEM_STD]

    spec = build_measurement_spec(item_map)
    try:
        model = semopy.Model(spec)
        model.fit(item_mat_tr, obj=config.CFA_ESTIMATOR)
        inspect_df = model.inspect(std_est=True)
        loadings = _loadings_from_inspect(inspect_df)
    except Exception as e:
        log.warning("cor_sem: within-split CFA fit failed (%s) — factor scores set to NaN", e)
        sem_cols = list(_RENAME_TO_SEM.values())
        return (pd.DataFrame(np.nan, index=item_mat_tr.index, columns=sem_cols),
                pd.DataFrame(np.nan, index=item_mat_te.index, columns=sem_cols))

    ref_means = item_mat_tr.mean()
    ref_stds  = item_mat_tr.std(ddof=0).replace(0, np.nan)

    scores_tr = score_from_loadings(item_mat_tr, loadings, ref_means, ref_stds)
    scores_te = score_from_loadings(item_mat_te, loadings, ref_means, ref_stds)
    return (
        scores_tr[_FACTORS].rename(columns=_RENAME_TO_SEM),
        scores_te[_FACTORS].rename(columns=_RENAME_TO_SEM),
    )


# =============================================================================
# Full-sample measurement model (Stage 9 descriptive report)
# =============================================================================

def fit_measurement_model(df: pd.DataFrame) -> dict:
    """
    Fit the 4-factor CFA on the full complete-case sample. Returns a dict
    with loadings, fit indices, reliability/validity tables (reusing
    construct_validation's generic functions with CFA loadings), and
    per-household factor scores.
    """
    import semopy

    item_map = build_item_map(df)
    spec = build_measurement_spec(item_map)
    log.info("CFA measurement spec:\n%s", spec)

    all_cols = [c for cols in item_map.values() for c in cols]
    complete = df[all_cols].dropna()
    log.info("CFA complete-case sample: n=%d (of %d)", len(complete), len(df))

    model = semopy.Model(spec)
    model.fit(complete, obj=config.CFA_ESTIMATOR)
    inspect_df = model.inspect(std_est=True)
    loadings = _loadings_from_inspect(inspect_df)
    _save_csv(loadings, "cfa_measurement_loadings")

    fit_row = _fit_indices_row(model, "measurement_model")
    fit_df = pd.DataFrame([fit_row])
    _save_csv(fit_df, "cfa_fit_indices")
    log.info("CFA fit: CFI=%.3f TLI=%.3f RMSEA=%.3f SRMR=%.3f (n=%d, DoF=%s)",
             fit_row.get("cfi", np.nan), fit_row.get("tli", np.nan),
             fit_row.get("rmsea", np.nan), fit_row.get("srmr", np.nan),
             len(complete), fit_row.get("dof"))

    # ── Reliability / validity, reusing construct_validation's generic functions ──
    rel_rows, ave_dict, htmt_data = [], {}, {}
    for short in _FACTORS:
        sub = loadings[loadings["factor"] == short]
        if sub.empty:
            continue
        items = sub["item"].tolist()
        item_mat = complete[items]
        htmt_data[short] = item_mat
        l = sub["std_loading"].values.astype(float)

        alpha = cronbach_alpha(item_mat)
        omega = mcdonald_omega(l, item_mat)
        cr    = composite_reliability(l)
        ave   = average_variance_extracted(l)
        ave_dict[short] = ave

        rel_rows.append({
            "construct": short,
            "n_items": len(items),
            "n_complete": int(item_mat.dropna(how="any").shape[0]),
            "cronbach_alpha": round(float(alpha), 4) if pd.notna(alpha) else np.nan,
            "mcdonald_omega": round(float(omega), 4) if pd.notna(omega) else np.nan,
            "composite_reliability": round(float(cr), 4) if pd.notna(cr) else np.nan,
            "ave": round(float(ave), 4) if pd.notna(ave) else np.nan,
            "mean_std_loading": round(float(np.mean(np.abs(l))), 4),
            "ave_ok": bool(ave >= config.MIN_AVE) if pd.notna(ave) else None,
        })
    rel_df = pd.DataFrame(rel_rows)
    _save_csv(rel_df, "cfa_reliability_validity")

    htmt_rows = []
    names = list(htmt_data.keys())
    for i, c1 in enumerate(names):
        for j, c2 in enumerate(names):
            if i >= j:
                continue
            val = htmt_ratio(htmt_data[c1], htmt_data[c2])
            htmt_rows.append({"construct_i": c1, "construct_j": c2,
                               "htmt": round(float(val), 4) if pd.notna(val) else np.nan,
                               "htmt_ok": bool(val < config.MAX_HTMT) if pd.notna(val) else None})
    htmt_df = pd.DataFrame(htmt_rows)
    _save_csv(htmt_df, "cfa_htmt_matrix")

    # Factor-correlation matrix from the fitted model (for Fornell-Larcker)
    corr_rows = inspect_df[(inspect_df["op"] == "~~") & (inspect_df["lval"] != inspect_df["rval"])
                            & inspect_df["lval"].isin(_FACTORS) & inspect_df["rval"].isin(_FACTORS)]
    corr = pd.DataFrame(1.0, index=_FACTORS, columns=_FACTORS)
    var_diag = inspect_df[(inspect_df["op"] == "~~") & (inspect_df["lval"] == inspect_df["rval"])
                           & inspect_df["lval"].isin(_FACTORS)].set_index("lval")["Estimate"]
    for _, r in corr_rows.iterrows():
        v1, v2 = var_diag.get(r["lval"], np.nan), var_diag.get(r["rval"], np.nan)
        if pd.notna(v1) and pd.notna(v2) and v1 > 0 and v2 > 0:
            rho = float(r["Estimate"]) / np.sqrt(v1 * v2)
            corr.loc[r["lval"], r["rval"]] = rho
            corr.loc[r["rval"], r["lval"]] = rho
    _save_csv(corr.reset_index().rename(columns={"index": "construct"}), "cfa_factor_correlations")

    fl_df = fornell_larcker_check(ave_dict, corr) if ave_dict else pd.DataFrame()
    if not fl_df.empty:
        _save_csv(fl_df, "cfa_fornell_larcker_check")

    # ── Factor scores (full sample, loading-weighted) ─────────────────────
    ref_means = complete.mean()
    ref_stds  = complete.std(ddof=0).replace(0, np.nan)
    scores = score_from_loadings(complete, loadings, ref_means, ref_stds)
    scores.index = complete.index
    scores_full = scores.reindex(df.index)

    # ── sem_fcp_latent / sem_aemc_latent / sem_bli_latent / sem_tcr_latent
    #    + sem_aev_score / sem_high_aev (route-comparison outcome proxy) ────
    scores_named = scores_full[_FACTORS].rename(columns=_RENAME_TO_SEM)
    if all(c in df.columns for c in _ROUTE1_SCORE_COLS):
        route1 = df[_ROUTE1_SCORE_COLS].rename(columns={
            "fcp_score": "FCP", "aemc_score": "AEMC", "bli_score": "BLI", "tcr_score": "TCR",
        })
        canon = sign_canonicalize_against_route1(scores_full[_FACTORS], route1)
        canon01 = minmax_scale_route_scores(canon, _FACTORS)
        aev = compute_route_aev(canon01, _FACTORS)
        scores_named["sem_aev_score"] = aev
        scores_named["sem_high_aev"]  = route_high_aev(aev, config.AEV_QUANTILE)
    else:
        log.warning("Route 1 composite scores absent from df — sem_aev_score/sem_high_aev not computed.")

    scores_named.to_csv(paths.LATENT_SCORES_ROUTE2_SEM)
    log.info("Saved Route 2 factor scores: %s", paths.LATENT_SCORES_ROUTE2_SEM.name)

    return {
        "model": model, "loadings": loadings, "fit": fit_df,
        "reliability": rel_df, "htmt": htmt_df, "fl_check": fl_df,
        "corr": corr, "ave_dict": ave_dict,
        "scores": scores_full, "scores_named": scores_named,
        "item_map": item_map, "complete": complete,
    }


# =============================================================================
# Second-order structural model (Option 2A) + observed-outcome check (2B)
# =============================================================================

def fit_structural_model(df: pd.DataFrame, item_map: dict[str, list[str]]) -> dict:
    """Second-order AEV =~ FCP + BLI + TCR + AEMC structural model (Option 2A)."""
    import semopy

    spec = build_structural_spec(item_map)
    all_cols = [c for cols in item_map.values() for c in cols]
    complete = df[all_cols].dropna()

    model = semopy.Model(spec)
    try:
        model.fit(complete, obj=config.CFA_ESTIMATOR)
    except Exception as e:
        log.error("Second-order structural model failed to fit: %s", e)
        return {"admissible": False, "reason": str(e)}

    inspect_df = model.inspect(std_est=True)
    struct_rows = inspect_df[(inspect_df["op"] == "~") & (inspect_df["rval"] == "AEV")].copy()
    struct_rows = struct_rows.rename(columns={"lval": "first_order_factor"})

    aev_var_row = inspect_df[(inspect_df["op"] == "~~") & (inspect_df["lval"] == "AEV")
                              & (inspect_df["rval"] == "AEV")]
    aev_var = float(aev_var_row["Estimate"].iloc[0]) if not aev_var_row.empty else np.nan

    # Admissibility check: non-degenerate AEV variance, no |std loading| > 1.2 (Heywood-ish)
    max_abs_std = struct_rows["Est. Std"].abs().max() if not struct_rows.empty else np.nan
    admissible = bool(
        pd.notna(aev_var) and aev_var > 1e-6 and
        (pd.isna(max_abs_std) or max_abs_std <= 1.2)
    )
    if not admissible:
        log.warning(
            "Second-order AEV structural model is INADMISSIBLE "
            "(aev_variance=%.6g, max|std loading|=%s) — likely a Heywood case "
            "reflecting weak inter-construct correlations (see cfa_factor_correlations.csv). "
            "Reporting estimates with this flag rather than treating them as reliable.",
            aev_var, max_abs_std,
        )

    fit_row = _fit_indices_row(model, "structural_model_2A")
    _save_csv(struct_rows, "structural_paths_2A")
    _save_csv(pd.DataFrame([fit_row]), "structural_fit_indices_2A")

    return {
        "admissible": admissible, "aev_variance": aev_var,
        "paths": struct_rows, "fit": fit_row, "model": model,
    }


def fit_observed_outcome_check(df: pd.DataFrame, scores: pd.DataFrame) -> dict:
    """
    Option 2B: regress the OBSERVED aev_score/high_aev (Route 1's outcome)
    on the four CFA factor scores. Flagged circular in the same spirit as
    Route 1's SEM_COR classification model.
    """
    work = df.join(scores.add_suffix("_cfa"), how="left")
    predictors = [f"{f}_cfa" for f in _FACTORS]

    rows = []
    if "aev_score" in work.columns:
        r = ols_path(work, "aev_score", predictors, label="Route2 Option2B: CFA scores -> AEV (circular)")
        rows.extend(r.get("rows", []))
    path_df = pd.DataFrame(rows)
    _save_csv(path_df, "structural_paths_2B_observed_outcome")

    med = {}
    if all(c in work.columns for c in ["aev_score"]) and \
       "FCP_cfa" in work.columns and "AEMC_cfa" in work.columns:
        med = bootstrap_mediation(
            work, "FCP_cfa", "AEMC_cfa", "aev_score",
            n_boot=config.SEM_BOOT_N, ci_level=0.95,
        )
        if med:
            _save_csv(pd.DataFrame([med]), "mediation_2B_cfa_scores")
    return {"paths": path_df, "mediation": med}


# =============================================================================
# Figures
# =============================================================================

def _plot_loadings(loadings: pd.DataFrame) -> None:
    if loadings.empty:
        return
    factors = [f for f in _FACTORS if f in loadings["factor"].unique()]
    fig, axes = plt.subplots(1, len(factors), figsize=(4 * len(factors), 5))
    if len(factors) == 1:
        axes = [axes]
    for ax, f in zip(axes, factors):
        sub = loadings[loadings["factor"] == f]
        ax.barh(sub["item"], sub["std_loading"].abs(), color=_PALETTE.get(f, "#999"), alpha=0.85)
        ax.axvline(config.MIN_LOADING, color="red", ls="--", lw=1, label=f"min={config.MIN_LOADING}")
        ax.set_xlim(0, 1.05)
        ax.set_title(f"{f} (CFA)", fontsize=9)
        ax.set_xlabel("|Standardized loading|")
        ax.legend(fontsize=7)
    fig.suptitle("Route 2: CFA standardized measurement loadings", fontsize=11)
    fig.tight_layout()
    _save_fig(fig, "cfa_loadings")


def _plot_fit_indices(fit_df: pd.DataFrame) -> None:
    if fit_df.empty:
        return
    metrics = [("cfi", config.MIN_CFI, ">="), ("tli", config.MIN_TLI, ">="),
               ("rmsea", config.MAX_RMSEA, "<="), ("srmr", config.MAX_SRMR, "<=")]
    labels = [m[0].upper() for m in metrics]
    vals = [fit_df.iloc[0].get(m[0], np.nan) for m in metrics]
    thresh = [m[1] for m in metrics]
    colors = ["#27AE60" if (v <= t if m[2] == "<=" else v >= t) else "#E74C3C"
              for (m, t, v) in zip(metrics, thresh, vals) if pd.notna(v)] or ["#999"] * len(metrics)

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, [v if pd.notna(v) else 0 for v in vals], color=colors, alpha=0.85)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{v:.3f}" if pd.notna(v) else "NA", ha="center", fontsize=9)
    ax.set_title("Route 2: CFA global fit indices\n(green = meets conventional threshold)")
    ax.set_ylabel("Value")
    fig.tight_layout()
    _save_fig(fig, "cfa_fit_indices")


def _plot_structural_paths(struct_result: dict) -> None:
    paths_df = struct_result.get("paths", pd.DataFrame())
    if paths_df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [_PALETTE.get(f, "#999") for f in paths_df["first_order_factor"]]
    bars = ax.barh(paths_df["first_order_factor"], paths_df["Est. Std"], color=colors, alpha=0.85)
    ax.axvline(0, color="black", lw=0.8)
    admissible = struct_result.get("admissible", False)
    title = "Route 2: second-order AEV structural loadings (Option 2A)"
    if not admissible:
        title += "\n⚠ INADMISSIBLE SOLUTION (Heywood case) — interpret with caution"
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Standardized loading on AEV")
    fig.tight_layout()
    _save_fig(fig, "structural_paths_2A")


# =============================================================================
# Main entry point
# =============================================================================

def run(df: pd.DataFrame) -> dict:
    """
    Route 2 full pipeline: CFA measurement model → reliability/validity →
    second-order structural model (2A) → observed-outcome robustness check (2B).

    Route 2 estimates REFLECTIVE latent variables through CFA/SEM — a
    comparative/robustness route evaluated cautiously against Route 1, not
    a claim that SEM is the primary valid measurement route (see
    `fit_structural_model`'s Heywood-case flag and the CFI/TLI/RMSEA/SRMR
    fit indices in `cfa_fit_indices.csv`).

    Resilient to failure: if the CFA measurement model itself cannot be
    fit (e.g. non-convergence, singular covariance), this logs the error,
    writes whatever diagnostics were produced, and returns {} rather than
    raising — callers (e.g. `household_stream.run`) should treat an empty
    dict as "Route 2 unavailable" and continue the pipeline.
    """
    paths.COR_SEM_TABLES.mkdir(parents=True, exist_ok=True)
    paths.COR_SEM_FIGURES.mkdir(parents=True, exist_ok=True)

    try:
        meas = fit_measurement_model(df)
    except Exception as e:
        log.error("Route 2 (SEM) CFA measurement model failed to fit: %s", e)
        _save_csv(pd.DataFrame([{"stage": "measurement_model", "error": str(e)}]),
                  "cor_sem_failure_diagnostic")
        return {}

    _plot_loadings(meas["loadings"])
    _plot_fit_indices(meas["fit"])

    struct = fit_structural_model(df, meas["item_map"])
    if struct.get("paths") is not None and not struct.get("paths", pd.DataFrame()).empty:
        _plot_structural_paths(struct)

    outcome_check = fit_observed_outcome_check(df, meas["scores"])

    log.info("Route 2 (COR-SEM) stream complete.")
    return {
        "measurement": meas,
        "structural_2a": struct,
        "observed_outcome_2b": outcome_check,
    }
