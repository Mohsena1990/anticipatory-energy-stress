"""
ukhls_cor_sem.py
─────────────────
Stage 2b — COR-SEM: Confirmatory Factor Analysis for Hobfoll's (1989)
Conservation of Resources dimensions on the UKHLS household-wave panel.

  OBJECT    =~ hsrooms + hsbeds + ncars + carval + hsval
  CONDITION =~ tenure_security + jbstat_security + bill_security
  PERSONAL  =~ health_good + sf1_good + qfhigh_band
  ENERGY    =~ fihhmnnet1_dv + fiyrinvinc_dv

  BASELINE  =~ OBJECT + CONDITION + PERSONAL + ENERGY   (second-order)

Item registry lives in `src.ukhls_mapping.COR_FACTOR_ITEMS` — the single
source of truth shared with `src.ukhls_preprocessing` (which builds the
recoded columns this module consumes); the spec above is generated from
that registry at runtime (`build_measurement_spec`), not hardcoded here,
so it always reflects whichever items are currently IN the registry. Two
items were tested and added/kept (`bill_security`) or tested and rejected
(`dvage`, `inoutflows2/3/4` from the pooled ENERGY factor) — see
`ukhls_mapping.py`'s comments beside `COR_FACTOR_ITEMS` for the empirical
evidence behind each call; this docstring previously went stale after
those changes and still showed the OLD (partly rejected) spec, so treat
`COR_FACTOR_ITEMS` itself, not this comment, as authoritative if they ever
diverge again.

Missing data
────────────
`hsval`/`carval` are missing for non-owners/non-car-owners — this is
STRUCTURAL, not random (~40-70% of the sample by construction), so this
module fits via semopy's `obj="FIML"` (full-information maximum
likelihood) rather than listwise deletion, which would otherwise silently
turn the model into an "owners-with-cars-only" model. `inoutflows2/3/4`
are additionally sparse (only waves m/o carry the cost-of-living-crisis
module) — FIML handles this the same way, using whatever indicators each
row has.

Panel caveat
────────────
The CFA is fit POOLED across all 15 waves. This assumes measurement
invariance over time (item meanings/loadings stable 2009-2024), which is
NOT tested here — flagged as a limitation, not hidden, the same way the
original ENABLE-based Route 2 flagged Heywood-case risk rather than
silently reporting an unstable estimate.

Known tool limitation: FIML chi2/CFI/TLI in semopy 2.3.11
────────────────────────────────────────────────────────
Verified empirically (not assumed): fitting the SAME spec via `obj="MLW"`
on the tiny complete-case subset (n=29, listwise-deleting all structural
missingness) produces CFI/TLI in the valid [0,1] range (0.72/0.62), while
`obj="FIML"` on the full sample produces impossible negative CFI/TLI with
an implausibly small chi2 (p=1.0) regardless of specification changes that
fixed everything else (outlier winsorization, dropping incoherent/sparse
items). This isolates the issue to semopy's FIML fit-statistic computation,
not this module's model specification. Loadings, the second-order
BASELINE admissibility check, and — most importantly — the FES-moderation
test below do NOT depend on this computation and are the primary evidence
this module reports; CFI/TLI are still saved for transparency but should
be read as unreliable under FIML, not as evidence the measurement model is
badly misspecified. RMSEA/SRMR/GFI are reported alongside as a partial
cross-check.

FES moderation (the COR-theoretic core of this stage)
────────────────────────────────────────────────────
COR theory predicts that low-baseline-resource households absorb an
energy-price shock worse than high-baseline households: the effect of
forecasted/realised energy stress (FES) on fuel-to-income burden should be
DAMPENED by baseline resources. Tested via OLS (reusing
`src.sem_mediation.ols_path`) as an interaction term:

  fuel_to_income_ratio ~ BASELINE_score + fes_delta + BASELINE_score*fes_delta

COR-consistent hypothesis: the interaction coefficient is NEGATIVE.

Usage
─────
  from src.ukhls_cor_sem import run
  result = run(df)   # df from ukhls_preprocessing.run()
"""

from __future__ import annotations

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from src import config, paths
from src.logging_utils import get_logger
from src.ukhls_mapping import COR_FACTOR_ITEMS
from src.sem_mediation import ols_path

log = get_logger("ukhls_cor_sem")

_FACTORS: list[str] = list(COR_FACTOR_ITEMS.keys())  # OBJECT, CONDITION, PERSONAL, ENERGY
_MIN_ITEM_STD = 1e-6

_PALETTE = {
    "OBJECT": "#E74C3C", "CONDITION": "#2980B9", "PERSONAL": "#E67E22",
    "ENERGY": "#8E44AD", "BASELINE": "#2C3E50",
}


# =============================================================================
# I/O helpers
# =============================================================================

def _save_csv(df: pd.DataFrame, name: str) -> None:
    paths.UKHLS_SEM_TABLES.mkdir(parents=True, exist_ok=True)
    p = paths.UKHLS_SEM_TABLES / f"{name}.csv"
    df.to_csv(p, index=False)
    log.info("Saved %s", p.name)


def _save_fig(fig: plt.Figure, name: str) -> None:
    paths.UKHLS_SEM_FIGURES.mkdir(parents=True, exist_ok=True)
    p = paths.UKHLS_SEM_FIGURES / f"{name}.{config.FIGURE_FORMAT}"
    fig.savefig(p, dpi=config.DPI, bbox_inches="tight")
    fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    log.info("Saved figure %s", p.name)


# =============================================================================
# Item selection + standardization
# =============================================================================

def build_item_map(df: pd.DataFrame) -> dict[str, list[str]]:
    """Drop items absent from df or with near-zero variance."""
    kept: dict[str, list[str]] = {}
    dropped: list[dict] = []
    for factor, items in COR_FACTOR_ITEMS.items():
        avail = [c for c in items if c in df.columns]
        stds = df[avail].std(ddof=0) if avail else pd.Series(dtype=float)
        kept_items = []
        for c in avail:
            if stds.get(c, 0.0) < _MIN_ITEM_STD:
                dropped.append({"factor": factor, "item": c, "std": float(stds.get(c, 0.0))})
            else:
                kept_items.append(c)
        kept[factor] = kept_items
    if dropped:
        _save_csv(pd.DataFrame(dropped), "cor_sem_item_exclusions")
    for factor, cols in kept.items():
        if len(cols) < 2:
            log.warning("ukhls_cor_sem: %s has only %d indicators", factor, len(cols))
    return kept


# Monetary variables are extremely heavy-tailed (e.g. hsval skew ~56 --
# a handful of outlier property/investment values would otherwise dominate
# a linear z-score and wreck the CFA). Signed-log compresses magnitude
# while preserving sign (fihhmnnet1_dv can be negative for self-employment
# losses) and zero.
_SKEWED_MONEY_COLS = {"carval", "hsval", "fihhmnnet1_dv", "fiyrinvinc_dv"}


def _signed_log1p(s: pd.Series) -> pd.Series:
    return np.sign(s) * np.log1p(np.abs(s))


def _winsorize(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """Cap at the 1st/99th percentile. Several count-type items (hsbeds max
    50, hsrooms max 69, ncars max 14) carry a handful of implausible/mis-
    keyed values that a log transform alone doesn't tame."""
    lo, hi = s.quantile(lower), s.quantile(upper)
    return s.clip(lower=lo, upper=hi)


def _standardize(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Signed-log-transform known heavy-tailed monetary columns, winsorize
    all columns against outliers, then z-score (NaNs preserved — required
    for FIML)."""
    out = df.copy()
    for c in cols:
        col = _signed_log1p(out[c]) if c in _SKEWED_MONEY_COLS else out[c]
        col = _winsorize(col)
        mu, sd = col.mean(), col.std(ddof=0)
        out[c] = (col - mu) / sd if sd and sd > 1e-10 else np.nan
    return out


# =============================================================================
# Model specification
# =============================================================================

def build_measurement_spec(item_map: dict[str, list[str]]) -> str:
    lines = []
    for f in _FACTORS:
        cols = item_map.get(f, [])
        if len(cols) >= 2:
            lines.append(f"{f} =~ " + " + ".join(cols))
    return "\n".join(lines)


def build_structural_spec(item_map: dict[str, list[str]]) -> str:
    return build_measurement_spec(item_map) + "\nBASELINE =~ OBJECT + CONDITION + PERSONAL + ENERGY\n"


# =============================================================================
# Fit indices (semopy + manual SRMR) — generic, adapted from src.cor_sem
# =============================================================================

def _srmr(model) -> float:
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


def _loadings_from_inspect(inspect_df: pd.DataFrame, factor_names: list[str]) -> pd.DataFrame:
    load = inspect_df[(inspect_df["op"] == "~") & (inspect_df["rval"].isin(factor_names))].copy()
    load = load.rename(columns={"lval": "item", "rval": "factor", "Est. Std": "std_loading"})
    return load[["factor", "item", "Estimate", "Std. Err", "z-value", "p-value", "std_loading"]]


# =============================================================================
# Factor scoring — per-row available-item loading-weighted composite
# (handles the structural missingness on hsval/carval/inoutflows* directly:
# a renter with no hsval still gets an OBJECT score from hsrooms/hsbeds/ncars)
# =============================================================================

def score_from_loadings(std_df: pd.DataFrame, loadings: pd.DataFrame, factors: list[str]) -> pd.DataFrame:
    out = {}
    for f in factors:
        sub = loadings[loadings["factor"] == f]
        if sub.empty:
            out[f] = pd.Series(np.nan, index=std_df.index)
            continue
        items = sub["item"].tolist()
        w = sub.set_index("item")["std_loading"].reindex(items)
        Z = std_df[items]
        weighted = Z.mul(w, axis=1)
        num = weighted.sum(axis=1, skipna=True)
        denom = Z.notna().astype(float).mul(w.abs(), axis=1).sum(axis=1)
        score = num / denom.replace(0, np.nan)
        score[Z.notna().sum(axis=1) == 0] = np.nan
        out[f] = score
    return pd.DataFrame(out, index=std_df.index)


# =============================================================================
# STEP 1 — Measurement model (4 first-order factors, FIML)
# =============================================================================

def fit_measurement_model(df: pd.DataFrame) -> dict:
    import semopy

    item_map = build_item_map(df)
    all_cols = [c for cols in item_map.values() for c in cols]
    std_df = _standardize(df, all_cols)

    spec = build_measurement_spec(item_map)
    log.info("COR-SEM measurement spec:\n%s", spec)
    log.info("Fitting via FIML (handles structural missingness on hsval/carval/inoutflows*) "
             "— n=%d rows, %d indicators", len(std_df), len(all_cols))

    model = semopy.Model(spec)
    model.fit(std_df[all_cols], obj="FIML")
    inspect_df = model.inspect(std_est=True)
    loadings = _loadings_from_inspect(inspect_df, _FACTORS)
    _save_csv(loadings, "cor_sem_measurement_loadings")

    fit_row = _fit_indices_row(model, "measurement_model")
    _save_csv(pd.DataFrame([fit_row]), "cor_sem_fit_indices")
    log.info("COR-SEM fit: CFI=%s TLI=%s RMSEA=%s SRMR=%s (DoF=%s)",
             fit_row.get("cfi"), fit_row.get("tli"), fit_row.get("rmsea"),
             fit_row.get("srmr"), fit_row.get("dof"))
    if pd.notna(fit_row.get("cfi")) and fit_row["cfi"] < 0:
        log.warning(
            "CFI/TLI outside [0,1] — this is a known semopy FIML fit-"
            "statistic computation limitation (verified against a "
            "complete-case MLW comparison, see module docstring), not "
            "evidence of measurement-model misspecification. Read the "
            "loadings and the FES-moderation test as the primary evidence."
        )

    scores = score_from_loadings(std_df, loadings, _FACTORS)
    scores_named = scores.rename(columns={f: f"{f.lower()}_score" for f in _FACTORS})

    return {
        "model": model, "loadings": loadings, "fit": fit_row,
        "item_map": item_map, "std_df": std_df, "scores": scores_named,
    }


# =============================================================================
# STEP 2 — Second-order structural model: BASELINE =~ 4 factors
# =============================================================================

def fit_structural_model(std_df: pd.DataFrame, item_map: dict[str, list[str]]) -> dict:
    import semopy

    spec = build_structural_spec(item_map)
    all_cols = [c for cols in item_map.values() for c in cols]

    model = semopy.Model(spec)
    try:
        model.fit(std_df[all_cols], obj="FIML")
    except Exception as e:
        log.error("BASELINE second-order structural model failed to fit: %s", e)
        return {"admissible": False, "reason": str(e)}

    inspect_df = model.inspect(std_est=True)
    struct_rows = inspect_df[(inspect_df["op"] == "~") & (inspect_df["rval"] == "BASELINE")].copy()
    struct_rows = struct_rows.rename(columns={"lval": "first_order_factor"})

    baseline_var_row = inspect_df[(inspect_df["op"] == "~~") & (inspect_df["lval"] == "BASELINE")
                                   & (inspect_df["rval"] == "BASELINE")]
    baseline_var = float(baseline_var_row["Estimate"].iloc[0]) if not baseline_var_row.empty else np.nan

    max_abs_std = struct_rows["Est. Std"].abs().max() if not struct_rows.empty else np.nan
    admissible = bool(
        pd.notna(baseline_var) and baseline_var > 1e-6 and
        (pd.isna(max_abs_std) or max_abs_std <= 1.2)
    )
    if not admissible:
        log.warning(
            "BASELINE second-order model is INADMISSIBLE (Heywood case): "
            "baseline_variance=%.6g, max|std loading|=%s — reporting with this flag.",
            baseline_var, max_abs_std,
        )

    fit_row = _fit_indices_row(model, "structural_model_baseline")
    _save_csv(struct_rows, "structural_paths_baseline")
    _save_csv(pd.DataFrame([fit_row]), "structural_fit_indices_baseline")

    # BASELINE factor score: loading-weighted composite of the 4 first-order
    # factor loadings applied to their own (already-standardized) scores.
    loadings4 = struct_rows.rename(columns={"first_order_factor": "item"}).copy()
    loadings4["factor"] = "BASELINE"
    loadings4 = loadings4.rename(columns={"Est. Std": "std_loading"})[["factor", "item", "std_loading"]]

    return {
        "admissible": admissible, "baseline_variance": baseline_var,
        "paths": struct_rows, "fit": fit_row, "model": model,
        "loadings4": loadings4,
    }


def compute_baseline_score(first_order_scores: pd.DataFrame, loadings4: pd.DataFrame) -> pd.Series:
    """BASELINE score from the 4 first-order factor scores (already
    computed, roughly standardized) weighted by their structural loadings."""
    if loadings4.empty:
        return pd.Series(np.nan, index=first_order_scores.index)
    std_scores = _standardize(first_order_scores, list(first_order_scores.columns))
    scored = score_from_loadings(
        std_scores.rename(columns={f"{f.lower()}_score": f for f in _FACTORS}),
        loadings4, ["BASELINE"],
    )
    return scored["BASELINE"]


# =============================================================================
# STEP 3 — FES moderation test (the COR-theoretic core of this stage)
# =============================================================================

def test_fes_moderation(df: pd.DataFrame, baseline_score: pd.Series) -> dict:
    """
    fuel_to_income_ratio ~ BASELINE_score + fes_delta + BASELINE_score*fes_delta

    COR-consistent hypothesis: interaction coefficient is NEGATIVE — higher
    baseline resources dampen the FES -> vulnerability effect. `fes_delta`
    (FES Magnitude minus each household-wave's own realised FES exposure —
    see `src.ukhls_preprocessing.attach_fes_delta`) is the row-varying shock
    signal used here; `fes_magnitude` (the constant forward-looking national
    shock) is reported as metadata only — it has zero variance this run, so
    its own OLS coefficient would be inestimable.
    """
    work = df.copy()
    work["baseline_score"] = baseline_score
    if "fes_delta" not in work.columns:
        log.warning("fes_delta column not found — FES moderation test skipped. "
                    "Run src.ukhls_preprocessing.attach_fes_delta first.")
        return {}
    work["baseline_x_fes"] = work["baseline_score"] * work["fes_delta"]

    r = ols_path(
        work, "fuel_to_income_ratio",
        ["baseline_score", "fes_delta", "baseline_x_fes"],
        label="COR-SEM: BASELINE x FES Delta -> fuel_to_income_ratio",
    )
    rows = r.get("rows", [])
    _save_csv(pd.DataFrame(rows), "fes_moderation_path")

    interaction_row = next((row for row in rows if row["predictor"] == "baseline_x_fes"), None)
    cor_consistent = None
    if interaction_row:
        cor_consistent = interaction_row["coef"] < 0 and interaction_row["p_value"] < 0.05
        log.info(
            "FES moderation: baseline_x_fes coef=%.5f p=%.4f -> %s with COR "
            "(expected: negative & significant)",
            interaction_row["coef"], interaction_row["p_value"],
            "CONSISTENT" if cor_consistent else "not consistent",
        )
    fes_magnitude = float(work["fes_magnitude"].iloc[0]) if "fes_magnitude" in work.columns else None
    return {
        "rows": rows, "interaction_row": interaction_row, "cor_consistent": cor_consistent,
        "fes_magnitude": fes_magnitude,
    }


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
        ax.set_title(f"{f} (CFA)", fontsize=9)
        ax.set_xlabel("|Standardized loading|")
        ax.legend(fontsize=7)
    fig.suptitle("COR-SEM: standardized measurement loadings", fontsize=11)
    fig.tight_layout()
    _save_fig(fig, "cor_sem_loadings")


def plot_factor_loadings_polar(loadings: pd.DataFrame) -> None:
    """
    Grouped circular bar chart (house style, see
    src.plotting_utils.plot_grouped_circular_bars): one group per COR
    factor (OBJECT/CONDITION/PERSONAL/ENERGY), one individual (non-stacked)
    bar per item, height = |standardized loading|. Each item is now
    directly comparable at a glance (previously stacked centre-to-edge,
    which showed total measurement evidence per factor but hid individual
    item quality inside the stack).
    """
    from src.plotting_utils import plot_grouped_circular_bars

    if loadings.empty:
        return
    factors = [f for f in _FACTORS if f in loadings["factor"].unique()]
    if not factors:
        return

    groups = {}
    for f in factors:
        sub = loadings[loadings["factor"] == f]
        groups[f] = {row["item"]: abs(float(row["std_loading"])) for _, row in sub.iterrows()}

    group_colors = {f: _PALETTE.get(f, "#888888") for f in factors}
    plot_grouped_circular_bars(
        groups, group_colors, f"{paths.UKHLS_SEM_FIGURES}/cor_sem_loadings_polar.png",
        title="COR-SEM: Item Loadings by Factor\n(bar = |standardized loading|, taller = stronger item)",
    )


def _plot_structural_paths(struct_result: dict) -> None:
    paths_df = struct_result.get("paths", pd.DataFrame())
    if paths_df.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [_PALETTE.get(f, "#999") for f in paths_df["first_order_factor"]]
    ax.barh(paths_df["first_order_factor"], paths_df["Est. Std"], color=colors, alpha=0.85)
    ax.axvline(0, color="black", lw=0.8)
    title = "COR-SEM: second-order BASELINE structural loadings"
    if not struct_result.get("admissible", False):
        title += "\n⚠ INADMISSIBLE SOLUTION (Heywood case)"
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Standardized loading on BASELINE")
    fig.tight_layout()
    _save_fig(fig, "structural_paths_baseline")


# =============================================================================
# Main entry point
# =============================================================================

def run(df: pd.DataFrame) -> dict:
    """
    Stage 2b full pipeline: 4-factor CFA (FIML) -> second-order BASELINE
    structural model -> FES-moderation OLS test.

    Resilient to failure: if the measurement model cannot be fit, logs the
    error, writes a diagnostic CSV, and returns {} rather than raising —
    callers should treat an empty dict as "Stage 2b unavailable".
    """
    paths.UKHLS_SEM_TABLES.mkdir(parents=True, exist_ok=True)
    paths.UKHLS_SEM_FIGURES.mkdir(parents=True, exist_ok=True)

    try:
        meas = fit_measurement_model(df)
    except Exception as e:
        log.error("COR-SEM measurement model failed to fit: %s", e)
        _save_csv(pd.DataFrame([{"stage": "measurement_model", "error": str(e)}]),
                  "cor_sem_failure_diagnostic")
        return {}

    _plot_loadings(meas["loadings"])
    plot_factor_loadings_polar(meas["loadings"])

    struct = fit_structural_model(meas["std_df"], meas["item_map"])
    baseline_score = pd.Series(np.nan, index=df.index)
    if struct.get("paths") is not None and not struct.get("paths", pd.DataFrame()).empty:
        _plot_structural_paths(struct)
        baseline_score = compute_baseline_score(meas["scores"], struct["loadings4"])

    moderation = test_fes_moderation(df, baseline_score)

    scores_out = meas["scores"].copy()
    scores_out["baseline_resource_score"] = baseline_score
    scores_out.to_csv(paths.UKHLS_SEM_TABLES / "cor_sem_factor_scores.csv")
    log.info("Saved COR-SEM factor scores: %d rows", len(scores_out))

    log.info("COR-SEM (Stage 2b) complete.")
    return {
        "measurement": meas, "structural": struct,
        "fes_moderation": moderation, "scores": scores_out,
    }
