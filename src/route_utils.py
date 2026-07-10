"""
route_utils.py
──────────────
Shared helpers used by all three COR estimation routes — Route 1 (formative
composites, `src.enable_preprocessing`), Route 2 (reflective CFA/SEM latent
variables, `src.cor_sem`), Route 3 (COR-aligned deep VAE latent
representations, `src.cor_vae`) — and by `src.route_comparison`, so the
"same AEV formula, different estimation method" logic and the FES safeguard
both live in exactly one place.

FES safeguard
─────────────
FES is a macro-level scenario-based signal simulation layer (see
`src.fes_scenarios` for the full nine-scenario robustness layer), not a
household-level predictor: the UK-only ENABLE sample has a single annual
value per scenario, shared by every household (zero within-sample
variation), so none of the 9 FES scenario variants or their underlying
z-score components may ever enter a household-level model (SEM, VAE,
CatBoost, SHAP, mediation, or route-comparison). `exclude_fes_columns()` is
the single choke point that strips/warns on this — call it wherever a
feature list is assembled. FES and HighAEV are linked only through the
scenario-conditioned interpretation matrix built in
`outputs/fes_highaev_interpretation/`, never as a model feature.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

log = logging.getLogger(__name__)

# The UK-only ENABLE sample has one value per FES scenario per year, shared
# by every household — these columns (and their underlying per-series
# z-score components) must never be used as household-level model features
# (see module docstring). Covers all 9 named scenarios from
# `src.fes_scenarios.SCENARIO_DEFS` plus the raw z-components/forecasts they
# are built from.
FES_COLUMNS: set[str] = {
    # Legacy / annual-context columns (attached by enable_preprocessing.attach_fes_context)
    "fes_core", "fes_macro", "fes_actual",
    # 9 named FES scenarios (src.fes_scenarios.SCENARIO_DEFS)
    "fes_vw_core", "fes_vw_macro",
    "fes_bayes_core", "fes_bayes_macro",
    "fes_bayes_core_std", "fes_bayes_macro_std",
    "fes_bayes_core_lb", "fes_bayes_core_ub",
    "fes_bayes_macro_lb", "fes_bayes_macro_ub",
    "fes_actual_A", "fes_actual_B", "fes_actual_C",
    # Underlying per-series z-score components (core/macro/actual)
    "z_gas_core", "z_electricity_core", "z_carbon_core", "z_unc_core",
    "z_gas_macro", "z_electricity_macro", "z_carbon_macro", "z_unc_macro",
    "z_gas_actual", "z_electricity_actual", "z_carbon_actual", "z_real_vol_actual",
    "z_rv_actual_A", "z_rv_actual_B", "z_rv_actual_C",
    "z_unc_gas_core", "z_unc_electricity_core", "z_unc_carbon_core",
    "z_unc_gas_macro", "z_unc_electricity_macro", "z_unc_carbon_macro",
    # Raw forecast / realised-volatility inputs to the FES construction
    "rv_actual_A", "rv_actual_B", "rv_actual_C", "real_vol_actual",
    "forecast_gas_core", "forecast_electricity_core", "forecast_carbon_core",
    "forecast_gas_macro", "forecast_electricity_macro", "forecast_carbon_macro",
    "pi_halfwidth_gas_core", "pi_halfwidth_electricity_core", "pi_halfwidth_carbon_core",
    "pi_halfwidth_gas_macro", "pi_halfwidth_electricity_macro", "pi_halfwidth_carbon_macro",
    "actual_gas", "actual_electricity", "actual_carbon",
}

CONSTRUCTS: list[str] = ["FCP", "AEMC", "BLI", "TCR"]


def exclude_fes_columns(feature_cols: list[str]) -> list[str]:
    """
    Strip any FES column out of a candidate feature list, warning loudly if
    one was present. Catches every column in `FES_COLUMNS` plus, as a
    defensive fallback, any column whose name contains "fes_" (case
    insensitive) that isn't already in the explicit set — covers future FES
    scenario variants without needing another edit here. Matches by prefix
    (not substring) so an unrelated household column that merely contains
    "fes" somewhere in its name isn't swept up by mistake. FES is a
    macro-level scenario-simulation signal, not a household-level predictor
    — see module docstring.
    """
    def _is_fes(col: str) -> bool:
        return col in FES_COLUMNS or col.lower().startswith("fes_")

    found = [c for c in feature_cols if _is_fes(c)]
    if found:
        log.warning(
            "FES column(s) %s found in a model feature list — removing "
            "them automatically. FES is a macro-level scenario-based "
            "signal simulation (identical for every UK household within a "
            "given scenario), not a household-level predictor.",
            found,
        )
    return [c for c in feature_cols if not _is_fes(c)]


def minmax_scale_route_scores(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Min-max scale each column in `cols` to [0, 1] independently. Constant
    (zero-range) columns become all-NaN rather than divide-by-zero."""
    out = df.copy()
    for col in cols:
        lo, hi = out[col].min(), out[col].max()
        if pd.isna(lo) or pd.isna(hi) or (hi - lo) < 1e-10:
            out[col] = np.nan
        else:
            out[col] = (out[col] - lo) / (hi - lo)
    return out


def sign_canonicalize_against_route1(
    route_df: pd.DataFrame,
    route1_df: pd.DataFrame,
    constructs: list[str] = None,
) -> pd.DataFrame:
    """
    Flip the sign of each construct column in `route_df` so that "higher =
    more of the construct" holds consistently with Route 1's composites,
    which are already correctly signed by item-level recoding construction.

    This matters for Route 2 (a CFA loading's sign is statistically free)
    and Route 3 (the VAE's COR-alignment loss is `1 - |corr(...)|`,
    sign-agnostic by design) — a latent dimension can converge to either
    polarity. The flip only changes sign convention, not correlation
    magnitude, and is a no-op when a route already agrees with Route 1.
    """
    constructs = constructs or CONSTRUCTS
    out = route_df.copy()
    for construct in constructs:
        if construct not in out.columns or construct not in route1_df.columns:
            continue
        mask = out[construct].notna() & route1_df[construct].notna()
        if mask.sum() < 10:
            continue
        r, _ = pearsonr(out.loc[mask, construct], route1_df.loc[mask, construct])
        if pd.notna(r) and r < 0:
            out[construct] = -out[construct]
    return out


def compute_route_aev(scores01: pd.DataFrame, constructs: list[str] = None) -> pd.Series:
    """
    Apply Route 1's AEV formula — mean(FCP, BLI, TCR, 1-AEMC) — to any
    route's four construct columns. `scores01` must already be scaled to
    [0, 1] (`minmax_scale_route_scores`) and sign-canonicalized against
    Route 1 (`sign_canonicalize_against_route1`); this keeps the formula
    fixed across routes while letting each route supply its own dimension
    estimates.
    """
    constructs = constructs or CONSTRUCTS
    fcp, aemc, bli, tcr = (scores01[c] for c in constructs)
    return pd.concat([fcp, bli, tcr, 1.0 - aemc], axis=1).mean(axis=1)


def route_high_aev(aev: pd.Series, quantile: float) -> pd.Series:
    """Route-specific binary HighAEV flag at that route's own P-quantile threshold."""
    thresh = aev.quantile(quantile)
    high = (aev >= thresh).astype("Int64")
    high[aev.isna()] = pd.NA
    return high
