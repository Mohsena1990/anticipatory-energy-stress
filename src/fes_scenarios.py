"""
fes_scenarios.py
─────────────────
FES scenario-based signal simulation layer (Stage 4 extension).

FES is a macro-level scenario-based signal simulation, not a household-level
predictor. The nine FES variants produced by `src.fes_calculator` are treated
here as nine named scenarios:

  Forecasted signal scenarios (6)
  ────────────────────────────────
    equal_core, equal_macro         — equal-weight z-sum
    vw_core, vw_macro               — inverse-volatility weighted z-sum
    bayesian_core, bayesian_macro   — scalar Kalman-filter posterior mean

  Realised benchmark scenarios (3)
  ─────────────────────────────────
    actual_A, actual_B, actual_C    — three realised-volatility proxies

This module asks four descriptive questions of that scenario set:
  1. Is 2017 low / moderate / high energy-carbon stress under each construction?
  2. Which component drives the signal — gas, electricity, carbon, or
     uncertainty / realised volatility?
  3. How close are the forecasted scenarios to the realised benchmarks?
  4. Does the macro interpretation change across construction assumptions?

It does NOT simulate household behaviour and is never used as a household-
level feature — see `src.route_utils.exclude_fes_columns`, which is the
single enforcement point that keeps every column referenced here out of
SEM, VAE, CatBoost, SHAP, mediation, and route comparison.

Outputs
───────
  outputs/fes/fes_scenario_summary.csv
  outputs/fes/fes_scenario_monthly_states.csv
  outputs/fes/fes_scenario_forecast_vs_actual_matrix.csv
  outputs/fes/fes_scenario_interpretation_notes.csv
  outputs/figures/fes_scenario_trajectories.png
  outputs/figures/fes_scenario_annual_ranking.png
  outputs/figures/fes_component_contribution_heatmap.png
  outputs/figures/fes_forecast_actual_distance_heatmap.png

  outputs/fes_highaev_interpretation/scenario_highaev_interpretation_matrix.csv
  outputs/figures/fes_highaev_interpretation_matrix.png
  (built separately, from `household_stream.py`, once HighAEV is available —
  see `build_scenario_highaev_interpretation_matrix`)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.logging_utils import get_logger

log = get_logger("fes_scenarios")
warnings.filterwarnings("ignore")

MONTHS_SHORT = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Historical-baseline classification thresholds (z-score scale, relative to
# the 2005–2016 training distribution used throughout `fes_calculator.py`).
HIST_LOW_THRESH  = -0.50
HIST_HIGH_THRESH = 0.50

# Relative magnitude gap below which the two largest components are
# considered "mixed" rather than one component clearly dominating.
_MIXED_REL_GAP = 0.10

_DPI = 150

# ══════════════════════════════════════════════════════════════════════════════
# Scenario registry — maps each of the 9 named scenarios to its columns in
# outputs/fes/fes_monthly_2017.csv (produced by src.fes_calculator.compute_fes)
# ══════════════════════════════════════════════════════════════════════════════

SCENARIO_DEFS: dict[str, dict] = {
    "equal_core": dict(
        scenario_id=1, fes_col="fes_core",
        scenario_group="forecasted", method_family="equal_weight", source_mode="core",
        gas_col="z_gas_core", elec_col="z_electricity_core",
        carbon_col="z_carbon_core", unc_col="z_unc_core", unc_label="uncertainty",
    ),
    "equal_macro": dict(
        scenario_id=2, fes_col="fes_macro",
        scenario_group="forecasted", method_family="equal_weight", source_mode="macro",
        gas_col="z_gas_macro", elec_col="z_electricity_macro",
        carbon_col="z_carbon_macro", unc_col="z_unc_macro", unc_label="uncertainty",
    ),
    "vw_core": dict(
        scenario_id=3, fes_col="fes_vw_core",
        scenario_group="forecasted", method_family="volatility_weighted", source_mode="core",
        gas_col="z_gas_core", elec_col="z_electricity_core",
        carbon_col="z_carbon_core", unc_col="z_unc_core", unc_label="uncertainty",
    ),
    "vw_macro": dict(
        scenario_id=4, fes_col="fes_vw_macro",
        scenario_group="forecasted", method_family="volatility_weighted", source_mode="macro",
        gas_col="z_gas_macro", elec_col="z_electricity_macro",
        carbon_col="z_carbon_macro", unc_col="z_unc_macro", unc_label="uncertainty",
    ),
    "bayesian_core": dict(
        scenario_id=5, fes_col="fes_bayes_core",
        scenario_group="forecasted", method_family="bayesian", source_mode="core",
        gas_col="z_gas_core", elec_col="z_electricity_core",
        carbon_col="z_carbon_core", unc_col="z_unc_core", unc_label="uncertainty",
    ),
    "bayesian_macro": dict(
        scenario_id=6, fes_col="fes_bayes_macro",
        scenario_group="forecasted", method_family="bayesian", source_mode="macro",
        gas_col="z_gas_macro", elec_col="z_electricity_macro",
        carbon_col="z_carbon_macro", unc_col="z_unc_macro", unc_label="uncertainty",
    ),
    "actual_A": dict(
        scenario_id=7, fes_col="fes_actual_A",
        scenario_group="actual", method_family="actual_benchmark", source_mode="actual",
        gas_col="z_gas_actual", elec_col="z_electricity_actual",
        carbon_col="z_carbon_actual", unc_col="z_rv_actual_A", unc_label="realised_volatility",
    ),
    "actual_B": dict(
        scenario_id=8, fes_col="fes_actual_B",
        scenario_group="actual", method_family="actual_benchmark", source_mode="actual",
        gas_col="z_gas_actual", elec_col="z_electricity_actual",
        carbon_col="z_carbon_actual", unc_col="z_rv_actual_B", unc_label="realised_volatility",
    ),
    "actual_C": dict(
        scenario_id=9, fes_col="fes_actual_C",
        scenario_group="actual", method_family="actual_benchmark", source_mode="actual",
        gas_col="z_gas_actual", elec_col="z_electricity_actual",
        carbon_col="z_carbon_actual", unc_col="z_rv_actual_C", unc_label="realised_volatility",
    ),
}

FORECASTED_SCENARIOS = [k for k, v in SCENARIO_DEFS.items() if v["scenario_group"] == "forecasted"]
ACTUAL_SCENARIOS     = [k for k, v in SCENARIO_DEFS.items() if v["scenario_group"] == "actual"]


# ══════════════════════════════════════════════════════════════════════════════
# Part 4 — stress-state classification
# ══════════════════════════════════════════════════════════════════════════════

def classify_historical_fes_state(value: Optional[float]) -> Optional[str]:
    """
    Historical-baseline stress state, relative to the 2005–2016 training
    z-score distribution used throughout `fes_calculator.py`.

      FES <= -0.50            -> "low"
      -0.50 < FES < +0.50     -> "moderate_neutral"
      FES >= +0.50            -> "high"
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if value <= HIST_LOW_THRESH:
        return "low"
    if value >= HIST_HIGH_THRESH:
        return "high"
    return "moderate_neutral"


def classify_relative_2017_state(series: pd.Series) -> pd.Series:
    """
    Within-series relative (tercile) classification. Applied either to a
    scenario's own 12 monthly FES values (which months are relatively higher
    or lower within that scenario) or to the 9 scenarios' annual means
    (which scenario reads relatively higher or lower than the others) — the
    same generic tercile logic serves both use cases.

    Uses percentile rank (average-rank, ties split evenly) rather than
    `pd.qcut` so that it degrades gracefully with small n or repeated values.
    """
    s = pd.Series(series).astype(float)
    pct = s.rank(method="average", pct=True)

    def _label(p: float) -> Optional[str]:
        if pd.isna(p):
            return None
        if p <= 1 / 3:
            return "low_2017_relative"
        if p <= 2 / 3:
            return "medium_2017_relative"
        return "high_2017_relative"

    return pct.apply(_label)


def identify_dominant_component(
    gas: float,
    electricity: float,
    carbon: float,
    uncertainty_or_volatility: float,
    unc_label: str = "uncertainty",
) -> str:
    """
    Compare absolute component magnitudes and return which one drives the
    signal: "gas", "electricity", "carbon", "uncertainty",
    "realised_volatility", or "mixed" (top two components are within
    `_MIXED_REL_GAP` relative magnitude of each other).
    """
    vals = {
        "gas": gas, "electricity": electricity, "carbon": carbon,
        unc_label: uncertainty_or_volatility,
    }
    valid = {k: abs(v) for k, v in vals.items() if v is not None and not pd.isna(v)}
    if not valid:
        return "mixed"

    ranked = sorted(valid.items(), key=lambda kv: kv[1], reverse=True)
    top_name, top_val = ranked[0]
    if len(ranked) > 1 and top_val > 0:
        second_val = ranked[1][1]
        if (top_val - second_val) / top_val < _MIXED_REL_GAP:
            return "mixed"
    return top_name


# ══════════════════════════════════════════════════════════════════════════════
# Part 3 — scenario output tables
# ══════════════════════════════════════════════════════════════════════════════

def build_fes_scenario_summary(fes_monthly_df: pd.DataFrame) -> pd.DataFrame:
    """Builds fes_scenario_summary.csv — one row per FES scenario."""
    rows = []
    for name, meta in SCENARIO_DEFS.items():
        fes_col = meta["fes_col"]
        if fes_col not in fes_monthly_df.columns:
            continue

        annual_mean = float(fes_monthly_df[fes_col].mean())
        gas_mean    = float(fes_monthly_df[meta["gas_col"]].mean())   if meta["gas_col"]   in fes_monthly_df.columns else np.nan
        elec_mean   = float(fes_monthly_df[meta["elec_col"]].mean())  if meta["elec_col"]  in fes_monthly_df.columns else np.nan
        carbon_mean = float(fes_monthly_df[meta["carbon_col"]].mean()) if meta["carbon_col"] in fes_monthly_df.columns else np.nan
        unc_mean    = float(fes_monthly_df[meta["unc_col"]].mean())   if meta["unc_col"]   in fes_monthly_df.columns else np.nan

        dominant = identify_dominant_component(
            gas_mean, elec_mean, carbon_mean, unc_mean, meta["unc_label"]
        )
        hist_state = classify_historical_fes_state(annual_mean)

        rows.append({
            "scenario_id":         meta["scenario_id"],
            "scenario_name":       name,
            "scenario_group":      meta["scenario_group"],
            "method_family":       meta["method_family"],
            "source_mode":         meta["source_mode"],
            "annual_mean_fes":     round(annual_mean, 5),
            "annual_stress_state_historical": hist_state,
            "dominant_component":  dominant,
            "gas_component_mean":         round(gas_mean, 5)    if not np.isnan(gas_mean)    else np.nan,
            "electricity_component_mean": round(elec_mean, 5)   if not np.isnan(elec_mean)   else np.nan,
            "carbon_component_mean":      round(carbon_mean, 5) if not np.isnan(carbon_mean) else np.nan,
            "uncertainty_or_volatility_mean": round(unc_mean, 5) if not np.isnan(unc_mean)   else np.nan,
        })

    summary_df = pd.DataFrame(rows).sort_values("scenario_id").reset_index(drop=True)

    # Relative-2017 state: where this scenario's annual mean ranks among the
    # 9 scenario constructions (robustness question 4 — does the macro
    # interpretation change across construction assumptions?).
    summary_df["annual_stress_state_2017_relative"] = classify_relative_2017_state(
        summary_df["annual_mean_fes"]
    )

    notes = []
    for _, row in summary_df.iterrows():
        rel = str(row["annual_stress_state_2017_relative"]).replace("_2017_relative", "")
        notes.append(
            f"Under the {row['scenario_name']} construction, 2017 UK macro "
            f"energy-carbon stress is classified as {row['annual_stress_state_historical']} "
            f"relative to the 2005-2016 baseline, and {rel} relative to the other "
            f"FES scenario constructions. The signal is primarily driven by the "
            f"{row['dominant_component']} component. This is a macro-level "
            f"signal-construction scenario, not a household-level predictor."
        )
    summary_df["interpretation_note"] = notes

    col_order = [
        "scenario_id", "scenario_name", "scenario_group", "method_family", "source_mode",
        "annual_mean_fes", "annual_stress_state_historical", "annual_stress_state_2017_relative",
        "dominant_component", "gas_component_mean", "electricity_component_mean",
        "carbon_component_mean", "uncertainty_or_volatility_mean", "interpretation_note",
    ]
    return summary_df[col_order]


def build_fes_scenario_monthly_states(fes_monthly_df: pd.DataFrame) -> pd.DataFrame:
    """Builds fes_scenario_monthly_states.csv — long format, one row per month per scenario."""
    parts = []
    for name, meta in SCENARIO_DEFS.items():
        fes_col = meta["fes_col"]
        needed = [fes_col, meta["gas_col"], meta["elec_col"], meta["carbon_col"], meta["unc_col"]]
        if not all(c in fes_monthly_df.columns for c in needed):
            continue

        sub = fes_monthly_df[["date"] + needed].copy()
        sub.columns = [
            "date", "fes_value", "gas_component", "electricity_component",
            "carbon_component", "uncertainty_or_volatility_component",
        ]
        sub["scenario_name"]  = name
        sub["scenario_group"] = meta["scenario_group"]
        sub["method_family"]  = meta["method_family"]
        sub["historical_state"]  = sub["fes_value"].apply(classify_historical_fes_state)
        sub["relative_2017_state"] = classify_relative_2017_state(sub["fes_value"])
        sub["dominant_component_month"] = sub.apply(
            lambda r: identify_dominant_component(
                r["gas_component"], r["electricity_component"],
                r["carbon_component"], r["uncertainty_or_volatility_component"],
                meta["unc_label"],
            ),
            axis=1,
        )
        parts.append(sub)

    long_df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    col_order = [
        "date", "scenario_name", "scenario_group", "method_family", "fes_value",
        "historical_state", "relative_2017_state", "dominant_component_month",
        "gas_component", "electricity_component", "carbon_component",
        "uncertainty_or_volatility_component",
    ]
    return long_df[col_order] if not long_df.empty else long_df


def _pairwise_metrics(f_vals: np.ndarray, a_vals: np.ndarray) -> Optional[dict]:
    try:
        from scipy import stats as sp
    except ImportError:
        sp = None

    mask = ~(np.isnan(f_vals) | np.isnan(a_vals))
    if mask.sum() < 3:
        return None
    f_m, a_m = f_vals[mask], a_vals[mask]

    mae  = float(np.mean(np.abs(f_m - a_m)))
    rmse = float(np.sqrt(np.mean((f_m - a_m) ** 2)))
    bias = float(np.mean(f_m - a_m))
    mdev = float(np.max(np.abs(f_m - a_m)))

    if sp is not None:
        try:
            pr, _ = sp.pearsonr(f_m, a_m)
            sr, _ = sp.spearmanr(f_m, a_m)
        except Exception:
            pr = sr = np.nan
    else:
        pr = sr = np.nan

    ss_res = float(np.sum((f_m - a_m) ** 2))
    ss_tot = float(np.sum((a_m - a_m.mean()) ** 2))
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 1e-10 else np.nan

    naive      = np.concatenate([[a_m[0]], a_m[:-1]])
    rmse_naive = float(np.sqrt(np.mean((naive - a_m) ** 2)))
    theil_u    = rmse / rmse_naive if rmse_naive > 1e-10 else np.nan

    return {
        "MAE": mae, "RMSE": rmse, "Bias": bias, "MaxDev": mdev,
        "Pearson_r": pr, "Spearman_r": sr, "R2": r2, "Theil_U": theil_u,
    }


def compare_forecast_scenarios_to_actual_benchmarks(fes_monthly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds fes_scenario_forecast_vs_actual_matrix.csv — long format:
    forecast_scenario | actual_benchmark | metric | value

    Six forecasted scenarios × three realised benchmark scenarios × 8 metrics.
    """
    rows = []
    for f_name in FORECASTED_SCENARIOS:
        f_col = SCENARIO_DEFS[f_name]["fes_col"]
        if f_col not in fes_monthly_df.columns:
            continue
        fv = fes_monthly_df[f_col].values.astype(float)

        for a_name in ACTUAL_SCENARIOS:
            a_col = SCENARIO_DEFS[a_name]["fes_col"]
            if a_col not in fes_monthly_df.columns:
                continue
            av = fes_monthly_df[a_col].values.astype(float)

            metrics = _pairwise_metrics(fv, av)
            if metrics is None:
                continue
            for metric_name, value in metrics.items():
                rows.append({
                    "forecast_scenario": f_name,
                    "actual_benchmark":  a_name,
                    "metric":            metric_name,
                    "value":             round(value, 5) if not (value is None or np.isnan(value)) else np.nan,
                })

    return pd.DataFrame(rows)


def build_fes_interpretation_notes(fes_scenario_summary_df: pd.DataFrame) -> pd.DataFrame:
    """Builds fes_scenario_interpretation_notes.csv — one row per scenario."""
    caution = (
        "This scenario represents a macro-level signal-construction assumption. "
        "It is not used as a household-level predictor and does not simulate "
        "household behavioural response."
    )
    rows = []
    for _, row in fes_scenario_summary_df.iterrows():
        stress_state = row["annual_stress_state_historical"]
        dominant     = row["dominant_component"]
        group_label  = "forecasted (ex-ante)" if row["scenario_group"] == "forecasted" else "realised (ex-post benchmark)"

        scenario_interpretation = (
            f"Under the {row['scenario_name']} {group_label} construction, 2017 UK "
            f"macro energy-carbon conditions read as {stress_state} stress relative "
            f"to the 2005-2016 baseline, driven primarily by the {dominant} component."
        )
        policy_relevance = (
            f"If this {group_label} construction were used for anticipatory energy-"
            f"poverty policy planning, it would flag 2017 as a {stress_state}-stress "
            f"period, informing how urgently pre-emptive household support measures "
            f"might be prioritised — a macro signal for timing/targeting, not a "
            f"household-level forecast."
        )
        rows.append({
            "scenario_name":          row["scenario_name"],
            "stress_state":           stress_state,
            "dominant_component":     dominant,
            "scenario_interpretation": scenario_interpretation,
            "policy_relevance":       policy_relevance,
            "caution_note":           caution,
        })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# Orchestrator — Stage 4 scenario simulation (called from fes_calculator.compute_fes)
# ══════════════════════════════════════════════════════════════════════════════

def run_fes_scenario_simulation(
    fes_monthly_df: pd.DataFrame,
    out_dir: str = "outputs/fes",
    figures_dir: str = "outputs/figures",
) -> dict:
    """
    Build all four FES scenario tables + four scenario figures from the
    already-computed monthly FES DataFrame (`fes_calculator.compute_fes`'s
    return value). Returns the four DataFrames for reuse (e.g. by
    `household_stream.py`'s scenario-to-HighAEV interpretation matrix).
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(figures_dir).mkdir(parents=True, exist_ok=True)

    summary_df = build_fes_scenario_summary(fes_monthly_df)
    summary_path = f"{out_dir}/fes_scenario_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    log.info("FES scenario summary saved -> %s (%d scenarios)", summary_path, len(summary_df))

    monthly_states_df = build_fes_scenario_monthly_states(fes_monthly_df)
    monthly_path = f"{out_dir}/fes_scenario_monthly_states.csv"
    monthly_states_df.to_csv(monthly_path, index=False)
    log.info("FES scenario monthly states saved -> %s (%d rows)", monthly_path, len(monthly_states_df))

    comparison_df = compare_forecast_scenarios_to_actual_benchmarks(fes_monthly_df)
    comparison_path = f"{out_dir}/fes_scenario_forecast_vs_actual_matrix.csv"
    comparison_df.to_csv(comparison_path, index=False)
    log.info("FES scenario forecast-vs-actual matrix saved -> %s (%d rows)", comparison_path, len(comparison_df))

    notes_df = build_fes_interpretation_notes(summary_df)
    notes_path = f"{out_dir}/fes_scenario_interpretation_notes.csv"
    notes_df.to_csv(notes_path, index=False)
    log.info("FES scenario interpretation notes saved -> %s", notes_path)

    try:
        plot_scenario_trajectories(fes_monthly_df, f"{figures_dir}/fes_scenario_trajectories.png")
    except Exception as e:
        log.warning("Scenario trajectories figure failed: %s", e)
    try:
        plot_scenario_annual_ranking(summary_df, f"{figures_dir}/fes_scenario_annual_ranking.png")
    except Exception as e:
        log.warning("Scenario annual ranking figure failed: %s", e)
    try:
        plot_component_contribution_heatmap(fes_monthly_df, f"{figures_dir}/fes_component_contribution_heatmap.png")
    except Exception as e:
        log.warning("Component contribution heatmap failed: %s", e)
    try:
        plot_forecast_actual_distance_heatmap(
            comparison_df, f"{figures_dir}/fes_forecast_actual_distance_heatmap.png", metric="RMSE"
        )
    except Exception as e:
        log.warning("Forecast-actual distance heatmap failed: %s", e)

    return {
        "summary": summary_df,
        "monthly_states": monthly_states_df,
        "comparison": comparison_df,
        "notes": notes_df,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Part 7 — scenario-to-HighAEV interpretation (household-level, interpretive only)
# ══════════════════════════════════════════════════════════════════════════════

def build_scenario_highaev_interpretation_matrix(
    fes_scenario_summary_df: pd.DataFrame,
    household_df: pd.DataFrame,
    route_comparison_csv: Optional[str] = None,
) -> pd.DataFrame:
    """
    Builds scenario_highaev_interpretation_matrix.csv.

    This does NOT model or predict HighAEV. HighAEV prevalence is already
    estimated (once) by the household routes; this function repeats that
    same prevalence across all 9 FES scenario rows and only varies the
    macro-context interpretation text — the FES scenario changes the
    interpretation, not the household label.
    """
    def _prevalence(col: str) -> float:
        if col not in household_df.columns:
            return np.nan
        s = household_df[col].dropna()
        return float(s.mean()) if len(s) else np.nan

    prev1 = _prevalence("high_aev")
    prev2 = _prevalence("sem_high_aev")
    prev3 = _prevalence("vae_high_aev")

    route_agreement_note = "Route comparison output not available for this run."
    if route_comparison_csv and Path(route_comparison_csv).exists():
        try:
            agree_df = pd.read_csv(route_comparison_csv)
            if "cohen_kappa" in agree_df.columns:
                kappa_mean = float(agree_df["cohen_kappa"].mean())
                route_agreement_note = (
                    f"Cross-route HighAEV agreement (mean Cohen's kappa across "
                    f"available route pairs): {kappa_mean:.3f}."
                )
            else:
                route_agreement_note = "Route comparison output present but no kappa column found."
        except Exception as e:
            route_agreement_note = f"Route comparison output could not be read: {e}"

    caution = (
        "This scenario represents a macro-level signal-construction assumption. "
        "It is not used as a household-level predictor and does not simulate "
        "household behavioural response. HighAEV labels do not change across "
        "FES scenarios — only the macro-stress interpretation does."
    )

    rows = []
    for _, row in fes_scenario_summary_df.iterrows():
        stress_state = row["annual_stress_state_historical"]
        interpretation = (
            f"Under this macro signal scenario ({row['scenario_name']}), the "
            f"HighAEV group represents households with structurally high adaptive "
            f"vulnerability observed under a {stress_state} national energy-carbon "
            f"stress context."
        )
        policy_implication = (
            f"Under {stress_state} macro stress ({row['scenario_name']}), the "
            f"HighAEV group is the policy-relevant target for anticipatory support — "
            f"whether that stress reading is itself high or low does not change who "
            f"is already structurally vulnerable."
        )
        rows.append({
            "scenario_name":         row["scenario_name"],
            "scenario_group":        row["scenario_group"],
            "annual_mean_fes":       row["annual_mean_fes"],
            "stress_state":          stress_state,
            "dominant_macro_signal": row["dominant_component"],
            "high_aev_prevalence_route1": round(prev1, 5) if not np.isnan(prev1) else np.nan,
            "high_aev_prevalence_route2_if_available": round(prev2, 5) if not np.isnan(prev2) else np.nan,
            "high_aev_prevalence_route3_if_available": round(prev3, 5) if not np.isnan(prev3) else np.nan,
            "route_agreement_note":  route_agreement_note,
            "interpretation":        interpretation,
            "policy_implication":    policy_implication,
            "caution_note":          caution,
        })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# Part 6 — figures
# ══════════════════════════════════════════════════════════════════════════════

_FORECAST_STYLES = {
    "equal_core":     dict(color="#8E44AD", ls="-",   marker="o"),
    "equal_macro":    dict(color="#E74C3C", ls="--",  marker="s"),
    "vw_core":        dict(color="#16A085", ls="-",   marker="^"),
    "vw_macro":       dict(color="#E91E63", ls="--",  marker="v"),
    "bayesian_core":  dict(color="#00ACC1", ls="-.",  marker="D"),
    "bayesian_macro": dict(color="#FF6F00", ls=":",   marker="P"),
}
_ACTUAL_STYLES = {
    "actual_A": dict(color="#2C3E50", ls="-",  marker="X"),
    "actual_B": dict(color="#7B241C", ls="--", marker="X"),
    "actual_C": dict(color="#145A32", ls="-.", marker="X"),
}
_STATE_COLOURS = {"low": "#2E86C1", "moderate_neutral": "#95A5A6", "high": "#C0392B"}
_GRID = "#EAECEE"


def _save_fig(fig: plt.Figure, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    fig.savefig(Path(path).with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    log.info("Figure saved -> %s", path)


def plot_scenario_trajectories(fes_monthly_df: pd.DataFrame, out_path: str) -> None:
    """Figure 1 — monthly trajectories of all nine FES scenarios across 2017."""
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    months = MONTHS_SHORT[:len(fes_monthly_df)]

    forecast_handles, forecast_labels = [], []
    for name in FORECASTED_SCENARIOS:
        col = SCENARIO_DEFS[name]["fes_col"]
        if col not in fes_monthly_df.columns:
            continue
        style = _FORECAST_STYLES[name]
        line, = ax.plot(months, fes_monthly_df[col].values, linewidth=2.0,
                         markersize=5, label=name, **style)
        forecast_handles.append(line)
        forecast_labels.append(name)

    actual_handles, actual_labels = [], []
    for name in ACTUAL_SCENARIOS:
        col = SCENARIO_DEFS[name]["fes_col"]
        if col not in fes_monthly_df.columns:
            continue
        style = _ACTUAL_STYLES[name]
        line, = ax.plot(months, fes_monthly_df[col].values, linewidth=2.6,
                         markersize=7, label=name, **style)
        actual_handles.append(line)
        actual_labels.append(name)

    ax.axhline(0, color="#95A5A6", linewidth=0.9)
    ax.set_title("FES Scenario Simulation — Monthly Trajectories, 2017",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Month (2017)", fontsize=11)
    ax.set_ylabel("FES value (z-score scale)", fontsize=11)
    ax.grid(True, color=_GRID, linewidth=0.8)
    ax.tick_params(axis="both", labelsize=9)

    leg1 = ax.legend(forecast_handles, forecast_labels, title="Forecasted signal scenarios",
                      loc="upper left", fontsize=8, title_fontsize=9, framealpha=0.92)
    ax.add_artist(leg1)
    ax.legend(actual_handles, actual_labels, title="Realised benchmark scenarios",
              loc="lower left", fontsize=8, title_fontsize=9, framealpha=0.92)

    _save_fig(fig, out_path)


def plot_scenario_annual_ranking(scenario_summary_df: pd.DataFrame, out_path: str) -> None:
    """Figure 2 — annual mean FES per scenario, ranked lowest to highest, stress state labelled."""
    df = scenario_summary_df.sort_values("annual_mean_fes").reset_index(drop=True)
    colours = [_STATE_COLOURS.get(s, "#95A5A6") for s in df["annual_stress_state_historical"]]

    fig, ax = plt.subplots(figsize=(13, 6.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    _short_state = {"low": "low", "moderate_neutral": "neutral", "high": "high"}
    bars = ax.bar(df["scenario_name"], df["annual_mean_fes"], color=colours,
                   edgecolor="white", linewidth=0.8, width=0.65)
    for bar, state, val in zip(bars, df["annual_stress_state_historical"], df["annual_mean_fes"]):
        label = _short_state.get(state, state)
        offset = 0.05 if val >= 0 else -0.05
        va = "bottom" if val >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
                label, ha="center", va=va, fontsize=9, fontweight="bold", rotation=90)

    ax.axhline(0, color="#5D6D7E", linewidth=1.0)
    ax.axhline(HIST_HIGH_THRESH, color="#C0392B", linewidth=0.8, linestyle=":", alpha=0.6)
    ax.axhline(HIST_LOW_THRESH, color="#2E86C1", linewidth=0.8, linestyle=":", alpha=0.6)
    ax.set_title("FES Scenario Simulation — Annual Ranking, 2017", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("Annual mean FES (z-score scale)", fontsize=11)
    ax.tick_params(axis="x", rotation=30, labelsize=9)
    ax.grid(axis="y", color=_GRID, linewidth=0.8)

    _save_fig(fig, out_path)


def plot_component_contribution_heatmap(fes_monthly_df: pd.DataFrame, out_path: str) -> None:
    """Figure 3 — annual mean z-score contribution: rows core/macro/actual, columns gas/elec/carbon/unc/FES total."""
    row_defs = {
        "core":   dict(gas="z_gas_core", elec="z_electricity_core", carbon="z_carbon_core",
                        unc="z_unc_core", fes="fes_core"),
        "macro":  dict(gas="z_gas_macro", elec="z_electricity_macro", carbon="z_carbon_macro",
                        unc="z_unc_macro", fes="fes_macro"),
        "actual": dict(gas="z_gas_actual", elec="z_electricity_actual", carbon="z_carbon_actual",
                        unc="z_real_vol_actual", fes="fes_actual"),
    }
    col_labels = ["Gas", "Electricity", "Carbon", "Uncertainty /\nRealised Vol", "FES total"]

    data = []
    row_labels = []
    for label, cols in row_defs.items():
        vals = [
            float(fes_monthly_df[cols[k]].mean()) if cols[k] in fes_monthly_df.columns else np.nan
            for k in ("gas", "elec", "carbon", "unc", "fes")
        ]
        data.append(vals)
        row_labels.append(label)

    arr = np.array(data, dtype=float)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    vmax = np.nanmax(np.abs(arr)) or 1.0
    im = ax.imshow(arr, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)
    plt.colorbar(im, ax=ax, label="Annual mean z-score contribution")

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=10)

    for r in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            v = arr[r, c]
            if not np.isnan(v):
                ax.text(c, r, f"{v:+.2f}", ha="center", va="center", fontsize=9, fontweight="bold")

    ax.set_title("FES Component Contribution — Annual Mean Z-Scores, 2017", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    _save_fig(fig, out_path)


def plot_forecast_actual_distance_heatmap(
    comparison_long_df: pd.DataFrame, out_path: str, metric: str = "RMSE"
) -> None:
    """Figure 4 — six forecasted scenarios × three actual benchmarks, cell = chosen metric."""
    if comparison_long_df.empty:
        return
    sub = comparison_long_df[comparison_long_df["metric"] == metric]
    if sub.empty:
        return

    pivot = sub.pivot(index="forecast_scenario", columns="actual_benchmark", values="value")
    pivot = pivot.reindex(index=[s for s in FORECASTED_SCENARIOS if s in pivot.index],
                          columns=[s for s in ACTUAL_SCENARIOS if s in pivot.columns])

    vals = pivot.values.astype(float)
    vmin, vmax = np.nanmin(vals), np.nanmax(vals)
    if np.isnan(vmin) or np.isnan(vmax):
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    im = ax.imshow(vals, cmap="YlOrRd", aspect="auto", vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label=f"{metric} (lower = closer to realised)")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(pivot.index, fontsize=9)

    mid = (vmin + vmax) / 2
    for r in range(len(pivot.index)):
        for c in range(len(pivot.columns)):
            v = vals[r, c]
            if not np.isnan(v):
                colour = "white" if v > mid else "black"
                ax.text(c, r, f"{v:.3f}", ha="center", va="center",
                        fontsize=9, fontweight="bold", color=colour)

    ax.set_title(f"Forecasted vs Realised FES Scenarios — {metric}", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Realised benchmark scenario", fontsize=10)
    ax.set_ylabel("Forecasted signal scenario", fontsize=10)
    plt.tight_layout()
    _save_fig(fig, out_path)


def plot_highaev_interpretation_matrix(matrix_df: pd.DataFrame, out_path: str) -> None:
    """Figure 5 — compact table figure linking FES scenarios to HighAEV prevalence/interpretation."""
    import textwrap

    if matrix_df.empty:
        return

    def _wrap(text: str, width: int = 34, max_chars: int = 170) -> str:
        text = str(text)
        if len(text) > max_chars:
            text = text[: max_chars - 1].rsplit(" ", 1)[0] + "…"
        return "\n".join(textwrap.wrap(text, width=width))

    header = ["Scenario", "Stress\nstate", "Dominant\nsignal", "HighAEV\nprev. (R1)",
              "Interpretation", "Policy implication"]

    cell_text = []
    max_lines = 1
    for _, row in matrix_df.iterrows():
        prev = row["high_aev_prevalence_route1"]
        prev_str = f"{prev:.1%}" if pd.notna(prev) else "n/a"
        interp = _wrap(row["interpretation"])
        policy = _wrap(row["policy_implication"])
        max_lines = max(max_lines, interp.count("\n") + 1, policy.count("\n") + 1)
        cell_text.append([
            row["scenario_name"], row["stress_state"], row["dominant_macro_signal"],
            prev_str, interp, policy,
        ])

    n_rows = len(cell_text) + 1  # + header
    row_h_inch = 0.20 * max_lines + 0.12
    title_h_inch = 0.5
    fig_h = n_rows * row_h_inch + title_h_inch
    fig = plt.figure(figsize=(20, fig_h))
    fig.patch.set_facecolor("white")

    ax = fig.add_axes([0.01, 0.01, 0.98, (fig_h - title_h_inch) / fig_h])
    ax.axis("off")

    col_widths = [0.08, 0.09, 0.10, 0.08, 0.325, 0.325]
    table = ax.table(cellText=cell_text, colLabels=header, loc="center",
                      cellLoc="left", colWidths=col_widths, bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#D5D8DC")
        cell.PAD = 0.015
        if r == 0:
            cell.set_facecolor("#2C3E50")
            cell.set_text_props(color="white", fontweight="bold", ha="center", va="center")
        else:
            cell.set_facecolor("#F8F9F9" if r % 2 == 0 else "white")
            cell.set_text_props(va="center")

    fig.suptitle(
        "FES Scenarios × HighAEV — Scenario-Conditioned Interpretation "
        "(not a household-level prediction)",
        fontsize=12, fontweight="bold", y=0.995,
    )
    _save_fig(fig, out_path)
