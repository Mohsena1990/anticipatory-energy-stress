"""
fes_calculator.py
─────────────────
Compute the equal-weighted Forecasted Energy-Carbon Stress (FES) index for
the target forecast year -- auto-detected from the data by
forecast_pipeline.py (see _compute_default_window), not a fixed calendar
year.

Two analytical baselines
─────────────────────────
  FES_core   — built from core-only model forecasts (target series only)
  FES_macro  — built from macro-augmented model forecasts (core + exogenous)
  FES_actual — built from the target year's realised values (benchmark)

Formula
───────
  FES_t = z(GasGrowth_t) + z(ElecGrowth_t) + z(CarbonGrowth_t) + z(Uncertainty_t)

  For FES_actual the uncertainty term is replaced by realised cross-sectional
  volatility:
    RealVol_t = std(z_gas_t, z_elec_t, z_carbon_t)   [across the three series]

Z-score standardisation
───────────────────────
  All z-scores use the TRAINING period (train_start..train_end, passed in
  by the caller) mean and std so that FES_core, FES_macro, and FES_actual
  are directly comparable.

    z(X_t) = (X_t − μ_train) / σ_train

  Uncertainty z-scores use the rolling 12-month std of the training series as
  the reference distribution for forecast interval half-widths.

Outputs (CSV only, named by the actual target_year)
────────────────────────────────────────────────────
  outputs/fes/fes_monthly_{target_year}.csv       — 12 rows × all z-components + FES
  outputs/fes/fes_summary_{target_year}.csv       — annual mean FES and components
  outputs/fes/fes_components_table.csv   — cross-baseline comparison table
  outputs/figures/fes_monthly_{target_year}.png   — FES time-series (equal-weight variants)
  outputs/figures/fes_components_{target_year}.png — component breakdown bars
  outputs/figures/forecast_vs_actual_{series}.png  — per-series forecast plot

Simplified from an earlier version that also computed volatility-weighted
(VW) and Bayesian-Kalman-filtered FES variants, plus a downstream 9-scenario
simulation (`src/fes_scenarios.py`) and TS-SHAP attribution
(`src/ts_shap.py`) — dropped as unused overhead; only the equal-weighted
FES_core/FES_macro/FES_actual are produced now. `fes_scenarios.py` and
`ts_shap.py` have since been removed from the repo entirely, recoverable
from git history if ever needed for comparison.
"""

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from src.logging_utils import get_logger

log = get_logger("fes")
warnings.filterwarnings("ignore")

SERIES         = ["gas", "electricity", "carbon"]
MODES          = ["core", "macro"]
# Legacy fallback defaults only -- forecast_pipeline.py always passes
# train_start/train_end/forecast_dates explicitly (computed dynamically from
# the data's own date range; see forecast_pipeline._compute_default_window),
# so these constants are never actually used by the real pipeline.
TRAIN_START    = "2005-01-01"
TRAIN_END      = "2016-12-01"
FORECAST_DATES = pd.date_range("2017-01-01", periods=12, freq="MS")


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _find_best_models(ranked_df: pd.DataFrame) -> dict:
    """Return {(series, mode): model_name} using the configured selection score."""
    best: dict = {}
    score_col = "selection_score" if "selection_score" in ranked_df.columns else "rank_score"
    for (series, mode), grp in ranked_df.groupby(["series_name", "mode"]):
        best[(series, mode)] = grp.loc[grp[score_col].idxmin(), "model"]
    log.info("Best models selected:")
    for (s, m), mdl in best.items():
        log.info(f"  [{s}][{m}] → {mdl}")
    return best


def _select_best_mode_per_series(ranked_df: pd.DataFrame, best: dict) -> dict:
    """
    Per series, pick whichever mode -- core or macro -- has the lower
    forecast error for its own best model. Independent of, and in addition
    to, _find_best_models' per-(series, mode) best-MODEL choice: this picks
    the best MODE for each series (gas/electricity/carbon may each end up
    core or macro), so a single composite forecast (FES_selected) can use
    gas's better mode, electricity's better mode, and carbon's better mode
    together instead of committing the whole index to one mode.

    Deliberately does NOT compare `selection_score`/`rank_score` across
    modes: those are ranks computed WITHIN each (series, mode) group (see
    model_evaluation.apply_selection_scores), so the winning model in every
    group always has rank 1 there regardless of how good or bad that group
    is overall -- comparing "minimum rank" across the core vs macro groups
    is therefore always a 1-vs-1 tie, which silently resolved to whichever
    mode pandas' groupby visited first (an earlier version of this function
    did exactly that, and always picked 'core'). Compares the two winning
    models' actual error on a shared, directly comparable scale across
    modes instead -- which error metric depends on `selection_basis`
    (read from ranked_df, set by model_evaluation.apply_selection_scores):

      'forecast_actual' -- uses forecast_actual_MAE (post-hoc accuracy
        against known target-year actuals). Only meaningful for a
        retrospective/backtested target year; deliberately hindsight and
        should not be the default (see forecast_pipeline.run's
        selection_basis default and its rationale).
      'validation' (default) -- uses RMSE from the genuine walk-forward
        backtest, computed using only pre-target-year information, so the
        mode choice stays honestly ex-ante even when forecast_actual_MAE
        happens to be available (e.g. when re-running FES for an
        already-realised year for reporting purposes).

    Previously this always preferred forecast_actual_MAE whenever present
    regardless of selection_basis, which meant FES_selected's per-series
    mode choice was silently hindsight-biased even when the caller had
    explicitly asked for genuine walk-forward ('validation') selection.
    """
    # selection_basis is set PER (series, mode) group by apply_selection_scores
    # (a group falls back to "validation_fallback" independently of its
    # neighbours when only that group's forecast_actual data is entirely
    # missing) -- reading a single global basis off ranked_df's first row and
    # applying it to every series/mode let one fallback group silently
    # downgrade every other, genuinely forecast_actual-tagged group to RMSE
    # too. Resolve metric_col per (series, mode) row instead.
    selected: dict = {}
    used_metrics: dict = {}
    for series in SERIES:
        scores: dict = {}
        for mode in MODES:
            model = best.get((series, mode))
            if model is None:
                continue
            row = ranked_df[
                (ranked_df["series_name"] == series)
                & (ranked_df["mode"] == mode)
                & (ranked_df["model"] == model)
            ]
            if row.empty:
                continue
            row0 = row.iloc[0]
            basis = row0.get("selection_basis", "validation")
            metric_col = (
                "forecast_actual_MAE"
                if basis == "forecast_actual" and "forecast_actual_MAE" in ranked_df.columns
                else "RMSE"
            )
            used_metrics[(series, mode)] = metric_col
            val = row0.get(metric_col, np.nan)
            if pd.isna(val) and metric_col != "RMSE":
                val = row0.get("RMSE", np.nan)
            if not pd.isna(val):
                scores[mode] = float(val)
        if scores:
            selected[series] = min(scores, key=scores.get)
    log.info("Per-series mode selected for FES_selected (metric per series/mode: %s): %s",
              used_metrics, selected)
    return selected


def _compute_series_weights(ranked_df: pd.DataFrame, selected_mode: dict) -> dict:
    """
    Data-driven weights for the new FES_Weighted variant -- added
    ALONGSIDE the existing equal-weighted variants (FES_core/FES_macro/
    FES_selected), never replacing them. An earlier version of this
    project's volatility-weighted/Bayesian-Kalman FES variants were
    deliberately dropped as unused overhead (see this module's own
    docstring); this is a narrower, more transparent reintroduction --
    ONE inverse-validation-RMSE weighted variant, computed openly and
    compared against the other three in fes_comparison_metrics.csv
    exactly like Core vs Macro vs Selected already are, so "does weighting
    actually help" stays an inspectable, falsifiable question rather than
    an assumed improvement.

    For each series, takes the RMSE of the model already chosen as that
    series' best (via _select_best_mode_per_series's own mode choice) --
    the genuine walk-forward validation RMSE, not a hindsight metric.
    weight = (1/RMSE) / mean(1/RMSE across series), normalized to mean 1.0
    so FES_Weighted sums to roughly the same scale as the unit-weighted
    variants (each of which implicitly uses weight=1 per series).
    """
    weights: dict = {}
    for series in SERIES:
        mode = selected_mode.get(series, "core")
        row = ranked_df[
            (ranked_df["series_name"] == series) & (ranked_df["mode"] == mode)
        ]
        if row.empty:
            weights[series] = np.nan
            continue
        # Whichever model won this (series, mode) -- lowest selection_score
        # if present, else rank_score.
        score_col = "selection_score" if "selection_score" in row.columns else "rank_score"
        best_row = row.loc[row[score_col].idxmin()]
        rmse = float(best_row.get("RMSE", np.nan))
        weights[series] = 1.0 / rmse if rmse and rmse > 1e-10 else np.nan

    valid = {s: w for s, w in weights.items() if not np.isnan(w)}
    if not valid:
        log.warning("No valid RMSE found for any series -- FES_Weighted falls back to equal weights (1.0 each)")
        return {s: 1.0 for s in SERIES}

    mean_inv_rmse = float(np.mean(list(valid.values())))
    normalized = {
        s: (w / mean_inv_rmse if not np.isnan(w) else 1.0)
        for s, w in weights.items()
    }
    log.info("FES_Weighted series weights (inverse validation-RMSE, normalized to mean 1.0): %s", normalized)
    return normalized


def _compute_actual_fes_for_window(
    core_df: pd.DataFrame, stats: dict, window_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Realised FES_actual (same formula as the FES_actual block in
    compute_fes) computed for an arbitrary calendar window instead of the
    forecast target window -- used to get FES_actual for the training-cutoff
    year (year x) as a baseline alongside FES_selected's year x+1 forecast,
    without re-running the whole compute_fes pipeline a second time.
    """
    rows = []
    for date in window_dates:
        row: dict = {"date": date}
        z_actual = []
        for series in SERIES:
            st  = stats[series]
            col = f"{series}_growth"
            sub = core_df[core_df["date"] == date]
            act_val = (
                float(sub[col].iloc[0])
                if not sub.empty and col in sub.columns and pd.notna(sub[col].iloc[0])
                else np.nan
            )
            z_a = _zscore_array(np.array([act_val]), st["mean"], st["std"])[0] \
                  if not np.isnan(act_val) else np.nan
            row[f"actual_{series}"]   = round(act_val, 5) if not np.isnan(act_val) else np.nan
            row[f"z_{series}_actual"] = round(float(z_a), 5) if not np.isnan(z_a) else np.nan
            z_actual.append(z_a)

        z_actual_arr = np.array([v for v in z_actual if not np.isnan(v)])
        real_vol = float(np.std(z_actual_arr)) if len(z_actual_arr) > 1 else np.nan
        row["real_vol_actual"] = round(real_vol, 5) if not np.isnan(real_vol) else np.nan
        rv_ref = stats.get("_real_vol", {"mean": 0.0, "std": 1.0})
        z_real_vol = _zscore_array(
            np.array([real_vol]), rv_ref["mean"], rv_ref["std"]
        )[0] if not np.isnan(real_vol) else np.nan
        row["z_real_vol_actual"] = round(float(z_real_vol), 5) if not np.isnan(z_real_vol) else np.nan

        if len(z_actual_arr) == 0:
            row["fes_actual"] = np.nan
        else:
            row["fes_actual"] = round(
                float(np.sum(z_actual_arr) + (z_real_vol if not np.isnan(z_real_vol) else 0)), 5
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _load_forecast(
    series: str, model: str, mode: str, forecast_dir: str
) -> Optional[pd.DataFrame]:
    path = (
        f"{forecast_dir}/{series}_growth_pct_forecasts_"
        f"{model.lower()}_{mode}.csv"
    )
    try:
        df = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
        return df
    except FileNotFoundError:
        log.warning(f"Forecast file not found: {path}")
        return None


def _training_stats(
    core_df: pd.DataFrame, train_start: str = TRAIN_START, train_end: str = TRAIN_END,
) -> dict:
    """
    Per-series training-period statistics for z-scoring. Defaults reproduce
    the original single-year window; a rolling walk-forward caller passes
    a different (train_start, train_end) per year.

    Returns
    -------
    {series: {"mean": float, "std": float,
              "unc_mean": float, "unc_std": float}}
    where unc_* are derived from the rolling 12-month std of the series
    (used as the reference for forecast PI half-width z-scores).
    """
    train = core_df[
        (core_df["date"] >= train_start) & (core_df["date"] <= train_end)
    ].set_index("date")

    stats: dict = {}
    for s in SERIES:
        col = f"{s}_growth"
        if col not in train.columns:
            log.warning(f"[{col}] not found in core dataset; using zeros")
            stats[s] = {"mean": 0.0, "std": 1.0, "unc_mean": 0.0, "unc_std": 1.0}
            continue

        series_vals = train[col].dropna()
        mean_s = float(series_vals.mean())
        std_s  = float(series_vals.std()) or 1.0

        # Reference for forecast uncertainty z-score:
        # rolling 12-month std of the series (historical intra-annual volatility)
        roll_std = series_vals.rolling(12, min_periods=6).std().dropna()
        unc_mean = float(roll_std.mean())
        unc_std  = float(roll_std.std()) or 1.0

        stats[s] = {
            "mean": mean_s, "std": std_s,
            "unc_mean": unc_mean, "unc_std": unc_std,
        }
        log.info(
            f"[{s}] train μ={mean_s:.4f}, σ={std_s:.4f} | "
            f"unc_ref μ={unc_mean:.4f}, σ={unc_std:.4f}"
        )

    z_train = pd.DataFrame(index=train.index)
    for s in SERIES:
        col = f"{s}_growth"
        if col in train.columns and s in stats:
            z_train[s] = _zscore_array(
                train[col].astype(float).to_numpy(),
                stats[s]["mean"],
                stats[s]["std"],
            )
    real_vol_ref = z_train.dropna(how="all").std(axis=1, ddof=0).dropna()
    if real_vol_ref.empty:
        stats["_real_vol"] = {"mean": 0.0, "std": 1.0}
    else:
        rv_mean = float(real_vol_ref.mean())
        rv_std = float(real_vol_ref.std()) or 1.0
        stats["_real_vol"] = {"mean": rv_mean, "std": rv_std}
        log.info(f"[actual real-vol] train μ={rv_mean:.4f}, σ={rv_std:.4f}")
    return stats


def _zscore_array(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (x - mean) / (std if std > 1e-10 else 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# Three actual FES benchmarks (options A / B / C)
# ══════════════════════════════════════════════════════════════════════════════

def _compute_actual_fes_variants(
    monthly_df: pd.DataFrame,
    stats: dict,
    core_df: pd.DataFrame,
    train_start: str = TRAIN_START,
    train_end: str = TRAIN_END,
) -> pd.DataFrame:
    """
    Compute three alternative volatility proxies for the uncertainty term in
    the actual FES (replacing PI half-width which is unavailable for actuals).

    Option A — rolling 3-month std of mean(z_gas, z_elec, z_carbon)
    Option B — cross-component std at each month t  (reproduces fes_actual)
    Option C — absolute shock: |mean_z_t − hist_mean_z| / hist_std_z

    Each option produces columns:
      rv_actual_{A/B/C}, z_rv_actual_{A/B/C}, fes_actual_{A/B/C}
    """
    df = monthly_df.copy()

    z_cols   = [f"z_{s}_actual" for s in SERIES]
    z_matrix = np.column_stack([df[c].values for c in z_cols]).astype(float)  # (12, 3)
    mean_z   = np.nanmean(z_matrix, axis=1)   # (12,) mean z per month

    # Option A: 3-month rolling std of mean_z
    rv_A = (
        pd.Series(mean_z)
        .rolling(window=3, min_periods=1)
        .std(ddof=1)
        .fillna(0.0)
        .values
    )

    # Option B: cross-component std (matches existing fes_actual logic)
    rv_B = np.array([
        float(np.nanstd(z_matrix[t], ddof=0))
        if not np.all(np.isnan(z_matrix[t])) else np.nan
        for t in range(len(df))
    ])

    # Option C: absolute shock vs training distribution
    train = core_df[
        (core_df["date"] >= train_start) & (core_df["date"] <= train_end)
    ].set_index("date")
    z_train_parts = []
    for s in SERIES:
        col = f"{s}_growth"
        if col in train.columns:
            v    = train[col].dropna().values.astype(float)
            mu   = stats[s]["mean"]
            sig  = max(stats[s]["std"], 1e-10)
            z_train_parts.append((v - mu) / sig)
    if z_train_parts:
        all_zt   = np.concatenate(z_train_parts)
        hist_mu  = float(np.nanmean(all_zt))
        hist_sig = float(np.nanstd(all_zt)) or 1.0
    else:
        hist_mu, hist_sig = 0.0, 1.0
    rv_C = np.abs(mean_z - hist_mu) / hist_sig

    rv_ref      = stats.get("_real_vol", {"mean": 0.0, "std": 1.0})
    rv_ref_mean = rv_ref["mean"]
    rv_ref_std  = max(rv_ref["std"], 1e-10)

    # np.nansum of an all-NaN row silently returns 0.0 -- mask those months
    # (no realised data at all, e.g. beyond the raw data's real-world
    # coverage) back to NaN so they don't fabricate a false fes_actual_A/B/C
    # data point, matching the fes_actual fix above.
    all_nan_month = np.all(np.isnan(z_matrix), axis=1)
    z_sum = np.nansum(z_matrix, axis=1)   # sum of actual component z-scores
    z_sum = np.where(all_nan_month, np.nan, z_sum)

    for label, rv in [("A", rv_A), ("B", rv_B), ("C", rv_C)]:
        rv_clean = np.where(np.isnan(rv), 0.0, rv)
        z_rv     = _zscore_array(rv_clean, rv_ref_mean, rv_ref_std)
        fes_opt  = np.where(all_nan_month, np.nan, z_sum + z_rv)
        df[f"rv_actual_{label}"]   = np.round(rv, 5)
        df[f"z_rv_actual_{label}"] = np.round(z_rv, 5)
        df[f"fes_actual_{label}"]  = np.round(fes_opt, 5)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# Comparison metrics: forecasted FES variants vs actual FES benchmarks
# ══════════════════════════════════════════════════════════════════════════════

def _compare_fes_variants(monthly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute MAE, RMSE, Bias, MaxDev, Pearson r, Spearman r, R², Theil U
    for every (forecasted FES variant) × (actual FES benchmark) pair.
    """
    try:
        from scipy import stats as sp
    except ImportError:
        sp = None

    forecasted = {
        "Equal_Core":     "fes_core",
        "Equal_Macro":    "fes_macro",
        "Equal_Selected": "fes_selected",
        "Equal_Weighted": "fes_weighted",
    }
    actuals = {
        "Actual_RollingVol": "fes_actual_A",
        "Actual_CrossComp":  "fes_actual_B",
        "Actual_AbsShock":   "fes_actual_C",
    }

    rows = []
    for f_lbl, f_col in forecasted.items():
        if f_col not in monthly_df.columns:
            continue
        fv = monthly_df[f_col].values.astype(float)

        for a_lbl, a_col in actuals.items():
            if a_col not in monthly_df.columns:
                continue
            av   = monthly_df[a_col].values.astype(float)
            mask = ~(np.isnan(fv) | np.isnan(av))
            if mask.sum() < 3:
                continue
            f_m, a_m = fv[mask], av[mask]

            mae  = float(np.mean(np.abs(f_m - a_m)))
            rmse = float(np.sqrt(np.mean((f_m - a_m) ** 2)))
            bias = float(np.mean(f_m - a_m))
            mdev = float(np.max(np.abs(f_m - a_m)))

            if sp is not None:
                try:
                    pr, pp = sp.pearsonr(f_m, a_m)
                    sr, _  = sp.spearmanr(f_m, a_m)
                except Exception:
                    pr = pp = sr = np.nan
            else:
                pr = pp = sr = np.nan

            ss_res = float(np.sum((f_m - a_m) ** 2))
            ss_tot = float(np.sum((a_m - a_m.mean()) ** 2))
            r2     = 1.0 - ss_res / ss_tot if ss_tot > 1e-10 else np.nan

            naive      = np.concatenate([[a_m[0]], a_m[:-1]])
            rmse_naive = float(np.sqrt(np.mean((naive - a_m) ** 2)))
            theil_u    = rmse / rmse_naive if rmse_naive > 1e-10 else np.nan

            rows.append({
                "FES_variant":      f_lbl,
                "Actual_benchmark": a_lbl,
                "N":                int(mask.sum()),
                "MAE":              round(mae,  5),
                "RMSE":             round(rmse, 5),
                "Bias":             round(bias, 5),
                "MaxDev":           round(mdev, 5),
                "Pearson_r":        round(pr,   5) if not np.isnan(pr)      else np.nan,
                "Pearson_p":        round(pp,   5) if not np.isnan(pp)      else np.nan,
                "Spearman_r":       round(sr,   5) if not np.isnan(sr)      else np.nan,
                "R2":               round(r2,   5) if not np.isnan(r2)      else np.nan,
                "Theil_U":          round(theil_u, 5) if not np.isnan(theil_u) else np.nan,
            })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# Main computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_fes(
    ranked_df: pd.DataFrame,
    core_csv: str = "data/processed/core_energy_carbon.csv",
    forecast_dir: str = "outputs_v2/forecasts",
    out_dir: str = "outputs_v2/fes",
    figures_dir: str = "outputs_v2/figures",
    train_start: str = TRAIN_START,
    train_end: str = TRAIN_END,
    forecast_dates: pd.DatetimeIndex = FORECAST_DATES,
    skip_diagnostics: bool = False,
) -> pd.DataFrame:
    """
    Compute FES_core, FES_macro, FES_actual for each month of the forecast
    target window.

    Parameters
    ----------
    ranked_df       : model_evaluation output (used to pick best model per series/mode)
    core_csv        : path to Dataset A (realised values)
    forecast_dir    : directory containing per-model-mode forecast CSVs
    out_dir         : where to save FES CSVs
    figures_dir     : where to save PNG figures
    train_start/train_end : z-scoring reference window (defaults reproduce
                      the original 2005-2016 single-year window)
    forecast_dates  : the 12-month target window (default: 2017); a rolling
                      walk-forward caller passes a different window per year
    skip_diagnostics : if True, skip all figure generation and the
                      summary/component tables -- still computes and saves
                      the monthly z-score DataFrame AND the cross-baseline
                      comparison metrics (cheap, no plotting; needed by the
                      rolling loop's FES-variant selection). Used by the
                      rolling walk-forward loop (run_rolling), where
                      generating the full ~15-figure diagnostic set once
                      per year x ~15 years would be excessive output volume.

    Returns
    -------
    Monthly FES DataFrame (12 rows, all components)
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(figures_dir).mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    core_df = pd.read_csv(core_csv, parse_dates=["date"])
    best    = _find_best_models(ranked_df)
    selected_mode = _select_best_mode_per_series(ranked_df, best)
    series_weights = _compute_series_weights(ranked_df, selected_mode)
    stats   = _training_stats(core_df, train_start, train_end)

    # ── Build per-series forecast arrays ─────────────────────────────────────
    # Storage: {(series, mode): DataFrame}
    forecasts: dict = {}
    for mode in MODES:
        for series in SERIES:
            model = best.get((series, mode))
            if model is None:
                log.warning(f"No best model found for ({series}, {mode})")
                continue
            df = _load_forecast(series, model, mode, forecast_dir)
            if df is not None:
                forecasts[(series, mode)] = df

    # ── Extract actual target-window values ───────────────────────────────────
    actual_target: dict = {}
    target_start, target_end = forecast_dates.min(), forecast_dates.max()
    for series in SERIES:
        col = f"{series}_growth"
        sub = core_df[
            (core_df["date"] >= target_start) & (core_df["date"] <= target_end)
        ].set_index("date")
        if col in sub.columns:
            actual_target[series] = sub[col].reindex(forecast_dates).values
        else:
            # Try to get from forecast CSV actual column
            for mode in MODES:
                key = (series, mode)
                if key in forecasts and "actual" in forecasts[key].columns:
                    actual_target[series] = (
                        forecasts[key].set_index("date")["actual"]
                        .reindex(forecast_dates).values
                    )
                    break
            else:
                log.warning(f"No actual target-window data for {series}; using NaN")
                actual_target[series] = np.full(12, np.nan)

    # ── Compute z-scores for each component and variant ───────────────────────
    monthly_rows = []

    for i, date in enumerate(forecast_dates):
        row: dict = {"date": date}

        # ── FES_core and FES_macro ────────────────────────────────────────────
        for mode in MODES:
            z_vals   = []
            z_unc    = []

            for series in SERIES:
                st  = stats[series]
                key = (series, mode)
                if key not in forecasts:
                    row[f"z_{series}_{mode}"]   = np.nan
                    row[f"unc_{series}_{mode}"]  = np.nan
                    row[f"z_unc_{series}_{mode}"] = np.nan
                    z_vals.append(np.nan)
                    z_unc.append(np.nan)
                    continue

                fc_df = forecasts[key].set_index("date")
                fc    = float(fc_df.loc[date, "forecast"])     if date in fc_df.index else np.nan
                lb    = float(fc_df.loc[date, "lower_bound"])  if date in fc_df.index else np.nan
                ub    = float(fc_df.loc[date, "upper_bound"])  if date in fc_df.index else np.nan

                z_series = _zscore_array(
                    np.array([fc]), st["mean"], st["std"]
                )[0]

                # Forecast uncertainty = PI half-width
                unc_hw = (ub - lb) / 2 if not (np.isnan(ub) or np.isnan(lb)) else np.nan
                z_unc_val = _zscore_array(
                    np.array([unc_hw]), st["unc_mean"], st["unc_std"]
                )[0] if not np.isnan(unc_hw) else np.nan

                row[f"forecast_{series}_{mode}"]   = round(fc, 5)
                row[f"z_{series}_{mode}"]          = round(z_series, 5)
                row[f"pi_halfwidth_{series}_{mode}"] = round(unc_hw, 5) if not np.isnan(unc_hw) else np.nan
                row[f"z_unc_{series}_{mode}"]      = round(z_unc_val, 5) if not np.isnan(z_unc_val) else np.nan

                z_vals.append(z_series)
                z_unc.append(z_unc_val)

            # Aggregate uncertainty z-score (mean across series)
            z_unc_agg = float(np.nanmean(z_unc))
            row[f"z_unc_{mode}"] = round(z_unc_agg, 5)

            fes_val = float(np.nansum(z_vals) + z_unc_agg)
            row[f"fes_{mode}"] = round(fes_val, 5)

        # ── FES_selected: per-series best-of-core/macro composite ────────────
        # Reuses the z_{series}_{mode}/z_unc_{series}_{mode} components just
        # computed above -- for each series, take whichever mode
        # _select_best_mode_per_series found better, instead of committing
        # every series to one mode the way FES_core/FES_macro each do.
        z_sel_vals, z_sel_unc = [], []
        for series in SERIES:
            m = selected_mode.get(series, "core")
            row[f"selected_mode_{series}"] = m
            z_sel = row.get(f"z_{series}_{m}", np.nan)
            unc_sel = row.get(f"z_unc_{series}_{m}", np.nan)
            row[f"z_{series}_selected"] = z_sel
            z_sel_vals.append(z_sel)
            z_sel_unc.append(unc_sel)
        z_sel_unc_agg = float(np.nanmean(z_sel_unc))
        row["z_unc_selected"] = round(z_sel_unc_agg, 5)
        row["fes_selected"] = round(float(np.nansum(z_sel_vals) + z_sel_unc_agg), 5)

        # ── FES_weighted: inverse-validation-RMSE weighted composite ─────────
        # ADDITIVE alongside FES_selected -- reuses the same per-series
        # best-mode z-scores, just scaled by each series' data-driven
        # weight (see _compute_series_weights) instead of an implicit 1.0.
        # Uncertainty term reused unweighted from FES_selected (weighting
        # it separately isn't well-motivated and keeps this bounded).
        z_wt_vals = []
        for series in SERIES:
            z_wt = series_weights.get(series, 1.0) * row.get(f"z_{series}_selected", np.nan)
            row[f"z_{series}_weighted"] = round(z_wt, 5) if pd.notna(z_wt) else np.nan
            z_wt_vals.append(z_wt)
        row["z_unc_weighted"] = round(z_sel_unc_agg, 5)
        row["fes_weighted"] = round(float(np.nansum(z_wt_vals) + z_sel_unc_agg), 5)

        # ── FES_actual ────────────────────────────────────────────────────────
        z_actual = []
        for series in SERIES:
            st = stats[series]
            act_val = actual_target[series][i] if i < len(actual_target[series]) else np.nan
            z_a = _zscore_array(np.array([act_val]), st["mean"], st["std"])[0] \
                  if not np.isnan(act_val) else np.nan
            row[f"actual_{series}"]   = round(float(act_val), 5) if not np.isnan(act_val) else np.nan
            row[f"z_{series}_actual"] = round(float(z_a), 5)     if not np.isnan(z_a)     else np.nan
            z_actual.append(z_a)

        # Realized volatility = cross-sectional std of z-scored actuals
        z_actual_arr = np.array([v for v in z_actual if not np.isnan(v)])
        real_vol = float(np.std(z_actual_arr)) if len(z_actual_arr) > 1 else np.nan

        row["real_vol_actual"] = round(real_vol, 5) if not np.isnan(real_vol) else np.nan
        rv_ref = stats.get("_real_vol", {"mean": 0.0, "std": 1.0})
        z_real_vol = _zscore_array(
            np.array([real_vol]), rv_ref["mean"], rv_ref["std"]
        )[0] if not np.isnan(real_vol) else np.nan
        row["z_real_vol_actual"] = (
            round(float(z_real_vol), 5) if not np.isnan(z_real_vol) else np.nan
        )

        if len(z_actual_arr) == 0:
            # No realised data at all for this month (e.g. still ahead of the
            # raw data's real-world coverage) -- np.nansum([nan, nan, nan])
            # silently returns 0.0, which would otherwise fabricate a false
            # "realised FES = 0" data point instead of leaving it unrealised.
            row["fes_actual"] = np.nan
        else:
            fes_actual = float(np.sum(z_actual_arr) + (
                z_real_vol if not np.isnan(z_real_vol) else 0
            ))
            row["fes_actual"] = round(fes_actual, 5)

        monthly_rows.append(row)

    monthly_df = pd.DataFrame(monthly_rows)

    # ── Three actual FES variants ─────────────────────────────────────────────
    monthly_df = _compute_actual_fes_variants(monthly_df, stats, core_df, train_start, train_end)

    # ── Save monthly CSV ──────────────────────────────────────────────────────
    target_year = int(forecast_dates[0].year)
    monthly_path = f"{out_dir}/fes_monthly_{target_year}.csv"
    monthly_df.to_csv(monthly_path, index=False)
    log.info(f"Monthly FES saved → {monthly_path}")

    # ── FES_actual for the training-cutoff year (year x) ──────────────────────
    # Baseline realised level for the year immediately before the forecast
    # target window, computed straight from realised data (not a forecast) --
    # sits alongside FES_selected's year x+1 forecast so downstream
    # consumers (src.ukhls_preprocessing.attach_fes_delta) have both "what
    # actually happened last year" and "what we forecast for next year"
    # available together.
    # Held out from `stats`: train_end == refit_end always lands on prior_year's
    # December (see forecast_pipeline._compute_default_window/run_rolling), so
    # `stats` itself includes prior_year in its mean/std -- z-scoring
    # prior_year's own realised values against it would be in-sample, not the
    # "held-out baseline" the caller relies on. Rebuild the reference window
    # ending one year earlier so prior_year is genuinely excluded.
    prior_year        = target_year - 1
    prior_stats_end   = f"{prior_year - 1}-12-01"
    prior_stats = (
        _training_stats(core_df, train_start, prior_stats_end)
        if pd.Timestamp(prior_stats_end) >= pd.Timestamp(train_start)
        else stats
    )
    prior_dates  = pd.DatetimeIndex([d - pd.DateOffset(years=1) for d in forecast_dates])
    prior_actual_df = _compute_actual_fes_for_window(core_df, prior_stats, prior_dates)
    # NOT "fes_monthly_*" -- that glob is reserved for compute_fes's own
    # target-year output and is exactly what paths.latest_fes_monthly_file()
    # scans (alphabetically, so a "fes_monthly_prior_actual_2024.csv" would
    # incorrectly outrank "fes_monthly_2025.csv" as the "latest" file).
    prior_actual_path = f"{out_dir}/fes_prior_actual_{prior_year}.csv"
    prior_actual_df.to_csv(prior_actual_path, index=False)
    fes_actual_prior_mean = float(prior_actual_df["fes_actual"].mean())
    log.info(
        "FES_actual, prior year %d (baseline for FES_selected's %d forecast): mean=%.5f -> %s",
        prior_year, target_year, fes_actual_prior_mean, prior_actual_path,
    )

    # ── Comparison metrics (cheap, no plotting -- kept even under
    # skip_diagnostics since the rolling walk-forward's FES-variant
    # selection needs these numbers per year) ─────────────────────────────────
    comparison_df = _compare_fes_variants(monthly_df)
    comp_path = f"{out_dir}/fes_comparison_metrics.csv"
    comparison_df.to_csv(comp_path, index=False)
    log.info("FES comparison metrics saved → %s  (%d rows)", comp_path, len(comparison_df))
    if not comparison_df.empty and "RMSE" in comparison_df.columns:
        best_rmse = comparison_df.groupby("FES_variant")["RMSE"].min()
        log.info("Best RMSE per variant:\n%s", best_rmse.to_string())

    if skip_diagnostics:
        return monthly_df

    # ── Build summary / component table ──────────────────────────────────────
    extra_summary_rows = [
        {"variant": "actual_prior_year", "component": "year",       "z_mean": prior_year},
        {"variant": "actual_prior_year", "component": "FES_TOTAL",  "z_mean": round(fes_actual_prior_mean, 5)},
    ]
    _save_summary(monthly_df, out_dir, target_year, extra_rows=extra_summary_rows)
    _save_component_table(monthly_df, best, out_dir, target_year, selected_mode=selected_mode)

    # ── Generate figures ──────────────────────────────────────────────────────
    _plot_fes_comparison(monthly_df, figures_dir, target_year)
    _plot_fes_components(monthly_df, figures_dir, target_year)
    _plot_forecasts_vs_actual(
        forecasts, actual_target, core_df, best, figures_dir,
        train_start=train_start, train_end=train_end, forecast_dates=forecast_dates,
    )

    try:
        _plot_metrics_heatmap(comparison_df, figures_dir)
    except Exception as e:
        log.warning(f"Metrics heatmap failed: {e}")

    # Polar model ranking charts
    try:
        from src.plotting_utils import plot_all_polar_charts
        plot_all_polar_charts(ranked_df, figures_dir, best=best)
    except Exception as e:
        log.warning(f"Polar charts failed: {e}")

    # Prediction interval comparison figure
    try:
        from src.plotting_utils import plot_prediction_intervals
        plot_prediction_intervals(
            forecast_dir=forecast_dir,
            out_path=f"{figures_dir}/prediction_intervals_{target_year}.png",
        )
    except Exception as e:
        log.warning(f"PI figure failed: {e}")

    # ── 6 static target-year model-comparison figures (3 series × 2 modes) ───
    try:
        from src.plotting_utils import plot_all_model_comparisons
        plot_all_model_comparisons(forecast_dir, figures_dir)
        log.info("Static target-year comparison figures complete")
    except Exception as e:
        log.warning(f"Static target-year comparison figures failed: {e}")
    # ── 6 interactive HTML timeline figures (3 series × 2 modes) ─────────────
    try:
        from src.plotting_utils import plot_all_interactive_forecasts
        plot_all_interactive_forecasts(
            core_df, forecast_dir, figures_dir,
            train_start=train_start, train_end=train_end,
        )
        log.info("Interactive timeline figures complete")
    except Exception as e:
        log.warning(f"Interactive timeline figures failed: {e}")

    return monthly_df


# ══════════════════════════════════════════════════════════════════════════════
# Summary tables
# ══════════════════════════════════════════════════════════════════════════════

def _save_summary(
    monthly_df: pd.DataFrame, out_dir: str, target_year: int,
    extra_rows: list[dict] | None = None,
) -> None:
    rows = []
    # Equal-weight, per-series-selected, data-driven-weighted, and actual
    # variants (have full component breakdown)
    for mode in ["core", "macro", "selected", "weighted", "actual"]:
        fes_col = f"fes_{mode}"
        if fes_col not in monthly_df.columns:
            continue
        fes_vals = monthly_df[fes_col].values

        for series in SERIES:
            z_col = f"z_{series}_{mode}"
            if z_col in monthly_df.columns:
                z_mean = float(monthly_df[z_col].mean())
            else:
                z_mean = np.nan
            rows.append({
                "variant":   mode,
                "component": f"{series}_growth_pct",
                "z_mean":    round(z_mean, 5),
            })

        unc_col = f"z_unc_{mode}" if mode in ("core", "macro", "selected", "weighted") else "z_real_vol_actual"
        if unc_col in monthly_df.columns:
            rows.append({
                "variant":   mode,
                "component": "uncertainty" if mode != "actual" else "real_vol",
                "z_mean":    round(float(monthly_df[unc_col].mean()), 5),
            })

        rows.append({
            "variant":   mode,
            "component": "FES_TOTAL",
            "z_mean":    round(float(np.nanmean(fes_vals)), 5),
        })

    # Extra rows the caller wants folded in (e.g. the prior-year FES_actual
    # baseline, which has no per-series/uncertainty breakdown of its own
    # since it isn't derived from monthly_df's target-year columns).
    if extra_rows:
        rows.extend(extra_rows)

    summary_df = pd.DataFrame(rows)
    path = f"{out_dir}/fes_summary_{target_year}.csv"
    summary_df.to_csv(path, index=False)
    log.info(f"FES summary saved → {path}")


def _save_component_table(
    monthly_df: pd.DataFrame,
    best: dict,
    out_dir: str,
    target_year: int,
    selected_mode: dict | None = None,
) -> None:
    """
    Create the cross-baseline comparison table:

    component          | FES_core | FES_macro | FES_selected | FES_actual
    ─────────────────────────────────────────────────────────────────────
    gas_growth_pct     |  z_mean  |  z_mean   |  z_mean      |  z_mean
    electricity_growth |  z_mean  |  z_mean   |  z_mean      |  z_mean
    carbon_log_return      |  z_mean  |  z_mean   |  z_mean      |  z_mean
    uncertainty        |  z_mean  |  z_mean   |  z_mean      |  z_mean
    FES (annual mean)  |  mean    |  mean     |  mean        |  mean

    FES_selected picks, independently for each series, whichever of
    FES_core/FES_macro validated better (see _select_best_mode_per_series) --
    so unlike FES_core/FES_macro it isn't tied to one "best model" dict
    entry per mode; `selected_mode` (the {series: 'core'|'macro'} choice) is
    reported in the "best model" row instead.
    """
    modes = ["core", "macro", "selected", "weighted", "actual"]
    selected_mode = selected_mode or {}
    rows = []
    for series in SERIES:
        row = {"component": f"{series}_growth_pct"}
        for mode in modes:
            col = f"z_{series}_{mode}"
            row[f"FES_{mode}"] = (
                round(float(monthly_df[col].mean()), 5) if col in monthly_df.columns else np.nan
            )
        rows.append(row)

    # Uncertainty row
    unc_row = {"component": "Uncertainty / RealVol"}
    for mode in ["core", "macro", "selected", "weighted"]:
        col = f"z_unc_{mode}"
        unc_row[f"FES_{mode}"] = (
            round(float(monthly_df[col].mean()), 5) if col in monthly_df.columns else np.nan
        )
    col_a = "z_real_vol_actual"
    unc_row["FES_actual"] = (
        round(float(monthly_df[col_a].mean()), 5) if col_a in monthly_df.columns else np.nan
    )
    rows.append(unc_row)

    # FES total row
    total_row = {"component": "FES_total (annual mean)"}
    for mode in modes:
        col = f"fes_{mode}"
        total_row[f"FES_{mode}"] = (
            round(float(monthly_df[col].mean()), 5) if col in monthly_df.columns else np.nan
        )
    rows.append(total_row)

    # Best model / mode info
    model_row = {"component": "--- best model ---"}
    for mode in ["core", "macro"]:
        mdl_list = [f"{s}:{best.get((s,mode),'?')}" for s in SERIES]
        model_row[f"FES_{mode}"] = " | ".join(mdl_list)
    model_row["FES_selected"] = " | ".join(f"{s}:{selected_mode.get(s,'?')}" for s in SERIES)
    model_row["FES_weighted"] = "inverse-RMSE weighted (see fes_variant_selection.csv)"
    model_row["FES_actual"] = "realised values"
    rows.append(model_row)

    comp_df = pd.DataFrame(rows)
    path = f"{out_dir}/fes_components_table.csv"
    comp_df.to_csv(path, index=False)
    log.info(f"FES components table saved → {path}")

    # Print to console
    print(f"\n── FES Component Table (z-score means, {target_year}) ──────────────────────────")
    print(comp_df.to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# Figures
# ══════════════════════════════════════════════════════════════════════════════

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_PALETTE = {
    "gas":         "#E67E22",
    "electricity": "#2980B9",
    "carbon":      "#27AE60",
    "core":        "#8E44AD",
    "macro":       "#E74C3C",
    "weighted":    "#F1C40F",
    "actual":      "#2C3E50",
    "grid":        "#EAECEE",
}
_DPI = 150

MONTHS_SHORT = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]


def _save_fig(fig: plt.Figure, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    fig.savefig(Path(path).with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    log.info(f"Figure saved → {path}")


def _plot_fes_comparison(monthly_df: pd.DataFrame, figures_dir: str, target_year: int) -> None:
    """Line chart: FES_core vs FES_macro vs FES_actual over 12 months of the target year."""
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    months = MONTHS_SHORT[:len(monthly_df)]

    for col, label, colour, ls in [
        ("fes_core",     "FES Core (core-only models)",      _PALETTE["core"],     "-"),
        ("fes_macro",    "FES Macro (core + exogenous)",     _PALETTE["macro"],    "--"),
        ("fes_weighted", "FES Weighted (inverse-RMSE)",      _PALETTE["weighted"], "-."),
        ("fes_actual",   f"FES Actual (realised {target_year} values)",_PALETTE["actual"], ":"),
    ]:
        if col in monthly_df.columns:
            ax.plot(months, monthly_df[col].values, color=colour,
                    linestyle=ls, linewidth=2.2, marker="o", markersize=5,
                    label=label)

    ax.axhline(0, color="#95A5A6", linewidth=0.9, linestyle="-")
    ax.set_title(f"UK Anticipatory Energy–Carbon Stress Index — {target_year} Monthly",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel(f"Month ({target_year})", fontsize=11)
    ax.set_ylabel("FES (sum of z-scores)", fontsize=11)
    ax.legend(framealpha=0.92, fontsize=10, loc="upper left")
    ax.grid(True, color=_PALETTE["grid"], linewidth=0.8)
    ax.tick_params(axis="both", labelsize=9)

    _save_fig(fig, f"{figures_dir}/fes_monthly_{target_year}.png")


def _plot_fes_components(monthly_df: pd.DataFrame, figures_dir: str, target_year: int) -> None:
    """
    Two-panel figure:
    Left  — grouped bar chart comparing z-score components across the three variants
    Right — FES total comparison bar
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor("white")

    # ── Left: component z-scores (annual mean) ────────────────────────────────
    ax = axes[0]
    ax.set_facecolor("white")

    components = [f"{s}_growth_pct" for s in SERIES] + ["uncertainty"]
    z_cols = {
        "FES_core":     [f"z_{s}_core" for s in SERIES] + ["z_unc_core"],
        "FES_macro":    [f"z_{s}_macro" for s in SERIES] + ["z_unc_macro"],
        "FES_weighted": [f"z_{s}_weighted" for s in SERIES] + ["z_unc_weighted"],
        "FES_actual":   [f"z_{s}_actual" for s in SERIES] + ["z_real_vol_actual"],
    }
    colours_bar = {
        "FES_core":     _PALETTE["core"],
        "FES_macro":    _PALETTE["macro"],
        "FES_weighted": _PALETTE["weighted"],
        "FES_actual":   _PALETTE["actual"],
    }

    x = np.arange(len(components))
    width = 0.19
    for k, (label, cols) in enumerate(z_cols.items()):
        vals = [
            float(monthly_df[c].mean()) if c in monthly_df.columns else 0.0
            for c in cols
        ]
        bars = ax.bar(x + (k - (len(z_cols) - 1) / 2) * width, vals, width,
                      label=label, color=colours_bar[label],
                      edgecolor="white", linewidth=0.6)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.03 if v >= 0 else -0.12),
                    f"{v:+.2f}", ha="center", fontsize=7, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(["Gas\nGrowth", "Elec\nGrowth", "Carbon\nGrowth", "Unc /\nRealVol"],
                       fontsize=9)
    ax.axhline(0, color="#95A5A6", linewidth=0.8)
    ax.set_title(f"FES Component Z-Scores ({target_year} annual mean)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Z-score", fontsize=10)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", color=_PALETTE["grid"], linewidth=0.8)
    # The value labels above/below each bar use a small FIXED offset, not a
    # fraction of that bar's own height -- for the most negative bars (e.g.
    # the Unc/RealVol group) the label lands beyond matplotlib's
    # auto-computed y-limit and gets clipped/overlaps the x-tick labels
    # below the axis. Pad the limits explicitly using the actual plotted
    # values (not just the default autoscale) so every label has room.
    all_vals = [
        float(monthly_df[c].mean()) if c in monthly_df.columns else 0.0
        for cols in z_cols.values() for c in cols
    ]
    y_lo, y_hi = min(0.0, *all_vals), max(0.0, *all_vals)
    pad = max(0.15 * (y_hi - y_lo), 0.15)
    ax.set_ylim(y_lo - pad, y_hi + pad)

    # ── Right: FES total bars ─────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("white")

    fes_labels  = ["FES Core", "FES Macro", "FES Weighted", "FES Actual"]
    fes_cols    = ["fes_core", "fes_macro", "fes_weighted", "fes_actual"]
    fes_colours = [_PALETTE["core"], _PALETTE["macro"], _PALETTE["weighted"], _PALETTE["actual"]]
    fes_vals    = [
        float(monthly_df[c].mean()) if c in monthly_df.columns else 0.0
        for c in fes_cols
    ]

    bars = ax2.bar(fes_labels, fes_vals, color=fes_colours,
                   edgecolor="white", linewidth=0.8, width=0.5)
    for bar, v in zip(bars, fes_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + (0.05 if v >= 0 else -0.15),
                 f"{v:+.3f}", ha="center", fontsize=11, fontweight="bold")

    ax2.axhline(0, color="#95A5A6", linewidth=0.8)
    ax2.set_title(f"Total FES — Annual Mean {target_year}", fontsize=12, fontweight="bold")
    ax2.set_ylabel("FES (sum of z-scores)", fontsize=10)
    ax2.grid(axis="y", color=_PALETTE["grid"], linewidth=0.8)
    # Same fixed-offset label-clipping issue as the left panel -- pad
    # explicitly so the most negative bar's label (e.g. FES Macro) doesn't
    # land on top of the x-tick labels below the axis.
    y_lo2, y_hi2 = min(0.0, *fes_vals), max(0.0, *fes_vals)
    pad2 = max(0.15 * (y_hi2 - y_lo2), 0.15)
    ax2.set_ylim(y_lo2 - pad2, y_hi2 + pad2)

    fig.suptitle(f"Anticipatory Energy–Carbon Stress Index — Component Analysis {target_year}",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save_fig(fig, f"{figures_dir}/fes_components_{target_year}.png")

def _plot_forecasts_vs_actual(
    forecasts: dict,
    actual_target: dict,
    core_df: pd.DataFrame,
    best: dict,
    figures_dir: str,
    train_start: str = TRAIN_START,
    train_end: str = TRAIN_END,
    forecast_dates: pd.DatetimeIndex = FORECAST_DATES,
) -> None:
    """
    One 3-panel figure showing all three series, core and macro forecasts,
    actual target-year values, plus the historical training-window baseline.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
    fig.patch.set_facecolor("white")

    months = MONTHS_SHORT
    target_year = int(forecast_dates[0].year)

    for ax, series in zip(axes, SERIES):
        ax.set_facecolor("white")
        col = f"{series}_growth"

        # Historical training window
        hist = core_df[
            (core_df["date"] >= train_start) & (core_df["date"] <= train_end)
        ].set_index("date")[col]
        if not hist.empty:
            ax.plot(hist.index, hist.values, color=_PALETTE[series],
                    linewidth=1.5, alpha=0.6, label=f"Historical {train_start[:4]}–{train_end[:4]}")

        x = np.arange(12)
        fc_dates = forecast_dates

        for mode, ls, col_mode in [("core", "-", _PALETTE["core"]),
                                    ("macro", "--", _PALETTE["macro"])]:
            key = (series, mode)
            if key in forecasts:
                df_fc = forecasts[key].set_index("date")
                fc = df_fc["forecast"].reindex(fc_dates).values
                lb = df_fc["lower_bound"].reindex(fc_dates).values
                ub = df_fc["upper_bound"].reindex(fc_dates).values
                model = best.get(key, "?")
                ax.plot(fc_dates, fc, color=col_mode, linestyle=ls,
                        linewidth=2, marker="o", markersize=4,
                        label=f"Forecast {mode} ({model})")
                ax.fill_between(fc_dates, lb, ub, alpha=0.12, color=col_mode)

        # Actual target-year values
        act = actual_target.get(series, np.full(12, np.nan))
        if not np.all(np.isnan(act)):
            ax.plot(fc_dates, act, color="#2C3E50", linestyle="none",
                    marker="o", markersize=5, zorder=5,
                    label=f"Actual {target_year}")

        ax.axvline(fc_dates.min(), color="#BDC3C7",
                   linewidth=1.0, linestyle=":")
        ax.set_title(f"{series.capitalize()} Growth (% YoY)",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, color=_PALETTE["grid"], linewidth=0.7)
        ax.axhline(0, color="#BDC3C7", linewidth=0.7)

    fig.suptitle(
        f"UK Energy–Carbon Forecast vs Actual {target_year}  "
        "(core-only | macro-augmented | realised values)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    _save_fig(fig, f"{figures_dir}/forecast_vs_actual_all_series.png")

    # Also individual series plots
    for series in SERIES:
        col = f"{series}_growth"
        hist = core_df[
            (core_df["date"] >= train_start) & (core_df["date"] <= train_end)
        ].set_index("date")[col]
        act  = actual_target.get(series, np.full(12, np.nan))

        fig2, ax2 = plt.subplots(figsize=(12, 4))
        fig2.patch.set_facecolor("white")
        ax2.set_facecolor("white")

        if not hist.empty:
            ax2.plot(hist.index, hist.values, color=_PALETTE[series],
                     linewidth=1.8, label=f"Historical {train_start[:4]}–{train_end[:4]}")

        for mode, ls, col_mode in [("core", "-", _PALETTE["core"]),
                                    ("macro", "--", _PALETTE["macro"])]:
            key = (series, mode)
            if key in forecasts:
                df_fc = forecasts[key].set_index("date")
                fc = df_fc["forecast"].reindex(forecast_dates).values
                lb = df_fc["lower_bound"].reindex(forecast_dates).values
                ub = df_fc["upper_bound"].reindex(forecast_dates).values
                model = best.get(key, "?")
                ax2.plot(forecast_dates, fc, color=col_mode, linestyle=ls,
                         linewidth=2, marker="o", markersize=5,
                         label=f"Forecast {mode} ({model})")
                ax2.fill_between(forecast_dates, lb, ub, alpha=0.15, color=col_mode)

        if not np.all(np.isnan(act)):
            ax2.plot(forecast_dates, act, color="#2C3E50", linestyle="none",
                     marker="o", markersize=5, zorder=5, label=f"Actual {target_year}")

        ax2.axvline(forecast_dates.min(), color="#BDC3C7",
                    linewidth=1.0, linestyle=":")
        ax2.axhline(0, color="#BDC3C7", linewidth=0.7)
        ax2.set_title(
            f"UK {series.capitalize()} Growth (% YoY) — Core vs Macro Forecast vs Actual",
            fontsize=13, fontweight="bold", pad=10,
        )
        ax2.set_ylabel("YoY Growth (%)", fontsize=11)
        ax2.legend(fontsize=9, loc="best", framealpha=0.9)
        ax2.grid(True, color=_PALETTE["grid"], linewidth=0.8)
        ax2.tick_params(axis="both", labelsize=9)
        _save_fig(fig2, f"{figures_dir}/forecast_vs_actual_{series}.png")


# ══════════════════════════════════════════════════════════════════════════════
# Robustness figures
# ══════════════════════════════════════════════════════════════════════════════

def _plot_metrics_heatmap(comparison_df: pd.DataFrame, figures_dir: str) -> None:
    """Heatmaps of RMSE and Pearson r for each forecasted FES × actual benchmark."""
    if comparison_df.empty:
        return

    for metric, cmap, invert, cbar_label in [
        ("RMSE",      "YlOrRd",  True,  "RMSE (lower = better)"),
        ("Pearson_r", "RdYlGn",  False, "Pearson r (higher = better)"),
    ]:
        if metric not in comparison_df.columns:
            continue
        try:
            pivot = comparison_df.pivot(
                index="FES_variant", columns="Actual_benchmark", values=metric
            )
        except Exception:
            continue

        fig, ax = plt.subplots(figsize=(9, 6))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        vals = pivot.values.astype(float)
        vmin, vmax = np.nanmin(vals), np.nanmax(vals)
        if np.isnan(vmin) or np.isnan(vmax):
            plt.close(fig)
            continue

        im = ax.imshow(vals, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, label=cbar_label)

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_yticks(range(len(pivot.index)))
        ax.set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=9)
        ax.set_yticklabels(pivot.index, fontsize=9)

        mid = (vmin + vmax) / 2
        for r in range(len(pivot.index)):
            for c in range(len(pivot.columns)):
                v = vals[r, c]
                if not np.isnan(v):
                    txt_colour = "white" if (invert and v > mid) else "black"
                    ax.text(c, r, f"{v:.3f}", ha="center", va="center",
                            fontsize=9, fontweight="bold", color=txt_colour)

        ax.set_title(
            f"FES Robustness — {metric}: Forecasted vs Actual Benchmarks",
            fontsize=12, fontweight="bold", pad=12,
        )
        ax.set_xlabel("Actual FES Benchmark", fontsize=10)
        ax.set_ylabel("Forecasted FES Variant", fontsize=10)
        plt.tight_layout()
        _save_fig(fig, f"{figures_dir}/fes_metrics_{metric.lower()}_heatmap.png")
