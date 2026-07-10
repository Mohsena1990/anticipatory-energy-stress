"""
fes_calculator.py
─────────────────
Compute three Forecasted Energy-Carbon Stress (FES) indices for 2017.

Three analytical baselines
──────────────────────────
  FES_core   — built from core-only model forecasts (target series only)
  FES_macro  — built from macro-augmented model forecasts (core + exogenous)
  FES_actual — built from realised 2017 values (benchmark)

Formula
───────
  FES_t = z(GasGrowth_t) + z(ElecGrowth_t) + z(CarbonGrowth_t) + z(Uncertainty_t)

  For FES_actual the uncertainty term is replaced by realised cross-sectional
  volatility:
    RealVol_t = std(z_gas_t, z_elec_t, z_carbon_t)   [across the three series]

Z-score standardisation
───────────────────────
  All z-scores use the TRAINING period (2005-2017) mean and std so that
  FES_core, FES_macro, and FES_actual are directly comparable.

    z(X_t) = (X_t − μ_train) / σ_train

  Uncertainty z-scores use the rolling 12-month std of the training series as
  the reference distribution for forecast interval half-widths.

Outputs (CSV only)
──────────────────
  outputs/fes/fes_monthly_2017.csv       — 12 rows × all z-components + FES
  outputs/fes/fes_summary_2017.csv       — annual mean FES and components
  outputs/fes/fes_components_table.csv   — cross-baseline comparison table
  outputs/figures/fes_monthly_2017.png   — FES time-series (3 variants)
  outputs/figures/fes_components_2017.png — component breakdown bars
  outputs/figures/forecast_vs_actual_{series}.png  — per-series forecast plot

FES scenario-based signal simulation
─────────────────────────────────────
The monthly DataFrame returned here (`fes_core`, `fes_macro`, the VW/Bayesian
robustness variants, and the three `fes_actual_{A,B,C}` benchmarks) is also
handed to `src.fes_scenarios.run_fes_scenario_simulation`, which treats all
nine variants as named scenarios in a macro-level scenario-simulation layer
— not household-level predictors. See `src/fes_scenarios.py` for the
scenario tables and figures this produces.
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


def _training_stats(core_df: pd.DataFrame) -> dict:
    """
    Per-series training-period (2005-2017) statistics for z-scoring.

    Returns
    -------
    {series: {"mean": float, "std": float,
              "unc_mean": float, "unc_std": float}}
    where unc_* are derived from the rolling 12-month std of the series
    (used as the reference for forecast PI half-width z-scores).
    """
    train = core_df[
        (core_df["date"] >= TRAIN_START) & (core_df["date"] <= TRAIN_END)
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
# Robustness: Volatility-Weighted FES
# ══════════════════════════════════════════════════════════════════════════════

def _vw_weights(stats: dict) -> dict:
    """
    Inverse-volatility weights: w_j = (1/σ_j) / Σ_k (1/σ_k).
    Less volatile series receive a higher weight — prevents high-σ series
    (e.g. carbon) from dominating the equal-weight index.
    """
    inv = {s: 1.0 / max(stats[s]["std"], 1e-10) for s in SERIES}
    total = sum(inv.values())
    return {s: v / total for s, v in inv.items()}


# ══════════════════════════════════════════════════════════════════════════════
# Robustness: Bayesian FES (scalar Kalman filter state-space model)
# ══════════════════════════════════════════════════════════════════════════════

def _estimate_kalman_params(z_matrix: np.ndarray) -> tuple:
    """
    Estimate Kalman process and observation noise from a training z-score matrix.

    Parameters
    ----------
    z_matrix : (T, n_series) — training-period z-scored component values

    Returns
    -------
    (sigma_process, sigma_obs) — noise standard deviations
    """
    mean_z = np.nanmean(z_matrix, axis=1)          # common-signal proxy (T,)
    valid  = ~np.isnan(mean_z)
    if valid.sum() > 2:
        diffs    = np.diff(mean_z[valid])
        sigma_Q  = float(np.std(diffs, ddof=1))
    else:
        sigma_Q  = 0.3

    residuals = z_matrix - mean_z[:, None]
    flat_res  = residuals[~np.all(np.isnan(residuals), axis=1)].ravel()
    sigma_R   = float(np.nanstd(flat_res)) if flat_res.size > 0 else 0.5

    return max(sigma_Q, 0.05), max(sigma_R, 0.05)


def _kalman_filter_fes(
    observations: np.ndarray,
    sigma_process: float = 0.3,
    sigma_obs: float     = 0.5,
) -> tuple:
    """
    Scalar Kalman filter for a latent FES state observed through n noisy signals.

    State-space model
    -----------------
      FES_t   = FES_{t-1} + w_t,   w_t ~ N(0, σ_Q²)   [random walk]
      z_j,t   = FES_t + v_j,t,     v_j,t ~ N(0, σ_R²)  [per-series noise]

    With n_v valid observations at time t the effective obs noise is σ_R²/n_v
    (information pooling: averaging n_v independent signals).

    Parameters
    ----------
    observations  : (T, n_series) z-scored values; NaN = series missing
    sigma_process : process noise std (magnitude of month-to-month FES change)
    sigma_obs     : per-series observation noise std

    Returns
    -------
    (filtered_means, filtered_stds) — posterior FES estimates, each shape (T,)
    """
    T, _ = observations.shape
    Q    = sigma_process ** 2
    R    = sigma_obs ** 2

    x_t = 0.0   # diffuse prior: neutral FES level
    P_t = 1.0   # high initial uncertainty

    x_filt = np.zeros(T)
    P_filt = np.zeros(T)

    for t in range(T):
        y   = observations[t]
        n_v = int((~np.isnan(y)).sum())

        # Predict
        x_pred = x_t
        P_pred = P_t + Q

        # Update
        if n_v > 0:
            y_mean  = float(np.nanmean(y))
            eff_R   = R / n_v                          # pooled obs noise
            K       = P_pred / (P_pred + eff_R)        # Kalman gain ∈ (0,1)
            x_t     = x_pred + K * (y_mean - x_pred)  # posterior mean
            P_t     = max((1.0 - K) * P_pred, 1e-8)   # posterior variance
        else:
            x_t = x_pred
            P_t = P_pred

        x_filt[t] = x_t
        P_filt[t] = P_t

    return x_filt, np.sqrt(P_filt)


# ══════════════════════════════════════════════════════════════════════════════
# Three actual FES benchmarks (options A / B / C)
# ══════════════════════════════════════════════════════════════════════════════

def _compute_actual_fes_variants(
    monthly_df: pd.DataFrame,
    stats: dict,
    core_df: pd.DataFrame,
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
        (core_df["date"] >= TRAIN_START) & (core_df["date"] <= TRAIN_END)
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

    z_sum = np.nansum(z_matrix, axis=1)   # sum of actual component z-scores

    for label, rv in [("A", rv_A), ("B", rv_B), ("C", rv_C)]:
        rv_clean = np.where(np.isnan(rv), 0.0, rv)
        z_rv     = _zscore_array(rv_clean, rv_ref_mean, rv_ref_std)
        fes_opt  = z_sum + z_rv
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
        "Equal_Core":   "fes_core",
        "Equal_Macro":  "fes_macro",
        "VW_Core":      "fes_vw_core",
        "VW_Macro":     "fes_vw_macro",
        "Bayes_Core":   "fes_bayes_core",
        "Bayes_Macro":  "fes_bayes_macro",
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
    forecast_dir: str = "outputs/forecasts",
    out_dir: str = "outputs/fes",
    figures_dir: str = "outputs/figures",
) -> pd.DataFrame:
    """
    Compute FES_core, FES_macro, FES_actual for each month of 2017.

    Parameters
    ----------
    ranked_df    : model_evaluation output (used to pick best model per series/mode)
    core_csv     : path to Dataset A (2005-2017 actual values)
    forecast_dir : directory containing per-model-mode forecast CSVs
    out_dir      : where to save FES CSVs
    figures_dir  : where to save PNG figures

    Returns
    -------
    Monthly FES DataFrame (12 rows, all components)
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(figures_dir).mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    core_df = pd.read_csv(core_csv, parse_dates=["date"])
    best    = _find_best_models(ranked_df)
    stats   = _training_stats(core_df)

    # ── VW weights ────────────────────────────────────────────────────────────
    vw_weights_map = _vw_weights(stats)
    log.info("VW weights: %s", {s: round(w, 4) for s, w in vw_weights_map.items()})

    # ── Training z-matrix for Kalman parameter estimation ────────────────────
    _train_sub = core_df[
        (core_df["date"] >= TRAIN_START) & (core_df["date"] <= TRAIN_END)
    ].set_index("date")
    _z_parts = []
    for _s in SERIES:
        _col = f"{_s}_growth"
        if _col in _train_sub.columns:
            _z_parts.append(
                _zscore_array(
                    _train_sub[_col].ffill().fillna(0.0).values.astype(float),
                    stats[_s]["mean"], stats[_s]["std"],
                )
            )
    _lens = [len(a) for a in _z_parts]
    if _z_parts and len(set(_lens)) == 1:
        _z_train_mat = np.column_stack(_z_parts)
    else:
        _z_train_mat = np.zeros((max(_lens or [144]), max(len(_z_parts), 3)))
    _sigma_Q, _sigma_R = _estimate_kalman_params(_z_train_mat)
    log.info("Kalman params: σ_process=%.4f, σ_obs=%.4f", _sigma_Q, _sigma_R)

    # ── Storage for Bayesian z-matrices (filled in monthly loop) ─────────────
    _z_fc_matrices: dict = {m: np.full((12, len(SERIES)), np.nan) for m in MODES}

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

    # ── Extract actual 2017 values ────────────────────────────────────────────
    actual_2017: dict = {}
    for series in SERIES:
        col = f"{series}_growth"
        sub = core_df[
            (core_df["date"] >= "2017-01-01") & (core_df["date"] <= "2017-12-01")
        ].set_index("date")
        if col in sub.columns:
            actual_2017[series] = sub[col].reindex(FORECAST_DATES).values
        else:
            # Try to get from forecast CSV actual column
            for mode in MODES:
                key = (series, mode)
                if key in forecasts and "actual" in forecasts[key].columns:
                    actual_2017[series] = (
                        forecasts[key].set_index("date")["actual"]
                        .reindex(FORECAST_DATES).values
                    )
                    break
            else:
                log.warning(f"No actual 2017 data for {series}; using NaN")
                actual_2017[series] = np.full(12, np.nan)

    # ── Compute z-scores for each component and variant ───────────────────────
    monthly_rows = []

    for i, date in enumerate(FORECAST_DATES):
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

            # ── VW-FES ───────────────────────────────────────────────────────
            _wts  = np.array([vw_weights_map[s] for s in SERIES], dtype=float)
            _zv   = np.array(z_vals, dtype=float)
            _zu   = np.array(z_unc,  dtype=float)
            _ok_z = ~np.isnan(_zv)
            _ok_u = ~np.isnan(_zu)
            if _ok_z.sum() > 0:
                _w_z       = _wts[_ok_z] / _wts[_ok_z].sum()
                _z_vw_val  = float(np.dot(_w_z, _zv[_ok_z]))
            else:
                _z_vw_val  = np.nan
            if _ok_u.sum() > 0:
                _w_u       = _wts[_ok_u] / _wts[_ok_u].sum()
                _z_vw_unc  = float(np.dot(_w_u, _zu[_ok_u]))
            else:
                _z_vw_unc  = 0.0
            _fes_vw = _z_vw_val + _z_vw_unc if not np.isnan(_z_vw_val) else np.nan
            row[f"fes_vw_{mode}"] = (
                round(_fes_vw, 5) if not np.isnan(_fes_vw) else np.nan
            )

            # ── Store z-scores for Bayesian filter ───────────────────────────
            for _j in range(len(SERIES)):
                _z_fc_matrices[mode][i, _j] = (
                    float(_zv[_j]) if not np.isnan(_zv[_j]) else np.nan
                )

        # ── FES_actual ────────────────────────────────────────────────────────
        z_actual = []
        for series in SERIES:
            st = stats[series]
            act_val = actual_2017[series][i] if i < len(actual_2017[series]) else np.nan
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

        fes_actual = float(np.nansum(z_actual) + (
            z_real_vol if not np.isnan(z_real_vol) else 0
        ))
        row["fes_actual"] = round(fes_actual, 5)

        monthly_rows.append(row)

    monthly_df = pd.DataFrame(monthly_rows)

    # ── Bayesian FES (Kalman filter on forecasted z-score matrices) ───────────
    for _mode in MODES:
        _x_filt, _std_filt = _kalman_filter_fes(
            _z_fc_matrices[_mode], _sigma_Q, _sigma_R
        )
        monthly_df[f"fes_bayes_{_mode}"]     = np.round(_x_filt, 5)
        monthly_df[f"fes_bayes_{_mode}_std"] = np.round(_std_filt, 5)
        monthly_df[f"fes_bayes_{_mode}_lb"]  = np.round(_x_filt - 1.96 * _std_filt, 5)
        monthly_df[f"fes_bayes_{_mode}_ub"]  = np.round(_x_filt + 1.96 * _std_filt, 5)
        log.info(
            "[Bayesian FES %s] mean=%.4f, mean_CI_half=%.4f",
            _mode, float(_x_filt.mean()), float((1.96 * _std_filt).mean()),
        )

    # ── Three actual FES variants ─────────────────────────────────────────────
    monthly_df = _compute_actual_fes_variants(monthly_df, stats, core_df)

    # ── Save monthly CSV ──────────────────────────────────────────────────────
    monthly_path = f"{out_dir}/fes_monthly_2017.csv"
    monthly_df.to_csv(monthly_path, index=False)
    log.info(f"Monthly FES saved → {monthly_path}")

    # ── Comparison metrics ────────────────────────────────────────────────────
    comparison_df = _compare_fes_variants(monthly_df)
    comp_path = f"{out_dir}/fes_comparison_metrics.csv"
    comparison_df.to_csv(comp_path, index=False)
    log.info("FES comparison metrics saved → %s  (%d rows)", comp_path, len(comparison_df))
    if not comparison_df.empty and "RMSE" in comparison_df.columns:
        best_rmse = comparison_df.groupby("FES_variant")["RMSE"].min()
        log.info("Best RMSE per variant:\n%s", best_rmse.to_string())

    # ── Build summary / component table ──────────────────────────────────────
    _save_summary(monthly_df, out_dir)
    _save_component_table(monthly_df, best, out_dir)

    # ── Generate figures ──────────────────────────────────────────────────────
    _plot_fes_comparison(monthly_df, figures_dir)
    _plot_fes_components(monthly_df, figures_dir)
    _plot_forecasts_vs_actual(forecasts, actual_2017, core_df, best, figures_dir)

    # Robustness comparison plots
    try:
        _plot_fes_robustness(monthly_df, figures_dir)
    except Exception as e:
        log.warning(f"Robustness plot failed: {e}")
    try:
        _plot_bayesian_uncertainty(monthly_df, figures_dir)
    except Exception as e:
        log.warning(f"Bayesian uncertainty plot failed: {e}")
    try:
        _plot_metrics_heatmap(comparison_df, figures_dir)
    except Exception as e:
        log.warning(f"Metrics heatmap failed: {e}")

    # Polar model ranking charts
    try:
        from src.plotting_utils import plot_all_polar_charts
        plot_all_polar_charts(ranked_df, figures_dir)
    except Exception as e:
        log.warning(f"Polar charts failed: {e}")

    # Prediction interval comparison figure
    try:
        from src.plotting_utils import plot_prediction_intervals
        plot_prediction_intervals(
            forecast_dir=forecast_dir,
            out_path=f"{figures_dir}/prediction_intervals_2017.png",
        )
    except Exception as e:
        log.warning(f"PI figure failed: {e}")

    # ── 6 static 2017 model-comparison figures (3 series × 2 modes) ──────────
    try:
        from src.plotting_utils import plot_all_2017_comparisons
        plot_all_2017_comparisons(forecast_dir, figures_dir)
        log.info("Static 2017 comparison figures complete")
    except Exception as e:
        log.warning(f"Static 2017 comparison figures failed: {e}")
    # ── 6 interactive HTML timeline figures (3 series × 2 modes) ─────────────
    try:
        from src.plotting_utils import plot_all_interactive_forecasts
        plot_all_interactive_forecasts(core_df, forecast_dir, figures_dir)
        log.info("Interactive timeline figures complete")
    except Exception as e:
        log.warning(f"Interactive timeline figures failed: {e}")

    # ── FES scenario-based signal simulation (9 named scenarios) ──────────────
    # Macro-context robustness/interpretation layer only — never a household
    # feature (see src.route_utils.exclude_fes_columns).
    try:
        from src.fes_scenarios import run_fes_scenario_simulation
        run_fes_scenario_simulation(monthly_df, out_dir=out_dir, figures_dir=figures_dir)
        log.info("FES scenario-based signal simulation complete")
    except Exception as e:
        log.warning(f"FES scenario simulation failed: {e}")

    return monthly_df


# ══════════════════════════════════════════════════════════════════════════════
# Summary tables
# ══════════════════════════════════════════════════════════════════════════════

def _save_summary(monthly_df: pd.DataFrame, out_dir: str) -> None:
    rows = []
    # Equal-weight and actual variants (have full component breakdown)
    for mode in ["core", "macro", "actual"]:
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

        unc_col = f"z_unc_{mode}" if mode in ("core", "macro") else "z_real_vol_actual"
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

    # VW and Bayesian variants (FES total only — no per-component z columns)
    for variant_col, variant_label in [
        ("fes_vw_core",    "vw_core"),
        ("fes_vw_macro",   "vw_macro"),
        ("fes_bayes_core", "bayes_core"),
        ("fes_bayes_macro","bayes_macro"),
    ]:
        if variant_col in monthly_df.columns:
            rows.append({
                "variant":   variant_label,
                "component": "FES_TOTAL",
                "z_mean":    round(float(np.nanmean(monthly_df[variant_col].values)), 5),
            })

    summary_df = pd.DataFrame(rows)
    path = f"{out_dir}/fes_summary_2017.csv"
    summary_df.to_csv(path, index=False)
    log.info(f"FES summary saved → {path}")


def _save_component_table(
    monthly_df: pd.DataFrame,
    best: dict,
    out_dir: str,
) -> None:
    """
    Create the cross-baseline comparison table:

    component          | FES_core | FES_macro | FES_actual
    ───────────────────────────────────────────────────────
    gas_growth_pct     |  z_mean  |  z_mean   |  z_mean
    electricity_growth |  z_mean  |  z_mean   |  z_mean
    carbon_log_return      |  z_mean  |  z_mean   |  z_mean
    uncertainty        |  z_mean  |  z_mean   |  z_mean
    FES (annual mean)  |  mean    |  mean     |  mean
    """
    rows = []
    for series in SERIES:
        row = {"component": f"{series}_growth_pct"}
        for mode in ["core", "macro", "actual"]:
            col = f"z_{series}_{mode}"
            row[f"FES_{mode}"] = (
                round(float(monthly_df[col].mean()), 5) if col in monthly_df.columns else np.nan
            )
        rows.append(row)

    # Uncertainty row
    unc_row = {"component": "Uncertainty / RealVol"}
    for mode in ["core", "macro"]:
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
    for mode in ["core", "macro", "actual"]:
        col = f"fes_{mode}"
        total_row[f"FES_{mode}"] = (
            round(float(monthly_df[col].mean()), 5) if col in monthly_df.columns else np.nan
        )
    rows.append(total_row)

    # Best model info
    model_row = {"component": "--- best model ---"}
    for mode in ["core", "macro"]:
        mdl_list = [f"{s}:{best.get((s,mode),'?')}" for s in SERIES]
        model_row[f"FES_{mode}"] = " | ".join(mdl_list)
    model_row["FES_actual"] = "realised values"
    rows.append(model_row)

    comp_df = pd.DataFrame(rows)
    path = f"{out_dir}/fes_components_table.csv"
    comp_df.to_csv(path, index=False)
    log.info(f"FES components table saved → {path}")

    # Print to console
    print("\n── FES Component Table (z-score means, 2017) ──────────────────────────")
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
    "actual":      "#2C3E50",
    "grid":        "#EAECEE",
}
_DPI = 150

MONTHS_SHORT = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]


def _save_fig(fig: plt.Figure, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Figure saved → {path}")


def _plot_fes_comparison(monthly_df: pd.DataFrame, figures_dir: str) -> None:
    """Line chart: FES_core vs FES_macro vs FES_actual over 12 months of 2017."""
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    months = MONTHS_SHORT[:len(monthly_df)]

    for col, label, colour, ls in [
        ("fes_core",   "FES Core (core-only models)",      _PALETTE["core"],   "-"),
        ("fes_macro",  "FES Macro (core + exogenous)",     _PALETTE["macro"],  "--"),
        ("fes_actual", "FES Actual (realised 2017 values)",_PALETTE["actual"], ":"),
    ]:
        if col in monthly_df.columns:
            ax.plot(months, monthly_df[col].values, color=colour,
                    linestyle=ls, linewidth=2.2, marker="o", markersize=5,
                    label=label)

    ax.axhline(0, color="#95A5A6", linewidth=0.9, linestyle="-")
    ax.set_title("UK Anticipatory Energy–Carbon Stress Index — 2017 Monthly",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Month (2017)", fontsize=11)
    ax.set_ylabel("FES (sum of z-scores)", fontsize=11)
    ax.legend(framealpha=0.92, fontsize=10, loc="upper left")
    ax.grid(True, color=_PALETTE["grid"], linewidth=0.8)
    ax.tick_params(axis="both", labelsize=9)

    _save_fig(fig, f"{figures_dir}/fes_monthly_2017.png")


def _plot_fes_components(monthly_df: pd.DataFrame, figures_dir: str) -> None:
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
        "FES_core":   [f"z_{s}_core" for s in SERIES] + ["z_unc_core"],
        "FES_macro":  [f"z_{s}_macro" for s in SERIES] + ["z_unc_macro"],
        "FES_actual": [f"z_{s}_actual" for s in SERIES] + ["z_real_vol_actual"],
    }
    colours_bar = {
        "FES_core":   _PALETTE["core"],
        "FES_macro":  _PALETTE["macro"],
        "FES_actual": _PALETTE["actual"],
    }

    x = np.arange(len(components))
    width = 0.25
    for k, (label, cols) in enumerate(z_cols.items()):
        vals = [
            float(monthly_df[c].mean()) if c in monthly_df.columns else 0.0
            for c in cols
        ]
        bars = ax.bar(x + (k - 1) * width, vals, width,
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
    ax.set_title("FES Component Z-Scores (2017 annual mean)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Z-score", fontsize=10)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", color=_PALETTE["grid"], linewidth=0.8)

    # ── Right: FES total bars ─────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("white")

    fes_labels  = ["FES Core", "FES Macro", "FES Actual"]
    fes_cols    = ["fes_core", "fes_macro", "fes_actual"]
    fes_colours = [_PALETTE["core"], _PALETTE["macro"], _PALETTE["actual"]]
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
    ax2.set_title("Total FES — Annual Mean 2017", fontsize=12, fontweight="bold")
    ax2.set_ylabel("FES (sum of z-scores)", fontsize=10)
    ax2.grid(axis="y", color=_PALETTE["grid"], linewidth=0.8)

    fig.suptitle("Anticipatory Energy–Carbon Stress Index — Component Analysis 2017",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save_fig(fig, f"{figures_dir}/fes_components_2017.png")

def _plot_forecasts_vs_actual(
    forecasts: dict,
    actual_2017: dict,
    core_df: pd.DataFrame,
    best: dict,
    figures_dir: str,
) -> None:
    """
    One 3-panel figure showing all three series, core and macro forecasts,
    actual 2017 values, plus the historical 2005-2017 baseline.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
    fig.patch.set_facecolor("white")

    months = MONTHS_SHORT

    for ax, series in zip(axes, SERIES):
        ax.set_facecolor("white")
        col = f"{series}_growth"

        # Historical 2005-2017
        hist = core_df[
            (core_df["date"] >= TRAIN_START) & (core_df["date"] <= TRAIN_END)
        ].set_index("date")[col]
        if not hist.empty:
            ax.plot(hist.index, hist.values, color=_PALETTE[series],
                    linewidth=1.5, alpha=0.6, label="Historical 2005–2016")

        x = np.arange(12)
        fc_dates = FORECAST_DATES

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

        # Actual 2017
        act = actual_2017.get(series, np.full(12, np.nan))
        if not np.all(np.isnan(act)):
            ax.plot(fc_dates, act, color="#2C3E50", linestyle="none",
                    marker="o", markersize=5, zorder=5,
                    label="Actual 2017")

        ax.axvline(pd.Timestamp("2017-01-01"), color="#BDC3C7",
                   linewidth=1.0, linestyle=":")
        ax.set_title(f"{series.capitalize()} Growth (% YoY)",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, color=_PALETTE["grid"], linewidth=0.7)
        ax.axhline(0, color="#BDC3C7", linewidth=0.7)

    fig.suptitle(
        "UK Energy–Carbon Forecast vs Actual 2017  "
        "(core-only | macro-augmented | realised values)",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    _save_fig(fig, f"{figures_dir}/forecast_vs_actual_all_series.png")

    # Also individual series plots
    for series in SERIES:
        col = f"{series}_growth"
        hist = core_df[
            (core_df["date"] >= TRAIN_START) & (core_df["date"] <= TRAIN_END)
        ].set_index("date")[col]
        act  = actual_2017.get(series, np.full(12, np.nan))

        fig2, ax2 = plt.subplots(figsize=(12, 4))
        fig2.patch.set_facecolor("white")
        ax2.set_facecolor("white")

        if not hist.empty:
            ax2.plot(hist.index, hist.values, color=_PALETTE[series],
                     linewidth=1.8, label="Historical 2005–2016")

        for mode, ls, col_mode in [("core", "-", _PALETTE["core"]),
                                    ("macro", "--", _PALETTE["macro"])]:
            key = (series, mode)
            if key in forecasts:
                df_fc = forecasts[key].set_index("date")
                fc = df_fc["forecast"].reindex(FORECAST_DATES).values
                lb = df_fc["lower_bound"].reindex(FORECAST_DATES).values
                ub = df_fc["upper_bound"].reindex(FORECAST_DATES).values
                model = best.get(key, "?")
                ax2.plot(FORECAST_DATES, fc, color=col_mode, linestyle=ls,
                         linewidth=2, marker="o", markersize=5,
                         label=f"Forecast {mode} ({model})")
                ax2.fill_between(FORECAST_DATES, lb, ub, alpha=0.15, color=col_mode)

        if not np.all(np.isnan(act)):
            ax2.plot(FORECAST_DATES, act, color="#2C3E50", linestyle="none",
                     marker="o", markersize=5, zorder=5, label="Actual 2017")

        ax2.axvline(pd.Timestamp("2017-01-01"), color="#BDC3C7",
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

def _plot_fes_robustness(monthly_df: pd.DataFrame, figures_dir: str) -> None:
    """Two-panel chart: all forecasted FES variants (top) and actual benchmarks (bottom)."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 11), sharex=True)
    fig.patch.set_facecolor("white")
    months = MONTHS_SHORT[:len(monthly_df)]

    # ── Top: forecasted variants ──────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor("white")
    top_series = [
        ("fes_core",        "Equal-Weight Core",  _PALETTE["core"],   "-",  "o"),
        ("fes_macro",       "Equal-Weight Macro",  _PALETTE["macro"],  "--", "s"),
        ("fes_vw_core",     "VW Core",            "#9B59B6",           "-",  "^"),
        ("fes_vw_macro",    "VW Macro",           "#E91E63",           "--", "v"),
        ("fes_bayes_core",  "Bayesian Core",      "#00ACC1",           "-.", "D"),
        ("fes_bayes_macro", "Bayesian Macro",     "#FF6F00",           ":",  "P"),
    ]
    for col, label, colour, ls, marker in top_series:
        if col in monthly_df.columns:
            ax.plot(months, monthly_df[col].values, color=colour, linestyle=ls,
                    linewidth=2.0, marker=marker, markersize=5, label=label)
    for _mode, _c in [("core", "#00ACC1"), ("macro", "#FF6F00")]:
        lb_c = f"fes_bayes_{_mode}_lb"
        ub_c = f"fes_bayes_{_mode}_ub"
        if lb_c in monthly_df.columns and ub_c in monthly_df.columns:
            ax.fill_between(months, monthly_df[lb_c].values, monthly_df[ub_c].values,
                            alpha=0.09, color=_c)
    ax.axhline(0, color="#95A5A6", linewidth=0.9)
    ax.set_title("Forecasted FES Robustness Variants — 2017 Monthly", fontsize=12, fontweight="bold")
    ax.set_ylabel("FES (z-score sum)", fontsize=10)
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax.grid(True, color=_PALETTE["grid"], linewidth=0.7)

    # ── Bottom: actual FES benchmarks ────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("white")
    bot_series = [
        ("fes_actual_A", "Actual A — Rolling Volatility",  "#2C3E50", "-",  "o"),
        ("fes_actual_B", "Actual B — Cross-Component Std", "#E74C3C", "--", "s"),
        ("fes_actual_C", "Actual C — Absolute Shock",      "#27AE60", "-.", "^"),
    ]
    for col, label, colour, ls, marker in bot_series:
        if col in monthly_df.columns:
            ax2.plot(months, monthly_df[col].values, color=colour, linestyle=ls,
                     linewidth=2.0, marker=marker, markersize=5, label=label)
    ax2.axhline(0, color="#95A5A6", linewidth=0.9)
    ax2.set_title("Actual FES Benchmarks — Three Volatility Options (2017)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Month (2017)", fontsize=10)
    ax2.set_ylabel("FES (z-score sum)", fontsize=10)
    ax2.legend(fontsize=9, loc="upper left")
    ax2.grid(True, color=_PALETTE["grid"], linewidth=0.7)

    fig.suptitle("FES Robustness Analysis — All Variants and Benchmarks",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save_fig(fig, f"{figures_dir}/fes_robustness_comparison.png")


def _plot_bayesian_uncertainty(monthly_df: pd.DataFrame, figures_dir: str) -> None:
    """Bayesian FES with 95% posterior credible intervals."""
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    months = MONTHS_SHORT[:len(monthly_df)]

    bayes_styles = {"core": ("#00ACC1", "-", "D"), "macro": ("#FF6F00", "--", "P")}
    for _mode, (_c, _ls, _mk) in bayes_styles.items():
        col = f"fes_bayes_{_mode}"
        lb  = f"fes_bayes_{_mode}_lb"
        ub  = f"fes_bayes_{_mode}_ub"
        if col in monthly_df.columns:
            ax.plot(months, monthly_df[col].values, color=_c, linestyle=_ls,
                    linewidth=2.2, marker=_mk, markersize=5,
                    label=f"Bayesian FES ({_mode})")
            if lb in monthly_df.columns and ub in monthly_df.columns:
                ax.fill_between(months, monthly_df[lb].values, monthly_df[ub].values,
                                alpha=0.18, color=_c, label=f"95 % CI ({_mode})")

    # Reference equal-weight lines
    for _mode, _c, _ls in [("core", _PALETTE["core"], "-"),
                            ("macro", _PALETTE["macro"], "--")]:
        col = f"fes_{_mode}"
        if col in monthly_df.columns:
            ax.plot(months, monthly_df[col].values, color=_c, linestyle=_ls,
                    linewidth=1.2, alpha=0.4, label=f"Equal-Weight ({_mode})")

    if "fes_actual_B" in monthly_df.columns:
        ax.plot(months, monthly_df["fes_actual_B"].values, color=_PALETTE["actual"],
                linestyle=":", linewidth=1.5, marker="o", markersize=4,
                label="Actual FES (cross-component)")

    ax.axhline(0, color="#95A5A6", linewidth=0.9)
    ax.set_title("Bayesian FES — Posterior Estimates with 95 % Credible Intervals (2017)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Month (2017)", fontsize=11)
    ax.set_ylabel("Latent FES (z-score)", fontsize=11)
    ax.legend(fontsize=9, loc="upper left", ncol=2)
    ax.grid(True, color=_PALETTE["grid"], linewidth=0.7)
    _save_fig(fig, f"{figures_dir}/fes_bayesian_uncertainty.png")


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
