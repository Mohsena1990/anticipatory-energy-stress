"""
fes_calculator.py
─────────────────
Compute three Forecasted Energy-Carbon Stress (FES) indices for 2018.

Three analytical baselines
──────────────────────────
  FES_core   — built from core-only model forecasts (target series only)
  FES_macro  — built from macro-augmented model forecasts (core + exogenous)
  FES_actual — built from realised 2018 values (benchmark)

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
  outputs/fes/fes_monthly_2018.csv       — 12 rows × all z-components + FES
  outputs/fes/fes_summary_2018.csv       — annual mean FES and components
  outputs/fes/fes_components_table.csv   — cross-baseline comparison table
  outputs/figures/fes_monthly_2018.png   — FES time-series (3 variants)
  outputs/figures/fes_components_2018.png — component breakdown bars
  outputs/figures/forecast_vs_actual_{series}.png  — per-series forecast plot
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
TRAIN_END      = "2017-12-01"
FORECAST_DATES = pd.date_range("2018-01-01", periods=12, freq="MS")


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _find_best_models(ranked_df: pd.DataFrame) -> dict:
    """Return {(series, mode): model_name} — lowest rank_score per group."""
    best: dict = {}
    for (series, mode), grp in ranked_df.groupby(["series_name", "mode"]):
        best[(series, mode)] = grp.loc[grp["rank_score"].idxmin(), "model"]
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
    return stats


def _zscore_array(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (x - mean) / (std if std > 1e-10 else 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# Main computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_fes(
    ranked_df: pd.DataFrame,
    core_csv: str = "data/raw/core_energy_carbon.csv",
    forecast_dir: str = "outputs/forecasts",
    out_dir: str = "outputs/fes",
    figures_dir: str = "outputs/figures",
) -> pd.DataFrame:
    """
    Compute FES_core, FES_macro, FES_actual for each month of 2018.

    Parameters
    ----------
    ranked_df    : model_evaluation output (used to pick best model per series/mode)
    core_csv     : path to Dataset A (2005-2018 actual values)
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

    # ── Extract actual 2018 values ────────────────────────────────────────────
    actual_2018: dict = {}
    for series in SERIES:
        col = f"{series}_growth"
        sub = core_df[
            (core_df["date"] >= "2018-01-01") & (core_df["date"] <= "2018-12-01")
        ].set_index("date")
        if col in sub.columns:
            actual_2018[series] = sub[col].reindex(FORECAST_DATES).values
        else:
            # Try to get from forecast CSV actual column
            for mode in MODES:
                key = (series, mode)
                if key in forecasts and "actual" in forecasts[key].columns:
                    actual_2018[series] = (
                        forecasts[key].set_index("date")["actual"]
                        .reindex(FORECAST_DATES).values
                    )
                    break
            else:
                log.warning(f"No actual 2018 data for {series}; using NaN")
                actual_2018[series] = np.full(12, np.nan)

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

        # ── FES_actual ────────────────────────────────────────────────────────
        z_actual = []
        for series in SERIES:
            st = stats[series]
            act_val = actual_2018[series][i] if i < len(actual_2018[series]) else np.nan
            z_a = _zscore_array(np.array([act_val]), st["mean"], st["std"])[0] \
                  if not np.isnan(act_val) else np.nan
            row[f"actual_{series}"]   = round(float(act_val), 5) if not np.isnan(act_val) else np.nan
            row[f"z_{series}_actual"] = round(float(z_a), 5)     if not np.isnan(z_a)     else np.nan
            z_actual.append(z_a)

        # Realized volatility = cross-sectional std of z-scored actuals
        z_actual_arr = np.array([v for v in z_actual if not np.isnan(v)])
        real_vol = float(np.std(z_actual_arr)) if len(z_actual_arr) > 1 else np.nan

        # Z-score the realized volatility using training cross-sectional dispersion
        # Compute training-period RealVol as reference
        row["real_vol_actual"] = round(real_vol, 5) if not np.isnan(real_vol) else np.nan

        fes_actual = float(np.nansum(z_actual) + (real_vol if not np.isnan(real_vol) else 0))
        row["fes_actual"] = round(fes_actual, 5)

        monthly_rows.append(row)

    monthly_df = pd.DataFrame(monthly_rows)

    # ── Save monthly CSV ──────────────────────────────────────────────────────
    monthly_path = f"{out_dir}/fes_monthly_2018.csv"
    monthly_df.to_csv(monthly_path, index=False)
    log.info(f"Monthly FES saved → {monthly_path}")

    # ── Build summary / component table ──────────────────────────────────────
    _save_summary(monthly_df, out_dir)
    _save_component_table(monthly_df, best, out_dir)

    # ── Generate figures ──────────────────────────────────────────────────────
    _plot_fes_comparison(monthly_df, figures_dir)
    _plot_fes_components(monthly_df, figures_dir)
    _plot_forecasts_vs_actual(forecasts, actual_2018, core_df, best, figures_dir)

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
            out_path=f"{figures_dir}/prediction_intervals_2018.png",
        )
    except Exception as e:
        log.warning(f"PI figure failed: {e}")

    # ── 6 static 2018 model-comparison figures (3 series × 2 modes) ──────────
    try:
        from src.plotting_utils import plot_all_2018_comparisons
        plot_all_2018_comparisons(forecast_dir, figures_dir)
        log.info("Static 2018 comparison figures complete")
    except Exception as e:
        log.warning(f"Static 2018 comparison figures failed: {e}")

    # ── 6 interactive HTML timeline figures (3 series × 2 modes) ─────────────
    try:
        from src.plotting_utils import plot_all_interactive_forecasts
        plot_all_interactive_forecasts(core_df, forecast_dir, figures_dir)
        log.info("Interactive timeline figures complete")
    except Exception as e:
        log.warning(f"Interactive timeline figures failed: {e}")

    return monthly_df


# ══════════════════════════════════════════════════════════════════════════════
# Summary tables
# ══════════════════════════════════════════════════════════════════════════════

def _save_summary(monthly_df: pd.DataFrame, out_dir: str) -> None:
    rows = []
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

        unc_col = f"z_unc_{mode}" if mode in ("core", "macro") else "real_vol_actual"
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

    summary_df = pd.DataFrame(rows)
    path = f"{out_dir}/fes_summary_2018.csv"
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
    carbon_growth      |  z_mean  |  z_mean   |  z_mean
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
    col_a = "real_vol_actual"
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
    print("\n── FES Component Table (z-score means, 2018) ──────────────────────────")
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
    """Line chart: FES_core vs FES_macro vs FES_actual over 12 months of 2018."""
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    months = MONTHS_SHORT[:len(monthly_df)]

    for col, label, colour, ls in [
        ("fes_core",   "FES Core (core-only models)",      _PALETTE["core"],   "-"),
        ("fes_macro",  "FES Macro (core + exogenous)",     _PALETTE["macro"],  "--"),
        ("fes_actual", "FES Actual (realised 2018 values)",_PALETTE["actual"], ":"),
    ]:
        if col in monthly_df.columns:
            ax.plot(months, monthly_df[col].values, color=colour,
                    linestyle=ls, linewidth=2.2, marker="o", markersize=5,
                    label=label)

    ax.axhline(0, color="#95A5A6", linewidth=0.9, linestyle="-")
    ax.set_title("UK Anticipatory Energy–Carbon Stress Index — 2018 Monthly",
                 fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Month (2018)", fontsize=11)
    ax.set_ylabel("FES (sum of z-scores)", fontsize=11)
    ax.legend(framealpha=0.92, fontsize=10, loc="upper left")
    ax.grid(True, color=_PALETTE["grid"], linewidth=0.8)
    ax.tick_params(axis="both", labelsize=9)

    _save_fig(fig, f"{figures_dir}/fes_monthly_2018.png")


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
        "FES_actual": [f"z_{s}_actual" for s in SERIES] + ["real_vol_actual"],
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
    ax.set_title("FES Component Z-Scores (2018 annual mean)", fontsize=12, fontweight="bold")
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
    ax2.set_title("Total FES — Annual Mean 2018", fontsize=12, fontweight="bold")
    ax2.set_ylabel("FES (sum of z-scores)", fontsize=10)
    ax2.grid(axis="y", color=_PALETTE["grid"], linewidth=0.8)

    fig.suptitle("Anticipatory Energy–Carbon Stress Index — Component Analysis 2018",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save_fig(fig, f"{figures_dir}/fes_components_2018.png")


def _plot_forecasts_vs_actual(
    forecasts: dict,
    actual_2018: dict,
    core_df: pd.DataFrame,
    best: dict,
    figures_dir: str,
) -> None:
    """
    One 3-panel figure showing all three series, core and macro forecasts,
    actual 2018 values, plus the historical 2005-2017 baseline.
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
                    linewidth=1.5, alpha=0.6, label="Historical 2005–2017")

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

        # Actual 2018
        act = actual_2018.get(series, np.full(12, np.nan))
        if not np.all(np.isnan(act)):
            ax.plot(fc_dates, act, color="#2C3E50", linestyle="none",
                    marker="D", markersize=6, zorder=5,
                    label="Actual 2018")

        ax.axvline(pd.Timestamp("2018-01-01"), color="#BDC3C7",
                   linewidth=1.0, linestyle=":")
        ax.set_title(f"{series.capitalize()} Growth (% YoY)",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, color=_PALETTE["grid"], linewidth=0.7)
        ax.axhline(0, color="#BDC3C7", linewidth=0.7)

    fig.suptitle(
        "UK Energy–Carbon Forecast vs Actual 2018  "
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
        act  = actual_2018.get(series, np.full(12, np.nan))

        fig2, ax2 = plt.subplots(figsize=(12, 4))
        fig2.patch.set_facecolor("white")
        ax2.set_facecolor("white")

        if not hist.empty:
            ax2.plot(hist.index, hist.values, color=_PALETTE[series],
                     linewidth=1.8, label="Historical 2005–2017")

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
                     marker="D", markersize=7, zorder=5, label="Actual 2018")

        ax2.axvline(pd.Timestamp("2018-01-01"), color="#BDC3C7",
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
