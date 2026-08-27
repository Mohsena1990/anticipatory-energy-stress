"""
forecast_pipeline.py
────────────────────
Macro forecasting + FES index pipeline (Stages 0–4).

Stages
──────
  Stage 0 : Load and parse UK energy/carbon/macro data from data/raw/
  Stage 1 : Preprocess core (gas, electricity, carbon) and macro datasets
  Stage 2 : Train all four time-series models in two modes per series
              core  — univariate (target series only)
              macro — multivariate (target + exogenous macro variables)
            4 models × 2 modes × 3 series = 24 model runs
  Stage 3 : Rank-aggregate models; select best per (series, mode)
  Stage 4 : FES construction — computes the equal-weighted
            FES_core / FES_macro / FES_actual; saves CSVs + figures

Simplified from an earlier version that also produced volatility-weighted/
Bayesian FES variants, a 9-scenario simulation, and a Stage 4b TS-SHAP
attribution step — all dropped as unused overhead (`src/fes_scenarios.py`
and `src/ts_shap.py` remain in the repo, unused, for reference).

Usage
─────
  python forecast_pipeline.py                           # full run
  python forecast_pipeline.py --fes-only                # skip training, recompute FES
  python forecast_pipeline.py --fast                    # fewer epochs (dev mode)
  python forecast_pipeline.py --skip-models LSTM TFT   # skip specific models
  python forecast_pipeline.py --series gas carbon       # specific series only
  python forecast_pipeline.py --tune                    # tune before training
  python forecast_pipeline.py --selection-basis validation
"""

from __future__ import annotations
import argparse
import time
from pathlib import Path
import numpy as np
import pandas as pd

from src.logging_utils import setup_logger, get_logger
setup_logger("energy_stress", log_file="outputs/logs/pipeline.log")
log = get_logger("forecast_pipeline")

ALL_MODELS = ["SARIMA", "Prophet", "LSTM", "TFT"]
ALL_SERIES = ["gas", "electricity", "carbon"]

FORECAST_DIR = "outputs/forecasts"
FES_DIR      = "outputs/fes"
FIGURES_DIR  = "outputs/figures"
TABLES_DIR   = "outputs/tables"
CORE_CSV     = "data/processed/core_energy_carbon.csv"


# ══════════════════════════════════════════════════════════════════════════════
# Console helpers
# ══════════════════════════════════════════════════════════════════════════════

def _banner(text: str) -> None:
    print("\n" + "═" * 70)
    print(f"  {text}")
    print("═" * 70)


def _stage(n: int, label: str) -> None:
    print(f"\n[STAGE {n}] {label}")
    log.info("=" * 40)
    log.info("STAGE %d: %s", n, label)
    log.info("=" * 40)


# ══════════════════════════════════════════════════════════════════════════════
# Stage 0 — Load raw data
# ══════════════════════════════════════════════════════════════════════════════

def stage0_load_data() -> None:
    from src.data_loader import load_core_dataset, load_macro_dataset
    load_core_dataset()
    load_macro_dataset()


# ══════════════════════════════════════════════════════════════════════════════
# Stage 1 — Preprocessing
# ══════════════════════════════════════════════════════════════════════════════

def stage1_preprocess():
    from src.preprocessing import preprocess_core, preprocess_macro
    core_full, core_train, core_test   = preprocess_core()
    macro_full, macro_train, macro_test = preprocess_macro()
    macro_full, macro_train, macro_test = _add_electricity_macro_lags(
        core_full, macro_full
    )
    return core_full, core_train, core_test, macro_full, macro_train, macro_test


def _add_electricity_macro_lags(
    core_full: pd.DataFrame, macro_full: pd.DataFrame,
    train_end: str = "2015-12-01", test_start: str = "2016-01-01", test_end: str = "2016-12-01",
):
    """Add 12-month electricity lag to macro features (avoids target
    leakage, one-time -- independent of the window), then split into
    train/test. Defaults reproduce the original single-year window; the
    rolling walk-forward loop passes a different window per year."""
    macro_full = macro_full.copy()
    if "electricity_growth" in core_full.columns:
        lag12 = core_full["electricity_growth"].shift(12)
        macro_full["electricity_growth_lag12"] = (
            lag12.reindex(macro_full.index).ffill().fillna(0.0)
        )
        macro_full.to_csv("data/processed/macro_processed.csv")
    macro_train = macro_full.loc[:train_end].copy()
    macro_test  = macro_full.loc[test_start:test_end].copy()
    return macro_full, macro_train, macro_test


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2 — Model training and evaluation
# ══════════════════════════════════════════════════════════════════════════════

def _get_series(
    series_name: str, core_train, core_test, core_full,
    train_end: str = "2016-12-01",
    forecast_start: str = "2017-01-01",
    forecast_end: str = "2017-12-01",
):
    """
    Extract per-series train/test/full splits plus actual target-year values.

    Parameters
    ----------
    train_end      : last month of the refit history (full_s)
    forecast_start/forecast_end : the 12-month target window to compare
                      forecasts against (defaults reproduce the original
                      single-year 2016-train/2017-forecast behavior)

    Returns
    -------
    train_s      : training period (from core_train, e.g. 2005-2015)
    test_s       : validation period (from core_test, e.g. 2016)
    full_s       : full history through train_end for final refit
    actual_target : realised target-window values (comparison column)
    eval_actual  : test-period actuals used for validation metrics
    """
    col = f"{series_name}_growth"
    train_s      = core_train[col].dropna()
    test_s       = core_test[col].dropna()
    eval_actual  = core_test[col].dropna()
    full_s       = core_full.loc[:train_end, col].dropna()
    actual_target = core_full.loc[forecast_start:forecast_end, col]

    log.info(
        "[%s] train=%d, test=%d, full=%d (through %s), actual_target=%d non-null (%s..%s)",
        series_name, len(train_s), len(test_s), len(full_s), train_end,
        actual_target.notna().sum(), forecast_start, forecast_end,
    )
    return train_s, test_s, full_s, actual_target, eval_actual


MODELS_DIR = "outputs/models"


def stage2_train_evaluate(
    series_names: list,
    models_to_run: list,
    core_train,
    core_test,
    core_full,
    macro_train,
    macro_full,
    fast: bool = False,
    model_params: dict | None = None,
    train_end: str = "2016-12-01",
    forecast_start: str = "2017-01-01",
    forecast_end: str = "2017-12-01",
    forecast_dir: str = FORECAST_DIR,
) -> list:
    """
    Train all models in both core and macro modes for every series.

    train_end/forecast_start/forecast_end default to the original
    single-year window; a rolling walk-forward caller passes a different
    window (and a year-specific forecast_dir, so each year's CSVs don't
    overwrite each other on the fixed filename pattern
    {series}_growth_pct_forecasts_{model}_{mode}.csv) per year — see
    forecast_pipeline.run_rolling.

    Returns list of result dicts (one per model × series × mode).
    """
    results: list = []
    model_params  = model_params or {}
    epochs_lstm   = 30  if fast else 100
    epochs_tft    = 20  if fast else 80

    from src.model_evaluation import merge_forecast_files

    for series in series_names:
        log.info("\n%s\nSeries: %s\n%s", "─" * 50, series.upper(), "─" * 50)

        train_s, test_s, full_s, actual_2017, eval_actual = _get_series(
            series, core_train, core_test, core_full,
            train_end=train_end, forecast_start=forecast_start, forecast_end=forecast_end,
        )

        # ── SARIMA ───────────────────────────────────────────────────────────
        if "SARIMA" in models_to_run:
            for use_macro in [False, True]:
                mode = "macro" if use_macro else "core"
                try:
                    from src.models.sarima_model import run_sarima
                    results.append(run_sarima(
                        series, train_s, test_s, full_s,
                        macro_train=macro_train, macro_full=macro_full,
                        use_macro=use_macro, actual_2017=actual_2017,
                        eval_actual=eval_actual,
                        return_model=True, forecast_dir=forecast_dir,
                        **model_params.get((series, mode, "SARIMA"), {}),
                    ))
                except Exception as e:
                    log.error("SARIMA-%s failed for %s: %s", mode, series, e, exc_info=True)

        # ── Prophet ──────────────────────────────────────────────────────────
        if "Prophet" in models_to_run:
            for use_regressors in [False, True]:
                mode = "macro" if use_regressors else "core"
                try:
                    from src.models.prophet_model import run_prophet
                    results.append(run_prophet(
                        series, train_s, test_s, full_s,
                        macro_train=macro_train, macro_full=macro_full,
                        use_regressors=use_regressors, actual_2017=actual_2017,
                        eval_actual=eval_actual,
                        return_model=True, forecast_dir=forecast_dir,
                        **model_params.get((series, mode, "Prophet"), {}),
                    ))
                except Exception as e:
                    log.error("Prophet-%s failed for %s: %s", mode, series, e, exc_info=True)

        # ── LSTM ─────────────────────────────────────────────────────────────
        if "LSTM" in models_to_run:
            for use_macro in [False, True]:
                mode = "macro" if use_macro else "core"
                try:
                    from src.models.lstm_model import run_lstm
                    _lstm_kw = {"macro_feature_set": "lstm"}
                    _lstm_kw.update(model_params.get((series, mode, "LSTM"), {}))
                    results.append(run_lstm(
                        series, train_s, test_s, full_s,
                        macro_train=macro_train, macro_full=macro_full,
                        use_macro=use_macro, actual_2017=actual_2017,
                        eval_actual=eval_actual, epochs=epochs_lstm,
                        save_dir=MODELS_DIR, forecast_dir=forecast_dir,
                        **_lstm_kw,
                    ))
                except Exception as e:
                    log.error("LSTM-%s failed for %s: %s", mode, series, e, exc_info=True)

        # ── TFT ──────────────────────────────────────────────────────────────
        if "TFT" in models_to_run:
            for use_macro in [False, True]:
                mode = "macro" if use_macro else "core"
                try:
                    from src.models.tft_model import run_tft
                    results.append(run_tft(
                        series, train_s, test_s, full_s,
                        macro_train=macro_train, macro_full=macro_full,
                        use_macro=use_macro, actual_2017=actual_2017,
                        eval_actual=eval_actual, epochs=epochs_tft,
                        save_dir=MODELS_DIR, forecast_dir=forecast_dir,
                        **model_params.get((series, mode, "TFT"), {}),
                    ))
                except Exception as e:
                    log.error("TFT-%s failed for %s: %s", mode, series, e, exc_info=True)

    merge_forecast_files(series_names, models_to_run, forecast_dir, forecast_dir)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Stage 3 — Evaluation + model ranking
# ══════════════════════════════════════════════════════════════════════════════

def stage3_evaluation(
    results: list,
    selection_basis: str = "forecast_actual",
    out_dir: str = TABLES_DIR,
    forecast_dir: str = FORECAST_DIR,
):
    """out_dir/forecast_dir default to the single-year path's fixed
    locations; the rolling walk-forward loop passes year-specific dirs so
    each year's metrics table and forecast-vs-actual read don't collide
    with (or get overwritten by) each other or the single-year path."""
    from src.model_evaluation import run_evaluation
    return run_evaluation(
        results,
        out_dir=out_dir,
        forecast_dir=forecast_dir,
        selection_basis=selection_basis,
    )


def _load_ranked_df_from_csv() -> pd.DataFrame:
    """Load the saved model metrics table when --fes-only skips training."""
    path = Path(TABLES_DIR) / "model_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Metrics table not found: {path}\n"
            "Run the full pipeline first (without --fes-only)."
        )
    df = pd.read_csv(path)
    log.info("Loaded ranked metrics from %s (%d rows)", path, len(df))
    return df


# ══════════════════════════════════════════════════════════════════════════════
# Stage 4 — FES computation
# ══════════════════════════════════════════════════════════════════════════════

def stage4_compute_fes(ranked_df: pd.DataFrame) -> None:
    from src.fes_calculator import compute_fes
    compute_fes(
        ranked_df=ranked_df,
        core_csv=CORE_CSV,
        forecast_dir=FORECAST_DIR,
        out_dir=FES_DIR,
        figures_dir=FIGURES_DIR,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Stage 4 (rolling) — walk-forward FES: train through year Y, forecast Y+1,
# repeat for every feasible Y. Makes FES Magnitude genuinely vary by
# household interview year instead of one fixed 2017-vintage constant --
# see src.ukhls_preprocessing.attach_fes_delta for how the household stream
# consumes this table.
# ══════════════════════════════════════════════════════════════════════════════

MIN_TRAIN_MONTHS = 24   # minimum split-train history before a year is "feasible"


def _feasible_as_of_years(core_full: pd.DataFrame, min_train_months: int = MIN_TRAIN_MONTHS) -> list:
    """
    Years Y such that: (a) the split-train window (data_start .. Y-1's
    December) has at least min_train_months of history, and (b) real data
    exists through Y's December for the final refit. Computed from the
    data's actual date range, not a hardcoded literal.
    """
    data_start = core_full.index.min()
    data_end   = core_full.index.max()

    years = []
    candidate = data_start.year + 1
    while True:
        split_train_end = pd.Timestamp(f"{candidate - 1}-12-01")
        refit_end        = pd.Timestamp(f"{candidate}-12-01")
        if refit_end > data_end:
            break
        months_avail = (
            (split_train_end.to_period("M") - data_start.to_period("M")).n + 1
        )
        if months_avail >= min_train_months:
            years.append(candidate)
        candidate += 1
    return years


def _plot_rolling_trend(rolling_df: pd.DataFrame, figures_dir: str) -> None:
    """One line chart: FES_core/FES_macro/FES_actual across all rolling
    years -- replaces per-year diagnostic figures (15+ years x the full
    single-year figure set would be excessive output volume)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path as _Path

    if rolling_df.empty:
        return
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for col, label, color in [
        ("fes_core", "FES Core (walk-forward)", "#8E44AD"),
        ("fes_macro", "FES Macro (walk-forward)", "#E74C3C"),
        ("fes_actual", "FES Actual (realised, target year)", "#2C3E50"),
    ]:
        if col in rolling_df.columns:
            ax.plot(rolling_df["target_year"], rolling_df[col], marker="o",
                     linewidth=2, label=label, color=color)
    ax.axhline(0, color="#95A5A6", linewidth=0.9)
    ax.set_xlabel("Target year (forecast made as of the prior December)")
    ax.set_ylabel("FES (sum of z-scores)")
    ax.set_title("Rolling Walk-Forward FES — Anticipated Shock by Target Year",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, color="#EAECEE", linewidth=0.8)
    _Path(figures_dir).mkdir(parents=True, exist_ok=True)
    out_path = f"{figures_dir}/fes_rolling_trend.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    log.info("Rolling FES trend figure saved → %s", out_path)


def _plot_rolling_metrics_heatmap(comparison_frames: list, figures_dir: str) -> None:
    """
    Supersedes the single-year (2017-only) fes_metrics_{metric}_heatmap.png
    -- same filenames, but now rows = every rolling as_of_year instead of a
    2x3 grid for one fixed year. Cell value = that year's mean RMSE/Pearson_r
    across the 3 realised-FES benchmarks (the same averaging
    _select_best_fes_variant uses for its own decision), one column per
    FES_variant (Equal_Core / Equal_Macro).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path as _Path

    if not comparison_frames:
        return
    all_comp = pd.concat(comparison_frames, ignore_index=True)

    for metric, cmap, invert, cbar_label in [
        ("RMSE", "YlOrRd", True, "Mean RMSE vs realised-FES benchmarks (lower = better)"),
        ("Pearson_r", "RdYlGn", False, "Mean Pearson r vs realised-FES benchmarks (higher = better)"),
    ]:
        if metric not in all_comp.columns:
            continue
        pivot = (
            all_comp.groupby(["as_of_year", "FES_variant"])[metric]
            .mean().reset_index()
            .pivot(index="as_of_year", columns="FES_variant", values=metric)
            .sort_index()
        )
        if pivot.empty:
            continue

        fig, ax = plt.subplots(figsize=(6, max(4, 0.4 * len(pivot))))
        fig.patch.set_facecolor("white")
        vals = pivot.values.astype(float)
        vmin, vmax = np.nanmin(vals), np.nanmax(vals)
        if np.isnan(vmin) or np.isnan(vmax):
            plt.close(fig)
            continue
        im = ax.imshow(vals, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, label=cbar_label)
        ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns, fontsize=9)
        ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index, fontsize=8)
        mid = (vmin + vmax) / 2
        for r in range(len(pivot.index)):
            for c in range(len(pivot.columns)):
                v = vals[r, c]
                if not np.isnan(v):
                    txt_colour = "white" if (invert and v > mid) else "black"
                    ax.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=8,
                            fontweight="bold", color=txt_colour)
        ax.set_xlabel("FES variant")
        ax.set_ylabel("Rolling as-of year")
        ax.set_title(f"FES Robustness by Year — {metric.replace('_', ' ')}", fontsize=12, fontweight="bold")
        fig.tight_layout()
        _Path(figures_dir).mkdir(parents=True, exist_ok=True)
        out_path = f"{figures_dir}/fes_metrics_{metric.lower()}_heatmap.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
        plt.close(fig)
        log.info("Rolling FES metrics heatmap saved → %s (%d years)", out_path, len(pivot))


_MODEL_COLORS = {
    "SARIMA":  "#3B0F70", "Prophet": "#B63679", "LSTM": "#F1605D", "TFT": "#FCB92C",
}  # fixed categorical colours (sampled from "inferno"), consistent across all 3 series charts


def _plot_model_selection_polar(model_selection_df: pd.DataFrame, series_name: str, figures_dir: str) -> None:
    """
    Polar chart: which model won each rolling year, for one series -- an
    inner ring for core mode, an outer ring for macro mode, each wedge
    coloured by the winning model. Distinct from Stage 1's per-year
    stacked-metric polar charts (those show ONE year's models compared by
    metric; this shows ALL years compared by which single model won).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path as _Path

    sub = model_selection_df[model_selection_df["series"] == series_name]
    if sub.empty:
        return
    years = sorted(sub["target_year"].unique())
    n = len(years)
    angles = np.array([i * 2 * np.pi / n for i in range(n)])
    width = (2 * np.pi / n) * 0.9

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for angle, year in zip(angles, years):
        for mode, (bottom, height) in [("core", (0, 1)), ("macro", (1, 1))]:
            row = sub[(sub["target_year"] == year) & (sub["mode"] == mode)]
            if row.empty:
                continue
            model = row["model"].iloc[0]
            ax.bar(angle, height, width=width, bottom=bottom,
                   color=_MODEL_COLORS.get(model, "#AAAAAA"), edgecolor="white",
                   linewidth=1.0, zorder=3)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels([str(y) for y in years], fontsize=9)
    ax.set_ylim(0, 2)
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["core", "macro"], fontsize=8, color="#555555")
    ax.grid(color="#EAECEE", linewidth=0.9, zorder=0)
    ax.spines["polar"].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="white", label=m)
               for m, c in _MODEL_COLORS.items()]
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=9, title="Model")
    ax.set_title(f"Which Model Was Selected, By Year — {series_name.capitalize()}\n"
                 "(inner ring = core mode, outer ring = macro mode)",
                 fontsize=12, fontweight="bold", pad=20)

    _Path(figures_dir).mkdir(parents=True, exist_ok=True)
    out_path = f"{figures_dir}/model_selection_polar_{series_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    log.info("Model-selection polar chart saved → %s (%d years)", out_path, n)


def _plot_forecast_performance_by_year(performance_df: pd.DataFrame, figures_dir: str) -> None:
    """How forecasting performed across the rolling walk-forward: the
    WINNING model's validation RMSE by target year, one panel per series,
    core vs macro mode as separate lines. Complements fes_rolling_trend.png
    (which shows the FES index itself) with the underlying forecast
    accuracy that FES is built from."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path as _Path

    if performance_df.empty:
        return
    series_list = [s for s in ALL_SERIES if s in performance_df["series"].unique()]
    if not series_list:
        return

    fig, axes = plt.subplots(1, len(series_list), figsize=(5.5 * len(series_list), 4.5), squeeze=False)
    axes = axes[0]
    fig.patch.set_facecolor("white")

    for ax, series_name in zip(axes, series_list):
        sub = performance_df[performance_df["series"] == series_name].sort_values("target_year")
        for mode, color in [("core", "#2E86AB"), ("macro", "#E67E22")]:
            m = sub[sub["mode"] == mode]
            if m.empty:
                continue
            ax.plot(m["target_year"], m["RMSE"], marker="o", linewidth=2, label=mode, color=color)
            for _, r in m.iterrows():
                ax.annotate(r["model"], (r["target_year"], r["RMSE"]),
                            textcoords="offset points", xytext=(0, 6), ha="center",
                            fontsize=6, color="#555", rotation=45)
        ax.set_title(series_name.capitalize(), fontsize=11, fontweight="bold")
        ax.set_xlabel("Target year")
        ax.set_ylabel("Winning model's validation RMSE")
        ax.legend(fontsize=8)
        ax.grid(True, color="#EAECEE", linewidth=0.8)

    fig.suptitle("How Forecasting Performed — Winning Model's RMSE by Year", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _Path(figures_dir).mkdir(parents=True, exist_ok=True)
    out_path = f"{figures_dir}/rolling_forecast_performance_by_year.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    log.info("Forecast performance-by-year figure saved → %s", out_path)


def run_rolling(
    years: list | None = None,
    series: list | None = None,
    models_to_run: list | None = None,
    fast: bool = False,
    selection_basis: str = "forecast_actual",
    min_train_months: int = MIN_TRAIN_MONTHS,
) -> pd.DataFrame:
    """
    Walk-forward rolling FES: for each feasible year Y, train through Y's
    December (refit_end), forecast Y+1's 12 months, producing one row of
    annual FES per year in outputs/fes/fes_rolling_yearly.csv.

    Cost (flagged, not hidden): ~n_years x 24 model fits (4 models x
    2 modes x 3 series). Recommend --fast for a first end-to-end pass.

    Each year's forecast CSVs/metrics land in year-specific subdirectories
    (outputs/forecasts_rolling/{year}/, outputs/tables/rolling/{year}/,
    outputs/fes/rolling/{year}/) so they don't overwrite each other or the
    single-year path's fixed-location outputs.
    """
    series        = series        or ALL_SERIES
    models_to_run = models_to_run or ALL_MODELS

    Path(FORECAST_DIR).mkdir(parents=True, exist_ok=True)
    Path(FES_DIR).mkdir(parents=True, exist_ok=True)
    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)

    _stage(0, "Loading raw UK data")
    stage0_load_data()

    _stage(1, "Preprocessing (core + macro, full history — computed once)")
    core_full, _, _, macro_full_raw, _, _ = stage1_preprocess()

    if years is None:
        years = _feasible_as_of_years(core_full, min_train_months)
    log.info("Rolling walk-forward FES: %d feasible years: %s", len(years), years)

    from src.preprocessing import split
    from src.fes_calculator import compute_fes

    rows = []
    comparison_frames = []
    monthly_frames = []
    model_selection_rows = []
    performance_rows = []
    for as_of_year in years:
        split_train_end  = f"{as_of_year - 1}-12-01"
        split_test_start = f"{as_of_year}-01-01"
        split_test_end   = f"{as_of_year}-12-01"
        refit_end        = f"{as_of_year}-12-01"
        forecast_dates   = pd.date_range(f"{as_of_year + 1}-01-01", periods=12, freq="MS")

        _stage(2, f"Rolling year as_of={as_of_year} "
                  f"(split train<{as_of_year}, test={as_of_year}, "
                  f"refit<={as_of_year}, forecast={as_of_year + 1})")

        core_train, core_test = split(core_full, split_train_end, split_test_start, split_test_end)
        macro_full, macro_train, macro_test = _add_electricity_macro_lags(
            core_full, macro_full_raw, split_train_end, split_test_start, split_test_end,
        )

        year_forecast_dir = f"{FORECAST_DIR}_rolling/{as_of_year}"
        year_tables_dir    = f"{TABLES_DIR}/rolling/{as_of_year}"
        year_fes_dir        = f"{FES_DIR}/rolling/{as_of_year}"

        try:
            results = stage2_train_evaluate(
                series, models_to_run, core_train, core_test, core_full,
                macro_train, macro_full, fast=fast,
                train_end=refit_end,
                forecast_start=forecast_dates.min().strftime("%Y-%m-%d"),
                forecast_end=forecast_dates.max().strftime("%Y-%m-%d"),
                forecast_dir=year_forecast_dir,
            )
            _, ranked_df, best = stage3_evaluation(
                results, selection_basis, out_dir=year_tables_dir, forecast_dir=year_forecast_dir,
            )

            # Which model won this year, per (series, mode) -- previously
            # discarded (the "_" in the old unpacking) even though it was
            # already computed fresh every year; kept here to answer "based
            # on the year, which model was selected" per series.
            for (series_name, mode), winner in best.items():
                model_selection_rows.append({
                    "as_of_year": as_of_year, "target_year": as_of_year + 1,
                    "series": series_name, "mode": mode, "model": winner,
                })
                win_row = ranked_df[
                    (ranked_df["series_name"] == series_name)
                    & (ranked_df["mode"] == mode)
                    & (ranked_df["model"] == winner)
                ]
                if not win_row.empty:
                    performance_rows.append({
                        "as_of_year": as_of_year, "target_year": as_of_year + 1,
                        "series": series_name, "mode": mode, "model": winner,
                        "RMSE": float(win_row["RMSE"].iloc[0]) if "RMSE" in win_row.columns else float("nan"),
                        "forecast_actual_RMSE": float(win_row["forecast_actual_RMSE"].iloc[0])
                            if "forecast_actual_RMSE" in win_row.columns and pd.notna(win_row["forecast_actual_RMSE"].iloc[0])
                            else float("nan"),
                    })

            monthly_df = compute_fes(
                ranked_df, core_csv=CORE_CSV, forecast_dir=year_forecast_dir,
                out_dir=year_fes_dir, figures_dir=year_fes_dir,
                train_start="2005-01-01", train_end=refit_end,
                forecast_dates=forecast_dates, skip_diagnostics=True,
            )
        except Exception as e:
            log.error("Rolling year as_of=%d failed: %s", as_of_year, e, exc_info=True)
            continue

        row = {"as_of_year": as_of_year, "target_year": as_of_year + 1}
        for col in ["fes_core", "fes_macro", "fes_actual"]:
            row[col] = float(monthly_df[col].mean()) if col in monthly_df.columns else float("nan")
        rows.append(row)

        # Keep the full 12-month detail too (not just its annual mean) --
        # `monthly_df` already has one row per calendar month of the
        # target year; collapsing straight to a mean here would throw away
        # exactly the resolution src.ukhls_preprocessing.attach_fes_delta
        # needs to give each household-wave row a FES Magnitude specific to
        # its own interview month, not just its interview year.
        month_detail = monthly_df[["date"] + [c for c in ["fes_core", "fes_macro", "fes_actual"] if c in monthly_df.columns]].copy()
        month_detail["as_of_year"] = as_of_year
        monthly_frames.append(month_detail)

        comp_path = f"{year_fes_dir}/fes_comparison_metrics.csv"
        if Path(comp_path).exists():
            comp = pd.read_csv(comp_path)
            comp["as_of_year"] = as_of_year
            comparison_frames.append(comp)

    rolling_df = pd.DataFrame(rows)
    rolling_path = f"{FES_DIR}/fes_rolling_yearly.csv"
    rolling_df.to_csv(rolling_path, index=False)
    log.info("Rolling FES table saved → %s (%d years)", rolling_path, len(rolling_df))

    if monthly_frames:
        monthly_rolling_df = pd.concat(monthly_frames, ignore_index=True)
        monthly_rolling_df["date"] = pd.to_datetime(monthly_rolling_df["date"])
        monthly_rolling_df["target_year"] = monthly_rolling_df["date"].dt.year
        monthly_rolling_df["target_month"] = monthly_rolling_df["date"].dt.month
        monthly_path = f"{FES_DIR}/fes_rolling_monthly.csv"
        monthly_rolling_df.to_csv(monthly_path, index=False)
        log.info("Rolling monthly FES table saved → %s (%d year x month rows)",
                  monthly_path, len(monthly_rolling_df))

    _select_best_fes_variant(comparison_frames, FES_DIR)

    _plot_rolling_trend(rolling_df, FIGURES_DIR)
    _plot_rolling_metrics_heatmap(comparison_frames, FIGURES_DIR)

    if model_selection_rows:
        model_selection_df = pd.DataFrame(model_selection_rows)
        model_selection_df.to_csv(f"{FES_DIR}/model_selection_by_year.csv", index=False)
        for series_name in series:
            _plot_model_selection_polar(model_selection_df, series_name, FIGURES_DIR)

    if performance_rows:
        performance_df = pd.DataFrame(performance_rows)
        performance_df.to_csv(f"{FES_DIR}/forecast_performance_by_year.csv", index=False)
        _plot_forecast_performance_by_year(performance_df, FIGURES_DIR)

    return rolling_df


def _select_best_fes_variant(comparison_frames: list, out_dir: str) -> str | None:
    """
    Pick ONE FES variant (Equal_Core or Equal_Macro) to use exclusively
    downstream, instead of carrying both -- lowest mean RMSE against the
    realised FES benchmarks, averaged across every rolling year AND every
    actual-FES benchmark (Actual_RollingVol/CrossComp/AbsShock) for a
    robust decision rather than one arbitrary year/benchmark pick.

    Saves outputs/fes/fes_variant_selection.csv (one row per variant, plus
    which one was chosen) so the decision is inspectable, not just logged.
    Returns the winning variant's fes_core/fes_macro column name (e.g.
    "fes_core"), or None if no comparison data was available.
    """
    if not comparison_frames:
        log.warning("No FES comparison metrics available -- cannot select a best variant.")
        return None

    all_comp = pd.concat(comparison_frames, ignore_index=True)
    by_variant = (
        all_comp.groupby("FES_variant")["RMSE"]
        .agg(mean_rmse="mean", n_comparisons="count")
        .reset_index()
        .sort_values("mean_rmse")
    )
    winner_label = by_variant.iloc[0]["FES_variant"]
    by_variant["chosen"] = by_variant["FES_variant"] == winner_label

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    sel_path = f"{out_dir}/fes_variant_selection.csv"
    by_variant.to_csv(sel_path, index=False)

    variant_to_col = {"Equal_Core": "fes_core", "Equal_Macro": "fes_macro"}
    winner_col = variant_to_col.get(winner_label)
    log.info(
        "FES variant selection (mean RMSE across %d years x benchmarks): %s -- "
        "winner = %s (%s), saved -> %s",
        len(comparison_frames), by_variant[["FES_variant", "mean_rmse"]].to_dict("records"),
        winner_label, winner_col, sel_path,
    )
    return winner_col


# ══════════════════════════════════════════════════════════════════════════════
# Public run() — callable from main.py
# ══════════════════════════════════════════════════════════════════════════════

def run(
    series: list | None = None,
    models_to_run: list | None = None,
    fast: bool = False,
    fes_only: bool = False,
    tune: bool = False,
    selection_basis: str = "forecast_actual",
) -> None:
    """
    Execute Stages 0–4.

    Parameters
    ----------
    series        : series to forecast (default: gas, electricity, carbon)
    models_to_run : models to train (default: all four)
    fast          : use fewer epochs for LSTM/TFT (development mode)
    fes_only      : skip training; recompute FES from existing forecast CSVs
    tune          : run hyperparameter tuning before final training
    selection_basis : 'forecast_actual' or 'validation'
    """
    series        = series        or ALL_SERIES
    models_to_run = models_to_run or ALL_MODELS

    Path("outputs/logs").mkdir(parents=True, exist_ok=True)
    Path(FORECAST_DIR).mkdir(parents=True, exist_ok=True)
    Path(FES_DIR).mkdir(parents=True, exist_ok=True)
    Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)
    Path(TABLES_DIR).mkdir(parents=True, exist_ok=True)
    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)

    if fes_only:
        _stage(4, "FES computation — using existing forecast CSVs")
        ranked_df = _load_ranked_df_from_csv()
        stage4_compute_fes(ranked_df)
        return

    _stage(0, "Loading raw UK data")
    stage0_load_data()

    _stage(1, "Preprocessing (core + macro)")
    (core_full, core_train, core_test,
     macro_full, macro_train, _) = stage1_preprocess()

    tuned_params: dict = {}
    if tune:
        _stage(2, "Hyperparameter tuning")
        from src.tuning import tune_models
        tuned_params, _ = tune_models(
            series, models_to_run,
            core_train, core_test, core_full,
            macro_train, macro_full,
            fast=fast, selection_basis=selection_basis,
            out_dir="outputs/tuning",
        )
        _print_tuning_winners(tuned_params)

    _stage(2, "Model training & evaluation (4 models × 2 modes × 3 series)")
    results = stage2_train_evaluate(
        series, models_to_run,
        core_train, core_test, core_full,
        macro_train, macro_full,
        fast=fast, model_params=tuned_params,
    )

    _stage(3, "Metrics & model ranking")
    _, ranked_df, best = stage3_evaluation(results, selection_basis)
    _print_best(best)

    _stage(4, "FES construction (equal-weighted)")
    stage4_compute_fes(ranked_df)


# ══════════════════════════════════════════════════════════════════════════════
# CLI helpers
# ══════════════════════════════════════════════════════════════════════════════

def _print_best(best: dict) -> None:
    print("\n── Best models ──")
    for (series, mode), winner in best.items():
        print(f"  {series:15s} [{mode:5s}] → {winner}")


def _print_tuning_winners(tuned_params: dict) -> None:
    if not tuned_params:
        print("  No tuned parameters selected.")
        return
    print("\n── Tuning winners ──")
    for (series, mode, model), params in sorted(tuned_params.items()):
        print(f"  {series:15s} [{mode:5s}] {model:8s} → {params}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Forecast pipeline — Stages 0–4 (training + FES)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--fast", action="store_true",
                        help="Fewer LSTM/TFT epochs (development mode)")
    parser.add_argument("--fes-only", action="store_true",
                        help="Skip training; recompute FES from existing forecast CSVs")
    parser.add_argument("--skip-models", nargs="+", default=[], metavar="MODEL",
                        help="Models to skip: SARIMA Prophet LSTM TFT")
    parser.add_argument("--series", nargs="+", default=ALL_SERIES,
                        choices=ALL_SERIES,
                        help="Series to forecast (default: all three)")
    parser.add_argument("--tune", action="store_true",
                        help="Hyperparameter tuning before final training")
    parser.add_argument("--selection-basis",
                        choices=["forecast_actual", "validation"],
                        default="forecast_actual",
                        help="Model selection criterion (default: forecast_actual)")
    parser.add_argument("--rolling", action="store_true",
                        help="Walk-forward rolling FES instead of the single-year "
                             "path: train through year Y, forecast Y+1, repeat for "
                             "every feasible Y (~n_years x 24 model fits -- see "
                             "run_rolling's docstring for the cost). Ignores "
                             "--fes-only/--tune.")
    args = parser.parse_args()

    models_to_run = [m for m in ALL_MODELS if m not in args.skip_models]

    _banner("Anticipatory Energy–Carbon Stress — Forecast Pipeline (Stages 0–4)")
    print(f"  Series      : {args.series}")
    print(f"  Models      : {models_to_run}")
    print(f"  Fast mode   : {args.fast}")
    print(f"  Rolling     : {args.rolling}")
    if not args.rolling:
        print(f"  FES only    : {args.fes_only}")
        print(f"  Tuning      : {args.tune}")
        print(f"  Selection   : {args.selection_basis}")

    t0 = time.time()
    if args.rolling:
        run_rolling(
            series=args.series,
            models_to_run=models_to_run,
            fast=args.fast,
            selection_basis=args.selection_basis,
        )
    else:
        run(
            series=args.series,
            models_to_run=models_to_run,
            fast=args.fast,
            fes_only=args.fes_only,
            tune=args.tune,
            selection_basis=args.selection_basis,
        )
    _banner(f"Forecast pipeline complete  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
