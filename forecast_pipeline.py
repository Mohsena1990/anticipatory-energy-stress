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
and `src/ts_shap.py` have since been removed from the repo entirely;
recoverable from git history if ever needed for comparison).

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
from src.config import DEFAULT_TARGET_YEAR
setup_logger("energy_stress", log_file="outputs_v2/logs/pipeline.log")
log = get_logger("forecast_pipeline")

ALL_MODELS = ["SARIMA", "Prophet", "LSTM", "TFT"]
ALL_SERIES = ["gas", "electricity", "carbon"]

FORECAST_DIR = "outputs_v2/forecasts"
FES_DIR      = "outputs_v2/fes"
FIGURES_DIR  = "outputs_v2/figures"
TABLES_DIR   = "outputs_v2/tables"
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


MODELS_DIR = "outputs_v2/models"


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

        train_s, test_s, full_s, actual_target, eval_actual = _get_series(
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
                        use_macro=use_macro, actual_target=actual_target,
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
                        use_regressors=use_regressors, actual_target=actual_target,
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
                        use_macro=use_macro, actual_target=actual_target,
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
                        use_macro=use_macro, actual_target=actual_target,
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
    selection_basis: str = "validation",
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
    """Load the saved model metrics table when --fes-only skips training.

    Filename must match model_evaluation.save_metrics_table's default
    ("model_metrics_comparison") -- this previously looked for
    "model_metrics.csv", a name nothing in the pipeline ever writes, so
    --fes-only always raised FileNotFoundError even right after a full run.
    """
    path = Path(TABLES_DIR) / "model_metrics_comparison.csv"
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

def stage4_compute_fes(
    ranked_df: pd.DataFrame,
    train_start: str = "2005-01-01",
    train_end: str = "2016-12-01",
    forecast_dates: pd.DatetimeIndex | None = None,
) -> None:
    from src.fes_calculator import compute_fes
    kwargs = {}
    if forecast_dates is not None:
        kwargs["forecast_dates"] = forecast_dates
    compute_fes(
        ranked_df=ranked_df,
        core_csv=CORE_CSV,
        forecast_dir=FORECAST_DIR,
        out_dir=FES_DIR,
        figures_dir=FIGURES_DIR,
        train_start=train_start,
        train_end=train_end,
        **kwargs,
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


def _compute_default_window(
    core_full: pd.DataFrame, min_train_months: int = MIN_TRAIN_MONTHS,
    override_target_year: int | None = None,
) -> tuple[str, str, str, str, pd.DatetimeIndex]:
    """
    "One year ahead" default window: train on everything through the most
    recent year that has a full December of real data (the latest entry in
    `_feasible_as_of_years`), validate on that year, then forecast the year
    right after it. Computed fresh from the data's own date range every
    run, so as new gas/electricity/carbon data lands (see
    src.data_loader._compute_forecast_end) the forecast window advances
    with it automatically instead of staying pinned to one fixed calendar
    year (e.g. 2017).

    override_target_year : if given, forecast exactly this year instead of
        the dynamically-detected latest one -- e.g. pin a delivery to 2025
        (the most recent fully-realised year) without changing the default
        for future runs. Must be a feasible target (its as-of year, i.e.
        override_target_year - 1, needs enough training history and real
        data through its December) or this raises ValueError.

    Returns
    -------
    (split_train_end, split_test_start, split_test_end, refit_end, forecast_dates)
    """
    years = _feasible_as_of_years(core_full, min_train_months)
    if not years:
        raise ValueError(
            "Not enough history in the core dataset to compute a default "
            f"forecast window (need >= {min_train_months} months before "
            "the first feasible validation year)."
        )
    if override_target_year is not None:
        as_of_year = override_target_year - 1
        if as_of_year not in years:
            raise ValueError(
                f"target_year={override_target_year} is not feasible with the "
                f"current data (needs as-of-year={as_of_year} in the feasible "
                f"set {years[0]}..{years[-1]}). Feasible target years: "
                f"{years[0] + 1}..{years[-1] + 1}."
            )
    else:
        as_of_year = years[-1]
    split_train_end  = f"{as_of_year - 1}-12-01"
    split_test_start = f"{as_of_year}-01-01"
    split_test_end   = f"{as_of_year}-12-01"
    refit_end        = f"{as_of_year}-12-01"
    forecast_dates   = pd.date_range(f"{as_of_year + 1}-01-01", periods=12, freq="MS")
    return split_train_end, split_test_start, split_test_end, refit_end, forecast_dates


def _infer_forecast_dates_from_csvs(forecast_dir: str, series_names: list) -> pd.DatetimeIndex:
    """Recover the forecast window from already-saved forecast CSVs.

    Used by --fes-only, which intentionally skips Stage 0/1 (no re-read of
    raw data) and so never recomputes the window from the core dataset --
    it must instead match whatever window the existing forecast CSVs were
    actually produced with.
    """
    for s in series_names:
        path = f"{forecast_dir}/{s}_growth_pct_forecasts_all.csv"
        if Path(path).exists():
            dates = pd.read_csv(path, parse_dates=["date"])["date"]
            if not dates.empty:
                return pd.DatetimeIndex(sorted(dates.unique())[-12:])
    raise FileNotFoundError(
        f"No forecast CSVs found under {forecast_dir} to infer the forecast "
        "window for --fes-only. Run the full pipeline first (without --fes-only)."
    )


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
        ("fes_weighted", "FES Weighted (inverse-RMSE, walk-forward)", "#F1C40F"),
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


def _plot_rolling_performance_polar(performance_df: pd.DataFrame, mode: str, figures_dir: str) -> None:
    """
    Grouped circular bar chart (house style, see plotting_utils.
    plot_grouped_circular_bars): one group per series (gas/electricity/
    carbon), one bar per rolling target year within that series, bar
    height = the winning model's validation RMSE that year, bar colour =
    which model won -- real per-year magnitude, not just an equal-height
    coloured ring segment (the previous design). One chart per mode
    (core/macro) since a single chart can't cleanly carry two RMSE scales
    at once; supersedes the old 3-per-series ring charts (2 files instead
    of 3, same total information).
    """
    from src.plotting_utils import plot_grouped_circular_bars, MODEL_COLORS, PALETTE

    sub = performance_df[performance_df["mode"] == mode]
    if sub.empty:
        return

    series_list = [s for s in ALL_SERIES if s in sub["series"].unique()]
    groups: dict = {}
    item_colors: dict = {}
    for series_name in series_list:
        s_sub = sub[sub["series"] == series_name].sort_values("target_year")
        groups[series_name] = {
            str(int(r["target_year"])): max(float(r["RMSE"]), 0.0)
            for _, r in s_sub.iterrows()
        }
        for _, r in s_sub.iterrows():
            item_colors[(series_name, str(int(r["target_year"])))] = MODEL_COLORS.get(r["model"], "#AAAAAA")

    group_colors = {s: PALETTE.get(s, "#888888") for s in series_list}
    models_seen = sorted(sub["model"].unique())
    out_path = f"{figures_dir}/model_selection_polar_{mode}.png"
    plot_grouped_circular_bars(
        groups, group_colors, out_path,
        title=f"Rolling Walk-Forward Performance — {mode.capitalize()} Mode\n"
              "bar = winning model's RMSE that year, colour = which model won",
        item_colors=item_colors,
        color_legend={m: MODEL_COLORS.get(m, "#AAAAAA") for m in models_seen},
    )
    log.info("Rolling performance polar chart saved → %s (%d series)", out_path, len(series_list))


def _plot_forecast_performance_by_year(performance_df: pd.DataFrame, figures_dir: str) -> None:
    """How forecasting performed across the rolling walk-forward: the
    WINNING model's validation RMSE by target year -- one figure PER
    SERIES (matching forecast_vs_actual_{series}.png's pattern), core vs
    macro mode shown as linestyle (solid/dashed), and each point MARKER
    coloured by which model actually won that year (plotting_utils.
    MODEL_COLORS, the same palette model_selection_polar_*.png uses)
    instead of naming the
    model as rotated text next to each point. Complements
    fes_rolling_trend.png (the FES index itself) with the underlying
    forecast accuracy that FES is built from."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path as _Path
    from src.plotting_utils import MODEL_COLORS

    if performance_df.empty:
        return
    series_list = [s for s in ALL_SERIES if s in performance_df["series"].unique()]
    if not series_list:
        return

    _Path(figures_dir).mkdir(parents=True, exist_ok=True)
    mode_styles = {"core": "-", "macro": "--"}

    for series_name in series_list:
        sub = performance_df[performance_df["series"] == series_name].sort_values("target_year")
        if sub.empty:
            continue

        fig, ax = plt.subplots(figsize=(7, 4.8))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        models_seen: set[str] = set()
        for mode, ls in mode_styles.items():
            m = sub[sub["mode"] == mode]
            if m.empty:
                continue
            ax.plot(m["target_year"], m["RMSE"], linestyle=ls, linewidth=1.6,
                     color="#B5B5B5", zorder=2)
            point_colors = [MODEL_COLORS.get(mdl, "#AAAAAA") for mdl in m["model"]]
            ax.scatter(m["target_year"], m["RMSE"], c=point_colors, s=90,
                        edgecolor="white", linewidth=1.2, zorder=3)
            models_seen.update(m["model"].unique())

        ax.set_title(f"How Forecasting Performed — {series_name.capitalize()}",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Target year")
        ax.set_ylabel("Winning model's validation RMSE")
        ax.grid(True, color="#EAECEE", linewidth=0.8)

        mode_handles = [
            plt.Line2D([0], [0], color="#7F7F7F", linestyle=ls, linewidth=1.8, label=mode)
            for mode, ls in mode_styles.items() if mode in sub["mode"].unique()
        ]
        model_handles = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=MODEL_COLORS.get(mdl, "#AAAAAA"),
                       markersize=9, markeredgecolor="white", label=mdl)
            for mdl in sorted(models_seen)
        ]
        ax.legend(handles=mode_handles + model_handles, fontsize=8, loc="best",
                   title="Line = mode, dot colour = winning model", title_fontsize=7.5)

        fig.tight_layout()
        out_path = f"{figures_dir}/rolling_forecast_performance_{series_name}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        fig.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
        plt.close(fig)
        log.info("Forecast performance-by-year figure saved → %s", out_path)


def run_rolling(
    years: list | None = None,
    series: list | None = None,
    models_to_run: list | None = None,
    fast: bool = False,
    selection_basis: str = "validation",
    min_train_months: int = MIN_TRAIN_MONTHS,
    max_target_year: int | None = DEFAULT_TARGET_YEAR,
    tune: bool = True,
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

    max_target_year : caps the dynamically-detected feasible years so the
        walk-forward stops at this target year (and `years` is not
        explicitly passed). Defaults to src.config.DEFAULT_TARGET_YEAR
        (2025) -- the raw price data updates independently of (and faster
        than) the UKHLS panel this project is built around, so leaving
        this uncapped would silently extend the rolling result past the
        years the household stream can actually use, into still-partial or
        panel-uncovered years. Pass a different year (or None, to remove
        the cap and run through every feasible year the data allows) to
        override for a single run. Ignored when `years` is given
        explicitly.

    tune : run src.tuning.tune_models ONCE (not once per rolling year --
        a literal per-year call would multiply this function's already
        ~n_years x 24-model-fit cost by another ~n_years, into many hours
        even in --fast mode, disproportionate given src.tuning's own
        docstring that broad grids on monthly data are more likely to
        overfit noise than help). Tuned on the LATEST feasible year's split
        (the most data-rich window), then those same hyperparameters are
        reused for every rolling year's stage2_train_evaluate call --
        a documented approximation: earlier years' much shorter training
        windows (as little as ~24 months, see the README's own Stage 1
        limitations) may not share the same optimal hyperparameters as the
        latest, data-richest window, but re-tuning every year is not
        proportionate. ON by default; pass tune=False (--no-tune) to skip.
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
        if max_target_year is not None:
            years = [y for y in years if y + 1 <= max_target_year]
    log.info("Rolling walk-forward FES: %d feasible years: %s", len(years), years)

    from src.preprocessing import split
    from src.fes_calculator import compute_fes

    tuned_params: dict = {}
    if tune and years:
        tune_as_of_year = years[-1]
        _stage(2, f"One-time hyperparameter tuning (on the latest rolling year, "
                  f"as_of={tune_as_of_year}) -- reused for every rolling year below")
        tune_split_train_end  = f"{tune_as_of_year - 1}-12-01"
        tune_split_test_start = f"{tune_as_of_year}-01-01"
        tune_split_test_end   = f"{tune_as_of_year}-12-01"
        tune_refit_end        = f"{tune_as_of_year}-12-01"
        tune_forecast_dates   = pd.date_range(f"{tune_as_of_year + 1}-01-01", periods=12, freq="MS")

        tune_core_train, tune_core_test = split(
            core_full, tune_split_train_end, tune_split_test_start, tune_split_test_end,
        )
        tune_macro_full, tune_macro_train, _ = _add_electricity_macro_lags(
            core_full, macro_full_raw, tune_split_train_end, tune_split_test_start, tune_split_test_end,
        )
        from src.tuning import tune_models
        try:
            tuned_params, _ = tune_models(
                series, models_to_run,
                tune_core_train, tune_core_test, core_full,
                tune_macro_train, tune_macro_full,
                fast=fast, selection_basis=selection_basis,
                out_dir="outputs_v2/tuning_rolling",
                full_train_end=tune_refit_end,
                forecast_start=tune_forecast_dates.min().strftime("%Y-%m-%d"),
                forecast_end=tune_forecast_dates.max().strftime("%Y-%m-%d"),
            )
            _print_tuning_winners(tuned_params)
        except Exception as e:
            log.error("Rolling one-time tuning failed -- continuing with untuned defaults: %s", e, exc_info=True)
            tuned_params = {}

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
                model_params=tuned_params,
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

        # NOTE: fes_selected is included here too -- a pre-existing gap
        # (found while adding fes_weighted below): _select_best_fes_variant
        # can choose "Equal_Selected" as the walk-forward's winning FES
        # variant (it's a candidate in _compare_fes_variants' benchmark),
        # but this table never actually captured a fes_selected column for
        # src.ukhls_preprocessing.attach_fes_delta to then read -- causing
        # a KeyError downstream whenever Equal_Selected wins. Fixed here
        # since fes_weighted needs the identical capture-list treatment
        # anyway.
        row = {"as_of_year": as_of_year, "target_year": as_of_year + 1}
        for col in ["fes_core", "fes_macro", "fes_selected", "fes_weighted", "fes_actual"]:
            row[col] = float(monthly_df[col].mean()) if col in monthly_df.columns else float("nan")
        rows.append(row)

        # Keep the full 12-month detail too (not just its annual mean) --
        # `monthly_df` already has one row per calendar month of the
        # target year; collapsing straight to a mean here would throw away
        # exactly the resolution src.ukhls_preprocessing.attach_fes_delta
        # needs to give each household-wave row a FES Magnitude specific to
        # its own interview month, not just its interview year.
        month_detail = monthly_df[["date"] + [c for c in ["fes_core", "fes_macro", "fes_selected", "fes_weighted", "fes_actual"] if c in monthly_df.columns]].copy()
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

    winner_col = _select_best_fes_variant(comparison_frames, FES_DIR)
    _save_selected_variant_by_year(comparison_frames, winner_col, FES_DIR)

    _plot_rolling_trend(rolling_df, FIGURES_DIR)

    if model_selection_rows:
        model_selection_df = pd.DataFrame(model_selection_rows)
        model_selection_df.to_csv(f"{FES_DIR}/model_selection_by_year.csv", index=False)

    if performance_rows:
        performance_df = pd.DataFrame(performance_rows)
        performance_df.to_csv(f"{FES_DIR}/forecast_performance_by_year.csv", index=False)
        _plot_forecast_performance_by_year(performance_df, FIGURES_DIR)
        for mode in ["core", "macro"]:
            _plot_rolling_performance_polar(performance_df, mode, FIGURES_DIR)

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

    variant_to_col = {
        "Equal_Core": "fes_core", "Equal_Macro": "fes_macro",
        "Equal_Selected": "fes_selected", "Equal_Weighted": "fes_weighted",
    }
    winner_col = variant_to_col.get(winner_label)
    log.info(
        "FES variant selection (mean RMSE across %d years x benchmarks): %s -- "
        "winner = %s (%s), saved -> %s",
        len(comparison_frames), by_variant[["FES_variant", "mean_rmse"]].to_dict("records"),
        winner_label, winner_col, sel_path,
    )
    return winner_col


def _save_selected_variant_by_year(comparison_frames: list, winner_col: str | None, out_dir: str) -> None:
    """
    Selected FES variant's RMSE/Pearson r by rolling year -- replaces the
    old fes_metrics_{rmse,pearson_r}_heatmap.png (which showed BOTH variants
    x 3 realised-FES benchmarks per year in one grid) with a single clean
    table for just the variant _select_best_fes_variant actually chose,
    averaged across the 3 benchmarks per year (same averaging that
    selection itself already uses, just kept per-year instead of collapsed
    to one overall number).
    """
    if not comparison_frames or winner_col is None:
        return
    variant_labels = {
        "fes_core": "Equal_Core", "fes_macro": "Equal_Macro",
        "fes_selected": "Equal_Selected", "fes_weighted": "Equal_Weighted",
    }
    winner_label = variant_labels.get(winner_col)
    if winner_label is None:
        return

    all_comp = pd.concat(comparison_frames, ignore_index=True)
    sub = all_comp[all_comp["FES_variant"] == winner_label]
    if sub.empty:
        return
    by_year = (
        sub.groupby("as_of_year")
        .agg(RMSE=("RMSE", "mean"), Pearson_r=("Pearson_r", "mean"))
        .reset_index()
    )
    by_year["target_year"] = by_year["as_of_year"] + 1
    by_year["FES_variant"] = winner_label
    by_year = by_year[["as_of_year", "target_year", "FES_variant", "RMSE", "Pearson_r"]].sort_values("as_of_year")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = f"{out_dir}/fes_metrics_selected_variant_by_year.csv"
    by_year.to_csv(out_path, index=False)
    log.info("Selected FES variant (%s) RMSE/Pearson r by year saved -> %s (%d years)",
              winner_label, out_path, len(by_year))


# ══════════════════════════════════════════════════════════════════════════════
# Public run() — callable from main.py
# ══════════════════════════════════════════════════════════════════════════════

def run(
    series: list | None = None,
    models_to_run: list | None = None,
    fast: bool = False,
    fes_only: bool = False,
    tune: bool = True,
    selection_basis: str = "validation",
    target_year: int | None = DEFAULT_TARGET_YEAR,
) -> None:
    """
    Execute Stages 0–4.

    Forecast window: train through target_year-1's December, validate on
    target_year-1, forecast target_year. Defaults to
    src.config.DEFAULT_TARGET_YEAR (2025) rather than fully auto-detecting
    from the raw price data's own latest date -- the UK gas/electricity/
    carbon series get updated independently of (and faster than) the UKHLS
    social-science panel this project is built around (currently 2009-2024
    interview years), so a purely data-driven "one year ahead" default can
    silently race past the year the household stream actually needs (see
    src.config.DEFAULT_TARGET_YEAR's comment for how this was found: raw
    price data reaching 2026-03 pushed the old fully-dynamic default to
    forecast 2026, one year past what the panel needs). Pass target_year
    explicitly to forecast a different year (must still be feasible given
    the data on hand); use --rolling for the full walk-forward backtest
    across every feasible historical year (what the household-panel stream
    actually uses for a genuinely per-interview-year FES signal).

    Parameters
    ----------
    series        : series to forecast (default: gas, electricity, carbon)
    models_to_run : models to train (default: all four)
    fast          : use fewer epochs for LSTM/TFT (development mode)
    fes_only      : skip training; recompute FES from existing forecast CSVs
    tune          : run hyperparameter tuning (src.tuning.tune_models) before
                    final training -- ON by default (was opt-in via --tune;
                    now opt out with --no-tune). Real accuracy lever, adds
                    real runtime -- see src.tuning's own docstring on why the
                    grids are kept compact.
    selection_basis : 'validation' (default) picks the best model/mode per
                    series using only pre-target-year backtest information
                    (RMSE-based rank_score) -- the genuinely ex-ante choice,
                    consistent with this project's anticipatory framing.
                    'forecast_actual' instead picks using the target year's
                    now-known actuals (forecast_actual_MAE) -- a hindsight
                    choice, useful only for retrospective "which model
                    would have been best" reporting on an already-realised
                    year, never for a genuinely future forecast (there are
                    no actuals yet to select on, so it silently falls back
                    to the validation rank in that case anyway -- see
                    model_evaluation.apply_selection_scores).
    target_year   : forecast exactly this year (default: DEFAULT_TARGET_YEAR,
                    2025 -- the most recent year aligned with the UKHLS
                    panel's own coverage). Pass a different year to
                    override for a single run without changing the
                    project-wide default in src.config. Ignored when
                    fes_only=True (that path infers its window from
                    existing forecast CSVs).
    """
    series        = series        or ALL_SERIES
    models_to_run = models_to_run or ALL_MODELS

    Path("outputs_v2/logs").mkdir(parents=True, exist_ok=True)
    Path(FORECAST_DIR).mkdir(parents=True, exist_ok=True)
    Path(FES_DIR).mkdir(parents=True, exist_ok=True)
    Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)
    Path(TABLES_DIR).mkdir(parents=True, exist_ok=True)
    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)

    if fes_only:
        _stage(4, "FES computation — using existing forecast CSVs")
        # _load_ranked_df_from_csv reads back whatever selection_basis/
        # selection_score was baked in by the LAST full run -- re-applying
        # apply_selection_scores here makes a `--fes-only --selection-basis X`
        # call actually take effect instead of silently keeping the old
        # basis (rank_score and forecast_actual_* are already in the saved
        # CSV and don't depend on selection_basis, so this is safe/cheap).
        from src.model_evaluation import apply_selection_scores, save_metrics_table
        ranked_df = _load_ranked_df_from_csv()
        ranked_df = apply_selection_scores(ranked_df, selection_basis)
        save_metrics_table(ranked_df, TABLES_DIR)
        forecast_dates = _infer_forecast_dates_from_csvs(FORECAST_DIR, series)
        refit_end = (forecast_dates.min() - pd.DateOffset(months=1)).strftime("%Y-%m-%d")
        log.info("--fes-only: inferred forecast window %s..%s from existing CSVs (refit_end=%s)",
                  forecast_dates.min().date(), forecast_dates.max().date(), refit_end)
        stage4_compute_fes(
            ranked_df, train_start="2005-01-01", train_end=refit_end,
            forecast_dates=forecast_dates,
        )
        return

    _stage(0, "Loading raw UK data")
    stage0_load_data()

    _stage(1, "Preprocessing (core + macro, full history)")
    core_full, _, _, macro_full_raw, _, _ = stage1_preprocess()

    split_train_end, split_test_start, split_test_end, refit_end, forecast_dates = (
        _compute_default_window(core_full, override_target_year=target_year)
    )
    forecast_start = forecast_dates.min().strftime("%Y-%m-%d")
    forecast_end   = forecast_dates.max().strftime("%Y-%m-%d")
    log.info(
        "Auto-detected forecast window: train<%s, validate=%s..%s, refit<=%s, forecast=%s..%s",
        split_train_end, split_test_start, split_test_end, refit_end, forecast_start, forecast_end,
    )

    from src.preprocessing import split as _split
    core_train, core_test = _split(core_full, split_train_end, split_test_start, split_test_end)
    macro_full, macro_train, _ = _add_electricity_macro_lags(
        core_full, macro_full_raw, split_train_end, split_test_start, split_test_end,
    )

    tuned_params: dict = {}
    if tune:
        _stage(2, "Hyperparameter tuning")
        from src.tuning import tune_models
        tuned_params, _ = tune_models(
            series, models_to_run,
            core_train, core_test, core_full,
            macro_train, macro_full,
            fast=fast, selection_basis=selection_basis,
            out_dir="outputs_v2/tuning",
            full_train_end=refit_end,
            forecast_start=forecast_start, forecast_end=forecast_end,
        )
        _print_tuning_winners(tuned_params)

    _stage(2, "Model training & evaluation (4 models × 2 modes × 3 series)")
    results = stage2_train_evaluate(
        series, models_to_run,
        core_train, core_test, core_full,
        macro_train, macro_full,
        fast=fast, model_params=tuned_params,
        train_end=refit_end, forecast_start=forecast_start, forecast_end=forecast_end,
    )

    _stage(3, "Metrics & model ranking")
    _, ranked_df, best = stage3_evaluation(results, selection_basis)
    _print_best(best)

    _stage(4, "FES construction (equal-weighted)")
    stage4_compute_fes(
        ranked_df, train_start="2005-01-01", train_end=refit_end,
        forecast_dates=forecast_dates,
    )


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
                        help="Hyperparameter tuning before final training -- ON by "
                             "default now, this flag is a harmless no-op kept for "
                             "backward compatibility. Use --no-tune to opt out.")
    parser.add_argument("--no-tune", action="store_true",
                        help="Skip hyperparameter tuning (opt out of the new "
                             "tuning-by-default behaviour). Single-year mode: skips "
                             "tuning entirely. --rolling mode: skips the one-time "
                             "tuning pass (see run_rolling's docstring).")
    parser.add_argument("--selection-basis",
                        choices=["forecast_actual", "validation"],
                        default="validation",
                        help="Model selection criterion. 'validation' (default) is "
                             "the genuine, non-hindsight walk-forward backtest choice. "
                             "'forecast_actual' selects using the target year's "
                             "now-known actuals -- hindsight, for retrospective "
                             "reporting on an already-realised year only.")
    parser.add_argument("--rolling", action="store_true",
                        help="Walk-forward rolling FES instead of the single-year "
                             "path: train through year Y, forecast Y+1, repeat for "
                             "every feasible Y (~n_years x 24 model fits, plus one "
                             "hyperparameter-tuning pass by default -- see "
                             "run_rolling's docstring for the cost). Ignores "
                             "--fes-only.")
    parser.add_argument("--target-year", type=int, default=DEFAULT_TARGET_YEAR, metavar="YYYY",
                        help="Single-year mode only: forecast exactly this year "
                             f"(default: {DEFAULT_TARGET_YEAR}, the most recent "
                             "year aligned with the UKHLS panel's own coverage -- "
                             "see src.config.DEFAULT_TARGET_YEAR). Pass a "
                             "different year to override for this run only, or "
                             "0 to fall back to the fully dynamic "
                             "latest-available-year detection instead.")
    parser.add_argument("--max-target-year", type=int, default=DEFAULT_TARGET_YEAR, metavar="YYYY",
                        help="--rolling only: cap the walk-forward so it stops at "
                             f"this target year (default: {DEFAULT_TARGET_YEAR}, "
                             "see src.config.DEFAULT_TARGET_YEAR) instead of "
                             "extending into a still-partial or panel-uncovered "
                             "year. Pass 0 to remove the cap entirely.")
    args = parser.parse_args()
    if args.max_target_year == 0:
        args.max_target_year = None   # explicit opt-out of the default cap
    if args.target_year == 0:
        args.target_year = None       # explicit opt-in to full dynamic detection
    tune = not args.no_tune           # tuning is ON by default; --no-tune opts out

    models_to_run = [m for m in ALL_MODELS if m not in args.skip_models]

    _banner("Anticipatory Energy–Carbon Stress — Forecast Pipeline (Stages 0–4)")
    print(f"  Series      : {args.series}")
    print(f"  Models      : {models_to_run}")
    print(f"  Fast mode   : {args.fast}")
    print(f"  Rolling     : {args.rolling}")
    if args.rolling:
        print(f"  Max target year : {args.max_target_year or '(uncapped)'}")
        print(f"  Tuning      : {tune} (one-time pass, reused across all rolling years)")
    else:
        print(f"  FES only    : {args.fes_only}")
        print(f"  Tuning      : {tune}")
        print(f"  Selection   : {args.selection_basis}")
        print(f"  Target year : {args.target_year or '(dynamic)'}")

    t0 = time.time()
    if args.rolling:
        run_rolling(
            series=args.series,
            models_to_run=models_to_run,
            fast=args.fast,
            selection_basis=args.selection_basis,
            max_target_year=args.max_target_year,
            tune=tune,
        )
    else:
        run(
            series=args.series,
            models_to_run=models_to_run,
            fast=args.fast,
            fes_only=args.fes_only,
            tune=tune,
            selection_basis=args.selection_basis,
            target_year=args.target_year,
        )
    _banner(f"Forecast pipeline complete  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
