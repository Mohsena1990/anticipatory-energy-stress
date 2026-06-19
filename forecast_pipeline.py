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
  Stage 4 : Compute FES_core / FES_macro / FES_actual; save CSVs + figures

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


def _add_electricity_macro_lags(core_full: pd.DataFrame, macro_full: pd.DataFrame):
    """Add 12-month electricity lag to macro features (avoids target leakage)."""
    macro_full = macro_full.copy()
    if "electricity_growth" in core_full.columns:
        lag12 = core_full["electricity_growth"].shift(12)
        macro_full["electricity_growth_lag12"] = (
            lag12.reindex(macro_full.index).ffill().fillna(0.0)
        )
        macro_full.to_csv("data/processed/macro_processed.csv")
    macro_train = macro_full.loc[: "2015-12-01"].copy()
    macro_test  = macro_full.loc["2016-01-01":"2016-12-01"].copy()
    return macro_full, macro_train, macro_test


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2 — Model training and evaluation
# ══════════════════════════════════════════════════════════════════════════════

def _get_series(series_name: str, core_train, core_test, core_full):
    """
    Extract per-series train/test/full splits plus actual 2017 values.

    Returns
    -------
    train_s     : training period (2005-2015)
    test_s      : validation period (2016)
    full_s      : full history through 2016 for final refit
    actual_2017 : realised 2017 values (comparison column in forecast CSVs)
    eval_actual : 2016 actuals used for validation metrics
    """
    col = f"{series_name}_growth"
    train_s     = core_train[col].dropna()
    test_s      = core_test[col].dropna()
    eval_actual = core_test[col].dropna()
    full_s      = core_full.loc[:"2016-12-01", col].dropna()
    actual_2017 = core_full.loc["2017-01-01":"2017-12-01", col]

    log.info(
        "[%s] train=%d, test=%d, full=%d, actual_2017=%d non-null",
        series_name, len(train_s), len(test_s), len(full_s),
        actual_2017.notna().sum(),
    )
    return train_s, test_s, full_s, actual_2017, eval_actual


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
) -> list:
    """
    Train all models in both core and macro modes for every series.

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
            series, core_train, core_test, core_full
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
                    results.append(run_lstm(
                        series, train_s, test_s, full_s,
                        macro_train=macro_train, macro_full=macro_full,
                        use_macro=use_macro, actual_2017=actual_2017,
                        eval_actual=eval_actual, epochs=epochs_lstm,
                        **model_params.get((series, mode, "LSTM"), {}),
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
                        **model_params.get((series, mode, "TFT"), {}),
                    ))
                except Exception as e:
                    log.error("TFT-%s failed for %s: %s", mode, series, e, exc_info=True)

    merge_forecast_files(series_names, models_to_run, FORECAST_DIR, FORECAST_DIR)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Stage 3 — Evaluation + model ranking
# ══════════════════════════════════════════════════════════════════════════════

def stage3_evaluation(results: list, selection_basis: str = "forecast_actual"):
    from src.model_evaluation import run_evaluation
    return run_evaluation(
        results,
        out_dir=TABLES_DIR,
        forecast_dir=FORECAST_DIR,
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

    _stage(4, "FES computation (core / macro / actual)")
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
    args = parser.parse_args()

    models_to_run = [m for m in ALL_MODELS if m not in args.skip_models]

    _banner("Anticipatory Energy–Carbon Stress — Forecast Pipeline (Stages 0–4)")
    print(f"  Series      : {args.series}")
    print(f"  Models      : {models_to_run}")
    print(f"  Fast mode   : {args.fast}")
    print(f"  FES only    : {args.fes_only}")
    print(f"  Tuning      : {args.tune}")
    print(f"  Selection   : {args.selection_basis}")

    t0 = time.time()
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
