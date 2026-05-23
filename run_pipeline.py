"""
run_pipeline.py
───────────────
Master orchestrator for the Anticipatory Energy–Carbon Stress pipeline.

Execution order
───────────────
  Stage 0 : Load and parse real UK data files from data/raw/
  Stage 1 : Preprocess Dataset A (core) and Dataset B (macro)
  Stage 2 : Train and evaluate all models in TWO modes per series:
              core  — target series only
              macro — target series + macro exogenous variables
            Each mode produces its own forecast CSV and metrics row.
  Stage 3 : Merge per-model forecast files; save combined metrics table

Usage
─────
  python run_pipeline.py [--fast] [--skip-models MODEL [MODEL ...]]
  python run_pipeline.py --series gas electricity carbon
  python run_pipeline.py --step6-only

  --fast          : fewer epochs / simpler settings (development mode)
  --skip-models   : skip one or more models (e.g. --skip-models LSTM TFT)
  --series        : run only specific series (default: gas electricity carbon)
  --tune          : run compact hyperparameter tuning before final training
  --step6-only    : skip model training and jump straight to Step 6
"""

from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd

from src.logging_utils import setup_logger, get_logger
setup_logger("energy_stress", log_file="outputs/logs/pipeline.log")
log = get_logger("pipeline")


def _banner(text: str) -> None:
    print("\n" + "═" * 70)
    print(f"  {text}")
    print("═" * 70)


def _stage(n: int, label: str) -> None:
    print(f"\n[STAGE {n}] {label}")
    log.info(f"{'='*40}")
    log.info(f"STAGE {n}: {label}")
    log.info(f"{'='*40}")


# ══════════════════════════════════════════════════════════════════════════════
# Stage 0 — Load real data
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
    """Add known-safe electricity target lag features for macro models."""
    macro_full = macro_full.copy()
    if "electricity_growth" in core_full.columns:
        lag12 = core_full["electricity_growth"].shift(12)
        macro_full["electricity_growth_lag12"] = (
            lag12.reindex(macro_full.index).ffill().fillna(0.0)
        )
        macro_full.to_csv("data/processed/macro_processed.csv")
    macro_train = macro_full.loc[: "2015-12-01"].copy()
    macro_test = macro_full.loc["2016-01-01":"2016-12-01"].copy()
    return macro_full, macro_train, macro_test


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2 — Model training and evaluation
# ══════════════════════════════════════════════════════════════════════════════

def _get_series(series_name: str, core_train, core_test, core_full):
    """
    Extract per-series splits.

    Returns
    -------
    train_s      : training series (up to end of 2015)
    test_s       : 2016 test series
    full_s       : history through 2016 only — used for the final refit
                   (2017 rows excluded to avoid leakage)
    actual_2017  : actual 2017 values for comparison column in forecast CSVs
                   (may contain NaN if source data does not reach 2017)
    """
    col = f"{series_name}_growth"
    model_col = col
    train_s = core_train[model_col].dropna()
    test_s  = core_test[model_col].dropna()
    eval_actual = core_test[col].dropna()

    # Restrict full to history through 2016 (refit before forecasting 2017)
    full_s = core_full.loc[:"2016-12-01", model_col].dropna()

    # Extract actual 2017 values for CSV comparison column
    actual_2017 = core_full.loc["2017-01-01":"2017-12-01", col]

    log.info(f"[{series_name}] train={len(train_s)}, test={len(test_s)}, "
             f"full={len(full_s)}, actual_2017={actual_2017.notna().sum()} non-null")
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
    Run all model–series combinations in BOTH core and macro modes.

    Each model is called twice per series:
      1. use_macro=False  → core-only forecast
      2. use_macro=True   → macro-augmented forecast with lagged series-specific macro values

    Returns list of result dicts (one per model × series × mode).
    """
    results: list = []
    model_params = model_params or {}
    epochs_lstm = 30  if fast else 100
    epochs_tft  = 20  if fast else 80

    from src.model_evaluation import merge_forecast_files

    for series in series_names:
        log.info(f"\n{'─'*50}")
        log.info(f"Series: {series.upper()}")
        log.info(f"{'─'*50}")

        train_s, test_s, full_s, actual_2017, eval_actual = _get_series(
            series, core_train, core_test, core_full
        )

        # ── SARIMA (core + macro) ─────────────────────────────────────────────
        if "SARIMA" in models_to_run:
            for use_macro in [False, True]:
                mode = "macro" if use_macro else "core"
                try:
                    from src.models.sarima_model import run_sarima
                    r = run_sarima(
                        series, train_s, test_s, full_s,
                        macro_train=macro_train,
                        macro_full=macro_full,
                        use_macro=use_macro,
                        actual_2017=actual_2017,
                        eval_actual=eval_actual,
                        **model_params.get((series, mode, "SARIMA"), {}),
                    )
                    results.append(r)
                except Exception as e:
                    log.error(f"SARIMA-{mode} failed for {series}: {e}", exc_info=True)

        # ── Prophet (core + macro) ────────────────────────────────────────────
        if "Prophet" in models_to_run:
            for use_regressors in [False, True]:
                mode = "macro" if use_regressors else "core"
                try:
                    from src.models.prophet_model import run_prophet
                    r = run_prophet(
                        series, train_s, test_s, full_s,
                        macro_train=macro_train,
                        macro_full=macro_full,
                        use_regressors=use_regressors,
                        actual_2017=actual_2017,
                        eval_actual=eval_actual,
                        **model_params.get((series, mode, "Prophet"), {}),
                    )
                    results.append(r)
                except Exception as e:
                    log.error(f"Prophet-{mode} failed for {series}: {e}", exc_info=True)

        # ── LSTM (core + macro) ───────────────────────────────────────────────
        if "LSTM" in models_to_run:
            for use_macro in [False, True]:
                mode = "macro" if use_macro else "core"
                try:
                    from src.models.lstm_model import run_lstm
                    r = run_lstm(
                        series, train_s, test_s, full_s,
                        macro_train=macro_train,
                        macro_full=macro_full,
                        use_macro=use_macro,
                        actual_2017=actual_2017,
                        eval_actual=eval_actual,
                        epochs=epochs_lstm,
                        **model_params.get((series, mode, "LSTM"), {}),
                    )
                    results.append(r)
                except Exception as e:
                    log.error(f"LSTM-{mode} failed for {series}: {e}", exc_info=True)

        # ── TFT (core + macro) ────────────────────────────────────────────────
        if "TFT" in models_to_run:
            for use_macro in [False, True]:
                mode = "macro" if use_macro else "core"
                try:
                    from src.models.tft_model import run_tft
                    r = run_tft(
                        series, train_s, test_s, full_s,
                        macro_train=macro_train,
                        macro_full=macro_full,
                        use_macro=use_macro,
                        actual_2017=actual_2017,
                        eval_actual=eval_actual,
                        epochs=epochs_tft,
                        **model_params.get((series, mode, "TFT"), {}),
                    )
                    results.append(r)
                except Exception as e:
                    log.error(f"TFT-{mode} failed for {series}: {e}", exc_info=True)

    # Merge per-model-mode files into combined per-series files
    merge_forecast_files(
        series_names, models_to_run,
        "outputs/forecasts", "outputs/forecasts",
    )

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Stage 3 — Evaluation + metrics table
# ══════════════════════════════════════════════════════════════════════════════

def stage3_evaluation(results: list, selection_basis: str):
    from src.model_evaluation import run_evaluation
    metrics_df, ranked_df, best = run_evaluation(
        results,
        out_dir="outputs/tables",
        forecast_dir="outputs/forecasts",
        selection_basis=selection_basis,
    )
    return metrics_df, ranked_df, best


# ══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Anticipatory Energy–Carbon Stress pipeline")
    parser.add_argument("--fast",         action="store_true",
                        help="Fewer epochs / simpler settings for development")
    parser.add_argument("--skip-models",  nargs="+", default=[],
                        metavar="MODEL",
                        help="Models to skip: SARIMA Prophet LSTM TFT")
    parser.add_argument("--series",       nargs="+",
                        default=["gas", "electricity", "carbon"],
                        help="Series to process (default: gas electricity carbon)")
    parser.add_argument("--step6-only",   action="store_true",
                        help="Skip training; jump to Step 6 (requires populated forecasts/)")
    parser.add_argument("--tune",         action="store_true",
                        help="Run compact hyperparameter tuning before final training")
    parser.add_argument("--selection-basis",
                        choices=["forecast_actual", "validation"],
                        default="forecast_actual",
                        help=("Model selection basis. 'forecast_actual' selects by saved "
                              "2017 forecast-vs-actual MAE when actuals exist; "
                              "'validation' selects by 2016 validation ranks."))
    args = parser.parse_args()

    ALL_MODELS   = ["SARIMA", "Prophet", "LSTM", "TFT"]
    models_to_run = [m for m in ALL_MODELS if m not in args.skip_models]

    _banner("Anticipatory Energy–Carbon Stress Index Pipeline")
    print(f"  Series       : {args.series}")
    print(f"  Models       : {models_to_run}")
    print(f"  Fast mode    : {args.fast}")
    print(f"  Step-6 only  : {args.step6_only}")
    print(f"  Tuning       : {args.tune}")
    print(f"  Selection    : {args.selection_basis}")

    t0 = time.time()

    if not args.step6_only:
        _stage(0, "Loading raw UK data")
        stage0_load_data()

        _stage(1, "Preprocessing")
        (core_full, core_train, core_test,
         macro_full, macro_train, macro_test) = stage1_preprocess()

        tuned_params = {}
        if args.tune:
            _stage(2, "Hyperparameter tuning")
            from src.tuning import tune_models
            tuned_params, tuning_df = tune_models(
                args.series,
                models_to_run,
                core_train,
                core_test,
                core_full,
                macro_train,
                macro_full,
                fast=args.fast,
                selection_basis=args.selection_basis,
                out_dir="outputs/tuning",
            )
            print("\n── Tuning winners ──")
            _print_tuning_winners(tuned_params)

        _stage(2, "Model training & evaluation (core + macro modes)")
        results = stage2_train_evaluate(
            args.series, models_to_run,
            core_train, core_test, core_full,
            macro_train, macro_full,
            fast=args.fast,
            model_params=tuned_params,
        )

        _stage(3, "Metrics & ranking")
        metrics_df, ranked_df, best = stage3_evaluation(
            results,
            selection_basis=args.selection_basis,
        )

        print(f"\n── Best models ({args.selection_basis}) ──")
        _print_best(best)

        _stage(4, "FES computation (core / macro / actual)")
        from src.fes_calculator import compute_fes
        compute_fes(
            ranked_df=ranked_df,
            core_csv="data/processed/core_energy_carbon.csv",
            forecast_dir="outputs/forecasts",
            out_dir="outputs/fes",
            figures_dir="outputs/figures",
        )

    elapsed = time.time() - t0
    _banner(f"Pipeline complete  ({elapsed:.1f}s)")


def _print_best(best: dict) -> None:
    for key, winner in best.items():
        series, mode = key if isinstance(key, tuple) else (key, "core")
        print(f"  {series:15s} [{mode:5s}] → {winner}")


def _print_tuning_winners(tuned_params: dict) -> None:
    if not tuned_params:
        print("  No tuned parameters were selected.")
        return
    for (series, mode, model), params in sorted(tuned_params.items()):
        print(f"  {series:15s} [{mode:5s}] {model:8s} → {params}")




if __name__ == "__main__":
    main()
