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
  --step6-only    : skip model training and jump straight to Step 6
"""

from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

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
    return core_full, core_train, core_test, macro_full, macro_train, macro_test


# ══════════════════════════════════════════════════════════════════════════════
# Stage 2 — Model training and evaluation
# ══════════════════════════════════════════════════════════════════════════════

def _get_series(series_name: str, core_train, core_test, core_full):
    """
    Extract per-series splits.

    Returns
    -------
    train_s      : training series (up to end of 2016)
    test_s       : 2017 test series
    full_s       : history through 2017 only — used for the final refit
                   (2018 rows excluded to avoid leakage)
    actual_2018  : actual 2018 values for comparison column in forecast CSVs
                   (may contain NaN if source data does not reach 2018)
    """
    col = f"{series_name}_growth"
    train_s = core_train[col].dropna()
    test_s  = core_test[col].dropna()

    # Restrict full to history through 2017 (refit before forecasting 2018)
    full_s = core_full.loc[:"2017-12-01", col].dropna()

    # Extract actual 2018 values for CSV comparison column
    actual_2018 = core_full.loc["2018-01-01":"2018-12-01", col]

    log.info(f"[{series_name}] train={len(train_s)}, test={len(test_s)}, "
             f"full={len(full_s)}, actual_2018={actual_2018.notna().sum()} non-null")
    return train_s, test_s, full_s, actual_2018


def stage2_train_evaluate(
    series_names: list,
    models_to_run: list,
    core_train,
    core_test,
    core_full,
    macro_train,
    macro_full,
    fast: bool = False,
) -> list:
    """
    Run all model–series combinations in BOTH core and macro modes.

    Each model is called twice per series:
      1. use_macro=False  → core-only forecast
      2. use_macro=True   → macro-augmented forecast with actual 2018 macro values

    Returns list of result dicts (one per model × series × mode).
    """
    results: list = []
    epochs_lstm = 30  if fast else 100
    epochs_tft  = 20  if fast else 80

    from src.model_evaluation import merge_forecast_files

    for series in series_names:
        log.info(f"\n{'─'*50}")
        log.info(f"Series: {series.upper()}")
        log.info(f"{'─'*50}")

        train_s, test_s, full_s, actual_2018 = _get_series(
            series, core_train, core_test, core_full
        )

        # ── SARIMA (core + macro) ─────────────────────────────────────────────
        if "SARIMA" in models_to_run:
            for use_macro in [False, True]:
                try:
                    from src.models.sarima_model import run_sarima
                    r = run_sarima(
                        series, train_s, test_s, full_s,
                        macro_train=macro_train,
                        macro_full=macro_full,
                        use_macro=use_macro,
                        actual_2018=actual_2018,
                    )
                    results.append(r)
                except Exception as e:
                    mode = "macro" if use_macro else "core"
                    log.error(f"SARIMA-{mode} failed for {series}: {e}", exc_info=True)

        # ── Prophet (core + macro) ────────────────────────────────────────────
        if "Prophet" in models_to_run:
            for use_regressors in [False, True]:
                try:
                    from src.models.prophet_model import run_prophet
                    r = run_prophet(
                        series, train_s, test_s, full_s,
                        macro_train=macro_train,
                        macro_full=macro_full,
                        use_regressors=use_regressors,
                        actual_2018=actual_2018,
                    )
                    results.append(r)
                except Exception as e:
                    mode = "macro" if use_regressors else "core"
                    log.error(f"Prophet-{mode} failed for {series}: {e}", exc_info=True)

        # ── LSTM (core + macro) ───────────────────────────────────────────────
        if "LSTM" in models_to_run:
            for use_macro in [False, True]:
                try:
                    from src.models.lstm_model import run_lstm
                    r = run_lstm(
                        series, train_s, test_s, full_s,
                        macro_train=macro_train,
                        macro_full=macro_full,
                        use_macro=use_macro,
                        actual_2018=actual_2018,
                        epochs=epochs_lstm,
                    )
                    results.append(r)
                except Exception as e:
                    mode = "macro" if use_macro else "core"
                    log.error(f"LSTM-{mode} failed for {series}: {e}", exc_info=True)

        # ── TFT (core + macro) ────────────────────────────────────────────────
        if "TFT" in models_to_run:
            for use_macro in [False, True]:
                try:
                    from src.models.tft_model import run_tft
                    r = run_tft(
                        series, train_s, test_s, full_s,
                        macro_train=macro_train,
                        macro_full=macro_full,
                        use_macro=use_macro,
                        actual_2018=actual_2018,
                        epochs=epochs_tft,
                    )
                    results.append(r)
                except Exception as e:
                    mode = "macro" if use_macro else "core"
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

def stage3_evaluation(results: list):
    from src.model_evaluation import run_evaluation
    metrics_df, ranked_df, best = run_evaluation(results, out_dir="outputs/tables")
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
    args = parser.parse_args()

    ALL_MODELS   = ["SARIMA", "Prophet", "LSTM", "TFT"]
    models_to_run = [m for m in ALL_MODELS if m not in args.skip_models]

    _banner("Anticipatory Energy–Carbon Stress Index Pipeline")
    print(f"  Series       : {args.series}")
    print(f"  Models       : {models_to_run}")
    print(f"  Fast mode    : {args.fast}")
    print(f"  Step-6 only  : {args.step6_only}")

    t0 = time.time()

    if not args.step6_only:
        _stage(0, "Loading raw UK data")
        stage0_load_data()

        _stage(1, "Preprocessing")
        (core_full, core_train, core_test,
         macro_full, macro_train, macro_test) = stage1_preprocess()

        _stage(2, "Model training & evaluation (core + macro modes)")
        results = stage2_train_evaluate(
            args.series, models_to_run,
            core_train, core_test, core_full,
            macro_train, macro_full,
            fast=args.fast,
        )

        _stage(3, "Metrics & ranking")
        metrics_df, ranked_df, best = stage3_evaluation(results)

        print("\n── Best models (2017 evaluation, rank-aggregation) ──")
        _print_best(best)

        _stage(4, "FES computation (core / macro / actual)")
        from src.fes_calculator import compute_fes
        compute_fes(
            ranked_df=ranked_df,
            core_csv="data/raw/core_energy_carbon.csv",
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


if __name__ == "__main__":
    main()
