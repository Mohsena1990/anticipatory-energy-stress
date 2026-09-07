"""
tuning.py
─────────
Compact hyperparameter tuning for the forecasting pipeline.

The search is intentionally small. With monthly data and short validation
windows, broad grids are more likely to overfit noise than improve the study.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import config
from src.logging_utils import get_logger

log = get_logger("tuning")


def _forecast_actual_mae(forecast_path: str | None) -> float:
    if not forecast_path:
        return float("nan")
    try:
        df = pd.read_csv(forecast_path)
    except FileNotFoundError:
        return float("nan")
    if not {"forecast", "actual"}.issubset(df.columns):
        return float("nan")
    comp = df[["forecast", "actual"]].dropna()
    if comp.empty:
        return float("nan")
    return float(np.mean(np.abs(comp["forecast"].to_numpy() - comp["actual"].to_numpy())))


def _score_result(result: dict, selection_basis: str) -> tuple[float, str]:
    if selection_basis == "forecast_actual":
        actual_mae = _forecast_actual_mae(result.get("forecast_path"))
        if np.isfinite(actual_mae):
            return actual_mae, "forecast_actual_MAE"
    return float(result["metrics"].get("MAE", np.inf)), "validation_MAE"


def _prophet_grid(series: str, run_mode: str, fast: bool) -> list[dict[str, Any]]:
    modes = ["additive"]
    if series == "gas":
        modes = ["additive", "multiplicative"]
    cps_values = [0.01, 0.05] if fast else [0.001, 0.01, 0.05, 0.1]
    sps_values = [1.0, 10.0] if not fast else [10.0]
    # regressor_prior_scale only affects macro mode (core has no
    # regressors) -- varying it for core would just duplicate identical
    # candidates. Includes Prophet's old implicit default (~10, effectively
    # unregularised) alongside tighter values, so tuning can confirm
    # whether the new 0.5 default actually beats the old behaviour rather
    # than just asserting it.
    reg_ps_values = [0.1, 0.5, 2.0, 10.0] if run_mode == "macro" else [None]
    return [
        {
            "changepoint_prior_scale": cps,
            "seasonality_prior_scale": sps,
            "seasonality_mode": mode,
            **({"regressor_prior_scale": reg_ps} if reg_ps is not None else {}),
        }
        for cps, sps, mode, reg_ps in product(cps_values, sps_values, modes, reg_ps_values)
    ]


def _lstm_grid(series: str, mode: str, fast: bool) -> list[dict[str, Any]]:
    if series == "electricity" and mode == "macro":
        lean = {"macro_feature_set": "lean"}
        demand = {"macro_feature_set": "demand"}
        if fast:
            return [
                {"lookback": 3, "dropout": 0.10, "learning_rate": 1e-3, "units_1": 64, "units_2": 32, **lean},
                {"lookback": 6, "dropout": 0.10, "learning_rate": 1e-3, "units_1": 64, "units_2": 32, **lean},
                {"lookback": 9, "dropout": 0.20, "learning_rate": 1e-3, "units_1": 96, "units_2": 48, **lean},
                {"lookback": 12, "dropout": 0.20, "learning_rate": 1e-3, "units_1": 128, "units_2": 64, **lean},
                {"lookback": 18, "dropout": 0.35, "learning_rate": 5e-4, "units_1": 128, "units_2": 64, **lean},
                {"lookback": 9, "dropout": 0.20, "learning_rate": 1e-3, "units_1": 96, "units_2": 48, **demand},
                {"lookback": 12, "dropout": 0.20, "learning_rate": 1e-3, "units_1": 128, "units_2": 64, **demand},
            ]
        return [
            {"lookback": 3, "dropout": 0.10, "learning_rate": 1e-3, "units_1": 64, "units_2": 32, **lean},
            {"lookback": 6, "dropout": 0.10, "learning_rate": 1e-3, "units_1": 64, "units_2": 32, **lean},
            {"lookback": 6, "dropout": 0.25, "learning_rate": 5e-4, "units_1": 96, "units_2": 48, **lean},
            {"lookback": 9, "dropout": 0.20, "learning_rate": 1e-3, "units_1": 96, "units_2": 48, **lean},
            {"lookback": 12, "dropout": 0.20, "learning_rate": 1e-3, "units_1": 128, "units_2": 64, **lean},
            {"lookback": 12, "dropout": 0.35, "learning_rate": 5e-4, "units_1": 128, "units_2": 64, **lean},
            {"lookback": 18, "dropout": 0.25, "learning_rate": 5e-4, "units_1": 128, "units_2": 64, **lean},
            {"lookback": 24, "dropout": 0.35, "learning_rate": 5e-4, "units_1": 96, "units_2": 48, **lean},
            {"lookback": 9, "dropout": 0.20, "learning_rate": 1e-3, "units_1": 96, "units_2": 48, **demand},
            {"lookback": 12, "dropout": 0.20, "learning_rate": 1e-3, "units_1": 128, "units_2": 64, **demand},
            {"lookback": 18, "dropout": 0.25, "learning_rate": 5e-4, "units_1": 128, "units_2": 64, **demand},
        ]

    if fast:
        return [
            {"lookback": 6, "dropout": 0.2, "learning_rate": 1e-3, "units_1": 64, "units_2": 32, "macro_feature_set": "lstm"},
            {"lookback": 12, "dropout": 0.2, "learning_rate": 1e-3, "units_1": 128, "units_2": 64, "macro_feature_set": "lstm"},
        ]
    return [
        {"lookback": 6, "dropout": 0.1, "learning_rate": 1e-3, "units_1": 64, "units_2": 32, "macro_feature_set": "lstm"},
        {"lookback": 12, "dropout": 0.2, "learning_rate": 1e-3, "units_1": 128, "units_2": 64, "macro_feature_set": "lstm"},
        {"lookback": 18, "dropout": 0.25, "learning_rate": 5e-4, "units_1": 128, "units_2": 64, "macro_feature_set": "lstm"},
    ]


def _tft_grid(fast: bool) -> list[dict[str, Any]]:
    if fast:
        return [
            {"lookback": 6, "hidden": 32, "n_heads": 4, "learning_rate": 1e-3},
            {"lookback": 12, "hidden": 64, "n_heads": 4, "learning_rate": 5e-4},
        ]
    return [
        {"lookback": 6, "hidden": 32, "n_heads": 4, "learning_rate": 1e-3},
        {"lookback": 12, "hidden": 64, "n_heads": 4, "learning_rate": 5e-4},
        {"lookback": 18, "hidden": 64, "n_heads": 4, "learning_rate": 5e-4},
    ]


def _candidate_grid(model: str, series: str, mode: str, fast: bool) -> list[dict[str, Any]]:
    if model == "Prophet":
        return _prophet_grid(series, mode, fast)
    if model == "LSTM":
        # Previously disabled entirely: tuning trained candidates for only
        # 60 epochs (20 in --fast) while the final production model trains
        # for 100 (30 fast) -- a real mismatch, since more epochs can shift
        # which dropout/lookback/lr combination actually generalises best,
        # so a candidate picked under 60-epoch training was never a
        # trustworthy predictor of 100-epoch behaviour. Re-enabled now that
        # tune_models trains candidates for the SAME epoch count as the
        # final model (see epochs_lstm below) -- the grid itself
        # (_lstm_grid) was already fully defined and correct, just unused.
        return _lstm_grid(series, mode, fast)
    if model == "TFT":
        return _tft_grid(fast)
    return [{}]


def tune_models(
    series_names: list[str],
    models_to_run: list[str],
    core_train: pd.DataFrame,
    core_test: pd.DataFrame,
    core_full: pd.DataFrame,
    macro_train: pd.DataFrame,
    macro_full: pd.DataFrame,
    fast: bool = False,
    selection_basis: str = "validation",
    out_dir: str = "outputs/tuning",
    full_train_end: str = "2016-12-01",
    forecast_start: str = "2017-01-01",
    forecast_end: str = "2017-12-01",
) -> tuple[dict[tuple[str, str, str], dict[str, Any]], pd.DataFrame]:
    """
    Tune model hyperparameters and return best params keyed by
    ``(series, mode, model)``.

    full_train_end/forecast_start/forecast_end default to the original
    single-year window; forecast_pipeline.run() passes the dynamically
    detected "latest year + 1" window instead so tuning is scored against
    whichever period is actually being forecast, not a fixed calendar year.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    forecast_root = Path(out_dir) / "forecasts"
    forecast_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    best_params: dict[tuple[str, str, str], dict[str, Any]] = {}

    # Matches the FINAL production epoch count exactly (config.LSTM_EPOCHS*)
    # -- previously 20/60 here vs. 30/100 in production, a mismatch that
    # made tuned LSTM hyperparameters unreliable (see _candidate_grid).
    epochs_lstm = config.LSTM_EPOCHS_FAST if fast else config.LSTM_EPOCHS
    epochs_tft = 15 if fast else 50
    mc_samples = 50 if fast else 100

    for series in series_names:
        col = f"{series}_growth"
        train_s = core_train[col].dropna()
        test_s = core_test[col].dropna()
        full_s = core_full.loc[:full_train_end, col].dropna()
        actual_target = core_full.loc[forecast_start:forecast_end, col]
        eval_actual = core_test[col].dropna()

        for mode in ["core", "macro"]:
            use_macro = mode == "macro"
            for model_name in models_to_run:
                grid = _candidate_grid(model_name, series, mode, fast)
                best_score = float("inf")
                best_row_params: dict[str, Any] = {}

                for i, params in enumerate(grid, start=1):
                    candidate_dir = forecast_root / series / mode / model_name.lower() / f"candidate_{i:02d}"
                    candidate_dir.mkdir(parents=True, exist_ok=True)
                    log.info(
                        f"Tuning {series}/{mode}/{model_name} candidate {i}/{len(grid)}: {params}"
                    )
                    try:
                        if model_name == "SARIMA":
                            from src.models.sarima_model import run_sarima

                            result = run_sarima(
                                series, train_s, test_s, full_s,
                                macro_train=macro_train,
                                macro_full=macro_full,
                                use_macro=use_macro,
                                actual_target=actual_target,
                                eval_actual=eval_actual,
                                forecast_dir=str(candidate_dir),
                            )
                        elif model_name == "Prophet":
                            from src.models.prophet_model import run_prophet

                            result = run_prophet(
                                series, train_s, test_s, full_s,
                                macro_train=macro_train,
                                macro_full=macro_full,
                                use_regressors=use_macro,
                                actual_target=actual_target,
                                eval_actual=eval_actual,
                                forecast_dir=str(candidate_dir),
                                **params,
                            )
                        elif model_name == "LSTM":
                            from src.models.lstm_model import run_lstm

                            result = run_lstm(
                                series, train_s, test_s, full_s,
                                macro_train=macro_train,
                                macro_full=macro_full,
                                use_macro=use_macro,
                                actual_target=actual_target,
                                eval_actual=eval_actual,
                                forecast_dir=str(candidate_dir),
                                epochs=epochs_lstm,
                                mc_samples=mc_samples,
                                **params,
                            )
                        elif model_name == "TFT":
                            from src.models.tft_model import run_tft

                            result = run_tft(
                                series, train_s, test_s, full_s,
                                macro_train=macro_train,
                                macro_full=macro_full,
                                use_macro=use_macro,
                                actual_target=actual_target,
                                eval_actual=eval_actual,
                                forecast_dir=str(candidate_dir),
                                epochs=epochs_tft,
                                **params,
                            )
                        else:
                            continue
                    except Exception as exc:
                        log.error(
                            f"Tuning failed for {series}/{mode}/{model_name} {params}: {exc}",
                            exc_info=True,
                        )
                        rows.append({
                            "series_name": series,
                            "mode": mode,
                            "model": model_name,
                            "candidate": i,
                            "status": "failed",
                            "error": str(exc),
                            **params,
                        })
                        continue

                    score, score_basis = _score_result(result, selection_basis)
                    row = {
                        "series_name": series,
                        "mode": mode,
                        "model": model_name,
                        "candidate": i,
                        "status": "ok",
                        "score": score,
                        "score_basis": score_basis,
                        "validation_MAE": result["metrics"].get("MAE"),
                        "validation_RMSE": result["metrics"].get("RMSE"),
                        "forecast_actual_MAE": _forecast_actual_mae(result.get("forecast_path")),
                        **params,
                    }
                    rows.append(row)

                    if score < best_score:
                        best_score = score
                        best_row_params = params.copy()

                if best_row_params:
                    best_params[(series, mode, model_name)] = best_row_params
                    log.info(
                        f"Best params for {series}/{mode}/{model_name}: "
                        f"{best_row_params} ({selection_basis} score={best_score:.4f})"
                    )

    tuning_df = pd.DataFrame(rows)
    tuning_path = Path(out_dir) / "model_tuning_results.csv"
    tuning_df.to_csv(tuning_path, index=False)
    log.info(f"Tuning results saved → {tuning_path}")
    return best_params, tuning_df
