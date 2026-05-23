"""
sarima_model.py
───────────────
SARIMA (core) and SARIMAX (macro) forecasting for a single time series.

Strategy
────────
  1. Use pmdarima.auto_arima to find the best (p,d,q)(P,D,Q,12) orders on
     the training window (evaluated without exogenous variables to keep order
     selection stable across both modes).
  2. Core mode  : SARIMA — no exogenous variables.
  3. Macro mode : SARIMAX with lagged, series-specific exogenous regressors.
     Gas uses lagged gas-futures signals; electricity uses lagged fuel-cost
     and electricity-growth signals.
  4. Evaluate on 2016 test set; refit on full history through 2016; forecast 2017.
  5. Save CSV with actual 2017 values for comparison.
     Filename: {series}_growth_pct_forecasts_sarima_core.csv
               {series}_growth_pct_forecasts_sarima_macro.csv
"""

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple

from src.logging_utils import get_logger
from src.metrics_utils import compute_all_metrics
from src.model_utils import (
    align_macro_for_series,
    index_forecast_to_yoy_growth,
    is_electricity,
    macro_cols_for_series,
)

log = get_logger("sarima")
warnings.filterwarnings("ignore")

FORECAST_PERIODS = 12
ALPHA            = 0.05
def _fit_auto(train_series: pd.Series) -> "pmdarima.ARIMA":
    """Auto-select SARIMA orders on training data (no exog — keeps order selection stable)."""
    import pmdarima as pm
    model = pm.auto_arima(
        train_series,
        start_p=0, start_q=0,
        max_p=3,   max_q=3,
        m=12,
        d=None, D=1,
        seasonal=True,
        information_criterion="aic",
        stepwise=True,
        suppress_warnings=True,
        error_action="ignore",
        n_fits=30,
    )
    log.info(f"SARIMA best order: {model.order} × {model.seasonal_order}")
    return model


def _align_macro(
    macro_df: pd.DataFrame,
    index: pd.DatetimeIndex,
    series_name: str,
) -> np.ndarray:
    """Align macro DataFrame to target index; return float64 array."""
    return align_macro_for_series(macro_df, index, series_name).values.astype(np.float64)


def _eval_on_2017(
    train: pd.Series,
    test: pd.Series,
    order: tuple,
    seasonal_order: tuple,
    exog_train: Optional[np.ndarray] = None,
    exog_test:  Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit SARIMA(X) on train and produce 12-step 2017 forecast."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = SARIMAX(
            train,
            exog=exog_train,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)

    fcast = res.get_forecast(steps=12, exog=exog_test)
    fc    = fcast.predicted_mean.values
    ci    = fcast.conf_int(alpha=ALPHA).values
    return fc, ci[:, 0], ci[:, 1]


def run_sarima(
    series_name: str,
    train: pd.Series,
    test: pd.Series,
    full: pd.Series,
    macro_train: Optional[pd.DataFrame] = None,
    macro_full:  Optional[pd.DataFrame] = None,
    use_macro: bool = False,
    actual_2017: Optional[pd.Series] = None,
    eval_actual: Optional[pd.Series] = None,
    forecast_dir: str = "outputs/forecasts",
) -> dict:
    """
    Full SARIMA/SARIMAX pipeline for one series.

    Parameters
    ----------
    series_name  : 'gas', 'electricity', or 'carbon'
    train        : history up to end of 2015
    test         : 2016 monthly series (12 obs)
    full         : history up to end of 2016 (no 2017 data)
    macro_train  : Dataset B aligned to training dates
    macro_full   : Dataset B for full period (must include 2016 rows for macro mode)
    use_macro    : fit SARIMAX with macro exogenous variables
    actual_2017  : actual 2017 target values for comparison column in CSV
    forecast_dir : output directory
    """
    mode = "macro" if use_macro else "core"
    log.info(f"[SARIMA-{mode.upper()}] Fitting on {series_name} (train={len(train)} obs)")

    # ── Order selection (always on core series, no exog) ─────────────────────
    try:
        import pmdarima as pm
        auto_model     = _fit_auto(train)
        order          = auto_model.order
        seasonal_order = auto_model.seasonal_order
    except ImportError:
        log.warning("pmdarima not available; using default SARIMA(1,1,1)(1,1,1,12)")
        order          = (1, 1, 1)
        seasonal_order = (1, 1, 1, 12)

    # ── Build exogenous arrays ────────────────────────────────────────────────
    exog_train = exog_test = exog_full = exog_2017 = None

    if use_macro and macro_train is not None and macro_full is not None:
        exog_train = _align_macro(macro_train, train.index, series_name)
        exog_test  = _align_macro(macro_full,  test.index, series_name)
        exog_full  = _align_macro(macro_full,  full.index, series_name)

        dates_2017 = pd.date_range("2017-01-01", periods=12, freq="MS")
        exog_2017  = _align_macro(macro_full, dates_2017, series_name)
        log.info(f"[SARIMA-MACRO] Using exogenous: {macro_cols_for_series(series_name, macro_full)}")

    # ── Evaluate on the 2016 validation split ────────────────────────────────
    fc_test, lb_test, ub_test = _eval_on_2017(
        train, test, order, seasonal_order,
        exog_train=exog_train,
        exog_test=exog_test,
    )
    test_arr = test.values
    if is_electricity(series_name):
        test_dates = pd.DatetimeIndex(test.index)
        hist_for_test = pd.concat([train, test]).sort_index()
        fc_test = index_forecast_to_yoy_growth(fc_test, test_dates, hist_for_test)
        lb_test = index_forecast_to_yoy_growth(lb_test, test_dates, hist_for_test)
        ub_test = index_forecast_to_yoy_growth(ub_test, test_dates, hist_for_test)
        test_arr = eval_actual.reindex(test_dates).values if eval_actual is not None else test_arr

    metrics = compute_all_metrics(
        actual=test_arr, forecast=fc_test,
        lower=lb_test, upper=ub_test,
        train_actual=train.values,
    )
    log.info(f"[SARIMA-{mode.upper()} {series_name}] 2016 validation: "
             f"MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}, "
             f"MAPE={metrics['MAPE']:.2f}%")

    # ── Refit on full history through 2016 → forecast 2017 ──────────────────
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_model = SARIMAX(
            full,
            exog=exog_full,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)

    fcast_2017 = final_model.get_forecast(steps=FORECAST_PERIODS, exog=exog_2017)
    fc_2017    = fcast_2017.predicted_mean.values
    ci_2017    = fcast_2017.conf_int(alpha=ALPHA).values
    lb_2017    = ci_2017[:, 0]
    ub_2017    = ci_2017[:, 1]
    forecast_dates = pd.date_range("2017-01-01", periods=FORECAST_PERIODS, freq="MS")
    if is_electricity(series_name):
        fc_2017 = index_forecast_to_yoy_growth(fc_2017, forecast_dates, full)
        lb_2017 = index_forecast_to_yoy_growth(lb_2017, forecast_dates, full)
        ub_2017 = index_forecast_to_yoy_growth(ub_2017, forecast_dates, full)

    df_out = pd.DataFrame({
        "date":        forecast_dates,
        "model":       "SARIMA",
        "mode":        mode,
        "forecast":    fc_2017.round(4),
        "lower_bound": lb_2017.round(4),
        "upper_bound": ub_2017.round(4),
    })

    if actual_2017 is not None:
        df_out["actual"] = actual_2017.reindex(forecast_dates).values

    Path(forecast_dir).mkdir(parents=True, exist_ok=True)
    out_path = f"{forecast_dir}/{series_name}_growth_pct_forecasts_sarima_{mode}.csv"
    df_out.to_csv(out_path, index=False)
    log.info(f"[SARIMA-{mode.upper()}] Forecast saved → {out_path}")

    return {
        "series":         series_name,
        "model":          "SARIMA",
        "mode":           mode,
        "metrics":        metrics,
        "forecast_path":  out_path,
        "order":          order,
        "seasonal_order": seasonal_order,
    }
