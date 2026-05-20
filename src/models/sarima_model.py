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
  3. Macro mode : SARIMAX — inflation_growth, weather_volatility, gdp_growth
     as exogenous regressors.
     2017 evaluation  : aligned 2017 macro slice as exog.
     2018 forecast    : actual 2018 macro values from macro_full as exog.
                        Real values are NEVER replaced with zeros.
  4. Evaluate on 2017 test set; refit on full 2005-2017; forecast 2018.
  5. Save CSV with actual 2018 values for comparison.
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

log = get_logger("sarima")
warnings.filterwarnings("ignore")

FORECAST_PERIODS = 12
ALPHA            = 0.05
MACRO_COLS       = ["inflation_growth", "weather_volatility", "gdp_growth"]


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
) -> np.ndarray:
    """Align macro DataFrame to target index; return float64 array."""
    cols = [c for c in MACRO_COLS if c in macro_df.columns]
    return macro_df[cols].reindex(index).ffill().bfill().values.astype(np.float64)


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
    actual_2018: Optional[pd.Series] = None,
    forecast_dir: str = "outputs/forecasts",
) -> dict:
    """
    Full SARIMA/SARIMAX pipeline for one series.

    Parameters
    ----------
    series_name  : 'gas', 'electricity', or 'carbon'
    train        : history up to end of 2016
    test         : 2017 monthly series (12 obs)
    full         : history up to end of 2017 (no 2018 data)
    macro_train  : Dataset B aligned to training dates
    macro_full   : Dataset B for full period (must include 2018 rows for macro mode)
    use_macro    : fit SARIMAX with macro exogenous variables
    actual_2018  : actual 2018 target values for comparison column in CSV
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
    exog_train = exog_test = exog_full = exog_2018 = None

    if use_macro and macro_train is not None and macro_full is not None:
        exog_train = _align_macro(macro_train, train.index)
        exog_test  = _align_macro(macro_full,  test.index)
        exog_full  = _align_macro(macro_full,  full.index)

        dates_2018 = pd.date_range("2018-01-01", periods=12, freq="MS")
        exog_2018  = _align_macro(macro_full, dates_2018)
        log.info(f"[SARIMA-MACRO] Using exogenous: {MACRO_COLS}")

    # ── Evaluate on 2017 ─────────────────────────────────────────────────────
    fc_test, lb_test, ub_test = _eval_on_2017(
        train, test, order, seasonal_order,
        exog_train=exog_train,
        exog_test=exog_test,
    )
    test_arr = test.values

    metrics = compute_all_metrics(
        actual=test_arr, forecast=fc_test,
        lower=lb_test, upper=ub_test,
        train_actual=train.values,
    )
    log.info(f"[SARIMA-{mode.upper()} {series_name}] 2017 eval: "
             f"MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}, "
             f"MAPE={metrics['MAPE']:.2f}%")

    # ── Refit on full 2005-2017 → forecast 2018 ──────────────────────────────
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

    fcast_2018 = final_model.get_forecast(steps=FORECAST_PERIODS, exog=exog_2018)
    fc_2018    = fcast_2018.predicted_mean.values
    ci_2018    = fcast_2018.conf_int(alpha=ALPHA).values
    lb_2018    = ci_2018[:, 0]
    ub_2018    = ci_2018[:, 1]

    forecast_dates = pd.date_range("2018-01-01", periods=FORECAST_PERIODS, freq="MS")
    df_out = pd.DataFrame({
        "date":        forecast_dates,
        "model":       "SARIMA",
        "mode":        mode,
        "forecast":    fc_2018.round(4),
        "lower_bound": lb_2018.round(4),
        "upper_bound": ub_2018.round(4),
    })

    if actual_2018 is not None:
        df_out["actual"] = actual_2018.reindex(forecast_dates).values

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
