"""
prophet_model.py
────────────────
Facebook Prophet forecasting for a single time series.

Strategy
────────
  1. Fit Prophet on training data (up to 2016).
  2. Core mode  : no extra regressors — pure trend + seasonality.
  3. Macro mode : inflation_growth, weather_volatility, gdp_growth added as
     additional regressors.  The 2018 future DataFrame is populated with the
     actual 2018 macro values from macro_full — not zeros, not forward-fills
     from 2017.
  4. Evaluate on 2017; refit on 2005-2017; forecast Jan-Dec 2018.
  5. Save CSV with actual 2018 values for comparison.
     Filename: {series}_growth_pct_forecasts_prophet_core.csv
               {series}_growth_pct_forecasts_prophet_macro.csv
"""

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from src.logging_utils import get_logger
from src.metrics_utils import compute_all_metrics

log = get_logger("prophet")
warnings.filterwarnings("ignore")

MACRO_COLS = ["inflation_growth", "weather_volatility", "gdp_growth"]


def _make_prophet_df(series: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"ds": series.index, "y": series.values})


def _add_regressors(
    prophet_df: pd.DataFrame,
    macro_df: Optional[pd.DataFrame],
    regressor_cols: list,
    series_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    if macro_df is None or not regressor_cols:
        return prophet_df
    macro_aligned = macro_df[regressor_cols].reindex(series_index).ffill().bfill()
    for col in regressor_cols:
        if col in macro_aligned.columns:
            prophet_df[col] = macro_aligned[col].values
    return prophet_df


def run_prophet(
    series_name: str,
    train: pd.Series,
    test: pd.Series,
    full: pd.Series,
    macro_train: Optional[pd.DataFrame] = None,
    macro_full: Optional[pd.DataFrame]  = None,
    use_regressors: bool = False,
    actual_2018: Optional[pd.Series] = None,
    forecast_dir: str = "outputs/forecasts",
) -> dict:
    """
    Full Prophet pipeline for one series.

    Parameters
    ----------
    series_name    : 'gas', 'electricity', or 'carbon'
    train          : history up to end of 2016
    test           : 2017 monthly series
    full           : history up to end of 2017 (no 2018 data)
    macro_train    : Dataset B aligned to training dates
    macro_full     : Dataset B for full period; must include 2018 rows so that
                     the 2018 forecast receives actual (not zero) macro values.
    use_regressors : include macro variables as additional regressors (macro mode)
    actual_2018    : actual 2018 target values for comparison column in CSV
    forecast_dir   : output directory
    """
    try:
        from prophet import Prophet
    except ImportError:
        log.error("prophet package not installed. Run: pip install prophet")
        raise

    mode = "macro" if use_regressors else "core"
    log.info(f"[Prophet-{mode.upper()}] Fitting on {series_name} "
             f"(train={len(train)} obs, regressors={use_regressors})")

    # ── Determine available macro columns ─────────────────────────────────────
    if use_regressors and macro_train is not None:
        available = [c for c in MACRO_COLS if c in macro_train.columns]
    else:
        available = []

    # ── Training ──────────────────────────────────────────────────────────────
    train_df = _make_prophet_df(train)
    if available:
        train_df = _add_regressors(train_df, macro_train, available, train.index)

    m = Prophet(
        interval_width=0.95,
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="multiplicative" if series_name == "gas" else "additive",
    )
    for col in available:
        m.add_regressor(col)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.fit(train_df)

    # ── Evaluation on 2017 ────────────────────────────────────────────────────
    test_future = m.make_future_dataframe(periods=12, freq="MS")
    if available and macro_full is not None:
        future_dates = pd.DatetimeIndex(test_future["ds"])
        for col in available:
            vals = macro_full[col].reindex(future_dates).ffill().bfill()
            test_future[col] = vals.values

    forecast_eval = m.predict(test_future)
    fc_test = forecast_eval.tail(12)["yhat"].values
    lb_test = forecast_eval.tail(12)["yhat_lower"].values
    ub_test = forecast_eval.tail(12)["yhat_upper"].values

    metrics = compute_all_metrics(
        actual=test.values, forecast=fc_test,
        lower=lb_test, upper=ub_test,
        train_actual=train.values,
    )
    log.info(f"[Prophet-{mode.upper()} {series_name}] 2017 eval: "
             f"MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}")

    # ── Refit on full 2005-2017 ───────────────────────────────────────────────
    full_df = _make_prophet_df(full)
    if available and macro_full is not None:
        full_df = _add_regressors(full_df, macro_full, available, full.index)

    m2 = Prophet(
        interval_width=0.95,
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="multiplicative" if series_name == "gas" else "additive",
    )
    for col in available:
        m2.add_regressor(col)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m2.fit(full_df)

    # ── Forecast 2018: future DataFrame uses actual 2018 macro values ─────────
    future_2018 = m2.make_future_dataframe(periods=12, freq="MS")
    if available and macro_full is not None:
        future_dates = pd.DatetimeIndex(future_2018["ds"])
        for col in available:
            # macro_full covers through 2018; reindex gives actual 2018 values
            vals = macro_full[col].reindex(future_dates).ffill().bfill()
            future_2018[col] = vals.values

    fcast_2018 = m2.predict(future_2018)
    rows_2018  = fcast_2018[fcast_2018["ds"].dt.year == 2018]

    forecast_dates = pd.to_datetime(rows_2018["ds"].values)
    df_out = pd.DataFrame({
        "date":        forecast_dates,
        "model":       "Prophet",
        "mode":        mode,
        "forecast":    rows_2018["yhat"].values.round(4),
        "lower_bound": rows_2018["yhat_lower"].values.round(4),
        "upper_bound": rows_2018["yhat_upper"].values.round(4),
    })

    if actual_2018 is not None:
        df_out["actual"] = actual_2018.reindex(
            pd.DatetimeIndex(forecast_dates)
        ).values

    Path(forecast_dir).mkdir(parents=True, exist_ok=True)
    out_path = f"{forecast_dir}/{series_name}_growth_pct_forecasts_prophet_{mode}.csv"
    df_out.to_csv(out_path, index=False)
    log.info(f"[Prophet-{mode.upper()}] Forecast saved → {out_path}")

    return {
        "series":        series_name,
        "model":         "Prophet",
        "mode":          mode,
        "metrics":       metrics,
        "forecast_path": out_path,
    }
