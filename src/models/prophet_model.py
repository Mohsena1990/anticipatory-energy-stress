"""
prophet_model.py
────────────────
Facebook Prophet forecasting for a single time series.

Strategy
────────
  1. Fit Prophet on training data (up to 2016).
  2. Core mode  : no extra regressors — pure trend + seasonality.
  3. Macro mode : lagged, series-specific regressors. Gas uses gas-futures
     signals; electricity uses lagged fuel-cost and electricity-growth signals.
  4. Evaluate on 2016; refit through 2016; forecast Jan-Dec 2017.
  5. Save CSV with actual 2017 values for comparison.
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
from src.model_utils import (
    align_macro_for_series,
    index_forecast_to_yoy_growth,
    is_electricity,
    macro_cols_for_series,
)

log = get_logger("prophet")
warnings.filterwarnings("ignore")

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
    macro_aligned = macro_df[regressor_cols].reindex(series_index).ffill().fillna(0.0)
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
    actual_2017: Optional[pd.Series] = None,
    eval_actual: Optional[pd.Series] = None,
    forecast_dir: str = "outputs/forecasts",
    changepoint_prior_scale: float = 0.05,
    seasonality_prior_scale: float = 10.0,
    seasonality_mode: Optional[str] = None,
    return_model: bool = False,
) -> dict:
    """
    Full Prophet pipeline for one series.

    Parameters
    ----------
    series_name    : 'gas', 'electricity', or 'carbon'
    train          : history up to end of 2015
    test           : 2016 monthly series
    full           : history up to end of 2016 (no 2017 data)
    macro_train    : Dataset B aligned to training dates
    macro_full     : Dataset B for full period; must include 2017 rows so that
                     the 2017 forecast receives actual (not zero) macro values.
    use_regressors : include macro variables as additional regressors (macro mode)
    actual_2017    : actual 2017 target values for comparison column in CSV
    forecast_dir   : output directory
    """
    try:
        from prophet import Prophet
    except ImportError:
        log.error("prophet package not installed. Run: pip install prophet")
        raise

    mode = "macro" if use_regressors else "core"
    train_seasonality_mode = seasonality_mode or "additive"
    final_seasonality_mode = (
        seasonality_mode
        or ("multiplicative" if series_name == "gas" else "additive")
    )
    log.info(f"[Prophet-{mode.upper()}] Fitting on {series_name} "
             f"(train={len(train)} obs, regressors={use_regressors}, "
             f"cps={changepoint_prior_scale}, sps={seasonality_prior_scale}, "
             f"seasonality={final_seasonality_mode})")

    # ── Determine available macro columns ─────────────────────────────────────
    if use_regressors and macro_train is not None:
        available = macro_cols_for_series(series_name, macro_train, feature_set="linear")
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
        seasonality_mode=train_seasonality_mode,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale,
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
        aligned = align_macro_for_series(macro_full, future_dates, series_name, feature_set="linear")
        for col in available:
            test_future[col] = aligned[col].values

    forecast_eval = m.predict(test_future)
    test_dates = pd.DatetimeIndex(test.index)
    fc_test = forecast_eval.tail(12)["yhat"].values
    lb_test = forecast_eval.tail(12)["yhat_lower"].values
    ub_test = forecast_eval.tail(12)["yhat_upper"].values
    test_arr = test.values
    if is_electricity(series_name):
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
    log.info(f"[Prophet-{mode.upper()} {series_name}] 2016 validation: "
             f"MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}")

    # ── Refit on full history through 2016 ───────────────────────────────────
    full_df = _make_prophet_df(full)
    if available and macro_full is not None:
        full_df = _add_regressors(full_df, macro_full, available, full.index)

    m2 = Prophet(
        interval_width=0.95,
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode=final_seasonality_mode,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale,
    )
    for col in available:
        m2.add_regressor(col)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m2.fit(full_df)

    # ── Forecast 2017 with lagged macro regressors ───────────────────────────
    future_2017 = m2.make_future_dataframe(periods=12, freq="MS")
    if available and macro_full is not None:
        future_dates = pd.DatetimeIndex(future_2017["ds"])
        aligned = align_macro_for_series(macro_full, future_dates, series_name, feature_set="linear")
        for col in available:
            future_2017[col] = aligned[col].values
    fcast_2017 = m2.predict(future_2017)
    rows_2017  = fcast_2017[fcast_2017["ds"].dt.year == 2017]

    forecast_dates = pd.to_datetime(rows_2017["ds"].values)
    fc_2017 = rows_2017["yhat"].values
    lb_2017 = rows_2017["yhat_lower"].values
    ub_2017 = rows_2017["yhat_upper"].values
    if is_electricity(series_name):
        fc_2017 = index_forecast_to_yoy_growth(fc_2017, pd.DatetimeIndex(forecast_dates), full)
        lb_2017 = index_forecast_to_yoy_growth(lb_2017, pd.DatetimeIndex(forecast_dates), full)
        ub_2017 = index_forecast_to_yoy_growth(ub_2017, pd.DatetimeIndex(forecast_dates), full)

    # Clip to ±3σ using the recent 60-month window rather than full history.
    # Full-history std is inflated by structural one-time events (e.g. EUA Phase I
    # collapse in 2007), making the full-history bounds too wide to be useful.
    _recent = full.iloc[-60:]
    _t_mean = _recent.mean()
    _t_std  = max(_recent.std(), 1.0)
    _lo, _hi = _t_mean - 3 * _t_std, _t_mean + 3 * _t_std
    fc_2017 = np.clip(fc_2017, _lo, _hi)
    lb_2017 = np.clip(lb_2017, _lo, _hi)
    ub_2017 = np.clip(ub_2017, _lo, _hi)

    df_out = pd.DataFrame({
        "date":        forecast_dates,
        "model":       "Prophet",
        "mode":        mode,
        "forecast":    fc_2017.round(4),
        "lower_bound": lb_2017.round(4),
        "upper_bound": ub_2017.round(4),
    })

    if actual_2017 is not None:
        df_out["actual"] = actual_2017.reindex(
            pd.DatetimeIndex(forecast_dates)
        ).values

    Path(forecast_dir).mkdir(parents=True, exist_ok=True)
    out_path = f"{forecast_dir}/{series_name}_growth_pct_forecasts_prophet_{mode}.csv"
    df_out.to_csv(out_path, index=False)
    log.info(f"[Prophet-{mode.upper()}] Forecast saved → {out_path}")

    result = {
        "series":        series_name,
        "model":         "Prophet",
        "mode":          mode,
        "metrics":       metrics,
        "forecast_path": out_path,
        "params": {
            "changepoint_prior_scale": changepoint_prior_scale,
            "seasonality_prior_scale": seasonality_prior_scale,
            "seasonality_mode": final_seasonality_mode,
        },
    }
    if return_model:
        result["fitted_model"]   = m2
        result["regressor_cols"] = list(available) if available else []
    return result
