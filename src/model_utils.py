"""
Shared model helpers for series-specific macro features and electricity handling.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


BASE_MACRO_COLS = [
    "inflation_growth_lag1",
    "weather_volatility_lag1",
    "gdp_growth_lag1",
]

ELECTRICITY_LEAN_COLS = BASE_MACRO_COLS + [
    "gas_futures_log_return_lag1",
    "gas_futures_yoy_growth_lag1",
    "electricity_growth_lag12",
    "post_2016_electricity_regime",
]

ELECTRICITY_DEMAND_COLS = ELECTRICITY_LEAN_COLS + [
    "electricity_demand_yoy_growth_lag1",
    "electricity_peak_yoy_growth_lag1",
]

ELECTRICITY_SUPPLY_DEMAND_COLS = ELECTRICITY_DEMAND_COLS + [
    "embedded_wind_generation_mean_yoy_growth_lag1",
    "embedded_solar_generation_mean_yoy_growth_lag1",
    "embedded_wind_capacity_mean_yoy_growth_lag1",
    "embedded_solar_capacity_mean_yoy_growth_lag1",
    "pump_storage_pumping_mean_yoy_growth_lag1",
    "interconnector_net_flow_mean_yoy_growth_lag1",
    "holiday_share_lag1",
]

SERIES_MACRO_COLS = {
    "gas": BASE_MACRO_COLS + [
        "gas_futures_log_return_lag1",
        "gas_futures_yoy_growth_lag1",
    ],
    "electricity": ELECTRICITY_LEAN_COLS,
    "carbon": BASE_MACRO_COLS + [
        "gas_futures_yoy_growth_lag1",
    ],
}

ELECTRICITY_MACRO_PROFILES = {
    "lean": ELECTRICITY_LEAN_COLS,
    "demand": ELECTRICITY_DEMAND_COLS,
    "supply_demand": ELECTRICITY_SUPPLY_DEMAND_COLS,
}


def macro_cols_for_series(
    series_name: str,
    macro_df: pd.DataFrame | None = None,
    feature_set: str = "lean",
) -> list[str]:
    if series_name.lower() == "electricity":
        cols = ELECTRICITY_MACRO_PROFILES.get(feature_set, ELECTRICITY_LEAN_COLS)
    else:
        cols = SERIES_MACRO_COLS.get(series_name.lower(), BASE_MACRO_COLS)
    if macro_df is None:
        return cols
    return [c for c in cols if c in macro_df.columns]


def align_macro_for_series(
    macro_df: pd.DataFrame,
    index: pd.DatetimeIndex,
    series_name: str,
    feature_set: str = "lean",
) -> pd.DataFrame:
    """Align lagged macro regressors without using future back-fill."""
    cols = macro_cols_for_series(series_name, macro_df, feature_set=feature_set)
    aligned = macro_df[cols].reindex(index).ffill()
    if aligned.isna().any().any():
        aligned = aligned.fillna(0.0)
    return aligned.astype(np.float64)


def is_electricity(series_name: str) -> bool:
    """True only for legacy runs that explicitly pass the CPI index as target."""
    return series_name.lower() == "electricity_index"


def index_forecast_to_yoy_growth(
    forecast_index: np.ndarray,
    forecast_dates: pd.DatetimeIndex,
    historical_index: pd.Series,
) -> np.ndarray:
    """
    Convert forecast CPI index levels into YoY growth rates using the actual
    index level from 12 months earlier.
    """
    hist = historical_index.sort_index()
    denom_dates = forecast_dates - pd.DateOffset(months=12)
    denom = hist.reindex(denom_dates).to_numpy(dtype=float)
    out = (np.asarray(forecast_index, dtype=float) / denom - 1.0) * 100.0
    return out
