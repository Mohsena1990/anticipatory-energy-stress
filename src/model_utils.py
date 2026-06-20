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

# Linear models (SARIMA, Prophet) also receive the 12-month lagged GBP/EUR
# depreciation signal, which captures the ~12-month regulatory transmission
# window from currency shock to UK retail electricity prices. Neural models
# (LSTM, TFT) use ELECTRICITY_LEAN_COLS — the lag-12 GBP feature adds noise
# for them given limited training data and fast-mode epochs.
ELECTRICITY_LINEAR_COLS = ELECTRICITY_LEAN_COLS + [
    "gbp_eur_yoy_change_lag12",
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

# Gas: neural models (LSTM, TFT) receive only stable macro signals — gas
# futures log-returns add high-frequency noise that overfits at n=116.
# Linear models (SARIMA, Prophet) can exploit the futures signal cleanly.
GAS_LEAN_COLS   = BASE_MACRO_COLS
GAS_LINEAR_COLS = BASE_MACRO_COLS + [
    "gas_futures_log_return_lag1",
    "gas_futures_yoy_growth_lag1",
]

# Carbon: TFT benefits from the gas/carbon price correlation via
# gas_futures_yoy_growth_lag1 (lean), but LSTM overfits this volatile signal
# at n=116 (carbon LSTM macro went from 76→76 with lean, worse than core).
# Linear models (Prophet, SARIMA) blow up when gas futures are OOD in 2017.
# Three-way split: TFT=lean (gas futures), LSTM=lstm (stable only), linear=BASE only.
CARBON_LEAN_COLS   = BASE_MACRO_COLS + ["gas_futures_yoy_growth_lag1"]
CARBON_LSTM_COLS   = BASE_MACRO_COLS
CARBON_LINEAR_COLS = BASE_MACRO_COLS

ALL_SERIES_PROFILES: dict[str, dict[str, list[str]]] = {
    "gas": {
        "lean":   GAS_LEAN_COLS,
        "linear": GAS_LINEAR_COLS,
    },
    "electricity": {
        "lean":         ELECTRICITY_LEAN_COLS,
        "linear":       ELECTRICITY_LINEAR_COLS,
        "demand":       ELECTRICITY_DEMAND_COLS,
        "supply_demand": ELECTRICITY_SUPPLY_DEMAND_COLS,
    },
    "carbon": {
        "lean":   CARBON_LEAN_COLS,
        "lstm":   CARBON_LSTM_COLS,
        "linear": CARBON_LINEAR_COLS,
    },
}

# Keep for any direct external references
ELECTRICITY_MACRO_PROFILES = ALL_SERIES_PROFILES["electricity"]


def macro_cols_for_series(
    series_name: str,
    macro_df: pd.DataFrame | None = None,
    feature_set: str = "lean",
) -> list[str]:
    profiles = ALL_SERIES_PROFILES.get(series_name.lower(), {})
    cols = profiles.get(feature_set, profiles.get("lean", BASE_MACRO_COLS))
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
