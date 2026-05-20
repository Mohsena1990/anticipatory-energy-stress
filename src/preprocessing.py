"""
preprocessing.py
────────────────
Data preprocessing pipeline for both Dataset A (core) and Dataset B (macro).

Input: cleaned CSVs produced by src/data_loader.py
         data/raw/core_energy_carbon.csv
         data/raw/macro_controls.csv

Steps
─────
  1. Load loader-produced CSVs
  2. Parse dates; set monthly frequency
  3. Missing-value audit and forward-fill
  4. Outlier detection (IQR, flagged but not removed by default)
  5. Stationarity check (ADF test)
  6. Log-return transformation (optional)
  7. Feature engineering: lags, rolling mean/std
  8. Train / test split  (2005-2016 train | 2017 test | 2018 forecast)
  9. Save processed datasets to data/processed/
"""

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple

from sympy import series

from src.logging_utils import get_logger

log = get_logger("preprocessing")

TRAIN_END = "2016-12-01"
TEST_START = "2017-01-01"
TEST_END   = "2017-12-01"


# ── Step 1-2: Load & parse ────────────────────────────────────────────────────

def load_and_parse(path: str, date_col: str = "date") -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    df[date_col] = pd.to_datetime(df[date_col]).dt.to_period("M").dt.to_timestamp()
    df = df.set_index(date_col)
    df.index = pd.DatetimeIndex(df.index)
    log.debug(f"Loaded {path}: {df.shape}, range {df.index.min()} – {df.index.max()}")
    return df


# ── Step 3: Missing values ────────────────────────────────────────────────────

def audit_missing(df: pd.DataFrame, label: str) -> pd.DataFrame:
    missing = df.isnull().sum()
    if missing.any():
        log.warning(f"[{label}] Missing values found:\n{missing[missing > 0]}")
        df = df.ffill().bfill()
        log.info(f"[{label}] Missing values filled via forward-fill + back-fill")
    else:
        log.info(f"[{label}] No missing values detected")
    return df


# ── Step 4: Outlier detection ─────────────────────────────────────────────────

def flag_outliers(df: pd.DataFrame, factor: float = 3.0) -> pd.DataFrame:
    """Flag (but do not remove) IQR outliers; adds boolean *_outlier columns."""
    numeric = df.select_dtypes(include="number")
    for col in numeric.columns:
        q1, q3 = numeric[col].quantile([0.25, 0.75])
        iqr     = q3 - q1
        lo, hi  = q1 - factor * iqr, q3 + factor * iqr
        mask    = (df[col] < lo) | (df[col] > hi)
        if mask.any():
            df[f"{col}_outlier"] = mask
            log.warning(f"Outliers in '{col}': {mask.sum()} rows flagged")
    return df


# ── Step 5: Stationarity ──────────────────────────────────────────────────────

def adf_check(series: pd.Series, label: str) -> dict:
    """Run the Augmented Dickey-Fuller test and log the result."""
    try:
        from statsmodels.tsa.stattools import adfuller
        result  = adfuller(series.dropna(), autolag="AIC")
        pval    = result[1]
        stat    = result[0]
        is_stat = pval < 0.05
        log.info(
            f"ADF [{label}]: stat={stat:.4f}, p={pval:.4f} → "
            f"{'STATIONARY' if is_stat else 'NON-STATIONARY'}"
        )
        return {"statistic": stat, "pvalue": pval, "stationary": is_stat}
    except ImportError:
        log.warning("statsmodels not installed; skipping ADF test")
        return {}


# ── Step 6: Transformations ───────────────────────────────────────────────────

def log_return(series: pd.Series) -> pd.Series:
    """Compute log-returns: log(x_t / x_{t-1})."""
    return np.log(series / series.shift(1))


def add_log_returns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[f"{col}_logret"] = log_return(df[col])
    return df


# ── Step 7: Feature engineering ───────────────────────────────────────────────

def add_features(
    df: pd.DataFrame,
    cols: list[str],
    lags: list[int] = [1, 3, 6, 12],
    windows: list[int] = [3, 6, 12],
) -> pd.DataFrame:
    """Add lag and rolling statistics for each target column."""
    for col in cols:
        if col not in df.columns:
            continue
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
        for win in windows:
            df[f"{col}_rm{win}"]  = df[col].rolling(win).mean()
            df[f"{col}_rstd{win}"] = df[col].rolling(win).std()

    # Calendar features
    df["month"]       = df.index.month
    df["month_sin"]   = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]   = np.cos(2 * np.pi * df["month"] / 12)
    df["year"]        = df.index.year
    df["time_index"]  = (df.index.year - df.index.year.min()) * 12 + df.index.month
    return df


# ── Step 8: Train / test split ────────────────────────────────────────────────

def split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train 2005-2016, test 2017)."""
    train = df.loc[:TRAIN_END].copy()
    test  = df.loc[TEST_START:TEST_END].copy()
    log.info(f"Train: {len(train)} rows | Test: {len(test)} rows")
    return train, test


# ── Step 9: Save ──────────────────────────────────────────────────────────────

def save_processed(df: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    log.info(f"Saved processed data → {path}")


# ── Public API ────────────────────────────────────────────────────────────────

def preprocess_core(
    raw_path: str = "data/raw/core_energy_carbon.csv",
    out_path: str = "data/processed/core_processed.csv",
    add_feats: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Full preprocessing pipeline for Dataset A.

    Returns
    -------
    (full_df, train_df, test_df)
    """
    TARGET_COLS = ["gas_growth", "electricity_growth", "carbon_growth"]

    df = load_and_parse(raw_path)
    df = audit_missing(df, "core")
    df = flag_outliers(df)

    for col in TARGET_COLS:
        adf_check(df[col], col)

    # df = add_log_returns(df, TARGET_COLS)
    def signed_log1p(series):
        return np.sign(series) * np.log1p(np.abs(series))
    
    
    df[f"{col}_signed_log"] = signed_log1p(df[col])

    if add_feats:
        df = add_features(df, TARGET_COLS)

    save_processed(df, out_path)

    train, test = split(df)
    return df, train, test


def preprocess_macro(
    raw_path: str = "data/raw/macro_controls.csv",
    out_path: str = "data/processed/macro_processed.csv",
    add_feats: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Full preprocessing pipeline for Dataset B.

    Returns
    -------
    (full_df, train_df, test_df)
    """
    MACRO_COLS = ["inflation_growth", "weather_volatility", "gdp_growth"]

    df = load_and_parse(raw_path)
    df = audit_missing(df, "macro")

    if add_feats:
        df = add_features(df, MACRO_COLS, lags=[1, 3], windows=[3, 6])

    save_processed(df, out_path)

    train, test = split(df)
    return df, train, test


if __name__ == "__main__":
    from src.logging_utils import setup_logger
    setup_logger()
    core_full, core_train, core_test = preprocess_core()
    print(core_train.tail(3))
