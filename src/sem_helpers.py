"""
sem_helpers.py
──────────────
Shared helpers for the SEM pipeline:
  - drop_feature        : drop redundant columns
  - assign_traditional_lihc : Low-Income High-Cost energy poverty flag
  - assign_hqrtm        : Household Quantile Regression Targeting Mechanism

Both energy-poverty classifiers accept the preprocessed DataFrame produced by
run_sem.py and return it with an additional 'risk_category' column:
    'energy_poor'   — both income and expenditure criteria met
    'at_risk'       — only one criterion met
    'not_poor'      — neither criterion met
"""

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from typing import Optional

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# Generic helpers
# ══════════════════════════════════════════════════════════════════════════════

def drop_feature(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Drop columns that exist in the DataFrame (silently skip missing ones)."""
    existing = [c for c in columns if c in df.columns]
    return df.drop(columns=existing, errors="ignore")


# ══════════════════════════════════════════════════════════════════════════════
# Traditional LIHC (Low Income High Cost)
# ══════════════════════════════════════════════════════════════════════════════

def assign_traditional_lihc(
    df: pd.DataFrame,
    exp_col: str = "total_expenditure",
    country_col: str = "Country",
    income_rule: str = "country_median_60",
    income_bracket_col: str = "income_bracket",
    exp_quantile: float = 0.80,
) -> pd.DataFrame:
    """
    Flag households as energy poor using the traditional LIHC method.

    Income threshold
    ────────────────
    'country_median_60' : low income if income_bracket ≤ 4  (proxy for < 60 %
                          of country median, using decile brackets)

    Expenditure threshold
    ─────────────────────
    High cost if total_expenditure > exp_quantile-th percentile within country.

    Returns
    -------
    DataFrame with added columns:
        low_income_flag   : bool
        high_cost_flag    : bool
        risk_category     : 'energy_poor' | 'at_risk' | 'not_poor'
    """
    out = df.copy()

    # ── Low income ────────────────────────────────────────────────────────────
    if income_rule == "country_median_60":
        out["low_income_flag"] = out[income_bracket_col].fillna(0).astype(int) <= 4
    else:
        out["low_income_flag"] = False

    # ── High cost (within-country percentile) ─────────────────────────────────
    threshold = out.groupby(country_col)[exp_col].transform(
        lambda x: x.quantile(exp_quantile)
    )
    out["high_cost_flag"] = out[exp_col] > threshold

    # ── Category ─────────────────────────────────────────────────────────────
    both   = out["low_income_flag"] & out["high_cost_flag"]
    either = out["low_income_flag"] | out["high_cost_flag"]
    out["risk_category"] = np.where(both, "energy_poor",
                             np.where(either, "at_risk", "not_poor"))

    return out


# ══════════════════════════════════════════════════════════════════════════════
# HQRTM  (Household Quantile Regression Targeting Mechanism)
# ══════════════════════════════════════════════════════════════════════════════

def assign_hqrtm(
    df: pd.DataFrame,
    qr_features: list,
    income_col: str = "equivalized_income",
    exp_col: str = "total_expenditure",
    country_col: str = "Country",
    income_rule: str = "bracket_lt4",
    income_bracket_col: str = "income_bracket",
    quantile: float = 0.65,
    add_country_effects: bool = True,
    margin_scale: float = 0.10,
) -> pd.DataFrame:
    """
    Assign energy poverty using Household Quantile Regression Targeting.

    The method estimates the q-th quantile of energy expenditure conditional on
    household characteristics.  A household is 'high-cost' if its actual
    expenditure exceeds the predicted quantile adjusted by margin_scale.

    Parameters
    ----------
    qr_features       : household feature columns for the quantile regression
    income_rule       : 'bracket_lt4' — low income if income_bracket < 4
                        'below_median' — below country-median equivalized income
    quantile          : quantile level (0.60 / 0.65 / 0.70 typical)
    add_country_effects : include country dummies in the QR model
    margin_scale      : threshold = predicted_q × (1 + margin_scale)

    Returns
    -------
    DataFrame with 'risk_category' : 'energy_poor' | 'at_risk' | 'not_poor'
    """
    try:
        from sklearn.linear_model import QuantileRegressor
        _HAS_SKLEARN_QR = True
    except ImportError:
        _HAS_SKLEARN_QR = False

    out = df.copy()

    # ── Low income flag ───────────────────────────────────────────────────────
    if income_rule == "bracket_lt4":
        low_income = out[income_bracket_col].fillna(5).astype(float) < 4
    elif income_rule == "below_median":
        med = out.groupby(country_col)[income_col].transform("median")
        low_income = out[income_col] < med
    else:
        low_income = pd.Series(False, index=out.index)

    # ── Build feature matrix ─────────────────────────────────────────────────
    X_cols = [c for c in qr_features if c in out.columns]
    X = out[X_cols].copy()

    if add_country_effects:
        country_dummies = pd.get_dummies(
            out[country_col].astype(str), prefix="c", drop_first=True
        )
        X = pd.concat([X, country_dummies], axis=1)

    # Encode any remaining string/categorical columns as integer codes
    for col in X.select_dtypes(include=["object", "category"]).columns:
        X[col] = X[col].astype("category").cat.codes.replace(-1, np.nan)

    X = X.fillna(X.median())
    X = X.astype(float)

    y = out[exp_col].fillna(out[exp_col].median()).values

    # ── Quantile regression ───────────────────────────────────────────────────
    if _HAS_SKLEARN_QR:
        qr = QuantileRegressor(quantile=quantile, alpha=0.0, solver="highs")
        try:
            qr.fit(X, y)
            predicted_q = qr.predict(X)
        except Exception:
            predicted_q = np.full(len(y), np.quantile(y, quantile))
    else:
        # Fallback: use global quantile
        predicted_q = np.full(len(y), np.quantile(y, quantile))

    out["_predicted_q"] = predicted_q
    threshold = predicted_q * (1 + margin_scale)
    high_cost = out[exp_col] > threshold

    # ── Category ─────────────────────────────────────────────────────────────
    both   = low_income & high_cost
    either = low_income | high_cost
    out["risk_category"] = np.where(both, "energy_poor",
                             np.where(either, "at_risk", "not_poor"))
    out["low_income_flag"] = low_income
    out["high_cost_flag"]  = high_cost
    out = out.drop(columns=["_predicted_q"], errors="ignore")

    return out
