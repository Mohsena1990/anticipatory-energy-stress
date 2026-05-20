"""
metrics_utils.py
────────────────
All point-forecast and probabilistic interval metrics used for model selection.

Metrics implemented
───────────────────
  Point    : MAE, RMSE, MAPE, SMAPE, MASE
  Interval : QuantileLoss, PredictionIntervalCoverage, WinklerScore, MSIS
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════════
# Point-forecast metrics
# ═══════════════════════════════════════════════════════════════════════════════

def mae(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(actual - forecast)))


def rmse(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((actual - forecast) ** 2)))


def mape(actual: np.ndarray, forecast: np.ndarray, eps: float = 1e-8) -> float:
    """Mean Absolute Percentage Error (%)."""
    return float(np.mean(np.abs((actual - forecast) / (np.abs(actual) + eps))) * 100)


def smape(actual: np.ndarray, forecast: np.ndarray, eps: float = 1e-8) -> float:
    """Symmetric Mean Absolute Percentage Error (%)."""
    num = np.abs(actual - forecast)
    den = (np.abs(actual) + np.abs(forecast)) / 2.0 + eps
    return float(np.mean(num / den) * 100)


def mase(
    actual: np.ndarray,
    forecast: np.ndarray,
    train_actual: np.ndarray,
    seasonality: int = 12,
) -> float:
    """
    Mean Absolute Scaled Error.
    Scaled by the MAE of the in-sample naive seasonal forecast.
    """
    naive_errors = np.abs(
        train_actual[seasonality:] - train_actual[:-seasonality]
    )
    scale = np.mean(naive_errors)
    if scale < 1e-10:
        return float("nan")
    return float(np.mean(np.abs(actual - forecast)) / scale)


# ═══════════════════════════════════════════════════════════════════════════════
# Probabilistic / interval metrics
# ═══════════════════════════════════════════════════════════════════════════════

def quantile_loss(
    actual: np.ndarray,
    quantile_forecast: np.ndarray,
    q: float,
) -> float:
    """
    Pinball (quantile) loss at level *q*.
    Lower is better; q=0.5 gives MAE/2.
    """
    errors = actual - quantile_forecast
    loss = np.where(errors >= 0, q * errors, (q - 1) * errors)
    return float(np.mean(loss))


def prediction_interval_coverage(
    actual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """
    Empirical coverage: fraction of actuals inside [lower, upper].
    Target = nominal level (e.g. 0.95 for a 95 % PI).
    """
    covered = np.sum((actual >= lower) & (actual <= upper))
    return float(covered / len(actual))


def winkler_score(
    actual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float = 0.05,
) -> float:
    """
    Winkler Score for a (1-alpha)*100 % prediction interval.
    Penalises both wide intervals and misses.  Lower is better.
    """
    width = upper - lower
    penalty = np.where(
        actual < lower, (2 / alpha) * (lower - actual),
        np.where(actual > upper, (2 / alpha) * (actual - upper), 0.0)
    )
    return float(np.mean(width + penalty))


def msis(
    actual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    train_actual: np.ndarray,
    alpha: float = 0.05,
    seasonality: int = 12,
) -> float:
    """
    Mean Scaled Interval Score (Gneiting & Raftery 2005).
    Scaled by the naive seasonal in-sample MAE.  Lower is better.
    """
    naive_mae = np.mean(
        np.abs(train_actual[seasonality:] - train_actual[:-seasonality])
    )
    if naive_mae < 1e-10:
        return float("nan")

    width   = upper - lower
    penalty = (2 / alpha) * (
        np.maximum(0, lower - actual) + np.maximum(0, actual - upper)
    )
    return float(np.mean(width + penalty) / naive_mae)


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience wrapper
# ═══════════════════════════════════════════════════════════════════════════════

def compute_all_metrics(
    actual: np.ndarray,
    forecast: np.ndarray,
    lower: Optional[np.ndarray],
    upper: Optional[np.ndarray],
    train_actual: np.ndarray,
    alpha: float = 0.05,
    seasonality: int = 12,
) -> dict:
    """
    Compute the full metric suite and return a flat dict.

    Parameters
    ----------
    actual      : observed test-set values
    forecast    : point forecast (median / mean)
    lower/upper : prediction interval bounds (may be None)
    train_actual: in-sample actuals for MASE / MSIS scaling
    alpha       : significance level for PI metrics (default 0.05 → 95 % PI)
    seasonality : seasonal period for naive benchmark (default 12 months)
    """
    metrics: dict = {
        "MAE":   mae(actual, forecast),
        "RMSE":  rmse(actual, forecast),
        "MAPE":  mape(actual, forecast),
        "SMAPE": smape(actual, forecast),
        "MASE":  mase(actual, forecast, train_actual, seasonality),
    }

    if lower is not None and upper is not None:
        # Treat forecast as 0.5 quantile for QuantileLoss
        metrics["QuantileLoss"] = quantile_loss(actual, forecast, q=0.5)
        metrics["PredictionIntervalCoverage"] = prediction_interval_coverage(
            actual, lower, upper
        )
        metrics["WinklerScore"] = winkler_score(actual, lower, upper, alpha)
        metrics["MSIS"]         = msis(actual, lower, upper, train_actual, alpha, seasonality)
    else:
        metrics["QuantileLoss"]              = float("nan")
        metrics["PredictionIntervalCoverage"] = float("nan")
        metrics["WinklerScore"]               = float("nan")
        metrics["MSIS"]                       = float("nan")

    return metrics
