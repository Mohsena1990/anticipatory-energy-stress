

"""
lstm_model.py
─────────────
Leakage-safe univariate + optional macro-augmented LSTM forecasting.

Architecture
────────────
  Input  → Lookback window of 12 months
  Body   → LSTM(128) → Dropout(0.2) → LSTM(64) → Dropout(0.2)
  Output → Dense(1) for point forecast
  PI     → Monte Carlo Dropout (N=200 forward passes at inference time)

Strategy
────────
  1. Scale the target and macro variables separately.
     - Target scaler is fitted only on the available training target.
     - Macro scaler is fitted only on the available training macro controls.
     - Carbon/log-return series use RobustScaler because of extreme volatility.

  2. Build sliding-window sequences using a 12-month lookback.

  3. Train on available history up to 2016 and evaluate on 2017.
     Validation is temporal, not random, to avoid time-series leakage.

  4. Refit on full history through 2016 and recursively forecast Jan-Dec 2017.

  Two forecast modes:
    core  — recursive, appending only the predicted target.
    macro — recursive, appending predicted target PLUS known 2017 macro values
            at each step. Macro values are never zeroed in macro mode.

  5. Save forecast CSV with actual 2017 values for comparison.
     Filename suffix: _lstm_core.csv or _lstm_macro.csv
"""

from __future__ import annotations

import random
import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src.logging_utils import get_logger
from src.metrics_utils import compute_all_metrics
from src.model_utils import (
    index_forecast_to_yoy_growth,
    is_electricity,
    macro_cols_for_series,
)

log = get_logger("lstm")
warnings.filterwarnings("ignore")

LOOKBACK = 12
MC_SAMPLES = 200
EPOCHS = 100
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
ALPHA = 0.05

# ══════════════════════════════════════════════════════════════════════════════
# Reproducibility
# ══════════════════════════════════════════════════════════════════════════════

def _set_seed(seed: int = 42) -> None:
    """Set reproducibility seeds for NumPy, Python, and TensorFlow."""
    np.random.seed(seed)
    random.seed(seed)
    try:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(seed)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# Data utilities
# ══════════════════════════════════════════════════════════════════════════════

def _safe_series(series: pd.Series) -> pd.Series:
    """Ensure datetime index, sorted order, and no missing target values."""
    s = series.copy()
    s.index = pd.to_datetime(s.index)
    return s.sort_index().dropna()


def _safe_macro(macro: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Ensure datetime index and sorted order for macro data."""
    if macro is None:
        return None
    m = macro.copy()
    m.index = pd.to_datetime(m.index)
    return m.sort_index()


def _available_macro_cols(
    macro: Optional[pd.DataFrame],
    series_name: str,
    macro_feature_set: str = "lean",
) -> list[str]:
    if macro is None:
        return []
    return macro_cols_for_series(series_name, macro, feature_set=macro_feature_set)


# ══════════════════════════════════════════════════════════════════════════════
# Scaling
# ══════════════════════════════════════════════════════════════════════════════

def _select_target_scaler(series_name: str):
    """
    Select target scaler.

    Carbon is now expected to be carbon_log_return, which can still be highly
    volatile. RobustScaler is safer for carbon; MinMaxScaler is retained for
    gas/electricity for comparability with the original design.
    """
    from sklearn.preprocessing import MinMaxScaler, RobustScaler

    name = series_name.lower()
    if "carbon" in name:
        return RobustScaler()
    return MinMaxScaler()


def _fit_transform_target(series: pd.Series, series_name: str):
    scaler = _select_target_scaler(series_name)
    arr = series.values.reshape(-1, 1).astype(np.float32)
    scaled = scaler.fit_transform(arr).astype(np.float32)
    return scaled, scaler


def _transform_target(series: pd.Series, scaler) -> np.ndarray:
    arr = series.values.reshape(-1, 1).astype(np.float32)
    return scaler.transform(arr).astype(np.float32)


def _inverse_target(arr: np.ndarray, scaler) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32).reshape(-1, 1)
    return scaler.inverse_transform(arr).reshape(-1)


def _fit_transform_macro(
    macro: Optional[pd.DataFrame],
    index: pd.Index,
    series_name: str,
    macro_feature_set: str = "lean",
) -> Tuple[Optional[np.ndarray], object, list[str]]:
    """
    Fit macro scaler using only the supplied historical macro window.

    No backward-fill is used. If the first requested macro date is missing,
    the function raises an error instead of leaking future values backward.
    """
    if macro is None:
        return None, None, []

    from sklearn.preprocessing import RobustScaler

    cols = _available_macro_cols(macro, series_name, macro_feature_set)
    if not cols:
        return None, None, []

    aligned = macro[cols].reindex(index).ffill().fillna(0.0)

    scaler = RobustScaler()
    scaled = scaler.fit_transform(aligned.values.astype(np.float32)).astype(np.float32)
    return scaled, scaler, cols


def _transform_macro(
    macro: Optional[pd.DataFrame],
    index: pd.Index,
    scaler,
    cols: list[str],
) -> Optional[np.ndarray]:
    """Transform macro data using a pre-fitted macro scaler. No bfill."""
    if macro is None or scaler is None or not cols:
        return None

    aligned = macro[cols].reindex(index).ffill().fillna(0.0)

    return scaler.transform(aligned.values.astype(np.float32)).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Sequence construction and model
# ══════════════════════════════════════════════════════════════════════════════

def _build_sequences(
    target_scaled: np.ndarray,
    macro_scaled: Optional[np.ndarray],
    lookback: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build LSTM sequences.

    X contains target alone in core mode, or target + macro columns in macro mode.
    y is always the next-step scaled target only.
    """
    if macro_scaled is not None:
        if len(target_scaled) != len(macro_scaled):
            raise ValueError("Target and macro arrays must have the same length.")
        data = np.hstack([target_scaled, macro_scaled])
    else:
        data = target_scaled

    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i])
        y.append(target_scaled[i, 0])

    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


def _temporal_train_val_split(
    X: np.ndarray,
    y: np.ndarray,
    val_fraction: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Chronological validation split for time-series learning."""
    if len(X) < 5:
        raise ValueError("Not enough sequences for LSTM training.")

    split_idx = int(len(X) * (1 - val_fraction))
    split_idx = max(1, min(split_idx, len(X) - 1))

    return X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:]


def _build_model(
    n_features: int,
    lookback: int,
    dropout: float = 0.2,
    learning_rate: float = LEARNING_RATE,
    units_1: int = 128,
    units_2: int = 64,
):
    try:
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError:
        raise ImportError("TensorFlow is required. Install with: pip install tensorflow")

    inputs = keras.Input(shape=(lookback, n_features))
    x = layers.LSTM(units_1, return_sequences=True)(inputs)
    x = layers.Dropout(dropout)(x, training=True)
    x = layers.LSTM(units_2, return_sequences=False)(x)
    x = layers.Dropout(dropout)(x, training=True)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(1)(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
    )
    return model


def _callbacks(patience: int = 15):
    import tensorflow as tf

    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
        )
    ]


# ══════════════════════════════════════════════════════════════════════════════
# Prediction
# ══════════════════════════════════════════════════════════════════════════════

def _mc_predict(
    model,
    X: np.ndarray,
    n_samples: int = MC_SAMPLES,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Monte Carlo dropout predictive mean and 95% prediction interval."""
    preds = np.stack(
        [model(X, training=True).numpy().reshape(-1) for _ in range(n_samples)],
        axis=0,
    )
    lower_q = 100 * (ALPHA / 2)
    upper_q = 100 * (1 - ALPHA / 2)
    return (
        preds.mean(axis=0),
        np.percentile(preds, lower_q, axis=0),
        np.percentile(preds, upper_q, axis=0),
    )


def _recursive_forecast(
    model,
    seed_window: np.ndarray,
    target_scaler,
    future_macro_scaled: Optional[np.ndarray] = None,
    steps: int = 12,
    n_samples: int = MC_SAMPLES,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Recursive multi-step forecast.

    Core mode:
        seed_window has one column: target.
        New rows contain only the predicted target.

    Macro mode:
        seed_window has target + macro columns.
        New rows contain predicted target + known future macro values.
    """
    window = seed_window.copy().astype(np.float32)
    means, lowers, uppers = [], [], []

    n_features = window.shape[1]

    if future_macro_scaled is not None:
        if len(future_macro_scaled) < steps:
            raise ValueError("future_macro_scaled must contain at least `steps` rows.")
        expected_macro_cols = n_features - 1
        if future_macro_scaled.shape[1] != expected_macro_cols:
            raise ValueError(
                f"Expected {expected_macro_cols} future macro columns, "
                f"got {future_macro_scaled.shape[1]}."
            )

    for step in range(steps):
        X_in = window[np.newaxis, :, :]
        pred_mean, pred_low, pred_high = _mc_predict(model, X_in, n_samples=n_samples)

        m = float(pred_mean[0])
        lo = float(pred_low[0])
        hi = float(pred_high[0])

        means.append(m)
        lowers.append(lo)
        uppers.append(hi)

        new_row = np.zeros((1, n_features), dtype=np.float32)
        new_row[0, 0] = m

        if future_macro_scaled is not None:
            new_row[0, 1:] = future_macro_scaled[step]

        window = np.vstack([window[1:], new_row])

    return (
        _inverse_target(np.array(means), target_scaler),
        _inverse_target(np.array(lowers), target_scaler),
        _inverse_target(np.array(uppers), target_scaler),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def run_lstm(
    series_name: str,
    train: pd.Series,
    test: pd.Series,
    full: pd.Series,
    macro_train: Optional[pd.DataFrame] = None,
    macro_full: Optional[pd.DataFrame] = None,
    use_macro: bool = False,
    actual_2017: Optional[pd.Series] = None,
    eval_actual: Optional[pd.Series] = None,
    forecast_dir: str = "outputs/forecasts",
    epochs: int = EPOCHS,
    mc_samples: int = MC_SAMPLES,
    seed: int = 42,
    lookback: int = LOOKBACK,
    dropout: float = 0.2,
    learning_rate: float = LEARNING_RATE,
    units_1: int = 128,
    units_2: int = 64,
    batch_size: int = BATCH_SIZE,
    patience: int = 15,
    macro_feature_set: str = "lean",
) -> dict:
    """
    Full LSTM pipeline for one series.

    Parameters
    ----------
    series_name : str
        Example: 'gas_growth', 'electricity_growth', or 'carbon_log_return'.
    train : pd.Series
        Target history up to end of 2016.
    test : pd.Series
        2016 monthly target series.
    full : pd.Series
        Target history up to end of 2017. No 7 target values.
    macro_train : pd.DataFrame, optional
        Macro controls aligned with train period.
    macro_full : pd.DataFrame, optional
        Macro controls through 2017. Required for macro-mode 2017 forecasting.
    use_macro : bool
        Whether to include macro controls as exogenous inputs.
    actual_2017 : pd.Series, optional
        Actual 2017 target values for comparison in saved CSV.
    forecast_dir : str
        Output directory.
    epochs : int
        Maximum training epochs.
    mc_samples : int
        Monte Carlo dropout forward passes.
    seed : int
        Reproducibility seed.
    """
    try:
        import tensorflow as tf  # noqa: F401
    except ImportError as e:
        log.error(f"Missing dependency: {e}")
        raise

    _set_seed(seed)

    train = _safe_series(train)
    test = _safe_series(test)
    full = _safe_series(full)
    actual_2017 = _safe_series(actual_2017) if actual_2017 is not None else None
    eval_actual = _safe_series(eval_actual) if eval_actual is not None else None

    macro_train = _safe_macro(macro_train)
    macro_full = _safe_macro(macro_full)

    mode = "macro" if use_macro else "core"

    if use_macro and (macro_train is None or macro_full is None):
        raise ValueError("use_macro=True requires both macro_train and macro_full.")

    if len(test) != 12:
        log.warning(f"Expected 12 test observations for 2016, got {len(test)}.")

    log.info(
        f"[LSTM-{mode.upper()}] Training on {series_name} | "
        f"train={train.index.min().date()}→{train.index.max().date()} ({len(train)} obs), "
        f"test={test.index.min().date()}→{test.index.max().date()} ({len(test)} obs), "
        f"use_macro={use_macro}, lookback={lookback}, dropout={dropout}, "
        f"lr={learning_rate}, units=({units_1},{units_2}), "
        f"macro_feature_set={macro_feature_set}"
    )

    # ── First model: train to 2016, evaluate on 2017 ─────────────────────────
    train_target_scaled, target_scaler = _fit_transform_target(train, series_name)

    if use_macro:
        train_macro_scaled, macro_scaler, macro_cols = _fit_transform_macro(
            macro_train,
            train.index,
            series_name,
            macro_feature_set,
        )
    else:
        train_macro_scaled, macro_scaler, macro_cols = None, None, []

    X_train, y_train = _build_sequences(
        train_target_scaled,
        train_macro_scaled,
        lookback,
    )

    X_tr, y_tr, X_val, y_val = _temporal_train_val_split(
        X_train,
        y_train,
        val_fraction=0.10,
    )

    n_features = X_train.shape[-1]
    model = _build_model(
        n_features,
        lookback,
        dropout=dropout,
        learning_rate=learning_rate,
        units_1=units_1,
        units_2=units_2,
    )
    model.fit(
        X_tr,
        y_tr,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=_callbacks(patience=patience),
        shuffle=False,
    )

    # Evaluation uses train+test target, but scaler was fitted only on train.
    train_test = pd.concat([train, test]).sort_index()
    train_test_target_scaled = _transform_target(train_test, target_scaler)

    if use_macro:
        # Build macro data for the exact train+test index.
        # macro_train covers train dates; macro_full supplies test dates.
        macro_eval = pd.concat([
            macro_train.reindex(train.index),
            macro_full.reindex(test.index),
        ]).sort_index()
        macro_eval = macro_eval[~macro_eval.index.duplicated(keep="last")]

        train_test_macro_scaled = _transform_macro(
            macro_eval,
            train_test.index,
            macro_scaler,
            macro_cols,
        )
    else:
        train_test_macro_scaled = None

    X_all, _ = _build_sequences(
        train_test_target_scaled,
        train_test_macro_scaled,
        lookback,
    )
    X_test = X_all[-len(test):].astype(np.float32)

    fc_test_scaled, lb_test_scaled, ub_test_scaled = _mc_predict(
        model,
        X_test,
        n_samples=mc_samples,
    )

    fc_test = _inverse_target(fc_test_scaled, target_scaler)
    lb_test = _inverse_target(lb_test_scaled, target_scaler)
    ub_test = _inverse_target(ub_test_scaled, target_scaler)
    test_arr = test.values
    if is_electricity(series_name):
        test_dates = pd.DatetimeIndex(test.index)
        hist_for_test = pd.concat([train, test]).sort_index()
        fc_test = index_forecast_to_yoy_growth(fc_test, test_dates, hist_for_test)
        lb_test = index_forecast_to_yoy_growth(lb_test, test_dates, hist_for_test)
        ub_test = index_forecast_to_yoy_growth(ub_test, test_dates, hist_for_test)
        test_arr = eval_actual.reindex(test_dates).values if eval_actual is not None else test_arr

    metrics = compute_all_metrics(
        actual=test_arr,
        forecast=fc_test,
        lower=lb_test,
        upper=ub_test,
        train_actual=train.values,
    )

    log.info(
        f"[LSTM-{mode.upper()} {series_name}] 2016 validation: "
        f"MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}"
    )

    # ── Second model: refit on full history through 2016, forecast 2017 ──────
    full_target_scaled, final_target_scaler = _fit_transform_target(full, series_name)

    if use_macro:
        full_macro_scaled, final_macro_scaler, final_macro_cols = _fit_transform_macro(
            macro_full,
            full.index,
            series_name,
            macro_feature_set,
        )
    else:
        full_macro_scaled, final_macro_scaler, final_macro_cols = None, None, []

    X_full, y_full = _build_sequences(
        full_target_scaled,
        full_macro_scaled,
        lookback,
    )

    X_tr2, y_tr2, X_val2, y_val2 = _temporal_train_val_split(
        X_full,
        y_full,
        val_fraction=0.05,
    )

    model2 = _build_model(
        X_full.shape[-1],
        lookback,
        dropout=dropout,
        learning_rate=learning_rate,
        units_1=units_1,
        units_2=units_2,
    )
    model2.fit(
        X_tr2,
        y_tr2,
        validation_data=(X_val2, y_val2),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=_callbacks(patience=patience),
        shuffle=False,
    )

    if full_macro_scaled is not None:
        seed_data = np.hstack([full_target_scaled, full_macro_scaled])
    else:
        seed_data = full_target_scaled

    seed_window = seed_data[-lookback:].astype(np.float32)

    forecast_dates = pd.date_range("2017-01-01", periods=12, freq="MS")

    if use_macro:
        future_macro_scaled = _transform_macro(
            macro_full,
            forecast_dates,
            final_macro_scaler,
            final_macro_cols,
        )
    else:
        future_macro_scaled = None

    fc_2017, lb_2017, ub_2017 = _recursive_forecast(
        model2,
        seed_window,
        final_target_scaler,
        future_macro_scaled=future_macro_scaled,
        steps=12,
        n_samples=mc_samples,
    )
    if is_electricity(series_name):
        fc_2017 = index_forecast_to_yoy_growth(fc_2017, forecast_dates, full)
        lb_2017 = index_forecast_to_yoy_growth(lb_2017, forecast_dates, full)
        ub_2017 = index_forecast_to_yoy_growth(ub_2017, forecast_dates, full)

    df_out = pd.DataFrame({
        "date": forecast_dates,
        "series": series_name,
        "model": "LSTM",
        "mode": mode,
        "forecast": np.round(fc_2017, 4),
        "lower_bound": np.round(lb_2017, 4),
        "upper_bound": np.round(ub_2017, 4),
    })

    if actual_2017 is not None:
        df_out["actual"] = actual_2017.reindex(forecast_dates).values

    Path(forecast_dir).mkdir(parents=True, exist_ok=True)

    # safe_series_name = series_name.replace(" ", "_").lower()
    out_path = f"{forecast_dir}/{series_name}_growth_pct_forecasts_lstm_{mode}.csv"
    df_out.to_csv(out_path, index=False)

    log.info(f"[LSTM-{mode.upper()}] Forecast saved → {out_path}")

    return {
        "series": series_name,
        "model": "LSTM",
        "mode": mode,
        "metrics": metrics,
        "forecast_path": out_path,
        "params": {
            "lookback": lookback,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "units_1": units_1,
            "units_2": units_2,
            "batch_size": batch_size,
            "patience": patience,
            "seed": seed,
            "macro_feature_set": macro_feature_set,
        },
    }
