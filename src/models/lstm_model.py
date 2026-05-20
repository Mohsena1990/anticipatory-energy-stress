"""
lstm_model.py
─────────────
Univariate + optional multivariate LSTM forecasting.

Architecture
────────────
  Input  → Lookback window of 12 months
  Body   → LSTM(128) → Dropout(0.2) → LSTM(64) → Dropout(0.2)
  Output → Dense(1) for point forecast
  PI     → Monte Carlo Dropout (N=200 forward passes at inference time)

Strategy
────────
  1. Normalise with sklearn MinMaxScaler per series.
  2. Build sliding-window sequences (lookback=12).
  3. Train on available history up to 2016; evaluate on 2017.
  4. Refit on full history up to 2017; forecast Jan-Dec 2018.

  Two forecast modes:
    core  — recursive, appending only the predicted target; exogenous = 0.
    macro — recursive, appending predicted target PLUS the known 2018 macro
            values (inflation_growth, weather_volatility, gdp_growth) at each
            step.  Zeros are never used for macro in this mode.

  5. Save forecast CSV with actual 2018 values for comparison.
     Filename suffix: _lstm_core.csv  or  _lstm_macro.csv
"""

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple

from src.logging_utils import get_logger
from src.metrics_utils import compute_all_metrics

log = get_logger("lstm")
warnings.filterwarnings("ignore")

LOOKBACK       = 12
MC_SAMPLES     = 200
EPOCHS         = 100
BATCH_SIZE     = 16
LEARNING_RATE  = 1e-3
ALPHA          = 0.05

MACRO_COLS = ["inflation_growth", "weather_volatility", "gdp_growth"]


def _build_sequences(data: np.ndarray, lookback: int) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i])
        y.append(data[i])
    return np.array(X), np.array(y)


def _build_model(n_features: int, lookback: int):
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError:
        raise ImportError("TensorFlow is required. pip install tensorflow")

    tf.random.set_seed(42)
    inputs = keras.Input(shape=(lookback, n_features))
    x = layers.LSTM(128, return_sequences=True)(inputs)
    x = layers.Dropout(0.2)(x, training=True)
    x = layers.LSTM(64, return_sequences=False)(x)
    x = layers.Dropout(0.2)(x, training=True)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(LEARNING_RATE), loss="mse")
    return model


def _mc_predict(model, X: np.ndarray, n_samples: int = MC_SAMPLES) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    preds = np.stack([model(X, training=True).numpy().flatten() for _ in range(n_samples)], axis=0)
    return preds.mean(axis=0), np.percentile(preds, 2.5, axis=0), np.percentile(preds, 97.5, axis=0)


def _recursive_forecast_core(
    model,
    seed_window: np.ndarray,
    scaler,
    steps: int = 12,
    n_samples: int = MC_SAMPLES,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Core-only recursive forecast.
    Non-target columns (macro) are set to zero in each appended row.
    """
    window = seed_window.copy()
    means, lowers, uppers = [], [], []

    for _ in range(steps):
        X_in = window[np.newaxis, :, :]
        step_preds = np.stack(
            [model(X_in, training=True).numpy()[0, 0] for _ in range(n_samples)]
        )
        m  = step_preds.mean()
        lo = np.percentile(step_preds, 2.5)
        hi = np.percentile(step_preds, 97.5)
        means.append(m); lowers.append(lo); uppers.append(hi)

        new_row = np.zeros((1, window.shape[1]))
        new_row[0, 0] = m
        window = np.vstack([window[1:], new_row])

    def inv(arr):
        dummy = np.zeros((len(arr), scaler.scale_.shape[0]))
        dummy[:, 0] = arr
        return scaler.inverse_transform(dummy)[:, 0]

    return inv(np.array(means)), inv(np.array(lowers)), inv(np.array(uppers))


def _recursive_forecast_macro(
    model,
    seed_window: np.ndarray,
    future_macro_scaled: np.ndarray,
    scaler,
    steps: int = 12,
    n_samples: int = MC_SAMPLES,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Macro-augmented recursive forecast.
    At each step, the new row contains:
      column 0 : predicted target (scaled)
      columns 1+: actual 2018 macro values for that month (scaled via training scaler)
    This is the correct approach — macro inputs are never zeroed out.
    """
    window = seed_window.copy()
    means, lowers, uppers = [], [], []

    for step in range(steps):
        X_in = window[np.newaxis, :, :]
        step_preds = np.stack(
            [model(X_in, training=True).numpy()[0, 0] for _ in range(n_samples)]
        )
        m  = step_preds.mean()
        lo = np.percentile(step_preds, 2.5)
        hi = np.percentile(step_preds, 97.5)
        means.append(m); lowers.append(lo); uppers.append(hi)

        new_row = np.zeros((1, window.shape[1]))
        new_row[0, 0] = m
        if step < len(future_macro_scaled):
            new_row[0, 1:] = future_macro_scaled[step]
        window = np.vstack([window[1:], new_row])

    def inv(arr):
        dummy = np.zeros((len(arr), scaler.scale_.shape[0]))
        dummy[:, 0] = arr
        return scaler.inverse_transform(dummy)[:, 0]

    return inv(np.array(means)), inv(np.array(lowers)), inv(np.array(uppers))


def run_lstm(
    series_name: str,
    train: pd.Series,
    test: pd.Series,
    full: pd.Series,
    macro_train: Optional[pd.DataFrame] = None,
    macro_full: Optional[pd.DataFrame]  = None,
    use_macro: bool = False,
    actual_2018: Optional[pd.Series] = None,
    forecast_dir: str = "outputs/forecasts",
    epochs: int = EPOCHS,
) -> dict:
    """
    Full LSTM pipeline for one series.

    Parameters
    ----------
    series_name  : 'gas', 'electricity', or 'carbon'
    train        : history up to end of 2016
    test         : 2017 monthly series (12 obs)
    full         : history up to end of 2017 (for final refit — no 2018 data)
    macro_train/full : Dataset B (optional exogenous); must include 2018 rows
                       in macro_full so 2018 macro values can be extracted.
    use_macro    : include macro vars as additional input features
    actual_2018  : actual 2018 target values for comparison column in CSV
    forecast_dir : output directory
    epochs       : training epochs
    """
    try:
        import tensorflow as tf
        from sklearn.preprocessing import MinMaxScaler
    except ImportError as e:
        log.error(f"Missing dependency: {e}")
        raise

    mode = "macro" if use_macro else "core"
    log.info(f"[LSTM-{mode.upper()}] Training on {series_name} "
             f"(train={len(train)}, use_macro={use_macro})")

    def _build_X(series: pd.Series, macro: Optional[pd.DataFrame]) -> np.ndarray:
        arr = series.values.reshape(-1, 1).astype(np.float32)
        if use_macro and macro is not None:
            cols = [c for c in MACRO_COLS if c in macro.columns]
            macro_arr = macro[cols].reindex(series.index).ffill().bfill().values
            arr = np.hstack([arr, macro_arr.astype(np.float32)])
        return arr

    train_arr = _build_X(train, macro_train)
    full_arr  = _build_X(full,  macro_full)
    n_features = train_arr.shape[1]

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_arr)
    full_scaled  = scaler.transform(full_arr)

    X_train, y_train = _build_sequences(train_scaled, LOOKBACK)
    X_train = X_train.astype(np.float32)
    y_train = y_train[:, 0].astype(np.float32)

    tf.random.set_seed(42)
    model = _build_model(n_features, LOOKBACK)
    model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        validation_split=0.1,
        verbose=0,
        callbacks=[
            __import__("tensorflow").keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=15, restore_best_weights=True
            )
        ],
    )

    # ── Evaluate on 2017 ─────────────────────────────────────────────────────
    if use_macro and macro_train is not None and macro_full is not None:
        combined_macro = pd.concat([macro_train, macro_full.reindex(test.index)])
    else:
        combined_macro = None

    train_test_scaled = scaler.transform(
        _build_X(pd.concat([train, test]), combined_macro)
    )
    X_all, _ = _build_sequences(train_test_scaled, LOOKBACK)
    X_test = X_all[-12:].astype(np.float32)
    fc_test_scaled, lb_test_scaled, ub_test_scaled = _mc_predict(model, X_test)

    def inv_col0(arr):
        dummy = np.zeros((len(arr), n_features))
        dummy[:, 0] = arr
        return scaler.inverse_transform(dummy)[:, 0]

    fc_test = inv_col0(fc_test_scaled)
    lb_test = inv_col0(lb_test_scaled)
    ub_test = inv_col0(ub_test_scaled)

    metrics = compute_all_metrics(
        actual=test.values, forecast=fc_test,
        lower=lb_test, upper=ub_test,
        train_actual=train.values,
    )
    log.info(f"[LSTM-{mode.upper()} {series_name}] 2017 eval: "
             f"MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}")

    # ── Refit on full 2005-2017 → forecast 2018 ──────────────────────────────
    X_full, y_full = _build_sequences(full_scaled, LOOKBACK)
    X_full = X_full.astype(np.float32)
    y_full = y_full[:, 0].astype(np.float32)

    model2 = _build_model(n_features, LOOKBACK)
    model2.fit(
        X_full, y_full,
        epochs=epochs,
        batch_size=BATCH_SIZE,
        validation_split=0.05,
        verbose=0,
        callbacks=[
            __import__("tensorflow").keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=15, restore_best_weights=True
            )
        ],
    )

    seed_window = full_scaled[-LOOKBACK:]

    if use_macro and macro_full is not None:
        # Extract and scale actual 2018 macro values using the training scaler.
        # MinMaxScaler scales each column independently, so using a dummy zero
        # for the target column is valid — we only use the macro columns.
        macro_2018_dates = pd.date_range("2018-01-01", periods=12, freq="MS")
        cols_avail = [c for c in MACRO_COLS if c in macro_full.columns]
        macro_2018_vals = (
            macro_full[cols_avail]
            .reindex(macro_2018_dates)
            .ffill().bfill()
            .values.astype(np.float32)
        )
        n_macro = len(cols_avail)
        dummy = np.zeros((12, n_features), dtype=np.float32)
        dummy[:, 1:1 + n_macro] = macro_2018_vals
        dummy_scaled = scaler.transform(dummy)
        future_macro_scaled = dummy_scaled[:, 1:1 + n_macro]

        fc_2018, lb_2018, ub_2018 = _recursive_forecast_macro(
            model2, seed_window, future_macro_scaled, scaler, steps=12
        )
    else:
        fc_2018, lb_2018, ub_2018 = _recursive_forecast_core(
            model2, seed_window, scaler, steps=12
        )

    forecast_dates = pd.date_range("2018-01-01", periods=12, freq="MS")
    df_out = pd.DataFrame({
        "date":        forecast_dates,
        "model":       "LSTM",
        "mode":        mode,
        "forecast":    fc_2018.round(4),
        "lower_bound": lb_2018.round(4),
        "upper_bound": ub_2018.round(4),
    })

    if actual_2018 is not None:
        df_out["actual"] = actual_2018.reindex(forecast_dates).values

    Path(forecast_dir).mkdir(parents=True, exist_ok=True)
    out_path = f"{forecast_dir}/{series_name}_growth_pct_forecasts_lstm_{mode}.csv"
    df_out.to_csv(out_path, index=False)
    log.info(f"[LSTM-{mode.upper()}] Forecast saved → {out_path}")

    return {
        "series":        series_name,
        "model":         "LSTM",
        "mode":          mode,
        "metrics":       metrics,
        "forecast_path": out_path,
    }
