

"""
lstm_model.py
─────────────
Leakage-safe univariate + optional macro-augmented LSTM forecasting.

Architecture
────────────
  Input  → Lookback window of 12 months
  Body   → LSTM(128) → Dropout(0.2) → LSTM(64) → Dropout(0.2)
  Output → Linear(32) → ReLU → Linear(1)
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
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════════════════
# Data utilities
# ══════════════════════════════════════════════════════════════════════════════

def _safe_series(series: pd.Series) -> pd.Series:
    s = series.copy()
    s.index = pd.to_datetime(s.index)
    return s.sort_index().dropna()


def _safe_macro(macro: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
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
    if macro is None or scaler is None or not cols:
        return None
    aligned = macro[cols].reindex(index).ffill().fillna(0.0)
    return scaler.transform(aligned.values.astype(np.float32)).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Sequence construction
# ══════════════════════════════════════════════════════════════════════════════

def _build_sequences(
    target_scaled: np.ndarray,
    macro_scaled: Optional[np.ndarray],
    lookback: int,
) -> Tuple[np.ndarray, np.ndarray]:
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
    if len(X) < 5:
        raise ValueError("Not enough sequences for LSTM training.")
    split_idx = int(len(X) * (1 - val_fraction))
    split_idx = max(1, min(split_idx, len(X) - 1))
    return X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:]


# ══════════════════════════════════════════════════════════════════════════════
# PyTorch model
# ══════════════════════════════════════════════════════════════════════════════

class _LSTMNet:
    """Thin wrapper so the rest of the code uses model(X) and model.train() like Keras."""

    def __init__(self, n_features, lookback, dropout=0.2, units_1=128, units_2=64):
        import torch
        import torch.nn as nn

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm1 = nn.LSTM(n_features, units_1, batch_first=True)
                self.drop1 = nn.Dropout(dropout)
                self.lstm2 = nn.LSTM(units_1, units_2, batch_first=True)
                self.drop2 = nn.Dropout(dropout)
                self.fc1   = nn.Linear(units_2, 32)
                self.relu  = nn.ReLU()
                self.fc2   = nn.Linear(32, 1)

            def forward(self, x):
                out, _ = self.lstm1(x)
                out = self.drop1(out)
                out, _ = self.lstm2(out)
                out = out[:, -1, :]  # last timestep
                out = self.drop2(out)
                out = self.relu(self.fc1(out))
                return self.fc2(out)

        self._net = _Net()
        self._torch = torch
        self._nn = nn

    def train(self):
        self._net.train()

    def eval(self):
        self._net.eval()

    def state_dict(self):
        return self._net.state_dict()

    def load_state_dict(self, state):
        self._net.load_state_dict(state)

    def parameters(self):
        return self._net.parameters()

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Forward pass; always uses current train/eval mode of inner net."""
        import torch
        x_t = torch.tensor(X.astype(np.float32))
        with torch.no_grad():
            out = self._net(x_t)
        return out.detach().numpy()


def _build_model(
    n_features: int,
    lookback: int,
    dropout: float = 0.2,
    learning_rate: float = LEARNING_RATE,
    units_1: int = 128,
    units_2: int = 64,
) -> _LSTMNet:
    return _LSTMNet(n_features, lookback, dropout=dropout, units_1=units_1, units_2=units_2)


def _fit_model(
    model: _LSTMNet,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int,
) -> None:
    import torch
    import torch.nn as nn

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    X_tr_t  = torch.tensor(X_tr)
    y_tr_t  = torch.tensor(y_tr).unsqueeze(1)
    X_val_t = torch.tensor(X_val)
    y_val_t = torch.tensor(y_val).unsqueeze(1)

    dataset = torch.utils.data.TensorDataset(X_tr_t, y_tr_t)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    best_val  = float("inf")
    best_state = None
    wait = 0

    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model._net(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model._net(X_val_t), y_val_t).item()

        if val_loss < best_val:
            best_val  = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)


# ══════════════════════════════════════════════════════════════════════════════
# Prediction
# ══════════════════════════════════════════════════════════════════════════════

def _mc_predict(
    model: _LSTMNet,
    X: np.ndarray,
    n_samples: int = MC_SAMPLES,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Monte Carlo dropout: keep dropout active during inference."""
    model.train()  # dropout stays on
    preds = np.stack(
        [model(X).reshape(-1) for _ in range(n_samples)],
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
    model: _LSTMNet,
    seed_window: np.ndarray,
    target_scaler,
    future_macro_scaled: Optional[np.ndarray] = None,
    steps: int = 12,
    n_samples: int = MC_SAMPLES,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

        m  = float(pred_mean[0])
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
    actual_target: Optional[pd.Series] = None,
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
    save_dir: Optional[str] = None,
) -> dict:
    """Full LSTM pipeline for one series (PyTorch backend)."""
    try:
        import torch  # noqa: F401
    except ImportError as e:
        log.error(f"Missing dependency: {e}")
        raise ImportError("PyTorch is required. Install with: pip install torch") from e

    _set_seed(seed)

    train       = _safe_series(train)
    test        = _safe_series(test)
    full        = _safe_series(full)
    actual_target = _safe_series(actual_target) if actual_target is not None else None
    eval_actual = _safe_series(eval_actual) if eval_actual is not None else None
    macro_train = _safe_macro(macro_train)
    macro_full  = _safe_macro(macro_full)

    mode = "macro" if use_macro else "core"

    if use_macro and (macro_train is None or macro_full is None):
        raise ValueError("use_macro=True requires both macro_train and macro_full.")

    if len(test) != 12:
        log.warning(f"Expected 12 test (validation) observations, got {len(test)}.")

    log.info(
        f"[LSTM-{mode.upper()}] Training on {series_name} | "
        f"train={train.index.min().date()}→{train.index.max().date()} ({len(train)} obs), "
        f"test={test.index.min().date()}→{test.index.max().date()} ({len(test)} obs), "
        f"use_macro={use_macro}, lookback={lookback}, dropout={dropout}, "
        f"lr={learning_rate}, units=({units_1},{units_2}), "
        f"macro_feature_set={macro_feature_set}"
    )

    # ── First model: train-to-2016, evaluate on 2016 ─────────────────────────
    train_target_scaled, target_scaler = _fit_transform_target(train, series_name)

    if use_macro:
        train_macro_scaled, macro_scaler, macro_cols = _fit_transform_macro(
            macro_train, train.index, series_name, macro_feature_set,
        )
    else:
        train_macro_scaled, macro_scaler, macro_cols = None, None, []

    X_train, y_train = _build_sequences(train_target_scaled, train_macro_scaled, lookback)
    X_tr, y_tr, X_val, y_val = _temporal_train_val_split(X_train, y_train, val_fraction=0.10)

    n_features = X_train.shape[-1]
    model = _build_model(n_features, lookback, dropout=dropout,
                         learning_rate=learning_rate, units_1=units_1, units_2=units_2)
    _fit_model(model, X_tr, y_tr, X_val, y_val,
               epochs=epochs, batch_size=batch_size,
               learning_rate=learning_rate, patience=patience)

    train_test = pd.concat([train, test]).sort_index()
    train_test_target_scaled = _transform_target(train_test, target_scaler)

    if use_macro:
        macro_eval = pd.concat([
            macro_train.reindex(train.index),
            macro_full.reindex(test.index),
        ]).sort_index()
        macro_eval = macro_eval[~macro_eval.index.duplicated(keep="last")]
        train_test_macro_scaled = _transform_macro(macro_eval, train_test.index, macro_scaler, macro_cols)
    else:
        train_test_macro_scaled = None

    X_all, _ = _build_sequences(train_test_target_scaled, train_test_macro_scaled, lookback)
    X_test = X_all[-len(test):].astype(np.float32)

    fc_test_scaled, lb_test_scaled, ub_test_scaled = _mc_predict(model, X_test, n_samples=mc_samples)

    fc_test = _inverse_target(fc_test_scaled, target_scaler)
    lb_test = _inverse_target(lb_test_scaled, target_scaler)
    ub_test = _inverse_target(ub_test_scaled, target_scaler)
    test_arr = test.values

    if is_electricity(series_name):
        test_dates = pd.DatetimeIndex(test.index)
        hist_for_test = pd.concat([train, test]).sort_index()
        fc_test  = index_forecast_to_yoy_growth(fc_test, test_dates, hist_for_test)
        lb_test  = index_forecast_to_yoy_growth(lb_test, test_dates, hist_for_test)
        ub_test  = index_forecast_to_yoy_growth(ub_test, test_dates, hist_for_test)
        test_arr = eval_actual.reindex(test_dates).values if eval_actual is not None else test_arr

    metrics = compute_all_metrics(
        actual=test_arr,
        forecast=fc_test,
        lower=lb_test,
        upper=ub_test,
        train_actual=train.values,
    )

    log.info(
        f"[LSTM-{mode.upper()} {series_name}] validation: "
        f"MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}"
    )

    # ── Second model: refit on full history through 2016, forecast 2017 ──────
    full_target_scaled, final_target_scaler = _fit_transform_target(full, series_name)

    if use_macro:
        full_macro_scaled, final_macro_scaler, final_macro_cols = _fit_transform_macro(
            macro_full, full.index, series_name, macro_feature_set,
        )
    else:
        full_macro_scaled, final_macro_scaler, final_macro_cols = None, None, []

    X_full, y_full = _build_sequences(full_target_scaled, full_macro_scaled, lookback)
    X_tr2, y_tr2, X_val2, y_val2 = _temporal_train_val_split(X_full, y_full, val_fraction=0.05)

    model2 = _build_model(X_full.shape[-1], lookback, dropout=dropout,
                          learning_rate=learning_rate, units_1=units_1, units_2=units_2)
    _fit_model(model2, X_tr2, y_tr2, X_val2, y_val2,
               epochs=epochs, batch_size=batch_size,
               learning_rate=learning_rate, patience=patience)

    if full_macro_scaled is not None:
        seed_data = np.hstack([full_target_scaled, full_macro_scaled])
    else:
        seed_data = full_target_scaled

    seed_window = seed_data[-lookback:].astype(np.float32)
    forecast_dates = pd.date_range(full.index.max() + pd.DateOffset(months=1), periods=12, freq="MS")

    if use_macro:
        future_macro_scaled = _transform_macro(
            macro_full, forecast_dates, final_macro_scaler, final_macro_cols,
        )
    else:
        future_macro_scaled = None

    fc_2017, lb_2017, ub_2017 = _recursive_forecast(
        model2, seed_window, final_target_scaler,
        future_macro_scaled=future_macro_scaled,
        steps=12, n_samples=mc_samples,
    )

    if is_electricity(series_name):
        fc_2017 = index_forecast_to_yoy_growth(fc_2017, forecast_dates, full)
        lb_2017 = index_forecast_to_yoy_growth(lb_2017, forecast_dates, full)
        ub_2017 = index_forecast_to_yoy_growth(ub_2017, forecast_dates, full)

    df_out = pd.DataFrame({
        "date":        forecast_dates,
        "series":      series_name,
        "model":       "LSTM",
        "mode":        mode,
        "forecast":    np.round(fc_2017, 4),
        "lower_bound": np.round(lb_2017, 4),
        "upper_bound": np.round(ub_2017, 4),
    })

    if actual_target is not None:
        df_out["actual"] = actual_target.reindex(forecast_dates).values

    Path(forecast_dir).mkdir(parents=True, exist_ok=True)
    out_path = f"{forecast_dir}/{series_name}_growth_pct_forecasts_lstm_{mode}.csv"
    df_out.to_csv(out_path, index=False)
    log.info(f"[LSTM-{mode.upper()}] Forecast saved → {out_path}")

    # ── Optionally save model and scalers for SHAP ────────────────────────────
    if save_dir is not None:
        try:
            import torch, joblib
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            _n_feat = X_full.shape[-1]
            torch.save(
                {
                    "state_dict": model2.state_dict(),
                    "n_features": _n_feat,
                    "lookback":   lookback,
                    "units_1":    units_1,
                    "units_2":    units_2,
                    "dropout":    dropout,
                },
                f"{save_dir}/{series_name}_lstm_{mode}.pt",
            )
            joblib.dump(
                final_target_scaler,
                f"{save_dir}/{series_name}_lstm_{mode}_target_scaler.pkl",
            )
            if use_macro and final_macro_scaler is not None:
                joblib.dump(
                    final_macro_scaler,
                    f"{save_dir}/{series_name}_lstm_{mode}_macro_scaler.pkl",
                )
                joblib.dump(
                    final_macro_cols,
                    f"{save_dir}/{series_name}_lstm_{mode}_macro_cols.pkl",
                )
            log.info(f"[LSTM-{mode.upper()}] Model saved → {save_dir}")
        except Exception as _e:
            log.warning(f"[LSTM-{mode.upper()}] Model save failed: {_e}")

    return {
        "series":        series_name,
        "model":         "LSTM",
        "mode":          mode,
        "metrics":       metrics,
        "forecast_path": out_path,
        "params": {
            "lookback":          lookback,
            "dropout":           dropout,
            "learning_rate":     learning_rate,
            "units_1":           units_1,
            "units_2":           units_2,
            "batch_size":        batch_size,
            "patience":          patience,
            "seed":              seed,
            "macro_feature_set": macro_feature_set,
        },
    }
