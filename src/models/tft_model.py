"""
tft_model.py
────────────
Temporal Fusion Transformer (TFT) forecasting.

Architecture summary
────────────────────
  Variable Selection Network  → selects informative input features
  Gated Residual Network (GRN) → non-linear feature processing
  LSTM encoder–decoder         → capture sequential dependencies
  Interpretable Multi-Head Attention → long-range patterns
  Quantile output heads        → 0.025, 0.5, 0.975  (95 % PI)

Implementation
──────────────
  Primary  : pytorch-forecasting TemporalFusionTransformer
  Fallback : lightweight PyTorch attention-LSTM that mirrors TFT outputs
             (activated automatically if pytorch-forecasting is not installed)

Forecast modes
──────────────
  core  — recursive forecast using predicted target only; macro cols = 0.
  macro — recursive forecast using predicted target + actual 2018 macro values
          (inflation_growth, weather_volatility, gdp_growth) at every step.
          Real macro values are scaled via the training z-score normaliser and
          fed into position 1+ of the input row, never zeroed out.

Output filename: {series}_growth_pct_forecasts_tft_core.csv
                 {series}_growth_pct_forecasts_tft_macro.csv
"""

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple

from src.logging_utils import get_logger
from src.metrics_utils import compute_all_metrics

log = get_logger("tft")
warnings.filterwarnings("ignore")

LOOKBACK       = 12
FORECAST_STEPS = 12
QUANTILES      = [0.025, 0.5, 0.975]
ALPHA          = 0.05
MACRO_COLS     = ["inflation_growth", "weather_volatility", "gdp_growth"]


# ════════════════════════════════════════════════════════════════════════════════
# Lightweight fallback: Attention-LSTM  (pure PyTorch, no extra deps)
# ════════════════════════════════════════════════════════════════════════════════

def _try_import_torch():
    try:
        import torch
        import torch.nn as nn
        return torch, nn
    except ImportError:
        raise ImportError("PyTorch is required. pip install torch")


class _AttentionLSTM:
    """
    Minimal TFT-inspired model:
      LSTM encoder → scaled dot-product attention → quantile MLP heads
    """

    def __init__(self, n_features: int, hidden: int = 64, n_heads: int = 4):
        torch, nn = _try_import_torch()
        self.torch = torch
        self.nn    = nn

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm    = nn.LSTM(n_features, hidden, num_layers=2,
                                       batch_first=True, dropout=0.2)
                self.attn    = nn.MultiheadAttention(hidden, n_heads, dropout=0.1,
                                                     batch_first=True)
                self.fc_q025 = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))
                self.fc_q50  = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))
                self.fc_q975 = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

            def forward(self, x):
                out, _  = self.lstm(x)
                ctx, _  = self.attn(out, out, out)
                last    = ctx[:, -1, :]
                return self.fc_q025(last), self.fc_q50(last), self.fc_q975(last)

        torch.manual_seed(42)
        self.model = Model()
        self.optim = torch.optim.Adam(self.model.parameters(), lr=5e-4)
        self.scaler_mean: Optional[np.ndarray] = None
        self.scaler_std:  Optional[np.ndarray] = None

    def _quantile_loss(self, pred, target, q):
        err = target - pred
        return self.torch.mean(self.torch.where(err >= 0, q * err, (q - 1) * err))

    def _normalise(self, arr: np.ndarray) -> np.ndarray:
        self.scaler_mean = arr.mean(axis=0)
        self.scaler_std  = arr.std(axis=0) + 1e-8
        return (arr - self.scaler_mean) / self.scaler_std

    def _inv(self, scaled_col: np.ndarray) -> np.ndarray:
        return scaled_col * self.scaler_std[0] + self.scaler_mean[0]

    def _make_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for i in range(LOOKBACK, len(data)):
            X.append(data[i - LOOKBACK:i])
            y.append(data[i, 0])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def fit(self, arr: np.ndarray, epochs: int = 120, batch_size: int = 16):
        torch = self.torch
        scaled = self._normalise(arr)
        X, y   = self._make_sequences(scaled)

        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X), torch.tensor(y).unsqueeze(1)
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                self.optim.zero_grad()
                q025, q50, q975 = self.model(xb)
                loss = (self._quantile_loss(q025, yb, 0.025)
                      + self._quantile_loss(q50,  yb, 0.5)
                      + self._quantile_loss(q975, yb, 0.975))
                loss.backward()
                self.torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optim.step()
                epoch_loss += loss.item()
            if (epoch + 1) % 20 == 0:
                log.debug(f"Epoch {epoch+1}/{epochs}  loss={epoch_loss/len(loader):.5f}")

    def predict(self, X_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        torch = self.torch
        self.model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_arr.astype(np.float32))
            q025, q50, q975 = self.model(X_t)
        return (
            self._inv(q025.numpy().flatten()),
            self._inv(q50.numpy().flatten()),
            self._inv(q975.numpy().flatten()),
        )

    def rolling_forecast(
        self,
        seed_window: np.ndarray,
        steps: int = 12,
        future_macro_scaled: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Recursive rolling forecast.

        Parameters
        ----------
        seed_window          : z-score scaled seed window, shape (LOOKBACK, n_features)
        steps                : number of steps to forecast (12 for 2018)
        future_macro_scaled  : z-score scaled 2018 macro values, shape (steps, n_macro).
                               When provided, each step's macro columns are filled with
                               the actual 2018 values instead of zeros (macro mode).
                               When None, macro columns stay zero (core mode).
        """
        torch = self.torch
        self.model.eval()
        window = seed_window.copy()
        means, lowers, uppers = [], [], []

        with torch.no_grad():
            for step in range(steps):
                x = torch.tensor(window[np.newaxis].astype(np.float32))
                q025, q50, q975 = self.model(x)
                lo  = self._inv(q025.numpy().flatten())[0]
                med = self._inv(q50.numpy().flatten())[0]
                hi  = self._inv(q975.numpy().flatten())[0]

                means.append(med)
                lowers.append(lo)
                uppers.append(hi)

                new_row = np.zeros((1, window.shape[1]))
                new_row[0, 0] = (med - self.scaler_mean[0]) / self.scaler_std[0]
                if future_macro_scaled is not None and step < len(future_macro_scaled):
                    new_row[0, 1:] = future_macro_scaled[step]
                window = np.vstack([window[1:], new_row])

        return np.array(means), np.array(lowers), np.array(uppers)


# ════════════════════════════════════════════════════════════════════════════════
# Primary TFT via pytorch-forecasting (when available)
# ════════════════════════════════════════════════════════════════════════════════

def _run_pytorch_forecasting(
    series_name: str,
    train: pd.Series,
    test: pd.Series,
    full: pd.Series,
    macro_full: Optional[pd.DataFrame],
    use_macro: bool,
    actual_2018: Optional[pd.Series],
    forecast_dir: str,
    epochs: int,
) -> dict:
    """Use the official pytorch-forecasting TFT."""
    from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
    from pytorch_forecasting.metrics import QuantileLoss as PTQLoss
    import pytorch_lightning as pl
    import torch

    mode = "macro" if use_macro else "core"
    log.info(f"[TFT/pytorch-forecasting] Fitting {series_name} ({mode})")

    series_col = f"{series_name}_value"
    df_train = pd.DataFrame({
        "time_idx": range(len(train)),
        "group":    series_name,
        series_col: train.values,
    })
    df_full = pd.DataFrame({
        "time_idx": range(len(full)),
        "group":    series_name,
        series_col: full.values,
    })

    time_varying_known_reals = []
    if use_macro and macro_full is not None:
        macro_aligned_full  = macro_full.reindex(full.index).ffill().bfill()
        macro_aligned_train = macro_full.reindex(train.index).ffill().bfill()
        cols_avail = [c for c in MACRO_COLS if c in macro_aligned_full.columns]
        for col in cols_avail:
            df_full[col]  = macro_aligned_full[col].values
            df_train[col] = macro_aligned_train[col].values
        time_varying_known_reals = cols_avail

    max_encoder_length    = LOOKBACK
    max_prediction_length = 12
    training_cutoff       = len(train) - max_prediction_length

    training = TimeSeriesDataSet(
        df_train[df_train.time_idx <= training_cutoff],
        time_idx="time_idx",
        target=series_col,
        group_ids=["group"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        time_varying_unknown_reals=[series_col],
        time_varying_known_reals=time_varying_known_reals,
        target_normalizer=__import__("pytorch_forecasting.data").data.encoders.GroupNormalizer(
            groups=["group"], transformation=None
        ),
    )
    val = TimeSeriesDataSet.from_dataset(training, df_train, predict=True, stop_randomization=True)

    train_dl = training.to_dataloader(train=True,  batch_size=32, num_workers=0)
    val_dl   = val.to_dataloader(     train=False, batch_size=32, num_workers=0)

    tft_model = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=1e-3,
        lstm_layers=2,
        hidden_size=32,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=16,
        loss=PTQLoss(quantiles=QUANTILES),
        log_interval=-1,
    )
    trainer = pl.Trainer(
        max_epochs=epochs,
        enable_model_summary=False,
        enable_progress_bar=False,
        logger=False,
    )
    trainer.fit(tft_model, train_dataloaders=train_dl, val_dataloaders=val_dl)

    raw_preds = tft_model.predict(val_dl, mode="quantiles", return_x=False)
    q025 = raw_preds[:, :, 0].numpy().flatten()[:12]
    q50  = raw_preds[:, :, 1].numpy().flatten()[:12]
    q975 = raw_preds[:, :, 2].numpy().flatten()[:12]

    metrics = compute_all_metrics(
        actual=test.values, forecast=q50, lower=q025, upper=q975,
        train_actual=train.values,
    )
    log.info(f"[TFT {series_name}] Test MAE={metrics['MAE']:.4f}")

    forecast_dates = pd.date_range("2018-01-01", periods=12, freq="MS")
    df_out = pd.DataFrame({
        "date":        forecast_dates,
        "model":       "TFT",
        "mode":        mode,
        "forecast":    q50[:12].round(4),
        "lower_bound": q025[:12].round(4),
        "upper_bound": q975[:12].round(4),
    })
    if actual_2018 is not None:
        df_out["actual"] = actual_2018.reindex(forecast_dates).values

    Path(forecast_dir).mkdir(parents=True, exist_ok=True)
    out_path = f"{forecast_dir}/{series_name}_growth_pct_forecasts_tft_{mode}.csv"
    df_out.to_csv(out_path, index=False)
    log.info(f"[TFT] Forecast saved → {out_path}")

    return {"series": series_name, "model": "TFT", "mode": mode,
            "metrics": metrics, "forecast_path": out_path}


# ════════════════════════════════════════════════════════════════════════════════
# Public API
# ════════════════════════════════════════════════════════════════════════════════

def run_tft(
    series_name: str,
    train: pd.Series,
    test: pd.Series,
    full: pd.Series,
    macro_train: Optional[pd.DataFrame] = None,
    macro_full: Optional[pd.DataFrame]  = None,
    use_macro: bool = False,
    actual_2018: Optional[pd.Series] = None,
    forecast_dir: str = "outputs/forecasts",
    epochs: int = 80,
) -> dict:
    """
    Full TFT pipeline for one series.

    Tries pytorch-forecasting first; falls back to the lightweight
    Attention-LSTM if that library is not installed.

    Parameters
    ----------
    series_name  : 'gas', 'electricity', or 'carbon'
    train        : history up to 2016
    test         : 2017 monthly series
    full         : history up to end of 2017 (no 2018 data)
    macro_*      : Dataset B; macro_full must include 2018 rows for macro mode
    use_macro    : pass macro vars as known-future covariates
    actual_2018  : actual 2018 target values for comparison column in CSV
    forecast_dir : output directory
    epochs       : training epochs
    """
    mode = "macro" if use_macro else "core"

    try:
        import pytorch_forecasting   # noqa
        return _run_pytorch_forecasting(
            series_name, train, test, full,
            macro_full if use_macro else None,
            use_macro,
            actual_2018,
            forecast_dir, epochs,
        )
    except ImportError:
        log.warning("pytorch-forecasting not installed; using fallback Attention-LSTM (TFT-inspired)")
    except Exception as e:
        log.warning(f"pytorch-forecasting TFT failed ({e}); falling back to Attention-LSTM")

    # ── Fallback: Attention-LSTM ──────────────────────────────────────────────
    log.info(f"[TFT-Fallback-{mode.upper()}] Training on {series_name} "
             f"(train={len(train)}, use_macro={use_macro})")

    def _build_arr(series: pd.Series, macro: Optional[pd.DataFrame]) -> np.ndarray:
        arr = series.values.reshape(-1, 1).astype(np.float64)
        if use_macro and macro is not None:
            cols = [c for c in MACRO_COLS if c in macro.columns]
            m    = macro[cols].reindex(series.index).ffill().bfill().values
            arr  = np.hstack([arr, m])
        return arr

    train_arr = _build_arr(train, macro_train)
    full_arr  = _build_arr(full,  macro_full)
    n_feats   = train_arr.shape[1]

    # Train on history up to 2016 for evaluation
    model = _AttentionLSTM(n_feats)
    model.fit(train_arr, epochs=epochs)

    scaled_train  = (train_arr - model.scaler_mean) / model.scaler_std
    seed_eval     = scaled_train[-LOOKBACK:]
    fc_test, lb_test, ub_test = model.rolling_forecast(seed_eval, steps=12)

    metrics = compute_all_metrics(
        actual=test.values, forecast=fc_test,
        lower=lb_test, upper=ub_test,
        train_actual=train.values,
    )
    log.info(f"[TFT-Fallback-{mode.upper()} {series_name}] 2017 eval: "
             f"MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}")

    # Refit on full history up to 2017
    model2 = _AttentionLSTM(n_feats)
    model2.fit(full_arr, epochs=epochs)

    scaled_full = (full_arr - model2.scaler_mean) / model2.scaler_std
    seed_2018   = scaled_full[-LOOKBACK:]

    future_macro_scaled = None
    if use_macro and macro_full is not None and n_feats > 1:
        macro_2018_dates = pd.date_range("2018-01-01", periods=12, freq="MS")
        cols_avail = [c for c in MACRO_COLS if c in macro_full.columns]
        macro_2018_vals = (
            macro_full[cols_avail]
            .reindex(macro_2018_dates)
            .ffill().bfill()
            .values.astype(np.float64)
        )
        # Z-score scale using the model2 scaler (columns 1+ correspond to macro)
        n_macro = len(cols_avail)
        future_macro_scaled = (
            (macro_2018_vals - model2.scaler_mean[1:1 + n_macro])
            / model2.scaler_std[1:1 + n_macro]
        )

    fc_2018, lb_2018, ub_2018 = model2.rolling_forecast(
        seed_2018, steps=12, future_macro_scaled=future_macro_scaled
    )

    forecast_dates = pd.date_range("2018-01-01", periods=12, freq="MS")
    df_out = pd.DataFrame({
        "date":        forecast_dates,
        "model":       "TFT",
        "mode":        mode,
        "forecast":    fc_2018.round(4),
        "lower_bound": lb_2018.round(4),
        "upper_bound": ub_2018.round(4),
    })
    if actual_2018 is not None:
        df_out["actual"] = actual_2018.reindex(forecast_dates).values

    Path(forecast_dir).mkdir(parents=True, exist_ok=True)
    out_path = f"{forecast_dir}/{series_name}_growth_pct_forecasts_tft_{mode}.csv"
    df_out.to_csv(out_path, index=False)
    log.info(f"[TFT-Fallback] Forecast saved → {out_path}")

    return {
        "series":        series_name,
        "model":         "TFT",
        "mode":          mode,
        "metrics":       metrics,
        "forecast_path": out_path,
    }
