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
  macro — recursive forecast using predicted target plus lagged, series-specific
          macro values. Gas uses gas-futures signals; electricity uses demand
          signals.

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
from src.model_utils import (
    align_macro_for_series,
    index_forecast_to_yoy_growth,
    is_electricity,
    macro_cols_for_series,
)

log = get_logger("tft")
warnings.filterwarnings("ignore")

LOOKBACK       = 12
FORECAST_STEPS = 12
QUANTILES      = [0.025, 0.5, 0.975]
ALPHA          = 0.05
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

    def __init__(
        self,
        n_features: int,
        hidden: int = 64,
        n_heads: int = 4,
        learning_rate: float = 5e-4,
        dropout: float = 0.2,
        attention_dropout: float = 0.1,
        lookback: int = LOOKBACK,
    ):
        torch, nn = _try_import_torch()
        self.torch = torch
        self.nn    = nn
        self.lookback = lookback

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm    = nn.LSTM(n_features, hidden, num_layers=2,
                                       batch_first=True, dropout=dropout)
                self.attn    = nn.MultiheadAttention(hidden, n_heads, dropout=attention_dropout,
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
        self.optim = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
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
        for i in range(self.lookback, len(data)):
            X.append(data[i - self.lookback:i])
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
        steps                : number of steps to forecast (12 for 2017)
        future_macro_scaled  : z-score scaled 2017 macro values, shape (steps, n_macro).
                               When provided, each step's macro columns are filled with
                               the actual 2017 values instead of zeros (macro mode).
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

def _build_tft_dataset(
    series_name: str, values: pd.Series, macro_full: Optional[pd.DataFrame],
    use_macro: bool, series_col: str, extra_time_idx: int = 0,
    future_dates: Optional[pd.DatetimeIndex] = None, last_value: float = 0.0,
) -> tuple:
    """
    Build a pytorch-forecasting-ready dataframe for `values`' date range,
    optionally extended by `extra_time_idx` additional future rows (target
    filled with `last_value` as a structural placeholder -- an "unknown
    real" is never read by the model as an input, only used to define the
    dataset's shape, so the placeholder value itself doesn't leak
    information). Returns (df, time_varying_known_reals).
    """
    n = len(values)
    df = pd.DataFrame({
        "time_idx": range(n + extra_time_idx),
        "group":    series_name,
        series_col: list(values.values) + [last_value] * extra_time_idx,
    })
    time_varying_known_reals: list = []
    if use_macro and macro_full is not None:
        full_index = values.index
        if extra_time_idx and future_dates is not None:
            full_index = full_index.append(future_dates)
        macro_aligned = align_macro_for_series(macro_full, full_index, series_name)
        cols_avail = macro_cols_for_series(series_name, macro_aligned)
        for col in cols_avail:
            df[col] = macro_aligned[col].values
        time_varying_known_reals = cols_avail
    return df, time_varying_known_reals


def _fit_tft(df_train_slice: pd.DataFrame, series_col: str, time_varying_known_reals: list,
             epochs: int) -> tuple:
    """Fit a fresh TemporalFusionTransformer on df_train_slice; returns (model, training_dataset)."""
    from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
    from pytorch_forecasting.data.encoders import GroupNormalizer
    from pytorch_forecasting.metrics import QuantileLoss as PTQLoss
    # pytorch_forecasting's TemporalFusionTransformer subclasses
    # lightning.pytorch.LightningModule (the current "lightning" package),
    # NOT the separate legacy "pytorch_lightning" package -- using the
    # latter's Trainer against this model raises "model must be a
    # LightningModule ... got TemporalFusionTransformer" since the two
    # packages define distinct, non-interchangeable LightningModule
    # classes despite the near-identical API. Verified empirically against
    # the installed pytorch-forecasting 1.8.0 / lightning 2.6.5.
    import lightning.pytorch as pl

    training = TimeSeriesDataSet(
        df_train_slice,
        time_idx="time_idx",
        target=series_col,
        group_ids=["group"],
        max_encoder_length=LOOKBACK,
        max_prediction_length=FORECAST_STEPS,
        time_varying_unknown_reals=[series_col],
        time_varying_known_reals=time_varying_known_reals,
        target_normalizer=GroupNormalizer(groups=["group"], transformation=None),
    )
    train_dl = training.to_dataloader(train=True, batch_size=32, num_workers=0)

    model = TemporalFusionTransformer.from_dataset(
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
        enable_checkpointing=False,
        logger=False,
    )
    trainer.fit(model, train_dataloaders=train_dl)
    return model, training


def _predict_tft(model, training_dataset, df_predict_from: pd.DataFrame) -> np.ndarray:
    """Predict quantiles for the last max_prediction_length rows of
    df_predict_from (its encoder = everything before that)."""
    from pytorch_forecasting import TimeSeriesDataSet

    predict_ds = TimeSeriesDataSet.from_dataset(
        training_dataset, df_predict_from, predict=True, stop_randomization=True,
    )
    predict_dl = predict_ds.to_dataloader(train=False, batch_size=32, num_workers=0)
    raw_preds = model.predict(predict_dl, mode="quantiles", return_x=False)
    return raw_preds[0].numpy()  # (prediction_length, n_quantiles)


def _run_pytorch_forecasting(
    series_name: str,
    train: pd.Series,
    test: pd.Series,
    full: pd.Series,
    macro_full: Optional[pd.DataFrame],
    use_macro: bool,
    actual_target: Optional[pd.Series],
    eval_actual: Optional[pd.Series],
    forecast_dir: str,
    epochs: int,
) -> dict:
    """
    Use the official pytorch-forecasting TFT, fit TWICE like every other
    model in this pipeline (SARIMA/Prophet/LSTM): once on `train` to
    validate against the genuinely held-out `test` period, then refit on
    `full` (train+test) to forecast the genuinely unknown 12 months beyond
    it. An earlier version of this function fit once on `train` alone and
    reused that SAME in-sample-tail prediction, mislabeled, as both the
    validation score AND the final forecast -- it never saw `test` or
    forecast anything beyond `train`'s own last 12 months. Fixed here.
    """
    mode = "macro" if use_macro else "core"
    log.info(f"[TFT/pytorch-forecasting] Fitting {series_name} ({mode})")

    series_col = f"{series_name}_value"

    # ── Step 1: fit on train, evaluate on the REAL held-out test period ──────
    eval_series = pd.concat([train, test])
    df_eval, kr_eval = _build_tft_dataset(series_name, eval_series, macro_full, use_macro, series_col)
    train_cutoff = len(train) - 1
    eval_model, eval_training_ds = _fit_tft(
        df_eval[df_eval.time_idx <= train_cutoff], series_col, kr_eval, epochs,
    )
    eval_quantiles = _predict_tft(eval_model, eval_training_ds, df_eval)
    q025_eval, q50_eval, q975_eval = eval_quantiles[:, 0], eval_quantiles[:, 1], eval_quantiles[:, 2]

    test_arr = test.values
    if is_electricity(series_name):
        test_dates = pd.DatetimeIndex(test.index)
        hist_for_test = eval_series.sort_index()
        q025_eval = index_forecast_to_yoy_growth(q025_eval, test_dates, hist_for_test)
        q50_eval = index_forecast_to_yoy_growth(q50_eval, test_dates, hist_for_test)
        q975_eval = index_forecast_to_yoy_growth(q975_eval, test_dates, hist_for_test)
        test_arr = eval_actual.reindex(test_dates).values if eval_actual is not None else test_arr

    metrics = compute_all_metrics(
        actual=test_arr, forecast=q50_eval, lower=q025_eval, upper=q975_eval,
        train_actual=train.values,
    )
    log.info(f"[TFT {series_name}] validation: MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}")

    # ── Step 2: refit on full (train+test), forecast genuinely beyond it ─────
    forecast_dates = pd.date_range(full.index.max() + pd.DateOffset(months=1), periods=FORECAST_STEPS, freq="MS")
    df_full, kr_full = _build_tft_dataset(
        series_name, full, macro_full, use_macro, series_col,
        extra_time_idx=FORECAST_STEPS, future_dates=forecast_dates, last_value=float(full.iloc[-1]),
    )
    final_cutoff = len(full) - 1
    final_model, final_training_ds = _fit_tft(
        df_full[df_full.time_idx <= final_cutoff], series_col, kr_full, epochs,
    )
    final_quantiles = _predict_tft(final_model, final_training_ds, df_full)
    q025_out, q50_out, q975_out = final_quantiles[:, 0], final_quantiles[:, 1], final_quantiles[:, 2]

    if is_electricity(series_name):
        q025_out = index_forecast_to_yoy_growth(q025_out, forecast_dates, full)
        q50_out = index_forecast_to_yoy_growth(q50_out, forecast_dates, full)
        q975_out = index_forecast_to_yoy_growth(q975_out, forecast_dates, full)

    df_out = pd.DataFrame({
        "date":        forecast_dates,
        "model":       "TFT",
        "mode":        mode,
        "forecast":    q50_out[:FORECAST_STEPS].round(4),
        "lower_bound": q025_out[:FORECAST_STEPS].round(4),
        "upper_bound": q975_out[:FORECAST_STEPS].round(4),
    })
    if actual_target is not None:
        df_out["actual"] = actual_target.reindex(forecast_dates).values

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
    actual_target: Optional[pd.Series] = None,
    eval_actual: Optional[pd.Series] = None,
    forecast_dir: str = "outputs_v2/forecasts",
    epochs: int = 80,
    hidden: int = 64,
    n_heads: int = 4,
    learning_rate: float = 5e-4,
    dropout: float = 0.2,
    attention_dropout: float = 0.1,
    batch_size: int = 16,
    lookback: int = LOOKBACK,
    save_dir: Optional[str] = None,
) -> dict:
    """
    Full TFT pipeline for one series.

    Tries pytorch-forecasting first; falls back to the lightweight
    Attention-LSTM if that library is not installed.

    Parameters
    ----------
    series_name  : 'gas', 'electricity', or 'carbon'
    train        : history up to 2015
    test         : 2016 monthly series
    full         : history up to end of 2017 (no 2017 data)
    macro_*      : Dataset B; macro_full must include 2017 rows for macro mode
    use_macro    : pass macro vars as known-future covariates
    actual_target  : actual 2017 target values for comparison column in CSV
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
            actual_target,
            eval_actual,
            forecast_dir, epochs,
        )
    except ImportError:
        log.warning("pytorch-forecasting not installed; using fallback Attention-LSTM (TFT-inspired)")
    except Exception as e:
        log.warning(f"pytorch-forecasting TFT failed ({e}); falling back to Attention-LSTM")

    # ── Fallback: Attention-LSTM ──────────────────────────────────────────────
    log.info(f"[TFT-Fallback-{mode.upper()}] Training on {series_name} "
             f"(train={len(train)}, use_macro={use_macro}, lookback={lookback}, "
             f"hidden={hidden}, heads={n_heads}, lr={learning_rate})")

    def _build_arr(series: pd.Series, macro: Optional[pd.DataFrame]) -> np.ndarray:
        arr = series.values.reshape(-1, 1).astype(np.float64)
        if use_macro and macro is not None:
            cols = macro_cols_for_series(series_name, macro)
            m    = align_macro_for_series(macro, series.index, series_name).values
            arr  = np.hstack([arr, m])
        return arr

    train_arr = _build_arr(train, macro_train)
    full_arr  = _build_arr(full,  macro_full)
    n_feats   = train_arr.shape[1]

    # Train on history up to 2016 for evaluation
    model = _AttentionLSTM(
        n_feats,
        hidden=hidden,
        n_heads=n_heads,
        learning_rate=learning_rate,
        dropout=dropout,
        attention_dropout=attention_dropout,
        lookback=lookback,
    )
    model.fit(train_arr, epochs=epochs, batch_size=batch_size)

    scaled_train = (train_arr - model.scaler_mean) / model.scaler_std
    seed_eval    = scaled_train[-lookback:]

    # Supply actual 2016 macro values during evaluation — consistent with how
    # the 2017 forecast is generated (zeroing macro columns would corrupt eval).
    eval_macro_scaled = None
    if use_macro and macro_full is not None and n_feats > 1:
        n_macro = n_feats - 1
        macro_eval_vals = align_macro_for_series(
            macro_full, pd.DatetimeIndex(test.index), series_name
        ).values.astype(np.float64)
        eval_macro_scaled = (
            (macro_eval_vals - model.scaler_mean[1:1 + n_macro])
            / model.scaler_std[1:1 + n_macro]
        )

    fc_test, lb_test, ub_test = model.rolling_forecast(seed_eval, steps=12,
                                                        future_macro_scaled=eval_macro_scaled)
    test_arr = test.values
    if is_electricity(series_name):
        test_dates = pd.DatetimeIndex(test.index)
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
    log.info(f"[TFT-Fallback-{mode.upper()} {series_name}] validation: "
             f"MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}")

    # Refit on full history through 2016
    model2 = _AttentionLSTM(
        n_feats,
        hidden=hidden,
        n_heads=n_heads,
        learning_rate=learning_rate,
        dropout=dropout,
        attention_dropout=attention_dropout,
        lookback=lookback,
    )
    model2.fit(full_arr, epochs=epochs, batch_size=batch_size)

    scaled_full = (full_arr - model2.scaler_mean) / model2.scaler_std
    seed_2017   = scaled_full[-lookback:]

    future_macro_scaled = None
    if use_macro and macro_full is not None and n_feats > 1:
        macro_2017_dates = pd.date_range(full.index.max() + pd.DateOffset(months=1), periods=12, freq="MS")
        cols_avail = macro_cols_for_series(series_name, macro_full)
        macro_2017_vals = align_macro_for_series(
            macro_full, macro_2017_dates, series_name
        ).values.astype(np.float64)
        # Z-score scale using the model2 scaler (columns 1+ correspond to macro)
        n_macro = len(cols_avail)
        future_macro_scaled = (
            (macro_2017_vals - model2.scaler_mean[1:1 + n_macro])
            / model2.scaler_std[1:1 + n_macro]
        )

    fc_2017, lb_2017, ub_2017 = model2.rolling_forecast(
        seed_2017, steps=12, future_macro_scaled=future_macro_scaled
    )

    forecast_dates = pd.date_range(full.index.max() + pd.DateOffset(months=1), periods=12, freq="MS")
    if is_electricity(series_name):
        fc_2017 = index_forecast_to_yoy_growth(fc_2017, forecast_dates, full)
        lb_2017 = index_forecast_to_yoy_growth(lb_2017, forecast_dates, full)
        ub_2017 = index_forecast_to_yoy_growth(ub_2017, forecast_dates, full)

    df_out = pd.DataFrame({
        "date":        forecast_dates,
        "model":       "TFT",
        "mode":        mode,
        "forecast":    fc_2017.round(4),
        "lower_bound": lb_2017.round(4),
        "upper_bound": ub_2017.round(4),
    })
    if actual_target is not None:
        df_out["actual"] = actual_target.reindex(forecast_dates).values

    Path(forecast_dir).mkdir(parents=True, exist_ok=True)
    out_path = f"{forecast_dir}/{series_name}_growth_pct_forecasts_tft_{mode}.csv"
    df_out.to_csv(out_path, index=False)
    log.info(f"[TFT-Fallback] Forecast saved → {out_path}")

    # ── Optionally save model for SHAP ────────────────────────────────────────
    if save_dir is not None:
        try:
            import torch
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            n_feats2 = full_arr.shape[1]  # model2 is trained on full_arr
            torch.save(
                {
                    "state_dict":   model2.model.state_dict(),
                    "n_features":   n_feats2,
                    "hidden":       hidden,
                    "n_heads":      n_heads,
                    "dropout":      dropout,
                    "scaler_mean":  model2.scaler_mean.tolist() if model2.scaler_mean is not None else None,
                    "scaler_std":   model2.scaler_std.tolist()  if model2.scaler_std  is not None else None,
                },
                f"{save_dir}/{series_name}_tft_{mode}.pt",
            )
            log.info(f"[TFT-Fallback] Model saved → {save_dir}")
        except Exception as _e:
            log.warning(f"[TFT-Fallback] Model save failed: {_e}")

    return {
        "series":        series_name,
        "model":         "TFT",
        "mode":          mode,
        "metrics":       metrics,
        "forecast_path": out_path,
        "params": {
            "hidden": hidden,
            "n_heads": n_heads,
            "learning_rate": learning_rate,
            "dropout": dropout,
            "attention_dropout": attention_dropout,
            "batch_size": batch_size,
            "lookback": lookback,
        },
    }
