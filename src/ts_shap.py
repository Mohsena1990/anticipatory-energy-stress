"""
ts_shap.py
──────────
SHAP-style feature attribution for the time-series forecasting models.

For each (series, mode) combination where a model has been selected, computes
feature importance / attribution appropriate to that model class:

  SARIMA / SARIMAX — exogenous variable coefficient attribution
  Prophet          — native component decomposition (trend + seasonality + regressors)
  LSTM             — gradient × input attribution over the lookback window
  TFT              — attention weight attribution over time steps

Outputs (per best model stream):
  outputs/shap/ts/{series}_{model}_{mode}_attribution.csv
  outputs/shap/ts/{series}_{model}_{mode}_attribution.png

Entry point
-----------
  from src.ts_shap import run_ts_shap
  run_ts_shap(best_models, results, core_full, macro_full, out_dir, figures_dir)

where
  best_models : dict  {(series, mode): model_name}
  results     : list  of dicts returned by run_* model functions
  core_full   : pd.DataFrame  full core dataset (2005-2017)
  macro_full  : pd.DataFrame  full macro dataset
"""

from __future__ import annotations

import warnings
import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
log = logging.getLogger("ts_shap")

FORECAST_DATES = pd.date_range("2017-01-01", periods=12, freq="MS")
MONTHS_SHORT   = ["Jan","Feb","Mar","Apr","May","Jun",
                  "Jul","Aug","Sep","Oct","Nov","Dec"]
_PALETTE = {"grid": "#EAECEE"}
_DPI = 150


# ══════════════════════════════════════════════════════════════════════════════
# I/O helpers
# ══════════════════════════════════════════════════════════════════════════════

def _ensure(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _save_fig(fig: plt.Figure, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("SHAP figure → %s", path)


def _save_csv(df: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info("SHAP CSV → %s", path)


# ══════════════════════════════════════════════════════════════════════════════
# Generic bar-chart for feature attribution
# ══════════════════════════════════════════════════════════════════════════════

def _plot_attribution_bar(
    importance_df: pd.DataFrame,
    title: str,
    out_path: str,
    top_n: int = 15,
) -> None:
    """Horizontal bar chart — mean absolute attribution, coloured by direction."""
    if importance_df.empty:
        return
    top = importance_df.head(top_n).copy()
    colours = ["#E74C3C" if v >= 0 else "#2980B9" for v in top["mean_attribution"]]
    fig, ax = plt.subplots(figsize=(9, max(4, min(top_n, len(top)) * 0.45)))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.barh(top["feature"][::-1], top["mean_abs_attribution"][::-1],
            color=colours[::-1], alpha=0.85)
    ax.set_xlabel("Mean |attribution| (impact on forecast)")
    ax.set_title(title)
    ax.grid(axis="x", color=_PALETTE["grid"], linewidth=0.8)
    plt.tight_layout()
    _save_fig(fig, out_path)


def _plot_attribution_heatmap(
    attr_df: pd.DataFrame,
    feature_col: str,
    value_col: str,
    month_col: str,
    title: str,
    out_path: str,
) -> None:
    """Heatmap of attribution over 2017 months × features."""
    if attr_df.empty:
        return
    try:
        pivot = attr_df.pivot(index=feature_col, columns=month_col, values=value_col)
        fig, ax = plt.subplots(figsize=(13, max(4, len(pivot) * 0.4)))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        vabs = np.nanmax(np.abs(pivot.values))
        im = ax.imshow(pivot.values, cmap="RdBu_r", aspect="auto",
                       vmin=-vabs, vmax=vabs)
        plt.colorbar(im, ax=ax, label="Attribution")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_yticks(range(len(pivot.index)))
        ax.set_xticklabels(
            [MONTHS_SHORT[m-1] if 1 <= m <= 12 else str(m)
             for m in pivot.columns],
            fontsize=8
        )
        ax.set_yticklabels(pivot.index, fontsize=8)
        ax.set_title(title, fontsize=11, fontweight="bold")
        plt.tight_layout()
        _save_fig(fig, out_path)
    except Exception as e:
        log.warning("Attribution heatmap failed: %s", e)


# ══════════════════════════════════════════════════════════════════════════════
# SARIMA / SARIMAX — exogenous coefficient attribution
# ══════════════════════════════════════════════════════════════════════════════

def shap_sarima(
    series_name: str,
    mode: str,
    fitted_model,        # statsmodels SARIMAXResults
    macro_full: Optional[pd.DataFrame],
    macro_cols: list,
    out_dir: str,
    figures_dir: str,
) -> Optional[pd.DataFrame]:
    """
    Exogenous variable attribution for SARIMAX macro mode.

    Attribution of variable j at time t:
      attr_j,t = beta_j × (X_j,t − X_j_mean_train)

    For core mode (no exogenous), skips gracefully.
    """
    if mode == "core" or not macro_cols or macro_full is None:
        log.info("[SARIMA-SHAP] Core mode — no exogenous features; skipping.")
        return None

    try:
        params = fitted_model.params
    except Exception as e:
        log.warning("[SARIMA-SHAP] Cannot access model params: %s", e)
        return None

    # Extract exogenous coefficients (last len(macro_cols) params)
    exog_params = {}
    for col in macro_cols:
        if col in params.index:
            exog_params[col] = float(params[col])

    if not exog_params:
        log.warning("[SARIMA-SHAP] No exogenous params found in model.")
        return None

    # Align macro values for 2017
    macro_2017 = macro_full.reindex(FORECAST_DATES).ffill().fillna(0.0)
    # Training mean per feature (for centring)
    from src.fes_calculator import TRAIN_START, TRAIN_END
    if not isinstance(macro_full.index, pd.DatetimeIndex):
        macro_full = macro_full.copy()
        macro_full.index = pd.to_datetime(macro_full.index)
    macro_train = macro_full[
        (macro_full.index >= TRAIN_START) & (macro_full.index <= TRAIN_END)
    ]

    rows = []
    for month_i, date in enumerate(FORECAST_DATES):
        for col, beta in exog_params.items():
            if col not in macro_2017.columns:
                continue
            x_t      = float(macro_2017.loc[date, col]) if date in macro_2017.index else np.nan
            x_mean   = float(macro_train[col].mean()) if col in macro_train.columns else 0.0
            attr_val = beta * (x_t - x_mean) if not np.isnan(x_t) else np.nan
            rows.append({
                "series":      series_name,
                "mode":        mode,
                "month":       month_i + 1,
                "date":        date,
                "feature":     col,
                "beta":        round(beta, 6),
                "x_t":         round(x_t, 5) if not np.isnan(x_t) else np.nan,
                "x_mean":      round(x_mean, 5),
                "attribution": round(attr_val, 6) if not np.isnan(attr_val) else np.nan,
            })

    if not rows:
        return None

    attr_df = pd.DataFrame(rows)
    out_base = f"{out_dir}/{series_name}_sarima_{mode}"
    _save_csv(attr_df, f"{out_base}_attribution.csv")

    # Importance summary
    imp = (
        attr_df.groupby("feature")["attribution"]
        .agg(mean_abs_attribution=lambda x: x.abs().mean(),
             mean_attribution="mean")
        .reset_index()
        .sort_values("mean_abs_attribution", ascending=False)
    )
    _save_csv(imp, f"{out_base}_importance.csv")

    _plot_attribution_bar(
        imp,
        title=(f"SARIMA Macro — Exogenous Attribution\n"
               f"{series_name.capitalize()} 2017 (β × ΔX, mean over months)"),
        out_path=f"{figures_dir}/{series_name}_sarima_{mode}_attribution.png",
    )
    _plot_attribution_heatmap(
        attr_df, "feature", "attribution", "month",
        title=f"SARIMA Macro Attribution Heatmap — {series_name.capitalize()} 2017",
        out_path=f"{figures_dir}/{series_name}_sarima_{mode}_heatmap.png",
    )

    log.info("[SARIMA-SHAP] %s-%s done — %d features", series_name, mode, len(imp))
    return imp


# ══════════════════════════════════════════════════════════════════════════════
# Prophet — native component decomposition
# ══════════════════════════════════════════════════════════════════════════════

def shap_prophet(
    series_name: str,
    mode: str,
    fitted_model,            # Prophet model object
    regressor_cols: list,
    macro_full: Optional[pd.DataFrame],
    out_dir: str,
    figures_dir: str,
) -> Optional[pd.DataFrame]:
    """
    Prophet component decomposition for 2017.

    Prophet's predict() directly returns columns for each component
    (trend, yearly, each regressor), making it naturally SHAP-compatible.
    """
    try:
        future = fitted_model.make_future_dataframe(periods=12, freq="MS")
        # Add regressor values for 2017
        if regressor_cols and macro_full is not None:
            from src.model_utils import align_macro_for_series
            future_dates = pd.DatetimeIndex(future["ds"])
            aligned = align_macro_for_series(
                macro_full, future_dates, series_name, feature_set="linear"
            )
            for col in regressor_cols:
                if col in aligned.columns:
                    future[col] = aligned[col].values
                else:
                    future[col] = 0.0

        forecast = fitted_model.predict(future)
        rows_2017 = forecast[forecast["ds"].dt.year == 2017].copy()
        rows_2017 = rows_2017.reset_index(drop=True)
    except Exception as e:
        log.warning("[Prophet-SHAP] Prediction failed for %s-%s: %s", series_name, mode, e)
        return None

    # Component columns available in Prophet output
    component_candidates = ["trend", "yearly", "weekly", "monthly"] + list(regressor_cols)
    component_cols = [c for c in component_candidates if c in rows_2017.columns]

    if not component_cols:
        log.warning("[Prophet-SHAP] No component columns found for %s-%s", series_name, mode)
        return None

    # Build attribution rows
    rows = []
    for idx, row in rows_2017.iterrows():
        for comp in component_cols:
            rows.append({
                "series":      series_name,
                "mode":        mode,
                "month":       idx + 1,
                "date":        row["ds"],
                "feature":     comp,
                "attribution": round(float(row[comp]), 6) if not pd.isna(row[comp]) else np.nan,
            })

    if not rows:
        return None

    attr_df = pd.DataFrame(rows)
    out_base = f"{out_dir}/{series_name}_prophet_{mode}"
    _save_csv(attr_df, f"{out_base}_attribution.csv")

    imp = (
        attr_df.groupby("feature")["attribution"]
        .agg(mean_abs_attribution=lambda x: x.abs().mean(),
             mean_attribution="mean")
        .reset_index()
        .sort_values("mean_abs_attribution", ascending=False)
    )
    _save_csv(imp, f"{out_base}_importance.csv")

    _plot_attribution_bar(
        imp,
        title=(f"Prophet — Component Decomposition\n"
               f"{series_name.capitalize()} 2017 (component contribution)"),
        out_path=f"{figures_dir}/{series_name}_prophet_{mode}_attribution.png",
    )
    _plot_attribution_heatmap(
        attr_df, "feature", "attribution", "month",
        title=f"Prophet Component Attribution Heatmap — {series_name.capitalize()} 2017",
        out_path=f"{figures_dir}/{series_name}_prophet_{mode}_heatmap.png",
    )

    log.info("[Prophet-SHAP] %s-%s done — components: %s", series_name, mode, component_cols)
    return imp


# ══════════════════════════════════════════════════════════════════════════════
# LSTM — gradient × input attribution
# ══════════════════════════════════════════════════════════════════════════════

def shap_lstm(
    series_name: str,
    mode: str,
    model_path: str,          # path to saved .pt checkpoint
    target_scaler_path: str,
    macro_scaler_path: Optional[str],
    macro_cols_path: Optional[str],
    full: pd.Series,           # training series up to 2016
    macro_full: Optional[pd.DataFrame],
    out_dir: str,
    figures_dir: str,
    lookback: int = 12,
) -> Optional[pd.DataFrame]:
    """
    Gradient × input attribution for LSTM over the lookback window.

    For each of the 12 2017 forecast steps:
      - Build the input sequence (12 time steps × n_features)
      - Compute ∂output/∂input_t for each (time_step, feature) pair
      - Attribution = gradient × (input − zero_baseline)
    Averaged over time steps gives per-feature importance.
    """
    if not Path(model_path).exists():
        log.warning("[LSTM-SHAP] Model not found at %s — skipping.", model_path)
        return None

    try:
        import torch
        import joblib
        from src.models.lstm_model import (
            _LSTMNet, _fit_transform_target, _transform_target,
            _fit_transform_macro, _transform_macro, _available_macro_cols,
        )
    except ImportError as e:
        log.warning("[LSTM-SHAP] Missing dependency: %s", e)
        return None

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location="cpu")
    n_features = checkpoint["n_features"]
    units_1    = checkpoint.get("units_1", 128)
    units_2    = checkpoint.get("units_2", 64)
    dropout    = checkpoint.get("dropout", 0.2)

    net = _LSTMNet(n_features, lookback, dropout=dropout, units_1=units_1, units_2=units_2)
    net._net.load_state_dict(checkpoint["state_dict"])
    net._net.eval()

    # Load scalers
    target_scaler = joblib.load(target_scaler_path)

    macro_scaler, macro_cols = None, []
    if macro_scaler_path and Path(macro_scaler_path).exists():
        macro_scaler = joblib.load(macro_scaler_path)
    if macro_cols_path and Path(macro_cols_path).exists():
        macro_cols = joblib.load(macro_cols_path)

    # Build seed window (last `lookback` steps of training data)
    full_scaled = _transform_target(full, target_scaler)   # (T, 1)
    if macro_full is not None and not isinstance(macro_full.index, pd.DatetimeIndex):
        macro_full = macro_full.copy()
        macro_full.index = pd.to_datetime(macro_full.index)
    if macro_scaler is not None and macro_cols and macro_full is not None:
        macro_scaled = _transform_macro(macro_full, full.index, macro_scaler, macro_cols)
        seed_data = np.hstack([full_scaled, macro_scaled]).astype(np.float32)
    else:
        seed_data = full_scaled.astype(np.float32)

    if len(seed_data) < lookback:
        log.warning("[LSTM-SHAP] Not enough history for lookback window.")
        return None

    seed_window = seed_data[-lookback:]   # (lookback, n_features)

    # Build feature names for each (time_step, feature) combination
    target_col = f"{series_name}_growth"
    feat_names = [target_col] + list(macro_cols)  # target first, then macro

    rows = []
    current_window = seed_window.copy()

    for step in range(12):
        x_tensor = torch.tensor(current_window[None, :, :],
                                dtype=torch.float32, requires_grad=True)

        # Forward pass
        out = net._net(x_tensor)  # shape (1, 1) via fc2

        # Backward pass: gradient of output w.r.t. input
        out.squeeze().backward()
        grad = x_tensor.grad.detach().numpy()[0]   # (lookback, n_features)

        # Gradient × input attribution (integrated from zero baseline)
        attr = grad * current_window   # element-wise product

        # Aggregate over time steps → per-feature importance
        for f_idx, f_name in enumerate(feat_names):
            if f_idx >= attr.shape[1]:
                continue
            # Mean attribution over lookback steps for this feature
            attr_f = float(np.mean(attr[:, f_idx]))
            rows.append({
                "series":      series_name,
                "mode":        mode,
                "step":        step + 1,
                "month":       step + 1,
                "date":        FORECAST_DATES[step],
                "feature":     f_name,
                "attribution": round(attr_f, 6),
            })

        # Advance window (append zero for next prediction step — simplified)
        next_val = np.zeros((1, n_features), dtype=np.float32)
        current_window = np.vstack([current_window[1:], next_val])

    if not rows:
        return None

    attr_df = pd.DataFrame(rows)
    out_base = f"{out_dir}/{series_name}_lstm_{mode}"
    _save_csv(attr_df, f"{out_base}_attribution.csv")

    imp = (
        attr_df.groupby("feature")["attribution"]
        .agg(mean_abs_attribution=lambda x: x.abs().mean(),
             mean_attribution="mean")
        .reset_index()
        .sort_values("mean_abs_attribution", ascending=False)
    )
    _save_csv(imp, f"{out_base}_importance.csv")

    _plot_attribution_bar(
        imp,
        title=(f"LSTM — Gradient×Input Attribution\n"
               f"{series_name.capitalize()} 2017 (mean over lookback)"),
        out_path=f"{figures_dir}/{series_name}_lstm_{mode}_attribution.png",
    )
    _plot_attribution_heatmap(
        attr_df, "feature", "attribution", "month",
        title=f"LSTM Attribution Heatmap — {series_name.capitalize()} 2017",
        out_path=f"{figures_dir}/{series_name}_lstm_{mode}_heatmap.png",
    )

    log.info("[LSTM-SHAP] %s-%s done — %d features", series_name, mode, len(imp))
    return imp


# ══════════════════════════════════════════════════════════════════════════════
# TFT — attention-weight attribution
# ══════════════════════════════════════════════════════════════════════════════

def shap_tft(
    series_name: str,
    mode: str,
    model_path: str,
    full: pd.Series,
    macro_full: Optional[pd.DataFrame],
    out_dir: str,
    figures_dir: str,
    lookback: int = 12,
) -> Optional[pd.DataFrame]:
    """
    Attention weight attribution for the TFT / Attention-LSTM fallback.

    Extracts multi-head attention weights from the model and uses them as
    a proxy for which time steps most influence each forecast step.
    Also uses gradient-based per-feature sensitivity when possible.
    """
    if not Path(model_path).exists():
        log.warning("[TFT-SHAP] Model not found at %s — skipping.", model_path)
        return None

    try:
        import torch
        import joblib
    except ImportError as e:
        log.warning("[TFT-SHAP] Missing dependency: %s", e)
        return None

    checkpoint  = torch.load(model_path, map_location="cpu")
    n_features  = checkpoint.get("n_features", 1)
    hidden      = checkpoint.get("hidden", 64)
    n_heads     = checkpoint.get("n_heads", 4)
    dropout     = checkpoint.get("dropout", 0.2)
    scaler_mean = checkpoint.get("scaler_mean")
    scaler_std  = checkpoint.get("scaler_std")

    # Rebuild the AttentionLSTM
    try:
        from src.models.tft_model import _AttentionLSTM
        tft = _AttentionLSTM(n_features, hidden=hidden, n_heads=n_heads, dropout=dropout)
        tft.model.load_state_dict(checkpoint["state_dict"])
        tft.model.eval()
        if scaler_mean is not None:
            tft.scaler_mean = np.array(scaler_mean)
            tft.scaler_std  = np.array(scaler_std)
    except Exception as e:
        log.warning("[TFT-SHAP] Model load failed: %s", e)
        return None

    # Build seed data (same normalisation as training)
    from src.model_utils import align_macro_for_series, macro_cols_for_series
    target_arr = full.values.reshape(-1, 1)
    if mode == "macro" and macro_full is not None:
        _cols = macro_cols_for_series(series_name, macro_full)
        macro_arr = align_macro_for_series(macro_full, full.index, series_name).values
        data_arr  = np.hstack([target_arr, macro_arr]).astype(np.float64)
        feat_names = [f"{series_name}_growth"] + list(_cols)
    else:
        data_arr   = target_arr.astype(np.float64)
        feat_names = [f"{series_name}_growth"]

    if tft.scaler_mean is not None and tft.scaler_std is not None:
        data_scaled = (data_arr - tft.scaler_mean) / tft.scaler_std
    else:
        data_scaled = data_arr.copy()

    seed_window = data_scaled[-lookback:].astype(np.float32)  # (lookback, n_features)

    rows = []
    attn_rows = []

    current_window = seed_window.copy()

    for step in range(12):
        x_t = torch.tensor(current_window[None, :, :], dtype=torch.float32,
                            requires_grad=True)

        # Forward pass (capture attention weights via hook)
        attn_weights = []

        def _attn_hook(module, input, output):
            # output[1] is the attention weight tensor: (batch, n_heads, T, T)
            if output[1] is not None:
                attn_weights.append(output[1].detach().numpy())

        hook = tft.model.attn.register_forward_hook(_attn_hook)
        try:
            q025, q50, q975 = tft.model(x_t)
            forecast_val = q50
        except Exception:
            hook.remove()
            continue
        hook.remove()

        # Gradient × input per feature
        forecast_val.sum().backward()
        if x_t.grad is not None:
            grad = x_t.grad.detach().numpy()[0]   # (lookback, n_features)
            attr = grad * current_window
        else:
            attr = np.zeros_like(current_window)

        for f_idx, f_name in enumerate(feat_names):
            if f_idx >= attr.shape[1]:
                continue
            attr_f = float(np.mean(attr[:, f_idx]))
            rows.append({
                "series":      series_name,
                "mode":        mode,
                "month":       step + 1,
                "date":        FORECAST_DATES[step],
                "feature":     f_name,
                "attribution": round(attr_f, 6),
            })

        # Attention: last query-position row → (lookback,)
        # PyTorch MHA returns (batch, T, T) when average_attn_weights=True (default)
        # or (batch, n_heads, T, T) when False — handle both shapes.
        if attn_weights:
            aw = attn_weights[0]   # numpy array from detach
            if aw.ndim == 4:
                # (batch, n_heads, T, T) — average over heads first
                aw_last = aw[0].mean(axis=0)[-1]   # (T,)
            elif aw.ndim == 3:
                # (batch, T, T) — heads already averaged by PyTorch
                aw_last = aw[0][-1]                # (T,)
            else:
                aw_last = np.zeros(lookback)
            aw_last = np.atleast_1d(aw_last)       # guard against scalar edge-cases
            for t_idx in range(len(aw_last)):
                attn_rows.append({
                    "series": series_name,
                    "mode":   mode,
                    "month":  step + 1,
                    "lag":    t_idx - lookback + 1,
                    "attention_weight": round(float(aw_last[t_idx]), 6),
                })

        next_val = np.zeros((1, n_features), dtype=np.float32)
        current_window = np.vstack([current_window[1:], next_val])

    if not rows:
        return None

    attr_df = pd.DataFrame(rows)
    out_base = f"{out_dir}/{series_name}_tft_{mode}"
    _save_csv(attr_df, f"{out_base}_attribution.csv")

    if attn_rows:
        attn_df = pd.DataFrame(attn_rows)
        _save_csv(attn_df, f"{out_base}_attention.csv")
        _plot_attention_heatmap(attn_df, series_name, mode, figures_dir)

    imp = (
        attr_df.groupby("feature")["attribution"]
        .agg(mean_abs_attribution=lambda x: x.abs().mean(),
             mean_attribution="mean")
        .reset_index()
        .sort_values("mean_abs_attribution", ascending=False)
    )
    _save_csv(imp, f"{out_base}_importance.csv")

    _plot_attribution_bar(
        imp,
        title=(f"TFT — Gradient×Input Attribution\n"
               f"{series_name.capitalize()} 2017"),
        out_path=f"{figures_dir}/{series_name}_tft_{mode}_attribution.png",
    )

    log.info("[TFT-SHAP] %s-%s done — %d features", series_name, mode, len(imp))
    return imp


def _plot_attention_heatmap(
    attn_df: pd.DataFrame, series_name: str, mode: str, figures_dir: str
) -> None:
    """Heatmap of mean attention weights: forecast step × lag."""
    try:
        pivot = attn_df.groupby(["month", "lag"])["attention_weight"].mean().unstack("lag")
        fig, ax = plt.subplots(figsize=(13, 5))
        fig.patch.set_facecolor("white")
        im = ax.imshow(pivot.values, cmap="viridis", aspect="auto")
        plt.colorbar(im, ax=ax, label="Attention weight")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(l) for l in pivot.columns], fontsize=8)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([MONTHS_SHORT[m-1] for m in pivot.index], fontsize=8)
        ax.set_xlabel("Lag (months before forecast)")
        ax.set_ylabel("Forecast month (2017)")
        ax.set_title(
            f"TFT Attention Weights — {series_name.capitalize()} {mode.capitalize()} 2017",
            fontsize=11, fontweight="bold",
        )
        plt.tight_layout()
        _save_fig(fig, f"{figures_dir}/{series_name}_tft_{mode}_attention.png")
    except Exception as e:
        log.warning("Attention heatmap failed: %s", e)


# ══════════════════════════════════════════════════════════════════════════════
# Main dispatcher
# ══════════════════════════════════════════════════════════════════════════════

def run_ts_shap(
    best_models: dict,
    results: list,
    core_full: pd.DataFrame,
    macro_full: Optional[pd.DataFrame],
    out_dir: str = "outputs/shap/ts",
    figures_dir: str = "outputs/figures/shap",
    models_dir: str = "outputs/models",
) -> None:
    """
    Run SHAP attribution for each selected best model.

    Parameters
    ----------
    best_models : {(series, mode): model_name}
    results     : list of result dicts from stage2_train_evaluate
    core_full   : full core DataFrame (2005-2017)
    macro_full  : full macro DataFrame
    out_dir     : where to save attribution CSVs
    figures_dir : where to save attribution plots
    models_dir  : where LSTM/TFT models were saved
    """
    _ensure(out_dir)
    _ensure(figures_dir)

    log.info("Running TS-SHAP for %d streams", len(best_models))

    # Index results by (series, model, mode) for fast lookup
    results_index: dict = {}
    for r in (results or []):
        key = (r.get("series"), r.get("model"), r.get("mode"))
        results_index[key] = r

    for (series, mode), model_name in best_models.items():
        log.info("[TS-SHAP] %s | %s | %s", series, mode, model_name)
        result = results_index.get((series, model_name, mode), {})

        # Extract full training series
        col = f"{series}_growth"
        full_series = (
            core_full.set_index("date")[col].loc[:"2016-12-01"].dropna()
            if "date" in core_full.columns
            else core_full[col].loc[:"2016-12-01"].dropna()
        )

        try:
            if model_name == "SARIMA":
                fitted = result.get("fitted_model")
                mcols  = result.get("macro_cols", [])
                if fitted is not None:
                    shap_sarima(series, mode, fitted, macro_full, mcols,
                                out_dir, figures_dir)
                else:
                    log.info("[TS-SHAP] SARIMA model object not in result; "
                             "run with return_model=True to enable SHAP.")

            elif model_name == "Prophet":
                fitted = result.get("fitted_model")
                rcols  = result.get("regressor_cols", [])
                if fitted is not None:
                    shap_prophet(series, mode, fitted, rcols, macro_full,
                                 out_dir, figures_dir)
                else:
                    log.info("[TS-SHAP] Prophet model object not in result.")

            elif model_name == "LSTM":
                mpath  = Path(models_dir) / f"{series}_lstm_{mode}.pt"
                tspath = Path(models_dir) / f"{series}_lstm_{mode}_target_scaler.pkl"
                mspath = Path(models_dir) / f"{series}_lstm_{mode}_macro_scaler.pkl"
                mcpath = Path(models_dir) / f"{series}_lstm_{mode}_macro_cols.pkl"
                shap_lstm(
                    series, mode,
                    str(mpath), str(tspath),
                    str(mspath) if mspath.exists() else None,
                    str(mcpath) if mcpath.exists() else None,
                    full_series, macro_full,
                    out_dir, figures_dir,
                )

            elif model_name == "TFT":
                mpath = Path(models_dir) / f"{series}_tft_{mode}.pt"
                shap_tft(series, mode, str(mpath), full_series, macro_full,
                         out_dir, figures_dir)

            else:
                log.info("[TS-SHAP] Unknown model %s — skipping.", model_name)

        except Exception as e:
            log.warning("[TS-SHAP] %s-%s-%s failed: %s", series, model_name, mode, e,
                        exc_info=True)

    log.info("[TS-SHAP] All streams complete.")
