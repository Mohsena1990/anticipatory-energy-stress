"""
plotting_utils.py
─────────────────
All figure-generation functions for the Anticipatory Energy Stress pipeline.

Figures produced
────────────────
  A. forecast_vs_actual_{series}.png        – historical + core/macro forecasts + actual 2017
  B. growth_components_bar.png              – GasGrowth / ElecGrowth / CarbonGrowth bars
  C. uncertainty_components_bar.png         – per-series uncertainty + average
  D. step6_pipeline_diagram.png             – flowchart of Step 6 stages
  E. model_ranking_polar_{series}_{mode}.png – polar/radar chart of normalised model metrics
  F. prediction_intervals_2017.png          – PI width comparison: core vs macro, all series
  G. fes_monthly_2017.png                   – FES timeline: core vs macro vs actual (Jan-Dec)
"""

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                    # headless rendering
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from pathlib import Path
from typing import Optional

from src.logging_utils import get_logger

log = get_logger("plotting")

# ── Style constants ────────────────────────────────────────────────────────────
PALETTE = {
    "gas":         "#E67E22",
    "electricity": "#2980B9",
    "carbon":      "#27AE60",
    "forecast":    "#8E44AD",
    "ci":          "#D7BDE2",
    "grid":        "#EAECEE",
}
FIGSIZE_WIDE = (14, 5)
FIGSIZE_BAR  = (10, 6)
DPI          = 150


def _save(fig: plt.Figure, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Figure saved → {path}")


def _actual_values_by_date(df: pd.DataFrame, forecast_dates: pd.DatetimeIndex) -> np.ndarray:
    """Return one actual value per forecast date, even when several models share dates."""
    if "actual" not in df.columns or df.empty:
        return np.full(len(forecast_dates), np.nan)

    actual = (
        df[["date", "actual"]]
        .dropna(subset=["actual"])
        .groupby("date", sort=True)["actual"]
        .first()
    )
    return actual.reindex(forecast_dates).values


# ── A. Forecast vs Actual ──────────────────────────────────────────────────────

def plot_forecast_vs_actual(
    actual: pd.Series,
    forecast_2017: np.ndarray,
    lower_2017: Optional[np.ndarray],
    upper_2017: Optional[np.ndarray],
    series_name: str,
    model_name: str,
    out_path: str,
    units: str = "",
) -> None:
    """
    Plot historical actuals plus 2017 forecast with prediction interval.

    Parameters
    ----------
    actual        : full history (2005-2017)
    forecast_2017 : 12-element array of 2017 point forecasts
    lower/upper   : 95 % PI bounds (may be None)
    series_name   : 'gas', 'electricity', or 'carbon'
    model_name    : name of selected model (for subtitle)
    out_path      : file path to save
    units         : y-axis label units
    """
    colour  = PALETTE.get(series_name, "#555555")
    fct_col = PALETTE["forecast"]
    dates_fc = pd.date_range("2017-01-01", periods=12, freq="MS")

    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    # Historical
    ax.plot(actual.index, actual.values, color=colour, linewidth=2.0,
            label=f"Actual ({actual.index.min().year}–{actual.index.max().year})", zorder=3)

    # Forecast
    ax.plot(dates_fc, forecast_2017, color=fct_col, linewidth=2.0,
            linestyle="--", marker="o", markersize=4, label=f"Forecast 2017 ({model_name})", zorder=4)

    # Prediction interval
    if lower_2017 is not None and upper_2017 is not None:
        ax.fill_between(
            dates_fc, lower_2017, upper_2017,
            alpha=0.25, color=fct_col, label="95 % Prediction Interval", zorder=2
        )

    # Vertical separator
    ax.axvline(pd.Timestamp("2017-01-01"), color="#BDC3C7", linewidth=1.2, linestyle=":", zorder=1)

    ax.set_title(
        f"{series_name.capitalize()} Price — Forecast vs Actual",
        fontsize=14, fontweight="bold", pad=12
    )
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel(units or series_name.capitalize(), fontsize=11)
    ax.legend(framealpha=0.9, fontsize=9)
    ax.grid(True, color=PALETTE["grid"], linewidth=0.8, zorder=0)
    ax.tick_params(axis="both", labelsize=9)

    _save(fig, out_path)


# ── B. Growth components bar chart ────────────────────────────────────────────

def plot_growth_components(
    gas_growth: float,
    elec_growth: float,
    carbon_growth: float,
    out_path: str,
) -> None:
    """Bar chart of annualised 2017 growth rates for all three core series."""
    labels  = ["Gas", "Electricity", "Carbon"]
    values  = [gas_growth * 100, elec_growth * 100, carbon_growth * 100]
    colours = [PALETTE["gas"], PALETTE["electricity"], PALETTE["carbon"]]
    edge    = ["#C0392B" if v < 0 else "#1A5276" for v in values]

    fig, ax = plt.subplots(figsize=FIGSIZE_BAR)
    bars = ax.bar(labels, values, color=colours, edgecolor=edge, linewidth=1.2, width=0.5)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (0.3 if val >= 0 else -0.8),
            f"{val:+.2f} %",
            ha="center", va="bottom", fontsize=11, fontweight="bold"
        )

    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_title("FES Growth Components — 2017 Annual Forecast vs 2017 Actual",
                 fontsize=13, fontweight="bold", pad=10)
    ax.set_ylabel("Growth Rate (%)", fontsize=11)
    ax.set_ylim(min(values) - 5, max(values) + 8)
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.8)
    ax.set_facecolor("white")

    _save(fig, out_path)


# ── C. Uncertainty components bar chart ───────────────────────────────────────

def plot_uncertainty_components(
    gas_unc: float,
    elec_unc: float,
    carbon_unc: float,
    out_path: str,
) -> None:
    """Bar chart of ForecastUncertainty for each series + average."""
    avg     = np.mean([gas_unc, elec_unc, carbon_unc])
    labels  = ["Gas", "Electricity", "Carbon", "Average"]
    values  = [gas_unc, elec_unc, carbon_unc, avg]
    colours = [PALETTE["gas"], PALETTE["electricity"], PALETTE["carbon"], "#7F8C8D"]

    fig, ax = plt.subplots(figsize=FIGSIZE_BAR)
    bars = ax.bar(labels, values, color=colours, edgecolor="#2C3E50", linewidth=0.8, width=0.5)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{val:.4f}" if val < 1 else f"{val:.3f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold"
        )

    ax.set_title("Forecast Uncertainty — (Upper − Lower) / Forecast  [2017 Annual]",
                 fontsize=12, fontweight="bold", pad=10)
    ax.set_ylabel("Uncertainty Ratio", fontsize=11)
    ax.set_ylim(0, max(values) * 1.25)
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.8)
    ax.set_facecolor("white")

    _save(fig, out_path)


# ── D. Pipeline flowchart ──────────────────────────────────────────────────────

def plot_pipeline_diagram(out_path: str) -> None:
    """
    Simple flowchart:
      Selected best models → Forecast 2017 → Growth calculation
      → Uncertainty calculation → Final FES components
    """
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis("off")
    fig.patch.set_facecolor("#FDFEFE")

    stages = [
        ("Step 6.1\nLoad & Validate\nData", 1.0,   "#2E86C1"),
        ("Step 6.2\nSelect Best\nModel",    3.0,   "#1A5276"),
        ("Step 6.3\nMonthly\nGrowth & Unc", 5.5,  "#117A65"),
        ("Step 6.4\nAnnual\nAggregation",   8.0,   "#76448A"),
        ("Step 6.5-6\nSave Tables\n& Figures", 10.5, "#B7950B"),
        ("Step 6.7\nFES Components\nJSON",  13.0,  "#943126"),
    ]

    BOX_W, BOX_H = 1.8, 1.0
    Y_CENTER = 2.0

    for label, x, colour in stages:
        rect = mpatches.FancyBboxPatch(
            (x - BOX_W / 2, Y_CENTER - BOX_H / 2),
            BOX_W, BOX_H,
            boxstyle="round,pad=0.1",
            facecolor=colour, edgecolor="white", linewidth=1.5,
        )
        ax.add_patch(rect)
        ax.text(x, Y_CENTER, label, ha="center", va="center",
                fontsize=7.5, color="white", fontweight="bold", wrap=True)

    # Arrows
    arrow_props = dict(arrowstyle="-|>", color="#555555", lw=1.5)
    xs = [s[1] for s in stages]
    for i in range(len(xs) - 1):
        ax.annotate(
            "", xy=(xs[i+1] - BOX_W / 2, Y_CENTER),
            xytext=(xs[i] + BOX_W / 2, Y_CENTER),
            arrowprops=arrow_props,
        )

    # Labels underneath
    bottom_labels = [
        (xs[0], "gas · electricity · carbon\n2005–2016 actual + forecasts"),
        (xs[1], "Rank-aggregation\nScore_m = Σ Rank(Metric_k)"),
        (xs[2], "GrowthRate = (FC₂₀₁₈ − Act₂₀₁₇) / Act₂₀₁₇\nUncertainty = (UB − LB) / FC"),
        (xs[3], "Annual avg\nforecast & bounds"),
        (xs[4], "CSV / XLSX\nPNG figures"),
        (xs[5], "GasGrowth · ElecGrowth\nCarbonGrowth · ForecastUnc"),
    ]
    for x, txt in bottom_labels:
        ax.text(x, Y_CENTER - 0.9, txt, ha="center", va="top",
                fontsize=6.5, color="#2C3E50", style="italic")

    ax.set_title("Step 6 Pipeline — Producing FES Components from Best Forecast Model",
                 fontsize=12, fontweight="bold", pad=6, color="#1C2833")

    _save(fig, out_path)


# ── Model comparison heatmap ───────────────────────────────────────────────────

def plot_metrics_heatmap(
    metrics_df: pd.DataFrame,
    out_path: str,
) -> None:
    """Heatmap of normalised metric values (lower = better, green)."""
    try:
        import seaborn as sns
    except ImportError:
        log.warning("seaborn not installed; skipping heatmap")
        return

    METRIC_COLS = [c for c in metrics_df.columns
                   if c not in ("series_name", "model", "rank_score")]

    n_series = metrics_df["series_name"].nunique()
    fig, axes = plt.subplots(1, n_series, figsize=(6 * n_series, 5))
    if n_series == 1:
        axes = [axes]

    for ax, (series, grp) in zip(axes, metrics_df.groupby("series_name")):
        pivot = grp.set_index("model")[METRIC_COLS].astype(float)
        # Normalise each column 0-1 (lower = greener)
        normed = (pivot - pivot.min()) / (pivot.max() - pivot.min() + 1e-10)
        sns.heatmap(
            normed, ax=ax, cmap="RdYlGn_r", annot=pivot.round(3),
            fmt=".3f", linewidths=0.5, vmin=0, vmax=1,
            cbar_kws={"label": "Normalised (lower=better)"},
        )
        ax.set_title(f"{series.capitalize()} — Model Metrics", fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=40, labelsize=8)

    fig.suptitle("Model Comparison Heatmap", fontsize=13, fontweight="bold", y=1.02)
    _save(fig, out_path)


# ── E. Polar model ranking chart ───────────────────────────────────────────────

def plot_model_ranking_polar(
    metrics_df: pd.DataFrame,
    series_name: str,
    mode: str,
    out_path: str,
) -> None:
    """
    Polar / radar chart showing normalised metric values for each model.

    Each spoke = one ranking metric.
    Each coloured line = one model.
    Closer to centre = better performance.

    Parameters
    ----------
    metrics_df  : ranked metrics table (from model_evaluation)
    series_name : 'gas', 'electricity', or 'carbon'
    mode        : 'core' or 'macro'
    out_path    : file path for PNG
    """
    RANK_METRICS = ["MAE", "RMSE", "SMAPE", "MASE",
                    "QuantileLoss", "WinklerScore", "MSIS",
                    "PredictionIntervalCoverage"]

    sub = metrics_df[
        (metrics_df["series_name"] == series_name) &
        (metrics_df["mode"] == mode)
    ].copy()

    if sub.empty:
        log.warning(f"No data for polar chart ({series_name}, {mode})")
        return

    cols_avail = [c for c in RANK_METRICS if c in sub.columns]
    if not cols_avail:
        return

    sub = sub.set_index("model")[cols_avail].astype(float)

    # Normalise: for each metric, 0 = best model, 1 = worst.
    # PredictionIntervalCoverage is higher-better → invert.
    normed = sub.copy()
    for col in cols_avail:
        col_min = sub[col].min()
        col_max = sub[col].max()
        rng = col_max - col_min + 1e-10
        if col == "PredictionIntervalCoverage":
            normed[col] = (col_max - sub[col]) / rng   # invert
        else:
            normed[col] = (sub[col] - col_min) / rng

    N = len(cols_avail)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]                                  # close the loop

    MODEL_COLOURS = [
        "#8E44AD", "#E74C3C", "#2980B9", "#27AE60",
        "#F39C12", "#16A085", "#C0392B",
    ]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor("white")

    short_labels = {
        "MAE": "MAE", "RMSE": "RMSE", "SMAPE": "SMAPE", "MASE": "MASE",
        "QuantileLoss": "QL", "WinklerScore": "Winkler", "MSIS": "MSIS",
        "PredictionIntervalCoverage": "PIC",
    }
    spoke_labels = [short_labels.get(c, c) for c in cols_avail]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(spoke_labels, fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "worst"], fontsize=8, color="#7F8C8D")
    ax.grid(color="#EAECEE", linewidth=0.8)
    ax.spines["polar"].set_color("#BDC3C7")

    for i, (model_name, row) in enumerate(normed.iterrows()):
        values = row.tolist() + row.tolist()[:1]
        colour = MODEL_COLOURS[i % len(MODEL_COLOURS)]
        ax.plot(angles, values, "o-", linewidth=2.0, color=colour,
                markersize=5, label=model_name)
        ax.fill(angles, values, alpha=0.06, color=colour)

    ax.legend(
        loc="upper right", bbox_to_anchor=(1.35, 1.15),
        fontsize=9, framealpha=0.9,
    )
    ax.set_title(
        f"{series_name.capitalize()} [{mode}]\n"
        f"Model Ranking Radar  (centre = best)",
        fontsize=12, fontweight="bold", pad=18,
    )

    _save(fig, out_path)


def plot_all_polar_charts(
    metrics_df: pd.DataFrame,
    figures_dir: str,
    series_names: list = None,
    modes: list = None,
) -> None:
    """Generate one polar chart per (series, mode) combination."""
    if series_names is None:
        series_names = ["gas", "electricity", "carbon"]
    if modes is None:
        modes = ["core", "macro"]

    for series in series_names:
        for mode in modes:
            out = f"{figures_dir}/model_ranking_polar_{series}_{mode}.png"
            try:
                plot_model_ranking_polar(metrics_df, series, mode, out)
            except Exception as e:
                log.warning(f"Polar chart failed ({series},{mode}): {e}")


# ── F. Prediction Interval comparison figure ───────────────────────────────────

def plot_prediction_intervals(
    forecast_dir: str,
    out_path: str,
    series_names: list = None,
) -> None:
    """
    Two-row figure showing 2017 forecast + prediction intervals for all series.

    Row 1 — core-only models
    Row 2 — macro-augmented models

    For each series, plots:
      - Point forecast (solid line)
      - 95 % PI shaded band
      - Actual 2017 values (diamonds, if available)

    Parameters
    ----------
    forecast_dir : directory containing *_forecasts_all.csv files
    out_path     : PNG output path
    series_names : list of series to plot (default: gas, electricity, carbon)
    """
    if series_names is None:
        series_names = ["gas", "electricity", "carbon"]

    MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    _PALETTE = {
        "gas":         "#E67E22",
        "electricity": "#2980B9",
        "carbon":      "#27AE60",
        "core":        "#8E44AD",
        "macro":       "#E74C3C",
        "actual":      "#2C3E50",
    }

    import pandas as pd

    fig, axes = plt.subplots(2, len(series_names),
                             figsize=(6 * len(series_names), 9),
                             sharey=False)
    fig.patch.set_facecolor("white")

    for col_idx, series in enumerate(series_names):
        path = f"{forecast_dir}/{series}_growth_pct_forecasts_all.csv"
        try:
            df_all = pd.read_csv(path, parse_dates=["date"])
        except FileNotFoundError:
            log.warning(f"Not found: {path}")
            for row_idx in range(2):
                axes[row_idx, col_idx].set_visible(False)
            continue

        forecast_dates = pd.date_range("2017-01-01", periods=12, freq="MS")

        for row_idx, mode in enumerate(["core", "macro"]):
            ax = axes[row_idx, col_idx]
            ax.set_facecolor("white")

            df_mode = df_all[df_all["mode"] == mode]
            colour  = _PALETTE[mode]

            # Plot each model's PI and forecast
            models_in_mode = df_mode["model"].unique()
            for midx, model in enumerate(models_in_mode):
                df_m = df_mode[df_mode["model"] == model].set_index("date")
                fc = df_m["forecast"].reindex(forecast_dates).values
                lb = df_m["lower_bound"].reindex(forecast_dates).values
                ub = df_m["upper_bound"].reindex(forecast_dates).values

                alpha_line = 0.9 if midx == 0 else 0.45
                alpha_fill = 0.15 if midx == 0 else 0.07
                lw = 2.0 if midx == 0 else 1.0

                ax.plot(MONTHS[:12], fc, color=colour, linewidth=lw,
                        linestyle="-", alpha=alpha_line,
                        label=f"{model} forecast")
                ax.fill_between(MONTHS[:12], lb, ub,
                                alpha=alpha_fill, color=colour)

            # Actual 2017
            act = _actual_values_by_date(df_mode, forecast_dates)
            if not all(pd.isna(act)):
                ax.plot(MONTHS[:12], act,
                        color=_PALETTE["actual"], marker="D",
                        markersize=6, linestyle="none", zorder=5,
                        label="Actual 2017")

            ax.axhline(0, color="#BDC3C7", linewidth=0.8)
            ax.set_title(
                f"{series.capitalize()} [{mode}]",
                fontsize=11, fontweight="bold",
            )
            ax.tick_params(axis="x", rotation=45, labelsize=8)
            ax.set_ylabel("YoY Growth (%)" if col_idx == 0 else "")
            ax.legend(fontsize=7.5, loc="best", framealpha=0.85, ncol=2)
            ax.grid(True, color="#EAECEE", linewidth=0.7)

    # Row labels
    for row_idx, label in enumerate(["CORE — target series only",
                                     "MACRO — core + exogenous"]):
        axes[row_idx, 0].set_ylabel(f"{label}\n\nYoY Growth (%)",
                                    fontsize=10, fontweight="bold")

    fig.suptitle(
        "Prediction Intervals — 2017 Forecast (core vs macro)  |  95 % PI shaded",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    _save(fig, out_path)


# ── H. Static 2017 model-comparison figure (6 total: 3 series × 2 modes) ─────

_MODEL_STYLES: dict = {
    "SARIMA":  {"color": "#2980B9", "ls": "-",   "marker": "o",  "lw": 2.0},
    "Prophet": {"color": "#E67E22", "ls": "--",  "marker": "s",  "lw": 2.0},
    "LSTM":    {"color": "#8E44AD", "ls": "-.",  "marker": "^",  "lw": 2.0},
    "TFT":     {"color": "#27AE60", "ls": ":",   "marker": "D",  "lw": 2.0},
}
_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def plot_2017_model_comparison(
    forecast_dir: str,
    series_name: str,
    mode: str,
    out_path: str,
    actual_2017: Optional["pd.Series"] = None,
) -> None:
    """
    Static 2017 monthly figure: all 4 models' point forecasts + 95 % PI,
    plus actual 2017 values for one (series, mode) pair.

    Parameters
    ----------
    forecast_dir : directory holding *_forecasts_all.csv
    series_name  : 'gas', 'electricity', or 'carbon'
    mode         : 'core' or 'macro'
    out_path     : PNG file path
    actual_2017  : optional 12-element actual-value Series for 2017
    """
    all_path = f"{forecast_dir}/{series_name}_growth_pct_forecasts_all.csv"
    try:
        df_all = pd.read_csv(all_path, parse_dates=["date"])
    except FileNotFoundError:
        log.warning(f"Not found for 2017 comparison: {all_path}")
        return

    df_mode = df_all[df_all["mode"] == mode].copy()
    if df_mode.empty:
        log.warning(f"No data for mode={mode} in {all_path}")
        return

    forecast_dates = pd.date_range("2017-01-01", periods=12, freq="MS")

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    plotted_any = False
    for model_name, style in _MODEL_STYLES.items():
        df_m = df_mode[df_mode["model"] == model_name].set_index("date")
        if df_m.empty:
            continue

        fc = df_m["forecast"].reindex(forecast_dates).values
        lb = df_m["lower_bound"].reindex(forecast_dates).values
        ub = df_m["upper_bound"].reindex(forecast_dates).values

        if all(pd.isna(fc)):
            continue

        ax.plot(
            _MONTHS, fc,
            color=style["color"], linestyle=style["ls"],
            linewidth=style["lw"], marker=style["marker"], markersize=5,
            label=f"{model_name}",
        )
        ax.fill_between(
            _MONTHS, lb, ub,
            alpha=0.10, color=style["color"],
        )
        plotted_any = True

    if not plotted_any:
        log.warning(f"No models plotted for ({series_name}, {mode})")
        plt.close(fig)
        return

    # Actual 2017 values
    act_vals = None
    if actual_2017 is not None and actual_2017.notna().any():
        act_vals = actual_2017.reindex(forecast_dates).values
    elif "actual" in df_mode.columns:
        act_vals = _actual_values_by_date(df_mode, forecast_dates)

    if act_vals is not None and not all(pd.isna(act_vals)):
        ax.plot(
            _MONTHS, act_vals,
            color="#2C3E50", linestyle="none",
            marker="D", markersize=8, markeredgewidth=1.5,
            markeredgecolor="white", zorder=6,
            label="Actual 2017",
        )

    ax.axhline(0, color="#BDC3C7", linewidth=0.8)
    mode_label = "Core-Only Models" if mode == "core" else "Macro-Augmented Models"
    ax.set_title(
        f"UK {series_name.capitalize()} Growth (%) — 2017 Monthly Forecast\n"
        f"{mode_label}  |  95 % PI shaded  |  Actual values ◆",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.set_xlabel("Month (2017)", fontsize=11)
    ax.set_ylabel("Growth Rate (%)", fontsize=11)
    ax.legend(fontsize=10, framealpha=0.92, loc="best")
    ax.grid(True, color=PALETTE["grid"], linewidth=0.8)
    ax.tick_params(axis="both", labelsize=10)

    _save(fig, out_path)


def plot_all_2017_comparisons(
    forecast_dir: str,
    figures_dir: str,
    series_names: list = None,
    modes: list = None,
) -> None:
    """Generate all 6 static 2017 model-comparison figures."""
    if series_names is None:
        series_names = ["gas", "electricity", "carbon"]
    if modes is None:
        modes = ["core", "macro"]

    for series in series_names:
        for mode in modes:
            out = f"{figures_dir}/forecast_2017_{series}_{mode}.png"
            try:
                plot_2017_model_comparison(forecast_dir, series, mode, out)
            except Exception as e:
                log.warning(f"2017 comparison figure failed ({series},{mode}): {e}")

# ── I. Interactive Plotly timeline (6 total: 3 series × 2 modes) ──────────────

def plot_interactive_forecast(
    historical: "pd.Series",
    forecast_dir: str,
    series_name: str,
    mode: str,
    out_path: str,
) -> None:
    """
    Interactive Plotly HTML figure: full 2005–2017 timeline.

    Layout
    ------
    - Solid grey area: historical 2005-2017 actual values
    - Vertical dashed line at 2017-01-01 separating history from forecast
    - One coloured trace per model (point forecast + shaded PI band)
    - Models are toggleable via the legend (click to show/hide)
    - Hover tooltip shows date, model, forecast value, PI bounds

    Parameters
    ----------
    historical   : full 2005-2017 actual Series (DatetimeIndex, name = col)
    forecast_dir : directory holding forecast CSVs
    series_name  : 'gas', 'electricity', or 'carbon'
    mode         : 'core' or 'macro'
    out_path     : .html output path
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        log.warning("plotly not installed; skipping interactive figure. "
                    "Run: pip install plotly")
        return

    all_path = f"{forecast_dir}/{series_name}_growth_pct_forecasts_all.csv"
    try:
        df_all = pd.read_csv(all_path, parse_dates=["date"])
    except FileNotFoundError:
        log.warning(f"Not found for interactive: {all_path}")
        return

    df_mode = df_all[df_all["mode"] == mode].copy()
    forecast_dates = pd.date_range("2017-01-01", periods=12, freq="MS")

    mode_label = "Core-Only" if mode == "core" else "Macro-Augmented"
    title = (
        f"UK {series_name.capitalize()} Growth (%) — Full Timeline 2005–2017  "
        f"[{mode_label} Models]"
    )

    fig = go.Figure()

    # ── Historical 2005-2017 ──────────────────────────────────────────────────
    if historical is not None and not historical.empty:
        fig.add_trace(go.Scatter(
            x=historical.index,
            y=historical.values,
            name="Historical 2005–2016",
            mode="lines",
            line=dict(color="#7F8C8D", width=2),
            fill="tozeroy",
            fillcolor="rgba(127,140,141,0.08)",
            hovertemplate="<b>%{x|%b %Y}</b><br>Actual: %{y:.3f}%<extra></extra>",
        ))

    # ── Actual 2017 ───────────────────────────────────────────────────────────
    act_col = "actual"
    if act_col in df_mode.columns:
        act_values = _actual_values_by_date(df_mode, forecast_dates)
        act_2017 = pd.Series(act_values, index=forecast_dates).dropna()
        if not act_2017.empty:
            fig.add_trace(go.Scatter(
                x=act_2017.index,
                y=act_2017.values,
                name="Actual 2017",
                mode="markers",
                marker=dict(symbol="diamond", size=10,
                            color="#2C3E50", line=dict(color="white", width=1.5)),
                hovertemplate="<b>%{x|%b %Y}</b><br>Actual: %{y:.3f}%<extra></extra>",
            ))

    # ── Per-model forecasts ───────────────────────────────────────────────────
    _PLOTLY_COLOURS = {
        "SARIMA":  "#2980B9",
        "Prophet": "#E67E22",
        "LSTM":    "#8E44AD",
        "TFT":     "#27AE60",
    }
    _DASH = {"SARIMA": "solid", "Prophet": "dash",
             "LSTM": "dashdot", "TFT": "dot"}

    for model_name, colour in _PLOTLY_COLOURS.items():
        df_m = df_mode[df_mode["model"] == model_name].set_index("date")
        if df_m.empty:
            continue

        fc = df_m["forecast"].reindex(forecast_dates)
        lb = df_m["lower_bound"].reindex(forecast_dates)
        ub = df_m["upper_bound"].reindex(forecast_dates)

        if fc.dropna().empty:
            continue

        rgba_fill = _hex_to_rgba(colour, 0.15)

        # Upper PI bound (invisible line for fill reference)
        fig.add_trace(go.Scatter(
            x=forecast_dates, y=ub.values,
            name=f"{model_name} PI upper",
            mode="lines",
            line=dict(width=0, color=colour),
            showlegend=False,
            hoverinfo="skip",
        ))
        # Lower PI bound with fill to upper
        fig.add_trace(go.Scatter(
            x=forecast_dates, y=lb.values,
            name=f"{model_name} 95% PI",
            mode="lines",
            line=dict(width=0, color=colour),
            fill="tonexty",
            fillcolor=rgba_fill,
            showlegend=True,
            legendgroup=model_name,
            hoverinfo="skip",
        ))
        # Point forecast
        fig.add_trace(go.Scatter(
            x=forecast_dates, y=fc.values,
            name=model_name,
            mode="lines+markers",
            line=dict(color=colour, width=2.5, dash=_DASH[model_name]),
            marker=dict(size=7, color=colour,
                        line=dict(color="white", width=1)),
            legendgroup=model_name,
            hovertemplate=(
                f"<b>{model_name}</b><br>"
                "<b>%{x|%b %Y}</b><br>"
                "Forecast: %{y:.3f}%<br>"
                "<extra></extra>"
            ),
        ))

    # ── Vertical separator at 2017-01-01 ─────────────────────────────────────
    split_date = pd.Timestamp("2017-01-01")
    fig.add_shape(
        type="line",
        x0=split_date,
        x1=split_date,
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line=dict(width=1.5, dash="dash", color="#BDC3C7"),
    )
    fig.add_annotation(
        x=split_date,
        y=1,
        xref="x",
        yref="paper",
        text="  2017 forecast →",
        showarrow=False,
        xanchor="left",
        yanchor="bottom",
        font=dict(size=11, color="#7F8C8D"),
    )

    fig.add_hline(y=0, line_width=1, line_color="#BDC3C7", line_dash="dot")

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, family="Arial Black"), x=0.5),
        xaxis=dict(
            title="Date",
            showgrid=True, gridcolor="#EAECEE",
            rangeslider=dict(visible=True),
            rangeselector=dict(
                buttons=[
                    dict(count=1,  label="1y",  step="year",  stepmode="backward"),
                    dict(count=5,  label="5y",  step="year",  stepmode="backward"),
                    dict(count=10, label="10y", step="year",  stepmode="backward"),
                    dict(step="all", label="All"),
                ]
            ),
        ),
        yaxis=dict(
            title="Growth Rate (%)",
            showgrid=True, gridcolor="#EAECEE", zeroline=False,
        ),
        legend=dict(
            orientation="v", x=1.02, y=1,
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#BDC3C7", borderwidth=1,
            font=dict(size=11),
        ),
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=60, r=200, t=80, b=60),
        height=550,
    )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn")
    log.info(f"Interactive figure saved → {out_path}")


def _hex_to_rgba(hex_colour: str, alpha: float) -> str:
    """Convert '#RRGGBB' to 'rgba(R,G,B,alpha)' string."""
    h = hex_colour.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def plot_all_interactive_forecasts(
    core_df: "pd.DataFrame",
    forecast_dir: str,
    figures_dir: str,
    series_names: list = None,
    modes: list = None,
) -> None:
    """Generate all 6 interactive HTML forecast timelines."""
    if series_names is None:
        series_names = ["gas", "electricity", "carbon"]
    if modes is None:
        modes = ["core", "macro"]

    TRAIN_START = "2005-01-01"
    TRAIN_END   = "2016-12-01"

    for series in series_names:
        col = f"{series}_growth"
        hist = (
            core_df[
                (core_df["date"] >= TRAIN_START) & (core_df["date"] <= TRAIN_END)
            ]
            .set_index("date")[col]
            if col in core_df.columns else pd.Series(dtype=float)
        )
        for mode in modes:
            out = f"{figures_dir}/interactive_{series}_{mode}.html"
            try:
                plot_interactive_forecast(hist, forecast_dir, series, mode, out)
            except Exception as e:
                log.warning(f"Interactive figure failed ({series},{mode}): {e}")


# ── G. FES timeline  ──────────────────────────────────────────────────────────
# (Implemented in fes_calculator.py — _plot_fes_comparison)
# Exposed here as a standalone wrapper for external use.

def plot_fes_timeline(
    fes_monthly: "pd.DataFrame",
    out_path: str,
) -> None:
    """
    Line chart: FES_core vs FES_macro vs FES_actual over Jan–Dec 2017.

    Parameters
    ----------
    fes_monthly : DataFrame with columns fes_core, fes_macro, fes_actual
    out_path    : PNG output path
    """
    import pandas as pd

    MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    _P = {"core": "#8E44AD", "macro": "#E74C3C", "actual": "#2C3E50"}

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    for col, label, colour, ls in [
        ("fes_core",   "FES Core (main — core-only models)",         _P["core"],   "-"),
        ("fes_macro",  "FES Macro (Robustness 1 — macro-augmented)", _P["macro"],  "--"),
        ("fes_actual", "FES Actual (Robustness 2 — realised prices)",_P["actual"], ":"),
    ]:
        if col in fes_monthly.columns:
            months_n = len(fes_monthly)
            ax.plot(MONTHS[:months_n], fes_monthly[col].values,
                    color=colour, linestyle=ls, linewidth=2.2,
                    marker="o", markersize=5, label=label)

    ax.axhline(0, color="#95A5A6", linewidth=0.9, linestyle="-")
    ax.set_title(
        "UK Anticipatory Energy–Carbon Stress Index — Monthly 2017\n"
        "FES_core (main)  |  FES_macro (Robustness 1)  |  FES_actual (Robustness 2)",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.set_xlabel("Month (2017)", fontsize=11)
    ax.set_ylabel("FES  (sum of training-period z-scores)", fontsize=11)
    ax.legend(framealpha=0.92, fontsize=10, loc="upper left")
    ax.grid(True, color=PALETTE["grid"], linewidth=0.8)

    _save(fig, out_path)
