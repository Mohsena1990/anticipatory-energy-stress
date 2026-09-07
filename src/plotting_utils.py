"""
plotting_utils.py
─────────────────
All figure-generation functions for the Anticipatory Energy Stress pipeline.

Figures produced (target year is auto-detected from the data every run --
see forecast_pipeline._compute_default_window -- not a fixed calendar year)
────────────────
  A. forecast_vs_actual_{series}.png        – historical + core/macro forecasts + actual target-year values
  B. growth_components_bar.png              – GasGrowth / ElecGrowth / CarbonGrowth bars
  C. uncertainty_components_bar.png         – per-series uncertainty + average
  D. step6_pipeline_diagram.png             – flowchart of Step 6 stages
  E. model_ranking_polar_{series}_{mode}.png – stacked polar bar chart of normalised model metrics
  F. prediction_intervals_{target_year}.png – PI width comparison: core vs macro, all series
  G. fes_monthly_{target_year}.png          – FES timeline: core vs macro vs actual (Jan-Dec)
"""

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                    # headless rendering
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as mpe
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
# Single source of truth for model->colour across the whole project (Stage 1
# polar charts, rolling performance figures) -- previously duplicated as
# forecast_pipeline._MODEL_COLORS; that module now imports this instead.
MODEL_COLORS = {
    "SARIMA":  "#3B0F70", "Prophet": "#B63679", "LSTM": "#F1605D", "TFT": "#FCB92C",
}
FIGSIZE_WIDE = (14, 5)
FIGSIZE_BAR  = (10, 6)
DPI          = 150


def _save(fig: plt.Figure, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    # metadata={"CreationDate": None} drops the run timestamp the pdf
    # backend embeds by default, so re-running the pipeline on unchanged
    # data doesn't dirty the pdf twin in git.
    fig.savefig(Path(path).with_suffix(".pdf"), bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)
    log.info(f"Figure saved → {path}")


def _infer_forecast_dates(df: pd.DataFrame, n: int = 12) -> pd.DatetimeIndex:
    """Recover the actual forecast window from a forecast CSV's own 'date'
    column (last n unique sorted dates), instead of assuming a fixed
    calendar year -- the window moves every run as new data lands (see
    forecast_pipeline._compute_default_window)."""
    dates = pd.to_datetime(df["date"]).sort_values().unique()
    return pd.DatetimeIndex(dates[-n:])


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


# ── Shared grouped circular bar chart (house style for every polar plot) ───────

def plot_grouped_circular_bars(
    groups: dict,
    group_colors: dict,
    out_path: str,
    title: str = "",
    item_colors: dict | None = None,
    highlight_groups: set | None = None,
    value_fmt: str = "{:.2f}",
    color_legend: dict | None = None,
) -> None:
    """
    Grouped circular bar chart: each group gets its own coloured arc +
    centred label, and each of its items is a separate (non-stacked) bar
    radiating from a common inner baseline circle -- reproduces the
    reference "A/B/C/D" circular bar plot style used consistently across
    every polar chart in this project (Stage 1 model ranking / rolling
    performance, COR-SEM loadings, COR-CVAE alignment) instead of each one
    inventing its own stacked-wedge scheme.

    Deliberately not a stacked chart: individual bars remove the
    ambiguous-total problem stacking has (a chart's visual "shortest bar"
    no longer needs to coincidentally agree with a separately-computed
    selection criterion). Which group is the real, external "winner" is
    shown structurally via `highlight_groups` (a bolder arc + label), not
    inferred from bar length.

    Parameters
    ----------
    groups           : {group_label: {item_label: value}}, in draw order.
                        All values must be >= 0.
    group_colors     : {group_label: hex colour} for arcs/labels and the
                        default bar colour.
    out_path         : PNG file path (a .pdf twin is also saved, via _save).
    title            : figure title.
    item_colors      : optional {(group_label, item_label): hex colour}
                        overriding a specific bar's colour (e.g. colour by
                        winning model while grouping by series).
    highlight_groups : group labels to draw with a bolder arc/label --
                        the actual externally-determined "selected"/"best"
                        group(s), independent of any bar's height.
    value_fmt        : format string for the small value label at each bar tip.
    color_legend     : optional {label: hex colour} legend box -- needed
                        whenever `item_colors` makes bar colour mean
                        something OTHER than group membership (e.g. bars
                        grouped by series but coloured by winning model).
    """
    highlight_groups = highlight_groups or set()
    item_colors = item_colors or {}

    group_labels = [g for g in groups if groups[g]]
    if not group_labels:
        log.warning("plot_grouped_circular_bars: no data to plot")
        return

    all_values = [v for g in group_labels for v in groups[g].values()]
    max_val = max(all_values) if all_values else 1.0
    if max_val <= 0:
        max_val = 1.0

    n_items_total = sum(len(groups[g]) for g in group_labels)
    n_groups = len(group_labels)

    gap_frac = 0.10  # fraction of the full circle spent on inter-group gaps
    gap_each = (2 * np.pi * gap_frac) / n_groups
    item_angle = (2 * np.pi * (1 - gap_frac)) / n_items_total

    # ── Angular layout ──────────────────────────────────────────────────────
    theta_offset = np.pi / 2   # first item starts at the top
    theta_direction = -1        # proceeds clockwise

    cursor = 0.0
    group_ranges: dict = {}
    item_angle_center: dict = {}
    for g in group_labels:
        items = list(groups[g].items())
        start = cursor
        for i, (item_label, _val) in enumerate(items):
            item_angle_center[(g, item_label)] = cursor + item_angle * (i + 0.5)
        cursor += len(items) * item_angle
        group_ranges[g] = (start, cursor)
        cursor += gap_each

    # ── Radial layout ────────────────────────────────────────────────────────
    # r0 (and everything inside it) is sized generously relative to max_val
    # so that fixed-size group-label text has room to breathe -- text at a
    # SMALL radius subtends a much larger angular span than the same text
    # further out, so a too-small inner circle makes adjacent group labels
    # collide regardless of how narrow their angular slice is.
    r0 = max_val * 0.55          # inner baseline circle bars grow from
    label_r = r0 + max_val * 1.20  # one consistent radius for every item label
    r_max = label_r * 1.08

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_theta_offset(theta_offset)
    ax.set_theta_direction(theta_direction)
    ax.set_ylim(0, r_max)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.spines["polar"].set_visible(False)

    def _visual_angle(theta: float) -> float:
        """Screen-space angle (standard math convention) a data-theta value
        renders at, after the offset/direction transform above -- needed to
        compute label rotation, since ax.text's rotation is in screen
        degrees and matplotlib does not adjust it for polar transforms."""
        return theta_offset + theta_direction * theta

    for g in group_labels:
        is_hl = g in highlight_groups
        gcolor = group_colors.get(g, "#888888")

        # Bars
        for item_label, val in groups[g].items():
            center = item_angle_center[(g, item_label)]
            bar_color = item_colors.get((g, item_label), gcolor)
            ax.bar(
                center, val, width=item_angle * 0.92, bottom=r0,
                color=bar_color, edgecolor="white", linewidth=0.8, zorder=3,
            )

        # Group arc (bold + wider when highlighted)
        start, end = group_ranges[g]
        pad = item_angle * 0.06
        arc_theta = np.linspace(start + pad, end - pad, 60)
        ax.plot(
            arc_theta, np.full_like(arc_theta, r0 * 0.90),
            color="#B8860B" if is_hl else gcolor,
            linewidth=7 if is_hl else 4,
            solid_capstyle="round", zorder=2,
        )

        # Group label, in the empty inner circle but close to its own arc
        # (not dead-centre) -- a small radius makes fixed-size text subtend
        # a wide angle and collide with neighbouring groups' labels.
        mid = (start + end) / 2
        ax.text(
            mid, r0 * 0.62, g, ha="center", va="center",
            fontsize=12 if is_hl else 11,
            fontweight="bold", color="#B8860B" if is_hl else "#333333",
        )

        # Item labels + leader lines for bars that don't reach label_r
        for item_label, val in groups[g].items():
            center = item_angle_center[(g, item_label)]
            bar_tip = r0 + val
            if label_r - bar_tip > max_val * 0.02:
                ax.plot([center, center], [bar_tip, label_r * 0.985],
                        color="#BBBBBB", linewidth=0.7, zorder=1)

            rot_deg = np.degrees(_visual_angle(center)) % 360
            flipped = 90 < rot_deg < 270
            ha = "right" if flipped else "left"
            label_rot = rot_deg + 180 if flipped else rot_deg
            ax.text(
                center, label_r, f"{item_label}",
                rotation=label_rot, rotation_mode="anchor",
                ha=ha, va="center", fontsize=8, color="#333333",
            )

            # Value, on the colourful bar itself (mid-radius of the wedge) --
            # white with a dark stroke so it stays legible against any of the
            # per-model/per-series bar colours.
            ax.text(
                center, r0 + val / 2, value_fmt.format(val),
                rotation=label_rot, rotation_mode="anchor",
                ha="center", va="center", fontsize=7.5, fontweight="bold",
                color="white", zorder=4,
                path_effects=[mpe.withStroke(linewidth=2, foreground="black")],
            )

    if color_legend:
        legend_handles = [
            mpatches.Patch(facecolor=c, edgecolor="white", label=lbl)
            for lbl, c in color_legend.items()
        ]
        ax.legend(
            handles=legend_handles, loc="center left", bbox_to_anchor=(1.08, 0.5),
            fontsize=9, framealpha=0.95, title="Bar colour", title_fontsize=9.5,
        )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=24)
    _save(fig, out_path)


# ── E. Polar model ranking chart ───────────────────────────────────────────────

MODEL_RANKING_METRICS = ["forecast_actual_MAE", "forecast_actual_RMSE", "forecast_actual_SMAPE"]
MODEL_RANKING_FALLBACK_METRICS = ["MAE", "RMSE", "SMAPE", "MASE",
                                  "QuantileLoss", "WinklerScore", "MSIS"]
MODEL_RANKING_LABELS = {
    "forecast_actual_MAE": "MAE", "forecast_actual_RMSE": "RMSE", "forecast_actual_SMAPE": "SMAPE",
    "MAE": "MAE", "RMSE": "RMSE", "SMAPE": "SMAPE", "MASE": "MASE",
    "QuantileLoss": "QuantileLoss", "WinklerScore": "WinklerScore", "MSIS": "MSIS",
}


def plot_model_ranking_polar(
    metrics_df: pd.DataFrame,
    series_name: str,
    mode: str,
    out_path: str,
    selected_model: str | None = None,
) -> None:
    """
    Grouped circular bar chart (house style, see plot_grouped_circular_bars):
    one group per model, one individual (non-stacked) bar per evaluation
    metric. The pipeline-selected model's group gets the bold highlighted
    arc + label -- shown structurally, not inferred from which bar/group
    looks shortest, so there is no possibility of the chart's visual and
    the real selection disagreeing (the failure mode of the earlier
    stacked-bar version: summing normalised metrics into one ambiguous
    total could, and did, order models differently than the pipeline's own
    forecast_actual_MAE-based selection).

    Every call uses the SAME fixed metric list (MODEL_RANKING_METRICS) so
    every model_ranking_polar_*.png in a run is directly comparable --
    falls back to MODEL_RANKING_FALLBACK_METRICS only when forecast-vs-
    actual data genuinely isn't available yet (a target year that hasn't
    been realised).

    Parameters
    ----------
    metrics_df     : ranked metrics table (from model_evaluation)
    series_name    : 'gas', 'electricity', or 'carbon'
    mode           : 'core' or 'macro'
    out_path       : file path for PNG
    selected_model : the model actually chosen by the pipeline for this
                      (series, mode) -- from fes_calculator._find_best_models.
    """
    sub = metrics_df[
        (metrics_df["series_name"] == series_name) &
        (metrics_df["mode"] == mode)
    ].copy()

    if sub.empty:
        log.warning(f"No data for polar chart ({series_name}, {mode})")
        return

    has_forecast_actual = (
        all(c in sub.columns for c in MODEL_RANKING_METRICS)
        and sub["forecast_actual_MAE"].notna().any()
    )
    metrics = MODEL_RANKING_METRICS if has_forecast_actual else MODEL_RANKING_FALLBACK_METRICS
    cols_avail = [c for c in metrics if c in sub.columns]
    if not cols_avail:
        log.warning(f"No ranking metrics available for polar chart ({series_name}, {mode})")
        return

    sub = sub.set_index("model")
    groups = {
        model: {MODEL_RANKING_LABELS.get(c, c): max(float(sub.loc[model, c]), 0.0) for c in cols_avail}
        for model in sub.index
    }
    group_colors = {model: MODEL_COLORS.get(model, "#888888") for model in sub.index}

    basis_label = "forecast-vs-actual accuracy" if has_forecast_actual else "validation-period fit"
    title = (
        f"{series_name.capitalize()} [{mode}] — Model Ranking ({basis_label})\n"
        f"shorter bar = better"
    )
    if selected_model is not None:
        # The pipeline's own selection can be made on a different metric than
        # the one drawn here (e.g. selection_basis='validation' picks by the
        # 2016 walk-forward backtest even when 2017 forecast-vs-actual bars
        # are shown, since the latter is hindsight and shouldn't drive a
        # genuine ex-ante choice -- see fes_calculator._find_best_models /
        # model_evaluation.apply_selection_scores). When that happens the
        # selected wedge is not necessarily the shortest one on THIS chart,
        # which looks like a bug if unlabelled, so say so explicitly instead
        # of leaving a bare "selected: X" next to "shorter bar = better".
        basis = sub["selection_basis"].iloc[0] if "selection_basis" in sub.columns else None
        if has_forecast_actual and basis not in (None, "forecast_actual"):
            title += f"  |  selected: {selected_model} (via {basis} backtest, not shown metrics)"
        else:
            title += f"  |  selected: {selected_model}"

    plot_grouped_circular_bars(
        groups, group_colors, out_path, title=title,
        highlight_groups={selected_model} if selected_model else None,
    )


def plot_all_polar_charts(
    metrics_df: pd.DataFrame,
    figures_dir: str,
    series_names: list = None,
    modes: list = None,
    best: dict | None = None,
) -> None:
    """Generate one polar chart per (series, mode) combination.

    best : {(series, mode): model} from fes_calculator._find_best_models --
           the actual pipeline selection, used to highlight the right wedge.
           Optional so existing callers without it keep working unhighlighted.
    """
    if series_names is None:
        series_names = ["gas", "electricity", "carbon"]
    if modes is None:
        modes = ["core", "macro"]

    for series in series_names:
        for mode in modes:
            out = f"{figures_dir}/model_ranking_polar_{series}_{mode}.png"
            selected = best.get((series, mode)) if best else None
            try:
                plot_model_ranking_polar(metrics_df, series, mode, out, selected_model=selected)
            except Exception as e:
                log.warning(f"Polar chart failed ({series},{mode}): {e}")


# ── F. Prediction Interval comparison figure ───────────────────────────────────
import pandas as pd
import matplotlib.pyplot as plt

def plot_prediction_intervals(
    forecast_dir: str,
    out_path: str,
    series_names: list = None,
) -> None:
    """
    Two-row figure showing the target-year forecast + prediction intervals
    for all series.

    Row 1 — core-only models
    Row 2 — macro-augmented models

    For each series, plots:
      - Point forecast (solid line)
      - 95 % PI shaded band
      - Actual values for the target year (diamonds, if available)

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
        "actual":      "#2C3E50",
    }

    _MODEL_STYLES: dict = {
        "SARIMA":  {"color": "#2980B9", "ls": "-",   "marker": "o",  "lw": 2.0},
        "Prophet": {"color": "#E67E22", "ls": "--",  "marker": "s",  "lw": 2.0},
        "LSTM":    {"color": "#8E44AD", "ls": "-.",  "marker": "^",  "lw": 2.0},
        "TFT":     {"color": "#27AE60", "ls": ":",   "marker": "D",  "lw": 2.0},
    }

    fig, axes = plt.subplots(2, len(series_names),
                             figsize=(6 * len(series_names), 9),
                             sharey=False)
    fig.patch.set_facecolor("white")

    target_year = None
    for col_idx, series in enumerate(series_names):
        path = f"{forecast_dir}/{series}_growth_pct_forecasts_all.csv"
        try:
            df_all = pd.read_csv(path, parse_dates=["date"])
        except FileNotFoundError:
            log.warning(f"Not found: {path}")
            for row_idx in range(2):
                axes[row_idx, col_idx].set_visible(False)
            continue

        forecast_dates = _infer_forecast_dates(df_all)
        if target_year is None:
            target_year = int(forecast_dates[0].year)

        for row_idx, mode in enumerate(["core", "macro"]):
            ax = axes[row_idx, col_idx]
            ax.set_facecolor("white")

            df_mode = df_all[df_all["mode"] == mode]

            # Loop through the pre-defined styles to map data by model string key
            for model, style in _MODEL_STYLES.items():
                df_m = df_mode[df_mode["model"] == model]
                if df_m.empty:
                    continue
                
                df_m = df_m.set_index("date")
                fc = df_m["forecast"].reindex(forecast_dates).values
                lb = df_m["lower_bound"].reindex(forecast_dates).values
                ub = df_m["upper_bound"].reindex(forecast_dates).values

                # Line plots now absorb individual settings from your style dict
                ax.plot(MONTHS[:12], fc, 
                        color=style["color"], 
                        linewidth=style["lw"],
                        linestyle=style["ls"], 
                        marker=style["marker"],
                        markersize=5,
                        alpha=0.9,
                        label=f"{model} forecast")
                
                # Fills map cleanly to your selected configurations
                ax.fill_between(MONTHS[:12], lb, ub,
                                alpha=0.12, color=style["color"])

            # Actual target-year values
            act = _actual_values_by_date(df_mode, forecast_dates)
            if not all(pd.isna(act)):
                ax.plot(MONTHS[:12], act,
                        color=_PALETTE["actual"], marker="o",
                        markersize=5, linestyle="none", zorder=5,
                        label=f"Actual {target_year}")

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
        f"Prediction Intervals — {target_year} Forecast (core vs macro)  |  95 % PI shaded",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    _save(fig, out_path)



# ── H. Static target-year model-comparison figure (6 total: 3 series × 2 modes) ─

_MODEL_STYLES: dict = {
    "SARIMA":  {"color": "#2980B9", "ls": "-",   "marker": "o",  "lw": 2.0},
    "Prophet": {"color": "#E67E22", "ls": "--",  "marker": "s",  "lw": 2.0},
    "LSTM":    {"color": "#8E44AD", "ls": "-.",  "marker": "^",  "lw": 2.0},
    "TFT":     {"color": "#27AE60", "ls": ":",   "marker": "D",  "lw": 2.0},
}
_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def plot_model_comparison(
    forecast_dir: str,
    series_name: str,
    mode: str,
    out_path: str,
    actual_target: Optional["pd.Series"] = None,
) -> None:
    """
    Static target-year monthly figure: all 4 models' point forecasts + 95 % PI,
    plus actual values for one (series, mode) pair. The target year itself
    is inferred from the forecast CSV's own dates, not hardcoded.

    Parameters
    ----------
    forecast_dir  : directory holding *_forecasts_all.csv
    series_name   : 'gas', 'electricity', or 'carbon'
    mode          : 'core' or 'macro'
    out_path      : PNG file path
    actual_target : optional 12-element actual-value Series for the target year
    """
    all_path = f"{forecast_dir}/{series_name}_growth_pct_forecasts_all.csv"
    try:
        df_all = pd.read_csv(all_path, parse_dates=["date"])
    except FileNotFoundError:
        log.warning(f"Not found for target-year comparison: {all_path}")
        return

    df_mode = df_all[df_all["mode"] == mode].copy()
    if df_mode.empty:
        log.warning(f"No data for mode={mode} in {all_path}")
        return

    forecast_dates = _infer_forecast_dates(df_mode)
    target_year = int(forecast_dates[0].year)

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

    # Actual target-year values
    act_vals = None
    if actual_target is not None and actual_target.notna().any():
        act_vals = actual_target.reindex(forecast_dates).values
    elif "actual" in df_mode.columns:
        act_vals = _actual_values_by_date(df_mode, forecast_dates)

    if act_vals is not None and not all(pd.isna(act_vals)):
        ax.plot(
            _MONTHS, act_vals,
            color="#2C3E50", linestyle="none",
            marker="o", markersize=6, markeredgewidth=1.0,
            markeredgecolor="white", zorder=6,
            label=f"Actual {target_year}",
        )

    ax.axhline(0, color="#BDC3C7", linewidth=0.8)
    mode_label = "Core-Only Models" if mode == "core" else "Macro-Augmented Models"
    ax.set_title(
        f"UK {series_name.capitalize()} Growth (%) — {target_year} Monthly Forecast\n"
        f"{mode_label}  |  95 % PI shaded  |  Actual values ●",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.set_xlabel(f"Month ({target_year})", fontsize=11)
    ax.set_ylabel("Growth Rate (%)", fontsize=11)
    ax.legend(fontsize=10, framealpha=0.92, loc="best")
    ax.grid(True, color=PALETTE["grid"], linewidth=0.8)
    ax.tick_params(axis="both", labelsize=10)

    _save(fig, out_path)


def plot_all_model_comparisons(
    forecast_dir: str,
    figures_dir: str,
    series_names: list = None,
    modes: list = None,
) -> None:
    """Generate all 6 static target-year model-comparison figures."""
    if series_names is None:
        series_names = ["gas", "electricity", "carbon"]
    if modes is None:
        modes = ["core", "macro"]

    for series in series_names:
        for mode in modes:
            out = f"{figures_dir}/forecast_comparison_{series}_{mode}.png"
            try:
                plot_model_comparison(forecast_dir, series, mode, out)
            except Exception as e:
                log.warning(f"Target-year comparison figure failed ({series},{mode}): {e}")

# ── I. Interactive Plotly timeline (6 total: 3 series × 2 modes) ──────────────

def plot_interactive_forecast(
    historical: "pd.Series",
    forecast_dir: str,
    series_name: str,
    mode: str,
    out_path: str,
) -> None:
    """
    Interactive Plotly HTML figure: full historical + target-year timeline.
    The historical range and target year are both inferred from the data
    (historical.index and the forecast CSV's own dates), not hardcoded.

    Layout
    ------
    - Solid grey area: historical actual values
    - Vertical dashed line separating history from the forecast
    - One coloured trace per model (point forecast + shaded PI band)
    - Models are toggleable via the legend (click to show/hide)
    - Hover tooltip shows date, model, forecast value, PI bounds

    Parameters
    ----------
    historical   : full historical actual Series (DatetimeIndex, name = col)
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
    if df_mode.empty:
        log.warning(f"No data for mode={mode} in {all_path}")
        return
    forecast_dates = _infer_forecast_dates(df_mode)
    target_year = int(forecast_dates[0].year)
    hist_start_year = int(historical.index.min().year) if historical is not None and not historical.empty else None
    hist_end_year   = int(historical.index.max().year) if historical is not None and not historical.empty else None
    hist_range_label = f"{hist_start_year}–{hist_end_year}" if hist_start_year else "history"

    mode_label = "Core-Only" if mode == "core" else "Macro-Augmented"
    title = (
        f"UK {series_name.capitalize()} Growth (%) — Full Timeline {hist_range_label}–{target_year}  "
        f"[{mode_label} Models]"
    )

    fig = go.Figure()

    # ── Historical ─────────────────────────────────────────────────────────
    if historical is not None and not historical.empty:
        fig.add_trace(go.Scatter(
            x=historical.index,
            y=historical.values,
            name=f"Historical {hist_range_label}",
            mode="lines",
            line=dict(color="#7F8C8D", width=2),
            fill="tozeroy",
            fillcolor="rgba(127,140,141,0.08)",
            hovertemplate="<b>%{x|%b %Y}</b><br>Actual: %{y:.3f}%<extra></extra>",
        ))

    # ── Actual target-year values ─────────────────────────────────────────
    act_col = "actual"
    if act_col in df_mode.columns:
        act_values = _actual_values_by_date(df_mode, forecast_dates)
        act_target = pd.Series(act_values, index=forecast_dates).dropna()
        if not act_target.empty:
            fig.add_trace(go.Scatter(
                x=act_target.index,
                y=act_target.values,
                name=f"Actual {target_year}",
                mode="markers",
                marker=dict(symbol="circle", size=10,
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

    # ── Vertical separator at the start of the forecast window ────────────
    split_date = forecast_dates.min()
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
        text=f"  {target_year} forecast →",
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
    train_start: str = "2005-01-01",
    train_end: str = "2016-12-01",
) -> None:
    """Generate all 6 interactive HTML forecast timelines.

    train_start/train_end default to the original single-year window but
    should be passed explicitly by the caller (src.fes_calculator.compute_fes
    does) so the historical range shown matches whatever window was
    actually used to train the current forecast.
    """
    if series_names is None:
        series_names = ["gas", "electricity", "carbon"]
    if modes is None:
        modes = ["core", "macro"]

    TRAIN_START = train_start
    TRAIN_END   = train_end

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

