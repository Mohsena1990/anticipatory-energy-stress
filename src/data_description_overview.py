"""
data_description_overview.py
─────────────────────────────
Descriptive distribution figures + summary-statistics tables for the two
input streams that feed Stage 1 forecasting (`reports/06_methodology.md`
Section 2.1) in its two information modes: **core** (`data/processed/
core_energy_carbon.csv` -- gas/electricity/carbon growth, built from the raw
price series in `data/raw/`) and **macro** (`data/processed/macro_controls.csv`
-- exogenous regressors, built from the raw ONS/National Grid ESO series in
`data/raw/`). No modeling -- purely "what does the data feeding the models
look like", feeding `reports/05_data_description.md`.

Usage
─────
  python -m src.data_description_overview
"""

from __future__ import annotations

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")

from src import paths
from src.logging_utils import get_logger

log = get_logger("data_description_overview")

_PALETTE = {"core": "#2980B9", "macro": "#E67E22", "grid": "#EAECEE", "median": "#27AE60"}

# Columns to exclude from the macro summary/figures: lagged duplicates
# (`_lag1`/`_lag12`) of a base column already covered, and the two
# regime/seasonal dummy variables (binary, not a continuous distribution --
# their "distribution" is just a share, already reported as text in
# reports/05_data_description.md Section 3.2).
_MACRO_EXCLUDE_SUFFIXES = ("_lag1", "_lag12")
_MACRO_EXCLUDE_COLS = {"date", "post_2016_electricity_regime", "winter_dummy"}

# A representative subset of macro columns for the panel figure -- the full
# ~26-column base set is summarised in the table, but a histogram grid of
# that many panels is unreadable, so the figure focuses on the variables
# most relevant to the FES macro regressors actually used by the models
# (src/fes_calculator.py, reports/06_methodology.md Section 2.1).
_MACRO_FIGURE_COLS = [
    ("inflation_growth", "CPI inflation growth (%, RPI-linked)"),
    ("gdp_growth", "GDP growth (%)"),
    ("gas_futures_price", "Gas futures price (pence/therm)"),
    ("electricity_demand_mean", "Electricity demand, monthly mean (MW)"),
    ("embedded_wind_generation_mean", "Embedded wind generation, monthly mean (MW)"),
    ("embedded_solar_generation_mean", "Embedded solar generation, monthly mean (MW)"),
    ("gbp_eur_rate", "GBP/EUR exchange rate"),
    ("weather_volatility", "Weather volatility (temperature-anomaly based)"),
    ("holiday_share", "Holiday share of days in month"),
]

_CORE_FIGURE_COLS = [
    ("gas_growth", "Gas price growth (%, YoY)"),
    ("electricity_index", "Electricity price index (2015=100)"),
    ("electricity_growth", "Electricity price growth (%, YoY)"),
    ("carbon_growth", "Carbon price growth (%, YoY)"),
]


def _save_csv(df: pd.DataFrame, name: str) -> None:
    paths.DATA_DESC_TABLES.mkdir(parents=True, exist_ok=True)
    p = paths.DATA_DESC_TABLES / f"{name}.csv"
    df.to_csv(p, index=False)
    log.info("Saved %s", p.name)


def _save_fig(fig: plt.Figure, name: str) -> None:
    paths.DATA_DESC_FIGURES.mkdir(parents=True, exist_ok=True)
    p = paths.DATA_DESC_FIGURES / f"{name}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    log.info("Saved figure %s", p.name)


def _summary_stats(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """count/mean/std/min/quartiles/max plus skew and excess kurtosis
    (scipy.stats, bias-corrected=False to match pandas' own convention) --
    the two shape statistics are what actually distinguish, e.g.,
    carbon_growth's heavy-tailed distribution from gas_growth's milder one,
    beyond what mean/std alone shows (reports/05_data_description.md
    Section 3.1 already narrates this qualitatively for carbon_growth; this
    table makes it quantitative and extends it to every core/macro column)."""
    rows = []
    for c in cols:
        s = df[c].dropna()
        if s.empty:
            continue
        rows.append({
            "variable": c,
            "n": int(s.shape[0]),
            "n_missing": int(df[c].isna().sum()),
            "mean": s.mean(),
            "std": s.std(),
            "min": s.min(),
            "p25": s.quantile(0.25),
            "median": s.median(),
            "p75": s.quantile(0.75),
            "max": s.max(),
            "skew": sp_stats.skew(s),
            "excess_kurtosis": sp_stats.kurtosis(s),
        })
    return pd.DataFrame(rows)


def _plot_distribution_grid(df: pd.DataFrame, specs: list[tuple[str, str]],
                             color: str, suptitle: str, name: str) -> None:
    specs = [(c, label) for c, label in specs if c in df.columns]
    n = len(specs)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.8 * nrows))
    fig.patch.set_facecolor("white")
    axes = np.atleast_1d(axes).flatten()

    for ax, (col, label) in zip(axes, specs):
        s = df[col].dropna()
        ax.hist(s, bins=40, color=color, alpha=0.85, edgecolor="white", linewidth=0.3)
        ax.axvline(s.median(), color=_PALETTE["median"], linestyle="--", linewidth=1.3,
                   label=f"median={s.median():.2f}")
        ax.axvline(s.mean(), color="#7F3C98", linestyle=":", linewidth=1.3,
                   label=f"mean={s.mean():.2f}")
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_ylabel("Months")
        ax.legend(fontsize=7)
        ax.grid(axis="y", color=_PALETTE["grid"])
    for ax in axes[len(specs):]:
        ax.set_axis_off()

    fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save_fig(fig, name)


# =============================================================================
# Core variables (data/processed/core_energy_carbon.csv, from data/raw)
# =============================================================================

def run_core() -> pd.DataFrame:
    df = pd.read_csv(paths.CORE_CSV, parse_dates=["date"])
    cols = [c for c in df.columns if c != "date"]

    summary = _summary_stats(df, cols)
    _save_csv(summary, "core_variable_summary_stats")

    _plot_distribution_grid(
        df, _CORE_FIGURE_COLS, _PALETTE["core"],
        "Core Variable Distributions (gas / electricity / carbon, monthly, "
        f"{df['date'].min():%Y-%m}–{df['date'].max():%Y-%m})",
        "core_variable_distributions",
    )
    return summary


# =============================================================================
# Macro variables (data/processed/macro_controls.csv, from data/raw)
# =============================================================================

def run_macro() -> pd.DataFrame:
    df = pd.read_csv(paths.MACRO_CSV, parse_dates=["date"])
    cols = [
        c for c in df.columns
        if c not in _MACRO_EXCLUDE_COLS and not c.endswith(_MACRO_EXCLUDE_SUFFIXES)
    ]

    summary = _summary_stats(df, cols)
    _save_csv(summary, "macro_variable_summary_stats")

    _plot_distribution_grid(
        df, _MACRO_FIGURE_COLS, _PALETTE["macro"],
        "Macro Variable Distributions (exogenous regressors, monthly, "
        f"{df['date'].min():%Y-%m}–{df['date'].max():%Y-%m})",
        "macro_variable_distributions",
    )
    return summary


# =============================================================================
# Main entry point
# =============================================================================

def run() -> None:
    log.info("Data description overview: core + macro variable distributions...")
    run_core()
    run_macro()
    log.info("Data description figures/tables saved -> %s", paths.DATA_DESC_OUT)


if __name__ == "__main__":
    paths.ensure_dirs()
    run()
