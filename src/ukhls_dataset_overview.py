"""
ukhls_dataset_overview.py
──────────────────────────
Descriptive "introduce the dataset" figures for the UKHLS household-wave
panel -- no modeling, just what's actually in the data before any COR-SEM/
CVAE/vulnerability machinery touches it: how big each wave is, when
households were actually interviewed (year AND month -- relevant now that
`src.ukhls_preprocessing.attach_fes_delta` uses month-level FES matching),
where they are, how complete each variable is, and what the key variables'
distributions look like.

Usage
─────
  from src.ukhls_dataset_overview import run
  run(df)   # df from ukhls_preprocessing.run(), before or after SEM/CVAE
"""

from __future__ import annotations

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from src import paths
from src.logging_utils import get_logger
from src.ukhls_mapping import GOR_LABELS
from src.ukhls_geo_maps import plot_choropleth

log = get_logger("ukhls_dataset_overview")

_PALETTE = {"bar": "#2E86AB", "grid": "#EAECEE", "accent": "#E67E22"}


def _save_csv(df: pd.DataFrame, name: str) -> None:
    paths.UKHLS_OVERVIEW_TABLES.mkdir(parents=True, exist_ok=True)
    p = paths.UKHLS_OVERVIEW_TABLES / f"{name}.csv"
    df.to_csv(p, index=False)
    log.info("Saved %s", p.name)


def _save_fig(fig: plt.Figure, name: str) -> None:
    paths.UKHLS_OVERVIEW_FIGURES.mkdir(parents=True, exist_ok=True)
    p = paths.UKHLS_OVERVIEW_FIGURES / f"{name}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    log.info("Saved figure %s", p.name)


# =============================================================================
# Figure 1 — Panel composition: wave sizes, interview timing, geography
# =============================================================================

def plot_panel_composition(df: pd.DataFrame) -> None:
    by_wave = df.groupby("wave").agg(
        n=("hidp", "size"), year=("interview_year", "median"),
    ).reset_index().sort_values("wave")
    _save_csv(by_wave, "dataset_rows_by_wave")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("white")

    axes[0].bar(by_wave["wave"], by_wave["n"], color=_PALETTE["bar"], alpha=0.85)
    for _, r in by_wave.iterrows():
        axes[0].annotate(f"{r['year']:.0f}", (r["wave"], r["n"]),
                          textcoords="offset points", xytext=(0, 4), ha="center", fontsize=7, color="#555")
    axes[0].set_xlabel("UKHLS wave")
    axes[0].set_ylabel("Household-wave rows")
    axes[0].set_title("Sample size by wave", fontsize=11, fontweight="bold")
    axes[0].grid(axis="y", color=_PALETTE["grid"])

    month_counts = df["interview_month"].value_counts().reindex(range(1, 13), fill_value=0)
    axes[1].bar(month_counts.index, month_counts.values, color=_PALETTE["accent"], alpha=0.85)
    axes[1].set_xticks(range(1, 13))
    axes[1].set_xlabel("Calendar month of interview")
    axes[1].set_ylabel("Household-wave rows")
    axes[1].set_title("Interview timing by calendar month\n(pooled across all waves)", fontsize=11, fontweight="bold")
    axes[1].grid(axis="y", color=_PALETTE["grid"])

    fig.suptitle("UKHLS Panel Composition", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save_fig(fig, "dataset_panel_composition")

    work = df.copy()
    work["region"] = work["gor_dv"].map(GOR_LABELS)
    by_region = work.groupby("region").agg(n=("hidp", "size")).reset_index()
    _save_csv(by_region, "dataset_rows_by_region")
    plot_choropleth(
        by_region, value_col="n",
        title="UKHLS Sample Size by UK Region\n(household-wave rows, all 15 waves pooled)",
        out_name="dataset_sample_size_by_region",
        tables_dir=paths.UKHLS_OVERVIEW_TABLES, figures_dir=paths.UKHLS_OVERVIEW_FIGURES,
        cmap="Blues", cbar_label="Household-wave rows", fmt="{:,.0f}",
    )


def plot_yearly_distribution(df: pd.DataFrame) -> None:
    """How the panel is distributed through calendar years -- distinct from
    'by wave' above: a UKHLS wave's ~24-month fieldwork window can straddle
    a year boundary, so 'by year' and 'by wave' are genuinely different
    views (and 'by year' is what interview_year-keyed joins like FES
    Magnitude/Delta actually operate on)."""
    work = df.copy()
    work["year"] = work["interview_year"].astype("Int64")
    by_year = work.groupby("year").agg(n=("hidp", "size")).reset_index()
    _save_csv(by_year, "dataset_rows_by_year")

    pivot = work.pivot_table(index="year", columns="wave", values="hidp", aggfunc="size", fill_value=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("white")

    axes[0].bar(by_year["year"].astype(str), by_year["n"], color=_PALETTE["bar"], alpha=0.85)
    axes[0].set_xlabel("Interview calendar year")
    axes[0].set_ylabel("Household-wave rows")
    axes[0].set_title("Sample size by calendar year", fontsize=11, fontweight="bold")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].grid(axis="y", color=_PALETTE["grid"])

    bottom = np.zeros(len(pivot))
    wave_cmap = plt.get_cmap("viridis")
    for i, wave in enumerate(pivot.columns):
        color = wave_cmap(i / max(len(pivot.columns) - 1, 1))
        axes[1].bar(pivot.index.astype(str), pivot[wave], bottom=bottom, color=color, label=wave, width=0.7)
        bottom += pivot[wave].values
    axes[1].set_xlabel("Interview calendar year")
    axes[1].set_ylabel("Household-wave rows")
    axes[1].set_title("Same totals, split by which wave\ncontributed each year's rows", fontsize=11, fontweight="bold")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].legend(title="Wave", fontsize=6, ncol=2, loc="upper right")
    axes[1].grid(axis="y", color=_PALETTE["grid"])

    fig.suptitle("UKHLS Panel Distribution Through the Years", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save_fig(fig, "dataset_rows_by_year")


# =============================================================================
# Figure 2 — Missingness across key variables
# =============================================================================

_MISSINGNESS_VARS = [
    "fuel_to_income_ratio", "high_fuel_vulnerable", "total_fuel_spend",
    "fihhmnnet1_dv", "fiyrinvinc_dv", "financial_strain_score",
    "tenure_security", "jbstat_security", "health_good", "sf1_good", "qfhigh_band",
    "hsval", "carval", "bill_security", "heatch_good",
    "inoutflows2", "inoutflows3", "inoutflows4",
]


def plot_missingness(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in _MISSINGNESS_VARS if c in df.columns]
    pct_missing = (df[cols].isna().mean() * 100).sort_values(ascending=False)
    out = pct_missing.reset_index()
    out.columns = ["variable", "pct_missing"]
    _save_csv(out, "dataset_missingness_by_variable")

    fig, ax = plt.subplots(figsize=(8, max(4, len(cols) * 0.35)))
    fig.patch.set_facecolor("white")
    colors = ["#E74C3C" if v > 50 else "#E67E22" if v > 10 else "#27AE60" for v in pct_missing.values]
    ax.barh(pct_missing.index[::-1], pct_missing.values[::-1], color=colors[::-1], alpha=0.85)
    for i, v in enumerate(pct_missing.values[::-1]):
        ax.text(v + 1, i, f"{v:.1f}%", va="center", fontsize=8)
    ax.set_xlabel("% missing (structural + item non-response, all waves pooled)")
    ax.set_title("Missingness by Variable", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.grid(axis="x", color=_PALETTE["grid"])
    fig.tight_layout()
    _save_fig(fig, "dataset_missingness")
    return out


# =============================================================================
# Figure 3 — Key variable distributions
# =============================================================================

_DIST_SPECS = [
    ("fuel_to_income_ratio", "Fuel-to-income ratio", False),
    ("financial_strain_score", "Financial/psychological strain (0-1)", False),
    ("dvage", "Age", False),
    ("fihhmnnet1_dv", "Household net monthly income (£, log10)", True),
    ("total_fuel_spend", "Annual fuel spend (£, log10)", True),
    ("fes_delta", "FES Delta (Magnitude - Current)", False),
]


def plot_key_distributions(df: pd.DataFrame) -> None:
    specs = [(c, label, log_scale) for c, label, log_scale in _DIST_SPECS if c in df.columns]
    n = len(specs)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.8 * nrows))
    fig.patch.set_facecolor("white")
    axes = np.atleast_1d(axes).flatten()

    for ax, (col, label, log_scale) in zip(axes, specs):
        s = df[col].dropna()
        if log_scale:
            s = np.log10(s[s > 0])
        ax.hist(s, bins=50, color=_PALETTE["bar"], alpha=0.85, edgecolor="white", linewidth=0.3)
        ax.axvline(s.median(), color=_PALETTE["accent"], linestyle="--", linewidth=1.2,
                   label=f"median={s.median():.2f}")
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_ylabel("Household-wave rows")
        ax.legend(fontsize=7)
        ax.grid(axis="y", color=_PALETTE["grid"])
    for ax in axes[len(specs):]:
        ax.set_axis_off()

    fig.suptitle("Key Variable Distributions (all waves pooled)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save_fig(fig, "dataset_key_distributions")


# =============================================================================
# Main entry point
# =============================================================================

def run(df: pd.DataFrame) -> None:
    log.info("Dataset overview: panel composition, missingness, key distributions...")
    plot_panel_composition(df)
    plot_yearly_distribution(df)
    plot_missingness(df)
    plot_key_distributions(df)
    log.info("Dataset overview figures saved -> %s", paths.UKHLS_OVERVIEW_FIGURES)
