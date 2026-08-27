"""
ukhls_policy_maps.py
──────────────────────
Stage 4 — policy geography maps: three visualisations connecting physical
resource depletion (Stage 2b's SEM Baseline Resource Stock) to
psychological loss-spiral risk (Stage 3's fuzzy vulnerability membership)
and the forecasted energy-price shock (FES Magnitude/Delta), aggregated to
the 12 UK Government Office Regions (`gor_dv`, `src.ukhls_mapping.GOR_LABELS`
— the only geography granularity available in this End User Licence data).

Design: real UK region boundaries, via src.ukhls_geo_maps
────────────────────────────────────────────────────────────
This environment has genuine outbound internet access (confirmed via a
direct `curl` test) -- so unlike the module's first version, these maps
are drawn on real UK NUTS1/Government-Office-Region boundaries
(`data/geo/uk_nuts1_regions.geojson`, ONS Open Geography Portal, OGL
v3.0 -- see `src/ukhls_geo_maps.py`), not a self-contained hex-cartogram.
`config.GOR_HEX_POSITIONS` is left in `config.py`, unused now, in case a
future environment lacks the boundary file or network access.

Adaptation flagged explicitly (not silently reinterpreted)
──────────────────────────────────────────────────────────
Map 1 substitutes regional vulnerability-outcome prevalence for the
originally-proposed "FES axis" — FES in this project is a single NATIONAL
scalar (one UK-wide forecast), with no regional variation to map. The
substitution preserves the underlying policy intent (low-resource +
high-vulnerability -> cash-transfer zones; high-resource +
high-vulnerability -> structural/infrastructure zones) using data that
actually varies spatially.

The three maps
──────────────
  1. Resource-to-Stress Hotspot Map — three real choropleths side by side:
     two continuous single-variable maps (mean SEM baseline_resource_score;
     regional high_fuel_vulnerable prevalence) for precise values, plus a
     third real bivariate-tier map (discrete 3x3 classification, same
     colour palette as its inset legend) that directly answers "which
     policy tier is this region in" without requiring a reader to
     cross-reference a disconnected swatch grid against the two colour maps.
  2. Fuzzy Membership Map — real choropleth of mean "Vulnerable to Loss"
     fuzzy membership (the % near the 0.5 boundary stays in the saved
     table, not baked into the map label).
  3. Vulnerability Vector Shift Map — arrow at each region's real
     centroid: current mean predicted vulnerability (under each
     household's own realised fes_current) -> forecast mean (under the
     shared fes_magnitude shock), reusing
     `src.ukhls_cor_cvae.counterfactual_fes_shift()`'s output.

Usage
─────
  from src.ukhls_policy_maps import run
  run(df, sem_scores, vuln_result, counterfactual_df)
"""

from __future__ import annotations

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrow
import pandas as pd

warnings.filterwarnings("ignore")

from src import config, paths
from src.logging_utils import get_logger
from src.ukhls_mapping import GOR_LABELS
from src.ukhls_geo_maps import load_region_boundaries, plot_choropleth, ATTRIBUTION, REGION_ABBREV

log = get_logger("ukhls_policy_maps")

_PALETTE = {"depleted": "#E74C3C", "vulnerable": "#E67E22", "resilient": "#27AE60",
            "grid": "#EAECEE", "neutral": "#DADFE1"}

# 3x3 bivariate palette: rows = baseline-resource tertile (0=low..2=high),
# cols = vulnerability tertile (0=low..2=high). Bottom-right corner (low
# baseline, high vulnerability) is dark red (cash-transfer zone); top-right
# corner (high baseline, high vulnerability) is purple (structural/
# infrastructure zone) — matching the colour language of the policy
# framework this map operationalises.
_BIVAR_PALETTE: dict[tuple[int, int], str] = {
    (0, 0): "#F2F0F7", (1, 0): "#DADAEB", (2, 0): "#BCBDDC",
    (0, 1): "#F4A582", (1, 1): "#C994C7", (2, 1): "#9E9AC8",
    (0, 2): "#B2182B", (1, 2): "#DD3497", (2, 2): "#6A51A3",
}


# =============================================================================
# I/O helpers (module-local, matching the convention every ukhls_* module uses)
# =============================================================================

def _save_csv(df: pd.DataFrame, name: str) -> None:
    paths.UKHLS_POLICY_MAPS_TABLES.mkdir(parents=True, exist_ok=True)
    p = paths.UKHLS_POLICY_MAPS_TABLES / f"{name}.csv"
    df.to_csv(p, index=False)
    log.info("Saved %s", p.name)


def _save_fig(fig: plt.Figure, name: str) -> None:
    paths.UKHLS_POLICY_MAPS_FIGURES.mkdir(parents=True, exist_ok=True)
    p = paths.UKHLS_POLICY_MAPS_FIGURES / f"{name}.{config.FIGURE_FORMAT}"
    fig.savefig(p, dpi=config.DPI, bbox_inches="tight")
    fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    log.info("Saved figure %s", p.name)


def _region_col(df: pd.DataFrame) -> pd.Series:
    return df["gor_dv"].map(GOR_LABELS)


def _tertile_codes(s: pd.Series) -> pd.Series:
    """0/1/2 tertile codes, degrading gracefully (duplicates="drop") when
    too few distinct values exist among only 12 regions."""
    try:
        codes = pd.qcut(s, 3, labels=False, duplicates="drop")
    except ValueError:
        return pd.Series(1, index=s.index)  # degenerate: everyone "mid"
    return codes.fillna(1).astype(int).clip(0, 2)


_LABEL_OFFSET_FRAC = {"London": (0.10, -0.11)}


def _label_regions(ax, merged: pd.DataFrame) -> None:
    """Abbreviated region-name label at each centroid, white-outlined so it
    reads over both light and dark fills regardless of which colormap/tier
    colour the region has. London is geographically tiny and sits inside
    South East, so its label is nudged clear with a leader line (same fix
    used by the vector-shift map below and by ukhls_geo_maps.plot_choropleth)."""
    minx, miny, maxx, maxy = merged.total_bounds
    map_extent = max(maxx - minx, maxy - miny)
    for _, row in merged.iterrows():
        c = row.geometry.centroid
        label_x, label_y = c.x, c.y
        if row["region"] in _LABEL_OFFSET_FRAC:
            dx_frac, dy_frac = _LABEL_OFFSET_FRAC[row["region"]]
            label_x = c.x + dx_frac * map_extent
            label_y = c.y + dy_frac * map_extent
            ax.plot([c.x, label_x], [c.y, label_y], color="#7F8C8D", linewidth=0.6, zorder=4)
        abbrev = REGION_ABBREV.get(row["region"], row["region"])
        ax.annotate(abbrev, xy=(label_x, label_y), ha="center", va="center",
                    fontsize=7, fontweight="bold", color="black", zorder=5,
                    path_effects=[pe.withStroke(linewidth=2.2, foreground="white")])


# =============================================================================
# Map 1 — Resource-to-Stress Hotspot Map (bivariate)
# =============================================================================


def plot_hotspot_map(df: pd.DataFrame, sem_scores: pd.DataFrame) -> pd.DataFrame:
    """Two side-by-side real UK maps -- mean SEM baseline_resource_score,
    and regional high_fuel_vulnerable prevalence -- substituted for the
    originally proposed single bivariate "FES axis" map, since FES here is
    a single national scalar with no regional variation (see module
    docstring), plus the 3x3 bivariate tier classification (which regions
    are cash-transfer vs. structural-infrastructure priority) as a table
    and legend."""
    work = df[["gor_dv", "high_fuel_vulnerable"]].join(
        sem_scores[["baseline_resource_score"]], how="left"
    )
    work["region"] = _region_col(work)
    by_region = work.groupby("region").agg(
        baseline_resource_mean=("baseline_resource_score", "mean"),
        vulnerable_pct=("high_fuel_vulnerable", lambda s: 100 * s.mean(skipna=True)),
        n=("high_fuel_vulnerable", "size"),
    ).reset_index()

    by_region["baseline_tertile"] = _tertile_codes(by_region["baseline_resource_mean"])
    by_region["vulnerable_tertile"] = _tertile_codes(by_region["vulnerable_pct"])
    tier_names = {0: "Low", 1: "Mid", 2: "High"}
    by_region["policy_tier"] = [
        f"{tier_names[b]} resource / {tier_names[v]} vulnerability"
        for b, v in zip(by_region["baseline_tertile"], by_region["vulnerable_tertile"])
    ]
    _save_csv(by_region, "map1_resource_stress_hotspot")

    gdf = load_region_boundaries()
    merged = gdf.merge(by_region, on="region", how="left")
    merged["tier_color"] = [
        _BIVAR_PALETTE.get((int(b), int(v)), "#DADFE1") if pd.notna(b) and pd.notna(v) else "#DADFE1"
        for b, v in zip(merged["baseline_tertile"], merged["vulnerable_tertile"])
    ]

    fig, axes = plt.subplots(1, 4, figsize=(22, 8), gridspec_kw={"width_ratios": [4, 4, 4, 1.4]})
    fig.patch.set_facecolor("white")

    merged.plot(column="baseline_resource_mean", cmap="PuBuGn", linewidth=0.8,
                edgecolor="white", ax=axes[0], legend=True,
                legend_kwds={"label": "Baseline resource score", "shrink": 0.55})
    axes[0].set_title("Baseline Resource Stock\n(COR-SEM, mean by region)", fontsize=11, fontweight="bold")
    axes[0].set_axis_off()
    _label_regions(axes[0], merged)

    merged.plot(column="vulnerable_pct", cmap="YlOrRd", linewidth=0.8,
                edgecolor="white", ax=axes[1], legend=True,
                legend_kwds={"label": "% vulnerable", "shrink": 0.55})
    axes[1].set_title("Vulnerability Prevalence\n(% high_fuel_vulnerable)", fontsize=11, fontweight="bold")
    axes[1].set_axis_off()
    _label_regions(axes[1], merged)

    # Third map: the ACTUAL bivariate policy-tier classification, drawn
    # with the same _BIVAR_PALETTE colours as the inset legend below --
    # previously this panel was a bare 3x3 swatch grid with no
    # corresponding real-map colours anywhere in the figure, so a reader
    # could not tell which tier a region colour on maps 1/2 belonged to.
    # This map is now the direct visual answer to "which policy tier is
    # this region in," with maps 1/2 kept alongside as the precise
    # continuous-value evidence the tiers were cut from.
    merged.plot(ax=axes[2], color=merged["tier_color"], edgecolor="white", linewidth=0.8)
    axes[2].set_title("Policy Tier\n(bivariate: resource x vulnerability)", fontsize=11, fontweight="bold")
    axes[2].set_axis_off()
    _label_regions(axes[2], merged)

    # Standalone 3x3 legend, same colours as map 3 -- kept as its own panel
    # rather than an inset, since Britain's landmass fills nearly the whole
    # map (no empty ocean corner to tuck a legend into without overlap).
    ax_legend = axes[3]
    ax_legend.set_facecolor("white")
    for b in range(3):
        for v in range(3):
            ax_legend.add_patch(plt.Rectangle((b, v), 1, 1, facecolor=_BIVAR_PALETTE[(b, v)]))
    ax_legend.set_xlim(0, 3)
    ax_legend.set_ylim(0, 3)
    ax_legend.set_xticks([0.5, 1.5, 2.5]); ax_legend.set_xticklabels(["Low", "Mid", "High"], fontsize=8)
    ax_legend.set_yticks([0.5, 1.5, 2.5]); ax_legend.set_yticklabels(["Low", "Mid", "High"], fontsize=8)
    ax_legend.set_xlabel("Resource", fontsize=8)
    ax_legend.set_ylabel("Vulnerability", fontsize=8)
    ax_legend.set_title("Legend", fontsize=9, fontweight="bold")
    ax_legend.tick_params(length=0)
    ax_legend.set_aspect("equal")
    for spine in ax_legend.spines.values():
        spine.set_color("#999999")

    fig.suptitle("Resource-to-Stress Hotspot: Baseline Resource Stock x Vulnerability Prevalence",
                 fontsize=13, fontweight="bold")
    fig.text(0.5, 0.03,
              "Dark red = low resource, high vulnerability (cash-transfer priority)   |   "
              "Purple = high resource, high vulnerability (structural/infrastructure priority)",
              fontsize=8, ha="center")
    fig.text(0.01, 0.01, ATTRIBUTION, fontsize=6, color="#7F8C8D")
    fig.subplots_adjust(bottom=0.12, wspace=0.15)
    _save_fig(fig, "policy_map1_resource_stress_hotspot")
    return by_region


# =============================================================================
# Map 2 — Fuzzy Membership Map
# =============================================================================

def plot_fuzzy_membership_map(df: pd.DataFrame, fuzzy_df: pd.DataFrame) -> pd.DataFrame:
    """Real UK choropleth of mean 'Vulnerable to Loss' fuzzy membership.
    The % of each region's households near the 0.5 boundary (the "about
    to tip" households) is saved in the table but not baked into the map
    label, to keep the map itself to one clean number per region."""
    work = df[["gor_dv"]].join(fuzzy_df[["fuzzy_vulnerable_to_loss"]], how="left")
    work["region"] = _region_col(work)
    by_region = work.groupby("region").agg(
        mean_vulnerable_to_loss=("fuzzy_vulnerable_to_loss", "mean"),
        pct_near_boundary=(
            "fuzzy_vulnerable_to_loss",
            lambda s: 100 * ((s - 0.5).abs() <= config.FES_BOUNDARY_ZONE).mean(skipna=True),
        ),
        n=("fuzzy_vulnerable_to_loss", "size"),
    ).reset_index()

    plot_choropleth(
        by_region, value_col="mean_vulnerable_to_loss",
        title="Fuzzy Membership Map\nMean 'Vulnerable to Loss' membership by region",
        out_name="policy_map2_fuzzy_membership",
        tables_dir=paths.UKHLS_POLICY_MAPS_TABLES, figures_dir=paths.UKHLS_POLICY_MAPS_FIGURES,
        cmap="YlOrRd", cbar_label="Mean 'Vulnerable to Loss' membership", fmt="{:.2f}",
    )
    return by_region


# =============================================================================
# Map 3 — Vulnerability Vector Shift Map
# =============================================================================

def plot_vector_shift_map(counterfactual_df: pd.DataFrame) -> pd.DataFrame:
    """Arrow at each region's real centroid: current mean predicted
    vulnerability (household's own realised fes_current) -> forecast mean
    (shared fes_magnitude shock), reusing
    src.ukhls_cor_cvae.counterfactual_fes_shift()'s output. The arrow is a
    schematic (length/colour = shift magnitude/direction), not a literal
    geographic displacement -- the region itself doesn't move."""
    if counterfactual_df.empty or "gor_dv" not in counterfactual_df.columns:
        log.warning("No counterfactual data (with gor_dv) available -- "
                    "Map 3 (Vulnerability Vector Shift) skipped.")
        return pd.DataFrame()

    work = counterfactual_df.copy()
    work["region"] = _region_col(work)
    by_region = work.groupby("region").agg(
        mean_current=("pred_prob_current", "mean"),
        mean_forecast=("pred_prob_forecast", "mean"),
        mean_shift=("pred_prob_shift", "mean"),
        n=("pred_prob_shift", "size"),
    ).reset_index()
    _save_csv(by_region, "map3_vulnerability_vector_shift")

    gdf = load_region_boundaries()
    merged = gdf.merge(by_region, on="region", how="left")

    fig, ax = plt.subplots(figsize=(8, 10))
    fig.patch.set_facecolor("white")
    merged.plot(ax=ax, color=_PALETTE["neutral"], edgecolor="white", linewidth=0.8)

    # Geographic extent varies a lot across UK regions (Scotland is huge,
    # London tiny) -- scale the arrow length to a fixed fraction of the
    # OVERALL map extent, not per-region size, so arrows stay comparable.
    minx, miny, maxx, maxy = merged.total_bounds
    map_extent = max(maxx - minx, maxy - miny)
    arrow_scale = 0.08 * map_extent
    shift_max = float(by_region["mean_shift"].abs().max(skipna=True)) or 1.0

    # London is geographically tiny and sits inside South East -- a
    # same-length arrow at its true centroid collides with South East's.
    # Nudge it into open space with a thin leader line (module-level
    # _LABEL_OFFSET_FRAC, shared with _label_regions above).
    for _, row in merged.iterrows():
        if pd.isna(row["mean_shift"]):
            continue
        c = row.geometry.centroid
        label_x, label_y = c.x, c.y
        if row["region"] in _LABEL_OFFSET_FRAC:
            dx_frac, dy_frac = _LABEL_OFFSET_FRAC[row["region"]]
            label_x = c.x + dx_frac * map_extent
            label_y = c.y + dy_frac * map_extent
            ax.plot([c.x, label_x], [c.y, label_y], color="#7F8C8D", linewidth=0.7, zorder=4)

        arrow_len = arrow_scale * (row["mean_shift"] / shift_max)
        arrow_color = _PALETTE["depleted"] if row["mean_shift"] >= 0 else _PALETTE["resilient"]
        ax.add_patch(FancyArrow(
            label_x - arrow_len / 2, label_y, arrow_len, 0,
            width=map_extent * 0.004, head_width=map_extent * 0.018, head_length=map_extent * 0.015,
            length_includes_head=True, facecolor=arrow_color, edgecolor=arrow_color, zorder=5,
        ))
        ax.annotate(f"{row['mean_shift']*100:+.2f}pp", xy=(label_x, label_y - map_extent * 0.016),
                    ha="center", fontsize=6, fontweight="bold", color="black")

    ax.set_title(
        "Vulnerability Vector Shift Map\nPredicted vulnerability: current realised FES -> forecast FES shock\n"
        "(red = rising risk, green = falling; SIMULATION on the trained CVAE, not an observation)",
        fontsize=11, fontweight="bold", pad=10,
    )
    ax.set_axis_off()
    fig.text(0.01, 0.01, ATTRIBUTION, fontsize=6, color="#7F8C8D")
    _save_fig(fig, "policy_map3_vulnerability_vector_shift")
    return by_region


# =============================================================================
# Main entry point
# =============================================================================

def run(
    df: pd.DataFrame, sem_scores: pd.DataFrame,
    vuln_result: dict, counterfactual_df: pd.DataFrame,
) -> dict:
    """
    Stage 4 full pipeline: the three policy geography maps.

    Parameters
    ----------
    df                : UKHLS household-wave panel (needs gor_dv,
                         high_fuel_vulnerable)
    sem_scores        : Stage 2b COR-SEM factor scores (needs
                         baseline_resource_score)
    vuln_result       : Stage 3 result dict (needs "fuzzy")
    counterfactual_df : Stage 2c's counterfactual FES shift table (needs
                         gor_dv, pred_prob_current, pred_prob_forecast,
                         pred_prob_shift) -- empty DataFrame if Stage 2c's
                         counterfactual query was skipped.
    """
    paths.UKHLS_POLICY_MAPS_TABLES.mkdir(parents=True, exist_ok=True)
    paths.UKHLS_POLICY_MAPS_FIGURES.mkdir(parents=True, exist_ok=True)

    log.info("Stage 4: Map 1 (resource-to-stress hotspot)...")
    map1 = plot_hotspot_map(df, sem_scores)

    log.info("Stage 4: Map 2 (fuzzy membership)...")
    fuzzy_df = vuln_result.get("fuzzy", pd.DataFrame())
    map2 = plot_fuzzy_membership_map(df, fuzzy_df) if not fuzzy_df.empty else pd.DataFrame()
    if fuzzy_df.empty:
        log.warning("No fuzzy c-means output available -- Map 2 skipped.")

    log.info("Stage 4: Map 3 (vulnerability vector shift)...")
    map3 = plot_vector_shift_map(counterfactual_df)

    log.info("Stage 4 (policy geography maps) complete.")
    return {"map1_hotspot": map1, "map2_fuzzy_membership": map2, "map3_vector_shift": map3}
