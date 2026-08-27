"""
ukhls_geo_maps.py
────────────────────
Real UK region boundary choropleth helper, shared by Stage 3
(src.ukhls_vulnerability_classification) and Stage 4 (src.ukhls_policy_maps).

Boundary data
─────────────
data/geo/uk_nuts1_regions.geojson -- ONS Open Geography Portal, "NUTS,
level 1 (January 2018) Boundaries UK BUC" (generalised/clipped resolution,
~712KB, downloaded via the portal's public ArcGIS FeatureServer). Exactly
the 12 areas UKHLS's gor_dv/GOR_LABELS already use (9 English regions +
Wales + Scotland + Northern Ireland) -- confirmed by direct inspection,
not assumed.

Licence: Office for National Statistics / Ordnance Survey data, released
under the Open Government Licence v3.0. Attribution (required by the
licence) is rendered directly on every figure this module produces, and
repeated in README.md's Stage 3/4 sections -- do not strip it.

Source registered here for a full pipeline re-run: this file is a
one-time download, not re-fetched by the pipeline itself (no network
access is assumed at pipeline-run time, only when this file was first
added to the repo).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd

from src import paths
from src.logging_utils import get_logger

log = get_logger("ukhls_geo_maps")

GEO_FILE = paths.ROOT / "data" / "geo" / "uk_nuts1_regions.geojson"

ATTRIBUTION = "Boundaries: ONS Open Geography Portal, NUTS1 (Jan 2018) Boundaries UK BUC -- OGL v3.0"

# nuts118nm (source) -> GOR_LABELS value (src.ukhls_mapping) -- the source
# file spells "The" with a capital T and appends "(England)" to some
# regions; UKHLS's own registry does not.
_NAME_FIXUPS = {
    "Yorkshire and The Humber": "Yorkshire and the Humber",
    "North East (England)": "North East",
    "North West (England)": "North West",
    "East Midlands (England)": "East Midlands",
    "West Midlands (England)": "West Midlands",
    "South East (England)": "South East",
    "South West (England)": "South West",
}


# Short codes for the 12 GOR_LABELS regions -- used to label maps where
# spelling out the full region name at each (often small/crowded) centroid
# would overlap; not an official ONS standard, just this project's own
# shorthand, kept in one place so every map uses the same abbreviations.
REGION_ABBREV: dict[str, str] = {
    "North East": "NE",
    "North West": "NW",
    "Yorkshire and the Humber": "YH",
    "East Midlands": "EM",
    "West Midlands": "WM",
    "East of England": "EE",
    "London": "LDN",
    "South East": "SE",
    "South West": "SW",
    "Wales": "WAL",
    "Scotland": "SCO",
    "Northern Ireland": "NI",
}


def load_region_boundaries() -> gpd.GeoDataFrame:
    """Load the 12 UK region boundaries, renamed to match GOR_LABELS values."""
    gdf = gpd.read_file(GEO_FILE)
    gdf["region"] = gdf["nuts118nm"].replace(_NAME_FIXUPS)
    return gdf[["region", "geometry"]]


def plot_choropleth(
    by_region: pd.DataFrame,
    value_col: str,
    title: str,
    out_name: str,
    tables_dir,
    figures_dir,
    region_col: str = "region",
    cmap: str = "YlOrRd",
    cbar_label: str | None = None,
    diverging: bool = False,
    fmt: str = "{:.1f}",
) -> gpd.GeoDataFrame:
    """
    Generic real-boundary choropleth. Merges `by_region` onto the UK
    region boundaries (matched by region name against GOR_LABELS values),
    colours by `value_col`, labels each region with its value at its
    centroid.

    Returns the merged GeoDataFrame (also saved as a CSV) so callers can
    verify the merge matched every region -- a name mismatch produces a
    silently blank/grey region, not an exception, so this is logged
    explicitly (see `n_missing` below) rather than left to be discovered
    visually.
    """
    gdf = load_region_boundaries()
    merged = gdf.merge(by_region, left_on="region", right_on=region_col, how="left")

    n_missing = int(merged[value_col].isna().sum())
    if n_missing:
        log.warning(
            "plot_choropleth(%s): %d/%d regions have no data after merge -- "
            "check region name matching (load_region_boundaries' _NAME_FIXUPS).",
            out_name, n_missing, len(merged),
        )

    fig, ax = plt.subplots(figsize=(8, 10))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    plot_cmap = "RdYlGn_r" if diverging else cmap
    vmin = vmax = None
    if diverging:
        vext = float(merged[value_col].abs().max(skipna=True)) or 1.0
        vmin, vmax = -vext, vext

    merged.plot(
        column=value_col, cmap=plot_cmap, linewidth=0.8, edgecolor="white",
        ax=ax, legend=True, missing_kwds={"color": "#DADFE1", "label": "No data"},
        vmin=vmin, vmax=vmax,
        legend_kwds={"label": cbar_label or value_col, "shrink": 0.6},
    )
    # London is geographically tiny and sits inside South East -- a label
    # at its true centroid collides with South East's label on every map
    # at this resolution, so it gets a small offset + leader line (same
    # fix already used for the vector-shift map in ukhls_policy_maps.py).
    minx, miny, maxx, maxy = merged.total_bounds
    map_extent = max(maxx - minx, maxy - miny)
    _label_offset_frac = {"London": (0.10, -0.11)}

    for _, row in merged.iterrows():
        if pd.notna(row[value_col]):
            c = row.geometry.centroid
            label_x, label_y = c.x, c.y
            if row["region"] in _label_offset_frac:
                dx_frac, dy_frac = _label_offset_frac[row["region"]]
                label_x = c.x + dx_frac * map_extent
                label_y = c.y + dy_frac * map_extent
                ax.plot([c.x, label_x], [c.y, label_y], color="#7F8C8D", linewidth=0.7, zorder=4)
            abbrev = REGION_ABBREV.get(row["region"], row["region"])
            ax.annotate(f"{abbrev}\n{fmt.format(row[value_col])}", xy=(label_x, label_y), ha="center",
                        fontsize=6.5, fontweight="bold", color="black", zorder=5,
                        linespacing=1.1, path_effects=None)

    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_axis_off()
    fig.text(0.01, 0.01, ATTRIBUTION, fontsize=6, color="#7F8C8D")

    Path(tables_dir).mkdir(parents=True, exist_ok=True)
    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    merged.drop(columns="geometry").to_csv(f"{tables_dir}/{out_name}.csv", index=False)

    fig.savefig(f"{figures_dir}/{out_name}.png", dpi=150, bbox_inches="tight")
    fig.savefig(f"{figures_dir}/{out_name}.pdf", bbox_inches="tight")
    plt.close(fig)
    log.info("Saved choropleth %s (%d/%d regions matched)", out_name, len(merged) - n_missing, len(merged))
    return merged
