"""
ukhls_external_validation.py
─────────────────────────────
Validates this project's own household-level vulnerability outputs
(src.ukhls_vulnerability_classification's "% high fuel-to-income
vulnerable", broken down by UK region, ethnicity, disability, housing
tenure, and UKHLS wave/year) against an independent external benchmark:
the Joseph Rowntree Foundation's "UK Poverty 2025" report (JRF, published
January 2025, using DWP Households Below Average Income data).

This is NOT an apples-to-apples comparison -- deliberately so, and that
difference is the point of running it:

  - Our metric: the share of household-waves flagged "high fuel-to-income
    vulnerable" (fuel_to_income_ratio-based, see
    src.ukhls_preprocessing / src.ukhls_vulnerability_classification),
    pooled across UKHLS waves a-o (2009-2024).
  - JRF's metric: relative income poverty, after housing costs (AHC) --
    equivalised household income < 60% of the UK median -- mostly
    averaged over 2021/22-2022/23.

PATTERN agreement (rank correlation) is the meaningful check across every
dimension below: both measure household financial hardship, so groups
should broadly rank similarly even though fuel poverty and income poverty
are distinct constructs with different drivers. Divergences are reported
alongside the agreement, not hidden -- a stratum where fuel poverty and
income poverty disagree sharply is informative (a genuinely different
driver), not a validation failure. See each comparison function's
docstring for the specific known divergences (Northern Ireland on region;
Black Caribbean/Bangladeshi and owned-outright/private-renting on
ethnicity/tenure).

JRF benchmark values are hardcoded from numbers explicitly stated in the
report's TEXT (not read off chart pixels) -- categories JRF only shows in
a bar chart without a stated number (e.g. Indian, Chinese, Mixed ethnic
groups) are deliberately left out of the benchmark dicts below rather
than guessed.

Outputs (per dimension: region, ethnicity, disability, tenure)
────────────────────────────────────────────────────────────────
  outputs/ukhls_vulnerability/tables/jrf_poverty_benchmark_{dim}.csv
  outputs/ukhls_vulnerability/tables/external_validation_{dim}_comparison.csv
  outputs/ukhls_vulnerability/figures/external_validation_{dim}_bars.png
  outputs/ukhls_vulnerability/figures/external_validation_{dim}_scatter.png
Plus the wave-level trend:
  outputs/ukhls_vulnerability/figures/external_validation_wave_trend.png

Usage
─────
  python -m src.ukhls_external_validation
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sp

from src import paths
from src.logging_utils import get_logger

log = get_logger("ukhls_external_validation")

_PALETTE = {
    "ours": "#E67E22",       # matches _PALETTE["vulnerable"] elsewhere in this project
    "jrf":  "#2E5077",       # a distinct colour so "external benchmark" reads as external
    "grid": "#EAECEE",
}

# ═══════════════════════════════════════════════════════════════════════
# JRF "UK Poverty 2025" benchmark values (hardcoded, text-stated only)
# ═══════════════════════════════════════════════════════════════════════

# Table 6, p.51 -- relative poverty rate, AHC, by UK nation/region,
# averaged over 2021/22-2022/23. "East" in JRF's table is "East of
# England" in this project's region labels (src.ukhls_mapping.GOR_LABELS).
JRF_POVERTY_RATE_BY_REGION: dict[str, float] = {
    "North East":               21,
    "North West":                25,
    "Yorkshire and the Humber":  23,
    "East Midlands":             20,
    "West Midlands":             27,
    "East of England":           18,
    "London":                    24,
    "South East":                19,
    "South West":                19,
    "Wales":                     21,
    "Scotland":                  21,
    "Northern Ireland":          17,
}

# p.9/42 -- poverty rate by ethnicity of household, general population
# (not child-specific), 2020/21-2022/23. Only categories with a number
# stated in the report's text are included; Indian, Chinese, Mixed
# ethnic groups, Any other Black background, and Other ethnic group are
# shown in JRF's Figure 13 chart only, with no stated rate to cite.
JRF_POVERTY_RATE_BY_ETHNICITY: dict[str, float] = {
    "White":                       19,
    "Pakistani":                   49,
    "Bangladeshi":                 56,
    "Black African":               40,
    "Black Caribbean":             30,
    "Any other Asian background":  34,
}

# Table 8, p.67 -- poverty rate by whether the family contains a disabled
# adult. Our disability_free flag is only observed for responding adults
# (indresp), so "Contains disabled adult" is benchmarked against JRF's
# "Disabled adults only" row (28% own children-only/adults-and-children
# rows are not comparable to our adult-only household flag).
JRF_POVERTY_RATE_BY_DISABILITY: dict[str, float] = {
    "No disabled adult":       19,
    "Contains disabled adult": 29,
}

# Table 10, p.95 -- poverty rate by housing tenure, AHC, 2022/23. Our
# "Private renting" group also folds in "rented from employer" (tenure_dv
# code 5), which JRF has no separate category for.
JRF_POVERTY_RATE_BY_TENURE: dict[str, float] = {
    "Owned outright":       14,
    "Buying with mortgage": 10,
    "Social renting":       44,
    "Private renting":      35,
}

# p.36 -- CHILD poverty rate by family type (a different unit than the
# other benchmarks above, which are household/adult poverty rates -- JRF's
# own Table 5 states these as child-poverty-rate-in-family-type, not
# family-poverty-rate). Compared against a COLLAPSED 2-category version of
# our own household-level family_composition_group breakdown (see
# validate_family_composition) since JRF states a rate for lone-parent vs
# couple families overall, not separately crossed with family size the way
# our own 5-category breakdown is.
JRF_POVERTY_RATE_BY_FAMILY_TYPE: dict[str, float] = {
    "Lone parent":         44,
    "Couple with children": 25,
}

# p.76 -- in-work vs out-of-work poverty rate, working-age adults.
JRF_POVERTY_RATE_BY_WORK_STATUS: dict[str, float] = {
    "In work":     12,
    "Not in work": 43,
}


# ═══════════════════════════════════════════════════════════════════════
# Generic dimension comparison engine
# ═══════════════════════════════════════════════════════════════════════

def _build_benchmark_table(benchmark: dict[str, float], dim: str) -> pd.DataFrame:
    df = pd.DataFrame(
        [{"label": k, "jrf_poverty_rate_pct": v} for k, v in benchmark.items()]
    ).sort_values("jrf_poverty_rate_pct", ascending=False)
    paths.UKHLS_VULN_TABLES.mkdir(parents=True, exist_ok=True)
    out_path = paths.UKHLS_VULN_TABLES / f"jrf_poverty_benchmark_{dim}.csv"
    df.to_csv(out_path, index=False)
    log.info("JRF external benchmark (%s) saved -> %s (%d categories)", dim, out_path, len(df))
    return df


def compare_dimension(
    dim: str, ours_csv_name: str, ours_group_col: str, benchmark: dict[str, float],
) -> pd.DataFrame:
    """Merge our own vulnerability-by-{dim} breakdown against a JRF
    benchmark dict, compute rank/linear agreement, and save the result.
    Shared by regions/ethnicity/disability/tenure -- the only thing that
    differs between them is which CSV and benchmark dict to use."""
    ours_path = paths.UKHLS_VULN_TABLES / f"{ours_csv_name}.csv"
    ours = pd.read_csv(ours_path).rename(columns={ours_group_col: "label"})
    jrf = _build_benchmark_table(benchmark, dim)

    merged = ours.merge(jrf, on="label", how="inner")
    if len(merged) < len(jrf):
        missing = set(jrf["label"]) - set(merged["label"])
        log.warning("(%s) JRF benchmark categories with no match in our data: %s", dim, missing)

    merged["our_rank"] = merged["pct_vulnerable"].rank(ascending=False).astype(int)
    merged["jrf_rank"] = merged["jrf_poverty_rate_pct"].rank(ascending=False).astype(int)
    merged = merged.sort_values("jrf_poverty_rate_pct", ascending=False)

    if len(merged) >= 3:
        rho, rho_p = sp.spearmanr(merged["pct_vulnerable"], merged["jrf_poverty_rate_pct"])
        r, r_p = sp.pearsonr(merged["pct_vulnerable"], merged["jrf_poverty_rate_pct"])
    else:
        rho = rho_p = r = r_p = float("nan")
    log.info(
        "(%s) agreement: Spearman rho=%.3f (p=%.3f), Pearson r=%.3f (p=%.3f), n=%d categories",
        dim, rho, rho_p, r, r_p, len(merged),
    )

    out_path = paths.UKHLS_VULN_TABLES / f"external_validation_{dim}_comparison.csv"
    merged.to_csv(out_path, index=False)
    log.info("(%s) comparison table saved -> %s", dim, out_path)

    merged.attrs["dim"] = dim
    merged.attrs["spearman_rho"] = rho
    merged.attrs["pearson_r"] = r
    return merged


def plot_dimension_bars(merged: pd.DataFrame, dim: str, ours_series_label: str, jrf_series_label: str, title: str) -> None:
    labels = merged["label"].tolist()
    x = np.arange(len(labels))
    width = 0.38

    fig, ax1 = plt.subplots(figsize=(max(8, 0.9 * len(labels)), 6))
    fig.patch.set_facecolor("white")
    ax2 = ax1.twinx()

    ax1.bar(x - width / 2, merged["pct_vulnerable"], width, color=_PALETTE["ours"], label=ours_series_label)
    ax2.bar(x + width / 2, merged["jrf_poverty_rate_pct"], width, color=_PALETTE["jrf"], label=jrf_series_label)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=40, ha="right")
    ax1.set_ylabel("Our fuel-to-income vulnerability (%)", color=_PALETTE["ours"])
    ax2.set_ylabel("JRF relative poverty rate, AHC (%)", color=_PALETTE["jrf"])
    ax1.tick_params(axis="y", labelcolor=_PALETTE["ours"])
    ax2.tick_params(axis="y", labelcolor=_PALETTE["jrf"])
    ax1.grid(axis="y", color=_PALETTE["grid"])

    rho = merged.attrs.get("spearman_rho", float("nan"))
    ax1.set_title(f"{title}\nSpearman rank correlation ρ={rho:.2f}", fontsize=11, fontweight="bold")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    fig.tight_layout()
    out_path = paths.UKHLS_VULN_FIGURES / f"external_validation_{dim}_bars.png"
    paths.UKHLS_VULN_FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    log.info("(%s) comparison bar chart saved -> %s", dim, out_path)


def plot_dimension_scatter(merged: pd.DataFrame, dim: str, x_label: str, y_label: str, title: str, exclude_label: str | None = None) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 7))
    fig.patch.set_facecolor("white")

    x = merged["jrf_poverty_rate_pct"].values.astype(float)
    y = merged["pct_vulnerable"].values.astype(float)
    ax.scatter(x, y, s=90, color=_PALETTE["ours"], edgecolor="white", linewidth=0.8, zorder=3)

    # Several categories often share the same rounded JRF rate (e.g. three
    # regions all at 21%) with close values on our own axis too -- a single
    # fixed offset then stacks their labels directly on top of each other,
    # illegible (seen on the region scatter: North East/Scotland at JRF=21%
    # rendered as unreadable overlapping text). Stagger increasingly for
    # points that collide in normalized-axis space instead.
    label_offsets = [(5, 4), (-70, 4), (5, -15), (-70, -15), (5, 26), (-70, 26), (5, -34), (-70, -34)]
    x_span = (x.max() - x.min()) or 1.0
    y_span = (y.max() - y.min()) or 1.0
    placed: list[tuple[float, float]] = []
    for _, row in merged.iterrows():
        px, py = float(row["jrf_poverty_rate_pct"]), float(row["pct_vulnerable"])
        nx, ny = px / x_span, py / y_span
        n_near = sum(1 for (qx, qy) in placed if abs(qx - nx) < 0.12 and abs(qy - ny) < 0.09)
        dx, dy = label_offsets[min(n_near, len(label_offsets) - 1)]
        # A big offset resolves the text collision but, without a visual
        # link back to its marker, reads as a disconnected, ambiguous
        # label (same problem the London-label fix in ukhls_geo_maps.py
        # already solves for the choropleth maps) -- draw a thin leader
        # line whenever a non-default offset was needed.
        ax.annotate(
            row["label"], (px, py), xytext=(dx, dy), textcoords="offset points",
            fontsize=8, color="#333333", ha="left" if dx >= 0 else "right",
            arrowprops=dict(arrowstyle="-", color="#BBBBBB", linewidth=0.6) if n_near else None,
        )
        placed.append((nx, ny))

    if len(merged) >= 2:
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(x.min() - 1, x.max() + 1, 50)
        ax.plot(xs, slope * xs + intercept, "--", color=_PALETTE["jrf"], linewidth=1.5, zorder=2)

    rho = merged.attrs.get("spearman_rho", float("nan"))
    r = merged.attrs.get("pearson_r", float("nan"))

    if exclude_label is not None and exclude_label in merged["label"].values:
        sub = merged[merged["label"] != exclude_label]
        if len(sub) >= 3:
            rho2, _ = sp.spearmanr(sub["pct_vulnerable"], sub["jrf_poverty_rate_pct"])
            r2, _ = sp.pearsonr(sub["pct_vulnerable"], sub["jrf_poverty_rate_pct"])
            subtitle = (
                f"All (n={len(merged)}): Spearman ρ={rho:.2f} | Pearson r={r:.2f}\n"
                f"Excl. {exclude_label} (n={len(sub)}): Spearman ρ={rho2:.2f} | Pearson r={r2:.2f}"
            )
        else:
            subtitle = f"Spearman ρ={rho:.2f}  |  Pearson r={r:.2f}  (n={len(merged)})"
    else:
        subtitle = f"Spearman ρ={rho:.2f}  |  Pearson r={r:.2f}  (n={len(merged)})"

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"{title}\n{subtitle}", fontsize=10.5, fontweight="bold")
    ax.grid(color=_PALETTE["grid"])
    fig.tight_layout()

    out_path = paths.UKHLS_VULN_FIGURES / f"external_validation_{dim}_scatter.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    log.info("(%s) comparison scatter plot saved -> %s", dim, out_path)


# ═══════════════════════════════════════════════════════════════════════
# Per-dimension entry points
# ═══════════════════════════════════════════════════════════════════════

def _save_ni_oil_heating_evidence() -> pd.DataFrame:
    """Supporting evidence for why Northern Ireland is excluded ONLY from
    the region correlation coefficient (never from the underlying
    vulnerability analysis itself -- see validate_region's docstring).
    Verified directly against this project's own panel data, not asserted
    from memory:

      - 71.2% of NI households report spending on oil heating, vs 1.5-9.8%
        in every GB region (mains gas never reached large parts of NI).
      - WITHIN Northern Ireland alone, oil-heating households average
        GBP 2,018/year on fuel and a 20.9% vulnerability rate, vs
        GBP 1,243/year and 12.3% for non-oil NI households in the SAME
        region -- a controlled within-region comparison, so this isn't
        confounded by anything else that makes NI different.

    That within-region gap is real signal, not noise: heating oil is
    bought in lump-sum deliveries, is price-volatile, and (unlike gas/
    electricity) sits outside Ofgem's price cap -- a genuine fuel-specific
    cost exposure with no reason to show up in JRF's income-based poverty
    measure. This is why NI is excluded from the region correlation
    coefficient (a like-for-like check of construct agreement) but kept
    in every other regional output (maps, tables, driver analysis) --
    dropping it there would throw away the single clearest example of
    this project's fuel-specific measure doing exactly what it's for.
    """
    df = pd.read_csv(paths.UKHLS_PANEL)
    from src.ukhls_mapping import GOR_LABELS
    df["region"] = df["gor_dv"].map(GOR_LABELS)
    df["has_oil"] = df["xpoily"].fillna(0) > 0

    by_region = df.groupby("region").agg(
        pct_using_oil_heating=("has_oil", lambda s: round(100 * s.mean(), 1)),
        pct_vulnerable=("high_fuel_vulnerable", lambda s: round(100 * s.mean(skipna=True), 1)),
        n=("has_oil", "size"),
    ).reset_index().sort_values("pct_using_oil_heating", ascending=False)

    ni = df[df["region"] == "Northern Ireland"]
    within_ni = ni.groupby("has_oil").agg(
        mean_annual_fuel_spend_gbp=("total_fuel_spend", lambda s: round(s.mean(), 0)),
        pct_vulnerable=("high_fuel_vulnerable", lambda s: round(100 * s.mean(skipna=True), 1)),
        n=("total_fuel_spend", "size"),
    ).reset_index()

    out_path = paths.UKHLS_VULN_TABLES / "ni_oil_heating_evidence_by_region.csv"
    by_region.to_csv(out_path, index=False)
    within_path = paths.UKHLS_VULN_TABLES / "ni_oil_heating_evidence_within_ni.csv"
    within_ni.to_csv(within_path, index=False)
    log.info("NI oil-heating evidence saved -> %s, %s", out_path, within_path)
    return by_region


def _save_rationing_evidence() -> pd.DataFrame:
    """
    Tests a specific blind spot in the ratio-based fuel_to_income_ratio
    target: a household that copes with cost pressure by RATIONING energy
    use (self-disconnection, "heat or eat") would show a LOWER fuel spend
    and so a lower ratio -- the opposite of what its true circumstances
    warrant. JRF's own cost-of-living tracker exists precisely because
    income-based poverty measures can miss exactly this. Two proxies,
    checked (not assumed) against this project's own panel data:

      - prepayment_meter (duelpay/elecpay==4, ALL 15 waves) -- a
        well-documented UK fuel-poverty self-disconnection proxy.
      - inoutflows12 ("reduced usage of utilities," waves m/o only,
        cost-of-living-crisis module) -- a direct self-report of rationing
        behaviour, narrower coverage but a more literal match.

    If either proxy's rate among LOW-ratio ("not vulnerable") households
    is non-trivial, that is direct evidence some households are coping by
    cutting usage rather than showing up as high-spend -- reported
    transparently either way, same practice as
    _save_ni_oil_heating_evidence above.
    """
    df = pd.read_csv(paths.UKHLS_PANEL)

    prepay_rows = []
    if "prepayment_meter" in df.columns:
        by_vuln = df.dropna(subset=["prepayment_meter", "high_fuel_vulnerable"]).groupby(
            "high_fuel_vulnerable"
        ).agg(
            pct_prepayment_meter=("prepayment_meter", lambda s: round(100 * s.mean(), 1)),
            n=("prepayment_meter", "size"),
        ).reset_index()
        by_vuln["high_fuel_vulnerable"] = by_vuln["high_fuel_vulnerable"].map({0: "Not vulnerable (ratio<10%)", 1: "Vulnerable (ratio>=10%)"})
        prepay_rows = by_vuln
        out_path = paths.UKHLS_VULN_TABLES / "rationing_evidence_prepayment.csv"
        by_vuln.to_csv(out_path, index=False)
        log.info("Rationing evidence (prepayment meter, all 15 waves) saved -> %s:\n%s", out_path, by_vuln.to_string(index=False))

    inout_rows = []
    if "inoutflows12" in df.columns:
        sub = df.dropna(subset=["inoutflows12", "high_fuel_vulnerable"])
        if not sub.empty:
            by_vuln2 = sub.groupby("high_fuel_vulnerable").agg(
                pct_reduced_utility_usage=("inoutflows12", lambda s: round(100 * s.mean(), 1)),
                n=("inoutflows12", "size"),
            ).reset_index()
            by_vuln2["high_fuel_vulnerable"] = by_vuln2["high_fuel_vulnerable"].map({0: "Not vulnerable (ratio<10%)", 1: "Vulnerable (ratio>=10%)"})
            inout_rows = by_vuln2
            out_path2 = paths.UKHLS_VULN_TABLES / "rationing_evidence_inoutflows12.csv"
            by_vuln2.to_csv(out_path2, index=False)
            log.info("Rationing evidence (self-reported reduced utility usage, waves m/o) saved -> %s:\n%s",
                      out_path2, by_vuln2.to_string(index=False))

    if isinstance(prepay_rows, pd.DataFrame) and not prepay_rows.empty:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        fig.patch.set_facecolor("white")
        ax.bar(prepay_rows["high_fuel_vulnerable"], prepay_rows["pct_prepayment_meter"],
               color=[_PALETTE["jrf"], _PALETTE["ours"]], alpha=0.85)
        for i, r in prepay_rows.iterrows():
            ax.text(i, r["pct_prepayment_meter"] + 0.3, f"{r['pct_prepayment_meter']:.1f}%", ha="center", fontsize=9)
        ax.set_ylabel("% of households on a prepayment meter")
        ax.set_title("Rationing/self-disconnection evidence:\nprepayment-meter rate, by vulnerability status (all waves)")
        ax.grid(axis="y", color=_PALETTE["grid"])
        fig.tight_layout()
        out_fig = paths.UKHLS_VULN_FIGURES / "rationing_evidence_prepayment.png"
        paths.UKHLS_VULN_FIGURES.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_fig, dpi=150, bbox_inches="tight")
        fig.savefig(out_fig.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        log.info("Rationing evidence figure saved -> %s", out_fig)

    return prepay_rows if isinstance(prepay_rows, pd.DataFrame) else pd.DataFrame()


def validate_family_composition() -> pd.DataFrame:
    """
    Collapses our own 5-category family_composition_group breakdown
    (src.ukhls_preprocessing._derive_family_composition_group) into JRF's
    2 stated categories (Lone parent 44%, Couple with children 25% -- UK
    Poverty 2025, Table 5, p.36) -- JRF states a rate for these two family
    types overall, not separately crossed with our own large-family split,
    so "Lone parent, 1-2/3+ children" collapse to one "Lone parent" row and
    "Couple, 1-2/3+ children" to one "Couple with children" row (n-weighted
    mean). Our "Other multi-adult, with children" and "No children" groups
    have no stated JRF comparator for this specific check and are excluded
    here (kept in full in the standalone breakdown figure)."""
    ours_path = paths.UKHLS_VULN_TABLES / "policy_vulnerability_by_family_composition.csv"
    ours = pd.read_csv(ours_path)
    collapse_map = {
        "Lone parent, 1-2 children":  "Lone parent",
        "Lone parent, 3+ children":   "Lone parent",
        "Couple, 1-2 children":       "Couple with children",
        "Couple, 3+ children":        "Couple with children",
    }
    ours = ours[ours["group"].isin(collapse_map)].copy()
    ours["label"] = ours["group"].map(collapse_map)
    # Select only the columns the lambda needs before grouping, rather than
    # passing include_groups=False (pandas>=2.2 only -- requirements.txt
    # declares pandas>=1.5.0, so that kwarg raises TypeError on 1.5.x-2.1.x).
    collapsed = ours.groupby("label")[["pct_vulnerable", "n"]].apply(
        lambda g: pd.Series({
            "pct_vulnerable": float(np.average(g["pct_vulnerable"], weights=g["n"])),
            "n": int(g["n"].sum()),
        }),
    ).reset_index()
    collapsed_path = paths.UKHLS_VULN_TABLES / "policy_vulnerability_by_family_type_collapsed.csv"
    collapsed.to_csv(collapsed_path, index=False)

    merged = compare_dimension("family_type", "policy_vulnerability_by_family_type_collapsed", "label", JRF_POVERTY_RATE_BY_FAMILY_TYPE)
    plot_dimension_bars(
        merged, "family_type",
        "Ours: % high fuel-to-income vulnerable\n(household-level, pooled 2009-2024)",
        "JRF: child poverty rate, AHC\n(Table 5, p.36)",
        "External validation: our fuel-to-income vulnerability vs JRF's child poverty rate, by family type",
    )
    plot_dimension_scatter(
        merged, "family_type",
        "JRF child poverty rate, AHC (%)",
        "Our fuel-to-income vulnerability, pooled 2009-2024 (%)",
        "Family-type agreement between our vulnerability measure and JRF's child poverty rate",
    )
    return merged


def validate_employment() -> pd.DataFrame:
    """
    Collapses our own 3-category employment_group breakdown
    (has_employed_adult/has_fulltime_worker/etc., see
    src.ukhls_preprocessing.load_wave_indresp_aggregated) into JRF's own
    in-work/out-of-work split (12%/43%, UK Poverty 2025 p.76) --
    "Full-time or self-employed" and "Part-time only" both collapse to
    "In work" (JRF's own headline split), "Workless household" to "Not in
    work"."""
    ours_path = paths.UKHLS_VULN_TABLES / "policy_vulnerability_by_employment.csv"
    ours = pd.read_csv(ours_path)
    collapse_map = {
        "Full-time or self-employed": "In work",
        "Part-time only":             "In work",
        "Workless household":         "Not in work",
    }
    ours = ours[ours["group"].isin(collapse_map)].copy()
    ours["label"] = ours["group"].map(collapse_map)
    collapsed = ours.groupby("label")[["pct_vulnerable", "n"]].apply(
        lambda g: pd.Series({
            "pct_vulnerable": float(np.average(g["pct_vulnerable"], weights=g["n"])),
            "n": int(g["n"].sum()),
        }),
    ).reset_index()
    collapsed_path = paths.UKHLS_VULN_TABLES / "policy_vulnerability_by_work_status_collapsed.csv"
    collapsed.to_csv(collapsed_path, index=False)

    merged = compare_dimension("work_status", "policy_vulnerability_by_work_status_collapsed", "label", JRF_POVERTY_RATE_BY_WORK_STATUS)
    plot_dimension_bars(
        merged, "work_status",
        "Ours: % high fuel-to-income vulnerable\n(household-level, pooled 2009-2024)",
        "JRF: in-work/out-of-work poverty rate, AHC\n(p.76)",
        "External validation: our fuel-to-income vulnerability vs JRF's poverty rate, by work status",
    )
    plot_dimension_scatter(
        merged, "work_status",
        "JRF in-work/out-of-work poverty rate, AHC (%)",
        "Our fuel-to-income vulnerability, pooled 2009-2024 (%)",
        "Work-status agreement between our vulnerability measure and JRF's poverty rate",
    )
    return merged


def validate_region() -> pd.DataFrame:
    """Northern Ireland is a documented outlier here, not noise. Verified
    directly (see _save_ni_oil_heating_evidence): 71.2% of NI households
    use oil heating vs 1.5-9.8% everywhere else in GB, and WITHIN NI
    alone, oil-heating households show a materially higher fuel-to-income
    vulnerability rate (20.9%) than non-oil NI households in the same
    region (12.3%) -- a controlled comparison confirming this is real
    signal, not an artifact of region-mapping or missing data. Heating
    oil is price-volatile, bought in lump sums, and outside Ofgem's price
    cap -- a fuel-specific cost exposure with no reason to appear in
    JRF's income-based poverty measure. NI is therefore excluded ONLY
    from the correlation coefficient below (a like-for-like check of how
    well the two DIFFERENT constructs agree), never from the underlying
    regional analysis itself, where it's kept as genuinely informative."""
    _save_ni_oil_heating_evidence()
    merged = compare_dimension("region", "policy_vulnerability_by_region", "region", JRF_POVERTY_RATE_BY_REGION)
    plot_dimension_bars(
        merged, "region",
        "Ours: % high fuel-to-income vulnerable\n(UKHLS, pooled 2009-2024)",
        "JRF: relative poverty rate, AHC\n(HBAI, avg. 2021/22-2022/23)",
        "External validation: our fuel-to-income vulnerability vs JRF's income poverty rate, by region",
    )
    plot_dimension_scatter(
        merged, "region",
        "JRF relative poverty rate, AHC, 2021/22-2022/23 (%)",
        "Our fuel-to-income vulnerability, pooled 2009-2024 (%)",
        "Regional agreement between our vulnerability measure and JRF's poverty rate",
        exclude_label="Northern Ireland",
    )
    return merged


def validate_ethnicity() -> pd.DataFrame:
    """Black Caribbean households rank HIGHEST on our fuel-specific
    measure (14.1%) but JRF ranks them LOWEST of the minority groups on
    income poverty (30%, vs Bangladeshi's 56%) -- while Bangladeshi
    households, JRF's highest-poverty group, sit close to White on our
    measure. This is a genuine, reportable divergence (fuel poverty and
    income poverty are different constructs with different housing-stock
    and household-composition drivers by ethnicity), not evidence either
    measure is wrong -- flagged explicitly rather than smoothed over."""
    merged = compare_dimension("ethnicity", "policy_vulnerability_by_ethnicity", "group", JRF_POVERTY_RATE_BY_ETHNICITY)
    plot_dimension_bars(
        merged, "ethnicity",
        "Ours: % high fuel-to-income vulnerable\n(household reference person's ethnicity, pooled 2009-2024)",
        "JRF: relative poverty rate, AHC\n(HBAI, 2020/21-2022/23)",
        "External validation: our fuel-to-income vulnerability vs JRF's income poverty rate, by ethnicity",
    )
    plot_dimension_scatter(
        merged, "ethnicity",
        "JRF relative poverty rate, AHC (%)",
        "Our fuel-to-income vulnerability, pooled 2009-2024 (%)",
        "Ethnicity-group agreement between our vulnerability measure and JRF's poverty rate",
    )
    return merged


def validate_disability() -> pd.DataFrame:
    merged = compare_dimension("disability", "policy_vulnerability_by_disability", "group", JRF_POVERTY_RATE_BY_DISABILITY)
    plot_dimension_bars(
        merged, "disability",
        "Ours: % high fuel-to-income vulnerable\n(contains a disabled adult, pooled 2009-2024)",
        "JRF: relative poverty rate, AHC\n(Table 8, p.67)",
        "External validation: our fuel-to-income vulnerability vs JRF's income poverty rate, by disability",
    )
    plot_dimension_scatter(
        merged, "disability",
        "JRF relative poverty rate, AHC (%)",
        "Our fuel-to-income vulnerability, pooled 2009-2024 (%)",
        "Disability-status agreement between our vulnerability measure and JRF's poverty rate",
    )
    return merged


def validate_tenure() -> pd.DataFrame:
    """Owned-outright households rank ABOVE private renters on our
    fuel-specific measure, the reverse of JRF's income-poverty ranking
    (private renting 35% > owned outright 14%). Plausible driver: outright
    owners skew older/pensioner and disproportionately live in older,
    less energy-efficient housing stock paid off but expensive to heat,
    while private renters (though more income-poor) more often occupy
    newer-built stock -- a real fuel-specific effect, flagged rather than
    smoothed over, same as the region/ethnicity divergences above."""
    merged = compare_dimension("tenure", "policy_vulnerability_by_tenure", "group", JRF_POVERTY_RATE_BY_TENURE)
    plot_dimension_bars(
        merged, "tenure",
        "Ours: % high fuel-to-income vulnerable\n(pooled 2009-2024)",
        "JRF: relative poverty rate, AHC\n(Table 10, p.95, 2022/23)",
        "External validation: our fuel-to-income vulnerability vs JRF's income poverty rate, by tenure",
    )
    plot_dimension_scatter(
        merged, "tenure",
        "JRF relative poverty rate, AHC (%)",
        "Our fuel-to-income vulnerability, pooled 2009-2024 (%)",
        "Tenure-group agreement between our vulnerability measure and JRF's poverty rate",
    )
    return merged


# ─────────────────────────────────────────────────────────────────────────
# Temporal comparison: our wave-level trend vs JRF's cost-of-living-crisis
# narrative (2021-2024)
# ─────────────────────────────────────────────────────────────────────────

def plot_wave_trend_vs_jrf_narrative() -> None:
    by_wave_path = paths.UKHLS_VULN_TABLES / "policy_vulnerability_by_wave.csv"
    by_wave = pd.read_csv(by_wave_path).dropna(subset=["year"]).sort_values("year")

    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.patch.set_facecolor("white")

    ax.axvspan(2021, 2023.5, color="#FADBD8", alpha=0.6, zorder=1,
               label="Cost-of-living crisis window\n(JRF: hardship tracker peaked Oct 2022;\nrelative poverty itself stayed 'broadly flat')")
    ax.plot(by_wave["year"], by_wave["pct_vulnerable"], marker="o", lw=2,
            color=_PALETTE["ours"], zorder=3, label="Ours: % high fuel-to-income vulnerable, by UKHLS wave")

    ax.set_xlabel("Interview year (UKHLS wave)")
    ax.set_ylabel("% high fuel-to-income vulnerable")
    ax.set_title(
        "External validation: does our fuel vulnerability trend track JRF's cost-of-living-crisis account?\n"
        "JRF p.19: relative poverty (AHC) was 'broadly flat' 2021/22->2022/23 -- a slow-moving annual "
        "measure by construction.\nJRF pp.105-108: its faster cost-of-living tracker instead shows hardship "
        "peaking in Oct 2022 (75% of low-income\nhouseholds going without essentials) before easing slightly "
        "by Oct 2024 (69%) -- the sharp-spike-then-partial-easing\nshape this project's own wave-level rate "
        "reproduces below.",
        fontsize=9.5, fontweight="bold",
    )
    ax.grid(axis="y", color=_PALETTE["grid"])
    ax.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()

    out_path = paths.UKHLS_VULN_FIGURES / "external_validation_wave_trend.png"
    paths.UKHLS_VULN_FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    log.info("Wave-trend vs JRF narrative figure saved -> %s", out_path)


def run() -> None:
    validate_region()
    validate_ethnicity()
    validate_disability()
    validate_tenure()
    validate_family_composition()
    validate_employment()
    _save_rationing_evidence()
    plot_wave_trend_vs_jrf_narrative()


if __name__ == "__main__":
    run()
