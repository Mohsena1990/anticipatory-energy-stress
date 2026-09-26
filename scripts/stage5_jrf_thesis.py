"""
Stage 5 thesis outputs: JRF comparison and Northern Ireland
(analysis_plan_rerun.md Stage 5, amendment A1 and logged deviations).

Builds on scripts/jrf_comparison_v2.py (same windows, same estimator):
  * PSU-bootstrap 95% CIs and within-dimension ranks for every JRF category
    (primary window; region and ethnicity also in the sensitivity window)
  * NI oil vs non-oil rates with CIs (region windows)
  * thesis tables T5.1-T5.4, with secondary suppression checks
  * figures F5.1-F5.3 and a results draft (markdown), numbers filled from the
    computed tables

Estimator: per-wave weighted rate (hh_xw) among in-window households,
averaged over waves by in-window n. Bootstrap: 2,000 replicates resampling
PSUs UK-wide.

Secondary suppression (thesis tables only): categories are published only
with unweighted n >= 100 (primary rule). Checked additionally:
  (i) complement risk: a dimension with exactly one masked category whose
      count could be recovered from a published total -> mask a second one;
 (ii) window differencing: sensitivity minus primary window n gives the
      2020/21 subgroup; if 0 < diff < 100 the sensitivity cell's rate and n
      are suppressed (diff < 10 would also reveal a small count).
All actions are logged in thesis_secondary_suppression_log.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import jrf_comparison_v2 as J  # noqa: E402
from src.paths import OUTPUTS_DIR  # noqa: E402

OUT = OUTPUTS_DIR / "stage5"
FIG = OUT / "figures"
REPORT = OUTPUTS_DIR / "reports"
N_BOOT = 2000
SEED = 20260926
MIN_N = 100
NAVY, ACCENT, MUTED, GRID = "#2E5077", "#D55E00", "#5F6B7A", "#E6E9ED"


def boot_rates(df: pd.DataFrame, dim: str, cats: list[str], window: tuple[str, str],
               flag: str = "high_fuel_vulnerable") -> pd.DataFrame:
    lo, hi = window
    win = df[(df.ym >= lo) & (df.ym <= hi) & df[dim].isin(cats) & df[flag].notna() & (df.hh_xw > 0)]
    win = win.merge(PSU, on=["hidp", "wave"], how="left", validate="1:1")
    waves = sorted(win.wave.unique())
    cell = win.assign(wy=win.hh_xw * win[flag], one=1).groupby(["psu", dim, "wave"])[["wy", "hh_xw", "one"]].sum()
    psus = cell.index.get_level_values("psu").unique()
    cols = pd.MultiIndex.from_product([cats, waves])

    def mat(c):
        return cell[c].unstack([dim, "wave"]).reindex(index=psus, columns=cols, fill_value=0).fillna(0).values
    WY, W, N = mat("wy"), mat("hh_xw"), mat("one")
    k, nw = len(cats), len(waves)

    def est(m):
        wy, w, n = m @ WY, m @ W, m @ N
        rate = np.divide(wy, w, out=np.full_like(wy, np.nan), where=w > 0).reshape(k, nw)
        n = n.reshape(k, nw)
        return 100 * np.nansum(rate * n, axis=1) / n.sum(axis=1)

    point = est(np.ones(len(psus)))
    rng = np.random.default_rng(SEED)
    boots = np.array([est(rng.multinomial(len(psus), np.full(len(psus), 1 / len(psus)))) for _ in range(N_BOOT)])
    ranks = (-boots).argsort(axis=1).argsort(axis=1) + 1
    prank = (-point).argsort().argsort() + 1
    nobs = win.groupby(dim).size().reindex(cats).fillna(0).astype(int)
    return pd.DataFrame(dict(category=cats, n=nobs.values, pct=point,
                             ci_low=np.nanpercentile(boots, 2.5, axis=0),
                             ci_high=np.nanpercentile(boots, 97.5, axis=0),
                             rank=prank, rank_ci_low=np.percentile(ranks, 2.5, axis=0).astype(int),
                             rank_ci_high=np.percentile(ranks, 97.5, axis=0).astype(int),
                             p_rank1=(ranks == 1).mean(axis=0)))


def comparison(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dim, spec in J.JRF.items():
        wins = [("primary", spec["window"])] + ([("sensitivity", spec["sensitivity_window"])]
                                                 if "sensitivity_window" in spec else [])
        cats = list(spec["values"])
        for wlabel, w in wins:
            b = boot_rates(df, dim, cats, w)
            b["dimension"], b["window"], b["ukhls_window"] = dim, wlabel, f"{w[0]} to {w[1]}"
            b["jrf_pct"] = b.category.map(spec["values"])
            b["jrf_rank"] = b.jrf_pct.rank(ascending=False, method="min").astype(int)
            rows.append(b)
    out = pd.concat(rows, ignore_index=True)
    return out[["dimension", "window", "ukhls_window", "category", "jrf_pct", "jrf_rank", "n", "pct",
                "ci_low", "ci_high", "rank", "rank_ci_low", "rank_ci_high", "p_rank1"]]


def ni_oil_rates(df: pd.DataFrame) -> pd.DataFrame:
    ni = df[df.region == "Northern Ireland"].copy()
    ni["oil_group"] = np.where(ni.fuelhave3 == 1, "Oil heating", np.where(ni.fuelhave3 == 0, "No oil", None))
    rows = []
    for wlabel, w in [("primary", J.FY_2021_23), ("sensitivity", J.FY_2020_23)]:
        b = boot_rates(ni, "oil_group", ["Oil heating", "No oil"], w)
        b["window"], b["ukhls_window"] = wlabel, f"{w[0]} to {w[1]}"
        rows.append(b)
    return pd.concat(rows, ignore_index=True)


def secondary_suppression(comp: pd.DataFrame, log: list[dict]) -> pd.DataFrame:
    comp = comp.copy()
    comp["suppressed"] = comp.n < MIN_N
    for r in comp[comp.suppressed].itertuples():
        log.append(dict(table="T5.2", dimension=r.dimension, window=r.window, category=r.category,
                        rule="primary: n < 100", action="rate, CI and rank suppressed"))
    # (i) complement: no dimension totals are published in T5.2, so a single
    # masked cell cannot be recovered by subtraction; still flag the pattern.
    for (dim, w), g in comp.groupby(["dimension", "window"]):
        if g.suppressed.sum() == 1:
            log.append(dict(table="T5.2", dimension=dim, window=w, category="(dimension)",
                            rule="complement check", action="one masked cell; no totals published -> no action"))
    # (ii) window differencing (region, ethnicity).
    for dim in comp[comp.window == "sensitivity"].dimension.unique():
        p = comp[(comp.dimension == dim) & (comp.window == "primary")].set_index("category")
        s = comp[(comp.dimension == dim) & (comp.window == "sensitivity")].set_index("category")
        for cat in s.index:
            diff = int(s.loc[cat, "n"] - p.loc[cat, "n"])
            if 0 < diff < MIN_N:
                idx = comp[(comp.dimension == dim) & (comp.window == "sensitivity") & (comp.category == cat)].index
                comp.loc[idx, "suppressed"] = True
                log.append(dict(table="T5.2", dimension=dim, window="sensitivity", category=cat,
                                rule=f"window differencing: 2020/21 subgroup n = {'<10' if diff < 10 else diff} (< 100)",
                                action="sensitivity-window rate, CI, rank and n suppressed"))
    cols = ["pct", "ci_low", "ci_high", "rank", "rank_ci_low", "rank_ci_high", "p_rank1"]
    comp[cols] = comp[cols].astype(object)
    comp.loc[comp.suppressed, cols] = "suppressed"
    comp["n"] = comp["n"].astype(object)
    comp.loc[comp.suppressed & (comp.window == "sensitivity"), "n"] = "suppressed"
    return comp


def fig_region(comp: pd.DataFrame, rho_excl: float, rho_all: float) -> None:
    r = comp[(comp.dimension == "region") & (comp.window == "primary")].copy()
    r[["pct", "ci_low", "ci_high"]] = r[["pct", "ci_low", "ci_high"]].astype(float)
    # Dodge regions that share a JRF value so their intervals do not overlap.
    r = r.sort_values(["jrf_pct", "pct"])
    r["x"] = r.jrf_pct + r.groupby("jrf_pct").cumcount().sub(
        (r.groupby("jrf_pct").jrf_pct.transform("size") - 1) / 2).mul(0.22)
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for _, x in r.iterrows():
        ni = x.category == "Northern Ireland"
        col = ACCENT if ni else NAVY
        ax.errorbar(x.x, x.pct, yerr=[[x.pct - x.ci_low], [x.ci_high - x.pct]], fmt="o", ms=6 if ni else 5,
                    color=col, ecolor=col, elinewidth=1, capsize=0, mec="white", mew=1, zorder=3)
    # Horizontal offsets to separate labels of regions sharing a JRF value.
    off = {"South East": (-6, 0, "right"), "East of England": (-6, 0, "right"),
           "North East": (6, -10, "left"), "Scotland": (-5, 9, "right"),
           "Northern Ireland": (7, 0, "left")}
    for _, x in r.iterrows():
        dx, dy, ha = off.get(x.category, (6, 0, "left"))
        ax.annotate(x.category, (x.x, x.pct), xytext=(dx, dy), textcoords="offset points",
                    fontsize=7.5, ha=ha, va="center",
                    color="#1F2933" if x.category == "Northern Ireland" else MUTED,
                    fontweight="bold" if x.category == "Northern Ireland" else "normal")
    ax.set_xlabel("JRF relative poverty rate, AHC (%), 2021/22–2022/23\n(regions with equal JRF rates offset slightly)")
    ax.set_ylabel("Fuel spend ≥ 10% of net income (%), weighted\ninterviews Apr 2021–Mar 2023, 95% CI")
    ax.set_xlim(15.5, 29)
    ax.set_ylim(0, max(r.ci_high) * 1.12)
    ax.grid(color=GRID, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.01, 0.02, f"Spearman ρ: all 12 regions {rho_all:.2f}; excluding NI {rho_excl:.2f}",
            transform=ax.transAxes, fontsize=7.5, color=MUTED)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"F5_1_region_vs_jrf.{ext}", dpi=220)
    plt.close(fig)


def fig_dimensions(comp: pd.DataFrame) -> None:
    dims = [("tenure", "Tenure (2022/23)"), ("family_type", "Family type (2022/23; JRF: child poverty)"),
            ("work_status", "Work status (2022/23; working-age)"), ("disability", "Disability (2022/23)"),
            ("ethnicity", "Ethnicity of HRP (2021/22–2022/23)")]
    fig, axes = plt.subplots(len(dims), 1, figsize=(7.2, 8.6),
                             gridspec_kw={"height_ratios": [4, 2, 2, 2, 6]}, sharex=True)
    for ax, (dim, title) in zip(axes, dims):
        d = comp[(comp.dimension == dim) & (comp.window == "primary")].copy()
        d = d[d.pct != "suppressed"]
        d[["pct", "ci_low", "ci_high"]] = d[["pct", "ci_low", "ci_high"]].astype(float)
        d = d.sort_values("jrf_pct")
        y = np.arange(len(d))
        ax.errorbar(d.pct, y, xerr=[d.pct - d.ci_low, d.ci_high - d.pct], fmt="o", ms=5, color=NAVY,
                    elinewidth=1, capsize=0, mec="white", mew=1)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{c}  (JRF {j:.0f}%)" for c, j in zip(d.category, d.jrf_pct)], fontsize=7.5)
        ax.set_title(title, fontsize=8.5, loc="left", color="#1F2933")
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(-0.6, len(d) - 0.4)
    axes[-1].set_xlabel("Fuel spend ≥ 10% of net income (%), weighted, 95% CI\n"
                        "Categories ordered by JRF poverty rate (highest at top)")
    axes[-1].set_xlim(0, None)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"F5_2_dimensions_vs_jrf.{ext}", dpi=220)
    plt.close(fig)


def fig_ni(oil: pd.DataFrame, ame: pd.DataFrame) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.4), gridspec_kw={"width_ratios": [1, 1.4]})
    o = oil[oil.window == "primary"]
    y = np.arange(len(o))
    a1.errorbar(o.pct, y, xerr=[o.pct - o.ci_low, o.ci_high - o.pct], fmt="o", ms=6, color=ACCENT,
                elinewidth=1, capsize=0, mec="white", mew=1)
    a1.set_yticks(y)
    a1.set_yticklabels([f"{c}\n(n = {n:,})" for c, n in zip(o.category, o.n)], fontsize=8)
    a1.set_xlim(0, float(o.ci_high.max()) * 1.2)
    a1.set_xlabel("Fuel spend ≥ 10% of income (%)\nNI, Apr 2021–Mar 2023, weighted", fontsize=8)
    a1.set_ylim(-0.6, len(o) - 0.4)
    a1.set_title("A. Northern Ireland by heating fuel", fontsize=9, loc="left")
    order = ["ni_a_regionFE_filled", "ni_b_oil_filled", "ni_b_oil_rural_filled",
             "ni_c_ni_x_oil_filled", "ni_c_ni_x_oil_rural_filled"]
    names = ["Region FE", "+ oil", "+ oil + rural", "+ NI × oil", "+ NI × oil + rural"]
    m = ame[(ame.spec == "primary") & (ame.contrast == "NI vs South East")].set_index("model").loc[order]
    yy = np.arange(len(m))[::-1]
    a2.errorbar(m.ame_pp, yy, xerr=[m.ame_pp - m.ci_low_pp, m.ci_high_pp - m.ame_pp], fmt="o", ms=5,
                color=NAVY, elinewidth=1, capsize=0, mec="white", mew=1)
    a2.axvline(0, color="#9AA5B1", lw=0.8)
    a2.set_yticks(yy)
    a2.set_yticklabels(names, fontsize=8)
    a2.set_xlabel("NI vs South East, average marginal effect (pp), 95% CI\n"
                  "driver model (Stage 3 primary), interviews 2010–2025", fontsize=8)
    a2.set_title("B. NI gap after accounting for oil and rurality", fontsize=9, loc="left")
    for ax in (a1, a2):
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"F5_3_ni_oil.{ext}", dpi=220)
    plt.close(fig)


def fmt_ci(p, lo, hi, d=1):
    return f"{p:.{d}f}% (95% CI {lo:.{d}f}–{hi:.{d}f})"


def main() -> None:
    global PSU
    for d in (OUT, FIG, REPORT):
        d.mkdir(parents=True, exist_ok=True)
    df = J.load()
    PSU = J.load_psu()
    comp = comparison(df)
    oil = ni_oil_rates(df)
    meta = pd.read_csv(OUTPUTS_DIR / "jrf" / "jrf_metadata.csv")
    agree = pd.read_csv(OUTPUTS_DIR / "jrf" / "jrf_agreement.csv")
    nioil = pd.read_csv(OUTPUTS_DIR / "jrf" / "ni_oil.csv")
    ame = pd.read_csv(OUTPUTS_DIR / "stage3" / "ni_oil_ame.csv")
    regci = pd.read_csv(OUTPUTS_DIR / "jrf" / "region_window_ci.csv")

    log: list[dict] = []
    t52 = secondary_suppression(comp, log)
    if not log:
        log.append(dict(table="T5.2/T5.4", dimension="all", window="all", category="all",
                        rule="primary n<100; complement; window differencing",
                        action="no cell triggered suppression"))
    pd.DataFrame(log).to_csv(OUT / "thesis_secondary_suppression_log.csv", index=False)

    meta.to_csv(OUT / "thesis_T5_1_jrf_metadata.csv", index=False)
    t52.to_csv(OUT / "thesis_T5_2_comparison.csv", index=False)
    agree.to_csv(OUT / "thesis_T5_3_agreement.csv", index=False)

    ni_rows = []
    for _, r in regci[regci.region == "Northern Ireland"].iterrows():
        ni_rows.append(dict(item=f"NI rate, {r.window} window ({r.ukhls_window}), outcome {r.outcome}",
                            value=r.pct_weighted, ci_low=r.ci95_low, ci_high=r.ci95_high,
                            detail=f"rank {int(r['rank'])} of 12 (95% CI {int(r.rank_ci95_low)}–{int(r.rank_ci95_high)}); "
                                   f"P(rank 1) = {r.p_rank1:.2f}; n = {int(r.n):,}"))
    w = nioil[nioil["sample"].str.startswith("primary")].iloc[0]
    ni_rows.append(dict(item="NI oil share, primary window (weighted)", value=w.oil_share_pct_weighted,
                        ci_low=np.nan, ci_high=np.nan, detail=f"unweighted {w.oil_share_pct_unweighted:.1f}%"))
    for _, o in oil.iterrows():
        ni_rows.append(dict(item=f"NI {o.category}, {o.window} window", value=o.pct, ci_low=o.ci_low,
                            ci_high=o.ci_high, detail=f"n = {int(o.n):,}"))
    for _, a in ame[(ame.spec == "primary") & ame.model.str.endswith("_filled")].iterrows():
        ni_rows.append(dict(item=f"AME {a.contrast} (pp), model {a.model}", value=a.ame_pp,
                            ci_low=a.ci_low_pp, ci_high=a.ci_high_pp, detail=f"n = {int(a.n):,}"))
    t54 = pd.DataFrame(ni_rows)
    t54.to_csv(OUT / "thesis_T5_4_northern_ireland.csv", index=False)

    ag = agree.set_index(["dimension", "window", "subset"])
    rho_all = ag.loc[("region", "primary", "all"), "spearman_rho"]
    rho_ex = ag.loc[("region", "primary", "excl. Northern Ireland"), "spearman_rho"]
    fig_region(t52, rho_ex, rho_all)
    fig_dimensions(t52)
    fig_ni(oil, ame)
    write_report(t52, oil, ame, regci, nioil, ag)
    print(t52.to_string(index=False))
    print(t54.round(2).to_string(index=False))
    print(pd.DataFrame(log).to_string(index=False))


def write_report(t52, oil, ame, regci, nioil, ag) -> None:
    def g(dim, cat, win="primary"):
        r = t52[(t52.dimension == dim) & (t52.window == win) & (t52.category == cat)].iloc[0]
        return r
    ni = regci[(regci.region == "Northern Ireland") & (regci.outcome == "primary")].set_index("window")
    nis1 = regci[(regci.region == "Northern Ireland") & (regci.outcome == "s1_lower_bound")].set_index("window")
    second = regci[(regci.window == "primary") & (regci.outcome == "primary") & (regci["rank"] == 2)].iloc[0]
    lowest = regci[(regci.window == "primary") & (regci.outcome == "primary") & (regci["rank"] == 12)].iloc[0]
    op = oil[oil.window == "primary"].set_index("category")
    am = ame[(ame.spec == "primary") & ame.model.str.endswith("_filled")].set_index(["model", "contrast"])
    A = lambda m, c="NI vs South East": am.loc[(m, c)]  # noqa: E731
    w = nioil[nioil["sample"].str.startswith("primary")].iloc[0]
    osr = pd.read_csv(OUTPUTS_DIR / "jrf" / "oil_share_by_region.csv")
    osr = osr[osr.region != "Northern Ireland"].oil_share_pct_unweighted
    rho = lambda d, s="all", win="primary": ag.loc[(d, win, s), "spearman_rho"]  # noqa: E731
    lines = [
        "# Stage 5 draft: fuel vulnerability, JRF income poverty and Northern Ireland",
        "",
        "*Generated by `scripts/stage5_jrf_thesis.py`; every number comes from `outputs_v2/`. "
        "Draft for the author's rewrite; not thesis text yet.*",
        "",
        "## Method in brief",
        "",
        "Fuel vulnerability (annual fuel spend ≥ 10% of net household income; routing-corrected, "
        "complete-case) is compared with JRF *UK Poverty 2025* income-poverty rates (relative poverty, "
        "after housing costs). UKHLS rates are time-matched to each JRF period using actual household "
        "interview dates and are weighted with the wave's household cross-sectional weight. "
        "Region and ethnicity use interviews from April 2021 to March 2023, matching JRF's two-year "
        "average (DWP excludes 2020/21); tenure, family type, work status and disability use April 2022 "
        "to March 2023. 95% confidence intervals and ranks come from 2,000 bootstrap replicates "
        "resampling primary sampling units. Categories with fewer than 100 households are suppressed. "
        "Family type (JRF: child poverty) and work status (JRF: working-age adults) compare different "
        "units and are read as directional only.",
        "",
        "## Agreement with JRF",
        "",
        f"- **Tenure** agrees most closely (Spearman ρ = {rho('tenure'):.2f}, four categories): "
        f"social renters {g('tenure','Social renting').pct:.1f}% and mortgage holders "
        f"{g('tenure','Buying with mortgage').pct:.1f}%. The exception is **outright owners**: third of four on "
        f"JRF income poverty (14%) but second on fuel vulnerability, at "
        f"{fmt_ci(g('tenure','Owned outright').pct, g('tenure','Owned outright').ci_low, g('tenure','Owned outright').ci_high)}, "
        f"above private renters ({g('tenure','Private renting').pct:.1f}%, JRF 35%).",
        f"- **Work status, family type and disability** rank in the same order as JRF: "
        f"workless {g('work_status','Not in work').pct:.1f}% vs in work {g('work_status','In work').pct:.1f}%; "
        f"lone parent {g('family_type','Lone parent').pct:.1f}% vs couple with children "
        f"{g('family_type','Couple with children').pct:.1f}%; households with a disabled adult "
        f"{g('disability','Contains disabled adult').pct:.1f}% vs none {g('disability','No disabled adult').pct:.1f}%.",
        f"- **Region** shows no agreement (ρ = {rho('region'):.2f} for all 12 regions; "
        f"{rho('region','excl. Northern Ireland'):.2f} excluding NI). The earlier finding that other "
        "regions track income poverty once NI is removed does not hold on the corrected outcome with "
        "time-matched, weighted rates.",
        f"- **Ethnicity**: the comparison is inconclusive (ρ = {rho('ethnicity'):.2f}; confidence intervals are "
        "wide for every minority group). Black Caribbean households have the highest point estimate, "
        f"{fmt_ci(g('ethnicity','Black Caribbean').pct, g('ethnicity','Black Caribbean').ci_low, g('ethnicity','Black Caribbean').ci_high)}, "
        f"while Bangladeshi households, highest on JRF income poverty (56%), are at "
        f"{g('ethnicity','Bangladeshi').pct:.1f}% (n = {int(g('ethnicity','Bangladeshi').n)}). "
        "The intervals overlap too widely to establish either agreement or divergence; not a basis for group "
        "targeting.",
        "",
        "## Northern Ireland",
        "",
        f"- NI has the highest rate of the 12 regions: "
        f"{fmt_ci(ni.loc['primary','pct_weighted'], ni.loc['primary','ci95_low'], ni.loc['primary','ci95_high'])}, "
        f"rank 1 (95% CI {int(ni.loc['primary','rank_ci95_low'])}–{int(ni.loc['primary','rank_ci95_high'])}), "
        f"ranked first in {100*ni.loc['primary','p_rank1']:.0f}% of bootstrap replicates. "
        f"The next region, {second.region}, is at {second.pct_weighted:.1f}% "
        f"({second.ci95_low:.1f}–{second.ci95_high:.1f}); the lowest, {lowest.region}, at {lowest.pct_weighted:.1f}%.",
        f"- This holds under the lower-bound outcome ({nis1.loc['primary','pct_weighted']:.1f}%, P(rank 1) = "
        f"{nis1.loc['primary','p_rank1']:.2f}) and in the April 2020–March 2023 window "
        f"({ni.loc['sensitivity','pct_weighted']:.1f}%, P(rank 1) = {ni.loc['sensitivity','p_rank1']:.2f}).",
        "- NI's high fuel vulnerability does not reflect higher income poverty: NI has the lowest JRF "
        "income-poverty rate of the UK nations (17%).",
        f"- Heating oil is the main fuel-system difference: {w.oil_share_pct_weighted:.0f}% of NI households "
        f"use oil (weighted, primary window), against {osr.min():.1f}–{osr.max():.1f}% in other regions (pooled, unweighted). "
        f"Within NI, oil-heated households are at "
        f"{fmt_ci(op.loc['Oil heating','pct'], op.loc['Oil heating','ci_low'], op.loc['Oil heating','ci_high'])} "
        f"against {fmt_ci(op.loc['No oil','pct'], op.loc['No oil','ci_low'], op.loc['No oil','ci_high'])} "
        "for other NI households.",
        f"- In the driver model (interviews 2010–2025, n = {int(A('ni_a_regionFE_filled').n):,}), the NI gap "
        f"relative to the South East is {A('ni_a_regionFE_filled').ame_pp:.1f} percentage points "
        f"({A('ni_a_regionFE_filled').ci_low_pp:.1f}–{A('ni_a_regionFE_filled').ci_high_pp:.1f}) with region "
        f"fixed effects only. Adding oil use reduces it to {A('ni_b_oil_filled').ame_pp:.1f} pp "
        f"({A('ni_b_oil_filled').ci_low_pp:.1f}–{A('ni_b_oil_filled').ci_high_pp:.1f}); oil itself is associated "
        f"with {A('ni_b_oil_filled','oil vs no oil').ame_pp:.1f} pp higher probability. Controlling for rural "
        f"location leaves the NI gap at {A('ni_b_oil_rural_filled').ame_pp:.1f} pp and the oil effect at "
        f"{A('ni_b_oil_rural_filled','oil vs no oil').ame_pp:.1f} pp. The oil penalty is no larger in NI than "
        f"elsewhere (NI × oil model: NI gap {A('ni_c_ni_x_oil_filled').ame_pp:.1f} pp).",
        f"- Reading: around two-thirds of NI's excess risk ({A('ni_a_regionFE_filled').ame_pp:.1f} → "
        f"{A('ni_b_oil_filled').ame_pp:.1f} pp) is accounted for by heating-oil use, a fuel-system exposure that "
        f"income-poverty measures do not capture; a gap of about {A('ni_b_oil_filled').ame_pp:.1f}–"
        f"{A('ni_b_oil_rural_filled').ame_pp:.1f} pp remains after controlling for oil and rurality.",
        "",
        "## Boundaries",
        "",
        "- Associations, not causal effects; oil use is not randomly assigned.",
        "- Oil shares outside NI are pooled unweighted figures; the NI share is weighted and window-matched.",
        "- Family type and work status compare different units from JRF (children; working-age adults).",
        "- Tables: `outputs_v2/stage5/thesis_T5_*.csv`. Figures: `outputs_v2/stage5/figures/F5_*.png`.",
    ]
    (REPORT / "stage5_jrf_ni_draft.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
