"""
Thesis asset bundle v2: figures (PNG, 300 dpi, 16 cm wide), tables (CSV),
documents and MANIFEST.md, zipped to outputs_v2/thesis_assets_v2.zip.

Aggregate outputs only. Figures are drawn from committed aggregate tables or
from new aggregates computed here (counts, weighted means, histograms with
bins of < 10 households dropped, PSU-bootstrap CIs for pooled social-group
rates). No new models: the ROC/calibration figures refit the pre-specified
Stage 6 models (and the logged post-hoc P3) exactly as in
scripts/stage6_prediction.py to recover their predictions.

Style: 16 cm (6.3 in) width, 8 pt text, navy/orange two-colour scheme (grey
for benchmarks and context), sequential navy ramp for magnitudes, orange/navy
diverging for change.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src.paths import OUTPUTS_DIR, UKHLS_PANEL, V1_OUTPUTS_DIR, UKHLS_RAW_DIR  # noqa: E402

O = OUTPUTS_DIR
B = O / "thesis_assets_v2"
FIG, TAB, DOC = B / "figures", B / "tables", B / "docs"
W = 6.3  # 16 cm
NAVY, ORANGE, GREY, LGREY, INK, MUTED = "#2E5077", "#D55E00", "#8A96A3", "#C5CCD3", "#1F2933", "#5F6B7A"
GRID = "#E6E9ED"
SEQ = LinearSegmentedColormap.from_list("navy", ["#F2F5F9", "#9FB3C8", NAVY, "#162A40"])
DIV = LinearSegmentedColormap.from_list("div", [NAVY, "#F2F2F2", ORANGE])
MODEL_COL = {"LSTM": NAVY, "Prophet": ORANGE, "SARIMA": GREY, "TFT": "#E6AB02"}  # validated: all pairs normal ΔE ≥ 18.5, CVD ≥ 15.3
GOR = {1: "North East", 2: "North West", 3: "Yorkshire and the Humber", 4: "East Midlands",
       5: "West Midlands", 6: "East of England", 7: "London", 8: "South East", 9: "South West",
       10: "Wales", 11: "Scotland", 12: "Northern Ireland"}
ABBR = {"North East": "NE", "North West": "NW", "Yorkshire and the Humber": "YH", "East Midlands": "EM",
        "West Midlands": "WM", "East of England": "EE", "London": "LDN", "South East": "SE",
        "South West": "SW", "Wales": "WAL", "Scotland": "SCO", "Northern Ireland": "NI"}
REG: list[dict] = []   # manifest registry

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.edgecolor": "#4B5563",
                     "axes.linewidth": 0.6, "font.family": "DejaVu Sans"})


def git_commit(path: Path) -> str:
    rel = str(path.relative_to(ROOT))
    out = subprocess.run(["git", "log", "-1", "--format=%h", "--", rel], cwd=ROOT, capture_output=True, text=True)
    return out.stdout.strip() or "untracked"


def clean(ax, grid="y"):
    ax.spines[["top", "right"]].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.6)
        ax.set_axisbelow(True)


def save(fig, name: str, number: str, desc: str, sources: list[str]) -> None:
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    REG.append(dict(kind="figure", file=f"figures/{name}.png", number=number, description=desc,
                    script="scripts/build_thesis_assets.py", sources=sources))


def table(df: pd.DataFrame, name: str, number: str, desc: str, sources: list[str]) -> None:
    df.to_csv(TAB / f"{name}.csv", index=False)
    REG.append(dict(kind="table", file=f"tables/{name}.csv", number=number, description=desc,
                    script="scripts/build_thesis_assets.py", sources=sources))


def rd(rel: str) -> pd.DataFrame:
    return pd.read_csv(O / rel)


def nm(s):
    return pd.to_numeric(s, errors="coerce")


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
_PANEL = None


def panel() -> pd.DataFrame:
    global _PANEL
    if _PANEL is None:
        cols = ["hidp", "wave", "psu", "hh_xw", "gor_dv", "interview_year", "interview_month",
                "fuel_to_income_ratio", "high_fuel_vulnerable", "fes_magnitude_growth3", "fes_delta_growth3",
                "finnow", "tenure_dv", "family_composition_group", "employment_group", "ethnicity_group",
                "disability_free"]
        df = pd.read_csv(UKHLS_PANEL, usecols=cols)
        df["region"] = df.gor_dv.map(GOR)
        _PANEL = df
    return _PANEL


def wave_mean(g: pd.DataFrame, col: str) -> float:
    g = g[g[col].notna() & (g.hh_xw > 0)]
    r = [np.average(x[col].astype(float), weights=x.hh_xw) for _, x in g.groupby("wave")]
    return float(np.mean(r)) if r else np.nan


def boot_wave_mean(df: pd.DataFrame, dim: str, col: str, n_boot=1000, seed=20260926) -> pd.DataFrame:
    """Pooled wave-mean weighted rate per category with PSU-bootstrap CIs."""
    d = df[df[dim].notna() & df[col].notna() & (df.hh_xw > 0)]
    cats = sorted(d[dim].unique())
    waves = sorted(d.wave.unique())
    cell = d.assign(wy=d.hh_xw * d[col], one=1).groupby(["psu", dim, "wave"])[["wy", "hh_xw", "one"]].sum()
    psus = cell.index.get_level_values("psu").unique()
    cols = pd.MultiIndex.from_product([cats, waves])
    mat = lambda c: cell[c].unstack([dim, "wave"]).reindex(index=psus, columns=cols, fill_value=0).fillna(0).values  # noqa: E731
    WY, Wt = mat("wy"), mat("hh_xw")
    k, nw = len(cats), len(waves)

    def est(m):
        wy, w = m @ WY, m @ Wt
        r = np.divide(wy, w, out=np.full_like(wy, np.nan), where=w > 0).reshape(k, nw)
        return 100 * np.nanmean(r, axis=1)
    pt = est(np.ones(len(psus)))
    rng = np.random.default_rng(seed)
    bs = np.array([est(rng.multinomial(len(psus), np.full(len(psus), 1 / len(psus)))) for _ in range(n_boot)])
    n = d.groupby(dim).size().reindex(cats)
    out = pd.DataFrame(dict(dimension=dim, category=cats, n=n.values, pct_weighted=pt,
                            ci_low=np.nanpercentile(bs, 2.5, axis=0), ci_high=np.nanpercentile(bs, 97.5, axis=0)))
    out.loc[out.n < 100, ["pct_weighted", "ci_low", "ci_high"]] = np.nan
    return out


def uk_map(ax, values: pd.Series, cmap, norm=None, fmt="{:.1f}", label=""):
    from src.ukhls_geo_maps import load_region_boundaries
    g = load_region_boundaries().merge(values.rename("v"), left_on="region", right_index=True, how="left")
    g.plot(column="v", cmap=cmap, norm=norm, ax=ax, edgecolor="white", linewidth=0.5,
           vmin=None if norm else values.min(), vmax=None if norm else values.max())
    pos = {"London": (1.55, 51.0), "South East": (-0.9, 50.95), "South West": (-3.7, 50.75)}  # lon/lat label anchors
    for _, r in g.iterrows():
        pt = r.geometry.representative_point()
        xy = pos.get(r.region, (pt.x, pt.y))
        kw = dict(arrowprops=dict(arrowstyle="-", color=INK, lw=0.5)) if r.region == "London" else {}
        ax.annotate(f"{ABBR[r.region]}\n{fmt.format(r.v)}", (r.geometry.centroid.x, r.geometry.centroid.y) if kw else xy,
                    xytext=xy, ha="center", va="center", fontsize=5.5, color=INK, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75), **kw)
    ax.set_axis_off()
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm or plt.Normalize(values.min(), values.max()))
    cb = plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.01)
    cb.set_label(label, fontsize=7)
    cb.ax.tick_params(labelsize=6.5)


# ---------------------------------------------------------------------------
# Chapter 3 figures
# ---------------------------------------------------------------------------
def fig3_1():
    fig, ax = plt.subplots(figsize=(W, 4.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 72); ax.set_axis_off()

    def box(x, y, w, h, text, fc="#EEF2F7", ec=NAVY, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=0.9))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=6.0, color=INK,
                fontweight="bold" if bold else "normal", wrap=True)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.8))
    box(2, 60, 30, 9, "Monthly gas, electricity and\ncarbon price growth, 2006–26", bold=True)
    box(68, 60, 30, 9, "UKHLS household panel, waves\na–o (339,201 household-waves)", bold=True)
    box(2, 44, 30, 11, "Stage 2: rolling core forecasts\nSARIMA, Prophet, LSTM, TFT;\nper-origin tuning; vs naive\nand seasonal-naive benchmarks")
    box(68, 44, 30, 11, "Stage 1: outcome audit\nrouting-aware fuel spend;\nS1/S2 bounds; actual interview\ndates; cross-sectional weights")
    box(35, 44, 30, 11, "FES (growth-only)\nDec Y−1 vintage, matched to\ninterview month; past-only\nz-scores; Delta = fcst − real.", fc="#FDF1EA", ec=ORANGE)
    box(2, 25, 30, 12, "Stage 3: driver model\nlogit, year FE, PSU-clustered;\nstrain components separate;\nNI-oil sequence (AMEs)")
    box(35, 25, 30, 12, "Stage 4: resources and H1\nformative composites (CFA\nfailed pre-set criteria);\nresources × Delta (H1)")
    box(68, 25, 30, 12, "Stage 5: JRF comparison\ntime-matched, weighted,\nPSU-bootstrap CIs;\nNorthern Ireland and oil")
    box(18, 6, 30, 12, "Stage 6: next-wave prediction\nP0 current burden (benchmark);\nP1 household; P2 = P1 + FES;\nheld-out m→n, n→o")
    box(52, 6, 30, 12, "Stage 7: scope, sensitivity\nequivalised-income check;\nv1 CVAE, fuzzy, SVM:\nappendix only (not rerun)", fc="#F5F6F8", ec=GREY)
    arrow(17, 60, 17, 55.5); arrow(83, 60, 83, 55.5); arrow(32, 49.5, 35, 49.5); arrow(68, 49.5, 65, 49.5)
    arrow(40, 44, 17, 37.5); arrow(50, 44, 50, 37.5); arrow(60, 44, 83, 37.5)
    arrow(50, 25, 33, 18.5); arrow(83, 25, 67, 18.5)
    save(fig, "fig3-1_pipeline_v2", "Figure 3-1",
         "Analysis pipeline, rerun v2 (new diagram; the v1 image had no source in the repo)", [])


def fig3_2():
    d = panel()
    t = pd.crosstab(d.wave, d.interview_year.astype(int))
    masked = t.where(t >= 10)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 2.8), gridspec_kw={"width_ratios": [2.3, 1]})
    im = a1.imshow(masked.values, cmap=SEQ, aspect="auto")
    a1.set_xticks(range(len(t.columns))); a1.set_xticklabels(t.columns, rotation=90, fontsize=6.5)
    a1.set_yticks(range(len(t.index))); a1.set_yticklabels(t.index, fontsize=6.5)
    a1.set_xlabel("Interview year (actual household interview date)"); a1.set_ylabel("UKHLS wave")
    a1.set_title("A. Household-waves by wave and interview year", loc="left")
    cb = plt.colorbar(im, ax=a1, fraction=0.04, pad=0.02); cb.ax.tick_params(labelsize=6); cb.set_label("households", fontsize=6.5)
    m = d.interview_month.dropna().astype(int).value_counts().sort_index()
    a2.bar(m.index, m.values / 1000, color=NAVY, width=0.7)
    a2.set_xticks(range(1, 13)); a2.set_xticklabels("JFMAMJJASOND", fontsize=6.5)
    a2.set_ylabel("household-waves (thousands)"); a2.set_title("B. Interview month", loc="left")
    clean(a2)
    fig.tight_layout()
    save(fig, "fig3-2_interview_timing", "Figure 3-2",
         "Interview timing from actual household interview dates (cells < 10 masked)", ["outputs_v2/ukhls_cleaned/ukhls_panel.csv (local, aggregated)"])
    return t


def fig3_3():
    t = rd("audit/missing_by_mode_wave.csv")
    t["pct"] = nm(t.pct_missing)
    fig, ax = plt.subplots(figsize=(W, 2.8))
    order = ["CAPI (face-to-face)", "CATI (telephone)", "CAWI (web)"]
    sty = {"CAPI (face-to-face)": (NAVY, "-", "o"), "CATI (telephone)": (GREY, "--", "s"), "CAWI (web)": (ORANGE, "-", "^")}
    waves = list("abcdefghijklmno")
    for mde in order:
        s = t[t["mode"] == mde].set_index("wave").reindex(waves)
        c, ls, mk = sty[mde]
        ax.plot(range(15), s.pct, color=c, ls=ls, marker=mk, ms=3.5, lw=1.4, label=mde)
    allw = t.groupby("wave").apply(lambda g: 100 * nm(g.missing).sum() / nm(g.n).sum()).reindex(waves)
    ax.plot(range(15), allw, color=INK, lw=0.8, ls=":", label="all modes")
    ax.set_xticks(range(15)); ax.set_xticklabels(waves)
    ax.set_xlabel("UKHLS wave"); ax.set_ylabel("% with fuel spend missing\n(item non-response)")
    ax.legend(frameon=False, ncol=4, loc="upper left")
    ax.set_ylim(0, 40)
    clean(ax)
    fig.tight_layout()
    save(fig, "fig3-3_missing_spend_wave_mode", "Figure 3-3",
         "Fuel-spend item non-response by interview mode and wave (households reporting electricity)",
         ["outputs_v2/audit/missing_by_mode_wave.csv"])


def fig3_4():
    d = panel()
    fig, axs = plt.subplots(1, 3, figsize=(W, 2.3))
    r = d.fuel_to_income_ratio.dropna()
    bins = np.arange(0, 0.41, 0.01)
    c, e = np.histogram(r.clip(upper=0.4), bins=bins)
    c = np.where(c < 10, 0, c)
    axs[0].bar(e[:-1], c / 1000, width=0.009, align="edge", color=NAVY)
    axs[0].axvline(0.10, color=ORANGE, lw=1.1, ls="--")
    axs[0].text(0.105, axs[0].get_ylim()[1] * 0.85, "10%", color=ORANGE, fontsize=7)
    axs[0].set_xlabel("Fuel-to-income ratio (≥ 0.4 in last bin)"); axs[0].set_ylabel("household-waves (thousands)")
    axs[0].set_title("A. Outcome", loc="left")
    for ax, col, lab, ttl in [(axs[1], "fes_magnitude_growth3", "FES magnitude (growth-only)", "B. FES magnitude"),
                              (axs[2], "fes_delta_growth3", "FES Delta (growth-only)", "C. FES Delta")]:
        x = d[col].dropna()
        cc, ee = np.histogram(x, bins=30)
        cc = np.where(cc < 10, 0, cc)
        ax.bar(ee[:-1], cc / 1000, width=np.diff(ee), align="edge", color=NAVY, edgecolor="white", lw=0.3)
        ax.set_xlabel(lab); ax.set_title(ttl, loc="left")
        clean(ax)
    clean(axs[0])
    fig.tight_layout()
    save(fig, "fig3-4_distributions_outcome_fes", "Figure 3-4",
         "Distribution of the fuel-to-income ratio (primary) and of FES as attached to households, interviews 2010+ for FES (bins < 10 dropped)",
         ["outputs_v2/ukhls_cleaned/ukhls_panel.csv (local, aggregated)"])


def fig3_5():
    d = panel()
    cnt = d.groupby("region").size()
    fig, ax = plt.subplots(figsize=(W * 0.62, 4.2))
    uk_map(ax, cnt / 1000, SEQ, fmt="{:.1f}k", label="household-waves (thousands)")
    ax.set_title("Household-waves by region, waves a–o pooled", loc="left")
    save(fig, "fig3-5_regional_counts", "Figure 3-5", "UKHLS household-waves by region", ["panel (aggregated)"])
    return cnt


def fig3_6():
    c = pd.read_csv(ROOT / "data/processed/core_energy_carbon.csv", parse_dates=["date"])
    fig, axs = plt.subplots(3, 1, figsize=(W, 4.2), sharex=True)
    for ax, col, ttl in zip(axs, ["gas_growth", "electricity_growth", "carbon_growth"],
                            ["Gas price growth (% y/y)", "Electricity price growth (% y/y)", "Carbon (EUA) price growth (% y/y, log return)"]):
        ax.plot(c.date, c[col], color=NAVY, lw=1)
        ax.axhline(0, color=LGREY, lw=0.7)
        ax.set_title(ttl, loc="left")
        clean(ax)
    axs[-1].set_xlabel("Month")
    fig.tight_layout()
    save(fig, "fig3-6_core_forecast_vars", "Figure 3-6", "Core forecasting series, May 2006–March 2026",
         ["data/processed/core_energy_carbon.csv"])
    return c


# ---------------------------------------------------------------------------
# Chapter 4 figures
# ---------------------------------------------------------------------------
def fig4_1():
    y = rd("fes_eval/thesis_table_relrmse_by_year.csv")
    fig, axs = plt.subplots(1, 3, figsize=(W, 2.5), sharey=True)
    for ax, s in zip(axs, ["gas", "electricity", "carbon"]):
        g = y[y.series == s]
        ax.axhline(1, color=GREY, lw=0.8)
        ax.plot(g.target_year, g.v2_core_relRMSE_vs_naive, color=NAVY, marker="o", ms=3, lw=1.2, label="vs naive")
        ax.plot(g.target_year, g.v2_core_relRMSE_vs_snaive, color=ORANGE, marker="s", ms=3, lw=1.0, ls="--",
                label="vs seasonal naive")
        ax.set_yscale("log"); ax.set_title(s.capitalize(), loc="left")
        ax.set_yticks([0.1, 0.3, 1, 3, 10]); ax.set_yticklabels(["0.1", "0.3", "1", "3", "10"])
        ax.set_xticks([2010, 2015, 2020, 2025])
        clean(ax)
    axs[0].set_ylabel("Relative RMSE, v2 core\n(< 1 = better than benchmark)")
    axs[0].legend(frameon=False, loc="upper left")
    fig.supxlabel("Target year", fontsize=8)
    fig.tight_layout()
    save(fig, "fig4-1_relrmse_by_year", "Figure 4-1", "v2 forecast accuracy relative to naive and seasonal-naive benchmarks, by target year",
         ["outputs_v2/fes_eval/thesis_table_relrmse_by_year.csv"])


def fig4_2():
    m = rd("fes/fes_rolling_monthly.csv")
    m["date"] = pd.to_datetime(m.date)
    m["forecast"] = m[[f"z_{s}_core" for s in ("gas", "electricity", "carbon")]].sum(axis=1)
    m["realised"] = m[[f"z_{s}_actual" for s in ("gas", "electricity", "carbon")]].sum(axis=1, min_count=3)
    fig, ax = plt.subplots(figsize=(W, 2.6))
    ax.axhline(0, color=LGREY, lw=0.7)
    ax.plot(m.date, m.forecast, color=NAVY, lw=1.3, label="Forecast (Dec Y−1 vintage)")
    ax.plot(m.date, m.realised, color=ORANGE, lw=1.1, ls="--", label="Realised")
    r = m[["forecast", "realised"]].corr().iloc[0, 1]
    ax.set_ylabel("Growth-only FES\n(sum of 3 z-scores)"); ax.set_xlabel("Target month")
    ax.legend(frameon=False, loc="upper left", title=f"r = {r:.2f}, 192 months", title_fontsize=7)
    clean(ax)
    fig.tight_layout()
    save(fig, "fig4-2_fes_growth3_forecast_vs_realised", "Figure 4-2",
         "Growth-only FES: forecast vs realised by target month, 2010–2025", ["outputs_v2/fes/fes_rolling_monthly.csv"])


def fig4_3():
    w = rd("descriptives/prevalence_by_wave.csv")
    fig, ax = plt.subplots(figsize=(W, 2.9))
    x = np.arange(len(w))
    ax.fill_between(x, w.s1_lower_bound_pct_weighted, w.primary_pct_weighted, color=NAVY, alpha=0.18, lw=0,
                    label="Lower bound (S1: non-response amounts as £0)")
    ax.plot(x, w.primary_pct_weighted, color=NAVY, lw=1.8, marker="o", ms=3.5, label="Primary (routing-corrected, complete-case)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a}\n{b % 100:02d}–{c % 100:02d}" for a, b, c in zip(w.wave, w.fieldwork_first_year, w.fieldwork_last_year)], fontsize=6.5)
    ax.set_ylabel("% households with fuel spend\n≥ 10% of net income (weighted)")
    ax.set_xlabel("UKHLS wave and fieldwork years (1st–99th percentile of interview dates, 20xx)")
    ax.set_ylim(0, 14.5)
    ax.legend(frameon=False, loc="lower left")
    clean(ax)
    fig.tight_layout()
    save(fig, "fig4-3_trend_wave_s1band", "Figure 4-3", "National trend by wave, weighted, with S1 lower bound",
         ["outputs_v2/descriptives/prevalence_by_wave.csv"])


def fig4_4():
    y = rd("descriptives/prevalence_by_interview_year.csv")
    y["p"], y["s"], y["n"] = nm(y.primary_pct_weighted), nm(y.s1_lower_bound_pct_weighted), nm(y.primary_n)
    full = y[y.n >= 1000]
    fig, ax = plt.subplots(figsize=(W, 2.9))
    ax.axvspan(2021.75, 2024.85, color=ORANGE, alpha=0.08, lw=0)
    ax.fill_between(full.interview_year, full.s, full.p, color=NAVY, alpha=0.18, lw=0, label="S1 lower bound")
    ax.plot(full.interview_year, full.p, color=NAVY, lw=1.8, marker="o", ms=3.5, label="Primary (weighted)")
    ax.annotate("JRF cost-of-living tracker (UK Poverty 2025, p.108):\nlow-income households going without essentials\n"
                "peaked at 75% in Oct 2022, 69% by Oct 2024", xy=(2022.8, 14.4), xytext=(2009.6, 13.4), fontsize=6.6,
                color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax.set_xticks(full.interview_year); ax.tick_params(axis="x", rotation=90)
    ax.set_ylim(0, 17); ax.set_ylabel("% households with fuel spend\n≥ 10% of net income"); ax.set_xlabel("Interview year")
    ax.legend(frameon=False, loc="lower left")
    clean(ax)
    fig.tight_layout()
    save(fig, "fig4-4_trend_vs_jrf_tracker", "Figure 4-4",
         "Trend by interview year with the JRF cost-of-living crisis window (shaded)", ["outputs_v2/descriptives/prevalence_by_interview_year.csv"])


def fig4_5():
    g = rd("descriptives/prevalence_by_group.csv").query("dimension == 'region'").set_index("group")
    fig, ax = plt.subplots(figsize=(W * 0.62, 4.2))
    uk_map(ax, nm(g.pct_weighted_wave_mean), SEQ, fmt="{:.1f}%", label="% fuel spend ≥ 10% of income")
    ax.set_title("Weighted prevalence by region, waves a–o (wave mean)", loc="left")
    save(fig, "fig4-5_region_map_weighted", "Figure 4-5", "Regional prevalence, weighted", ["outputs_v2/descriptives/prevalence_by_group.csv"])


def fig4_6():
    h = rd("descriptives/prevalence_region_by_year.csv")
    h["v"] = nm(h.pct_weighted)
    order = rd("descriptives/prevalence_by_group.csv").query("dimension == 'region'")
    order = order.assign(v=nm(order.pct_weighted_wave_mean)).sort_values("v", ascending=False).group.tolist()
    p = h.pivot(index="region", columns="interview_year", values="v").reindex(order)
    p = p.loc[:, [c for c in p.columns if p[c].notna().any()]]
    fig, ax = plt.subplots(figsize=(W, 3.4))
    im = ax.imshow(p.values, cmap=SEQ, aspect="auto", vmin=np.nanmin(p.values), vmax=np.nanmax(p.values))
    for i in range(p.shape[0]):
        for j in range(p.shape[1]):
            v = p.values[i, j]
            if np.isnan(v):
                ax.text(j, i, "–", ha="center", va="center", fontsize=5.5, color=MUTED)
            else:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=5.5,
                        color="white" if v > np.nanpercentile(p.values, 70) else INK)
    ax.set_xticks(range(p.shape[1])); ax.set_xticklabels(p.columns.astype(int), rotation=90, fontsize=6.5)
    ax.set_yticks(range(p.shape[0])); ax.set_yticklabels(p.index, fontsize=6.8)
    cb = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.01); cb.set_label("%", fontsize=7); cb.ax.tick_params(labelsize=6.5)
    ax.set_xlabel("Interview year (– = fewer than 100 households, suppressed)")
    fig.tight_layout()
    save(fig, "fig4-6_region_year_heatmap_masked", "Figure 4-6", "Weighted prevalence by region and interview year (cells n < 100 masked)",
         ["outputs_v2/descriptives/prevalence_region_by_year.csv"])


def fig4_7():
    c = rd("stage7/regional_change_early_late.csv").set_index("region").change_pp
    lim = float(np.abs(c).max())
    fig, ax = plt.subplots(figsize=(W * 0.62, 4.2))
    uk_map(ax, c, DIV, norm=TwoSlopeNorm(0, -lim, lim), fmt="{:+.1f}", label="change (pp), waves a–e → k–o")
    ax.set_title("Change in prevalence, waves a–e to k–o", loc="left")
    save(fig, "fig4-7_region_change_map", "Figure 4-7", "Change in weighted regional prevalence (navy = fall, orange = rise)",
         ["outputs_v2/stage7/regional_change_early_late.csv"])


def fig4_8():
    r = rd("resources/resource_by_region.csv").set_index("region").R_primary_weighted_mean
    d = panel()
    f = pd.Series({reg: wave_mean(g, "finnow") for reg, g in d.dropna(subset=["region"]).groupby("region")})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 3.6))
    uk_map(a1, r, SEQ, fmt="{:+.2f}", label="resource composite (SD units, sum of 3)")
    a1.set_title("A. Household resources (O+C+P)", loc="left")
    uk_map(a2, f, SEQ, fmt="{:.2f}", label="mean current financial difficulty (1–5)")
    a2.set_title("B. Current financial difficulty", loc="left")
    fig.tight_layout()
    save(fig, "fig4-8_resource_and_findifficulty_by_region", "Figure 4-8",
         "Weighted mean resource composite and current financial difficulty by region",
         ["outputs_v2/resources/resource_by_region.csv", "panel (aggregated)"])
    return f


def social_rates() -> pd.DataFrame:
    d = panel().copy()
    ten = {1: "Owned outright", 2: "Buying with mortgage", 3: "Social renting", 4: "Social renting",
           5: "Private renting", 6: "Private renting", 7: "Private renting"}
    d["tenure"] = d.tenure_dv.map(ten)
    d["disability"] = np.where(d.disability_free.isna(), None,
                               np.where(d.disability_free < 1, "Contains disabled adult", "No disabled adult"))
    out = [boot_wave_mean(d, dim, "high_fuel_vulnerable") for dim in
           ["region", "tenure", "family_composition_group", "employment_group", "ethnicity_group", "disability"]]
    return pd.concat(out, ignore_index=True)


def fig4_9(sr: pd.DataFrame):
    dims = [("disability", "Disability"), ("employment_group", "Employment"), ("family_composition_group", "Family"),
            ("tenure", "Tenure"), ("ethnicity_group", "Ethnicity of HRP")]
    sizes = [sr[sr.dimension == d].pct_weighted.notna().sum() for d, _ in dims]
    fig, axs = plt.subplots(len(dims), 1, figsize=(W, 7.2), sharex=True, gridspec_kw={"height_ratios": sizes})
    for ax, (dim, ttl) in zip(axs, dims):
        g = sr[sr.dimension == dim].dropna(subset=["pct_weighted"]).sort_values("pct_weighted")
        yy = np.arange(len(g))
        ax.errorbar(g.pct_weighted, yy, xerr=[g.pct_weighted - g.ci_low, g.ci_high - g.pct_weighted], fmt="o", ms=3.5,
                    color=NAVY, elinewidth=0.9, capsize=0, mec="white", mew=0.6)
        ax.set_yticks(yy); ax.set_yticklabels(g.category, fontsize=7)
        ax.set_title(ttl, loc="left"); ax.set_ylim(-0.6, len(g) - 0.4)
        clean(ax, "x")
    axs[-1].set_xlabel("% fuel spend ≥ 10% of net income, weighted (mean of waves a–o)\nwith 95% PSU-bootstrap CI")
    axs[-1].set_xlim(0, None)
    fig.tight_layout()
    save(fig, "fig4-9_social_groups_panel_ci", "Figure 4-9",
         "Pooled weighted prevalence by social group with 95% CIs (categories n < 100 suppressed)", ["panel (aggregated, bootstrap)"])


def fig4_10():
    p = rd("stage7/prepayment_by_vulnerability.csv")
    fig, ax = plt.subplots(figsize=(W * 0.6, 2.4))
    lab = ["Not vulnerable\n(< 10%)", "Vulnerable\n(≥ 10%)"]
    ax.barh(lab, p.pct_prepayment_weighted, color=[GREY, ORANGE], height=0.55)
    for i, v in enumerate(p.pct_prepayment_weighted):
        ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=7.5)
    ax.set_xlabel("% on a prepayment meter (weighted)"); ax.set_xlim(0, 30)
    clean(ax, "x")
    fig.tight_layout()
    save(fig, "fig4-10_prepayment", "Figure 4-10", "Prepayment-meter use by vulnerability status", ["outputs_v2/stage7/prepayment_by_vulnerability.csv"])


def fig4_11():
    d = rd("stage3/coefficients.csv").query("spec == 'primary' and model == 'main'").copy()
    d = d[~d.term.str.startswith("interview_year_")]
    cont = d.sd_in_sample.notna()
    d["x"] = np.where(cont, d.OR_per_sd, d.OR)
    d["lo"] = np.where(cont, np.exp(np.log(d.OR_ci_low) * d.sd_in_sample), d.OR_ci_low)
    d["hi"] = np.where(cont, np.exp(np.log(d.OR_ci_high) * d.sd_in_sample), d.OR_ci_high)
    d = d.sort_values("x")
    fig, ax = plt.subplots(figsize=(W, 4.6))
    yy = np.arange(len(d))
    col = np.where(d.term.str.startswith("fes_"), ORANGE, NAVY)
    for i, (x, lo, hi, c) in enumerate(zip(d.x, d.lo, d.hi, col)):
        ax.plot([lo, hi], [i, i], color=c, lw=1)
        ax.plot(x, i, "o", color=c, ms=4, mec="white", mew=0.6)
    ax.axvline(1, color=GREY, lw=0.8, ls="--")
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{l}{' (per SD)' if c else ''}" for l, c in zip(d.label, d.sd_in_sample.notna())], fontsize=7)
    ax.set_xscale("log"); t = [0.4, 0.5, 0.7, 1, 1.5, 2, 3]
    ax.set_xticks(t); ax.set_xticklabels([f"{v:g}" for v in t]); ax.minorticks_off()
    ax.set_xlabel(f"Odds ratio with 95% CI (PSU-clustered); continuous terms per SD\nn = {int(d.n.iloc[0]):,}; interview-year fixed effects not shown")
    clean(ax, "x")
    fig.tight_layout()
    save(fig, "fig4-11_driver_forest", "Figure 4-11", "Primary driver model odds ratios (FES in orange)", ["outputs_v2/stage3/coefficients.csv"])


def fig4_12():
    s = rd("stage4/h1_slopes.csv").query("model == 'primary'")
    bb = rd("stage4/h1_buffering_bound.csv").iloc[0]
    pr = rd("stage4/h1_logit_prob_slopes.csv")
    sd = bb.sd_delta
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 2.6))
    x = np.arange(3)
    lab = ["p10", "p50", "p90"]
    v = 100 * s.delta_slope * sd
    a1.errorbar(x, v, yerr=[v - 100 * s.ci_low * sd, 100 * s.ci_high * sd - v], fmt="o", color=NAVY, ms=4, capsize=0, elinewidth=1)
    a1.axhline(0, color=GREY, lw=0.7)
    a1.set_xticks(x); a1.set_xticklabels(lab); a1.set_xlabel("Resource composite percentile")
    a1.set_ylabel("Change in fuel-to-income ratio\n(pp of income per SD of Delta)")
    a1.set_title("A. Primary (OLS): no buffering", loc="left"); a1.set_xlim(-0.5, 2.5)
    clean(a1)
    a2.errorbar(x, pr.pp_per_sd_delta, yerr=[pr.pp_per_sd_delta - pr.ci_low, pr.ci_high - pr.pp_per_sd_delta], fmt="s", color=GREY,
                ms=4, capsize=0, elinewidth=1)
    a2.axhline(0, color=GREY, lw=0.7)
    a2.set_xticks(x); a2.set_xticklabels(lab); a2.set_xlabel("Resource composite percentile")
    a2.set_ylabel("Change in P(vulnerable)\n(pp per SD of Delta)")
    a2.set_title("B. Sensitivity (logit, probability scale)", loc="left"); a2.set_xlim(-0.5, 2.5)
    clean(a2)
    fig.tight_layout()
    save(fig, "fig4-12_h1_delta_slopes", "Figure 4-12",
         "H1: predicted FES Delta slopes at resource percentiles (95% CIs)",
         ["outputs_v2/stage4/h1_slopes.csv", "outputs_v2/stage4/h1_logit_prob_slopes.csv", "outputs_v2/stage4/h1_buffering_bound.csv"])


def fig4_13():
    t = rd("stage5/thesis_T5_2_comparison.csv").query("dimension == 'region' and window == 'primary'").copy()
    for c in ["pct", "ci_low", "ci_high"]:
        t[c] = nm(t[c])
    ag = rd("stage5/thesis_T5_3_agreement.csv").query("dimension == 'region' and window == 'primary'").set_index("subset")
    t = t.sort_values(["jrf_pct", "pct"])
    t["x"] = t.jrf_pct + t.groupby("jrf_pct").cumcount().sub((t.groupby("jrf_pct").jrf_pct.transform("size") - 1) / 2).mul(0.24)
    fig, ax = plt.subplots(figsize=(W, 3.9))
    for _, r in t.iterrows():
        c = ORANGE if r.category == "Northern Ireland" else NAVY
        ax.plot([r.x, r.x], [r.ci_low, r.ci_high], color=c, lw=1)
        ax.plot(r.x, r.pct, "o", color=c, ms=4.5, mec="white", mew=0.6)
    lab = {  # (dx points, dy points, ha)
        "Northern Ireland": (6, 0, "left"), "East of England": (-6, 0, "right"), "South East": (-6, -4, "right"),
        "South West": (6, -4, "left"), "East Midlands": (-6, 6, "right"), "North East": (-6, -8, "right"),
        "Scotland": (-6, 9, "right"), "Wales": (6, 5, "left"), "Yorkshire and the Humber": (6, 0, "left"),
        "London": (6, 0, "left"), "North West": (6, 0, "left"), "West Midlands": (-6, 0, "right")}
    for _, r in t.iterrows():
        dx, dy, ha = lab[r.category]
        ax.annotate(r.category, (r.x, r.pct), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center",
                    fontsize=6.6, color=INK if r.category == "Northern Ireland" else MUTED,
                    fontweight="bold" if r.category == "Northern Ireland" else "normal")
    ax.set_xlim(15.5, 28.5); ax.set_ylim(0, 18.5)
    ax.set_xlabel("JRF relative poverty rate, AHC (%), 2021/22–2022/23 (equal JRF rates offset slightly)")
    ax.set_ylabel("Fuel spend ≥ 10% of income (%), weighted\ninterviews Apr 2021–Mar 2023, 95% CI")
    ax.text(0.01, 0.02, f"Spearman ρ: all 12 regions {ag.loc['all','spearman_rho']:.2f}; excluding NI "
            f"{ag.loc['excl. Northern Ireland','spearman_rho']:.2f}", transform=ax.transAxes, fontsize=6.8, color=MUTED)
    clean(ax, "both")
    fig.tight_layout()
    save(fig, "fig4-13_jrf_regions", "Figure 4-13", "Regional fuel vulnerability vs JRF income poverty (time-matched)",
         ["outputs_v2/stage5/thesis_T5_2_comparison.csv", "outputs_v2/stage5/thesis_T5_3_agreement.csv"])


def fig4_14():
    t = rd("stage5/thesis_T5_2_comparison.csv").query("window == 'primary'").copy()
    for c in ["pct", "ci_low", "ci_high"]:
        t[c] = nm(t[c])
    dims = [("tenure", "Tenure (2022/23)"), ("family_type", "Family type (2022/23; JRF: child poverty)"),
            ("work_status", "Work status (2022/23; working-age)"), ("disability", "Disability (2022/23)"),
            ("ethnicity", "Ethnicity of HRP (2021/22–2022/23)")]
    fig, axs = plt.subplots(len(dims), 1, figsize=(W, 6.2), sharex=True, gridspec_kw={"height_ratios": [4, 2, 2, 2, 6]})
    for ax, (dim, ttl) in zip(axs, dims):
        g = t[t.dimension == dim].dropna(subset=["pct"]).sort_values("jrf_pct")
        yy = np.arange(len(g))
        ax.errorbar(g.pct, yy, xerr=[g.pct - g.ci_low, g.ci_high - g.pct], fmt="o", ms=3.5, color=NAVY, elinewidth=0.9,
                    capsize=0, mec="white", mew=0.6)
        ax.set_yticks(yy); ax.set_yticklabels([f"{c} (JRF {j:.0f}%)" for c, j in zip(g.category, g.jrf_pct)], fontsize=6.8)
        ax.set_title(ttl, loc="left"); ax.set_ylim(-0.6, len(g) - 0.4)
        clean(ax, "x")
    axs[-1].set_xlabel("Fuel spend ≥ 10% of income (%), weighted, 95% CI\ncategories ordered by JRF poverty rate (highest at top)")
    axs[-1].set_xlim(0, None)
    fig.tight_layout()
    save(fig, "fig4-14_jrf_other_dims", "Figure 4-14", "Other JRF dimensions, time-matched, with 95% CIs",
         ["outputs_v2/stage5/thesis_T5_2_comparison.csv"])


def fig4_15():
    n4 = rd("stage5/thesis_T5_4_northern_ireland.csv")
    ame = rd("stage3/ni_oil_ame.csv").query("spec == 'primary' and contrast == 'NI vs South East'").set_index("model")
    oil = n4[n4.item.str.contains("NI (Oil heating|No oil), primary", regex=True)].copy()
    oil["lab"] = oil.item.str.replace(", primary window", "").str.replace("NI ", "") + "\n(" + oil.detail + ")"
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(W, 2.6), gridspec_kw={"width_ratios": [1, 1.35]})
    yy = np.arange(len(oil))
    a1.errorbar(oil.value, yy, xerr=[oil.value - oil.ci_low, oil.ci_high - oil.value], fmt="o", color=ORANGE, ms=4.5,
                elinewidth=1, capsize=0, mec="white", mew=0.6)
    a1.set_yticks(yy); a1.set_yticklabels(oil.lab, fontsize=6.8); a1.set_ylim(-0.6, len(oil) - 0.4)
    a1.set_xlim(0, 25); a1.set_xlabel("Fuel spend ≥ 10% of income (%)\nNI, Apr 2021–Mar 2023, weighted")
    a1.set_title("A. NI by heating fuel", loc="left"); clean(a1, "x")
    order = ["ni_a_regionFE_filled", "ni_b_oil_filled", "ni_b_oil_rural_filled", "ni_c_ni_x_oil_filled", "ni_c_ni_x_oil_rural_filled"]
    names = ["Region FE", "+ oil", "+ oil + rural", "+ NI × oil", "+ NI × oil + rural"]
    m = ame.loc[order]
    y2 = np.arange(len(m))[::-1]
    a2.errorbar(m.ame_pp, y2, xerr=[m.ame_pp - m.ci_low_pp, m.ci_high_pp - m.ame_pp], fmt="o", color=NAVY, ms=4.5,
                elinewidth=1, capsize=0, mec="white", mew=0.6)
    a2.axvline(0, color=GREY, lw=0.7)
    a2.set_yticks(y2); a2.set_yticklabels(names, fontsize=7)
    a2.set_xlabel("NI vs South East: average marginal\neffect (pp), 95% CI; interviews 2010–25")
    a2.set_title("B. NI gap after oil and rurality", loc="left"); clean(a2, "x")
    fig.tight_layout()
    save(fig, "fig4-15_ni_oil", "Figure 4-15", "Northern Ireland: rates by heating fuel and the NI gap across models",
         ["outputs_v2/stage5/thesis_T5_4_northern_ireland.csv", "outputs_v2/stage3/ni_oil_ame.csv"])


def prediction_preds():
    import stage6_prediction as S
    pairs, _ = S.transitions()
    tr_all = pairs[~pairs.transition.isin(S.VALIDATION)].copy()
    va_all = pairs[pairs.transition.isin(S.VALIDATION)].copy()
    S.build_composites(tr_all, va_all)
    allp = pd.concat([tr_all, va_all])
    need = ["y", "psu"] + S.P0 + S.P1 + [S.FES_T1]
    cs = ~allp[need].isna().any(axis=1)
    tr, va = tr_all[cs.loc[tr_all.index]].copy(), va_all[cs.loc[va_all.index]].copy()
    S.standardise(tr, va, sorted(set(S.P0 + S.P1 + [S.FES_T1])))
    models = {"P0": S.P0, "P1": S.P1, "P2": S.P1 + [S.FES_T1], "P3": S.P0 + S.P1}
    return va.y.values.astype(int), {m: S.fit_predict(tr, va, c)[1] for m, c in models.items()}


def fig4_16_17(y, preds):
    from sklearn.metrics import roc_auc_score, roc_curve
    met = rd("stage6/metrics.csv").query("metric == 'auc'").set_index("model").estimate
    p3 = rd("stage6/posthoc_p3_metrics.csv").query("metric == 'auc' and label == 'POST-HOC / EXPLORATORY'").estimate.iloc[0]
    for m in ["P0", "P1", "P2"]:
        assert abs(roc_auc_score(y, preds[m]) - met[m]) < 1e-9, f"AUC mismatch {m}"
    assert abs(roc_auc_score(y, preds["P3"]) - p3) < 1e-9
    sty = {"P0": (GREY, "--", "P0 current burden (benchmark)"), "P1": (NAVY, "-", "P1 household predictors"),
           "P2": (ORANGE, ":", "P2 = P1 + FES"), "P3": (INK, "-.", "P3 = P0 + P1 (post-hoc)")}
    fig, ax = plt.subplots(figsize=(W * 0.62, W * 0.6))
    for m, p in preds.items():
        f, t, _ = roc_curve(y, p)
        c, ls, lab = sty[m]
        ax.plot(f, t, color=c, ls=ls, lw=1.4, label=f"{lab}, AUC {roc_auc_score(y, p):.3f}")
    ax.plot([0, 1], [0, 1], color=LGREY, lw=0.7)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.legend(frameon=False, loc="lower right", fontsize=6.6)
    clean(ax, "both")
    fig.tight_layout()
    save(fig, "fig4-16_roc_p0_p3", "Figure 4-16", "ROC curves, validation transitions m→n and n→o (n = 19,960)",
         ["outputs_v2/stage6/metrics.csv", "outputs_v2/stage6/posthoc_p3_metrics.csv"])
    fig, ax = plt.subplots(figsize=(W * 0.62, W * 0.6))
    top = 0
    for m, p in preds.items():
        d = pd.DataFrame({"p": p, "y": y})
        d["b"] = pd.qcut(d.p.rank(method="first"), 10, labels=False)
        g = d.groupby("b").agg(p=("p", "mean"), y=("y", "mean"))
        top = max(top, g.p.max(), g.y.max())
        c, ls, lab = sty[m]
        ax.plot(100 * g.p, 100 * g.y, color=c, ls=ls, lw=1.2, marker="o", ms=2.8, label=lab)
    lim = 100 * top * 1.08
    ax.plot([0, lim], [0, lim], color=LGREY, lw=0.7)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("Mean predicted risk by decile (%)"); ax.set_ylabel("Observed next-wave vulnerability (%)")
    ax.legend(frameon=False, loc="upper left", fontsize=6.6)
    clean(ax, "both")
    fig.tight_layout()
    save(fig, "fig4-17_calibration", "Figure 4-17", "Calibration by decile of predicted risk, validation transitions",
         ["outputs_v2/stage6/calibration.csv"])


def fig4_18():
    e = rd("stage7/sensitivity_equivalised_income_only.csv")
    fig, ax = plt.subplots(figsize=(W * 0.6, 2.5))
    ax.bar(e.household_size.astype(str), e.pct_flip, color=NAVY, width=0.6)
    for x, v in zip(e.household_size.astype(str), e.pct_flip):
        ax.text(x, v + 1, f"{v:.1f}%", ha="center", fontsize=7)
    ax.set_xlabel("Household size"); ax.set_ylabel("% whose 10% flag changes\n(all changes are into vulnerability)")
    ax.set_ylim(0, 55)
    clean(ax)
    fig.tight_layout()
    save(fig, "fig4-18_equivalisation_sensitivity", "Figure 4-18", "Sensitivity to equivalising income only (fuel spend not equivalised)",
         ["outputs_v2/stage7/sensitivity_equivalised_income_only.csv"])


def fig4_19():
    t = rd("stage7/prevalence_by_fes_tercile.csv")
    fig, axs = plt.subplots(1, 2, figsize=(W * 0.8, 2.4), sharey=True)
    for ax, f, ttl in zip(axs, ["fes_magnitude_growth3", "fes_delta_growth3"], ["A. FES magnitude", "B. FES Delta"]):
        g = t[t.fes == f]
        ax.bar(g.tercile, g.pct_vulnerable_weighted, color=NAVY, width=0.55)
        for x, v in zip(g.tercile, g.pct_vulnerable_weighted):
            ax.text(x, v + 0.2, f"{v:.1f}%", ha="center", fontsize=7)
        ax.set_title(ttl, loc="left"); ax.set_xlabel("Tercile (growth-only)")
        clean(ax)
    axs[0].set_ylabel("% fuel spend ≥ 10% of income\n(weighted, interviews 2010+)"); axs[0].set_ylim(0, 11)
    fig.tight_layout()
    save(fig, "fig4-19_fes_tercile", "Figure 4-19", "Prevalence by FES tercile (descriptive)", ["outputs_v2/stage7/prevalence_by_fes_tercile.csv"])


# ---------------------------------------------------------------------------
# Appendix figures
# ---------------------------------------------------------------------------
def figA_radial():
    p = rd("fes/forecast_performance_by_year.csv").query("mode == 'core'")
    series = ["gas", "electricity", "carbon"]
    years = sorted(p.target_year.unique())
    fig = plt.figure(figsize=(W * 0.8, W * 0.86))
    ax = fig.add_subplot(111, projection="polar")
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)   # clockwise from 12 o'clock; years run clockwise
    gap = np.deg2rad(8)
    sector = (2 * np.pi - len(series) * gap) / len(series)
    step = sector / len(years)
    for si, s in enumerate(series):
        g = p[p.series == s].set_index("target_year")
        start = si * (sector + gap) + gap / 2
        for yi, yv in enumerate(years):
            if yv not in g.index:
                continue
            th = start + (yi + 0.5) * step
            ax.bar(th, 1.0, width=step * 0.92, bottom=1.6, color=MODEL_COL[g.loc[yv, "model"]], edgecolor="white", lw=0.4)
            if yv in (years[0], years[-1]) or yv % 5 == 0:
                deg = np.rad2deg(th)
                rot = -deg if deg <= 90 or deg >= 270 else 180 - deg   # keep text upright
                ax.text(th, 2.95, str(yv), ha="center", va="center", fontsize=6, color=MUTED, rotation=rot,
                        rotation_mode="anchor")
        mid = start + sector / 2
        ax.text(mid, 3.55, s.capitalize(), ha="center", va="center", fontsize=8, fontweight="bold", color=INK)
    ax.set_yticks([]); ax.set_xticks([]); ax.spines["polar"].set_visible(False); ax.set_ylim(0, 3.8)
    counts = p.model.value_counts()
    ax.text(0, 0, f"{len(p)} series-years\n{years[0]}–{years[-1]}", ha="center", va="center", fontsize=7, color=MUTED)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in MODEL_COL.values()]
    ax.legend(handles, [f"{m} ({counts.get(m, 0)})" for m in MODEL_COL], frameon=False, loc="lower center",
              bbox_to_anchor=(0.5, -0.1), ncol=4, fontsize=7, title="Winning core model (number of target years won)",
              title_fontsize=7)
    save(fig, "figA_radial_winners", "Figure A-1",
         "Winning rolling core model by series and target year (v2), read clockwise from 12 o'clock",
         ["outputs_v2/fes/forecast_performance_by_year.csv"])


def figA_macro():
    m = pd.read_csv(ROOT / "data/processed/macro_controls.csv")
    cols = [c for c in ["inflation_growth", "gdp_growth", "gas_futures_price", "weather_volatility",
                        "gbp_eur", "elec_demand", "wind_generation", "solar_generation", "holiday_share"] if c in m.columns]
    if len(cols) < 4:
        cols = [c for c in m.columns if c != "date" and pd.api.types.is_numeric_dtype(m[c])][:9]
    n = len(cols)
    rows = int(np.ceil(n / 3))
    fig, axs = plt.subplots(rows, 3, figsize=(W, 1.7 * rows))
    for ax, c in zip(axs.flat, cols):
        ax.hist(m[c].dropna(), bins=30, color=NAVY, edgecolor="white", lw=0.3)
        ax.set_title(c, loc="left", fontsize=7); clean(ax)
    for ax in list(axs.flat)[n:]:
        ax.set_visible(False)
    fig.tight_layout()
    save(fig, "figA_macro_covariates", "Figure A-2", "Macro covariates (descriptive; not used by the v2 core-only forecasts)",
         ["data/processed/macro_controls.csv"])


def figA_trend_year():
    y = rd("descriptives/prevalence_by_interview_year.csv")
    y["p"], y["s"], y["n"] = nm(y.primary_pct_weighted), nm(y.s1_lower_bound_pct_weighted), nm(y.primary_n)
    full, part = y[y.n >= 1000], y[y.n < 1000]
    fig, ax = plt.subplots(figsize=(W, 2.8))
    ax.fill_between(full.interview_year, full.s, full.p, color=NAVY, alpha=0.18, lw=0, label="S1 lower bound")
    ax.plot(full.interview_year, full.p, color=NAVY, lw=1.8, marker="o", ms=3.5, label="Primary (weighted)")
    ax.plot(part.interview_year, part.p, "o", mfc="white", mec=NAVY, ms=4, ls="none", label="Partial year (n < 1,000)")
    ax.set_xticks(y.interview_year); ax.tick_params(axis="x", rotation=90)
    ax.set_ylim(0, 18); ax.set_xlabel("Interview year"); ax.set_ylabel("% fuel spend ≥ 10% of income")
    ax.legend(frameon=False, loc="lower left"); clean(ax)
    fig.tight_layout()
    save(fig, "figA_trend_interview_year", "Figure A-3", "Trend by interview year (supplementary)",
         ["outputs_v2/descriptives/prevalence_by_interview_year.csv"])


def _trim(img: np.ndarray) -> np.ndarray:
    rgb = img[..., :3]
    mask = (rgb < 0.97).any(axis=2)
    ys, xs = np.where(mask)
    return img[max(ys.min() - 5, 0):ys.max() + 5, max(xs.min() - 5, 0):xs.max() + 5]


def figA_v1():
    imgs = [(V1_OUTPUTS_DIR / "ukhls_cor_cvae/figures/cvae_sem_alignment_heatmap.png", "A. COR-CVAE vs COR-SEM factor alignment"),
            (V1_OUTPUTS_DIR / "ukhls_policy_maps/figures/policy_map2_fuzzy_membership.png", "B. Fuzzy 'vulnerable to loss' membership")]
    val = pd.read_csv(V1_OUTPUTS_DIR / "ukhls_vulnerability/tables/stage3_validation_against_objective_ratio.csv")
    fig = plt.figure(figsize=(W, 5.0))
    fig.text(0.01, 0.985, "v1 exploratory analyses (submitted draft): not re-estimated in v2; "
             "computed on the v1 outcome, SEM and FES timing", fontsize=7.5, color=ORANGE, va="top")
    for i, (p, ttl) in enumerate(imgs):
        ax = fig.add_axes([0.01 + i * 0.5, 0.30, 0.48, 0.60])
        ax.imshow(_trim(plt.imread(p))); ax.set_axis_off(); ax.set_title(ttl, fontsize=7.5, loc="left")
    ax = fig.add_axes([0.05, 0.03, 0.9, 0.18]); ax.set_axis_off()
    names = {"fuzzy_resource_depleted": "Fuzzy c-means: resource-depleted membership",
             "oneclass_anomaly_score": "One-class SVM anomaly score"}
    cell = [[names.get(r.method, r.method), f"{int(r.n):,}", f"{r.pearson_r_vs_ratio:.2f}", f"{r.spearman_r_vs_ratio:.2f}",
             f"{r.auc_vs_high_fuel_vulnerable:.2f}"] for r in val.itertuples()]
    tab = ax.table(cellText=cell, colLabels=["Score (v1)", "n", "Pearson r\nvs ratio", "Spearman ρ\nvs ratio", "AUC vs\n10% flag"],
                   loc="center", cellLoc="center", colWidths=[0.42, 0.12, 0.15, 0.15, 0.13])
    tab.auto_set_font_size(False); tab.set_fontsize(6.5); tab.scale(1, 1.5)
    for (r, c), k in tab.get_celld().items():
        k.set_edgecolor(LGREY); k.set_linewidth(0.5)
        if r == 0:
            k.set_facecolor("#EEF2F7")
    ax.set_title("C. Fuzzy c-means and one-class SVM scores vs the v1 fuel-to-income outcome", fontsize=7.5, loc="left")
    save(fig, "figA_v1_exploratory", "Figure A-4", "v1 exploratory models (CVAE, fuzzy c-means, one-class SVM), labelled v1",
         [str(p.relative_to(ROOT)) for p, _ in imgs] + ["outputs/ukhls_vulnerability/tables/stage3_validation_against_objective_ratio.csv"])


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def tables(t32_counts: pd.DataFrame, reg_counts: pd.Series, core: pd.DataFrame, sr: pd.DataFrame) -> None:
    w = rd("descriptives/prevalence_by_wave.csv")
    t = pd.DataFrame({"wave": w.wave, "fieldwork_period": w.fieldwork_first_year.astype(str) + "–" + w.fieldwork_last_year.astype(str),
                      "household_waves": w.n_households, "analytical_n_primary": w.primary_n,
                      "analytical_n_s1": w.s1_lower_bound_n, "analytical_n_v1": w.v1_n})
    t.loc[len(t)] = ["total", "", t.household_waves.sum(), t.analytical_n_primary.sum(), t.analytical_n_s1.sum(), t.analytical_n_v1.sum()]
    table(t, "T3-2_wave_obs_analytical_n", "Table 3-2", "Household-waves and analytical n by wave", ["outputs_v2/descriptives/prevalence_by_wave.csv"])

    table(rd("audit/sample_flow_reconciliation.csv"), "T3-3_sample_flow", "Table 3-3", "Sample flow 339,201 → analytical n (primary, S1, S2, v1)",
          ["outputs_v2/audit/sample_flow_reconciliation.csv"])

    r = pd.DataFrame([
        ("fuelhave1–4", "all households", "fuels used (electricity, gas, oil, other)", "defines which amounts are required"),
        ("fuelduel", "electricity AND gas used", "1 one bill / 2 separate", "−8 = not dual-fuel (not missing); DK/refused → separate amounts asked"),
        ("xpduely", "fuelduel = 1", "annual combined gas+electricity £", "−1/−2/−9 = item non-response → spend missing"),
        ("xpgasy, xpelecy", "fuelduel = 2 or DK/refused, or single-fuel household", "annual £", "−8 = fuel not used (structural 0); −1/−2/−9 → missing"),
        ("xpoily", "oil used (fuelhave3 = 1)", "annual £", "−8 = not used (0); non-response → missing"),
        ("xpsfly", "other fuel used (fuelhave4 = 1)", "annual £", "−8 = not used (0); non-response → missing"),
    ], columns=["variable", "asked if", "content", "v2 treatment"])
    table(r, "T3-4_fuel_code_routing", "Table 3-4", "Fuel-expenditure routing and code treatment (v1 errors: −8 fuelduel dropped; non-response zero-filled)",
          ["outputs_v2/audit_fuel_codes.csv"])

    al = rd("resources/composite_alpha.csv").set_index("domain").cronbach_alpha
    sa = rd("descriptives/strain_structure.csv").set_index("metric").value
    ms = pd.DataFrame([
        ("Outcome", "high_fuel_vulnerable", "annual fuel spend / (12 × monthly net income) ≥ 0.10", "routing-aware spend, complete-case; income < £1,200 excluded; ratio capped at 1", ""),
        ("Strain", "finnow", "current financial situation, 1 comfortable … 5 very difficult", "household mean of adults", f"{sa['cronbach_alpha_primary_items']:.2f} (3-item composite; not used as a scale)"),
        ("Strain", "scghq1_dv", "GHQ-12 Likert 0–36 (higher = more distress)", "household mean", ""),
        ("Strain", "finfut_risk", "financial expectations: 0 better / 0.5 same / 1 worse", "household mean; always with age", ""),
        ("Resources: OBJECT", "hsrooms, hsbeds, ncars, carval, hsval", "log(1+x) for £ items; z-scored", "mean of z (≥ 50% observed), re-standardised", f"{al['OBJECT']:.2f}"),
        ("Resources: CONDITION", "tenure_security, jbstat_security, bill_security", "0–1 security codings", "as above", f"{al['CONDITION']:.2f}"),
        ("Resources: PERSONAL", "sf1_good (self-rated health), health_good (no long-standing illness), qfhigh_band", "higher = better", "as above", f"{al['PERSONAL']:.2f}"),
        ("Resources: ENERGY", "fihhmnnet1_dv, fiyrinvinc_dv", "log(1+x)", "as above", f"{al['ENERGY']:.2f}"),
        ("Disability", "health + disdif1–12", "long-standing illness and ≥ 1 substantial difficulty", "household: any observed adult", ""),
        ("Oil use", "fuelhave3", "1 = uses heating oil", "", ""),
        ("Rural", "urban_dv", "2 = rural", "missing filled from adjacent wave if no move", ""),
        ("FES", "fes_magnitude_growth3, fes_delta_growth3", "sum of 3 growth z-scores (past-only moments); Delta = forecast − realised (m−1)", "Dec Y−1 vintage; interviews 2010+", ""),
    ], columns=["construct", "items", "coding", "construction", "Cronbach alpha (descriptive)"])
    table(ms, "T3-5_measures", "Table 3-5", "Measures: items, coding, construction and alpha (formative indices)",
          ["outputs_v2/resources/composite_alpha.csv", "outputs_v2/descriptives/strain_structure.csv"])

    rows = []
    for c in ["gas_growth", "electricity_index", "electricity_growth", "carbon_growth"]:
        s = core[c]
        rows.append(dict(variable=c, n=int(s.notna().sum()), missing=int(s.isna().sum()), mean=s.mean(), sd=s.std(),
                         median=s.median(), skew=s.skew(), excess_kurtosis=s.kurt()))
    table(pd.DataFrame(rows).round(3), "T3-6_forecasting_variables", "Table 3-6", "Core forecasting series, May 2006–March 2026",
          ["data/processed/core_energy_carbon.csv"])

    table(rd("stage5/thesis_T5_1_jrf_metadata.csv"), "T3-7_jrf_metadata", "Table 3-7", "JRF benchmark metadata and matching windows",
          ["outputs_v2/stage5/thesis_T5_1_jrf_metadata.csv"])

    fa = rd("fes_eval/thesis_table_forecast_accuracy.csv")
    pi = rd("fes_eval/uncertainty_pi.csv")[["version", "series", "pi95_coverage_pct"]]
    table(fa.merge(pi, on=["version", "series"], how="left").round(4), "T4-1_forecast_accuracy_pi", "Table 4-1",
          "Forecast accuracy (relative RMSE, DM p) and 95% PI coverage", ["outputs_v2/fes_eval/thesis_table_forecast_accuracy.csv", "outputs_v2/fes_eval/uncertainty_pi.csv"])

    table(sr.round(3), "T4-2_social_regional_rates", "Table 4-2", "Pooled weighted prevalence by region and social group, 95% PSU-bootstrap CI (n < 100 suppressed)",
          ["panel (aggregated, bootstrap)"])

    table(rd("stage3/thesis_table_primary.csv"), "T4-3_driver_model", "Table 4-3", "Primary driver model (logit, year FE, PSU-clustered)",
          ["outputs_v2/stage3/thesis_table_primary.csv"])
    table(rd("stage3/thesis_table_per_sd.csv"), "T4-4_per_sd", "Table 4-4", "Continuous predictors ranked by |log OR per SD|",
          ["outputs_v2/stage3/thesis_table_per_sd.csv"])

    h = rd("stage4/h1_coefficients.csv").assign(part="coefficients")
    hs = rd("stage4/h1_slopes.csv").assign(part="Delta slopes")
    hb = rd("stage4/h1_buffering_bound.csv").assign(part="buffering bound")
    hd = rd("stage4/h1_decision.csv").assign(part="verdict")
    hp = rd("stage4/h1_logit_prob_slopes.csv").assign(part="logit probability-scale slopes (footnote)")
    table(pd.concat([hd, h, hs, hb, hp], ignore_index=True), "T4-5_h1", "Table 4-5", "H1: moderation, slopes, buffering bound, verdicts",
          ["outputs_v2/stage4/"])

    c = rd("stage5/thesis_T5_2_comparison.csv").assign(part="comparison")
    a = rd("stage5/thesis_T5_3_agreement.csv").assign(part="agreement")
    table(pd.concat([c, a], ignore_index=True), "T4-6_jrf_comparison_agreement", "Table 4-6", "JRF comparison with CIs and agreement",
          ["outputs_v2/stage5/thesis_T5_2_comparison.csv", "outputs_v2/stage5/thesis_T5_3_agreement.csv"])

    table(rd("stage3/ni_oil_ame.csv").query("spec == 'primary'"), "T4-7_ni_oil_ame", "Table 4-7", "NI-oil sequence: average marginal effects (pp)",
          ["outputs_v2/stage3/ni_oil_ame.csv"])

    tp = rd("stage6/thesis_table_prediction.csv")
    p3 = rd("stage6/posthoc_p3_metrics.csv").query("label == 'POST-HOC / EXPLORATORY'")
    g = p3.set_index("metric")
    f = lambda k, pct=False: (f"{100*g.loc[k,'estimate']:.1f} [{100*g.loc[k,'ci_low']:.1f}, {100*g.loc[k,'ci_high']:.1f}]"  # noqa: E731
                              if pct else f"{g.loc[k,'estimate']:.3f} [{g.loc[k,'ci_low']:.3f}, {g.loc[k,'ci_high']:.3f}]")
    tp.loc[len(tp)] = {"Model": "P3 = P0 + P1 (POST-HOC / EXPLORATORY)", "ROC-AUC [95% CI]": f("auc"), "PR-AUC [95% CI]": f("prauc"),
                       "Calibration slope": "", "Calibration-in-the-large": "", "Top 5%: sensitivity / PPV (%)": "",
                       "Top 10%: sensitivity / PPV (%)": f"{f('sens10', True)} / {f('ppv10', True)}",
                       "n validation": int(g.iloc[0].n_validation), "prevalence (%)": tp["prevalence (%)"].iloc[0]}
    dl = rd("stage6/delta_auc.csv")
    tp["note"] = ""
    tp.loc[tp.Model.str.startswith("P2"), "note"] = f"no improvement over P1 (ΔAUC {dl.set_index('comparison').loc['P2 - P1','delta_auc']:.4f})"
    table(tp, "T4-8_prediction_p0_p3", "Table 4-8", "Next-wave prediction, P0–P3 (P3 post-hoc)",
          ["outputs_v2/stage6/thesis_table_prediction.csv", "outputs_v2/stage6/posthoc_p3_metrics.csv", "outputs_v2/stage6/delta_auc.csv"])

    co = rd("stage3/coefficients.csv")
    fes = co[(co.model.isin(["main", "main_twoway_cluster"])) & co.term.isin(["fes_delta_growth3", "fes_delta"])]
    rob = [dict(estimate="FES Delta OR", specification=f"{r.spec} ({r.model})", value=r.OR, ci_low=r.OR_ci_low,
                ci_high=r.OR_ci_high, n=r.n) for r in fes.itertuples()]
    ame = rd("stage3/ni_oil_ame.csv").query("spec == 'primary' and contrast == 'NI vs South East'")
    rob += [dict(estimate="NI gap AME (pp)", specification=r.model, value=r.ame_pp, ci_low=r.ci_low_pp, ci_high=r.ci_high_pp, n=r.n)
            for r in ame.itertuples()]
    wv = rd("stage5/thesis_T5_4_northern_ireland.csv")
    rob += [dict(estimate="NI rate (%)", specification=r.item, value=r.value, ci_low=r.ci_low, ci_high=r.ci_high, n=None)
            for r in wv[wv.item.str.startswith("NI rate")].itertuples()]
    rob += [dict(estimate="H1 verdict", specification=r.model, value=r.verdict, ci_low=None, ci_high=None, n=r.n)
            for r in rd("stage4/h1_decision.csv").itertuples()]
    pb = rd("stage6/p2b_sensitivity.csv")
    rob += [dict(estimate="Prediction AUC (P2b sample)", specification=r.model, value=r.auc, ci_low=None, ci_high=None, n=r.n_validation)
            for r in pb.itertuples()]
    trend = rd("descriptives/prevalence_by_wave.csv")
    for col, lab in [("primary_pct_weighted", "primary"), ("s1_lower_bound_pct_weighted", "S1 lower bound"),
                     ("s2_plus_elec_nr_pct_weighted", "S2 incl. electricity not reported"), ("v1_pct_weighted", "v1 rule")]:
        for wv_ in ["a", "l", "n", "o"]:
            rob.append(dict(estimate=f"Prevalence wave {wv_} (%)", specification=lab,
                            value=trend.set_index("wave").loc[wv_, col], ci_low=None, ci_high=None, n=None))
    table(pd.DataFrame(rob), "T4-9_robustness_summary", "Table 4-9", "Robustness summary across pre-specified sensitivities",
          ["outputs_v2/stage3/coefficients.csv", "outputs_v2/stage3/ni_oil_ame.csv", "outputs_v2/stage5/thesis_T5_4_northern_ireland.csv",
           "outputs_v2/stage4/h1_decision.csv", "outputs_v2/stage6/p2b_sensitivity.csv", "outputs_v2/descriptives/prevalence_by_wave.csv"])

    pr = rd("stage3/thesis_table_primary.csv").set_index("term")
    ag = rd("stage5/thesis_T5_3_agreement.csv").set_index(["dimension", "window", "subset"]).spearman_rho
    hv = rd("stage4/h1_decision.csv").set_index("model")
    met = rd("stage6/metrics.csv").query("metric == 'auc'").set_index("model").estimate
    hyp = pd.DataFrame([
        ("H1 Resource moderation (COR buffering)", "Not supported", "author-confirmed",
         f"R × Delta = {hv.loc['primary','interaction']:.6f} (p = {hv.loc['primary','p']:.2f}); buffering ≤ 17% of slope"),
        ("H2 FES independent predictor", "Supported (small effect)", "PROPOSED — author to confirm",
         f"FES Delta OR {pr.loc['fes_delta_growth3','OR']:.3f} [{pr.loc['fes_delta_growth3','OR_ci_low']:.3f}, {pr.loc['fes_delta_growth3','OR_ci_high']:.3f}]; robust across sensitivities"),
        ("H3 Household financial position dominates macro stress", "Supported (reframed: current financial difficulty and employment security)", "PROPOSED — author to confirm",
         f"financial difficulty OR {pr.loc['finnow','OR']:.2f} per point; employment security OR {pr.loc['jbstat_security','OR']:.2f}; FES OR per SD 0.93"),
        ("H4 External validation with JRF", "Partially supported", "PROPOSED — author to confirm",
         f"tenure ρ {ag.loc[('tenure','primary','all')]:.2f}; family, work, disability same direction; region ρ {ag.loc[('region','primary','all')]:.2f} (excl. NI {ag.loc[('region','primary','excl. Northern Ireland')]:.2f}); ethnicity inconclusive"),
        ("H5 Prospective prediction", "Partially supported", "author-confirmed",
         f"P1 AUC {met['P1']:.3f}; benchmark P0 {met['P0']:.3f}; FES: no improvement"),
    ], columns=["hypothesis", "verdict", "status", "key statistics"])
    table(hyp, "T4-10_hypothesis_verdicts", "Table 4-10", "Hypothesis verdicts (H2–H4 wording proposed)",
          ["outputs_v2/stage3/thesis_table_primary.csv", "outputs_v2/stage5/thesis_T5_3_agreement.csv",
           "outputs_v2/stage4/h1_decision.csv", "outputs_v2/stage6/metrics.csv"])

    table(rd("fes_eval/appendix_table_mase.csv"), "TA-1_mase", "Table A-1", "MASE (appendix)", ["outputs_v2/fes_eval/appendix_table_mase.csv"])
    table(rd("stage6/per_transition_auc.csv"), "TA-2_per_transition_auc", "Table A-2", "Per-transition AUC", ["outputs_v2/stage6/per_transition_auc.csv"])
    table(co[~co.term.str.startswith(("interview_year_", "interview_month_"))], "TA-3_driver_sensitivities", "Table A-3",
          "All driver-model specifications and NI-oil models (coefficients)", ["outputs_v2/stage3/coefficients.csv"])
    table(rd("stage3/model_summary.csv"), "TA-4_driver_model_fit", "Table A-4", "Driver-model N, events, pseudo-R²", ["outputs_v2/stage3/model_summary.csv"])
    table(rd("fes_eval/thesis_table_relrmse_by_year.csv"), "TA-5_relrmse_by_year", "Table A-5", "Relative RMSE by target year",
          ["outputs_v2/fes_eval/thesis_table_relrmse_by_year.csv"])
    table(rd("fes_eval/diebold_mariano.csv"), "TA-6_diebold_mariano", "Table A-6", "Diebold–Mariano tests", ["outputs_v2/fes_eval/diebold_mariano.csv"])
    table(rd("stage6/calibration.csv"), "TA-7_calibration", "Table A-7", "Calibration slope and intercept", ["outputs_v2/stage6/calibration.csv"])
    table(rd("stage6/sample_flow.csv"), "TA-8_prediction_sample_flow", "Table A-8", "Prediction sample flow", ["outputs_v2/stage6/sample_flow.csv"])
    table(rd("resources/cfa_fit.csv"), "TA-9_cfa_fit", "Table A-9", "CFA fit (failed pre-registered criteria)", ["outputs_v2/resources/cfa_fit.csv"])
    table(rd("resources/cfa_loadings.csv"), "TA-10_cfa_loadings", "Table A-10", "CFA loadings", ["outputs_v2/resources/cfa_loadings.csv"])
    table(rd("stage7/sensitivity_equivalised_income_only.csv"), "TA-11_equivalised_income", "Table A-11",
          "Sensitivity to equivalising income only", ["outputs_v2/stage7/sensitivity_equivalised_income_only.csv"])
    table(rd("jrf/region_window_ci.csv"), "TA-12_region_window_ci", "Table A-12", "Regional rates, JRF windows, rank CIs",
          ["outputs_v2/jrf/region_window_ci.csv"])
    table(t32_counts.reset_index().rename(columns={"wave": "wave"}).melt(id_vars="wave", var_name="interview_year", value_name="households")
          .assign(households=lambda x: x.households.where(x.households >= 10, other=np.nan)).dropna(),
          "TA-13_interview_timing", "Table A-13", "Household-waves by wave × interview year (cells < 10 omitted)", ["panel (aggregated)"])
    table(reg_counts.rename("household_waves").reset_index(), "TA-14_regional_counts", "Table A-14", "Household-waves by region",
          ["panel (aggregated)"])


# ---------------------------------------------------------------------------
def docs() -> None:
    for src, name, num in [(O / "results_inventory.csv", "results_inventory.csv", "Inventory"),
                           (O / "reports/v1_to_v2_change_summary.md", "v1_to_v2_change_summary.md", "Change summary"),
                           (O / "reports/stage4_h1_draft.md", "stage4_h1_draft.md", "Draft (Ch4 H1)"),
                           (O / "reports/stage5_jrf_ni_draft.md", "stage5_jrf_ni_draft.md", "Draft (Ch4 JRF/NI)"),
                           (O / "reports/stage6_prediction_draft.md", "stage6_prediction_draft.md", "Draft (Ch4 prediction)"),
                           (O / "reports/appendix_scope_note.md", "appendix_scope_note.md", "Appendix note"),
                           (ROOT / "analysis_plan_rerun.md", "analysis_plan_rerun.md", "Plan + deviation log")]:
        shutil.copy(src, DOC / name)
        REG.append(dict(kind="document", file=f"docs/{name}", number=num, description="copied", script="(copied)",
                        sources=[str(src.relative_to(ROOT))]))


def bundle_check() -> list[str]:
    """Suppression and row-level checks over every CSV in the bundle."""
    import re
    # 71 model specifications x ~20 coefficients: aggregate, not row-level.
    ROW_OK = {"tables/TA-3_driver_sensitivities.csv"}
    probs = []
    for f in (B).rglob("*.csv"):
        d = pd.read_csv(f, dtype=str, keep_default_na=False)
        rel = f.relative_to(B)
        if len(d) > 1000 and str(rel) not in ROW_OK:
            probs.append(f"{rel}: {len(d)} rows (possible row-level data)")
        if any(re.fullmatch(r"(hidp|pidp|hrpid|uid|respondent_id|pno)", c.lower()) for c in d.columns):
            probs.append(f"{rel}: identifier column")
        for c in d.columns:
            if re.fullmatch(r"(n|n_.*|.*_n|.*count.*|households|household_waves|missing|observed|n validation|analytical_n.*|n_events|n_psu|n_both|n_complete.*|n_train|n_validation)", c.lower()) \
                    and c.lower() not in {"n_boot", "n_categories"}:
                v = pd.to_numeric(d[c], errors="coerce").abs()
                if ((v >= 1) & (v < 10)).any():
                    probs.append(f"{rel}: column '{c}' has counts 1-9")
        for c in d.columns:
            if d[c].str.contains(r"hidp=|pidp=|hrpid=", regex=True).any():
                probs.append(f"{rel}: identifier text in '{c}'")
    return probs


def manifest() -> None:
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    lines = ["# MANIFEST — thesis_assets_v2", "",
             f"Built by `scripts/build_thesis_assets.py` (commit `{git_commit(Path(__file__))}`) at repository HEAD `{head}` (branch `rerun-v2`), 2026-09-26.",
             "Aggregate outputs only; every CSV passed the bundle suppression check (counts 1–9, identifiers, row-level size).",
             "Figures: PNG, 300 dpi, drawn at 16 cm width. 'Source commit' = last commit of each source file.", "",
             "| File | Thesis number | Description | Source script | Source file(s) → commit |", "|---|---|---|---|---|"]
    for r in REG:
        srcs = []
        for s in r["sources"]:
            p = ROOT / s.split(" ")[0]
            if "panel" in s and ("local" in s or s.startswith("panel")):
                srcs.append("UKHLS panel (row-level, local only, never in git; aggregated in this script)")
            else:
                srcs.append(f"`{s}` → `{git_commit(p)}`" if p.exists() else f"`{s}`")
        lines.append(f"| `{r['file']}` | {r['number']} | {r['description']} | `{r['script']}` | {'; '.join(srcs) or '—'} |")
    (B / "MANIFEST.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    if B.exists():
        shutil.rmtree(B)
    for d in (FIG, TAB, DOC):
        d.mkdir(parents=True)
    print("ch3 figures", flush=True)
    fig3_1(); t32 = fig3_2(); fig3_3(); fig3_4(); reg = fig3_5(); core = fig3_6()
    print("ch4 figures", flush=True)
    fig4_1(); fig4_2(); fig4_3(); fig4_4(); fig4_5(); fig4_6(); fig4_7(); fig4_8()
    sr = social_rates()
    fig4_9(sr); fig4_10(); fig4_11(); fig4_12(); fig4_13(); fig4_14(); fig4_15()
    y, preds = prediction_preds()
    fig4_16_17(y, preds); fig4_18(); fig4_19()
    print("appendix", flush=True)
    figA_radial(); figA_macro(); figA_trend_year(); figA_v1()
    print("tables", flush=True)
    tables(t32, reg, core, sr)
    docs()
    probs = bundle_check()
    if probs:
        print("\n".join(probs))
        sys.exit("bundle check FAILED; not zipping")
    manifest()
    z = O / "thesis_assets_v2.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(B.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(B.parent))
    print(f"OK: {len(REG)} items; bundle check passed; {z} ({z.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
