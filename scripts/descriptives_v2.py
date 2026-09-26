"""
FES-independent descriptives for rerun v2 (analysis_plan_rerun.md, A1).

Weighted rates use each wave's household cross-sectional weight (`hh_xw`;
rows with zero weight are outside that wave's cross-sectional population
and are dropped from weighted estimates only). Pooled group rates are the
mean of per-wave weighted rates, each wave counting equally, so the pooled
figure is not dominated by the large early waves. Cells with fewer than
100 unweighted households are masked.

Outputs (outputs_v2/descriptives/):
  prevalence_by_wave.csv          primary / S1 / S2 / v1, unweighted + weighted
  prevalence_by_group.csv         region, tenure, family, employment,
                                  ethnicity, disability (pooled over waves)
  prevalence_region_by_year.csv   interview year x region, masked n<100
  strain_structure.csv            strain item correlations, alpha, coverage
  strain_item_correlations.csv    finnow, finfut_risk, GHQ, xphsdba, age (household level)
  trend_primary_with_s1_band.{png,pdf}
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
from src.paths import OUTPUTS_DIR, UKHLS_PANEL  # noqa: E402

OUT = OUTPUTS_DIR / "descriptives"
MIN_CELL_N = 100
VARIANTS = {"primary": "high_fuel_vulnerable", "s1_lower_bound": "high_fuel_vulnerable_s1",
            "s2_plus_elec_nr": "high_fuel_vulnerable_s2", "v1": "high_fuel_vulnerable_v1"}
GOR = {1: "North East", 2: "North West", 3: "Yorkshire and the Humber", 4: "East Midlands",
       5: "West Midlands", 6: "East of England", 7: "London", 8: "South East", 9: "South West",
       10: "Wales", 11: "Scotland", 12: "Northern Ireland"}
TENURE = {1: "Owned outright", 2: "Buying with mortgage", 3: "Social renting",
          4: "Social renting", 5: "Private renting", 6: "Private renting",
          7: "Private renting", 8: "Other"}


def load() -> pd.DataFrame:
    cols = (["hidp", "wave", "interview_year", "interview_month", "gor_dv", "tenure_dv",
             "family_composition_group", "employment_group", "ethnicity_group",
             "disability_free", "hh_xw", "fuelhave3", "fuel_to_income_ratio",
             "financial_strain_score", "financial_strain_score_v1",
             "financial_strain_score_lag1", "finnow", "finfut_risk", "scghq1_dv", "xphsdba", "dvage"]
            + list(VARIANTS.values()))
    df = pd.read_csv(UKHLS_PANEL, usecols=lambda c: c in cols)
    df["region"] = df.gor_dv.map(GOR)
    df["tenure"] = df.tenure_dv.map(TENURE)
    df["disability"] = np.where(df.disability_free.isna(), None,
                                np.where(df.disability_free < 1, "Contains disabled adult",
                                         "No disabled adult"))
    return df


def wrate(flag: pd.Series, w: pd.Series) -> float:
    ok = flag.notna() & (w > 0)
    return float(100 * np.average(flag[ok].astype(float), weights=w[ok])) if ok.any() else np.nan


def by_wave(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for wave, g in df.groupby("wave"):
        r = {"wave": wave, "start_year": 2009 + "abcdefghijklmno".index(wave),
             "n_households": len(g), "n_zero_weight": int((g.hh_xw == 0).sum())}
        for name, col in VARIANTS.items():
            f = g[col]
            r[f"{name}_n"] = int(f.notna().sum())
            r[f"{name}_pct_unweighted"] = float(100 * f.mean())
            r[f"{name}_pct_weighted"] = wrate(f, g.hh_xw)
        rows.append(r)
    return pd.DataFrame(rows)


def pooled_group(df: pd.DataFrame, dim: str, flag: str = "high_fuel_vulnerable") -> pd.DataFrame:
    rows = []
    for grp, g in df.dropna(subset=[dim]).groupby(dim):
        per_wave = [wrate(gw[flag], gw.hh_xw) for _, gw in g.groupby("wave") if gw[flag].notna().any()]
        n = int(g[flag].notna().sum())
        rows.append(dict(dimension=dim, group=grp, n=n,
                         pct_unweighted=float(100 * g[flag].mean()),
                         pct_weighted_wave_mean=float(np.nanmean(per_wave)) if per_wave else np.nan,
                         masked=n < MIN_CELL_N))
    out = pd.DataFrame(rows)
    out.loc[out.masked, ["pct_unweighted", "pct_weighted_wave_mean"]] = np.nan
    return out


def region_by_year(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (reg, yr), g in df.dropna(subset=["region"]).groupby(["region", "interview_year"]):
        n = int(g.high_fuel_vulnerable.notna().sum())
        rows.append(dict(region=reg, interview_year=int(yr), n=n,
                         pct_weighted=wrate(g.high_fuel_vulnerable, g.hh_xw) if n >= MIN_CELL_N else np.nan,
                         pct_unweighted=float(100 * g.high_fuel_vulnerable.mean()) if n >= MIN_CELL_N else np.nan,
                         masked=n < MIN_CELL_N))
    return pd.DataFrame(rows)


def strain_structure(df: pd.DataFrame) -> pd.DataFrame:
    items = ["finnow", "finfut_risk", "scghq1_dv"]
    x = df[items].dropna()
    z = (x - x.min()) / (x.max() - x.min())
    k = len(items)
    alpha = k / (k - 1) * (1 - z.var().sum() / z.sum(axis=1).var())
    corr = x.corr()
    rows = [dict(metric="cronbach_alpha_primary_items", value=alpha, n=len(x))]
    for i in range(k):
        for j in range(i + 1, k):
            rows.append(dict(metric=f"r({items[i]},{items[j]})", value=corr.iloc[i, j], n=len(x)))
    for c in ["financial_strain_score", "financial_strain_score_v1", "financial_strain_score_lag1"]:
        rows.append(dict(metric=f"coverage_{c}", value=df[c].notna().mean(), n=int(df[c].notna().sum())))
    rows.append(dict(metric="r(primary, v1)", value=df.financial_strain_score.corr(df.financial_strain_score_v1),
                     n=int(df[["financial_strain_score", "financial_strain_score_v1"]].dropna().shape[0])))
    rows.append(dict(metric="r(primary, lag1)", value=df.financial_strain_score.corr(df.financial_strain_score_lag1),
                     n=int(df[["financial_strain_score", "financial_strain_score_lag1"]].dropna().shape[0])))
    return pd.DataFrame(rows)


def trend_figure(w: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x = w.start_year
    ax.fill_between(x, w.s1_lower_bound_pct_weighted, w.primary_pct_weighted,
                    color="#2E5077", alpha=0.18, lw=0,
                    label="Lower bound (S1: non-response amounts counted as £0)")
    ax.plot(x, w.primary_pct_weighted, color="#2E5077", lw=2.2, marker="o", ms=4,
            label="Primary (complete-case, routing-corrected)")
    for xi, yi, lab in zip(x, w.primary_pct_weighted, w.wave):
        ax.annotate(lab, (xi, yi), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=7, color="#555")
    ax.set_xlabel("UKHLS wave (fieldwork start year)")
    ax.set_ylabel("% households with fuel spend ≥ 10% of net income\n(weighted)")
    ax.set_xticks(x)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(axis="y", color="#EAECEE")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, max(w.primary_pct_weighted.max(), w.s1_lower_bound_pct_weighted.max()) * 1.15)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"trend_primary_with_s1_band.{ext}", dpi=200)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load()
    w = by_wave(df)
    w.to_csv(OUT / "prevalence_by_wave.csv", index=False)
    trend_figure(w)
    groups = pd.concat([pooled_group(df, d) for d in
                        ["region", "tenure", "family_composition_group", "employment_group",
                         "ethnicity_group", "disability"]], ignore_index=True)
    groups.to_csv(OUT / "prevalence_by_group.csv", index=False)
    region_by_year(df).to_csv(OUT / "prevalence_region_by_year.csv", index=False)
    strain_structure(df).to_csv(OUT / "strain_structure.csv", index=False)
    items = ["finnow", "finfut_risk", "scghq1_dv", "xphsdba", "dvage"]
    pd.concat({"pearson": df[items].corr(), "spearman": df[items].corr("spearman")}).round(4).to_csv(
        OUT / "strain_item_correlations.csv")

    pd.set_option("display.width", 200)
    cols = ["wave", "start_year"] + [f"{v}_pct_weighted" for v in VARIANTS] + ["primary_pct_unweighted", "primary_n"]
    print(w[cols].round(2).to_string(index=False))
    print(groups.round(2).to_string(index=False))
    print(strain_structure(df).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
