"""
JRF UK Poverty 2025 comparison, rerun v2 (analysis_plan_rerun.md Stage 5).
Descriptive only: no FES-dependent quantities.

Every JRF value was checked against `UK Poverty 2025.pdf` (page and table
recorded in the metadata). UKHLS rates are time-matched to the JRF period:
households interviewed within the matching financial-year window.

Weighted rate for a window = per-wave weighted rate (household
cross-sectional weight `hh_xw`) among that wave's in-window households,
averaged over waves with each wave weighted by its in-window n. Cells with
fewer than 100 unweighted households are masked.

Outputs (outputs_v2/jrf/):
  jrf_metadata.csv          one row per dimension
  jrf_comparison.csv        category-level JRF vs UKHLS (weighted + unweighted)
  jrf_agreement.csv         Spearman/Pearson per dimension (weighted rates)
  ni_oil.csv                NI oil share and oil vs non-oil rates
  region_window_ci.csv      weighted regional rates in the region window
                            (Apr 2020-Mar 2023) with PSU-bootstrap 95% CIs
                            and rank distributions (1 = highest)

Bootstrap: 2,000 replicates resampling primary sampling units (psu) with
replacement across the whole sample (PSUs are not nested in regions because
households move), recomputing the window estimator for all 12 regions.
Households interviewed more than once in the window stay within their PSU.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.paths import OUTPUTS_DIR, UKHLS_PANEL, UKHLS_RAW_DIR  # noqa: E402

OUT = OUTPUTS_DIR / "jrf"
MIN_CELL_N = 100
GOR = {1: "North East", 2: "North West", 3: "Yorkshire and the Humber", 4: "East Midlands",
       5: "West Midlands", 6: "East of England", 7: "London", 8: "South East", 9: "South West",
       10: "Wales", 11: "Scotland", 12: "Northern Ireland"}
TENURE = {1: "Owned outright", 2: "Buying with mortgage", 3: "Social renting", 4: "Social renting",
          5: "Private renting", 6: "Private renting", 7: "Private renting"}

FY_2020_23 = ("2020-04", "2023-03")
FY_2022_23 = ("2022-04", "2023-03")

JRF = {
    "region": dict(
        values={"North East": 21, "North West": 25, "Yorkshire and the Humber": 23,
                "East Midlands": 20, "West Midlands": 27, "East of England": 18, "London": 24,
                "South East": 19, "South West": 19, "Wales": 21, "Scotland": 21,
                "Northern Ireland": 17},
        population="People (all ages)", measure="Relative poverty, AHC",
        period="'2021–2023' (HBAI 3-year average, FY 2020/21–2022/23)", source="Table 6, p.51",
        window=FY_2020_23, ukhls_unit="Household (gor_dv)",
        ukhls_definition="Region of household",
        notes="Table title reads 2021–2023; read as HBAI 3-year average ending 2022/23. "
              "JRF 'East' = East of England."),
    "ethnicity": dict(
        values={"White": 19, "Pakistani": 49, "Bangladeshi": 56, "Black African": 40,
                "Black Caribbean": 30, "Any other Asian background": 34},
        population="People in households, by ethnicity of household head",
        measure="Relative poverty, AHC", period="FY 2020/21–2022/23 (3-year average)",
        source="p.9 and p.42 (text); Figure 13, p.43", window=FY_2020_23,
        ukhls_unit="Household (ethnicity of household reference person)",
        ukhls_definition="ethnicity_group of HRP",
        notes="Only categories with a rate stated in JRF text are compared."),
    "tenure": dict(
        values={"Owned outright": 14, "Buying with mortgage": 10, "Social renting": 44,
                "Private renting": 35},
        population="People", measure="Relative poverty, AHC", period="FY 2022/23",
        source="Table 10, p.95", window=FY_2022_23, ukhls_unit="Household (tenure_dv)",
        ukhls_definition="Social = LA + housing association; private incl. rented from employer; "
                         "'Other' tenure excluded",
        notes=""),
    "disability": dict(
        values={"No disabled adult": 19, "Contains disabled adult": 29},
        population="People, by disability mix of family", measure="Relative poverty, AHC",
        period="FY 2022/23", source="Table 8, p.67", window=FY_2022_23,
        ukhls_unit="Household with >=1 adult disability status observed",
        ukhls_definition="Disabled = health==1 and any disdif1-12; household contains a disabled "
                         "adult if any observed adult is disabled",
        notes="JRF 'Disabled adults only' (29) vs 'No one is disabled' (19). UKHLS does not observe "
              "child disability, so JRF's child rows (28, 36) are not compared. Directional (n=2)."),
    "family_type": dict(
        values={"Lone parent": 44, "Couple with children": 25},
        population="CHILDREN, by family type", measure="Child relative poverty, AHC",
        period="FY 2022/23", source="Table 5, p.36", window=FY_2022_23,
        ukhls_unit="Household with dependent children",
        ukhls_definition="family_composition_group lone parent (any size) vs couple (any size); "
                         "other multi-adult excluded",
        notes="Unit mismatch: JRF rate is per child, UKHLS rate per household. Directional (n=2)."),
    "work_status": dict(
        values={"Not in work": 54, "In work": 15},
        population="WORKING-AGE ADULTS, by household work status",
        measure="Relative poverty, AHC", period="FY 2022/23 (latest year in report)",
        source="p.77 (text)", window=FY_2022_23,
        ukhls_unit="Household with >=1 respondent aged 16-64",
        ukhls_definition="Workless = no responding adult in paid/self-employment",
        notes="Corrected from v1 (12/43). Unit mismatch: JRF per working-age adult, UKHLS per "
              "household. Directional (n=2)."),
}


def working_age_households() -> pd.DataFrame:
    frames = []
    for w in "abcdefghijklmno":
        d = pd.read_stata(UKHLS_RAW_DIR / f"{w}_indresp.dta", columns=[f"{w}_hidp", f"{w}_dvage"],
                          convert_categoricals=False)
        d.columns = ["hidp", "dvage"]
        g = d.assign(wa=d.dvage.between(16, 64)).groupby("hidp").wa.any().reset_index()
        g["wave"] = w
        frames.append(g)
    return pd.concat(frames).rename(columns={"wa": "has_working_age_adult"})


def load() -> pd.DataFrame:
    cols = ["hidp", "wave", "interview_year", "interview_month", "gor_dv", "tenure_dv",
            "family_composition_group", "employment_group", "ethnicity_group", "disability_free",
            "hh_xw", "fuelhave3", "high_fuel_vulnerable", "high_fuel_vulnerable_s1"]
    df = pd.read_csv(UKHLS_PANEL, usecols=cols)
    df = df.merge(working_age_households(), on=["hidp", "wave"], how="left", validate="1:1")
    df["ym"] = pd.to_datetime(dict(year=df.interview_year, month=df.interview_month, day=1),
                              errors="coerce").dt.strftime("%Y-%m")
    df["region"] = df.gor_dv.map(GOR)
    df["ethnicity"] = df.ethnicity_group
    df["tenure"] = df.tenure_dv.map(TENURE)
    df["disability"] = np.where(df.disability_free.isna(), None,
                                np.where(df.disability_free < 1, "Contains disabled adult",
                                         "No disabled adult"))
    fam = df.family_composition_group.fillna("")
    df["family_type"] = np.select([fam.str.startswith("Lone parent"), fam.str.startswith("Couple")],
                                  ["Lone parent", "Couple with children"], None)
    df["work_status"] = np.where(~df.has_working_age_adult.astype("boolean").fillna(False) | df.employment_group.isna(), None,
                                 np.where(df.employment_group == "Workless household",
                                          "Not in work", "In work"))
    return df


def window_rate(g: pd.DataFrame, flag: str) -> tuple[float, float, int]:
    g = g[g[flag].notna()]
    n = len(g)
    if n == 0:
        return np.nan, np.nan, 0
    per_wave, sizes = [], []
    for _, gw in g.groupby("wave"):
        ok = gw.hh_xw > 0
        if ok.any():
            per_wave.append(np.average(gw.loc[ok, flag].astype(float), weights=gw.loc[ok, "hh_xw"]))
            sizes.append(len(gw))
    weighted = float(100 * np.average(per_wave, weights=sizes)) if per_wave else np.nan
    return weighted, float(100 * g[flag].mean()), n


def compare(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    meta, rows, agree = [], [], []
    for dim, spec in JRF.items():
        lo, hi = spec["window"]
        win = df[(df.ym >= lo) & (df.ym <= hi)]
        for cat, jrf_val in spec["values"].items():
            g = win[win[dim] == cat]
            w, u, n = window_rate(g, "high_fuel_vulnerable")
            s1w, _, _ = window_rate(g, "high_fuel_vulnerable_s1")
            masked = n < MIN_CELL_N
            rows.append(dict(dimension=dim, category=cat, jrf_poverty_rate_pct=jrf_val,
                             ukhls_n=n, fuel_vuln_pct_weighted=np.nan if masked else w,
                             fuel_vuln_pct_unweighted=np.nan if masked else u,
                             fuel_vuln_s1_pct_weighted=np.nan if masked else s1w,
                             waves_in_window=",".join(sorted(g.wave.unique())), masked=masked))
        meta.append(dict(dimension=dim, jrf_population=spec["population"], jrf_measure=spec["measure"],
                         jrf_period=spec["period"], jrf_source=spec["source"],
                         ukhls_window=f"interviews {lo} to {hi}", ukhls_unit=spec["ukhls_unit"],
                         ukhls_definition=spec["ukhls_definition"],
                         ukhls_waves=",".join(sorted(win.wave.unique())),
                         n_categories=len(spec["values"]), notes=spec["notes"]))
    comp = pd.DataFrame(rows)
    for dim, sub in comp.groupby("dimension", sort=False):
        for label, s in [("all", sub), ("excl. Northern Ireland", sub[sub.category != "Northern Ireland"])]:
            if label != "all" and dim != "region":
                continue
            s = s.dropna(subset=["fuel_vuln_pct_weighted"])
            k = len(s)
            rho = stats.spearmanr(s.jrf_poverty_rate_pct, s.fuel_vuln_pct_weighted)[0] if k > 2 else np.nan
            r = stats.pearsonr(s.jrf_poverty_rate_pct, s.fuel_vuln_pct_weighted)[0] if k > 2 else np.nan
            same_order = (np.sign(np.diff(s.jrf_poverty_rate_pct.values))
                          == np.sign(np.diff(s.fuel_vuln_pct_weighted.values))).all() if k == 2 else np.nan
            agree.append(dict(dimension=dim, subset=label, n_categories=k, spearman_rho=rho,
                              pearson_r=r, two_group_same_direction=same_order))
    return pd.DataFrame(meta), comp, pd.DataFrame(agree)


def load_psu() -> pd.DataFrame:
    frames = []
    for w in "abcdefghijklmno":
        d = pd.read_stata(UKHLS_RAW_DIR / f"{w}_hhresp.dta", columns=[f"{w}_hidp", f"{w}_psu"],
                          convert_categoricals=False)
        d.columns = ["hidp", "psu"]
        frames.append(d.assign(wave=w))
    return pd.concat(frames)


def region_window_ci(df: pd.DataFrame, n_boot: int = 2000, seed: int = 20260926) -> pd.DataFrame:
    lo, hi = FY_2020_23
    regions = list(GOR.values())
    out = []
    for flag, label in [("high_fuel_vulnerable", "primary"), ("high_fuel_vulnerable_s1", "s1_lower_bound")]:
        win = df[(df.ym >= lo) & (df.ym <= hi) & df.region.notna() & df[flag].notna() & (df.hh_xw > 0)]
        win = win.merge(load_psu(), on=["hidp", "wave"], how="left", validate="1:1")
        cell = win.assign(wy=win.hh_xw * win[flag], one=1).groupby(["psu", "region", "wave"])[
            ["wy", "hh_xw", "one"]].sum()
        psus = cell.index.get_level_values("psu").unique()
        cols = pd.MultiIndex.from_product([regions, sorted(win.wave.unique())])
        WY = cell.wy.unstack(["region", "wave"]).reindex(index=psus, columns=cols, fill_value=0).fillna(0).values
        W = cell.hh_xw.unstack(["region", "wave"]).reindex(index=psus, columns=cols, fill_value=0).fillna(0).values
        N = cell.one.unstack(["region", "wave"]).reindex(index=psus, columns=cols, fill_value=0).fillna(0).values
        n_w = len(cols.levels[1])

        def estimate(mult: np.ndarray) -> np.ndarray:
            wy, w, n = mult @ WY, mult @ W, mult @ N
            rate = np.divide(wy, w, out=np.full_like(wy, np.nan), where=w > 0).reshape(len(regions), n_w)
            n = n.reshape(len(regions), n_w)
            return 100 * np.nansum(rate * n, axis=1) / n.sum(axis=1)

        point = estimate(np.ones(len(psus)))
        rng = np.random.default_rng(seed)
        boots = np.array([estimate(rng.multinomial(len(psus), np.full(len(psus), 1 / len(psus))))
                          for _ in range(n_boot)])
        ranks = (-boots).argsort(axis=1).argsort(axis=1) + 1
        point_rank = (-point).argsort().argsort() + 1
        for i, r in enumerate(regions):
            out.append(dict(outcome=label, region=r, n=int(win[win.region == r].shape[0]),
                            n_psu=int((N.reshape(len(psus), len(regions), n_w)[:, i, :].sum(axis=1) > 0).sum()),
                            pct_weighted=point[i],
                            ci95_low=np.percentile(boots[:, i], 2.5), ci95_high=np.percentile(boots[:, i], 97.5),
                            rank=int(point_rank[i]),
                            rank_ci95_low=int(np.percentile(ranks[:, i], 2.5)),
                            rank_ci95_high=int(np.percentile(ranks[:, i], 97.5)),
                            p_rank1=float((ranks[:, i] == 1).mean()),
                            n_boot=n_boot, psu_resampled=len(psus)))
    return pd.DataFrame(out)


def ni_oil(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ni = df[df.region == "Northern Ireland"]
    for label, sub in [("all waves (wave-mean)", ni),
                       ("FY 2020/21-2022/23 window", ni[(ni.ym >= FY_2020_23[0]) & (ni.ym <= FY_2020_23[1])])]:
        valid = sub[sub.fuelhave3.notna()]
        share_u = 100 * (valid.fuelhave3 == 1).mean()
        share_w = np.average([np.average(gw.fuelhave3 == 1, weights=gw.hh_xw)
                              for _, gw in valid[valid.hh_xw > 0].groupby("wave")]) * 100
        rec = dict(sample=label, n_households=len(valid), oil_share_pct_unweighted=share_u,
                   oil_share_pct_weighted=share_w)
        for grp, m in [("oil", valid.fuelhave3 == 1), ("non_oil", valid.fuelhave3 == 0)]:
            w, u, n = window_rate(valid[m], "high_fuel_vulnerable")
            rec.update({f"{grp}_n": n, f"{grp}_vuln_pct_weighted": w, f"{grp}_vuln_pct_unweighted": u})
        rows.append(rec)
    gb = df[df.fuelhave3.notna()]
    other = [dict(region=r, oil_share_pct_unweighted=100 * (g.fuelhave3 == 1).mean())
             for r, g in gb.groupby("region")]
    out = pd.DataFrame(rows)
    out.attrs["other_regions"] = pd.DataFrame(other)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load()
    meta, comp, agree = compare(df)
    meta.to_csv(OUT / "jrf_metadata.csv", index=False)
    comp.to_csv(OUT / "jrf_comparison.csv", index=False)
    agree.to_csv(OUT / "jrf_agreement.csv", index=False)
    ni = ni_oil(df)
    ni.to_csv(OUT / "ni_oil.csv", index=False)
    ni.attrs["other_regions"].to_csv(OUT / "oil_share_by_region.csv", index=False)
    ci = region_window_ci(df)
    ci.to_csv(OUT / "region_window_ci.csv", index=False)
    print(ci.sort_values(["outcome", "rank"]).round(2).to_string(index=False))

    pd.set_option("display.width", 220)
    print(meta[["dimension", "jrf_period", "ukhls_window", "ukhls_waves"]].to_string(index=False))
    print(comp.round(2).to_string(index=False))
    print(agree.round(3).to_string(index=False))
    print(ni.round(2).T.to_string())
    print(ni.attrs["other_regions"].round(1).to_string(index=False))


if __name__ == "__main__":
    main()
