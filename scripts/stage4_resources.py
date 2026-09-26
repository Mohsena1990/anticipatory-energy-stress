"""
Stage 4 resources (analysis_plan_rerun.md): correlated four-factor
first-order CFA by complete-case ML. One attempt. If the pre-registered
fit criteria fail, household resources are measured by unit-weighted
standardised composites instead. No H1 interaction here (FES-dependent,
held until Stage 2 is approved).

Pre-registered fit criteria: CFI >= 0.90, RMSEA <= 0.08, SRMR <= 0.08, no
Heywood case (negative residual variance or |std loading| >= 1), every
standardised loading >= 0.30 with the expected (positive) sign.
Marker indicators (loading fixed to 1, orientation): OBJECT hsrooms,
CONDITION tenure_security, PERSONAL sf1_good, ENERGY fihhmnnet1_dv.

Outputs (outputs_v2/resources/):
  cfa_fit.csv, cfa_loadings.csv, cfa_factor_correlations.csv,
  cfa_sample_composition.csv, resource_decision.csv,
  resource_scores.csv (hidp, wave, domain scores, R_primary, R_with_energy),
  composite_item_correlations.csv, resource_by_region.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import semopy

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.paths import OUTPUTS_DIR, UKHLS_PANEL  # noqa: E402

OUT = OUTPUTS_DIR / "resources"
FACTORS = {  # marker first
    "OBJECT": ["hsrooms", "hsbeds", "ncars", "carval", "hsval"],
    "CONDITION": ["tenure_security", "jbstat_security", "bill_security"],
    "PERSONAL": ["sf1_good", "health_good", "qfhigh_band"],
    "ENERGY": ["fihhmnnet1_dv", "fiyrinvinc_dv"],
}
LABELS = {"health_good": "No long-standing illness/disability", "sf1_good": "Self-rated general health",
          "qfhigh_band": "Highest qualification band", "hsrooms": "Rooms", "hsbeds": "Bedrooms",
          "ncars": "Cars", "carval": "Car value (log)", "hsval": "House value (log)",
          "tenure_security": "Tenure security", "jbstat_security": "Employment-status security",
          "bill_security": "Bill-payment security", "fihhmnnet1_dv": "Net household income (log)",
          "fiyrinvinc_dv": "Investment income (log)"}
MONETARY = ["hsval", "carval", "fihhmnnet1_dv", "fiyrinvinc_dv"]
ITEMS = [i for v in FACTORS.values() for i in v]
GOR = {1: "North East", 2: "North West", 3: "Yorkshire and the Humber", 4: "East Midlands",
       5: "West Midlands", 6: "East of England", 7: "London", 8: "South East", 9: "South West",
       10: "Wales", 11: "Scotland", 12: "Northern Ireland"}
TENURE = {1: "Owned outright", 2: "Owned with mortgage", 3: "Social rent", 4: "Social rent",
          5: "Private rent", 6: "Private rent", 7: "Private rent", 8: "Other"}


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    x = df[ITEMS].astype(float).copy()
    for c in MONETARY:
        x[c] = np.log1p(x[c].clip(lower=0))
    return x


def srmr(model: semopy.Model, data: pd.DataFrame) -> float:
    obs = model.vars["observed"]
    s = data[obs].cov().values
    sigma = model.calc_sigma()[0]
    d = np.sqrt(np.diag(s))
    r_s, r_m = s / np.outer(d, d), sigma / np.outer(d, d)
    idx = np.tril_indices(len(obs))
    return float(np.sqrt(np.mean((r_s[idx] - r_m[idx]) ** 2)))


def fit_cfa(x: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame, bool, list[str]]:
    cc = x.dropna()
    z = (cc - cc.mean()) / cc.std()
    desc = "\n".join(f"{f} =~ " + " + ".join(items) for f, items in FACTORS.items())
    names = list(FACTORS)
    desc += "\n" + "\n".join(f"{a} ~~ {b}" for i, a in enumerate(names) for b in names[i + 1:])
    model = semopy.Model(desc)
    model.fit(z, obj="MLW")
    st = semopy.calc_stats(model).T.iloc[:, 0]
    est = model.inspect(std_est=True)
    loads = est[(est.op == "~") & est.rval.isin(names)].copy()
    loads = loads.rename(columns={"lval": "item", "rval": "factor"})[
        ["factor", "item", "Estimate", "Est. Std", "Std. Err", "p-value"]]
    loads["label"] = loads.item.map(LABELS)
    fcorr = est[(est.op == "~~") & est.lval.isin(names) & est.rval.isin(names) & (est.lval != est.rval)][
        ["lval", "rval", "Est. Std"]].rename(columns={"Est. Std": "factor_correlation"})
    resid = est[(est.op == "~~") & (est.lval == est.rval) & est.lval.isin(ITEMS)]
    fit = dict(n_complete_case=len(cc), chi2=float(st["chi2"]), dof=float(st["DoF"]),
               CFI=float(st["CFI"]), TLI=float(st["TLI"]), RMSEA=float(st["RMSEA"]),
               SRMR=srmr(model, z))
    fails = []
    if not fit["CFI"] >= 0.90: fails.append(f"CFI {fit['CFI']:.3f} < 0.90")
    if not fit["RMSEA"] <= 0.08: fails.append(f"RMSEA {fit['RMSEA']:.3f} > 0.08")
    if not fit["SRMR"] <= 0.08: fails.append(f"SRMR {fit['SRMR']:.3f} > 0.08")
    if (resid.Estimate.astype(float) < 0).any() or (loads["Est. Std"].astype(float).abs() >= 1).any():
        fails.append("Heywood case")
    weak = loads[loads["Est. Std"].astype(float) < 0.30]
    for r in weak.itertuples():
        fails.append(f"{r.factor}:{r.item} std loading {float(r._4):.2f} < 0.30")
    return fit, loads, fcorr, not fails, fails


def composites(x: pd.DataFrame) -> pd.DataFrame:
    z = (x - x.mean()) / x.std()
    out = pd.DataFrame(index=x.index)
    for f, items in FACTORS.items():
        obs = z[items].notna().sum(axis=1)
        score = z[items].mean(axis=1).where(obs >= len(items) / 2)
        out[f] = (score - score.mean()) / score.std()
    out["R_primary"] = out[["OBJECT", "CONDITION", "PERSONAL"]].sum(axis=1, min_count=3)
    out["R_with_energy"] = out[list(FACTORS)].sum(axis=1, min_count=4)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(UKHLS_PANEL, usecols=["hidp", "wave", "gor_dv", "tenure_dv", "hh_xw"] + ITEMS)
    x = prepare(df)

    fit, loads, fcorr, ok, fails = fit_cfa(x)
    pd.DataFrame([fit]).to_csv(OUT / "cfa_fit.csv", index=False)
    loads.to_csv(OUT / "cfa_loadings.csv", index=False)
    fcorr.to_csv(OUT / "cfa_factor_correlations.csv", index=False)
    cc = x.dropna().index
    comp = pd.DataFrame({
        "tenure_all": df.tenure_dv.map(TENURE).value_counts(normalize=True),
        "tenure_complete_case": df.loc[cc, "tenure_dv"].map(TENURE).value_counts(normalize=True)}).fillna(0)
    comp.to_csv(OUT / "cfa_sample_composition.csv")

    decision = "CFA factor scores" if ok else "unit-weighted standardised composites (CFA failed pre-registered criteria)"
    pd.DataFrame([dict(decision=decision, cfa_passes=ok, failures="; ".join(fails))]).to_csv(
        OUT / "resource_decision.csv", index=False)

    # The plan calls for composites only on CFA failure; they are computed
    # either way, but only used downstream when the CFA fails.
    scores = composites(x)
    scores.insert(0, "wave", df.wave)
    scores.insert(0, "hidp", df.hidp)
    scores.to_csv(OUT / "resource_scores.csv", index=False)
    scores[list(FACTORS) + ["R_primary", "R_with_energy"]].corr().to_csv(OUT / "composite_item_correlations.csv")

    reg = scores.assign(region=df.gor_dv.map(GOR), w=df.hh_xw).dropna(subset=["region", "R_primary"])
    reg = reg[reg.w > 0]
    (reg.groupby("region").apply(lambda g: pd.Series({
        "n": len(g), "R_primary_weighted_mean": np.average(g.R_primary, weights=g.w),
        **{f"{f}_weighted_mean": np.average(g[f].fillna(0), weights=g.w) for f in FACTORS}}))
        .reset_index().to_csv(OUT / "resource_by_region.csv", index=False))

    pd.set_option("display.width", 200)
    print(pd.Series(fit).round(4).to_string())
    print(loads.round(3).to_string(index=False))
    print(fcorr.round(3).to_string(index=False))
    print(comp.round(3).to_string())
    print("DECISION:", decision, "|", "; ".join(fails))
    print(scores[list(FACTORS) + ["R_primary", "R_with_energy"]].describe().T[["count", "mean", "std"]].round(3))


if __name__ == "__main__":
    main()
