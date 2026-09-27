"""
Stage 3 driver models (analysis_plan_rerun.md, incl. logged deviations).

Outcome: high_fuel_vulnerable (A1 primary; S1/S2 outcome sensitivities).
Primary specification (logit):
  v1 controls, with the strain composite replaced by its components entered
  separately (finnow = current financial difficulty, scghq1_dv = GHQ-12
  distress, finfut_risk = financial expectations; age always included),
  + bill_security, + FES = fes_delta_growth3 (growth-only, primary per
  deviation 2026-09-26), + interview-year fixed effects.
  SEs clustered on PSU.
Sensitivities: two-way clustering (PSU x interview year-month); strain
  composite without xphsdba; v1 composite with xphsdba; lagged components;
  lagged composites; + calendar-month FE; 4-term FES; S1 and S2 outcomes.
NI-oil sequence (each specification): region FE (ref South East) ->
  + oil (fuelhave3) -> + NI x oil; (b) and (c) also with rural
  (urban_dv, adjacent-wave filled); primary also with observed urban_dv only.
  Nested models share one estimation sample within each block.

FES models cover interviews 2010+ (no Dec-2008 origin); N is reported per
model. Households with any missing model variable are excluded
(complete case), counts in sample_flow.csv.

Outputs (outputs_v2/stage3/): coefficients.csv, model_summary.csv,
  year_fe.csv, ni_oil_sequence.csv, thesis_table_primary.csv,
  sample_flow.csv, figures/primary_or_forest.{png,pdf}
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.paths import OUTPUTS_DIR, UKHLS_PANEL  # noqa: E402

OUT = OUTPUTS_DIR / "stage3"
CONTROLS = ["tenure_security", "jbstat_security", "health_good", "sf1_good", "qfhigh_band",
            "dvage", "heatch", "hsbeds", "hsrooms", "ncars", "bill_security",
            "lone_parent", "large_family", "ieqmoecd_dv", "workless_household"]
STRAIN = ["finnow", "scghq1_dv", "finfut_risk"]
LABELS = {
    "finnow": "Current financial difficulty (1-5)",
    "scghq1_dv": "Psychological distress, GHQ-12 (0-36)",
    "finfut_risk": "Financial expectations: worse off (0-1)",
    "finnow_lag1": "Current financial difficulty, previous wave",
    "scghq1_dv_lag1": "GHQ-12 distress, previous wave",
    "finfut_risk_lag1": "Financial expectations, previous wave",
    "financial_strain_score": "Strain composite (no bill arrears)",
    "financial_strain_score_v1": "Strain composite, v1 (incl. bill arrears)",
    "financial_strain_score_lag1": "Strain composite, previous wave",
    "financial_strain_score_v1_lag1": "Strain composite v1, previous wave",
    "fes_delta_growth3": "FES Delta (growth-only)",
    "fes_delta": "FES Delta (4-term)",
    "tenure_security": "Tenure security", "jbstat_security": "Employment-status security",
    "health_good": "No long-standing illness/disability", "sf1_good": "Self-rated general health",
    "qfhigh_band": "Highest qualification band", "dvage": "Age (mean of adults)",
    "heatch": "Has central heating", "hsbeds": "Bedrooms", "hsrooms": "Rooms", "ncars": "Cars",
    "bill_security": "Bill-payment security", "lone_parent": "Lone-parent household",
    "large_family": "Large family (3+ children)", "ieqmoecd_dv": "OECD equivalence scale",
    "workless_household": "Workless household", "oil": "Uses heating oil",
    "ni_x_oil": "Northern Ireland x heating oil", "rural": "Rural location",
    "region_Northern Ireland": "Northern Ireland (vs South East)",
}
GOR = {1: "North East", 2: "North West", 3: "Yorkshire and the Humber", 4: "East Midlands",
       5: "West Midlands", 6: "East of England", 7: "London", 8: "South East", 9: "South West",
       10: "Wales", 11: "Scotland", 12: "Northern Ireland"}

SPECS = {  # name: (strain terms, fes term, outcome, extra FE)
    "primary": (STRAIN, "fes_delta_growth3", "high_fuel_vulnerable", []),
    "sens_composite": (["financial_strain_score"], "fes_delta_growth3", "high_fuel_vulnerable", []),
    "sens_composite_v1": (["financial_strain_score_v1"], "fes_delta_growth3", "high_fuel_vulnerable", []),
    "sens_lagged_components": ([f"{c}_lag1" for c in STRAIN], "fes_delta_growth3", "high_fuel_vulnerable", []),
    "sens_lagged_composite": (["financial_strain_score_lag1"], "fes_delta_growth3", "high_fuel_vulnerable", []),
    "sens_lagged_composite_v1": (["financial_strain_score_v1_lag1"], "fes_delta_growth3", "high_fuel_vulnerable", []),
    "sens_month_fe": (STRAIN, "fes_delta_growth3", "high_fuel_vulnerable", ["interview_month"]),
    "sens_fes_4term": (STRAIN, "fes_delta", "high_fuel_vulnerable", []),
    "sens_outcome_s1": (STRAIN, "fes_delta_growth3", "high_fuel_vulnerable_s1", []),
    "sens_outcome_s2": (STRAIN, "fes_delta_growth3", "high_fuel_vulnerable_s2", []),
    # qfhigh_band is the only missing control for 61% of the complete-case loss
    # (author decision 2026-09-26): primary specification without it.
    "sens_no_qualification": (STRAIN, "fes_delta_growth3", "high_fuel_vulnerable", []),
}
DROP_CONTROLS = {"sens_no_qualification": ["qfhigh_band"]}


def load() -> pd.DataFrame:
    df = pd.read_csv(UKHLS_PANEL, low_memory=False)
    df["region"] = df["gor_dv"].map(GOR)
    df["oil"] = (df["fuelhave3"] == 1).astype(float).where(df["fuelhave3"].notna())
    df["ni"] = (df["region"] == "Northern Ireland").astype(float).where(df["region"].notna())
    df["ni_x_oil"] = df["ni"] * df["oil"]
    df["rural_observed"] = (df["urban_dv"] == 2).astype(float).where(df["urban_dv"].notna())
    df["year_month"] = df["interview_year"] * 100 + df["interview_month"]
    return df


def design(df: pd.DataFrame, terms: list[str], fe: list[str]) -> pd.DataFrame:
    X = df[terms].astype(float)
    for f in fe:
        ref = {"interview_year": 2010, "region": "South East", "interview_month": 1}[f]
        d = pd.get_dummies(df[f], prefix=f, dtype=float)
        d = d.drop(columns=[f"{f}_{ref}" if f != "interview_year" else f"{f}_{float(ref)}"], errors="ignore")
        if f == "interview_year":
            d = d.drop(columns=[c for c in d.columns if c.endswith("_2010.0")], errors="ignore")
        X = pd.concat([X, d], axis=1)
    return sm.add_constant(X)


def fit(df: pd.DataFrame, y: str, terms: list[str], fe: list[str], two_way: bool = False):
    X = design(df, terms, fe)
    X = X.loc[:, X.std() > 0].assign(const=1.0)
    groups = (np.column_stack([pd.factorize(df["psu"])[0], pd.factorize(df["year_month"])[0]])
              if two_way else pd.factorize(df["psu"])[0])
    res = sm.Logit(df[y].astype(float), X).fit(disp=0, maxiter=300, method="newton",
                                              cov_type="cluster", cov_kwds={"groups": groups})
    return res


def ame(res, contrast: str) -> tuple[float, float]:
    """Average marginal effect (percentage points) with delta-method SE using
    the model's (clustered) covariance.
      contrast="ni":  P(NI) - P(South East), averaged over all sample rows; all
                      region dummies set to 0 in both arms, NI x oil = oil in
                      the NI arm and 0 otherwise.
      contrast="oil": P(oil) - P(no oil); NI x oil = NI in the oil arm."""
    X = pd.DataFrame(res.model.exog, columns=res.model.exog_names)
    X1, X0 = X.copy(), X.copy()
    if contrast == "ni":
        reg = [c for c in X if c.startswith("region_")]
        X1[reg] = 0.0
        X0[reg] = 0.0
        X1["region_Northern Ireland"] = 1.0
        if "ni_x_oil" in X:
            X1["ni_x_oil"] = X["oil"]
            X0["ni_x_oil"] = 0.0
    else:
        X1["oil"], X0["oil"] = 1.0, 0.0
        if "ni_x_oil" in X:
            ni = X.get("region_Northern Ireland", 0.0)
            X1["ni_x_oil"], X0["ni_x_oil"] = ni, 0.0
    b = res.params.values
    p1 = 1 / (1 + np.exp(-X1.values @ b))
    p0 = 1 / (1 + np.exp(-X0.values @ b))
    est = float(np.mean(p1 - p0))
    grad = (X1.values * (p1 * (1 - p1))[:, None] - X0.values * (p0 * (1 - p0))[:, None]).mean(axis=0)
    se = float(np.sqrt(grad @ res.cov_params().values @ grad))
    return 100 * est, 100 * se


def tidy(res, model: str, spec: str, df: pd.DataFrame, cluster: str, y: str) -> tuple[pd.DataFrame, dict]:
    ci = res.conf_int()
    rows = []
    for t in res.params.index:
        if t == "const":
            continue
        sd = df[t].std() if t in df.columns and df[t].nunique() > 2 else np.nan
        b = res.params[t]
        rows.append(dict(spec=spec, model=model, term=t, label=LABELS.get(t, t), coef=b,
                         se=res.bse[t], z=res.tvalues[t], p=res.pvalues[t], OR=np.exp(b),
                         OR_ci_low=np.exp(ci.loc[t, 0]), OR_ci_high=np.exp(ci.loc[t, 1]),
                         sd_in_sample=sd, OR_per_sd=np.exp(b * sd) if pd.notna(sd) else np.nan,
                         n=int(res.nobs)))
    summ = dict(spec=spec, model=model, outcome=y, n=int(res.nobs), n_events=int(res.model.endog.sum()),
                n_psu=int(df["psu"].nunique()), interview_years=f"{int(df.interview_year.min())}-{int(df.interview_year.max())}",
                pseudo_r2_mcfadden=res.prsquared, llf=res.llf, aic=res.aic, cluster=cluster,
                converged=bool(res.mle_retvals.get("converged", True)))
    return pd.DataFrame(rows), summ


AMES: list[dict] = []


def run_spec(df, spec, strain, fes, y, extra_fe, coefs, summs, flows):
    terms = [c for c in CONTROLS if c not in DROP_CONTROLS.get(spec, [])] + strain + [fes]
    need = terms + [y, "psu", "interview_year", "year_month"] + extra_fe
    base = df.dropna(subset=need)
    flows.append(dict(spec=spec, step="complete case on model variables (FES => 2010+)", n=len(base)))
    fe = ["interview_year"] + extra_fe
    res = fit(base, y, terms, fe)
    c, s = tidy(res, "main", spec, base, "psu", y)
    coefs.append(c); summs.append(s)
    if spec == "primary":
        res2 = fit(base, y, terms, fe, two_way=True)
        c, s = tidy(res2, "main_twoway_cluster", spec, base, "psu x interview year-month", y)
        coefs.append(c); summs.append(s)
        yfe = res.params.filter(like="interview_year_")
        pd.DataFrame({"term": yfe.index, "coef": yfe.values, "OR": np.exp(yfe.values),
                      "p": res.pvalues[yfe.index].values}).to_csv(OUT / "year_fe.csv", index=False)

    # NI-oil sequence: common sample within each block.
    blocks = [("filled", "rural", True)]
    if spec == "primary":
        blocks.append(("observed_only", "rural_observed", False))
    for block, rural_col, with_no_urban in blocks:
        s0 = base.dropna(subset=["region", "oil", rural_col])
        flows.append(dict(spec=spec, step=f"NI-oil sample (rural {block})", n=len(s0)))
        sub = s0.rename(columns={rural_col: "rural_tmp"})
        seq = []
        if with_no_urban:
            seq += [("a_regionFE", [])]
        seq += [("b_oil", ["oil"]), ("b_oil_rural", ["oil", "rural_tmp"]),
                ("c_ni_x_oil", ["oil", "ni_x_oil"]), ("c_ni_x_oil_rural", ["oil", "ni_x_oil", "rural_tmp"])]
        for name, extra in seq:
            r = fit(sub, y, terms + extra, fe + ["region"])
            c, s = tidy(r, f"ni_{name}_{block}", spec, sub, "psu", y)
            for contrast in ["ni"] + (["oil"] if "oil" in extra else []):
                est, se = ame(r, contrast)
                AMES.append(dict(spec=spec, model=f"ni_{name}_{block}", n=int(r.nobs),
                                 contrast={"ni": "NI vs South East", "oil": "oil vs no oil"}[contrast],
                                 ame_pp=est, se_pp=se, ci_low_pp=est - 1.96 * se, ci_high_pp=est + 1.96 * se))
            c["term"] = c["term"].replace({"rural_tmp": "rural"})
            c["label"] = c["term"].map(lambda t: LABELS.get(t, t))
            coefs.append(c); summs.append(s)


def forest(coef: pd.DataFrame) -> None:
    d = coef[(coef.spec == "primary") & (coef.model == "main")].copy()
    d = d[~d.term.str.startswith("interview_year_")].sort_values("OR_per_sd", na_position="first")
    d["x"] = d.OR_per_sd.fillna(d.OR)
    d["lo"] = np.where(d.OR_per_sd.notna(), np.exp(np.log(d.OR_ci_low) * d.sd_in_sample), d.OR_ci_low)
    d["hi"] = np.where(d.OR_per_sd.notna(), np.exp(np.log(d.OR_ci_high) * d.sd_in_sample), d.OR_ci_high)
    d = d.sort_values("x")
    fig, ax = plt.subplots(figsize=(7.5, 0.34 * len(d) + 1.4))
    yy = np.arange(len(d))
    ax.errorbar(d.x, yy, xerr=[d.x - d.lo, d.hi - d.x], fmt="o", color="#2E5077", ms=4, capsize=2, lw=1)
    ax.axvline(1, color="#888", lw=0.8, ls="--")
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{l}{' (per SD)' if pd.notna(s) else ''}" for l, s in zip(d.label, d.OR_per_sd)], fontsize=8)
    ax.set_xscale("log")
    ticks = [0.4, 0.5, 0.7, 1, 1.5, 2, 3]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("Odds ratio, 95% CI (PSU-clustered); continuous terms per SD")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    (OUT / "figures").mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / "figures" / f"primary_or_forest.{ext}", dpi=200)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load()
    flows = [dict(spec="all", step="all household-wave rows", n=len(df)),
             dict(spec="all", step="primary outcome observed", n=int(df.high_fuel_vulnerable.notna().sum())),
             dict(spec="all", step="outcome and FES (2010+)",
                  n=int((df.high_fuel_vulnerable.notna() & df.fes_delta_growth3.notna()).sum()))]
    coefs, summs = [], []
    for spec, (strain, fes, y, extra_fe) in SPECS.items():
        print("fitting", spec, flush=True)
        run_spec(df, spec, strain, fes, y, extra_fe, coefs, summs, flows)
    coef = pd.concat(coefs, ignore_index=True)
    coef_nofe = coef[~coef.term.str.startswith(("interview_year_", "interview_month_", "region_"))
                     | (coef.term == "region_Northern Ireland")]
    coef_nofe.to_csv(OUT / "coefficients.csv", index=False)
    coef[coef.term.str.startswith("region_")].to_csv(OUT / "region_fe.csv", index=False)
    summ = pd.DataFrame(summs)
    summ.to_csv(OUT / "model_summary.csv", index=False)
    pd.DataFrame(flows).to_csv(OUT / "sample_flow.csv", index=False)

    keep = ["region_Northern Ireland", "oil", "ni_x_oil", "rural"]
    ni = coef[coef.model.str.startswith("ni_") & coef.term.isin(keep)]
    ni_w = ni.pivot_table(index=["spec", "model", "n"], columns="term",
                          values=["OR", "OR_ci_low", "OR_ci_high", "p"]).reset_index()
    ni_w.columns = ["_".join([c for c in col if c]).strip("_") for col in ni_w.columns]
    ni_w.to_csv(OUT / "ni_oil_sequence.csv", index=False)

    pd.DataFrame(AMES).to_csv(OUT / "ni_oil_ame.csv", index=False)

    prim = coef_nofe[(coef_nofe.spec == "primary") & (coef_nofe.model == "main")]
    # Per-SD comparison: continuous predictors ranked by |log OR per SD|.
    cont = prim[prim.sd_in_sample.notna()].copy()
    cont["log_OR_per_sd"] = np.log(cont.OR_per_sd)
    cont["OR_per_sd_ci_low"] = np.exp(np.log(cont.OR_ci_low) * cont.sd_in_sample)
    cont["OR_per_sd_ci_high"] = np.exp(np.log(cont.OR_ci_high) * cont.sd_in_sample)
    cont["note"] = np.where(cont.term == "ieqmoecd_dv",
                            "partly mechanical: outcome uses unequivalised income, so larger households "
                            "have more income per fuel need", "")
    cont = cont.reindex(cont.log_OR_per_sd.abs().sort_values(ascending=False).index)
    cont[["label", "term", "sd_in_sample", "OR_per_sd", "OR_per_sd_ci_low", "OR_per_sd_ci_high",
          "log_OR_per_sd", "p", "note", "n"]].to_csv(OUT / "thesis_table_per_sd.csv", index=False)
    prim[["label", "term", "OR", "OR_ci_low", "OR_ci_high", "p", "OR_per_sd", "sd_in_sample", "n"]] \
        .sort_values("OR_per_sd", ascending=False).to_csv(OUT / "thesis_table_primary.csv", index=False)
    forest(coef)

    pd.set_option("display.width", 250)
    print(summ[["spec", "model", "n", "n_events", "pseudo_r2_mcfadden", "converged"]].round(4).to_string(index=False))
    print(prim[["label", "OR", "OR_ci_low", "OR_ci_high", "p", "OR_per_sd"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
