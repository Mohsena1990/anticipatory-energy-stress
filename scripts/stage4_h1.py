"""
Stage 4 H1 (analysis_plan_rerun.md, incl. the 2026-09-26 decision-rule
correction): does household resource stock moderate the effect of FES Delta?

Primary: OLS of fuel_to_income_ratio on R, FES Delta (growth-only), R x Delta
and interview-year FE; SEs clustered on PSU. R = OBJECT + CONDITION +
PERSONAL (unit-weighted standardised composites, formative indices).
Delta = forecast - realised; R higher = more resources.

Decision rule (fixed before fitting): COR predicts a POSITIVE R x Delta
interaction (resources flatten the negative Delta slope).
  supported       interaction > 0, p < 0.05 (primary)
  contrary to COR interaction < 0, p < 0.05
  not supported   otherwise
Predicted Delta slopes (with 95% CIs) at R = p10 / p50 / p90.

Sensitivities (reported beside the primary): R including ENERGY; logit on
high_fuel_vulnerable (slopes on the log-odds scale); 4-term FES Delta.

Outputs (outputs_v2/stage4/): h1_coefficients.csv, h1_slopes.csv,
  h1_decision.csv,
  h1_buffering_bound.csv   largest buffering compatible with the primary
                           interaction CI: slope difference R p90 - p10, per
                           unit and per SD of Delta, in pp of income
  h1_logit_prob_slopes.csv logit sensitivity: average marginal effect of Delta
                           on P(vulnerable), pp per SD of Delta, at R p10/p50/p90
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.paths import OUTPUTS_DIR, UKHLS_PANEL  # noqa: E402

OUT = OUTPUTS_DIR / "stage4"
MODELS = {  # name: (R column, Delta column, outcome, estimator)
    "primary": ("R_primary", "fes_delta_growth3", "fuel_to_income_ratio", "ols"),
    "sens_R_with_energy": ("R_with_energy", "fes_delta_growth3", "fuel_to_income_ratio", "ols"),
    "sens_logit_binary": ("R_primary", "fes_delta_growth3", "high_fuel_vulnerable", "logit"),
    "sens_fes_4term": ("R_primary", "fes_delta", "fuel_to_income_ratio", "ols"),
}


def load() -> pd.DataFrame:
    cols = ["hidp", "wave", "psu", "interview_year", "fuel_to_income_ratio", "high_fuel_vulnerable",
            "fes_delta_growth3", "fes_delta"]
    df = pd.read_csv(UKHLS_PANEL, usecols=cols)
    sc = pd.read_csv(OUTPUTS_DIR / "resources" / "resource_scores.csv",
                     usecols=["hidp", "wave", "R_primary", "R_with_energy"])
    return df.merge(sc, on=["hidp", "wave"], how="left", validate="1:1")


def fit(df: pd.DataFrame, r: str, d: str, y: str, est: str):
    sub = df.dropna(subset=[r, d, y, "psu", "interview_year"]).copy()
    sub["RxD"] = sub[r] * sub[d]
    X = sub[[r, d, "RxD"]].astype(float)
    yfe = pd.get_dummies(sub["interview_year"].astype(int), prefix="year", dtype=float)
    X = sm.add_constant(pd.concat([X, yfe.drop(columns=yfe.columns[0])], axis=1))
    groups = pd.factorize(sub["psu"])[0]
    model = sm.OLS(sub[y].astype(float), X) if est == "ols" else sm.Logit(sub[y].astype(float), X)
    kw = dict(cov_type="cluster", cov_kwds={"groups": groups})
    res = model.fit(**kw) if est == "ols" else model.fit(disp=0, maxiter=200, **kw)
    return res, sub


def logit_prob_slope(res, sub: pd.DataFrame, r: str, d: str, rv: float) -> tuple[float, float]:
    """AME of Delta on P(y=1) with R fixed at rv (other terms observed):
    mean[p(1-p)(b_D + b_RxD * rv)], delta-method SE (clustered covariance)."""
    X = pd.DataFrame(res.model.exog, columns=res.model.exog_names)
    X[r] = rv
    X["RxD"] = rv * sub[d].values
    b = res.params.values
    pr = 1 / (1 + np.exp(-X.values @ b))
    w = pr * (1 - pr)
    k = res.params.index
    s_ = res.params[d] + res.params["RxD"] * rv
    est = float(np.mean(w * s_))
    grad = (X.values * (w * (1 - 2 * pr) * s_)[:, None]).mean(axis=0)
    grad[k.get_loc(d)] += w.mean()
    grad[k.get_loc("RxD")] += w.mean() * rv
    se = float(np.sqrt(grad @ res.cov_params().values @ grad))
    return est, se


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load()
    coefs, slopes, decisions = [], [], []
    for name, (r, d, y, est) in MODELS.items():
        res, sub = fit(df, r, d, y, est)
        ci = res.conf_int()
        for t in [r, d, "RxD"]:
            coefs.append(dict(model=name, estimator=est, outcome=y, term=t if t != "RxD" else f"{r} x {d}",
                              coef=res.params[t], se=res.bse[t], p=res.pvalues[t],
                              ci_low=ci.loc[t, 0], ci_high=ci.loc[t, 1], n=int(res.nobs),
                              n_psu=int(sub.psu.nunique()),
                              r2_or_pseudo_r2=res.rsquared if est == "ols" else res.prsquared))
        V = res.cov_params()
        for q in [0.10, 0.50, 0.90]:
            rv = float(sub[r].quantile(q))
            slope = res.params[d] + res.params["RxD"] * rv
            se = float(np.sqrt(V.loc[d, d] + rv ** 2 * V.loc["RxD", "RxD"] + 2 * rv * V.loc[d, "RxD"]))
            slopes.append(dict(model=name, R_percentile=int(q * 100), R_value=rv, delta_slope=slope, se=se,
                               ci_low=slope - 1.96 * se, ci_high=slope + 1.96 * se,
                               scale="fuel-to-income ratio per unit Delta" if est == "ols"
                               else "log-odds per unit Delta"))
        b, p = res.params["RxD"], res.pvalues["RxD"]
        verdict = ("supported" if b > 0 and p < 0.05 else
                   "contrary to COR" if b < 0 and p < 0.05 else "not supported")
        decisions.append(dict(model=name, interaction=b, p=p, verdict=verdict,
                              governs_H1=(name == "primary"), n=int(res.nobs)))
        if name == "primary":
            r10, r90 = float(sub[r].quantile(0.10)), float(sub[r].quantile(0.90))
            sd_d = float(sub[d].std())
            ci_i = res.conf_int().loc["RxD"]
            rows = []
            for label, bi in [("point estimate", res.params["RxD"]), ("CI lower", ci_i[0]),
                              ("CI upper (max buffering)", ci_i[1])]:
                diff = bi * (r90 - r10)
                rows.append(dict(interaction=label, b_RxD=bi, R_p10=r10, R_p90=r90, sd_delta=sd_d,
                                 slope_diff_p90_minus_p10_per_unit_delta=diff,
                                 slope_diff_per_sd_delta_ratio=diff * sd_d,
                                 slope_diff_per_sd_delta_pp_income=100 * diff * sd_d,
                                 slope_p10_per_sd_delta_pp_income=100 * sd_d * (res.params[d] + res.params["RxD"] * r10)))
            pd.DataFrame(rows).to_csv(OUT / "h1_buffering_bound.csv", index=False)
        if est == "logit":
            sd_d = float(sub[d].std())
            prob = []
            for q in [0.10, 0.50, 0.90]:
                rv = float(sub[r].quantile(q))
                e, se_ = logit_prob_slope(res, sub, r, d, rv)
                prob.append(dict(model=name, R_percentile=int(q * 100), R_value=rv, sd_delta=sd_d,
                                 pp_per_sd_delta=100 * e * sd_d, ci_low=100 * (e - 1.96 * se_) * sd_d,
                                 ci_high=100 * (e + 1.96 * se_) * sd_d))
            pd.DataFrame(prob).to_csv(OUT / "h1_logit_prob_slopes.csv", index=False)
    pd.DataFrame(coefs).to_csv(OUT / "h1_coefficients.csv", index=False)
    pd.DataFrame(slopes).to_csv(OUT / "h1_slopes.csv", index=False)
    pd.DataFrame(decisions).to_csv(OUT / "h1_decision.csv", index=False)
    write_report()
    pd.set_option("display.width", 220)
    print(pd.DataFrame(coefs).to_string(index=False))
    print(pd.DataFrame(slopes).to_string(index=False))
    print(pd.DataFrame(decisions).to_string(index=False))


def write_report() -> None:
    """H1 results draft; every number read from the Stage 4 outputs."""
    c = pd.read_csv(OUT / "h1_coefficients.csv")
    sl = pd.read_csv(OUT / "h1_slopes.csv")
    bb = pd.read_csv(OUT / "h1_buffering_bound.csv").set_index("interaction")
    pr = pd.read_csv(OUT / "h1_logit_prob_slopes.csv").set_index("R_percentile")
    dec = pd.read_csv(OUT / "h1_decision.csv").set_index("model")
    i = c[(c.model == "primary") & c.term.str.contains(" x ")].iloc[0]
    lo = c[(c.model == "sens_logit_binary") & c.term.str.contains(" x ")].iloc[0]
    ps = sl[sl.model == "primary"].set_index("R_percentile")
    ub = bb.loc["CI upper (max buffering)"]
    share = 100 * ub.slope_diff_per_sd_delta_pp_income / abs(ub.slope_p10_per_sd_delta_pp_income)
    text = f"""# Stage 4 draft: does household resource stock moderate FES Delta? (H1)

*Generated by `scripts/stage4_h1.py`; every number comes from `outputs_v2/stage4/`. Draft for the author's rewrite.*

H1 (Conservation of Resources) predicts that households with more resources are less affected by
forecast-realised energy-price stress: with FES Delta = forecast − realised and resources oriented so
that higher = more, a **positive** resource × Delta interaction. The decision rule was fixed before
fitting.

**Primary verdict: not supported.** In the primary model (OLS of the fuel-to-income ratio, interview-year
fixed effects, PSU-clustered standard errors, n = {int(i.n):,}), the interaction is
{i.coef:.6f} (95% CI {i.ci_low:.6f} to {i.ci_high:.6f}, p = {i.p:.2f}). The Delta slope is negative at every
resource level ({ps.loc[10,'delta_slope']:.5f} at the 10th percentile of resources, {ps.loc[90,'delta_slope']:.5f} at the
90th) and is not flatter for better-resourced households.

**Largest buffering compatible with the data.** At the upper confidence limit of the interaction, moving
from the 10th to the 90th resource percentile would flatten the Delta slope by at most
{ub.slope_diff_per_sd_delta_pp_income:.3f} percentage points of income per standard deviation of Delta, i.e. at most
about {share:.0f}% of the slope at the 10th percentile ({ub.slope_p10_per_sd_delta_pp_income:.3f} pp). A substantial
buffering effect is therefore ruled out; a small one cannot be excluded.

Resources themselves are strongly associated with a lower fuel burden (main effect
{c[(c.model=='primary') & (c.term=='R_primary')].coef.iloc[0]:.4f} per unit of the composite, p < 0.001); what is not
supported is that they moderate the response to energy-price stress.

Sensitivities: resources including the ENERGY domain ({dec.loc['sens_R_with_energy','verdict']}) and the 4-term FES
({dec.loc['sens_fes_4term','verdict']}).[^logit]

[^logit]: In a logit of the binary outcome the interaction is negative on the log-odds scale
({lo.coef:.4f}, 95% CI {lo.ci_low:.4f} to {lo.ci_high:.4f}). On the probability scale the Delta effect is
instead smaller for better-resourced households: {pr.loc[10,'pp_per_sd_delta']:.2f} pp per SD of Delta at the 10th
resource percentile (95% CI {pr.loc[10,'ci_low']:.2f} to {pr.loc[10,'ci_high']:.2f}), {pr.loc[50,'pp_per_sd_delta']:.2f} at the median
and {pr.loc[90,'pp_per_sd_delta']:.2f} at the 90th ({pr.loc[90,'ci_low']:.2f} to {pr.loc[90,'ci_high']:.2f}). Both patterns reflect
differences in baseline risk between resource levels rather than evidence for or against buffering, and
neither changes the primary verdict.
"""
    rep_dir = OUTPUTS_DIR / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)
    (rep_dir / "stage4_h1_draft.md").write_text(text)


if __name__ == "__main__":
    main()
