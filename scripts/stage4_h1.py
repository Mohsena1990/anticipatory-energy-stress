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
  h1_decision.csv
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
    pd.DataFrame(coefs).to_csv(OUT / "h1_coefficients.csv", index=False)
    pd.DataFrame(slopes).to_csv(OUT / "h1_slopes.csv", index=False)
    pd.DataFrame(decisions).to_csv(OUT / "h1_decision.csv", index=False)
    pd.set_option("display.width", 220)
    print(pd.DataFrame(coefs).to_string(index=False))
    print(pd.DataFrame(slopes).to_string(index=False))
    print(pd.DataFrame(decisions).to_string(index=False))


if __name__ == "__main__":
    main()
