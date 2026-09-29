"""
Stage 8, steps 2-5 (analysis_plan_rerun.md, amendment A7; EXPLORATORY, post hoc).

Anticipatory burden projection. For each linked transition t -> t+1 (A6 linking
and split), the household's burden at t+1 is projected with information
available at its wave-t interview month T:

  2. fuel-mix exposure: spend at t split into gas / electricity / oil / other;
     price multiplier M = sum_f share_f * (mean P_f[T..T+11] / mean P_f[T-12..T-1])
     from the step-1 forecasts (scripts/stage8_price_forecast.py);
  3. income projection: labour income x latest known AWE growth, benefit and
     pension income x the statutory CPI uprating known at T, other unchanged;
     sensitivity S-support adds the Energy Bills Support Scheme and the
     pensioner cost-of-living payment after their announcement;
  4. P_struct = P(b_{t+1} >= 10%) combining training household noise with the
     price-forecast uncertainty known at T;
  5. logistic models (training-only standardisation, one common sample), A6
     metrics on all validation transitions and on the incident subset (not
     vulnerable at t), calibration by year, crisis subset.

Outputs (outputs_v3/anticipation/, aggregate tables only): sample_flow.csv,
metrics.csv, delta_auc.csv, calibration.csv, prevalence_by_year.csv,
coefficients.csv, decision.csv, figures/prevalence_by_year.{png,pdf}
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import average_precision_score, roc_auc_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src.paths import UKHLS_PANEL  # noqa: E402
from stage4_resources import FACTORS  # noqa: E402
from stage6_prediction import (ITEMS, P0, P1, VALIDATION, WAVES, build_composites,  # noqa: E402
                               calibration, fit_predict, standardise, topk)
from stage8_price_forecast import EXT, OUT, read_ons  # noqa: E402

RAW_UKHLS = ROOT / "data" / "raw" / "ukhls"
N_BOOT = 2000
SEED = 20260929
THRESH = 0.10
FLOOR = 1e-3
FUELS = ["gas", "electricity", "liquid_fuels", "solid_fuels"]
INCOME_EXTRA = ["fihhmnsben_dv", "fihhmnpen_dv", "fihhmnlabnet_dv"]
SPEND_COLS = ["xpduely", "xpgasy", "xpelecy", "xpoily", "xpsfly", "fuelduel", "fuelduel_nr",
              "fuelhave1", "fuelhave2", "fuelhave3", "fuelhave4", "total_fuel_spend", "fihhmnnet1_dv"]
CRISIS = (pd.Timestamp("2021-01-01"), pd.Timestamp("2022-06-01"))

# Energy Bills Support Scheme: GBP 400 over Oct 2022-Mar 2023, announced 26 May 2022.
EBSS = dict(known_from="2022-06-01", first="2022-10-01", last="2023-03-01", amount=400.0)
# Pensioner cost-of-living payment (GBP 300, HRP aged >= 66): (known from, paid in month)
PENSIONER_PAYMENTS = [("2022-06-01", "2022-11-01"), ("2022-12-01", "2023-11-01")]


# ── data ─────────────────────────────────────────────────────────────────────

def income_components() -> pd.DataFrame:
    frames = []
    for w in WAVES:
        cols = [f"{w}_hidp"] + [f"{w}_{c}" for c in INCOME_EXTRA]
        d = pd.read_stata(RAW_UKHLS / f"{w}_hhresp.dta", columns=cols, convert_categoricals=False)
        d.columns = ["hidp"] + INCOME_EXTRA
        d["wave"] = w
        frames.append(d)
    out = pd.concat(frames, ignore_index=True)
    # Benefit and pension income cannot be negative, so any negative value is a
    # missing code. Net labour income can be (self-employment losses): only the
    # UKHLS missing codes are cleared there.
    for c in ("fihhmnsben_dv", "fihhmnpen_dv"):
        out.loc[out[c] < 0, c] = np.nan
    out.loc[out.fihhmnlabnet_dv.isin([-1, -2, -7, -8, -9, -10, -11, -20, -21]), "fihhmnlabnet_dv"] = np.nan
    return out


def load_panel() -> pd.DataFrame:
    cols = (["hidp", "wave", "hrpid", "psu", "interview_year", "interview_month", "dvage",
             "fuel_to_income_ratio", "high_fuel_vulnerable"] + SPEND_COLS
            + [c for c in P1 if c not in FACTORS] + ITEMS)
    df = pd.read_csv(UKHLS_PANEL, usecols=lambda c: c in set(cols))
    df = df.merge(income_components(), on=["hidp", "wave"], how="left")
    df["T"] = pd.to_datetime(dict(year=df.interview_year, month=df.interview_month, day=1), errors="coerce")
    return df


def fuel_split(df: pd.DataFrame) -> pd.DataFrame:
    """Spend at t by fuel, following the A1 routing; combined dual-fuel bills are
    split by the wave median gas share among separate-bill households."""
    elec, gas = df.fuelhave1 == 1, df.fuelhave2 == 1
    both = elec & gas
    duel_nr = df.fuelduel_nr.fillna(False).astype(bool)
    sep = both & ((df.fuelduel == 2) | duel_nr) & (df.xpgasy + df.xpelecy > 0)
    share = (df.xpgasy / (df.xpgasy + df.xpelecy)).where(sep)
    wave_share = share.groupby(df.wave).transform("median")
    parts = pd.DataFrame(0.0, index=df.index, columns=FUELS)
    comb = both & (df.fuelduel == 1)
    parts.loc[comb, "gas"] = df.xpduely * wave_share
    parts.loc[comb, "electricity"] = df.xpduely * (1 - wave_share)
    s2 = both & ((df.fuelduel == 2) | duel_nr)
    parts.loc[s2, "gas"] = df.xpgasy
    parts.loc[s2, "electricity"] = df.xpelecy
    parts.loc[elec & ~gas, "electricity"] = df.xpelecy
    parts.loc[gas & ~elec, "gas"] = df.xpgasy
    parts.loc[df.fuelhave3 == 1, "liquid_fuels"] = df.xpoily
    parts.loc[df.fuelhave4 == 1, "solid_fuels"] = df.xpsfly
    tot = parts.sum(axis=1)
    shares = parts.div(tot.where(tot > 0), axis=0)
    shares[tot <= 0] = np.nan
    return shares.add_prefix("share_")


# ── projection ───────────────────────────────────────────────────────────────

def macro_by_origin(origins: pd.DatetimeIndex) -> pd.DataFrame:
    """Earnings growth and benefit uprating known at each origin T."""
    cpi = read_ons(EXT / "cpi_all_items_d7bt.csv")
    awe = read_ons(EXT / "awe_total_pay_kab9.csv")
    cpi_yoy = (cpi / cpi.shift(12) - 1).dropna()
    mo = pd.DateOffset(months=1)
    rows = []
    for T in origins:
        a_last = T - 3 * mo
        g = awe.get(a_last, np.nan) / awe.get(a_last - 12 * mo, np.nan) - 1
        april_year = T.year if T.month <= 4 else T.year + 1
        sept = pd.Timestamp(f"{april_year - 1}-09-01")
        known = cpi_yoy.loc[:T - 2 * mo]
        u = cpi_yoy[sept] if sept <= T - 2 * mo and sept in cpi_yoy.index else (known.iloc[-1] if len(known) else np.nan)
        rows.append(dict(T=T, earn_growth=g, uprating=u))
    return pd.DataFrame(rows)


def attach_price_ratios(df: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    wide = prices.pivot(index="origin", columns="series",
                        values=["ratio_forecast", "ratio_naive", "ratio_realised", "sigma_ratio"])
    wide.columns = [f"{v}_{s}" for v, s in wide.columns]
    return df.merge(wide, left_on="T", right_index=True, how="left")


def overlap_months(T: pd.Series, first: str, last: str) -> pd.Series:
    a, b = pd.Timestamp(first), pd.Timestamp(last)
    start = T.where(T > a, a)
    end_w = T + pd.DateOffset(months=11)
    end = end_w.where(end_w < b, b)
    n = (end.dt.year - start.dt.year) * 12 + end.dt.month - start.dt.month + 1
    return n.clip(lower=0)


def project(df: pd.DataFrame, support: bool) -> pd.DataFrame:
    shares = df[[f"share_{f}" for f in FUELS]].values
    out = {}
    for v in ("forecast", "naive", "realised"):
        R = np.exp(df[[f"ratio_{v}_{f}" for f in FUELS]].values)
        M = np.nansum(shares * R, axis=1)
        # no spend split, or a fuel the household uses has no ratio -> missing
        M[np.isnan(shares).all(axis=1) | (np.isnan(R) & (shares > 0)).any(axis=1)] = np.nan
        out[v] = M
    lab = df.fihhmnlabnet_dv.fillna(0)
    ben = df.fihhmnsben_dv.fillna(0) + df.fihhmnpen_dv.fillna(0)
    other = df.fihhmnnet1_dv - lab - ben
    inc = 12 * (lab * (1 + df.earn_growth) + ben * (1 + df.uprating) + other)
    spend = df.total_fuel_spend
    extra_inc = pd.Series(0.0, index=df.index)
    spend_cut = pd.Series(0.0, index=df.index)
    if support:
        known = df["T"] >= pd.Timestamp(EBSS["known_from"])
        spend_cut = EBSS["amount"] * overlap_months(df["T"], EBSS["first"], EBSS["last"]) / 6 * known
        for known_from, paid in PENSIONER_PAYMENTS:
            p = pd.Timestamp(paid)
            in_w = (df["T"] <= p) & (df["T"] + pd.DateOffset(months=11) >= p) & (df["T"] >= pd.Timestamp(known_from))
            extra_inc += 300.0 * (in_w & (df.dvage >= 66))
    res = pd.DataFrame(index=df.index)
    denom = inc + extra_inc
    for v, M in out.items():
        b = ((spend * M - spend_cut).clip(lower=0)) / denom
        b[denom < 1200] = np.nan
        res[f"burden_{v}"] = b.clip(upper=1.0)
    for v, M in out.items():
        res[f"multiplier_{v}"] = M
    res["income_ratio"] = denom / (12 * df.fihhmnnet1_dv)
    sig = df[[f"sigma_ratio_{f}" for f in FUELS]].values
    res["sigma_hh"] = np.nansum(np.nan_to_num(shares) * sig, axis=1)
    return res


def p_struct(log_b: np.ndarray, sigma: np.ndarray, resid: np.ndarray) -> np.ndarray:
    """mean_j Phi((log b + e_j - log 0.1) / sigma); sigma floored to keep it finite."""
    s = np.maximum(sigma, 1e-3)[:, None]
    z = (log_b[:, None] + resid[None, :] - np.log(THRESH)) / s
    return norm.cdf(z).mean(axis=1)


# ── transitions and models ───────────────────────────────────────────────────

def transitions(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    flow, pairs = [], []
    for a, b in zip(WAVES[:-1], WAVES[1:]):
        t = df[df.wave == a].dropna(subset=["hrpid"]).drop_duplicates("hrpid", keep=False)
        t1 = df[df.wave == b].dropna(subset=["hrpid"]).drop_duplicates("hrpid", keep=False)
        m = t.merge(t1[["hrpid", "high_fuel_vulnerable", "fuel_to_income_ratio", "interview_year"]]
                    .rename(columns={"high_fuel_vulnerable": "y", "fuel_to_income_ratio": "burden_t1",
                                     "interview_year": "interview_year_t1"}), on="hrpid")
        m["transition"] = f"{a}->{b}"
        pairs.append(m)
        flow.append(dict(step=f"linked transitions {a}->{b}", n=len(m)))
    return pd.concat(pairs, ignore_index=True), flow


def logb(x: pd.Series) -> pd.Series:
    return np.log(x.clip(lower=FLOOR))


def bootstrap(va: pd.DataFrame, preds: dict, contrasts: list[tuple[str, str]]):
    y = va.y.values.astype(int)
    inc = (va.high_fuel_vulnerable.values == 0)
    crisis = va.crisis.values
    subsets = {"all": np.ones(len(y), bool), "incident": inc, "crisis": crisis, "crisis_incident": crisis & inc}
    codes, uniq = pd.factorize(va.psu)
    rng = np.random.default_rng(SEED)
    auc = {(m, s): [] for m in preds for s in subsets}
    for _ in range(N_BOOT):
        w = rng.multinomial(len(uniq), np.full(len(uniq), 1 / len(uniq)))[codes]
        for s, mask in subsets.items():
            k = mask & (w > 0)
            for m, p in preds.items():
                auc[(m, s)].append(roc_auc_score(y[k], p[k], sample_weight=w[k]))
    rows, drows = [], []
    for s, mask in subsets.items():
        for m, p in preds.items():
            a = np.array(auc[(m, s)])
            rows.append(dict(model=m, subset=s, n=int(mask.sum()), prevalence=y[mask].mean(),
                             auc=roc_auc_score(y[mask], p[mask]), auc_ci_low=np.percentile(a, 2.5),
                             auc_ci_high=np.percentile(a, 97.5),
                             prauc=average_precision_score(y[mask], p[mask]),
                             sens10=topk(y[mask], p[mask], 0.10)[0], ppv10=topk(y[mask], p[mask], 0.10)[1]))
        for m1, m2 in contrasts:
            d = np.array(auc[(m1, s)]) - np.array(auc[(m2, s)])
            drows.append(dict(contrast=f"{m1} - {m2}", subset=s,
                              delta_auc=roc_auc_score(y[mask], preds[m1][mask]) - roc_auc_score(y[mask], preds[m2][mask]),
                              ci_low=np.percentile(d, 2.5), ci_high=np.percentile(d, 97.5),
                              share_boot_gt0=(d > 0).mean()))
    return pd.DataFrame(rows), pd.DataFrame(drows)


def build_pairs(price_file: Path | None = None) -> tuple[pd.DataFrame, list[dict]]:
    """Linked transitions with the A7 projections attached (steps 2-3).
    `price_file` swaps in another set of step-1 window ratios (A9 cap forecasts)."""
    prices = pd.read_csv(price_file or OUT / "price_forecasts.csv", parse_dates=["origin"])
    df = load_panel()
    df = pd.concat([df, fuel_split(df)], axis=1)
    macro = macro_by_origin(pd.DatetimeIndex(df["T"].dropna().unique()))
    df = df.merge(macro, on="T", how="left")
    df = attach_price_ratios(df, prices)
    proj = project(df, support=False)
    proj_s = project(df, support=True).add_suffix("_support")
    df = pd.concat([df, proj, proj_s], axis=1)

    pairs, flow = transitions(df)
    pairs["crisis"] = pairs["T"].between(*CRISIS)
    for v in ("forecast", "naive", "realised"):
        pairs[f"logb_{v}"] = logb(pairs[f"burden_{v}"])
    pairs["logb_forecast_support"] = logb(pairs["burden_forecast_support"])
    pairs["logb_t"] = logb(pairs.fuel_to_income_ratio)
    return pairs, flow


def main() -> None:
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    pairs, flow = build_pairs()

    tr_all = pairs[~pairs.transition.isin(VALIDATION)].copy()
    va_all = pairs[pairs.transition.isin(VALIDATION)].copy()
    build_composites(tr_all, va_all)
    allp = pd.concat([tr_all, va_all])
    need = (["y", "psu"] + P0 + P1 + ["logb_forecast", "logb_naive", "logb_realised",
                                      "logb_forecast_support", "sigma_hh"])
    miss = allp[need].isna()
    for c in need:
        if miss[c].any():
            flow.append(dict(step=f"  missing {c}", n=int(miss[c].sum())))
    cs = ~miss.any(axis=1)
    tr, va = tr_all[cs.loc[tr_all.index]].copy(), va_all[cs.loc[va_all.index]].copy()
    flow += [dict(step="common sample", n=int(cs.sum())), dict(step="  training", n=len(tr)),
             dict(step="  validation", n=len(va)),
             dict(step="  validation incident (not vulnerable at t)", n=int((va.high_fuel_vulnerable == 0).sum())),
             dict(step="  validation crisis (t interview Jan 2021-Jun 2022)", n=int(va.crisis.sum()))]

    # step 4: structural probability (no fitting on the outcome except the noise residuals)
    resid = (logb(tr.burden_t1) - tr.logb_realised).dropna().values
    rng = np.random.default_rng(SEED)
    resid = rng.choice(resid, size=min(3000, len(resid)), replace=False)
    va["p_struct"] = p_struct(va.logb_forecast.values, va.sigma_hh.values, resid)
    va["p_struct_naive"] = p_struct(va.logb_naive.values, va.sigma_hh.values, resid)

    FLAG = ["high_fuel_vulnerable"]
    specs = {
        "P0 (A6 benchmark)": P0,
        "P1 (A6 social)": P1,
        "P3 = P0 + P1": P0 + P1,
        "A_naive": ["logb_naive"] + FLAG,
        "A_fut": ["logb_forecast"] + FLAG,
        "A_fut_support": ["logb_forecast_support"] + FLAG,
        "A_full = A_fut + P1": ["logb_forecast"] + FLAG + P1,
        "A_naive_full = A_naive + P1": ["logb_naive"] + FLAG + P1,
        "A_oracle (realised prices; bound)": ["logb_realised"] + FLAG,
    }
    cont = sorted({c for v in specs.values() for c in v})
    standardise(tr, va, cont)
    preds, coefs, cal = {}, [], []
    for name, cols in specs.items():
        res, p = fit_predict(tr, va, cols)
        preds[name] = p
        ci = res.conf_int()
        for c in res.params.index:
            coefs.append(dict(model=name, term=c, coef=res.params[c], ci_low=ci.loc[c, 0],
                              ci_high=ci.loc[c, 1], p=res.pvalues[c]))
        cal.append(dict(model=name, **calibration(va.y.values.astype(int), np.clip(p, 1e-6, 1 - 1e-6))))
    for name in ("p_struct", "p_struct_naive"):
        preds[name] = va[name].values
        cal.append(dict(model=name, **calibration(va.y.values.astype(int), np.clip(va[name].values, 1e-6, 1 - 1e-6))))

    contrasts = [("A_fut", "A_naive"), ("A_fut", "P0 (A6 benchmark)"), ("A_naive", "P0 (A6 benchmark)"),
                 ("A_full = A_fut + P1", "P3 = P0 + P1"), ("A_full = A_fut + P1", "A_naive_full = A_naive + P1"),
                 ("A_fut_support", "A_fut"), ("p_struct", "p_struct_naive"),
                 ("A_oracle (realised prices; bound)", "A_naive")]
    metrics, delta = bootstrap(va, preds, contrasts)

    # prevalence tracking by t+1 interview year
    prev = []
    for yr, g in va.groupby("interview_year_t1"):
        if len(g) < 100:
            continue
        idx = va.index.get_indexer(g.index)
        row = dict(interview_year_t1=int(yr), n=len(g), actual=g.y.mean())
        for m in ("P0 (A6 benchmark)", "P3 = P0 + P1", "A_naive", "A_fut", "A_full = A_fut + P1", "p_struct"):
            row[m] = preds[m][idx].mean()
        prev.append(row)
    prev = pd.DataFrame(prev)

    acc = pd.read_csv(OUT / "price_accuracy.csv")
    ge = acc[(acc.period == "all 2009+") & acc.series.isin(["gas", "electricity"])]
    rule_i = bool(((ge.rel_rmse < 1) & (ge.dm_p < 0.05)).all())
    d_inc = delta[(delta.contrast == "A_fut - A_naive") & (delta.subset == "incident")].iloc[0]
    rule_ii = bool(d_inc.ci_low > 0)
    decision = pd.DataFrame([
        dict(criterion="(i) futures beat no-change, gas and electricity, DM p<0.05 (all 2009+)",
             result=rule_i, detail="; ".join(f"{r.series}: rel RMSE {r.rel_rmse:.3f}, DM p {r.dm_p:.3f}"
                                             for r in ge.itertuples())),
        dict(criterion="(ii) incident AUC(A_fut) - AUC(A_naive) > 0, 95% CI excludes 0",
             result=rule_ii, detail=f"{d_inc.delta_auc:.4f} [{d_inc.ci_low:.4f}, {d_inc.ci_high:.4f}]"),
        dict(criterion="Anticipation adds value (both)", result=rule_i and rule_ii, detail=""),
    ])

    pd.DataFrame(flow).to_csv(OUT / "sample_flow.csv", index=False)
    metrics.to_csv(OUT / "metrics.csv", index=False)
    delta.to_csv(OUT / "delta_auc.csv", index=False)
    pd.DataFrame(cal).to_csv(OUT / "calibration.csv", index=False)
    pd.DataFrame(coefs).to_csv(OUT / "coefficients.csv", index=False)
    prev.to_csv(OUT / "prevalence_by_year.csv", index=False)
    decision.to_csv(OUT / "decision.csv", index=False)
    plot_prevalence(prev)

    pd.set_option("display.width", 200)
    print(pd.DataFrame(flow).to_string(index=False))
    print(metrics.round(4).to_string(index=False))
    print(delta.round(4).to_string(index=False))
    print(pd.DataFrame(cal).round(3).to_string(index=False))
    print(prev.round(4).to_string(index=False))
    print(decision.to_string(index=False))


def plot_prevalence(prev: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(prev.interview_year_t1, 100 * prev.actual, color="#1f1f1f", lw=2, marker="o", label="Actual")
    style = {"P3 = P0 + P1": ("#8A96A3", "--"), "A_naive": ("#2E5077", ":"), "A_full = A_fut + P1": ("#D55E00", "-")}
    for m, (c, ls) in style.items():
        ax.plot(prev.interview_year_t1, 100 * prev[m], color=c, ls=ls, lw=1.5, marker="o", ms=3, label=m)
    ax.set_ylabel("High fuel vulnerability at t+1 (%)")
    ax.set_xlabel("t+1 interview year (validation transitions)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / "figures" / f"prevalence_by_year.{ext}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
