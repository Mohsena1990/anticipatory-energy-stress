"""
Stage 9 (analysis_plan_rerun.md, amendment A10; EXPLORATORY, post hoc):
analyses requested by the Risk Analysis reviewers.

  1. sample flow (Stage 8 common sample + A10 exclusions)
  2. decision curve analysis, incl. two income-based policies (INC_logit, INC_rule)
  3. Brier score, scaled Brier, AUC / PR-AUC, calibration by subset; paired deltas
  4. construct validation against non-monetary hardship (arrears, warmth, damp)
  5. missing fuel spending: who is missing, IPW, multiple imputation (MICE, M = 20)
  6. frozen-information sensitivity for the national index (P2_frozen, P2_fixed12)
  7. equivalisation sensitivity for prediction (V_eq)

Models, transitions and the common sample are Stage 8's (fit_stage8 /
validation_predictions in stage8_anticipation.py). Inference: 2,000 bootstrap
replicates resampling wave-t PSUs, seed 20261022, paired for every difference.

Outputs (outputs_v4/review/, aggregate tables only): sample_flow.csv,
dca_curves.csv, dca_contrasts.csv, metrics_extra.csv, delta_metrics.csv,
variable_availability.csv, construct_concurrent.csv, construct_concurrent_or.csv,
construct_predictive.csv, missing_by_hardship.csv, missingness_model.csv,
ipw_prevalence.csv, ipw_driver.csv, ipw_ame.csv, ipw_prediction.csv,
mi_prevalence.csv, mi_driver.csv, mi_diagnostics.csv, p2_frozen.csv,
p2_frozen_lead_time.csv, equiv_prediction.csv, figures/dca.{pdf,png}

Usage: python scripts/stage9_review.py [sections...]   (default: all of 2-7)
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src.paths import UKHLS_PANEL  # noqa: E402
import stage3_drivers as s3  # noqa: E402
import stage6_prediction as s6  # noqa: E402
from stage4_resources import FACTORS  # noqa: E402
from stage6_prediction import (BINARY, FES_T1, P0, P1, VALIDATION, WAVES, build_composites,  # noqa: E402
                               calibration, fit_predict, standardise)
from stage8_anticipation import RAW_UKHLS, fit_stage8, logb, validation_predictions  # noqa: E402

OUT = ROOT / "outputs_v4" / "review"
FIG = OUT / "figures"
N_BOOT = 2000
SEED = 20261022
CHUNK = 250
MODELS = ["P0", "P1", "P3", "A_naive", "A_fut", "A_naive_full", "A_full", "A_oracle", "INC_logit", "INC_rule"]
PT_GRID = np.round(np.arange(0.01, 0.4001, 0.01), 2)
PT_CONTRAST = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
DCA_CONTRASTS = [("A_full", "A_naive_full"), ("A_full", "P3"), ("A_full", "INC_logit"), ("A_full", "INC_rule"),
                 ("P0", "INC_logit"), ("A_full", "treat_all")]
DELTA_CONTRASTS = [("A_naive", "P0"), ("A_fut", "A_naive"), ("A_naive_full", "P3"), ("A_full", "A_naive_full"),
                   ("A_full", "P3"), ("A_oracle", "A_naive")]
HARDSHIP = ["arrears", "heat_inadequate", "warm_afford", "damp_pensioner"]
TENURE = {1: "Owned outright", 2: "Buying with mortgage", 3: "Social renting", 4: "Social renting",
          5: "Private renting", 6: "Private renting", 7: "Private renting", 8: "Other"}
DRIVER_TERMS = ["finnow", "fes_delta_growth3", "jbstat_security", "bill_security", "lone_parent", "large_family"]
NESTED = ["P0", "P1", "P3", "A_naive_full", "A_full"]
M_IMP, MICE_CYCLES = 20, 10
COMPONENTS = ["xpduely", "xpgasy", "xpelecy", "xpoily", "xpsfly"]
NAVY, ACCENT, GREY, GRID = "#2E5077", "#D55E00", "#8A96A3", "#E6E9ED"
FLOW: list[dict] = []


# ── bootstrap machinery (vectorised over replicates) ────────────────────────

def boot_weights(psu) -> np.ndarray:
    """(N_BOOT, n) PSU-multinomial replicate weights, seed 20261022."""
    codes, uniq = pd.factorize(pd.Series(psu))
    rng = np.random.default_rng(SEED)
    return rng.multinomial(len(uniq), np.full(len(uniq), 1 / len(uniq)), size=N_BOOT)[:, codes].astype(np.float32)


def _by_chunk(W, fn, y, p):
    return np.concatenate([fn(W[i:i + CHUNK].astype(np.float64), y, p) for i in range(0, len(W), CHUNK)])


def _tie_groups(s):
    return np.r_[0, np.flatnonzero(np.diff(s)) + 1]


def _auc(W, y, p):
    o = np.argsort(p, kind="mergesort")
    st = _tie_groups(p[o])
    Wo = W[:, o]
    pos = np.add.reduceat(Wo * y[o], st, axis=1)
    neg = np.add.reduceat(Wo * (1 - y[o]), st, axis=1)
    below = np.cumsum(neg, axis=1) - neg
    return (pos * (below + 0.5 * neg)).sum(1) / (pos.sum(1) * neg.sum(1))


def _ap(W, y, p):
    o = np.argsort(-p, kind="mergesort")
    st = _tie_groups(-p[o])
    Wo = W[:, o]
    tp = np.cumsum(np.add.reduceat(Wo * y[o], st, axis=1), axis=1)
    fp = np.cumsum(np.add.reduceat(Wo * (1 - y[o]), st, axis=1), axis=1)
    den = tp + fp
    prec = np.divide(tp, den, out=np.zeros_like(tp), where=den > 0)
    rec = tp / tp[:, -1:]
    return (np.diff(rec, axis=1, prepend=0) * prec).sum(1)


def _brier(W, y, p):
    return (W @ ((y - p) ** 2)) / W.sum(1)


METRIC_FN = {"auc": _auc, "prauc": _ap, "brier": _brier}


def point(metric, y, p, w=None):
    w = np.ones((1, len(y))) if w is None else np.asarray(w, float)[None, :]
    return float(METRIC_FN[metric](w, y.astype(float), p.astype(float))[0])


def boot(metric, W, y, p):
    return _by_chunk(W, METRIC_FN[metric], y.astype(float), p.astype(float))


def nb(W, y, p, pt):
    t = (p >= pt).astype(float)
    return (W @ (y * t) - (W @ ((1 - y) * t)) * pt / (1 - pt)) / W.sum(1)


def ci(a):
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


# ── data helpers ─────────────────────────────────────────────────────────────

def read_panel(cols) -> pd.DataFrame:
    return pd.read_csv(UKHLS_PANEL, usecols=lambda c: c in set(["hidp", "wave", "hrpid"] + list(cols)))


def lookup(d: pd.DataFrame, src: pd.DataFrame, col: str, at: str = "t") -> pd.Series:
    """Value of `col` for the household-wave of each row of a transition frame:
    at="t" keys on (hidp, wave); at="t1" keys on (next wave, hrpid), with hrpid
    duplicated within a wave excluded as in the transition linking."""
    if at == "t":
        s = src.drop_duplicates(["hidp", "wave"]).set_index(["hidp", "wave"])[col]
        keys = pd.MultiIndex.from_arrays([d.hidp, d.wave])
    else:
        u = src.dropna(subset=["hrpid"])
        u = u[~u.duplicated(["wave", "hrpid"], keep=False)]
        s = u.set_index(["wave", "hrpid"])[col]
        keys = pd.MultiIndex.from_arrays([d.transition.str[-1], d.hrpid])
    return pd.Series(s.reindex(keys).values, index=d.index)


def raw_household() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hardship items and interview mode from each wave's hhresp, HRP age from
    indresp. Returns (household-wave table, variable availability)."""
    frames, avail_rows = [], {}
    labels = {}
    for w in WAVES:
        with pd.io.stata.StataReader(RAW_UKHLS / f"{w}_hhresp.dta") as r:
            vlab = r.variable_labels()
        want = [c for c in ("xphsdba", "hheat", "pdepi2", "pdeph1", "matdepj", "hholdmodedv") if f"{w}_{c}" in vlab]
        d = pd.read_stata(RAW_UKHLS / f"{w}_hhresp.dta", columns=[f"{w}_hidp"] + [f"{w}_{c}" for c in want],
                          convert_categoricals=False)
        d.columns = ["hidp"] + want
        for c in want:
            labels.setdefault(c, vlab[f"{w}_{c}"])
        out = pd.DataFrame({"hidp": d.hidp, "wave": w})
        rules = {"arrears": ("xphsdba", [1, 2, 3], lambda x: x >= 2),
                 "heat_inadequate": ("hheat", [1, 2], lambda x: x == 2),
                 "warm_afford": ("pdepi2", [0, 1], lambda x: x == 1),
                 "damp_pensioner": ("pdeph1", [0, 1], lambda x: x == 0),
                 "bills_afford": ("matdepj", [1, 2, 3, 4], lambda x: x == 2)}
        for item, (var, valid, f) in rules.items():
            if var in d:
                ok = d[var].isin(valid)
                out[item] = f(d[var]).astype(float).where(ok)
                a = avail_rows.setdefault(var, {"waves": "", "n_valid": 0})
                a["waves"] += w
                a["n_valid"] += int(ok.sum())
            else:
                out[item] = np.nan
        out["mode"] = d["hholdmodedv"] if "hholdmodedv" in d else np.nan
        frames.append(out)
    hh = pd.concat(frames, ignore_index=True)

    ages = []
    for w in WAVES:
        a = pd.read_stata(RAW_UKHLS / f"{w}_indresp.dta", columns=["pidp", f"{w}_dvage"], convert_categoricals=False)
        a.columns = ["hrpid", "hrp_age"]
        a["wave"] = w
        ages.append(a[a.hrp_age >= 0])
    ages = pd.concat(ages, ignore_index=True).drop_duplicates(["wave", "hrpid"])

    concept = {"xphsdba": "(i) household bill arrears", "hheat": "(ii) accommodation adequately heated (any reason)",
               "pdepi2": "(ii) home not kept warm: no money for this (pensioner deprivation module)",
               "pdeph1": "(iii) damp-free home (pensioner deprivation module; not damp/condensation/rot)",
               "matdepj": "related, not used: cannot afford to keep up with bills (material deprivation)"}
    rows = [dict(concept=concept[v], variable=v, file="hhresp", label=labels.get(v, ""), waves_available=a["waves"],
                 n_valid=a["n_valid"]) for v, a in avail_rows.items()]
    rows += [
        dict(concept="(i) gas- or electricity-specific arrears", variable="not found", file="hhresp, indresp",
             label="no variable whose name or label matches arrear/behind/(gas|electric|fuel).*(bill|debt) "
                   "other than xphsdba (all bills)", waves_available="", n_valid=0),
        dict(concept="(ii) cannot keep home adequately warm, all households (can't afford)", variable="not found",
             file="hhresp, indresp", label="matdep*/cdep*/hhdep* contain no warmth item; the afford-to-keep-warm item "
             "exists only in the pensioner module (pdepi*); hheat has no affordability reason", waves_available="",
             n_valid=0),
        dict(concept="(iii) damp, condensation or rot (hsprb* family)", variable="not found", file="hhresp, indresp",
             label="no hsprb* variables; only label match for damp is pdeph* (pensioners: 'damp-free home'); "
                   "no match for condensation or rot", waves_available="", n_valid=0),
        dict(concept="(iv) inability to pay for heating / disconnection", variable="not found", file="hhresp, indresp",
             label="no variable whose name or label matches disconnect or afford.*(fuel|gas|elec|energ)",
             waves_available="", n_valid=0),
    ]
    return hh, ages, pd.DataFrame(rows)


# ── sections 2-3: DCA, Brier, paired deltas ──────────────────────────────────

def add_income_policies(fit: dict, vp: pd.DataFrame, pan: pd.DataFrame) -> float:
    tr, va = fit["tr_raw"].copy(), fit["va_raw"].copy()
    for d in (tr, va):
        d["ieq"] = lookup(d, pan, "ieqmoecd_dv")
        d["hhsize"] = lookup(d, pan, "hhsize")
    # ieqmoecd_dv missing for a handful of transitions: training median for the household size
    # (deviation logged), so the Stage 8 common sample is unchanged.
    med = tr.groupby("hhsize").ieq.median()
    for d, lab in ((tr, "training"), (va, "validation")):
        miss = d.ieq.isna()
        FLOW.append(dict(step=f"  A10: {lab} transitions with ieqmoecd_dv missing, filled by household-size median",
                         n=int(miss.sum())))
        d.loc[miss, "ieq"] = d.loc[miss, "hhsize"].map(med)
    for d in (tr, va):
        d["eq_income"] = d.fihhmnnet1_dv / d.ieq
        d["inc_log"] = np.log1p(d.fihhmnnet1_dv.clip(lower=0))
        d["inceq_log"] = np.log1p(d.eq_income.clip(lower=0))
    n_miss = int(tr.inceq_log.isna().sum() + va.inceq_log.isna().sum())
    if n_miss:
        raise SystemExit("equivalised income missing in the Stage 8 common sample: decide before continuing")
    X = ["inc_log", "inceq_log"]
    tr_s, va_s = tr[["y"] + X].copy(), va[["y"] + X].copy()
    standardise(tr_s, va_s, X)
    _, p = fit_predict(tr_s, va_s, X)
    vp["INC_logit"] = p
    cut = tr.eq_income.quantile(0.20)
    vp["INC_rule"] = (va.eq_income <= cut).astype(float).values
    vp["eq_income"] = va.eq_income.values
    FLOW.append(dict(step="  A10: INC_rule cut-off = training 20th percentile of equivalised monthly net income (GBP, "
                          f"{cut:.0f})", n=len(tr)))
    return float(tr.y.mean())


def sec_dca_metrics(fit: dict, vp: pd.DataFrame, prev_train: float) -> None:
    y_all = vp.y.values.astype(float)
    W_all = boot_weights(vp.psu.values)
    subsets = {"all": np.ones(len(vp), bool), "incident": vp.incident.values}
    curves, contr, met, delta = [], [], [], []
    for s, m in subsets.items():
        y, W = y_all[m], W_all[:, m]
        n, pi = int(m.sum()), y.mean()
        scores = {k: vp[k].values[m] for k in MODELS}
        for k, p in list(scores.items()) + [("treat_all", np.ones(n)), ("treat_none", np.zeros(n) - 1)]:
            for pt in PT_GRID:
                t = p >= pt
                tp, fp = float((y * t).sum()), float(((1 - y) * t).sum())
                curves.append(dict(subset=s, model=k, p_t=pt, net_benefit=tp / n - fp / n * pt / (1 - pt),
                                   tp_per_1000=1000 * tp / n, fp_per_1000=1000 * fp / n, n_treated=int(t.sum()), n=n))
        allp = dict(scores, treat_all=np.ones(n))
        for a, b in DCA_CONTRASTS:
            for pt in PT_CONTRAST:
                pa = nb(np.ones((1, n)), y, allp[a], pt)[0] - nb(np.ones((1, n)), y, allp[b], pt)[0]
                d = 1000 * (nb(W, y, allp[a], pt) - nb(W, y, allp[b], pt))
                lo, hi = ci(d)
                contr.append(dict(subset=s, contrast=f"{a} - {b}", p_t=pt, delta_nb_per_1000=1000 * pa, ci_low=lo,
                                  ci_high=hi, share_boot_gt0=float((d > 0).mean())))
        bt = {}
        brier_null = float(np.mean((y - prev_train) ** 2))
        for k, p in scores.items():
            row = dict(subset=s, model=k, n=n, prevalence=pi)
            for metric in ("brier", "auc", "prauc"):
                if metric == "brier" and k == "INC_rule":
                    row.update(brier=np.nan, brier_ci_low=np.nan, brier_ci_high=np.nan)
                    continue
                bt[(k, metric)] = boot(metric, W, y, p)
                lo, hi = ci(bt[(k, metric)])
                row.update({metric: point(metric, y, p), f"{metric}_ci_low": lo, f"{metric}_ci_high": hi})
            row["scaled_brier"] = 1 - row["brier"] / brier_null if k != "INC_rule" else np.nan
            if k != "INC_rule":
                c = calibration(y.astype(int), np.clip(p, 1e-6, 1 - 1e-6))
                row.update(cal_slope=c["slope"], cal_slope_ci_low=c["slope_ci_low"], cal_slope_ci_high=c["slope_ci_high"],
                           citl=c["intercept"], citl_ci_low=c["intercept_ci_low"], citl_ci_high=c["intercept_ci_high"])
            met.append(row)
        for a, b in DELTA_CONTRASTS:
            for metric in ("auc", "prauc", "brier"):
                d = bt[(a, metric)] - bt[(b, metric)]
                lo, hi = ci(d)
                delta.append(dict(subset=s, contrast=f"{a} - {b}", metric=metric,
                                  delta=point(metric, y, scores[a]) - point(metric, y, scores[b]), ci_low=lo, ci_high=hi,
                                  share_boot_gt0=float((d > 0).mean())))
    cols = ["subset", "model", "n", "prevalence", "brier", "brier_ci_low", "brier_ci_high", "scaled_brier", "auc",
            "auc_ci_low", "auc_ci_high", "prauc", "prauc_ci_low", "prauc_ci_high", "cal_slope", "cal_slope_ci_low",
            "cal_slope_ci_high", "citl", "citl_ci_low", "citl_ci_high"]
    curves = pd.DataFrame(curves)
    plot_dca(curves)
    # within-row disclosure control: with a TP or FP count of 1-9, net benefit and the
    # other rate would let it be recovered from n_treated, so all three are suppressed
    tp, fp = curves.tp_per_1000 * curves.n / 1000, curves.fp_per_1000 * curves.n / 1000
    small = ((tp.round() >= 1) & (tp.round() < 10)) | ((fp.round() >= 1) & (fp.round() < 10))
    curves = curves.astype({"net_benefit": object, "tp_per_1000": object, "fp_per_1000": object})
    curves.loc[small, ["net_benefit", "tp_per_1000", "fp_per_1000"]] = "<10"
    curves.to_csv(OUT / "dca_curves.csv", index=False)
    pd.DataFrame(contr).to_csv(OUT / "dca_contrasts.csv", index=False)
    pd.DataFrame(met).reindex(columns=cols).to_csv(OUT / "metrics_extra.csv", index=False)
    pd.DataFrame(delta).to_csv(OUT / "delta_metrics.csv", index=False)


def plot_dca(curves: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # Okabe-Ito hues in fixed order, line style as the second cue; other models recede in grey.
    style = {"A_full": (ACCENT, "-", 2.0), "A_naive_full": (NAVY, "--", 1.6), "P3": ("#009E73", "-.", 1.6),
             "P0": ("#56B4E9", ":", 1.8), "INC_logit": ("#CC79A7", "-", 1.4), "INC_rule": ("#E69F00", "--", 1.4),
             "treat_all": ("#1f1f1f", ":", 1.2)}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=False)
    for ax, (s, title) in zip(axes, [("all", "All validation transitions"),
                                     ("incident", "Incident (not vulnerable at t)")]):
        g = curves[curves.subset == s]
        for k in ["P1", "A_naive", "A_fut", "A_oracle"]:
            d = g[g.model == k]
            ax.plot(d.p_t, 1000 * d.net_benefit, color="#C5CCD3", lw=0.9, zorder=1)
        for k, (c, ls, lw) in style.items():
            d = g[g.model == k]
            ax.plot(d.p_t, 1000 * d.net_benefit, color=c, ls=ls, lw=lw, label=k.replace("_", " "), zorder=3)
        ax.axhline(0, color="#888", lw=0.8)
        top = 1000 * g[g.model == "A_full"].net_benefit.max()
        ax.set_ylim(-0.25 * top, 1.15 * top)
        ax.set_xlim(0, 0.40)
        ax.set_xlabel("Threshold probability $p_t$")
        ax.set_title(title, fontsize=9, loc="left")
        ax.grid(color=GRID, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Net benefit per 1,000 households")
    axes[1].legend(frameon=False, fontsize=7.5, loc="upper right", title="grey: P1, A naive, A fut, A oracle",
                   title_fontsize=6.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"dca.{ext}", dpi=220)
    plt.close(fig)


# ── section 4: construct validation ─────────────────────────────────────────

def quintile_within_wave(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby("wave")[col].transform(lambda x: pd.qcut(x.rank(method="first"), 5, labels=False) + 1)


def rate_lin(g: pd.DataFrame, flag: str, wcol: str, combine: str = "equal", var: bool = False,
             size_incl_zero: bool = False):
    """Weighted rate (%) as the repository estimators: per-wave weighted rate,
    then the mean over waves ("equal") or weighted by each wave's n ("n").
    var=True adds a PSU-clustered linearisation variance (%^2)."""
    g = g[g[flag].notna()]
    sizes = g.groupby("wave").size() if size_incl_zero else None
    g = g[g[wcol] > 0]
    if g.empty:
        return (np.nan, np.nan) if var else np.nan
    codes, uniq = pd.factorize(g.wave.values)
    y, w = g[flag].values.astype(float), g[wcol].values.astype(float)
    sw, swy = np.bincount(codes, w), np.bincount(codes, w * y)
    r = swy / sw
    if combine == "equal":
        a = np.full(len(uniq), 1 / len(uniq))
    else:
        nw = (sizes.reindex(uniq).values if size_incl_zero else np.bincount(codes)).astype(float)
        a = nw / nw.sum()
    est = 100 * float((a * r).sum())
    if not var:
        return est
    psi = a[codes] * w * (y - r[codes]) / sw[codes]
    tot = pd.Series(psi).groupby(g.psu.values).sum().values
    G = len(tot)
    return est, 1e4 * G / (G - 1) * float(((tot - tot.mean()) ** 2).sum())


def logit_cluster(d: pd.DataFrame, y: str, X: pd.DataFrame):
    X = sm.add_constant(X.astype(float), has_constant="add")
    X = X.loc[:, (X.std() > 0) | (X.columns == "const")]
    return sm.Logit(d[y].astype(float), X).fit(disp=0, maxiter=300, method="newton", cov_type="cluster",
                                               cov_kwds={"groups": pd.factorize(d.psu)[0]})


def sec_construct(fit: dict, vp: pd.DataFrame, hh: pd.DataFrame, ages: pd.DataFrame, avail: pd.DataFrame) -> dict:
    avail.to_csv(OUT / "variable_availability.csv", index=False)
    pan = read_panel(["psu", "hh_xw", "high_fuel_vulnerable", "fihhmnnet1_dv", "ieqmoecd_dv", "hhsize",
                      "interview_year"])
    pan = pan.merge(hh, on=["hidp", "wave"], how="left").merge(ages, on=["wave", "hrpid"], how="left")
    pan["eq_income"] = pan.fihhmnnet1_dv / pan.ieqmoecd_dv
    pan["inc_q"] = quintile_within_wave(pan, "eq_income")
    conc, ors, decision = [], [], {}
    for item in HARDSHIP:
        d = pan[pan[item].notna() & pan.high_fuel_vulnerable.notna()].copy()
        if d.empty:
            continue
        p1 = rate_lin(d[d.high_fuel_vulnerable == 1], item, "hh_xw")
        p0 = rate_lin(d[d.high_fuel_vulnerable == 0], item, "hh_xw")
        conc.append(dict(item=item, waves="".join(sorted(d.wave.unique())), n=len(d), prev_H_if_V1=p1,
                         prev_H_if_V0=p0, ratio=p1 / p0))
        res = logit_cluster(d, item, d[["high_fuel_vulnerable"]])
        c = res.conf_int().loc["high_fuel_vulnerable"]
        ors.append(dict(item=item, adjusted="no", **{"or": np.exp(res.params["high_fuel_vulnerable"])},
                        ci_low=np.exp(c[0]), ci_high=np.exp(c[1]), n=int(res.nobs)))
        da = d.dropna(subset=["inc_q", "hhsize", "hrp_age", "interview_year"])
        X = pd.concat([da[["high_fuel_vulnerable", "hhsize", "hrp_age"]],
                       pd.get_dummies(da.inc_q.astype(int), prefix="incq", drop_first=True, dtype=float),
                       pd.get_dummies(da.interview_year.astype(int), prefix="yr", drop_first=True, dtype=float)], axis=1)
        res = logit_cluster(da, item, X)
        c = res.conf_int().loc["high_fuel_vulnerable"]
        ors.append(dict(item=item, adjusted="yes", **{"or": np.exp(res.params["high_fuel_vulnerable"])},
                        ci_low=np.exp(c[0]), ci_high=np.exp(c[1]), n=int(res.nobs)))
        decision[item] = dict(concurrent=bool(np.exp(c[0]) > 1), p_adj=float(res.pvalues["high_fuel_vulnerable"]))
    pd.DataFrame(conc).to_csv(OUT / "construct_concurrent.csv", index=False)
    pd.DataFrame(ors).to_csv(OUT / "construct_concurrent_or.csv", index=False)

    # 4c predictive validity on the validation transitions
    va = fit["va_raw"]
    preds = {k: vp[k].values for k in MODELS}
    preds.update(b_hat_constant_price=va.burden_naive.values, b_hat_forecast_price=va.burden_forecast.values,
                 b_t=va.fuel_to_income_ratio.values)
    rows = []
    for item in HARDSHIP:
        h1 = lookup(va, pan, item, at="t1").values
        h0 = lookup(va, pan, item, at="t").values
        decision.setdefault(item, dict(concurrent=False, p_adj=np.nan))
        decision[item]["predictive"] = False
        for s, m in [("all", ~np.isnan(h1)), ("new_hardship (H=0 at t)", ~np.isnan(h1) & (h0 == 0))]:
            n = int(m.sum())
            FLOW.append(dict(step=f"  A10 4c: validation transitions with {item} at t+1, {s}", n=n))
            if n == 0 or np.nansum(h1[m]) == 0:
                continue
            y = h1[m]
            W = boot_weights(vp.psu.values[m])
            for k, p in preds.items():
                lo, hi = ci(boot("auc", W, y, p[m]))
                rows.append(dict(item=item, subset=s, predictor=k, n=n, prevalence=float(y.mean()),
                                 auc=point("auc", y, p[m]), auc_ci_low=lo, auc_ci_high=hi))
                if k == "b_hat_forecast_price" and s == "all":
                    decision[item]["predictive"] = bool(lo > 0.5)
    pd.DataFrame(rows).to_csv(OUT / "construct_predictive.csv", index=False)
    return decision


# ── section 5: missing data ─────────────────────────────────────────────────

def prev_wave_status(df: pd.DataFrame) -> pd.Series:
    u = df.dropna(subset=["hrpid"])
    u = u[~u.duplicated(["wave", "hrpid"], keep=False)]
    v = u.set_index(["wave", "hrpid"]).high_fuel_vulnerable
    prev_w = df.wave.map({b: a for a, b in zip(WAVES[:-1], WAVES[1:])})
    linked = pd.Series(pd.MultiIndex.from_arrays([prev_w, df.hrpid]).isin(v.index), index=df.index)
    linked &= ~df.duplicated(["wave", "hrpid"], keep=False) & df.hrpid.notna()
    val = pd.Series(v.reindex(pd.MultiIndex.from_arrays([prev_w, df.hrpid])).values, index=df.index)
    return pd.Series(np.select([linked & (val == 1), linked & (val == 0), linked], ["vulnerable", "not vulnerable",
                                "linked, not observed"], "not linked / first wave"), index=df.index)


def adjacent_spend(df: pd.DataFrame, step: int) -> pd.Series:
    u = df.dropna(subset=["hrpid"])
    u = u[~u.duplicated(["wave", "hrpid"], keep=False)]
    s = np.log1p(u.set_index(["wave", "hrpid"]).total_fuel_spend)
    order = {w: i for i, w in enumerate(WAVES)}
    adj = df.wave.map(lambda w: WAVES[order[w] + step] if 0 <= order[w] + step < len(WAVES) else None)
    return pd.Series(s.reindex(pd.MultiIndex.from_arrays([adj, df.hrpid])).values, index=df.index)


def cat_dummies(s: pd.Series, prefix: str, ref) -> pd.DataFrame:
    d = pd.get_dummies(s.fillna("missing").astype(str), prefix=prefix, dtype=float)
    return d.drop(columns=[f"{prefix}_{ref}"], errors="ignore")


def fill_ind(s: pd.Series, name: str) -> pd.DataFrame:
    return pd.DataFrame({name: s.fillna(s.median()), f"{name}_missing": s.isna().astype(float)})


def fit_glm(df: pd.DataFrame, y: str, terms: list[str], fe: list[str], w: np.ndarray):
    X = s3.design(df, terms, fe)
    X = X.loc[:, X.std() > 0].assign(const=1.0)
    return sm.GLM(df[y].astype(float), X, family=sm.families.Binomial(), var_weights=np.asarray(w)).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(df["psu"])[0]})


def ame_w(res, contrast: str, w=None) -> tuple[float, float]:
    """stage3_drivers.ame with an optional (normalised) weight on the average."""
    X = pd.DataFrame(res.model.exog, columns=res.model.exog_names)
    X1, X0 = X.copy(), X.copy()
    if contrast == "ni":
        reg = [c for c in X if c.startswith("region_")]
        X1[reg] = 0.0
        X0[reg] = 0.0
        X1["region_Northern Ireland"] = 1.0
    else:
        X1["oil"], X0["oil"] = 1.0, 0.0
    w = np.ones(len(X)) if w is None else np.asarray(w, float)
    w = w / w.sum()
    b = res.params.values
    p1, p0 = 1 / (1 + np.exp(-X1.values @ b)), 1 / (1 + np.exp(-X0.values @ b))
    est = float(np.sum(w * (p1 - p0)))
    grad = ((X1.values * (p1 * (1 - p1))[:, None] - X0.values * (p0 * (1 - p0))[:, None]) * w[:, None]).sum(axis=0)
    return 100 * est, 100 * float(np.sqrt(grad @ res.cov_params().values @ grad))


def cells(df: pd.DataFrame) -> list[tuple]:
    rows = []
    for wv, g in df.groupby("wave"):
        rows.append(("wave", wv, g))
    for reg, g in df.dropna(subset=["region"]).groupby("region"):
        rows.append(("region", reg, g))
    win = df[(df["T"] >= "2021-04-01") & (df["T"] <= "2023-03-01")]
    for reg, g in win.dropna(subset=["region"]).groupby("region"):
        rows.append(("region_apr2021_mar2023", reg, g))
    for yr, g in df.dropna(subset=["interview_year"]).groupby("interview_year"):
        rows.append(("interview_year", str(int(yr)), g))
    return rows


def prevalence_table(df: pd.DataFrame, flag: str, wcol: str, var: bool = False) -> list[dict]:
    out = []
    for dim, grp, g in cells(df):
        combine = {"wave": "equal", "region": "equal"}.get(dim, "n")
        r = rate_lin(g, flag, wcol, combine, var=var, size_incl_zero=(dim == "region_apr2021_mar2023"))
        est, v = r if var else (r, np.nan)
        out.append(dict(dimension=dim, group=grp, est=est, var=v, n=int(g[flag].notna().sum())))
    return out


def sec_missing(fit: dict, vp: pd.DataFrame, hh: pd.DataFrame, ages: pd.DataFrame) -> dict:
    t0 = time.time()
    df = s3.load()
    df = df.merge(hh[["hidp", "wave", "mode"]], on=["hidp", "wave"], how="left").merge(ages, on=["wave", "hrpid"],
                                                                                       how="left")
    df["T"] = pd.to_datetime(dict(year=df.interview_year, month=df.interview_month, day=1), errors="coerce")
    df["eq_income"] = df.fihhmnnet1_dv / df.ieqmoecd_dv
    df["inc_q"] = quintile_within_wave(df, "eq_income")
    df["tenure"] = df.tenure_dv.map(TENURE)
    df["prev_V"] = prev_wave_status(df)
    elig = df.fuel_outcome_exclusion.isin(["in_scope", "item_nonresponse"])
    df["missing"] = (df.fuel_outcome_exclusion == "item_nonresponse").astype(float)
    FLOW.extend([dict(step="A10 5: household-waves eligible for the spend items (in scope or item nonresponse)",
                      n=int(elig.sum())),
                 dict(step="  of which spend item nonresponse (missing)", n=int(df.missing[elig].sum()))])
    E = df[elig].copy()

    # 5a. who is missing
    # finnow is the household mean over adults: grouped by rounding to the nearest category (half up)
    dims = {"finnow": np.floor(E.finnow + 0.5).map(lambda v: f"{int(v)}" if pd.notna(v) else "missing"),
            "xphsdba": E.xphsdba.map({1: "up to date", 2: "behind with some", 3: "behind with all"}).fillna("missing"),
            "prev_wave_V": E.prev_V,
            "income_quintile": E.inc_q.map(lambda v: f"Q{int(v)}" if pd.notna(v) else "missing"),
            "tenure": E.tenure.fillna("missing"), "wave": E.wave}
    rows = []
    for dim, s in dims.items():
        for grp, g in E.groupby(s):
            rows.append(dict(dimension=dim, group=grp, n=len(g), n_missing=int(g.missing.sum()),
                             pct_missing=100 * g.missing.mean()))
    pd.DataFrame(rows).to_csv(OUT / "missing_by_hardship.csv", index=False)
    mode = E["mode"].map({1: "capi", 2: "cati", 3: "cawi"})
    mode = mode.where(E["mode"].notna() | E.wave.isin(list("cdefghijklmno")), "not recorded")
    X = pd.concat([cat_dummies(dims["finnow"], "finnow", "1"), cat_dummies(dims["xphsdba"], "xphsdba", "up to date"),
                   cat_dummies(E.prev_V, "prevV", "not vulnerable"),
                   cat_dummies(dims["income_quintile"], "incq", "Q3"),
                   cat_dummies(E.tenure, "tenure", "Owned outright"), cat_dummies(E.wave, "wave", "a"),
                   cat_dummies(E.region, "region", "South East"),
                   cat_dummies(E.employment_group, "employment", "Full-time or self-employed"),
                   cat_dummies(mode, "mode", "capi"), fill_ind(E.hhsize, "hhsize"), fill_ind(E.hrp_age, "hrp_age")],
                  axis=1)
    res = logit_cluster(E, "missing", X)
    cint = res.conf_int()
    pd.DataFrame([dict(term=t, **{"or": np.exp(res.params[t])}, ci_low=np.exp(cint.loc[t, 0]),
                       ci_high=np.exp(cint.loc[t, 1]), p=res.pvalues[t]) for t in res.params.index if t != "const"]
                 ).to_csv(OUT / "missingness_model.csv", index=False)
    p_obs = 1 - res.predict()
    obs = E.missing.values == 0
    rate_obs = obs.mean()
    w_model = rate_obs / p_obs
    lo, hi = np.percentile(w_model[obs], [1, 99])
    df.loc[E.index, "w_ipw"] = np.clip(w_model, lo, hi)
    w_prev = E.hh_xw.values * rate_obs / p_obs
    pos = obs & (E.hh_xw.values > 0)
    lo, hi = np.percentile(w_prev[pos], [1, 99])
    df.loc[E.index, "w_prev_ipw"] = np.where(E.hh_xw.values > 0, np.clip(w_prev, lo, hi), 0.0)
    print(f"  missingness model done ({time.time() - t0:.0f}s)", flush=True)

    # 5b. IPW
    prim = prevalence_table(df, "high_fuel_vulnerable", "hh_xw")
    ipw = prevalence_table(df, "high_fuel_vulnerable", "w_prev_ipw")
    pd.DataFrame([dict(dimension=a["dimension"], group=a["group"], prev_primary=a["est"], prev_ipw=b["est"], n=a["n"])
                  for a, b in zip(prim, ipw)]).to_csv(OUT / "ipw_prevalence.csv", index=False)

    terms = s3.CONTROLS + s3.STRAIN + ["fes_delta_growth3"]
    y = "high_fuel_vulnerable"
    base = df.dropna(subset=terms + [y, "psu", "interview_year", "year_month"])
    r_p = s3.fit(base, y, terms, ["interview_year"])
    r_i = fit_glm(base, y, terms, ["interview_year"], base.w_ipw.values)
    ci_i = r_i.conf_int()
    pd.DataFrame([dict(term=t, or_primary=np.exp(r_p.params[t]), or_ipw=np.exp(r_i.params[t]),
                       ci_low_ipw=np.exp(ci_i.loc[t, 0]), ci_high_ipw=np.exp(ci_i.loc[t, 1])) for t in DRIVER_TERMS]
                 ).to_csv(OUT / "ipw_driver.csv", index=False)
    s0 = base.dropna(subset=["region", "oil", "rural"])
    rb_p = s3.fit(s0, y, terms + ["oil"], ["interview_year", "region"])
    rb_i = fit_glm(s0, y, terms + ["oil"], ["interview_year", "region"], s0.w_ipw.values)
    rows = []
    for c, lab in [("ni", "NI vs South East"), ("oil", "oil vs no oil")]:
        ep, sp = ame_w(rb_p, c)
        ei, si = ame_w(rb_i, c, s0.w_ipw.values)
        rows.append(dict(model="ni_b_oil_filled", contrast=lab, ame_primary_pp=ep, ci_low_primary=ep - 1.96 * sp,
                         ci_high_primary=ep + 1.96 * sp, ame_ipw_pp=ei, ci_low_ipw=ei - 1.96 * si,
                         ci_high_ipw=ei + 1.96 * si, n=int(rb_p.nobs)))
    pd.DataFrame(rows).to_csv(OUT / "ipw_ame.csv", index=False)
    print(f"  IPW driver/AME done ({time.time() - t0:.0f}s)", flush=True)

    va = fit["va_raw"]
    w_t1 = lookup(va, df, "w_ipw", at="t1").values
    FLOW.append(dict(step="  A10 5b: validation transitions without an IPW weight at t+1", n=int(np.isnan(w_t1).sum())))
    rows = []
    for s, m in [("all", np.ones(len(vp), bool)), ("incident", vp.incident.values)]:
        for k in NESTED:
            rows.append(dict(subset=s, model=k, auc_primary=roc_auc_score(vp.y.values[m], vp[k].values[m]),
                             auc_ipw=roc_auc_score(vp.y.values[m], vp[k].values[m], sample_weight=w_t1[m])))
    pd.DataFrame(rows).to_csv(OUT / "ipw_prediction.csv", index=False)

    # 5c. multiple imputation
    mi = run_mi(df, E.index)
    print(f"  MI imputations done ({time.time() - t0:.0f}s)", flush=True)
    pool_mi(df, mi, prim, r_p, terms)
    print(f"  MI pooling done ({time.time() - t0:.0f}s)", flush=True)
    return dict(prim=prim, ipw=ipw)


def run_mi(df: pd.DataFrame, idx: pd.Index) -> list[pd.Series]:
    """MICE over the fuel components (log scale, Bayesian normal linear model,
    parameters drawn per imputation). Returns, per imputation, V for every
    eligible household-wave (observed values unchanged)."""
    df["next_spend"] = adjacent_spend(df, +1)
    df["prev_spend"] = adjacent_spend(df, -1)
    E = df.loc[idx]
    elec, gas = E.fuelhave1 == 1, E.fuelhave2 == 1
    duel_nr = E.fuelduel_nr.fillna(False).astype(bool)
    both = elec & gas
    sep = both & ((E.fuelduel == 2) | duel_nr)
    app = {"xpduely": both & (E.fuelduel == 1), "xpgasy": sep, "xpelecy": sep | (elec & ~gas),
           "xpoily": E.fuelhave3 == 1, "xpsfly": E.fuelhave4 == 1}
    app = {c: a.values for c, a in app.items()}
    L = {c: np.where(app[c], np.log1p(E[c].clip(lower=0).values), 0.0) for c in COMPONENTS}
    mis = {c: app[c] & np.isnan(L[c]) for c in COMPONENTS}
    for c in COMPONENTS:
        FLOW.append(dict(step=f"  A10 5c: {c} applicable", n=int(app[c].sum())))
        FLOW.append(dict(step=f"    {c} missing (imputed)", n=int(mis[c].sum())))
    rng_lo = {c: np.nanmin(L[c][app[c] & ~mis[c]]) for c in COMPONENTS}
    rng_hi = {c: np.nanmax(L[c][app[c] & ~mis[c]]) for c in COMPONENTS}
    Xb = pd.concat([fill_ind(np.log1p(E.fihhmnnet1_dv.clip(lower=0)), "log_income"), fill_ind(E.hhsize, "hhsize"),
                    cat_dummies(E.tenure, "tenure", "Owned outright"), cat_dummies(E.region, "region", "South East"),
                    E[["fuelhave3", "fuelhave4"]].fillna(0), pd.DataFrame({"gas_use": gas.astype(float),
                                                                          "separate_bills": sep.astype(float)}),
                    fill_ind(E.prepayment_meter, "prepay"), fill_ind(E.hsrooms, "rooms"), fill_ind(E.heatch, "heatch"),
                    cat_dummies(E.interview_year, "year", "2015.0"), cat_dummies(E.interview_month, "month", "1.0"),
                    fill_ind(E.finnow, "finnow"), fill_ind(E.prev_spend, "prev_spend"),
                    fill_ind(E.next_spend, "next_spend")], axis=1).astype(float)
    Xb.insert(0, "const", 1.0)
    Xb = Xb.values
    rng = np.random.default_rng(SEED)
    inc12 = 12 * E.fihhmnnet1_dv.values
    out = []
    for m in range(M_IMP):
        cur = {c: L[c].copy() for c in COMPONENTS}
        for c in COMPONENTS:
            cur[c][mis[c]] = np.nanmean(L[c][app[c] & ~mis[c]])
        for _ in range(MICE_CYCLES):
            for c in COMPONENTS:
                if not mis[c].any():
                    continue
                others = [o for o in COMPONENTS if o != c]
                Z = np.column_stack([Xb] + [cur[o] for o in others] + [app[o].astype(float) for o in others])
                fit_rows = app[c] & ~mis[c]
                Zt, yt = Z[fit_rows], cur[c][fit_rows]
                keep = Zt.std(axis=0) > 0
                keep[0] = True
                Zt = Zt[:, keep]
                XtX = Zt.T @ Zt
                # tiny ridge: applicability indicators can duplicate base columns (e.g. oil use)
                XtX += 1e-6 * np.mean(np.diag(XtX)) * np.eye(Zt.shape[1])
                chol = np.linalg.cholesky(XtX)
                beta = np.linalg.solve(XtX, Zt.T @ yt)
                resid = yt - Zt @ beta
                dof = len(yt) - Zt.shape[1]
                s2 = float(resid @ resid) / rng.chisquare(dof)
                # L^-T z has covariance (X'X)^-1
                b_draw = beta + np.sqrt(s2) * np.linalg.solve(chol.T, rng.standard_normal(len(beta)))
                Zm = Z[mis[c]][:, keep]
                cur[c][mis[c]] = np.clip(Zm @ b_draw + np.sqrt(s2) * rng.standard_normal(mis[c].sum()),
                                         rng_lo[c], rng_hi[c])
        amt = {c: np.where(app[c], np.expm1(cur[c]), 0.0) for c in COMPONENTS}
        total = np.where(app["xpduely"], amt["xpduely"], 0.0) + np.where(sep, amt["xpgasy"] + amt["xpelecy"], 0.0)
        total += np.where((elec & ~gas).values, amt["xpelecy"], 0.0) + amt["xpoily"] + amt["xpsfly"]
        ratio = np.clip(total / inc12, None, 1.0)
        ratio[~(inc12 >= 1200)] = np.nan
        v = np.where(np.isnan(ratio), np.nan, (ratio >= 0.10).astype(float))
        if m == 0:
            obs = E.high_fuel_vulnerable.notna().values
            agree = np.mean(v[obs] == E.high_fuel_vulnerable.values[obs])
            FLOW.append(dict(step=f"  A10 5c check: recomputed V equals primary V on observed rows (per million; rest = round-off at 10%: "
                                  f"{int(round(agree * 1e6))})", n=int(obs.sum())))
            imputed = ~obs & ~np.isnan(v)
            FLOW.append(dict(step="  A10 5c: eligible household-waves with V imputed (missing spend, income guard met)",
                             n=int(imputed.sum())))
        v = np.where(E.high_fuel_vulnerable.notna().values, E.high_fuel_vulnerable.values, v)
        out.append(pd.Series(v, index=idx))
        print(f"    imputation {m + 1}/{M_IMP}", flush=True)
    return out


def rubin(q: np.ndarray, u: np.ndarray) -> dict:
    M = len(q)
    qbar, ubar, b = q.mean(), u.mean(), q.var(ddof=1)
    t = ubar + (1 + 1 / M) * b
    r = (1 + 1 / M) * b / ubar if ubar > 0 else np.inf
    dof = (M - 1) * (1 + 1 / r) ** 2 if b > 0 else np.inf
    half = stats.t.ppf(0.975, dof) * np.sqrt(t)
    fmi = (r + 2 / (dof + 3)) / (r + 1) if b > 0 else 0.0
    return dict(est=qbar, ci_low=qbar - half, ci_high=qbar + half, fmi=fmi)


def pool_mi(df: pd.DataFrame, mi: list[pd.Series], prim: list[dict], r_p, terms: list[str]) -> None:
    observed = df.high_fuel_vulnerable.notna()
    tabs = []
    for v in mi:
        df["V_mi"] = np.nan
        df.loc[v.index, "V_mi"] = v.values
        tabs.append(prevalence_table(df, "V_mi", "hh_xw", var=True))
    df["imputed"] = np.nan
    df.loc[mi[0].index, "imputed"] = (~observed.loc[mi[0].index] & mi[0].notna()).astype(float)
    rows, diag = [], []
    for i, a in enumerate(prim):
        r = rubin(np.array([t[i]["est"] for t in tabs]), np.array([t[i]["var"] for t in tabs]))
        rows.append(dict(dimension=a["dimension"], group=a["group"], prev_primary=a["est"], prev_mi=r["est"],
                         ci_low=r["ci_low"], ci_high=r["ci_high"]))
        diag.append(dict(table="mi_prevalence", dimension=a["dimension"], group_or_term=a["group"],
                         n=tabs[0][i]["n"], share_outcome_imputed=np.nan, fmi=r["fmi"]))
    # share imputed per prevalence cell: unweighted share of the cell's V that was imputed
    df["V_mi"] = np.nan
    df.loc[mi[0].index, "V_mi"] = mi[0].values
    for dgn, (_, _, g) in zip(diag, cells(df)):
        g = g[g.V_mi.notna()]
        dgn["share_outcome_imputed"] = float(g.imputed.mean()) if len(g) else np.nan
    pd.DataFrame(rows).to_csv(OUT / "mi_prevalence.csv", index=False)

    y = "V_mi"
    q, u, share_d, n_d = {t: [] for t in DRIVER_TERMS}, {t: [] for t in DRIVER_TERMS}, [], []
    for k, v in enumerate(mi):
        df["V_mi"] = np.nan
        df.loc[v.index, "V_mi"] = v.values
        base = df.dropna(subset=terms + [y, "psu", "interview_year", "year_month"])
        res = s3.fit(base, y, terms, ["interview_year"])
        for t in DRIVER_TERMS:
            q[t].append(res.params[t])
            u[t].append(res.bse[t] ** 2)
        share_d.append(float(base.imputed.mean()))
        n_d.append(len(base))
        print(f"    driver model {k + 1}/{M_IMP}", flush=True)
    rows = []
    for t in DRIVER_TERMS:
        r = rubin(np.array(q[t]), np.array(u[t]))
        rows.append(dict(term=t, or_primary=np.exp(r_p.params[t]), or_mi=np.exp(r["est"]), ci_low_mi=np.exp(r["ci_low"]),
                         ci_high_mi=np.exp(r["ci_high"])))
        diag.append(dict(table="mi_driver", dimension="primary driver model", group_or_term=t, n=int(np.mean(n_d)),
                         share_outcome_imputed=float(np.mean(share_d)), fmi=r["fmi"]))
    pd.DataFrame(rows).to_csv(OUT / "mi_driver.csv", index=False)
    pd.DataFrame(diag).to_csv(OUT / "mi_diagnostics.csv", index=False)


# ── section 6: frozen-information P2 ────────────────────────────────────────

def sec_p2() -> None:
    global FLOW
    pairs, _ = s6.transitions()
    pan = read_panel(["interview_month"])
    pairs["interview_month_t1"] = lookup(pairs, pan, "interview_month", at="t1")
    tr_all = pairs[~pairs.transition.isin(VALIDATION)].copy()
    va_all = pairs[pairs.transition.isin(VALIDATION)].copy()
    build_composites(tr_all, va_all)
    allp = pd.concat([tr_all, va_all])
    cs = ~allp[["y", "psu"] + P0 + P1 + [FES_T1]].isna().any(axis=1)

    monthly = pd.read_csv(ROOT / "outputs_v2" / "fes" / "fes_rolling_monthly.csv")
    monthly["g3"] = monthly[["z_gas_core", "z_electricity_core", "z_carbon_core"]].sum(axis=1, min_count=3)
    g3 = monthly.set_index(["as_of_year", "target_month"]).g3
    annual = monthly.groupby("as_of_year").g3.mean()

    def frozen(d):
        vint = d.interview_year - 1          # latest December vintage before the wave-t interview month T
        out, fix = [], []
        for v, mth in zip(vint, d.interview_month):
            if pd.isna(v) or int(v) not in annual.index:
                out.append(np.nan); fix.append(np.nan)
                continue
            v = int(v)
            months = range(int(mth), 13) if pd.notna(mth) else range(1, 13)
            out.append(np.mean([g3[(v, k)] for k in months]))
            fix.append(g3[(v, 12)])
        d["fes_frozen"], d["fes_fixed12"] = out, fix
        d["lead_published"] = d.interview_month_t1
        d["lead_frozen"] = (d.interview_year_t1 - vint - 1) * 12 + d.interview_month_t1

    for d in (tr_all, va_all):
        frozen(d)
    # reproduce Stage 6 P1/P2 on the published common sample
    tr0, va0 = tr_all[cs.loc[tr_all.index]].copy(), va_all[cs.loc[va_all.index]].copy()
    standardise(tr0, va0, sorted(set(P0 + P1 + [FES_T1])))
    ref = pd.read_csv(ROOT / "outputs_v2" / "stage6" / "metrics.csv").query("metric == 'auc'").set_index("model")
    for k, cols in [("P1", P1), ("P2", P1 + [FES_T1])]:
        _, p = fit_predict(tr0, va0, cols)
        a = roc_auc_score(va0.y, p)
        FLOW.append(dict(step=f"  A10 6 check: {k} AUC on Stage 6 sample {a:.4f} (published {ref.loc[k, 'estimate']:.4f})",
                         n=len(va0)))
    allp = pd.concat([tr_all, va_all])
    cs2 = cs & allp.fes_frozen.notna() & allp.fes_fixed12.notna()
    FLOW += [dict(step="A10 6: Stage 6 common sample (P0/P1/P2)", n=int(cs.sum())),
             dict(step="  minus transitions with no vintage before T (wave-t interview in 2009)",
                  n=int((cs & ~cs2).sum())),
             dict(step="  A10 6 sample", n=int(cs2.sum()))]
    tr, va = tr_all[cs2.loc[tr_all.index]].copy(), va_all[cs2.loc[va_all.index]].copy()
    FLOW += [dict(step="    training", n=len(tr)), dict(step="    validation", n=len(va))]
    lead = []
    for k, col in [("P2 (as published)", "lead_published"), ("P2_frozen", "lead_frozen"), ("P2_fixed12", "lead_frozen")]:
        x = va[col]
        lead.append(dict(model=k, sample="validation", lead_mean_months=x.mean(), lead_min=x.min(), lead_max=x.max(),
                         definition="months from the vintage's December origin to the t+1 interview month",
                         target=("t+1 interview month" if k.startswith("P2 (") else
                                 "mean of months T..Dec of T's year" if k == "P2_frozen" else "December of T's year (h=12)")))
    pd.DataFrame(lead).to_csv(OUT / "p2_frozen_lead_time.csv", index=False)
    standardise(tr, va, sorted(set(P1 + [FES_T1, "fes_frozen", "fes_fixed12"])))
    specs = {"P1": P1, "P2 (as published)": P1 + [FES_T1], "P2_frozen": P1 + ["fes_frozen"],
             "P2_fixed12": P1 + ["fes_fixed12"]}
    y = va.y.values.astype(float)
    W = boot_weights(va.psu.values)
    preds = {k: fit_predict(tr, va, cols)[1] for k, cols in specs.items()}
    bt = {k: boot("auc", W, y, p) for k, p in preds.items()}
    rows = []
    for k, p in preds.items():
        lo, hi = ci(bt[k])
        d = bt[k] - bt["P1"]
        dlo, dhi = ci(d)
        rows.append(dict(model=k, auc=point("auc", y, p), auc_ci_low=lo, auc_ci_high=hi,
                         delta_vs_P1=point("auc", y, p) - point("auc", y, preds["P1"]),
                         delta_ci_low=dlo if k != "P1" else np.nan, delta_ci_high=dhi if k != "P1" else np.nan, n=len(va)))
    pd.DataFrame(rows).to_csv(OUT / "p2_frozen.csv", index=False)


# ── section 7: equivalisation sensitivity ───────────────────────────────────

def sec_equiv(fit: dict, vp: pd.DataFrame) -> None:
    global FLOW
    pan = read_panel(["fuel_to_income_ratio_equivalised", "high_fuel_vulnerable_equivalised", "ieqmoecd_dv"])
    frames = {}
    for name in ("tr_raw", "va_raw"):
        d = fit[name].copy()
        d["req_t"] = lookup(d, pan, "fuel_to_income_ratio_equivalised")
        d["veq_t"] = lookup(d, pan, "high_fuel_vulnerable_equivalised")
        d["veq_t1"] = lookup(d, pan, "high_fuel_vulnerable_equivalised", at="t1")
        ieq = lookup(d, pan, "ieqmoecd_dv")
        d["logb_naive_eq"] = logb((d.burden_naive * ieq).clip(upper=1.0))
        d["logb_fut_eq"] = logb((d.burden_forecast * ieq).clip(upper=1.0))
        frames[name] = d
    tr, va = frames["tr_raw"], frames["va_raw"]
    need = ["req_t", "veq_t", "veq_t1", "logb_naive_eq", "logb_fut_eq"]
    ok_tr, ok_va = tr[need].notna().all(axis=1), va[need].notna().all(axis=1)
    FLOW += [dict(step="A10 7: Stage 8 common sample without V_eq at t or t+1", n=int((~ok_tr).sum() + (~ok_va).sum())),
             dict(step="  A10 7 sample: training", n=int(ok_tr.sum())),
             dict(step="  A10 7 sample: validation", n=int(ok_va.sum()))]
    tr, va = tr[ok_tr].copy(), va[ok_va].copy()
    vpe = vp.loc[va.index]
    specs = {"P0": ["req_t", "veq_t"], "P1": P1, "P3": ["req_t", "veq_t"] + P1,
             "A_naive_full": ["logb_naive_eq", "veq_t"] + P1, "A_full": ["logb_fut_eq", "veq_t"] + P1}
    cont = ["req_t", "logb_naive_eq", "logb_fut_eq"] + [c for c in P1 if c not in BINARY and c not in FACTORS]
    standardise(tr, va, cont)
    tr["y"], va["y"] = tr.veq_t1.astype(float), va.veq_t1.astype(float)
    preds = {"V_eq": {k: fit_predict(tr, va, cols)[1] for k, cols in specs.items()},
             "V (primary, same sample)": {k: vpe[k].values for k in specs}}
    ys = {"V_eq": va.y.values, "V (primary, same sample)": vpe.y.values.astype(float)}
    incid = {"V_eq": va.veq_t.values == 0, "V (primary, same sample)": vpe.incident.values}
    W_all = boot_weights(va.psu.values)
    rows = []
    for outc, pr in preds.items():
        for s in ("all", "incident"):
            m = np.ones(len(va), bool) if s == "all" else incid[outc]
            y, W = ys[outc][m], W_all[:, m]
            bt = {k: boot("auc", W, y, p[m]) for k, p in pr.items()}
            for k, p in pr.items():
                lo, hi = ci(bt[k])
                rows.append(dict(outcome=outc, subset=s, model=k, prevalence=float(y.mean()), auc=point("auc", y, p[m]),
                                 auc_ci_low=lo, auc_ci_high=hi, n=int(m.sum())))
            for a, b in [("A_full", "A_naive_full"), ("A_full", "P3")]:
                lo, hi = ci(bt[a] - bt[b])
                rows.append(dict(outcome=outc, subset=s, model=f"{a} - {b}", prevalence=np.nan,
                                 auc=point("auc", y, pr[a][m]) - point("auc", y, pr[b][m]), auc_ci_low=lo,
                                 auc_ci_high=hi, n=int(m.sum())))
    pd.DataFrame(rows)[["outcome", "subset", "model", "prevalence", "auc", "auc_ci_low", "auc_ci_high", "n"]] \
        .to_csv(OUT / "equiv_prediction.csv", index=False)


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    sections = set(sys.argv[1:]) or {"2", "4", "5", "6", "7"}
    FIG.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    fit = fit_stage8()
    vp = validation_predictions(fit)
    FLOW.extend(fit["flow"])
    print(f"Stage 8 fit ({time.time() - t0:.0f}s)", flush=True)
    pan = read_panel(["ieqmoecd_dv", "hhsize"])
    prev_train = add_income_policies(fit, vp, pan)
    if "2" in sections:
        sec_dca_metrics(fit, vp, prev_train)
        print(f"sections 2-3 done ({time.time() - t0:.0f}s)", flush=True)
    if sections & {"4", "5"}:
        hh, ages, avail = raw_household()
    if "4" in sections:
        dec = sec_construct(fit, vp, hh, ages, avail)
        print("section 4 decision inputs:", dec, flush=True)
        print(f"section 4 done ({time.time() - t0:.0f}s)", flush=True)
    if "6" in sections:
        sec_p2()
        print(f"section 6 done ({time.time() - t0:.0f}s)", flush=True)
    if "7" in sections:
        sec_equiv(fit, vp)
        print(f"section 7 done ({time.time() - t0:.0f}s)", flush=True)
    if "5" in sections:
        sec_missing(fit, vp, hh, ages)
        print(f"section 5 done ({time.time() - t0:.0f}s)", flush=True)
    if sections >= {"2", "4", "5", "6", "7"}:
        pd.DataFrame(FLOW).to_csv(OUT / "sample_flow.csv", index=False)
    else:
        pd.DataFrame(FLOW).to_csv(OUT / f"sample_flow_partial_{'_'.join(sorted(sections))}.csv", index=False)
    print(pd.DataFrame(FLOW).to_string(index=False))


if __name__ == "__main__":
    main()
