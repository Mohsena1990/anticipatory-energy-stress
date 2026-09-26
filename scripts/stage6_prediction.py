"""
Stage 6: next-wave prediction (analysis_plan_rerun.md, amendment A6).

Transitions t -> t+1 linked by hrpid (duplicate hrpid within a wave excluded).
Outcome: high_fuel_vulnerable at t+1. Train: a->b ... l->m; validate: m->n, n->o.

  P0  benchmark: fuel_to_income_ratio_t, high_fuel_vulnerable_t
  P1  main: finnow, scghq1_dv, finfut_risk, dvage, heatch, lone_parent,
      large_family, workless_household, OBJECT/CONDITION/PERSONAL/ENERGY
      composites (all at t)
  P2  P1 + fes_magnitude_growth3 at the t+1 interview month (Dec Y_{t+1}-1
      vintage)
  P2b sensitivity: P2 + fes_delta_growth3 at t (own common sample; compared
      with P1 refitted on that sample)

Standardisation uses training-transition statistics only; resource
composites are rebuilt from items on the same basis. P0/P1/P2 share one
common sample. Unpenalised logistic regression.

Validation metrics: ROC-AUC and PR-AUC with 95% CIs (2,000 bootstrap
replicates resampling wave-t PSUs), calibration slope and
calibration-in-the-large, per-transition AUC, sensitivity and PPV at the top
5% and 10% of predicted risk, paired bootstrap delta AUC.

Outputs (outputs_v2/stage6/): sample_flow.csv, metrics.csv, delta_auc.csv,
  per_transition_auc.csv, calibration.csv, top_k.csv, coefficients.csv,
  thesis_table_prediction.csv, figures/roc.{png,pdf}, figures/calibration.{png,pdf}
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src.paths import OUTPUTS_DIR, UKHLS_PANEL  # noqa: E402
from stage4_resources import FACTORS, MONETARY  # noqa: E402

OUT = OUTPUTS_DIR / "stage6"
WAVES = "abcdefghijklmno"
VALIDATION = {"m->n", "n->o"}
N_BOOT = 2000
SEED = 20260926
NAVY, ACCENT, GREY, GRID = "#2E5077", "#D55E00", "#8A96A3", "#E6E9ED"

P0 = ["fuel_to_income_ratio", "high_fuel_vulnerable"]
P1 = ["finnow", "scghq1_dv", "finfut_risk", "dvage", "heatch", "lone_parent", "large_family",
      "workless_household", "OBJECT", "CONDITION", "PERSONAL", "ENERGY"]
FES_T1 = "fes_magnitude_growth3_t1"
FES_DELTA_T = "fes_delta_growth3"
MODELS = {"P0": P0, "P1": P1, "P2": P1 + [FES_T1]}
BINARY = {"high_fuel_vulnerable", "heatch", "lone_parent", "large_family", "workless_household"}
ITEMS = [i for v in FACTORS.values() for i in v]


def transitions() -> tuple[pd.DataFrame, list[dict]]:
    cols = (["hidp", "wave", "hrpid", "psu", "interview_year", "interview_month", "fuel_to_income_ratio",
             "high_fuel_vulnerable", "fes_magnitude_growth3", "fes_delta_growth3"]
            + [c for c in P1 if c not in FACTORS] + ITEMS)
    df = pd.read_csv(UKHLS_PANEL, usecols=lambda c: c in set(cols))
    flow, pairs = [], []
    for a, b in zip(WAVES[:-1], WAVES[1:]):
        t = df[df.wave == a].dropna(subset=["hrpid"]).drop_duplicates("hrpid", keep=False)
        t1 = df[df.wave == b].dropna(subset=["hrpid"]).drop_duplicates("hrpid", keep=False)
        m = t.merge(t1[["hrpid", "high_fuel_vulnerable", "fes_magnitude_growth3", "interview_year"]]
                    .rename(columns={"high_fuel_vulnerable": "y", "fes_magnitude_growth3": FES_T1,
                                     "interview_year": "interview_year_t1"}), on="hrpid")
        m["transition"] = f"{a}->{b}"
        pairs.append(m)
        flow.append(dict(step=f"linked transitions {a}->{b}", n=len(m), wave_t_households=len(df[df.wave == a])))
    return pd.concat(pairs, ignore_index=True), flow


def build_composites(tr: pd.DataFrame, va: pd.DataFrame) -> None:
    """Resource composites with training-transition statistics only."""
    x_tr, x_va = tr[ITEMS].astype(float).copy(), va[ITEMS].astype(float).copy()
    for c in MONETARY:
        x_tr[c], x_va[c] = np.log1p(x_tr[c].clip(lower=0)), np.log1p(x_va[c].clip(lower=0))
    mu, sd = x_tr.mean(), x_tr.std()
    z_tr, z_va = (x_tr - mu) / sd, (x_va - mu) / sd
    for f, items in FACTORS.items():
        def dom(z):
            return z[items].mean(axis=1).where(z[items].notna().sum(axis=1) >= len(items) / 2)
        d_tr, d_va = dom(z_tr), dom(z_va)
        m, s = d_tr.mean(), d_tr.std()
        tr[f], va[f] = (d_tr - m) / s, (d_va - m) / s


def standardise(tr: pd.DataFrame, va: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in cols:
        if c in BINARY or c in FACTORS:
            continue
        m, s = tr[c].mean(), tr[c].std()
        tr[c], va[c] = (tr[c] - m) / s, (va[c] - m) / s
        rows.append(dict(variable=c, train_mean=m, train_sd=s))
    return pd.DataFrame(rows)


def fit_predict(tr, va, cols):
    X = sm.add_constant(tr[cols].astype(float))
    res = sm.Logit(tr.y.astype(float), X).fit(disp=0, maxiter=200)
    p = res.predict(sm.add_constant(va[cols].astype(float), has_constant="add"))
    return res, np.asarray(p)


def topk(y, p, k):
    n = int(np.ceil(k * len(p)))
    idx = np.argsort(-p, kind="stable")[:n]
    tp = y[idx].sum()
    return tp / y.sum(), tp / n


def calibration(y, p):
    lp = np.log(p / (1 - p))
    slope = sm.Logit(y, sm.add_constant(lp)).fit(disp=0)
    citl = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=lp).fit()
    return dict(slope=slope.params[1], slope_ci_low=slope.conf_int()[1][0], slope_ci_high=slope.conf_int()[1][1],
                intercept=citl.params[0], intercept_ci_low=citl.conf_int()[0][0],
                intercept_ci_high=citl.conf_int()[0][1])


def bootstrap(va: pd.DataFrame, preds: dict[str, np.ndarray]):
    y = va.y.values.astype(int)
    psu_codes, uniq = pd.factorize(va.psu)
    rng = np.random.default_rng(SEED)
    stats = {m: {"auc": [], "prauc": [], "sens5": [], "ppv5": [], "sens10": [], "ppv10": []} for m in preds}
    per_tr = {(m, t): [] for m in preds for t in sorted(VALIDATION)}
    for _ in range(N_BOOT):
        w = rng.multinomial(len(uniq), np.full(len(uniq), 1 / len(uniq)))[psu_codes]
        keep = w > 0
        idx = np.repeat(np.arange(len(y)), w)
        for m, p in preds.items():
            stats[m]["auc"].append(roc_auc_score(y[keep], p[keep], sample_weight=w[keep]))
            stats[m]["prauc"].append(average_precision_score(y[keep], p[keep], sample_weight=w[keep]))
            s5, v5 = topk(y[idx], p[idx], 0.05)
            s10, v10 = topk(y[idx], p[idx], 0.10)
            stats[m]["sens5"].append(s5); stats[m]["ppv5"].append(v5)
            stats[m]["sens10"].append(s10); stats[m]["ppv10"].append(v10)
            for t in sorted(VALIDATION):
                mk = keep & (va.transition.values == t)
                per_tr[(m, t)].append(roc_auc_score(y[mk], p[mk], sample_weight=w[mk]))
    return stats, per_tr


def main() -> None:
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    pairs, flow = transitions()
    flow.append(dict(step="all linked transitions", n=len(pairs)))
    flow.append(dict(step="  of which outcome at t+1 missing", n=int(pairs.y.isna().sum())))
    tr_all = pairs[~pairs.transition.isin(VALIDATION)].copy()
    va_all = pairs[pairs.transition.isin(VALIDATION)].copy()
    build_composites(tr_all, va_all)
    allp = pd.concat([tr_all, va_all])

    need = ["y", "psu"] + P0 + P1 + [FES_T1]
    miss = allp[need].isna()
    for c in need:
        only = int((miss[c] & (miss.sum(axis=1) == 1)).sum())
        if miss[c].any():
            only_txt = "<10" if 0 < only < 10 else f"{only:,}"
            flow.append(dict(step=f"  missing {c} (only this missing: {only_txt})", n=int(miss[c].sum())))
    cs = ~miss.any(axis=1)
    flow.append(dict(step="common sample (P0/P1/P2)", n=int(cs.sum())))
    tr, va = tr_all[cs.loc[tr_all.index]].copy(), va_all[cs.loc[va_all.index]].copy()
    flow += [dict(step="  training transitions (a->b ... l->m)", n=len(tr)),
             dict(step="  validation transitions (m->n, n->o)", n=len(va)),
             dict(step="  validation prevalence (%)", n=round(100 * va.y.mean(), 2))]
    scal = standardise(tr, va, sorted(set(P0 + P1 + [FES_T1])))
    scal.to_csv(OUT / "standardisation_train_stats.csv", index=False)

    preds, coefs = {}, []
    for name, cols in MODELS.items():
        res, p = fit_predict(tr, va, cols)
        preds[name] = p
        ci = res.conf_int()
        for t in res.params.index:
            coefs.append(dict(model=name, term=t, coef=res.params[t], OR=np.exp(res.params[t]),
                              OR_ci_low=np.exp(ci.loc[t, 0]), OR_ci_high=np.exp(ci.loc[t, 1]),
                              p=res.pvalues[t], n_train=int(res.nobs)))
    pd.DataFrame(coefs).to_csv(OUT / "coefficients.csv", index=False)

    y = va.y.values.astype(int)
    stats, per_tr = bootstrap(va, preds)
    ci = lambda a: (np.percentile(a, 2.5), np.percentile(a, 97.5))  # noqa: E731
    met, cal, topr = [], [], []
    for m, p in preds.items():
        pt = dict(auc=roc_auc_score(y, p), prauc=average_precision_score(y, p))
        s5, v5 = topk(y, p, 0.05)
        s10, v10 = topk(y, p, 0.10)
        pt.update(sens5=s5, ppv5=v5, sens10=s10, ppv10=v10)
        for k, v in pt.items():
            lo, hi = ci(stats[m][k])
            met.append(dict(model=m, metric=k, estimate=v, ci_low=lo, ci_high=hi, n_validation=len(y),
                            prevalence=y.mean()))
        cal.append(dict(model=m, **calibration(y, p)))
    metrics = pd.DataFrame(met)
    metrics.to_csv(OUT / "metrics.csv", index=False)
    pd.DataFrame(cal).to_csv(OUT / "calibration.csv", index=False)

    d_rows = []
    for a, b in [("P1", "P0"), ("P2", "P1"), ("P2", "P0")]:
        d = np.array(stats[a]["auc"]) - np.array(stats[b]["auc"])
        d_rows.append(dict(comparison=f"{a} - {b}", delta_auc=roc_auc_score(y, preds[a]) - roc_auc_score(y, preds[b]),
                           ci_low=np.percentile(d, 2.5), ci_high=np.percentile(d, 97.5),
                           p_two_sided=2 * min((d <= 0).mean(), (d >= 0).mean())))
    pd.DataFrame(d_rows).to_csv(OUT / "delta_auc.csv", index=False)

    pt_rows = []
    for (m, t), arr in per_tr.items():
        mk = va.transition.values == t
        pt_rows.append(dict(model=m, transition=t, n=int(mk.sum()), prevalence=y[mk].mean(),
                            auc=roc_auc_score(y[mk], preds[m][mk]), ci_low=np.percentile(arr, 2.5),
                            ci_high=np.percentile(arr, 97.5)))
    pd.DataFrame(pt_rows).to_csv(OUT / "per_transition_auc.csv", index=False)

    # P2b sensitivity: own common sample, compared with P1 on that sample.
    need_b = need + [FES_DELTA_T]
    csb = ~allp[need_b].isna().any(axis=1)
    trb = tr_all[csb.loc[tr_all.index]].copy()
    vab = va_all[csb.loc[va_all.index]].copy()
    standardise(trb, vab, sorted(set(P1 + [FES_T1, FES_DELTA_T])))
    _, p1b = fit_predict(trb, vab, P1)
    _, p2b = fit_predict(trb, vab, P1 + [FES_T1, FES_DELTA_T])
    yb = vab.y.values.astype(int)
    flow.append(dict(step="P2b common sample (adds FES Delta at t; loses 2009 wave-t interviews)", n=int(csb.sum())))
    pd.DataFrame([dict(model="P1 (P2b sample)", auc=roc_auc_score(yb, p1b), n_train=len(trb), n_validation=len(vab)),
                  dict(model="P2b", auc=roc_auc_score(yb, p2b), n_train=len(trb), n_validation=len(vab))]
                 ).to_csv(OUT / "p2b_sensitivity.csv", index=False)
    pd.DataFrame(flow).to_csv(OUT / "sample_flow.csv", index=False)

    thesis_table(metrics, pd.DataFrame(cal), pd.DataFrame(d_rows))
    figures(y, preds, va)
    pd.set_option("display.width", 220)
    print(pd.DataFrame(flow).to_string(index=False))
    print(metrics.round(4).to_string(index=False))
    print(pd.DataFrame(cal).round(3).to_string(index=False))
    print(pd.DataFrame(d_rows).round(4).to_string(index=False))
    print(pd.DataFrame(pt_rows).round(4).to_string(index=False))
    print(pd.read_csv(OUT / "p2b_sensitivity.csv").round(4).to_string(index=False))


def thesis_table(metrics, cal, delta) -> None:
    rows = []
    for m in MODELS:
        g = metrics[metrics.model == m].set_index("metric")
        c = cal.set_index("model").loc[m]
        f = lambda k, pct=False: (f"{100*g.loc[k,'estimate']:.1f} [{100*g.loc[k,'ci_low']:.1f}, {100*g.loc[k,'ci_high']:.1f}]"  # noqa: E731
                                  if pct else f"{g.loc[k,'estimate']:.3f} [{g.loc[k,'ci_low']:.3f}, {g.loc[k,'ci_high']:.3f}]")
        rows.append({"Model": {"P0": "P0 current burden (benchmark)", "P1": "P1 household predictors",
                               "P2": "P2 = P1 + FES (growth-only magnitude, t+1)"}[m],
                     "ROC-AUC [95% CI]": f("auc"), "PR-AUC [95% CI]": f("prauc"),
                     "Calibration slope": f"{c.slope:.2f} [{c.slope_ci_low:.2f}, {c.slope_ci_high:.2f}]",
                     "Calibration-in-the-large": f"{c.intercept:.2f} [{c.intercept_ci_low:.2f}, {c.intercept_ci_high:.2f}]",
                     "Top 5%: sensitivity / PPV (%)": f"{f('sens5', True)} / {f('ppv5', True)}",
                     "Top 10%: sensitivity / PPV (%)": f"{f('sens10', True)} / {f('ppv10', True)}",
                     "n validation": int(g.iloc[0].n_validation),
                     "prevalence (%)": round(100 * g.iloc[0].prevalence, 2)})
    pd.DataFrame(rows).to_csv(OUT / "thesis_table_prediction.csv", index=False)


def figures(y, preds, va) -> None:
    col = {"P0": GREY, "P1": NAVY, "P2": ACCENT}
    lab = {"P0": "P0 current burden (benchmark)", "P1": "P1 household predictors", "P2": "P2 = P1 + FES"}
    ls = {"P0": "--", "P1": "-", "P2": ":"}
    fig, ax = plt.subplots(figsize=(5.2, 5))
    for m, p in preds.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=col[m], lw=2, ls=ls[m], label=f"{lab[m]} (AUC {roc_auc_score(y, p):.3f})")
    ax.plot([0, 1], [0, 1], color="#C5CCD3", lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    ax.grid(color=GRID, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Next-wave fuel vulnerability, validation transitions m→n, n→o", fontsize=8.5, loc="left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / "figures" / f"roc.{ext}", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.2, 5))
    top = 0.0
    for m, p in preds.items():
        d = pd.DataFrame({"p": p, "y": y})
        d["bin"] = pd.qcut(d.p.rank(method="first"), 10, labels=False)
        g = d.groupby("bin").agg(p=("p", "mean"), y=("y", "mean"))
        top = max(top, g.p.max(), g.y.max())
        ax.plot(100 * g.p, 100 * g.y, color=col[m], lw=1.6, ls=ls[m], marker="o", ms=4, label=lab[m])
    lim = 100 * top * 1.08
    ax.plot([0, lim], [0, lim], color="#C5CCD3", lw=0.8)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Mean predicted risk by decile (%)")
    ax.set_ylabel("Observed next-wave vulnerability (%)")
    ax.legend(frameon=False, fontsize=7.5, loc="upper left")
    ax.grid(color=GRID, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / "figures" / f"calibration.{ext}", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
