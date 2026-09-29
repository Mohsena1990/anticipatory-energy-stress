"""
Stage 8b (analysis_plan_rerun.md, amendment A8; EXPLORATORY, post hoc).

Conditional tests of the anticipatory burden projection (A7):

  idea 6  forward-chaining validation: every transition from e->f to n->o is
          predicted by models fitted on all earlier transitions only;
  idea 1  shock vs calm origins (ex-ante: |forecast window ratio| >= 0.10 for
          gas or electricity): H8.1 incident AUC(A_fut) - AUC(A_naive) at shock
          origins; H8.2a/b step-1 forecast vs no-change at shock origins (DM);
  idea 2  separate terms (log b_t, log M, log income ratio): H8.3 on the A6 split;
  idea 4  decision metrics: H8.4 top-10% sensitivity among newly vulnerable
          households, A_fut vs A_naive (A6 split); prevalence error by year.

Holm correction over H8.1, H8.2a, H8.2b, H8.3, H8.4.

Outputs (outputs_v3/anticipation_a8/, aggregate tables only): hypotheses.csv,
forward_chain_auc.csv, fold_auc.csv, shock_sensitivity.csv, price_shock_accuracy.csv,
split_metrics.csv, decision_metrics.csv, prevalence_error.csv, calibration.csv,
figures/fold_gain.{png,pdf}
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from stage6_prediction import (P0, P1, VALIDATION, WAVES, build_composites,  # noqa: E402
                               calibration, fit_predict, standardise)
from stage8_anticipation import build_pairs, logb  # noqa: E402
from stage8_price_forecast import OUT as A7_OUT, dm_test  # noqa: E402

OUT = ROOT / "outputs_v3" / "anticipation_a8"
N_BOOT = 2000
SEED = 20260929
SHOCK = 0.10
SHOCK_SENS = (0.05, 0.20)
FIRST_FOLD = "e->f"
FLAG = ["high_fuel_vulnerable"]
SPECS = {
    "P0": P0,
    "P3 = P0 + P1": P0 + P1,
    "A_naive": ["logb_naive"] + FLAG,
    "A_fut": ["logb_forecast"] + FLAG,
    "A_sep_naive": ["logb_t", "logM_naive", "log_income_ratio"] + FLAG,
    "A_sep_fut": ["logb_t", "logM_forecast", "log_income_ratio"] + FLAG,
}
NEED = sorted({c for v in SPECS.values() for c in v} | {"y", "psu"})


def shock_origins(threshold: float) -> set:
    pf = pd.read_csv(A7_OUT / "price_forecasts.csv", parse_dates=["origin"])
    w = pf[pf.series.isin(["gas", "electricity"])].pivot(index="origin", columns="series", values="ratio_forecast")
    return set(w.index[(w.abs() >= threshold).any(axis=1)])


def fit_fold(tr: pd.DataFrame, va: pd.DataFrame) -> pd.DataFrame:
    """Fit every SPECS model on tr (training-only statistics), predict va."""
    tr, va = tr.copy(), va.copy()
    build_composites(tr, va)
    tr, va = tr.dropna(subset=NEED), va.dropna(subset=NEED)
    raw_cols = ["y", "psu", "high_fuel_vulnerable", "T", "transition", "interview_year_t1", "shock"]
    out = va[raw_cols].copy()
    standardise(tr, va, sorted({c for v in SPECS.values() for c in v}))
    for name, cols in SPECS.items():
        _, p = fit_predict(tr, va, cols)
        out[name] = p
    out["n_train"] = len(tr)
    return out


def boot_auc_diff(d: pd.DataFrame, m1: str, m2: str, mask: np.ndarray, rng) -> np.ndarray:
    y = d.y.values.astype(int)
    codes, uniq = pd.factorize(d.psu)
    out = []
    for _ in range(N_BOOT):
        w = rng.multinomial(len(uniq), np.full(len(uniq), 1 / len(uniq)))[codes]
        k = mask & (w > 0)
        out.append(roc_auc_score(y[k], d[m1].values[k], sample_weight=w[k])
                   - roc_auc_score(y[k], d[m2].values[k], sample_weight=w[k]))
    return np.array(out)


def top10_incident_sens(y, p, inc, w):
    """Sensitivity among incident positives when the top 10% (weighted) of all
    households are flagged."""
    order = np.argsort(-p, kind="stable")
    cw = np.cumsum(w[order])
    flagged = np.zeros(len(p), bool)
    flagged[order[cw <= 0.10 * cw[-1]]] = True
    pos = (y == 1) & inc
    return (w * (flagged & pos)).sum() / (w * pos).sum(), (w * (flagged & pos)).sum() / w.sum()


def boot_sens_diff(d, m1, m2, rng):
    y = d.y.values.astype(int)
    inc = d.high_fuel_vulnerable.values == 0
    codes, uniq = pd.factorize(d.psu)
    out = []
    for _ in range(N_BOOT):
        w = rng.multinomial(len(uniq), np.full(len(uniq), 1 / len(uniq)))[codes].astype(float)
        out.append(top10_incident_sens(y, d[m1].values, inc, w)[0] - top10_incident_sens(y, d[m2].values, inc, w)[0])
    return np.array(out)


def summarise(name, point, boot):
    return dict(test=name, estimate=point, ci_low=np.percentile(boot, 2.5), ci_high=np.percentile(boot, 97.5),
                p_one_sided=max((boot <= 0).mean(), 1 / N_BOOT))


def holm(p: pd.Series) -> pd.Series:
    order = p.sort_values().index
    adj, running = {}, 0.0
    m = len(p)
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * p[k]))
        adj[k] = running
    return pd.Series(adj)[p.index]


def prevalence_error(d: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    rows = []
    for yr, g in d.groupby("interview_year_t1"):
        if len(g) < 100:
            continue
        rows.append(dict(interview_year_t1=int(yr), n=len(g), actual=g.y.mean(), **{m: g[m].mean() for m in models}))
    return pd.DataFrame(rows)


def main() -> None:
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    pairs, _ = build_pairs()
    pairs["logM_forecast"] = np.log(pairs.multiplier_forecast)
    pairs["logM_naive"] = np.log(pairs.multiplier_naive)
    pairs["log_income_ratio"] = np.log(pairs.income_ratio.where(pairs.income_ratio > 0))
    shocks = shock_origins(SHOCK)
    pairs["shock"] = pairs["T"].isin(shocks)
    order = [f"{a}->{b}" for a, b in zip(WAVES[:-1], WAVES[1:])]

    # ── idea 6: forward chaining ────────────────────────────────────────────
    folds = []
    for k in order[order.index(FIRST_FOLD):]:
        prior = order[:order.index(k)]
        folds.append(fit_fold(pairs[pairs.transition.isin(prior)], pairs[pairs.transition == k]))
        print(f"fold {k}: validation n={len(folds[-1])}", flush=True)
    fc = pd.concat(folds, ignore_index=True)
    inc = fc.high_fuel_vulnerable.values == 0

    fold_rows = []
    for k, g in fc.groupby("transition", sort=False):
        gi = g[g.high_fuel_vulnerable == 0]
        fold_rows.append(dict(transition=k, n=len(g), n_incident=len(gi), share_shock=g.shock.mean(),
                              prevalence_incident=gi.y.mean(),
                              auc_inc_A_fut=roc_auc_score(gi.y, gi.A_fut), auc_inc_A_naive=roc_auc_score(gi.y, gi.A_naive),
                              auc_inc_P0=roc_auc_score(gi.y, gi.P0),
                              gain_inc_fut_vs_naive=roc_auc_score(gi.y, gi.A_fut) - roc_auc_score(gi.y, gi.A_naive)))
    fold_df = pd.DataFrame(fold_rows)

    fc_rows, hyp = [], []
    for stratum, smask in {"shock": fc.shock.values, "calm": ~fc.shock.values}.items():
        for sub, m in {"incident": smask & inc, "all": smask}.items():
            y = fc.y.values[m]
            row = dict(stratum=stratum, subset=sub, n=int(m.sum()), prevalence=y.mean())
            for mod in SPECS:
                row[f"auc_{mod}"] = roc_auc_score(y, fc[mod].values[m])
            b = boot_auc_diff(fc, "A_fut", "A_naive", m, rng)
            s = summarise("", row["auc_A_fut"] - row["auc_A_naive"], b)
            row.update(delta_fut_vs_naive=s["estimate"], ci_low=s["ci_low"], ci_high=s["ci_high"], p_one_sided=s["p_one_sided"])
            fc_rows.append(row)
            if stratum == "shock" and sub == "incident":
                hyp.append(summarise("H8.1 forward-chained, shock origins, incident: AUC A_fut - A_naive",
                                     s["estimate"], b))
    fc_auc = pd.DataFrame(fc_rows)

    sens_rows = []
    for th in SHOCK_SENS:
        so = fc["T"].isin(shock_origins(th)).values
        for stratum, sm in {"shock": so, "calm": ~so}.items():
            m = sm & inc
            sens_rows.append(dict(threshold=th, stratum=stratum, n_incident=int(m.sum()),
                                  delta_fut_vs_naive=roc_auc_score(fc.y[m], fc.A_fut[m]) - roc_auc_score(fc.y[m], fc.A_naive[m])))
    sens_df = pd.DataFrame(sens_rows)

    # ── idea 1: price accuracy at shock origins ─────────────────────────────
    pf = pd.read_csv(A7_OUT / "price_forecasts.csv", parse_dates=["origin"])
    price_rows = []
    for series in ("gas", "electricity"):
        for stratum in ("shock", "calm"):
            g = pf[(pf.series == series) & pf.ratio_realised.notna() & (pf.origin >= "2009-01-01")]
            g = g[g.origin.isin(shocks) == (stratum == "shock")]
            ef, en = (g.ratio_forecast - g.ratio_realised).values, (g.ratio_naive - g.ratio_realised).values
            stat, _ = dm_test(ef, en)
            p1 = float(stats.t.cdf(stat, df=len(ef) - 1))
            price_rows.append(dict(series=series, stratum=stratum, n_origins=len(g),
                                   rmse_forecast=np.sqrt(np.mean(ef ** 2)), rmse_naive=np.sqrt(np.mean(en ** 2)),
                                   rel_rmse=np.sqrt(np.mean(ef ** 2)) / np.sqrt(np.mean(en ** 2)),
                                   dm_stat=stat, p_one_sided=p1))
            if stratum == "shock":
                hyp.append(dict(test=f"H8.2{'a' if series == 'gas' else 'b'} step-1 {series}, shock origins: forecast beats no-change (DM)",
                                estimate=price_rows[-1]["rel_rmse"], ci_low=np.nan, ci_high=np.nan, p_one_sided=p1))
    price_df = pd.DataFrame(price_rows)

    # ── ideas 2 and 4: A6 split ─────────────────────────────────────────────
    sp = fit_fold(pairs[~pairs.transition.isin(VALIDATION)], pairs[pairs.transition.isin(VALIDATION)])
    sinc = sp.high_fuel_vulnerable.values == 0
    split_rows = []
    for sub, m in {"all": np.ones(len(sp), bool), "incident": sinc}.items():
        for mod in SPECS:
            split_rows.append(dict(subset=sub, model=mod, n=int(m.sum()), auc=roc_auc_score(sp.y[m], sp[mod][m])))
    split_df = pd.DataFrame(split_rows)
    a = lambda mod: roc_auc_score(sp.y[sinc], sp[mod][sinc])  # noqa: E731
    b = boot_auc_diff(sp, "A_sep_fut", "A_sep_naive", sinc, rng)
    hyp.append(summarise("H8.3 A6 split, incident: AUC A_sep_fut - A_sep_naive", a("A_sep_fut") - a("A_sep_naive"), b))
    b2 = boot_auc_diff(sp, "A_sep_fut", "A_fut", sinc, rng)
    extra = [summarise("(reported) A6 split, incident: AUC A_sep_fut - A_fut", a("A_sep_fut") - a("A_fut"), b2)]

    y, w1 = sp.y.values.astype(int), np.ones(len(sp))
    dec_rows = []
    for mod in SPECS:
        sens, per = top10_incident_sens(y, sp[mod].values, sinc, w1)
        dec_rows.append(dict(model=mod, top10_sensitivity_incident=sens, incident_found_per_1000_screened=1000 * per,
                             incident_positives=int(((y == 1) & sinc).sum()),
                             incident_found=int(round(per * len(sp)))))
    dec_df = pd.DataFrame(dec_rows).set_index("model")
    bs = boot_sens_diff(sp, "A_fut", "A_naive", rng)
    hyp.append(summarise("H8.4 A6 split: top-10% sensitivity among incident, A_fut - A_naive",
                         dec_df.loc["A_fut", "top10_sensitivity_incident"] - dec_df.loc["A_naive", "top10_sensitivity_incident"], bs))
    bs0 = boot_sens_diff(sp, "A_fut", "P0", rng)
    extra.append(summarise("(reported) A6 split: top-10% sensitivity among incident, A_fut - P0",
                           dec_df.loc["A_fut", "top10_sensitivity_incident"] - dec_df.loc["P0", "top10_sensitivity_incident"], bs0))

    cal = [dict(model=m, **calibration(sp.y.values.astype(int), np.clip(sp[m].values, 1e-6, 1 - 1e-6))) for m in SPECS]
    models_prev = ["P0", "P3 = P0 + P1", "A_naive", "A_fut", "A_sep_fut"]
    prev_rows = []
    for design, d in {"A6 split": sp, "forward-chained": fc}.items():
        pe = prevalence_error(d, models_prev)
        pe.insert(0, "design", design)
        prev_rows.append(pe)
    prev = pd.concat(prev_rows, ignore_index=True)
    mae = prev.groupby("design").apply(lambda g: pd.Series({m: (g[m] - g.actual).abs().mean() for m in models_prev}))

    hyp = pd.DataFrame(hyp)
    hyp["p_holm"] = holm(hyp.p_one_sided).values
    hyp["survives_holm"] = hyp.p_holm < 0.05
    rule = bool(hyp[hyp.test.str.startswith(("H8.1", "H8.2a", "H8.2b"))].survives_holm.all())
    hyp = pd.concat([hyp, pd.DataFrame(extra),
                     pd.DataFrame([dict(test="DECISION: anticipation useful when a price shock is forecast (H8.1, H8.2a, H8.2b survive Holm)",
                                        estimate=float(rule))])], ignore_index=True)

    hyp.to_csv(OUT / "hypotheses.csv", index=False)
    fc_auc.to_csv(OUT / "forward_chain_auc.csv", index=False)
    fold_df.to_csv(OUT / "fold_auc.csv", index=False)
    sens_df.to_csv(OUT / "shock_sensitivity.csv", index=False)
    price_df.to_csv(OUT / "price_shock_accuracy.csv", index=False)
    split_df.to_csv(OUT / "split_metrics.csv", index=False)
    dec_df.reset_index().to_csv(OUT / "decision_metrics.csv", index=False)
    prev.to_csv(OUT / "prevalence_error.csv", index=False)
    mae.to_csv(OUT / "prevalence_mae.csv")
    pd.DataFrame(cal).to_csv(OUT / "calibration.csv", index=False)
    plot_folds(fold_df)

    pd.set_option("display.width", 220)
    for t in (hyp, fc_auc, fold_df, sens_df, price_df, split_df, dec_df, pd.DataFrame(cal), prev, mae):
        print(t.round(4).to_string())
        print()


def plot_folds(fold_df: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = np.arange(len(fold_df))
    ax.bar(x, 1000 * fold_df.gain_inc_fut_vs_naive,
           color=np.where(fold_df.share_shock >= 0.5, "#D55E00", "#8A96A3"))
    ax.axhline(0, color="#555555", lw=0.8)
    ax.set_xticks(x, fold_df.transition)
    ax.set_ylabel("AUC gain from price forecast\n(x 1000, newly vulnerable)")
    ax.set_xlabel("Forward-chained validation transition (orange: most interviews at shock origins)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / "figures" / f"fold_gain.{ext}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
