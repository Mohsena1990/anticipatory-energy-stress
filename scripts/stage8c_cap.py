"""
Stage 8c (analysis_plan_rerun.md, amendment A9; EXPLORATORY, post hoc).

Adds Ofgem's announced price caps to the step-1 retail price forecast and re-tests
whether forecasting helps identify newly vulnerable households.

  F_cap      months in a cap period known at origin T follow the cap (with the
             Energy Price Guarantee where it applied and was known); later months
             follow the A7 futures path from the last known cap month.
  F_capflat  known cap months, then flat (announcements only, no market forecast).

A cap period counts as known from T = first month - 1: every cap since 2019 was
announced at least ~5 weeks before it started, so this is never early.

Tests (Holm over five): H9.1a/b F_cap vs no-change, H9.2a/b F_cap vs F_capflat
(gas/electricity, origins T >= 2019-03, one-sided DM, HAC lag 12); H9.3
forward-chained incident AUC(A_cap) - AUC(A_naive), T >= 2019-03.

Outputs (outputs_v3/anticipation_a9/, aggregate tables only): cap_monthly.csv,
price_forecasts_cap.csv, price_accuracy_cap.csv, price_accuracy_cap_by_horizon.csv,
hypotheses.csv, forward_chain_auc.csv, split_metrics.csv, calibration.csv,
figures/cap_window_ratio.{png,pdf}
"""
from __future__ import annotations

import re
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
import stage8b_conditional as s8b  # noqa: E402
from stage6_prediction import VALIDATION, WAVES, calibration  # noqa: E402
from stage8_anticipation import build_pairs, logb  # noqa: E402
from stage8_price_forecast import (EPG_SCHEDULE, EXT, H_MAX, MIN_ERRORS, OUT as A7_OUT,  # noqa: E402
                                   _at, dm_test, load_prices, window_ratio)

OUT = ROOT / "outputs_v3" / "anticipation_a9"
CAP_FILE = EXT / "ofgem_default_tariff_cap_level_v1.31.xlsx"
CAP_START = pd.Timestamp("2019-01-01")
TEST_FROM = pd.Timestamp("2019-03-01")
FUELS = ("gas", "electricity")
# Published typical dual-fuel cap (old typical consumption) in the EPG periods.
PUBLISHED_DUAL = {("2022-10-01", "2022-12-01"): 3549, ("2023-01-01", "2023-03-01"): 4279,
                  ("2023-04-01", "2023-06-01"): 3280}
MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun",
                                       "jul", "aug", "sep", "oct", "nov", "dec"], 1)}


# ── caps ─────────────────────────────────────────────────────────────────────

def _parse_month(txt: str) -> pd.Timestamp:
    mon, yr = re.match(r"([A-Za-z]+)\s+(\d{4})", txt.strip()).groups()
    return pd.Timestamp(year=int(yr), month=MONTHS[mon[:3].lower()], day=1)


def load_caps() -> pd.DataFrame:
    """Monthly direct-debit typical-consumption cap level (incl. VAT) per fuel."""
    d = pd.read_excel(CAP_FILE, sheet_name="1b Historical level tables", header=None)
    blocks = {"electricity": (2, 32), "gas": (64, 94)}
    rows = []
    for fuel, (c0, c1) in blocks.items():
        assert str(d.iat[9, c0 - 1 if fuel == "electricity" else c0 - 1]).startswith(
            "Electricity: Single" if fuel == "electricity" else "Gas"), f"unexpected layout for {fuel}"
        assert d.iat[29, 1] == "Typical consumption" and d.iat[45, 1] == "Total inc VAT"
        for label, level in zip(d.iloc[29, c0:c1], d.iloc[45, c0:c1]):
            if not isinstance(label, str):
                continue
            a, b = label.split(" - ")
            first, last = _parse_month(a), _parse_month(b)
            for m in pd.date_range(first, last, freq="MS"):
                rows.append(dict(fuel=fuel, month=m, period_first=first, known_from=first - pd.DateOffset(months=1),
                                 cap=float(level)))
    caps = pd.DataFrame(rows)
    return caps[caps.month >= CAP_START].reset_index(drop=True)


def epg_level(origin: pd.Timestamp, month: pd.Timestamp) -> int | None:
    level = None
    for known_from, first, last, lvl in EPG_SCHEDULE:  # later announcements override
        if origin >= pd.Timestamp(known_from) and pd.Timestamp(first) <= month <= pd.Timestamp(last):
            level = lvl
    return level


def effective(cap: float, month: pd.Timestamp, origin: pd.Timestamp) -> float:
    lvl = epg_level(origin, month)
    if lvl is None:
        return cap
    for (a, b), dual in PUBLISHED_DUAL.items():
        if pd.Timestamp(a) <= month <= pd.Timestamp(b):
            return cap * min(1.0, lvl / dual)
    return cap


# ── forecasts ────────────────────────────────────────────────────────────────

def cap_forecasts(px: dict, a7: pd.DataFrame, caps: pd.DataFrame) -> pd.DataFrame:
    mo = pd.DateOffset(months=1)
    rows = []
    for fuel in FUELS:
        p = px[fuel]
        last_cpi = p.index.max()
        cf = caps[caps.fuel == fuel].set_index("month")
        for r in a7[a7.series == fuel].itertuples():
            T = r.origin
            L = T - 2 * mo
            pL = _at(p, L)
            fut = {L + h * mo: pL + getattr(r, f"h{h}_forecast") for h in range(1, H_MAX + 1)}
            cap_path, flat_path = dict(fut), {m: pL for m in fut}
            n_known = 0
            if L >= CAP_START and L in cf.index:
                eff_L = effective(cf.at[L, "cap"], L, T)
                m_star = L
                for h in range(1, H_MAX + 1):
                    m = L + h * mo
                    if m in cf.index and cf.at[m, "known_from"] <= T:
                        cap_path[m] = flat_path[m] = pL + np.log(effective(cf.at[m, "cap"], m, T) / eff_L)
                        m_star, n_known = m, h
                    else:
                        break
                base_fut = pL if m_star == L else fut[m_star]
                for h in range(n_known + 1, H_MAX + 1):
                    m = L + h * mo
                    cap_path[m] = cap_path[m_star] + fut[m] - base_fut if m_star != L else fut[m]
                    flat_path[m] = flat_path[m_star] if m_star != L else pL
            realised = r.ratio_realised
            rows.append(dict(series=fuel, origin=T, fallback=r.fallback, n_cap_months_known=n_known,
                             ratio_forecast=window_ratio(cap_path, p, T, known_to=L),
                             ratio_capflat=window_ratio(flat_path, p, T, known_to=L),
                             ratio_futures=r.ratio_forecast, ratio_naive=r.ratio_naive, ratio_realised=realised,
                             **{f"h{h}_cap": cap_path[L + h * mo] - pL for h in range(1, H_MAX + 1)},
                             **{f"h{h}_fut": fut[L + h * mo] - pL for h in range(1, H_MAX + 1)},
                             **{f"h{h}_realised": _at(p, L + h * mo) - pL if L + h * mo <= last_cpi else np.nan
                                for h in range(1, H_MAX + 1)}))
    df = pd.DataFrame(rows).sort_values(["series", "origin"]).reset_index(drop=True)
    sig = []
    for _, g in df.groupby("series", sort=False):
        for _, r in g.iterrows():
            known = g[(g.origin + pd.DateOffset(months=11) <= r.origin - 2 * mo) & g.ratio_realised.notna()]
            e = (known.ratio_forecast - known.ratio_realised).values
            if len(e) >= MIN_ERRORS:
                sig.append(np.sqrt(np.mean(e ** 2)))
            else:
                sig.append(float(px[r.series].loc[:r.origin - 2 * mo].diff(12).dropna().std()))
    df["sigma_ratio"] = sig
    return df


def one_sided_dm(e1, e2) -> tuple[float, float]:
    stat, _ = dm_test(e1, e2)
    return stat, float(stats.t.cdf(stat, df=len(e1) - 1))


def price_accuracy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    rows, hyp = [], []
    periods = {"cap regime (T >= 2019-03)": df.origin >= TEST_FROM, "all 2009+": df.origin >= "2009-01-01"}
    comps = {"F_cap vs no-change": ("ratio_forecast", "ratio_naive"),
             "F_cap vs F_capflat": ("ratio_forecast", "ratio_capflat"),
             "F_capflat vs no-change": ("ratio_capflat", "ratio_naive"),
             "F_futures vs no-change": ("ratio_futures", "ratio_naive"),
             "F_cap vs F_futures": ("ratio_forecast", "ratio_futures")}
    for fuel in FUELS:
        for pname, pmask in periods.items():
            g = df[pmask & (df.series == fuel) & df.ratio_realised.notna()]
            for cname, (a, b) in comps.items():
                e1, e2 = (g[a] - g.ratio_realised).values, (g[b] - g.ratio_realised).values
                stat, p1 = one_sided_dm(e1, e2)
                rows.append(dict(series=fuel, period=pname, comparison=cname, n_origins=len(g),
                                 rmse_a=np.sqrt(np.mean(e1 ** 2)), rmse_b=np.sqrt(np.mean(e2 ** 2)),
                                 rel_rmse=np.sqrt(np.mean(e1 ** 2)) / np.sqrt(np.mean(e2 ** 2)),
                                 dm_stat=stat, p_one_sided=p1))
                if pname.startswith("cap regime") and cname in ("F_cap vs no-change", "F_cap vs F_capflat"):
                    tag = ("H9.1" if cname == "F_cap vs no-change" else "H9.2") + ("a" if fuel == "gas" else "b")
                    hyp.append(dict(test=f"{tag} {fuel}, T >= 2019-03: {cname} (DM)", estimate=rows[-1]["rel_rmse"],
                                    ci_low=np.nan, ci_high=np.nan, p_one_sided=p1))
    hrows = []
    for fuel in FUELS:
        g = df[(df.series == fuel) & (df.origin >= TEST_FROM)]
        for h in range(1, H_MAX + 1):
            gh = g.dropna(subset=[f"h{h}_realised"])
            real = gh[f"h{h}_realised"]
            rm = lambda e: float(np.sqrt((e ** 2).mean()))  # noqa: E731
            hrows.append(dict(series=fuel, h=h, n=len(gh), share_cap_known=(gh.n_cap_months_known >= h).mean(),
                              rmse_cap=rm(gh[f"h{h}_cap"] - real), rmse_futures=rm(gh[f"h{h}_fut"] - real),
                              rmse_naive=rm(-real), rel_cap_vs_naive=rm(gh[f"h{h}_cap"] - real) / rm(-real)))
    return pd.DataFrame(rows), pd.DataFrame(hrows), hyp


def plot(df: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for ax, s in zip(axes, FUELS):
        g = df[(df.series == s) & (df.origin >= "2018-01-01")]
        ax.plot(g.origin, 100 * np.expm1(g.ratio_realised), color="#1f1f1f", lw=1.8, label="Realised")
        ax.plot(g.origin, 100 * np.expm1(g.ratio_forecast), color="#D55E00", lw=1.4, label="Caps + futures")
        ax.plot(g.origin, 100 * np.expm1(g.ratio_futures), color="#2E5077", lw=1.1, ls=":", label="Futures only")
        ax.plot(g.origin, 100 * np.expm1(g.ratio_naive), color="#8A96A3", lw=1.1, ls="--", label="No change")
        ax.axhline(0, color="#cccccc", lw=0.8)
        ax.axvline(CAP_START, color="#cccccc", lw=0.8, ls="--")
        ax.set_ylabel(f"{s.capitalize()}: % change,\nnext 12 m vs last 12 m")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, ncol=4, loc="upper left", fontsize=8)
    axes[1].set_xlabel("Forecast origin (interview month)")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / "figures" / f"cap_window_ratio.{ext}", dpi=200)
    plt.close(fig)


# ── households ───────────────────────────────────────────────────────────────

def main() -> None:
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    px = load_prices()
    a7 = pd.read_csv(A7_OUT / "price_forecasts.csv", parse_dates=["origin"])
    caps = load_caps()
    caps.to_csv(OUT / "cap_monthly.csv", index=False)
    capf = cap_forecasts(px, a7, caps)
    acc, acc_h, hyp = price_accuracy(capf)
    acc.to_csv(OUT / "price_accuracy_cap.csv", index=False)
    acc_h.to_csv(OUT / "price_accuracy_cap_by_horizon.csv", index=False)
    plot(capf)

    # step-1 file for build_pairs: cap ratios for gas/electricity, A7 rows for other fuels
    keep = ["series", "origin", "fallback", "ratio_forecast", "ratio_naive", "ratio_realised", "sigma_ratio"]
    price_file = OUT / "price_forecasts_cap.csv"
    pd.concat([capf[keep], a7[~a7.series.isin(FUELS)][keep]]).to_csv(price_file, index=False)

    pairs, _ = build_pairs()
    pairs_cap, _ = build_pairs(price_file)
    assert (pairs.hrpid.values == pairs_cap.hrpid.values).all() and (pairs.transition.values == pairs_cap.transition.values).all()
    pairs["logb_cap"] = logb(pairs_cap.burden_forecast)
    pairs["shock"] = pairs["T"].isin(s8b.shock_origins(s8b.SHOCK))

    flag = ["high_fuel_vulnerable"]
    s8b.SPECS = {"P0": s8b.P0, "A_naive": ["logb_naive"] + flag, "A_fut": ["logb_forecast"] + flag,
                 "A_cap": ["logb_cap"] + flag}
    s8b.NEED = sorted({c for v in s8b.SPECS.values() for c in v} | {"y", "psu"})
    rng = np.random.default_rng(s8b.SEED)

    order = [f"{a}->{b}" for a, b in zip(WAVES[:-1], WAVES[1:])]
    folds = []
    for k in order[order.index(s8b.FIRST_FOLD):]:
        folds.append(s8b.fit_fold(pairs[pairs.transition.isin(order[:order.index(k)])], pairs[pairs.transition == k]))
        print(f"fold {k}: n={len(folds[-1])}", flush=True)
    fc = pd.concat(folds, ignore_index=True)
    inc = fc.high_fuel_vulnerable.values == 0
    capreg = (fc["T"] >= TEST_FROM).values

    fc_rows = []
    strata = {"cap regime (T >= 2019-03)": capreg, "pre-cap (T < 2019-03)": ~capreg,
              "cap regime, shock origins": capreg & fc.shock.values, "all": np.ones(len(fc), bool)}
    for sname, smask in strata.items():
        m = smask & inc
        y = fc.y.values[m]
        row = dict(stratum=sname, subset="incident", n=int(m.sum()), prevalence=y.mean(),
                   **{f"auc_{mod}": roc_auc_score(y, fc[mod].values[m]) for mod in s8b.SPECS})
        for a, b in (("A_cap", "A_naive"), ("A_cap", "A_fut")):
            bt = s8b.boot_auc_diff(fc, a, b, m, rng)
            s = s8b.summarise("", row[f"auc_{a}"] - row[f"auc_{b}"], bt)
            row[f"delta_{a}_vs_{b}"], row[f"ci_{a}_vs_{b}"] = s["estimate"], f"[{s['ci_low']:.4f}, {s['ci_high']:.4f}]"
            row[f"p_{a}_vs_{b}"] = s["p_one_sided"]
            if sname.startswith("cap regime (") and (a, b) == ("A_cap", "A_naive"):
                hyp.append(s8b.summarise("H9.3 forward-chained, T >= 2019-03, incident: AUC A_cap - A_naive",
                                         s["estimate"], bt))
        fc_rows.append(row)
    fc_auc = pd.DataFrame(fc_rows)

    sp = s8b.fit_fold(pairs[~pairs.transition.isin(VALIDATION)], pairs[pairs.transition.isin(VALIDATION)])
    sinc = sp.high_fuel_vulnerable.values == 0
    split = pd.DataFrame([dict(subset=sub, model=mod, auc=roc_auc_score(sp.y[m], sp[mod][m]))
                          for sub, m in {"all": np.ones(len(sp), bool), "incident": sinc}.items() for mod in s8b.SPECS])
    cal = pd.DataFrame([dict(model=m, **calibration(sp.y.values.astype(int), np.clip(sp[m].values, 1e-6, 1 - 1e-6)))
                        for m in s8b.SPECS])

    hyp = pd.DataFrame(hyp)
    hyp["p_holm"] = s8b.holm(hyp.p_one_sided).values
    hyp["survives_holm"] = hyp.p_holm < 0.05
    rule = bool(hyp[hyp.test.str.startswith(("H9.1a", "H9.1b", "H9.3"))].survives_holm.all())
    hyp = pd.concat([hyp, pd.DataFrame([dict(test="DECISION: forecasting useful (H9.1a, H9.1b, H9.3 survive Holm)",
                                             estimate=float(rule))])], ignore_index=True)

    hyp.to_csv(OUT / "hypotheses.csv", index=False)
    fc_auc.to_csv(OUT / "forward_chain_auc.csv", index=False)
    split.to_csv(OUT / "split_metrics.csv", index=False)
    cal.to_csv(OUT / "calibration.csv", index=False)

    pd.set_option("display.width", 250)
    for t in (hyp, acc, acc_h, fc_auc, split, cal):
        print(t.round(4).to_string(index=False))
        print()


if __name__ == "__main__":
    main()
