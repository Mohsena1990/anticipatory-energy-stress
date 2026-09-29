"""
Stage 8, step 1 (analysis_plan_rerun.md, amendment A7; EXPLORATORY, post hoc).

Retail energy price-level forecasts from wholesale markets, made at every monthly
origin T with the information available at T:

  retail CPI levels (ONS MM23: gas D7DU, electricity D7DT, liquid fuels D7DV,
  solid fuels D7DW) to L = T-2; NBP gas futures and Brent to T-1.

Direct OLS per horizon h = 1..13 on an expanding window (pairs whose target
month is <= L only). Gas and electricity use the NBP futures features g1, g2;
liquid fuels use Brent; solid fuels are no-change. The Energy Price Guarantee
caps the gas/electricity forecast for the months and origins in which it was
announced policy.

Benchmark: no-change level. Evaluated on the log window ratio
log(mean P[T..T+11] / mean P[T-12..T-1]), the price change that separates the
spend windows of consecutive UKHLS interviews.

Outputs (outputs_v3/anticipation/):
  price_forecasts.csv        origin x series: forecast / naive / realised window
                             ratios, sigma of past errors, fallback flag
  price_accuracy.csv         RMSE, relative RMSE and DM test by series and period
  price_accuracy_by_horizon.csv
  figures/price_window_ratio.{png,pdf}
"""
from __future__ import annotations

import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
EXT = RAW / "anticipation"
OUT = ROOT / "outputs_v3" / "anticipation"

SERIES = {"gas": "cpi_gas_d7du.csv", "electricity": "cpi_electricity_d7dt.csv",
          "liquid_fuels": "cpi_liquid_fuels_d7dv.csv", "solid_fuels": "cpi_solid_fuels_d7dw.csv"}
H_MAX = 13
MIN_PAIRS = 24
MIN_ERRORS = 12
FIRST_ORIGIN, LAST_ORIGIN = "2008-01-01", "2026-09-01"

# Energy Price Guarantee, typical-bill approximation relative to the April 2022 cap
# (£1,971): £2,500 -> gas x1.36, electricity x1.17; £3,000 -> gas x1.70, electricity x1.33.
EPG_MULT = {2500: {"gas": 1.36, "electricity": 1.17}, 3000: {"gas": 1.70, "electricity": 1.33}}
EPG_BASE_MONTH = pd.Timestamp("2022-04-01")
# (first origin at which the policy was known, first month, last month, EPG level)
EPG_SCHEDULE = [
    ("2022-10-01", "2022-10-01", "2023-03-01", 2500),   # announced 8 Sep 2022
    ("2022-12-01", "2023-04-01", "2024-03-01", 3000),   # Autumn Statement 17 Nov 2022
    ("2023-04-01", "2023-04-01", "2023-06-01", 2500),   # Budget 15 Mar 2023
]


def read_ons(path: Path) -> pd.Series:
    raw = pd.read_csv(path, header=None, names=["k", "v"], dtype=str)
    raw = raw[raw.k.str.match(r"^\d{4} [A-Z]{3}$", na=False)]
    idx = pd.to_datetime(raw.k, format="%Y %b")
    return pd.Series(raw.v.astype(float).values, index=idx).sort_index()


def read_nbp() -> pd.Series:
    d = pd.read_csv(RAW / "UK NBP Natural Gas Quaterly Futures Historical Data UK.csv", encoding="utf-8-sig")
    idx = pd.to_datetime(d["Date"], format="%d/%m/%Y")
    return pd.Series(d["Price"].astype(str).str.replace(",", "").astype(float).values, index=idx).sort_index()


def read_brent() -> pd.Series:
    d = pd.read_csv(EXT / "brent_fred_mcoilbrenteu.csv", parse_dates=["observation_date"])
    return d.set_index("observation_date")["MCOILBRENTEU"].astype(float).sort_index()


def load_prices() -> dict[str, pd.Series]:
    out = {s: np.log(read_ons(EXT / f)) for s, f in SERIES.items()}
    out["_nbp"] = np.log(read_nbp())
    out["_brent"] = np.log(read_brent())
    return out


def _at(s: pd.Series, m: pd.Timestamp) -> float:
    return float(s.get(m, np.nan))


def _mean(s: pd.Series, a: pd.Timestamp, b: pd.Timestamp) -> float:
    w = s.loc[a:b]
    n_expected = (b.year - a.year) * 12 + b.month - a.month + 1
    return float(w.mean()) if len(w.dropna()) == n_expected else np.nan


def features(series: str, px: dict, L: pd.Timestamp) -> np.ndarray | None:
    """Features for last retail month L (wholesale known to L+1)."""
    mo = pd.DateOffset(months=1)
    p = px[series]
    if series in ("gas", "electricity"):
        f = px["_nbp"]
        emb = _mean(f, L - 8 * mo, L - 3 * mo)
        g1 = _at(f, L + mo) - emb
        g2 = emb - _mean(f, L - 20 * mo, L - 15 * mo) - (_at(p, L) - _at(p, L - 12 * mo))
        x = np.array([g1, g2])
    elif series == "liquid_fuels":
        b = px["_brent"]
        x = np.array([_at(b, L + mo) - _mean(b, L - mo, L)])
    else:
        return None
    return None if np.isnan(x).any() else x


def epg_cap(series: str, origin: pd.Timestamp, month: pd.Timestamp, base_level: float) -> float:
    """Log-level ceiling from announced EPG policy, or +inf."""
    if series not in ("gas", "electricity") or np.isnan(base_level):
        return np.inf
    level = None
    for known_from, first, last, lvl in EPG_SCHEDULE:  # later announcements override earlier
        if origin >= pd.Timestamp(known_from) and pd.Timestamp(first) <= month <= pd.Timestamp(last):
            level = lvl
    return np.inf if level is None else base_level + np.log(EPG_MULT[level][series])


def forecast_origin(series: str, px: dict, T: pd.Timestamp, feat_cache: dict) -> tuple[dict, bool]:
    """Forecast log level for months L+1..L+13 (L = T-2). Returns ({month: logP}, fallback)."""
    mo = pd.DateOffset(months=1)
    p = px[series]
    L = T - 2 * mo
    pL = _at(p, L)
    x_now = feature_lookup(series, px, L, feat_cache)
    fallback = False
    out = {}
    base = _at(p, EPG_BASE_MONTH) if L >= EPG_BASE_MONTH else np.nan
    for h in range(1, H_MAX + 1):
        target = L + h * mo
        pred = np.nan
        if x_now is not None:
            X, y = [], []
            Lp = L - h * mo
            while True:
                xf = feature_lookup(series, px, Lp, feat_cache)
                if xf is None and Lp < pd.Timestamp("2006-01-01"):
                    break
                yv = _at(p, Lp + h * mo) - _at(p, Lp)
                if xf is not None and not np.isnan(yv):
                    X.append(xf); y.append(yv)
                Lp = Lp - mo
            if len(y) >= MIN_PAIRS:
                X = np.column_stack([np.ones(len(y)), np.array(X)])
                beta, *_ = np.linalg.lstsq(X, np.array(y), rcond=None)
                pred = pL + float(np.r_[1.0, x_now] @ beta)
        if np.isnan(pred):
            fallback = fallback or series in ("gas", "electricity", "liquid_fuels")
            pred = pL
        out[target] = min(pred, epg_cap(series, T, target, base))
    return out, fallback


def feature_lookup(series, px, L, cache):
    key = (series, L)
    if key not in cache:
        cache[key] = features(series, px, L)
    return cache[key]


def window_ratio(level_path: dict, p: pd.Series, T: pd.Timestamp, known_to: pd.Timestamp) -> float:
    """log(mean P[T..T+11] / mean P[T-12..T-1]); months after known_to come from level_path."""
    mo = pd.DateOffset(months=1)

    def lvl(m):
        return np.exp(_at(p, m)) if m <= known_to else np.exp(level_path[m])
    prev = [lvl(T - k * mo) for k in range(12, 0, -1)]
    nxt = [lvl(T + k * mo) for k in range(0, 12)]
    return float(np.log(np.mean(nxt) / np.mean(prev)))


def run_price_forecasts(px: dict | None = None) -> pd.DataFrame:
    px = px or load_prices()
    mo = pd.DateOffset(months=1)
    origins = pd.date_range(FIRST_ORIGIN, LAST_ORIGIN, freq="MS")
    rows, cache = [], {}
    for s in SERIES:
        p = px[s]
        last_cpi = p.index.max()
        for T in origins:
            L = T - 2 * mo
            if L > last_cpi:
                continue
            path, fb = forecast_origin(s, px, T, cache)
            naive = {m: _at(p, L) for m in path}
            realised = np.nan
            if T + 11 * mo <= last_cpi:
                realised = window_ratio({}, p, T, known_to=last_cpi)
            rows.append(dict(series=s, origin=T, fallback=fb,
                             ratio_forecast=window_ratio(path, p, T, known_to=L),
                             ratio_naive=window_ratio(naive, p, T, known_to=L),
                             ratio_realised=realised,
                             **{f"h{h}_forecast": path[L + h * mo] - _at(p, L) for h in range(1, H_MAX + 1)},
                             **{f"h{h}_realised": _at(p, L + h * mo) - _at(p, L) for h in range(1, H_MAX + 1)}))
    df = pd.DataFrame(rows)
    # sigma_f(T): RMS of out-of-sample window-ratio errors whose outcome is known by T-2
    sig = []
    for s, g in df.groupby("series"):
        g = g.sort_values("origin")
        for _, r in g.iterrows():
            known = g[(g.origin + pd.DateOffset(months=11) <= r.origin - 2 * mo) & g.ratio_realised.notna()]
            e = (known.ratio_forecast - known.ratio_realised).values
            if len(e) >= MIN_ERRORS:
                sig.append(np.sqrt(np.mean(e ** 2)))
            else:
                hist = px[s].loc[:r.origin - 2 * mo].diff(12).dropna()
                sig.append(float(hist.std()))
    df = df.sort_values(["series", "origin"]).reset_index(drop=True)
    df["sigma_ratio"] = sig
    return df


def dm_test(e1: np.ndarray, e2: np.ndarray, lag: int = 12) -> tuple[float, float]:
    """Diebold-Mariano on squared-error loss with Newey-West variance (Bartlett, `lag`)."""
    d = e1 ** 2 - e2 ** 2
    n = len(d)
    dc = d - d.mean()
    v = np.sum(dc * dc) / n
    for k in range(1, lag + 1):
        v += 2 * (1 - k / (lag + 1)) * np.sum(dc[k:] * dc[:-k]) / n
    stat = d.mean() / np.sqrt(v / n)
    return float(stat), float(2 * stats.t.sf(abs(stat), df=n - 1))


def evaluate(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, hrows = [], []
    ev = df[df.ratio_realised.notna() & (df.origin >= "2009-01-01")]
    periods = {"all 2009+": lambda o: o.year >= 2009, "2009-2020": lambda o: o.year <= 2020,
               "2021+": lambda o: o.year >= 2021}
    for s, g in ev.groupby("series"):
        for pname, f in periods.items():
            gg = g[g.origin.map(f)]
            e_f = (gg.ratio_forecast - gg.ratio_realised).values
            e_n = (gg.ratio_naive - gg.ratio_realised).values
            stat, p = dm_test(e_f, e_n) if s != "solid_fuels" else (np.nan, np.nan)
            rows.append(dict(series=s, period=pname, n_origins=len(gg),
                             rmse_forecast=np.sqrt(np.mean(e_f ** 2)), rmse_naive=np.sqrt(np.mean(e_n ** 2)),
                             rel_rmse=np.sqrt(np.mean(e_f ** 2)) / np.sqrt(np.mean(e_n ** 2)),
                             dm_stat=stat, dm_p=p,
                             corr_forecast_change=np.corrcoef(gg.ratio_forecast - gg.ratio_naive,
                                                              gg.ratio_realised - gg.ratio_naive)[0, 1]
                             if s != "solid_fuels" else np.nan,
                             fallback_origins=int(gg.fallback.sum())))
        for h in range(1, H_MAX + 1):
            gh = g.dropna(subset=[f"h{h}_realised"])
            ef = gh[f"h{h}_forecast"] - gh[f"h{h}_realised"]
            en = -gh[f"h{h}_realised"]
            hrows.append(dict(series=s, h=h, n=len(gh), rmse_forecast=np.sqrt((ef ** 2).mean()),
                              rmse_naive=np.sqrt((en ** 2).mean()),
                              rel_rmse=np.sqrt((ef ** 2).mean()) / np.sqrt((en ** 2).mean())))
    return pd.DataFrame(rows), pd.DataFrame(hrows)


def plot(df: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for ax, s in zip(axes, ["gas", "electricity"]):
        g = df[(df.series == s) & (df.origin >= "2009-01-01")]
        ax.plot(g.origin, 100 * np.expm1(g.ratio_realised), color="#1f1f1f", lw=1.6, label="Realised")
        ax.plot(g.origin, 100 * np.expm1(g.ratio_forecast), color="#D55E00", lw=1.3, label="Futures forecast")
        ax.plot(g.origin, 100 * np.expm1(g.ratio_naive), color="#8A96A3", lw=1.1, ls="--", label="No change")
        ax.axhline(0, color="#cccccc", lw=0.8)
        ax.set_ylabel(f"{s.capitalize()}: % change,\nnext 12 m vs last 12 m")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, ncol=3, loc="upper left")
    axes[1].set_xlabel("Forecast origin (interview month)")
    fig.tight_layout()
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / "figures" / f"price_window_ratio.{ext}", dpi=200)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = run_price_forecasts()
    df.to_csv(OUT / "price_forecasts.csv", index=False)
    acc, acc_h = evaluate(df)
    acc.to_csv(OUT / "price_accuracy.csv", index=False)
    acc_h.to_csv(OUT / "price_accuracy_by_horizon.csv", index=False)
    plot(df)
    print(acc.round(4).to_string(index=False))
    print(acc_h.pivot(index="h", columns="series", values="rel_rmse").round(3).to_string())


if __name__ == "__main__":
    main()
