"""
Stage 8 diagnostics (motivation for amendment A7; descriptive, no models).

Why the pre-specified FES could not work as an anticipatory predictor:

  1. timing: correlation of the monthly high-vulnerability rate (interview
     months with >= 200 households, 2010+) with retail gas/electricity price
     growth shifted by -6..18 months (positive = price growth BEFORE the interview);
  2. ceiling: validation waves n and o, AUC of the true interview-month
     vulnerability rate used as the only predictor (the best any national
     monthly signal could do) and of fes_magnitude_growth3;
  3. forecasts: v2 selected-model relative RMSE vs no-change by horizon.

Outputs (outputs_v3/diagnostics/, aggregate only): lag_correlation.csv,
monthly_signal_ceiling.csv, v2_forecast_by_horizon.csv
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.paths import UKHLS_PANEL  # noqa: E402

OUT = ROOT / "outputs_v3" / "diagnostics"
CORE = ROOT / "data" / "processed" / "core_processed.csv"
V2 = ROOT / "outputs_v2"
LAGS = [-6, 0, 3, 6, 9, 12, 18]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    p = pd.read_csv(UKHLS_PANEL, usecols=["wave", "interview_year", "interview_month",
                                          "high_fuel_vulnerable", "fes_magnitude_growth3"])
    p = p.dropna(subset=["high_fuel_vulnerable", "interview_year", "interview_month"])
    p["date"] = pd.to_datetime(dict(year=p.interview_year.astype(int), month=p.interview_month.astype(int), day=1))
    core = pd.read_csv(CORE, parse_dates=["date"]).set_index("date")

    m = p.groupby("date").high_fuel_vulnerable.agg(["mean", "size"])
    m = m[(m["size"] >= 200) & (m.index >= "2010-01-01")]
    rows = []
    for s in ("gas_growth", "electricity_growth"):
        for lag in LAGS:
            x = core[s].shift(lag).reindex(m.index)
            ok = x.notna()
            rows.append(dict(series=s.replace("_growth", ""), months_before_interview=lag,
                             correlation=np.corrcoef(m["mean"][ok], x[ok])[0, 1], n_months=int(ok.sum())))
    pd.DataFrame(rows).to_csv(OUT / "lag_correlation.csv", index=False)

    v = p[p.wave.isin(["n", "o"])].copy()
    v["month_rate"] = v.groupby("date").high_fuel_vulnerable.transform("mean")
    vf = v.dropna(subset=["fes_magnitude_growth3"])
    pd.DataFrame([
        dict(predictor="true interview-month vulnerability rate (oracle)", auc=roc_auc_score(v.high_fuel_vulnerable, v.month_rate),
             n_households=len(v), n_months=v.date.nunique()),
        dict(predictor="fes_magnitude_growth3", auc=roc_auc_score(vf.high_fuel_vulnerable, vf.fes_magnitude_growth3),
             n_households=len(vf), n_months=vf.date.nunique()),
    ]).to_csv(OUT / "monthly_signal_ceiling.csv", index=False)

    sel = pd.read_csv(V2 / "fes" / "model_selection_by_year.csv")
    frames = []
    for f in (V2 / "forecasts_rolling").glob("*/*_growth_pct_forecasts_*_core.csv"):
        mm = re.search(r"/(\d{4})/(\w+?)_growth_pct_forecasts_(\w+)_core\.csv$", str(f))
        if not mm:
            continue
        y, s, mod = int(mm.group(1)), mm.group(2), mm.group(3)
        name = {"lstm": "LSTM", "tft": "TFT", "sarima": "SARIMA", "prophet": "Prophet"}[mod]
        if sel[(sel.as_of_year == y) & (sel.series == s)].model.iloc[0] != name:
            continue
        d = pd.read_csv(f, parse_dates=["date"])
        col = f"{s}_growth"
        d["actual"] = core[col].reindex(d.date).values
        d["naive"] = core.loc[f"{y}-12-01", col]
        d["series"], d["h"] = s, d.date.dt.month
        frames.append(d.dropna(subset=["actual"]))
    D = pd.concat(frames)
    rows = []
    for (s, h), g in D.groupby(["series", "h"]):
        rm, rn = np.sqrt(((g.forecast - g.actual) ** 2).mean()), np.sqrt(((g.naive - g.actual) ** 2).mean())
        rows.append(dict(series=s, h=h, n_origins=len(g), rel_rmse_vs_naive=rm / rn))
    pd.DataFrame(rows).to_csv(OUT / "v2_forecast_by_horizon.csv", index=False)
    print(pd.read_csv(OUT / "lag_correlation.csv").round(3).to_string(index=False))
    print(pd.read_csv(OUT / "monthly_signal_ceiling.csv").round(3).to_string(index=False))
    print(pd.read_csv(OUT / "v2_forecast_by_horizon.csv").pivot(index="h", columns="series",
                                                                 values="rel_rmse_vs_naive").round(2).to_string())


if __name__ == "__main__":
    main()
