"""
Stage 2 evaluation (analysis_plan_rerun.md): rolling forecasts vs realised
values on the three growth terms only (gas, electricity, carbon), v1 vs v2.

Versions compared
  v1_core   submitted-draft-v1 rolling run, core-mode winner per series/origin
  v1_macro  same run, macro-mode winner: the variant v1 attached to
            households (fes_variant_selection.csv chose Equal_Macro by
            hindsight RMSE over all years)
  v2_core   rerun: core only, hyperparameters tuned per origin
  naive     no-change benchmark: growth observed at the origin (Dec of
            as_of_year) repeated for all 12 target months
  snaive    seasonal naive: target month m of year Y+1 forecast by the
            observed growth in month m of the origin year Y

Accuracy is on the three growth terms only. Forecast uncertainty (the
fourth FES term) is reported separately as 95% PI coverage and width.
Diebold-Mariano: squared-error loss over the 192 non-overlapping target
months (each forecast once, horizons 1-12), Newey-West variance with lag
11, Harvey-Leybourne-Newbold correction at h=12 (conservative).
MASE: MAE scaled by the in-sample one-step naive MAE of the training data
up to each origin (Hyndman & Koehler 2006), averaged over origins.

Outputs (outputs_v2/fes_eval/):
  forecast_accuracy_by_year.csv   version x series x target_year
  forecast_accuracy_pooled.csv    version x series (all target years), incl.
                                  relative RMSE/MAE vs naive and snaive, MASE
  diebold_mariano.csv             each model version vs naive / snaive
  uncertainty_pi.csv              95% PI coverage and mean width
  model_wins.csv                  winner counts per version x series
  fes_annual.csv                  annual mean FES per target year, v1 vs v2
"""
from __future__ import annotations

import sys
from pathlib import Path

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.paths import OUTPUTS_DIR, V1_OUTPUTS_DIR, CORE_CSV  # noqa: E402

OUT = OUTPUTS_DIR / "fes_eval"
SERIES = ("gas", "electricity", "carbon")


def load_forecasts(outputs_dir: Path, mode: str, label: str) -> pd.DataFrame:
    sel_path = outputs_dir / "fes" / "model_selection_by_year.csv"
    if not sel_path.exists():
        print(f"[skip] {label}: {sel_path} not found")
        return pd.DataFrame()
    sel = pd.read_csv(sel_path)
    sel = sel[sel["mode"] == mode]
    frames = []
    for r in sel.itertuples():
        f = (outputs_dir / "forecasts_rolling" / str(r.as_of_year)
             / f"{r.series}_growth_pct_forecasts_{r.model.lower()}_{mode}.csv")
        if not f.exists():
            print(f"[missing] {f}")
            continue
        d = pd.read_csv(f, parse_dates=["date"])
        d = d[[c for c in ["date", "forecast", "lower_bound", "upper_bound"] if c in d]]
        d["series"], d["as_of_year"], d["model"] = r.series, r.as_of_year, r.model
        frames.append(d)
    out = pd.concat(frames, ignore_index=True)
    out["version"] = label
    return out


def naive_forecasts(core: pd.DataFrame, origins: list[int]) -> pd.DataFrame:
    rows = []
    for y in origins:
        origin = pd.Timestamp(f"{y}-12-01")
        for s in SERIES:
            last = core.loc[:origin, f"{s}_growth"].dropna()
            if last.empty:
                continue
            for d in pd.date_range(f"{y + 1}-01-01", periods=12, freq="MS"):
                rows.append(dict(date=d, forecast=float(last.iloc[-1]), series=s,
                                 as_of_year=y, model="no-change", version="naive"))
    return pd.DataFrame(rows)


def seasonal_naive_forecasts(core: pd.DataFrame, origins: list[int]) -> pd.DataFrame:
    rows = []
    for y in origins:
        for s in SERIES:
            for m in range(1, 13):
                v = core[f"{s}_growth"].get(pd.Timestamp(f"{y}-{m:02d}-01"), np.nan)
                rows.append(dict(date=pd.Timestamp(f"{y + 1}-{m:02d}-01"), forecast=float(v),
                                 series=s, as_of_year=y, model="seasonal-no-change",
                                 version="snaive"))
    return pd.DataFrame(rows)


def mase_scale(core: pd.DataFrame, series: str, origin_year: int) -> float:
    hist = core.loc[:pd.Timestamp(f"{origin_year}-12-01"), f"{series}_growth"].dropna()
    return float(hist.diff().abs().mean())


def diebold_mariano(e1: np.ndarray, e2: np.ndarray, h: int = 12) -> tuple[float, float]:
    """DM test of equal squared-error loss; positive stat => model 1 worse."""
    from scipy import stats
    d = e1 ** 2 - e2 ** 2
    T = len(d)
    dc = d - d.mean()
    lrv = np.sum(dc ** 2) / T
    for k in range(1, h):
        w = 1 - k / h
        lrv += 2 * w * np.sum(dc[k:] * dc[:-k]) / T
    dm = d.mean() / np.sqrt(lrv / T)
    hln = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    stat = dm * hln
    return float(stat), float(2 * stats.t.sf(abs(stat), df=T - 1))


def metrics(g: pd.DataFrame) -> pd.Series:
    e = g["forecast"] - g["actual"]
    ok = g[["forecast", "actual"]].dropna()
    return pd.Series({
        "n_months": int(len(ok)),
        "RMSE": float(np.sqrt(np.mean(e ** 2))),
        "MAE": float(np.mean(np.abs(e))),
        "sign_agreement_pct": float(100 * np.mean(np.sign(ok.forecast) == np.sign(ok.actual))),
        "pearson_r": float(ok.forecast.corr(ok.actual)) if len(ok) > 2 else np.nan,
    })


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    core = pd.read_csv(CORE_CSV, index_col=0, parse_dates=True).sort_index()
    actual = (core[[f"{s}_growth" for s in SERIES]]
              .rename(columns=lambda c: c.replace("_growth", ""))
              .stack().rename("actual").reset_index()
              .rename(columns={"level_0": "date", core.index.name or "index": "date",
                               "level_1": "series"}))
    actual.columns = ["date", "series", "actual"]

    fc = pd.concat([
        load_forecasts(V1_OUTPUTS_DIR, "core", "v1_core"),
        load_forecasts(V1_OUTPUTS_DIR, "macro", "v1_macro"),
        load_forecasts(OUTPUTS_DIR, "core", "v2_core"),
    ], ignore_index=True)
    origins = sorted(fc["as_of_year"].unique())
    fc = pd.concat([fc, naive_forecasts(core, origins),
                    seasonal_naive_forecasts(core, origins)], ignore_index=True)
    fc["target_year"] = fc["date"].dt.year
    fc = fc.merge(actual, on=["date", "series"], how="left")
    fc = fc.dropna(subset=["actual"])

    by_year = (fc.groupby(["version", "series", "target_year"]).apply(metrics).reset_index())
    by_year.to_csv(OUT / "forecast_accuracy_by_year.csv", index=False)
    pooled = fc.groupby(["version", "series"]).apply(metrics).reset_index()
    for bench in ["naive", "snaive"]:
        b = pooled[pooled.version == bench].set_index("series")
        pooled[f"relRMSE_vs_{bench}"] = pooled.RMSE / pooled.series.map(b.RMSE)
        pooled[f"relMAE_vs_{bench}"] = pooled.MAE / pooled.series.map(b.MAE)
    fc["abs_err"] = (fc.forecast - fc.actual).abs()
    per_origin = fc.groupby(["version", "series", "as_of_year"]).abs_err.mean().reset_index()
    per_origin["scale"] = [mase_scale(core, s, y) for s, y in zip(per_origin.series, per_origin.as_of_year)]
    per_origin["mase"] = per_origin.abs_err / per_origin.scale
    pooled = pooled.merge(per_origin.groupby(["version", "series"]).mase.mean().rename("MASE").reset_index(),
                          on=["version", "series"])

    dm_rows = []
    for (ver, s), g in fc[~fc.version.isin(["naive", "snaive"])].groupby(["version", "series"]):
        for bench in ["naive", "snaive"]:
            b = fc[(fc.version == bench) & (fc.series == s)][["date", "forecast"]]
            m = g[["date", "forecast", "actual"]].merge(b, on="date", suffixes=("", "_b")).dropna().sort_values("date")
            stat, pv = diebold_mariano((m.forecast - m.actual).values, (m.forecast_b - m.actual).values)
            dm_rows.append(dict(version=ver, series=s, benchmark=bench, n_months=len(m),
                                DM_stat=stat, p_value=pv,
                                verdict=("worse than benchmark" if stat > 0 and pv < 0.05 else
                                         "better than benchmark" if stat < 0 and pv < 0.05 else
                                         "not significantly different")))
    dm = pd.DataFrame(dm_rows)
    dm.to_csv(OUT / "diebold_mariano.csv", index=False)

    unc = fc.dropna(subset=["lower_bound", "upper_bound"]) if "lower_bound" in fc else pd.DataFrame()
    if not unc.empty:
        unc = unc.assign(covered=(unc.actual >= unc.lower_bound) & (unc.actual <= unc.upper_bound),
                         width=unc.upper_bound - unc.lower_bound)
        (unc.groupby(["version", "series"])
            .agg(n_months=("covered", "size"), pi95_coverage_pct=("covered", lambda x: 100 * x.mean()),
                 mean_pi_width=("width", "mean"), median_pi_width=("width", "median"))
            .reset_index().to_csv(OUT / "uncertainty_pi.csv", index=False))
    # Share of target years in which each version beats the naive benchmark.
    nv = by_year[by_year.version == "naive"].set_index(["series", "target_year"]).RMSE
    by_year["beats_naive"] = by_year.apply(
        lambda r: np.nan if r.version == "naive"
        else float(r.RMSE < nv.get((r.series, r.target_year), np.nan)), axis=1)
    pooled = pooled.merge(
        by_year.groupby(["version", "series"]).beats_naive.mean().mul(100)
        .rename("pct_years_beating_naive").reset_index(), on=["version", "series"])
    pooled.to_csv(OUT / "forecast_accuracy_pooled.csv", index=False)

    wins = (fc[~fc.version.isin(["naive", "snaive"])].drop_duplicates(["version", "series", "as_of_year"])
            .groupby(["version", "series", "model"]).size().rename("n_origins").reset_index())
    wins.to_csv(OUT / "model_wins.csv", index=False)

    ann = []
    for label, d in [("v1", V1_OUTPUTS_DIR), ("v2", OUTPUTS_DIR)]:
        f = d / "fes" / "fes_rolling_yearly.csv"
        if f.exists():
            t = pd.read_csv(f)
            keep = [c for c in ["as_of_year", "target_year", "fes_core", "fes_macro", "fes_actual"] if c in t]
            ann.append(t[keep].assign(version=label))
    if ann:
        pd.concat(ann).to_csv(OUT / "fes_annual.csv", index=False)

    # Thesis tables (author decision 2026-09-26): relative RMSE vs naive and
    # seasonal naive with DM p-values in the main text; MASE in an appendix;
    # per-year relative RMSE to show where gains are concentrated.
    models = ["v1_core", "v1_macro", "v2_core"]
    dmw = dm.pivot_table(index=["version", "series"], columns="benchmark", values="p_value").add_prefix("DM_p_vs_")
    main_tab = (pooled[pooled.version.isin(models)]
                .set_index(["version", "series"])[["RMSE", "relRMSE_vs_naive", "relRMSE_vs_snaive"]]
                .join(dmw).reset_index()
                [["series", "version", "RMSE", "relRMSE_vs_naive", "DM_p_vs_naive",
                  "relRMSE_vs_snaive", "DM_p_vs_snaive"]]
                .sort_values(["series", "version"]))
    main_tab.to_csv(OUT / "thesis_table_forecast_accuracy.csv", index=False)
    (pooled[pooled.version.isin(models + ["naive", "snaive"])][["series", "version", "MAE", "MASE"]]
        .sort_values(["series", "version"]).to_csv(OUT / "appendix_table_mase.csv", index=False))
    by = by_year.set_index(["series", "target_year", "version"]).RMSE.unstack("version")
    per_year = pd.concat({f"{m}_relRMSE_vs_{b}": by[m] / by[b] for m in models for b in ["naive", "snaive"]},
                         axis=1).reset_index()
    per_year.to_csv(OUT / "thesis_table_relrmse_by_year.csv", index=False)

    pd.set_option("display.width", 200)
    cols = ["version", "series", "RMSE", "MAE", "relRMSE_vs_naive", "relRMSE_vs_snaive", "MASE",
            "sign_agreement_pct", "pearson_r"]
    print(pooled[cols].round(3).to_string(index=False))
    print(dm.round(3).to_string(index=False))
    if (OUT / "uncertainty_pi.csv").exists():
        print(pd.read_csv(OUT / "uncertainty_pi.csv").round(2).to_string(index=False))
    print(wins.pivot_table(index=["version", "series"], columns="model", values="n_origins",
                           fill_value=0).to_string())


if __name__ == "__main__":
    main()
