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

Outputs (outputs_v2/fes_eval/):
  forecast_accuracy_by_year.csv   version x series x target_year
  forecast_accuracy_pooled.csv    version x series (all target years)
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
        d = pd.read_csv(f, parse_dates=["date"])[["date", "forecast"]]
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
    fc = pd.concat([fc, naive_forecasts(core, origins)], ignore_index=True)
    fc["target_year"] = fc["date"].dt.year
    fc = fc.merge(actual, on=["date", "series"], how="left")
    fc = fc.dropna(subset=["actual"])

    by_year = (fc.groupby(["version", "series", "target_year"]).apply(metrics).reset_index())
    by_year.to_csv(OUT / "forecast_accuracy_by_year.csv", index=False)
    pooled = fc.groupby(["version", "series"]).apply(metrics).reset_index()
    # Share of target years in which each version beats the naive benchmark.
    nv = by_year[by_year.version == "naive"].set_index(["series", "target_year"]).RMSE
    by_year["beats_naive"] = by_year.apply(
        lambda r: np.nan if r.version == "naive"
        else float(r.RMSE < nv.get((r.series, r.target_year), np.nan)), axis=1)
    pooled = pooled.merge(
        by_year.groupby(["version", "series"]).beats_naive.mean().mul(100)
        .rename("pct_years_beating_naive").reset_index(), on=["version", "series"])
    pooled.to_csv(OUT / "forecast_accuracy_pooled.csv", index=False)

    wins = (fc[fc.version != "naive"].drop_duplicates(["version", "series", "as_of_year"])
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

    pd.set_option("display.width", 200)
    print(pooled.round(3).to_string(index=False))
    print(wins.pivot_table(index=["version", "series"], columns="model", values="n_origins",
                           fill_value=0).to_string())


if __name__ == "__main__":
    main()
