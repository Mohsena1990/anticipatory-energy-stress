"""
model_evaluation.py
───────────────────
Aggregate per-model metric results, compute rank-aggregation scores,
and select the best model per series × mode.

Rank-aggregation rule
─────────────────────
  For each metric k, rank models from best (1) to worst (N).
  Score_m = sum_k Rank(Metric_k) * Weight_k
  Lowest total score = best model.

  Metrics used:
    Lower-is-better : MAE, RMSE, MAPE, SMAPE, MASE, QuantileLoss, WinklerScore, MSIS
    Higher-is-better: PredictionIntervalCoverage  (rank is INVERTED before summing)

  Ranking is done separately within each (series, mode) group so that
  core models are ranked against each other and macro models against each other.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

from src.logging_utils import get_logger

log = get_logger("model_evaluation")

# MAPE is excluded from ranking: growth rates can be near zero, causing MAPE to
# explode (e.g. 3000 %).  It is kept in the table for reporting only.
RANK_LOWER_BETTER = [
    "MAE", "RMSE", "SMAPE", "MASE",
    "QuantileLoss", "WinklerScore", "MSIS",
]
RANK_HIGHER_BETTER = ["PredictionIntervalCoverage"]
ALL_METRICS        = ["MAE", "RMSE", "MAPE", "SMAPE", "MASE",
                      "QuantileLoss", "WinklerScore", "MSIS",
                      "PredictionIntervalCoverage"]
RANK_WEIGHTS = {
    "MAE": 0.25,
    "RMSE": 0.25,
    "SMAPE": 0.15,
    "MASE": 0.10,
    "QuantileLoss": 0.10,
    "WinklerScore": 0.05,
    "MSIS": 0.05,
    "PredictionIntervalCoverage": 0.05,
}


def build_metrics_table(results: list) -> pd.DataFrame:
    """
    Convert a list of per-model result dicts into a tidy DataFrame.

    Each dict should have:
      { 'series': str, 'model': str, 'mode': str,
        'metrics': { metric_name: float, ... } }
    """
    rows = []
    for r in results:
        row = {
            "series_name": r["series"],
            "model":       r["model"],
            "mode":        r.get("mode", "core"),
        }
        row.update(r["metrics"])
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    metric_cols = [m for m in ALL_METRICS if m in df.columns]
    return df[["series_name", "model", "mode"] + metric_cols]


def rank_aggregate(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (series, mode) group, compute per-model rank-aggregation scores.
    Ranking within mode ensures core vs macro are compared fairly within mode.

    Returns DataFrame with 'rank_score' column; lowest = best.
    """
    result_rows = []
    for (series, mode), grp in metrics_df.groupby(["series_name", "mode"]):
        grp   = grp.copy().reset_index(drop=True)
        score = pd.Series(0.0, index=grp.index)

        for metric in RANK_LOWER_BETTER:
            if metric in grp.columns and grp[metric].notna().any():
                ranks = grp[metric].rank(
                    method="average", ascending=True, na_option="bottom"
                )
                score += ranks * RANK_WEIGHTS[metric]

        for metric in RANK_HIGHER_BETTER:
            if metric in grp.columns and grp[metric].notna().any():
                ranks = grp[metric].rank(
                    method="average", ascending=False, na_option="bottom"
                )
                score += ranks * RANK_WEIGHTS[metric]

        grp["rank_score"] = score
        result_rows.append(grp)

    ranked = pd.concat(result_rows, ignore_index=True)
    return ranked.sort_values(["series_name", "mode", "rank_score"]).reset_index(drop=True)


def _smape(actual: np.ndarray, forecast: np.ndarray) -> float:
    denom = np.abs(actual) + np.abs(forecast)
    valid = denom > 1e-12
    if not valid.any():
        return float("nan")
    return float(np.mean(2.0 * np.abs(forecast[valid] - actual[valid]) / denom[valid]) * 100.0)


def attach_forecast_actual_metrics(
    ranked_df: pd.DataFrame,
    forecast_dir: str = "outputs_v2/forecasts",
) -> pd.DataFrame:
    """
    Add retrospective forecast-horizon metrics from the saved 2017 forecast CSVs.

    The validation metrics in ``ranked_df`` are computed on the 2016 test split.
    These columns use the optional ``actual`` column in the 2017 forecast files,
    so they are only suitable for retrospective reporting/selection.
    """
    ranked_df = ranked_df.copy()
    metrics: list[dict] = []

    for row in ranked_df.itertuples(index=False):
        path = (
            Path(forecast_dir)
            / f"{row.series_name}_growth_pct_forecasts_{row.model.lower()}_{row.mode}.csv"
        )
        item = {
            "series_name": row.series_name,
            "model": row.model,
            "mode": row.mode,
            "forecast_actual_MAE": np.nan,
            "forecast_actual_RMSE": np.nan,
            "forecast_actual_SMAPE": np.nan,
        }
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            log.warning(f"Forecast file not found for actual comparison: {path}")
            metrics.append(item)
            continue

        if not {"forecast", "actual"}.issubset(df.columns):
            metrics.append(item)
            continue

        comp = df[["forecast", "actual"]].dropna()
        if comp.empty:
            metrics.append(item)
            continue

        actual = comp["actual"].to_numpy(dtype=float)
        forecast = comp["forecast"].to_numpy(dtype=float)
        err = forecast - actual
        item["forecast_actual_MAE"] = float(np.mean(np.abs(err)))
        item["forecast_actual_RMSE"] = float(np.sqrt(np.mean(err**2)))
        item["forecast_actual_SMAPE"] = _smape(actual, forecast)
        metrics.append(item)

    actual_df = pd.DataFrame(metrics)
    return ranked_df.merge(
        actual_df,
        on=["series_name", "model", "mode"],
        how="left",
    )


def apply_selection_scores(
    ranked_df: pd.DataFrame,
    selection_basis: str = "validation",
) -> pd.DataFrame:
    """
    Add ``selection_score`` used to pick best models.

    ``validation`` uses the 2016 validation rank. ``forecast_actual`` uses the
    saved 2017 forecast-vs-actual MAE when available and falls back to the
    validation rank when actuals are absent.
    """
    ranked_df = ranked_df.copy()
    ranked_df["selection_basis"] = selection_basis
    ranked_df["selection_score"] = ranked_df["rank_score"]

    if selection_basis != "forecast_actual":
        return ranked_df

    for (_, _), grp in ranked_df.groupby(["series_name", "mode"]):
        idx = grp.index
        if "forecast_actual_MAE" not in grp.columns or not grp["forecast_actual_MAE"].notna().any():
            ranked_df.loc[idx, "selection_basis"] = "validation_fallback"
            continue

        ranked_df.loc[idx, "selection_score"] = grp["forecast_actual_MAE"].rank(
            method="average", ascending=True, na_option="bottom"
        )

    return ranked_df


def select_best_models(ranked_df: pd.DataFrame) -> dict:
    """
    Return a mapping { (series_name, mode) → best_model_name }.
    """
    best: dict = {}
    score_col = "selection_score" if "selection_score" in ranked_df.columns else "rank_score"
    for (series, mode), grp in ranked_df.groupby(["series_name", "mode"]):
        winner = grp.loc[grp[score_col].idxmin(), "model"]
        best[(series, mode)] = winner
        log.info(f"Best model [{series}][{mode}]: {winner} "
                 f"({score_col}={grp[score_col].min():.3f})")
    return best


def save_metrics_table(
    metrics_df: pd.DataFrame,
    out_dir: str = "outputs_v2/tables",
    filename: str = "model_metrics_comparison",
) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    csv_path = f"{out_dir}/{filename}.csv"
    metrics_df.to_csv(csv_path, index=False)
    log.info(f"Metrics table saved → {csv_path}")


def run_evaluation(
    results: list,
    out_dir: str = "outputs_v2/tables",
    forecast_dir: str = "outputs_v2/forecasts",
    selection_basis: str = "validation",
) -> tuple:
    """
    Full evaluation pipeline.

    Returns
    -------
    (metrics_df, ranked_df, best_models_dict)
    """
    log.info(f"Evaluating {len(results)} model results across all series and modes")

    if not results:
        log.error("No model results to evaluate — all models failed.")
        return pd.DataFrame(), pd.DataFrame(), {}

    metrics_df = build_metrics_table(results)
    ranked_df  = rank_aggregate(metrics_df)
    ranked_df  = attach_forecast_actual_metrics(ranked_df, forecast_dir)
    ranked_df  = apply_selection_scores(ranked_df, selection_basis)
    best       = select_best_models(ranked_df)

    save_metrics_table(ranked_df, out_dir)

    print(f"\n── Model ranking ({selection_basis}) ──────────────────────────────────")
    for col_name in [
        "series_name", "mode", "model", "MAE", "RMSE", "MAPE",
        "rank_score", "forecast_actual_MAE", "selection_score",
    ]:
        if col_name not in ranked_df.columns:
            ranked_df[col_name] = float("nan")
    display_cols = [
        c for c in [
            "series_name", "mode", "model", "MAE", "RMSE", "MAPE",
            "rank_score", "forecast_actual_MAE", "selection_score",
        ]
        if c in ranked_df.columns
    ]
    print(ranked_df[display_cols].to_string(index=False))

    return metrics_df, ranked_df, best


def merge_forecast_files(
    series_names: list,
    models: list,
    forecast_dir: str = "outputs_v2/forecasts",
    out_dir: str = "outputs_v2/forecasts",
    modes: list = None,
) -> dict:
    """
    Merge per-model-mode forecast CSVs into combined per-series files.

    Reads : {forecast_dir}/{series}_growth_pct_forecasts_{model_lower}_{mode}.csv
    Writes: {out_dir}/{series}_growth_pct_forecasts_all.csv
              (contains all models and modes; use 'model' and 'mode' columns to filter)

    Returns
    -------
    Dict mapping series_name → combined DataFrame
    """
    if modes is None:
        modes = ["core", "macro"]

    combined: dict = {}

    for series in series_names:
        dfs = []
        for model in models:
            for mode in modes:
                path = (
                    f"{forecast_dir}/{series}_growth_pct_forecasts_"
                    f"{model.lower()}_{mode}.csv"
                )
                try:
                    df = pd.read_csv(path, parse_dates=["date"])
                    if "mode" not in df.columns:
                        df["mode"] = mode
                    dfs.append(df)
                    log.debug(f"Loaded {path} ({len(df)} rows)")
                except FileNotFoundError:
                    log.warning(f"Forecast file not found: {path}")

        if dfs:
            merged = pd.concat(dfs, ignore_index=True)
            out_path = f"{out_dir}/{series}_growth_pct_forecasts_all.csv"
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            merged.to_csv(out_path, index=False)
            log.info(f"Combined forecasts ({len(merged)} rows) → {out_path}")
            combined[series] = merged

            # Also write separate core / macro combined files for convenience
            for mode in modes:
                sub = merged[merged["mode"] == mode]
                if not sub.empty:
                    sub_path = f"{out_dir}/{series}_growth_pct_forecasts_{mode}.csv"
                    sub.to_csv(sub_path, index=False)
                    log.info(f"  {mode} combined → {sub_path}")
        else:
            log.error(f"No forecast files found for series '{series}'")

    return combined
