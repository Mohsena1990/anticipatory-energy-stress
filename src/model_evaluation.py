"""
model_evaluation.py
───────────────────
Aggregate per-model metric results, compute rank-aggregation scores,
and select the best model per series × mode.

Rank-aggregation rule
─────────────────────
  For each metric k, rank models from best (1) to worst (N).
  Score_m = sum_k Rank(Metric_k)
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
                score += grp[metric].rank(method="average", ascending=True, na_option="bottom")

        for metric in RANK_HIGHER_BETTER:
            if metric in grp.columns and grp[metric].notna().any():
                score += grp[metric].rank(method="average", ascending=False, na_option="bottom")

        grp["rank_score"] = score
        result_rows.append(grp)

    ranked = pd.concat(result_rows, ignore_index=True)
    return ranked.sort_values(["series_name", "mode", "rank_score"]).reset_index(drop=True)


def select_best_models(ranked_df: pd.DataFrame) -> dict:
    """
    Return a mapping { (series_name, mode) → best_model_name }.
    """
    best: dict = {}
    for (series, mode), grp in ranked_df.groupby(["series_name", "mode"]):
        winner = grp.loc[grp["rank_score"].idxmin(), "model"]
        best[(series, mode)] = winner
        log.info(f"Best model [{series}][{mode}]: {winner} "
                 f"(score={grp['rank_score'].min():.1f})")
    return best


def save_metrics_table(
    metrics_df: pd.DataFrame,
    out_dir: str = "outputs/tables",
    filename: str = "model_metrics_comparison",
) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    csv_path = f"{out_dir}/{filename}.csv"
    metrics_df.to_csv(csv_path, index=False)
    log.info(f"Metrics table saved → {csv_path}")


def run_evaluation(
    results: list,
    out_dir: str = "outputs/tables",
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
    best       = select_best_models(ranked_df)

    save_metrics_table(ranked_df, out_dir)

    print("\n── Model ranking (2017 evaluation) ──────────────────────────────────")
    for col_name in ["series_name", "mode", "model", "MAE", "RMSE", "MAPE", "rank_score"]:
        if col_name not in ranked_df.columns:
            ranked_df[col_name] = float("nan")
    display_cols = [c for c in ["series_name", "mode", "model", "MAE", "RMSE", "MAPE", "rank_score"]
                    if c in ranked_df.columns]
    print(ranked_df[display_cols].to_string(index=False))

    return metrics_df, ranked_df, best


def merge_forecast_files(
    series_names: list,
    models: list,
    forecast_dir: str = "outputs/forecasts",
    out_dir: str = "outputs/forecasts",
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
