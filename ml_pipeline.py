"""
ml_pipeline.py
──────────────
ML classification + SHAP explainability stream (Stages 9–10).

Stages
──────
  Stage 9  : CatBoost classification of High Adaptive Energy Vulnerability (HighAEV)
             Trains on COR-derived behavioural scores and household characteristics.
             Saves predictions, confusion matrix, ROC curve, PR curve.
  Stage 10 : SHAP explainability
             Feature importance, beeswarm plots, dependence plots, household examples.

Input
─────
  outputs/enable_cleaned/enable_aev_scored.csv   (produced by household_stream.py)
  Pass a DataFrame directly when calling run(df=...) from main.py to skip disk I/O.

Output
──────
  outputs/ml_classification/
  outputs/shap/

Prerequisite
────────────
  Run household_stream.py (Stages 5–8) first so the scored dataset exists.

Usage
─────
  python ml_pipeline.py               # load from CSV, run both stages
  python ml_pipeline.py --no-shap    # skip SHAP (Stage 10)
"""

from __future__ import annotations
import argparse
import time
import pandas as pd

from src.logging_utils import setup_logger, get_logger
from src import paths
setup_logger("energy_stress", log_file="outputs/logs/pipeline.log")
log = get_logger("ml_pipeline")


# ══════════════════════════════════════════════════════════════════════════════
# Console helpers
# ══════════════════════════════════════════════════════════════════════════════

def _banner(text: str) -> None:
    print("\n" + "═" * 70)
    print(f"  {text}")
    print("═" * 70)


def _stage(n: int, label: str) -> None:
    print(f"\n[STAGE {n}] {label}")
    log.info("=" * 40)
    log.info("STAGE %d: %s", n, label)
    log.info("=" * 40)


# ══════════════════════════════════════════════════════════════════════════════
# Public run() — callable from main.py
# ══════════════════════════════════════════════════════════════════════════════

def run(df: pd.DataFrame | None = None, run_shap: bool = True) -> None:
    """
    Execute Stages 9–10.

    Parameters
    ----------
    df       : scored household DataFrame from household_stream.run().
               If None, loaded from outputs/enable_cleaned/enable_aev_scored.csv.
    run_shap : if True (default), run Stage 10 SHAP explainability.
    """
    paths.ensure_dirs()

    if df is None:
        if not paths.ENABLE_SCORED.exists():
            raise FileNotFoundError(
                f"Scored dataset not found: {paths.ENABLE_SCORED}\n"
                "Run household_stream.py (Stages 5–8) first."
            )
        df = pd.read_csv(paths.ENABLE_SCORED)
        log.info("Loaded scored dataset: %d rows, %d columns", len(df), df.shape[1])

    _stage(9, "CatBoost classification of High Adaptive Energy Vulnerability")
    from src.ml_classification import run as run_ml
    comparison_df, all_models = run_ml(df)

    if run_shap:
        _stage(10, "SHAP explainability")
        from src.shap_explainability import run as run_shap_fn

        # Run SHAP on the controls-only primary model (generalizable prediction)
        primary_key = "Controls_Only" if "Controls_Only" in all_models else "SEM_COR"
        if primary_key in all_models:
            primary = all_models[primary_key]
            log.info("SHAP: %s model (primary generalizable predictor)", primary_key)
            run_shap_fn(
                model=primary["model"],
                X_test=primary["X_test"],
                y_test=primary["y_test"],
                y_prob=primary["y_prob"],
                cat_idx=primary["cat_idx"],
                model_label=primary_key.lower(),
            )

        # Also run SHAP on the best non-circular model by test ROC-AUC
        circular_keys = {"SEM_COR", "Hybrid_SEM_AE"}
        if not comparison_df.empty:
            non_circular = comparison_df[~comparison_df["model_key"].isin(circular_keys)]
            if not non_circular.empty:
                best_row = non_circular.sort_values("roc_auc", ascending=False).iloc[0]
                best_key = best_row["model_key"]
                if best_key != primary_key and best_key in all_models:
                    best = all_models[best_key]
                    log.info("SHAP: best non-circular model = %s (ROC-AUC=%.4f)",
                             best_key, best_row["roc_auc"])
                    run_shap_fn(
                        model=best["model"],
                        X_test=best["X_test"],
                        y_test=best["y_test"],
                        y_prob=best["y_prob"],
                        cat_idx=best["cat_idx"],
                        model_label=best_key.lower(),
                    )


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ML pipeline — Stages 9–10 (CatBoost + SHAP)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--no-shap", action="store_true",
        help="Skip SHAP explainability (Stage 10)",
    )
    args = parser.parse_args()

    _banner("Anticipatory Energy–Carbon Stress — ML Pipeline (Stages 9–10)")
    print(f"  SHAP : {'disabled (--no-shap)' if args.no_shap else 'enabled'}")

    t0 = time.time()
    run(run_shap=not args.no_shap)
    _banner(f"ML pipeline complete  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
