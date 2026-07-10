"""
ml_pipeline.py
──────────────
ML classification + SHAP explainability stream (Stages 10–11).

Stages
──────
  Stage 10 : CatBoost classification of High Adaptive Energy Vulnerability (HighAEV)
             Trains one model per COR estimation route (Controls_Only,
             Route1_Composite, Route2_SEM, Route3_VAE, AllRoutes_Hybrid).
             Target is Route 1's high_aev unless explicitly configured
             otherwise. FES columns are excluded from every feature set
             (see src.route_utils.exclude_fes_columns) — FES is contextual
             macro background, never a household-level predictor.
             Controls_Only is the only fully generalizable predictor;
             Route1_Composite is circular by construction; Route2_SEM/
             Route3_VAE are construct-overlap / representation-validation
             models (not fully independent of Route 1); AllRoutes_Hybrid
             contains multiple overlapping/circular components.
             Saves predictions, confusion matrix, ROC curve, PR curve.
  Stage 11 : SHAP explainability
             Run primarily on Controls_Only. Also runs on the best route
             model, labelled as construct-overlap / route-validation
             explanation (not a generalizable-prediction claim).
             Feature importance, beeswarm plots, dependence plots, household examples.
             (Route 3's own prediction-head SHAP, over its z1–z4 latent
             dimensions, is produced separately by src.cor_vae during
             Stage 8 of household_stream.py.)

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
  Run household_stream.py (Stages 5–9) first so the scored dataset exists.

Usage
─────
  python ml_pipeline.py               # load from CSV, run both stages (10–11)
  python ml_pipeline.py --no-shap    # skip SHAP (Stage 11)
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
    Execute Stages 10–11.

    Parameters
    ----------
    df       : scored household DataFrame from household_stream.run().
               If None, loaded from outputs/enable_cleaned/enable_aev_scored.csv.
    run_shap : if True (default), run Stage 11 SHAP explainability.
    """
    paths.ensure_dirs()

    if df is None:
        if not paths.ENABLE_SCORED.exists():
            raise FileNotFoundError(
                f"Scored dataset not found: {paths.ENABLE_SCORED}\n"
                "Run household_stream.py (Stages 5–9) first."
            )
        df = pd.read_csv(paths.ENABLE_SCORED)
        log.info("Loaded scored dataset: %d rows, %d columns", len(df), df.shape[1])

    _stage(10, "CatBoost classification of High Adaptive Energy Vulnerability (per route)")
    from src.ml_classification import run as run_ml
    comparison_df, all_models = run_ml(df)

    if run_shap:
        _stage(11, "SHAP explainability")
        from src.shap_explainability import run as run_shap_fn

        # Run SHAP on the controls-only primary model (generalizable prediction)
        primary_key = "Controls_Only" if "Controls_Only" in all_models else "Route1_Composite"
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

        # Also run SHAP on the best route model by test ROC-AUC, EXCLUDING the
        # fully circular ones (Route1_Composite / AllRoutes_Hybrid). Route2_SEM
        # and Route3_VAE are not fully independent of Route 1 either (shared
        # item pool / Route 1 alignment target & label), so this second SHAP
        # run is a construct-overlap / route-validation explanation — not a
        # claim of generalizable prediction.
        circular_keys = {"Route1_Composite", "AllRoutes_Hybrid"}
        if not comparison_df.empty:
            non_circular = comparison_df[~comparison_df["model_key"].isin(circular_keys)]
            if not non_circular.empty:
                best_row = non_circular.sort_values("roc_auc", ascending=False).iloc[0]
                best_key = best_row["model_key"]
                if best_key != primary_key and best_key in all_models:
                    best = all_models[best_key]
                    log.info("SHAP: best construct-overlap/route-validation model = %s (ROC-AUC=%.4f)",
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
        description="ML pipeline — Stages 10–11 (CatBoost + SHAP)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--no-shap", action="store_true",
        help="Skip SHAP explainability (Stage 11)",
    )
    args = parser.parse_args()

    _banner("Anticipatory Energy–Carbon Stress — ML Pipeline (Stages 10–11)")
    print(f"  SHAP : {'disabled (--no-shap)' if args.no_shap else 'enabled'}")

    t0 = time.time()
    run(run_shap=not args.no_shap)
    _banner(f"ML pipeline complete  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
