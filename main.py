"""
main.py
───────
Master orchestrator for the full Anticipatory Energy–Carbon Stress pipeline.

Three sub-pipelines
───────────────────
  forecast_pipeline.py  — Stages 0–4 : macro data → 4 models → FES index + figures
  household_stream.py   — Stages 5–8 : ENABLE UK survey → constructs → SEM → latent
  ml_pipeline.py        — Stages 9–10: CatBoost HighAEV classifier → SHAP

Run individually
────────────────
  python forecast_pipeline.py [options]   # Stages 0–4
  python household_stream.py [--no-fes]  # Stages 5–8
  python ml_pipeline.py      [--no-shap] # Stages 9–10

Run via orchestrator
────────────────────
  python main.py                              # full pipeline (all stages)
  python main.py --stage forecast             # Stages 0–4 only
  python main.py --stage household            # Stages 5–8 only
  python main.py --stage ml                  # Stages 9–10 only

Forecast options (active when --stage forecast or all)
──────────────────────────────────────────────────────
  --fast                         fewer LSTM/TFT epochs (development mode)
  --skip-models LSTM TFT         skip specific forecasting models
  --series gas electricity       run specific series only
  --tune                         hyperparameter tuning before training
  --fes-only                     skip training; recompute FES from saved CSVs
  --selection-basis validation   model selection criterion

Household options (active when --stage household or all)
────────────────────────────────────────────────────────
  --no-fes    skip FES context attachment (if forecast pipeline has not run)

ML options (active when --stage ml or all)
──────────────────────────────────────────
  --no-shap   skip SHAP explainability (Stage 10)
"""

from __future__ import annotations
import argparse
import time

from src.logging_utils import setup_logger, get_logger
setup_logger("energy_stress", log_file="outputs/logs/pipeline.log")
log = get_logger("main")


# ══════════════════════════════════════════════════════════════════════════════
# Console helpers
# ══════════════════════════════════════════════════════════════════════════════

def _banner(text: str) -> None:
    print("\n" + "═" * 70)
    print(f"  {text}")
    print("═" * 70)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Anticipatory Energy–Carbon Stress — full pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Sub-pipeline scripts can also be run independently:\n"
            "  python forecast_pipeline.py --help\n"
            "  python household_stream.py  --help\n"
            "  python ml_pipeline.py       --help\n"
        ),
    )

    p.add_argument(
        "--stage",
        choices=["all", "forecast", "household", "ml"],
        default="all",
        help=(
            "all       : run all three sub-pipelines in sequence (default)\n"
            "forecast  : Stages 0–4 (macro models + FES)\n"
            "household : Stages 5–8 (ENABLE UK survey stream)\n"
            "ml        : Stages 9–10 (CatBoost + SHAP)"
        ),
    )

    # ── Forecast options ──────────────────────────────────────────────────────
    fg = p.add_argument_group("forecast options (Stages 0–4)")
    fg.add_argument("--fast", action="store_true",
                    help="Fewer LSTM/TFT epochs (development mode)")
    fg.add_argument("--fes-only", action="store_true",
                    help="Skip training; recompute FES from existing forecast CSVs")
    fg.add_argument("--skip-models", nargs="+", default=[], metavar="MODEL",
                    help="Models to skip: SARIMA Prophet LSTM TFT")
    fg.add_argument("--series", nargs="+", default=None,
                    choices=["gas", "electricity", "carbon"],
                    help="Series to forecast (default: all three)")
    fg.add_argument("--tune", action="store_true",
                    help="Hyperparameter tuning before final training")
    fg.add_argument("--selection-basis",
                    choices=["forecast_actual", "validation"],
                    default="forecast_actual",
                    help="Model selection criterion (default: forecast_actual)")

    # ── Household options ─────────────────────────────────────────────────────
    hg = p.add_argument_group("household options (Stages 5–8)")
    hg.add_argument("--no-fes", action="store_true",
                    help="Skip FES context attachment to household data")

    # ── ML options ────────────────────────────────────────────────────────────
    mg = p.add_argument_group("ML options (Stages 9–10)")
    mg.add_argument("--no-shap", action="store_true",
                    help="Skip SHAP explainability (Stage 10)")

    return p


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    ALL_MODELS    = ["SARIMA", "Prophet", "LSTM", "TFT"]
    models_to_run = [m for m in ALL_MODELS if m not in args.skip_models]
    series        = args.series or ["gas", "electricity", "carbon"]

    _banner("Anticipatory Energy–Carbon Stress Index — Full Pipeline")
    print(f"  Stage       : {args.stage}")
    if args.stage in ("all", "forecast"):
        print(f"  Series      : {series}")
        print(f"  Models      : {models_to_run}")
        print(f"  Fast mode   : {args.fast}")
        print(f"  FES only    : {args.fes_only}")
        print(f"  Tuning      : {args.tune}")
        print(f"  Selection   : {args.selection_basis}")
    if args.stage in ("all", "household"):
        print(f"  FES context : {'disabled' if args.no_fes else 'enabled'}")
    if args.stage in ("all", "ml"):
        print(f"  SHAP        : {'disabled' if args.no_shap else 'enabled'}")

    t0 = time.time()
    household_df = None

    # ── Stages 0–4 : Forecast pipeline ───────────────────────────────────────
    if args.stage in ("all", "forecast"):
        from forecast_pipeline import run as run_forecast
        run_forecast(
            series=series,
            models_to_run=models_to_run,
            fast=args.fast,
            fes_only=args.fes_only,
            tune=args.tune,
            selection_basis=args.selection_basis,
        )

    # ── Stages 5–8 : Household stream ────────────────────────────────────────
    if args.stage in ("all", "household"):
        from household_stream import run as run_household
        household_df = run_household(attach_fes=not args.no_fes)

    # ── Stages 9–10 : ML pipeline ────────────────────────────────────────────
    if args.stage in ("all", "ml"):
        from ml_pipeline import run as run_ml
        # Pass df in-memory when household ran in this session (avoids re-loading CSV)
        run_ml(df=household_df, run_shap=not args.no_shap)

    _banner(f"Pipeline complete  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
