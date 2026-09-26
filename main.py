"""
main.py
───────
Master orchestrator for the full Anticipatory Energy–Carbon Stress pipeline.

Two sub-pipelines
─────────────────
  forecast_pipeline.py  — Stage 1 (0–4): macro data → 4 models → equal-weighted
                           FES index (single-year, or --rolling for a
                           walk-forward year-by-year forecast)
  household_stream.py   — Stage 2–5 : UKHLS (Study 6614) household panel →
                           COR-SEM + FES-conditioned COR-CVAE (Stage 2) →
                           fuzzy/one-class vulnerability identification +
                           driver analysis (Stage 3) → policy geography
                           maps (Stage 4)

Supersedes the ENABLE-based household stream and its separate
ml_pipeline.py CatBoost/SHAP stage (see project plan:
/home/mohsen/.claude/plans/linked-tinkering-moonbeam.md) — Stage 3 now
folds vulnerability identification and driver analysis into one script.
The legacy ENABLE-based scripts (ml_pipeline.py, src/enable_preprocessing.py,
src/cor_sem.py, src/cor_vae.py, src/construct_validation.py,
src/construct_mapping.py, src/unsupervised_latent.py, src/route_comparison.py,
src/route_utils.py, src/ml_classification.py, src/shap_explainability.py,
src/ts_shap.py, src/fes_scenarios.py) have been removed from the repo
entirely -- recoverable from git history if ever needed for comparison.
src/sem_mediation.py was trimmed to just its one still-reused function
(`ols_path`, called by src.ukhls_cor_sem) rather than removed outright,
since that one function isn't ENABLE-specific.

Run individually
────────────────
  python forecast_pipeline.py [options]      # Stage 1
  python household_stream.py [--skip-cvae]   # Stage 2–5

Run via orchestrator
────────────────────
  python main.py                              # full pipeline (all stages)
  python main.py --stage forecast             # Stage 1 only
  python main.py --stage household            # Stage 2–5 only

Forecast options (active when --stage forecast or all)
──────────────────────────────────────────────────────
  --fast                         fewer LSTM/TFT epochs (development mode)
  --skip-models LSTM TFT         skip specific forecasting models
  --series gas electricity       run specific series only
  --tune                         hyperparameter tuning before training
  --fes-only                     skip training; recompute FES from saved CSVs
  --selection-basis validation   model selection criterion
  --target-year 2025             single-year mode: forecast this year (default: 2025)
  --max-target-year 2025         --rolling only: cap the walk-forward at this year (default: 2025)

Household options (active when --stage household or all)
──────────────────────────────────────────────────────────
  --skip-cvae  skip Stage 2c (COR-CVAE, the most expensive stage)
"""

from __future__ import annotations
import argparse
import time

from src.logging_utils import setup_logger, get_logger
from src.config import DEFAULT_TARGET_YEAR
setup_logger("energy_stress", log_file="outputs_v2/logs/pipeline.log")
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
        ),
    )

    p.add_argument(
        "--stage",
        choices=["all", "forecast", "household"],
        default="all",
        help=(
            "all       : run both sub-pipelines in sequence (default)\n"
            "forecast  : Stage 1 (macro models + FES)\n"
            "household : Stage 2–5 (UKHLS panel, COR-SEM/CVAE, vulnerability identification, forward prediction)"
        ),
    )

    # ── Forecast options ──────────────────────────────────────────────────────
    fg = p.add_argument_group("forecast options (Stage 1)")
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
                    help="Hyperparameter tuning before final training -- ON by "
                         "default now, this flag is a harmless no-op kept for "
                         "backward compatibility. Use --no-tune to opt out.")
    fg.add_argument("--no-tune", action="store_true",
                    help="Skip hyperparameter tuning (opt out of the new "
                         "tuning-by-default behaviour). Applies to both single-year "
                         "and --rolling mode (rolling tunes once and reuses the "
                         "result across every rolling year -- see "
                         "forecast_pipeline.run_rolling's docstring).")
    fg.add_argument("--selection-basis",
                    choices=["forecast_actual", "validation"],
                    default="validation",
                    help="Model selection criterion. 'validation' (default) is "
                         "the genuine, non-hindsight walk-forward backtest choice. "
                         "'forecast_actual' selects using the target year's "
                         "now-known actuals -- hindsight, for retrospective "
                         "reporting on an already-realised year only.")
    fg.add_argument("--rolling", action="store_true",
                    help="Walk-forward rolling FES (train through year Y, forecast "
                         "Y+1, repeat for every feasible Y) instead of the single-year "
                         "path -- ~n_years x 24 model fits, 45min+ even in --fast mode. "
                         "Opt-in only, never runs by default. Ignores --fes-only/--tune.")
    fg.add_argument("--target-year", type=int, default=DEFAULT_TARGET_YEAR, metavar="YYYY",
                    help="Single-year mode only: forecast exactly this year "
                         f"(default: {DEFAULT_TARGET_YEAR}, aligned with the UKHLS "
                         "panel's own coverage -- see src.config.DEFAULT_TARGET_YEAR). "
                         "Pass 0 for the fully dynamic latest-available-year detection.")
    fg.add_argument("--max-target-year", type=int, default=DEFAULT_TARGET_YEAR, metavar="YYYY",
                    help="--rolling only: cap the walk-forward at this target year "
                         f"(default: {DEFAULT_TARGET_YEAR}). Pass 0 to remove the cap.")

    # ── Household options ─────────────────────────────────────────────────────
    hg = p.add_argument_group("household options (Stage 2–5)")
    hg.add_argument("--skip-cvae", action="store_true",
                    help="Skip Stage 2c (COR-CVAE, the most expensive stage)")

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
    target_year      = None if args.target_year == 0 else args.target_year
    max_target_year  = None if args.max_target_year == 0 else args.max_target_year
    tune             = not args.no_tune   # tuning is ON by default; --no-tune opts out

    _banner("Anticipatory Energy–Carbon Stress Index — Full Pipeline")
    print(f"  Stage       : {args.stage}")
    if args.stage in ("all", "forecast"):
        print(f"  Series      : {series}")
        print(f"  Models      : {models_to_run}")
        print(f"  Fast mode   : {args.fast}")
        print(f"  Rolling     : {args.rolling}")
        if args.rolling:
            print(f"  Max target year : {max_target_year or '(uncapped)'}")
        else:
            print(f"  FES only    : {args.fes_only}")
            print(f"  Tuning      : {tune}")
            print(f"  Selection   : {args.selection_basis}")
            print(f"  Target year : {target_year or '(dynamic)'}")
    if args.stage in ("all", "household"):
        print(f"  COR-CVAE    : {'disabled' if args.skip_cvae else 'enabled'}")

    t0 = time.time()

    # ── Stage 1 : Forecast pipeline ──────────────────────────────────────────
    if args.stage in ("all", "forecast"):
        if args.rolling:
            from forecast_pipeline import run_rolling
            run_rolling(
                series=series, models_to_run=models_to_run,
                fast=args.fast, selection_basis=args.selection_basis,
                max_target_year=max_target_year,
                tune=tune,
            )
        else:
            from forecast_pipeline import run as run_forecast
            run_forecast(
                series=series,
                models_to_run=models_to_run,
                fast=args.fast,
                fes_only=args.fes_only,
                tune=tune,
                selection_basis=args.selection_basis,
                target_year=target_year,
            )

    # ── Stage 2–5 : Household stream (UKHLS COR-SEM/CVAE + vulnerability + forward prediction) ──
    if args.stage in ("all", "household"):
        from household_stream import run as run_household
        run_household(skip_cvae=args.skip_cvae)

    _banner(f"Pipeline complete  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
