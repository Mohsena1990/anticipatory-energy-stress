"""
household_stream.py
───────────────────
ENABLE UK household survey stream (Stages 5–8).

Stages
──────
  Stage 5 : Load and preprocess ENABLE.EU UK household survey data
            Builds construct scores, AEV composite, HighAEV binary target,
            energy-poverty flags, and attaches FES context columns.
  Stage 6 : Construct validation — Cronbach α, McDonald's ω, AVE, HTMT, EFA
  Stage 7 : COR path analysis + bootstrap mediation
            (Insecurity → Resource Preservation → Thermal Discomfort)
  Stage 8 : Unsupervised latent robustness (PCA / EFA / linear autoencoder)

Output
──────
  outputs/enable_cleaned/enable_aev_scored.csv   ← consumed by ml_pipeline.py
  outputs/construct_validation/
  outputs/sem_mediation/
  outputs/unsupervised_latent_robustness/

Prerequisite
────────────
  Stage 5 attaches FES context by default (--no-fes to skip).
  Requires forecast_pipeline.py to have run first so that
  outputs/fes/fes_monthly_2017.csv exists.

Usage
─────
  python household_stream.py              # full run (with FES attachment)
  python household_stream.py --no-fes    # skip FES context attachment
"""

from __future__ import annotations
import argparse
import time
import pandas as pd

from src.logging_utils import setup_logger, get_logger
setup_logger("energy_stress", log_file="outputs/logs/pipeline.log")
log = get_logger("household_stream")


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

def run(attach_fes: bool = True) -> pd.DataFrame:
    """
    Execute Stages 5–8.

    Parameters
    ----------
    attach_fes : if True (default), attach FES context columns from
                 outputs/fes/fes_monthly_2017.csv to the household DataFrame.
                 Set to False if forecast_pipeline.py has not been run yet.

    Returns
    -------
    df : scored and annotated ENABLE UK DataFrame
         (also saved to outputs/enable_cleaned/enable_aev_scored.csv)
    """
    from src import paths
    paths.ensure_dirs()

    _stage(5, "Load and preprocess ENABLE UK household data")
    from src.enable_preprocessing import run as run_enable
    df = run_enable(attach_fes=attach_fes)
    log.info("Stage 5 complete — %d households, %d columns", len(df), df.shape[1])

    _stage(6, "Construct validation (Cronbach α, ω, AVE, HTMT, EFA)")
    from src.construct_validation import run as run_validation
    run_validation(df)

    _stage(7, "COR path analysis + bootstrap mediation")
    from src.sem_mediation import run as run_sem
    run_sem(df)

    _stage(8, "Unsupervised latent robustness (PCA / EFA / autoencoder)")
    from src.unsupervised_latent import run as run_latent
    run_latent(df)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Household stream — Stages 5–8 (ENABLE UK survey)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--no-fes", action="store_true",
        help="Skip attaching FES context (use when forecast_pipeline has not been run)",
    )
    args = parser.parse_args()

    _banner("Anticipatory Energy–Carbon Stress — Household Stream (Stages 5–8)")
    print(f"  FES context : {'disabled (--no-fes)' if args.no_fes else 'enabled'}")

    t0 = time.time()
    run(attach_fes=not args.no_fes)
    _banner(f"Household stream complete  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
