"""
household_stream.py
───────────────────
ENABLE UK household survey stream (Stages 5–9).

Three comparable COR estimation routes
───────────────────────────────────────
  Route 1 — COR Composite Route  : operationalises COR-informed formative
                                    composites (Stage 6)
  Route 2 — COR-Informed SEM     : estimates reflective latent variables
                                    through CFA/SEM (Stage 7)
  Route 3 — COR-Informed VAE     : learns COR-aligned deep latent
                                    representations through a VAE (Stage 8)
  Cross-route comparison         : Stage 9

FES is macro-level contextual background only (identical annual value for
every UK household in this sample) — it is attached to the household
DataFrame as metadata/context in Stage 5 and is never used as a feature in
any of the three routes' models. See `src.route_utils` for the shared
safeguard that enforces this in the ML stream.

Stages
──────
  Stage 5 : Load and preprocess ENABLE.EU UK household survey data.
            Cleans missing/refusal codes, applies the shared codebook
            recoding rules from `src.construct_mapping`, builds normalized
            item columns n_{item}, and attaches FES as context metadata
            only (--no-fes to skip).
  Stage 6 : Route 1 — COR Composite Route.
            Formative composite scores (fcp_score, aemc_score, bli_score,
            tcr_score) → aev_score → high_aev.  Includes construct
            validation (Cronbach α, McDonald's ω, CR, AVE, HTMT,
            Fornell–Larcker, EFA), COR path analysis + bootstrap mediation,
            and empirical-recovery robustness (PCA / EFA / linear
            autoencoder alignment).
  Stage 7 : Route 2 — COR-Informed SEM.
            CFA measurement model + second-order structural model
            (sem_fcp_latent, sem_aemc_latent, sem_bli_latent,
            sem_tcr_latent) → sem_aev_score → sem_high_aev.  Comparative/
            robustness route, evaluated cautiously (see CFA fit indices).
  Stage 8 : Route 3 — COR-Informed VAE (skippable via --skip-vae).
            Theory-informed VAE latents (vae_fcp_latent, vae_aemc_latent,
            vae_bli_latent, vae_tcr_latent) → vae_aev_score →
            vae_high_aev.  Not independent of Route 1 (its alignment
            target and prediction label both come from Route 1).
  Stage 9 : Cross-route comparison (Route 1 vs Route 2 vs Route 3) — the
            central comparison mechanism; degrades gracefully if Route 2
            and/or Route 3 are unavailable.

After Stage 9, if `outputs/fes/fes_scenario_summary.csv` exists (produced by
`forecast_pipeline.py`'s Stage 4 scenario simulation), this module also
builds `outputs/fes_highaev_interpretation/scenario_highaev_interpretation_matrix.csv`
— a scenario-conditioned interpretation table linking each of the 9 FES
scenarios to the already-estimated HighAEV prevalence. This does not model
or predict HighAEV: the household label doesn't change across scenarios,
only the macro-stress context it is interpreted under.

Output
──────
  outputs/enable_cleaned/enable_aev_scored.csv   ← consumed by ml_pipeline.py
                                                    (includes Route 1 + Route 2
                                                    + Route 3 score columns
                                                    once Stages 7/8 complete)
  outputs/construct_validation/
  outputs/sem_mediation/
  outputs/unsupervised_latent_robustness/
  outputs/cor_sem/
  outputs/cor_vae/
  outputs/route_comparison/
  outputs/fes_highaev_interpretation/

Prerequisite
────────────
  Stage 5 attaches FES context by default (--no-fes to skip).
  Requires forecast_pipeline.py to have run first so that
  outputs/fes/fes_monthly_2017.csv exists.

Usage
─────
  python household_stream.py               # full run (with FES attachment)
  python household_stream.py --no-fes      # skip FES context attachment
  python household_stream.py --skip-vae    # skip Stage 8 (Route 3 VAE, the
                                            # most expensive stage) — Stage 9
                                            # falls back to a 2-route
                                            # (Route 1 vs 2) comparison
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

def run(attach_fes: bool = True, skip_vae: bool = False) -> pd.DataFrame:
    """
    Execute Stages 5–9.

    Parameters
    ----------
    attach_fes : if True (default), attach FES context columns from
                 outputs/fes/fes_monthly_2017.csv to the household DataFrame
                 as metadata only — FES is never used as a model feature in
                 any route (see `src.route_utils.exclude_fes_columns`).
                 Set to False if forecast_pipeline.py has not been run yet.
    skip_vae   : if True, skip Stage 8 (Route 3 COR-VAE) — the most
                 expensive stage. Stage 9's cross-route comparison then
                 degrades to a 2-route (Route 1 vs Route 2) comparison.

    Returns
    -------
    df : scored and annotated ENABLE UK DataFrame — Route 1 columns always
         present; Route 2 (sem_*) / Route 3 (vae_*) columns present when
         those routes succeeded.
         (also saved to outputs/enable_cleaned/enable_aev_scored.csv)
    """
    from src import paths
    paths.ensure_dirs()

    _stage(5, "Load and preprocess ENABLE UK household data (FES attached as context only)")
    from src.enable_preprocessing import run as run_enable
    df = run_enable(attach_fes=attach_fes)
    log.info("Stage 5 complete — %d households, %d columns", len(df), df.shape[1])

    _stage(6, "Route 1: COR Composite Route (formative composites, validation, path analysis, empirical recovery)")
    from src.construct_validation import run as run_validation
    run_validation(df)
    from src.sem_mediation import run as run_sem
    run_sem(df)
    from src.unsupervised_latent import run as run_latent
    run_latent(df)

    _stage(7, "Route 2: COR-informed SEM (reflective CFA measurement + structural model)")
    from src.cor_sem import run as run_cor_sem
    try:
        sem_result = run_cor_sem(df)
    except Exception as e:
        log.error("Route 2 (SEM) failed entirely: %s — continuing without Route 2 scores.", e)
        sem_result = None
    route2_scores = sem_result.get("measurement", {}).get("scores_named") if sem_result else None
    if route2_scores is not None:
        df = df.join(route2_scores, how="left")
        log.info("Route 2 scores merged into household frame: %s", list(route2_scores.columns))
    else:
        log.warning("Route 2 (SEM) unavailable this run — Stage 9 will compare available routes only.")

    route3_scores = None
    if skip_vae:
        log.info("Stage 8 skipped (--skip-vae)")
    else:
        _stage(8, "Route 3: COR-informed VAE (COR-aligned deep latent representations)")
        from src.cor_vae import run as run_cor_vae
        try:
            vae_result = run_cor_vae(df)
        except Exception as e:
            log.error("Route 3 (VAE) failed entirely: %s — continuing without Route 3 scores.", e)
            vae_result = None
        route3_scores = vae_result.get("scores_named") if vae_result else None
        if route3_scores is not None:
            df = df.join(route3_scores, how="left")
            log.info("Route 3 scores merged into household frame: %s", list(route3_scores.columns))
        else:
            log.warning("Route 3 (VAE) unavailable this run — Stage 9 will compare available routes only.")

    _stage(9, "Cross-route comparison (Route 1 vs Route 2 vs Route 3)")
    from src.route_comparison import run as run_route_comparison
    run_route_comparison(df, route2_scores=route2_scores, route3_scores=route3_scores)

    # FES scenario <-> HighAEV interpretation matrix — scenario-conditioned
    # interpretation only, never a household-level prediction. Requires
    # forecast_pipeline.py to have already produced fes_scenario_summary.csv.
    try:
        if paths.FES_SCENARIO_SUMMARY_FILE.exists():
            from src.fes_scenarios import (
                build_scenario_highaev_interpretation_matrix,
                plot_highaev_interpretation_matrix,
            )
            scenario_summary_df = pd.read_csv(paths.FES_SCENARIO_SUMMARY_FILE)
            route_comparison_csv = paths.ROUTE_COMPARISON_TABLES / "cross_route_outcome_agreement.csv"
            matrix_df = build_scenario_highaev_interpretation_matrix(
                scenario_summary_df, df, route_comparison_csv=str(route_comparison_csv),
            )
            paths.FES_HIGHAEV_OUT.mkdir(parents=True, exist_ok=True)
            matrix_df.to_csv(paths.FES_HIGHAEV_MATRIX, index=False)
            log.info("Scenario-HighAEV interpretation matrix saved: %s", paths.FES_HIGHAEV_MATRIX)
            plot_highaev_interpretation_matrix(
                matrix_df, str(paths.FIGURES_DIR / "fes_highaev_interpretation_matrix.png"),
            )
        else:
            log.warning(
                "FES scenario summary not found (%s) — skipping scenario-HighAEV "
                "interpretation matrix. Run forecast_pipeline.py first.",
                paths.FES_SCENARIO_SUMMARY_FILE,
            )
    except Exception as e:
        log.warning("Scenario-HighAEV interpretation matrix failed (non-fatal): %s", e)

    # Persist the fully merged dataset (Route 1 + any available Route 2/3
    # columns) so downstream consumers (standalone ml_pipeline.py runs,
    # notebooks, chapter reports) see the complete picture.
    df.to_csv(paths.ENABLE_SCORED, index=False)
    log.info("Saved merged scored dataset: %s (%d rows, %d columns)",
             paths.ENABLE_SCORED.name, len(df), df.shape[1])

    return df


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Household stream — Stages 5–9 (ENABLE UK survey, 3 COR routes)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--no-fes", action="store_true",
        help="Skip attaching FES context (use when forecast_pipeline has not been run)",
    )
    parser.add_argument(
        "--skip-vae", action="store_true",
        help="Skip Stage 8 (Route 3 COR-VAE) — the most expensive stage",
    )
    args = parser.parse_args()

    _banner("Anticipatory Energy–Carbon Stress — Household Stream (Stages 5–9)")
    print(f"  FES context : {'disabled (--no-fes)' if args.no_fes else 'enabled (context only, never a model feature)'}")
    print(f"  Route 3 VAE : {'disabled (--skip-vae)' if args.skip_vae else 'enabled'}")

    t0 = time.time()
    run(attach_fes=not args.no_fes, skip_vae=args.skip_vae)
    _banner(f"Household stream complete  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
