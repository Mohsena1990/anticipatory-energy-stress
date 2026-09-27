"""
household_stream.py
───────────────────
LEGACY (v1). The submitted-draft-v1 household stream. The v2 rerun does not
use it: v2 runs the per-stage scripts in scripts/ (see README.md and
analysis_plan_rerun.md), and the COR-SEM / COR-CVAE / fuzzy / SVM / policy-map
results below are superseded (CVAE, fuzzy and SVM survive only as a labelled
v1 appendix figure). src/paths.py now points OUTPUTS_DIR at outputs_v2/, so
the "outputs/" paths listed under Output refer to the v1 run.

UKHLS (Understanding Society, UK Data Service Study 6614) household panel
stream — Stage 2 (latent variable extraction), Stage 3 (vulnerability
identification), Stage 4 (policy geography maps), and Stage 5 (forward
vulnerability prediction).

Supersedes the ENABLE.EU-based household stream (see project plan:
/home/mohsen/.claude/plans/linked-tinkering-moonbeam.md). ENABLE was a
single UK cross-section (2017 only) — every household shared one FES
value, so forecasted/actual prices could only ever be background context.
UKHLS is a 15-wave panel (waves a-o, ~2009-2024): each household's
interview year differs, so realised energy-price growth is a genuine
row-level signal, not one shared constant. The legacy ENABLE-based modules
(enable_preprocessing, cor_sem, cor_vae, construct_validation,
unsupervised_latent, route_comparison, ml_pipeline/ml_classification,
route_utils, shap_explainability, ts_shap) have been removed from the
repo entirely -- they are not called from this stream and are recoverable
from git history if ever needed for comparison.

Stages
──────
  Stage 2a : Build the UKHLS household-wave panel (src.ukhls_preprocessing)
             — load/clean 15 waves, compute fuel_to_income_ratio and the
             high_fuel_vulnerable target, attach realised price-growth
             context per household's interview year.
  Stage 2b : COR-SEM (src.ukhls_cor_sem) — 4-factor CFA (Object/Condition/
             Personal/Energy) -> second-order BASELINE resource-stock
             factor -> FES-moderation OLS test (does baseline resource
             stock dampen the effect of energy-price stress on fuel
             burden?).
  Stage 2c : FES-conditioned COR-CVAE (src.ukhls_cor_cvae, skippable via
             --skip-cvae) — 4-dim latent space aligned to the Stage 2b SEM
             factor scores, with FES as a genuine encoder+decoder
             conditioning variable, plus a counterfactual P10/P90 FES
             scenario query.
  Stage 3  : Vulnerability identification (fuzzy c-means + one-class SVM,
             both validated against the objective fuel_to_income_ratio), a
             transparent logistic-regression driver analysis, and
             spread/driver policy figures (src.ukhls_vulnerability_classification).
  Stage 4  : Policy geography maps (src.ukhls_policy_maps) — a
             resource-to-stress hotspot map, a fuzzy-membership map, and a
             vulnerability vector-shift map, drawn on the 12 UK Government
             Office Regions' real boundaries (src.ukhls_geo_maps, ONS Open
             Geography Portal).
  Stage 5  : Forward vulnerability prediction (src.ukhls_forward_prediction)
             — answers "who is about to become vulnerable, and roughly
             when," not just "who is vulnerable now and why." Links
             households across consecutive waves via hrpid (household
             reference person's pidp), trains/walk-forward-validates a
             logistic model on known 2009-2024 wave-to-wave transitions,
             then applies it to the most recent wave to predict each
             household's own next-year vulnerability -- genuinely forward,
             not yet observed.

Output
──────
  outputs/ukhls_cleaned/ukhls_panel.csv   ← full panel + SEM/CVAE scores
  outputs/ukhls_cor_sem/
  outputs/ukhls_cor_cvae/
  outputs/ukhls_vulnerability/
  outputs/ukhls_policy_maps/
  outputs/ukhls_forward_prediction/

Prerequisite
────────────
  Stage 2a attaches realised price context from
  data/processed/core_energy_carbon.csv — run forecast_pipeline.py first
  (Stage 0-1) so that file exists; otherwise price columns are NaN.

Usage
─────
  python household_stream.py               # full run (Stage 2 + 3 + 4 + 5)
  python household_stream.py --skip-cvae   # skip Stage 2c (COR-CVAE, the
                                             # most expensive stage) — Stage
                                             # 3/4/5 then run on SEM scores only
"""

from __future__ import annotations
import argparse
import time
import pandas as pd

from src.logging_utils import setup_logger, get_logger
setup_logger("energy_stress", log_file="outputs_v2/logs/pipeline.log")
log = get_logger("household_stream")


# ══════════════════════════════════════════════════════════════════════════════
# Console helpers
# ══════════════════════════════════════════════════════════════════════════════

def _banner(text: str) -> None:
    print("\n" + "═" * 70)
    print(f"  {text}")
    print("═" * 70)


def _stage(n: str, label: str) -> None:
    print(f"\n[STAGE {n}] {label}")
    log.info("=" * 40)
    log.info("STAGE %s: %s", n, label)
    log.info("=" * 40)


# ══════════════════════════════════════════════════════════════════════════════
# Public run() — callable from main.py
# ══════════════════════════════════════════════════════════════════════════════

def run(skip_cvae: bool = False) -> pd.DataFrame:
    """
    Execute Stage 2 (panel build -> COR-SEM -> COR-CVAE), Stage 3
    (vulnerability identification), Stage 4 (policy geography maps), then
    Stage 5 (forward vulnerability prediction).

    Parameters
    ----------
    skip_cvae : if True, skip Stage 2c (COR-CVAE) — the most expensive
                stage. Stage 3 then runs on SEM factor scores only (no
                CVAE latent columns).

    Returns
    -------
    df : UKHLS household-wave panel joined with Stage 2b SEM factor scores
         and (unless skipped) Stage 2c CVAE latent scores.
         (also saved to outputs/ukhls_cleaned/ukhls_panel.csv)
    """
    from src import paths
    paths.ensure_dirs()

    _stage("2a", "Build UKHLS household-wave panel (15 waves, fuel-to-income target, realised price context)")
    from src.ukhls_preprocessing import run as run_panel
    df = run_panel()
    log.info("Stage 2a complete — %d household-wave rows, %d columns", len(df), df.shape[1])

    _stage("2a-overview", "Dataset overview (panel composition, missingness, key distributions)")
    from src.ukhls_dataset_overview import run as run_overview
    run_overview(df)

    _stage("2b", "COR-SEM (Object/Condition/Personal/Energy -> Baseline Resource Stock -> FES moderation)")
    from src.ukhls_cor_sem import run as run_sem
    sem_result = run_sem(df)
    sem_scores = sem_result.get("scores") if sem_result else None
    if sem_scores is None or sem_scores.empty:
        log.error("Stage 2b (COR-SEM) unavailable this run — Stage 2c/3 will be degraded.")
        sem_scores = pd.DataFrame(index=df.index)
    else:
        log.info("Stage 2b scores: %s", list(sem_scores.columns))

    cvae_scores = pd.DataFrame(index=df.index)
    counterfactual_df = pd.DataFrame()
    if skip_cvae:
        log.info("Stage 2c skipped (--skip-cvae)")
    else:
        _stage("2c", "FES-conditioned COR-CVAE (COR-aligned latent space, counterfactual FES scenario query)")
        from src.ukhls_cor_cvae import run as run_cvae
        cvae_result = run_cvae(df, sem_scores)
        if cvae_result:
            cvae_scores = cvae_result.get("scores", cvae_scores)
            counterfactual_df = cvae_result.get("counterfactual", counterfactual_df)
            log.info("Stage 2c scores: %s", list(cvae_scores.columns))
        else:
            log.warning("Stage 2c (COR-CVAE) unavailable this run — Stage 3 runs on SEM scores only.")

    _stage(3, "Vulnerability identification (fuzzy c-means + one-class SVM, driver analysis, policy figures)")
    from src.ukhls_vulnerability_classification import run as run_vulnerability
    vuln_result = run_vulnerability(df, sem_scores, cvae_scores)

    _stage(4, "Policy geography maps (real UK region boundaries: hotspot / fuzzy membership / vulnerability shift)")
    from src.ukhls_policy_maps import run as run_policy_maps
    run_policy_maps(df, sem_scores, vuln_result, counterfactual_df)

    _stage(5, "Forward vulnerability prediction (hrpid-linked wave transitions, 2009-2024 -> next-year prediction)")
    from src.ukhls_forward_prediction import run as run_forward_prediction
    run_forward_prediction(df, sem_scores)

    df = df.join(sem_scores, how="left").join(cvae_scores, how="left")
    df.to_csv(paths.UKHLS_PANEL, index=False)
    log.info("Saved merged UKHLS panel + latent scores: %s (%d rows, %d columns)",
             paths.UKHLS_PANEL.name, len(df), df.shape[1])

    return df


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Household stream — UKHLS Stage 2 (COR-SEM + COR-CVAE) + Stage 3 "
                     "(vulnerability identification) + Stage 4 (policy geography maps)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--skip-cvae", action="store_true",
        help="Skip Stage 2c (COR-CVAE) — the most expensive stage",
    )
    args = parser.parse_args()

    _banner("Anticipatory Energy–Carbon Stress — Household Stream (UKHLS Stage 2-5)")
    print(f"  COR-CVAE : {'disabled (--skip-cvae)' if args.skip_cvae else 'enabled'}")

    t0 = time.time()
    run(skip_cvae=args.skip_cvae)
    _banner(f"Household stream complete  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
