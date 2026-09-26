"""
Statistical disclosure control for tracked UKHLS-derived aggregate tables
in outputs_v2/ (analysis_plan_rerun.md, author decision 2026-09-26).

Rules
  * frequency counts 1-9 -> "<10" (zeros kept)
  * a rate is suppressed when its denominator n < 10, or when the numerator
    it implies (rate x n, unweighted) is 1-9
  * identifiers (hidp etc.) and row-level detail text are removed

Usage
  python scripts/suppress_small_cells.py          apply in place
  python scripts/suppress_small_cells.py --check  exit 1 if anything is left, or if
                                                 any tracked/staged outputs_v2 CSV is
                                                 not a known aggregate table

Re-run after regenerating any table, before committing it.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs_v2"
THRESHOLD = 10
SUP = "<10"

# file -> spec. counts: frequency-count columns ("*" = every numeric column
# except the listed keys). rates: {rate_col: (denominator_col, scale)}, scale
# 100 for percentages, 1 for proportions; numerator implied = rate/scale*n.
SPEC: dict[str, dict] = {
    "audit/elec_not_reported_rent_check.csv": dict(
        counts=["n", "n_waves_c_to_o", "elecpay_asked_n", "elecpay_included_in_rent_n",
                "gaspay_asked_n", "gaspay_included_in_rent_n", "duelpay_asked_n",
                "duelpay_included_in_rent_n"],
        rates={"elecpay_included_in_rent_pct_of_asked": ("elecpay_asked_n", 100),
               "gaspay_included_in_rent_pct_of_asked": ("gaspay_asked_n", 100),
               "duelpay_included_in_rent_pct_of_asked": ("duelpay_asked_n", 100),
               "pct_renting": ("n", 100), "pct_social_rent": ("n", 100),
               "pct_private_rent": ("n", 100), "pct_owner": ("n", 100),
               "pct_with_positive_reported_spend": ("n", 100)}),
    "audit/fuel_status_by_region_wave.csv": dict(counts="*", keys=["region", "wave"]),
    "audit/fuelduel_by_region_year.csv": dict(counts="*", keys=["region", "interview_year"]),
    "audit/zero_filled_components_by_region_wave.csv": dict(counts="*", keys=["region", "wave"]),
    "audit/ni_oil_lost_by_wave.csv": dict(counts="*", keys=["wave"]),
    "audit/gap_explanations.csv": dict(counts=["n"], drop_text={"detail": r"hidp="}),
    "audit/indicative_prevalence.csv": dict(
        counts=["v1_n", "a1_primary_n", "s1_lower_bound_n", "s2_plus_elec_nr_n"],
        rates={"v1_pct": ("v1_n", 100), "a1_primary_pct": ("a1_primary_n", 100),
               "s1_lower_bound_pct": ("s1_lower_bound_n", 100),
               "s2_plus_elec_nr_pct": ("s2_plus_elec_nr_n", 100)}),
    "audit/missing_by_mode_wave.csv": dict(
        counts=["observed", "missing", "n"],
        rates={"pct_missing": ("n", 100), "pct_of_wave": ("n", None)}),
    "audit/missing_vs_observed_spend.csv": dict(
        counts=["missing", "observed"],
        rates={"pct_missing_within_group": ("__missing_plus_observed", 100)}),
    "audit/sample_flow_reconciliation.csv": dict(counts=["n_change_or_total", "remaining"]),
    "audit_fuel_codes.csv": dict(counts=["n"]),
    "descriptives/prevalence_by_group.csv": dict(
        counts=["n"], rates={"pct_unweighted": ("n", 100), "pct_weighted_wave_mean": ("n", 100)}),
    "descriptives/prevalence_by_wave.csv": dict(
        counts=["n_households", "n_zero_weight", "primary_n", "s1_lower_bound_n",
                "s2_plus_elec_nr_n", "v1_n"]),
    "descriptives/prevalence_region_by_year.csv": dict(
        counts=["n"], rates={"pct_weighted": ("n", 100), "pct_unweighted": ("n", 100)}),
    "jrf/jrf_comparison.csv": dict(
        counts=["ukhls_n"],
        rates={"fuel_vuln_pct_weighted": ("ukhls_n", 100), "fuel_vuln_pct_unweighted": ("ukhls_n", 100),
               "fuel_vuln_s1_pct_weighted": ("ukhls_n", 100)}),
    "jrf/ni_oil.csv": dict(
        counts=["n_households", "oil_n", "non_oil_n"],
        rates={"oil_vuln_pct_weighted": ("oil_n", 100), "oil_vuln_pct_unweighted": ("oil_n", 100),
               "non_oil_vuln_pct_weighted": ("non_oil_n", 100),
               "non_oil_vuln_pct_unweighted": ("non_oil_n", 100),
               "oil_share_pct_unweighted": ("n_households", 100)}),
    "jrf/region_window_ci.csv": dict(counts=["n", "n_psu"],
                                     rates={"pct_weighted": ("n", 100)}),
    "resources/resource_by_region.csv": dict(counts=["n"]),
    "resources/resource_composite_by_region.csv": dict(counts=["n"]),
    "resources/cfa_fit.csv": dict(counts=["n_complete_case"]),
    "resources/cfa_sample_composition.csv": dict(
        proportions_of={"tenure_complete_case": ("resources/cfa_fit.csv", "n_complete_case")}),
    "descriptives/strain_structure.csv": dict(counts=["n"]),
    "fes_eval/fes_coverage_by_interview_year.csv": dict(
        counts=["n", "n_fes", "n_outcome", "n_outcome_and_fes"]),
    "fes_eval/fes_attached_v1_vs_v2.csv": dict(counts=["n_both"]),
    "stage3/coefficients.csv": dict(counts=["n"]),
    "stage3/region_fe.csv": dict(counts=["n"]),
    "stage3/model_summary.csv": dict(counts=["n", "n_events", "n_psu"]),
    "stage3/sample_flow.csv": dict(counts=["n"]),
    "stage3/ni_oil_sequence.csv": dict(counts=["n"]),
    "stage3/thesis_table_primary.csv": dict(counts=["n"]),
    "stage3/thesis_table_per_sd.csv": dict(counts=["n"]),
    "stage3/ni_oil_ame.csv": dict(counts=["n"]),
    "stage4/h1_coefficients.csv": dict(counts=["n", "n_psu"]),
    "stage4/h1_decision.csv": dict(counts=["n"]),
    "stage5/thesis_T5_2_comparison.csv": dict(counts=["n"]),
    "stage6/sample_flow.csv": dict(counts=["n", "wave_t_households"]),
    "stage6/posthoc_p3_metrics.csv": dict(counts=["n_train", "n_validation"]),
    "stage7/sensitivity_equivalised_income_only.csv": dict(counts=["n"]),
    "resources/composite_alpha.csv": dict(counts=["n_complete"]),
    # Built only from the suppressed tables above; count-type values verified
    # to contain no household count of 1-9 when the inventory was built.
    "results_inventory.csv": dict(counts=["n"]),
    "stage7/prepayment_by_vulnerability.csv": dict(counts=["n"]),
    "stage7/regional_change_early_late.csv": dict(counts=["n_early", "n_late"]),
    "stage7/prevalence_by_fes_tercile.csv": dict(counts=["n"]),
    "stage6/metrics.csv": dict(counts=["n_validation"]),
    "stage6/per_transition_auc.csv": dict(counts=["n"]),
    "stage6/coefficients.csv": dict(counts=["n_train"]),
    "stage6/p2b_sensitivity.csv": dict(counts=["n_train", "n_validation"]),
    "stage6/thesis_table_prediction.csv": dict(counts=["n validation"]),
    "descriptives/prevalence_by_interview_year.csv": dict(
        counts=["primary_n", "s1_lower_bound_n"],
        rates={"primary_pct_weighted": ("primary_n", 100),
               "s1_lower_bound_pct_weighted": ("s1_lower_bound_n", 100)}),
}
# Tables with no UKHLS counts or rates at risk (correlations, loadings,
# metadata, national aggregates over >=100 households) -- listed so --check
# can confirm every tracked table was considered.
EXEMPT = {"fes_eval/prophet_fallbacks.csv", "stage6/posthoc_p3_delta_auc.csv", "stage6/calibration.csv", "stage6/delta_auc.csv", "stage6/standardisation_train_stats.csv",
          "stage5/thesis_T5_1_jrf_metadata.csv", "stage5/thesis_T5_3_agreement.csv",
          "stage5/thesis_T5_4_northern_ireland.csv",  # every n in its text column is >= 545
          "stage5/thesis_secondary_suppression_log.csv",
          "stage4/h1_slopes.csv", "stage4/h1_buffering_bound.csv", "stage4/h1_logit_prob_slopes.csv", "stage3/year_fe.csv", "fes_eval/thesis_table_forecast_accuracy.csv",
          "fes_eval/appendix_table_mase.csv", "fes_eval/thesis_table_relrmse_by_year.csv",
          "fes_eval/forecast_accuracy_by_year.csv", "fes_eval/forecast_accuracy_pooled.csv",
          "fes_eval/diebold_mariano.csv", "fes_eval/uncertainty_pi.csv", "fes_eval/model_wins.csv",
          "fes_eval/fes_annual.csv",  # macro price series only, no UKHLS data
          "descriptives/strain_item_correlations.csv", "jrf/jrf_agreement.csv",
          "jrf/jrf_metadata.csv", "jrf/oil_share_by_region.csv",
          "resources/cfa_factor_correlations.csv", "resources/cfa_loadings.csv",
          "resources/composite_item_correlations.csv", "resources/resource_decision.csv"}


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def small(x: pd.Series) -> pd.Series:
    v = _num(x).abs()
    return (v >= 1) & (v < THRESHOLD)


def apply(rel: str, spec: dict, check: bool) -> list[str]:
    path = OUT / rel
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    problems = []
    keys = spec.get("keys", [])
    counts = ([c for c in df.columns if c not in keys and _num(df[c]).notna().any()]
              if spec.get("counts") == "*" else spec.get("counts", []))
    denom = {}
    if "__missing_plus_observed" in str(spec.get("rates", {})):
        denom["__missing_plus_observed"] = _num(df["missing"]) + _num(df["observed"])

    mask_rows = pd.Series(False, index=df.index)
    for rate, (den, scale) in spec.get("rates", {}).items():
        if rate not in df:
            continue
        n = denom.get(den, _num(df.get(den, pd.Series(np.nan, index=df.index))))
        bad = (n < THRESHOLD)
        if scale:
            bad |= small(_num(df[rate]) / scale * n)
        bad &= df[rate].ne("") & df[rate].ne(SUP)
        if check and bad.any():
            problems.append(f"{rel}:{rate} {int(bad.sum())} cells")
        df.loc[bad, rate] = SUP
        mask_rows |= bad
    for c in counts:
        if c not in df:
            continue
        bad = small(df[c])
        if check and bad.any():
            problems.append(f"{rel}:{c} {int(bad.sum())} cells")
        df.loc[bad, c] = SUP
    for col, pattern in spec.get("drop_text", {}).items():
        bad = df[col].str.contains(pattern, regex=True, na=False)
        if check and bad.any():
            problems.append(f"{rel}:{col} identifier text")
        df.loc[bad, col] = "[removed: row-level detail]"
    for col, (ref_file, ref_col) in spec.get("proportions_of", {}).items():
        n_total = float(pd.read_csv(OUT / ref_file)[ref_col].iloc[0])
        implied = _num(df[col]) * n_total
        bad = small(implied) | ((_num(df[col]) == 0) & df[col].ne("0") & df[col].ne("0.0"))
        bad &= df[col].ne(SUP)
        if check and bad.any():
            problems.append(f"{rel}:{col} {int(bad.sum())} cells")
        df.loc[bad, col] = SUP
    if not check:
        df.to_csv(path, index=False)
    return problems


def main() -> None:
    check = "--check" in sys.argv
    tracked = {str(p.relative_to(OUT)) for p in OUT.rglob("*.csv")
               if not re.search(r"(ukhls_cleaned|resource_scores|/fes/|forecasts|tuning|tables/rolling|/logs/)", str(p))}
    unconsidered = sorted(tracked - set(SPEC) - EXEMPT)
    problems = []
    for rel, spec in SPEC.items():
        if (OUT / rel).exists():
            problems += apply(rel, spec, check)
    if unconsidered:
        problems += [f"not in SPEC/EXEMPT: {u}" for u in unconsidered]
    # Guard: every CSV tracked or staged under outputs_v2 must be a known
    # aggregate table. Stops row-level files slipping in via `git add -f`.
    import subprocess
    listed = subprocess.run(["git", "ls-files", "--cached", "outputs_v2"], cwd=ROOT,
                            capture_output=True, text=True).stdout.split()
    known = set(SPEC) | EXEMPT | {"fes/fes_rolling_monthly.csv", "fes/fes_rolling_yearly.csv",
                                  "fes/model_selection_by_year.csv", "fes/tuned_params_by_origin.csv",
                                  "fes/forecast_performance_by_year.csv"}
    for f in listed:
        if f.endswith(".csv") and f[len("outputs_v2/"):] not in known:
            problems.append(f"tracked/staged CSV not a known aggregate table: {f}")
    if check:
        print("\n".join(problems) if problems else "OK: no unsuppressed small cells")
        sys.exit(1 if problems else 0)
    print(f"suppression applied to {len(SPEC)} tables"
          + (f"; NOT CONSIDERED: {unconsidered}" if unconsidered else ""))


if __name__ == "__main__":
    main()
