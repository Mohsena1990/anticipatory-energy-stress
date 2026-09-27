"""
Results inventory: every number the thesis will quote, with label, value,
95% CI, N, unit, source file and the commit that last changed that file.

Reads committed outputs only; fails if a source file is untracked or has
uncommitted changes. Output: outputs_v2/results_inventory.csv
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
O = ROOT / "outputs_v2"
ROWS: list[dict] = []
_COMMIT: dict[str, str] = {}


def commit_of(rel: str) -> str:
    if rel not in _COMMIT:
        tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=ROOT, capture_output=True).returncode == 0
        dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=ROOT).returncode != 0
        if not tracked or dirty:
            sys.exit(f"source not committed cleanly: {rel}")
        _COMMIT[rel] = subprocess.run(["git", "log", "-1", "--format=%h", "--", rel], cwd=ROOT,
                                      capture_output=True, text=True).stdout.strip()
    return _COMMIT[rel]


def num(x):
    v = pd.to_numeric(pd.Series([x]), errors="coerce").iloc[0]
    return None if pd.isna(v) else float(v)


def add(section, label, value, file, ci_low=None, ci_high=None, n=None, unit="", note=""):
    rel = str((O / file).relative_to(ROOT))
    v = num(value)
    ROWS.append(dict(section=section, label=label, value=v if v is not None else value,
                     ci_low=num(ci_low) if ci_low is not None else None,
                     ci_high=num(ci_high) if ci_high is not None else None,
                     n=num(n) if n is not None else None, unit=unit, source_file=rel,
                     commit=commit_of(rel), note=note))


def rd(file):
    return pd.read_csv(O / file)


def main() -> None:
    # 1. Sample and outcome.
    f = "audit/sample_flow_reconciliation.csv"
    for r in rd(f).itertuples():
        add("1 Sample", f"{r.step}: {r.description}", r.n_change_or_total, f, unit="household-waves")
    f = "audit/ni_oil_lost_by_wave.csv"
    r = rd(f).query("wave == 'all'").iloc[0]
    add("1 Sample", "NI oil households (all waves)", r.ni_oil_households, f, unit="household-waves")
    add("1 Sample", "NI oil households dropped by the v1 fuelduel rule", r.ni_oil_dropped_by_fuelduel_rule, f,
        unit="household-waves")

    # 2. Trend.
    f = "descriptives/prevalence_by_wave.csv"
    for r in rd(f).itertuples():
        per = f"wave {r.wave} ({int(r.fieldwork_first_year)}-{int(r.fieldwork_last_year)})"
        add("2 Trend (by wave, main)", f"{per}: primary, weighted", r.primary_pct_weighted, f, n=r.primary_n, unit="%")
        add("2 Trend (by wave, main)", f"{per}: S1 lower bound, weighted", r.s1_lower_bound_pct_weighted, f,
            n=r.s1_lower_bound_n, unit="%")
    f = "descriptives/prevalence_by_interview_year.csv"
    for r in rd(f).itertuples():
        add("2 Trend (by interview year, supplementary)", f"{r.interview_year}: primary, weighted",
            r.primary_pct_weighted, f, n=r.primary_n, unit="%", note="partial year" if num(r.primary_n) < 1000 else "")
        add("2 Trend (by interview year, supplementary)", f"{r.interview_year}: S1 lower bound, weighted",
            r.s1_lower_bound_pct_weighted, f, n=r.s1_lower_bound_n, unit="%")

    # 3. Pooled group prevalence.
    f = "descriptives/prevalence_by_group.csv"
    for r in rd(f).itertuples():
        add("3 Group prevalence (pooled, wave-mean weighted)", f"{r.dimension}: {r.group}",
            r.pct_weighted_wave_mean, f, n=r.n, unit="%")

    # 4. Forecasting and FES.
    f = "fes_eval/thesis_table_forecast_accuracy.csv"
    for r in rd(f).itertuples():
        add("4 Forecast accuracy", f"{r.series} {r.version}: relative RMSE vs naive", r.relRMSE_vs_naive, f,
            note=f"DM p = {r.DM_p_vs_naive:.3f}")
        add("4 Forecast accuracy", f"{r.series} {r.version}: relative RMSE vs seasonal naive", r.relRMSE_vs_snaive, f,
            note=f"DM p = {r.DM_p_vs_snaive:.3f}")
    f = "fes_eval/appendix_table_mase.csv"
    for r in rd(f).itertuples():
        add("4 Forecast accuracy (appendix)", f"{r.series} {r.version}: MASE", r.MASE, f)
    f = "fes_eval/thesis_table_relrmse_by_year.csv"
    for r in rd(f).itertuples():
        add("4 Forecast accuracy by year", f"{r.series} {r.target_year}: v2 relative RMSE vs naive",
            r.v2_core_relRMSE_vs_naive, f)
    f = "fes_eval/uncertainty_pi.csv"
    for r in rd(f).itertuples():
        add("4 Forecast uncertainty", f"{r.series} {r.version}: 95% PI coverage", r.pi95_coverage_pct, f,
            n=r.n_months, unit="% of months")
    f = "fes_eval/model_wins.csv"
    for r in rd(f).itertuples():
        add("4 Model selection", f"{r.version} {r.series}: origins won by {r.model}", r.n_origins, f, unit="origins")
    f = "fes_eval/prophet_fallbacks.csv"
    r = rd(f).iloc[0]
    add("4 Model selection", "Prophet L-BFGS -> Newton fallbacks", r.newton_fallbacks, f,
        n=r.cmdstan_optimise_calls, note=f"{r.pct_fallback}% of optimiser calls; {int(r.failed_fits)} failed fits")
    f = "fes_eval/fes_coverage_by_interview_year.csv"
    for r in rd(f).itertuples():
        add("4 FES coverage", f"{int(r.interview_year)}: households with outcome and FES", r.n_outcome_and_fes, f,
            unit="household-waves")
    f = "fes_eval/fes_attached_v1_vs_v2.csv"
    for r in rd(f).itertuples():
        add("4 FES v1 vs v2", f"{r.variable}: row-level correlation v1 vs v2", r.r_v1_v2, f, n=r.n_both)

    # 5. Driver model.
    f = "stage3/sample_flow.csv"
    for r in rd(f).query("spec in ['all', 'primary']").itertuples():
        add("5 Driver model sample", f"{r.spec}: {r.step}", r.n, f, unit="household-waves")
    f = "stage3/model_summary.csv"
    for r in rd(f).query("model in ['main', 'main_twoway_cluster']").itertuples():
        add("5 Driver model fit", f"{r.spec} ({r.model}): McFadden pseudo-R2", r.pseudo_r2_mcfadden, f, n=r.n)
    f = "stage3/thesis_table_primary.csv"
    for r in rd(f).itertuples():
        add("5 Driver model (primary)", f"{r.label}: OR", r.OR, f, r.OR_ci_low, r.OR_ci_high, r.n,
            note=f"p = {r.p:.3g}")
    f = "stage3/thesis_table_per_sd.csv"
    for r in rd(f).itertuples():
        add("5 Driver model (per SD)", f"{r.label}: OR per SD", r.OR_per_sd, f, r.OR_per_sd_ci_low,
            r.OR_per_sd_ci_high, r.n, note=r.note if isinstance(r.note, str) else "")
    f = "stage3/coefficients.csv"
    key = ["fes_delta_growth3", "fes_delta", "finnow", "financial_strain_score", "financial_strain_score_v1",
           "finnow_lag1", "financial_strain_score_lag1", "jbstat_security", "workless_household"]
    c = rd(f)
    for r in c[(c.model == "main") & c.term.isin(key) & (c.spec != "primary")].itertuples():
        add("5 Driver model (sensitivities)", f"{r.spec}: {r.label}: OR", r.OR, f, r.OR_ci_low, r.OR_ci_high, r.n,
            note=f"p = {r.p:.3g}")
    for r in c[(c.model == "main_twoway_cluster") & (c.term == "fes_delta_growth3")].itertuples():
        add("5 Driver model (sensitivities)", f"primary, two-way clustered: {r.label}: OR", r.OR, f,
            r.OR_ci_low, r.OR_ci_high, r.n)
    f = "stage3/ni_oil_ame.csv"
    for r in rd(f).query("spec == 'primary'").itertuples():
        add("5 NI-oil sequence (AME)", f"{r.model}: {r.contrast}", r.ame_pp, f, r.ci_low_pp, r.ci_high_pp, r.n, unit="pp")

    # 6. Resources.
    f = "resources/cfa_fit.csv"
    r = rd(f).iloc[0]
    for k in ["CFI", "TLI", "RMSEA", "SRMR"]:
        add("6 Resources (CFA, failed criteria)", f"CFA {k}", r[k], f, n=r.n_complete_case)
    f = "resources/composite_alpha.csv"
    for r in rd(f).itertuples():
        add("6 Resources (formative indices)", f"{r.domain}: Cronbach alpha (descriptive)", r.cronbach_alpha, f,
            n=r.n_complete)
    f = "descriptives/strain_structure.csv"
    for r in rd(f).itertuples():
        add("6 Strain items", r.metric, r.value, f, n=r.n)

    # 7. H1.
    f = "stage4/h1_coefficients.csv"
    for r in rd(f).itertuples():
        add("7 H1", f"{r.model}: {r.term}", r.coef, f, r.ci_low, r.ci_high, r.n, note=f"p = {r.p:.3g}")
    f = "stage4/h1_decision.csv"
    for r in rd(f).itertuples():
        add("7 H1", f"{r.model}: verdict", r.verdict, f, n=r.n,
            note="governs H1" if r.governs_H1 else "sensitivity")
    f = "stage4/h1_slopes.csv"
    for r in rd(f).itertuples():
        add("7 H1 slopes", f"{r.model}: Delta slope at R p{r.R_percentile}", r.delta_slope, f, r.ci_low, r.ci_high,
            unit=r.scale)
    f = "stage4/h1_buffering_bound.csv"
    for r in rd(f).itertuples():
        add("7 H1 buffering bound", f"{r.interaction}: slope difference p90-p10 per SD Delta",
            r.slope_diff_per_sd_delta_pp_income, f, unit="pp of income")
    r = rd(f).iloc[0]
    add("7 H1 buffering bound", "Delta slope at R p10 per SD Delta", r.slope_p10_per_sd_delta_pp_income, f,
        unit="pp of income")
    f = "stage4/h1_logit_prob_slopes.csv"
    for r in rd(f).itertuples():
        add("7 H1 (footnote)", f"logit: Delta effect at R p{r.R_percentile}", r.pp_per_sd_delta, f, r.ci_low,
            r.ci_high, unit="pp per SD of Delta")

    # 8. JRF and NI.
    f = "stage5/thesis_T5_2_comparison.csv"
    for r in rd(f).itertuples():
        add("8 JRF comparison", f"{r.dimension} ({r.window}, {r.ukhls_window}): {r.category}", r.pct, f,
            r.ci_low, r.ci_high, r.n, unit="%",
            note=f"JRF {r.jrf_pct}% (rank {r.jrf_rank}); UKHLS rank {r['rank'] if False else r.rank} "
                 f"[{r.rank_ci_low}-{r.rank_ci_high}]")
    f = "stage5/thesis_T5_3_agreement.csv"
    for r in rd(f).itertuples():
        v = r.spearman_rho if pd.notna(r.spearman_rho) else ("same direction" if r.two_group_same_direction else "")
        add("8 JRF agreement", f"{r.dimension} ({r.window}, {r.subset}): Spearman rho", v, f,
            note=f"{r.n_categories} categories")
    f = "stage5/thesis_T5_4_northern_ireland.csv"
    for r in rd(f).itertuples():
        add("8 Northern Ireland", r.item, r.value, f, r.ci_low if pd.notna(r.ci_low) else None,
            r.ci_high if pd.notna(r.ci_high) else None, note=r.detail)
    f = "jrf/oil_share_by_region.csv"
    for r in rd(f).itertuples():
        add("8 Oil share by region (pooled, unweighted)", r.region, r.oil_share_pct_unweighted, f, unit="%")

    # 9. Prediction.
    f = "stage6/sample_flow.csv"
    for r in rd(f).itertuples():
        add("9 Prediction sample", r.step.strip(), r.n, f)
    f = "stage6/metrics.csv"
    for r in rd(f).itertuples():
        add("9 Prediction", f"{r.model}: {r.metric}", r.estimate, f, r.ci_low, r.ci_high, r.n_validation)
    f = "stage6/delta_auc.csv"
    for r in rd(f).itertuples():
        add("9 Prediction", f"delta AUC {r.comparison}", r.delta_auc, f, r.ci_low, r.ci_high,
            note="report as 'no improvement' without significance language" if r.comparison == "P2 - P1" else "")
    f = "stage6/per_transition_auc.csv"
    for r in rd(f).itertuples():
        add("9 Prediction", f"{r.model} {r.transition}: AUC", r.auc, f, r.ci_low, r.ci_high, r.n)
    f = "stage6/calibration.csv"
    for r in rd(f).itertuples():
        add("9 Prediction calibration", f"{r.model}: calibration slope", r.slope, f, r.slope_ci_low, r.slope_ci_high)
        add("9 Prediction calibration", f"{r.model}: calibration-in-the-large", r.intercept, f,
            r.intercept_ci_low, r.intercept_ci_high)
    f = "stage6/p2b_sensitivity.csv"
    for r in rd(f).itertuples():
        add("9 Prediction", f"{r.model}: AUC (P2b sample)", r.auc, f, n=r.n_validation)
    f = "stage6/posthoc_p3_metrics.csv"
    for r in rd(f).query("label == 'POST-HOC / EXPLORATORY'").itertuples():
        add("9 Prediction (POST-HOC)", f"{r.model}: {r.metric}", r.estimate, f, r.ci_low, r.ci_high, r.n_validation,
            note="POST-HOC / EXPLORATORY")
    f = "stage6/posthoc_p3_delta_auc.csv"
    for r in rd(f).itertuples():
        add("9 Prediction (POST-HOC)", f"delta AUC {r.comparison}", r.delta_auc, f, r.ci_low, r.ci_high,
            note="POST-HOC / EXPLORATORY")

    # 10. Stage 7 sensitivity and descriptive refreshes.
    f = "stage7/sensitivity_equivalised_income_only.csv"
    for r in rd(f).itertuples():
        add("10 Sensitivity to equivalising income only", f"household size {r.household_size}: % flag changes",
            r.pct_flip, f, n=r.n, unit="%")
    f = "stage7/prepayment_by_vulnerability.csv"
    for r in rd(f).itertuples():
        add("10 Descriptive", f"prepayment meter, {r.status}", r.pct_prepayment_weighted, f, n=r.n, unit="%")
    f = "stage7/regional_change_early_late.csv"
    for r in rd(f).itertuples():
        add("10 Descriptive", f"{r.region}: change waves a-e to k-o", r.change_pp, f, unit="pp",
            note=f"{r.pct_early_waves_a_e:.2f}% -> {r.pct_late_waves_k_o:.2f}%")
    f = "stage7/prevalence_by_fes_tercile.csv"
    for r in rd(f).itertuples():
        add("10 Descriptive", f"{r.fes} tercile {r.tercile}", r.pct_vulnerable_weighted, f, n=r.n, unit="%",
            note=f"FES {r.fes_range}")

    inv = pd.DataFrame(ROWS)
    inv.insert(0, "id", [f"R{i:04d}" for i in range(1, len(inv) + 1)])
    inv.to_csv(O / "results_inventory.csv", index=False)
    print(len(inv), "entries;", inv.section.nunique(), "sections;", inv.commit.nunique(), "source commits")
    print(inv.groupby("section").size().to_string())


if __name__ == "__main__":
    main()
