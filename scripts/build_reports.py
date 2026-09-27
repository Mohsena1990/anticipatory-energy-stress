"""
Build README.md and reports/01-06 from reports/templates/*.md.

Every table in the reports is rendered here from the committed CSVs, so the
reports always show the current outputs; nothing is typed in by hand.
Templates use:
  {{TABLE:key}}   a markdown table rendered from a CSV (keys defined below)
  {{P}}           relative path prefix to the repository root ('' or '../')

Needs the thesis bundle (python scripts/build_thesis_assets.py) because the
thesis tables are read from outputs_v2/thesis_assets_v2/tables/.

  python scripts/build_reports.py                 # all documents
  python scripts/build_reports.py README.md       # one template
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
O = ROOT / "outputs_v2"
B = O / "thesis_assets_v2" / "tables"
TEMPLATES = ROOT / "reports" / "templates"


# ---------------------------------------------------------------- formatting
def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def fi(x):
    """integer with thousands separator; passes '<10' etc. through."""
    v = num(x)
    if v is None or v != v:
        return "" if (x is None or pd.isna(x)) else str(x)
    return f"{int(round(v)):,}"


def ff(x, d=2):
    v = num(x)
    if v is not None and v != v:
        return ""
    if v is None:
        return "" if (x is None or (isinstance(x, float) and pd.isna(x))) else str(x)
    return f"{v:,.{d}f}"


def fp(x):
    v = num(x)
    if v is not None and v != v:
        return ""
    if v is None:
        return "" if pd.isna(x) else str(x)
    if v < 0.001:
        return "<0.001"
    return f"{v:.3f}"


def ci(v, lo, hi, d=2):
    if num(v) is None:
        return ""
    if num(lo) is None or num(hi) is None:
        return ff(v, d)
    return f"{ff(v, d)} [{ff(lo, d)}, {ff(hi, d)}]"


def md(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for v in r.values:
            s = "" if (v is None or (isinstance(v, float) and pd.isna(v))) else str(v)
            cells.append(s.replace("|", "\\|").replace("\n", " "))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def rd(p):
    return pd.read_csv(p, dtype=str, keep_default_na=False, na_values=[""])


T: dict[str, str] = {}

# ---------------------------------------------------------------- Chapter 3
d = rd(B / "T3-2_wave_obs_analytical_n.csv")
T["T3-2"] = md(pd.DataFrame({
    "Wave": d.wave, "Fieldwork": d.fieldwork_period.fillna(""),
    "Household-waves": d.household_waves.map(fi),
    "Analytical n, primary": d.analytical_n_primary.map(fi),
    "Analytical n, S1 lower bound": d.analytical_n_s1.map(fi),
    "Analytical n, v1 rule": d.analytical_n_v1.map(fi)}))

d = rd(B / "T3-3_sample_flow.csv")
T["T3-3"] = md(pd.DataFrame({
    "Step": d.step, "Description": d.description.str.strip(),
    "Change / total": d.n_change_or_total.map(fi), "Remaining": d.remaining.map(fi)}))

T["T3-4"] = md(rd(B / "T3-4_fuel_code_routing.csv").rename(columns={
    "variable": "Variable", "asked if": "Asked if", "content": "Content",
    "v2 treatment": "v2 treatment"}))

T["T3-5"] = md(rd(B / "T3-5_measures.csv").fillna("").rename(columns={
    "construct": "Construct", "items": "Items", "coding": "Coding",
    "construction": "Construction", "Cronbach alpha (descriptive)": "Cronbach α (descriptive)"}))

d = rd(B / "T3-6_forecasting_variables.csv")
T["T3-6"] = md(pd.DataFrame({
    "Series": d.variable, "n": d.n, "Missing": d.missing, "Mean": d["mean"].map(lambda x: ff(x, 2)),
    "SD": d.sd.map(lambda x: ff(x, 2)), "Median": d["median"].map(lambda x: ff(x, 2)),
    "Skew": d["skew"].map(lambda x: ff(x, 2)), "Excess kurtosis": d.excess_kurtosis.map(lambda x: ff(x, 2))}))

d = rd(B / "T3-7_jrf_metadata.csv").fillna("")
T["T3-7"] = md(pd.DataFrame({
    "Dimension": d.dimension, "Window": d.window, "JRF population": d.jrf_population,
    "JRF measure": d.jrf_measure, "JRF period": d.jrf_period, "JRF source": d.jrf_source,
    "UKHLS window": d.ukhls_window, "UKHLS unit": d.ukhls_unit,
    "UKHLS definition": d.ukhls_definition, "Waves": d.ukhls_waves,
    "Categories": d.n_categories, "Notes": d.notes}))

# ---------------------------------------------------------------- Chapter 4
d = rd(B / "T4-1_forecast_accuracy_pi.csv")
T["T4-1"] = md(pd.DataFrame({
    "Series": d.series, "Version": d.version, "RMSE": d.RMSE.map(lambda x: ff(x, 2)),
    "Rel. RMSE vs naive": d.relRMSE_vs_naive.map(lambda x: ff(x, 3)),
    "DM p vs naive": d.DM_p_vs_naive.map(fp),
    "Rel. RMSE vs seasonal naive": d.relRMSE_vs_snaive.map(lambda x: ff(x, 3)),
    "DM p vs seasonal naive": d.DM_p_vs_snaive.map(fp),
    "95% PI coverage (%)": d.pi95_coverage_pct.map(lambda x: ff(x, 1))}))

d = rd(B / "T4-2_social_regional_rates.csv")
T["T4-2"] = md(pd.DataFrame({
    "Dimension": d.dimension, "Category": d.category, "n": d.n.map(fi),
    "Weighted % [95% CI]": [ci(a, b, c, 1) for a, b, c in zip(d.pct_weighted, d.ci_low, d.ci_high)]}))

d = rd(B / "T4-3_driver_model.csv")
T["T4-3"] = md(pd.DataFrame({
    "Predictor": d.label, "Term": "`" + d.term + "`",
    "OR [95% CI]": [ci(a, b, c, 3) for a, b, c in zip(d.OR, d.OR_ci_low, d.OR_ci_high)],
    "p": d.p.map(fp), "OR per SD": d.OR_per_sd.map(lambda x: ff(x, 3)),
    "SD in sample": d.sd_in_sample.map(lambda x: ff(x, 3))}))

d = rd(B / "T4-4_per_sd.csv").fillna("")
T["T4-4"] = md(pd.DataFrame({
    "Predictor": d.label, "SD": d.sd_in_sample.map(lambda x: ff(x, 3)),
    "OR per SD [95% CI]": [ci(a, b, c, 3) for a, b, c in zip(d.OR_per_sd, d.OR_per_sd_ci_low, d.OR_per_sd_ci_high)],
    "log OR per SD": d.log_OR_per_sd.map(lambda x: ff(x, 3)), "p": d.p.map(fp), "Note": d.note}))

h = rd(B / "T4-5_h1.csv")
v = h[h.part == "verdict"]
T["T4-5a"] = md(pd.DataFrame({
    "Model": v.model, "R × Delta interaction": v.interaction.map(lambda x: f"{float(x):.7f}"),
    "p": v.p.map(fp), "Verdict": v.verdict, "Governs H1": v.governs_H1, "n": v.n.map(fi)}))
c = h[h.part == "coefficients"]
T["T4-5b"] = md(pd.DataFrame({
    "Model": c.model, "Estimator": c.estimator, "Outcome": c.outcome, "Term": "`" + c.term + "`",
    "Coefficient [95% CI]": [f"{float(a):.6f} [{float(b):.6f}, {float(e):.6f}]" for a, b, e in zip(c.coef, c.ci_low, c.ci_high)],
    "SE": c.se.map(lambda x: f"{float(x):.6f}"), "p": c.p.map(fp),
    "R² / pseudo-R²": c.r2_or_pseudo_r2.map(lambda x: ff(x, 4)), "PSUs": c.n_psu.map(fi)}))
s = h[h.part == "Delta slopes"]
T["T4-5c"] = md(pd.DataFrame({
    "Model": s.model, "R percentile": s.R_percentile.map(lambda x: f"p{int(float(x))}"),
    "R value": s.R_value.map(lambda x: ff(x, 3)),
    "Delta slope [95% CI]": [f"{float(a):.6f} [{float(b):.6f}, {float(e):.6f}]" for a, b, e in zip(s.delta_slope, s.ci_low, s.ci_high)],
    "Scale": s.scale}))
bb = h[h.part == "buffering bound"]
T["T4-5d"] = md(pd.DataFrame({
    "Interaction value": bb.interaction, "b (R × Delta)": bb.b_RxD.map(lambda x: f"{float(x):.7f}"),
    "R p10 → p90": [f"{float(a):.2f} → {float(b):.2f}" for a, b in zip(bb.R_p10, bb.R_p90)],
    "SD of Delta": bb.sd_delta.map(lambda x: ff(x, 3)),
    "Slope change p10→p90 per SD Delta (ratio units)": bb.slope_diff_per_sd_delta_ratio.map(lambda x: f"{float(x):.6f}"),
    "Same, pp of income": bb.slope_diff_per_sd_delta_pp_income.map(lambda x: ff(x, 3)),
    "Slope at p10 per SD Delta (pp)": bb.slope_p10_per_sd_delta_pp_income.map(lambda x: ff(x, 3)),
    "Change as % of p10 slope (+ = flatter, i.e. buffering)": [f"{100*float(a)/abs(float(b)):+.0f}%" for a, b in zip(bb.slope_diff_per_sd_delta_pp_income, bb.slope_p10_per_sd_delta_pp_income)]}))
lp = h[h.part.str.startswith("logit probability")]
T["T4-5e"] = md(pd.DataFrame({
    "R percentile": lp.R_percentile.map(lambda x: f"p{int(float(x))}"),
    "R value": lp.R_value.map(lambda x: ff(x, 3)),
    "Delta effect, pp per SD [95% CI]": [ci(a, b, e, 2) for a, b, e in zip(lp.pp_per_sd_delta, lp.ci_low, lp.ci_high)]}))
_sl = rd(O / "stage4/h1_slopes.csv")
assert abs(float(_sl.ci_low.iloc[0]) - float(s.ci_low.iloc[0])) < 1e-12, "slope CI column check"

d = rd(B / "T4-6_jrf_comparison_agreement.csv")
cmp_ = d[d.part == "comparison"]
T["T4-6a"] = md(pd.DataFrame({
    "Dimension": cmp_.dimension, "Window": cmp_.window + " (" + cmp_.ukhls_window + ")",
    "Category": cmp_.category,
    "JRF % (rank)": [f"{ff(a,0)} ({ff(b,0)})" for a, b in zip(cmp_.jrf_pct, cmp_.jrf_rank)],
    "n": cmp_.n.map(fi),
    "Fuel-vulnerable % [95% CI]": [ci(a, b, e, 1) for a, b, e in zip(cmp_.pct, cmp_.ci_low, cmp_.ci_high)],
    "Rank [95% CI]": [f"{ff(a,0)} [{ff(b,0)}–{ff(e,0)}]" for a, b, e in zip(cmp_["rank"], cmp_.rank_ci_low, cmp_.rank_ci_high)],
    "P(rank 1)": cmp_.p_rank1.map(lambda x: ff(x, 3))}))
ag = d[d.part == "agreement"].fillna("")
T["T4-6b"] = md(pd.DataFrame({
    "Dimension": ag.dimension, "Window": ag.window, "Subset": ag.subset,
    "Categories": ag.n_categories.map(fi),
    "Spearman ρ": ag.spearman_rho.map(lambda x: ff(x, 2)), "Pearson r": ag.pearson_r.map(lambda x: ff(x, 2)),
    "Two groups, same direction": ag.two_group_same_direction}))

d = rd(B / "T4-7_ni_oil_ame.csv")
T["T4-7"] = md(pd.DataFrame({
    "Model": "`" + d.model + "`", "n": d.n.map(fi), "Contrast": d.contrast,
    "AME, pp [95% CI]": [ci(a, b, e, 2) for a, b, e in zip(d.ame_pp, d.ci_low_pp, d.ci_high_pp)],
    "SE (pp)": d.se_pp.map(lambda x: ff(x, 2))}))

T["T4-8"] = md(rd(B / "T4-8_prediction_p0_p3.csv").fillna(""))

d = rd(B / "T4-9_robustness_summary.csv")
def _val(r):
    if num(r.value) is None:
        return r.value
    dd = 3 if r.estimate.endswith("OR") or r.estimate.startswith("Prediction") else 2
    return ci(r.value, r.ci_low, r.ci_high, dd)
T["T4-9"] = md(pd.DataFrame({
    "Estimate": d.estimate, "Specification": d.specification,
    "Value [95% CI]": [_val(r) for r in d.itertuples()], "n": d.n.map(fi)}))

T["T4-10"] = md(rd(B / "T4-10_hypothesis_verdicts.csv").rename(columns={
    "hypothesis": "Hypothesis", "verdict": "Verdict", "status": "Status", "key statistics": "Key statistics"}))

# ---------------------------------------------------------------- Appendix
d = rd(B / "TA-1_mase.csv")
T["TA-1"] = md(pd.DataFrame({"Series": d.series, "Version": d.version,
                             "MAE": d.MAE.map(lambda x: ff(x, 2)), "MASE": d.MASE.map(lambda x: ff(x, 3))}))

d = rd(B / "TA-2_per_transition_auc.csv")
T["TA-2"] = md(pd.DataFrame({"Model": d.model, "Transition": d.transition.str.replace("->", "→"), "n": d.n.map(fi),
                             "Prevalence (%)": d.prevalence.map(lambda x: ff(100 * float(x), 1)),
                             "AUC [95% CI]": [ci(a, b, e, 3) for a, b, e in zip(d.auc, d.ci_low, d.ci_high)]}))

d = rd(B / "TA-3_driver_sensitivities.csv")
m = d[d.model == "main"].copy()
m["cell"] = [ff(o, 3) for o in m.OR]
order = list(dict.fromkeys(m.label))
specs = list(dict.fromkeys(m.spec))
pv = m.pivot_table(index="label", columns="spec", values="cell", aggfunc="first").reindex(order)[specs]
ns = m.groupby("spec").n.first().reindex(specs).map(fi)
pv.loc["n"] = ns
pv = pv.reset_index().rename(columns={"label": "Predictor (OR, main model)"})
T["TA-3"] = md(pv.fillna("—"))
tw = d[d.model == "main_twoway_cluster"]
T["TA-3-twoway"] = md(pd.DataFrame({
    "Predictor": tw.label,
    "OR [95% CI], two-way clustered (PSU × interview year-month)": [ci(a, b, e, 3) for a, b, e in zip(tw.OR, tw.OR_ci_low, tw.OR_ci_high)],
    "p": tw.p.map(fp)}))
nio = d[(d.spec == "primary") & d.model.str.startswith("ni_") & d.term.str.contains("oil|Northern|rural", regex=True)]
T["TA-3-nioil"] = md(pd.DataFrame({
    "Model": "`" + nio.model + "`", "Term": nio.label.fillna(nio.term),
    "OR [95% CI]": [ci(a, b, e, 3) for a, b, e in zip(nio.OR, nio.OR_ci_low, nio.OR_ci_high)],
    "p": nio.p.map(fp), "n": nio.n.map(fi)}))

d = rd(B / "TA-4_driver_model_fit.csv")
T["TA-4"] = md(pd.DataFrame({
    "Specification": d.spec, "Model": "`" + d.model + "`", "n": d.n.map(fi), "Events": d.n_events.map(fi),
    "PSUs": d.n_psu.map(fi), "Interview years": d.interview_years,
    "McFadden R²": d.pseudo_r2_mcfadden.map(lambda x: ff(x, 4)), "AIC": d.aic.map(lambda x: ff(x, 0)),
    "Clustering": d.cluster, "Converged": d.converged}))

d = rd(B / "TA-5_relrmse_by_year.csv")
T["TA-5"] = md(pd.DataFrame({
    "Series": d.series, "Target year": d.target_year,
    "v1 core vs naive": d.v1_core_relRMSE_vs_naive.map(lambda x: ff(x, 3)),
    "v1 core vs s-naive": d.v1_core_relRMSE_vs_snaive.map(lambda x: ff(x, 3)),
    "v1 macro vs naive": d.v1_macro_relRMSE_vs_naive.map(lambda x: ff(x, 3)),
    "v1 macro vs s-naive": d.v1_macro_relRMSE_vs_snaive.map(lambda x: ff(x, 3)),
    "v2 core vs naive": d.v2_core_relRMSE_vs_naive.map(lambda x: ff(x, 3)),
    "v2 core vs s-naive": d.v2_core_relRMSE_vs_snaive.map(lambda x: ff(x, 3))}))

d = rd(B / "TA-6_diebold_mariano.csv")
T["TA-6"] = md(pd.DataFrame({
    "Version": d.version, "Series": d.series, "Benchmark": d.benchmark.replace({"snaive": "seasonal naive"}),
    "Months": d.n_months, "DM statistic": d.DM_stat.map(lambda x: ff(x, 3)), "p": d.p_value.map(fp),
    "Verdict": d.verdict}))

d = rd(B / "TA-7_calibration.csv")
T["TA-7"] = md(pd.DataFrame({
    "Model": d.model,
    "Calibration slope [95% CI]": [ci(a, b, e, 3) for a, b, e in zip(d.slope, d.slope_ci_low, d.slope_ci_high)],
    "Calibration-in-the-large [95% CI]": [ci(a, b, e, 3) for a, b, e in zip(d.intercept, d.intercept_ci_low, d.intercept_ci_high)]}))

d = rd(B / "TA-8_prediction_sample_flow.csv").fillna("")
T["TA-8"] = md(pd.DataFrame({"Step": d.step.str.replace("->", "→"),
                             "n": [x if x == "10.54" else fi(x) for x in d.n],
                             "Wave-t households": d.wave_t_households.map(fi)}))

d = rd(B / "TA-9_cfa_fit.csv")
T["TA-9"] = md(pd.DataFrame({
    "n (complete case)": d.n_complete_case.map(fi), "χ²": d.chi2.map(lambda x: ff(x, 1)), "df": d.dof.map(fi),
    "CFI": d.CFI.map(lambda x: ff(x, 3)), "TLI": d.TLI.map(lambda x: ff(x, 3)),
    "RMSEA": d.RMSEA.map(lambda x: ff(x, 3)), "SRMR": d.SRMR.map(lambda x: ff(x, 3)),
    "Pre-registered criteria": ["CFI ≥ 0.90, RMSEA ≤ 0.08, SRMR ≤ 0.08, no Heywood case, all std. loadings ≥ 0.30 → **failed**"]}))

d = rd(B / "TA-10_cfa_loadings.csv")
T["TA-10"] = md(pd.DataFrame({
    "Factor": d.factor, "Item": "`" + d.item + "`", "Label": d.label,
    "Estimate": d.Estimate.map(lambda x: ff(x, 3)), "Std. estimate": d["Est. Std"].map(lambda x: ff(x, 3)),
    "SE": d["Std. Err"].map(lambda x: ff(x, 4) if num(x) is not None else "fixed (marker)"),
    "p": d["p-value"].map(lambda x: fp(x) if num(x) is not None else "—")}))

d = rd(B / "TA-11_equivalised_income.csv")
T["TA-11"] = md(pd.DataFrame({
    "Household size": d.household_size, "n": d.n.map(fi),
    "Flagged, primary (%)": d.pct_flagged_primary_weighted.map(lambda x: ff(x, 1)),
    "Flagged, income equivalised (%)": d.pct_flagged_equivalised_income_weighted.map(lambda x: ff(x, 1)),
    "Flip (%)": d.pct_flip.map(lambda x: ff(x, 1)), "In → out (%)": d.pct_flip_in_to_out.map(lambda x: ff(x, 1)),
    "Out → in (%)": d.pct_flip_out_to_in.map(lambda x: ff(x, 1))}))

d = rd(B / "TA-12_region_window_ci.csv")
T["TA-12"] = md(pd.DataFrame({
    "Window": d.window + " (" + d.ukhls_window + ")", "Outcome": d.outcome, "Region": d.region,
    "n": d.n.map(fi), "PSUs": d.n_psu.map(fi),
    "Weighted % [95% CI]": [ci(a, b, e, 1) for a, b, e in zip(d.pct_weighted, d.ci95_low, d.ci95_high)],
    "Rank [95% CI]": [f"{a} [{b}–{e}]" for a, b, e in zip(d["rank"], d.rank_ci95_low, d.rank_ci95_high)],
    "P(rank 1)": d.p_rank1.map(lambda x: ff(x, 3))}))

d = rd(B / "TA-13_interview_timing.csv")
d["households"] = d.households.map(fi)
pt = d.pivot_table(index="wave", columns="interview_year", values="households", aggfunc="first").fillna("—")
T["TA-13"] = md(pt.reset_index().rename(columns={"wave": "Wave"}))

d = rd(B / "TA-14_regional_counts.csv")
T["TA-14"] = md(pd.DataFrame({"Region": d.region, "Household-waves": d.household_waves.map(fi)}))

# ---------------------------------------------------------------- supporting (tracked, non-bundle)
d = rd(O / "stage5/thesis_T5_4_northern_ireland.csv").fillna("")
T["T5-4"] = md(pd.DataFrame({"Item": d["item"], "Value [95% CI]": [ci(a, b, e, 1) for a, b, e in zip(d.value, d.ci_low, d.ci_high)],
                             "Detail": d.detail}))

d = rd(O / "descriptives/prevalence_by_wave.csv")
T["prev_wave"] = md(pd.DataFrame({
    "Wave": d.wave, "Fieldwork": d.fieldwork_first_year + "–" + d.fieldwork_last_year,
    "Primary n": d.primary_n.map(fi), "Primary % (weighted)": d.primary_pct_weighted.map(lambda x: ff(x, 1)),
    "S1 lower bound % (weighted)": d.s1_lower_bound_pct_weighted.map(lambda x: ff(x, 1)),
    "S2 % (weighted)": d.s2_plus_elec_nr_pct_weighted.map(lambda x: ff(x, 1)),
    "v1 rule % (weighted)": d.v1_pct_weighted.map(lambda x: ff(x, 1)),
    "Primary % (unweighted)": d.primary_pct_unweighted.map(lambda x: ff(x, 1))}))

d = rd(O / "descriptives/prevalence_by_interview_year.csv")
T["prev_year"] = md(pd.DataFrame({
    "Interview year": d.interview_year, "Waves": d.waves, "Primary n": d.primary_n.map(fi),
    "Primary % (weighted)": d.primary_pct_weighted.map(lambda x: ff(x, 1)),
    "S1 lower bound % (weighted)": d.s1_lower_bound_pct_weighted.map(lambda x: ff(x, 1))}))

d = rd(O / "stage7/prepayment_by_vulnerability.csv")
T["prepay"] = md(pd.DataFrame({"Status": d.status, "n": d.n.map(fi),
                               "Prepayment meter, weighted (%)": d.pct_prepayment_weighted.map(lambda x: ff(x, 1)),
                               "Prepayment meter, unweighted (%)": d.pct_prepayment_unweighted.map(lambda x: ff(x, 1))}))

d = rd(O / "stage7/prevalence_by_fes_tercile.csv")
T["fes_tercile"] = md(pd.DataFrame({"FES term": "`" + d.fes + "`", "Tercile": d.tercile, "n": d.n.map(fi),
                                    "FES range": d.fes_range, "Fuel-vulnerable, weighted (%)": d.pct_vulnerable_weighted.map(lambda x: ff(x, 1))}))

d = rd(O / "stage7/regional_change_early_late.csv")
T["reg_change"] = md(pd.DataFrame({"Region": d.region, "n, waves a–e": d.n_early.map(fi), "n, waves k–o": d.n_late.map(fi),
                                   "%, waves a–e": d.pct_early_waves_a_e.map(lambda x: ff(x, 1)),
                                   "%, waves k–o": d.pct_late_waves_k_o.map(lambda x: ff(x, 1)),
                                   "Change (pp)": d.change_pp.map(lambda x: ff(x, 1))}))

d = rd(O / "resources/resource_by_region.csv")
T["res_region"] = md(pd.DataFrame({"Region": d.region, "n": d.n.map(fi),
                                   "R (primary)": d.R_primary_weighted_mean.map(lambda x: ff(x, 2)),
                                   "OBJECT": d.OBJECT_weighted_mean.map(lambda x: ff(x, 2)),
                                   "CONDITION": d.CONDITION_weighted_mean.map(lambda x: ff(x, 2)),
                                   "PERSONAL": d.PERSONAL_weighted_mean.map(lambda x: ff(x, 2)),
                                   "ENERGY": d.ENERGY_weighted_mean.map(lambda x: ff(x, 2))}))

d = rd(O / "resources/composite_alpha.csv")
T["alpha"] = md(pd.DataFrame({"Domain": d.domain, "Items": d.n_items, "n complete": d.n_complete.map(fi),
                              "Cronbach α": d.cronbach_alpha.map(lambda x: ff(x, 2))}))

d = rd(O / "resources/cfa_factor_correlations.csv")
T["cfa_corr"] = md(pd.DataFrame({"Factor": d.lval, "Factor ": d.rval, "Correlation": d.factor_correlation.map(lambda x: ff(x, 3))}))

d = rd(O / "resources/composite_item_correlations.csv")
d = d.rename(columns={d.columns[0]: ""})
for col in d.columns[1:]:
    d[col] = d[col].map(lambda x: ff(x, 2))
T["comp_corr"] = md(d)

d = rd(O / "resources/resource_decision.csv")
T["res_decision"] = md(d.rename(columns={"decision": "Decision", "cfa_passes": "CFA passes", "failures": "Failures"}))

d = rd(O / "descriptives/strain_structure.csv")
T["strain"] = md(pd.DataFrame({"Metric": d.metric, "Value": d.value.map(lambda x: ff(x, 3)), "n": d.n.map(fi)}))

d = rd(O / "descriptives/strain_item_correlations.csv")
d.columns = ["Type", "Item"] + list(d.columns[2:])
T["strain_corr"] = md(d)

d = rd(O / "fes_eval/forecast_accuracy_pooled.csv")
T["fc_pooled"] = md(pd.DataFrame({
    "Version": d.version, "Series": d.series, "RMSE": d.RMSE.map(lambda x: ff(x, 2)), "MAE": d.MAE.map(lambda x: ff(x, 2)),
    "Sign agreement (%)": d.sign_agreement_pct.map(lambda x: ff(x, 1)), "Pearson r": d.pearson_r.map(lambda x: ff(x, 3)),
    "Rel. RMSE vs naive": d.relRMSE_vs_naive.map(lambda x: ff(x, 3)), "Rel. RMSE vs s-naive": d.relRMSE_vs_snaive.map(lambda x: ff(x, 3)),
    "MASE": d.MASE.map(lambda x: ff(x, 3)), "Years beating naive (%)": d.pct_years_beating_naive.map(lambda x: ff(x, 1))}))

d = rd(O / "fes_eval/uncertainty_pi.csv")
T["pi"] = md(pd.DataFrame({"Version": d.version, "Series": d.series, "Months": d.n_months,
                           "95% PI coverage (%)": d.pi95_coverage_pct.map(lambda x: ff(x, 1)),
                           "Mean PI width": d.mean_pi_width.map(lambda x: ff(x, 2)),
                           "Median PI width": d.median_pi_width.map(lambda x: ff(x, 2))}))

d = rd(O / "fes_eval/model_wins.csv")
pw = d.pivot_table(index=["version", "series"], columns="model", values="n_origins", aggfunc="first").fillna("0").reset_index()
T["wins"] = md(pw.rename(columns={"version": "Version", "series": "Series"}))

d = rd(O / "fes_eval/fes_attached_v1_vs_v2.csv")
T["fes_v1v2"] = md(pd.DataFrame({"Variable": "`" + d.variable + "`", "n (both)": d.n_both.map(fi),
                                 "r (v1, v2)": d.r_v1_v2.map(lambda x: ff(x, 3)),
                                 "Mean v1": d.mean_v1.map(lambda x: ff(x, 3)), "Mean v2": d.mean_v2.map(lambda x: ff(x, 3)),
                                 "SD v1": d.sd_v1.map(lambda x: ff(x, 3)), "SD v2": d.sd_v2.map(lambda x: ff(x, 3))}))

d = rd(O / "fes_eval/fes_coverage_by_interview_year.csv")
T["fes_cov"] = md(pd.DataFrame({"Interview year": d.interview_year.map(lambda x: str(int(float(x)))),
                                "Households": d.n.map(fi), "With FES": d.n_fes.map(fi),
                                "With outcome and FES": d.n_outcome_and_fes.map(fi),
                                "Mean magnitude": d.fes_magnitude_mean.map(lambda x: ff(x, 2)),
                                "Mean current": d.fes_current_mean.map(lambda x: ff(x, 2)),
                                "Mean Delta": d.fes_delta_mean.map(lambda x: ff(x, 2))}))

d = rd(O / "fes_eval/prophet_fallbacks.csv")
T["prophet"] = md(d.drop(columns=["source"]))

d = rd(O / "fes/fes_rolling_yearly.csv")
T["fes_yearly"] = md(pd.DataFrame({"Origin (Dec)": d.as_of_year, "Target year": d.target_year,
                                   "FES core (4-term, annual mean)": d.fes_core.map(lambda x: ff(x, 2)),
                                   "FES weighted": d.fes_weighted.map(lambda x: ff(x, 2)),
                                   "FES realised": d.fes_actual.map(lambda x: ff(x, 2))}))

d = rd(O / "fes/model_selection_by_year.csv")
ps = d.pivot_table(index="target_year", columns="series", values="model", aggfunc="first").reset_index()
T["model_sel"] = md(ps.rename(columns={"target_year": "Target year"}))

d = rd(O / "stage6/delta_auc.csv")
T["delta_auc"] = md(pd.DataFrame({"Comparison": d.comparison,
                                  "ΔAUC [95% CI]": [ci(a, b, e, 4) for a, b, e in zip(d.delta_auc, d.ci_low, d.ci_high)]}))
d = rd(O / "stage6/posthoc_p3_delta_auc.csv")
T["p3_delta"] = md(pd.DataFrame({"Comparison": d.comparison,
                                 "ΔAUC [95% CI]": [ci(a, b, e, 4) for a, b, e in zip(d.delta_auc, d.ci_low, d.ci_high)],
                                 "Label": d.label}))
d = rd(O / "stage6/coefficients.csv")
T["s6_coef"] = md(pd.DataFrame({"Model": d.model, "Term": "`" + d.term + "`",
                                "OR [95% CI]": [ci(a, b, e, 3) for a, b, e in zip(d.OR, d.OR_ci_low, d.OR_ci_high)],
                                "p": d.p.map(fp)}))
d = rd(O / "stage6/p2b_sensitivity.csv")
T["p2b"] = md(pd.DataFrame({"Model": d.model, "AUC": d.auc.map(lambda x: ff(x, 4)),
                            "n train": d.n_train.map(fi), "n validation": d.n_validation.map(fi)}))
d = rd(O / "stage6/standardisation_train_stats.csv")
T["s6_std"] = md(pd.DataFrame({"Variable": "`" + d.variable + "`", "Training mean": d.train_mean.map(lambda x: ff(x, 3)),
                               "Training SD": d.train_sd.map(lambda x: ff(x, 3))}))

d = rd(O / "stage3/year_fe.csv")
T["year_fe"] = md(pd.DataFrame({"Interview year": d.term.str.replace("interview_year_", "").str.replace(".0", "", regex=False),
                                "OR (ref. 2010)": d.OR.map(lambda x: ff(x, 3)), "p": d.p.map(fp)}))
d = rd(O / "stage3/sample_flow.csv")
T["s3_flow"] = md(pd.DataFrame({"Specification": d.spec, "Step": d.step, "n": d.n.map(fi)}))

d = rd(O / "audit/indicative_prevalence.csv")
T["indicative"] = md(pd.DataFrame({"Level": d.level, "Group": d.group,
                                   "v1 n": d.v1_n.map(fi), "v1 %": d.v1_pct.map(lambda x: ff(x, 1)),
                                   "Primary n": d.a1_primary_n.map(fi), "Primary %": d.a1_primary_pct.map(lambda x: ff(x, 1)),
                                   "S1 n": d.s1_lower_bound_n.map(fi), "S1 %": d.s1_lower_bound_pct.map(lambda x: ff(x, 1)),
                                   "S2 n": d.s2_plus_elec_nr_n.map(fi), "S2 %": d.s2_plus_elec_nr_pct.map(lambda x: ff(x, 1))}))
d = rd(O / "audit/ni_oil_lost_by_wave.csv")
T["ni_lost"] = md(pd.DataFrame({"Wave": d.wave, "NI households": d.ni_households.map(fi),
                                "NI oil households": d.ni_oil_households.map(fi),
                                "Oil households dropped by v1 `fuelduel` rule": d.ni_oil_dropped_by_fuelduel_rule.map(fi),
                                "…of which in v2 primary sample": d.of_which_in_A1_primary.map(fi),
                                "Oil households kept by v1": d.ni_oil_kept_by_v1.map(fi)}))
d = rd(O / "audit/gap_explanations.csv").fillna("")
T["gaps"] = md(d.rename(columns={"gap": "Gap", "explanation": "Explanation", "n": "n", "detail": "Detail"}))
d = rd(O / "audit/elec_not_reported_rent_check.csv")
T["elec_nr"] = md(pd.DataFrame({"Group": d.group, "n": d.n.map(fi), "n (waves c–o)": d.n_waves_c_to_o.map(fi),
                                "Renting (%)": d.pct_renting.map(lambda x: ff(x, 1)), "Owner (%)": d.pct_owner.map(lambda x: ff(x, 1)),
                                "Gas in rent (% of asked)": d.gaspay_included_in_rent_pct_of_asked.map(lambda x: ff(x, 1)),
                                "Elec. in rent (% of asked)": d.elecpay_included_in_rent_pct_of_asked.map(lambda x: ff(x, 1))}))
d = rd(O / "audit/missing_vs_observed_spend.csv")
T["miss_obs"] = md(pd.DataFrame({"Dimension": d.dimension, "Group": d.group, "Missing": d["missing"].map(fi),
                                 "Observed": d.observed.map(fi),
                                 "% missing in group": d.pct_missing_within_group.map(lambda x: ff(x, 1))}))
d = rd(O / "audit/missing_by_mode_wave.csv")
T["miss_mode"] = md(pd.DataFrame({"Wave": d.wave, "Mode": d["mode"], "Observed": d.observed.map(fi), "Missing": d["missing"].map(fi),
                                  "n": d.n.map(fi), "% missing": d.pct_missing.map(lambda x: ff(x, 1)),
                                  "% of wave": d.pct_of_wave.map(lambda x: ff(x, 1))}))

d = rd(O / "jrf/oil_share_by_region.csv")
T["oil_region"] = md(pd.DataFrame({"Region": d.region, "Oil heating, pooled unweighted (%)": d.oil_share_pct_unweighted.map(lambda x: ff(x, 1))}))
d = rd(O / "jrf/ni_oil.csv")
T["ni_oil"] = md(pd.DataFrame({"Sample": d["sample"], "NI households": d.n_households.map(fi),
                               "Oil share, weighted (%)": d.oil_share_pct_weighted.map(lambda x: ff(x, 1)),
                               "Oil share, unweighted (%)": d.oil_share_pct_unweighted.map(lambda x: ff(x, 1)),
                               "Oil: n / weighted % vulnerable": [f"{fi(a)} / {ff(b,1)}" for a, b in zip(d.oil_n, d.oil_vuln_pct_weighted)],
                               "No oil: n / weighted % vulnerable": [f"{fi(a)} / {ff(b,1)}" for a, b in zip(d.non_oil_n, d.non_oil_vuln_pct_weighted)]}))

d = rd(O / "stage5/thesis_secondary_suppression_log.csv")
T["supp_log"] = md(d)


# ---------------------------------------------------------------- fill
def fill(text: str, prefix: str) -> str:
    def rep(mo):
        key = mo.group(1)
        if key not in T:
            raise KeyError(f"unknown table key {key}")
        return T[key]
    text = re.sub(r"\{\{TABLE:([A-Za-z0-9_\-]+)\}\}", rep, text)
    text = text.replace("{{P}}", prefix)
    left = re.findall(r"\{\{[^}]*\}\}", text)
    if left:
        raise ValueError(f"unfilled placeholders: {left}")
    return text


TARGETS = {
    "README.md": ("README.md", ""),
    "01_outputs_catalog.md": ("reports/01_outputs_catalog.md", "../"),
    "02_findings_report.md": ("reports/02_findings_report.md", "../"),
    "03_policy_brief.md": ("reports/03_policy_brief.md", "../"),
    "04_journal_submission_materials.md": ("reports/04_journal_submission_materials.md", "../"),
    "05_data_description.md": ("reports/05_data_description.md", "../"),
    "06_methodology.md": ("reports/06_methodology.md", "../"),
}

if __name__ == "__main__":
    which = sys.argv[1:] or list(TARGETS)
    for name in which:
        src = TEMPLATES / name
        if not src.exists():
            continue
        dst, prefix = TARGETS[name]
        (ROOT / dst).write_text(fill(src.read_text(), prefix))
        print("wrote", dst)
