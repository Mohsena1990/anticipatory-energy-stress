# Journal Submission Materials (rerun v2)

**Working title:** *Who Becomes Fuel Vulnerable, and When? Forecast-Conditioned Household Risk Modelling in the UK*
**Target scope:** *Energy Policy* (Elsevier) or comparable energy–social-science journals.
**Source material:** [`02_findings_report.md`](02_findings_report.md) (results, all tables and figures), [`06_methodology.md`](06_methodology.md) (methods), [`03_policy_brief.md`](03_policy_brief.md) (policy translation), `outputs_v2/results_inventory.csv` (every number, with source and commit).
**Status:** branch `rerun-v2`, analysis frozen 2026-09-26. This document replaces the v1 version, whose abstract, hypothesis results and highlights are superseded (see §8). The wording of the H2–H4 verdicts is proposed and awaits the author's confirmation.

---

## 0. Abstract

*(≈260 words.)*

UK fuel-poverty policy relies on realised, income-based measures, leaving open whether anticipated energy-price stress helps identify vulnerable households. We attach a forecast of energy-price stress (FES), built from rolling walk-forward forecasts of gas, electricity and carbon price growth with no look-ahead, to 339,201 household-waves of the UK Household Longitudinal Study (interviews 2009–2025). The outcome is a routing-aware measure of fuel spend at or above 10% of net income. Weighted prevalence fell from 12.0% (2009–11 wave) to 6.5% (2020–22) and rose to 12.5% in the 2022–24 wave. The forecasts did not significantly beat a no-change benchmark. In logit models with year fixed effects and PSU-clustered errors (n = 221,877), current financial difficulty (OR 1.65 per point) and employment security (OR 0.17) dominated; forecast-minus-realised stress had a small, robust association (OR 0.93 per SD). A pre-registered test found no evidence that household resources buffer price stress (interaction p = 0.16), and any buffering larger than 17% of the stress slope is ruled out. A time-matched comparison with Joseph Rowntree Foundation income-poverty rates agreed by tenure (ρ = 0.80) but not by region (ρ = −0.10). Northern Ireland had the highest fuel vulnerability (14.4%) but the lowest income poverty, and heating-oil use accounted for about two-thirds of its excess risk. On held-out transitions between the three most recent waves, household predictors forecast next-wave vulnerability with AUC 0.739, below current fuel burden alone (0.780), and FES added nothing. Fuel vulnerability is distinct from income poverty and is driven chiefly by household finances and heating fuel. For early warning, a household's current fuel burden is the most useful signal.

---

## 1. Research questions

The research questions are those of the submitted draft (plan principle: keep the RQs, recompute the evidence).

### RQ1 — Forecast integration

**Can a forecast of energy-price stress be attached to households without look-ahead, so that it varies by household interview month, and how good are the forecasts against simple benchmarks?**

*Evidence.* Rolling core forecasts re-tuned at every origin; December Y−1 vintage matched to interview month; accuracy against naive and seasonal-naive forecasts with Diebold–Mariano tests; prediction-interval coverage (Tables 4-1, A-1, A-5, A-6; Figures 4-1, 4-2, A-1).
*Answer.* Yes, it can be attached without look-ahead (324,055 household-waves, interviews 2010–2025). But the forecasts are weak: none significantly beats the no-change forecast, and the intervals cover only 28–48% of outcomes.

### RQ2 — Resource moderation

**Within a Conservation of Resources (COR) framework, do household resources moderate the effect of forecast–realised price stress on the fuel burden?**

*Evidence.* A pre-registered CFA (failed its criteria) → formative resource indices; OLS with R × FES Delta; Delta slopes at resource percentiles; buffering bound (Tables 4-5, A-9, A-10; Figure 4-12).
*Answer.* No. The interaction is null, and a substantial buffering effect is ruled out (H1).

### RQ3 — Drivers of vulnerability

**Which household factors are associated with fuel vulnerability, and how does forecast price stress compare with household financial position?**

*Evidence.* Logit with year FE and PSU clustering, 11 pre-specified specifications, per-SD comparison (Tables 4-3, 4-4, 4-9, A-3, A-4; Figure 4-11).
*Answer.* Current financial difficulty and employment security dominate. FES Delta has a small, robust association (H2, H3).

### RQ4 — Geographic and social structure, and income poverty

**How is fuel vulnerability distributed by region and social group, and does it match an independent income-poverty benchmark?**

*Evidence.* Weighted prevalence with PSU-bootstrap CIs; time-matched JRF comparison with rank intervals; NI-oil sequence with AMEs (Tables 4-2, 4-6, 4-7, A-12; Figures 4-3 to 4-10, 4-13 to 4-15).
*Answer.* Partly. Tenure agrees, regions do not, and Northern Ireland's heating-oil exposure is the clearest divergence (H4).

### RQ5 — Prospective prediction

**Can next-wave fuel vulnerability be predicted before it is observed, and does the forecast add to household information?**

*Evidence.* Linked transitions; training a→b … l→m, validation m→n and n→o; P0 benchmark, P1 household, P2 + FES on one common sample; AUC, PR-AUC, calibration, top-k (Tables 4-8, A-2, A-7, A-8; Figures 4-16, 4-17).
*Answer.* Moderately. The household model does not beat current fuel burden, and FES adds nothing (H5).

---

## 2. Hypotheses

Each hypothesis is stated as specified. Its test and decision rule were fixed before fitting, except where §9 of the plan logs a change.

### H1 — Resource buffering (COR)

**Statement.** Household resources moderate the association between forecast–realised price stress and the fuel burden: better-resourced households are less affected.
**Test.** OLS of the fuel-to-income ratio on R (OBJECT + CONDITION + PERSONAL, formative), FES Delta (growth-only), R × Delta and year FE; PSU-clustered SEs; n = 269,372. With Delta = forecast − realised and R oriented so that higher = more resources, COR predicts a **positive** interaction.
**Result — not supported.** Interaction −0.000034 [−0.000081, 0.000014], p = 0.16. The Delta slope is negative at every resource level and no flatter at high R (p10 −0.00045, p90 −0.00063). At the upper confidence limit, buffering is at most 0.018 pp of income per SD of Delta, 17% of the p10 slope. Sensitivities (R with ENERGY; 4-term FES) are also not supported. The logit on the binary outcome is negative on the log-odds scale but smaller at high R on the probability scale; both reflect baseline-risk differences. Resources have a strong main effect (−0.0047 per unit R).

### H2 — FES as an independent predictor

**Statement.** Forecast price stress is associated with household fuel vulnerability net of household characteristics.
**Test.** Primary driver logit (Table 4-3); 10 sensitivities (Table 4-9).
**Result — supported, small effect (proposed wording).** FES Delta OR 0.972 [0.962, 0.981] per unit, 0.93 per SD. It is robust across every specification (0.968–0.974), including two-way clustering, month FE, lagged strain, the 4-term FES, and the S1 and S2 outcomes. Vulnerability is higher when realised stress exceeds the forecast. The association is identified from within-year variation across interview months. FES does **not** improve next-wave prediction (H5).

### H3 — Household financial position dominates the price environment

**Statement.** Household financial position is more strongly associated with fuel vulnerability than forecast price stress.
**Test.** Same model; per-SD comparison (Table 4-4).
**Result — supported, reframed (proposed wording).**

- Current financial difficulty: OR 1.65 per point [1.61, 1.69], 1.59 per SD.
- Employment-status security: OR 0.17 [0.15, 0.20], 0.63 per SD.
- FES Delta: 0.93 per SD.

The v1 strain composite (OR 6.79) is not a scale (α = 0.27) and is not used. GHQ distress and financial expectations are slightly protective once current difficulty is included.

### H4 — Agreement with an independent income-poverty benchmark

**Statement.** A fuel-specific vulnerability measure broadly agrees in rank with JRF income poverty, with divergence attributable to identifiable structural factors.
**Test.** Weighted UKHLS rates time-matched to the JRF periods; Spearman and Pearson correlations; bootstrap rank intervals (Tables 4-6, A-12).
**Result — partially supported (proposed wording).**

- Tenure: ρ = 0.80, with outright owners the exception.
- Disability, family type and work status: same direction.
- Region: ρ = −0.10 (0.18 excluding NI).
- Ethnicity: inconclusive (ρ = 0.26, wide CIs).

Northern Ireland is the clearest structural divergence: highest on fuel vulnerability (14.4%, P(rank 1) = 0.95), lowest on income poverty. Its gap with the South East falls from 6.8 to 2.4 pp once oil use is controlled.

### H5 — Prospective predictability

**Statement.** Next-wave fuel vulnerability can be predicted from information available before it is observed, and FES adds to household predictors.
**Test.** P0 (current burden, benchmark), P1 (household predictors), P2 (P1 + FES for the next interview month) on one common sample; training-only standardisation; validation on m→n and n→o (n = 19,960); 2,000 PSU-bootstrap replicates (Table 4-8).
**Result — partially supported.**

- P1: AUC 0.739 [0.727, 0.751]; PR-AUC 0.280 against a 0.105 baseline.
- P0 is better: AUC 0.780; ΔAUC P1 − P0 −0.041 [−0.053, −0.029].
- FES: no improvement (ΔAUC −0.0006).
- Post-hoc P3 = P0 + P1: 0.781, no gain over P0.

### Summary table

{{TABLE:T4-10}}

---

## 3. Contributions

1. **A routing-aware fuel-spend outcome for UKHLS.** Treating the not-asked code of the combined-bill question as missing drops almost all off-gas-grid households (84% of NI oil users) and understates prevalence in every wave. We document the routing, the correct treatment of every code (Tables 3-3 and 3-4) and lower and upper bounds.
2. **Forecast-conditioned exposure without look-ahead.** Rolling core forecasts are re-tuned at each origin and attached from the vintage published before each interview year, with past-only standardisation. We also show the forecasts do not beat a naive benchmark, and we report that plainly.
3. **A pre-registered test of COR buffering with a quantified bound.** Rather than a bare non-significant interaction, we report the largest buffering compatible with the data.
4. **Benchmarked prediction.** Household prediction is compared with a current-burden benchmark on one common sample with leakage control. The benchmark wins, which changes the practical case for complex early-warning models.
5. **Time-matched external comparison.** Fuel vulnerability is compared with JRF income poverty by time-matched window, with weights and bootstrap rank intervals. The Northern Ireland divergence is decomposed with average marginal effects.

## 4. Achievements

- **Scale:** 339,201 household-waves (15 waves, interviews 2009–2025); primary analytical n 286,902; driver model n 221,877; 261,759 linked transitions.
- **Forecasting:** 16 origins × 3 series × 4 models, with per-origin tuning; formal evaluation against two benchmarks.
- **Transparency:** a pre-registered plan with a dated deviation log; 622 quotable numbers with source file and commit; a thesis bundle with a MANIFEST of source commits.
- **Disclosure control:** no row-level data in git; small-cell suppression on every tracked table, with checks for secondary disclosure.

## 5. Highlights

*(Elsevier: 3–5 bullets, each ≤ 85 characters including spaces.)*

- Correcting survey routing restores off-gas households to UK fuel vulnerability. *(79)*
- Financial difficulty, not forecast energy-price stress, dominates household risk. *(81)*
- Household resources do not buffer forecast-realised energy-price stress. *(72)*
- Current fuel burden predicts next-wave vulnerability better than household data. *(80)*
- Heating oil accounts for two-thirds of Northern Ireland's excess fuel vulnerability. *(84)*

## 6. Suggested keywords

Fuel poverty; energy vulnerability; household panel data; Understanding Society; energy-price forecasting; Conservation of Resources theory; early warning; income poverty; Northern Ireland; heating oil.

## 7. Figure and table plan for the manuscript

Thesis numbering, with suggested placement: **M** = main text, **S** = supplementary. All files are in `outputs_v2/thesis_assets_v2/` and are rendered in [`02_findings_report.md`](02_findings_report.md) or [`05_data_description.md`](05_data_description.md).

| Item | Content | Placement |
|---|---|---|
| Figure 3-1 | Analysis pipeline | M |
| Figure 3-2 | Interview timing | S |
| Figure 3-3 | Missing spend by mode and wave | S |
| Figure 3-4 | Distributions of the outcome and FES | S |
| Figure 3-5 | Household-waves by region | S |
| Figure 3-6 | Core forecasting series | S |
| Figure 4-1 | Relative RMSE by year | M |
| Figure 4-2 | Growth-only FES, forecast vs realised | M |
| Figure 4-3 | National trend with S1 band | M |
| Figure 4-4 | Trend by interview year, JRF window | S |
| Figure 4-5 | Regional prevalence map | M |
| Figure 4-6 | Region × year heatmap | S |
| Figure 4-7 | Regional change map | S |
| Figure 4-8 | Resources and financial difficulty by region | S |
| Figure 4-9 | Social groups with CIs | M |
| Figure 4-10 | Prepayment by status | S |
| Figure 4-11 | Driver forest plot | M |
| Figure 4-12 | H1 Delta slopes | M |
| Figure 4-13 | Regions vs JRF | M |
| Figure 4-14 | Other dimensions vs JRF | S |
| Figure 4-15 | Northern Ireland and oil | M |
| Figure 4-16 | ROC P0–P3 | M |
| Figure 4-17 | Calibration | S |
| Figure 4-18 | Equivalised-income sensitivity | S |
| Figure 4-19 | FES terciles | S |
| Figures A-1 to A-4 | Winning models; macro covariates; trend by year; v1 exploratory (labelled v1) | S |
| Tables 3-2 to 3-7 | Wave n; sample flow; routing; measures; forecasting series; JRF metadata | 3-3, 3-4 M; others S |
| Table 4-1 | Forecast accuracy and PI coverage | M |
| Table 4-2 | Prevalence by region and social group | S |
| Table 4-3 | Primary driver model | M |
| Table 4-4 | Per-SD ranking | S |
| Table 4-5 | H1 | M (verdicts and bound); S (slopes, logit) |
| Table 4-6 | JRF comparison and agreement | M (agreement); S (categories) |
| Table 4-7 | NI-oil AMEs | M |
| Table 4-8 | Prediction P0–P3 | M |
| Table 4-9 | Robustness summary | S |
| Table 4-10 | Hypothesis verdicts | M |
| Tables A-1 to A-14 | MASE; per-transition AUC; all driver specifications; model fit; relative RMSE by year; DM tests; calibration; prediction sample flow; CFA fit and loadings; equivalised income; regional window CIs; interview timing; regional counts | S |

## 8. Notes for the manuscript's limitations section

- **Associations only.** FES varies by interview year-month only; with year FE its coefficient is identified from within-year variation across months.
- **Weak forecasts.** No series beats a no-change forecast significantly; PI coverage is 28–48%; the forecasts missed the 2022 shock.
- **Resources.** The CFA failed its pre-registered criteria, and its complete-case sample was 99% owner-occupiers. The composites are formative indices.
- **Missing spend.** The outcome is complete-case; missingness rises to 20% in wave o. The S1 lower bound is reported throughout.
- **Unequivalised threshold.** Equivalising income alone would flip 45.8% of 5+ person households into vulnerability.
- **JRF comparisons** differ in construct, and for family type and work status in unit. Ethnicity is inconclusive.
- **Prediction** is validated on two crisis-era transitions, and all models under-predict risk there.
- **Post-hoc analyses** are labelled: P3, `sens_no_qualification`, the H5 verdict wording.
- **Corrections to v1 must be disclosed** if any v1 result was circulated:
  - the outcome routing error;
  - look-ahead and hindsight in the v1 FES;
  - the mislabelled health variables;
  - the disability variable (a consent flag);
  - the uncorrected JRF work-status values;
  - the "sign flips across refits" account of H1, which came from code changes, not refits.

  The full list is in `outputs_v2/reports/v1_to_v2_change_summary.md`.
