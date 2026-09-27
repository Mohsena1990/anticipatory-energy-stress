# Findings Report: Anticipatory Fuel Stress Watch (rerun v2)

**Scope.** The v2 results for RQ1–RQ5, with every Chapter 4 and appendix table and every results figure. The data tables of Chapter 3 are in [`05_data_description.md`](05_data_description.md) and the methods in [`06_methodology.md`](06_methodology.md).
**Status.** Branch `rerun-v2`, analysis frozen 2026-09-26. v1 (tag `submitted-draft-v1`) is superseded. Every number below is a row of `outputs_v2/results_inventory.csv` or is read from the table shown next to it.
**Figures** are the thesis versions in `outputs_v2/thesis_assets_v2/figures/`. This is a local build (`python scripts/build_thesis_assets.py`) and is not tracked in git. **Tables** are rendered from the CSVs in `outputs_v2/thesis_assets_v2/tables/`, or from the tracked `outputs_v2/` file named in the caption.

---

## 1. Executive summary

{{TABLE:T4-10}}

*Table 4-10. Hypothesis verdicts. H1 and H5 wording is confirmed by the author; H2–H4 wording is proposed.*

1. **The outcome changed.** v1 dropped households routed out of the combined-bill question (`fuelduel` = −8), which removed almost every off-gas-grid household, including 84% of NI oil users. v1 also set non-response to £0. The corrected outcome has 286,902 household-waves (v1: 255,324) and a higher prevalence in every wave.
2. **Trend.** Weighted prevalence falls from 12.0% (wave a) to 6.5% (wave l) and rises to 12.5% (wave n, fieldwork 2022–24). The peak by interview year is 2023 (14.7%).
3. **Forecasts are weak.** No v2 series forecast significantly beats a no-change forecast. Prediction intervals cover 28–48% of outcomes, so the primary FES uses the three growth terms only.
4. **FES has a small, robust association with vulnerability** (H2): OR 0.972 per unit of FES Delta, 0.93 per SD, stable across all 11 specifications (0.968–0.974).
5. **Household finances dominate** (H3). Current financial difficulty (OR 1.65 per point) and employment-status security (OR 0.17) have far larger effects than FES.
6. **Resources do not buffer FES** (H1, not supported). The interaction is null, and a substantial buffering effect is ruled out.
7. **The JRF comparison is mixed** (H4). Tenure agrees; three two-group dimensions point in the same direction; regions do not agree; ethnicity is inconclusive. Northern Ireland has the highest fuel vulnerability but the lowest income poverty, and heating oil accounts for about two-thirds of its excess risk.
8. **Prediction** (H5, partially supported). The household model reaches AUC 0.739 on held-out waves, but the current fuel burden alone reaches 0.780, and FES adds nothing.

---

## 2. What changed from v1

The full list, with commits and the effect on each thesis claim, is in [`outputs_v2/reports/v1_to_v2_change_summary.md`](../outputs_v2/reports/v1_to_v2_change_summary.md). In brief:

| Area | v1 | v2 |
|---|---|---|
| Outcome | `fuelduel` = −8 read as missing (household dropped); non-response set to £0 | Routing-aware spend, non-response missing (complete-case); S1 and S2 bounds |
| Interview timing | Sample (issue) month | Actual household interview date |
| FES | Macro variant chosen by hindsight; tuned once on 2024; forecast for Y+1 given to year-Y interviews | Core only, tuned per origin, Dec Y−1 vintage, past-only z-scores, growth-only primary |
| Weights | Unweighted | Per-wave household cross-sectional weights |
| Strain | 4-item composite (α 0.27) with bill arrears counted twice | Components entered separately |
| Driver model | No year effects, unclustered, N = 103,621 | Year FE, PSU clustering, region FE and oil in the NI sequence, N = 221,877 |
| Resources | Second-order SEM (degenerate) | CFA failed pre-set criteria → formative composites |
| JRF | Pooled 2009–2024, unweighted | Time-matched windows, weighted, bootstrap CIs |
| Prediction | One model, AUC only | P0 benchmark, P1, P2 on one common sample; calibration, PR-AUC, top-k |
| Disability | `healthlink` (a record-linkage consent flag, wave a only) | Long-standing illness plus any substantial difficulty (`disdif`) |

---

## 3. Forecasts and FES (RQ1)

### 3.1 Accuracy against benchmarks

![Figure 4-1. Relative RMSE by target year](../outputs_v2/thesis_assets_v2/figures/fig4-1_relrmse_by_year.png)

*Figure 4-1. v2 forecast accuracy relative to naive and seasonal-naive benchmarks, by target year.*

{{TABLE:T4-1}}

*Table 4-1. Forecast accuracy on the three growth terms (192 target months per series, 2010–2025) and 95% prediction-interval coverage. v1_macro is the variant v1 attached to households.*

- **Against the naive (no-change) forecast, no version is significantly better** (all Diebold–Mariano p ≥ 0.17). The v2 point estimates are below 1 for gas (0.71) and electricity (0.83) and about 1 for carbon (0.99).
- **The gas gain is concentrated in 2023.** Relative RMSE was 0.30 that year, but v2 beats naive in only 6 of 16 target years (Table A-5; `forecast_accuracy_pooled.csv`, `pct_years_beating_naive` = 37.5%).
- **Against seasonal naive**, v2 is better for all three series, significantly so only for carbon (p = 0.002).
- **The v1 variant attached to households (v1_macro) was worse than naive for gas** (relative RMSE 1.58).
- **Prediction intervals are far too narrow.** v2 coverage is 27.6% (carbon), 32.3% (gas) and 48.4% (electricity) against a nominal 95%. This is why the uncertainty term was removed from the primary FES (deviation log, 2026-09-26).

{{TABLE:fc_pooled}}

*Pooled accuracy, all versions and benchmarks (`outputs_v2/fes_eval/forecast_accuracy_pooled.csv`). MASE scales by the in-sample one-step naive error, so values above 1 are expected at 1–12-month horizons; compare across rows, not against 1.*

{{TABLE:pi}}

*Prediction-interval coverage and width (`outputs_v2/fes_eval/uncertainty_pi.csv`).*

### 3.2 Which model wins

![Figure A-1. Winning model by series and target year](../outputs_v2/thesis_assets_v2/figures/figA_radial_winners.png)

*Figure A-1. Winning rolling core model by series and target year (v2), read clockwise from 12 o'clock.*

{{TABLE:wins}}

*Number of forecast origins won by each model (`outputs_v2/fes_eval/model_wins.csv`). In v2, the LSTM wins 36 of 48 series-origins.*

![Winning core model and validation RMSE, polar view](../outputs_v2/figures/model_selection_polar_core.png)

*Pipeline figure `outputs_v2/figures/model_selection_polar_core.png`: bar = winning model's validation RMSE for that target year, colour = winning model.*

| Gas | Electricity | Carbon |
|---|---|---|
| ![gas](../outputs_v2/figures/rolling_forecast_performance_gas.png) | ![electricity](../outputs_v2/figures/rolling_forecast_performance_electricity.png) | ![carbon](../outputs_v2/figures/rolling_forecast_performance_carbon.png) |

*Pipeline figures `outputs_v2/figures/rolling_forecast_performance_{gas,electricity,carbon}.png`: the winning model's validation RMSE by target year, points coloured by model.*

### 3.3 The FES signal attached to households

![Figure 4-2. Growth-only FES forecast vs realised](../outputs_v2/thesis_assets_v2/figures/fig4-2_fes_growth3_forecast_vs_realised.png)

*Figure 4-2. Growth-only FES: forecast vs realised by target month, 2010–2025.*

![Rolling FES by target year](../outputs_v2/figures/fes_rolling_trend.png)

*Pipeline figure `outputs_v2/figures/fes_rolling_trend.png`: annual mean of the 4-term rolling FES (forecast) against the realised index. The forecast misses the 2022 shock: realised 15.4, forecast 2.1.*

The FES attached to households differs substantially from v1's. The v2 FES Delta correlates 0.16 with v1's, because v1 gave year-Y interviews a forecast that used data up to December Y (look-ahead) and chose the macro variant by hindsight.

{{TABLE:fes_v1v2}}

*FES attached to households, v1 vs v2 (`outputs_v2/fes_eval/fes_attached_v1_vs_v2.csv`).*

<details><summary><b>Table A-1. MASE</b></summary>

{{TABLE:TA-1}}

</details>

<details><summary><b>Table A-5. Relative RMSE by target year</b></summary>

{{TABLE:TA-5}}

</details>

<details><summary><b>Table A-6. Diebold–Mariano tests (squared-error loss, Newey–West lag 11, HLN correction)</b></summary>

{{TABLE:TA-6}}

</details>

---

## 4. How many households are fuel vulnerable, and who (RQ4)

### 4.1 Trend

![Figure 4-3. National trend by wave with S1 band](../outputs_v2/thesis_assets_v2/figures/fig4-3_trend_wave_s1band.png)

*Figure 4-3. National trend by wave, weighted, with the S1 lower bound.*

{{TABLE:prev_wave}}

*Prevalence by wave, all outcome definitions (`outputs_v2/descriptives/prevalence_by_wave.csv`). Waves span two to three calendar years; say "wave n (fieldwork 2022–24)", not "2022".*

![Figure 4-4. Trend by interview year with the JRF crisis window](../outputs_v2/thesis_assets_v2/figures/fig4-4_trend_vs_jrf_tracker.png)

*Figure 4-4. Trend by interview year, with the JRF cost-of-living crisis window shaded.*

![Figure A-3. Trend by interview year, supplementary](../outputs_v2/thesis_assets_v2/figures/figA_trend_interview_year.png)

*Figure A-3. Trend by interview year (supplementary). 2025 has only 223 households with an observed outcome.*

{{TABLE:prev_year}}

*Prevalence by interview year (`outputs_v2/descriptives/prevalence_by_interview_year.csv`).*

The trend falls from the start of the panel to a low around 2016–2021 and rises sharply in 2022–2023, the period of the energy-price crisis. The S1 lower bound is 1–3 points lower but has the same shape. The same shapes appear in the tracked pipeline figures `outputs_v2/descriptives/trend_primary_with_s1_band.png` and `trend_by_interview_year_supplementary.png`:

| By wave | By interview year |
|---|---|
| ![](../outputs_v2/descriptives/trend_primary_with_s1_band.png) | ![](../outputs_v2/descriptives/trend_by_interview_year_supplementary.png) |

### 4.2 Regions

![Figure 4-5. Regional prevalence map](../outputs_v2/thesis_assets_v2/figures/fig4-5_region_map_weighted.png)

*Figure 4-5. Regional prevalence, weighted (pooled mean of per-wave weighted rates).*

![Figure 4-6. Region × interview-year heatmap](../outputs_v2/thesis_assets_v2/figures/fig4-6_region_year_heatmap_masked.png)

*Figure 4-6. Weighted prevalence by region and interview year (cells with n < 100 masked).*

![Figure 4-7. Change in regional prevalence](../outputs_v2/thesis_assets_v2/figures/fig4-7_region_change_map.png)

*Figure 4-7. Change in weighted regional prevalence, waves a–e to k–o (navy = fall, orange = rise).*

{{TABLE:reg_change}}

*Regional change, waves a–e vs k–o (`outputs_v2/stage7/regional_change_early_late.csv`). NI shows the largest fall (−8.9 pp) but remains the highest region.*

### 4.3 Social groups

![Figure 4-9. Social groups with CIs](../outputs_v2/thesis_assets_v2/figures/fig4-9_social_groups_panel_ci.png)

*Figure 4-9. Pooled weighted prevalence by social group, with 95% CIs (categories with n < 100 suppressed).*

{{TABLE:T4-2}}

*Table 4-2. Pooled weighted prevalence by region and social group, with 95% PSU-bootstrap CIs.*

The largest gradients are by employment (workless 15.0% vs full-time or self-employed 3.5%), by family type (lone parents with 1–2 children 15.1% vs couples with 1–2 children 3.8%) and by tenure (outright owners 11.1% and social renters 11.1% vs mortgage holders 4.2%). Households with a disabled adult are at 10.5% against 7.6%. Among ethnic groups, Black Caribbean (14.7%) and Pakistani (13.8%) households have the highest pooled rates.

### 4.4 Resources and financial difficulty by region

![Figure 4-8. Resource composite and financial difficulty by region](../outputs_v2/thesis_assets_v2/figures/fig4-8_resource_and_findifficulty_by_region.png)

*Figure 4-8. Weighted mean resource composite and current financial difficulty by region.*

![Resource composite by region, pipeline figure](../outputs_v2/resources/figures/resource_composite_by_region.png)

*Pipeline figure `outputs_v2/resources/figures/resource_composite_by_region.png`.*

{{TABLE:res_region}}

*Weighted mean domain composites by region (`outputs_v2/resources/resource_by_region.csv`; composites are standardised, 0 = sample mean). The South East is highest and the North East lowest on the primary composite R.*

### 4.5 Prepayment meters

![Figure 4-10. Prepayment by vulnerability](../outputs_v2/thesis_assets_v2/figures/fig4-10_prepayment.png)

*Figure 4-10. Prepayment-meter use by vulnerability status.*

{{TABLE:prepay}}

*`outputs_v2/stage7/prepayment_by_vulnerability.csv`. Prepayment use is higher among vulnerable households (24.4% vs 13.6%). Households that ration energy on a prepayment meter can fall below the 10% line, so the ratio measure can miss some hardship.*

---

## 5. Drivers of fuel vulnerability (RQ3; H2, H3)

![Figure 4-11. Driver-model forest plot](../outputs_v2/thesis_assets_v2/figures/fig4-11_driver_forest.png)

*Figure 4-11. Primary driver model, odds ratios (FES in orange). Pipeline version: `outputs_v2/stage3/figures/primary_or_forest.png`.*

![Primary driver model forest plot, pipeline version](../outputs_v2/stage3/figures/primary_or_forest.png)

{{TABLE:T4-3}}

*Table 4-3. Primary driver model: logit of `high_fuel_vulnerable`, interview-year fixed effects, SEs clustered on PSU (8,801 PSUs); n = 221,877 household-waves, interviews 2010–2025; 17,209 events; McFadden R² 0.153.*

{{TABLE:T4-4}}

*Table 4-4. Continuous predictors ranked by the absolute log odds ratio per SD.*

**Reading the model.**

- **Current financial difficulty** is the strongest household-finance term: OR 1.65 per point on the 1–5 scale (1.59 per SD).
- **Employment-status security** is the strongest protective factor: OR 0.17 (0.63 per SD).
- **The OECD equivalence scale** has the largest per-SD effect (OR 0.47 per SD). It is partly mechanical: the outcome uses unequivalised income, so larger households have more income per unit of fuel need.
- **GHQ distress and financial expectations are slightly protective** once current difficulty is in the model. They were never one scale with it (α = 0.27; `finnow`–`finfut_risk` r = 0.02).
- **Housing size raises risk** (bedrooms OR 1.33 per room, rooms 1.15), as does tenure security (2.03). This is consistent with larger, owner-occupied homes costing more to heat relative to income. Cars are protective (0.86).
- **Health.** No long-standing illness raises odds (1.25). Self-rated health is null (1.03). v1 had mislabelled the first as self-rated health.
- **Workless household** is null (1.05) net of employment security. Lone parents (1.50) and large families (1.96) are at higher risk.
- **FES Delta** is small but robust: OR 0.972 [0.962, 0.981] per unit, 0.93 per SD. Delta = forecast − realised, so vulnerability is higher when realised stress exceeds what was forecast.

{{TABLE:year_fe}}

*Interview-year fixed effects, primary model (reference 2010; `outputs_v2/stage3/year_fe.csv`). Odds are lowest in 2016–2021 and highest in 2023 (OR 1.58).*

### 5.1 Sensitivities

{{TABLE:TA-3}}

*Table A-3 (main models). Odds ratios for every pre-specified sensitivity. Composite and lagged specifications replace the three strain components with composites or lagged terms (their rows are blank for the components and filled for the composite terms). Full file: `TA-3_driver_sensitivities.csv`, 1,450 rows.*

{{TABLE:TA-3-twoway}}

*Table A-3 (two-way clustered model). Same point estimates as the primary model; SEs clustered on PSU and on interview year-month (the unit at which FES varies).*

<details><summary><b>Table A-4. Driver-model N, events, PSUs and pseudo-R² (all 71 models)</b></summary>

{{TABLE:TA-4}}

</details>

---

## 6. Resources and the buffering hypothesis (RQ2; H1)

### 6.1 Measuring resources

The pre-registered correlated four-factor CFA (complete-case ML) failed the fixed criteria on its only attempt.

{{TABLE:TA-9}}

*Table A-9. CFA fit.*

{{TABLE:TA-10}}

*Table A-10. CFA loadings. Employment-status security loads negatively on CONDITION (−0.74), bill security is near zero, and the ENERGY marker is a Heywood case (std. loading 1.00).*

{{TABLE:res_decision}}

{{TABLE:cfa_corr}}

*Factor correlations (`outputs_v2/resources/cfa_factor_correlations.csv`).*

The complete-case CFA sample is 99% owner-occupiers (private renters 0.1%, social renters 0.6%; `cfa_sample_composition.csv`), because house and car values are asked only of owners. Resources are therefore measured by **unit-weighted formative indices**: the mean of standardised items per domain, requiring at least 50% of items observed, then re-standardised. α is descriptive only.

{{TABLE:alpha}}

*Cronbach's α per domain (`outputs_v2/resources/composite_alpha.csv`).*

{{TABLE:comp_corr}}

*Correlations of domain composites and of R (`outputs_v2/resources/composite_item_correlations.csv`). R_primary = OBJECT + CONDITION + PERSONAL; R_with_energy adds ENERGY.*

### 6.2 H1: does R moderate the FES Delta slope?

![Figure 4-12. H1 Delta slopes](../outputs_v2/thesis_assets_v2/figures/fig4-12_h1_delta_slopes.png)

*Figure 4-12. H1: predicted FES Delta slopes at resource percentiles, with 95% CIs.*

Decision rule, fixed before fitting: COR predicts a **positive** R × Delta interaction (resources flatten the negative Delta slope). *Supported* if positive with p < 0.05; *contrary to COR* if negative with p < 0.05; *not supported* otherwise.

{{TABLE:T4-5a}}

*Table 4-5 (verdicts). OLS of the fuel-to-income ratio on R, FES Delta (growth-only), R × Delta and interview-year FE; SEs clustered on PSU (9,775 PSUs); n = 269,372.*

{{TABLE:T4-5b}}

*Table 4-5 (coefficients).*

{{TABLE:T4-5c}}

*Table 4-5 (Delta slopes at the 10th, 50th and 90th percentiles of R).*

{{TABLE:T4-5d}}

*Table 4-5 (buffering bound). At the upper confidence limit of the interaction, moving from the 10th to the 90th resource percentile flattens the Delta slope by at most 0.018 percentage points of income per SD of Delta, about 17% of the slope at the 10th percentile.*

{{TABLE:T4-5e}}

*Table 4-5 (logit footnote, probability scale). The logit interaction is negative on the log-odds scale (−0.0035, p < 0.001), yet on the probability scale the Delta effect is smaller for better-resourced households. Both patterns reflect differences in baseline risk, not buffering, and neither changes the primary verdict.*

**Verdict: H1 not supported.** Resources have a strong main effect (−0.0047 per unit of R, p < 0.001), but they do not change how the fuel burden responds to forecast–realised price stress. A substantial buffering effect is ruled out; a small one (at most 17% of the slope) cannot be excluded.

---

## 7. Comparison with JRF income poverty, and Northern Ireland (RQ4; H4)

### 7.1 Agreement by dimension

![Figure 4-13. Regions vs JRF](../outputs_v2/thesis_assets_v2/figures/fig4-13_jrf_regions.png)

*Figure 4-13. Regional fuel vulnerability vs JRF income poverty (time-matched). Pipeline version: `outputs_v2/stage5/figures/F5_1_region_vs_jrf.png`.*

![Figure 4-14. Other dimensions vs JRF](../outputs_v2/thesis_assets_v2/figures/fig4-14_jrf_other_dims.png)

*Figure 4-14. Other JRF dimensions, time-matched, with 95% CIs. Pipeline version: `outputs_v2/stage5/figures/F5_2_dimensions_vs_jrf.png`.*

| F5.1 Region vs JRF | F5.2 Other dimensions |
|---|---|
| ![](../outputs_v2/stage5/figures/F5_1_region_vs_jrf.png) | ![](../outputs_v2/stage5/figures/F5_2_dimensions_vs_jrf.png) |

{{TABLE:T4-6a}}

*Table 4-6 (comparison). UKHLS rates are weighted, time-matched to the JRF period, with 95% CIs and rank intervals from 2,000 PSU bootstrap replicates. Rank 1 = highest.*

{{TABLE:T4-6b}}

*Table 4-6 (agreement).*

- **Tenure** agrees most closely (ρ = 0.80). Outright owners are the exception: third of four on JRF income poverty (14%) but second on fuel vulnerability (13.8% [12.7, 15.0]), above private renters (11.4%; JRF 35%).
- **Disability, family type and work status** rank in the JRF order: a disabled adult 14.3% vs none 9.5%; lone parent 20.8% vs couple with children 7.8%; not in work 22.5% vs in work 7.0%. The last two compare different units (JRF children and working-age adults vs UKHLS households) and are directional only.
- **Region** does not agree (ρ = −0.10 for all 12 regions; 0.18 without NI). v1's "0.73 excluding NI" does not hold on the corrected, weighted, time-matched rates.
- **Ethnicity** is inconclusive (ρ = 0.26). Every minority-group CI is wide (e.g. Black Caribbean 17.8% [10.2, 26.6]; Bangladeshi 11.3%, n = 223).

<details><summary><b>Table A-12. Regional rates in both JRF windows and both outcomes, with rank CIs</b></summary>

{{TABLE:TA-12}}

</details>

### 7.2 Northern Ireland and heating oil

![Figure 4-15. Northern Ireland and oil](../outputs_v2/thesis_assets_v2/figures/fig4-15_ni_oil.png)

*Figure 4-15. Northern Ireland: rates by heating fuel, and the NI gap across models. Pipeline version: `outputs_v2/stage5/figures/F5_3_ni_oil.png`.*

![F5.3 NI oil, pipeline version](../outputs_v2/stage5/figures/F5_3_ni_oil.png)

{{TABLE:T5-4}}

*Thesis Table T5.4, Northern Ireland (`outputs_v2/stage5/thesis_T5_4_northern_ireland.csv`).*

{{TABLE:T4-7}}

*Table 4-7. NI-oil sequence: average marginal effects in percentage points (delta-method CIs, PSU-clustered). Models: (a) region FE; (b) + oil use; (c) + NI × oil; "rural" adds `urban_dv` (adjacent-wave filled); "observed_only" uses unfilled `urban_dv`.*

{{TABLE:TA-3-nioil}}

*NI-oil models, odds ratios of the NI, oil, NI × oil and rural terms (primary specification; from `TA-3_driver_sensitivities.csv`).*

{{TABLE:ni_oil}}

*Oil share and oil vs non-oil vulnerability within NI (`outputs_v2/jrf/ni_oil.csv`).*

{{TABLE:oil_region}}

*Oil heating by region, pooled unweighted (`outputs_v2/jrf/oil_share_by_region.csv`).*

**Reading.** NI has the highest rate in the JRF window, 14.4% [12.1, 17.0]. It ranks first in 95% of bootstrap replicates, and this holds under the S1 lower bound (13.2%) and in the wider window (14.2%). NI also has the lowest JRF income-poverty rate of the UK nations (17%). About 73% of NI households heat with oil (weighted, JRF window), against 0.3–10.6% elsewhere. Within NI, oil-heated households are at 18.2% against 8.6%.

In the driver model, the NI gap relative to the South East falls from 6.8 pp to 2.4 pp once oil use is controlled. Oil itself adds 5.8 pp (4.9 pp once rural location is also controlled). The oil penalty is no larger in NI than elsewhere (NI × oil OR 0.94, p = 0.66). Around two-thirds of NI's excess risk is accounted for by oil use; a gap of about 2.4–2.5 pp remains. These are associations; oil use is not randomly assigned.

---

## 8. Next-wave prediction (RQ5; H5)

![Figure 4-16. ROC curves](../outputs_v2/thesis_assets_v2/figures/fig4-16_roc_p0_p3.png)

*Figure 4-16. ROC curves on the validation transitions m→n and n→o (n = 19,960).*

![Figure 4-17. Calibration](../outputs_v2/thesis_assets_v2/figures/fig4-17_calibration.png)

*Figure 4-17. Calibration by decile of predicted risk, validation transitions.*

| Stage 6 ROC (pipeline) | Stage 6 calibration (pipeline) |
|---|---|
| ![](../outputs_v2/stage6/figures/roc.png) | ![](../outputs_v2/stage6/figures/calibration.png) |

{{TABLE:T4-8}}

*Table 4-8. Next-wave prediction. P0 = this wave's fuel burden (benchmark); P1 = household predictors (financial difficulty, GHQ, financial expectations, age, central heating, lone parent, large family, workless, four resource composites); P2 = P1 + growth-only FES magnitude for the household's next interview month (December vintage before that year); P3 = P0 + P1 (post-hoc, exploratory). Training a→b … l→m (n = 170,895); standardisation uses training statistics only.*

{{TABLE:delta_auc}}

*Paired PSU-bootstrap ΔAUC (`outputs_v2/stage6/delta_auc.csv`). P2 − P1 is reported as "no improvement", without significance language (author decision; its CI lies just below zero).*

{{TABLE:p3_delta}}

*Post-hoc P3 comparisons (`outputs_v2/stage6/posthoc_p3_delta_auc.csv`).*

{{TABLE:p2b}}

*Sensitivity P2b: adding FES Delta observed at wave t, on its own common sample (`outputs_v2/stage6/p2b_sensitivity.csv`).*

{{TABLE:TA-2}}

*Table A-2. AUC by validation transition.*

{{TABLE:TA-7}}

*Table A-7. Calibration slope (coefficient on logit p) and calibration-in-the-large (intercept with logit p as offset), log-odds scale.*

<details><summary><b>Model coefficients, P0–P2 (training fit)</b></summary>

{{TABLE:s6_coef}}

*`outputs_v2/stage6/coefficients.csv`. Continuous predictors are z-scored with training statistics, so their ORs are per training SD.*

</details>

<details><summary><b>Table A-8. Prediction sample flow</b></summary>

{{TABLE:TA-8}}

</details>

**Verdict: H5 partially supported.** Next-wave vulnerability can be predicted before it is observed with moderate discrimination (P1 AUC 0.739; PR-AUC 0.280 against a no-skill baseline of 0.105). But the simple benchmark is better:

- P0 has AUC 0.780, ΔAUC P1 − P0 = −0.041 [−0.053, −0.029].
- Among the 10% of households with the highest predicted risk, P0 captures 38.6% of next-wave cases (PPV 40.6%), against 30.9% for P1 (PPV 32.5%).
- The forecast adds nothing (P2 − P1 = −0.0006), nor does FES Delta at wave t (P2b).
- All models under-predict risk in the crisis-era validation waves (calibration-in-the-large 0.39–0.45). P0 is over-dispersed (slope 0.78).
- The post-hoc P3 = P0 + P1 gives no gain over P0 alone (AUC 0.781; ΔAUC 0.0008).

For early warning, the most informative signal is a household's current fuel burden.

---

## 9. Robustness and scope

{{TABLE:T4-9}}

*Table 4-9. Robustness summary across the pre-specified sensitivities.*

### 9.1 Equivalising income

![Figure 4-18. Equivalisation sensitivity](../outputs_v2/thesis_assets_v2/figures/fig4-18_equivalisation_sensitivity.png)

*Figure 4-18. Sensitivity to equivalising income only (fuel spend not equivalised). Pipeline version: `outputs_v2/stage7/figures/sensitivity_equivalised_income_only.png`.*

![Equivalisation sensitivity, pipeline version](../outputs_v2/stage7/figures/sensitivity_equivalised_income_only.png)

{{TABLE:TA-11}}

*Table A-11. Dividing income by the modified-OECD scale while leaving fuel spend unequivalised flags many more large households: 45.8% of 5+ person households change status, all into vulnerability. The primary flag stays unequivalised, like the official 10% definition.*

### 9.2 FES terciles (descriptive)

![Figure 4-19. FES tercile](../outputs_v2/thesis_assets_v2/figures/fig4-19_fes_tercile.png)

*Figure 4-19. Prevalence by FES tercile (descriptive).*

{{TABLE:fes_tercile}}

*`outputs_v2/stage7/prevalence_by_fes_tercile.csv`. Raw prevalence is flat across FES terciles (8.1 / 9.1 / 8.8% for magnitude). The small FES Delta association in Table 4-3 is not visible in these raw terciles; it is estimated net of interview year and household characteristics.*

### 9.3 Scope: v1 exploratory models

![Figure A-4. v1 exploratory models](../outputs_v2/thesis_assets_v2/figures/figA_v1_exploratory.png)

*Figure A-4. v1 exploratory models (CVAE, fuzzy c-means, one-class SVM), labelled v1. They were computed on the v1 outcome, the abandoned second-order SEM and v1 FES timing, and were not re-estimated in v2. No main-text claim rests on them ([`appendix_scope_note.md`](../outputs_v2/reports/appendix_scope_note.md)). Dropped: the vector-shift map, the hotspot tiers, regional driver models, and v1 forward-risk maps by month and region.*

![Figure A-2. Macro covariates](../outputs_v2/thesis_assets_v2/figures/figA_macro_covariates.png)

*Figure A-2. Macro covariates (descriptive; not used by the v2 core-only forecasts).*

---

## 10. Caveats

- **Identification of FES effects.** FES is national and varies only by interview year and month. With year FE, the FES Delta coefficient is identified from within-year variation across interview months (about 185 year-month clusters). It is an association.
- **Forecast quality.** The forecasts do not beat a no-change forecast significantly, and they missed the 2022 shock. FES Delta is therefore dominated by realised stress in crisis months (mean Delta in 2022 = −7.1).
- **Missing spend.** The primary outcome is complete-case. Missing spend rises to 20% of in-scope households in wave o. S1 bounds the effect of treating non-response as £0.
- **Resources** are formative indices, not validated latent measures.
- **JRF** comparisons are time-matched but differ in construct (fuel vs income poverty), in unit (households vs people, children, working-age adults) and in data source.
- **Small cells** are suppressed (counts < 10; rates on n < 100 masked in group tables). Some ethnic groups do not appear in the JRF comparison for this reason or because JRF gives no stated rate.
- **Post-hoc items** (P3; the H5 wording) are labelled in the deviation log.

## 11. What the evidence establishes

1. After correcting the outcome, fuel vulnerability in UKHLS follows the energy-price cycle: it falls through the 2010s and roughly doubles in 2022–2024 fieldwork.
2. Household financial position, above all current financial difficulty and employment security, is far more strongly associated with fuel vulnerability than forecast price stress.
3. Forecast–realised price stress has a small, robust association with vulnerability, but it neither interacts with resources (H1) nor improves next-wave prediction (H5).
4. Fuel vulnerability is not income poverty: tenure agrees, but regions do not, and Northern Ireland's heating-oil exposure is the clearest source of divergence.
5. For targeting, a household's current fuel burden is the best available predictor of its fuel vulnerability at the next wave.
