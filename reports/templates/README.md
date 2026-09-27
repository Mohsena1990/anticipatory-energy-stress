# Anticipatory Fuel Stress Watch (AFSW)

### Who Becomes Fuel Vulnerable, and When? Forecast-Conditioned Household Risk Modelling in the UK

**Status: rerun v2, analysis frozen 2026-09-26** (branch `rerun-v2`). The submitted thesis draft is tag `submitted-draft-v1`. v2 recomputes every result after an audit found errors in the v1 outcome, FES timing and several measures. The research questions (RQ1–RQ5) are unchanged. The evidence and several conclusions are not.

| Where to look | What it holds |
|---|---|
| [`analysis_plan_rerun.md`](analysis_plan_rerun.md) | Pre-specified v2 plan, amendments A1 and A6, and the full deviation log (§9) |
| [`outputs_v2/reports/v1_to_v2_change_summary.md`](outputs_v2/reports/v1_to_v2_change_summary.md) | What changed from v1 and the effect on each thesis claim |
| [`outputs_v2/results_inventory.csv`](outputs_v2/results_inventory.csv) | Every quotable number (622 rows, IDs R0001–R0622), with source file and commit |
| [`reports/`](reports/) | Outputs catalog, findings report, policy brief, journal materials, data description, methodology (all v2) |
| `outputs_v2/thesis_assets_v2/` | Thesis figures (300 dpi) and tables with thesis numbering, plus a MANIFEST. A local build, not in git; see [Thesis asset bundle](#thesis-asset-bundle) |
| `outputs/` | v1 outputs, read-only reference. Superseded by `outputs_v2/`; do not quote |

---

## Research questions

| RQ | Question | Answered in |
|---|---|---|
| RQ1 | Can a forecast of energy-price stress (FES) be attached to households without look-ahead, and how good are the forecasts? | Stage 2 |
| RQ2 | Do household resources buffer the effect of forecast-realised price stress (Conservation of Resources, H1)? | Stage 4 |
| RQ3 | What drives fuel vulnerability, and does household financial position matter more than the price environment (H2, H3)? | Stage 3 |
| RQ4 | How is fuel vulnerability distributed by region and social group, and does it match JRF income poverty (H4)? | Descriptives, Stage 5 |
| RQ5 | Can next-wave fuel vulnerability be predicted before it is observed, and does the forecast help (H5)? | Stage 6 |

**Outcome.** A household is *fuel vulnerable* when annual fuel spend is at least 10% of net household income (`high_fuel_vulnerable`). v2 builds the spend from the UKHLS questionnaire routing, treats item non-response as missing (complete-case), and reports two bounds: S1 sets non-response to £0 (a lower bound), and S2 adds households that did not report electricity.

**FES (Forecasted Energy Stress).** The sum of the z-scored forecast growth of gas, electricity and carbon prices (growth-only, 3 terms; the 4-term index with forecast uncertainty is a sensitivity). A household interviewed in calendar year Y receives the forecast made in December Y−1 for its interview month. **FES Delta** = forecast − realised stress in the month before the interview, both z-scored with past-only moments.

---

## Headline results (v2)

{{TABLE:T4-10}}

*Table 4-10. Hypothesis verdicts. H2–H4 wording is proposed and awaits the author's confirmation.*

- **Trend.** Weighted prevalence falls from 12.0% (wave a, fieldwork 2009–11) to 6.5% (wave l, 2020–22) and returns to 12.5% in wave n (2022–24) and 12.4% in wave o. By interview year the peak is 2023 (14.7%). The S1 lower bound follows the same shape.
- **Drivers (H3).** Current financial difficulty is the dominant strain term (OR 1.65 per point [1.61, 1.69]). Employment-status security is the strongest protective factor (OR 0.17 [0.15, 0.20]). The v1 strain composite is not a scale (α = 0.27), so its v1 OR of 6.79 is not reported.
- **FES (H2).** FES Delta has a small, robust association: OR 0.972 [0.962, 0.981] per unit, 0.93 per SD. Vulnerability is higher when realised stress exceeds the forecast.
- **Buffering (H1).** Not supported. The resource × Delta interaction is −0.000034 [−0.000081, 0.000014] (p = 0.16). Any buffering compatible with the data is at most 17% of the Delta slope.
- **JRF comparison (H4).** Tenure agrees (ρ = 0.80); disability, family type and work status point in the same direction; region does not agree (ρ = −0.10; 0.18 without Northern Ireland); ethnicity is inconclusive.
- **Northern Ireland.** Highest regional rate in the JRF window, 14.4% [12.1, 17.0], ranked first in 95% of bootstrap replicates, despite having the lowest income poverty. Heating-oil use accounts for about two-thirds of NI's excess risk: the NI gap vs the South East falls from 6.8 to 2.4 percentage points once oil use is controlled.
- **Prediction (H5).** Household predictors reach AUC 0.739 [0.727, 0.751] on held-out transitions, but this wave's fuel burden alone does better (AUC 0.780 [0.768, 0.792]). Adding FES gives no improvement (ΔAUC −0.0006).
- **Forecasts (RQ1).** No v2 forecast significantly beats a no-change (naive) forecast. v2 beats the seasonal-naive benchmark significantly for carbon only (Diebold–Mariano p = 0.002). The 95% prediction intervals cover 28–48% of outcomes, so the uncertainty term was dropped from the primary FES.

---

## Pipeline

![Figure 3-1. Analysis pipeline, rerun v2](outputs_v2/thesis_assets_v2/figures/fig3-1_pipeline_v2.png)

*Figure 3-1. Analysis pipeline, rerun v2.*

| Stage | What it does | Script | Outputs |
|---|---|---|---|
| Panel | Builds the 339,201-row household-wave panel from UKHLS waves a–o; attaches FES at the Dec Y−1 vintage | `src/ukhls_preprocessing.py` | `outputs_v2/ukhls_cleaned/` (row-level, never in git) |
| 1 Outcome audit | Audits fuel-expenditure codes against the questionnaire routing; sample flow; missing-spend patterns | `scripts/audit_ukhls_codes.py` | `outputs_v2/audit_fuel_codes.csv`, `outputs_v2/audit/` |
| 2 FES | Rolling core forecasts (SARIMA, Prophet, LSTM, TFT), tuned per origin; evaluation against naive and seasonal-naive benchmarks | `forecast_pipeline.py --rolling --core-only --tune-per-origin`, `scripts/stage2_fes_evaluation.py` | `outputs_v2/fes/`, `outputs_v2/fes_eval/`, `outputs_v2/figures/` |
| Descriptives | Weighted prevalence by wave, interview year, region, social group | `scripts/descriptives_v2.py` | `outputs_v2/descriptives/` |
| 3 Drivers | Logit with year FE and PSU clustering; 10 sensitivities; NI-oil sequence with AMEs | `scripts/stage3_drivers.py` | `outputs_v2/stage3/` |
| 4 Resources, H1 | Four-factor CFA (failed pre-set criteria) → formative composites; resource × FES Delta | `scripts/stage4_resources.py`, `scripts/stage4_h1.py` | `outputs_v2/resources/`, `outputs_v2/stage4/` |
| 5 JRF, NI | Time-matched, weighted comparison with JRF *UK Poverty 2025*; PSU-bootstrap CIs and ranks | `scripts/jrf_comparison_v2.py`, `scripts/stage5_jrf_thesis.py` | `outputs_v2/jrf/`, `outputs_v2/stage5/` |
| 6 Prediction | P0 benchmark, P1 household, P2 + FES on one common sample; held-out m→n, n→o | `scripts/stage6_prediction.py`, `scripts/stage6_p3_posthoc.py`, `scripts/stage6_draft.py` | `outputs_v2/stage6/` |
| 7 Scope | Equivalised-income sensitivity; descriptive refreshes; appendix note | `scripts/stage7_scope.py` | `outputs_v2/stage7/` |
| Reporting | Results inventory; thesis bundle; small-cell suppression; README and reports (tables rendered from the CSVs) | `scripts/build_results_inventory.py`, `scripts/build_thesis_assets.py`, `scripts/suppress_small_cells.py`, `scripts/build_reports.py` | `outputs_v2/results_inventory.csv`, `outputs_v2/thesis_assets_v2/`, `README.md`, `reports/` |

---

## How to run (v2)

`src/paths.py` points `OUTPUTS_DIR` at `outputs_v2/`. The raw UKHLS files (`data/raw/ukhls/*.dta`, UK Data Service SN 6614) are licensed and not in the repository.

```bash
pip install -r requirements.txt

# Stage 2 first: rolling walk-forward forecasts, core mode, re-tuned at every origin
python forecast_pipeline.py --rolling --core-only --tune-per-origin

# Panel build (reads the rolling FES table; Dec Y-1 vintage, month-matched)
python -m src.ukhls_preprocessing

# Stage 1 audit, Stage 2 evaluation, descriptives
python scripts/audit_ukhls_codes.py
python scripts/stage2_fes_evaluation.py
python scripts/descriptives_v2.py

# Stages 3-7
python scripts/stage3_drivers.py
python scripts/stage4_resources.py
python scripts/stage4_h1.py
python scripts/jrf_comparison_v2.py
python scripts/stage5_jrf_thesis.py
python scripts/stage6_prediction.py
python scripts/stage6_p3_posthoc.py        # post-hoc, exploratory
python scripts/stage6_draft.py
python scripts/stage7_scope.py

# Disclosure control, inventory, thesis bundle
python scripts/suppress_small_cells.py      # apply; --check must pass before any output commit
python scripts/build_results_inventory.py   # needs committed outputs
python scripts/build_thesis_assets.py       # writes outputs_v2/thesis_assets_v2/ and the .zip
python scripts/build_reports.py             # renders README.md and reports/01-06 from reports/templates/
```

**Disclosure control.** Row-level UKHLS files never enter git (`.gitignore` plus a local pre-commit hook that runs `suppress_small_cells.py --check`). Every tracked table has counts of 1–9 shown as `<10`, rates on fewer than 10 cases suppressed, and category rates on fewer than 100 households masked.

**Legacy v1 entry points.** `main.py` and `household_stream.py` still run the v1 architecture (COR-SEM, COR-CVAE, fuzzy c-means, one-class SVM, policy maps, v1 forward prediction). v2 does not use them. Their results are superseded; the CVAE, fuzzy and SVM outputs are kept only as a v1 appendix (Figure A-4).

---

## Results, with every figure and table

The figures below are the thesis versions from `outputs_v2/thesis_assets_v2/figures/` (300 dpi, 16 cm wide). The pipeline figures from the tracked output folders follow in [Pipeline figures](#pipeline-figures-tracked-in-git). Every table is also rendered in [`reports/02_findings_report.md`](reports/02_findings_report.md).

### Data and sample (thesis Chapter 3)

<table>
<tr>
<td align="center" width="50%"><img src="outputs_v2/thesis_assets_v2/figures/fig3-2_interview_timing.png" width="400"><br><sub>Figure 3-2. Interview timing from actual household interview dates (cells &lt; 10 masked)</sub></td>
<td align="center" width="50%"><img src="outputs_v2/thesis_assets_v2/figures/fig3-3_missing_spend_wave_mode.png" width="400"><br><sub>Figure 3-3. Fuel-spend item non-response by interview mode and wave</sub></td>
</tr>
<tr>
<td align="center" width="50%"><img src="outputs_v2/thesis_assets_v2/figures/fig3-4_distributions_outcome_fes.png" width="400"><br><sub>Figure 3-4. Distribution of the fuel-to-income ratio and of FES as attached to households</sub></td>
<td align="center" width="50%"><img src="outputs_v2/thesis_assets_v2/figures/fig3-5_regional_counts.png" width="400"><br><sub>Figure 3-5. UKHLS household-waves by region</sub></td>
</tr>
<tr>
<td align="center" width="50%" colspan="2"><img src="outputs_v2/thesis_assets_v2/figures/fig3-6_core_forecast_vars.png" width="400"><br><sub>Figure 3-6. Core forecasting series, May 2006–March 2026</sub></td>
</tr>
</table>

**Table 3-3. Sample flow, 339,201 household-waves to the analytical n.**

{{TABLE:T3-3}}

<details><summary><b>Table 3-2. Household-waves and analytical n by wave</b></summary>

{{TABLE:T3-2}}

</details>

<details><summary><b>Table 3-4. Fuel-expenditure routing and code treatment</b></summary>

{{TABLE:T3-4}}

</details>

<details><summary><b>Table 3-5. Measures: items, coding, construction and α</b></summary>

{{TABLE:T3-5}}

</details>

<details><summary><b>Table 3-6. Core forecasting series, May 2006–March 2026</b></summary>

{{TABLE:T3-6}}

</details>

<details><summary><b>Table 3-7. JRF benchmark metadata and matching windows</b></summary>

{{TABLE:T3-7}}

</details>

### Stage 2: forecasts and FES (RQ1)

<table>
<tr>
<td align="center" width="50%"><img src="outputs_v2/thesis_assets_v2/figures/fig4-1_relrmse_by_year.png" width="400"><br><sub>Figure 4-1. Forecast accuracy relative to naive and seasonal-naive benchmarks, by target year</sub></td>
<td align="center" width="50%"><img src="outputs_v2/thesis_assets_v2/figures/fig4-2_fes_growth3_forecast_vs_realised.png" width="400"><br><sub>Figure 4-2. Growth-only FES: forecast vs realised by target month, 2010–2025</sub></td>
</tr>
</table>

**Table 4-1. Forecast accuracy (relative RMSE, Diebold–Mariano p) and 95% prediction-interval coverage.** Relative RMSE below 1 means the model beats the benchmark.

{{TABLE:T4-1}}

<details><summary><b>Table A-1. MASE</b></summary>

{{TABLE:TA-1}}

</details>

<details><summary><b>Table A-5. Relative RMSE by target year</b></summary>

{{TABLE:TA-5}}

</details>

<details><summary><b>Table A-6. Diebold–Mariano tests</b></summary>

{{TABLE:TA-6}}

</details>

### Descriptives: trend, regions, social groups (RQ4)

<table>
<tr>
<td align="center" width="50%"><img src="outputs_v2/thesis_assets_v2/figures/fig4-3_trend_wave_s1band.png" width="400"><br><sub>Figure 4-3. National trend by wave, weighted, with S1 lower bound</sub></td>
<td align="center" width="50%"><img src="outputs_v2/thesis_assets_v2/figures/fig4-4_trend_vs_jrf_tracker.png" width="400"><br><sub>Figure 4-4. Trend by interview year with the JRF cost-of-living crisis window shaded</sub></td>
</tr>
<tr>
<td align="center" width="50%"><img src="outputs_v2/thesis_assets_v2/figures/fig4-5_region_map_weighted.png" width="400"><br><sub>Figure 4-5. Regional prevalence, weighted</sub></td>
<td align="center" width="50%"><img src="outputs_v2/thesis_assets_v2/figures/fig4-6_region_year_heatmap_masked.png" width="400"><br><sub>Figure 4-6. Weighted prevalence by region and interview year (cells n &lt; 100 masked)</sub></td>
</tr>
<tr>
<td align="center" width="50%"><img src="outputs_v2/thesis_assets_v2/figures/fig4-7_region_change_map.png" width="400"><br><sub>Figure 4-7. Change in weighted regional prevalence, waves a–e to k–o (navy = fall, orange = rise)</sub></td>
<td align="center" width="50%"><img src="outputs_v2/thesis_assets_v2/figures/fig4-8_resource_and_findifficulty_by_region.png" width="400"><br><sub>Figure 4-8. Weighted mean resource composite and current financial difficulty by region</sub></td>
</tr>
<tr>
<td align="center" width="50%"><img src="outputs_v2/thesis_assets_v2/figures/fig4-9_social_groups_panel_ci.png" width="400"><br><sub>Figure 4-9. Pooled weighted prevalence by social group, 95% CIs (n &lt; 100 suppressed)</sub></td>
<td align="center" width="50%"><img src="outputs_v2/thesis_assets_v2/figures/fig4-10_prepayment.png" width="400"><br><sub>Figure 4-10. Prepayment-meter use by vulnerability status</sub></td>
</tr>
</table>

<details><summary><b>Table 4-2. Pooled weighted prevalence by region and social group, 95% PSU-bootstrap CI</b></summary>

{{TABLE:T4-2}}

</details>

### Stage 3: drivers (RQ3; H2, H3)

<table>
<tr>
<td align="center" width="100%"><img src="outputs_v2/thesis_assets_v2/figures/fig4-11_driver_forest.png" width="560"><br><sub>Figure 4-11. Primary driver model, odds ratios (FES in orange)</sub></td>
</tr>
</table>

**Table 4-3. Primary driver model** (logit of `high_fuel_vulnerable`; interview-year fixed effects; SEs clustered on PSU; n = 221,877 household-waves, interviews 2010–2025; 17,209 events; McFadden R² 0.153).

{{TABLE:T4-3}}

<details><summary><b>Table 4-4. Continuous predictors ranked by |log OR per SD|</b></summary>

{{TABLE:T4-4}}

</details>

<details><summary><b>Table A-3. All driver-model specifications (OR, main model)</b></summary>

{{TABLE:TA-3}}

The full table (`TA-3_driver_sensitivities.csv`, 1,450 rows) also holds every term of the two-way-clustered and NI-oil models. They are rendered in [`reports/02_findings_report.md`](reports/02_findings_report.md).

</details>

<details><summary><b>Table A-4. Driver-model N, events and pseudo-R²</b></summary>

{{TABLE:TA-4}}

</details>

### Stage 4: resources and H1 (RQ2)

<table>
<tr>
<td align="center" width="100%"><img src="outputs_v2/thesis_assets_v2/figures/fig4-12_h1_delta_slopes.png" width="560"><br><sub>Figure 4-12. H1: predicted FES Delta slopes at resource percentiles (95% CIs)</sub></td>
</tr>
</table>

**Table 4-5. H1 verdicts** (OLS of the fuel-to-income ratio; year FE; PSU-clustered; n = 269,372). The coefficients, the slopes and the buffering bound are in the collapsed parts below.

{{TABLE:T4-5a}}

<details><summary><b>Table 4-5, continued: coefficients, Delta slopes, buffering bound, logit footnote</b></summary>

{{TABLE:T4-5b}}

{{TABLE:T4-5c}}

{{TABLE:T4-5d}}

Logit sensitivity on the probability scale:

{{TABLE:T4-5e}}

</details>

<details><summary><b>Table A-9. CFA fit (failed the pre-registered criteria)</b></summary>

{{TABLE:TA-9}}

</details>

<details><summary><b>Table A-10. CFA loadings</b></summary>

{{TABLE:TA-10}}

</details>

### Stage 5: JRF comparison and Northern Ireland (RQ4; H4)

<table>
<tr>
<td align="center" width="50%"><img src="outputs_v2/thesis_assets_v2/figures/fig4-13_jrf_regions.png" width="400"><br><sub>Figure 4-13. Regional fuel vulnerability vs JRF income poverty (time-matched)</sub></td>
<td align="center" width="50%"><img src="outputs_v2/thesis_assets_v2/figures/fig4-14_jrf_other_dims.png" width="400"><br><sub>Figure 4-14. Other JRF dimensions, time-matched, with 95% CIs</sub></td>
</tr>
<tr>
<td align="center" width="100%" colspan="2"><img src="outputs_v2/thesis_assets_v2/figures/fig4-15_ni_oil.png" width="560"><br><sub>Figure 4-15. Northern Ireland: rates by heating fuel and the NI gap across models</sub></td>
</tr>
</table>

**Table 4-7. NI-oil sequence: average marginal effects (percentage points).**

{{TABLE:T4-7}}

<details><summary><b>Table 4-6. JRF comparison with CIs, and agreement</b></summary>

{{TABLE:T4-6a}}

{{TABLE:T4-6b}}

</details>

<details><summary><b>Table A-12. Regional rates in the JRF windows, with rank CIs</b></summary>

{{TABLE:TA-12}}

</details>

### Stage 6: next-wave prediction (RQ5; H5)

<table>
<tr>
<td align="center" width="50%"><img src="outputs_v2/thesis_assets_v2/figures/fig4-16_roc_p0_p3.png" width="400"><br><sub>Figure 4-16. ROC curves, validation transitions m→n and n→o (n = 19,960)</sub></td>
<td align="center" width="50%"><img src="outputs_v2/thesis_assets_v2/figures/fig4-17_calibration.png" width="400"><br><sub>Figure 4-17. Calibration by decile of predicted risk, validation transitions</sub></td>
</tr>
</table>

**Table 4-8. Next-wave prediction, P0–P3** (training a→b … l→m, n = 170,895; validation m→n and n→o, n = 19,960; 95% CIs from 2,000 PSU bootstrap replicates; P3 is post-hoc).

{{TABLE:T4-8}}

<details><summary><b>Table A-2. Per-transition AUC</b></summary>

{{TABLE:TA-2}}

</details>

<details><summary><b>Table A-7. Calibration slope and intercept</b></summary>

{{TABLE:TA-7}}

</details>

<details><summary><b>Table A-8. Prediction sample flow</b></summary>

{{TABLE:TA-8}}

</details>

### Stage 7 and robustness

<table>
<tr>
<td align="center" width="50%"><img src="outputs_v2/thesis_assets_v2/figures/fig4-18_equivalisation_sensitivity.png" width="400"><br><sub>Figure 4-18. Sensitivity to equivalising income only (fuel spend not equivalised)</sub></td>
<td align="center" width="50%"><img src="outputs_v2/thesis_assets_v2/figures/fig4-19_fes_tercile.png" width="400"><br><sub>Figure 4-19. Prevalence by FES tercile (descriptive)</sub></td>
</tr>
</table>

<details><summary><b>Table 4-9. Robustness summary across pre-specified sensitivities</b></summary>

{{TABLE:T4-9}}

</details>

<details><summary><b>Table A-11. Sensitivity to equivalising income only</b></summary>

{{TABLE:TA-11}}

</details>

### Appendix figures and tables

<table>
<tr>
<td align="center" width="50%"><img src="outputs_v2/thesis_assets_v2/figures/figA_radial_winners.png" width="400"><br><sub>Figure A-1. Winning rolling core model by series and target year, read clockwise from 12 o'clock</sub></td>
<td align="center" width="50%"><img src="outputs_v2/thesis_assets_v2/figures/figA_macro_covariates.png" width="400"><br><sub>Figure A-2. Macro covariates (descriptive; not used by the v2 core-only forecasts)</sub></td>
</tr>
<tr>
<td align="center" width="50%"><img src="outputs_v2/thesis_assets_v2/figures/figA_trend_interview_year.png" width="400"><br><sub>Figure A-3. Trend by interview year (supplementary)</sub></td>
<td align="center" width="50%"><img src="outputs_v2/thesis_assets_v2/figures/figA_v1_exploratory.png" width="400"><br><sub>Figure A-4. v1 exploratory models (CVAE, fuzzy c-means, one-class SVM), labelled v1, not re-estimated</sub></td>
</tr>
</table>

<details><summary><b>Table A-13. Household-waves by wave × interview year (cells &lt; 10 omitted)</b></summary>

{{TABLE:TA-13}}

</details>

<details><summary><b>Table A-14. Household-waves by region</b></summary>

{{TABLE:TA-14}}

</details>

### Pipeline figures (tracked in git)

These are drawn by the analysis scripts and committed with their PDF twins. Several are earlier renderings of the thesis figures above.

<table>
<tr>
<td align="center" width="50%"><img src="outputs_v2/figures/fes_rolling_trend.png" width="400"><br><sub>Rolling walk-forward FES by target year: forecast (4-term, annual mean) vs realised. <code>outputs_v2/figures/fes_rolling_trend.png</code></sub></td>
<td align="center" width="50%"><img src="outputs_v2/figures/model_selection_polar_core.png" width="400"><br><sub>Winning core model and its validation RMSE by series and target year. <code>outputs_v2/figures/model_selection_polar_core.png</code></sub></td>
</tr>
<tr>
<td align="center" width="50%"><img src="outputs_v2/figures/rolling_forecast_performance_gas.png" width="400"><br><sub>Winning model's validation RMSE by year, gas</sub></td>
<td align="center" width="50%"><img src="outputs_v2/figures/rolling_forecast_performance_electricity.png" width="400"><br><sub>Winning model's validation RMSE by year, electricity</sub></td>
</tr>
<tr>
<td align="center" width="50%"><img src="outputs_v2/figures/rolling_forecast_performance_carbon.png" width="400"><br><sub>Winning model's validation RMSE by year, carbon</sub></td>
<td align="center" width="50%"><img src="outputs_v2/descriptives/trend_primary_with_s1_band.png" width="400"><br><sub>Trend by wave, primary with S1 band. <code>outputs_v2/descriptives/</code></sub></td>
</tr>
<tr>
<td align="center" width="50%"><img src="outputs_v2/descriptives/trend_by_interview_year_supplementary.png" width="400"><br><sub>Trend by interview year, supplementary. <code>outputs_v2/descriptives/</code></sub></td>
<td align="center" width="50%"><img src="outputs_v2/resources/figures/resource_composite_by_region.png" width="400"><br><sub>Resource composite by region (weighted means). <code>outputs_v2/resources/figures/</code></sub></td>
</tr>
<tr>
<td align="center" width="50%"><img src="outputs_v2/stage3/figures/primary_or_forest.png" width="400"><br><sub>Primary driver model, odds-ratio forest plot. <code>outputs_v2/stage3/figures/</code></sub></td>
<td align="center" width="50%"><img src="outputs_v2/stage5/figures/F5_1_region_vs_jrf.png" width="400"><br><sub>F5.1 Region vs JRF. <code>outputs_v2/stage5/figures/</code></sub></td>
</tr>
<tr>
<td align="center" width="50%"><img src="outputs_v2/stage5/figures/F5_2_dimensions_vs_jrf.png" width="400"><br><sub>F5.2 Other dimensions vs JRF</sub></td>
<td align="center" width="50%"><img src="outputs_v2/stage5/figures/F5_3_ni_oil.png" width="400"><br><sub>F5.3 Northern Ireland and heating oil</sub></td>
</tr>
<tr>
<td align="center" width="50%"><img src="outputs_v2/stage6/figures/roc.png" width="400"><br><sub>Stage 6 ROC curves (P0, P1, P2). <code>outputs_v2/stage6/figures/</code></sub></td>
<td align="center" width="50%"><img src="outputs_v2/stage6/figures/calibration.png" width="400"><br><sub>Stage 6 calibration</sub></td>
</tr>
<tr>
<td align="center" width="50%"><img src="outputs_v2/stage7/figures/sensitivity_equivalised_income_only.png" width="400"><br><sub>Sensitivity to equivalising income only. <code>outputs_v2/stage7/figures/</code></sub></td>
<td align="center" width="50%"></td>
</tr>
</table>

---

## Thesis asset bundle

`python scripts/build_thesis_assets.py` writes `outputs_v2/thesis_assets_v2/` (29 figures, 30 tables, 7 documents and `MANIFEST.md`) and `outputs_v2/thesis_assets_v2.zip`. The MANIFEST records each file's thesis number, source script, source files and their commits. All CSVs in the bundle pass the suppression check.

The bundle is a local build product and is **ignored by git** (`.gitignore`). The thesis figures embedded in this README and in `reports/` therefore display only after the bundle has been built. Five figures (3-2, 3-4, 3-5, 4-9, and the financial-difficulty panel of 4-8) and three tables (4-2, A-13, A-14) aggregate the row-level panel inside the build script, so rebuilding them needs the licensed UKHLS data.

---

## Directory structure

```
anticipatory-energy-stress/
├── analysis_plan_rerun.md       # v2 plan, amendments, deviation log
├── forecast_pipeline.py         # Stage 2 forecasts (--rolling --core-only --tune-per-origin)
├── main.py, household_stream.py # v1 orchestrators (legacy; not used by v2)
├── scripts/                     # v2 analysis, one script per stage (see Pipeline)
├── src/
│   ├── paths.py                 # OUTPUTS_DIR = outputs_v2, V1_OUTPUTS_DIR = outputs
│   ├── ukhls_preprocessing.py   # panel build, routing-aware outcome, FES attachment
│   ├── ukhls_mapping.py         # wave/variable registry, recodes
│   ├── fes_calculator.py        # FES construction (rolling helpers)
│   ├── models/                  # sarima_model.py, prophet_model.py, lstm_model.py, tft_model.py
│   └── ...                      # data_loader, tuning, model_evaluation, plotting_utils, v1 modules
├── data/                        # raw and processed series; data/raw/ukhls/ is licensed and ignored
├── outputs_v2/                  # v2 results (aggregate tables tracked; row-level files ignored)
│   ├── audit/, audit_fuel_codes.csv      # Stage 1
│   ├── fes/, fes_eval/, figures/, models/, tables/rolling/, forecasts_rolling/, tuning_rolling/   # Stage 2
│   ├── descriptives/                      # trend and group prevalence
│   ├── stage3/, resources/, stage4/       # drivers, resources, H1
│   ├── jrf/, stage5/                      # JRF comparison, NI
│   ├── stage6/, stage7/                   # prediction, scope
│   ├── reports/                           # v2 drafts, change summary, appendix note
│   ├── results_inventory.csv, logs/
│   └── thesis_assets_v2/                  # local build (ignored)
├── outputs/                     # v1 outputs, read-only
└── reports/                     # 01-06 reports (v2), generated from reports/templates/ by scripts/build_reports.py
```

The file-by-file description of every output is in [`reports/01_outputs_catalog.md`](reports/01_outputs_catalog.md).

---

## Limitations

- **Associations, not causal effects.** FES varies by interview year and month only; with year fixed effects the FES Delta coefficient comes from within-year variation across interview months.
- **Forecast skill is weak.** No v2 forecast beats the naive benchmark significantly; prediction intervals are too narrow (coverage 28–48%).
- **Resources are formative indices.** The CFA failed the pre-registered criteria (CFI 0.78, a Heywood case, low loadings) and its complete-case sample is 99% owner-occupiers, so the composites describe resource levels rather than measure latent constructs. α (0.32–0.60) is descriptive only.
- **Complete-case outcome.** Missing fuel spend among in-scope households rises from 7–8% (waves b–e) to 20% (wave o), and is higher in telephone interviews (Figure 3-3). S1 (non-response as £0) is reported as a lower bound throughout.
- **The 10% threshold uses unequivalised income.** Equivalising income alone would flag many more large households (45.8% of 5+ person households change status, all into vulnerability).
- **JRF comparisons differ in unit and construct.** Family type (children) and work status (working-age adults) are directional only. Ethnicity CIs are too wide for conclusions.
- **Prediction** is evaluated on two held-out transitions during the cost-of-living crisis; all models under-predict risk there (calibration-in-the-large 0.39–0.45).
- **Post-hoc items are labelled.** P3 (P0 + P1) was added after Stage 6 results were seen and is exploratory.

---

## Data and licence

Understanding Society (UKHLS), UK Data Service SN 6614, End User Licence: raw and row-level data are not redistributed. JRF *UK Poverty 2025* (January 2025): values are taken from the report text and tables with page references (`outputs_v2/jrf/jrf_metadata.csv`). Price series: ONS (Open Government Licence v3.0) and market futures data. Region boundaries: ONS Open Geography Portal (OGL v3.0). Full provenance is in [`reports/05_data_description.md`](reports/05_data_description.md).
