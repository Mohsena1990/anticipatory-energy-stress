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

| Hypothesis | Verdict | Status | Key statistics |
|---|---|---|---|
| H1 Resource moderation (COR buffering) | Not supported | author-confirmed | R × Delta = -0.000034 (p = 0.16); buffering ≤ 17% of slope |
| H2 FES independent predictor | Supported (small effect) | PROPOSED — author to confirm | FES Delta OR 0.972 [0.962, 0.981]; robust across sensitivities |
| H3 Household financial position dominates macro stress | Supported (reframed: current financial difficulty and employment security) | PROPOSED — author to confirm | financial difficulty OR 1.65 per point; employment security OR 0.17; FES OR per SD 0.93 |
| H4 External validation with JRF | Partially supported | PROPOSED — author to confirm | tenure ρ 0.80; family, work, disability same direction; region ρ -0.10 (excl. NI 0.18); ethnicity inconclusive |
| H5 Prospective prediction | Partially supported | author-confirmed | P1 AUC 0.739; benchmark P0 0.780; FES: no improvement |

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

| Step | Description | Change / total | Remaining |
|---|---|---|---|
| 0 | All UKHLS household-wave rows (waves a-o) | 339,201 |  |
| 1 | Fuel-use module nonresponse (fuelhave* < 0) | -1,326 | 337,875 |
| 2 | No fuel reported (fuelhave96 / none mentioned) | -1,182 | 336,693 |
| 3 | Electricity not reported: gas only [S2 adds back] | -4,278 | 332,415 |
| 4 | Electricity not reported: oil/other only [S2 adds back] | -4,003 | 328,412 |
| 5a | Item nonresponse, first missing amount = xpduely [S1 adds back as 0] | -16,684 | 311,728 |
| 5b | Item nonresponse, first missing amount = xpgasy [S1 adds back as 0] | -15,619 | 296,109 |
| 5c | Item nonresponse, first missing amount = xpelecy [S1 adds back as 0] | -6,284 | 289,825 |
| 5d | Item nonresponse, first missing amount = xpoily [S1 adds back as 0] | -256 | 289,569 |
| 5e | Item nonresponse, first missing amount = xpsfly [S1 adds back as 0] | -221 | 289,348 |
| 6 | Household net income missing (sentinel code) | -13 | 289,335 |
| 7 | Annual net income < £1,200 guard (never logged in v1) | -2,433 | 286,902 |
| = | Primary analytical n (fuel_to_income_ratio non-missing) | 286,902 |  |
| info | of which ratio capped at 1.0 (kept, not excluded) | 2,106 |  |
| info | v1 analytical n for comparison | 255,324 |  |
| info | S1 (lower-bound) analytical n | 325,190 |  |
| info | S2 (+elec-not-reported) analytical n | 294,310 |  |
| v1.0 | All rows | 339,201 |  |
| v1.1 | fuelduel = -8 inapplicable (not dual-fuel) -> dropped | -61,392 |  |
| v1.2 | fuelduel DK/refused/missing -> dropped | -3,691 |  |
| v1.3 | fuelduel = 1 and xpduely nonresponse -> dropped | -16,684 |  |
| v1.4 | = rows with v1 spend | 257,434 |  |
| v1.5 | income missing | -10 |  |
| v1.6 | income < £1,200 guard (not logged in v1) | -2,100 |  |
| v1.7 | = v1 analytical n | 255,324 |  |

<details><summary><b>Table 3-2. Household-waves and analytical n by wave</b></summary>

| Wave | Fieldwork | Household-waves | Analytical n, primary | Analytical n, S1 lower bound | Analytical n, v1 rule |
|---|---|---|---|---|---|
| a | 2009–2011 | 30,169 | 25,649 | 28,881 | 23,440 |
| b | 2010–2012 | 30,484 | 26,936 | 29,307 | 23,609 |
| c | 2011–2013 | 27,751 | 24,765 | 26,945 | 21,694 |
| d | 2012–2014 | 25,817 | 23,260 | 25,180 | 20,382 |
| e | 2013–2015 | 24,325 | 21,970 | 23,653 | 19,244 |
| f | 2014–2016 | 24,454 | 20,625 | 23,458 | 18,732 |
| g | 2015–2017 | 23,033 | 19,814 | 22,205 | 17,847 |
| h | 2016–2018 | 21,746 | 18,818 | 20,938 | 16,776 |
| i | 2017–2019 | 20,048 | 17,004 | 19,298 | 15,149 |
| j | 2018–2020 | 19,252 | 16,171 | 18,473 | 14,282 |
| k | 2019–2021 | 18,139 | 14,884 | 17,268 | 13,167 |
| l | 2020–2022 | 16,856 | 13,431 | 15,912 | 11,900 |
| m | 2021–2023 | 16,156 | 12,478 | 15,152 | 11,168 |
| n | 2022–2024 | 21,385 | 16,411 | 20,072 | 14,752 |
| o | 2023–2025 | 19,586 | 14,686 | 18,448 | 13,182 |
| total |  | 339,201 | 286,902 | 325,190 | 255,324 |

</details>

<details><summary><b>Table 3-4. Fuel-expenditure routing and code treatment</b></summary>

| Variable | Asked if | Content | v2 treatment |
|---|---|---|---|
| fuelhave1–4 | all households | fuels used (electricity, gas, oil, other) | defines which amounts are required |
| fuelduel | electricity AND gas used | 1 one bill / 2 separate | −8 = not dual-fuel (not missing); DK/refused → separate amounts asked |
| xpduely | fuelduel = 1 | annual combined gas+electricity £ | −1/−2/−9 = item non-response → spend missing |
| xpgasy, xpelecy | fuelduel = 2 or DK/refused, or single-fuel household | annual £ | −8 = fuel not used (structural 0); −1/−2/−9 → missing |
| xpoily | oil used (fuelhave3 = 1) | annual £ | −8 = not used (0); non-response → missing |
| xpsfly | other fuel used (fuelhave4 = 1) | annual £ | −8 = not used (0); non-response → missing |

</details>

<details><summary><b>Table 3-5. Measures: items, coding, construction and α</b></summary>

| Construct | Items | Coding | Construction | Cronbach α (descriptive) |
|---|---|---|---|---|
| Outcome | high_fuel_vulnerable | annual fuel spend / (12 × monthly net income) ≥ 0.10 | routing-aware spend, complete-case; income < £1,200 excluded; ratio capped at 1 |  |
| Strain | finnow | current financial situation, 1 comfortable … 5 very difficult | household mean of adults | 0.27 (3-item composite; not used as a scale) |
| Strain | scghq1_dv | GHQ-12 Likert 0–36 (higher = more distress) | household mean |  |
| Strain | finfut_risk | financial expectations: 0 better / 0.5 same / 1 worse | household mean; always with age |  |
| Resources: OBJECT | hsrooms, hsbeds, ncars, carval, hsval | log(1+x) for £ items; z-scored | mean of z (≥ 50% observed), re-standardised | 0.60 |
| Resources: CONDITION | tenure_security, jbstat_security, bill_security | 0–1 security codings | as above | 0.34 |
| Resources: PERSONAL | sf1_good (self-rated health), health_good (no long-standing illness), qfhigh_band | higher = better | as above | 0.58 |
| Resources: ENERGY | fihhmnnet1_dv, fiyrinvinc_dv | log(1+x) | as above | 0.32 |
| Disability | health + disdif1–12 | long-standing illness and ≥ 1 substantial difficulty | household: any observed adult |  |
| Oil use | fuelhave3 | 1 = uses heating oil |  |  |
| Rural | urban_dv | 2 = rural | missing filled from adjacent wave if no move |  |
| FES | fes_magnitude_growth3, fes_delta_growth3 | sum of 3 growth z-scores (past-only moments); Delta = forecast − realised (m−1) | Dec Y−1 vintage; interviews 2010+ |  |

</details>

<details><summary><b>Table 3-6. Core forecasting series, May 2006–March 2026</b></summary>

| Series | n | Missing | Mean | SD | Median | Skew | Excess kurtosis |
|---|---|---|---|---|---|---|---|
| gas_growth | 239 | 0 | 9.39 | 29.76 | 1.40 | 2.42 | 7.01 |
| electricity_index | 239 | 0 | 119.01 | 47.80 | 100.20 | 1.10 | 0.09 |
| electricity_growth | 239 | 0 | 7.80 | 15.49 | 5.70 | 1.85 | 4.85 |
| carbon_growth | 239 | 0 | 5.98 | 149.33 | 3.78 | 0.28 | 8.87 |

</details>

<details><summary><b>Table 3-7. JRF benchmark metadata and matching windows</b></summary>

| Dimension | Window | JRF population | JRF measure | JRF period | JRF source | UKHLS window | UKHLS unit | UKHLS definition | Waves | Categories | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| region | primary | People (all ages) | Relative poverty, AHC | '2021–2023': HBAI '3-year' average of FY 2021/22 and 2022/23 only (DWP excludes 2020/21) | Table 6, p.51; exclusion of 2020/21: note p.43, Annex p.162 | interviews 2021-04 to 2023-03 | Household (gor_dv) | Region of household | k,l,m,n,o | 12 | JRF 'East' = East of England. Sensitivity window adds Apr 2020-Mar 2021. |
| region | sensitivity | People (all ages) | Relative poverty, AHC | '2021–2023': HBAI '3-year' average of FY 2021/22 and 2022/23 only (DWP excludes 2020/21) | Table 6, p.51; exclusion of 2020/21: note p.43, Annex p.162 | interviews 2020-04 to 2023-03 | Household (gor_dv) | Region of household | j,k,l,m,n,o | 12 | JRF 'East' = East of England. Sensitivity window adds Apr 2020-Mar 2021. |
| ethnicity | primary | People in households, by ethnicity of household head | Relative poverty, AHC | FY 2021/22 and 2022/23 (text says 2020/21-2022/23; note p.43: 2020/21 excluded) | p.9 and p.42 (text); Figure 13 and note, p.43; Annex p.162 | interviews 2021-04 to 2023-03 | Household (ethnicity of household reference person) | ethnicity_group of HRP | k,l,m,n,o | 6 | Only categories with a rate stated in JRF text are compared. |
| ethnicity | sensitivity | People in households, by ethnicity of household head | Relative poverty, AHC | FY 2021/22 and 2022/23 (text says 2020/21-2022/23; note p.43: 2020/21 excluded) | p.9 and p.42 (text); Figure 13 and note, p.43; Annex p.162 | interviews 2020-04 to 2023-03 | Household (ethnicity of household reference person) | ethnicity_group of HRP | j,k,l,m,n,o | 6 | Only categories with a rate stated in JRF text are compared. |
| tenure | primary | People | Relative poverty, AHC | FY 2022/23 | Table 10, p.95 | interviews 2022-04 to 2023-03 | Household (tenure_dv) | Social = LA + housing association; private incl. rented from employer; 'Other' tenure excluded | l,m,n,o | 4 |  |
| disability | primary | People, by disability mix of family | Relative poverty, AHC | FY 2022/23 | Table 8, p.67 | interviews 2022-04 to 2023-03 | Household with >=1 adult disability status observed | Disabled = health==1 and any disdif1-12; household contains a disabled adult if any observed adult is disabled | l,m,n,o | 2 | JRF 'Disabled adults only' (29) vs 'No one is disabled' (19). UKHLS does not observe child disability, so JRF's child rows (28, 36) are not compared. Directional (n=2). |
| family_type | primary | CHILDREN, by family type | Child relative poverty, AHC | FY 2022/23 | Table 5, p.36 | interviews 2022-04 to 2023-03 | Household with dependent children | family_composition_group lone parent (any size) vs couple (any size); other multi-adult excluded | l,m,n,o | 2 | Unit mismatch: JRF rate is per child, UKHLS rate per household. Directional (n=2). |
| work_status | primary | WORKING-AGE ADULTS, by household work status | Relative poverty, AHC | FY 2022/23 (latest year in report) | p.77 (text) | interviews 2022-04 to 2023-03 | Household with >=1 respondent aged 16-64 | Workless = no responding adult in paid/self-employment | l,m,n,o | 2 | Corrected from v1 (12/43). Unit mismatch: JRF per working-age adult, UKHLS per household. Directional (n=2). |

</details>

### Stage 2: forecasts and FES (RQ1)

<table>
<tr>
<td align="center" width="50%"><img src="outputs_v2/thesis_assets_v2/figures/fig4-1_relrmse_by_year.png" width="400"><br><sub>Figure 4-1. Forecast accuracy relative to naive and seasonal-naive benchmarks, by target year</sub></td>
<td align="center" width="50%"><img src="outputs_v2/thesis_assets_v2/figures/fig4-2_fes_growth3_forecast_vs_realised.png" width="400"><br><sub>Figure 4-2. Growth-only FES: forecast vs realised by target month, 2010–2025</sub></td>
</tr>
</table>

**Table 4-1. Forecast accuracy (relative RMSE, Diebold–Mariano p) and 95% prediction-interval coverage.** Relative RMSE below 1 means the model beats the benchmark.

| Series | Version | RMSE | Rel. RMSE vs naive | DM p vs naive | Rel. RMSE vs seasonal naive | DM p vs seasonal naive | 95% PI coverage (%) |
|---|---|---|---|---|---|---|---|
| carbon | v1_core | 48.29 | 1.028 | 0.845 | 0.788 | 0.054 | 28.6 |
| carbon | v1_macro | 55.47 | 1.180 | 0.403 | 0.905 | 0.551 | 37.5 |
| carbon | v2_core | 46.47 | 0.989 | 0.905 | 0.758 | 0.002 | 27.6 |
| electricity | v1_core | 16.99 | 0.973 | 0.899 | 0.740 | 0.200 | 60.4 |
| electricity | v1_macro | 13.97 | 0.800 | 0.458 | 0.609 | 0.140 | 49.5 |
| electricity | v2_core | 14.41 | 0.825 | 0.526 | 0.627 | 0.143 | 48.4 |
| gas | v1_core | 43.32 | 1.228 | 0.172 | 0.956 | 0.710 | 56.2 |
| gas | v1_macro | 55.79 | 1.582 | 0.202 | 1.231 | 0.424 | 68.8 |
| gas | v2_core | 25.11 | 0.712 | 0.373 | 0.554 | 0.130 | 32.3 |

<details><summary><b>Table A-1. MASE</b></summary>

| Series | Version | MAE | MASE |
|---|---|---|---|
| carbon | naive | 38.44 | 1.020 |
| carbon | snaive | 51.78 | 1.390 |
| carbon | v1_core | 36.43 | 0.914 |
| carbon | v1_macro | 38.89 | 0.940 |
| carbon | v2_core | 37.16 | 0.960 |
| electricity | naive | 9.19 | 5.693 |
| electricity | snaive | 14.46 | 8.684 |
| electricity | v1_core | 11.22 | 6.749 |
| electricity | v1_macro | 8.60 | 5.291 |
| electricity | v2_core | 9.06 | 5.556 |
| gas | naive | 15.74 | 6.326 |
| gas | snaive | 26.05 | 9.817 |
| gas | v1_core | 20.41 | 7.970 |
| gas | v1_macro | 24.75 | 9.555 |
| gas | v2_core | 13.50 | 5.419 |

</details>

<details><summary><b>Table A-5. Relative RMSE by target year</b></summary>

| Series | Target year | v1 core vs naive | v1 core vs s-naive | v1 macro vs naive | v1 macro vs s-naive | v2 core vs naive | v2 core vs s-naive |
|---|---|---|---|---|---|---|---|
| carbon | 2010 | 1.157 | 0.605 | 0.764 | 0.400 | 1.159 | 0.607 |
| carbon | 2011 | 2.588 | 2.721 | 3.265 | 3.432 | 0.905 | 0.952 |
| carbon | 2012 | 0.509 | 0.241 | 0.752 | 0.356 | 1.424 | 0.674 |
| carbon | 2013 | 0.864 | 1.664 | 1.338 | 2.576 | 1.091 | 2.100 |
| carbon | 2014 | 0.838 | 0.584 | 0.994 | 0.692 | 0.901 | 0.628 |
| carbon | 2015 | 1.070 | 0.829 | 0.866 | 0.672 | 0.567 | 0.440 |
| carbon | 2016 | 1.104 | 0.894 | 0.991 | 0.803 | 0.982 | 0.796 |
| carbon | 2017 | 0.645 | 0.443 | 0.807 | 0.554 | 1.475 | 1.014 |
| carbon | 2018 | 1.247 | 1.091 | 1.576 | 1.379 | 1.289 | 1.128 |
| carbon | 2019 | 0.398 | 0.412 | 0.182 | 0.189 | 0.543 | 0.562 |
| carbon | 2020 | 1.227 | 0.344 | 0.904 | 0.254 | 1.118 | 0.314 |
| carbon | 2021 | 1.000 | 0.640 | 0.522 | 0.334 | 0.710 | 0.454 |
| carbon | 2022 | 0.395 | 0.398 | 0.474 | 0.477 | 0.576 | 0.580 |
| carbon | 2023 | 1.352 | 0.346 | 1.122 | 0.287 | 1.126 | 0.288 |
| carbon | 2024 | 1.144 | 0.827 | 0.752 | 0.543 | 1.247 | 0.901 |
| carbon | 2025 | 0.658 | 0.360 | 0.978 | 0.535 | 0.571 | 0.313 |
| electricity | 2010 | 1.847 | 0.721 | 1.148 | 0.448 | 1.086 | 0.424 |
| electricity | 2011 | 1.038 | 0.813 | 1.093 | 0.856 | 1.046 | 0.819 |
| electricity | 2012 | 0.338 | 0.328 | 0.649 | 0.631 | 0.701 | 0.681 |
| electricity | 2013 | 0.481 | 0.319 | 0.724 | 0.480 | 1.952 | 1.295 |
| electricity | 2014 | 0.962 | 0.941 | 1.162 | 1.137 | 0.623 | 0.610 |
| electricity | 2015 | 12.015 | 0.862 | 11.426 | 0.819 | 9.804 | 0.703 |
| electricity | 2016 | 42.620 | 15.713 | 44.338 | 16.346 | 16.869 | 6.219 |
| electricity | 2017 | 0.830 | 0.828 | 0.897 | 0.896 | 1.170 | 1.168 |
| electricity | 2018 | 0.741 | 0.417 | 0.631 | 0.355 | 0.870 | 0.490 |
| electricity | 2019 | 1.034 | 0.789 | 0.913 | 0.697 | 0.921 | 0.703 |
| electricity | 2020 | 1.629 | 0.894 | 1.645 | 0.903 | 1.629 | 0.894 |
| electricity | 2021 | 0.791 | 0.737 | 0.909 | 0.846 | 0.892 | 0.831 |
| electricity | 2022 | 1.279 | 1.016 | 1.086 | 0.863 | 1.221 | 0.970 |
| electricity | 2023 | 0.552 | 0.550 | 0.563 | 0.561 | 0.333 | 0.332 |
| electricity | 2024 | 4.225 | 0.537 | 2.368 | 0.301 | 4.285 | 0.545 |
| electricity | 2025 | 2.513 | 1.296 | 0.805 | 0.415 | 0.678 | 0.350 |
| gas | 2010 | 10.387 | 0.572 | 7.159 | 0.395 | 1.438 | 0.079 |
| gas | 2011 | 1.392 | 1.128 | 1.175 | 0.952 | 1.232 | 0.998 |
| gas | 2012 | 0.943 | 0.646 | 1.476 | 1.011 | 0.498 | 0.341 |
| gas | 2013 | 1.623 | 0.543 | 0.745 | 0.249 | 4.146 | 1.386 |
| gas | 2014 | 0.846 | 0.817 | 0.847 | 0.818 | 0.913 | 0.882 |
| gas | 2015 | 2.879 | 1.008 | 2.896 | 1.014 | 3.190 | 1.117 |
| gas | 2016 | 3.377 | 1.318 | 4.876 | 1.904 | 3.377 | 1.318 |
| gas | 2017 | 1.075 | 0.819 | 0.585 | 0.446 | 1.052 | 0.802 |
| gas | 2018 | 0.410 | 0.359 | 1.155 | 1.012 | 0.536 | 0.470 |
| gas | 2019 | 0.873 | 0.980 | 0.657 | 0.738 | 0.569 | 0.639 |
| gas | 2020 | 1.700 | 0.814 | 1.503 | 0.720 | 1.463 | 0.701 |
| gas | 2021 | 1.024 | 1.002 | 0.768 | 0.752 | 1.029 | 1.007 |
| gas | 2022 | 1.143 | 0.888 | 1.215 | 0.944 | 1.178 | 0.915 |
| gas | 2023 | 1.259 | 1.286 | 1.707 | 1.743 | 0.298 | 0.304 |
| gas | 2024 | 1.561 | 0.230 | 2.171 | 0.321 | 1.507 | 0.223 |
| gas | 2025 | 0.747 | 0.355 | 2.440 | 1.160 | 0.490 | 0.233 |

</details>

<details><summary><b>Table A-6. Diebold–Mariano tests</b></summary>

| Version | Series | Benchmark | Months | DM statistic | p | Verdict |
|---|---|---|---|---|---|---|
| v1_core | carbon | naive | 192 | 0.196 | 0.845 | not significantly different |
| v1_core | carbon | seasonal naive | 192 | -1.939 | 0.054 | not significantly different |
| v1_core | electricity | naive | 192 | -0.127 | 0.899 | not significantly different |
| v1_core | electricity | seasonal naive | 192 | -1.286 | 0.200 | not significantly different |
| v1_core | gas | naive | 192 | 1.369 | 0.173 | not significantly different |
| v1_core | gas | seasonal naive | 192 | -0.372 | 0.710 | not significantly different |
| v1_macro | carbon | naive | 192 | 0.838 | 0.403 | not significantly different |
| v1_macro | carbon | seasonal naive | 192 | -0.597 | 0.551 | not significantly different |
| v1_macro | electricity | naive | 192 | -0.743 | 0.458 | not significantly different |
| v1_macro | electricity | seasonal naive | 192 | -1.483 | 0.140 | not significantly different |
| v1_macro | gas | naive | 192 | 1.279 | 0.202 | not significantly different |
| v1_macro | gas | seasonal naive | 192 | 0.800 | 0.425 | not significantly different |
| v2_core | carbon | naive | 192 | -0.119 | 0.905 | not significantly different |
| v2_core | carbon | seasonal naive | 192 | -3.185 | 0.002 | better than benchmark |
| v2_core | electricity | naive | 192 | -0.635 | 0.526 | not significantly different |
| v2_core | electricity | seasonal naive | 192 | -1.469 | 0.143 | not significantly different |
| v2_core | gas | naive | 192 | -0.893 | 0.373 | not significantly different |
| v2_core | gas | seasonal naive | 192 | -1.521 | 0.130 | not significantly different |

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

| Dimension | Category | n | Weighted % [95% CI] |
|---|---|---|---|
| region | East Midlands | 21,104 | 9.4 [8.5, 10.2] |
| region | East of England | 24,393 | 7.3 [6.6, 8.0] |
| region | London | 33,423 | 6.7 [6.1, 7.3] |
| region | North East | 11,085 | 9.0 [8.0, 9.9] |
| region | North West | 29,472 | 9.1 [8.4, 9.7] |
| region | Northern Ireland | 16,646 | 18.0 [16.8, 19.4] |
| region | Scotland | 25,951 | 10.3 [9.5, 11.1] |
| region | South East | 33,992 | 6.6 [6.1, 7.1] |
| region | South West | 23,661 | 7.4 [6.8, 8.1] |
| region | Wales | 19,274 | 10.4 [9.5, 11.3] |
| region | West Midlands | 23,778 | 9.9 [9.0, 10.7] |
| region | Yorkshire and the Humber | 23,905 | 9.1 [8.3, 9.9] |
| tenure | Buying with mortgage | 98,150 | 4.2 [4.0, 4.5] |
| tenure | Owned outright | 101,166 | 11.1 [10.8, 11.6] |
| tenure | Private renting | 35,200 | 8.9 [8.3, 9.5] |
| tenure | Social renting | 50,611 | 11.1 [10.6, 11.6] |
| family_composition_group | Couple, 1-2 children | 47,733 | 3.8 [3.5, 4.0] |
| family_composition_group | Couple, 3+ children | 9,748 | 5.2 [4.5, 6.1] |
| family_composition_group | Lone parent, 1-2 children | 14,579 | 15.1 [14.1, 16.2] |
| family_composition_group | Lone parent, 3+ children | 1,929 | 13.6 [11.2, 16.4] |
| family_composition_group | No children | 192,368 | 9.7 [9.5, 10.0] |
| family_composition_group | Other multi-adult, with children | 20,374 | 3.8 [3.5, 4.2] |
| employment_group | Full-time or self-employed | 151,626 | 3.5 [3.3, 3.6] |
| employment_group | Part-time only | 29,607 | 10.0 [9.5, 10.6] |
| employment_group | Workless household | 104,093 | 15.0 [14.6, 15.5] |
| ethnicity_group | Any other Asian background | 2,144 | 6.2 [4.6, 8.0] |
| ethnicity_group | Any other Black background | 349 | 14.6 [9.5, 19.9] |
| ethnicity_group | Bangladeshi | 3,246 | 9.3 [7.5, 11.3] |
| ethnicity_group | Black African | 5,508 | 11.2 [9.7, 12.7] |
| ethnicity_group | Black Caribbean | 5,615 | 14.7 [12.6, 17.2] |
| ethnicity_group | Chinese | 1,099 | 7.5 [4.6, 11.3] |
| ethnicity_group | Indian | 7,646 | 8.0 [6.8, 9.5] |
| ethnicity_group | Mixed/multiple ethnic groups | 3,989 | 9.9 [8.0, 11.9] |
| ethnicity_group | Other ethnic group | 1,872 | 10.7 [7.2, 15.0] |
| ethnicity_group | Pakistani | 5,584 | 13.8 [12.2, 15.5] |
| ethnicity_group | White | 229,098 | 8.8 [8.6, 9.1] |
| disability | Contains disabled adult | 98,015 | 10.5 [10.2, 10.9] |
| disability | No disabled adult | 186,994 | 7.6 [7.4, 7.9] |

</details>

### Stage 3: drivers (RQ3; H2, H3)

<table>
<tr>
<td align="center" width="100%"><img src="outputs_v2/thesis_assets_v2/figures/fig4-11_driver_forest.png" width="560"><br><sub>Figure 4-11. Primary driver model, odds ratios (FES in orange)</sub></td>
</tr>
</table>

**Table 4-3. Primary driver model** (logit of `high_fuel_vulnerable`; interview-year fixed effects; SEs clustered on PSU; n = 221,877 household-waves, interviews 2010–2025; 17,209 events; McFadden R² 0.153).

| Predictor | Term | OR [95% CI] | p | OR per SD | SD in sample |
|---|---|---|---|---|---|
| Current financial difficulty (1-5) | `finnow` | 1.649 [1.607, 1.691] | <0.001 | 1.595 | 0.934 |
| Bedrooms | `hsbeds` | 1.335 [1.285, 1.386] | <0.001 | 1.341 | 1.015 |
| Tenure security | `tenure_security` | 2.034 [1.767, 2.340] | <0.001 | 1.186 | 0.240 |
| Rooms | `hsrooms` | 1.149 [1.106, 1.195] | <0.001 | 1.158 | 1.054 |
| No long-standing illness/disability | `health_good` | 1.253 [1.185, 1.325] | <0.001 | 1.097 | 0.412 |
| Age (mean of adults) | `dvage` | 1.003 [1.001, 1.005] | 0.013 | 1.050 | 16.703 |
| Self-rated general health | `sf1_good` | 1.026 [0.918, 1.147] | 0.649 | 1.006 | 0.235 |
| Psychological distress, GHQ-12 (0-36) | `scghq1_dv` | 0.991 [0.987, 0.995] | <0.001 | 0.956 | 4.976 |
| Financial expectations: worse off (0-1) | `finfut_risk` | 0.818 [0.762, 0.879] | <0.001 | 0.946 | 0.278 |
| FES Delta (growth-only) | `fes_delta_growth3` | 0.972 [0.962, 0.981] | <0.001 | 0.932 | 2.459 |
| Bill-payment security | `bill_security` | 0.587 [0.518, 0.664] | <0.001 | 0.931 | 0.134 |
| Cars | `ncars` | 0.856 [0.824, 0.889] | <0.001 | 0.848 | 1.057 |
| Highest qualification band | `qfhigh_band` | 0.548 [0.509, 0.590] | <0.001 | 0.802 | 0.368 |
| Employment-status security | `jbstat_security` | 0.174 [0.154, 0.197] | <0.001 | 0.630 | 0.264 |
| OECD equivalence scale | `ieqmoecd_dv` | 0.275 [0.255, 0.297] | <0.001 | 0.468 | 0.589 |
| Has central heating | `heatch` | 1.087 [1.013, 1.167] | 0.020 |  |  |
| Lone-parent household | `lone_parent` | 1.499 [1.379, 1.629] | <0.001 |  |  |
| Large family (3+ children) | `large_family` | 1.963 [1.738, 2.218] | <0.001 |  |  |
| Workless household | `workless_household` | 1.046 [0.957, 1.144] | 0.318 |  |  |

<details><summary><b>Table 4-4. Continuous predictors ranked by |log OR per SD|</b></summary>

| Predictor | SD | OR per SD [95% CI] | log OR per SD | p | Note |
|---|---|---|---|---|---|
| OECD equivalence scale | 0.589 | 0.468 [0.447, 0.489] | -0.760 | <0.001 | partly mechanical: outcome uses unequivalised income, so larger households have more income per fuel need |
| Current financial difficulty (1-5) | 0.934 | 1.595 [1.557, 1.633] | 0.467 | <0.001 |  |
| Employment-status security | 0.264 | 0.630 [0.609, 0.651] | -0.462 | <0.001 |  |
| Bedrooms | 1.015 | 1.341 [1.290, 1.393] | 0.293 | <0.001 |  |
| Highest qualification band | 0.368 | 0.802 [0.780, 0.824] | -0.221 | <0.001 |  |
| Tenure security | 0.240 | 1.186 [1.147, 1.227] | 0.171 | <0.001 |  |
| Cars | 1.057 | 0.848 [0.815, 0.883] | -0.164 | <0.001 |  |
| Rooms | 1.054 | 1.158 [1.112, 1.206] | 0.147 | <0.001 |  |
| No long-standing illness/disability | 0.412 | 1.097 [1.072, 1.123] | 0.093 | <0.001 |  |
| Bill-payment security | 0.134 | 0.931 [0.916, 0.947] | -0.071 | <0.001 |  |
| FES Delta (growth-only) | 2.459 | 0.932 [0.910, 0.954] | -0.071 | <0.001 |  |
| Financial expectations: worse off (0-1) | 0.278 | 0.946 [0.927, 0.965] | -0.056 | <0.001 |  |
| Age (mean of adults) | 16.703 | 1.050 [1.010, 1.091] | 0.049 | 0.013 |  |
| Psychological distress, GHQ-12 (0-36) | 4.976 | 0.956 [0.936, 0.976] | -0.045 | <0.001 |  |
| Self-rated general health | 0.235 | 1.006 [0.980, 1.033] | 0.006 | 0.649 |  |

</details>

<details><summary><b>Table A-3. All driver-model specifications (OR, main model)</b></summary>

| Predictor (OR, main model) | primary | sens_composite | sens_composite_v1 | sens_lagged_components | sens_lagged_composite | sens_lagged_composite_v1 | sens_month_fe | sens_fes_4term | sens_outcome_s1 | sens_outcome_s2 | sens_no_qualification |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Tenure security | 2.034 | 1.639 | 1.624 | 2.038 | 1.696 | 1.723 | 2.036 | 2.034 | 2.101 | 2.016 | 1.908 |
| Employment-status security | 0.174 | 0.153 | 0.152 | 0.145 | 0.146 | 0.147 | 0.174 | 0.174 | 0.178 | 0.178 | 0.163 |
| No long-standing illness/disability | 1.253 | 1.293 | 1.293 | 1.269 | 1.306 | 1.304 | 1.250 | 1.253 | 1.241 | 1.257 | 1.257 |
| Self-rated general health | 1.026 | 0.980 | 0.959 | 0.937 | 0.899 | 0.881 | 1.029 | 1.026 | 1.046 | 1.027 | 0.906 |
| Highest qualification band | 0.548 | 0.522 | 0.521 | 0.550 | 0.523 | 0.523 | 0.549 | 0.548 | 0.553 | 0.555 | — |
| Age (mean of adults) | 1.003 | 0.999 | 0.999 | 1.002 | 0.999 | 1.000 | 1.003 | 1.003 | 1.004 | 1.003 | 1.007 |
| Has central heating | 1.087 | 1.102 | 1.103 | 1.065 | 1.072 | 1.074 | 1.087 | 1.087 | 1.083 | 1.041 | 1.108 |
| Bedrooms | 1.335 | 1.315 | 1.315 | 1.300 | 1.296 | 1.296 | 1.335 | 1.335 | 1.305 | 1.327 | 1.320 |
| Rooms | 1.149 | 1.127 | 1.127 | 1.152 | 1.129 | 1.127 | 1.148 | 1.149 | 1.146 | 1.150 | 1.123 |
| Cars | 0.856 | 0.821 | 0.818 | 0.848 | 0.816 | 0.818 | 0.855 | 0.856 | 0.866 | 0.857 | 0.836 |
| Bill-payment security | 0.587 | 0.415 | 0.694 | 0.436 | 0.360 | 0.417 | 0.585 | 0.587 | 0.573 | 0.583 | 0.605 |
| Lone-parent household | 1.499 | 1.531 | 1.533 | 1.505 | 1.543 | 1.533 | 1.501 | 1.499 | 1.600 | 1.503 | 1.544 |
| Large family (3+ children) | 1.963 | 1.926 | 1.930 | 2.092 | 2.005 | 2.014 | 1.967 | 1.963 | 1.971 | 1.964 | 1.982 |
| OECD equivalence scale | 0.275 | 0.280 | 0.280 | 0.261 | 0.270 | 0.269 | 0.275 | 0.275 | 0.284 | 0.276 | 0.281 |
| Workless household | 1.046 | 0.948 | 0.947 | 0.937 | 0.899 | 0.898 | 1.047 | 1.046 | 1.021 | 1.055 | 1.038 |
| Current financial difficulty (1-5) | 1.649 | — | — | — | — | — | 1.648 | 1.649 | 1.639 | 1.646 | 1.674 |
| Psychological distress, GHQ-12 (0-36) | 0.991 | — | — | — | — | — | 0.991 | 0.991 | 0.990 | 0.991 | 0.989 |
| Financial expectations: worse off (0-1) | 0.818 | — | — | — | — | — | 0.818 | 0.818 | 0.843 | 0.819 | 0.837 |
| FES Delta (growth-only) | 0.972 | 0.973 | 0.973 | 0.972 | 0.973 | 0.973 | 0.968 | — | 0.974 | 0.972 | 0.973 |
| Strain composite (no bill arrears) | — | 4.756 | — | — | — | — | — | — | — | — | — |
| Strain composite, v1 (incl. bill arrears) | — | — | 7.515 | — | — | — | — | — | — | — | — |
| Current financial difficulty, previous wave | — | — | — | 1.502 | — | — | — | — | — | — | — |
| GHQ-12 distress, previous wave | — | — | — | 0.989 | — | — | — | — | — | — | — |
| Financial expectations, previous wave | — | — | — | 0.994 | — | — | — | — | — | — | — |
| Strain composite, previous wave | — | — | — | — | 4.052 | — | — | — | — | — | — |
| Strain composite v1, previous wave | — | — | — | — | — | 5.393 | — | — | — | — | — |
| FES Delta (4-term) | — | — | — | — | — | — | — | 0.970 | — | — | — |
| n | 221,877 | 233,980 | 234,102 | 183,078 | 194,151 | 194,692 | 221,877 | 221,877 | 249,200 | 227,007 | 248,802 |

The full table (`TA-3_driver_sensitivities.csv`, 1,450 rows) also holds every term of the two-way-clustered and NI-oil models. They are rendered in [`reports/02_findings_report.md`](reports/02_findings_report.md).

</details>

<details><summary><b>Table A-4. Driver-model N, events and pseudo-R²</b></summary>

| Specification | Model | n | Events | PSUs | Interview years | McFadden R² | AIC | Clustering | Converged |
|---|---|---|---|---|---|---|---|---|---|
| primary | `main` | 221,877 | 17,209 | 8,801 | 2010-2025 | 0.1531 | 102,581 | psu | True |
| primary | `main_twoway_cluster` | 221,877 | 17,209 | 8,801 | 2010-2025 | 0.1531 | 102,581 | psu x interview year-month | True |
| primary | `ni_a_regionFE_filled` | 221,778 | 17,197 | 8,801 | 2010-2025 | 0.1586 | 101,869 | psu | True |
| primary | `ni_b_oil_filled` | 221,778 | 17,197 | 8,801 | 2010-2025 | 0.1626 | 101,388 | psu | True |
| primary | `ni_b_oil_rural_filled` | 221,778 | 17,197 | 8,801 | 2010-2025 | 0.1631 | 101,336 | psu | True |
| primary | `ni_c_ni_x_oil_filled` | 221,778 | 17,197 | 8,801 | 2010-2025 | 0.1626 | 101,390 | psu | True |
| primary | `ni_c_ni_x_oil_rural_filled` | 221,778 | 17,197 | 8,801 | 2010-2025 | 0.1631 | 101,338 | psu | True |
| primary | `ni_b_oil_observed_only` | 221,611 | 17,176 | 8,800 | 2010-2025 | 0.1625 | 101,295 | psu | True |
| primary | `ni_b_oil_rural_observed_only` | 221,611 | 17,176 | 8,800 | 2010-2025 | 0.1629 | 101,243 | psu | True |
| primary | `ni_c_ni_x_oil_observed_only` | 221,611 | 17,176 | 8,800 | 2010-2025 | 0.1625 | 101,297 | psu | True |
| primary | `ni_c_ni_x_oil_rural_observed_only` | 221,611 | 17,176 | 8,800 | 2010-2025 | 0.1629 | 101,245 | psu | True |
| sens_composite | `main` | 233,980 | 19,022 | 9,224 | 2010-2025 | 0.1408 | 113,425 | psu | True |
| sens_composite | `ni_a_regionFE_filled` | 233,873 | 19,008 | 9,223 | 2010-2025 | 0.1459 | 112,697 | psu | True |
| sens_composite | `ni_b_oil_filled` | 233,873 | 19,008 | 9,223 | 2010-2025 | 0.1499 | 112,171 | psu | True |
| sens_composite | `ni_b_oil_rural_filled` | 233,873 | 19,008 | 9,223 | 2010-2025 | 0.1503 | 112,120 | psu | True |
| sens_composite | `ni_c_ni_x_oil_filled` | 233,873 | 19,008 | 9,223 | 2010-2025 | 0.1499 | 112,173 | psu | True |
| sens_composite | `ni_c_ni_x_oil_rural_filled` | 233,873 | 19,008 | 9,223 | 2010-2025 | 0.1503 | 112,122 | psu | True |
| sens_composite_v1 | `main` | 234,102 | 19,055 | 9,233 | 2010-2025 | 0.1403 | 113,637 | psu | True |
| sens_composite_v1 | `ni_a_regionFE_filled` | 233,995 | 19,041 | 9,232 | 2010-2025 | 0.1455 | 112,909 | psu | True |
| sens_composite_v1 | `ni_b_oil_filled` | 233,995 | 19,041 | 9,232 | 2010-2025 | 0.1495 | 112,381 | psu | True |
| sens_composite_v1 | `ni_b_oil_rural_filled` | 233,995 | 19,041 | 9,232 | 2010-2025 | 0.1499 | 112,331 | psu | True |
| sens_composite_v1 | `ni_c_ni_x_oil_filled` | 233,995 | 19,041 | 9,232 | 2010-2025 | 0.1495 | 112,383 | psu | True |
| sens_composite_v1 | `ni_c_ni_x_oil_rural_filled` | 233,995 | 19,041 | 9,232 | 2010-2025 | 0.1499 | 112,333 | psu | True |
| sens_lagged_components | `main` | 183,078 | 13,518 | 8,130 | 2010-2025 | 0.1456 | 82,494 | psu | True |
| sens_lagged_components | `ni_a_regionFE_filled` | 183,013 | 13,507 | 8,130 | 2010-2025 | 0.1512 | 81,913 | psu | True |
| sens_lagged_components | `ni_b_oil_filled` | 183,013 | 13,507 | 8,130 | 2010-2025 | 0.1559 | 81,465 | psu | True |
| sens_lagged_components | `ni_b_oil_rural_filled` | 183,013 | 13,507 | 8,130 | 2010-2025 | 0.1565 | 81,406 | psu | True |
| sens_lagged_components | `ni_c_ni_x_oil_filled` | 183,013 | 13,507 | 8,130 | 2010-2025 | 0.1559 | 81,467 | psu | True |
| sens_lagged_components | `ni_c_ni_x_oil_rural_filled` | 183,013 | 13,507 | 8,130 | 2010-2025 | 0.1565 | 81,407 | psu | True |
| sens_lagged_composite | `main` | 194,151 | 15,018 | 8,407 | 2010-2025 | 0.1380 | 91,190 | psu | True |
| sens_lagged_composite | `ni_a_regionFE_filled` | 194,079 | 15,006 | 8,406 | 2010-2025 | 0.1438 | 90,540 | psu | True |
| sens_lagged_composite | `ni_b_oil_filled` | 194,079 | 15,006 | 8,406 | 2010-2025 | 0.1484 | 90,060 | psu | True |
| sens_lagged_composite | `ni_b_oil_rural_filled` | 194,079 | 15,006 | 8,406 | 2010-2025 | 0.1490 | 90,001 | psu | True |
| sens_lagged_composite | `ni_c_ni_x_oil_filled` | 194,079 | 15,006 | 8,406 | 2010-2025 | 0.1484 | 90,062 | psu | True |
| sens_lagged_composite | `ni_c_ni_x_oil_rural_filled` | 194,079 | 15,006 | 8,406 | 2010-2025 | 0.1490 | 90,003 | psu | True |
| sens_lagged_composite_v1 | `main` | 194,692 | 15,067 | 8,416 | 2010-2025 | 0.1379 | 91,486 | psu | True |
| sens_lagged_composite_v1 | `ni_a_regionFE_filled` | 194,620 | 15,055 | 8,415 | 2010-2025 | 0.1438 | 90,824 | psu | True |
| sens_lagged_composite_v1 | `ni_b_oil_filled` | 194,620 | 15,055 | 8,415 | 2010-2025 | 0.1483 | 90,346 | psu | True |
| sens_lagged_composite_v1 | `ni_b_oil_rural_filled` | 194,620 | 15,055 | 8,415 | 2010-2025 | 0.1489 | 90,287 | psu | True |
| sens_lagged_composite_v1 | `ni_c_ni_x_oil_filled` | 194,620 | 15,055 | 8,415 | 2010-2025 | 0.1483 | 90,348 | psu | True |
| sens_lagged_composite_v1 | `ni_c_ni_x_oil_rural_filled` | 194,620 | 15,055 | 8,415 | 2010-2025 | 0.1489 | 90,289 | psu | True |
| sens_month_fe | `main` | 221,877 | 17,209 | 8,801 | 2010-2025 | 0.1538 | 102,519 | psu | True |
| sens_month_fe | `ni_a_regionFE_filled` | 221,778 | 17,197 | 8,801 | 2010-2025 | 0.1593 | 101,807 | psu | True |
| sens_month_fe | `ni_b_oil_filled` | 221,778 | 17,197 | 8,801 | 2010-2025 | 0.1633 | 101,328 | psu | True |
| sens_month_fe | `ni_b_oil_rural_filled` | 221,778 | 17,197 | 8,801 | 2010-2025 | 0.1637 | 101,278 | psu | True |
| sens_month_fe | `ni_c_ni_x_oil_filled` | 221,778 | 17,197 | 8,801 | 2010-2025 | 0.1633 | 101,330 | psu | True |
| sens_month_fe | `ni_c_ni_x_oil_rural_filled` | 221,778 | 17,197 | 8,801 | 2010-2025 | 0.1637 | 101,280 | psu | True |
| sens_fes_4term | `main` | 221,877 | 17,209 | 8,801 | 2010-2025 | 0.1531 | 102,576 | psu | True |
| sens_fes_4term | `ni_a_regionFE_filled` | 221,778 | 17,197 | 8,801 | 2010-2025 | 0.1587 | 101,864 | psu | True |
| sens_fes_4term | `ni_b_oil_filled` | 221,778 | 17,197 | 8,801 | 2010-2025 | 0.1627 | 101,384 | psu | True |
| sens_fes_4term | `ni_b_oil_rural_filled` | 221,778 | 17,197 | 8,801 | 2010-2025 | 0.1631 | 101,332 | psu | True |
| sens_fes_4term | `ni_c_ni_x_oil_filled` | 221,778 | 17,197 | 8,801 | 2010-2025 | 0.1627 | 101,385 | psu | True |
| sens_fes_4term | `ni_c_ni_x_oil_rural_filled` | 221,778 | 17,197 | 8,801 | 2010-2025 | 0.1631 | 101,334 | psu | True |
| sens_outcome_s1 | `main` | 249,200 | 17,320 | 8,950 | 2010-2025 | 0.1480 | 107,228 | psu | True |
| sens_outcome_s1 | `ni_a_regionFE_filled` | 249,076 | 17,308 | 8,950 | 2010-2025 | 0.1544 | 106,370 | psu | True |
| sens_outcome_s1 | `ni_b_oil_filled` | 249,076 | 17,308 | 8,950 | 2010-2025 | 0.1588 | 105,829 | psu | True |
| sens_outcome_s1 | `ni_b_oil_rural_filled` | 249,076 | 17,308 | 8,950 | 2010-2025 | 0.1592 | 105,777 | psu | True |
| sens_outcome_s1 | `ni_c_ni_x_oil_filled` | 249,076 | 17,308 | 8,950 | 2010-2025 | 0.1588 | 105,829 | psu | True |
| sens_outcome_s1 | `ni_c_ni_x_oil_rural_filled` | 249,076 | 17,308 | 8,950 | 2010-2025 | 0.1592 | 105,779 | psu | True |
| sens_outcome_s2 | `main` | 227,007 | 17,406 | 8,868 | 2010-2025 | 0.1521 | 104,233 | psu | True |
| sens_outcome_s2 | `ni_a_regionFE_filled` | 226,907 | 17,394 | 8,868 | 2010-2025 | 0.1563 | 103,673 | psu | True |
| sens_outcome_s2 | `ni_b_oil_filled` | 226,907 | 17,394 | 8,868 | 2010-2025 | 0.1600 | 103,223 | psu | True |
| sens_outcome_s2 | `ni_b_oil_rural_filled` | 226,907 | 17,394 | 8,868 | 2010-2025 | 0.1604 | 103,172 | psu | True |
| sens_outcome_s2 | `ni_c_ni_x_oil_filled` | 226,907 | 17,394 | 8,868 | 2010-2025 | 0.1600 | 103,225 | psu | True |
| sens_outcome_s2 | `ni_c_ni_x_oil_rural_filled` | 226,907 | 17,394 | 8,868 | 2010-2025 | 0.1604 | 103,174 | psu | True |
| sens_no_qualification | `main` | 248,802 | 20,416 | 9,291 | 2010-2025 | 0.1484 | 120,317 | psu | True |
| sens_no_qualification | `ni_a_regionFE_filled` | 248,687 | 20,400 | 9,291 | 2010-2025 | 0.1571 | 119,029 | psu | True |
| sens_no_qualification | `ni_b_oil_filled` | 248,687 | 20,400 | 9,291 | 2010-2025 | 0.1614 | 118,417 | psu | True |
| sens_no_qualification | `ni_b_oil_rural_filled` | 248,687 | 20,400 | 9,291 | 2010-2025 | 0.1622 | 118,315 | psu | True |
| sens_no_qualification | `ni_c_ni_x_oil_filled` | 248,687 | 20,400 | 9,291 | 2010-2025 | 0.1615 | 118,415 | psu | True |
| sens_no_qualification | `ni_c_ni_x_oil_rural_filled` | 248,687 | 20,400 | 9,291 | 2010-2025 | 0.1622 | 118,315 | psu | True |

</details>

### Stage 4: resources and H1 (RQ2)

<table>
<tr>
<td align="center" width="100%"><img src="outputs_v2/thesis_assets_v2/figures/fig4-12_h1_delta_slopes.png" width="560"><br><sub>Figure 4-12. H1: predicted FES Delta slopes at resource percentiles (95% CIs)</sub></td>
</tr>
</table>

**Table 4-5. H1 verdicts** (OLS of the fuel-to-income ratio; year FE; PSU-clustered; n = 269,372). The coefficients, the slopes and the buffering bound are in the collapsed parts below.

| Model | R × Delta interaction | p | Verdict | Governs H1 | n |
|---|---|---|---|---|---|
| primary | -0.0000338 | 0.163 | not supported | True | 269,372 |
| sens_R_with_energy | 0.0000180 | 0.400 | not supported | False | 269,372 |
| sens_logit_binary | -0.0035192 | <0.001 | contrary to COR | False | 269,372 |
| sens_fes_4term | -0.0000070 | 0.755 | not supported | False | 269,372 |

<details><summary><b>Table 4-5, continued: coefficients, Delta slopes, buffering bound, logit footnote</b></summary>

| Model | Estimator | Outcome | Term | Coefficient [95% CI] | SE | p | R² / pseudo-R² | PSUs |
|---|---|---|---|---|---|---|---|---|
| primary | ols | fuel_to_income_ratio | `R_primary` | -0.004678 [-0.004851, -0.004506] | 0.000088 | <0.001 | 0.0446 | 9,775 |
| primary | ols | fuel_to_income_ratio | `fes_delta_growth3` | -0.000545 [-0.000693, -0.000398] | 0.000075 | <0.001 | 0.0446 | 9,775 |
| primary | ols | fuel_to_income_ratio | `R_primary x fes_delta_growth3` | -0.000034 [-0.000081, 0.000014] | 0.000024 | 0.163 | 0.0446 | 9,775 |
| sens_R_with_energy | ols | fuel_to_income_ratio | `R_with_energy` | -0.005366 [-0.005505, -0.005227] | 0.000071 | <0.001 | 0.0810 | 9,775 |
| sens_R_with_energy | ols | fuel_to_income_ratio | `fes_delta_growth3` | -0.000547 [-0.000696, -0.000397] | 0.000076 | <0.001 | 0.0810 | 9,775 |
| sens_R_with_energy | ols | fuel_to_income_ratio | `R_with_energy x fes_delta_growth3` | 0.000018 [-0.000024, 0.000060] | 0.000021 | 0.400 | 0.0810 | 9,775 |
| sens_logit_binary | logit | high_fuel_vulnerable | `R_primary` | -0.240102 [-0.249174, -0.231029] | 0.004629 | <0.001 | 0.0511 | 9,775 |
| sens_logit_binary | logit | high_fuel_vulnerable | `fes_delta_growth3` | -0.031140 [-0.039455, -0.022825] | 0.004242 | <0.001 | 0.0511 | 9,775 |
| sens_logit_binary | logit | high_fuel_vulnerable | `R_primary x fes_delta_growth3` | -0.003519 [-0.005608, -0.001430] | 0.001066 | <0.001 | 0.0511 | 9,775 |
| sens_fes_4term | ols | fuel_to_income_ratio | `R_primary` | -0.004666 [-0.004840, -0.004492] | 0.000089 | <0.001 | 0.0446 | 9,775 |
| sens_fes_4term | ols | fuel_to_income_ratio | `fes_delta` | -0.000564 [-0.000707, -0.000421] | 0.000073 | <0.001 | 0.0446 | 9,775 |
| sens_fes_4term | ols | fuel_to_income_ratio | `R_primary x fes_delta` | -0.000007 [-0.000051, 0.000037] | 0.000022 | 0.755 | 0.0446 | 9,775 |

| Model | R percentile | R value | Delta slope [95% CI] | Scale |
|---|---|---|---|---|
| primary | p10 | -2.959 | -0.000446 [-0.000672, -0.000219] | fuel-to-income ratio per unit Delta |
| primary | p50 | 0.329 | -0.000557 [-0.000701, -0.000412] | fuel-to-income ratio per unit Delta |
| primary | p90 | 2.506 | -0.000630 [-0.000795, -0.000465] | fuel-to-income ratio per unit Delta |
| sens_R_with_energy | p10 | -3.614 | -0.000612 [-0.000876, -0.000347] | fuel-to-income ratio per unit Delta |
| sens_R_with_energy | p50 | 0.344 | -0.000540 [-0.000683, -0.000398] | fuel-to-income ratio per unit Delta |
| sens_R_with_energy | p90 | 3.366 | -0.000486 [-0.000626, -0.000346] | fuel-to-income ratio per unit Delta |
| sens_logit_binary | p10 | -2.959 | -0.020728 [-0.030373, -0.011083] | log-odds per unit Delta |
| sens_logit_binary | p50 | 0.329 | -0.032299 [-0.040738, -0.023861] | log-odds per unit Delta |
| sens_logit_binary | p90 | 2.506 | -0.039960 [-0.050385, -0.029535] | log-odds per unit Delta |
| sens_fes_4term | p10 | -2.959 | -0.000543 [-0.000758, -0.000328] | fuel-to-income ratio per unit Delta |
| sens_fes_4term | p50 | 0.329 | -0.000566 [-0.000706, -0.000426] | fuel-to-income ratio per unit Delta |
| sens_fes_4term | p90 | 2.506 | -0.000581 [-0.000740, -0.000423] | fuel-to-income ratio per unit Delta |

| Interaction value | b (R × Delta) | R p10 → p90 | SD of Delta | Slope change p10→p90 per SD Delta (ratio units) | Same, pp of income | Slope at p10 per SD Delta (pp) | Change as % of p10 slope (+ = flatter, i.e. buffering) |
|---|---|---|---|---|---|---|---|
| point estimate | -0.0000338 | -2.96 → 2.51 | 2.375 | -0.000438 | -0.044 | -0.106 | -41% |
| CI lower | -0.0000812 | -2.96 → 2.51 | 2.375 | -0.001053 | -0.105 | -0.106 | -100% |
| CI upper (max buffering) | 0.0000136 | -2.96 → 2.51 | 2.375 | 0.000177 | 0.018 | -0.106 | +17% |

Logit sensitivity on the probability scale:

| R percentile | R value | Delta effect, pp per SD [95% CI] |
|---|---|---|
| p10 | -2.959 | -0.61 [-0.89, -0.33] |
| p50 | 0.329 | -0.52 [-0.65, -0.38] |
| p90 | 2.506 | -0.41 [-0.51, -0.30] |

</details>

<details><summary><b>Table A-9. CFA fit (failed the pre-registered criteria)</b></summary>

| n (complete case) | χ² | df | CFI | TLI | RMSEA | SRMR | Pre-registered criteria |
|---|---|---|---|---|---|---|---|
| 143,770 | 51,361.8 | 59 | 0.783 | 0.714 | 0.078 | 0.066 | CFI ≥ 0.90, RMSEA ≤ 0.08, SRMR ≤ 0.08, no Heywood case, all std. loadings ≥ 0.30 → **failed** |

</details>

<details><summary><b>Table A-10. CFA loadings</b></summary>

| Factor | Item | Label | Estimate | Std. estimate | SE | p |
|---|---|---|---|---|---|---|
| OBJECT | `hsrooms` | Rooms | 1.000 | 0.544 | fixed (marker) | — |
| OBJECT | `hsbeds` | Bedrooms | 1.230 | 0.669 | 0.0087 | <0.001 |
| OBJECT | `ncars` | Cars | 0.736 | 0.400 | 0.0068 | <0.001 |
| OBJECT | `carval` | Car value (log) | 0.598 | 0.325 | 0.0065 | <0.001 |
| OBJECT | `hsval` | House value (log) | 0.938 | 0.510 | 0.0074 | <0.001 |
| CONDITION | `tenure_security` | Tenure security | 1.000 | 0.512 | fixed (marker) | — |
| CONDITION | `jbstat_security` | Employment-status security | -1.450 | -0.743 | 0.0141 | <0.001 |
| CONDITION | `bill_security` | Bill-payment security | -0.058 | -0.030 | 0.0063 | <0.001 |
| PERSONAL | `sf1_good` | Self-rated general health | 1.000 | 0.720 | fixed (marker) | — |
| PERSONAL | `health_good` | No long-standing illness/disability | 0.903 | 0.651 | 0.0075 | <0.001 |
| PERSONAL | `qfhigh_band` | Highest qualification band | 0.358 | 0.258 | 0.0048 | <0.001 |
| ENERGY | `fihhmnnet1_dv` | Net household income (log) | 1.000 | 1.000 | fixed (marker) | — |
| ENERGY | `fiyrinvinc_dv` | Investment income (log) | 0.141 | 0.141 | 0.0058 | <0.001 |

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

| Model | n | Contrast | AME, pp [95% CI] | SE (pp) |
|---|---|---|---|---|
| `ni_a_regionFE_filled` | 221,778 | NI vs South East | 6.82 [5.76, 7.88] | 0.54 |
| `ni_b_oil_filled` | 221,778 | NI vs South East | 2.39 [1.40, 3.37] | 0.50 |
| `ni_b_oil_filled` | 221,778 | oil vs no oil | 5.84 [4.88, 6.79] | 0.49 |
| `ni_b_oil_rural_filled` | 221,778 | NI vs South East | 2.51 [1.52, 3.51] | 0.51 |
| `ni_b_oil_rural_filled` | 221,778 | oil vs no oil | 4.93 [3.96, 5.89] | 0.49 |
| `ni_c_ni_x_oil_filled` | 221,778 | NI vs South East | 2.63 [1.22, 4.03] | 0.72 |
| `ni_c_ni_x_oil_filled` | 221,778 | oil vs no oil | 5.90 [4.89, 6.92] | 0.52 |
| `ni_c_ni_x_oil_rural_filled` | 221,778 | NI vs South East | 2.59 [1.19, 3.99] | 0.72 |
| `ni_c_ni_x_oil_rural_filled` | 221,778 | oil vs no oil | 4.95 [3.92, 5.97] | 0.52 |
| `ni_b_oil_observed_only` | 221,611 | NI vs South East | 2.39 [1.40, 3.38] | 0.50 |
| `ni_b_oil_observed_only` | 221,611 | oil vs no oil | 5.83 [4.87, 6.79] | 0.49 |
| `ni_b_oil_rural_observed_only` | 221,611 | NI vs South East | 2.52 [1.53, 3.52] | 0.51 |
| `ni_b_oil_rural_observed_only` | 221,611 | oil vs no oil | 4.92 [3.95, 5.88] | 0.49 |
| `ni_c_ni_x_oil_observed_only` | 221,611 | NI vs South East | 2.65 [1.23, 4.06] | 0.72 |
| `ni_c_ni_x_oil_observed_only` | 221,611 | oil vs no oil | 5.90 [4.88, 6.92] | 0.52 |
| `ni_c_ni_x_oil_rural_observed_only` | 221,611 | NI vs South East | 2.62 [1.20, 4.03] | 0.72 |
| `ni_c_ni_x_oil_rural_observed_only` | 221,611 | oil vs no oil | 4.94 [3.92, 5.97] | 0.52 |

<details><summary><b>Table 4-6. JRF comparison with CIs, and agreement</b></summary>

| Dimension | Window | Category | JRF % (rank) | n | Fuel-vulnerable % [95% CI] | Rank [95% CI] | P(rank 1) |
|---|---|---|---|---|---|---|---|
| region | primary (2021-04 to 2023-03) | North East | 21 (5) | 1,047 | 8.8 [6.5, 11.3] | 7 [3–12] | 0.001 |
| region | primary (2021-04 to 2023-03) | North West | 25 (2) | 2,757 | 7.9 [6.6, 9.4] | 9 [6–12] | 0.000 |
| region | primary (2021-04 to 2023-03) | Yorkshire and the Humber | 23 (4) | 2,367 | 9.5 [7.7, 11.3] | 6 [3–10] | 0.000 |
| region | primary (2021-04 to 2023-03) | East Midlands | 20 (8) | 1,996 | 10.5 [8.6, 12.5] | 5 [2–8] | 0.004 |
| region | primary (2021-04 to 2023-03) | West Midlands | 27 (1) | 2,203 | 10.6 [8.6, 12.6] | 4 [2–8] | 0.004 |
| region | primary (2021-04 to 2023-03) | East of England | 18 (11) | 2,377 | 8.6 [7.2, 10.1] | 8 [5–11] | 0.000 |
| region | primary (2021-04 to 2023-03) | London | 24 (3) | 2,676 | 6.7 [5.0, 8.7] | 12 [8–12] | 0.000 |
| region | primary (2021-04 to 2023-03) | South East | 19 (9) | 3,433 | 7.4 [6.2, 8.7] | 11 [7–12] | 0.000 |
| region | primary (2021-04 to 2023-03) | South West | 19 (9) | 2,343 | 7.8 [6.4, 9.3] | 10 [6–12] | 0.000 |
| region | primary (2021-04 to 2023-03) | Wales | 21 (5) | 1,691 | 11.4 [9.4, 13.6] | 2 [1–6] | 0.027 |
| region | primary (2021-04 to 2023-03) | Scotland | 21 (5) | 2,677 | 11.2 [9.4, 13.2] | 3 [2–6] | 0.018 |
| region | primary (2021-04 to 2023-03) | Northern Ireland | 17 (12) | 1,562 | 14.4 [12.1, 17.0] | 1 [1–2] | 0.946 |
| region | sensitivity (2020-04 to 2023-03) | North East | 21 (5) | 1,607 | 8.1 [6.3, 10.1] | 7 [3–11] | 0.000 |
| region | sensitivity (2020-04 to 2023-03) | North West | 25 (2) | 4,189 | 7.1 [6.1, 8.2] | 9 [6–12] | 0.000 |
| region | sensitivity (2020-04 to 2023-03) | Yorkshire and the Humber | 23 (4) | 3,614 | 8.4 [7.0, 9.8] | 6 [3–9] | 0.000 |
| region | sensitivity (2020-04 to 2023-03) | East Midlands | 20 (8) | 3,094 | 9.2 [7.6, 11.0] | 5 [2–8] | 0.000 |
| region | sensitivity (2020-04 to 2023-03) | West Midlands | 27 (1) | 3,438 | 9.4 [7.9, 10.8] | 4 [2–7] | 0.000 |
| region | sensitivity (2020-04 to 2023-03) | East of England | 18 (11) | 3,635 | 7.7 [6.4, 9.1] | 8 [4–11] | 0.000 |
| region | sensitivity (2020-04 to 2023-03) | London | 24 (3) | 4,311 | 6.3 [4.8, 8.0] | 12 [7–12] | 0.000 |
| region | sensitivity (2020-04 to 2023-03) | South East | 19 (9) | 5,235 | 6.5 [5.6, 7.5] | 11 [8–12] | 0.000 |
| region | sensitivity (2020-04 to 2023-03) | South West | 19 (9) | 3,596 | 7.0 [5.9, 8.2] | 10 [6–12] | 0.000 |
| region | sensitivity (2020-04 to 2023-03) | Wales | 21 (5) | 2,643 | 9.4 [7.9, 11.1] | 3 [2–7] | 0.000 |
| region | sensitivity (2020-04 to 2023-03) | Scotland | 21 (5) | 4,017 | 10.3 [8.9, 11.8] | 2 [2–5] | 0.002 |
| region | sensitivity (2020-04 to 2023-03) | Northern Ireland | 17 (12) | 2,305 | 14.2 [12.1, 16.4] | 1 [1–1] | 0.999 |
| ethnicity | primary (2021-04 to 2023-03) | White | 19 (6) | 21,646 | 9.3 [8.7, 9.8] | 5 [3–6] | 0.000 |
| ethnicity | primary (2021-04 to 2023-03) | Pakistani | 49 (2) | 459 | 12.5 [8.2, 17.4] | 2 [1–5] | 0.108 |
| ethnicity | primary (2021-04 to 2023-03) | Bangladeshi | 56 (1) | 223 | 11.3 [4.6, 18.6] | 3 [1–6] | 0.080 |
| ethnicity | primary (2021-04 to 2023-03) | Black African | 40 (3) | 380 | 11.1 [7.2, 15.7] | 4 [1–6] | 0.034 |
| ethnicity | primary (2021-04 to 2023-03) | Black Caribbean | 30 (5) | 445 | 17.8 [10.2, 26.6] | 1 [1–4] | 0.770 |
| ethnicity | primary (2021-04 to 2023-03) | Any other Asian background | 34 (4) | 158 | 7.8 [3.0, 14.0] | 6 [2–6] | 0.009 |
| ethnicity | sensitivity (2020-04 to 2023-03) | White | 19 (6) | 33,189 | 8.2 [7.8, 8.7] | 5 [3–6] | 0.000 |
| ethnicity | sensitivity (2020-04 to 2023-03) | Pakistani | 49 (2) | 733 | 11.1 [7.5, 15.0] | 2 [1–5] | 0.084 |
| ethnicity | sensitivity (2020-04 to 2023-03) | Bangladeshi | 56 (1) | 362 | 9.0 [4.5, 14.1] | 4 [1–6] | 0.029 |
| ethnicity | sensitivity (2020-04 to 2023-03) | Black African | 40 (3) | 627 | 10.2 [7.1, 13.8] | 3 [1–5] | 0.036 |
| ethnicity | sensitivity (2020-04 to 2023-03) | Black Caribbean | 30 (5) | 743 | 17.0 [10.2, 25.2] | 1 [1–3] | 0.848 |
| ethnicity | sensitivity (2020-04 to 2023-03) | Any other Asian background | 34 (4) | 261 | 6.3 [2.7, 10.6] | 6 [3–6] | 0.002 |
| tenure | primary (2022-04 to 2023-03) | Owned outright | 14 (3) | 5,753 | 13.8 [12.7, 15.0] | 2 [1–2] | 0.348 |
| tenure | primary (2022-04 to 2023-03) | Buying with mortgage | 10 (4) | 4,602 | 5.9 [5.1, 6.8] | 4 [4–4] | 0.000 |
| tenure | primary (2022-04 to 2023-03) | Social renting | 44 (1) | 2,035 | 14.3 [12.2, 16.5] | 1 [1–2] | 0.648 |
| tenure | primary (2022-04 to 2023-03) | Private renting | 35 (2) | 1,667 | 11.4 [9.4, 13.5] | 3 [2–3] | 0.004 |
| disability | primary (2022-04 to 2023-03) | No disabled adult | 19 (2) | 9,071 | 9.5 [8.8, 10.4] | 2 [2–2] | 0.000 |
| disability | primary (2022-04 to 2023-03) | Contains disabled adult | 29 (1) | 4,797 | 14.3 [13.0, 15.6] | 1 [1–1] | 1.000 |
| family_type | primary (2022-04 to 2023-03) | Lone parent | 44 (1) | 584 | 20.8 [16.8, 25.1] | 1 [1–1] | 1.000 |
| family_type | primary (2022-04 to 2023-03) | Couple with children | 25 (2) | 2,317 | 7.8 [6.4, 9.4] | 2 [2–2] | 0.000 |
| work_status | primary (2022-04 to 2023-03) | Not in work | 54 (1) | 1,656 | 22.5 [19.6, 25.3] | 1 [1–1] | 1.000 |
| work_status | primary (2022-04 to 2023-03) | In work | 15 (2) | 8,178 | 7.0 [6.3, 7.7] | 2 [2–2] | 0.000 |

| Dimension | Window | Subset | Categories | Spearman ρ | Pearson r | Two groups, same direction |
|---|---|---|---|---|---|---|
| region | primary | all | 12 | -0.10 | -0.26 |  |
| region | primary | excl. Northern Ireland | 11 | 0.18 | 0.08 |  |
| region | sensitivity | all | 12 | -0.10 | -0.31 |  |
| region | sensitivity | excl. Northern Ireland | 11 | 0.18 | 0.11 |  |
| ethnicity | primary | all | 6 | 0.26 | 0.05 |  |
| ethnicity | sensitivity | all | 6 | 0.14 | -0.06 |  |
| tenure | primary | all | 4 | 0.80 | 0.58 |  |
| disability | primary | all | 2 |  |  | True |
| family_type | primary | all | 2 |  |  | True |
| work_status | primary | all | 2 |  |  | True |

</details>

<details><summary><b>Table A-12. Regional rates in the JRF windows, with rank CIs</b></summary>

| Window | Outcome | Region | n | PSUs | Weighted % [95% CI] | Rank [95% CI] | P(rank 1) |
|---|---|---|---|---|---|---|---|
| primary (2021-04 to 2023-03) | primary | North East | 1,047 | 198 | 8.8 [6.5, 11.3] | 7 [3–12] | 0.001 |
| primary (2021-04 to 2023-03) | primary | North West | 2,757 | 607 | 7.9 [6.6, 9.4] | 9 [6–12] | 0.000 |
| primary (2021-04 to 2023-03) | primary | Yorkshire and the Humber | 2,367 | 478 | 9.5 [7.7, 11.3] | 6 [3–10] | 0.000 |
| primary (2021-04 to 2023-03) | primary | East Midlands | 1,996 | 426 | 10.5 [8.6, 12.5] | 5 [2–8] | 0.004 |
| primary (2021-04 to 2023-03) | primary | West Midlands | 2,203 | 526 | 10.6 [8.6, 12.6] | 4 [2–8] | 0.004 |
| primary (2021-04 to 2023-03) | primary | East of England | 2,377 | 564 | 8.6 [7.2, 10.1] | 8 [5–11] | 0.000 |
| primary (2021-04 to 2023-03) | primary | London | 2,676 | 1,065 | 6.7 [5.0, 8.7] | 12 [8–12] | 0.000 |
| primary (2021-04 to 2023-03) | primary | South East | 3,433 | 759 | 7.4 [6.2, 8.7] | 11 [7–12] | 0.000 |
| primary (2021-04 to 2023-03) | primary | South West | 2,343 | 501 | 7.8 [6.4, 9.3] | 10 [6–12] | 0.000 |
| primary (2021-04 to 2023-03) | primary | Wales | 1,691 | 316 | 11.4 [9.4, 13.6] | 2 [1–6] | 0.027 |
| primary (2021-04 to 2023-03) | primary | Scotland | 2,677 | 457 | 11.2 [9.4, 13.2] | 3 [2–6] | 0.018 |
| primary (2021-04 to 2023-03) | primary | Northern Ireland | 1,562 | 890 | 14.4 [12.1, 17.0] | 1 [1–2] | 0.946 |
| primary (2021-04 to 2023-03) | s1_lower_bound | North East | 1,264 | 216 | 7.5 [5.4, 9.6] | 7 [3–11] | 0.000 |
| primary (2021-04 to 2023-03) | s1_lower_bound | North West | 3,388 | 636 | 6.5 [5.4, 7.7] | 9 [6–12] | 0.000 |
| primary (2021-04 to 2023-03) | s1_lower_bound | Yorkshire and the Humber | 2,919 | 508 | 7.8 [6.4, 9.4] | 6 [3–10] | 0.000 |
| primary (2021-04 to 2023-03) | s1_lower_bound | East Midlands | 2,421 | 450 | 8.8 [7.1, 10.5] | 4 [2–8] | 0.000 |
| primary (2021-04 to 2023-03) | s1_lower_bound | West Midlands | 2,744 | 568 | 8.7 [7.1, 10.3] | 5 [2–8] | 0.000 |
| primary (2021-04 to 2023-03) | s1_lower_bound | East of England | 2,945 | 611 | 7.0 [5.8, 8.2] | 8 [5–11] | 0.000 |
| primary (2021-04 to 2023-03) | s1_lower_bound | London | 3,449 | 1,200 | 5.2 [3.8, 6.7] | 12 [9–12] | 0.000 |
| primary (2021-04 to 2023-03) | s1_lower_bound | South East | 4,200 | 802 | 6.0 [5.0, 7.1] | 11 [7–12] | 0.000 |
| primary (2021-04 to 2023-03) | s1_lower_bound | South West | 2,856 | 531 | 6.5 [5.3, 7.7] | 10 [6–12] | 0.000 |
| primary (2021-04 to 2023-03) | s1_lower_bound | Wales | 2,085 | 328 | 9.2 [7.5, 10.9] | 3 [2–7] | 0.003 |
| primary (2021-04 to 2023-03) | s1_lower_bound | Scotland | 3,159 | 468 | 9.5 [8.0, 11.2] | 2 [2–6] | 0.004 |
| primary (2021-04 to 2023-03) | s1_lower_bound | Northern Ireland | 1,745 | 953 | 13.2 [11.1, 15.5] | 1 [1–1] | 0.993 |
| sensitivity (2020-04 to 2023-03) | primary | North East | 1,607 | 212 | 8.1 [6.3, 10.1] | 7 [3–11] | 0.000 |
| sensitivity (2020-04 to 2023-03) | primary | North West | 4,189 | 640 | 7.1 [6.1, 8.2] | 9 [6–12] | 0.000 |
| sensitivity (2020-04 to 2023-03) | primary | Yorkshire and the Humber | 3,614 | 505 | 8.4 [7.0, 9.8] | 6 [3–9] | 0.000 |
| sensitivity (2020-04 to 2023-03) | primary | East Midlands | 3,094 | 452 | 9.2 [7.6, 11.0] | 5 [2–8] | 0.000 |
| sensitivity (2020-04 to 2023-03) | primary | West Midlands | 3,438 | 574 | 9.4 [7.9, 10.8] | 4 [2–7] | 0.000 |
| sensitivity (2020-04 to 2023-03) | primary | East of England | 3,635 | 604 | 7.7 [6.4, 9.1] | 8 [4–11] | 0.000 |
| sensitivity (2020-04 to 2023-03) | primary | London | 4,311 | 1,209 | 6.3 [4.8, 8.0] | 12 [7–12] | 0.000 |
| sensitivity (2020-04 to 2023-03) | primary | South East | 5,235 | 805 | 6.5 [5.6, 7.5] | 11 [8–12] | 0.000 |
| sensitivity (2020-04 to 2023-03) | primary | South West | 3,596 | 533 | 7.0 [5.9, 8.2] | 10 [6–12] | 0.000 |
| sensitivity (2020-04 to 2023-03) | primary | Wales | 2,643 | 330 | 9.4 [7.9, 11.1] | 3 [2–7] | 0.000 |
| sensitivity (2020-04 to 2023-03) | primary | Scotland | 4,017 | 469 | 10.3 [8.9, 11.8] | 2 [2–5] | 0.002 |
| sensitivity (2020-04 to 2023-03) | primary | Northern Ireland | 2,305 | 990 | 14.2 [12.1, 16.4] | 1 [1–1] | 0.999 |
| sensitivity (2020-04 to 2023-03) | s1_lower_bound | North East | 1,909 | 228 | 7.1 [5.5, 8.8] | 6 [2–10] | 0.000 |
| sensitivity (2020-04 to 2023-03) | s1_lower_bound | North West | 5,121 | 663 | 5.9 [4.9, 6.8] | 10 [7–12] | 0.000 |
| sensitivity (2020-04 to 2023-03) | s1_lower_bound | Yorkshire and the Humber | 4,392 | 530 | 7.0 [5.9, 8.3] | 7 [3–9] | 0.000 |
| sensitivity (2020-04 to 2023-03) | s1_lower_bound | East Midlands | 3,692 | 473 | 7.8 [6.5, 9.4] | 3 [2–8] | 0.000 |
| sensitivity (2020-04 to 2023-03) | s1_lower_bound | West Midlands | 4,191 | 605 | 7.8 [6.5, 9.0] | 4 [2–7] | 0.000 |
| sensitivity (2020-04 to 2023-03) | s1_lower_bound | East of England | 4,464 | 643 | 6.3 [5.2, 7.4] | 8 [5–11] | 0.000 |
| sensitivity (2020-04 to 2023-03) | s1_lower_bound | London | 5,494 | 1,325 | 5.0 [3.8, 6.4] | 12 [8–12] | 0.000 |
| sensitivity (2020-04 to 2023-03) | s1_lower_bound | South East | 6,347 | 845 | 5.3 [4.5, 6.1] | 11 [8–12] | 0.000 |
| sensitivity (2020-04 to 2023-03) | s1_lower_bound | South West | 4,309 | 558 | 5.9 [5.0, 6.9] | 9 [6–12] | 0.000 |
| sensitivity (2020-04 to 2023-03) | s1_lower_bound | Wales | 3,210 | 338 | 7.7 [6.5, 9.1] | 5 [2–8] | 0.000 |
| sensitivity (2020-04 to 2023-03) | s1_lower_bound | Scotland | 4,685 | 479 | 8.8 [7.5, 10.3] | 2 [2–5] | 0.001 |
| sensitivity (2020-04 to 2023-03) | s1_lower_bound | Northern Ireland | 2,555 | 1,046 | 13.0 [11.0, 15.1] | 1 [1–1] | 0.999 |

</details>

### Stage 6: next-wave prediction (RQ5; H5)

<table>
<tr>
<td align="center" width="50%"><img src="outputs_v2/thesis_assets_v2/figures/fig4-16_roc_p0_p3.png" width="400"><br><sub>Figure 4-16. ROC curves, validation transitions m→n and n→o (n = 19,960)</sub></td>
<td align="center" width="50%"><img src="outputs_v2/thesis_assets_v2/figures/fig4-17_calibration.png" width="400"><br><sub>Figure 4-17. Calibration by decile of predicted risk, validation transitions</sub></td>
</tr>
</table>

**Table 4-8. Next-wave prediction, P0–P3** (training a→b … l→m, n = 170,895; validation m→n and n→o, n = 19,960; 95% CIs from 2,000 PSU bootstrap replicates; P3 is post-hoc).

| Model | ROC-AUC [95% CI] | PR-AUC [95% CI] | Calibration slope | Calibration-in-the-large | Top 5%: sensitivity / PPV (%) | Top 10%: sensitivity / PPV (%) | n validation | prevalence (%) | note |
|---|---|---|---|---|---|---|---|---|---|
| P0 current burden (benchmark) | 0.780 [0.768, 0.792] | 0.334 [0.314, 0.359] | 0.78 [0.74, 0.82] | 0.39 [0.34, 0.44] | 22.0 [20.6, 23.6] / 46.4 [43.3, 50.1] | 38.6 [36.6, 40.4] / 40.6 [38.4, 43.2] | 19960 | 10.54 |  |
| P1 household predictors | 0.739 [0.727, 0.751] | 0.280 [0.261, 0.302] | 0.93 [0.88, 0.98] | 0.45 [0.40, 0.49] | 18.8 [17.3, 20.3] / 39.6 [36.1, 43.1] | 30.9 [29.0, 32.6] / 32.5 [30.1, 34.7] | 19960 | 10.54 |  |
| P2 = P1 + FES (growth-only magnitude, t+1) | 0.739 [0.726, 0.750] | 0.278 [0.259, 0.300] | 0.93 [0.87, 0.98] | 0.42 [0.37, 0.47] | 18.7 [17.2, 20.2] / 39.4 [36.0, 42.7] | 30.7 [28.8, 32.4] / 32.3 [29.8, 34.6] | 19960 | 10.54 | no improvement over P1 (ΔAUC -0.0006) |
| P3 = P0 + P1 (POST-HOC / EXPLORATORY) | 0.781 [0.770, 0.792] | 0.341 [0.320, 0.365] |  |  |  | 39.0 [37.1, 40.6] / 41.1 [38.6, 43.6] | 19960 | 10.54 |  |

<details><summary><b>Table A-2. Per-transition AUC</b></summary>

| Model | Transition | n | Prevalence (%) | AUC [95% CI] |
|---|---|---|---|---|
| P0 | m→n | 9,075 | 9.6 | 0.781 [0.763, 0.800] |
| P0 | n→o | 10,885 | 11.3 | 0.777 [0.762, 0.792] |
| P1 | m→n | 9,075 | 9.6 | 0.738 [0.720, 0.756] |
| P1 | n→o | 10,885 | 11.3 | 0.740 [0.725, 0.754] |
| P2 | m→n | 9,075 | 9.6 | 0.737 [0.719, 0.755] |
| P2 | n→o | 10,885 | 11.3 | 0.740 [0.725, 0.754] |

</details>

<details><summary><b>Table A-7. Calibration slope and intercept</b></summary>

| Model | Calibration slope [95% CI] | Calibration-in-the-large [95% CI] |
|---|---|---|
| P0 | 0.783 [0.743, 0.824] | 0.386 [0.335, 0.436] |
| P1 | 0.930 [0.878, 0.981] | 0.446 [0.399, 0.494] |
| P2 | 0.926 [0.875, 0.978] | 0.420 [0.373, 0.468] |

</details>

<details><summary><b>Table A-8. Prediction sample flow</b></summary>

| Step | n | Wave-t households |
|---|---|---|
| linked transitions a→b | 21,886 | 30,169 |
| linked transitions b→c | 24,404 | 30,484 |
| linked transitions c→d | 22,961 | 27,751 |
| linked transitions d→e | 22,034 | 25,817 |
| linked transitions e→f | 19,771 | 24,325 |
| linked transitions f→g | 20,001 | 24,454 |
| linked transitions g→h | 19,288 | 23,033 |
| linked transitions h→i | 17,954 | 21,746 |
| linked transitions i→j | 17,083 | 20,048 |
| linked transitions j→k | 16,234 | 19,252 |
| linked transitions k→l | 15,021 | 18,139 |
| linked transitions l→m | 14,253 | 16,856 |
| linked transitions m→n | 13,725 | 16,156 |
| linked transitions n→o | 17,144 | 21,385 |
| all linked transitions | 261,759 |  |
|   of which outcome at t+1 missing | 36,602 |  |
|   missing y (only this missing: 19,945) | 36,602 |  |
|   missing fuel_to_income_ratio (only this missing: 0) | 35,291 |  |
|   missing high_fuel_vulnerable (only this missing: 0) | 35,291 |  |
|   missing finnow (only this missing: 20) | 1,115 |  |
|   missing scghq1_dv (only this missing: 9,282) | 14,901 |  |
|   missing finfut_risk (only this missing: 1,694) | 3,997 |  |
|   missing dvage (only this missing: <10) | 730 |  |
|   missing heatch (only this missing: 28) | 773 |  |
|   missing workless_household (only this missing: 0) | 724 |  |
|   missing OBJECT (only this missing: 1,525) | 3,237 |  |
|   missing CONDITION (only this missing: 0) | 306 |  |
|   missing PERSONAL (only this missing: 12) | 1,402 |  |
|   missing ENERGY (only this missing: 0) | <10 |  |
| common sample (P0/P1/P2) | 190,855 |  |
|   training transitions (a→b ... l→m) | 170,895 |  |
|   validation transitions (m→n, n→o) | 19,960 |  |
|   validation prevalence (%) | 10.54 |  |
| P2b common sample (adds FES Delta at t; loses 2009 wave-t interviews) | 183,179 |  |

</details>

### Stage 7 and robustness

<table>
<tr>
<td align="center" width="50%"><img src="outputs_v2/thesis_assets_v2/figures/fig4-18_equivalisation_sensitivity.png" width="400"><br><sub>Figure 4-18. Sensitivity to equivalising income only (fuel spend not equivalised)</sub></td>
<td align="center" width="50%"><img src="outputs_v2/thesis_assets_v2/figures/fig4-19_fes_tercile.png" width="400"><br><sub>Figure 4-19. Prevalence by FES tercile (descriptive)</sub></td>
</tr>
</table>

<details><summary><b>Table 4-9. Robustness summary across pre-specified sensitivities</b></summary>

| Estimate | Specification | Value [95% CI] | n |
|---|---|---|---|
| FES Delta OR | primary (main) | 0.972 [0.962, 0.981] | 221,877 |
| FES Delta OR | primary (main_twoway_cluster) | 0.972 [0.958, 0.985] | 221,877 |
| FES Delta OR | sens_composite (main) | 0.973 [0.964, 0.982] | 233,980 |
| FES Delta OR | sens_composite_v1 (main) | 0.973 [0.964, 0.982] | 234,102 |
| FES Delta OR | sens_lagged_components (main) | 0.972 [0.962, 0.983] | 183,078 |
| FES Delta OR | sens_lagged_composite (main) | 0.973 [0.962, 0.983] | 194,151 |
| FES Delta OR | sens_lagged_composite_v1 (main) | 0.973 [0.963, 0.983] | 194,692 |
| FES Delta OR | sens_month_fe (main) | 0.968 [0.959, 0.978] | 221,877 |
| FES Delta OR | sens_fes_4term (main) | 0.970 [0.961, 0.980] | 221,877 |
| FES Delta OR | sens_outcome_s1 (main) | 0.974 [0.965, 0.983] | 249,200 |
| FES Delta OR | sens_outcome_s2 (main) | 0.972 [0.963, 0.981] | 227,007 |
| FES Delta OR | sens_no_qualification (main) | 0.973 [0.964, 0.981] | 248,802 |
| NI gap AME (pp) | ni_a_regionFE_filled | 6.82 [5.76, 7.88] | 221,778 |
| NI gap AME (pp) | ni_b_oil_filled | 2.39 [1.40, 3.37] | 221,778 |
| NI gap AME (pp) | ni_b_oil_rural_filled | 2.51 [1.52, 3.51] | 221,778 |
| NI gap AME (pp) | ni_c_ni_x_oil_filled | 2.63 [1.22, 4.03] | 221,778 |
| NI gap AME (pp) | ni_c_ni_x_oil_rural_filled | 2.59 [1.19, 3.99] | 221,778 |
| NI gap AME (pp) | ni_b_oil_observed_only | 2.39 [1.40, 3.38] | 221,611 |
| NI gap AME (pp) | ni_b_oil_rural_observed_only | 2.52 [1.53, 3.52] | 221,611 |
| NI gap AME (pp) | ni_c_ni_x_oil_observed_only | 2.65 [1.23, 4.06] | 221,611 |
| NI gap AME (pp) | ni_c_ni_x_oil_rural_observed_only | 2.62 [1.20, 4.03] | 221,611 |
| NI rate (%) | NI rate, primary window (2021-04 to 2023-03), outcome primary | 14.42 [12.09, 16.96] |  |
| NI rate (%) | NI rate, primary window (2021-04 to 2023-03), outcome s1_lower_bound | 13.20 [11.11, 15.54] |  |
| NI rate (%) | NI rate, sensitivity window (2020-04 to 2023-03), outcome primary | 14.17 [12.09, 16.40] |  |
| NI rate (%) | NI rate, sensitivity window (2020-04 to 2023-03), outcome s1_lower_bound | 13.02 [10.99, 15.05] |  |
| H1 verdict | primary | not supported | 269,372 |
| H1 verdict | sens_R_with_energy | not supported | 269,372 |
| H1 verdict | sens_logit_binary | contrary to COR | 269,372 |
| H1 verdict | sens_fes_4term | not supported | 269,372 |
| Prediction AUC (P2b sample) | P1 (P2b sample) | 0.739 [, ] | 19,960 |
| Prediction AUC (P2b sample) | P2b | 0.740 [, ] | 19,960 |
| Prevalence wave a (%) | primary | 11.98 [, ] |  |
| Prevalence wave l (%) | primary | 6.52 [, ] |  |
| Prevalence wave n (%) | primary | 12.47 [, ] |  |
| Prevalence wave o (%) | primary | 12.42 [, ] |  |
| Prevalence wave a (%) | S1 lower bound | 10.81 [, ] |  |
| Prevalence wave l (%) | S1 lower bound | 5.56 [, ] |  |
| Prevalence wave n (%) | S1 lower bound | 10.22 [, ] |  |
| Prevalence wave o (%) | S1 lower bound | 9.94 [, ] |  |
| Prevalence wave a (%) | S2 incl. electricity not reported | 11.93 [, ] |  |
| Prevalence wave l (%) | S2 incl. electricity not reported | 6.41 [, ] |  |
| Prevalence wave n (%) | S2 incl. electricity not reported | 12.27 [, ] |  |
| Prevalence wave o (%) | S2 incl. electricity not reported | 12.23 [, ] |  |
| Prevalence wave a (%) | v1 rule | 10.77 [, ] |  |
| Prevalence wave l (%) | v1 rule | 5.69 [, ] |  |
| Prevalence wave n (%) | v1 rule | 11.42 [, ] |  |
| Prevalence wave o (%) | v1 rule | 11.43 [, ] |  |

</details>

<details><summary><b>Table A-11. Sensitivity to equivalising income only</b></summary>

| Household size | n | Flagged, primary (%) | Flagged, income equivalised (%) | Flip (%) | In → out (%) | Out → in (%) |
|---|---|---|---|---|---|---|
| 1 | 73,682 | 15.9 | 15.9 | 0.0 | 0.0 | 0.0 |
| 2 | 99,863 | 6.5 | 17.4 | 11.5 | 0.0 | 11.5 |
| 3 | 45,329 | 5.3 | 22.1 | 18.2 | 0.0 | 18.2 |
| 4 | 43,354 | 3.6 | 27.0 | 25.3 | 0.0 | 25.3 |
| 5+ | 24,478 | 4.3 | 46.4 | 45.8 | 0.0 | 45.8 |

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

| Wave | 2009 | 2010 | 2011 | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a | 15,146 | 14,029 | 994 | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| b | — | 17,742 | 12,205 | 537 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| c | — | — | 16,336 | 10,820 | 591 | — | — | — | — | — | — | — | — | — | — | — | — |
| d | — | — | — | 15,044 | 10,013 | 760 | — | — | — | — | — | — | — | — | — | — | — |
| e | — | — | — | — | 13,936 | 9,698 | 691 | — | — | — | — | — | — | — | — | — | — |
| f | — | — | — | — | — | 12,672 | 10,549 | 1,233 | — | — | — | — | — | — | — | — | — |
| g | — | — | — | — | — | — | 12,145 | 9,963 | 925 | — | — | — | — | — | — | — | — |
| h | — | — | — | — | — | — | — | 11,813 | 8,902 | 1,031 | — | — | — | — | — | — | — |
| i | — | — | — | — | — | — | — | — | 10,960 | 8,243 | 845 | — | — | — | — | — | — |
| j | — | — | — | — | — | — | — | — | — | 10,619 | 8,092 | 541 | — | — | — | — | — |
| k | — | — | — | — | — | — | — | — | — | — | 10,141 | 7,660 | 338 | — | — | — | — |
| l | — | — | — | — | — | — | — | — | — | — | — | 9,887 | 6,704 | 265 | — | — | — |
| m | — | — | — | — | — | — | — | — | — | — | — | — | 9,549 | 6,299 | 308 | — | — |
| n | — | — | — | — | — | — | — | — | — | — | — | — | — | 11,043 | 9,698 | 644 | — |
| o | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 10,758 | 8,528 | 300 |

</details>

<details><summary><b>Table A-14. Household-waves by region</b></summary>

| Region | Household-waves |
|---|---|
| East Midlands | 24,339 |
| East of England | 28,545 |
| London | 41,318 |
| North East | 12,643 |
| North West | 34,392 |
| Northern Ireland | 21,486 |
| Scotland | 30,538 |
| South East | 40,315 |
| South West | 27,323 |
| Wales | 22,768 |
| West Midlands | 27,592 |
| Yorkshire and the Humber | 27,761 |

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
