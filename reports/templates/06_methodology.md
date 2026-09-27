# Methodology (rerun v2)

**Project:** Anticipatory Fuel Stress Watch (AFSW) — forecasting anticipatory energy–carbon stress and household fuel vulnerability in the UK.
**Scope:** the v2 methods for every stage: specifications, estimators, validation designs, disclosure control and software, with the implementing script for each. The specification was fixed in [`analysis_plan_rerun.md`](../analysis_plan_rerun.md) before any code change; every later change is in its deviation log (§9), summarised in §10 below.
**Status:** branch `rerun-v2`, analysis frozen 2026-09-26. Results are in [`02_findings_report.md`](02_findings_report.md), data in [`05_data_description.md`](05_data_description.md).

---

## 1. Overview

![Figure 3-1. Analysis pipeline, rerun v2](../outputs_v2/thesis_assets_v2/figures/fig3-1_pipeline_v2.png)

*Figure 3-1. Analysis pipeline, rerun v2 (`outputs_v2/thesis_assets_v2/figures/`, local build).*

| Stage | Question | Script(s) | Output folder |
|---|---|---|---|
| 1 Outcome audit | Is the v1 outcome correct? | `scripts/audit_ukhls_codes.py` | `outputs_v2/audit/` |
| 2 FES | Can price stress be forecast and attached without look-ahead? | `forecast_pipeline.py --rolling --core-only --tune-per-origin`; `src/ukhls_preprocessing.py`; `scripts/stage2_fes_evaluation.py` | `outputs_v2/fes/`, `fes_eval/` |
| Descriptives | Prevalence by wave, year, region, group | `scripts/descriptives_v2.py` | `outputs_v2/descriptives/` |
| 3 Drivers | What is associated with fuel vulnerability? (H2, H3) | `scripts/stage3_drivers.py` | `outputs_v2/stage3/` |
| 4 Resources, H1 | Do resources buffer FES? (H1) | `scripts/stage4_resources.py`, `scripts/stage4_h1.py` | `outputs_v2/resources/`, `stage4/` |
| 5 JRF, NI | Does fuel vulnerability match income poverty? (H4) | `scripts/jrf_comparison_v2.py`, `scripts/stage5_jrf_thesis.py` | `outputs_v2/jrf/`, `stage5/` |
| 6 Prediction | Can next-wave vulnerability be predicted? (H5) | `scripts/stage6_prediction.py`, `scripts/stage6_p3_posthoc.py` | `outputs_v2/stage6/` |
| 7 Scope | Sensitivities and appendix scope | `scripts/stage7_scope.py` | `outputs_v2/stage7/` |

The work stopped after each stage for the author's approval (plan §8).

---

## 2. Stage 2 — Forecasting and the FES index

### 2.1 Forecasting models

Four model families are fit for each of three monthly series (year-on-year % growth of gas, electricity and carbon prices; `data/processed/core_energy_carbon.csv`, May 2006–March 2026) in **core mode only**: the target series, with no exogenous regressors. v1's macro mode was dropped (plan Stage 2).

- **SARIMA** (`src/models/sarima_model.py`): order chosen by `pmdarima.auto_arima`, seasonal period 12.
- **Prophet** (`src/models/prophet_model.py`): yearly seasonality, no weekly or daily seasonality. Additive and multiplicative seasonality are both candidates for gas. Prophet's `cmdstan` optimiser fell back to Newton in 205 of 1,325 calls (15.5%), with no failed fits:

{{TABLE:prophet}}

- **LSTM** (`src/models/lstm_model.py`): two stacked LSTM layers (128, 64) with dropout 0.2, a 12-month lookback, and 95% intervals from Monte Carlo dropout (200 passes).
- **TFT** (`src/models/tft_model.py`): `pytorch-forecasting`'s Temporal Fusion Transformer trained with `lightning`.

### 2.2 Rolling walk-forward design with per-origin tuning

For each origin December Y (Y = 2009 … 2024), every model is **re-tuned on data up to that origin only**, fit, and used to forecast the 12 months of Y+1. The first origin is December 2009: the December 2008 origin had only 20 months of history before its validation year, fewer than the 24 required. Candidate grids per series and origin are SARIMA 1, Prophet 8 (16 for gas), LSTM 3 and TFT 3 (`outputs_v2/tuning_rolling/`). The winning model per series and origin is chosen on walk-forward validation error, never on realised target-year accuracy:

{{TABLE:model_sel}}

*Winning core model by target year (`outputs_v2/fes/model_selection_by_year.csv`).*

### 2.3 The FES index

For target month *t* of origin *Y*:

```
FES_growth3(t) = z(gas_t) + z(elec_t) + z(carbon_t)                      (primary)
FES_4term(t)   = FES_growth3(t) + z(uncertainty_t)                       (sensitivity)
```

Each z uses the **past-only** mean and SD of that series: an expanding window from the start of the series through the December Y origin, with at least 36 months (`src/ukhls_preprocessing.py::_past_only_moments`). The uncertainty term is the forecast's prediction-interval half-width, z-scored. The growth-only index became primary because v2 95% intervals cover only 28–48% of outcomes (deviation, 2026-09-26, decided on forecast diagnostics before any household model).

{{TABLE:fes_yearly}}

*Annual means of the rolling FES (`outputs_v2/fes/fes_rolling_yearly.csv`; 4-term core index, the inverse-RMSE weighted variant, and the realised index). The macro column is empty in a core-only run.*

### 2.4 Attaching FES to households (`src/ukhls_preprocessing.py::attach_fes_delta`)

- **Vintage.** A household interviewed in calendar year Y receives the forecast from origin **December Y−1**, for its own interview month. Nothing after the origin is used, and the origin precedes every interview it is attached to. Interview year and month come from the actual household interview date. If the month is unknown, the annual mean of the same vintage is used.
- **FES magnitude** = the forecast index for that month (`fes_magnitude_growth3`; `fes_magnitude` for the 4-term index).
- **FES current** = the realised growth in month *m*−1 (the last complete month before the interview), z-scored with the same past-only moments. It has three terms, because realised uncertainty has no counterpart.
- **FES Delta** = magnitude − current (`fes_delta_growth3`, primary). Negative Delta means realised stress exceeded the forecast.
- Interviews in 2009 have no feasible origin and no FES (15,146 rows). They are excluded from FES models only.

### 2.5 Forecast evaluation (`scripts/stage2_fes_evaluation.py`)

Accuracy is computed on the three growth terms only, over 192 non-overlapping target months per series (2010–2025, horizons 1–12). Versions: v1 core, v1 macro (the variant v1 attached to households, chosen by hindsight), and v2 core. Benchmarks:

- **naive**: the growth observed at the origin (December Y), repeated for all 12 months;
- **seasonal naive**: month *m* of Y+1 is forecast by the observed growth in month *m* of Y.

Metrics: RMSE, MAE, sign agreement, Pearson r, relative RMSE and MAE against each benchmark, and MASE (MAE scaled by the in-sample one-step naive MAE up to each origin, averaged over origins). **Diebold–Mariano** tests use squared-error loss, Newey–West variance with lag 11, and the Harvey–Leybourne–Newbold correction at h = 12 (conservative). **Uncertainty** is reported as 95% PI coverage and width.

---

## 3. Outcome, sample and measures

### 3.1 Outcome (amendment A1)

The routing-aware spend, the complete-case rule, and the S1 and S2 bounds are defined in [`05_data_description.md`](05_data_description.md) §4.4 (Tables 3-3 and 3-4). The ratio = spend ÷ (12 × net monthly household income). It is missing if annual income is below £1,200 and capped at 1.0. `high_fuel_vulnerable` = ratio ≥ 0.10, the official 10% threshold with unequivalised income.

- **S1 (lower bound):** item non-response set to £0 (v1's zero-fill), with the routing otherwise correct.
- **S2:** adds households that do not report electricity (gas-only; oil- or other-fuel-only), with spend as reported.

### 3.2 Measures

{{TABLE:T3-5}}

*Table 3-5. Measures. The strain components enter separately (deviation 2026-09-26): the three-item composite has α = 0.27, `finnow`–`finfut_risk` r = 0.02, and `finfut_risk` correlates with age (r = 0.32). Financial expectations are therefore always estimated with age controlled.*

### 3.3 Weights and pooled rates (`scripts/descriptives_v2.py`)

Weighted rates use each wave's household cross-sectional weight (`hhdenus_xw`, `hhdenub_xw`, `hhdenui_xw`, `hhdeng2_xw`). A **pooled** rate is the mean of the per-wave weighted rates, each wave counting equally, so the large early waves do not dominate. Categories with fewer than 100 unweighted households are masked. The 95% CIs for pooled group rates (Table 4-2) come from a PSU bootstrap in `scripts/build_thesis_assets.py`.

---

## 4. Stage 3 — Driver models (`scripts/stage3_drivers.py`)

**Primary specification.** Logit of `high_fuel_vulnerable` on:

- current financial difficulty (`finnow`), GHQ-12 distress (`scghq1_dv`) and financial expectations (`finfut_risk`), entered separately;
- bill-payment security;
- tenure and employment-status security, health (`health_good`, `sf1_good`), qualification band, age, central heating, bedrooms, rooms, cars;
- lone parent, large family, workless household, the OECD equivalence scale;
- **FES Delta (growth-only)**;
- **interview-year fixed effects**.

Standard errors are clustered on PSU. Complete case; FES restricts the sample to interviews from 2010. The n by specification:

{{TABLE:s3_flow}}

*Estimation samples (`outputs_v2/stage3/sample_flow.csv`). Highest qualification is the only missing control for 26,925 of the 52,251 households lost to complete-case estimation, hence the `sens_no_qualification` specification.*

**Pre-specified sensitivities.**

- Two-way clustering on PSU × interview year-month (the unit at which FES varies; about 185 clusters).
- The strain composite without `xphsdba`, and the v1 composite with it.
- Lagged strain components, and lagged composites (previous wave, linked by reference person).
- Calendar-month fixed effects.
- The 4-term FES.
- The S1 and S2 outcomes.
- Without qualification (post-hoc, logged).

**NI-oil sequence.** Region fixed effects (reference South East) are added, then:

- (a) region FE only;
- (b) + oil use (`fuelhave3` = 1);
- (c) + NI × oil.

Models (b) and (c) are also fit with rural location (`urban_dv`, adjacent-wave filled where there is no evidence of a move; an observed-only variant for the primary specification). Nested models share one estimation sample. The NI gap and the oil effect are reported as **average marginal effects** in percentage points, with delta-method CIs from the PSU-clustered covariance.

**Reporting.** OR, 95% CI and p for every term; OR per SD for continuous predictors; N, events, PSUs and McFadden pseudo-R² for every model (Table A-4). **Identification.** With year FE, the FES Delta coefficient is identified only from within-year variation across interview months. This is reported as a limitation.

---

## 5. Stage 4 — Resources and H1

### 5.1 CFA and the fallback rule (`scripts/stage4_resources.py`)

A correlated four-factor first-order CFA (OBJECT, CONDITION, PERSONAL, ENERGY) was fit by complete-case ML (`semopy`), with **one attempt only**. Monetary items were transformed with log(1 + max(x, 0)) and all items standardised. Markers were fixed to 1 so that higher = more resource: `hsrooms`, `tenure_security`, `sf1_good`, `fihhmnnet1_dv`. **Pre-registered criteria:** CFI ≥ 0.90, RMSEA ≤ 0.08, SRMR ≤ 0.08, no Heywood case, and every standardised loading ≥ 0.30 with the expected sign.

{{TABLE:TA-9}}

*Table A-9. CFA fit.*

{{TABLE:TA-10}}

*Table A-10. CFA loadings.*

{{TABLE:res_decision}}

The complete-case CFA sample is 99% owner-occupiers (house value is asked only of owners), a second reason not to read it as a population measurement model. Following the rule, resources are measured by **unit-weighted formative indices**. For each domain: the mean of the standardised items, where at least 50% are observed, then re-standardised. **R** = OBJECT + CONDITION + PERSONAL (primary); **R_with_energy** adds ENERGY (sensitivity). Higher always means more resources. α is descriptive. The 3 × 3 hotspot tiers were dropped; regional resources are shown as continuous weighted means.

### 5.2 H1 (`scripts/stage4_h1.py`)

**Primary model:** OLS of the fuel-to-income ratio on R, FES Delta (growth-only), R × Delta and interview-year FE, with SEs clustered on PSU; n = 269,372.

**Decision rule** (corrected 2026-09-26, before any H1 fit, once Delta's sign convention was fixed). COR predicts a **positive** interaction: resources flatten the negative Delta slope.

- *Supported* if the interaction is positive with p < 0.05.
- *Contrary to COR* if it is negative with p < 0.05.
- *Not supported* otherwise.

**Also reported:**

- predicted Delta slopes with 95% CIs at the 10th, 50th and 90th percentiles of R;
- a **buffering bound**: the flattening of the slope from p10 to p90 at the upper confidence limit of the interaction, per SD of Delta, in percentage points of income and as a share of the p10 slope.

**Sensitivities:** R with ENERGY; the 4-term FES; a logit on the binary outcome (slopes on the log-odds scale, and on the probability scale as a footnote).

---

## 6. Stage 5 — JRF comparison (`scripts/jrf_comparison_v2.py`, `scripts/stage5_jrf_thesis.py`)

- **Windows.** Households are selected by actual interview date within the JRF period. Region and ethnicity use April 2021–March 2023; DWP excludes 2020/21 from three-year averages, so April 2020–March 2023 is a sensitivity. Tenure, disability, family type and work status use FY 2022/23. Metadata: Table 3-7 in [`05_data_description.md`](05_data_description.md) §6.
- **Units.** Work status is restricted to households with at least one respondent aged 16–64. Family type compares lone parents with couples with children, of any size. Disability compares households containing a disabled adult with households where none is disabled, among households with at least one adult's status observed.
- **Estimator.** The per-wave weighted rate among in-window households, averaged over waves with each wave weighted by its in-window n.
- **Uncertainty.** 2,000 bootstrap replicates resampling PSUs UK-wide give 95% CIs, rank intervals and P(rank 1).
- **Agreement.** Spearman ρ and Pearson r for dimensions with more than two categories (region all 12 and without NI; ethnicity six; tenure four). Same direction for the two-group dimensions.
- **Disclosure.** Categories with fewer than 100 households are masked. For the thesis tables, secondary suppression was also checked: complement risk, and the 2020/21 subgroup recoverable by differencing the two windows. No cell triggered it:

{{TABLE:supp_log}}

- **Northern Ireland.**
  - Oil share: weighted within NI in the window; pooled unweighted for other regions.
  - Oil vs non-oil rates within NI, with CIs.
  - The NI gap across the Stage 3 NI-oil sequence (AMEs).

---

## 7. Stage 6 — Next-wave prediction (`scripts/stage6_prediction.py`; amendment A6)

**Transitions.** Consecutive-wave pairs *t* → *t*+1 linked by the household reference person (`hrpid`). Reference persons appearing twice in a wave are excluded. The outcome is `high_fuel_vulnerable` at *t*+1. **Split:** train a→b … l→m; validate m→n and n→o.

**Models** (unpenalised logistic regression, one common sample):

- **P0 (benchmark):** `fuel_to_income_ratio` and `high_fuel_vulnerable` at *t*.
- **P1:**
  - `finnow`, `scghq1_dv`, `finfut_risk`;
  - `dvage`, `heatch`, lone parent, large family, workless household;
  - the four resource composites, all at *t*.
- **P2:** P1 + `fes_magnitude_growth3` for the household's *t*+1 interview month, from the December (Y<sub>t+1</sub> − 1) vintage, which is published before the outcome year begins.
- **P2b (sensitivity):** P2 + `fes_delta_growth3` at *t*, on its own common sample; it loses 2009 wave-*t* interviews.
- **P3 = P0 + P1:** post-hoc and exploratory, added after Stage 6 results were seen. No pre-specified conclusion depends on it.

**Leakage control.** Continuous predictors and FES are z-scored with training-transition statistics only. The resource composites are rebuilt from items using training statistics. Binary predictors are left unscaled.

{{TABLE:s6_std}}

*Training means and SDs (`outputs_v2/stage6/standardisation_train_stats.csv`).*

{{TABLE:TA-8}}

*Table A-8. Prediction sample flow: 261,759 linked transitions → common sample 190,855 (training 170,895; validation 19,960).*

**Metrics (validation set).**

- ROC-AUC and PR-AUC (with the prevalence baseline), each with 95% CIs from 2,000 bootstrap replicates resampling wave-*t* PSUs.
- Calibration slope (coefficient on logit *p*) and calibration-in-the-large (intercept with logit *p* as an offset).
- AUC per validation transition.
- Sensitivity and PPV at the top 5% and 10% of predicted risk.
- Paired bootstrap ΔAUC for P1 − P0 and P2 − P1.

---

## 8. Stage 7 — Scope (`scripts/stage7_scope.py`)

- **Sensitivity to equivalising income only.** The primary 10% flag is compared with the same flag using income divided by the modified-OECD scale (`ieqmoecd_dv`). Fuel spend is not equivalised, hence the relabelling from v1's "equivalisation check". Reported by household size as flags, flips, and flip direction.
- **Descriptive refreshes (no models):** prepayment-meter use by status; regional change, waves a–e vs k–o; prevalence by tercile of the growth-only FES magnitude and Delta.
- **Appendix scope.** The CVAE, fuzzy c-means and one-class SVM are v1 exploratory results and were not re-estimated. The vector-shift map, hotspot tiers, regional driver models and v1 forward-risk maps are dropped ([`appendix_scope_note.md`](../outputs_v2/reports/appendix_scope_note.md)).

---

## 9. Disclosure control and reproducibility

- **Row-level data** (the UKHLS panel, resource scores) never enter git: `.gitignore` guards, and history was rewritten on 2026-09-26 to remove earlier row-level files.
- **Small cells** (`scripts/suppress_small_cells.py`): counts of 1–9 are shown as `<10`; rates are suppressed if the denominator is below 10 or the implied numerator is 1–9; identifiers are removed. `--check` must pass before any output commit, and a local pre-commit hook enforces it. Group tables also mask categories with n < 100.
- **Results inventory** (`scripts/build_results_inventory.py`): builds `outputs_v2/results_inventory.csv`, 622 numbers with source file and commit, from committed outputs only. It fails on untracked or modified sources.
- **Thesis bundle** (`scripts/build_thesis_assets.py`): figures at 300 dpi and 16 cm width, tables, documents and a MANIFEST with source commits. The ROC and calibration figures refit the pre-specified Stage 6 models exactly as in `scripts/stage6_prediction.py`.
- **Seeds and software.**
  - Fixed seeds: NumPy, PyTorch and TensorFlow, 42 (`src/config.py`).
  - Python with `pandas`/`numpy`; `statsmodels` (logit, OLS, clustered SEs); `semopy` (CFA); `scikit-learn` (ROC-AUC, PR-AUC); `scipy`.
  - Forecasting: `pmdarima`, `prophet`, `torch`, `pytorch-forecasting` and `lightning`.
  - Figures and maps: `matplotlib` and `geopandas`.

---

## 10. Deviations from the plan (summary)

The full log with dates is `analysis_plan_rerun.md` §9. Items decided after results had been seen are marked.

| Stage | Deviation | Results seen first? |
|---|---|---|
| 1 | Fixed outcome (A1) with S1 and S2 bounds | Indicative prevalence only |
| 1 | Prepayment flag routing fix | No |
| 2 | Realised stress = month *m*−1, same past-only moments | No |
| 2 | Growth-only FES added, then made primary (PI coverage 28–48%) | Forecast diagnostics only |
| 2 | 2009 interviews excluded from FES models (first origin Dec 2009) | No |
| 2/5 | Interview timing from the actual interview date, not the sample month | No |
| 3/4 | Extra missing codes; `sf1_good` from `scsf1` | No |
| 3 | Strain components entered separately; `finfut_risk` read as financial expectations | Item correlations and α |
| 3 | Region FE, oil and rural terms; NI-oil sequence; AMEs instead of log-odds shares | Descriptive NI rates; Stage 3 ORs for the AME decision |
| 3 | `sens_no_qualification` added | Yes (primary model) |
| 3 | Two-way clustering on PSU × interview year-month | No |
| 4 | Monetary items log-transformed | No |
| 4 | Composites described as formative indices; hotspot tiers dropped | CFA and map seen |
| 4 | H1 direction corrected (positive = buffering) | Stage 3 main effect only |
| 5 | Disability variable replaced (`healthlink` → `health` + `disdif`) | v1 rates only |
| 5 | JRF windows defined; region/ethnicity window corrected to Apr 2021–Mar 2023 | Yes (earlier window) |
| all | Small-cell suppression on all tracked tables | n/a |
| 6 | P0 as benchmark only; strain components; FES term and timing; training-only standardisation; common sample (A6) | Stages 3–5 seen; no Stage 6 model |
| 6 | P3 = P0 + P1, post-hoc | **Yes** |
| 6 | H5 verdict wording "partially supported" | Yes |
| 7 | v1 CVAE, fuzzy, SVM to the appendix as archived results; descriptive refreshes | Yes |
