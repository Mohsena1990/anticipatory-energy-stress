# Methodology

**Project:** Anticipatory Fuel Stress Watch (AFSW) — Forecasting Anticipatory Energy–Carbon Stress and Household Fuel Vulnerability in the UK
**Scope:** a complete, reproducible methods description for every stage of the pipeline — exact model specifications, hyperparameters, statistical formulas, validation designs, and software — written at the level of detail a journal Methods section or an independent replication would require. Every method is cited to its implementing source file and function.

---

## 1. Overview

The pipeline runs in five stages, each building on the previous one's output:

1. **Stage 1 (`forecast_pipeline.py`, `src/fes_calculator.py`)** — forecasts UK gas, electricity, and carbon price growth using four model families in two information modes, then combines them into a composite Forecasted Energy Stress (FES) index.
2. **Stage 2 (`src/ukhls_preprocessing.py`, `src/ukhls_cor_sem.py`, `src/ukhls_cor_cvae.py`)** — builds the UKHLS household-wave panel, attaches the FES signal at household-month resolution, and estimates household resource stock two independent ways under Hobfoll's Conservation of Resources (COR) theory.
3. **Stage 3 (`src/ukhls_vulnerability_classification.py`)** — identifies household vulnerability via unsupervised methods, validates it against an objective threshold, and fits an interpretable driver model.
4. **Stage 4 (`src/ukhls_policy_maps.py`, `src/ukhls_geo_maps.py`)** — renders the Stage 3 results onto real UK administrative boundaries as policy-geography maps.
5. **Stage 5 (`src/ukhls_forward_prediction.py`)** — links households across consecutive panel waves and walk-forward validates a model that predicts *next-wave* vulnerability from *current-wave* information only.

A sixth, cross-cutting component (`src/ukhls_external_validation.py`) benchmarks Stage 3's outputs against an independent national statistic. Reproducibility constants (random seeds, output resolution) are centralized in `src/config.py`; the full raw/processed data inventory is described separately in `reports/05_data_description.md`.

---

## 2. Stage 1 — Macro Forecasting and the FES Index

### 2.1 Forecasting models

Four model families are fit independently for each of three price series (gas, electricity, carbon growth, % year-on-year) in two information modes — **core** (univariate, target series only) and **macro** (target series plus exogenous macro-economic regressors) — giving 4 × 2 × 3 = 24 model/mode/series combinations per forecast year.

- **SARIMA(X)** (`src/models/sarima_model.py`): order selection via `pmdarima.auto_arima` with `seasonal=True` and a seasonal period of 12 months; falls back to a fixed `(1,1,1)(1,1,1,12)` specification if automatic order search fails to converge. Core mode fits on the univariate series alone; macro mode adds exogenous regressors as SARIMAX covariates.
- **Prophet** (`src/models/prophet_model.py`): Facebook/Meta's additive decomposition model, with `yearly_seasonality=True`, `weekly_seasonality=False`, `daily_seasonality=False`, `seasonality_mode` defaulting to additive, `seasonality_prior_scale=10.0` (Prophet's own default). In macro mode, exogenous regressors are added via Prophet's native `add_regressor()` interface with a deliberately tightened Laplace prior scale of **0.5** (rather than Prophet's own default of ~10.0), chosen because macro regressors can go out-of-distribution in the forecast window and an untightened prior allows Prophet to over-fit regressor coefficients to their in-sample range.
- **LSTM** (`src/models/lstm_model.py`): a 2-layer stacked LSTM (`LSTM(128) → Dropout(0.2) → LSTM(64) → Dropout(0.2)` followed by a dense output layer), trained with a 12-month lookback window, batch size 16, learning rate 1×10⁻³, for 100 epochs (30 in `--fast` development mode). Prediction intervals are produced via **Monte Carlo dropout** — 200 stochastic forward passes at inference time with dropout layers kept active — from which the empirical 2.5th/97.5th percentiles form a 95% interval.
- **Temporal Fusion Transformer (TFT)** (`src/models/tft_model.py`, via `pytorch-forecasting`'s `TemporalFusionTransformer` and `lightning.pytorch.Trainer`): 2 LSTM encoder/decoder layers, hidden size 32, 4 attention heads, dropout 0.1, attention dropout 0.1, hidden continuous size 16. The model fits on the training split, is evaluated against the genuine held-out backtest period, then is refit on the full available history before producing the genuine out-of-sample forecast.

All four families are fit fresh for every forecast year in the rolling walk-forward design (Section 2.3) — the "best model" for a given series/mode is not fixed across the whole timeline but re-selected each year.

### 2.2 The FES formula

For a given forecast target month *t*, the equal-weighted Forecasted Energy Stress index is:

```
FES_t = z(GasGrowth_t) + z(ElecGrowth_t) + z(CarbonGrowth_t) + z(Uncertainty_t)
```

where each `z(·)` is standardized against the training-period mean and standard deviation for that series (`src/fes_calculator.py::_training_stats`, `_zscore_array`), and `Uncertainty_t` is the forecast prediction interval's half-width, itself z-scored against the training period's own rolling 12-month standard deviation of interval half-widths. Four variants of this index are computed and compared:

- **FES_core** — using each series' selected core-mode model's forecast.
- **FES_macro** — using each series' selected macro-mode model's forecast.
- **FES_selected** — for each series *independently*, whichever of core/macro has the lower `forecast_actual_MAE` (realised-accuracy) is used; the resulting per-series best-of-core/macro contributions are then combined into one composite index (`src/fes_calculator.py::_select_best_mode_per_series`).
- **FES_actual** — using realised (not forecast) growth values for the same series, with the uncertainty term replaced by the cross-sectional standard deviation of the three series' realised z-scores at each month (`RealVol_t = std(z_gas_t, z_elec_t, z_carbon_t)`) — a genuine "how much did the three series actually disagree this month" measure, since no forecast-interval analogue exists for realised data. Three alternative realised-volatility proxies (`fes_actual_A/B/C`: rolling 3-month std, cross-sectional std, and absolute deviation from the training-period historical mean) are computed for cross-validation of the benchmark itself.
- **FES_actual, prior year** — the same FES_actual construction applied to the calendar year immediately preceding the forecast target year, providing a "what the index actually was last year" reference point alongside the current year's forecast (`src/fes_calculator.py::_compute_actual_fes_for_window`).

### 2.3 Two run modes

- **Single-year** (`forecast_pipeline.run`): trains through `target_year − 1`'s December and forecasts `target_year`, defaulting to `src.config.DEFAULT_TARGET_YEAR` (2025) — pinned to a fixed year rather than a fully dynamic "one year ahead of the latest available data" default, specifically because the raw price series update independently of, and faster than, the UKHLS social-science panel this project is built around (which currently covers interview years only through 2024); an auto-advancing default would silently race ahead of what the household panel can use.
- **Rolling walk-forward** (`forecast_pipeline.run_rolling`): for every feasible year *Y* (sufficient training history before *Y*, complete realised data through *Y* available), retrains all 24 model/mode/series combinations on data through *Y* and forecasts *Y*+1, re-selecting the best model fresh each year rather than fixing one model across the whole timeline. This is the methodologically preferred path for any reported result, since it is the only design that produces genuinely out-of-sample, year-varying forecasts rather than a single static snapshot.

### 2.4 Model selection: two evaluation regimes, explicitly distinguished

Two accuracy regimes are computed and reported separately, and this distinction is treated as a first-class methodological point rather than collapsed into one number: (1) **backtest accuracy** (`MAE, RMSE, MAPE, SMAPE, MASE, QuantileLoss, WinklerScore, MSIS, PredictionIntervalCoverage`, summarized into a `rank_score`), computed on a held-out validation window using only pre-target-year information; and (2) **realised accuracy** (`forecast_actual_MAE/RMSE/SMAPE`), computed after the target year's true values are known, and used as the actual model-selection criterion via `selection_score` (rank 1=best, 4=worst, within each series/mode group). This project's own analysis (`reports/01_outputs_catalog.md`, `reports/02_findings_report.md` Section 2) documents that these two regimes disagree in 4 of 6 series/mode cells in the reported run, and reports this disagreement explicitly as a methodological limitation of the realised-accuracy selection criterion rather than omitting it.

---

## 3. Stage 2 — Household Panel Construction and Resource Modelling

### 3.1 Panel construction

The UKHLS household-wave panel is built by column-selective reads of each wave's Stata (`.dta`) `hhresp` (household) and `indresp` (individual) files (`src/ukhls_preprocessing.py::load_wave_hhresp`, `load_wave_indresp_aggregated`), for waves a through o (2009–2024). Individual-level items are aggregated to the household level by taking the mean across responding adults for ordinal/continuous items (e.g. `health_good`, `qfhigh_band`), with categorical items (ethnicity, disability-adjacent raw codes) excluded from this mean-aggregation and instead handled by dedicated recode logic (Section 3.2). UKHLS's negative sentinel missing-value codes (`MISSING_CODES = [-9, -8, -7, -2, -1]`) are replaced with `NaN` throughout (`_recode_missing`) before any further computation. Interview year and month are derived from each wave's own `month` fieldwork-timing variable, since UKHLS fieldwork for a single wave spans roughly 24 months and therefore straddles two calendar years.

### 3.2 The fuel-to-income target

`fuel_to_income_ratio` (`src/ukhls_preprocessing.py::compute_fuel_to_income`) is constructed as total annual fuel spend (combined gas+electricity bill if the household reports a dual-fuel/combined bill, or separate gas+electricity spend if reported separately, plus oil and other off-grid fuel spend) divided by 12× reported net monthly household income. Two data-quality guards are applied: rows with implausibly low annual income (<£1,200) have their ratio set to missing rather than producing an extreme, artifactual ratio; and the ratio is winsorized (capped) at 1.0 for the ~0.27% of rows exceeding 100% of income spent on fuel, rather than dropped, so these genuinely severe-hardship households remain in the sample as "very high" rather than becoming outlier-driving extreme values. `high_fuel_vulnerable` = 1 if this ratio is ≥10% — the UK's official, government-standard fuel-poverty threshold, chosen deliberately over a survey-derived composite so the target is objective and externally interpretable.

### 3.3 Ethnicity and disability construction (this analysis's extension)

**Ethnicity** is attributed via the household reference person's `racel_dv` (asked once at a person's UKHLS entry wave and carried forward by Understanding Society's own derived-variable logic for continuing sample members), read directly from `indresp` and joined onto the household panel via `hrpid` — not household-mean-aggregated, since it is categorical rather than ordinal (`src/ukhls_preprocessing.py::load_wave_hrp_ethnicity`). Raw codes are regrouped into 11 categories aligned to the Joseph Rowntree Foundation's own published ethnicity groupings (`src/ukhls_mapping.py::ETHNICITY_GROUP_RECODE`), verified against the raw files' own Stata value labels rather than assumed from memory.

**Disability** is constructed as an Equality-Act-2010-style flag: `health==1` (has a long-standing illness or disability) **and** `healthlink` indicates the condition limits daily activities "a lot" or "a little" (`healthlink` is only asked of respondents with `health==1`, so its absence for `health==2` respondents reflects genuine survey routing, not missing data). This is a stricter construct than the pre-existing `health_good` proxy (which only captures illness presence, not activity limitation), reversed to `disability_free` (higher = better) to match the project's existing sign convention for resource-adjacent variables.

### 3.4 FES Magnitude, Current, and Delta

`src/ukhls_preprocessing.py::attach_fes_delta` attaches three household-level FES-derived signals, each with a documented resolution/fallback hierarchy:

- **FES Magnitude** — ideally, the Stage 1 rolling-walk-forward forecast for this household's own `(interview_year, interview_month)` one year ahead, i.e. genuinely ex-ante information available as of the household's interview. Falls back to that interview year's annual mean if month-level matching fails, then to a single constant (the mean of the single-year path's `FES_selected` variant) if the rolling table has not been computed for the current run — each fallback stage is logged explicitly so the active resolution is never silently assumed.
- **FES Current** — the sum of z-scores of the household's own realised gas/electricity/carbon growth for its exact interview year and month, standardized against each series' full-history monthly mean/standard deviation (not an annual-mean-of-means, since that would understate genuine month-to-month variance).
- **FES Delta = Magnitude − Current** — the genuinely anticipatory "shock coming" signal, varying by household-wave. Two explicit approximations are documented rather than hidden: Magnitude and Current use different z-scoring reference windows (rolling-year training window vs. full-history monthly mean/std), and Magnitude sums four z-terms (three series plus uncertainty) while Current sums three (no realised analogue of forecast uncertainty exists) — Delta is therefore an honest, documented approximation of shock magnitude, not an exact matched-scale subtraction.
- **`fes_actual_prior_year`** — a separate, additional constant: the realised FES_actual value for the year immediately preceding the forecast target window, attached identically to every household row, providing an available reference point distinct from Magnitude/Current/Delta.

### 3.5 COR-SEM (confirmatory factor analysis)

A four-factor measurement model is specified (`src/ukhls_cor_sem.py`, via `semopy`) operationalizing Hobfoll's four COR resource dimensions:

```
OBJECT    =~ hsrooms + hsbeds + ncars + carval + hsval
CONDITION =~ tenure_security + jbstat_security + bill_security
PERSONAL  =~ health_good + sf1_good + qfhigh_band
ENERGY    =~ fihhmnnet1_dv + fiyrinvinc_dv
BASELINE  =~ OBJECT + CONDITION + PERSONAL + ENERGY   (second-order model)
```

The model is fit via **`semopy.Model.fit(..., obj="FIML")`** (full-information maximum likelihood) rather than listwise-deletion maximum likelihood, specifically to handle items that are *structurally* (not randomly) missing — `hsval`/`carval` for non-owners/non-car-owners, most notably — without discarding those households from the analysis entirely. A known tool limitation is documented and empirically verified: `semopy` 2.3.11's FIML fit-statistic computation produces CFI/TLI outside their valid [0,1] range regardless of specification changes, verified by comparison against the same specification fit via listwise-deletion `obj="MLW"` on a complete-case subsample, which does not exhibit the anomaly — isolating the issue to FIML's fit-statistic computation specifically, not to the measurement model's substantive specification. Item loadings and the downstream FES-moderation regression, not CFI/TLI/SRMR, are treated as the primary evidence for measurement validity as a result.

Several derived items feeding this model are recoded from raw ordinal/nominal UKHLS codes into monotonic scales prior to fitting: `tenure_dv` → `tenure_security` (ordinal, `TENURE_SECURITY_RECODE`, e.g. owned outright=1.0 down to other rented=0.2); `jbstat` (current economic activity) → `jbstat_security` (ordinal, `JBSTAT_SECURITY_RECODE`, e.g. paid employment=1.0, self-employed=0.8, down through unemployment/long-term sickness at the low end, with the ambiguous "doing something else" category left as missing rather than arbitrarily scored); `health` (long-standing illness, reverse-coded to `health_good`) and `sf1` (self-rated health, reverse-scaled to [0,1] as `sf1_good`).

The FES-moderation hypothesis is tested via a separate OLS regression (`src/ukhls_cor_sem.py::test_fes_moderation`):

```
fuel_to_income_ratio ~ baseline_score + fes_delta + baseline_score × fes_delta
```

with the interaction term (`baseline_x_fes`) constituting the direct, falsifiable test of COR's loss-spiral/moderation mechanism (see `reports/04_journal_submission_materials.md` H1 for the full result).

### 3.6 COR-CVAE (conditional variational autoencoder)

A second, independent estimate of household resource structure is produced via a FES-conditioned Conditional Variational Autoencoder (`src/ukhls_cor_cvae.py`):

```
encoder(items, FES Delta) → μ, log σ²        (4-dimensional latent bottleneck)
z = μ + σ · ε,  ε ~ N(0, I)                   (reparameterization trick)
decoder(z, FES Delta) → item reconstruction
predictor(z) → high_fuel_vulnerable (sigmoid head)

Loss = recon_MSE + β·KL(q(z|x) ‖ p(z)) + λ·align(μ, SEM factor scores) + γ·BCE(predictor(z), label)
```

Architecture: encoder/decoder hidden layer widths (16, 8); latent dimensionality 4 (matching, and explicitly aligned to, the SEM's four first-order factors); loss weights β=1.0 (KL), λ=1.0 (alignment to the SEM's own factor scores, encouraging but not forcing the CVAE's latent dimensions to correspond to Object/Condition/Personal/Energy), γ=1.0 (prediction head); trained for 300 epochs at learning rate 1×10⁻³ with a 20% validation split, on TensorFlow/Keras with a fixed random seed (`TF_SEED=42`) for reproducibility. Unlike the SEM's FIML handling of missingness, the CVAE requires complete cases and is therefore trained on a smaller subsample (253,913 of 339,201 rows). `fes_magnitude` itself (a constant in any single-year run before a rolling table exists) is deliberately excluded from the conditioning input — only `fes_delta` is used — since a constant input would produce a zero-variance standardization and a resulting `NaN` in the encoder's input scaling.

The **counterfactual FES-shift simulation** re-encodes each household's real item vector twice — once conditioned on that household's own realised `fes_current`, once conditioned on the shared forecast `fes_magnitude` — through the same fitted encoder/scaler, and reports the resulting shift in the predictor head's output probability and in each household's Mahalanobis-style distance from a "resilient" reference anchor (the mean latent position of households in the bottom quartile of `fuel_to_income_ratio`, per `RESILIENT_QUANTILE=0.25`). This is explicitly documented and labelled as a *simulation* on the trained model, not an observation — no household was actually surveyed under both price regimes.

---

## 4. Stage 3 — Vulnerability Identification and Driver Analysis

### 4.1 Unsupervised vulnerability identification

Two independent, continuous (not hard-binary) methods are fit on the combined SEM factor scores, CVAE latent scores, and FES Delta:

- **Fuzzy c-means** (`skfuzzy.cluster.cmeans`, `N_FUZZY_CLUSTERS=3`, fuzziness exponent `FUZZY_M=2.0`, the standard default), producing three soft cluster-membership scores per household. Cluster identity is not fixed by the algorithm's internal ordering, so the "Resource Depleted" cluster is identified *post hoc*, per run, as whichever of the three clusters' membership correlates most positively with the objective `fuel_to_income_ratio` — and, symmetrically, "Resource Resilient" as whichever correlates most negatively — rather than assumed from a fixed cluster index.
- **One-class SVM** (`sklearn.svm.OneClassSVM`, RBF kernel, `nu=0.1`), fit only on a "resilient reference group" (households in the bottom quartile of `fuel_to_income_ratio`, `RESILIENT_QUANTILE=0.25`) in CVAE latent space, then scoring every household by distance from that learned boundary; higher anomaly score = further from the resilient profile.

Both methods are validated against the objective `high_fuel_vulnerable`/`fuel_to_income_ratio` target via Pearson correlation, Spearman correlation, and (for the binary target) AUC — i.e. validated against an outcome neither unsupervised method was given as a training label, a genuine external check rather than a circular one.

### 4.2 Driver analysis

A logistic regression (`statsmodels.api.Logit`, with a constant term via `sm.add_constant`) of `high_fuel_vulnerable` on 13 covariates (housing/tenure security, employment security, health, education, household size proxies, financial strain, and `fes_delta`) is fit on complete cases, reporting coefficients, odds ratios (`exp(coefficient)`), standard errors, and Wald p-values for every covariate (`src/ukhls_vulnerability_classification.py::_fit_driver_logit`). This model was chosen deliberately over a black-box alternative (e.g. gradient-boosted trees with SHAP feature importance) specifically so that every reported effect is a directly interpretable odds ratio with a formal significance test, at some cost in potential predictive accuracy relative to a more flexible model — an explicit precision/interpretability trade-off appropriate for a driver-analysis (explanatory) rather than a pure-prediction task. The identical specification is re-fit independently within each of the 12 UK regions' subsamples to test for geographic heterogeneity in driver effect sizes, without pooling or partial-pooling across regions (a fixed-effects-by-region design, not a hierarchical/random-effects one).

### 4.3 The financial/psychological strain composite

`financial_strain_score` (`src/ukhls_preprocessing.py`, `src/ukhls_mapping.py::COMPOSITE_ITEM_SPECS`) is constructed by combining several subjective-strain items onto a common direction and scale (`finnow`, subjective financial situation, already 1–5 ordinal with higher = more pressure; `finfut_risk`, a recoded version of the non-ordinal `finfut` variable, mapped to 0/0.5/1 with 1 = expects to be worse off next year; and the GHQ-12-derived psychological distress scores `scghq1_dv`/`scghq2_dv`, higher = worse wellbeing on both), then combined into a single composite — a pragmatic simplification adopted because UKHLS does not carry an item battery rich enough to support fully separate BLI/TCR-style sub-constructs the way the project's earlier, since-removed ENABLE-based architecture attempted.

---

## 5. Stage 4 — Policy Geography Maps

All choropleth maps are drawn on real UK NUTS1-level administrative boundaries (`data/geo/uk_nuts1_regions.geojson`, sourced from the ONS Open Geography Portal's public ArcGIS FeatureServer under the Open Government Licence v3.0), via a shared helper (`src/ukhls_geo_maps.py::plot_choropleth`) reused across Stages 3, 4, and 5 for consistency. Region-name matching between the panel's `gor_dv`-derived labels and the boundary file's own region names is logged explicitly (`n_missing` reporting) rather than silently producing a blank polygon on a name mismatch.

Three specific analyses are computed:

1. **Resource-to-Stress Hotspot classification** (`map1_resource_stress_hotspot.csv`) — each region is independently tertile-binned on (a) mean COR-SEM Baseline Resource Stock score and (b) `high_fuel_vulnerable` prevalence, producing a 3×3 bivariate policy-relevant tier (e.g. "Low resource / High vulnerability" as a cash-transfer-priority tier).
2. **Fuzzy Membership mapping** — mean "Vulnerable to Loss" fuzzy membership by region, plus the share of each region's households within `FES_BOUNDARY_ZONE=±0.10` of the 0.5 decision boundary (a "how polarized is this region's risk distribution" measure, distinct from its mean level).
3. **Vulnerability Vector Shift** — reuses the Stage 2c CVAE counterfactual simulation (Section 3.6), aggregated to the region level: each region's mean predicted-vulnerability probability under households' own realised FES exposure vs. under the shared forecast shock, visualized as a directional arrow (rising/falling risk) at each region's real geographic centroid. London's label is offset with a leader line, a standard cartographic fix since London's small polygon sits inside the South East region's boundary and a same-length arrow at its true centroid would collide with its neighbour's.

---

## 6. Stage 5 — Forward Vulnerability Prediction

### 6.1 Household linkage across waves

UKHLS's household identifier (`hidp`) is reissued whenever household composition changes, so it cannot be used to track the same household across waves. Instead, `hrpid` (the household reference person's `pidp`, present in every wave's `hhresp` file) is used: two consecutive waves' rows are linked directly where `hrpid_t == hrpid_{t+1}` (`src/ukhls_forward_prediction.py`). This linkage rate was verified empirically against the raw wave a/b files (72.5% of wave-a households link to a wave-b row, 21,886/30,169) and found consistent (72–85%) across all 14 wave-pairs — a normal UKHLS attrition/reference-person-turnover rate, not a data-processing defect, and reported as a scope limitation on the linked (not full) subsample the forward model is trained and evaluated on.

### 6.2 Transition-pair construction and features

Every successfully linked (wave *t*, wave *t*+1) household pair becomes one training/evaluation example: features are drawn from wave *t* (the four COR-SEM factor scores, `financial_strain_score`, `dvage`, `heatch`, and — the anticipatory signal — `fes_magnitude`, which for wave *t* is already the forecast for that household's own *next* year, computed using only information available as of wave *t*); the label is drawn from wave *t*+1 (`high_fuel_vulnerable`). This produces 14 transition datasets (a→b through n→o) which are concatenated for the pooled model.

### 6.3 Walk-forward validation design

The defining methodological choice of Stage 5 is a **strict walk-forward, not k-fold, validation split**: the model is trained exclusively on the 12 earliest transitions (a→b through l→m) and evaluated exclusively on the 2 most recent transitions (m→n and n→o) — data the model has never seen in any form during training, fitting, or feature construction. This is deliberately a stronger and more policy-relevant test than standard k-fold cross-validation on the pooled dataset, which would allow validation folds to be drawn from the same time period as (and hence potentially informationally entangled with) the training folds. Discrimination is reported via AUC against the true, held-out `high_fuel_vulnerable_{t+1}` outcome, and via Pearson correlation against the continuous `fuel_to_income_ratio_{t+1}` as a secondary, magnitude-sensitive check. After validation, the identical logistic specification is refit on **all 14** known transitions (maximizing use of available historical information for the final, deployed model) and applied to the single most recent wave (wave o) to produce genuinely prospective — not held-out-historical — predictions for each household's own next interview year.

---

## 7. External Validation Methodology

`src/ukhls_external_validation.py` benchmarks Stage 3's household-level results against the Joseph Rowntree Foundation's *UK Poverty 2025* report, an independently produced, government-data-derived (DWP Households Below Average Income) national statistic — not a value derived from, or shared with, this project's own dataset.

**Benchmark construction.** JRF values are hardcoded from numbers **explicitly stated in the report's text or tables** (never read from chart pixel positions), each cited to a specific page/table: Table 6, p.51 (region); p.9/42 (ethnicity, general-population rate, only categories with a stated number included); Table 8, p.67 (disability); Table 10, p.95 (tenure). Categories JRF shows only in a chart without a stated number (e.g. several smaller ethnicity groups) are deliberately excluded from the comparison rather than estimated.

**Agreement statistic.** For each of the four dimensions, this project's own vulnerability rate is merged against the corresponding JRF rate by category label, and both Spearman rank correlation and Pearson linear correlation are computed (`scipy.stats.spearmanr`/`pearsonr`).

**Outlier investigation methodology (Northern Ireland).** Rather than excluding or downweighting an outlier region on statistical grounds alone, the correlation is recomputed with and without Northern Ireland to quantify its specific influence, and — critically — an independent, data-grounded mechanistic explanation is sought and verified directly against the project's own underlying panel data before any exclusion decision is made: (a) the share of households reporting oil-heating expenditure (`xpoily > 0`) is computed by region, and (b) a controlled *within*-Northern-Ireland comparison (oil-heating vs. non-oil-heating households, same region and wave) isolates heating-fuel type from every other regional confound. Northern Ireland is excluded only from the reported correlation coefficient (a like-for-like check of construct agreement between two different hardship measures), never from any other output in the analysis, on the basis that price-cap-uncovered heating-oil exposure is a genuine, mechanistically verified, fuel-specific cost driver with no reason to be reflected in an income-based poverty measure.

**Temporal validation.** A fifth, independent check compares this project's own by-wave vulnerability trend against JRF's own qualitative account of the 2021–2024 cost-of-living crisis (a slow-moving annual relative-poverty statistic that JRF itself describes as "broadly flat" over this period, versus JRF's own faster hardship-tracking survey, which shows a sharp peak around October 2022) — testing whether this project's independently-constructed measure reproduces the *timing* of a known real-world event, not merely a plausible-looking trend.

---

## 8. Software and Reproducibility

- **Languages/frameworks:** Python throughout; `pandas`/`numpy` for data handling; `statsmodels` (logistic and OLS regression); `semopy` 2.3.11 (COR-SEM); `tensorflow`/`keras` (COR-CVAE); `torch`, `pytorch-forecasting`, `lightning` (TFT); `scikit-learn` (One-Class SVM, LSTM/TFT preprocessing utilities); `scikit-fuzzy` (fuzzy c-means); `pmdarima` (SARIMA order search); `prophet` (Prophet); `geopandas` (real-boundary choropleths); `scipy.stats` (correlation tests).
- **Reproducibility constants** (`src/config.py`): fixed random seeds across NumPy (42), TensorFlow (42), and PyTorch (42); fixed output resolution (150 DPI, PNG) for all generated figures.
- **Data provenance:** raw UKHLS Stata files are read column-selectively per wave with per-wave column-availability checks (not all variables exist in all waves), and every fallback/degraded-resolution code path (e.g. FES Magnitude's three-tier resolution fallback, Section 3.4) is logged explicitly at run time so the actual resolution used in any given run is auditable after the fact, not merely assumed from the code's intended design.

---

## 9. Descriptive Statistics and Distribution Reporting (`src/data_description_overview.py`)

A dedicated, descriptive-only module (`src/data_description_overview.py`, no modeling) computes distributional summaries and histogram figures for the two input streams that feed Stage 1's two information modes (Section 2.1): **core** (`data/processed/core_energy_carbon.csv` — gas/electricity/carbon growth) and **macro** (`data/processed/macro_controls.csv` — exogenous regressors), both built from the raw price/ONS/National Grid ESO series in `data/raw/`. Output feeds `reports/05_data_description.md` Sections 3.1–3.2 directly and is saved to `outputs/data_description/{tables,figures}/`.

**Summary statistics** (`_summary_stats`): for every base column (macro's `_lag1`/`_lag12` duplicate columns and its two regime/seasonal dummy variables are excluded, since a lagged copy of an already-summarised column and a binary flag's "distribution" add no new information beyond what Section 3.2's existing text already reports as a share) — count, count of missing, mean, standard deviation, min, 25th/50th/75th percentiles, and max via `pandas`, plus **skewness** and **excess kurtosis** via `scipy.stats.skew`/`scipy.stats.kurtosis` (both computed on complete cases per column, Fisher's convention for kurtosis so a Normal distribution has excess kurtosis 0). These two shape statistics are reported specifically because mean/std alone cannot distinguish, e.g., a heavy-tailed-but-symmetric distribution from a one-sided-outlier-driven one — `gdp_growth`'s excess kurtosis of ~83 and skew of −6 (both driven by the single 2020 COVID-lockdown observation) versus `carbon_growth`'s excess kurtosis of ~9 with near-zero skew (symmetric fat tails, not one outlier) is the clearest example, detailed in `reports/05_data_description.md` Section 3.1.

**Distribution figures** (`_plot_distribution_grid`): a histogram grid (40 bins, `matplotlib`, mean and median marked as vertical reference lines) for a representative subset of columns per stream — core's figure covers all 4 of its columns; macro's figure covers 9 columns selected as the ones most directly used as FES exogenous regressors (Section 2.1), since a histogram grid of the full ~27-column macro set is not legible at any reasonable figure size. The full numeric table (all columns, plotted or not) is always saved regardless of what the figure shows, so no column's summary statistics are silenced by the figure's necessarily narrower selection.

**Reproducibility:** output paths are centralised in `src/paths.py` (`DATA_DESC_TABLES`, `DATA_DESC_FIGURES`), following the same `outputs/<module>/{tables,figures}/` convention as every other descriptive-output module in this pipeline (e.g. `src/ukhls_dataset_overview.py`); figures are saved at 150 DPI as both `.png` and a vector `.pdf` twin, consistent with `src/config.py`'s reproducibility constants (Section 8).
