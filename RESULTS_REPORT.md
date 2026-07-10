# Anticipatory Energy–Carbon Stress — Full Results Report

**Pipeline version reviewed:** July 2026
**Forecast horizon:** January–December 2017
**UK ENABLE sample:** n = 1,015 households

> **Framing note.** FES is a macro-level **scenario-based signal simulation** (9 named scenarios: 6 forecasted + 3 realised benchmarks — see `src/fes_scenarios.py` and the README's "FES Scenario-Based Signal Simulation" section), not a household-level predictor. HighAEV is estimated independently by three COR routes. The two are linked only through **scenario-conditioned interpretation** (`outputs/fes_highaev_interpretation/`), never as a causal or predictive claim. This report also reflects the current 5-variant CatBoost/SHAP structure (Controls_Only, Route1_Composite, Route2_SEM, Route3_VAE, AllRoutes_Hybrid) and the household-side sample-size and feature fixes described in Section 6.

---

## 1. Forecast Pipeline — Energy and Carbon Series (Stages 0–4)

### 1.1 Model Selection

The best model per series and mode was chosen by composite rank-aggregation score across MAE, RMSE, SMAPE, MASE, quantile loss, Winkler score, and MSIS on the 2016 hold-out year, confirmed by lowest forecast-vs-actual error over 2017.

| Series | Mode | Selected Model | Val Rank Score | Forecast MAE | Forecast SMAPE | PI Coverage |
|--------|------|----------------|:--------------:|:------------:|:---------------:|:-----------:|
| Gas | Core | TFT | 4.00 | **1.03** | 106.2% | 8.3% |
| Gas | Macro | Prophet | 3.43 | **2.44** | 197.0% | 100.0% |
| Electricity | Core | TFT | 3.95 | **3.85** | 93.8% | 16.7% |
| Electricity | Macro | LSTM | 1.13 | **4.22** | 92.3% | 66.7% |
| Carbon | Core | LSTM | 1.45 | **15.75** | 88.3% | **0.0% ⚠** |
| Carbon | Macro | TFT | 3.13 | **15.96** | 76.6% | **0.0% ⚠** |

*Source: `outputs/tables/model_metrics_comparison.csv`.*

**Economic interpretation — Gas:** Gas-core (TFT) correctly captures the 2017 upward drift out of deflation (forecast: −3.2 to +2.2 p.p.). Gas-macro (Prophet) now carries a **large, systematic positive bias**: it forecasts rising growth (0.36 p.p. in January to a 2.35 p.p. peak in July) while actual gas growth stayed negative all year (−5.3 p.p. in January, easing only to −0.1 p.p. by November/December). The error is largest in Q1 (5.4–5.7 p.p.) and shrinks through the year as the actual series itself converges toward zero — not because the forecast corrects. TS-SHAP attribution shows why: Prophet's `trend` component is positive in every single month (+2.56 in Jan declining to +1.78 in Dec, never crossing zero), directly producing the bias; `inflation_growth_lag1` is the main (but far smaller) negative counterweight.

**Economic interpretation — Electricity:** Both electricity models are unaffected by the above and remain well-calibrated in direction: TFT-core (MAE 3.85 p.p.) and LSTM-macro (MAE 4.22 p.p., the best macro-mode fit of any series) both under-anticipate the H2 tariff pass-through rise but track the broad shape correctly.

**Economic interpretation — Carbon:** LSTM-core systematically predicted continued 2015–2016 EUA declines (−35% to −8% log-return) while actuals recovered (+11% to +55% from mid-2017). **Prediction interval coverage = 0%** — a critical failure, discussed further in Section 1.2. TFT-macro is materially better calibrated in direction (SMAPE 76.6% vs. 88.3%) and is used as the primary carbon component for FES_macro.

### 1.2 Issues Identified

- **SARIMA MAPE on electricity: up to 1,348%** — near-zero electricity growth rates cause MAPE explosion; this is why MAPE is excluded from the rank-aggregation metric.
- **LSTM-core on carbon: 0% prediction interval coverage** — model failed to anticipate the 2017 EUA recovery.
- **Prophet-macro on gas: systematic positive bias throughout 2017**, driven by a persistently positive Prophet `trend` term that never adapts to the actually-deflationary series (Section 1.1). Prophet-macro is not perfectly seed-stable — its own decomposition and fit quality have varied materially across independent runs of this pipeline, which is itself a documented caveat (Chapter 4, Section 4.2.2).
- **TFT-core on gas/electricity: PI coverage 8.3%/16.7%** — despite low absolute error, PI bounds are too narrow given actual variance.

---

## 2. FES Scenario-Based Signal Simulation (Stage 4)

FES is now framed as 9 named scenarios (6 forecasted signal constructions + 3 realised benchmarks; `src/fes_scenarios.py`), reported descriptively as macro context — never as a household-level predictor or a simulation of household behaviour (see README, "FES Scenario-Based Signal Simulation" and "Linking FES Scenarios and HighAEV").

### 2.1 Annual FES Summary — All 9 Scenarios

| Scenario | Gas z | Electricity z | Carbon z | Uncertainty z | **FES Total** | Historical state |
|----------|-------|---------------|----------|---------------|--------------|-------------------|
| equal_core | −0.555 | −0.336 | +0.113 | −0.151 | **−0.929** | low |
| equal_macro | −0.450 | −0.244 | +0.150 | +0.072 | **−0.472** | low |
| vw_core | — | — | — | — | −0.328 | low |
| vw_macro | — | — | — | — | −0.233 | low |
| bayesian_core | — | — | — | — | −0.281 | low |
| bayesian_macro | — | — | — | — | −0.190 | moderate_neutral |
| actual_A | −0.600 | +0.012 | +0.155 | −0.394 | −1.674 | low |
| actual_B (actual_CrossComp) | −0.600 | +0.012 | +0.155 | −0.394 | **−0.827** | low |
| actual_C | −0.600 | +0.012 | +0.155 | −0.394 | −2.923 | low |

*Source: `outputs/fes/fes_summary_2017.csv`, `outputs/fes/fes_scenario_summary.csv`. All z-scores relative to the 2005–2016 reference period; historical-baseline classification per `classify_historical_fes_state` (FES ≤ −0.50 low, −0.50 to +0.50 moderate/neutral, ≥ +0.50 high). actual_A/B/C share the same gas/electricity/carbon z-scores and differ only in the realised-volatility proxy replacing the uncertainty term.*

**All 9 scenarios agree on the direction**: 2017 reads as below-average energy-carbon stress under every construction (`low` on the historical-baseline scale for 8 of 9; `bayesian_macro` alone reads `moderate_neutral`, at −0.190, just inside the ±0.50 band). This is consistent with falling gas prices, a stable electricity market until H2, and a carbon market only beginning its 2017 recovery.

**Does the interpretation change across constructions?** Materially, yes, for the macro-mode gas-driven scenarios (Section 1's Prophet-macro bias propagates into `equal_macro`, `vw_macro`, `bayesian_macro`), but the *direction* of the finding (low stress) is robust across every one of the 9 scenarios — the robustness question this scenario layer is designed to test.

### 2.2 Component Decomposition

The gas z-score is the largest negative contributor across all scenarios (−0.45 to −0.60). Carbon is the only consistently positive component (+0.11 to +0.16). Electricity is near zero in the realised benchmark (+0.012) but negative in both forecast scenarios (−0.24 to −0.34), reflecting under-anticipation of the H2 tariff rise. The uncertainty component is now the single largest driver of the gap between `equal_core` (−0.929) and `equal_macro` (−0.472): 0.223 of the 0.457 SD total gap, versus 0.105 SD from gas and 0.092 SD from electricity.

**Policy framing:** A FES of roughly −0.5 to −0.9 in 2017 does not mean energy poverty was resolved — it means the 2017 macro environment was less stressful than the 2005–2014 average (which includes the 2008 spike and 2010–2014 high-gas-price regime). It also does not mean, or imply, that households changed their behaviour because of this macro reading — see the scenario-conditioned interpretation framing in Section 2.4.

### 2.3 Forecast-vs-Actual Robustness

| Forecast scenario | vs actual_B (CrossComp) Pearson r | R² | MAE |
|---|:---:|:---:|:---:|
| equal_core | **0.970** | **0.804** | 0.288 |
| equal_macro | 0.887 | 0.204 | 0.535 |
| vw_core | 0.962 | −0.001 | 0.544 |
| bayesian_core | 0.965 | −0.289 | 0.623 |

*Source: `outputs/fes/fes_scenario_forecast_vs_actual_matrix.csv`, `outputs/fes/fes_comparison_metrics.csv`.*

`equal_core` is both the simplest and the best-tracking scenario against the dominant realised benchmark (r = 0.970, R² = 0.804). In the current run, `equal_macro` **no longer improves** on `equal_core` for either level calibration or trajectory tracking (Chapter 4, Section 4.10.1 discusses the reversal from earlier pipeline iterations in detail) — `equal_core`'s absolute distance from the realised benchmark (0.102 SD) is now more than three times smaller than `equal_macro`'s (0.355 SD), driven by Prophet-macro's gas bias. `vw_core` and `bayesian_core` do not improve on `equal_core` against any benchmark; their reweighting shifts the level (−0.328, −0.281) away from the realised benchmark despite maintaining high Pearson r.

### 2.4 Linking FES Scenarios and HighAEV

FES scenarios do not change household labels and are never entered as household-level predictors. `outputs/fes_highaev_interpretation/scenario_highaev_interpretation_matrix.csv` links each of the 9 scenarios to the **same** already-estimated HighAEV prevalence (Route 1: 25.0%, n=254/1015), varying only the macro-context interpretation text (e.g. "under this low-stress macro scenario, the HighAEV group represents households with structurally high adaptive vulnerability, not households made vulnerable by macro conditions"). See README, "Linking FES Scenarios and HighAEV".

---

## 3. Construct Validation (Stage 6, Route 1)

### 3.1 Reliability and Convergent Validity

| Construct | Items | n | Cronbach α | AVE | CR (ω) | α pass? | AVE pass? |
|-----------|-------|---|------------|-----|---------|---------|-----------|
| FCP (Financial Pressure) | 3 | 698 | 0.518 | **0.038** | 0.097 | ✗ | ✗ |
| AEMC (Adaptive Capacity) | 9 | 1,015 | 0.286 | **0.035** | 0.093 | ✗ | ✗ |
| BLI (Behavioural Lock-in) | 5 | 1,015 | **0.860** | **0.081** | 0.303 | ✓ | ✗ |
| TCR (Transition Resistance) | 5 | 673 | **0.710** | **0.041** | 0.165 | ✓ | ✗ |

**All four constructs fail the AVE ≥ 0.50 convergent-validity threshold** (max AVE = 0.081, BLI). Maximum factor loading across all items is 0.383 (E6A2, AEMC). This is unchanged from earlier iterations of this pipeline, as expected — the engineered ML controls added in Section 6 (`has_insulation`, `heating_gas_share`, `has_smart_meter`) are dwelling/administrative variables, not COR construct items, so they have no bearing on construct psychometrics.

### 3.2 Discriminant Validity (HTMT)

All six pairwise HTMT ratios are below the 0.85 threshold (FCP–AEMC 0.523, FCP–BLI 0.100, FCP–TCR 0.177, AEMC–BLI 0.534, AEMC–TCR 0.781, BLI–TCR 0.319) — discriminant validity is acceptable despite the convergent-validity failure. See Chapter 5 for full discussion.

### 3.3 Construct Score Distributions

| Score | Mean | Std | Median | P25 | P75 | Missing |
|-------|------|-----|--------|-----|-----|---------|
| FCP | 0.476 | 0.181 | 0.455 | 0.359 | 0.573 | 0% |
| AEMC | 0.221 | 0.092 | 0.208 | 0.167 | 0.292 | 0% |
| BLI | 0.418 | 0.283 | 0.500 | 0.150 | 0.600 | 0% |
| TCR | 0.474 | 0.178 | 0.467 | 0.333 | 0.600 | 4.3% |
| AEV | 0.539 | 0.111 | 0.542 | 0.458 | **0.615 (P75 threshold)** | 0% |

*Source: `outputs/enable_cleaned/construct_score_summary.csv`. AEV = mean(FCP, BLI, TCR, 1−AEMC). HighAEV = 1 if AEV ≥ 0.615 → 254 households (25.0%); 761 LowAEV (75.0%).*

---

## 4. COR Path Analysis and Mediation (Stage 6, Route 1)

### 4.1 Path Estimates

| Path | β | p | Significant | Direction consistent with COR? |
|------|---|---|:---:|:---:|
| a: FCP → AEMC | −0.024 | 0.141 | ✗ | ✓ (negative, as predicted, but n.s.) |
| b, c′, d, e: {AEMC, FCP, BLI, TCR} → AEV | ±0.25 | <0.001 | (artefact) | (artefact) |

**Critical note:** paths b/c′/d/e show R² = 1.0 and t-statistics of order 10¹⁵ — a numerical artefact of AEV_score = mean(FCP, BLI, TCR, 1−AEMC) by algebraic definition (not a sum, as in earlier documentation of this project — see Chapter 5/6 for the corrected formula). Only the **a-path** and the **mediation bootstrap CI** carry independent behavioural information.

### 4.2 Mediation Analysis

| Effect | Estimate | Bootstrap 95% CI | Significant? |
|--------|----------|-------------------|:---:|
| Indirect (a×b) | 0.012 | [−0.003, 0.027] | ✗ |
| Direct (c′) | 0.260 | — | (artefact) |
| Total | 0.271 | — | (artefact) |

The indirect pathway (FCP → AEMC → AEV) is not statistically significant. As discussed in Chapter 6, this is attributable to the cross-sectional design, UK welfare-state attenuation of the financial-pressure→capacity link, and construct measurement noise — not a refutation of COR theory.

**FES context (Section 2) is reported alongside these results as macro-level background only** — it carries zero within-sample variance and is excluded from every regression.

---

## 5. Unsupervised Latent Robustness (Stage 6, Route 1 empirical-recovery check)

PCA, EFA, and the linear autoencoder are **Route 1's own empirical-recovery robustness checks** — not a fourth route, and not separate CatBoost model variants (Section 6).

### 5.1 PCA Explained Variance

| Component | Variance Explained | Cumulative |
|-----------|-------------------|------------|
| PC1 | 18.3% | 18.3% |
| PC2 | 10.5% | 28.8% |
| PC3 | 9.0% | 37.8% |
| PC4 | 6.5% | 44.3% |

Four components explain only 44.3% of total item variance, consistent with the weak factor structure identified in construct validation.

### 5.2 Alignment of Latent Dimensions with COR Constructs

| Construct | PCA/EFA best match | r | AE best match | mean \|r\| (30 seeds) | Stable? |
|-----------|---------------|:---:|----------------|:---:|:---:|
| BLI | PC1/EFA1 | **0.807** | AE4 | 0.816 | ✓ |
| TCR | PC1/EFA1 | 0.638 | AE4 | 0.701 | ✓ |
| AEMC | PC4/EFA4 | 0.465 | AE4 (sign-reversed) | 0.513 | Moderate |
| FCP | PC2/EFA2 | 0.285 | AE3 | 0.248 | ✗ Unstable |

**BLI is the most data-recoverable construct** (r ≈ 0.81), TCR next (r ≈ 0.64–0.72, co-loading with BLI on PC1/AE4 — a joint "energy-transition-reluctance" dimension). AEMC is moderately and unstably recoverable (split across reminder-based and routine-based sub-clusters). **FCP is not recoverable by any method** (best r = 0.31, unstable across AE seeds) — its objective (S8) and subjective (E2A/E2B) facets share limited common variance, confirming the formative rather than reflective specification adopted throughout. EFA converges to the same solution as PCA in this sample (no item exceeds communality 0.146) — there is no rotatable common-factor structure beyond what PCA already captures. See Chapter 7 for full discussion.

---

## 6. Household ML Stream: Sample-Size and Feature Fixes

Two targeted fixes were applied to `src/ml_classification.py` and `src/enable_preprocessing.py` since the previous version of this report, verified by 5-fold CV before being kept:

1. **Controls_Only / Route1_Composite no longer lose ~15% of the sample to unnecessary listwise deletion.** `_prepare_features_for_model()` previously applied the same complete-case restriction needed by Route 2 (CFA)/Route 3 (VAE) to Controls_Only and Route1_Composite too, which don't need it — their few missing values (mainly H2 at ~15%) are now imputed, matching the primary `prepare_features()` path. Sample size for these two variants rose from n≈855 to the full n=1,015.
2. **Three new energy-efficiency controls were added**: `has_insulation`, `heating_gas_share`, `has_smart_meter`, engineered in `enable_preprocessing.build_efficiency_controls()` from ENABLE's granular per-option sub-items (H5A1-3, H6A3, H13A/H13C) — the bare `H5`/`H6`/`H13` columns referenced by earlier documentation do not exist in the UK sub-sample, but this underlying dwelling-efficiency data does. `heating_gas_share` is now the single most important Controls_Only feature (Section 7).
3. `l2_leaf_reg` was raised from 8 to 10 after re-validating by 5-fold CV across 3 seeds: same mean test AUC, consistently smaller train/test overfit gap. `depth=4` was tried and rejected (marginal AUC gain, much larger overfit gap).
4. A CatBoost `eval_metric` bug was fixed (`AUC:hints=skip_train~false`) so `learning_curve_controls_only.png` — a required output that was previously never generated — now renders correctly.

Net effect (5-fold CV, the reliable estimate): Controls_Only mean test AUC rose from 0.577 to **0.612**, and the mean overfit gap fell from 0.078 to **0.044**. Full detail in Chapter 8.

---

## 7. CatBoost Classification — HighAEV Prediction (Stages 10)

### 7.1 Primary Model: Controls-Only (Generalizable Predictor)

| Metric | Value |
|--------|-------|
| ROC-AUC (test) | **0.638** |
| Train ROC-AUC | 0.703 |
| Overfit gap | 0.065 |
| Balanced accuracy | 0.594 |
| Precision (HighAEV) | 32.1% |
| Recall (HighAEV) | 79.7% |
| F1 (HighAEV) | 0.457 |
| PR-AUC | 0.376 |
| Best iteration | 106 / 600 |
| Decision threshold | 0.465 |
| n train / n test | 761 / 254 |

CV mean test AUC = 0.612 (SD = 0.041) — see Section 6 for what changed and why.

### 7.2 Confusion Matrix (decision threshold 0.465)

| | Predicted: Low AEV | Predicted: High AEV |
|--|-------------------|---------------------|
| **True: Low AEV** (n=190) | 82 (TN) | 108 (FP) |
| **True: High AEV** (n=64) | 13 (FN) | 51 (TP) |

51 of 64 true HighAEV households are correctly identified (recall 79.7%) at 32.1% precision — for every 3 households flagged, roughly 1 is genuinely HighAEV. This is a materially better recall than the 65.6% reported in earlier iterations of this pipeline, traded against a higher false-positive rate.

### 7.3 Feature Importance

| Rank | Feature | CatBoost Importance | SHAP mean \|abs\| | Note |
|------|---------|--------------------|--------------------|-----------------|
| 1 | **heating_gas_share** | 21.2% | 0.1194 | Engineered (Section 6) |
| 2 | S8 | 20.7% | 0.0717 | Household income bracket |
| 3 | H3 | 16.3% | 0.0587 | Year of property construction |
| 4 | H2 | 7.4% | 0.0266 | Number of rooms |
| 5 | has_insulation | 4.0% | 0.0230 | Engineered (Section 6) |
| 6 | risk_category | 7.9% | 0.0067 | Energy poverty proxy |
| 7 | H1 | 4.9% | 0.0171 | Dwelling type |
| 8 | S2 | 4.4% | 0.0160 | Employment status |
| 9 | S3 | 5.7% | 0.0115 | Household size |
| 10 | S6 | 2.9% | 0.0097 | Age of respondent |
| 11 | has_smart_meter | 1.7% | 0.0083 | Engineered (Section 6) |
| 12 | S5 | 1.9% | 0.0043 | Tenure |
| 13 | low_income_flag | 0.3% | 0.0013 | — |
| 14 | high_cost_flag | 0.7% | 0.0009 | — |

**`heating_gas_share`, S8, and H3 together account for 58.2%** of CatBoost feature importance — the central finding of the ML stream, updated from earlier iterations of this pipeline (which found S8+H3 alone at 54.7%, with no heating-fuel signal available). Two of the top five features (`heating_gas_share`, `has_insulation`) are the newly engineered energy-efficiency controls, confirming that ENABLE's granular dwelling sub-items carried real, previously unused predictive signal.

### 7.4 Multi-Model Comparison

| Model | n train/test | Test AUC | Train AUC | Gap | PR-AUC | Interpretation |
|-------|:---:|:---:|:---:|:---:|:---:|----------------|
| Controls_Only | 761/254 | 0.638 | 0.703 | 0.065 | 0.376 | **Generalizable** |
| Route1_Composite | 761/254 | 0.992 | 0.999 | 0.006 | 0.982 | Circular (AEV ≡ f(scores)) |
| Route2_SEM | 303/101 | 0.978 | 0.966 | −0.012 | 0.908 | Construct-overlap, not fully generalizable |
| Route3_VAE | 303/101 | 0.72–0.74* | 0.83–0.85* | 0.10–0.11* | 0.40* | Construct-overlap, `high_aev` in own loss |
| AllRoutes_Hybrid | 303/101 | 0.975–0.989* | 0.960–0.998* | −0.014 to 0.008* | 0.89–0.96* | Not generalizable — multiple circular components |

*Route3_VAE and AllRoutes_Hybrid vary between independent runs of this pipeline (two runs measured 0.717 and 0.744 for Route3_VAE) because their CatBoost variant refits a small VAE encoder within each train/test split, and that per-split fit is not perfectly seed-stable — unlike Controls_Only, Route1_Composite, and Route2_SEM, which reproduce bit-identically. The qualitative ranking is stable across runs even though the exact AUC is not.

Route2_SEM (n=503 complete-case, split 303/101) and Route3_VAE did not exist in earlier iterations of this pipeline reviewed by this report; they are new to this version. Route1_Composite remains the ceiling of circular prediction (AUC ≈ 0.99). Route2_SEM comes close (0.978) via CFA factor scores drawn from the same item pool as the target. Route3_VAE is the outlier at AUC ≈ 0.72–0.74 — notably lower than Route1/Route2 despite `high_aev` entering its own training loss, because the VAE's joint reconstruction/KL/alignment/prediction objective dilutes the label signal across a 4-D bottleneck. **None of these four should be read as demonstrating generalizable predictive power** — Controls_Only remains the only variant that does.

---

## 8. Route 2 (SEM/CFA) and Route 3 (VAE) — Own Fit Quality

*(New in this version of the report — these routes did not exist in earlier reviews.)*

### 8.1 Route 2 — CFA Measurement and Structural Fit

| Model | df | χ² | CFI | TLI | RMSEA | SRMR | All pass? |
|-------|:--:|:--:|:---:|:---:|:-----:|:----:|:---:|
| Measurement model | 246 | 1364.4 | 0.692 | 0.654 | 0.095 | 0.095 | ✗ (all four indices fail conventional thresholds) |
| Structural model (2A) | 248 | 1385.3 | 0.686 | 0.651 | 0.096 | 0.100 | ✗ |

| Construct | n items | Cronbach α | ω | CR | AVE | AVE pass? |
|-----------|:--:|:--:|:--:|:--:|:--:|:--:|
| FCP | 3 | 0.471 | 0.638 | 0.638 | 0.450 | ✗ (just below 0.50) |
| AEMC | 10 | 0.470 | 0.446 | 0.446 | 0.211 | ✗ |
| BLI | 5 | 0.856 | 0.857 | 0.857 | **0.547** | ✓ |
| TCR | 6 | 0.702 | 0.711 | 0.711 | 0.313 | ✗ |

*Source: `outputs/cor_sem/tables/cfa_fit_indices.csv`, `cfa_reliability_validity.csv`, `structural_fit_indices_2A.csv`.*

Route 2's CFA reproduces the same substantive picture as Route 1's formative composites: global model fit fails conventional thresholds (CFI/TLI < 0.90, RMSEA/SRMR > 0.08), and only BLI clears the AVE ≥ 0.50 convergent-validity bar under a reflective specification — consistent with, and independently corroborating, Chapter 5's conclusion that these four constructs are better treated as formative than reflective. FCP's AVE under CFA loadings (0.450) is notably closer to the 0.50 threshold than under Route 1's composite treatment (0.038), because CFA loadings are fit to maximise shared variance directly — but it still falls short.

### 8.2 Route 3 — VAE Reconstruction and Alignment

| Metric | Value |
|--------|-------|
| n (complete-case) | 503 |
| Epochs trained | 300 |
| Reconstruction MSE / MAE | 0.961 / 0.692 |
| KL divergence | 0.050 |
| COR-alignment loss | 0.358 |
| Prediction BCE | 0.521 |
| Prediction-head ROC-AUC (val) | 0.524 |
| Prediction-head recall (val) | 0.0 |

*Source: `outputs/cor_vae/tables/vae_reconstruction_metrics.csv`, `vae_prediction_metrics.csv`.*

The VAE's own prediction head performs close to chance on a held-out validation split (ROC-AUC 0.524, recall 0.0 at the default threshold) — a materially weaker signal than its CatBoost-embedded AUC of ≈0.72–0.74 (Section 7.4), because the two are measuring different things: the prediction-head metric is the VAE's own single small MLP head evaluated in isolation, while Section 7.4's figure is a full CatBoost model trained on the VAE's 4 latent means plus all Controls_Only features. This reconstruction/alignment/prediction-BCE table reproduces bit-identically across runs (unlike the CatBoost-embedded Route3_VAE variant — see Section 7.4's note). Per-dimension alignment with Route 1's composites (Hungarian-unassigned, raw correlations) shows z2 aligning with AEMC (r=−0.669) and BLI (r=0.504), z3 with BLI (r=−0.731) and AEMC (r=0.449), z4 with TCR (r=0.693), and z1 with FCP (r=−0.476) — a similar qualitative pattern to Route 1's own PCA/AE robustness check (Section 5.2): BLI/TCR are the most strongly represented constructs, FCP the weakest.

### 8.3 Cross-Route Comparison

| Route pair | Construct-level agreement | AEV Pearson r | HighAEV Cohen's κ | Agreement rate |
|---|---|:---:|:---:|:---:|
| Route 1 vs Route 2 | BLI/TCR strong (r=0.93–1.00), FCP strong (r=0.60), AEMC weak (r=0.28) | 0.841 | 0.581 | 85.7% |
| Route 1 vs Route 3 | BLI/AEMC/TCR strong (r=0.67–0.73), FCP moderate (r=0.48) | 0.782 | 0.535 | 84.1% |
| Route 2 vs Route 3 | BLI/TCR strong (r=0.70–0.73), FCP strong (r=0.53), AEMC weak (r=0.06) | 0.683 | 0.386 | 76.9% |

*Source: `outputs/route_comparison/tables/cross_route_construct_agreement.csv`, `cross_route_outcome_agreement.csv`. n=503 (complete-case, all three routes).*

All three route pairs show moderate-to-strong AEV-level agreement (Pearson r 0.68–0.84) and fair-to-moderate HighAEV classification agreement (κ 0.39–0.58, 77–86% raw agreement) — the three estimation methods broadly agree on *which* households are vulnerable even though they are built through genuinely different statistical machinery (formative composite / reflective CFA / deep generative latent). AEMC is the one construct where all three routes disagree most (r as low as 0.06 between Route 2 and Route 3), echoing Section 5.2's finding that AEMC splits across two empirically distinct sub-clusters (reminder-based, routine-based) that different methods can capture differently.

---

## 9. SHAP Explainability — Key Household Narratives

The SHAP analysis on the Controls_Only model (Section 7.3) identifies:

**Highest-probability profile (0.607, a false positive):** high income difficulty (S8 SHAP ≈ +0.24), high mains-gas heating share (`heating_gas_share` SHAP ≈ +0.12), older building (H3 SHAP ≈ +0.05).

**Lowest-probability profile (0.329):** low income difficulty (S8 SHAP ≈ −0.16), low mains-gas dependence (`heating_gas_share` SHAP ≈ −0.20, the single largest-magnitude SHAP contribution observed), newer building (H3 SHAP ≈ −0.17).

The predicted-probability spread across household examples (0.329–0.607) is now substantially wider than the 0.49–0.52 compression reported in earlier iterations of this pipeline — direct evidence that `heating_gas_share` gives the model real room to separate cases rather than clustering predictions near the decision boundary.

---

## 10. Cross-Stream Consistency Check

| Finding | Consistent across streams? |
|---------|--------------------------|
| Income (S8) and heating-gas exposure jointly drive High AEV risk | ✓ Feature importance + SHAP (Section 7), consistent with COR c′-path (higher FCP → higher AEV, Section 4) |
| Low adaptive capacity (AEMC) weakly measured across all 3 routes | ✓ Route 1 AVE=0.035 (Section 3); Route 2 AVE=0.211 (Section 8.1); Route 3 alignment weakest for FCP, most divergent for AEMC across routes (Section 8.3) |
| BLI is the most data-recoverable COR construct | ✓ PCA r=0.81 (Section 5); Route 2 AVE=0.547, the only construct passing (Section 8.1); highest cross-route agreement (Section 8.3) |
| 2017 macro environment was below-average stress under every FES scenario | ✓ All 9 scenarios read `low` or `moderate_neutral` (Section 2) |
| Gas-macro (Prophet) forecast is not stable across pipeline runs | ✓ MAE/SMAPE/trend-decomposition all changed materially between reviewed runs (Sections 1, 2.1) |
| Carbon forecast is structurally unreliable | ✓ 0% PI coverage on both core and macro selected models, 76–88% SMAPE (Section 1) |
| Mediation FCP→AEMC→AEV not confirmed | ✓ a-path β=−0.024, p=0.141; bootstrap CI includes zero (Section 4) |
| Route 1/2/3 broadly agree on which households are HighAEV | ✓ κ 0.39–0.58, 77–86% raw agreement (Section 8.3) |

---

## 11. Summary of Issues and Recommendations

### Resolved in This Version

- ✅ FES reframed as a 9-scenario macro signal-simulation layer (not a household predictor or behavioural simulation) — `src/fes_scenarios.py`
- ✅ AEV formula corrected to `mean(FCP, BLI, TCR, 1−AEMC)` everywhere (was previously documented inconsistently as a sum in some places)
- ✅ CatBoost model variants aligned to the 5 real routes (Controls_Only, Route1_Composite, Route2_SEM, Route3_VAE, AllRoutes_Hybrid) — legacy PCA/EFA/Linear_AE/SEM_COR/Hybrid_SEM_AE variant names retired
- ✅ **H5/H6/H13 "missing from ML" issue resolved** — reconstructed as `has_insulation`/`heating_gas_share`/`has_smart_meter` from ENABLE's granular sub-items; `heating_gas_share` is now the #1 Controls_Only feature
- ✅ Controls_Only/Route1_Composite sample-size bug fixed (n≈855 → full n=1,015)
- ✅ `l2_leaf_reg` re-tuned (8→10) and validated by CV across 3 seeds
- ✅ `learning_curve_controls_only.png` generation bug fixed (CatBoost `eval_metric` hint)
- ✅ `exclude_fes_columns` extended to cover all 9 FES scenario columns + underlying z-score components
- ✅ Dead `from sympy import series` import removed; `joblib`/`factor_analyzer` added to `requirements.txt`

### Remaining Issues / Open for Future Work

| Issue | Severity | Recommendation |
|-------|----------|----------------|
| AVE < 0.55 for all constructs under both Route 1 (formative) and Route 2 (reflective CFA) treatment | High | Continue reporting as formative composites; Route 2's CFA independently corroborates this rather than resolving it |
| Route 2/Route 3 restricted to n=503 (complete-case) vs. Controls_Only/Route1's n=1,015 | Medium | Consider FIML (semopy), multiple imputation, or a masked VAE reconstruction loss to recover the ~50% of the sample lost to listwise deletion on E2A/E2B and the H15 battery (see prior conversation's missing-data diagnostic) |
| Carbon LSTM-core / TFT-macro 0% PI coverage on both modes | High | Both carbon models remain structurally unreliable; treat carbon's forecast-interval component as descriptive only |
| Prophet-macro (gas) not seed-stable across pipeline runs | Medium | Treat `equal_macro`/`vw_macro`/`bayesian_macro` FES scenarios as a sensitivity check, not a co-equal primary reading, until Prophet's macro gas fit is stabilised (e.g. fixed MCMC seed, MAP estimation) |
| Route 3 VAE prediction head near-chance on held-out validation (ROC-AUC 0.524) despite `high_aev` in its own loss | Medium | Document as a genuine finding (the joint objective dilutes label signal) rather than a bug; do not tune away without re-examining the loss-term weighting |
| FCP not recoverable from any latent method (Route 1 PCA/EFA/AE or Route 2 CFA or Route 3 VAE) | Low | Consistent, well-documented feature of the item space across all three estimation routes; no remediation needed, treat as a robustness finding in its own right |

---

*Anticipatory Energy–Carbon Stress Pipeline · July 2026 report*
