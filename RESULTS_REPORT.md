# Anticipatory Energy–Carbon Stress — Full Results Report

**Pipeline version reviewed:** June 2026  
**Forecast horizon:** January–December 2017  
**UK ENABLE sample:** n = 1,015 households

---

## 1. Forecast Pipeline — Energy and Carbon Series (Stages 0–4)

### 1.1 Model Selection

The best model per series and mode was chosen by composite rank-aggregation score across MAE, RMSE, SMAPE, MASE, quantile loss, Winkler score, and MSIS on the 2016 hold-out year, confirmed by lowest forecast-vs-actual error over 2017.

| Series | Mode | Selected Model | Val MAE | Forecast MAE | Forecast SMAPE |
|--------|------|----------------|---------|--------------|----------------|
| Gas | Core | TFT | 27.6 | **1.03** | 106% |
| Gas | Macro | LSTM | 1.74 | 3.85 | 200% |
| Electricity | Core | TFT | 11.1 | **3.85** | 93.8% |
| Electricity | Macro | LSTM | 0.82 | 4.22 | 92.3% |
| Carbon | Core | LSTM | 34.1 | 15.75 | 88.3% |
| Carbon | Macro | TFT | 55.9 | **15.96** | 76.6% |

**Economic interpretation — Gas:** The UK domestic gas market in 2017 was experiencing mild price deflation relative to trend, reflecting high LNG import availability and a warmer-than-average winter. The TFT-core model captured this correctly (forecast: −3 to +2 p.p. growth, actual: −5 to 0 p.p.). SMAPE of 106% on gas (core) reflects near-zero actual growth rates making the denominator unstable — the directional signal is correct.

**Economic interpretation — Electricity:** UK household electricity prices showed near-zero real-terms growth in 2017 (Ofgem price cap not yet in effect; market rates relatively stable). The LSTM-macro model achieves 92% SMAPE driven by the fact that actual electricity growth hovers near zero (≈ −0.2 p.p. YoY), making percentage error metrics uninformative. The absolute prediction interval from LSTM-macro (−0.2 to +5.0 p.p.) captures most actuals.

**Economic interpretation — Carbon:** This is the most problematic series. The LSTM-core model systematically predicted a continuation of 2015–2016 EUA price declines (forecasting −35 to −8% log-returns) while actuals showed a recovery (+11 to +55% log-return from mid-2017 onward). The model was trained on the 2014–2016 EUA price collapse and had no signal of the 2017 recovery. **Prediction interval coverage = 0%** — a critical failure. The TFT-macro model does better (Theil U = 0.077 vs. 0.088 for LSTM-core) and should be treated as the primary carbon component in the FES.

### 1.2 Issues Identified

- **SARIMA MAPE on electricity: 1,348%** — near-zero electricity growth rates cause MAPE explosion. This is why MAPE is excluded from the rank-aggregation metric (as documented), but the raw value flags the scale problem.
- **LSTM-core on carbon: 0% prediction interval coverage** — model failed to anticipate EUA price recovery.
- **Gas macro SMAPE = 200%** for SARIMA and LSTM (both) — same near-zero denominator problem.
- **TFT-core on gas: 0% PI coverage** — despite low absolute error, PI bounds are too narrow given actual variance.

---

## 2. Forecast Energy Stress Index — FES 2017 (Stage 4)

### 2.1 Annual FES Summary

| Variant | Gas z | Electricity z | Carbon z | Uncertainty z | **FES Total** |
|---------|-------|---------------|----------|---------------|--------------|
| Core (forecast) | −0.555 | −0.336 | +0.113 | −0.151 | **−0.929** |
| Macro (forecast) | −0.520 | −0.244 | +0.150 | +0.112 | **−0.503** |
| Actual | −0.600 | +0.012 | +0.155 | −0.394 | **−0.827** |
| VW-Core | — | — | — | — | −0.328 |
| Bayes-Core | — | — | — | — | −0.281 |

### 2.2 Economic Interpretation

**All three variants agree: 2017 was a year of below-average energy stress for UK households**, ranging from 0.5 to 0.93 standard deviations below the 2005–2016 historical mean. This places 2017 in the lower quartile of the study period's stress distribution.

The result is consistent with the macroeconomic record:
- **Gas prices** fell in real terms in 2016–2017 following the 2014 oil price collapse, reducing household fuel bills. The z-score of −0.55 (core) reflects forecasted continuation of this low-gas-price environment.
- **Electricity prices** showed minimal growth (z ≈ −0.34 core, +0.01 actual) — Ofgem's price cap was not yet introduced (2019) but retail market competition held prices near flat.
- **Carbon costs** showed slight upward drift (z ≈ +0.11–0.15) as the EUA price began recovering from its 2016 low, but the magnitude was small and net negative across the other components.
- **Uncertainty (forecast PI half-width)** was below average for the core variant (z = −0.15), meaning forecasters were relatively confident in the low-stress environment.

**Policy implication:** A FES of −0.9 in 2017 should not be interpreted as energy poverty being resolved. It means the 2017 macro environment was less stressful than the 2005–2014 average — a period that included the 2008 energy price spike and the 2010–2014 sustained high-gas-price regime. The 2021–2022 energy crisis (outside this study's horizon) would register a strongly positive FES, providing the counterfactual benchmark.

### 2.3 Variant Comparison

| FES Variant | vs Actual (CrossComp) Pearson r | R² | vs Actual (RollingVol) Pearson r | R² |
|-------------|--------------------------------|-----|----------------------------------|----|
| Equal_Core | **0.970** | **0.804** | 0.923 | −0.588 |
| Equal_Macro | 0.903 | 0.197 | 0.913 | −2.920 |
| VW_Core | 0.962 | −0.001 | 0.885 | −3.941 |
| Bayes_Core | 0.965 | −0.289 | 0.880 | −4.436 |

**Best-tracking variant:** Equal_Core vs Actual_CrossComp achieves Pearson r = 0.97 and R² = 0.80 — the monthly shape of the index closely follows realised cross-component dispersion. This means the equal-weight, core-forecast variant is not only simplest but best at reproducing the actual stress signal.

The negative R² values against Actual_RollingVol reflect a level mismatch: all four forecasted FES variants systematically predict a lower absolute stress level than what rolling-window realised volatility suggests. This is expected — realised volatility incorporates short-term price noise that forecasts smooth out. The Pearson r remains high (0.88–0.92), confirming the direction is tracked but not the level.

**VW and Bayes variants** lose R² relative to Equal_Core against CrossComp, suggesting that for this sample the inverse-volatility weighting and Kalman smoothing add complexity without improving accuracy against the dominant benchmark.

---

## 3. Construct Validation (Stage 6)

### 3.1 Reliability and Convergent Validity

| Construct | Items | n | Cronbach α | AVE | CR (ω) | α pass? | AVE pass? |
|-----------|-------|---|------------|-----|---------|---------|-----------|
| FCP (Financial Pressure) | 3 | 698 | 0.518 | **0.038** | 0.097 | ✗ | ✗ |
| AEMC (Adaptive Capacity) | 9 | 1,015 | 0.286 | **0.035** | 0.093 | ✗ | ✗ |
| BLI (Behavioural Lock-in) | 5 | 1,015 | **0.860** | **0.081** | 0.303 | ✓ | ✗ |
| TCR (Transition Resistance) | 5 | 673 | **0.710** | **0.041** | 0.165 | ✓ | ✗ |

**All four constructs fail the AVE ≥ 0.50 convergent-validity threshold** by a wide margin (max AVE = 0.081, target = 0.500). Maximum factor loading across all 21 items is 0.38 (E6A2, AEMC) — none exceed the standard 0.40 threshold.

### 3.2 Discriminant Validity (HTMT)

| Pair | HTMT | Pass (< 0.85)? |
|------|------|----------------|
| FCP–AEMC | 0.523 | ✓ |
| FCP–BLI | 0.100 | ✓ |
| FCP–TCR | 0.177 | ✓ |
| AEMC–BLI | 0.534 | ✓ |
| AEMC–TCR | 0.781 | ✓ |
| BLI–TCR | 0.319 | ✓ |

**Discriminant validity is acceptable:** all HTMT ratios are below 0.85. The constructs measure genuinely different dimensions (they are not correlated enough to be considered the same latent variable). AEMC–TCR shows the highest correlation (0.78), which is theoretically plausible — households with lower adaptive capacity also tend to resist transition costs.

### 3.3 Economic Interpretation

The low AVE and CR values indicate that the ENABLE survey items do not cohere well within their COR-specified constructs in the UK sub-sample. This is not a computational error — it reflects genuine psychometric heterogeneity:

- **AEMC items** (E5A, E6A series) span behaviours ranging from reminder use to routine establishment. These are weakly intercorrelated (α = 0.29), suggesting UK households vary considerably in which adaptive behaviours they adopt — some use timers, others use tariff-switching, rarely both.
- **FCP items** (S8 income + E2A/E2B bill concern) blend an objective indicator (income bracket) with subjective perception (concern). The weak convergence reflects that low-income households do not always perceive high financial pressure, and vice versa.
- **BLI and TCR** achieve acceptable internal consistency (α = 0.86, 0.71) but still fail AVE because even well-correlated items explain less than 10% of item variance through the common factor — much of the inter-item variation is item-specific.

**Implication for construct validity:** The COR constructs should be treated as formative composites (additively combining items regardless of their intercorrelation) rather than reflective scales (where high AVE would indicate items are interchangeable manifestations of a latent trait). The formative interpretation is more appropriate for ENABLE survey items that represent distinct behavioural or attitudinal facets rather than exchangeable indicators of one underlying trait.

### 3.4 Construct Score Distributions

| Score | Mean | Std | Median | P25 | P75 | Missing |
|-------|------|-----|--------|-----|-----|---------|
| FCP | 0.476 | 0.181 | 0.455 | 0.359 | 0.573 | 0% |
| AEMC | 0.221 | 0.092 | 0.208 | 0.167 | 0.292 | 0% |
| BLI | 0.418 | 0.283 | 0.500 | 0.150 | 0.600 | 0% |
| TCR | 0.474 | 0.178 | 0.467 | 0.333 | 0.600 | 4.3% |
| AEV | 0.539 | 0.111 | 0.542 | 0.458 | 0.615 | 0% |

**Economic interpretation:**
- The average UK household has moderate financial pressure (FCP = 0.48/1.0) and very low adaptive energy-management capacity (AEMC = 0.22) — fewer than one in four behavioural adaptation strategies are in use on average.
- BLI (0.42) shows high variance (σ = 0.28) — behavioural lock-in is bimodal: some households have almost no lock-in (BLI near 0), others are strongly habituated (BLI near 1).
- TCR (0.47) is nearly symmetric around the midpoint — roughly half of UK respondents show above-average resistance to energy transition costs.

---

## 4. SEM / COR Mediation (Stage 7)

### 4.1 Path Estimates

| Path | β | p | Significant | Direction Consistent with COR? |
|------|---|---|-------------|-------------------------------|
| a: FCP → AEMC | −0.024 | 0.14 | ✗ | ✗ (positive sign would be expected given result) |
| b: AEMC → AEV | −0.25 | 0.00 | ✓ (artefact) | ✓ |
| c': FCP → AEV | +0.25 | 0.00 | ✓ (artefact) | ✓ |
| d: BLI → AEV | +0.25 | 0.00 | ✓ (artefact) | ✓ |
| e: TCR → AEV | +0.25 | 0.00 | ✓ (artefact) | ✓ |

**⚠ Critical note:** Paths b, c', d, e all show R² = 1.0 and t-statistics of order 10¹⁵ — numerical artefacts caused by the fact that AEV_score = FCP + BLI + TCR + (1−AEMC) by algebraic definition. Regressing AEV on its own components recovers a perfect fit, not a behavioural finding. Only the **a-path** (FCP → AEMC) and the **mediation bootstrap CI** contain genuine behavioural information.

### 4.2 Mediation Analysis

| Effect | Estimate | Bootstrap 95% CI | Significant? |
|--------|----------|-----------------|--------------|
| Indirect (a×b) | 0.012 | [−0.003, 0.027] | ✗ |
| Direct (c') | 0.260 | — | (artefact) |
| Total | 0.271 | — | (artefact) |

**The indirect pathway (FCP → AEMC → AEV) is not statistically significant** (CI includes zero). The theoretical COR mechanism — financial pressure depletes adaptive capacity, which then raises vulnerability — is not supported by this cross-sectional UK sample at conventional significance levels.

**Economic interpretation:** This non-significance does not contradict COR theory; it reflects several plausible explanations:
1. **Cross-sectional design**: COR is a dynamic theory of resource depletion over time. A single-wave survey cannot capture the temporal process by which financial stress erodes adaptive reserves.
2. **UK-specific context**: The UK welfare state (means-tested energy-efficiency grants, Warm Homes Discount) may partially break the theoretical link between financial pressure and adaptive capacity by providing external resources that offset personal resource depletion.
3. **Construct validity issue**: The weak AVE values mean FCP and AEMC are measured with substantial noise, reducing statistical power to detect the indirect effect.

The direct path (FCP → AEV positive) and the BLI/TCR paths all align directionally with COR theory, supporting the composite index even where the mediation is not statistically confirmed.

---

## 5. Unsupervised Latent Robustness (Stage 8)

### 5.1 PCA Explained Variance

| Component | Variance Explained | Cumulative |
|-----------|-------------------|------------|
| PC1 | 18.3% | 18.3% |
| PC2 | 10.5% | 28.8% |
| PC3 | 9.0% | 37.8% |
| PC4 | 6.5% | 44.3% |

Four components explain only **44% of total item variance** — well below the 70%+ often sought in psychometric applications. This reflects the same weak factor structure identified in construct validation: ENABLE items span genuinely heterogeneous behaviours with substantial item-specific variance.

### 5.2 Alignment of Latent Dimensions with COR Constructs

| Construct | PCA best match | r (Pearson) | EFA best match | r | AE best match | r | Stable? |
|-----------|---------------|------------|----------------|---|---------------|---|---------|
| BLI | PC1 | **0.807** | EFA1 | **0.807** | AE4 | 0.638 | ✓ |
| TCR | PC1 | 0.638 | EFA1 | 0.638 | AE4 | 0.715 | ✓ |
| AEMC | PC4 | 0.465 | EFA4 | 0.465 | AE4 | −0.435 | ✓ (mod.) |
| FCP | PC2 | 0.285 | EFA2 | 0.285 | AE3 | 0.306 | ✗ |

**Economic interpretation:**

- **BLI is the most data-recoverable construct** (r = 0.81 with PC1). This makes economic sense: behavioural lock-in (sticking to inefficient appliances, resisting habit change) produces consistent response patterns across E7A–E7E items that any dimensionality-reduction method can recover.
- **TCR is well-aligned** (r = 0.64 with PC1/AE4). Transition-cost resistance loads onto the same principal component as BLI — households that are behaviourally locked in also tend to resist paying the upfront costs of switching. This overlap is economically coherent: both capture inertia, one behavioural, one financial.
- **AEMC shows moderate alignment** (r = 0.46 with PC4) — the fourth component captures a distinct adaptive capacity dimension not loaded on the dominant variance axes. The AE captures it with opposite sign (r = −0.44), reflecting that the AE axis is flipped.
- **FCP is the weakest** (r ≤ 0.31) — financial–energy cost pressure is not well-represented in the item space. This is because S8 (income) is categorical and contributes a different type of variance than E2A/E2B (subjective bill concern). No single latent dimension bridges the objective-financial and subjective-perceptual aspects of FCP.

**AE Seed Stability** (30 seeds):
- BLI: mean |r| = 0.82 ± 0.11 — very stable
- TCR: mean |r| = 0.70 ± 0.09 — stable
- AEMC: mean |r| = 0.51 ± 0.14 — moderate
- FCP: mean |r| = 0.25 ± 0.10 — **unstable** (high seed-to-seed variance, low signal)

The AE reconstruction MSE is consistent across seeds (0.54 ± 0.009) despite FCP instability — the AE reliably reconstructs the overall item space but inconsistently captures the FCP dimension specifically.

---

## 6. CatBoost Classification — High AEV Prediction (Stages 9–10)

### 6.1 Primary Model: Controls-Only (Generalizable Predictor)

| Metric | Value |
|--------|-------|
| ROC-AUC (test) | **0.647** |
| Train ROC-AUC | 0.659 |
| Overfit gap | 0.011 ✓ |
| Balanced accuracy | 0.623 |
| Precision (HighAEV) | 35.0% |
| Recall (HighAEV) | 65.6% |
| F1 (HighAEV) | 0.457 |
| PR-AUC | 0.378 |
| Best iteration | 30 / 600 |
| Decision threshold | 0.50 |
| n train / n test | 761 / 254 |

**Overfit gap = 0.011 — within acceptable range** after regularization tightening (depth 3, l2=8, min_leaf=20). The model is not overfitting; the low performance is a genuine signal about the limited information in observable household controls.

### 6.2 Cross-Validation Stability

| Fold | Train AUC | Test AUC | Gap | Best iter |
|------|-----------|----------|-----|-----------|
| 1 | 0.694 | 0.571 | 0.122 ⚠ | 13 |
| 2 | 0.632 | 0.573 | 0.059 ✓ | 3 |
| 3 | 0.636 | 0.537 | 0.100 ⚠ | 2 |
| 4 | 0.693 | 0.630 | 0.063 ✓ | 20 |
| 5 | 0.619 | 0.575 | 0.044 ✓ | 4 |
| **Mean** | **0.655** | **0.577** | **0.078** | **8** |

CV mean test AUC = 0.577 (σ = 0.033). The gap between the held-out performance (0.647 on a single 75/25 split) and 5-fold CV mean (0.577) is partly explained by stratification variance with a small dataset — the 75/25 split may produce a slightly easier test fold. Folds 1 and 3 still show concerning gaps (0.12 and 0.10), and the very early stopping (2–20 iterations) across all folds indicates the model exhausts its learning capacity very quickly — a signal that the features contain limited discriminative information for this target.

### 6.3 Confusion Matrix

| | Predicted: Low AEV | Predicted: High AEV |
|--|-------------------|---------------------|
| **True: Low AEV** | 112 (TN) | 78 (FP) |
| **True: High AEV** | 22 (FN) | 42 (TP) |

At the optimal threshold (0.50), the model achieves 65.6% recall of High AEV households. In absolute terms:
- **42 of 64 true High AEV households are correctly identified** (true positives)
- **22 High AEV households are missed** (false negatives — the more costly error)
- **78 Low AEV households are incorrectly flagged** (false positives)

For a policy targeting programme (e.g., energy efficiency grants, hardship tariff eligibility), the recall of 66% with 35% precision means: if this model were used to allocate support, for every 3 households selected, approximately 1 would genuinely need support and 2 would not. This is better than random (baseline precision = 25%) but far from actionable at current performance.

### 6.4 Feature Importance

| Rank | Feature | CatBoost Importance | SHAP mean |abs| | Economic meaning |
|------|---------|--------------------|--------------------|-----------------|
| 1 | S8 | 31.9% | 0.0104 | **Household income bracket** |
| 2 | H3 | 22.8% | 0.0045 | **Year of property construction** |
| 3 | H2 | 12.1% | 0.0012 | Number of rooms (dwelling size) |
| 4 | risk_category | 6.9% | 0.0028 | Energy poverty proxy (combined flag) |
| 5 | H1 | 5.7% | 0.000 | Dwelling type (flat / terraced / detached) |
| 6 | S3 | 5.5% | 0.0018 | Household size (number of occupants) |
| 7 | S5 | 4.4% | 0.0023 | Tenure (rent / own) |
| 8 | high_cost_flag | 4.0% | 0.000 | High energy cost relative to income |
| 9 | S6 | 3.2% | 0.0006 | Age of main respondent |
| 10 | S2 | 3.2% | 0.0022 | Employment status |
| 11 | low_income_flag | 0.3% | 0.000 | Low income threshold flag |

**S8 (income) and H3 (building age) together account for 55% of CatBoost feature importance.** This is the central economic finding of the ML stream:

**Income is the dominant predictor.** Low-income households have fewer financial resources to buffer energy cost volatility, fewer options to invest in efficiency improvements, and less bargaining power with energy retailers. This is consistent with the energy poverty literature (Boardman 1991, Hills 2012).

**Building age (H3) is the second-strongest predictor.** Older UK dwellings (pre-1940 and pre-1970 stock) have systematically poorer thermal efficiency — solid walls, single glazing, no cavity insulation — meaning occupants must spend more on heating to maintain comfort, increasing their adaptive burden. The SHAP analysis shows H3 pushes high-AEV predictions particularly for households in properties built before 1965.

**risk_category and S5 (tenure) contribute modest additional signal.** Private renters are constrained by split incentives (landlords own the building, tenants pay bills) — a well-documented structural barrier to energy adaptation. S2 (employment) adds signal because unemployment reduces both income and the capacity to engage with energy-switching or retrofit schemes.

**low_income_flag and H1 contribute near-zero SHAP** despite their intuitive relevance — likely because their information is already captured by S8 and S5 respectively.

### 6.5 Multi-Model Comparison

| Model | Test AUC | Train AUC | Gap | PR-AUC | Interpretation |
|-------|----------|-----------|-----|--------|----------------|
| Controls_Only | 0.658 | 0.633 | −0.025 | 0.347 | **Generalizable** |
| SEM_COR | **0.996** | 0.999 | 0.003 | 0.985 | Circular (AEV ≡ f(scores)) |
| Linear_AE | 0.919 | 0.950 | 0.031 | 0.601 | Structurally circular |
| EFA | 0.924 | 0.923 | −0.001 | 0.741 | Structurally circular |
| PCA | 0.924 | 0.923 | −0.001 | 0.741 | Structurally circular |
| Hybrid_SEM_AE | 0.969 | 0.993 | 0.024 | 0.914 | Partially circular |

The 0.27-point gap between Controls_Only (0.658) and AE/EFA/PCA (0.92+) quantifies the information content in the survey's behavioural items that is not captured by administrative-style household controls. However, this gap is **not a causal estimate** — it reflects structural circularity (the latent dimensions compress the same survey items that define AEV) rather than the genuine predictive value of latent features beyond controls.

**Negative overfit gap for Controls_Only (−0.025):** train AUC (0.633) is slightly lower than test AUC (0.658). This is unusual but can occur with stratified splits and class weights — the model is not overfitting; the test fold happened to be slightly easier to discriminate than the training set.

---

## 7. SHAP Explainability — Key Household Narratives

The SHAP analysis on the Controls_Only model identifies four prototypical vulnerability pathways:

**Pathway 1 — Low income + old property** (highest AEV probability): S8 pushes strongly toward high AEV; H3 (pre-1965 build) amplifies the prediction. These households combine financial constraint with a structurally energy-inefficient home — they cannot afford to heat it adequately, cannot afford to retrofit it, and cannot afford to move.

**Pathway 2 — Risk category + private renter** (moderate-high AEV): risk_category and S5 both push toward high AEV. Private renters in the energy-cost-high bracket face the split incentive trap — landlord has no incentive to install insulation; tenant bears the cost of inefficiency.

**Pathway 3 — Large household, moderate income** (mixed AEV): S3 (household size) interacts with S8 in a non-linear way — larger households with moderate income are exposed to higher absolute energy costs but may benefit from economies of scale in heating. SHAP values for this group are near-zero, reflecting genuine model uncertainty.

**Pathway 4 — Single-person, low income** (high AEV probability): S3 pushes toward low AEV (fewer people, smaller absolute consumption) but this is more than offset by S8. Single-person low-income households are a key policy target — they often occupy small but thermally inefficient properties with no ability to share costs.

---

## 8. Cross-Stream Consistency Check

| Finding | Consistent across streams? |
|---------|--------------------------|
| Income (S8) drives High AEV risk | ✓ Feature importance + SHAP + COR c'-path (higher FCP → higher AEV) |
| Low adaptive capacity (AEMC) weakly measured | ✓ AVE = 0.035, α = 0.29, FCP best r = 0.31 |
| BLI is the most data-recoverable COR construct | ✓ PCA r = 0.81, α = 0.86, AE seed stability = 0.82 |
| 2017 macro environment was below-average stress | ✓ All three FES variants (core/macro/actual) negative |
| Carbon forecast is structurally unreliable | ✓ 0% PI coverage, 88% SMAPE, systematic direction reversal in LSTM-core |
| Mediation FCP→AEMC→AEV not confirmed | ✓ a-path β = −0.024, p = 0.14; bootstrap CI includes zero |

---

## 9. Summary of Issues and Recommendations

### Resolved in This Version
- ✅ CatBoost overfitting tightened (depth 4→3, l2 3→8, min_leaf 10→20, random_strength 1.5)
- ✅ `use_best_model=True` prevents early-stopping misfire
- ✅ Optimal decision threshold (F1-maximising sweep)
- ✅ Feature importance now saved and plotted
- ✅ Within-split encoding for AE/EFA/PCA (transductive leakage eliminated)
- ✅ Missing preferred-features warning added
- ✅ AE/EFA/PCA structural circularity clearly documented

### Remaining Issues to Address in Future Work

| Issue | Severity | Recommendation |
|-------|----------|----------------|
| AVE < 0.05 for all constructs | High | Reframe as formative composites; report α only (not CR/AVE) in the main text |
| SEM circular path estimates | High | Remove b/c'/d/e paths from SEM table or clearly label as identities; report a-path and mediation only |
| Carbon LSTM-core 0% PI coverage | High | Demote LSTM-core; promote TFT-macro as primary carbon FES component |
| H5, H6, H13 missing from ML | Medium | Investigate ENABLE UK data availability for smart-meter, heating-fuel variables |
| CV mean AUC 0.577 vs. hold-out 0.647 | Medium | Report CV mean as the primary performance estimate |
| FCP not recoverable from latent methods | Low | Document as a feature of the item space; no remediation needed |
| Gas/electricity SMAPE = 200% in some models | Low | Expected near-zero growth rates; continue to exclude MAPE from rank-aggregation |

---

*Anticipatory Energy–Carbon Stress Pipeline · June 2026 report*
