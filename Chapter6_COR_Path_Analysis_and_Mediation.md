# Chapter 6: COR Path Analysis and Mediation

## 6.1 Revised COR Pathway

The Conservation of Resources (COR) framework posits that vulnerability to energy stress arises through a sequence of resource loss and depletion. In the theoretical model developed for this dissertation, Financial–Energy Cost Pressure (FCP) represents an external resource threat; Adaptive Energy-Management Capacity (AEMC) represents the internal resource pool that can be depleted by that threat; and Adaptive Energy Vulnerability (AEV) is the outcome state — elevated when resources are depleted or unavailable, reduced when adaptation is high. Behavioural Lock-in (BLI) and Transition-Cost Resistance (TCR) are direct structural contributors to AEV, representing habitual and attitudinal constraints that limit the household's capacity to re-organise energy behaviour.

The full pathway tested in this chapter is:

```
[FES 2017 = −0.929 core / −0.472 macro]   ← macro context (not in regression)
           ↓
FCP  ──[a]──▶  AEMC  ──[b]──▶  AEV
 └──────────────[c']──────────────▶
BLI  ──[d]──────────────────────▶
TCR  ──[e]──────────────────────▶
```

Path **a** is the theoretically central mediation pathway: financial pressure depleting adaptive capacity. Paths **b**, **c'**, **d**, and **e** constitute the structural antecedents of AEV. Figure `outputs/sem_mediation/figures/cor_path_diagram.png` presents the full path diagram with estimated coefficients; Figure `outputs/sem_mediation/figures/cor_path_coefficients.png` displays coefficient magnitudes and confidence intervals.

---

## 6.2 FES as Contextual Background

The Forecast Energy Stress Index (FES) derived in Chapter 4 is a macro-level indicator whose value in 2017 is identical for every UK household: FES_core = −0.929, FES_macro = −0.472, FES_actual = −0.827 (Table `outputs/sem_mediation/tables/fes_context_summary.csv`). Because all UK respondents share the same annual macro-stress background, the FES has zero within-sample variance at the household level and cannot enter any cross-sectional regression as a predictor. A coefficient for FES would be mathematically undefined.

The FES therefore serves exclusively as **contextual framing** for interpreting household-level results — a scenario-conditioned interpretation, not a predictor or causal driver of AEV/HighAEV. The consistent finding that all 2017 FES variants are below the 2005–2016 historical mean (negative z-scores) implies that the 2017 household survey was conducted in a macro environment of below-average energy-carbon stress. This has a specific interpretation for AEV results: the vulnerability captured by AEV in 2017 reflects structural household characteristics (income, housing stock, behaviour, attitudes) rather than acute macro-energy-price pressure. The identified high-vulnerability households would be expected to exhibit even greater stress during macro-elevated periods such as 2021–2022 — a scenario-conditioned observation about which macro context these already-estimated vulnerabilities would be interpreted under, not a prediction of how households would behave if FES rose. FES_core/FES_macro/FES_actual here are 3 of the 9 named FES scenarios in the current pipeline (`src/fes_scenarios.py`); the full scenario-to-HighAEV interpretation matrix is `outputs/fes_highaev_interpretation/scenario_highaev_interpretation_matrix.csv`.

---

## 6.3 Path a: FCP → AEMC

Path a is the theoretically pivotal link in the COR mediation chain: the hypothesis that financial pressure erodes adaptive energy-management capacity. It is the only path in the model that carries genuine behavioural information independent of the AEV algebraic identity (see Section 6.4).

**Estimated coefficient:** β = **−0.024**, SE = 0.016, t = −1.474, p = 0.141, n = 1,015, R² = 0.002 (Table `outputs/sem_mediation/tables/sem_path_estimates.csv`).

The coefficient is in the **predicted direction** — negative, indicating that higher FCP (greater financial-energy cost pressure) is associated with lower AEMC (lower adaptive capacity) — but it does not reach statistical significance at the conventional α = 0.05 level. The 95% confidence interval approximately spans [−0.055, 0.008], including zero. The effect size is small (β = −0.024 on normalised 0–1 scales, R² = 0.002), indicating that financial pressure explains only 0.2% of the variance in adaptive capacity within this sample.

**Interpretation.** The non-significance of path a does not necessarily refute COR theory. Three explanations are consistent with the null result:

1. *Cross-sectional design*. COR is a temporal resource-depletion theory. A single-wave survey captures a static cross-section, not the longitudinal process by which sustained financial pressure gradually erodes the household's capacity to maintain adaptive behaviours. A panel study following households through the 2021–2022 energy crisis would be better positioned to detect this pathway.

2. *UK welfare-state attenuation*. UK households with high FCP may benefit from targeted support mechanisms — the Warm Homes Discount, Energy Company Obligation schemes, and means-tested bill assistance — that partially compensate for financial resource depletion. These state-provided resources could break the direct FCP → AEMC link by supplying external adaptive resources that offset the threat, reducing the within-sample correlation.

3. *Construct imprecision*. The FCP composite has low internal reliability (α = 0.518, AVE = 0.038, Chapter 5) and AEMC has the weakest reliability of all constructs (α = 0.286). Measurement error in both variables attenuates regression coefficients toward zero. Even a genuine moderate-strength FCP–AEMC relationship would be difficult to detect when measurement noise is high.

Despite the non-significance, the directional consistency is noted: no evidence contradicts COR theory on this pathway; the data simply do not provide sufficient signal to confirm it statistically within this design.

---

## 6.4 Direct Paths to AEV: Structural Circularity

Paths b through e — AEMC → AEV, FCP → AEV (c'), BLI → AEV (d), and TCR → AEV (e) — are estimated jointly in an OLS regression of AEV_score on all four construct scores.

**Results:** β = ±0.25 for all paths, t-statistics of order 10¹⁵, R² = 1.000, p < 0.001 for all predictors (Table `outputs/sem_mediation/tables/sem_path_estimates.csv`).

These results are **not interpretable as independent causal effects**. They are the mechanical consequence of the AEV formula:

> AEV_i = mean(FCP_i, BLI_i, TCR_i, 1 − AEMC_i) = ¼FCP + ¼BLI + ¼TCR − ¼AEMC + ¼

When AEV is regressed on its own constituent components, the OLS estimator recovers the exact algebraic weights (0.25 per component) with zero residual by construction. The R² = 1.000 is a definitional artefact, not an empirical finding. The t-statistics in the range of 10¹⁵ reflect numerical precision at the floating-point level, not scientific evidence.

**Reporting convention adopted.** These paths are documented in Table `outputs/sem_mediation/tables/sem_path_estimates.csv` for methodological transparency — to confirm that the pipeline correctly identifies the algebraic structure — but they are **excluded from substantive interpretation** in this chapter. Any regression of AEV on its definitional components will recover this same result regardless of the sample, survey instrument, or country. Researchers wishing to test structural paths to AEV independently must use a theoretically specified outcome variable that is not algebraically constructed from the predictor set.

---

## 6.5 Mediation Analysis: FCP → AEMC → AEV

The bootstrap mediation analysis tests whether the effect of FCP on AEV operates (at least partly) through the depletion of AEMC. This indirect pathway is not subject to the structural circularity problem because path a (FCP → AEMC) is estimated independently of the AEV definition (Figure `outputs/sem_mediation/figures/mediation_effects.png`, Table `outputs/sem_mediation/tables/mediation_effects.csv`).

**Results:**
- Indirect effect (a × b): **0.012**
- Direct effect (c', definitional artefact, reported for completeness): 0.260
- Total effect: 0.271
- Bootstrap 95% percentile CI: [**−0.003, 0.027**], n = 2,000 resamples
- Significance: **Not significant** (CI includes zero)

The indirect effect of 0.012 represents the estimated change in AEV for a one-unit increase in FCP that operates via AEMC depletion. Although positive in the expected direction (higher FCP → lower AEMC → higher AEV), the 95% confidence interval straddles zero, indicating that the mediation pathway cannot be distinguished from a null effect at conventional significance levels.

The bootstrap distribution of the indirect effect (Figure `outputs/sem_mediation/figures/mediation_effects.png`) is right-skewed and centred just above zero, with the lower tail extending into negative territory. This is consistent with a true indirect effect that is small in magnitude and imprecisely estimated given the measurement limitations of the current instruments.

**Practical significance.** Even if statistically significant, an indirect effect of 0.012 on a 0–1 scale would represent a negligible practical effect. A household at the extreme of FCP (maximum financial pressure, FCP = 1.0) would, via AEMC depletion, experience only 0.012 additional AEV units compared to a household at minimum FCP, holding everything else equal. Given the SD of AEV = 0.111, this represents approximately 0.11 SD — a small effect even in the most optimistic interpretation.

---

## 6.6 What COR Evidence Is Supported?

Table 6.1 summarises the empirical status of each COR pathway relative to theoretical predictions and observed results.

**Table 6.1. COR Pathway Evidence Assessment**

| Pathway | COR Prediction | Empirical Result | Status |
|---------|---------------|-----------------|--------|
| FCP as resource threat | Financial pressure constitutes a primary stressor in energy contexts | FCP mean = 0.476, IQR [0.36–0.57]; 22.5% of households in highest-difficulty income bracket | **Supported** |
| FCP → AEMC (depletion) | Financial stress depletes adaptive capacity | β = −0.024, p = 0.141; direction correct but not significant | **Directionally consistent; statistically inconclusive** |
| AEMC as protective buffer | Higher adaptive capacity reduces vulnerability | AEMC mean = 0.221; high AEMC is rare in UK sample; (1−AEMC) enters AEV positively by construction | **Supported theoretically; formula-enforced** |
| BLI → AEV (lock-in elevates vulnerability) | Habitual constraints raise adaptive energy vulnerability | BLI contributes equally (0.25 weight) to AEV by construction; directional sign consistent | **Theoretically supported; algebraically enforced** |
| TCR → AEV (resistance elevates vulnerability) | Transition resistance raises vulnerability | TCR contributes equally to AEV; AEMC–TCR HTMT = 0.78 (closely linked) | **Theoretically supported; algebraically enforced** |
| Mediation: FCP → AEMC → AEV | AEMC mediates the FCP–AEV relationship | Indirect = 0.012, CI [−0.003, 0.027]; not significant | **Not statistically supported** |

*Source: `outputs/sem_mediation/tables/sem_path_estimates.csv` and `outputs/sem_mediation/tables/mediation_effects.csv`. Algebraically enforced = result from structural definition of AEV, not independent empirical evidence.*

Three summary positions emerge:

**Supported by both theory and empirical data:** FCP as an energy-stress resource threat is validated by the construct distribution and the UK sample profile (heavily skewed toward financial difficulty). The AEMC mean of 0.221 confirms that adaptive resource availability is genuinely low in this sample, consistent with COR theory's prediction that resource-deprived populations exhibit low adaptive capacity.

**Theoretically supported, empirically inconclusive:** The FCP → AEMC depletion pathway is the core COR mechanism, but cross-sectional data, measurement noise, and UK welfare provisions collectively limit the ability to detect it. The null result is informative — it motivates longitudinal follow-up — but does not refute the mechanism.

**Artefactual and non-interpretable:** Paths b, c', d, and e from construct scores to AEV_score are definitional rather than empirical. They confirm that the AEV formula has been correctly implemented but cannot be used to claim causal evidence for any structural path.

---

## 6.7 Chapter Summary: COR Evidence on Adaptive Energy Vulnerability

The COR pathway analysis reveals three key findings relevant to understanding the nature of adaptive energy vulnerability in UK households.

First, the 2017 macro environment (FES ≈ −0.93 SD below average) represents a relatively benign energy-stress backdrop against which household vulnerability is assessed. Households classified as HighAEV (n = 254) in this sample are structurally vulnerable — defined by income difficulty, habitual constraints, and transition resistance — rather than acutely stressed by contemporaneous price pressure. This distinction matters for policy: structural vulnerability requires structural interventions (income support, building retrofit, behavioural-change programmes) rather than temporary bill relief.

Second, the COR mediation chain — financial pressure eroding adaptive capacity, which then amplifies vulnerability — is directionally consistent with theory (path a: β = −0.024, negative) but statistically inconclusive (p = 0.141). The failure to confirm this pathway within a cross-sectional design is an acknowledged limitation of this research. The mechanism remains theoretically plausible and empirically non-refuted; its confirmation would require longitudinal tracking of the same households across periods of varying macro energy stress.

Third, the structural circularity in the AEV definition is acknowledged and managed through careful reporting. Because AEV is defined as a mean of its components, regression of AEV on those components produces perfect fit by construction. The meaningful statistical test — the only one in this chapter not contaminated by circularity — is path a. That this single informative test is directionally correct, even if not significant, provides modest positive evidence that the COR constructs are at least coherently specified relative to the framework's theoretical logic.

Together, these findings address RQ4's extension into empirical COR evidence and set the stage for Chapter 7, which asks whether the COR construct boundaries are recoverable using purely data-driven methods.

---

*Selected output references used in this chapter:*
- *Path estimates: `outputs/sem_mediation/tables/sem_path_estimates.csv`*
- *Mediation: `outputs/sem_mediation/tables/mediation_effects.csv`*
- *COR validation: `outputs/sem_mediation/tables/cor_mechanism_validation.csv`*
- *FES context: `outputs/sem_mediation/tables/fes_context_summary.csv`*
- *Figures: `outputs/sem_mediation/figures/cor_path_diagram.png`, `outputs/sem_mediation/figures/cor_path_coefficients.png`, `outputs/sem_mediation/figures/mediation_effects.png`*
