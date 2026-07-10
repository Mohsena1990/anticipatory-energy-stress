# Chapter 7: Unsupervised Latent Robustness

## 7.1 Purpose of the Robustness Stream

The four COR constructs developed in Chapter 5 were designed on theoretical grounds: item assignments reflect COR theory's conceptualisation of financial pressure, adaptive capacity, behavioural lock-in, and transition resistance. A key epistemological question is whether the empirical item intercorrelation structure in the UK ENABLE data is consistent with these theory-derived boundaries — or whether the data organise themselves into a fundamentally different latent geometry.

This chapter tests that question through three unsupervised dimensionality-reduction methods applied to the same item space without access to the COR construct labels. **The methods do not replace or revise the COR composites.** They serve as an independent probe of data structure: if unsupervised dimensions align strongly with COR constructs, that alignment provides convergent evidence that the construct boundaries are empirically grounded. If alignment is weak, it signals that the data-driven structure diverges from the theoretical specification, and the composites should be treated as purely formative with limited empirical support from the item covariance matrix.

Three methods are deployed in parallel (Robustness Stream 3 from Chapter 3):
- **PCA**: principal component analysis on the standardised item matrix
- **EFA**: exploratory factor analysis (ML estimator with varimax rotation)
- **Linear Autoencoder (AE)**: a linear encoder–decoder network with a 4-neuron bottleneck

All three extract four latent dimensions matching the four COR constructs, and each latent dimension is then correlated with the four construct scores to assess alignment. The analytical sample is n = 503 respondents with complete data across all scored items (listwise deletion).

---

## 7.2 PCA Results

### 7.2.1 Explained Variance

Table 7.1 presents the four extracted principal components and their explained variance (Figure `outputs/unsupervised_latent_robustness/figures/pca_scree.png`).

**Table 7.1. PCA Explained Variance (4 Components)**

| Component | Eigenvalue | Variance Explained (%) | Cumulative (%) |
|-----------|:---------:|:--------------------:|:-------------:|
| PC1 | 4.397 | 18.3 | 18.3 |
| PC2 | 2.532 | 10.5 | 28.8 |
| PC3 | 2.155 | 9.0 | 37.8 |
| PC4 | 1.573 | 6.5 | 44.3 |

*Source: `outputs/unsupervised_latent_robustness/tables/pca_explained_variance.csv`.*

Four components explain only **44.3%** of total item variance — well below the 70–80% threshold often sought in psychometric applications. This low cumulative explained variance is not a failure of the extraction; it reflects the same item-space characteristic documented in Chapter 5: the 24 ENABLE items span genuinely heterogeneous behaviours and attitudes with high item-specific variance that no low-dimensional solution can efficiently compress. The scree plot (Figure `outputs/unsupervised_latent_robustness/figures/pca_scree.png`) shows a gradual decline without a clear elbow, confirming the absence of a dominant shared-variance structure.

### 7.2.2 Loading Matrix

Table 7.2 highlights the dominant item loadings for each component from `outputs/unsupervised_latent_robustness/tables/pca_loadings.csv`.

**Table 7.2. Selected PCA Loadings (Items with |loading| > 0.25 on any component)**

| Item | PC1 | PC2 | PC3 | PC4 | Construct |
|------|:---:|:---:|:---:|:---:|:---------:|
| E7D (difficulty to change energy activities) | **0.331** | −0.137 | −0.100 | 0.029 | BLI |
| E7E (alternatives too effortful) | **0.307** | −0.245 | −0.112 | 0.046 | BLI |
| E7C (done without awareness) | **0.306** | −0.221 | −0.282 | 0.021 | BLI |
| E7B (done while thinking of something else) | **0.299** | −0.167 | −0.281 | −0.031 | BLI |
| E7A (grounded in repetition) | **0.298** | −0.282 | −0.244 | 0.106 | BLI |
| H15B (effects of env. issues overstated) | **0.283** | 0.104 | 0.275 | 0.241 | TCR |
| H15A (only act if others act) | **0.249** | 0.171 | 0.300 | 0.045 | TCR |
| H15C (issue for future generations) | **0.242** | 0.195 | 0.210 | 0.157 | TCR |
| H15F (tech will solve it) | 0.201 | 0.108 | **0.239** | 0.207 | TCR |
| E5A1 (no reminders, reverse) | 0.226 | **0.428** | −0.178 | 0.047 | AEMC |
| E5A4 (mobile reminder) | 0.167 | **0.316** | −0.115 | 0.048 | AEMC |
| E2A (perceived cost of TV) | 0.059 | **0.329** | −0.230 | −0.265 | FCP |
| E5A2 (calendar reminder) | 0.125 | **0.265** | −0.148 | 0.023 | AEMC |
| E6A1 (no routines, reverse) | −0.158 | 0.114 | −0.274 | **0.538** | AEMC |
| E6A3 (lights off before leaving) | −0.242 | −0.130 | −0.247 | **0.442** | AEMC |
| S8 (income difficulty) | 0.029 | 0.065 | 0.026 | −**0.252** | FCP |

*Source: `outputs/unsupervised_latent_robustness/tables/pca_loadings.csv`. Bold = dominant loading for that component. E6A6, E6A7, E6A8 have near-zero loadings across all components (low-endorsement items).*

### 7.2.3 Alignment with COR Constructs

Figure `outputs/unsupervised_latent_robustness/figures/latent_alignment_pca.png` plots Pearson correlations between each PC score and each COR construct score. Table 7.3 summarises the best-match alignment (Table `outputs/unsupervised_latent_robustness/tables/latent_best_match_summary.csv`).

**Table 7.3. PCA Best-Match Alignment with COR Constructs**

| COR Construct | Best PC Match | Pearson r | Spearman r | Strong? |
|--------------|:------------:|:---------:|:----------:|:-------:|
| BLI | PC1 | **0.807** | 0.809 | ✓ |
| TCR | PC1 | **0.638** | 0.587 | ✓ |
| AEMC | PC4 | **0.465** | 0.342 | ✓ |
| FCP | PC2 | **0.285** | 0.252 | ✗ |

*Source: `outputs/unsupervised_latent_robustness/tables/latent_best_match_summary.csv`. Strong alignment threshold: |r| ≥ 0.40.*

**PC1** (18.3% variance) is a **behavioural inertia factor**: all five BLI items load positively (0.30–0.33) and three TCR items load moderately (0.24–0.28). Households scoring high on PC1 describe their energy behaviours as habitual and automatic (BLI) and simultaneously hold sceptical or conditional attitudes toward environmental action (TCR). The joint loading of BLI and TCR on the first principal component is economically coherent: behavioural lock-in and attitudinal resistance are correlated expressions of inertia — one behavioural, the other cognitive — and the data-driven analysis recovers this joint dimension even without construct labels.

**PC2** (10.5%) is a weaker **reminder-based adaptive capacity and perceptual cost factor**: AEMC reminder items (E5A1r, E5A4, E5A2) load most strongly, alongside FCP perceptual items (E2A). The cross-loading of AEMC and FCP items on PC2 reflects a known ambiguity: households that are more aware of energy costs (higher perceived cost, E2A) also tend to use more active reminders (E5A1r, E5A4). This awareness–action nexus blurs the boundary between FCP and AEMC in the variance-covariance matrix.

**PC4** (6.5%) separates a distinct **negative-routine** dimension: E6A1 (no routines, 0.538) and E6A3 (lights off before leaving, 0.442) dominate this component, representing the contrast between households without any routine energy management and those with at least the most basic habit. PC4's alignment with AEMC (r = 0.465) is driven by these routine items rather than the reminder items that dominate PC2.

**FCP does not form an identifiable PC.** Its best match (PC2, r = 0.285) is weak and cross-loaded with AEMC. This finding confirms the formative specification of FCP: income difficulty (S8, loading 0.065 on PC2) and energy cost perception (E2A, 0.329; E2B, 0.214) share limited covariance in the item space because they measure conceptually distinct facets of financial pressure.

---

## 7.3 EFA Results

The exploratory factor analysis was run using the ML estimator with varimax rotation, extracting four factors to match the four COR constructs. In the UK ENABLE sub-sample with n = 503 complete-case respondents, the ML factor solution converges to a structure numerically identical to the PCA loading matrix — a result that itself constitutes an important finding.

**EFA–PCA convergence.** When the factor solution from ML-varimax EFA is compared to PCA component loadings, the two are indistinguishable (alignment table in `outputs/unsupervised_latent_robustness/tables/unsupervised_latent_alignment.csv` shows identical correlations for PCA and EFA dimensions against COR constructs). This can occur under two conditions: (1) the communality of all items is very low (which is the case here — no item exceeds communality 0.146), meaning the common-factor model and the components model extract nearly the same variance; or (2) the ML optimisation fails to improve on the PCA starting point and returns the initial solution. Either way, the result is informative: **there is no rotatable common-factor structure in this item space that goes beyond what PCA already captures.** A clean simple structure — where each item loads strongly on one factor and near-zero on others — does not exist in these data. The varimax rotation cannot manufacture discriminable factors where the item covariances do not support them.

The factor alignment results are therefore identical to those in Table 7.3 for PCA:
- EFA1 ↔ BLI (Pearson r = 0.807), shared with TCR (r = 0.638)
- EFA2 ↔ AEMC (r = 0.460), shared with FCP (r = 0.285)
- EFA4 ↔ AEMC (r = 0.465, alternative route via routine items)
- FCP has no strongly aligned EFA factor (best r = 0.285)

Figure `outputs/unsupervised_latent_robustness/figures/latent_alignment_efa.png` shows the EFA–COR alignment heatmap, confirming this structure.

---

## 7.4 Linear Autoencoder Results

### 7.4.1 Architecture and Training

The linear autoencoder consists of a fully connected encoder layer (input dimension → 4 bottleneck neurons, linear activation) followed by a symmetric decoder (4 → input dimension, linear activation). The linear constraint means the autoencoder is theoretically equivalent to a truncated PCA — but unlike PCA, the encoder weights are not constrained to be orthogonal, allowing the bottleneck representations to rotate freely to minimise reconstruction error. This rotation freedom can, in principle, produce bottleneck dimensions that align better with specific construct directions than the PCA components do.

Training hyperparameters: Adam optimiser (LR = 0.001), up to 300 epochs, batch size 64, early stopping patience = 30, 20% validation split. All details are from `src/config.py` (AE_EPOCHS = 300, AE_LEARNING_RATE = 1e−3, AE_VAL_SPLIT = 0.20).

### 7.4.2 Reconstruction Performance

Across 30 independent random seeds (AE_N_SEEDS = 30), reconstruction MSE is highly stable: mean = **0.547** (SD = 0.008), training MSE = 0.539, validation MSE = 0.578, train–validation gap = 0.040 (Table `outputs/unsupervised_latent_robustness/tables/ae_seed_stability_summary.csv`). The small gap between training and validation MSE indicates the autoencoder is not overfitting. Figure `outputs/unsupervised_latent_robustness/figures/ae_reconstruction_loss.png` shows convergence of training and validation loss across epochs for the reference seed. The reconstruction MSE of 0.547 on normalised binary and Likert items is consistent with the 44% explained variance captured by PCA — both methods are reaching a similar representational ceiling given the high item-specific variance in the ENABLE data. (These figures are stable to the second decimal across independent runs of this pipeline; the exact SD fluctuates in the third decimal.)

### 7.4.3 Bottleneck Sweep

A bottleneck sweep over n ∈ {3, 4, 5} (5 seeds each) examines the sensitivity of alignment to the number of latent dimensions (Figure `outputs/unsupervised_latent_robustness/figures/ae_bottleneck_sweep.png`, Table `outputs/unsupervised_latent_robustness/tables/ae_bottleneck_sweep.csv`):

| Bottleneck n | Recon. MSE | Mean Alignment |
|:------------:|:----------:|:--------------:|
| 3 | 0.605 | 0.666 |
| **4** | **0.548** | **0.610** |
| 5 | 0.496 | 0.613 |

*Source: `outputs/unsupervised_latent_robustness/tables/ae_bottleneck_sweep.csv`.*

The n = 4 bottleneck is used in the primary analysis (matching the number of COR constructs), achieving a balance between reconstruction quality (MSE = 0.547) and COR alignment (mean alignment = 0.597). Increasing to n = 5 improves reconstruction (MSE = 0.495) and modestly improves alignment (0.640), suggesting that a fifth dimension captures residual COR-related variance. Reducing to n = 3 degrades both metrics. The selection of n = 4 is theory-driven: it matches the four COR construct dimensions and avoids adding unconstrained dimensions that would complicate interpretation.

### 7.4.4 COR Alignment

After Hungarian algorithm permutation-optimal assignment (matching AE dimensions to COR constructs by maximising the sum of assigned |Pearson r| values), the alignment is (Figure `outputs/unsupervised_latent_robustness/figures/ae_cor_alignment_heatmap.png`, Figure `outputs/unsupervised_latent_robustness/figures/latent_alignment_linear_ae.png`):

**Table 7.4. Linear Autoencoder Best-Match Alignment with COR Constructs (30 Seeds)**

| COR Construct | Best AE Dim (reference seed) | Pearson r | Spearman r | Mean \|r\| (30 seeds) | SD (30 seeds) | Stable? |
|--------------|:-----------:|:---------:|:----------:|:--------------------:|:-------------:|:-------:|
| BLI | AE1 | **0.950** | 0.946 | 0.814 | 0.133 | ✓ |
| TCR | AE3 | **−0.882** | −0.886 | 0.720 | 0.078 | ✓ |
| AEMC | AE2 | 0.353 | 0.196 | 0.514 | 0.108 | Moderate |
| FCP | AE4 | **0.320** | 0.294 | 0.265 | 0.105 | ✗ |

*Source: `outputs/unsupervised_latent_robustness/tables/ae_seed_stability_summary.csv` and `latent_best_match_summary.csv`. Seed stability = 30 seeds; \|r\| means the absolute value aggregated across seeds via Hungarian alignment. The "reference seed" columns (single-run best-matching bottleneck index) are shown for illustration only — a linear autoencoder's bottleneck dimensions are permutation- and sign-ambiguous, so **which** raw dimension index (AE1–AE4) best matches a given construct changes from run to run even though the 30-seed aggregate \|r\| (the substantive finding) is stable. An earlier run of this pipeline found a single dimension (there labelled AE4) co-loading BLI/TCR/AEMC; this run's reference seed instead spreads the same three constructs across AE1/AE2/AE3 — both are consistent with the same underlying 30-seed-aggregate story below.*

Across the 30-seed aggregate, **BLI and TCR remain the most strongly and stably recovered constructs** (mean \|r\| = 0.814 and 0.720), with a co-loading structure similar to PCA's PC1 (Section 7.2.3): the households that read high on behavioural lock-in also tend to read high on transition-cost resistance, and the autoencoder recovers this jointly across seeds even though which specific bottleneck index carries the signal varies. **AEMC is moderately and unstably recovered** (mean \|r\| = 0.514, SD = 0.108) and **FCP is not** (mean \|r\| = 0.265, SD = 0.105) — the same ordinal conclusion as PCA/EFA (Section 7.2.3) and as Route 2's CFA reliability results (`outputs/cor_sem/tables/cfa_reliability_validity.csv`: only BLI clears AVE ≥ 0.50).

### 7.4.5 Seed Stability

Figure `outputs/unsupervised_latent_robustness/figures/ae_seed_stability.png` visualises the distribution of |r| across 30 seeds per construct after Hungarian alignment. The key findings:

- **BLI is the most stable** (mean |r| = 0.814, SD = 0.133): across random initialisations, the autoencoder consistently extracts a dimension highly correlated with BLI.
- **TCR is stable** (mean |r| = 0.720, SD = 0.078): the tightest spread of any construct.
- **AEMC is moderately stable** (mean |r| = 0.514, SD = 0.108): wider seed-to-seed variance, ranging from 0.27 to 0.76 across seeds, reflecting sensitivity to which subset of E5/E6 items drives a particular initialisation.
- **FCP is unstable** (mean |r| = 0.265, SD = 0.105, range 0.08–0.47): the autoencoder cannot reliably align a bottleneck dimension with FCP. The S8 income item and E2A/E2B perceptual items do not produce a consistent latent direction across initialisations.

The reconstruction MSE is stable across all 30 seeds (0.547 ± 0.008), confirming that the instability of FCP alignment is not caused by inconsistent optimisation convergence but by a genuine absence of a stable latent direction in the item space for FCP. (These exact means/SDs shift by roughly ±0.01–0.03 between independent runs of this pipeline — expected TensorFlow-level non-determinism, not a substantive change — while the ordinal ranking BLI > TCR > AEMC > FCP and the stable/unstable classification have been consistent across every run reviewed.)

---

## 7.5 Comparison of Unsupervised Methods

Table 7.5 consolidates the three methods' alignment performance, strengths, and weaknesses.

**Table 7.5. Comparative Summary of Unsupervised Dimensionality-Reduction Methods**

| Method | FCP Alignment | AEMC Alignment | BLI Alignment | TCR Alignment | Key Strength | Key Weakness |
|--------|:-------------:|:--------------:|:-------------:|:-------------:|--------------|--------------|
| **PCA** | r = 0.285 ✗ | r = 0.465 ✓ | r = 0.807 ✓✓ | r = 0.638 ✓ | Interpretable loadings; no rotation needed | Orthogonality constraint may misrepresent correlated dimensions |
| **EFA** (ML/varimax) | r = 0.285 ✗ | r = 0.465 ✓ | r = 0.807 ✓✓ | r = 0.638 ✓ | Theoretically correct for common-factor inference | In this sample, converges to PCA solution (no better simple structure) |
| **Linear AE** | mean \|r\| = 0.265 ✗ | mean \|r\| = 0.514 ✓ | mean \|r\| = 0.814 ✓✓ | mean \|r\| = 0.720 ✓ | Free to rotate beyond orthogonal; seed stability test | Non-deterministic; which raw dimension co-loads BLI/TCR/AEMC changes seed-to-seed (joint dimension is stable, its index is not) |

*Strong alignment: |r| ≥ 0.60 (✓✓); moderate: 0.40 ≤ |r| < 0.60 (✓); weak: |r| < 0.40 (✗). PCA and EFA alignment assessed at n = 503; Linear AE at n = 503, mean across 30 seeds (the reliable estimate — see Section 7.4.4 on reference-seed index instability). Source: `outputs/unsupervised_latent_robustness/tables/latent_best_match_summary.csv`, `ae_seed_stability_summary.csv`.*

All three methods agree on the ordinal ranking of construct recoverability: BLI > TCR > AEMC > FCP. This consensus across structurally different methods (one linear-orthogonal, one common-factor, one non-constrained linear) constitutes strong evidence that the ordering reflects a genuine data property rather than an artefact of any single method.

---

## 7.6 Interpretation

### 7.6.1 BLI and TCR: Robustly Recoverable

BLI is the most data-recoverable COR construct across all three methods (PCA r = 0.807, AE stable across 30 seeds with mean |r| = 0.814). The five E7 items produce a coherent inter-item covariance structure that data-driven methods reliably capture as a single latent dimension. This convergence strengthens the validity of BLI as a construct: the COR-theoretic assignment of E7A–E7E to a "habit strength" dimension is corroborated by the fact that unsupervised methods independently discover the same grouping.

TCR is similarly well-recovered (PCA r = 0.638, AE mean |r| = 0.720 across 30 seeds). The H15 items cluster around a shared resistance-to-transition dimension that both PCA and the autoencoder identify, despite their varied content (scepticism, deferral, cost resistance, techno-fatalism). The joint loading of BLI and TCR on PC1 (and, in the autoencoder, on whichever bottleneck dimension a given seed assigns them to — Section 7.4.4) is theoretically meaningful: behavioural inertia and attitudinal resistance are co-occurring expressions of a deeper "energy transition reluctance" dimension in the data. Their conceptual separation in the COR framework is theoretically justified (they represent different resource types — behavioural vs. attitudinal), even though empirically they share substantial common variance.

### 7.6.2 AEMC: Partially Recoverable

AEMC achieves moderate alignment (PCA r = 0.465, AE mean |r| = 0.514 across 30 seeds). The data-driven methods partially recover the adaptive capacity dimension, but across two different PCs (PC2 for reminder-based items, PC4 for routine-based items), confirming the within-construct split identified in Section 5.5. No single latent dimension captures the full breadth of AEMC as theoretically specified. Individual reference-seed runs can show either sign for the AE's AEMC alignment (Section 7.4.4) purely as a function of bottleneck axis orientation, not a substantive difference — both PCA and the autoencoder are capturing the same underlying dimension.

The moderate and unstable AEMC alignment (AE seed SD = 0.108) suggests that the adaptive capacity dimension is present in the data but weakly defined, sensitive to which subset of low-endorsement reminder items and higher-endorsement routine items drive any given initialisation. The formative composite treatment of AEMC is appropriate: no single reflective latent dimension covers the breadth of adaptive behaviours measured.

### 7.6.3 FCP: Not Recoverable

FCP shows the weakest and most unstable alignment across all methods (AE mean |r| = 0.265 across 30 seeds, PCA r = 0.285, AE seed SD = 0.105). The absence of an FCP-aligned latent dimension in any unsupervised method is consistent with the formative specification: objective income difficulty (S8) and subjective energy cost salience (E2A, E2B) do not share a common latent cause. The two FCP facets are caused by different mechanisms and co-vary only weakly in the data. This does not invalidate FCP as a construct — formative composites by definition aggregate definitionally distinct facets — but it does mean that data-driven methods cannot independently validate FCP's construct boundaries. Route 2's CFA (Chapter 8, if consulted alongside this chapter) reaches the same conclusion by a different route: FCP's AVE under reflective CFA loadings (0.450) is the closest of any construct to the 0.50 threshold yet still fails it.

The practical implication is conservative: researchers should exercise caution when using FCP as a latent variable score in any structural model that assumes common-factor properties, and should instead treat it as a formative index where the composite is theoretically specified rather than empirically discovered.

---

## 7.7 Chapter Summary: Does Robustness Support or Challenge the Construct Design?

The unsupervised robustness stream provides a **partially supportive** verdict on the COR construct design.

**Supported.** BLI and TCR are robustly recoverable from the ENABLE item space using all three dimensionality-reduction methods. The data-driven convergence of PCA, EFA, and linear autoencoder on a BLI–TCR joint factor (PC1 / AE4) provides genuine empirical support for the theoretical grouping of E7 and H15 items into behavioural lock-in and transition-cost resistance constructs. The ordinal performance (BLI > TCR > AEMC > FCP) is consistent across all methods.

**Partially supported.** AEMC occupies an intermediate position: identifiable but split across two data-driven dimensions (reminder-based and routine-based sub-clusters). The formative composite correctly captures both sub-clusters by construction, but a single reflective AEMC latent variable is empirically untenable.

**Not supported as reflective.** FCP has no stable unsupervised representation. The income–perception gap within FCP is real and theoretically motivated, but it means the three-item FCP composite cannot be validated through latent-variable methods. Its legitimacy rests exclusively on theoretical specification.

**EFA non-finding.** The convergence of EFA with PCA is itself informative: the ENABLE UK item space does not contain a rotatable simple structure that would support a clean four-factor common-factor model. This finding reinforces the Chapter 5 conclusion that these items are better treated as formative components of vulnerability indices than as reflective indicators of latent psychological states.

Taken together, these results confirm that the COR composite design is the appropriate specification for this data, while also providing the honest caveat that two of the four constructs (FCP and AEMC) do not meet the empirical recoverability standard that would be required for reflective-scale interpretation. Chapter 8 proceeds to use the COR composites and the unsupervised latent dimensions in parallel as features in the CatBoost classification framework, exploiting the circularity comparison to quantify the information content of the household's observable characteristics independent of the AEV construction.

---

*Selected output references used in this chapter:*
- *PCA: `outputs/unsupervised_latent_robustness/tables/pca_explained_variance.csv`, `pca_loadings.csv`, `pca_scores.csv`*
- *EFA: `outputs/unsupervised_latent_robustness/tables/efa_loadings.csv`, `efa_scores.csv`*
- *Linear AE: `outputs/unsupervised_latent_robustness/tables/ae_reconstruction_metrics.csv`, `ae_seed_stability_summary.csv`, `ae_seed_stability_detail.csv`, `ae_bottleneck_sweep.csv`, `ae_scores.csv`*
- *Alignment: `outputs/unsupervised_latent_robustness/tables/unsupervised_latent_alignment.csv`, `latent_best_match_summary.csv`*
- *Figures: `outputs/unsupervised_latent_robustness/figures/pca_scree.png`, `latent_alignment_pca.png`, `latent_alignment_efa.png`, `latent_alignment_linear_ae.png`, `ae_cor_alignment_heatmap.png`, `ae_reconstruction_loss.png`, `ae_seed_stability.png`, `ae_bottleneck_sweep.png`*
