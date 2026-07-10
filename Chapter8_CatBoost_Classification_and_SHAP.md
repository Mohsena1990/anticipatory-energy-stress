# Chapter 8: CatBoost Classification and SHAP Results

## 8.1 Prediction Objective

The machine learning stream asks a fundamentally different question from the COR pathway and unsupervised robustness analyses: **which observable household characteristics — income, housing, demographics, and risk indicators — can predict whether a household falls in the top quartile of Adaptive Energy Vulnerability?**

The binary outcome is **HighAEV**: 1 if the household's AEV score ≥ 0.615 (the 75th percentile threshold established in Chapter 5), 0 otherwise. In the analytical sample, 254 households (25.0%) are classified HighAEV and 761 (75.0%) are LowAEV, yielding a 3:1 class imbalance that is handled through class-weighted training (positive class weight ≈ 3.0 in CatBoost).

The prediction task has clear policy relevance: a model that can identify HighAEV households from administrative-style data (housing registers, income records, building age databases) could guide the targeting of energy-efficiency interventions without requiring household surveys. The extent to which AUC exceeds 0.50 quantifies how much information observable characteristics contain about vulnerability, independently of the COR survey instruments. The SHAP analysis then identifies which household features drive predictions, providing the interpretive layer for RQ5.

---

## 8.2 Model Variants

Five CatBoost variants are trained and evaluated — one controls-only baseline plus one per COR estimation route (Table `outputs/ml_classification/tables/model_comparison_table.csv`):

**Table 8.1. CatBoost Model Variants**

| Model Label | Feature Set | Interpretation Status |
|-------------|-------------|----------------------|
| **Controls_Only** | Household controls + energy-poverty proxies + 3 engineered energy-efficiency controls (S8, H1, H2, H3, S2, S3, S5, S6, risk_category, low_income_flag, high_cost_flag, has_insulation, heating_gas_share, has_smart_meter) | **Generalizable** — features do not derive from the survey items that define AEV |
| **Route1_Composite** | Controls + Route 1 formative composite scores (FCP, AEMC, BLI, TCR) | Circular — construct scores are algebraic components of the target |
| **Route2_SEM** | Controls + Route 2 CFA factor scores (fitted within-split; Section 8.2.1) | Construct-overlap / representation-validation — not fully generalizable |
| **Route3_VAE** | Controls + Route 3 VAE latent means (fitted within-split) | Construct-overlap / representation-validation — not fully generalizable |
| **AllRoutes_Hybrid** | Controls + Route 1 + Route 2 + Route 3 scores | Not generalizable — multiple overlapping/circular components |

*Source: `outputs/ml_classification/tables/model_comparison_table.csv`.*

All variants use identical CatBoost hyperparameters: depth = 3, L2 regularisation = 10, learning rate = 0.02, 600 maximum iterations, subsample = 0.75, column sample by level = 0.70, min_data_in_leaf = 20, random_strength = 1.5. Early stopping via `use_best_model = True` on a stratified 75/25 train–test split. `l2_leaf_reg` was raised from an earlier value of 8 to 10 after 5-fold CV re-validation across 3 seeds showed the same mean test AUC with a consistently smaller train/test overfit gap (`src/ml_classification.py`).

### 8.2.1 A note on sample size across variants

Controls_Only and Route1_Composite train on the **full n = 1,015** households (761 train / 254 test) — their features are lightly missing (only H2 at ~15%) and are imputed rather than dropped, matching the philosophy of the primary `prepare_features()` path. Route2_SEM, Route3_VAE, and AllRoutes_Hybrid are restricted to the **n = 503 complete-case** sample required by the CFA/VAE item matrix (Chapters 5–7 document why: E2A/E2B and the H15 battery carry 15–23% item-level missingness) — 303 train / 101 test after the same 75/25 split. This is a genuine asymmetry across variants, not a bug: Controls_Only's larger, unrestricted sample is part of why it is the more trustworthy generalizable estimate.

---

## 8.3 Why Controls_Only Is the Primary Model

The four non-Controls models all include features derived from the same ENABLE survey items that define AEV. This creates **structural circularity** of different degrees: Route 1's construct scores (FCP, BLI, TCR, AEMC) are algebraic components of the AEV formula; Route 2's CFA factor scores and Route 3's VAE latents are estimated from the same item matrix, and Route 3's latents are additionally trained with `high_aev` directly in the VAE's own loss function. A model that includes these features as predictors of AEV is, to varying degrees, predicting the target from its own ingredients or from representations of the same source items.

**Controls_Only is the sole model without this circularity.** Its features — income bracket (S8), building age (H3), number of rooms (H2), dwelling type (H1), household size (S3), tenure (S5), age (S6), employment (S2), three administrative-risk flags, and three engineered energy-efficiency controls (insulation presence, mains-gas heating share, smart-meter ownership; Section 8.2 and Chapter 3, Section 3.11.2) — are observable characteristics that do not enter the AEV formula. Its test AUC therefore measures the genuine out-of-sample predictive value of household administrative and dwelling-efficiency data for vulnerability classification.

The AUC gap between Controls_Only and the circular/construct-overlap models quantifies the additional information that COR survey responses (or their compressed representations) add about vulnerability beyond observable characteristics. This gap is real and substantial, but for Route 2/Route 3 it cannot be attributed to genuine out-of-sample predictive generalisability — it reflects the survey-to-AEV information dependency documented in Section 8.2.1's leakage-free fitting note.

---

## 8.4 Model Performance

### 8.4.1 Controls_Only Performance (Primary Model)

**Table 8.2. Controls_Only CatBoost Performance — Held-Out Test Set**

| Metric | Value |
|--------|:-----:|
| ROC-AUC (test) | **0.638** |
| ROC-AUC (train) | 0.703 |
| Overfit gap (train − test) | **0.065** |
| PR-AUC | 0.376 |
| Balanced accuracy | 0.614 |
| Precision (HighAEV) | 0.321 |
| Recall (HighAEV) | **0.797** |
| F1 (HighAEV) | 0.457 |
| Accuracy | 0.524 |
| Decision threshold | 0.465 |
| Best iteration | 106 / 600 |
| n train | 761 |
| n test | 254 |

*Source: `outputs/ml_classification/tables/catboost_performance.csv`.*

The test AUC of **0.638** indicates moderate but genuine discriminability: the model correctly ranks HighAEV households above LowAEV households roughly 64% of the time, against a random-chance baseline of 50%. The overfit gap of 0.065 is a little wider than the very tight (0.01–0.03) gaps seen in earlier iterations of this pipeline, but remains below the 0.10 threshold flagged as concerning elsewhere in this dissertation (Section 8.5).

**Best iteration is now 106** (out of 600 available), a substantial change from earlier iterations of this pipeline where the model exhausted its learning capacity within 6–30 iterations. This is the direct consequence of adding `heating_gas_share` as a Controls_Only feature (Chapter 3, Section 3.11.2; Chapter 5's documented "H5/H6/H13 absent" gap is now closed via `enable_preprocessing.build_efficiency_controls()`): the model now has a genuinely informative continuous predictor to keep exploiting well past the point where the older, sparser control set would have plateaued.

### 8.4.2 Full Model Comparison

**Table 8.3. CatBoost Model Comparison — Test Set Performance**

| Model | n train / test | Test AUC | Train AUC | Gap | PR-AUC | Balanced Acc. | Recall | Precision | F1 |
|-------|:---------------:|:--------:|:---------:|:---:|:------:|:------------:|:------:|:---------:|:--:|
| **Controls_Only** | 761 / 254 | **0.638** | 0.703 | 0.065 | 0.376 | 0.594 | 0.594 | 0.330 | 0.425 |
| Route1_Composite | 761 / 254 | 0.992 | 0.999 | 0.006 | 0.982 | 0.948 | 0.969 | 0.816 | 0.886 |
| Route2_SEM | 303 / 101 | 0.978 | 0.966 | −0.012 | 0.908 | 0.924 | 0.944 | 0.680 | 0.791 |
| Route3_VAE | 303 / 101 | 0.744 | 0.846 | 0.101 | 0.398 | 0.692 | 0.722 | 0.317 | 0.441 |
| AllRoutes_Hybrid | 303 / 101 | 0.989 | 0.998 | 0.008 | 0.960 | 0.914 | 0.889 | 0.762 | 0.821 |

*Source: `outputs/ml_classification/tables/model_comparison_table.csv` and `outputs/ml_classification/figures/model_comparison_bar.png`, `model_comparison_heatmap.png`. Unlike Controls_Only, Route1_Composite, and Route2_SEM (all reproduced bit-identically across independent runs of this pipeline), **Route3_VAE and AllRoutes_Hybrid vary by run** — `_fit_latent_train_test()` refits a fresh small VAE encoder within each train/test split, and that per-split VAE fit is not perfectly seed-stable (a repeat run measured Route3_VAE at AUC 0.717 versus 0.744 here). The qualitative conclusion (Route3_VAE is materially weaker than Route1/Route2, all four non-Controls variants are non-generalizable) is stable across runs even though the exact AUC is not.*

Route1_Composite achieves AUC = 0.992 — near-perfect discrimination — because the COR construct scores are algebraically related to the AEV target (`mean(FCP, BLI, TCR, 1−AEMC)`, Chapter 5). This is the practical ceiling of circular prediction for this pipeline. Route2_SEM (AUC = 0.978) and AllRoutes_Hybrid (AUC = 0.989) are close behind — both draw on CFA factor scores estimated from the same item pool that defines the target, and AllRoutes_Hybrid additionally combines all three routes' representations. Route3_VAE is the outlier among the four non-Controls variants at AUC ≈ 0.72–0.74 (run-dependent; see table note) — despite `high_aev` entering the VAE's own training loss (Section 8.2), its four latent dimensions are compressed to a 4-D bottleneck jointly optimising reconstruction, KL divergence, and COR-alignment, which dilutes the label signal relative to Route 1's direct algebraic construction or Route 2's supervised-adjacent factor scores.

The 0.079 AUC gap between Controls_Only (0.638) and Route2_SEM (0.978, the strongest non-circular-by-construction variant) represents the information that self-reported behavioural and attitudinal survey responses contribute to vulnerability prediction above and beyond what administrative and dwelling-efficiency data can provide. Even granting that most of this gap reflects the survey-to-AEV dependency rather than genuine generalisable signal, it confirms that demographic, housing, and efficiency controls alone cannot capture the full heterogeneity of vulnerability within this sample.

---

## 8.5 Cross-Validation and Overfitting

The 5-fold stratified cross-validation (Figure `outputs/ml_classification/figures/cv_overfitting_diagnostics.png`) assesses whether the held-out performance generalises across different data partitions.

**Table 8.4. 5-Fold Cross-Validation Results — Controls_Only**

| Fold | Train AUC | Test AUC | Gap | Balanced Acc. | F1 | Best Iter |
|------|:---------:|:--------:|:---:|:------------:|:--:|:---------:|
| 1 | 0.664 | 0.605 | 0.059 | 0.555 | 0.387 | 11 |
| 2 | 0.622 | 0.641 | −0.018 | 0.584 | 0.414 | 3 |
| 3 | 0.614 | 0.576 | 0.038 | 0.580 | 0.424 | 3 |
| 4 | 0.700 | 0.669 | 0.031 | 0.636 | 0.470 | 44 |
| 5 | 0.680 | 0.570 | 0.109 | 0.590 | 0.417 | 19 |
| **Mean** | **0.656** | **0.612** | **0.044** | **0.589** | **0.423** | **16** |
| **SD** | 0.035 | 0.041 | — | 0.028 | 0.028 | — |

*Source: `outputs/ml_classification/tables/catboost_cv_results.csv`.*

The CV mean test AUC of **0.612** (SD = 0.041) is closer to the single 75/25 held-out estimate (0.638) than in earlier iterations of this pipeline (where CV mean was 0.577 against a 0.647 held-out split) — both the level and the gap between the two estimates have improved after the Controls_Only sample-size fix (Section 8.2.1) and the addition of `heating_gas_share`. The CV estimate remains the more conservative and reliable figure for generalisation claims.

Fold 5 shows a train–test gap of 0.109, just above the 0.10 threshold flagged elsewhere as concerning; fold 2 shows a small *negative* gap (test AUC exceeds train AUC), which can occur with stratified small-sample splits and class weighting when the test fold happens to be easier to discriminate. The mean overfit gap of 0.044 remains comfortably under the 0.05 "acceptable" convention. Best iterations now range from 3–44 across folds (mean 16) rather than the near-instantaneous 2–20 (mean 8) seen previously, consistent with the model having a genuinely richer feature to keep exploiting.

---

## 8.6 Confusion Matrix and Class Errors

Figure `outputs/ml_classification/figures/confusion_matrix.png` shows the confusion matrix at the F1-maximising decision threshold of 0.465.

**Table 8.5. Confusion Matrix — Controls_Only (n = 254 test households)**

| | Predicted: LowAEV | Predicted: HighAEV |
|--|:-----------------:|:------------------:|
| **Actual: LowAEV** (n = 190) | 82 (TN, 43.2%) | 108 (FP, 56.8%) |
| **Actual: HighAEV** (n = 64) | 13 (FN, 20.3%) | 51 (TP, 79.7%) |

*Source: `outputs/ml_classification/tables/confusion_matrix.csv`, `classification_report.csv`.*

At this threshold the model correctly identifies **51 of the 64 true HighAEV households** in the test set (recall = 79.7%), a substantial improvement over earlier iterations of this pipeline (65.6%). This comes at the cost of a higher false-positive rate: 108 of 190 LowAEV households (56.8%) are flagged as high-risk, versus 41.1% previously. The F1-maximising threshold (0.465, close to but below the naive 0.50) is deliberately trading precision for recall given the 3:1 class imbalance and the policy framing in Section 8.9 — for a screening tool, missing a genuinely vulnerable household (false negative) is typically more costly than over-including a marginal one (false positive).

Precision is 32.1%: roughly one in three households flagged by the model on the basis of controls alone is genuinely HighAEV — better than the 25% random-selection baseline, and comparable to earlier iterations of this pipeline, but recall has improved markedly at a moderate precision cost. The ROC-AUC of 0.638 and PR-AUC of 0.376 both remain above their respective random baselines (0.50 and 0.25).

---

## 8.7 SHAP Global Feature Importance

SHAP (SHapley Additive exPlanations) values decompose each prediction into additive feature contributions, providing model-agnostic attribution of predictions to individual features. Global importance is measured as the mean absolute SHAP value across all test households (Figure `outputs/ml_classification/figures/feature_importance_controls_only.png`, Table `outputs/shap/tables/controls_only_shap_feature_importance.csv`).

**Table 8.6. SHAP Global Feature Importance — Controls_Only**

| Rank | Feature | CatBoost Importance (%) | Mean \|SHAP\| | Positive Share | Description |
|------|---------|:-----------------------:|:------------:|:--------------:|-------------|
| 1 | **heating_gas_share** | 21.2 | **0.1194** | 37.4% | % of home heating from mains gas (engineered, Section 3.11.2) |
| 2 | **S8** | 20.7 | **0.0717** | 31.1% | Household income difficulty bracket |
| 3 | **H3** | 16.3 | 0.0587 | 80.3% | Year of property construction |
| 4 | H2 | 7.4 | 0.0266 | 61.4% | Number of rooms in dwelling |
| 5 | has_insulation | 4.0 | 0.0230 | 30.7% | Any attic/cavity/external-wall insulation (engineered) |
| 6 | risk_category | 7.9 | 0.0067 | 42.9% | Energy poverty proxy (combined flag) |
| 7 | H1 | 4.9 | 0.0171 | 73.6% | Dwelling type (flat/terraced/detached) |
| 8 | S2 | 4.4 | 0.0160 | 11.8% | Employment status |
| 9 | S3 | 5.7 | 0.0115 | 46.1% | Household size (number of occupants) |
| 10 | S6 | 2.9 | 0.0097 | 31.5% | Age of main respondent |
| 11 | has_smart_meter | 1.7 | 0.0083 | 30.7% | Electricity or heating smart meter (engineered) |
| 12 | S5 | 1.9 | 0.0043 | 48.8% | Tenure (owner-occupier vs. renter) |
| 13 | low_income_flag | 0.3 | 0.0013 | 50.8% | Low income threshold flag |
| 14 | high_cost_flag | 0.7 | 0.0009 | 63.0% | High energy cost relative to income |

*Source: `outputs/shap/tables/controls_only_shap_feature_importance.csv` and `outputs/ml_classification/tables/feature_importance_controls_only.csv`. Positive share = fraction of households for whom this feature pushes toward HighAEV prediction. Rank/CatBoost-importance ordering can differ slightly (e.g. risk_category ranks 6th by CatBoost importance but 11th by mean \|SHAP\|) because the two metrics weight split-frequency and prediction-magnitude differently.*

**`heating_gas_share` is now the single most important feature** (21.2% CatBoost importance, largest mean \|SHAP\|), ahead of income difficulty. This engineered feature — reconstructed from ENABLE's per-option heating-fuel sub-items after the bare `H6` column was found not to exist in the UK sub-sample (Chapter 3, Section 3.11.2) — closes what was previously a documented gap in the Controls_Only feature set and is the main reason the model's discriminability and best-iteration count both improved (Sections 8.4.1, 8.5). Its positive share of 37.4% indicates that for most households a *higher* mains-gas heating share pushes the prediction *down* (gas heating is typically cheaper per unit than electric resistance heating in the UK), while the minority with high gas share and other compounding risk factors are pushed toward HighAEV.

**S8 (income difficulty)** remains the second-dominant feature (20.7% CatBoost importance) with a positive-share of 31.1%, consistent with the pattern documented in earlier iterations of this pipeline: for the majority of households in moderate-to-comfortable income brackets, income pushes the SHAP contribution toward lower HighAEV probability, while the minority in the hardest-income bracket receive strongly positive contributions.

**H3 (building age)** is third (16.3%) with a positive-share of 80.3% — a more one-sided pattern than S8, reflecting that older UK building stock consistently carries a thermal-efficiency penalty. **`has_insulation`** (rank 5, 4.0%) is the second engineered feature to place in the top five, complementing `heating_gas_share` and confirming that ENABLE's granular dwelling sub-items (insulation type, heating fuel mix, smart-meter ownership) collectively carry real, previously-unused predictive signal once reconstructed into usable controls.

---

## 8.8 SHAP Household Examples

Table `outputs/shap/tables/controls_only_shap_household_examples.csv` reports the five highest- and five lowest-probability test households by predicted HighAEV probability.

The **highest predicted probability is 0.607** (a true LowAEV case, i.e. a false positive), driven by S8 SHAP ≈ +0.237, `heating_gas_share` SHAP ≈ +0.118, and H3 SHAP ≈ +0.045 — a household combining high income difficulty, a high mains-gas heating share, and an older building. The next-highest cases (0.574–0.578) follow the same pattern: S8 in the +0.20 to +0.25 range dominates, with `heating_gas_share` a consistent secondary contributor (+0.10 to +0.12).

The **lowest predicted probability is 0.329**, driven by S8 SHAP ≈ −0.156, `heating_gas_share` SHAP ≈ −0.196 (the largest-magnitude single contribution observed in either direction), and H3 SHAP ≈ −0.173 — a household with low income difficulty, low mains-gas dependence (likely electric or low-consumption heating), and a newer building.

The predicted-probability range across these ten examples (0.329–0.607) is **substantially wider** than the tightly compressed 0.49–0.52 range reported in earlier iterations of this pipeline — direct evidence that `heating_gas_share` gives the model materially more room to separate cases rather than clustering all predictions near the decision boundary.

---

## 8.9 Chapter Summary: Answering RQ5

**RQ5 asks: Which household characteristics best predict High Adaptive Energy Vulnerability, and how can SHAP explain the predictive contribution of financial, housing, behavioural, and energy-related features?**

The CatBoost classification framework identifies mains-gas heating share (`heating_gas_share`), household income (S8), and building age (H3) as the three dominant predictors of HighAEV from observable administrative and dwelling-efficiency controls, together accounting for 58.2% of CatBoost feature importance. Their dominance sharpens the structural picture from earlier iterations of this pipeline: vulnerability is a function of financial capacity, physical building infrastructure, **and now explicitly the household's heating-fuel exposure** — a dimension that earlier iterations could not test because the relevant ENABLE sub-items had not yet been reconstructed into usable features (Chapter 3, Section 3.11.2).

The Controls_Only model achieves a test AUC of **0.638** (CV mean 0.612), balanced accuracy 0.594, recall 0.797, and precision 0.321 — genuine out-of-sample predictive signal in administrative and dwelling-efficiency data, free of the structural circularity that inflates the other four model variants. The AUC exceeds the random-chance baseline (0.50) but remains below the levels required for reliable stand-alone policy targeting (conventionally ≥ 0.75 for high-stakes classification); the five-fold cross-validation confirms this is a genuine signal ceiling rather than an artefact of a favourable train/test split, though the ceiling itself has risen relative to earlier iterations of this pipeline.

The five-fold cross-validation (CV mean test AUC = 0.612) and the model's longer average best-iteration count (16, versus 8 previously) together indicate that the current Controls_Only feature set has more real signal to exploit than earlier iterations, without materially changing the qualitative conclusion that observable administrative data alone cannot substitute for the COR survey instruments.

For policy applications, the practical implication is dual:

1. **Screening value.** A household identified as high-risk by administrative and dwelling-efficiency controls now has approximately a 4-in-5 chance of the model catching a genuinely HighAEV case (recall = 79.7%), at roughly 1-in-3 precision — a meaningfully better screening recall than earlier iterations of this pipeline (65.6%), at the cost of flagging more LowAEV households (56.8% false-positive rate among LowAEV cases) than before.

2. **Residual vulnerability.** The 13 false-negative households — genuinely HighAEV but not flagged by controls — represent the structurally hidden vulnerable population whose risk is driven by behavioural inertia (high BLI) or attitudinal resistance (high TCR) rather than observable financial, housing, and heating-fuel characteristics. Reaching these households still requires survey-based identification tools that can measure the latent COR dimensions directly (Route 2/Route 3, Section 8.4.2).

The full model comparison (Table 8.3) quantifies the decomposition of information sources in vulnerability prediction: administrative and efficiency controls contribute AUC = 0.638 above the 0.50 random baseline; Route 1's COR composite scores push this to 0.992 (the ceiling of circular prediction); Route 2's CFA factor scores achieve a comparably high 0.978 while drawing on the same item pool as the target (construct-overlap, not fully independent); Route 3's VAE latents, despite `high_aev` entering their own training loss, achieve a comparatively modest and run-dependent 0.72–0.74 — the joint reconstruction/KL/alignment/prediction objective dilutes rather than concentrates the label signal relative to Route 1 and Route 2. None of the four non-Controls variants should be read as demonstrating genuinely generalizable predictive power beyond Controls_Only; they instead document, at different points on the circularity spectrum, how much information the COR survey instruments carry about a target partly or wholly constructed from those same instruments.

---

*Selected output references used in this chapter:*
- *Model performance: `outputs/ml_classification/tables/catboost_performance.csv`, `model_comparison_table.csv`, `catboost_cv_results.csv`, `classification_report.csv`, `confusion_matrix.csv`*
- *Feature importance: `outputs/ml_classification/tables/feature_importance_controls_only.csv`*
- *SHAP tables: `outputs/shap/tables/controls_only_shap_feature_importance.csv`, `controls_only_shap_household_examples.csv`*
- *Figures (classification): `outputs/ml_classification/figures/roc_curve.png`, `pr_curve.png`, `confusion_matrix.png`, `cv_overfitting_diagnostics.png`, `feature_importance_controls_only.png`, `model_comparison_bar.png`, `model_comparison_heatmap.png`, `learning_curve_controls_only.png`*
- *Figures (SHAP): `outputs/shap/figures/controls_only_shap_bar_importance.png`, `controls_only_shap_beeswarm.png`, `controls_only_shap_dependence_top_features.png`*
- *Route 2/Route 3 context: `outputs/cor_sem/tables/cfa_fit_indices.csv`, `outputs/cor_vae/tables/vae_reconstruction_metrics.csv`, `outputs/route_comparison/tables/cross_route_outcome_agreement.csv`*
