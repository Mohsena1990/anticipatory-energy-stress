# Forecasting Anticipatory Energy–Carbon Stress and High Adaptive Energy Vulnerability in the UK

**A Conservation of Resources and Explainable Machine Learning Framework**

---

## Research Objective

To examine how forecasted energy–carbon stress creates a macro-level anticipatory pressure environment, how UK households adapt through behavioural change, and how these behavioural changes may increase or reduce energy-poverty vulnerability.

---

## Conceptual Logic

```
Forecasted Energy–Carbon Stress  (macro-level contextual background)
           ↓
Household Behavioural Adaptation
           ↓
High Adaptive Energy Vulnerability
           ↓
Energy-Poverty Risk Interpretation
```

The framework does **not** claim that FES directly causes household outcomes.
In the UK-only ENABLE dataset, FES is a shared annual macro-stress variable with
zero within-household variation.  It is treated as contextual background only.

---

## Why the Old Construct Labels Were Wrong

The previous version mislabelled H12 and H15 variables as "thermal discomfort":

| Variable | Correct interpretation (ENABLE codebook) | Old mislabel |
|----------|------------------------------------------|--------------|
| H12A | Proportion of incandescent bulbs in household | Heating satisfaction |
| H12B | Proportion of energy-efficient bulbs | Heating satisfaction |
| H15A–H15G | Environmental attitudes, lifestyle compromise, transition-cost resistance | Thermal comfort barriers |

The C-block variables (C1, C3, C4, C5, C7) are the correct heating/thermal
constraint variables but are largely absent from the UK sub-sample.

The revised framework replaces "thermal discomfort" with the codebook-correct
**Transition-Cost Resistance** construct (H15A–H15F), and introduces a new
**Energy Behavioural Lock-in** construct (E7A–E7E).

---

## How to Run

### Run Everything

```bash
python main.py                        # full pipeline (all 10 stages)
python main.py --fast                 # fast dev mode (fewer LSTM/TFT epochs)
python main.py --stage forecast       # Stages 0–4 only (models + FES)
python main.py --stage household      # Stages 5–8 only (ENABLE survey)
python main.py --stage ml             # Stages 9–10 only (CatBoost + SHAP)
```

### Run Each Sub-Pipeline Independently

```bash
# Stages 0–4: load data → train 4 models × 2 modes × 3 series → FES
python forecast_pipeline.py
python forecast_pipeline.py --fast                    # fewer epochs
python forecast_pipeline.py --skip-models LSTM TFT   # skip specific models
python forecast_pipeline.py --series gas carbon       # specific series only
python forecast_pipeline.py --fes-only                # skip training, recompute FES only
python forecast_pipeline.py --tune                    # hyperparameter tuning first
python forecast_pipeline.py --selection-basis validation

# Stages 5–8: ENABLE household survey stream
python household_stream.py             # with FES context attachment
python household_stream.py --no-fes   # skip FES attachment

# Stages 9–10: ML classification + SHAP
python ml_pipeline.py                  # CatBoost + SHAP
python ml_pipeline.py --no-shap        # skip SHAP (Stage 10)
```

---

## Full Pipeline

```
Stage 0–1: Load & Preprocess (macro)
    data_loader.py          -- parse UK macro series → growth rates
    preprocessing_macro.py  -- stationarity, ADF, feature engineering
           ↓
Stage 2: Forecast models (TWO modes per series)
    ┌──────────────────────────┬───────────────────────────────────┐
    │  Mode: CORE              │  Mode: MACRO                      │
    │  Target only             │  Target + lagged exogenous        │
    │  SARIMA                  │  SARIMAX                          │
    │  Prophet                 │  Prophet + macro regressors       │
    │  LSTM (univariate)       │  LSTM (multivariate)              │
    │  TFT (core only)         │  TFT (future covariates)          │
    └──────────────────────────┴───────────────────────────────────┘
    Train: 2005–2015  |  Evaluate: 2016  |  Forecast: Jan–Dec 2017
           ↓
Stage 3: Rank-aggregation model selection
    model_evaluation.py
    Metrics: MAE, RMSE, SMAPE, MASE, QL, PIC, WinklerScore, MSIS
    Note: MAPE excluded (near-zero growth rates cause explosion)
           ↓
Stage 4: FES construction
    fes_construction.py
    FES_core   → best core-only model per series
    FES_macro  → best macro-augmented model per series  (Robustness 1)
    FES_actual → realised 2017 values                   (Robustness 2)
    All z-scores use the same 2005–2016 reference distribution
           ↓
Stage 5: ENABLE household preprocessing
    enable_preprocessing.py
    UK sub-sample (Country=11, n≈1,015)
    Codebook-guided construct construction: FCP, AEMC, BLI, TCR
    AEV composite + HighAEV binary target
           ↓
Stage 6: Construct validation
    construct_validation.py
    Cronbach α, McDonald ω, CR, AVE, HTMT, Fornell–Larcker, EFA
           ↓
Stage 7: COR path analysis + mediation
    sem_mediation.py
    OLS directional associations + bootstrap mediation CIs
           ↓
Stage 8: Unsupervised latent robustness
    unsupervised_latent.py
    PCA / EFA / linear autoencoder vs. COR construct scores
           ↓
Stage 9: CatBoost classification
    ml_classification.py
    Target: high_aev  (AEV ≥ 75th percentile)
           ↓
Stage 10: SHAP explainability
    shap_explainability.py
    Which features drive predictions toward high AEV?
```

---

## Macro-FES Stream

### Core series (YoY growth rates)

| Series | Transformation |
|--------|---------------|
| Gas | YoY % — ONS RPI, used as-is |
| Electricity | CPI index (2015=100) → YoY % = (Index_t − Index_{t-12}) / Index_{t-12} × 100 |
| Carbon | Log-return × 100 = log(Price_t / Price_{t-12}) × 100 |

**Why carbon uses log-return:** percentage growth explodes when early EUA prices
approach zero.  Log-return is robust to near-zero denominators and bounded
below even in low-price regimes.

### Macro control variables (Dataset B)

CPIH housing-energy index, monthly GVA growth, UK temperature anomaly volatility,
NBP gas futures, grid electricity demand, wind/solar generation, and lagged
versions of all the above.

**Key rule:** macro variables never enter the FES equation directly.  They are
lagged exogenous inputs that help SARIMAX/Prophet/multivariate-LSTM produce
better core-series forecasts.

### FES Formula

```
FES_core_t   = z(GasGrowth_t^core)   + z(ElecGrowth_t^core)   + z(CarbonLR_t^core)   + z(Uncertainty_t^core)
FES_macro_t  = z(GasGrowth_t^macro)  + z(ElecGrowth_t^macro)  + z(CarbonLR_t^macro)  + z(Uncertainty_t^macro)
FES_actual_t = z(GasGrowth_t^actual) + z(ElecGrowth_t^actual) + z(CarbonLR_t^actual) + z(RealVol_t)
```

- `FES_actual` uses **realised volatility** in place of forecast uncertainty
  (it has no prediction intervals).
- All z-scores reference the same **2005–2016 training distribution** so that
  the three variants are directly comparable.
- **Annual FES** = mean(FES_Jan, …, FES_Dec) — same value for every UK household.

### Why FES is contextual, not household-level

The annual FES 2017 value is identical for all 1,015 UK respondents.  There is
zero within-country variation.  Entering FES as a household-level regression
predictor would be methodologically incoherent.  FES is therefore:

- Reported as contextual macro-stress background (three scenarios).
- Compared descriptively across FES_core, FES_macro, FES_actual.
- **NOT** included as a predictor in any regression or ML model.

---

## ENABLE UK Household Stream

### Data

ENABLE.EU household survey — UK sub-sample: Country = 11, n ≈ 1,015.

Missing/refusal codes replaced with NaN: `[9, 98, 99, 999, 9999, 99999]`

### Revised COR Constructs

The framework follows Conservation of Resources theory (Hobfoll 1989):
anticipated resource loss triggers defensive behavioural responses, which
in turn shape vulnerability outcomes.

| Construct | Short | Items (core) | Direction | Replaces |
|-----------|-------|-------------|-----------|---------|
| Financial–Energy Cost Pressure | FCP | S8, E2A, E2B | higher = more pressure | Perceived Energy Insecurity |
| Adaptive Energy-Management Capacity | AEMC | E5A2–4, E6A2–4, E6A6–8 (+ reverse: E5A1, E6A1) | higher = better capacity | Resource Preservation |
| Energy Behavioural Lock-in | BLI | E7A, E7B, E7C, E7D, E7E | higher = stronger lock-in | — (new) |
| Transition-Cost Resistance | TCR | H15A–C, H15E–F (+ reverse: H15D) | higher = more resistant | Thermal Discomfort (mislabel corrected) |

**Recoding rules:**
- `E5A1` ("I do not use reminders") → reverse-coded: 1 = no reminders → 0 adaptive capacity
- `E6A1` ("I do not have routines") → reverse-coded: 1 = no routines → 0 adaptive capacity
- `H15D` ("willing to make lifestyle compromises") → reverse-coded so high willingness = low resistance
- `H9` (heating control method) → recoded 0–3 scale: programmer + thermostat (3) > thermostat only (2) > programmer only (2) > manual (1) > none (0)

### High Adaptive Energy Vulnerability (HighAEV)

```
AEV_i = FCP_i + BLI_i + TCR_i + (1 − AEMC_i)

HighAEV_i = 1  if AEV_i ≥ P75(AEV)
           = 0  otherwise
```

All component scores are normalized to [0, 1] before combining.
A household has **high adaptive energy vulnerability** when it combines:
- Higher financial/energy cost pressure
- Stronger behavioural lock-in (hard to change habits)
- Greater resistance to transition costs
- Weaker adaptive energy-management capacity

### Optional Thermal Hardship Extension (future)

For completeness or journal robustness, a **Thermal Energy Poverty Risk** index
can be constructed from C-block variables (C1, C3, C4A–M, C5A–I, C7A–F) if
sufficient UK data are available.  These variables are the correct heating and
thermal constraint indicators.  **Do not mix them with H12/H15.**

---

## Construct Validation

Before SEM or ML, the construct module checks:

| Check | Metric | Threshold |
|-------|--------|-----------|
| Internal consistency | Cronbach's α | ≥ 0.60 |
| Reliability | McDonald's ω, CR | (descriptive) |
| Convergent validity | AVE | ≥ 0.50 |
| Convergent validity | Factor loadings | ≥ 0.40 |
| Discriminant validity | HTMT ratio | < 0.85 |
| Discriminant validity | Fornell–Larcker | AVE_i > r²_ij |

---

## SEM / Mediation Stream

Estimable COR-consistent directional pathways:

```
  FES 2017 (contextual background — NOT a predictor)
       ↓
  [Financial–Energy Cost Pressure]  ──a──►  [Adaptive Energy-Management Capacity]
         │                                              │
         │◄─── BLI ─────────────────────────────►      │
         │                                              ↓
         └──────────── c' (direct) ──────────►  [High Adaptive Energy Vulnerability]
                                                         ▲
  [Transition-Cost Resistance] ──────── e ──────────────┘
```

Estimation approach: OLS path analysis with bootstrap mediation (n=2000,
95% percentile CI) for the primary FCP → AEMC → AEV indirect pathway.

**Wording:** these estimates **test COR-consistent directional associations**.
They do **not** prove causal relationships.

---

## Unsupervised Latent Robustness

The unsupervised ML stream does NOT replace COR theory.  It tests whether the
theory-derived household vulnerability dimensions are recoverable from the
empirical structure of the data.

| Method | Description |
|--------|-------------|
| PCA | Linear variance decomposition (scree plot) |
| EFA | Exploratory factor analysis with Varimax rotation |
| Linear autoencoder | Keras encoder with 4-neuron linear bottleneck |

**Alignment metric:**  `corr(COR_construct_score, ML_latent_dimension)`

A Pearson r ≥ 0.40 between a COR construct score and a data-derived latent
dimension indicates the theory-specified dimension is empirically recoverable.

---

## CatBoost Classification

**Target:** `high_aev = 1` if AEV ≥ 75th percentile

**Features (no leakage):**

| Group | Variables |
|-------|----------|
| Construct scores | fcp_score, aemc_score, bli_score, tcr_score |
| Energy poverty | risk_category, low_income_flag, high_cost_flag |
| Household controls | S8, H1, H2, H3, S2, S3, S5, S6 |
| Energy variables | H5 (insulation), H6 (heating source), H13 (smart meter) |

**Excluded:**
- `aev_score` and all direct AEV components (target leakage)
- `fes_core/macro/actual` (constant across UK sample — zero variance)

---

## SHAP Explainability

SHAP values quantify how much each feature pushes a prediction above or below
the model's base rate.

**Correct interpretation:** SHAP explains the model's predictions.  It does
**not** prove causal influence on energy vulnerability.  The interpretation
focuses on: which financial, behavioural, household, and transition-attitude
variables are the most informative predictors, and in which direction do they
push predictions toward high AEV?

---

## Directory Structure

```
anticipatory-energy-stress/
│
├── src/
│   ├── config.py                   # Central configuration constants
│   ├── paths.py                    # All file path definitions
│   ├── construct_mapping.py        # Revised COR construct variable map
│   ├── enable_preprocessing.py     # ENABLE loading, recoding, AEV scoring
│   ├── construct_validation.py     # α, ω, AVE, HTMT, EFA
│   ├── sem_mediation.py            # COR path analysis + bootstrap mediation
│   ├── unsupervised_latent.py      # PCA / EFA / linear autoencoder
│   ├── ml_classification.py        # CatBoost → HighAEV
│   ├── shap_explainability.py      # SHAP analysis
│   │
│   ├── data_loader.py              # Parse raw UK macro data files
│   ├── preprocessing.py            # Macro feature engineering
│   ├── metrics_utils.py            # MAE, RMSE, SMAPE, MASE, PI metrics
│   ├── model_evaluation.py         # Rank-aggregation model selection
│   ├── fes_calculator.py           # FES index construction
│   ├── model_utils.py              # Macro alignment helpers
│   ├── plotting_utils.py           # Forecast / FES figures
│   ├── logging_utils.py            # Rotating file + coloured console log
│   ├── tuning.py                   # Hyperparameter grid search
│   │
│   └── models/
│       ├── sarima_model.py         # SARIMA / SARIMAX
│       ├── prophet_model.py        # Facebook Prophet
│       ├── lstm_model.py           # LSTM with MC Dropout
│       └── tft_model.py            # TFT / Attention-LSTM fallback
│
├── data/
│   ├── raw/                        # ONS, ICE, ENABLE source files
│   └── social_science_data/
│       └── ENABLE.EU_dataset_survey of households.xlsx
│
├── outputs/
│   ├── macro_forecasts/            # Per-model forecast CSVs
│   ├── fes/                        # fes_monthly_2017.csv, fes_annual_context.csv
│   ├── enable_cleaned/             # Cleaned UK data + construct scores
│   ├── construct_validation/       # α/ω/AVE/HTMT/EFA tables + figures
│   ├── sem_mediation/              # Path estimates, mediation, FES context
│   ├── unsupervised_latent_robustness/  # PCA/EFA/AE alignment tables
│   ├── ml_classification/          # CatBoost performance + predictions
│   ├── shap/                       # SHAP importance + plots
│   ├── figures/                    # Shared figures
│   ├── tables/                     # Shared tables
│   └── logs/
│
├── notebooks/
│   └── monitoring.ipynb
│
├── main.py                         # Master orchestrator — runs all three sub-pipelines
├── forecast_pipeline.py            # Stages 0–4: macro data → 4 models → FES index
├── household_stream.py             # Stages 5–8: ENABLE UK survey → constructs → SEM → latent
├── ml_pipeline.py                  # Stages 9–10: CatBoost HighAEV classifier → SHAP
└── requirements.txt
```

---

## Required Outputs Reference

### Forecast tables (`outputs/macro_forecasts/`)

```
{series}_growth_pct_forecasts_{model}_{mode}.csv
Columns: date | model | mode | forecast | lower_bound | upper_bound | actual
```

### FES (`outputs/fes/`)

| File | Description |
|------|-------------|
| `fes_monthly_2017.csv` | 12 rows × all z-components + FES_core, FES_macro, FES_actual |
| `fes_annual_context.csv` | Annual mean FES + FES context summary |
| `fes_component_decomposition.csv` | Cross-baseline component comparison |

### ENABLE (`outputs/enable_cleaned/`)

| File | Description |
|------|-------------|
| `item_diagnostics.csv` | Per-item missingness, variance, inclusion flag |
| `construct_variable_map.csv` | Item-to-construct mapping with directions |
| `construct_score_summary.csv` | Descriptive stats for all construct scores |
| `h12_codebook_correction.csv` | Documents H12 mislabel correction |
| `enable_aev_scored.csv` | Full UK dataset with all scores + HighAEV |

### Construct validation (`outputs/construct_validation/`)

| File | Description |
|------|-------------|
| `construct_reliability.csv` | α, ω, CR, AVE per construct |
| `factor_loadings.csv` | EFA loadings per item |
| `ave_cr_table.csv` | AVE and CR per construct |
| `htmt_matrix.csv` | HTMT ratios between construct pairs |
| `fornell_larcker_check.csv` | FL criterion per construct pair |
| `construct_correlation_matrix.csv` | Pearson r between composite scores |

### SEM / mediation (`outputs/sem_mediation/`)

| File | Description |
|------|-------------|
| `sem_path_estimates.csv` | OLS β, SE, t, p for all paths |
| `mediation_effects.csv` | a×b indirect + bootstrap 95% CI |
| `cor_mechanism_validation.csv` | COR pathway directional support summary |
| `fes_context_summary.csv` | FES annual values + methodology note |

### Unsupervised latent (`outputs/unsupervised_latent_robustness/`)

| File | Description |
|------|-------------|
| `pca_explained_variance.csv` | Scree data |
| `pca_loadings.csv` | PCA component loadings |
| `efa_loadings.csv` | EFA factor loadings |
| `unsupervised_latent_alignment.csv` | Pearson/Spearman r: COR scores vs. latents |
| `latent_best_match_summary.csv` | Best-matching latent dim per construct |

### ML classification (`outputs/ml_classification/`)

| File | Description |
|------|-------------|
| `ml_feature_list.csv` | Feature names + categorical flag |
| `catboost_performance.csv` | Accuracy, balanced accuracy, F1, ROC-AUC, PR-AUC |
| `confusion_matrix.csv` | 2×2 test-set confusion matrix |
| `classification_report.csv` | Per-class precision, recall, F1 |
| `enable_ml_predictions.csv` | Full-sample predictions + probabilities |

### SHAP (`outputs/shap/`)

| File | Description |
|------|-------------|
| `shap_feature_importance.csv` | Features ranked by mean |SHAP| |
| `shap_household_examples.csv` | SHAP breakdown for extreme-probability households |

---

## Required Figures

| Figure | Description |
|--------|-------------|
| `forecast_vs_actual_gas.png` | Historical + 2017 gas forecast vs. actual |
| `forecast_vs_actual_electricity.png` | Same for electricity |
| `forecast_vs_actual_carbon.png` | Same for carbon (log-return) |
| `fes_monthly_2017.png` | Monthly FES_core, FES_macro, FES_actual |
| `fes_annual_context.png` | Annual FES bar chart across three variants |
| `fes_component_analysis.png` | Z-score component decomposition |
| `construct_missingness.png` | Per-item missing rate |
| `construct_variability.png` | Per-item std (grey = zero variance) |
| `construct_score_distributions.png` | Histograms of FCP, AEMC, BLI, TCR, AEV |
| `factor_loadings.png` | EFA loadings bar chart per construct |
| `reliability_ave_cr.png` | α, ω, CR, AVE grouped bar chart |
| `htmt_matrix.png` | HTMT heatmap |
| `construct_correlation_matrix.png` | Pearson r heatmap |
| `cor_path_diagram.png` | COR path diagram with OLS coefficients |
| `cor_path_coefficients.png` | Path coefficient bar chart |
| `mediation_effects.png` | Indirect / direct / total effect with CI |
| `fes_context_bar.png` | FES annual values comparison |
| `latent_alignment_pca.png` | Alignment heatmap: PCA vs. COR |
| `latent_alignment_efa.png` | Alignment heatmap: EFA vs. COR |
| `latent_alignment_linear_ae.png` | Alignment heatmap: autoencoder vs. COR |
| `pca_scree.png` | PCA scree plot |
| `confusion_matrix.png` | HighAEV confusion matrix |
| `roc_curve.png` | ROC curve |
| `pr_curve.png` | Precision–Recall curve |
| `predicted_risk_distribution.png` | Predicted probability by true AEV class |
| `shap_bar_importance.png` | Mean |SHAP| bar chart (top 15) |
| `shap_beeswarm.png` | SHAP beeswarm plot |
| `shap_dependence_top_features.png` | SHAP dependence plots: top-3 features |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Full pipeline (all stages)
python main.py

# 3. Macro forecasting only (Stages 0–4)
python main.py --stage forecast

# 4. Household stream only (Stages 5–10)
#    (requires FES pipeline to have run first)
python main.py --stage household

# 5. Fast development mode (fewer LSTM/TFT epochs)
python main.py --fast

# 6. Skip specific models
python main.py --skip-models LSTM TFT

# 7. Specific series only
python main.py --series gas electricity
```

---

## Methodological Limitations

| Limitation | Impact |
|------------|--------|
| FES has no within-UK household variation | FES cannot be a household-level predictor; treated as contextual background |
| H12/H15 mislabelled in prior version | Construct redesign corrects this; old results are not comparable |
| C-block heating variables largely absent from UK sub-sample | Thermal energy poverty index deferred to optional robustness extension |
| ENABLE survey is cross-sectional | Path estimates test directional association, not temporal causality |
| SHAP explains model predictions | SHAP values are not causal effect estimates |
| Carbon percentage growth unstable near zero prices | Carbon uses log-return instead of percentage growth |
| Single country UK sub-sample | Results may not generalise across ENABLE's 11 countries |

---

## Dependencies

| Library | Purpose |
|---------|---------|
| `pandas`, `numpy` | Data wrangling |
| `scipy`, `statsmodels` | Statistical tests, OLS path analysis |
| `pmdarima` | SARIMA / SARIMAX auto-selection |
| `scikit-learn` | Scaling, train/test split, PCA, metrics |
| `factor_analyzer` | EFA (optional; falls back to PCA) |
| `prophet` | Prophet forecasting |
| `tensorflow` | LSTM (MC Dropout) + linear autoencoder |
| `torch`, `pytorch-forecasting` | TFT / Attention-LSTM fallback |
| `catboost` | Gradient boosting for HighAEV prediction |
| `shap` | SHAP explainability |
| `matplotlib`, `seaborn` | Static figures |
| `plotly` | Interactive HTML forecast timelines |
| `openpyxl` | Reading ENABLE.EU .xlsx file |
| `jupyter` | Monitoring notebook |

---

## Citation / Project Context

**Title:** Forecasting Anticipatory Energy–Carbon Stress and High Adaptive Energy Vulnerability in the UK: A Conservation of Resources and Explainable Machine Learning Framework

**Alternative:** Anticipatory Policy Intelligence for Energy Poverty: Forecasted Energy–Carbon Stress, Household Adaptive Capacity, and Just Transition Vulnerability

**Framework components:**

1. **FES_core** — primary anticipatory stress signal from core energy forecasts
2. **Robustness 1 (FES_macro)** — macro-augmented forecast
3. **Robustness 2 (FES_actual)** — realised-price benchmark
4. **Construct validation** — codebook-corrected COR constructs with psychometric checks
5. **COR path analysis** — FCP → AEMC → HighAEV, bootstrap mediation
6. **Unsupervised robustness** — PCA / EFA / linear autoencoder alignment
7. **CatBoost + SHAP** — explainable risk prediction for High Adaptive Energy Vulnerability

The z-score standardisation across all three FES variants uses the same 2005–2016
training distribution, ensuring differences reflect genuine methodological variation
rather than scaling artefacts.

---

*Anticipatory Energy–Carbon Stress Pipeline · UK 2005–2017*
