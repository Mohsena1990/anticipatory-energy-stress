# Anticipatory Energy-Carbon Stress Index

A production-grade Python pipeline for forecasting UK energy and carbon growth
rates (2005-2017 training window, 2018 monthly forecast horizon), constructing
the **Forecasted Energy-Carbon Stress Index (FES)**, and linking macro-level
stress to household-level behavioural responses via COR theory.

---

## Three Analytical Baselines

| Baseline | Role | Input |
|----------|------|-------|
| **FES_core** | *Main model* — primary anticipatory stress index | Core-only forecasts (gas, electricity, carbon growth) |
| **FES_macro** | *Robustness 1* — tests whether macro context improves signals | Core + exogenous macro (inflation, weather, GDP) as model inputs |
| **FES_actual** | *Robustness 2* — realised-price ground truth benchmark | Realised 2018 growth values |

**Key rule**: inflation, weather, and GDP never enter the FES equation directly.
They are exogenous inputs that help models produce better core forecasts.

### FES Formula

```
FES_core_t   = z(GasGrowth_t)        + z(ElecGrowth_t)        + z(CarbonGrowth_t)        + z(Uncertainty_t)
FES_macro_t  = z(Gas_hat_macro_t)    + z(Elec_hat_macro_t)    + z(Carbon_hat_macro_t)    + z(Uncertainty_t)
FES_actual_t = z(GasGrowth_actual_t) + z(ElecGrowth_actual_t) + z(CarbonGrowth_actual_t) + RealVol_t
```

All z-scores use the same **2005-2017 training-period reference** (mean and std)
so that the three variants are directly comparable.

---

## Full Pipeline

```
Stage 0-1: Load & Preprocess
    data_loader.py      -- parse raw UK data files into growth rate series
    preprocessing.py    -- stationarity checks, ADF, feature engineering
           |
           v
Stage 2: Forecast models (TWO modes per series)
    +--------------------------+-----------------------------+
    |  Mode: CORE              |  Mode: MACRO                |
    |  Target only             |  Target + exogenous         |
    |  SARIMA                  |  SARIMAX                    |
    |  Prophet (no regressors) |  Prophet (macro regressors) |
    |  LSTM (univariate)       |  LSTM (multivariate)        |
    |  TFT (core only)         |  TFT (future covariates)    |
    +--------------------------+-----------------------------+
    Train: 2005-2016  |  Evaluate: 2017  |  Refit: 2005-2017
    Forecast horizon: Jan-Dec 2018
           |
           v
Stage 3: Rank-aggregation model selection
    model_evaluation.py
    Score_m = sum_k Rank(Metric_k)
    Metrics: MAE, RMSE, SMAPE, MASE, QuantileLoss,
             PredictionIntervalCoverage, WinklerScore, MSIS
    Note: MAPE excluded (growth rates near zero cause explosion)
           |
           v
Stage 4: FES construction
    fes_calculator.py
    FES_core   -> best core-only model per series
    FES_macro  -> best macro model per series   (Robustness 1)
    FES_actual -> realised 2018 values           (Robustness 2)
    z-scores referenced to 2005-2017 training statistics
           |
           v
Stage 5: Social SEM -- COR composite-score path analysis
    run_sem.py
    Data: ENABLE.EU UK household survey (n=1,015, Country=11)
    FES 2018 annual mean -> contextual macro-stress background
    COR pathway (OLS path analysis + bootstrap mediation):
        Insecurity -> Resource Preservation -> Thermal Discomfort
           |
           v
Stage 6: ML + SHAP -- high discomfort risk prediction
    run_ml_shap.py
    CatBoost classifier: COR scores + household controls -> high_discomfort
    SHAP explainability: which features drive the prediction?
```

---

## Macro-to-Micro Link (COR Theory)

The anticipatory energy-carbon stress index captures the **macro-level** energy
market context.  Under Conservation of Resources theory (Hobfoll 1989), this
shared stress environment shapes **micro-level** household responses:

| COR construct | Observable measure |
|---------------|--------------------|
| Perceived resource threat | **Insecurity** (energy worry, cost perception, income difficulty) |
| Defensive resource protection | **Resource Preservation** (thermostat control, energy-saving behaviours) |
| Consequences of depletion | **Thermal Discomfort** (heating dissatisfaction, comfort barriers) |

FES 2018 annual mean is the same for all UK households (zero within-country
variance), so it is treated as **contextual macro-stress exposure** — NOT as a
household-level predictor.  The estimable behavioural pathway is the COR chain:

```
  FES 2018 (contextual background)
       |
  [Insecurity]  --a-->  [Resource Preservation]  --b-->  [Thermal Discomfort]
       |______________________c' (direct)__________________________|
```

### COR Items (UK sub-sample validated)

| Construct | Variables | Note |
|-----------|-----------|------|
| **Insecurity** | S8, E2A, E2B, E7A, E7B, E7C, E7D, E7E | Income difficulty + energy worry scale |
| **Resource Preservation** | H9, E5A1-E5A5, E5A9, E6A1-E6A8 | Thermostat control + energy-saving behaviours |
| **Thermal Discomfort** | H12A, H12B, H15A-H15E | Heating satisfaction + comfort barriers |

> Items C3, C4A-C4M, C1A, C1B, C5A-C5I, C7A-C7F are entirely absent from
> the UK sub-sample of this dataset and are therefore not used.

---

## Directory Structure

```
anticipatory-energy-stress/
|
+-- src/
|   +-- data_loader.py          # Parse UK raw files; core series -> YoY/MoM growth %
|   +-- preprocessing.py        # Missing values, ADF stationarity, feature engineering
|   +-- metrics_utils.py        # MAE, RMSE, SMAPE, MASE, QL, PIC, WS, MSIS
|   +-- model_evaluation.py     # Rank-aggregation selection (MAPE excluded from ranking)
|   +-- fes_calculator.py       # Three FES variants (core / macro / actual)
|   +-- plotting_utils.py       # All figures: forecast, FES, polar ranking, PI, interactive
|   +-- logging_utils.py        # Rotating file + coloured console logger
|   +-- run_sem.py              # Stage 5: COR path analysis + FES context
|   +-- run_ml_shap.py          # Stage 6: CatBoost + SHAP explainability
|   +-- sem_helpers.py          # (legacy) LIHC / HQRTM helpers; not imported by run_sem
|   +-- models/
|       +-- sarima_model.py     # SARIMA (core) / SARIMAX (macro)
|       +-- prophet_model.py    # Prophet -- core or with macro regressors
|       +-- lstm_model.py       # LSTM with MC Dropout; core and macro recursive forecast
|       +-- tft_model.py        # TFT / Attention-LSTM fallback; macro-aware rolling forecast
|
+-- data/
|   +-- raw/
|   |   +-- gas.csv                                        # ONS RPI YoY % gas
|   |   +-- electricity.csv                                # ONS CPI electricity index
|   |   +-- Carbon Emissions Futures Historical Data UK.csv # EUA futures price
|   |   +-- cpih08_188.xlsx                                # CPIH housing-energy index
|   |   +-- mgdp.csv                                       # ONS monthly GDP growth
|   |   +-- monthly-temperature-anomalies.csv              # UK temperature anomalies
|   |   +-- core_energy_carbon.csv                         # -> Dataset A (generated)
|   |   +-- macro_controls.csv                             # -> Dataset B (generated)
|   +-- social_science_data/
|       +-- ENABLE.EU_dataset_survey of households.xlsx    # ENABLE.EU survey (all countries)
|   +-- processed/
|
+-- outputs/
|   +-- logs/
|   +-- tables/
|   +-- figures/                # Forecast, FES, polar ranking, PI, 2018 comparisons
|   +-- forecasts/              # {series}_growth_pct_forecasts_{model}_{mode}.csv
|   +-- fes/                    # fes_monthly_2018.csv, fes_summary_2018.csv
|   +-- social_sem/             # Stage 5 outputs
|   |   +-- tables/             # 8 CSV tables (item diagnostics, path estimates, mediation...)
|   |   +-- figures/            # 7 figures (COR path diagram, distributions, FES context...)
|   |   +-- enable_fes_cor_scored.csv
|   |   +-- README_social_sem.md
|   +-- ml_shap/                # Stage 6 outputs
|       +-- tables/             # 5 CSV tables (performance, SHAP importance...)
|       +-- figures/            # 7 figures (beeswarm, ROC, PR, confusion matrix...)
|       +-- enable_ml_predictions.csv
|       +-- README_ml_shap.md
|
+-- notebooks/
|   +-- monitoring.ipynb        # Interactive dashboard (8 sections)
|
+-- run_pipeline.py             # Master pipeline orchestrator (Stages 0-4)
+-- requirements.txt
+-- README.md
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full forecasting + FES pipeline (Stages 0-4)
python run_pipeline.py

# 3. Fast development mode (fewer epochs)
python run_pipeline.py --fast

# 4. Skip specific models
python run_pipeline.py --skip-models LSTM TFT

# 5. Specific series only
python run_pipeline.py --series gas electricity

# 6. Stage 5: Social SEM + COR path analysis
python -m src.run_sem

# 7. Stage 6: CatBoost + SHAP explainability
python -m src.run_ml_shap

# 8. Launch monitoring notebook
jupyter notebook notebooks/monitoring.ipynb
```

---

## Data Sources (`data/raw/`)

| File | Series | Transformation | Source |
|------|--------|---------------|--------|
| `gas.csv` | Gas price change | Already YoY % -- used as-is | ONS MM23 CZDA |
| `electricity.csv` | Electricity CPI index (2015=100) | MoM % = (Index_t - Index_{t-1}) / Index_{t-1} x 100 | ONS MM23 D7DT |
| `Carbon...csv` | EUA futures price (EUR/tonne) | YoY % = (Price_t - Price_{t-12}) / Price_{t-12} x 100 | ICE / Investing.com |
| `cpih08_188.xlsx` | CPIH housing-energy index | MoM % | ONS CPIH Table 7 |
| `mgdp.csv` | Monthly GVA growth | Already MoM % -- used as-is | ONS Monthly GDP |
| `monthly-temperature-anomalies.csv` | Temperature anomaly -- UK filtered | RollingStd(12) | Our World in Data |
| `ENABLE.EU_dataset_...xlsx` | Household survey (11 countries) | UK sub-sample: Country==11, n=1,015 | ENABLE.EU project |

> All three core series (gas, electricity, carbon) are expressed as **growth rates (%)**
> for conceptual consistency and FES z-score validity.

---

## Outputs Reference

### Forecast CSVs (`outputs/forecasts/`)

```
{series}_growth_pct_forecasts_{model}_{mode}.csv
```
Columns: `date | model | mode | forecast | lower_bound | upper_bound | actual`

### FES CSVs (`outputs/fes/`)

| File | Description |
|------|-------------|
| `fes_monthly_2018.csv` | 12 rows x all z-components + FES_core, FES_macro, FES_actual |
| `fes_summary_2018.csv` | Annual mean FES and per-component z-score means |
| `fes_components_table.csv` | Cross-baseline component comparison table |

### Social SEM (`outputs/social_sem/`)

| File | Description |
|------|-------------|
| `tables/annual_fes_context.csv` | FES annual values + methodological notes |
| `tables/item_diagnostics.csv` | Per-item validity, missingness, variance |
| `tables/construct_variable_map.csv` | Item-to-construct mapping |
| `tables/cor_score_summary.csv` | Construct-level descriptive statistics |
| `tables/path_model_estimates.csv` | OLS path coefficients, SE, t, p |
| `tables/mediation_effects.csv` | Indirect a*b effect with bootstrap 95% CI |
| `tables/cor_mechanism_validation.csv` | COR pathway support summary |
| `tables/fes_context_summary.csv` | FES scenario descriptive comparison |
| `enable_fes_cor_scored.csv` | Final household dataset with all COR scores |
| `figures/cor_path_diagram.png` | COR path diagram with OLS coefficients |
| `figures/cor_score_distributions.png` | Histograms of the three composite scores |
| `figures/cor_path_coefficients.png` | Bar chart: a, b, c' path coefficients |
| `figures/mediation_effects.png` | Indirect / direct / total effect with CI |
| `figures/fes_context_bar.png` | Three FES annual scenario comparison |
| `figures/construct_item_missingness.png` | Per-item missing rate |
| `figures/construct_item_variability.png` | Per-item std (grey = zero variance, excluded) |

### ML + SHAP (`outputs/ml_shap/`)

| File | Description |
|------|-------------|
| `tables/ml_feature_list.csv` | Features used + categorical flag |
| `tables/catboost_performance.csv` | Accuracy, balanced accuracy, F1, ROC-AUC, PR-AUC |
| `tables/confusion_matrix.csv` | 2x2 test-set confusion matrix |
| `tables/classification_report.csv` | Per-class precision, recall, F1 |
| `tables/shap_feature_importance.csv` | Features ranked by mean abs SHAP |
| `enable_ml_predictions.csv` | Full-sample predictions + probabilities |
| `figures/shap_beeswarm.png` | SHAP beeswarm: feature impact distribution |
| `figures/shap_bar_importance.png` | Mean abs SHAP bar chart (top 15) |
| `figures/shap_dependence_top3.png` | SHAP dependence plots: top-3 features |
| `figures/confusion_matrix.png` | Confusion matrix heatmap |
| `figures/roc_curve.png` | ROC curve with AUC |
| `figures/pr_curve.png` | Precision-Recall curve with Average Precision |
| `figures/predicted_risk_distribution.png` | Predicted probability by true label |

---

## Metrics Reference

| Metric | Used in ranking? | Note |
|--------|-----------------|------|
| MAE | Yes | Primary point-forecast metric |
| RMSE | Yes | Penalises large errors |
| SMAPE | Yes | Symmetric -- robust to near-zero actuals |
| MASE | Yes | Relative to naive seasonal baseline |
| QuantileLoss | Yes | Pinball loss at q=0.5 |
| PredictionIntervalCoverage | Yes | % actuals inside 95% PI |
| WinklerScore | Yes | PI width + miss penalties |
| MSIS | Yes | Scaled interval score |
| **MAPE** | **No** | Excluded -- growth rates near zero cause explosion |

---

## Dependencies

| Library | Purpose |
|---------|---------|
| `pandas`, `numpy` | Data wrangling |
| `scipy` | Statistical utilities |
| `statsmodels`, `pmdarima` | SARIMA / SARIMAX; OLS path analysis |
| `scikit-learn` | MinMaxScaler, train/test split, evaluation metrics |
| `prophet` | Facebook Prophet forecasting |
| `tensorflow` | LSTM (Keras + MC Dropout) |
| `torch`, `pytorch-forecasting` | TFT / Attention-LSTM fallback |
| `catboost` | Gradient boosting for discomfort risk prediction |
| `shap` | SHAP explainability (CatBoost native + beeswarm plots) |
| `matplotlib`, `seaborn` | Static figures |
| `plotly` | Interactive HTML forecast timelines |
| `openpyxl` | Reading ENABLE.EU .xlsx survey file |
| `jupyter` | Monitoring notebook |

---

## Citation / Project Context

This pipeline implements the full Anticipatory Energy-Carbon Stress methodology:

1. **FES_core** -- primary anticipatory stress signal from core energy forecasts
2. **Robustness 1 (FES_macro)** -- alternative forecast augmented by macro exogenous inputs
3. **Robustness 2 (FES_actual)** -- realised-price benchmark validating forecast accuracy
4. **COR path analysis** -- insecurity -> preservation -> discomfort under annual FES context
5. **ML + SHAP** -- CatBoost-based risk prediction with feature-level SHAP explanations

The z-score standardisation across all three FES variants uses the same 2005-2017
training distribution, ensuring differences reflect genuine methodological variation
rather than scaling artefacts.

---

*Anticipatory Energy-Carbon Stress Index Pipeline -- UK 2005-2018*
