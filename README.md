# Forecasting Anticipatory Energy–Carbon Stress and High Adaptive Energy Vulnerability in the UK

**A Conservation of Resources and Explainable Machine Learning Framework**

---

## Research Objective

To construct a macro-level, scenario-based simulation of anticipatory UK energy–carbon stress (FES), to estimate household-level High Adaptive Energy Vulnerability (HighAEV) independently through three comparable COR estimation routes, and to interpret HighAEV's policy relevance under alternative macro energy–carbon stress scenarios.

---

## Conceptual Logic

```
Forecasted gas/electricity/carbon growth
           ↓
FES scenario-based signal simulation
           ↓
2017 UK macro energy–carbon stress context
           ↓
Three household vulnerability routes:
  Route 1 — COR Composite
  Route 2 — COR-informed SEM
  Route 3 — COR-informed VAE
           ↓
Cross-route comparison + HighAEV
           ↓
Scenario-conditioned interpretation and policy relevance
```

The framework does **not** claim that FES causes household outcomes, predicts
household vulnerability, or that households change behaviour because FES
rises or falls. FES is a macro-level scenario simulation; HighAEV is
estimated independently by the household routes. The two are linked only
through scenario-conditioned interpretation: FES scenarios describe the
macro-context under which the already-estimated HighAEV group is
interpreted, not an input to estimating it. In the UK-only ENABLE dataset,
every FES scenario is a shared annual value with zero within-household
variation, which is exactly why it cannot be a household-level predictor —
see [FES Scenario-Based Signal Simulation](#fes-scenario-based-signal-simulation)
and [Linking FES Scenarios and HighAEV](#linking-fes-scenarios-and-highaev).

---

**Transition-Cost Resistance** construct (H15A–H15F), and introduces a new
**Energy Behavioural Lock-in** construct (E7A–E7E).

---

## How to Run

### Run Everything

```bash
python main.py                        # full pipeline (all 12 stages: 0–11)
python main.py --fast                 # fast dev mode (fewer LSTM/TFT epochs)
python main.py --stage forecast       # Stages 0–4 only (models + FES — contextual background only)
python main.py --stage household      # Stages 5–9 only (ENABLE survey, 3 COR routes)
python main.py --stage ml             # Stages 10–11 only (CatBoost + SHAP)
```

### Run Each Sub-Pipeline Independently

```bash
# Stages 0–4 + 4b (TS-SHAP): load data → train 4 models × 2 modes × 3 series → FES → attribution
python forecast_pipeline.py
python forecast_pipeline.py --fast                    # fewer epochs
python forecast_pipeline.py --skip-models LSTM TFT   # skip specific models
python forecast_pipeline.py --series gas carbon       # specific series only
python forecast_pipeline.py --fes-only                # skip training, recompute FES only
python forecast_pipeline.py --tune                    # hyperparameter tuning first
python forecast_pipeline.py --selection-basis validation

# Stages 5–9: ENABLE household survey stream, 3 COR estimation routes
python household_stream.py             # with FES context attachment (metadata only, never a feature)
python household_stream.py --no-fes   # skip FES attachment
python household_stream.py --skip-vae # skip Stage 8 (Route 3 VAE, most expensive stage)

# Stages 10–11: ML classification (one model per route) + SHAP
python ml_pipeline.py                  # CatBoost + SHAP
python ml_pipeline.py --no-shap        # skip SHAP (Stage 11)
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
Stage 4: FES construction and scenario-based signal simulation
    fes_calculator.py + fes_scenarios.py
    3 method families (equal-weight / volatility-weighted / Bayesian) × 2
    modes (core / macro) = 6 forecasted signal scenarios, plus 3 realised
    benchmark scenarios = 9 named FES scenarios total (see
    [FES Scenario-Based Signal Simulation](#fes-scenario-based-signal-simulation)):
    ┌──────────────────────────────────────────────────────────────────┐
    │  equal_core, equal_macro     → equal-weight z-sum                │
    │  vw_core, vw_macro           → inverse-volatility weighted z-sum │
    │  bayesian_core, bayesian_macro → scalar Kalman filter posterior  │
    └──────────────────────────────────────────────────────────────────┘
    Realised benchmark scenarios (3 options, no PI available for actuals):
      actual_A — rolling 3-month std of mean(z_gas, z_elec, z_carbon)
      actual_B — cross-component std at each month (≈ existing fes_actual)
      actual_C — absolute shock: |mean_z − hist_mean| / hist_std
    Comparison metrics (6 forecasted scenarios × 3 realised benchmarks = 18 pairs):
      MAE, RMSE, Bias, MaxDev, Pearson r, Spearman r, R², Theil's U
    → outputs/fes/fes_comparison_metrics.csv (legacy)
    → outputs/fes/fes_scenario_summary.csv, fes_scenario_monthly_states.csv,
      fes_scenario_forecast_vs_actual_matrix.csv, fes_scenario_interpretation_notes.csv
    This is a macro-level signal-construction robustness layer — it is never
    a household-level predictor and never simulates household behaviour.
           ↓
Stage 4b: TS-SHAP attribution (run within forecast_pipeline, post-FES)
    ts_shap.py
    6 streams = 3 series × 2 modes (core / macro)
    SARIMA macro  → coefficient × (X_t − X̄_train)
    Prophet       → native component decomposition
    LSTM          → gradient × input over lookback window
    TFT           → attention weights + gradient × input
    → outputs/shap/ts/  +  outputs/figures/shap/
           ↓
Stage 5: ENABLE household data preparation
    enable_preprocessing.py
    UK sub-sample (Country=11, n≈1,015); clean missing/refusal codes;
    apply shared codebook recoding rules from construct_mapping.py; build
    normalized item columns n_{item}; attach FES as CONTEXT METADATA only
    (--no-fes to skip) — FES is never used as a model feature downstream.
           ↓
Stage 6: Route 1 — COR Composite Route
    enable_preprocessing.py + construct_validation.py + sem_mediation.py
    + unsupervised_latent.py
    Operationalises COR-informed FORMATIVE composites (not latent-variable
    extraction): fcp_score, aemc_score, bli_score, tcr_score → aev_score
    → high_aev (P75 threshold).  Includes construct validation (Cronbach
    α, McDonald ω, CR, AVE, HTMT, Fornell–Larcker, EFA), COR path analysis
    + bootstrap mediation, and empirical-recovery robustness (PCA / EFA /
    linear autoencoder alignment against the composite scores).
    → Main transparent, primary route.
           ↓
Stage 7: Route 2 — COR-informed SEM
    cor_sem.py
    Estimates REFLECTIVE latent variables through CFA/SEM (semopy, MLW):
    sem_fcp_latent, sem_aemc_latent, sem_bli_latent, sem_tcr_latent →
    sem_aev_score → sem_high_aev (route-specific P75). Reliability/
    validity (α, ω, CR, AVE, HTMT, Fornell–Larcker on CFA loadings) →
    second-order structural model AEV =~ FCP + BLI + TCR + AEMC (Option
    2A, with Heywood-case detection) → observed-outcome robustness check
    (Option 2B). Comparative/robustness route, evaluated cautiously — not
    presented as the primary measurement route regardless of fit quality.
    Resilient to failure: writes diagnostic CSVs and lets the pipeline
    continue if the CFA cannot be fit.
           ↓
Stage 8: Route 3 — COR-informed VAE (skippable via --skip-vae)
    cor_vae.py
    Learns COR-aligned deep latent REPRESENTATIONS through a VAE:
    vae_fcp_latent, vae_aemc_latent, vae_bli_latent, vae_tcr_latent →
    vae_aev_score → vae_high_aev (route-specific P75). Loss = reconstruction
    + beta·KL + lambda·COR-alignment (against Route 1 composites, training
    split only) + gamma·HighAEV prediction. Seed stability + prediction-head
    SHAP (over z1–z4). Not independent of Route 1 (alignment target +
    prediction label both come from Route 1). Resilient to failure: writes
    a diagnostic CSV and lets the pipeline continue if training fails.
           ↓
Stage 9: Cross-route comparison — the central comparison mechanism
    route_comparison.py
    Consumes each route's own already-computed outcome proxy (does not
    re-estimate anything). Route 1 vs 2 vs 3: per-construct Pearson r +
    Spearman r (after sign-canonicalization against Route 1), AEV/HighAEV
    agreement (Pearson r + Cohen's kappa + raw agreement rate), structured
    qualitative comparison table. Degrades gracefully to whichever routes
    are available (e.g. Route 1 vs 2 only if Route 3 was skipped/failed).
           ↓
Stage 10: CatBoost classification (one model per route)
    ml_classification.py
    Target: Route 1's high_aev (AEV ≥ 75th percentile) unless configured
    otherwise. FES columns and every route's own outcome proxy are
    excluded from every feature set (src.route_utils safeguards).
    Controls_Only model: the only fully generalizable predictor.
    Multi-model comparison: Controls_Only, Route1_Composite (circular),
    Route2_SEM / Route3_VAE (construct-overlap / representation-validation,
    not fully independent of Route 1), AllRoutes_Hybrid (not generalizable
    — multiple overlapping/circular components).
    Overfitting diagnostics: learning curve + 5-fold CV
           ↓
Stage 11: SHAP explainability (household-level)
    shap_explainability.py
    Which household features drive predictions toward high AEV?
    Run primarily on Controls_Only; also run on the best route model,
    labelled as construct-overlap / route-validation explanation, not a
    generalizable-prediction claim.
    (Route 3's own prediction-head SHAP, over z1–z4, is produced
    separately during Stage 8.)
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

### FES Formula — Construction Families

The FES equation below is built from **3 method families** (equal-weight,
volatility-weighted, Bayesian) in **2 modes** each (core / macro) = **6
forecasted scenarios**, plus **3 realised benchmark scenarios** = **9 named
FES scenarios** total (some earlier notes/figures call the 3 method
families "4 streamlines" — that count came from listing `equal_core` and
`equal_macro` as two separate rows; see
[FES Scenario-Based Signal Simulation](#fes-scenario-based-signal-simulation)
below for the definitive 9-scenario framing used throughout the rest of the
pipeline).

#### Equal-weight streamlines (original method)

```
FES_equal_core_t  = z(GasGrowth_t^core)  + z(ElecGrowth_t^core)  + z(CarbonLR_t^core)  + z(Uncertainty_t^core)
FES_equal_macro_t = z(GasGrowth_t^macro) + z(ElecGrowth_t^macro) + z(CarbonLR_t^macro) + z(Uncertainty_t^macro)
```

#### Volatility-Weighted FES (VW-FES)

Downweights series with high historical variability so that carbon (the most
volatile) does not dominate the index:

```
w_j = (1/σ_j) / Σ_k(1/σ_k)
VW-FES_t = Σ_j w_j × z_j,t   (weights renormalised over valid observations each month)
```

#### Bayesian FES (scalar Kalman filter)

A state-space model where the latent FES evolves as a random walk and each
z-scored component is a noisy observation of the latent state:

```
FES_t = FES_{t-1} + w_t,   w_t ~ N(0, σ²_Q)      (process equation)
z_j,t = FES_t    + v_j,t,  v_j,t ~ N(0, σ²_R)    (observation equation)
```

Parameters σ_Q and σ_R are estimated from the training z-matrix (2005–2016).
With n_v valid observations at month t the effective observation noise is
σ²_R / n_v (information pooling).  The filter returns a posterior mean and
posterior standard deviation, providing natural uncertainty bounds.

#### Actual FES benchmarks (3 options)

For actuals, prediction intervals are unavailable.  Three proxy volatility
terms replace the uncertainty component:

| Option | Formula |
|--------|---------|
| A | Rolling 3-month std of mean(z_gas, z_elec, z_carbon) |
| B | Cross-component std at each month t (reproduces original `fes_actual`) |
| C | Absolute shock: \|mean_z_t − hist_mean_z\| / hist_std_z |

#### Robustness comparison

All 6 forecasted FES scenarios are evaluated against all 3 realised
benchmark scenarios (6 × 3 = 18 pairs) using:
MAE, RMSE, Bias, MaxDev, Pearson r (+ p-value), Spearman r, R², Theil's U.
Results are saved to `outputs/fes/fes_comparison_metrics.csv` (legacy
labels) and, in the scenario-named long format, to
`outputs/fes/fes_scenario_forecast_vs_actual_matrix.csv`.

---

- All z-scores reference the same **2005–2016 training distribution** so that
  all scenarios are directly comparable.
- **Annual FES** = mean(FES_Jan, …, FES_Dec) — same value for every UK household.

### FES Scenario-Based Signal Simulation

The nine FES variants are used as a scenario-based signal simulation layer.
The six forecasted FES variants represent alternative anticipatory
macro-stress signals generated under different forecasting and
index-construction assumptions: equal-weight core, equal-weight macro,
volatility-weighted core, volatility-weighted macro, Bayesian core and
Bayesian macro. The three actual FES variants provide realised benchmark
scenarios. These scenarios are not treated as household-level predictors;
rather, they define alternative macro-contexts under which household
adaptive vulnerability is interpreted.

This simulation asks whether the 2017 UK energy–carbon stress context is
classified differently when the signal is constructed using equal
weighting, inverse-volatility weighting, Bayesian/Kalman smoothing, or
realised benchmark definitions. The output is therefore a robustness and
interpretation layer for macro stress, not a behavioural simulation of
households.

Concretely, `src/fes_scenarios.py` treats the nine variants as named
scenarios (`equal_core`, `equal_macro`, `vw_core`, `vw_macro`,
`bayesian_core`, `bayesian_macro`, `actual_A`, `actual_B`, `actual_C`) and
answers four descriptive questions for each:

1. Is 2017 low, moderate or high energy–carbon stress under this
   construction? (`classify_historical_fes_state` — relative to the
   2005–2016 training distribution: FES ≤ −0.50 → low, −0.50 < FES < +0.50 →
   moderate/neutral, FES ≥ +0.50 → high.)
2. Which component drives the signal — gas, electricity, carbon, or
   uncertainty/realised volatility? (`identify_dominant_component`, with a
   "mixed" result when the top two components are within 10% of each
   other.)
3. How close are the six forecasted scenarios to the three realised
   benchmark scenarios? (`compare_forecast_scenarios_to_actual_benchmarks`.)
4. Does the macro interpretation change across equal-weight,
   volatility-weighted and Bayesian constructions? (the within-2017
   relative/tercile state, `classify_relative_2017_state`, applied both
   across a scenario's own 12 months and across the 9 scenarios' annual
   means.)

### How the FES Scenario Simulation Is Shown

The scenario simulation is reported through four main visual outputs.
First, monthly scenario trajectories show the evolution of all forecasted
and realised FES variants across 2017. Second, an annual scenario-ranking
plot compares the mean FES value of each scenario against the zero-stress
baseline. Third, a component-contribution heatmap shows whether gas,
electricity, carbon, uncertainty or realised volatility drives the
aggregate signal. Fourth, forecast-vs-actual distance heatmaps compare the
six forecasted scenarios against the three realised benchmark scenarios.
Together, these figures show whether the macro-stress interpretation is
robust across alternative FES construction assumptions.

- `outputs/figures/fes_scenario_trajectories.png`
- `outputs/figures/fes_scenario_annual_ranking.png`
- `outputs/figures/fes_component_contribution_heatmap.png`
- `outputs/figures/fes_forecast_actual_distance_heatmap.png`
- `outputs/figures/fes_highaev_interpretation_matrix.png` (built once
  HighAEV is available — see
  [Linking FES Scenarios and HighAEV](#linking-fes-scenarios-and-highaev))

### TS-SHAP Attribution (Stage 4b)

`src/ts_shap.py` provides time-series SHAP attribution for the best-selected
model in each of the 6 streams (3 series × 2 modes).  Attribution methods are
model-specific:

| Model | Method |
|-------|--------|
| SARIMA core | Skipped — no exogenous features |
| SARIMA macro | β_j × (X_j,t − X̄_j,train) for each macro regressor |
| Prophet (any mode) | Native Prophet component decomposition (trend + seasonality + regressors) |
| LSTM (any mode) | Gradient × input integrated over the lookback window |
| TFT / Attention-LSTM (any mode) | Attention weights + gradient × input |

Outputs per stream: `{series}_{model}_{mode}_attribution.csv` and
`{series}_{model}_{mode}_importance.csv` in `outputs/shap/ts/`, with matching
bar-chart and heatmap figures in `outputs/figures/shap/`.

---

### Why FES is contextual, not household-level

Every one of the 9 FES scenario values (equal/VW/Bayesian × core/macro,
plus the 3 realised benchmarks) is identical for all 1,015 UK respondents.
There is zero within-country variation. Entering FES as a household-level
regression predictor would be methodologically incoherent. FES is
therefore:

- Reported as a macro-level scenario-based signal simulation (9 scenarios).
- Compared descriptively across all 9 scenarios (see
  [FES Scenario-Based Signal Simulation](#fes-scenario-based-signal-simulation)).
- Linked to HighAEV only through scenario-conditioned interpretation (see
  [Linking FES Scenarios and HighAEV](#linking-fes-scenarios-and-highaev)).
- **NOT** included as a predictor in any regression or ML model.

---

## ENABLE UK Household Stream

### FES is contextual, not a household predictor

FES is a macro-level scenario-based signal simulation, not a household-level
predictor. In this UK-only ENABLE sample every household shares the same
annual FES value within a given scenario (zero within-sample variation), so
the equal-weight scenario's annual value is attached to the household
DataFrame in Stage 5 as metadata/context and reported descriptively — it
must never be used as a household-level feature in Route 2's SEM, Route 3's
VAE, CatBoost, SHAP, mediation, or cross-route comparison. `src/route_utils.py`
is the single choke point (`exclude_fes_columns`) that enforces this across
all 9 FES scenario columns (and their underlying z-score components) in the
ML stream.

### Three Comparable COR Estimation Routes

The same four theoretical COR dimensions (FCP, AEMC, BLI, TCR) and each
route's own AEV/HighAEV outcome proxy are estimated from the same survey
items via three genuinely different methods, so the pipeline can ask "does
the COR pathway survive a change of estimation method?" rather than relying
on a single scoring choice. Cross-route comparison (Stage 9) is the central
comparison mechanism.

| Route | Method | Module | Stage | Score columns |
|-------|--------|--------|-------|----------------|
| **Route 1 — COR Composite** | Operationalises COR-informed **formative composites** (row-mean of normalized items) — not latent-variable extraction. Main transparent, primary route. | `enable_preprocessing.py` + `construct_validation.py` + `sem_mediation.py` + `unsupervised_latent.py` | 6 | `fcp_score`, `aemc_score`, `bli_score`, `tcr_score`, `aev_score`, `high_aev` |
| **Route 2 — COR-Informed SEM** | Estimates **reflective latent variables** through CFA/SEM. Comparative/robustness route, evaluated cautiously. | `cor_sem.py` | 7 | `sem_fcp_latent`, `sem_aemc_latent`, `sem_bli_latent`, `sem_tcr_latent`, `sem_aev_score`, `sem_high_aev` |
| **Route 3 — COR-Informed VAE** | Learns **COR-aligned deep latent representations** through a VAE. Not independent of Route 1 (alignment target + prediction label). | `cor_vae.py` | 8 (`--skip-vae` to skip) | `vae_fcp_latent`, `vae_aemc_latent`, `vae_bli_latent`, `vae_tcr_latent`, `vae_aev_score`, `vae_high_aev` |
| **Cross-route comparison** | Per-construct + outcome agreement across all three routes — central comparison mechanism | `route_comparison.py` | 9 | (reads the columns above; does not re-estimate) |

`src/construct_mapping.py` defines the single shared item vocabulary
(item lists, recoding rules, construct registry) that all three routes draw
on — each route estimates FCP/AEMC/BLI/TCR differently from the same items.
`src/route_utils.py` provides the shared cross-route helpers:
`compute_route_aev` (Route 1's `mean(FCP, BLI, TCR, 1−AEMC)` formula, reused
by Route 2/3 to build their own outcome proxy), `sign_canonicalize_against_route1`,
`minmax_scale_route_scores`, and `exclude_fes_columns`.

Route 2's `sem_fcp_latent`/… and Route 3's `vae_fcp_latent`/… columns are
merged onto the household DataFrame in `household_stream.py` once each
route succeeds, so `outputs/enable_cleaned/enable_aev_scored.csv` contains
Route 1 + (available) Route 2 + Route 3 columns side by side.

### Linking FES Scenarios and HighAEV

FES scenarios do not change household labels and are not entered as
household-level predictors. Instead, they provide the macro-context for
interpreting HighAEV. The HighAEV group represents households whose
adaptive vulnerability is structurally high, while the FES scenario
describes whether the surrounding national energy–carbon stress signal is
low, moderate or high under a particular signal-construction assumption.
This allows the study to ask whether vulnerability is only visible under
high macro stress, or whether it is already present even when aggregate
stress appears low.

The link between FES and HighAEV is therefore scenario-conditioned and
interpretive. The household model estimates who is vulnerable; the FES
scenario layer explains what macro stress context those vulnerabilities
are interpreted under.

Once the household routes have produced HighAEV (Stage 6/7/8) and Stage 4's
`fes_scenario_summary.csv` exists, `household_stream.py` builds
`outputs/fes_highaev_interpretation/scenario_highaev_interpretation_matrix.csv`
(`src.fes_scenarios.build_scenario_highaev_interpretation_matrix`) — one row
per FES scenario, each carrying the *same* already-estimated HighAEV
prevalence (Route 1, and Route 2/3 where available) alongside that
scenario's stress state and dominant macro signal. It does not model or
predict HighAEV. Wording follows: *"Under this macro signal scenario, the
HighAEV group represents households with structurally high adaptive
vulnerability observed under a low/moderate/high national energy–carbon
stress context."* It never states that FES causes HighAEV, predicts
household vulnerability, or that households changed behaviour because FES
increased.

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
AEV_i = mean(FCP_i, BLI_i, TCR_i, 1 − AEMC_i)

HighAEV_i = 1  if AEV_i ≥ P75(AEV)
           = 0  otherwise
```

All component scores are normalized to [0, 1] before combining; the
row-wise mean keeps AEV on the same [0, 1] scale (rather than [0, 4] under
a sum) and is missing-aware. `src.route_utils.compute_route_aev` applies
this same mean formula for Route 1, Route 2, and Route 3.
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

## Construct Validation (Route 1, Stage 6)

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

## SEM / Mediation Stream (Route 1, Stage 6)

OLS path analysis on Route 1's formative composite scores — deliberately
not a latent-variable SEM (see Route 2 below for the CFA/structural-model
equivalent). Estimable COR-consistent directional pathways:

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

## Unsupervised Latent Robustness (Route 1, Stage 6)

This stream does NOT replace COR theory, and is Route 1's own internal
robustness check (not a fourth "route" in its own right).  It tests whether
the theory-derived household vulnerability dimensions are recoverable from
the empirical structure of the data.

| Method | Description |
|--------|-------------|
| PCA | Linear variance decomposition (scree plot) |
| EFA | Exploratory factor analysis with Varimax rotation |
| Linear autoencoder | Keras encoder with 4-neuron linear bottleneck |

**Alignment metric:**  `corr(COR_construct_score, ML_latent_dimension)`

A Pearson r ≥ 0.40 between a COR construct score and a data-derived latent
dimension indicates the theory-specified dimension is empirically recoverable.

`build_item_matrix()` and `fit_encode_train_test()` from this module are
also reused by `src.ml_classification` for leakage-free PCA/EFA/linear-AE
feature fitting, and by Route 3 (`src.cor_vae`) to build its item matrix.

---

## Route 2 — COR-Informed SEM (Stage 7)

Where Route 1 **operationalises COR-informed formative composites**
(row-means of normalized items, validated post-hoc via EFA — not
latent-variable extraction), Route 2 **estimates reflective latent
variables through CFA/SEM**, specified upfront and measured with error via
`semopy` (maximum-likelihood, MLW estimator). Route 2 is a
comparative/robustness route, evaluated cautiously against Route 1 — its
fit indices and AVE are reported honestly rather than assumed adequate
(see [Methodological Limitations](#methodological-limitations)).

**Measurement model:**

```
FCP  =~ n_S8 + n_E2A + n_E2B
AEMC =~ n_E5A1 + n_E5A2 + n_E5A3 + n_E5A4 + n_E6A1 + n_E6A2 + n_E6A3
        + n_E6A4 + n_E6A6 + n_E6A7 + n_E6A8
BLI  =~ n_E7A + n_E7B + n_E7C + n_E7D + n_E7E
TCR  =~ n_H15A + n_H15B + n_H15C + n_H15D + n_H15E + n_H15F
```

Global fit is checked against conventional thresholds: CFI ≥ 0.90, TLI ≥
0.90, RMSEA ≤ 0.08, SRMR ≤ 0.08 (SRMR computed manually — semopy has no
built-in SRMR). Reliability/validity (α, ω, CR, AVE, HTMT, Fornell–Larcker)
is recomputed on the CFA loadings, reusing `construct_validation`'s generic
functions.

**Second-order structural model (Option 2A, primary):**

```
AEV =~ FCP + BLI + TCR + AEMC
```

AEMC enters directly (not reverse-scored) — a CFA loading's sign is free, so
a negative AEV=~AEMC loading is the latent-variable equivalent of Route 1's
formative `(1 − AEMC)` inversion. Second-order models with weakly-correlated
first-order factors can produce an inadmissible ("Heywood") solution
(near-zero or negative AEV factor variance); `fit_structural_model()`
detects this and flags the result rather than silently reporting an
unstable estimate.

**Observed-outcome robustness check (Option 2B):** regresses Route 1's
observed `aev_score`/`high_aev` on the four CFA factor scores — flagged
circular in the same spirit as Route 1's classification model, since the
scores are estimated from the same items that define the target.

**Route 2's own outcome proxy:** after fitting, the CFA factor scores
(`sem_fcp_latent`, `sem_aemc_latent`, `sem_bli_latent`, `sem_tcr_latent`)
are sign-canonicalized against Route 1's composites and min-max scaled to
`[0, 1]` (`src.route_utils`), then Route 1's AEV formula is applied to
produce `sem_aev_score` and, at Route 2's own 75th-percentile threshold,
`sem_high_aev`. Both are saved to `route2_factor_scores.csv` alongside the
four latent columns and consumed directly by Stage 9 (cross-route
comparison) and by `ml_classification`'s `Route2_SEM` model variant.

**Out-of-sample factor scoring** (`fit_cfa_train_test()`, used by
`src.ml_classification`) uses a Thurstone-style loading-weighted composite
of standardized items (train-split means/SDs/loadings only) rather than
semopy's built-in `Model.predict_factors()` — the latter inverts a joint
covariance system that is frequently singular for train-only subsamples of
this size.

**Resilience:** if the CFA measurement model itself fails to fit (e.g.
non-convergence or a singular covariance matrix), `cor_sem.run()` logs the
error, writes `cor_sem_failure_diagnostic.csv`, and returns `{}` rather than
raising — `household_stream.py` treats this as "Route 2 unavailable" and
continues to Stage 8, and Stage 9's cross-route comparison degrades
gracefully to whichever routes are present.

---

## Route 3 — COR-Informed VAE (Stage 8, skippable via `--skip-vae`)

Route 3 **learns COR-aligned deep latent representations through a VAE** —
a theory-informed Variational Autoencoder over the same 25-item set used by
Routes 1 and 2:

```
Raw items → Encoder (Dense/ReLU) → (mu, logvar)
          → Reparameterize: z = mu + exp(0.5·logvar)·eps
          → Decoder (mirror of encoder) → reconstruction
          → Prediction head (small MLP): z → HighAEV (sigmoid)
```

A plain VAE's four latent dimensions are not identifiable with any specific
COR construct, so this module forces interpretability via a COR-alignment
penalty against Route 1's composite scores (z1≈FCP, z2≈AEMC, z3≈BLI,
z4≈TCR — persisted/returned as `vae_fcp_latent`, `vae_aemc_latent`,
`vae_bli_latent`, `vae_tcr_latent`):

```
L = reconstruction_MSE
  + beta   * KL[N(mu,sigma) || N(0,I)]
  + lambda * COR_alignment_loss        (mean_i 1 - |corr(mu_i, cor_composite_i)|)
  + gamma  * BCE(prediction_head(z), high_aev)
```

This is an intentional, documented dependency of Route 3 on Route 1's
composite scores (used only as the alignment target and prediction label,
computed from the training split only inside `fit_vae_train_test`). The
four-term joint loss doesn't fit into a single Keras `.fit()` call cleanly,
so training uses a manual `tf.GradientTape` loop, full-batch (the ~500–750
complete-case sample is small and the alignment term's correlation is only
stable over a large-enough batch).

Reported per run: reconstruction MSE/MAE, KL divergence, alignment r per
construct, prediction-head ROC-AUC/PR-AUC on a held-out validation split,
Hungarian-aligned seed stability across `VAE_N_SEEDS` seeds, and SHAP
(KernelExplainer) on the 4-dim prediction head (labelled explicitly as VAE
prediction-head SHAP over the z1–z4 latent dimensions, not household-level
SHAP).

**Route 3's own outcome proxy:** the encoder's deterministic mean `mu`
(renamed to `vae_fcp_latent`/…) is sign-canonicalized against Route 1's
composites and min-max scaled to `[0, 1]`, then Route 1's AEV formula
produces `vae_aev_score` and, at Route 3's own 75th-percentile threshold,
`vae_high_aev`. Both are saved to `route3_vae_scores.csv` alongside the four
latent columns and consumed directly by Stage 9 (cross-route comparison)
and by `ml_classification`'s `Route3_VAE` model variant. Route 3 is **not
independent of Route 1** — its alignment target and prediction label both
come from Route 1's own scores/label — which is why Route2_SEM/Route3_VAE
are reported as construct-overlap / representation-validation models, not
generalizable predictors (see [CatBoost Classification](#catboost-classification)).

`fit_vae_train_test()` encodes both splits via the trained encoder's
deterministic mean `mu` (not the stochastic sample `z`) for stable
downstream ML features, and is used by `src.ml_classification`'s
`Route3_VAE`/`AllRoutes_Hybrid` model variants.

**Resilience:** if VAE training itself fails, `cor_vae.run()` logs the
error, writes `cor_vae_failure_diagnostic.csv`, and returns `{}` — treated
by `household_stream.py` as "Route 3 unavailable," same as `--skip-vae`.

---

## Cross-Route Comparison (Stage 9) — the central comparison mechanism

`route_comparison.py` does not re-estimate anything — each route already
computes its own outcome proxy upstream (Route 1: `aev_score`/`high_aev` in
`enable_preprocessing.py`; Route 2: `sem_aev_score`/`sem_high_aev` in
`cor_sem.py`; Route 3: `vae_aev_score`/`vae_high_aev` in `cor_vae.py`, all
via the shared `src.route_utils.compute_route_aev` formula). This module
only consumes those already-computed per-household scores and asks two
questions:

1. **Per-construct agreement** — how correlated are the three routes'
   FCP/AEMC/BLI/TCR estimates with each other? Both **Pearson r and
   Spearman r** are reported (`weak`/`moderate`/`strong` banding on Pearson).
2. **Outcome agreement** — how much do the three routes agree on which
   households are "High AEV"? **Pearson r** on the continuous AEV proxy;
   **Cohen's kappa** + raw agreement rate on the binary HighAEV proxy.

**Sign canonicalization:** Route 2's `sem_*_latent` and Route 3's
`vae_*_latent` construct columns are sign-canonicalized against Route 1's
composites before comparison (`src.route_utils.sign_canonicalize_against_route1`)
— a CFA loading's sign (Route 2) or a VAE latent's alignment correlation
(Route 3) is statistically free, so a dimension can converge to either
polarity. Each route's own `aev_score`/`high_aev`-equivalent column was
already computed with this same canonicalization applied internally, so it
is used as-is here.

A structured qualitative comparison table (`route_comparison_summary.csv`)
also records each route's main logic, how it represents the four
dimensions, whether measurement error and non-linearity are modelled, and
its intended role (main transparent/primary route / comparative-robustness
route / AI-oriented extension not independent of Route 1) — as data, not
prose.

**Graceful degradation:** availability is determined per route (Route 1 is
always present; Route 2/Route 3 are present only if their `scores_named`
DataFrame was returned or their persisted CSV exists). Only pairs between
*available* routes are compared — if Route 2 and/or Route 3 are unavailable
(fit failure, or Route 3 skipped via `--skip-vae`), the comparison
degrades to Route 1 alone or Route 1 vs Route 2, rather than emitting
placeholder rows for a route that never ran.

---

## CatBoost Classification (Stage 10)

**Target:** Route 1's `high_aev = 1` if `aev_score` ≥ 75th percentile,
unless explicitly configured otherwise.

### Model variants

One model per COR estimation route (see [Three Comparable COR Estimation
Routes](#three-comparable-cor-estimation-routes)), plus the controls-only
baseline. PCA/EFA/plain-linear-autoencoder latents are **not** separate
model variants here — they remain Route 1's internal empirical-recovery
check (`src.unsupervised_latent`), not standalone ML feature sets.

| Model key | Features | Interpretation |
|-----------|----------|----------------|
| **Controls_Only** | Energy-poverty proxies + household controls | **The only fully generalizable predictor** — no construct/latent features |
| Route1_Composite | Route 1 formative composite scores + controls | *Circular by construction* (see below) — theory validation only |
| Route2_SEM | Route 2 CFA factor scores (`fit_cfa_train_test`) + controls | Construct-overlap / representation-validation model — not fully generalizable (CFA scores from the same item pool as the target) |
| Route3_VAE | Route 3 VAE latent means (`fit_vae_train_test`) + controls | Construct-overlap / representation-validation model — not fully generalizable (HighAEV is already in the VAE's own training loss) |
| AllRoutes_Hybrid | Route 1 + Route 2 + Route 3 scores + controls | **Not generalizable prediction** — multiple overlapping/circular components present |

### Why Route1_Composite performance is near-perfect by construction

`high_aev` is defined as a **deterministic function** of the four Route 1
composite scores: `AEV = mean(FCP, BLI, TCR, 1 − AEMC)`, then thresholded at
P75. Using those same scores to predict `high_aev` recovers the target's own
components — this is circular, not generalizable prediction.
Route1_Composite is retained in the comparison table as a *theoretical
construct validation*, not as a predictive claim. Route2_SEM and Route3_VAE
are not circular by that same construction, but they are not fully
independent either — Route 2's factor scores are estimated from the same
item pool that defines the target, and Route 3's latents are both aligned
against Route 1's composites and trained with `high_aev` in their own loss
— so both are reported as **construct-overlap / representation-validation
models**, not generalizable predictors.

**The Controls_Only model is the only fully generalizable, meaningful
out-of-sample predictor.**

### Features — Controls_Only (primary model)

| Group | Variables |
|-------|----------|
| Energy poverty | risk_category, low_income_flag, high_cost_flag |
| Household controls | S8, H1, H2, H3, S2, S3, S5, S6 |
| Energy-efficiency controls | has_insulation, heating_gas_share, has_smart_meter — engineered by `enable_preprocessing.build_efficiency_controls()` from ENABLE's per-option sub-items (H5A1-3, H6A3, H13A/H13C); the bare H5/H6/H13 columns referenced by earlier versions of this table do not exist in the UK sub-sample, but this underlying data does |

**Always excluded** (`src/ml_classification.py`'s `_LEAKAGE_COLS`, enforced
defensively on every model variant's feature list via
`src.route_utils.exclude_fes_columns` plus an explicit leakage filter):
- `aev_score` / `high_aev` (Route 1's raw composite + target)
- `sem_aev_score` / `sem_high_aev` (Route 2's outcome proxy)
- `vae_aev_score` / `vae_high_aev` (Route 3's outcome proxy)
- `fes_core` / `fes_macro` / `fes_actual` — FES is contextual macro
  background (identical across the UK sample), never a household-level
  feature; a warning is logged if one is ever found in a feature list

### CatBoost regularization

Hyperparameters were tuned for the ~800–1,000-sample dataset size.  Depth 3 and
`l2_leaf_reg` 10 are the two strongest levers against overfitting at this scale.
`l2_leaf_reg` was raised from 8 to 10 after re-validating by 5-fold CV across
3 seeds: same mean test AUC as 8, consistently smaller train/test overfit gap.
`depth=4` was also tried and rejected — a marginal AUC gain came with a much
larger overfit gap, conflicting with the priority on generalisation over
training accuracy at this sample size.

| Hyperparameter | Value | Purpose |
|----------------|-------|---------|
| `iterations` | 600 | Upper bound (early stopping used) |
| `learning_rate` | 0.02 | Slower learning rate, compensated by more iterations |
| `depth` | 3 | Shallower trees — primary anti-overfitting lever |
| `l2_leaf_reg` | 10 | Stronger L2 regularization on leaf weights |
| `subsample` | 0.75 | Row subsampling per tree |
| `colsample_bylevel` | 0.7 | Feature subsampling per split |
| `min_data_in_leaf` | 20 | Prevents splits on very small subgroups |
| `random_strength` | 1.5 | Bayesian noise added to split scoring |
| `early_stopping_rounds` | 40 | Stops when validation AUC does not improve |
| `use_best_model` | True | Predictions always use best iteration, not last |

### Optimal decision threshold

The default 0.5 classification threshold undershoots recall on the minority
class (HighAEV ≈ 25%) with imbalanced data.  At the end of training, the
threshold is swept from 0.05 to 0.95 and the value maximising F1 on the test
set is selected.  The chosen threshold is stored in `decision_threshold` in
`catboost_performance.csv`.

### Leakage-free encoder/factor/VAE training (Route2_SEM / Route3_VAE / AllRoutes_Hybrid)

Pre-computed latent scores trained on the full dataset (including test
households) are a mild form of transductive leakage. For each model variant
that uses latent features, the relevant route's estimator is instead fitted
exclusively on the CatBoost training split and applied to the held-out test
split: `fit_encode_train_test` (PCA/EFA/AE, `src/unsupervised_latent.py`),
`fit_cfa_train_test` (Route 2, `src/cor_sem.py`), and `fit_vae_train_test`
(Route 3, `src/cor_vae.py`) — dispatched by `_fit_latent_train_test()` in
`src/ml_classification.py`. Route 3's fitting additionally slices Route 1's
composite scores and `high_aev` labels to the training rows only, since its
alignment target and prediction head both need them.
The high AUC of the Route1_Composite variant remains primarily driven by
**structural circularity** — its features are (or are estimated from) the
same item-level data that defines the AEV target — not by transductive
leakage.

### Overfitting diagnostics

1. **Learning curve** (`learning_curve_controls_only.png`) — train vs. validation
   AUC over boosting iterations with early-stopping marker.

2. **5-fold stratified cross-validation** (`catboost_cv_results.csv`,
   `cv_overfitting_diagnostics.png`) — reports train/test AUC per fold and
   the overfit gap (train_AUC − test_AUC).  A gap > 0.05 signals excessive
   memorisation; > 0.10 is critical.

3. **Train metrics + threshold in performance table** — `catboost_performance.csv`
   includes `train_roc_auc`, `overfit_gap_auc`, and `decision_threshold`.

4. **Feature importance** — `feature_importance_controls_only.csv` and
   `feature_importance_controls_only.png` rank features by CatBoost's
   PredictionValuesChange importance metric.

---

## SHAP Explainability (Stage 11)

SHAP values quantify how much each feature pushes a prediction above or below
the model's base rate.

**Correct interpretation:** SHAP explains the model's predictions.  It does
**not** prove causal influence on energy vulnerability.  The interpretation
focuses on: which financial, behavioural, household, and transition-attitude
variables are the most informative predictors, and in which direction do they
push predictions toward high AEV?

Stage 11 runs `shap_explainability.py` primarily on the Controls_Only
model. It also runs on the best route model from the comparison table
(excluding the fully circular Route1_Composite/AllRoutes_Hybrid) — this
second run is labelled a construct-overlap / route-validation explanation,
not a generalizable-prediction claim, since Route2_SEM/Route3_VAE are
themselves not fully independent of Route 1.

Route 3 (VAE) additionally has its own **prediction-head SHAP**, computed
during Stage 8 via `shap.KernelExplainer` on the 4-dim latent `mu`
(explicitly labelled as SHAP over the VAE's z1–z4 latent dimensions, not
household-level feature SHAP) and routed through the same
`shap_explainability.run()` pipeline via its `precomputed_shap_values`
parameter (which lets non-CatBoost models reuse the table/plotting code
unchanged).

---

## Directory Structure

```
anticipatory-energy-stress/
│
├── src/
│   ├── config.py                   # Central configuration constants
│   ├── paths.py                    # All file path definitions
│   ├── construct_mapping.py        # Shared COR item vocabulary across all 3 routes
│   ├── route_utils.py              # Shared cross-route helpers + FES safeguard
│   ├── enable_preprocessing.py     # Route 1: ENABLE loading, recoding, formative composite scoring
│   ├── construct_validation.py     # Route 1: α, ω, AVE, HTMT, EFA
│   ├── sem_mediation.py            # Route 1: COR path analysis + bootstrap mediation
│   ├── unsupervised_latent.py      # Route 1: PCA / EFA / linear autoencoder
│   ├── cor_sem.py                  # Route 2: CFA measurement + second-order structural model
│   ├── cor_vae.py                  # Route 3: theory-informed VAE
│   ├── route_comparison.py         # Cross-route (1 vs 2 vs 3) comparison
│   ├── ml_classification.py        # CatBoost → HighAEV (one model per route)
│   ├── shap_explainability.py      # SHAP analysis
│   │
│   ├── data_loader.py              # Parse raw UK macro data files
│   ├── preprocessing.py            # Macro feature engineering
│   ├── metrics_utils.py            # MAE, RMSE, SMAPE, MASE, PI metrics
│   ├── model_evaluation.py         # Rank-aggregation model selection
│   ├── fes_calculator.py           # FES index: 6 forecasted + 3 actual scenarios + robustness comparison
│   ├── fes_scenarios.py            # FES scenario simulation layer: 9 named scenarios, stress-state classification, scenario↔HighAEV interpretation
│   ├── ts_shap.py                  # TS-SHAP attribution for selected models
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
│   ├── forecasts/                  # Per-model forecast CSVs
│   ├── fes/                        # FES monthly/summary CSVs + comparison metrics + 9-scenario tables
│   ├── fes_highaev_interpretation/ # FES scenario ↔ HighAEV interpretation matrix (interpretive only)
│   ├── models/                     # Saved model checkpoints for TS-SHAP (.pt, .pkl)
│   ├── shap/
│   │   ├── ts/                     # TS-SHAP attribution CSVs per stream
│   │   └── (catboost SHAP outputs)
│   ├── enable_cleaned/             # Cleaned UK data + Route 1/2/3 score columns (merged)
│   ├── construct_validation/       # Route 1: α/ω/AVE/HTMT/EFA tables + figures
│   ├── sem_mediation/              # Route 1: path estimates, mediation, FES context
│   ├── unsupervised_latent_robustness/  # Route 1: PCA/EFA/AE alignment tables
│   ├── cor_sem/
│   │   ├── tables/                 # Route 2: CFA loadings, fit indices, structural paths
│   │   └── figures/
│   ├── cor_vae/
│   │   ├── tables/                 # Route 3: reconstruction/KL/alignment/prediction metrics
│   │   └── figures/
│   ├── route_comparison/
│   │   ├── tables/                 # Cross-route construct/outcome agreement
│   │   └── figures/
│   ├── ml_classification/          # CatBoost performance + predictions (per route)
│   ├── figures/                    # Shared figures (incl. figures/shap/ for TS-SHAP)
│   ├── tables/                     # Shared tables
│   └── logs/
│
├── notebooks/
│   └── monitoring.ipynb
│
├── main.py                         # Master orchestrator — runs all three sub-pipelines
├── forecast_pipeline.py            # Stages 0–4: macro data → 4 models → FES index (contextual background)
├── household_stream.py             # Stages 5–9: ENABLE UK survey → 3 COR routes → comparison
├── ml_pipeline.py                  # Stages 10–11: CatBoost HighAEV classifier (per route) → SHAP
├── Chapter3_Data_and_Methodology.md              # Dissertation write-up drafts, one per
├── Chapter4_Forecasting_and_FES_Results.md       # pipeline stream — mirror the outputs
├── Chapter5_ENABLE_Construct_Redesign_and_Validation.md  # above, not re-derived here
├── Chapter6_COR_Path_Analysis_and_Mediation.md
├── Chapter7_Unsupervised_Latent_Robustness.md
├── Chapter8_CatBoost_Classification_and_SHAP.md
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
| `fes_monthly_2017.csv` | 12 rows × all z-components + all 6 forecasted FES scenarios + 3 realised benchmarks |
| `fes_summary_2017.csv` | Annual mean per scenario + method note |
| `fes_components_table.csv` | Cross-baseline component comparison |
| `fes_comparison_metrics.csv` | 6 forecasted scenarios × 3 realised benchmarks × 9 metrics (legacy labels) |
| `fes_scenario_summary.csv` | One row per FES scenario (9 rows): annual mean, historical + relative-2017 stress state, dominant component, interpretation note |
| `fes_scenario_monthly_states.csv` | Long format, one row per month per scenario (108 rows): FES value, historical/relative state, dominant component that month |
| `fes_scenario_forecast_vs_actual_matrix.csv` | Long format: 6 forecasted scenarios × 3 realised benchmarks × 8 metrics |
| `fes_scenario_interpretation_notes.csv` | One row per scenario: stress state, dominant component, interpretation, policy relevance, caution note |

**Scenario columns in `fes_monthly_2017.csv`** (named scenarios per `src/fes_scenarios.py`; scenario name → column):

| Scenario name | Column | Description |
|--------|--------|-------------|
| `equal_core` / `equal_macro` | `fes_core` / `fes_macro` | Equal-weight FES |
| `vw_core` / `vw_macro` | `fes_vw_core` / `fes_vw_macro` | Volatility-weighted FES (VW-FES) |
| `bayesian_core` / `bayesian_macro` | `fes_bayes_core` / `fes_bayes_macro` | Bayesian (Kalman) FES posterior mean (+ `_std`/`_lb`/`_ub`) |
| `actual_A` / `actual_B` / `actual_C` | `fes_actual_A` / `fes_actual_B` / `fes_actual_C` | Three realised-benchmark FES scenarios |

### FES ↔ HighAEV interpretation (`outputs/fes_highaev_interpretation/`)

| File | Description |
|------|-------------|
| `scenario_highaev_interpretation_matrix.csv` | One row per FES scenario (9 rows): stress state, dominant macro signal, already-estimated HighAEV prevalence per route, route-agreement note, interpretation, policy implication, caution note. Interpretive only — never a household-level prediction. |

### TS-SHAP (`outputs/shap/ts/` + `outputs/figures/shap/`)

| File pattern | Description |
|---|---|
| `{series}_{model}_{mode}_attribution.csv` | Step-by-step attribution per feature per 2017 month |
| `{series}_{model}_{mode}_importance.csv` | Features ranked by mean absolute attribution |
| `{series}_{model}_{mode}_importance.png` | Horizontal bar chart of mean \|attribution\| |
| `{series}_{model}_{mode}_heatmap.png` | Attribution heatmap: features × 2017 months |

### ENABLE (`outputs/enable_cleaned/`)

| File | Description |
|------|-------------|
| `item_diagnostics.csv` | Per-item missingness, variance, inclusion flag |
| `construct_variable_map.csv` | Item-to-construct mapping with directions |
| `construct_score_summary.csv` | Descriptive stats for all construct scores |
| `h12_codebook_correction.csv` | Documents H12 mislabel correction |
| `enable_aev_scored.csv` | Full UK dataset — Route 1 scores + HighAEV always present; Route 2 (`sem_*_latent`, `sem_aev_score`, `sem_high_aev`) and Route 3 (`vae_*_latent`, `vae_aev_score`, `vae_high_aev`) columns merged in once `household_stream.py` runs and those routes succeed; FES columns present as context metadata only |

### Construct validation — Route 1 (`outputs/construct_validation/`)

| File | Description |
|------|-------------|
| `construct_reliability.csv` | α, ω, CR, AVE per construct |
| `factor_loadings.csv` | EFA loadings per item |
| `ave_cr_table.csv` | AVE and CR per construct |
| `htmt_matrix.csv` | HTMT ratios between construct pairs |
| `fornell_larcker_check.csv` | FL criterion per construct pair |
| `construct_correlation_matrix.csv` | Pearson r between composite scores |

### SEM / mediation — Route 1 (`outputs/sem_mediation/`)

| File | Description |
|------|-------------|
| `sem_path_estimates.csv` | OLS β, SE, t, p for all paths |
| `mediation_effects.csv` | a×b indirect + bootstrap 95% CI |
| `cor_mechanism_validation.csv` | COR pathway directional support summary |
| `fes_context_summary.csv` | FES annual values + methodology note |

### Unsupervised latent — Route 1 (`outputs/unsupervised_latent_robustness/`)

| File | Description |
|------|-------------|
| `pca_explained_variance.csv` | Scree data |
| `pca_loadings.csv` | PCA component loadings |
| `efa_loadings.csv` | EFA factor loadings |
| `unsupervised_latent_alignment.csv` | Pearson/Spearman r: COR scores vs. latents |
| `latent_best_match_summary.csv` | Best-matching latent dim per construct |

### Route 2 — COR-SEM (`outputs/cor_sem/tables/`)

| File | Description |
|------|-------------|
| `cfa_measurement_loadings.csv` | Standardized CFA loadings per item/factor |
| `cfa_fit_indices.csv` | DoF, χ², CFI, TLI, RMSEA, GFI, AIC, BIC, SRMR + threshold pass flags |
| `cfa_item_exclusions.csv` | Items dropped for near-zero variance in the complete-case CFA sample |
| `cfa_reliability_validity.csv` | α, ω, CR, AVE per factor (CFA loadings) |
| `cfa_htmt_matrix.csv` | HTMT ratios between CFA factors |
| `cfa_factor_correlations.csv` | Fitted inter-factor correlation matrix |
| `cfa_fornell_larcker_check.csv` | FL criterion on CFA AVE/correlations |
| `structural_paths_2A.csv` | Second-order AEV structural loadings (Option 2A) |
| `structural_fit_indices_2A.csv` | Fit indices for the structural model |
| `structural_paths_2B_observed_outcome.csv` | OLS: observed AEV ~ CFA factor scores (circular check) |
| `mediation_2B_cfa_scores.csv` | Bootstrap mediation on CFA factor scores |
| `cor_sem_failure_diagnostic.csv` | Written only if the CFA measurement model fails to fit — records the error so the pipeline can continue |
| `route2_factor_scores.csv` | Per-household `sem_fcp_latent`/`sem_aemc_latent`/`sem_bli_latent`/`sem_tcr_latent` CFA factor scores + `sem_aev_score`/`sem_high_aev` (Route 2's own outcome proxy) |

### Route 3 — COR-VAE (`outputs/cor_vae/tables/`)

| File | Description |
|------|-------------|
| `vae_training_history.csv` | Per-epoch train/val loss components (total, recon, KL, align, BCE) |
| `vae_reconstruction_metrics.csv` | Final recon MSE/MAE, KL, alignment loss, prediction BCE |
| `vae_cor_alignment.csv` | Pearson/Spearman r: VAE latent dims vs. Route 1 composites |
| `vae_prediction_metrics.csv` | Prediction-head accuracy/F1/ROC-AUC/PR-AUC on held-out validation split |
| `vae_seed_stability.csv` | Hungarian-aligned |r| per construct across `VAE_N_SEEDS` seeds |
| `cor_vae_failure_diagnostic.csv` | Written only if VAE training fails — records the error so the pipeline can continue |
| `route3_vae_scores.csv` | Per-household `vae_fcp_latent`/`vae_aemc_latent`/`vae_bli_latent`/`vae_tcr_latent` (mu) latent scores + `vae_aev_score`/`vae_high_aev` (Route 3's own outcome proxy) |

### Cross-route comparison (`outputs/route_comparison/tables/`)

| File | Description |
|------|-------------|
| `cross_route_construct_agreement.csv` | Pearson r + Spearman r per construct, each *available* route pair |
| `cross_route_outcome_agreement.csv` | AEV Pearson r + HighAEV Cohen's kappa/agreement rate, each *available* route pair |
| `route_comparison_summary.csv` | Structured qualitative comparison (logic, dimensions, metrics, role) per route |

### ML classification (`outputs/ml_classification/`)

| File | Description |
|------|-------------|
| `ml_feature_list.csv` | Feature names + categorical flag |
| `catboost_performance.csv` | Accuracy, balanced accuracy, F1, ROC-AUC, PR-AUC, train AUC, overfit gap, decision threshold |
| `catboost_cv_results.csv` | Per-fold train/test AUC, overfit gap, best iteration (5-fold stratified CV) |
| `confusion_matrix.csv` | 2×2 test-set confusion matrix |
| `classification_report.csv` | Per-class precision, recall, F1 |
| `model_comparison_table.csv` | All 5 model variants (Controls_Only + one per route + hybrid) with metrics + interpretation |
| `feature_importance_controls_only.csv` | PredictionValuesChange importance per feature |
| `enable_ml_predictions.csv` | Full-sample predictions + probabilities |

### SHAP (`outputs/shap/`)

| File | Description |
|------|-------------|
| `shap_feature_importance.csv` | Features ranked by mean |SHAP| |
| `shap_household_examples.csv` | SHAP breakdown for extreme-probability households |
| `route3_vae_shap_feature_importance.csv` | Route 3 prediction-head SHAP, ranked by mean |SHAP| over z1–z4 |
| `route3_vae_shap_household_examples.csv` | Route 3 SHAP breakdown for extreme-probability households |

---

## Required Figures

| Figure | Description |
|--------|-------------|
| `forecast_vs_actual_gas.png` | Historical + 2017 gas forecast vs. actual |
| `forecast_vs_actual_electricity.png` | Same for electricity |
| `forecast_vs_actual_carbon.png` | Same for carbon (log-return) |
| `fes_monthly_2017.png` | Monthly FES for the equal/macro + actual baselines |
| `fes_components_2017.png` | Z-score component decomposition across baselines |
| `fes_robustness_variants.png` | All 6 forecasted FES scenarios + 3 actual benchmarks overlay |
| `fes_bayesian_uncertainty_core.png` | Bayesian FES posterior ± 95% band (core) |
| `fes_bayesian_uncertainty_macro.png` | Bayesian FES posterior ± 95% band (macro) |
| `fes_metrics_rmse_heatmap.png` | Heatmap: RMSE of 6 forecasted scenarios vs 3 actual benchmarks |
| `fes_metrics_pearson_r_heatmap.png` | Heatmap: Pearson r of 6 forecasted scenarios vs 3 actual benchmarks |
| `fes_scenario_trajectories.png` | Monthly trajectories of all 9 named FES scenarios, grouped forecasted vs realised |
| `fes_scenario_annual_ranking.png` | Annual mean FES per scenario, ranked, labelled by stress state |
| `fes_component_contribution_heatmap.png` | Annual mean z-score contribution: core/macro/actual × gas/electricity/carbon/uncertainty/FES total |
| `fes_forecast_actual_distance_heatmap.png` | 6 forecasted scenarios × 3 realised benchmarks, RMSE distance |
| `fes_highaev_interpretation_matrix.png` | Compact table figure: FES scenario × stress state × dominant signal × HighAEV prevalence × interpretation |
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
| `cfa_loadings.png` (`outputs/cor_sem/figures/`) | Route 2: standardized CFA loadings per factor |
| `cfa_fit_indices.png` (`outputs/cor_sem/figures/`) | Route 2: CFI/TLI/RMSEA/SRMR vs. threshold |
| `structural_paths_2A.png` (`outputs/cor_sem/figures/`) | Route 2: second-order AEV structural loadings (flags Heywood cases) |
| `vae_training_curves.png` (`outputs/cor_vae/figures/`) | Route 3: total/recon/KL/alignment loss curves |
| `vae_cor_alignment_heatmap.png` (`outputs/cor_vae/figures/`) | Route 3: VAE latent dims vs. Route 1 composites |
| `vae_seed_stability.png` (`outputs/cor_vae/figures/`) | Route 3: |r| boxplots per construct across seeds |
| `vae_prediction_curves.png` (`outputs/cor_vae/figures/`) | Route 3: prediction-head ROC + PR curves |
| `cross_route_construct_agreement.png` (`outputs/route_comparison/figures/`) | Route 1 vs 2 vs 3: per-construct Pearson r heatmap |
| `cross_route_outcome_agreement.png` (`outputs/route_comparison/figures/`) | Route pairs: AEV Pearson r + HighAEV Cohen's kappa |
| `cross_route_aev_distributions.png` (`outputs/route_comparison/figures/`) | AEV proxy distribution overlay, all 3 routes |
| `confusion_matrix.png` | HighAEV confusion matrix |
| `roc_curve.png` | ROC curve |
| `pr_curve.png` | Precision–Recall curve |
| `predicted_risk_distribution.png` | Predicted probability by true AEV class |
| `shap_bar_importance.png` | Mean |SHAP| bar chart (top 15) |
| `shap_beeswarm.png` | SHAP beeswarm plot |
| `route3_vae_shap_bar_importance.png` (`outputs/shap/figures/`) | Route 3 prediction-head mean |SHAP| bar chart |
| `route3_vae_shap_beeswarm.png` (`outputs/shap/figures/`) | Route 3 prediction-head SHAP beeswarm |
| `route3_vae_shap_dependence_top_features.png` (`outputs/shap/figures/`) | Route 3 SHAP dependence, top latent dims |
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

# 4. Household stream only (Stages 5–9: 3 COR routes + comparison)
#    (requires FES pipeline to have run first)
python main.py --stage household

# 5. Fast development mode (fewer LSTM/TFT epochs)
python main.py --fast

# 6. Skip specific models
python main.py --skip-models LSTM TFT

# 7. Specific series only
python main.py --series gas electricity

# 8. Skip Route 3 (VAE, the most expensive household-stream stage)
python main.py --stage household --skip-vae
```

---

## Methodological Limitations

| Limitation | Impact |
|------------|--------|
| FES has no within-UK household variation | FES cannot be a household-level predictor; treated as a macro-level scenario-based signal simulation |
| **FES scenarios are macro-context simulations, not household behavioural simulations** | The 9 FES scenarios should not be interpreted as causal household response simulations — they simulate alternative macro-level anticipatory stress signals only |
| **FES is not used as a household-level predictor** | The integration between FES and HighAEV is scenario-conditioned and interpretive (`outputs/fes_highaev_interpretation/`), not causal |
| **HighAEV prevalence does not change across FES scenarios unless route-specific labels are compared** | FES changes the interpretation of vulnerability, not the household classification — every row of `scenario_highaev_interpretation_matrix.csv` carries the same HighAEV prevalence per route |
| **PCA/EFA/linear AE are empirical-recovery robustness checks only** | They support or qualify Route 1's construct scores; they are not independent estimation routes and are not separate ML model variants |
| **Scenario-conditioned interpretation is not scenario-conditioned prediction** | The study can discuss which HighAEV households would be policy-relevant under alternative macro stress contexts, but it does not estimate how households would respond if FES changed |
| H12/H15 mislabelled in prior version | Construct redesign corrects this; old results are not comparable |
| C-block heating variables largely absent from UK sub-sample | Thermal energy poverty index deferred to optional robustness extension |
| ENABLE survey is cross-sectional | Path estimates test directional association, not temporal causality |
| SHAP explains model predictions | SHAP values are not causal effect estimates |
| Carbon percentage growth unstable near zero prices | Carbon uses log-return instead of percentage growth |
| Single country UK sub-sample | Results may not generalise across ENABLE's 11 countries |
| **Construct AVE < 0.05 for all four constructs** | All fail the AVE ≥ 0.50 convergent-validity threshold; items load weakly on their intended factor (max loading 0.38).  Interpret construct scores as formative composites, not reflective scales |
| **SEM path estimates to AEV are deterministic** | Because AEV_score = mean(FCP, BLI, TCR, 1−AEMC) by definition, regressing AEV on these four scores produces R² = 1.0 and astronomically large t-statistics (numerical artefact of near-perfect collinearity).  Only the a-path (FCP → AEMC) and mediation CIs are meaningfully estimated |
| **FCP→AEMC a-path direction inconsistent with COR** | Estimated coefficient is positive (higher financial pressure associates with higher adaptive capacity) contrary to COR depletion theory.  The association is not significant (p = 0.14), so no causal claim should be made |
| **Carbon forecast direction reversal (LSTM core)** | LSTM-core forecasts consistently negative carbon growth (−35 to −8 z-unit) while 2017 actuals swung from −6 to +55; prediction interval coverage = 0%.  Carbon SMAPE ≈ 88%. Use TFT-macro (Theil U = 0.077) for the primary carbon FES component |
| **Bare H5/H6/H13 columns absent; reconstructed from sub-items** | The single-column H5/H6/H13 fields don't exist in the UK sub-sample, but the underlying insulation/heating-fuel/smart-meter data does, as ENABLE's per-option sub-items (H5A1-3, H6A3, H13A/H13C). `enable_preprocessing.build_efficiency_controls()` now derives `has_insulation`/`heating_gas_share`/`has_smart_meter` from these; `heating_gas_share` is CatBoost's #1-2 most important Controls_Only feature (~21%, on par with S8) |
| **`heating_gas_share` has a handful of corrupted values** | `load_enable_uk()`'s global `MISSING_CODES` replacement ([9, 98, 99, 999, 9999, 99999] → NaN) also zeroes out genuine 98%/99% answers on this item, since 98/99 double as survey-wide "don't know" sentinels elsewhere. Affects only a few rows out of 1,015 (negligible for reported metrics), but is not literally exact |
| **Route1_Composite / AllRoutes_Hybrid: structural circularity** | High AUC is driven by item-level or score-level overlap with the AEV target, not by genuine out-of-sample predictive power.  Controls_Only is the only fully generalizable predictor |
| **Route 2 second-order structural model can be inadmissible (Heywood case)** | Weakly-correlated first-order factors can drive the AEV factor variance to near-zero or negative; `fit_structural_model()` detects this and flags the result rather than reporting an unstable estimate — check `structural_fit_indices_2A.csv` before interpreting Option 2A |
| **Route 3 prediction head trains directly against `high_aev`** | Unlike Route 1/2's circularity (scores summed/loaded into the target), Route 3's BCE loss term optimizes the latent space against the label itself — its high ROC-AUC reflects supervised signal baked into training, not only representation quality |
| **Route 2 / Route 3 depend on Route 1's outputs** | Route 2's Option 2B check and Route 3's COR-alignment loss + prediction label both use Route 1's composite scores / `high_aev` as inputs — the three routes are not fully independent estimation strategies |

---

## Dependencies

| Library | Purpose |
|---------|---------|
| `pandas`, `numpy` | Data wrangling |
| `scipy`, `statsmodels` | Statistical tests, OLS path analysis |
| `pmdarima` | SARIMA / SARIMAX auto-selection |
| `scikit-learn` | Scaling, train/test split, PCA, metrics |
| `factor_analyzer` | EFA (optional; falls back to PCA) |
| `semopy` | Route 2: CFA measurement model + second-order structural model |
| `prophet` | Prophet forecasting |
| `tensorflow` | LSTM (MC Dropout), Route 1 linear autoencoder, Route 3 VAE |
| `torch`, `pytorch-forecasting` | TFT / Attention-LSTM fallback |
| `catboost` | Gradient boosting for HighAEV prediction |
| `shap` | SHAP explainability (CatBoost + Route 3 prediction-head KernelExplainer) |
| `matplotlib`, `seaborn` | Static figures |
| `plotly` | Interactive HTML forecast timelines |
| `openpyxl` | Reading ENABLE.EU .xlsx file |
| `jupyter` | Monitoring notebook |

---

## Citation / Project Context

**Title:** Forecasting Anticipatory Energy–Carbon Stress and High Adaptive Energy Vulnerability in the UK: A Conservation of Resources and Explainable Machine Learning Framework

**Alternative:** Anticipatory Policy Intelligence for Energy Poverty: Forecasted Energy–Carbon Stress, Household Adaptive Capacity, and Just Transition Vulnerability

**Framework components:**

1. **FES scenario-based signal simulation (9 scenarios)** — equal-weight, volatility-weighted (VW-FES) and Bayesian (Kalman) constructions, each in core/macro mode (6 forecasted scenarios), plus 3 realised benchmark scenarios (A/B/C) — a macro-context robustness layer, not a household-level predictor or behavioural simulation
2. **TS-SHAP** — feature attribution for the best model in each of 6 forecast streams
3. **Route 1 — COR Composite** — codebook-corrected COR constructs (formative scores), psychometric checks, OLS path analysis + bootstrap mediation, PCA/EFA/linear-AE empirical-recovery robustness (checks only, not a fourth route)
4. **Route 2 — COR-Informed SEM** — CFA measurement model + second-order structural model for AEV, semopy MLW estimation
5. **Route 3 — COR-Informed VAE** — theory-informed variational autoencoder with a COR-alignment penalty and a HighAEV prediction head
6. **Cross-route comparison** — per-construct and outcome agreement across all three estimation routes
7. **CatBoost + SHAP** — explainable risk prediction for High Adaptive Energy Vulnerability, one model per route (Controls_Only, Route1_Composite, Route2_SEM, Route3_VAE, AllRoutes_Hybrid only — FES is never a feature)
8. **Scenario-conditioned interpretation** — `outputs/fes_highaev_interpretation/` links the 9 FES scenarios to the already-estimated HighAEV prevalence descriptively, never as a prediction

The z-score standardisation across all 9 FES scenarios uses the same 2005–2016
training distribution, ensuring differences reflect genuine methodological variation
rather than scaling artefacts.

In summary: the macro stream constructs 9 FES scenarios that simulate
alternative macro stress signals, not household behaviour; the household
stream estimates HighAEV through three independent routes; ML predicts
HighAEV from household-level controls and route-specific representations,
never from FES; SHAP explains model predictions, not causal effects;
PCA/EFA/linear AE are retained only as Route 1 empirical-recovery
robustness; and FES and HighAEV are linked only through scenario-conditioned
interpretation and policy relevance.

**The household model estimates who is vulnerable. The FES scenario layer
explains the macro stress context under which that vulnerability is
interpreted.**

---

*Anticipatory Energy–Carbon Stress Pipeline · UK 2005–2017*
