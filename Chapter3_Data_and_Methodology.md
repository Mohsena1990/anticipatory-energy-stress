# Chapter 3: Data and Methodology

---

## 3.1 Overall Research Design

This study pursues a mixed-methods analytical pipeline that integrates macro-level energy price forecasting with household-level psychometric modelling. The central empirical claim is that anticipatory energy stress — the stress households experience in expectation of future energy cost increases — can be operationalised both as a macro index derived from forecasted energy and carbon prices and as a composite household vulnerability score grounded in Conservation of Resources (COR) theory. The methodology is designed so that each analytical stream informs the next and so that the same theoretical framework unifies quantitative results produced by very different methods.

The pipeline proceeds across twelve analytically distinct stages. First, three macro energy series — natural gas price growth, electricity price growth, and carbon allowance log-returns — are assembled from UK administrative and market sources covering January 2005 to December 2017. A set of exogenous macro controls is assembled alongside these core series. Second, four core forecasting models using only target-series history generate twelve-month forecasts for 2017 along with 95% prediction intervals. Third, models are evaluated on a 2016 hold-out year and the best-performing model per series is selected by rank aggregation. Fourth, the best-model forecasts are combined into the primary Forecasted Energy-Carbon Stress (FES_core) index. Fifth, the exogenous macro controls are introduced as regressors in macro-augmented variants of the same models; the resulting FES_macro index constitutes **Robustness Check 1**, testing sensitivity to the inclusion of market and economic covariates. Sixth, the three actual 2017 energy-carbon values are used to construct FES_actual benchmarks; these constitute **Robustness Check 2**, providing a ground-truth comparison against which both FES_core and FES_macro are evaluated. Seventh, a model-specific time-series attribution procedure (TS-SHAP) identifies which historical and exogenous features drove each selected forecast. Eighth, the ENABLE.EU household survey is loaded, restricted to the UK sub-sample, and cleaned before construct items are directionally recoded and normalised. Ninth, four revised COR-based composite scores — Financial–Energy Cost Pressure (FCP), Adaptive Energy-Management Capacity (AEMC), Energy Behavioural Lock-in (BLI), and Transition-Cost Resistance (TCR) — are constructed, validated psychometrically, and submitted to OLS path analysis (the SEM stream). Tenth, three unsupervised dimensionality-reduction methods — PCA, EFA, and a linear autoencoder — are applied to the same item pool without any theoretical guidance; their data-driven latent dimensions are then compared against the COR composite scores, constituting **Robustness Check 3**: a test of whether the theory-specified construct structure is recoverable from the raw empirical covariance of the items. Eleventh, CatBoost classification with five model variants — a controls-only baseline plus one variant per COR estimation route (Route1_Composite, Route2_SEM, Route3_VAE) and an AllRoutes_Hybrid combination — compares the predictive value of the theory-driven route representations of household vulnerability; PCA/EFA/the linear autoencoder from the Ninth stage remain Route 1 empirical-recovery robustness checks, not separate model variants. Twelfth, SHAP explainability attributes predictions from the meaningful generalizable classifier to specific household characteristics.

The three robustness checks are sequenced to test increasing depth of methodological challenge: Robustness 1 tests whether macro information changes the macro stress signal; Robustness 2 tests whether the forecasted stress signal tracks realised stress; Robustness 3 tests whether the theoretical household constructs are recoverable from the data without imposing any COR structure. Together, they provide convergent evidence that the primary results are not artefacts of any single modelling assumption.

---

## 3.2 Macro Data

### 3.2.1 Gas Data

The primary domestic gas series is the Retail Prices Index (RPI) gas sub-component published monthly by the UK Office for National Statistics (ONS). The ONS series reports the percentage change in domestic gas prices over the preceding twelve months, expressed as a year-over-year (YoY) growth rate. Because the series is already expressed as YoY percentage growth, no further transformation is required: the values are used directly as `gas_growth`. The series spans January 2005 to December 2017, yielding 156 monthly observations that capture the 2008 energy price spike, the elevated 2010–2013 gas price environment, the sustained deflation following the global oil price collapse of mid-2014, and the low-growth period of 2015–2017. An extended internal loading window beginning in January 2003 is used in the parser to accommodate 12-month lags needed for other macro series, but the gas growth values themselves require no back-fill from January 2005.

### 3.2.2 Electricity Data

The electricity series originates from the ONS Consumer Prices Index (CPI), specifically sub-index 04.5.1 "Electricity" with 2015 as the base year (2015 = 100). Unlike the gas series, this is an index level, not a pre-computed growth rate. To make it conceptually consistent with the gas and carbon series — all of which represent YoY growth — it is transformed:

$$\text{ElectricityGrowth}_t = \frac{\text{Index}_t - \text{Index}_{t-12}}{\text{Index}_{t-12}} \times 100$$

The raw index is loaded from January 2003 so that 12-month lagged values are available for computing the full growth series from January 2005. The index level itself (`electricity_index`) is retained alongside the growth series for models that forecast the administered index level first and derive growth after forecasting. The near-zero electricity growth rates in 2016–2017 introduce particular metric instability in percentage-based evaluation, a point addressed explicitly in the model selection procedure.

### 3.2.3 Carbon Data

The carbon series is derived from European Union Allowance (EUA) futures closing prices sourced from ICE/Investing.com, expressed in EUR per tonne. EUA Phase I trading commenced in April 2005, so the first usable year-over-year observation becomes available from April 2006; rows prior to the first valid observation are dropped rather than back-filled.

A deliberate decision is made to use the log-return transformation rather than simple percentage growth:

$$\text{CarbonLogReturn}_t = \log\left(\frac{\text{Price}_t}{\text{Price}_{t-12}}\right) \times 100$$

The motivation is statistical. EUA prices in Phase I (2005–2007) exhibited extreme collapse near zero as the market discovered over-allocation; similar near-zero episodes occurred after Phase II adjustments. Simple percentage growth $(P_t - P_{t-12})/P_{t-12}$ becomes numerically explosive when the denominator approaches zero, whereas the log-return is bounded and symmetric around zero, making it better-suited to modelling and z-scoring. The log-return is multiplied by 100 to place it on a percentage scale comparable to the other two growth series. The `carbon_growth` column contains log-returns throughout and this should be borne in mind when interpreting raw coefficient magnitudes.

### 3.2.4 Macro Controls

Beyond the three core energy-carbon series, a second dataset assembles exogenous macro controls used as regressors in the macro-augmented forecasting models and as the basis for Robustness Check 1. These controls are not FES components; they serve as explanatory context for the three energy-carbon targets.

**CPIH housing-energy inflation** (`inflation_growth`) is derived from the ONS CPIH sub-index for housing energy (series 08_188), transformed to a monthly YoY percentage change, capturing the broader consumer price environment affecting energy-adjacent household expenditure.

**Monthly GVA growth** (`gdp_growth`) comes from the ONS monthly GVA indicator, using the period-on-period percentage change in chained-volume-measure seasonally adjusted GVA — a proxy for macroeconomic activity that co-moves with energy demand.

**Weather volatility** (`weather_volatility`) is the 12-month rolling standard deviation of monthly UK temperature anomalies. High rolling standard deviation indicates greater-than-usual seasonal temperature instability, associated with elevated and uncertain energy demand.

**NBP natural gas futures** (`gas_futures_price`, `gas_futures_log_return`, `gas_futures_yoy_growth`) from UK National Balancing Point quarterly futures data embed forward-looking market expectations about gas prices, capturing anticipatory information not present in the historical spot series.

**Electricity demand and supply** variables are constructed from Elexon half-hourly settlement data, aggregated to monthly means and peaks. These include mean and peak national demand, England and Wales demand, embedded wind and solar generation, embedded wind and solar capacity, pump storage pumping, and net interconnector flows, with YoY growth rates derived for all volume series. These capture the supply-demand balance that is a direct driver of wholesale electricity prices.

**GBP/EUR exchange rate** (`gbp_eur_rate`, `gbp_eur_yoy_change`, `gbp_eur_mom_change`) from the ONS XUMAERS monthly average sterling series. EUA carbon allowances are priced in EUR, so sterling depreciation directly raises compliance costs for UK generators and inflates the cost of electricity imported via EU interconnectors. Both YoY and month-on-month changes are included, plus a 12-month lagged YoY change to capture regulatory transmission lags in UK retail electricity pricing — the post-Brexit GBP depreciation of H2 2016 is a concrete motivating episode.

**Lagged variants** (one-month lags for the full variable list, and a 12-month lag specifically for GBP/EUR YoY change) are generated automatically. Two structural dummies are also added: `post_2016_electricity_regime` (binary, from January 2016 onward) and `winter_dummy` (November to March). All lagged columns whose first row would be missing are filled with 0.0 to keep the regressor array rectangular without introducing look-ahead.

---

## 3.3 Forecasting Design

### 3.3.1 Forecast Horizon and Data Splits

The training period covers January 2005 to December 2015, providing 132 monthly observations. The validation (hold-out) period covers January to December 2016 (12 observations) against which each fitted model is evaluated before any 2017 values are seen. The forecast period is January to December 2017 — a 12-step-ahead horizon that constitutes the operationalisation year for the FES index.

The year 2017 is chosen as the forecast horizon because the ENABLE.EU survey was fielded in 2016–2017, making 2017 the most relevant period for interpreting household-level HighAEV under the corresponding macro energy-carbon stress context (a scenario-conditioned interpretation, not a causal or predictive link — see Section 3.4). The validation year 2016 provides a fully out-of-sample test that precedes the forecast horizon, preventing look-ahead bias in model selection. Z-score standardisation for all FES scenarios uses the combined training and validation period (January 2005 to December 2016) as the reference window, ensuring that FES_core, FES_macro, and FES_actual are anchored to the same distributional baseline.

### 3.3.2 Core Models (Primary Stream)

Four model classes are fitted to each of the three energy-carbon series using only the target series history — no exogenous regressors. These constitute the primary forecasting stream.

**SARIMA** (Seasonal Autoregressive Integrated Moving Average) captures lagged autoregressive dependencies, moving average error correction, and annual cycles driven by heating-season demand. Prediction intervals are produced analytically from the fitted residual variance.

**Prophet** represents the series as a sum of a piecewise-linear trend, a Fourier-basis annual seasonality component, and an irregular remainder. It is robust to structural breaks via changepoint detection and quantifies uncertainty through Monte Carlo simulation of trend uncertainty.

**LSTM** (Long Short-Term Memory) uses a 12-month lookback window, 100 training epochs, batch size 16, and learning rate 0.001. Two LSTM layers with dropout are used. Prediction intervals are approximated via Monte Carlo dropout: 200 forward passes with dropout active generate a distribution from which the 2.5th and 97.5th percentiles define the 95% interval.

**TFT** (Temporal Fusion Transformer) combines multi-head self-attention with gating mechanisms. In the core variant, the TFT receives only lags of the target series. TFT natively generates quantile forecasts; the 5th and 95th quantile predictions serve as the 95% interval bounds.

### 3.3.3 Macro-Augmented Models — Robustness Check 1

The first robustness check tests whether incorporating exogenous macroeconomic information systematically changes the FES signal relative to the core (target-history-only) estimates. Four macro-augmented variants are fitted for each series, using the same model architectures as the core stream but with the exogenous controls from Section 3.2.4 as additional inputs.

**SARIMAX** extends SARIMA by admitting exogenous variables as additive regressors estimated jointly with the ARIMA orders, providing a direct interpretable link between macro controls and the forecast.

**Prophet with regressors** incorporates exogenous series as additive regression terms within the Prophet decomposition, attributing residual variation to specific macro drivers after trend and seasonality are removed.

**Multivariate LSTM** uses the full macro control set as additional input channels at each time step of the lookback window, learning joint temporal dynamics across energy price growth rates and their macroeconomic context.

**TFT with covariates** includes exogenous controls as known future inputs (structural dummies, projected values) and observed inputs (contemporaneous values assumed to persist forward). TFT's variable selection network assigns learned importance weights to each covariate, making it well-suited to the high-dimensional macro control set.

The FES index constructed from these macro-augmented model forecasts (**FES_macro**) is compared directly against the core-only FES (**FES_core**) throughout. If the two variants produce similar stress signals and similar tracking of actual 2017 prices, it suggests that the core price-history information is sufficient and the macro controls add limited incremental signal in this forecast year. If they diverge materially, the macro covariates are changing the stress characterisation in ways that carry substantive implications for how market-forward information modifies anticipatory household stress.

### 3.3.4 Model Evaluation

All eight models (four core, four macro-augmented) are evaluated on the 2016 validation year using a battery covering both point-forecast accuracy and probabilistic interval quality:

- **MAE** (Mean Absolute Error): average absolute deviation.
- **RMSE** (Root Mean Squared Error): penalises large errors more heavily.
- **SMAPE** (Symmetric Mean Absolute Percentage Error): bounded and less sensitive to near-zero denominators than MAPE.
- **MASE** (Mean Absolute Scaled Error): MAE scaled by the seasonal naïve baseline MAE (period 12), making it scale-free across series.
- **Quantile loss**: pinball loss at the 5th and 95th quantiles.
- **Winkler score**: interval scoring rule that rewards narrow intervals that still cover the actual value.
- **MSIS** (Mean Scaled Interval Score): interval analogue of MASE.
- **Prediction interval coverage**: proportion of 2016 actuals inside the 95% prediction interval.

**MAPE is explicitly excluded from model selection ranking.** Near-zero electricity and gas growth rates produce MAPE values in the thousands of percentage points, rendering it uninformative for ranking. MAPE is retained in output tables for reporting purposes only.

Model selection uses rank aggregation: within each (series, mode) group, models are ranked from best to worst on each metric, and a weighted sum of ranks is computed. Weights: MAE and RMSE each 0.25; SMAPE 0.15; MASE, quantile loss, Winkler score, MSIS, and coverage each receive smaller weights. The model with the lowest composite score is selected as the best for that series–mode combination.

---

## 3.4 FES Construction

### 3.4.1 Equal-Weight FES (Primary Index)

The Forecasted Energy-Carbon Stress index for each month $t$ in 2017 is the sum of four z-scored components:

$$\text{FES}_t = z(\hat{G}_t) + z(\hat{E}_t) + z(\hat{C}_t) + z(\hat{U}_t)$$

where $\hat{G}_t$, $\hat{E}_t$, $\hat{C}_t$ are the best-model point forecasts for gas growth, electricity growth, and carbon log-return, and $\hat{U}_t$ is a measure of aggregate forecast uncertainty. Z-scores use the training-period statistics (2005–2016 mean and standard deviation) for the energy-carbon series. Positive FES values indicate above-historical-average stress; negative values indicate below-average stress.

The uncertainty component $\hat{U}_t$ is the mean of the z-scored prediction-interval half-widths across the three series, where each half-width is z-scored relative to the rolling 12-month standard deviation of the training series. This makes uncertainty a fourth substantive dimension of the index rather than merely a confidence qualifier.

### 3.4.2 FES_core and FES_macro

**FES_core** is built from the best-performing core (target-history-only) model for each series; it represents what a forecaster could derive from price history alone. **FES_macro** uses the best macro-augmented model for each series; it incorporates market intelligence such as futures curves, demand patterns, and exchange rate dynamics. The comparison of FES_core against FES_macro constitutes the primary output of Robustness Check 1 in the macro stream: if the two variants co-move closely over 2017, the additional exogenous information does not materially change the stress characterisation; if they diverge, the macro context is driving a different reading of anticipatory household stress.

### 3.4.3 Volatility-Weighted and Bayesian FES

Two further weighting schemes are implemented as additional robustness variants within the FES construction.

**Inverse-volatility (IV) weighting** tests sensitivity to the potentially dominant influence of the high-volatility carbon series:

$$w_j = \frac{1/\hat{\sigma}_j}{\sum_k 1/\hat{\sigma}_k}, \quad j \in \{\text{gas, electricity, carbon}\}$$

where $\hat{\sigma}_j$ is the training-period standard deviation of series $j$. Under this scheme, the carbon series receives a lower weight, preventing extreme EUA price swings from overwhelming the FES signal. The IV-weighted variants are labelled **FES_vw_core** and **FES_vw_macro**.

**Bayesian / Kalman FES** treats the monthly stress value as a latent scalar state variable observed through three noisy z-scored signals under a random-walk state-space model:

$$\text{FES}_t = \text{FES}_{t-1} + w_t, \quad w_t \sim \mathcal{N}(0, \sigma_Q^2)$$
$$z_{j,t} = \text{FES}_t + v_{j,t}, \quad v_{j,t} \sim \mathcal{N}(0, \sigma_R^2)$$

Process noise $\sigma_Q$ and observation noise $\sigma_R$ are estimated from the training-period z-score matrix. The filter produces posterior monthly FES estimates with 95% credible intervals. These Bayesian variants (**FES_bayes_core** and **FES_bayes_macro**) provide a probabilistically-grounded alternative to the sum-of-z-scores construction.

### 3.4.4 Actual FES Benchmarks — Robustness Check 2

Robustness Check 2 tests whether the forecasted FES indices track realised 2017 energy stress. Three actual FES benchmarks are constructed from observed 2017 values. The benchmark design must replace the forecast prediction-interval half-width — which is undefined for actual values — with a measure of realised uncertainty:

- **Option A (rolling volatility):** 3-month rolling standard deviation of the mean across the three z-scored actual values, capturing short-run temporal instability.
- **Option B (cross-component dispersion):** cross-sectional standard deviation of the three z-scored actual values at each month $t$, capturing how much the three series diverge within a given month.
- **Option C (absolute shock):** absolute deviation of the monthly mean z-score from the training-period mean, normalised by the training-period standard deviation.

These are labelled `fes_actual_A`, `fes_actual_B`, and `fes_actual_C`. Option B is the primary benchmark because it is computed from the same z-score matrix as the forecasted variants and is most directly comparable to the equal-weight FES structure.

Each of the six forecasted FES variants is compared against each actual benchmark using: MAE, RMSE, mean bias, maximum absolute deviation, Pearson r, Spearman r, R-squared, and Theil's U (RMSE relative to a naïve persistence forecast). High Pearson r with low RMSE against the primary actual benchmark (Option B) constitutes passing Robustness Check 2.

---

## 3.5 Time-Series Attribution (TS-SHAP)

To identify which historical and exogenous features drive the selected forecasts, a model-class-specific attribution procedure is applied to the best-performing model for each (series, mode) combination. Four approaches are used:

**SARIMA/SARIMAX:** the product of the estimated exogenous coefficient and the deviation of each regressor from its training-period mean at each time step — an additive decomposition of the forecast into per-variable macro contributions.

**Prophet:** native component decomposition extracting trend, seasonality, and per-regressor contributions as additive terms at each time step.

**LSTM:** gradient × input saliency — the gradient of the output with respect to each input time step, multiplied by the actual input value, computed across the full 12-month lookback window to produce a (time steps × features) attribution matrix.

**TFT:** attention weight matrix (which historical time steps the model attends to most strongly) combined with gradient × input attribution (feature-level). TFT's multi-head attention is applied after variable selection, so high attention weights at specific time steps carry interpretable meaning.

The purpose is to identify which lagged energy-carbon histories and which macro covariates (NBP futures, exchange rates, electricity demand, weather volatility) most strongly drove each selected forecast, bridging the gap between black-box model outputs and interpretable economic attribution.

---

## 3.6 ENABLE Household Data

### 3.6.1 UK Sub-Sample

Household-level data come from the ENABLE.EU multi-country survey of residential energy behaviour, conducted across eleven European countries in 2016–2017 using a standardised questionnaire covering energy attitudes, household financial circumstances, energy-saving behaviours, appliance adoption, and dwelling characteristics. This study uses exclusively the United Kingdom sub-sample, identified by country code 11, comprising approximately 1,015 households. The UK sub-sample is substantively aligned with the macro pipeline: the energy-carbon series, NBP futures, Ofgem-regulated electricity market, and GBP/EUR dynamics are all UK-specific. Using a different national sub-sample would introduce an incoherence between the macro-stress index and the household vulnerability assessment.

### 3.6.2 Missing Value Codes

The ENABLE.EU dataset uses numeric sentinel values (9, 98, 99, 999, 9999, and 99999) to indicate non-response, refusal, or not-applicable responses. These are replaced with `NaN` at the point of loading. Items with high missingness rates or zero post-cleaning variance are documented in item diagnostics tables and excluded from composite scoring rather than imputed.

### 3.6.3 Codebook Corrections

**H12 items** were previously described in earlier versions of this framework as thermal discomfort indicators. The codebook makes clear that H12A captures the proportion of incandescent lightbulbs and H12B the proportion of energy-efficient lightbulbs. These measure lightbulb adoption — a transition-behaviour proxy — and are excluded from the core construct scoring.

**H15 items** were similarly mislabelled as thermal discomfort in earlier iterations. H15A through H15G are explicitly environmental and energy-transition attitude items (social conditionality, scepticism about impacts, temporal displacement, lifestyle change willingness, cost resistance, techno-optimism, and environmental-economic beliefs). They are correctly assigned to the Transition-Cost Resistance (TCR) construct in the revised framework.

**C-block variables** (C1A–C7F), which contain thermal comfort and dwelling refurbishment items and would ideally anchor a thermal discomfort construct, are largely absent in the UK sub-sample, most likely due to interviewer skip-logic or not-applicable routing for certain housing types. The C-block is documented but not incorporated. The thermal discomfort dimension from earlier framework versions is replaced in full by Transition-Cost Resistance.

---

## 3.7 Revised COR Constructs

Conservation of Resources theory (Hobfoll, 1989) posits that individuals strive to acquire, retain, and protect valued resources, and that resource threat or loss triggers psychological and behavioural stress. In the energy context, financial pressure on energy spending constitutes a resource threat; adaptive energy-management behaviours represent resource acquisition and protection; behavioural lock-in prevents efficient resource deployment; and resistance to transition costs reflects anticipated threat from the energy transition. Four constructs operationalise this structure.

### 3.7.1 Financial–Energy Cost Pressure (FCP)

FCP (replacing "Perceived Energy Insecurity") measures the household's subjective experience of financial difficulty and perceived energy cost burden. Core items: **S8** (difficulty living on present income), **E2A** (perceived cost of running a TV for one hour), **E2B** (perceived cost of running a washing machine). Optional items (S7, E1, E3A–E3C) are included when available and non-degenerate. Higher scores indicate greater resource threat.

### 3.7.2 Adaptive Energy-Management Capacity (AEMC)

AEMC (replacing "Resource Preservation") captures the household's active capacity to organise and execute energy-saving actions through reminders and habitual routines. Core items: reminder behaviours (E5A2–E5A4) and routine behaviours (E6A2–E6A4, E6A6–E6A8). Two items are reverse-coded before scoring: **E5A1** ("I do not use reminders") and **E6A1** ("I do not have routines") so that endorsing non-use contributes zero to adaptive capacity. Optional items include H9 (heating control method, recoded on a 0–3 scale where automated controls score highest) and additional reminder and tariff-switching items. AEMC enters the AEV composite as $(1 - \text{AEMC})$ so that weak adaptive capacity contributes positively to vulnerability.

### 3.7.3 Energy Behavioural Lock-in (BLI)

BLI (a new construct) operationalises the degree to which energy-consumption behaviours have become habitual, automatic, and resistant to deliberate change. Items from the E7 battery: **E7A** (behaviours through repetition), **E7B** (behaviours while thinking about something else — automaticity), **E7C** (behaviours without full awareness), **E7D** (perceived difficulty changing behaviours), **E7E** (behaviours maintained because alternatives require too much effort). Higher BLI indicates stronger consumption inertia.

### 3.7.4 Transition-Cost Resistance (TCR)

TCR (replacing "Thermal Discomfort" following the codebook corrections in Section 3.6.3) captures reluctance to bear personal costs of the environmental transition and scepticism about environmental urgency. Core items: **H15A** (social conditionality), **H15B** (scepticism about environmental impacts), **H15C** (temporal displacement), **H15E** (cost resistance to government policy), **H15F** (techno-optimism fatalism). **H15D** is reverse-coded (high willingness to make lifestyle changes = low resistance). H15G (positive environmental-economic belief) is excluded from the core composite to avoid attenuating the resistance signal.

### 3.7.5 High Adaptive Energy Vulnerability (HighAEV)

The four construct scores are combined into a composite AEV score as a row-wise mean across the four vulnerability-direction components (all component scores normalised to [0, 1]):

$$\text{AEV}_i = \text{mean}\bigl(\text{FCP}_i,\; \text{BLI}_i,\; \text{TCR}_i,\; (1 - \text{AEMC}_i)\bigr)$$

Row-wise mean (rather than sum) gracefully accommodates partial item missingness: when a component score is unavailable, the composite is computed from the available components rather than set to missing, keeping AEV on the [0, 1] interval. The binary classification target **HighAEV** is defined at the 75th percentile of the AEV distribution within the UK sample:

$$\text{HighAEV}_i = \mathbf{1}\bigl[\text{AEV}_i \geq Q_{0.75}(\text{AEV})\bigr]$$

Approximately 25% of the UK sample is classified as high-vulnerability. HighAEV is a relative vulnerability indicator — identifying the top quartile of the composite distribution within this sample — rather than an absolute poverty threshold.

---

## 3.8 Construct Validation

The four constructs are evaluated on normalised item scores within the UK sub-sample using a standard psychometric battery.

**Internal consistency:** Cronbach's alpha ($\alpha$, minimum acceptable threshold 0.60) and McDonald's omega ($\omega$, estimated from first-factor EFA loadings and relaxing the tau-equivalence assumption of alpha). Both are computed from the normalised item matrix.

**Convergent validity:** Composite Reliability (CR, Jöreskog's $\rho_c$, acceptable above 0.70) and Average Variance Extracted (AVE, acceptable above 0.50 per Fornell and Larcker, 1981). Both CR and AVE are computed from squared standardised first-factor loadings.

**Discriminant validity:** HTMT ratio (Henseler et al., 2015; values below 0.85 support discriminant validity) and the Fornell-Larcker criterion (AVE of each construct must exceed the squared Pearson correlation with every other construct).

**Factor structure:** Single-factor EFA for each construct using maximum likelihood estimation with varimax rotation (falling back to a PCA-based loading approximation when the `factor_analyzer` package is unavailable). Factor loadings below 0.40 are flagged. The proportion of variance explained by the first factor provides an additional convergent validity indicator.

**A critical limitation must be stated explicitly.** Across all four constructs, AVE falls below the conventional 0.50 threshold. This reflects a genuine mismatch between the item pools and the requirements of a reflective measurement model: the ENABLE.EU items were designed for cross-European descriptive research rather than reflective COR construct operationalisation, and items within each scale do not share a strong enough common factor to meet AVE standards. Rather than abandoning the constructs or narrowing item selection at the cost of content validity, this study adopts the position that **the four constructs are most appropriately treated as formative or composite vulnerability scores rather than reflective latent scales**. Under a formative interpretation, items are understood as distinct causes of the latent construct rather than parallel reflections of it, and AVE is not the appropriate validity criterion. Alpha, omega, and factor loadings are retained as descriptive coherence indicators, but substantive conclusions are drawn at the composite score level. This framing is maintained consistently through all downstream analyses.

---

## 3.9 COR Path Analysis and Mediation (SEM Stream)

### 3.9.1 Why OLS Rather Than Full SEM

The COR-specified directional pathways are estimated using Ordinary Least Squares (OLS) regression rather than Structural Equation Modelling in its conventional sense. Full SEM requires a reflective measurement model with adequate AVE to separate measurement error from structural relationships. Since AVE is below threshold for all four constructs (Section 3.8), estimating a simultaneous measurement and structural model would be methodologically inappropriate: the measurement model component would be fitting a misspecified factor structure. OLS path analysis bypasses this by treating each composite score as a directly observed variable, estimating only the structural relationships among the already-constructed scores. Results are interpreted as COR-consistent directional associations — evidence that the observed data pattern is consistent with the theorised resource dynamics — rather than as causal structural effects.

### 3.9.2 Implementation

All path regressions are implemented via `statsmodels` OLS with heteroskedasticity-consistent standard errors. Each regression uses listwise-complete rows across the outcome and all specified predictors. For each model, the reported statistics are: regression coefficient $\beta$, standard error, t-statistic, p-value, and model-level $R^2$. Significance is assessed at the conventional $\alpha = 0.05$ level. The FES contextual values — the annual FES score for 2017 — are identical for all UK households and carry zero within-sample variance; they are excluded from all regressions and reported as contextual background only.

### 3.9.3 Estimated Pathways

Five directional paths constitute the COR path model:

- **a-path**: FCP → AEMC. COR predicts a negative coefficient: financial resource threat depletes the capacity for organised adaptive action.
- **b-path**: AEMC → AEV composite. Higher adaptive capacity reduces overall vulnerability (negative coefficient).
- **c'-path**: FCP → AEV composite (direct, controlling for AEMC). Higher financial pressure directly increases AEV (positive coefficient).
- **d-path**: BLI → AEV composite. Stronger behavioural lock-in increases AEV (positive coefficient).
- **e-path**: TCR → AEV composite. Greater transition-cost resistance increases AEV (positive coefficient).

### 3.9.4 Bootstrap Mediation

Bootstrap mediation for the theoretically primary FCP → AEMC → AEV pathway is estimated using the percentile method with 2,000 bootstrap samples drawn with replacement from the complete-case UK sub-sample. Each bootstrap sample yields an estimate of the indirect effect $a \times b$ (a-path coefficient × b-path coefficient), computed via two sequential OLS regressions within that sample. The 2.5th and 97.5th percentiles of the 2,000 bootstrap indirect-effect estimates define the 95% confidence interval. A confidence interval that excludes zero indicates statistically significant mediation.

### 3.9.5 Structural Circularity Caveat

The interpretation of paths to the AEV composite requires explicit caution. AEV is an arithmetic function of FCP, BLI, TCR, and $(1 - \text{AEMC})$. Regressing AEV on any subset of these component scores is partly a regression of an outcome on its own inputs. Specifically, a model regressing AEV simultaneously on all four component scores will produce R-squared approaching 1.0 by definition — not because the model has discovered an empirical relationship, but because it has recovered the composite formula. The regression coefficients in such a model reflect the algebraic structure of that formula rather than independent empirical associations.

This circularity constrains interpretation without eliminating the value of the path analysis. The most empirically meaningful path is **FCP → AEMC** (a-path), which tests a directional relationship between two distinct composite measures before any composite outcome is involved. Mediation results are interpreted with explicit acknowledgement that the dependent variable (AEV) is mechanically derived from the same items as the mediator (AEMC) and the other predictors (BLI, TCR). This limitation is documented throughout the results chapter.

---

## 3.10 Unsupervised Latent Robustness — Robustness Check 3

### 3.10.1 Purpose and Logic

Robustness Check 3 asks a different question from Robustness Checks 1 and 2. Those checks tested sensitivity of the macro stress signal to modelling choices; Robustness Check 3 tests whether the COR-derived household construct structure is recoverable from the raw empirical covariance of the items, without any theoretical guidance. If the same four vulnerability dimensions that COR theory specifies emerge from data-driven dimensionality reduction applied blindly to the item pool, this constitutes convergent evidence that the construct structure reflects genuine latent covariance in the data — not merely theoretical imposition. If the data-driven dimensions do not align with the COR constructs, it suggests the COR framework is identifying a structure that the data-alone approach cannot reproduce, which would itself be an important diagnostic result.

Three unsupervised methods are applied to the full pool of normalised construct items jointly — that is, across all items from all four constructs simultaneously, with no within-construct grouping imposed.

### 3.10.2 Item Matrix Preparation

Before any unsupervised method is applied, the item matrix is assembled from all available normalised construct item columns (`n_{item}`) — a total of 27 core items spanning FCP (S8, E2A, E2B), AEMC (E5A1–E5A4, E6A1–E6A8), BLI (E7A–E7E), and TCR (H15A–H15F). Listwise deletion of rows with any missing value is applied across this joint item pool, producing a complete-case item matrix. All items are standardised (zero mean, unit variance) before being passed to each unsupervised method.

### 3.10.3 Principal Component Analysis

PCA extracts four orthogonal linear combinations of the standardised items that maximise successive variance components (four components matching the number of COR constructs). Eigenvalues, explained variance ratios, and cumulative explained variance are reported. Component scores for each household are retained for downstream alignment analysis and CatBoost classification.

### 3.10.4 Exploratory Factor Analysis

EFA is performed using maximum likelihood estimation with varimax rotation, extracting four factors (matching the number of COR constructs). When the `factor_analyzer` package is available, it is used directly; otherwise, a PCA-based loading approximation is substituted. Factor loadings, communalities, eigenvalues, and explained variance ratios are reported. Factor scores for each household are computed from the fitted factor solution and retained for alignment and classification.

### 3.10.5 Linear Autoencoder

The linear autoencoder is a two-layer feedforward neural network with the following architecture: the input layer receives the full standardised item vector (dimension = number of available items, typically 27); a single hidden dense layer with four linear neurons forms the **bottleneck** (the learned latent representation); a second dense layer with linear activations reconstructs the original item vector from the bottleneck. Both the encoder and decoder use strictly linear activations, making the autoencoder mathematically equivalent in expressivity to a rank-4 PCA — but trained by gradient descent rather than eigendecomposition. The training objective is mean squared reconstruction error (MSE).

**Training protocol.** The model is compiled with the Adam optimiser at learning rate 0.001. The item matrix is split 80/20 into training and validation subsets. Training runs for up to 300 epochs with batch size 64 and early stopping with patience 30, restoring the best-validation-MSE weights. After training, the encoder sub-model is used to project all households (including those in the held-out validation split) into the four-dimensional bottleneck space. Three MSE values are logged: overall reconstruction MSE, training-split MSE, and validation-split MSE. The gap between validation and training MSE is reported as a diagnostic for overfitting.

**Seed stability assessment.** Because gradient-based training is sensitive to random initialisation, the autoencoder is trained 30 times with different random seeds (seeds 0 to 29). Each run produces a different latent representation, reflecting the rotational non-uniqueness of linear autoencoders. For each seed, the four bottleneck dimensions are aligned to the four COR construct scores via the Hungarian algorithm (see Section 3.10.6) and the aligned absolute Pearson correlations are recorded. The mean and standard deviation of the aligned |r| across 30 seeds are reported per construct; small standard deviation across seeds indicates that the alignment is a stable feature of the data structure rather than an artefact of one particular initialisation.

**Bottleneck size sweep.** To test robustness to the choice of four latent dimensions, the autoencoder is also trained with bottleneck sizes of 3 and 5 neurons (five random seeds each). For each bottleneck size, mean reconstruction MSE and mean Hungarian-aligned |r| (averaged across constructs and seeds) are reported. This tests whether the four-dimensional choice is appropriate or whether a different number of latent dimensions better captures the item covariance structure.

### 3.10.6 Alignment Analysis: COR Constructs vs. Data-Driven Dimensions

All three unsupervised methods produce multi-dimensional representations that are subject to sign and permutation ambiguity — dimension 1 from PCA might correspond to BLI in the COR framework, or it might not correspond to any single construct. To identify the best-matching assignment of unsupervised dimensions to COR constructs, the **Hungarian algorithm** is applied to the matrix of absolute Pearson correlations between each data-driven dimension and each COR composite score.

Specifically, for a given unsupervised method with $D$ latent dimensions and $K = 4$ COR construct scores, the absolute correlation matrix $|r|$ of shape $(D \times K)$ is computed. The Hungarian algorithm finds the one-to-one assignment of dimensions to constructs that maximises the sum of assigned |r| values. The resulting assigned |r| per construct indicates how strongly the best-matching data-driven dimension correlates with the COR composite for that construct. Both Pearson and Spearman correlations are reported; a strong alignment (|r| ≥ 0.40) indicates that the corresponding COR construct dimension is empirically recoverable from the data-driven method.

The full pairwise (latent dimension × COR construct) correlation matrix is also reported without the Hungarian assignment, allowing inspection of whether construct scores correlate with multiple data-driven dimensions or with none.

### 3.10.7 Leakage-Free Encoding for CatBoost

When the unsupervised latent representations are used as features in CatBoost classification (Section 3.11), a leakage-free within-split encoding protocol is applied. Rather than fitting the unsupervised method on the full dataset and then using those representations in cross-validation, the encoder (PCA, EFA, or autoencoder) is re-fitted **on the training fold only** within each cross-validation iteration. The fitted encoder is then used to transform both the training and test folds. This eliminates transductive leakage that would otherwise arise from test-fold items influencing the latent space through the pre-fitted encoder, ensuring that the test-fold latent features are genuinely out-of-sample.

---

## 3.11 CatBoost Classification

### 3.11.1 Target and Classifier

The binary classification target is **HighAEV** (Section 3.7.5). CatBoost is selected because it handles mixed numerical and categorical features natively, is robust at the approximately 1,015-household sample size, and provides stable SHAP-compatible feature importances. All models are evaluated using stratified 5-fold cross-validation, reporting AUC-ROC, F1 score, precision, and recall per fold. Training versus test performance is explicitly logged to diagnose overfitting. CatBoost hyperparameters are tuned for the ~800-household training set: 600 boosting iterations, learning rate 0.02, maximum tree depth 3, $\ell_2$ leaf regularisation 8, subsample ratio 0.75, column sample rate 0.70, minimum data in leaf 20, and random strength 1.5. These conservative regularisation settings reflect the priority of generalisation over training accuracy at this sample size.

### 3.11.2 Model Variants

> **Terminology note (current pipeline).** The five-variant set below
> (`src/ml_classification.py`) is: **Controls_Only**, **Route1_Composite**,
> **Route2_SEM**, **Route3_VAE**, **AllRoutes_Hybrid**. An earlier iteration
> of this pipeline (predating the Route 2 SEM / Route 3 VAE
> implementation) instead compared six variants — `Controls_Only`,
> `SEM_COR` (= today's `Route1_Composite`), `Linear_AE`, `EFA`, `PCA`, and
> `Hybrid_SEM_AE` — with PCA/EFA/the linear autoencoder each as a
> standalone CatBoost feature set. In the current pipeline, **PCA, EFA, and
> the linear autoencoder are retained only as Route 1 empirical-recovery
> robustness checks** (`outputs/unsupervised_latent_robustness/`) — they
> assess whether the theory-guided COR composite dimensions are recoverable
> from the item structure, and are not standalone routes or separate ML
> model variants.

Five model variants are compared, one per COR estimation route plus the controls-only baseline:

| Variant | Features | Status |
|---------|----------|--------|
| **Controls_Only** | Household controls + energy-poverty proxies (no construct scores) | Primary generalisable predictor |
| **Route1_Composite** | Route 1 COR formative composite scores (FCP, AEMC, BLI, TCR) + controls | Theory-driven — structurally circular |
| **Route2_SEM** | Route 2 CFA factor scores + controls | Construct-overlap / representation-validation — not fully generalizable |
| **Route3_VAE** | Route 3 VAE latent means + controls | Construct-overlap / representation-validation — not fully generalizable |
| **AllRoutes_Hybrid** | Route 1 + Route 2 + Route 3 scores + controls | Not generalizable — multiple overlapping/circular components |

**Controls** include: income difficulty (S8), household size and type (H1, H2, H3), employment status (S2, S3), financial situation (S5, S6), energy-poverty flags (low-income indicator, high-cost indicator, LIHC risk category), and three engineered energy-efficiency controls — insulation presence, mains-gas heating share, and smart-meter ownership — reconstructed by `enable_preprocessing.build_efficiency_controls()` from ENABLE's per-option sub-items (H5A1-3, H6A3, H13A/H13C), since the single-column H5/H6/H13 fields referenced in earlier drafts of this table do not exist in the UK sub-sample. The heating-gas-share feature turns out to be one of the two most important Controls_Only features (on par with income difficulty). FES columns (and their underlying z-score components, across all 9 FES scenarios) are excluded from all variants because they carry zero within-UK-sample variance and would contribute no information — see `src.route_utils.exclude_fes_columns`.

### 3.11.3 Circularity and Generalisability

The Controls_Only model is the only variant that produces genuinely generalisable predictions. Its features are observable household characteristics and proxies that do not overlap with the items or composite formulas used to construct HighAEV. Any predictive signal in this model therefore reflects a meaningful empirical association between observable household characteristics and elevated vulnerability.

The Route1_Composite model includes the four construct scores that collectively determine the AEV composite (`mean(FCP, BLI, TCR, 1-AEMC)`) and thereby the HighAEV binary target. Because HighAEV is a deterministic function of these four scores, the Route1_Composite classifier is effectively given access to a monotone transformation of the target variable — near-perfect performance is algebraically expected, not empirically informative. This variant documents the upper bound of what the COR composite approach achieves by construction.

Route2_SEM and Route3_VAE use latent features derived from the same item pool (Route 2) or trained with `high_aev` directly in their own loss (Route 3), creating partial but not complete circularity — they are reported as construct-overlap / representation-validation models, not fully independent generalizable predictors. AllRoutes_Hybrid combines all three routes' scores and is therefore not a generalizable-prediction claim either. The performance comparison across all five variants — and the gap between Route1_Composite and Controls_Only — together quantify the extent to which predictive accuracy is attributable to structural circularity versus genuine generalisation.

---

## 3.12 SHAP Explainability

SHAP (SHapley Additive exPlanations) values are computed for the **Controls_Only** model and, for comparison, for the best-performing available route model excluding the fully circular Route1_Composite/AllRoutes_Hybrid (i.e. whichever of Route2_SEM/Route3_VAE scores highest) — labelled a construct-overlap / route-validation explanation, not a generalizable-prediction claim. SHAP values decompose each prediction into additive feature contributions satisfying efficiency, symmetry, and linearity, making them consistent attribution measures across heterogeneous feature types.

For the Controls_Only model, SHAP identifies which observable household characteristics — income difficulty, dwelling type, employment, fuel type, appliance ownership, energy-poverty proxy — most strongly drive HighAEV classification without any construct-score inputs. Global feature importance is summarised via mean absolute SHAP values across all households. Individual-level beeswarm plots display the direction and magnitude of each feature's contribution per respondent, allowing detection of asymmetric effects (e.g., features that are predictive only at extreme values). The SHAP analysis bridges the macro FES stream and the household vulnerability assessment by revealing whether the characteristics most predictive of high adaptive energy vulnerability align with the income-stress and adaptive-capacity dimensions theorised under COR.

---

## 3.13 Ethics and Limitations

**Ethical considerations.** The ENABLE.EU survey was conducted under the ethical frameworks of the participating national institutions and in compliance with applicable data protection regulations. The UK sub-sample data are anonymised and do not contain direct personal identifiers. This study uses the data solely for academic research on household energy vulnerability, consistent with the stated purposes of the ENABLE.EU project.

**Limitations of the macro forecasting stream.** The year 2017 fell in a low-stress period relative to the 2005–2016 training window; the resulting negative FES values may partly reflect this historical coincidence rather than a structural property of the models. The carbon forecast is particularly problematic: models trained on the 2014–2016 EUA price collapse systematically underpredict the 2017 EUA recovery, with zero prediction interval coverage for the core LSTM model. This failure reflects the well-known limitation of autoregressive models at structural breaks — the 2017 EUA recovery was driven by EU market reform expectations that could not be extrapolated from recent price history. The FES should be treated as a conditional stress index contingent on the selected forecast model, not an objective measure of macro-energy stress.

**Limitations of the household data stream.** The UK ENABLE sample of approximately 1,015 households is sufficient for exploratory psychometric and path analysis but is modest by large-survey standards. The cross-sectional design prevents causal inference: all associations among constructs and between the macro FES context and household vulnerability are correlational. The temporal misalignment between FES 2017 and the survey (collected 2016–2017) is addressed by using the annual mean FES as a contextual scalar for all UK households; however, this means FES carries zero within-sample variance and cannot function as a predictor in any regression model.

The AVE failure across all four constructs is the most consequential psychometric limitation. It implies that item pools do not converge on a sufficiently strong single latent factor to justify reflective SEM. The formative/composite interpretation adopted here is theoretically defensible and practically necessary, but it means path coefficients should not be interpreted as structural effects and construct score comparisons across contexts require care. Future research should develop purpose-built reflective items for energy-domain COR constructs, or apply item response theory models that can accommodate the mixed item formats present in the ENABLE.EU battery.

The structural circularity in AEV regressions and in the SEM_COR and Hybrid CatBoost variants is a feature of the composite design, not a methodological error. It is documented transparently throughout so that readers can calibrate their interpretation of R-squared values and classification accuracies accordingly. The three-stream robustness framework is designed precisely to make these distinctions visible: Robustness Check 3 separates what the data-driven methods can recover independently from what the COR framework imposes through its composite construction.
