# Chapter 4: Macro Forecasting and the Forecast Energy Stress Index

> **Terminology note (current pipeline).** This chapter refers to each FES
> construction as a "variant" (`Equal_Core`, `Equal_Macro`, `VW_Core`,
> `VW_Macro`, `Bayes_Core`, `Bayes_Macro`, `Actual_RollingVol`,
> `Actual_CrossComp`, `Actual_AbsShock`). In the current pipeline
> (`src/fes_scenarios.py`), these are named **scenarios** in a macro-level
> scenario-based signal simulation layer — `equal_core`, `equal_macro`,
> `vw_core`, `vw_macro`, `bayesian_core`, `bayesian_macro`, `actual_A`,
> `actual_B`, `actual_C` — with identical underlying numbers to what is
> reported below. The results and interpretation in this chapter are about
> the macro stress signal only; none of them are, or imply, a household-level
> prediction (see Chapter 5 onward for the independently-estimated HighAEV
> routes, and the README's "Linking FES Scenarios and HighAEV" section for
> how the two are connected only through scenario-conditioned
> interpretation).

## 4.1 Descriptive Behaviour of Gas, Electricity, and Carbon Growth

Before evaluating model performance, it is useful to characterise the statistical properties of the three energy series during the study period and forecast horizon. Figure `outputs/figures/forecast_vs_actual_all_series.png` plots each series from the training window (2005–2015) through the validation year (2016) and into the 2017 forecast horizon alongside out-of-sample actuals.

**Gas (ONS RPI, year-on-year percentage growth).** The training period encompasses two distinct regimes: a high-price, high-growth era from 2005 through 2013, punctuated by the 2008 commodity super-cycle spike (growth exceeding +25 p.p. YoY in some months), followed by a prolonged decline as global LNG supply expanded and the 2014 oil-price collapse transmitted into retail gas markets. By early 2016 gas growth had turned negative, and in 2017 it remained in mild deflation throughout the year, moving monotonically from −5.3 p.p. in January toward near-zero (−0.1 p.p.) by December. This upward drift — gas becoming less deflationary as the 2014–2016 trough base recedes — is the primary signal that forecasting models are required to capture. The overall 2017 trajectory is therefore a gentle upward trend within a negative range, ending the year close to, but not reaching, zero growth.

**Electricity (ONS CPI 04.5.1, 12-month percentage change).** UK household electricity prices showed a more complex seasonal-and-level structure. During the training window, electricity growth oscillated between approximately −5 p.p. and +10 p.p. with strong quarterly seasonality reflecting billing cycle timing and the tariff-change patterns of the regulated retail market. In 2017, actual growth exhibited two phases: a low-to-negative phase in Q1 (−0.2 p.p.) rising to a moderate-positive plateau in Q3–Q4 (+9.0 to +11.4 p.p.), reflecting the pass-through of wholesale energy costs into retail bills in the second half of the year. The annual mean z-score of actual electricity growth in 2017 (z = +0.01 relative to the 2005–2016 reference period) is close to the long-run average, masking this within-year divergence: early 2017 electricity growth was below average while late 2017 was above average.

**Carbon (ICE EUA futures, log-return).** Carbon log-returns are the most volatile and structurally challenging series. The training window spans three structural regimes: Phase I over-allocation (pre-2012, EUA prices near zero, log-returns explosive), Phase II correction (2012–2014), and the 2015–2016 collapse as EUA prices fell toward the €4–5/tonne floor. Models trained on this history absorb a dominant downward-trend and high volatility signal. The 2017 actual carbon series represents a sharp structural break: EUA prices began recovering in mid-2017 (supported by expectations of the forthcoming Market Stability Reserve mechanism), producing log-returns ranging from −22% in April to +55% in November. This regime change — from a multi-year collapse to a nascent recovery — is not predictable from historical patterns alone, creating the most challenging forecast environment of the three series.

---

## 4.2 Forecasting Performance: Gas

Figure `outputs/figures/forecast_vs_actual_gas.png` overlays all eight gas forecasts (four model families × two modes) against the 2017 actuals, with 95% prediction intervals shown. Figure `outputs/figures/model_ranking_polar_gas_core.png` and `outputs/figures/model_ranking_polar_gas_macro.png` display the multi-metric rank profiles in polar-chart form, one per mode.

### 4.2.1 Core Mode (Target History Only)

The Temporal Fusion Transformer (TFT) achieved the lowest 2017 forecast error in core mode, with a mean absolute error of **1.03 p.p.** and a SMAPE of 106.2% (`outputs/forecasts/gas_growth_pct_forecasts_core.csv`). The SMAPE value is inflated by near-zero denominators when actual gas growth is close to zero — a known instability that motivated the exclusion of MAPE from the rank-aggregation criterion in Chapter 3 (Section 3.4.2). The directional story is more informative: TFT-core correctly identifies the upward trend from January's trough (forecast: −3.2 p.p.) through the year, capturing the gradual unwinding of gas deflation, though it becomes moderately optimistic in H2, predicting a return to positive growth (forecast: +2.2 p.p. in December) when actuals remained marginally negative (−0.1 p.p.).

LSTM-core ranked second on 2017 forecast MAE (3.80 p.p.) but achieved the strongest validation-year (2016) performance across all eight gas models, with a rank-aggregation composite score of 1.10 — the best on the multi-metric leaderboard (Table 4.1). This divergence between validation rank and forecast accuracy is consequential: LSTM-core generalises well to the validation period but loses precision over the 12-month forecast horizon, possibly because its 12-step-ahead roll-forward accumulates error as gas growth approaches zero and the model's learned dynamics push toward a flat trend rather than the gentle U-shaped recovery that occurred.

SARIMA-core (forecast MAE: 6.12 p.p.) and Prophet-core (forecast MAE: 6.30 p.p.) both predicted continued strong gas deflation beyond what materialised, generating systematic positive bias in the second half of the year. The SARIMA 95% PI grew very wide (±40 p.p. by December), capturing the actuals but providing little actionable signal. Prediction interval coverage for TFT-core is only 8.3% — the intervals are too narrow — while Prophet-core and SARIMA-core achieve 100% coverage, albeit at the cost of extremely wide intervals.

### 4.2.2 Macro Mode (With Exogenous Regressors)

Prophet-macro was selected as the best gas model in macro mode with a 2017 forecast MAE of **2.44 p.p.** and SMAPE of **197.0%** (Figure `outputs/figures/shap/gas_prophet_macro_attribution.png`). Inspection of monthly errors shows a **systematic positive bias across the entire year**, not a cancellation pattern: the model forecasts gas growth rising from 0.36 p.p. in January to a mid-year peak of 2.35 p.p. in July, while actual gas growth remained negative throughout, from −5.3 p.p. in January easing only to −0.1 p.p. by November–December. The error is largest in Q1 (5.66 p.p. in January, 5.42 p.p. in February) and shrinks steadily as the year progresses (1.08–1.45 p.p. in Q3, 0.09–1.17 p.p. in Q4), reflecting the actual series' own convergence toward zero rather than any correction in the model's upward-biased trajectory. LSTM-macro achieved a 2017 MAE of 3.85 p.p. with a validation rank score of 1.10 (Prophet-macro's is 3.43), comparable to LSTM-core on validation but still behind Prophet-macro on 2017 forecast accuracy despite Prophet-macro's own large positive bias.

Prophet-macro adds interpretable macro decomposition that LSTM and TFT do not provide in the same form, though the current fit's decomposition differs materially from earlier runs of this pipeline (Prophet's fit is not perfectly seed-stable). The **trend component is the dominant driver throughout the year**, not just in H1: it contributes a steadily *declining but always positive* attribution, from +2.56 in January to +1.78 in December — this persistently positive trend is the direct source of the model's systematic upward bias against a series that actually stayed in deflation all year. `inflation_growth_lag1` is the main negative macro contributor across all 12 months (−0.15 to −0.40), economically plausible because low CPIH deflates the price index against which gas prices are measured, but not large enough to offset the trend. Gas futures (log-return lag) switch from a small positive contribution in H1 (+0.04 to +0.24, Jan–Jun) to a small negative one in H2 (−0.04 to −0.26, Jul–Dec), tracking the NBP forward curve moving from backwardation to contango; GDP growth and the yearly seasonal term contribute smaller, more erratic adjustments, with the yearly term showing a distinct positive spike (+0.78) in July.

---

## 4.3 Forecasting Performance: Electricity

Figure `outputs/figures/forecast_vs_actual_electricity.png` and polar charts `outputs/figures/model_ranking_polar_electricity_core.png` / `outputs/figures/model_ranking_polar_electricity_macro.png` summarise electricity performance across all eight models.

### 4.3.1 Core Mode

TFT-core again ranks first on 2017 forecast accuracy for electricity, with a MAE of **3.85 p.p.** and SMAPE of 93.8% (`outputs/forecasts/electricity_growth_pct_forecasts_core.csv`). The model correctly predicts near-zero to low-positive electricity growth in H1 but fails to anticipate the sharp rise in H2 (actual +9 to +11 p.p. in Q3–Q4, forecast +4 to +6 p.p.). This systematic underestimate of late-year electricity price pass-through is economically understandable: the model has no information about the timing of supplier tariff adjustments and treats the H2 rise as beyond its learned seasonal baseline.

LSTM-core (forecast MAE: 5.86 p.p.) shows similar qualitative behaviour but with higher absolute error and a validation rank of 3.05 — third across the electricity core field. SARIMA-core produces the pathological result noted in the data issues section: its electricity SMAPE reaches **1,348%** on the 2016 validation period, driven by months in which actual electricity growth is effectively zero and SARIMA predicts a small but non-zero value. This extreme SMAPE is the exact case for which MAPE-family metrics were excluded from the rank-aggregation scoring. Prophet-core (forecast MAE: 8.32 p.p., SMAPE: 187.6%) performs worst on 2017 accuracy.

### 4.3.2 Macro Mode

LSTM-macro was selected as best electricity model in macro mode with a 2017 forecast MAE of **4.22 p.p.** and SMAPE of **92.3%** — the most competitive macro model across all three series. Its validation rank score is 1.10, again the best in the macro electricity field, making it the only model where validation ranking and 2017 selection are entirely consistent. The macro regressors providing positive information are documented via TS-SHAP in Section 4.6: weather volatility (which signals demand-side heating pressure that propagates into retail prices), gas futures (cross-energy price correlation), and a post-2016 electricity regime indicator.

SARIMA-macro (forecast MAE: 6.97 p.p.) and Prophet-macro (forecast MAE: 4.69 p.p.) are both outperformed by LSTM-macro on absolute error. TFT-macro (forecast MAE: 5.84 p.p.) performs comparably to SARIMA but with wider prediction intervals that nonetheless achieve 50% coverage — better than SARIMA-macro or TFT-core for electricity.

---

## 4.4 Forecasting Performance: Carbon

Carbon is the most difficult series and yields the most instructive failure. Figure `outputs/figures/forecast_vs_actual_carbon.png` plots all eight carbon forecasts against the 2017 actual EUA log-returns, and `outputs/figures/prediction_intervals_2017.png` shows the full prediction interval comparison across series.

### 4.4.1 Core Mode: Structural Forecast Failure

LSTM-core was selected as best in core mode (forecast MAE: **15.75 p.p.**, SMAPE: 88.3%) — yet this comes with a critical caveat: **prediction interval coverage = 0%** over the entire 12-month horizon (`outputs/forecasts/carbon_growth_pct_forecasts_core.csv`). Every single one of the 12 monthly actuals fell outside the model's 95% Monte Carlo dropout interval. The model consistently predicted a continuation of the 2015–2016 EUA price collapse, forecasting log-returns in the range of −35% to −8%, while actuals recovered from −22% in April to +55% in November. The LSTM had no mechanism to anticipate the 2017 EUA regime shift — its gradient saliency attributions (TS-SHAP, Figure `outputs/figures/shap/carbon_lstm_core_attribution.png`) show a mean absolute attribution of only 0.009, indicating the model is predicting near its training mean and responding very weakly to input history. This near-trivial gradient signal is a diagnostic of a model in extrapolation mode: it has been trained on persistent decline, and declining momentum is what it outputs.

TFT-core (forecast MAE: 27.5 p.p., SMAPE: 155%) achieves 50% PI coverage — better than LSTM-core — but at the cost of much higher absolute error and very wide intervals. Prophet-core covers 100% of actuals (interval width spanning ±130 p.p.) but with a MAE of 20.4 p.p. SARIMA-core achieves 100% coverage with a similar interval width of ±200 p.p. — effectively non-informative prediction intervals.

### 4.4.2 Macro Mode: TFT Outperforms

TFT-macro was selected as the best carbon model in macro mode with a forecast MAE of **15.96 p.p.** and SMAPE of **76.6%** — the best SMAPE of any carbon model, confirming it as the model with the most accurate proportional tracking of the 2017 recovery (`outputs/forecasts/carbon_growth_pct_forecasts_macro.csv`). However, TFT-macro also achieves 0% PI coverage, indicating that while its point forecasts are directionally better (mean carbon growth correctly positive), uncertainty bounds remain too narrow relative to the realised volatility. LSTM-macro (forecast MAE: 20.8 p.p., SMAPE: 112.2%, PI coverage: 0%) performs worse on both MAE and SMAPE than TFT-macro.

The macro regressors provide a modest but meaningful directional improvement. TS-SHAP analysis (Figure `outputs/figures/shap/carbon_tft_macro_attribution.png`) shows `gdp_growth_lag1` contributing positively throughout the year — economic growth supports carbon demand and EUA prices — while `inflation_growth_lag1` is also positive (rising prices co-move with energy commodity prices). The `carbon_growth` own-history feature contributes small negative attributions, suggesting a mild mean-reversion expectation. Taken together, macro context correctly biases TFT toward positive carbon territory, though the magnitude of the 2017 recovery remains beyond what any model in the suite can quantitatively predict.

**Implication for FES construction.** Given the 0% PI coverage of LSTM-core, TFT-macro is used as the primary carbon component when building FES_macro. For FES_core, the LSTM-core point forecast is retained (as the lowest-MAE core model) but flagged as carrying structurally unreliable prediction intervals; the uncertainty component for FES_core therefore reflects overly narrow carbon PI half-widths.

---

## 4.5 Model Selection Summary

Table 4.1 consolidates the model selection outcomes across all six series-mode combinations. Selection was made on the basis of 2017 forecast accuracy against realised values (selection_basis = forecast_actual), with validation-period rank-aggregation scores providing secondary context. The rank-aggregation score (weighted sum of ranks across MAE, RMSE, SMAPE, MASE, quantile loss, Winkler score, MSIS, and PI coverage) covers the 2016 hold-out year; a lower validation rank indicates better validation performance.

**Table 4.1. Model Selection Summary — Best Model Per Series and Mode**

| Series | Mode | Selected Model | Val Rank Score | Forecast MAE (p.p.) | Forecast SMAPE (%) | PI Coverage |
|--------|------|----------------|:--------------:|:-------------------:|:-----------------:|:-----------:|
| Gas | Core | TFT | 4.00 | **1.03** | 106.2 | 8.3% |
| Gas | Macro | Prophet | 3.43 | **2.44** | 197.0 | 100.0% |
| Electricity | Core | TFT | 3.95 | **3.85** | 93.8 | 16.7% |
| Electricity | Macro | LSTM | 1.10 | **4.22** | 92.3 | 66.7% |
| Carbon | Core | LSTM | 1.45 | **15.75** | 88.3 | **0.0% ⚠** |
| Carbon | Macro | TFT | 3.13 | **15.96** | 76.6 | **0.0% ⚠** |

*Note: Val Rank Score is the composite multi-metric rank on the 2016 hold-out period (lower = better performance in 2016). Selection was confirmed by lowest 2017 forecast MAE. Forecast MAE and SMAPE are computed against 2017 realised values. MAPE is excluded throughout (near-zero denominators produce explosive values, as evidenced by SARIMA electricity MAPE = 1,348%). PI Coverage is the fraction of 12 monthly actuals falling within the 95% prediction interval. Source: `outputs/tables/model_metrics_comparison.csv`.*

A notable pattern in Table 4.1 is the absence of consistent validation-to-forecast correspondence. TFT-core has the worst validation rank for gas (4.00) yet the best 2017 accuracy (MAE=1.03). Conversely, LSTM-core has the best validation rank (1.10) but second-worst 2017 gas accuracy. This disconnect reflects the structural properties of 2017 gas series: TFT's attention mechanism appears to weight the short-run upward trend signal correctly even when its 2016 rolling-window performance was poor, while LSTM's recurrent memory retains a stronger imprint of the pre-2016 deflation regime. For electricity macro, LSTM-macro is the only model where the best validation rank (1.10) and best forecast accuracy (MAE=4.22) coincide. For carbon, the 2017 regime shift means that neither validation nor forecast accuracy can be interpreted as a reliable forward-looking signal.

---

## 4.6 TS-SHAP Forecast Attribution

TS-SHAP attributions are computed per model class using architecture-appropriate methods: TFT attention-weighted gradient×input, Prophet additive component decomposition, and LSTM gradient×input (saliency). Outputs are in `outputs/figures/shap/` and supporting tables in `outputs/shap/ts/`. Attribution figures are presented per series-mode pair (e.g., `outputs/figures/shap/gas_tft_core_attribution.png`).

### 4.6.1 Gas (TFT-Core)

The TFT-core gas model has a single input feature in core mode — the gas growth history — and the gradient×input attribution reflects how historical growth values at different lags influence each forecast step. Attribution values are negative in January–February 2017 (−0.040, −0.018), turn positive from March onward, and peak in May–June (0.120, 0.123), before decaying back toward zero by December (0.004). This pattern indicates that recent historical gas growth (the negative values in H2 2016) initially pushes the forecast downward, but the model correctly identifies the levelling-off signal and shifts to an upward influence as the forecast horizon extends. The mean absolute attribution across the year is 0.056, indicating moderate responsiveness to historical input — the model is not near its training mean in the way that LSTM-core carbon is (mean |attribution| = 0.009). The attention-heatmap (`outputs/figures/shap/gas_tft_core_attention.png`) confirms that short-to-medium lags (3–9 months back) receive the highest attention weights for gas, consistent with the model learning a 6–9-month inertia structure in gas price dynamics.

### 4.6.2 Gas (Prophet-Macro)

The Prophet-macro decomposition (`outputs/figures/shap/gas_prophet_macro_attribution.png`, `outputs/shap/ts/gas_prophet_macro_importance.csv`) provides the richest multi-feature attribution of any selected model, and in the current fit it also explains directly why the model carries a large positive bias against an actually-deflationary gas series. The **trend component dominates every month of 2017** (mean |attribution| = 2.17, an order of magnitude larger than any other feature), declining smoothly from +2.56 in January to +1.78 in December but never turning negative — this persistently positive trend is the direct mechanism behind the model's systematic overestimate of gas growth (Section 4.2.2). `inflation_growth_lag1` is the largest negative contributor (mean |attribution| = 0.31), consistently between −0.15 and −0.40 across all 12 months, reflecting the deflationary CPIH macro environment, but an order of magnitude too small to offset the trend term. `gas_futures_log_return_lag1` (mean |attribution| = 0.12) provides the clearest time-varying signal: positive in H1 (+0.04 to +0.24, forward curve in backwardation) and negative in H2 (−0.04 to −0.26, forward curve moving into contango as the summer LNG glut dampened forward prices). The `yearly` seasonal term is small in most months but spikes to +0.78 in July. `gdp_growth_lag1` and `gas_futures_yoy_growth_lag1` contribute the smallest and most erratic adjustments (mean |attribution| = 0.05 and 0.01 respectively).

### 4.6.3 Electricity (TFT-Core and LSTM-Macro)

The TFT-core electricity attention heatmap (`outputs/figures/shap/electricity_tft_core_attention.png`) shows the model heavily weighting the 12-month lag — the same calendar month one year ago — consistent with strong annual seasonality in electricity billing. For LSTM-macro electricity, the importance table (`outputs/shap/ts/electricity_lstm_macro_importance.csv`) reveals `weather_volatility_lag1` as the highest-attribution macro variable (mean |attribution| = 0.00082), followed by `gas_futures_yoy_growth_lag1` (0.00070) and `gas_futures_log_return_lag1` (0.00058). The `post_2016_electricity_regime` indicator — a structural break dummy capturing the regime shift following 2016 regulatory changes — contributes modestly (0.00023) and is consistently positive, representing the model's learned response to the H2 2017 electricity price elevation. Together, these attributions indicate that LSTM-macro's advantage over LSTM-core for electricity forecasting comes primarily from incorporating weather uncertainty (which drives heating-related demand) and gas-to-power price linkages, with a secondary contribution from regime awareness.

### 4.6.4 Carbon (LSTM-Core and TFT-Macro)

The carbon LSTM-core attribution is instructive precisely because of its near-absence: mean absolute attribution across all 12 months is only 0.009 (Figure `outputs/figures/shap/carbon_lstm_core_attribution.png`). The gradient×input values are slightly negative in early months (−0.020 in January, −0.008 in February) and weakly positive from March onward, peaking at +0.018 in May before decaying to near-zero. This pattern indicates the model extrapolates a gentle upward trend from within-sample dynamics but lacks the amplitude to predict the EUA recovery that actually materialised. In effect, the LSTM has learned to predict mean-reversion from the 2015–2016 lows, but at a pace orders of magnitude slower than what the market delivered.

TFT-macro carbon attribution (`outputs/figures/shap/carbon_tft_macro_attribution.png`) shows `gdp_growth_lag1` as the most important positive macro driver (attribution approximately +0.008 to +0.015 across all months), `inflation_growth_lag1` as a secondary positive driver (+0.005 to +0.006), and `weather_volatility_lag1` as a consistent negative pull (−0.002 to −0.004). The `carbon_growth` own-history feature is a small negative contributor (mean attribution −0.001), reflecting a mean-reversion component in the TFT's representation of carbon momentum. The macro context thus biases the TFT toward a slightly positive EUA outlook — consistent with an expanding economy with rising energy demand — but cannot reproduce the 2017 structural recovery, which was driven by anticipation of the 2019 Market Stability Reserve mechanism rather than any contemporaneously observable macro variable.

---

## 4.7 FES Monthly Dynamics

Figure `outputs/figures/fes_monthly_2017.png` plots all FES variants month by month across 2017, together with the three actual stress benchmarks (Options A–C). Table `outputs/fes/fes_monthly_2017.csv` provides the full numerical record.

The monthly trajectories reveal three qualitatively distinct dynamics:

**FES_core (equal-weight, core models)** follows a monotonically ascending path from −1.631 in January to −0.253 in December. This is driven primarily by the gas component: TFT-core gas forecasts drift steadily from −3.2 p.p. (January) to +2.2 p.p. (December), pulling the gas z-score upward from −0.716 to −0.384, and the uncertainty z-score gradually rises as expanding TFT prediction intervals widen the PI half-width throughout the year. The electricity and carbon z-scores contribute less within-year variation (electricity z is nearly constant, carbon z steps gently upward). The result is a smooth, monotone FES_core trajectory — aesthetically clean but potentially underestimating month-to-month energy market volatility.

**FES_macro (equal-weight, macro models)** is notably less smooth than FES_core. It begins at −0.887 in January, rises fairly steadily to a mid-year peak of −0.185 in July, then eases back down to −0.384 by December — a hump-shaped rather than monotone path. The mid-year peak is driven primarily by the gas component: Prophet-macro's persistently positive trend attribution (Section 4.6.2) combines with the carbon component's own rise through the summer, before easing off in H2 as the carbon and gas contributions partially reverse. The uncertainty term (z_unc_macro) stays small and positive throughout (+0.017 to +0.125, peaking in July) rather than swinging into strongly negative territory the way FES_core's uncertainty term does — a materially calmer uncertainty profile than in earlier runs of this pipeline. The macro variant therefore still captures different within-year dynamics from the core variant, but the specific shape is sensitive to the Prophet-macro gas refit and should not be over-interpreted as a stable structural feature.

**Actual stress benchmarks.** Option B (cross-component standard deviation, `fes_actual_B`) is the most behaviour-rich series among the three actual benchmarks. It starts at −1.849 in January — reflecting very low cross-component dispersion when all three energy series are weakly negative — and rises through H2, reaching a brief positive value of +0.164 in November 2017, driven by the simultaneous occurrence of high carbon log-returns (+55.4%, z = +0.348) and elevated electricity growth (+11.4 p.p., z = +0.500). November 2017 is the only month in the 2017 horizon where the actual stress benchmark crosses above zero, signalling that the carbon recovery and electricity price rise briefly combined to create a stress episode that forecasting models did not anticipate. Option A (rolling 3-month volatility, `fes_actual_A`) stays persistently more negative than Option B, reflecting the smoothing effect of a 3-month rolling window. Option C (absolute shock, `fes_actual_C`) is the most volatile of the three.

The core FES and macro FES are both more negative than the actual benchmarks in H1 (below-average stress correctly forecast) but fail to rise with the actual benchmarks in H2 because models underestimated the carbon recovery and electricity price lift.

---

## 4.8 FES Annual Comparison

Table 4.2 summarises the annual mean FES values for each variant, derived from `outputs/fes/fes_summary_2017.csv` and `outputs/fes/fes_components_table.csv`. Figure `outputs/figures/fes_bayesian_uncertainty.png` overlays Bayes_Core and Bayes_Macro with their posterior standard deviation bands.

**Table 4.2. Annual FES Summary — 2017 Component z-Scores and Totals**

| Variant | Gas z | Electricity z | Carbon z | Uncertainty z | **FES Total** |
|---------|:-----:|:------------:|:--------:|:------------:|:------------:|
| Equal_Core | −0.555 | −0.336 | +0.113 | −0.151 | **−0.929** |
| Equal_Macro | −0.450 | −0.244 | +0.150 | +0.072 | **−0.472** |
| Actual (CrossComp) | −0.600 | +0.012 | +0.155 | −0.394 | **−0.827** |
| VW_Core | — | — | — | — | **−0.328** |
| VW_Macro | — | — | — | — | **−0.233** |
| Bayes_Core | — | — | — | — | **−0.281** |
| Bayes_Macro | — | — | — | — | **−0.190** |

*Source: `outputs/fes/fes_summary_2017.csv`. All z-scores computed relative to the 2005–2016 reference period. For VW and Bayes variants, individual component z-scores are the same as their equal-weight counterparts; only the aggregation weights differ.*

All seven variants agree on the sign and broad magnitude of the result: **2017 was a year of below-average energy-carbon stress for UK households**, ranging from −0.19 to −0.93 standard deviations below the 2005–2016 historical mean. This places 2017 in approximately the lowest quartile of the historical stress distribution, consistent with the macroeconomic record of falling gas prices, stable electricity retail markets, and a carbon market only beginning to recover from its 2016 nadir.

The gas z-score is the largest negative contributor across all variants (−0.45 to −0.60), reflecting both the directional consensus and the magnitude of the 2017 gas price decline. Carbon is the only component with a consistently positive z-score (+0.11 to +0.16), as EUA log-returns were above the historical average even before accounting for the full magnitude of the H2 recovery. Electricity sits near zero in the actual benchmark (+0.012) but is negative in both forecast variants (−0.24 to −0.34) because neither TFT-core nor LSTM-macro fully capture the late-year electricity price rise. The uncertainty component diverges directionally between core (−0.151, reflecting narrow TFT prediction intervals) and macro (+0.072, reflecting wider but less extreme macro-model uncertainty than in earlier runs), and this divergence accounts for roughly 0.22 SD of the 0.457 SD gap between Equal_Core (−0.929) and Equal_Macro (−0.472) — still the single largest contributor to that gap, alongside a 0.105 SD gas-component contribution.

---

## 4.9 FES Component Decomposition

Figure `outputs/figures/fes_components_2017.png` decomposes each FES variant into its four z-score components as stacked bar charts for each of the twelve months.

Three structural observations stand out from the decomposition.

First, **gas dominates the negative contribution throughout the year** across all variants. In the equal-weight specification, gas z-scores range from −0.716 (January) to −0.384 (December), consistently pulling the total FES downward. The magnitude of the gas contribution is largest in Q1 (contributing approximately −0.7 SD to the total) and smallest by year-end (−0.38 SD), driven by the base-effect unwinding documented in Section 4.1.

Second, **carbon provides the only consistently positive contribution**, and its positive role grows over the course of the year. In core mode, carbon z-scores rise from +0.109 (January) to +0.257 (December) as the TFT-macro and LSTM-core point forecasts converge toward increasingly positive EUA territory. In the actual benchmark, carbon z-scores peak in November (+0.348) before receding slightly in December (+0.207), reflecting the actual EUA price trajectory. The growing positive carbon contribution partially offsets the gas component in H2, explaining why the actual FES benchmark rises from −1.849 in January toward −0.035 in December despite all three variants showing a consistently negative annual average.

Third, **the electricity and uncertainty components diverge between forecast variants and actuals in a directionally important way**. The electricity z-score in actual benchmark data switches from negative in Q1 (−0.708) to positive in Q3–Q4 (+0.250 to +0.500), but in both forecasting variants the electricity z-score remains negative throughout (core: −0.603 to −0.064; macro: −0.400 to −0.064). This systematic underestimation of the H2 electricity price rise means FES_core and FES_macro fail to register the late-year stress contribution that the actual data captures. Policymakers relying on real-time forecasts in mid-2017 would have underestimated the degree to which electricity prices were beginning to pressure household budgets in Q3–Q4.

---

## 4.10 FES Robustness Comparison

Two robustness checks are embedded in the FES construction: macro augmentation (Robustness 1) and actual realised benchmarks (Robustness 2). Figure `outputs/figures/fes_robustness_comparison.png` presents a direct visual comparison; `outputs/figures/fes_metrics_rmse_heatmap.png` and `outputs/figures/fes_metrics_pearson_r_heatmap.png` display RMSE and Pearson correlation heatmaps across all FES-variant × actual-benchmark combinations. Full metrics are in `outputs/fes/fes_comparison_metrics.csv`.

### 4.10.1 Robustness 1: Macro-Augmented FES

FES_macro (annual mean = −0.472) is approximately 0.457 SD less negative than FES_core (−0.929); the actual crosscomp benchmark sits at −0.827, still numerically between the two (−0.929 < −0.827 < −0.472) but now much closer to FES_core. Whereas earlier runs of this pipeline found FES_macro landing closer to the actual benchmark than FES_core, the current fit reverses that: |FES_core − Actual| = 0.102 SD versus |FES_macro − Actual| = 0.355 SD — **FES_core is now more than three times closer to the realised benchmark than FES_macro**, driven by Prophet-macro's systematic positive gas bias (Section 4.2.2) pulling the macro index well past the actual level. Trajectory tracking also favours the core variant: Equal_Macro versus Actual_CrossComp achieves Pearson r = 0.887 and R² = 0.204, compared to Equal_Core's r = 0.970 and R² = 0.804.

This finding reverses the previous interpretation: in the current fit, macro augmentation does not improve level calibration, and it still does not improve trajectory tracking. The practical implication is that the equal-weight core-only specification is the more defensible default for both annual-planning and month-by-month FES monitoring in this run, with FES_macro retained primarily as a robustness/sensitivity check on the core signal — its divergence from FES_core (and from FES_macro's own prior-run behaviour) is itself informative about how sensitive the macro-augmented gas forecast is to Prophet's stochastic fitting.

### 4.10.2 Robustness 2: Realised Actual Benchmarks

Table 4.3 reports the full set of FES comparison metrics against all three actual benchmarks for the four primary forecast variants. The most important finding is that **Equal_Core tracks Actual_CrossComp best of any variant-benchmark pair**, achieving Pearson r = 0.970, R² = 0.804, MAE = 0.288, and Theil's U = 1.154.

**Table 4.3. FES Tracking Metrics Against Actual Benchmarks (Selected Variants)**

| FES Variant | vs Actual_CrossComp | | | vs Actual_RollingVol | | vs Actual_AbsShock | |
|-------------|:-------------------:|:--:|:--:|:--------------------:|:--:|:--------------:|:--:|
| | Pearson r | R² | MAE | Pearson r | R² | Pearson r | R² |
| Equal_Core | **0.970** | **0.804** | **0.288** | 0.923 | −0.588 | 0.802 | −4.147 |
| Equal_Macro | 0.887 | 0.204 | 0.535 | 0.924 | −3.027 | 0.514 | −14.93 |
| VW_Core | 0.962 | −0.001 | 0.544 | 0.885 | −3.941 | 0.836 | −19.65 |
| Bayes_Core | 0.965 | −0.289 | 0.623 | 0.880 | −4.436 | 0.830 | −21.69 |

*Source: `outputs/fes/fes_comparison_metrics.csv`. N = 12 months (Jan–Dec 2017). Pearson r and R² are computed on monthly FES series. MAE is in z-score units. Negative R² indicates the benchmark mean is a better predictor than the FES variant. Theil's U > 1 indicates the forecast is worse than a naive walk.*

The R² values against Actual_RollingVol and Actual_AbsShock are uniformly negative, indicating systematic level mismatches: all forecast FES variants predict more negative (lower stress) values than these two benchmarks suggest. This is expected: rolling-window volatility and absolute shock benchmarks incorporate short-term price noise that smooth 12-month forecasts suppress. The Pearson correlations with these benchmarks remain high (0.88–0.92), confirming that directional tracking is maintained even when level accuracy fails.

VW_Core and Bayes_Core do not improve on Equal_Core against any benchmark. Against Actual_CrossComp, they achieve near-zero or negative R² despite high Pearson r, because their reweighting introduces level shifts (VW_Core = −0.328, Bayes_Core = −0.281 vs. Equal_Core = −0.929) that move them away from the benchmark mean. This result is particularly striking for Bayes_Core: the Kalman smoother adds temporal stability but at the cost of a level bias that widens the gap to actuals. The simplest specification — equal-weight, core-only forecasts — proves to be the best-performing FES variant for 2017 by the dominant benchmark metric.

---

## 4.11 Chapter Summary: Addressing RQ1–RQ3

### RQ1: How Can UK Gas, Electricity, and Carbon Growth Be Forecasted and Integrated into a 2017 FES Index?

This chapter demonstrates a full forecasting pipeline for the three energy-carbon growth series spanning the period 2005–2017, with the 2016 hold-out year used for validation and 2017 as the out-of-sample forecast horizon. Four model architectures — SARIMA, Prophet, LSTM (with Monte Carlo dropout uncertainty), and Temporal Fusion Transformer (quantile uncertainty) — are trained in two modes (core: target series only; macro: augmented with CPIH, GVA growth, weather volatility, NBP gas futures, electricity demand, and GBP/EUR exchange rate). Model selection is confirmed by 2017 forecast accuracy.

The winning models — TFT-core for gas (MAE = 1.03 p.p.), TFT-core for electricity (MAE = 3.85 p.p.), and LSTM-core for carbon (MAE = 15.75 p.p.) in core mode; Prophet-macro for gas (MAE = 2.44 p.p.), LSTM-macro for electricity (MAE = 4.22 p.p.), and TFT-macro for carbon (MAE = 15.96 p.p.) in macro mode — are translated into monthly z-scores relative to the 2005–2016 reference period and aggregated into the FES via the equal-weight specification:

> FES_t = z(gas_t) + z(electricity_t) + z(carbon_t) + z(uncertainty_t)

The resulting index signals a below-average energy-carbon stress environment in 2017 across all model variants, consistent with falling gas retail prices, near-neutral electricity growth, and a carbon market only beginning to recover from its 2016 nadir. The three actual realised benchmarks (cross-component dispersion, rolling volatility, absolute shock) confirm this directional finding: actual 2017 energy stress was below the 2005–2016 mean in 11 of 12 months, with only November 2017 briefly crossing into mildly above-average territory owing to the confluence of the EUA price recovery and elevated electricity pass-through costs.

### RQ2: Do Macro-Augmented Models Improve Alignment Between Forecasted and Realised 2017 Stress Versus Core Models?

The answer is **mixed and instrument-dependent**. At the series level, macro regressors provide a directional improvement for carbon forecasting: TFT-macro achieves a lower SMAPE (76.6%) than LSTM-core (88.3%) and avoids the complete directional failure that LSTM-core exhibits. For gas, Prophet-macro still achieves a lower absolute MAE (2.44 p.p.) than LSTM-macro (3.85 p.p.) and remains the selected macro model, but its own forecast now carries a large, systematic positive bias against an actually-deflationary series (Section 4.2.2) — the macro selection is the "least-bad" option among the four macro-mode gas models, not a well-calibrated forecast in absolute terms. For electricity, LSTM-macro (MAE = 4.22 p.p.) is competitive with but not dominant over TFT-core (MAE = 3.85 p.p.).

At the FES level, macro augmentation moves the annual index from −0.929 (Equal_Core) to −0.472 (Equal_Macro) — a 0.457 SD shift — but in the current fit this **overshoots** the actual benchmark (−0.827) rather than approaching it: FES_core's absolute distance from actual (0.102 SD) is now more than three times smaller than FES_macro's (0.355 SD). Trajectory tracking also favours the core variant: Equal_Macro achieves R² = 0.204 versus Equal_Core's R² = 0.804 against the Actual_CrossComp benchmark. Macro augmentation therefore improves neither level calibration nor month-by-month coherence in this run. The uncertainty component — which turns from negative (core: −0.151) to small positive (macro: +0.072) when macro models are used — remains the single largest contributor to the level shift (0.223 of the 0.457 SD gap), with the gas component contributing a further 0.105 SD via Prophet-macro's persistent positive bias (Section 4.6.2).

Overall, macro augmentation functions in this run as a **sensitivity/robustness check that reveals fragility rather than improvement**: it narrows the level gap for carbon (TFT-macro's better-calibrated direction) but widens it for gas (Prophet-macro's systematic overestimate), and it does not improve trajectory tracking. This is consistent with the well-documented finding in the energy forecasting literature that macro regressors can improve mean-level calibration in stable periods but add noise — or, as here, introduce their own bias — in structurally uncertain environments; it also illustrates that Prophet's macro-augmented fit for a given series is not fully stable across independent training runs, which is itself a relevant caveat for treating any single FES_macro reading as definitive.

### RQ3: How Do Alternative FES Specifications Compare?

Equal_Core emerges as both the simplest and the best-performing FES variant against the dominant benchmark (Actual_CrossComp). The R² of 0.804 and Pearson r = 0.970 achieved by Equal_Core significantly outperform all other variants. Inverse-volatility weighting (VW_Core) and Kalman smoothing (Bayes_Core) improve the FES level in the sense of producing less extreme negative values (−0.328 and −0.281 versus −0.929), but they introduce level biases that yield near-zero or negative R² against actuals. The equal-weight specification avoids over-fitting the index weights to historical volatility patterns that may not persist into the forecast horizon.

The three actual benchmarks themselves also convey important methodological information. Actual_CrossComp (Option B, cross-component standard deviation) proves to be the most discriminating benchmark — it tracks the within-year stress dynamics most closely and produces the best R² against all forecast variants. Actual_RollingVol (Option A) is dominated by short-term noise and yields uniformly negative R² for all forecast variants. Actual_AbsShock (Option C) is the most generous benchmark by Pearson r but penalises all variants severely on R² owing to level mismatches.

In sum, the FES methodology is validated as directionally robust across all specifications (all variants agree that 2017 energy stress was below average), while the equal-weight, core-forecast variant provides the best empirical tracking of realised stress patterns. The macro-augmented and alternative-weighting variants serve their intended role as robustness checks, confirming the directional finding while revealing the sensitivity of level and temporal precision to modelling choices. These results set the stage for Chapter 5, which turns from the macro environment to the household level, using the 2017 FES context as the backdrop against which individual-level anticipatory vulnerability is assessed.

---

*Selected output references used in this chapter:*
- *Forecast series: `outputs/forecasts/gas_growth_pct_forecasts_all.csv`, `electricity_growth_pct_forecasts_all.csv`, `carbon_growth_pct_forecasts_all.csv`*
- *FES tables: `outputs/fes/fes_monthly_2017.csv`, `outputs/fes/fes_summary_2017.csv`, `outputs/fes/fes_components_table.csv`, `outputs/fes/fes_comparison_metrics.csv`*
- *Figures: `outputs/figures/forecast_vs_actual_gas.png`, `forecast_vs_actual_electricity.png`, `forecast_vs_actual_carbon.png`, `forecast_vs_actual_all_series.png`, `prediction_intervals_2017.png`, `fes_monthly_2017.png`, `fes_components_2017.png`, `fes_robustness_comparison.png`, `fes_bayesian_uncertainty.png`, `fes_metrics_rmse_heatmap.png`, `fes_metrics_pearson_r_heatmap.png`*
- *Model selection: `outputs/tables/model_metrics_comparison.csv`*
- *TS-SHAP: `outputs/shap/ts/gas_tft_core_attribution.csv`, `gas_prophet_macro_attribution.csv`, `carbon_lstm_core_attribution.csv`, `carbon_tft_macro_attribution.csv`, `electricity_lstm_macro_importance.csv`*
