# Findings Report: Anticipatory Fuel Stress Watch (rerun v2)

**Scope.** The v2 results for RQ1–RQ5, with every Chapter 4 and appendix table and every results figure. The data tables of Chapter 3 are in [`05_data_description.md`](05_data_description.md) and the methods in [`06_methodology.md`](06_methodology.md).
**Status.** Branch `rerun-v2`, analysis frozen 2026-09-26. v1 (tag `submitted-draft-v1`) is superseded. Every number below is a row of `outputs_v2/results_inventory.csv` or is read from the table shown next to it.
**Figures** are the thesis versions in `outputs_v2/thesis_assets_v2/figures/`. This is a local build (`python scripts/build_thesis_assets.py`) and is not tracked in git. **Tables** are rendered from the CSVs in `outputs_v2/thesis_assets_v2/tables/`, or from the tracked `outputs_v2/` file named in the caption.

---

## 1. Executive summary

| Hypothesis | Verdict | Status | Key statistics |
|---|---|---|---|
| H1 Resource moderation (COR buffering) | Not supported | author-confirmed | R × Delta = -0.000034 (p = 0.16); buffering ≤ 17% of slope |
| H2 FES independent predictor | Supported (small effect) | PROPOSED — author to confirm | FES Delta OR 0.972 [0.962, 0.981]; robust across sensitivities |
| H3 Household financial position dominates macro stress | Supported (reframed: current financial difficulty and employment security) | PROPOSED — author to confirm | financial difficulty OR 1.65 per point; employment security OR 0.17; FES OR per SD 0.93 |
| H4 External validation with JRF | Partially supported | PROPOSED — author to confirm | tenure ρ 0.80; family, work, disability same direction; region ρ -0.10 (excl. NI 0.18); ethnicity inconclusive |
| H5 Prospective prediction | Partially supported | author-confirmed | P1 AUC 0.739; benchmark P0 0.780; FES: no improvement |

*Table 4-10. Hypothesis verdicts. H1 and H5 wording is confirmed by the author; H2–H4 wording is proposed.*

1. **The outcome changed.** v1 dropped households routed out of the combined-bill question (`fuelduel` = −8), which removed almost every off-gas-grid household, including 84% of NI oil users. v1 also set non-response to £0. The corrected outcome has 286,902 household-waves (v1: 255,324) and a higher prevalence in every wave.
2. **Trend.** Weighted prevalence falls from 12.0% (wave a) to 6.5% (wave l) and rises to 12.5% (wave n, fieldwork 2022–24). The peak by interview year is 2023 (14.7%).
3. **Forecasts are weak.** No v2 series forecast significantly beats a no-change forecast. Prediction intervals cover 28–48% of outcomes, so the primary FES uses the three growth terms only.
4. **FES has a small, robust association with vulnerability** (H2): OR 0.972 per unit of FES Delta, 0.93 per SD, stable across all 11 specifications (0.968–0.974).
5. **Household finances dominate** (H3). Current financial difficulty (OR 1.65 per point) and employment-status security (OR 0.17) have far larger effects than FES.
6. **Resources do not buffer FES** (H1, not supported). The interaction is null, and a substantial buffering effect is ruled out.
7. **The JRF comparison is mixed** (H4). Tenure agrees; three two-group dimensions point in the same direction; regions do not agree; ethnicity is inconclusive. Northern Ireland has the highest fuel vulnerability but the lowest income poverty, and heating oil accounts for about two-thirds of its excess risk.
8. **Prediction** (H5, partially supported). The household model reaches AUC 0.739 on held-out waves, but the current fuel burden alone reaches 0.780, and FES adds nothing.

---

## 2. What changed from v1

The full list, with commits and the effect on each thesis claim, is in [`outputs_v2/reports/v1_to_v2_change_summary.md`](../outputs_v2/reports/v1_to_v2_change_summary.md). In brief:

| Area | v1 | v2 |
|---|---|---|
| Outcome | `fuelduel` = −8 read as missing (household dropped); non-response set to £0 | Routing-aware spend, non-response missing (complete-case); S1 and S2 bounds |
| Interview timing | Sample (issue) month | Actual household interview date |
| FES | Macro variant chosen by hindsight; tuned once on 2024; forecast for Y+1 given to year-Y interviews | Core only, tuned per origin, Dec Y−1 vintage, past-only z-scores, growth-only primary |
| Weights | Unweighted | Per-wave household cross-sectional weights |
| Strain | 4-item composite (α 0.27) with bill arrears counted twice | Components entered separately |
| Driver model | No year effects, unclustered, N = 103,621 | Year FE, PSU clustering, region FE and oil in the NI sequence, N = 221,877 |
| Resources | Second-order SEM (degenerate) | CFA failed pre-set criteria → formative composites |
| JRF | Pooled 2009–2024, unweighted | Time-matched windows, weighted, bootstrap CIs |
| Prediction | One model, AUC only | P0 benchmark, P1, P2 on one common sample; calibration, PR-AUC, top-k |
| Disability | `healthlink` (a record-linkage consent flag, wave a only) | Long-standing illness plus any substantial difficulty (`disdif`) |

---

## 3. Forecasts and FES (RQ1)

### 3.1 Accuracy against benchmarks

![Figure 4-1. Relative RMSE by target year](../outputs_v2/thesis_assets_v2/figures/fig4-1_relrmse_by_year.png)

*Figure 4-1. v2 forecast accuracy relative to naive and seasonal-naive benchmarks, by target year.*

| Series | Version | RMSE | Rel. RMSE vs naive | DM p vs naive | Rel. RMSE vs seasonal naive | DM p vs seasonal naive | 95% PI coverage (%) |
|---|---|---|---|---|---|---|---|
| carbon | v1_core | 48.29 | 1.028 | 0.845 | 0.788 | 0.054 | 28.6 |
| carbon | v1_macro | 55.47 | 1.180 | 0.403 | 0.905 | 0.551 | 37.5 |
| carbon | v2_core | 46.47 | 0.989 | 0.905 | 0.758 | 0.002 | 27.6 |
| electricity | v1_core | 16.99 | 0.973 | 0.899 | 0.740 | 0.200 | 60.4 |
| electricity | v1_macro | 13.97 | 0.800 | 0.458 | 0.609 | 0.140 | 49.5 |
| electricity | v2_core | 14.41 | 0.825 | 0.526 | 0.627 | 0.143 | 48.4 |
| gas | v1_core | 43.32 | 1.228 | 0.172 | 0.956 | 0.710 | 56.2 |
| gas | v1_macro | 55.79 | 1.582 | 0.202 | 1.231 | 0.424 | 68.8 |
| gas | v2_core | 25.11 | 0.712 | 0.373 | 0.554 | 0.130 | 32.3 |

*Table 4-1. Forecast accuracy on the three growth terms (192 target months per series, 2010–2025) and 95% prediction-interval coverage. v1_macro is the variant v1 attached to households.*

- **Against the naive (no-change) forecast, no version is significantly better** (all Diebold–Mariano p ≥ 0.17). The v2 point estimates are below 1 for gas (0.71) and electricity (0.83) and about 1 for carbon (0.99).
- **The gas gain is concentrated in 2023.** Relative RMSE was 0.30 that year, but v2 beats naive in only 6 of 16 target years (Table A-5; `forecast_accuracy_pooled.csv`, `pct_years_beating_naive` = 37.5%).
- **Against seasonal naive**, v2 is better for all three series, significantly so only for carbon (p = 0.002).
- **The v1 variant attached to households (v1_macro) was worse than naive for gas** (relative RMSE 1.58).
- **Prediction intervals are far too narrow.** v2 coverage is 27.6% (carbon), 32.3% (gas) and 48.4% (electricity) against a nominal 95%. This is why the uncertainty term was removed from the primary FES (deviation log, 2026-09-26).

| Version | Series | RMSE | MAE | Sign agreement (%) | Pearson r | Rel. RMSE vs naive | Rel. RMSE vs s-naive | MASE | Years beating naive (%) |
|---|---|---|---|---|---|---|---|---|---|
| naive | carbon | 46.99 | 38.44 | 60.4 | 0.447 | 1.000 | 0.767 | 1.020 |  |
| naive | electricity | 17.46 | 9.19 | 76.6 | 0.454 | 1.000 | 0.760 | 5.693 |  |
| naive | gas | 35.27 | 15.74 | 75.5 | 0.433 | 1.000 | 0.778 | 6.326 |  |
| snaive | carbon | 61.27 | 51.78 | 51.0 | 0.158 | 1.304 | 1.000 | 1.390 | 25.0 |
| snaive | electricity | 22.96 | 14.46 | 50.5 | -0.033 | 1.315 | 1.000 | 8.684 | 0.0 |
| snaive | gas | 45.31 | 26.05 | 51.6 | -0.053 | 1.285 | 1.000 | 9.817 | 12.5 |
| v1_core | carbon | 48.29 | 36.43 | 58.3 | 0.294 | 1.028 | 0.788 | 0.914 | 43.8 |
| v1_core | electricity | 16.99 | 11.22 | 60.4 | 0.262 | 0.973 | 0.740 | 6.749 | 43.8 |
| v1_core | gas | 43.32 | 20.41 | 60.9 | 0.272 | 1.228 | 0.956 | 7.970 | 31.2 |
| v1_macro | carbon | 55.47 | 38.89 | 54.7 | 0.188 | 1.180 | 0.905 | 0.940 | 75.0 |
| v1_macro | electricity | 13.97 | 8.60 | 57.8 | 0.545 | 0.800 | 0.609 | 5.291 | 50.0 |
| v1_macro | gas | 55.79 | 24.75 | 62.5 | 0.190 | 1.582 | 1.231 | 9.555 | 31.2 |
| v2_core | carbon | 46.47 | 37.16 | 57.8 | 0.337 | 0.989 | 0.758 | 0.960 | 50.0 |
| v2_core | electricity | 14.41 | 9.06 | 62.0 | 0.448 | 0.825 | 0.627 | 5.556 | 43.8 |
| v2_core | gas | 25.11 | 13.50 | 71.4 | 0.609 | 0.712 | 0.554 | 5.419 | 37.5 |

*Pooled accuracy, all versions and benchmarks (`outputs_v2/fes_eval/forecast_accuracy_pooled.csv`). MASE scales by the in-sample one-step naive error, so values above 1 are expected at 1–12-month horizons; compare across rows, not against 1.*

| Version | Series | Months | 95% PI coverage (%) | Mean PI width | Median PI width |
|---|---|---|---|---|---|
| v1_core | carbon | 192 | 28.6 | 77.53 | 17.29 |
| v1_core | electricity | 192 | 60.4 | 22.14 | 17.60 |
| v1_core | gas | 192 | 56.2 | 27.02 | 13.46 |
| v1_macro | carbon | 192 | 37.5 | 71.85 | 18.11 |
| v1_macro | electricity | 192 | 49.5 | 12.75 | 10.77 |
| v1_macro | gas | 192 | 68.8 | 33.78 | 21.52 |
| v2_core | carbon | 192 | 27.6 | 45.78 | 17.20 |
| v2_core | electricity | 192 | 48.4 | 13.54 | 6.72 |
| v2_core | gas | 192 | 32.3 | 11.27 | 7.34 |

*Prediction-interval coverage and width (`outputs_v2/fes_eval/uncertainty_pi.csv`).*

### 3.2 Which model wins

![Figure A-1. Winning model by series and target year](../outputs_v2/thesis_assets_v2/figures/figA_radial_winners.png)

*Figure A-1. Winning rolling core model by series and target year (v2), read clockwise from 12 o'clock.*

| Version | Series | LSTM | Prophet | SARIMA | TFT |
|---|---|---|---|---|---|
| v1_core | carbon | 13 | 2 | 0 | 1 |
| v1_core | electricity | 7 | 1 | 3 | 5 |
| v1_core | gas | 10 | 1 | 2 | 3 |
| v1_macro | carbon | 12 | 1 | 1 | 2 |
| v1_macro | electricity | 6 | 4 | 5 | 1 |
| v1_macro | gas | 2 | 2 | 7 | 5 |
| v2_core | carbon | 14 | 1 | 0 | 1 |
| v2_core | electricity | 9 | 4 | 2 | 1 |
| v2_core | gas | 13 | 1 | 1 | 1 |

*Number of forecast origins won by each model (`outputs_v2/fes_eval/model_wins.csv`). In v2, the LSTM wins 36 of 48 series-origins.*

![Winning core model and validation RMSE, polar view](../outputs_v2/figures/model_selection_polar_core.png)

*Pipeline figure `outputs_v2/figures/model_selection_polar_core.png`: bar = winning model's validation RMSE for that target year, colour = winning model.*

| Gas | Electricity | Carbon |
|---|---|---|
| ![gas](../outputs_v2/figures/rolling_forecast_performance_gas.png) | ![electricity](../outputs_v2/figures/rolling_forecast_performance_electricity.png) | ![carbon](../outputs_v2/figures/rolling_forecast_performance_carbon.png) |

*Pipeline figures `outputs_v2/figures/rolling_forecast_performance_{gas,electricity,carbon}.png`: the winning model's validation RMSE by target year, points coloured by model.*

### 3.3 The FES signal attached to households

![Figure 4-2. Growth-only FES forecast vs realised](../outputs_v2/thesis_assets_v2/figures/fig4-2_fes_growth3_forecast_vs_realised.png)

*Figure 4-2. Growth-only FES: forecast vs realised by target month, 2010–2025.*

![Rolling FES by target year](../outputs_v2/figures/fes_rolling_trend.png)

*Pipeline figure `outputs_v2/figures/fes_rolling_trend.png`: annual mean of the 4-term rolling FES (forecast) against the realised index. The forecast misses the 2022 shock: realised 15.4, forecast 2.1.*

The FES attached to households differs substantially from v1's. The v2 FES Delta correlates 0.16 with v1's, because v1 gave year-Y interviews a forecast that used data up to December Y (look-ahead) and chose the macro variant by hindsight.

| Variable | n (both) | r (v1, v2) | Mean v1 | Mean v2 | SD v1 | SD v2 |
|---|---|---|---|---|---|---|
| `fes_magnitude` | 324,055 | 0.554 | -0.185 | -0.984 | 2.143 | 1.639 |
| `fes_current` | 324,055 | 0.914 | -0.090 | -0.162 | 2.002 | 3.136 |
| `fes_delta` | 324,055 | 0.163 | -0.095 | -0.823 | 1.625 | 2.597 |

*FES attached to households, v1 vs v2 (`outputs_v2/fes_eval/fes_attached_v1_vs_v2.csv`).*

<details><summary><b>Table A-1. MASE</b></summary>

| Series | Version | MAE | MASE |
|---|---|---|---|
| carbon | naive | 38.44 | 1.020 |
| carbon | snaive | 51.78 | 1.390 |
| carbon | v1_core | 36.43 | 0.914 |
| carbon | v1_macro | 38.89 | 0.940 |
| carbon | v2_core | 37.16 | 0.960 |
| electricity | naive | 9.19 | 5.693 |
| electricity | snaive | 14.46 | 8.684 |
| electricity | v1_core | 11.22 | 6.749 |
| electricity | v1_macro | 8.60 | 5.291 |
| electricity | v2_core | 9.06 | 5.556 |
| gas | naive | 15.74 | 6.326 |
| gas | snaive | 26.05 | 9.817 |
| gas | v1_core | 20.41 | 7.970 |
| gas | v1_macro | 24.75 | 9.555 |
| gas | v2_core | 13.50 | 5.419 |

</details>

<details><summary><b>Table A-5. Relative RMSE by target year</b></summary>

| Series | Target year | v1 core vs naive | v1 core vs s-naive | v1 macro vs naive | v1 macro vs s-naive | v2 core vs naive | v2 core vs s-naive |
|---|---|---|---|---|---|---|---|
| carbon | 2010 | 1.157 | 0.605 | 0.764 | 0.400 | 1.159 | 0.607 |
| carbon | 2011 | 2.588 | 2.721 | 3.265 | 3.432 | 0.905 | 0.952 |
| carbon | 2012 | 0.509 | 0.241 | 0.752 | 0.356 | 1.424 | 0.674 |
| carbon | 2013 | 0.864 | 1.664 | 1.338 | 2.576 | 1.091 | 2.100 |
| carbon | 2014 | 0.838 | 0.584 | 0.994 | 0.692 | 0.901 | 0.628 |
| carbon | 2015 | 1.070 | 0.829 | 0.866 | 0.672 | 0.567 | 0.440 |
| carbon | 2016 | 1.104 | 0.894 | 0.991 | 0.803 | 0.982 | 0.796 |
| carbon | 2017 | 0.645 | 0.443 | 0.807 | 0.554 | 1.475 | 1.014 |
| carbon | 2018 | 1.247 | 1.091 | 1.576 | 1.379 | 1.289 | 1.128 |
| carbon | 2019 | 0.398 | 0.412 | 0.182 | 0.189 | 0.543 | 0.562 |
| carbon | 2020 | 1.227 | 0.344 | 0.904 | 0.254 | 1.118 | 0.314 |
| carbon | 2021 | 1.000 | 0.640 | 0.522 | 0.334 | 0.710 | 0.454 |
| carbon | 2022 | 0.395 | 0.398 | 0.474 | 0.477 | 0.576 | 0.580 |
| carbon | 2023 | 1.352 | 0.346 | 1.122 | 0.287 | 1.126 | 0.288 |
| carbon | 2024 | 1.144 | 0.827 | 0.752 | 0.543 | 1.247 | 0.901 |
| carbon | 2025 | 0.658 | 0.360 | 0.978 | 0.535 | 0.571 | 0.313 |
| electricity | 2010 | 1.847 | 0.721 | 1.148 | 0.448 | 1.086 | 0.424 |
| electricity | 2011 | 1.038 | 0.813 | 1.093 | 0.856 | 1.046 | 0.819 |
| electricity | 2012 | 0.338 | 0.328 | 0.649 | 0.631 | 0.701 | 0.681 |
| electricity | 2013 | 0.481 | 0.319 | 0.724 | 0.480 | 1.952 | 1.295 |
| electricity | 2014 | 0.962 | 0.941 | 1.162 | 1.137 | 0.623 | 0.610 |
| electricity | 2015 | 12.015 | 0.862 | 11.426 | 0.819 | 9.804 | 0.703 |
| electricity | 2016 | 42.620 | 15.713 | 44.338 | 16.346 | 16.869 | 6.219 |
| electricity | 2017 | 0.830 | 0.828 | 0.897 | 0.896 | 1.170 | 1.168 |
| electricity | 2018 | 0.741 | 0.417 | 0.631 | 0.355 | 0.870 | 0.490 |
| electricity | 2019 | 1.034 | 0.789 | 0.913 | 0.697 | 0.921 | 0.703 |
| electricity | 2020 | 1.629 | 0.894 | 1.645 | 0.903 | 1.629 | 0.894 |
| electricity | 2021 | 0.791 | 0.737 | 0.909 | 0.846 | 0.892 | 0.831 |
| electricity | 2022 | 1.279 | 1.016 | 1.086 | 0.863 | 1.221 | 0.970 |
| electricity | 2023 | 0.552 | 0.550 | 0.563 | 0.561 | 0.333 | 0.332 |
| electricity | 2024 | 4.225 | 0.537 | 2.368 | 0.301 | 4.285 | 0.545 |
| electricity | 2025 | 2.513 | 1.296 | 0.805 | 0.415 | 0.678 | 0.350 |
| gas | 2010 | 10.387 | 0.572 | 7.159 | 0.395 | 1.438 | 0.079 |
| gas | 2011 | 1.392 | 1.128 | 1.175 | 0.952 | 1.232 | 0.998 |
| gas | 2012 | 0.943 | 0.646 | 1.476 | 1.011 | 0.498 | 0.341 |
| gas | 2013 | 1.623 | 0.543 | 0.745 | 0.249 | 4.146 | 1.386 |
| gas | 2014 | 0.846 | 0.817 | 0.847 | 0.818 | 0.913 | 0.882 |
| gas | 2015 | 2.879 | 1.008 | 2.896 | 1.014 | 3.190 | 1.117 |
| gas | 2016 | 3.377 | 1.318 | 4.876 | 1.904 | 3.377 | 1.318 |
| gas | 2017 | 1.075 | 0.819 | 0.585 | 0.446 | 1.052 | 0.802 |
| gas | 2018 | 0.410 | 0.359 | 1.155 | 1.012 | 0.536 | 0.470 |
| gas | 2019 | 0.873 | 0.980 | 0.657 | 0.738 | 0.569 | 0.639 |
| gas | 2020 | 1.700 | 0.814 | 1.503 | 0.720 | 1.463 | 0.701 |
| gas | 2021 | 1.024 | 1.002 | 0.768 | 0.752 | 1.029 | 1.007 |
| gas | 2022 | 1.143 | 0.888 | 1.215 | 0.944 | 1.178 | 0.915 |
| gas | 2023 | 1.259 | 1.286 | 1.707 | 1.743 | 0.298 | 0.304 |
| gas | 2024 | 1.561 | 0.230 | 2.171 | 0.321 | 1.507 | 0.223 |
| gas | 2025 | 0.747 | 0.355 | 2.440 | 1.160 | 0.490 | 0.233 |

</details>

<details><summary><b>Table A-6. Diebold–Mariano tests (squared-error loss, Newey–West lag 11, HLN correction)</b></summary>

| Version | Series | Benchmark | Months | DM statistic | p | Verdict |
|---|---|---|---|---|---|---|
| v1_core | carbon | naive | 192 | 0.196 | 0.845 | not significantly different |
| v1_core | carbon | seasonal naive | 192 | -1.939 | 0.054 | not significantly different |
| v1_core | electricity | naive | 192 | -0.127 | 0.899 | not significantly different |
| v1_core | electricity | seasonal naive | 192 | -1.286 | 0.200 | not significantly different |
| v1_core | gas | naive | 192 | 1.369 | 0.173 | not significantly different |
| v1_core | gas | seasonal naive | 192 | -0.372 | 0.710 | not significantly different |
| v1_macro | carbon | naive | 192 | 0.838 | 0.403 | not significantly different |
| v1_macro | carbon | seasonal naive | 192 | -0.597 | 0.551 | not significantly different |
| v1_macro | electricity | naive | 192 | -0.743 | 0.458 | not significantly different |
| v1_macro | electricity | seasonal naive | 192 | -1.483 | 0.140 | not significantly different |
| v1_macro | gas | naive | 192 | 1.279 | 0.202 | not significantly different |
| v1_macro | gas | seasonal naive | 192 | 0.800 | 0.425 | not significantly different |
| v2_core | carbon | naive | 192 | -0.119 | 0.905 | not significantly different |
| v2_core | carbon | seasonal naive | 192 | -3.185 | 0.002 | better than benchmark |
| v2_core | electricity | naive | 192 | -0.635 | 0.526 | not significantly different |
| v2_core | electricity | seasonal naive | 192 | -1.469 | 0.143 | not significantly different |
| v2_core | gas | naive | 192 | -0.893 | 0.373 | not significantly different |
| v2_core | gas | seasonal naive | 192 | -1.521 | 0.130 | not significantly different |

</details>

---

## 4. How many households are fuel vulnerable, and who (RQ4)

### 4.1 Trend

![Figure 4-3. National trend by wave with S1 band](../outputs_v2/thesis_assets_v2/figures/fig4-3_trend_wave_s1band.png)

*Figure 4-3. National trend by wave, weighted, with the S1 lower bound.*

| Wave | Fieldwork | Primary n | Primary % (weighted) | S1 lower bound % (weighted) | S2 % (weighted) | v1 rule % (weighted) | Primary % (unweighted) |
|---|---|---|---|---|---|---|---|
| a | 2009–2011 | 25,649 | 12.0 | 10.8 | 11.9 | 10.8 | 12.8 |
| b | 2010–2012 | 26,936 | 9.4 | 8.6 | 9.3 | 8.5 | 10.0 |
| c | 2011–2013 | 24,765 | 9.2 | 8.5 | 9.2 | 8.4 | 9.7 |
| d | 2012–2014 | 23,260 | 9.5 | 8.8 | 9.4 | 8.6 | 9.9 |
| e | 2013–2015 | 21,970 | 10.0 | 9.3 | 9.9 | 9.2 | 10.0 |
| f | 2014–2016 | 20,625 | 8.4 | 7.5 | 8.3 | 7.6 | 8.5 |
| g | 2015–2017 | 19,814 | 6.8 | 6.2 | 6.8 | 6.3 | 7.1 |
| h | 2016–2018 | 18,818 | 6.5 | 5.8 | 6.4 | 6.0 | 6.4 |
| i | 2017–2019 | 17,004 | 6.3 | 5.6 | 6.2 | 5.7 | 6.1 |
| j | 2018–2020 | 16,171 | 6.7 | 5.9 | 6.6 | 6.0 | 6.6 |
| k | 2019–2021 | 14,884 | 6.5 | 5.6 | 6.4 | 5.5 | 6.2 |
| l | 2020–2022 | 13,431 | 6.5 | 5.6 | 6.4 | 5.7 | 5.9 |
| m | 2021–2023 | 12,478 | 7.6 | 6.3 | 7.5 | 6.7 | 7.2 |
| n | 2022–2024 | 16,411 | 12.5 | 10.2 | 12.3 | 11.4 | 12.0 |
| o | 2023–2025 | 14,686 | 12.4 | 9.9 | 12.2 | 11.4 | 11.9 |

*Prevalence by wave, all outcome definitions (`outputs_v2/descriptives/prevalence_by_wave.csv`). Waves span two to three calendar years; say "wave n (fieldwork 2022–24)", not "2022".*

![Figure 4-4. Trend by interview year with the JRF crisis window](../outputs_v2/thesis_assets_v2/figures/fig4-4_trend_vs_jrf_tracker.png)

*Figure 4-4. Trend by interview year, with the JRF cost-of-living crisis window shaded.*

![Figure A-3. Trend by interview year, supplementary](../outputs_v2/thesis_assets_v2/figures/figA_trend_interview_year.png)

*Figure A-3. Trend by interview year (supplementary). 2025 has only 223 households with an observed outcome.*

| Interview year | Waves | Primary n | Primary % (weighted) | S1 lower bound % (weighted) |
|---|---|---|---|---|
| 2009 | a | 12,774 | 12.4 | 11.2 |
| 2010 | a,b,c | 27,594 | 10.3 | 9.4 |
| 2011 | a,b,c | 26,339 | 9.5 | 8.7 |
| 2012 | b,c,d | 23,704 | 9.1 | 8.4 |
| 2013 | c,d,e | 22,178 | 10.2 | 9.5 |
| 2014 | d,e,f | 20,267 | 9.3 | 8.5 |
| 2015 | e,f,g | 19,855 | 7.8 | 7.0 |
| 2016 | f,g,h | 19,704 | 6.1 | 5.5 |
| 2017 | g,h,i | 17,849 | 6.3 | 5.6 |
| 2018 | h,i,j | 16,830 | 6.4 | 5.7 |
| 2019 | i,j,k | 15,829 | 6.5 | 5.6 |
| 2020 | j,k,l | 14,693 | 6.6 | 5.7 |
| 2021 | k,l,m | 12,933 | 6.2 | 5.2 |
| 2022 | l,m,n | 13,533 | 9.3 | 7.7 |
| 2023 | m,n,o | 15,705 | 14.7 | 11.8 |
| 2024 | n,o | 6,892 | 11.5 | 9.2 |
| 2025 | o | 223 | 15.7 | 12.8 |

*Prevalence by interview year (`outputs_v2/descriptives/prevalence_by_interview_year.csv`).*

The trend falls from the start of the panel to a low around 2016–2021 and rises sharply in 2022–2023, the period of the energy-price crisis. The S1 lower bound is 1–3 points lower but has the same shape. The same shapes appear in the tracked pipeline figures `outputs_v2/descriptives/trend_primary_with_s1_band.png` and `trend_by_interview_year_supplementary.png`:

| By wave | By interview year |
|---|---|
| ![](../outputs_v2/descriptives/trend_primary_with_s1_band.png) | ![](../outputs_v2/descriptives/trend_by_interview_year_supplementary.png) |

### 4.2 Regions

![Figure 4-5. Regional prevalence map](../outputs_v2/thesis_assets_v2/figures/fig4-5_region_map_weighted.png)

*Figure 4-5. Regional prevalence, weighted (pooled mean of per-wave weighted rates).*

![Figure 4-6. Region × interview-year heatmap](../outputs_v2/thesis_assets_v2/figures/fig4-6_region_year_heatmap_masked.png)

*Figure 4-6. Weighted prevalence by region and interview year (cells with n < 100 masked).*

![Figure 4-7. Change in regional prevalence](../outputs_v2/thesis_assets_v2/figures/fig4-7_region_change_map.png)

*Figure 4-7. Change in weighted regional prevalence, waves a–e to k–o (navy = fall, orange = rise).*

| Region | n, waves a–e | n, waves k–o | %, waves a–e | %, waves k–o | Change (pp) |
|---|---|---|---|---|---|
| Northern Ireland | 7,848 | 3,923 | 23.9 | 15.0 | -8.9 |
| Wales | 8,421 | 4,621 | 12.9 | 10.1 | -2.8 |
| North West | 12,718 | 7,258 | 11.2 | 9.0 | -2.2 |
| North East | 4,899 | 2,760 | 11.2 | 9.1 | -2.1 |
| West Midlands | 9,988 | 5,961 | 11.3 | 10.0 | -1.3 |
| Scotland | 10,750 | 7,035 | 11.7 | 10.8 | -0.9 |
| Yorkshire and the Humber | 9,841 | 6,159 | 10.3 | 9.6 | -0.6 |
| South West | 9,705 | 6,340 | 8.1 | 7.9 | -0.2 |
| East Midlands | 9,082 | 5,362 | 10.5 | 10.4 | -0.1 |
| East of England | 10,394 | 6,304 | 8.0 | 8.3 | 0.2 |
| South East | 14,178 | 8,930 | 7.0 | 7.2 | 0.3 |
| London | 14,699 | 7,213 | 7.4 | 7.7 | 0.3 |

*Regional change, waves a–e vs k–o (`outputs_v2/stage7/regional_change_early_late.csv`). NI shows the largest fall (−8.9 pp) but remains the highest region.*

### 4.3 Social groups

![Figure 4-9. Social groups with CIs](../outputs_v2/thesis_assets_v2/figures/fig4-9_social_groups_panel_ci.png)

*Figure 4-9. Pooled weighted prevalence by social group, with 95% CIs (categories with n < 100 suppressed).*

| Dimension | Category | n | Weighted % [95% CI] |
|---|---|---|---|
| region | East Midlands | 21,104 | 9.4 [8.5, 10.2] |
| region | East of England | 24,393 | 7.3 [6.6, 8.0] |
| region | London | 33,423 | 6.7 [6.1, 7.3] |
| region | North East | 11,085 | 9.0 [8.0, 9.9] |
| region | North West | 29,472 | 9.1 [8.4, 9.7] |
| region | Northern Ireland | 16,646 | 18.0 [16.8, 19.4] |
| region | Scotland | 25,951 | 10.3 [9.5, 11.1] |
| region | South East | 33,992 | 6.6 [6.1, 7.1] |
| region | South West | 23,661 | 7.4 [6.8, 8.1] |
| region | Wales | 19,274 | 10.4 [9.5, 11.3] |
| region | West Midlands | 23,778 | 9.9 [9.0, 10.7] |
| region | Yorkshire and the Humber | 23,905 | 9.1 [8.3, 9.9] |
| tenure | Buying with mortgage | 98,150 | 4.2 [4.0, 4.5] |
| tenure | Owned outright | 101,166 | 11.1 [10.8, 11.6] |
| tenure | Private renting | 35,200 | 8.9 [8.3, 9.5] |
| tenure | Social renting | 50,611 | 11.1 [10.6, 11.6] |
| family_composition_group | Couple, 1-2 children | 47,733 | 3.8 [3.5, 4.0] |
| family_composition_group | Couple, 3+ children | 9,748 | 5.2 [4.5, 6.1] |
| family_composition_group | Lone parent, 1-2 children | 14,579 | 15.1 [14.1, 16.2] |
| family_composition_group | Lone parent, 3+ children | 1,929 | 13.6 [11.2, 16.4] |
| family_composition_group | No children | 192,368 | 9.7 [9.5, 10.0] |
| family_composition_group | Other multi-adult, with children | 20,374 | 3.8 [3.5, 4.2] |
| employment_group | Full-time or self-employed | 151,626 | 3.5 [3.3, 3.6] |
| employment_group | Part-time only | 29,607 | 10.0 [9.5, 10.6] |
| employment_group | Workless household | 104,093 | 15.0 [14.6, 15.5] |
| ethnicity_group | Any other Asian background | 2,144 | 6.2 [4.6, 8.0] |
| ethnicity_group | Any other Black background | 349 | 14.6 [9.5, 19.9] |
| ethnicity_group | Bangladeshi | 3,246 | 9.3 [7.5, 11.3] |
| ethnicity_group | Black African | 5,508 | 11.2 [9.7, 12.7] |
| ethnicity_group | Black Caribbean | 5,615 | 14.7 [12.6, 17.2] |
| ethnicity_group | Chinese | 1,099 | 7.5 [4.6, 11.3] |
| ethnicity_group | Indian | 7,646 | 8.0 [6.8, 9.5] |
| ethnicity_group | Mixed/multiple ethnic groups | 3,989 | 9.9 [8.0, 11.9] |
| ethnicity_group | Other ethnic group | 1,872 | 10.7 [7.2, 15.0] |
| ethnicity_group | Pakistani | 5,584 | 13.8 [12.2, 15.5] |
| ethnicity_group | White | 229,098 | 8.8 [8.6, 9.1] |
| disability | Contains disabled adult | 98,015 | 10.5 [10.2, 10.9] |
| disability | No disabled adult | 186,994 | 7.6 [7.4, 7.9] |

*Table 4-2. Pooled weighted prevalence by region and social group, with 95% PSU-bootstrap CIs.*

The largest gradients are by employment (workless 15.0% vs full-time or self-employed 3.5%), by family type (lone parents with 1–2 children 15.1% vs couples with 1–2 children 3.8%) and by tenure (outright owners 11.1% and social renters 11.1% vs mortgage holders 4.2%). Households with a disabled adult are at 10.5% against 7.6%. Among ethnic groups, Black Caribbean (14.7%) and Pakistani (13.8%) households have the highest pooled rates.

### 4.4 Resources and financial difficulty by region

![Figure 4-8. Resource composite and financial difficulty by region](../outputs_v2/thesis_assets_v2/figures/fig4-8_resource_and_findifficulty_by_region.png)

*Figure 4-8. Weighted mean resource composite and current financial difficulty by region.*

![Resource composite by region, pipeline figure](../outputs_v2/resources/figures/resource_composite_by_region.png)

*Pipeline figure `outputs_v2/resources/figures/resource_composite_by_region.png`.*

| Region | n | R (primary) | OBJECT | CONDITION | PERSONAL | ENERGY |
|---|---|---|---|---|---|---|
| East Midlands | 23,946 | -0.17 | -0.04 | 0.01 | -0.14 | -0.05 |
| East of England | 28,023 | 0.08 | 0.08 | 0.04 | -0.04 | 0.06 |
| London | 39,144 | -0.32 | -0.31 | -0.21 | 0.20 | 0.03 |
| North East | 12,435 | -0.63 | -0.29 | -0.10 | -0.24 | -0.20 |
| North West | 33,570 | -0.33 | -0.13 | -0.05 | -0.15 | -0.09 |
| Northern Ireland | 20,912 | -0.24 | -0.02 | -0.07 | -0.16 | -0.25 |
| Scotland | 29,999 | -0.37 | -0.31 | 0.00 | -0.06 | -0.09 |
| South East | 39,541 | 0.26 | 0.16 | 0.06 | 0.04 | 0.17 |
| South West | 26,895 | 0.02 | 0.05 | 0.04 | -0.06 | 0.09 |
| Wales | 22,228 | -0.27 | -0.04 | -0.01 | -0.21 | -0.13 |
| West Midlands | 26,860 | -0.17 | -0.02 | -0.03 | -0.13 | -0.03 |
| Yorkshire and the Humber | 27,047 | -0.38 | -0.18 | -0.06 | -0.13 | -0.12 |

*Weighted mean domain composites by region (`outputs_v2/resources/resource_by_region.csv`; composites are standardised, 0 = sample mean). The South East is highest and the North East lowest on the primary composite R.*

### 4.5 Prepayment meters

![Figure 4-10. Prepayment by vulnerability](../outputs_v2/thesis_assets_v2/figures/fig4-10_prepayment.png)

*Figure 4-10. Prepayment-meter use by vulnerability status.*

| Status | n | Prepayment meter, weighted (%) | Prepayment meter, unweighted (%) |
|---|---|---|---|
| not vulnerable (<10%) | 214,249 | 13.6 | 13.1 |
| vulnerable (>=10%) | 19,686 | 24.4 | 24.7 |

*`outputs_v2/stage7/prepayment_by_vulnerability.csv`. Prepayment use is higher among vulnerable households (24.4% vs 13.6%). Households that ration energy on a prepayment meter can fall below the 10% line, so the ratio measure can miss some hardship.*

---

## 5. Drivers of fuel vulnerability (RQ3; H2, H3)

![Figure 4-11. Driver-model forest plot](../outputs_v2/thesis_assets_v2/figures/fig4-11_driver_forest.png)

*Figure 4-11. Primary driver model, odds ratios (FES in orange). Pipeline version: `outputs_v2/stage3/figures/primary_or_forest.png`.*

![Primary driver model forest plot, pipeline version](../outputs_v2/stage3/figures/primary_or_forest.png)

| Predictor | Term | OR [95% CI] | p | OR per SD | SD in sample |
|---|---|---|---|---|---|
| Current financial difficulty (1-5) | `finnow` | 1.649 [1.607, 1.691] | <0.001 | 1.595 | 0.934 |
| Bedrooms | `hsbeds` | 1.335 [1.285, 1.386] | <0.001 | 1.341 | 1.015 |
| Tenure security | `tenure_security` | 2.034 [1.767, 2.340] | <0.001 | 1.186 | 0.240 |
| Rooms | `hsrooms` | 1.149 [1.106, 1.195] | <0.001 | 1.158 | 1.054 |
| No long-standing illness/disability | `health_good` | 1.253 [1.185, 1.325] | <0.001 | 1.097 | 0.412 |
| Age (mean of adults) | `dvage` | 1.003 [1.001, 1.005] | 0.013 | 1.050 | 16.703 |
| Self-rated general health | `sf1_good` | 1.026 [0.918, 1.147] | 0.649 | 1.006 | 0.235 |
| Psychological distress, GHQ-12 (0-36) | `scghq1_dv` | 0.991 [0.987, 0.995] | <0.001 | 0.956 | 4.976 |
| Financial expectations: worse off (0-1) | `finfut_risk` | 0.818 [0.762, 0.879] | <0.001 | 0.946 | 0.278 |
| FES Delta (growth-only) | `fes_delta_growth3` | 0.972 [0.962, 0.981] | <0.001 | 0.932 | 2.459 |
| Bill-payment security | `bill_security` | 0.587 [0.518, 0.664] | <0.001 | 0.931 | 0.134 |
| Cars | `ncars` | 0.856 [0.824, 0.889] | <0.001 | 0.848 | 1.057 |
| Highest qualification band | `qfhigh_band` | 0.548 [0.509, 0.590] | <0.001 | 0.802 | 0.368 |
| Employment-status security | `jbstat_security` | 0.174 [0.154, 0.197] | <0.001 | 0.630 | 0.264 |
| OECD equivalence scale | `ieqmoecd_dv` | 0.275 [0.255, 0.297] | <0.001 | 0.468 | 0.589 |
| Has central heating | `heatch` | 1.087 [1.013, 1.167] | 0.020 |  |  |
| Lone-parent household | `lone_parent` | 1.499 [1.379, 1.629] | <0.001 |  |  |
| Large family (3+ children) | `large_family` | 1.963 [1.738, 2.218] | <0.001 |  |  |
| Workless household | `workless_household` | 1.046 [0.957, 1.144] | 0.318 |  |  |

*Table 4-3. Primary driver model: logit of `high_fuel_vulnerable`, interview-year fixed effects, SEs clustered on PSU (8,801 PSUs); n = 221,877 household-waves, interviews 2010–2025; 17,209 events; McFadden R² 0.153.*

| Predictor | SD | OR per SD [95% CI] | log OR per SD | p | Note |
|---|---|---|---|---|---|
| OECD equivalence scale | 0.589 | 0.468 [0.447, 0.489] | -0.760 | <0.001 | partly mechanical: outcome uses unequivalised income, so larger households have more income per fuel need |
| Current financial difficulty (1-5) | 0.934 | 1.595 [1.557, 1.633] | 0.467 | <0.001 |  |
| Employment-status security | 0.264 | 0.630 [0.609, 0.651] | -0.462 | <0.001 |  |
| Bedrooms | 1.015 | 1.341 [1.290, 1.393] | 0.293 | <0.001 |  |
| Highest qualification band | 0.368 | 0.802 [0.780, 0.824] | -0.221 | <0.001 |  |
| Tenure security | 0.240 | 1.186 [1.147, 1.227] | 0.171 | <0.001 |  |
| Cars | 1.057 | 0.848 [0.815, 0.883] | -0.164 | <0.001 |  |
| Rooms | 1.054 | 1.158 [1.112, 1.206] | 0.147 | <0.001 |  |
| No long-standing illness/disability | 0.412 | 1.097 [1.072, 1.123] | 0.093 | <0.001 |  |
| Bill-payment security | 0.134 | 0.931 [0.916, 0.947] | -0.071 | <0.001 |  |
| FES Delta (growth-only) | 2.459 | 0.932 [0.910, 0.954] | -0.071 | <0.001 |  |
| Financial expectations: worse off (0-1) | 0.278 | 0.946 [0.927, 0.965] | -0.056 | <0.001 |  |
| Age (mean of adults) | 16.703 | 1.050 [1.010, 1.091] | 0.049 | 0.013 |  |
| Psychological distress, GHQ-12 (0-36) | 4.976 | 0.956 [0.936, 0.976] | -0.045 | <0.001 |  |
| Self-rated general health | 0.235 | 1.006 [0.980, 1.033] | 0.006 | 0.649 |  |

*Table 4-4. Continuous predictors ranked by the absolute log odds ratio per SD.*

**Reading the model.**

- **Current financial difficulty** is the strongest household-finance term: OR 1.65 per point on the 1–5 scale (1.59 per SD).
- **Employment-status security** is the strongest protective factor: OR 0.17 (0.63 per SD).
- **The OECD equivalence scale** has the largest per-SD effect (OR 0.47 per SD). It is partly mechanical: the outcome uses unequivalised income, so larger households have more income per unit of fuel need.
- **GHQ distress and financial expectations are slightly protective** once current difficulty is in the model. They were never one scale with it (α = 0.27; `finnow`–`finfut_risk` r = 0.02).
- **Housing size raises risk** (bedrooms OR 1.33 per room, rooms 1.15), as does tenure security (2.03). This is consistent with larger, owner-occupied homes costing more to heat relative to income. Cars are protective (0.86).
- **Health.** No long-standing illness raises odds (1.25). Self-rated health is null (1.03). v1 had mislabelled the first as self-rated health.
- **Workless household** is null (1.05) net of employment security. Lone parents (1.50) and large families (1.96) are at higher risk.
- **FES Delta** is small but robust: OR 0.972 [0.962, 0.981] per unit, 0.93 per SD. Delta = forecast − realised, so vulnerability is higher when realised stress exceeds what was forecast.

| Interview year | OR (ref. 2010) | p |
|---|---|---|
| 2011 | 0.848 | <0.001 |
| 2012 | 0.865 | <0.001 |
| 2013 | 0.956 | 0.212 |
| 2014 | 0.924 | 0.037 |
| 2015 | 0.815 | <0.001 |
| 2016 | 0.647 | <0.001 |
| 2017 | 0.610 | <0.001 |
| 2018 | 0.609 | <0.001 |
| 2019 | 0.666 | <0.001 |
| 2020 | 0.629 | <0.001 |
| 2021 | 0.588 | <0.001 |
| 2022 | 0.778 | <0.001 |
| 2023 | 1.578 | <0.001 |
| 2024 | 1.241 | <0.001 |
| 2025 | 1.325 | 0.211 |

*Interview-year fixed effects, primary model (reference 2010; `outputs_v2/stage3/year_fe.csv`). Odds are lowest in 2016–2021 and highest in 2023 (OR 1.58).*

### 5.1 Sensitivities

| Predictor (OR, main model) | primary | sens_composite | sens_composite_v1 | sens_lagged_components | sens_lagged_composite | sens_lagged_composite_v1 | sens_month_fe | sens_fes_4term | sens_outcome_s1 | sens_outcome_s2 | sens_no_qualification |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Tenure security | 2.034 | 1.639 | 1.624 | 2.038 | 1.696 | 1.723 | 2.036 | 2.034 | 2.101 | 2.016 | 1.908 |
| Employment-status security | 0.174 | 0.153 | 0.152 | 0.145 | 0.146 | 0.147 | 0.174 | 0.174 | 0.178 | 0.178 | 0.163 |
| No long-standing illness/disability | 1.253 | 1.293 | 1.293 | 1.269 | 1.306 | 1.304 | 1.250 | 1.253 | 1.241 | 1.257 | 1.257 |
| Self-rated general health | 1.026 | 0.980 | 0.959 | 0.937 | 0.899 | 0.881 | 1.029 | 1.026 | 1.046 | 1.027 | 0.906 |
| Highest qualification band | 0.548 | 0.522 | 0.521 | 0.550 | 0.523 | 0.523 | 0.549 | 0.548 | 0.553 | 0.555 | — |
| Age (mean of adults) | 1.003 | 0.999 | 0.999 | 1.002 | 0.999 | 1.000 | 1.003 | 1.003 | 1.004 | 1.003 | 1.007 |
| Has central heating | 1.087 | 1.102 | 1.103 | 1.065 | 1.072 | 1.074 | 1.087 | 1.087 | 1.083 | 1.041 | 1.108 |
| Bedrooms | 1.335 | 1.315 | 1.315 | 1.300 | 1.296 | 1.296 | 1.335 | 1.335 | 1.305 | 1.327 | 1.320 |
| Rooms | 1.149 | 1.127 | 1.127 | 1.152 | 1.129 | 1.127 | 1.148 | 1.149 | 1.146 | 1.150 | 1.123 |
| Cars | 0.856 | 0.821 | 0.818 | 0.848 | 0.816 | 0.818 | 0.855 | 0.856 | 0.866 | 0.857 | 0.836 |
| Bill-payment security | 0.587 | 0.415 | 0.694 | 0.436 | 0.360 | 0.417 | 0.585 | 0.587 | 0.573 | 0.583 | 0.605 |
| Lone-parent household | 1.499 | 1.531 | 1.533 | 1.505 | 1.543 | 1.533 | 1.501 | 1.499 | 1.600 | 1.503 | 1.544 |
| Large family (3+ children) | 1.963 | 1.926 | 1.930 | 2.092 | 2.005 | 2.014 | 1.967 | 1.963 | 1.971 | 1.964 | 1.982 |
| OECD equivalence scale | 0.275 | 0.280 | 0.280 | 0.261 | 0.270 | 0.269 | 0.275 | 0.275 | 0.284 | 0.276 | 0.281 |
| Workless household | 1.046 | 0.948 | 0.947 | 0.937 | 0.899 | 0.898 | 1.047 | 1.046 | 1.021 | 1.055 | 1.038 |
| Current financial difficulty (1-5) | 1.649 | — | — | — | — | — | 1.648 | 1.649 | 1.639 | 1.646 | 1.674 |
| Psychological distress, GHQ-12 (0-36) | 0.991 | — | — | — | — | — | 0.991 | 0.991 | 0.990 | 0.991 | 0.989 |
| Financial expectations: worse off (0-1) | 0.818 | — | — | — | — | — | 0.818 | 0.818 | 0.843 | 0.819 | 0.837 |
| FES Delta (growth-only) | 0.972 | 0.973 | 0.973 | 0.972 | 0.973 | 0.973 | 0.968 | — | 0.974 | 0.972 | 0.973 |
| Strain composite (no bill arrears) | — | 4.756 | — | — | — | — | — | — | — | — | — |
| Strain composite, v1 (incl. bill arrears) | — | — | 7.515 | — | — | — | — | — | — | — | — |
| Current financial difficulty, previous wave | — | — | — | 1.502 | — | — | — | — | — | — | — |
| GHQ-12 distress, previous wave | — | — | — | 0.989 | — | — | — | — | — | — | — |
| Financial expectations, previous wave | — | — | — | 0.994 | — | — | — | — | — | — | — |
| Strain composite, previous wave | — | — | — | — | 4.052 | — | — | — | — | — | — |
| Strain composite v1, previous wave | — | — | — | — | — | 5.393 | — | — | — | — | — |
| FES Delta (4-term) | — | — | — | — | — | — | — | 0.970 | — | — | — |
| n | 221,877 | 233,980 | 234,102 | 183,078 | 194,151 | 194,692 | 221,877 | 221,877 | 249,200 | 227,007 | 248,802 |

*Table A-3 (main models). Odds ratios for every pre-specified sensitivity. Composite and lagged specifications replace the three strain components with composites or lagged terms (their rows are blank for the components and filled for the composite terms). Full file: `TA-3_driver_sensitivities.csv`, 1,450 rows.*

| Predictor | OR [95% CI], two-way clustered (PSU × interview year-month) | p |
|---|---|---|
| Tenure security | 2.034 [1.769, 2.337] | <0.001 |
| Employment-status security | 0.174 [0.152, 0.200] | <0.001 |
| No long-standing illness/disability | 1.253 [1.183, 1.327] | <0.001 |
| Self-rated general health | 1.026 [0.918, 1.147] | 0.647 |
| Highest qualification band | 0.548 [0.509, 0.590] | <0.001 |
| Age (mean of adults) | 1.003 [1.001, 1.005] | 0.016 |
| Has central heating | 1.087 [1.010, 1.171] | 0.026 |
| Bedrooms | 1.335 [1.281, 1.390] | <0.001 |
| Rooms | 1.149 [1.107, 1.194] | <0.001 |
| Cars | 0.856 [0.823, 0.890] | <0.001 |
| Bill-payment security | 0.587 [0.519, 0.664] | <0.001 |
| Lone-parent household | 1.499 [1.370, 1.639] | <0.001 |
| Large family (3+ children) | 1.963 [1.742, 2.213] | <0.001 |
| OECD equivalence scale | 0.275 [0.255, 0.297] | <0.001 |
| Workless household | 1.046 [0.948, 1.154] | 0.366 |
| Current financial difficulty (1-5) | 1.649 [1.606, 1.692] | <0.001 |
| Psychological distress, GHQ-12 (0-36) | 0.991 [0.987, 0.995] | <0.001 |
| Financial expectations: worse off (0-1) | 0.818 [0.763, 0.878] | <0.001 |
| FES Delta (growth-only) | 0.972 [0.958, 0.985] | <0.001 |

*Table A-3 (two-way clustered model). Same point estimates as the primary model; SEs clustered on PSU and on interview year-month (the unit at which FES varies).*

<details><summary><b>Table A-4. Driver-model N, events, PSUs and pseudo-R² (all 71 models)</b></summary>

| Specification | Model | n | Events | PSUs | Interview years | McFadden R² | AIC | Clustering | Converged |
|---|---|---|---|---|---|---|---|---|---|
| primary | `main` | 221,877 | 17,209 | 8,801 | 2010-2025 | 0.1531 | 102,581 | psu | True |
| primary | `main_twoway_cluster` | 221,877 | 17,209 | 8,801 | 2010-2025 | 0.1531 | 102,581 | psu x interview year-month | True |
| primary | `ni_a_regionFE_filled` | 221,778 | 17,197 | 8,801 | 2010-2025 | 0.1586 | 101,869 | psu | True |
| primary | `ni_b_oil_filled` | 221,778 | 17,197 | 8,801 | 2010-2025 | 0.1626 | 101,388 | psu | True |
| primary | `ni_b_oil_rural_filled` | 221,778 | 17,197 | 8,801 | 2010-2025 | 0.1631 | 101,336 | psu | True |
| primary | `ni_c_ni_x_oil_filled` | 221,778 | 17,197 | 8,801 | 2010-2025 | 0.1626 | 101,390 | psu | True |
| primary | `ni_c_ni_x_oil_rural_filled` | 221,778 | 17,197 | 8,801 | 2010-2025 | 0.1631 | 101,338 | psu | True |
| primary | `ni_b_oil_observed_only` | 221,611 | 17,176 | 8,800 | 2010-2025 | 0.1625 | 101,295 | psu | True |
| primary | `ni_b_oil_rural_observed_only` | 221,611 | 17,176 | 8,800 | 2010-2025 | 0.1629 | 101,243 | psu | True |
| primary | `ni_c_ni_x_oil_observed_only` | 221,611 | 17,176 | 8,800 | 2010-2025 | 0.1625 | 101,297 | psu | True |
| primary | `ni_c_ni_x_oil_rural_observed_only` | 221,611 | 17,176 | 8,800 | 2010-2025 | 0.1629 | 101,245 | psu | True |
| sens_composite | `main` | 233,980 | 19,022 | 9,224 | 2010-2025 | 0.1408 | 113,425 | psu | True |
| sens_composite | `ni_a_regionFE_filled` | 233,873 | 19,008 | 9,223 | 2010-2025 | 0.1459 | 112,697 | psu | True |
| sens_composite | `ni_b_oil_filled` | 233,873 | 19,008 | 9,223 | 2010-2025 | 0.1499 | 112,171 | psu | True |
| sens_composite | `ni_b_oil_rural_filled` | 233,873 | 19,008 | 9,223 | 2010-2025 | 0.1503 | 112,120 | psu | True |
| sens_composite | `ni_c_ni_x_oil_filled` | 233,873 | 19,008 | 9,223 | 2010-2025 | 0.1499 | 112,173 | psu | True |
| sens_composite | `ni_c_ni_x_oil_rural_filled` | 233,873 | 19,008 | 9,223 | 2010-2025 | 0.1503 | 112,122 | psu | True |
| sens_composite_v1 | `main` | 234,102 | 19,055 | 9,233 | 2010-2025 | 0.1403 | 113,637 | psu | True |
| sens_composite_v1 | `ni_a_regionFE_filled` | 233,995 | 19,041 | 9,232 | 2010-2025 | 0.1455 | 112,909 | psu | True |
| sens_composite_v1 | `ni_b_oil_filled` | 233,995 | 19,041 | 9,232 | 2010-2025 | 0.1495 | 112,381 | psu | True |
| sens_composite_v1 | `ni_b_oil_rural_filled` | 233,995 | 19,041 | 9,232 | 2010-2025 | 0.1499 | 112,331 | psu | True |
| sens_composite_v1 | `ni_c_ni_x_oil_filled` | 233,995 | 19,041 | 9,232 | 2010-2025 | 0.1495 | 112,383 | psu | True |
| sens_composite_v1 | `ni_c_ni_x_oil_rural_filled` | 233,995 | 19,041 | 9,232 | 2010-2025 | 0.1499 | 112,333 | psu | True |
| sens_lagged_components | `main` | 183,078 | 13,518 | 8,130 | 2010-2025 | 0.1456 | 82,494 | psu | True |
| sens_lagged_components | `ni_a_regionFE_filled` | 183,013 | 13,507 | 8,130 | 2010-2025 | 0.1512 | 81,913 | psu | True |
| sens_lagged_components | `ni_b_oil_filled` | 183,013 | 13,507 | 8,130 | 2010-2025 | 0.1559 | 81,465 | psu | True |
| sens_lagged_components | `ni_b_oil_rural_filled` | 183,013 | 13,507 | 8,130 | 2010-2025 | 0.1565 | 81,406 | psu | True |
| sens_lagged_components | `ni_c_ni_x_oil_filled` | 183,013 | 13,507 | 8,130 | 2010-2025 | 0.1559 | 81,467 | psu | True |
| sens_lagged_components | `ni_c_ni_x_oil_rural_filled` | 183,013 | 13,507 | 8,130 | 2010-2025 | 0.1565 | 81,407 | psu | True |
| sens_lagged_composite | `main` | 194,151 | 15,018 | 8,407 | 2010-2025 | 0.1380 | 91,190 | psu | True |
| sens_lagged_composite | `ni_a_regionFE_filled` | 194,079 | 15,006 | 8,406 | 2010-2025 | 0.1438 | 90,540 | psu | True |
| sens_lagged_composite | `ni_b_oil_filled` | 194,079 | 15,006 | 8,406 | 2010-2025 | 0.1484 | 90,060 | psu | True |
| sens_lagged_composite | `ni_b_oil_rural_filled` | 194,079 | 15,006 | 8,406 | 2010-2025 | 0.1490 | 90,001 | psu | True |
| sens_lagged_composite | `ni_c_ni_x_oil_filled` | 194,079 | 15,006 | 8,406 | 2010-2025 | 0.1484 | 90,062 | psu | True |
| sens_lagged_composite | `ni_c_ni_x_oil_rural_filled` | 194,079 | 15,006 | 8,406 | 2010-2025 | 0.1490 | 90,003 | psu | True |
| sens_lagged_composite_v1 | `main` | 194,692 | 15,067 | 8,416 | 2010-2025 | 0.1379 | 91,486 | psu | True |
| sens_lagged_composite_v1 | `ni_a_regionFE_filled` | 194,620 | 15,055 | 8,415 | 2010-2025 | 0.1438 | 90,824 | psu | True |
| sens_lagged_composite_v1 | `ni_b_oil_filled` | 194,620 | 15,055 | 8,415 | 2010-2025 | 0.1483 | 90,346 | psu | True |
| sens_lagged_composite_v1 | `ni_b_oil_rural_filled` | 194,620 | 15,055 | 8,415 | 2010-2025 | 0.1489 | 90,287 | psu | True |
| sens_lagged_composite_v1 | `ni_c_ni_x_oil_filled` | 194,620 | 15,055 | 8,415 | 2010-2025 | 0.1483 | 90,348 | psu | True |
| sens_lagged_composite_v1 | `ni_c_ni_x_oil_rural_filled` | 194,620 | 15,055 | 8,415 | 2010-2025 | 0.1489 | 90,289 | psu | True |
| sens_month_fe | `main` | 221,877 | 17,209 | 8,801 | 2010-2025 | 0.1538 | 102,519 | psu | True |
| sens_month_fe | `ni_a_regionFE_filled` | 221,778 | 17,197 | 8,801 | 2010-2025 | 0.1593 | 101,807 | psu | True |
| sens_month_fe | `ni_b_oil_filled` | 221,778 | 17,197 | 8,801 | 2010-2025 | 0.1633 | 101,328 | psu | True |
| sens_month_fe | `ni_b_oil_rural_filled` | 221,778 | 17,197 | 8,801 | 2010-2025 | 0.1637 | 101,278 | psu | True |
| sens_month_fe | `ni_c_ni_x_oil_filled` | 221,778 | 17,197 | 8,801 | 2010-2025 | 0.1633 | 101,330 | psu | True |
| sens_month_fe | `ni_c_ni_x_oil_rural_filled` | 221,778 | 17,197 | 8,801 | 2010-2025 | 0.1637 | 101,280 | psu | True |
| sens_fes_4term | `main` | 221,877 | 17,209 | 8,801 | 2010-2025 | 0.1531 | 102,576 | psu | True |
| sens_fes_4term | `ni_a_regionFE_filled` | 221,778 | 17,197 | 8,801 | 2010-2025 | 0.1587 | 101,864 | psu | True |
| sens_fes_4term | `ni_b_oil_filled` | 221,778 | 17,197 | 8,801 | 2010-2025 | 0.1627 | 101,384 | psu | True |
| sens_fes_4term | `ni_b_oil_rural_filled` | 221,778 | 17,197 | 8,801 | 2010-2025 | 0.1631 | 101,332 | psu | True |
| sens_fes_4term | `ni_c_ni_x_oil_filled` | 221,778 | 17,197 | 8,801 | 2010-2025 | 0.1627 | 101,385 | psu | True |
| sens_fes_4term | `ni_c_ni_x_oil_rural_filled` | 221,778 | 17,197 | 8,801 | 2010-2025 | 0.1631 | 101,334 | psu | True |
| sens_outcome_s1 | `main` | 249,200 | 17,320 | 8,950 | 2010-2025 | 0.1480 | 107,228 | psu | True |
| sens_outcome_s1 | `ni_a_regionFE_filled` | 249,076 | 17,308 | 8,950 | 2010-2025 | 0.1544 | 106,370 | psu | True |
| sens_outcome_s1 | `ni_b_oil_filled` | 249,076 | 17,308 | 8,950 | 2010-2025 | 0.1588 | 105,829 | psu | True |
| sens_outcome_s1 | `ni_b_oil_rural_filled` | 249,076 | 17,308 | 8,950 | 2010-2025 | 0.1592 | 105,777 | psu | True |
| sens_outcome_s1 | `ni_c_ni_x_oil_filled` | 249,076 | 17,308 | 8,950 | 2010-2025 | 0.1588 | 105,829 | psu | True |
| sens_outcome_s1 | `ni_c_ni_x_oil_rural_filled` | 249,076 | 17,308 | 8,950 | 2010-2025 | 0.1592 | 105,779 | psu | True |
| sens_outcome_s2 | `main` | 227,007 | 17,406 | 8,868 | 2010-2025 | 0.1521 | 104,233 | psu | True |
| sens_outcome_s2 | `ni_a_regionFE_filled` | 226,907 | 17,394 | 8,868 | 2010-2025 | 0.1563 | 103,673 | psu | True |
| sens_outcome_s2 | `ni_b_oil_filled` | 226,907 | 17,394 | 8,868 | 2010-2025 | 0.1600 | 103,223 | psu | True |
| sens_outcome_s2 | `ni_b_oil_rural_filled` | 226,907 | 17,394 | 8,868 | 2010-2025 | 0.1604 | 103,172 | psu | True |
| sens_outcome_s2 | `ni_c_ni_x_oil_filled` | 226,907 | 17,394 | 8,868 | 2010-2025 | 0.1600 | 103,225 | psu | True |
| sens_outcome_s2 | `ni_c_ni_x_oil_rural_filled` | 226,907 | 17,394 | 8,868 | 2010-2025 | 0.1604 | 103,174 | psu | True |
| sens_no_qualification | `main` | 248,802 | 20,416 | 9,291 | 2010-2025 | 0.1484 | 120,317 | psu | True |
| sens_no_qualification | `ni_a_regionFE_filled` | 248,687 | 20,400 | 9,291 | 2010-2025 | 0.1571 | 119,029 | psu | True |
| sens_no_qualification | `ni_b_oil_filled` | 248,687 | 20,400 | 9,291 | 2010-2025 | 0.1614 | 118,417 | psu | True |
| sens_no_qualification | `ni_b_oil_rural_filled` | 248,687 | 20,400 | 9,291 | 2010-2025 | 0.1622 | 118,315 | psu | True |
| sens_no_qualification | `ni_c_ni_x_oil_filled` | 248,687 | 20,400 | 9,291 | 2010-2025 | 0.1615 | 118,415 | psu | True |
| sens_no_qualification | `ni_c_ni_x_oil_rural_filled` | 248,687 | 20,400 | 9,291 | 2010-2025 | 0.1622 | 118,315 | psu | True |

</details>

---

## 6. Resources and the buffering hypothesis (RQ2; H1)

### 6.1 Measuring resources

The pre-registered correlated four-factor CFA (complete-case ML) failed the fixed criteria on its only attempt.

| n (complete case) | χ² | df | CFI | TLI | RMSEA | SRMR | Pre-registered criteria |
|---|---|---|---|---|---|---|---|
| 143,770 | 51,361.8 | 59 | 0.783 | 0.714 | 0.078 | 0.066 | CFI ≥ 0.90, RMSEA ≤ 0.08, SRMR ≤ 0.08, no Heywood case, all std. loadings ≥ 0.30 → **failed** |

*Table A-9. CFA fit.*

| Factor | Item | Label | Estimate | Std. estimate | SE | p |
|---|---|---|---|---|---|---|
| OBJECT | `hsrooms` | Rooms | 1.000 | 0.544 | fixed (marker) | — |
| OBJECT | `hsbeds` | Bedrooms | 1.230 | 0.669 | 0.0087 | <0.001 |
| OBJECT | `ncars` | Cars | 0.736 | 0.400 | 0.0068 | <0.001 |
| OBJECT | `carval` | Car value (log) | 0.598 | 0.325 | 0.0065 | <0.001 |
| OBJECT | `hsval` | House value (log) | 0.938 | 0.510 | 0.0074 | <0.001 |
| CONDITION | `tenure_security` | Tenure security | 1.000 | 0.512 | fixed (marker) | — |
| CONDITION | `jbstat_security` | Employment-status security | -1.450 | -0.743 | 0.0141 | <0.001 |
| CONDITION | `bill_security` | Bill-payment security | -0.058 | -0.030 | 0.0063 | <0.001 |
| PERSONAL | `sf1_good` | Self-rated general health | 1.000 | 0.720 | fixed (marker) | — |
| PERSONAL | `health_good` | No long-standing illness/disability | 0.903 | 0.651 | 0.0075 | <0.001 |
| PERSONAL | `qfhigh_band` | Highest qualification band | 0.358 | 0.258 | 0.0048 | <0.001 |
| ENERGY | `fihhmnnet1_dv` | Net household income (log) | 1.000 | 1.000 | fixed (marker) | — |
| ENERGY | `fiyrinvinc_dv` | Investment income (log) | 0.141 | 0.141 | 0.0058 | <0.001 |

*Table A-10. CFA loadings. Employment-status security loads negatively on CONDITION (−0.74), bill security is near zero, and the ENERGY marker is a Heywood case (std. loading 1.00).*

| Decision | CFA passes | Failures |
|---|---|---|
| unit-weighted standardised composites (CFA failed pre-registered criteria) | False | CFI 0.783 < 0.90; Heywood case; CONDITION:jbstat_security std loading -0.74 < 0.30; CONDITION:bill_security std loading -0.03 < 0.30; PERSONAL:qfhigh_band std loading 0.26 < 0.30; ENERGY:fiyrinvinc_dv std loading 0.14 < 0.30 |

| Factor | Factor  | Correlation |
|---|---|---|
| OBJECT | CONDITION | 0.024 |
| OBJECT | PERSONAL | 0.234 |
| OBJECT | ENERGY | 0.437 |
| CONDITION | PERSONAL | -0.489 |
| CONDITION | ENERGY | -0.339 |
| PERSONAL | ENERGY | 0.213 |

*Factor correlations (`outputs_v2/resources/cfa_factor_correlations.csv`).*

The complete-case CFA sample is 99% owner-occupiers (private renters 0.1%, social renters 0.6%; `cfa_sample_composition.csv`), because house and car values are asked only of owners. Resources are therefore measured by **unit-weighted formative indices**: the mean of standardised items per domain, requiring at least 50% of items observed, then re-standardised. α is descriptive only.

| Domain | Items | n complete | Cronbach α |
|---|---|---|---|
| OBJECT | 5 | 163,645 | 0.60 |
| CONDITION | 3 | 333,100 | 0.34 |
| PERSONAL | 3 | 297,044 | 0.58 |
| ENERGY | 2 | 336,881 | 0.32 |

*Cronbach's α per domain (`outputs_v2/resources/composite_alpha.csv`).*

|  | OBJECT | CONDITION | PERSONAL | ENERGY | R_primary | R_with_energy |
|---|---|---|---|---|---|---|
| OBJECT | 1.00 | 0.37 | 0.28 | 0.44 | 0.75 | 0.75 |
| CONDITION | 0.37 | 1.00 | 0.25 | 0.35 | 0.74 | 0.70 |
| PERSONAL | 0.28 | 0.25 | 1.00 | 0.22 | 0.70 | 0.63 |
| ENERGY | 0.44 | 0.35 | 0.22 | 1.00 | 0.47 | 0.72 |
| R_primary | 0.75 | 0.74 | 0.70 | 0.47 | 1.00 | 0.95 |
| R_with_energy | 0.75 | 0.70 | 0.63 | 0.72 | 0.95 | 1.00 |

*Correlations of domain composites and of R (`outputs_v2/resources/composite_item_correlations.csv`). R_primary = OBJECT + CONDITION + PERSONAL; R_with_energy adds ENERGY.*

### 6.2 H1: does R moderate the FES Delta slope?

![Figure 4-12. H1 Delta slopes](../outputs_v2/thesis_assets_v2/figures/fig4-12_h1_delta_slopes.png)

*Figure 4-12. H1: predicted FES Delta slopes at resource percentiles, with 95% CIs.*

Decision rule, fixed before fitting: COR predicts a **positive** R × Delta interaction (resources flatten the negative Delta slope). *Supported* if positive with p < 0.05; *contrary to COR* if negative with p < 0.05; *not supported* otherwise.

| Model | R × Delta interaction | p | Verdict | Governs H1 | n |
|---|---|---|---|---|---|
| primary | -0.0000338 | 0.163 | not supported | True | 269,372 |
| sens_R_with_energy | 0.0000180 | 0.400 | not supported | False | 269,372 |
| sens_logit_binary | -0.0035192 | <0.001 | contrary to COR | False | 269,372 |
| sens_fes_4term | -0.0000070 | 0.755 | not supported | False | 269,372 |

*Table 4-5 (verdicts). OLS of the fuel-to-income ratio on R, FES Delta (growth-only), R × Delta and interview-year FE; SEs clustered on PSU (9,775 PSUs); n = 269,372.*

| Model | Estimator | Outcome | Term | Coefficient [95% CI] | SE | p | R² / pseudo-R² | PSUs |
|---|---|---|---|---|---|---|---|---|
| primary | ols | fuel_to_income_ratio | `R_primary` | -0.004678 [-0.004851, -0.004506] | 0.000088 | <0.001 | 0.0446 | 9,775 |
| primary | ols | fuel_to_income_ratio | `fes_delta_growth3` | -0.000545 [-0.000693, -0.000398] | 0.000075 | <0.001 | 0.0446 | 9,775 |
| primary | ols | fuel_to_income_ratio | `R_primary x fes_delta_growth3` | -0.000034 [-0.000081, 0.000014] | 0.000024 | 0.163 | 0.0446 | 9,775 |
| sens_R_with_energy | ols | fuel_to_income_ratio | `R_with_energy` | -0.005366 [-0.005505, -0.005227] | 0.000071 | <0.001 | 0.0810 | 9,775 |
| sens_R_with_energy | ols | fuel_to_income_ratio | `fes_delta_growth3` | -0.000547 [-0.000696, -0.000397] | 0.000076 | <0.001 | 0.0810 | 9,775 |
| sens_R_with_energy | ols | fuel_to_income_ratio | `R_with_energy x fes_delta_growth3` | 0.000018 [-0.000024, 0.000060] | 0.000021 | 0.400 | 0.0810 | 9,775 |
| sens_logit_binary | logit | high_fuel_vulnerable | `R_primary` | -0.240102 [-0.249174, -0.231029] | 0.004629 | <0.001 | 0.0511 | 9,775 |
| sens_logit_binary | logit | high_fuel_vulnerable | `fes_delta_growth3` | -0.031140 [-0.039455, -0.022825] | 0.004242 | <0.001 | 0.0511 | 9,775 |
| sens_logit_binary | logit | high_fuel_vulnerable | `R_primary x fes_delta_growth3` | -0.003519 [-0.005608, -0.001430] | 0.001066 | <0.001 | 0.0511 | 9,775 |
| sens_fes_4term | ols | fuel_to_income_ratio | `R_primary` | -0.004666 [-0.004840, -0.004492] | 0.000089 | <0.001 | 0.0446 | 9,775 |
| sens_fes_4term | ols | fuel_to_income_ratio | `fes_delta` | -0.000564 [-0.000707, -0.000421] | 0.000073 | <0.001 | 0.0446 | 9,775 |
| sens_fes_4term | ols | fuel_to_income_ratio | `R_primary x fes_delta` | -0.000007 [-0.000051, 0.000037] | 0.000022 | 0.755 | 0.0446 | 9,775 |

*Table 4-5 (coefficients).*

| Model | R percentile | R value | Delta slope [95% CI] | Scale |
|---|---|---|---|---|
| primary | p10 | -2.959 | -0.000446 [-0.000672, -0.000219] | fuel-to-income ratio per unit Delta |
| primary | p50 | 0.329 | -0.000557 [-0.000701, -0.000412] | fuel-to-income ratio per unit Delta |
| primary | p90 | 2.506 | -0.000630 [-0.000795, -0.000465] | fuel-to-income ratio per unit Delta |
| sens_R_with_energy | p10 | -3.614 | -0.000612 [-0.000876, -0.000347] | fuel-to-income ratio per unit Delta |
| sens_R_with_energy | p50 | 0.344 | -0.000540 [-0.000683, -0.000398] | fuel-to-income ratio per unit Delta |
| sens_R_with_energy | p90 | 3.366 | -0.000486 [-0.000626, -0.000346] | fuel-to-income ratio per unit Delta |
| sens_logit_binary | p10 | -2.959 | -0.020728 [-0.030373, -0.011083] | log-odds per unit Delta |
| sens_logit_binary | p50 | 0.329 | -0.032299 [-0.040738, -0.023861] | log-odds per unit Delta |
| sens_logit_binary | p90 | 2.506 | -0.039960 [-0.050385, -0.029535] | log-odds per unit Delta |
| sens_fes_4term | p10 | -2.959 | -0.000543 [-0.000758, -0.000328] | fuel-to-income ratio per unit Delta |
| sens_fes_4term | p50 | 0.329 | -0.000566 [-0.000706, -0.000426] | fuel-to-income ratio per unit Delta |
| sens_fes_4term | p90 | 2.506 | -0.000581 [-0.000740, -0.000423] | fuel-to-income ratio per unit Delta |

*Table 4-5 (Delta slopes at the 10th, 50th and 90th percentiles of R).*

| Interaction value | b (R × Delta) | R p10 → p90 | SD of Delta | Slope change p10→p90 per SD Delta (ratio units) | Same, pp of income | Slope at p10 per SD Delta (pp) | Change as % of p10 slope (+ = flatter, i.e. buffering) |
|---|---|---|---|---|---|---|---|
| point estimate | -0.0000338 | -2.96 → 2.51 | 2.375 | -0.000438 | -0.044 | -0.106 | -41% |
| CI lower | -0.0000812 | -2.96 → 2.51 | 2.375 | -0.001053 | -0.105 | -0.106 | -100% |
| CI upper (max buffering) | 0.0000136 | -2.96 → 2.51 | 2.375 | 0.000177 | 0.018 | -0.106 | +17% |

*Table 4-5 (buffering bound). At the upper confidence limit of the interaction, moving from the 10th to the 90th resource percentile flattens the Delta slope by at most 0.018 percentage points of income per SD of Delta, about 17% of the slope at the 10th percentile.*

| R percentile | R value | Delta effect, pp per SD [95% CI] |
|---|---|---|
| p10 | -2.959 | -0.61 [-0.89, -0.33] |
| p50 | 0.329 | -0.52 [-0.65, -0.38] |
| p90 | 2.506 | -0.41 [-0.51, -0.30] |

*Table 4-5 (logit footnote, probability scale). The logit interaction is negative on the log-odds scale (−0.0035, p < 0.001), yet on the probability scale the Delta effect is smaller for better-resourced households. Both patterns reflect differences in baseline risk, not buffering, and neither changes the primary verdict.*

**Verdict: H1 not supported.** Resources have a strong main effect (−0.0047 per unit of R, p < 0.001), but they do not change how the fuel burden responds to forecast–realised price stress. A substantial buffering effect is ruled out; a small one (at most 17% of the slope) cannot be excluded.

---

## 7. Comparison with JRF income poverty, and Northern Ireland (RQ4; H4)

### 7.1 Agreement by dimension

![Figure 4-13. Regions vs JRF](../outputs_v2/thesis_assets_v2/figures/fig4-13_jrf_regions.png)

*Figure 4-13. Regional fuel vulnerability vs JRF income poverty (time-matched). Pipeline version: `outputs_v2/stage5/figures/F5_1_region_vs_jrf.png`.*

![Figure 4-14. Other dimensions vs JRF](../outputs_v2/thesis_assets_v2/figures/fig4-14_jrf_other_dims.png)

*Figure 4-14. Other JRF dimensions, time-matched, with 95% CIs. Pipeline version: `outputs_v2/stage5/figures/F5_2_dimensions_vs_jrf.png`.*

| F5.1 Region vs JRF | F5.2 Other dimensions |
|---|---|
| ![](../outputs_v2/stage5/figures/F5_1_region_vs_jrf.png) | ![](../outputs_v2/stage5/figures/F5_2_dimensions_vs_jrf.png) |

| Dimension | Window | Category | JRF % (rank) | n | Fuel-vulnerable % [95% CI] | Rank [95% CI] | P(rank 1) |
|---|---|---|---|---|---|---|---|
| region | primary (2021-04 to 2023-03) | North East | 21 (5) | 1,047 | 8.8 [6.5, 11.3] | 7 [3–12] | 0.001 |
| region | primary (2021-04 to 2023-03) | North West | 25 (2) | 2,757 | 7.9 [6.6, 9.4] | 9 [6–12] | 0.000 |
| region | primary (2021-04 to 2023-03) | Yorkshire and the Humber | 23 (4) | 2,367 | 9.5 [7.7, 11.3] | 6 [3–10] | 0.000 |
| region | primary (2021-04 to 2023-03) | East Midlands | 20 (8) | 1,996 | 10.5 [8.6, 12.5] | 5 [2–8] | 0.004 |
| region | primary (2021-04 to 2023-03) | West Midlands | 27 (1) | 2,203 | 10.6 [8.6, 12.6] | 4 [2–8] | 0.004 |
| region | primary (2021-04 to 2023-03) | East of England | 18 (11) | 2,377 | 8.6 [7.2, 10.1] | 8 [5–11] | 0.000 |
| region | primary (2021-04 to 2023-03) | London | 24 (3) | 2,676 | 6.7 [5.0, 8.7] | 12 [8–12] | 0.000 |
| region | primary (2021-04 to 2023-03) | South East | 19 (9) | 3,433 | 7.4 [6.2, 8.7] | 11 [7–12] | 0.000 |
| region | primary (2021-04 to 2023-03) | South West | 19 (9) | 2,343 | 7.8 [6.4, 9.3] | 10 [6–12] | 0.000 |
| region | primary (2021-04 to 2023-03) | Wales | 21 (5) | 1,691 | 11.4 [9.4, 13.6] | 2 [1–6] | 0.027 |
| region | primary (2021-04 to 2023-03) | Scotland | 21 (5) | 2,677 | 11.2 [9.4, 13.2] | 3 [2–6] | 0.018 |
| region | primary (2021-04 to 2023-03) | Northern Ireland | 17 (12) | 1,562 | 14.4 [12.1, 17.0] | 1 [1–2] | 0.946 |
| region | sensitivity (2020-04 to 2023-03) | North East | 21 (5) | 1,607 | 8.1 [6.3, 10.1] | 7 [3–11] | 0.000 |
| region | sensitivity (2020-04 to 2023-03) | North West | 25 (2) | 4,189 | 7.1 [6.1, 8.2] | 9 [6–12] | 0.000 |
| region | sensitivity (2020-04 to 2023-03) | Yorkshire and the Humber | 23 (4) | 3,614 | 8.4 [7.0, 9.8] | 6 [3–9] | 0.000 |
| region | sensitivity (2020-04 to 2023-03) | East Midlands | 20 (8) | 3,094 | 9.2 [7.6, 11.0] | 5 [2–8] | 0.000 |
| region | sensitivity (2020-04 to 2023-03) | West Midlands | 27 (1) | 3,438 | 9.4 [7.9, 10.8] | 4 [2–7] | 0.000 |
| region | sensitivity (2020-04 to 2023-03) | East of England | 18 (11) | 3,635 | 7.7 [6.4, 9.1] | 8 [4–11] | 0.000 |
| region | sensitivity (2020-04 to 2023-03) | London | 24 (3) | 4,311 | 6.3 [4.8, 8.0] | 12 [7–12] | 0.000 |
| region | sensitivity (2020-04 to 2023-03) | South East | 19 (9) | 5,235 | 6.5 [5.6, 7.5] | 11 [8–12] | 0.000 |
| region | sensitivity (2020-04 to 2023-03) | South West | 19 (9) | 3,596 | 7.0 [5.9, 8.2] | 10 [6–12] | 0.000 |
| region | sensitivity (2020-04 to 2023-03) | Wales | 21 (5) | 2,643 | 9.4 [7.9, 11.1] | 3 [2–7] | 0.000 |
| region | sensitivity (2020-04 to 2023-03) | Scotland | 21 (5) | 4,017 | 10.3 [8.9, 11.8] | 2 [2–5] | 0.002 |
| region | sensitivity (2020-04 to 2023-03) | Northern Ireland | 17 (12) | 2,305 | 14.2 [12.1, 16.4] | 1 [1–1] | 0.999 |
| ethnicity | primary (2021-04 to 2023-03) | White | 19 (6) | 21,646 | 9.3 [8.7, 9.8] | 5 [3–6] | 0.000 |
| ethnicity | primary (2021-04 to 2023-03) | Pakistani | 49 (2) | 459 | 12.5 [8.2, 17.4] | 2 [1–5] | 0.108 |
| ethnicity | primary (2021-04 to 2023-03) | Bangladeshi | 56 (1) | 223 | 11.3 [4.6, 18.6] | 3 [1–6] | 0.080 |
| ethnicity | primary (2021-04 to 2023-03) | Black African | 40 (3) | 380 | 11.1 [7.2, 15.7] | 4 [1–6] | 0.034 |
| ethnicity | primary (2021-04 to 2023-03) | Black Caribbean | 30 (5) | 445 | 17.8 [10.2, 26.6] | 1 [1–4] | 0.770 |
| ethnicity | primary (2021-04 to 2023-03) | Any other Asian background | 34 (4) | 158 | 7.8 [3.0, 14.0] | 6 [2–6] | 0.009 |
| ethnicity | sensitivity (2020-04 to 2023-03) | White | 19 (6) | 33,189 | 8.2 [7.8, 8.7] | 5 [3–6] | 0.000 |
| ethnicity | sensitivity (2020-04 to 2023-03) | Pakistani | 49 (2) | 733 | 11.1 [7.5, 15.0] | 2 [1–5] | 0.084 |
| ethnicity | sensitivity (2020-04 to 2023-03) | Bangladeshi | 56 (1) | 362 | 9.0 [4.5, 14.1] | 4 [1–6] | 0.029 |
| ethnicity | sensitivity (2020-04 to 2023-03) | Black African | 40 (3) | 627 | 10.2 [7.1, 13.8] | 3 [1–5] | 0.036 |
| ethnicity | sensitivity (2020-04 to 2023-03) | Black Caribbean | 30 (5) | 743 | 17.0 [10.2, 25.2] | 1 [1–3] | 0.848 |
| ethnicity | sensitivity (2020-04 to 2023-03) | Any other Asian background | 34 (4) | 261 | 6.3 [2.7, 10.6] | 6 [3–6] | 0.002 |
| tenure | primary (2022-04 to 2023-03) | Owned outright | 14 (3) | 5,753 | 13.8 [12.7, 15.0] | 2 [1–2] | 0.348 |
| tenure | primary (2022-04 to 2023-03) | Buying with mortgage | 10 (4) | 4,602 | 5.9 [5.1, 6.8] | 4 [4–4] | 0.000 |
| tenure | primary (2022-04 to 2023-03) | Social renting | 44 (1) | 2,035 | 14.3 [12.2, 16.5] | 1 [1–2] | 0.648 |
| tenure | primary (2022-04 to 2023-03) | Private renting | 35 (2) | 1,667 | 11.4 [9.4, 13.5] | 3 [2–3] | 0.004 |
| disability | primary (2022-04 to 2023-03) | No disabled adult | 19 (2) | 9,071 | 9.5 [8.8, 10.4] | 2 [2–2] | 0.000 |
| disability | primary (2022-04 to 2023-03) | Contains disabled adult | 29 (1) | 4,797 | 14.3 [13.0, 15.6] | 1 [1–1] | 1.000 |
| family_type | primary (2022-04 to 2023-03) | Lone parent | 44 (1) | 584 | 20.8 [16.8, 25.1] | 1 [1–1] | 1.000 |
| family_type | primary (2022-04 to 2023-03) | Couple with children | 25 (2) | 2,317 | 7.8 [6.4, 9.4] | 2 [2–2] | 0.000 |
| work_status | primary (2022-04 to 2023-03) | Not in work | 54 (1) | 1,656 | 22.5 [19.6, 25.3] | 1 [1–1] | 1.000 |
| work_status | primary (2022-04 to 2023-03) | In work | 15 (2) | 8,178 | 7.0 [6.3, 7.7] | 2 [2–2] | 0.000 |

*Table 4-6 (comparison). UKHLS rates are weighted, time-matched to the JRF period, with 95% CIs and rank intervals from 2,000 PSU bootstrap replicates. Rank 1 = highest.*

| Dimension | Window | Subset | Categories | Spearman ρ | Pearson r | Two groups, same direction |
|---|---|---|---|---|---|---|
| region | primary | all | 12 | -0.10 | -0.26 |  |
| region | primary | excl. Northern Ireland | 11 | 0.18 | 0.08 |  |
| region | sensitivity | all | 12 | -0.10 | -0.31 |  |
| region | sensitivity | excl. Northern Ireland | 11 | 0.18 | 0.11 |  |
| ethnicity | primary | all | 6 | 0.26 | 0.05 |  |
| ethnicity | sensitivity | all | 6 | 0.14 | -0.06 |  |
| tenure | primary | all | 4 | 0.80 | 0.58 |  |
| disability | primary | all | 2 |  |  | True |
| family_type | primary | all | 2 |  |  | True |
| work_status | primary | all | 2 |  |  | True |

*Table 4-6 (agreement).*

- **Tenure** agrees most closely (ρ = 0.80). Outright owners are the exception: third of four on JRF income poverty (14%) but second on fuel vulnerability (13.8% [12.7, 15.0]), above private renters (11.4%; JRF 35%).
- **Disability, family type and work status** rank in the JRF order: a disabled adult 14.3% vs none 9.5%; lone parent 20.8% vs couple with children 7.8%; not in work 22.5% vs in work 7.0%. The last two compare different units (JRF children and working-age adults vs UKHLS households) and are directional only.
- **Region** does not agree (ρ = −0.10 for all 12 regions; 0.18 without NI). v1's "0.73 excluding NI" does not hold on the corrected, weighted, time-matched rates.
- **Ethnicity** is inconclusive (ρ = 0.26). Every minority-group CI is wide (e.g. Black Caribbean 17.8% [10.2, 26.6]; Bangladeshi 11.3%, n = 223).

<details><summary><b>Table A-12. Regional rates in both JRF windows and both outcomes, with rank CIs</b></summary>

| Window | Outcome | Region | n | PSUs | Weighted % [95% CI] | Rank [95% CI] | P(rank 1) |
|---|---|---|---|---|---|---|---|
| primary (2021-04 to 2023-03) | primary | North East | 1,047 | 198 | 8.8 [6.5, 11.3] | 7 [3–12] | 0.001 |
| primary (2021-04 to 2023-03) | primary | North West | 2,757 | 607 | 7.9 [6.6, 9.4] | 9 [6–12] | 0.000 |
| primary (2021-04 to 2023-03) | primary | Yorkshire and the Humber | 2,367 | 478 | 9.5 [7.7, 11.3] | 6 [3–10] | 0.000 |
| primary (2021-04 to 2023-03) | primary | East Midlands | 1,996 | 426 | 10.5 [8.6, 12.5] | 5 [2–8] | 0.004 |
| primary (2021-04 to 2023-03) | primary | West Midlands | 2,203 | 526 | 10.6 [8.6, 12.6] | 4 [2–8] | 0.004 |
| primary (2021-04 to 2023-03) | primary | East of England | 2,377 | 564 | 8.6 [7.2, 10.1] | 8 [5–11] | 0.000 |
| primary (2021-04 to 2023-03) | primary | London | 2,676 | 1,065 | 6.7 [5.0, 8.7] | 12 [8–12] | 0.000 |
| primary (2021-04 to 2023-03) | primary | South East | 3,433 | 759 | 7.4 [6.2, 8.7] | 11 [7–12] | 0.000 |
| primary (2021-04 to 2023-03) | primary | South West | 2,343 | 501 | 7.8 [6.4, 9.3] | 10 [6–12] | 0.000 |
| primary (2021-04 to 2023-03) | primary | Wales | 1,691 | 316 | 11.4 [9.4, 13.6] | 2 [1–6] | 0.027 |
| primary (2021-04 to 2023-03) | primary | Scotland | 2,677 | 457 | 11.2 [9.4, 13.2] | 3 [2–6] | 0.018 |
| primary (2021-04 to 2023-03) | primary | Northern Ireland | 1,562 | 890 | 14.4 [12.1, 17.0] | 1 [1–2] | 0.946 |
| primary (2021-04 to 2023-03) | s1_lower_bound | North East | 1,264 | 216 | 7.5 [5.4, 9.6] | 7 [3–11] | 0.000 |
| primary (2021-04 to 2023-03) | s1_lower_bound | North West | 3,388 | 636 | 6.5 [5.4, 7.7] | 9 [6–12] | 0.000 |
| primary (2021-04 to 2023-03) | s1_lower_bound | Yorkshire and the Humber | 2,919 | 508 | 7.8 [6.4, 9.4] | 6 [3–10] | 0.000 |
| primary (2021-04 to 2023-03) | s1_lower_bound | East Midlands | 2,421 | 450 | 8.8 [7.1, 10.5] | 4 [2–8] | 0.000 |
| primary (2021-04 to 2023-03) | s1_lower_bound | West Midlands | 2,744 | 568 | 8.7 [7.1, 10.3] | 5 [2–8] | 0.000 |
| primary (2021-04 to 2023-03) | s1_lower_bound | East of England | 2,945 | 611 | 7.0 [5.8, 8.2] | 8 [5–11] | 0.000 |
| primary (2021-04 to 2023-03) | s1_lower_bound | London | 3,449 | 1,200 | 5.2 [3.8, 6.7] | 12 [9–12] | 0.000 |
| primary (2021-04 to 2023-03) | s1_lower_bound | South East | 4,200 | 802 | 6.0 [5.0, 7.1] | 11 [7–12] | 0.000 |
| primary (2021-04 to 2023-03) | s1_lower_bound | South West | 2,856 | 531 | 6.5 [5.3, 7.7] | 10 [6–12] | 0.000 |
| primary (2021-04 to 2023-03) | s1_lower_bound | Wales | 2,085 | 328 | 9.2 [7.5, 10.9] | 3 [2–7] | 0.003 |
| primary (2021-04 to 2023-03) | s1_lower_bound | Scotland | 3,159 | 468 | 9.5 [8.0, 11.2] | 2 [2–6] | 0.004 |
| primary (2021-04 to 2023-03) | s1_lower_bound | Northern Ireland | 1,745 | 953 | 13.2 [11.1, 15.5] | 1 [1–1] | 0.993 |
| sensitivity (2020-04 to 2023-03) | primary | North East | 1,607 | 212 | 8.1 [6.3, 10.1] | 7 [3–11] | 0.000 |
| sensitivity (2020-04 to 2023-03) | primary | North West | 4,189 | 640 | 7.1 [6.1, 8.2] | 9 [6–12] | 0.000 |
| sensitivity (2020-04 to 2023-03) | primary | Yorkshire and the Humber | 3,614 | 505 | 8.4 [7.0, 9.8] | 6 [3–9] | 0.000 |
| sensitivity (2020-04 to 2023-03) | primary | East Midlands | 3,094 | 452 | 9.2 [7.6, 11.0] | 5 [2–8] | 0.000 |
| sensitivity (2020-04 to 2023-03) | primary | West Midlands | 3,438 | 574 | 9.4 [7.9, 10.8] | 4 [2–7] | 0.000 |
| sensitivity (2020-04 to 2023-03) | primary | East of England | 3,635 | 604 | 7.7 [6.4, 9.1] | 8 [4–11] | 0.000 |
| sensitivity (2020-04 to 2023-03) | primary | London | 4,311 | 1,209 | 6.3 [4.8, 8.0] | 12 [7–12] | 0.000 |
| sensitivity (2020-04 to 2023-03) | primary | South East | 5,235 | 805 | 6.5 [5.6, 7.5] | 11 [8–12] | 0.000 |
| sensitivity (2020-04 to 2023-03) | primary | South West | 3,596 | 533 | 7.0 [5.9, 8.2] | 10 [6–12] | 0.000 |
| sensitivity (2020-04 to 2023-03) | primary | Wales | 2,643 | 330 | 9.4 [7.9, 11.1] | 3 [2–7] | 0.000 |
| sensitivity (2020-04 to 2023-03) | primary | Scotland | 4,017 | 469 | 10.3 [8.9, 11.8] | 2 [2–5] | 0.002 |
| sensitivity (2020-04 to 2023-03) | primary | Northern Ireland | 2,305 | 990 | 14.2 [12.1, 16.4] | 1 [1–1] | 0.999 |
| sensitivity (2020-04 to 2023-03) | s1_lower_bound | North East | 1,909 | 228 | 7.1 [5.5, 8.8] | 6 [2–10] | 0.000 |
| sensitivity (2020-04 to 2023-03) | s1_lower_bound | North West | 5,121 | 663 | 5.9 [4.9, 6.8] | 10 [7–12] | 0.000 |
| sensitivity (2020-04 to 2023-03) | s1_lower_bound | Yorkshire and the Humber | 4,392 | 530 | 7.0 [5.9, 8.3] | 7 [3–9] | 0.000 |
| sensitivity (2020-04 to 2023-03) | s1_lower_bound | East Midlands | 3,692 | 473 | 7.8 [6.5, 9.4] | 3 [2–8] | 0.000 |
| sensitivity (2020-04 to 2023-03) | s1_lower_bound | West Midlands | 4,191 | 605 | 7.8 [6.5, 9.0] | 4 [2–7] | 0.000 |
| sensitivity (2020-04 to 2023-03) | s1_lower_bound | East of England | 4,464 | 643 | 6.3 [5.2, 7.4] | 8 [5–11] | 0.000 |
| sensitivity (2020-04 to 2023-03) | s1_lower_bound | London | 5,494 | 1,325 | 5.0 [3.8, 6.4] | 12 [8–12] | 0.000 |
| sensitivity (2020-04 to 2023-03) | s1_lower_bound | South East | 6,347 | 845 | 5.3 [4.5, 6.1] | 11 [8–12] | 0.000 |
| sensitivity (2020-04 to 2023-03) | s1_lower_bound | South West | 4,309 | 558 | 5.9 [5.0, 6.9] | 9 [6–12] | 0.000 |
| sensitivity (2020-04 to 2023-03) | s1_lower_bound | Wales | 3,210 | 338 | 7.7 [6.5, 9.1] | 5 [2–8] | 0.000 |
| sensitivity (2020-04 to 2023-03) | s1_lower_bound | Scotland | 4,685 | 479 | 8.8 [7.5, 10.3] | 2 [2–5] | 0.001 |
| sensitivity (2020-04 to 2023-03) | s1_lower_bound | Northern Ireland | 2,555 | 1,046 | 13.0 [11.0, 15.1] | 1 [1–1] | 0.999 |

</details>

### 7.2 Northern Ireland and heating oil

![Figure 4-15. Northern Ireland and oil](../outputs_v2/thesis_assets_v2/figures/fig4-15_ni_oil.png)

*Figure 4-15. Northern Ireland: rates by heating fuel, and the NI gap across models. Pipeline version: `outputs_v2/stage5/figures/F5_3_ni_oil.png`.*

![F5.3 NI oil, pipeline version](../outputs_v2/stage5/figures/F5_3_ni_oil.png)

| Item | Value [95% CI] | Detail |
|---|---|---|
| NI rate, primary window (2021-04 to 2023-03), outcome primary | 14.4 [12.1, 17.0] | rank 1 of 12 (95% CI 1–2); P(rank 1) = 0.95; n = 1,562 |
| NI rate, primary window (2021-04 to 2023-03), outcome s1_lower_bound | 13.2 [11.1, 15.5] | rank 1 of 12 (95% CI 1–1); P(rank 1) = 0.99; n = 1,745 |
| NI rate, sensitivity window (2020-04 to 2023-03), outcome primary | 14.2 [12.1, 16.4] | rank 1 of 12 (95% CI 1–1); P(rank 1) = 1.00; n = 2,305 |
| NI rate, sensitivity window (2020-04 to 2023-03), outcome s1_lower_bound | 13.0 [11.0, 15.1] | rank 1 of 12 (95% CI 1–1); P(rank 1) = 1.00; n = 2,555 |
| NI oil share, primary window (weighted) | 72.5 | unweighted 66.1% |
| NI Oil heating, primary window | 18.2 [14.8, 21.7] | n = 1,017 |
| NI No oil, primary window | 8.6 [5.7, 11.9] | n = 545 |
| NI Oil heating, sensitivity window | 16.6 [13.8, 19.4] | n = 1,535 |
| NI No oil, sensitivity window | 10.3 [7.2, 13.6] | n = 770 |
| AME NI vs South East (pp), model ni_a_regionFE_filled | 6.8 [5.8, 7.9] | n = 221,778 |
| AME NI vs South East (pp), model ni_b_oil_filled | 2.4 [1.4, 3.4] | n = 221,778 |
| AME oil vs no oil (pp), model ni_b_oil_filled | 5.8 [4.9, 6.8] | n = 221,778 |
| AME NI vs South East (pp), model ni_b_oil_rural_filled | 2.5 [1.5, 3.5] | n = 221,778 |
| AME oil vs no oil (pp), model ni_b_oil_rural_filled | 4.9 [4.0, 5.9] | n = 221,778 |
| AME NI vs South East (pp), model ni_c_ni_x_oil_filled | 2.6 [1.2, 4.0] | n = 221,778 |
| AME oil vs no oil (pp), model ni_c_ni_x_oil_filled | 5.9 [4.9, 6.9] | n = 221,778 |
| AME NI vs South East (pp), model ni_c_ni_x_oil_rural_filled | 2.6 [1.2, 4.0] | n = 221,778 |
| AME oil vs no oil (pp), model ni_c_ni_x_oil_rural_filled | 4.9 [3.9, 6.0] | n = 221,778 |

*Thesis Table T5.4, Northern Ireland (`outputs_v2/stage5/thesis_T5_4_northern_ireland.csv`).*

| Model | n | Contrast | AME, pp [95% CI] | SE (pp) |
|---|---|---|---|---|
| `ni_a_regionFE_filled` | 221,778 | NI vs South East | 6.82 [5.76, 7.88] | 0.54 |
| `ni_b_oil_filled` | 221,778 | NI vs South East | 2.39 [1.40, 3.37] | 0.50 |
| `ni_b_oil_filled` | 221,778 | oil vs no oil | 5.84 [4.88, 6.79] | 0.49 |
| `ni_b_oil_rural_filled` | 221,778 | NI vs South East | 2.51 [1.52, 3.51] | 0.51 |
| `ni_b_oil_rural_filled` | 221,778 | oil vs no oil | 4.93 [3.96, 5.89] | 0.49 |
| `ni_c_ni_x_oil_filled` | 221,778 | NI vs South East | 2.63 [1.22, 4.03] | 0.72 |
| `ni_c_ni_x_oil_filled` | 221,778 | oil vs no oil | 5.90 [4.89, 6.92] | 0.52 |
| `ni_c_ni_x_oil_rural_filled` | 221,778 | NI vs South East | 2.59 [1.19, 3.99] | 0.72 |
| `ni_c_ni_x_oil_rural_filled` | 221,778 | oil vs no oil | 4.95 [3.92, 5.97] | 0.52 |
| `ni_b_oil_observed_only` | 221,611 | NI vs South East | 2.39 [1.40, 3.38] | 0.50 |
| `ni_b_oil_observed_only` | 221,611 | oil vs no oil | 5.83 [4.87, 6.79] | 0.49 |
| `ni_b_oil_rural_observed_only` | 221,611 | NI vs South East | 2.52 [1.53, 3.52] | 0.51 |
| `ni_b_oil_rural_observed_only` | 221,611 | oil vs no oil | 4.92 [3.95, 5.88] | 0.49 |
| `ni_c_ni_x_oil_observed_only` | 221,611 | NI vs South East | 2.65 [1.23, 4.06] | 0.72 |
| `ni_c_ni_x_oil_observed_only` | 221,611 | oil vs no oil | 5.90 [4.88, 6.92] | 0.52 |
| `ni_c_ni_x_oil_rural_observed_only` | 221,611 | NI vs South East | 2.62 [1.20, 4.03] | 0.72 |
| `ni_c_ni_x_oil_rural_observed_only` | 221,611 | oil vs no oil | 4.94 [3.92, 5.97] | 0.52 |

*Table 4-7. NI-oil sequence: average marginal effects in percentage points (delta-method CIs, PSU-clustered). Models: (a) region FE; (b) + oil use; (c) + NI × oil; "rural" adds `urban_dv` (adjacent-wave filled); "observed_only" uses unfilled `urban_dv`.*

| Model | Term | OR [95% CI] | p | n |
|---|---|---|---|---|
| `ni_a_regionFE_filled` | Northern Ireland (vs South East) | 2.523 [2.218, 2.870] | <0.001 | 221,778 |
| `ni_b_oil_filled` | Uses heating oil | 2.124 [1.917, 2.353] | <0.001 | 221,778 |
| `ni_b_oil_filled` | Northern Ireland (vs South East) | 1.466 [1.263, 1.702] | <0.001 | 221,778 |
| `ni_b_oil_rural_filled` | Uses heating oil | 1.923 [1.726, 2.143] | <0.001 | 221,778 |
| `ni_b_oil_rural_filled` | Rural location | 1.181 [1.106, 1.261] | <0.001 | 221,778 |
| `ni_b_oil_rural_filled` | Northern Ireland (vs South East) | 1.492 [1.286, 1.731] | <0.001 | 221,778 |
| `ni_c_ni_x_oil_filled` | Uses heating oil | 2.144 [1.916, 2.398] | <0.001 | 221,778 |
| `ni_c_ni_x_oil_filled` | Northern Ireland x heating oil | 0.943 [0.730, 1.219] | 0.656 | 221,778 |
| `ni_c_ni_x_oil_filled` | Northern Ireland (vs South East) | 1.525 [1.221, 1.905] | <0.001 | 221,778 |
| `ni_c_ni_x_oil_rural_filled` | Uses heating oil | 1.929 [1.713, 2.173] | <0.001 | 221,778 |
| `ni_c_ni_x_oil_rural_filled` | Northern Ireland x heating oil | 0.981 [0.759, 1.269] | 0.886 | 221,778 |
| `ni_c_ni_x_oil_rural_filled` | Rural location | 1.180 [1.105, 1.260] | <0.001 | 221,778 |
| `ni_c_ni_x_oil_rural_filled` | Northern Ireland (vs South East) | 1.511 [1.209, 1.888] | <0.001 | 221,778 |
| `ni_b_oil_observed_only` | Uses heating oil | 2.122 [1.915, 2.351] | <0.001 | 221,611 |
| `ni_b_oil_observed_only` | Northern Ireland (vs South East) | 1.467 [1.263, 1.704] | <0.001 | 221,611 |
| `ni_b_oil_rural_observed_only` | Uses heating oil | 1.921 [1.724, 2.141] | <0.001 | 221,611 |
| `ni_b_oil_rural_observed_only` | Rural location | 1.181 [1.106, 1.261] | <0.001 | 221,611 |
| `ni_b_oil_rural_observed_only` | Northern Ireland (vs South East) | 1.494 [1.287, 1.734] | <0.001 | 221,611 |
| `ni_c_ni_x_oil_observed_only` | Uses heating oil | 2.143 [1.915, 2.398] | <0.001 | 221,611 |
| `ni_c_ni_x_oil_observed_only` | Northern Ireland x heating oil | 0.939 [0.726, 1.214] | 0.631 | 221,611 |
| `ni_c_ni_x_oil_observed_only` | Northern Ireland (vs South East) | 1.531 [1.226, 1.913] | <0.001 | 221,611 |
| `ni_c_ni_x_oil_rural_observed_only` | Uses heating oil | 1.929 [1.712, 2.172] | <0.001 | 221,611 |
| `ni_c_ni_x_oil_rural_observed_only` | Northern Ireland x heating oil | 0.977 [0.755, 1.265] | 0.862 | 221,611 |
| `ni_c_ni_x_oil_rural_observed_only` | Rural location | 1.180 [1.105, 1.261] | <0.001 | 221,611 |
| `ni_c_ni_x_oil_rural_observed_only` | Northern Ireland (vs South East) | 1.517 [1.214, 1.896] | <0.001 | 221,611 |

*NI-oil models, odds ratios of the NI, oil, NI × oil and rural terms (primary specification; from `TA-3_driver_sensitivities.csv`).*

| Sample | NI households | Oil share, weighted (%) | Oil share, unweighted (%) | Oil: n / weighted % vulnerable | No oil: n / weighted % vulnerable |
|---|---|---|---|---|---|
| all waves (wave-mean) | 21,433 | 70.4 | 74.5 | 12,503 / 20.8 | 4,143 / 14.9 |
| primary window Apr 2021-Mar 2023 | 2,260 | 72.5 | 66.1 | 1,017 / 18.2 | 545 / 8.6 |
| sensitivity window Apr 2020-Mar 2023 | 3,324 | 65.9 | 67.2 | 1,535 / 16.6 | 770 / 10.3 |

*Oil share and oil vs non-oil vulnerability within NI (`outputs_v2/jrf/ni_oil.csv`).*

| Region | Oil heating, pooled unweighted (%) |
|---|---|
| East Midlands | 4.3 |
| East of England | 8.4 |
| London | 0.3 |
| North East | 2.1 |
| North West | 1.7 |
| Northern Ireland | 74.5 |
| Scotland | 7.4 |
| South East | 4.3 |
| South West | 9.9 |
| Wales | 10.6 |
| West Midlands | 4.0 |
| Yorkshire and the Humber | 2.7 |

*Oil heating by region, pooled unweighted (`outputs_v2/jrf/oil_share_by_region.csv`).*

**Reading.** NI has the highest rate in the JRF window, 14.4% [12.1, 17.0]. It ranks first in 95% of bootstrap replicates, and this holds under the S1 lower bound (13.2%) and in the wider window (14.2%). NI also has the lowest JRF income-poverty rate of the UK nations (17%). About 73% of NI households heat with oil (weighted, JRF window), against 0.3–10.6% elsewhere. Within NI, oil-heated households are at 18.2% against 8.6%.

In the driver model, the NI gap relative to the South East falls from 6.8 pp to 2.4 pp once oil use is controlled. Oil itself adds 5.8 pp (4.9 pp once rural location is also controlled). The oil penalty is no larger in NI than elsewhere (NI × oil OR 0.94, p = 0.66). Around two-thirds of NI's excess risk is accounted for by oil use; a gap of about 2.4–2.5 pp remains. These are associations; oil use is not randomly assigned.

---

## 8. Next-wave prediction (RQ5; H5)

![Figure 4-16. ROC curves](../outputs_v2/thesis_assets_v2/figures/fig4-16_roc_p0_p3.png)

*Figure 4-16. ROC curves on the validation transitions m→n and n→o (n = 19,960).*

![Figure 4-17. Calibration](../outputs_v2/thesis_assets_v2/figures/fig4-17_calibration.png)

*Figure 4-17. Calibration by decile of predicted risk, validation transitions.*

| Stage 6 ROC (pipeline) | Stage 6 calibration (pipeline) |
|---|---|
| ![](../outputs_v2/stage6/figures/roc.png) | ![](../outputs_v2/stage6/figures/calibration.png) |

| Model | ROC-AUC [95% CI] | PR-AUC [95% CI] | Calibration slope | Calibration-in-the-large | Top 5%: sensitivity / PPV (%) | Top 10%: sensitivity / PPV (%) | n validation | prevalence (%) | note |
|---|---|---|---|---|---|---|---|---|---|
| P0 current burden (benchmark) | 0.780 [0.768, 0.792] | 0.334 [0.314, 0.359] | 0.78 [0.74, 0.82] | 0.39 [0.34, 0.44] | 22.0 [20.6, 23.6] / 46.4 [43.3, 50.1] | 38.6 [36.6, 40.4] / 40.6 [38.4, 43.2] | 19960 | 10.54 |  |
| P1 household predictors | 0.739 [0.727, 0.751] | 0.280 [0.261, 0.302] | 0.93 [0.88, 0.98] | 0.45 [0.40, 0.49] | 18.8 [17.3, 20.3] / 39.6 [36.1, 43.1] | 30.9 [29.0, 32.6] / 32.5 [30.1, 34.7] | 19960 | 10.54 |  |
| P2 = P1 + FES (growth-only magnitude, t+1) | 0.739 [0.726, 0.750] | 0.278 [0.259, 0.300] | 0.93 [0.87, 0.98] | 0.42 [0.37, 0.47] | 18.7 [17.2, 20.2] / 39.4 [36.0, 42.7] | 30.7 [28.8, 32.4] / 32.3 [29.8, 34.6] | 19960 | 10.54 | no improvement over P1 (ΔAUC -0.0006) |
| P3 = P0 + P1 (POST-HOC / EXPLORATORY) | 0.781 [0.770, 0.792] | 0.341 [0.320, 0.365] |  |  |  | 39.0 [37.1, 40.6] / 41.1 [38.6, 43.6] | 19960 | 10.54 |  |

*Table 4-8. Next-wave prediction. P0 = this wave's fuel burden (benchmark); P1 = household predictors (financial difficulty, GHQ, financial expectations, age, central heating, lone parent, large family, workless, four resource composites); P2 = P1 + growth-only FES magnitude for the household's next interview month (December vintage before that year); P3 = P0 + P1 (post-hoc, exploratory). Training a→b … l→m (n = 170,895); standardisation uses training statistics only.*

| Comparison | ΔAUC [95% CI] |
|---|---|
| P1 - P0 | -0.0409 [-0.0531, -0.0285] |
| P2 - P1 | -0.0006 [-0.0009, -0.0002] |
| P2 - P0 | -0.0414 [-0.0538, -0.0292] |

*Paired PSU-bootstrap ΔAUC (`outputs_v2/stage6/delta_auc.csv`). P2 − P1 is reported as "no improvement", without significance language (author decision; its CI lies just below zero).*

| Comparison | ΔAUC [95% CI] | Label |
|---|---|---|
| P3 - P0 | 0.0008 [-0.0081, 0.0099] | POST-HOC / EXPLORATORY |
| P3 - P1 | 0.0417 [0.0353, 0.0482] | POST-HOC / EXPLORATORY |

*Post-hoc P3 comparisons (`outputs_v2/stage6/posthoc_p3_delta_auc.csv`).*

| Model | AUC | n train | n validation |
|---|---|---|---|
| P1 (P2b sample) | 0.7392 | 163,219 | 19,960 |
| P2b | 0.7395 | 163,219 | 19,960 |

*Sensitivity P2b: adding FES Delta observed at wave t, on its own common sample (`outputs_v2/stage6/p2b_sensitivity.csv`).*

| Model | Transition | n | Prevalence (%) | AUC [95% CI] |
|---|---|---|---|---|
| P0 | m→n | 9,075 | 9.6 | 0.781 [0.763, 0.800] |
| P0 | n→o | 10,885 | 11.3 | 0.777 [0.762, 0.792] |
| P1 | m→n | 9,075 | 9.6 | 0.738 [0.720, 0.756] |
| P1 | n→o | 10,885 | 11.3 | 0.740 [0.725, 0.754] |
| P2 | m→n | 9,075 | 9.6 | 0.737 [0.719, 0.755] |
| P2 | n→o | 10,885 | 11.3 | 0.740 [0.725, 0.754] |

*Table A-2. AUC by validation transition.*

| Model | Calibration slope [95% CI] | Calibration-in-the-large [95% CI] |
|---|---|---|
| P0 | 0.783 [0.743, 0.824] | 0.386 [0.335, 0.436] |
| P1 | 0.930 [0.878, 0.981] | 0.446 [0.399, 0.494] |
| P2 | 0.926 [0.875, 0.978] | 0.420 [0.373, 0.468] |

*Table A-7. Calibration slope (coefficient on logit p) and calibration-in-the-large (intercept with logit p as offset), log-odds scale.*

<details><summary><b>Model coefficients, P0–P2 (training fit)</b></summary>

| Model | Term | OR [95% CI] | p |
|---|---|---|---|
| P0 | `const` | 0.052 [0.050, 0.053] | <0.001 |
| P0 | `fuel_to_income_ratio` | 1.419 [1.386, 1.454] | <0.001 |
| P0 | `high_fuel_vulnerable` | 5.938 [5.543, 6.361] | <0.001 |
| P1 | `const` | 0.047 [0.043, 0.051] | <0.001 |
| P1 | `finnow` | 1.257 [1.230, 1.284] | <0.001 |
| P1 | `scghq1_dv` | 1.004 [0.985, 1.023] | 0.690 |
| P1 | `finfut_risk` | 1.035 [1.015, 1.055] | <0.001 |
| P1 | `dvage` | 1.346 [1.307, 1.385] | <0.001 |
| P1 | `heatch` | 0.959 [0.891, 1.032] | 0.263 |
| P1 | `lone_parent` | 1.478 [1.376, 1.589] | <0.001 |
| P1 | `large_family` | 0.860 [0.777, 0.952] | 0.004 |
| P1 | `workless_household` | 1.715 [1.622, 1.813] | <0.001 |
| P1 | `OBJECT` | 1.373 [1.341, 1.406] | <0.001 |
| P1 | `CONDITION` | 0.918 [0.898, 0.939] | <0.001 |
| P1 | `PERSONAL` | 0.968 [0.947, 0.990] | 0.005 |
| P1 | `ENERGY` | 0.382 [0.370, 0.394] | <0.001 |
| P2 | `const` | 0.047 [0.043, 0.051] | <0.001 |
| P2 | `finnow` | 1.257 [1.231, 1.284] | <0.001 |
| P2 | `scghq1_dv` | 1.004 [0.985, 1.023] | 0.711 |
| P2 | `finfut_risk` | 1.035 [1.015, 1.055] | <0.001 |
| P2 | `dvage` | 1.345 [1.307, 1.384] | <0.001 |
| P2 | `heatch` | 0.958 [0.890, 1.031] | 0.256 |
| P2 | `lone_parent` | 1.479 [1.376, 1.589] | <0.001 |
| P2 | `large_family` | 0.860 [0.777, 0.952] | 0.004 |
| P2 | `workless_household` | 1.715 [1.623, 1.813] | <0.001 |
| P2 | `OBJECT` | 1.373 [1.341, 1.406] | <0.001 |
| P2 | `CONDITION` | 0.918 [0.898, 0.939] | <0.001 |
| P2 | `PERSONAL` | 0.968 [0.946, 0.990] | 0.004 |
| P2 | `ENERGY` | 0.381 [0.369, 0.394] | <0.001 |
| P2 | `fes_magnitude_growth3_t1` | 1.013 [0.995, 1.032] | 0.165 |

*`outputs_v2/stage6/coefficients.csv`. Continuous predictors are z-scored with training statistics, so their ORs are per training SD.*

</details>

<details><summary><b>Table A-8. Prediction sample flow</b></summary>

| Step | n | Wave-t households |
|---|---|---|
| linked transitions a→b | 21,886 | 30,169 |
| linked transitions b→c | 24,404 | 30,484 |
| linked transitions c→d | 22,961 | 27,751 |
| linked transitions d→e | 22,034 | 25,817 |
| linked transitions e→f | 19,771 | 24,325 |
| linked transitions f→g | 20,001 | 24,454 |
| linked transitions g→h | 19,288 | 23,033 |
| linked transitions h→i | 17,954 | 21,746 |
| linked transitions i→j | 17,083 | 20,048 |
| linked transitions j→k | 16,234 | 19,252 |
| linked transitions k→l | 15,021 | 18,139 |
| linked transitions l→m | 14,253 | 16,856 |
| linked transitions m→n | 13,725 | 16,156 |
| linked transitions n→o | 17,144 | 21,385 |
| all linked transitions | 261,759 |  |
|   of which outcome at t+1 missing | 36,602 |  |
|   missing y (only this missing: 19,945) | 36,602 |  |
|   missing fuel_to_income_ratio (only this missing: 0) | 35,291 |  |
|   missing high_fuel_vulnerable (only this missing: 0) | 35,291 |  |
|   missing finnow (only this missing: 20) | 1,115 |  |
|   missing scghq1_dv (only this missing: 9,282) | 14,901 |  |
|   missing finfut_risk (only this missing: 1,694) | 3,997 |  |
|   missing dvage (only this missing: <10) | 730 |  |
|   missing heatch (only this missing: 28) | 773 |  |
|   missing workless_household (only this missing: 0) | 724 |  |
|   missing OBJECT (only this missing: 1,525) | 3,237 |  |
|   missing CONDITION (only this missing: 0) | 306 |  |
|   missing PERSONAL (only this missing: 12) | 1,402 |  |
|   missing ENERGY (only this missing: 0) | <10 |  |
| common sample (P0/P1/P2) | 190,855 |  |
|   training transitions (a→b ... l→m) | 170,895 |  |
|   validation transitions (m→n, n→o) | 19,960 |  |
|   validation prevalence (%) | 10.54 |  |
| P2b common sample (adds FES Delta at t; loses 2009 wave-t interviews) | 183,179 |  |

</details>

**Verdict: H5 partially supported.** Next-wave vulnerability can be predicted before it is observed with moderate discrimination (P1 AUC 0.739; PR-AUC 0.280 against a no-skill baseline of 0.105). But the simple benchmark is better:

- P0 has AUC 0.780, ΔAUC P1 − P0 = −0.041 [−0.053, −0.029].
- Among the 10% of households with the highest predicted risk, P0 captures 38.6% of next-wave cases (PPV 40.6%), against 30.9% for P1 (PPV 32.5%).
- The forecast adds nothing (P2 − P1 = −0.0006), nor does FES Delta at wave t (P2b).
- All models under-predict risk in the crisis-era validation waves (calibration-in-the-large 0.39–0.45). P0 is over-dispersed (slope 0.78).
- The post-hoc P3 = P0 + P1 gives no gain over P0 alone (AUC 0.781; ΔAUC 0.0008).

For early warning, the most informative signal is a household's current fuel burden.

---

## 9. Robustness and scope

| Estimate | Specification | Value [95% CI] | n |
|---|---|---|---|
| FES Delta OR | primary (main) | 0.972 [0.962, 0.981] | 221,877 |
| FES Delta OR | primary (main_twoway_cluster) | 0.972 [0.958, 0.985] | 221,877 |
| FES Delta OR | sens_composite (main) | 0.973 [0.964, 0.982] | 233,980 |
| FES Delta OR | sens_composite_v1 (main) | 0.973 [0.964, 0.982] | 234,102 |
| FES Delta OR | sens_lagged_components (main) | 0.972 [0.962, 0.983] | 183,078 |
| FES Delta OR | sens_lagged_composite (main) | 0.973 [0.962, 0.983] | 194,151 |
| FES Delta OR | sens_lagged_composite_v1 (main) | 0.973 [0.963, 0.983] | 194,692 |
| FES Delta OR | sens_month_fe (main) | 0.968 [0.959, 0.978] | 221,877 |
| FES Delta OR | sens_fes_4term (main) | 0.970 [0.961, 0.980] | 221,877 |
| FES Delta OR | sens_outcome_s1 (main) | 0.974 [0.965, 0.983] | 249,200 |
| FES Delta OR | sens_outcome_s2 (main) | 0.972 [0.963, 0.981] | 227,007 |
| FES Delta OR | sens_no_qualification (main) | 0.973 [0.964, 0.981] | 248,802 |
| NI gap AME (pp) | ni_a_regionFE_filled | 6.82 [5.76, 7.88] | 221,778 |
| NI gap AME (pp) | ni_b_oil_filled | 2.39 [1.40, 3.37] | 221,778 |
| NI gap AME (pp) | ni_b_oil_rural_filled | 2.51 [1.52, 3.51] | 221,778 |
| NI gap AME (pp) | ni_c_ni_x_oil_filled | 2.63 [1.22, 4.03] | 221,778 |
| NI gap AME (pp) | ni_c_ni_x_oil_rural_filled | 2.59 [1.19, 3.99] | 221,778 |
| NI gap AME (pp) | ni_b_oil_observed_only | 2.39 [1.40, 3.38] | 221,611 |
| NI gap AME (pp) | ni_b_oil_rural_observed_only | 2.52 [1.53, 3.52] | 221,611 |
| NI gap AME (pp) | ni_c_ni_x_oil_observed_only | 2.65 [1.23, 4.06] | 221,611 |
| NI gap AME (pp) | ni_c_ni_x_oil_rural_observed_only | 2.62 [1.20, 4.03] | 221,611 |
| NI rate (%) | NI rate, primary window (2021-04 to 2023-03), outcome primary | 14.42 [12.09, 16.96] |  |
| NI rate (%) | NI rate, primary window (2021-04 to 2023-03), outcome s1_lower_bound | 13.20 [11.11, 15.54] |  |
| NI rate (%) | NI rate, sensitivity window (2020-04 to 2023-03), outcome primary | 14.17 [12.09, 16.40] |  |
| NI rate (%) | NI rate, sensitivity window (2020-04 to 2023-03), outcome s1_lower_bound | 13.02 [10.99, 15.05] |  |
| H1 verdict | primary | not supported | 269,372 |
| H1 verdict | sens_R_with_energy | not supported | 269,372 |
| H1 verdict | sens_logit_binary | contrary to COR | 269,372 |
| H1 verdict | sens_fes_4term | not supported | 269,372 |
| Prediction AUC (P2b sample) | P1 (P2b sample) | 0.739 [, ] | 19,960 |
| Prediction AUC (P2b sample) | P2b | 0.740 [, ] | 19,960 |
| Prevalence wave a (%) | primary | 11.98 [, ] |  |
| Prevalence wave l (%) | primary | 6.52 [, ] |  |
| Prevalence wave n (%) | primary | 12.47 [, ] |  |
| Prevalence wave o (%) | primary | 12.42 [, ] |  |
| Prevalence wave a (%) | S1 lower bound | 10.81 [, ] |  |
| Prevalence wave l (%) | S1 lower bound | 5.56 [, ] |  |
| Prevalence wave n (%) | S1 lower bound | 10.22 [, ] |  |
| Prevalence wave o (%) | S1 lower bound | 9.94 [, ] |  |
| Prevalence wave a (%) | S2 incl. electricity not reported | 11.93 [, ] |  |
| Prevalence wave l (%) | S2 incl. electricity not reported | 6.41 [, ] |  |
| Prevalence wave n (%) | S2 incl. electricity not reported | 12.27 [, ] |  |
| Prevalence wave o (%) | S2 incl. electricity not reported | 12.23 [, ] |  |
| Prevalence wave a (%) | v1 rule | 10.77 [, ] |  |
| Prevalence wave l (%) | v1 rule | 5.69 [, ] |  |
| Prevalence wave n (%) | v1 rule | 11.42 [, ] |  |
| Prevalence wave o (%) | v1 rule | 11.43 [, ] |  |

*Table 4-9. Robustness summary across the pre-specified sensitivities.*

### 9.1 Equivalising income

![Figure 4-18. Equivalisation sensitivity](../outputs_v2/thesis_assets_v2/figures/fig4-18_equivalisation_sensitivity.png)

*Figure 4-18. Sensitivity to equivalising income only (fuel spend not equivalised). Pipeline version: `outputs_v2/stage7/figures/sensitivity_equivalised_income_only.png`.*

![Equivalisation sensitivity, pipeline version](../outputs_v2/stage7/figures/sensitivity_equivalised_income_only.png)

| Household size | n | Flagged, primary (%) | Flagged, income equivalised (%) | Flip (%) | In → out (%) | Out → in (%) |
|---|---|---|---|---|---|---|
| 1 | 73,682 | 15.9 | 15.9 | 0.0 | 0.0 | 0.0 |
| 2 | 99,863 | 6.5 | 17.4 | 11.5 | 0.0 | 11.5 |
| 3 | 45,329 | 5.3 | 22.1 | 18.2 | 0.0 | 18.2 |
| 4 | 43,354 | 3.6 | 27.0 | 25.3 | 0.0 | 25.3 |
| 5+ | 24,478 | 4.3 | 46.4 | 45.8 | 0.0 | 45.8 |

*Table A-11. Dividing income by the modified-OECD scale while leaving fuel spend unequivalised flags many more large households: 45.8% of 5+ person households change status, all into vulnerability. The primary flag stays unequivalised, like the official 10% definition.*

### 9.2 FES terciles (descriptive)

![Figure 4-19. FES tercile](../outputs_v2/thesis_assets_v2/figures/fig4-19_fes_tercile.png)

*Figure 4-19. Prevalence by FES tercile (descriptive).*

| FES term | Tercile | n | FES range | Fuel-vulnerable, weighted (%) |
|---|---|---|---|---|
| `fes_magnitude_growth3` | low | 91,376 | -2.34 to -1.59 | 8.1 |
| `fes_magnitude_growth3` | middle | 91,376 | -1.59 to -0.49 | 9.1 |
| `fes_magnitude_growth3` | high | 91,376 | -0.49 to 6.47 | 8.8 |
| `fes_delta_growth3` | low | 91,376 | -15.59 to -0.63 | 8.7 |
| `fes_delta_growth3` | middle | 91,376 | -0.63 to 0.24 | 8.8 |
| `fes_delta_growth3` | high | 91,376 | 0.24 to 3.50 | 8.3 |

*`outputs_v2/stage7/prevalence_by_fes_tercile.csv`. Raw prevalence is flat across FES terciles (8.1 / 9.1 / 8.8% for magnitude). The small FES Delta association in Table 4-3 is not visible in these raw terciles; it is estimated net of interview year and household characteristics.*

### 9.3 Scope: v1 exploratory models

![Figure A-4. v1 exploratory models](../outputs_v2/thesis_assets_v2/figures/figA_v1_exploratory.png)

*Figure A-4. v1 exploratory models (CVAE, fuzzy c-means, one-class SVM), labelled v1. They were computed on the v1 outcome, the abandoned second-order SEM and v1 FES timing, and were not re-estimated in v2. No main-text claim rests on them ([`appendix_scope_note.md`](../outputs_v2/reports/appendix_scope_note.md)). Dropped: the vector-shift map, the hotspot tiers, regional driver models, and v1 forward-risk maps by month and region.*

![Figure A-2. Macro covariates](../outputs_v2/thesis_assets_v2/figures/figA_macro_covariates.png)

*Figure A-2. Macro covariates (descriptive; not used by the v2 core-only forecasts).*

---

## 10. Caveats

- **Identification of FES effects.** FES is national and varies only by interview year and month. With year FE, the FES Delta coefficient is identified from within-year variation across interview months (about 185 year-month clusters). It is an association.
- **Forecast quality.** The forecasts do not beat a no-change forecast significantly, and they missed the 2022 shock. FES Delta is therefore dominated by realised stress in crisis months (mean Delta in 2022 = −7.1).
- **Missing spend.** The primary outcome is complete-case. Missing spend rises to 20% of in-scope households in wave o. S1 bounds the effect of treating non-response as £0.
- **Resources** are formative indices, not validated latent measures.
- **JRF** comparisons are time-matched but differ in construct (fuel vs income poverty), in unit (households vs people, children, working-age adults) and in data source.
- **Small cells** are suppressed (counts < 10; rates on n < 100 masked in group tables). Some ethnic groups do not appear in the JRF comparison for this reason or because JRF gives no stated rate.
- **Post-hoc items** (P3; the H5 wording) are labelled in the deviation log.

## 11. What the evidence establishes

1. After correcting the outcome, fuel vulnerability in UKHLS follows the energy-price cycle: it falls through the 2010s and roughly doubles in 2022–2024 fieldwork.
2. Household financial position, above all current financial difficulty and employment security, is far more strongly associated with fuel vulnerability than forecast price stress.
3. Forecast–realised price stress has a small, robust association with vulnerability, but it neither interacts with resources (H1) nor improves next-wave prediction (H5).
4. Fuel vulnerability is not income poverty: tenure agrees, but regions do not, and Northern Ireland's heating-oil exposure is the clearest source of divergence.
5. For targeting, a household's current fuel burden is the best available predictor of its fuel vulnerability at the next wave.
