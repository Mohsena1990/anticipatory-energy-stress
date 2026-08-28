# Outputs Catalog: File-by-File Analysis

**Project:** Anticipatory Fuel Stress Watch (AFSW) — Forecasting Anticipatory Energy–Carbon Stress and Household Fuel Vulnerability in the UK
**Scope:** every table (CSV) and figure (PNG/interactive HTML) under `outputs/` — 150 CSV/PNG files plus 6 interactive HTML companions, 156 in total — organized by pipeline stage, in the order the pipeline produces them. Every file is named individually and explicitly by its exact filename (no filename is abbreviated with a pattern or shorthand), followed by a precise, numbers-grounded analysis of its contents: what it structurally contains, the specific values in it, and anything notable, surprising, or worth flagging.

---

## Stage 1 — Macro Forecasting and the FES Index

### `outputs/tables/model_metrics_comparison.csv`

**Structure:** 24 data rows (3 series × 4 models × 2 modes). Columns: `series_name, model, mode, MAE, RMSE, MAPE, SMAPE, MASE, QuantileLoss, WinklerScore, MSIS, PredictionIntervalCoverage, rank_score, forecast_actual_MAE, forecast_actual_RMSE, forecast_actual_SMAPE, selection_basis, selection_score`.

This single file carries **two independent accuracy regimes** that frequently disagree with each other: (1) a backtest/walk-forward composite built from `MAE/RMSE/MAPE/SMAPE/MASE/QuantileLoss/WinklerScore/MSIS/PredictionIntervalCoverage`, summarized into `rank_score` (lower = better in backtest); and (2) accuracy of the final 2025 forecast against the *realised* 2025 actual, `forecast_actual_MAE/RMSE/SMAPE`, summarized into `selection_score` (1=best, 4=worst within each series/mode group) — every row shows `selection_basis=forecast_actual`, confirming this second regime is what actually drives model selection downstream.

**Winner by realised `forecast_actual_RMSE` per series/mode** (this is what is actually used by the pipeline):

| Series | Mode | Winner | RMSE | Runner-up | Worst |
|---|---|---|---|---|---|
| gas | core | LSTM | 7.60 | Prophet 12.54 | SARIMA 92.01 |
| gas | macro | LSTM | 11.86 | Prophet 31.62 | TFT 41.98 |
| electricity | core | LSTM | 4.14 | TFT 5.43 | SARIMA 46.84 |
| electricity | macro | LSTM | 4.54 | TFT 5.32 | SARIMA 19.76 |
| carbon | core | LSTM | 15.00 | SARIMA 24.45 | TFT 49.75 |
| carbon | macro | **TFT** | 13.79 | LSTM 20.62 | Prophet 101.98 |

This matches `outputs/fes/fes_components_table.csv`'s "best model" row exactly, confirming the two files are internally consistent with each other.

**Critical finding — backtest and reality disagree sharply in 4 of 6 cells.** Electricity-core: the backtest ranks SARIMA best (`rank_score`=1.0), but against realised 2025 data SARIMA is the single worst model in the entire 24-cell grid (RMSE 46.84 vs. LSTM's 4.14, an ~11× gap). Gas-core (backtest favours LSTM narrowly at 1.15 over SARIMA's 1.85; reality gives SARIMA an RMSE of 92.01 vs. LSTM's 7.60) and gas-macro (backtest favours SARIMA at 1.0 over LSTM's 3.15; reality reverses this completely, LSTM RMSE 11.86 vs. SARIMA's 37.83) show the same pattern. Carbon-macro is the one case where the backtest's narrow preference for LSTM (rank_score 1.45 vs. TFT's 2.15) flips to TFT once real outcomes are known (RMSE 13.79 vs. 20.62). Only electricity-macro has backtest and realised-accuracy rankings agree (LSTM best both ways). Because model selection uses `forecast_actual_RMSE` — computed *after* the 2025 actuals were already known — the FES pipeline's "best model" choice is, in a strict sense, a hindsight pick: it is not necessarily what a genuinely blind, pre-2025 forecast exercise would have selected. This is the single most important methodological caveat attached to Stage 1's headline numbers.

SARIMA is the consistent underperformer against real 2025 outcomes specifically for gas-core and electricity-core, despite scoring well or best in backtest for exactly those two cells — its wide, unstable long-horizon extrapolations (documented in the forecast-comparison figures below) are the direct cause. Prophet has the worst backtest scores in 5 of 6 cells (rank_score 3.65–3.95) — consistent with its flat, low-variance forecasts that track no turning point in any series.

---

### `outputs/forecasts/` (33 files)

Every per-model file shares columns `date, model, mode, forecast, lower_bound, upper_bound, actual` (a `series` column is present only in the raw LSTM/TFT per-model files, and is blank in the three aggregate files per series — a minor, harmless schema inconsistency worth noting for anyone querying these files programmatically). Each series has 8 per-model files (4 models × 2 modes) plus 3 aggregates (`_core.csv` = 4-model concatenation for core mode, `_macro.csv` = 4-model concatenation for macro mode, `_all.csv` = concatenation of both, 96 rows), for 11 files per series × 3 series = 33 files total.

#### Gas (`gas_growth_pct_forecasts_*.csv`)

Actual monthly gas growth 2025: −12.4%, −12.4%, −12.3%, +12.5%, +12.6%, +12.6%, +13.3%, +13.3%, +13.3%, +2.2%, +2.2%, +2.1% — a sharp regime shift from strongly negative (winter) to strongly positive (spring/summer) then back to near-zero (autumn/winter).

- **`outputs/forecasts/gas_growth_pct_forecasts_sarima_core.csv`** — 12 monthly rows. Mean forecast 80.84%, sign-matches the actual all 12 months but massively over-forecasts magnitude, reaching 95.9% by April (actual 12.5%) and 144% by October (actual 2.2%). Average prediction-interval width 90.37 percentage points, with the interval's upper bound reaching as high as ~203% by October/November; only 3 of 12 months (25%) have the actual value fall inside the stated interval, despite that interval's enormous width — a sign the point forecast itself, not just the uncertainty band, is badly miscalibrated.
- **`outputs/forecasts/gas_growth_pct_forecasts_sarima_macro.csv`** — tamer than its core counterpart (mean 33.50%) but still overshoots the actual materially; average PI width 84.53; 9 of 12 months (75%) covered.
- **`outputs/forecasts/gas_growth_pct_forecasts_prophet_core.csv`** — essentially flat, hovering 10–11% in every month regardless of the actual's dramatic swings; the widest average PI of any gas model (118.65); 12 of 12 months (100%) covered — a band so wide it is uninformative even though nominally "always correct."
- **`outputs/forecasts/gas_growth_pct_forecasts_prophet_macro.csv`** — noisier than Prophet-core rather than flat (63% in March, −22% in June, 39% in December), still 9/12 correct signs; narrower average PI (67.46) than core; 75% coverage.
- **`outputs/forecasts/gas_growth_pct_forecasts_lstm_core.csv`** — **the winning file for gas.** A smooth, monotonic-looking trend from −7.8% (January) up to +7.2% (December), correct sign in all 12 months, by far the tightest average PI of any gas model (11.77); coverage 5/12 (42%) — the interval is well-calibrated in width but occasionally too narrow for the actual's sharpest swings (e.g. the April/May regime jump).
- **`outputs/forecasts/gas_growth_pct_forecasts_lstm_macro.csv`** — a cautionary case: its RMSE (11.86) looks respectable in the metrics table, but it gets the sign wrong in 9 of 12 months, staying persistently negative (−1.2% to −2.7%) while the actual is mostly strongly positive. Its low RMSE reflects low variance/a "safe, small forecast" strategy, not genuine directional skill. Average PI width only 3.69 — the narrowest of any gas model — and 0 of 12 months (0%) have the actual fall inside it, the clearest single case of prediction-interval overconfidence in the whole forecasts directory.
- **`outputs/forecasts/gas_growth_pct_forecasts_tft_core.csv`** — drifts upward through the year, reaching ~70% by December against an actual of 2.1%; 11 of 12 correct signs; average PI width 56.10, 8/12 (67%) coverage.
- **`outputs/forecasts/gas_growth_pct_forecasts_tft_macro.csv`** — 12 of 12 correct signs, but RMSE 41.98 — the worst-performing macro-mode model for gas by realised accuracy despite its perfect sign record, illustrating that sign accuracy and magnitude accuracy are not the same thing; average PI width 45.71, 7/12 (58%) coverage.
- **`outputs/forecasts/gas_growth_pct_forecasts_core.csv`** — 48-row concatenation of the four core-mode model files above (used by the core-mode comparison and prediction-interval figures).
- **`outputs/forecasts/gas_growth_pct_forecasts_macro.csv`** — 48-row concatenation of the four macro-mode model files above.
- **`outputs/forecasts/gas_growth_pct_forecasts_all.csv`** — 96-row concatenation of both modes together (used by `interactive_gas_core.html`/`interactive_gas_macro.html` and the historical-context figure).

#### Electricity (`electricity_growth_pct_forecasts_*.csv`)

Actual monthly electricity growth 2025: −8.78%, −8.78%, −8.83%, +4.63%, +4.57%, +4.57%, +7.99%, +7.99%, +8.04%, +2.70%, +2.81%, +2.75%.

- **`outputs/forecasts/electricity_growth_pct_forecasts_sarima_core.csv`** — the single largest backtest-vs-reality contradiction in the whole dataset (best backtest rank_score, worst realised RMSE of all 24 model/series/mode cells, 46.84). Flat around −9% to −11% for January–March, then an abrupt jump to 51–55% from April onward against an actual that only reaches 4.6–8.0%; 12/12 correct signs but badly wrong on magnitude; average PI width 44.32, 3/12 (25%) coverage.
- **`outputs/forecasts/electricity_growth_pct_forecasts_sarima_macro.csv`** — same directional pathology, tamer (mean 16.64%); average PI width 39.14, 9/12 (75%) coverage.
- **`outputs/forecasts/electricity_growth_pct_forecasts_prophet_core.csv`** — flat around 7.95% mean, 9/12 correct signs, average PI width 61.82, 12/12 (100%) coverage (uninformatively wide, mirroring Prophet's gas-core behaviour).
- **`outputs/forecasts/electricity_growth_pct_forecasts_prophet_macro.csv`** — mean 7.67%, 9/12 correct signs, average PI width 47.76, 11/12 (92%) coverage.
- **`outputs/forecasts/electricity_growth_pct_forecasts_lstm_core.csv`** — **the winning file for electricity** (RMSE 4.14). Mean forecast 3.04%, 12/12 correct signs, by far the tightest average PI of any electricity model (5.29), 4/12 (33%) coverage — like gas-LSTM-core, a well-shaped but occasionally too-narrow interval.
- **`outputs/forecasts/electricity_growth_pct_forecasts_lstm_macro.csv`** — close second (RMSE 4.54), mean 1.14%, 12/12 correct signs, average PI width 7.11, 5/12 (42%) coverage.
- **`outputs/forecasts/electricity_growth_pct_forecasts_tft_core.csv`** — competitive (RMSE 5.43), mean 1.70%, 11/12 correct signs, average PI width 17.42, 9/12 (75%) coverage.
- **`outputs/forecasts/electricity_growth_pct_forecasts_tft_macro.csv`** — RMSE 5.32, mean 4.69%, 10/12 correct signs, average PI width 6.50 (tight), 6/12 (50%) coverage.
- **`outputs/forecasts/electricity_growth_pct_forecasts_core.csv`** — 48-row concatenation of the four core-mode electricity model files.
- **`outputs/forecasts/electricity_growth_pct_forecasts_macro.csv`** — 48-row concatenation of the four macro-mode electricity model files.
- **`outputs/forecasts/electricity_growth_pct_forecasts_all.csv`** — 96-row concatenation of both modes.

#### Carbon (`carbon_growth_pct_forecasts_*.csv`)

Actual monthly carbon growth 2025: +25.14%, +21.65%, +9.32%, −3.90%, −5.07%, +2.36%, +4.82%, +3.76%, +14.07%, +19.24%, +20.00%, +18.10% — the most volatile and least monotonic of the three series, with a mid-year dip into negative territory (April–May) before a strong Q3–Q4 recovery.

- **`outputs/forecasts/carbon_growth_pct_forecasts_sarima_core.csv`** — flat and negative (mean −11.24%) all year; only 2 of 12 months (17%) correct sign, missing the entire recovery; average PI width 169.12, but 12/12 (100%) coverage because the interval is so wide it is uninformative rather than accurate.
- **`outputs/forecasts/carbon_growth_pct_forecasts_sarima_macro.csv`** — near-identical to core (mean −11.30%), same 2/12 sign accuracy, average PI width 167.10, 12/12 (100%) coverage.
- **`outputs/forecasts/carbon_growth_pct_forecasts_prophet_core.csv`** — **a striking artifact**: `lower_bound=−99.7815` and `upper_bound=137.9618` are *literally identical, to four decimal places, in every one of the 12 monthly rows* — Prophet has collapsed to a frozen, non-time-varying uncertainty band rather than genuinely recalibrating month to month. Mean point forecast 39.13%, 10/12 correct signs, but RMSE 76.81 — the worst of any carbon model; 12/12 (100%) coverage, again uninformative given the band's width.
- **`outputs/forecasts/carbon_growth_pct_forecasts_prophet_macro.csv`** — the same frozen-band artifact (`lower_bound=−99.7815`, `upper_bound=137.9618` in every row), mean point forecast 29.15%, 10/12 correct signs, RMSE 67.07 (second-worst carbon model), 12/12 (100%) coverage.
- **`outputs/forecasts/carbon_growth_pct_forecasts_lstm_core.csv`** — mean forecast ≈0.00%, 9/12 correct signs, RMSE 15.00 — the best core-mode carbon model; average PI width 15.93, 5/12 (42%) coverage.
- **`outputs/forecasts/carbon_growth_pct_forecasts_lstm_macro.csv`** — only 2 of 12 correct signs (mean −6.36% against a series that is mostly positive), RMSE 20.62; average PI width 11.02 (the tightest of any carbon model, but badly miscalibrated in direction), 2/12 (17%) coverage.
- **`outputs/forecasts/carbon_growth_pct_forecasts_tft_core.csv`** — forecast rises essentially monotonically all year, from 7.3% in January to 91.5% by December, chasing the real recovery but badly overshooting its magnitude by year-end (actual December value 18.10%); this produces the **worst core-mode RMSE for carbon of any model (49.75)** despite a respectable 10/12 correct-sign record; average PI width 118.72, 10/12 (83%) coverage.
- **`outputs/forecasts/carbon_growth_pct_forecasts_tft_macro.csv`** — **the standout result of the entire forecasts directory**: RMSE 13.79, beating every other model in every other mode for carbon, including its own core counterpart (49.75) and LSTM-macro (20.62) — the only instance across all 24 model/series/mode cells where a macro-mode model outright wins over every alternative. 10/12 correct signs, average PI width 45.17, 11/12 (92%) coverage — a well-calibrated, accurate file.
- **`outputs/forecasts/carbon_growth_pct_forecasts_core.csv`** — 48-row concatenation of the four core-mode carbon model files.
- **`outputs/forecasts/carbon_growth_pct_forecasts_macro.csv`** — 48-row concatenation of the four macro-mode carbon model files.
- **`outputs/forecasts/carbon_growth_pct_forecasts_all.csv`** — 96-row concatenation of both modes.

---

### `outputs/fes/` (5 files)

#### `outputs/fes/fes_components_table.csv`

**Structure:** 5 rows (3 growth-rate components + uncertainty + total) × 4 columns (FES_core, FES_macro, FES_selected, FES_actual), plus one additional text row reporting the "best model" per mode.

| Component | FES_core | FES_macro | FES_selected | FES_actual |
|---|---|---|---|---|
| gas_growth_pct (z-score) | −0.20837 | −0.37860 | −0.20837 | −0.19390 |
| electricity_growth_pct (z-score) | −0.32264 | −0.44250 | −0.32264 | −0.41105 |
| carbon_growth_pct (z-score) | −0.03758 | +0.06007 | −0.03758 | +0.03237 |
| Uncertainty/RealVol (z-score) | −0.49218 | −0.49485 | −0.49218 | −0.57706 |
| **FES_total (annual mean)** | **−1.06077** | **−1.25588** | **−1.06077** | **−1.14962** |
| "best model" row | gas:LSTM \| electricity:LSTM \| carbon:LSTM | gas:LSTM \| electricity:LSTM \| carbon:TFT | gas:core \| electricity:core \| carbon:core | realised values |

**Key finding: `FES_selected` is bit-for-bit identical to `FES_core`**, confirmed at every decimal place across all five component rows, and independently confirmed across all 12 months in `fes_monthly_2025.csv`'s `selected_mode_gas`/`selected_mode_electricity`/`selected_mode_carbon` columns, which all read `"core"` for every month. **The per-series best-of-core/macro selection mechanism has, in this run, never once selected the macro-mode variant for any of the three series** — despite this same table's own "best model" row showing carbon's true best model is macro-mode TFT. This means the selection logic is choosing at the whole-mode level in a way that does not surface carbon's genuine macro-mode advantage into the composite index. Carbon is also the only component where core and macro disagree on sign (core −0.038, macro +0.060), and the realised actual (+0.032) sits on the macro side of that disagreement — i.e., FES_core gets the *direction* of carbon's 2025 contribution to stress wrong, and FES_selected inherits that error by construction. On the annual-mean total, FES_macro's −1.256 overshoots the realised −1.150 by 0.106, while FES_core's −1.061 undershoots it by 0.089 — nominally closer in absolute terms, though Section 2/6 below shows this doesn't mean core is the better-validated variant overall.

#### `outputs/fes/fes_summary_2025.csv`

**Structure:** long-format table, columns `variant, component, z_mean` — 20 rows covering core/macro/selected/actual variants × 5 components each, plus 2 additional rows for a prior-year actual baseline. Restates the same numbers as `fes_components_table.csv` in a different (row-per-observation) shape, and adds: `actual_prior_year, year, 2024.0` and **`actual_prior_year, FES_TOTAL, −2.582`**. This means realised 2025 stress (annual mean −1.150) is markedly less severe than realised 2024 stress (−2.582) — a roughly 55% reduction in the index's magnitude, indicating a substantial year-on-year easing of the combined gas/electricity/carbon price-and-uncertainty shock.

#### `outputs/fes/fes_prior_actual_2024.csv`

**Structure:** 12 monthly rows for calendar year 2024, columns `date, actual_gas, z_gas_actual, actual_electricity, z_electricity_actual, actual_carbon, z_carbon_actual, real_vol_actual, z_real_vol_actual, fes_actual`. Monthly `fes_actual` ranges from −3.080 (June, the worst month) to −1.787 (November, the best month) — every single month in 2024 is more negative (more stressed) than every month in the 2025 actual series described below. Gas and electricity actuals are deeply negative for much of 2024 (gas −38.3% and electricity −21.07% in April–June specifically) — a sharp deflationary/base-effect period consistent with 2022–23's price spike washing out of the year-on-year comparison base.

#### `outputs/fes/fes_monthly_2025.csv`

**Structure:** the widest file in the project — 12 monthly rows × 53 columns. Column groups: `forecast_*_core/z_*_core/pi_halfwidth_*_core/z_unc_*_core` for each of gas/electricity/carbon plus `fes_core`; a mirrored `*_macro` block plus `fes_macro`; a `selected_mode_*`/`z_*_selected` block plus `fes_selected`; an `actual_*` block plus `fes_actual`; and three further realised-volatility benchmark blocks (`rv_actual_A/B/C`, `z_rv_actual_A/B/C`, `fes_actual_A/B/C`) corresponding to the RollingVol/CrossComp/AbsShock definitions used in `fes_comparison_metrics.csv`.

**Monthly trajectories** (matching `fes_monthly_2025.png`): `fes_core` improves almost monotonically from −1.933 (January) to −0.683 (December). `fes_macro` is flatter, from −1.600 (January) to −1.376 (December) — much less improvement over the year than core implies. `fes_actual` follows a materially different, non-monotonic shape: a dip to roughly −1.79 around March, a sharp jump to about −0.97 by April, continued improvement to a peak (least-negative point) of roughly −0.71 by September, and then **a relapse back to approximately −1.15 to −1.16 by October–December** — a reversal that neither FES_core nor FES_macro anticipates, since both continue trending toward less-negative values right through December. This end-of-year divergence between forecast and reality is the single largest miss visible anywhere in the Stage 1 output set.

#### `outputs/fes/fes_comparison_metrics.csv`

**Structure:** 9 rows = 3 forecast variants (Equal_Core, Equal_Macro, Equal_Selected) × 3 realised-actual benchmark definitions (Actual_RollingVol, Actual_CrossComp, Actual_AbsShock), N=12 each. Columns: `FES_variant, Actual_benchmark, N, MAE, RMSE, Bias, MaxDev, Pearson_r, Pearson_p, Spearman_r, R2, Theil_U`.

| Variant | Benchmark | MAE | RMSE | Pearson r | R² | Theil U |
|---|---|---|---|---|---|---|
| Equal_Core | RollingVol | 0.577 | 0.700 | 0.727 | 0.247 | 1.019 |
| Equal_Core | CrossComp | 0.222 | 0.284 | 0.742 | 0.236 | 1.076 |
| Equal_Core | AbsShock | 0.224 | 0.296 | 0.740 | **−0.850** | **1.718** |
| Equal_Macro | RollingVol | 0.521 | 0.711 | **0.813** | 0.223 | 1.035 |
| Equal_Macro | CrossComp | 0.215 | 0.245 | **0.801** | 0.434 | 0.926 |
| Equal_Macro | AbsShock | **0.135** | **0.170** | 0.778 | 0.387 | 0.989 |
| Equal_Selected | RollingVol / CrossComp / AbsShock | identical to Equal_Core to the last reported decimal, since Selected≡Core in this run | | | | |

**FES_Macro beats FES_Core (and hence FES_Selected) on nearly every aggregate-index robustness metric** — a higher Pearson correlation against all three realised-stress benchmarks (0.813 vs. 0.727 on RollingVol, 0.801 vs. 0.742 on CrossComp, 0.778 vs. 0.740 on AbsShock) and a lower MAE/RMSE on two of the three (CrossComp, AbsShock) — even though the per-series model-comparison table favoured core-mode models in 5 of 6 series/mode cells. This is a genuine, unresolved tension between per-series accuracy and aggregate-index tracking quality. **FES_Core/Selected also posts a negative R² (−0.850) and a Theil's U of 1.718 against the AbsShock benchmark specifically** — meaning it performs worse than simply predicting the benchmark's own mean, and worse than a naive persistence (no-change) forecast, on that one definition of realised stress. FES_Macro's Theil's U values (0.926–1.035) all sit close to 1.0 — modest predictive value over a naive persistence forecast, not a decisive beat, but consistently better-behaved than Core/Selected.

---

### `outputs/figures/` (21 PNG files, each with a duplicate vector `.pdf` twin, plus 6 interactive `.html` companions)

#### FES-level figures

- **`outputs/figures/fes_components_2025.png`** — a two-panel bar chart. Left panel: 4 components (Gas Growth, Elec Growth, Carbon Growth, Uncertainty/RealVol) as grouped z-score bars for FES_core (purple), FES_macro (red/orange), FES_actual (navy), with value labels matching `fes_components_table.csv` exactly (−0.21/−0.38/−0.19 for gas; −0.32/−0.44/−0.41 for electricity; −0.04/+0.06/+0.03 for carbon; −0.49/−0.49/−0.58 for uncertainty). Right panel: three annual-mean total bars (Core −1.061, Macro −1.256, Actual −1.150), visually confirming Core undershoots and Macro overshoots the realised total by roughly symmetric margins. This session's earlier fix (explicit y-axis padding for the most negative bars) is visible in the final version — the −0.58 and −1.256 value labels sit fully clear of the x-axis tick labels rather than overlapping them.
- **`outputs/figures/fes_monthly_2025.png`** — a line chart of the three FES variants across the 12 months of 2025 (Core solid, Macro dashed, Actual dotted/marked), visually confirming the actual series' March dip, April jump, September peak, and October–December relapse described above, and that neither forecast variant anticipates the year-end reversal — both continue trending toward less-negative values through December while the actual line turns back down.
- **`outputs/figures/fes_metrics_pearson_r_heatmap.png`** — a 3×3 heatmap (rows = FES variant, columns = realised-benchmark definition), reproducing `fes_comparison_metrics.csv`'s Pearson r column exactly. Equal_Macro/RollingVol (r=0.813) renders as the single greenest/best cell; Equal_Core and Equal_Selected/RollingVol (r=0.727, identical since Selected≡Core) render as the reddest/worst cells. The RollingVol column is visually the hardest benchmark to hit for every variant regardless of core/macro choice — all three variants' RollingVol cells are the weakest in their respective rows.
- **`outputs/figures/fes_metrics_rmse_heatmap.png`** — the RMSE-column companion heatmap; Equal_Macro/AbsShock (RMSE=0.170) is the single best (palest) cell on the whole grid; the RollingVol column is again the worst (~0.70–0.71) across all three variants, matching the Pearson heatmap's pattern that RollingVol is a uniformly harder benchmark, independent of which FES variant is being scored.

#### Forecast-comparison figures (6 files: one per series × mode)

- **`outputs/figures/forecast_comparison_gas_core.png`** — all four core-mode gas models plotted against the 2025 actual (black markers). SARIMA (blue) visibly breaks away from the other three from April onward, its line and shaded prediction-interval band dominating the chart's y-axis scale (reaching above 200% by October/November) and compressing the other three models' more accurate trajectories into a visually flat cluster near the bottom of the plot. TFT (green, dotted) also drifts upward late in the year (~70% by December) with a narrower band than SARIMA's. LSTM (purple) and Prophet (orange) stay visually closest to the actual's cluster near 0–15% for most of the year.
- **`outputs/figures/forecast_comparison_gas_macro.png`** — SARIMA's macro-mode band is tamer than its core counterpart (~65% maximum vs. core's 144%) but still overshoots materially. Prophet-macro is noticeably noisier/zigzagging (63% in March, −22% in June, +39% in December) rather than the flat line seen in Prophet-core. LSTM-macro is visibly flat and slightly negative throughout, sitting below the actual's positive summer cluster — the chart makes its poor sign-accuracy (3/12) immediately apparent even though its numeric RMSE looks competitive.
- **`outputs/figures/forecast_comparison_electricity_core.png`** — the same SARIMA blow-up pattern as gas-core (flat around −9% to −11% for Jan–Mar, jumping to 51–55% from April), with a correspondingly wide shaded band. LSTM and TFT visibly track the actual's shape (rise from −9% to +8% then settle to +2–3%) most closely of any model, with the tightest bands on the whole chart.
- **`outputs/figures/forecast_comparison_electricity_macro.png`** — SARIMA's macro band is again tamer (~20% peak) than its core counterpart but still visibly overshoots; LSTM-macro and TFT-macro remain the tightest, most accurate-looking bands.
- **`outputs/figures/forecast_comparison_carbon_core.png`** — TFT is the visual outlier here: a steadily climbing dotted green line from roughly 7% to 92% over the year, diverging furthest from the actual by year-end of any model in this chart. SARIMA sits flat and negative (~−10%) throughout, visibly missing the actual's positive Q4 recovery entirely. Prophet's shaded band is a literally unchanged rectangle in width and vertical position across the whole year (from about −100% to +138%) — the frozen-PI artifact noted in the underlying CSVs is directly visible here as a static, non-adapting shaded region.
- **`outputs/figures/forecast_comparison_carbon_macro.png`** — TFT-macro (the actual best-performing carbon model) shows a far tighter, better-tracking band than its own core counterpart, rising from ~13% to ~35% rather than core's 7%-to-92% overshoot; SARIMA-macro and Prophet-macro repeat their core-mode pathologies (flat-negative and frozen-band respectively) in tamer form.

#### Forecast-vs-actual, long-history context figures (4 files)

- **`outputs/figures/forecast_vs_actual_gas.png`** — plots the full 2005–2024 historical monthly gas growth series (a continuous coloured line) alongside only the winning core/macro model forecasts and the 2025 actual, clustered at the chart's right edge past a dotted vertical line marking the forecast origin. Historical gas swings range from about −38% to +133%, with the 2022 energy-crisis spike visible as a large, sustained plateau around 2022–2023 — against which the entire 2025 forecast/actual cluster (roughly −13% to +13%) occupies a small, comparatively calm sliver at the chart's right edge.
- **`outputs/figures/forecast_vs_actual_electricity.png`** — same construction for electricity; historical range approximately −21% to +67%, with a similar 2022-crisis plateau, against which 2025's cluster (−9% to +8%) is again visually minor by comparison.
- **`outputs/figures/forecast_vs_actual_carbon.png`** — historical carbon range is dramatically wider than the other two series, roughly −680% to +735%, reflecting the extreme volatility of the EU-ETS carbon market particularly in its 2007–2009 formative years; the 2025 forecast/actual cluster (roughly −15% to +25%) is a tiny, nearly-invisible sliver by comparison. This chart is the clearest visual explanation for why carbon's prediction intervals are proportionally so much wider than gas/electricity's throughout every other Stage 1 figure, even though 2025 itself was a comparatively tame year for carbon.
- **`outputs/figures/forecast_vs_actual_all_series.png`** — a 3-panel combined version of the three figures above (gas, electricity, carbon side by side), used as a single at-a-glance historical-context summary figure.

#### `outputs/figures/prediction_intervals_2025.png`

A 6-panel composite grid (rows = core/macro, columns = gas/electricity/carbon), essentially the six `forecast_comparison_*` figures re-rendered together with a shared visual layout, making two patterns easy to see at a glance across the whole grid simultaneously: SARIMA's prediction intervals explode in core mode but tame considerably in macro mode for every series, and carbon's prediction intervals are dramatically wider than gas or electricity's in every model/mode cell. At 566 KB this is the largest PNG in the project, appropriately so given it is a 6-way composite; no rendering artifacts (mis-clipped legends, overlapping text, or cut-off axes) were observed in any of the six panels.

#### Model-ranking polar figures (6 files: one per series × mode)

- **`outputs/figures/model_ranking_polar_gas_core.png`** — a circular "wind-rose" bar chart with 4 colour-coded quadrants (one per model: SARIMA, Prophet, LSTM, TFT), each containing 3 wedges (MAE, RMSE, SMAPE) using the realised `forecast_actual_*` metrics; wedge length is proportional to error magnitude (shorter = better). Title states "selected: LSTM". Wedge values are cross-validated exactly against `model_metrics_comparison.csv`'s `forecast_actual_*` columns for gas-core.
- **`outputs/figures/model_ranking_polar_gas_macro.png`** — same construction for gas-macro; title states "selected: LSTM".
- **`outputs/figures/model_ranking_polar_electricity_core.png`** — same construction for electricity-core; title states "selected: LSTM"; TFT's wedges (MAE 5.11, RMSE 5.43) sit nearly as short as LSTM's, visually the closest competition of any of the six charts.
- **`outputs/figures/model_ranking_polar_electricity_macro.png`** — same construction for electricity-macro; title states "selected: LSTM".
- **`outputs/figures/model_ranking_polar_carbon_core.png`** — same construction for carbon-core; title states "selected: LSTM"; SARIMA's SMAPE wedge (180.55) and RMSE wedge (24.45), and TFT's SMAPE wedge (159.76), dwarf LSTM's corresponding wedges (12.07 MAE, 15.00 RMSE, 159.26 SMAPE — note LSTM's own SMAPE is not actually small in absolute terms here, a reminder that SMAPE can behave erratically when actual values pass through or near zero, as carbon's did in April–May 2025).
- **`outputs/figures/model_ranking_polar_carbon_macro.png`** — the one chart of the six whose title states **"selected: TFT"** rather than LSTM, visually confirming carbon-macro's status as the sole case where a non-LSTM model wins; TFT's three wedges are visibly the shortest of the four models on this particular chart.

These six charts are the clearest single visual confirmation, across the whole Stage 1 output set, that SARIMA, TFT, and Prophet each occasionally balloon to 2–10× the winning model's error depending on series and mode.

#### Interactive HTML companions (6 files)

- **`outputs/figures/interactive_gas_core.html`**, **`outputs/figures/interactive_gas_macro.html`**, **`outputs/figures/interactive_electricity_core.html`**, **`outputs/figures/interactive_electricity_macro.html`**, **`outputs/figures/interactive_carbon_core.html`**, **`outputs/figures/interactive_carbon_macro.html`** — Plotly-based interactive charts, one per series/mode, combining the long-history context view (`forecast_vs_actual_*`) and the forecast-comparison view (`forecast_comparison_*`) into a single toggle/zoomable chart. Each contains traces for `Historical 2006–2024`, `Actual 2025`, and per-model `{Model}` / `{Model} 95% PI` / `{Model} PI upper` series for all four models, letting a user isolate individual models' traces interactively. These add genuine exploratory value beyond the static PNGs (no static figure combines the full historical range, all four models, and their prediction intervals in one view simultaneously) but surface no numeric values beyond what is already reported in the underlying CSVs.

---

## Stage 2a-overview — Dataset Overview

### `outputs/ukhls_dataset_overview/tables/`

#### `outputs/ukhls_dataset_overview/tables/dataset_rows_by_wave.csv`

**Structure:** 15 rows (waves a–o), columns `wave, n, year`. Row counts: a=30,169 (2009), b=30,484 (2010), c=27,751 (2011), d=25,817 (2012), e=24,325 (2013), f=24,454 (2014), g=23,033 (2015), h=21,746 (2016), i=20,048 (2017), j=19,252 (2018), k=18,139 (2019), l=16,856 (2020), m=16,156 (2021), n=21,385 (2022), o=19,586 (2023). Sum = 339,201, matching the full merged panel exactly. Counts decline broadly and monotonically from wave a through wave m (2021, the low point at 16,156), before rising again at waves n (21,385) and o (19,586) — most plausibly a refreshment/boost sample being added to the panel in its most recent two waves.

#### `outputs/ukhls_dataset_overview/tables/dataset_rows_by_year.csv`

**Structure:** 16 rows, columns `year, n`, covering 2009–2024. Counts: 2009=15,908; 2010=33,213; 2011=28,739; 2012=26,508; 2013=27,413; 2014=23,014; 2015=21,505; 2016=22,531; 2017=20,898; 2018=19,520; 2019=18,825; 2020=17,482; 2021=16,377; 2022=18,210; 2023=20,619; 2024=8,439. Sum = 339,201, also matching the full panel. This table spans one more calendar year (2024) than the wave table above, because each wave's fieldwork runs continuously across roughly 24 months and therefore straddles two calendar years — confirmed directly by the interview-month histogram in `dataset_panel_composition.png` below. This is a genuine reconciliation point rather than a data error: wave-level and year-level totals both sum to 339,201 but partition the same rows differently.

#### `outputs/ukhls_dataset_overview/tables/dataset_rows_by_region.csv` and `outputs/ukhls_dataset_overview/tables/dataset_sample_size_by_region.csv`

**Structure:** identical content (12 regions, columns `region, n`), differing only in sort order. Counts: East Midlands 24,339; East of England 28,545; London 41,318; North East 12,643; North West 34,392; Northern Ireland 21,486; Scotland 30,538; South East 40,315; South West 27,323; Wales 22,768; West Midlands 27,592; Yorkshire and the Humber 27,761. Sum = 339,020 — **181 rows (0.05%) short of the full panel**, implying a small number of rows carry an unmapped or missing region code that is silently excluded from this particular aggregation; worth a minor data-quality note even though it is immaterial to any of the project's substantive conclusions. London is the largest region by sample (41,318, a >3.2× multiple of North East's 12,643), consistent with population-representative regional stratification.

#### `outputs/ukhls_dataset_overview/tables/dataset_missingness_by_variable.csv`

**Structure:** 18 rows, columns `variable, pct_missing`. Ranked from highest to lowest: `inoutflows2` 96.857%, `inoutflows4` 96.857%, `inoutflows3` 96.857% (all three identical to 14 decimal places, strongly implying they are jointly missing across the same rows/waves rather than independently missing); `sf1_good` 56.684%; `hsval` 37.332%; `carval` 34.847%; `high_fuel_vulnerable` 24.728% and `fuel_to_income_ratio` 24.728% (identical, as expected since one is derived from the other); `total_fuel_spend` 24.106%; `qfhigh_band` 11.412%; then a cluster of low-missingness variables all under 1%: `jbstat_security` 0.960%, `health_good` 0.809%, `fiyrinvinc_dv` 0.684%, `tenure_security` 0.617%, `bill_security` 0.473%, `heatch_good` 0.420%, `financial_strain_score` 0.074%, `fihhmnnet1_dv` 0.004% (essentially complete). The three `inoutflows2/3/4` variables at ~97% missingness are near-unusable structurally (asked only in a small subset of waves) and are already excluded from the project's pooled ENERGY factor and most models for exactly this reason.

### `outputs/ukhls_dataset_overview/figures/`

- **`outputs/ukhls_dataset_overview/figures/dataset_panel_composition.png`** — a two-panel figure. Left panel, "Sample size by wave": a bar chart of waves a–o with corresponding start years labelled, matching `dataset_rows_by_wave.csv` exactly (peak near wave b/2010 at ~30k, trough at wave m/2021 at ~16k, recovering at n/o). Right panel, "Interview timing by calendar month (pooled across all waves)": a bar chart of household-wave rows by interview month 1–12, relatively flat at roughly 26,700–31,000 rows per month, with a peak in February (~31,000) and a trough in August (~26,700). This right-hand panel is the direct explanation for the wave/calendar-year reconciliation point above: since interviews for a single wave are spread fairly evenly across the whole year, a wave's rows necessarily straddle two calendar years.
- **`outputs/ukhls_dataset_overview/figures/dataset_missingness.png`** — the missingness table above, re-rendered as a horizontal bar chart with human-readable labels added this session (mapping raw variable codes to plain-English descriptions): "Used savings (past year)" 96.9%, "New borrowing, family/friends" 96.9%, "New borrowing, bank/credit card" 96.9%, "General health satisfaction (good)" 56.7%, "House value" 37.3%, "Car value" 34.8%, "High fuel vulnerability (flag)" 24.7%, "Fuel-to-income ratio" 24.7%, "Annual fuel spend" 24.1%, "Educational qualification level" 11.4%, then a green (low-missingness) cluster: "Employment status security" 1.0%, "Self-rated health (good)" 0.8%, "Individual annual investment income" 0.7%, "Housing tenure security" 0.6%, "Bill payment security" 0.5%, "Has central heating (good)" 0.4%, "Financial/psychological strain" 0.1%, "Household net monthly income" 0.0%.
- **`outputs/ukhls_dataset_overview/figures/dataset_key_distributions.png`** — a 6-panel histogram grid, "Key Variable Distributions (all waves pooled)": (1) fuel-to-income ratio — heavily right-skewed, median 0.04, almost all mass below 0.2; (2) financial/psychological strain (0–1) — right-skewed, median 0.25; (3) age — broad and roughly bimodal, median 48.50, with peaks in the 30s and again in the 60s–70s; (4) household net monthly income (£, log10) — roughly normal, median 3.40 (≈£2,512/month); (5) annual fuel spend (£, log10) — roughly normal, median 3.08 (≈£1,202/year); (6) FES Delta (Magnitude minus Current) — a visibly non-smooth, multi-modal/clumped distribution ranging roughly −9 to +3, median **−0.68**, with a large spike near 0/−1 but also distinct smaller clusters around −9, −7, and +2 to +3. This clumped shape suggests FES Delta effectively takes a small number of discrete values rather than varying continuously — worth flagging as a modelling consideration wherever FES Delta is used as a continuous regressor (as it is in the SEM-moderation test and the driver-analysis logistic regression).
- **`outputs/ukhls_dataset_overview/figures/dataset_rows_by_year.png`** — two panels: a simple bar chart of rows by calendar year (matching `dataset_rows_by_year.csv` exactly), and the same totals stacked and colour-coded by which wave contributed them, visually confirming that years 2010–2013 and 2022–2024 each draw from two adjacent waves, while wave a alone contributes to both 2009 and 2010 and wave o alone contributes to both 2023 and 2024.
- **`outputs/ukhls_dataset_overview/figures/dataset_sample_size_by_region.png`** — a real UK NUTS1-boundary choropleth map, values matching `dataset_rows_by_region.csv` exactly: London 41,318 (darkest/highest), South East 40,315, North West 34,392, Scotland 30,538, East of England 28,545, Yorkshire and the Humber 27,761, West Midlands 27,592, South West 27,323, East Midlands 24,339, Wales 22,768, Northern Ireland 21,486, North East 12,643 (lightest/lowest).

---

## Stage 2b — COR-SEM

### `outputs/ukhls_cor_sem/tables/`

#### `outputs/ukhls_cor_sem/tables/cor_sem_measurement_loadings.csv`

**Structure:** 13 rows, columns `factor, item, Estimate, Std. Err, z-value, p-value, std_loading`. Each factor's first (marker) item has a fixed Estimate=1.0 and no SE/z/p.

| Factor | Item | Std. loading | z-value |
|---|---|---|---|
| OBJECT | hsrooms (marker) | 0.5758 | — |
| OBJECT | hsbeds | 0.6705 | 271.33 |
| OBJECT | ncars | 0.6675 | 270.68 |
| OBJECT | carval | 0.4271 | 197.89 |
| OBJECT | hsval | 0.5521 | 240.09 |
| CONDITION | tenure_security (marker) | 0.3720 | — |
| CONDITION | jbstat_security | 0.4495 | 157.85 |
| CONDITION | bill_security | **0.2263** | 102.16 |
| PERSONAL | health_good (marker) | 0.5924 | — |
| PERSONAL | sf1_good | 0.8489 | 210.80 |
| PERSONAL | qfhigh_band | 0.3770 | 178.92 |
| ENERGY | fihhmnnet1_dv (marker) | 0.5832 | — |
| ENERGY | fiyrinvinc_dv | 0.3671 | 184.90 |

All non-marker items are significant at p≈0.0 (n≈339,000, so even small effects reach extreme significance — the loading magnitudes, not the p-values, are the informative quantity here). `bill_security` (0.226) is the single weakest loading in the entire measurement model, below the conventional 0.3–0.4 acceptability threshold; `tenure_security` (0.372), `qfhigh_band` (0.377), and `fiyrinvinc_dv` (0.367) also sit right at that boundary. CONDITION is, on balance, the weakest-measured of the four factors — all three of its items are at or below the 0.4 line.

#### `outputs/ukhls_cor_sem/tables/cor_sem_fit_indices.csv`

**Structure:** single row, columns `dof, chi2, chi2_p_value, cfi, tli, rmsea, gfi, aic, bic, srmr, cfi_ok, tli_ok, rmsea_ok, srmr_ok`. Values: `dof=59, chi2=9.94316, chi2_p_value=1.0, cfi=−0.53766, tli=−1.03284, rmsea=0.0, gfi=0.7843, aic=62.64627, bic=406.14541, srmr=0.07858, cfi_ok=False, tli_ok=False, rmsea_ok=True, srmr_ok=True`.

**CFI (−0.538) and TLI (−1.033) both fall well outside their mathematically valid [0,1] range**, and the pipeline's own logic correctly flags both as `False`. This coexists with a chi² (9.94 on 59 degrees of freedom) so small that `chi2_p_value` rounds to 1.0 — i.e. by the chi² test alone the model is not rejected at all, while by CFI/TLI it appears to fail badly. This combination is a known signature of a degenerate baseline/independence-model computation or of badly-scaled input variables (raw £-valued indicators like `hsval`/`carval`/`fihhmnnet1_dv` mixed with 0/1-coded security indicators) destabilizing `semopy`'s FIML likelihood — documented in the project's own source code as a known `semopy` 2.3.11 limitation. RMSEA=0.0 (a suspiciously exact floor value) and SRMR=0.079 both look nominally acceptable in isolation, but their coexistence with the invalid CFI/TLI values means the fit-index block as a whole should not be read as a clean pass.

#### `outputs/ukhls_cor_sem/tables/structural_fit_indices_baseline.csv`

**Structure:** single row, same column set as above but for the second-order structural model. Values: `dof=61, chi2=79.95503, chi2_p_value=0.0522, cfi=1.59414, tli=1.75971, rmsea=0.00096, gfi=−0.73452, aic=−88.64553, bic=233.38491, srmr=9214088570.31261, cfi_ok=True, tli_ok=True, rmsea_ok=True, srmr_ok=False`.

**This is the more severe of the two fit-index anomalies in the project.** SRMR of **9,214,088,570.31** (≈9.2 billion) is an obviously invalid value for a statistic that should normally range roughly 0–1, correctly flagged `srmr_ok=False`. GFI (−0.735) is also outside its valid [0,1] range but has no corresponding `_ok` flag to catch it. Most notably, **CFI (1.594) and TLI (1.760) both exceed the mathematical maximum of 1.0 for these indices, yet are marked `cfi_ok=True` and `tli_ok=True`** — strong evidence the pipeline's validity check tests only a lower bound (e.g. "≥0.90") and does not test an upper bound, allowing an out-of-range value to pass as "acceptable" silently. `chi2_p_value=0.0522` is borderline non-significant at the conventional 5% threshold.

#### `outputs/ukhls_cor_sem/tables/structural_paths_baseline.csv`

**Structure:** 4 rows (one per first-order factor's path from the second-order BASELINE factor), columns `first_order_factor, op, rval, Estimate, Est. Std, Std. Err, z-value, p-value`.

| Path | Estimate | Est. Std | z-value | p-value |
|---|---|---|---|---|
| OBJECT ~ BASELINE (marker) | 1.0 | **0.7115** | — | — |
| CONDITION ~ BASELINE | 132.341 | **0.9931** | 452.28 | 0.0 |
| PERSONAL ~ BASELINE | 547.253 | **0.99999741** | 454.81 | 0.0 |
| ENERGY ~ BASELINE | 183.403 | **0.99998643** | 454.80 | 0.0 |

**This is a Heywood-case-like pattern.** OBJECT loads at a plausible, sub-unity 0.7115 on the second-order BASELINE factor, but CONDITION, PERSONAL, and ENERGY all load at essentially 0.99–1.00 — i.e. three of the four first-order COR resource dimensions appear empirically indistinguishable from the overall BASELINE construct itself, which is not a substantively credible result. The very large raw (unstandardized) Estimates for these three paths (132–547, versus OBJECT's fixed marker value of 1.0) point to the same unstandardized-scale mismatch implicated in the fit-index anomalies above as the most likely underlying cause. This should be read as a numerical identification/scaling artifact, not as evidence that Condition, Personal, and Energy resources are truly interchangeable with the household's overall baseline resource stock.

#### `outputs/ukhls_cor_sem/tables/cor_sem_factor_scores.csv`

**Structure:** 339,201 rows × 6 columns (an unnamed index plus `object_score, condition_score, personal_score, energy_score, baseline_resource_score`). Descriptive statistics: `object_score` mean=−0.0538, sd=0.7297, range [−2.639, 2.946]; `condition_score` mean=−0.0018, sd=0.6711, range **[−3.963, 1.013]**; `personal_score` mean=0.0011, sd=0.8087, range [−2.029, 1.610]; `energy_score` mean=0.0001, sd=0.7946, range [−4.238, 2.311]; `baseline_resource_score` mean=−0.0004, sd=0.7156, range [−3.050, 2.392]. Small counts of missing scores remain despite FIML: `object_score` 393 NaN, `condition_score` 120 NaN, `personal_score` 2,373 NaN, `energy_score` 15 NaN, `baseline_resource_score` 1 NaN (all out of 339,201). `condition_score`'s distribution is notably asymmetric — its positive tail (max 1.013) is less than a third the length of its negative tail (min −3.963) — a >4:1 imbalance unlike the other three factors, whose positive and negative extremes are comparable in magnitude; this distributional oddity is consistent with CONDITION's weaker measurement quality documented above.

#### `outputs/ukhls_cor_sem/tables/fes_moderation_path.csv`

**Structure:** 3 rows (one per predictor in a single shared regression), columns `path, predictor, outcome, coef, std_err, t_stat, p_value, r_squared, n, significant`. All three rows share `path="COR-SEM: BASELINE x FES Delta -> fuel_to_income_ratio"`, `outcome=fuel_to_income_ratio`, `n=255,324`, `r_squared=0.1085`.

| Predictor | Coefficient | Std. Err | t-stat | p-value | Significant |
|---|---|---|---|---|---|
| baseline_score | −0.02521 | 0.00016 | −158.4866 | 0.0 | Yes |
| fes_delta | −0.00063 | 0.00005 | −11.789 | 0.0 | Yes |
| baseline_x_fes (interaction) | +0.00004 | 0.00008 | 0.5124 | **0.6084** | **No** |

**The central moderation hypothesis this regression was designed to test is not supported.** The interaction term (`baseline_x_fes`) is statistically indistinguishable from zero (p=0.608). Only the two additive main effects are significant: a higher baseline resource stock is associated with a lower fuel-to-income ratio, as theoretically expected (coefficient −0.025), and a higher FES Delta is associated with a very small reduction in the ratio (coefficient −0.0006). Together the model explains just 10.85% of variance in fuel_to_income_ratio (R²=0.1085). The honest characterization of this result is "additive effects of baseline resources and FES Delta, no significant moderation" — a meaningful correction to any framing of this analysis as demonstrating that FES *moderates* (differentially affects low- vs. high-resource households) rather than simply *adds to* the effect of baseline resources.

### `outputs/ukhls_cor_sem/figures/`

- **`outputs/ukhls_cor_sem/figures/cor_sem_loadings.png`** — four side-by-side horizontal bar charts (one per factor: OBJECT, CONDITION, PERSONAL, ENERGY), each item's absolute standardized loading plotted against a red dashed "min=0.4" reference line. Confirms visually: OBJECT — hsval 0.55, carval 0.43 (barely clears the line), ncars 0.67, hsbeds 0.67, hsrooms 0.58 (all clear); CONDITION — bill_security 0.23 (well below the line), jbstat_security 0.45 (clears), tenure_security 0.37 (just below); PERSONAL — qfhigh_band 0.38 (just below), sf1_good 0.85 (well clear), health_good 0.59 (clears); ENERGY — fiyrinvinc_dv 0.37 (just below), fihhmnnet1_dv 0.58 (clears).
- **`outputs/ukhls_cor_sem/figures/cor_sem_loadings_polar.png`** — the same 13 loadings re-rendered in the project's house-style circular bar chart, grouped and colour-coded by factor (OBJECT, CONDITION, PERSONAL, ENERGY as four coloured arcs), with each item's exact rounded loading value displayed on its own bar. Purely a re-visualization of the loadings table; no new numbers.
- **`outputs/ukhls_cor_sem/figures/structural_paths_baseline.png`** — a horizontal bar chart, "COR-SEM: second-order BASELINE structural loadings," visually confirming the near-1.0-height bars for ENERGY, PERSONAL, and CONDITION against OBJECT's distinctly shorter (~0.71) bar — the single clearest visual representation of the Heywood-case-like pattern documented above.

---

## Stage 2c — COR-CVAE

### `outputs/ukhls_cor_cvae/tables/`

#### `outputs/ukhls_cor_cvae/tables/cvae_latent_scores.csv`

**Structure:** 253,913 rows × 5 columns (an unnamed index plus `cvae_object_z, cvae_condition_z, cvae_personal_z, cvae_energy_z`), no missing values. The row count (253,913) is notably smaller than the SEM's factor-score table (339,201) because the CVAE, unlike the SEM's FIML estimator, requires complete cases and is trained/scored only on households with no missing values across its input items. Descriptive statistics: `cvae_object_z` mean=−0.0180, sd=0.1426, range [−0.610, 0.763]; `cvae_condition_z` mean=0.0348, sd=0.1146, range [−0.251, 0.918]; `cvae_personal_z` mean=0.0053, sd=0.1560, range [−1.064, 0.758]; `cvae_energy_z` mean=0.0747, sd=0.1828, range [−0.122, 1.290]. `cvae_energy_z` is markedly right-skewed (its positive extreme, 1.290, is over ten times its negative extreme's magnitude, 0.122) — a distributional shape unlike the other three, roughly symmetric latent dimensions. The naming convention (`cvae_object_z` etc.) is a positional assignment (z1→object, z2→condition, z3→personal, z4→energy) rather than a verified empirical correspondence — see the alignment table below for how well that naming actually holds up.

#### `outputs/ukhls_cor_cvae/tables/cvae_sem_alignment.csv`

**Structure:** 16 rows (4 latent dimensions × 4 SEM factors), columns `latent_dim, sem_factor, pearson_r`. Full matrix:

| Latent dim | object_score | condition_score | personal_score | energy_score |
|---|---|---|---|---|
| z1 (cvae_object_z) | **−0.7643** | −0.5018 | −0.2722 | −0.6125 |
| z2 (cvae_condition_z) | −0.4557 | **−0.8298** | −0.3446 | −0.4864 |
| z3 (cvae_personal_z) | 0.3081 | 0.4391 | **0.6713** | 0.2863 |
| z4 (cvae_energy_z) | −0.6362 | −0.5943 | −0.3583 | **−0.6478** |

(Bold = the "intended" diagonal pairing implied by the naming convention.)

**Two findings worth flagging.** First, a sign inconsistency: the four intended diagonal correlations are all reasonably strong in magnitude (|r|=0.65–0.83), but three of the four (z1↔object −0.764, z2↔condition −0.830, z4↔energy −0.648) are *negative* while z3↔personal is *positive* (+0.671). VAE latent-dimension sign is mathematically arbitrary, so this is not an error, but it does mean the `cvae_*_z` naming convention (which mirrors the SEM's `*_score` names) implies a same-direction interpretation that does not actually hold for three of the four dimensions without sign-flipping first. Second, and more substantively, alignment quality is uneven: z2 and z3 are cleanly, dedicatedly aligned to their own factor (z2's condition correlation, −0.830, is well clear of its next-highest correlate at −0.456; z3's personal correlation, 0.671, is well clear of its next-highest at 0.439), while **z1 and z4 are diffuse** — z1 correlates almost as strongly with the energy factor (−0.6125) as with its own object factor (−0.7643), and even fairly strongly with condition (−0.5018); z4 correlates nearly as strongly with object (−0.6362) and condition (−0.5943) as with its own energy factor (−0.6478). The CVAE's object and energy latent dimensions have not cleanly disentangled from one another. The weakest alignment pairs overall are z3↔energy_score (0.2863) and z1↔personal_score (−0.2722), both under |r|=0.3.

#### `outputs/ukhls_cor_cvae/tables/cvae_training_history.csv`

**Structure:** 300 rows (epoch 0–299) × 11 columns: `epoch, train_total, val_total, train_recon, val_recon, train_kl, val_kl, train_align, val_align, train_bce, val_bce`. Epoch 0: `train_total=6.0778, val_total=5.9106, train_recon=1.4424, val_recon=1.4092, train_kl=3.3054, val_kl=3.1761, train_align=0.5358, val_align=0.5377, train_bce=0.7942, val_bce=0.7876`. Epoch 299 (final): `train_total=1.6856, val_total=1.6771, train_recon=0.9917, val_recon=0.9859, train_kl=0.0679, val_kl=0.0675, train_align=0.2728, val_align=0.2703, train_bce=0.3533, val_bce=0.3534`. Global range across all 300 epochs: `train_total` [1.6856, 6.0778]; `train_kl` [0.0679, 3.3054]; `train_align` [0.2728, 0.5677] — note that align's maximum (0.5677) occurs *mid-training* (around epoch 30–40), not at epoch 0 (0.5358), meaning it rises before it falls, consistent with an annealed alignment-loss weighting schedule commonly used to avoid early posterior collapse in conditional VAEs. The train/validation gap is consistently tiny throughout (e.g. final total loss 1.6856 vs. 1.6771, under 0.5% relative difference) — no evidence of overfitting anywhere in the 300-epoch run.

#### `outputs/ukhls_cor_cvae/tables/cvae_counterfactual_fes_shift.csv`

**Structure:** 253,913 rows × 11 columns: `hidp, wave, interview_year, gor_dv, row_index, pred_prob_current, pred_prob_forecast, pred_prob_shift, dist_from_resilient_current, dist_from_resilient_forecast, dist_shift`. `gor_dv` has 97 missing values; all other columns are complete. `pred_prob_current` mean=0.2901, sd=0.0147, range [0.1994, 0.3575]; `pred_prob_forecast` mean=0.2905, sd=0.0146, range [0.2091, 0.3545]; `pred_prob_shift` mean=**+0.000457**, sd=0.00384, range **[−0.0504, +0.0585]**; `dist_from_resilient_current` mean=0.2577, sd=0.1876; `dist_from_resilient_forecast` mean=0.2459, sd=0.1752; `dist_shift` mean=**−0.01182**, sd=0.05015, range [−0.7017, +0.4203]. On average, swapping in the forecast FES conditioning produces only a tiny change in predicted vulnerability probability (+0.046 percentage points on a ~29% base rate) and a small mean *improvement* in resilience-distance (households move, on average, slightly closer to the "resilient" reference point) — directionally consistent with FES Delta's negative median (−0.68) noted in `dataset_key_distributions.png` (recalling FES Delta = Magnitude − Current, so a negative median means forecast conditions are typically milder than currently realised conditions). However, the individual-level range is wide (probability shift from −5.0 to +5.9 percentage points; distance shift from −0.70 to +0.42), meaning this near-null population average conceals substantial heterogeneity in how differently individual households respond to the counterfactual scenario.

### `outputs/ukhls_cor_cvae/figures/`

- **`outputs/ukhls_cor_cvae/figures/cvae_sem_alignment_heatmap.png`** — a signed heatmap (green-to-red diverging colormap, −1 to +1) of the alignment matrix above; the values match `cvae_sem_alignment.csv` exactly to 2 decimal places.
- **`outputs/ukhls_cor_cvae/figures/cvae_alignment_polar.png`** — the project's house-style circular bar chart, |r| per latent dimension shown as four colour-coded quadrants (one per z1–z4), each containing four wedges (one per SEM factor). This session's fix now displays each pairing's exact |r| value directly on its colourful bar (previously the values were absent from the chart entirely) — confirms z2/condition (0.83) and z1/object (0.76) as the two strongest pairings, and z1/personal and z3/energy (both in the 0.27–0.29 range) as the two weakest.
- **`outputs/ukhls_cor_cvae/figures/cvae_training_curves.png`** — a 4-panel line plot (total/reconstruction/KL/alignment losses, solid=train, dashed=validation) across all 300 epochs. Total and reconstruction losses decay smoothly and monotonically; KL decays steeply in the first ~50 epochs then flattens near-zero for the remainder (consistent with the latent posterior converging close to the prior); alignment shows the rise-then-fall hump described above, peaking near epoch 30–40 before declining to its final value. Train and validation curves are visually indistinguishable on all four panels throughout — no visible overfitting.

---

## Stage 2 (merged panel)

### `outputs/ukhls_cleaned/ukhls_panel.csv`

339,201 data rows × 62 columns (339,202 lines including the header). Confirmed full column list: `hidp, hrpid, month, gor_dv, fuelduel, xpduely, xpgasy, xpelecy, xpoily, xpsfly, fihhmngrs_dv, fihhmnnet1_dv, tenure_dv, hsbeds, ctband_dv, heatch, xphsdba, xphsdct, hsrooms, ncars, hsval, wave, tenure_security, heatch_good, bill_security, finnow, scghq1_dv, scghq2_dv, dvage, fiyrinvinc_dv, finfut_risk, jbstat_security, health_good, sf1_good, qfhigh_band, disability_free, ethnicity_group, interview_year, interview_month, carval, duelpay, elecpay, quarter, inoutflows1, inoutflows2, inoutflows3, inoutflows4, inoutflows10, htpmp, inoutflows12, total_fuel_spend, fuel_to_income_ratio, high_fuel_vulnerable, high_fuel_vulnerable_relative, financial_strain_score, gas_growth, electricity_growth, carbon_growth, fes_magnitude, fes_actual_prior_year, fes_current, fes_delta`. This is the single merged file every later stage reads from; it confirms `disability_free` and `ethnicity_group` (this session's additions) are correctly present. Notably, all six raw `inoutflows*` variables are retained even though half of them (`inoutflows2/3/4`) are 96.9% missing — a deliberate raw-data-retention choice distinct from the derived-variable pruning applied in the SEM/driver-analysis feature lists.

---

## Stage 3 — Vulnerability Identification

### `outputs/ukhls_vulnerability/tables/`

#### `outputs/ukhls_vulnerability/tables/stage3_vulnerability_scores.csv`

**Structure:** 339,201 rows, columns: an unnamed index, `fuzzy_resource_depleted, fuzzy_resource_resilient, fuzzy_vulnerable_to_loss` (three fuzzy c-means membership scores summing to 1 per row), `fuzzy_partition_coefficient` (a single constant, 0.49365691..., broadcast to every row — a global model-fit statistic, not a per-observation quantity), and `oneclass_anomaly_score`. 85,288 rows (25.1%) are NaN across all five score columns, leaving 253,913 valid rows — matching the CVAE's complete-case count exactly. Among valid rows, a hard cluster assignment via arg-max of the three fuzzy memberships gives: "resilient" dominant in 105,709 rows (41.6%), "vulnerable to loss" in 90,377 (35.6%), and "resource depleted" in only 57,827 (22.8%). `fuzzy_resource_depleted` itself has mean 0.236, min 0.0046, max 0.968. `oneclass_anomaly_score` is strongly right-skewed: mean −24.3, median −44.9, max 772.75 — most households score as broadly "typical," with a long tail of extreme anomaly scores.

#### `outputs/ukhls_vulnerability/tables/stage3_validation_against_objective_ratio.csv`

**Structure:** 2 rows, columns `method, n, pearson_r_vs_ratio, spearman_r_vs_ratio, auc_vs_high_fuel_vulnerable`.

| Method | n | Pearson r | Spearman r | AUC |
|---|---|---|---|---|
| fuzzy_resource_depleted | 253,913 | 0.2595 | 0.3362 | **0.7499** |
| oneclass_anomaly_score | 253,913 | 0.1555 | 0.0150 | 0.5666 |

Fuzzy c-means "Resource Depleted" membership clearly outperforms as a vulnerability proxy (AUC 0.75, a genuinely useful discriminator against the objective fuel/income≥10% target). The one-class SVM anomaly score is much weaker — AUC 0.567 is barely above the 0.5 chance line, and a Spearman correlation of just 0.015 indicates essentially no rank-order relationship with the objective ratio despite a somewhat higher Pearson correlation (0.156); the anomaly detector appears to be picking up a different, more general kind of statistical "unusualness" than fuel poverty specifically.

#### `outputs/ukhls_vulnerability/tables/driver_analysis_logistic_regression.csv`

**Structure:** 13 rows (one per predictor), columns `predictor, coef, odds_ratio, std_err, p_value, significant, n`; n=103,646 for all rows (the national, pooled model).

| Predictor | Coefficient | Odds ratio | p-value | Significant |
|---|---|---|---|---|
| financial_strain_score | 1.9237 | **6.85** | 0.0 | Yes |
| tenure_security | 0.3388 | 1.40 | 0.0 | Yes |
| health_good | 0.2365 | 1.27 | 0.0 | Yes |
| hsrooms | 0.0865 | 1.09 | 0.0 | Yes |
| hsbeds | 0.0515 | 1.05 | 0.0001 | Yes |
| fes_delta | 0.0222 | 1.02 | 0.182 | No |
| dvage | 0.0134 | 1.01 | 0.0 | Yes |
| heatch | −0.0411 | 0.96 | 0.436 | No |
| sf1_good | −0.0882 | 0.92 | 0.112 | No |
| bill_security | −0.2487 | 0.78 | 0.0005 | Yes |
| ncars | −0.2841 | 0.75 | 0.0 | Yes |
| qfhigh_band | −0.5363 | 0.58 | 0.0 | Yes |
| jbstat_security | −2.0342 | **0.13** | 0.0 | Yes |

Financial/psychological strain dominates every other predictor by a wide margin (OR 6.85). Two signs are worth flagging as genuinely counter-intuitive rather than glossed over: `health_good` (self-rated health "good") *raises* the odds of vulnerability (OR 1.27, significant), and both `hsrooms` and `hsbeds` (more rooms/bedrooms) also raise odds — plausibly because larger, older homes cost more to heat, and self-rated "good health" may correlate with older homeowners in exactly this kind of housing stock, though this project does not test that specific mechanism directly. `tenure_security` raising risk (OR 1.40) is similarly non-obvious at first glance. Only `fes_delta`, `heatch`, and `sf1_good` are not statistically significant.

#### `outputs/ukhls_vulnerability/tables/driver_analysis_by_region.csv`

**Structure:** 156 rows (13 predictors × 12 regions), same columns as the national table above plus a `region` column. `financial_strain_score` is the top-ranked driver in every one of the 12 regions, but its magnitude varies enormously: highest in Northern Ireland (OR 14.67, n=1,879) and South East (OR 14.57, n=12,880), lowest in London (OR 2.86, n=15,231) — over a 5× spread in effect size by geography. `jbstat_security` is consistently and strongly protective everywhere (OR range approximately 0.10–0.21), the most universal driver after financial strain. Regional idiosyncrasies include Northern Ireland's `health_good` flipping to a negative, non-significant coefficient (−0.163) unlike the national/most-region pattern, and `tenure_security` being non-significant in about half the regions while large and significant in North East (OR 3.24) specifically. Smaller-sample regions (Northern Ireland n=1,879, North East n=4,577) carry correspondingly larger standard errors (e.g. NI's financial_strain_score std_err=0.662 vs. London's 0.269) and should be read with that extra uncertainty in mind.

#### `outputs/ukhls_vulnerability/tables/policy_vulnerability_by_wave.csv`

**Structure:** 15 rows (waves a–o), columns `wave, year, pct_vulnerable, n`. Trajectory: 11.57% (2009) declining to a trough of 5.09% (2020, wave l, n=16,856), then spiking sharply to 6.31% (2021), 10.83% (2022), and 10.62% (2023) — a clear U-shape with the cost-of-living-crisis spike as the dominant late-series feature.

#### `outputs/ukhls_vulnerability/tables/policy_vulnerability_by_region.csv`

**Structure:** 12 rows, columns `region, pct_vulnerable, n`, ranked descending: Northern Ireland 15.62% (n=21,486); West Midlands 9.75% (n=27,592); Wales 9.12% (n=22,768); North West 8.86% (n=34,392); North East 8.51% (n=12,643); Scotland 8.39% (n=30,538); Yorkshire and the Humber 8.37% (n=27,761); East Midlands 8.05% (n=24,339); London 6.88% (n=41,318); East of England 6.50% (n=28,545); South West 6.08% (n=27,323); South East 5.96% (n=40,315). Northern Ireland is a stark outlier, more than 5.8 percentage points above the next-highest region and roughly 2.6× the rate of the lowest.

#### `outputs/ukhls_vulnerability/tables/policy_vulnerability_by_ethnicity.csv`

**Structure:** 11 rows, columns `group, pct_vulnerable, n` (ethnicity of the household reference person): Black Caribbean 14.06% (n=6,751); Pakistani 13.37% (n=6,797); Any other Black background 13.18% (n=426); Black African 10.52% (n=6,748); Other ethnic group 10.36% (n=2,321); Mixed/multiple ethnic groups 10.19% (n=4,816); Bangladeshi 9.50% (n=3,930); Indian 8.67% (n=9,178); White 7.76% (n=265,950); Chinese 7.26% (n=1,408); Any other Asian background 6.66% (n=2,570). White households dominate the sample (265,950 of ~339k rows). Bangladeshi ranking only 7th here, despite ranking 1st in JRF's independently-published income-poverty statistics, is the single largest cross-source divergence found anywhere in this project (see Section 6/external validation below).

#### `outputs/ukhls_vulnerability/tables/policy_vulnerability_by_disability.csv`

**Structure:** 2 rows, columns `group, pct_vulnerable, n`: "Contains disabled adult" 12.86% (n=14,119) vs. "No disabled adult" 10.33% (n=15,840). Total n (29,959) is far smaller than the region/ethnicity tables' (~254k) because the disability flag is observed only for responding adults (indresp), not the full household sample.

#### `outputs/ukhls_vulnerability/tables/policy_vulnerability_by_tenure.csv`

**Structure:** 5 rows, columns `group, pct_vulnerable, n`: Other 13.10% (n=1,173); Social renting 11.28% (n=58,904); **Owned outright 9.87% (n=118,208)**; Private renting 8.50% (n=43,788); Buying with mortgage 4.15% (n=115,035). Owned outright ranking above Private renting is the notable finding here — counter to the typical income-poverty severity ordering (where outright ownership usually ranks least poor), consistent with a fuel-to-income measure picking up older, asset-rich-but-heating-cost-exposed outright owners that an income-only lens does not flag.

#### `outputs/ukhls_vulnerability/tables/policy_vulnerability_by_fes_tier.csv`

**Structure:** 3 rows, columns `fes_tier, pct_vulnerable, mean_fes, n`: Low stress 7.82% (mean_fes=−2.51, n=86,042); Moderate stress 8.05% (mean_fes=−0.59, n=85,186); High stress 7.94% (mean_fes=+0.36, n=84,096). Essentially flat across terciles of national FES stress — household-level vulnerability does not discriminate meaningfully by the macro price-stress environment in this simple three-way split, consistent with `fes_delta`'s non-significance in the driver-analysis regression above.

#### `outputs/ukhls_vulnerability/tables/policy_vulnerability_by_region_map.csv`

**Structure:** the same data as `policy_vulnerability_by_region.csv`, reordered geographically (North East through Northern Ireland) rather than by rank, feeding the choropleth map figure rather than the ranked bar chart.

#### `outputs/ukhls_vulnerability/tables/policy_map_baseline_resource_by_region.csv` and `outputs/ukhls_vulnerability/tables/policy_map_financial_strain_by_region.csv`

**Structure:** both files contain the identical 12-region set with both `baseline_resource_score` and `financial_strain_score` columns present in each — **these two files are byte-identical**, a data-export redundancy worth flagging (harmless in practice, since each downstream figure correctly reads only its own intended column, but worth cleaning up). `baseline_resource_score` ranges from South East (0.168, highest) to North East (−0.134, lowest); London is slightly negative (−0.065) despite its economic profile, more plausibly reflecting a needs-adjusted resource construct than raw income. `financial_strain_score`'s true range is very narrow — 0.2659 (Scotland, lowest) to 0.2818 (West Midlands, highest), only a 0.016-point, ~6% relative spread — which its corresponding choropleth map visually over-dramatizes with full colour saturation (see figure notes below).

#### `outputs/ukhls_vulnerability/tables/policy_temporal_change_by_region.csv` and `outputs/ukhls_vulnerability/tables/policy_temporal_change_map.csv`

**Structure:** identical data (12 rows, columns `region, pct_vulnerable_early, pct_vulnerable_late, pct_point_change`), differing only in row order. Comparing waves a–e (2009–2013) to waves k–o (2019–2023): Northern Ireland improved by far the most (20.76%→10.54%, **−10.21 percentage points**); West Midlands −2.37pp, Wales −2.36pp, North East −3.38pp, North West −2.22pp all improved materially; **South West is the only region to get worse** (6.45%→6.68%, +0.23pp); South East is nearly flat (−0.13pp).

#### `outputs/ukhls_vulnerability/tables/policy_vulnerability_region_year_heatmap.csv`

**Structure:** 12 regions × 17 year columns (2009–2024). **Data-quality flag:** Northern Ireland's 2024 cell reads exactly 0.000000%, against a neighbouring 2022–2023 trend of roughly 11–12% and every other region's plausible 5.7–12.3% range in 2024 — almost certainly a sparse-or-zero-sample artifact for that specific cell rather than a genuine finding, and should be caveated or excluded from any presentation of this table rather than read at face value. Otherwise, Northern Ireland dominates the early years of this heatmap (23% in 2009, 22% in 2011/2012, 21% in 2013 — the darkest cells on the entire grid) before falling into the same range as other regions from around 2016 onward, visually corroborating the large temporal improvement reported above.

#### JRF external-validation benchmark tables (4 files)

- **`outputs/ukhls_vulnerability/tables/jrf_poverty_benchmark_region.csv`** — 12 rows, JRF's own relative-poverty rate (AHC) by UK nation/region, averaged 2021/22–2022/23 (Table 6, p.51): West Midlands 27 (highest); North West 25; London 24; Yorkshire and the Humber 23; Scotland, North East, Wales all 21 (tied); East Midlands 20; South West, South East both 19 (tied); East of England 18; **Northern Ireland 17 (lowest of all 12)**.
- **`outputs/ukhls_vulnerability/tables/jrf_poverty_benchmark_ethnicity.csv`** — 6 rows, only ethnicity groups with a number explicitly stated in JRF's report text (p.9/42), 2020/21–2022/23: Bangladeshi 56 (highest); Pakistani 49; Black African 40; Any other Asian background 34; Black Caribbean 30; White 19 (lowest). Indian, Chinese, Mixed/multiple, Any other Black background, and Other ethnic group are deliberately excluded — JRF shows these only in a chart (Figure 13/25) with no stated rate to cite, and the project's own methodology avoids reading numbers off chart pixels.
- **`outputs/ukhls_vulnerability/tables/jrf_poverty_benchmark_disability.csv`** — 2 rows, JRF Table 8 (p.67): "Contains disabled adult" 29; "No disabled adult" 19.
- **`outputs/ukhls_vulnerability/tables/jrf_poverty_benchmark_tenure.csv`** — 4 rows, JRF Table 10 (p.95), 2022/23: Social renting 44 (highest); Private renting 35; Owned outright 14; Buying with mortgage 10 (lowest).

#### External-validation comparison tables (4 files)

- **`outputs/ukhls_vulnerability/tables/external_validation_region_comparison.csv`** — 12 rows, merging `policy_vulnerability_by_region.csv` with the JRF region benchmark, plus computed rank columns. Correlation across all 12 regions: Pearson r=−0.10, Spearman ρ=0.33. **Excluding Northern Ireland (n=11): Pearson r=0.68, Spearman ρ=0.73** — Northern Ireland alone flips the sign of the linear correlation and roughly doubles the rank correlation, the single most important statistic in the whole external-validation exercise.
- **`outputs/ukhls_vulnerability/tables/external_validation_ethnicity_comparison.csv`** — 6 rows. Correlation: Pearson r=0.27, Spearman ρ=0.14 (n=6) — the weakest of the four dimensions. Bangladeshi is JRF's #1 most-poor ethnic group (56%) but only #4 of 6 on this project's measure (9.50%); Black Caribbean is the mirror-image divergence, #1 here (14.06%) but #5 of 6 on JRF (30%).
- **`outputs/ukhls_vulnerability/tables/external_validation_disability_comparison.csv`** — 2 rows. Rank order agrees exactly, but with only 2 categories the correlation coefficient is mathematically undefined (reported and displayed as NaN in the corresponding figures).
- **`outputs/ukhls_vulnerability/tables/external_validation_tenure_comparison.csv`** — 4 rows. Correlation: Pearson r=0.68, Spearman ρ=0.80 (n=4) — the strongest external agreement of the four dimensions. Owned outright is the one rank-swap (2nd here vs. 3rd on JRF), consistent with the fuel-vs-income divergence discussed above.

#### Northern Ireland oil-heating evidence tables (2 files)

- **`outputs/ukhls_vulnerability/tables/ni_oil_heating_evidence_by_region.csv`** — 12 rows, columns `region, pct_using_oil_heating, pct_vulnerable, n`: Northern Ireland 71.2% oil-heating usage, 15.6% vulnerable (n=21,486) — the next-highest region for oil-heating usage is Wales at only 9.8% (n=22,768), a **7.3× gap**; South West 9.0%/6.1%; East of England 7.7%/6.5%; Scotland 6.7%/8.4%; East Midlands 3.9%/8.0%; South East 3.8%/6.0%; West Midlands 3.6%/9.8%; Yorkshire and the Humber 2.3%/8.4%; North East 1.8%/8.5%; North West 1.5%/8.9%; London 0.1%/6.9%. This is the key exogenous variable explaining Northern Ireland's outlier position in every other regional table in this project.
- **`outputs/ukhls_vulnerability/tables/ni_oil_heating_evidence_within_ni.csv`** — 2 rows, a controlled within-Northern-Ireland comparison by oil-heating status: `has_oil=False` — mean annual fuel spend £1,243, 12.3% vulnerable, n=6,183; `has_oil=True` — mean annual fuel spend £2,017, **20.9% vulnerable**, n=15,303 (6,183+15,303=21,486, matching Northern Ireland's total n exactly). Oil-heating NI households spend 62% more annually on fuel and are vulnerable at nearly double the rate of non-oil NI households in the same wave and region — this within-region comparison isolates oil-heating exposure from every other NI-specific confound and is the strongest single piece of evidence for the causal story that oil-market volatility (lump-sum purchased, price-volatile, outside Ofgem's price cap) drives Northern Ireland's outlier status.

### `outputs/ukhls_vulnerability/figures/` (22 files, each with a matching vector `.pdf`)

- **`outputs/ukhls_vulnerability/figures/driver_analysis_by_region.png`** — a horizontal bar chart, "Does the strongest national driver hold everywhere?", showing `financial_strain_score`'s odds ratio by region sorted descending: Northern Ireland (~14.7) and South East (~14.6) highest, East Midlands (9.6), North East (8.8), North West/Scotland/West Midlands (~7.9), Yorkshire (7.5), South West (7.2), East of England (6.6), Wales (5.3), and **London lowest (~2.9)** — a roughly 5× spread across regions, with a reference line at OR=1.
- **`outputs/ukhls_vulnerability/figures/policy_driver_odds_ratios.png`** — the national logistic-regression forest/bar plot, colour-coded green (protective) / red (risk-raising) / grey (non-significant), with human-readable predictor labels. Financial/psychological strain's bar is overwhelmingly the longest (OR≈6.85). Confirms the sign pattern in `driver_analysis_logistic_regression.csv` exactly: green = employment status security, educational qualification, number of cars, bill payment security; grey = general health satisfaction, has central heating, FES Delta; red = age, number of bedrooms/rooms, self-rated health (good), housing tenure security, financial/psychological strain.
- **`outputs/ukhls_vulnerability/figures/policy_vulnerability_by_wave.png`** — a line chart, 2009–2023 (waves a–o), each point annotated with its calendar year, visually confirming the U-shape and cost-of-living-crisis spike. **The chart's title states "2009-2024," but the last plotted data point (wave o) is labelled 2023 and no 2024 data exists in the underlying wave-level table** — a minor title/labelling inaccuracy worth correcting.
- **`outputs/ukhls_vulnerability/figures/policy_vulnerability_by_region.png`** — a ranked horizontal bar chart of the 12 regions, coloured red (above median) or green (below median) at a visual split between Scotland (8.39%, red) and Yorkshire and the Humber (8.37%, green) — an essentially coin-flip threshold given how close those two values are. Northern Ireland's bar is roughly 60% longer than the next-longest (West Midlands).
- **`outputs/ukhls_vulnerability/figures/policy_vulnerability_by_ethnicity.png`** — the ethnicity breakdown as a ranked horizontal bar chart, red/green split at the median; confirms Black Caribbean highest and Any other Asian background lowest.
- **`outputs/ukhls_vulnerability/figures/policy_vulnerability_by_disability.png`** — a 2-bar chart: red "Contains disabled adult" (12.86%) vs. green "No disabled adult" (10.33%).
- **`outputs/ukhls_vulnerability/figures/policy_vulnerability_by_tenure.png`** — a 5-bar chart; "Owned outright" is coloured red (above median), ranking visually just below Social renting and above Private renting — the counter-intuitive tenure ordering discussed above is directly visible here.
- **`outputs/ukhls_vulnerability/figures/policy_vulnerability_by_fes_tier.png`** — a 3-bar chart with percentage labels printed directly on the bars (7.8%, 8.0%, 7.9%), visually confirming the near-flat, non-monotonic pattern across low/moderate/high national FES-stress terciles.
- **`outputs/ukhls_vulnerability/figures/policy_vulnerability_region_year_heatmap.png`** — a 12×17 (region × year, 2009–2024) annotated heatmap, green (low, ~0–5%) through red (high, ~20%+). Northern Ireland's row is visually the only band of dark-red cells in the early years, fading to green/yellow like every other region by around 2016, before its rightmost (2024) cell renders as a stark, isolated dark-green "0" against its own row's otherwise-elevated recent trend — visually confirming the likely data-sparsity artifact flagged in the underlying CSV.
- **`outputs/ukhls_vulnerability/figures/policy_vulnerability_by_region_map.png`** — a real UK NUTS1 choropleth (ONS Open Geography Portal boundaries), each region labelled with its abbreviation and percentage, colour scale roughly 6–14%+. Northern Ireland renders as a solid, visually isolated dark-maroon polygon; the next-darkest region (West Midlands, orange) is several shades lighter. London's label uses a leader line to point to its small polygon, following the same convention used elsewhere in the project's mapping code.
- **`outputs/ukhls_vulnerability/figures/policy_map_baseline_resource_by_region.png`** — a diverging blue(low)-to-teal/green(high) choropleth of `baseline_resource_score`, roughly −0.13 to +0.17; South East (0.17) and East of England/South West (~0.08–0.09) are darkest teal (highest resource); North East (−0.13) is lightest/pinkish (lowest resource); London's slightly negative score (−0.07) may read as counter-intuitive given the city's income profile but reflects a needs-adjusted latent construct rather than raw income.
- **`outputs/ukhls_vulnerability/figures/policy_map_financial_strain_by_region.png`** — a red-scale choropleth of `financial_strain_score`, with values printed to two decimal places on nearly every region (mostly "0.27" or "0.28"). Given the underlying true range is only 0.2659–0.2818 (a spread of just 0.016), the colour intensities exaggerate a very small underlying difference — West Midlands, Wales, Northern Ireland, London, and Yorkshire appear most "red," Scotland and South East appear palest, but the practically meaningful difference between the reddest and palest region is small; a genuine readability/interpretation caveat for anyone using this map at face value.
- **`outputs/ukhls_vulnerability/figures/policy_temporal_change_map.png`** — a diverging green(improved)-to-red(worsened) choropleth of percentage-point change, early vs. late waves. Northern Ireland is the only deep-green (most-improved, −10.2pp) region; South West is the only region tinted toward "worse" (pale yellow-green, +0.2pp — essentially neutral but technically positive). **The chart's title states "waves k-o [2019-2024]," which is again one year past the data's actual 2023 endpoint** — the same title-labelling issue as the by-wave figure above.
- **`outputs/ukhls_vulnerability/figures/external_validation_wave_trend.png`** — a line chart of this project's own by-wave vulnerability rate (2009–2023), with a shaded "cost-of-living crisis" window (2021–2023) and a title that quotes JRF's own report (pp.19, 105–108) directly: JRF's slower relative-poverty measure was "broadly flat" 2021/22→2022/23, while its faster cost-of-living hardship tracker peaked at 75% of low-income households going without essentials in October 2022, easing to 69% by October 2024. This project's own line visually reproduces that same sharp-spike-then-partial-easing shape (6.31%→10.83%→10.62% across 2021/2022/2023).
- **`outputs/ukhls_vulnerability/figures/external_validation_region_bars.png`** — a dual-axis grouped bar chart (orange bars = this project's fuel-vulnerability rate, navy bars = JRF's income-poverty rate) for all 12 regions, sorted by JRF rate; the title states the two correlation results ("All (n=12): Spearman ρ=0.33 | Pearson r=−0.10" and "Excl. Northern Ireland (n=11): Spearman ρ=0.73 | Pearson r=0.68") directly in the chart itself.
- **`outputs/ukhls_vulnerability/figures/external_validation_region_scatter.png`** — a labelled scatter plot of the same 12 regions (JRF rate on x, our rate on y) with a dashed best-fit line; the region-label-collision fix applied this session is visible here — North East, Scotland, and Wales cluster tightly around JRF=21% and their labels now stagger apart with thin leader lines back to their markers rather than overlapping illegibly as they did before the fix.
- **`outputs/ukhls_vulnerability/figures/external_validation_ethnicity_bars.png`** — the dual-axis bar-chart companion for ethnicity (6 groups with a JRF-stated rate), title states "Spearman ρ=0.14 | Pearson r=0.27 (n=6)".
- **`outputs/ukhls_vulnerability/figures/external_validation_ethnicity_scatter.png`** — the scatter-plot companion; Bangladeshi and Black Caribbean are visually the two points furthest from the best-fit line, the clearest visual representation of the ethnicity divergence discussed throughout this catalog.
- **`outputs/ukhls_vulnerability/figures/external_validation_disability_bars.png`** — the dual-axis bar-chart companion for disability (2 groups); title literally displays **"Spearman ρ=nan | Pearson r=nan (n=2)"** — mathematically correct for a 2-point comparison, but this reads as an error message rather than an expected result to an unprepared viewer, and would benefit from an explanatory caption if shown outside a technical audience.
- **`outputs/ukhls_vulnerability/figures/external_validation_disability_scatter.png`** — the scatter-plot companion; the same "ρ=nan/r=nan" title issue applies, and a dashed best-fit line is still drawn through the 2 points despite the correlation coefficient being formally undefined — this could be misread as showing a meaningful fitted trend from only two data points.
- **`outputs/ukhls_vulnerability/figures/external_validation_tenure_bars.png`** — the dual-axis bar-chart companion for tenure (4 groups); title states "Spearman ρ=0.80 | Pearson r=0.68 (n=4)" — the strongest external-validation result of the four dimensions.
- **`outputs/ukhls_vulnerability/figures/external_validation_tenure_scatter.png`** — the scatter-plot companion; "Owned outright" is visibly the one point sitting well above the best-fit line — much more vulnerable on this project's fuel measure than its comparatively low JRF income-poverty rate would predict, the clearest single-point illustration of the tenure divergence discussed throughout this catalog.

---

## Stage 4 — Policy Geography Maps

### `outputs/ukhls_policy_maps/tables/`

#### `outputs/ukhls_policy_maps/tables/map1_resource_stress_hotspot.csv`

**Structure:** 12 rows, columns `region, baseline_resource_mean, vulnerable_pct, n, baseline_tertile, vulnerable_tertile, policy_tier`.

| Region | Baseline resource (mean) | % vulnerable | n | Policy tier |
|---|---|---|---|---|
| Northern Ireland | −0.0585 | **15.62%** | 21,486 | **Low resource / High vulnerability** |
| West Midlands | −0.0191 | 9.75% | 27,592 | Mid resource / High vulnerability |
| Wales | −0.0474 | 9.12% | 22,768 | Mid resource / High vulnerability |
| North West | −0.0459 | 8.86% | 34,392 | Mid resource / High vulnerability |
| North East | −0.1340 | 8.51% | 12,643 | Low resource / Mid vulnerability |
| Scotland | −0.0373 | 8.39% | 30,538 | Mid resource / Mid vulnerability |
| Yorkshire and the Humber | −0.0583 | 8.37% | 27,761 | Low resource / Mid vulnerability |
| East Midlands | −0.0080 | 8.05% | 24,339 | High resource / Mid vulnerability |
| London | −0.0655 | 6.88% | 41,318 | Low resource / Low vulnerability |
| East of England | 0.0915 | 6.50% | 28,545 | High resource / Low vulnerability |
| South West | 0.0843 | 6.08% | 27,323 | High resource / Low vulnerability |
| South East | **0.1683** | **5.96%** | 40,315 | High resource / Low vulnerability |

Northern Ireland is the only region classified "Low resource / High vulnerability" — the tier the accompanying map treats as a cash-transfer-priority zone — even though its baseline resource score (−0.0585) is roughly mid-pack, not the very worst (North East's −0.1340 is lower); NI's placement in this tier is driven by its vulnerability rate, not its resource score. The Midlands/North/Wales form a contiguous "Mid resource / High vulnerability" band (a structural/infrastructure-priority tier, distinct from NI's cash-transfer tier), while the three southern English regions (South East, South West, East of England) cluster tightly as "High resource / Low vulnerability."

#### `outputs/ukhls_policy_maps/tables/policy_map2_fuzzy_membership.csv`

**Structure:** 12 rows, columns `region, mean_vulnerable_to_loss, pct_near_boundary, n`.

| Region | Mean fuzzy membership | % near boundary | n |
|---|---|---|---|
| Yorkshire and the Humber | 0.3694 | 27.16% | 27,761 |
| Wales | 0.3690 | 26.02% | 22,768 |
| North East | 0.3685 | 27.83% | 12,643 |
| East Midlands | 0.3668 | 26.14% | 24,339 |
| West Midlands | 0.3643 | 25.41% | 27,592 |
| Scotland | 0.3642 | 24.70% | 30,538 |
| North West | 0.3593 | 26.06% | 34,392 |
| East of England | 0.3561 | 22.36% | 28,545 |
| South West | 0.3553 | 21.67% | 27,323 |
| London | 0.3512 | 24.15% | 41,318 |
| South East | 0.3473 | 21.70% | 40,315 |
| Northern Ireland | **0.3430** | **8.37%** | 21,486 |

Northern Ireland has the *lowest* mean fuzzy membership (0.343) despite having by far the highest hard-threshold vulnerability prevalence elsewhere in this project — an apparent contradiction resolved by the "% near boundary" column: NI's near-boundary share (8.37%) is less than a third of every other region's (21.7%–27.8%). NI's fuzzy-membership distribution is more polarized/bimodal than every other region's — its households sort more decisively into "clearly vulnerable" or "clearly not," with far fewer borderline cases — while every GB region clusters tightly together (0.347–0.369, a narrow ~2.2-percentage-point band), showing membership is fairly uniform across Great Britain with NI standing apart on both the level and the shape of its distribution.

#### `outputs/ukhls_policy_maps/tables/map3_vulnerability_vector_shift.csv`

**Structure:** 12 rows, columns `region, mean_current, mean_forecast, mean_shift, n` (using each household's own realised FES vs. the shared forecast shock — the CVAE counterfactual — a simulation, not observed data). Total n across regions = 231,840.

| Region | Current prob. | Forecast prob. | Shift | n |
|---|---|---|---|---|
| South East | 0.29090 | 0.29148 | **+0.000573 (largest)** | 31,315 |
| South West | 0.29157 | 0.29212 | +0.000547 | 19,610 |
| East of England | 0.29074 | 0.29126 | +0.000521 | 21,150 |
| North West | 0.29000 | 0.29047 | +0.000467 | 28,537 |
| East Midlands | 0.29142 | 0.29188 | +0.000465 | 19,376 |
| Wales | 0.29085 | 0.29131 | +0.000460 | 17,279 |
| London | 0.28570 | 0.28615 | +0.000446 | 32,524 |
| West Midlands | 0.29020 | 0.29062 | +0.000412 | 22,106 |
| Yorkshire and the Humber | 0.29004 | 0.29045 | +0.000412 | 22,976 |
| North East | 0.28987 | 0.29026 | +0.000391 | 10,579 |
| Northern Ireland | 0.28837 | 0.28871 | +0.000341 | 6,131 |
| Scotland | 0.29226 | 0.29256 | **+0.000299 (smallest)** | 22,233 |

Every region shifts positively (rising risk) under the shared forecast shock — no region shows a decline, consistent with the map's "red = rising risk" framing (no green arrows anywhere on the map). The magnitudes are tiny in absolute terms (0.03–0.06 percentage points). South East has the *largest* shift despite having the *lowest* current baseline vulnerability prevalence anywhere in the project (5.96% in the region table above), while Scotland has the *smallest* shift despite mid-pack baseline vulnerability — the shift magnitude is essentially decoupled from, if anything mildly inversely related to, current prevalence, suggesting South East/South West/East households' own FES trajectories are comparatively more sensitive to the shared shock scenario, while Scotland and Northern Ireland are comparatively insulated from it. London's current level (0.2857) is the lowest of all 12 regions in this particular table (this differs slightly from the region table's percentage because this is a continuous predicted probability, not a binary-threshold prevalence).

### `outputs/ukhls_policy_maps/figures/`

- **`outputs/ukhls_policy_maps/figures/policy_map1_resource_stress_hotspot.png`** — two side-by-side real UK region maps (mean baseline resource score; regional vulnerability prevalence) plus a 3×3 bivariate policy-tier legend combining both, confirming Northern Ireland's unique "Low resource / High vulnerability" tier classification visually.
- **`outputs/ukhls_policy_maps/figures/policy_map2_fuzzy_membership.png`** — a choropleth of mean fuzzy "vulnerable to loss" membership; Northern Ireland is visually the lowest-shaded region, though the more analytically important number (its unusually low near-boundary share) is not itself mapped in this figure and only appears in the underlying table.
- **`outputs/ukhls_policy_maps/figures/policy_map3_vulnerability_vector_shift.png`** — arrows drawn at each region's real geographic centroid, all pointing the same direction (rising risk) but of varying length; South East's arrow is the longest on the map, Scotland's the shortest, matching the table above exactly. London's arrow/label uses the same leader-line nudge as other London-labelled maps in this project, since London's small geographic area sits inside South East and a same-length arrow at its true centroid would otherwise collide with its neighbour.

---

## Stage 5 — Forward Vulnerability Prediction

### `outputs/ukhls_forward_prediction/tables/`

#### `outputs/ukhls_forward_prediction/tables/stage5_transition_pairs.csv`

**Structure:** 261,759 rows, columns `as_of_wave, target_wave, hrpid, gor_dv, interview_year_t, interview_month_t, interview_year_t1, high_fuel_vulnerable_t1, fuel_to_income_ratio_t1, object_score, condition_score, personal_score, energy_score, financial_strain_score, dvage, heatch, fes_magnitude`. Covers all 14 consecutive wave-to-wave transitions (a→b through n→o). Row counts per transition: a→b 21,886; b→c 24,404; c→d 22,961; d→e 22,034; e→f 19,771; f→g 20,001; g→h 19,288; h→i 17,954; i→j 17,083; j→k 16,234; k→l 15,021; l→m 14,253; m→n 13,725; n→o 17,144 — declining steadily from b→c (the peak) to m→n (the trough), reflecting cumulative sample attrition through the panel, before an uptick at n→o consistent with a refreshment/boost sample entering in the most recent wave. `high_fuel_vulnerable_t1` distribution: 184,940 not-vulnerable, 14,624 vulnerable, 62,195 missing (attrition/non-response on the target wave) — a raw ~7.3% vulnerability rate among non-missing target labels, consistent with the ~6–16% regional prevalence range seen throughout Stage 3/4. Train/validation reconciliation: training uses the 12 earliest transitions (a→b … l→m), 230,890 raw rows reducing to 177,408 after listwise-deleting rows with any missing predictor or target — an exact match to `n_train` in `stage5_validation_metrics.csv` below; validation uses the 2 held-out transitions (m→n, n→o), 30,869 raw rows reducing to 21,161 complete cases — an exact match to `n_validation`.

#### `outputs/ukhls_forward_prediction/tables/stage5_validation_metrics.csv`

**Structure:** single row, columns `validation_pairs, n_train, n_validation, auc_vs_high_fuel_vulnerable_t1, pearson_r_vs_fuel_to_income_ratio_t1`. Values: `validation_pairs="m->n, n->o"`, `n_train=177,408`, `n_validation=21,161`, **`auc=0.7581`**, **`pearson_r=0.2653`**. An AUC of 0.758 on genuinely held-out, walk-forward (not fit) transitions is a solid discriminative result for a logistic model predicting a binary next-wave outcome from cross-sectional predictors — comfortably above chance (0.5) and in the acceptable-to-good range by conventional social-science standards. The much weaker Pearson correlation (0.265) against the continuous `fuel_to_income_ratio_t1` indicates the model is considerably better at rank-ordering/classifying who crosses the high-vulnerability threshold than at explaining variance in the continuous ratio itself.

#### `outputs/ukhls_forward_prediction/tables/stage5_driver_coefficients.csv`

**Structure:** 8 rows, columns `predictor, coef, odds_ratio, p_value, significant` (final model, refit on all 14 known transitions).

| Predictor | Coefficient | Odds ratio | p-value | Significant |
|---|---|---|---|---|
| **fes_magnitude** | 3.9513 | **52.00** | <0.001 | Yes |
| financial_strain_score | 1.5808 | 4.86 | <0.001 | Yes |
| object_score | 0.3559 | 1.43 | <0.001 | Yes |
| dvage | 0.0180 | 1.018 | <0.001 | Yes |
| heatch | −0.0503 | 0.951 | 0.1491 | No |
| personal_score | −0.0662 | 0.936 | <0.001 | Yes |
| condition_score | −0.2661 | 0.766 | <0.001 | Yes |
| energy_score | **−1.1798** | **0.307** | <0.001 | Yes |

`fes_magnitude`'s odds ratio of 52.0 dominates the model by a wide margin — a one-unit increase is associated with a greater-than-50-fold increase in the odds of becoming high-fuel-vulnerable next wave. This is structurally significant: it is the entire mechanism by which the forecast shock enters Stage 5's predictions (see `stage5_forward_predictions.csv` below, where this signal is a single constant applied to every household — meaning it cannot, on its own, differentiate between households or regions). `energy_score`'s negative coefficient (OR 0.307) is the most counter-intuitive-looking sign at first glance — higher "energy score" *reducing* vulnerability odds — plausible if `energy_score` indexes energy-efficiency/asset adequacy rather than raw energy spending, in which case a protective sign is sensible, but this warrants a precise definitional check before quoting externally. `condition_score` is also negative (OR 0.766), plausibly for the same asset/quality-index reason. `heatch` is the only non-significant predictor (p=0.149), with an odds ratio (0.951) close to 1 — no measurable independent effect once the other seven covariates are controlled for. `dvage` has a small but highly significant positive effect (OR 1.018 per year) — a modest, real age gradient in forward risk.

#### `outputs/ukhls_forward_prediction/tables/stage5_forward_prediction_by_month.csv`

**Structure:** 12 rows, columns `interview_month, mean_predicted_probability, n`, total n=19,140.

| Month | Mean predicted probability | n |
|---|---|---|
| Jan | 7.02% | 1,603 |
| Feb | 7.31% | 1,652 |
| Mar | 6.81% | 1,644 |
| **Apr** | **7.36% (highest)** | 1,616 |
| May | 6.84% | 1,589 |
| Jun | 6.78% | 1,584 |
| Jul | 6.69% | 1,521 |
| Aug | 6.60% | 1,598 |
| **Sep** | **6.44% (lowest)** | 1,577 |
| Oct | 7.02% | 1,601 |
| Nov | 7.06% | 1,592 |
| Dec | 6.72% | 1,563 |

April is peak-risk (7.36%) and September the trough (6.44%) — a spread of 0.92 percentage points, roughly 14% relative. Winter/early-spring target months (Jan, Feb, Apr, Oct, Nov, all ≥7.0%) generally run higher than mid/late-summer months (Jul, Aug, Sep, all ≤6.7%), plausibly tracking UKHLS interview timing relative to heating-season bills and income patterns feeding into each household's own characteristics combined with the fixed forecast-shock term.

#### `outputs/ukhls_forward_prediction/tables/stage5_forward_prediction_map.csv`

**Structure:** 12 rows, columns `region, mean_predicted_probability, n`, n summing to 19,134 of the full 19,140 households (6 have missing/unmapped region — see the household-level file below).

| Region | Mean predicted probability | n |
|---|---|---|
| **Northern Ireland** | **7.74% (highest)** | 1,083 |
| Wales | 7.51% | 1,185 |
| North West | 7.37% | 1,976 |
| Yorkshire and the Humber | 7.33% | 1,638 |
| West Midlands | 7.11% | 1,603 |
| North East | 7.00% | 706 |
| East of England | 6.92% | 1,728 |
| Scotland | 6.74% | 1,858 |
| East Midlands | 6.72% | 1,429 |
| London | 6.62% | 1,829 |
| South East | 6.23% | 2,394 |
| **South West** | **6.18% (lowest)** | 1,705 |

This forward-looking ranking closely mirrors the historical prevalence ranking documented throughout Stage 3/4 (Northern Ireland highest, South East/South West lowest at both) — the model projects forward the same existing geography of disadvantage rather than predicting any reshuffling. The spread (6.18%–7.74%, 1.56 percentage points) is much narrower than the historical prevalence spread (5.96%–15.62%, ~9.66 points), reflecting natural model shrinkage toward the shared constant forecast-shock term applied to every household.

#### `outputs/ukhls_forward_prediction/tables/stage5_forward_predictions.csv`

**Structure:** 19,140 household rows, columns `hrpid, wave, gor_dv, region, interview_year, interview_month, predicted_target_year, fes_magnitude_used, predicted_vulnerable_probability`. Summary statistics: mean predicted probability **6.89%**, **median only 4.01%** (a materially right-skewed distribution), min 0.16%, max 87.97%, standard deviation 9.02 percentage points. 228 households (1.2%) are predicted above 50%; 1,063 (5.6%) above 20%; 3,555 (18.6%) above 10%. `predicted_target_year` splits into two cohorts: 10,937 households interviewed in 2023 predicted for 2024, and 8,203 households interviewed in 2024 predicted for 2025. `fes_magnitude_used` is a single constant value, **−1.0607666666666666**, applied identically to every one of the 19,140 rows — confirming this is the shared/uniform forecast shock (as opposed to Stage 4's Map 3, which uses each household's own realised FES). Six households have blank `gor_dv`/`region` (unmapped GOR codes), excluded from the by-region aggregate table above; one of these six unmapped households has a notably high individual predicted probability of 37.37%. The large gap between the mean (6.89%) and median (4.01%), together with the maximum of 87.97%, shows a small number of households carry very concentrated predicted risk — an obvious candidate group for direct, targeted outreach distinct from a purely geographic targeting approach.

### `outputs/ukhls_forward_prediction/figures/`

- **`outputs/ukhls_forward_prediction/figures/stage5_validation_roc.png`** — an ROC curve (true positive rate vs. false positive rate) for the Stage 5 logistic model evaluated on the held-out wave transitions, titled "Stage 5 Walk-Forward Validation (held-out wave transitions — genuinely forward-predicted, not fit)." A red curve for the model against a dashed grey chance diagonal; legend confirms "Stage 5 model (AUC=0.758)", matching `stage5_validation_metrics.csv` exactly. The curve rises steeply at low false-positive rates (true-positive rate ≈0.6 by a false-positive rate of ≈0.2) before flattening — a typical, well-behaved discrimination shape, comfortably bowed above the diagonal across its entire range with no crossing or inversion.
- **`outputs/ukhls_forward_prediction/figures/stage5_forward_prediction_by_month.png`** — a bar chart, "Which Month Carries the Highest Predicted Risk," x-axis = target month, y-axis = mean predicted probability (%), bars coloured red/green by whether they sit above or below roughly the 6.7–6.8% mark (red: Jan, Feb, Apr, May, Oct, Nov, Dec; green: Mar, Jun, Jul, Aug, Sep). Data labels on each bar match the underlying CSV exactly, confirming April as the tallest bar and September as the shortest.
- **`outputs/ukhls_forward_prediction/figures/stage5_forward_prediction_map.png`** — a choropleth of the UK's 12 regions, "Predicted Forward Vulnerability by UK Region," redesigned this session to display percentages (e.g. "7.7%") rather than a two-decimal probability fraction — resolving the earlier problem where most regions rounded to an indistinguishable "0.07." Colour scale spans roughly 6.2%–7.6%; labels match the region table exactly, with Northern Ireland darkest/highest and South East/South West both lightest/lowest (tied visually at the low end of the colour scale).

---

## Summary of Data-Quality and Methodological Flags Found Across All Files

1. **Model selection in Stage 1 is hindsight-based** (`outputs/tables/model_metrics_comparison.csv`): the pipeline's "best model" choice uses post-hoc realised accuracy, which disagrees with the genuine walk-forward backtest ranking in 4 of 6 series/mode cells — most dramatically for electricity-core, where the backtest's top pick (SARIMA) is the worst real-world performer by a factor of ~11.
2. **`FES_selected` never differs from `FES_core`** (`outputs/fes/fes_components_table.csv`, `outputs/fes/fes_monthly_2025.csv`) — the per-series core/macro selection mechanism has not, in this run, picked macro for any of the three series, even though carbon's own best model is macro-mode.
3. **FES_Macro validates better than FES_Core/Selected** against realised-stress benchmarks (`outputs/fes/fes_comparison_metrics.csv`), an unresolved tension with the per-series accuracy results.
4. **Two SEM fit-index tables report values outside their valid mathematical range** (`outputs/ukhls_cor_sem/tables/cor_sem_fit_indices.csv`: CFI/TLI outside [0,1]; `outputs/ukhls_cor_sem/tables/structural_fit_indices_baseline.csv`: SRMR≈9.2 billion, and CFI/TLI >1 marked "ok" by the pipeline's own validity flags, which appear to test only a lower bound).
5. **A Heywood-case-like pattern** in the second-order SEM structural paths (`outputs/ukhls_cor_sem/tables/structural_paths_baseline.csv`: three of four factors loading ~0.99–1.00 on BASELINE).
6. **The FES-moderation interaction term is non-significant** (`outputs/ukhls_cor_sem/tables/fes_moderation_path.csv`, p=0.608) — the project's own "FES-moderation" framing is not supported by this result; only additive main effects hold.
7. **CVAE-SEM latent alignment is partial** (`outputs/ukhls_cor_cvae/tables/cvae_sem_alignment.csv`): two of four latent dimensions (object, energy) are diffuse rather than cleanly disentangled from each other.
8. **Two duplicate table exports**: `outputs/ukhls_vulnerability/tables/policy_map_baseline_resource_by_region.csv` and `outputs/ukhls_vulnerability/tables/policy_map_financial_strain_by_region.csv` are byte-identical.
9. **A likely data-sparsity artifact**: Northern Ireland's 2024 cell in `outputs/ukhls_vulnerability/tables/policy_vulnerability_region_year_heatmap.csv` reads exactly 0.0% against a neighbouring ~11–12% trend.
10. **Two chart titles state "2024" as an endpoint where the underlying wave data only extends to 2023**: `outputs/ukhls_vulnerability/figures/policy_vulnerability_by_wave.png` and `outputs/ukhls_vulnerability/figures/policy_temporal_change_map.png`.
11. **Two external-validation charts display "ρ=nan"/"r=nan"** (`outputs/ukhls_vulnerability/figures/external_validation_disability_bars.png` and `...disability_scatter.png`) — mathematically correct for n=2 disability categories, but reads as an error without an explanatory caption.
12. **A minor schema inconsistency** in `outputs/forecasts/`: the `series` column is populated in the 24 raw per-model CSVs but blank in the 9 aggregate (`_core`/`_macro`/`_all`) CSVs.
13. **A small region-mapping gap**: `outputs/ukhls_dataset_overview/tables/dataset_rows_by_region.csv`/`dataset_sample_size_by_region.csv` sum to 339,020, 181 rows (0.05%) short of the full 339,201-row panel.
