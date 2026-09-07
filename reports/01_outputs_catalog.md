# Outputs Catalog: File-by-File Analysis

**Project:** Anticipatory Fuel Stress Watch (AFSW) — Forecasting Anticipatory Energy–Carbon Stress and Household Fuel Vulnerability in the UK
**Scope:** every curated table (CSV) and figure (PNG/interactive HTML) under `outputs/` — 230 CSV/PNG files (156 CSV, 74 PNG, each PNG with a duplicate vector `.pdf` twin not separately counted here) plus 6 interactive HTML companions, 236 in total, verified by direct file count as of this update (up from 137 in the previous count — the growth is Stage 1's new rolling walk-forward FES/model-selection tables and figures, described below, plus Stage 2/3's new family-composition, employment-status, and equivalisation-robustness outputs) — organized by pipeline stage, in the order the pipeline produces them. Per-candidate hyperparameter-tuning dumps (`outputs/tuning/`, `outputs/tuning_rolling/`) and the per-target-year raw forecast/table dumps (`outputs/forecasts_rolling/{year}/`, `outputs/tables/rolling/{year}/`) are intermediate artifacts of the tuning and rolling-year loops, not separately catalogued file-by-file here — their *summarized* form (the winning hyperparameters actually used, the year-by-year rolling results) is what feeds the curated tables below. Every curated file is named individually and explicitly by its exact filename (no filename is abbreviated with a pattern or shorthand), followed by a precise, numbers-grounded analysis of its contents: what it structurally contains, the specific values in it, and anything notable, surprising, or worth flagging.

---

## Stage 0 — Data Description (Core & Macro Raw-Data Distributions)

Descriptive-only outputs (no modeling) produced by `src/data_description_overview.py`, documented methodologically in `reports/06_methodology.md` Section 9 and narratively in `reports/05_data_description.md` Sections 3.1–3.2. Summarizes the distributions of the **core** (gas/electricity/carbon, `data/processed/core_energy_carbon.csv`) and **macro** (exogenous regressors, `data/processed/macro_controls.csv`) variables that feed Stage 1's two forecasting modes — both derived directly from the raw price/ONS/National Grid ESO series in `data/raw/` (Section 2 of `reports/05_data_description.md`).

### `outputs/data_description/tables/`

#### `outputs/data_description/tables/core_variable_summary_stats.csv`

**Structure:** 4 rows (one per `core_energy_carbon.csv` column), columns `variable, n, n_missing, mean, std, min, p25, median, p75, max, skew, excess_kurtosis`. Zero missing values in any of the four columns (n=239 throughout, matching the file's full row count exactly).

`gas_growth` (skew 2.40) and `electricity_growth` (skew 1.84) are both markedly right-skewed — large price increases are more extreme/frequent than large decreases over this window. `carbon_growth` has the mildest skew of the four (0.27) despite having by far the widest range (−670.93 to +734.73) and the largest excess kurtosis (8.66) — its extremity is driven by symmetric fat tails, not one-sided outliers, a distinction the mean/std/min/max figures already in `reports/05_data_description.md` Section 3.1 could not on their own distinguish. `electricity_index` (a price *level*, not a growth rate) has by far the lowest excess kurtosis (0.07, near-Normal), consistent with its visibly multi-modal (not fat-tailed) histogram shape described below.

#### `outputs/data_description/tables/macro_variable_summary_stats.csv`

**Structure:** 27 rows (every base `macro_controls.csv` column except its `_lag1`/`_lag12` duplicates and its two regime/seasonal dummy variables), same column set as the core table above.

`gdp_growth` carries the single largest excess kurtosis in the table (82.99) alongside a strongly negative skew (−6.02) — both driven by one extreme observation, the 2020 COVID-lockdown GDP collapse (min −19.2%) sitting far outside an otherwise tightly clustered series (p25/median/p75 all within [−0.1%, +0.5%]). The `_yoy_growth` columns for embedded wind/solar generation and capacity are uniformly right-skewed with large excess kurtosis (up to 88.86 for `embedded_solar_generation_mean_yoy_growth`) — the same "growth off a small base is mechanically volatile" pattern documented for `carbon_growth` above, since embedded solar/wind capacity was small early in the panel. Confirms, independently of `reports/05_data_description.md`'s existing text, that the electricity-demand/generation-derived columns (`electricity_demand_mean`, the `embedded_wind_*`/`embedded_solar_*` columns, `pump_storage_pumping_mean`, `interconnector_net_flow_mean`, `holiday_share`, and their `_yoy_growth` derivatives) carry exactly **48 missing (NaN) rows each** — the pre-2009 period before `historic_demand_2009_2024.csv`'s coverage begins — correcting an earlier draft of the data-description report, which had stated these were zero-filled rather than genuinely missing for that period.

### `outputs/data_description/figures/`

- **`outputs/data_description/figures/core_variable_distributions.png`** — a 2×2 histogram grid (mean dotted, median dashed) for `gas_growth`, `electricity_index`, `electricity_growth`, `carbon_growth`, 2006-05–2026-03 (239 months). `carbon_growth`'s panel is visibly the widest and most sharply peaked of the four (mass concentrated near zero, with thin bars extending out to roughly ±700), the clearest visual confirmation of its high-excess-kurtosis, low-skew shape described above. `electricity_index`'s panel is visibly multi-modal rather than single-peaked — a pre-2021 cluster around 60–100, a distinct 2022-crisis-era cluster around 125–145, and a separate cluster around 190–240 for the most recent months — rather than one smooth bell shape.
- **`outputs/data_description/figures/macro_variable_distributions.png`** — a 3×3 histogram grid for 9 representative macro columns selected as the ones most directly used as FES exogenous regressors: `inflation_growth`, `gdp_growth`, `gas_futures_price`, `electricity_demand_mean`, `embedded_wind_generation_mean`, `embedded_solar_generation_mean`, `gbp_eur_rate`, `weather_volatility`, `holiday_share`. `gdp_growth`'s panel is visually dominated by a single tall spike near zero with one isolated low outlier bar far to the left (the COVID observation); `holiday_share`'s panel is a sparse, near-zero-inflated distribution (median exactly 0.0) with three small clusters corresponding to short/long/leap-adjusted holiday months; `gbp_eur_rate` and `embedded_solar_generation_mean` both show visibly multi-modal shapes rather than single peaks, consistent with genuine structural regime shifts (post-Brexit-referendum sterling weakness; the ramp-up of UK solar deployment from near-zero in the early 2010s) rather than sampling noise.

---

## Stage 1 — Macro Forecasting and the FES Index

### `outputs/tables/model_metrics_comparison.csv`

**Structure:** 24 data rows (3 series × 4 models × 2 modes). Columns: `series_name, model, mode, MAE, RMSE, MAPE, SMAPE, MASE, QuantileLoss, WinklerScore, MSIS, PredictionIntervalCoverage, rank_score, forecast_actual_MAE, forecast_actual_RMSE, forecast_actual_SMAPE, selection_basis, selection_score`.

This single file carries **two independent accuracy regimes**: (1) a backtest/walk-forward composite built from `MAE/RMSE/MAPE/SMAPE/MASE/QuantileLoss/WinklerScore/MSIS/PredictionIntervalCoverage`, summarized into `rank_score` (lower = better in backtest); and (2) accuracy of the final 2025 forecast against the *realised* 2025 actual, `forecast_actual_MAE/RMSE/SMAPE`. Every row in the file currently on disk shows `selection_basis=forecast_actual` and a `selection_score` (1=best, 4=worst within each series/mode group) built from the second, hindsight regime — because this is the **single-year diagnostic path** (`forecast_pipeline.run`, no `--rolling`), run here with `--selection-basis forecast_actual` explicitly, precisely so the two regimes' disagreement could be inspected side by side (see the critical finding below). This is a deliberate, one-off retrospective illustration, not this project's default or reported behaviour: `forecast_pipeline.run`'s own default (and the only behaviour `run_rolling` — the reported path, see the new subsection below — supports) is `selection_basis='validation'`, which selects strictly on pre-target-year `rank_score` and never on `forecast_actual_*`, precisely to avoid the hindsight bias this file's `forecast_actual` configuration deliberately exposes.

**Winner by realised `forecast_actual_RMSE` per series/mode** (this is what is actually used by the pipeline):

| Series | Mode | Winner | RMSE | Runner-up | Worst |
|---|---|---|---|---|---|
| gas | core | LSTM | 7.60 | Prophet 11.99 | SARIMA 92.01 |
| gas | macro | LSTM | 14.77 | TFT 17.17 | SARIMA 37.83 |
| electricity | core | **TFT** | 6.01 | LSTM 6.80 | SARIMA 46.84 |
| electricity | macro | LSTM | 4.65 | TFT 7.03 | SARIMA 19.76 |
| carbon | core | **Prophet** | 9.93 | LSTM 12.09 | TFT 45.78 |
| carbon | macro | TFT | 19.11 | Prophet 19.36 | SARIMA 24.47 |

Two cells' winners differ from `fes_components_table.csv`'s own "best model" row below (which is based on `forecast_actual_MAE`, not the `forecast_actual_RMSE` this table ranks on — a different error metric can pick a different winner at the margin, not a bug): that row instead reports electricity-core as Prophet's win and carbon-core/macro split as Prophet/TFT. Both readings agree the *close, model-selection-sensitive* cells are electricity-core (TFT/LSTM/Prophet all within 1 RMSE point of each other, 6.01–6.95) and carbon-macro (TFT/Prophet within 0.25, 19.11 vs. 19.36) — genuine near-ties, not a discrepancy to resolve.

**Critical finding, on this diagnostic run — backtest and reality disagree sharply in most cells.** Electricity-core: the backtest ranks LSTM best (`rank_score` 1.45, RMSE 8.89), but SARIMA is the single worst model against realised 2025 data (`forecast_actual_RMSE` 46.84, vs. LSTM's 6.80). Gas-core (backtest favours LSTM, RMSE 8.20; SARIMA's realised RMSE is 92.01 vs. LSTM's 7.60) and gas-macro (backtest favours SARIMA, RMSE 12.67; reality instead favours LSTM, realised RMSE 14.77 vs. SARIMA's 37.83) show the same reversal pattern; carbon-core's backtest winner (TFT, RMSE 14.97) is realised-accuracy's *worst* performer (45.78) while Prophet — backtest's worst (RMSE 44.64) — turns out realised-best (9.93). Only electricity-macro's backtest winner (LSTM) stays the realised-accuracy winner too. **This is exactly why `forecast_pipeline`'s default (and `run_rolling`'s only) selection basis is `'validation'`, not `'forecast_actual'`**: choosing on post-hoc realised accuracy — as this one diagnostic run deliberately does, to make the disagreement visible — is a hindsight pick that a genuinely blind, pre-2025 forecast exercise could not have made. The reported rolling walk-forward path (new subsection below) never makes this choice; its own year-by-year winners are picked purely on pre-target-year `rank_score`, and the same backtest/realised divergence shows up there too (as a property of the underlying models, not of the selection mechanism) — reported transparently as context rather than as a live methodological flaw.

SARIMA is the worst-or-near-worst realised performer in 5 of 6 cells (only gas-macro's LSTM/SARIMA gap is smaller) — its wide, unstable long-horizon extrapolations (documented in the forecast-comparison figures below) are the direct cause, independent of which selection basis is used. LSTM wins or is runner-up in 5 of 6 cells by realised RMSE, the closest thing to a consistent performer in this diagnostic run; TFT and Prophet each win exactly one cell (electricity-core and carbon-core respectively) rather than clustering at the bottom the way SARIMA does.

---

### `outputs/forecasts/` (33 files)

Every per-model file shares columns `date, model, mode, forecast, lower_bound, upper_bound, actual` (a `series` column is present only in the raw LSTM/TFT per-model files, and is blank in the three aggregate files per series — a minor, harmless schema inconsistency worth noting for anyone querying these files programmatically). Each series has 8 per-model files (4 models × 2 modes) plus 3 aggregates (`_core.csv` = 4-model concatenation for core mode, `_macro.csv` = 4-model concatenation for macro mode, `_all.csv` = concatenation of both, 96 rows), for 11 files per series × 3 series = 33 files total.

#### Gas (`gas_growth_pct_forecasts_*.csv`)

Actual monthly gas growth 2025: −12.4%, −12.4%, −12.3%, +12.5%, +12.6%, +12.6%, +13.3%, +13.3%, +13.3%, +2.2%, +2.2%, +2.1% — a sharp regime shift from strongly negative (winter) to strongly positive (spring/summer) then back to near-zero (autumn/winter). Now that hyperparameter tuning runs by default (`forecast_pipeline`'s `tune=True` default), every model's point forecasts and intervals below differ from an untuned run — the numbers here are current as of this run, not necessarily stable across every future rerun.

- **`outputs/forecasts/gas_growth_pct_forecasts_sarima_core.csv`** — 12 monthly rows. Mean forecast 80.84%, sign-matches the actual all 12 months but massively over-forecasts magnitude. Average prediction-interval width 90.37 percentage points; only 3 of 12 months (25%) have the actual value fall inside the stated interval despite that width — the point forecast itself, not just the uncertainty band, is badly miscalibrated. The single worst realised performer for gas (RMSE 92.01).
- **`outputs/forecasts/gas_growth_pct_forecasts_sarima_macro.csv`** — tamer than core (mean 33.50%) but still overshoots materially; average PI width 84.53; 9/12 (75%) covered; RMSE 37.83, again the worst macro-mode gas model.
- **`outputs/forecasts/gas_growth_pct_forecasts_prophet_core.csv`** — flat, hovering ~7% regardless of the actual's dramatic swings; 9/12 correct signs; the widest average PI of any gas model (125.50); 12/12 (100%) covered — a band so wide it is uninformative even though nominally "always correct." RMSE 11.99, second-best core-mode gas model.
- **`outputs/forecasts/gas_growth_pct_forecasts_prophet_macro.csv`** — 8/12 correct signs; average PI width 87.87; 11/12 (92%) coverage; RMSE 18.37.
- **`outputs/forecasts/gas_growth_pct_forecasts_lstm_core.csv`** — **the winning file for gas** (RMSE 7.60, the lowest of any gas model/mode cell). Mean forecast 3.47%, 12/12 correct signs, by far the tightest average PI of any gas model (11.77); coverage 5/12 (42%) — well-calibrated in width but occasionally too narrow for the sharpest swings (the April/May regime jump).
- **`outputs/forecasts/gas_growth_pct_forecasts_lstm_macro.csv`** — a cautionary case, same pathology as before but now more pronounced: mean forecast −6.99% (persistently negative) against an actual that is mostly strongly positive, getting the sign wrong in 9 of 12 months (only 3/12 correct). RMSE 14.77 still wins the macro-mode gas cell — its low RMSE again reflects a low-variance "safe, small forecast" strategy, not directional skill. Average PI width only 7.38, the narrowest of any gas model, and 0 of 12 months (0%) have the actual fall inside it — the clearest single case of prediction-interval overconfidence in the whole forecasts directory, unchanged in kind from the previous run even though the exact numbers shifted.
- **`outputs/forecasts/gas_growth_pct_forecasts_tft_core.csv`** — mean forecast 11.85%, 10/12 correct signs; average PI width 43.14, 9/12 (75%) coverage; RMSE 19.98, mid-pack.
- **`outputs/forecasts/gas_growth_pct_forecasts_tft_macro.csv`** — mean forecast 9.49%, 10/12 correct signs; average PI width 61.05, 12/12 (100%) coverage; RMSE 17.17, second-best macro-mode gas model.
- **`outputs/forecasts/gas_growth_pct_forecasts_core.csv`** — 48-row concatenation of the four core-mode model files above (used by the core-mode comparison and prediction-interval figures).
- **`outputs/forecasts/gas_growth_pct_forecasts_macro.csv`** — 48-row concatenation of the four macro-mode model files above.
- **`outputs/forecasts/gas_growth_pct_forecasts_all.csv`** — 96-row concatenation of both modes together (used by `interactive_gas_core.html`/`interactive_gas_macro.html` and the historical-context figure).

#### Electricity (`electricity_growth_pct_forecasts_*.csv`)

Actual monthly electricity growth 2025: −8.78%, −8.78%, −8.83%, +4.63%, +4.57%, +4.57%, +7.99%, +7.99%, +8.04%, +2.70%, +2.81%, +2.75%.

- **`outputs/forecasts/electricity_growth_pct_forecasts_sarima_core.csv`** — flat around −9% to −11% for January–March, then an abrupt jump to overshoot the actual's 4.6–8.0% range from April onward; 12/12 correct signs but badly wrong on magnitude; average PI width 44.32, 3/12 (25%) coverage. The worst realised performer for electricity, RMSE 46.84 — by far the largest single RMSE gap to the next-worst model of any series/mode cell.
- **`outputs/forecasts/electricity_growth_pct_forecasts_sarima_macro.csv`** — same directional pathology, tamer (mean 16.64%); average PI width 39.14, 9/12 (75%) coverage; RMSE 19.76, again worst macro-mode electricity model.
- **`outputs/forecasts/electricity_growth_pct_forecasts_prophet_core.csv`** — mean 2.47%, 9/12 correct signs, average PI width 67.04, 12/12 (100%) coverage (uninformatively wide, mirroring Prophet's gas-core behaviour). RMSE 6.95, close to but behind the two winners below.
- **`outputs/forecasts/electricity_growth_pct_forecasts_prophet_macro.csv`** — mean 2.36%, 7/12 correct signs, average PI width 48.72, 12/12 (100%) coverage; RMSE 8.29.
- **`outputs/forecasts/electricity_growth_pct_forecasts_lstm_core.csv`** — mean forecast −2.25%, persistently negative against an actual that turns strongly positive from April, getting only 3/12 signs correct; tightest average PI of any electricity model (5.26), 0/12 (0%) coverage — the same overconfident-narrow-interval pathology seen in gas-LSTM-macro. Despite the poor sign record, RMSE (6.80) is still competitive (runner-up) because the magnitude stays close to the (comparatively small) true values.
- **`outputs/forecasts/electricity_growth_pct_forecasts_lstm_macro.csv`** — mean 1.56%, 11/12 correct signs, average PI width 12.65, 10/12 (83%) coverage — genuinely well-calibrated. RMSE 4.65, **the winning file for electricity-macro** and the single lowest realised RMSE across all 24 model/series/mode cells this run.
- **`outputs/forecasts/electricity_growth_pct_forecasts_tft_core.csv`** — mean −0.09%, 8/12 correct signs, average PI width 12.92, 9/12 (75%) coverage. RMSE 6.01, **the winning file for electricity-core** — narrowly ahead of LSTM-core (6.80) and Prophet-core (6.95) despite a weaker sign-accuracy record, illustrating again that sign accuracy and RMSE are not the same axis.
- **`outputs/forecasts/electricity_growth_pct_forecasts_tft_macro.csv`** — mean 0.43%, 6/12 correct signs, average PI width 8.52, 3/12 (25%) coverage; RMSE 7.03.
- **`outputs/forecasts/electricity_growth_pct_forecasts_core.csv`** — 48-row concatenation of the four core-mode electricity model files.
- **`outputs/forecasts/electricity_growth_pct_forecasts_macro.csv`** — 48-row concatenation of the four macro-mode electricity model files.
- **`outputs/forecasts/electricity_growth_pct_forecasts_all.csv`** — 96-row concatenation of both modes.

#### Carbon (`carbon_growth_pct_forecasts_*.csv`)

Actual monthly carbon growth 2025: +25.14%, +21.65%, +9.32%, −3.90%, −5.07%, +2.36%, +4.82%, +3.76%, +14.07%, +19.24%, +20.00%, +18.10% — the most volatile and least monotonic of the three series, with a mid-year dip into negative territory (April–May) before a strong Q3–Q4 recovery.

- **`outputs/forecasts/carbon_growth_pct_forecasts_sarima_core.csv`** — flat and negative (mean −11.24%) all year; only 2 of 12 months (17%) correct sign, missing the entire recovery; average PI width 169.12, but 12/12 (100%) coverage because the interval is so wide it is uninformative rather than accurate. RMSE 24.45.
- **`outputs/forecasts/carbon_growth_pct_forecasts_sarima_macro.csv`** — near-identical to core (mean −11.30%), same 2/12 sign accuracy, average PI width 167.10, 12/12 (100%) coverage; RMSE 24.47.
- **`outputs/forecasts/carbon_growth_pct_forecasts_prophet_core.csv`** — mean point forecast 12.49%, 10/12 correct signs, average PI width 237.74 (the widest of any file in the whole forecasts directory), 12/12 (100%) coverage. **The winning file for carbon-core**, RMSE 9.93 — the single lowest realised RMSE of any carbon model/mode cell, a reversal from an earlier run where Prophet-core was carbon's worst model; the width of its interval means this accuracy is on the point forecast, not a sign the model is well-calibrated in general.
- **`outputs/forecasts/carbon_growth_pct_forecasts_prophet_macro.csv`** — mean 27.66%, 10/12 correct signs, RMSE 19.36 (essentially tied with TFT-macro's 19.11, the macro-mode winner below), average PI width 237.74 (identical width to Prophet-core, again the frozen/near-frozen-band pattern), 12/12 (100%) coverage.
- **`outputs/forecasts/carbon_growth_pct_forecasts_lstm_core.csv`** — mean forecast 9.51%, 9/12 correct signs, RMSE 12.09 — competitive, runner-up to Prophet-core; average PI width 26.44, 10/12 (83%) coverage.
- **`outputs/forecasts/carbon_growth_pct_forecasts_lstm_macro.csv`** — only 2 of 12 correct signs (mean −8.61% against a series that is mostly positive), RMSE 22.77; average PI width 14.94 (tight, but badly miscalibrated in direction), 3/12 (25%) coverage — the same directional-overconfidence pattern flagged elsewhere for LSTM-macro cells.
- **`outputs/forecasts/carbon_growth_pct_forecasts_tft_core.csv`** — forecast rises essentially monotonically all year, mean 48.36%, chasing the real recovery but badly overshooting; 10/12 correct signs but the **worst core-mode RMSE for carbon of any model (45.78)** — the single largest backtest-favoured-but-realised-worst reversal in this run (TFT-core wins the pre-target-year `rank_score` for carbon-core at 14.97, the lowest of any of the four models, yet is realised-accuracy's worst). Average PI width 122.77, 11/12 (92%) coverage.
- **`outputs/forecasts/carbon_growth_pct_forecasts_tft_macro.csv`** — mean 21.64%, 10/12 correct signs, average PI width 40.41, 6/12 (50%) coverage. **The winning file for carbon-macro**, RMSE 19.11 — narrowly ahead of Prophet-macro (19.36), a near-tie rather than the decisive win TFT-macro posted in an earlier run.
- **`outputs/forecasts/carbon_growth_pct_forecasts_core.csv`** — 48-row concatenation of the four core-mode carbon model files.
- **`outputs/forecasts/carbon_growth_pct_forecasts_macro.csv`** — 48-row concatenation of the four macro-mode carbon model files.
- **`outputs/forecasts/carbon_growth_pct_forecasts_all.csv`** — 96-row concatenation of both modes.

---

### `outputs/fes/` (11 files) — single-year diagnostic path

#### `outputs/fes/fes_components_table.csv`

**Structure:** 5 rows (3 growth-rate components + uncertainty + total) × 5 columns (FES_core, FES_macro, FES_selected, **FES_weighted**, FES_actual) — one column wider than before, since `FES_weighted` (Section 2.2, `reports/06_methodology.md`) is a newly-added fourth variant, computed alongside (never replacing) the three equal-weighted ones. Plus one additional text row reporting the "best model" per mode.

| Component | FES_core | FES_macro | FES_selected | FES_weighted | FES_actual |
|---|---|---|---|---|---|
| gas_growth_pct (z-score) | −0.20837 | −0.08728 | −0.20837 | −0.33268 | −0.19390 |
| electricity_growth_pct (z-score) | −0.35869 | −0.41618 | −0.41618 | −0.46207 | −0.41105 |
| carbon_growth_pct (z-score) | +0.04342 | +0.10274 | +0.04342 | +0.01273 | +0.03237 |
| Uncertainty/RealVol (z-score) | +1.51309 | +0.68934 | +0.07994 | +0.07994 | −0.57706 |
| **FES_total (annual mean)** | **+0.98946** | **+0.28862** | **−0.50118** | **−0.70207** | **−1.14962** |
| "best model" row | gas:LSTM \| electricity:Prophet \| carbon:Prophet | gas:Prophet \| electricity:LSTM \| carbon:TFT | gas:core \| electricity:macro \| carbon:core | inverse-RMSE weighted (see `fes_variant_selection.csv`) | realised values |

**Every number in this table changed from an earlier run, several by sign.** `FES_selected` is **no longer identical to `FES_core`** — the per-series bug fix documented in `reports/06_methodology.md` Section 2.2 (an earlier version of `_select_best_mode_per_series` always compared on `forecast_actual_MAE` regardless of the requested `selection_basis`, silently hindsight-biasing the mode choice) now genuinely picks the better mode per series: electricity's contribution now comes from macro mode (−0.416, matching `FES_macro`'s own electricity term exactly) while gas and carbon still come from core. The **uncertainty term is the most dramatic change**: `FES_core`'s own uncertainty z-score flipped from strongly negative (−0.492, an earlier run) to strongly positive (**+1.513**), which alone accounts for most of `FES_core`'s annual-mean total flipping sign, from −1.061 (stressed, roughly in line with reality) to **+0.989 (comfortably *below*-average stress)** — the opposite direction from the realised actual (−1.150). `FES_macro` shows the same sign flip on its total (−1.256 → +0.289), though less extreme. **`FES_selected` and `FES_weighted` are the only two variants that still land on the correct (negative/stressed) side of the realised actual** (−0.501 and −0.702 respectively, vs. actual −1.150) — both undershoot the *magnitude* of realised stress, but neither gets the *direction* wrong the way Core/Macro now do. This directly explains the `fes_comparison_metrics.csv` finding below, where Selected/Weighted decisively outperform Core/Macro on every realised-stress benchmark.

#### `outputs/fes/fes_summary_2025.csv`

**Structure:** long-format table, columns `variant, component, z_mean` — 25 rows covering core/macro/selected/**weighted**/actual variants × 5 components each (one more variant, hence 5 more rows than before), plus 2 additional rows for a prior-year actual baseline: `actual_prior_year, year, 2024.0` and **`actual_prior_year, FES_TOTAL, −2.69529`** (revised from an earlier run's −2.582 — `_compute_actual_fes_for_window`'s reference window was corrected this iteration to genuinely exclude the prior year from its own z-scoring baseline, Section 3.4 of `reports/06_methodology.md`; the prior-year figure is therefore now out-of-sample rather than partly in-sample). Realised 2025 stress (annual mean −1.150) is still markedly less severe than realised 2024 stress (−2.695) — a roughly 57% reduction in the index's magnitude, indicating a substantial year-on-year easing of the combined gas/electricity/carbon price-and-uncertainty shock, materially unchanged in direction from the earlier estimate even though the exact 2024 baseline shifted.

#### `outputs/fes/fes_prior_actual_2024.csv`

**Structure:** 12 monthly rows for calendar year 2024, columns `date, actual_gas, z_gas_actual, actual_electricity, z_electricity_actual, actual_carbon, z_carbon_actual, real_vol_actual, z_real_vol_actual, fes_actual`. Monthly `fes_actual` ranges from −3.202 (June, the worst month) to −1.896 (November, the best month) — consistent with the revised annual mean above, and every single month in 2024 is still more negative (more stressed) than every month in the 2025 actual series described below.

#### `outputs/fes/fes_monthly_2025.csv`

**Structure:** the widest file in the project — 12 monthly rows, now including a `fes_weighted` column (plus its own `z_*_weighted`/`z_unc_weighted` component columns) alongside the existing `fes_core`/`fes_macro`/`fes_selected`/`fes_actual` blocks and the three realised-volatility benchmark blocks (`rv_actual_A/B/C`, `fes_actual_A/B/C`).

**Monthly trajectories** (matching `fes_monthly_2025.png`): `fes_core` is now **positive throughout the entire year** (+0.64 in January rising to +1.12 by November, +1.12 in December) — i.e. core-mode's own forecast reads 2025 as comfortably below-average stress every single month. `fes_macro` oscillates around zero (−0.69 to +1.18, sign-flipping month to month) rather than trending either way. `fes_actual`, by contrast, stays **negative every month** (−0.79 to −1.79) — genuinely above-average stress throughout 2025, with the same non-monotonic shape as before: a dip to −1.79 in March, a jump to −0.97 by April, a peak (least-negative) around −0.79 to −0.84 in July–September, then **a relapse to −1.13 to −1.16 by October–December**. `fes_selected` and `fes_weighted` are the two variants that track this negative range at all (roughly −0.25 to −1.64 and −0.38 to −1.64 respectively) — both correctly negative every month, unlike Core/Macro, though both still run persistently less negative than the actual (i.e. still understate the true severity), especially early in the year. This is a materially different — and, on direction, better — read than an earlier run, where `fes_core`/`fes_macro` were both negative and only diverged from `fes_actual` on shape (the October–December relapse), not on sign.

#### `outputs/fes/fes_comparison_metrics.csv`

**Structure:** 12 rows = 4 forecast variants (Equal_Core, Equal_Macro, Equal_Selected, **Equal_Weighted**) × 3 realised-actual benchmark definitions (Actual_RollingVol, Actual_CrossComp, Actual_AbsShock), N=12 each — one more variant (and 3 more rows) than before.

| Variant | Benchmark | MAE | RMSE | Pearson r | R² | Theil U |
|---|---|---|---|---|---|---|
| Equal_Core | RollingVol | 2.436 | 2.540 | 0.627 | −8.915 | 3.698 |
| Equal_Core | CrossComp | 2.139 | 2.154 | 0.630 | −42.928 | 8.161 |
| Equal_Core | AbsShock | 2.144 | 2.151 | 0.617 | −96.741 | 12.491 |
| Equal_Macro | RollingVol | 1.735 | 1.931 | 0.267 | −4.729 | 2.811 |
| Equal_Macro | CrossComp | 1.438 | 1.562 | 0.102 | −22.102 | 5.918 |
| Equal_Macro | AbsShock | 1.444 | 1.545 | 0.205 | −49.376 | 8.967 |
| Equal_Selected | RollingVol | 0.945 | 1.109 | **0.839** | −0.891 | 1.615 |
| Equal_Selected | CrossComp | 0.648 | 0.673 | 0.838 | −3.282 | 2.548 |
| Equal_Selected | AbsShock | 0.654 | 0.676 | 0.816 | −8.662 | 3.927 |
| Equal_Weighted | RollingVol | **0.772** | **0.912** | 0.824 | **−0.278** | **1.328** |
| Equal_Weighted | CrossComp | **0.456** | **0.503** | 0.824 | **−1.390** | **1.904** |
| Equal_Weighted | AbsShock | **0.482** | **0.523** | 0.805 | **−4.780** | **3.037** |

**`Equal_Selected` and `Equal_Weighted` decisively beat `Equal_Core`/`Equal_Macro` on every single metric, every benchmark** — a complete reversal from an earlier run, where `Equal_Selected` was bit-for-bit identical to `Equal_Core` (the mode-selection bug meant "Selected" wasn't actually selecting anything) and `Equal_Macro` was the best of the three. Now: Pearson r against RollingVol is 0.839 (Selected) / 0.824 (Weighted) vs. 0.627 (Core) / 0.267 (Macro); MAE against AbsShock is 0.654 (Selected) / 0.482 (Weighted) vs. 2.144 (Core) / 1.444 (Macro) — roughly a 3–4× error reduction. `Equal_Weighted` posts the single best MAE/RMSE/Theil's U on every benchmark (e.g. Theil's U 1.328 vs. Selected's 1.615 on RollingVol), while `Equal_Selected` posts the single best Pearson r on two of three (0.839 RollingVol, 0.838 CrossComp) — the two new/fixed variants essentially trade off "smallest average error" (Weighted) against "best rank/shape tracking" (Selected), both clearly ahead of the unweighted, un-selected Core/Macro baselines. **R² is negative for every variant on every benchmark** (as low as −96.7 for Core/AbsShock, "only" −0.278 for Weighted/RollingVol) — with only 12 monthly observations, R² is extremely sensitive to any systematic bias in the point forecast, so the *relative* ranking between variants (Weighted/Selected far less negative than Core/Macro) is the informative signal here, not the absolute R² values themselves. This diagnostic-run finding is directly consistent with the reported rolling walk-forward path below, where `Equal_Weighted`'s year-by-year RMSE (mean 2.79) also comes in close behind the winning `Equal_Macro` (2.75) and clearly ahead of `Equal_Selected` (2.94) and `Equal_Core` (3.24) — the *specific* ranking of Selected vs. Weighted vs. Macro differs between the single-year diagnostic and the 16-year rolling average (expected, given N=12 vs. N=16×12 months of evidence), but both runs agree the un-selected `Equal_Core` is the weakest variant.

---

### `outputs/fes/` — Rolling walk-forward path (`--rolling`, the reported results)

The six files below are produced only by `forecast_pipeline.run_rolling` and are **the methodologically preferred, reported Stage 1 results** (`reports/06_methodology.md` Section 2.3) — every feasible year 2009→2024 retrains all 24 model/mode/series combinations through that year and forecasts the next, so the numbers here reflect 16 independent forecast-year exercises rather than the single-year diagnostic path's one 2025 snapshot. Regenerated together in one run (all six files share a Sep-4 03:03 timestamp, strictly after the single-year files above), so they are internally consistent with each other and reflect the current code's defaults: `selection_basis='validation'` (never hindsight) and `tune=True` (one hyperparameter-tuning pass on the latest rolling year, reused across all 16, per `run_rolling`'s own documented cost/benefit tradeoff).

#### `outputs/fes/fes_variant_selection.csv`

**Structure:** 4 rows, columns `FES_variant, mean_rmse, n_comparisons, chosen`, one row per equal-weighted variant, each compared against 3 realised-actual benchmarks over 16 years (`n_comparisons=48`).

| FES_variant | mean_rmse | chosen |
|---|---|---|
| **Equal_Macro** | **2.749** | **True** |
| Equal_Weighted | 2.790 | False |
| Equal_Selected | 2.940 | False |
| Equal_Core | 3.236 | False |

**`Equal_Macro` wins the walk-forward variant competition** and is the variant actually used downstream for every household-level FES Magnitude/Delta computation in Stage 2 (`attach_fes_delta`, `reports/06_methodology.md` Section 3.4) — a different winner from either of the single-year diagnostic run's own top variants (Selected/Weighted, above), a reminder that a single target year's ranking need not generalise to the 16-year average, and vice versa. `Equal_Weighted` is a close second (2.790, only 1.5% worse than Macro) and clearly ahead of `Equal_Selected` (2.940) and `Equal_Core` (3.236, the worst of the four in the rolling average, consistent with the single-year run's own finding that un-selected Core is weakest).

#### `outputs/fes/model_selection_by_year.csv`

**Structure:** 96 rows (16 years × 3 series × 2 modes), columns `as_of_year, target_year, series, mode, model` — the specific model that won each of the 96 individual walk-forward cells, chosen fresh every year rather than fixed across the timeline.

**LSTM wins 50 of 96 cells (52%)** — SARIMA 18 (19%), TFT 17 (18%), Prophet 11 (11%). LSTM's dominance is strongest in carbon (25 of 32 series/mode/year cells, 78%) and in core mode generally (30 of 48, 63%, vs. 20 of 48 in macro mode, 42%) — macro mode is where SARIMA picks up most of its wins (13 of 18), suggesting exogenous regressors help SARIMA's relative competitiveness more than they help LSTM's. No single model wins every year for any series/mode combination — even carbon-core, LSTM's strongest cell, has 3 non-LSTM winners across the 16 years (2 Prophet, 1 TFT) — confirming the rolling design's premise that a fixed "best model" chosen once would not have been correct for every year.

#### `outputs/fes/forecast_performance_by_year.csv`

**Structure:** 96 rows (same grouping as above), columns `as_of_year, target_year, series, mode, model, RMSE, forecast_actual_RMSE` — for each cell, the winning model's own genuine walk-forward validation RMSE (`RMSE`, pre-target-year information only) alongside how that same model then performed against the target year's realised actual (`forecast_actual_RMSE`) once it existed.

Mean walk-forward validation RMSE by series/mode: gas core 10.10 / macro 15.03; electricity core 7.21 / macro 8.28; carbon core 35.70 / macro 38.36 — core mode has the lower (better) mean validation RMSE in every one of the three series, an average pattern the single 2025 diagnostic year alone could not establish. Mean realised (`forecast_actual_RMSE`) accuracy of those same validation-selected winners tells a partly different story: gas core 23.57 / macro 28.58 (core still better), electricity core 12.16 / **macro 10.13** (macro now *better* on realised accuracy despite worse validation RMSE), carbon core 40.31 / macro 42.20 (core still better, narrowly). **This persistent backtest-vs-realised gap is a property of the underlying forecasting problem, not of the (now honest) selection mechanism** — it is the same phenomenon flagged in the single-year diagnostic's "Critical finding" above, still visible here even though `run_rolling` never once uses `forecast_actual_RMSE` to choose a winner. 2022 (as_of_year, target_year 2023) is the single hardest year to forecast across the board — e.g. gas-core's winning model's realised RMSE spikes to 145.07 that year (vs. a 16-year mean of 23.57), squarely the 2022–23 energy-crisis price shock landing exactly in a forecast-target window.

#### `outputs/fes/fes_metrics_selected_variant_by_year.csv`

**Structure:** 16 rows (one per rolling year), columns `as_of_year, target_year, FES_variant, RMSE, Pearson_r` — every row's `FES_variant` reads `Equal_Macro`, confirming this file reports the single winning variant's (not all four candidates') year-by-year accuracy, replacing what an earlier iteration described as a two-heatmap variant×benchmark grid.

Year-by-year RMSE ranges from a best of 0.510 (as_of 2017, forecasting 2018 — the calmest year in the whole series) to a worst of **13.923** (as_of 2021, forecasting 2022 — the energy-crisis year, over 27× the best year's error) — the single clearest illustration in the whole Stage 1 output set of how forecast difficulty is not remotely uniform across time. Pearson r swings even more dramatically, from a strong +0.771 (as_of 2024, forecasting 2025 — the most recent, best-tracked year) down to −0.911 (as_of 2019, forecasting the COVID-disrupted 2020) and −0.615 (as_of 2016). The 16-year mean Pearson r is **−0.103** — slightly negative on average, driven by several sharply negative years pulling down what more recent years' strongly positive correlations (2024→2025: +0.771; 2020→2021: +0.400) would otherwise suggest — a genuinely mixed track record that should temper how much confidence is placed in `Equal_Macro`'s within-year shape-tracking, even though it wins on RMSE.

#### `outputs/fes/fes_rolling_yearly.csv` and `outputs/fes/fes_rolling_monthly.csv`

**Structure:** `fes_rolling_yearly.csv` has 16 rows (one per `as_of_year`/`target_year` pair, 2009→2010 through 2024→2025), columns `as_of_year, target_year, fes_core, fes_macro, fes_selected, fes_weighted, fes_actual` (annual means); `fes_rolling_monthly.csv` is the same data at 12×16=192-row month resolution, the file `src.ukhls_preprocessing.attach_fes_delta` actually joins against for household-month-resolution FES Magnitude (`reports/06_methodology.md` Section 3.4).

All four forecast variants and the realised actual swing from comfortably positive (2017→2018: `fes_core` +0.86, `fes_actual` +0.18, the calmest year) to sharply negative (2021→2022: every variant strongly negative, `fes_actual` reaching **−15.36** — an order of magnitude more extreme than any other year in the 16-year series, the clearest single number in this file confirming 2022 as the energy-crisis outlier) and back. `fes_weighted` is the most volatile variant across the 16 years (ranging −4.65 to +3.06), consistent with it amplifying whichever series currently has the lowest validation RMSE rather than smoothing across all three equally the way the unweighted variants do.

---

### `outputs/figures/` (27 PNG files, each with a duplicate vector `.pdf` twin, plus 6 interactive `.html` companions)

#### FES-level figures (single-year diagnostic path)

- **`outputs/figures/fes_components_2025.png`** — a two-panel bar chart, now with a 4th bar group per component (Core/Macro/**Weighted**/Actual — Selected is omitted from this specific figure per `_plot_fes_components`'s own grouping). Left panel: 4 components (Gas Growth, Elec Growth, Carbon Growth, Uncertainty/RealVol) as grouped z-score bars, value labels matching `fes_components_table.csv` exactly. The uncertainty bar for Core is now the single tallest bar in the whole left panel (+1.513, positive) rather than one of the more negative bars as in an earlier run. Right panel: four annual-mean total bars (Core +0.989, Macro +0.289, Weighted −0.702, Actual −1.150) — visually, Core and Macro now sit on the *opposite side of zero* from Actual, while Weighted is the only forecast bar sharing Actual's sign, a materially different and more visually striking pattern than an earlier run's "Core undershoots, Macro overshoots, both same sign as Actual" story.
- **`outputs/figures/fes_monthly_2025.png`** — a line chart of the four FES variants across the 12 months of 2025 (Core solid, Macro dashed, **Weighted dash-dot**, Actual dotted/marked). Core's line now sits visibly *above* zero for the entire year while Actual's stays *below* zero for the entire year — the two lines never cross, a visually unambiguous confirmation that Core's forecast disagrees with reality on basic direction, not just magnitude or shape. Weighted's line is the forecast variant that tracks closest to Actual's shape and sign throughout, though it still sits measurably less negative (nearer zero) than Actual in most months.
- **`outputs/figures/fes_metrics_pearson_r_heatmap.png`** — now a 4×3 heatmap (rows = 4 FES variants, columns = 3 realised-benchmark definitions), reproducing `fes_comparison_metrics.csv`'s Pearson r column exactly. Equal_Selected/RollingVol (r=0.839) renders as the single greenest/best cell; Equal_Macro/CrossComp (r=0.102) renders as the reddest/worst — a complete reshuffle from an earlier run, where Equal_Macro's row was uniformly the best and Equal_Core/Equal_Selected's (then-identical) row was uniformly the worst.
- **`outputs/figures/fes_metrics_rmse_heatmap.png`** — the RMSE-column companion heatmap, now 4×3; Equal_Weighted/CrossComp (RMSE=0.503) is the single best (palest) cell; Equal_Core/AbsShock (RMSE=2.151) is the worst — again a full reshuffle, with the two new/fixed variants (Selected, Weighted) occupying every one of the palest cells and Core/Macro occupying every one of the darkest, visually confirming the comparison-metrics table's finding of a decisive Selected/Weighted-over-Core/Macro split.

#### FES-level figures (rolling walk-forward path — the reported results)

- **`outputs/figures/fes_rolling_trend.png`** — a line chart of `fes_core`/`fes_macro`/**`fes_weighted`**/`fes_actual` (purple/red/gold/navy, per `src/fes_calculator.py`'s `_PALETTE`) across all 16 rolling target years, 2010–2025. The 2022 spike in `fes_actual` (to −15.36, per `fes_rolling_yearly.csv` above) renders as a single sharp downward needle dwarfing every other year on the chart — visually the same "2022 is the outlier" finding as the single-year diagnostic's forecast-vs-actual figures below, but now shown as part of a genuine 16-year trend rather than inferred from one year's historical-context plot.
- **`outputs/figures/model_selection_polar_core.png`** and **`outputs/figures/model_selection_polar_macro.png`** — the two "wind-rose" charts described in the project README (groups = 3 series, items = the 16 rolling target years, bar height = that year's winning model's validation RMSE, bar colour = which model won, per `plotting_utils.plot_grouped_circular_bars`). Carbon's group is visibly the tallest/most-colour-varied of the three in both charts (mean validation RMSE 35.7 core / 38.4 macro, an order of magnitude above gas/electricity's single-digit-to-teens RMSEs) — the same carbon-is-hardest-to-forecast pattern the single-year diagnostic path's own `forecast_comparison_carbon_*` figures show, now confirmed across 16 years rather than one. LSTM's colour (per `MODEL_COLORS`) dominates the core-mode chart's bars (30 of 48) more visibly than the macro-mode chart's (20 of 48, with more SARIMA-coloured bars interspersed), matching `model_selection_by_year.csv`'s win-count breakdown exactly.
- **`outputs/figures/rolling_forecast_performance_gas.png`**, **`outputs/figures/rolling_forecast_performance_electricity.png`**, **`outputs/figures/rolling_forecast_performance_carbon.png`** — one figure per series (replacing an earlier combined 3-panel design), each plotting the winning model's validation RMSE by rolling year with core/macro shown as solid/dashed linestyle and each point coloured by which model won that year (`forecast_performance_by_year.csv`). Carbon's figure has the largest year-to-year swings of the three (RMSE ranging roughly 11–74 across the 16 years, per the underlying table); gas and electricity's figures are visibly calmer, with gas's SARIMA-coloured points concentrated in macro mode specifically (matching the model-selection win-count finding that SARIMA's wins skew macro).

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

- **`outputs/figures/model_ranking_polar_gas_core.png`** — a circular "wind-rose" bar chart with 4 colour-coded quadrants (one per model: SARIMA, Prophet, LSTM, TFT), each containing 3 wedges (MAE, RMSE, SMAPE) using the realised `forecast_actual_*` metrics; wedge length is proportional to error magnitude (shorter = better). Title states **"selected: LSTM"** (MAE 7.27) — SARIMA's wedges (MAE 77.69, RMSE 92.01) dominate the chart's outer ring.
- **`outputs/figures/model_ranking_polar_gas_macro.png`** — title now states **"selected: Prophet"** (MAE 12.81) rather than LSTM — a genuine three-way near-tie with TFT (13.38) and LSTM (13.45) visibly rendering as almost-equal-length wedges for those two models' MAE quadrants, and SARIMA (31.59) clearly the odd one out.
- **`outputs/figures/model_ranking_polar_electricity_core.png`** — title now states **"selected: Prophet"** (MAE 5.27), edging out TFT (5.36) and LSTM (6.47) — the closest three-way competition of any of the six charts, all three non-SARIMA quadrants rendering as visibly similar-length wedges.
- **`outputs/figures/model_ranking_polar_electricity_macro.png`** — title states **"selected: LSTM"** (MAE 3.82), the single lowest MAE of any cell in this figure set.
- **`outputs/figures/model_ranking_polar_carbon_core.png`** — title now states **"selected: Prophet"** (MAE 8.81), edging out LSTM (8.82) by a razor-thin margin (0.003) — visually indistinguishable wedge lengths for those two models; TFT is now this cell's *worst* model (MAE 41.49, RMSE 45.78), a reversal from an earlier run where TFT was mid-pack and LSTM won outright.
- **`outputs/figures/model_ranking_polar_carbon_macro.png`** — title states **"selected: TFT"** (MAE 15.29), narrowly ahead of Prophet (16.87) — the sole cell of the six where a macro-mode-favouring pattern (TFT beating LSTM here specifically) persists from an earlier run, though the margin over Prophet is much tighter than before.

Three of the six titles changed from an earlier run (gas-macro and carbon-core flipping to Prophet, electricity-core flipping to Prophet from LSTM) — a direct visual consequence of hyperparameter tuning now running by default (`tune=True`) plus the `_select_best_mode_per_series` fix, both of which shift each model's fitted forecast enough to change the ranking in several already-close cells. These six charts remain the clearest single visual confirmation, across the whole Stage 1 output set, that SARIMA in particular balloons to several multiples of the winning model's error regardless of which model wins a given cell.

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

**Structure:** single row, columns `dof, chi2, chi2_p_value, cfi, tli, rmsea, gfi, aic, bic, srmr, cfi_ok, tli_ok, rmsea_ok, srmr_ok`. Latest run: `dof=59, chi2=9.94, chi2_p_value=1.0, cfi=−0.426, tli=−0.885, rmsea=0.0, gfi=0.775, aic=62.65, bic=406.15, srmr=0.0786, cfi_ok=False, tli_ok=False, rmsea_ok=True, srmr_ok=True`.

**CFI and TLI both fall well outside their mathematically valid [0,1] range**, and the pipeline's own logic correctly flags both as `False`. This coexists with a chi² (9.94 on 59 degrees of freedom) so small that `chi2_p_value` rounds to 1.0 — i.e. by the chi² test alone the model is not rejected at all, while by CFI/TLI it appears to fail badly. This combination is a known signature of a degenerate baseline/independence-model computation or of badly-scaled input variables (raw £-valued indicators like `hsval`/`carval`/`fihhmnnet1_dv` mixed with 0/1-coded security indicators) destabilizing `semopy`'s FIML likelihood — documented in the project's own source code as a known `semopy` 2.3.11 limitation. **CFI/TLI also vary run-to-run on materially the same data** (observed range across three recent refits: CFI −0.14 to −0.54, TLI −0.51 to −1.03) — a further, independent piece of evidence that these two statistics specifically, not the measurement model as a whole, are unreliable under this tool/dataset combination. RMSEA=0.0 (a suspiciously exact floor value) and SRMR=0.079 both look nominally acceptable in isolation, but their coexistence with the invalid CFI/TLI values means the fit-index block as a whole should not be read as a clean pass.

#### `outputs/ukhls_cor_sem/tables/structural_fit_indices_baseline.csv`

**Structure:** single row, same column set as above but for the second-order structural model. Latest run: `dof=61, chi2=83.84, chi2_p_value=0.0278, cfi=1.664, tli=1.849, rmsea=0.0011, gfi=−0.923, aic=−117.11, bic=204.92, srmr=718,007,968,755.89, cfi_ok=True, tli_ok=True, rmsea_ok=True, srmr_ok=False`.

**This is the more severe of the two fit-index anomalies in the project.** SRMR (hundreds of billions — the exact magnitude itself varies wildly between refits, e.g. 9.2 billion in one run vs. 718 billion in another, both equally invalid) is an obviously invalid value for a statistic that should normally range roughly 0–1, correctly flagged `srmr_ok=False`. GFI (negative) is also outside its valid [0,1] range but has no corresponding `_ok` flag to catch it. Most notably, **CFI and TLI both exceed the mathematical maximum of 1.0 for these indices, yet are marked `cfi_ok=True` and `tli_ok=True`** — strong evidence the pipeline's validity check tests only a lower bound (e.g. "≥0.90") and does not test an upper bound, allowing an out-of-range value to pass as "acceptable" silently.

#### `outputs/ukhls_cor_sem/tables/structural_paths_baseline.csv`

**Structure:** 4 rows (one per first-order factor's path from the second-order BASELINE factor), columns `first_order_factor, op, rval, Estimate, Est. Std, Std. Err, z-value, p-value`.

| Path | Estimate | Est. Std | z-value | p-value |
|---|---|---|---|---|
| OBJECT ~ BASELINE (marker) | 1.0 | **0.774** | — | — |
| CONDITION ~ BASELINE | −497.70 | **−1.0000** | −45.99 | 0.0 |
| PERSONAL ~ BASELINE | −568.45 | **−1.0000** | −45.99 | 0.0 |
| ENERGY ~ BASELINE | −1209.09 | **−1.0000** | −45.99 | 0.0 |

**This is a Heywood-case-like pattern, and — discovered this iteration — its *sign* is not stable across refits.** OBJECT loads at a plausible, sub-unity 0.77 on the second-order BASELINE factor (fixed positive by identification convention), but CONDITION, PERSONAL, and ENERGY all load at essentially ±1.00 — i.e. three of the four first-order COR resource dimensions appear empirically indistinguishable in magnitude from the overall BASELINE construct itself, which is not a substantively credible result. **In this run all three are negative** (unlike an earlier run's positive ≈0.99–1.00 values shown in a prior draft of this catalog) — since flipping all three free loadings' sign simultaneously fits the data identically well, this is not new information about the data, but direct confirmation that the second-order model's overall sign is arbitrary under the current specification. The very large raw (unstandardized) Estimates for these three paths (hundreds, vs. OBJECT's fixed marker value of 1.0) point to the same unstandardized-scale mismatch implicated in the fit-index anomalies above. This should be read as a numerical identification/scaling artifact requiring an explicit sign constraint to resolve — not as evidence that Condition, Personal, and Energy resources are truly interchangeable with the household's overall baseline resource stock, and not as evidence for any particular *direction* of that relationship either.

#### `outputs/ukhls_cor_sem/tables/cor_sem_factor_scores.csv`

**Structure:** 339,201 rows × 6 columns (an unnamed index plus `object_score, condition_score, personal_score, energy_score, baseline_resource_score`). Descriptive statistics (first-order factor scores are stable across refits, matching an earlier run to 3+ decimal places): `object_score` mean=−0.0538, sd=0.7297, range [−2.639, 2.946]; `condition_score` mean=−0.0018, sd=0.6711, range **[−3.963, 1.013]**; `personal_score` mean=0.0011, sd=0.8087, range [−2.029, 1.610]; `energy_score` mean=0.0001, sd=0.7946, range [−4.238, 2.311]; `baseline_resource_score` mean=0.0003, sd=**0.5028**, range [−2.392, 3.050]. Small counts of missing scores remain despite FIML: `object_score` 393 NaN, `condition_score` 120 NaN, `personal_score` 2,373 NaN, `energy_score` 15 NaN, `baseline_resource_score` 1 NaN (all out of 339,201). **`baseline_resource_score`'s own spread is not stable across refits** (sd 0.50 in this run vs. 0.72 in an earlier one, and its sign relative to the first-order scores is not fixed either) — a direct downstream consequence of the sign/scale instability documented in `structural_paths_baseline.csv` above; any regional or driver-analysis figure built from this column should be read with that instability in mind. `condition_score`'s distribution is notably asymmetric — its positive tail (max 1.013) is less than a third the length of its negative tail (min −3.963) — a >4:1 imbalance unlike the other three factors; this distributional oddity is consistent with CONDITION's weaker measurement quality documented above and is stable across refits (unlike the second-order score).

#### `outputs/ukhls_cor_sem/tables/fes_moderation_path.csv`

**Structure:** 3 rows (one per predictor in a single shared regression), columns `path, predictor, outcome, coef, std_err, t_stat, p_value, r_squared, n, significant`. All three rows share `path="COR-SEM: BASELINE x FES Delta -> fuel_to_income_ratio"`, `outcome=fuel_to_income_ratio`, `n=255,324`.

**This table's values are not stable across refits of the pipeline on materially the same data — reported here as three separate refits observed during this project's development, not as one authoritative row, since a single number would misrepresent how settled this result is:**

| Predictor | Refit A | Refit B | Refit C (latest) |
|---|---|---|---|
| baseline_score | −0.02521 (p=0.0) | −0.03163 (p=0.0) | +0.03961 (p=0.0) |
| fes_delta | −0.00063 (p=0.0) | −0.00088 (p=0.0) | −0.00080 (p=0.0) |
| baseline_x_fes (interaction) | +0.00004 (p=**0.608**) | +0.00049 (p<0.0001) | **−0.00073** (p<0.0001) |
| R² | 0.1085 | 0.0813 | 0.1375 |

**`fes_delta`'s main effect is the one stable, trustworthy number in this table** — negative and significant in all three refits, coefficient magnitude within a narrow band (−0.0006 to −0.0009). `baseline_score` and the interaction term (`baseline_x_fes`) are both significant in the two more recent refits but **flip sign between them** — direct evidence, not an assumption, that this specific regression's headline "moderation" result is currently unresolved rather than a settled null or a settled positive finding. See `structural_paths_baseline.csv` above for the diagnosed mechanism (the second-order model's unfixed sign) and `reports/06_methodology.md` §3.5 for the full methodological discussion. **Do not cite a single row of this table as "the" result without this context.**

### `outputs/ukhls_cor_sem/figures/`

- **`outputs/ukhls_cor_sem/figures/cor_sem_loadings.png`** — four side-by-side horizontal bar charts (one per factor: OBJECT, CONDITION, PERSONAL, ENERGY), each item's absolute standardized loading plotted against a red dashed "min=0.4" reference line. Confirms visually: OBJECT — hsval 0.55, carval 0.43 (barely clears the line), ncars 0.67, hsbeds 0.67, hsrooms 0.58 (all clear); CONDITION — bill_security 0.23 (well below the line), jbstat_security 0.45 (clears), tenure_security 0.37 (just below); PERSONAL — qfhigh_band 0.38 (just below), sf1_good 0.85 (well clear), health_good 0.59 (clears); ENERGY — fiyrinvinc_dv 0.37 (just below), fihhmnnet1_dv 0.58 (clears).
- **`outputs/ukhls_cor_sem/figures/cor_sem_loadings_polar.png`** — the same 13 loadings re-rendered in the project's house-style circular bar chart, grouped and colour-coded by factor (OBJECT, CONDITION, PERSONAL, ENERGY as four coloured arcs), with each item's exact rounded loading value displayed on its own bar. Purely a re-visualization of the loadings table; no new numbers.
- **`outputs/ukhls_cor_sem/figures/structural_paths_baseline.png`** — a horizontal bar chart, "COR-SEM: second-order BASELINE structural loadings," visually confirming the near-1.0-height bars for ENERGY, PERSONAL, and CONDITION against OBJECT's distinctly shorter (~0.71) bar — the single clearest visual representation of the Heywood-case-like pattern documented above.

---

## Stage 2c — COR-CVAE

### `outputs/ukhls_cor_cvae/tables/`

#### `outputs/ukhls_cor_cvae/tables/cvae_latent_scores.csv`

**Structure:** 253,913 rows × 5 columns (an unnamed index plus `cvae_object_z, cvae_condition_z, cvae_personal_z, cvae_energy_z`), no missing values. The row count (253,913) is notably smaller than the SEM's factor-score table (339,201) because the CVAE, unlike the SEM's FIML estimator, requires complete cases and is trained/scored only on households with no missing values across its input items. Latest run (post the household-grouped train/validation split fix, `reports/06_methodology.md` §3.6): `cvae_object_z` mean=0.0234, sd=0.2259, range [−1.369, 0.615]; `cvae_condition_z` mean=−0.0796, sd=0.2031, range [−1.222, 0.799]; `cvae_personal_z` mean=−0.0325, sd=0.1560, range [−0.625, 0.750]; `cvae_energy_z` mean=0.0058, sd=0.1924, range [−0.856, 0.993]. The naming convention (`cvae_object_z` etc.) is a positional assignment (z1→object, z2→condition, z3→personal, z4→energy) rather than a verified empirical correspondence — see the alignment table below for how well that naming actually holds up (spoiler: not for z4).

#### `outputs/ukhls_cor_cvae/tables/cvae_sem_alignment.csv`

**Structure:** 16 rows (4 latent dimensions × 4 SEM factors), columns `latent_dim, sem_factor, pearson_r`. Latest run (post the grouped-split fix, `reports/06_methodology.md` §3.6):

| Latent dim | object_score | condition_score | personal_score | energy_score |
|---|---|---|---|---|
| z1 (cvae_object_z) | **−0.792** | −0.412 | −0.425 | −0.624 |
| z2 (cvae_condition_z) | 0.454 | **0.803** | 0.397 | 0.559 |
| z3 (cvae_personal_z) | 0.393 | 0.278 | **0.752** | 0.285 |
| z4 (cvae_energy_z) | **0.752** | 0.588 | 0.139 | 0.693 |

(Bold = the "intended" diagonal pairing implied by the naming convention.)

**Three of four dimensions are now cleanly, dedicatedly aligned — an improvement over an earlier run — but the fourth has gotten worse, not better.** z1 (object, −0.792, clear of its next-highest at −0.624), z2 (condition, 0.803, clear of its next-highest at 0.559), and z3 (personal, 0.752, clear of its next-highest at 0.393) all show a clean, well-separated alignment to their intended factor, each a stronger separation than an earlier run of this same architecture. **z4 does not**: its largest correlation is now with `object_score` (0.752), *exceeding* its correlation with its own intended factor `energy_score` (0.693) — bolded in the table above at its "intended" cell for consistency, but this is in fact the *second*-largest value in that row, not the largest. This is worse than an earlier run's finding of "partial entanglement" (where z4's own-factor correlation was still the largest, just not by much) — z4 has moved to being more accurately described as aligned with object than with energy. Combined with `object_score`'s reproducible sign anomaly in the driver-analysis and forward-prediction models (Stage 3/5 below), this suggests the object/asset resource dimension is genuinely harder for both estimation methods to cleanly separate from the rest of the resource construct, not an artifact specific to either method.

#### `outputs/ukhls_cor_cvae/tables/cvae_training_history.csv`

**Structure:** 300 rows (epoch 0–299) × 11 columns: `epoch, train_total, val_total, train_recon, val_recon, train_kl, val_kl, train_align, val_align, train_bce, val_bce`. Latest run (post the household-grouped split fix — the train/validation gap here is the more honest, "real" gap now that leakage is closed): epoch 0: `train_total=11.42, val_total=10.82, train_recon=1.445, val_recon=1.406, train_kl=8.236, val_kl=7.697, train_align=0.796, val_align=0.794, train_bce=0.945, val_bce=0.926`. Epoch 299 (final): `train_total=1.715, val_total=1.707, train_recon=0.994, val_recon=0.988, train_kl=0.123, val_kl=0.121, train_align=0.240, val_align=0.240, train_bce=0.358, val_bce=0.358`. `train_align`'s maximum (0.816) occurs at epoch 18, not epoch 0, meaning it rises before it falls — the same annealed alignment-loss shape as before the split fix, now on genuinely held-out validation households rather than partially-leaked ones. The train/validation gap remains small throughout (final total loss 1.715 vs. 1.707, under 0.5% relative difference) — reassuring given the split fix removed the specific leakage path that could have inflated this number; the gap staying small even after closing that path is itself evidence the model generalizes rather than memorizes.

#### `outputs/ukhls_cor_cvae/tables/cvae_counterfactual_fes_shift.csv`

**Structure:** 253,913 rows × 11 columns: `hidp, wave, interview_year, gor_dv, row_index, pred_prob_current, pred_prob_forecast, pred_prob_shift, dist_from_resilient_current, dist_from_resilient_forecast, dist_shift`. Latest run (post the grouped-split fix): `pred_prob_current` mean=0.2826, sd=0.0201, range [0.161, 0.340]; `pred_prob_forecast` mean=0.2825, sd=0.0202, range [0.161, 0.343]; `pred_prob_shift` mean=**−0.0001**, sd=0.0030, range **[−4.4, +3.9 percentage points]**; `dist_from_resilient_current` mean=0.357, sd=0.202; `dist_from_resilient_forecast` mean=0.358, sd=0.202; `dist_shift` mean=**+0.0009**, sd=0.0413, range [−0.693, +0.530]. The population-mean probability shift is essentially zero and has **flipped sign** relative to an earlier run (which found a small positive mean shift) — consistent with the same FES-attachment-bug correction discussed under `fes_moderation_path.csv` above, which affects exactly the realised/forecast FES values this simulation swaps between. The individual-level range remains wide (probability shift roughly ±4 percentage points, distance shift −0.69 to +0.53), meaning the near-null population average conceals substantial heterogeneity in how differently individual households respond to the counterfactual scenario — this heterogeneity finding is the more robust part of this table; the population-average *sign* should not be leaned on given it has changed between runs.

### `outputs/ukhls_cor_cvae/figures/`

- **`outputs/ukhls_cor_cvae/figures/cvae_sem_alignment_heatmap.png`** — a signed heatmap (green-to-red diverging colormap, −1 to +1) of the alignment matrix above; the values match `cvae_sem_alignment.csv` exactly to 2 decimal places.
- **`outputs/ukhls_cor_cvae/figures/cvae_alignment_polar.png`** — the project's house-style circular bar chart, |r| per latent dimension shown as four colour-coded quadrants (one per z1–z4), each containing four wedges (one per SEM factor), each pairing's exact |r| value displayed directly on its bar. Latest run: z2/condition (0.80) and z4/object (0.75, note — *not* z4's own-factor pairing, see the alignment table above) are the two strongest pairings; z3/energy (0.29) and z3/condition (0.28) the weakest.
- **`outputs/ukhls_cor_cvae/figures/cvae_training_curves.png`** — a 4-panel line plot (total/reconstruction/KL/alignment losses, solid=train, dashed=validation) across all 300 epochs. Total and reconstruction losses decay smoothly and monotonically; KL decays steeply in the first ~50 epochs then flattens near-zero for the remainder (consistent with the latent posterior converging close to the prior); alignment shows the rise-then-fall hump described above, peaking near epoch 30–40 before declining to its final value. Train and validation curves are visually indistinguishable on all four panels throughout — no visible overfitting.

---

## Stage 2 (merged panel)

### `outputs/ukhls_cleaned/ukhls_panel.csv`

339,201 data rows × **95** columns (339,202 lines including the header) — up from 62 in an earlier version of this panel, reflecting this iteration's additions: `family_composition_group`, `lone_parent`, `large_family`, `prepayment_meter` (household/family composition, Section 3.3 of `reports/06_methodology.md`); `has_employed_adult`, `has_fulltime_worker`, `has_parttime_worker`, `has_selfemployed_worker`, `workless_household`, `employment_group` (employment status); `ieqmoecd_dv` (equivalisation factor); plus the COR-SEM/COR-CVAE latent scores (`object_score` through `cvae_energy_z`) merged back onto the raw/derived panel columns, which an earlier version of this catalog entry did not yet include in its column count. This is the single merged file every later stage reads from. All six raw `inoutflows*` variables are retained even though half of them (`inoutflows2/3/4`) are ~97% missing — a deliberate raw-data-retention choice distinct from the derived-variable pruning applied in the SEM/driver-analysis feature lists.

---

## Stage 3 — Vulnerability Identification

### `outputs/ukhls_vulnerability/tables/`

#### `outputs/ukhls_vulnerability/tables/stage3_vulnerability_scores.csv`

**Structure:** 339,201 rows, columns: an unnamed index, `fuzzy_resource_depleted, fuzzy_resource_resilient, fuzzy_vulnerable_to_loss` (three fuzzy c-means membership scores summing to 1 per row), `fuzzy_partition_coefficient` (a single constant, 0.4559, broadcast to every row — a global model-fit statistic, not a per-observation quantity), and `oneclass_anomaly_score`. 85,288 rows (25.1%) are NaN across all five score columns, leaving 253,913 valid rows — matching the CVAE's complete-case count exactly. Among valid rows, a hard cluster assignment via arg-max of the three fuzzy memberships gives: "vulnerable to loss" dominant in 96,546 rows (38.0%), "resource resilient" in 87,318 (34.4%), and "resource depleted" in 70,049 (27.6%). `oneclass_anomaly_score` is strongly right-skewed: mean −60.0, median −81.9, max 771.5 — most households score as broadly "typical," with a long tail of extreme anomaly scores.

#### `outputs/ukhls_vulnerability/tables/stage3_validation_against_objective_ratio.csv`

**Structure:** 2 rows, columns `method, n, pearson_r_vs_ratio, spearman_r_vs_ratio, auc_vs_high_fuel_vulnerable`.

| Method | n | Pearson r | Spearman r | AUC |
|---|---|---|---|---|
| fuzzy_resource_depleted | 253,913 | 0.2615 | 0.3681 | **0.7448** |
| oneclass_anomaly_score | 253,913 | 0.1722 | 0.1326 | 0.6208 |

Fuzzy c-means "Resource Depleted" membership clearly outperforms as a vulnerability proxy (AUC 0.74, a genuinely useful discriminator against the objective fuel/income≥10% target). The one-class SVM anomaly score is weaker but has improved somewhat relative to an earlier run (AUC 0.62 vs. an earlier 0.57), and its Spearman correlation (0.133) now shows a real, if modest, rank-order relationship with the objective ratio; the anomaly detector still appears to be picking up a somewhat different, more general kind of statistical "unusualness" than fuel poverty specifically, but less exclusively so than in an earlier run.

#### `outputs/ukhls_vulnerability/tables/driver_analysis_logistic_regression.csv`

**Structure:** **17** rows (one per predictor — extended this iteration from 13, adding `large_family`, `lone_parent`, `workless_household`, and the equivalisation factor `ieqmoecd_dv`), columns `predictor, coef, odds_ratio, std_err, p_value, significant, n`; n=103,621 for all rows (the national, pooled model).

| Predictor | Coefficient | Odds ratio | p-value | Significant |
|---|---|---|---|---|
| financial_strain_score | 1.9270 | **6.87** | 0.0 | Yes |
| large_family | 0.5175 | 1.68 | 0.0 | Yes |
| tenure_security | 0.4468 | 1.56 | 0.0 | Yes |
| health_good | 0.3192 | 1.38 | 0.0 | Yes |
| hsbeds | 0.2843 | 1.33 | 0.0 | Yes |
| workless_household | 0.2060 | 1.23 | 0.0 | Yes |
| lone_parent | 0.2013 | 1.22 | 0.0 | Yes |
| hsrooms | 0.0993 | 1.10 | 0.0 | Yes |
| dvage | −0.0065 | 0.99 | 0.0 | Yes |
| fes_delta | −0.0609 | **0.94** | **0.0** | **Yes** |
| heatch | −0.0748 | 0.93 | 0.165 | No |
| sf1_good | −0.1895 | 0.83 | 0.0006 | Yes |
| ncars | −0.1898 | 0.83 | 0.0 | Yes |
| bill_security | −0.2406 | 0.79 | 0.0011 | Yes |
| qfhigh_band | −0.6954 | 0.50 | 0.0 | Yes |
| ieqmoecd_dv | −1.2870 | 0.28 | 0.0 | Yes |
| jbstat_security | −1.8113 | **0.16** | 0.0 | Yes |

Financial/psychological strain dominates every other predictor by a wide margin (OR 6.87). **`fes_delta` is now significant** (OR 0.94, p<0.0001) — a correction from an earlier run of this same analysis, which found it non-significant (OR 1.02, p=0.182) under a data-pipeline defect since fixed (`reports/02_findings_report.md` §8's changelog). Two signs are worth flagging as genuinely counter-intuitive rather than glossed over: `health_good` (self-rated health "good") *raises* the odds of vulnerability (OR 1.38, significant), and both `hsrooms` and `hsbeds` (more rooms/bedrooms) also raise odds — plausibly because larger, older homes cost more to heat, and self-rated "good health" may correlate with older homeowners in exactly this kind of housing stock, though this project does not test that specific mechanism directly. `tenure_security` raising risk (OR 1.56) is similarly non-obvious at first glance. Among the new covariates, `large_family` (OR 1.68) and `lone_parent` (OR 1.22) raise vulnerability odds as expected, `workless_household` (OR 1.23) is directionally consistent with `jbstat_security`'s protective effect, and `ieqmoecd_dv` (OR 0.28) is strongly protective. Only `heatch` is not statistically significant in this run.

#### `outputs/ukhls_vulnerability/tables/driver_analysis_by_region.csv`

**Structure:** now 204 rows (17 predictors × 12 regions, up from 156/13 — see the national table above for the 4 covariates added this iteration), same columns as the national table above plus a `region` column. Region-level numbers below are from an earlier run predating this iteration's covariate additions and `fes_delta` correction; the qualitative pattern (financial strain dominant everywhere, magnitude varying substantially by region) is expected to hold, but exact odds ratios should be re-pulled from the current file before being quoted precisely. `financial_strain_score` is the top-ranked driver in every one of the 12 regions, but its magnitude varies enormously: highest in Northern Ireland (OR 14.67, n=1,879) and South East (OR 14.57, n=12,880), lowest in London (OR 2.86, n=15,231) — over a 5× spread in effect size by geography. `jbstat_security` is consistently and strongly protective everywhere (OR range approximately 0.10–0.21), the most universal driver after financial strain. Regional idiosyncrasies include Northern Ireland's `health_good` flipping to a negative, non-significant coefficient (−0.163) unlike the national/most-region pattern, and `tenure_security` being non-significant in about half the regions while large and significant in North East (OR 3.24) specifically. Smaller-sample regions (Northern Ireland n=1,879, North East n=4,577) carry correspondingly larger standard errors (e.g. NI's financial_strain_score std_err=0.662 vs. London's 0.269) and should be read with that extra uncertainty in mind.

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

#### `outputs/ukhls_vulnerability/tables/policy_vulnerability_by_family_composition.csv` — new this iteration

**Structure:** 6 rows, columns `group, pct_vulnerable, n`: **Lone parent, 1-2 children 15.13% (n=16,378)**; Lone parent, 3+ children 13.18% (n=2,165); No children 9.00% (n=229,491); Couple, 3+ children 5.27% (n=11,160); Other multi-adult, with children 4.32% (n=24,477); Couple, 1-2 children 3.67% (n=55,392). Lone-parent households are, by a wide margin, the most fuel-vulnerable family-composition group — consistent with `driver_analysis_logistic_regression.csv`'s significant `lone_parent` covariate (OR 1.22) and with JRF's own child-poverty statistics (external validation below). Collapses to `policy_vulnerability_by_family_type_collapsed.csv` (2 rows: Lone parent 14.90%/n=18,543 n-weighted mean, Couple with children 3.94%/n=66,552) for the JRF comparison.

#### `outputs/ukhls_vulnerability/tables/policy_vulnerability_by_employment.csv` — new this iteration

**Structure:** 3 rows, columns `group, pct_vulnerable, n`: **Workless household 14.54% (n=124,412)**; Part-time only 9.29% (n=36,234); Full-time or self-employed 3.25% (n=176,250). A clean, monotonic gradient — full employment is strongly protective, consistent with `jbstat_security`'s status as the strongest protective driver in the logistic regression above. Collapses to `policy_vulnerability_by_work_status_collapsed.csv` (2 rows: Not in work 14.54%/n=124,412, In work 4.28%/n=212,484) for the JRF comparison.

#### Rationing/self-disconnection evidence tables (2 files) — new this iteration

- **`outputs/ukhls_vulnerability/tables/rationing_evidence_prepayment.csv`** — 2 rows, columns `high_fuel_vulnerable, pct_prepayment_meter, n`: Not vulnerable (ratio<10%) 10.5% on a prepayment meter (n=235,066); Vulnerable (ratio≥10%) 19.0% (n=20,258). Tests a specific blind spot in the ratio-based target: a household that copes by rationing energy use would show a *lower*, not higher, spend ratio. The prepayment-meter rate running *higher*, not lower, among already-flagged-vulnerable households is reassuring — some rationing behaviour is occurring, but it is not large or systematic enough to be hiding a materially different population within "not vulnerable."
- **`outputs/ukhls_vulnerability/tables/rationing_evidence_inoutflows12.csv`** — 2 rows, the same test using a direct self-report ("reduced usage of utilities," waves m/o's cost-of-living-crisis module only, far smaller n): Not vulnerable 33.5% (n=1,925); Vulnerable 40.4% (n=371). Same direction, same reassuring conclusion, on a narrower but more literal proxy.

#### Equivalisation robustness check (2 files) — new this iteration

`src.ukhls_vulnerability_classification.run_equivalisation_robustness_check` (`reports/06_methodology.md` Section 4.2) tests whether the project's primary, deliberately **unequivalised** `fuel_to_income_ratio` target (mirroring the UK's own official 10%-of-income fuel-poverty definition) is silently distorted by household size — the same concern JRF's own income-poverty methodology uses equivalisation (Modified OECD scale, via `ieqmoecd_dv`) to correct for.

- **`outputs/ukhls_vulnerability/tables/equivalisation_robustness_check_correlations.csv`** — 2 rows, columns `method, n, pearson_r_vs_equivalised_ratio, auc_vs_high_fuel_vulnerable_equivalised`: `fuzzy_resource_depleted` (n=253,716) correlates 0.208 with the equivalised ratio and reaches AUC=0.6695 against the equivalised binary label; `oneclass_anomaly_score` correlates 0.151 and reaches AUC=0.552. Both are markedly **weaker** against the equivalised ratio than the same two methods' own validation against the primary, unequivalised ratio (Stage 3's headline `stage3_validation_against_objective_ratio.csv`: fuzzy r=0.262/AUC=0.745, one-class r=0.172/AUC=0.621) — expected, since the unsupervised methods were never tuned toward the equivalised construct, and this comparison is a robustness check on the *target*, not a re-validation of the clustering methods themselves.
- **`outputs/ukhls_vulnerability/tables/equivalisation_robustness_check_flips_by_hhsize.csv`** — 5 rows, columns `hhsize_bucket, pct_flipped, n`: 1-person households 0.0% flipped (n=61,422 — by construction, equivalising a 1-person household's income changes it by a fixed divisor close to 1, essentially never crossing the 10% threshold differently); 2-person 10.5% (n=88,030); 3-person 16.6% (n=42,006); 4-person 22.9% (n=40,525); **5+-person 42.5% (n=23,142)** — a clean, monotonically increasing gradient. **This is a genuine, substantial size-sensitivity finding**: for the largest households, equivalisation changes the `high_fuel_vulnerable` classification for over 4 in 10 rows — exactly the kind of size-driven misclassification JRF's own methodology is designed to avoid, and a concrete, quantified caveat on this project's choice to keep the primary target unequivalised (a choice made deliberately, to match the UK's own official fuel-poverty standard, not by oversight — but one whose cost is now measured rather than assumed).

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

#### JRF external-validation benchmark tables (6 files, up from 4)

- **`outputs/ukhls_vulnerability/tables/jrf_poverty_benchmark_region.csv`** — 12 rows, JRF's own relative-poverty rate (AHC) by UK nation/region, averaged 2021/22–2022/23 (Table 6, p.51): West Midlands 27 (highest); North West 25; London 24; Yorkshire and the Humber 23; Scotland, North East, Wales all 21 (tied); East Midlands 20; South West, South East both 19 (tied); East of England 18; **Northern Ireland 17 (lowest of all 12)**.
- **`outputs/ukhls_vulnerability/tables/jrf_poverty_benchmark_ethnicity.csv`** — 6 rows, only ethnicity groups with a number explicitly stated in JRF's report text (p.9/42), 2020/21–2022/23: Bangladeshi 56 (highest); Pakistani 49; Black African 40; Any other Asian background 34; Black Caribbean 30; White 19 (lowest). Indian, Chinese, Mixed/multiple, Any other Black background, and Other ethnic group are deliberately excluded — JRF shows these only in a chart (Figure 13/25) with no stated rate to cite, and the project's own methodology avoids reading numbers off chart pixels.
- **`outputs/ukhls_vulnerability/tables/jrf_poverty_benchmark_disability.csv`** — 2 rows, JRF Table 8 (p.67): "Contains disabled adult" 29; "No disabled adult" 19.
- **`outputs/ukhls_vulnerability/tables/jrf_poverty_benchmark_tenure.csv`** — 4 rows, JRF Table 10 (p.95), 2022/23: Social renting 44 (highest); Private renting 35; Owned outright 14; Buying with mortgage 10 (lowest).
- **`outputs/ukhls_vulnerability/tables/jrf_poverty_benchmark_family_type.csv`** — **new this iteration**, 2 rows, JRF Table 5 (p.36) — Lone parent 44 (highest), Couple with children 25. Note: these are JRF's stated *child* poverty rates by family type, a different unit from the household/adult-level rates in every other benchmark table here.
- **`outputs/ukhls_vulnerability/tables/jrf_poverty_benchmark_work_status.csv`** — **new this iteration**, 2 rows, JRF p.76 — Not in work 43 (highest), In work 12.

#### External-validation comparison tables (6 files, up from 4)

- **`outputs/ukhls_vulnerability/tables/external_validation_region_comparison.csv`** — 12 rows, merging `policy_vulnerability_by_region.csv` with the JRF region benchmark, plus computed rank columns. Correlation across all 12 regions: Pearson r=−0.10, Spearman ρ=0.33. **Excluding Northern Ireland (n=11): Pearson r=0.68, Spearman ρ=0.73** — Northern Ireland alone flips the sign of the linear correlation and roughly doubles the rank correlation, the single most important statistic in the whole external-validation exercise.
- **`outputs/ukhls_vulnerability/tables/external_validation_ethnicity_comparison.csv`** — 6 rows. Correlation: Pearson r=0.27, Spearman ρ=0.14 (n=6) — the weakest of the six dimensions. Bangladeshi is JRF's #1 most-poor ethnic group (56%) but only #4 of 6 on this project's measure (9.50%); Black Caribbean is the mirror-image divergence, #1 here (14.06%) but #5 of 6 on JRF (30%).
- **`outputs/ukhls_vulnerability/tables/external_validation_disability_comparison.csv`** — 2 rows. Rank order agrees exactly, but with only 2 categories the correlation coefficient is mathematically undefined (reported and displayed as NaN in the corresponding figures).
- **`outputs/ukhls_vulnerability/tables/external_validation_tenure_comparison.csv`** — 4 rows. Correlation: Pearson r=0.68, Spearman ρ=0.80 (n=4) — the strongest external agreement among dimensions with n>2. Owned outright is the one rank-swap (2nd here vs. 3rd on JRF), consistent with the fuel-vs-income divergence discussed above.
- **`outputs/ukhls_vulnerability/tables/external_validation_family_type_comparison.csv`** — **new this iteration**, 2 rows (built from `policy_vulnerability_by_family_type_collapsed.csv`, which n-weight-averages this project's 5-category family-composition breakdown down to JRF's 2). Correlation: Pearson r=1.00, Spearman ρ=1.00 (n=2) — ranks agree (Lone parent higher on both measures), though n=2 is a necessarily weak test.
- **`outputs/ukhls_vulnerability/tables/external_validation_work_status_comparison.csv`** — **new this iteration**, 2 rows (built from `policy_vulnerability_by_work_status_collapsed.csv`). Correlation: Pearson r=1.00, Spearman ρ=1.00 (n=2) — ranks agree (Not in work higher on both measures), same n=2 caveat.

#### Northern Ireland oil-heating evidence tables (2 files)

- **`outputs/ukhls_vulnerability/tables/ni_oil_heating_evidence_by_region.csv`** — 12 rows, columns `region, pct_using_oil_heating, pct_vulnerable, n`: Northern Ireland 71.2% oil-heating usage, 15.6% vulnerable (n=21,486) — the next-highest region for oil-heating usage is Wales at only 9.8% (n=22,768), a **7.3× gap**; South West 9.0%/6.1%; East of England 7.7%/6.5%; Scotland 6.7%/8.4%; East Midlands 3.9%/8.0%; South East 3.8%/6.0%; West Midlands 3.6%/9.8%; Yorkshire and the Humber 2.3%/8.4%; North East 1.8%/8.5%; North West 1.5%/8.9%; London 0.1%/6.9%. This is the key exogenous variable explaining Northern Ireland's outlier position in every other regional table in this project.
- **`outputs/ukhls_vulnerability/tables/ni_oil_heating_evidence_within_ni.csv`** — 2 rows, a controlled within-Northern-Ireland comparison by oil-heating status: `has_oil=False` — mean annual fuel spend £1,243, 12.3% vulnerable, n=6,183; `has_oil=True` — mean annual fuel spend £2,017, **20.9% vulnerable**, n=15,303 (6,183+15,303=21,486, matching Northern Ireland's total n exactly). Oil-heating NI households spend 62% more annually on fuel and are vulnerable at nearly double the rate of non-oil NI households in the same wave and region — this within-region comparison isolates oil-heating exposure from every other NI-specific confound and is the strongest single piece of evidence for the causal story that oil-market volatility (lump-sum purchased, price-volatile, outside Ofgem's price cap) drives Northern Ireland's outlier status.

### `outputs/ukhls_vulnerability/figures/` (30 files, up from 22, each with a matching vector `.pdf`)

- **`outputs/ukhls_vulnerability/figures/driver_analysis_by_region.png`** — a horizontal bar chart, "Does the strongest national driver hold everywhere?", showing `financial_strain_score`'s odds ratio by region sorted descending: Northern Ireland (~14.7) and South East (~14.6) highest, East Midlands (9.6), North East (8.8), North West/Scotland/West Midlands (~7.9), Yorkshire (7.5), South West (7.2), East of England (6.6), Wales (5.3), and **London lowest (~2.9)** — a roughly 5× spread across regions, with a reference line at OR=1.
- **`outputs/ukhls_vulnerability/figures/policy_driver_odds_ratios.png`** — the national logistic-regression forest/bar plot, colour-coded green (protective) / red (risk-raising) / grey (non-significant), with human-readable predictor labels. Financial/psychological strain's bar is overwhelmingly the longest (OR≈6.85). Confirms the sign pattern in `driver_analysis_logistic_regression.csv` exactly: green = employment status security, educational qualification, number of cars, bill payment security; grey = general health satisfaction, has central heating, FES Delta; red = age, number of bedrooms/rooms, self-rated health (good), housing tenure security, financial/psychological strain.
- **`outputs/ukhls_vulnerability/figures/policy_vulnerability_by_wave.png`** — a line chart, 2009–2023 (waves a–o), each point annotated with its calendar year, visually confirming the U-shape and cost-of-living-crisis spike. **The chart's title states "2009-2024," but the last plotted data point (wave o) is labelled 2023 and no 2024 data exists in the underlying wave-level table** — a minor title/labelling inaccuracy worth correcting.
- **`outputs/ukhls_vulnerability/figures/policy_vulnerability_by_region.png`** — a ranked horizontal bar chart of the 12 regions, coloured red (above median) or green (below median) at a visual split between Scotland (8.39%, red) and Yorkshire and the Humber (8.37%, green) — an essentially coin-flip threshold given how close those two values are. Northern Ireland's bar is roughly 60% longer than the next-longest (West Midlands).
- **`outputs/ukhls_vulnerability/figures/policy_vulnerability_by_ethnicity.png`** — the ethnicity breakdown as a ranked horizontal bar chart, red/green split at the median; confirms Black Caribbean highest and Any other Asian background lowest.
- **`outputs/ukhls_vulnerability/figures/policy_vulnerability_by_disability.png`** — a 2-bar chart: red "Contains disabled adult" (12.86%) vs. green "No disabled adult" (10.33%).
- **`outputs/ukhls_vulnerability/figures/policy_vulnerability_by_tenure.png`** — a 5-bar chart; "Owned outright" is coloured red (above median), ranking visually just below Social renting and above Private renting — the counter-intuitive tenure ordering discussed above is directly visible here.
- **`outputs/ukhls_vulnerability/figures/policy_vulnerability_by_family_composition.png`** — new this iteration, a 6-bar ranked chart matching `policy_vulnerability_by_family_composition.csv` exactly: Lone parent (1-2 children) tallest at 15.13%, Couple (1-2 children) shortest at 3.67% — visually a clean split between the two lone-parent categories (both red, above-median) and the four remaining categories (green), rather than a smooth monotonic gradient by household size alone.
- **`outputs/ukhls_vulnerability/figures/policy_vulnerability_by_employment.png`** — new this iteration, a 3-bar chart matching `policy_vulnerability_by_employment.csv`: Workless household (14.54%, red) roughly 4.5× Full-time or self-employed (3.25%, green), with Part-time only (9.29%) in between — the same clean monotonic employment-security gradient visible in `jbstat_security`'s driver-regression coefficient, now shown as its own standalone breakdown for the first time.
- **`outputs/ukhls_vulnerability/figures/rationing_evidence_prepayment.png`** — new this iteration, a 2-bar chart ("Not vulnerable" vs. "Vulnerable"), data labels printed on each bar matching `rationing_evidence_prepayment.csv` exactly (10.5%, 19.0%) — visually confirming the reassuring direction (prepayment-meter usage is *higher*, not lower, among already-flagged-vulnerable households) at a glance.
- **`outputs/ukhls_vulnerability/figures/equivalisation_robustness_check_flips_by_hhsize.png`** — new this iteration, a 5-bar chart ("1" through "5+" household-size buckets), data labels matching `equivalisation_robustness_check_flips_by_hhsize.csv` exactly (0.0%, 10.5%, 16.6%, 22.9%, 42.5%) — a visually unambiguous monotonic staircase, the clearest single chart in the project for communicating "the unequivalised target misclassifies large households disproportionately."
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
- **`outputs/ukhls_vulnerability/figures/external_validation_family_type_bars.png`** — new this iteration, the dual-axis bar-chart companion for family type (2 groups: Lone parent 14.90%/JRF 44%, Couple with children 3.94%/JRF 25%); title states "Spearman ρ=1.00 | Pearson r=1.00 (n=2)" — mathematically a perfect correlation with only 2 points, so the coefficient itself carries little statistical weight beyond confirming the ranks agree.
- **`outputs/ukhls_vulnerability/figures/external_validation_family_type_scatter.png`** — the scatter-plot companion; with n=2 the "best-fit line" passes exactly through both points by construction — a visual reminder (same caveat as the disability scatter above) that a 2-point fit line should not be read as evidence of a robust linear relationship.
- **`outputs/ukhls_vulnerability/figures/external_validation_work_status_bars.png`** — new this iteration, the dual-axis bar-chart companion for work status (2 groups: Not in work 14.54%/JRF 43%, In work 4.28%/JRF 12%); title states "Spearman ρ=1.00 | Pearson r=1.00 (n=2)", the same n=2 caveat as family type above.
- **`outputs/ukhls_vulnerability/figures/external_validation_work_status_scatter.png`** — the scatter-plot companion; same exact-fit-through-2-points caveat as family type's scatter.

---

## Stage 4 — Policy Geography Maps

### `outputs/ukhls_policy_maps/tables/`

#### `outputs/ukhls_policy_maps/tables/map1_resource_stress_hotspot.csv`

**Structure:** 12 rows, columns `region, baseline_resource_mean, vulnerable_pct, n, baseline_tertile, vulnerable_tertile, policy_tier`.

| Region | Baseline resource (mean) | % vulnerable | n | Policy tier |
|---|---|---|---|---|
| Northern Ireland | 0.0929 | **15.62%** | 21,486 | **High resource / High vulnerability** |
| West Midlands | 0.0365 | 9.75% | 27,592 | High resource / High vulnerability |
| Wales | 0.0644 | 9.12% | 22,768 | High resource / High vulnerability |
| North West | 0.0248 | 8.86% | 34,392 | Mid resource / High vulnerability |
| North East | 0.0533 | 8.51% | 12,643 | High resource / Mid vulnerability |
| Scotland | −0.0512 | 8.39% | 30,538 | Low resource / Mid vulnerability |
| Yorkshire and the Humber | 0.0262 | 8.37% | 27,761 | Mid resource / Mid vulnerability |
| East Midlands | 0.0244 | 8.05% | 24,339 | Mid resource / Mid vulnerability |
| London | −0.0597 | 6.88% | 41,318 | Low resource / Low vulnerability |
| East of England | −0.0150 | 6.50% | 28,545 | Mid resource / Low vulnerability |
| South West | −0.0172 | 6.08% | 27,323 | Low resource / Low vulnerability |
| South East | **−0.0577** | **5.96%** | 40,315 | Low resource / Low vulnerability |

**The bivariate tier classification has changed substantially between refits** — Northern Ireland now falls in "High resource / High vulnerability" rather than the earlier run's "Low resource / High vulnerability," and several other regions' tiers have shifted too. This is a direct, visible downstream consequence of `baseline_resource_score`'s sign/scale instability documented under `cor_sem_factor_scores.csv` above — the *ranking* by vulnerability (right-hand column) is stable across refits (Northern Ireland highest, South East lowest, matching every other regional table in this project), but the *baseline-resource* half of this bivariate classification, and hence the specific tier label assigned to each region, is not currently reliable and should not be quoted without this caveat until the underlying second-order model is sign-constrained.

#### `outputs/ukhls_policy_maps/tables/policy_map2_fuzzy_membership.csv`

**Structure:** 12 rows, columns `region, mean_vulnerable_to_loss, pct_near_boundary, n`.

| Region | Mean fuzzy membership | % near boundary | n |
|---|---|---|---|
| North East | 0.3900 | 26.64% | 12,643 |
| East Midlands | 0.3898 | 25.38% | 24,339 |
| Yorkshire and the Humber | 0.3897 | 25.70% | 27,761 |
| Wales | 0.3886 | 24.01% | 22,768 |
| Scotland | 0.3881 | 23.79% | 30,538 |
| West Midlands | 0.3847 | 23.77% | 27,592 |
| North West | 0.3837 | 25.50% | 34,392 |
| South West | 0.3827 | 21.49% | 27,323 |
| East of England | 0.3765 | 20.48% | 28,545 |
| South East | 0.3700 | 21.08% | 40,315 |
| Northern Ireland | 0.3628 | **8.13%** | 21,486 |
| London | **0.3580** | 22.10% | 41,318 |

**The two statistics in this table now tell different stories about which region is the outlier — a change from an earlier run, where Northern Ireland was both.** London, not Northern Ireland, now has the *lowest mean membership* (0.358) — but Northern Ireland's near-boundary share (8.13%) remains, by a wide margin, the lowest of any region, less than half of London's (22.1%) and well under a third of every other GB region's (20.5%–26.6%). Northern Ireland's fuzzy-membership distribution is still the most polarized/bimodal of any region — its households sort more decisively into "clearly vulnerable" or "clearly not," with far fewer borderline cases — a finding that has held stable across refits even as the specific mean-membership ranking above has shifted. This is a useful illustration of why the *shape* statistic (near-boundary share) and the *level* statistic (mean membership) should be read as two separate questions, not assumed to move together.

#### `outputs/ukhls_policy_maps/tables/map3_vulnerability_vector_shift.csv`

**Structure:** 12 rows, columns `region, mean_current, mean_forecast, mean_shift, n` (using each household's own realised FES vs. the shared forecast shock — the CVAE counterfactual — a simulation, not observed data).

| Region | Current prob. | Forecast prob. | Shift | n |
|---|---|---|---|---|
| London | 0.28033 | 0.28044 | +0.000115 (only region to rise) | 32,524 |
| South East | 0.28686 | 0.28685 | −0.000004 (~flat) | 31,315 |
| East of England | 0.28498 | 0.28495 | −0.000030 | 21,150 |
| South West | 0.28545 | 0.28539 | −0.000063 | 19,610 |
| Scotland | 0.28196 | 0.28186 | −0.000104 | 22,233 |
| West Midlands | 0.28185 | 0.28168 | −0.000167 | 22,106 |
| North West | 0.28101 | 0.28084 | −0.000177 | 28,537 |
| Yorkshire and the Humber | 0.28125 | 0.28106 | −0.000188 | 22,976 |
| Wales | 0.28165 | 0.28143 | −0.000224 | 17,279 |
| East Midlands | 0.28249 | 0.28228 | −0.000218 | 19,376 |
| North East | 0.27927 | 0.27904 | −0.000232 | 10,579 |
| Northern Ireland | 0.28203 | 0.28167 | **−0.000363 (largest)** | 6,131 |

**This table's direction has reversed relative to an earlier run of the same simulation** — an earlier run found every region shifting positively (rising risk) under the shared forecast shock; the current run finds every region except London shifting negatively (falling risk), and even London's shift is close to zero. This flip is a direct downstream consequence of the FES-attachment correction discussed under `fes_moderation_path.csv` above, which changed the realised/forecast FES values this simulation swaps between. The magnitudes remain tiny in absolute terms in both runs (well under 0.05 percentage points), so the more robust reading of this table across both runs is "the counterfactual shift is consistently small everywhere," not a specific claim about its direction — Northern Ireland has the largest-magnitude shift of any region in the current run, consistent with its exceptional current vulnerability level.

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

**Structure:** single row, columns `validation_pairs, n_train, n_validation, auc_vs_high_fuel_vulnerable_t1, pearson_r_vs_fuel_to_income_ratio_t1`. Latest run: `validation_pairs="m->n, n->o"`, `n_train=177,408`, `n_validation=21,161`, **`auc=0.7615`**, **`pearson_r=0.2719`**. An AUC of ~0.76 on genuinely held-out, walk-forward (not fit) transitions is a solid discriminative result for a logistic model predicting a binary next-wave outcome from cross-sectional predictors — comfortably above chance (0.5) and in the acceptable-to-good range by conventional social-science standards. **This value fluctuates slightly across successive reruns of the same specification** (observed range across recent runs: 0.756–0.762) — the logistic model itself has no fixed random seed, and it is fit on upstream SEM/CVAE factor scores that themselves vary slightly run to run (Section 3.6's own documented CVAE stochasticity); read the AUC as "approximately 0.76," not as a number stable to the 4th decimal place. The much weaker Pearson correlation (~0.27) against the continuous `fuel_to_income_ratio_t1` indicates the model is considerably better at rank-ordering/classifying who crosses the high-vulnerability threshold than at explaining variance in the continuous ratio itself.

#### `outputs/ukhls_forward_prediction/tables/stage5_driver_coefficients.csv`

**Structure:** **11** rows (up from 8 — extended this iteration with `lone_parent`, `large_family`, `workless_household`), columns `predictor, coef, odds_ratio, p_value, significant` (final model, refit on all 14 known transitions).

| Predictor | Coefficient | Odds ratio | p-value | Significant |
|---|---|---|---|---|
| financial_strain_score | 1.6052 | **4.98** | <0.001 | Yes |
| lone_parent | 0.4256 | 1.53 | <0.001 | Yes |
| workless_household | 0.3570 | 1.43 | <0.001 | Yes |
| object_score | 0.3501 | 1.42 | <0.001 | Yes |
| **fes_magnitude** | 0.0662 | **1.07** | <0.001 | Yes |
| dvage | 0.0124 | 1.01 | <0.001 | Yes |
| large_family | −0.0359 | 0.96 | 0.456 | No |
| personal_score | −0.0746 | 0.93 | <0.001 | Yes |
| heatch | −0.0867 | 0.92 | 0.013 | Yes |
| condition_score | −0.1205 | 0.89 | <0.001 | Yes |
| energy_score | **−1.1684** | **0.31** | <0.001 | Yes |

**`fes_magnitude`'s coefficient has been corrected this iteration** — an earlier run of this table reported OR≈52 (a >50-fold odds increase per unit), which was a symptom of the same FES-attachment defect corrected under `fes_moderation_path.csv` above, not a genuine effect. The current, corrected coefficient (OR ~1.03–1.07, drifting slightly between reruns of this specification for the same reason the validation AUC above does) is small but significant — directionally consistent with the earlier finding, just far more plausible in magnitude. **A related, genuinely positive change**: `fes_magnitude_used` in `stage5_forward_predictions.csv` below is no longer a single constant applied to every household — it now takes **24 distinct values** (varying by each household's own interview month/target-year cohort), meaning the forecast signal genuinely differentiates between households now, not only in this model's structure but in its actual applied predictions too. Three of the four COR-SEM factors (`personal_score` OR 0.93, `condition_score` OR 0.89, `energy_score` OR 0.31) behave as "more resource → lower future risk" predicts, with `energy_score` dominant; `object_score` (OR 1.42) is the one consistent exception, raising rather than lowering predicted future risk — the same pattern found in Stage 3's contemporaneous driver analysis (`driver_analysis_logistic_regression.csv` above), now confirmed prospectively too. Among the new covariates, `lone_parent` (OR 1.53) and `workless_household` (OR 1.43) are both significant risk-raisers; `large_family` is not significant in this specification (p=0.46, OR 0.96), unlike its significant, risk-raising effect in Stage 3's contemporaneous model (OR 1.68) — a genuine contemporaneous-vs-prospective divergence for this one covariate specifically, not an inconsistency to reconcile. `heatch` is a small but now-significant protective factor (p=0.013), a modest change from a non-significant reading in an earlier run.

#### `outputs/ukhls_forward_prediction/tables/stage5_forward_prediction_by_month.csv`

**Structure:** 12 rows, columns `interview_month, mean_predicted_probability, n`, total n=19,140.

| Month | Mean predicted probability | n |
|---|---|---|
| Jan | 6.76% | 1,603 |
| Feb | 7.00% | 1,652 |
| Mar | 6.62% | 1,644 |
| **Apr** | **7.48% (highest)** | 1,616 |
| May | 6.96% | 1,589 |
| Jun | 6.90% | 1,584 |
| Jul | 6.90% | 1,521 |
| Aug | 6.81% | 1,598 |
| Sep | 6.67% | 1,577 |
| **Oct** | **7.44%** | 1,601 |
| Nov | **7.45%** | 1,592 |
| **Dec** | **7.01% (2nd/3rd close behind Nov/Oct/Apr)** | 1,563 |

April, October, and November now cluster together as the highest-risk months (7.44–7.48%), with March the trough (6.62%) — a spread of 0.86 percentage points, a similar shape to an earlier run (which found April highest, September lowest) but with a slightly different specific ranking, consistent with `fes_magnitude_used` now genuinely varying by household (below) rather than being a single flat number.

#### `outputs/ukhls_forward_prediction/tables/stage5_forward_prediction_map.csv`

**Structure:** 12 rows, columns `region, mean_predicted_probability, n, mean_predicted_probability_pct`.

| Region | Mean predicted probability | n |
|---|---|---|
| **Northern Ireland** | **7.63% (highest)** | 1,083 |
| Wales | 7.57% | 1,185 |
| North West | 7.51% | 1,976 |
| Yorkshire and the Humber | 7.45% | 1,638 |
| West Midlands | 7.22% | 1,603 |
| North East | 7.15% | 706 |
| East of England | 7.04% | 1,728 |
| Scotland | 6.84% | 1,858 |
| East Midlands | 6.92% | 1,429 |
| London | 6.68% | 1,829 |
| South East | 6.36% | 2,394 |
| **South West** | **6.36% (lowest, tied with South East)** | 1,705 |

This forward-looking ranking closely mirrors the historical prevalence ranking documented throughout Stage 3/4 (Northern Ireland highest, South East/South West lowest at both), stable relative to an earlier run — the model projects forward the same existing geography of disadvantage rather than predicting any reshuffling. The spread (6.36%–7.63%, 1.27 percentage points) is much narrower than the historical prevalence spread (5.96%–15.62%, ~9.66 points), reflecting natural model shrinkage toward the forecast-shock term's own, now genuinely household-varying but still comparatively narrow, range.

#### `outputs/ukhls_forward_prediction/tables/stage5_forward_predictions.csv`

**Structure:** 19,140 household rows, columns `hrpid, wave, gor_dv, region, interview_year, interview_month, predicted_target_year, fes_magnitude_used, predicted_vulnerable_probability`. Latest run: mean predicted probability **7.00%**, **median only 4.00%** (a materially right-skewed distribution), min 0.15%, max 90.4%, standard deviation 9.07 percentage points. 220 households (1.15%) are predicted above 50%; 1,152 (6.0%) above 20%; 3,710 (19.4%) above 10%. `predicted_target_year` splits into two cohorts: 10,937 households interviewed in 2023 predicted for 2024, and 8,203 households interviewed in 2024 predicted for 2025 (unchanged from an earlier run). **`fes_magnitude_used` is a genuine improvement this iteration: it now takes 24 distinct values** (ranging −2.28 to +4.80) rather than the single constant (−1.06) an earlier run applied identically to every household — the forecast shock now varies by each household's own interview-month/target-year cohort, meaning it can genuinely contribute to differentiating between households' predictions, not only to the model's fitted coefficient. The large gap between the mean (7.00%) and median (4.00%), together with the maximum (90.4%), shows a small number of households carry very concentrated predicted risk — an obvious candidate group for direct, targeted outreach distinct from a purely geographic targeting approach.

### `outputs/ukhls_forward_prediction/figures/`

- **`outputs/ukhls_forward_prediction/figures/stage5_validation_roc.png`** — an ROC curve (true positive rate vs. false positive rate) for the Stage 5 logistic model evaluated on the held-out wave transitions, titled "Stage 5 Walk-Forward Validation (held-out wave transitions — genuinely forward-predicted, not fit)." A red curve for the model against a dashed grey chance diagonal; legend confirms "Stage 5 model (AUC=0.760)", matching `stage5_validation_metrics.csv` exactly. The curve rises steeply at low false-positive rates (true-positive rate ≈0.6 by a false-positive rate of ≈0.2) before flattening — a typical, well-behaved discrimination shape, comfortably bowed above the diagonal across its entire range with no crossing or inversion.
- **`outputs/ukhls_forward_prediction/figures/stage5_forward_prediction_by_month.png`** — a bar chart, "Which Month Carries the Highest Predicted Risk," x-axis = target month, y-axis = mean predicted probability (%), bars coloured red/green by whether they sit above or below roughly the 6.7–6.8% mark (red: Jan, Feb, Apr, May, Oct, Nov, Dec; green: Mar, Jun, Jul, Aug, Sep). Data labels on each bar match the underlying CSV exactly, confirming April as the tallest bar and September as the shortest.
- **`outputs/ukhls_forward_prediction/figures/stage5_forward_prediction_map.png`** — a choropleth of the UK's 12 regions, "Predicted Forward Vulnerability by UK Region," redesigned this session to display percentages (e.g. "7.7%") rather than a two-decimal probability fraction — resolving the earlier problem where most regions rounded to an indistinguishable "0.07." Colour scale spans roughly 6.2%–7.6%; labels match the region table exactly, with Northern Ireland darkest/highest and South East/South West both lightest/lowest (tied visually at the low end of the colour scale).

---

## Summary of Data-Quality and Methodological Flags Found Across All Files

**Changelog, this iteration — several fixes that changed numbers throughout this catalog, Stage 1 especially:**

- **Stage 1 model selection now defaults to `'validation'` (genuine walk-forward, pre-target-year information only), not `'forecast_actual'` (hindsight).** `forecast_pipeline.run`'s own default flipped, and `run_rolling` (the reported path) never supports anything else. `outputs/tables/model_metrics_comparison.csv` on disk still shows `selection_basis=forecast_actual` because that single-year diagnostic file was deliberately regenerated with the old basis passed explicitly, to keep the backtest-vs-realised disagreement inspectable side by side — it is a one-off illustration, not this project's current default behaviour.
- **`_select_best_mode_per_series` no longer silently ignores the requested `selection_basis`.** An earlier version always compared candidate modes on `forecast_actual_MAE` whenever it was present, regardless of what basis the caller asked for — meaning `FES_selected` was hindsight-biased even under nominally honest `'validation'` selection, and (compounded by a separate bug) had in practice never once picked macro-mode for any series, making `FES_selected` bit-for-bit identical to `FES_core`. Both are now fixed: `FES_selected` genuinely differs from `FES_core` (electricity's contribution now comes from macro mode) and only uses `forecast_actual_MAE` under the explicit, non-default `'forecast_actual'` basis.
- **A new `FES_weighted` variant** (inverse-validation-RMSE per series, normalized to mean 1.0) was added alongside the three existing equal-weighted variants, never replacing them — reported side-by-side in `fes_comparison_metrics.csv`/`fes_variant_selection.csv` so its value is an inspectable, falsifiable comparison rather than an assumed improvement.
- **A crash in `attach_fes_delta`** (`KeyError: 'fes_selected'`, a column-rename step that attached `fes_selected` but not its sibling `fes_weighted`) was fixed. This is the most likely cause of several corrections seen throughout this catalog's Stage 2–5 sections relative to an earlier run: `fes_delta`'s significance flipping from non-significant to significant in `driver_analysis_logistic_regression.csv`, `stage5_driver_coefficients.csv`'s `fes_magnitude` coefficient shrinking from an implausible OR≈52 to a plausible OR≈1.03, and `fes_magnitude_used` in `stage5_forward_predictions.csv` now taking 24 distinct values instead of one constant.
- **One household-wave row with zero observed CFA indicators** was degenerating into a 0×0 covariance submatrix inside `semopy`'s FIML fit, producing hundreds of spurious LAPACK stderr lines per run (`"On entry to DPOTRI parameter number 4 had an illegal value"`) with no effect on any actual estimate — now dropped before fitting. A log-noise fix only, included here for completeness.

1. **Model selection in Stage 1's single-year *diagnostic* file is hindsight-based by deliberate, explicit configuration, not by default** (`outputs/tables/model_metrics_comparison.csv`, run with `--selection-basis forecast_actual` specifically to expose the disagreement): its "best model" choice, on that basis, disagrees with the genuine walk-forward backtest ranking in most of the 6 series/mode cells — most dramatically for carbon-core, where the backtest's top pick (TFT) is realised-accuracy's *worst* performer. The reported rolling walk-forward path (`outputs/fes/model_selection_by_year.csv`) never makes a hindsight choice; the same backtest-vs-realised divergence still shows up there too, as a property of the forecasting problem rather than of the (now honest) selection mechanism.
2. **`FES_selected`/`FES_weighted` now genuinely outperform `FES_core`/`FES_macro`** on the single-year diagnostic's own realised-stress validation (`outputs/fes/fes_comparison_metrics.csv`: Pearson r up to 0.84 vs. 0.10–0.63, MAE cut by roughly 3–4×) — a reversal from an earlier run where the mode-selection bug above made `FES_selected` identical to `FES_core` and left `FES_macro` as the best of only three candidates. On the reported rolling path, however, `FES_macro` is the variant that actually wins the 16-year mean-RMSE comparison (`outputs/fes/fes_variant_selection.csv`, 2.749 vs. Weighted's 2.790) — the single-year and 16-year rankings genuinely disagree on which single variant is "best," a finding reported rather than reconciled, since N=12 (one year) and N=16×12 (the rolling average) are different, both legitimate, evidence bases.
3. **`FES_core`'s own uncertainty term flipped sign in the single-year diagnostic run** (`outputs/fes/fes_components_table.csv`: −0.492 → +1.513), flipping `FES_core`'s and `FES_macro`'s annual-mean totals from correctly-negative (roughly matching the realised actual's stressed direction) to incorrectly-positive — while `FES_selected`/`FES_weighted` stayed on the correct side. Worth watching on any future rerun, since hyperparameter tuning now runs by default and can shift the underlying models' prediction-interval widths (which drive the uncertainty z-score) run to run.
4. **Two SEM fit-index tables report values outside their valid mathematical range** (`outputs/ukhls_cor_sem/tables/cor_sem_fit_indices.csv`: CFI/TLI outside [0,1]; `outputs/ukhls_cor_sem/tables/structural_fit_indices_baseline.csv`: an implausible SRMR, and CFI/TLI >1 marked "ok" by the pipeline's own validity flags, which appear to test only a lower bound) — and CFI/TLI themselves vary run-to-run, not just relative to their valid range.
5. **The second-order SEM structural model's sign is not identified**, discovered this iteration by deliberate replication (`outputs/ukhls_cor_sem/tables/structural_paths_baseline.csv`: three of four factors loading |≈1.0| on BASELINE, with a sign that flips between refits) — this is stronger than the earlier "Heywood-case-like pattern" framing, since it directly explains why the interaction term below is unresolved rather than merely imprecise.
6. **The FES-moderation interaction term is now significant in every recent refit, but its sign is not stable across those refits** (`outputs/ukhls_cor_sem/tables/fes_moderation_path.csv`: +0.00049 then −0.00073, both p<0.0001) — a correction from an earlier run's stable non-significant finding (p=0.608). Only the two main effects (`fes_delta`, always negative and significant; `baseline_score`, always significant but sign-unstable) are currently trustworthy from this table.
7. **CVAE-SEM latent alignment has improved for three of four dimensions but worsened for the fourth** (`outputs/ukhls_cor_cvae/tables/cvae_sem_alignment.csv`): z1/object, z2/condition, z3/personal are now all cleanly separated from their next-highest correlate; z4/energy has moved from partially entangled to actively mis-aligned — its largest correlation is now with `object_score`, not its own intended `energy_score`.
8. **`object_score` is a reproducible exception to "more resource → lower vulnerability"** across both the contemporaneous driver model (`driver_analysis_logistic_regression.csv`) and the prospective forward-prediction model (`stage5_driver_coefficients.csv`) — it consistently raises, rather than lowers, predicted vulnerability, plausibly reflecting the heating cost of larger/older owned housing.
9. **Two duplicate table exports**: `outputs/ukhls_vulnerability/tables/policy_map_baseline_resource_by_region.csv` and `outputs/ukhls_vulnerability/tables/policy_map_financial_strain_by_region.csv` were byte-identical in an earlier run; not re-verified this iteration.
10. **A likely data-sparsity artifact**: Northern Ireland's 2024 cell in `outputs/ukhls_vulnerability/tables/policy_vulnerability_region_year_heatmap.csv` reads exactly 0.0% against a neighbouring ~11–12% trend (as of an earlier run; not re-verified this iteration).
11. **Two external-validation charts display "ρ=nan"/"r=nan"** (`outputs/ukhls_vulnerability/figures/external_validation_disability_bars.png` and `...disability_scatter.png`) — mathematically correct for n=2 disability categories, but reads as an error without an explanatory caption. The same will apply to the two new n=2 dimensions (family type, work status) added this iteration.
12. **A minor schema inconsistency** in `outputs/forecasts/`: the `series` column is populated in the 24 raw per-model CSVs but blank in the 9 aggregate (`_core`/`_macro`/`_all`) CSVs.
13. **A small region-mapping gap**: `outputs/ukhls_dataset_overview/tables/dataset_rows_by_region.csv`/`dataset_sample_size_by_region.csv` sum to 339,020, 181 rows (0.05%) short of the full 339,201-row panel (as of an earlier run; not re-verified this iteration).
