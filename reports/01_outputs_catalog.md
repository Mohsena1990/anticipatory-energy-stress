# Outputs Catalog: File-by-File Analysis

**Project:** Anticipatory Fuel Stress Watch (AFSW) — Forecasting Anticipatory Energy–Carbon Stress and Household Fuel Vulnerability in the UK
**Scope:** every table (CSV) and figure (PNG/HTML) under `outputs/`, organized by pipeline stage, in the order the pipeline produces them. Each entry states the exact filename first, then a precise analysis of its contents.

---

## Stage 1 — Macro Forecasting and the FES Index

### `outputs/tables/model_metrics_comparison.csv`

24 rows (3 series × 4 models × 2 modes). Carries **two independent accuracy regimes** that frequently disagree: a backtest/walk-forward composite (`MAE/RMSE/MAPE/SMAPE/MASE/...` + `rank_score`, lower = better) and the realised-2025 accuracy (`forecast_actual_MAE/RMSE/SMAPE` + `selection_score`), the latter of which is what actually drives model selection (`selection_basis=forecast_actual` on every row).

Winner by realised `forecast_actual_RMSE` per series/mode:

| Series | Mode | Winner | RMSE | Runner-up | Worst |
|---|---|---|---|---|---|
| gas | core | LSTM | 7.60 | Prophet 12.54 | SARIMA 92.01 |
| gas | macro | LSTM | 11.86 | Prophet 31.62 | TFT 41.98 |
| electricity | core | LSTM | 4.14 | TFT 5.43 | SARIMA 46.84 |
| electricity | macro | LSTM | 4.54 | TFT 5.32 | SARIMA 19.76 |
| carbon | core | LSTM | 15.00 | SARIMA 24.45 | TFT 49.75 |
| carbon | macro | **TFT** | 13.79 | LSTM 20.62 | Prophet 101.98 |

This matches `fes_components_table.csv`'s "best model" row exactly.

**Critical finding — backtest and reality disagree sharply in 4 of 6 cells.** Electricity-core: backtest ranks SARIMA best (rank_score 1.0), but against realised 2025 data SARIMA is the worst model by a factor of ~11 (RMSE 46.84 vs LSTM's 4.14). Gas-core and gas-macro show the same pattern (SARIMA backtest-favoured, real-world worst). Carbon-macro is the one case where the backtest's narrow preference for LSTM (1.45 vs TFT's 2.15) flips to TFT once real outcomes are known. Only electricity-macro has backtest and reality agree. **Because model selection uses `forecast_actual_RMSE` — computed after the 2025 actuals were already known — the FES pipeline's "best model" choice is a hindsight pick, not what a genuinely blind 2025 forecast would have selected.** This is the single most important methodological caveat in Stage 1.

---

### `outputs/forecasts/` (33 files)

Per series (gas, electricity, carbon): 4 models × 2 modes = 8 per-model CSVs, plus 3 aggregates (`_core.csv`, `_macro.csv`, `_all.csv`). All share columns `date, model, mode, forecast, lower_bound, upper_bound, actual` (a `series` column exists only in the raw LSTM/TFT files, blank in the aggregates — a minor schema inconsistency).

**Gas** — actual monthly growth 2025: −12.4, −12.4, −12.3, +12.5, +12.6, +12.6, +13.3, +13.3, +13.3, +2.2, +2.2, +2.1% (a sharp winter-negative → spring/summer-positive → autumn-flat regime shift).

- `gas_growth_pct_forecasts_sarima_core.csv` — mean forecast 80.84%, sign-matches actual all 12 months but massively over-forecasts magnitude (up to 144% by October vs 2.2% actual); average PI width 90.4, only 25% coverage.
- `gas_growth_pct_forecasts_sarima_macro.csv` — tamer than core (mean 33.5%) but still overshoots; 75% coverage.
- `gas_growth_pct_forecasts_prophet_core.csv` — essentially flat (~10–11% every month, ignoring the actual regime shift); widest, static PI band; 100% coverage (uninformatively wide).
- `gas_growth_pct_forecasts_prophet_macro.csv` — noisier than core (63% in March, −22% in June, 39% in December) rather than flat; 75% coverage.
- `gas_growth_pct_forecasts_lstm_core.csv` — **the winner.** Smooth trend from −7.8% (Jan) to +7.2% (Dec), correct sign all 12 months, tightest PI (11.8 width); 42% coverage.
- `gas_growth_pct_forecasts_lstm_macro.csv` — low RMSE (11.86) but only 3/12 correct signs — it stays persistently negative (−1.2 to −2.7%) while gas is mostly positive; **its low RMSE reflects low variance, not genuine skill**; 0% PI coverage (actual never falls inside its own interval — overconfident).
- `gas_growth_pct_forecasts_tft_core.csv` — drifts upward late in the year (~70% by December); 11/12 correct signs.
- `gas_growth_pct_forecasts_tft_macro.csv` — moderate performance, 12/12 correct signs, RMSE 41.98 (worst core-adjacent model here).
- `gas_growth_pct_forecasts_core.csv`, `gas_growth_pct_forecasts_macro.csv`, `gas_growth_pct_forecasts_all.csv` — 4-model concatenations (aggregate views used by downstream plotting).

**Electricity** — actual monthly growth 2025: −8.78, −8.78, −8.83, +4.63, +4.57, +4.57, +7.99, +7.99, +8.04, +2.70, +2.81, +2.75%.

- `electricity_growth_pct_forecasts_sarima_core.csv` — the single largest backtest-vs-reality contradiction in the dataset: best backtest rank_score (1.0) yet worst realised RMSE (46.84) of all 24 model/series/mode cells. Flat around −9 to −11% for Jan–Mar, then an abrupt jump to 51–55% from April.
- `electricity_growth_pct_forecasts_sarima_macro.csv` — same pathology, tamer (mean 16.6%); 75% coverage.
- `electricity_growth_pct_forecasts_prophet_core.csv` / `_macro.csv` — flat ~8%, 9/12 correct signs, very wide PI (100%/92% coverage).
- `electricity_growth_pct_forecasts_lstm_core.csv` — **the winner** (RMSE 4.14), tightest PI (5.3 width), 12/12 correct signs.
- `electricity_growth_pct_forecasts_lstm_macro.csv` — close second (RMSE 4.54), also 12/12 correct signs.
- `electricity_growth_pct_forecasts_tft_core.csv` — competitive (RMSE 5.43), tight PI, 11/12 correct signs.
- `electricity_growth_pct_forecasts_tft_macro.csv` — RMSE 5.32, 10/12 correct signs.
- `electricity_growth_pct_forecasts_core.csv`, `_macro.csv`, `_all.csv` — aggregates.

**Carbon** — actual monthly growth 2025: +25.1, +21.7, +9.3, −3.9, −5.1, +2.4, +4.8, +3.8, +14.1, +19.2, +20.0, +18.1% (the most volatile series, with a mid-year dip into negative territory).

- `carbon_growth_pct_forecasts_sarima_core.csv` / `_macro.csv` — flat and negative (~−11%) all year; only 2/12 correct signs, missing the entire recovery; PI so wide (167–169) it achieves 100% coverage uninformatively.
- `carbon_growth_pct_forecasts_prophet_core.csv` / `_macro.csv` — **a striking artifact**: `lower_bound=−99.78` and `upper_bound=137.96` are literally identical across all 12 rows in both modes — Prophet has collapsed to a frozen constant band rather than a time-varying forecast; 10/12 signs correct but RMSE is the worst or near-worst in the series (76.8 core / 67.1 macro).
- `carbon_growth_pct_forecasts_lstm_core.csv` — mean forecast ≈0.0%, 9/12 correct signs, RMSE 15.00 — best core model.
- `carbon_growth_pct_forecasts_lstm_macro.csv` — only 2/12 correct signs (mean −6.4% vs. a series that's mostly positive), RMSE 20.62.
- `carbon_growth_pct_forecasts_tft_core.csv` — monotonically rising forecast (7.3% Jan → 91.5% Dec) chasing the recovery but badly overshooting by year-end; **worst core RMSE for carbon (49.75)** despite 10/12 correct signs.
- `carbon_growth_pct_forecasts_tft_macro.csv` — **the outlier win**: RMSE 13.79, beating every other model/mode including LSTM-macro (20.62) — the only case in the whole grid where a macro model beats its own core counterpart and every other model.
- `carbon_growth_pct_forecasts_core.csv`, `_macro.csv`, `_all.csv` — aggregates.

---

### `outputs/fes/` (5 files)

**`outputs/fes/fes_components_table.csv`** — component breakdown (z-scores) for FES_core, FES_macro, FES_selected, FES_actual:

| Component | Core | Macro | Selected | Actual |
|---|---|---|---|---|
| gas_growth_pct | −0.208 | −0.379 | −0.208 | −0.194 |
| electricity_growth_pct | −0.323 | −0.443 | −0.323 | −0.411 |
| carbon_growth_pct | −0.038 | +0.060 | −0.038 | +0.032 |
| Uncertainty/RealVol | −0.492 | −0.495 | −0.492 | −0.577 |
| **FES_total (annual mean)** | **−1.061** | **−1.256** | **−1.061** | **−1.150** |

**Key finding: `FES_selected` is bit-for-bit identical to `FES_core`**, confirmed across all 12 months in `fes_monthly_2025.csv`'s `selected_mode_gas/electricity/carbon` columns (all read `"core"`). The per-series best-of-core/macro selection never once picked macro — even though the model comparison table shows carbon's true best model is macro/TFT. Carbon is also the only component where core and macro disagree on sign (core −0.038 vs macro +0.060), and the realised actual (+0.032) sits on the macro side — i.e. the core variant gets the *direction* of carbon's 2025 effect wrong; macro gets it right, but `FES_selected` still inherits core's (wrong-signed) carbon contribution.

**`outputs/fes/fes_summary_2025.csv`** — same numbers restructured long-format, plus `actual_prior_year, FES_TOTAL, −2.582` (2024). Realised 2025 stress (−1.150) is markedly less severe than realised 2024 (−2.582) — roughly a 55% reduction in magnitude, indicating a substantial year-on-year easing of the energy-cost/carbon-price shock.

**`outputs/fes/fes_prior_actual_2024.csv`** — 12 monthly rows for 2024. `fes_actual` ranges from −3.080 (June) to −1.787 (November); every 2024 month is more negative than every 2025 month, consistent with the year-on-year easing above. Gas/electricity actuals are deeply negative through much of 2024 (gas −38.3%, electricity −21.1% in Apr–Jun) — a sharp deflationary base-effect period as 2022–23's price spike washes out of the year-on-year comparison.

**`outputs/fes/fes_monthly_2025.csv`** — 12 rows, 53 columns (the widest file in the project): parallel `_core`/`_macro`/`_selected`/`_actual` blocks plus three additional realised-volatility benchmark blocks (`rv_actual_A/B/C`). Monthly trajectories: `fes_core` improves almost monotonically from −1.933 (Jan) to −0.683 (Dec); `fes_macro` is flatter, from −1.600 to −1.376; `fes_actual` shows a materially different shape — a dip around March (~−1.79), a jump to ~−0.97 by April, continued improvement to a peak of ~−0.71 by September, then **a relapse back to −1.15/−1.16 in October–December that neither FES_core nor FES_macro anticipates** (both keep trending toward less-negative values through December). This end-of-year miss is the single biggest divergence between forecast and actual in the whole FES series.

**`outputs/fes/fes_comparison_metrics.csv`** — 9 rows (3 forecast variants × 3 realised-benchmark definitions), N=12 each:

| Variant | Benchmark | MAE | RMSE | Pearson r | R² | Theil U |
|---|---|---|---|---|---|---|
| Equal_Core | RollingVol | 0.577 | 0.700 | 0.727 | 0.247 | 1.019 |
| Equal_Core | CrossComp | 0.222 | 0.284 | 0.742 | 0.236 | 1.076 |
| Equal_Core | AbsShock | 0.224 | 0.296 | 0.740 | **−0.850** | **1.718** |
| Equal_Macro | RollingVol | 0.521 | 0.711 | **0.813** | 0.223 | 1.035 |
| Equal_Macro | CrossComp | 0.215 | 0.245 | **0.801** | 0.434 | 0.926 |
| Equal_Macro | AbsShock | **0.135** | **0.170** | 0.778 | 0.387 | 0.989 |
| Equal_Selected | (all three) | ≈ identical to Equal_Core (they are the same series) | | | | |

**Tension worth flagging: FES_Macro beats FES_Core (and hence Selected) on nearly every aggregate-index robustness metric** — higher Pearson r against all three benchmarks (0.813 vs 0.727, 0.801 vs 0.742, 0.778 vs 0.740) and lower error on two of three — even though the per-series model comparison table favoured core-mode models for 5 of 6 series/mode cells. **FES_Core/Selected also has a negative R² (−0.850) and Theil's U of 1.718 against the AbsShock benchmark** — performing worse than predicting the mean, and worse than a naive random-walk forecast, on that specific definition of realised stress.

---

### `outputs/figures/` (21 PNG + matching PDFs + 6 interactive HTML)

- **`fes_components_2025.png`** — two-panel bar chart (component z-scores left, annual totals right). Confirms the `fes_components_table.csv` numbers exactly. Value labels for the most negative bars were recently fixed this session (previously clipped/overlapping the x-axis tick labels — see `src/fes_calculator.py`'s ylim-padding fix).
- **`fes_monthly_2025.png`** — line chart of the three FES variants over 2025, visually confirming the March dip / April jump / September peak / Oct–Dec relapse in the actual series described above, and that neither Core nor Macro anticipates the relapse.
- **`fes_metrics_pearson_r_heatmap.png`**, **`fes_metrics_rmse_heatmap.png`** — 3×3 heatmaps (variant × benchmark) reproducing `fes_comparison_metrics.csv` exactly; Equal_Macro/RollingVol (r=0.813) is the greenest/best cell, Equal_Core-and-Selected/RollingVol (r=0.727) the reddest/worst in the Pearson map; RollingVol is the hardest benchmark to hit regardless of variant choice.
- **`forecast_comparison_{gas,electricity,carbon}_{core,macro}.png`** (6 files) — visually confirm SARIMA's runaway prediction intervals (dominating the plot's y-axis scale in gas/electricity-core), Prophet's frozen carbon PI band, and TFT's late-year overshoot on carbon-core.
- **`forecast_vs_actual_{gas,electricity,carbon}.png`** and **`forecast_vs_actual_all_series.png`** — full 2005–2024 historical context. Carbon's historical range (−680% to +735%, 2007–09 EU-ETS-era volatility) dwarfs the 2025 forecast/actual cluster (roughly −15% to +25%), explaining why carbon PIs are proportionally so wide even though 2025 itself was comparatively tame.
- **`prediction_intervals_2025.png`** — 6-panel composite (rows=core/macro, columns=gas/electricity/carbon) of the same data as the six `forecast_comparison_*` figures, for at-a-glance comparison. No rendering issues observed.
- **`model_ranking_polar_{gas,electricity,carbon}_{core,macro}.png`** (6 files) — circular bar charts, one per series/mode, values cross-validated exactly against `model_metrics_comparison.csv`'s `forecast_actual_*` columns; the clearest visual confirmation that SARIMA/TFT/Prophet occasionally balloon to 2–10× the winning model's error.
- **`interactive_{gas,electricity,carbon}_{core,macro}.html`** (6 files) — Plotly interactive versions merging the historical-context and forecast-comparison views with toggleable per-model traces; add exploratory value beyond the static PNGs but surface no numbers not already in the CSVs.

---

## Stage 2a-overview — Dataset Overview

### `outputs/ukhls_dataset_overview/tables/`

**`dataset_rows_by_wave.csv`** — 15 rows (waves a–o). Row counts: a=30,169 (2009) → declining broadly to a trough at m=16,156 (2021) → rising again at n=21,385 (2022) and o=19,586 (2023), a refreshment-sample pattern. Sum = 339,201, matching the full panel exactly.

**`dataset_rows_by_year.csv`** — 16 rows, 2009–2024, summing to the same 339,201. Notably spans one more calendar year than the wave table (2024 has 8,439 rows) because each wave's fieldwork actually straddles two calendar years — confirmed by the interview-month histogram below. This is a reconciliation note, not a data error.

**`dataset_rows_by_region.csv`** / **`dataset_sample_size_by_region.csv`** (identical content, different sort order) — 12 regions; London highest (41,318), North East lowest (12,643), a 3.2× spread. Sum = 339,020, **181 rows short of the full panel** — a small number of rows have an unmapped region code, silently dropped from this aggregation (worth a data-quality note, though immaterial at 0.05% of rows).

**`dataset_missingness_by_variable.csv`** — 18 variables. Highest missingness: `inoutflows2/3/4` (all three at an identical 96.857%, implying jointly-missing rows/waves — these variables are asked only in a small subset of waves) — the project's own docstrings already flag these as excluded from most models. `sf1_good` 56.7%, `hsval` 37.3%, `carval` 34.8%. Lowest-missingness cluster (all <1%): `jbstat_security`, `health_good`, `fiyrinvinc_dv`, `tenure_security`, `bill_security`, `heatch_good`, `financial_strain_score`, `fihhmnnet1_dv` (0.004%, essentially complete).

### `outputs/ukhls_dataset_overview/figures/`

- **`dataset_panel_composition.png`** — sample size by wave (left) confirms the table above; interview-month histogram (right) is nearly flat (~26,700–31,000/month, peak February, trough August) — this near-uniformity is exactly why month-resolution FES matching is viable, and explains the wave/calendar-year straddling noted above.
- **`dataset_missingness.png`** — the same missingness data with human-readable labels (added this session): "Used savings (past year)" 96.9%, "New borrowing, family/friends" 96.9%, "New borrowing, bank/credit card" 96.9%, "General health satisfaction (good)" 56.7%, "House value" 37.3%, "Car value" 34.8%, down to "Household net monthly income" 0.0%.
- **`dataset_key_distributions.png`** — 6-panel histogram grid. Fuel-to-income ratio: median 0.04, heavily right-skewed. Financial strain: median 0.25. Age: bimodal (30s and 60s–70s peaks), median 48.5. Income (log10): roughly normal, median ≈£2,512/month. Fuel spend (log10): roughly normal, median ≈£1,202/year. **FES Delta: a visibly clumped, multi-modal distribution** (median −0.68, distinct clusters near 0/−1, −9, −7, and +2 to +3) rather than smooth/continuous — worth flagging as a modeling consideration, since the FES-moderation regression treats it as a continuous predictor.
- **`dataset_rows_by_year.png`** — confirms the by-year table and visually shows which waves contribute to which calendar years (the straddling pattern).
- **`dataset_sample_size_by_region.png`** — UK choropleth, London/South East darkest (highest sample), North East lightest (lowest).

---

## Stage 2b — COR-SEM

### `outputs/ukhls_cor_sem/tables/`

**`cor_sem_measurement_loadings.csv`** — 13 items across 4 factors, all non-marker items significant at p≈0:

| Factor | Item | Std. loading |
|---|---|---|
| OBJECT | hsrooms (marker) / hsbeds / ncars / carval / hsval | 0.576 / 0.671 / 0.668 / 0.427 / 0.552 |
| CONDITION | tenure_security (marker) / jbstat_security / bill_security | 0.372 / 0.450 / **0.226** |
| PERSONAL | health_good (marker) / sf1_good / qfhigh_band | 0.592 / 0.849 / 0.377 |
| ENERGY | fihhmnnet1_dv (marker) / fiyrinvinc_dv | 0.583 / 0.367 |

`bill_security` (0.226) is the weakest loading in the entire model, below the conventional 0.3–0.4 acceptability threshold — visually confirmed against a "min=0.4" reference line in both loading figures. CONDITION is the weakest-measured factor overall (all three of its items sit at or below that line).

**`cor_sem_fit_indices.csv`** (measurement model) — single row: `dof=59, chi2=9.943, chi2_p=1.0, CFI=−0.538, TLI=−1.033, RMSEA=0.0, GFI=0.784, SRMR=0.079`. **CFI and TLI are both far outside their valid [0,1] range** — flagged `False` by the pipeline's own `_ok` columns. This combination (near-zero chi² alongside deeply negative incremental fit indices) is consistent with a known `semopy` 2.3.11 limitation under FIML with unstandardized, differently-scaled indicators (raw £ values like `hsval`/`carval`/`fihhmnnet1_dv` mixed with 0/1-coded security items) — already documented in the project's own code comments as a fit-statistic-computation limitation, not necessarily evidence of measurement misspecification. RMSEA (0.0, a suspiciously exact floor value) and SRMR (0.079) are nominally acceptable but their coexistence with nonsensical CFI/TLI undermines confidence in the reported fit as a whole.

**`structural_fit_indices_baseline.csv`** (second-order model) — single row: `dof=61, chi2=79.955, chi2_p=0.052, CFI=1.594, TLI=1.760, RMSEA=0.001, GFI=−0.735, SRMR=9,214,088,570.31`. **SRMR of ~9.2 billion is an obviously invalid value** (correctly flagged `srmr_ok=False`); GFI (−0.735) is also out of its valid range but has no corresponding `_ok` flag; CFI/TLI both exceed 1.0 (mathematically out of range) yet are marked `cfi_ok=True`/`tli_ok=True` — **the pipeline's own validity check appears to test only a lower bound, not an upper one, so it is passing invalid values as "acceptable."** This should be treated as a second, more severe instance of unreliable SEM fit-index reporting.

**`structural_paths_baseline.csv`** — BASELINE → 4 first-order factors: OBJECT 0.712 (a normal, sub-unity loading); **CONDITION 0.993, PERSONAL 0.99997, ENERGY 0.99999** — three of the four factors load on BASELINE at essentially 1.0, a Heywood-case-like pattern indicating a numerical identification/scaling problem in the second-order model, not a genuine empirical finding that three of four COR resource dimensions are perfectly interchangeable with the overall baseline construct.

**`cor_sem_factor_scores.csv`** — 339,201 rows. `condition_score`'s distribution is notably asymmetric (range −3.963 to +1.013, a >4:1 tail-length imbalance) compared to the other three factors, consistent with CONDITION's problematic measurement above. Missing-score counts are small (1–2,373 rows per factor, out of 339,201) despite FIML.

**`fes_moderation_path.csv`** — the FES-moderation regression (n=255,324, R²=0.1085):

| Predictor | Coefficient | p-value | Significant |
|---|---|---|---|
| baseline_score | −0.02521 | 0.0 | Yes |
| fes_delta | −0.00063 | 0.0 | Yes |
| baseline_x_fes (interaction) | +0.00004 | **0.608** | **No** |

**The moderation hypothesis itself is not supported** — the interaction term is statistically indistinguishable from zero. Only two additive main effects are significant (higher baseline resource → lower fuel-to-income ratio, as expected; higher FES Delta → slightly lower ratio), and together they explain under 11% of variance. This is an important corrective to the "FES-moderation" framing used elsewhere in the project: the evidence supports additive effects, not moderation.

### `outputs/ukhls_cor_sem/figures/`

- **`cor_sem_loadings.png`** — four bar charts (one per factor) with a "min=0.4" reference line; visually confirms `bill_security` (0.23), `tenure_security` (0.37), `qfhigh_band` (0.38), and `fiyrinvinc_dv` (0.37) all sit below the acceptability line.
- **`cor_sem_loadings_polar.png`** — the same loadings in the project's house-style circular bar chart; purely a re-visualization.
- **`structural_paths_baseline.png`** — bar chart visually confirming the near-1.0 CONDITION/PERSONAL/ENERGY bars against OBJECT's distinctly shorter (~0.71) bar — the clearest single visual of the Heywood-case-like anomaly above.

---

## Stage 2c — COR-CVAE

### `outputs/ukhls_cor_cvae/tables/`

**`cvae_latent_scores.csv`** — 253,913 rows (a smaller, complete-case subsample vs. the SEM's 339,201, since the CVAE lacks FIML's missing-data handling). `cvae_energy_z` is notably right-skewed (range −0.122 to +1.290) unlike the other three, roughly symmetric latent dimensions.

**`cvae_sem_alignment.csv`** — full 4×4 latent-dim × SEM-factor correlation matrix:

| | object | condition | personal | energy |
|---|---|---|---|---|
| z1 (object) | **−0.764** | −0.502 | −0.272 | −0.613 |
| z2 (condition) | −0.456 | **−0.830** | −0.345 | −0.486 |
| z3 (personal) | 0.308 | 0.439 | **0.671** | 0.286 |
| z4 (energy) | −0.636 | −0.594 | −0.358 | **−0.648** |

Two findings worth flagging: **(1) sign inconsistency** — three of four "own-factor" diagonal correlations are negative (z1, z2, z4) while z3 is positive; since VAE latent sign is arbitrary this isn't wrong, but it means the `cvae_*_z` naming convention implies same-direction comparability with the SEM `*_score` columns that doesn't actually hold without sign-flipping. **(2) partial, not full, disentanglement** — z2 (condition, −0.830 vs next-highest −0.456) and z3 (personal, 0.671 vs next-highest 0.439) show clean, well-separated alignment; z1 and z4 are diffuse — z1 correlates almost as strongly with energy (−0.613) as with its own object factor (−0.764), and z4 correlates nearly as strongly with object (−0.636) and condition (−0.594) as with its own energy factor (−0.648). The object/energy latent dimensions have not cleanly separated.

**`cvae_training_history.csv`** — 300 epochs. Total loss falls smoothly from 6.078 (epoch 0) to 1.686 (epoch 299); reconstruction loss 1.442→0.992; KL falls steeply and monotonically to 0.068 by epoch ~50 then stays flat (consistent with the latent posterior converging close to the prior — a possible posterior-collapse risk, though alignment loss stays informative so likely benign). **Alignment loss is non-monotonic**: it rises from 0.536 (epoch 0) to a peak of ~0.568 around epoch 30–40, before falling to 0.273 by epoch 299 — consistent with an annealed alignment-weighting schedule. Train/validation gaps are tiny throughout (<0.5% relative) — no evidence of overfitting.

**`cvae_counterfactual_fes_shift.csv`** — 253,913 rows. Mean predicted-probability shift (forecast vs. current FES conditioning) is tiny: +0.00046 (0.046 percentage points) on a ~29% base rate; mean distance-from-resilient shift is −0.0118 (a small average improvement). But individual-level ranges are wide (probability shift −0.050 to +0.059; distance shift −0.702 to +0.420) — **the near-null population average masks substantial household-level heterogeneity.**

### `outputs/ukhls_cor_cvae/figures/`

- **`cvae_sem_alignment_heatmap.png`**, **`cvae_alignment_polar.png`** — direct re-visualizations of the alignment matrix above; the polar chart now displays each pairing's exact |r| value on the colourful bar itself (fixed this session — previously values were absent from the chart entirely).
- **`cvae_training_curves.png`** — 4-panel line plot confirming the smooth total/reconstruction/KL decay and the non-monotonic, rise-then-fall alignment curve described above.

---

## Stage 2 (merged panel)

### `outputs/ukhls_cleaned/ukhls_panel.csv`

339,201 rows × 62 columns (confirmed, including this session's additions `ethnicity_group` and `disability_free`). Full column list spans household/individual raw variables (hidp, hrpid, gor_dv, fuel/income variables, tenure_dv, etc.), COR-SEM-recoded derived columns, the new ethnicity/disability columns, FES columns (`fes_magnitude`, `fes_actual_prior_year`, `fes_current`, `fes_delta`), and the panel target (`fuel_to_income_ratio`, `high_fuel_vulnerable`). Notably retains all six `inoutflows*` raw variables even though three (`inoutflows2/3/4`) are 96.9% missing — a raw-data-retention choice rather than derived-variable pruning.

---

## Stage 3 — Vulnerability Identification

### `outputs/ukhls_vulnerability/tables/`

**`stage3_vulnerability_scores.csv`** — 339,201 rows, 5 score columns (3 fuzzy c-means memberships + 1 partition coefficient + 1 one-class SVM anomaly score). 85,288 rows (25.1%) are entirely missing (253,913 valid, matching the CVAE's complete-case count). Hard cluster assignment (arg-max) among valid rows: "resilient" 41.6%, "vulnerable to loss" 35.6%, "resource depleted" only 22.8%. `oneclass_anomaly_score` is strongly right-skewed (mean −24.3, median −44.9, max 772.75).

**`stage3_validation_against_objective_ratio.csv`**:

| Method | Pearson r | Spearman r | AUC |
|---|---|---|---|
| fuzzy_resource_depleted | 0.260 | 0.336 | **0.750** |
| oneclass_anomaly_score | 0.156 | 0.015 | 0.567 |

Fuzzy c-means clearly outperforms as a vulnerability proxy; the one-class SVM anomaly score is barely better than chance (AUC 0.567) and essentially uncorrelated in rank terms (Spearman 0.015) — it is detecting a different kind of "unusualness" than fuel poverty specifically.

**`driver_analysis_logistic_regression.csv`** — 13 predictors, n=103,646:

| Predictor | Odds ratio | p-value | Significant |
|---|---|---|---|
| financial_strain_score | **6.85** | 0.0 | Yes |
| tenure_security | 1.40 | 0.0 | Yes |
| health_good | 1.27 | 0.0 | Yes |
| hsrooms | 1.09 | 0.0 | Yes |
| hsbeds | 1.05 | 0.0001 | Yes |
| fes_delta | 1.02 | 0.182 | No |
| dvage | 1.01 | 0.0 | Yes |
| heatch | 0.96 | 0.436 | No |
| sf1_good | 0.92 | 0.112 | No |
| bill_security | 0.78 | 0.0005 | Yes |
| ncars | 0.75 | 0.0 | Yes |
| qfhigh_band | 0.58 | 0.0 | Yes |
| jbstat_security | **0.13** | 0.0 | Yes |

Financial/psychological strain dominates by a wide margin. Two counter-intuitive but plausible signs: `health_good` and `hsrooms`/`hsbeds` all *raise* odds of vulnerability — larger, older homes cost more to heat, and self-rated "good health" may correlate with older homeowners in larger, harder-to-heat properties. `fes_delta`, `heatch`, and `sf1_good` are not significant.

**`driver_analysis_by_region.csv`** — the same regression re-fit per region (156 rows). `financial_strain_score` is the top driver everywhere but its magnitude ranges from OR 14.67 (Northern Ireland, n=1,879) and 14.57 (South East, n=12,880) down to OR 2.86 (London, n=15,231) — a 5× spread in effect size by geography. `jbstat_security` is consistently strongly protective everywhere (OR 0.10–0.21). Smaller-n regions (NI, North East) carry proportionally wider standard errors.

**`policy_vulnerability_by_wave.csv`** — U-shaped trajectory: 11.57% (2009) → trough of 5.09% (2020) → spikes to 10.83% (2022) and 10.62% (2023), the cost-of-living crisis.

**`policy_vulnerability_by_region.csv`** — Northern Ireland 15.62% (rank 1) down to South East 5.96% (rank 12); NI is >5.8 points above the next-highest region (West Midlands, 9.75%).

**`policy_vulnerability_by_ethnicity.csv`** — Black Caribbean 14.06% (n=6,751) highest, down to Any other Asian background 6.66% (n=2,570) lowest; White (7.76%, n=265,950 — the dominant sample group) sits near the low end.

**`policy_vulnerability_by_disability.csv`** — Contains disabled adult 12.86% (n=14,119) vs. No disabled adult 10.33% (n=15,840); total n much smaller than region/ethnicity tables since disability is observed only for responding adults.

**`policy_vulnerability_by_tenure.csv`** — Other 13.10% > Social renting 11.28% > **Owned outright 9.87%** > Private renting 8.50% > Buying with mortgage 4.15%. Owned outright ranking above Private renting is notable — a fuel-to-income measure picks up asset-rich-but-heating-cost-exposed older/outright owners that a pure income measure would not.

**`policy_vulnerability_by_fes_tier.csv`** — Low 7.82%, Moderate 8.05%, High 7.94% — essentially flat across national FES-stress terciles, consistent with `fes_delta`'s non-significance in the driver regression: household vulnerability is dominated by household-level financial strain, not by the macro time-varying price-shock signal.

**`policy_vulnerability_by_region_map.csv`** — same data as the region table, geographically reordered for the choropleth.

**`policy_map_baseline_resource_by_region.csv`** and **`policy_map_financial_strain_by_region.csv`** — **these two files are byte-identical** (both contain both scores for all 12 regions) — a harmless but avoidable data-export duplication. `baseline_resource_score` ranges from South East (0.168, highest) to North East (−0.134, lowest); `financial_strain_score` has a very narrow true range (0.2659–0.2818, ~6% relative spread) that its choropleth (below) visually over-dramatizes.

**`policy_temporal_change_by_region.csv`** / **`policy_temporal_change_map.csv`** (same data, different order) — Northern Ireland improved by far the most (20.76%→10.54%, **−10.21pp**); South West is the only region that got worse (6.45%→6.68%, +0.23pp); South East nearly flat (−0.13pp).

**`policy_vulnerability_region_year_heatmap.csv`** — 12 regions × 17 years. **Data-quality flag**: Northern Ireland's 2024 cell reads exactly 0.000000% against a neighbouring 11–12% trend in 2022–2023 — almost certainly a sparse/zero-sample artifact for that cell, not a genuine finding, and should be caveated or excluded from any presentation of this table.

**JRF external-validation benchmark tables** — `jrf_poverty_benchmark_region.csv`, `jrf_poverty_benchmark_ethnicity.csv`, `jrf_poverty_benchmark_disability.csv`, `jrf_poverty_benchmark_tenure.csv` — independently-published JRF relative-poverty rates (AHC), hardcoded with page/table citations (region: Table 6 p.51; ethnicity: p.9/42; disability: Table 8 p.67; tenure: Table 10 p.95).

**`external_validation_region_comparison.csv`** — merged rates + ranks for 12 regions. Correlation **all regions**: Pearson r=−0.10, Spearman ρ=0.33; **excluding Northern Ireland**: Pearson r=0.68, Spearman ρ=0.73. Northern Ireland alone (rank 1 on our measure, rank 12/last on JRF's) flips the correlation sign.

**`external_validation_ethnicity_comparison.csv`** — 6 groups with a JRF-stated rate. Pearson r=0.27, Spearman ρ=0.14 — weak. Bangladeshi is JRF's #1 most-poor group (56%) but only #4 on this project's measure (9.50%); Black Caribbean is the mirror-image divergence (#1 here at 14.06%, #5 of 6 on JRF at 30%).

**`external_validation_disability_comparison.csv`** — 2 groups, rank order agrees, correlation mathematically undefined at n=2 (both figures literally display "ρ=nan"/"r=nan" in their titles — see figure notes below).

**`external_validation_tenure_comparison.csv`** — 4 groups. Pearson r=0.68, Spearman ρ=0.80 (n=4) — the strongest external agreement of the four dimensions. Owned outright is the one rank-swap (2nd here, 3rd on JRF).

**`ni_oil_heating_evidence_by_region.csv`** — Northern Ireland: 71.2% of households use oil heating, vs. the next-highest region (Wales) at only 9.8% — a 7.3× gap, and the exogenous factor behind NI's outlier vulnerability rate.

**`ni_oil_heating_evidence_within_ni.csv`** — within Northern Ireland: oil-heating households spend £2,017/year on fuel and are 20.9% vulnerable, vs. £1,243/year and 12.3% for non-oil NI households in the same wave/region — a controlled, within-region comparison isolating oil-heating exposure as the driver.

### `outputs/ukhls_vulnerability/figures/`

- **`driver_analysis_by_region.png`** — financial_strain_score odds ratio by region: Northern Ireland (~14.7) and South East (~14.6) highest, London lowest (~2.9) — a ~5× spread.
- **`policy_driver_odds_ratios.png`** — national forest/bar plot with human-readable labels, colour-coded by significance/direction; confirms the CSV's sign pattern exactly.
- **`policy_vulnerability_by_wave.png`** — confirms the U-shape/spike; **title says "2009–2024" but the data only extends to 2023 (wave o)** — a minor labeling inaccuracy worth correcting.
- **`policy_vulnerability_by_region.png`** — Northern Ireland's bar is ~60% longer than the next-longest.
- **`policy_vulnerability_by_ethnicity.png`**, **`policy_vulnerability_by_disability.png`**, **`policy_vulnerability_by_tenure.png`** — bar-chart renderings of the three new breakdowns, confirming the rankings above; Owned outright is coloured "above median" (red) in the tenure chart.
- **`policy_vulnerability_by_fes_tier.png`** — visually confirms the near-flat 7.8/8.0/7.9% pattern.
- **`policy_vulnerability_region_year_heatmap.png`** — 12×17 heatmap; Northern Ireland's row is the only band of dark-red (high-vulnerability) cells in the early years, and its 2024 cell is a stark, likely-artifactual dark green "0" against its own row's trend.
- **`policy_vulnerability_by_region_map.png`** — real UK choropleth; Northern Ireland renders as a solid, visually isolated dark maroon polygon.
- **`policy_map_baseline_resource_by_region.png`** — South East/East of England/South West darkest (highest resource); North East lightest; London's slightly negative score may read as counter-intuitive given its income profile — reflects a needs-adjusted resource construct, not raw income.
- **`policy_map_financial_strain_by_region.png`** — visually implies more regional contrast than the underlying ~0.016-point true spread supports; a readability/interpretation caveat.
- **`policy_temporal_change_map.png`** — Northern Ireland is the only deep-green (most-improved) region; **title says "waves k-o [2019-2024]"**, again one year past the data's actual 2023 endpoint.
- **`external_validation_wave_trend.png`** — this project's wave-level trend with the 2021–2023 cost-of-living-crisis window shaded and JRF's own hardship-tracker findings quoted in the title; visually reproduces JRF's sharp-spike-then-partial-easing shape.
- **`external_validation_{region,ethnicity,disability,tenure}_bars.png`** and **`_scatter.png`** (8 files) — dual-axis bar charts and annotated scatter plots for all four validation dimensions, each titled with its exact correlation statistic. The disability pair literally displays "ρ=nan / r=nan" in its title (mathematically correct for n=2, but reads as an error to an unprepared viewer) while still drawing a dashed best-fit line through the two points — worth a caption caveat if this figure is shown to a non-technical audience. The region scatter's label-collision fix (applied this session — colliding labels now stagger with leader lines back to their markers) is visible in the North East/Scotland/Wales cluster.

---

## Stage 4 — Policy Geography Maps

### `outputs/ukhls_policy_maps/tables/`

**`map1_resource_stress_hotspot.csv`** — 12 regions with `baseline_resource_mean`, `vulnerable_pct`, tertile-based `policy_tier`. Northern Ireland is the only region classified "Low resource / High vulnerability" (the cash-transfer-priority tier); the Midlands/North/Wales form a contiguous "Mid resource / High vulnerability" band (structural/infrastructure-priority tier); South East/South West/East of England cluster as "High resource / Low vulnerability."

**`policy_map2_fuzzy_membership.csv`** — mean fuzzy "vulnerable to loss" membership and % of households near the 0.5 decision boundary, by region. Northern Ireland has the *lowest* mean membership (0.343) but by far the *lowest* near-boundary share (8.37% vs. 21.7–27.8% everywhere else) — its distribution is more polarized/bimodal (households sorted decisively rather than clustering near the boundary), while every GB region clusters tightly (0.347–0.369).

**`map3_vulnerability_vector_shift.csv`** — current vs. forecast predicted probability (the CVAE counterfactual) by region. Every region shifts positively (rising risk) under the shared forecast shock; South East has the *largest* shift (+0.057pp) despite the *lowest* current baseline vulnerability, while Scotland has the *smallest* shift (+0.030pp) despite mid-pack baseline vulnerability — the shift magnitude is essentially decoupled from (if anything, mildly inversely related to) current prevalence.

### `outputs/ukhls_policy_maps/figures/`

- **`policy_map1_resource_stress_hotspot.png`** — two side-by-side real maps plus the 3×3 bivariate policy-tier legend, confirming Northern Ireland's unique tier classification.
- **`policy_map2_fuzzy_membership.png`** — choropleth of mean fuzzy membership; Northern Ireland visually distinct as the lowest, but the near-boundary share (not directly mapped here) is the more analytically interesting number.
- **`policy_map3_vulnerability_vector_shift.png`** — arrows at each region's real centroid; all arrows point the same (rising-risk) direction, of varying length; South East's arrow is the longest, Scotland's the shortest.

---

## Stage 5 — Forward Vulnerability Prediction

### `outputs/ukhls_forward_prediction/tables/`

**`stage5_transition_pairs.csv`** (261,759 rows) — 14 wave-to-wave transitions (a→b through n→o). Row counts decline from 24,404 (b→c) to 13,725 (m→n) — cumulative attrition — before an uptick at n→o (17,144), a refreshment-sample effect. `high_fuel_vulnerable_t1`: 184,940 not-vulnerable, 14,624 vulnerable, 62,195 missing (attrition on the target wave) — a ~7.3% raw vulnerability rate among non-missing labels.

**`stage5_validation_metrics.csv`** — held out on m→n and n→o: n_train=177,408, n_validation=21,161, **AUC=0.758**, Pearson r=0.265 (vs. the continuous ratio). An AUC of 0.758 on genuinely unseen, walk-forward-held-out data is a solid discriminative result; the much weaker Pearson r suggests the model is considerably better at classifying who crosses the vulnerability threshold than at explaining variance in the continuous ratio.

**`stage5_driver_coefficients.csv`** — 8 predictors (final model, refit on all 14 transitions):

| Predictor | Odds ratio | p-value | Significant |
|---|---|---|---|
| **fes_magnitude** | **52.00** | <0.001 | Yes |
| financial_strain_score | 4.86 | <0.001 | Yes |
| object_score | 1.43 | <0.001 | Yes |
| dvage | 1.018 | <0.001 | Yes |
| heatch | 0.951 | 0.149 | No |
| personal_score | 0.936 | <0.001 | Yes |
| condition_score | 0.766 | <0.001 | Yes |
| energy_score | 0.307 | <0.001 | Yes |

`fes_magnitude`'s odds ratio of 52.0 dominates the model — a one-unit increase is associated with a >50-fold increase in the odds of becoming vulnerable next wave. Since (per `stage5_forward_predictions.csv` below) `fes_magnitude_used` is a single constant applied to every household, **this coefficient is the entire mechanism by which the forecast shock enters Stage 5's predictions, but it cannot by itself differentiate between households or regions** — that differentiation comes entirely from the other 7 covariates. `energy_score`'s and `condition_score`'s protective (< 1) odds ratios are plausible only if these index asset/efficiency quality rather than raw consumption — worth a definitional check.

**`stage5_forward_prediction_by_month.csv`** — 12 months, n=19,140 total. April is peak-risk (7.36%), September the trough (6.44%) — a 0.92-percentage-point, ~14% relative spread. Winter/early-spring months (Jan, Feb, Apr, Oct, Nov, all ≥7.0%) generally exceed mid/late-summer months (Jul, Aug, Sep, all ≤6.7%).

**`stage5_forward_prediction_map.csv`** — 12 regions (n=19,134 of 19,140; 6 households have unmapped region). Northern Ireland highest (7.74%), South West lowest (6.18%) — a 1.56pp spread, narrower than the historical prevalence spread (5.96–15.62%, ~9.66pp) due to model shrinkage toward the shared constant shock term. This forward ranking closely mirrors the historical ranking (NI highest, SE/SW lowest at both) — the model projects forward the existing geography of disadvantage rather than predicting a reshuffling.

**`stage5_forward_predictions.csv`** (19,140 households) — mean predicted probability 6.89%, **median only 4.01%** (right-skewed distribution), min 0.16%, max 87.97%. 228 households (1.2%) predicted >50%; 3,555 (18.6%) >10%. `fes_magnitude_used` = a single constant, −1.0608, applied identically to every row. `predicted_target_year` splits into 10,937 households (interviewed 2023) → predicted for 2024, and 8,203 (interviewed 2024) → predicted for 2025.

### `outputs/ukhls_forward_prediction/figures/`

- **`stage5_validation_roc.png`** — ROC curve for the held-out m→n/n→o validation, AUC=0.758 matching the CSV exactly; a well-behaved, comfortably-above-diagonal curve with no crossing or inversion.
- **`stage5_forward_prediction_by_month.png`** — bar chart confirming April's peak and September's trough, colour-split roughly at the ~6.7–6.8% mark.
- **`stage5_forward_prediction_map.png`** — the map redesigned this session to show percentages (e.g. "7.7%") instead of raw probability fractions, resolving the earlier problem where most regions rounded to an indistinguishable "0.07."

---

## Summary of data-quality and methodological flags found across all 150 files

1. **Model selection is hindsight-based** (Stage 1): the FES pipeline's "best model" choice uses post-hoc realised accuracy, which disagrees with the genuine backtest ranking in 4 of 6 series/mode cells — most dramatically for electricity-core, where the backtest's top pick (SARIMA) is the worst real-world performer by a factor of 11.
2. **`FES_selected` never differs from `FES_core`** — the per-series core/macro selection mechanism has not, in this run, picked macro for any of the three series, even though carbon's own best model is macro-mode.
3. **FES_Macro (the aggregate index) validates better than FES_Core/Selected** against realised-stress benchmarks, an unresolved tension with the per-series accuracy results.
4. **Two SEM fit-index tables report values outside their valid mathematical range** (CFI/TLI outside [0,1]; one SRMR value of ~9.2 billion), and the pipeline's own "ok" flags do not catch an upper-bound violation (CFI/TLI >1 marked `True`).
5. **A Heywood-case-like pattern** in the second-order SEM structural paths (three of four factors loading ~0.99–1.00 on BASELINE).
6. **The FES-moderation interaction term is non-significant** (p=0.608) — the project's own "FES-moderation" framing is not supported by this result; only additive main effects hold.
7. **CVAE-SEM latent alignment is partial**: two of four latent dimensions (object, energy) are diffuse rather than cleanly disentangled from each other.
8. **Two duplicate table exports** (`policy_map_baseline_resource_by_region.csv` and `policy_map_financial_strain_by_region.csv` are byte-identical).
9. **A likely data-sparsity artifact**: Northern Ireland's 2024 cell in the region×year heatmap reads exactly 0.0% against a neighbouring ~11–12% trend.
10. **Two chart titles state "2024" as an endpoint where the underlying wave data only extends to 2023** (`policy_vulnerability_by_wave.png`, `policy_temporal_change_map.png`).
11. **Two external-validation charts display "ρ=nan"/"r=nan"** (mathematically correct for n=2 disability categories, but reads as an error without a caption explaining why).
