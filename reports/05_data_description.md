# Data Description: Sources, Coverage, and Distributions (rerun v2)

**Project:** Anticipatory Fuel Stress Watch (AFSW) — forecasting anticipatory energy–carbon stress and household fuel vulnerability in the UK.
**Scope:** every raw and processed *input* to the v2 analysis: coverage, structure, descriptive statistics, and how the UKHLS household panel is turned into the analytical sample. Includes all Chapter 3 thesis tables and data figures (Tables 3-2 to 3-7, A-13, A-14; Figures 3-2 to 3-6, A-2). Generated *outputs* are catalogued in [`01_outputs_catalog.md`](01_outputs_catalog.md), methods in [`06_methodology.md`](06_methodology.md).
**Status:** branch `rerun-v2`, analysis frozen 2026-09-26. Thesis figures are in `outputs_v2/thesis_assets_v2/figures/`, a local build (`python scripts/build_thesis_assets.py`) that is not tracked in git.

---

## 1. Overview

The project draws on five kinds of input:

1. **Raw energy-market and macro-economic time series** (`data/raw/`, 9 files): UK gas, electricity and carbon prices, inflation, GDP, electricity demand, temperature and exchange rates, from as early as 1975 to 2026.
2. **Processed time series** (`data/processed/`, 4 files): the raw series cleaned onto a common monthly index. v2 forecasts use **only the core file** (`core_energy_carbon.csv`: gas, electricity and carbon growth). The macro controls are described for completeness; they fed v1's macro-mode models, which v2 dropped (plan Stage 2).
3. **UKHLS household panel** (`data/raw/ukhls/`, 30 Stata files): Understanding Society waves a–o, household (`hhresp`) and individual (`indresp`) files. Licensed (UK Data Service SN 6614) and not in the repository.
4. **Region boundaries** (`data/geo/`, 1 GeoJSON file): NUTS1 regions for the maps in Figures 4-5 and 4-7.
5. **JRF *UK Poverty 2025*** (`UK Poverty 2025.pdf` in the repository root): the external income-poverty benchmark. Values are taken from the text and tables and recorded with page references in `outputs_v2/jrf/jrf_metadata.csv`.

---
## 2. Raw Macro-Economic and Energy-Market Data (`data/raw/`)

### 2.1 `Carbon Emissions Futures Historical Data UK.csv`

**Structure:** 253 rows, columns `Date, Price, Open, High, Low, Vol., Change %`. `Price`/`Open`/`High`/`Low` are numeric; `Vol.` and `Change %` are stored as strings (e.g. `"6.44K"`, `"2.57%"`) requiring parsing before use.

**Coverage:** monthly frequency, **2005-05-01 to 2026-05-01** (dates recorded DD/MM/YYYY). The 253-day gap distribution (147×31 days, 84×30, 16×28, 5×29) confirms consistent month-start sampling with no missing months.

**Distribution (`Price`, EUR/tonne CO₂):** mean 28.33, std 27.92, min 0.01, median 15.84, max 95.88 — a right-skewed distribution consistent with the EU-ETS carbon market's historical trajectory from near-worthless allowances in its early (2005–2012) phase to substantially higher prices from the mid-2010s onward.

### 2.2 `UK NBP Natural Gas Quaterly Futures Historical Data UK.csv`

**Structure:** identical 7-column schema to the carbon file above.

**Coverage:** 268 rows, monthly frequency, **2004-02-01 to 2026-05-01**.

**Distribution (`Price`, pence/therm):** mean 71.52, std 68.66, min 11.82, median 54.04, max 544.68 — the maximum (544.68p/therm) corresponds to the extreme 2022 European gas-price spike; the wide gap between mean/median and the maximum reflects a small number of crisis months dominating the tail.

### 2.3 `gas.csv` (ONS series CZDA, dataset MM23)

**Structure:** a 2-column ONS statistical export (`Title`, `RPI:Percentage change over 12 months - Gas`), with an 8-row metadata header (Title, CDID, Source dataset ID, PreUnit, Unit=%, Release date 22-04-2026, Next release 20 May 2026, Important notes) preceding the data body. The data body itself mixes three frequencies in a single column (annual, quarterly, monthly rows are interleaved by label, not separated into distinct columns), requiring frequency-filtering before use — the project's own loader (`src/data_loader.py`) handles this filtering.

**Coverage:** 38 annual rows (1988–2025), 153 quarterly rows (1988 Q1–2026 Q1), 459 monthly rows (1988 JAN–2026 MAR).

**Distribution (monthly RPI 12-month % change in gas prices):** mean 6.35%, std 22.01, min −38.3%, median 1.90%, max 132.9% — a heavily right-skewed distribution (the median is far below the mean), reflecting that large gas-price increases are more extreme and more frequent than large decreases over this 38-year window.

### 2.4 `electricity.csv` (ONS series D7DT, dataset MM23)

**Structure:** same layout as `gas.csv`; column `CPI INDEX 04.5.1 : ELECTRICITY 2015=100`. Unit: index, base year=100. Release date 22-04-2026.

**Coverage:** 38 annual, 153 quarterly, 459 monthly rows, 1988–2026 (same three-frequency structure as gas.csv).

**Distribution (monthly index level, 2015=100):** mean 83.30, std 50.83, min 33.1, median 64.9, max 240.9 — this is a price *level* index, not a growth rate; the project's own preprocessing (`src/data_loader.py`) derives `electricity_growth` (12-month % change) from this level series, which appears in the processed data (Section 3.1).

### 2.5 `cpih08_188.xlsx` (ONS CPIH aggregate 04 — Housing, water, electricity, gas & other fuels)

**Structure:** an Excel workbook with two sheets — `Dataset` (a single UK-level time series) and `Metadata` (47 descriptive rows: release date 18 Feb 2026, monthly frequency, unit "Index: 2015=100", Open Government Licence v3.0).

**Coverage:** 373 monthly observations, **January 1988 to January 2019** — notably, **this file's coverage stops in 2019**, unlike every other ONS series in this data collection, which extend through 2026. Any use of this series for post-2019 analysis would require either an updated extract or reliance on the overlapping macro-controls series instead.

**Distribution:** mean 75.48, std 18.74, min 38.7, median 73.8, max 106.6; zero missing values across all 373 observations.

### 2.6 `historic_demand_2009_2024.csv` (National Grid ESO half-hourly electricity demand)

**Structure:** the largest raw input file (26MB), 279,264 data rows × 24 columns: `settlement_date, settlement_period` (1–50, with 46 or 50 rather than the nominal 48 on clock-change days — confirming genuine half-hourly granularity with correct daylight-saving handling), core demand columns (`nd`=national demand, `tsd`=transmission system demand, `england_wales_demand`), embedded generation columns (wind/solar generation and capacity), storage/pumping, a full set of interconnector flow columns (`ifa_flow`, `ifa2_flow`, `britned_flow`, `moyle_flow`, `east_west_flow`, `nemo_flow`, `nsl_flow`, `eleclink_flow`, `viking_flow`, `greenlink_flow`), and an `is_holiday` flag.

**Coverage:** **2009-01-01 to 2024-12-05**, half-hourly (the finest-grained raw data source in the project, aggregated up to monthly means for use in `macro_controls.csv`).

**Distribution (units MW, implied by National Grid ESO convention):** `nd` (national demand) mean 31,186.6, std 7,827.3, min 13,367, max 59,095; `tsd` mean 32,627.8, std 7,710.0, min 0 (a data artefact — almost certainly an early recording gap rather than a genuine zero-demand half-hour), max 60,147; `england_wales_demand` mean 28,389.0, std 7,087.6, min 0, max 53,325. Zero nulls in the three core demand columns and in the date/period columns; several interconnector-flow columns are legitimately NaN in early rows because those physical interconnectors (e.g. NSL to Norway, ElecLink, Viking, Greenlink) had not yet been commissioned — a genuine absence-of-infrastructure, not a missing-data problem.

### 2.7 `mgdp.csv` (ONS Monthly GDP by industry, chained-volume-measure, seasonally adjusted)

**Structure:** a wide ONS export, 357 total rows × 208 columns; the first 7 rows are metadata (Title, CDID, PreUnit, Unit, Release Date, Next release, Important Notes), followed by 350 monthly observation rows.

**Coverage:** **January 1997 to February 2026**.

**Distribution (headline "Gross Value Added — Monthly (Index 1dp), CVM SA"):** mean 84.08, std 11.29, min 61.6, median 83.2, max 103.0; zero missing values. The file also carries a specific, domain-relevant "Electricity, gas, steam and air conditioning supply" sub-index: level mean 223.53 (std 68.67, min 90.0, max 362.5), and its own period-on-period growth column (mean −0.20%, std 3.30, min −10.1%, max 12.3%, n=349 non-null — the first observation has no prior period to compute growth against).

### 2.8 `monthly-temperature-anomalies.csv` (global temperature anomaly data, multi-country)

**Structure:** a global panel (Our World in Data-style format), 220,668 total rows across 213 distinct countries/entities, columns `Entity, Code, Day, Temperature anomaly`.

**Coverage (UK subset only, `Entity="United Kingdom"`, `Code="GBR"`):** 1,036 rows, **1940-01-15 to 2026-04-15**, monthly (observation day is consistently the 15th of each month).

**Distribution (UK temperature anomaly, °C relative to a historical baseline):** mean −0.44°C, std 1.32, min −6.99, median −0.39, max 3.65; zero missing values in the UK slice. In v1 this series fed the temperature-volatility control used by the macro-mode forecasts; v2 forecasts are core-only and do not use it.

### 2.9 `series-190626.csv` (ONS GBP/EUR exchange rate, series THAP, dataset MRET)

**Structure:** same 2-column ONS export format as `gas.csv`/`electricity.csv`; column `Average Sterling exchange rate: Euro XUMAERS`.

**Coverage:** 51 annual rows (1975–2025), 205 quarterly rows (1975 Q1–2026 Q1), 616 monthly rows (1975 JAN–2026 APR) — but only **352 of the 616 monthly rows are non-null**, since the Euro (and its ECU precursor) did not exist before the mid-1990s; genuine monthly values begin in **January 1997**.

**Distribution (monthly, non-null values only):** mean 1.3064, std 0.1728, min 1.0867, median 1.2214, max 1.6994.

---

## 3. Processed Time-Series Data (`data/processed/`)

### 3.1 `core_energy_carbon.csv` — the primary gas/electricity/carbon growth-rate file

This is the only input to the v2 forecasts (Stage 2, core mode): every model is fit to these three growth series.

**Structure:** 239 rows × 5 columns — `date, gas_growth, electricity_index, electricity_growth, carbon_growth`.

**Coverage:** **2006-05-01 to 2026-03-01**, monthly, zero missing values in any column.

**Distribution:**

| Column | Mean | Std | Min | 25th pctile | Median | 75th pctile | Max | Skew | Excess kurtosis |
|---|---|---|---|---|---|---|---|---|---|
| gas_growth (%) | 9.39 | 29.76 | −38.30 | −5.95 | 1.40 | 13.10 | 132.90 | 2.40 | 6.84 |
| electricity_index | 119.01 | 47.80 | 60.70 | 84.90 | 100.20 | 130.10 | 240.90 | 1.09 | 0.07 |
| electricity_growth (%) | 7.80 | 15.49 | −21.07 | −0.30 | 5.70 | 9.65 | 66.71 | 1.84 | 4.72 |
| carbon_growth (%) | 5.98 | 149.33 | −670.93 | −31.69 | 3.78 | 35.81 | 734.73 | 0.27 | 8.66 |

`carbon_growth`'s standard deviation (149.33) is roughly 25× its mean (5.98), and its range (−670.93 to +734.73) is by far the widest of the three series — a direct consequence of computing percentage growth off a very low base price in the carbon market's early (pre-2013) years, when small absolute price movements translate into enormous percentage swings. It is also why carbon's forecasts carry the widest prediction intervals of the three series (mean 95% PI width 45.8 percentage points in v2; `outputs_v2/fes_eval/uncertainty_pi.csv`).

The skew/excess-kurtosis columns quantify this more precisely: `carbon_growth`'s own skew (0.27) is actually the mildest of the four — its extreme range is driven by symmetric fat tails (excess kurtosis 8.66, the highest of the four) rather than one-sided outliers — whereas `gas_growth` (skew 2.40) and `electricity_growth` (skew 1.84) are both markedly right-skewed, consistent with the "large increases more extreme/frequent than large decreases" pattern already noted for gas's raw RPI series in Section 2.3. `electricity_index` is the one column here that is a price *level*, not a growth rate, and its near-zero excess kurtosis (0.07) reflects a genuinely multi-modal rather than fat-tailed shape — visible as three separate clusters in the figure below (a pre-2021 ~60–100 regime, a 2022-crisis-era jump to ~125–145, and a distinct ~190–240 cluster for the most recent months).


The values above are population moments with quantiles, from the tracked descriptive table `outputs/data_description/tables/core_variable_summary_stats.csv`. The thesis version (Table 3-6) uses sample-corrected skew and kurtosis, which is why its values are slightly larger:

| Series | n | Missing | Mean | SD | Median | Skew | Excess kurtosis |
|---|---|---|---|---|---|---|---|
| gas_growth | 239 | 0 | 9.39 | 29.76 | 1.40 | 2.42 | 7.01 |
| electricity_index | 239 | 0 | 119.01 | 47.80 | 100.20 | 1.10 | 0.09 |
| electricity_growth | 239 | 0 | 7.80 | 15.49 | 5.70 | 1.85 | 4.85 |
| carbon_growth | 239 | 0 | 5.98 | 149.33 | 3.78 | 0.28 | 8.87 |

*Table 3-6. Core forecasting series, May 2006–March 2026 (`outputs_v2/thesis_assets_v2/tables/T3-6_forecasting_variables.csv`).*

![Figure 3-6. Core forecasting series](../outputs_v2/thesis_assets_v2/figures/fig3-6_core_forecast_vars.png)

*Figure 3-6. Core forecasting series, May 2006–March 2026: the three growth series forecast in Stage 2.*

![Core variable distributions](../outputs/data_description/figures/core_variable_distributions.png)

*Monthly histograms for the four `core_energy_carbon.csv` columns, with mean (dotted) and median (dashed) marked (`src/data_description_overview.py::run_core`; `outputs/data_description/figures/core_variable_distributions.{png,pdf}`). This descriptive module predates v2, but it describes the inputs, which v2 did not change.*

### 3.2 `macro_controls.csv`

**Structure:** 255 rows × 48 columns, **2005-01-01 to 2026-03-01**, monthly.

**Content categories:** level variables (`gas_futures_price`, `gbp_eur_rate`, electricity-demand means/peaks, embedded wind/solar generation and capacity means, `pump_storage_pumping_mean`, `interconnector_net_flow_mean`); growth/return transforms (`inflation_growth`, `gdp_growth`, `gas_futures_log_return`, `gas_futures_yoy_growth`, and a `_yoy_growth` variant for each demand/generation series, plus `gbp_eur_yoy_change`/`gbp_eur_mom_change`); a full set of `_lag1` (and some `_lag12`) lagged versions of the above; and two regime/seasonal dummy variables (`post_2016_electricity_regime`, `winter_dummy`).

**Distribution (selected columns):** `inflation_growth` mean 1.56%, max 6.63%; `gdp_growth` mean 0.13%, max 9.30%; `gas_futures_price` mean 72.82, max 544.68 (consistent with the raw gas-futures file in Section 2.2); `gbp_eur_rate` mean 1.224, max 1.508; `electricity_demand_mean` mean 31,215.8, max 43,693.0; `holiday_share` mean 0.0206, max 0.10; `post_2016_electricity_regime` mean 0.482 (≈48% of the sample period falls after 2016); `winter_dummy` mean 0.424 (≈42% of months are winter months, consistent with a 4-of-12-months winter definition plus some rounding).

**Full distribution table (all 28 base, non-lagged, non-dummy columns):**

| Variable | Mean | Std | Min | p25 | Median | p75 | Max | N (non-null) | N missing | Skew | Excess kurtosis |
|---|---|---|---|---|---|---|---|---|---|---|---|
| inflation_growth (%) | 1.559 | 1.695 | −1.107 | 0.000 | 1.509 | 2.345 | 6.627 | 243 | 12 | 0.80 | 0.05 |
| weather_volatility | 1.030 | 0.277 | 0.425 | 0.849 | 0.977 | 1.216 | 1.990 | 250 | 5 | 0.90 | 1.35 |
| gdp_growth (%) | 0.131 | 1.628 | −19.200 | −0.100 | 0.200 | 0.500 | 9.300 | 255 | 0 | −6.02 | 82.99 |
| gas_futures_price (p/therm) | 72.818 | 69.800 | 11.820 | 42.695 | 55.140 | 71.495 | 544.680 | 255 | 0 | 4.11 | 20.04 |
| gas_futures_log_return | 0.594 | 19.595 | −72.367 | −8.497 | 0.425 | 8.548 | 92.102 | 255 | 0 | 0.56 | 3.54 |
| gas_futures_yoy_growth (%) | 29.045 | 102.429 | −79.788 | −27.826 | 2.849 | 45.050 | 563.319 | 254 | 1 | 2.64 | 8.12 |
| electricity_demand_mean (MW) | 31,215.79 | 4,879.28 | 21,610.89 | 27,939.98 | 31,283.00 | 34,230.74 | 43,693.05 | 207 | 48 | 0.19 | −0.32 |
| electricity_demand_peak (MW) | 42,626.72 | 6,946.77 | 29,061.00 | 37,518.00 | 41,944.00 | 47,058.00 | 59,095.00 | 207 | 48 | 0.15 | −0.53 |
| electricity_tsd_mean (MW) | 32,708.79 | 4,876.02 | 22,610.57 | 28,884.16 | 33,059.80 | 35,544.76 | 45,555.60 | 207 | 48 | 0.33 | −0.23 |
| england_wales_demand_mean (MW) | 28,435.45 | 4,365.60 | 19,599.77 | 25,554.38 | 28,733.63 | 31,021.38 | 39,286.27 | 207 | 48 | 0.13 | −0.37 |
| embedded_wind_generation_mean (MW) | 1,373.36 | 688.80 | 297.32 | 787.69 | 1,253.97 | 1,855.66 | 3,446.89 | 207 | 48 | 0.48 | −0.57 |
| embedded_solar_generation_mean (MW) | 838.35 | 777.36 | 0.00 | 213.18 | 537.90 | 1,520.41 | 2,874.29 | 207 | 48 | 0.72 | −0.76 |
| embedded_wind_capacity_mean (MW) | 4,613.34 | 1,972.44 | 1,403.00 | 2,129.98 | 5,305.00 | 6,527.00 | 6,622.00 | 207 | 48 | −0.38 | −1.56 |
| embedded_solar_capacity_mean (MW) | 9,334.03 | 6,096.99 | 0.00 | 2,675.34 | 12,372.00 | 13,080.00 | 17,194.07 | 207 | 48 | −0.39 | −1.38 |
| pump_storage_pumping_mean (MW) | 300.11 | 94.08 | 29.33 | 241.06 | 333.10 | 373.70 | 492.01 | 207 | 48 | −0.62 | −0.44 |
| interconnector_net_flow_mean (MW) | 1,809.72 | 1,399.86 | −2,999.90 | 1,126.13 | 2,103.65 | 2,696.33 | 5,390.54 | 207 | 48 | −0.78 | 0.97 |
| holiday_share | 0.021 | 0.027 | 0.000 | 0.000 | 0.000 | 0.032 | 0.100 | 207 | 48 | 0.88 | −0.68 |
| electricity_demand_yoy_growth (%) | −1.319 | 5.387 | −19.286 | −4.743 | −1.570 | 1.750 | 22.246 | 195 | 60 | 0.40 | 2.69 |
| electricity_peak_yoy_growth (%) | −2.003 | 4.310 | −16.695 | −4.544 | −2.371 | 0.340 | 15.813 | 195 | 60 | 0.39 | 1.82 |
| embedded_wind_generation_mean_yoy_growth (%) | 11.934 | 36.488 | −60.141 | −8.382 | 5.964 | 30.785 | 185.971 | 195 | 60 | 1.10 | 2.37 |
| embedded_solar_generation_mean_yoy_growth (%) | 190.995 | 730.238 | −34.676 | 1.813 | 26.739 | 61.244 | 8,395.819 | 180 | 75 | 8.61 | 88.86 |
| embedded_wind_capacity_mean_yoy_growth (%) | 10.119 | 16.017 | −20.110 | 0.283 | 4.681 | 15.481 | 71.101 | 195 | 60 | 1.94 | 4.31 |
| embedded_solar_capacity_mean_yoy_growth (%) | 115.048 | 304.374 | 0.000 | 5.496 | 11.421 | 69.050 | 1,435.128 | 180 | 75 | 3.30 | 9.66 |
| pump_storage_pumping_mean_yoy_growth (%) | 1.374 | 60.976 | −83.761 | −18.165 | −4.818 | 5.730 | 576.187 | 195 | 60 | 7.03 | 57.95 |
| interconnector_net_flow_mean_yoy_growth (%) | −29.168 | 652.032 | −4,587.436 | −35.509 | 6.357 | 50.606 | 5,719.615 | 195 | 60 | 0.88 | 44.99 |
| gbp_eur_rate | 1.224 | 0.116 | 1.087 | 1.147 | 1.177 | 1.263 | 1.508 | 255 | 0 | 1.20 | 0.10 |
| gbp_eur_yoy_change (%) | −0.913 | 6.562 | −20.342 | −4.273 | 0.095 | 3.046 | 15.007 | 255 | 0 | −0.49 | 0.38 |
| gbp_eur_mom_change (%) | −0.072 | 1.686 | −8.288 | −0.931 | 0.068 | 1.016 | 3.742 | 255 | 0 | −0.94 | 2.69 |

Two shape patterns stand out: `gdp_growth`'s excess kurtosis (82.99, by far the largest in the table) and strongly negative skew (−6.02) are both driven by a single extreme observation — the 2020 COVID-lockdown GDP collapse (min −19.2%) sitting far from an otherwise tightly clustered series (p25/median/p75 all within [−0.1, 0.5]); and the `_yoy_growth` columns for wind/solar generation and capacity are all right-skewed with large excess kurtosis (up to 88.86 for `embedded_solar_generation_mean_yoy_growth`), reflecting that year-on-year growth off a small base is mechanically volatile in the same way `carbon_growth` is in Section 3.1 — early in the panel, embedded solar/wind capacity was small enough that modest absolute MW increases produce enormous percentage swings.


![Figure A-2. Macro covariates](../outputs_v2/thesis_assets_v2/figures/figA_macro_covariates.png)

*Figure A-2. Macro covariates (descriptive only; not used by the v2 core-only forecasts).*

![Macro variable distributions](../outputs/data_description/figures/macro_variable_distributions.png)

*Monthly histograms for 9 representative `macro_controls.csv` columns, 2005-01 to 2026-03 (`src/data_description_overview.py::run_macro`; `outputs/data_description/figures/macro_variable_distributions.{png,pdf}`; all columns in `outputs/data_description/tables/macro_variable_summary_stats.csv`).*

**A data-handling note, corrected against the current data file:** the electricity-demand/generation-derived columns (`electricity_demand_mean`, `electricity_demand_peak`, `electricity_tsd_mean`, `england_wales_demand_mean`, the `embedded_wind_*`/`embedded_solar_*` columns, `pump_storage_pumping_mean`, `interconnector_net_flow_mean`, `holiday_share`, and their `_yoy_growth` derivatives) are genuinely **NaN, not 0**, for the 48 rows (2005-01 to 2008-12) before `historic_demand_2009_2024.csv`'s 2009 coverage begins — confirmed directly against `data/processed/macro_controls.csv` for this update (an earlier draft of this report stated these were zero-filled; that is no longer accurate for the current data file and has been corrected here). The two solar columns are the one exception with a genuine, non-missing **0**: `embedded_solar_generation_mean`/`embedded_solar_capacity_mean` are legitimately 0.0 for 15 rows in the early-2010s, reflecting that grid-connected embedded solar capacity in Great Britain was genuinely near-zero before roughly 2011, not a fill artefact.

### 3.3 `core_processed.csv` — feature-engineered expansion of `core_energy_carbon.csv`

**Structure:** the same 239 rows and the same 2006-05-01–2026-03-01 date range as `core_energy_carbon.csv`, expanded from 5 to **69 columns**.

**Added features:** calendar encodings (`month`, `sin_month`/`cos_month`, `month_sin`/`month_cos` — appears twice under slightly different naming, worth noting as a minor redundancy — `winter_dummy`, `time_index`, `year`); an STL-style seasonal decomposition of gas and electricity growth into `_trend`/`_seasonal`/`_irregular` components; regime dummy variables (`post_2016_electricity_regime`, `post_2008_regime`, `low_price_regime`); boolean outlier flags for the growth series and their decomposition components; signed-log transforms (e.g. `gas_growth_signed_log`, preserving sign while compressing magnitude for series that can be negative); and a full lag/rolling-statistics feature set (`_lag1/_lag3/_lag6/_lag12`, `_rm3/_rstd3/_rm6/_rstd6/_rm12/_rstd12` — rolling mean/std at 3/6/12-month windows) for each of the three growth series. This expansion is purely additive feature engineering — no rows are added, removed, or altered in their underlying values relative to `core_energy_carbon.csv`.

### 3.4 `macro_processed.csv` — feature-engineered expansion of `macro_controls.csv`

**Structure:** the same 255 rows and 2005-01-01–2026-03-01 date range as `macro_controls.csv`, expanded from 48 to **158 columns**.

**Added features:** calendar encodings (`month`, `sin_month`/`cos_month`, `time_index`); for every `_lag1` macro control already present, a further set of second-order lag/rolling features (`_lag1_lag1`, `_lag1_lag3`, `_lag1_rm3`, `_lag1_rstd3`, `_lag1_rm6`, `_lag1_rstd6`); rolling statistics on the `post_2016_electricity_regime` dummy itself; and one cross-reference column, `electricity_growth_lag12`, linking back to the energy-carbon series' own 12-month-lagged value. Again, purely additive feature engineering relative to `macro_controls.csv` — identical rows and dates, no value alterations.


---

## 4. UKHLS Household Panel Survey Data (`data/raw/ukhls/`)

30 Stata (`.dta`) files: one household-level (`hhresp`) and one individual-level (`indresp`) file per wave, for all 15 waves (a through o). Every wave's row and variable count below was verified independently by reading each file's Stata header metadata directly (not estimated or taken from prior documentation), covering **all 15 waves, not a sample**.

### 4.1 Household-level files (`{wave}_hhresp.dta`)

| Wave | Variables | Rows | Wave | Variables | Rows | Wave | Variables | Rows |
|---|---|---|---|---|---|---|---|---|
| a | 227 | 30,169 | f | 448 | 24,454 | k | 317 | 18,139 |
| b | 227 | 30,484 | g | 341 | 23,033 | l | 507 | 16,856 |
| c | 225 | 27,751 | h | 525 | 21,746 | m | 537 | 16,156 |
| d | 527 | 25,817 | i | 334 | 20,048 | n | 600 | 21,385 |
| e | 290 | 24,325 | j | 496 | 19,252 | o | 466 | 19,586 |

**Total across all 15 waves: 339,201 household-wave rows** — this independently confirms (via a completely separate data-reading approach: raw Stata header metadata, not the pipeline's own processed panel) the same 339,201 figure already reported throughout this project's outputs and README, a useful cross-check that the two counting methods agree exactly.

Two patterns are worth noting: (1) row counts decline fairly steadily from wave a's 30,169 down to a trough of 16,156 at wave m, before rising again at wave n (21,385) — consistent with cumulative panel attrition followed by a sample refreshment/boost around wave n; and (2) the number of variables collected per wave grows substantially over the panel's life (227 at wave a to 600 at wave n), reflecting additional questionnaire modules introduced over the study's 15-year span — meaning later waves are structurally richer in available covariates than earlier ones, a consideration for any analysis (like the pooled CFA) that pools across all 15 waves and can only use variables available in every wave it includes.

### 4.2 Individual-level files (`{wave}_indresp.dta`)

| Wave | Variables | Rows | Wave | Variables | Rows | Wave | Variables | Rows |
|---|---|---|---|---|---|---|---|---|
| a | 1,393 | 50,994 | f | 2,111 | 45,192 | k | 3,380 | 32,008 |
| b | 1,660 | 54,568 | g | 2,850 | 42,170 | l | 2,773 | 29,271 |
| c | 3,102 | 49,692 | h | 2,168 | 39,294 | m | 3,219 | 27,998 |
| d | 2,111 | 47,074 | i | 3,180 | 36,058 | n | 3,974 | 35,471 |
| e | 2,625 | 44,837 | j | 2,484 | 34,319 | o | 3,483 | 32,849 |

**Total across all 15 waves: 601,795 individual-wave rows.**

Individual-level files are far wider than household-level files (up to 3,974 variables at wave n, vs. a maximum of 600 for hhresp), reflecting the much larger individual questionnaire instrument, and are correspondingly larger on disk (up to 176MB for a single wave's `indresp.dta`). This project's own preprocessing reads only a deliberately small, named subset of these thousands of available individual-level variables per wave (documented in full in `src/ukhls_mapping.py`'s variable registry) — the vast majority of each `indresp` file's content is not used by this analysis, which is a normal and expected consequence of UKHLS's breadth as a general-purpose household panel rather than a survey purpose-built for this project's specific research questions.


### 4.3 From raw files to the v2 panel

The panel has one row per `hhresp` household-wave (339,201 rows, 127 columns; `src/ukhls_preprocessing.py`). Individual items from `indresp` are aggregated to the household: ordinal and continuous items by the mean across responding adults, employment flags by the maximum ("does any adult work"), and disability as "any observed adult is disabled". Ethnicity is that of the household reference person. The panel is row-level, licensed data. It is written to `outputs_v2/ukhls_cleaned/ukhls_panel.csv` and never committed.

v2 corrected the following input errors (plan §9; [`v1_to_v2_change_summary.md`](../outputs_v2/reports/v1_to_v2_change_summary.md)):

| Item | v1 | v2 |
|---|---|---|
| Interview timing | `month` = the *sample* (address issue) month | Actual household interview date `intdatey`/`intdatem`. Only 24–75% of households per wave are interviewed in their sample month, and 3.5–10% in a different calendar year. 7 rows lack a date (sample month used); 300 interviews fall in 2025 |
| Missing codes | −10/−11/−20/−21 kept as values (e.g. wave f `ncars` and `carval` = −10 for 2,468 households) | Recoded to missing, with −1/−2/−7/−8/−9 |
| Self-rated health (`sf1_good`) | Interviewer `sf1`, < 11% observed after wave e | Self-completion `scsf1` where valid, else `sf1`; 95–100% observed per wave |
| Disability | `healthlink` (adult health-record-linkage consent, wave a only) | Long-standing illness (`health` = 1) and any substantial difficulty (`disdif1`–`12`); observed for 97–99.8% of households per wave |
| Survey weights | None | Household cross-sectional weight per wave: `hhdenus_xw` (a), `hhdenub_xw` (b–e), `hhdenui_xw` (f–m), `hhdeng2_xw` (n–o). Zero-weight rows are outside that wave's population and are dropped from weighted estimates only |
| Oil use | `xpoily` > 0 | `fuelhave3` = 1 |
| Prepayment flag | Missed electricity-only households | Electricity-only households routed to `elecpay` (coverage 223,494 → 264,542 rows) |
| Rural location | Not used | `urban_dv` (2 = rural); missing values filled from the adjacent wave of the same reference person only when there is no evidence of a move |

### 4.4 The outcome: routing-aware fuel spend

Fuel spend is asked through a routed sequence. v1 read `fuelduel` = −8 (not asked, because the household does not have both gas and electricity) as missing and dropped the household. That removed almost every off-gas-grid household, including 13,463 of 15,968 Northern Ireland oil users. v1 also set item non-response on the separate amounts to £0. Table 3-4 gives the routing and the v2 treatment.

| Variable | Asked if | Content | v2 treatment |
|---|---|---|---|
| fuelhave1–4 | all households | fuels used (electricity, gas, oil, other) | defines which amounts are required |
| fuelduel | electricity AND gas used | 1 one bill / 2 separate | −8 = not dual-fuel (not missing); DK/refused → separate amounts asked |
| xpduely | fuelduel = 1 | annual combined gas+electricity £ | −1/−2/−9 = item non-response → spend missing |
| xpgasy, xpelecy | fuelduel = 2 or DK/refused, or single-fuel household | annual £ | −8 = fuel not used (structural 0); −1/−2/−9 → missing |
| xpoily | oil used (fuelhave3 = 1) | annual £ | −8 = not used (0); non-response → missing |
| xpsfly | other fuel used (fuelhave4 = 1) | annual £ | −8 = not used (0); non-response → missing |

*Table 3-4. Fuel-expenditure routing and code treatment. Every code is listed in `outputs_v2/audit_fuel_codes.csv` (171 rows).*

**Spend (amendment A1).** For electricity and gas households: `xpduely` if `fuelduel` = 1, otherwise `xpgasy` + `xpelecy`. For electricity-only households: `xpelecy`. Plus `xpoily` if oil is used and `xpsfly` if another fuel is used. Item non-response on any required amount makes spend missing (complete-case). Households that do not report electricity are excluded from the primary sample (S2 adds them back). **Ratio:** spend ÷ (12 × `fihhmnnet1_dv`). The ratio is missing if annual income is below £1,200 and capped at 1.0 (2,106 capped rows kept). `high_fuel_vulnerable` = ratio ≥ 0.10.

| Step | Description | Change / total | Remaining |
|---|---|---|---|
| 0 | All UKHLS household-wave rows (waves a-o) | 339,201 |  |
| 1 | Fuel-use module nonresponse (fuelhave* < 0) | -1,326 | 337,875 |
| 2 | No fuel reported (fuelhave96 / none mentioned) | -1,182 | 336,693 |
| 3 | Electricity not reported: gas only [S2 adds back] | -4,278 | 332,415 |
| 4 | Electricity not reported: oil/other only [S2 adds back] | -4,003 | 328,412 |
| 5a | Item nonresponse, first missing amount = xpduely [S1 adds back as 0] | -16,684 | 311,728 |
| 5b | Item nonresponse, first missing amount = xpgasy [S1 adds back as 0] | -15,619 | 296,109 |
| 5c | Item nonresponse, first missing amount = xpelecy [S1 adds back as 0] | -6,284 | 289,825 |
| 5d | Item nonresponse, first missing amount = xpoily [S1 adds back as 0] | -256 | 289,569 |
| 5e | Item nonresponse, first missing amount = xpsfly [S1 adds back as 0] | -221 | 289,348 |
| 6 | Household net income missing (sentinel code) | -13 | 289,335 |
| 7 | Annual net income < £1,200 guard (never logged in v1) | -2,433 | 286,902 |
| = | Primary analytical n (fuel_to_income_ratio non-missing) | 286,902 |  |
| info | of which ratio capped at 1.0 (kept, not excluded) | 2,106 |  |
| info | v1 analytical n for comparison | 255,324 |  |
| info | S1 (lower-bound) analytical n | 325,190 |  |
| info | S2 (+elec-not-reported) analytical n | 294,310 |  |
| v1.0 | All rows | 339,201 |  |
| v1.1 | fuelduel = -8 inapplicable (not dual-fuel) -> dropped | -61,392 |  |
| v1.2 | fuelduel DK/refused/missing -> dropped | -3,691 |  |
| v1.3 | fuelduel = 1 and xpduely nonresponse -> dropped | -16,684 |  |
| v1.4 | = rows with v1 spend | 257,434 |  |
| v1.5 | income missing | -10 |  |
| v1.6 | income < £1,200 guard (not logged in v1) | -2,100 |  |
| v1.7 | = v1 analytical n | 255,324 |  |

*Table 3-3. Sample flow from 339,201 household-waves to the primary analytical n of 286,902, with the S1, S2 and v1 flows.*

| Wave | Fieldwork | Household-waves | Analytical n, primary | Analytical n, S1 lower bound | Analytical n, v1 rule |
|---|---|---|---|---|---|
| a | 2009–2011 | 30,169 | 25,649 | 28,881 | 23,440 |
| b | 2010–2012 | 30,484 | 26,936 | 29,307 | 23,609 |
| c | 2011–2013 | 27,751 | 24,765 | 26,945 | 21,694 |
| d | 2012–2014 | 25,817 | 23,260 | 25,180 | 20,382 |
| e | 2013–2015 | 24,325 | 21,970 | 23,653 | 19,244 |
| f | 2014–2016 | 24,454 | 20,625 | 23,458 | 18,732 |
| g | 2015–2017 | 23,033 | 19,814 | 22,205 | 17,847 |
| h | 2016–2018 | 21,746 | 18,818 | 20,938 | 16,776 |
| i | 2017–2019 | 20,048 | 17,004 | 19,298 | 15,149 |
| j | 2018–2020 | 19,252 | 16,171 | 18,473 | 14,282 |
| k | 2019–2021 | 18,139 | 14,884 | 17,268 | 13,167 |
| l | 2020–2022 | 16,856 | 13,431 | 15,912 | 11,900 |
| m | 2021–2023 | 16,156 | 12,478 | 15,152 | 11,168 |
| n | 2022–2024 | 21,385 | 16,411 | 20,072 | 14,752 |
| o | 2023–2025 | 19,586 | 14,686 | 18,448 | 13,182 |
| total |  | 339,201 | 286,902 | 325,190 | 255,324 |

*Table 3-2. Household-waves and analytical n by wave. Each wave's fieldwork spans two to three calendar years.*

| Gap | Explanation | n | Detail |
|---|---|---|---|
| status_sum_off_by_one | Stage 1 status counts omitted the A2 rows (v1 kept, A1 kept, spend differs): fuelduel=2 with gas not reported, so v1 adds xpgasy while A1 counts electricity only. No row-level detail is written out. | <10 |  |
| v1_spend_rows_minus_v1_analytical_n | Rows with a v1 spend (same + zero-filled + A2) minus v1 analytical n: removed by the income step inside compute_fuel_to_income. | 2110 | income missing=10; income<£1,200=2100 |
| a1_spend_rows_minus_a1_analytical_n | Same gap for the A1 primary outcome. | 2446 | income missing=13; income<£1,200=2433 |

*Gap explanations (`outputs_v2/audit/gap_explanations.csv`): the income step removes 2,446 rows with spend under the v2 rule (2,110 under v1).*

| Wave | NI households | NI oil households | Oil households dropped by v1 `fuelduel` rule | …of which in v2 primary sample | Oil households kept by v1 |
|---|---|---|---|---|---|
| a | 1,292 | 1,010 | 834 | 678 | 176 |
| b | 2,227 | 1,776 | 1,436 | 1,234 | 340 |
| c | 2,054 | 1,637 | 1,322 | 1,139 | 315 |
| d | 1,831 | 1,435 | 1,162 | 1,025 | 273 |
| e | 1,666 | 1,268 | 1,013 | 895 | 255 |
| f | 1,510 | 1,159 | 988 | 591 | 169 |
| g | 1,434 | 1,070 | 939 | 556 | 131 |
| h | 1,382 | 1,005 | 871 | 578 | 134 |
| i | 1,316 | 955 | 824 | 644 | 131 |
| j | 1,245 | 899 | 782 | 609 | 117 |
| k | 1,154 | 814 | 709 | 537 | 105 |
| l | 1,043 | 723 | 629 | 441 | 94 |
| m | 1,041 | 708 | 624 | 415 | 84 |
| n | 1,179 | 767 | 679 | 449 | 88 |
| o | 1,112 | 742 | 651 | 428 | 91 |
| all | 21,486 | 15,968 | 13,463 | 10,219 | 2,503 |

*Northern Ireland oil households dropped by the v1 `fuelduel` rule, by wave (`outputs_v2/audit/ni_oil_lost_by_wave.csv`).*

| Level | Group | v1 n | v1 % | Primary n | Primary % | S1 n | S1 % | S2 n | S2 % |
|---|---|---|---|---|---|---|---|---|---|
| all | UK | 255,324 | 7.9 | 286,902 | 9.0 | 325,190 | 8.0 | 294,310 | 8.9 |
| region | East Midlands | 19,474 | 8.0 | 21,109 | 9.0 | 23,725 | 8.1 | 21,312 | 9.0 |
| region | East of England | 21,253 | 6.5 | 24,400 | 7.1 | 27,736 | 6.3 | 24,741 | 7.0 |
| region | London | 32,783 | 6.9 | 33,478 | 7.2 | 39,797 | 6.1 | 33,724 | 7.1 |
| region | Missing region | 97 | 12.4 | 120 | 15.8 | 153 | 12.4 | 122 | 15.6 |
| region | North East | 10,642 | 8.5 | 11,086 | 8.8 | 12,359 | 8.0 | 11,190 | 8.8 |
| region | North West | 28,712 | 8.9 | 29,474 | 9.3 | 33,441 | 8.3 | 29,766 | 9.3 |
| region | Northern Ireland | 6,172 | 15.6 | 16,646 | 18.2 | 17,550 | 17.4 | 20,143 | 15.9 |
| region | Scotland | 22,340 | 8.4 | 25,951 | 9.7 | 29,040 | 8.7 | 26,891 | 9.6 |
| region | South East | 31,465 | 6.0 | 34,004 | 6.3 | 39,241 | 5.5 | 34,394 | 6.3 |
| region | South West | 19,699 | 6.1 | 23,663 | 7.1 | 26,545 | 6.4 | 24,040 | 7.0 |
| region | Wales | 17,366 | 9.1 | 19,278 | 10.1 | 21,685 | 9.1 | 19,894 | 9.9 |
| region | West Midlands | 22,230 | 9.8 | 23,786 | 10.3 | 26,822 | 9.2 | 24,008 | 10.3 |
| region | Yorkshire and the Humber | 23,091 | 8.4 | 23,907 | 8.9 | 27,096 | 7.9 | 24,085 | 8.8 |
| wave | a | 23,440 | 11.6 | 25,649 | 12.8 | 28,881 | 11.5 | 26,199 | 12.7 |
| wave | b | 23,609 | 8.7 | 26,936 | 10.0 | 29,307 | 9.2 | 27,431 | 9.9 |
| wave | c | 21,694 | 8.4 | 24,765 | 9.7 | 26,945 | 9.0 | 25,199 | 9.6 |
| wave | d | 20,382 | 8.6 | 23,260 | 9.9 | 25,180 | 9.2 | 23,600 | 9.9 |
| wave | e | 19,244 | 9.0 | 21,970 | 10.0 | 23,653 | 9.3 | 22,282 | 10.0 |
| wave | f | 18,732 | 7.5 | 20,625 | 8.5 | 23,458 | 7.6 | 21,270 | 8.4 |
| wave | g | 17,847 | 6.4 | 19,814 | 7.1 | 22,205 | 6.4 | 20,363 | 7.0 |
| wave | h | 16,776 | 5.9 | 18,818 | 6.4 | 20,938 | 5.8 | 19,325 | 6.3 |
| wave | i | 15,149 | 5.4 | 17,004 | 6.1 | 19,298 | 5.4 | 17,399 | 6.0 |
| wave | j | 14,282 | 5.8 | 16,171 | 6.6 | 18,473 | 5.8 | 16,586 | 6.5 |
| wave | k | 13,167 | 5.3 | 14,884 | 6.2 | 17,268 | 5.4 | 15,353 | 6.2 |
| wave | l | 11,900 | 5.1 | 13,431 | 5.9 | 15,912 | 5.1 | 13,958 | 5.8 |
| wave | m | 11,168 | 6.3 | 12,478 | 7.2 | 15,152 | 5.9 | 13,036 | 7.0 |
| wave | n | 14,752 | 10.8 | 16,411 | 12.0 | 20,072 | 9.9 | 17,044 | 11.8 |
| wave | o | 13,182 | 10.6 | 14,686 | 11.9 | 18,448 | 9.5 | 15,265 | 11.7 |

*Indicative unweighted prevalence under each outcome definition (`outputs_v2/audit/indicative_prevalence.csv`). The weighted trend used in the thesis is in [`02_findings_report.md`](02_findings_report.md) §4.1.*

| Group | n | n (waves c–o) | Renting (%) | Owner (%) | Gas in rent (% of asked) | Elec. in rent (% of asked) |
|---|---|---|---|---|---|---|
| gas_only | 4,278 | 3,595 | 28.8 | 70.3 | 1.3 |  |
| oil_or_other_only | 4,003 | 3,526 | 20.1 | 79.3 |  |  |
| primary_in_scope (reference) | 289,348 | 236,196 | 30.0 | 69.4 | 0.4 | 0.5 |

*Households that do not report electricity (`outputs_v2/audit/elec_not_reported_rent_check.csv`). Both groups are mostly owner-occupiers, so their reported spend is probably incomplete rather than included in rent. They are excluded from the primary sample and added back in S2.*

### 4.5 Missing fuel spend

Spend is missing for 39,064 of the 328,412 in-scope household-waves (11.9%). The share rises over the panel (7–8% in waves b–e, 20.4% in wave o) and is much higher in telephone interviews.

![Figure 3-3. Missing spend by mode and wave](../outputs_v2/thesis_assets_v2/figures/fig3-3_missing_spend_wave_mode.png)

*Figure 3-3. Fuel-spend item non-response by interview mode and wave (households reporting electricity).*

<details><summary><b>Missing spend by interview mode and wave</b> (<code>outputs_v2/audit/missing_by_mode_wave.csv</code>)</summary>

| Wave | Mode | Observed | Missing | n | % missing | % of wave |
|---|---|---|---|---|---|---|
| a | mode not recorded | 26,037 | 3,335 | 29,372 | 11.4 | 100.0 |
| b | mode not recorded | 27,115 | 2,409 | 29,524 | 8.2 | 100.0 |
| c | CAPI (face-to-face) | 24,588 | 2,097 | 26,685 | 7.9 | 98.4 |
| c | CATI (telephone) | 299 | 122 | 421 | 29.0 | 1.6 |
| c | mode missing/other | 0 | <10 | <10 | <10 | <10 |
| d | CAPI (face-to-face) | 23,019 | 1,801 | 24,820 | 7.3 | 98.0 |
| d | CATI (telephone) | 358 | 140 | 498 | 28.1 | 2.0 |
| d | mode missing/other | <10 | <10 | 17 | <10 | 0.1 |
| e | CAPI (face-to-face) | 21,796 | 1,576 | 23,372 | 6.7 | 98.2 |
| e | CATI (telephone) | 287 | 116 | 403 | 28.8 | 1.7 |
| e | mode missing/other | 18 | 12 | 30 | 40.0 | 0.1 |
| f | CAPI (face-to-face) | 20,245 | 2,799 | 23,044 | 12.1 | 97.5 |
| f | CATI (telephone) | 448 | 94 | 542 | 17.3 | 2.3 |
| f | mode missing/other | 42 | <10 | 48 | <10 | 0.2 |
| g | CAPI (face-to-face) | 18,771 | 2,160 | 20,931 | 10.3 | 93.7 |
| g | CATI (telephone) | 257 | 59 | 316 | 18.7 | 1.4 |
| g | CAWI (web) | 878 | 223 | 1,101 | 20.3 | 4.9 |
| h | CAPI (face-to-face) | 13,181 | 1,376 | 14,557 | 9.5 | 69.0 |
| h | CATI (telephone) | 178 | 37 | 215 | 17.2 | 1.0 |
| h | CAWI (web) | 5,574 | 760 | 6,334 | 12.0 | 30.0 |
| i | CAPI (face-to-face) | 8,396 | 1,122 | 9,518 | 11.8 | 48.9 |
| i | CATI (telephone) | 55 | 31 | 86 | 36.0 | 0.4 |
| i | CAWI (web) | 8,688 | 1,181 | 9,869 | 12.0 | 50.7 |
| j | CAPI (face-to-face) | 7,024 | 907 | 7,931 | 11.4 | 42.6 |
| j | CATI (telephone) | 54 | 21 | 75 | 28.0 | 0.4 |
| j | CAWI (web) | 9,220 | 1,409 | 10,629 | 13.3 | 57.0 |
| k | CAPI (face-to-face) | 3,869 | 493 | 4,362 | 11.3 | 25.0 |
| k | CATI (telephone) | 1,244 | 300 | 1,544 | 19.4 | 8.8 |
| k | CAWI (web) | 9,925 | 1,646 | 11,571 | 14.2 | 66.2 |
| l | CAPI (face-to-face) | 405 | 58 | 463 | 12.5 | 2.9 |
| l | CATI (telephone) | 2,454 | 513 | 2,967 | 17.3 | 18.4 |
| l | CAWI (web) | 10,713 | 1,956 | 12,669 | 15.4 | 78.7 |
| m | CAPI (face-to-face) | 484 | 132 | 616 | 21.4 | 4.0 |
| m | CATI (telephone) | 1,388 | 322 | 1,710 | 18.8 | 11.1 |
| m | CAWI (web) | 10,771 | 2,279 | 13,050 | 17.5 | 84.9 |
| n | CAPI (face-to-face) | 2,610 | 628 | 3,238 | 19.4 | 15.9 |
| n | CATI (telephone) | 606 | 179 | 785 | 22.8 | 3.8 |
| n | CAWI (web) | 13,467 | 2,929 | 16,396 | 17.9 | 80.3 |
| o | CAPI (face-to-face) | 1,688 | 322 | 2,010 | 16.0 | 10.8 |
| o | CATI (telephone) | 300 | 111 | 411 | 27.0 | 2.2 |
| o | CAWI (web) | 12,888 | 3,387 | 16,275 | 20.8 | 87.1 |

</details>

<details><summary><b>Missing vs observed spend by region, tenure, income quintile and wave</b> (<code>outputs_v2/audit/missing_vs_observed_spend.csv</code>)</summary>

| Dimension | Group | Missing | Observed | % missing in group |
|---|---|---|---|---|
| region | East Midlands | 2,684 | 21,273 | 11.2 |
| region | East of England | 3,377 | 24,606 | 12.1 |
| region | London | 6,492 | 33,976 | 16.0 |
| region | Missing region | 33 | 137 | 19.4 |
| region | North East | 1,290 | 11,166 | 10.4 |
| region | North West | 4,064 | 29,715 | 12.0 |
| region | Northern Ireland | 919 | 16,762 | 5.2 |
| region | Scotland | 3,142 | 26,139 | 10.7 |
| region | South East | 5,329 | 34,285 | 13.5 |
| region | South West | 2,927 | 23,816 | 10.9 |
| region | Wales | 2,449 | 19,401 | 11.2 |
| region | West Midlands | 3,108 | 23,975 | 11.5 |
| region | Yorkshire and the Humber | 3,250 | 24,097 | 11.9 |
| tenure | Housing assoc rented | 2,416 | 21,589 | 10.1 |
| tenure | Local authority rent | 3,678 | 29,476 | 11.1 |
| tenure | Missing tenure | 568 | 980 | 36.7 |
| tenure | Other | 318 | 736 | 30.2 |
| tenure | Owned outright | 12,506 | 101,930 | 10.9 |
| tenure | Owned with mortgage | 13,140 | 98,767 | 11.7 |
| tenure | Rented from employer | 523 | 2,647 | 16.5 |
| tenure | Rented private furnished | 2,458 | 8,709 | 22.0 |
| tenure | Rented private unfurnished | 3,457 | 24,514 | 12.4 |
| income_band | Q1 | 8,980 | 56,681 | 13.7 |
| income_band | Q2 | 6,874 | 58,781 | 10.5 |
| income_band | Q3 | 6,881 | 58,771 | 10.5 |
| income_band | Q4 | 7,525 | 58,130 | 11.5 |
| income_band | Q5 | 8,795 | 56,864 | 13.4 |
| income_band | income missing | <10 | 121 | <10 |
| wave | a | 3,335 | 26,037 | 11.4 |
| wave | b | 2,409 | 27,115 | 8.2 |
| wave | c | 2,226 | 24,887 | 8.2 |
| wave | d | 1,950 | 23,385 | 7.7 |
| wave | e | 1,704 | 22,101 | 7.2 |
| wave | f | 2,899 | 20,735 | 12.3 |
| wave | g | 2,442 | 19,906 | 10.9 |
| wave | h | 2,173 | 18,933 | 10.3 |
| wave | i | 2,334 | 17,139 | 12.0 |
| wave | j | 2,337 | 16,298 | 12.5 |
| wave | k | 2,439 | 15,038 | 14.0 |
| wave | l | 2,527 | 13,572 | 15.7 |
| wave | m | 2,733 | 12,643 | 17.8 |
| wave | n | 3,736 | 16,683 | 18.3 |
| wave | o | 3,820 | 14,876 | 20.4 |
| all | all | 39,064 | 289,348 | 11.9 |

</details>

Missingness is highest in London (16.0%) and among private renters in furnished lets (22.0%), lowest in Northern Ireland (5.2%), and U-shaped in income (13.7% in the lowest quintile, 13.4% in the highest). S1, which counts non-response as £0, is therefore reported as a lower bound throughout.

### 4.6 Interview timing and regions

![Figure 3-2. Interview timing](../outputs_v2/thesis_assets_v2/figures/fig3-2_interview_timing.png)

*Figure 3-2. Interview timing from actual household interview dates (cells < 10 masked).*

| Wave | 2009 | 2010 | 2011 | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| a | 15,146 | 14,029 | 994 | — | — | — | — | — | — | — | — | — | — | — | — | — | — |
| b | — | 17,742 | 12,205 | 537 | — | — | — | — | — | — | — | — | — | — | — | — | — |
| c | — | — | 16,336 | 10,820 | 591 | — | — | — | — | — | — | — | — | — | — | — | — |
| d | — | — | — | 15,044 | 10,013 | 760 | — | — | — | — | — | — | — | — | — | — | — |
| e | — | — | — | — | 13,936 | 9,698 | 691 | — | — | — | — | — | — | — | — | — | — |
| f | — | — | — | — | — | 12,672 | 10,549 | 1,233 | — | — | — | — | — | — | — | — | — |
| g | — | — | — | — | — | — | 12,145 | 9,963 | 925 | — | — | — | — | — | — | — | — |
| h | — | — | — | — | — | — | — | 11,813 | 8,902 | 1,031 | — | — | — | — | — | — | — |
| i | — | — | — | — | — | — | — | — | 10,960 | 8,243 | 845 | — | — | — | — | — | — |
| j | — | — | — | — | — | — | — | — | — | 10,619 | 8,092 | 541 | — | — | — | — | — |
| k | — | — | — | — | — | — | — | — | — | — | 10,141 | 7,660 | 338 | — | — | — | — |
| l | — | — | — | — | — | — | — | — | — | — | — | 9,887 | 6,704 | 265 | — | — | — |
| m | — | — | — | — | — | — | — | — | — | — | — | — | 9,549 | 6,299 | 308 | — | — |
| n | — | — | — | — | — | — | — | — | — | — | — | — | — | 11,043 | 9,698 | 644 | — |
| o | — | — | — | — | — | — | — | — | — | — | — | — | — | — | 10,758 | 8,528 | 300 |

*Table A-13. Household-waves by wave × interview year (cells < 10 omitted).*

![Figure 3-5. Regional counts](../outputs_v2/thesis_assets_v2/figures/fig3-5_regional_counts.png)

*Figure 3-5. UKHLS household-waves by region.*

| Region | Household-waves |
|---|---|
| East Midlands | 24,339 |
| East of England | 28,545 |
| London | 41,318 |
| North East | 12,643 |
| North West | 34,392 |
| Northern Ireland | 21,486 |
| Scotland | 30,538 |
| South East | 40,315 |
| South West | 27,323 |
| Wales | 22,768 |
| West Midlands | 27,592 |
| Yorkshire and the Humber | 27,761 |

*Table A-14. Household-waves by region. Northern Ireland is over-represented relative to its population share (21,486 household-waves). Weighted estimates correct for this.*

### 4.7 Measures

| Construct | Items | Coding | Construction | Cronbach α (descriptive) |
|---|---|---|---|---|
| Outcome | high_fuel_vulnerable | annual fuel spend / (12 × monthly net income) ≥ 0.10 | routing-aware spend, complete-case; income < £1,200 excluded; ratio capped at 1 |  |
| Strain | finnow | current financial situation, 1 comfortable … 5 very difficult | household mean of adults | 0.27 (3-item composite; not used as a scale) |
| Strain | scghq1_dv | GHQ-12 Likert 0–36 (higher = more distress) | household mean |  |
| Strain | finfut_risk | financial expectations: 0 better / 0.5 same / 1 worse | household mean; always with age |  |
| Resources: OBJECT | hsrooms, hsbeds, ncars, carval, hsval | log(1+x) for £ items; z-scored | mean of z (≥ 50% observed), re-standardised | 0.60 |
| Resources: CONDITION | tenure_security, jbstat_security, bill_security | 0–1 security codings | as above | 0.34 |
| Resources: PERSONAL | sf1_good (self-rated health), health_good (no long-standing illness), qfhigh_band | higher = better | as above | 0.58 |
| Resources: ENERGY | fihhmnnet1_dv, fiyrinvinc_dv | log(1+x) | as above | 0.32 |
| Disability | health + disdif1–12 | long-standing illness and ≥ 1 substantial difficulty | household: any observed adult |  |
| Oil use | fuelhave3 | 1 = uses heating oil |  |  |
| Rural | urban_dv | 2 = rural | missing filled from adjacent wave if no move |  |
| FES | fes_magnitude_growth3, fes_delta_growth3 | sum of 3 growth z-scores (past-only moments); Delta = forecast − realised (m−1) | Dec Y−1 vintage; interviews 2010+ |  |

*Table 3-5. Measures: items, coding, construction and Cronbach's α. α is descriptive for the formative resource indices.*

The strain items are not one scale. Current financial difficulty (`finnow`) and financial expectations (`finfut_risk`) are almost uncorrelated, and `finfut_risk` tracks age:

| Metric | Value | n |
|---|---|---|
| cronbach_alpha_primary_items | 0.271 | 312,395 |
| r(finnow,finfut_risk) | 0.021 | 312,395 |
| r(finnow,scghq1_dv) | 0.348 | 312,395 |
| r(finfut_risk,scghq1_dv) | 0.102 | 312,395 |
| coverage_financial_strain_score | 0.991 | 336,282 |
| coverage_financial_strain_score_v1 | 0.999 | 338,950 |
| coverage_financial_strain_score_lag1 | 0.769 | 260,782 |
| r(primary, v1) | 0.955 | 336,282 |
| r(primary, lag1) | 0.566 | 259,408 |

*Strain structure (`outputs_v2/descriptives/strain_structure.csv`).*

| Type | Item | finnow | finfut_risk | scghq1_dv | xphsdba | dvage |
|---|---|---|---|---|---|---|
| pearson | finnow | 1.0 | 0.0233 | 0.3489 | 0.3084 | -0.2133 |
| pearson | finfut_risk | 0.0233 | 1.0 | 0.1026 | -0.0368 | 0.3095 |
| pearson | scghq1_dv | 0.3489 | 0.1026 | 1.0 | 0.1664 | -0.0804 |
| pearson | xphsdba | 0.3084 | -0.0368 | 0.1664 | 1.0 | -0.1448 |
| pearson | dvage | -0.2133 | 0.3095 | -0.0804 | -0.1448 | 1.0 |
| spearman | finnow | 1.0 | 0.0077 | 0.3129 | 0.2719 | -0.2184 |
| spearman | finfut_risk | 0.0077 | 1.0 | 0.0868 | -0.0423 | 0.3285 |
| spearman | scghq1_dv | 0.3129 | 0.0868 | 1.0 | 0.1406 | -0.0981 |
| spearman | xphsdba | 0.2719 | -0.0423 | 0.1406 | 1.0 | -0.155 |
| spearman | dvage | -0.2184 | 0.3285 | -0.0981 | -0.155 | 1.0 |

*Household-level correlations of the strain items, bill arrears (`xphsdba`) and age (`outputs_v2/descriptives/strain_item_correlations.csv`).*

### 4.8 The outcome and FES as attached to households

![Figure 3-4. Distributions of the outcome and FES](../outputs_v2/thesis_assets_v2/figures/fig3-4_distributions_outcome_fes.png)

*Figure 3-4. Distribution of the fuel-to-income ratio (primary) and of FES as attached to households, interviews 2010 onwards for FES (bins < 10 dropped).*

FES is attached from the December Y−1 forecast origin. The first origin with at least 24 months of training history is December 2009, so the 15,146 household-waves interviewed in 2009 have no FES and are excluded from FES models only. 324,055 rows carry FES, and 274,128 have both the outcome and FES.

| Interview year | Households | With FES | With outcome and FES | Mean magnitude | Mean current | Mean Delta |
|---|---|---|---|---|---|---|
| 2009 | 15,146 | 0 | 0 |  |  |  |
| 2010 | 31,775 | 31,775 | 27,594 | -2.22 | -2.43 | 0.21 |
| 2011 | 29,535 | 29,535 | 26,339 | -3.06 | -0.54 | -2.52 |
| 2012 | 26,401 | 26,401 | 23,704 | -0.48 | -0.46 | -0.03 |
| 2013 | 24,540 | 24,540 | 22,178 | -2.82 | -0.60 | -2.23 |
| 2014 | 23,130 | 23,130 | 20,267 | -0.66 | -0.39 | -0.27 |
| 2015 | 23,385 | 23,385 | 19,855 | -1.40 | -1.63 | 0.23 |
| 2016 | 23,009 | 23,009 | 19,704 | -0.79 | -1.86 | 1.07 |
| 2017 | 20,787 | 20,787 | 17,849 | -1.80 | -0.60 | -1.20 |
| 2018 | 19,893 | 19,893 | 16,830 | 0.15 | 0.49 | -0.35 |
| 2019 | 19,078 | 19,078 | 15,829 | -0.08 | 0.01 | -0.09 |
| 2020 | 18,088 | 18,088 | 14,693 | -0.67 | -1.74 | 1.07 |
| 2021 | 16,591 | 16,591 | 12,933 | -0.78 | -0.48 | -0.30 |
| 2022 | 17,607 | 17,607 | 13,533 | 2.25 | 9.38 | -7.13 |
| 2023 | 20,764 | 20,764 | 15,705 | 1.10 | 3.05 | -1.95 |
| 2024 | 9,172 | 9,172 | 6,892 | -1.21 | -3.17 | 1.97 |
| 2025 | 300 | 300 | 223 | -1.72 | -1.48 | -0.24 |

*FES coverage and means by interview year (`outputs_v2/fes_eval/fes_coverage_by_interview_year.csv`). Mean Delta is most negative in 2022 (−7.1): realised stress far exceeded the forecast.*

---

## 5. Geographic Boundary Data (`data/geo/uk_nuts1_regions.geojson`)

**Structure:** a GeoJSON `FeatureCollection` with an explicitly stated coordinate reference system (`{"type": "name", "properties": {"name": "EPSG:4326"}}` — WGS84, the standard latitude/longitude system — stated in the file itself rather than assumed by convention).

**Content:** 12 features, one per NUTS1 region, matching the UK's standard 12-region/nation breakdown used throughout this project:

| NUTS1 code | Region name | Geometry type |
|---|---|---|
| UKC | North East (England) | MultiPolygon |
| UKD | North West (England) | MultiPolygon |
| UKE | Yorkshire and The Humber | Polygon |
| UKF | East Midlands (England) | Polygon |
| UKG | West Midlands (England) | Polygon |
| UKH | East of England | MultiPolygon |
| UKI | London | Polygon |
| UKJ | South East (England) | MultiPolygon |
| UKK | South West (England) | MultiPolygon |
| UKL | Wales | MultiPolygon |
| UKM | Scotland | MultiPolygon |
| UKN | Northern Ireland | MultiPolygon |

8 of the 12 regions are `MultiPolygon` geometries (reflecting offshore islands or exclaves — e.g. Scotland's islands, South West England's islands), while 4 (Yorkshire and the Humber, East Midlands, West Midlands, London) are simple, contiguous `Polygon` geometries. Each feature additionally carries ONS-style metadata: `OBJECTID`, `nuts118cd`, `nuts118nm`, a British National Grid centroid (`bng_e`/`bng_n`), a WGS84 centroid (`long`/`lat`), computed shape area/length, and a `GlobalID` GUID — sourced from the ONS Open Geography Portal's public ArcGIS FeatureServer under the Open Government Licence v3.0.


---

## 6. External Benchmark: Joseph Rowntree Foundation, *UK Poverty 2025*

**Source.** JRF, *UK Poverty 2025* (January 2025), based on DWP Households Below Average Income. The measure is relative poverty after housing costs (AHC): equivalised household income below 60% of the median. The PDF is in the repository root (`UK Poverty 2025.pdf`). Every value was checked against it, and the page or table is recorded in `outputs_v2/jrf/jrf_metadata.csv`.

**Values used** (stated in the text or tables; values shown only in charts are not read off):

| Dimension | Categories and JRF rate (%) | Source |
|---|---|---|
| Region | North East 21, North West 25, Yorkshire and the Humber 23, East Midlands 20, West Midlands 27, East of England 18, London 24, South East 19, South West 19, Wales 21, Scotland 21, Northern Ireland 17 | Table 6, p.51 |
| Ethnicity (household head) | White 19, Pakistani 49, Bangladeshi 56, Black African 40, Black Caribbean 30, Any other Asian background 34 | p.9 and p.42 |
| Tenure | Owned outright 14, Buying with mortgage 10, Social renting 44, Private renting 35 | Table 10, p.95 |
| Disability | Disabled adults only 29, No one disabled 19 | Table 8, p.67 |
| Family type (children) | Lone parent 44, Couple with children 25 | Table 5, p.36 |
| Work status (working-age adults) | Not in work 54, In work 15 (v1 used 43 and 12, which are not in the report) | p.77 |

**Time matching.** UKHLS rates are computed for households interviewed within the matching financial-year window, from actual interview dates:

| Dimension | Window | JRF population | JRF measure | JRF period | JRF source | UKHLS window | UKHLS unit | UKHLS definition | Waves | Categories | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| region | primary | People (all ages) | Relative poverty, AHC | '2021–2023': HBAI '3-year' average of FY 2021/22 and 2022/23 only (DWP excludes 2020/21) | Table 6, p.51; exclusion of 2020/21: note p.43, Annex p.162 | interviews 2021-04 to 2023-03 | Household (gor_dv) | Region of household | k,l,m,n,o | 12 | JRF 'East' = East of England. Sensitivity window adds Apr 2020-Mar 2021. |
| region | sensitivity | People (all ages) | Relative poverty, AHC | '2021–2023': HBAI '3-year' average of FY 2021/22 and 2022/23 only (DWP excludes 2020/21) | Table 6, p.51; exclusion of 2020/21: note p.43, Annex p.162 | interviews 2020-04 to 2023-03 | Household (gor_dv) | Region of household | j,k,l,m,n,o | 12 | JRF 'East' = East of England. Sensitivity window adds Apr 2020-Mar 2021. |
| ethnicity | primary | People in households, by ethnicity of household head | Relative poverty, AHC | FY 2021/22 and 2022/23 (text says 2020/21-2022/23; note p.43: 2020/21 excluded) | p.9 and p.42 (text); Figure 13 and note, p.43; Annex p.162 | interviews 2021-04 to 2023-03 | Household (ethnicity of household reference person) | ethnicity_group of HRP | k,l,m,n,o | 6 | Only categories with a rate stated in JRF text are compared. |
| ethnicity | sensitivity | People in households, by ethnicity of household head | Relative poverty, AHC | FY 2021/22 and 2022/23 (text says 2020/21-2022/23; note p.43: 2020/21 excluded) | p.9 and p.42 (text); Figure 13 and note, p.43; Annex p.162 | interviews 2020-04 to 2023-03 | Household (ethnicity of household reference person) | ethnicity_group of HRP | j,k,l,m,n,o | 6 | Only categories with a rate stated in JRF text are compared. |
| tenure | primary | People | Relative poverty, AHC | FY 2022/23 | Table 10, p.95 | interviews 2022-04 to 2023-03 | Household (tenure_dv) | Social = LA + housing association; private incl. rented from employer; 'Other' tenure excluded | l,m,n,o | 4 |  |
| disability | primary | People, by disability mix of family | Relative poverty, AHC | FY 2022/23 | Table 8, p.67 | interviews 2022-04 to 2023-03 | Household with >=1 adult disability status observed | Disabled = health==1 and any disdif1-12; household contains a disabled adult if any observed adult is disabled | l,m,n,o | 2 | JRF 'Disabled adults only' (29) vs 'No one is disabled' (19). UKHLS does not observe child disability, so JRF's child rows (28, 36) are not compared. Directional (n=2). |
| family_type | primary | CHILDREN, by family type | Child relative poverty, AHC | FY 2022/23 | Table 5, p.36 | interviews 2022-04 to 2023-03 | Household with dependent children | family_composition_group lone parent (any size) vs couple (any size); other multi-adult excluded | l,m,n,o | 2 | Unit mismatch: JRF rate is per child, UKHLS rate per household. Directional (n=2). |
| work_status | primary | WORKING-AGE ADULTS, by household work status | Relative poverty, AHC | FY 2022/23 (latest year in report) | p.77 (text) | interviews 2022-04 to 2023-03 | Household with >=1 respondent aged 16-64 | Workless = no responding adult in paid/self-employment | l,m,n,o | 2 | Corrected from v1 (12/43). Unit mismatch: JRF per working-age adult, UKHLS per household. Directional (n=2). |

*Table 3-7. JRF benchmark metadata and matching windows. Region and ethnicity use April 2021–March 2023, because DWP excludes 2020/21 from its three-year averages (JRF note p.43; Annex p.162). April 2020–March 2023 is a sensitivity. Tenure, disability, family type and work status use FY 2022/23.*

---

## 7. Summary: Data Provenance and Licensing

| Source | Provider | Licence | Coverage used |
|---|---|---|---|
| Carbon and gas futures prices | Market data provider | As obtained; not redistributed | 2004/2005–2026 |
| Gas and electricity price indices, GDP, exchange rate | Office for National Statistics | Open Government Licence v3.0 | 1975/1988/1997–2026 |
| CPIH housing and energy aggregate | ONS | Open Government Licence v3.0 | 1988–2019 only |
| Half-hourly electricity demand | National Grid ESO | Public dataset | 2009–2024 |
| Temperature anomalies | Multi-country climate dataset (Our World in Data style) | Public dataset | 1940–2026 (UK subset) |
| UKHLS household panel | Understanding Society, UK Data Service SN 6614 | End User Licence; raw and row-level data not redistributed | Waves a–o; interviews 2009–2025 |
| UK NUTS1 boundaries | ONS Open Geography Portal | Open Government Licence v3.0 | 12 regions |
| External poverty benchmark | JRF, *UK Poverty 2025* | Published report; stated values used with citation | FY 2021/22–2022/23 (varies by table) |

Only the core series enter the v2 forecasts. Only aggregate UKHLS outputs that pass the suppression check (`scripts/suppress_small_cells.py --check`) are tracked in git.
