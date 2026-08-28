"""
data_loader.py
──────────────
Loads and parses the UK data files from data/raw/ and produces two cleaned,
standardised monthly DataFrames:

    Dataset A  (core)  — data/processed/core_energy_carbon.csv
        date | gas_growth | electricity_index | electricity_growth | carbon_growth


    Dataset B  (macro) — data/raw/macro_controls.csv
        date | inflation_growth | weather_volatility | gdp_growth

All three core series are expressed as YEAR-OVER-YEAR growth rates (%)
so they are conceptually consistent with each other:

    gas_growth         : ONS RPI "% change over 12 months – Gas"
                         → already a YoY % change; used as-is.

    electricity_growth : ONS CPI Index 04.5.1 (2015 = 100)
                         → the raw series is an INDEX LEVEL, not growth.
                         Transformation:
                           ElecGrowth_t = (Elec_t − Elec_{t-12}) / Elec_{t-12} × 100
                         This gives the same conceptual basis as gas_growth.

    carbon_log_return
      : ICE/Investing.com EUA futures closing price (EUR/tonne)
                         → the raw series is a PRICE LEVEL, not growth.
                         Transformation:
                           CarbonLogReturn_t = log(Carbon_t / Carbon_{t-12})
                         EUA Phase I launched Apr-2005; rows before the first
                         valid YoY carbon value are dropped rather than back-filled.

Macro series (Dataset B)
────────────────────────
    inflation_growth   : CPIH housing-energy index → monthly % change.
    gdp_growth         : ONS period-on-period monthly GVA growth (already %).
    weather_volatility : Monthly temperature anomaly for the UNITED KINGDOM,
                         filtered from the anomaly dataset, then expressed as
                         rolling 12-month standard deviation of the anomaly.

Date range
──────────
    TRAIN_START  = 2005-01-01
    FORECAST_END = 2017-12-01

    Internally, parsers load an extended window (from 2003) when 12-month
    lags are needed to compute YoY growth for the 2005 start of the series.
"""

from __future__ import annotations
import re
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from src.logging_utils import get_logger

log = get_logger("data_loader")
warnings.filterwarnings("ignore")

# ── Date range of interest ────────────────────────────────────────────────────
TRAIN_START  = "2005-01-01"

# Extended start for computing 12-month lags without losing 2005 data
_RAW_START   = "2003-01-01"


def _detect_raw_max_date(path: str) -> pd.Timestamp:
    """Cheaply find the last parseable 'YYYY MON' monthly date in an
    ONS-style CSV (used for gas.csv / electricity.csv / mgdp.csv)."""
    df = pd.read_csv(path)
    title_col = df.columns[0]
    mask = df[title_col].astype(str).str.match(r"^\d{4} [A-Z]{3}$")
    dates = pd.to_datetime(df.loc[mask, title_col].str.strip(), format="%Y %b")
    return dates.max()


def _detect_carbon_max_date(path: str) -> pd.Timestamp:
    df = pd.read_csv(path)
    dates = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    return dates.max().to_period("M").to_timestamp()


def _compute_forecast_end(
    gas_path: str = "data/raw/gas.csv",
    elec_path: str = "data/raw/electricity.csv",
    carbon_path: str = "data/raw/Carbon Emissions Futures Historical Data UK.csv",
) -> str:
    """
    Dynamically detect the latest month for which ALL THREE core raw
    sources (gas, electricity, carbon) have real data, instead of a
    hardcoded cutoff. The raw ONS/futures files are live-updated series
    that already extend far past 2017 -- hardcoding "2017-12-01" here
    silently truncated data/processed/core_energy_carbon.csv (and every
    downstream file) to a single forecast year's worth of history, which
    is why the UKHLS household-wave panel (2009-2024) could only get
    realised price-growth context for waves through ~2017.

    forecast_pipeline.py's own train/validate/forecast window is now ALSO
    computed dynamically (see forecast_pipeline._compute_default_window),
    directly from however far the CSVs this function controls extend -- so
    as this end date advances, the forecast window advances with it
    ("one year ahead of the latest complete year" instead of a fixed
    calendar year). Only the raw-source detection itself lives here; the
    resulting default train/validate/forecast split lives in
    forecast_pipeline.py.
    """
    try:
        gas_max = _detect_raw_max_date(gas_path)
        elec_max = _detect_raw_max_date(elec_path)
        carbon_max = _detect_carbon_max_date(carbon_path)
        common_end = min(gas_max, elec_max, carbon_max)
        end_str = common_end.strftime("%Y-%m-01")
        log.info("Detected common core-data end date: %s (gas=%s, elec=%s, carbon=%s)",
                 end_str, gas_max.date(), elec_max.date(), carbon_max.date())
        return end_str
    except Exception as e:
        log.warning("Could not auto-detect data end date (%s) -- falling back to 2017-12-01", e)
        return "2017-12-01"


FORECAST_END = _compute_forecast_end()


# ══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ══════════════════════════════════════════════════════════════════════════════

def _load_ons_monthly_raw(path: str, start: str = _RAW_START) -> pd.Series:
    """
    Parse an ONS CSV with 'YYYY MON' row labels.
    Returns a raw numeric Series from *start* to FORECAST_END.
    Column index 1 is assumed to hold the data values.
    """
    df        = pd.read_csv(path)
    title_col = df.columns[0]
    val_col   = df.columns[1]

    mask    = df[title_col].astype(str).str.match(r"^\d{4} [A-Z]{3}$")
    monthly = df[mask].copy()
    monthly["date"] = pd.to_datetime(monthly[title_col].str.strip(), format="%Y %b")
    monthly = monthly.set_index("date").sort_index()
    monthly[val_col] = pd.to_numeric(monthly[val_col], errors="coerce")

    return monthly[val_col].loc[start:FORECAST_END].ffill().bfill()


# ══════════════════════════════════════════════════════════════════════════════
# Dataset A parsers
# ══════════════════════════════════════════════════════════════════════════════

def _parse_gas_growth(path: str) -> pd.Series:
    """
    Gas: ONS RPI "% change over 12 months – Gas".
    Already a YoY growth rate — used as-is.
    """
    series = _load_ons_monthly_raw(path, start=TRAIN_START)
    series.name = "gas_growth"

    log.info(
        f"[gas_growth] {len(series)} obs | "
        f"{series.index.min().date()} to {series.index.max().date()} | "
        f"YoY % (source) | mean={series.mean():.3f}, "
        f"range=[{series.min():.3f}, {series.max():.3f}]"
    )
    return series


def _parse_electricity_growth(path: str) -> pd.Series:
    """
    Electricity: ONS CPI Index 04.5.1 – ELECTRICITY (2015 = 100).

    The raw series is an INDEX LEVEL.  It is transformed to a year-over-year
    growth rate so it is conceptually consistent with gas_growth:

        ElecGrowth_t = (Index_t − Index_{t-12}) / Index_{t-12} × 100

    An extended window (from 2003) is loaded internally so that YoY values
    are non-null from 2005-01-01 onwards.
    """
    raw = _load_ons_monthly_raw(path, start=_RAW_START)

    growth = raw.pct_change(12) * 100
    growth.name = "electricity_growth"

    # growth = growth.loc[TRAIN_START:FORECAST_END].ffill().bfill()
    growth = growth.loc[TRAIN_START:FORECAST_END]


    log.info(
        f"[electricity_growth] {len(growth)} obs | "
        f"{growth.index.min().date()} to {growth.index.max().date()} | "
        f"YoY % (pct_change 12) from CPI index | "
        f"mean={growth.mean():.3f}, range=[{growth.min():.3f}, {growth.max():.3f}]"
    )
    log.info(
        "  Transformation: CPI electricity index (2015=100) → "
        "YoY growth % = (Index_t − Index_{t-12}) / Index_{t-12} × 100"
    )
    return growth


def _parse_electricity_core(path: str) -> pd.DataFrame:
    """
    Return the electricity CPI index level plus YoY growth.

    The ONS series is CPI INDEX 04.5.1 : ELECTRICITY 2015=100.  The index level
    is kept for models that forecast the administered CPI index first and derive
    YoY growth after forecasting.
    """
    raw = _load_ons_monthly_raw(path, start=_RAW_START)
    out = pd.DataFrame(index=raw.index)
    out["electricity_index"] = raw
    out["electricity_growth"] = raw.pct_change(12) * 100
    return out.loc[TRAIN_START:FORECAST_END]


def _parse_carbon_growth(path: str) -> pd.Series:
    """
    Carbon: ICE/Investing.com EUA futures closing price (EUR/tonne).

    The raw series is a PRICE LEVEL.  It is transformed to a year-over-year
    growth rate for consistency with gas_growth and electricity_growth:

        CarbonGrowth_t = (Price_t − Price_{t-12}) / Price_{t-12} × 100

    EUA Phase I began Apr-2005.  The 12 months before the first valid YoY are
    left missing and dropped when the core dataset is assembled.
    """
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df = df.set_index("date").sort_index()

    price_str = df["Price"].astype(str).str.replace(",", "").str.strip()
    raw_price = pd.to_numeric(price_str, errors="coerce")

    # Resample to month-start frequency before YoY computation
    raw_price = raw_price.resample("MS").last().ffill()

    # growth = raw_price.pct_change(12) * 100
    # growth.name = "carbon_growth"

    growth = np.log(raw_price / raw_price.shift(12)) * 100
    growth.name = "carbon_growth"

    growth = growth.loc[TRAIN_START:FORECAST_END]
    # growth = growth.loc[TRAIN_START:FORECAST_END].ffill().bfill()

    log.info(
        f"[carbon_log_return] {len(growth)} obs | "
        f"{growth.index.min().date()} to {growth.index.max().date()} | "
        f"YoY % from EUA price | "
        f"mean={growth.mean():.3f}, range=[{growth.min():.3f}, {growth.max():.3f}]"
    )
    log.info(
        "  Transformation: EUA futures price (EUR/tonne) → "
        "  YoY log return = log(Price_t / Price_t-12) × 100 "
    )
    return growth


# ══════════════════════════════════════════════════════════════════════════════
# Dataset B parsers
# ══════════════════════════════════════════════════════════════════════════════

def _parse_inflation_growth(path: str) -> pd.Series:
    """
    Parse cpih08_188.xlsx and return monthly CPIH housing-energy growth rate.

    Layout (0-indexed rows):
        row 2: header row  — "Geography" | ... | "Jan-08" | "Feb-08" | ...
        row 3: UK data row — "United Kingdom" | ... | 85.2 | 86.1 | ...

    Transformation:
        InflationGrowth_t = (CPIH_t − CPIH_{t-1}) / CPIH_{t-1} × 100
    """
    df_raw = pd.read_excel(path, header=None)

    header_row   = df_raw.iloc[2].tolist()
    date_pattern = re.compile(r"^[A-Z][a-z]{2}-\d{2}$")
    date_col_idx = [i for i, v in enumerate(header_row) if date_pattern.match(str(v))]

    if not date_col_idx:
        raise ValueError(
            "No date columns found in CPIH xlsx — expected 'Jan-08' format in row 2"
        )

    uk_row = df_raw.iloc[3]
    dates  = pd.to_datetime(
        [str(header_row[i]) for i in date_col_idx], format="%b-%y"
    )
    values = pd.to_numeric(
        [uk_row.iloc[i] for i in date_col_idx], errors="coerce"
    )

    cpih = pd.Series(
        values,
        index=pd.DatetimeIndex(dates, freq="MS"),
        name="cpih_index",
    ).sort_index()

    full_idx = pd.date_range(TRAIN_START, FORECAST_END, freq="MS")
    cpih = cpih.reindex(full_idx).ffill()

    growth = cpih.pct_change(12) * 100
    growth.name = "inflation_growth"
    growth = growth.loc[TRAIN_START:FORECAST_END]
    # growth = growth.ffill().bfill().loc[TRAIN_START:FORECAST_END]

    log.info(
        f"[inflation_growth] {len(growth)} obs | "
        f"{growth.index.min().date()} to {growth.index.max().date()} | "
        f"mean={growth.mean():.4f}%, "
        f"range=[{growth.min():.4f}, {growth.max():.4f}]"
    )
    return growth


def _parse_gdp_growth(path: str) -> pd.Series:
    """
    Parse mgdp.csv → month-on-month GDP growth rate.
    ONS 'period on period growth' column is already stationary %.
    """
    df        = pd.read_csv(path)
    title_col = df.columns[0]

    mom_col = "Gross Value Added - Monthly (period on period growth) :CVM SA"
    yoy_col = "Gross Value Added - Monthly (period on period 1 year ago growth ) :CVM SA"

    if mom_col in df.columns:
        growth_col, growth_type = mom_col, "MoM"
    elif yoy_col in df.columns:
        growth_col, growth_type = yoy_col, "YoY"
        log.warning("GDP MoM growth column not found; falling back to YoY")
    else:
        raise ValueError("Cannot find any GDP growth column in mgdp.csv")

    mask    = df[title_col].astype(str).str.match(r"^\d{4} [A-Z]{3}$")
    monthly = df[mask].copy()
    monthly["date"] = pd.to_datetime(monthly[title_col].str.strip(), format="%Y %b")
    monthly = monthly.set_index("date").sort_index()

    series = pd.to_numeric(monthly[growth_col], errors="coerce")
    series.name = "gdp_growth"
    series = series.loc[TRAIN_START:FORECAST_END].ffill()

    log.info(
        f"[gdp_growth] {len(series)} obs | "
        f"{series.index.min().date()} to {series.index.max().date()} | "
        f"type={growth_type}, mean={series.mean():.4f}%, "
        f"range=[{series.min():.4f}, {series.max():.4f}]"
    )
    return series


def _parse_weather_volatility(
    path: str,
    target_dates: pd.DatetimeIndex,
) -> pd.Series:
    """
    Parse monthly temperature anomaly CSV, filter to United Kingdom, and
    derive weather_volatility = RollingStd(anomaly, 12 months).

    Supported file formats (tried in order)
    ────────────────────────────────────────
    Format A — long panel (one row per country-month):
        columns: country/entity, year, month, anomaly/temperature/value
        → filter rows where country ≈ "United Kingdom"

    Format B — wide (one row per country, monthly columns):
        columns: country, Jan-2005, Feb-2005, ... or 2005-01, ...
        → filter row where country ≈ "United Kingdom"

    Format C — HadUK-Grid spatial (63k cells, 12 monthly average columns):
        columns: "tas January" ... "tas December"
        → all rows are UK; compute national mean per calendar month,
          then simulate inter-annual variation via AR(1)

    UK labels recognised: "United Kingdom", "UK", "England", "Great Britain",
                           "GBR", "Britain", "Scotland" (first match wins).
    """
    df = pd.read_csv(path, low_memory=False)
    cols_map = {c.strip().lower(): c for c in df.columns}

    UK_PATTERNS = re.compile(
        r"united kingdom|great britain|\buk\b|england|scotland|wales|gbr",
        re.IGNORECASE,
    )

    anomaly_series: pd.Series | None = None

    # ── Format A: long panel ──────────────────────────────────────────────────
    country_col = next(
        (cols_map[k] for k in ["country", "entity", "region", "area", "nation",
                                "territory", "country name"]
         if k in cols_map),
        None,
    )

    year_col = next(
        (cols_map[k] for k in ["year", "yr"] if k in cols_map), None
    )
    month_col = next(
        (cols_map[k] for k in ["month", "mo", "mon"] if k in cols_map), None
    )
    date_col = next(
        (cols_map[k] for k in ["date", "time", "datetime", "day"] if k in cols_map), None
    )

    val_col = next(
        (cols_map[k] for k in [
            "temperature anomaly",           # "Temperature anomaly" (space variant)
            "anomaly", "temperature_anomaly", "temp_anomaly",
            "value", "temperature", "temp", "celsius",
        ] if k in cols_map),
        None,
    )

    if country_col is not None and val_col is not None:
        uk_mask = df[country_col].astype(str).str.contains(UK_PATTERNS, na=False)
        df_uk = df[uk_mask].copy()

        if df_uk.empty:
            log.warning("[weather] UK filter returned 0 rows; using all rows")
            df_uk = df.copy()
        else:
            log.info(f"[weather] Filtered to UK: {len(df_uk)} rows "
                     f"(country col='{country_col}')")

        if year_col and month_col:
            df_uk["_date"] = pd.to_datetime(
                df_uk[[year_col, month_col]]
                .rename(columns={year_col: "year", month_col: "month"})
                .assign(day=1)
            )
        elif date_col:
            df_uk["_date"] = pd.to_datetime(df_uk[date_col])
        else:
            log.warning("[weather] Cannot construct date from columns; "
                        "falling back to HadUK spatial approach")
            df_uk = None

        if df_uk is not None:
            df_uk = df_uk.dropna(subset=["_date"])
            df_uk = df_uk.set_index("_date").sort_index()
            raw = pd.to_numeric(df_uk[val_col], errors="coerce")
            anomaly_series = raw.resample("MS").mean()

    # ── Format B: wide with country rows ─────────────────────────────────────
    if anomaly_series is None and country_col is not None and val_col is None:
        uk_mask = df[country_col].astype(str).str.contains(UK_PATTERNS, na=False)
        df_uk = df[uk_mask].copy()
        if not df_uk.empty:
            # Remaining columns should be date-like
            date_cols = [c for c in df.columns if c != country_col]
            try:
                melted = df_uk.melt(
                    id_vars=[country_col], value_vars=date_cols,
                    var_name="_raw_date", value_name="_val",
                )
                melted["_date"] = pd.to_datetime(melted["_raw_date"],
                                                  infer_datetime_format=True,
                                                  errors="coerce")
                melted = melted.dropna(subset=["_date"]).set_index("_date").sort_index()
                raw = pd.to_numeric(melted["_val"], errors="coerce")
                anomaly_series = raw.resample("MS").mean()
                log.info("[weather] Parsed wide-format UK temperature data")
            except Exception as e:
                log.warning(f"[weather] Wide-format parsing failed: {e}")

    # ── Format C: HadUK spatial grid (63k cells × 12 monthly averages) ───────
    if anomaly_series is None:
        log.info("[weather] Using HadUK spatial grid format")
        MONTH_COLS = {
            m: f"tas {name}" for m, name in enumerate(
                ["January","February","March","April","May","June",
                 "July","August","September","October","November","December"], 1
            )
        }
        seasonal_mean: dict[int, float] = {}
        for m, col in MONTH_COLS.items():
            if col in df.columns:
                seasonal_mean[m] = float(df[col].mean())
            else:
                seasonal_mean[m] = 10.0
                log.warning(f"[weather] Column '{col}' not found; using 10.0")

        log.info(
            "[weather] Seasonal means (Jan–Dec): "
            + ", ".join(f"{seasonal_mean[m]:.1f}" for m in range(1, 13))
        )

        rng     = np.random.default_rng(42)
        n       = len(target_dates)
        phi, sigma = 0.70, 0.40
        anom    = np.zeros(n)
        for t in range(1, n):
            anom[t] = phi * anom[t - 1] + rng.normal(0, sigma)

        temperature = np.array(
            [seasonal_mean[dt.month] + anom[i] for i, dt in enumerate(target_dates)]
        )
        anomaly_series = pd.Series(
            [temperature[i] - seasonal_mean[dt.month]
             for i, dt in enumerate(target_dates)],
            index=target_dates,
        )

    # ── Align to target dates ─────────────────────────────────────────────────
    anomaly_aligned = anomaly_series.reindex(target_dates).ffill()

    # weather_volatility = rolling std of anomaly (12-month window)
    weather_vol = anomaly_aligned.rolling(window=12, min_periods=6).std()
    weather_vol.name = "weather_volatility"

    log.info(
        f"[weather_volatility] {len(weather_vol)} obs (UK-filtered) | "
        f"mean={weather_vol.mean():.4f}, "
        f"range=[{weather_vol.min():.4f}, {weather_vol.max():.4f}]"
    )
    log.info(
        "  Transformation: RollingStd(UK temperature anomaly, 12 months). "
        "Captures unusual weather instability relevant to energy stress."
    )
    return weather_vol


def _parse_gbp_eur_rate(path: str) -> pd.DataFrame:
    """
    ONS XUMAERS — Monthly average Sterling/Euro exchange rate (EUR per GBP).

    Relevance to electricity forecasting
    ─────────────────────────────────────
    EUA carbon allowances are priced in EUR; a weaker GBP raises compliance costs
    for UK generators directly. UK interconnectors (France, Netherlands) transmit
    European electricity prices in EUR — GBP depreciation inflates that import cost.
    The post-Brexit GBP fall in H2 2016 (1.38 → 1.15 EUR/GBP) preceded the
    observed 2017 electricity price surge.

    Returns monthly:
        gbp_eur_rate        : level (EUR per GBP)
        gbp_eur_yoy_change  : YoY % change (depreciation momentum)
        gbp_eur_mom_change  : MoM % change (short-term signal)
    """
    raw = _load_ons_monthly_raw(path, start=_RAW_START)

    out = pd.DataFrame(index=raw.index)
    out["gbp_eur_rate"]       = raw
    out["gbp_eur_yoy_change"] = raw.pct_change(12) * 100
    out["gbp_eur_mom_change"] = raw.pct_change(1) * 100

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.loc[TRAIN_START:FORECAST_END]

    log.info(
        f"[gbp_eur_rate] {len(out)} obs | "
        f"{out.index.min().date()} to {out.index.max().date()} | "
        f"mean={out['gbp_eur_rate'].mean():.4f}, "
        f"range=[{out['gbp_eur_rate'].min():.4f}, {out['gbp_eur_rate'].max():.4f}]"
    )
    log.info(
        "  Source: ONS XUMAERS (MRET) — average monthly EUR/GBP rate. "
        "YoY & MoM changes capture GBP depreciation relevant to EUR-priced energy inputs."
    )
    return out


def _parse_gas_futures(path: str) -> pd.DataFrame:
    """
    UK NBP Natural Gas Quarterly Futures.

    Columns:
        Date | Price | Open | High | Low | Vol. | Change %

    Returns monthly:
        gas_futures_price
        gas_futures_log_return
        gas_futures_yoy_growth
    """
    df = pd.read_csv(path)

    df["date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date").sort_index()

    price = pd.to_numeric(df["Price"], errors="coerce")
    price = price[price > 0]

    monthly_price = price.resample("MS").last().ffill()

    out = pd.DataFrame(index=monthly_price.index)
    out["gas_futures_price"] = monthly_price
    out["gas_futures_log_return"] = np.log(monthly_price / monthly_price.shift(1)) * 100
    out["gas_futures_yoy_growth"] = monthly_price.pct_change(12) * 100

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.loc[TRAIN_START:FORECAST_END]

    log.info(
        f"[gas_futures] {len(out)} obs | "
        f"{out.index.min().date()} to {out.index.max().date()}"
    )

    return out

def _parse_electricity_demand(path: str) -> pd.DataFrame:
    """
    UK electricity demand file.

    Key columns:
        settlement_date
        settlement_period
        nd
        tsd
        england_wales_demand

    Returns monthly:
        electricity_demand_mean
        electricity_demand_peak
        electricity_tsd_mean
        england_wales_demand_mean
        electricity_demand_yoy_growth
        electricity_peak_yoy_growth
        embedded_wind_generation_mean
        embedded_solar_generation_mean
        embedded_wind_capacity_mean
        embedded_solar_capacity_mean
        pump_storage_pumping_mean
        interconnector_net_flow_mean
        holiday_share
    """
    df = pd.read_csv(path)

    df["date"] = pd.to_datetime(df["settlement_date"], errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date").sort_index()

    numeric_cols = [
        "nd",
        "tsd",
        "england_wales_demand",
        "embedded_wind_generation",
        "embedded_wind_capacity",
        "embedded_solar_generation",
        "embedded_solar_capacity",
        "pump_storage_pumping",
        "is_holiday",
    ]
    interconnector_cols = [
        "ifa_flow",
        "ifa2_flow",
        "britned_flow",
        "moyle_flow",
        "east_west_flow",
        "nemo_flow",
        "nsl_flow",
        "eleclink_flow",
        "viking_flow",
        "greenlink_flow",
    ]
    for col in numeric_cols + interconnector_cols:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    monthly = pd.DataFrame(index=df.resample("MS").mean(numeric_only=True).index)

    monthly["electricity_demand_mean"] = df["nd"].resample("MS").mean()
    monthly["electricity_demand_peak"] = df["nd"].resample("MS").max()
    monthly["electricity_tsd_mean"] = df["tsd"].resample("MS").mean()
    monthly["england_wales_demand_mean"] = df["england_wales_demand"].resample("MS").mean()
    monthly["embedded_wind_generation_mean"] = (
        df["embedded_wind_generation"].resample("MS").mean()
    )
    monthly["embedded_solar_generation_mean"] = (
        df["embedded_solar_generation"].resample("MS").mean()
    )
    monthly["embedded_wind_capacity_mean"] = (
        df["embedded_wind_capacity"].resample("MS").mean()
    )
    monthly["embedded_solar_capacity_mean"] = (
        df["embedded_solar_capacity"].resample("MS").mean()
    )
    monthly["pump_storage_pumping_mean"] = (
        df["pump_storage_pumping"].resample("MS").mean()
    )
    monthly["interconnector_net_flow_mean"] = (
        df[interconnector_cols].sum(axis=1, min_count=1).resample("MS").mean()
    )
    monthly["holiday_share"] = df["is_holiday"].resample("MS").mean()

    monthly["electricity_demand_yoy_growth"] = (
        monthly["electricity_demand_mean"].pct_change(12) * 100
    )

    monthly["electricity_peak_yoy_growth"] = (
        monthly["electricity_demand_peak"].pct_change(12) * 100
    )
    for col in [
        "embedded_wind_generation_mean",
        "embedded_solar_generation_mean",
        "embedded_wind_capacity_mean",
        "embedded_solar_capacity_mean",
        "pump_storage_pumping_mean",
        "interconnector_net_flow_mean",
    ]:
        monthly[f"{col}_yoy_growth"] = monthly[col].pct_change(12) * 100

    monthly = monthly.replace([np.inf, -np.inf], np.nan)
    monthly = monthly.loc[TRAIN_START:FORECAST_END]

    log.info(
        f"[electricity_demand] {len(monthly)} obs | "
        f"{monthly.index.min().date()} to {monthly.index.max().date()}"
    )

    return monthly



# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def load_core_dataset(
    gas_path:    str = "data/raw/gas.csv",
    elec_path:   str = "data/raw/electricity.csv",
    carbon_path: str = "data/raw/Carbon Emissions Futures Historical Data UK.csv",
    save_path:   str = "data/processed/core_energy_carbon.csv",
) -> pd.DataFrame:
    """
    Parse Dataset A — all three core series returned as YoY growth rates (%).

    Series
    ──────
    gas_growth         : ONS RPI % change over 12 months (source already YoY %)
    electricity_growth : (CPI_t − CPI_{t-12}) / CPI_{t-12} × 100
    carbon_log_return  : log(EUA_t − EUA_{t-12}) / EUA_{t-12} × 100

    Range: TRAIN_START (2005-01-01) to FORECAST_END (2017-12-01)

    Returns
    -------
    pd.DataFrame  columns: date, gas_growth, electricity_index,
                           electricity_growth, carbon_growth
    """
    log.info("=" * 60)
    log.info("Loading Dataset A — core energy-carbon series (all as YoY growth %)")
    log.info("=" * 60)

    gas  = _parse_gas_growth(gas_path)
    elec = _parse_electricity_core(elec_path)
    carb = _parse_carbon_growth(carbon_path)

    target_idx = pd.date_range(TRAIN_START, FORECAST_END, freq="MS")

    df = pd.DataFrame({
        "gas_growth":         gas.reindex(target_idx),
        "electricity_index":  elec["electricity_index"].reindex(target_idx),
        "electricity_growth": elec["electricity_growth"].reindex(target_idx),
        "carbon_growth":      carb.reindex(target_idx),
    })
    missing_before = df[["gas_growth", "electricity_growth", "carbon_growth"]].isna().sum()
    if missing_before.any():
        log.warning(
            "Core target rows with unavailable source history will be dropped:\n"
            f"{missing_before[missing_before > 0]}"
        )
    df = df.dropna(subset=["gas_growth", "electricity_index", "electricity_growth", "carbon_growth"])
    df = df.reset_index().rename(columns={"index": "date"})
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    log.info(
        f"Core dataset: {len(df)} rows, "
        f"{df['date'].min().date()} to {df['date'].max().date()}"
    )
    log.info("  All series are YoY growth rates (%) — conceptually consistent.")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        log.info(f"Core dataset saved → {save_path}")

    return df


def load_macro_dataset(
    cpi_path:  str = "data/raw/cpih08_188.xlsx",
    gdp_path:  str = "data/raw/mgdp.csv",
    temp_path: str = "data/raw/monthly-temperature-anomalies.csv",
    gas_futures_path: str = "data/raw/UK NBP Natural Gas Quaterly Futures Historical Data UK.csv",
    elec_demand_path: str = "data/raw/historic_demand_2009_2024.csv",
    gbp_eur_path: str = "data/raw/series-190626.csv",
    save_path: str = "data/processed/macro_controls.csv",
) -> pd.DataFrame:
    """
    Parse Dataset B — exogenous / contextual controls.

    inflation_growth  : CPIH housing-energy monthly % change
    weather_volatility: rolling 12-month std of UK temperature anomaly
    gdp_growth        : ONS period-on-period monthly GVA growth %

    These are exogenous controls only; they do NOT appear as FES components.

    Returns
    -------
    pd.DataFrame  columns: date, inflation_growth, weather_volatility, gdp_growth
    """
    log.info("=" * 60)
    log.info("Loading Dataset B — macro controls (UK-specific)")
    log.info("=" * 60)

    inf_growth = _parse_inflation_growth(cpi_path)
    gdp_growth = _parse_gdp_growth(gdp_path)

    target_dates = pd.date_range(TRAIN_START, FORECAST_END, freq="MS")
    weather_vol  = _parse_weather_volatility(temp_path, target_dates)
    gas_futures  = _parse_gas_futures(gas_futures_path)
    elec_demand  = _parse_electricity_demand(elec_demand_path)
    gbp_eur      = _parse_gbp_eur_rate(gbp_eur_path)

    df = pd.DataFrame(index=target_dates)
    df.index.name = "date"
    df["inflation_growth"]   = inf_growth.reindex(target_dates)
    df["weather_volatility"] = weather_vol.reindex(target_dates)
    df["gdp_growth"]         = gdp_growth.reindex(target_dates)
    df = df.join(gas_futures.reindex(target_dates))
    df = df.join(elec_demand.reindex(target_dates))
    df = df.join(gbp_eur.reindex(target_dates))

    # cpih08_188.xlsx only has real data through Jan-2019, and
    # historic_demand_2009_2024.csv only through Dec-2024 -- both are the
    # weakest links among the macro sources. Now that FORECAST_END is
    # detected dynamically (often 2025/2026), inflation_growth/elec-demand
    # columns go STALE (forward-filled, not real) past those dates. This
    # is a real limitation, not hidden: macro controls are only used as
    # exogenous regressors for "macro mode" forecasting (never as FES
    # components directly), so staleness degrades but doesn't break that
    # use, and it does not affect the core gas/electricity/carbon series
    # the UKHLS panel's attach_price_context() actually consumes.
    stale_after = {"inflation_growth": pd.Timestamp("2019-01-01"),
                   "electricity_demand_mean": pd.Timestamp("2024-12-01")}
    for col, cutoff in stale_after.items():
        if col in df.columns and target_dates.max() > cutoff:
            log.warning("%s has no real data after %s -- forward-filled "
                        "(stale) for %d months beyond that.",
                        col, cutoff.date(), (target_dates.max().year - cutoff.year) * 12
                        + (target_dates.max().month - cutoff.month))

    df = df.ffill()

    lag_cols = [
        "inflation_growth",
        "weather_volatility",
        "gdp_growth",
        "gas_futures_log_return",
        "gas_futures_yoy_growth",
        "electricity_demand_yoy_growth",
        "electricity_peak_yoy_growth",
        "embedded_wind_generation_mean_yoy_growth",
        "embedded_solar_generation_mean_yoy_growth",
        "embedded_wind_capacity_mean_yoy_growth",
        "embedded_solar_capacity_mean_yoy_growth",
        "pump_storage_pumping_mean_yoy_growth",
        "interconnector_net_flow_mean_yoy_growth",
        "holiday_share",
        "gbp_eur_yoy_change",
        "gbp_eur_mom_change",
    ]
    for col in lag_cols:
        if col in df.columns:
            df[f"{col}_lag1"] = df[col].shift(1)

    # lag-12 for GBP/EUR: ~12-month regulatory transmission window for UK retail electricity prices
    if "gbp_eur_yoy_change" in df.columns:
        df["gbp_eur_yoy_change_lag12"] = df["gbp_eur_yoy_change"].shift(12)

    # Neutral first-month values avoid future back-fill while keeping exog arrays rectangular.
    neutral_cols = [c for c in df.columns if c.endswith("_lag1") or c.endswith("_lag12")]
    df[neutral_cols] = df[neutral_cols].fillna(0.0)

    df["post_2016_electricity_regime"] = (df.index >= "2016-01-01").astype(int)
    df["winter_dummy"] = df.index.month.isin([11, 12, 1, 2, 3]).astype(int)

    df = df.reset_index()

    

    log.info(
        f"Macro dataset: {len(df)} rows, "
        f"{df['date'].min().date()} to {df['date'].max().date()}"
    )
    log.info(f"  inflation_growth  : mean={df['inflation_growth'].mean():.4f}%")
    log.info(f"  weather_volatility: mean={df['weather_volatility'].mean():.4f}")
    log.info(f"  gdp_growth        : mean={df['gdp_growth'].mean():.4f}%")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        log.info(f"Macro dataset saved → {save_path}")

    return df


if __name__ == "__main__":
    from src.logging_utils import setup_logger
    setup_logger()

    core  = load_core_dataset()
    macro = load_macro_dataset()

    print("\nCore dataset (head + tail):")
    print(pd.concat([core.head(3), core.tail(3)]).to_string(index=False))

    print("\nMacro dataset (head + tail):")
    print(pd.concat([macro.head(3), macro.tail(3)]).to_string(index=False))
