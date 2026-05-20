"""
data_loader.py
──────────────
Loads and parses the UK data files from data/raw/ and produces two cleaned,
standardised monthly DataFrames:

    Dataset A  (core)  — data/raw/core_energy_carbon.csv
        date | gas_growth | electricity_growth | carbon_growth

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

    carbon_growth      : ICE/Investing.com EUA futures closing price (EUR/tonne)
                         → the raw series is a PRICE LEVEL, not growth.
                         Transformation:
                           CarbonGrowth_t = (Carbon_t − Carbon_{t-12}) / Carbon_{t-12} × 100
                         EUA Phase I launched Apr-2005; the first 12 months are
                         back-filled so the series aligns with 2005-01-01.

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
    FORECAST_END = 2018-12-01

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
FORECAST_END = "2018-12-01"

# Extended start for computing 12-month lags without losing 2005 data
_RAW_START   = "2003-01-01"


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

    growth = growth.loc[TRAIN_START:FORECAST_END].ffill().bfill()

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


def _parse_carbon_growth(path: str) -> pd.Series:
    """
    Carbon: ICE/Investing.com EUA futures closing price (EUR/tonne).

    The raw series is a PRICE LEVEL.  It is transformed to a year-over-year
    growth rate for consistency with gas_growth and electricity_growth:

        CarbonGrowth_t = (Price_t − Price_{t-12}) / Price_{t-12} × 100

    EUA Phase I began Apr-2005.  The 12 months before the first valid YoY
    (i.e. Apr 2005 → Mar 2006) are back-filled from the first valid value.
    """
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df = df.set_index("date").sort_index()

    price_str = df["Price"].astype(str).str.replace(",", "").str.strip()
    raw_price = pd.to_numeric(price_str, errors="coerce")

    # Resample to month-start frequency before YoY computation
    raw_price = raw_price.resample("MS").last().ffill()

    growth = raw_price.pct_change(12) * 100
    growth.name = "carbon_growth"

    growth = growth.loc[TRAIN_START:FORECAST_END].ffill().bfill()

    log.info(
        f"[carbon_growth] {len(growth)} obs | "
        f"{growth.index.min().date()} to {growth.index.max().date()} | "
        f"YoY % from EUA price | "
        f"mean={growth.mean():.3f}, range=[{growth.min():.3f}, {growth.max():.3f}]"
    )
    log.info(
        "  Transformation: EUA futures price (EUR/tonne) → "
        "YoY growth % = (Price_t − Price_{t-12}) / Price_{t-12} × 100"
    )
    return growth


# ══════════════════════════════════════════════════════════════════════════════
# Dataset B parsers
# ══════════════════════════════════════════════════════════════════════════════

def _parse_inflation_growth(path: str) -> pd.Series:
    """
    Parse cpih08_18.xlsx and return monthly CPIH housing-energy growth rate.

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
    cpih = cpih.reindex(full_idx).bfill()

    growth = cpih.pct_change(12) * 100
    growth.name = "inflation_growth"
    growth = growth.ffill().bfill().loc[TRAIN_START:FORECAST_END]

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
    series = series.loc[TRAIN_START:FORECAST_END].ffill().bfill()

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
    anomaly_aligned = anomaly_series.reindex(target_dates).ffill().bfill()

    # weather_volatility = rolling std of anomaly (12-month window)
    weather_vol = anomaly_aligned.rolling(window=12, min_periods=12).std().bfill()
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


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def load_core_dataset(
    gas_path:    str = "data/raw/gas.csv",
    elec_path:   str = "data/raw/electricity.csv",
    carbon_path: str = "data/raw/Carbon Emissions Futures Historical Data UK.csv",
    save_path:   str = "data/raw/core_energy_carbon.csv",
) -> pd.DataFrame:
    """
    Parse Dataset A — all three core series returned as YoY growth rates (%).

    Series
    ──────
    gas_growth         : ONS RPI % change over 12 months (source already YoY %)
    electricity_growth : (CPI_t − CPI_{t-12}) / CPI_{t-12} × 100
    carbon_growth      : (EUA_t − EUA_{t-12}) / EUA_{t-12} × 100

    Range: TRAIN_START (2005-01-01) to FORECAST_END (2018-12-01)

    Returns
    -------
    pd.DataFrame  columns: date, gas_growth, electricity_growth, carbon_growth
    """
    log.info("=" * 60)
    log.info("Loading Dataset A — core energy-carbon series (all as YoY growth %)")
    log.info("=" * 60)

    gas  = _parse_gas_growth(gas_path)
    elec = _parse_electricity_growth(elec_path)
    carb = _parse_carbon_growth(carbon_path)

    target_idx = pd.date_range(TRAIN_START, FORECAST_END, freq="MS")

    df = pd.DataFrame({
        "gas_growth":         gas.reindex(target_idx),
        "electricity_growth": elec.reindex(target_idx),
        "carbon_growth":      carb.reindex(target_idx),
    })
    df = df.ffill().bfill()
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
    save_path: str = "data/raw/macro_controls.csv",
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

    df = pd.DataFrame(index=target_dates)
    df.index.name = "date"
    df["inflation_growth"]   = inf_growth.reindex(target_dates)
    df["weather_volatility"] = weather_vol.reindex(target_dates)
    df["gdp_growth"]         = gdp_growth.reindex(target_dates)

    df = df.ffill().bfill().reset_index()

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
