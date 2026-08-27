"""
ukhls_preprocessing.py
────────────────────────
Understanding Society (UKHLS, UK Data Service Study 6614) household panel:
loading, cleaning, fuel-poverty target construction, and a single COR-style
explanatory composite. Replaces `enable_preprocessing.py` as the household
stream's data source (see project plan:
/home/mohsen/.claude/plans/linked-tinkering-moonbeam.md).

Why this replaces ENABLE
─────────────────────────
ENABLE.EU is a single UK cross-section (year 2017 only) — every household
shares one FES value, so forecasted/actual prices could only ever be
background context. UKHLS is a 15-wave panel (waves a-o, ~2009-2023): each
household's interview year differs, so real price growth genuinely varies
across rows and can be a legitimate row-level model feature.

Pipeline steps
──────────────
  1. Per wave: load hhresp (household) + indresp (individual) with
     column-selective reads (never load the full ~2000-column indresp
     files), recode UKHLS missing sentinels to NaN
  2. Aggregate individual-level items (finnow/finfut/GHQ) to household
     level (mean across responding adults)
  3. Merge household + aggregated-individual per wave; stack all waves
     into one long household-wave panel
  4. Approximate each household-wave's interview calendar year/month from
     the wave's fieldwork start year + the `month` variable
  5. Compute total annual fuel spend (handles combined vs. separate gas/
     electricity billing) and the fuel-to-income ratio
  6. Binary target: high_fuel_vulnerable = 1 if ratio >= 10% (standard UK
     fuel-poverty threshold); also a within-wave P75 relative version
  7. Build one COR-style explanatory composite (financial_strain_score) —
     a feature/context column, NOT the target
  8. Attach realised national gas/electricity/carbon growth for each
     household's interview year, then FES Magnitude (the forecast
     pipeline's forward-looking national shock) and FES Delta (Magnitude
     minus each household-wave's own realised exposure) — see
     `attach_fes_delta()` below for the full COR-theoretic rationale
  9. Export the panel CSV consumed by household_stream.py's Stage 2b/2c/3

Usage
─────
  from src.ukhls_preprocessing import run
  df = run()          # returns the full household-wave panel
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from src import paths
from src.logging_utils import get_logger
from src.ukhls_mapping import (
    WAVE_LETTERS, WAVE_FIELDWORK_START_YEAR, MISSING_CODES,
    HH_IDENTIFIER, HH_TIMING_VARS, HH_GEOGRAPHY_VARS,
    HH_FUEL_EXPENDITURE_VARS, HH_INCOME_VARS, HH_HOUSING_VARS,
    HH_HARDSHIP_VARS, HH_COPING_VARS_RECENT_ONLY, HH_COPING_AVAILABLE_WAVES,
    HH_OBJECT_VARS,
    IND_IDENTIFIER, IND_HH_LINK, IND_FINANCIAL_VARS, IND_WELLBEING_VARS,
    IND_CONDITION_VARS, IND_PERSONAL_VARS, IND_ENERGY_VARS,
    JBSTAT_SECURITY_RECODE, QFHIGH_BAND_RECODE, TENURE_SECURITY_RECODE,
    HEATCH_GOOD_RECODE, BILL_SECURITY_RECODE,
    COR_FACTOR_ITEMS,
    FUEL_POVERTY_RATIO_THRESHOLD, FUEL_POVERTY_RELATIVE_QUANTILE,
)

log = get_logger("ukhls_preprocessing")


# =============================================================================
# STEP 0 — Cheap column discovery (avoids requiring columns absent from a
# given wave, e.g. `inoutflows*` which only exist in waves m/o)
# =============================================================================

def _available_columns(path: Path) -> set[str]:
    with pd.io.stata.StataReader(path) as r:
        return set(r.variable_labels().keys())


def _wave_path(wave: str, filetype: str) -> Path:
    return paths.UKHLS_RAW_DIR / f"{wave}_{filetype}.dta"


# =============================================================================
# STEP 1 — Missing-code recoding
# =============================================================================

def _recode_missing(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Replace UKHLS negative sentinel codes with NaN in the given columns."""
    for col in cols:
        if col in df.columns:
            df[col] = df[col].replace(MISSING_CODES, np.nan)
    return df


# =============================================================================
# STEP 2 — Per-wave household file
# =============================================================================

def load_wave_hhresp(wave: str) -> pd.DataFrame:
    """Load one wave's hhresp file, selecting only variables this project
    needs (plus whichever of those are actually present in that wave)."""
    path = _wave_path(wave, "hhresp")
    if not path.exists():
        raise FileNotFoundError(f"UKHLS hhresp not found for wave {wave}: {path}")

    wanted_bare = (
        [HH_IDENTIFIER] + HH_TIMING_VARS + HH_GEOGRAPHY_VARS
        + HH_FUEL_EXPENDITURE_VARS + HH_INCOME_VARS + HH_HOUSING_VARS
        + HH_HARDSHIP_VARS + HH_COPING_VARS_RECENT_ONLY + HH_OBJECT_VARS
    )
    prefixed = [f"{wave}_{c}" for c in wanted_bare]
    available = _available_columns(path)
    present = [c for c in prefixed if c in available]
    missing = sorted(set(prefixed) - set(present))
    if missing:
        log.debug("wave %s hhresp: %d requested vars absent (%s)",
                   wave, len(missing), missing[:5])

    df = pd.read_stata(path, columns=present, convert_categoricals=False)
    df = df.rename(columns={c: c[len(wave) + 1:] for c in present})
    df["wave"] = wave

    numeric_cols = [c for c in df.columns if c not in ("wave",)]
    df = _recode_missing(df, numeric_cols)

    if "tenure_dv" in df.columns:
        df["tenure_security"] = df["tenure_dv"].map(TENURE_SECURITY_RECODE)
    if "heatch" in df.columns:
        df["heatch_good"] = df["heatch"].map(HEATCH_GOOD_RECODE)
    if "xphsdba" in df.columns:
        df["bill_security"] = df["xphsdba"].map(BILL_SECURITY_RECODE)
    return df


# =============================================================================
# STEP 3 — Per-wave individual file, aggregated to household level
# =============================================================================

def load_wave_indresp_aggregated(wave: str) -> pd.DataFrame:
    """Load one wave's indresp file (column-selective), aggregate person-
    level items to household level (mean across responding adults)."""
    path = _wave_path(wave, "indresp")
    if not path.exists():
        raise FileNotFoundError(f"UKHLS indresp not found for wave {wave}: {path}")

    ind_vars = (
        IND_FINANCIAL_VARS + IND_WELLBEING_VARS
        + IND_CONDITION_VARS + IND_PERSONAL_VARS + IND_ENERGY_VARS
    )
    prefixed = [f"{wave}_{c}" for c in ind_vars] + [f"{wave}_{IND_HH_LINK}"]
    available = _available_columns(path)
    present = [c for c in prefixed if c in available]
    if f"{wave}_{IND_HH_LINK}" not in present:
        raise ValueError(f"wave {wave} indresp missing household link column")

    read_cols = [IND_IDENTIFIER] + present
    read_cols = [c for c in dict.fromkeys(read_cols) if c in available or c == IND_IDENTIFIER]

    df = pd.read_stata(path, columns=read_cols, convert_categoricals=False)
    df = df.rename(columns={c: c[len(wave) + 1:] for c in present})
    df = _recode_missing(df, ind_vars)

    if "finfut" in df.columns:
        df["finfut_risk"] = df["finfut"].map({1: 0.0, 3: 0.5, 2: 1.0})
    if "jbstat" in df.columns:
        df["jbstat_security"] = df["jbstat"].map(JBSTAT_SECURITY_RECODE)
    if "health" in df.columns:
        # health: 1=has long-standing illness/disability, 2=no -> reverse
        # so higher=better personal resource (no illness=1, illness=0).
        df["health_good"] = df["health"].map({1: 0.0, 2: 1.0})
    if "sf1" in df.columns:
        # sf1: 1=excellent...5=poor -> reverse to higher=better, [0,1] scale.
        df["sf1_good"] = (5.0 - df["sf1"]) / 4.0
    if "qfhigh_dv" in df.columns:
        df["qfhigh_band"] = df["qfhigh_dv"].map(QFHIGH_BAND_RECODE)

    # Recoded categorical items (finfut, jbstat, health, sf1, qfhigh_dv)
    # are replaced by their derived ordinal/reversed columns for
    # aggregation; the raw categorical codes are not meaningfully mean-able.
    RECODED = {"finfut", "jbstat", "health", "sf1", "qfhigh_dv"}
    DERIVED = ["finfut_risk", "jbstat_security", "health_good", "sf1_good", "qfhigh_band"]
    agg_cols = [c for c in ind_vars if c in df.columns and c not in RECODED]
    agg_cols += [c for c in DERIVED if c in df.columns]
    agg = df.groupby(HH_IDENTIFIER)[agg_cols].mean()
    log.info("wave %s: %d respondents -> %d households (individual aggregation)",
              wave, len(df), len(agg))
    return agg.reset_index()


# =============================================================================
# STEP 4 — Build one wave's household panel rows
# =============================================================================

def build_wave_panel(wave: str) -> pd.DataFrame:
    hh = load_wave_hhresp(wave)
    ind_agg = load_wave_indresp_aggregated(wave)
    panel = hh.merge(ind_agg, on=HH_IDENTIFIER, how="left")

    start_year = WAVE_FIELDWORK_START_YEAR[wave]
    if "month" in panel.columns:
        m = panel["month"].astype("float64")
        panel["interview_year"] = np.where(
            m.notna(), start_year + ((m - 1) // 12).fillna(0), start_year
        ).astype(float)
        panel["interview_month"] = np.where(
            m.notna(), ((m - 1) % 12) + 1, np.nan
        )
    else:
        panel["interview_year"] = float(start_year)
        panel["interview_month"] = np.nan

    return panel


def build_full_panel(waves: list[str] | None = None) -> pd.DataFrame:
    waves = waves or WAVE_LETTERS
    frames = []
    for wave in waves:
        try:
            frames.append(build_wave_panel(wave))
        except FileNotFoundError as e:
            log.warning("Skipping wave %s: %s", wave, e)
    panel = pd.concat(frames, ignore_index=True, sort=False)
    log.info("Full panel: %d household-wave rows across %d waves",
              len(panel), len(frames))
    return panel


# =============================================================================
# STEP 5 — Fuel-to-income ratio
# =============================================================================

def compute_fuel_to_income(df: pd.DataFrame) -> pd.DataFrame:
    """
    Total annual fuel spend: combined gas+electricity bill (fuelduel==1)
    OR separate gas+electricity spend (fuelduel==2), plus oil/other fuel
    for off-grid heating. NaN when the billing-type question itself was
    unanswered (not the same as a genuine zero).
    """
    combined = df.get("xpduely", pd.Series(np.nan, index=df.index))
    separate = (
        df.get("xpgasy", pd.Series(0.0, index=df.index)).fillna(0)
        + df.get("xpelecy", pd.Series(0.0, index=df.index)).fillna(0)
    )
    fuelduel = df.get("fuelduel", pd.Series(np.nan, index=df.index))
    elec_gas_spend = pd.Series(
        np.where(fuelduel == 1, combined, np.where(fuelduel == 2, separate, np.nan)),
        index=df.index,
    )

    oil = df.get("xpoily", pd.Series(0.0, index=df.index)).fillna(0)
    other = df.get("xpsfly", pd.Series(0.0, index=df.index)).fillna(0)
    total_fuel_spend = elec_gas_spend + oil + other
    total_fuel_spend[elec_gas_spend.isna()] = np.nan
    df["total_fuel_spend"] = total_fuel_spend

    annual_net_income = df.get("fihhmnnet1_dv", pd.Series(np.nan, index=df.index)) * 12
    ratio = total_fuel_spend / annual_net_income

    # Guard against implausible near-zero-income denominators: a handful of
    # rows report annual net income under GBP 1,200 (GBP 100/month) --
    # almost certainly a transient reporting artifact (e.g. a month with
    # unreported benefit income), not a real sustained income level. Left
    # unguarded, dividing by these produces ratios up to 2,778x income,
    # which silently destroys Pearson-correlation-based validation
    # (discovered via Stage 3's own validation step) even though the
    # binary high_fuel_vulnerable threshold below is unaffected by it.
    _MIN_PLAUSIBLE_ANNUAL_INCOME = 1200.0
    ratio[annual_net_income < _MIN_PLAUSIBLE_ANNUAL_INCOME] = np.nan

    # Hard sanity cap: spending >100% of net income on fuel alone for a
    # full year, while not physically impossible, is implausible at scale
    # and affects only ~0.27% of rows -- winsorized rather than dropped so
    # these households (real, severe hardship in most cases) stay in the
    # sample as "very high" rather than becoming absurd outliers.
    n_capped = int((ratio > 1.0).sum())
    if n_capped:
        log.info("fuel_to_income_ratio: capping %d rows (%.2f%%) at 1.0 "
                  "(uncapped max was %.1f)", n_capped,
                  100.0 * n_capped / ratio.notna().sum(), ratio.max())
    ratio = ratio.clip(upper=1.0)

    df["fuel_to_income_ratio"] = ratio

    n_valid = int(ratio.notna().sum())
    log.info("fuel_to_income_ratio: %d/%d valid (%.1f%%)",
              n_valid, len(df), 100.0 * n_valid / len(df))
    return df


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    high_fuel_vulnerable: absolute UK fuel-poverty threshold (ratio >= 10%).
    high_fuel_vulnerable_relative: within-wave P75 version, for continuity
    with the old AEV_QUANTILE convention.
    """
    ratio = df["fuel_to_income_ratio"]
    df["high_fuel_vulnerable"] = (ratio >= FUEL_POVERTY_RATIO_THRESHOLD).astype("Int64")
    df.loc[ratio.isna(), "high_fuel_vulnerable"] = pd.NA

    wave_thresh = df.groupby("wave")["fuel_to_income_ratio"].transform(
        lambda x: x.quantile(FUEL_POVERTY_RELATIVE_QUANTILE)
    )
    df["high_fuel_vulnerable_relative"] = (ratio >= wave_thresh).astype("Int64")
    df.loc[ratio.isna(), "high_fuel_vulnerable_relative"] = pd.NA

    n_high = int((df["high_fuel_vulnerable"] == 1).sum())
    n_valid = int(ratio.notna().sum())
    log.info("high_fuel_vulnerable (ratio>=%.0f%%): %d/%d (%.1f%%)",
              FUEL_POVERTY_RATIO_THRESHOLD * 100, n_high, n_valid,
              100.0 * n_high / n_valid if n_valid else 0.0)
    return df


# =============================================================================
# STEP 6 — COR-style explanatory composite (NOT the target)
# =============================================================================

def _normalize_01(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    if pd.isna(lo) or pd.isna(hi) or (hi - lo) < 1e-10:
        return pd.Series(np.nan, index=s.index, dtype=float)
    return (s - lo) / (hi - lo)


def build_financial_strain_composite(df: pd.DataFrame) -> pd.DataFrame:
    """
    One combined financial/psychological strain composite (context/feature,
    NOT the target): finnow + finfut_risk + GHQ distress + bill arrears,
    each min-max normalized across the full panel, row-mean (missing-aware).
    UKHLS lacks item batteries as rich as ENABLE's per-dimension blocks, so
    this collapses the old 4-construct COR design into a single score.
    """
    components = []
    for col in ["finnow", "finfut_risk", "scghq1_dv", "xphsdba"]:
        if col in df.columns:
            components.append(_normalize_01(df[col]))
    if not components:
        log.warning("No components available for financial_strain_score")
        df["financial_strain_score"] = np.nan
        return df
    df["financial_strain_score"] = pd.concat(components, axis=1).mean(axis=1)
    return df


# =============================================================================
# STEP 7 — Attach realised price context (Phase 1: realised only)
# =============================================================================

def attach_price_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach realised national gas/electricity/carbon growth for each
    household-wave row, from the processed core series.

    Month-level first, annual-mean fallback: `interview_month` (derived in
    `build_wave_panel` from UKHLS's own `month` fieldwork-timing variable)
    has ~100% coverage across all 15 waves, and `data/processed/
    core_energy_carbon.csv` is itself a genuinely monthly series -- so each
    household-wave row is joined directly on its own (interview_year,
    interview_month) to that specific calendar month's realised growth,
    rather than the whole year's mean smeared across every wave interviewed
    that year regardless of month. Rows whose exact month falls outside the
    core series' covered range (typically none, in practice) fall back to
    that year's annual mean, then to NaN -- the same honest-degradation
    pattern used throughout this module, logged rather than silent.
    """
    if not paths.CORE_CSV.exists():
        log.warning("Core price series not found: %s — price columns will be NaN. "
                    "Run forecast_pipeline.py's Stage 0-1 first.", paths.CORE_CSV)
        for col in ["gas_growth", "electricity_growth", "carbon_growth"]:
            df[col] = np.nan
        return df

    price_cols = [c for c in ["gas_growth", "electricity_growth", "carbon_growth"]]
    core = pd.read_csv(paths.CORE_CSV, index_col=0, parse_dates=True)
    core["year"] = core.index.year
    core["month"] = core.index.month
    price_cols = [c for c in price_cols if c in core.columns]

    monthly = core.set_index(["year", "month"])[price_cols]
    annual  = core.groupby("year")[price_cols].mean()

    min_year, max_year = int(annual.index.min()), int(annual.index.max())
    log.info("Core price series covers %d-%d (monthly resolution)", min_year, max_year)

    month_joined = df.merge(
        monthly, left_on=["interview_year", "interview_month"], right_index=True, how="left",
    )
    n_month_matched = month_joined[price_cols[0]].notna().sum() if price_cols else 0

    df = df.merge(annual, left_on="interview_year", right_index=True, how="left")
    for col in price_cols:
        df[col] = month_joined[col].combine_first(df[col])

    out_of_range = ((df["interview_year"] < min_year) | (df["interview_year"] > max_year)).sum()
    if out_of_range:
        log.warning(
            "%d household-wave rows have interview_year outside the core price "
            "series range (%d-%d) — price columns NaN for those rows. Extend "
            "data/processed/core_energy_carbon.csv's date range to cover the "
            "full panel window.",
            out_of_range, min_year, max_year,
        )
    log.info(
        "Price context: %d/%d rows matched at month resolution, %d additional rows "
        "fell back to their interview year's annual mean",
        n_month_matched, len(df),
        int(df[price_cols[0]].notna().sum() - n_month_matched) if price_cols else 0,
    )
    return df


# =============================================================================
# STEP 8 — FES Magnitude / Delta (forward-looking shock vs. realised baseline)
# =============================================================================

def attach_fes_delta(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach FES Magnitude (the forward-looking national energy-price shock)
    and FES Delta (Magnitude minus each household-wave's own realised
    exposure) -- operationalising Hobfoll's COR theory's requirement that
    FES act as an explicit exogenous shock (direct effect + interaction
    with Baseline Resource Stock in the SEM; a genuine conditioning signal
    in the CVAE) rather than the flat per-row constant `attach_price_context`
    alone would give every household in a single-year cross-section.
    Requires `attach_price_context()` to have already merged
    gas_growth/electricity_growth/carbon_growth.

    FES Magnitude (varies per household-wave, by interview_year AND
    interview_month where available)
    -----------------------------------------------------------------
    Three-level fallback, each stage logged so the active resolution is
    always visible, not just assumed:

      1. `outputs/fes/fes_rolling_monthly.csv` (from
         `forecast_pipeline.run_rolling`, which already computes FES at
         12-month resolution per as_of_year internally -- this just stops
         discarding that detail at an annual-mean step). Each household-wave
         row is joined on its own (interview_year, interview_month) against
         (as_of_year, target_month) -- UKHLS's `month` fieldwork-timing
         variable gives ~100% interview_month coverage, and energy prices
         have real within-year seasonality, so this is a materially
         different signal from a flat annual value for waves fielded in
         different months of the same year.
      2. `outputs/fes/fes_rolling_yearly.csv` (annual mean of the same
         walk-forward run) for any row that didn't match at month
         resolution (e.g. missing interview_month).
      3. A single constant (mean `fes_core` over `outputs/fes/
         fes_monthly_2017.csv`'s 12 forecast-target months) if `run_rolling`
         hasn't been executed at all yet -- the original, cruder behavior,
         kept so the pipeline still runs (with a logged warning).

    In every case, "what was forecast for [this row's target period], using
    only data available as of [this row's as_of_year]" -- a properly
    ex-ante, per-household anticipatory shock, not one fixed constant
    applied to every wave regardless of when it was fielded.

    FES Current (varies per household-wave, by interview_year AND
    interview_month where available)
    -------------------------------------------------------------
    Sum of z-scores of this household's own realised gas_growth/
    electricity_growth/carbon_growth -- month-resolution values from
    `attach_price_context` (which itself prefers the exact (interview_year,
    interview_month) match against the core series, annual-mean fallback) --
    standardized against those three series' own FULL-HISTORY MONTHLY
    mean/std (every month present in `data/processed/core_energy_carbon.csv`,
    not just the panel's own 2009-2023 window, and not annual-mean-of-means,
    which would understate month-to-month variance).

    FES Delta = FES Magnitude - FES Current
    -----------------------------------------
    Varies per household-wave through BOTH terms now (when the rolling
    table is available) -- FES Magnitude no longer needs the
    zero-variance/StandardScaler caveats that applied when it was a
    single constant, though callers should still not assume every row has
    a value: rows whose interview_year falls outside the rolling table's
    covered years get NaN (documented, same pattern as
    `attach_price_context`'s out-of-range handling).

    Two approximations, documented rather than hidden (matching this
    project's existing practice, e.g. the FIML CFI/TLI limitation flagged
    in `src.ukhls_cor_sem`):
      (a) FES Current's z-scoring window (full history) differs from the
          forecast pipeline's own train-window baseline for a given
          rolling year -- not reused directly here.
      (b) FES Magnitude sums 4 z-terms (gas/elec/carbon/uncertainty); FES
          Current sums 3 (no realised analogue of forecast-uncertainty
          exists). Delta is therefore a reasonable, honestly-approximate
          proxy for shock size, not an exact matched-scale subtraction.
    """
    price_cols = ["gas_growth", "electricity_growth", "carbon_growth"]

    # Which of fes_core/fes_macro forecast_pipeline.run_rolling selected as
    # the single best-performing variant (lowest mean RMSE vs realised FES
    # across every rolling year) -- shared by both the monthly and annual
    # branches below. Falls back to fes_core if the selection file doesn't
    # exist yet.
    fes_col = "fes_core"
    if paths.FES_VARIANT_SELECTION_FILE.exists():
        selection = pd.read_csv(paths.FES_VARIANT_SELECTION_FILE)
        chosen_row = selection[selection["chosen"]]
        if not chosen_row.empty:
            variant_to_col = {"Equal_Core": "fes_core", "Equal_Macro": "fes_macro"}
            fes_col = variant_to_col.get(chosen_row.iloc[0]["FES_variant"], "fes_core")

    if paths.FES_ROLLING_MONTHLY_FILE.exists():
        # Best resolution: each household-wave row gets the forecast for
        # ITS OWN interview month one year ahead (as_of_year=interview_year,
        # target_month=interview_month) instead of a single value shared by
        # every wave interviewed anywhere in that year. Energy prices have
        # real within-year seasonality, so this is a materially different
        # (and more honestly ex-ante) signal than the annual mean, using
        # rolling-forecast output that was already being computed and
        # previously discarded at the annual-average step.
        monthly_rolling = pd.read_csv(paths.FES_ROLLING_MONTHLY_FILE)
        mag_by_year_month = monthly_rolling.set_index(["as_of_year", "target_month"])[fes_col]
        df["fes_magnitude"] = df.set_index(["interview_year", "interview_month"]).index.map(mag_by_year_month)

        # Fall back to that interview year's annual mean (from the same
        # monthly table, or the separate annual table) for any row whose
        # exact month didn't match -- e.g. interview_month is NaN.
        if paths.FES_ROLLING_FILE.exists():
            annual_rolling = pd.read_csv(paths.FES_ROLLING_FILE)
            mag_by_year = annual_rolling.set_index("as_of_year")[fes_col]
        else:
            mag_by_year = monthly_rolling.groupby("as_of_year")[fes_col].mean()
        n_month_matched = df["fes_magnitude"].notna().sum()
        df["fes_magnitude"] = df["fes_magnitude"].combine_first(df["interview_year"].map(mag_by_year))

        log.info(
            "FES magnitude: rolling walk-forward MONTHLY table (%s, column=%s, "
            "%d/%d rows matched at (year, month) resolution, %d additional rows "
            "fell back to their interview year's annual mean, %d/%d total matched)",
            paths.FES_ROLLING_MONTHLY_FILE.name, fes_col, n_month_matched, len(df),
            int(df["fes_magnitude"].notna().sum() - n_month_matched),
            df["fes_magnitude"].notna().sum(), len(df),
        )
    elif paths.FES_ROLLING_FILE.exists():
        rolling = pd.read_csv(paths.FES_ROLLING_FILE)
        mag_by_year = rolling.set_index("as_of_year")[fes_col]
        df["fes_magnitude"] = df["interview_year"].map(mag_by_year)
        log.info(
            "FES magnitude: rolling walk-forward table (%s, column=%s, %d years covered, "
            "%d/%d household-wave rows matched)",
            paths.FES_ROLLING_FILE.name, fes_col, mag_by_year.notna().sum(),
            df["fes_magnitude"].notna().sum(), len(df),
        )
    elif paths.FES_MONTHLY_FILE.exists():
        monthly = pd.read_csv(paths.FES_MONTHLY_FILE)
        df["fes_magnitude"] = float(monthly["fes_core"].mean())
        log.warning(
            "Rolling FES table not found (%s) -- falling back to a single "
            "constant FES Magnitude from %s. Run "
            "forecast_pipeline.run_rolling() for a genuinely year-varying signal.",
            paths.FES_ROLLING_FILE, paths.FES_MONTHLY_FILE.name,
        )
    else:
        log.warning("No FES file found (rolling or single-year) -- fes_magnitude/"
                    "fes_current/fes_delta will be NaN. Run forecast_pipeline.py first.")
        df["fes_magnitude"] = np.nan
        df["fes_current"]   = np.nan
        df["fes_delta"]     = np.nan
        return df

    if not paths.CORE_CSV.exists():
        log.warning("Core price series not found: %s -- fes_current/fes_delta "
                    "will be NaN.", paths.CORE_CSV)
        df["fes_current"] = np.nan
        df["fes_delta"]   = np.nan
        return df

    core = pd.read_csv(paths.CORE_CSV, index_col=0, parse_dates=True)

    # z-scored against the core series' own full-history MONTHLY mean/std
    # (not an annual-mean-of-means) -- df[col] itself is now month-resolution
    # (attach_price_context joins by (interview_year, interview_month)), so
    # the z-scoring reference has to be at the same resolution or the scale
    # won't match (monthly values are noisier than annual averages).
    z_parts = []
    for col in price_cols:
        if col not in df.columns or col not in core.columns:
            continue
        mu    = float(core[col].mean())
        sigma = float(core[col].std()) or 1.0
        z_parts.append((df[col] - mu) / sigma)

    df["fes_current"] = (
        pd.concat(z_parts, axis=1).sum(axis=1, skipna=True, min_count=1)
        if z_parts else np.nan
    )
    df["fes_delta"] = df["fes_magnitude"] - df["fes_current"]

    log.info(
        "FES magnitude range [%.4f, %.4f] (unique values=%d) | "
        "fes_current range [%.4f, %.4f] | fes_delta range [%.4f, %.4f] "
        "(%d/%d rows valid)",
        df["fes_magnitude"].min(skipna=True), df["fes_magnitude"].max(skipna=True),
        df["fes_magnitude"].nunique(),
        df["fes_current"].min(skipna=True), df["fes_current"].max(skipna=True),
        df["fes_delta"].min(skipna=True), df["fes_delta"].max(skipna=True),
        df["fes_delta"].notna().sum(), len(df),
    )
    return df


# =============================================================================
# Main entry point
# =============================================================================

def run(waves: list[str] | None = None) -> pd.DataFrame:
    """
    Full UKHLS panel-building pipeline.

    Returns
    -------
    df : household-wave panel with fuel_to_income_ratio,
         high_fuel_vulnerable (+ relative version), financial_strain_score,
         and realised price-growth context columns.
    """
    paths.ensure_dirs()

    log.info("Building UKHLS household-wave panel (waves: %s)", waves or WAVE_LETTERS)
    df = build_full_panel(waves)

    df = compute_fuel_to_income(df)
    df = build_target(df)
    df = build_financial_strain_composite(df)
    df = attach_price_context(df)
    df = attach_fes_delta(df)

    paths.UKHLS_OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(paths.UKHLS_PANEL, index=False)
    log.info("Saved UKHLS panel: %s (%d rows, %d columns)",
              paths.UKHLS_PANEL.name, len(df), df.shape[1])

    return df


if __name__ == "__main__":
    from src.logging_utils import setup_logger
    setup_logger("energy_stress", log_file="outputs/logs/pipeline.log")
    run()
