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
background context. UKHLS is a 15-wave panel (waves a-o, ~2009-2024): each
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
    HH_IDENTIFIER, HH_LINK_VARS, HH_TIMING_VARS, HH_GEOGRAPHY_VARS,
    HH_FUEL_EXPENDITURE_VARS, FUEL_NONRESPONSE_CODES, FUEL_AMOUNT_VARS,
    HH_INCOME_VARS, HH_HOUSING_VARS, HH_HARDSHIP_VARS, HH_COPING_VARS_RECENT_ONLY, HH_COPING_AVAILABLE_WAVES,
    HH_OBJECT_VARS, HH_FAMILY_VARS, HH_EQUIVALISATION_VARS,
    IND_IDENTIFIER, IND_HH_LINK, IND_FINANCIAL_VARS, IND_WELLBEING_VARS,
    IND_CONDITION_VARS, IND_PERSONAL_VARS, IND_ENERGY_VARS,
    IND_DISABILITY_VARS, IND_ETHNICITY_VAR, ETHNICITY_GROUP_RECODE,
    IND_EMPLOYMENT_VARS, JBSTAT_EMPLOYED_CODES,
    JBSTAT_SECURITY_RECODE, QFHIGH_BAND_RECODE, TENURE_SECURITY_RECODE,
    HEATCH_GOOD_RECODE, BILL_SECURITY_RECODE,
    HHTYPE_LONE_PARENT_CODES, HHTYPE_COUPLE_WITH_CHILDREN_CODES,
    HHTYPE_OTHER_WITH_CHILDREN_CODES, LARGE_FAMILY_MIN_CHILDREN,
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
        [HH_IDENTIFIER] + HH_LINK_VARS + HH_TIMING_VARS + HH_GEOGRAPHY_VARS
        + HH_FUEL_EXPENDITURE_VARS + HH_INCOME_VARS + HH_HOUSING_VARS
        + HH_HARDSHIP_VARS + HH_COPING_VARS_RECENT_ONLY + HH_OBJECT_VARS
        + HH_FAMILY_VARS + HH_EQUIVALISATION_VARS
    )
    prefixed = [f"{wave}_{c}" for c in wanted_bare]
    available = _available_columns(path)
    present = [c for c in prefixed if c in available]
    # Household cross-sectional weight: name varies by wave (hhdenus_xw,
    # hhdenub_xw, hhdenui_xw, hhdeng2_xw) -- loaded as `hh_xw`.
    xw_cols = sorted(c for c in available if c.startswith(f"{wave}_") and c.endswith("_xw"))
    present += xw_cols[:1]
    missing = sorted(set(prefixed) - set(present))
    if missing:
        log.debug("wave %s hhresp: %d requested vars absent (%s)",
                   wave, len(missing), missing[:5])

    df = pd.read_stata(path, columns=present, convert_categoricals=False)
    df = df.rename(columns={c: c[len(wave) + 1:] for c in present})
    if xw_cols:
        df = df.rename(columns={xw_cols[0][len(wave) + 1:]: "hh_xw"})
        df["hh_xw_name"] = xw_cols[0][len(wave) + 1:]
    df["wave"] = wave

    # Keep item nonresponse distinguishable from -8 inapplicable for the fuel
    # amounts: the generic recode below turns both into NaN.
    for col in FUEL_AMOUNT_VARS:
        if col in df.columns:
            df[f"{col}_nr"] = df[col].isin(FUEL_NONRESPONSE_CODES)

    numeric_cols = [c for c in df.columns if c not in ("wave",) and not c.endswith("_nr")]
    df = _recode_missing(df, numeric_cols)

    if "tenure_dv" in df.columns:
        df["tenure_security"] = df["tenure_dv"].map(TENURE_SECURITY_RECODE)
    if "heatch" in df.columns:
        df["heatch_good"] = df["heatch"].map(HEATCH_GOOD_RECODE)
    if "xphsdba" in df.columns:
        df["bill_security"] = df["xphsdba"].map(BILL_SECURITY_RECODE)

    if "hhtype_dv" in df.columns and "nkids_dv" in df.columns:
        df["family_composition_group"] = _derive_family_composition_group(
            df["hhtype_dv"], df["nkids_dv"]
        )
        df["lone_parent"] = df["hhtype_dv"].isin(HHTYPE_LONE_PARENT_CODES).astype(float)
        df.loc[df["hhtype_dv"].isna(), "lone_parent"] = np.nan
        df["large_family"] = (df["nkids_dv"] >= LARGE_FAMILY_MIN_CHILDREN).astype(float)
        df.loc[df["nkids_dv"].isna(), "large_family"] = np.nan

    # Prepayment-meter flag -- same fuelduel branching compute_fuel_to_income
    # uses (combined-bill households answer duelpay, separate-bill
    # households answer elecpay). A well-documented UK fuel-poverty proxy
    # for self-disconnection/rationing (see README's rationing-evidence
    # section) -- available across ALL 15 waves, unlike inoutflows12
    # (waves m/o only).
    if "fuelduel" in df.columns:
        duelpay = df.get("duelpay", pd.Series(np.nan, index=df.index))
        elecpay = df.get("elecpay", pd.Series(np.nan, index=df.index))
        fuelduel = df["fuelduel"]
        # duelpay==4 / elecpay==4 collapse a NaN (unanswered) sub-question to
        # False rather than propagating missingness, unlike compute_fuel_to_income's
        # analogous branching above -- guard each comparison so an unanswered
        # duelpay/elecpay stays NaN instead of silently reading as "not prepay".
        duelpay_flag = np.where(duelpay.isna(), np.nan, (duelpay == 4).astype(float))
        elecpay_flag = np.where(elecpay.isna(), np.nan, (elecpay == 4).astype(float))
        # Electricity-only households skip fuelduel (-8) and answer elecpay
        # directly -- same routing as the A1 outcome.
        elec_only = (df["fuelhave1"] == 1) & (df["fuelhave2"] == 0)
        prepay = pd.Series(
            np.where(fuelduel == 1, duelpay_flag,
                     np.where((fuelduel == 2) | elec_only, elecpay_flag, np.nan)),
            index=df.index,
        )
        df["prepayment_meter"] = prepay.astype(float)

    return df


def _derive_family_composition_group(hhtype_dv: pd.Series, nkids_dv: pd.Series) -> pd.Series:
    """hhtype_dv + nkids_dv -> JRF-comparable family-type group (UK Poverty
    2025, Table 5, p.36: family type x large-family cross-cut) -- see
    src.ukhls_mapping's HHTYPE_*_CODES comment for the code-group rationale.
    Explicit mask assignment (not np.select) -- np.select's choicelist
    (strings) and a np.nan default don't share a common numpy dtype and
    raise a TypeError under recent numpy, so the "no match" case is left as
    the Series' own NaN default instead of passed through np.select."""
    out = pd.Series(np.nan, index=hhtype_dv.index, dtype=object)
    out[nkids_dv.fillna(0) == 0] = "No children"
    out[hhtype_dv.isin(HHTYPE_OTHER_WITH_CHILDREN_CODES)] = "Other multi-adult, with children"
    out[hhtype_dv.isin(HHTYPE_COUPLE_WITH_CHILDREN_CODES)] = "Couple, 1-2 children"
    out[hhtype_dv.isin(HHTYPE_COUPLE_WITH_CHILDREN_CODES) & (nkids_dv >= LARGE_FAMILY_MIN_CHILDREN)] = "Couple, 3+ children"
    out[hhtype_dv.isin(HHTYPE_LONE_PARENT_CODES)] = "Lone parent, 1-2 children"
    out[hhtype_dv.isin(HHTYPE_LONE_PARENT_CODES) & (nkids_dv >= LARGE_FAMILY_MIN_CHILDREN)] = "Lone parent, 3+ children"
    out[hhtype_dv.isna()] = np.nan
    return out


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
        + IND_DISABILITY_VARS + IND_EMPLOYMENT_VARS
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
    disdif = [f"disdif{i}" for i in range(1, 13) if f"disdif{i}" in df.columns]
    if "health" in df.columns and disdif:
        # Equality-Act-style disability (JRF/FRS definition): a long-standing
        # illness or disability (health==1) AND at least one substantial
        # difficulty (any disdif1-12 mentioned). health==2 -> not disabled.
        # health==1 with every difficulty item observed and none mentioned
        # -> not disabled; otherwise unknown. Consistent across waves even
        # though disdif is asked of everyone in later waves but only of
        # health==1 respondents in wave a. Reversed to disability_free
        # (higher=better) to match health_good/sf1_good.
        any_difficulty = (df[disdif] == 1).any(axis=1)
        all_observed = df[disdif].notna().all(axis=1)
        disabled = np.select(
            [df["health"] == 2,
             (df["health"] == 1) & any_difficulty,
             (df["health"] == 1) & all_observed],
            [0.0, 1.0, 0.0], default=np.nan)
        df["disability_free"] = 1.0 - disabled

    # Recoded categorical items (finfut, jbstat, health, sf1, qfhigh_dv,
    # healthlink) are replaced by their derived ordinal/reversed columns
    # for aggregation; the raw categorical codes are not meaningfully
    # mean-able. jbft_dv/jbsemp/jbterm1 are likewise per-person category
    # codes (full-time/part-time, employee/self-employed, permanent/
    # temporary) -- averaging them produces a numerically meaningless code
    # (e.g. 1.5 for a mixed full-/part-time household); jbft_dv/jbsemp are
    # separately MAX-aggregated into has_fulltime_worker/has_selfemployed_worker
    # below, so they don't need a mean-aggregated column at all. jbhrs
    # (hours worked) stays mean-aggregated -- it's a genuine continuous value.
    RECODED = {"finfut", "jbstat", "health", "sf1", "qfhigh_dv"} | set(IND_DISABILITY_VARS)
    NOT_MEANABLE_CATEGORICAL = {"jbft_dv", "jbsemp", "jbterm1"}
    DERIVED = ["finfut_risk", "jbstat_security", "health_good", "sf1_good",
               "qfhigh_band", "disability_free"]
    agg_cols = [c for c in ind_vars
                if c in df.columns and c not in RECODED and c not in NOT_MEANABLE_CATEGORICAL]
    agg_cols += [c for c in DERIVED if c in df.columns]
    agg = df.groupby(HH_IDENTIFIER)[agg_cols].mean()

    # Employment status/hours (jbft_dv/jbsemp/jbstat) are per-person
    # categorical flags -- meaningful at household level as "does ANY
    # responding adult have property X" (max), not "average code value"
    # (mean, meaningless for a categorical). A separate MAX-aggregated
    # groupby, joined onto the mean-aggregated one above. jbft_dv/jbsemp
    # are legitimately inapplicable (NaN) for non-workers, same
    # structurally-missing-not-at-random pattern as hsval/carval elsewhere
    # in this project -- households with zero employed adults correctly
    # get has_parttime_worker/has_selfemployed_worker = 0 (via fillna(0)
    # below, only once has_employed_adult establishes the household has no
    # employed adult at all) rather than NaN.
    emp_flags = pd.DataFrame(index=df.index)
    if "jbstat" in df.columns:
        emp_flags["is_employed"] = df["jbstat"].isin(JBSTAT_EMPLOYED_CODES).astype(float)
    if "jbft_dv" in df.columns:
        emp_flags["is_fulltime"] = (df["jbft_dv"] == 1).astype(float)
        emp_flags["is_parttime"] = (df["jbft_dv"] == 2).astype(float)
    if "jbsemp" in df.columns:
        emp_flags["is_selfemployed"] = (df["jbsemp"] == 2).astype(float)
    if not emp_flags.empty:
        emp_flags[HH_IDENTIFIER] = df[HH_IDENTIFIER]
        emp_agg = emp_flags.groupby(HH_IDENTIFIER).max()
        rename_map = {
            "is_employed": "has_employed_adult",
            "is_fulltime": "has_fulltime_worker",
            "is_parttime": "has_parttime_worker",
            "is_selfemployed": "has_selfemployed_worker",
        }
        emp_agg = emp_agg.rename(columns=rename_map)
        if "has_employed_adult" in emp_agg.columns:
            no_employed = emp_agg["has_employed_adult"] == 0
            for c in ["has_fulltime_worker", "has_parttime_worker", "has_selfemployed_worker"]:
                if c in emp_agg.columns:
                    emp_agg.loc[no_employed & emp_agg[c].isna(), c] = 0.0
        agg = agg.join(emp_agg, how="left")

        if "has_employed_adult" in agg.columns:
            # workless_household -- JRF's headline "workless" category
            # (p.76), approximated as "no responding adult in paid/self
            # employment." Caveat (documented in README): this isn't
            # JRF's working-age-only definition, so a pensioner-only
            # household with no employed adult also reads as workless here.
            agg["workless_household"] = 1.0 - agg["has_employed_adult"]

            # 3-category grouping matching JRF's in-work/out-of-work +
            # part-time framing (pp.76-86) -- self-employed or any
            # full-time worker outranks a part-time-only household, since
            # JRF finds part-time work carries a materially higher poverty
            # rate than full-time (22% vs 8%). Explicit mask assignment
            # (not np.select) -- see _derive_family_composition_group's
            # comment for why: string choices + a np.nan default don't
            # share a common numpy dtype under recent numpy.
            emp_group = pd.Series(np.nan, index=agg.index, dtype=object)
            emp_group[agg["has_employed_adult"] == 1] = "Part-time only"
            # agg.get(col, 0) returns a bare Python 0 (not a length-matched
            # Series) when jbft_dv AND jbsemp were both absent from this
            # wave's raw file, collapsing fulltime_or_self to a scalar bool
            # instead of a boolean mask -- build explicit zero-filled Series
            # so the `==` comparisons always broadcast elementwise.
            fulltime_col = agg["has_fulltime_worker"] if "has_fulltime_worker" in agg.columns \
                else pd.Series(0, index=agg.index)
            selfemp_col = agg["has_selfemployed_worker"] if "has_selfemployed_worker" in agg.columns \
                else pd.Series(0, index=agg.index)
            fulltime_or_self = (fulltime_col == 1) | (selfemp_col == 1)
            emp_group[fulltime_or_self] = "Full-time or self-employed"
            emp_group[agg["workless_household"] == 1] = "Workless household"
            emp_group[agg["has_employed_adult"].isna()] = np.nan
            agg["employment_group"] = emp_group

    log.info("wave %s: %d respondents -> %d households (individual aggregation)",
              wave, len(df), len(agg))
    return agg.reset_index()


def load_wave_hrp_ethnicity(wave: str) -> pd.DataFrame:
    """
    Household reference person's ethnicity group for one wave -- read
    directly from indresp and NOT household-mean-aggregated (racel_dv is
    categorical, not ordinal/continuous, so averaging it across household
    members would be meaningless; it's also asked once and carried
    forward per person by Understanding Society's own derived-variable
    logic, so no wave-to-wave imputation is needed here either).

    Returns columns [IND_IDENTIFIER, "ethnicity_group"] -- joined onto the
    household panel via hrpid in build_wave_panel, matching JRF's own
    convention of defining ethnicity-based poverty rates by "households
    headed by someone from a X background" (UK Poverty 2025, pp.42-49).
    """
    path = _wave_path(wave, "indresp")
    if not path.exists():
        raise FileNotFoundError(f"UKHLS indresp not found for wave {wave}: {path}")

    col = f"{wave}_{IND_ETHNICITY_VAR}"
    available = _available_columns(path)
    if col not in available:
        return pd.DataFrame(columns=[IND_IDENTIFIER, "ethnicity_group"])

    df = pd.read_stata(path, columns=[IND_IDENTIFIER, col], convert_categoricals=False)
    df = df.rename(columns={col: IND_ETHNICITY_VAR})
    df = _recode_missing(df, [IND_ETHNICITY_VAR])
    df["ethnicity_group"] = df[IND_ETHNICITY_VAR].map(ETHNICITY_GROUP_RECODE)
    return df[[IND_IDENTIFIER, "ethnicity_group"]]


# =============================================================================
# STEP 4 — Build one wave's household panel rows
# =============================================================================

def build_wave_panel(wave: str) -> pd.DataFrame:
    hh = load_wave_hhresp(wave)
    ind_agg = load_wave_indresp_aggregated(wave)
    panel = hh.merge(ind_agg, on=HH_IDENTIFIER, how="left")

    if "hrpid" in panel.columns:
        eth = load_wave_hrp_ethnicity(wave)
        panel = panel.merge(eth, left_on="hrpid", right_on=IND_IDENTIFIER, how="left")
        panel = panel.drop(columns=[IND_IDENTIFIER], errors="ignore")

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

_MIN_PLAUSIBLE_ANNUAL_INCOME = 1200.0

# Outcome variants (analysis_plan_rerun.md amendment A1). Suffix "" is the
# primary outcome; the others are sensitivity / comparison versions.
FUEL_OUTCOME_VARIANTS = ("", "_s1", "_s2", "_v1")


def _routed_fuel_spend(df: pd.DataFrame, nonresp_as_zero: bool) -> pd.Series:
    """Annual fuel spend following the UKHLS questionnaire routing.

    fuelduel is asked only of households using BOTH electricity and gas, so
    single-fuel households are routed straight to xpelecy / xpgasy. A required
    amount that is item nonresponse gives NaN, or 0 when nonresp_as_zero
    (the S1 lower bound). Structural -8 zeros (fuel not used) are 0.
    """
    def amount(col: str) -> pd.Series:
        s = df.get(col, pd.Series(np.nan, index=df.index))
        if nonresp_as_zero:
            nr = df.get(f"{col}_nr", pd.Series(False, index=df.index))
            s = s.where(~nr, 0.0)
        return s

    elec, gas = df["fuelhave1"] == 1, df["fuelhave2"] == 1
    fuelduel = df["fuelduel"]
    duel_nr = df.get("fuelduel_nr", pd.Series(False, index=df.index))

    elec_gas = pd.Series(0.0, index=df.index)
    both = elec & gas
    elec_gas[both & (fuelduel == 1)] = amount("xpduely")
    # Separate bills, or billing type unknown (xpgasy/xpelecy are then asked).
    elec_gas[both & ((fuelduel == 2) | duel_nr)] = amount("xpgasy") + amount("xpelecy")
    elec_gas[elec & ~gas] = amount("xpelecy")
    elec_gas[gas & ~elec] = amount("xpgasy")

    oil = pd.Series(0.0, index=df.index)
    oil[df["fuelhave3"] == 1] = amount("xpoily")
    other = pd.Series(0.0, index=df.index)
    other[df["fuelhave4"] == 1] = amount("xpsfly")
    return elec_gas + oil + other


def _v1_fuel_spend(df: pd.DataFrame) -> pd.Series:
    """The submitted-draft-v1 rule, kept only as a comparison column: drops
    every household with fuelduel missing (incl. -8, i.e. not dual-fuel) and
    zero-fills nonresponse on the separate/oil/other amounts."""
    combined = df.get("xpduely", pd.Series(np.nan, index=df.index))
    separate = (df.get("xpgasy", pd.Series(0.0, index=df.index)).fillna(0)
                + df.get("xpelecy", pd.Series(0.0, index=df.index)).fillna(0))
    fuelduel = df["fuelduel"]
    elec_gas = pd.Series(
        np.where(fuelduel == 1, combined, np.where(fuelduel == 2, separate, np.nan)),
        index=df.index)
    total = (elec_gas + df.get("xpoily", pd.Series(0.0, index=df.index)).fillna(0)
             + df.get("xpsfly", pd.Series(0.0, index=df.index)).fillna(0))
    total[elec_gas.isna()] = np.nan
    return total


def _fuel_outcome_exclusion(df: pd.DataFrame, primary_raw: pd.Series) -> pd.Series:
    """First A1 exclusion reason per row ("in_scope" if none), in the order
    used by the sample-flow reconciliation."""
    elec, gas = df["fuelhave1"] == 1, df["fuelhave2"] == 1
    oil, other = df["fuelhave3"] == 1, df["fuelhave4"] == 1
    reason = pd.Series("in_scope", index=df.index)
    reason[primary_raw.isna()] = "item_nonresponse"
    reason[~elec & ~gas & (oil | other)] = "elec_not_reported_oil_or_other_only"
    reason[gas & ~elec] = "elec_not_reported_gas_only"
    reason[~elec & ~gas & ~oil & ~other] = "no_fuel_reported"
    reason[df[["fuelhave1", "fuelhave2", "fuelhave3", "fuelhave4"]].isna().any(axis=1)] = \
        "fuelhave_module_nonresponse"
    return reason


def _ratio(spend: pd.Series, annual_net_income: pd.Series) -> pd.Series:
    # Implausible near-zero or negative incomes (< GBP 1,200/yr) are set to
    # NaN; >100% of income is capped at 1.0 (kept as severe hardship).
    ratio = spend / annual_net_income
    ratio[annual_net_income < _MIN_PLAUSIBLE_ANNUAL_INCOME] = np.nan
    return ratio.clip(upper=1.0)


def compute_fuel_to_income(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fuel-to-income ratio, amendment A1 (analysis_plan_rerun.md).

    Primary (`fuel_to_income_ratio`): routing-aware spend, complete-case on
    every required amount, electricity use reported.
    `_s1`: lower bound -- item nonresponse amounts set to 0.
    `_s2`: primary + households not reporting electricity (gas-only,
           oil/other-only), spend as reported.
    `_v1`: submitted-draft-v1 rule, for comparison only.
    `fuel_outcome_exclusion` records why a row is outside the primary outcome.
    """
    primary_raw = _routed_fuel_spend(df, nonresp_as_zero=False)
    reason = _fuel_outcome_exclusion(df, primary_raw)
    in_scope = reason == "in_scope"
    elec_not_reported = reason.str.startswith("elec_not_reported")

    spend = {
        "": primary_raw.where(in_scope),
        "_s1": _routed_fuel_spend(df, nonresp_as_zero=True).where(
            in_scope | (reason == "item_nonresponse")),
        "_s2": primary_raw.where(in_scope | elec_not_reported),
        "_v1": _v1_fuel_spend(df),
    }
    annual_net_income = df.get("fihhmnnet1_dv", pd.Series(np.nan, index=df.index)) * 12
    for suffix in FUEL_OUTCOME_VARIANTS:
        df[f"total_fuel_spend{suffix}"] = spend[suffix]
        df[f"fuel_to_income_ratio{suffix}"] = _ratio(spend[suffix], annual_net_income)

    income_excluded = in_scope & df["fuel_to_income_ratio"].isna()
    reason[income_excluded & annual_net_income.isna()] = "income_missing"
    reason[income_excluded & annual_net_income.notna()] = "income_below_1200"
    df["fuel_outcome_exclusion"] = reason

    n_capped = int((primary_raw.where(in_scope) / annual_net_income)
                   .where(annual_net_income >= _MIN_PLAUSIBLE_ANNUAL_INCOME).gt(1.0).sum())
    log.info("fuel_to_income_ratio: %d rows capped at 1.0", n_capped)
    for r, n in reason.value_counts().items():
        log.info("fuel_outcome_exclusion: %-40s %7d", r, n)
    for suffix in FUEL_OUTCOME_VARIANTS:
        n_valid = int(df[f"fuel_to_income_ratio{suffix}"].notna().sum())
        log.info("fuel_to_income_ratio%-4s: %d/%d valid (%.1f%%)",
                  suffix, n_valid, len(df), 100.0 * n_valid / len(df))
    return df


def build_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    high_fuel_vulnerable: absolute UK fuel-poverty threshold (ratio >= 10%).
    high_fuel_vulnerable_relative: within-wave P75 version, for continuity
    with the old AEV_QUANTILE convention.
    """
    for suffix in FUEL_OUTCOME_VARIANTS:
        r = df[f"fuel_to_income_ratio{suffix}"]
        df[f"high_fuel_vulnerable{suffix}"] = (r >= FUEL_POVERTY_RATIO_THRESHOLD).astype("Int64")
        df.loc[r.isna(), f"high_fuel_vulnerable{suffix}"] = pd.NA

    ratio = df["fuel_to_income_ratio"]

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


def compute_equivalised_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Secondary, ROBUSTNESS-CHECK ratio -- NOT a replacement for
    fuel_to_income_ratio/high_fuel_vulnerable above. The UK's own official
    10%-of-income fuel-poverty definition is itself unequivalised, so this
    project's primary target stays unequivalised too (anchored to that
    government standard). But an unequivalised ratio can misclassify
    across household sizes/compositions -- exactly what JRF's own poverty
    methodology (Annex 1, Table 18) uses equivalisation to correct for
    -- so this ratio equivalises net income by ieqmoecd_dv (Modified OECD
    equivalence scale, the same scale family JRF uses) and is used
    downstream only as a sensitivity check
    (src.ukhls_vulnerability_classification.run_equivalisation_robustness_check),
    never as a modelling target.
    """
    if "ieqmoecd_dv" not in df.columns:
        log.warning("ieqmoecd_dv not found -- fuel_to_income_ratio_equivalised will be NaN")
        df["fuel_to_income_ratio_equivalised"] = np.nan
        df["high_fuel_vulnerable_equivalised"] = pd.NA
        return df

    annual_net_income = df.get("fihhmnnet1_dv", pd.Series(np.nan, index=df.index)) * 12
    equivalised_income = annual_net_income / df["ieqmoecd_dv"]
    ratio_eq = df["total_fuel_spend"] / equivalised_income
    ratio_eq[annual_net_income < 1200.0] = np.nan
    ratio_eq = ratio_eq.clip(upper=1.0)
    df["fuel_to_income_ratio_equivalised"] = ratio_eq

    df["high_fuel_vulnerable_equivalised"] = (ratio_eq >= FUEL_POVERTY_RATIO_THRESHOLD).astype("Int64")
    df.loc[ratio_eq.isna(), "high_fuel_vulnerable_equivalised"] = pd.NA

    n_valid = int(ratio_eq.notna().sum())
    log.info("fuel_to_income_ratio_equivalised: %d/%d valid (%.1f%%)",
              n_valid, len(df), 100.0 * n_valid / len(df) if len(df) else 0.0)
    return df


# =============================================================================
# STEP 6 — COR-style explanatory composite (NOT the target)
# =============================================================================

def _normalize_01(s: pd.Series) -> pd.Series:
    lo, hi = s.min(), s.max()
    if pd.isna(lo) or pd.isna(hi) or (hi - lo) < 1e-10:
        return pd.Series(np.nan, index=s.index, dtype=float)
    return (s - lo) / (hi - lo)


STRAIN_ITEMS = ["finnow", "finfut_risk", "scghq1_dv"]
STRAIN_ITEMS_V1 = STRAIN_ITEMS + ["xphsdba"]


def build_financial_strain_composite(df: pd.DataFrame) -> pd.DataFrame:
    """
    Financial/psychological strain: row mean (missing-aware) of min-max
    normalised items (analysis_plan_rerun.md Stage 3).

    financial_strain_score       primary: finnow + finfut_risk + GHQ distress.
                                 Bill arrears (xphsdba) is left out because
                                 it already enters the driver model as
                                 bill_security.
    financial_strain_score_v1    submitted-draft-v1 version incl. xphsdba
                                 (sensitivity A).
    financial_strain_score_lag1  the same household's primary strain at the
                                 previous wave, linked by hrpid exactly as
                                 Stage 5 links transitions (sensitivity B).
    """
    norm = {c: _normalize_01(df[c]) for c in STRAIN_ITEMS_V1 if c in df.columns}
    if not norm:
        log.warning("No components available for financial_strain_score")
        for c in ["financial_strain_score", "financial_strain_score_v1", "financial_strain_score_lag1"]:
            df[c] = np.nan
        return df
    df["financial_strain_score"] = pd.concat(
        [norm[c] for c in STRAIN_ITEMS if c in norm], axis=1).mean(axis=1)
    df["financial_strain_score_v1"] = pd.concat(list(norm.values()), axis=1).mean(axis=1)

    df["financial_strain_score_lag1"] = np.nan
    if "hrpid" in df.columns:
        next_wave = {w: WAVE_LETTERS[i + 1] for i, w in enumerate(WAVE_LETTERS[:-1])}
        prev = (df[["wave", "hrpid", "financial_strain_score"]]
                .dropna(subset=["hrpid"]).drop_duplicates(subset=["wave", "hrpid"], keep=False))
        prev = prev.assign(wave=prev["wave"].map(next_wave)).dropna(subset=["wave"])
        lag = df[["wave", "hrpid"]].merge(
            prev.rename(columns={"financial_strain_score": "lag"}),
            on=["wave", "hrpid"], how="left")["lag"]
        df["financial_strain_score_lag1"] = lag.values
    log.info("financial_strain_score: %d valid | _v1: %d | _lag1: %d",
             df["financial_strain_score"].notna().sum(),
             df["financial_strain_score_v1"].notna().sum(),
             df["financial_strain_score_lag1"].notna().sum())
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

FES_SERIES = ("gas", "electricity", "carbon")
FES_MIN_ZSCORE_MONTHS = 36   # analysis_plan_rerun.md Stage 2: expanding window, min 36 months


def _past_only_moments(core: pd.DataFrame, origin_end: pd.Timestamp) -> dict | None:
    """Mean/SD of each growth series over data up to origin_end only
    (expanding window from the start of the series)."""
    hist = core.loc[:origin_end]
    moments = {}
    for s in FES_SERIES:
        vals = hist[f"{s}_growth"].dropna()
        if len(vals) < FES_MIN_ZSCORE_MONTHS:
            return None
        moments[s] = (float(vals.mean()), float(vals.std()) or 1.0)
    return moments


def attach_fes_delta(df: pd.DataFrame) -> pd.DataFrame:
    """
    FES Magnitude, FES Current and FES Delta per household-wave
    (analysis_plan_rerun.md Stage 2).

    Magnitude: core-mode rolling forecast from the December (Y-1) origin
      (`as_of_year` = Y-1, trained on data through Dec Y-1), for the
      household's own interview month in year Y. Nothing after the origin
      is used, and the origin precedes every interview it is attached to.
      `fes_magnitude` is the 4-term index (3 growth z-terms + forecast
      uncertainty); `fes_magnitude_growth3` drops the uncertainty term.
    Current: realised growth in the last complete month before the
      interview (month m-1), z-scored with the same past-only moments as the
      forecast (expanding window through Dec Y-1). Three terms: no realised
      analogue of forecast uncertainty exists.
    Delta: Magnitude - Current (and the growth-only analogue).
    Households interviewed in years with no feasible origin (too little
    history) get NaN and are logged.
    """
    for col in ["fes_magnitude", "fes_magnitude_growth3", "fes_current",
                "fes_delta", "fes_delta_growth3"]:
        df[col] = np.nan
    df["fes_vintage_as_of_year"] = df["interview_year"] - 1

    if not paths.FES_ROLLING_MONTHLY_FILE.exists():
        log.warning("No rolling monthly FES (%s) -- FES columns left NaN. Run "
                    "forecast_pipeline.py --rolling --core-only --tune-per-origin.",
                    paths.FES_ROLLING_MONTHLY_FILE)
        return df

    monthly = pd.read_csv(paths.FES_ROLLING_MONTHLY_FILE)
    assert (monthly["target_year"] == monthly["as_of_year"] + 1).all(), \
        "rolling FES rows must forecast the year after their origin"
    monthly["fes_core_growth3"] = monthly[[f"z_{s}_core" for s in FES_SERIES]].sum(
        axis=1, min_count=len(FES_SERIES))
    by_month = monthly.set_index(["as_of_year", "target_month"])
    key = pd.MultiIndex.from_arrays([df["fes_vintage_as_of_year"], df["interview_month"]])
    df["fes_magnitude"] = key.map(by_month["fes_core"])
    df["fes_magnitude_growth3"] = key.map(by_month["fes_core_growth3"])
    # Interview month unknown: annual mean of the SAME vintage.
    by_year = monthly.groupby("as_of_year")[["fes_core", "fes_core_growth3"]].mean()
    no_month = df["interview_month"].isna()
    df.loc[no_month, "fes_magnitude"] = df.loc[no_month, "fes_vintage_as_of_year"].map(by_year["fes_core"])
    df.loc[no_month, "fes_magnitude_growth3"] = df.loc[no_month, "fes_vintage_as_of_year"].map(
        by_year["fes_core_growth3"])

    core = pd.read_csv(paths.CORE_CSV, index_col=0, parse_dates=True).sort_index()
    obs_date = pd.to_datetime(dict(year=df["interview_year"], month=df["interview_month"], day=1),
                              errors="coerce") - pd.DateOffset(months=1)
    current = pd.Series(np.nan, index=df.index)
    for vintage in df["fes_vintage_as_of_year"].dropna().unique():
        moments = _past_only_moments(core, pd.Timestamp(f"{int(vintage)}-12-01"))
        if moments is None:
            continue
        rows = df.index[df["fes_vintage_as_of_year"] == vintage]
        z = [(obs_date[rows].map(core[f"{s}_growth"]) - m) / sd for s, (m, sd) in moments.items()]
        current[rows] = pd.concat(z, axis=1).sum(axis=1, min_count=len(FES_SERIES))
    df["fes_current"] = current
    df["fes_delta"] = df["fes_magnitude"] - df["fes_current"]
    df["fes_delta_growth3"] = df["fes_magnitude_growth3"] - df["fes_current"]

    covered = sorted(monthly["target_year"].unique())
    n_ok = int(df["fes_delta"].notna().sum())
    log.info("FES (core, Dec Y-1 vintage): target years %s-%s; %d/%d rows with fes_delta",
             covered[0], covered[-1], n_ok, len(df))
    for y, n in df.loc[df["fes_delta"].isna(), "interview_year"].value_counts().sort_index().items():
        log.info("FES missing: interview_year=%s  rows=%d", y, n)
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
    df = compute_equivalised_ratio(df)
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
    setup_logger("energy_stress", log_file="outputs_v2/logs/pipeline.log")
    run()
