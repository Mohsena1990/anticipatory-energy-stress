"""
ukhls_mapping.py
─────────────────
Variable registry for the Understanding Society (UKHLS, UK Data Service
Study 6614) household panel — waves a–o (≈2009–2024, wave o's fieldwork
spans both calendar years). Parallels
`construct_mapping.py`'s role in the old ENABLE-based design: the single
shared vocabulary that `src.ukhls_preprocessing` draws on, so item lists,
missing-value conventions, and per-wave availability notes live in exactly
one place.

Variable naming convention
───────────────────────────
Every UKHLS variable is prefixed with its wave letter in the raw .dta files
(e.g. `o_xpduely` in wave o). This module refers to variables by their
BARE name (`xpduely`); `src.ukhls_preprocessing` adds/strips the wave
prefix when reading each wave file.

Missing-value convention
─────────────────────────
Standard UKHLS negative sentinel codes (distinct from ENABLE's positive
9/98/99 codes):
  -9 missing, -8 inapplicable, -7 proxy/phone, -2 refusal, -1 don't know
All of these mean "no substantive answer" and are recoded to NaN.

Wave fieldwork years (approximate)
────────────────────────────────────
Each UKHLS wave fields over a ~24-month window; WAVE_FIELDWORK_START_YEAR
gives the first calendar year of that window. Combined with the `month`
variable (1–12 = start year, 13–24 = start year + 1), this gives an
approximate calendar month for each interview — sufficient for Phase 1's
annual price-growth attachment. Exact day-level alignment is a Phase 2
refinement (see project plan).
"""

from __future__ import annotations

# =============================================================================
# Wave list and approximate fieldwork years
# =============================================================================

WAVE_LETTERS: list[str] = list("abcdefghijklmno")

WAVE_FIELDWORK_START_YEAR: dict[str, int] = {
    letter: 2009 + i for i, letter in enumerate(WAVE_LETTERS)
}

# =============================================================================
# Missing-value convention
# =============================================================================

MISSING_CODES: list[int] = [-9, -8, -7, -2, -1]

# =============================================================================
# Household-level (hhresp) variables — present in every wave a–o unless noted
# =============================================================================

HH_IDENTIFIER = "hidp"

# Household reference person's pidp -- hidp itself is wave-specific (reissued
# whenever household composition changes), so it cannot link the same
# household across waves. hrpid CAN: matching hrpid_t == hrpid_{t+1} directly
# at the household level identifies "the same reference person still heads a
# household in the next wave" without needing an indresp lookup. Verified
# against the raw waves: present in every wave's hhresp file, ~72-73% direct
# match rate wave-to-wave (normal UKHLS attrition). Used by
# src.ukhls_forward_prediction to build wave-to-wave transition pairs.
HH_LINK_VARS: list[str] = ["hrpid"]

HH_TIMING_VARS: list[str] = ["month", "quarter"]

HH_GEOGRAPHY_VARS: list[str] = ["gor_dv"]

# Fuel expenditure — combined bill (fuelduel==1) OR separate gas+electricity
# (fuelduel==2), plus oil/other fuel for off-grid heating households.
HH_FUEL_EXPENDITURE_VARS: list[str] = [
    "fuelduel",   # 1=combined bill, 2=separate bills
    "xpduely",    # combined gas+electricity annual spend (fuelduel==1)
    "xpgasy",     # gas annual spend (fuelduel==2)
    "xpelecy",    # electricity annual spend (fuelduel==2, or elec-only)
    "xpoily",     # oil annual spend
    "xpsfly",     # other/solid fuel annual spend
    "duelpay",    # combined-bill payment method (4=prepayment meter)
    "elecpay",    # electricity payment method (4=prepayment meter)
    # Fuels used (1=mentioned). These define the questionnaire routing:
    # fuelduel is asked only if elec AND gas are used, so fuelduel=-8 means
    # "not dual-fuel", not missing (analysis_plan_rerun.md amendment A1).
    "fuelhave1",  # electricity
    "fuelhave2",  # gas
    "fuelhave3",  # oil
    "fuelhave4",  # other fuel, incl. solid fuel
]

# Fuel amounts whose item nonresponse (-1 DK, -2 refused, -9 missing) must be
# told apart from -8 inapplicable before the generic missing-code recode.
FUEL_NONRESPONSE_CODES: list[int] = [-1, -2, -9]
FUEL_AMOUNT_VARS: list[str] = ["fuelduel", "xpduely", "xpgasy", "xpelecy", "xpoily", "xpsfly"]

HH_INCOME_VARS: list[str] = [
    "fihhmngrs_dv",   # gross household income, month before interview
    "fihhmnnet1_dv",  # net household income, month before interview
]

HH_HOUSING_VARS: list[str] = [
    "tenure_dv",   # housing tenure (1=owned outright ... 8=other)
    "hsbeds",      # number of bedrooms
    "ctband_dv",   # council tax band (property size/value proxy)
    "heatch",      # has central heating (1=yes, 2=no)
    "htpmp",       # has heat pump (later waves only)
]

# Family composition (household-level derived variables, ~100% coverage
# verified against o_hhresp.dta) -- JRF UK Poverty 2025 devotes a full
# section (pp.34-41) to family type/size as one of the strongest poverty
# predictors (lone-parent child poverty 44% vs 25% couple; large families
# [3+ children] 45%), a dimension this project had none of until now.
HH_FAMILY_VARS: list[str] = [
    "hhtype_dv",   # Composition of household, LFS-version
    "nkids_dv",    # Number of children in household
    "nch02_dv", "nch34_dv", "nch511_dv", "nch1215_dv",  # child age bands
    "hhsize",      # Household size, incl. absent members
]

# Modified OECD equivalence scale -- the same family of scale JRF's own
# Annex 1 (Table 18) uses to equivalise income for household size/
# composition. 99.8% coverage verified against o_hhresp.dta. Used as a
# plain control + a secondary robustness-check ratio, NOT to replace the
# primary fuel_to_income_ratio (which mirrors the UK's own unequivalised
# official 10%-of-income fuel-poverty definition).
HH_EQUIVALISATION_VARS: list[str] = ["ieqmoecd_dv"]

# Household-level hardship/coping — general "problems paying bills" ordinal
# is preferred over the binary council-tax-specific item where available.
HH_HARDSHIP_VARS: list[str] = [
    "xphsdba",   # problems paying bills: 1=up to date, 2=behind some, 3=behind all
    "xphsdct",   # problems paying council tax: 1=yes, 2=no (all waves)
]

# =============================================================================
# COR resource dimensions (Hobfoll 1989) — Object / Condition / Personal /
# Energy. Feeds the Stage 2 COR-SEM (src.ukhls_cor_sem) measurement model.
# =============================================================================

# Object resources: physical/material assets. hsbeds already in
# HH_HOUSING_VARS. hsval/carval are STRUCTURALLY missing for non-owners/
# non-car-owners (not random) — the CFA must use FIML, not row-drop.
HH_OBJECT_VARS: list[str] = ["hsrooms", "ncars", "carval", "hsval"]

# Condition resources (individual-level, aggregated to household): economic
# activity as an employment-security proxy. tenure_dv (housing security)
# already in HH_HOUSING_VARS.
IND_CONDITION_VARS: list[str] = ["jbstat"]

# Personal resources (individual-level): human capital / health.
IND_PERSONAL_VARS: list[str] = ["dvage", "health", "sf1", "qfhigh_dv"]

# Employment status/hours (individual-level, aggregated to household via
# "does ANY adult have property X" -- see
# src.ukhls_preprocessing.load_wave_indresp_aggregated). jbstat (already in
# IND_CONDITION_VARS) only gives an employment-SECURITY ordinal; JRF UK
# Poverty 2025's "Work and poverty" section (pp.76-86) instead breaks
# poverty out by work STATUS (in-work 12% vs out-of-work 43%; full-time 8%
# vs part-time 22%), a dimension this project had none of until now.
# jbft_dv/jbsemp verified against o_indresp.dta's value labels: jbft_dv
# 1=FT employee/2=PT employee (employees only); jbsemp 1=Employee/
# 2=Self-employed. jbhrs kept for potential future hours-based analysis;
# jbsic07_cc (industry sector) deliberately left out of this pass -- see
# README limitations.
IND_EMPLOYMENT_VARS: list[str] = ["jbft_dv", "jbsemp", "jbhrs", "jbterm1"]

# jbstat raw codes counted as "in paid or self employment" for the
# household-level workless_household flag (src.ukhls_preprocessing) --
# matches JBSTAT_SECURITY_RECODE's two highest-security codes.
JBSTAT_EMPLOYED_CODES: set[int] = {1, 2}

# Disability (individual-level, aggregated to household): `health` alone
# (already in IND_PERSONAL_VARS, RECODED to health_good) only captures
# long-standing illness/disability ever, not whether it limits daily
# activity -- healthlink adds that, together giving an Equality-Act-2010-
# style disability flag (JRF's own definition, UK Poverty 2025 p.65:
# "a physical or mental impairment which has a substantial and long-term
# adverse effect on the ability to carry out normal day-to-day
# activities"), which health alone conflates with milder/non-limiting
# conditions. Kept as a separate list (not folded into IND_PERSONAL_VARS)
# because it's excluded from the SEM's PERSONAL factor -- this is a new
# descriptive breakdown dimension, not a COR-SEM indicator.
IND_DISABILITY_VARS: list[str] = ["healthlink"]

# Energy resources (individual-level, additional to income already in
# HH_INCOME_VARS and inoutflows already in HH_COPING_VARS_RECENT_ONLY).
IND_ENERGY_VARS: list[str] = ["fiyrinvinc_dv"]

# jbstat (Current economic activity) -> ordinal employment-security score,
# higher = more secure/stable. Judgment-call banding, documented for
# transparency: paid employment is most secure; unemployment/long-term
# sickness least; leave/furlough/apprenticeship treated as "still
# attached to a job" (moderately secure); "doing something else" (97) is
# too ambiguous to score and left NaN.
JBSTAT_SECURITY_RECODE: dict[int, float] = {
    2: 1.0,    # Paid employment (ft/pt)
    1: 0.8,    # Self employed
    11: 0.8,   # On apprenticeship
    12: 0.8,   # On furlough (still employed)
    5: 0.7,    # On maternity leave
    14: 0.7,   # On shared parental leave
    15: 0.7,   # On adoption leave
    4: 0.6,    # Retired
    10: 0.4,   # Unpaid, family business
    7: 0.4,    # Full-time student
    9: 0.3,    # Govt training scheme
    6: 0.3,    # Family care or home
    13: 0.2,   # Temporarily laid off / short-term working
    3: 0.0,    # Unemployed
    8: 0.0,    # LT sick or disabled
    # 97 "Doing something else" -- deliberately unmapped (-> NaN)
}

# qfhigh_dv (Highest educational qualification) -> ordinal education-
# resource band, higher = more human capital. Full UKHLS codebook (verified
# in wave o's data dictionary): 1-6 = degree-level, 7-12 = A-level/Highers-
# level, 13-16 = GCSE/O-level or below, 96 = no qualification.
QFHIGH_BAND_RECODE: dict[int, float] = {
    **{c: 1.00 for c in [1, 2, 3, 4, 5, 6]},     # degree-level
    **{c: 0.66 for c in [7, 8, 9, 10, 11, 12]},  # A-level / Highers
    **{c: 0.33 for c in [13, 14, 15, 16]},       # GCSE/O-level or below
    96: 0.00,                                     # no qualification
}

# heatch (has central heating): 1=yes, 2=no -> 1.0/0.0. Computed but NOT
# assigned to a COR factor -- empirically tested in CONDITION and rejected
# (standardized loading 0.012, see COR_FACTOR_ITEMS comment above).
HEATCH_GOOD_RECODE: dict[int, float] = {1: 1.0, 2: 0.0}

# xphsdba ("problems paying bills"): 1=up to date, 2=behind on some bills,
# 3=behind on all bills -> reversed to 1.0/0.5/0.0 so higher=more secure,
# same direction as tenure_security/jbstat_security.
BILL_SECURITY_RECODE: dict[int, float] = {1: 1.0, 2: 0.5, 3: 0.0}

# tenure_dv raw codes are NOT ordinal in security terms (e.g. code 3 "LA
# rent" sits between codes for owned/other rented, but is less secure than
# both) -> explicit security recode, higher = more housing-tenure security.
TENURE_SECURITY_RECODE: dict[int, float] = {
    1: 1.0,   # Owned outright
    2: 0.8,   # Owned with mortgage
    3: 0.5,   # Local authority rent
    4: 0.5,   # Housing assoc rented
    5: 0.4,   # Rented from employer
    6: 0.3,   # Rented private unfurnished
    7: 0.3,   # Rented private furnished
    8: 0.2,   # Other
}

# tenure_dv -> a coarser, JRF-comparable tenure grouping (UK Poverty 2025,
# Table 10, p.95: Owned outright / Buying with mortgage / Social renting /
# Private renting) -- a NEW descriptive-breakdown dimension, distinct from
# TENURE_SECURITY_RECODE's ordinal security score which already feeds the
# SEM's CONDITION factor. Codes 5-7 (rented from employer, private
# unfurnished/furnished) all map to "Private renting", the closest match;
# JRF's own table has no separate "rented from employer" category.
TENURE_GROUP_RECODE: dict[int, str] = {
    1: "Owned outright",
    2: "Buying with mortgage",
    3: "Social renting",   # Local authority rent
    4: "Social renting",   # Housing assoc rented
    5: "Private renting",  # Rented from employer
    6: "Private renting",  # Rented private unfurnished
    7: "Private renting",  # Rented private furnished
    8: "Other",
}

# =============================================================================
# Ethnicity (individual-level, attributed via household reference person)
# =============================================================================
# racel_dv is asked once (at a person's entry wave) and carried forward by
# Understanding Society's own derived-variable logic for continuing sample
# members -- confirmed populated in both wave a and wave o's raw files.
# Unlike other individual-level items in this project it is NOT
# household-mean-aggregated (it's categorical, not ordinal/continuous):
# it's read directly for the household reference person (hrpid) only,
# matching how JRF's own report defines ethnicity-based poverty rates
# ("households headed by someone from a X background" -- UK Poverty 2025,
# pp.42-49) -- see src.ukhls_preprocessing.load_wave_hrp_ethnicity.
IND_ETHNICITY_VAR = "racel_dv"

# racel_dv numeric codes -> group labels, aligned to JRF's own ethnicity
# categories (UK Poverty 2025, Figure 13/25) for direct comparability.
# Codes/labels confirmed against data/raw/ukhls/a_indresp.dta's Stata
# value labels (convert_categoricals=True), not guessed from memory.
ETHNICITY_GROUP_RECODE: dict[int, str] = {
    1:  "White",                          # British/English/Scottish/Welsh/NI
    2:  "White",                          # Irish
    4:  "White",                          # Any other White background
    5:  "Mixed/multiple ethnic groups",   # White and Black Caribbean
    6:  "Mixed/multiple ethnic groups",   # White and Black African
    7:  "Mixed/multiple ethnic groups",   # White and Asian
    8:  "Mixed/multiple ethnic groups",   # Any other Mixed background
    9:  "Indian",
    10: "Pakistani",
    11: "Bangladeshi",
    12: "Chinese",
    13: "Any other Asian background",
    14: "Black Caribbean",
    15: "Black African",
    16: "Any other Black background",
    17: "Other ethnic group",             # Arab
    97: "Other ethnic group",             # Any other ethnic group
}

# =============================================================================
# COR factor item registry — final (post-recode) column names each Stage 2
# COR-SEM first-order factor draws on. `src.ukhls_cor_sem` builds its
# semopy measurement spec directly from this registry.
# =============================================================================

COR_FACTOR_ITEMS: dict[str, list[str]] = {
    "OBJECT":    ["hsrooms", "hsbeds", "ncars", "carval", "hsval"],
    "CONDITION": ["tenure_security", "jbstat_security", "bill_security"],
    "PERSONAL":  ["health_good", "sf1_good", "qfhigh_band"],
    "ENERGY":    ["fihhmnnet1_dv", "fiyrinvinc_dv"],
}
# Two candidate revisions were EMPIRICALLY TESTED (re-fit Stage 2b, compared
# fit indices) and rejected -- kept here as a documented negative result,
# not silently dropped, since the next person tempted by the same "obvious"
# fix should not have to re-discover this:
#   - heatch (has central heating) added to CONDITION: standardized loading
#     0.012 (essentially zero) -- central heating is near-universal in the
#     UK sample, leaving almost no variance to correlate with anything.
#     REJECTED, reverted.
#   - sf1_good excluded from PERSONAL (see its coverage-collapse note
#     below): CFI went from 0.17 -> -2.37 and TLI from -0.14 -> -3.63 (both
#     WORSE, not better), while SRMR improved 0.081 -> 0.071. Net evidence
#     does not support the hypothesis that sf1_good's sparsity drives the
#     degenerate FIML fit -- REJECTED, reverted; item kept in the model.
#
# bill_security (from xphsdba, "problems paying bills": 1=up to
# date/2=behind some/3=behind all, recoded to 1.0/0.5/0.0 so higher=more
# secure, same direction as tenure_security/jbstat_security) added to
# CONDITION instead -- TESTED and KEPT: standardized loading 0.226 (real,
# not near-zero), 99.5% coverage, and it strengthened the other two
# CONDITION items too (tenure_security 0.254->0.372, jbstat_security
# 0.314->0.449) -- CONDITION went from a fragile 2-item factor to a
# genuinely 3-item one. Global fit indices (CFI/TLI) did not improve
# alongside this (they degrade with dof under the same known FIML
# limitation documented in ukhls_cor_sem.py regardless of which items are
# in the spec) -- read the loadings as the evidence for this specific
# change, not CFI/TLI, which the module's own docstring already flags as
# unreliable under FIML.
#
# dvage deliberately excluded from PERSONAL: diagnostic CFA runs showed it
# loads with the OPPOSITE sign to health_good/sf1_good/qfhigh_band (older
# age correlates with worse health/lower recent qualifications in this
# population, not "more resource" in the same direction as the other three
# items) -- age is kept as a plain demographic control for Stage 3 rather
# than forced into a reflective factor it doesn't cohere with.
#
# inoutflows2/3/4 deliberately excluded from the pooled ENERGY factor:
# they only exist in waves m/o (~7% coverage), which collapsed the
# multi-item complete-case overlap to single digits and produced degenerate
# FIML fit statistics (chi2 essentially 0, impossible negative CFI/TLI).
# fihhmnnet1_dv/fiyrinvinc_dv alone are present for ~99% of the panel and
# keep ENERGY identifiable pooled across all 15 waves.
ENERGY_DEPLETION_ITEMS_RECENT_ONLY: list[str] = ["inoutflows2", "inoutflows3", "inoutflows4"]
# inoutflows2/3/4 (used savings / new borrowing) are resource-DEPLETION
# indicators -- expected to load NEGATIVELY on ENERGY. semopy estimates
# loading sign directly (reflective CFA), so no pre-inversion is needed;
# this is a validation expectation, not a preprocessing step. Also sparse
# (waves m/o only, see HH_COPING_AVAILABLE_WAVES) -- the ENERGY factor
# stays identifiable in every wave via income/investment income alone.

# Crisis-era financial coping behaviours — ONLY present in waves m, o
# (cost-of-living-crisis module). NaN in all other waves; CatBoost handles
# missing numerics natively, no imputation needed.
HH_COPING_VARS_RECENT_ONLY: list[str] = [
    "inoutflows1",   # reduced spending
    "inoutflows12",  # reduced usage of utilities (electricity/gas/water)
    "inoutflows2",   # used savings
    "inoutflows3",   # new borrowing (bank/credit card)
    "inoutflows4",   # new borrowing (family/friends)
    "inoutflows10",  # new or increased welfare benefits
]
HH_COPING_AVAILABLE_WAVES: set[str] = {"m", "o"}

ALL_HH_VARS: list[str] = (
    HH_TIMING_VARS + HH_GEOGRAPHY_VARS + HH_FUEL_EXPENDITURE_VARS
    + HH_INCOME_VARS + HH_HOUSING_VARS + HH_HARDSHIP_VARS
    + HH_COPING_VARS_RECENT_ONLY + HH_OBJECT_VARS
    + HH_FAMILY_VARS + HH_EQUIVALISATION_VARS
)

# hhtype_dv (LFS-version household composition) -> JRF-comparable family
# groups (UK Poverty 2025, Table 5, p.36: family type x large-family
# cross-cut). Codes verified against o_hhresp.dta's Stata value labels
# directly, not guessed: 4/5 = lone parent (1 ADULT + 1/2+ children only --
# genuinely one adult present, unlike the codes below). 10/11/12 = couple
# + 1/2/3+ children. 18/20/21/23 (2+ adults with children, explicitly NOT
# a couple -- e.g. siblings or a parent+grandparent sharing a household)
# are deliberately kept in their OWN "other multi-adult with children"
# bucket rather than folded into lone-parent (would misclassify a
# 2+-adult household as single-adult) or couple (there is no couple
# relationship in these codes) -- JRF has no exact matching category
# either. Large-family split (3+ children) applied afterwards using
# nkids_dv, since hhtype_dv's own child-count granularity stops at "3 or
# more" for couples but "2 or more" for lone parents.
HHTYPE_LONE_PARENT_CODES: set[int] = {4, 5}
HHTYPE_COUPLE_WITH_CHILDREN_CODES: set[int] = {10, 11, 12}
HHTYPE_OTHER_WITH_CHILDREN_CODES: set[int] = {18, 20, 21, 23}
LARGE_FAMILY_MIN_CHILDREN: int = 3

# =============================================================================
# Individual-level (indresp) variables — merged to household level
# =============================================================================

IND_IDENTIFIER = "pidp"
IND_HH_LINK    = "hidp"

# Subjective financial situation. finnow: 1=comfortable...5=very difficult
# (already ordinal, higher=more pressure). finfut is NOT ordinal
# (1=better, 2=worse, 3=same) — recoded separately, see
# `src.ukhls_preprocessing.recode_finfut`.
IND_FINANCIAL_VARS: list[str] = ["finnow", "finfut"]

# GHQ-12 psychological distress (Likert 0-36 and caseness 0-12 derived
# scores; higher = worse wellbeing on both).
IND_WELLBEING_VARS: list[str] = ["scghq1_dv", "scghq2_dv"]

ALL_IND_VARS: list[str] = (
    IND_FINANCIAL_VARS + IND_WELLBEING_VARS
    + IND_CONDITION_VARS + IND_PERSONAL_VARS + IND_ENERGY_VARS
    + IND_EMPLOYMENT_VARS
)

# =============================================================================
# COR-style explanatory composite (context/feature — NOT the target)
# ─────────────────────────────────────────────────────────────────
# Approximate re-mapping of the old ENABLE COR dimensions onto UKHLS items.
# Unlike the old 4-dimension AEV composite, this is a single combined
# "financial/psychological strain" composite: UKHLS lacks an item battery
# rich enough (10+ items per dimension) to support the old BLI/TCR-style
# separate constructs, so all available proxies are folded into one score.
# =============================================================================

COMPOSITE_ITEM_SPECS: dict[str, str] = {
    "finnow":     "higher=more financial pressure (already 1-5 ordinal)",
    "finfut_risk": "recoded 0/0.5/1: 1=expects to be worse off next year",
    "scghq1_dv":  "higher=worse psychological distress (0-36)",
    "xphsdba":    "higher=more behind on bills (1-3 ordinal)",
}

# =============================================================================
# Fuel-poverty target
# =============================================================================

# Standard UK fuel-poverty definition: fuel spend >= 10% of net household
# income. Documented alongside a within-wave P75 relative version for
# continuity with the old AEV_QUANTILE convention.
FUEL_POVERTY_RATIO_THRESHOLD: float = 0.10
FUEL_POVERTY_RELATIVE_QUANTILE: float = 0.75

# Government Office Region codes (gor_dv), verified against the UKHLS data
# dictionary. Used for policy-figure region labelling.
GOR_LABELS: dict[int, str] = {
    1: "North East", 2: "North West", 3: "Yorkshire and the Humber",
    4: "East Midlands", 5: "West Midlands", 6: "East of England",
    7: "London", 8: "South East", 9: "South West",
    10: "Wales", 11: "Scotland", 12: "Northern Ireland",
}

# =============================================================================
# Household controls for ML (non-composite, observable characteristics)
# =============================================================================

ML_CONTROL_VARS: list[str] = [
    "tenure_dv", "hsbeds", "ctband_dv", "heatch", "htpmp",
    "gor_dv", "duelpay", "elecpay",
]
