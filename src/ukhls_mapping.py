"""
ukhls_mapping.py
─────────────────
Variable registry for the Understanding Society (UKHLS, UK Data Service
Study 6614) household panel — waves a–o (≈2009–2023). Parallels
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
]

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
)

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
