"""
paths.py
────────
Centralised path management for all pipeline I/O.

No other module should hard-code a file path.  Import from here instead.
Call ensure_dirs() once at pipeline startup to create all output folders.
"""

from __future__ import annotations
from pathlib import Path

# Project root (parent of src/)
ROOT: Path = Path(__file__).parent.parent

# =============================================================================
# Raw input data
# =============================================================================
RAW_DIR    = ROOT / "data" / "raw"
SOCIAL_DIR = ROOT / "data" / "social_science_data"

UKHLS_RAW_DIR    = RAW_DIR / "ukhls"

GAS_FILE         = RAW_DIR / "gas.csv"
ELECTRICITY_FILE = RAW_DIR / "electricity.csv"
CARBON_FILE      = RAW_DIR / "Carbon Emissions Futures Historical Data UK.csv"
CPIH_FILE        = RAW_DIR / "cpih08_188.xlsx"
GDP_FILE         = RAW_DIR / "mgdp.csv"
TEMPERATURE_FILE = RAW_DIR / "monthly-temperature-anomalies.csv"
GAS_FUTURES_FILE = RAW_DIR / "UK NBP Natural Gas Quaterly Futures Historical Data UK.csv"
ELEC_DEMAND_FILE = RAW_DIR / "historic_demand_2009_2024.csv"

# =============================================================================
# Processed intermediate data
# =============================================================================
PROCESSED_DIR     = ROOT / "data" / "processed"
CORE_CSV          = PROCESSED_DIR / "core_energy_carbon.csv"
MACRO_CSV         = PROCESSED_DIR / "macro_controls.csv"
CORE_PROC_CSV     = PROCESSED_DIR / "core_processed.csv"
MACRO_PROC_CSV    = PROCESSED_DIR / "macro_processed.csv"

# =============================================================================
# Outputs — macro forecasting stream
# =============================================================================
OUTPUTS_DIR  = ROOT / "outputs"
FES_DIR      = OUTPUTS_DIR / "fes"

FES_MONTHLY_FILE  = FES_DIR / "fes_monthly_2017.csv"
FES_SUMMARY_FILE  = FES_DIR / "fes_annual_context.csv"
FES_COMP_FILE     = FES_DIR / "fes_component_decomposition.csv"
# Rolling walk-forward FES (forecast_pipeline.run_rolling): one row per
# (as_of_year, target_year) -- consumed by
# src.ukhls_preprocessing.attach_fes_delta to give each household-wave a
# genuinely year-varying FES Magnitude instead of one fixed constant.
FES_ROLLING_FILE  = FES_DIR / "fes_rolling_yearly.csv"
# Same rolling walk-forward run, kept at its native 12-month-per-year
# resolution instead of collapsed to one annual mean -- one row per
# (as_of_year, target_year, target_month). Preferred over FES_ROLLING_FILE
# when present, since UKHLS interview_month has ~100% coverage and this
# gives each household-wave a FES Magnitude specific to its own interview
# month rather than a single value shared by every wave in the same year.
FES_ROLLING_MONTHLY_FILE = FES_DIR / "fes_rolling_monthly.csv"
# Which of fes_core/fes_macro to use exclusively (lowest mean RMSE vs
# realised FES across every rolling year) -- see
# forecast_pipeline._select_best_fes_variant.
FES_VARIANT_SELECTION_FILE = FES_DIR / "fes_variant_selection.csv"

# =============================================================================
# Outputs — UKHLS household panel stream (Study 6614, waves a-o)
# =============================================================================
UKHLS_OUT    = OUTPUTS_DIR / "ukhls_cleaned"
UKHLS_PANEL  = UKHLS_OUT / "ukhls_panel.csv"

# Stage 2a(ii) — dataset-overview/introduction figures (panel composition,
# missingness, key-variable distributions) -- descriptive only, no modeling
UKHLS_OVERVIEW_OUT     = OUTPUTS_DIR / "ukhls_dataset_overview"
UKHLS_OVERVIEW_TABLES  = UKHLS_OVERVIEW_OUT / "tables"
UKHLS_OVERVIEW_FIGURES = UKHLS_OVERVIEW_OUT / "figures"

# Stage 2b — COR-SEM (Object/Condition/Personal/Energy -> Baseline Resource Stock)
UKHLS_SEM_OUT     = OUTPUTS_DIR / "ukhls_cor_sem"
UKHLS_SEM_TABLES  = UKHLS_SEM_OUT / "tables"
UKHLS_SEM_FIGURES = UKHLS_SEM_OUT / "figures"

# Stage 2c — FES-conditioned COR-CVAE
UKHLS_CVAE_OUT     = OUTPUTS_DIR / "ukhls_cor_cvae"
UKHLS_CVAE_TABLES  = UKHLS_CVAE_OUT / "tables"
UKHLS_CVAE_FIGURES = UKHLS_CVAE_OUT / "figures"

# Stage 3 — vulnerability identification (fuzzy c-means / one-class / CatBoost)
UKHLS_VULN_OUT     = OUTPUTS_DIR / "ukhls_vulnerability"
UKHLS_VULN_TABLES  = UKHLS_VULN_OUT / "tables"
UKHLS_VULN_FIGURES = UKHLS_VULN_OUT / "figures"

# Stage 4 — policy geography maps (real UK region boundaries, 12 GOR regions)
UKHLS_POLICY_MAPS_OUT     = OUTPUTS_DIR / "ukhls_policy_maps"
UKHLS_POLICY_MAPS_TABLES  = UKHLS_POLICY_MAPS_OUT / "tables"
UKHLS_POLICY_MAPS_FIGURES = UKHLS_POLICY_MAPS_OUT / "figures"

# =============================================================================
# Outputs — SEM / mediation. Only `ols_path` is actually imported/called from
# src.sem_mediation (by src.ukhls_cor_sem); the other functions that reference
# these paths (bootstrap_mediation, run(), plot_path_diagram, ...) are dead
# code -- nothing currently reachable ever writes here. Not eagerly created
# by ensure_dirs() any more (outputs/sem_mediation/ was confirmed empty and
# removed); the constants stay only so src/sem_mediation.py's own
# _save_csv/_save_fig helpers don't break if that dead code is ever revived.
# =============================================================================
SEM_TABLES  = OUTPUTS_DIR / "sem_mediation" / "tables"
SEM_FIGURES = OUTPUTS_DIR / "sem_mediation" / "figures"

# =============================================================================
# Shared outputs
# =============================================================================
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR  = OUTPUTS_DIR / "tables"
LOGS_DIR    = OUTPUTS_DIR / "logs"
LOG_FILE    = LOGS_DIR / "pipeline.log"


def ensure_dirs() -> None:
    """Create all output directories that do not already exist."""
    dirs = [
        PROCESSED_DIR,
        FES_DIR,
        UKHLS_OUT,
        UKHLS_OVERVIEW_TABLES, UKHLS_OVERVIEW_FIGURES,
        UKHLS_SEM_TABLES, UKHLS_SEM_FIGURES,
        UKHLS_CVAE_TABLES, UKHLS_CVAE_FIGURES,
        UKHLS_VULN_TABLES, UKHLS_VULN_FIGURES,
        UKHLS_POLICY_MAPS_TABLES, UKHLS_POLICY_MAPS_FIGURES,
        FIGURES_DIR, TABLES_DIR, LOGS_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
