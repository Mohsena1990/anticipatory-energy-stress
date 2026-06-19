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

GAS_FILE         = RAW_DIR / "gas.csv"
ELECTRICITY_FILE = RAW_DIR / "electricity.csv"
CARBON_FILE      = RAW_DIR / "Carbon Emissions Futures Historical Data UK.csv"
CPIH_FILE        = RAW_DIR / "cpih08_188.xlsx"
GDP_FILE         = RAW_DIR / "mgdp.csv"
TEMPERATURE_FILE = RAW_DIR / "monthly-temperature-anomalies.csv"
GAS_FUTURES_FILE = RAW_DIR / "UK NBP Natural Gas Quaterly Futures Historical Data UK.csv"
ELEC_DEMAND_FILE = RAW_DIR / "historic_demand_2009_2024.csv"
ENABLE_FILE      = SOCIAL_DIR / "ENABLE.EU_dataset_survey of households.xlsx"

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
FORECAST_DIR = OUTPUTS_DIR / "macro_forecasts"   # renamed from outputs/forecasts
FES_DIR      = OUTPUTS_DIR / "fes"

FES_MONTHLY_FILE  = FES_DIR / "fes_monthly_2017.csv"
FES_SUMMARY_FILE  = FES_DIR / "fes_annual_context.csv"
FES_COMP_FILE     = FES_DIR / "fes_component_decomposition.csv"

# =============================================================================
# Outputs — ENABLE household stream
# =============================================================================
ENABLE_OUT   = OUTPUTS_DIR / "enable_cleaned"
ENABLE_CLEAN = ENABLE_OUT / "enable_uk_cleaned.csv"
ENABLE_SCORED = ENABLE_OUT / "enable_aev_scored.csv"

# =============================================================================
# Outputs — construct validation
# =============================================================================
CONSTRUCT_VAL   = OUTPUTS_DIR / "construct_validation"
CV_TABLES       = CONSTRUCT_VAL / "tables"
CV_FIGURES      = CONSTRUCT_VAL / "figures"

# =============================================================================
# Outputs — SEM / mediation
# =============================================================================
SEM_OUT     = OUTPUTS_DIR / "sem_mediation"
SEM_TABLES  = SEM_OUT / "tables"
SEM_FIGURES = SEM_OUT / "figures"

# =============================================================================
# Outputs — unsupervised latent robustness
# =============================================================================
LATENT_OUT     = OUTPUTS_DIR / "unsupervised_latent_robustness"
LATENT_TABLES  = LATENT_OUT / "tables"
LATENT_FIGURES = LATENT_OUT / "figures"

# =============================================================================
# Outputs — supervised ML classification
# =============================================================================
ML_OUT     = OUTPUTS_DIR / "ml_classification"
ML_TABLES  = ML_OUT / "tables"
ML_FIGURES = ML_OUT / "figures"
ML_PREDS   = ML_OUT / "enable_ml_predictions.csv"

# =============================================================================
# Outputs — SHAP explainability
# =============================================================================
SHAP_OUT     = OUTPUTS_DIR / "shap"
SHAP_TABLES  = SHAP_OUT / "tables"
SHAP_FIGURES = SHAP_OUT / "figures"

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
        FORECAST_DIR, FES_DIR,
        ENABLE_OUT,
        CV_TABLES, CV_FIGURES,
        SEM_TABLES, SEM_FIGURES,
        LATENT_TABLES, LATENT_FIGURES,
        ML_TABLES, ML_FIGURES,
        SHAP_TABLES, SHAP_FIGURES,
        FIGURES_DIR, TABLES_DIR, LOGS_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
