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

# FES scenario-based signal simulation (9 named scenarios — see src/fes_scenarios.py)
FES_SCENARIO_SUMMARY_FILE   = FES_DIR / "fes_scenario_summary.csv"
FES_SCENARIO_MONTHLY_FILE   = FES_DIR / "fes_scenario_monthly_states.csv"
FES_SCENARIO_COMPARISON_FILE = FES_DIR / "fes_scenario_forecast_vs_actual_matrix.csv"
FES_SCENARIO_NOTES_FILE     = FES_DIR / "fes_scenario_interpretation_notes.csv"

# Scenario-conditioned interpretation of FES against HighAEV (interpretive
# only — never a household-level prediction, see src/fes_scenarios.py)
FES_HIGHAEV_OUT     = OUTPUTS_DIR / "fes_highaev_interpretation"
FES_HIGHAEV_MATRIX  = FES_HIGHAEV_OUT / "scenario_highaev_interpretation_matrix.csv"

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
# Outputs — unsupervised latent scores (per-household, for ML input)
# =============================================================================
LATENT_SCORES_PCA = LATENT_TABLES / "pca_scores.csv"
LATENT_SCORES_EFA = LATENT_TABLES / "efa_scores.csv"
LATENT_SCORES_AE  = LATENT_TABLES / "ae_scores.csv"

# =============================================================================
# Outputs — Route 2: COR-Informed SEM (CFA + structural model)
# =============================================================================
COR_SEM_OUT     = OUTPUTS_DIR / "cor_sem"
COR_SEM_TABLES  = COR_SEM_OUT / "tables"
COR_SEM_FIGURES = COR_SEM_OUT / "figures"
LATENT_SCORES_ROUTE2_SEM = COR_SEM_TABLES / "route2_factor_scores.csv"

# =============================================================================
# Outputs — Route 3: COR-Informed VAE
# =============================================================================
COR_VAE_OUT     = OUTPUTS_DIR / "cor_vae"
COR_VAE_TABLES  = COR_VAE_OUT / "tables"
COR_VAE_FIGURES = COR_VAE_OUT / "figures"
LATENT_SCORES_ROUTE3_VAE = COR_VAE_TABLES / "route3_vae_scores.csv"

# =============================================================================
# Outputs — cross-route comparison
# =============================================================================
ROUTE_COMPARISON_OUT     = OUTPUTS_DIR / "route_comparison"
ROUTE_COMPARISON_TABLES  = ROUTE_COMPARISON_OUT / "tables"
ROUTE_COMPARISON_FIGURES = ROUTE_COMPARISON_OUT / "figures"

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
        FES_HIGHAEV_OUT,
        ENABLE_OUT,
        CV_TABLES, CV_FIGURES,
        SEM_TABLES, SEM_FIGURES,
        LATENT_TABLES, LATENT_FIGURES,
        COR_SEM_TABLES, COR_SEM_FIGURES,
        COR_VAE_TABLES, COR_VAE_FIGURES,
        ROUTE_COMPARISON_TABLES, ROUTE_COMPARISON_FIGURES,
        ML_TABLES, ML_FIGURES,
        SHAP_TABLES, SHAP_FIGURES,
        FIGURES_DIR, TABLES_DIR, LOGS_DIR,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
