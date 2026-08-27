"""
config.py
─────────
Central configuration for the Anticipatory Energy–Carbon Stress pipeline.

All hard-coded constants live here.  Import this module rather than
scattering literals across the codebase.
"""

from __future__ import annotations

# =============================================================================
# Reproducibility
# =============================================================================
RANDOM_SEED: int = 42
NUMPY_SEED: int  = 42
TF_SEED: int     = 42
TORCH_SEED: int  = 42

# =============================================================================
# Date windows (macro forecasting pipeline)
# =============================================================================
TRAIN_START:    str = "2005-01-01"   # first usable month after 12-month lag
TRAIN_END:      str = "2015-12-01"   # last month used for model training
VALIDATION_END: str = "2016-12-01"   # hold-out evaluation year
FORECAST_START: str = "2017-01-01"   # 12-month forecast horizon start
FORECAST_END:   str = "2017-12-01"   # 12-month forecast horizon end

# Z-score reference period: training + validation window, strictly pre-forecast.
# Using the same reference across FES_core, FES_macro, and FES_actual ensures
# that differences between the three variants reflect genuine methodological
# variation rather than scaling artefacts.
ZSCORE_REF_START: str = "2005-01-01"
ZSCORE_REF_END:   str = "2016-12-01"

# =============================================================================
# ENABLE.EU household survey
# =============================================================================
UK_COUNTRY_CODE: int     = 11
MISSING_CODES: list[int] = [9, 98, 99, 999, 9999, 99999]

# High Adaptive Energy Vulnerability threshold.
# A household is classified HighAEV=1 if its composite AEV score falls at or
# above the 75th percentile within the UK sample.
AEV_QUANTILE: float = 0.75

# Energy poverty (Low-Income High-Cost proxy)
LOW_INCOME_THRESHOLD: int   = 4    # income bracket ≤ 4 ~ below 60 % of median
HIGH_COST_PERCENTILE: float = 0.80

# =============================================================================
# Macro forecasting models
# =============================================================================
FORECAST_PERIODS: int   = 12
CI_ALPHA: float         = 0.05    # 95 % prediction intervals
MC_SAMPLES: int         = 200     # Monte Carlo dropout samples (LSTM)
LSTM_LOOKBACK: int      = 12
LSTM_EPOCHS: int        = 100
LSTM_EPOCHS_FAST: int   = 30
BATCH_SIZE: int         = 16
LEARNING_RATE: float    = 1e-3

# =============================================================================
# Construct validation thresholds (guidelines, not hard gates)
# =============================================================================
MIN_ALPHA: float   = 0.60   # Cronbach's α – minimum acceptable reliability
MIN_AVE: float     = 0.50   # Average Variance Extracted – convergent validity
MAX_HTMT: float    = 0.85   # HTMT ratio – discriminant validity ceiling
MIN_LOADING: float = 0.40   # Factor loading – inclusion threshold

# =============================================================================
# Unsupervised latent robustness stream
# =============================================================================
N_LATENT_DIMS: int           = 4        # autoencoder bottleneck neurons
AE_EPOCHS: int               = 300
AE_LEARNING_RATE: float      = 1e-3
AE_VAL_SPLIT: float          = 0.20     # fraction held out for val MSE
AE_N_SEEDS: int              = 30       # seed stability runs
AE_BOTTLENECK_SIZES: tuple   = (3, 4, 5)
AE_SWEEP_SEEDS: int          = 5        # seeds per bottleneck size in sweep
AE_FORCE_CPU: bool           = True     # True = bypass XLA/Triton GPU issues
PCA_COMPONENTS: int          = 4
EFA_FACTORS: int             = 4

# =============================================================================
# Route 2 — COR-Informed SEM (semopy CFA + structural model)
# =============================================================================
CFA_ESTIMATOR: str = "MLW"   # semopy default continuous ML estimator
MIN_CFI: float      = 0.90
MIN_TLI: float      = 0.90
MAX_RMSEA: float    = 0.08
MAX_SRMR: float     = 0.08
SEM_BOOT_N: int      = 2000   # bootstrap resamples for Route 2 mediation (mirrors sem_mediation.py)

# =============================================================================
# Route 3 — COR-Informed VAE
# =============================================================================
VAE_HIDDEN_DIMS: tuple  = (16, 8)   # encoder/decoder hidden layer widths
VAE_BETA: float         = 1.0       # KL divergence weight
VAE_LAMBDA_ALIGN: float = 1.0       # COR-alignment loss weight
VAE_GAMMA_PRED: float   = 1.0       # HighAEV prediction-head loss weight
VAE_EPOCHS: int         = 300
VAE_EPOCHS_FAST: int    = 60        # used with --skip-vae / --fast dev mode
VAE_LEARNING_RATE: float = 1e-3
VAE_VAL_SPLIT: float    = 0.20
VAE_N_SEEDS: int        = 10        # seed-stability runs (fewer than AE_N_SEEDS: costlier joint loss)
# VAE_LATENT_DIMS reuses N_LATENT_DIMS (4) above — not duplicated.

# =============================================================================
# UKHLS Stage 2c — FES-conditioned COR-CVAE
# =============================================================================
UKHLS_VAE_HIDDEN_DIMS: tuple  = (16, 8)
UKHLS_VAE_BETA: float         = 1.0
UKHLS_VAE_LAMBDA_ALIGN: float = 1.0
UKHLS_VAE_GAMMA_PRED: float   = 1.0
UKHLS_VAE_EPOCHS: int         = 300
UKHLS_VAE_EPOCHS_FAST: int    = 60
UKHLS_VAE_LEARNING_RATE: float = 1e-3
UKHLS_VAE_VAL_SPLIT: float    = 0.20
UKHLS_VAE_N_SEEDS: int        = 10
UKHLS_N_LATENT_DIMS: int      = 4   # aligned to OBJECT/CONDITION/PERSONAL/ENERGY

# =============================================================================
# UKHLS Stage 3 — vulnerability identification
# =============================================================================
N_FUZZY_CLUSTERS: int      = 3      # Resource Depleted / Vulnerable to Loss / Resource Resilient
FUZZY_M: float              = 2.0    # FCM fuzziness exponent (standard default)
RESILIENT_QUANTILE: float   = 0.25   # bottom-quartile fuel_to_income_ratio -> One-Class SVM reference group

# =============================================================================
# UKHLS Stage 4 — policy geography hex-cartogram positions (UNUSED)
# =============================================================================
# Superseded by real UK region boundaries (data/geo/uk_nuts1_regions.geojson,
# src/ukhls_geo_maps.py) -- kept here, unused, as a self-contained fallback
# reference in case boundary-file/internet access is ever unavailable again.
# Hand-placed (col, row) grid positions approximating true relative UK
# geography (row 0 = north; col increases eastward).
GOR_HEX_POSITIONS: dict[str, tuple[int, int]] = {
    "Scotland":                 (2, 0),
    "Northern Ireland":         (0, 1),
    "North East":               (3, 1),
    "North West":               (2, 2),
    "Yorkshire and the Humber": (3, 2),
    "Wales":                    (1, 3),
    "West Midlands":            (2, 3),
    "East Midlands":            (3, 3),
    "East of England":          (4, 3),
    "South West":               (1, 4),
    "London":                   (3, 4),
    "South East":               (3, 5),
}
FES_BOUNDARY_ZONE: float = 0.10   # +/- band around 0.5 fuzzy membership for the "near boundary" annotation

# =============================================================================
# Output
# =============================================================================
DPI: int = 150
FIGURE_FORMAT: str = "png"
