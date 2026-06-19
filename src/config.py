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
N_LATENT_DIMS: int      = 4     # autoencoder bottleneck neurons
AE_EPOCHS: int          = 300
AE_LEARNING_RATE: float = 1e-3
PCA_COMPONENTS: int     = 4
EFA_FACTORS: int        = 4

# =============================================================================
# Output
# =============================================================================
DPI: int = 150
FIGURE_FORMAT: str = "png"
