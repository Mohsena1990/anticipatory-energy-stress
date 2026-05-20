# Social SEM Pipeline -- UK ENABLE.EU x FES 2018

## Methodology

### Why annual FES is used

The Forecasted Energy-Carbon Stress Index (FES) is computed at the UK-country
level as the annual mean of 12 monthly forecasts for 2018.  Every UK household
in the ENABLE.EU survey receives the same FES value because FES is a macro
indicator of the anticipated energy-market stress environment shared by all
UK residents.

The interview-date variable (T3) returns '#NULL!' for all UK respondents in this
dataset, making monthly matching impossible.  Even if dates were available, all
UK interviews fall within 2018, so matching to a specific forecast month would
not add identifying variation -- all households would still receive the same
annual contextual value.

### Why moderation with FES is not estimated

Because FES is constant for all UK households (zero within-country variance),
it cannot be estimated as a household-level predictor.  OLS regression would
absorb it into the intercept.  Any apparent interaction (FES x insecurity) would
be entirely non-identified within the UK annual sample.

To identify FES as a predictor, one would need either:
  (a) Cross-country variation in FES (different country-level FES values), or
  (b) Longitudinal variation (multiple survey years with different FES values).

### FES as contextual macro-stress exposure

FES is treated as the anticipated stress environment in which UK households make
energy decisions.  It defines the shared economic context of 2018, not a
property that varies across households.  This is consistent with stress exposure
research where contextual stressors affect all members of a community equally.

Three FES scenarios are reported descriptively:
  - **fes_core**  : primary forecast using core energy-price models
  - **fes_macro** : alternative forecast augmented with macroeconomic inputs
  - **fes_actual**: realised 2018 energy prices (benchmark)

### COR construct operationalisation

COR latent constructs are operationalised as composite scores:

| Construct | Items | Source |
|-----------|-------|--------|
| Insecurity | S8, E2A, E2B, E7A-E7E | Income difficulty + energy worry scale |
| Resource Preservation | H9, E5A1-E5A5, E5A9, E6A1-E6A8 | Thermostat control + energy behaviours |
| Thermal Discomfort | H12A, H12B, H15A-H15E | Heating satisfaction + comfort barriers |

Items are normalised to [0, 1] and averaged row-wise.  Items with zero variance
(all UK respondents gave the same answer) are excluded.

Note: Traditional ENABLE.EU thermal discomfort items (C1A, C1B, C5A-C5I,
C7A-C7F) are entirely missing in the UK sub-sample of this dataset.  UK-specific
substitutes (H12, H15) are used.

### Composite-score path analysis vs full latent SEM

This pipeline uses **composite-score path analysis** (OLS regression on derived
composite scores), not full latent SEM.  Full latent SEM would simultaneously
estimate measurement model parameters (factor loadings) and structural path
parameters, accounting for measurement error.  Composite-score path analysis
treats each composite as an observed variable, which underestimates standard
errors but is computationally simpler and more transparent.

The path model estimates:

  M1: preservation_score ~ insecurity_score + controls
  M2: discomfort_score ~ preservation_score + insecurity_score + controls
  M0: discomfort_score ~ insecurity_score + controls  (total effect for mediation)

### Mediation

The indirect COR pathway (Insecurity -> Preservation -> Discomfort) is quantified
as the product of path coefficients a x b, with 95% bootstrap confidence
intervals (1,000 resamples, seed 42).

## Outputs

| File | Description |
|------|-------------|
| `tables/annual_fes_context.csv` | FES annual values and methodological notes |
| `tables/item_diagnostics.csv` | Per-item validity, missingness, variance |
| `tables/construct_variable_map.csv` | Item-to-construct mapping with roles |
| `tables/cor_score_summary.csv` | Construct-level descriptive statistics |
| `tables/path_model_estimates.csv` | OLS path coefficients, SE, t, p |
| `tables/mediation_effects.csv` | Indirect effect with bootstrap CI |
| `tables/cor_mechanism_validation.csv` | COR pathway support summary |
| `tables/fes_context_summary.csv` | FES scenario descriptive comparison |
| `enable_fes_cor_scored.csv` | Final household dataset with all scores |

## Key wording

"The current UK-only ENABLE analysis does not identify household-level variation
in FES because annual FES is common to all UK respondents.  Therefore, FES is
interpreted as a contextual annual macro-stress environment, while the estimable
behavioural mechanism is the COR pathway linking insecurity, resource
preservation, and discomfort."
