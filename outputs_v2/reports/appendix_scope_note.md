# Appendix scope note (Stage 7)

*analysis_plan_rerun.md, Stage 7. Written 2026-09-26.*

- **Moved to an appendix, not re-estimated in v2:** the COR-CVAE (latent resource representation and
  its alignment with the COR-SEM factors), fuzzy c-means vulnerability membership, and the one-class SVM
  anomaly scores. They are exploratory v1 results computed on the v1 outcome (which dropped off-gas-grid
  households and zero-filled non-response), the v1 second-order SEM (abandoned in v2), and v1 FES timing.
  They must be labelled as v1 exploratory analyses in the appendix, and no main-text claim rests on them.
  Source: tag `submitted-draft-v1`, `outputs/ukhls_cor_cvae/`, `outputs/ukhls_vulnerability/`.
  Re-estimating them on the v2 outcome is possible but was not part of the plan.
- **Dropped:** the vector-shift map (CVAE counterfactual FES shift). Not reported anywhere.
- **Dropped with the SEM route:** the second-order baseline-resource factor and its map (replaced by
  unit-weighted formative composites, Stage 4).
- **Retained, relabelled:** the equivalisation check, now "sensitivity to equivalising income only"
  (fuel spend is not equivalised). Recomputed on the v2 outcome: `outputs_v2/stage7/`.
- **Not re-estimated in v2 (v1 figures to drop or replace):** regional driver models by region
  (v1 Fig. 4-36, built on the v1 strain composite); v1 forward-risk maps by month and region and the
  "1.1% above 50% risk" figure (v1 Stage 5 scoring of wave o; v2 Stage 6 evaluates prediction on held-out
  transitions and does not score a future wave).
