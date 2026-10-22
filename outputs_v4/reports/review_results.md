# A10 review analyses: results (EXPLORATORY, POST HOC)

Amendment A10 (`analysis_plan_rerun.md`, 2026-10-22). Code: `scripts/stage9_review.py`; Stage 8
models from `fit_stage8()` / `validation_predictions()` in `scripts/stage8_anticipation.py`.
Tables: `outputs_v4/review/`. All A10 analyses were specified after the v2 and v3 results were
seen. Deviations from the A10 text are listed in §9 of the plan, dated 2026-10-22.

**Common design.** Stage 8 common sample: 190,292 linked transitions; 170,420 training (a→b … l→m)
and 19,872 validation (m→n, n→o), of which 18,122 are incident (not vulnerable at t). Validation
prevalence is 10.57% (incident 7.52%). The refactored `validation_predictions()` reproduces
`outputs_v3/anticipation/metrics.csv` to 4 decimals for all 8 models × {all, incident} (AUC,
PR-AUC, n). Inference uses 2,000 wave-t PSU bootstraps, seed 20261022, paired. Because the seed
differs from Stage 8's (20260929), CIs differ from `outputs_v3` in the third decimal; point
estimates are identical.

## 1. Sample flow (`sample_flow.csv`)

- The Stage 8 flow is reproduced exactly (common sample 190,292).
- INC models: `ieqmoecd_dv` is missing for 84 transitions (69 training, 15 validation), filled
  with the training household-size median. No transitions are dropped.
- §5: 325,966 household-waves are eligible for the spend items; 39,064 (12.0%) are item
  nonresponse. V is imputed for 38,288 of them (the rest fail the £1,200 income guard).
- §6: 7,676 Stage 6 transitions with a 2009 wave-t interview have no December vintage before T
  and are dropped from all four §6 models (A10 sample 183,179; validation unchanged at 19,960).
- §7: 151 transitions without `V_eq` at t or t+1 are dropped (validation 19,843).

## 2. Decision curve analysis (`dca_curves.csv`, `dca_contrasts.csv`, `figures/dca.*`)

- A_full has the highest net benefit (NB) of any valid model across most of the threshold range.
  At p_t = 0.10 (all transitions), NB per 1,000 households is A_full 48.6, A_naive_full 47.5,
  P3 40.2, P0 31.4, INC_logit 32.8, INC_rule 22.8 and treat-all 6.3. A_full beats both treat-all
  and treat-none at 38 of 40 thresholds (all) and 31 of 40 (incident).
- **A_full − A_naive_full** (value of the price forecast, with characteristics): +1.1 per 1,000 at
  p_t = 0.10 [95% CI 0.3, 2.0] and +1.2 at 0.20 [0.5, 1.9]. The CI includes 0 at 0.05, 0.25 and
  0.30. Incident: +1.2 [0.3, 2.2] at 0.10 and +1.3 [0.6, 2.0] at 0.20.
- **A_full − P3**: +8.4 per 1,000 at p_t = 0.10 [6.6, 10.3] and +3.1 at 0.20 [2.0, 4.2]. Incident:
  +9.3 [7.2, 11.3] at 0.10. The CI includes 0 only at p_t = 0.30.
- **Income-based targeting.** A_full − INC_logit is +15.8 per 1,000 at 0.10 [13.2, 18.4] (all).
  Incident: +8.2 [6.1, 10.5] at 0.10, with the CI including 0 from p_t = 0.20. A_full − INC_rule is
  +25.9 at 0.10 [22.9, 29.0]. INC_rule (lowest training quintile of equivalised income) treats
  9.4% of validation households, with sensitivity 28.2% and PPV 31.9% (incident: 13.5% and 19.6%).
- **P0 − INC_logit** is mixed. On all transitions P0 is better at 0.05 and at 0.15–0.30 but not at
  0.10 (−1.4 [−3.9, 1.0]). On incident transitions INC_logit is better at 0.10–0.20 (e.g. −10.7
  [−12.7, −8.6] at 0.10), because P0 gives every incident household the same flag and only its
  burden varies.
- A_oracle (realised prices; an upper bound, not a usable predictor) is the highest curve at
  about half of the thresholds.

## 3. Brier score and paired metric differences (`metrics_extra.csv`, `delta_metrics.csv`)

- Brier scores on all transitions: A_full 0.0810 [0.0777, 0.0844], A_naive_full 0.0813, P3
  0.0832, P0 0.0860, INC_logit 0.0893. Scaled Brier: A_full 0.152, P3 0.129, P0 0.099, INC_logit
  0.065. Incident: A_full 0.0658 (scaled 0.054); P0's scaled Brier is −0.0001.
- INC_logit AUC is 0.759 [0.748, 0.771] and PR-AUC 0.287 (all). Incident: AUC 0.708, PR-AUC
  0.170. INC_rule AUC (0/1 indicator) is 0.606 (all) and 0.545 (incident).
- Paired deltas on all transitions, AUC / PR-AUC / Brier:
  - A_naive − P0: +0.0030 [0.0012, 0.0047] / +0.0043 / −0.0030
  - A_fut − A_naive: +0.0027 [0.0014, 0.0040] / +0.0030 [0.0008, 0.0052] / −0.0004 [−0.0006, −0.0003]
  - A_naive_full − P3: +0.0128 [0.0067, 0.0190]
  - A_full − A_naive_full: +0.0020 [0.0011, 0.0028] / +0.0029 / −0.0004
  - A_full − P3: +0.0147 [0.0087, 0.0210]
  - A_oracle − A_naive: +0.0069 [0.0036, 0.0103]

  Every AUC/PR-AUC delta is > 0 with a CI excluding 0, and every Brier delta is < 0 with a CI
  excluding 0, on both subsets. One borderline case: the incident PR-AUC of A_naive − P0 has
  CI [0.0001, 0.0163].
- Calibration (all): slopes 0.78 (A_naive, P0) to 0.93 (P1); A_full 0.84 [0.81, 0.88]. All models
  under-predict (calibration-in-the-large 0.14 to 0.77), consistent with the 2022–23 rise in
  prevalence after the training period. Incident: A_full slope 0.99 [0.92, 1.06], CITL 0.44.
  P0's incident slope is 4.74, because its incident predictions vary only through the burden at t.

## 4. Construct validation (`variable_availability.csv`, `construct_*.csv`)

- **Items found.** `xphsdba` (bill arrears; all waves); `hheat` (accommodation adequately heated;
  waves a,b,d,f,h,i,j,k,l,n; no affordability reason); `pdepi2` (home not kept warm: no money;
  **pensioner module only**, waves d,f,h,j,l,n); `pdeph1` (damp-free home; pensioners, waves
  d,f,h,j,l). **Not found:** `hsprb*` / damp, condensation or rot for all households; gas- or
  electricity-specific arrears; disconnection or inability to pay for heating. The planned "warm"
  and "damp" items were therefore replaced by the closest available items (deviation log).
- **Concurrent validity** (weighted mean of per-wave rates, H if V = 1 vs V = 0):
  - arrears 12.5% vs 5.4% (ratio 2.29); adjusted OR 1.48 [1.39, 1.57], n = 269,089
  - heat_inadequate 14.1% vs 6.1% (2.31); adjusted OR 1.34 [1.26, 1.43]
  - warm_afford 3.2% vs 1.5% (2.09); adjusted OR 1.42 [1.11, 1.80], n = 33,773 pensioner
    households
  - damp_pensioner 6.0% vs 4.4% (1.37); unadjusted OR 1.45 [1.21, 1.74], adjusted OR 1.12
    [0.90, 1.38] (not significant)
- **Predictive validity** (validation, H at t+1; `b_hat_forecast_price` = projected burden at
  forecast prices):
  - arrears (n = 19,869, prevalence 6.2%): AUC 0.592 [0.574, 0.611]; A_full p̂ 0.639; INC_logit 0.627
  - heat_inadequate (m→n only, n = 9,026): 0.586 [0.566, 0.607]
  - warm_afford (m→n pensioners, n = 3,397): 0.577 [0.519, 0.632]

  Newly in arrears (H = 0 at t, n = 18,683, 2.7%): b̂ AUC 0.558 [0.531, 0.584]. On this subset INC_logit
  (0.596) is higher than A_full (0.578), with overlapping CIs.
- The projected burden adds little over the current burden b_t for these items (e.g. arrears
  b_t 0.599). The model probabilities that include P1 characteristics predict hardship better
  than the burden alone.
- **Decision rule: met.** Three items (arrears, heat_inadequate, warm_afford) are each associated
  with V at P < 0.05 after adjustment, and each is predicted at t+1 by the projected burden with a
  95% CI above 0.5. Caveats: two of the three items are substitutes for the planned definitions,
  and warm_afford covers pensioners only.

## 5. Missing fuel spending (`missing_by_hardship.csv`, `missingness_model.csv`, `ipw_*.csv`, `mi_*.csv`)

- **Who is missing.** Nonresponse rises from 7–8% (waves b–e) to 20.6% (wave o). It is 15.5% in
  private renting vs 11.0% for outright owners, and 14.0% in income Q1 vs 10.7–12.2% in Q2–Q5.
  It does not rise with financial difficulty or arrears: 10.8% when behind with bills vs 11.8%
  when up to date. Households vulnerable at the previous wave are missing at the same rate as
  non-vulnerable ones (7.3% vs 7.3%).
- **Missingness model** (adjusted ORs): previous-wave status not observed 6.10 [5.88, 6.33];
  `xphsdba` missing 32.5 (general item nonresponse); CATI mode 1.52 and web 1.22 vs face to face;
  private renting 1.40; income Q1 1.23 vs Q3; finnow 4–5 0.84–0.85 vs 1; previously vulnerable
  1.06 [0.99, 1.12]. The mode "not recorded" OR is not identified (waves a/b only).
- **IPW** (stabilised, truncated 1st/99th percentile). Prevalence changes by at most 0.41 pp per
  cell, except interview year 2025 (n = 223 partial year: −1.1 pp). NI 18.04% → 18.14% (pooled) and 14.42% → 14.81% (Apr 2021–Mar 2023 window). 2022 vs 2023:
  9.28/14.66 → 9.49/14.76. Driver ORs barely move (finnow 1.649 → 1.641; fes_delta_growth3 0.972
  → 0.971; jbstat_security 0.174 → 0.176). NI AME in model (b): 2.39 pp → 2.26 pp [1.27, 3.24];
  oil AME 5.84 → 5.82 pp. Validation AUCs change by ≤ 0.001 (A_full 0.7956 → 0.7952).
- **MI** (20 chained-equation imputations). Prevalence is 0.1–0.8 pp higher in every cell (e.g.
  NI 18.34% [17.10, 19.58]; 2023 14.89% [14.18, 15.60] vs 2022 9.56% [8.91, 10.21]). Imputed
  households are therefore somewhat more burdened than observed ones. Driver ORs are slightly
  attenuated: finnow 1.600 [1.562, 1.638], jbstat_security 0.191, lone_parent 1.432. Share of V
  imputed: 11.0% in the driver model and 5–22% per prevalence cell (highest in waves n–o).
  FMI: 0.06–0.12 (driver ORs), 0.02–0.20 (prevalence).
- **Decision rule: met.** (i) NI is ranked first in both the pooled and the window ranking, and
  2023 > 2022, under both IPW and MI. (ii) All six driver ORs stay on the same side of 1 with CIs
  overlapping the primary CIs, under both IPW and MI. (iii) The IPW AUC ordering
  P1 < P0 < P3 < A_naive_full < A_full is unchanged on both subsets.

## 6. Frozen-information P2 (`p2_frozen.csv`, `p2_frozen_lead_time.csv`)

- Check: P1 and P2 refitted on the Stage 6 common sample reproduce `outputs_v2/stage6` (AUC 0.7391
  and 0.7385).
- On the A10 sample (2009 wave-t interviews dropped; validation n = 19,960), AUC:
  - P1 0.7392 [0.7274, 0.7515]
  - P2 as published 0.7383 (Δ vs P1 −0.0009 [−0.0014, −0.0004])
  - P2_frozen 0.7368 (Δ −0.0024 [−0.0028, −0.0020])
  - P2_fixed12 0.7365 (Δ −0.0027 [−0.0031, −0.0022])
- Mean lead time from the vintage origin to the t+1 interview: 6.0 months as published (1–12) vs
  17.9 months frozen (6–29).
- Restricting the national index to information available at the wave-t interview makes it
  slightly *worse* than P1, not better. The published P2's null result is therefore not an
  artefact of its timing. No decision rule was pre-specified for this section.

## 7. Equivalisation sensitivity (`equiv_prediction.csv`)

- V_eq prevalence is 25.3% (all) and 17.4% (incident), against 10.6% / 7.5% for V on the same
  sample.
- V_eq AUC (all): P0 0.756, P1 0.678, P3 0.757, A_naive_full 0.767, A_full 0.771 [0.762, 0.779].
  Incident: P0 0.675, A_full 0.699 [0.689, 0.711].
- A_full − A_naive_full: +0.0032 [0.0024, 0.0039] (all) and +0.0055 [0.0040, 0.0070] (incident),
  against +0.0020 / +0.0030 for V.
- A_full − P3: +0.0134 [0.0096, 0.0178] (all) and +0.0250 [0.0170, 0.0337] (incident).
- The ranking and the sign of every contrast are unchanged under equivalised income, at lower
  AUC levels. No decision rule was pre-specified for this section.

## Not found or not run

- Damp, condensation or rot (`hsprb*`), fuel-specific arrears, and disconnection or inability to
  pay for heating: not in the licensed `hhresp`/`indresp` files. The pensioner damp-free item was
  used as a proxy, for concurrent validity only.
- Predictive validity for damp: not possible, because the item is not asked in waves n/o. The
  "newly hardshipped" subset for heat_inadequate and warm_afford: not possible, because neither is
  asked in wave m.
- An individual-level hardship item (`indresp`) was searched for but none matched, so no
  household aggregation was needed.
- The palette validator of the plotting guidelines could not run (no Node.js). The DCA figure
  uses the Okabe–Ito colour-blind-safe hues with line style as a second cue.
