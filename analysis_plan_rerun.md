# Analysis plan — rerun v2

Fixed: 2026-09-26, before any code change (see git history: this file is committed
on branch `rerun-v2` ahead of every code commit).
Baseline for comparison: tag `submitted-draft-v1` (commit `bb4791f`), which matches
the submitted thesis PDF.

**Principle.** Keep the research questions (RQ1–RQ5). Recompute the evidence. Do
not carry over the old conclusions. Any specification changed after results are
seen is logged in §9 with the reason and the date, and is labelled post hoc
in the thesis.

**Output rule.** All new outputs go to `outputs_v2/`. `outputs/` is read-only
reference material; nothing in it may be overwritten.

Items marked **[proposed]** were not in the supervisor/author rerun plan. They are
defaults filled in so that the specification is complete before results exist.
They need approval at the end of Stage 0. Changing them later counts as a
deviation (§9).

---

## Stage 1 — Outcome audit (no fixes)

Variables: `xpduely`, `xpgasy`, `xpelecy`, `xpoily`, `xpsfly`, `fuelduel`, plus
`fuelhave1–4, 96` as routing context.

1. For every wave a–o, list each raw value < 0 and each coded value, with its UKHLS
   value label. Classify each code as:
   - not applicable / routed out (e.g. −8 inapplicable);
   - item nonresponse (−1 don't know, −2 refused, −9 missing);
   - other (−7 proxy, −10 not available for IEMB, and so on).
2. Compare the current code's treatment of each code (kept as 0, set to NaN, or
   the row dropped) with the correct treatment. Count the values that are
   currently zero or dropped but are really missing, by region × wave.
3. `fuelduel` frequencies by region × interview year.
4. Northern Ireland: count oil households (`fuelhave3 == 1`) lost to the
   `fuelduel`-NaN rule, by wave.
5. Deliverable: `outputs_v2/audit_fuel_codes.csv` plus a one-paragraph verdict
   (the outcome is correct, or it needs fixing). If a fix is needed, the fixed
   outcome becomes the primary outcome for every later stage, and the old outcome
   is reported only as a comparison.

### Amendment A1 (2026-09-26): fixed outcome, set after the Stage 1 audit

Status: approved by the author on 2026-09-26. Indicative Stage 1 prevalence had
already been seen (logged in §9). Committed before any Stage 2 code.

**Primary outcome, v2.** Spend is the routing-aware sum:

- **electricity + gas households:**
  - `fuelduel` = 1 → `xpduely`;
  - `fuelduel` = 2, or don't know / refused → `xpgasy` + `xpelecy`.
- **electricity-only households:** `xpelecy`.
- **plus oil and other fuel:** `xpoily` if `fuelhave3` = 1, and `xpsfly` if
  `fuelhave4` = 1. Otherwise these are structural zeros (−8).

Rules:

1. **Item nonresponse** (−1, −2, −9) on any required amount means the household's
   spend is **missing**. The primary analysis is complete-case.
   - *Sensitivity S1, lower bound:* those amounts are set to £0 (the v1
     zero-fill), with the routing otherwise correct.
2. **Electricity not reported** (`fuelhave1` ≠ 1 while some other fuel is
   reported): this covers gas-only households *and* oil- or other-fuel-only
   households. Their spend is probably incomplete, so they are **excluded from the
   primary analysis**.
   - *Sensitivity S2* includes both groups, with spend as reported.
3. **Excluded from all versions:** households where the fuel-use module is
   nonresponse, and households reporting no fuel.
4. **Denominator and guards unchanged from v1:** 12 × `fihhmnnet1_dv`. The ratio is
   set to NaN if annual income is below £1,200, and capped at 1.0. The
   high-vulnerability threshold stays at 10%.
5. **Oil use** is defined as `fuelhave3` = 1, never as `xpoily` > 0.

**Additional Stage 1 deliverables (audit, no inference):**

- A sample-flow reconciliation from 339,201 rows to the analytical n, one line per
  exclusion, including the £1,200 income guard.
- An explanation of the 1-row and ~2,100-row gaps in the Stage 1 counts.
- A table comparing missing-spend and observed-spend households by region,
  tenure (`tenure_dv`), income band (within-wave quintile of `fihhmnnet1_dv`) and
  wave.
- Fuel included in rent: `elecpay`, `gaspay` and `duelpay` code 5, available from
  waves c–o (absent in a and b). Report it, together with tenure, for the gas-only group and the
  electricity-not-reported group.

**Northern Ireland.** No NI text in the thesis is rewritten until the Stage 5
time-matched JRF comparison has been recomputed on the fixed sample. Stage 5
reports:

- the NI oil share;
- oil vs non-oil vulnerability rates within NI.

Both are given weighted (per-wave household `_xw` weight: `hhdenus_xw` in wave a,
`hhdenub_xw` in b–e, `hhdenui_xw` in f–m, `hhdeng2_xw` in n–o) and unweighted.

## Stage 2 — FES

- **Mode:** core only (target series only, no exogenous macro regressors).
  FES-macro, FES-selected and FES-weighted are dropped from the main text.
- **Backtest:** rolling core backtest. Hyperparameters are tuned only on data
  available before each forecast origin (per-origin tuning).
  - **Fallback if infeasible** (runtime or failures): reuse the existing
    `fes_core` from `outputs/` and log the tuning limitation in §9 and in the
    thesis.
- **Vintage:** households interviewed in calendar year Y get the forecast made
  at origin **December Y−1**. An interview in month m of Y gets that vintage's
  target month (Y, m).
  - **[proposed]** Interviews in 2009 (no Dec-2008 origin with enough history, if
    that turns out to be the case) are logged and excluded from FES models only.
- **Standardisation:** z-scores for each component use **past-only rolling**
  moments: mean and SD over data up to the forecast origin only.
  - **[proposed]** Expanding window, minimum 36 months.
- **FES Delta:** anticipated FES magnitude minus the realised FES exposure
  observable at the interview. Both use the same past-only z-scores.
- **Forecast-vs-realised evaluation:** on the three growth terms only (gas,
  electricity, carbon). The uncertainty term is excluded because it has no
  realised counterpart.
  - Metrics: RMSE, MAE, sign agreement, Pearson r, per series and per target
    year.

## Stage 3 — Driver model (logit, outcome `high_fuel_vulnerable`)

- **Primary:** financial strain composite **without `xphsdba`**, i.e.
  mean(`finnow`, `finfut_risk`, `scghq1_dv`), with each component scaled as in v1.
  `bill_security` stays as a separate predictor, which removes the double count.
  All other predictors are as in v1, plus **year fixed effects** (interview year).
- **SEs:** clustered on `psu`. Use two-way clustering on `psu` × interview month
  if the estimator supports it. If it doesn't, cluster on `psu` only and log it.
- **Sensitivity A:** the original strain composite (with `xphsdba`).
- **Sensitivity B:** lagged strain (the household's strain at its previous wave,
  linked through `hrpid` as in Stage 5) in place of contemporaneous strain.
- **Sensitivity C:** primary + calendar-month fixed effects.
- **Reporting:** a full table with OR, 95% CI, p, N and McFadden pseudo-R² for
  every model. State the reason rows are dropped from 339,201 to N.
- **Label fix:** `health_good` = *no long-standing illness or disability*
  (UKHLS `health`). `sf1_good` = *self-rated general health* (UKHLS `sf1`).
  Relabel every table and figure.
- **[proposed]** Identification note: with year FE, the FES Delta coefficient
  comes only from within-year variation across interview months. This is
  reported as a limitation, not fixed.

## Stage 4 — Resources and H1

- Fit a **correlated four-factor first-order CFA** (OBJECT, CONDITION, PERSONAL,
  ENERGY) by **complete-case ML**. There is no second-order BASELINE factor.
- **Orientation:** positive marker indicators, not income:
  - **[proposed]** OBJECT → `hsrooms`, CONDITION → `tenure_security`,
    PERSONAL → `sf1_good`, ENERGY → `fihhmnnet1_dv`.
  - Each marker is fixed at loading 1, and higher always means more resource.
- **Fit criteria [proposed]:** the CFA is defensible if CFI ≥ 0.90, RMSEA ≤ 0.08,
  SRMR ≤ 0.08, there are no Heywood cases, and all standardised loadings are
  ≥ 0.30 with the expected sign.
  - **One attempt only.** If the fit fails, use **unit-weighted standardised
    composites** (mean of z-scored indicators per domain) instead. Report the
    failed fit.
- **H1 primary:** resource composite R = OBJECT + CONDITION + PERSONAL
  (sum of standardised domain scores). Model:
  - **[proposed]** OLS of `fuel_to_income_ratio` on R, FES Delta, R × FES Delta
    and year FE, with SEs clustered on `psu`. The sample and outcome are those
    fixed in Stage 1.
- **H1 sensitivity:** R including ENERGY. **[proposed]** Also a logit on
  `high_fuel_vulnerable` with the same terms.
- **H1 decision rule [proposed]:** ~~*Supported* if the interaction is negative with
  p < 0.05; *contradicted* if positive with p < 0.05; *not supported* otherwise.~~
  **Superseded 2026-09-26 (see §9):** the direction was written before FES Delta's
  sign convention was fixed.
  - **Current rule.** FES Delta = forecast − realised, and R is oriented so higher =
    more resources. COR predicts a **positive** R × Delta interaction: resources
    flatten the negative Delta slope.
  - *Supported* if the interaction is positive with p < 0.05 under the primary
    specification.
  - *Contrary to COR* if it is negative with p < 0.05.
  - *Not supported* otherwise.
  - Also report predicted Delta slopes, with 95% CIs, at low, median and high R
    (10th, 50th and 90th percentiles of R in the estimation sample).
  - Primary FES term = `fes_delta_growth3`.
  - Report the sensitivity results beside the primary, not in place of it.
- Regenerate **only** the maps that depend on the resource score: the
  baseline-resource map and the resource × vulnerability hotspot map.

## Stage 5 — JRF external comparison

- **Metadata table:** one row per comparison, with columns for
  - dimension;
  - JRF population (households / individuals / children);
  - JRF measure;
  - JRF period and page;
  - matching UKHLS window;
  - UKHLS unit.
  Saved as `outputs_v2/jrf_metadata.csv`.
- **Rates:** weighted per-wave rates using each wave's household cross-sectional
  weight (`hhdenus_xw` in wave a; later waves use the wave-specific `_xw` name,
  resolved and listed in the metadata). Rates are then averaged over the matching
  window.
- **Work status:** JRF values corrected to workless **54%** and in-work **15%**
  (author-supplied). The source page is recorded in the metadata table and
  checked against `UK Poverty 2025.pdf`.
- **Disability:** fix the sample. The v1 subgroup rates were both above the pooled
  rate, which means the denominators were wrong.
  - **[proposed]** Unit = household-wave with disability status observed for at
    least one adult. Report the excluded count.
- **Masking:** cells with unweighted n < 100 are masked in every table and map,
  including the NI 2024 heatmap cell.

## Stage 6 — Next-wave prediction

- Same `hrpid`-linked transitions and walk-forward split as v1: train on the
  earlier transitions, validate on m→n and n→o.
- Nested models:
  - **P0:** current burden only (`fuel_to_income_ratio` at t).
    **[proposed]** Also the `high_fuel_vulnerable` flag at t.
  - **P1:** P0 + social variables (financial strain, age, `heatch`, lone parent,
    large family, workless, resource composites from Stage 4).
  - **P2:** P1 + FES (magnitude and Delta from Stage 2).
- **Metrics, per model:**
  - ROC-AUC with 95% CI (**[proposed]** 1,000 bootstrap resamples, clustered by
    household);
  - PR-AUC and the prevalence baseline;
  - calibration slope and intercept on the validation set;
  - AUC for each validation transition separately;
  - sensitivity and PPV at the top 5% and top 10% of predicted risk;
  - ΔAUC P1−P0 and P2−P1 with bootstrap CIs.

### Amendment A6 (2026-09-26): Stage 6 specification, fixed before any Stage 6 code

- **Transitions.** Consecutive-wave pairs t → t+1 linked by `hrpid`. Reference
  persons appearing twice within a wave are excluded, as in v1. Outcome:
  `high_fuel_vulnerable` (A1 primary) at t+1.
- **Split.** Train on transitions a→b … l→m; validate on m→n and n→o.
- **P0 (benchmark only, not merged into the main model):** `fuel_to_income_ratio`
  and `high_fuel_vulnerable` at t.
- **P1 (main model):** household predictors at t:
  - `finnow`, `scghq1_dv`, `finfut_risk` (strain components separately, as in
    Stage 3);
  - `dvage`, `heatch`, `lone_parent`, `large_family`, `workless_household`;
  - the four resource domain composites (OBJECT, CONDITION, PERSONAL, ENERGY),
    rebuilt as in (a).
- **P2 = P1 + `fes_magnitude_growth3`** for the household's t+1 interview month.
  This is the December (Y_{t+1} − 1) vintage, published before the outcome year
  begins. The t+1 interview month is a scheduling quantity, not outcome
  information.
  - *Sensitivity P2b:* P2 + `fes_delta_growth3` observed at wave t, on its own
    common sample. It loses transitions whose wave-t interview was in 2009,
    which have no FES.
- **(a) Standardisation uses training-transition statistics only.**
  - Continuous predictors are z-scored with training means and SDs.
  - Resource composites are rebuilt: items z-scored with training statistics
    (monetary items log(1 + max(x, 0)) first); the domain mean is taken where
    ≥ 50% of items are observed; the domain score is re-standardised with
    training statistics.
  - FES terms are z-scored with training statistics.
  - Binary predictors are left unscaled.
- **(b) One common sample.** P0, P1 and P2 are fitted and compared on
  transitions with the t+1 outcome and all P0, P1 and P2 predictors observed.
  Report the number of linked transitions, the exclusions by reason, and N for
  training and validation.
- **Estimator.** Unpenalised logistic regression.
- **Metrics (validation set).**
  - ROC-AUC with a 95% CI from 2,000 bootstrap replicates resampling wave-t PSUs.
  - PR-AUC and the prevalence baseline.
  - Calibration slope (coefficient on logit p) and calibration-in-the-large
    (intercept with logit p as an offset).
  - AUC for each validation transition separately.
  - Sensitivity and PPV at the top 5% and top 10% of predicted risk.
  - Paired bootstrap ΔAUC for P1 − P0 and P2 − P1.

## Stage 7 — Scope

- CVAE, fuzzy c-means and one-class SVM move to an appendix. No main-text claims
  rest on them.
- The vector-shift map is dropped.
- The equivalisation check is kept, relabelled *"sensitivity to equivalising
  income only"* (the fuel spend is not equivalised).

## 8. Reporting after each stage (2–7)

After each stage, report:

- the commit hash;
- the files changed;
- a table of old vs new key numbers;
- deviations from this plan.

The work stops after each stage for approval.

## 9. Deviation log

| Date | Stage | Plan said | Did instead | Reason | Seen results first? |
|---|---|---|---|---|---|
| 2026-09-26 | 1 | "If a fix is needed, the fixed outcome becomes primary" (fix not specified) | Amendment A1 defines the fixed outcome and sensitivities S1 and S2 | The Stage 1 audit found that v1 zero-fills nonresponse and drops routed-out (−8) `fuelduel` households | Yes: indicative prevalence under a draft rule (UK 8.89% vs 7.93%) |
| 2026-09-26 | 1 | (not in plan) | `prepayment_meter` routes electricity-only households to `elecpay`, the same routing fix as A1: coverage rises from 223,494 to 264,542 rows | Same `fuelduel` = −8 bug as the outcome. It feeds thesis Fig. 4-35 | No: prepayment rates not yet computed |
| 2026-09-26 | 2 | "Realised FES exposure observable at the interview" | Interpreted as realised growth in month m−1, the last complete month before the interview month m. It is z-scored with the same past-only moments as the forecast (expanding window through the Dec Y−1 origin) | Month m is not complete at the interview | No |
| 2026-09-26 | 2 | FES = 3 growth z-terms + uncertainty (eq. 1) | Primary `fes_magnitude` / `fes_delta` keep the 4-term index. Added sensitivity: `fes_magnitude_growth3` / `fes_delta_growth3` (growth terms only), because FES Current has no uncertainty term | Delta subtracts a 3-term sum from a 4-term sum | No |
| 2026-09-26 | 2 | [proposed] 2009 interviews excluded if no Dec-2008 origin | Confirmed: as_of 2008 has 20 months of history before its validation year (< 24 required), so the first origin is Dec 2009 → 2010 interviews. All 2009 interviews have no FES | Data start May 2006 | No |
| 2026-09-26 | 5 | Fix the disability *sample* | Also replaced the *variable*. v1 built disability from `healthlink`, which is the adult health-record-linkage consent (label "adult health consent capi"), not activity limitation. It exists only in wave a, so v1's disability comparison was 2009-only. v2: disabled = `health` = 1 and any of `disdif1`–`12` mentioned (Equality-Act/FRS style). Household = contains a disabled adult if any observed adult is disabled (unit as proposed). Observed for 97–99.8% of households per wave | Wrong variable found while fixing the sample | Partly: v1 rates had been seen; v2 rates computed only after the definition was fixed |
| 2026-09-26 | 5 | Matching UKHLS window per JRF period (unspecified) | Region and ethnicity: interviews Apr 2020–Mar 2023 (HBAI 3-year average; JRF Table 6 title '2021–2023' read as FYE 2021–2023). Tenure, disability, family and work: Apr 2022–Mar 2023. Work status restricted to households with ≥1 respondent aged 16–64 (JRF population: working-age adults). Family type: lone parent vs couple, any size. Window rate = per-wave weighted rates averaged by in-window n | Definitions needed to time-match | Set before computing v2 comparison rates |
| 2026-09-26 | 3/4 | (data cleaning; not in plan) | (a) Added UKHLS codes −10/−11/−20/−21 to the missing codes: wave f carried `ncars` = −10 and `carval` = −10 for 2,468 households as real values in v1. (b) `sf1_good` now uses self-completion `scsf1` where valid, else `sf1`: v1 read `sf1` only, which is <11% observed after wave e, so self-rated health was effectively waves a–e only. Coverage is now 95–100% per wave. The outcome is unchanged (verified) | Input errors found while preparing the Stage 4 CFA. Both variables feed the driver model and the resource model | No: found before any v2 resource or driver model was fitted |
| 2026-09-26 | 4 | CFA/composites on indicators as coded | Monetary indicators (`hsval`, `carval`, `fihhmnnet1_dv`, `fiyrinvinc_dv`) transformed with log(1 + max(x, 0)) before standardising, for the CFA and the composites alike. All indicators are z-scored on the estimation sample. Composites: mean of available standardised items in a domain if ≥ 50% of that domain's items are observed, then the domain score is re-standardised | Extreme skew (`hsval` max £70M) violates ML normality and would dominate unit-weighted composites | No: set before any Stage 4 fit |
| 2026-09-26 | 3 | Primary: strain composite without `xphsdba` | **Primary driver specification enters the strain components separately:** `finnow` (current financial difficulty, 1–5), `scghq1_dv` (GHQ-12 Likert, 0–36) and `finfut_risk` (0 better / 0.5 same / 1 worse, household mean), with `bill_security` as before. Sensitivities: the composite without `xphsdba`, and the v1 composite with `xphsdba`. The lagged-strain check likewise uses the three lagged components separately, with lagged composites as sensitivity. `finfut` coding verified against the UKHLS labels | Composite α = 0.27. `finnow`–`finfut_risk` r = 0.02 (household level) and 0.03 (person level). `finfut_risk` tracks age (r = 0.32), so the items are not one scale. Author decision 2026-09-26 | Yes: item correlations and α seen; no driver model fitted |
| 2026-09-26 | 4 | Composites as fallback measures of latent resources | The unit-weighted composites are described as **formative indices**. α is reported as description, not as a validity test. **The hotspot tier classification (3×3 tertile map) is dropped.** Regional resources are shown as continuous composite values (weighted mean, `resource_composite_by_region`) | CFA failed. Tertiles over 12 regions exaggerate small differences. Author decision 2026-09-26 | Yes: CFA and map seen |
| 2026-09-26 | all | (not in plan) | Small-cell suppression on every tracked `outputs_v2` aggregate table (`scripts/suppress_small_cells.py`): counts 1–9 → '<10'; rates with denominator < 10 or implied numerator 1–9 suppressed; identifiers removed. `--check` must pass before any output commit. Primary suppression only: no secondary suppression against differencing across tables | Author decision 2026-09-26 (statistical disclosure control before pushing) | n/a |
| 2026-09-26 | 5 | Region/ethnicity window Apr 2020–Mar 2023 (my earlier reading of JRF Table 6) | **Corrected:** primary window = interviews **Apr 2021–Mar 2023**, because DWP excludes 2020/21 from 3-year averages (JRF note p.43 and p.47; Annex p.162). Apr 2020–Mar 2023 kept as sensitivity. Tenure, family type, work status and disability stay FY 2022/23 (Apr 2022–Mar 2023). Region CIs and ranks reported for both windows | Author correction 2026-09-26, confirmed in the PDF | Yes: the Apr 2020–Mar 2023 results had been seen |
| 2026-09-26 | 3 | `finfut_risk` as a strain component | Interpreted and labelled as **financial expectations** (0 better / 0.5 same / 1 worse), always estimated with age controlled (`dvage` is in every driver specification) | `finfut_risk` correlates with age (r = 0.32), not with current difficulty. Author decision | Yes: item correlations seen; no driver model fitted |
| 2026-09-26 | 3 | Driver model predictors as in v1 (+ year FE) | Add **region fixed effects** (reference: South East) and an **oil-heating indicator** (`fuelhave3` = 1). Report the NI coefficient in three nested models: (a) region FE, no oil terms; (b) + oil main effect; (c) + NI × oil. **In (b) and (c), also fit a version adding `urban_dv` (urban/rural household location).** Report the oil coefficient (and NI × oil) with and without `urban_dv`. `urban_dv` is coded 1 urban / 2 rural in all waves. Missing in 330 wave-k NI households (29% of NI in that wave). **Imputation of missing `urban_dv`:** fill from the same household's value in the adjacent wave, linked by `hrpid` as in Stage 5; prefer the previous wave, else the next wave. Only where there is no evidence of a move between the two interviews. Evidence of a move is any of: (i) `gor_dv` differs between the two waves; (ii) `origadd` switches between 1 and 2 (waves b–o); (iii) the reference person reports a move-in date (`mvyr`/`mvmnth` in the later wave) on or after the earlier wave's interview month. `mvyr` is asked mainly of movers and new entrants, so its absence is not treated as proof of no move, only as absence of evidence. Report the number recovered from the previous wave, from the next wave and still missing, overall and for NI wave k. Models with `urban_dv` use the filled variable; a sensitivity uses observed `urban_dv` only | Tests whether oil dependence accounts for NI's excess vulnerability, and whether the oil effect is a rurality effect (UKHLS NI sample ≈ 50% rural vs ≈ 22% elsewhere). v1 had no region terms. Author decisions 2026-09-26 | Descriptive NI oil rates and urban/rural frequencies seen; no driver model fitted |
| 2026-09-26 | 2/5 | Interview timing from `interview_year`/`interview_month` (as in v1) | **These now come from the actual household interview date** (`intdatey`/`intdatem`), not the UKHLS *sample month* (`month`, the address issue month) that v1 used. Sample year/month are kept as `sample_year`/`sample_month`. Affects the FES vintage and month (Stage 2), the JRF time windows (Stage 5) and by-year tables. Only 24–75% of households per wave are interviewed in their sample month; 3.5–10% in a different calendar year. The interview date is missing for 7 rows (sample month used). 300 interviews fall in 2025 | Timing error found while attaching FES (2,468 wave-f households had no sample month) | No: found before any FES-dependent model; JRF and by-year tables rerun |
| 2026-09-26 | 2/3/6 | Primary FES = 4-term index (3 growth z-terms + forecast uncertainty); growth-only as sensitivity | **Primary FES = growth-only 3-term index** (`fes_magnitude_growth3`, `fes_delta_growth3`) in every household model (Stages 3, 4-H1, 6). The 4-term index (`fes_magnitude`, `fes_delta`) becomes the sensitivity | v2 95% prediction-interval coverage is 28–48% (gas 32%, electricity 48%, carbon 28%), so the uncertainty term is mis-scaled. Author decision 2026-09-26, based on forecast diagnostics only | No: forecast diagnostics only; no household model had been fitted |
| 2026-09-26 | 3 | Two-way clustering on PSU × 'interview month' | Second dimension = interview **year-month** (the unit at which FES varies, ≈185 clusters). Calendar month alone would give only 12 clusters. Observed-only `urban_dv` variant fitted for the primary specification only; the filled `urban_dv` block is fitted for every specification. Nested NI-oil models share one estimation sample within each block | Clarification of scope | No: set in the script before the first fit |
| 2026-09-26 | 4 | H1 supported if the R × Delta interaction is **negative** (p < 0.05) | **Supported if positive** (p < 0.05); significant negative = 'contrary to COR'; otherwise 'not supported'. Plus predicted Delta slopes at R p10/p50/p90 | Delta = forecast − realised and R is oriented higher = more resources, so COR buffering means resources flatten the negative Delta slope, i.e. a positive interaction. The original rule predates the fixed sign convention. Author decision | No: set before any Stage 4 H1 fit. Stage 3 main-effect Delta results had been seen |
| 2026-09-26 | 3 | (not in plan) | Added sensitivity `sens_no_qualification`: the primary specification without `qfhigh_band` | Highest qualification is the only missing control for 26,925 of the 52,251 households lost to complete-case estimation (61%; 11.7% missing among 274,128 with outcome and FES). Author decision | Yes: the primary model had been seen |
| 2026-09-26 | 3 | NI-oil decomposition reported as the share of log-odds | NI gap reported as **average marginal effects** (percentage points, NI vs South East, delta-method CIs with PSU-clustered covariance) for models a, b, b+rural, c, c+rural; log-odds share not used in the thesis. Also AME of oil. Per-SD comparison table of continuous predictors, with the equivalence scale flagged as partly mechanical | Author decision | Yes: Stage 3 ORs had been seen |
| 2026-09-26 | 6 | P1 = P0 + social variables | **P0 is a benchmark only; P1 excludes current burden** (A6) | Author decision: compare social prediction against the current-burden benchmark rather than nesting it | Stage 3–5 results seen; no Stage 6 model fitted |
| 2026-09-26 | 6 | P1 'financial strain' (composite) | Strain components entered separately in P1 (A6), consistent with the Stage 3 decision | Composite is not a coherent scale (α = 0.27) | No Stage 6 model fitted |
| 2026-09-26 | 6 | P2 = P1 + FES (terms unspecified) | P2 = P1 + `fes_magnitude_growth3` at the t+1 interview month (Dec Y_{t+1} − 1 vintage); P2b adds `fes_delta_growth3` at t | Pre-specification of the FES term and its timing (author request) | No Stage 6 model fitted |
| 2026-09-26 | 6 | Standardisation unspecified | All predictor standardisation and resource composites use training-transition statistics only (A6 (a)) | Avoid leakage from validation waves | No Stage 6 model fitted |
| 2026-09-26 | 6 | Sample per model | One common sample for P0/P1/P2 (A6 (b)); N and exclusions reported | Comparability of metrics across models | No Stage 6 model fitted |
| 2026-09-26 | 6 | (not in plan) | **POST-HOC / EXPLORATORY:** P3 = P0 + P1 on the A6 common sample, same split and training-only standardisation. Reported: ΔAUC vs P0 and vs P1 (paired PSU bootstrap), PR-AUC, top-10% sensitivity/PPV. Outputs are labelled `posthoc_`. No conclusion of the pre-specified analysis depends on it | Author request after seeing Stage 6 (P0 > P1) | **Yes**: Stage 6 results seen |
| 2026-09-26 | 6 | H5 verdict wording (unspecified) | H5 = **partially supported**. P2 − P1 reported as 'no improvement (ΔAUC −0.0006)', without significance language | Author decision | Yes |
| 2026-09-26 | 7 | CVAE/fuzzy/SVM to appendix | Moved to the appendix **as v1 archived exploratory results, not re-estimated** on the v2 outcome; vector-shift map dropped. Descriptive refreshes (no models) of v1 figures the thesis quotes: prepayment by status, regional early/late change, prevalence by FES tercile. Regional driver models, forward-risk maps by month/region and the '1.1% above 50%' figure are not re-estimated (to be dropped) | Re-estimation was not in the plan; analysis freeze follows Stage 7 | Yes |

## 10. Analysis freeze

Frozen 2026-09-26 after Stage 7, on the author's instruction. No new models unless the author asks. Every number the thesis quotes is in `outputs_v2/results_inventory.csv` (built by `scripts/build_results_inventory.py` from committed outputs; each row records its source file and commit). The v1 → v2 changes and their effect on each thesis claim are in `outputs_v2/reports/v1_to_v2_change_summary.md`.

## Amendment A7 (2026-09-29): anticipatory burden projection — EXPLORATORY, POST HOC

Status: written before any A7 code was run, but **after** the v2 results (Stages 2–6) and the
2026-09-29 diagnostics were seen (FES forecasts ≈ damped no-change; vulnerability tracks price
growth 9–12 months *earlier*; a month-level national signal cannot raise AUC much). Everything
in A7 is labelled exploratory. It does not replace any pre-specified result. Outputs go to
`outputs_v3/anticipation/`.

**Idea.** Use the forecast to project each household's *next* burden, instead of adding a
national index as a predictor. Spend at t+1 covers the 12 months before the t+1 interview, so it
is partly determined by prices that are already known or priced in wholesale markets at t.

**Information set at a wave-t interview in month T.** Retail CPI to T−2 (publication lag);
NBP gas futures and Brent to T−1; average earnings (AWE, KAB9) to T−3; announced policy only
after its announcement date. The t+1 interview is assumed at T+12, so spend windows are
W_t = [T−12, T−1] and W_{t+1} = [T, T+11].

1. **Price-level forecast (per series, per origin).** Target: log retail CPI level
   (gas D7DU, electricity D7DT, liquid fuels D7DV; solid fuels D7DW is no-change) at horizons
   h = 1…13 from L = T−2. Direct OLS per horizon, expanding window, fitted only on pairs whose
   target month is ≤ L:
   - gas and electricity: Δ_h log P = a_h + b_h·g1 + c_h·g2, with
     g1 = log F(T−1) − mean log F(L−8…L−3) (futures vs wholesale already embedded in retail),
     g2 = mean log F(L−8…L−3) − mean log F(L−20…L−15) − Δ12 log P(L) (wholesale change not yet passed through);
   - liquid fuels: Δ_h log P = a_h + b_h·(log Brent(T−1) − log Brent(L−2));
   - fewer than 24 training pairs → no-change forecast (counted).
   - **Energy Price Guarantee**: from origins after its announcement, the gas and electricity
     forecast is capped at the EPG level relative to CPI in April 2022 (typical-bill
     approximation: £2,500 → gas ×1.36, electricity ×1.17; £3,000 → gas ×1.70, electricity ×1.33)
     for the months and origins in which that EPG level was announced policy.
   - Benchmark: no-change level (P(L) held flat). Evaluation: RMSE of the log window ratio
     log(mean P(W_{t+1}) / mean P(W_t)) over monthly origins, Diebold–Mariano with HAC
     (lag 12); also by horizon and for 2009–2020 vs 2021–2025 origins.
2. **Fuel-mix exposure.** Household spend at t is split into gas, electricity, oil and other
   (dual-fuel combined bills split by the wave's median gas share among separate-bill
   households). Price multiplier M = Σ_f share_f · (mean P_f(W_{t+1}) / mean P_f(W_t)).
3. **Income projection.** Net labour income × (1 + latest known AWE 12-month growth);
   social-benefit + pension income × (1 + uprating), uprating = September CPI before the April
   in W_{t+1} if published by T−1, else latest known CPI 12-month rate (statutory default;
   the 2013–15 1% cap and 2016–19 freeze are not modelled); other income unchanged.
   *Sensitivity S-support:* Energy Bills Support Scheme (−£400 spend, Oct 2022–Mar 2023, known
   from June 2022 origins) and the pensioner cost-of-living payment (+£300 for HRP aged ≥ 66,
   winters 2022/23 and 2023/24, after announcement). Means-tested payments are not modelled
   (receipt not identifiable).
   Projected burden b̂ = spend_t · M / projected annual income (A1 guards: NaN if income
   < £1,200, capped at 1).
4. **Uncertainty.** P_struct = P(b_{t+1} ≥ 10%) = mean_j Φ((log b̂ + e_j − log 0.1)/σ_hh), where e_j
   are training residuals log b_{t+1} − log b̂_oracle (household noise, projection with realised
   prices) and σ_hh = Σ_f share_f·σ_f(T), σ_f(T) = RMS of that series' out-of-sample window-ratio
   errors known by T (≥ 12 errors, else SD of the naive errors).
5. **Evaluation (A6 transitions and split).** Common sample with y, P0, P1 and projection inputs.
   Logistic models with training-only standardisation:
   - P0 (A6 benchmark), P1, P3 = P0 + P1;
   - A_naive: log b̂ (no-change prices, income step) + flag_t;
   - A_fut: log b̂ (futures forecast) + flag_t;
   - A_full: A_fut + P1;
   - A_oracle (realised prices; upper bound, not a valid predictor);
   - P_struct used directly (no fitting).
   Metrics as A6 (AUC, PR-AUC, calibration, top-k, 2,000 PSU bootstraps), on all validation
   transitions and on the **incident** subset (not vulnerable at t). Calibration-in-the-large and
   predicted vs actual prevalence by t+1 interview year. Crisis subset: wave-t interviews
   Jan 2021–Jun 2022.

**Decision rule (fixed now).** Anticipation *adds value* if both hold:
(i) the futures model beats no-change on the window-ratio RMSE for gas and electricity
(DM p < 0.05); and (ii) on the incident validation subset, AUC(A_fut) − AUC(A_naive) > 0 with a
95% PSU-bootstrap CI excluding 0. Secondary: A_fut vs P0, A_full vs P3, calibration by year.

## Amendment A8 (2026-09-29): conditional anticipation tests — EXPLORATORY, POST HOC

Status: written after the A7 results were seen (criterion (i) failed: DM p = 0.16; criterion (ii)
met) and before any A8 code was run. The only A8-relevant quantity inspected beforehand is the
distribution of the ex-ante forecast (78 of 213 origins since 2009 meet the shock definition
below). No outcome was inspected. Outputs go to `outputs_v3/anticipation_a8/`. All A8 tests are
reported, significant or not.

**Shock definition (ex ante).** Origin T is a *shock origin* if the step-1 forecast window ratio
for gas or electricity is at least 10% in absolute value (|log ratio| ≥ 0.10). A transition is
a shock transition if its wave-t interview month is a shock origin. Sensitivity thresholds: 0.05
and 0.20 (reported, not tested).

**Idea 6 — forward-chaining validation.** For every transition k from e→f to n→o, fit on all
earlier transitions (a→b … the one before k) and predict k. Standardisation and resource
composites use that fold's training transitions only. Predictions are stacked across folds.
Model definitions as A7.

**Idea 1 — conditional value of anticipation.** On the stacked forward-chained predictions:
- H8.1: incident transitions (not vulnerable at t) at shock origins: AUC(A_fut) − AUC(A_naive) > 0.
- H8.2a / H8.2b: step-1 window-ratio RMSE, forecast vs no-change, at shock origins only
  (gas / electricity), Diebold–Mariano (HAC lag 12), one-sided (forecast better).
- Reported without a test: the same contrasts at calm origins, and by fold.

**Idea 2 — separate terms.** A_sep_fut: log b_t + log M_fut + log(projected/current income) +
flag_t. A_sep_naive: the same with log M_naive. On the A6 split (train a→b … l→m, validate m→n,
n→o):
- H8.3: incident AUC(A_sep_fut) − AUC(A_sep_naive) > 0.
- Reported: A_sep_fut vs A_fut, and calibration of both.

**Idea 4 — decision metrics.** A6 split, incident validation transitions:
- H8.4: sensitivity in the top 10% of predicted risk (ranked over all validation transitions),
  A_fut − A_naive > 0.
- Reported: A_fut − P0 on the same metric; newly vulnerable households found per 1,000
  households screened; mean absolute error of predicted vs actual prevalence by t+1
  interview year (A6 split and forward-chained).

**Inference.** 2,000 PSU bootstraps for the AUC and sensitivity contrasts (one-sided
p = share of replicates ≤ 0), DM p for H8.2. Holm correction over H8.1, H8.2a, H8.2b, H8.3
and H8.4 at α = 0.05.

**Decision rule.** *Anticipation is useful for vulnerability identification when a price shock is
forecast* if H8.1, H8.2a and H8.2b all survive Holm. H8.3 and H8.4 are supporting evidence, not
part of the headline rule.

## Amendment A9 (2026-09-29): announced price caps — EXPLORATORY, POST HOC

Status: written after the A8 results were seen (H8.2a/b failed Holm: gas p = 0.066,
electricity p = 0.071 one-sided) and before any A9 code was run. Inspected beforehand: only
the Ofgem cap levels themselves (input data), no forecast errors and no outcomes. Outputs go
to `outputs_v3/anticipation_a9/`.

**Data.** Ofgem, *Default tariff cap level* model v1.31 (August 2026), sheet *1b Historical
level tables*: direct-debit ("other payment method") typical-consumption annual level incl.
VAT, single-rate electricity and gas, every cap period (file in `data/raw/anticipation/`).
Cap periods from January 2019 only (earlier columns are indicative, pre-cap).

**Information rule (conservative, no announcement dates needed).** A cap period counts as
known at origin T from T = (first month of the period − 1). Every cap since 2019 was announced
at least about five weeks before it started, so this is never earlier than the real
announcement.

**Energy Price Guarantee.** Effective level = cap × min(1, EPG / published typical dual-fuel
cap) for Oct–Dec 2022 (£3,549), Jan–Mar 2023 (£4,279) and Apr–Jun 2023 (£3,280), with the EPG
level (£2,500 / £3,000) known per the A7 schedule. Months without a known cap keep the A7
ceiling.

**Forecasts (gas, electricity; L = T − 2).**
- *F_cap:* months in a known cap period follow the cap, log P(m) = log P(L) + log(eff(m) / eff(L)).
  Later months follow the A7 futures path from the last known cap month:
  F_cap(m) = F_cap(m*) + F_fut(m) − F_fut(m*). If L < January 2019, F_cap = F_fut.
- *F_capflat:* known cap months as F_cap, then flat. Isolates the value of the futures beyond
  the announcements.
- Benchmark: no-change, as A7.

**Tests.** Holm correction over five tests at α = 0.05:
- H9.1a (gas) / H9.1b (electricity): window-ratio RMSE, F_cap vs no-change, origins
  T ≥ March 2019 (L within the cap regime), Diebold–Mariano HAC lag 12, one-sided.
- H9.2a / H9.2b: F_cap vs F_capflat, same origins and test.
- H9.3: forward-chained (A8 design), incident transitions with T ≥ March 2019:
  AUC(A_cap) − AUC(A_naive) > 0, 2,000 PSU bootstraps. A_cap is A_fut with the F_cap
  window ratios.

**Reported, not tested.** F_cap vs no-change over all 2009+ origins (the A7 criterion (i)
re-run); relative RMSE by horizon; A_cap vs A_fut; A_cap at shock origins; A6-split AUC
and calibration.

**Decision rule.** *Forecasting is useful for household vulnerability identification* if H9.1a,
H9.1b and H9.3 all survive Holm. H9.2 says whether market (futures) information adds anything
beyond the announced caps.
