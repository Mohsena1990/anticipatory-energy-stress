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
- **H1 decision rule [proposed]:**
  - *Supported* if the interaction is negative with p < 0.05 in the primary model.
  - *Contradicted* if it is positive with p < 0.05.
  - *Not supported* otherwise.
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
