# v1 → v2 change summary

*Analysis rerun on branch `rerun-v2` against tag `submitted-draft-v1` (the submitted draft).
Plan and every deviation: `analysis_plan_rerun.md`. Every v2 number below is a row of
`outputs_v2/results_inventory.csv` (IDs R####; IDs refer to that file as committed). Prepared
2026-09-26; the analysis is frozen from here.*

## 1. What changed and why

### Data and measurement (errors found in v1)

| Area | v1 | v2 | Why it mattered | Commit |
|---|---|---|---|---|
| Fuel-spend outcome | `fuelduel` = −8 (not asked: household does not have both gas and electricity) read as missing → household dropped; nonresponse on separate amounts set to £0 | Routing-aware spend; nonresponse = missing (complete-case); S1 lower bound (nonresponse as £0) and S2 (+ households not reporting electricity) as sensitivities | v1 dropped 45,656 households whose spend is fully observed under the v2 rule (Stage 1 audit total; the tracked region × wave table sums to 45,634 because cells < 10 are suppressed), almost every off-gas-grid household, incl. 84% of NI oil users (13,463; R0027); understated spend for 13,742 | `1a4b335`, `f2563af` |
| Analytical n | 255,324 (R0015) | 286,902 (R0013) | as above | `f2563af` |
| Disability | `healthlink` = health-record **linkage consent**, wave a only | Equality-Act definition: long-standing illness + any `disdif` difficulty; 97–99.8% coverage per wave | v1 "disability" was a 2009-only consent flag | `96311b6` |
| Self-rated health | interviewer `sf1`, < 11% observed after wave e | self-completion `scsf1`, else `sf1`; 95–100% coverage | v1 self-rated health effectively waves a–e only | `01222a9` |
| Health labels | `health_good` labelled "self-rated health" | `health_good` = no long-standing illness/disability; `sf1_good` = self-rated general health | mislabelled in v1 table/figure | `8750734` |
| Missing codes | −10/−11/−20/−21 kept as values | recoded missing | wave f `ncars`/`carval` = −10 for 2,468 households | `01222a9` |
| Interview timing | `month` = **sample month** (address issue) | actual interview date `intdatey`/`intdatem` | only 24–75% interviewed in sample month; 3.5–10% in another year; affects FES timing and JRF windows | `6c7bb23` |
| Survey weights | unweighted | per-wave household cross-sectional weights, averaged by wave | representativeness | `342f03d` |
| JRF work-status benchmark | 12% / 43% | 15% / 54% (p.77, working-age adults) | v1 values not in the report | `74d19f7` |

### Methods

| Area | v1 | v2 | Why | Commit |
|---|---|---|---|---|
| FES timing | year-Y interview given the forecast for Y+1 made with data to Dec Y (future information) | Dec Y−1 vintage for year-Y interviews; realised stress at month m−1 | no look-ahead | `f89aac7` |
| FES variant | `fes_macro`, chosen by hindsight RMSE over all years | core only; **growth-only 3-term index primary**, 4-term sensitivity | hindsight selection; PI coverage 28–48% shows the uncertainty term is mis-scaled | `f89aac7`, `b95d82f` |
| Forecast tuning | tuned once on 2024, reused for all origins | re-tuned at each origin on data to that origin | look-ahead | `f89aac7` |
| Forecast evaluation | model rankings only | relative RMSE vs naive and seasonal naive, Diebold–Mariano, PI coverage | skill needs a benchmark | `1eaf6e4`, `5d8a7d6` |
| Strain | composite of finnow, finfut, GHQ, bill arrears (bill arrears also a separate predictor) | components entered separately; composites as sensitivities | α = 0.27 (R0393); double counting of arrears | `3ed013c`, `8750734` |
| Driver model | no year effects, unclustered SEs, N = 103,621 | year FE, PSU-clustered SEs (+ two-way, month FE), N = 221,877; NI-oil sequence with AMEs | inference and comparability | `8750734`, `aa64bb6` |
| Resources | second-order SEM (degenerate: loadings ±1, Heywood) | correlated 4-factor CFA → failed pre-registered criteria (CFI 0.78, R0385) → unit-weighted **formative** composites | v1 structure not identified | `0383b95` |
| H1 | interaction sign "flips across refits" | fixed rule, fixed direction (positive = buffering), CI-based buffering bound | v1 sign changed with code revisions, not refits | `9d22783`, `d93a131` |
| JRF comparison | pooled 2009–2024, unweighted, v1 outcome | time-matched to JRF periods (region/ethnicity Apr 2021–Mar 2023; others 2022/23), weighted, PSU-bootstrap CIs | like-for-like comparison | `14088d6`, `c57efcd` |
| Prediction | single model, AUC only | P0 benchmark (current burden), P1, P2 (+FES) on one common sample; training-only standardisation; PR-AUC, calibration, top-k | benchmark and leakage control | `5a503c5`, `44fd61d` |
| Scope | CVAE, fuzzy, SVM, vector-shift map in main text | CVAE/fuzzy/SVM → appendix (v1, not re-estimated); vector-shift map dropped | not identified / not needed for claims | `132ee69` |

### Governance

- Row-level UKHLS files and ENABLE.EU microdata removed from all git history; `.gitignore` guards and a
  pre-commit check block row-level CSVs. GitHub cache purge (orphaned commits incl. `99dfcf2`) to be requested by the author.
- Small-cell suppression (counts 1–9, rates on n < 10) on every tracked table; thesis tables checked for
  complement and window-differencing disclosure (no cell triggered).

## 2. Effect on each thesis claim

Status: **holds** (direction and substance unchanged), **updated** (number changes), **changed**
(substance changes), **not supported**, **dropped**.

| v1 claim (location) | Status | v2 | IDs |
|---|---|---|---|
| 339,201 household-wave observations, 2009–2024 (Abstract, 3.2) | updated | 339,201 rows; interviews 2009–2025 (300 in early 2025); primary analytical n 286,902 | R0013 |
| Trend 11.57% (2009) → 5.09% (2020) → 10.83% (2022) → 10.62% (2023) (Abstract, 4.4) | updated | by wave, weighted: a 12.0% → l 6.5% → n 12.5% → o 12.4%; S1 lower bound 10.8 / 5.6 / 10.2 / 9.9. By interview year the peak is 2023 (14.7%). Say "wave n (fieldwork 2022–24)", not "2022" | R0028, R0050, R0054, R0056, R0086 |
| Financial/psychological strain dominant, OR 6.79 (Abstract, 4.8, H3) | changed | the composite is not a scale (α 0.27). Current financial difficulty is the dominant strain term: OR 1.65 per point [1.61, 1.69], 1.60 per SD; GHQ and financial expectations slightly protective once it is included. Composites: 4.76 (no arrears), 7.52 (v1 definition) | R0294, R0330, R0334 |
| Employment security strongest protective factor, OR 0.16 | holds | 0.17 [0.15, 0.20] | R0307 |
| Workless household OR 1.23 | not supported | 1.05 [0.96, 1.14] net of employment security | R0312 |
| Education protective, OR 0.50 | holds | 0.55 [0.51, 0.59] | R0306 |
| Lone parent 1.22; large family 1.68 | updated | 1.50 [1.38, 1.63]; 1.96 [1.74, 2.22] | R0310, R0311 |
| "Good self-rated health raises odds" | changed | that term is *no long-standing illness* (OR 1.25); self-rated health is null (1.03) | R0298, R0300 |
| FES Delta OR 0.94, p < 0.0001 (Abstract, H2) | updated | growth-only FES Delta OR 0.972 [0.962, 0.981], per SD 0.93; robust to all sensitivities. Correlates 0.16 with v1's signal. Interpretation: vulnerability rises when realised stress exceeds the forecast | R0303 |
| FES magnitude OR 1.07 for next wave (Abstract) | not supported | adding FES to next-wave prediction: no improvement (ΔAUC −0.0006) | R0576 |
| H1 COR moderation inconclusive, sign unstable across refits | changed | **not supported** (primary interaction −0.000034 [−0.000081, 0.000014]); largest buffering compatible with data ≤ 0.018 pp of income per SD Delta (≤ 17% of slope). The v1 "sign flips across refits" came from code changes, not refits | R0404, R0414, R0432 |
| AUC 0.7615 on held-out transitions supports proactive identification (Abstract, H5) | changed | **H5 partially supported.** P1 AUC 0.739 [0.727, 0.751]; the current-burden benchmark P0 is better, 0.780 [0.768, 0.792] (ΔAUC −0.041); FES adds nothing. Post-hoc P0+P1: 0.781, no gain over P0 | R0557, R0563, R0569, R0575, R0592 |
| NI highest prevalence, 15.62% | updated | JRF window (Apr 2021–Mar 2023): 14.4% [12.1, 17.0], rank 1 (CI 1–2), P(rank 1) 0.95; pooled wave-mean 18.0% | R0493 |
| NI lowest on JRF income poverty; heating-oil dependence explains the mismatch; 71.2% use oil | changed (wording) | oil share 72.5% (weighted, JRF window); oil vs non-oil in NI 18.2% vs 8.6%; around two-thirds of NI's excess risk (6.8 → 2.4 pp) accounted for by oil use; ~2.4–2.5 pp remains after oil and rurality. Avoid "explains" | R0497–R0499, R0368–R0372 |
| NI largest reduction over time | holds | −8.9 pp (a–e to k–o) | R0605 |
| Regional agreement with JRF: ρ 0.33 all, 0.73 excluding NI (H4) | not supported | ρ −0.10 all, 0.18 excluding NI (time-matched, weighted, corrected outcome) | R0483, R0484 |
| Tenure agreement ρ 0.80; outright owners more vulnerable than private renters | holds | ρ 0.80; outright owners 13.8% [12.7, 15.0] vs private renters 11.4% (2022/23) | R0489, R0473, R0476 |
| Family type, work status: rank agreement | holds | same direction (lone parent 20.8% vs couple 7.8%; not in work 22.5% vs in work 7.0%; 2022/23) | section 8 |
| Disability: directional agreement (12.86% vs 10.33%) | updated | v1 figure was 2009-only and used a consent item. 2022/23: 14.3% vs 9.5%; same direction as JRF | section 8 |
| Ethnicity divergence (Black Caribbean highest; ρ 0.14) | changed (wording) | **inconclusive**: ρ 0.26, wide CIs (Black Caribbean 17.8% [10.2, 26.6]; Bangladeshi 11.3%, n = 223) | R0487, section 8 |
| Social patterning by region/tenure/family/employment/ethnicity/disability (Table 4-3) | updated | pooled weighted rates in `prevalence_by_group.csv` | section 3 |
| Prepayment meter 24.9% vs 12.9% | holds | 24.4% vs 13.6% (weighted; prepayment flag also fixed for electricity-only households) | R0603, R0604 |
| Equivalisation flips up to 42.5% (5+ households) | updated, relabelled | "sensitivity to equivalising income only": 45.8% for 5+; all flips are into vulnerability | R0602 |
| FES tercile 9.3 / 7.2 / 7.3% | updated | flat: 8.1 / 9.1 / 8.8% (growth-only magnitude) | section 10 |
| LSTM strongest, 50 of 96 cells | updated | core only: LSTM wins 36 of 48 series-origins; no v2 forecast significantly beats naive; v2 beats seasonal naive for carbon only (DM p = 0.002); gas gain concentrated in 2023 | R0135, R0136, R0147, R0148, section 4 |
| FES-selected/weighted correct sign in 2025; FES-macro best rolling RMSE 2.749 | dropped | variants not used in v2 (core only, no hindsight selection) | – |
| COR-SEM loadings; second-order baseline factor; resource maps | changed | CFA fails pre-registered criteria; formative composites (α 0.60 / 0.34 / 0.58 / 0.32, descriptive); continuous regional composite map; hotspot tiers dropped | R0385, R0389–R0392 |
| CVAE alignment (3 of 4 dimensions) | moved | appendix, v1 exploratory, not re-estimated | – |
| Vector-shift map; London largest simulated shift | dropped | – | – |
| Regional financial-strain ORs (Fig 4-36; NI and SE highest) | dropped | not re-estimated (built on the v1 composite) | – |
| Forward risk by month (October highest) and region; 1.1% above 50% risk | dropped | v2 evaluates held-out transitions; no future-wave scoring | – |

## 3. Where things are

- Draft text: `outputs_v2/reports/stage4_h1_draft.md`, `stage5_jrf_ni_draft.md`, `stage6_prediction_draft.md`,
  `appendix_scope_note.md`.
- Thesis tables: `outputs_v2/stage3/thesis_table_*.csv`, `stage5/thesis_T5_*.csv`,
  `stage6/thesis_table_prediction.csv`, `fes_eval/thesis_table_*.csv`.
- Figures: `outputs_v2/descriptives/` (trend), `stage3/figures/`, `stage5/figures/`, `stage6/figures/`,
  `stage7/figures/`.
- Every quotable number: `outputs_v2/results_inventory.csv`.
