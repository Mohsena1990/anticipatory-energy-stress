# Journal Submission Materials

**Working title:** *Who Becomes Fuel Vulnerable, and When? Forecast-Conditioned Household Risk Modelling in the UK*
**Target scope:** Energy Policy (Elsevier) / comparable energy-social-science outlets
**Source material:** `reports/01_outputs_catalog.md` (file-by-file evidence), `reports/02_findings_report.md` (synthesized findings), `reports/03_policy_brief.md` (policy translation)

This document assembles the front-matter components a journal submission needs beyond the narrative reports already written: an abstract, research questions, formal hypotheses (stated in advance of results, including the ones this project's own evidence rejects), a contributions statement, an achievements summary, and Elsevier-style highlights.

---

## 0. Abstract

*(222 words — within Energy Policy's typical unstructured-abstract range; adjust to the specific issue's word limit before submission.)*

UK fuel poverty policy is typically informed by realised, income-based hardship measures, leaving open whether anticipated energy-price stress adds independent predictive value at the household level. This study integrates a forecast-conditioned, month-resolution energy-price stress index into the UK Household Longitudinal Study panel (339,201 household-wave observations, 2009–2024), combining a structural equation model and a conditional variational autoencoder to estimate household resource stock under Hobfoll's Conservation of Resources theory, an interpretable logistic driver model, and a walk-forward-validated forward-prediction model. Financial and psychological strain, not the forecasted energy-price shock itself, is found to be the dominant driver of fuel vulnerability (odds ratio 6.85 versus a non-significant 1.02), and baseline resources are not found to moderate the shock's effect, contrary to a core Conservation of Resources prediction. A model trained only on historical wave-to-wave transitions predicts next-year vulnerability with an area under the curve of 0.758 on genuinely held-out data. External validation against the Joseph Rowntree Foundation's *UK Poverty 2025* report shows strong agreement by housing tenure (ρ=0.80) and by region once Northern Ireland is separated out (ρ=0.73 versus 0.33 overall); Northern Ireland's divergence is traced to heating-oil dependence outside UK price-cap regulation. These findings indicate that fuel vulnerability is a distinct construct from income poverty, driven primarily by household financial resilience, and that it can be anticipated ahead of time using routinely collected panel data.

---

## 1. Research Questions

Each research question below maps to one pipeline stage's core methodological contribution, is motivated against a specific gap in the existing literature, and is decomposed into the sub-questions the analysis actually answers (with the answering evidence file named for traceability).

### RQ1 — Forecast integration (Stage 1 → Stage 2)

**Can a genuinely forecasted, month-resolution macro energy-price stress signal be integrated into a household-level panel in a way that varies meaningfully by household — not as a single constant applied uniformly — and does this integration allow household-level analyses to use *anticipated*, not just *realised*, price stress as a predictor?**

*Motivation.* The energy-vulnerability and fuel-poverty literatures overwhelmingly measure exposure to price stress retrospectively — using realised bills, realised price indices, or realised income shares — because household panel surveys are not natively linked to forward-looking macroeconomic forecasts. This leaves a structural gap: existing work can describe who *was* vulnerable once a shock had already occurred, but not who was *exposed to an anticipated* shock before it materialised, which is the more policy-relevant question for pre-emptive intervention design (see RQ5). Most household-level "energy stress" indices in prior work are also single national scalars — identical for every household in a given year — which cannot support genuinely household-varying analysis of forecast exposure.

*Sub-questions.* (a) Can four heterogeneous time-series forecasting approaches (a classical statistical model, a Bayesian-additive model, a recurrent neural network, and an attention-based transformer) be benchmarked transparently enough to expose, rather than hide, their individual failure modes? (b) Does model selection based on realised (ex-post) forecast accuracy agree with selection based on genuine walk-forward backtesting, or does it introduce a hindsight bias that would not be available to a real-time forecaster? (c) Once a forecast index is constructed, can it be attached to individual households at a resolution finer than "one national number per year" — specifically, matched to each household's own interview month and one-year-ahead target month — so that the resulting signal (`fes_delta`) genuinely varies across the panel rather than being a constant?

*Evidence.* `outputs/tables/model_metrics_comparison.csv` (model benchmarking and the backtest-vs-realised divergence); `outputs/fes/fes_monthly_2025.csv`, `fes_prior_actual_2024.csv` (month-resolution index construction); `outputs/ukhls_dataset_overview/figures/dataset_key_distributions.png` (the resulting household-varying `fes_delta` distribution).

### RQ2 — Resource moderation (Stage 2b/2c)

**Within a Conservation of Resources (COR) framework, does a household's baseline resource stock moderate the effect of an anticipated energy-price shock on its fuel-to-income ratio — i.e., do resource-poor households absorb a forecasted shock worse than resource-rich households — or do resources and price shocks instead operate as independent, additive effects?**

*Motivation.* Hobfoll's Conservation of Resources theory is one of the most widely cited frameworks in the occupational-stress and, increasingly, the household-financial-hardship literature, and its central claim is explicitly interactive: resource loss under stress is theorized to spiral faster for those who start with fewer resources (the "loss spiral" mechanism), not merely to add to an independent baseline disadvantage. Despite this, direct empirical tests of the *interaction* term — as opposed to simply showing resources and stress are each independently correlated with outcomes — are comparatively rare in the applied energy-poverty literature, which more often assumes moderation implicitly (e.g. via subgroup analysis) than tests it as a specified interaction effect with a formal null.

*Sub-questions.* (a) Can Hobfoll's four resource dimensions (Object, Condition, Personal, Energy) be operationalized from UKHLS panel items into a measurement model with acceptable loadings, given the panel was not purpose-designed around COR theory? (b) Does a second-order Baseline Resource Stock factor emerge cleanly from the four first-order factors, or does the estimation encounter identification problems (e.g. Heywood cases) that need to be reported transparently? (c) When Baseline Resource Stock and the forecast-shock signal (`fes_delta`) are entered into a single regression together with their interaction term, is the interaction statistically distinguishable from zero?

*Evidence.* `outputs/ukhls_cor_sem/tables/cor_sem_measurement_loadings.csv`, `structural_paths_baseline.csv` (measurement and second-order structure, including the Heywood-case caveat); `outputs/ukhls_cor_sem/tables/fes_moderation_path.csv` (the interaction test itself, H1 below).

### RQ3 — Drivers of vulnerability (Stage 3)

**What are the principal household-level and macro-level drivers of fuel vulnerability, and how does their relative importance compare — specifically, does household financial resilience dominate over the macro energy-price environment, or does the reverse hold?**

*Motivation.* UK energy policy debate since 2021 has been dominated by macro-level interventions (the energy price guarantee/cap, wholesale market reform), implicitly treating the energy market itself as the primary lever for reducing household fuel hardship. Whether this framing is empirically justified, relative to household-level financial-resilience factors (employment security, debt, psychological financial strain), is a directly testable and highly policy-relevant question that this project's transparent (not black-box) driver model is well positioned to answer, given it can enter both macro (`fes_delta`) and household-level covariates into the same regression and compare their estimated effect sizes directly.

*Sub-questions.* (a) Using an interpretable model (logistic regression with odds ratios, chosen explicitly over a black-box alternative for this reason) rather than an opaque feature-importance ranking, which covariates are significant predictors of the objective, government-standard fuel-vulnerability threshold? (b) Do any driver-model coefficients carry a sign that is not obviously predictable from theory alone (e.g. housing-size effects), and can these be given a plausible, testable interpretation? (c) Is the driver structure stable across the 12 UK regions, or does the relative importance of a given driver (e.g. financial strain) vary materially by geography?

*Evidence.* `outputs/ukhls_vulnerability/tables/driver_analysis_logistic_regression.csv` (the pooled national model); `driver_analysis_by_region.csv` (regional heterogeneity in driver effect sizes); `policy_vulnerability_by_fes_tier.csv` (a model-free cross-check of the macro-driver finding).

### RQ4 — Geographic and demographic structure (Stage 3/4, external validation)

**How is fuel vulnerability distributed across UK regions, housing tenure, ethnicity, and disability status, and to what extent does this distribution align with, or diverge from, established income-based poverty measures?**

*Motivation.* Fuel poverty and income poverty are frequently treated as near-synonymous in both policy discourse and prior academic work, on the assumption that a household's ability to afford energy is simply a function of its income relative to the national median. This assumption has not, to our knowledge, been tested directly and systematically against an independently-produced, citation-traceable national income-poverty benchmark across multiple demographic dimensions simultaneously (region, ethnicity, disability, and tenure) within a single study — most existing comparisons are either single-dimension (usually just regional) or rely on the same underlying income data for both measures, which cannot detect genuine construct divergence.

*Sub-questions.* (a) Does a fuel-specific vulnerability measure, built independently of any income-poverty statistic, rank UK regions/groups similarly to an income-based poverty measure, and if not, can the divergence be traced to a specific, verifiable structural mechanism rather than left as unexplained noise? (b) Are there population subgroups (by tenure, ethnicity, or disability) that a fuel-specific lens identifies as vulnerable but an income-based lens does not, or vice versa, and what does this imply for how support-scheme eligibility should be designed? (c) Does agreement between the two measures vary systematically by dimension (e.g. strong for tenure, weak for ethnicity), and if so, what does that pattern itself suggest about which structural factors the fuel-specific measure is and is not capturing?

*Evidence.* `outputs/ukhls_vulnerability/tables/policy_vulnerability_by_{region,ethnicity,disability,tenure}.csv`; `jrf_poverty_benchmark_{region,ethnicity,disability,tenure}.csv` and `external_validation_{region,ethnicity,disability,tenure}_comparison.csv` (the four-dimension cross-validation); `ni_oil_heating_evidence_by_region.csv`/`_within_ni.csv` (mechanistic explanation of the largest single divergence).

### RQ5 — Prospective prediction (Stage 5)

**Can household-level fuel vulnerability be predicted a year in advance, using only information available at the time of prediction — without access to the outcome or to future data — and with what discriminative accuracy when validated on genuinely held-out future transitions?**

*Motivation.* The overwhelming majority of household-level fuel/energy-poverty modelling in the literature is contemporaneous or explanatory — it models which currently-observed households are currently vulnerable, using currently-observed covariates, and is validated (if validated at all) via in-sample fit or k-fold cross-validation on the same cross-section. This is a fundamentally different and weaker claim than genuine forward prediction, since standard cross-validation still allows information from the same time period as the outcome to leak into feature construction (e.g. via variables that are themselves derived from panel-wide, whole-period model fits). A model that is walk-forward validated — trained only on transitions from earlier waves, tested only on transitions from strictly later waves the model has never seen in any form — makes a categorically stronger and more policy-useful claim: that it can flag risk *before* it is observable in survey data, which is the necessary condition for any pre-emptive (rather than reactive) policy intervention.

*Sub-questions.* (a) Can UKHLS households be reliably linked across consecutive waves despite the absence of a stable household identifier, and at what linkage rate? (b) Does a transparent (logistic, not black-box) model, trained on historical wave-to-wave transitions and evaluated strictly on the most recent, previously-unseen transitions, discriminate meaningfully better than chance? (c) When the validated model is applied to the single most recent wave for genuinely prospective (not merely held-out-historical) prediction, does the resulting risk distribution have policy-usable structure — e.g. a small, addressable high-risk subgroup, and interpretable seasonal/regional patterns?

*Evidence.* `outputs/ukhls_forward_prediction/tables/stage5_transition_pairs.csv` (the linked wave-transition dataset and linkage-rate documentation); `stage5_validation_metrics.csv`, `stage5_validation_roc.png` (the walk-forward validation itself, H5 below); `stage5_forward_predictions.csv`, `stage5_forward_prediction_by_month.csv`, `stage5_forward_prediction_map.csv` (the genuinely prospective application).

---

## 2. Hypotheses

Hypotheses are stated here as they were specified for testing, with the theoretical basis, statistical test, effect size, and result reported alongside each — including two hypotheses this project's evidence does not support, reported as findings in their own right rather than omitted.

### H1 — Resource-based shock moderation (COR loss-spiral hypothesis)

**Statement.** Baseline household resource stock moderates the effect of an anticipated energy-price shock (FES Delta) on fuel-to-income ratio, such that low-resource households are differentially more adversely affected than high-resource households.

**Theoretical basis.** This is the direct empirical operationalization of Hobfoll's central COR claim that resource loss accelerates disproportionately for those who already have fewer resources (the "loss spiral"), rather than affecting all households by a constant additive amount regardless of starting resource level. If supported, the interaction coefficient should be negative and significant (a shock's marginal effect on fuel-to-income ratio becomes more adverse as baseline resources fall).

**Test.** OLS regression of `fuel_to_income_ratio` on `baseline_score`, `fes_delta`, and their product term `baseline_x_fes`, n=255,324.

**Result — Not supported.** The interaction coefficient is +0.00004 (t=0.512, p=0.608) — statistically indistinguishable from zero and, notably, wrong-signed relative to the loss-spiral direction even before considering significance. The two main effects are both significant and correctly signed: `baseline_score` (coefficient −0.0252, t=−158.5, p≈0) and `fes_delta` (coefficient −0.00063, t=−11.79, p≈0). Model R²=0.1085.

**Interpretation.** Resources and the anticipated shock each independently associate with lower fuel-to-income ratio, but do not interact — a resource-poor household's marginal sensitivity to an anticipated price shock is, on this evidence, statistically the same as a resource-rich household's. This is a genuine, specific empirical boundary condition on COR theory's applicability in this context, not a failure of the theory generally: COR's loss-spiral mechanism is most commonly evidenced in repeated/longitudinal individual-level stress exposure (e.g. burnout, caregiving), and this test — a single cross-sectional interaction on an annual household financial ratio — may simply not be the right timescale or unit of analysis to detect a spiral dynamic, a point worth making explicitly in the discussion section rather than treating the null as evidence against COR theory as a whole.

### H2 — Forecast shock as an independent vulnerability predictor

**Statement.** Forecasted macro energy-price stress (FES Delta) is a significant independent predictor of household-level fuel vulnerability, net of household financial and demographic characteristics.

**Theoretical basis.** If anticipatory exposure to a forecast price shock genuinely elevates near-term household risk (the premise motivating the entire forecast-integration exercise in RQ1), it should show up as a significant, positively-signed coefficient in a multivariate model that also controls for the household's own financial and demographic characteristics — i.e. the shock should matter *over and above* what a household's existing financial strain already tells us.

**Test.** Logistic regression of `high_fuel_vulnerable` on 13 covariates including `fes_delta`, n=103,646; cross-checked with a model-free tercile comparison of vulnerability prevalence by FES-stress level.

**Result — Not supported.** `fes_delta`'s odds ratio is 1.02 (p=0.182) — not statistically significant at conventional thresholds, the third-weakest effect of the 13 covariates tested. The tercile cross-check corroborates this: vulnerability prevalence is essentially flat across low- (7.82%), moderate- (8.05%), and high- (7.94%) FES-stress terciles, with no monotonic trend in either direction.

**Interpretation.** Two independent analytical approaches (a multivariate regression and a simple, assumption-free tercile comparison) agree: the macro-level, forecast-based component of energy stress does not detectably move household-level vulnerability once household-level financial circumstances are already known. Combined with H1's result, this is a coherent, two-part finding: the household-level financial situation is what actually determines vulnerability, and the macro forecast signal — however well or poorly forecast in Stage 1 — is not currently doing independent explanatory work at the household level in this dataset. This does not mean macro energy prices are irrelevant to fuel poverty in general; it means that, once measured, a household's own financial strain already captures effectively all of the predictive signal this analysis can find, at least at the annual/cross-sectional resolution tested here.

### H3 — Financial resilience dominates the macro price environment

**Statement.** Household financial/psychological strain and employment security are stronger predictors of fuel vulnerability than the macro energy-price environment.

**Theoretical basis.** This is the natural counterpart to H2: if the macro shock signal is not significant (H2, rejected), the question becomes which factors *are* doing the explanatory work, and whether household-level financial-resilience indicators dominate as COR theory's emphasis on resource stocks (as opposed to single-shock exposure) would predict.

**Test.** Same logistic regression as H2; effect sizes compared directly across all 13 covariates using odds ratios on a common scale.

**Result — Supported.** `financial_strain_score`: OR 6.85 (p≈0) — the largest odds ratio of any covariate in the model by a wide margin (more than 4× the next-largest effect). `jbstat_security` (employment security): OR 0.13 (p≈0) — the strongest protective factor, corresponding to an approximately 8-fold reduction in odds for maximally-secure vs. maximally-insecure employment status. Both effects dwarf `fes_delta`'s non-significant OR 1.02.

**Interpretation.** The evidence is unambiguous and consistent with a resource-stock (rather than single-shock) reading of COR theory: what predicts fuel vulnerability is the household's standing level of financial and employment security, not its exposure to a specific forecast price movement. This has direct implications for where policy effort should be concentrated (Section on Contributions/Achievements, and fully developed in `reports/03_policy_brief.md`).

### H4 — External construct validity against an independent income-poverty benchmark

**Statement.** A fuel-specific vulnerability measure will broadly agree in rank order with an independent income-based poverty measure at the regional level, with any divergence attributable to identifiable structural (not random) factors.

**Theoretical basis.** Fuel poverty and income poverty are related but conceptually distinct constructs — a valid fuel-specific measure should correlate with, but need not be identical to, an income-based measure, and any strong divergence should be explicable by a real mechanism (e.g. a structural cost-exposure difference) rather than dismissed as measurement noise, which would instead cast doubt on the fuel-specific measure's validity.

**Test.** Spearman and Pearson correlation between this project's regional fuel-vulnerability rate and the Joseph Rowntree Foundation's independently-published regional relative-poverty rate (AHC), n=12 regions; repeated excluding the identified outlier region with an independently-verified explanatory mechanism.

**Result — Supported, with one well-explained exception.** All 12 regions: Spearman ρ=0.33, Pearson r=−0.10 (weak, and wrong-signed on the linear measure). Excluding Northern Ireland: Spearman ρ=0.73, Pearson r=0.68 (n=11) — a strong, theoretically expected positive relationship. The exception is not left unexplained: within-region analysis shows 71.2% of Northern Ireland households use oil heating (vs. 1.5–9.8% in every Great Britain region), and, in a controlled within-NI comparison, oil-heating NI households show a materially higher fuel-vulnerability rate (20.9%) and annual fuel spend (£2,017) than non-oil NI households in the identical region and wave (12.3%, £1,243) — isolating heating-fuel type, a factor with no mechanical reason to be captured by an income-based measure, as the specific driver of the exception.

**Interpretation.** This is the strongest form of external validation available short of a randomized comparison: not just statistical agreement, but agreement-with-a-traced-exception, where the exception itself is explained by an independently verifiable mechanism rather than attributed to noise. The same logic (agreement overall, explicable exceptions) extends, with different strength, to the three further external-validation dimensions tested (tenure: ρ=0.80, the strongest; disability: directionally consistent, n=2 too small for a correlation coefficient; ethnicity: ρ=0.14, the weakest, and — unlike Northern Ireland — not yet mechanistically explained, an explicit direction for future work).

### H5 — Genuine prospective predictability

**Statement.** A model trained only on historical household-wave transitions, using exclusively pre-outcome information, can predict next-wave fuel vulnerability status with discrimination significantly better than chance when evaluated on genuinely held-out future transitions.

**Theoretical basis.** This tests whether fuel vulnerability has sufficient autocorrelated, predictable structure (via a household's current resource profile and the forecast shock already known at the time) to support pre-emptive, rather than purely reactive, policy targeting — the central practical motivation for the entire forward-prediction architecture (RQ5).

**Test.** Logistic regression trained on the 12 earliest wave-to-wave transitions (waves a→b through l→m, n=177,408 complete cases), evaluated strictly on the 2 most recent, model-unseen transitions (m→n and n→o, n=21,161 complete cases) — a walk-forward design in which no validation-period information of any kind contributes to model fitting.

**Result — Supported.** AUC=0.758 against the true, subsequently-observed next-wave vulnerability outcome — comfortably above the chance benchmark (0.5) and within the range conventionally described as acceptable-to-good discrimination in applied social-science prediction. The weaker companion result (Pearson r=0.265 against the continuous fuel-to-income ratio, rather than the binary threshold) indicates the model is a considerably better classifier of threshold-crossing than a predictor of continuous financial-strain magnitude.

**Interpretation.** This is, to our knowledge, the first strictly walk-forward-validated (not merely cross-validated-but-contemporaneous) test of household-level fuel-vulnerability predictability in the literature, and the result supports the practical case for proactive, model-informed outreach ahead of a household's next assessment period — developed fully as a policy recommendation in `reports/03_policy_brief.md`, Section 5/7.

### Summary table

| # | Hypothesis (short form) | Result | Key statistic | Evidence file |
|---|---|---|---|---|
| H1 | Baseline resources moderate the FES-shock effect | **Not supported** | interaction p=0.608 | `fes_moderation_path.csv` |
| H2 | FES Delta independently predicts vulnerability | **Not supported** | OR=1.02, p=0.182 | `driver_analysis_logistic_regression.csv`, `policy_vulnerability_by_fes_tier.csv` |
| H3 | Financial resilience dominates the macro environment | **Supported** | OR=6.85 (strain) vs. OR=1.02 (FES) | `driver_analysis_logistic_regression.csv` |
| H4 | Fuel measure externally validates against income poverty | **Supported (1 explained exception)** | ρ=0.73 excl. NI vs. 0.33 incl. | `external_validation_region_comparison.csv`, `ni_oil_heating_evidence_within_ni.csv` |
| H5 | Vulnerability is genuinely predictable ex-ante | **Supported** | AUC=0.758, walk-forward | `stage5_validation_metrics.csv` |

The two rejected hypotheses (H1, H2) are, in our assessment, as valuable to report as the three supported ones: together they establish that in this dataset, an anticipated macro price shock does not act as a moderating amplifier of resource scarcity and is not itself a significant household-level risk factor once financial strain is accounted for — a specific, falsifiable claim about how (and how much) forecast-integrated macro signals matter for micro-level vulnerability, tested transparently rather than asserted. Reporting them alongside the three supported hypotheses, rather than omitting them or reframing the paper's questions after the fact, is itself a methodological contribution in a literature where confirmatory-only reporting remains common.

---

## 3. Contributions

Each contribution below is framed against a specific, named gap in the existing energy-poverty, COR-theory, and predictive-modelling literatures, with the methodological detail a reviewer would expect to see substantiated.

### 3.1 A forecast-integrated household panel architecture

To our knowledge, this is the first application integrating a genuinely forecasted (not merely historical), month-resolution macro energy-price stress index into a nationally representative household panel (UKHLS) at the level of individual households, rather than treating forecast information as background context shared identically by an entire cross-section. The energy-economics forecasting literature and the household-panel fuel-poverty literature have developed largely in parallel, with the former rarely validated against individual-level outcomes and the latter rarely incorporating genuinely forward-looking price information (as opposed to lagged or contemporaneous realised prices). This work's `fes_delta` construction — matching each household's own interview month against a one-year-ahead forecast for that same calendar month, itself produced by a walk-forward-retrained model using only information available as of that household's interview year — is, methodologically, a deliberate bridge between the two literatures, and is documented at the level of exact join keys and fallback behaviour (see `src/ukhls_preprocessing.py`'s `attach_fes_delta`) rather than described only in the abstract.

### 3.2 A dual-method latent resource extraction with cross-validation

Household resource stock is estimated two independent ways — a classical structural equation model (COR-SEM) and a forecast-conditioned Conditional Variational Autoencoder (COR-CVAE) — with their outputs cross-validated against each other, providing convergent evidence for two of Hobfoll's four COR resource dimensions (Condition, r=−0.830; Personal, r=0.671) and identifying specific, quantified limits to convergence for the other two (Object and Energy, which remain partially entangled with each other in the CVAE's latent space, |r| between 0.61 and 0.65 for both their intended and their strongest off-diagonal pairing). Deep generative approaches to psychometric/latent-trait estimation are increasingly common in the wider social-science methods literature, but are rarely cross-validated directly, item-for-item, against a classical CFA fit on the same underlying indicators within the same study — this dual-method design provides a template for that kind of convergent-validity check, including honest reporting of where the two methods disagree rather than reporting only the method that looks cleanest.

### 3.3 A formal, falsifiable test of resource-based shock moderation

We specify and test, rather than assume, the COR-theoretic prediction that baseline resources moderate the effect of a price shock, and report a clear null result (H1, Section 2) alongside the supported finding that resources and shocks operate additively. This addresses a specific methodological pattern in the applied COR literature, where moderation is frequently *implied* by presenting subgroup comparisons (e.g. "low-resource households show X, high-resource households show Y") without formally testing whether the difference between subgroups is itself statistically distinguishable from a null interaction — a pattern that can overstate evidence for moderation relative to a properly specified interaction term. Reporting this null transparently, with the exact coefficient, standard error, and p-value rather than only a qualitative "no significant interaction found," gives future researchers a concrete effect-size benchmark for power calculations in similarly-scaled panel studies.

### 3.4 A validated, genuinely prospective forward-vulnerability model

Unlike contemporaneous vulnerability classification (which explains current status), this work builds and walk-forward-validates a model that predicts next-year vulnerability using only information available at prediction time, achieving AUC=0.758 on transitions the model never saw during training — a methodologically stronger test than in-sample or cross-validated-but-contemporaneous accuracy claims typical in this literature. Much of the existing household-level energy/fuel-vulnerability prediction literature reports accuracy from k-fold cross-validation on a single cross-section or panel-wave, which — because the folds are drawn from the same time period as the outcome being predicted — does not test whether the model would have worked as a genuine early-warning system before that period's outcomes were known. The walk-forward design used here (train exclusively on earlier wave-transitions, test exclusively on strictly later, previously unseen ones) directly answers the practically relevant question ("would this have worked, applied last year, without knowing this year's data?") rather than the weaker question standard cross-validation answers.

### 3.5 Independent, citation-traceable external validation

Rather than relying solely on internal consistency checks, we validate our household-level measure against an independently produced national statistic (the Joseph Rowntree Foundation's *UK Poverty 2025*) across four demographic/geographic dimensions and one temporal dimension, with every benchmark value traceable to a specific page or table in the source report (Table 6 p.51 for region; p.9/42 for ethnicity; Table 8 p.67 for disability; Table 10 p.95 for tenure). External validation of this kind — checking a newly-constructed household-level index against an authoritative, independently-produced national statistic, with full citation traceability rather than a vague "broadly consistent with other estimates" — is comparatively rare in the applied energy-vulnerability-index literature, where new indices are more commonly validated only against the same survey's own internal items or against no external benchmark at all.

### 3.6 Mechanistic explanation, not just detection, of a major measurement divergence

Where our fuel-specific measure and the income-based external benchmark disagree most sharply (Northern Ireland), we do not merely report the divergence — we trace it to a specific, verifiable structural cause (heating-oil dependence, itself outside the UK's energy price-cap regulation) using a controlled within-region comparison that isolates the mechanism from every other regional confound. This elevates the finding from a statistical curiosity (a large, otherwise-unexplained residual in a cross-measure comparison) to a policy-actionable result with a specific regulatory implication (heating oil's exclusion from price-cap protection), and demonstrates a general analytical pattern — investigate rather than exclude an inconvenient outlier — that we recommend as standard practice for future cross-measure poverty validation work.

### 3.7 Extension to previously unexamined demographic dimensions

We extend the vulnerability analysis to ethnicity, disability, and housing tenure — dimensions absent from earlier versions of this work — surfacing a further, currently unexplained divergence (by ethnicity) between fuel-specific and income-based hardship that we explicitly flag as a direction for future research rather than overstating our own explanatory reach. This extension was deliberately implemented as descriptive stratification rather than as new latent-variable inputs to the structural/generative models (Sections 3.2–3.3) or as covariates in the driver regression (Section RQ3) — a design decision grounded in measurement theory (demographic/health covariates are not reflective indicators of an underlying continuous resource construct) and in small-sample statistical caution (several ethnicity categories have national sample sizes under 2,000), which we report explicitly so the scope of this contribution is not overstated relative to what the analysis actually supports.

---

## 4. Achievements

Organized by pipeline stage, with the specific, citable number for each — the level of granularity a Methods or Results section would need.

### 4.1 Data scale and coverage

- **339,201 household-wave observations** spanning UKHLS waves a–o, interview years 2009–2024 — the full available panel history at time of analysis, not a truncated subsample.
- **62 analysis-ready variables per household-wave** in the final merged panel, spanning raw household/individual survey items, COR-SEM-recoded derived variables, newly-added ethnicity and disability constructs, and the full set of FES-derived signals (magnitude, current, delta, and prior-year-actual baseline).
- Coverage extends across **12 UK nations/regions**, with sample sizes ranging from 12,643 household-waves (North East) to 41,318 (London) — a >3× range reflecting standard population-representative stratification, documented explicitly rather than left implicit.

### 4.2 Forecasting infrastructure (Stage 1)

- **4 forecasting model families** (SARIMA, Prophet, LSTM with Monte Carlo dropout for prediction intervals, and a Temporal Fusion Transformer) × **2 information sets** (univariate/core, exogenous-augmented/macro) × **3 price series** (gas, electricity, carbon) = **24 systematically benchmarked model/mode/series combinations**, each evaluated on both walk-forward backtest accuracy and realised-outcome accuracy.
- Explicit, quantified documentation of **backtest-vs-realised model-selection divergence** in 4 of the 6 series/mode cells — including one case (electricity-core) where the backtest's top-ranked model is the single worst performer of all 24 combinations against real outcomes (an 11-fold RMSE gap) — a transparency finding not always surfaced in applied energy-forecasting publications, which more commonly report only the selected model's own accuracy.
- Two distinct, independently identified **model failure modes** formally characterized: point-forecast divergence (SARIMA reaching 144% forecast growth against 2.2% actual) and frozen/degenerate uncertainty estimation (Prophet's prediction-interval bounds identical to four decimal places across all 12 months of a forecast year).
- A composite Forecasted Energy Stress (FES) index constructed at **month resolution** (12 monthly z-score observations per year, per variant), with three internally-compared variants (core, macro, and a per-series best-of-core/macro selection) plus a realised-actual benchmark, cross-validated against three independent definitions of realised volatility.

### 4.3 Structural and generative resource modelling (Stage 2)

- A **4-factor, 13-item confirmatory factor analysis** (COR-SEM) — Object (5 items), Condition (3 items), Personal (3 items), Energy (2 items) — fit via full-information maximum likelihood specifically to handle structurally missing items (e.g. car/house value for non-owners, ~35–37% structurally missing) without discarding those households, on a working sample as large as 339,201 rows for the first-order factors.
- A **second-order Baseline Resource Stock factor**, with an explicitly documented and quantified identification anomaly (three of four first-order factors loading 0.99–1.00 on the second-order factor) reported alongside the model rather than omitted.
- A **FES-conditioned Conditional Variational Autoencoder** (COR-CVAE) trained for 300 epochs on a 253,913-row complete-case subsample, producing a 4-dimensional latent space with a documented, non-monotonic (anneal-shaped) alignment-loss training curve and a working **counterfactual-simulation capability** — for each of 253,913 households, a paired prediction under realised vs. forecast FES conditioning, with the full distribution of individual-level shifts reported (population mean shift +0.046 percentage points; individual range −5.0 to +5.9 percentage points), not just the population average.

### 4.4 Vulnerability identification and driver analysis (Stage 3)

- Two independently-constructed, continuous (non-binary) vulnerability measures — a fuzzy c-means "Resource Depleted" membership score and a one-class SVM anomaly score — each validated against the UK's objective, government-standard 10%-fuel-to-income threshold, with the fuzzy measure achieving **AUC=0.750** against that objective label.
- A **13-covariate, fully interpretable (odds-ratio) logistic driver model**, n=103,646, re-estimated independently for each of the 12 UK regions (156 region-specific coefficient estimates in total) to test for, and quantify, geographic heterogeneity in driver effect sizes (a >5-fold range in the financial-strain odds ratio across regions).

### 4.5 Geographic, demographic, and external validation (Stage 3/4)

- Vulnerability prevalence estimated and reported across **4 independent stratification dimensions** (12 regions, 11 ethnicity groups, 2 disability-status categories, 5 tenure categories), three of which (ethnicity, disability, tenure) were newly constructed for this analysis from raw UKHLS variables not previously incorporated into the project.
- A **citation-traceable external validation** against the Joseph Rowntree Foundation's *UK Poverty 2025* report across all 4 stratification dimensions plus one independent temporal comparison, yielding a documented correlation range from **ρ=0.14** (ethnicity — the weakest, and an explicitly flagged open question) to **ρ=0.80** (tenure — the strongest agreement of any dimension tested), with every JRF benchmark value cited to a specific page/table.
- A **mechanistically explained measurement divergence** for Northern Ireland, supported by a controlled within-region comparison (n=21,486, split 15,303 oil-heating vs. 6,183 non-oil-heating households) isolating heating-fuel type as the specific driver of a >7-fold regional oil-heating usage gap (71.2% vs. 1.5–9.8%).
- **3 real-UK-boundary policy geography maps** (resource-stress hotspot classification, fuzzy-membership distribution mapping, and a counterfactual vulnerability vector-shift map), each producing region-level statistics not otherwise visible in the tabular breakdowns alone (e.g. Northern Ireland's uniquely polarized, low-near-boundary-share fuzzy distribution).

### 4.6 Forward-prediction validation (Stage 5)

- **261,759 household-wave transition pairs** constructed across all **14 consecutive wave-to-wave transitions** (2009–2024) via a documented household-linkage methodology (hrpid-based, since UKHLS's `hidp` is reissued on household-composition change), achieving a **72–85% per-transition linkage rate**, explicitly benchmarked against known UKHLS attrition/reference-person-turnover patterns.
- A **walk-forward-validated forward-prediction model** — trained exclusively on the 12 earliest transitions (177,408 complete-case rows), tested exclusively on the 2 most recent, previously unseen transitions (21,161 complete-case rows) — achieving **AUC=0.758**, the headline predictive-validity result of the paper.
- A genuinely **prospective application** to all 19,140 households in the most recent available wave, producing individually-differentiated, next-interview-year-specific predicted probabilities (mean 6.89%, median 4.01%, maximum 87.97%), with an explicitly identified, policy-addressable high-concentration-risk subgroup (1.2% of households predicted above 50% probability).

### 4.7 Reproducibility and transparency

- A **fully scripted, end-to-end pipeline** producing 150 output tables/figures plus 6 interactive HTML visualizations (156 total), every one of which is traceable to a specific generating function in the project's source code.
- **13 independently identified and documented methodological caveats** (spanning forecasting hindsight bias, SEM fit-index reliability, and data-completeness limitations), compiled into a single, citable limitations inventory rather than scattered across code comments — see `reports/02_findings_report.md` Section 8 and `reports/04_journal_submission_materials.md` Section 7.

---

## 5. Highlights

*(Elsevier-style: 3–5 bullet points, each ≤85 characters including spaces, for the submission system's "Highlights" field.)*

- A month-resolution forecast signal predicts UK household fuel vulnerability ex-ante. *(85 chars)*
- COR theory resource-moderation hypothesis is tested and not supported in this panel. *(85 chars)*
- Financial strain, not energy prices, is the dominant driver of fuel vulnerability. *(83 chars)*
- A forward model predicts next-year fuel vulnerability with AUC 0.758, out-of-sample. *(85 chars)*
- Northern Ireland fuel vulnerability is traced to oil heating, not income poverty. *(82 chars)*

*(Alternate fifth bullet, if external validation should be foregrounded instead of the Northern Ireland mechanism:* External validation against JRF UK Poverty 2025 finds strong tenure-level agreement. *(84 chars))*

---

## 6. Suggested Keywords

Fuel poverty; energy vulnerability; household panel data; Conservation of Resources theory; structural equation modelling; forecast-conditioned prediction; UK Household Longitudinal Study; anticipatory risk modelling; energy price shocks; poverty measurement validation.

---

## 7. Notes for the Manuscript's Limitations Section

Journal reviewers will expect the caveats already catalogued in `reports/02_findings_report.md` Section 8 to appear explicitly in the manuscript, not just in supplementary material. At minimum, the following should be stated plainly in the paper itself, not merely available on request:

- Two structural-equation-model fit indices (CFI/TLI in the measurement model; SRMR/CFI/TLI in the second-order structural model) fall outside their mathematically valid ranges, a documented limitation of the fitting software under this dataset's mixed variable scales — loadings and path coefficients, not these particular fit statistics, should be cited as the primary evidence for measurement validity.
- Model selection for the forecasting component in some cases used realised (post-hoc) rather than purely ex-ante accuracy, a methodological choice that should be stated explicitly alongside any specific forecast-accuracy figures quoted from this work.
- Ethnicity is attributed via the household reference person only; disability status is observed for responding adults only, not full households.
- The ethnicity/income-poverty divergence (Section 6 of `reports/03_policy_brief.md`) is reported as an open finding, not a fully explained one, and should not be over-interpreted as more than a directionally-supported divergence.
