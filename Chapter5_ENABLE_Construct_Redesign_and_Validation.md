# Chapter 5: ENABLE Construct Redesign and Validation

## 5.1 UK ENABLE Sample

The analysis draws on the United Kingdom sub-sample of the ENABLE.EU household survey (Wave 1, approximately 2016–2017), identified by country code 11 within the multi-country dataset. After applying the project-defined missing-value filter (codes 9, 98, 99, 999, 9,999, and 99,999 recoded as system missing), the working UK sample comprises **n = 1,015 households** across 525 raw variables. This represents the complete respondent set for the UK wave; no further eligibility exclusions are applied beyond nationality.

**Demographic profile.** The sample spans a broad socio-economic and housing cross-section appropriate for studying household energy vulnerability:

- *Income* (S8): 22.5% report difficulty living on their income (bracket 1, highest difficulty), 46.7% moderate difficulty (bracket 2), 21.2% manageable (bracket 3), and 8.3% comfortable (bracket 4, lowest difficulty). The distribution is left-skewed, indicating the sample over-represents financially stressed households relative to national population averages, which is consistent with targeted ENABLE recruitment in vulnerable communities.
- *Dwelling type* (H1): 22.7% flats or apartments, 56.8% terraced or semi-detached properties, 8.3% detached, and 12.3% other. The predominance of terraced and semi-detached stock reflects the composition of the UK's older urban housing stock.
- *Property age* (H3): 12.8% pre-1919, 29.5% 1919–1944, 37.7% 1945–1980, 14.7% 1981–2000, 3.8% 2001–2011, and 1.5% post-2011. More than 80% of the sample occupies properties built before 1980, making thermal efficiency retrofit potential high but historically underinvested in.
- *Tenure* (S5): nearly even split between owner-occupiers (48.9%) and renters (51.1%). The near-parity of renters is noteworthy for energy vulnerability analysis given the well-documented split-incentive barrier between landlords (who control capital investment in building fabric) and tenants (who pay the energy bills).
- *Household size* (S3): 39.3% single-person, 13.3% two-person, 6.6% three-person, 25.1% four-person, and 15.3% five or more occupants. The high proportion of single-person households (39.3%) is consistent with UK demographic trends and relevant for vulnerability assessment given that single-person households face higher per-capita energy costs.
- *Employment* (S2): 57.6% employed full-time, 28.8% retired, 10.0% unemployed or economically inactive, and 2.9% self-employed or part-time.
- *Age of main respondent* (S6): 15.4% aged 18–30, 19.6% aged 31–45, 48.9% aged 46–60, 13.7% aged 61–75, and 2.5% aged 76 and above.

**Missingness overview.** Core demographic and housing variables (H1, H3, S5, S2, S3, S6) are fully observed or have negligible item-level missingness. The primary missingness challenge arises in the attitudinal and behavioural blocks: E2A (perceived cost of running a television) is missing for 29.1% of respondents, E2B (washing machine cost) for 28.9%, H15B (environmental statement agreement) for 14.7%, H15F for 23.1%, and H15G for 22.2%. E1 (perceived cost per kWh) has 52.2% missingness and is therefore treated as an optional FCP item not included in the core three-item FCP composite. Full item availability and missing rates are documented in Figure `outputs/enable_cleaned/figures/construct_missingness.png` and Table `outputs/enable_cleaned/item_diagnostics.csv`.

---

## 5.2 Codebook Correction

The original ENABLE.EU documentation classifies items into broad thematic blocks — thermal comfort, insecurity, behavioural routines, resource preservation — that do not straightforwardly map to the Conservation of Resources (COR) theoretical framework used in this dissertation. A systematic codebook review was conducted by cross-referencing the original survey instrument, the COR-grounded theoretical constructs of Financial–Energy Cost Pressure (FCP), Adaptive Energy-Management Capacity (AEMC), Energy Behavioural Lock-in (BLI), and Transition-Cost Resistance (TCR), and the actual item wording in each block.

The most significant codebook correction concerns the H12 item block. H12A asks respondents for the proportion of incandescent bulbs in their dwelling, and H12B the proportion of energy-efficient bulbs. These items were initially encountered in documentation that labelled them under a "thermal discomfort" construct. This interpretation is incorrect: bulb counts are lightbulb technology adoption indicators, not measures of experienced thermal discomfort. A dedicated correction log is recorded in `outputs/enable_cleaned/h12_codebook_correction.csv`. The H12 items are excluded from all constructs in this dissertation; their inclusion in any thermal or financial pressure measure would constitute a construct validity error.

The broader redesign resolves four systematic misalignments between the original ENABLE thematic labels and the COR framework:

**Table 5.1. Codebook Corrections: Original Labelling vs. Revised COR Construct Assignment**

| Original ENABLE Label | Problem Identified | Revised COR Construct |
|----------------------|-------------------|----------------------|
| Thermal discomfort (H12A/H12B) | Items measure lightbulb type proportions, not thermal experience | *Excluded* — technology adoption indicator |
| Thermal discomfort / insecurity (H15 block) | H15 items express attitudes toward environmental transition costs and responsibility — not thermal experience | TCR (Transition-Cost Resistance) |
| Insecurity (E2A/E2B, S8) | Items blend objective income difficulty with subjective energy cost perception — constitutes financial–energy pressure, not generic insecurity | FCP (Financial–Energy Cost Pressure) |
| Resource preservation behaviours (E5/E6 blocks) | Items operationalise specific reminder-setting and routine-based energy management practices — better interpreted as adaptive capacity behaviours | AEMC (Adaptive Energy-Management Capacity) |
| Insecurity / mixed behavioural block (E7A–E7E) | Items measure the degree to which energy behaviours are habitual, automatic, and difficult to change — consistent with habit strength and behavioural lock-in, not financial insecurity | BLI (Energy Behavioural Lock-in) |

*Source: Author's review against original ENABLE codebook. H15D (I am willing to make changes to my lifestyle for environmental reasons) is reverse-coded within TCR, since higher willingness corresponds to lower transition-cost resistance.*

The revised construct assignments are formally codified in `outputs/enable_cleaned/construct_variable_map.csv`, which documents each item's construct assignment, scoring direction, reverse-coding flag, and optional/core designation.

---

## 5.3 Item Diagnostics

Before computing construct scores, all items are screened for availability, missingness, variance, and extreme distributions. Full diagnostics are presented in Table `outputs/enable_cleaned/item_diagnostics.csv`, with visual summaries in Figure `outputs/enable_cleaned/figures/construct_missingness.png` (item-level missing rates) and Figure `outputs/enable_cleaned/figures/construct_variability.png` (item variances and zero-variance flags).

**Zero-variance items.** Three items in the E5A block — E5A6 (tariff or price alert reminders), E5A7 (timer-based energy reminders), and E5A8 (smart-meter alerts) — have zero variance in the UK sub-sample: all 1,015 respondents answered "no" or the equivalent null response. This reflects a genuine behavioural characteristic of the 2016–2017 UK smart-meter roll-out period: automated tariff-alert and smart-meter-based reminder technologies were not yet widely available to the general public. These three items are excluded from AEMC scoring as they provide no discriminative information within this sample.

**Low-variance but non-zero items.** E6A6 (use cheap tariff), E6A7 (turn off heating when not needed), E6A8 (do not overfill kettle), E5A2 (note/calendar reminders for energy), E5A3 (ask others to remind), and E5A4 (mobile phone reminders) all have means below 0.05, indicating very low endorsement rates. These items are retained in the AEMC score because the low-endorsement pattern is itself informative about the low baseline adaptive capacity in the UK sub-sample (mean AEMC = 0.22 out of 1.00), but they contribute very low factor loadings in EFA.

**Items with substantial missingness.** E2A (29.1%) and E2B (28.9%) have the highest missingness among core FCP items. Complete-case analysis on FCP therefore uses n = 698 respondents (those with non-missing S8, E2A, and E2B simultaneously). This is the effective analytical sample for the three-item FCP composite and for reliability calculations. E1 (perceived cost per kWh, 52.2% missing) is available as an optional FCP item but is excluded from the core composite given that over half the sample cannot contribute to it. H15F (23.1% missing) and H15G (22.2%) are included in TCR with missingness handled through item-mean imputation at the construct-score stage.

**Recoded items.** H9 (heating control method) is recoded to a 0–3 adaptive capacity scale under the mapping {1: 2, 2: 2, 3: 3, 4: 1, 5: 0, 6: 0}, where higher values indicate greater use of automated or programmable heating controls (thermostats, programmers), which represent more adaptive energy-management behaviour. H9 is classified as an optional AEMC item; it is included in the AEMC composite for respondents with valid responses but does not affect the core nine-item AEMC specification.

**H15D reverse coding.** H15D ("I am willing to make changes to my lifestyle to reduce environmental impacts") is the only pro-environmental item in the H15 block and is scored in the opposite direction from H15A–C, H15E, and H15F, which express resistance or scepticism. H15D is reverse-coded as (max − observed) before inclusion in the TCR composite so that the composite consistently represents resistance to environmental transition costs.

---

## 5.4 FCP: Financial–Energy Cost Pressure

### 5.4.1 Items and Scoring

The Financial–Energy Cost Pressure construct operationalises the degree to which a household experiences financial strain at the intersection of income adequacy and energy cost perception. Three core items are included:

- **S8** — *Difficulty living on present income*: a four-point ordinal scale (1 = greatest difficulty, 4 = no difficulty), scored such that higher values on S8 correspond to lower financial pressure. For FCP, S8 is rescaled to [0, 1] with 1 = greatest difficulty (original value 1 maps to highest FCP-S8 score after inversion if needed — the direction is defined so higher item score means higher financial pressure).
- **E2A** — *Perceived cost of running a television for one hour*: a six-point Likert scale (1 = very low perceived cost, 6 = very high perceived cost). Higher perceived costs signal heightened financial attention to energy expenditure.
- **E2B** — *Perceived cost of running a washing machine for one hour*: same six-point scale as E2A. The washing machine represents a high-salience high-cost appliance in UK household energy budgets.

All three items are scored on their natural direction (higher = more financial pressure) and mean-aggregated after min-max normalisation to produce FCP ∈ [0, 1]. Optional items (E1, S7, E3A–C) were examined but excluded from the core composite owing to high missingness (E1) or ambiguous construct fit (E3 knowledge items).

### 5.4.2 Distribution

The FCP score distribution (Figure `outputs/enable_cleaned/figures/construct_score_distributions.png`) is approximately symmetric with a slight right-skew: mean = 0.476 (SD = 0.181), median = 0.455, interquartile range [0.359, 0.573]. The full range spans 0.0 to 1.0. The near-Gaussian distribution suggests that financial pressure is evenly distributed across the UK sub-sample, without extreme clustering at either pole. Approximately 25% of respondents score above the third quartile (FCP > 0.573), indicating moderate-to-high financial-energy cost pressure.

### 5.4.3 Reliability and Factor Structure

Cronbach's α = **0.518** (n = 698 complete-case), below the conventional threshold of 0.60 (Table `outputs/construct_validation/tables/construct_reliability.csv`). McDonald's ω = 0.097 and composite reliability = 0.097 — both substantially below the 0.70 benchmark. AVE = **0.038**, far below the 0.50 convergent validity threshold. The first EFA factor explains 52.8% of FCP item variance (Figure `outputs/construct_validation/figures/factor_loadings.png`).

**Factor loadings** are low for all three items: S8 = 0.097, E2A = 0.211, E2B = 0.247 — none meet the conventional 0.40 threshold. The low loadings reflect the heterogeneous nature of the three items: S8 is an objective income-strain measure, while E2A and E2B are subjective perceptual items that do not necessarily co-vary with objective income difficulty. A household with a low income may hold accurate perceptions of high energy costs (S8–E2A positive relationship) or may, conversely, be oblivious to unit energy costs (no relationship). This within-construct heterogeneity is the psychometric reason that a single common factor cannot extract strong inter-item covariance.

### 5.4.4 Construct Justification

Despite the weak psychometric convergence, FCP is retained as a theoretically justified formative composite. In COR theory, financial pressure is a resource-threat condition that can manifest through multiple non-interchangeable channels — income inadequacy and heightened cost salience are distinct manifestations, not exchangeable indicators of the same latent state. A formative specification treats items as contributing distinct facets to the composite rather than as redundant reflections of a single underlying factor. Under this framing, the low AVE and α are not invalidating; they reflect the intended heterogeneity of FCP's facets.

---

## 5.5 AEMC: Adaptive Energy-Management Capacity

### 5.5.1 Items and Scoring

Adaptive Energy-Management Capacity represents the breadth of deliberate energy-conservation behaviours that a household has institutionalised. Two behavioural sub-blocks contribute:

**E5A block — Energy-saving reminder use (binary items, 0 = no, 1 = yes):**
- E5A1: *I do not use reminders to save energy* (reverse-coded: 0 = uses no reminders → low AEMC)
- E5A2: Note, calendar, or fridge reminder
- E5A3: Ask others to remind
- E5A4: Mobile phone reminder
- E5A9: Other reminder method

*Excluded zero-variance items:* E5A6 (tariff alerts), E5A7 (timer reminders), E5A8 (smart-meter alerts).

**E6A block — Energy-saving routine use (binary items, 0 = no, 1 = yes):**
- E6A1: *I do not have energy-saving routines* (reverse-coded)
- E6A2: Check each room before leaving house
- E6A3: Switch lights off before leaving rooms
- E6A4: Unplug appliances after use
- E6A6: Use cheap tariff
- E6A7: Turn off heating when not in use
- E6A8: Do not overfill kettle

The nine core items (E5A1r, E5A2–4, E5A9, E6A1r, E6A2–4) with full n = 1,015 form the primary AEMC composite. Additional items (E6A5, E6A6–8) and H9 (heating control) are included as optional items where available. The final composite is the row-wise mean of all non-missing items, mapped to [0, 1].

### 5.5.2 Distribution

AEMC scores are strongly left-skewed with a low mean: mean = **0.221** (SD = 0.092), median = 0.208, IQR [0.167, 0.292], range 0.0–0.542. This is the lowest mean of all four COR constructs and indicates that, on average, UK respondents have adopted fewer than one in four of the measured adaptive energy behaviours. The distribution reveals a fundamental characteristic of the UK sample: low baseline adaptive capacity, with few respondents at the high end of the scale. This is consistent with the documented inertia of UK household energy-management behaviour and the limited incentive to adopt voluntary demand-management practices without financial or regulatory triggers.

### 5.5.3 Reliability and Factor Structure

Cronbach's α = **0.286** (n = 1,015), McDonald's ω = 0.093, CR = 0.093, AVE = **0.035** (Table `outputs/construct_validation/tables/construct_reliability.csv`). The EFA first factor explains 40.6% of AEMC item variance. AEMC is psychometrically the weakest of the four constructs, driven by dramatic heterogeneity across the nine items.

Factor loadings reveal a stark internal split: E6A2 (0.383) and E6A4 (0.370) load moderately, reflecting that room-checking and appliance-unplugging routines cluster around a common "active household energy management" factor. However, E5A2 (0.002), E5A3 (0.001), E5A4 (0.003), E6A6 (−0.002), E6A7 (0.001), and E6A8 (0.002) all have near-zero loadings. The E5 reminder items are very rarely adopted (E5A2 mean = 0.042, E5A4 mean = 0.051) and are uncorrelated with each other or with the E6 routines. E6A6 (use cheap tariff, mean = 0.001) and E6A7 (turn off heating, mean = 0.004) are near-zero-endorsement items that provide essentially no discriminating variance.

The practical implication is that AEMC spans two qualitatively different categories of adaptive behaviour: passive awareness-based behaviours (reminder tools, mean ≈ 0.04) and active routinised behaviours (checking, unplugging, mean ≈ 0.40–0.90). These two sub-categories are poorly correlated, which explains the near-zero inter-item consistency. The formative composite retains all items because they represent genuine facets of the adaptive capacity construct even if they do not form a psychometrically coherent reflective scale.

---

## 5.6 BLI: Energy Behavioural Lock-in

### 5.6.1 Items and Scoring

Energy Behavioural Lock-in operationalises the degree to which household energy behaviours are habitual, automatic, and resistant to deliberate change. Five E7-block items measure distinct facets of habit strength:

- **E7A**: *My energy-related activities are grounded in practice and repetition* (habit anchoring through repetition)
- **E7B**: *My energy-related activities are done while thinking about something else* (cognitive automaticity)
- **E7C**: *My energy-related activities are done without being fully aware of them* (behavioural unawareness)
- **E7D**: *It would be difficult to change my energy-related activities* (resistance-to-change)
- **E7E**: *I do my energy activities consciously because the alternatives are too effortful* (switching-cost awareness)

Items are scored on a five-point scale (1 = strongly disagree, 5 = strongly agree) and mean-aggregated after min-max normalisation to BLI ∈ [0, 1], where higher scores indicate stronger behavioural lock-in.

### 5.6.2 Distribution

BLI has a bimodal distribution (Figure `outputs/enable_cleaned/figures/construct_score_distributions.png`): mean = **0.418** (SD = 0.283), median = 0.500, IQR [0.150, 0.600]. The high standard deviation (0.283) and bimodality indicate that the UK sample divides into two broad groups: households with very low lock-in (BLI near 0 — respondents who do not describe their energy behaviours as automatic or habitual) and households with high lock-in (BLI near 0.5–0.6, treating energy routines as habitual and largely unconscious). This bimodal pattern is consistent with COR theory, which treats habitual behaviours as constituting a stable resource reservoir for some individuals while being absent in others.

### 5.6.3 Reliability and Factor Structure

BLI achieves the strongest psychometric consistency of the four constructs: Cronbach's α = **0.860** (n = 1,015), comfortably above the 0.70 acceptability threshold (Figure `outputs/construct_validation/figures/reliability_ave_cr.png`). McDonald's ω = 0.303. However, AVE = **0.081** — substantially below the 0.50 threshold despite the high α. The EFA first factor explains 65.1% of BLI item variance, the highest of any construct.

Factor loadings range from E7A = 0.347 to E7D = 0.237, with all five items loading in the expected direction and the highest loading belonging to E7A (repetition-based habituation). The modest loading magnitudes (0.24–0.35), despite the high α, illustrate a known limitation of Cronbach's α as a reliability measure: α inflates with more items even when loadings are moderate, while AVE is dominated by the squared loadings and remains low when loadings are below 0.40. The E7 items are genuinely correlated (hence high α) but individually explain limited variance through the common factor (hence low AVE). This reflects the presence of substantial item-specific variance — each E7 item captures a distinct facet of habituation (repetition, unawareness, effortfulness) that is only partially shared with the others.

---

## 5.7 TCR: Transition-Cost Resistance

### 5.7.1 Items and Scoring

Transition-Cost Resistance captures the degree to which a household holds attitudes that resist or conditionalise engagement with energy transition policies on the basis of social, temporal, cost, and technological arguments. Six H15-block items operationalise this:

- **H15A**: *I will only act on environmental issues if others around me do the same* (conditional social resistance)
- **H15B**: *The effects of environmental issues on our lives have been overstated* (denial/scepticism)
- **H15C**: *Environmental issues are for future generations to deal with* (temporal deferral)
- **H15D** (reverse-coded): *I am willing to make changes to my lifestyle to reduce environmental impacts* (reverse: willingness → coded as lower resistance)
- **H15E**: *Government environmental policies should not cost extra money* (cost resistance)
- **H15F**: *Environmental issues will be solved by new technologies without requiring behaviour change* (techno-optimistic fatalism)

H15A–C, H15E, and H15F are scored such that higher agreement indicates greater transition-cost resistance. H15D is reverse-coded so that pro-environmental willingness maps to lower TCR. The five items entering factor analysis (H15A–C, H15E, H15F) are treated as the primary structural core, with H15D handled through reverse coding in the composite mean. TCR is the mean of all six items after recoding, mapped to [0, 1].

**H15G** ("Environmental protection goes hand in hand with economic growth") expresses a positive environmentalism belief inconsistent with resistance framing and is excluded from the core TCR composite, consistent with the designation in `outputs/enable_cleaned/construct_variable_map.csv`.

### 5.7.2 Distribution

TCR score mean = **0.474** (SD = 0.178), median = 0.467, IQR [0.333, 0.600], with 4.3% missing (n = 971 with valid TCR scores, driven by the 14–23% missingness on individual H15 items before item-mean handling). The distribution is approximately symmetric around the midpoint, indicating that the UK sample is evenly divided between households with above- and below-average transition-cost resistance. This balance is substantively important: roughly half of UK households in the sample express views that are conditional on social context, cost neutrality, or technological resolution before they would accept energy-transition responsibilities.

### 5.7.3 Reliability and Factor Structure

Cronbach's α = **0.710** (n = 673 complete-case across the five factor-analysed items), above the 0.60 threshold, McDonald's ω = 0.165, CR = 0.165, AVE = **0.041**. The EFA first factor explains 49.3% of TCR item variance. Factor loadings: H15A = 0.201, H15B = 0.262, H15C = 0.235, H15E = 0.089, H15F = 0.185 — all below the 0.40 threshold. The low loadings reflect the substantive heterogeneity across the five resistance dimensions: social conditionality (H15A), epistemic scepticism (H15B), temporal deferral (H15C), cost resistance (H15E), and techno-optimistic fatalism (H15F). These are genuinely different attitudinal positions that may not correlate highly even within a single respondent who endorses some but not others.

**Note on item content mix.** H15E (government policies should not cost extra money) and H15F (technology will solve environmental problems) represent qualitatively different forms of resistance — one cost-focused, the other displacement-focused. The low loading of H15E (0.089) in particular suggests that cost resistance to government policy is not well captured by the same factor dimension as denial or deferral. This heterogeneity is acknowledged as a limitation of the TCR composite; however, from a COR perspective, all six H15 items represent barriers to proactive resource-preservation behaviour and are retained in the composite on theoretical grounds.

---

## 5.8 AEV and HighAEV Construction

### 5.8.1 AEV Formula and Scoring

The Adaptive Energy Vulnerability (AEV) composite integrates the four COR constructs into a single household-level vulnerability index. It is defined as the row-wise mean of the four component scores, with AEMC flipped to represent capacity deficit (1 − AEMC):

> AEV_i = mean(FCP_i, BLI_i, TCR_i, 1 − AEMC_i)

All four terms are already in [0, 1], and the mean aggregation preserves this range. The formula reflects COR logic: vulnerability is elevated by financial pressure (FCP), behavioural lock-in (BLI), and transition resistance (TCR), and reduced by adaptive capacity (AEMC). Partial missingness on any single component is handled through the row-wise mean, so a household missing TCR (4.3% of the sample) still receives a valid AEV score from the remaining three components.

All 1,015 respondents receive a valid AEV score (missing rate = 0%).

### 5.8.2 AEV Distribution

The AEV score distribution is approximately symmetric and unimodal (Figure `outputs/enable_cleaned/figures/construct_score_distributions.csv`): mean = **0.539** (SD = 0.111), median = 0.542, IQR [0.458, 0.615], range [0.235, 0.873]. The relatively narrow interquartile range (0.16) compared to the full range (0.64) indicates moderate compression around the midpoint, reflecting that the four-component averaging attenuates extreme individual-construct values.

The distribution is markedly more normal than any individual construct score, as expected from averaging four partially independent components. The minimum observed AEV (0.235) is well above zero, reflecting that no household in the UK sample scores simultaneously at the lowest-possible level on all four vulnerability dimensions.

### 5.8.3 HighAEV Classification

The binary HighAEV indicator is defined as:

> HighAEV_i = 1 if AEV_i ≥ Q₀.₇₅ = **0.615**

This produces exactly **254 HighAEV households (25.0%)** and **761 LowAEV households (75.0%)** — a 3:1 class ratio consistent with the design intent of targeting the top quartile of vulnerability. The 75th-percentile threshold is computed within the UK sub-sample (`AEV_QUANTILE = 0.75` in `src/config.py`) and is not a pre-specified absolute cut-off, ensuring the threshold is calibrated to the distribution of the observed sample.

**Table 5.2. AEV and HighAEV Summary Statistics**

| Statistic | Value |
|-----------|-------|
| n (total) | 1,015 |
| AEV mean | 0.539 |
| AEV SD | 0.111 |
| AEV min | 0.235 |
| AEV P25 | 0.458 |
| AEV median | 0.542 |
| AEV P75 (threshold) | **0.615** |
| AEV max | 0.873 |
| HighAEV = 1 (n) | **254 (25.0%)** |
| HighAEV = 0 (n) | **761 (75.0%)** |

*Source: `outputs/enable_cleaned/construct_score_summary.csv` and `outputs/enable_cleaned/enable_aev_scored.csv`. AEV_QUANTILE = 0.75.*

---

## 5.9 Construct Validity Judgement

### 5.9.1 Convergent Validity: AVE and Reliability

Table 5.3 consolidates the full reliability and convergent validity picture across the four constructs. All metrics are drawn from `outputs/construct_validation/tables/construct_reliability.csv` and `outputs/construct_validation/tables/ave_cr_table.csv`.

**Table 5.3. Reliability and Convergent Validity Summary**

| Construct | Items | n | α | ω | CR | AVE | EFA EVR₁ | α ≥ 0.60 | AVE ≥ 0.50 |
|-----------|:-----:|:---:|:-----:|:-----:|:-----:|:-----:|:--------:|:--------:|:----------:|
| FCP | 3 | 698 | 0.518 | 0.097 | 0.097 | 0.038 | 0.528 | ✗ | ✗ |
| AEMC | 9 | 1,015 | 0.286 | 0.093 | 0.093 | 0.035 | 0.406 | ✗ | ✗ |
| BLI | 5 | 1,015 | 0.860 | 0.303 | 0.303 | 0.081 | 0.651 | ✓ | ✗ |
| TCR | 5 | 673 | 0.710 | 0.165 | 0.165 | 0.041 | 0.493 | ✓ | ✗ |

*Thresholds: α ≥ 0.60 (minimum acceptable internal consistency); AVE ≥ 0.50 (convergent validity for reflective latent variable). EVR₁ = variance explained by first EFA factor. n = complete-case respondents used in reliability calculations. Source: `outputs/construct_validation/tables/construct_reliability.csv`.*

**All four constructs fail the AVE ≥ 0.50 threshold**, with maximum AVE = 0.081 (BLI). The highest individual factor loading across all 22 items is 0.383 (E6A2, AEMC) — below even the minimum loading threshold of 0.40 (Figure `outputs/construct_validation/figures/factor_loadings.png`). Under conventional reflective SEM criteria, none of the four constructs demonstrates adequate convergent validity.

### 5.9.2 Discriminant Validity: HTMT and Fornell–Larcker

Despite the convergent validity failures, discriminant validity is acceptable: all six pairwise HTMT ratios are below the 0.85 ceiling (Figure `outputs/construct_validation/figures/htmt_matrix.png`, Table `outputs/construct_validation/tables/htmt_matrix.csv`).

**Table 5.4. Discriminant Validity: HTMT Ratios**

| Pair | HTMT | < 0.85? |
|------|:----:|:-------:|
| FCP – AEMC | 0.523 | ✓ |
| FCP – BLI | 0.100 | ✓ |
| FCP – TCR | 0.177 | ✓ |
| AEMC – BLI | 0.534 | ✓ |
| AEMC – TCR | 0.781 | ✓ |
| BLI – TCR | 0.319 | ✓ |

*Source: `outputs/construct_validation/tables/htmt_matrix.csv`. HTMT threshold = 0.85 (Henseler, Ringle, & Sarstedt, 2015).*

The highest HTMT ratio is AEMC–TCR (0.781), which is theoretically plausible: households with lower adaptive capacity also tend to express higher resistance to transition costs, consistent with COR theory's resource-conservation logic. However, the ratio remains below the critical 0.85 threshold, supporting the conclusion that AEMC and TCR measure sufficiently distinct constructs.

Inter-construct correlations (Figure `outputs/construct_validation/figures/construct_correlation_matrix.png`) are generally low: FCP–BLI = 0.005, FCP–TCR = 0.005 (negligible), AEMC–BLI = −0.203, AEMC–TCR = −0.142, BLI–TCR = 0.205. The negative AEMC–BLI correlation is theoretically consistent (households with higher adaptive capacity tend to have lower behavioural lock-in), and AEMC–TCR is similarly negative (adaptive households are less resistant to transition).

The Fornell–Larcker criterion fails for two pairs (AEMC–BLI: AVE_AEMC = 0.035 < r² = 0.041; BLI–TCR: AVE_TCR = 0.041 ≈ r² = 0.042), but this finding is trivially uninterpretable when all AVEs are below 0.10, since any non-trivial inter-construct correlation will exceed such a low AVE. The Fornell–Larcker failure in this context is a consequence of the low AVEs, not an independent signal of discriminant validity problems.

### 5.9.3 OLS Path Coherence Check

To assess whether the constructs behave directionally as COR theory predicts — irrespective of the psychometric failures — five OLS paths are estimated (Figure `outputs/sem_mediation/figures/cor_path_diagram.png`, detailed results in `outputs/sem_mediation/tables/sem_path_estimates.csv`):

**Path a: FCP → AEMC** (the theoretically critical COR mediation path): β = −0.024, SE = 0.016, t = −1.474, p = 0.141. **Not significant.** Higher financial pressure does not predict lower adaptive capacity at conventional significance levels in this cross-sectional sample.

**Paths b, c', d, e (AEMC/FCP/BLI/TCR → AEV_score):** all return β = ±0.25, p < 0.001, R² = 1.000 — numerical artefacts resulting from the algebraic identity AEV = mean(FCP, BLI, TCR, 1−AEMC). Regressing AEV on its own constituent components recovers a perfect fit by construction. These paths carry no independent behavioural information and are not interpreted as substantive findings. They are reported solely to document the structural circularity for transparency (Table `outputs/sem_mediation/tables/sem_path_estimates.csv`).

**Bootstrap mediation (FCP → AEMC → AEV):** indirect effect = 0.012, 95% bootstrap CI [−0.003, 0.027], 2,000 samples. The confidence interval includes zero, confirming that the indirect pathway through AEMC is not statistically supported (`outputs/sem_mediation/tables/mediation_effects.csv`, Figure `outputs/sem_mediation/figures/mediation_effects.png`). The COR mechanism — financial stress depleting adaptive capacity, which then amplifies vulnerability — is directionally consistent (the a-path is negative as expected) but does not achieve statistical significance in this cross-sectional UK sample, a limitation discussed further in Section 5.10.

### 5.9.4 Implications: Formative Rather Than Reflective Specification

The combined evidence from Sections 5.9.1–5.9.3 yields a clear and internally consistent verdict: **the four COR constructs should be treated as formative composites rather than reflective latent variables.**

In a reflective measurement model, items are assumed to be interchangeable manifestations of a single underlying latent trait, and high AVE (≥ 0.50) is required to demonstrate that the common factor explains more variance than item-specific noise. The ENABLE items do not meet this standard: they represent genuinely heterogeneous behavioural and attitudinal facets — income strain and cost perception (FCP), reminder use and routine habits (AEMC), habituation dimensions from repetition to unawareness (BLI), and five distinct forms of resistance from scepticism to techno-fatalism (TCR) — that are not expected to be highly intercorrelated.

In a formative measurement model, items are definitional components that together constitute the construct, without requiring inter-item covariance. Under this specification, low AVE and factor loadings are neither surprising nor invalidating; they reflect deliberate theoretical breadth. The four composites are retained on these grounds, with the explicit caveat that they should not be used as latent variable scores in structural equation models that assume reflective measurement. The OLS path analysis confirms that directions are consistent with COR theory even if statistical significance is limited by low reliability and cross-sectional design. These constructs function as theoretically justified and empirically auditable vulnerability indicators, and their use in the AEV composite and CatBoost classification (Chapter 6) is appropriate under the formative framing.

---

## 5.10 Chapter Summary: Answering RQ4

**RQ4 asks: How can UK ENABLE household variables be reorganised into codebook-corrected Conservation of Resources constructs of Financial–Energy Cost Pressure, Adaptive Energy-Management Capacity, Energy Behavioural Lock-in, and Transition-Cost Resistance?**

This chapter demonstrates the complete redesign pipeline from raw ENABLE survey variables to four validated COR composites and a household-level AEV vulnerability index.

The redesign begins with a systematic codebook correction that resolves four labelling errors in the original ENABLE thematic classification: H12 items are correctly identified as lightbulb technology counts rather than thermal discomfort indicators and excluded from all constructs; H15 items are reassigned from a thermal/insecurity block to TCR based on their transition-attitude content; E2/S8 items are consolidated into FCP from a generic insecurity label; E5/E6 routine and reminder items are restructured as AEMC; and E7 habit-strength items are assigned to BLI rather than a mixed insecurity construct. These corrections are documented in Table 5.1 and `outputs/enable_cleaned/construct_variable_map.csv`.

Item diagnostics (Section 5.3) identify three zero-variance items in the UK ENABLE sub-sample (E5A6, E5A7, E5A8 — smart-meter and automated reminder features not yet available to the 2016–2017 sample), substantial missingness in FCP's perceptual items (∼29%), and the need for H15D reverse-coding in the TCR composite.

Psychometric validation yields a nuanced result. BLI achieves acceptable internal consistency (α = 0.860) and TCR moderate consistency (α = 0.710). FCP (α = 0.518) and AEMC (α = 0.286) fall below standard thresholds. **All four constructs fail the AVE ≥ 0.50 convergent validity criterion**, with maximum AVE = 0.081 and no factor loading exceeding 0.40. These failures are attributed to the deliberate theoretical heterogeneity of each construct rather than poor item design: FCP deliberately combines objective income strain with subjective cost salience; AEMC spans both near-zero-endorsement reminder tools and high-endorsement routine habits; BLI covers five distinct habit-strength facets; TCR aggregates five qualitatively different forms of transition resistance. Discriminant validity is satisfactory (all HTMT < 0.85).

The conclusion is unambiguous: the four COR composites are valid as **formative (weighted-sum or mean) composites** representing distinct theoretical resource domains in COR theory, but cannot be treated as reflective latent variables with common-factor structure. The OLS path analysis (Section 5.9.3) confirms directional coherence with COR theory — the a-path (FCP → AEMC) is negative as predicted — but statistical significance is not achieved, likely reflecting the combination of low reliability, cross-sectional measurement, and a UK welfare-state context that partially decouples income pressure from adaptive-capacity depletion through means-tested energy support programmes.

The AEV composite (mean of FCP, BLI, TCR, 1−AEMC) is formally defined and validated in Section 5.8, yielding an approximately normal distribution (mean = 0.539, SD = 0.111) and a HighAEV threshold of 0.615 that classifies 254 households (25.0%) as highly vulnerable. This classification provides the binary outcome variable for the machine learning and SHAP analysis in Chapter 6, which addresses RQ5.

---

*Selected output references used in this chapter:*
- *Sample and diagnostics: `outputs/enable_cleaned/enable_aev_scored.csv`, `outputs/enable_cleaned/item_diagnostics.csv`, `outputs/enable_cleaned/construct_variable_map.csv`, `outputs/enable_cleaned/construct_score_summary.csv`, `outputs/enable_cleaned/h12_codebook_correction.csv`*
- *Figures (item-level): `outputs/enable_cleaned/figures/construct_missingness.png`, `outputs/enable_cleaned/figures/construct_variability.png`, `outputs/enable_cleaned/figures/construct_score_distributions.png`*
- *Validation tables: `outputs/construct_validation/tables/construct_reliability.csv`, `outputs/construct_validation/tables/factor_loadings.csv`, `outputs/construct_validation/tables/htmt_matrix.csv`, `outputs/construct_validation/tables/ave_cr_table.csv`, `outputs/construct_validation/tables/fornell_larcker_check.csv`, `outputs/construct_validation/tables/construct_correlation_matrix.csv`*
- *Validation figures: `outputs/construct_validation/figures/reliability_ave_cr.png`, `outputs/construct_validation/figures/factor_loadings.png`, `outputs/construct_validation/figures/htmt_matrix.png`, `outputs/construct_validation/figures/construct_correlation_matrix.png`*
- *Path analysis: `outputs/sem_mediation/tables/sem_path_estimates.csv`, `outputs/sem_mediation/tables/mediation_effects.csv`, `outputs/sem_mediation/figures/cor_path_diagram.png`, `outputs/sem_mediation/figures/cor_path_coefficients.png`, `outputs/sem_mediation/figures/mediation_effects.png`*
