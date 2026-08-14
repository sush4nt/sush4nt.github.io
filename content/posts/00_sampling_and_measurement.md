# Sampling & Audience Measurement: Ground-Up Reference

**Date:** July 19, 2026
**Purpose:** This is new territory, so this document is built ground-up — intuition before formalism, one running example threaded through every section so concepts connect rather than sitting as isolated formulas. Targets the JD's "Sophisticated Sampling," "Audience Measurement," and "Universe Projections" requirements.

**Why this domain is different from your usual ML work:** Your XGBoost/RTB/fraud work asks "given this data, predict Y." This domain asks a prior question: **"is the data itself a trustworthy mirror of the population we care about — and if not, how do we mathematically correct for that before we even start modeling?"** Get the sampling/weighting wrong, and every downstream model — however good — is confidently wrong about the wrong population.

---

## Table of Contents

- [0. Running Example: The Streaming Panel](#0-running-example-the-streaming-panel)
- [1. The Core Problem: Sample vs. Population](#1-the-core-problem-sample-vs-population)
- [2. Simple Random Sampling: The Baseline](#2-simple-random-sampling-the-baseline)
- [3. Stratified Sampling](#3-stratified-sampling)
- [4. PPS (Probability Proportional to Size) Sampling](#4-pps-probability-proportional-to-size-sampling)
- [5. Weighting & Calibration](#5-weighting--calibration)
- [6. Universe Projection](#6-universe-projection)
- [7. Synthetic Population Generation](#7-synthetic-population-generation)
- [8. Reach & Frequency Modeling](#8-reach--frequency-modeling)
- [9. Bias-Variance Trade-offs in High-Variance Datasets](#9-bias-variance-trade-offs-in-high-variance-datasets)
- [10. Interview Narrative: Tying It Together](#10-interview-narrative-tying-it-together)
- [11. Summary Table: Quick Reference](#11-summary-table-quick-reference)

---

## 0. Running Example: The Streaming Panel

Every section below reuses this scenario, with consistent numbers, so you can build one mental model rather than re-deriving intuition each time.

**The setup:** You run audience measurement for a streaming platform. There are **130 million US households** (the "universe" / population you actually care about). You can't track all of them directly, so you recruit a **panel of 10,000 households** who agree to have their viewing tracked. You use this panel to estimate things like "what % of US households watched the season finale last night" and "how many total households saw at least one ad in this campaign."

The entire discipline in this document exists to answer one question honestly: **how do you make inferences about 130 million households from a panel of 10,000, given that the panel was never a perfect random slice of the population to begin with?**

---

## 1. The Core Problem: Sample vs. Population

**Surface:** A sample is only useful to the extent it represents the population you're trying to describe. Two things can break that: (1) **sampling error** — random noise from only seeing a subset, which shrinks as sample size grows, and (2) **sampling bias** — systematic mismatch between who's in your sample and who's in the population, which does *not* shrink no matter how large your sample gets.

**In-Depth:**
- **Sampling error** is the "expected" kind of imperfection — even a perfectly random sample of 10,000 households won't give you the *exact* true population percentage, just an estimate with some margin of error around it. This is quantifiable and shrinks with $\sqrt{n}$.
- **Sampling bias** is the dangerous kind — e.g., if your streaming panel over-recruits tech-savvy, younger households (because panel sign-up happens via an app), then no matter how large you grow the panel, it will systematically over-estimate viewership among young/tech-savvy households and under-estimate everyone else. **More data does not fix bias — it just makes you more confidently wrong.**
- This is the single most important distinction in this entire domain, and it's exactly what stratification, PPS, and weighting/calibration (Sections 3–5) exist to address: they're all techniques to convert a *biased-by-default* real-world sample into one that behaves, for estimation purposes, like it were unbiased.

**Interview-ready one-liner:** "The core discipline here is separating sampling error, which shrinks with more data, from sampling bias, which doesn't — a bigger biased panel just gives you a more confident wrong answer. Most of stratified/PPS sampling and calibration is about designing around or correcting for bias, not reducing error through brute-force scale."

---

## 2. Simple Random Sampling: The Baseline

**Surface:** Every household in the population has an equal, independent chance of being selected. It's the theoretical baseline every other method is compared against — unbiased by construction, but often impractical or inefficient in the real world.

**In-Depth — the math you should have ready:**

For estimating a proportion $p$ (e.g., % of households that watched the finale) from a simple random sample of size $n$:

$$\hat{p} = \frac{\text{households in sample who watched}}{n}$$

$$\text{SE}(\hat{p}) = \sqrt{\frac{p(1-p)}{n}}$$

**Worked example:** Suppose the *true* (unknown) population viewership is 20%. From your 10,000-household panel, 2,050 watched:
$$\hat{p} = 2050/10000 = 0.205$$
$$\text{SE} = \sqrt{\frac{0.205 \times 0.795}{10000}} \approx 0.004 = 0.4\%$$

So your 95% confidence interval is roughly $20.5\% \pm 0.8\%$ — a tight, trustworthy estimate **if and only if** the panel is genuinely a random slice of the 130 million households.

**Why simple random sampling is rarely used as-is in practice:**
1. **It's inefficient for subgroup analysis.** If you also want a reliable estimate for a small subgroup (e.g., households in a specific state, or with a specific demographic), a simple random sample might only capture a handful of them by chance — too few to say anything precise.
2. **True random sampling of a real population is often infeasible.** You can't literally pick 10,000 random US households and force them onto a panel — recruitment is voluntary, which immediately introduces self-selection bias (Section 1's bias problem).

This is the motivating gap that stratified sampling and PPS sampling exist to close.

---

## 3. Stratified Sampling

**Surface:** Instead of sampling from the whole population as one pool, you split the population into non-overlapping subgroups ("strata") — e.g., by region, age bracket, household size — and sample *within* each stratum, usually so each stratum is adequately represented rather than left to random chance.

**In-Depth — why and how:**

**The problem it solves:** In the running example, suppose the West region is 20% of US households, but by pure chance a simple random sample might only capture 15% or 25% West-region households — noisy for that subgroup and for any regional breakdown you need to report. Stratification removes that randomness from the *allocation* step: you deliberately decide how many households to sample from each region, guaranteeing adequate representation.

**Two allocation choices, both worth knowing:**

| Allocation Type | Rule | When to Use |
|---|---|---|
| **Proportional allocation** | Sample size per stratum ∝ stratum's share of the population (e.g., West = 20% of households → 20% of panel from West) | Default choice when you want overall population estimates and don't have a specific reason to over/under-sample a stratum |
| **Optimal (Neyman) allocation** | Sample size per stratum ∝ stratum size **and** stratum's internal variance — allocate more sample to strata that are more heterogeneous/variable | When some strata are much noisier than others internally (e.g., viewership behavior is highly variable in one region, very consistent in another) — you get more precision per sampled unit by over-sampling the noisy stratum |

**Formula — why stratification reduces variance (the intuition, not just the label):**

The variance of a stratified estimator is a *weighted average of within-stratum variances*:
$$\text{Var}(\hat{p}_{strat}) = \sum_h W_h^2 \cdot \frac{p_h(1-p_h)}{n_h}$$
where $W_h$ is stratum $h$'s population weight and $n_h$ its sample size.

Compare this to simple random sampling's variance, which is driven by the *overall* population variance (which includes *between-stratum* variance — differences in viewership rate across regions). **Stratification removes the between-stratum variance component entirely from your error** — you're only left with the (usually smaller) within-stratum variance, averaged. This is the actual mathematical reason stratified sampling is more precise than simple random sampling for the same sample size, whenever strata differ meaningfully from each other.

**Worked example:** Say the true viewership rate is 20% overall, but it's not uniform — West region: 30%, Rest of country: 17%. A simple random sample's variance has to "absorb" that 30% vs 17% spread as noise. A stratified sample that guarantees, say, exactly 2,000 of your 10,000 households are West-region and 8,000 are elsewhere (proportional to the 20%/80% population split) removes the randomness of *how many* West households you happened to get — you only have residual noise from *within* each region, which is smaller.

**Interview-ready one-liner:** "Stratified sampling fixes the allocation problem, not the recruitment-bias problem — it guarantees your sample mirrors known population subgroup proportions, which both reduces variance (by removing between-stratum variance from your error) and guarantees you have enough sample size in every subgroup you need to report on. It requires knowing the true stratum proportions in the population ahead of time, usually from census or other reliable benchmarks."

---

## 4. PPS (Probability Proportional to Size) Sampling

**Surface:** Instead of giving every unit an equal chance of selection (like simple random sampling), you give units a selection probability *proportional to some known size measure* — e.g., a household's likelihood of being sampled is proportional to its household size, or a business's likelihood of being sampled is proportional to its revenue. This is common when the population units vary hugely in "size" and you want your sample's total to be an efficient, low-variance estimator of the population total.

**In-Depth — why this matters and where it's genuinely different from stratification:**

Stratification groups units into buckets and controls *how many* you sample per bucket. PPS instead controls *the probability of selection for each individual unit*, scaled to a size measure, *before* you even define buckets.

**Motivating case:** Suppose instead of household viewership, you're measuring **ad impressions delivered by publisher websites**, and you want to estimate total ad spend flowing through a set of publishers. Publisher size (in terms of traffic) varies enormously — a handful of publishers carry a huge share of total traffic, and a long tail carries very little. If you did simple random sampling of publishers (treating a tiny blog and a massive news site as equally likely to be selected), you'd very likely under-represent the handful of huge publishers who actually account for most of the traffic/spend — and your total-spend estimate would be dominated by noise from which large publishers happened to get randomly included.

**PPS fixes this directly:** give each publisher a selection probability proportional to its known traffic/size. A publisher with 10x the traffic of another is 10x as likely to be sampled. This means large, high-impact units are reliably captured (not left to chance), while the estimator is mathematically corrected (via the Horvitz-Thompson estimator, below) so it remains unbiased despite the unequal selection probabilities.

**The core correction — Horvitz-Thompson estimator:**

Once you sample with unequal probabilities, you can't just average — you have to weight each observation by the *inverse* of its selection probability to keep the estimator unbiased:
$$\hat{T} = \sum_{i \in \text{sample}} \frac{y_i}{\pi_i}$$
where $y_i$ is the value for unit $i$ (e.g., its ad spend) and $\pi_i$ is its probability of selection. Units that were *less* likely to be selected get *more* weight per observed unit when they do show up — this exactly counterbalances the unequal selection so the total estimate stays unbiased.

**Worked example:** A publisher with a 1% selection probability that gets sampled contributes $y_i / 0.01 = 100 \times y_i$ to the total estimate (it's "standing in" for roughly 100 similar unsampled units). A publisher with a 50% selection probability contributes $y_i / 0.5 = 2 \times y_i$. Large, near-certainly-sampled publishers get counted close to their actual value; small, rarely-sampled publishers get scaled up to represent the many similar small publishers not in the sample.

**Interview-ready one-liner:** "PPS sampling is the right tool when population units vary hugely in size and a few large units drive most of the total you're trying to estimate — sampling with probability proportional to a known size measure ensures those high-impact units are reliably captured, and the Horvitz-Thompson correction (weighting by inverse selection probability) keeps the resulting estimator unbiased despite the unequal selection probabilities."

**Stratified vs. PPS — the distinction to have crisp:**

| | Stratified | PPS |
|---|---|---|
| **Controls** | How many units sampled *per subgroup* | The *individual selection probability* per unit |
| **Best for** | Population naturally divides into meaningful subgroups you need represented | A few large units dominate the total you're estimating (skewed size distribution) |
| **Typical domain example** | Households by region/demographic | Publishers by traffic, advertisers by spend, businesses by revenue |
| **Can combine?** | Yes — commonly used together (e.g., stratify by region, then PPS-sample publishers within each region by traffic) | |

---

## 5. Weighting & Calibration

**Surface:** Even a carefully designed panel drifts out of alignment with the population over time (people drop off panels non-randomly, recruitment always has some self-selection). Weighting is the post-hoc correction: you adjust how much each panel household "counts" so that, in aggregate, the weighted panel matches known population benchmarks on key characteristics.

**In-Depth — Post-Stratification:**

The simplest calibration method. You know the *true* population proportions for some characteristic (e.g., age bracket, from census data), and you know your *panel's* proportions for the same characteristic. You compute a weight per group so the panel's weighted proportions match the population's:

$$\text{weight}_h = \frac{\text{true population proportion in group } h}{\text{observed panel proportion in group } h}$$

**Worked example:** Suppose census data says 35-to-54-year-old household heads are 30% of US households, but your streaming panel — because of how it recruits — only has 20% of households in that bracket (they're under-represented, maybe because panel sign-up skews toward tech-comfortable younger and older demographics). The post-stratification weight for that group is:
$$\text{weight} = 30\% / 20\% = 1.5$$
Every household in that bracket in your panel now "counts as" 1.5 households when you compute any weighted estimate — correcting for their under-representation.

**In-Depth — Raking / Iterative Proportional Fitting (IPF):**

Post-stratification works cleanly for **one** characteristic at a time. The real problem: you usually have *multiple* characteristics you need to simultaneously match to population benchmarks (age **and** region **and** household size **and** income bracket), and you typically don't have a full population benchmark for every *combination* of these (the joint distribution) — census data often gives you marginal totals for each characteristic separately, not the full cross-tab.

**Raking (IPF) solves this iteratively:**
1. Adjust weights so the panel matches the population margin for characteristic 1 (e.g., age) — same as post-stratification.
2. Now adjust weights again so the panel matches the population margin for characteristic 2 (e.g., region) — but this adjustment will slightly disturb your age-margin match from step 1.
3. Re-adjust for characteristic 1 again — it's now slightly off after step 2's adjustment.
4. Keep alternating between characteristics, each time nudging weights to match that characteristic's known margin, until the weights converge (stop changing much) and the panel simultaneously matches *all* the marginal population benchmarks reasonably well, even though you never had — or needed — the full joint population distribution.

**Why this is the practically important algorithm to know cold:** almost every real panel-calibration and survey-weighting system (Nielsen-style audience panels included) relies on some form of raking, precisely because you can usually get separate census margins for age, region, income, etc., but essentially never a full joint population cross-tab at the granularity you'd want.

**Interview-ready one-liner:** "Post-stratification corrects a panel's weights against a single known population benchmark. Raking (iterative proportional fitting) extends that to multiple characteristics at once by alternating adjustments across each characteristic's margin until the weights converge — this is the standard approach when you have separate marginal population benchmarks (age, region, income) but not the full joint distribution, which is the normal real-world situation with census-style data."

**A failure mode worth naming proactively:** if weights become extreme (a small handful of panel households end up with very large weights because they're the only representatives of an under-recruited group), your *effective sample size* shrinks — a few high-weight households can dominate the variance of your estimates even though your nominal panel size is large. Production systems typically cap or trim extreme weights and accept a small amount of residual bias in exchange for controlling variance — a direct instance of the bias-variance trade-off covered in Section 9.

---

## 6. Universe Projection

**Surface:** Once your panel is properly weighted/calibrated, "universe projection" is the final step of scaling a sample-level estimate up to an absolute population-level number — e.g., turning "20.5% of the panel watched" into "26.65 million US households watched," using a known total population ("universe") size as the scaling benchmark.

**In-Depth:**

$$\text{Projected Total} = \hat{p} \times \text{Universe Size}$$

**Worked example (continuing from Section 2):** Panel estimate $\hat{p} = 20.5\%$, universe size (total US households, from census/big-data benchmark) = 130,000,000.
$$\text{Projected households that watched} = 0.205 \times 130{,}000{,}000 = 26{,}650{,}000$$

This looks trivially simple — and the multiplication is — but the **entire rest of this document exists to make sure $\hat{p}$ is trustworthy before this step.** This is worth saying explicitly in an interview: universe projection is the *easy* arithmetic at the end of a pipeline whose real difficulty is everything upstream (unbiased/representative sampling design, correct weighting/calibration). A common interview trap is treating "universe projection" as its own hard technical problem — the honest, senior-level answer is that it's simple math gated entirely by the *quality of the calibrated estimate* feeding into it.

**Where "big-data benchmarks" come in (beyond census):** Modern audience measurement increasingly calibrates/validates panel-based estimates against large-scale digital "big data" sources (e.g., set-top-box data, smart TV ACR data, app-level telemetry) that cover far more households than any panel but often lack the rich demographic/behavioral detail panels collect. A common modern pattern: use panel data (rich detail, smaller scale) *calibrated against* big-data sources (larger scale, less detail) to get both richness and scale — sometimes called "hybrid" or "big-data-enhanced panel" methodology. If your interviewer is at a media-measurement company, this hybrid approach is very likely part of their actual methodology and a strong thing to reference.

**Interview-ready one-liner:** "Universe projection itself is simple — you multiply your calibrated sample estimate by the known total population size. The actual technical difficulty lives entirely upstream, in making sure that estimate is unbiased and properly weighted. Increasingly, that upstream calibration also involves reconciling panel-based estimates against larger-scale but less-detailed big-data sources — using the panel's richness and the big-data source's scale together, rather than relying on either alone."

---

## 7. Synthetic Population Generation

**Surface:** Sometimes you don't just want a *weighted estimate* from your panel — you want a full **synthetic dataset** that behaves like the true population at the individual-record level, for use as ML training data. Synthetic population generation creates simulated individual records (synthetic households/people) whose *joint* distribution of characteristics matches known population benchmarks — useful when your actual panel is too small, or too biased, to serve directly as a representative training set for a downstream model.

**In-Depth — how this differs from weighting:**

Weighting (Section 5) keeps your *real* panel records but adjusts how much each counts. Synthetic population generation instead **creates new, simulated records** — this is useful specifically when:
- You need a training set at a scale or granularity your real panel can't provide (e.g., you need household-level synthetic records for every US county, but your panel only has meaningful sample size at the national/regional level).
- You need to protect privacy — synthetic records that match the population's statistical properties without being tied to any single real household.
- Your real panel has structural gaps (near-zero coverage of some subgroup) that weighting alone can't fix — weighting a subgroup with almost no panel representation just inflates a tiny number of records to huge weights (the extreme-weight problem from Section 5); generating synthetic records for that subgroup based on known marginal/joint characteristics can be more stable.

**Common approaches (know these at a conceptual level):**

| Approach | How It Works | Trade-off |
|---|---|---|
| **Iterative Proportional Fitting → microsimulation** | Use raking-style fitting to estimate the joint distribution across characteristics, then sample synthetic individual records from that fitted joint distribution | Conceptually simple, extends naturally from Section 5's raking; can struggle with very high-dimensional joint distributions |
| **Copula-based generation** | Model the marginal distribution of each characteristic separately, then use a copula to stitch them into a realistic joint distribution with appropriate correlations | More flexible for continuous/mixed variable types; more statistically involved to implement correctly |
| **Generative ML approaches (GANs, VAEs, or modern generative models)** | Train a generative model directly on the (small, real) panel data to learn to produce realistic synthetic records, ideally constrained to match known population benchmarks | Can capture complex, non-linear relationships between characteristics better than IPF/copula approaches; needs enough real data to train on and careful validation that synthetic records don't just memorize/leak real panel households |

**Interview-ready one-liner:** "Weighting adjusts real panel records; synthetic population generation creates new simulated records that match known population benchmarks at the joint-distribution level — I'd reach for this when the real panel has structural coverage gaps that weighting alone would just inflate into unstable, high-variance weights, or when I need training data at a granularity the real panel can't support. IPF-based microsimulation is the most direct extension of standard calibration techniques; generative ML approaches can capture more complex relationships but need careful validation against real benchmarks and privacy-leakage checks."

---

## 8. Reach & Frequency Modeling

**Surface:** "Reach" is the number (or %) of *unique* people/households exposed to a campaign at least once. "Frequency" is the average number of times each exposed person/household was exposed. These are the two foundational metrics of media measurement — nearly every audience-measurement conversation eventually returns to reach and frequency, because they answer the two most basic media-planning questions: "how many people did we reach?" and "how many times, on average, did we reach each of them?"

**In-Depth — why this is a genuinely modeling problem, not just a count:**

If you had perfect individual-level exposure logs for every person, reach and frequency would be trivial counting exercises. The actual difficulty: you almost never have that. You have **sample-level exposure data** (from a panel, or from partial digital exposure logs) and need to *model/estimate* reach and frequency for the full population — and this is where sampling/weighting (Sections 2–6) directly feeds into media measurement's core deliverable.

**A classic modeling tool: reach curves and the "frequency distribution" problem.**
- As ad spend/impressions increase, reach grows but with diminishing returns (you increasingly re-reach the same people rather than finding new ones) — this is typically modeled with a saturating curve (e.g., a form resembling $\text{Reach}(n) = R_{max}(1 - e^{-\lambda n})$, though various functional forms exist across the field).
- Getting from "total impressions delivered" to "unique reach" requires assumptions or models about the *frequency distribution* — how impressions are distributed across individuals (are they concentrated on a few heavily-targeted people, or spread evenly?). Common approaches include fitting a known distributional form (e.g., a negative binomial distribution is a classic choice for modeling frequency distributions) to observed sample-level frequency data, then using that fitted distribution to back out an estimated reach for the full population/campaign.

**Worked example (conceptual, tying back to the panel):** Your 10,000-household panel shows a campaign delivered impressions to panel households with an average frequency of 3.2 exposures among those reached. You fit a negative binomial distribution to that panel-level frequency pattern, then use total campaign impressions delivered (a number you know precisely, e.g., from ad-server logs) plus the fitted distribution shape to estimate total *unique* households reached across the full 130 million household universe — this is a direct, practical fusion of (a) precise census-style impression totals and (b) a statistical distribution shape learned from your representative, calibrated panel.

**Panel calibration's role here (tying back to Section 5):** if your panel is poorly calibrated, your *fitted frequency distribution shape* is wrong — e.g., you'd systematically mis-estimate whether exposure is concentrated or spread out — and every downstream reach/frequency estimate for the full campaign inherits that error. This is a good concrete example to have ready if asked "why does calibration matter downstream."

**Interview-ready one-liner:** "Reach and frequency sound like simple counts, but at scale you rarely have full population-level exposure logs — you're modeling the frequency distribution from panel-level data (often with a distribution like negative binomial) and using that shape, combined with precisely known total impressions, to estimate unique reach across the full population. This is exactly where panel calibration quality directly determines the accuracy of the business-facing reach/frequency numbers."

---

## 9. Bias-Variance Trade-offs in High-Variance Datasets

**Surface:** This is the same bias-variance trade-off you already know from ML model fitting — but applied to *sampling and weighting design decisions* rather than model complexity. Every technique in this document (stratification, PPS, weighting, trimming extreme weights) is, underneath, a bias-variance trade-off decision.

**In-Depth — mapping the familiar ML concept onto this domain:**

| ML Concept (familiar) | Sampling/Weighting Analogue |
|---|---|
| Model too simple → high bias, low variance | Unweighted/uncorrected sample from a biased panel → systematically wrong, but "stable" wrong (won't change much across resamples) |
| Model too complex/overfit → low bias, high variance | Aggressive weighting that perfectly matches every population benchmark, including on very sparse subgroups → unbiased in theory, but a few extreme weights make variance explode (Section 5's extreme-weight problem) |
| Regularization trades a bit of bias for a lot of variance reduction | **Weight trimming/capping** — deliberately introduce a small amount of bias (by capping extreme weights below what perfect calibration would require) in exchange for a large reduction in variance; a very standard production practice in survey/panel weighting |
| Cross-validation to pick the right complexity | Comparing calibrated estimates against held-out validation benchmarks (e.g., a separate big-data source) to tune how aggressively to weight/trim |

**Why "high-variance datasets" specifically shows up in this JD's language:** AdTech/media data is naturally high-variance — a small number of "heavy users" or high-spend advertisers can dominate any raw aggregate (this is exactly the PPS motivating scenario from Section 4). Naive equal-probability sampling or unweighted aggregation on this kind of data produces wildly unstable estimates purely from which heavy-hitters happened to be included — this is a variance problem, and PPS/stratification are, at their core, variance-reduction techniques that happen to also need bias-correction machinery (Horvitz-Thompson, calibration weights) to remain valid.

**Interview-ready one-liner:** "This is the same bias-variance trade-off as model fitting, just applied one level upstream — to the sampling and weighting design instead of model complexity. Weight trimming is the clearest example: you deliberately accept a small, controlled amount of bias in exchange for a large reduction in variance from extreme weights, exactly the way regularization trades bias for variance in a model. In high-variance AdTech data specifically — where a small number of heavy users or high-spend advertisers dominate — PPS and stratified designs exist precisely to control that variance without giving up on an unbiased estimator."

---

## 10. Interview Narrative: Tying It Together

**"Walk me through how you'd design a sampling methodology for a new audience panel."**
> I'd start by identifying the population subgroups where I know I need reliable estimates and stratify on those — using known population proportions from census or reliable benchmarks for allocation. If certain units (households, publishers, advertisers) vary hugely in size and disproportionately drive the metric I care about, I'd layer in PPS sampling so those high-impact units are reliably captured, using Horvitz-Thompson weighting to keep the estimator unbiased. Recruitment will still introduce some self-selection bias no matter how careful the design — that's what post-recruitment calibration, typically raking against multiple population margins, is for.

**"How do you handle panels that have drifted out of alignment with the population over time?"**
> Post-stratification or raking against updated population benchmarks — raking specifically when I need to match multiple characteristics simultaneously but only have separate marginal benchmarks, not a full joint population distribution, which is the normal situation with census-style data. I'd also watch for extreme weights as a sign the panel has structural coverage gaps in some subgroup, and consider weight trimming, or in more severe cases synthetic population generation, rather than letting a handful of massively-overweighted panel members dominate variance.

**"How would you project a sample estimate to a total population number?"**
> That final step — multiplying the calibrated estimate by the known universe size — is simple arithmetic. The real work is upstream: making sure the estimate is unbiased through sampling design and calibration. I'd also increasingly look to reconcile panel-based estimates against larger-scale but less detailed big-data sources, since that hybrid approach gives you both the panel's behavioral richness and the big-data source's scale.

**"How is this related to the bias-variance trade-off you'd use in model selection?"**
> It's the identical trade-off, one level upstream. Weight trimming is the cleanest example — you accept a small amount of controlled bias to avoid variance exploding from a few extreme weights, exactly like regularization in a model. In AdTech data specifically, a small number of heavy users or high-spend advertisers can dominate any naive aggregate, which is a variance problem — PPS and stratified sampling are fundamentally variance-control techniques that need bias-correction machinery to stay valid.

---

## 11. Summary Table: Quick Reference

| Concept | Key Insight | Interview Trigger |
|---|---|---|
| **Sampling error vs. bias** | Error shrinks with n; bias doesn't — more data just makes bias more confident | "Why not just use a bigger sample?" |
| **Simple random sampling** | Unbiased baseline, but inefficient for subgroups and rarely achievable in practice (voluntary panels self-select) | "What's the simplest sampling method and its limits?" |
| **Stratified sampling** | Controls allocation across known subgroups; removes between-stratum variance from the error | "How do you ensure subgroup representation?" |
| **PPS sampling** | Selection probability ∝ a size measure; Horvitz-Thompson weighting (inverse of selection prob) keeps it unbiased | "How do you sample when a few units dominate the total?" |
| **Post-stratification** | Single-characteristic weight correction: population proportion ÷ panel proportion | "How do you correct one demographic imbalance?" |
| **Raking (IPF)** | Iteratively adjusts weights across multiple characteristics until convergence; standard when you have marginal but not joint population benchmarks | "How do you calibrate against multiple census margins at once?" |
| **Universe projection** | Calibrated estimate × known population size — simple math gated by upstream estimate quality | "How do you scale a sample stat to a population number?" |
| **Synthetic population generation** | Creates new simulated records matching population joint distribution; used for coverage gaps, privacy, or scale that weighting can't fix | "How do you handle a subgroup with almost no panel coverage?" |
| **Reach & frequency** | Modeling the exposure-frequency distribution (e.g., negative binomial) from panel data, combined with known total impressions, to estimate unique population reach | "How do you estimate campaign reach without full exposure logs?" |
| **Bias-variance in sampling** | Same trade-off as ML model fitting, applied to sampling/weighting design; weight trimming is the clearest example | "How does this relate to bias-variance trade-off in modeling?" |

---

**Next:** This document is intuition/framework-first, appropriately for ground-up new territory. A natural Phase 2 (matching your two-phase prep pattern) would be worked numerical problems — e.g., "given this panel composition and these census margins, compute the raking weights by hand for two iterations" — once this framework feels solid on a first pass.
