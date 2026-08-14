---
title: "Core Statistical Concepts for Senior ML Scientists"
date: 2026-07-19T00:00:00+05:30
draft: true
tags: ["statistics", "interview-prep", "ml-fundamentals"]
summary: "A worked-example-driven refresher on core statistics, using a running coin-toss example to build fast, genuine intuition."
---

# Core Statistical Concepts for Senior ML Scientists

**Purpose**: A crisp reference for defending statistical foundations in senior-level interviews. Built via worked examples (coin toss as running thread) rather than abstract tables — the goal is recall speed and genuine intuition, not exhaustive coverage.

---

## Table of Contents
1. Bernoulli vs Binomial — Distinguishing the Two (Coin Toss)
2. Probability Distributions & Their ML Roles
3. Hypothesis Testing & P-Values (Worked: 7/10 vs 70/100)
4. Confidence Intervals
5. Bias-Variance Decomposition
6. Probability vs Likelihood, then Maximum Likelihood Estimation (MLE)
7. Bayesian Inference vs Frequentist Thinking
8. Multiple Hypothesis Correction
9. Power Analysis
10. Law of Large Numbers & Central Limit Theorem
11. Statistical Significance vs Practical Significance

*(Regularization is folded into Section 5 and the Bayesian prior connection in Section 7. Flag if you want it pulled out into its own standalone deep-dive.)*

---

## 1. Bernoulli vs Binomial — Distinguishing the Two

This distinction trips people up constantly in interviews because both live in "coin toss land." Get this crisp once, and it never confuses you again.

### The single toss: Bernoulli

Flip **one** coin. Outcome is 0 or 1.

$$X \sim \text{Bernoulli}(p), \quad P(X=1) = p, \quad P(X=0) = 1-p$$

- **One trial. One random variable. One number as the outcome (0 or 1).**
- Mean = $p$, Variance = $p(1-p)$

**Coin toss framing**: "I flip a coin once. Did it land heads?" That single flip *is* a Bernoulli random variable.

### The repeated toss: Binomial

Flip the coin **n times**, independently, same $p$ each time. Count the total number of heads.

$$X \sim \text{Binomial}(n, p), \quad P(X=k) = \binom{n}{k} p^k (1-p)^{n-k}$$

- **n trials. One random variable = the SUM of n Bernoullis.** The outcome is a count (0 to n), not just 0/1.
- Mean = $np$, Variance = $np(1-p)$

**Coin toss framing**: "I flip a coin 10 times. How many heads did I get?" That count is Binomial(10, 0.5).

### The relationship (this is the part people forget to say out loud)

$$\text{Binomial}(n, p) = \sum_{i=1}^{n} \text{Bernoulli}_i(p)$$

A Binomial random variable is literally **the sum of n i.i.d. Bernoulli random variables.** Bernoulli is the atomic unit; Binomial is what you get when you aggregate it.

| | Bernoulli | Binomial |
|---|---|---|
| Number of trials | 1 | n |
| Outcome | 0 or 1 | count of successes, 0 to n |
| Parameter(s) | $p$ | $n, p$ |
| Relationship | the building block | sum of $n$ Bernoullis |

**Where this matters in ML**: A single prediction ("will this one transaction be fraud?") is Bernoulli. Aggregate model behavior across a batch ("how many of these 1000 transactions did we flag?") is Binomial. When you build a confidence interval on a *conversion rate*, you're implicitly reasoning about a Binomial count divided by n — that's why naive normal approximations break down at low $p$ or small $n$ (the underlying object is discrete and only asymptotically Gaussian).

---

## 2. Probability Distributions & Their ML Roles

### Intuition

A distribution is an assumption about how your data (or your errors) were generated. Every loss function you use is quietly making this assumption for you. Knowing the distribution tells you:
- What loss function is "correct" (matches MLE, see Section 6)
- What confidence intervals should look like
- What happens when the assumption is violated

### Gaussian (Normal)

$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

- Continuous, symmetric, light tails (extreme values are *very* rare)
- **Why ML defaults to it**: squared-error loss silently assumes Gaussian noise. Central Limit Theorem also means sums/averages of almost anything tend toward Gaussian — this is why it shows up everywhere even when the raw data isn't Gaussian.
- **Breaks when**: data has outliers or heavy tails (fraud amounts, latencies). A single huge outlier dominates squared loss. Fix: Huber loss, or model with Student-t / Laplace instead.

### Bernoulli / Binomial

Covered above. **Why ML uses it**: binary classification (fraud / not fraud, click / no click) — binary cross-entropy is the exact negative log-likelihood of a Bernoulli. Aggregate metrics like CTR are Binomial counts.

### Categorical / Multinomial

Generalizes Bernoulli/Binomial to $k > 2$ outcomes.
- Categorical = single draw among $k$ classes (like Bernoulli, but $k$-way)
- Multinomial = $n$ draws, counts per class (like Binomial, but $k$-way)
- **Why ML uses it**: multi-class softmax + cross-entropy is MLE under Categorical.

### Poisson

$$P(X=k) = \frac{\lambda^k e^{-\lambda}}{k!}$$

- Counts of rare events in a fixed window (impressions, arrivals, defects)
- Mean = Variance = $\lambda$ (a distinctive signature — if your count data has variance >> mean, Poisson is the wrong model; that's "overdispersion," and Negative Binomial is the fix)
- **Why ML uses it**: modeling ad impressions, queue arrivals, event counts in a time window.

### Quick decision rule (what to reach for)

- Binary outcome → Bernoulli
- Count of successes out of n trials → Binomial
- One of k categories → Categorical
- Counts of events in time/space → Poisson
- Continuous, symmetric, well-behaved → Gaussian
- Continuous, heavy-tailed / outlier-prone → Student-t / Laplace

---

## 3. Hypothesis Testing & P-Values

### Intuition, built from scratch

You flip a coin **10 times** and get **7 heads**. Is the coin biased, or is this just normal randomness from a fair coin?

The null hypothesis $H_0$: the coin is fair, $p = 0.5$.

The p-value answers exactly one question: **"If the coin really is fair, how likely was I to see a result this extreme (7 or more heads out of 10) just by chance?"**

It is **not** "the probability the coin is fair." That's the single most common misstatement — avoid it out loud in an interview.

### Worked example 1: 7 heads out of 10 tosses

Under $H_0$: $X \sim \text{Binomial}(10, 0.5)$

$$P(X=7) = \binom{10}{7}(0.5)^7(0.5)^3 = 0.1172$$

For a p-value we sum the tail — everything as extreme or more extreme than what we saw (two-sided, so both 7+ heads and the mirror-image 3− heads):

$$P(X \geq 7) = 0.1719 \quad\Rightarrow\quad \text{two-sided p-value} = 2 \times 0.1719 = 0.344$$

**A p-value of 0.34.** Nowhere near the conventional 0.05 threshold. Getting 7/10 heads from a fair coin is unremarkable — it happens about a third of the time. **We do not reject $H_0$.**

### Worked example 2: 70 heads out of 100 tosses

Same proportion (70%). Under $H_0$: $X \sim \text{Binomial}(100, 0.5)$

$$P(X \geq 70) = 0.0000393 \quad\Rightarrow\quad \text{two-sided p-value} = 0.0000785$$

**A p-value of 0.000078.** Overwhelming evidence against $H_0$. **We reject it — this coin is almost certainly biased.**

### Why the same 70% proportion gives wildly different conclusions

This is the entire lesson, stated precisely:

| | n=10, 7 heads | n=100, 70 heads |
|---|---|---|
| Proportion | 70% | 70% |
| Std. error of proportion | $\sqrt{0.5 \times 0.5/10} = 0.158$ | $\sqrt{0.5\times0.5/100} = 0.05$ |
| z-score (how many SEs from 50%) | 1.26 | 4.00 |
| p-value | 0.344 | 0.000078 |

**The standard error shrinks as $n$ grows** ($SE = \sqrt{p(1-p)/n}$, so it scales like $1/\sqrt{n}$). The *same deviation from 50%* becomes many more standard errors away as $n$ increases, because your estimate of the true rate gets more precise with more data. At n=10, a 70% result is barely 1.26 SEs out — totally plausible noise. At n=100, 70% is a full 4 SEs out — essentially impossible under a fair coin.

**The one-line takeaway to say out loud**: *p-values conflate effect size with sample size — the same observed effect becomes "more significant" purely by collecting more data, which is exactly why you always report effect size and confidence interval alongside the p-value, never the p-value alone.*

### Hypothesis testing — the master template

Every hypothesis test follows the same five-step skeleton. Below is the template, then the test statistics slotted in with the **why** for each.

**The five steps (always the same):**
1. State $H_0$ (no effect / no difference) and $H_1$ (the effect you suspect)
2. Choose a test statistic that measures "how far is my data from what $H_0$ predicts"
3. Derive/assume the distribution of that statistic under $H_0$
4. Compute the p-value: how extreme is my observed statistic under that distribution
5. Compare to significance level $\alpha$ (typically 0.05); reject $H_0$ if p < $\alpha$

**Which test statistic, and why:**

| Test | Used when | Test statistic | Why this statistic |
|---|---|---|---|
| One-sample z / binomial test | Comparing a proportion or count to a known value (our coin example) | $z = \frac{\hat{p}-p_0}{\sqrt{p_0(1-p_0)/n}}$ | Measures deviation in standard-error units; exact binomial for small n, normal approx for large n |
| Two-sample t-test | Comparing means of two groups (model A vs model B accuracy) | $t = \frac{\bar{X}_1-\bar{X}_2}{s_p\sqrt{1/n_1+1/n_2}}$ | Standardizes the mean difference by its estimated variability; t-distribution corrects for extra uncertainty from estimating variance with small samples |
| Paired t-test | Same units measured twice (before/after, same test set two models) | $t = \frac{\bar{d}}{s_d/\sqrt{n}}$ on differences $d_i$ | Removes between-subject variance by differencing first — much more powerful than unpaired when pairing is valid |
| Chi-square test | Independence or goodness-of-fit for categorical data (is feature X independent of label Y?) | $\chi^2 = \sum \frac{(O-E)^2}{E}$ | Sums squared, normalized deviations of observed vs expected counts across all categories at once |
| ANOVA / F-test | Comparing means across 3+ groups (multiple model variants) | $F = \frac{\text{between-group variance}}{\text{within-group variance}}$ | If groups truly differ, between-group spread should dwarf within-group noise — F captures that ratio directly |

**ANOVA, expanded**: with $k$ groups and $n$ total observations,

$$F = \frac{MSB}{MSW} = \frac{\sum_j n_j(\bar X_j - \bar X)^2 / (k-1)}{\sum_j\sum_i (X_{ij}-\bar X_j)^2 / (n-k)}$$

$MSB$ (mean square between) measures how spread out the group means are from the grand mean $\bar X$; $MSW$ (mean square within) measures ordinary noise inside each group. A large $F$ means the groups differ more than noise alone would explain. Why not just run pairwise t-tests instead? Because running $\binom{k}{2}$ t-tests inflates the false-positive rate (see Multiple Testing below) — ANOVA tests "any difference among all groups" in one shot, and you only drill into pairwise comparisons (with correction) if the F-test is significant.

**The pattern to internalize**: every test statistic is some version of **(observed − expected) / (uncertainty in that estimate)**. Once you see that shape, you can always reconstruct the right test rather than memorizing formulas.

### Pitfalls (the "why it fails" list, kept short)

- **Multiple comparisons**: test 100 features independently at $\alpha=0.05$, expect ~5 false positives by chance alone. Fix: Bonferroni or Benjamini-Hochberg (FDR).
- **Peeking / early stopping**: checking the p-value repeatedly and stopping when it dips below 0.05 inflates false positive rate massively. Fix: pre-register sample size, or use sequential testing methods designed for peeking.
- **Statistically significant ≠ practically significant**: with n=1,000,000 even a 0.01% lift can hit p<0.05. Always pair with effect size.

---

## 4. Confidence Intervals

### Intuition — the natural companion to a p-value

A point estimate alone ($\hat p = 0.7$) hides how much you should trust it. A confidence interval answers: *"What range of values is consistent with my data, given the noise in my sample?"* It's the same machinery as hypothesis testing, run in reverse — instead of testing one fixed value of $p$, you're asking which values of $p$ *would not* have been rejected by your data.

**The precise (and commonly butchered) definition**: a 95% CI means that if you repeated the sampling process many times and built an interval each time, 95% of those intervals would contain the true parameter. It does **not** mean "95% probability the true value is in this specific interval" — that's a Bayesian credible-interval statement (Section 7), a fine everyday shorthand but technically a different claim.

### The formula (Wald interval)

$$\hat p \;\pm\; z_{\alpha/2}\sqrt{\frac{\hat p(1-\hat p)}{n}}$$

Same shape as every test statistic so far: **estimate ± (critical value × standard error)**. $z_{\alpha/2}=1.96$ for 95% confidence.

### Worked example: continuing the coin, both sample sizes

**7 heads out of 10** ($\hat p = 0.7$):

$$SE = \sqrt{\frac{0.7 \times 0.3}{10}} = 0.145 \quad\Rightarrow\quad 95\%\text{ CI} = 0.7 \pm 1.96(0.145) = (0.42,\ 0.98)$$

A huge interval — it comfortably contains 0.5. This is the *same conclusion as the p-value* (0.34, not significant) arrived at from the other direction: the data is consistent with a wide range of true $p$, including a fair coin.

**70 heads out of 100** ($\hat p = 0.7$):

$$SE = \sqrt{\frac{0.7 \times 0.3}{100}} = 0.046 \quad\Rightarrow\quad 95\%\text{ CI} = 0.7 \pm 1.96(0.046) = (0.61,\ 0.79)$$

Much tighter — 0.5 is nowhere near this interval, agreeing with the earlier p-value of 0.000078. **Same $\hat p$, same 95% confidence level, radically different interval width — purely a function of $n$.** This is the direct visual counterpart to why standard error drives everything in Section 3: more data doesn't just shrink your p-value, it shrinks the range of plausible true values around your estimate.

### Why the Wald interval can mislead — Wilson score interval

The Wald formula above uses $\hat p$ itself to estimate the standard error, which breaks down for small $n$ or $p$ near 0/1 (it can even produce impossible bounds outside $[0,1]$). The **Wilson score interval** corrects for this by inverting the actual hypothesis test rather than plugging in a point estimate:

$$\frac{\hat p + \frac{z^2}{2n} \;\pm\; z\sqrt{\frac{\hat p(1-\hat p)}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}$$

For the n=10 case, Wilson gives a center of 0.644 (pulled toward 0.5, unlike Wald's uncorrected 0.7) with interval (0.40, 0.89) — narrower and better-behaved than Wald's (0.42, 0.98). For n=100 the two methods nearly agree (0.60–0.78 vs 0.61–0.79), since Wald's approximation improves as $n$ grows. **Rule of thumb**: use Wilson (or exact binomial) instead of Wald whenever $n$ is small or the observed rate is near 0 or 1 — exactly the regime of rare-event rates like fraud or CTR.

### Width scales with $1/\sqrt{n}$ — the one number to remember

To halve your CI width, you need **4x** the data, not 2x — interval width shrinks with $\sqrt{n}$, not $n$. This is the direct practical consequence for planning: if a metric's confidence interval is too wide to act on, quadrupling sample size only gets you to half the width, and the same $1/\sqrt{n}$ scaling is exactly what drives the sample-size formula in Power Analysis (Section 9).

---

## 5. Bias-Variance Decomposition

### Intuition — the archery target

Imagine shooting arrows at a bullseye, repeated across many different training sets (many "practice sessions").

- **High bias**: your arrows cluster tightly together, but far from the bullseye. You're consistently wrong in the same way — the model is too simple to capture the true pattern (underfitting).
- **High variance**: your arrows scatter widely around the bullseye, sometimes near it, sometimes far. Small changes in the training set swing your predictions wildly — the model is too flexible, memorizing noise (overfitting).
- **Irreducible error**: even a perfect archer with a perfect bow can't hit dead-center every time — wind, bow imperfections. This is the noise floor in the problem itself; no model can remove it.

### The decomposition

$$\underbrace{\mathbb{E}[(y-\hat f(x))^2]}_{\text{total expected error}} = \underbrace{(\mathbb{E}[\hat f(x)] - f(x))^2}_{\text{Bias}^2} + \underbrace{\mathbb{E}[(\hat f(x) - \mathbb{E}[\hat f(x)])^2]}_{\text{Variance}} + \sigma^2$$

- **Bias**: how far the *average* prediction (averaged over many retrainings on different datasets) is from the truth
- **Variance**: how much predictions *swing* across those different retrainings
- **$\sigma^2$**: noise you can never remove, no matter the model

### Why more complexity trades one for the other

A deeper decision tree fits the training set almost perfectly → low bias. But retrain it on a slightly different sample of the same population and you get a very different tree → high variance. A linear model barely changes across resamples (low variance) but may never capture a nonlinear pattern (high bias).

**This is why regularization exists**: L2/L1 penalties deliberately accept a bit more bias (shrinking coefficients away from their unconstrained best fit) in exchange for a large reduction in variance — usually a good trade when variance was the dominant error source. Ensembling (Random Forest) attacks the opposite direction: it averages many high-variance, low-bias trees, cancelling out variance while barely touching bias.

**One-line interview answer**: *"Total error is bias² + variance + irreducible noise. Simple models underfit (bias-dominated), complex models overfit (variance-dominated), and regularization or ensembling is just a deliberate move along that tradeoff — you're not eliminating error, you're reshaping which kind you're willing to tolerate."*

---

## 6. Probability vs Likelihood, then MLE

### The distinction people gloss over

**Probability**: parameters are fixed and known; we ask about the data.
*"Given a fair coin ($p=0.5$), what's the probability of 7 heads in 10 tosses?"* — a question about data, given a fixed model.

**Likelihood**: data is fixed (it already happened); we ask about the parameters.
*"Given that I observed 7 heads in 10 tosses, how plausible is $p=0.5$? How plausible is $p=0.7$? Which value of $p$ makes my observed data most probable?"* — a question about the parameter, given fixed, already-observed data.

**Same formula, different variable held fixed:**

$$P(X=k \mid p) \quad \text{vs.} \quad L(p \mid X=k)$$

Numerically identical expression, $\binom{n}{k}p^k(1-p)^{n-k}$ — but in probability you plug in a known $p$ and vary $k$ (what data might occur); in likelihood you plug in the observed $k$ and vary $p$ (which parameter best explains it). This is the entire idea. Likelihood is a function *of the parameter*, not a probability distribution over parameters (it doesn't need to integrate to 1).

### Worked example: MLE for a coin

You toss a coin 10 times and observe 7 heads. What value of $p$ maximizes the likelihood of seeing exactly this data?

$$L(p) = \binom{10}{7} p^7 (1-p)^3$$

Take the log (turns the product into a sum, easier to differentiate, and the maximizing $p$ is unchanged since log is monotonic):

$$\ell(p) = \log\binom{10}{7} + 7\log p + 3\log(1-p)$$

Differentiate and set to zero:

$$\frac{d\ell}{dp} = \frac{7}{p} - \frac{3}{1-p} = 0 \;\Rightarrow\; 7(1-p) = 3p \;\Rightarrow\; p = \frac{7}{10} = 0.7$$

**The MLE is exactly the observed proportion, $\hat{p} = k/n$.** This isn't a coincidence — it's what "maximum likelihood" means in the simplest possible case: the parameter value under which your actual data was the single most probable outcome is just the empirical frequency.

Notice the tie back to Section 3: this $\hat p = 0.7$ is the same value we hypothesis-tested against $H_0: p=0.5$. MLE gives you the *best point estimate*; hypothesis testing asks whether that estimate is *significantly different* from some baseline. Two different questions, same coin, same data.

### Generalizing: why "maximize log-likelihood" is everywhere in ML

$$\hat\theta_{\text{MLE}} = \arg\max_\theta \sum_{i=1}^n \log p(x_i \mid \theta)$$

Every standard ML loss is this, in disguise:

- **Squared loss** ⟺ MLE assuming Gaussian noise around predictions
- **Binary cross-entropy** ⟺ MLE assuming Bernoulli-distributed labels (exactly our coin, but now $p$ is a function of input features via logistic regression)
- **Categorical cross-entropy** ⟺ MLE assuming Categorical-distributed labels

So "minimize cross-entropy" and "find the coin-bias-like parameter that makes the observed labels most probable" are the same operation — cross-entropy loss is just $-\log L(\theta)$ for a Bernoulli/Categorical likelihood, summed over your dataset instead of over 10 coin flips.

### Key properties (kept crisp, no padding)

- **Consistent**: as $n \to \infty$, $\hat\theta_{\text{MLE}} \to \theta^*$ (true value)
- **Asymptotically efficient**: achieves the lowest possible variance (Cramér-Rao bound) among unbiased estimators, for large $n$
- **Invariant to reparameterization**: MLE of $\sigma^2$ is just (MLE of $\sigma$)² — you don't need to redo the optimization in a transformed parameterization

---

## 7. Bayesian Inference vs Frequentist Thinking

### The core philosophical split

**Frequentist**: the true parameter $p$ is a fixed, unknown constant. Probability describes long-run frequency of outcomes across repeated experiments. There is no "probability that $p=0.6$" — $p$ either is 0.6 or it isn't. You estimate it and build confidence intervals describing your *procedure's* reliability, not $p$ itself.

**Bayesian**: the parameter $p$ is treated as a random variable with its own distribution, reflecting your belief/uncertainty about it. You start with a **prior** belief, observe data, and update to a **posterior** belief via Bayes' theorem. It's meaningful to say "there's a 90% probability $p$ is between 0.6 and 0.7" — that statement is about your belief, licensed by treating $p$ as random.

### Bayes' theorem

$$P(\theta \mid D) = \frac{P(D\mid\theta)\,P(\theta)}{P(D)} \;\propto\; \underbrace{P(D\mid\theta)}_{\text{likelihood}} \times \underbrace{P(\theta)}_{\text{prior}}$$

Posterior ∝ Likelihood × Prior. The likelihood is exactly the same object from Section 6 (MLE); Bayesian inference doesn't replace likelihood, it multiplies it by a prior and renormalizes.

### Worked example: continuing the coin

You observed 7 heads, 3 tails (n=10). MLE said $\hat p = 0.7$ — but that ignored any prior belief about the coin.

**Weak/uninformative prior**: Beta(1,1) — uniform, "I have no idea if it's fair." Using the Beta-Binomial conjugate relationship:

$$\text{Posterior} = \text{Beta}(1+7,\ 1+3) = \text{Beta}(8,4), \quad \text{posterior mean} = \frac{8}{12} = 0.667$$

Barely moved from the MLE (0.7) — a weak prior gets swamped by data almost immediately.

**Strong prior**: Beta(20,20) — "I've handled thousands of coins, most are close to fair, I'd need strong evidence to think otherwise."

$$\text{Posterior} = \text{Beta}(20+7,\ 20+3) = \text{Beta}(27,23), \quad \text{posterior mean} = \frac{27}{50} = 0.54$$

The strong prior pulls the estimate much closer to 0.5, barely nudged by 10 flips. **This is the whole idea in one worked number**: a prior acts like "pretend data" you already believed in (here, equivalent to having already seen 20 heads/20 tails), and real data has to out-vote it to shift your belief.

### The regularization connection (ties back to Section 5)

This is the single highest-leverage fact to say out loud in an interview: **L2 regularization is exactly MAP estimation (maximum a posteriori) under a Gaussian prior on the weights, and L1 is MAP under a Laplace prior.**

$$\hat\theta_{\text{MAP}} = \arg\max_\theta \; \log P(D\mid\theta) + \log P(\theta)$$

- Gaussian prior on $\theta$ → $\log P(\theta) \propto -\lambda\|\theta\|_2^2$ → this is exactly the L2 penalty term
- Laplace prior on $\theta$ → $\log P(\theta) \propto -\lambda\|\theta\|_1$ → this is exactly the L1 penalty term

So "add L2 regularization" and "assume weights are a priori centered near zero, Gaussian-distributed" are the same statement. Ridge/Lasso aren't ad hoc engineering tricks — they're Bayesian priors in frequentist clothing. This is also why L1's prior (Laplace, sharply peaked at 0) induces sparsity while L2's prior (Gaussian, smoothly peaked) shrinks but rarely zeroes out — the shape of the prior directly explains the shape of the regularization effect.

### Confidence Interval vs Credible Interval (the distinction people blur)

Section 4 built the frequentist 95% CI: repeat the sampling many times, and 95% of the resulting intervals contain the true fixed $p$ — a statement about the *procedure*, not about this one interval. The **Bayesian credible interval** looks similar numerically but means something different: given the data you actually observed, there's a 95% probability $\theta$ itself lies in this interval — a direct probability statement about the parameter, licensed because Bayesian thinking treats $\theta$ as random rather than fixed.

Concretely, our Beta(8,4) posterior from earlier directly gives a 95% credible interval by taking its 2.5th and 97.5th percentiles — no repeated-sampling story required, just "read the probability off the posterior distribution." This is why "there's a 95% chance the true value is in this interval" is technically a *Bayesian* statement even when people casually say it about a frequentist CI — a common, mostly harmless conflation, but worth being precise about if pressed.

### When each mindset wins in production

- **Frequentist**: large-data A/B tests where you want a well-calibrated, repeatable decision procedure without injecting subjective belief
- **Bayesian**: small-data regimes (early-stage fraud model with few labeled cases), or when you have genuine prior knowledge worth encoding (e.g., known base rates), or when you need a full distribution over outcomes for downstream decision-making (e.g., bidding under uncertainty in RTB)

---

## 8. Multiple Hypothesis Correction

### Intuition

Test one fair coin at $\alpha=0.05$: 5% chance of a false alarm. Test **20 independent fair coins** at $\alpha=0.05$ each: expected false alarms = $20 \times 0.05 = 1$. You will almost certainly "discover" at least one biased coin that isn't — pure noise, amplified by volume of testing.

This is exactly the failure mode when you test hundreds of features for correlation with a label, or run dozens of metrics in one A/B test dashboard, and report whichever one crossed p<0.05.

### Bonferroni correction

Simplest fix: divide your significance threshold by the number of tests $m$.

$$\alpha_{\text{adjusted}} = \frac{\alpha}{m}$$

Testing 20 coins at overall 5% false-positive budget → each individual test needs $p < 0.05/20 = 0.0025$ to be called significant. **Conservative** — controls the probability of *any* false positive (family-wise error rate), at the cost of missing real effects (lower power) when $m$ is large.

### Benjamini-Hochberg (FDR control)

Less conservative, controls the **expected proportion of false positives among your discoveries** rather than the probability of any false positive at all:

1. Sort all $m$ p-values ascending: $p_{(1)} \le p_{(2)} \le \dots \le p_{(m)}$
2. Find the largest $k$ such that $p_{(k)} \le \frac{k}{m}\alpha$
3. Reject $H_0$ for all tests with $p \le p_{(k)}$

**Why this is looser than Bonferroni**: the threshold grows with rank $k$, so later (larger) p-values get a more lenient bar than Bonferroni's flat $\alpha/m$ — you tolerate a controlled amount of false discoveries in exchange for catching more true effects.

### When to use which

- **Bonferroni**: few tests, need near-zero tolerance for any false positive (e.g., a single go/no-go launch decision)
- **BH/FDR**: many tests, exploratory setting where some false positives are acceptable if the discovery rate stays controlled (e.g., screening 500 features for a model, or monitoring 50 metrics in an experimentation platform)

---

## 9. Power Analysis

### Intuition

Two ways a hypothesis test can fail, laid out as a 2×2:

| | $H_0$ actually true | $H_0$ actually false |
|---|---|---|
| **Reject $H_0$** | Type I error (false positive), rate = $\alpha$ | Correct rejection — **Power** = $1-\beta$ |
| **Fail to reject** | Correct — true negative | Type II error (false negative), rate = $\beta$ |

**Power** = $P(\text{detect the effect} \mid \text{effect is real})$. A power analysis answers: *"How much data do I need to reliably detect an effect of a given size, if it's really there?"* Run an underpowered test and you'll frequently conclude "no significant difference" when a real difference existed — you just didn't collect enough data to see it.

### Sample size formula (two-proportion test)

$$n = \frac{(z_{\alpha/2} + z_\beta)^2 \big[p_1(1-p_1) + p_2(1-p_2)\big]}{(p_1-p_2)^2}$$

### Worked example

You want to detect whether a new fraud model changes the flagged rate from $p_1=0.50$ to $p_2=0.60$, at $\alpha=0.05$ (two-sided, $z_{\alpha/2}=1.96$) and 80% power ($z_\beta=0.84$):

$$n = \frac{(1.96+0.84)^2\big[0.5(0.5)+0.6(0.4)\big]}{(0.1)^2} = \frac{7.84 \times 0.49}{0.01} \approx 385 \text{ per group}$$

**You'd need ~385 samples per group** to reliably detect a 10-point swing in rate at 80% power. Notice the direct link to Section 3: this is exactly why the 7/10-heads coin experiment (n=10) couldn't distinguish a fair coin from a 60%-biased one — it was wildly underpowered for that effect size. Smaller detectable effect sizes require dramatically larger $n$, since $n$ scales with $1/(p_1-p_2)^2$.

### Why this matters in production

Before launching an A/B test, power analysis tells you the minimum sample size (and therefore runtime) needed to trust a null result. Skipping this is how teams ship "no significant difference, ship it" conclusions off tests that never had a chance of detecting the effect size that mattered.

---

## 10. Law of Large Numbers & Central Limit Theorem

### Law of Large Numbers (LLN) — intuition

As you flip a coin more and more times, the observed proportion of heads converges to the true $p$.

$$\bar X_n = \frac{1}{n}\sum_{i=1}^n X_i \;\xrightarrow{P}\; \mathbb{E}[X] \quad \text{as } n \to \infty$$

10 flips can easily give 70% heads by chance (we saw this — p=0.34, plausible). 10,000 flips giving 70% heads is essentially impossible if $p=0.5$ — the sample average has locked onto the true rate. LLN is *why* "more data → more trustworthy estimate" is true at all; it's the formal guarantee behind that intuition.

### Central Limit Theorem (CLT) — intuition

Regardless of the underlying distribution of individual $X_i$ (coin flips are Bernoulli, not Gaussian!), the distribution of the **sample mean** approaches Gaussian as $n$ grows:

$$\bar X_n \;\approx\; \mathcal{N}\!\left(\mu,\ \frac{\sigma^2}{n}\right) \quad \text{for large } n$$

This is exactly why we could use a z-score / normal approximation for the 70/100 coin example even though a single coin flip is nowhere near Gaussian — sum enough Bernoullis together and the *sum's* distribution smooths into a bell curve. CLT is the bridge that lets Gaussian-based methods (z-tests, confidence intervals) apply almost universally to averages and sums, even when raw data isn't Gaussian at all.

### LLN vs CLT — the distinction

- **LLN** says the estimate converges to the truth (tells you *where* it's heading)
- **CLT** says *how* it fluctuates around that truth along the way, and that the fluctuation shape is Gaussian with spread shrinking as $\sigma/\sqrt{n}$ (tells you the *shape and speed* of convergence)

### Why this matters in ML

- **Mini-batch training**: a batch gradient is a sample mean of per-example gradients. LLN says larger batches give gradient estimates closer to the true (full-batch) gradient; CLT says the noise in that estimate is approximately Gaussian with variance shrinking as $1/\text{batch size}$ — directly explaining why larger batches produce smoother, less noisy training and why learning-rate scaling rules with batch size exist.
- **Bootstrap / bagging**: averaging predictions across resampled datasets relies on the same convergence logic — variance of the ensemble average shrinks as you add more bootstrap samples.
- **Monitoring dashboards**: a daily fraud rate computed from a handful of transactions is noisy (small n, CLT hasn't "kicked in" yet); the same metric over a week of volume is far more stable — same phenomenon as the 7/10 vs 70/100 coin gap.

---

## 11. Statistical Significance vs Practical Significance

### Intuition

A p-value tells you whether an effect is *real* (unlikely to be pure chance). It says nothing about whether the effect is *big enough to matter*. These are orthogonal questions, and conflating them is one of the most common production mistakes.

### Worked example

An A/B test runs on 10,000,000 users. Control CTR = 5.00%, treatment CTR = 5.01% — a 0.01 percentage-point lift. With that much volume, the standard error is tiny, and this can easily produce $p < 0.001$: **highly statistically significant.**

But a 0.01pp lift might translate to negligible revenue impact once you account for the engineering cost of shipping and maintaining the change. **Statistically real, practically irrelevant.**

Conversely, with n=50 users, a genuinely large improvement (say 5% → 8% CTR) might yield $p=0.25$ — not significant, purely because the sample was too small to detect it (an underpowered test, see Section 8) — even though the underlying effect, if real, would clearly matter practically.

### Effect size — the piece that closes the gap

**Cohen's d** (for comparing two means):

$$d = \frac{\bar X_1 - \bar X_2}{s_{\text{pooled}}}$$

Rule-of-thumb magnitudes: $d\approx0.2$ small, $0.5$ medium, $0.8$ large. Effect size is scale-free and sample-size-independent — unlike a p-value, it doesn't automatically shrink just because you collected more data. **Always report effect size and a confidence interval alongside the p-value**, never the p-value in isolation; the p-value answers "is it real," the effect size answers "does it matter."

### One-line synthesis across Tier 1 + Tier 2

Every concept in this document eventually answers one of three questions: **what does the data look like (distributions), is an observed pattern real or noise (hypothesis testing, p-values, power, multiple testing), and how much should I trust my estimate (MLE, Bayesian inference, CLT/LLN, bias-variance).** Practical significance is the reminder that "real" and "worth acting on" are not the same claim.

---

## Checkpoint

Tier 1 + Tier 2 core concepts are now in place, all threaded through the same coin-toss example where possible so the numbers reinforce each other across sections (7/10 → p=0.34 → MLE $\hat p=0.7$ → Bayesian posterior 0.667/0.54 → power analysis referencing why n=10 couldn't detect a 60% bias).

Remaining open items, your call on priority:
1. Standalone deep-dive on Regularization if the folded-in coverage (Sections 5 & 7) isn't enough
2. Causal inference, sample size calculations, or other Tier 3 items if you want them before Tuesday
3. Or move on entirely — pivot to Day 3–5 prep (STAR stories, production narratives, case study framework) per your five-day plan
